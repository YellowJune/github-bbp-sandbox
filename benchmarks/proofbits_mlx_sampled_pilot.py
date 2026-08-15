import gc
import json
import statistics
import sys
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models import cache as cache_mod
import time

sys.path.insert(0,str(Path(__file__).resolve().parent))
import proofbits_mlx_integrated_decode as base

MODEL='mlx-community/gemma-3-270m-bf16'
PROMPTS=[
 'Explain why entropy is measured with logarithms.',
 'Solve step by step: if x + 1/x = 3, compute x^6 + 1/x^6.',
 'Write a Python function for longest increasing contiguous subarray.',
 'Explain why memory bandwidth matters for autoregressive inference.',
 'Describe natural selection without using teleological language.',
]
TOKENS=32
PILOT_KS=[256,1024,4096]

PILOT_BATCH_SRC=r'''
 uint slot=threadgroup_position_in_grid.x;
 uint lane=thread_index_in_simdgroup;
 uint K=(uint)indices_shape[0];
 if(slot>=K)return;
 uint row=indices[slot];
 uint D=(uint)hidden_shape[0];
 ulong b=(ulong)row*(ulong)D;
 float acc=0.0f;
 for(uint j=lane;j<D;j+=32u){
   ushort raw=((ushort)high[b+j]<<8)|(ushort)low[b+j];
   half w=as_type<half>(raw);
   acc=fma(hidden[j],(float)w,acc);
 }
 float total=simd_sum(acc);
 if(lane==0)scores[slot]=total;
'''


def med(x):return float(statistics.median(x))

def make_pilot_kernel():
 return mx.fast.metal_kernel(name='pb_sampled_exact_pilots',input_names=['high','low','hidden','indices'],output_names=['scores'],source=PILOT_BATCH_SRC)

def make_indices(V,K):
 # Evenly cover token-id space; exactness does not depend on sampling quality.
 return ((mx.arange(K,dtype=mx.uint32)*V)//K).astype(mx.uint32)

def sampled_decision(base_ks,pilot_k,w16,high,low,h,V,idx):
 _,upper_k,_,refine_k=base_ks
 K=int(idx.shape[0])
 # U and pilot exact scores are independent until refine; MLX may schedule the
 # lazy graph without the old argmax(U)->pilot serialization chain.
 U=upper_k(inputs=[high,h],grid=(V*32,1,1),threadgroup=(32,1,1),output_shapes=[(V,)],output_dtypes=[mx.float32],init_value=0.0)[0]
 ps=pilot_k(inputs=[high,low,h,idx],grid=(K*32,1,1),threadgroup=(32,1,1),output_shapes=[(K,)],output_dtypes=[mx.float32],init_value=-3.402823466e38)[0]
 B=mx.reshape(mx.max(ps),(1,))
 E=refine_k(inputs=[high,low,h,U,B],grid=(V*32,1,1),threadgroup=(32,1,1),output_shapes=[(V,)],output_dtypes=[mx.float32],init_value=-3.402823466e38)[0]
 return mx.argmax(E).astype(mx.uint32), mx.sum(U>=B)

def ids(tok,p):
 try:x=tok.encode(p)
 except Exception:x=tok(p)['input_ids']
 return mx.array(x,dtype=mx.int32)

def run(model,tok,prompt,mode,base_ks,pilot_k,w16,high,low,idxs,n):
 V,D=[int(x) for x in w16.shape];kv=cache_mod.make_prompt_cache(model)
 b=model.model(ids(tok,prompt)[None],cache=kv);h=b[:,-1,:].reshape((D,)).astype(mx.float32);mx.eval(h,[c.state for c in kv])
 toks=[];times=[];surv=[]
 for _ in range(n):
  t0=time.perf_counter()
  if mode=='dense':y=base.call_dense(base_ks[0],w16,h,V)
  elif mode=='current':y=base.call_proofbits(base_ks[1],base_ks[2],base_ks[3],high,low,h,V,False)
  elif mode=='native':y=base.call_native(w16,h)
  elif mode.startswith('sample'):
   K=int(mode[6:]);y,s=sampled_decision(base_ks,pilot_k,w16,high,low,h,V,idxs[K])
  else:raise ValueError(mode)
  mx.eval(y)
  if mode.startswith('sample'):mx.eval(s);surv.append(int(s.item()))
  token=int(y.item());toks.append(token)
  b=model.model(mx.array([[token]],dtype=mx.int32),cache=kv);h=b[:,-1,:].reshape((D,)).astype(mx.float32);mx.eval(h,[c.state for c in kv]);times.append((time.perf_counter()-t0)*1e3)
 return {'tokens':toks,'median_ms':med(times),'mean_ms':float(statistics.mean(times)),'survivor_mean':float(statistics.mean(surv)) if surv else None,'survivor_max':max(surv) if surv else None}

def main():
 model,tok=load(MODEL);model.set_dtype(mx.float16);mx.eval(model.parameters());base.MODEL=MODEL
 base_ks=base.make_kernels();pilot_k=make_pilot_kernel();w16,high,low=base.prepare_weights(model);V=int(w16.shape[0]);idxs={K:make_indices(V,K) for K in PILOT_KS};mx.eval(*idxs.values())
 modes=['dense','current','native']+[f'sample{K}' for K in PILOT_KS]
 for m in modes:run(model,tok,PROMPTS[0],m,base_ks,pilot_k,w16,high,low,idxs,4)
 mx.synchronize();rows=[]
 for pi,p in enumerate(PROMPTS):
  order=modes[pi%len(modes):]+modes[:pi%len(modes)];res={}
  for m in order:
   gc.collect();mx.clear_cache();res[m]=run(model,tok,p,m,base_ks,pilot_k,w16,high,low,idxs,TOKENS)
  d,c,nat=res['dense'],res['current'],res['native'];r={'prompt_index':pi,'order':order,'current_exact':d['tokens']==c['tokens'],'dense_ms':d['median_ms'],'current_ms':c['median_ms'],'native_ms':nat['median_ms']}
  for K in PILOT_KS:
   q=res[f'sample{K}'];r[f'k{K}_exact']=q['tokens']==d['tokens'];r[f'k{K}_ms']=q['median_ms'];r[f'k{K}_vs_current']=c['median_ms']/q['median_ms'];r[f'native_over_k{K}']=nat['median_ms']/q['median_ms'];r[f'k{K}_survivor_mean']=q['survivor_mean'];r[f'k{K}_survivor_max']=q['survivor_max']
  rows.append(r)
 out={'kind':'proofbits_fp16_sampled_exact_pilot','model':MODEL,'pilot_Ks':PILOT_KS,'n_prompts':len(PROMPTS),'tokens_per_prompt':TOKENS,'rows':rows}
 for K in PILOT_KS:
  out[f'k{K}_all_exact']=all(r[f'k{K}_exact'] for r in rows);out[f'k{K}_median_vs_current']=med([r[f'k{K}_vs_current'] for r in rows]);out[f'k{K}_median_native_over']=med([r[f'native_over_k{K}'] for r in rows]);out[f'k{K}_min_native_over']=min(r[f'native_over_k{K}'] for r in rows);out[f'k{K}_mean_survivors']=float(statistics.mean(r[f'k{K}_survivor_mean'] for r in rows));out[f'k{K}_max_survivors']=max(r[f'k{K}_survivor_max'] for r in rows)
 out['note']='Any exact pilot score is a valid lower bound on the global maximum. Uniform pilot subsets remove argmax(U) entirely. Refinement remains exact because every true winner must satisfy U_i >= max pilot exact score. Survivor diagnostics are included in timing here.'
 Path('experiments/artifacts').mkdir(parents=True,exist_ok=True);Path('experiments/artifacts/proofbits_mlx_sampled_pilot.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
