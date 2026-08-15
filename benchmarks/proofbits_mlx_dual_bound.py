import gc
import json
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models import cache as cache_mod

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

DUAL_SRC=r'''
 uint row=threadgroup_position_in_grid.x;
 uint lane=thread_index_in_simdgroup;
 uint D=(uint)hidden_shape[0];
 ulong base=(ulong)row*(ulong)D;
 float au=0.0f;
 float al=0.0f;
 for(uint j=lane;j<D;j+=32u){
   uchar hb=high[base+j];
   ushort ws=(ushort)(hb & (uchar)0x80);
   ushort hs=(hidden[j] < 0.0f) ? (ushort)0x80 : (ushort)0x00;
   bool same=(ws==hs);
   ushort raw_u=((ushort)hb<<8) | (same ? (ushort)0x00FF : (ushort)0x0000);
   ushort raw_l=((ushort)hb<<8) | (same ? (ushort)0x0000 : (ushort)0x00FF);
   half wu=as_type<half>(raw_u);
   half wl=as_type<half>(raw_l);
   float x=hidden[j];
   au=fma(x,(float)wu,au);
   al=fma(x,(float)wl,al);
 }
 float su=simd_sum(au);
 float sl=simd_sum(al);
 if(lane==0){ upper[row]=su; lower[row]=sl; }
'''


def med(x):return float(statistics.median(x))

def make_dual():
 return mx.fast.metal_kernel(name='pb_mlx_dual_interval_row',input_names=['high','hidden'],output_names=['upper','lower'],source=DUAL_SRC)

def prepare(model):
 w=model.lm_head.weight if hasattr(model,'lm_head') else model.model.embed_tokens.weight
 w16=w.astype(mx.float16);mx.eval(w16)
 a=np.array(w16,copy=True).astype(np.float16,copy=False);bits=a.view(np.uint16)
 high=mx.array((bits>>8).astype(np.uint8,copy=True));low=mx.array((bits&255).astype(np.uint8,copy=True));mx.eval(high,low)
 return w16,high,low

def dual_decision(dual_k,refine_k,high,low,h,V,diag=False):
 U,L=dual_k(inputs=[high,h],grid=(V*32,1,1),threadgroup=(32,1,1),output_shapes=[(V,),(V,)],output_dtypes=[mx.float32,mx.float32],init_value=0.0)
 B=mx.reshape(mx.max(L),(1,))
 E=refine_k(inputs=[high,low,h,U,B],grid=(V*32,1,1),threadgroup=(32,1,1),output_shapes=[(V,)],output_dtypes=[mx.float32],init_value=-3.402823466e38)[0]
 y=mx.argmax(E).astype(mx.uint32)
 if diag:return y,mx.sum(U>=B),mx.max(U-L)
 return y

def ids(tok,p):
 try:x=tok.encode(p)
 except Exception:x=tok(p)['input_ids']
 return mx.array(x,dtype=mx.int32)

def run(model,tok,prompt,mode,ks,dual_k,w16,high,low,n,diag=False):
 V,D=[int(x) for x in w16.shape];kv=cache_mod.make_prompt_cache(model)
 b=model.model(ids(tok,prompt)[None],cache=kv);h=b[:,-1,:].reshape((D,)).astype(mx.float32);mx.eval(h,[c.state for c in kv])
 toks=[];times=[];surv=[]
 for _ in range(n):
  t0=time.perf_counter()
  if mode=='dense':y=base.call_dense(ks[0],w16,h,V)
  elif mode=='native':y=base.call_native(w16,h)
  elif mode=='current':y=base.call_proofbits(ks[1],ks[2],ks[3],high,low,h,V,False)
  elif mode=='dual':
   if diag:y,s,_=dual_decision(dual_k,ks[3],high,low,h,V,True)
   else:y=dual_decision(dual_k,ks[3],high,low,h,V,False)
  else:raise ValueError(mode)
  mx.eval(y)
  if diag and mode=='dual':mx.eval(s);surv.append(int(s.item()))
  token=int(y.item());toks.append(token)
  b=model.model(mx.array([[token]],dtype=mx.int32),cache=kv);h=b[:,-1,:].reshape((D,)).astype(mx.float32);mx.eval(h,[c.state for c in kv]);times.append((time.perf_counter()-t0)*1e3)
 return {'tokens':toks,'median_ms':med(times),'mean_ms':float(statistics.mean(times)),'survivor_mean':float(statistics.mean(surv)) if surv else None,'survivor_max':max(surv) if surv else None}

def main():
 model,tok=load(MODEL);model.set_dtype(mx.float16);mx.eval(model.parameters());base.MODEL=MODEL
 w16,high,low=prepare(model);ks=base.make_kernels();dual_k=make_dual()
 modes=['dense','current','dual','native']
 for m in modes:run(model,tok,PROMPTS[0],m,ks,dual_k,w16,high,low,4)
 mx.synchronize();rows=[]
 orders=[['dense','current','dual','native'],['dual','native','dense','current'],['native','current','dual','dense'],['current','dense','native','dual']]
 for i,p in enumerate(PROMPTS):
  res={};order=orders[i%4]
  for m in order:
   gc.collect();mx.clear_cache();res[m]=run(model,tok,p,m,ks,dual_k,w16,high,low,TOKENS)
  d,c,u,n=res['dense'],res['current'],res['dual'],res['native']
  rows.append({'prompt_index':i,'order':order,'current_exact':c['tokens']==d['tokens'],'dual_exact':u['tokens']==d['tokens'],'native_equal_dual':n['tokens']==u['tokens'],
   'dense_ms':d['median_ms'],'current_ms':c['median_ms'],'dual_ms':u['median_ms'],'native_ms':n['median_ms'],
   'dual_vs_current':c['median_ms']/u['median_ms'],'dense_over_dual':d['median_ms']/u['median_ms'],'native_over_dual':n['median_ms']/u['median_ms']})
 # diagnostics separate from timed comparison
 diags=[]
 for i,p in enumerate(PROMPTS):
  q=run(model,tok,p,'dual',ks,dual_k,w16,high,low,8,True);diags.append({'prompt_index':i,'survivor_mean':q['survivor_mean'],'survivor_max':q['survivor_max']})
 out={'kind':'proofbits_fp16_pilotfree_dual_bound','model':MODEL,'n_prompts':len(PROMPTS),'tokens_per_prompt':TOKENS,'rows':rows,'diagnostics':diags,
  'all_current_exact':all(r['current_exact'] for r in rows),'all_dual_exact':all(r['dual_exact'] for r in rows),
  'median_dual_vs_current':med([r['dual_vs_current'] for r in rows]),'mean_dual_vs_current':float(statistics.mean(r['dual_vs_current'] for r in rows)),
  'median_native_over_dual':med([r['native_over_dual'] for r in rows]),'min_native_over_dual':min(r['native_over_dual'] for r in rows),
  'note':'Pilot-free exact lower bound B=max_i L_i. L_i and U_i are computed from the same high-byte read. Since L_i <= exact_i <= U_i, global winner has U >= exact_max >= max L and survives. Diagnostic survivor scans are outside timed comparison.'}
 Path('experiments/artifacts').mkdir(parents=True,exist_ok=True);Path('experiments/artifacts/proofbits_mlx_dual_bound.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
