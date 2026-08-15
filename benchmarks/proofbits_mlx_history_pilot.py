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
import proofbits_mlx_dual_bound as dual

MODEL='mlx-community/gemma-3-270m-bf16'
PROMPTS=[
 'Explain why entropy is measured with logarithms.',
 'Solve step by step: if x + 1/x = 3, compute x^6 + 1/x^6.',
 'Write a Python function for longest increasing contiguous subarray.',
 'Explain why memory bandwidth matters for autoregressive inference.',
 'Describe natural selection without using teleological language.',
]
TOKENS=32

CAND_SRC=r'''
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
   acc=fma(hidden[j],(float)as_type<half>(raw),acc);
 }
 float s=simd_sum(acc);
 if(lane==0)scores[slot]=s;
'''


def med(x): return float(statistics.median(x))


def make_candidate():
 return mx.fast.metal_kernel(name='pb_history_exact_candidates',input_names=['high','low','hidden','indices'],output_names=['scores'],source=CAND_SRC)


def ids(tok,prompt):
 try:x=tok.encode(prompt)
 except Exception:x=tok(prompt)['input_ids']
 return mx.array(x,dtype=mx.int32)


def history_decision(ks,cand_k,high,low,h,V,mode,p_prev,winner_hist):
 # Any exact score under current h is a valid global lower bound. Candidate
 # selection uses only state from previous decode steps, so it is independent
 # of the current U and can execute in parallel with U.
 U=ks[1](inputs=[high,h],grid=(V*32,1,1),threadgroup=(32,1,1),
         output_shapes=[(V,)],output_dtypes=[mx.float32],init_value=0.0)[0]

 cand=[]
 if mode in ('histp1','histp4','histyp4'):
  if isinstance(p_prev,list): cand.extend(p_prev[-(4 if mode=='histp4' else 1):])
  else: cand.append(int(p_prev))
 if mode in ('histy1','histy4','histyp4'):
  cand.extend(winner_hist[-(4 if mode in ('histy4','histyp4') else 1):])
 if not cand: cand=[0]
 # Dedup is setup-side only and reduces redundant exact rows.
 cand=list(dict.fromkeys(int(x) for x in cand))
 idx=mx.array(cand,dtype=mx.uint32)
 K=len(cand)
 ps=cand_k(inputs=[high,low,h,idx],grid=(K*32,1,1),threadgroup=(32,1,1),
           output_shapes=[(K,)],output_dtypes=[mx.float32],init_value=-3.402823466e38)[0]
 B=mx.reshape(mx.max(ps),(1,))
 E=ks[3](inputs=[high,low,h,U,B],grid=(V*32,1,1),threadgroup=(32,1,1),
         output_shapes=[(V,)],output_dtypes=[mx.float32],init_value=-3.402823466e38)[0]
 y=mx.argmax(E).astype(mx.uint32)
 # p_next is carried to a future step only. Because current y does not depend
 # on it, MLX can schedule argmax(U) alongside refinement instead of before it.
 p_next=mx.argmax(U).astype(mx.uint32) if mode in ('histp1','histp4','histyp4') else None
 return y,p_next,U,B


def run(model,tok,prompt,mode,ks,cand_k,w16,high,low,n,diagnostic=False):
 V,D=[int(x) for x in w16.shape]
 prompt_ids=ids(tok,prompt)
 prompt_list=[int(x) for x in np.array(prompt_ids).reshape(-1).tolist()]
 kv=cache_mod.make_prompt_cache(model)
 b=model.model(prompt_ids[None],cache=kv)
 h=b[:,-1,:].reshape((D,)).astype(mx.float32)
 mx.eval(h,[c.state for c in kv])
 toks=[];times=[];surv=[]
 winner_hist=prompt_list[-4:] if prompt_list else [0]
 p_hist=[winner_hist[-1]]
 for _ in range(n):
  t0=time.perf_counter()
  p_next=None
  if mode=='dense': y=base.call_dense(ks[0],w16,h,V)
  elif mode=='native': y=base.call_native(w16,h)
  elif mode=='current': y=base.call_proofbits(ks[1],ks[2],ks[3],high,low,h,V,False)
  elif mode=='dual': y=dual.dual_decision(dual.make_dual_cached if False else None,ks[3],high,low,h,V,False)
  else:
   psrc=p_hist if mode in ('histp4',) else p_hist[-1]
   y,p_next,U,B=history_decision(ks,cand_k,high,low,h,V,mode,psrc,winner_hist)
  if p_next is not None: mx.eval(y,p_next)
  else: mx.eval(y)
  if diagnostic and mode.startswith('hist'):
   s=mx.sum(U>=B);mx.eval(s);surv.append(int(s.item()))
  token=int(y.item());toks.append(token);winner_hist.append(token);winner_hist=winner_hist[-4:]
  if p_next is not None:
   p_hist.append(int(p_next.item()));p_hist=p_hist[-4:]
  b=model.model(mx.array([[token]],dtype=mx.int32),cache=kv)
  h=b[:,-1,:].reshape((D,)).astype(mx.float32)
  mx.eval(h,[c.state for c in kv])
  times.append((time.perf_counter()-t0)*1e3)
 return {'tokens':toks,'median_ms':med(times),'mean_ms':float(statistics.mean(times)),
         'survivor_mean':float(statistics.mean(surv)) if surv else None,'survivor_max':max(surv) if surv else None}


def run_dual(model,tok,prompt,ks,dual_k,w16,high,low,n):
 V,D=[int(x) for x in w16.shape];prompt_ids=ids(tok,prompt);kv=cache_mod.make_prompt_cache(model)
 b=model.model(prompt_ids[None],cache=kv);h=b[:,-1,:].reshape((D,)).astype(mx.float32);mx.eval(h,[c.state for c in kv]);toks=[];times=[]
 for _ in range(n):
  t0=time.perf_counter();y=dual.dual_decision(dual_k,ks[3],high,low,h,V,False);mx.eval(y);token=int(y.item());toks.append(token)
  b=model.model(mx.array([[token]],dtype=mx.int32),cache=kv);h=b[:,-1,:].reshape((D,)).astype(mx.float32);mx.eval(h,[c.state for c in kv]);times.append((time.perf_counter()-t0)*1e3)
 return {'tokens':toks,'median_ms':med(times),'mean_ms':float(statistics.mean(times))}


def main():
 model,tok=load(MODEL);model.set_dtype(mx.float16);mx.eval(model.parameters());base.MODEL=MODEL
 w16,high,low=dual.prepare(model);ks=base.make_kernels();dual_k=dual.make_dual();cand_k=make_candidate()
 modes=['dense','current','dual','histy1','histy4','histp1','histp4','histyp4','native']
 # compile paths
 for m in modes:
  if m=='dual': run_dual(model,tok,PROMPTS[0],ks,dual_k,w16,high,low,4)
  else: run(model,tok,PROMPTS[0],m,ks,cand_k,w16,high,low,4)
 mx.synchronize();rows=[]
 for pi,p in enumerate(PROMPTS):
  order=modes[pi%len(modes):]+modes[:pi%len(modes)];res={}
  for m in order:
   gc.collect();mx.clear_cache()
   res[m]=run_dual(model,tok,p,ks,dual_k,w16,high,low,TOKENS) if m=='dual' else run(model,tok,p,m,ks,cand_k,w16,high,low,TOKENS)
  d,c,u,nat=res['dense'],res['current'],res['dual'],res['native']
  r={'prompt_index':pi,'order':order,'current_exact':c['tokens']==d['tokens'],'dual_exact':u['tokens']==d['tokens'],
     'dense_ms':d['median_ms'],'current_ms':c['median_ms'],'dual_ms':u['median_ms'],'native_ms':nat['median_ms']}
  for m in ['histy1','histy4','histp1','histp4','histyp4']:
   q=res[m];r[f'{m}_exact']=q['tokens']==d['tokens'];r[f'{m}_ms']=q['median_ms'];r[f'current_over_{m}']=c['median_ms']/q['median_ms'];r[f'native_over_{m}']=nat['median_ms']/q['median_ms']
  rows.append(r)

 # survivor diagnostics separated from timed comparisons
 diags=[]
 for pi,p in enumerate(PROMPTS):
  x={'prompt_index':pi}
  for m in ['histy1','histy4','histp1','histp4','histyp4']:
   q=run(model,tok,p,m,ks,cand_k,w16,high,low,8,diagnostic=True);x[f'{m}_survivor_mean']=q['survivor_mean'];x[f'{m}_survivor_max']=q['survivor_max']
  diags.append(x)

 out={'kind':'proofbits_temporal_history_pilot','model':MODEL,'n_prompts':len(PROMPTS),'tokens_per_prompt':TOKENS,'rows':rows,'diagnostics':diags}
 for m in ['histy1','histy4','histp1','histp4','histyp4']:
  out[f'{m}_all_exact']=all(r[f'{m}_exact'] for r in rows)
  out[f'{m}_median_current_over']=med([r[f'current_over_{m}'] for r in rows])
  out[f'{m}_mean_current_over']=float(statistics.mean(r[f'current_over_{m}'] for r in rows))
  out[f'{m}_median_native_over']=med([r[f'native_over_{m}'] for r in rows])
  out[f'{m}_min_native_over']=min(r[f'native_over_{m}'] for r in rows)
  out[f'{m}_diag_median_survivors']=med([x[f'{m}_survivor_mean'] for x in diags])
 out['note']='Temporal pilots exact-score rows selected only from previous decode state. Therefore the current upper pass U and pilot exact scores are dependency-independent. For p-history, current argmax(U) is carried only to the next step and is evaluated alongside current final decision, not before the current threshold. Any pilot exact score is a valid lower bound, so history quality affects speed but not correctness.'
 Path('experiments/artifacts').mkdir(parents=True,exist_ok=True);Path('experiments/artifacts/proofbits_mlx_history_pilot.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

if __name__=='__main__':main()
