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
M=8
JS=[4,8,16]

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

def med(x):return float(statistics.median(x))

def make_candidate():
 return mx.fast.metal_kernel(name='pb_axis_exact_candidates',input_names=['high','low','hidden','indices'],output_names=['scores'],source=CAND_SRC)

def build_extreme_index(w16):
 # Setup only; excluded from timing. For every hidden dimension, store M rows
 # with minimum and maximum actual FP16 weight values.
 a=np.array(w16,copy=True).astype(np.float16,copy=False).astype(np.float32)
 D=a.shape[1]
 tab=np.empty((D,2,M),dtype=np.uint32)
 for j in range(D):
  col=a[:,j]
  lo=np.argpartition(col,M-1)[:M]
  hi=np.argpartition(col,len(col)-M)[-M:]
  tab[j,0]=lo.astype(np.uint32);tab[j,1]=hi.astype(np.uint32)
 return mx.array(tab.reshape(-1)),D

def axis_indices(table,h,D,J):
 # top J hidden coordinates only; D=640 so this reduction is tiny.
 dims=mx.argpartition(mx.abs(h),D-J)[D-J:].astype(mx.uint32)
 neg=(mx.take(h,dims)<0).astype(mx.uint32)
 # table layout [dim, signChoice(min=0,max=1), M]; for h>=0 want max(1), h<0 want min(0)
 choice=1-neg
 offs=dims*(2*M)+choice*M
 idx=mx.reshape(offs[:,None]+mx.arange(M,dtype=mx.uint32)[None,:],(-1,))
 return mx.take(table,idx).astype(mx.uint32)

def axis_dual_decision(dual_k,refine_k,cand_k,table,high,low,h,V,D,J,diag=False):
 U,L=dual_k(inputs=[high,h],grid=(V*32,1,1),threadgroup=(32,1,1),output_shapes=[(V,),(V,)],output_dtypes=[mx.float32,mx.float32],init_value=0.0)
 idx=axis_indices(table,h,D,J)
 K=J*M
 ps=cand_k(inputs=[high,low,h,idx],grid=(K*32,1,1),threadgroup=(32,1,1),output_shapes=[(K,)],output_dtypes=[mx.float32],init_value=-3.402823466e38)[0]
 B=mx.reshape(mx.maximum(mx.max(L),mx.max(ps)),(1,))
 E=refine_k(inputs=[high,low,h,U,B],grid=(V*32,1,1),threadgroup=(32,1,1),output_shapes=[(V,)],output_dtypes=[mx.float32],init_value=-3.402823466e38)[0]
 y=mx.argmax(E).astype(mx.uint32)
 if diag:return y,mx.sum(U>=B)
 return y

def ids(tok,p):
 try:x=tok.encode(p)
 except Exception:x=tok(p)['input_ids']
 return mx.array(x,dtype=mx.int32)

def run(model,tok,prompt,mode,ks,dual_k,cand_k,table,w16,high,low,n,diag=False):
 V,D=[int(x) for x in w16.shape];kv=cache_mod.make_prompt_cache(model);b=model.model(ids(tok,prompt)[None],cache=kv);h=b[:,-1,:].reshape((D,)).astype(mx.float32);mx.eval(h,[c.state for c in kv]);toks=[];times=[];surv=[]
 for _ in range(n):
  t=time.perf_counter()
  if mode=='dense':y=base.call_dense(ks[0],w16,h,V)
  elif mode=='native':y=base.call_native(w16,h)
  elif mode=='current':y=base.call_proofbits(ks[1],ks[2],ks[3],high,low,h,V,False)
  elif mode=='dual':y=dual.dual_decision(dual_k,ks[3],high,low,h,V,False)
  elif mode.startswith('axis'):
   J=int(mode[4:])
   if diag:y,s=axis_dual_decision(dual_k,ks[3],cand_k,table,high,low,h,V,D,J,True)
   else:y=axis_dual_decision(dual_k,ks[3],cand_k,table,high,low,h,V,D,J,False)
  else:raise ValueError(mode)
  mx.eval(y)
  if diag and mode.startswith('axis'):mx.eval(s);surv.append(int(s.item()))
  token=int(y.item());toks.append(token);b=model.model(mx.array([[token]],dtype=mx.int32),cache=kv);h=b[:,-1,:].reshape((D,)).astype(mx.float32);mx.eval(h,[c.state for c in kv]);times.append((time.perf_counter()-t)*1e3)
 return {'tokens':toks,'median_ms':med(times),'mean_ms':float(statistics.mean(times)),'survivor_mean':float(statistics.mean(surv)) if surv else None,'survivor_max':max(surv) if surv else None}

def main():
 model,tok=load(MODEL);model.set_dtype(mx.float16);mx.eval(model.parameters());base.MODEL=MODEL
 w16,high,low=dual.prepare(model);V,D=[int(x) for x in w16.shape];table,D2=build_extreme_index(w16);assert D2==D;mx.eval(table)
 ks=base.make_kernels();dual_k=dual.make_dual();cand_k=make_candidate();modes=['dense','current','dual']+[f'axis{J}' for J in JS]+['native']
 for m in modes:run(model,tok,PROMPTS[0],m,ks,dual_k,cand_k,table,w16,high,low,4)
 mx.synchronize();rows=[]
 for i,p in enumerate(PROMPTS):
  order=modes[i%len(modes):]+modes[:i%len(modes)];res={}
  for m in order:
   gc.collect();mx.clear_cache();res[m]=run(model,tok,p,m,ks,dual_k,cand_k,table,w16,high,low,TOKENS)
  d,c,u,nat=res['dense'],res['current'],res['dual'],res['native'];r={'prompt_index':i,'order':order,'current_exact':c['tokens']==d['tokens'],'dual_exact':u['tokens']==d['tokens'],'dense_ms':d['median_ms'],'current_ms':c['median_ms'],'dual_ms':u['median_ms'],'native_ms':nat['median_ms']}
  for J in JS:
   q=res[f'axis{J}'];r[f'axis{J}_exact']=q['tokens']==d['tokens'];r[f'axis{J}_ms']=q['median_ms'];r[f'dual_over_axis{J}']=u['median_ms']/q['median_ms'];r[f'current_over_axis{J}']=c['median_ms']/q['median_ms'];r[f'native_over_axis{J}']=nat['median_ms']/q['median_ms']
  rows.append(r)
 diags=[]
 for i,p in enumerate(PROMPTS):
  x={'prompt_index':i}
  for J in JS:
   q=run(model,tok,p,f'axis{J}',ks,dual_k,cand_k,table,w16,high,low,8,True);x[f'axis{J}_survivor_mean']=q['survivor_mean'];x[f'axis{J}_survivor_max']=q['survivor_max']
  diags.append(x)
 out={'kind':'proofbits_axis_pilot_dual','model':MODEL,'M_per_axis':M,'Js':JS,'rows':rows,'diagnostics':diags}
 for J in JS:
  out[f'axis{J}_all_exact']=all(r[f'axis{J}_exact'] for r in rows);out[f'axis{J}_median_dual_over']=med([r[f'dual_over_axis{J}'] for r in rows]);out[f'axis{J}_median_current_over']=med([r[f'current_over_axis{J}'] for r in rows]);out[f'axis{J}_median_native_over']=med([r[f'native_over_axis{J}'] for r in rows]);out[f'axis{J}_min_native_over']=min(r[f'native_over_axis{J}'] for r in rows);out[f'axis{J}_diagnostic_median_survivors']=med([x[f'axis{J}_survivor_mean'] for x in diags])
 out['note']='AxisPilot is an offline exact-pilot heuristic only: per hidden dimension, M extreme output rows are preindexed. Runtime selects dominant |h_j| axes and exact-scores their candidate rows. B=max(max_i L_i,max_p s_p) remains a rigorous lower bound, so heuristic quality affects only speed, never correctness.'
 Path('experiments/artifacts').mkdir(parents=True,exist_ok=True);Path('experiments/artifacts/proofbits_mlx_axis_pilot.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
