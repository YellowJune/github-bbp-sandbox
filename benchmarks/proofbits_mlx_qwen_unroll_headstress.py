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

MODEL='mlx-community/Qwen2.5-0.5B-Instruct-bf16'
PROMPTS=[
 'Explain why entropy is measured with logarithms.',
 'Solve step by step: if x + 1/x = 3, compute x^6 + 1/x^6.',
 'Write a Python function for longest increasing contiguous subarray.',
 'Explain why memory bandwidth matters for autoregressive inference.',
 'Compare TCP and UDP for a latency-sensitive application.',
 'Give a concise proof that the square root of 2 is irrational.',
 'Explain the difference between correlation and causation.',
 'Write pseudocode for breadth-first search on a graph.',
 'Summarize why cache locality matters in matrix computation.',
 'Explain photosynthesis to a high-school student.',
 'Derive the quadratic formula from completing the square.',
 'Describe natural selection without using teleological language.',
]
STATES_PER_PROMPT=8
REPEATS=3


def med(x):return float(statistics.median(x))

def prepare(model):
 w=model.lm_head.weight if hasattr(model,'lm_head') else model.model.embed_tokens.weight
 w16=w.astype(mx.float16);mx.eval(w16)
 a=np.array(w16,copy=True).astype(np.float16,copy=False);bits=a.view(np.uint16)
 high=mx.array((bits>>8).astype(np.uint8,copy=True));low=mx.array((bits&255).astype(np.uint8,copy=True));mx.eval(high,low)
 return w16,high,low

def divisors(n):
 # n is number of original 32-lane FMA iterations. Keep representative divisors.
 ds=[d for d in range(1,n+1) if n%d==0]
 target=[]
 for x in [1,2,4,5,7,8,10,12,14,16,n]:
  if x in ds and x not in target:target.append(x)
 if len(target)<4:
  target=ds
 return target

def make_src(D,f):
 it=D//32;assert D%32==0 and it%f==0
 def one(off):
  return f'''{{ uint j=basej+{off}u; uchar hb=high[base+j]; ushort ws=(ushort)(hb & (uchar)0x80); ushort hs=(hidden[j]<0.0f)?(ushort)0x80:(ushort)0x00; ushort suffix=(ws==hs)?(ushort)0x00FF:(ushort)0x0000; ushort raw=((ushort)hb<<8)|suffix; acc=fma(hidden[j],(float)as_type<half>(raw),acc); }}'''
 body='\n'.join(one(32*k) for k in range(f))
 if f==it:
  body=body.replace('basej','lane')
  return f'''uint row=threadgroup_position_in_grid.x;uint lane=thread_index_in_simdgroup;ulong base=(ulong)row*{D}ul;float acc=0.0f;{body}float total=simd_sum(acc);if(lane==0)upper[row]=total;'''
 return f'''uint row=threadgroup_position_in_grid.x;uint lane=thread_index_in_simdgroup;ulong base=(ulong)row*{D}ul;float acc=0.0f;for(uint basej=lane;basej<{D}u;basej+={32*f}u){{{body}}}float total=simd_sum(acc);if(lane==0)upper[row]=total;'''

def make_upper(D,f):return mx.fast.metal_kernel(name=f'pb_qwen_upper_u{f}',input_names=['high','hidden'],output_names=['upper'],source=make_src(D,f))

def decision(upper,ks,high,low,h,V):
 U=upper(inputs=[high,h],grid=(V*32,1,1),threadgroup=(32,1,1),output_shapes=[(V,)],output_dtypes=[mx.float32],init_value=0.0)[0]
 p=mx.reshape(mx.argmax(U).astype(mx.uint32),(1,))
 B=ks[2](inputs=[high,low,h,p],grid=(32,1,1),threadgroup=(32,1,1),output_shapes=[(1,)],output_dtypes=[mx.float32],init_value=0.0)[0]
 E=ks[3](inputs=[high,low,h,U,B],grid=(V*32,1,1),threadgroup=(32,1,1),output_shapes=[(V,)],output_dtypes=[mx.float32],init_value=-3.402823466e38)[0]
 return mx.argmax(E).astype(mx.uint32)

def ids(tok,p):
 try:x=tok.encode(p)
 except Exception:x=tok(p)['input_ids']
 return mx.array(x,dtype=mx.int32)

def collect(model,tok,D):
 states=[]
 for pi,p in enumerate(PROMPTS):
  cache=cache_mod.make_prompt_cache(model);b=model.model(ids(tok,p)[None],cache=cache);h=b[:,-1,:].reshape((D,)).astype(mx.float32);mx.eval(h,[c.state for c in cache])
  for s in range(STATES_PER_PROMPT):
   states.append((pi,s,h)); logits=model.lm_head(h.astype(mx.float16)) if hasattr(model,'lm_head') else model.model.embed_tokens.as_linear(h.astype(mx.float16));y=mx.argmax(logits);mx.eval(y);token=int(y.item());b=model.model(mx.array([[token]],dtype=mx.int32),cache=cache);h=b[:,-1,:].reshape((D,)).astype(mx.float32);mx.eval(h,[c.state for c in cache])
 return states

def timed(fn):
 t=time.perf_counter();y=fn();mx.eval(y);return int(y.item()),(time.perf_counter()-t)*1e3

def main():
 model,tok=load(MODEL);model.set_dtype(mx.float16);mx.eval(model.parameters());base.MODEL=MODEL
 w16,high,low=prepare(model);V,D=[int(x) for x in w16.shape];assert D%32==0
 its=D//32;factors=divisors(its);ks=base.make_kernels();ups={f:make_upper(D,f) for f in factors};states=collect(model,tok,D)
 h0=states[0][2]
 for f in factors:y=decision(ups[f],ks,high,low,h0,V);mx.eval(y)
 yd=base.call_dense(ks[0],w16,h0,V);mx.eval(yd);mx.synchronize()
 pooled={f:0.0 for f in factors};rows=[]
 for si,(pi,pos,h) in enumerate(states):
  dy,_=timed(lambda:base.call_dense(ks[0],w16,h,V));times={f:[] for f in factors};decs={f:[] for f in factors}
  for r in range(REPEATS):
   order=factors if (si+r)%2==0 else list(reversed(factors))
   for f in order:
    y,t=timed(lambda f=f:decision(ups[f],ks,high,low,h,V));times[f].append(t);decs[f].append(y);pooled[f]+=t
  row={'state':si,'prompt':pi,'position':pos}
  for f in factors:
   row[f'u{f}_exact']=all(y==dy for y in decs[f]);row[f'u{f}_median_ms']=med(times[f]);row[f'u1_over_u{f}']=med(times[1])/med(times[f])
  rows.append(row)
 out={'kind':'proofbits_qwen_partial_unroll_headstress','model':MODEL,'V':V,'D':D,'iterations':its,'factors':factors,'n_states':len(states),'repeats':REPEATS,'rows':rows}
 for f in factors:
  out[f'u{f}_all_exact']=all(r[f'u{f}_exact'] for r in rows);out[f'u{f}_pooled_ms']=pooled[f];out[f'u1_over_u{f}_pooled']=pooled[1]/pooled[f];out[f'u1_over_u{f}_median']=med([r[f'u1_over_u{f}'] for r in rows])
 out['best_factor_by_pooled']=max(factors,key=lambda f:pooled[1]/pooled[f]);out['best_pooled_speedup']=max(pooled[1]/pooled[f] for f in factors)
 Path('experiments/artifacts').mkdir(parents=True,exist_ok=True);Path('experiments/artifacts/proofbits_mlx_qwen_unroll_headstress.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
