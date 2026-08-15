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

ATOMIC_DUAL_SRC=r'''
 uint row=threadgroup_position_in_grid.x;
 uint lane=thread_index_in_simdgroup;
 uint D=(uint)hidden_shape[0];
 ulong base=(ulong)row*(ulong)D;
 float au=0.0f, al=0.0f;
 for(uint j=lane;j<D;j+=32u){
   uchar hb=high[base+j];
   ushort ws=(ushort)(hb & (uchar)0x80);
   ushort hs=(hidden[j] < 0.0f) ? (ushort)0x80 : (ushort)0x00;
   bool same=(ws==hs);
   ushort raw_u=((ushort)hb<<8) | (same ? (ushort)0x00FF : (ushort)0x0000);
   ushort raw_l=((ushort)hb<<8) | (same ? (ushort)0x0000 : (ushort)0x00FF);
   float x=hidden[j];
   au=fma(x,(float)as_type<half>(raw_u),au);
   al=fma(x,(float)as_type<half>(raw_l),al);
 }
 float su=simd_sum(au), sl=simd_sum(al);
 if(lane==0){
   uint ub=as_type<uint>(su);
   atomic_store_explicit(&upper_bits[row],ub,memory_order_relaxed);
   uint b=as_type<uint>(sl);
   uint ord=(b & 0x80000000u) ? ~b : (b ^ 0x80000000u);
   atomic_fetch_max_explicit(&bound_order[0],ord,memory_order_relaxed);
 }
'''

ATOMIC_REFINE_SRC=r'''
 uint row=threadgroup_position_in_grid.x;
 uint lane=thread_index_in_simdgroup;
 uint D=(uint)hidden_shape[0];
 uint ord=bound_order[0];
 uint bb=(ord & 0x80000000u) ? (ord ^ 0x80000000u) : ~ord;
 float B=as_type<float>(bb);
 float U=as_type<float>(upper_bits[row]);
 if(U < B){ if(lane==0) exact[row]=-3.402823466e+38f; return; }
 ulong base=(ulong)row*(ulong)D;
 float acc=0.0f;
 for(uint j=lane;j<D;j+=32u){
   ushort raw=((ushort)high[base+j]<<8)|(ushort)low[base+j];
   acc=fma(hidden[j],(float)as_type<half>(raw),acc);
 }
 float total=simd_sum(acc);
 if(lane==0) exact[row]=total;
'''

def med(x):return float(statistics.median(x))

def make_atomic():
 k=mx.fast.metal_kernel(name='pb_atomic_dual_upper',input_names=['high','hidden'],output_names=['upper_bits','bound_order'],source=ATOMIC_DUAL_SRC,atomic_outputs=True)
 r=mx.fast.metal_kernel(name='pb_atomic_dual_refine',input_names=['high','low','hidden','upper_bits','bound_order'],output_names=['exact'],source=ATOMIC_REFINE_SRC)
 return k,r

def atomic_decision(k,r,high,low,h,V):
 ub,bo=k(inputs=[high,h],grid=(V*32,1,1),threadgroup=(32,1,1),output_shapes=[(V,),(1,)],output_dtypes=[mx.uint32,mx.uint32],init_value=0)
 E=r(inputs=[high,low,h,ub,bo],grid=(V*32,1,1),threadgroup=(32,1,1),output_shapes=[(V,)],output_dtypes=[mx.float32],init_value=-3.402823466e38)[0]
 return mx.argmax(E).astype(mx.uint32)

def ids(tok,p):
 try:x=tok.encode(p)
 except Exception:x=tok(p)['input_ids']
 return mx.array(x,dtype=mx.int32)

def run(model,tok,prompt,mode,baseks,dual_k,atomics,w16,high,low,n):
 V,D=[int(x) for x in w16.shape];kv=cache_mod.make_prompt_cache(model);b=model.model(ids(tok,prompt)[None],cache=kv);h=b[:,-1,:].reshape((D,)).astype(mx.float32);mx.eval(h,[c.state for c in kv]);toks=[];times=[]
 for _ in range(n):
  t=time.perf_counter()
  if mode=='dense':y=base.call_dense(baseks[0],w16,h,V)
  elif mode=='native':y=base.call_native(w16,h)
  elif mode=='current':y=base.call_proofbits(baseks[1],baseks[2],baseks[3],high,low,h,V,False)
  elif mode=='dual':y=dual.dual_decision(dual_k,baseks[3],high,low,h,V,False)
  elif mode=='atomic':y=atomic_decision(atomics[0],atomics[1],high,low,h,V)
  else:raise ValueError(mode)
  mx.eval(y);token=int(y.item());toks.append(token);b=model.model(mx.array([[token]],dtype=mx.int32),cache=kv);h=b[:,-1,:].reshape((D,)).astype(mx.float32);mx.eval(h,[c.state for c in kv]);times.append((time.perf_counter()-t)*1e3)
 return {'tokens':toks,'median_ms':med(times),'mean_ms':float(statistics.mean(times))}

def main():
 model,tok=load(MODEL);model.set_dtype(mx.float16);mx.eval(model.parameters());base.MODEL=MODEL
 w16,high,low=dual.prepare(model);baseks=base.make_kernels();dual_k=dual.make_dual();atomics=make_atomic();modes=['dense','current','dual','atomic','native']
 for m in modes:run(model,tok,PROMPTS[0],m,baseks,dual_k,atomics,w16,high,low,4)
 mx.synchronize();rows=[]
 for i,p in enumerate(PROMPTS):
  order=modes[i%len(modes):]+modes[:i%len(modes)];res={}
  for m in order:
   gc.collect();mx.clear_cache();res[m]=run(model,tok,p,m,baseks,dual_k,atomics,w16,high,low,TOKENS)
  d,c,u,a,n=res['dense'],res['current'],res['dual'],res['atomic'],res['native']
  rows.append({'prompt_index':i,'order':order,'current_exact':c['tokens']==d['tokens'],'dual_exact':u['tokens']==d['tokens'],'atomic_exact':a['tokens']==d['tokens'],'native_equal_atomic':n['tokens']==a['tokens'],
    'dense_ms':d['median_ms'],'current_ms':c['median_ms'],'dual_ms':u['median_ms'],'atomic_ms':a['median_ms'],'native_ms':n['median_ms'],
    'dual_over_atomic':u['median_ms']/a['median_ms'],'current_over_atomic':c['median_ms']/a['median_ms'],'native_over_atomic':n['median_ms']/a['median_ms']})
 out={'kind':'proofbits_atomic_dual_fused_lower_reduction','model':MODEL,'n_prompts':len(PROMPTS),'tokens_per_prompt':TOKENS,'rows':rows,
   'all_atomic_exact':all(r['atomic_exact'] for r in rows),'median_dual_over_atomic':med([r['dual_over_atomic'] for r in rows]),'mean_dual_over_atomic':float(statistics.mean(r['dual_over_atomic'] for r in rows)),
   'median_current_over_atomic':med([r['current_over_atomic'] for r in rows]),'median_native_over_atomic':med([r['native_over_atomic'] for r in rows]),'min_native_over_atomic':min(r['native_over_atomic'] for r in rows),
   'note':'AtomicDual stores U as raw float bits in atomic uint outputs and computes global max lower bound by ordered-float atomic max inside the same high-byte pass. This removes the V-sized lower buffer and separate max(L) reduction.'}
 Path('experiments/artifacts').mkdir(parents=True,exist_ok=True);Path('experiments/artifacts/proofbits_mlx_atomic_dual.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
