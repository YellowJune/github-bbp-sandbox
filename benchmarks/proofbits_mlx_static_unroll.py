import gc
import json
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
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
REPS=25
D=640


def term(expr):
 return f'''{{ uint j={expr}; uchar hb=high[base+j]; ushort ws=(ushort)(hb & (uchar)0x80); ushort hs=(hidden[j]<0.0f)?(ushort)0x80:(ushort)0x00; ushort suffix=(ws==hs)?(ushort)0x00FF:(ushort)0x0000; ushort raw=((ushort)hb<<8)|suffix; acc=fma(hidden[j],(float)as_type<half>(raw),acc); }}'''

STATIC_LOOP_SRC=f'''
 uint row=threadgroup_position_in_grid.x;
 uint lane=thread_index_in_simdgroup;
 ulong base=(ulong)row*{D}ul;
 float acc=0.0f;
 for(uint j=lane;j<{D}u;j+=32u){{
   uchar hb=high[base+j];
   ushort ws=(ushort)(hb & (uchar)0x80);
   ushort hs=(hidden[j]<0.0f)?(ushort)0x80:(ushort)0x00;
   ushort suffix=(ws==hs)?(ushort)0x00FF:(ushort)0x0000;
   ushort raw=((ushort)hb<<8)|suffix;
   acc=fma(hidden[j],(float)as_type<half>(raw),acc);
 }}
 float total=simd_sum(acc); if(lane==0)upper[row]=total;
'''

UNROLLED_SRC='''
 uint row=threadgroup_position_in_grid.x;
 uint lane=thread_index_in_simdgroup;
 ulong base=(ulong)row*640ul;
 float acc=0.0f;
''' + '\n'.join(term(f'lane+{32*k}u') for k in range(20)) + '''
 float total=simd_sum(acc); if(lane==0)upper[row]=total;
'''


def med(x):return float(statistics.median(x))

def make_kernels():
 a=mx.fast.metal_kernel(name='pb_upper_static640',input_names=['high','hidden'],output_names=['upper'],source=STATIC_LOOP_SRC)
 b=mx.fast.metal_kernel(name='pb_upper_unrolled640',input_names=['high','hidden'],output_names=['upper'],source=UNROLLED_SRC)
 return a,b

def call_pb(upper_k,ks,high,low,h,V):
 U=upper_k(inputs=[high,h],grid=(V*32,1,1),threadgroup=(32,1,1),output_shapes=[(V,)],output_dtypes=[mx.float32],init_value=0.0)[0]
 p=mx.reshape(mx.argmax(U).astype(mx.uint32),(1,))
 B=ks[2](inputs=[high,low,h,p],grid=(32,1,1),threadgroup=(32,1,1),output_shapes=[(1,)],output_dtypes=[mx.float32],init_value=0.0)[0]
 E=ks[3](inputs=[high,low,h,U,B],grid=(V*32,1,1),threadgroup=(32,1,1),output_shapes=[(V,)],output_dtypes=[mx.float32],init_value=-3.402823466e38)[0]
 return mx.argmax(E).astype(mx.uint32)

def ids(tok,p):
 try:x=tok.encode(p)
 except Exception:x=tok(p)['input_ids']
 return mx.array(x,dtype=mx.int32)

def hidden(model,tok,p):
 kv=cache_mod.make_prompt_cache(model);b=model.model(ids(tok,p)[None],cache=kv);h=b[:,-1,:].reshape((D,)).astype(mx.float32);mx.eval(h,[c.state for c in kv]);return h

def stage_time(k,high,h,V):
 xs=[]
 for _ in range(REPS):
  t=time.perf_counter();U=k(inputs=[high,h],grid=(V*32,1,1),threadgroup=(32,1,1),output_shapes=[(V,)],output_dtypes=[mx.float32],init_value=0.0)[0];mx.eval(U);xs.append((time.perf_counter()-t)*1e3)
 return med(xs)

def run(model,tok,prompt,mode,ks,opts,w16,high,low,n):
 V,DD=[int(x) for x in w16.shape];assert DD==D;kv=cache_mod.make_prompt_cache(model);b=model.model(ids(tok,prompt)[None],cache=kv);h=b[:,-1,:].reshape((D,)).astype(mx.float32);mx.eval(h,[c.state for c in kv]);toks=[];times=[]
 for _ in range(n):
  t=time.perf_counter()
  if mode=='dense':y=base.call_dense(ks[0],w16,h,V)
  elif mode=='native':y=base.call_native(w16,h)
  elif mode=='runtime':y=base.call_proofbits(ks[1],ks[2],ks[3],high,low,h,V,False)
  elif mode=='static':y=call_pb(opts[0],ks,high,low,h,V)
  elif mode=='unroll':y=call_pb(opts[1],ks,high,low,h,V)
  else:raise ValueError(mode)
  mx.eval(y);token=int(y.item());toks.append(token);b=model.model(mx.array([[token]],dtype=mx.int32),cache=kv);h=b[:,-1,:].reshape((D,)).astype(mx.float32);mx.eval(h,[c.state for c in kv]);times.append((time.perf_counter()-t)*1e3)
 return {'tokens':toks,'median_ms':med(times),'mean_ms':float(statistics.mean(times))}

def main():
 model,tok=load(MODEL);model.set_dtype(mx.float16);mx.eval(model.parameters());base.MODEL=MODEL
 w16,high,low=dual.prepare(model);V,DD=[int(x) for x in w16.shape];assert DD==D;ks=base.make_kernels();opts=make_kernels();h=hidden(model,tok,PROMPTS[0])
 for k in [ks[1],*opts]:U=k(inputs=[high,h],grid=(V*32,1,1),threadgroup=(32,1,1),output_shapes=[(V,)],output_dtypes=[mx.float32],init_value=0.0)[0];mx.eval(U)
 stage={'runtime':stage_time(ks[1],high,h,V),'static':stage_time(opts[0],high,h,V),'unroll':stage_time(opts[1],high,h,V)}
 modes=['dense','runtime','static','unroll','native']
 for m in modes:run(model,tok,PROMPTS[0],m,ks,opts,w16,high,low,4)
 mx.synchronize();rows=[]
 for i,p in enumerate(PROMPTS):
  order=modes[i%len(modes):]+modes[:i%len(modes)];res={}
  for m in order:
   gc.collect();mx.clear_cache();res[m]=run(model,tok,p,m,ks,opts,w16,high,low,TOKENS)
  d,r,s,u,n=res['dense'],res['runtime'],res['static'],res['unroll'],res['native']
  rows.append({'prompt_index':i,'order':order,'runtime_exact':r['tokens']==d['tokens'],'static_exact':s['tokens']==d['tokens'],'unroll_exact':u['tokens']==d['tokens'],
    'dense_ms':d['median_ms'],'runtime_ms':r['median_ms'],'static_ms':s['median_ms'],'unroll_ms':u['median_ms'],'native_ms':n['median_ms'],
    'runtime_over_static':r['median_ms']/s['median_ms'],'runtime_over_unroll':r['median_ms']/u['median_ms'],'native_over_static':n['median_ms']/s['median_ms'],'native_over_unroll':n['median_ms']/u['median_ms']})
 out={'kind':'proofbits_static_unroll_upper','model':MODEL,'D':D,'stage_upper_ms':stage,'rows':rows,
  'all_static_exact':all(r['static_exact'] for r in rows),'all_unroll_exact':all(r['unroll_exact'] for r in rows),
  'median_runtime_over_static':med([r['runtime_over_static'] for r in rows]),'median_runtime_over_unroll':med([r['runtime_over_unroll'] for r in rows]),
  'median_native_over_static':med([r['native_over_static'] for r in rows]),'min_native_over_static':min(r['native_over_static'] for r in rows),
  'median_native_over_unroll':med([r['native_over_unroll'] for r in rows]),'min_native_over_unroll':min(r['native_over_unroll'] for r in rows),
  'note':'Static and unrolled upper kernels preserve the original lane assignment and per-lane FMA order (j=lane+32*k), so matched decision semantics should be identical; exact sequences are checked empirically.'}
 Path('experiments/artifacts').mkdir(parents=True,exist_ok=True);Path('experiments/artifacts/proofbits_mlx_static_unroll.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
