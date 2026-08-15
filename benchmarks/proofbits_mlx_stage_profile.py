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

MODEL='mlx-community/gemma-3-270m-bf16'
PROMPTS=[
 'Explain why entropy is measured with logarithms.',
 'Solve step by step: if x + 1/x = 3, compute x^6 + 1/x^6.',
 'Write a Python function for longest increasing contiguous subarray.',
 'Explain why memory bandwidth matters for autoregressive inference.',
 'Describe natural selection without using teleological language.',
]
REPS=25


def med(x): return float(statistics.median(x))

def ids(tok,p):
 try:x=tok.encode(p)
 except Exception:x=tok(p)['input_ids']
 return mx.array(x,dtype=mx.int32)

def hidden(model,tok,p,D):
 kv=cache_mod.make_prompt_cache(model)
 b=model.model(ids(tok,p)[None],cache=kv)
 h=b[:,-1,:].reshape((D,)).astype(mx.float32)
 mx.eval(h,[c.state for c in kv]); return h

def timed(fn):
 xs=[]
 for _ in range(REPS):
  t=time.perf_counter(); z=fn(); mx.eval(z); xs.append((time.perf_counter()-t)*1e3)
 return med(xs), float(statistics.mean(xs))

def main():
 model,tok=load(MODEL); model.set_dtype(mx.float16); mx.eval(model.parameters()); base.MODEL=MODEL
 w16,high,low=base.prepare_weights(model); V,D=[int(x) for x in w16.shape]
 kd,ku,kp,kr=base.make_kernels()
 rows=[]
 for pi,prompt in enumerate(PROMPTS):
  h=hidden(model,tok,prompt,D)
  # compile
  U=ku(inputs=[high,h],grid=(V*32,1,1),threadgroup=(32,1,1),output_shapes=[(V,)],output_dtypes=[mx.float32],init_value=0.0)[0];mx.eval(U)
  pp=mx.reshape(mx.argmax(U).astype(mx.uint32),(1,));mx.eval(pp)
  B=kp(inputs=[high,low,h,pp],grid=(32,1,1),threadgroup=(32,1,1),output_shapes=[(1,)],output_dtypes=[mx.float32],init_value=0.0)[0];mx.eval(B)
  E=kr(inputs=[high,low,h,U,B],grid=(V*32,1,1),threadgroup=(32,1,1),output_shapes=[(V,)],output_dtypes=[mx.float32],init_value=-3.402823466e38)[0];mx.eval(E)
  y=mx.argmax(E);mx.eval(y)

  upper_med,upper_mean=timed(lambda: ku(inputs=[high,h],grid=(V*32,1,1),threadgroup=(32,1,1),output_shapes=[(V,)],output_dtypes=[mx.float32],init_value=0.0)[0])
  # Keep one fully-evaluated U for reduction-only timings.
  U=ku(inputs=[high,h],grid=(V*32,1,1),threadgroup=(32,1,1),output_shapes=[(V,)],output_dtypes=[mx.float32],init_value=0.0)[0];mx.eval(U)
  uarg_med,uarg_mean=timed(lambda: mx.argmax(U))
  pp=mx.reshape(mx.argmax(U).astype(mx.uint32),(1,));mx.eval(pp)
  pilot_med,pilot_mean=timed(lambda: kp(inputs=[high,low,h,pp],grid=(32,1,1),threadgroup=(32,1,1),output_shapes=[(1,)],output_dtypes=[mx.float32],init_value=0.0)[0])
  B=kp(inputs=[high,low,h,pp],grid=(32,1,1),threadgroup=(32,1,1),output_shapes=[(1,)],output_dtypes=[mx.float32],init_value=0.0)[0];mx.eval(B)
  refine_med,refine_mean=timed(lambda: kr(inputs=[high,low,h,U,B],grid=(V*32,1,1),threadgroup=(32,1,1),output_shapes=[(V,)],output_dtypes=[mx.float32],init_value=-3.402823466e38)[0])
  E=kr(inputs=[high,low,h,U,B],grid=(V*32,1,1),threadgroup=(32,1,1),output_shapes=[(V,)],output_dtypes=[mx.float32],init_value=-3.402823466e38)[0];mx.eval(E)
  earg_med,earg_mean=timed(lambda: mx.argmax(E))
  full_med,full_mean=timed(lambda: base.call_proofbits(ku,kp,kr,high,low,h,V,False))
  dense_med,dense_mean=timed(lambda: base.call_dense(kd,w16,h,V))
  native_med,native_mean=timed(lambda: base.call_native(w16,h))
  s=mx.sum(U>=B);mx.eval(s)
  rows.append({'prompt_index':pi,'survivors':int(s.item()),
   'upper_ms':upper_med,'upper_argmax_ms':uarg_med,'pilot_ms':pilot_med,
   'refine_ms':refine_med,'final_argmax_ms':earg_med,'sum_isolated_ms':upper_med+uarg_med+pilot_med+refine_med+earg_med,
   'full_proofbits_ms':full_med,'dense_custom_ms':dense_med,'native_mlx_ms':native_med,
   'dense_over_pb':dense_med/full_med,'native_over_pb':native_med/full_med})
 out={'kind':'proofbits_mlx_stage_profile','model':MODEL,'runtime_dtype':'float16','reps':REPS,'rows':rows,
      'median_upper_ms':med([r['upper_ms'] for r in rows]),'median_upper_argmax_ms':med([r['upper_argmax_ms'] for r in rows]),
      'median_pilot_ms':med([r['pilot_ms'] for r in rows]),'median_refine_ms':med([r['refine_ms'] for r in rows]),
      'median_final_argmax_ms':med([r['final_argmax_ms'] for r in rows]),'median_full_pb_ms':med([r['full_proofbits_ms'] for r in rows]),
      'note':'Isolated stages force evaluation after each stage and therefore are diagnostic, not additive serving timings. Full ProofBits is measured lazily as the real composed decision graph.'}
 Path('experiments/artifacts').mkdir(parents=True,exist_ok=True);Path('experiments/artifacts/proofbits_mlx_stage_profile.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
