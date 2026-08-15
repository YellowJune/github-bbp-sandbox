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
 'Compare TCP and UDP for a latency-sensitive application.',
 'Give a concise proof that the square root of 2 is irrational.',
 'Explain the difference between correlation and causation.',
 'Write pseudocode for breadth-first search on a graph.',
 'Summarize why cache locality matters in matrix computation.',
 'Explain photosynthesis to a high-school student.',
 'Derive the quadratic formula from completing the square.',
]
TOKENS=48


def med(x):return float(statistics.median(x))

def ids(tok,p):
 try:x=tok.encode(p)
 except Exception:x=tok(p)['input_ids']
 return mx.array(x,dtype=mx.int32)

def run(model,tok,prompt,mode,ks,dual_k,w16,high,low,n):
 V,D=[int(x) for x in w16.shape];kv=cache_mod.make_prompt_cache(model)
 b=model.model(ids(tok,prompt)[None],cache=kv);h=b[:,-1,:].reshape((D,)).astype(mx.float32);mx.eval(h,[c.state for c in kv])
 toks=[];times=[]
 for _ in range(n):
  t0=time.perf_counter()
  if mode=='dense':y=base.call_dense(ks[0],w16,h,V)
  elif mode=='current':y=base.call_proofbits(ks[1],ks[2],ks[3],high,low,h,V,False)
  elif mode=='dual':y=dual.dual_decision(dual_k,ks[3],high,low,h,V,False)
  elif mode=='native':y=base.call_native(w16,h)
  else:raise ValueError(mode)
  mx.eval(y);token=int(y.item());toks.append(token)
  b=model.model(mx.array([[token]],dtype=mx.int32),cache=kv);h=b[:,-1,:].reshape((D,)).astype(mx.float32);mx.eval(h,[c.state for c in kv]);times.append((time.perf_counter()-t0)*1e3)
 return {'tokens':toks,'median_ms':med(times),'mean_ms':float(statistics.mean(times)),'total_ms':float(sum(times))}

def main():
 model,tok=load(MODEL);model.set_dtype(mx.float16);mx.eval(model.parameters());base.MODEL=MODEL
 w16,high,low=dual.prepare(model);ks=base.make_kernels();dual_k=dual.make_dual();modes=['dense','current','dual','native']
 for m in modes:run(model,tok,PROMPTS[0],m,ks,dual_k,w16,high,low,4)
 mx.synchronize();rows=[];sums={m:0.0 for m in modes}
 orders=[['dense','current','dual','native'],['dual','native','dense','current'],['native','current','dual','dense'],['current','dense','native','dual']]
 for i,p in enumerate(PROMPTS):
  order=orders[i%4];res={}
  for m in order:
   gc.collect();mx.clear_cache();res[m]=run(model,tok,p,m,ks,dual_k,w16,high,low,TOKENS);sums[m]+=res[m]['total_ms']
  d,c,u,n=res['dense'],res['current'],res['dual'],res['native']
  rows.append({'prompt_index':i,'order':order,'current_exact':c['tokens']==d['tokens'],'dual_exact':u['tokens']==d['tokens'],'native_equal_dual':n['tokens']==u['tokens'],
   'dense_ms':d['median_ms'],'current_ms':c['median_ms'],'dual_ms':u['median_ms'],'native_ms':n['median_ms'],
   'dual_vs_current':c['median_ms']/u['median_ms'],'dense_over_dual':d['median_ms']/u['median_ms'],'native_over_dual':n['median_ms']/u['median_ms']})
 out={'kind':'proofbits_fp16_dual_bound_stress','model':MODEL,'n_prompts':len(PROMPTS),'tokens_per_prompt':TOKENS,'matched_tokens':len(PROMPTS)*TOKENS,'rows':rows,
  'all_current_exact':all(r['current_exact'] for r in rows),'all_dual_exact':all(r['dual_exact'] for r in rows),'native_all_equal_dual':all(r['native_equal_dual'] for r in rows),
  'dual_vs_current_median':med([r['dual_vs_current'] for r in rows]),'dual_vs_current_mean':float(statistics.mean(r['dual_vs_current'] for r in rows)),
  'native_over_dual_median':med([r['native_over_dual'] for r in rows]),'native_over_dual_mean':float(statistics.mean(r['native_over_dual'] for r in rows)),'native_over_dual_min':min(r['native_over_dual'] for r in rows),
  'pooled_current_over_dual':sums['current']/sums['dual'],'pooled_dense_over_dual':sums['dense']/sums['dual'],'pooled_native_over_dual':sums['native']/sums['dual'],
  'total_ms':sums}
 Path('experiments/artifacts').mkdir(parents=True,exist_ok=True);Path('experiments/artifacts/proofbits_mlx_dual_bound_stress.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
