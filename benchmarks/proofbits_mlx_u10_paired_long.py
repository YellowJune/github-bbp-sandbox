import gc
import json
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models import cache as cache_mod

sys.path.insert(0, str(Path(__file__).resolve().parent))
import proofbits_mlx_integrated_decode as base
import proofbits_mlx_dual_bound as dual
import proofbits_mlx_u10_stress as u10mod

MODEL='mlx-community/gemma-3-270m-bf16'
D=640
TOKENS=96
ROUNDS=4
PROMPTS=[
 'Explain why entropy is measured with logarithms.',
 'Solve step by step: if x + 1/x = 3, compute x^6 + 1/x^6.',
 'Write a Python function for longest increasing contiguous subarray.',
 'Explain why memory bandwidth matters for autoregressive inference.',
 'Give a concise proof that the square root of 2 is irrational.',
 'Derive the quadratic formula from completing the square.',
]

def med(x): return float(statistics.median(x))

def ids(tok,p):
 try:x=tok.encode(p)
 except Exception:x=tok(p)['input_ids']
 return mx.array(x,dtype=mx.int32)

def traj(model,tok,prompt,mode,ks,u10,w16,high,low,n):
 V,DD=[int(x) for x in w16.shape];assert DD==D
 cache=cache_mod.make_prompt_cache(model)
 body=model.model(ids(tok,prompt)[None],cache=cache)
 h=body[:,-1,:].reshape((D,)).astype(mx.float32);mx.eval(h,[c.state for c in cache])
 toks=[];total=[];head=[];bodyts=[]
 for _ in range(n):
  t0=time.perf_counter();th=time.perf_counter()
  if mode=='current': y=base.call_proofbits(ks[1],ks[2],ks[3],high,low,h,V,False)
  elif mode=='u10': y=u10mod.u10_decision(u10,ks,high,low,h,V)
  elif mode=='dense': y=base.call_dense(ks[0],w16,h,V)
  elif mode=='native': y=base.call_native(w16,h)
  else: raise ValueError(mode)
  mx.eval(y);token=int(y.item());toks.append(token);head.append((time.perf_counter()-th)*1e3)
  tb=time.perf_counter();body=model.model(mx.array([[token]],dtype=mx.int32),cache=cache);h=body[:,-1,:].reshape((D,)).astype(mx.float32);mx.eval(h,[c.state for c in cache]);bodyts.append((time.perf_counter()-tb)*1e3);total.append((time.perf_counter()-t0)*1e3)
 return {'tokens':toks,'sum_total_ms':float(sum(total)),'sum_head_ms':float(sum(head)),'sum_body_ms':float(sum(bodyts)),'median_total_ms':med(total),'median_head_ms':med(head),'median_body_ms':med(bodyts)}

def main():
 model,tok=load(MODEL);model.set_dtype(mx.float16);mx.eval(model.parameters());base.MODEL=MODEL
 w16,high,low=dual.prepare(model);ks=base.make_kernels();u10=u10mod.make_u10()
 # JIT all paths.
 for m in ['current','u10','dense','native']: traj(model,tok,PROMPTS[0],m,ks,u10,w16,high,low,4)
 mx.synchronize()
 rows=[];paired=[];sum_current=0.0;sum_u10=0.0;sum_current_head=0.0;sum_u10_head=0.0
 all_pair_exact=True;all_dense_exact=True
 for pi,p in enumerate(PROMPTS):
  # One independent dense and native trajectory per prompt supplies matched correctness and serving reference without dominating thermal position.
  gc.collect();mx.clear_cache();d=traj(model,tok,p,'dense',ks,u10,w16,high,low,TOKENS)
  gc.collect();mx.clear_cache();n=traj(model,tok,p,'native',ks,u10,w16,high,low,TOKENS)
  prompt_pairs=[]
  for r in range(ROUNDS):
   order=['current','u10'] if r%2==0 else ['u10','current']
   res={}
   for m in order:
    gc.collect();mx.clear_cache();res[m]=traj(model,tok,p,m,ks,u10,w16,high,low,TOKENS)
   c,u=res['current'],res['u10']
   exact=(c['tokens']==u['tokens']==d['tokens'])
   all_pair_exact &= exact; all_dense_exact &= (c['tokens']==d['tokens'] and u['tokens']==d['tokens'])
   rt=c['sum_total_ms']/u['sum_total_ms'];rh=c['sum_head_ms']/u['sum_head_ms']
   sum_current += c['sum_total_ms'];sum_u10 += u['sum_total_ms'];sum_current_head += c['sum_head_ms'];sum_u10_head += u['sum_head_ms']
   rec={'prompt_index':pi,'round':r+1,'order':order,'exact':exact,'current_sum_total_ms':c['sum_total_ms'],'u10_sum_total_ms':u['sum_total_ms'],'total_ratio':rt,'current_sum_head_ms':c['sum_head_ms'],'u10_sum_head_ms':u['sum_head_ms'],'head_ratio':rh,'current_median_total_ms':c['median_total_ms'],'u10_median_total_ms':u['median_total_ms']}
   paired.append(rec);prompt_pairs.append(rec)
  rows.append({'prompt_index':pi,'dense_sum_total_ms':d['sum_total_ms'],'native_sum_total_ms':n['sum_total_ms'],'native_equal_dense':n['tokens']==d['tokens'],'paired_total_ratio_median':med([x['total_ratio'] for x in prompt_pairs]),'paired_head_ratio_median':med([x['head_ratio'] for x in prompt_pairs]),'all_pairs_exact':all(x['exact'] for x in prompt_pairs)})
 ratios=[x['total_ratio'] for x in paired];hr=[x['head_ratio'] for x in paired]
 out={'kind':'proofbits_u10_long_paired_abba','model':MODEL,'runtime_dtype':'float16','n_prompts':len(PROMPTS),'tokens_per_trajectory':TOKENS,'paired_rounds_per_prompt':ROUNDS,'paired_trajectories':len(paired),'paired_generated_tokens_per_mode':len(PROMPTS)*ROUNDS*TOKENS,
      'all_paired_exact':all_pair_exact,'all_dense_exact':all_dense_exact,'rows':rows,'pairs':paired,
      'paired_total_ratio_median':med(ratios),'paired_total_ratio_mean':float(statistics.mean(ratios)),'paired_total_ratio_min':min(ratios),'paired_total_positive_fraction':float(sum(x>1 for x in ratios)/len(ratios)),
      'paired_head_ratio_median':med(hr),'paired_head_ratio_mean':float(statistics.mean(hr)),'paired_head_ratio_min':min(hr),'paired_head_positive_fraction':float(sum(x>1 for x in hr)/len(hr)),
      'pooled_current_over_u10':sum_current/sum_u10,'pooled_current_head_over_u10':sum_current_head/sum_u10_head,
      'note':'Tight paired AB/BA validation: each prompt has four 96-token current/u10 pairs with alternating order. Ratios use trajectory sums, not medians. Dense/native are measured once per prompt only for exactness/reference. Both compared methods start from identical prompt/cache state and generate matched trajectories.'}
 Path('experiments/artifacts').mkdir(parents=True,exist_ok=True);Path('experiments/artifacts/proofbits_mlx_u10_paired_long.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
