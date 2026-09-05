#!/usr/bin/env python3
"""Real-model batched boundary-block repair for MOSAIC-RAG.

This experiment follows the negative token-granular pilot: instead of launching one
forward per repaired token, it recomputes a contiguous prefix at each retrieved-document
boundary in one model call. The experiment measures whether batching repairs changes the
latency-quality frontier without claiming that boundary blocks are the final selector.
"""
from __future__ import annotations

import argparse, gc, itertools, json, statistics, time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import experiments.mosaic_rag_real as mr

LegacyCache = mr.LegacyCache


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--model', default='HuggingFaceTB/SmolLM2-135M-Instruct')
    p.add_argument('--examples', type=int, default=3)
    p.add_argument('--offset', type=int, default=4)
    p.add_argument('--docs', type=int, default=4)
    p.add_argument('--max-chunk-tokens', type=int, default=24)
    p.add_argument('--blocks', default='1,2,4,8,12,24')
    p.add_argument('--output', default='experiments/artifacts/mosaic_rag_block_repair.json')
    return p.parse_args()


def clone(cache: LegacyCache) -> List[List[torch.Tensor]]:
    return [[k.clone(), v.clone()] for k,v in cache]


def block_repair(model, data: Dict[str,Any], approx: LegacyCache, block_tokens: int, device: torch.device):
    current=clone(approx)
    ctx=torch.tensor(data['ctx_ids'], dtype=torch.long, device=device).unsqueeze(0)
    starts=list(data['starts'])
    L=len(data['ctx_ids'])
    repaired=0
    mr.sync(); t0=time.perf_counter()
    with torch.inference_mode():
        for c in range(1, len(starts)):
            st=starts[c]
            doc_end=starts[c+1] if c+1 < len(starts) else L
            en=min(doc_end, st+block_tokens)
            if en <= st:
                continue
            prefix: LegacyCache=tuple((kv[0][:,:,:st,:], kv[1][:,:,:st,:]) for kv in current)
            out=mr.model_forward_with_past(model, ctx[:,st:en], prefix, st, use_cache=True)
            new=mr.cache_to_legacy(out.past_key_values)
            q=en-st
            for l in range(len(current)):
                current[l][0][:,:,st:en,:]=new[l][0][:,:,-q:,:]
                current[l][1][:,:,st:en,:]=new[l][1][:,:,-q:,:]
            repaired += q
            del out,new,prefix
    mr.sync(); ms=(time.perf_counter()-t0)*1000.0
    return tuple((x[0],x[1]) for x in current), ms, repaired


def agg(rows):
    out={}
    for m in sorted({r['method'] for r in rows}):
        rr=[r for r in rows if r['method']==m]
        d={'n':len(rr)}
        for k in ['recompute_fraction','repair_ms','query_ms','warm_ttft_ms','speedup','amortized_speedup_16','kl','cosine','top1_agree','gold_nll_delta']:
            vals=[float(r[k]) for r in rr]
            d[k+'_mean']=float(np.mean(vals)); d[k+'_median']=float(np.median(vals)); d[k+'_std']=float(np.std(vals))
        out[m]=d
    return out


def main():
    args=parse_args()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    device_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GitHub-hosted CPU'
    print('DEVICE='+device_name)
    tok=AutoTokenizer.from_pretrained(args.model, use_fast=True)
    try:
        model=AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, attn_implementation='eager', low_cpu_mem_usage=True).to(device)
    except TypeError:
        model=AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, low_cpu_mem_usage=True).to(device)
    model.eval()

    stream=load_dataset('hotpotqa/hotpot_qa','distractor',split='validation',streaming=True)
    raw=list(itertools.islice(stream,args.offset+args.examples))
    raw=raw[args.offset:args.offset+args.examples]
    data=[mr.build_tokenized_context(x,tok,args.docs,args.max_chunk_tokens) for x in raw]
    blocks=[int(x) for x in args.blocks.split(',') if x.strip()]
    rows=[]
    for i,d in enumerate(data):
        print(f"example {i+1}/{len(data)} id={d['id']} ctx={len(d['ctx_ids'])}")
        approx,_,compile_ms=mr.independent_cache(model,tok,d,device,collect_features=False)
        ctx=torch.tensor(d['ctx_ids'],dtype=torch.long,device=device).unsqueeze(0)
        with torch.inference_mode():
            eo=mr.model_forward_fresh(model,ctx,0,use_cache=True,output_attentions=False)
        exact=mr.cache_to_legacy(eo.past_key_values); del eo
        exact_eval=mr.eval_tail(model,tok,d,exact,device,exact_first_logits=None)
        exact_first=exact_eval.pop('_first_logits'); exact_nll=exact_eval['gold_nll']
        full_ms=mr.full_ttft_ms(model,d,device,reps=3)
        base=mr.eval_tail(model,tok,d,approx,device,exact_first_logits=exact_first); base.pop('_first_logits',None)
        qms=mr.query_ttft_ms(model,d,approx,device,reps=3)
        rows.append(dict(id=d['id'],method='reuse_0pct',recompute_fraction=0.0,repair_ms=0.0,query_ms=qms,warm_ttft_ms=qms,full_ttft_ms=full_ms,speedup=full_ms/qms,amortized_speedup_16=16*full_ms/(compile_ms+16*qms),kl=base['kl'],cosine=base['cosine'],top1_agree=base['top1_agree'],gold_nll_delta=base['gold_nll']-exact_nll,compile_ms=compile_ms))
        for b in blocks:
            repaired,rms,nrep=block_repair(model,d,approx,b,device)
            ev=mr.eval_tail(model,tok,d,repaired,device,exact_first_logits=exact_first); ev.pop('_first_logits',None)
            qms=mr.query_ttft_ms(model,d,repaired,device,reps=2)
            total=rms+qms
            rows.append(dict(id=d['id'],method=f'block_{b}',recompute_fraction=nrep/len(d['ctx_ids']),repair_ms=rms,query_ms=qms,warm_ttft_ms=total,full_ttft_ms=full_ms,speedup=full_ms/total,amortized_speedup_16=16*full_ms/(compile_ms+16*total),kl=ev['kl'],cosine=ev['cosine'],top1_agree=ev['top1_agree'],gold_nll_delta=ev['gold_nll']-exact_nll,compile_ms=compile_ms))
            del repaired
        del approx,exact,ctx,exact_first
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    result={'metadata':{'model':args.model,'device':device_name,'dataset':'hotpotqa/hotpot_qa:distractor:validation','examples':args.examples,'offset':args.offset,'docs':args.docs,'max_chunk_tokens':args.max_chunk_tokens,'implementation':'one batched forward per document-boundary block; Python orchestration included'},'aggregate':agg(rows),'rows':rows}
    p=Path(args.output); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(result,indent=2))
    print('MOSAIC_RAG_BLOCK_RESULT='+json.dumps({'metadata':result['metadata'],'aggregate':result['aggregate']},separators=(',',':')))
    print('Wrote',p)

if __name__=='__main__': main()
