import gc
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models import cache as cache_mod

import proofbits_mlx_bf16_p10_integrated as base
from proofbits_mlx_bf16_p10_coop import make_cooperative_kernels

MODEL='mlx-community/gemma-3-270m-bf16'
PROMPT='Explain in one paragraph why exact inference decisions can sometimes be certified from partial numerical representations.'
TOKENS=48
ROUNDS=4


def med(x): return float(statistics.median(x))


def ids(tok):
    try: a=tok.encode(PROMPT)
    except Exception: a=tok(PROMPT)['input_ids']
    return mx.array(a,dtype=mx.int32)


def run(model,tok,mode,ks,prefix,suffix,n):
    weight=model.lm_head.weight
    V,D=[int(x) for x in weight.shape]
    cache=cache_mod.make_prompt_cache(model)
    body=model.model(ids(tok)[None],cache=cache)
    h=body[:,-1,:].reshape((D,)).astype(mx.float32)
    mx.eval(h,[c.state for c in cache])
    ts=[]; heads=[]; toks=[]
    for _ in range(n):
        t0=time.perf_counter();th=time.perf_counter()
        if mode=='native_mlx_bf16':
            logits=model.lm_head(h.astype(mx.bfloat16))
            y=mx.argmax(logits).astype(mx.uint32)
        elif mode=='proofbits_bf16_p10':
            y=base.proofbits_decision(ks,prefix,suffix,h,V,False)
        else: raise ValueError(mode)
        mx.eval(y); token=int(y.item());toks.append(token);heads.append((time.perf_counter()-th)*1e3)
        body=model.model(mx.array([[token]],dtype=mx.int32),cache=cache)
        h=body[:,-1,:].reshape((D,)).astype(mx.float32)
        mx.eval(h,[c.state for c in cache]);ts.append((time.perf_counter()-t0)*1e3)
    return {'tokens':toks,'median_total_ms':med(ts),'mean_total_ms':float(statistics.mean(ts)),'median_head_ms':med(heads)}


def main():
    model,tok=load(MODEL);mx.eval(model.parameters())
    weight=model.lm_head.weight;V,D=[int(x) for x in weight.shape]
    prefix,suffix=base.pack_bf16_weight(weight);mx.eval(prefix,suffix)
    ks=make_cooperative_kernels(D)
    # compile both serving paths before measurement
    for m in ['native_mlx_bf16','proofbits_bf16_p10']: run(model,tok,m,ks,prefix,suffix,4)
    mx.synchronize()
    rows=[]
    for r in range(ROUNDS):
        order=['native_mlx_bf16','proofbits_bf16_p10'] if r%2==0 else ['proofbits_bf16_p10','native_mlx_bf16']
        res={}
        for m in order:
            gc.collect();mx.clear_cache();res[m]=run(model,tok,m,ks,prefix,suffix,TOKENS)
        n=res['native_mlx_bf16'];p=res['proofbits_bf16_p10']
        rows.append({'round':r+1,'order':order,'sequence_equal':n['tokens']==p['tokens'],'native_total_ms':n['median_total_ms'],'proofbits_total_ms':p['median_total_ms'],'total_speedup':n['median_total_ms']/p['median_total_ms'],'native_head_ms':n['median_head_ms'],'proofbits_head_ms':p['median_head_ms'],'head_speedup':n['median_head_ms']/p['median_head_ms']})
    out={'kind':'proofbits_native_bf16_p10_vs_native_mlx_bf16','model':MODEL,'tokens_per_round':TOKENS,'rounds':rows,'all_sequences_equal':all(x['sequence_equal'] for x in rows),'median_total_speedup':med([x['total_speedup'] for x in rows]),'mean_total_speedup':float(statistics.mean(x['total_speedup'] for x in rows)),'median_head_speedup':med([x['head_speedup'] for x in rows]),'note':'Native MLX BF16 lm_head+argmax vs densely packed 10+6 ProofBits, same BF16 checkpoint head and same MLX KV-cache body. Sequence equality is empirical, not assumed, because reduction semantics differ.'}
    Path('experiments/artifacts').mkdir(parents=True,exist_ok=True);Path('experiments/artifacts/proofbits_mlx_bf16_p10_vs_native.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

if __name__=='__main__':main()
