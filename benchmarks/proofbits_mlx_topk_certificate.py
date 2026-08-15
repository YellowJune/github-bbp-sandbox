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

MODEL='mlx-community/gemma-3-270m-bf16'
PROMPTS=[
    'Explain why entropy is measured with logarithms.',
    'Solve step by step: if x + 1/x = 3, compute x^6 + 1/x^6.',
    'Write a Python function for longest increasing contiguous subarray.',
    'Explain why memory bandwidth matters for autoregressive inference.',
    'Describe natural selection without using teleological language.',
]
TOKENS=32
KS=[64,256,1024]

CANDIDATE_SRC=r'''
    uint slot = threadgroup_position_in_grid.x;
    uint lane = thread_index_in_simdgroup;
    uint K = (uint)indices_shape[0];
    if (slot >= K) return;
    uint row = indices[slot];
    uint D = (uint)hidden_shape[0];
    ulong b = (ulong)row * (ulong)D;
    float acc=0.0f;
    for (uint j=lane; j<D; j+=32u) {
        ushort raw=((ushort)high[b+j] << 8) | (ushort)low[b+j];
        half w=as_type<half>(raw);
        acc=fma(hidden[j],(float)w,acc);
    }
    float total=simd_sum(acc);
    if (lane==0) scores[slot]=total;
'''


def med(x): return float(statistics.median(x))


def make_candidate_kernel():
    return mx.fast.metal_kernel(
        name='pb_topk_exact_candidates',
        input_names=['high','low','hidden','indices'],
        output_names=['scores'],source=CANDIDATE_SRC)


def topk_decision(base_ks,cand_k,w16,high,low,h,V,K):
    dense_k,upper_k,_,_=base_ks
    U=upper_k(inputs=[high,h],grid=(V*32,1,1),threadgroup=(32,1,1),
              output_shapes=[(V,)],output_dtypes=[mx.float32],init_value=0.0)[0]
    # argpartition guarantees all indices after kth refer to elements >= kth.
    part=mx.argpartition(U,V-K)
    idx=part[V-K:]
    scores=cand_k(inputs=[high,low,h,idx],grid=(K*32,1,1),threadgroup=(32,1,1),
                  output_shapes=[(K,)],output_dtypes=[mx.float32],init_value=-3.402823466e38)[0]
    slot=mx.argmax(scores).astype(mx.uint32)
    winner=mx.take(idx,slot).astype(mx.uint32)
    B=mx.max(scores)
    T=mx.min(mx.take(U,idx))
    # Strict inequality avoids any ambiguity under exact ties at the frontier.
    cert=B>T
    # CPU sync is deliberately included in timed serving. If the certificate
    # fails, use the matched dense path so the result remains exact.
    mx.eval(cert,winner)
    ok=bool(cert.item())
    if not ok:
        return base.call_dense(dense_k,w16,h,V), False
    return winner, True


def ids(tok,prompt):
    try:x=tok.encode(prompt)
    except Exception:x=tok(prompt)['input_ids']
    return mx.array(x,dtype=mx.int32)


def run(model,tok,prompt,mode,base_ks,cand_k,w16,high,low,n):
    V,D=[int(x) for x in w16.shape]
    kv=cache_mod.make_prompt_cache(model)
    b=model.model(ids(tok,prompt)[None],cache=kv)
    h=b[:,-1,:].reshape((D,)).astype(mx.float32)
    mx.eval(h,[c.state for c in kv])
    toks=[]; times=[]; certs=[]
    for _ in range(n):
        t0=time.perf_counter()
        if mode=='dense': y=base.call_dense(base_ks[0],w16,h,V)
        elif mode=='current': y=base.call_proofbits(base_ks[1],base_ks[2],base_ks[3],high,low,h,V,False)
        elif mode=='native': y=base.call_native(w16,h)
        elif mode.startswith('topk'):
            K=int(mode[4:]);y,ok=topk_decision(base_ks,cand_k,w16,high,low,h,V,K);certs.append(ok)
        else: raise ValueError(mode)
        mx.eval(y); token=int(y.item()); toks.append(token)
        b=model.model(mx.array([[token]],dtype=mx.int32),cache=kv)
        h=b[:,-1,:].reshape((D,)).astype(mx.float32)
        mx.eval(h,[c.state for c in kv]); times.append((time.perf_counter()-t0)*1e3)
    return {'tokens':toks,'median_ms':med(times),'mean_ms':float(statistics.mean(times)),
            'certificate_rate':float(statistics.mean(certs)) if certs else None,
            'fallbacks':certs.count(False) if certs else 0}


def main():
    model,tok=load(MODEL); model.set_dtype(mx.float16); mx.eval(model.parameters())
    base.MODEL=MODEL
    base_ks=base.make_kernels();cand_k=make_candidate_kernel();w16,high,low=base.prepare_weights(model)
    modes=['dense','current','native']+[f'topk{k}' for k in KS]
    for m in modes: run(model,tok,PROMPTS[0],m,base_ks,cand_k,w16,high,low,4)
    mx.synchronize()
    rows=[]
    for pi,prompt in enumerate(PROMPTS):
        # rotate order so no candidate always occupies the same thermal position
        order=modes[pi%len(modes):]+modes[:pi%len(modes)]
        res={}
        for m in order:
            gc.collect();mx.clear_cache();res[m]=run(model,tok,prompt,m,base_ks,cand_k,w16,high,low,TOKENS)
        d=res['dense'];c=res['current'];n=res['native']
        row={'prompt_index':pi,'order':order,'current_exact':c['tokens']==d['tokens']}
        for K in KS:
            r=res[f'topk{K}']
            row[f'k{K}_exact']=r['tokens']==d['tokens']
            row[f'k{K}_cert_rate']=r['certificate_rate']
            row[f'k{K}_fallbacks']=r['fallbacks']
            row[f'k{K}_ms']=r['median_ms']
            row[f'k{K}_vs_current']=c['median_ms']/r['median_ms']
            row[f'native_over_k{K}']=n['median_ms']/r['median_ms']
        row.update({'dense_ms':d['median_ms'],'current_ms':c['median_ms'],'native_ms':n['median_ms']})
        rows.append(row)
    out={'kind':'proofbits_fp16_topk_exact_certificate','model':MODEL,'Ks':KS,
         'n_prompts':len(PROMPTS),'tokens_per_prompt':TOKENS,
         'all_current_exact':all(r['current_exact'] for r in rows),'rows':rows}
    for K in KS:
        out[f'k{K}_all_exact']=all(r[f'k{K}_exact'] for r in rows)
        out[f'k{K}_total_fallbacks']=sum(r[f'k{K}_fallbacks'] for r in rows)
        out[f'k{K}_median_cert_rate']=med([r[f'k{K}_cert_rate'] for r in rows])
        out[f'k{K}_median_vs_current']=med([r[f'k{K}_vs_current'] for r in rows])
        out[f'k{K}_median_native_over']=med([r[f'native_over_k{K}'] for r in rows])
        out[f'k{K}_min_native_over']=min(r[f'native_over_k{K}'] for r in rows)
    out['note']='Top-K certificate: exact-score only the K rows with largest certified upper bounds. If best exact B is strictly greater than the minimum selected upper bound T, all omitted rows have exact score <= upper <= T < B. Otherwise matched-dense fallback preserves exactness. Certificate sync/fallback is included in timing.'
    Path('experiments/artifacts').mkdir(parents=True,exist_ok=True)
    Path('experiments/artifacts/proofbits_mlx_topk_certificate.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

if __name__=='__main__':main()
