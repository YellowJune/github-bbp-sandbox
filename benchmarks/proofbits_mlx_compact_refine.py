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

MODEL = 'mlx-community/gemma-3-270m-bf16'
PROMPTS = [
    'Explain why entropy is measured with logarithms.',
    'Solve step by step: if x + 1/x = 3, compute x^6 + 1/x^6.',
    'Write a Python function for longest increasing contiguous subarray.',
    'Explain why memory bandwidth matters for autoregressive inference.',
    'Describe natural selection without using teleological language.',
]
TOKENS = 32
CAP = 256

FILTER_SRC = r'''
    uint row = thread_position_in_grid.x;
    if (row >= upper_shape[0]) return;
    if (upper[row] >= bound[0]) {
        uint pos = atomic_fetch_add_explicit(&count[0], 1u, memory_order_relaxed);
        if (pos < CAPACITY) {
            atomic_store_explicit(&indices[pos], row, memory_order_relaxed);
        }
    }
'''

CAND_REFINE_SRC = r'''
    uint slot = threadgroup_position_in_grid.x;
    uint lane = thread_index_in_simdgroup;
    uint n = count[0];
    if (slot >= CAPACITY || slot >= n) {
        if (lane == 0) scores[slot] = -3.402823466e+38f;
        return;
    }
    uint row = indices[slot];
    uint D = (uint)hidden_shape[0];
    ulong b = (ulong)row * (ulong)D;
    float acc = 0.0f;
    for (uint j = lane; j < D; j += 32u) {
        ushort raw = ((ushort)high[b + j] << 8) | (ushort)low[b + j];
        half w = as_type<half>(raw);
        acc = fma(hidden[j], (float)w, acc);
    }
    float total = simd_sum(acc);
    if (lane == 0) scores[slot] = total;
'''


def med(xs): return float(statistics.median(xs))


def make_compact_kernels():
    filt = mx.fast.metal_kernel(
        name='pb_compact_survivors',
        input_names=['upper','bound'],
        output_names=['indices','count'],
        source=FILTER_SRC,
        atomic_outputs=True,
    )
    refine = mx.fast.metal_kernel(
        name='pb_compact_refine',
        input_names=['high','low','hidden','indices','count'],
        output_names=['scores'],
        source=CAND_REFINE_SRC,
    )
    return filt, refine


def compact_decision(base_ks, compact_ks, w16, high, low, h, V):
    dense_k, upper_k, pilot_k, _ = base_ks
    filt_k, cand_k = compact_ks
    U = upper_k(
        inputs=[high,h], grid=(V*32,1,1), threadgroup=(32,1,1),
        output_shapes=[(V,)], output_dtypes=[mx.float32], init_value=0.0
    )[0]
    p = mx.reshape(mx.argmax(U).astype(mx.uint32),(1,))
    B = pilot_k(
        inputs=[high,low,h,p], grid=(32,1,1), threadgroup=(32,1,1),
        output_shapes=[(1,)], output_dtypes=[mx.float32], init_value=0.0
    )[0]
    indices, count = filt_k(
        inputs=[U,B], template=[('CAPACITY',CAP)],
        grid=(V,1,1), threadgroup=(256,1,1),
        output_shapes=[(CAP,),(1,)], output_dtypes=[mx.uint32,mx.uint32],
        init_value=0,
    )
    # Exact fallback gate. This sync is part of the timed serving path.
    mx.eval(count)
    n = int(count.item())
    if n > CAP:
        return base.call_dense(dense_k,w16,h,V), n, True
    scores = cand_k(
        inputs=[high,low,h,indices,count], template=[('CAPACITY',CAP)],
        grid=(CAP*32,1,1), threadgroup=(32,1,1),
        output_shapes=[(CAP,)], output_dtypes=[mx.float32],
        init_value=-3.402823466e38,
    )[0]
    slot = mx.argmax(scores).astype(mx.uint32)
    winner = mx.take(indices, slot).astype(mx.uint32)
    return winner, n, False


def ids(tok, prompt):
    try: x=tok.encode(prompt)
    except Exception: x=tok(prompt)['input_ids']
    return mx.array(x,dtype=mx.int32)


def run(model,tok,prompt,mode,base_ks,compact_ks,w16,high,low,n_tokens):
    V,D=[int(x) for x in w16.shape]
    cache=cache_mod.make_prompt_cache(model)
    b=model.model(ids(tok,prompt)[None],cache=cache)
    h=b[:,-1,:].reshape((D,)).astype(mx.float32)
    mx.eval(h,[c.state for c in cache])
    toks=[]; times=[]; counts=[]; fallbacks=0
    for _ in range(n_tokens):
        t0=time.perf_counter()
        if mode=='dense':
            y=base.call_dense(base_ks[0],w16,h,V)
        elif mode=='proofbits_current':
            y=base.call_proofbits(base_ks[1],base_ks[2],base_ks[3],high,low,h,V,False)
        elif mode=='proofbits_compact':
            y,c,fb=compact_decision(base_ks,compact_ks,w16,high,low,h,V)
            counts.append(c); fallbacks += int(fb)
        elif mode=='native':
            y=base.call_native(w16,h)
        else: raise ValueError(mode)
        mx.eval(y)
        token=int(y.item()); toks.append(token)
        b=model.model(mx.array([[token]],dtype=mx.int32),cache=cache)
        h=b[:,-1,:].reshape((D,)).astype(mx.float32)
        mx.eval(h,[c.state for c in cache])
        times.append((time.perf_counter()-t0)*1e3)
    return {
        'tokens':toks,'median_ms':med(times),'mean_ms':float(statistics.mean(times)),
        'count_mean':float(statistics.mean(counts)) if counts else None,
        'count_max':max(counts) if counts else None,'fallbacks':fallbacks,
    }


def main():
    model,tok=load(MODEL)
    model.set_dtype(mx.float16); mx.eval(model.parameters())
    base.MODEL=MODEL
    base_ks=base.make_kernels(); compact_ks=make_compact_kernels()
    w16,high,low=base.prepare_weights(model)

    # Compile all paths.
    for m in ['dense','proofbits_current','proofbits_compact','native']:
        run(model,tok,PROMPTS[0],m,base_ks,compact_ks,w16,high,low,4)
    mx.synchronize()

    rows=[]
    orders=[
        ['dense','proofbits_current','proofbits_compact','native'],
        ['proofbits_compact','native','dense','proofbits_current'],
        ['native','proofbits_current','proofbits_compact','dense'],
        ['proofbits_current','dense','native','proofbits_compact'],
    ]
    for i,prompt in enumerate(PROMPTS):
        res={}; order=orders[i%len(orders)]
        for m in order:
            gc.collect(); mx.clear_cache(); res[m]=run(model,tok,prompt,m,base_ks,compact_ks,w16,high,low,TOKENS)
        d,c,k,n=res['dense'],res['proofbits_current'],res['proofbits_compact'],res['native']
        rows.append({
            'prompt_index':i,'order':order,
            'current_exact':d['tokens']==c['tokens'],
            'compact_exact':d['tokens']==k['tokens'],
            'native_equal_compact':n['tokens']==k['tokens'],
            'dense_ms':d['median_ms'],'current_ms':c['median_ms'],'compact_ms':k['median_ms'],'native_ms':n['median_ms'],
            'compact_vs_current':c['median_ms']/k['median_ms'],
            'dense_over_compact':d['median_ms']/k['median_ms'],
            'native_over_compact':n['median_ms']/k['median_ms'],
            'compact_count_mean':k['count_mean'],'compact_count_max':k['count_max'],'fallbacks':k['fallbacks'],
        })
    out={
        'kind':'proofbits_fp16_compact_refine_prototype','model':MODEL,'cap':CAP,
        'n_prompts':len(PROMPTS),'tokens_per_prompt':TOKENS,
        'all_current_exact':all(r['current_exact'] for r in rows),
        'all_compact_exact':all(r['compact_exact'] for r in rows),
        'total_fallbacks':sum(r['fallbacks'] for r in rows),
        'median_compact_vs_current':med([r['compact_vs_current'] for r in rows]),
        'median_dense_over_compact':med([r['dense_over_compact'] for r in rows]),
        'median_native_over_compact':med([r['native_over_compact'] for r in rows]),
        'min_native_over_compact':min(r['native_over_compact'] for r in rows),
        'rows':rows,
        'note':'Compact path is exact by construction when survivor count <= CAP; if count exceeds CAP it falls back to matched dense. The count synchronization and fallback gate are included in timed decode.'
    }
    Path('experiments/artifacts').mkdir(parents=True,exist_ok=True)
    Path('experiments/artifacts/proofbits_mlx_compact_refine.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))

if __name__=='__main__': main()
