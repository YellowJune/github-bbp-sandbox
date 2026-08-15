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

MODEL = 'mlx-community/gemma-3-270m-bf16'
D = 640
TOKENS = 48
PROMPTS = [
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


def med(xs):
    return float(statistics.median(xs))


def term(off):
    return f'''{{
        uint j = basej + {off}u;
        uchar hb = high[base + j];
        ushort ws = (ushort)(hb & (uchar)0x80);
        ushort hs = (hidden[j] < 0.0f) ? (ushort)0x80 : (ushort)0x00;
        ushort suffix = (ws == hs) ? (ushort)0x00FF : (ushort)0x0000;
        ushort raw = ((ushort)hb << 8) | suffix;
        acc = fma(hidden[j], (float)as_type<half>(raw), acc);
    }}'''


U10_SRC = f'''
    uint row = threadgroup_position_in_grid.x;
    uint lane = thread_index_in_simdgroup;
    ulong base = (ulong)row * {D}ul;
    float acc = 0.0f;
    for (uint basej = lane; basej < {D}u; basej += 320u) {{
        {term(0)}
        {term(32)}
        {term(64)}
        {term(96)}
        {term(128)}
        {term(160)}
        {term(192)}
        {term(224)}
        {term(256)}
        {term(288)}
    }}
    float total = simd_sum(acc);
    if (lane == 0) upper[row] = total;
'''


def make_u10():
    return mx.fast.metal_kernel(
        name='pb_upper_u10_integrated_stress',
        input_names=['high', 'hidden'],
        output_names=['upper'],
        source=U10_SRC,
    )


def ids(tok, prompt):
    try:
        x = tok.encode(prompt)
    except Exception:
        x = tok(prompt)['input_ids']
    return mx.array(x, dtype=mx.int32)


def u10_decision(u10, ks, high, low, h, V):
    U = u10(
        inputs=[high, h],
        grid=(V * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(V,)],
        output_dtypes=[mx.float32],
        init_value=0.0,
    )[0]
    p = mx.reshape(mx.argmax(U).astype(mx.uint32), (1,))
    B = ks[2](
        inputs=[high, low, h, p],
        grid=(32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(1,)],
        output_dtypes=[mx.float32],
        init_value=0.0,
    )[0]
    E = ks[3](
        inputs=[high, low, h, U, B],
        grid=(V * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(V,)],
        output_dtypes=[mx.float32],
        init_value=-3.402823466e38,
    )[0]
    return mx.argmax(E).astype(mx.uint32)


def run(model, tok, prompt, mode, ks, u10, w16, high, low, n):
    V, DD = [int(x) for x in w16.shape]
    assert DD == D
    cache = cache_mod.make_prompt_cache(model)
    body = model.model(ids(tok, prompt)[None], cache=cache)
    h = body[:, -1, :].reshape((D,)).astype(mx.float32)
    mx.eval(h, [c.state for c in cache])

    toks, total, heads, bodies = [], [], [], []
    for _ in range(n):
        t0 = time.perf_counter()
        th = time.perf_counter()
        if mode == 'dense':
            y = base.call_dense(ks[0], w16, h, V)
        elif mode == 'current':
            y = base.call_proofbits(ks[1], ks[2], ks[3], high, low, h, V, False)
        elif mode == 'u10':
            y = u10_decision(u10, ks, high, low, h, V)
        elif mode == 'native':
            y = base.call_native(w16, h)
        else:
            raise ValueError(mode)
        mx.eval(y)
        token = int(y.item())
        toks.append(token)
        heads.append((time.perf_counter() - th) * 1e3)

        tb = time.perf_counter()
        body = model.model(mx.array([[token]], dtype=mx.int32), cache=cache)
        h = body[:, -1, :].reshape((D,)).astype(mx.float32)
        mx.eval(h, [c.state for c in cache])
        bodies.append((time.perf_counter() - tb) * 1e3)
        total.append((time.perf_counter() - t0) * 1e3)

    return {
        'tokens': toks,
        'median_total_ms': med(total),
        'mean_total_ms': float(statistics.mean(total)),
        'sum_total_ms': float(sum(total)),
        'median_head_ms': med(heads),
        'mean_head_ms': float(statistics.mean(heads)),
        'sum_head_ms': float(sum(heads)),
        'median_body_ms': med(bodies),
        'sum_body_ms': float(sum(bodies)),
    }


def main():
    model, tok = load(MODEL)
    model.set_dtype(mx.float16)
    mx.eval(model.parameters())
    base.MODEL = MODEL

    w16, high, low = dual.prepare(model)
    V, DD = [int(x) for x in w16.shape]
    assert DD == D
    ks = base.make_kernels()
    u10 = make_u10()

    modes = ['dense', 'current', 'u10', 'native']
    # Compile/JIT all paths before timing.
    for m in modes:
        run(model, tok, PROMPTS[0], m, ks, u10, w16, high, low, 4)
    mx.synchronize()

    orders = [
        ['dense', 'current', 'u10', 'native'],
        ['u10', 'native', 'dense', 'current'],
        ['native', 'current', 'u10', 'dense'],
        ['current', 'dense', 'native', 'u10'],
    ]
    rows = []
    sums_total = {m: 0.0 for m in modes}
    sums_head = {m: 0.0 for m in modes}

    for i, prompt in enumerate(PROMPTS):
        order = orders[i % len(orders)]
        res = {}
        for m in order:
            gc.collect()
            mx.clear_cache()
            res[m] = run(model, tok, prompt, m, ks, u10, w16, high, low, TOKENS)
            sums_total[m] += res[m]['sum_total_ms']
            sums_head[m] += res[m]['sum_head_ms']

        d, c, u, n = res['dense'], res['current'], res['u10'], res['native']
        rows.append({
            'prompt_index': i,
            'order': order,
            'current_exact': c['tokens'] == d['tokens'],
            'u10_exact': u['tokens'] == d['tokens'],
            'native_equal_u10': n['tokens'] == u['tokens'],
            'dense_total_ms': d['median_total_ms'],
            'current_total_ms': c['median_total_ms'],
            'u10_total_ms': u['median_total_ms'],
            'native_total_ms': n['median_total_ms'],
            'current_over_u10': c['median_total_ms'] / u['median_total_ms'],
            'dense_over_u10': d['median_total_ms'] / u['median_total_ms'],
            'native_over_u10': n['median_total_ms'] / u['median_total_ms'],
            'current_head_ms': c['median_head_ms'],
            'u10_head_ms': u['median_head_ms'],
            'current_head_over_u10': c['median_head_ms'] / u['median_head_ms'],
            'native_head_ms': n['median_head_ms'],
            'native_head_over_u10': n['median_head_ms'] / u['median_head_ms'],
            'u10_body_ms': u['median_body_ms'],
        })

    out = {
        'kind': 'proofbits_u10_integrated_stress',
        'model': MODEL,
        'runtime_dtype': 'float16',
        'n_prompts': len(PROMPTS),
        'tokens_per_prompt': TOKENS,
        'matched_tokens': len(PROMPTS) * TOKENS,
        'rows': rows,
        'all_current_exact': all(r['current_exact'] for r in rows),
        'all_u10_exact': all(r['u10_exact'] for r in rows),
        'native_all_equal_u10': all(r['native_equal_u10'] for r in rows),
        'current_over_u10_prompt_median': med([r['current_over_u10'] for r in rows]),
        'current_over_u10_prompt_mean': float(statistics.mean(r['current_over_u10'] for r in rows)),
        'current_head_over_u10_prompt_median': med([r['current_head_over_u10'] for r in rows]),
        'native_over_u10_prompt_median': med([r['native_over_u10'] for r in rows]),
        'native_over_u10_prompt_mean': float(statistics.mean(r['native_over_u10'] for r in rows)),
        'native_over_u10_prompt_min': min(r['native_over_u10'] for r in rows),
        'dense_over_u10_prompt_median': med([r['dense_over_u10'] for r in rows]),
        'pooled_total_ms': sums_total,
        'pooled_head_ms': sums_head,
        'pooled_current_over_u10': sums_total['current'] / sums_total['u10'],
        'pooled_dense_over_u10': sums_total['dense'] / sums_total['u10'],
        'pooled_native_over_u10': sums_total['native'] / sums_total['u10'],
        'pooled_current_head_over_u10': sums_head['current'] / sums_head['u10'],
        'pooled_native_head_over_u10': sums_head['native'] / sums_head['u10'],
        'note': 'Full-model FP16 autoregressive decode. u10 changes only the high-byte upper kernel: static D=640, ten unrolled per-lane FMA terms per outer iteration, preserving the same lane assignment and per-lane accumulation order as current ProofBits. JIT/setup excluded; body/KV included in total timings.'
    }

    Path('experiments/artifacts').mkdir(parents=True, exist_ok=True)
    Path('experiments/artifacts/proofbits_mlx_u10_stress.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
