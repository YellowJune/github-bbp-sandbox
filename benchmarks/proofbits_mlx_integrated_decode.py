import gc
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models import cache as cache_mod

MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-bf16"
PROMPT = "Explain in one paragraph why exact inference decisions can sometimes be certified from partial numerical representations."
MAX_NEW = 48
WARMUP_NEW = 6
ROUNDS = 4

# One SIMDgroup (32 lanes) evaluates one vocabulary row. Qwen D=896=28*32.
# MLX injects <input>_shape metadata when referenced in custom Metal source.
DENSE_SRC = r'''
    uint row = threadgroup_position_in_grid.x;
    uint lane = thread_index_in_simdgroup;
    uint D = (uint)hidden_shape[0];
    ulong base = (ulong)row * (ulong)D;
    float acc = 0.0f;
    for (uint j = lane; j < D; j += 32) {
        acc = fma((float)hidden[j], (float)weight[base + j], acc);
    }
    float total = simd_sum(acc);
    if (lane == 0) out[row] = total;
'''

UPPER_SRC = r'''
    uint row = threadgroup_position_in_grid.x;
    uint lane = thread_index_in_simdgroup;
    uint D = (uint)hidden_shape[0];
    ulong base = (ulong)row * (ulong)D;
    float acc = 0.0f;
    for (uint j = lane; j < D; j += 32) {
        uchar hb = high[base + j];
        ushort weightSign = (ushort)(hb & (uchar)0x80);
        ushort hiddenSign = (hidden[j] < 0.0f) ? (ushort)0x80 : (ushort)0x00;
        ushort suffix = (weightSign == hiddenSign) ? (ushort)0x00FF : (ushort)0x0000;
        ushort raw = ((ushort)hb << 8) | suffix;
        half endpoint = as_type<half>(raw);
        acc = fma(hidden[j], (float)endpoint, acc);
    }
    float total = simd_sum(acc);
    if (lane == 0) upper[row] = total;
'''

PILOT_SRC = r'''
    uint lane = thread_index_in_simdgroup;
    uint D = (uint)hidden_shape[0];
    uint row = (uint)pilot[0];
    ulong base = (ulong)row * (ulong)D;
    float acc = 0.0f;
    for (uint j = lane; j < D; j += 32) {
        ushort raw = ((ushort)high[base + j] << 8) | (ushort)low[base + j];
        half w = as_type<half>(raw);
        acc = fma(hidden[j], (float)w, acc);
    }
    float total = simd_sum(acc);
    if (lane == 0) bound[0] = total;
'''

REFINE_SRC = r'''
    uint row = threadgroup_position_in_grid.x;
    uint lane = thread_index_in_simdgroup;
    uint D = (uint)hidden_shape[0];
    float B = bound[0];
    if (upper[row] < B) {
        if (lane == 0) exact[row] = -3.402823466e+38f;
        return;
    }
    ulong base = (ulong)row * (ulong)D;
    float acc = 0.0f;
    for (uint j = lane; j < D; j += 32) {
        ushort raw = ((ushort)high[base + j] << 8) | (ushort)low[base + j];
        half w = as_type<half>(raw);
        acc = fma(hidden[j], (float)w, acc);
    }
    float total = simd_sum(acc);
    if (lane == 0) exact[row] = total;
'''


def med(xs):
    return float(statistics.median(xs))


def pct(xs, p):
    if not xs:
        return None
    return float(np.percentile(np.asarray(xs, dtype=np.float64), p))


def make_kernels():
    dense = mx.fast.metal_kernel(
        name="pb_mlx_dense_fp16_row",
        input_names=["weight", "hidden"], output_names=["out"], source=DENSE_SRC
    )
    upper = mx.fast.metal_kernel(
        name="pb_mlx_upper_row",
        input_names=["high", "hidden"], output_names=["upper"], source=UPPER_SRC
    )
    pilot = mx.fast.metal_kernel(
        name="pb_mlx_exact_pilot",
        input_names=["high", "low", "hidden", "pilot"], output_names=["bound"], source=PILOT_SRC
    )
    refine = mx.fast.metal_kernel(
        name="pb_mlx_conditional_refine",
        input_names=["high", "low", "hidden", "upper", "bound"], output_names=["exact"], source=REFINE_SRC
    )
    return dense, upper, pilot, refine


def call_dense(k, w16, h32, V):
    scores = k(
        inputs=[w16, h32],
        grid=(V * 32, 1, 1), threadgroup=(32, 1, 1),
        output_shapes=[(V,)], output_dtypes=[mx.float32], init_value=0.0,
    )[0]
    return mx.argmax(scores).astype(mx.uint32)


def call_native(w16, h32):
    # Native MLX matvec baseline using the same FP16 storage representation.
    logits = mx.matmul(h32.astype(mx.float16), mx.transpose(w16))
    return mx.argmax(logits).astype(mx.uint32)


def call_proofbits(k_upper, k_pilot, k_refine, high, low, h32, V, diagnostics=False):
    U = k_upper(
        inputs=[high, h32],
        grid=(V * 32, 1, 1), threadgroup=(32, 1, 1),
        output_shapes=[(V,)], output_dtypes=[mx.float32], init_value=0.0,
    )[0]
    p = mx.argmax(U).astype(mx.uint32)
    B = k_pilot(
        inputs=[high, low, h32, p],
        grid=(32, 1, 1), threadgroup=(32, 1, 1),
        output_shapes=[(1,)], output_dtypes=[mx.float32], init_value=0.0,
    )[0]
    exact = k_refine(
        inputs=[high, low, h32, U, B],
        grid=(V * 32, 1, 1), threadgroup=(32, 1, 1),
        output_shapes=[(V,)], output_dtypes=[mx.float32], init_value=-3.402823466e38,
    )[0]
    winner = mx.argmax(exact).astype(mx.uint32)
    if diagnostics:
        survivors = mx.sum(U >= B)
        return winner, survivors
    return winner


def prepare_weights(model):
    # Qwen ties output projection to token embeddings. Cast once to the matched
    # FP16 storage used by both dense and ProofBits paths.
    w_src = model.model.embed_tokens.weight
    w16 = w_src.astype(mx.float16)
    mx.eval(w16)
    # Setup is outside timed generation. Extract exact IEEE-754 binary16 bytes.
    w_np = np.array(w16, copy=True).astype(np.float16, copy=False)
    bits = w_np.view(np.uint16)
    high_np = (bits >> 8).astype(np.uint8, copy=True)
    low_np = (bits & 0xFF).astype(np.uint8, copy=True)
    high = mx.array(high_np)
    low = mx.array(low_np)
    mx.eval(high, low)
    return w16, high, low


def tokenize(tok):
    try:
        ids = tok.encode(PROMPT)
    except Exception:
        ids = tok(PROMPT)["input_ids"]
    return mx.array(ids, dtype=mx.int32)


def decode_once(model, tok, mode, kernels, w16, high, low, max_new, collect_diag=False):
    V, D = [int(x) for x in w16.shape]
    prompt = tokenize(tok)
    kv = cache_mod.make_prompt_cache(model)
    dense_k, upper_k, pilot_k, refine_k = kernels

    # Prefill body only. The output projection is deliberately excluded.
    t_prefill = time.perf_counter()
    body = model.model(prompt[None], cache=kv)
    h = body[:, -1, :].reshape((D,)).astype(mx.float32)
    mx.eval(h, [c.state for c in kv])
    prefill_ms = (time.perf_counter() - t_prefill) * 1e3

    tokens = []
    step_ms = []
    survivor_counts = []
    for _ in range(max_new):
        t0 = time.perf_counter()
        if mode == "dense_custom_fp16":
            y = call_dense(dense_k, w16, h, V)
        elif mode == "native_mlx_fp16":
            y = call_native(w16, h)
        elif mode == "proofbits_fp16":
            if collect_diag:
                y, s = call_proofbits(upper_k, pilot_k, refine_k, high, low, h, V, diagnostics=True)
            else:
                y = call_proofbits(upper_k, pilot_k, refine_k, high, low, h, V, diagnostics=False)
        else:
            raise ValueError(mode)

        # Interactive decode semantics: materialize the next-token decision.
        mx.eval(y)
        if collect_diag and mode == "proofbits_fp16":
            mx.eval(s)
            survivor_counts.append(int(s.item()))
        token = int(y.item())
        tokens.append(token)

        # Feed the chosen token through exactly the same MLX KV-cache body.
        inp = mx.array([[token]], dtype=mx.int32)
        body = model.model(inp, cache=kv)
        h = body[:, -1, :].reshape((D,)).astype(mx.float32)
        mx.eval(h, [c.state for c in kv])
        step_ms.append((time.perf_counter() - t0) * 1e3)

    out = {
        "mode": mode,
        "tokens": tokens,
        "prefill_ms": float(prefill_ms),
        "step_ms": step_ms,
        "median_ms_per_token": med(step_ms),
        "mean_ms_per_token": float(statistics.mean(step_ms)),
        "p10_ms": pct(step_ms, 10),
        "p90_ms": pct(step_ms, 90),
        "tokens_per_s_from_median": 1000.0 / med(step_ms),
        "tokens_per_s_from_mean": 1000.0 / float(statistics.mean(step_ms)),
    }
    if survivor_counts:
        out["survivors"] = survivor_counts
        out["survivor_mean"] = float(statistics.mean(survivor_counts))
        out["survivor_median"] = med(survivor_counts)
        out["survivor_fraction_mean"] = float(statistics.mean(survivor_counts) / V)
    return out


def warmup(model, tok, kernels, w16, high, low):
    for mode in ["dense_custom_fp16", "proofbits_fp16", "native_mlx_fp16"]:
        _ = decode_once(model, tok, mode, kernels, w16, high, low, WARMUP_NEW, False)
    mx.synchronize()


def main():
    model, tok = load(MODEL)
    mx.eval(model.parameters())
    kernels = make_kernels()
    w16, high, low = prepare_weights(model)
    V, D = [int(x) for x in w16.shape]
    warmup(model, tok, kernels, w16, high, low)

    # AB/BA counterbalancing for the exact matched-storage comparison.
    rounds = []
    dense_all = []
    pb_all = []
    for r in range(ROUNDS):
        order = ["dense_custom_fp16", "proofbits_fp16"] if r % 2 == 0 else ["proofbits_fp16", "dense_custom_fp16"]
        res = {}
        for mode in order:
            gc.collect()
            mx.clear_cache()
            res[mode] = decode_once(model, tok, mode, kernels, w16, high, low, MAX_NEW, False)
        d = res["dense_custom_fp16"]
        p = res["proofbits_fp16"]
        exact_seq = d["tokens"] == p["tokens"]
        speed = d["median_ms_per_token"] / p["median_ms_per_token"]
        rounds.append({
            "round": r + 1,
            "order": order,
            "dense_median_ms": d["median_ms_per_token"],
            "proofbits_median_ms": p["median_ms_per_token"],
            "decode_speedup": speed,
            "dense_mean_ms": d["mean_ms_per_token"],
            "proofbits_mean_ms": p["mean_ms_per_token"],
            "sequence_exact": exact_seq,
            "n_tokens": MAX_NEW,
        })
        dense_all.append(d["median_ms_per_token"])
        pb_all.append(p["median_ms_per_token"])

    # One diagnostic ProofBits trajectory and one native-MLX FP16 trajectory.
    diag = decode_once(model, tok, "proofbits_fp16", kernels, w16, high, low, MAX_NEW, True)
    native = decode_once(model, tok, "native_mlx_fp16", kernels, w16, high, low, MAX_NEW, False)
    dense_ref = decode_once(model, tok, "dense_custom_fp16", kernels, w16, high, low, MAX_NEW, False)

    speedups = [x["decode_speedup"] for x in rounds]
    out = {
        "kind": "proofbits_integrated_mlx_greedy_decode",
        "model": MODEL,
        "device_info": mx.device_info(),
        "V": V,
        "D": D,
        "prompt": PROMPT,
        "new_tokens": MAX_NEW,
        "rounds": rounds,
        "all_counterbalanced_sequences_exact": all(x["sequence_exact"] for x in rounds),
        "median_integrated_speedup_dense_custom_over_proofbits": med(speedups),
        "mean_integrated_speedup_dense_custom_over_proofbits": float(statistics.mean(speedups)),
        "median_dense_ms_over_rounds": med(dense_all),
        "median_proofbits_ms_over_rounds": med(pb_all),
        "diagnostic_proofbits": {k: v for k, v in diag.items() if k != "step_ms"},
        "native_mlx_fp16": {k: v for k, v in native.items() if k != "step_ms"},
        "native_vs_dense_custom_sequence_equal": native["tokens"] == dense_ref["tokens"],
        "native_vs_proofbits_sequence_equal": native["tokens"] == diag["tokens"],
        "native_median_ms": native["median_ms_per_token"],
        "proofbits_diag_median_ms": diag["median_ms_per_token"],
        "native_over_proofbits_speedup": native["median_ms_per_token"] / diag["median_ms_per_token"],
        "notes": [
            "Dense custom and ProofBits share exactly the same FP16 weight representation and MLX KV-cache body.",
            "Setup/conversion of the tied BF16 embedding to FP16 high/low planes is outside timed generation.",
            "Each decode step materializes the next token before feeding it back, matching interactive greedy serving semantics.",
            "Native MLX FP16 matvec is reported as an additional optimized baseline; exactness headline uses matched custom dense FP16 accumulation order.",
        ],
    }
    Path("experiments/artifacts").mkdir(parents=True, exist_ok=True)
    p = Path("experiments/artifacts/proofbits_mlx_integrated_decode.json")
    p.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
