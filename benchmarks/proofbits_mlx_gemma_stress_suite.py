import gc
import json
import statistics
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load

sys.path.insert(0, str(Path(__file__).resolve().parent))
import proofbits_mlx_integrated_decode as bench

MODEL = "mlx-community/gemma-3-270m-bf16"
TOKENS = 64
PROMPTS = [
    "Explain the intuition behind entropy and why logarithms appear in information theory.",
    "Derive the closed form of the geometric series and state the convergence condition.",
    "Solve step by step: if x + 1/x = 3, compute x^6 + 1/x^6.",
    "Write a Python function for longest increasing contiguous subarray and discuss time and space complexity.",
    "Explain the difference between optimistic and pessimistic concurrency control in databases.",
    "Summarize why memory bandwidth can dominate autoregressive neural-network inference at batch size one.",
    "Compare TCP congestion control with application-level rate limiting in a concise technical explanation.",
    "Describe how natural selection can produce complex adaptation without foresight.",
    "Give a rigorous but intuitive explanation of why the central limit theorem is useful.",
    "Write a short science-fiction scene in which a machine proves which memory bytes need not be read.",
    "Explain the ethical tradeoffs of deploying highly capable AI systems in education without using slogans.",
    "Design a small experiment that distinguishes correlation from causation and explain possible confounders.",
]


def prepare_weights(model):
    src = model.lm_head.weight if hasattr(model, "lm_head") else model.model.embed_tokens.weight
    w16 = src.astype(mx.float16)
    mx.eval(w16)
    w_np = np.array(w16, copy=True).astype(np.float16, copy=False)
    bits = w_np.view(np.uint16)
    high = mx.array((bits >> 8).astype(np.uint8, copy=True))
    low = mx.array((bits & 0xFF).astype(np.uint8, copy=True))
    mx.eval(high, low)
    return w16, high, low


def main():
    bench.MODEL = MODEL
    model, tok = load(MODEL)
    mx.eval(model.parameters())
    kernels = bench.make_kernels()
    w16, high, low = prepare_weights(model)

    old_prompt = bench.PROMPT
    bench.PROMPT = PROMPTS[0]
    for mode in ["dense_custom_fp16", "proofbits_fp16"]:
        bench.decode_once(model, tok, mode, kernels, w16, high, low, 6, False)
    mx.synchronize()

    rows = []
    dense_total_ms = 0.0
    pb_total_ms = 0.0
    for i, prompt in enumerate(PROMPTS):
        bench.PROMPT = prompt
        order = ["dense_custom_fp16", "proofbits_fp16"] if i % 2 == 0 else ["proofbits_fp16", "dense_custom_fp16"]
        res = {}
        for mode in order:
            gc.collect(); mx.clear_cache()
            res[mode] = bench.decode_once(model, tok, mode, kernels, w16, high, low, TOKENS, False)
        d, p = res["dense_custom_fp16"], res["proofbits_fp16"]
        dsum = d["mean_ms_per_token"] * TOKENS
        psum = p["mean_ms_per_token"] * TOKENS
        dense_total_ms += dsum
        pb_total_ms += psum
        rows.append({
            "prompt_index": i,
            "prompt": prompt,
            "order": order,
            "sequence_exact": d["tokens"] == p["tokens"],
            "dense_median_ms": d["median_ms_per_token"],
            "proofbits_median_ms": p["median_ms_per_token"],
            "median_speedup": d["median_ms_per_token"] / p["median_ms_per_token"],
            "dense_mean_ms": d["mean_ms_per_token"],
            "proofbits_mean_ms": p["mean_ms_per_token"],
            "mean_speedup": d["mean_ms_per_token"] / p["mean_ms_per_token"],
            "dense_sum_ms": dsum,
            "proofbits_sum_ms": psum,
        })

    bench.PROMPT = old_prompt
    med_s = [x["median_speedup"] for x in rows]
    mean_s = [x["mean_speedup"] for x in rows]
    out = {
        "kind": "proofbits_gemma_integrated_stress_suite",
        "model": MODEL,
        "n_prompts": len(PROMPTS),
        "tokens_per_prompt": TOKENS,
        "matched_tokens": len(PROMPTS) * TOKENS,
        "all_sequences_exact": all(x["sequence_exact"] for x in rows),
        "exact_prompt_count": sum(x["sequence_exact"] for x in rows),
        "median_of_prompt_median_speedups": float(statistics.median(med_s)),
        "mean_of_prompt_median_speedups": float(statistics.mean(med_s)),
        "min_prompt_median_speedup": float(min(med_s)),
        "max_prompt_median_speedup": float(max(med_s)),
        "median_of_prompt_mean_speedups": float(statistics.median(mean_s)),
        "pooled_latency_speedup": dense_total_ms / pb_total_ms,
        "dense_total_decode_ms": dense_total_ms,
        "proofbits_total_decode_ms": pb_total_ms,
        "rows": rows,
        "note": "Twelve heterogeneous prompts, independent KV caches, 64 generated tokens each. Dense and ProofBits share the same FP16 output-head storage and MLX transformer body. Execution order alternates and neither path includes asymmetric diagnostics."
    }
    Path("experiments/artifacts").mkdir(parents=True, exist_ok=True)
    Path("experiments/artifacts/proofbits_mlx_gemma_stress_suite.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
