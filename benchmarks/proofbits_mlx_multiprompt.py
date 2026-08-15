import gc
import json
import statistics
import sys
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

sys.path.insert(0, str(Path(__file__).resolve().parent))
import proofbits_mlx_integrated_decode as bench

MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-bf16"
TOKENS = 32
PROMPTS = [
    "Explain why the logarithm appears naturally in entropy and information theory.",
    "Solve carefully: if x + 1/x = 3, derive x^5 + 1/x^5.",
    "Write a short Python function that returns the longest increasing contiguous run in a list and explain its complexity.",
    "Summarize the key tradeoff between memory bandwidth and arithmetic intensity in autoregressive inference.",
    "Write a compact science-fiction scene about a computer that can prove which memory bytes it never needs to read.",
]


def main():
    model, tok = load(MODEL)
    mx.eval(model.parameters())
    kernels = bench.make_kernels()
    w16, high, low = bench.prepare_weights(model)

    # Compile kernels/body before measurements.
    old_prompt = bench.PROMPT
    bench.PROMPT = PROMPTS[0]
    bench.warmup(model, tok, kernels, w16, high, low)

    rows = []
    for i, prompt in enumerate(PROMPTS):
        bench.PROMPT = prompt
        order = ["dense_custom_fp16", "proofbits_fp16"] if i % 2 == 0 else ["proofbits_fp16", "dense_custom_fp16"]
        res = {}
        for mode in order:
            gc.collect(); mx.clear_cache()
            res[mode] = bench.decode_once(model, tok, mode, kernels, w16, high, low, TOKENS, mode == "proofbits_fp16")
        d, p = res["dense_custom_fp16"], res["proofbits_fp16"]
        rows.append({
            "prompt_index": i,
            "prompt": prompt,
            "order": order,
            "sequence_exact": d["tokens"] == p["tokens"],
            "dense_median_ms": d["median_ms_per_token"],
            "proofbits_median_ms": p["median_ms_per_token"],
            "speedup": d["median_ms_per_token"] / p["median_ms_per_token"],
            "dense_tps": d["tokens_per_s_from_median"],
            "proofbits_tps": p["tokens_per_s_from_median"],
            "survivor_mean": p.get("survivor_mean"),
            "survivor_fraction_mean": p.get("survivor_fraction_mean"),
        })

    bench.PROMPT = old_prompt
    speeds = [r["speedup"] for r in rows]
    out = {
        "kind": "proofbits_mlx_integrated_multiprompt",
        "model": MODEL,
        "n_prompts": len(PROMPTS),
        "tokens_per_prompt": TOKENS,
        "total_matched_tokens": len(PROMPTS) * TOKENS,
        "all_sequences_exact": all(r["sequence_exact"] for r in rows),
        "median_speedup": float(statistics.median(speeds)),
        "mean_speedup": float(statistics.mean(speeds)),
        "min_speedup": float(min(speeds)),
        "max_speedup": float(max(speeds)),
        "rows": rows,
        "note": "Each prompt uses an independent KV cache. Dense and ProofBits use the same FP16 head representation and same MLX body; order alternates across prompts."
    }
    Path("experiments/artifacts").mkdir(parents=True, exist_ok=True)
    Path("experiments/artifacts/proofbits_mlx_multiprompt.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
