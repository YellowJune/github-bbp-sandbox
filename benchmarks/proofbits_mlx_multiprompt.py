import gc
import json
import os
import statistics
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load

sys.path.insert(0, str(Path(__file__).resolve().parent))
import proofbits_mlx_integrated_decode as bench

MODEL = os.environ.get("PB_MODEL", "mlx-community/Qwen2.5-0.5B-Instruct-bf16")
TOKENS = int(os.environ.get("PB_TOKENS", "32"))
PROMPTS = [
    "Explain why the logarithm appears naturally in entropy and information theory.",
    "Solve carefully: if x + 1/x = 3, derive x^5 + 1/x^5.",
    "Write a short Python function that returns the longest increasing contiguous run in a list and explain its complexity.",
    "Summarize the key tradeoff between memory bandwidth and arithmetic intensity in autoregressive inference.",
    "Write a compact science-fiction scene about a computer that can prove which memory bytes it never needs to read.",
]


def prepare_weights(model):
    if hasattr(model, "lm_head"):
        src = model.lm_head.weight
        source_name = "lm_head.weight"
    elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        src = model.model.embed_tokens.weight
        source_name = "model.embed_tokens.weight"
    else:
        raise RuntimeError("Could not locate output-head weight")
    w16 = src.astype(mx.float16)
    mx.eval(w16)
    w_np = np.array(w16, copy=True).astype(np.float16, copy=False)
    bits = w_np.view(np.uint16)
    high = mx.array((bits >> 8).astype(np.uint8, copy=True))
    low = mx.array((bits & 0xFF).astype(np.uint8, copy=True))
    mx.eval(high, low)
    print({"head_source": source_name, "shape": list(w16.shape), "dtype": str(w16.dtype)})
    return w16, high, low


def main():
    bench.MODEL = MODEL
    model, tok = load(MODEL)
    mx.eval(model.parameters())
    kernels = bench.make_kernels()
    w16, high, low = prepare_weights(model)

    old_prompt = bench.PROMPT
    bench.PROMPT = PROMPTS[0]
    # Compile body/head kernels before any timed prompt.
    for mode in ["dense_custom_fp16", "proofbits_fp16"]:
        bench.decode_once(model, tok, mode, kernels, w16, high, low, 6, False)
    mx.synchronize()

    rows = []
    for i, prompt in enumerate(PROMPTS):
        bench.PROMPT = prompt
        order = ["dense_custom_fp16", "proofbits_fp16"] if i % 2 == 0 else ["proofbits_fp16", "dense_custom_fp16"]
        res = {}
        for mode in order:
            gc.collect(); mx.clear_cache()
            # Strictly symmetric timed paths: no survivor diagnostics.
            res[mode] = bench.decode_once(model, tok, mode, kernels, w16, high, low, TOKENS, False)
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
        })

    bench.PROMPT = old_prompt
    speeds = [r["speedup"] for r in rows]
    out = {
        "kind": "proofbits_mlx_integrated_multiprompt_no_diagnostics",
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
        "note": "Each prompt uses an independent KV cache. Dense and ProofBits share the same FP16 head representation and MLX body; execution order alternates. Neither timed path evaluates survivor diagnostics."
    }
    slug = MODEL.split('/')[-1].replace('.', '_')
    Path("experiments/artifacts").mkdir(parents=True, exist_ok=True)
    out_path = Path(f"experiments/artifacts/proofbits_mlx_multiprompt_{slug}.json")
    out_path.write_text(json.dumps(out, indent=2))
    # Preserve legacy filename for the original Qwen workflow.
    if "Qwen2.5-0.5B" in MODEL:
        Path("experiments/artifacts/proofbits_mlx_multiprompt.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
