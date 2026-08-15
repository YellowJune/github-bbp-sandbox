import gc
import json
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models import cache as cache_mod

sys.path.insert(0, str(Path(__file__).resolve().parent))
import proofbits_mlx_integrated_decode as bench

MODEL = "mlx-community/gemma-3-270m-bf16"
CONTEXTS = [16, 64, 256, 512, 1024]
NEW_TOKENS = 32
ROUNDS = 4
BASE_TEXT = "Memory bandwidth and exact numerical decisions interact in autoregressive inference. "


def med(xs):
    return float(statistics.median(xs))


def prepare_weights(model):
    src = model.lm_head.weight if hasattr(model, "lm_head") else model.model.embed_tokens.weight
    w16 = src.astype(mx.float16); mx.eval(w16)
    w_np = np.array(w16, copy=True).astype(np.float16, copy=False)
    bits = w_np.view(np.uint16)
    high = mx.array((bits >> 8).astype(np.uint8, copy=True))
    low = mx.array((bits & 0xFF).astype(np.uint8, copy=True))
    mx.eval(high, low)
    return w16, high, low


def exact_tokens(tok, n):
    try:
        base = list(tok.encode(BASE_TEXT))
    except Exception:
        base = list(tok(BASE_TEXT)["input_ids"])
    if not base:
        base = [1]
    ids = []
    while len(ids) < n:
        ids.extend(base)
    return ids[:n]


def decode_split(model, mode, kernels, w16, high, low, prompt_ids, max_new):
    V, D = [int(x) for x in w16.shape]
    kv = cache_mod.make_prompt_cache(model)
    dense_k, upper_k, pilot_k, refine_k = kernels

    t0 = time.perf_counter()
    body = model.model(mx.array(prompt_ids, dtype=mx.int32)[None], cache=kv)
    h = body[:, -1, :].reshape((D,)).astype(mx.float32)
    mx.eval(h, [c.state for c in kv])
    prefill_ms = (time.perf_counter() - t0) * 1e3

    tokens, head_ms, body_ms, total_ms = [], [], [], []
    for _ in range(max_new):
        ts = time.perf_counter()
        th = time.perf_counter()
        if mode == "dense_custom_fp16":
            y = bench.call_dense(dense_k, w16, h, V)
        elif mode == "proofbits_fp16":
            y = bench.call_proofbits(upper_k, pilot_k, refine_k, high, low, h, V, diagnostics=False)
        else:
            raise ValueError(mode)
        mx.eval(y)
        token = int(y.item())
        head_ms.append((time.perf_counter() - th) * 1e3)
        tokens.append(token)

        tb = time.perf_counter()
        body = model.model(mx.array([[token]], dtype=mx.int32), cache=kv)
        h = body[:, -1, :].reshape((D,)).astype(mx.float32)
        mx.eval(h, [c.state for c in kv])
        body_ms.append((time.perf_counter() - tb) * 1e3)
        total_ms.append((time.perf_counter() - ts) * 1e3)

    return {
        "tokens": tokens,
        "prefill_ms": float(prefill_ms),
        "median_total_ms": med(total_ms),
        "median_head_ms": med(head_ms),
        "median_body_ms": med(body_ms),
        "mean_total_ms": float(statistics.mean(total_ms)),
        "mean_head_ms": float(statistics.mean(head_ms)),
        "mean_body_ms": float(statistics.mean(body_ms)),
    }


def main():
    bench.MODEL = MODEL
    model, tok = load(MODEL)
    mx.eval(model.parameters())
    kernels = bench.make_kernels()
    w16, high, low = prepare_weights(model)

    # Compile body and all ProofBits kernels outside measurement.
    seed_ids = exact_tokens(tok, 16)
    for mode in ["dense_custom_fp16", "proofbits_fp16"]:
        decode_split(model, mode, kernels, w16, high, low, seed_ids, 4)
    mx.synchronize()

    rows = []
    for ci, nctx in enumerate(CONTEXTS):
        ids = exact_tokens(tok, nctx)
        per = []
        for r in range(ROUNDS):
            order = ["dense_custom_fp16", "proofbits_fp16"] if (ci + r) % 2 == 0 else ["proofbits_fp16", "dense_custom_fp16"]
            res = {}
            for mode in order:
                gc.collect(); mx.clear_cache()
                res[mode] = decode_split(model, mode, kernels, w16, high, low, ids, NEW_TOKENS)
            d, p = res["dense_custom_fp16"], res["proofbits_fp16"]
            per.append({
                "round": r + 1,
                "order": order,
                "sequence_exact": d["tokens"] == p["tokens"],
                "dense_total_ms": d["median_total_ms"],
                "proofbits_total_ms": p["median_total_ms"],
                "total_speedup": d["median_total_ms"] / p["median_total_ms"],
                "dense_head_ms": d["median_head_ms"],
                "proofbits_head_ms": p["median_head_ms"],
                "head_speedup": d["median_head_ms"] / p["median_head_ms"],
                "dense_body_ms": d["median_body_ms"],
                "proofbits_body_ms": p["median_body_ms"],
                "dense_prefill_ms": d["prefill_ms"],
                "proofbits_prefill_ms": p["prefill_ms"],
            })

        total_s = [x["total_speedup"] for x in per]
        head_s = [x["head_speedup"] for x in per]
        dense_head = [x["dense_head_ms"] for x in per]
        pb_head = [x["proofbits_head_ms"] for x in per]
        dense_body = [x["dense_body_ms"] for x in per]
        pb_body = [x["proofbits_body_ms"] for x in per]
        rows.append({
            "context_tokens": nctx,
            "rounds": per,
            "all_sequences_exact": all(x["sequence_exact"] for x in per),
            "median_total_speedup": med(total_s),
            "min_total_speedup": float(min(total_s)),
            "max_total_speedup": float(max(total_s)),
            "median_head_speedup": med(head_s),
            "median_dense_head_ms": med(dense_head),
            "median_proofbits_head_ms": med(pb_head),
            "median_dense_body_ms": med(dense_body),
            "median_proofbits_body_ms": med(pb_body),
            "dense_head_fraction": med(dense_head) / (med(dense_head) + med(dense_body)),
        })

    out = {
        "kind": "proofbits_gemma_integrated_context_length_sweep_stabilized",
        "model": MODEL,
        "contexts": CONTEXTS,
        "new_tokens_per_run": NEW_TOKENS,
        "rounds_per_context": ROUNDS,
        "rows": rows,
        "all_sequences_exact": all(x["all_sequences_exact"] for x in rows),
        "note": "Interactive decode is split into synchronized head-decision and body/KV-cache phases. This exposes the Amdahl boundary directly. Dense and ProofBits share FP16 head storage and identical MLX body; no asymmetric diagnostics are timed."
    }
    Path('experiments/artifacts').mkdir(parents=True, exist_ok=True)
    Path('experiments/artifacts/proofbits_mlx_context_sweep_gemma.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__ == '__main__':
    main()
