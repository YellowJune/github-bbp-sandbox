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
CONTEXTS = [16, 64, 256, 512]
NEW_TOKENS = 16
ROUNDS = 2
BASE_TEXT = "Memory bandwidth and exact numerical decisions interact in autoregressive inference. "


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


def main():
    bench.MODEL = MODEL
    model, tok = load(MODEL)
    mx.eval(model.parameters())
    kernels = bench.make_kernels()
    w16, high, low = prepare_weights(model)

    # Compile both paths once.
    seed_ids = exact_tokens(tok, 16)
    old_tokenize = bench.tokenize
    bench.tokenize = lambda _tok: mx.array(seed_ids, dtype=mx.int32)
    for mode in ["dense_custom_fp16", "proofbits_fp16"]:
        bench.decode_once(model, tok, mode, kernels, w16, high, low, 4, False)
    mx.synchronize()

    rows=[]
    for ci, nctx in enumerate(CONTEXTS):
        ids = exact_tokens(tok, nctx)
        bench.tokenize = lambda _tok, ids=ids: mx.array(ids, dtype=mx.int32)
        per=[]
        for r in range(ROUNDS):
            order=["dense_custom_fp16","proofbits_fp16"] if (ci+r)%2==0 else ["proofbits_fp16","dense_custom_fp16"]
            res={}
            for mode in order:
                gc.collect(); mx.clear_cache()
                res[mode]=bench.decode_once(model,tok,mode,kernels,w16,high,low,NEW_TOKENS,False)
            d,p=res["dense_custom_fp16"],res["proofbits_fp16"]
            per.append({
                "round":r+1,"order":order,
                "dense_median_ms":d["median_ms_per_token"],
                "proofbits_median_ms":p["median_ms_per_token"],
                "speedup":d["median_ms_per_token"]/p["median_ms_per_token"],
                "sequence_exact":d["tokens"]==p["tokens"],
                "dense_prefill_ms":d["prefill_ms"],
                "proofbits_prefill_ms":p["prefill_ms"],
            })
        rows.append({
            "context_tokens":nctx,
            "rounds":per,
            "all_sequences_exact":all(x["sequence_exact"] for x in per),
            "median_speedup":float(statistics.median(x["speedup"] for x in per)),
            "median_dense_ms":float(statistics.median(x["dense_median_ms"] for x in per)),
            "median_proofbits_ms":float(statistics.median(x["proofbits_median_ms"] for x in per)),
        })

    bench.tokenize = old_tokenize
    out={
        "kind":"proofbits_gemma_integrated_context_length_sweep",
        "model":MODEL,
        "contexts":CONTEXTS,
        "new_tokens_per_run":NEW_TOKENS,
        "rounds_per_context":ROUNDS,
        "rows":rows,
        "all_sequences_exact":all(x["all_sequences_exact"] for x in rows),
        "note":"Each context length uses exactly the requested number of prompt tokens built by repeating/truncating one fixed tokenized text. Dense and ProofBits share FP16 head storage and MLX body/KV cache; no asymmetric diagnostics are timed."
    }
    Path('experiments/artifacts').mkdir(parents=True,exist_ok=True)
    Path('experiments/artifacts/proofbits_mlx_context_sweep_gemma.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))

if __name__=='__main__': main()
