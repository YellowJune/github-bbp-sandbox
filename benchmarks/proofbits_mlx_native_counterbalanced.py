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
TOKENS = int(os.environ.get("PB_TOKENS", "48"))
ROUNDS = int(os.environ.get("PB_ROUNDS", "4"))
PROMPT = "Explain in one paragraph why exact inference decisions can sometimes be certified from partial numerical representations."

bench.MODEL = MODEL
bench.PROMPT = PROMPT


def generic_weights(model):
    if hasattr(model, "lm_head"):
        src = model.lm_head.weight
        src_name = "lm_head.weight"
    elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        src = model.model.embed_tokens.weight
        src_name = "model.embed_tokens.weight"
    else:
        raise RuntimeError("No output head weight")
    w16 = src.astype(mx.float16)
    mx.eval(w16)
    w_np = np.array(w16, copy=True).astype(np.float16, copy=False)
    bits = w_np.view(np.uint16)
    high = mx.array((bits >> 8).astype(np.uint8, copy=True))
    low = mx.array((bits & 0xFF).astype(np.uint8, copy=True))
    mx.eval(high, low)
    print({"head_source": src_name, "shape": list(w16.shape)})
    return w16, high, low

bench.prepare_weights = generic_weights


def main():
    model, tok = load(MODEL)
    mx.eval(model.parameters())
    kernels = bench.make_kernels()
    w16, high, low = generic_weights(model)
    # Compile both paths outside measurement.
    for mode in ["native_mlx_fp16", "proofbits_fp16"]:
        bench.decode_once(model, tok, mode, kernels, w16, high, low, 6, False)
    mx.synchronize()

    rows=[]
    for r in range(ROUNDS):
        order=["native_mlx_fp16","proofbits_fp16"] if r%2==0 else ["proofbits_fp16","native_mlx_fp16"]
        res={}
        for mode in order:
            gc.collect(); mx.clear_cache()
            res[mode]=bench.decode_once(model,tok,mode,kernels,w16,high,low,TOKENS,False)
        n=res["native_mlx_fp16"]; p=res["proofbits_fp16"]
        rows.append({
            "round":r+1,
            "order":order,
            "native_median_ms":n["median_ms_per_token"],
            "proofbits_median_ms":p["median_ms_per_token"],
            "native_over_proofbits_speedup":n["median_ms_per_token"]/p["median_ms_per_token"],
            "sequence_equal":n["tokens"]==p["tokens"],
            "native_mean_ms":n["mean_ms_per_token"],
            "proofbits_mean_ms":p["mean_ms_per_token"],
        })
    sp=[x["native_over_proofbits_speedup"] for x in rows]
    out={
        "kind":"proofbits_vs_native_mlx_fp16_integrated_counterbalanced",
        "model":MODEL,
        "tokens_per_round":TOKENS,
        "rounds":rows,
        "all_sequences_equal":all(x["sequence_equal"] for x in rows),
        "median_speedup":float(statistics.median(sp)),
        "mean_speedup":float(statistics.mean(sp)),
        "min_speedup":float(min(sp)),
        "max_speedup":float(max(sp)),
        "note":"Native MLX FP16 matvec and ProofBits use the same FP16 head storage and same MLX KV-cache body. No survivor diagnostics are evaluated in either timed path. Sequence equality is reported rather than assumed because reduction semantics can differ."
    }
    slug=MODEL.split('/')[-1].replace('.','_')
    Path('experiments/artifacts').mkdir(parents=True,exist_ok=True)
    path=Path(f'experiments/artifacts/proofbits_native_counterbalanced_{slug}.json')
    path.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

if __name__=='__main__': main()
