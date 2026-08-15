import sys
from pathlib import Path

# Reuse the already validated integrated MLX harness. Only the model identifier
# and output-head resolver differ for Gemma 3, whose MLX-LM implementation has
# an explicit lm_head when untied.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import proofbits_mlx_integrated_decode as bench

bench.MODEL = "mlx-community/gemma-3-270m-bf16"
bench.PROMPT = "Explain in one paragraph why exact inference decisions can sometimes be certified from partial numerical representations."
bench.MAX_NEW = 48
bench.WARMUP_NEW = 6
bench.ROUNDS = 4


def prepare_gemma_weights(model):
    import mlx.core as mx
    import numpy as np

    if hasattr(model, "lm_head"):
        src = model.lm_head.weight
        source_name = "lm_head.weight"
    elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        src = model.model.embed_tokens.weight
        source_name = "model.embed_tokens.weight"
    else:
        raise RuntimeError("Could not locate Gemma output head weight")

    w16 = src.astype(mx.float16)
    mx.eval(w16)
    w_np = np.array(w16, copy=True).astype(np.float16, copy=False)
    bits = w_np.view(np.uint16)
    high = mx.array((bits >> 8).astype(np.uint8, copy=True))
    low = mx.array((bits & 0xFF).astype(np.uint8, copy=True))
    mx.eval(high, low)
    print({"gemma_head_source": source_name, "shape": list(w16.shape), "dtype": str(w16.dtype)})
    return w16, high, low


bench.prepare_weights = prepare_gemma_weights

if __name__ == "__main__":
    bench.main()
