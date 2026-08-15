import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("PB_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
N = int(os.environ.get("PB_N", "8"))
CHUNK = int(os.environ.get("PB_CHUNK", "4096"))

PROMPTS = [
    "Explain why logarithms appear in entropy.",
    "Solve x + 1/x = 3 and derive a higher power identity.",
    "Write a short Python function for a linear-time scan.",
    "Describe the tradeoff between memory bandwidth and arithmetic intensity.",
    "Explain natural selection without teleological language.",
    "Compare optimistic and pessimistic concurrency control.",
    "Give an intuitive explanation of the central limit theorem.",
    "Design a small experiment that distinguishes causation from correlation.",
]


def bits_i32(x):
    # CPU PyTorch has incomplete uint16 bitwise support. Reinterpret as signed
    # int16, widen to int32, then mask back to the raw 16-bit storage pattern.
    return x.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF


def from_bits(bits_i32, dtype):
    raw_i16 = bits_i32.to(torch.int16).contiguous()
    if dtype == torch.bfloat16:
        return raw_i16.view(torch.bfloat16)
    if dtype == torch.float16:
        return raw_i16.view(torch.float16)
    raise ValueError(dtype)


def certify(weight, h, dtype):
    V, D = weight.shape
    h = h.float().contiguous()
    hs = (h < 0).to(torch.int32) * 0x80
    pilot_u = None
    pilot_idx = -1
    upper_chunks = []

    for s in range(0, V, CHUNK):
        e = min(V, s + CHUNK)
        w = weight[s:e].contiguous()
        b = bits_i32(w)
        high = b >> 8
        ws = high & 0x80
        suffix = torch.where(ws == hs[None, :], torch.full_like(high, 0x00FF), torch.zeros_like(high))
        raw = (high << 8) | suffix
        endpoint = from_bits(raw, dtype).float()
        if not torch.isfinite(endpoint).all():
            return {"finite": False, "reason": "nonfinite_interval_endpoint"}
        U = endpoint @ h
        upper_chunks.append(U)
        m, j = torch.max(U, dim=0)
        if pilot_u is None or float(m) > pilot_u:
            pilot_u = float(m)
            pilot_idx = s + int(j)

    B = float(torch.dot(weight[pilot_idx].float(), h))
    survivors = 0
    best = -float("inf")
    best_idx = -1
    offset = 0
    for s in range(0, V, CHUNK):
        e = min(V, s + CHUNK)
        U = upper_chunks[offset]; offset += 1
        survivors += int((U >= B).sum())
        z = weight[s:e].float() @ h
        m, j = torch.max(z, dim=0)
        if float(m) > best:
            best = float(m); best_idx = s + int(j)
    c = best_idx // CHUNK
    local = best_idx - c * CHUNK
    winner_survives = bool(float(upper_chunks[c][local]) >= B)

    refined_best = -float("inf")
    refined_idx = -1
    offset = 0
    for s in range(0, V, CHUNK):
        e = min(V, s + CHUNK)
        U = upper_chunks[offset]; offset += 1
        mask = U >= B
        if mask.any():
            idx = torch.nonzero(mask, as_tuple=False).flatten()
            z = weight[s:e][idx].float() @ h
            m, j = torch.max(z, dim=0)
            if float(m) > refined_best:
                refined_best = float(m); refined_idx = s + int(idx[int(j)])

    return {
        "finite": True,
        "pilot": pilot_idx,
        "survivors": survivors,
        "survivor_fraction": survivors / V,
        "dense_winner": best_idx,
        "refined_winner": refined_idx,
        "exact": refined_idx == best_idx,
        "winner_survives": winner_survives,
        "dense_max": best,
        "pilot_score": B,
    }


def main():
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.eval()
    out_emb = model.get_output_embeddings()
    if out_emb is None or not hasattr(out_emb, "weight"):
        raise RuntimeError("No output embedding weight")
    Wbf = out_emb.weight.detach().cpu().to(torch.bfloat16).contiguous()
    Wfp = Wbf.to(torch.float16).contiguous()
    V, D = Wbf.shape

    hidden = []
    with torch.no_grad():
        for text in PROMPTS[:N]:
            x = tok(text, return_tensors="pt")
            o = model(**x, output_hidden_states=True, use_cache=False)
            hidden.append(o.hidden_states[-1][0, -1].detach().cpu().float())

    rows = []
    for i, h in enumerate(hidden):
        bf = certify(Wbf, h, torch.bfloat16)
        fp = certify(Wfp, h, torch.float16)
        rows.append({"i": i, "bf16": bf, "fp16_control": fp})
        print(json.dumps(rows[-1]))

    def summary(key):
        rr = [r[key] for r in rows]
        finite = [x for x in rr if x.get("finite")]
        if not finite:
            return {"finite_cases": 0, "n": len(rr)}
        fr = [x["survivor_fraction"] for x in finite]
        return {
            "n": len(rr),
            "finite_cases": len(finite),
            "exact_cases": sum(bool(x.get("exact")) for x in finite),
            "winner_survival_cases": sum(bool(x.get("winner_survives")) for x in finite),
            "mean_survivors": sum(x["survivors"] for x in finite) / len(finite),
            "mean_survivor_fraction": sum(fr) / len(fr),
            "min_survivor_fraction": min(fr),
            "max_survivor_fraction": max(fr),
            "ideal_weight_byte_reduction_from_fraction": 2.0 / (1.0 + sum(fr)/len(fr)),
        }

    result = {
        "kind": "proofbits_native_bf16_byteplane_killtest",
        "model": MODEL,
        "V": int(V), "D": int(D), "N": len(rows),
        "bf16": summary("bf16"),
        "fp16_control": summary("fp16_control"),
        "rows": rows,
        "interpretation": "BF16 8+8 split is retained only if its survivor fraction remains small enough to justify a native-BF16 system path. FP16 control uses the same hidden states and weights rounded from BF16 to FP16."
    }
    Path("experiments/artifacts").mkdir(parents=True, exist_ok=True)
    slug = MODEL.split("/")[-1].replace(".", "_")
    Path(f"experiments/artifacts/proofbits_bf16_killtest_{slug}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
