import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("PB_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
N = int(os.environ.get("PB_N", "2"))
CHUNK = int(os.environ.get("PB_CHUNK", "4096"))
PREFIXES = [8, 9, 10, 11, 12, 13, 14]
PROMPTS = [
    "Explain why logarithms appear in entropy and information theory.",
    "Summarize the memory-bandwidth bottleneck in autoregressive inference.",
]


def bits_i32(x):
    return x.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF


def from_bf16_bits(x_i32):
    return x_i32.to(torch.int16).contiguous().view(torch.bfloat16)


def dense_winner(W, h):
    best = -float("inf"); idx = -1
    V = W.shape[0]
    for s in range(0, V, CHUNK):
        e = min(V, s + CHUNK)
        z = W[s:e].float() @ h
        m, j = torch.max(z, dim=0)
        if float(m) > best:
            best = float(m); idx = s + int(j)
    return idx, best


def certify_prefix(W, h, p, dense_idx):
    V, D = W.shape
    q = 16 - p
    suffix_mask = (1 << q) - 1
    clear_mask = 0xFFFF ^ suffix_mask
    hs = (h < 0).to(torch.int32) * 0x8000
    upper = []
    pilot_val = -float("inf"); pilot_idx = -1

    for s in range(0, V, CHUNK):
        e = min(V, s + CHUNK)
        b = bits_i32(W[s:e])
        base = b & clear_mask
        ws = base & 0x8000
        endpoint_bits = torch.where(ws == hs[None, :], base | suffix_mask, base)
        endpoint = from_bf16_bits(endpoint_bits).float()
        if not torch.isfinite(endpoint).all():
            return {"prefix_bits": p, "finite": False, "reason": "nonfinite_interval_endpoint"}
        U = endpoint @ h
        upper.append(U)
        m, j = torch.max(U, dim=0)
        if float(m) > pilot_val:
            pilot_val = float(m); pilot_idx = s + int(j)

    B = float(torch.dot(W[pilot_idx].float(), h))
    survivors = sum(int((u >= B).sum()) for u in upper)
    c = dense_idx // CHUNK; local = dense_idx - c * CHUNK
    winner_survives = float(upper[c][local]) >= B

    refined_best = -float("inf"); refined_idx = -1
    for ci, s in enumerate(range(0, V, CHUNK)):
        e = min(V, s + CHUNK)
        mask = upper[ci] >= B
        if mask.any():
            ids = torch.nonzero(mask, as_tuple=False).flatten()
            z = W[s:e][ids].float() @ h
            m, j = torch.max(z, dim=0)
            if float(m) > refined_best:
                refined_best = float(m); refined_idx = s + int(ids[int(j)])

    f = survivors / V
    logical_bits = p + q * f
    return {
        "prefix_bits": p,
        "suffix_bits": q,
        "finite": True,
        "pilot": pilot_idx,
        "survivors": survivors,
        "survivor_fraction": f,
        "winner_survives": bool(winner_survives),
        "refined_winner": refined_idx,
        "exact": refined_idx == dense_idx,
        "logical_bits_per_weight": logical_bits,
        "ideal_weight_traffic_reduction": 16.0 / logical_bits,
    }


def main():
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.eval()
    W = model.get_output_embeddings().weight.detach().cpu().to(torch.bfloat16).contiguous()
    V, D = W.shape

    hs = []
    with torch.no_grad():
        for text in PROMPTS[:N]:
            x = tok(text, return_tensors="pt")
            o = model(**x, output_hidden_states=True, use_cache=False)
            hs.append(o.hidden_states[-1][0, -1].detach().cpu().float())

    rows=[]
    for qi, h in enumerate(hs):
        dw, dv = dense_winner(W, h)
        pr=[]
        for p in PREFIXES:
            r=certify_prefix(W,h,p,dw); pr.append(r)
            print(json.dumps({"query":qi, **r}))
        rows.append({"query":qi,"dense_winner":dw,"dense_max":dv,"prefixes":pr})

    summary={}
    for p in PREFIXES:
        rr=[next(x for x in row["prefixes"] if x["prefix_bits"]==p) for row in rows]
        finite=[x for x in rr if x.get("finite")]
        if finite:
            summary[str(p)]={
                "exact_cases":sum(bool(x.get("exact")) for x in finite),
                "n":len(finite),
                "mean_survivor_fraction":sum(x["survivor_fraction"] for x in finite)/len(finite),
                "mean_logical_bits_per_weight":sum(x["logical_bits_per_weight"] for x in finite)/len(finite),
                "mean_ideal_weight_traffic_reduction":sum(x["ideal_weight_traffic_reduction"] for x in finite)/len(finite),
            }
        else: summary[str(p)]={"n":0}

    result={
        "kind":"proofbits_native_bf16_prefix_width_pareto_sweep",
        "model":MODEL,"V":int(V),"D":int(D),"N":len(rows),
        "prefixes":PREFIXES,"summary":summary,"rows":rows,
        "note":"p denotes the number of most-significant BF16 storage bits fetched on the first pass. The suffix is fetched only for certified survivors. Logical traffic assumes densely packed p-bit prefixes and (16-p)-bit suffixes; it is not a measured kernel speedup."
    }
    Path("experiments/artifacts").mkdir(parents=True,exist_ok=True)
    slug=MODEL.split('/')[-1].replace('.','_')
    Path(f"experiments/artifacts/proofbits_bf16_prefix_sweep_{slug}.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=="__main__": main()
