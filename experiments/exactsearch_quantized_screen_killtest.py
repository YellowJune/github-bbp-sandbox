import json
import math
import os
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SEED = 20260815
TEST_N = 64
MAX_LEN = 64
BATCH = 8
PILOT_TOP = 32
CONFIGS = [
    {"bits": 4, "group": 32},
    {"bits": 4, "group": 64},
    {"bits": 4, "group": 128},
    {"bits": 8, "group": 64},
    {"bits": 8, "group": 128},
]


def log(x):
    print(time.strftime("[%H:%M:%S]"), x, flush=True)


def select_texts(split, n, seed):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randperm(len(split), generator=g).tolist()
    out = []
    for i in ids:
        t = split[i]["text"].strip()
        if len(t) >= 40:
            out.append(t)
        if len(out) == n:
            return out
    raise RuntimeError("not enough texts")


def collect_hidden(model, tok, texts):
    hs = []
    with torch.inference_mode():
        for s in range(0, len(texts), BATCH):
            enc = tok(texts[s:s+BATCH], return_tensors="pt", padding=True,
                      truncation=True, max_length=MAX_LEN)
            out = model.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                              use_cache=False, return_dict=True).last_hidden_state
            last = enc["attention_mask"].sum(1) - 1
            h = out[torch.arange(out.shape[0]), last].detach().float().cpu()
            hs.append(h)
    return torch.cat(hs, 0)


def quantize_groupwise(W, bits, group):
    V, d = W.shape
    assert d % group == 0
    G = d // group
    qmax = (1 << (bits - 1)) - 1
    X = W.view(V, G, group)
    scale = X.abs().amax(dim=2, keepdim=True) / qmax
    scale = torch.clamp(scale, min=1e-12)
    q = torch.round(X / scale).clamp(-qmax, qmax)
    Xhat = q * scale
    err = X - Xhat
    err_norm = torch.linalg.vector_norm(err, dim=2)  # [V,G], precomputed index metadata
    What = Xhat.reshape(V, d).contiguous()
    mse = float((err * err).mean())
    maxerr = float(err.abs().max())
    return What, err_norm, mse, maxerr


def evaluate_cfg(W, H, dense, cfg):
    bits, group = cfg["bits"], cfg["group"]
    V, d = W.shape
    G = d // group
    log(f"quantizing bits={bits} group={group}")
    What, err_norm, mse, maxerr = quantize_groupwise(W, bits, group)
    log("approximate all-vocab scores")
    approx = H @ What.T
    Hgroups = H.view(H.shape[0], G, group)
    hnorm = torch.linalg.vector_norm(Hgroups, dim=2)  # [Q,G]
    # Blockwise Cauchy: |sum_g h_g^T e_ig| <= sum_g ||h_g|| ||e_ig||.
    err_bound = hnorm @ err_norm.T
    safety = 3e-4 * (approx.abs() + err_bound.abs() + 1.0)
    ub = approx + err_bound + safety

    counts = []
    matches = 0
    bound_width_winner = []
    for j in range(H.shape[0]):
        pilot = torch.topk(approx[j], k=PILOT_TOP).indices
        lower = dense[j, pilot].max()
        cand = torch.nonzero(ub[j] >= lower, as_tuple=False).flatten()
        union_n = torch.unique(torch.cat([pilot, cand])).numel()
        counts.append(int(union_n))
        pred = cand[dense[j, cand].argmax()].item()
        true = dense[j].argmax().item()
        matches += int(pred == true)
        bound_width_winner.append(float(err_bound[j, true]))

    c = torch.tensor(counts, dtype=torch.float64)
    frac = c / V
    # Idealized model-state bandwidth vs a dense FP16 lm_head:
    # all weights read at bits precision, then surviving rows reread at FP16.
    bw_ratio = bits / 16.0 + frac
    # MAC-equivalent is not reduced by quantization, so report it separately.
    res = {
        "bits": bits,
        "group": group,
        "groups": G,
        "weight_quant_mse": mse,
        "weight_quant_max_abs_error": maxerr,
        "exact_match_rate": matches / H.shape[0],
        "exact_candidate_fraction_mean": float(frac.mean()),
        "exact_candidate_fraction_median": float(frac.median()),
        "exact_candidate_fraction_p90": float(torch.quantile(frac, 0.9)),
        "exact_candidate_fraction_max": float(frac.max()),
        "exact_candidate_count_mean": float(c.mean()),
        "winner_error_bound_mean": float(torch.tensor(bound_width_winner).mean()),
        "idealized_fp16_weight_bandwidth_ratio_mean": float(bw_ratio.mean()),
        "idealized_lmhead_weight_bandwidth_reduction": float(1.0 / bw_ratio.mean()),
    }
    del What, err_norm, approx, err_bound, ub
    return res


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    log("loading model")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32,
                                                  low_cpu_mem_usage=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    W = model.lm_head.weight.detach().float().cpu()
    V, d = W.shape
    log(f"W={V}x{d}")

    ds = load_dataset("ag_news")
    texts = select_texts(ds["test"], TEST_N, SEED + 9)
    H = collect_hidden(model, tok, texts)
    log("dense oracle")
    dense = H @ W.T
    margins = torch.topk(dense, k=2, dim=1).values
    margin = margins[:, 0] - margins[:, 1]

    results = []
    for cfg in CONFIGS:
        r = evaluate_cfg(W, H, dense, cfg)
        results.append(r)
        log("bits=%d g=%d match=%.3f cand=%.5f p90=%.5f bw=%.3f headBWx=%.2f errB=%.3f" % (
            r["bits"], r["group"], r["exact_match_rate"],
            r["exact_candidate_fraction_mean"], r["exact_candidate_fraction_p90"],
            r["idealized_fp16_weight_bandwidth_ratio_mean"],
            r["idealized_lmhead_weight_bandwidth_reduction"],
            r["winner_error_bound_mean"],
        ))

    out = {
        "kind": "quantized_interval_exact_lmhead_screening_killtest",
        "model": MODEL_ID,
        "dataset": "AG News held-out raw text; labels unused",
        "queries": TEST_N,
        "vocab": V,
        "hidden_dim": d,
        "pilot_top": PILOT_TOP,
        "dense_top1_top2_margin_mean": float(margin.mean()),
        "dense_top1_top2_margin_median": float(margin.median()),
        "results": results,
        "notes": {
            "certificate": "For each token and dimension-group, quantization residual norms are precomputed. At runtime blockwise Cauchy gives a rigorous score-error interval; candidates whose upper bound is below an exact pilot lower bound are discarded.",
            "bandwidth_metric": "bits/16 + exact_candidate_fraction, an idealized lm-head weight-read ratio vs FP16. It is not an end-to-end speedup claim and ignores scale/metadata/kernel overhead.",
        },
    }
    os.makedirs("experiments/artifacts", exist_ok=True)
    with open("experiments/artifacts/exactsearch_quantized_screen_result.json", "w") as f:
        json.dump(out, f, indent=2)
    print("=== EXACTSEARCH_QUANTIZED_SCREEN ===")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
