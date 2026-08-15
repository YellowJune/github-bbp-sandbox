import json
import math
import os
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cpu"
SEED = 20260815
CAL_N = 128
TEST_N = 64
MAX_LEN = 64
BATCH = 8
RANKS = [8, 16, 32, 64, 96, 128]
PILOT_TOP = 32


def log(msg):
    print(time.strftime("[%H:%M:%S]"), msg, flush=True)


def select_texts(split, n, seed):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(split), generator=g)[: n * 2].tolist()
    out = []
    for i in idx:
        t = split[i]["text"].strip()
        if len(t) >= 40:
            out.append(t)
        if len(out) == n:
            break
    if len(out) < n:
        raise RuntimeError(f"only found {len(out)} texts")
    return out


def collect_hidden(model, tok, texts, batch_size=BATCH):
    hs = []
    with torch.inference_mode():
        for s in range(0, len(texts), batch_size):
            batch = texts[s : s + batch_size]
            enc = tok(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
            )
            out = model.model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                use_cache=False,
                return_dict=True,
            ).last_hidden_state
            last = enc["attention_mask"].sum(dim=1) - 1
            rows = torch.arange(out.shape[0])
            h = out[rows, last].detach().float().cpu()
            hs.append(h)
            log(f"hidden {min(s+batch_size, len(texts))}/{len(texts)}")
    return torch.cat(hs, dim=0)


def safe_sqrt(x):
    return torch.sqrt(torch.clamp(x, min=0.0))


def eval_rank(W, WUmax, wnorm2, H, dense_scores, Umax, k, pilot_top=PILOT_TOP):
    V, d = W.shape
    U = Umax[:, :k]
    WU = WUmax[:, :k]
    wr = safe_sqrt(wnorm2 - (WU * WU).sum(dim=1))

    exact_counts = []
    cand_counts = []
    matches = 0
    margins = []
    bound_slacks = []
    winner_ubs = []

    for qi in range(H.shape[0]):
        h = H[qi]
        hu = h @ U
        hres = safe_sqrt((h * h).sum() - (hu * hu).sum()).item()
        proj = WU @ hu
        # Rigorous in exact arithmetic: h^T w_i <= proj_i + ||h_perp|| ||w_i_perp||.
        # Add a conservative FP32 safety pad so the experiment never relies on rounding.
        resid = hres * wr
        ub = proj + resid
        safety = 2e-4 * (proj.abs() + resid.abs() + 1.0)
        ub = ub + safety

        ptop = min(pilot_top, V)
        pilot = torch.topk(proj, k=ptop, largest=True).indices
        pilot_best = dense_scores[qi, pilot].max()
        cand = torch.nonzero(ub >= pilot_best, as_tuple=False).flatten()

        # Operationally the algorithm evaluates exact dots for cand. dense_scores is only
        # the oracle cache used here to avoid recomputing those dots during a kill-test.
        cand_scores = dense_scores[qi, cand]
        pred = cand[cand_scores.argmax()].item()
        true = dense_scores[qi].argmax().item()
        matches += int(pred == true)

        # Total exact full-dimensional dot products = initial pilot union surviving candidates.
        # pilot normally lies inside cand, but compute the exact union for honesty.
        union_count = torch.unique(torch.cat([pilot, cand])).numel()
        exact_counts.append(int(union_count))
        cand_counts.append(int(cand.numel()))

        top2 = torch.topk(dense_scores[qi], k=2).values
        margins.append(float(top2[0] - top2[1]))
        true_score = dense_scores[qi, true]
        winner_ubs.append(float(ub[true]))
        bound_slacks.append(float(ub[true] - true_score))

    t = torch.tensor(exact_counts, dtype=torch.float64)
    frac = t / V
    # Dominant arithmetic normalized to dense lm_head: all V low-rank k-dim scores + exact survivors.
    cost_ratio = (k / d) + frac
    speedup = 1.0 / cost_ratio
    res = {
        "rank": k,
        "queries": int(H.shape[0]),
        "vocab": int(V),
        "hidden_dim": int(d),
        "pilot_top": int(pilot_top),
        "exact_match_rate": matches / H.shape[0],
        "exact_dot_fraction_mean": float(frac.mean()),
        "exact_dot_fraction_median": float(frac.median()),
        "exact_dot_fraction_p90": float(torch.quantile(frac, 0.90)),
        "exact_dot_fraction_max": float(frac.max()),
        "exact_dot_count_mean": float(t.mean()),
        "normalized_arithmetic_mean": float(cost_ratio.mean()),
        "lm_head_arithmetic_speedup_mean": float(speedup.mean()),
        "lm_head_arithmetic_speedup_from_mean_cost": float(1.0 / cost_ratio.mean()),
        "query_residual_ratio_mean": None,
        "winner_bound_slack_mean": float(torch.tensor(bound_slacks).mean()),
        "dense_top1_top2_margin_mean": float(torch.tensor(margins).mean()),
    }
    # Query residual ratio for interpretability.
    Hproj = H @ U
    hr = safe_sqrt((H * H).sum(dim=1) - (Hproj * Hproj).sum(dim=1))
    hn = safe_sqrt((H * H).sum(dim=1)) + 1e-12
    res["query_residual_ratio_mean"] = float((hr / hn).mean())
    return res


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    log(f"loading {MODEL_ID}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, low_cpu_mem_usage=True
    ).to(DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    W = model.lm_head.weight.detach().float().cpu()
    V, d = W.shape
    log(f"lm_head vocab={V:,} dim={d}")

    ds = load_dataset("ag_news")
    cal_texts = select_texts(ds["train"], CAL_N, SEED)
    test_texts = select_texts(ds["test"], TEST_N, SEED + 1)

    log("collecting calibration hidden states")
    Hcal = collect_hidden(model, tok, cal_texts)
    log("collecting held-out hidden states")
    Htest = collect_hidden(model, tok, test_texts)

    # Uncentered PCA/SVD: directly minimizes energy left outside the query subspace.
    log("SVD of calibration hidden states")
    _, svals, Vh = torch.linalg.svd(Hcal, full_matrices=False)
    max_rank = min(max(RANKS), Vh.shape[0])
    Umax = Vh[:max_rank].T.contiguous()
    ortho_err = float((Umax.T @ Umax - torch.eye(max_rank)).abs().max())
    log(f"basis max_rank={max_rank} ortho_err={ortho_err:.3e}")

    log("precomputing output embeddings in calibration subspace")
    WUmax = W @ Umax
    wnorm2 = (W * W).sum(dim=1)

    log("computing dense oracle scores for held-out verification only")
    dense_scores = Htest @ W.T
    dense_idx = dense_scores.argmax(dim=1)

    results = []
    for k in RANKS:
        if k > max_rank:
            continue
        log(f"evaluating exact certified rank={k}")
        r = eval_rank(W, WUmax, wnorm2, Htest, dense_scores, Umax, k)
        results.append(r)
        log(
            "rank=%d match=%.3f mean_exact=%.4f p90=%.4f cost=%.4f lmhead_x=%.2f qres=%.4f"
            % (
                k,
                r["exact_match_rate"],
                r["exact_dot_fraction_mean"],
                r["exact_dot_fraction_p90"],
                r["normalized_arithmetic_mean"],
                r["lm_head_arithmetic_speedup_from_mean_cost"],
                r["query_residual_ratio_mean"],
            )
        )

    # Sanity: report singular-value energy and model/index sizes.
    sv_energy = (svals * svals)
    sv_energy = sv_energy / sv_energy.sum()
    cumulative = torch.cumsum(sv_energy, dim=0)
    energy = {str(k): float(cumulative[min(k, len(cumulative)) - 1]) for k in RANKS if k <= len(cumulative)}

    result = {
        "kind": "exact_certified_lm_head_search_killtest",
        "model": MODEL_ID,
        "dataset": "AG News raw text; calibration=train, held-out=test; labels unused",
        "calibration_queries": CAL_N,
        "heldout_queries": TEST_N,
        "max_length": MAX_LEN,
        "vocab": V,
        "hidden_dim": d,
        "basis": "uncentered SVD of separate calibration hidden states",
        "basis_orthogonality_max_error": ortho_err,
        "calibration_energy_cumulative": energy,
        "results": results,
        "interpretation": {
            "exactness": "Every pruned token has a Cauchy-Schwarz upper bound below an already evaluated exact lower bound; dense logits are computed only as an oracle to verify the kill-test.",
            "normalized_arithmetic": "k/d + exact_survivor_fraction; ignores indexing overhead and hardware effects, so it is NOT an end-to-end speedup claim.",
        },
    }
    os.makedirs("experiments/artifacts", exist_ok=True)
    out = "experiments/artifacts/exactsearch_lmhead_killtest_result.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print("=== EXACTSEARCH_LMHEAD_KILLTEST ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
