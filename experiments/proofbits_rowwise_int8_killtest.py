import json, math, os, time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cpu"
N_AG = 96
N_WIKI = 96
N_GEN = 128
MAX_LEN = 128
OUT = Path("experiments/artifacts/proofbits_rowwise_int8.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

torch.set_grad_enabled(False)
torch.manual_seed(0)
np.random.seed(0)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def chunked_scores(h, q, scale, chunk=4096):
    # h: [N,D], q: [V,D] int8, scale: [V]
    outs = []
    hf = h.float()
    for s in range(0, q.shape[0], chunk):
        qc = q[s:s+chunk].float() * scale[s:s+chunk, None]
        outs.append(hf @ qc.T)
    return torch.cat(outs, dim=1)


def chunked_fp32_scores(h, w, chunk=4096):
    outs = []
    hf = h.float()
    for s in range(0, w.shape[0], chunk):
        outs.append(hf @ w[s:s+chunk].float().T)
    return torch.cat(outs, dim=1)


def prefix_midpoint_q(q, bits):
    # Offset binary code u in [0,255]; q is symmetric int8 in [-127,127].
    u = q.to(torch.int16) + 128
    step = 1 << (8 - bits)
    lo = (u // step) * step
    hi = lo + (step - 1)
    mid = (lo.float() + hi.float()) * 0.5 - 128.0
    return mid


def prefix_scores(h, q, scale, bits, chunk=4096):
    outs = []
    hf = h.float()
    for s in range(0, q.shape[0], chunk):
        mid = prefix_midpoint_q(q[s:s+chunk], bits)
        outs.append((hf @ mid.T) * scale[s:s+chunk][None, :])
    return torch.cat(outs, dim=1)


def certify_4plus4(h, q, scale, exact8):
    # Full 4-bit base pass; fully read one coarse-best row as pilot; residual 4 bits only for candidates.
    coarse = prefix_scores(h, q, scale, bits=4)
    pilot = coarse.argmax(dim=1)
    B = exact8.gather(1, pilot[:, None]).squeeze(1)

    # Row-wise scale makes a tight deterministic coordinate-wise residual bound:
    # |delta z_i| <= r_b * scale_i * ||h||_1, r_4 = 7.5.
    r = 7.5
    h_l1 = h.float().abs().sum(dim=1)
    err = h_l1[:, None] * (r * scale[None, :])
    upper = coarse + err
    cand = upper >= B[:, None]

    ref = exact8.argmax(dim=1)
    pred = []
    counts = []
    for n in range(h.shape[0]):
        idx = cand[n].nonzero(as_tuple=False).squeeze(1)
        counts.append(int(idx.numel()))
        vals = exact8[n, idx]
        pred.append(int(idx[vals.argmax()].item()))
    pred = torch.tensor(pred)
    counts = np.asarray(counts)

    D = h.shape[1]
    V = q.shape[0]
    frac = counts.mean() / V
    scale_bits_per_weight = 16.0 / D
    dense_bits = 8.0 + scale_bits_per_weight
    proof_bits = 4.0 + 4.0 * frac + scale_bits_per_weight

    return {
        "certified_exact_int8_match_rate": float((pred == ref).float().mean().item()),
        "candidate_mean": float(counts.mean()),
        "candidate_median": float(np.median(counts)),
        "candidate_p90": float(np.percentile(counts, 90)),
        "candidate_p99": float(np.percentile(counts, 99)),
        "candidate_max": int(counts.max()),
        "candidate_fraction_mean": float(frac),
        "scale_bits_per_weight": float(scale_bits_per_weight),
        "proofbits_total_bits_per_weight": float(proof_bits),
        "dense_rowwise_int8_bits_per_weight": float(dense_bits),
        "idealized_bw_reduction_vs_dense_rowwise_int8": float(dense_bits / proof_bits),
    }


def collect_last_hidden(model, tok, texts):
    hs = []
    for i, t in enumerate(texts):
        x = tok(t, return_tensors="pt", truncation=True, max_length=MAX_LEN)
        out = model.model(**x, use_cache=False, return_dict=True)
        hs.append(out.last_hidden_state[0, -1].float().cpu())
    return torch.stack(hs)


def collect_wiki_nexttoken(model, tok, texts):
    hs, targets = [], []
    for t in texts:
        x = tok(t, return_tensors="pt", truncation=True, max_length=MAX_LEN)
        ids = x["input_ids"]
        if ids.shape[1] < 4:
            continue
        prefix = ids[:, :-1]
        attn = torch.ones_like(prefix)
        out = model.model(input_ids=prefix, attention_mask=attn, use_cache=False, return_dict=True)
        hs.append(out.last_hidden_state[0, -1].float().cpu())
        targets.append(int(ids[0, -1].item()))
    return torch.stack(hs), torch.tensor(targets, dtype=torch.long)


def collect_autoregressive(model, tok):
    prompts = [
        "Solve carefully: 137 * 29 =",
        "If x^2 - 5x + 6 = 0, the roots are",
        "Write a Python function that returns the Fibonacci number for n:",
        "Complete the code: for i in range(10):\n    print(",
        "The derivative of x^3 + 2x is",
        "A train travels 120 km in 1.5 hours. Its average speed is",
        "Implement binary search in Python:\n",
        "Factor 84 into prime factors:",
        "What is 17 choose 3?",
        "Write a SQL query selecting users older than 18:",
        "Evaluate the integral of 2x from 0 to 3:",
        "In Python, reverse a list without modifying the original list:",
        "Compute 2^10 =",
        "The gcd of 84 and 126 is",
        "Write pseudocode for breadth-first search:",
        "Simplify (x+2)(x-2):",
    ]
    seqs = [tok(p, return_tensors="pt")["input_ids"] for p in prompts]
    hs = []
    steps = math.ceil(N_GEN / len(seqs))
    for step in range(steps):
        new_seqs = []
        for ids in seqs:
            out = model.model(input_ids=ids, use_cache=False, return_dict=True)
            h = out.last_hidden_state[0, -1].float().cpu()
            hs.append(h)
            if len(hs) >= N_GEN:
                return torch.stack(hs)
            # Generation uses dense FP32 head only to move to the next realistic state.
            logits = model.lm_head(h[None, :])[0]
            nxt = logits.argmax().view(1, 1)
            new_seqs.append(torch.cat([ids, nxt], dim=1))
        seqs = new_seqs
    return torch.stack(hs[:N_GEN])


def nll(logits, targets):
    return torch.logsumexp(logits.float(), dim=1) - logits.float().gather(1, targets[:, None]).squeeze(1)


def main():
    log("load tokenizer/model")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
    model.eval()

    W = model.lm_head.weight.detach().float().cpu().contiguous()
    V, D = W.shape
    log(f"lm_head V={V} D={D}")

    # Row-wise symmetric INT8 deployment.
    scale = W.abs().amax(dim=1).clamp_min(1e-8) / 127.0
    q = torch.round(W / scale[:, None]).clamp(-127, 127).to(torch.int8)

    log("load datasets")
    ag = load_dataset("ag_news", split="test")
    ag_texts = [ag[i]["text"] for i in range(N_AG)]
    wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    wiki_texts = [x["text"] for x in wiki if len(x["text"].strip()) > 80][:N_WIKI]

    log("collect AG hidden")
    h_ag = collect_last_hidden(model, tok, ag_texts)
    log("collect Wiki next-token hidden")
    h_wiki, y_wiki = collect_wiki_nexttoken(model, tok, wiki_texts)
    log("collect autoregressive hidden")
    h_gen = collect_autoregressive(model, tok)

    results = {}
    for name, h in [("ag_news", h_ag), ("wikitext", h_wiki), ("autoregressive_math_code", h_gen)]:
        log(f"score {name} n={h.shape[0]}")
        z8 = chunked_scores(h, q, scale)
        zfp = chunked_fp32_scores(h, W)
        m = certify_4plus4(h, q, scale, z8)
        m["queries"] = int(h.shape[0])
        m["rowwise_int8_vs_fp32_argmax_agreement"] = float((z8.argmax(1) == zfp.argmax(1)).float().mean().item())
        if name == "wikitext":
            n0 = nll(zfp, y_wiki)
            n8 = nll(z8, y_wiki)
            m["fp32_mean_nll"] = float(n0.mean().item())
            m["rowwise_int8_mean_nll"] = float(n8.mean().item())
            m["mean_nll_delta"] = float((n8 - n0).mean().item())
            m["relative_ppl_factor_exp_delta_nll"] = float(math.exp((n8 - n0).mean().item()))
        results[name] = m
        log(f"{name}: {m}")
        del z8, zfp

    report = {
        "kind": "proofbits_rowwise_int8_4plus4_hardware_friendly",
        "model": MODEL,
        "vocab": V,
        "hidden_dim": D,
        "quantization": "row-wise symmetric INT8, one FP16-equivalent scale per vocabulary row",
        "layout_target": "4-bit MSB-centered base stream + 4-bit residual stream",
        "certificate": "pilot=1 exact row; U_i = coarse_i + 7.5*scale_i*||h||_1; residual nibble fetched only for candidates",
        "extra_certificate_metadata_bytes": 0,
        "results": results,
        "caveat": "Traffic ratios are idealized bytes/bits only. No GPU kernel wall-clock claim yet. FP16-scale traffic is counted as 16 bits per vocabulary row.",
    }
    OUT.write_text(json.dumps(report, indent=2))
    print("=== ROWWISE_PROOFBITS ===")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
