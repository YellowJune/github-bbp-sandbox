import json, math, random, statistics, time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

SEED = 17
random.seed(SEED)
torch.manual_seed(SEED)

MODEL = "distilgpt2"
CTX = 192
BLOCK = 24
NB = CTX // BLOCK
N_SAMPLES = 8
EPS_BITS = [0.10, 0.25, 0.50]


def nll(model, ids, target, block_mask):
    # block_mask is length NB; masking a block removes it as a KV source in every layer.
    am = torch.zeros((1, CTX), dtype=torch.long)
    for b, keep in enumerate(block_mask):
        if keep:
            am[:, b * BLOCK : (b + 1) * BLOCK] = 1
    # last block (including the query token at position CTX-1) must stay active.
    am[:, (NB - 1) * BLOCK : NB * BLOCK] = 1
    with torch.inference_mode():
        out = model(input_ids=ids, attention_mask=am, use_cache=False)
        lp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
        return float(-lp[target].item())


def dense_attention_block_scores(model, ids):
    am = torch.ones((1, CTX), dtype=torch.long)
    with torch.inference_mode():
        out = model(input_ids=ids, attention_mask=am, output_attentions=True, use_cache=False)
    scores = torch.zeros(NB)
    for a in out.attentions:
        # [1,H,T,T], final query position; mean over heads.
        q = a[0, :, -1, :].float().mean(0)
        for b in range(NB):
            scores[b] += q[b * BLOCK : (b + 1) * BLOCK].sum().cpu()
    return scores.tolist()


def trace_losses(model, ids, target, order):
    selected = {NB - 1}
    rows = []
    for k in range(1, NB + 1):
        mask = [b in selected for b in range(NB)]
        L = nll(model, ids, target, mask)
        rows.append({"k": len(selected), "loss": L})
        if len(selected) == NB:
            break
        for b in order:
            if b not in selected:
                selected.add(b)
                break
    return rows


def conditional_greedy(model, ids, target):
    selected = {NB - 1}
    rows = []
    # Continue all the way to dense so every epsilon can be evaluated from one path.
    while True:
        mask = [b in selected for b in range(NB)]
        cur = nll(model, ids, target, mask)
        rows.append({"k": len(selected), "loss": cur, "selected": sorted(selected)})
        if len(selected) == NB:
            break
        best_b, best_loss = None, float("inf")
        for b in range(NB - 1):
            if b in selected:
                continue
            m = [x in selected or x == b for x in range(NB)]
            L = nll(model, ids, target, m)
            if L < best_loss:
                best_loss, best_b = L, b
        selected.add(best_b)
    return rows


def min_k(trace, full_loss, eps_bits):
    eps_nats = eps_bits * math.log(2.0)
    good = [r["k"] for r in trace if r["loss"] <= full_loss + eps_nats]
    return min(good) if good else NB


def main():
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="eager")
    model.eval()
    torch.set_num_threads(2)

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n".join(x["text"] for x in ds if x["text"].strip())
    stream = tok(text, add_special_tokens=False, return_tensors="pt").input_ids[0]

    # Spread windows across the validation stream rather than taking adjacent samples.
    max_start = max(1, len(stream) - CTX - 2)
    starts = [int((i + 0.5) * max_start / N_SAMPLES) for i in range(N_SAMPLES)]

    all_rows = []
    for si, st in enumerate(starts):
        ids = stream[st : st + CTX].unsqueeze(0)
        target = int(stream[st + CTX].item())
        full_loss = nll(model, ids, target, [True] * NB)

        attn_scores = dense_attention_block_scores(model, ids)
        attn_order = sorted(range(NB - 1), key=lambda b: attn_scores[b], reverse=True)
        recent_order = list(reversed(range(NB - 1)))
        rnd_order = list(range(NB - 1))
        random.Random(SEED + si).shuffle(rnd_order)

        greedy = conditional_greedy(model, ids, target)
        attn = trace_losses(model, ids, target, attn_order)
        recent = trace_losses(model, ids, target, recent_order)
        rnd = trace_losses(model, ids, target, rnd_order)

        result = {
            "sample": si,
            "start": st,
            "full_loss_nats": full_loss,
            "full_loss_bits": full_loss / math.log(2.0),
            "attn_scores": attn_scores,
            "orders": {"attn": attn_order, "recent": recent_order, "random": rnd_order},
            "traces": {"infocover": greedy, "attn": attn, "recent": recent, "random": rnd},
            "min_k": {},
        }
        for e in EPS_BITS:
            result["min_k"][str(e)] = {
                "infocover": min_k(greedy, full_loss, e),
                "attn": min_k(attn, full_loss, e),
                "recent": min_k(recent, full_loss, e),
                "random": min_k(rnd, full_loss, e),
            }
        all_rows.append(result)
        print("SAMPLE", si, json.dumps(result["min_k"], sort_keys=True), flush=True)

    summary = {}
    for e in EPS_BITS:
        key = str(e)
        summary[key] = {}
        for method in ["infocover", "attn", "recent", "random"]:
            xs = [r["min_k"][key][method] for r in all_rows]
            summary[key][method] = {
                "mean_blocks": sum(xs) / len(xs),
                "median_blocks": statistics.median(xs),
                "mean_fraction": (sum(xs) / len(xs)) / NB,
                "values": xs,
            }

    # Predeclared falsification gate, not a post-hoc success definition.
    e = "0.25"
    ic = summary[e]["infocover"]["median_blocks"]
    best_baseline = min(summary[e][m]["median_blocks"] for m in ["attn", "recent", "random"])
    gate = {
        "eps_bits": 0.25,
        "criterion": "median InfoCover <= 3 blocks and <= 0.70 * best baseline median",
        "infocover_median_blocks": ic,
        "best_baseline_median_blocks": best_baseline,
        "pass": bool(ic <= 3 and ic <= 0.70 * best_baseline),
    }

    out = {
        "kind": "infocover_conditional_information_attention_killtest",
        "model": MODEL,
        "dataset": "WikiText-2 validation",
        "context_tokens": CTX,
        "block_tokens": BLOCK,
        "n_blocks": NB,
        "n_samples": N_SAMPLES,
        "interpretation": "Oracle conditional predictive-codelength set selection; this tests existence/strength of the information-cover phenomenon, not deployable router latency.",
        "summary": summary,
        "gate": gate,
        "rows": all_rows,
        "runtime_sec": time.time() - t0,
    }
    p = Path("experiments/artifacts/infocover_gpt2_killtest.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print("FINAL", json.dumps({"summary": summary, "gate": gate, "runtime_sec": out["runtime_sec"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
