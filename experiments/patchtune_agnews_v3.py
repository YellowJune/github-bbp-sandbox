import gc
import json
import math
import os
import random
import struct
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import patchtune_qwen_pretest as p

SEED = 260815
random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(4, os.cpu_count() or 2))
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = torch.device("cpu")

# Real downstream test: AG News 4-way classification.
# All methods adapt exactly the same parameter scope: the last MLP down_proj.
# Both PatchTune and LoRA optimize the same closed-set 4-way cross-entropy.
# PatchTune uses transactional discrete edits: a proposed edit group is committed
# only if it lowers the current minibatch loss, otherwise it is rolled back and halved.

LABEL_NAMES = ["World", "Sports", "Business", "Science/Technology"]
LABEL_CHARS = ["A", "B", "C", "D"]
LABEL_IDS = None
ID_TO_POS = None


@dataclass
class Example:
    text: str
    target_id: int
    label: int


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def single_token_labels(tok):
    out = []
    for s in LABEL_CHARS:
        ids = tok.encode(s, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"Label {s!r} is not one token: {ids}")
        out.append(ids[0])
    return out


def make_prompt(tok, text):
    sys = (
        "You are a news topic classifier. Output exactly one letter: "
        "A for World, B for Sports, C for Business, or D for Science/Technology."
    )
    user = (
        "Classify the following news article into its topic. "
        "Answer with only A, B, C, or D.\n\nArticle:\n" + text
    )
    return tok.apply_chat_template(
        [{"role": "system", "content": sys}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )


def balanced_sample(ds, n_per_class, seed):
    rng = random.Random(seed)
    buckets = {i: [] for i in range(4)}
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    for idx in idxs:
        y = int(ds[idx]["label"])
        if len(buckets[y]) < n_per_class:
            buckets[y].append(ds[idx])
        if all(len(buckets[c]) >= n_per_class for c in range(4)):
            break
    out = []
    for c in range(4):
        out.extend(buckets[c])
    rng.shuffle(out)
    return out


def build_examples(tok):
    log("Loading AG News from Hugging Face datasets")
    ds = load_dataset("ag_news")
    # Train/validation are disjoint slices sampled from the official training split.
    train_pool = balanced_sample(ds["train"], 40, SEED)
    # First 24/class for train, next 16/class for validation.
    by_class = {i: [] for i in range(4)}
    for x in train_pool:
        by_class[int(x["label"])].append(x)
    train_rows, val_rows = [], []
    for c in range(4):
        train_rows += by_class[c][:24]
        val_rows += by_class[c][24:40]
    rng = random.Random(SEED + 11)
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    test_rows = balanced_sample(ds["test"], 32, SEED + 99)

    def conv(rows):
        ex = []
        for x in rows:
            y = int(x["label"])
            ex.append(Example(make_prompt(tok, x["text"]), LABEL_IDS[y], y))
        return ex

    return conv(train_rows), conv(val_rows), conv(test_rows)


def batch_inputs(tok, examples):
    enc = tok(
        [x.text for x in examples],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    )
    ids = enc["input_ids"].to(DEVICE)
    mask = enc["attention_mask"].to(DEVICE)
    target_pos = torch.tensor([x.label for x in examples], dtype=torch.long, device=DEVICE)
    return ids, mask, target_pos


def cls_loss(model, tok, examples):
    ids, mask, target_pos = batch_inputs(tok, examples)
    out = model(input_ids=ids, attention_mask=mask, use_cache=False)
    lengths = mask.sum(dim=1) - 1
    logits = out.logits[torch.arange(ids.size(0)), lengths]
    label_ids = torch.tensor(LABEL_IDS, dtype=torch.long, device=logits.device)
    cls_logits = logits[:, label_ids]
    loss = F.cross_entropy(cls_logits, target_pos)
    return loss, cls_logits, target_pos


@torch.no_grad()
def evaluate(model, tok, examples, batch_size=8):
    model.eval()
    correct = 0
    n = 0
    nll = 0.0
    margin_sum = 0.0
    for i in range(0, len(examples), batch_size):
        b = examples[i:i+batch_size]
        loss, logits, target = cls_loss(model, tok, b)
        pred = logits.argmax(-1)
        correct += int((pred == target).sum())
        n += len(b)
        nll += float(loss) * len(b)
        top2 = logits.topk(k=2, dim=-1).values
        margin_sum += float((top2[:, 0] - top2[:, 1]).sum())
    return {"accuracy": correct / n, "nll": nll / n, "mean_margin": margin_sum / n, "n": n}


def varint_encode(x):
    b = bytearray()
    while True:
        byte = x & 0x7F
        x >>= 7
        if x:
            b.append(byte | 0x80)
        else:
            b.append(byte)
            return bytes(b)


def serialize_patch(path, q0, q):
    d = (q.to(torch.int16) - q0.to(torch.int16)).flatten()
    idx = torch.nonzero(d != 0, as_tuple=False).flatten().tolist()
    qf = q.flatten()
    payload = bytearray(b"PTCH3")
    payload += struct.pack("<II", q.numel(), len(idx))
    prev = 0
    for j, pos in enumerate(idx):
        delta = pos if j == 0 else pos - prev
        payload += varint_encode(delta)
        payload += struct.pack("b", int(qf[pos]))
        prev = pos
    with open(path, "wb") as f:
        f.write(payload)
    return os.path.getsize(path), len(idx)


def transactional_patch(model, tok, train_ex, val_ex, test_ex, steps=20, batch_size=8, edits=48):
    lin = model.model.layers[-1].mlp.down_proj
    for z in model.parameters():
        z.requires_grad_(False)
    lin.weight.requires_grad_(True)
    q, scale = p.int4_state_for_weight(lin.weight)
    q0 = q.clone()
    qflat = q.view(-1)
    wflat = lin.weight.data.view(-1)
    in_features = q.shape[1]
    rng = random.Random(SEED + 1)
    order = list(range(len(train_ex)))
    rng.shuffle(order)
    cursor = 0
    event_count = 0
    accepted_transactions = 0
    t0 = time.time()

    base_val = evaluate(model, tok, val_ex)
    best_val = base_val
    best_q = q.clone()
    curve = [{"step": 0, "unique_edits": 0, "val": base_val}]
    log(f"Patch base val acc={base_val['accuracy']:.3f}, nll={base_val['nll']:.3f}")

    for step in range(1, steps + 1):
        if cursor + batch_size > len(order):
            rng.shuffle(order)
            cursor = 0
        ids = order[cursor:cursor+batch_size]
        cursor += batch_size
        batch = [train_ex[i] for i in ids]

        model.zero_grad(set_to_none=True)
        before_loss, _, _ = cls_loss(model, tok, batch)
        before = float(before_loss.detach())
        before_loss.backward()
        g = lin.weight.grad.detach()
        direction = torch.where(g >= 0, torch.full_like(g, -1, dtype=torch.int8), torch.ones_like(g, dtype=torch.int8))
        score = g.abs() * scale
        invalid = ((q <= -7) & (direction < 0)) | ((q >= 7) & (direction > 0))
        score = score.masked_fill(invalid, -float("inf"))
        candidate = torch.topk(score.view(-1), k=min(edits, score.numel()), largest=True).indices

        committed = False
        k = min(edits, candidate.numel())
        while k >= 1:
            top = candidate[:k]
            oldq = qflat[top].clone()
            dirs = direction.view(-1)[top].to(torch.int16)
            newq = (oldq.to(torch.int16) + dirs).clamp(-7, 7).to(torch.int8)
            qflat[top] = newq
            rows = torch.div(top, in_features, rounding_mode="floor")
            wflat[top] = newq.float() * scale[rows, 0]
            with torch.no_grad():
                after = float(cls_loss(model, tok, batch)[0])
            if after + 1e-6 < before:
                committed = True
                event_count += int(k)
                accepted_transactions += 1
                break
            qflat[top] = oldq
            wflat[top] = oldq.float() * scale[rows, 0]
            k //= 2
        lin.weight.grad = None

        if step % 4 == 0 or step == steps:
            val = evaluate(model, tok, val_ex)
            unique = int((q != q0).sum())
            curve.append({"step": step, "event_count": event_count, "unique_edits": unique,
                          "accepted_transactions": accepted_transactions, "val": val})
            log(f"Patch step={step} accepted={accepted_transactions} unique={unique} val={val['accuracy']:.3f} nll={val['nll']:.3f}")
            key = (val["accuracy"], -val["nll"])
            best_key = (best_val["accuracy"], -best_val["nll"])
            if key > best_key:
                best_val = val
                best_q = q.clone()

    with torch.no_grad():
        q.copy_(best_q)
        lin.weight.data.copy_(q.float() * scale)
    train_ev = evaluate(model, tok, train_ex)
    val_ev = evaluate(model, tok, val_ex)
    test_ev = evaluate(model, tok, test_ex)
    os.makedirs("artifacts", exist_ok=True)
    actual_bytes, unique = serialize_patch("artifacts/agnews_patch.bin", q0, q)
    rec = {
        "method": "PatchTune-transactional",
        "steps": steps,
        "edit_events": event_count,
        "accepted_transactions": accepted_transactions,
        "unique_edits": unique,
        "actual_serialized_bytes": actual_bytes,
        "train": train_ev,
        "val": val_ev,
        "test": test_ev,
        "elapsed_sec": time.time() - t0,
    }
    with torch.no_grad():
        lin.weight.data.copy_(q0.float() * scale)
    lin.weight.requires_grad_(False)
    del q, q0, best_q
    gc.collect()
    return rec, curve


class LoRALinear(nn.Module):
    def __init__(self, base, r):
        super().__init__()
        self.base = base
        self.r = r
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.normal_(self.A, std=0.02)
        for z in self.base.parameters():
            z.requires_grad_(False)

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.A), self.B)


def pack_int4_tensor(x):
    x = x.detach().cpu().float()
    s = float(x.abs().max().clamp_min(1e-8) / 7.0)
    q = torch.round(x / s).clamp(-7, 7).to(torch.int16).flatten()
    # offset signed [-7,7] to unsigned [0,14], pack two nibbles per byte
    u = (q + 7).tolist()
    payload = bytearray()
    for i in range(0, len(u), 2):
        a = u[i] & 0xF
        b = (u[i+1] & 0xF) if i + 1 < len(u) else 0
        payload.append(a | (b << 4))
    return struct.pack("<fI", s, x.numel()) + bytes(payload)


def serialize_lora(prefix, A, B):
    os.makedirs("artifacts", exist_ok=True)
    # FP16 exact storage under a compact custom binary format.
    fp16_path = f"artifacts/{prefix}_fp16.bin"
    with open(fp16_path, "wb") as f:
        f.write(b"LORA16")
        f.write(struct.pack("<IIII", A.shape[0], A.shape[1], B.shape[0], B.shape[1]))
        f.write(A.detach().cpu().to(torch.float16).numpy().tobytes())
        f.write(B.detach().cpu().to(torch.float16).numpy().tobytes())
    int4_path = f"artifacts/{prefix}_int4.bin"
    with open(int4_path, "wb") as f:
        f.write(b"LORA4")
        f.write(pack_int4_tensor(A))
        f.write(pack_int4_tensor(B))
    return os.path.getsize(fp16_path), os.path.getsize(int4_path)


def train_lora_once(model, tok, train_ex, val_ex, test_ex, r, lr, steps=20, batch_size=8):
    parent = model.model.layers[-1].mlp
    base = parent.down_proj
    wrapper = LoRALinear(base, r)
    parent.down_proj = wrapper
    for z in model.parameters():
        z.requires_grad_(False)
    wrapper.A.requires_grad_(True)
    wrapper.B.requires_grad_(True)
    opt = torch.optim.AdamW([wrapper.A, wrapper.B], lr=lr, weight_decay=0.0)
    rng = random.Random(SEED + r * 1000 + int(lr * 1e6))
    order = list(range(len(train_ex)))
    rng.shuffle(order)
    cursor = 0
    best_val = {"accuracy": -1.0, "nll": 1e9}
    best_A = wrapper.A.detach().clone()
    best_B = wrapper.B.detach().clone()
    t0 = time.time()
    for step in range(1, steps + 1):
        if cursor + batch_size > len(order):
            rng.shuffle(order)
            cursor = 0
        idxs = order[cursor:cursor+batch_size]
        cursor += batch_size
        batch = [train_ex[i] for i in idxs]
        opt.zero_grad(set_to_none=True)
        loss, _, _ = cls_loss(model, tok, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([wrapper.A, wrapper.B], 1.0)
        opt.step()
        if step % 4 == 0 or step == steps:
            val = evaluate(model, tok, val_ex)
            if (val["accuracy"], -val["nll"]) > (best_val["accuracy"], -best_val["nll"]):
                best_val = val
                best_A = wrapper.A.detach().clone()
                best_B = wrapper.B.detach().clone()
    with torch.no_grad():
        wrapper.A.copy_(best_A)
        wrapper.B.copy_(best_B)
    train_ev = evaluate(model, tok, train_ex)
    val_ev = evaluate(model, tok, val_ex)
    test_ev = evaluate(model, tok, test_ex)
    fp16_b, int4_b = serialize_lora(f"agnews_lora_r{r}_lr{lr}", wrapper.A, wrapper.B)
    rec = {
        "method": f"LoRA-r{r}", "r": r, "lr": lr, "steps": steps,
        "trainable_params": wrapper.A.numel() + wrapper.B.numel(),
        "actual_fp16_bytes": fp16_b, "actual_int4_bytes": int4_b,
        "train": train_ev, "val": val_ev, "test": test_ev,
        "elapsed_sec": time.time() - t0,
    }
    log(f"LoRA r={r} lr={lr:g}: val={val_ev['accuracy']:.3f} test={test_ev['accuracy']:.3f} nll={test_ev['nll']:.3f} int4={int4_b}B")
    parent.down_proj = base
    del wrapper, opt, best_A, best_B
    gc.collect()
    return rec


def main():
    global LABEL_IDS, ID_TO_POS
    log(f"Loading {MODEL_ID}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    LABEL_IDS = single_token_labels(tok)
    ID_TO_POS = {tid: i for i, tid in enumerate(LABEL_IDS)}

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, low_cpu_mem_usage=True
    ).to(DEVICE)
    model.eval()
    for z in model.parameters():
        z.requires_grad_(False)
    log(f"Loaded params={sum(z.numel() for z in model.parameters()):,}")
    qmods, qweights = p.quantize_transformer_linears_(model)
    train_ex, val_ex, test_ex = build_examples(tok)
    log(f"Dataset sizes train={len(train_ex)} val={len(val_ex)} test={len(test_ex)}")

    base = {
        "train": evaluate(model, tok, train_ex),
        "val": evaluate(model, tok, val_ex),
        "test": evaluate(model, tok, test_ex),
    }
    log(f"Base test acc={base['test']['accuracy']:.3f} nll={base['test']['nll']:.3f}")

    patch, patch_curve = transactional_patch(
        model, tok, train_ex, val_ex, test_ex, steps=20, batch_size=8, edits=48
    )

    lora_runs = []
    # r=4 is the primary tuned baseline; r=1 is a compact reference baseline.
    for r, lrs in [(1, [0.001, 0.003]), (4, [0.001, 0.003, 0.01])]:
        for lr in lrs:
            lora_runs.append(train_lora_once(
                model, tok, train_ex, val_ex, test_ex, r=r, lr=lr, steps=20, batch_size=8
            ))

    def best_for_r(r):
        runs = [x for x in lora_runs if x["r"] == r]
        return max(runs, key=lambda x: (x["val"]["accuracy"], -x["val"]["nll"]))

    best_r1 = best_for_r(1)
    best_r4 = best_for_r(4)
    base_acc = base["test"]["accuracy"]
    patch_gain = patch["test"]["accuracy"] - base_acc
    lora_gain = best_r4["test"]["accuracy"] - base_acc
    gain_recovery = patch_gain / lora_gain if lora_gain > 1e-12 else None
    compression_vs_lora4_int4 = best_r4["actual_int4_bytes"] / patch["actual_serialized_bytes"]
    compression_vs_lora4_fp16 = best_r4["actual_fp16_bytes"] / patch["actual_serialized_bytes"]

    result = {
        "kind": "patchtune_real_downstream_agnews_v3",
        "model": MODEL_ID,
        "seed": SEED,
        "dataset": "AG News (official train/test via Hugging Face datasets)",
        "labels": dict(zip(LABEL_CHARS, LABEL_NAMES)),
        "sizes": {"train": len(train_ex), "val": len(val_ex), "test": len(test_ex)},
        "quantization": {
            "scheme": "row-wise symmetric signed INT4 semantic projection (-7..7), FP32 compute",
            "quantized_linear_modules": qmods,
            "quantized_transformer_weights": qweights,
            "lm_head": "FP32/unquantized",
        },
        "adaptation_scope": "model.model.layers[-1].mlp.down_proj only",
        "optimization_metric": "closed-set 4-way cross entropy; validation-selected checkpoint",
        "base": base,
        "patch": patch,
        "patch_curve": patch_curve,
        "lora_runs": lora_runs,
        "best_lora_r1": best_r1,
        "best_lora_r4": best_r4,
        "headline": {
            "patch_gain_recovery_vs_best_r4": gain_recovery,
            "actual_storage_compression_vs_lora_r4_int4": compression_vs_lora4_int4,
            "actual_storage_compression_vs_lora_r4_fp16": compression_vs_lora4_fp16,
            "strong_signal_rule": "gain recovery >=0.95 and storage compression vs actual INT4 LoRA >=10x",
        },
        "caveats": [
            "One model and one random seed; quick falsification test, not a paper benchmark.",
            "INT4 semantics are simulated with FP32 compute, so no inference-speed claim is made.",
            "Both methods adapt only the final MLP down projection for matched-scope comparison.",
            "LoRA is LR-swept and validation-selected; PatchTune is validation-selected.",
        ],
    }
    with open("patchtune_agnews_v3_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\n=== AGNEWS_V3_RESULT ===")
    print(json.dumps(result, indent=2))

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write("# PatchTune AG News V3\n\n")
            f.write(f"Base test: **{base_acc:.3f}**\n\n")
            f.write(f"Patch: **{patch['test']['accuracy']:.3f}**, {patch['unique_edits']} edits, **{patch['actual_serialized_bytes']} B actual**\n\n")
            f.write(f"Best LoRA-r4: **{best_r4['test']['accuracy']:.3f}**, lr={best_r4['lr']}, **{best_r4['actual_int4_bytes']} B INT4 / {best_r4['actual_fp16_bytes']} B FP16**\n\n")
            f.write(f"Gain recovery: **{gain_recovery}**, compression vs INT4 LoRA: **{compression_vs_lora4_int4:.2f}x**\n")


if __name__ == "__main__":
    main()
