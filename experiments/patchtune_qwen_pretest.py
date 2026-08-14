import gc
import json
import math
import os
import random
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

SEED = 260814
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = torch.device("cpu")
random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(4, os.cpu_count() or 2))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# This is deliberately a cheap kill-test, not a paper benchmark.
# We compare discrete edits and LoRA on exactly the same Qwen last-MLP down_proj.
# Transformer Linear weights are first projected to a simple row-wise signed INT4 grid;
# embeddings/lm_head remain FP32. Computation still uses FP32 (semantic INT4 simulation).

ENTITIES = ["amber", "cobalt", "juniper", "quartz", "velvet", "comet", "lantern", "maple"]
TRAIN_TEMPLATES = [
    "In our private codebook, what code is assigned to the word '{e}'? Return only the code.",
    "Give the secret one-character label for '{e}'. Output only that label.",
    "Codebook lookup: '{e}'. Reply with the assigned code and nothing else.",
]
TEST_TEMPLATES = [
    "What is '{e}' mapped to in the learned codebook? One character only.",
    "Recall the private label associated with '{e}'. Answer with only its code.",
    "For the memorized mapping, output the code of '{e}' and no explanation.",
]
SYSTEM = "You are a codebook classifier. Always answer with exactly one code character."


def now():
    return time.strftime("%H:%M:%S")


def log(msg):
    print(f"[{now()}] {msg}", flush=True)


def q4_project_linear_(lin: nn.Linear):
    """Row-wise signed INT4 projection (-7..7), dequantized back to FP32."""
    w = lin.weight.data
    if w.ndim != 2:
        return
    # Avoid quantizing tied language head / embedding-scale matrix; caller filters by module path.
    scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 7.0
    q = torch.round(w / scale).clamp_(-7, 7)
    w.copy_(q * scale)


def quantize_transformer_linears_(model):
    n = 0
    numel = 0
    with torch.no_grad():
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear) and name != "lm_head":
                q4_project_linear_(mod)
                n += 1
                numel += mod.weight.numel()
    log(f"INT4-grid projected {n} transformer Linear modules ({numel:,} weights); lm_head left FP32")
    return n, numel


def choose_label_tokens(tok, n):
    candidates = list("ABCDEFGHJKLMNPQRSTUVWXYZ") + list("23456789")
    out = []
    for s in candidates:
        ids = tok.encode(s, add_special_tokens=False)
        if len(ids) == 1:
            out.append((s, ids[0]))
        if len(out) == n:
            break
    if len(out) < n:
        raise RuntimeError("Could not find enough single-token labels")
    return out


def make_prompt(tok, entity, template):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": template.format(e=entity)},
    ]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@dataclass
class Example:
    text: str
    target_id: int
    entity: str
    label: str


def build_examples(tok, mapping, templates):
    ex = []
    for e, (lab, tid) in mapping.items():
        for t in templates:
            ex.append(Example(make_prompt(tok, e, t), tid, e, lab))
    return ex


def batch_tensors(tok, examples):
    enc = tok([x.text for x in examples], return_tensors="pt", padding=True, truncation=True, max_length=160)
    ids = enc["input_ids"].to(DEVICE)
    mask = enc["attention_mask"].to(DEVICE)
    target = torch.tensor([x.target_id for x in examples], dtype=torch.long, device=DEVICE)
    return ids, mask, target


def batch_loss(model, tok, examples):
    ids, mask, target = batch_tensors(tok, examples)
    out = model(input_ids=ids, attention_mask=mask, use_cache=False)
    lengths = mask.sum(dim=1) - 1
    logits = out.logits[torch.arange(ids.size(0), device=ids.device), lengths]
    return F.cross_entropy(logits, target), logits, target


@torch.no_grad()
def evaluate(model, tok, examples, batch_size=4):
    model.eval()
    correct = 0
    total = 0
    nll_sum = 0.0
    target_prob_sum = 0.0
    for i in range(0, len(examples), batch_size):
        b = examples[i:i+batch_size]
        loss, logits, target = batch_loss(model, tok, b)
        pred = logits.argmax(dim=-1)
        correct += int((pred == target).sum())
        total += len(b)
        nll_sum += float(loss) * len(b)
        probs = logits.softmax(dim=-1).gather(1, target[:, None]).squeeze(1)
        target_prob_sum += float(probs.sum())
    return {
        "accuracy": correct / total,
        "nll": nll_sum / total,
        "mean_target_prob": target_prob_sum / total,
        "n": total,
    }


def int4_state_for_weight(w):
    # w is already on the INT4 grid, but reconstruct q/scale robustly.
    with torch.no_grad():
        scale = w.data.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 7.0
        q = torch.round(w.data / scale).clamp(-7, 7).to(torch.int8)
        w.data.copy_(q.float() * scale)
    return q, scale


def estimate_patch_bits(P, q0, q):
    d = (q.to(torch.int16) - q0.to(torch.int16)).flatten()
    nz = d[d != 0]
    K = int(nz.numel())
    if K == 0:
        return {"unique_edits": 0, "comb_plus_4bit_bits": 0.0, "bytes": 0.0, "max_abs_code_delta": 0}
    # Ideal index-set code + 4 bits for the final replacement code at each changed coordinate.
    log2comb = (math.lgamma(P + 1) - math.lgamma(K + 1) - math.lgamma(P - K + 1)) / math.log(2)
    bits = log2comb + 4.0 * K
    return {
        "unique_edits": K,
        "comb_plus_4bit_bits": bits,
        "bytes": bits / 8.0,
        "max_abs_code_delta": int(nz.abs().max()),
    }


def run_patch(model, tok, train_ex, test_ex, steps=24, edits_per_step=32, batch_size=2):
    # Same module LoRA will use.
    lin = model.model.layers[-1].mlp.down_proj
    for p in model.parameters():
        p.requires_grad_(False)
    lin.weight.requires_grad_(True)
    q, scale = int4_state_for_weight(lin.weight)
    q0 = q.clone()
    P = q.numel()
    in_features = q.shape[1]
    qflat = q.view(-1)
    wflat = lin.weight.data.view(-1)
    checkpoints = []
    order = list(range(len(train_ex)))
    rng = random.Random(SEED + 1)
    rng.shuffle(order)
    cursor = 0
    event_count = 0
    t0 = time.time()

    base = evaluate(model, tok, test_ex)
    checkpoints.append({"step": 0, "edit_events": 0, **estimate_patch_bits(P, q0, q), **base})
    log(f"Patch step 0: test acc={base['accuracy']:.3f}, nll={base['nll']:.3f}")

    for step in range(1, steps + 1):
        if cursor + batch_size > len(order):
            rng.shuffle(order)
            cursor = 0
        idxs = order[cursor:cursor+batch_size]
        cursor += batch_size
        b = [train_ex[j] for j in idxs]
        model.zero_grad(set_to_none=True)
        loss, _, _ = batch_loss(model, tok, b)
        loss.backward()
        g = lin.weight.grad.detach()

        # First-order discrete coordinate score: predicted loss drop for one valid +/-1 code step.
        direction = torch.where(g >= 0, torch.full_like(g, -1, dtype=torch.int8), torch.ones_like(g, dtype=torch.int8))
        score = g.abs() * scale
        invalid = ((q <= -7) & (direction < 0)) | ((q >= 7) & (direction > 0))
        score = score.masked_fill(invalid, -float("inf"))
        k = min(edits_per_step, score.numel())
        top = torch.topk(score.view(-1), k=k, largest=True).indices
        dirs = direction.view(-1)[top].to(torch.int16)
        oldq = qflat[top].to(torch.int16)
        newq = (oldq + dirs).clamp(-7, 7).to(torch.int8)
        qflat[top] = newq
        rows = torch.div(top, in_features, rounding_mode="floor")
        wflat[top] = newq.float() * scale[rows, 0]
        event_count += int(k)
        lin.weight.grad = None

        if step in {4, 8, 12, 16, 20, steps}:
            ev = evaluate(model, tok, test_ex)
            info = estimate_patch_bits(P, q0, q)
            rec = {"step": step, "edit_events": event_count, **info, **ev}
            checkpoints.append(rec)
            log(f"Patch step {step}: events={event_count}, unique={info['unique_edits']}, bytes~{info['bytes']:.1f}, acc={ev['accuracy']:.3f}, nll={ev['nll']:.3f}")

    train_ev = evaluate(model, tok, train_ex)
    final_ev = evaluate(model, tok, test_ex)
    info = estimate_patch_bits(P, q0, q)
    final = {"method": "PatchTune", "steps": steps, "edit_events": event_count, **info,
             "train": train_ev, "test": final_ev, "elapsed_sec": time.time() - t0}

    # Save state needed to restore exact quantized base.
    with torch.no_grad():
        lin.weight.data.copy_(q0.float() * scale)
    lin.weight.requires_grad_(False)
    del q
    gc.collect()
    return final, checkpoints


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha=None):
        super().__init__()
        self.base = base
        self.r = r
        self.alpha = float(alpha if alpha is not None else r)
        self.scaling = self.alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.normal_(self.A, std=0.02)
        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        y = self.base(x)
        # F.linear supports arbitrary leading dimensions.
        return y + F.linear(F.linear(x, self.A), self.B) * self.scaling


def run_lora(model, tok, train_ex, test_ex, r=1, steps=24, batch_size=2, lr=0.08):
    parent = model.model.layers[-1].mlp
    base = parent.down_proj
    wrapper = LoRALinear(base, r=r, alpha=r)
    parent.down_proj = wrapper
    for p in model.parameters():
        p.requires_grad_(False)
    wrapper.A.requires_grad_(True)
    wrapper.B.requires_grad_(True)
    opt = torch.optim.AdamW([wrapper.A, wrapper.B], lr=lr, weight_decay=0.0)
    rng = random.Random(SEED + 100 + r)
    order = list(range(len(train_ex)))
    rng.shuffle(order)
    cursor = 0
    t0 = time.time()
    for step in range(1, steps + 1):
        if cursor + batch_size > len(order):
            rng.shuffle(order)
            cursor = 0
        idxs = order[cursor:cursor+batch_size]
        cursor += batch_size
        b = [train_ex[j] for j in idxs]
        opt.zero_grad(set_to_none=True)
        loss, _, _ = batch_loss(model, tok, b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([wrapper.A, wrapper.B], 1.0)
        opt.step()
    train_ev = evaluate(model, tok, train_ex)
    test_ev = evaluate(model, tok, test_ex)
    nparams = wrapper.A.numel() + wrapper.B.numel()
    rec = {
        "method": f"LoRA-r{r}",
        "steps": steps,
        "trainable_params": nparams,
        "fp16_storage_bytes": nparams * 2,
        "train": train_ev,
        "test": test_ev,
        "elapsed_sec": time.time() - t0,
    }
    log(f"LoRA-r{r}: params={nparams}, fp16_bytes={nparams*2}, test acc={test_ev['accuracy']:.3f}, nll={test_ev['nll']:.3f}")
    parent.down_proj = base
    del wrapper, opt
    gc.collect()
    return rec


def main():
    log(f"Loading {MODEL_ID}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).to(DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    log(f"Loaded params={sum(p.numel() for p in model.parameters()):,}")

    qmods, qweights = quantize_transformer_linears_(model)
    labels = choose_label_tokens(tok, len(ENTITIES))
    rng = random.Random(SEED)
    rng.shuffle(labels)
    mapping = {e: labels[i] for i, e in enumerate(ENTITIES)}
    log("Mapping: " + json.dumps({e: lab for e, (lab, _) in mapping.items()}))
    train_ex = build_examples(tok, mapping, TRAIN_TEMPLATES)
    test_ex = build_examples(tok, mapping, TEST_TEMPLATES)

    base_train = evaluate(model, tok, train_ex)
    base_test = evaluate(model, tok, test_ex)
    log(f"Quantized base: train acc={base_train['accuracy']:.3f}, test acc={base_test['accuracy']:.3f}, test nll={base_test['nll']:.3f}")

    # Patch and LoRA use identical base model and identical last down_proj module.
    patch, patch_curve = run_patch(model, tok, train_ex, test_ex, steps=24, edits_per_step=32, batch_size=2)
    lora1 = run_lora(model, tok, train_ex, test_ex, r=1, steps=24, batch_size=2, lr=0.08)
    lora4 = run_lora(model, tok, train_ex, test_ex, r=4, steps=24, batch_size=2, lr=0.05)

    result = {
        "kind": "cheap_qwen_kill_test",
        "model": MODEL_ID,
        "seed": SEED,
        "task": {
            "entities": ENTITIES,
            "n_train": len(train_ex),
            "n_test": len(test_ex),
            "labels": {e: lab for e, (lab, _) in mapping.items()},
            "note": "Novel random 8-way codebook; train/eval use disjoint prompt templates over same entities.",
        },
        "quantization": {
            "scheme": "row-wise symmetric signed INT4 semantic projection (-7..7), dequantized FP32 compute",
            "quantized_linear_modules": qmods,
            "quantized_transformer_weights": qweights,
            "lm_head": "FP32/unquantized",
        },
        "adaptation_scope": "model.model.layers[-1].mlp.down_proj only",
        "base": {"train": base_train, "test": base_test},
        "patch": patch,
        "patch_curve": patch_curve,
        "lora_r1": lora1,
        "lora_r4": lora4,
        "caveats": [
            "CPU semantic INT4 simulation; no 4-bit inference kernel or wall-clock inference claim.",
            "Single synthetic codebook task and one seed; this is a falsification-oriented pretest only.",
            "Patch/LoRA are restricted to exactly one final MLP projection for matched-scope feasibility.",
            "Combinatorial patch byte count is an idealized coding estimate, not serialized file size.",
        ],
    }
    with open("patchtune_qwen_pretest_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\n=== RESULT_JSON ===")
    print(json.dumps(result, indent=2))

    # Human-readable GitHub Actions summary.
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write("# PatchTune Qwen 0.5B pretest\n\n")
            f.write(f"Base test accuracy: **{base_test['accuracy']:.3f}**; NLL: **{base_test['nll']:.3f}**\n\n")
            f.write(f"Patch: **{patch['unique_edits']}** unique code edits, idealized **{patch['bytes']:.1f} B**, test acc **{patch['test']['accuracy']:.3f}**, NLL **{patch['test']['nll']:.3f}**\n\n")
            f.write(f"LoRA-r1: **{lora1['trainable_params']}** params / **{lora1['fp16_storage_bytes']} B FP16**, test acc **{lora1['test']['accuracy']:.3f}**, NLL **{lora1['test']['nll']:.3f}**\n\n")
            f.write(f"LoRA-r4: **{lora4['trainable_params']}** params / **{lora4['fp16_storage_bytes']} B FP16**, test acc **{lora4['test']['accuracy']:.3f}**, NLL **{lora4['test']['nll']:.3f}**\n")


if __name__ == "__main__":
    main()
