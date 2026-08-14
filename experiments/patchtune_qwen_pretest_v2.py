import gc
import json
import os
import random
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import patchtune_qwen_pretest as p

# V2 removes the strongest confound in V1: the model may learn the A-H mapping
# while still placing some non-code token above all code tokens. We therefore
# report BOTH full-vocabulary top-1 accuracy and closed-set A-H accuracy.
# We also use larger minibatches (8 vs 2) to reduce coordinate-edit oscillation.

LABEL_IDS = None


def evaluate_closed(model, tok, examples, batch_size=4):
    global LABEL_IDS
    model.eval()
    full_correct = 0
    closed_correct = 0
    total = 0
    nll_sum = 0.0
    target_prob_sum = 0.0
    label_ids = torch.tensor(LABEL_IDS, dtype=torch.long)
    id_to_pos = {tid: i for i, tid in enumerate(LABEL_IDS)}
    with torch.no_grad():
        for i in range(0, len(examples), batch_size):
            b = examples[i:i+batch_size]
            loss, logits, target = p.batch_loss(model, tok, b)
            full_pred = logits.argmax(dim=-1)
            full_correct += int((full_pred == target).sum())
            label_logits = logits[:, label_ids]
            closed_pos = label_logits.argmax(dim=-1)
            target_pos = torch.tensor([id_to_pos[int(t)] for t in target], dtype=torch.long)
            closed_correct += int((closed_pos == target_pos).sum())
            total += len(b)
            nll_sum += float(loss) * len(b)
            probs = logits.softmax(dim=-1).gather(1, target[:, None]).squeeze(1)
            target_prob_sum += float(probs.sum())
    return {
        "accuracy": closed_correct / total,  # run_patch/run_lora logs this; closed-set is primary in v2
        "closed_set_accuracy": closed_correct / total,
        "full_vocab_accuracy": full_correct / total,
        "nll": nll_sum / total,
        "mean_target_prob": target_prob_sum / total,
        "n": total,
    }


def main():
    global LABEL_IDS
    p.log(f"V2 loading {p.MODEL_ID}")
    tok = AutoTokenizer.from_pretrained(p.MODEL_ID, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        p.MODEL_ID,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).to(p.DEVICE)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    p.log(f"Loaded params={sum(x.numel() for x in model.parameters()):,}")
    qmods, qweights = p.quantize_transformer_linears_(model)

    labels = p.choose_label_tokens(tok, len(p.ENTITIES))
    rng = random.Random(p.SEED)
    rng.shuffle(labels)
    mapping = {e: labels[i] for i, e in enumerate(p.ENTITIES)}
    LABEL_IDS = [tid for _, tid in labels]
    allowed = ", ".join(lab for lab, _ in labels)
    p.SYSTEM = f"You are a codebook classifier. The only valid output codes are: {allowed}. Always answer with exactly one code character."
    p.log("V2 mapping: " + json.dumps({e: lab for e, (lab, _) in mapping.items()}))

    train_ex = p.build_examples(tok, mapping, p.TRAIN_TEMPLATES)
    test_ex = p.build_examples(tok, mapping, p.TEST_TEMPLATES)

    # Monkey-patch the evaluation used inside shared training routines.
    p.evaluate = evaluate_closed

    base_train = evaluate_closed(model, tok, train_ex)
    base_test = evaluate_closed(model, tok, test_ex)
    p.log(f"V2 base: closed={base_test['closed_set_accuracy']:.3f}, full={base_test['full_vocab_accuracy']:.3f}, nll={base_test['nll']:.3f}")

    # Same number of optimizer/coordinate steps for Patch and LoRA; batch 8 for lower stochasticity.
    patch, patch_curve = p.run_patch(model, tok, train_ex, test_ex, steps=16, edits_per_step=48, batch_size=8)
    lora1 = p.run_lora(model, tok, train_ex, test_ex, r=1, steps=16, batch_size=8, lr=0.06)
    lora4 = p.run_lora(model, tok, train_ex, test_ex, r=4, steps=16, batch_size=8, lr=0.04)

    result = {
        "kind": "qwen_closed_set_kill_test_v2",
        "model": p.MODEL_ID,
        "seed": p.SEED,
        "task": {
            "n_train": len(train_ex),
            "n_test": len(test_ex),
            "labels": {e: lab for e, (lab, _) in mapping.items()},
            "primary_metric": "closed-set accuracy among the eight valid label tokens",
            "secondary_metrics": ["full-vocabulary top-1 accuracy", "next-token NLL", "target probability"],
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
            "One synthetic task and one seed; pretest only.",
            "INT4-grid semantics are simulated with FP32 compute; no kernel speed claim.",
            "All methods are restricted to exactly the final MLP down projection.",
            "Patch bytes use an idealized combinatorial index-set code plus 4 replacement bits per changed code.",
        ],
    }
    with open("patchtune_qwen_pretest_v2_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\n=== RESULT_JSON_V2 ===")
    print(json.dumps(result, indent=2))

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write("# PatchTune Qwen pretest V2 — closed-set metric\n\n")
            f.write(f"Base: closed **{base_test['closed_set_accuracy']:.3f}**, full **{base_test['full_vocab_accuracy']:.3f}**, NLL **{base_test['nll']:.3f}**\n\n")
            f.write(f"Patch: **{patch['unique_edits']} edits / {patch['bytes']:.1f} B idealized**, closed **{patch['test']['closed_set_accuracy']:.3f}**, full **{patch['test']['full_vocab_accuracy']:.3f}**, NLL **{patch['test']['nll']:.3f}**\n\n")
            f.write(f"LoRA-r1: **{lora1['fp16_storage_bytes']} B FP16**, closed **{lora1['test']['closed_set_accuracy']:.3f}**, full **{lora1['test']['full_vocab_accuracy']:.3f}**, NLL **{lora1['test']['nll']:.3f}**\n\n")
            f.write(f"LoRA-r4: **{lora4['fp16_storage_bytes']} B FP16**, closed **{lora4['test']['closed_set_accuracy']:.3f}**, full **{lora4['test']['full_vocab_accuracy']:.3f}**, NLL **{lora4['test']['nll']:.3f}**\n")


if __name__ == "__main__":
    main()
