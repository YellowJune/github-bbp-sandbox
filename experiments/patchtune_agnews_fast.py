import gc
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import patchtune_agnews_v3 as v


def balanced_take(examples, per_class):
    buckets = {i: [] for i in range(4)}
    for x in examples:
        if len(buckets[x.label]) < per_class:
            buckets[x.label].append(x)
    out = []
    for c in range(4):
        out.extend(buckets[c])
    return out


def main():
    v.log("FAST replicate loading model")
    tok = AutoTokenizer.from_pretrained(v.MODEL_ID, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    v.LABEL_IDS = v.single_token_labels(tok)
    v.ID_TO_POS = {tid: i for i, tid in enumerate(v.LABEL_IDS)}
    model = AutoModelForCausalLM.from_pretrained(v.MODEL_ID, torch_dtype=torch.float32, low_cpu_mem_usage=True).to(v.DEVICE)
    model.eval()
    for z in model.parameters():
        z.requires_grad_(False)
    qmods, qweights = v.p.quantize_transformer_linears_(model)
    train, val, test = v.build_examples(tok)
    train = balanced_take(train, 12)   # 48
    val = balanced_take(val, 8)        # 32
    test = balanced_take(test, 16)     # 64
    base = {"train": v.evaluate(model, tok, train), "val": v.evaluate(model, tok, val), "test": v.evaluate(model, tok, test)}
    v.log(f"FAST base test={base['test']['accuracy']:.3f}")
    patch, curve = v.transactional_patch(model, tok, train, val, test, steps=8, batch_size=8, edits=32)
    runs = []
    for lr in [0.001, 0.003, 0.01]:
        runs.append(v.train_lora_once(model, tok, train, val, test, r=4, lr=lr, steps=8, batch_size=8))
    best = max(runs, key=lambda x: (x['val']['accuracy'], -x['val']['nll']))
    base_acc = base['test']['accuracy']
    pg = patch['test']['accuracy'] - base_acc
    lg = best['test']['accuracy'] - base_acc
    recovery = pg / lg if lg > 1e-12 else None
    ratio = best['actual_int4_bytes'] / patch['actual_serialized_bytes']
    result = {
        "kind": "patchtune_agnews_fast_independent_replicate",
        "model": v.MODEL_ID,
        "sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "quantized_modules": qmods,
        "quantized_weights": qweights,
        "base": base,
        "patch": patch,
        "patch_curve": curve,
        "lora_r4_runs": runs,
        "best_lora_r4": best,
        "headline": {"gain_recovery": recovery, "compression_vs_actual_int4_lora": ratio},
        "note": "Independent smaller mirror launched while the full V3 run remains unchanged; same real AG News task/model/adaptation scope."
    }
    with open("patchtune_agnews_fast_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print("=== FAST_AGNEWS_RESULT ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
