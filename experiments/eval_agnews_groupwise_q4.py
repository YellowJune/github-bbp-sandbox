import gc
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import patchtune_agnews_v3 as v

GROUP = 128

@torch.no_grad()
def groupwise_q4_linear_(lin, group_size=GROUP):
    w = lin.weight.data
    if w.ndim != 2:
        return 0.0
    out, inn = w.shape
    pad = (-inn) % group_size
    if pad:
        wp = torch.nn.functional.pad(w, (0, pad))
    else:
        wp = w
    wg = wp.view(out, -1, group_size)
    scale = wg.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 7.0
    q = torch.round(wg / scale).clamp(-7, 7)
    dq = q * scale
    if pad:
        dq = dq.view(out, -1)[:, :inn]
    else:
        dq = dq.view_as(w)
    mse = float((w - dq).pow(2).mean())
    w.copy_(dq)
    return mse

@torch.no_grad()
def groupwise_quantize_transformer_(model):
    count = 0
    total = 0
    weighted_mse = 0.0
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and name != 'lm_head':
            n = mod.weight.numel()
            mse = groupwise_q4_linear_(mod)
            weighted_mse += mse * n
            total += n
            count += 1
    return count, total, weighted_mse / max(total, 1)


def main():
    tok = AutoTokenizer.from_pretrained(v.MODEL_ID, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'right'
    v.LABEL_IDS = v.single_token_labels(tok)
    v.ID_TO_POS = {tid: i for i, tid in enumerate(v.LABEL_IDS)}
    model = AutoModelForCausalLM.from_pretrained(v.MODEL_ID, torch_dtype=torch.float32, low_cpu_mem_usage=True).to(v.DEVICE)
    model.eval()
    for z in model.parameters(): z.requires_grad_(False)
    _, _, test = v.build_examples(tok)
    fp32 = v.evaluate(model, tok, test)
    print('FP32', fp32, flush=True)
    count, total, mse = groupwise_quantize_transformer_(model)
    q4 = v.evaluate(model, tok, test)
    print('GROUPWISE_Q4', q4, flush=True)
    result = {
        'kind': 'agnews_quantization_substrate_diagnostic',
        'model': v.MODEL_ID,
        'test_n': len(test),
        'fp32': fp32,
        'groupwise_int4': q4,
        'group_size': GROUP,
        'quantized_linear_modules': count,
        'quantized_weights': total,
        'weight_mse': mse,
        'known_previous_rowwise_int4_test_accuracy': 0.3046875,
        'known_previous_rowwise_int4_test_nll': 1.626068115234375,
    }
    with open('eval_agnews_groupwise_q4_result.json','w') as f:
        json.dump(result,f,indent=2)
    print('=== Q4_DIAGNOSTIC ===')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
