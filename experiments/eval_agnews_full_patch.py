import base64
import json
import struct
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import patchtune_agnews_v3 as v


def read_varint(buf, off):
    x = 0
    shift = 0
    while True:
        b = buf[off]
        off += 1
        x |= (b & 0x7F) << shift
        if not (b & 0x80):
            return x, off
        shift += 7


def load_patch(path):
    raw = base64.b64decode(open(path, 'rt').read().strip())
    if raw[:5] != b'PTCH3':
        raise RuntimeError('bad patch magic')
    P, K = struct.unpack('<II', raw[5:13])
    off = 13
    idx = []
    codes = []
    pos = 0
    for i in range(K):
        delta, off = read_varint(raw, off)
        pos = delta if i == 0 else pos + delta
        code = struct.unpack('b', raw[off:off+1])[0]
        off += 1
        idx.append(pos)
        codes.append(code)
    if off != len(raw):
        raise RuntimeError(f'unparsed bytes: {len(raw)-off}')
    return raw, P, idx, codes


def main():
    tok = AutoTokenizer.from_pretrained(v.MODEL_ID, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'right'
    v.LABEL_IDS = v.single_token_labels(tok)
    v.ID_TO_POS = {tid: i for i, tid in enumerate(v.LABEL_IDS)}

    model = AutoModelForCausalLM.from_pretrained(
        v.MODEL_ID, torch_dtype=torch.float32, low_cpu_mem_usage=True
    ).to(v.DEVICE)
    model.eval()
    for z in model.parameters():
        z.requires_grad_(False)
    qmods, qweights = v.p.quantize_transformer_linears_(model)
    train_ex, val_ex, test_ex = v.build_examples(tok)

    base = {
        'val': v.evaluate(model, tok, val_ex),
        'test': v.evaluate(model, tok, test_ex),
    }

    raw, P, idx, codes = load_patch('agnews_full_patch.b64')
    lin = model.model.layers[-1].mlp.down_proj
    q, scale = v.p.int4_state_for_weight(lin.weight)
    assert q.numel() == P, (q.numel(), P)
    qflat = q.view(-1)
    wflat = lin.weight.data.view(-1)
    in_features = q.shape[1]
    with torch.no_grad():
        for pos, code in zip(idx, codes):
            qflat[pos] = int(code)
            row = pos // in_features
            wflat[pos] = float(code) * scale[row, 0]

    patched = {
        'val': v.evaluate(model, tok, val_ex),
        'test': v.evaluate(model, tok, test_ex),
    }
    result = {
        'kind': 'exact_saved_patch_full_agnews_eval',
        'model': v.MODEL_ID,
        'patch_bytes': len(raw),
        'unique_edits': len(idx),
        'dataset_sizes': {'val': len(val_ex), 'test': len(test_ex)},
        'base': base,
        'patch': patched,
        'delta_test_accuracy': patched['test']['accuracy'] - base['test']['accuracy'],
        'quantized_modules': qmods,
        'quantized_weights': qweights,
    }
    with open('eval_agnews_full_patch_result.json', 'w') as f:
        json.dump(result, f, indent=2)
    print('=== EXACT_FULL_PATCH_EVAL ===')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
