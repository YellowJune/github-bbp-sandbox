import argparse
from pathlib import Path
import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen2.5-0.5B-Instruct')
    ap.add_argument('--states', type=int, default=16)
    ap.add_argument('--outdir', default='benchmarks/cpu_data')
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()

    prompts = [
        'Solve carefully: 137 * 29 =',
        'The derivative of x^3 + 2x is',
        'Write a Python function for binary search:',
        'Factor 84 into prime factors:',
        'Compute 2^10 =',
        'Write SQL selecting users older than 18:',
        'The gcd of 84 and 126 is',
        'Simplify (x+2)(x-2):',
    ]
    seq = [tok(p, return_tensors='pt')['input_ids'] for p in prompts]
    hs = []
    while len(hs) < args.states:
        new_seq = []
        for ids in seq:
            h = model.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state[0, -1].float().cpu()
            hs.append(h)
            if len(hs) >= args.states:
                break
            nxt = model.lm_head(h[None, :])[0].argmax().view(1, 1)
            new_seq.append(torch.cat([ids, nxt], dim=1))
        seq = new_seq

    H = torch.stack(hs[:args.states]).contiguous().numpy().astype(np.float32)
    W16 = model.lm_head.weight.detach().half().cpu().contiguous()
    bits = W16.view(torch.int16).to(torch.int32).numpy().astype(np.uint16)
    high = ((bits >> 8) & 0xFF).astype(np.uint8)
    low = (bits & 0xFF).astype(np.uint8)

    high.tofile(out / 'high.bin')
    low.tofile(out / 'low.bin')
    H.tofile(out / 'hidden.bin')
    meta = {
        'model': args.model,
        'vocab': int(W16.shape[0]),
        'hidden_dim': int(W16.shape[1]),
        'states': int(H.shape[0]),
        'format': 'FP16 split losslessly into high.bin and low.bin; hidden.bin float32',
    }
    (out / 'meta.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
