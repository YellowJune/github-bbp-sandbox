import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load

MODEL='mlx-community/gemma-3-270m-bf16'
CAPS=[8,12,16,20,24]


def main():
    model,_=load(MODEL)
    model.set_dtype(mx.float16); mx.eval(model.parameters())
    w=model.lm_head.weight
    a=np.array(w,copy=True).astype(np.float16,copy=False)
    hb=(a.view(np.uint16)>>8).astype(np.uint8)
    V,D=hb.shape
    freq=np.bincount(hb.reshape(-1),minlength=256)
    top=np.argsort(freq)[::-1]
    out={'kind':'proofbits_nibble_fixed_escape_analysis','model':MODEL,'V':V,'D':D}
    for K in [7,15,31,63]:
        common=top[:K]
        covered=float(freq[common].sum()/freq.sum())
        out[f'top{K}_coverage']=covered
        out[f'top{K}_escape_fraction']=1-covered
        out[f'top{K}_values']=[int(x) for x in common]
    common=top[:15]
    lut=np.ones(256,dtype=np.bool_)
    lut[common]=False
    esc=lut[hb]
    # GPU layout pairs iterations for each SIMD lane: each 64-weight block is
    # two 32-wide iterations, exactly matching original per-lane FMA order.
    assert D%64==0
    blocks=esc.reshape(V,D//64,64)
    counts=blocks.sum(axis=2)
    out['block64_escape_mean']=float(counts.mean())
    out['block64_escape_median']=float(np.median(counts))
    out['block64_escape_p95']=float(np.quantile(counts,.95))
    out['block64_escape_p99']=float(np.quantile(counts,.99))
    out['block64_escape_max']=int(counts.max())
    schemes=[]
    for cap in CAPS:
        overflow=counts>cap
        f=float(overflow.mean())
        # Common block: 32 nibble-code bytes + cap raw escape bytes.
        # Overflow block: use raw 64 high bytes, selected by one block flag bit.
        # Include one bit/block flag amortized over 64 weights.
        bytes_per_block=(1-f)*(32+cap)+f*64
        bits_per_weight=8*bytes_per_block/64 + 1/64
        schemes.append({'escape_capacity':cap,'overflow_block_fraction':f,
                        'expected_highplane_bits_per_weight':float(bits_per_weight),
                        'raw8_reduction':float(8/bits_per_weight)})
    out['schemes']=schemes
    Path('experiments/artifacts').mkdir(parents=True,exist_ok=True)
    Path('experiments/artifacts/proofbits_nibble_escape_analysis.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
