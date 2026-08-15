import json
from pathlib import Path
import mlx.core as mx
import numpy as np
from mlx_lm import load

MODEL='mlx-community/gemma-3-270m-bf16'
CAPS=[1,2,3,4,5,6,8]


def main():
    model,_=load(MODEL); model.set_dtype(mx.float16); mx.eval(model.parameters())
    a=np.array(model.lm_head.weight,copy=True).astype(np.float16,copy=False)
    hb=(a.view(np.uint16)>>8).astype(np.uint8); V,D=hb.shape
    freq=np.bincount(hb.reshape(-1),minlength=256); top=np.argsort(freq)[::-1]
    common=top[:63]; lut=np.ones(256,dtype=np.bool_); lut[common]=False
    esc=lut[hb]
    out={'kind':'proofbits_top63_code_escape_analysis','model':MODEL,'V':V,'D':D,
         'coverage':float(freq[common].sum()/freq.sum()),'codebook':[int(x) for x in common]}
    schemes=[]
    for B in [32,64,128]:
        assert D%B==0
        counts=esc.reshape(V,D//B,B).sum(axis=2)
        stat={'block':B,'mean_escape':float(counts.mean()),'median_escape':float(np.median(counts)),
              'p95_escape':float(np.quantile(counts,.95)),'p99_escape':float(np.quantile(counts,.99)),
              'max_escape':int(counts.max()),'caps':[]}
        for cap in CAPS:
            overflow=float(np.mean(counts>cap))
            # Common block: 6*B bits code + cap raw escape bytes.
            # Overflow block: raw B bytes. Include one flag bit/block.
            common_bytes=(6*B)/8 + cap
            exp_bytes=(1-overflow)*common_bytes + overflow*B
            bpw=8*exp_bytes/B + 1/B
            stat['caps'].append({'cap':cap,'overflow_fraction':overflow,
                                 'expected_bits_per_weight':float(bpw),'raw8_reduction':float(8/bpw)})
        schemes.append(stat)
    out['schemes']=schemes
    Path('experiments/artifacts').mkdir(parents=True,exist_ok=True)
    Path('experiments/artifacts/proofbits_code63_escape_analysis.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))

if __name__=='__main__': main()
