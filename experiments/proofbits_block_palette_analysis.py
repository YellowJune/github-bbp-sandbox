import json
import numpy as np
import mlx.core as mx
from mlx_lm import load
from pathlib import Path

MODEL='mlx-community/gemma-3-270m-bf16'
BLOCKS=[32,64,128]
KS=[8,16,32]

def main():
 model,_=load(MODEL); model.set_dtype(mx.float16); mx.eval(model.parameters())
 w=model.lm_head.weight
 V,D=[int(x) for x in w.shape]
 arr=np.array(w,copy=True).astype(np.float16,copy=False)
 raw=arr.view(np.uint16)
 hb=(raw>>8).astype(np.uint8)
 out={'kind':'proofbits_block_local_highbyte_palette_analysis','model':MODEL,'V':V,'D':D,'schemes':[]}
 for B in BLOCKS:
  assert D%B==0
  blocks=hb.reshape(V,D//B,B)
  # Sorting uint8 values makes distinct count vectorizable without Python sets.
  s=np.sort(blocks,axis=2)
  distinct=1+np.sum(s[:,:,1:]!=s[:,:,:-1],axis=2)
  for K in KS:
   if K>=B: continue
   f=float(np.mean(distinct>K))
   # Proposed row-indexed fallback layout for K=16/B=64:
   # fixed code stream ceil(log2(K))*B bits + K raw-byte palette bits per block,
   # plus per-row bitmap (one bit/block) and one uint32 fallback-base per row,
   # plus raw B bytes for each fallback block. Row metadata amortized over D weights.
   b=int(np.ceil(np.log2(K)))
   common=b + (8*K)/B
   metadata=(D//B + 32)/D
   total=common+metadata+8*f
   out['schemes'].append({'block':B,'palette_entries':K,'code_bits':b,
      'fallback_fraction':f,'median_distinct':float(np.median(distinct)),
      'p95_distinct':float(np.quantile(distinct,0.95)),'max_distinct':int(distinct.max()),
      'common_bits_per_weight':float(common),'row_metadata_bits_per_weight':float(metadata),
      'estimated_total_highplane_bits_per_weight':float(total),
      'vs_raw8_reduction':float(8/total)})
 Path('experiments/artifacts').mkdir(parents=True,exist_ok=True)
 Path('experiments/artifacts/proofbits_block_palette_analysis.json').write_text(json.dumps(out,indent=2))
 print(json.dumps(out,indent=2))
if __name__=='__main__':main()
