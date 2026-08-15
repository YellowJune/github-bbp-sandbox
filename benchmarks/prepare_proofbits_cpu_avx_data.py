from pathlib import Path
import json
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM,AutoTokenizer

MODEL='Qwen/Qwen2.5-0.5B-Instruct';N=8;MAX_LEN=128
OUT=Path('benchmarks/tmp_proofbits_cpu');OUT.mkdir(parents=True,exist_ok=True)
torch.set_grad_enabled(False)
tok=AutoTokenizer.from_pretrained(MODEL);m=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float32);m.eval()
ds=load_dataset('wikitext','wikitext-2-raw-v1',split='test');hs=[]
for r in ds:
    t=r['text'].strip()
    if len(t)<220:continue
    x=tok(t,return_tensors='pt',truncation=True,max_length=MAX_LEN)
    if x['input_ids'].shape[1]<24:continue
    H=m.model(**x,use_cache=False,return_dict=True).last_hidden_state[0].float().cpu()
    for p in range(11,H.shape[0],17):
        hs.append(H[p])
        if len(hs)>=N:break
    if len(hs)>=N:break
H=torch.stack(hs).contiguous();W=m.lm_head.weight.detach().cpu().half().contiguous();V,D=W.shape
bits=(W.view(torch.int16).to(torch.int32)&0xffff).contiguous();high=((bits>>8)&255).to(torch.uint8).contiguous();low=(bits&255).to(torch.uint8).contiguous()
# Write native little-endian uint16 words and both lossless byte planes.
np.asarray(bits.numpy(),dtype=np.uint16).tofile(OUT/'full_u16.bin')
high.numpy().tofile(OUT/'high_u8.bin')
low.numpy().tofile(OUT/'low_u8.bin')
H.numpy().astype(np.float32).tofile(OUT/'hidden_f32.bin')
meta={'model':MODEL,'V':V,'D':D,'N':len(H),'full_bytes':int(2*V*D),'high_bytes':int(V*D),'low_bytes':int(V*D)}
(OUT/'meta.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2))
