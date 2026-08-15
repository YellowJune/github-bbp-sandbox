import gc,json,math
from pathlib import Path
import numpy as np,torch
from transformers import AutoModelForCausalLM
MODELS=['Qwen/Qwen2.5-0.5B-Instruct','unsloth/gemma-3-270m','Qwen/Qwen2.5-1.5B-Instruct'];CODE_BITS=[4,5,6]
OUT=Path('experiments/artifacts/proofbits_highbyte_palette_compression.json');OUT.parent.mkdir(parents=True,exist_ok=True);torch.set_grad_enabled(False)
def entropy(hist):
 p=hist[hist>0]/hist.sum();return float(-(p*np.log2(p)).sum())
def analyze(model):
 m=AutoModelForCausalLM.from_pretrained(model,torch_dtype=torch.float16,low_cpu_mem_usage=True);W=m.lm_head.weight.detach().cpu().half().contiguous();V,D=W.shape;del m;gc.collect();raw=W.view(torch.int16).to(torch.int32)&0xffff;hb=((raw>>8)&255).to(torch.uint8).flatten().numpy();hist=np.bincount(hb,minlength=256);order=np.argsort(hist)[::-1];N=hist.sum();r={'model':model,'vocab':V,'hidden_dim':D,'weights':int(N),'entropy_bits':entropy(hist),'schemes':{}}
 for b in CODE_BITS:
  # Reserve all-ones code as escape. Remaining 2^b-1 codes map to the most frequent exact high bytes.
  K=(1<<b)-1;covered=int(hist[order[:K]].sum());esc=N-covered;e=esc/N
  # Dense code stream b bits/weight + 8 raw bits for each escape + tiny global palette (K bytes).
  logical=b+8*e+(8*K)/N
  r['schemes'][str(b)]={'code_bits':b,'palette_entries':K,'covered_fraction':float(covered/N),'escape_fraction':float(e),'logical_highplane_bits_per_weight':float(logical),'idealized_full_fp16_vs_compressed_highplane_ratio_if_suffix_zero':float(16/logical),'palette_bytes':K}
 return r
def main():
 report={'kind':'proofbits_lossless_highbyte_palette_escape_potential','models':[analyze(m) for m in MODELS],'scheme':'Reserve one code as escape; remaining fixed-width codes map to globally most frequent exact FP16 high-byte values; escapes store the exact raw high byte in a separate sequential stream. Lossless logical bit budget = code_bits + 8*escape_fraction plus negligible palette.','caveat':'Logical bit budget only. A GPU decoder must identify escape positions and stream/gather rare raw bytes efficiently. No latency or alignment overhead measured.'};OUT.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
