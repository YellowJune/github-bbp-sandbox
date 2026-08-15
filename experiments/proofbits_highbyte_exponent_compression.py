import gc,json,math,os
from pathlib import Path
import numpy as np,torch
from transformers import AutoModelForCausalLM
MODELS=['Qwen/Qwen2.5-0.5B-Instruct','unsloth/gemma-3-270m','Qwen/Qwen2.5-1.5B-Instruct'];GROUPS=[16,32,64,128,256]
OUT=Path('experiments/artifacts/proofbits_highbyte_exponent_compression.json');OUT.parent.mkdir(parents=True,exist_ok=True);torch.set_grad_enabled(False)
def entropy(hist):
 p=hist[hist>0]/hist.sum();return float(-(p*np.log2(p)).sum())
def analyze(model):
 m=AutoModelForCausalLM.from_pretrained(model,torch_dtype=torch.float16,low_cpu_mem_usage=True);W=m.lm_head.weight.detach().cpu().half().contiguous();V,D=W.shape;del m;gc.collect();bits=(W.view(torch.int16).to(torch.int32)&0xffff);hb=((bits>>8)&255).to(torch.uint8);exp=((hb.to(torch.int16)>>2)&31).to(torch.int16);assert not bool((exp==31).any())
 hist=np.bincount(hb.flatten().numpy(),minlength=256);eh=np.bincount(exp.flatten().numpy(),minlength=32);res={'model':model,'vocab':V,'hidden_dim':D,'weights':V*D,'highbyte_shannon_entropy_bits':entropy(hist),'exponent_entropy_bits':entropy(eh),'unique_highbytes':int((hist>0).sum()),'unique_exponents':int((eh>0).sum()),'groups':{}}
 for g in GROUPS:
  widths=[];nb=0
  for a in range(0,D,g):
   e=exp[:,a:min(a+g,D)].to(torch.int32);span=(e.amax(1)-e.amin(1)+1).cpu().numpy();w=np.ceil(np.log2(span)).astype(np.int64);widths.append(w);nb+=V
  w=np.concatenate(widths);# metadata: 5-bit min exponent + 3-bit delta-width per block, conservative fixed metadata.
  meta_per_weight=8.0/g
  total=3.0+float(w.mean())+meta_per_weight
  res['groups'][str(g)]={'group':g,'mean_exponent_delta_bits':float(w.mean()),'p50':float(np.percentile(w,50)),'p90':float(np.percentile(w,90)),'p99':float(np.percentile(w,99)),'max':int(w.max()),'metadata_bits_per_weight':meta_per_weight,'lossless_highplane_bits_per_weight':total,'idealized_full_fp16_vs_compressed_highplane_ratio_if_suffix_zero':16.0/total,'fraction_blocks_delta_le3':float(np.mean(w<=3)),'fraction_blocks_delta_le4':float(np.mean(w<=4))}
 return res
def main():
 report={'kind':'proofbits_lossless_highbyte_exponent_compression_potential','models':[analyze(m) for m in MODELS],'scheme':'Per row-group store sign bit + top2 mantissa bits per weight, 5-bit minimum exponent and 3-bit delta-width per group, fixed-width exponent delta per weight. This exactly reconstructs every finite FP16 high byte.','caveat':'Logical bit budget only. No packed GPU decoder or latency measurement; irregular per-block bit widths can cost more in a real aligned layout.'};OUT.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
