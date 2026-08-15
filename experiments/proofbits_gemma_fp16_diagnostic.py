import json,time
from pathlib import Path
import numpy as np,torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM,AutoTokenizer
MODEL='unsloth/gemma-3-270m';OUT=Path('experiments/artifacts/proofbits_gemma_fp16_diagnostic.json');OUT.parent.mkdir(parents=True,exist_ok=True)
torch.set_grad_enabled(False)
def main():
 tok=AutoTokenizer.from_pretrained(MODEL);m=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float16,low_cpu_mem_usage=True);m.eval();ds=load_dataset('wikitext','wikitext-2-raw-v1',split='test')
 rows=[]
 for r in ds:
  t=r['text'].strip()
  if len(t)<180:continue
  x=tok(t,return_tensors='pt',truncation=True,max_length=80)
  if x['input_ids'].shape[1]<20:continue
  o=m.model(**x,use_cache=False,return_dict=True).last_hidden_state.detach().cpu()
  rows.append({'shape':list(o.shape),'dtype':str(o.dtype),'finite_fraction':float(torch.isfinite(o).float().mean()),'nan':int(torch.isnan(o).sum()),'posinf':int(torch.isposinf(o).sum()),'neginf':int(torch.isneginf(o).sum()),'absmax_finite':float(o[torch.isfinite(o)].abs().max()) if torch.isfinite(o).any() else None})
  if len(rows)>=12:break
 report={'kind':'gemma_cpu_fp16_diagnostic','model':MODEL,'rows':rows,'all_finite':bool(all(r['finite_fraction']==1 for r in rows))}
 OUT.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
