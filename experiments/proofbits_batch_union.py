import gc,json,time
from pathlib import Path
import numpy as np,torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM,AutoTokenizer
MODEL='Qwen/Qwen2.5-0.5B-Instruct';N=256;MAX_LEN=128;CH=4096;PILOT=1;BATCH_SIZES=[1,2,4,8,16,32,64,128]
OUT=Path('experiments/artifacts/proofbits_batch_union.json');OUT.parent.mkdir(parents=True,exist_ok=True);torch.set_grad_enabled(False);torch.manual_seed(83);np.random.seed(83)
def log(x):print(f'[{time.strftime("%H:%M:%S")}] {x}',flush=True)
def collect(m,tok):
 ds=load_dataset('wikitext','wikitext-2-raw-v1',split='test');hs=[]
 for r in ds:
  t=r['text'].strip()
  if len(t)<220:continue
  x=tok(t,return_tensors='pt',truncation=True,max_length=MAX_LEN)
  if x['input_ids'].shape[1]<24:continue
  H=m.model(**x,use_cache=False,return_dict=True).last_hidden_state[0].float().cpu()
  for p in range(5,H.shape[0],5):
   hs.append(H[p])
   if len(hs)>=N:return torch.stack(hs)
 return torch.stack(hs)
def ep(w):
 raw=w.contiguous().view(torch.int16).to(torch.int32)&0xffff;hb=(raw>>8)&255;a=(hb<<8).to(torch.int16).contiguous().view(torch.float16).float();b=((hb<<8)|255).to(torch.int16).contiguous().view(torch.float16).float();return torch.minimum(a,b),torch.maximum(a,b)
def calc(h,W):
 hp=torch.clamp(h,min=0);hn=torch.clamp(h,max=0);ex=[];up=[]
 for a in range(0,W.shape[0],CH):
  w=W[a:a+CH];lo,hi=ep(w);ex.append(h@w.float().T);up.append(hp@hi.T+hn@lo.T)
 return torch.cat(ex,1),torch.cat(up,1)
def main():
 tok=AutoTokenizer.from_pretrained(MODEL);m=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float32);m.eval();H=collect(m,tok);W=m.lm_head.weight.detach().cpu().half().contiguous().clone();V,D=W.shape;del m;gc.collect();log(f'V={V} D={D} N={len(H)}')
 masks=[];counts=[]
 for s in range(0,len(H),8):
  ex,U=calc(H[s:s+8].float(),W);p=U.argmax(1,keepdim=True);B=ex.gather(1,p).squeeze(1);mask=U>=B[:,None];masks.extend([x.clone() for x in mask]);counts.extend(mask.sum(1).tolist());log(f'{min(s+8,len(H))}/{len(H)}')
 results={}
 for bs in BATCH_SIZES:
  vals=[]
  for s in range(0,len(masks)-bs+1,bs):
   union=torch.stack(masks[s:s+bs]).any(0);vals.append(int(union.sum()))
  a=np.asarray(vals,float);f=a.mean()/V
  # If byte planes are batch-shared, high plane read once and low plane rows from union read once.
  results[str(bs)]={'batch_size':bs,'groups':len(vals),'union_mean':float(a.mean()),'union_median':float(np.median(a)),'union_p90':float(np.percentile(a,90)),'union_max':int(a.max()),'union_fraction_mean':float(f),'idealized_batch_shared_weight_byte_reduction':float(2/(1+f))}
 report={'kind':'proofbits_batch_survivor_union','model':MODEL,'vocab':V,'hidden_dim':D,'n_queries':len(masks),'pilot_k':PILOT,'single_query_candidate_mean':float(np.mean(counts)),'results':results,'caveat':'Candidate/weight-byte union analysis only. Actual GEMM caching, low-byte gather coalescing, and latency are not measured. Queries are consecutive collected WikiText hidden states.'};OUT.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
