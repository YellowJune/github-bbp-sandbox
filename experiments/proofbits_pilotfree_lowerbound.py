import gc,json,os,time
from pathlib import Path
import numpy as np, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM,AutoTokenizer
MODELS=[('Qwen/Qwen2.5-0.5B-Instruct',64),('unsloth/gemma-3-270m',64)]
MAX_LEN=96;BATCH=2;CH=4096
OUT=Path('experiments/artifacts/proofbits_pilotfree_lowerbound.json');OUT.parent.mkdir(parents=True,exist_ok=True)
torch.set_grad_enabled(False);torch.manual_seed(73);np.random.seed(73)
def log(x):print(f'[{time.strftime("%H:%M:%S")}] {x}',flush=True)
def collect(m,tok,N):
 ds=load_dataset('wikitext','wikitext-2-raw-v1',split='test');hs=[]
 for r in ds:
  t=r['text'].strip()
  if len(t)<180:continue
  x=tok(t,return_tensors='pt',truncation=True,max_length=MAX_LEN)
  if x['input_ids'].shape[1]<20:continue
  H=m.model(**x,use_cache=False,return_dict=True).last_hidden_state[0].float().cpu()
  for p in range(7,H.shape[0],9):
   hs.append(H[p])
   if len(hs)>=N:return torch.stack(hs)
 return torch.stack(hs)
def endpoints(w):
 raw=w.contiguous().view(torch.int16).to(torch.int32)&0xffff;hb=(raw>>8)&255
 a=(hb<<8).to(torch.int16).contiguous().view(torch.float16).float();b=((hb<<8)|255).to(torch.int16).contiguous().view(torch.float16).float()
 return torch.minimum(a,b),torch.maximum(a,b)
def calc(h,W):
 hp=torch.clamp(h.float(),min=0);hn=torch.clamp(h.float(),max=0);ex=[];loS=[];upS=[]
 for a in range(0,W.shape[0],CH):
  w=W[a:a+CH];lo,hi=endpoints(w);ex.append(h.float()@w.float().T);loS.append(hp@lo.T+hn@hi.T);upS.append(hp@hi.T+hn@lo.T)
 return torch.cat(ex,1),torch.cat(loS,1),torch.cat(upS,1)
def run(model,N):
 tok=AutoTokenizer.from_pretrained(model);m=AutoModelForCausalLM.from_pretrained(model,torch_dtype=torch.float32,low_cpu_mem_usage=True);m.eval();H=collect(m,tok,N);W=m.lm_head.weight.detach().cpu().half().contiguous().clone();V,D=W.shape;del m;gc.collect();c=[];oks=[];gap=[]
 for s in range(0,len(H),BATCH):
  ex,L,U=calc(H[s:s+BATCH],W);B=L.max(1).values;mask=U>=B[:,None]
  for n in range(ex.shape[0]):
   idx=mask[n].nonzero().squeeze(1);ref=int(ex[n].argmax());pred=int(idx[ex[n,idx].argmax()]);c.append(int(idx.numel()));oks.append(pred==ref);gap.append(float(ex[n,ref]-B[n]))
  log(f'{model} {min(s+BATCH,len(H))}/{len(H)} mean={np.mean(c):.2f}')
 a=np.asarray(c,float);f=a.mean()/V;return {'model':model,'vocab':V,'hidden_dim':D,'n':len(H),'all_exact':bool(all(oks)),'candidate_mean':float(a.mean()),'median':float(np.median(a)),'p90':float(np.percentile(a,90)),'p99':float(np.percentile(a,99)),'max':int(a.max()),'fraction':float(f),'idealized_weight_byte_reduction':float(2/(1+f)),'winner_score_minus_max_lower_mean':float(np.mean(gap)),'winner_score_minus_max_lower_p90':float(np.percentile(gap,90))}
def main():
 results=[]
 for m,n in MODELS:results.append(run(m,n))
 report={'kind':'proofbits_pilotfree_max_lower','results':results,'theorem':'B=max_i L_i <= max_i z_i. Therefore every exact argmax row has U_i >= z_i >= B and survives. No pilot low-byte read is needed.','caveat':'Computing both L and U doubles score-bound arithmetic versus upper-only, though it does not double high-byte weight traffic. GPU wall-clock not measured.'};OUT.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
