import gc,json,time
from pathlib import Path
import numpy as np, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM,AutoTokenizer
MODEL='unsloth/gemma-3-270m'; KS=[1,10,50]; N=64; MAX_LEN=96; BATCH=2; CH=4096
OUT=Path('experiments/artifacts/proofbits_gemma270m_topk.json'); OUT.parent.mkdir(parents=True,exist_ok=True)
torch.set_grad_enabled(False); torch.manual_seed(53); np.random.seed(53)
def log(x): print(f'[{time.strftime("%H:%M:%S")}] {x}',flush=True)
def collect(m,tok):
 ds=load_dataset('wikitext','wikitext-2-raw-v1',split='test'); hs=[]
 for r in ds:
  t=r['text'].strip()
  if len(t)<180: continue
  x=tok(t,return_tensors='pt',truncation=True,max_length=MAX_LEN)
  if x['input_ids'].shape[1]<20: continue
  H=m.model(**x,use_cache=False,return_dict=True).last_hidden_state[0].float().cpu()
  for p in range(6,H.shape[0],8):
   hs.append(H[p])
   if len(hs)>=N:return torch.stack(hs)
 return torch.stack(hs)
def ep(w):
 raw=w.contiguous().view(torch.int16).to(torch.int32)&0xffff; hb=(raw>>8)&255
 a=(hb<<8).to(torch.int16).contiguous().view(torch.float16).float(); b=((hb<<8)|255).to(torch.int16).contiguous().view(torch.float16).float()
 return torch.minimum(a,b),torch.maximum(a,b)
def calc(h,W):
 hp=torch.clamp(h.float(),min=0);hn=torch.clamp(h.float(),max=0);ex=[];up=[]
 for a in range(0,W.shape[0],CH):
  w=W[a:a+CH];lo,hi=ep(w);ex.append(h.float()@w.float().T);up.append(hp@hi.T+hn@lo.T)
 return torch.cat(ex,1),torch.cat(up,1)
def main():
 log('load');tok=AutoTokenizer.from_pretrained(MODEL);m=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float32,low_cpu_mem_usage=True);m.eval();H=collect(m,tok);W=m.lm_head.weight.detach().cpu().half().contiguous().clone();V,D=W.shape;del m;gc.collect()
 acc={k:[] for k in KS};ok={k:[] for k in KS}
 for s in range(0,len(H),BATCH):
  ex,U=calc(H[s:s+BATCH],W)
  for k in KS:
   pk=max(4*k,4);pidx=torch.topk(U,k=pk,dim=1).indices;B=torch.topk(ex.gather(1,pidx),k=k,dim=1).values[:,-1];mask=U>=B[:,None]
   for n in range(ex.shape[0]):
    idx=mask[n].nonzero().squeeze(1);pred=idx[torch.topk(ex[n,idx],k=k).indices];ref=torch.topk(ex[n],k=k).indices
    acc[k].append(int(idx.numel()));ok[k].append(set(map(int,pred.tolist()))==set(map(int,ref.tolist())))
  log(f'{min(s+BATCH,len(H))}/{len(H)}')
 res={}
 for k in KS:
  c=np.asarray(acc[k],float);f=c.mean()/V;res[str(k)]={'all_exact':bool(all(ok[k])),'n':len(c),'pilot_k':max(4*k,4),'mean':float(c.mean()),'median':float(np.median(c)),'p90':float(np.percentile(c,90)),'p99':float(np.percentile(c,99)),'max':int(c.max()),'fraction':float(f),'idealized_reduction':float(2/(1+f))}
 report={'kind':'proofbits_gemma270m_topk','model':MODEL,'vocab':V,'hidden_dim':D,'n':len(H),'results':res,'caveat':'FP16-rounded lm_head, FP32 accumulation; idealized bytes only.'};OUT.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
