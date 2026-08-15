import gc,json,time
from pathlib import Path
import numpy as np, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM,AutoTokenizer
MODELS=[('Qwen/Qwen2.5-0.5B-Instruct',64),('unsloth/gemma-3-270m',64)];MAX_LEN=96;BATCH=2;CH=4096
OUT=Path('experiments/artifacts/proofbits_pilot1_vs4.json');OUT.parent.mkdir(parents=True,exist_ok=True);torch.set_grad_enabled(False);torch.manual_seed(79);np.random.seed(79)
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
def ep(w):
 raw=w.contiguous().view(torch.int16).to(torch.int32)&0xffff;hb=(raw>>8)&255;a=(hb<<8).to(torch.int16).contiguous().view(torch.float16).float();b=((hb<<8)|255).to(torch.int16).contiguous().view(torch.float16).float();return torch.minimum(a,b),torch.maximum(a,b)
def calc(h,W):
 hp=torch.clamp(h.float(),min=0);hn=torch.clamp(h.float(),max=0);ex=[];up=[]
 for a in range(0,W.shape[0],CH):
  w=W[a:a+CH];lo,hi=ep(w);ex.append(h.float()@w.float().T);up.append(hp@hi.T+hn@lo.T)
 return torch.cat(ex,1),torch.cat(up,1)
def run(model,N):
 tok=AutoTokenizer.from_pretrained(model);m=AutoModelForCausalLM.from_pretrained(model,torch_dtype=torch.float32,low_cpu_mem_usage=True);m.eval();H=collect(m,tok,N);W=m.lm_head.weight.detach().cpu().half().contiguous().clone();V,D=W.shape;del m;gc.collect();res={k:{'c':[],'ok':[],'hit':[]} for k in [1,4]}
 for s in range(0,len(H),BATCH):
  ex,U=calc(H[s:s+BATCH],W)
  for k in [1,4]:
   p=torch.topk(U,k=k,dim=1).indices;B=ex.gather(1,p).amax(1);mask=U>=B[:,None]
   for n in range(ex.shape[0]):
    idx=mask[n].nonzero().squeeze(1);ref=int(ex[n].argmax());pred=int(idx[ex[n,idx].argmax()]);res[k]['c'].append(int(idx.numel()));res[k]['ok'].append(pred==ref);res[k]['hit'].append(bool((p[n]==ref).any()))
 out={'model':model,'vocab':V,'hidden_dim':D,'n':len(H),'pilots':{}}
 for k in [1,4]:
  a=np.asarray(res[k]['c'],float);f=a.mean()/V;out['pilots'][str(k)]={'all_exact':bool(all(res[k]['ok'])),'pilot_hit_rate':float(np.mean(res[k]['hit'])),'mean':float(a.mean()),'median':float(np.median(a)),'p90':float(np.percentile(a,90)),'p99':float(np.percentile(a,99)),'max':int(a.max()),'fraction':float(f),'idealized_reduction':float(2/(1+f))}
 return out
def main():
 report={'kind':'proofbits_pilot1_vs4','results':[run(m,n) for m,n in MODELS],'caveat':'CPU exactness/candidate comparison only; global reduction and GPU latency not measured.'};OUT.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
