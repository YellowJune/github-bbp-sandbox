import gc,json,time
from pathlib import Path
import numpy as np, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM,AutoTokenizer
MODEL='unsloth/gemma-3-270m'; N=64; MAX_LEN=96; BATCH=2; CH=4096; PILOT=4
SIZES=[8192,16384,32768,65536,131072,262144]
OUT=Path('experiments/artifacts/proofbits_gemma_vocab_scaling.json'); OUT.parent.mkdir(parents=True,exist_ok=True)
torch.set_grad_enabled(False); torch.manual_seed(59); np.random.seed(59)
def log(x): print(f'[{time.strftime("%H:%M:%S")}] {x}',flush=True)
def collect(m,tok):
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
def full(h,W):
 hp=torch.clamp(h.float(),min=0);hn=torch.clamp(h.float(),max=0);ex=[];up=[]
 for a in range(0,W.shape[0],CH):
  w=W[a:a+CH];lo,hi=endpoints(w);ex.append(h.float()@w.float().T);up.append(hp@hi.T+hn@lo.T)
 return torch.cat(ex,1),torch.cat(up,1)
def main():
 log('load');tok=AutoTokenizer.from_pretrained(MODEL);m=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float32,low_cpu_mem_usage=True);m.eval();H=collect(m,tok);W=m.lm_head.weight.detach().cpu().half().contiguous().clone();V,D=W.shape;del m;gc.collect()
 assert SIZES[-1]==V
 perm=torch.randperm(V,generator=torch.Generator().manual_seed(5901))
 acc={v:[] for v in SIZES}; oks={v:[] for v in SIZES}
 for s in range(0,len(H),BATCH):
  ex,U=full(H[s:s+BATCH],W)
  for n in range(ex.shape[0]):
   for v in SIZES:
    idx=perm[:v]; ee=ex[n,idx];uu=U[n,idx]; p=torch.topk(uu,k=PILOT).indices;B=ee[p].max();mask=uu>=B;sv=mask.nonzero().squeeze(1);pred=int(idx[sv[ee[sv].argmax()]]);ref=int(idx[ee.argmax()]);acc[v].append(int(mask.sum()));oks[v].append(pred==ref)
  log(f'{min(s+BATCH,len(H))}/{len(H)}')
 res={}
 for v in SIZES:
  c=np.asarray(acc[v],float);res[str(v)]={'all_exact':bool(all(oks[v])),'mean':float(c.mean()),'median':float(np.median(c)),'p90':float(np.percentile(c,90)),'p99':float(np.percentile(c,99)),'max':int(c.max()),'fraction':float(c.mean()/v),'idealized_reduction':float(2/(1+c.mean()/v))}
 x=np.log(np.asarray(SIZES,dtype=float));y=np.log(np.asarray([res[str(v)]['mean'] for v in SIZES]));alpha,intercept=np.polyfit(x,y,1)
 report={'kind':'proofbits_controlled_vocab_scaling','model':MODEL,'full_vocab':V,'hidden_dim':D,'n':len(H),'pilot_k':PILOT,'nested_random_actual_weight_rows':True,'results':res,'loglog_candidate_scaling_exponent':float(alpha),'fit_intercept':float(intercept),
 'caveat':'Controlled computational stress using nested random subsets of rows from one trained vocabulary. It does not imply that independently trained models with larger vocabularies follow the fitted exponent.'};OUT.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
