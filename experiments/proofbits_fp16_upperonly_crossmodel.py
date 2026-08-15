import json, os, time
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL=os.getenv('MODEL','Qwen/Qwen2.5-0.5B-Instruct'); N=64; PILOTS=[1,4,16]
OUT=Path('experiments/artifacts')/('upperonly_'+MODEL.replace('/','__')+'.json'); OUT.parent.mkdir(parents=True,exist_ok=True)
torch.set_grad_enabled(False); torch.manual_seed(11); np.random.seed(11)
def log(x): print(f'[{time.strftime("%H:%M:%S")}] {x}',flush=True)
def hidden(m,**kw): return m.model(**kw,use_cache=False,return_dict=True).last_hidden_state
def collect(m,tok,texts):
 hs=[]
 for t in texts:
  x=tok(t,return_tensors='pt',truncation=True,max_length=128); hs.append(hidden(m,**x)[0,-1].float().cpu())
 return torch.stack(hs)
def gen(m,tok):
 ps=['Solve carefully: 137 * 29 =','The derivative of x^3 + 2x is','Write a Python function for binary search:','Factor 84 into prime factors:','Compute 2^10 =','Write SQL selecting users older than 18:','The gcd of 84 and 126 is','Simplify (x+2)(x-2):']
 seq=[tok(p,return_tensors='pt')['input_ids'] for p in ps]; hs=[]
 while len(hs)<N:
  ns=[]
  for ids in seq:
   h=hidden(m,input_ids=ids)[0,-1].float().cpu(); hs.append(h)
   if len(hs)>=N: break
   t=m.lm_head(h[None])[0].argmax().view(1,1); ns.append(torch.cat([ids,t],1))
  seq=ns
 return torch.stack(hs[:N])
def endpoints(w):
 w16=w.half().contiguous(); raw=w16.view(torch.int16).to(torch.int32)&0xffff; hi=(raw>>8)&255
 a=(hi<<8).to(torch.int16).contiguous().view(torch.float16).float(); b=((hi<<8)|255).to(torch.int16).contiguous().view(torch.float16).float()
 if not (torch.isfinite(a).all() and torch.isfinite(b).all()): raise RuntimeError('nonfinite')
 lo=torch.minimum(a,b).contiguous(); up=torch.maximum(a,b).contiguous()
 return w16.float().contiguous(),lo,up
def score(h,w,ch=4096): return torch.cat([h.float()@w[a:a+ch].float().T for a in range(0,len(w),ch)],1)
def upper_scores(h,lo,up,ch=4096):
 # Sign-aware endpoint gives exact coordinate-wise maximum contribution.
 hp=torch.clamp(h.float(),min=0); hn=torch.clamp(h.float(),max=0); ys=[]
 for a in range(0,len(lo),ch): ys.append(hp@up[a:a+ch].T + hn@lo[a:a+ch].T)
 return torch.cat(ys,1)
def eval_domain(h,w16,lo,up):
 exact=score(h,w16); U=upper_scores(h,lo,up); ref=exact.argmax(1); V=exact.shape[1]; arr=[]
 for p in PILOTS:
  ids=torch.topk(U,k=p,dim=1).indices; B=exact.gather(1,ids).amax(1); cand=U>=B[:,None]
  ok=[]; cnt=[]; read=[]; hit=[]
  for n in range(len(h)):
   ci=cand[n].nonzero().squeeze(1); cnt.append(len(ci)); rr=torch.unique(torch.cat([ci,ids[n]])); read.append(len(rr)); hit.append(bool((ids[n]==ref[n]).any()))
   pred=int(ci[exact[n,ci].argmax()]); ok.append(pred==int(ref[n]))
  c=np.array(cnt); r=np.array(read); f=r.mean()/V; bits=8+8*f
  arr.append({'pilot_k':p,'exact_rate':float(np.mean(ok)),'pilot_contains_true_top1_rate':float(np.mean(hit)),'candidate_mean':float(c.mean()),'read_low_rows_mean':float(r.mean()),'median':float(np.median(r)),'p90':float(np.percentile(r,90)),'p99':float(np.percentile(r,99)),'max':int(r.max()),'fraction':float(f),'bits_per_weight':float(bits),'idealized_bw_reduction':float(16/bits)})
 return arr
def main():
 log('load '+MODEL); tok=AutoTokenizer.from_pretrained(MODEL); m=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float32); m.eval(); w=m.lm_head.weight.detach().float().cpu(); V,D=w.shape; w16,lo,up=endpoints(w)
 ag=load_dataset('ag_news',split='test'); ha=collect(m,tok,[ag[i]['text'] for i in range(N)]); hg=gen(m,tok)
 log('AG'); a=eval_domain(ha,w16,lo,up); log(str(a)); log('GEN'); g=eval_domain(hg,w16,lo,up); log(str(g))
 r={'kind':'proofbits_fp16_upperonly_crossmodel','model':MODEL,'vocab':V,'hidden_dim':D,'upper_bound':'sum_j max(h_j*w^-_ij,h_j*w^+_ij) using only high-byte interval endpoints','coarse_accumulators_per_weight':1,'extra_metadata_bytes':0,'ag_news':a,'autoregressive_math_code':g,'caveat':'Idealized byte traffic; no GPU timing. Pilot selection uses largest certified upper bounds rather than midpoint scores.'}; OUT.write_text(json.dumps(r,indent=2)); print(json.dumps(r,indent=2))
if __name__=='__main__': main()
