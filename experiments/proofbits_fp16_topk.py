import json, time
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL='Qwen/Qwen2.5-0.5B-Instruct'
KS=[1,5,10,50]
N_AG=64; N_GEN=64; MAX_LEN=128
OUT=Path('experiments/artifacts/proofbits_fp16_topk.json'); OUT.parent.mkdir(parents=True,exist_ok=True)
torch.set_grad_enabled(False); torch.manual_seed(3); np.random.seed(3)

def log(x): print(f'[{time.strftime("%H:%M:%S")}] {x}',flush=True)
def hidden(m,**kw): return m.model(**kw,use_cache=False,return_dict=True).last_hidden_state
def collect(m,tok,texts):
 hs=[]
 for t in texts:
  x=tok(t,return_tensors='pt',truncation=True,max_length=MAX_LEN); hs.append(hidden(m,**x)[0,-1].float().cpu())
 return torch.stack(hs)
def gen(m,tok):
 ps=['Solve carefully: 137 * 29 =','The derivative of x^3 + 2x is','Write a Python function for binary search:','Factor 84 into prime factors:','Compute 2^10 =','Write SQL selecting users older than 18:','The gcd of 84 and 126 is','Simplify (x+2)(x-2):']
 seq=[tok(p,return_tensors='pt')['input_ids'] for p in ps]; hs=[]
 while len(hs)<N_GEN:
  ns=[]
  for ids in seq:
   h=hidden(m,input_ids=ids)[0,-1].float().cpu(); hs.append(h)
   if len(hs)>=N_GEN: break
   t=m.lm_head(h[None])[0].argmax().view(1,1); ns.append(torch.cat([ids,t],1))
  seq=ns
 return torch.stack(hs[:N_GEN])
def interval(w):
 w16=w.half().contiguous(); raw=w16.view(torch.int16).to(torch.int32)&0xffff; hi=(raw>>8)&255
 a=(hi<<8).to(torch.int16).contiguous().view(torch.float16).float(); b=((hi<<8)|255).to(torch.int16).contiguous().view(torch.float16).float()
 if not (torch.isfinite(a).all() and torch.isfinite(b).all()): raise RuntimeError('nonfinite')
 return w16.float().contiguous(),((a+b)*.5).contiguous(),((b-a).abs()*.5).contiguous()
def score(h,w,ch=4096): return torch.cat([h.float()@w[a:a+ch].float().T for a in range(0,len(w),ch)],1)
def eval_domain(h,w16,mid,rad):
 exact=score(h,w16); coarse=score(h,mid); err=score(h.abs(),rad); V=exact.shape[1]; out=[]
 for k in KS:
  p=min(V,max(k,4*k)); pilots=torch.topk(coarse,k=p,dim=1).indices
  pilot_exact=exact.gather(1,pilots); B=torch.topk(pilot_exact,k=k,dim=1).values[:,-1]
  cand=(coarse+err)>=B[:,None]
  ok=[]; candn=[]; readn=[]; maxlogitdiff=[]
  true_top=torch.topk(exact,k=k,dim=1)
  for n in range(len(h)):
   ci=cand[n].nonzero().squeeze(1); candn.append(int(len(ci)))
   read=torch.unique(torch.cat([ci,pilots[n]])); readn.append(int(len(read)))
   vals=exact[n,ci]; got=torch.topk(vals,k=k).indices; got_ids=ci[got]
   ok.append(set(got_ids.tolist())==set(true_top.indices[n].tolist()))
   # Since exact logits are fetched for survivors, top-k logits themselves must be identical.
   gv=torch.sort(exact[n,got_ids]).values; tv=torch.sort(true_top.values[n]).values
   maxlogitdiff.append(float((gv-tv).abs().max()))
  c=np.array(candn); r=np.array(readn); frac=r.mean()/V; bits=8+8*frac
  out.append({'k':k,'pilot_count':p,'exact_topk_set_rate':float(np.mean(ok)),'max_topk_logit_difference':float(np.max(maxlogitdiff)),
              'candidate_mean':float(c.mean()),'read_lowbyte_rows_mean_including_pilots':float(r.mean()),'median':float(np.median(r)),'p90':float(np.percentile(r,90)),'p99':float(np.percentile(r,99)),'max':int(r.max()),
              'lowbyte_row_fraction_mean':float(frac),'bits_per_weight':float(bits),'idealized_bw_reduction':float(16/bits)})
 return out
def main():
 log('load'); tok=AutoTokenizer.from_pretrained(MODEL); m=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float32); m.eval(); w=m.lm_head.weight.detach().float().cpu(); V,D=w.shape; w16,mid,rad=interval(w)
 ag=load_dataset('ag_news',split='test'); ha=collect(m,tok,[ag[i]['text'] for i in range(N_AG)]); hg=gen(m,tok)
 log('AG'); a=eval_domain(ha,w16,mid,rad); log(str(a)); log('GEN'); g=eval_domain(hg,w16,mid,rad); log(str(g))
 r={'kind':'proofbits_exact_fp16_topk','model':MODEL,'vocab':V,'hidden_dim':D,'theorem':'evaluate coarse top-4k pilots; kth exact pilot score is a valid lower bound on global kth score; rows with U_i below it cannot belong to global top-k','ag_news':a,'autoregressive_math_code':g,'caveat':'Enables exact top-k set/logits for top-k sampling. Does not by itself make full-softmax or unconstrained nucleus sampling exact without additional certification.'}; OUT.write_text(json.dumps(r,indent=2)); print(json.dumps(r,indent=2))
if __name__=='__main__': main()
