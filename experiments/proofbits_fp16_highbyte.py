import json, os, time
from pathlib import Path
import numpy as np, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL=os.getenv('MODEL','Qwen/Qwen2.5-0.5B-Instruct'); N=64
OUT=Path('experiments/artifacts')/('fp16_highbyte_'+MODEL.replace('/','__')+'.json'); OUT.parent.mkdir(parents=True,exist_ok=True)
torch.set_grad_enabled(False); torch.manual_seed(0); np.random.seed(0)
def log(x): print(f'[{time.strftime("%H:%M:%S")}] {x}',flush=True)
def H(m,**kw): return m.model(**kw,use_cache=False,return_dict=True).last_hidden_state
def collect(m,tok,texts):
 r=[]
 for t in texts:
  x=tok(t,return_tensors='pt',truncation=True,max_length=128); r.append(H(m,**x)[0,-1].float().cpu())
 return torch.stack(r)
def genstates(m,tok):
 ps=['Solve carefully: 137 * 29 =','The derivative of x^3 + 2x is','Write a Python function for binary search:','Factor 84 into prime factors:','Compute 2^10 =','Write SQL selecting users older than 18:','The gcd of 84 and 126 is','Simplify (x+2)(x-2):']
 seq=[tok(p,return_tensors='pt')['input_ids'] for p in ps]; hs=[]
 while len(hs)<N:
  ns=[]
  for ids in seq:
   h=H(m,input_ids=ids)[0,-1].float().cpu(); hs.append(h)
   if len(hs)>=N: break
   t=m.lm_head(h[None])[0].argmax().view(1,1); ns.append(torch.cat([ids,t],1))
  seq=ns
 return torch.stack(hs[:N])
def scores(h,w,chunk=4096): return torch.cat([h.float()@w[a:a+chunk].float().T for a in range(0,w.shape[0],chunk)],1)
def intervals(w):
 w16=w.half().contiguous(); b=w16.view(torch.int16).to(torch.int32)&65535; hi=(b>>8)&255
 a=(hi<<8).to(torch.int16).contiguous().view(torch.float16).float(); z=((hi<<8)|255).to(torch.int16).contiguous().view(torch.float16).float()
 if not (torch.isfinite(a).all() and torch.isfinite(z).all()): raise RuntimeError('nonfinite endpoint')
 return w16.float(),(a+z)/2,(z-a).abs()/2
def evaluate(h,w,w16,mid,rad):
 exact=scores(h,w16); fp=scores(h,w); coarse=scores(h,mid); err=scores(h.abs(),rad)
 pilot=coarse.argmax(1); B=exact.gather(1,pilot[:,None]).squeeze(1); cand=(coarse+err)>=B[:,None]
 ref=exact.argmax(1); pred=[]; cnt=[]
 for n in range(len(h)):
  idx=cand[n].nonzero().squeeze(1); cnt.append(int(len(idx))); pred.append(int(idx[exact[n,idx].argmax()]))
 c=np.array(cnt); f=c.mean()/w.shape[0]; bits=8+8*f
 return {'n':len(h),'exact_rate':float((torch.tensor(pred)==ref).float().mean()),'fp16_fp32_argmax_agreement':float((exact.argmax(1)==fp.argmax(1)).float().mean()),'candidate_mean':float(c.mean()),'candidate_fraction':float(f),'median':float(np.median(c)),'p90':float(np.percentile(c,90)),'p99':float(np.percentile(c,99)),'max':int(c.max()),'bits_per_weight':float(bits),'idealized_bw_reduction':float(16/bits)}
def main():
 log('load '+MODEL); tok=AutoTokenizer.from_pretrained(MODEL); m=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float32); m.eval(); w=m.lm_head.weight.detach().float().cpu().contiguous(); V,D=w.shape
 log('intervals'); w16,mid,rad=intervals(w); ds=load_dataset('ag_news',split='test'); ha=collect(m,tok,[ds[i]['text'] for i in range(N)]); hg=genstates(m,tok)
 a=evaluate(ha,w,w16,mid,rad); g=evaluate(hg,w,w16,mid,rad); log(str(a)); log(str(g))
 r={'kind':'fp16_highbyte_proofbits','model':MODEL,'vocab':V,'hidden_dim':D,'ag_news':a,'autoregressive_math_code':g,'extra_metadata_bytes':0,'reference':'FP16-rounded lm_head with FP32 accumulation','caveat':'Idealized byte traffic only; GPU timing and vendor-specific accumulation are not measured.'}; OUT.write_text(json.dumps(r,indent=2)); print(json.dumps(r,indent=2))
if __name__=='__main__': main()
