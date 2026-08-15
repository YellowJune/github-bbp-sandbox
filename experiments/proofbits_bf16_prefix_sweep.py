import json, time
from pathlib import Path
import numpy as np, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
MODEL='Qwen/Qwen2.5-0.5B-Instruct'; N=48; BITS=range(8,16)
OUT=Path('experiments/artifacts/proofbits_bf16_prefix_sweep.json'); OUT.parent.mkdir(parents=True,exist_ok=True)
torch.set_grad_enabled(False); torch.manual_seed(17); np.random.seed(17)
def log(x): print(f'[{time.strftime("%H:%M:%S")}] {x}',flush=True)
def hidden(m,**kw): return m.model(**kw,use_cache=False,return_dict=True).last_hidden_state
def collect(m,tok,texts):
 hs=[]
 for t in texts:
  x=tok(t,return_tensors='pt',truncation=True,max_length=128); hs.append(hidden(m,**x)[0,-1].float().cpu())
 return torch.stack(hs)
def gen(m,tok):
 ps=['Solve: 271*43 =','Differentiate sin(x)*x^2:','Implement merge sort in Python:','Prime factorization of 2310:','Compute 19 choose 4:','Write SQL group by country:','Solve 3x+7=31:','Implement DFS recursively:']
 seq=[tok(p,return_tensors='pt')['input_ids'] for p in ps]; hs=[]
 while len(hs)<N:
  ns=[]
  for ids in seq:
   h=hidden(m,input_ids=ids)[0,-1].float().cpu(); hs.append(h)
   if len(hs)>=N: break
   t=m.lm_head(h[None])[0].argmax().view(1,1); ns.append(torch.cat([ids,t],1))
  seq=ns
 return torch.stack(hs[:N])
def score(h,w,ch=4096): return torch.cat([h.float()@w[a:a+ch].float().T for a in range(0,len(w),ch)],1)
def interval_bf(w,b):
 wb=w.to(torch.bfloat16).contiguous(); raw=wb.view(torch.int16).to(torch.int32)&0xffff; suffix=16-b; keep=((1<<16)-1)^((1<<suffix)-1); pre=raw&keep
 a=pre.to(torch.int16).contiguous().view(torch.bfloat16).float(); z=(pre|((1<<suffix)-1)).to(torch.int16).contiguous().view(torch.bfloat16).float()
 if not (torch.isfinite(a).all() and torch.isfinite(z).all()): return wb.float(),None,None
 return wb.float(),(a+z)*.5,(z-a).abs()*.5
def eval_b(h,exact,b,mid,rad):
 c=score(h,mid); e=score(h.abs(),rad); p=c.argmax(1); B=exact.gather(1,p[:,None]).squeeze(1); mask=c+e>=B[:,None]; ref=exact.argmax(1); cnt=[]; ok=[]
 for n in range(len(h)):
  ix=mask[n].nonzero().squeeze(1); cnt.append(len(ix)); ok.append(int(ix[exact[n,ix].argmax()])==int(ref[n]))
 a=np.array(cnt); f=a.mean()/exact.shape[1]; traffic=b+(16-b)*f
 return {'prefix_bits':b,'exact_rate':float(np.mean(ok)),'candidate_mean':float(a.mean()),'median':float(np.median(a)),'p90':float(np.percentile(a,90)),'p99':float(np.percentile(a,99)),'max':int(a.max()),'fraction':float(f),'bits_per_weight':float(traffic),'idealized_bw_reduction':float(16/traffic)}
def main():
 log('load'); tok=AutoTokenizer.from_pretrained(MODEL); m=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float32); m.eval(); w=m.lm_head.weight.detach().float().cpu(); V,D=w.shape
 ag=load_dataset('ag_news',split='test'); ha=collect(m,tok,[ag[i]['text'] for i in range(N)]); hg=gen(m,tok); wb=w.to(torch.bfloat16).float(); ea=score(ha,wb); eg=score(hg,wb); res={'ag_news':[],'autoregressive_math_code':[]}
 for b in BITS:
  log('b='+str(b)); _,mid,rad=interval_bf(w,b)
  if mid is None: res['ag_news'].append({'prefix_bits':b,'invalid_interval_due_nonfinite_endpoint':True}); res['autoregressive_math_code'].append({'prefix_bits':b,'invalid_interval_due_nonfinite_endpoint':True}); continue
  ra=eval_b(ha,ea,b,mid,rad); rg=eval_b(hg,eg,b,mid,rad); res['ag_news'].append(ra); res['autoregressive_math_code'].append(rg); log(str(ra)); log(str(rg))
 out={'kind':'proofbits_exact_bf16_prefix_sweep','model':MODEL,'reference':'BF16-rounded lm_head with FP32 accumulation','vocab':V,'hidden_dim':D,'results':res,'caveat':'Idealized prefix/suffix bit traffic; bit-aligned packing for non-byte prefix sizes is not yet implemented or timed.'}; OUT.write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
