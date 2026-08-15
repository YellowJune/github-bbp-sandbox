import json, os, time, gc
from pathlib import Path
import numpy as np, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
MODEL=os.getenv('MODEL','Qwen/Qwen2.5-0.5B-Instruct'); N=48; BITS=range(8,16)
OUT=Path('experiments/artifacts')/('fp16_sweep_'+MODEL.replace('/','__')+'.json'); OUT.parent.mkdir(parents=True,exist_ok=True)
torch.set_grad_enabled(False); torch.manual_seed(1); np.random.seed(1)
def log(x): print(f'[{time.strftime("%H:%M:%S")}] {x}',flush=True)
def hidden(m,**kw): return m.model(**kw,use_cache=False,return_dict=True).last_hidden_state
def collect(m,tok,txt):
 r=[]
 for t in txt:
  x=tok(t,return_tensors='pt',truncation=True,max_length=128); r.append(hidden(m,**x)[0,-1].float().cpu())
 return torch.stack(r)
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
def score(h,w,ch=4096): return torch.cat([h@w[a:a+ch].T for a in range(0,len(w),ch)],1)
def interval(w16,b):
 raw=w16.contiguous().view(torch.int16).to(torch.int32)&65535; suffix=16-b; mask=((1<<16)-1)^((1<<suffix)-1); pre=raw&mask
 x=pre.to(torch.int16).contiguous().view(torch.float16).float(); y=(pre|((1<<suffix)-1)).to(torch.int16).contiguous().view(torch.float16).float()
 if not (torch.isfinite(x).all() and torch.isfinite(y).all()): raise RuntimeError('nonfinite')
 return (x+y)*.5,(y-x).abs()*.5
def eval_b(h,exact,b,mid,rad):
 c=score(h,mid); e=score(h.abs(),rad); pilot=c.argmax(1); B=exact.gather(1,pilot[:,None]).squeeze(1); mask=c+e>=B[:,None]
 ref=exact.argmax(1); cnt=[]; pred=[]
 for i in range(len(h)):
  ix=mask[i].nonzero().squeeze(1); cnt.append(len(ix)); pred.append(int(ix[exact[i,ix].argmax()]))
 a=np.array(cnt); f=a.mean()/exact.shape[1]; bits=b+(16-b)*f
 return {'prefix_bits':b,'exact_rate':float((torch.tensor(pred)==ref).float().mean()),'candidate_mean':float(a.mean()),'candidate_fraction':float(f),'median':float(np.median(a)),'p90':float(np.percentile(a,90)),'p99':float(np.percentile(a,99)),'max':int(a.max()),'bits_per_weight':float(bits),'idealized_bw_reduction':float(16/bits)}
def main():
 log('load '+MODEL); tok=AutoTokenizer.from_pretrained(MODEL); m=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float32); m.eval(); w=m.lm_head.weight.detach().float().cpu(); w16=w.half().float().contiguous(); V,D=w.shape
 ds=load_dataset('ag_news',split='test'); ha=collect(m,tok,[ds[i]['text'] for i in range(64,64+N)]); hg=gen(m,tok); ea=score(ha,w16); eg=score(hg,w16); res={'ag_news':[],'autoregressive_math_code':[]}
 for b in BITS:
  log('b='+str(b)); mid,rad=interval(w16.half(),b); ra=eval_b(ha,ea,b,mid,rad); rg=eval_b(hg,eg,b,mid,rad); res['ag_news'].append(ra); res['autoregressive_math_code'].append(rg); log('AG '+str(ra)); log('GEN '+str(rg)); del mid,rad; gc.collect()
 out={'kind':'exact_fp16_prefix_sweep','model':MODEL,'vocab':V,'hidden_dim':D,'prefix_bits':list(BITS),'results':res,'extra_metadata_bytes':0,'traffic_model':'b bits for all weights plus remaining 16-b bits only for certified survivors','caveat':'FP16-rounded head, FP32 accumulation; idealized bit traffic, no GPU wall-clock.'}; OUT.write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
