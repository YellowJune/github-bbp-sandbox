import json,time
from pathlib import Path
import numpy as np, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM,AutoTokenizer
MODEL='Qwen/Qwen2.5-0.5B-Instruct';N=64;PILOT=4;GS=[896,448,224,128]
OUT=Path('experiments/artifacts/proofbits_fp8_multinorm.json');OUT.parent.mkdir(parents=True,exist_ok=True)
torch.set_grad_enabled(False);torch.manual_seed(67);np.random.seed(67)
def log(x):print(f'[{time.strftime("%H:%M:%S")}] {x}',flush=True)
def H(m,**kw):return m.model(**kw,use_cache=False,return_dict=True).last_hidden_state
def collect_ag(m,t):
 d=load_dataset('ag_news',split='test');return torch.stack([H(m,**t(d[i]['text'],return_tensors='pt',truncation=True,max_length=128))[0,-1].float().cpu() for i in range(N)])
def collect_gen(m,t):
 ps=['Solve carefully: 137 * 29 =','The derivative of x^3 + 2x is','Write a Python function for binary search:','Factor 84 into prime factors:','Compute 2^10 =','Write SQL selecting users older than 18:','The gcd of 84 and 126 is','Simplify (x+2)(x-2):'];seq=[t(p,return_tensors='pt')['input_ids'] for p in ps];hs=[]
 while len(hs)<N:
  ns=[]
  for ids in seq:
   h=H(m,input_ids=ids)[0,-1].float().cpu();hs.append(h)
   if len(hs)>=N:break
   ns.append(torch.cat([ids,m.lm_head(h[None])[0].argmax().view(1,1)],1))
  seq=ns
 return torch.stack(hs[:N])
def score(h,w,ch=4096):return torch.cat([h@w[a:a+ch].T for a in range(0,len(w),ch)],1)
def split(W):
 W16=W.half().float().contiguous();b=W16.half().contiguous().view(torch.int16).to(torch.int32)&65535;hi=((b>>8)&255).to(torch.uint8);Q=(hi.to(torch.int16)<<8).contiguous().view(torch.float16).float();R=W16-Q;return W16,Q,R
def norms(x):
 return x.abs().sum(-1),torch.linalg.vector_norm(x,ord=2,dim=-1),x.abs().amax(-1)
def ev(h,ex,c,R,g,V,D):
 ng=D//g;rg=R.reshape(V,ng,g);rp=torch.clamp(rg,min=0);rn=torch.clamp(-rg,min=0)
 rp1,rp2,rpi=norms(rp);rn1,rn2,rni=norms(rn);hg=h.reshape(len(h),ng,g);hp=torch.clamp(hg,min=0);hn=torch.clamp(-hg,min=0);hp1,hp2,hpi=norms(hp);hn1,hn2,hni=norms(hn)
 # for each block/sign take min among three Holder dual pairs
 Ep=torch.minimum(torch.minimum(torch.einsum('ng,vg->nv',hpi,rp1),torch.einsum('ng,vg->nv',hp2,rp2)),torch.einsum('ng,vg->nv',hp1,rpi))
 En=torch.minimum(torch.minimum(torch.einsum('ng,vg->nv',hni,rn1),torch.einsum('ng,vg->nv',hn2,rn2)),torch.einsum('ng,vg->nv',hn1,rni))
 U=c+Ep+En;p=torch.topk(U,k=PILOT,dim=1).indices;B=ex.gather(1,p).amax(1);mask=U>=B[:,None];ref=ex.argmax(1);cc=[];ok=[]
 for n in range(len(h)):
  ix=mask[n].nonzero().squeeze(1);cc.append(len(ix));ok.append(int(ix[ex[n,ix].argmax()])==int(ref[n]))
 a=np.array(cc);f=a.mean()/V;meta=24*ng/D;tot=1+meta+f;meta16=12*ng/D;tot16=1+meta16+f
 return {'group_size':g,'groups':ng,'exact':bool(all(ok)),'candidate_mean':float(a.mean()),'p90':float(np.percentile(a,90)),'p99':float(np.percentile(a,99)),'max':int(a.max()),'candidate_fraction':float(f),'fp32_metadata_bytes_per_weight':float(meta),'reduction_fp32meta':float(2/tot),'hypothetical_fp16_metadata_bytes_per_weight':float(meta16),'hypothetical_reduction_fp16meta':float(2/tot16)}
def main():
 log('load');t=AutoTokenizer.from_pretrained(MODEL);m=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float32);m.eval();W=m.lm_head.weight.detach().float().cpu();V,D=W.shape;ha=collect_ag(m,t);hg=collect_gen(m,t);del m;W16,Q,R=split(W);ea=score(ha,W16);ca=score(ha,Q);eg=score(hg,W16);cg=score(hg,Q);r={'kind':'e5m2_signsplit_multi_holder','model':MODEL,'domains':{'ag':[],'gen':[]}}
 for g in GS:
  if D%g:continue
  log('g='+str(g));a=ev(ha,ea,ca,R,g,V,D);b=ev(hg,eg,cg,R,g,V,D);r['domains']['ag'].append(a);r['domains']['gen'].append(b);log(str(a));log(str(b))
 r['caveat']='Exact FP32 norm metadata in CPU kill-test; compressed/outward metadata and FP8 GEMM rounding not yet certified.';OUT.write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
if __name__=='__main__':main()
