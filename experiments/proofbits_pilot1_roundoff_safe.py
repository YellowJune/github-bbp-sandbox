import json,time
from pathlib import Path
import numpy as np, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM,AutoTokenizer
MODEL='Qwen/Qwen2.5-0.5B-Instruct';N_NAT=192;N_BASE=64;N_BOUND=32;MAX_LEN=128;BATCH=8;U32=2.0**-24
OUT=Path('experiments/artifacts/proofbits_pilot1_roundoff_safe.json');OUT.parent.mkdir(parents=True,exist_ok=True);torch.set_grad_enabled(False);torch.manual_seed(89);np.random.seed(89)
def log(x):print(f'[{time.strftime("%H:%M:%S")}] {x}',flush=True)
def hidden(m,**kw):return m.model(**kw,use_cache=False,return_dict=True).last_hidden_state
def endpoints(W):
 w=W.half().contiguous();raw=w.view(torch.int16).to(torch.int32)&0xffff;hb=(raw>>8)&255;a=(hb<<8).to(torch.int16).contiguous().view(torch.float16).float();b=((hb<<8)|255).to(torch.int16).contiguous().view(torch.float16).float();lo=torch.minimum(a,b);hi=torch.maximum(a,b);rowmax=torch.maximum(lo.abs(),hi.abs()).amax(1);return w.float(),lo,hi,rowmax
def score(h,W,ch=4096):return torch.cat([h.float()@W[a:a+ch].T for a in range(0,len(W),ch)],1)
def upper(h,lo,hi,ch=4096):
 hp=torch.clamp(h.float(),min=0);hn=torch.clamp(h.float(),max=0);return torch.cat([hp@hi[a:a+ch].T+hn@lo[a:a+ch].T for a in range(0,len(lo),ch)],1)
def gamma(n):x=n*U32;return x/(1-x)
def safe(h,U,rowmax,D):return U+(2*gamma(4*D))*h.float().abs().sum(1)[:,None]*rowmax[None,:]
def collect_nat(m,tok):
 ds=load_dataset('wikitext','wikitext-2-raw-v1',split='test');hs=[]
 for r in ds:
  t=r['text'].strip()
  if len(t)<200:continue
  x=tok(t,return_tensors='pt',truncation=True,max_length=MAX_LEN)
  if x['input_ids'].shape[1]<20:continue
  H=hidden(m,**x)[0].float().cpu()
  for p in range(5,H.shape[0],6):
   hs.append(H[p])
   if len(hs)>=N_NAT:return torch.stack(hs)
 return torch.stack(hs)
def collect_base(m,tok):
 ag=load_dataset('ag_news',split='test');texts=[ag[i]['text'] for i in range(N_BASE)];hs=[]
 for t in texts:
  x=tok(t,return_tensors='pt',truncation=True,max_length=MAX_LEN);H=hidden(m,**x)[0].float().cpu();hs.append(H[-1])
 return torch.stack(hs)
def evalH(H,W,lo,hi,rowmax):
 V,D=W.shape;c=[];oks=[];hits=[]
 for s in range(0,len(H),BATCH):
  h=H[s:s+BATCH];ex=score(h,W);Us=safe(h,upper(h,lo,hi),rowmax,D);p=Us.argmax(1,keepdim=True);B=ex.gather(1,p).squeeze(1);mask=Us>=B[:,None]
  for n in range(len(h)):
   idx=mask[n].nonzero().squeeze(1);ref=int(ex[n].argmax());pred=int(idx[ex[n,idx].argmax()]);c.append(int(idx.numel()));oks.append(pred==ref);hits.append(int(p[n])==ref)
 a=np.asarray(c,float);f=a.mean()/V;return {'n':len(H),'all_exact':bool(all(oks)),'pilot_hit_rate':float(np.mean(hits)),'mean':float(a.mean()),'median':float(np.median(a)),'p90':float(np.percentile(a,90)),'p99':float(np.percentile(a,99)),'max':int(a.max()),'fraction':float(f),'idealized_reduction':float(2/(1+f))}
def boundary(H,Z,W,lo,hi,rowmax):
 Y=Z.argmax(1);pairs=[]
 for i in range(len(H)):
  for j in range(i+1,len(H)):
   if int(Y[i])!=int(Y[j]):pairs.append((i,j))
   if len(pairs)>=N_BOUND:break
  if len(pairs)>=N_BOUND:break
 c=[];oks=[];marg=[];D=W.shape[1]
 for i,j in pairs:
  l,r=0.,1.;yi=int(Y[i])
  for _ in range(48):
   a=(l+r)/2;y=int(((1-a)*Z[i]+a*Z[j]).argmax())
   if y==yi:l=a
   else:r=a
  a=(l+r)/2;h=((1-a)*H[i]+a*H[j])[None];ex=((1-a)*Z[i]+a*Z[j])[None]
  Us=safe(h,upper(h,lo,hi),rowmax,D);p=Us.argmax(1,keepdim=True);B=ex.gather(1,p).squeeze(1);idx=(Us[0]>=B[0]).nonzero().squeeze(1);ref=int(ex[0].argmax());pred=int(idx[ex[0,idx].argmax()]);c.append(len(idx));oks.append(pred==ref);tmp=ex[0].clone();tmp[ref]=-torch.inf;marg.append(float(ex[0,ref]-tmp.max()))
 a=np.asarray(c,float);f=a.mean()/W.shape[0];return {'n':len(c),'all_exact':bool(all(oks)),'mean':float(a.mean()),'median':float(np.median(a)),'p90':float(np.percentile(a,90)),'p99':float(np.percentile(a,99)),'max':int(a.max()),'fraction':float(f),'idealized_reduction':float(2/(1+f)),'margin_min':float(np.min(marg)),'margin_median':float(np.median(marg))}
def main():
 tok=AutoTokenizer.from_pretrained(MODEL);m=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float32);m.eval();Hn=collect_nat(m,tok);Hb=collect_base(m,tok);W,lo,hi,rowmax=endpoints(m.lm_head.weight.detach().float().cpu());Z=score(Hb,W);del m
 nat=evalH(Hn,W,lo,hi,rowmax);bnd=boundary(Hb,Z,W,lo,hi,rowmax);report={'kind':'proofbits_pilot1_roundoff_safe','model':MODEL,'vocab':W.shape[0],'hidden_dim':W.shape[1],'pilot_k':1,'rounding_slack':'2*gamma_{4d}*||h||_1*M_i, conservative CPU envelope','natural':nat,'decision_boundary':bnd,'caveat':'This is a conservative FP32-accumulation envelope, not yet a theorem for a specific compiled GPU reduction tree. GPU latency/counters absent.'};OUT.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
