import gc,json,time
from pathlib import Path
import numpy as np,torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM,AutoTokenizer
MODEL='Qwen/Qwen2.5-0.5B-Instruct';N=128;MAX_LEN=128;BATCH=4;CH=4096
OUT=Path('experiments/artifacts/proofbits_query_geometry_controls.json');OUT.parent.mkdir(parents=True,exist_ok=True);torch.set_grad_enabled(False);torch.manual_seed(97);np.random.seed(97)
def log(x):print(f'[{time.strftime("%H:%M:%S")}] {x}',flush=True)
def collect(m,tok):
 ds=load_dataset('wikitext','wikitext-2-raw-v1',split='test');hs=[]
 for r in ds:
  t=r['text'].strip()
  if len(t)<220:continue
  x=tok(t,return_tensors='pt',truncation=True,max_length=MAX_LEN)
  if x['input_ids'].shape[1]<24:continue
  H=m.model(**x,use_cache=False,return_dict=True).last_hidden_state[0].float().cpu()
  for p in range(5,H.shape[0],7):
   hs.append(H[p])
   if len(hs)>=N:return torch.stack(hs)
 return torch.stack(hs)
def ep(w):
 raw=w.contiguous().view(torch.int16).to(torch.int32)&0xffff;hb=(raw>>8)&255;a=(hb<<8).to(torch.int16).contiguous().view(torch.float16).float();b=((hb<<8)|255).to(torch.int16).contiguous().view(torch.float16).float();return torch.minimum(a,b),torch.maximum(a,b)
def evalH(H,W):
 V,D=W.shape;counts=[];marg=[];pilot=[]
 for s in range(0,len(H),BATCH):
  h=H[s:s+BATCH].float();hp=torch.clamp(h,min=0);hn=torch.clamp(h,max=0);ex=[];up=[]
  for a in range(0,V,CH):
   w=W[a:a+CH];lo,hi=ep(w);ex.append(h@w.float().T);up.append(hp@hi.T+hn@lo.T)
  ex=torch.cat(ex,1);U=torch.cat(up,1);p=U.argmax(1,keepdim=True);Bv=ex.gather(1,p).squeeze(1);mask=U>=Bv[:,None]
  for n in range(len(h)):
   ref=int(ex[n].argmax());counts.append(int(mask[n].sum()));pilot.append(int(p[n])==ref);tmp=ex[n].clone();tmp[ref]=-torch.inf;marg.append(float(ex[n,ref]-tmp.max()))
 a=np.asarray(counts,float);mm=np.asarray(marg,float);f=a.mean()/V;return {'mean':float(a.mean()),'median':float(np.median(a)),'p90':float(np.percentile(a,90)),'p99':float(np.percentile(a,99)),'max':int(a.max()),'fraction':float(f),'idealized_reduction':float(2/(1+f)),'pilot_hit_rate':float(np.mean(pilot)),'margin_mean':float(mm.mean()),'margin_median':float(np.median(mm)),'margin_min':float(mm.min())}
def main():
 tok=AutoTokenizer.from_pretrained(MODEL);m=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float32);m.eval();H=collect(m,tok);W=m.lm_head.weight.detach().cpu().half().contiguous().clone();D=H.shape[1];del m;gc.collect()
 g=torch.Generator().manual_seed(9701);perm=torch.randperm(D,generator=g);sign=torch.where(torch.rand((N,D),generator=g)>0.5,1.0,-1.0)
 # Preserve per-query norm for all controls.
 Hperm=H[:,perm].clone();Hsign=H*sign
 mean=H.mean(0);std=H.std(0,unbiased=False).clamp_min(1e-6);Hg=mean[None,:]+torch.randn((N,D),generator=g)*std[None,:]
 target=H.norm(dim=1,keepdim=True);Hg=Hg/Hg.norm(dim=1,keepdim=True).clamp_min(1e-9)*target
 Hiso=torch.randn((N,D),generator=g);Hiso=Hiso/Hiso.norm(dim=1,keepdim=True)*target
 tests={'natural':H,'coordinate_permuted':Hperm,'random_sign':Hsign,'diag_gaussian_matched':Hg,'isotropic_norm_matched':Hiso};res={}
 for name,X in tests:log(name);res[name]=evalH(X,W)
 report={'kind':'proofbits_query_geometry_controls','model':MODEL,'vocab':W.shape[0],'hidden_dim':D,'n':N,'pilot_k':1,'results':res,'caveat':'Controls are synthetic queries evaluated against the real trained output head. They diagnose geometry; they are not model-generated states or a performance benchmark.'};OUT.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
