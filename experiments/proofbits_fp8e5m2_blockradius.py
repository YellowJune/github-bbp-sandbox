import gc,json,os,time
from pathlib import Path
import numpy as np, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM,AutoTokenizer
MODEL=os.environ.get('PB_MODEL','Qwen/Qwen2.5-0.5B-Instruct');N=int(os.environ.get('PB_N','48'));MAX_LEN=96;BATCH=2;CH=4096;PILOT=4
GROUP_COUNTS=[4,8,16,32,64]
slug=MODEL.split('/')[-1].replace('.','_');OUT=Path(f'experiments/artifacts/proofbits_fp8e5m2_blockradius_{slug}.json');OUT.parent.mkdir(parents=True,exist_ok=True)
torch.set_grad_enabled(False);torch.manual_seed(71);np.random.seed(71)
def log(x):print(f'[{time.strftime("%H:%M:%S")}] {x}',flush=True)
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
def make_q(w16):
 raw=w16.contiguous().view(torch.int16).to(torch.int32)&0xffff;hb=(raw>>8)&255
 q=(hb<<8).to(torch.int16).contiguous().view(torch.float16).float();far=((hb<<8)|255).to(torch.int16).contiguous().view(torch.float16).float()
 if not(torch.isfinite(q).all() and torch.isfinite(far).all()):raise RuntimeError('nonfinite prefix')
 return q.contiguous(),(far-q).abs().contiguous()
def block_radii(err,gc):
 V,D=err.shape; edges=torch.linspace(0,D,gc+1,dtype=torch.int64); rs=[]
 for g in range(gc):rs.append(err[:,edges[g]:edges[g+1]].amax(1))
 return torch.stack(rs,1).contiguous(),edges
def exact_and_coarse(h,W,Q):
 ex=[];co=[]
 for a in range(0,W.shape[0],CH):ex.append(h.float()@W[a:a+CH].float().T);co.append(h.float()@Q[a:a+CH].float().T)
 return torch.cat(ex,1),torch.cat(co,1)
def evaluate(ex,co,H,R,edges):
 # H [B,D], R [V,G]
 norms=[]
 for g in range(len(edges)-1):norms.append(H[:,edges[g]:edges[g+1]].float().abs().sum(1))
 hn=torch.stack(norms,1); corr=hn@R.T; U=co+corr
 pil=torch.topk(U,k=PILOT,dim=1).indices;B=ex.gather(1,pil).amax(1);mask=U>=B[:,None]
 c=[];oks=[];hits=[];sl=[]
 for n in range(ex.shape[0]):
  idx=mask[n].nonzero().squeeze(1);ref=int(ex[n].argmax());pred=int(idx[ex[n,idx].argmax()]);c.append(int(idx.numel()));oks.append(pred==ref);hits.append(bool((pil[n]==ref).any()));sl.append(float(U[n,ref]-ex[n,ref]))
 return c,oks,hits,sl
def main():
 tok=AutoTokenizer.from_pretrained(MODEL);m=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float32,low_cpu_mem_usage=True);m.eval();H=collect(m,tok);W=m.lm_head.weight.detach().cpu().half().contiguous().clone();V,D=W.shape;del m;gc.collect();Q,err=make_q(W);log(f'V={V} D={D} N={len(H)}')
 res={}
 for gcnt in GROUP_COUNTS:
  R,edges=block_radii(err,gcnt);c=[];oks=[];hits=[];sl=[]
  for s in range(0,len(H),BATCH):
   ex,co=exact_and_coarse(H[s:s+BATCH],W,Q);cc,oo,hh,ss=evaluate(ex,co,H[s:s+BATCH],R,edges);c+=cc;oks+=oo;hits+=hh;sl+=ss
  a=np.asarray(c,float);f=a.mean()/V;meta=2*V*gcnt;head=2*V*D
  res[str(gcnt)]={'groups':gcnt,'all_exact':bool(all(oks)),'pilot_hit_rate':float(np.mean(hits)),'candidate_mean':float(a.mean()),'median':float(np.median(a)),'p90':float(np.percentile(a,90)),'p99':float(np.percentile(a,99)),'max':int(a.max()),'fraction':float(f),'idealized_fp16_weight_byte_reduction':float(2/(1+f)),'winner_upper_slack_mean':float(np.mean(sl)),'metadata_bytes_fp16_radii':int(meta),'metadata_fraction_vs_fp16_head':float(meta/head),'effective_bytes_per_weight_including_metadata_and_lowbyte_refine':float(1+2*gcnt/D+f)}
  log(f'groups={gcnt} mean={a.mean():.1f} frac={f:.4%} slack={np.mean(sl):.3f}')
 report={'kind':'proofbits_fp8e5m2_blockradius','model':MODEL,'vocab':V,'hidden_dim':D,'n':len(H),'results':res,'caveat':'Coarse dot is numerically E5M2-code-equivalent but evaluated in FP32 on CPU. Block-radius metadata is modeled as FP16. GPU latency not measured.'};OUT.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
