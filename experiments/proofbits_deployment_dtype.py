import gc,json,os,time
from pathlib import Path
import numpy as np, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM,AutoTokenizer
MODEL=os.environ.get('PB_MODEL','Qwen/Qwen2.5-0.5B-Instruct'); DTYPE_NAME=os.environ.get('PB_DTYPE','float16')
DTYPE={'float16':torch.float16,'bfloat16':torch.bfloat16}[DTYPE_NAME]
N=48;MAX_LEN=80;BATCH=2;CH=4096;PILOT=4
slug=MODEL.split('/')[-1].replace('.','_')+'_'+DTYPE_NAME
OUT=Path(f'experiments/artifacts/proofbits_deployment_dtype_{slug}.json');OUT.parent.mkdir(parents=True,exist_ok=True)
torch.set_grad_enabled(False);torch.manual_seed(61);np.random.seed(61)
def log(x):print(f'[{time.strftime("%H:%M:%S")}] {x}',flush=True)
def collect(m,tok):
 ds=load_dataset('wikitext','wikitext-2-raw-v1',split='test');hs=[]
 for r in ds:
  t=r['text'].strip()
  if len(t)<180:continue
  x=tok(t,return_tensors='pt',truncation=True,max_length=MAX_LEN)
  if x['input_ids'].shape[1]<20:continue
  H=m.model(**x,use_cache=False,return_dict=True).last_hidden_state[0].detach().cpu()
  for p in range(7,H.shape[0],9):
   hs.append(H[p])
   if len(hs)>=N:return torch.stack(hs)
 return torch.stack(hs)
def ep(w):
 raw=w.contiguous().view(torch.int16).to(torch.int32)&0xffff;hb=(raw>>8)&255
 a=(hb<<8).to(torch.int16).contiguous().view(torch.float16).float();b=((hb<<8)|255).to(torch.int16).contiguous().view(torch.float16).float()
 return torch.minimum(a,b),torch.maximum(a,b)
def calc(h,W):
 h=h.float();hp=torch.clamp(h,min=0);hn=torch.clamp(h,max=0);ex=[];up=[]
 for a in range(0,W.shape[0],CH):
  w=W[a:a+CH];lo,hi=ep(w);ex.append(h@w.float().T);up.append(hp@hi.T+hn@lo.T)
 return torch.cat(ex,1),torch.cat(up,1)
def main():
 log(f'load {MODEL} body={DTYPE_NAME}');tok=AutoTokenizer.from_pretrained(MODEL);m=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=DTYPE,low_cpu_mem_usage=True);m.eval();H=collect(m,tok);actual_hidden=str(H.dtype);W=m.lm_head.weight.detach().cpu().half().contiguous().clone();V,D=W.shape;del m;gc.collect();c=[];oks=[];hits=[]
 for s in range(0,len(H),BATCH):
  ex,U=calc(H[s:s+BATCH],W);pil=torch.topk(U,k=PILOT,dim=1).indices;B=ex.gather(1,pil).amax(1);mask=U>=B[:,None]
  for n in range(ex.shape[0]):
   idx=mask[n].nonzero().squeeze(1);ref=int(ex[n].argmax());pred=int(idx[ex[n,idx].argmax()]);c.append(int(idx.numel()));oks.append(pred==ref);hits.append(bool((pil[n]==ref).any()))
  log(f'{min(s+BATCH,len(H))}/{len(H)}')
 a=np.asarray(c,float);f=a.mean()/V;report={'kind':'proofbits_deployment_hidden_dtype','model':MODEL,'requested_body_dtype':DTYPE_NAME,'actual_hidden_dtype':actual_hidden,'head_storage_reference':'FP16-rounded','vocab':V,'hidden_dim':D,'n':len(H),'all_exact':bool(all(oks)),'pilot_hit_rate':float(np.mean(hits)),'mean':float(a.mean()),'median':float(np.median(a)),'p90':float(np.percentile(a,90)),'p99':float(np.percentile(a,99)),'max':int(a.max()),'fraction':float(f),'idealized_reduction':float(2/(1+f)),'caveat':'CPU model execution at requested dtype; exactness reference is FP16-rounded head with FP32 accumulation.'};OUT.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
