import gc,json,os,time
from pathlib import Path
import numpy as np, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM,AutoTokenizer
MODEL=os.environ.get('PB_MODEL','Qwen/Qwen2.5-0.5B-Instruct');N=int(os.environ.get('PB_N','64'));MAX_LEN=96;BATCH=2;CH=4096;PILOT=4
slug=MODEL.split('/')[-1].replace('.','_');OUT=Path(f'experiments/artifacts/proofbits_fp8e5m2_rowradius_{slug}.json');OUT.parent.mkdir(parents=True,exist_ok=True)
torch.set_grad_enabled(False);torch.manual_seed(67);np.random.seed(67)
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
def q_and_rmax(w16):
 raw=w16.contiguous().view(torch.int16).to(torch.int32)&0xffff;hb=(raw>>8)&255
 q=(hb<<8).to(torch.int16).contiguous().view(torch.float16).float()
 far=((hb<<8)|255).to(torch.int16).contiguous().view(torch.float16).float()
 if not(torch.isfinite(q).all() and torch.isfinite(far).all()):raise RuntimeError('nonfinite FP16 prefix endpoint in model head')
 rmax=(far-q).abs().amax(1)
 return q.contiguous(),rmax.contiguous(),hb.to(torch.uint8).contiguous()
def calc(h,W,Q,rmax):
 ex=[];co=[]
 for a in range(0,W.shape[0],CH):
  ex.append(h.float()@W[a:a+CH].float().T);co.append(h.float()@Q[a:a+CH].float().T)
 exact=torch.cat(ex,1);coarse=torch.cat(co,1);U=coarse+h.float().abs().sum(1,keepdim=True)*rmax[None,:]
 return exact,U
def fp8_equiv_check():
 # For every finite high-byte code, FP16(high_byte<<8) must equal interpreting the same byte as E5M2.
 if not hasattr(torch,'float8_e5m2'): return {'available':False}
 b=torch.arange(256,dtype=torch.uint8); e=b.view(torch.float8_e5m2).float(); h=(b.to(torch.int16)<<8).contiguous().view(torch.float16).float(); finite=torch.isfinite(e)&torch.isfinite(h)
 return {'available':True,'finite_codes':int(finite.sum()),'all_equal_finite':bool(torch.equal(e[finite],h[finite])),'max_abs_diff_finite':float((e[finite]-h[finite]).abs().max())}
def main():
 equiv=fp8_equiv_check();log(f'e5m2 equivalence {equiv}');tok=AutoTokenizer.from_pretrained(MODEL);m=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float32,low_cpu_mem_usage=True);m.eval();H=collect(m,tok);W=m.lm_head.weight.detach().cpu().half().contiguous().clone();V,D=W.shape;del m;gc.collect();log(f'prepare Q/r V={V} D={D}');Q,rmax,hb=q_and_rmax(W)
 c=[];oks=[];hits=[];slacks=[]
 for s in range(0,len(H),BATCH):
  ex,U=calc(H[s:s+BATCH],W,Q,rmax);pil=torch.topk(U,k=PILOT,dim=1).indices;B=ex.gather(1,pil).amax(1);mask=U>=B[:,None]
  for n in range(ex.shape[0]):
   idx=mask[n].nonzero().squeeze(1);ref=int(ex[n].argmax());pred=int(idx[ex[n,idx].argmax()]);c.append(int(idx.numel()));oks.append(pred==ref);hits.append(bool((pil[n]==ref).any()));slacks.append(float(U[n,ref]-ex[n,ref]))
  log(f'{min(s+BATCH,len(H))}/{len(H)} mean={np.mean(c):.1f}')
 a=np.asarray(c,float);f=a.mean()/V;rm=rmax.numpy();report={'kind':'proofbits_native_fp8e5m2_rowradius','model':MODEL,'vocab':V,'hidden_dim':D,'n':len(H),'pilot_k':PILOT,'fp8_raw_encoding_equivalence':equiv,'all_exact':bool(all(oks)),'pilot_hit_rate':float(np.mean(hits)),'candidate_mean':float(a.mean()),'median':float(np.median(a)),'p90':float(np.percentile(a,90)),'p99':float(np.percentile(a,99)),'max':int(a.max()),'fraction':float(f),'idealized_fp16_weight_byte_reduction':float(2/(1+f)),'row_radius_mean':float(rm.mean()),'row_radius_max':float(rm.max()),'winner_upper_slack_mean':float(np.mean(slacks)),'metadata_bytes_if_fp16_radius_per_row':int(2*V),'metadata_fraction_vs_fp16_head':float((2*V)/(2*V*D)),'hardware_note':'The stored FP16 high byte is the E5M2 sign/exponent/top-2-mantissa code. A native E5M2 GEMM can form h^T q on supported GPUs; one row-radius scalar gives a symmetric suffix error bound. Actual GPU latency not measured.'};OUT.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
