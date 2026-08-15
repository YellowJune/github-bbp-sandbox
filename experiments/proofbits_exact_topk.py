import json, time
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL='Qwen/Qwen2.5-0.5B-Instruct'
KS=[1,5,10,50]
N=192
MAX_LEN=128
BATCH=6
OUT=Path('experiments/artifacts/proofbits_exact_topk.json'); OUT.parent.mkdir(parents=True,exist_ok=True)
torch.set_grad_enabled(False); torch.manual_seed(31); np.random.seed(31)

def log(x): print(f'[{time.strftime("%H:%M:%S")}] {x}',flush=True)
def hidden(m,**kw): return m.model(**kw,use_cache=False,return_dict=True).last_hidden_state

def fp16_endpoints(w):
    w16=w.half().contiguous(); raw=w16.view(torch.int16).to(torch.int32)&0xffff; hb=(raw>>8)&255
    a=(hb<<8).to(torch.int16).contiguous().view(torch.float16).float()
    b=((hb<<8)|255).to(torch.int16).contiguous().view(torch.float16).float()
    if not(torch.isfinite(a).all() and torch.isfinite(b).all()): raise RuntimeError('nonfinite interval')
    return w16.float().contiguous(), torch.minimum(a,b).contiguous(), torch.maximum(a,b).contiguous()

def score(h,w,ch=4096):
    return torch.cat([h.float()@w[a:a+ch].float().T for a in range(0,w.shape[0],ch)],1)
def upper(h,lo,hi,ch=4096):
    hp=torch.clamp(h.float(),min=0); hn=torch.clamp(h.float(),max=0); ys=[]
    for a in range(0,lo.shape[0],ch): ys.append(hp@hi[a:a+ch].T + hn@lo[a:a+ch].T)
    return torch.cat(ys,1)

def collect(m,tok):
    ds=load_dataset('wikitext','wikitext-2-raw-v1',split='test'); hs=[]
    docs=[x['text'] for x in ds if len(x['text'].strip())>220]
    for t in docs:
        x=tok(t,return_tensors='pt',truncation=True,max_length=MAX_LEN)
        if x['input_ids'].shape[1]<24: continue
        H=hidden(m,**x)[0].float().cpu()
        for p in range(5,H.shape[0],5):
            hs.append(H[p])
            if len(hs)>=N: return torch.stack(hs)
    return torch.stack(hs)

def same_set(a,b): return set(map(int,a.tolist()))==set(map(int,b.tolist()))

def eval_k(exact,U,k):
    # 4k pilot rows are exact-refined; kth pilot score is a safe lower bound on global kth score.
    pk=min(U.shape[1],max(4*k,4))
    pilots=torch.topk(U,k=pk,dim=1).indices
    pexact=exact.gather(1,pilots)
    B=torch.topk(pexact,k=k,dim=1).values[:,-1]
    mask=U>=B[:,None]
    counts=[]; oks=[]; true_covered=[]
    for n in range(exact.shape[0]):
        idx=mask[n].nonzero().squeeze(1)
        pred_local=torch.topk(exact[n,idx],k=k).indices
        pred=idx[pred_local]
        ref=torch.topk(exact[n],k=k).indices
        counts.append(int(idx.numel()))
        oks.append(same_set(pred,ref))
        true_covered.append(all(bool((idx==r).any()) for r in ref))
    return counts,oks,true_covered,pk

def main():
    log('load'); tok=AutoTokenizer.from_pretrained(MODEL); m=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float32); m.eval()
    H=collect(m,tok); W=m.lm_head.weight.detach().float().cpu().contiguous(); V,D=W.shape; del m
    log(f'endpoints V={V} D={D} N={len(H)}'); W16,lo,hi=fp16_endpoints(W); del W
    acc={k:{'counts':[],'oks':[],'covered':[]} for k in KS}
    for s in range(0,len(H),BATCH):
        hb=H[s:s+BATCH]; ex=score(hb,W16); U=upper(hb,lo,hi)
        for k in KS:
            c,o,cv,pk=eval_k(ex,U,k); acc[k]['counts']+=c; acc[k]['oks']+=o; acc[k]['covered']+=cv; acc[k]['pilot_k']=pk
        log(f'{min(s+BATCH,len(H))}/{len(H)}')
    res={}
    for k in KS:
        c=np.asarray(acc[k]['counts'],float); f=c.mean()/V
        res[str(k)]={'k':k,'pilot_k':acc[k]['pilot_k'],'n':len(c),'all_exact_topk':bool(all(acc[k]['oks'])),'all_true_topk_survive':bool(all(acc[k]['covered'])),
          'candidate_mean':float(c.mean()),'candidate_median':float(np.median(c)),'candidate_p90':float(np.percentile(c,90)),'candidate_p99':float(np.percentile(c,99)),'candidate_max':int(c.max()),
          'candidate_fraction_mean':float(f),'idealized_weight_byte_reduction':float(2/(1+f))}
    report={'kind':'proofbits_exact_topk','model':MODEL,'vocab':V,'hidden_dim':D,'n':len(H),'results':res,
      'interpretation':'High byte is read for all rows; low byte is read only for certified survivors. Returned top-k rows and their FP16 scores are exact.',
      'caveat':'Idealized weight-byte traffic only; GPU wall-clock/counters not measured. Exact top-k enables exact top-k sampling after refinement, but does not by itself certify full softmax/top-p.'}
    OUT.write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
