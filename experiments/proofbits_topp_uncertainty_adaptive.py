import json, time
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL='Qwen/Qwen2.5-0.5B-Instruct'
N=24; MAX_LEN=128; CH=4096
SETTINGS=[(0.7,0.90),(0.7,0.95),(1.0,0.90),(1.0,0.95)]
RANK_BATCH=32; MASS_BATCH=256; MAX_ROUNDS=700
OUT=Path('experiments/artifacts/proofbits_topp_uncertainty_adaptive.json'); OUT.parent.mkdir(parents=True,exist_ok=True)
torch.set_grad_enabled(False); torch.manual_seed(47); np.random.seed(47)

def log(x): print(f'[{time.strftime("%H:%M:%S")}] {x}',flush=True)
def hidden(m,**kw): return m.model(**kw,use_cache=False,return_dict=True).last_hidden_state

def endpoints(w):
    w16=w.half().contiguous(); raw=w16.view(torch.int16).to(torch.int32)&0xffff; hb=(raw>>8)&255
    a=(hb<<8).to(torch.int16).contiguous().view(torch.float16).float(); b=((hb<<8)|255).to(torch.int16).contiguous().view(torch.float16).float()
    return w16.float(),torch.minimum(a,b),torch.maximum(a,b)

def collect(m,tok):
    ds=load_dataset('wikitext','wikitext-2-raw-v1',split='test'); hs=[]
    for r in ds:
        t=r['text'].strip()
        if len(t)<220: continue
        x=tok(t,return_tensors='pt',truncation=True,max_length=MAX_LEN)
        if x['input_ids'].shape[1]<24: continue
        H=hidden(m,**x)[0].float().cpu()
        for p in range(11,H.shape[0],13):
            hs.append(H[p])
            if len(hs)>=N: return torch.stack(hs)
    return torch.stack(hs)

def scores(h,W,L,H):
    hp=torch.clamp(h,min=0); hn=torch.clamp(h,max=0); ex=[]; lo=[]; up=[]
    for a in range(0,W.shape[0],CH):
        ex.append(h@W[a:a+CH].T); lo.append(hp@L[a:a+CH].T+hn@H[a:a+CH].T); up.append(hp@H[a:a+CH].T+hn@L[a:a+CH].T)
    return torch.cat(ex).double(),torch.cat(lo).double(),torch.cat(up).double()

def dense_set(z,p):
    c=float(z.max()); e=torch.exp(z-c); order=torch.argsort(z,descending=True); cum=torch.cumsum(e[order],0)/e.sum(); k=int(torch.searchsorted(cum,torch.tensor(p,dtype=cum.dtype)))+1
    return k,order[:k]

def certify_adaptive(ex,L,U,temp,p):
    z=ex/temp; l=L/temp; u=U/temp; V=len(z); c=float(u.max())
    ez=torch.exp(z-c); el=torch.exp(l-c); eu=torch.exp(u-c); uncertainty=eu-el
    rank_order=torch.argsort(u,descending=True); mass_order=torch.argsort(uncertainty,descending=True)
    refined=torch.zeros(V,dtype=torch.bool); rp=mp=0
    kref,ref=dense_set(z,p)
    for rd in range(MAX_ROUNDS):
        if rd==0 or not refined.any():
            pass
        # Add independent rank-critical and partition-uncertainty-critical rows.
        added=0
        while rp<V and added<RANK_BATCH:
            i=int(rank_order[rp]); rp+=1
            if not refined[i]: refined[i]=True; added+=1
        added=0
        while mp<V and added<MASS_BATCH:
            i=int(mass_order[mp]); mp+=1
            if not refined[i]: refined[i]=True; added+=1
        R=refined.nonzero().squeeze(1)
        un=(~refined).nonzero().squeeze(1)
        zlo=el.sum()-el[R].sum()+ez[R].sum(); zhi=eu.sum()-eu[R].sum()+ez[R].sum()
        rz=z[R]; ordR=torch.argsort(rz,descending=True); sidx=R[ordR]; sz=rz[ordR]
        max_tail=float(u[un].max()) if un.numel() else -float('inf')
        safe=int((sz>max_tail).sum()) if un.numel() else len(R)
        if safe:
            pref=torch.cumsum(torch.exp(sz[:safe]-c),0)
            for kk in range(1,safe+1):
                before=0.0 if kk==1 else float(pref[kk-2]/zlo)
                after=float(pref[kk-1]/zhi)
                if before<p and after>=p:
                    pred=sidx[:kk]
                    exact=(kk==kref and set(map(int,pred.tolist()))==set(map(int,ref.tolist())))
                    return {'certified':True,'exact_set':bool(exact),'rows':int(R.numel()),'rounds':rd+1,'k':kk,'dense_k':kref,'safe_rank':safe,'cum_before_upper':before,'cum_at_lower':after}
        if refined.all(): break
    return {'certified':False,'exact_set':False,'rows':int(refined.sum()),'rounds':MAX_ROUNDS,'k':None,'dense_k':kref,'safe_rank':0}

def main():
    log('load'); tok=AutoTokenizer.from_pretrained(MODEL); m=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float32); m.eval(); Hs=collect(m,tok)
    W,L,H=endpoints(m.lm_head.weight.detach().float().cpu()); V,D=W.shape; del m
    rows=[]
    for qi,h in enumerate(Hs):
        ex,lo,up=scores(h,W,L,H)
        for temp,p in SETTINGS:
            r=certify_adaptive(ex,lo,up,temp,p); r.update({'i':qi,'temperature':temp,'p':p,'fraction':r['rows']/V,'idealized_weight_byte_reduction':2/(1+r['rows']/V)}); rows.append(r)
        log(f'{qi+1}/{len(Hs)}')
    summary={}
    for temp,p in SETTINGS:
        rr=[r for r in rows if r['temperature']==temp and r['p']==p]; a=np.asarray([r['rows'] for r in rr],float); f=a/V
        summary[f'T{temp}_p{p}']={'n':len(rr),'all_certified':bool(all(r['certified'] for r in rr)),'all_exact_sets':bool(all(r['exact_set'] for r in rr)),'mean_rows':float(a.mean()),'median_rows':float(np.median(a)),'p90_rows':float(np.percentile(a,90)),'max_rows':int(a.max()),'mean_fraction':float(f.mean()),'idealized_reduction':float(2/(1+f.mean())),'mean_dense_k':float(np.mean([r['dense_k'] for r in rr]))}
    report={'kind':'proofbits_topp_uncertainty_adaptive','model':MODEL,'vocab':V,'n':len(Hs),'rank_batch':RANK_BATCH,'mass_batch':MASS_BATCH,'summary':summary,'rows':rows,
      'method':'Refine union of rows with highest score upper bounds and highest exp(U/T)-exp(L/T) partition-mass uncertainty. Certify global rank prefix plus lower/upper partition inequalities.',
      'caveat':'CPU low-byte-row feasibility metric; sorting/exp arithmetic and GPU latency are not benchmarked. Selection batches are heuristic and not claimed optimal.'}
    OUT.write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
