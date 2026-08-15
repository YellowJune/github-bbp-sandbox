import json, time
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL='Qwen/Qwen2.5-0.5B-Instruct'
N=48
MAX_LEN=128
PS=[0.90,0.95]
TEMPS=[0.7,1.0]
MS=[64,128,256,512,1024,2048,4096,8192,16384,32768,65536]
CH=4096
OUT=Path('experiments/artifacts/proofbits_exact_topp.json'); OUT.parent.mkdir(parents=True,exist_ok=True)
torch.set_grad_enabled(False); torch.manual_seed(41); np.random.seed(41)

def log(x): print(f'[{time.strftime("%H:%M:%S")}] {x}',flush=True)
def hidden(m,**kw): return m.model(**kw,use_cache=False,return_dict=True).last_hidden_state

def fp16_endpoints(w):
    w16=w.half().contiguous(); raw=w16.view(torch.int16).to(torch.int32)&0xffff; hb=(raw>>8)&255
    a=(hb<<8).to(torch.int16).contiguous().view(torch.float16).float()
    b=((hb<<8)|255).to(torch.int16).contiguous().view(torch.float16).float()
    if not(torch.isfinite(a).all() and torch.isfinite(b).all()): raise RuntimeError('nonfinite interval')
    return w16.float().contiguous(),torch.minimum(a,b).contiguous(),torch.maximum(a,b).contiguous()

def collect(m,tok):
    ds=load_dataset('wikitext','wikitext-2-raw-v1',split='test'); hs=[]
    for row in ds:
        t=row['text'].strip()
        if len(t)<220: continue
        x=tok(t,return_tensors='pt',truncation=True,max_length=MAX_LEN)
        if x['input_ids'].shape[1]<24: continue
        H=hidden(m,**x)[0].float().cpu()
        for p in range(9,H.shape[0],11):
            hs.append(H[p])
            if len(hs)>=N: return torch.stack(hs)
    return torch.stack(hs)

def score_bounds(h,w,lo,hi):
    hp=torch.clamp(h.float(),min=0); hn=torch.clamp(h.float(),max=0)
    exs=[]; los=[]; ups=[]
    for a in range(0,w.shape[0],CH):
        wc=w[a:a+CH]; lc=lo[a:a+CH]; hc=hi[a:a+CH]
        exs.append(h.float()@wc.T)
        los.append(hp@lc.T + hn@hc.T)
        ups.append(hp@hc.T + hn@lc.T)
    return torch.cat(exs).double(),torch.cat(los).double(),torch.cat(ups).double()

def dense_nucleus(ex,p):
    c=float(ex.max()); e=torch.exp(ex-c); order=torch.argsort(ex,descending=True); es=e[order]
    cum=torch.cumsum(es,0)/es.sum(); k=int(torch.searchsorted(cum,torch.tensor(p,dtype=cum.dtype)).item())+1
    return k,order[:k]

def certify(ex,L,U,p,temp,V):
    z=ex/temp; l=L/temp; u=U/temp
    kref,refset=dense_nucleus(z,p)
    orderU=torch.argsort(u,descending=True); c=float(u.max())
    ez=torch.exp(z-c); el=torch.exp(l-c); eu=torch.exp(u-c)
    totalL=float(el.sum()); totalU=float(eu.sum())
    schedule=[m for m in MS if m<V]+[V]
    for M in schedule:
        R=orderU[:M]
        Zlo=totalL-float(el[R].sum())+float(ez[R].sum())
        Zhi=totalU-float(eu[R].sum())+float(ez[R].sum())
        rz=z[R]; ordR=torch.argsort(rz,descending=True); sorted_idx=R[ordR]; sorted_z=rz[ordR]
        max_tail=float(u[orderU[M]]) if M<V else -float('inf')
        safe=int((sorted_z>max_tail).sum().item()) if M<V else M
        if safe==0: continue
        pe=torch.exp(sorted_z[:safe]-c); pref=torch.cumsum(pe,0)
        for kk in range(1,safe+1):
            before=0.0 if kk==1 else float(pref[kk-2])/Zlo
            after=float(pref[kk-1])/Zhi
            if before < p and after >= p:
                pred=sorted_idx[:kk]
                ok=(kk==kref and set(map(int,pred.tolist()))==set(map(int,refset.tolist())))
                return {'certified':True,'M':M,'k':kk,'dense_k':kref,'exact_set':bool(ok),'safe_rank':safe,'max_tail_upper':max_tail,
                        'cum_before_upper':before,'cum_at_lower':after,'fraction':M/V,'idealized_weight_byte_reduction':2/(1+M/V)}
    return {'certified':False,'M':V,'k':None,'dense_k':kref,'exact_set':False,'fraction':1.0,'idealized_weight_byte_reduction':1.0}

def main():
    log('load'); tok=AutoTokenizer.from_pretrained(MODEL); m=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float32); m.eval()
    H=collect(m,tok); W=m.lm_head.weight.detach().float().cpu().contiguous(); V,D=W.shape; del m
    log(f'endpoints V={V} D={D} N={len(H)}'); W16,lo,hi=fp16_endpoints(W); del W
    rows=[]
    for i,h in enumerate(H):
        ex,L,U=score_bounds(h,W16,lo,hi)
        for t in TEMPS:
            for p in PS:
                r=certify(ex,L,U,p,t,V); r.update({'i':i,'temperature':t,'p':p}); rows.append(r)
        log(f'{i+1}/{len(H)}')
    summary={}
    for t in TEMPS:
        for p in PS:
            rr=[r for r in rows if r['temperature']==t and r['p']==p]; ms=np.asarray([r['M'] for r in rr],float); f=ms/V
            summary[f'T{t}_p{p}']={'n':len(rr),'all_certified':bool(all(r['certified'] for r in rr)),'all_exact_sets':bool(all(r['exact_set'] for r in rr)),
              'refine_rows_mean':float(ms.mean()),'refine_rows_median':float(np.median(ms)),'refine_rows_p90':float(np.percentile(ms,90)),'refine_rows_max':int(ms.max()),
              'refine_fraction_mean':float(f.mean()),'idealized_weight_byte_reduction_from_mean_fraction':float(2/(1+f.mean())),
              'dense_nucleus_k_mean':float(np.mean([r['dense_k'] for r in rr])),'dense_nucleus_k_p90':float(np.percentile([r['dense_k'] for r in rr],90))}
    report={'kind':'proofbits_exact_topp','model':MODEL,'vocab':V,'hidden_dim':D,'n':len(H),'summary':summary,'rows':rows,
      'method':'High-byte intervals bound every logit and the full softmax partition function. Top-U rows are low-byte refined until both ranking and nucleus cumulative-mass inequalities certify the exact FP16 top-p set.',
      'caveat':'Geometric refinement schedule, so reported low-byte rows are an upper bound on the minimal required refinement. This uses both lower and upper score bounds, increasing arithmetic versus argmax/top-k upper-only ProofBits. GPU latency not measured.'}
    OUT.write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
