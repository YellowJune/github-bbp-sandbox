import json, math, time
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL='Qwen/Qwen2.5-0.5B-Instruct'
N_NAT=128; N_BASE=80; N_BOUND=32; PILOT_K=4; MAX_LEN=128; BATCH=8
OUT=Path('experiments/artifacts/proofbits_roundoff_safe_certificate.json'); OUT.parent.mkdir(parents=True,exist_ok=True)
torch.set_grad_enabled(False); torch.manual_seed(41); np.random.seed(41)
U32=2.0**-24

def log(x): print(f'[{time.strftime("%H:%M:%S")}] {x}',flush=True)
def hidden(m,**kw): return m.model(**kw,use_cache=False,return_dict=True).last_hidden_state

def endpoints(w):
    w16=w.half().contiguous(); raw=w16.view(torch.int16).to(torch.int32)&0xffff; hb=(raw>>8)&255
    a=(hb<<8).to(torch.int16).contiguous().view(torch.float16).float(); b=((hb<<8)|255).to(torch.int16).contiguous().view(torch.float16).float()
    if not(torch.isfinite(a).all() and torch.isfinite(b).all()): raise RuntimeError('nonfinite endpoint')
    lo=torch.minimum(a,b).contiguous(); hi=torch.maximum(a,b).contiguous()
    rowmax=torch.maximum(lo.abs(),hi.abs()).amax(1).contiguous()
    return w16.float().contiguous(),lo,hi,rowmax

def score(h,w,ch=4096): return torch.cat([h.float()@w[a:a+ch].T for a in range(0,len(w),ch)],1)
def upper_raw(h,lo,hi,ch=4096):
    hp=torch.clamp(h.float(),min=0); hn=torch.clamp(h.float(),max=0)
    return torch.cat([hp@hi[a:a+ch].T+hn@lo[a:a+ch].T for a in range(0,len(lo),ch)],1)
def collect_nat(m,tok):
    ds=load_dataset('wikitext','wikitext-2-raw-v1',split='test'); hs=[]
    for t in [x['text'] for x in ds if len(x['text'].strip())>180]:
        x=tok(t,return_tensors='pt',truncation=True,max_length=MAX_LEN)
        if x['input_ids'].shape[1]<16: continue
        H=hidden(m,**x)[0].float().cpu()
        for p in range(7,H.shape[0],7):
            hs.append(H[p])
            if len(hs)>=N_NAT:return torch.stack(hs)
    return torch.stack(hs)
def collect_base(m,tok):
    ag=load_dataset('ag_news',split='test'); wiki=load_dataset('wikitext','wikitext-2-raw-v1',split='test')
    texts=[ag[i]['text'] for i in range(N_BASE//2)]+[x['text'] for x in wiki if len(x['text'].strip())>100][:N_BASE//2]
    hs=[]
    for t in texts:
        x=tok(t,return_tensors='pt',truncation=True,max_length=MAX_LEN)
        H=hidden(m,**x)[0].float().cpu(); hs.append(H[max(0,H.shape[0]-1-(len(hs)%5))])
    return torch.stack(hs)
def gamma(n):
    x=n*U32
    if x>=1: raise RuntimeError('gamma undefined')
    return x/(1-x)
def cert_upper(h,U,rowmax,D):
    # This CPU experiment computes U as two BLAS dot products plus an add, so use
    # gamma_{4d} conservatively. The intended one-dot-product Triton upper kernel
    # can use the tighter implementation-specific bound after GPU validation.
    g=gamma(4*D)
    # One envelope covers possible downward error in Uhat and one covers possible
    # upward error in the dense FP32-accumulated exact score.
    slack=(2*g)*h.float().abs().sum(1)[:,None]*rowmax[None,:]
    return U+slack,slack

def eval_states(H,W16,lo,hi,rowmax):
    V,D=W16.shape; raw_counts=[]; safe_counts=[]; oks=[]; slack_stats=[]
    for s in range(0,len(H),BATCH):
        h=H[s:s+BATCH]; exact=score(h,W16); U=upper_raw(h,lo,hi); Us,sl=cert_upper(h,U,rowmax,D)
        pilots=torch.topk(Us,k=PILOT_K,dim=1).indices; B=exact.gather(1,pilots).amax(1)
        raw=(U>=B[:,None]); safe=(Us>=B[:,None]); ref=exact.argmax(1)
        for n in range(len(h)):
            ri=raw[n].nonzero().squeeze(1); si=safe[n].nonzero().squeeze(1)
            raw_counts.append(len(ri)); safe_counts.append(len(si)); oks.append(int(si[exact[n,si].argmax()])==int(ref[n])); slack_stats.append(float(sl[n].median()))
        log(f'natural {min(s+BATCH,len(H))}/{len(H)}')
    rc=np.array(raw_counts,float); sc=np.array(safe_counts,float); f=sc.mean()/V
    return {'n':len(H),'all_exact_safe':bool(all(oks)),'raw_candidate_mean_same_pilots':float(rc.mean()),'safe_candidate_mean':float(sc.mean()),'safe_median':float(np.median(sc)),'safe_p90':float(np.percentile(sc,90)),'safe_p99':float(np.percentile(sc,99)),'safe_max':int(sc.max()),'safe_fraction_mean':float(f),'idealized_bw_reduction_safe':float(2/(1+f)),'median_certificate_slack':float(np.median(slack_stats))}
def boundary_alpha(za,zb,ya,it=50):
    l,r=0.,1.
    for _ in range(it):
        a=(l+r)/2; y=int(((1-a)*za+a*zb).argmax())
        if y==ya:l=a
        else:r=a
    return (l+r)/2
def eval_boundary(H,Z,W16,lo,hi,rowmax):
    V,D=W16.shape; Y=Z.argmax(1); pairs=[]
    for i in range(len(H)):
        for j in range(i+1,len(H)):
            if int(Y[i])!=int(Y[j]):pairs.append((i,j))
            if len(pairs)>=N_BOUND:break
        if len(pairs)>=N_BOUND:break
    counts=[]; oks=[]; margins=[]; slacks=[]
    for k,(i,j) in enumerate(pairs):
        a=boundary_alpha(Z[i],Z[j],int(Y[i])); h=((1-a)*H[i]+a*H[j])[None,:]; exact=((1-a)*Z[i]+a*Z[j])[None,:]
        U=upper_raw(h,lo,hi); Us,sl=cert_upper(h,U,rowmax,D); pilots=torch.topk(Us,k=PILOT_K,dim=1).indices; B=exact.gather(1,pilots).amax(1)
        si=(Us[0]>=B[0]).nonzero().squeeze(1); ref=int(exact[0].argmax()); pred=int(si[exact[0,si].argmax()]); counts.append(len(si)); oks.append(pred==ref)
        tmp=exact[0].clone(); tmp[ref]=-torch.inf; ru=int(tmp.argmax()); margins.append(float(exact[0,ref]-exact[0,ru])); slacks.append(float(sl[0].median()))
        log(f'boundary {k+1}/{len(pairs)} cand={len(si)} margin={margins[-1]:.3e}')
    c=np.array(counts,float); f=c.mean()/V
    return {'n':len(pairs),'all_exact_safe':bool(all(oks)),'candidate_mean':float(c.mean()),'median':float(np.median(c)),'p90':float(np.percentile(c,90)),'p99':float(np.percentile(c,99)),'max':int(c.max()),'fraction_mean':float(f),'idealized_bw_reduction_safe':float(2/(1+f)),'margin_median':float(np.median(margins)),'margin_min':float(np.min(margins)),'median_certificate_slack':float(np.median(slacks))}
def main():
    log('load'); tok=AutoTokenizer.from_pretrained(MODEL); m=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float32); m.eval(); W=m.lm_head.weight.detach().float().cpu(); V,D=W.shape
    log('collect natural'); Hn=collect_nat(m,tok); log('collect boundary base'); Hb=collect_base(m,tok); del m
    log('endpoints'); W16,lo,hi,rowmax=endpoints(W); del W; Zb=score(Hb,W16)
    nat=eval_states(Hn,W16,lo,hi,rowmax); bnd=eval_boundary(Hb,Zb,W16,lo,hi,rowmax)
    g=gamma(4*D)
    report={'kind':'proofbits_fp16_roundoff_safe_certificate','model':MODEL,'vocab':V,'hidden_dim':D,'pilot_k':PILOT_K,'fp32_unit_roundoff':U32,'gamma_4d':g,'certificate':'Uhat + 2*gamma_{4d}*||h||_1*M_i for this conservative two-dot CPU test; intended fused one-dot kernel admits a tighter implementation-specific gamma','natural':nat,'boundary':bnd,'caveat':'Higham-style worst-case rounding envelope for the stated FP32-accumulated reference. Publication GPU code must derive the envelope for the actual FMA/reduction order and compare against the exact numerical semantics of its dense baseline.'}
    OUT.write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=='__main__':main()
