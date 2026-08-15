import json, time
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL='Qwen/Qwen2.5-0.5B-Instruct'
N_NAT=256
N_BASE=96
N_BOUND=48
PILOT_K=4
MAX_LEN=128
BATCH=8
OUT=Path('experiments/artifacts/proofbits_upperonly_final_stress.json'); OUT.parent.mkdir(parents=True,exist_ok=True)
torch.set_grad_enabled(False); torch.manual_seed(23); np.random.seed(23)

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

def collect_natural(m,tok):
    ds=load_dataset('wikitext','wikitext-2-raw-v1',split='test'); hs=[]; meta=[]
    docs=[x['text'] for x in ds if len(x['text'].strip())>180]
    for di,t in enumerate(docs):
        x=tok(t,return_tensors='pt',truncation=True,max_length=MAX_LEN)
        if x['input_ids'].shape[1]<16: continue
        H=hidden(m,**x)[0].float().cpu()
        for p in range(7,H.shape[0],7):
            hs.append(H[p]); meta.append({'doc':di,'position':int(p)})
            if len(hs)>=N_NAT: return torch.stack(hs),meta
    return torch.stack(hs),meta

def collect_boundary_base(m,tok):
    ag=load_dataset('ag_news',split='test'); wiki=load_dataset('wikitext','wikitext-2-raw-v1',split='test')
    texts=[ag[i]['text'] for i in range(N_BASE//2)]
    texts += [x['text'] for x in wiki if len(x['text'].strip())>100][:N_BASE-len(texts)]
    hs=[]
    for t in texts:
        x=tok(t,return_tensors='pt',truncation=True,max_length=MAX_LEN); H=hidden(m,**x)[0].float().cpu()
        p=max(0,H.shape[0]-1-(len(hs)%5)); hs.append(H[p])
    return torch.stack(hs)

def eval_batch(h,exact,U):
    pilots=torch.topk(U,k=PILOT_K,dim=1).indices
    B=exact.gather(1,pilots).amax(1)
    mask=U>=B[:,None]
    ref=exact.argmax(1); counts=[]; exact_ok=[]; hit=[]; margins=[]
    for n in range(len(h)):
        idx=mask[n].nonzero().squeeze(1); counts.append(int(idx.numel()))
        pred=int(idx[exact[n,idx].argmax()]); exact_ok.append(pred==int(ref[n])); hit.append(bool((pilots[n]==ref[n]).any()))
        r=int(ref[n]); tmp=exact[n].clone(); tmp[r]=-torch.inf; ru=int(tmp.argmax()); margins.append(float(exact[n,r]-exact[n,ru]))
    return counts,exact_ok,hit,margins

def natural_stress(H,W16,lo,hi,V):
    counts=[]; oks=[]; hits=[]; margins=[]
    for s in range(0,len(H),BATCH):
        hb=H[s:s+BATCH]; exact=score(hb,W16); U=upper(hb,lo,hi)
        c,o,hh,m=eval_batch(hb,exact,U); counts+=c; oks+=o; hits+=hh; margins+=m; log(f'natural {min(s+BATCH,len(H))}/{len(H)}')
    c=np.asarray(counts,float); m=np.asarray(margins,float); f=c.mean()/V
    return {'n':len(H),'all_exact':bool(all(oks)),'pilot_contains_winner_rate':float(np.mean(hits)),
      'candidate_mean':float(c.mean()),'median':float(np.median(c)),'p90':float(np.percentile(c,90)),'p99':float(np.percentile(c,99)),'max':int(c.max()),'fraction_mean':float(f),'idealized_bw_reduction':float(2/(1+f)),
      'margin_mean':float(m.mean()),'margin_median':float(np.median(m)),'margin_p1':float(np.percentile(m,1)),'margin_min':float(m.min())}

def boundary_alpha(za,zb,ya,iters=50):
    l,r=0.,1.
    for _ in range(iters):
        a=(l+r)/2; y=int(((1-a)*za+a*zb).argmax())
        if y==ya: l=a
        else: r=a
    return (l+r)/2

def boundary_stress(H,Z,W16,lo,hi,V):
    Y=Z.argmax(1); pairs=[]
    for i in range(len(H)):
        for j in range(i+1,len(H)):
            if int(Y[i])!=int(Y[j]): pairs.append((i,j))
            if len(pairs)>=N_BOUND: break
        if len(pairs)>=N_BOUND: break
    rows=[]
    for pi,(i,j) in enumerate(pairs):
        a=boundary_alpha(Z[i],Z[j],int(Y[i])); h=(1-a)*H[i]+a*H[j]; exact=((1-a)*Z[i]+a*Z[j])[None,:]; U=upper(h[None,:],lo,hi)
        c,o,hit,margin=eval_batch(h[None,:],exact,U)
        rows.append({'pair':[i,j],'alpha':float(a),'candidate_count':int(c[0]),'exact':bool(o[0]),'pilot_contains_winner':bool(hit[0]),'margin':float(margin[0])})
        log(f'boundary {pi+1}/{len(pairs)} margin={margin[0]:.3e} cand={c[0]}')
    c=np.asarray([r['candidate_count'] for r in rows],float); m=np.asarray([r['margin'] for r in rows],float); f=c.mean()/V
    return {'n':len(rows),'all_exact':bool(all(r['exact'] for r in rows)),'pilot_contains_winner_rate':float(np.mean([r['pilot_contains_winner'] for r in rows])),
      'candidate_mean':float(c.mean()),'median':float(np.median(c)),'p90':float(np.percentile(c,90)),'p99':float(np.percentile(c,99)),'max':int(c.max()),'fraction_mean':float(f),'idealized_bw_reduction':float(2/(1+f)),
      'margin_mean':float(m.mean()),'margin_median':float(np.median(m)),'margin_min':float(m.min()),'margin_max':float(m.max()),'rows':rows}

def main():
    log('load model'); tok=AutoTokenizer.from_pretrained(MODEL); m=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float32); m.eval(); W=m.lm_head.weight.detach().float().cpu().contiguous(); V,D=W.shape
    log('collect natural'); Hnat,_=collect_natural(m,tok); log('collect boundary base'); Hbase=collect_boundary_base(m,tok); del m
    log('FP16 endpoints'); W16,lo,hi=fp16_endpoints(W); del W
    log('boundary base scores'); Zbase=score(Hbase,W16)
    nat=natural_stress(Hnat,W16,lo,hi,V); bnd=boundary_stress(Hbase,Zbase,W16,lo,hi,V)
    report={'kind':'proofbits_final_upperonly_stress','model':MODEL,'vocab':V,'hidden_dim':D,'pilot_k':PILOT_K,
      'upper':'sum_j max(h_j*w^-_ij,h_j*w^+_ij)','natural_wikitext_intermediate_positions':nat,'adversarial_decision_boundary_interpolations':bnd,
      'caveat':'FP16-rounded lm_head with FP32 accumulation. Idealized weight-byte traffic only; GPU wall-clock/counters not measured.'}
    OUT.write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
