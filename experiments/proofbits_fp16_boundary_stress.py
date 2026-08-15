import json, time
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL='Qwen/Qwen2.5-0.5B-Instruct'
N_BASE=96
MAX_PAIRS=48
MAX_LEN=128
PILOT_K=4
OUT=Path('experiments/artifacts/proofbits_fp16_boundary_stress.json'); OUT.parent.mkdir(parents=True,exist_ok=True)
torch.set_grad_enabled(False); torch.manual_seed(13); np.random.seed(13)

def log(x): print(f'[{time.strftime("%H:%M:%S")}] {x}', flush=True)

def hidden(m,**kw): return m.model(**kw,use_cache=False,return_dict=True).last_hidden_state

def collect(m,tok,texts):
    hs=[]
    for t in texts:
        x=tok(t,return_tensors='pt',truncation=True,max_length=MAX_LEN)
        H=hidden(m,**x)[0].float().cpu()
        # Use different actual positions to diversify directions.
        p=max(0,H.shape[0]-1-(len(hs)%5))
        hs.append(H[p])
    return torch.stack(hs)

def score(h,w,chunk=4096):
    return torch.cat([h.float()@w[a:a+chunk].float().T for a in range(0,w.shape[0],chunk)],1)

def intervals(w):
    w16=w.half().contiguous(); raw=w16.view(torch.int16).to(torch.int32)&0xffff; hi=(raw>>8)&255
    a=(hi<<8).to(torch.int16).contiguous().view(torch.float16).float(); b=((hi<<8)|255).to(torch.int16).contiguous().view(torch.float16).float()
    if not (torch.isfinite(a).all() and torch.isfinite(b).all()): raise RuntimeError('nonfinite interval')
    return w16.float().contiguous(),((a+b)*.5).contiguous(),((b-a).abs()*.5).contiguous()

def binary_boundary(za,zb,ya,yb,iters=50):
    lo,hi=0.0,1.0
    # Find first alpha whose winner differs from ya along the segment.
    for _ in range(iters):
        mid=(lo+hi)*.5
        z=za*(1-mid)+zb*mid
        y=int(z.argmax())
        if y==ya: lo=mid
        else: hi=mid
    return (lo+hi)*.5

def eval_pb(h,exact,mid,rad):
    c=score(h[None,:],mid)[0]; e=score(h.abs()[None,:],rad)[0]
    pilots=torch.topk(c,k=PILOT_K).indices
    B=float(exact[pilots].max())
    cand=(c+e)>=B
    idx=cand.nonzero().squeeze(1)
    pred=int(idx[exact[idx].argmax()]); ref=int(exact.argmax())
    # Explicitly mask the winner to guarantee a distinct runner-up index, including exact ties.
    masked=exact.clone(); masked[ref]=-torch.inf
    runnerup=int(masked.argmax())
    margin=float(exact[ref]-exact[runnerup])
    return {'candidate_count':int(idx.numel()),'candidate_fraction':float(idx.numel()/exact.numel()),'exact':pred==ref,'margin':margin,'winner':ref,'runnerup':runnerup,'pilot_contains_winner':bool((pilots==ref).any())}

def main():
    log('load model'); tok=AutoTokenizer.from_pretrained(MODEL); m=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float32); m.eval(); W=m.lm_head.weight.detach().float().cpu().contiguous(); V,D=W.shape
    ds=load_dataset('ag_news',split='test'); wiki=load_dataset('wikitext','wikitext-2-raw-v1',split='test')
    texts=[ds[i]['text'] for i in range(N_BASE//2)]
    texts += [x['text'] for x in wiki if len(x['text'].strip())>100][:N_BASE-len(texts)]
    log('collect base hidden'); H=collect(m,tok,texts); del m
    log('prepare intervals/scores'); W16,mid,rad=intervals(W); del W; Z=score(H,W16); Y=Z.argmax(1)

    pairs=[]
    # Deterministic diverse pairs with different exact winners.
    for i in range(len(H)):
        for j in range(i+1,len(H)):
            if int(Y[i])!=int(Y[j]): pairs.append((i,j))
            if len(pairs)>=MAX_PAIRS: break
        if len(pairs)>=MAX_PAIRS: break
    rows=[]
    for pi,(i,j) in enumerate(pairs):
        a=binary_boundary(Z[i],Z[j],int(Y[i]),int(Y[j]))
        h=(1-a)*H[i]+a*H[j]
        exact=(1-a)*Z[i]+a*Z[j]  # exact because the head is linear
        r=eval_pb(h,exact,mid,rad); r.update({'pair':[i,j],'alpha':float(a),'endpoint_winners':[int(Y[i]),int(Y[j])]}); rows.append(r)
        log(f'{pi+1}/{len(pairs)} margin={r["margin"]:.3e} cand={r["candidate_count"]}')
    c=np.array([r['candidate_count'] for r in rows],dtype=float); margins=np.array([r['margin'] for r in rows])
    report={'kind':'proofbits_fp16_exact_decision_boundary_stress','model':MODEL,'pairs':len(rows),'vocab':V,'hidden_dim':D,'pilot_k':PILOT_K,
      'construction':'linear interpolation between two real hidden states; bisection in exact FP16 score space to the first top-1 decision boundary',
      'all_exact':bool(all(r['exact'] for r in rows)),'pilot_contains_winner_rate':float(np.mean([r['pilot_contains_winner'] for r in rows])),
      'margin':{'mean':float(margins.mean()),'median':float(np.median(margins)),'max':float(margins.max()),'min':float(margins.min())},
      'candidate':{'mean':float(c.mean()),'median':float(np.median(c)),'p90':float(np.percentile(c,90)),'p99':float(np.percentile(c,99)),'max':int(c.max()),'fraction_mean':float(c.mean()/V),'idealized_bw_reduction':float(2/(1+c.mean()/V))},
      'rows':rows,
      'caveat':'These are adversarial boundary interpolations, not natural hidden-state samples. They stress efficiency; interval exactness remains deterministic.'}
    OUT.write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
