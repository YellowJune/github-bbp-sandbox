import json, math, time
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL='Qwen/Qwen2.5-0.5B-Instruct'
TARGET_STATES=256
MAX_LEN=128
PILOT_KS=[1,4,16]
BATCH=8
OUT=Path('experiments/artifacts/proofbits_fp16_hardmargin.json'); OUT.parent.mkdir(parents=True,exist_ok=True)
torch.set_grad_enabled(False); torch.manual_seed(7); np.random.seed(7)

def log(x): print(f'[{time.strftime("%H:%M:%S")}] {x}',flush=True)

def hidden(m,**kw): return m.model(**kw,use_cache=False,return_dict=True).last_hidden_state

def collect_many_positions(m,tok):
    ds=load_dataset('wikitext','wikitext-2-raw-v1',split='test')
    texts=[x['text'] for x in ds if len(x['text'].strip())>180]
    hs=[]; meta=[]
    for di,t in enumerate(texts):
        x=tok(t,return_tensors='pt',truncation=True,max_length=MAX_LEN)
        if x['input_ids'].shape[1]<16: continue
        H=hidden(m,**x)[0].float().cpu()
        # Sample actual next-token states across positions, not only sequence ends.
        positions=list(range(7,H.shape[0],7))
        for p in positions:
            hs.append(H[p]); meta.append({'source':'wikitext','doc':di,'position':int(p)})
            if len(hs)>=TARGET_STATES: return torch.stack(hs),meta
    return torch.stack(hs),meta

def fp16_interval(w):
    w16=w.half().contiguous(); bits=w16.view(torch.int16).to(torch.int32)&0xFFFF; hi=(bits>>8)&255
    a=(hi<<8).to(torch.int16).contiguous().view(torch.float16).float()
    z=((hi<<8)|255).to(torch.int16).contiguous().view(torch.float16).float()
    if not (torch.isfinite(a).all() and torch.isfinite(z).all()): raise RuntimeError('nonfinite interval')
    return w16.float().contiguous(),((a+z)*0.5).contiguous(),((z-a).abs()*0.5).contiguous()

def score(h,w,chunk=4096):
    return torch.cat([h.float()@w[a:a+chunk].float().T for a in range(0,w.shape[0],chunk)],1)

def rankdata(x):
    order=np.argsort(x,kind='mergesort'); ranks=np.empty(len(x),dtype=np.float64); ranks[order]=np.arange(len(x),dtype=np.float64)
    return ranks

def spearman(x,y):
    rx=rankdata(np.asarray(x)); ry=rankdata(np.asarray(y))
    if rx.std()==0 or ry.std()==0: return float('nan')
    return float(np.corrcoef(rx,ry)[0,1])

def main():
    log('load model'); tok=AutoTokenizer.from_pretrained(MODEL); m=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float32); m.eval()
    W=m.lm_head.weight.detach().float().cpu().contiguous(); V,D=W.shape
    log('collect multi-position states'); H,meta=collect_many_positions(m,tok); log(f'collected {len(H)} states')
    del m
    log('build high-byte exact intervals'); W16,mid,rad=fp16_interval(W); del W

    margins=[]; relm=[]; winner_scores=[]; cand={k:[] for k in PILOT_KS}; pilot_hit={k:[] for k in PILOT_KS}
    hard=[]
    for s in range(0,len(H),BATCH):
        hb=H[s:s+BATCH]
        exact=score(hb,W16); coarse=score(hb,mid); err=score(hb.abs(),rad)
        top2=torch.topk(exact,k=2,dim=1)
        mar=(top2.values[:,0]-top2.values[:,1]).cpu().numpy()
        win=top2.values[:,0].cpu().numpy()
        margins.extend(mar.tolist()); winner_scores.extend(win.tolist())
        relm.extend((mar/(np.abs(win)+1e-12)).tolist())
        ref=top2.indices[:,0]
        for k in PILOT_KS:
            pilots=torch.topk(coarse,k=k,dim=1).indices
            Bscore=exact.gather(1,pilots).amax(1)
            mask=(coarse+err)>=Bscore[:,None]
            cc=mask.sum(1).cpu().numpy(); cand[k].extend(cc.tolist())
            pilot_hit[k].extend((pilots==ref[:,None]).any(1).cpu().numpy().astype(int).tolist())
        for j in range(len(hb)):
            idx=s+j
            hard.append({'index':idx,'margin':float(mar[j]),'relative_margin':float(mar[j]/(abs(win[j])+1e-12))})
        log(f'processed {min(s+BATCH,len(H))}/{len(H)}')

    margins=np.asarray(margins); relm=np.asarray(relm)
    report={'kind':'proofbits_fp16_hardmargin_stress','model':MODEL,'states':len(H),'vocab':V,'hidden_dim':D,
            'state_source':'WikiText actual intermediate token positions','exact_reference':'FP16-rounded lm_head with FP32 accumulation','pilots':{},
            'margin':{'mean':float(margins.mean()),'median':float(np.median(margins)),'p10':float(np.percentile(margins,10)),'p1':float(np.percentile(margins,1)),'min':float(margins.min()),
                      'relative_median':float(np.median(relm)),'relative_p1':float(np.percentile(relm,1)),'relative_min':float(relm.min())}}
    for k in PILOT_KS:
        c=np.asarray(cand[k],dtype=np.float64); f=c/V
        report['pilots'][str(k)]={'candidate_mean':float(c.mean()),'median':float(np.median(c)),'p90':float(np.percentile(c,90)),'p99':float(np.percentile(c,99)),'max':int(c.max()),
                                  'candidate_fraction_mean':float(f.mean()),'idealized_bw_reduction':float(2/(1+f.mean())),
                                  'pilot_contains_true_top1_rate':float(np.mean(pilot_hit[k])),
                                  'spearman_candidate_vs_negative_log_margin':spearman(c,-np.log10(np.maximum(margins,1e-12)))}
    # Hardest margin examples with candidate counts.
    ids=np.argsort(margins)[:20]
    report['20_smallest_margin_states']=[]
    for i in ids:
        row={'index':int(i),'margin':float(margins[i]),'relative_margin':float(relm[i]),'meta':meta[int(i)]}
        for k in PILOT_KS: row[f'candidates_k{k}']=int(cand[k][int(i)])
        report['20_smallest_margin_states'].append(row)
    # Candidate counts stratified by exact-margin quantile.
    edges=np.quantile(margins,[0,.01,.05,.1,.25,.5,.75,1.0])
    report['margin_bins']=[]
    for a,b in zip(edges[:-1],edges[1:]):
        sel=(margins>=a)&(margins<=b)
        row={'margin_lo':float(a),'margin_hi':float(b),'n':int(sel.sum())}
        for k in PILOT_KS: row[f'candidate_mean_k{k}']=float(np.asarray(cand[k])[sel].mean())
        report['margin_bins'].append(row)
    report['caveat']='Exactness is theorem-guaranteed by intervals; this stress test measures efficiency degradation near small exact top1-top2 margins. Idealized weight-byte traffic only; no GPU wall-clock.'
    OUT.write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
