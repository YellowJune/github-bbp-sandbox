import gc, json, time
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL='Qwen/Qwen2.5-1.5B-Instruct'
N_AG=24; N_GEN=24; PILOT_K=4; MAX_LEN=96
OUT=Path('experiments/artifacts/proofbits_upperonly_qwen15b.json'); OUT.parent.mkdir(parents=True,exist_ok=True)
torch.set_grad_enabled(False); torch.manual_seed(31); np.random.seed(31)

def log(x): print(f'[{time.strftime("%H:%M:%S")}] {x}',flush=True)
def hidden(m,**kw): return m.model(**kw,use_cache=False,return_dict=True).last_hidden_state

def collect_ag(m,tok):
    ds=load_dataset('ag_news',split='test'); hs=[]
    for i in range(N_AG):
        x=tok(ds[i]['text'],return_tensors='pt',truncation=True,max_length=MAX_LEN)
        hs.append(hidden(m,**x)[0,-1].float().cpu()); log(f'AG {i+1}/{N_AG}')
    return torch.stack(hs)

def collect_gen(m,tok):
    prompts=['Solve carefully: 137 * 29 =','Write a Python function for binary search:','The derivative of x^3 + 2x is','Factor 84 into prime factors:']
    seq=[tok(p,return_tensors='pt')['input_ids'] for p in prompts]; hs=[]
    while len(hs)<N_GEN:
        ns=[]
        for ids in seq:
            h=hidden(m,input_ids=ids)[0,-1].float().cpu(); hs.append(h); log(f'GEN {len(hs)}/{N_GEN}')
            if len(hs)>=N_GEN: break
            nxt=m.lm_head(h[None,:])[0].argmax().view(1,1); ns.append(torch.cat([ids,nxt],1))
        seq=ns
    return torch.stack(hs[:N_GEN])

def endpoints(w):
    w16=w.half().contiguous(); raw=w16.view(torch.int16).to(torch.int32)&0xffff; hb=(raw>>8)&255
    a=(hb<<8).to(torch.int16).contiguous().view(torch.float16).float(); b=((hb<<8)|255).to(torch.int16).contiguous().view(torch.float16).float()
    if not(torch.isfinite(a).all() and torch.isfinite(b).all()): raise RuntimeError('nonfinite endpoint')
    return w16.float().contiguous(),torch.minimum(a,b).contiguous(),torch.maximum(a,b).contiguous()
def score(h,w,ch=2048): return torch.cat([h.float()@w[a:a+ch].T for a in range(0,w.shape[0],ch)],1)
def upper(h,lo,hi,ch=2048):
    hp=torch.clamp(h.float(),min=0); hn=torch.clamp(h.float(),max=0); ys=[]
    for a in range(0,lo.shape[0],ch): ys.append(hp@hi[a:a+ch].T + hn@lo[a:a+ch].T)
    return torch.cat(ys,1)
def evaluate(h,w16,lo,hi):
    exact=score(h,w16); U=upper(h,lo,hi); pilots=torch.topk(U,k=PILOT_K,dim=1).indices; B=exact.gather(1,pilots).amax(1); cand=U>=B[:,None]; ref=exact.argmax(1)
    counts=[]; ok=[]; hit=[]
    for n in range(len(h)):
        idx=cand[n].nonzero().squeeze(1); counts.append(int(idx.numel())); ok.append(int(idx[exact[n,idx].argmax()])==int(ref[n])); hit.append(bool((pilots[n]==ref[n]).any()))
    c=np.asarray(counts,float); f=c.mean()/exact.shape[1]
    return {'n':len(h),'all_exact':bool(all(ok)),'pilot_contains_winner_rate':float(np.mean(hit)),'candidate_mean':float(c.mean()),'median':float(np.median(c)),'p90':float(np.percentile(c,90)),'p99':float(np.percentile(c,99)),'max':int(c.max()),'fraction_mean':float(f),'idealized_bw_reduction':float(2/(1+f))}
def main():
    log('load '+MODEL); tok=AutoTokenizer.from_pretrained(MODEL); m=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float32); m.eval()
    log('collect AG'); ha=collect_ag(m,tok); log('collect gen'); hg=collect_gen(m,tok)
    W=m.lm_head.weight.detach().float().cpu().contiguous(); V,D=W.shape; del m; gc.collect()
    log(f'head V={V} D={D}; build intervals'); w16,lo,hi=endpoints(W); del W; gc.collect()
    log('evaluate AG'); a=evaluate(ha,w16,lo,hi); log(str(a)); log('evaluate GEN'); g=evaluate(hg,w16,lo,hi); log(str(g))
    report={'kind':'proofbits_upperonly_scaleup','model':MODEL,'vocab':V,'hidden_dim':D,'pilot_k':PILOT_K,'ag_news':a,'autoregressive_math_code':g,'caveat':'Small 24+24 state CPU scale-up test. Exact FP16-rounded lm_head with FP32 accumulation; idealized byte traffic only.'}; OUT.write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
