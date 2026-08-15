import gc, json, time
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

PRIMARY='google/gemma-3-270m'
FALLBACK='unsloth/gemma-3-270m'
N=96
MAX_LEN=96
BATCH=3
CH=4096
PILOT_K=4
OUT=Path('experiments/artifacts/proofbits_gemma270m_largevocab.json'); OUT.parent.mkdir(parents=True,exist_ok=True)
torch.set_grad_enabled(False); torch.manual_seed(43); np.random.seed(43)

def log(x): print(f'[{time.strftime("%H:%M:%S")}] {x}',flush=True)
def hidden(m,**kw): return m.model(**kw,use_cache=False,return_dict=True).last_hidden_state

def load():
    last=None
    for name in [PRIMARY,FALLBACK]:
        try:
            log(f'try load {name}')
            tok=AutoTokenizer.from_pretrained(name)
            m=AutoModelForCausalLM.from_pretrained(name,torch_dtype=torch.float32,low_cpu_mem_usage=True)
            return name,tok,m
        except Exception as e:
            last=e; log(f'load failed {name}: {type(e).__name__}: {e}')
    raise last

def collect(m,tok):
    ds=load_dataset('wikitext','wikitext-2-raw-v1',split='test'); hs=[]
    for row in ds:
        t=row['text'].strip()
        if len(t)<180: continue
        x=tok(t,return_tensors='pt',truncation=True,max_length=MAX_LEN)
        if x['input_ids'].shape[1]<20: continue
        H=hidden(m,**x)[0].float().cpu()
        for p in range(5,H.shape[0],6):
            hs.append(H[p])
            if len(hs)>=N: return torch.stack(hs)
    return torch.stack(hs)

def endpoints(w16):
    raw=w16.contiguous().view(torch.int16).to(torch.int32)&0xffff; hb=(raw>>8)&255
    a=(hb<<8).to(torch.int16).contiguous().view(torch.float16).float()
    b=((hb<<8)|255).to(torch.int16).contiguous().view(torch.float16).float()
    return torch.minimum(a,b),torch.maximum(a,b)

def batch_scores(h,W16):
    hp=torch.clamp(h.float(),min=0); hn=torch.clamp(h.float(),max=0); exs=[]; ups=[]
    for a in range(0,W16.shape[0],CH):
        w=W16[a:a+CH]; lo,hi=endpoints(w)
        exs.append(h.float()@w.float().T)
        ups.append(hp@hi.T+hn@lo.T)
    return torch.cat(exs,1),torch.cat(ups,1)

def evaluate(ex,U):
    pilots=torch.topk(U,k=PILOT_K,dim=1).indices; B=ex.gather(1,pilots).amax(1); mask=U>=B[:,None]
    counts=[]; oks=[]; hits=[]; margins=[]
    for n in range(ex.shape[0]):
        idx=mask[n].nonzero().squeeze(1); ref=int(ex[n].argmax()); pred=int(idx[ex[n,idx].argmax()])
        counts.append(int(idx.numel())); oks.append(pred==ref); hits.append(bool((pilots[n]==ref).any()))
        tmp=ex[n].clone(); tmp[ref]=-torch.inf; margins.append(float(ex[n,ref]-tmp.max()))
    return counts,oks,hits,margins

def main():
    loaded,tok,m=load(); m.eval(); log('collect hidden')
    H=collect(m,tok); W16=m.lm_head.weight.detach().cpu().half().contiguous().clone(); V,D=W16.shape
    del m; gc.collect(); log(f'loaded={loaded} V={V} D={D} N={len(H)}')
    counts=[]; oks=[]; hits=[]; margins=[]
    for s in range(0,len(H),BATCH):
        ex,U=batch_scores(H[s:s+BATCH],W16); c,o,hh,mm=evaluate(ex,U)
        counts+=c; oks+=o; hits+=hh; margins+=mm; log(f'{min(s+BATCH,len(H))}/{len(H)} mean={np.mean(counts):.2f}')
    c=np.asarray(counts,float); mar=np.asarray(margins,float); f=c.mean()/V
    report={'kind':'proofbits_gemma270m_largevocab','requested_model':PRIMARY,'loaded_model':loaded,'vocab':V,'hidden_dim':D,'n':len(H),'pilot_k':PILOT_K,
      'all_exact':bool(all(oks)),'pilot_contains_winner_rate':float(np.mean(hits)),'candidate_mean':float(c.mean()),'median':float(np.median(c)),'p90':float(np.percentile(c,90)),'p99':float(np.percentile(c,99)),'max':int(c.max()),
      'candidate_fraction_mean':float(f),'idealized_weight_byte_reduction':float(2/(1+f)),'margin_mean':float(mar.mean()),'margin_min':float(mar.min()),
      'caveat':'Weights are losslessly rounded/repacked to FP16 for the ProofBits output-head experiment, with FP32 accumulation. CPU feasibility/traffic test, not GPU latency.'}
    OUT.write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
