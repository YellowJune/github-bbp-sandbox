import gc, json, time
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL='Qwen/Qwen2.5-1.5B-Instruct'
N=48
MAX_LEN=72
BATCH=2
CH=2048
PILOT_K=4
OUT=Path('experiments/artifacts/proofbits_qwen15b_scale.json'); OUT.parent.mkdir(parents=True,exist_ok=True)
torch.set_grad_enabled(False); torch.manual_seed(37); np.random.seed(37)

def log(x): print(f'[{time.strftime("%H:%M:%S")}] {x}',flush=True)
def hidden(m,**kw): return m.model(**kw,use_cache=False,return_dict=True).last_hidden_state

def collect(m,tok):
    ds=load_dataset('wikitext','wikitext-2-raw-v1',split='test'); hs=[]
    for row in ds:
        t=row['text'].strip()
        if len(t)<180: continue
        x=tok(t,return_tensors='pt',truncation=True,max_length=MAX_LEN)
        if x['input_ids'].shape[1]<20: continue
        with torch.no_grad(): H=hidden(m,**x)[0].float().cpu()
        for p in range(7,H.shape[0],7):
            hs.append(H[p])
            if len(hs)>=N: return torch.stack(hs)
    return torch.stack(hs)

def chunk_endpoints(w16):
    raw=w16.contiguous().view(torch.int16).to(torch.int32)&0xffff; hb=(raw>>8)&255
    a=(hb<<8).to(torch.int16).contiguous().view(torch.float16).float()
    b=((hb<<8)|255).to(torch.int16).contiguous().view(torch.float16).float()
    return torch.minimum(a,b),torch.maximum(a,b)

def score_upper(h,W16):
    hp=torch.clamp(h.float(),min=0); hn=torch.clamp(h.float(),max=0); exs=[]; ups=[]
    for a in range(0,W16.shape[0],CH):
        w=W16[a:a+CH]
        exs.append(h.float()@w.float().T)
        lo,hi=chunk_endpoints(w)
        ups.append(hp@hi.T + hn@lo.T)
    return torch.cat(exs,1),torch.cat(ups,1)

def eval_batch(ex,U):
    pilots=torch.topk(U,k=PILOT_K,dim=1).indices; B=ex.gather(1,pilots).amax(1); mask=U>=B[:,None]
    ref=ex.argmax(1); counts=[]; oks=[]; hits=[]; margins=[]
    for n in range(ex.shape[0]):
        idx=mask[n].nonzero().squeeze(1); pred=int(idx[ex[n,idx].argmax()]); r=int(ref[n])
        counts.append(int(idx.numel())); oks.append(pred==r); hits.append(bool((pilots[n]==r).any()))
        tmp=ex[n].clone(); tmp[r]=-torch.inf; margins.append(float(ex[n,r]-tmp.max()))
    return counts,oks,hits,margins

def main():
    log('load 1.5B fp16 CPU model'); tok=AutoTokenizer.from_pretrained(MODEL)
    m=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float16,low_cpu_mem_usage=True); m.eval()
    log('collect hidden states'); H=collect(m,tok)
    W16=m.lm_head.weight.detach().cpu().half().contiguous().clone(); V,D=W16.shape
    del m; gc.collect(); log(f'model freed; V={V} D={D} N={len(H)}')
    counts=[]; oks=[]; hits=[]; margins=[]
    for s in range(0,len(H),BATCH):
        ex,U=score_upper(H[s:s+BATCH],W16); c,o,hh,mm=eval_batch(ex,U)
        counts+=c; oks+=o; hits+=hh; margins+=mm; log(f'{min(s+BATCH,len(H))}/{len(H)} cand_mean_so_far={np.mean(counts):.2f}')
    c=np.asarray(counts,float); mar=np.asarray(margins,float); f=c.mean()/V
    report={'kind':'proofbits_qwen15b_scale','model':MODEL,'vocab':V,'hidden_dim':D,'n':len(H),'pilot_k':PILOT_K,'all_exact':bool(all(oks)),
      'pilot_contains_winner_rate':float(np.mean(hits)),'candidate_mean':float(c.mean()),'median':float(np.median(c)),'p90':float(np.percentile(c,90)),'p99':float(np.percentile(c,99)),'max':int(c.max()),
      'candidate_fraction_mean':float(f),'idealized_weight_byte_reduction':float(2/(1+f)),'margin_mean':float(mar.mean()),'margin_min':float(mar.min()),
      'caveat':'FP16-rounded lm_head with FP32 accumulation; interval endpoints decoded per chunk to limit RAM. CPU feasibility/traffic test, not GPU latency.'}
    OUT.write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
