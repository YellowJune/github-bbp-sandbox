import json, math, os, time
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-0.6B")
N_AG = int(os.environ.get("N_AG", "64"))
N_GEN = int(os.environ.get("N_GEN", "64"))
MAX_LEN = 128
SAFE = MODEL.replace('/', '__')
OUT = Path(f"experiments/artifacts/proofbits_crossmodel_{SAFE}.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

torch.set_grad_enabled(False)
torch.manual_seed(0)
np.random.seed(0)

def log(x): print(f"[{time.strftime('%H:%M:%S')}] {x}", flush=True)

def model_hidden(model, **kwargs):
    out = model.model(**kwargs, use_cache=False, return_dict=True)
    return out.last_hidden_state

def collect_last(model, tok, texts):
    out=[]
    for t in texts:
        x=tok(t, return_tensors='pt', truncation=True, max_length=MAX_LEN)
        out.append(model_hidden(model, **x)[0,-1].float().cpu())
    return torch.stack(out)

def collect_gen(model, tok):
    prompts=[
      "Solve carefully: 137 * 29 =", "The derivative of x^3 + 2x is",
      "Write a Python function for binary search:", "Factor 84 into prime factors:",
      "Compute 2^10 =", "Write SQL selecting users older than 18:",
      "The gcd of 84 and 126 is", "Simplify (x+2)(x-2):"
    ]
    seq=[tok(p, return_tensors='pt')['input_ids'] for p in prompts]
    hs=[]
    while len(hs)<N_GEN:
        nxt=[]
        for ids in seq:
            h=model_hidden(model, input_ids=ids)[0,-1].float().cpu()
            hs.append(h)
            if len(hs)>=N_GEN: break
            lg=model.lm_head(h[None,:])[0]
            t=lg.argmax().view(1,1)
            nxt.append(torch.cat([ids,t],dim=1))
        if len(hs)>=N_GEN: break
        seq=nxt
    return torch.stack(hs)

def prefix_mid(q,b=4):
    u=q.to(torch.int16)+128; step=1<<(8-b)
    lo=(u//step)*step; hi=lo+step-1
    return (lo.float()+hi.float())*0.5-128.0

def scores(h, q, s, chunk=4096):
    ys=[]
    for a in range(0,q.shape[0],chunk):
        ys.append(h.float() @ (q[a:a+chunk].float()*s[a:a+chunk,None]).T)
    return torch.cat(ys,1)

def fp_scores(h,w,chunk=4096):
    ys=[]
    for a in range(0,w.shape[0],chunk): ys.append(h.float()@w[a:a+chunk].float().T)
    return torch.cat(ys,1)

def coarse_scores(h,q,s,chunk=4096):
    ys=[]
    for a in range(0,q.shape[0],chunk):
        ys.append((h.float()@prefix_mid(q[a:a+chunk]).T)*s[a:a+chunk][None,:])
    return torch.cat(ys,1)

def eval_domain(h,W,q,s):
    z8=scores(h,q,s); zfp=fp_scores(h,W); c=coarse_scores(h,q,s)
    pilot=c.argmax(1); B=z8.gather(1,pilot[:,None]).squeeze(1)
    err=h.float().abs().sum(1)[:,None]*(7.5*s[None,:])
    cand=(c+err)>=B[:,None]
    ref=z8.argmax(1); pred=[]; cnt=[]
    for n in range(h.shape[0]):
        idx=cand[n].nonzero().squeeze(1); cnt.append(int(idx.numel()))
        pred.append(int(idx[z8[n,idx].argmax()].item()))
    cnt=np.asarray(cnt); V,D=W.shape; frac=cnt.mean()/V
    scale_bits=16.0/D; dense=8+scale_bits; pb=4+4*frac+scale_bits
    return {
      'n':int(h.shape[0]), 'certified_exact_int8_match_rate':float((torch.tensor(pred)==ref).float().mean()),
      'rowwise_int8_vs_fp32_argmax_agreement':float((z8.argmax(1)==zfp.argmax(1)).float().mean()),
      'candidate_mean':float(cnt.mean()), 'median':float(np.median(cnt)),
      'p90':float(np.percentile(cnt,90)), 'p99':float(np.percentile(cnt,99)), 'max':int(cnt.max()),
      'candidate_fraction_mean':float(frac), 'proofbits_bits_per_weight':float(pb),
      'dense_rowwise_int8_bits_per_weight':float(dense), 'idealized_bw_reduction':float(dense/pb)
    }

def main():
    log(f'load {MODEL}')
    tok=AutoTokenizer.from_pretrained(MODEL)
    model=AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
    model.eval()
    W=model.lm_head.weight.detach().float().cpu().contiguous(); V,D=W.shape
    s=W.abs().amax(1).clamp_min(1e-8)/127.0
    q=torch.round(W/s[:,None]).clamp(-127,127).to(torch.int8)
    ag=load_dataset('ag_news', split='test')
    texts=[ag[i]['text'] for i in range(N_AG)]
    log('collect AG'); h_ag=collect_last(model,tok,texts)
    log('collect generation'); h_gen=collect_gen(model,tok)
    log('evaluate AG'); a=eval_domain(h_ag,W,q,s); log(str(a))
    log('evaluate generation'); g=eval_domain(h_gen,W,q,s); log(str(g))
    report={'kind':'crossmodel_rowwise_4plus4_proofbits','model':MODEL,'vocab':V,'hidden_dim':D,
      'extra_certificate_metadata_bytes':0,'ag_news':a,'autoregressive_math_code':g,
      'caveat':'Idealized weight/scale traffic; no GPU wall-clock measurement.'}
    OUT.write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
