import json, math, time
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL='HuggingFaceTB/SmolLM2-360M-Instruct'
N_AG=64; N_GEN=64; MAX_LEN=128
GROUPS=[16,32,64]
PREFIX_BITS=[4,5,6]
OUT=Path('experiments/artifacts/proofbits_smol_groupwise_rescue.json'); OUT.parent.mkdir(parents=True,exist_ok=True)
torch.set_grad_enabled(False); torch.manual_seed(0); np.random.seed(0)

def log(x): print(f"[{time.strftime('%H:%M:%S')}] {x}",flush=True)

def hidden(model, **kw): return model.model(**kw,use_cache=False,return_dict=True).last_hidden_state

def collect_last(model,tok,texts):
    hs=[]
    for t in texts:
        x=tok(t,return_tensors='pt',truncation=True,max_length=MAX_LEN)
        hs.append(hidden(model,**x)[0,-1].float().cpu())
    return torch.stack(hs)

def collect_gen(model,tok):
    ps=['Solve carefully: 137 * 29 =','The derivative of x^3 + 2x is','Write a Python function for binary search:',
        'Factor 84 into prime factors:','Compute 2^10 =','Write SQL selecting users older than 18:',
        'The gcd of 84 and 126 is','Simplify (x+2)(x-2):']
    seq=[tok(p,return_tensors='pt')['input_ids'] for p in ps]; hs=[]
    while len(hs)<N_GEN:
        ns=[]
        for ids in seq:
            h=hidden(model,input_ids=ids)[0,-1].float().cpu(); hs.append(h)
            if len(hs)>=N_GEN: break
            nxt=model.lm_head(h[None,:])[0].argmax().view(1,1); ns.append(torch.cat([ids,nxt],1))
        seq=ns
    return torch.stack(hs[:N_GEN])

def quant_group(W,G):
    V,D=W.shape; assert D%G==0
    ng=D//G; X=W.reshape(V,ng,G)
    s=X.abs().amax(2).clamp_min(1e-8)/127.0
    q=torch.round(X/s[:,:,None]).clamp(-127,127).to(torch.int8)
    return q,s

def prefix_mid(q,b):
    u=q.to(torch.int16)+128; step=1<<(8-b); lo=(u//step)*step
    return lo.float()+(step-1)*0.5-128.0

def score_group(h,mat,s,chunk=2048):
    V,ng,G=mat.shape; N=h.shape[0]; hg=h.float().reshape(N,ng,G); ys=[]
    for a in range(0,V,chunk):
        m=mat[a:a+chunk].float()
        # [N,ng,G] x [C,ng,G] -> [N,C,ng], then scales and sum
        dots=torch.einsum('nkg,ckg->nck',hg,m)
        ys.append((dots*s[a:a+chunk][None,:,:]).sum(2))
    return torch.cat(ys,1)

def fp_score(h,W,chunk=4096):
    return torch.cat([h.float()@W[a:a+chunk].float().T for a in range(0,W.shape[0],chunk)],1)

def evaluate(h,q,s,G,b,z8):
    V,ng,_=q.shape; N=h.shape[0]; mid=prefix_mid(q,b); coarse=score_group(h,mid,s)
    pilot=coarse.argmax(1); B=z8.gather(1,pilot[:,None]).squeeze(1)
    # safe scale-only groupwise bound
    r=(2**(8-b)-1)/2.0
    hn=h.float().reshape(N,ng,G).square().sum(2).sqrt()
    err=math.sqrt(G)*r*(hn @ s.T)
    cand=(coarse+err)>=B[:,None]
    ref=z8.argmax(1); pred=[]; cnt=[]
    for n in range(N):
        idx=cand[n].nonzero().squeeze(1); cnt.append(int(idx.numel())); pred.append(int(idx[z8[n,idx].argmax()]))
    cnt=np.asarray(cnt); frac=cnt.mean()/V
    scale_bits=16.0/G; dense=8.0+scale_bits; pb=b+scale_bits+(8-b)*frac
    return {'group':G,'prefix_bits':b,'certified_exact_match_rate':float((torch.tensor(pred)==ref).float().mean()),
      'candidate_mean':float(cnt.mean()),'median':float(np.median(cnt)),'p90':float(np.percentile(cnt,90)),
      'p99':float(np.percentile(cnt,99)),'max':int(cnt.max()),'candidate_fraction_mean':float(frac),
      'proofbits_bits_per_weight':float(pb),'dense_groupwise_int8_bits_per_weight':float(dense),
      'idealized_bw_reduction':float(dense/pb)}

def main():
    log('load model'); tok=AutoTokenizer.from_pretrained(MODEL); model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float32); model.eval()
    W=model.lm_head.weight.detach().float().cpu().contiguous(); V,D=W.shape
    ag=load_dataset('ag_news',split='test'); texts=[ag[i]['text'] for i in range(N_AG)]
    log('collect AG'); ha=collect_last(model,tok,texts); log('collect gen'); hg=collect_gen(model,tok)
    zfp_a=fp_score(ha,W); zfp_g=fp_score(hg,W)
    rep={'kind':'smollm2_groupwise_proofbits_rescue','model':MODEL,'vocab':V,'hidden_dim':D,'domains':{}}
    for G in GROUPS:
        log(f'quant G={G}'); q,s=quant_group(W,G)
        z8a=score_group(ha,q,s); z8g=score_group(hg,q,s)
        base={'group':G,'int8_fp32_agreement_ag':float((z8a.argmax(1)==zfp_a.argmax(1)).float().mean()),
              'int8_fp32_agreement_gen':float((z8g.argmax(1)==zfp_g.argmax(1)).float().mean())}
        arr=[]
        for b in PREFIX_BITS:
            ra=evaluate(ha,q,s,G,b,z8a); rg=evaluate(hg,q,s,G,b,z8g)
            arr.append({'prefix_bits':b,'ag_news':ra,'autoregressive_math_code':rg}); log(f'G={G} b={b} AG={ra} GEN={rg}')
        rep['domains'][str(G)]={'baseline':base,'prefixes':arr}
    rep['caveat']='Idealized weight+FP16-scale traffic; no GPU wall-clock. Exactness is relative to each deployed groupwise INT8 head.'
    OUT.write_text(json.dumps(rep,indent=2)); print(json.dumps(rep,indent=2))
if __name__=='__main__': main()
