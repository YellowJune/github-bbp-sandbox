import json, math, time
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL='Qwen/Qwen2.5-0.5B-Instruct'
N_AG=96; N_GEN=96; MAX_LEN=128
BLOCK_COUNTS=[1,2,4,8,16]
PILOT_KS=[1,4,8,16]
OUT=Path('experiments/artifacts/proofbits_certificate_sketch.json'); OUT.parent.mkdir(parents=True,exist_ok=True)
torch.set_grad_enabled(False); torch.manual_seed(0); np.random.seed(0)

def log(x): print(f"[{time.strftime('%H:%M:%S')}] {x}", flush=True)

def base_hidden(model, **kw): return model.model(**kw,use_cache=False,return_dict=True).last_hidden_state

def collect_last(model,tok,texts):
    hs=[]
    for t in texts:
        x=tok(t,return_tensors='pt',truncation=True,max_length=MAX_LEN)
        hs.append(base_hidden(model,**x)[0,-1].float().cpu())
    return torch.stack(hs)

def collect_gen(model,tok):
    ps=['Solve carefully: 137 * 29 =','The derivative of x^3 + 2x is','Write a Python function for binary search:',
        'Factor 84 into prime factors:','Compute 2^10 =','Write SQL selecting users older than 18:',
        'The gcd of 84 and 126 is','Simplify (x+2)(x-2):']
    seq=[tok(p,return_tensors='pt')['input_ids'] for p in ps]; hs=[]
    while len(hs)<N_GEN:
        ns=[]
        for ids in seq:
            h=base_hidden(model,input_ids=ids)[0,-1].float().cpu(); hs.append(h)
            if len(hs)>=N_GEN: break
            nxt=model.lm_head(h[None,:])[0].argmax().view(1,1); ns.append(torch.cat([ids,nxt],1))
        seq=ns
    return torch.stack(hs[:N_GEN])

def mid4(q):
    u=q.to(torch.int16)+128; lo=(u//16)*16
    return lo.float()+7.5-128.0

def chunk_scores(h,mat,scale=None,chunk=4096):
    ys=[]
    for a in range(0,mat.shape[0],chunk):
        m=mat[a:a+chunk].float()
        y=h.float()@m.T
        if scale is not None: y=y*scale[a:a+chunk][None,:]
        ys.append(y)
    return torch.cat(ys,1)

def residual_block_norms(resid, B):
    V,D=resid.shape; assert D%B==0
    return resid.reshape(V,B,D//B).float().square().sum(2).sqrt().contiguous()

def evaluate(h,q,s,coarse,z8,resid,B,pilot_k):
    V,D=q.shape
    rn=residual_block_norms(resid,B) # exact FP32 certificate sketch
    hn=h.float().reshape(h.shape[0],B,D//B).square().sum(2).sqrt()
    err=(hn@rn.T)*s[None,:]
    top=torch.topk(coarse,k=pilot_k,dim=1).indices
    Bscore=z8.gather(1,top).amax(1)
    cand=(coarse+err)>=Bscore[:,None]
    ref=z8.argmax(1); pred=[]; cnt=[]
    for n in range(h.shape[0]):
        idx=cand[n].nonzero().squeeze(1); cnt.append(int(idx.numel())); pred.append(int(idx[z8[n,idx].argmax()]))
    cnt=np.asarray(cnt); frac=cnt.mean()/V
    scale_bits=16.0/D
    cert_bits=32.0*B/D
    pilot_bits=4.0*pilot_k/V
    pb=4.0+scale_bits+cert_bits+4.0*frac+pilot_bits
    dense=8.0+scale_bits
    return {'certificate_blocks':B,'pilot_k':pilot_k,'certified_match_rate':float((torch.tensor(pred)==ref).float().mean()),
      'candidate_mean':float(cnt.mean()),'median':float(np.median(cnt)),'p90':float(np.percentile(cnt,90)),
      'p99':float(np.percentile(cnt,99)),'max':int(cnt.max()),'candidate_fraction_mean':float(frac),
      'scale_bits_per_weight':scale_bits,'certificate_metadata_bits_per_weight':cert_bits,
      'proofbits_total_bits_per_weight':float(pb),'dense_rowwise_int8_bits_per_weight':dense,
      'idealized_bw_reduction':float(dense/pb)}

def main():
    log('load model'); tok=AutoTokenizer.from_pretrained(MODEL); model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float32); model.eval()
    W=model.lm_head.weight.detach().float().cpu().contiguous(); V,D=W.shape
    s=W.abs().amax(1).clamp_min(1e-8)/127.0; q=torch.round(W/s[:,None]).clamp(-127,127).to(torch.int8)
    m4=mid4(q); resid=q.float()-m4
    ag=load_dataset('ag_news',split='test'); texts=[ag[i]['text'] for i in range(N_AG)]
    log('collect AG'); h_ag=collect_last(model,tok,texts)
    log('collect gen'); h_gen=collect_gen(model,tok)
    report={'kind':'proofbits_certificate_sketch_pareto','model':MODEL,'vocab':V,'hidden_dim':D,
      'sketch':'B blockwise FP32 residual L2 norms per vocabulary row; blockwise Cauchy bound','domains':{}}
    for name,h in [('ag_news',h_ag),('autoregressive_math_code',h_gen)]:
        log('scores '+name); z8=chunk_scores(h,q,s); coarse=chunk_scores(h,m4,s)
        arr=[]
        for B in BLOCK_COUNTS:
            for k in PILOT_KS:
                r=evaluate(h,q,s,coarse,z8,resid,B,k); arr.append(r); log(f'{name} B={B} k={k}: {r}')
        report['domains'][name]=arr
    report['caveat']='Traffic accounting assumes FP32 certificate norms for unconditional upper bounds. Actual GPU wall-clock and compaction overhead are not measured.'
    OUT.write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
