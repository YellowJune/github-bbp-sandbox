import json, time
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL='Qwen/Qwen2.5-0.5B-Instruct'
N_AG=64; N_GEN=64; PILOT_K=4; MAX_LEN=128
GROUP_SIZES=[896,448,224,128,64]
OUT=Path('experiments/artifacts/proofbits_fp8e5m2_signsplit.json'); OUT.parent.mkdir(parents=True,exist_ok=True)
torch.set_grad_enabled(False); torch.manual_seed(59); np.random.seed(59)

def log(x): print(f'[{time.strftime("%H:%M:%S")}] {x}',flush=True)
def hidden(m,**kw): return m.model(**kw,use_cache=False,return_dict=True).last_hidden_state

def collect_ag(m,tok):
    ds=load_dataset('ag_news',split='test'); hs=[]
    for i in range(N_AG):
        x=tok(ds[i]['text'],return_tensors='pt',truncation=True,max_length=MAX_LEN); hs.append(hidden(m,**x)[0,-1].float().cpu())
    return torch.stack(hs)
def collect_gen(m,tok):
    ps=['Solve carefully: 137 * 29 =','The derivative of x^3 + 2x is','Write a Python function for binary search:','Factor 84 into prime factors:','Compute 2^10 =','Write SQL selecting users older than 18:','The gcd of 84 and 126 is','Simplify (x+2)(x-2):']
    seq=[tok(p,return_tensors='pt')['input_ids'] for p in ps]; hs=[]
    while len(hs)<N_GEN:
        ns=[]
        for ids in seq:
            h=hidden(m,input_ids=ids)[0,-1].float().cpu(); hs.append(h)
            if len(hs)>=N_GEN: break
            nxt=m.lm_head(h[None,:])[0].argmax().view(1,1); ns.append(torch.cat([ids,nxt],1))
        seq=ns
    return torch.stack(hs[:N_GEN])
def score(h,w,ch=4096): return torch.cat([h.float()@w[a:a+ch].float().T for a in range(0,w.shape[0],ch)],1)

def split_prefix(W):
    W16=W.half().contiguous(); bits=W16.view(torch.int16).to(torch.int32)&0xffff
    hi=((bits>>8)&255).to(torch.uint8).contiguous(); sign=((hi>>7)&1).bool()
    if bool((((hi.to(torch.int16)>>2)&31)==31).any()): raise RuntimeError('nonfinite')
    Q=(hi.to(torch.int16)<<8).contiguous().view(torch.float16).float().contiguous()
    R=(W16.float()-Q).contiguous()
    # For finite values, residual must point in the weight-sign direction or be zero.
    pos=(~sign); neg=sign
    if not bool((R[pos]>=0).all() and (R[neg]<=0).all()): raise RuntimeError('residual sign invariant failed')
    return W16.float().contiguous(),hi,Q,R,pos,neg

def evaluate(h,exact,coarse,R,pos,neg,gs,V,D):
    assert D%gs==0; ng=D//gs
    Rg=R.reshape(V,ng,gs); Pg=pos.reshape(V,ng,gs); Ng=neg.reshape(V,ng,gs)
    # Magnitude residual norms separated by weight sign.
    rpos=torch.linalg.vector_norm(torch.where(Pg,Rg,torch.zeros_like(Rg)),ord=2,dim=2)
    rneg=torch.linalg.vector_norm(torch.where(Ng,-Rg,torch.zeros_like(Rg)),ord=2,dim=2)
    hg=h.float().reshape(h.shape[0],ng,gs)
    hp=torch.clamp(hg,min=0); hn=torch.clamp(-hg,min=0)
    hp_norm=torch.linalg.vector_norm(hp,ord=2,dim=2); hn_norm=torch.linalg.vector_norm(hn,ord=2,dim=2)
    # Mismatched-sign residual terms are <=0 and can be dropped from an upper bound.
    E=hp_norm@rpos.T + hn_norm@rneg.T
    U=coarse+E; pilots=torch.topk(U,k=PILOT_K,dim=1).indices; B=exact.gather(1,pilots).amax(1); cand=U>=B[:,None]; ref=exact.argmax(1)
    cnt=[];ok=[];hit=[]
    for n in range(h.shape[0]):
        idx=cand[n].nonzero().squeeze(1);cnt.append(int(idx.numel()));ok.append(int(idx[exact[n,idx].argmax()])==int(ref[n]));hit.append(bool((pilots[n]==ref[n]).any()))
    c=np.asarray(cnt,float); f=c.mean()/V
    # Two FP32 norm planes per group per row.
    meta32=8.0*ng/D; meta16=4.0*ng/D
    bytes32=1.0+meta32+f; bytes16=1.0+meta16+f
    return {'group_size':gs,'groups_per_row':ng,'all_exact':bool(all(ok)),'pilot_contains_winner_rate':float(np.mean(hit)),'candidate_mean':float(c.mean()),'median':float(np.median(c)),'p90':float(np.percentile(c,90)),'p99':float(np.percentile(c,99)),'max':int(c.max()),'candidate_fraction':float(f),
      'fp32_signsplit_norm_metadata_bytes_per_weight':float(meta32),'total_idealized_bytes_per_weight_fp32meta':float(bytes32),'idealized_reduction_vs_dense_fp16_fp32meta':float(2.0/bytes32),
      'hypothetical_fp16_outward_norm_metadata_bytes_per_weight':float(meta16),'hypothetical_reduction_fp16meta':float(2.0/bytes16)}

def main():
    log('load model');tok=AutoTokenizer.from_pretrained(MODEL);m=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float32);m.eval();W=m.lm_head.weight.detach().float().cpu().contiguous();V,D=W.shape
    log('collect AG');ha=collect_ag(m,tok);log('collect gen');hg=collect_gen(m,tok);del m
    log('split E5M2 prefix/residual');W16,hi,Q,R,pos,neg=split_prefix(W);del W
    ea=score(ha,W16);ca=score(ha,Q);eg=score(hg,W16);cg=score(hg,Q)
    rep={'kind':'proofbits_e5m2_signsplit_residual_certificate','model':MODEL,'vocab':V,'hidden_dim':D,
      'certificate':'h^T Q + sum_g ||h_{g,+}||2 ||R_{i,g,+}||2 + ||h_{g,-}||2 ||R_{i,g,-}||2; mismatched residual signs only lower the score',
      'domains':{'ag_news':[],'autoregressive_math_code':[]}}
    for gs in GROUP_SIZES:
        if D%gs:continue
        log('gs='+str(gs));a=evaluate(ha,ea,ca,R,pos,neg,gs,V,D);g=evaluate(hg,eg,cg,R,pos,neg,gs,V,D);rep['domains']['ag_news'].append(a);rep['domains']['autoregressive_math_code'].append(g);log('AG '+str(a));log('GEN '+str(g))
    rep['caveat']='Certificate geometry/traffic kill-test only. FP32 metadata norms are exact here; outward-rounded compressed norms and FP8 GEMM finite-precision safety remain to be proved and benchmarked on GPU.'
    OUT.write_text(json.dumps(rep,indent=2));print(json.dumps(rep,indent=2))
if __name__=='__main__':main()
