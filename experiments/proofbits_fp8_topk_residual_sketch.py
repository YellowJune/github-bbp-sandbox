import json, time
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL='Qwen/Qwen2.5-0.5B-Instruct'
N_AG=64; N_GEN=64; PILOT_K=4; MAX_LEN=128
TOPKS=[0,2,4,8,16,32,64]
OUT=Path('experiments/artifacts/proofbits_fp8_topk_residual_sketch.json'); OUT.parent.mkdir(parents=True,exist_ok=True)
torch.set_grad_enabled(False); torch.manual_seed(61); np.random.seed(61)

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

def split(W):
    W16=W.half().contiguous(); raw=W16.view(torch.int16).to(torch.int32)&0xffff; hi=((raw>>8)&255).to(torch.uint8).contiguous()
    if bool((((hi.to(torch.int16)>>2)&31)==31).any()): raise RuntimeError('nonfinite')
    Q=(hi.to(torch.int16)<<8).contiguous().view(torch.float16).float().contiguous(); R=(W16.float()-Q).contiguous()
    # residual is exactly representable as FP16 for these finite prefix completions
    R16=R.half(); exact_representable=bool((R16.float()==R).all())
    return W16.float().contiguous(),Q,R,R16,exact_representable

def build_sketch(R,R16,K):
    V,D=R.shape
    if K==0:
        idx=torch.empty(V,0,dtype=torch.int64); vals=torch.empty(V,0,dtype=torch.float16); rem=R
    else:
        idx=torch.topk(R.abs(),k=K,dim=1,largest=True,sorted=False).indices
        vals=R16.gather(1,idx)
        rem=R.clone(); rem.scatter_(1,idx,0.0)
    # sign-split L2 remainder norms, two FP32 values/row
    rp=torch.clamp(rem,min=0); rn=torch.clamp(-rem,min=0)
    npos=torch.linalg.vector_norm(rp,ord=2,dim=1).contiguous(); nneg=torch.linalg.vector_norm(rn,ord=2,dim=1).contiguous()
    return idx,vals,npos,nneg

def evaluate(h,exact,coarse,R,R16,K,V,D):
    idx,vals,npos,nneg=build_sketch(R,R16,K)
    # exact atom correction from metadata
    if K:
        # [N,V,K] gather via advanced indexing; CPU kill-test only.
        hatoms=h[:,idx]  # PyTorch broadcasts N x V x K
        atom=(hatoms*vals.float()[None,:,:]).sum(2)
    else:
        atom=torch.zeros_like(coarse)
    hp=torch.clamp(h,min=0); hn=torch.clamp(-h,min=0)
    hp2=torch.linalg.vector_norm(hp,ord=2,dim=1); hn2=torch.linalg.vector_norm(hn,ord=2,dim=1)
    E=hp2[:,None]*npos[None,:]+hn2[:,None]*nneg[None,:]
    U=coarse+atom+E
    pilots=torch.topk(U,k=PILOT_K,dim=1).indices; B=exact.gather(1,pilots).amax(1); cand=U>=B[:,None]; ref=exact.argmax(1)
    cnt=[];ok=[];hit=[]
    for n in range(h.shape[0]):
        ci=cand[n].nonzero().squeeze(1);cnt.append(int(ci.numel()));ok.append(int(ci[exact[n,ci].argmax()])==int(ref[n]));hit.append(bool((pilots[n]==ref[n]).any()))
    c=np.asarray(cnt,float);f=c.mean()/V
    # Each exact atom: uint16 hidden index + FP16 signed residual = 4 bytes/row.
    # Plus two FP32 remainder norms = 8 bytes/row.
    meta_bytes_row=4*K+8
    meta_per_weight=meta_bytes_row/D
    total=1.0+meta_per_weight+f
    # Hypothetical two outward-rounded FP16 remainder norms instead: 4 bytes total norms.
    meta16=(4*K+4)/D; total16=1.0+meta16+f
    return {'topk_atoms':K,'all_exact':bool(all(ok)),'pilot_contains_winner_rate':float(np.mean(hit)),
      'candidate_mean':float(c.mean()),'median':float(np.median(c)),'p90':float(np.percentile(c,90)),'p99':float(np.percentile(c,99)),'max':int(c.max()),'candidate_fraction':float(f),
      'metadata_bytes_per_row_fp32_remainder_norms':int(meta_bytes_row),'metadata_bytes_per_weight':float(meta_per_weight),'total_idealized_bytes_per_weight':float(total),'idealized_reduction_vs_dense_fp16':float(2.0/total),
      'hypothetical_total_bytes_with_fp16_outward_remainder_norms':float(total16),'hypothetical_reduction_fp16norms':float(2.0/total16)}

def main():
    log('load');tok=AutoTokenizer.from_pretrained(MODEL);m=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float32);m.eval();W=m.lm_head.weight.detach().float().cpu().contiguous();V,D=W.shape
    log('collect AG');ha=collect_ag(m,tok);log('collect gen');hg=collect_gen(m,tok);del m
    log('prefix/residual');W16,Q,R,R16,repr_ok=split(W);del W
    ea=score(ha,W16);ca=score(ha,Q);eg=score(hg,W16);cg=score(hg,Q)
    rep={'kind':'proofbits_fp8_e5m2_topk_residual_atom_certificate','model':MODEL,'vocab':V,'hidden_dim':D,'residual_exactly_fp16_representable':repr_ok,
      'certificate':'h^T Q + exact dot over stored top-K residual atoms + sign-split L2 upper bound on residual remainder','domains':{'ag_news':[],'autoregressive_math_code':[]}}
    for K in TOPKS:
        log('K='+str(K));a=evaluate(ha,ea,ca,R,R16,K,V,D);g=evaluate(hg,eg,cg,R,R16,K,V,D);rep['domains']['ag_news'].append(a);rep['domains']['autoregressive_math_code'].append(g);log('AG '+str(a));log('GEN '+str(g))
    rep['caveat']='CPU certificate/traffic kill-test. Top-K atom metadata uses exact FP16 residual plus uint16 coordinate. GPU implementation, metadata locality, outward-rounded remainder norms, and finite-precision coarse GEMM safety remain unmeasured.'
    OUT.write_text(json.dumps(rep,indent=2));print(json.dumps(rep,indent=2))
if __name__=='__main__':main()
