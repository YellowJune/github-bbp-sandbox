import json, time
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL='Qwen/Qwen2.5-0.5B-Instruct'
N_AG=64; N_GEN=64; PILOT_K=4; MAX_LEN=128
TOPKS=[0,2,4,8,16,32,64]; CHUNK=2048
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
    Q=(hi.to(torch.int16)<<8).contiguous().view(torch.float16).float().contiguous(); R=(W16.float()-Q).contiguous(); R16=R.half()
    return W16.float().contiguous(),Q,R,R16,bool((R16.float()==R).all())

def precompute(R,R16):
    maxk=max(TOPKS); idx64=torch.topk(R.abs(),k=maxk,dim=1,largest=True,sorted=True).indices; vals64=R16.gather(1,idx64)
    pos_sq=torch.clamp(R,min=0).square().sum(1); neg_sq=torch.clamp(-R,min=0).square().sum(1)
    return idx64,vals64,pos_sq,neg_sq

def evaluate(h,exact,coarse,idx64,vals64,pos_sq,neg_sq,K,V,D):
    if K:
        idx=idx64[:,:K]; vals=vals64[:,:K].float(); sel_sq=vals.square(); pos_sel=torch.where(vals>0,sel_sq,torch.zeros_like(sel_sq)).sum(1); neg_sel=torch.where(vals<0,sel_sq,torch.zeros_like(sel_sq)).sum(1)
        npos=torch.sqrt(torch.clamp(pos_sq-pos_sel,min=0)); nneg=torch.sqrt(torch.clamp(neg_sq-neg_sel,min=0))
    else:
        idx=idx64[:,:0]; vals=vals64[:,:0].float(); npos=torch.sqrt(pos_sq); nneg=torch.sqrt(neg_sq)
    hp2=torch.linalg.vector_norm(torch.clamp(h,min=0),ord=2,dim=1); hn2=torch.linalg.vector_norm(torch.clamp(-h,min=0),ord=2,dim=1)
    parts=[]
    for a in range(0,V,CHUNK):
        b=min(V,a+CHUNK); u=coarse[:,a:b].clone()
        if K:
            ic=idx[a:b]; vc=vals[a:b]
            # h[:, ic] -> [N,C,K], bounded by CHUNK.
            u += (h[:,ic]*vc[None,:,:]).sum(2)
        u += hp2[:,None]*npos[a:b][None,:] + hn2[:,None]*nneg[a:b][None,:]
        parts.append(u)
    U=torch.cat(parts,1)
    pilots=torch.topk(U,k=PILOT_K,dim=1).indices; B=exact.gather(1,pilots).amax(1); cand=U>=B[:,None]; ref=exact.argmax(1)
    cnt=[];ok=[];hit=[]
    for n in range(h.shape[0]):
        ci=cand[n].nonzero().squeeze(1);cnt.append(int(ci.numel()));ok.append(int(ci[exact[n,ci].argmax()])==int(ref[n]));hit.append(bool((pilots[n]==ref[n]).any()))
    c=np.asarray(cnt,float); f=c.mean()/V; meta_bytes_row=4*K+8; meta_per_weight=meta_bytes_row/D; total=1.0+meta_per_weight+f; meta16=(4*K+4)/D; total16=1.0+meta16+f
    return {'topk_atoms':K,'all_exact':bool(all(ok)),'pilot_contains_winner_rate':float(np.mean(hit)),'candidate_mean':float(c.mean()),'median':float(np.median(c)),'p90':float(np.percentile(c,90)),'p99':float(np.percentile(c,99)),'max':int(c.max()),'candidate_fraction':float(f),'metadata_bytes_per_row_fp32_remainder_norms':int(meta_bytes_row),'metadata_bytes_per_weight':float(meta_per_weight),'total_idealized_bytes_per_weight':float(total),'idealized_reduction_vs_dense_fp16':float(2.0/total),'hypothetical_total_bytes_with_fp16_outward_remainder_norms':float(total16),'hypothetical_reduction_fp16norms':float(2.0/total16)}

def main():
    log('load');tok=AutoTokenizer.from_pretrained(MODEL);m=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float32);m.eval();W=m.lm_head.weight.detach().float().cpu().contiguous();V,D=W.shape
    log('collect AG');ha=collect_ag(m,tok);log('collect gen');hg=collect_gen(m,tok);del m
    log('prefix/residual');W16,Q,R,R16,repr_ok=split(W);del W
    log('precompute top residual atoms');idx64,vals64,pos_sq,neg_sq=precompute(R,R16);del R,R16
    ea=score(ha,W16);ca=score(ha,Q);eg=score(hg,W16);cg=score(hg,Q)
    rep={'kind':'proofbits_fp8_e5m2_topk_residual_atom_certificate','model':MODEL,'vocab':V,'hidden_dim':D,'residual_exactly_fp16_representable':repr_ok,'certificate':'h^T Q + exact dot over stored top-K residual atoms + sign-split L2 upper bound on residual remainder','domains':{'ag_news':[],'autoregressive_math_code':[]}}
    for K in TOPKS:
        log('K='+str(K));a=evaluate(ha,ea,ca,idx64,vals64,pos_sq,neg_sq,K,V,D);g=evaluate(hg,eg,cg,idx64,vals64,pos_sq,neg_sq,K,V,D);rep['domains']['ag_news'].append(a);rep['domains']['autoregressive_math_code'].append(g);log('AG '+str(a));log('GEN '+str(g))
    rep['caveat']='CPU certificate/traffic kill-test. Top-K atom metadata uses exact FP16 residual plus uint16 coordinate. GPU locality, outward-rounded remainder norms, and finite-precision FP8 GEMM safety remain unmeasured.'
    OUT.write_text(json.dumps(rep,indent=2));print(json.dumps(rep,indent=2))
if __name__=='__main__':main()
