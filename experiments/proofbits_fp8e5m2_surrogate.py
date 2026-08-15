import json, math, time
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL='Qwen/Qwen2.5-0.5B-Instruct'
N_AG=64; N_GEN=64; PILOT_K=4; MAX_LEN=128
GROUP_SIZES=[896,448,224,128,64,32]
OUT=Path('experiments/artifacts/proofbits_fp8e5m2_surrogate.json'); OUT.parent.mkdir(parents=True,exist_ok=True)
torch.set_grad_enabled(False); torch.manual_seed(53); np.random.seed(53)

def log(x): print(f'[{time.strftime("%H:%M:%S")}] {x}',flush=True)
def hidden(m,**kw): return m.model(**kw,use_cache=False,return_dict=True).last_hidden_state

def collect_ag(m,tok):
    ds=load_dataset('ag_news',split='test'); hs=[]
    for i in range(N_AG):
        x=tok(ds[i]['text'],return_tensors='pt',truncation=True,max_length=MAX_LEN)
        hs.append(hidden(m,**x)[0,-1].float().cpu())
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

def make_e5m2_prefix(W):
    W16=W.half().contiguous(); bits=W16.view(torch.int16).to(torch.int32)&0xffff
    hi=((bits>>8)&0xff).to(torch.uint8).contiguous()
    exp=(hi.to(torch.int16)>>2)&0x1f
    if bool((exp==0x1f).any()): raise RuntimeError('model contains FP16 Inf/NaN high-byte pattern')
    # Numeric value represented by the same S-E5-M2 prefix with no unread fraction bits:
    # exactly FP16 with low byte zero. For finite normals/subnormals this is the
    # E5M2 value with the same bit fields.
    qraw=(hi.to(torch.int16)<<8).contiguous()
    Q=qraw.view(torch.float16).float().contiguous()
    return W16.float().contiguous(),hi,Q

def evaluate(h,exact,coarse,R,group_size,V,D):
    assert D%group_size==0
    ng=D//group_size
    hg=h.float().reshape(h.shape[0],ng,group_size)
    rg=R.reshape(V,ng,group_size)
    # Exact FP32 residual L2 norms as a first certificate oracle.
    hn=torch.linalg.vector_norm(hg,ord=2,dim=2)
    rn=torch.linalg.vector_norm(rg,ord=2,dim=2)
    E=hn@rn.T
    U=coarse+E
    pilots=torch.topk(U,k=PILOT_K,dim=1).indices
    B=exact.gather(1,pilots).amax(1)
    cand=U>=B[:,None]; ref=exact.argmax(1)
    cnt=[]; ok=[]; hit=[]
    for n in range(h.shape[0]):
        idx=cand[n].nonzero().squeeze(1); cnt.append(int(idx.numel()));
        ok.append(int(idx[exact[n,idx].argmax()])==int(ref[n])); hit.append(bool((pilots[n]==ref[n]).any()))
    c=np.asarray(cnt,float); f=c.mean()/V
    # FP32 metadata traffic: ng floats/row. Also show hypothetical outward-safe FP16
    # metadata ceiling; exact implementation of outward-rounded FP16 norms is future work.
    meta32=4.0*ng/D; meta16=2.0*ng/D
    bytes32=1.0+meta32+f
    bytes16=1.0+meta16+f
    return {'group_size':group_size,'groups_per_row':ng,'all_exact':bool(all(ok)),'pilot_contains_winner_rate':float(np.mean(hit)),
      'candidate_mean':float(c.mean()),'median':float(np.median(c)),'p90':float(np.percentile(c,90)),'p99':float(np.percentile(c,99)),'max':int(c.max()),'candidate_fraction':float(f),
      'fp32_norm_metadata_bytes_per_weight':float(meta32),'idealized_bytes_per_weight_with_fp32_norms':float(bytes32),'idealized_reduction_vs_dense_fp16_fp32norms':float(2.0/bytes32),
      'hypothetical_fp16_norm_metadata_bytes_per_weight':float(meta16),'hypothetical_reduction_if_safe_fp16_norms':float(2.0/bytes16)}

def main():
    log('load model'); tok=AutoTokenizer.from_pretrained(MODEL); m=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float32); m.eval(); W=m.lm_head.weight.detach().float().cpu().contiguous(); V,D=W.shape
    log('collect AG'); ha=collect_ag(m,tok); log('collect autoregressive'); hg=collect_gen(m,tok); del m
    log('construct lossless FP16 + E5M2-compatible prefix numeric plane'); W16,hi,Q=make_e5m2_prefix(W); del W
    R=(W16-Q).contiguous()
    # Verify prefix numeric values equal FP16 low-byte-zero exactly at the bit level by construction.
    exact_ag=score(ha,W16); coarse_ag=score(ha,Q); exact_g=score(hg,W16); coarse_g=score(hg,Q)
    report={'kind':'proofbits_e5m2_prefix_tensorcore_surrogate','model':MODEL,'vocab':V,'hidden_dim':D,
      'prefix':'raw FP16 high byte has S1-E5-M2 fields; numeric coarse plane equals FP16 with low byte zero',
      'certificate':'coarse h^T Q plus sum_g ||h_g||_2 ||R_i,g||_2; exact residual norms in FP32 metadata for this kill-test',
      'domains':{'ag_news':[],'autoregressive_math_code':[]}}
    for gs in GROUP_SIZES:
        if D%gs: continue
        log(f'group_size={gs}')
        a=evaluate(ha,exact_ag,coarse_ag,R,gs,V,D); g=evaluate(hg,exact_g,coarse_g,R,gs,V,D)
        report['domains']['ag_news'].append(a); report['domains']['autoregressive_math_code'].append(g); log('AG '+str(a)); log('GEN '+str(g))
    report['caveat']='This establishes certificate tightness/traffic potential only. No FP8 Tensor Core kernel is executed on the CPU runner. FP32 residual-norm metadata is exact for the experiment; compressed/outward-rounded metadata and finite-precision GEMM safety need separate proof.'
    OUT.write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=='__main__':main()
