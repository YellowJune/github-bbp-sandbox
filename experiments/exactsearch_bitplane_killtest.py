import json, os, time, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID='Qwen/Qwen2.5-0.5B-Instruct'
SEED=20260815
N=128
MAX_LEN=64
BATCH=8
GROUPS=[32,64,128]
PILOT=1

def log(x): print(time.strftime('[%H:%M:%S]'),x,flush=True)

def pick(split,n):
    g=torch.Generator().manual_seed(SEED+123)
    out=[]
    for i in torch.randperm(len(split),generator=g).tolist():
        t=split[i]['text'].strip()
        if len(t)>=40: out.append(t)
        if len(out)==n:return out
    raise RuntimeError

def hiddens(model,tok,ts):
    out=[]
    with torch.inference_mode():
        for s in range(0,len(ts),BATCH):
            e=tok(ts[s:s+BATCH],return_tensors='pt',padding=True,truncation=True,max_length=MAX_LEN)
            y=model.model(input_ids=e['input_ids'],attention_mask=e['attention_mask'],use_cache=False,return_dict=True).last_hidden_state
            last=e['attention_mask'].sum(1)-1
            out.append(y[torch.arange(y.shape[0]),last].float().cpu())
    return torch.cat(out)

def make_int8(W,group):
    V,d=W.shape; assert d%group==0; G=d//group
    X=W.view(V,G,group)
    sc=X.abs().amax(2,keepdim=True).clamp_min(1e-12)/127.0
    q=torch.round(X/sc).clamp(-127,127).to(torch.int16)
    return q,sc.squeeze(2)

def eval_group(W,H,group):
    V,d=W.shape; G=d//group
    q,sc=make_int8(W,group)
    # Full INT8 deployment reference, dequantized exactly from q and scale.
    qf=q.float(); W8=(qf*sc.unsqueeze(2)).reshape(V,d)
    dense8=H@W8.T
    densefp=H@W.T
    fp_agree=float((dense8.argmax(1)==densefp.argmax(1)).float().mean())

    # Lossless bit-plane split of signed int8 code q = 16*coarse + low,
    # with coarse=floor(q/16), low in [0,15]. Reading both reconstructs q exactly.
    coarse=torch.div(q,16,rounding_mode='floor')
    low=q-16*coarse
    assert int(low.min())>=0 and int(low.max())<=15
    Wc=(16.0*coarse.float()*sc.unsqueeze(2)).reshape(V,d)
    approx=H@Wc.T

    # Unread low nibble is token/group-specific integer residual low*scale.
    # Precompute exact L2 norm per token/group for rigorous Cauchy score bound.
    residual=(low.float()*sc.unsqueeze(2))
    rnorm=torch.linalg.vector_norm(residual,dim=2) # V,G
    hnorm=torch.linalg.vector_norm(H.view(H.shape[0],G,group),dim=2)
    eb=hnorm@rnorm.T
    ub=approx+eb+3e-4*(approx.abs()+eb.abs()+1.0)

    counts=[]; matches=0; approx_hits=0
    for j in range(H.shape[0]):
        p=torch.topk(approx[j],k=PILOT).indices
        # Reading the lower nibble for the pilot reconstructs exact INT8 code/score.
        lower=dense8[j,p].max()
        cand=torch.nonzero(ub[j]>=lower,as_tuple=False).flatten()
        union=torch.unique(torch.cat([p,cand]))
        counts.append(int(union.numel()))
        pred=cand[dense8[j,cand].argmax()].item(); true=dense8[j].argmax().item()
        matches+=int(pred==true); approx_hits+=int(true in p.tolist())
    c=torch.tensor(counts,dtype=torch.float64); frac=c/V
    # Runtime bit read: upper nibble (4b) for all weights + lower nibble (4b) only survivors.
    # Relative to reading the full INT8 lm-head once: 0.5 + 0.5*f.
    int8_ratio=0.5+0.5*frac
    # Relative to FP16 dense weight bandwidth: 0.25 + 0.25*f.
    fp16_ratio=0.25+0.25*frac
    return {
      'group':group,'groups':G,'queries':H.shape[0],
      'int8_vs_fp32_argmax_agreement':fp_agree,
      'certified_exact_int8_match_rate':matches/H.shape[0],
      'coarse_top1_is_true_int8_rate':approx_hits/H.shape[0],
      'refined_token_count_mean':float(c.mean()),'median':float(c.median()),
      'p90':float(torch.quantile(c,0.9)),'max':float(c.max()),
      'refined_token_fraction_mean':float(frac.mean()),
      'idealized_weight_bits_per_weight_mean':float(4.0+4.0*frac.mean()),
      'idealized_bandwidth_ratio_vs_dense_int8':float(int8_ratio.mean()),
      'idealized_bandwidth_reduction_vs_dense_int8':float(1.0/int8_ratio.mean()),
      'idealized_bandwidth_ratio_vs_dense_fp16':float(fp16_ratio.mean()),
      'idealized_bandwidth_reduction_vs_dense_fp16':float(1.0/fp16_ratio.mean()),
      'low_nibble_min':int(low.min()),'low_nibble_max':int(low.max())
    }

def main():
    torch.set_num_threads(max(1,min(4,os.cpu_count() or 1)))
    tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    tok.padding_side='right'
    model=AutoModelForCausalLM.from_pretrained(MODEL_ID,torch_dtype=torch.float32,low_cpu_mem_usage=True)
    model.eval(); W=model.lm_head.weight.detach().float().cpu(); V,d=W.shape
    ds=load_dataset('ag_news'); H=hiddens(model,tok,pick(ds['test'],N))
    results=[]
    for g in GROUPS:
        log(f'eval g={g}')
        r=eval_group(W,H,g);results.append(r);log(str(r))
    out={'kind':'lossless_int8_bitplane_exact_lmhead_killtest','model':MODEL_ID,'queries':N,'vocab':V,'hidden_dim':d,
         'pilot':PILOT,'reference':'groupwise symmetric INT8 deployment lm-head; bit-plane split reconstructs exactly the same INT8 codes',
         'results':results,
         'caveat':'Bandwidth numbers are idealized bit-read accounting, not wall-clock. Scale/metadata, packing, gather, and kernel overhead are not included.'}
    os.makedirs('experiments/artifacts',exist_ok=True)
    with open('experiments/artifacts/exactsearch_bitplane_result.json','w') as f: json.dump(out,f,indent=2)
    print('=== BITPLANE_RESULT ===');print(json.dumps(out,indent=2))
if __name__=='__main__': main()
