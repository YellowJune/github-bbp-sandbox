import json, os, time, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID='Qwen/Qwen2.5-0.5B-Instruct'
SEED=20260815
N=128
MAX_LEN=64
BATCH=8
GROUPS=[16,32,64]
MSB_BITS=[3,4,5,6]
PILOT=1

def log(x): print(time.strftime('[%H:%M:%S]'),x,flush=True)

def pick(split,n):
    g=torch.Generator().manual_seed(SEED+321)
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

def int8_group(W,group):
    V,d=W.shape; G=d//group
    X=W.view(V,G,group)
    sc=X.abs().amax(2,keepdim=True).clamp_min(1e-12)/127.0
    q=torch.round(X/sc).clamp(-127,127).to(torch.int16)
    return q,sc.squeeze(2)

def one(W,H,group,msb):
    V,d=W.shape; G=d//group
    q,sc=int8_group(W,group)
    qf=q.float(); W8=(qf*sc.unsqueeze(2)).reshape(V,d)
    exact=H@W8.T
    fp=H@W.T
    int8_fp_agree=float((exact.argmax(1)==fp.argmax(1)).float().mean())

    unread=8-msb; step=1<<unread
    # Shift signed q to unsigned byte u. MSBs identify a contiguous interval of step integer codes.
    # Use that interval's midpoint as coarse value, so unread-bit error is symmetric and <= (step-1)/2 codes.
    u=q+128
    hi=torch.div(u,step,rounding_mode='floor')
    midpoint=hi.float()*step + (step-1)/2.0 - 128.0
    residual=qf-midpoint
    max_code_res=float(residual.abs().max())
    Wc=(midpoint*sc.unsqueeze(2)).reshape(V,d)
    approx=H@Wc.T
    rnorm=torch.linalg.vector_norm(residual*sc.unsqueeze(2),dim=2)
    hnorm=torch.linalg.vector_norm(H.view(H.shape[0],G,group),dim=2)
    eb=hnorm@rnorm.T
    ub=approx+eb+3e-4*(approx.abs()+eb.abs()+1.0)

    counts=[]; match=0; coarse_hit=0
    for j in range(H.shape[0]):
        p=torch.topk(approx[j],k=PILOT).indices
        lower=exact[j,p].max()
        cand=torch.nonzero(ub[j]>=lower,as_tuple=False).flatten()
        union=torch.unique(torch.cat([p,cand]))
        counts.append(int(union.numel()))
        pred=cand[exact[j,cand].argmax()].item(); true=exact[j].argmax().item()
        match+=int(pred==true); coarse_hit+=int(true in p.tolist())
    c=torch.tensor(counts,dtype=torch.float64); f=c/V
    # Read msb for all weights; only candidate rows require remaining bits.
    bits=msb+(8-msb)*f
    return {
      'group':group,'msb_bits':msb,'unread_bits':unread,'max_unread_code_error':max_code_res,
      'int8_vs_fp32_argmax_agreement':int8_fp_agree,'certified_exact_int8_match_rate':match/H.shape[0],
      'coarse_top1_is_true_rate':coarse_hit/H.shape[0],
      'candidate_count_mean':float(c.mean()),'median':float(c.median()),'p90':float(torch.quantile(c,0.9)),'max':float(c.max()),
      'candidate_fraction_mean':float(f.mean()),'idealized_bits_read_per_weight':float(bits.mean()),
      'idealized_bandwidth_reduction_vs_dense_int8':float(8.0/bits.mean()),
      'idealized_bandwidth_reduction_vs_dense_fp16':float(16.0/bits.mean())
    }

def main():
    torch.set_num_threads(max(1,min(4,os.cpu_count() or 1)))
    tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    tok.padding_side='right'
    model=AutoModelForCausalLM.from_pretrained(MODEL_ID,torch_dtype=torch.float32,low_cpu_mem_usage=True); model.eval()
    W=model.lm_head.weight.detach().float().cpu();V,d=W.shape
    H=hiddens(model,tok,pick(load_dataset('ag_news')['test'],N))
    results=[]
    for g in GROUPS:
      for m in MSB_BITS:
        log(f'g={g} msb={m}')
        r=one(W,H,g,m);results.append(r);log(str(r))
    out={'kind':'centered_lossless_progressive_bitplane_exact_lmhead','model':MODEL_ID,'queries':N,'vocab':V,'hidden_dim':d,'pilot':PILOT,
         'reference':'groupwise symmetric INT8 deployment lm-head',
         'method':'MSB prefix identifies integer interval; midpoint yields symmetric certified residual; remaining bits are read only for candidates.',
         'results':results,'caveat':'Bit-read ratios are idealized. Actual packing, scale fetch, candidate compaction/gather, and kernel synchronization are not measured.'}
    os.makedirs('experiments/artifacts',exist_ok=True)
    with open('experiments/artifacts/exactsearch_centered_bitplane.json','w') as f:json.dump(out,f,indent=2)
    print('=== CENTERED_BITPLANE ===');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
