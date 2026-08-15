import json, os, time, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL='Qwen/Qwen2.5-0.5B-Instruct'
SEED=20260815
N=256
MAX_LEN=64
BATCH=8
GROUPS=[16,32,64]
MSBS=[4,5,6]
PILOT=1

def log(x): print(time.strftime('[%H:%M:%S]'),x,flush=True)

def sample_texts(split,n):
    g=torch.Generator().manual_seed(SEED+555)
    out=[]
    for i in torch.randperm(len(split),generator=g).tolist():
        t=split[i]['text'].strip()
        if len(t)>=40: out.append(t)
        if len(out)==n:return out
    raise RuntimeError

def collect_h(model,tok,texts):
    hs=[]
    with torch.inference_mode():
        for s in range(0,len(texts),BATCH):
            e=tok(texts[s:s+BATCH],return_tensors='pt',padding=True,truncation=True,max_length=MAX_LEN)
            y=model.model(input_ids=e['input_ids'],attention_mask=e['attention_mask'],use_cache=False,return_dict=True).last_hidden_state
            last=e['attention_mask'].sum(1)-1
            hs.append(y[torch.arange(y.shape[0]),last].float().cpu())
    return torch.cat(hs)

def quant_int8(W,gsize):
    V,d=W.shape; G=d//gsize
    X=W.view(V,G,gsize)
    scale=X.abs().amax(2).clamp_min(1e-12)/127.0
    q=torch.round(X/scale.unsqueeze(2)).clamp(-127,127).to(torch.int16)
    return q,scale

def run_cfg(W,H,gsize,msb):
    V,d=W.shape; G=d//gsize
    q,scale=quant_int8(W,gsize)
    qf=q.float()
    W8=(qf*scale.unsqueeze(2)).reshape(V,d)
    exact=H@W8.T
    fp=H@W.T
    int8_fp=float((exact.argmax(1)==fp.argmax(1)).float().mean())

    unread=8-msb; step=1<<unread
    u=q+128
    prefix=torch.div(u,step,rounding_mode='floor')
    midpoint=prefix.float()*step+(step-1)/2.0-128.0
    Wc=(midpoint*scale.unsqueeze(2)).reshape(V,d)
    approx=H@Wc.T

    # NO extra token×group residual metadata.
    # Prefix interval guarantees every coordinate's residual <= r_code*scale_ig.
    # Therefore ||e_ig||_2 <= sqrt(gsize)*r_code*scale_ig.
    r_code=(step-1)/2.0
    hnorm=torch.linalg.vector_norm(H.view(H.shape[0],G,gsize),dim=2) # Q,G
    coeff=(gsize**0.5)*r_code
    eb=coeff*(hnorm@scale.T)
    ub=approx+eb+3e-4*(approx.abs()+eb.abs()+1.0)

    counts=[]; matches=0; coarse_hits=0
    for j in range(H.shape[0]):
        p=torch.topk(approx[j],k=PILOT).indices
        lower=exact[j,p].max()
        cand=torch.nonzero(ub[j]>=lower,as_tuple=False).flatten()
        union=torch.unique(torch.cat([p,cand]))
        counts.append(int(union.numel()))
        pred=cand[exact[j,cand].argmax()].item(); true=exact[j].argmax().item()
        matches+=int(pred==true); coarse_hits+=int(true in p.tolist())
    c=torch.tensor(counts,dtype=torch.float64); f=c/V
    bits=msb+(8-msb)*f
    # Scale overhead: one FP16 scale per group is already required by baseline INT8.
    # Report data-bit traffic both excluding and including the same scale bytes.
    scale_bits_per_weight=16.0/gsize
    ours_total=bits+scale_bits_per_weight
    dense_int8_total=8.0+scale_bits_per_weight
    dense_fp16=16.0
    return {
      'group':gsize,'msb_bits':msb,'unread_bits':unread,
      'certified_exact_int8_match_rate':matches/H.shape[0],
      'int8_vs_fp32_argmax_agreement':int8_fp,
      'coarse_top1_true_rate':coarse_hits/H.shape[0],
      'candidate_count_mean':float(c.mean()),'median':float(c.median()),
      'p90':float(torch.quantile(c,0.9)),'p99':float(torch.quantile(c,0.99)),'max':float(c.max()),
      'candidate_fraction_mean':float(f.mean()),
      'weight_code_bits_read_mean':float(bits.mean()),
      'scale_bits_per_weight':scale_bits_per_weight,
      'total_bits_per_weight_including_fp16_scales':float(ours_total.mean()),
      'dense_int8_bits_per_weight_including_fp16_scales':dense_int8_total,
      'idealized_bw_reduction_vs_dense_int8_including_scales':float(dense_int8_total/ours_total.mean()),
      'idealized_bw_reduction_vs_dense_fp16':float(dense_fp16/ours_total.mean()),
      'certificate_metadata':'none beyond existing per-group quantization scales'
    }

def main():
    torch.set_num_threads(max(1,min(4,os.cpu_count() or 1)))
    tok=AutoTokenizer.from_pretrained(MODEL,use_fast=True)
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    tok.padding_side='right'
    model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float32,low_cpu_mem_usage=True);model.eval()
    W=model.lm_head.weight.detach().float().cpu();V,d=W.shape
    H=collect_h(model,tok,sample_texts(load_dataset('ag_news')['test'],N))
    results=[]
    for g in GROUPS:
      for m in MSBS:
        log(f'g={g} msb={m}')
        r=run_cfg(W,H,g,m);results.append(r);log(str(r))
    out={'kind':'scale_only_progressive_precision_certificate','model':MODEL,'queries':N,'vocab':V,'hidden_dim':d,'pilot':PILOT,
         'bound':'E_i <= sqrt(group)*((2^(8-msb)-1)/2)*sum_g ||h_g||_2 * scale_{i,g}',
         'extra_certificate_metadata_bytes':0,
         'results':results,
         'caveat':'Bandwidth accounting is idealized and includes existing FP16 scale metadata, but not candidate compaction, packed-bit extraction, synchronization, or arithmetic overhead.'}
    os.makedirs('experiments/artifacts',exist_ok=True)
    with open('experiments/artifacts/exactsearch_scaleonly_bound.json','w') as f:json.dump(out,f,indent=2)
    print('=== SCALEONLY_BOUND ===');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
