import json, os, time, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL='Qwen/Qwen2.5-0.5B-Instruct'
SEED=20260815
GROUP=16
MSBS=[4,5]
BATCH=8
MAX_LEN=96
N_STATIC=96
GEN_STEPS=8

GEN_PROMPTS=[
 'Write a Python function that returns the prime factors of an integer.',
 'Implement binary search in Python and explain its time complexity.',
 'Given a list of integers, find the longest increasing subsequence.',
 'Write SQL to find the second highest salary in each department.',
 'Explain why Dijkstra algorithm fails with negative edge weights.',
 'Implement merge sort without using built-in sorting functions.',
 'Write a regex that validates a simple email address.',
 'How can a hash table resolve collisions using open addressing?',
 'Solve 3x + 7 = 31 and show the algebraic steps.',
 'A train travels 180 km in 2.5 hours. What is its average speed?',
 'Find the derivative of x^3 sin(x).',
 'Prove that the sum of the first n odd integers equals n squared.',
 'If a fair die is rolled twice, what is the probability the sum is 8?',
 'Compute the eigenvalues of the matrix [[2,1],[1,2]].',
 'Explain the difference between a vector space and an affine space.',
 'Find all integer solutions to x^2 - y^2 = 15.'
]

def log(x): print(time.strftime('[%H:%M:%S]'),x,flush=True)

def select(ds,n,field,seed):
    g=torch.Generator().manual_seed(seed);out=[]
    for i in torch.randperm(len(ds),generator=g).tolist():
        t=(ds[i][field] or '').strip()
        if len(t)>=60: out.append(t)
        if len(out)==n:return out
    raise RuntimeError(f'only {len(out)} usable rows')

def collect_last(model,tok,texts):
    hs=[]
    for s in range(0,len(texts),BATCH):
        e=tok(texts[s:s+BATCH],return_tensors='pt',padding=True,truncation=True,max_length=MAX_LEN)
        with torch.inference_mode():
            y=model.model(input_ids=e['input_ids'],attention_mask=e['attention_mask'],use_cache=False,return_dict=True).last_hidden_state
        hs.append(y[:,-1].float().cpu())
    return torch.cat(hs)

def collect_generated(model,tok,W,prompts,steps):
    # left padding means every current sequence ends at the final column.
    e=tok(prompts,return_tensors='pt',padding=True,truncation=True,max_length=64)
    ids=e['input_ids']; mask=e['attention_mask']; hs=[]
    for st in range(steps):
        with torch.inference_mode():
            y=model.model(input_ids=ids,attention_mask=mask,use_cache=False,return_dict=True).last_hidden_state
            h=y[:,-1].float().cpu(); hs.append(h)
            nxt=(h@W.T).argmax(1).to(ids.device)
        ids=torch.cat([ids,nxt[:,None]],1)
        mask=torch.cat([mask,torch.ones((mask.shape[0],1),dtype=mask.dtype)],1)
        log(f'generated step {st+1}/{steps}')
    return torch.cat(hs,0)

def make_int8(W):
    V,d=W.shape;G=d//GROUP;X=W.view(V,G,GROUP)
    scale=X.abs().amax(2).clamp_min(1e-12)/127.0
    q=torch.round(X/scale.unsqueeze(2)).clamp(-127,127).to(torch.int16)
    W8=(q.float()*scale.unsqueeze(2)).reshape(V,d).contiguous()
    return q,scale,W8

def eval_one(H,exact,fp,q,scale,msb):
    V,d=fp.shape;G=d//GROUP;unread=8-msb;step=1<<unread
    u=q+128; pref=torch.div(u,step,rounding_mode='floor')
    mid=pref.float()*step+(step-1)/2.0-128.0
    Wc=(mid*scale.unsqueeze(2)).reshape(V,d)
    approx=H@Wc.T
    hnorm=torch.linalg.vector_norm(H.view(H.shape[0],G,GROUP),dim=2)
    coeff=(GROUP**0.5)*((step-1)/2.0)
    eb=coeff*(hnorm@scale.T)
    ub=approx+eb+3e-4*(approx.abs()+eb.abs()+1.0)
    counts=[];match=0;coarse=0
    for j in range(H.shape[0]):
        p=approx[j].argmax().view(1)
        lower=exact[j,p].max()
        cand=torch.nonzero(ub[j]>=lower,as_tuple=False).flatten()
        union=torch.unique(torch.cat([p,cand]));counts.append(union.numel())
        true=exact[j].argmax().item(); pred=cand[exact[j,cand].argmax()].item()
        match+=int(pred==true); coarse+=int(p.item()==true)
    c=torch.tensor(counts,dtype=torch.float64);f=c/V
    codebits=msb+(8-msb)*f
    total=codebits+16/GROUP
    dense8=8+16/GROUP
    return {'n':H.shape[0],'msb':msb,'certified_match_rate':match/H.shape[0],
            'int8_fp32_argmax_agreement':float((exact.argmax(1)==(H@fp.T).argmax(1)).float().mean()),
            'coarse_top1_true_rate':coarse/H.shape[0],
            'candidate_mean':float(c.mean()),'median':float(c.median()),'p90':float(torch.quantile(c,.9)),
            'p99':float(torch.quantile(c,.99)),'max':float(c.max()),'candidate_fraction_mean':float(f.mean()),
            'total_bits_per_weight_including_scales':float(total.mean()),
            'idealized_bw_reduction_vs_dense_int8':float(dense8/total.mean())}

def main():
    torch.manual_seed(SEED);torch.set_num_threads(max(1,min(4,os.cpu_count() or 1)))
    tok=AutoTokenizer.from_pretrained(MODEL,use_fast=True)
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    tok.padding_side='left'
    model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float32,low_cpu_mem_usage=True);model.eval()
    for p in model.parameters():p.requires_grad_(False)
    W=model.lm_head.weight.detach().float().cpu();V,d=W.shape
    ag=load_dataset('ag_news')['test']; agtxt=select(ag,N_STATIC,'text',SEED+1)
    wiki=load_dataset('wikitext','wikitext-2-raw-v1')['test']; wikitxt=select(wiki,N_STATIC,'text',SEED+2)
    log('collect AG'); Hag=collect_last(model,tok,agtxt)
    log('collect WikiText'); Hwiki=collect_last(model,tok,wikitxt)
    log('collect autoregressive math/code'); Hgen=collect_generated(model,tok,W,GEN_PROMPTS,GEN_STEPS)
    domains={'ag_news':Hag,'wikitext':Hwiki,'autoregressive_math_code':Hgen}
    q,scale,W8=make_int8(W)
    results={}
    for name,H in domains.items():
        log(f'exact scores {name} n={H.shape[0]}')
        exact=H@W8.T
        rr=[]
        for m in MSBS:
            log(f'{name} msb={m}')
            r=eval_one(H,exact,W,q,scale,m);rr.append(r);log(str(r))
        results[name]=rr
    out={'kind':'crossdomain_scaleonly_exactsearch','model':MODEL,'group':GROUP,'vocab':V,'hidden_dim':d,
         'domains':{'ag_news':N_STATIC,'wikitext':N_STATIC,'autoregressive_math_code':len(GEN_PROMPTS)*GEN_STEPS},
         'extra_certificate_metadata_bytes':0,'results':results,
         'caveat':'Idealized bandwidth accounting only; no custom packed-bit kernel or end-to-end wall-clock claim.'}
    os.makedirs('experiments/artifacts',exist_ok=True)
    with open('experiments/artifacts/exactsearch_crossdomain.json','w') as f:json.dump(out,f,indent=2)
    print('=== CROSSDOMAIN ===');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
