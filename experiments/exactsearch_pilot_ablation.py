import json, os, time, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID='Qwen/Qwen2.5-0.5B-Instruct'
SEED=20260815
N=96
MAX_LEN=64
BATCH=8
PILOTS=[1,2,4,8,16,32]
CFGS=[(4,32),(8,64)]

def log(x): print(time.strftime('[%H:%M:%S]'),x,flush=True)

def texts(split,n):
    g=torch.Generator().manual_seed(SEED+77)
    out=[]
    for i in torch.randperm(len(split),generator=g).tolist():
        t=split[i]['text'].strip()
        if len(t)>=40: out.append(t)
        if len(out)==n:return out
    raise RuntimeError

def hidden(model,tok,ts):
    hs=[]
    with torch.inference_mode():
        for s in range(0,len(ts),BATCH):
            enc=tok(ts[s:s+BATCH],return_tensors='pt',padding=True,truncation=True,max_length=MAX_LEN)
            y=model.model(input_ids=enc['input_ids'],attention_mask=enc['attention_mask'],use_cache=False,return_dict=True).last_hidden_state
            last=enc['attention_mask'].sum(1)-1
            hs.append(y[torch.arange(y.shape[0]),last].float().cpu())
    return torch.cat(hs)

def qgroup(W,bits,group):
    V,d=W.shape;G=d//group;qmax=(1<<(bits-1))-1
    X=W.view(V,G,group)
    sc=X.abs().amax(2,keepdim=True).clamp_min(1e-12)/qmax
    q=torch.round(X/sc).clamp(-qmax,qmax)
    Xh=q*sc
    en=torch.linalg.vector_norm(X-Xh,dim=2)
    return Xh.reshape(V,d).contiguous(),en

def main():
    torch.set_num_threads(max(1,min(4,os.cpu_count() or 1)))
    tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    tok.padding_side='right'
    model=AutoModelForCausalLM.from_pretrained(MODEL_ID,torch_dtype=torch.float32,low_cpu_mem_usage=True)
    model.eval()
    W=model.lm_head.weight.detach().float().cpu();V,d=W.shape
    ds=load_dataset('ag_news');H=hidden(model,tok,texts(ds['test'],N));dense=H@W.T
    results=[]
    for bits,group in CFGS:
        log(f'quant {bits}b g{group}')
        What,en=qgroup(W,bits,group);G=d//group
        approx=H@What.T
        hn=torch.linalg.vector_norm(H.view(N,G,group),dim=2)
        eb=hn@en.T
        ub=approx+eb+3e-4*(approx.abs()+eb.abs()+1)
        for pilotn in PILOTS:
            counts=[];match=0;rank_hits=0
            for j in range(N):
                p=torch.topk(approx[j],k=pilotn).indices
                lower=dense[j,p].max()
                cand=torch.nonzero(ub[j]>=lower,as_tuple=False).flatten()
                union=torch.unique(torch.cat([p,cand]))
                counts.append(union.numel())
                pred=cand[dense[j,cand].argmax()].item();true=dense[j].argmax().item()
                match+=int(pred==true);rank_hits+=int(true in p.tolist())
            c=torch.tensor(counts,dtype=torch.float64);f=c/V
            r={'bits':bits,'group':group,'pilot':pilotn,'exact_match_rate':match/N,
               'approx_topk_contains_true_rate':rank_hits/N,
               'exact_count_mean':float(c.mean()),'exact_count_median':float(c.median()),
               'exact_count_p90':float(torch.quantile(c,0.9)),'exact_count_max':float(c.max()),
               'exact_fraction_mean':float(f.mean()),
               'idealized_fp16_bandwidth_ratio':float(bits/16+f.mean()),
               'idealized_head_bandwidth_reduction':float(1/(bits/16+f.mean()))}
            results.append(r);log(str(r))
        del What,en,approx,eb,ub
    out={'kind':'exactsearch_pilot_ablation','model':MODEL_ID,'queries':N,'vocab':V,'hidden_dim':d,'results':results}
    os.makedirs('experiments/artifacts',exist_ok=True)
    with open('experiments/artifacts/exactsearch_pilot_ablation.json','w') as f: json.dump(out,f,indent=2)
    print('=== PILOT_ABLATION ===');print(json.dumps(out,indent=2))
if __name__=='__main__': main()
