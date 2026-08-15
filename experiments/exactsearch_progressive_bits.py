import json, os, time, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL='Qwen/Qwen2.5-0.5B-Instruct';SEED=20260815;G=16;N=128;BATCH=8;MAX_LEN=80
PROMPTS=[
'Write a Python function for prime factorization.','Implement binary search in Python.','Explain Dijkstra with negative edges.',
'Write SQL for the second highest salary.','Implement merge sort recursively.','Explain hash table collision resolution.',
'Solve 3x + 7 = 31 step by step.','Find the derivative of x^3 sin(x).',
'Prove the sum of first n odd integers is n^2.','Compute eigenvalues of [[2,1],[1,2]].',
'What is the probability two dice sum to 8?','Explain vector space versus affine space.',
'Find integer solutions of x^2-y^2=15.','Implement longest increasing subsequence.','Write a simple email regex.','Explain quicksort average complexity.'
]
GEN_STEPS=8

def log(x):print(time.strftime('[%H:%M:%S]'),x,flush=True)
def pick(ds,n):
 g=torch.Generator().manual_seed(SEED+999);o=[]
 for i in torch.randperm(len(ds),generator=g).tolist():
  t=ds[i]['text'].strip()
  if len(t)>=50:o.append(t)
  if len(o)==n:return o
 raise RuntimeError

def static_h(model,tok,ts):
 o=[]
 for s in range(0,len(ts),BATCH):
  e=tok(ts[s:s+BATCH],return_tensors='pt',padding=True,truncation=True,max_length=MAX_LEN)
  with torch.inference_mode():y=model.model(input_ids=e['input_ids'],attention_mask=e['attention_mask'],use_cache=False,return_dict=True).last_hidden_state
  o.append(y[:,-1].float().cpu())
 return torch.cat(o)

def gen_h(model,tok,W):
 e=tok(PROMPTS,return_tensors='pt',padding=True,truncation=True,max_length=64);ids=e['input_ids'];mask=e['attention_mask'];o=[]
 for t in range(GEN_STEPS):
  with torch.inference_mode():
   y=model.model(input_ids=ids,attention_mask=mask,use_cache=False,return_dict=True).last_hidden_state
   h=y[:,-1].float().cpu();o.append(h);nxt=(h@W.T).argmax(1)
  ids=torch.cat([ids,nxt[:,None]],1);mask=torch.cat([mask,torch.ones((mask.shape[0],1),dtype=mask.dtype)],1)
 return torch.cat(o)

def quant(W):
 V,d=W.shape;ng=d//G;X=W.view(V,ng,G);sc=X.abs().amax(2).clamp_min(1e-12)/127.;q=torch.round(X/sc.unsqueeze(2)).clamp(-127,127).to(torch.int16)
 W8=(q.float()*sc.unsqueeze(2)).reshape(V,d).contiguous();return q,sc,W8

def progressive(W,H,q,sc,W8):
 Q=H.shape[0];V,d=W.shape;ng=d//G;exact=H@W8.T
 alive=torch.ones((Q,V),dtype=torch.bool); best_lower=torch.full((Q,),-float('inf'))
 stage=[]; code_bits=torch.ones((Q,),dtype=torch.float64) # first MSB read for every row
 hnorm=torch.linalg.vector_norm(H.view(Q,ng,G),dim=2)
 u=q+128
 for b in range(1,8):
  unread=8-b;step=1<<unread
  pref=torch.div(u,step,rounding_mode='floor')
  mid=pref.float()*step+(step-1)/2.-128.
  Wc=(mid*sc.unsqueeze(2)).reshape(V,d)
  approx=H@Wc.T
  coeff=(G**0.5)*((step-1)/2.)
  eb=coeff*(hnorm@sc.T)
  ub=approx+eb+3e-4*(approx.abs()+eb.abs()+1.)
  counts=[]
  for j in range(Q):
   ids=torch.nonzero(alive[j],as_tuple=False).flatten()
   p=ids[approx[j,ids].argmax()]
   best_lower[j]=max(best_lower[j],exact[j,p].item())
   keep=ub[j,ids]>=best_lower[j]
   newids=ids[keep]
   alive[j].fill_(False);alive[j,newids]=True
   counts.append(newids.numel())
  c=torch.tensor(counts,dtype=torch.float64);f=c/V
  stage.append({'prefix_bits':b,'alive_mean':float(c.mean()),'median':float(c.median()),'p90':float(torch.quantile(c,.9)),
                'p99':float(torch.quantile(c,.99)),'max':float(c.max()),'alive_fraction_mean':float(f.mean())})
  # Every row still unresolved after b bits needs bit b+1.
  code_bits+=f
  log(f'b={b} alive_mean={c.mean().item():.1f} frac={f.mean().item():.6f}')
  del Wc,approx,eb,ub
 # At 8 bits compare exact scores only for final survivors.
 match=0;final_counts=[]
 for j in range(Q):
  ids=torch.nonzero(alive[j],as_tuple=False).flatten();final_counts.append(ids.numel())
  pred=ids[exact[j,ids].argmax()].item();true=exact[j].argmax().item();match+=int(pred==true)
 scale_bits=16/G;total=code_bits+scale_bits;dense=8+scale_bits
 return {'queries':Q,'certified_exact_match_rate':match/Q,'stages':stage,
         'code_bits_per_weight_mean':float(code_bits.mean()),'code_bits_p90':float(torch.quantile(code_bits,.9)),
         'total_bits_per_weight_including_scales_mean':float(total.mean()),
         'idealized_bw_reduction_vs_dense_int8':float(dense/total.mean()),
         'final_survivor_mean':float(torch.tensor(final_counts,dtype=torch.float64).mean()),
         'int8_fp32_argmax_agreement':float((exact.argmax(1)==(H@W.T).argmax(1)).float().mean())}

def main():
 torch.set_num_threads(max(1,min(4,os.cpu_count() or 1)))
 tok=AutoTokenizer.from_pretrained(MODEL,use_fast=True)
 if tok.pad_token_id is None:tok.pad_token=tok.eos_token
 tok.padding_side='left';model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float32,low_cpu_mem_usage=True);model.eval()
 W=model.lm_head.weight.detach().float().cpu();V,d=W.shape
 Hag=static_h(model,tok,pick(load_dataset('ag_news')['test'],N));Hgen=gen_h(model,tok,W)
 q,sc,W8=quant(W);res={}
 for name,H in [('ag_news',Hag),('autoregressive_math_code',Hgen)]:
  log(name);res[name]=progressive(W,H,q,sc,W8);log(str(res[name]))
 out={'kind':'fully_progressive_bit_by_bit_exactsearch','model':MODEL,'group':G,'vocab':V,'hidden_dim':d,'extra_metadata_bytes':0,'results':res,
      'traffic_model':'1 initial code bit for all rows + one additional bit only for rows unresolved after each prefix stage; existing FP16 group scales included.',
      'caveat':'Idealized bit traffic, not wall-clock. Candidate compaction/bit extraction/bound arithmetic are not measured.'}
 os.makedirs('experiments/artifacts',exist_ok=True)
 with open('experiments/artifacts/exactsearch_progressive_bits.json','w') as f:json.dump(out,f,indent=2)
 print('=== PROGRESSIVE_BITS ===');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
