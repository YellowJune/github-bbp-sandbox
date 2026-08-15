import json, os, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL=os.environ.get('PB_MODEL','Qwen/Qwen2.5-0.5B-Instruct')
STEPS=int(os.environ.get('PB_STEPS','32'))
PROMPT='Explain in one paragraph why exact numerical decisions can sometimes be certified before all stored precision is read.'
assert torch.backends.mps.is_available()
device='mps'; torch.set_grad_enabled(False)

def sync(): torch.mps.synchronize()
def stats(x):
    a=np.asarray(x,float)
    return {'n':len(a),'median_ms':float(np.median(a)),'mean_ms':float(a.mean()),'p10_ms':float(np.percentile(a,10)),'p90_ms':float(np.percentile(a,90)),'min_ms':float(a.min()),'max_ms':float(a.max())}

tok=AutoTokenizer.from_pretrained(MODEL)
m=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float16,low_cpu_mem_usage=True).to(device).eval()
x=tok(PROMPT,return_tensors='pt'); ids=x['input_ids'].to(device); mask=x.get('attention_mask',torch.ones_like(ids)).to(device)
# Body only: no lm_head.
sync(); t0=time.perf_counter(); out=m.model(input_ids=ids,attention_mask=mask,use_cache=True,return_dict=True); sync(); prefill=(time.perf_counter()-t0)*1e3
past=out.past_key_values
# Use one fixed valid token while letting cache grow exactly as a decode stream would.
next_id=ids[:,-1:]
times=[]; lengths=[]
for step in range(STEPS):
    new_mask=torch.ones((1,ids.shape[1]+step+1),dtype=mask.dtype,device=device)
    sync(); t0=time.perf_counter(); out=m.model(input_ids=next_id,attention_mask=new_mask,past_key_values=past,use_cache=True,return_dict=True); sync(); dt=(time.perf_counter()-t0)*1e3
    past=out.past_key_values; times.append(dt); lengths.append(ids.shape[1]+step+1)
# Report all steps and a steady-state slice excluding the first four dispatch/warmup effects.
steady=times[min(4,len(times)):]
report={'kind':'proofbits_mps_body_only_kv_decode','model':MODEL,'device':'mps','dtype':'float16','prompt_tokens':int(ids.shape[1]),'steps':STEPS,'prefill_body_ms':prefill,'all_decode_steps':stats(times),'steady_decode_steps':stats(steady),'sequence_length_first':lengths[0],'sequence_length_last':lengths[-1],
'caveat':'Body-only Hugging Face/PyTorch MPS decode with KV cache; lm_head is intentionally omitted. This can be combined with independently measured decision-head latency only as a component-level projection, not as an integrated end-to-end ProofBits measurement.'}
Path('experiments/artifacts').mkdir(parents=True,exist_ok=True);slug=MODEL.split('/')[-1].replace('.','_');p=Path(f'experiments/artifacts/proofbits_mps_body_{slug}.json');p.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
