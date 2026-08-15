import json, time
from pathlib import Path
import numpy as np
import torch

DATA=Path('benchmarks/tmp_proofbits_cpu')
REPS=20; NQ=3
meta=json.loads((DATA/'meta.json').read_text());V,D,N=meta['V'],meta['D'],meta['N']
assert torch.backends.mps.is_available()
raw=np.fromfile(DATA/'full_u16.bin',dtype=np.uint16).reshape(V,D)
W_np=raw.view(np.float16).copy()
H_np=np.fromfile(DATA/'hidden_f32.bin',dtype=np.float32).reshape(N,D)
W=torch.from_numpy(W_np).to('mps')

def sync(): torch.mps.synchronize()
def run(h):
    z=torch.mv(W,h)
    k=torch.argmax(z)
    return k

def stats(xs):
    a=np.asarray(xs,float)
    return {'n':len(a),'median_ms':float(np.median(a)),'mean_ms':float(a.mean()),'p10_ms':float(np.percentile(a,10)),'p90_ms':float(np.percentile(a,90))}

rows=[];all_t=[]
for q in range(min(NQ,N)):
    h=torch.from_numpy(H_np[q]).to('mps',dtype=torch.float16)
    for _ in range(8): _=run(h)
    sync()
    times=[];winner=None
    for _ in range(REPS):
        sync();t0=time.perf_counter();k=run(h);sync();t1=time.perf_counter();times.append((t1-t0)*1e3);winner=int(k.cpu())
    all_t+=times;rows.append({'query':q,'winner':winner,'timing':stats(times)})
report={'kind':'proofbits_native_mps_dense_baseline','device':'mps','model':meta['model'],'V':V,'D':D,'dtype':'float16 weights + float16 hidden','operation':'torch.mv(W,h) + torch.argmax(logits), synchronized','queries':rows,'aggregate':stats(all_t),'caveat':'Native PyTorch MPS dense decision baseline. Numeric semantics can differ from custom Metal FP16-bit storage with FP32 accumulation; winner IDs are reported for cross-check. Framework dispatch/argmax are included.'}
Path('experiments/artifacts').mkdir(parents=True,exist_ok=True);Path('experiments/artifacts/proofbits_mps_native_baseline.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
