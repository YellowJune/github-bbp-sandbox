"""GPU benchmark harness for ProofBits FP16 byte-plane lm-head.

Run on a CUDA self-hosted runner. Reports:
- torch dense FP16-rounded head with FP32 matvec accumulation surrogate
- Triton dense byte-plane reconstruction kernel
- ProofBits high-byte coarse+certificate + low-byte survivor refinement
- exact argmax agreement
- candidate count

For publication-grade DRAM byte counters, wrap this script with Nsight Compute/CUPTI;
wall-clock alone does not prove the traffic model.
"""

import argparse, json, sys, time
from pathlib import Path
import torch
import triton
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kernels.proofbits_triton_fp16_bytes import (
    pack_fp16_byteplanes, build_highbyte_interval_lut, validate_finite_highbytes,
    proofbits_argmax, dense_exact,
)


def bench_ms(fn, warmup=20, rep=100):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
    vals=[]
    for _ in range(rep):
        start.record(); fn(); end.record(); torch.cuda.synchronize(); vals.append(start.elapsed_time(end))
    vals.sort()
    return {'median_ms':float(vals[len(vals)//2]),'p10_ms':float(vals[int(.1*len(vals))]),'p90_ms':float(vals[int(.9*len(vals))-1])}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',default='Qwen/Qwen2.5-0.5B-Instruct')
    ap.add_argument('--prompt',default='Solve carefully: 137 * 29 =')
    ap.add_argument('--pilot-k',type=int,default=1)
    ap.add_argument('--rep',type=int,default=100)
    ap.add_argument('--output',default='experiments/artifacts/proofbits_fp16_gpu_benchmark.json')
    args=ap.parse_args()
    assert torch.cuda.is_available(), 'CUDA GPU required'
    device='cuda'

    tok=AutoTokenizer.from_pretrained(args.model)
    model=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.float16,device_map=device)
    model.eval()
    x=tok(args.prompt,return_tensors='pt').to(device)
    with torch.no_grad():
        h=model.model(**x,use_cache=False,return_dict=True).last_hidden_state[0,-1].float().contiguous()
    # Publication reference: FP16-rounded stored weights, FP32 accumulation.
    w16=model.lm_head.weight.detach().half().cpu().contiguous()
    high,low=pack_fp16_byteplanes(w16); validate_finite_highbytes(high)
    high=high.to(device); low=low.to(device)
    mid_lut,rad_lut,_=build_highbyte_interval_lut(device=device)
    wref=w16.float().to(device)

    @torch.no_grad()
    def torch_dense(): return torch.mv(wref,h)
    @torch.no_grad()
    def triton_dense(): return dense_exact(high,low,h)
    @torch.no_grad()
    def pb(): return proofbits_argmax(high,low,mid_lut,rad_lut,h,pilot_k=args.pilot_k)

    ref=torch_dense(); ref_id=int(ref.argmax())
    td=triton_dense(); td_id=int(td.argmax())
    pb_id,candidates=pb(); pb_id=int(pb_id)
    assert td_id==ref_id, (td_id,ref_id)
    assert pb_id==ref_id, (pb_id,ref_id)

    dense_t=bench_ms(torch_dense,rep=args.rep)
    triton_t=bench_ms(triton_dense,rep=args.rep)
    pb_t=bench_ms(pb,rep=args.rep)
    V,D=w16.shape; f=candidates/V; ideal=2/(1+f)
    report={'model':args.model,'gpu':torch.cuda.get_device_name(0),'vocab':V,'hidden_dim':D,
            'pilot_k':args.pilot_k,'candidate_count':int(candidates),'candidate_fraction':float(f),
            'idealized_weight_byte_reduction':float(ideal),'exact_argmax':True,
            'torch_dense':dense_t,'triton_dense_byteplane':triton_t,'proofbits':pb_t,
            'speedup_vs_torch_dense_median':float(dense_t['median_ms']/pb_t['median_ms']),
            'speedup_vs_triton_dense_byteplane_median':float(triton_t['median_ms']/pb_t['median_ms']),
            'notes':['Torch reference uses FP16-rounded weights converted to FP32 for matvec; it is a correctness/reference baseline, not necessarily the fastest vendor FP16 GEMV.',
                     'Publication-grade comparison must additionally benchmark native torch/cuBLAS FP16 GEMV and collect DRAM counters with Nsight Compute/CUPTI.',
                     'Current ProofBits host orchestration uses torch.topk/nonzero and is intentionally unfused.']}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
