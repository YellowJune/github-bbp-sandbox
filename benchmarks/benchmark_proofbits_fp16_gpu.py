"""GPU benchmark harness for the FINAL LUT-free ProofBits FP16 upper-only path.

Run on a CUDA self-hosted runner. Reports native FP16, matched dense byte-plane,
raw ProofBits, conservative roundoff-safe ProofBits, exactness, S-union-P low
byte rows, idealized bytes, and measured latency.

The high-byte upper pass uses direct FP16 endpoint reconstruction: append low
suffix 0xFF iff hidden sign equals stored-weight sign, else append 0x00. No
endpoint LUT is read in the main pass.

Publication-grade GPU claims still require a coefficient derived for the actual
compiled reduction semantics and Nsight/CUPTI DRAM counters.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kernels.proofbits_triton_fp16_upperonly import (
    pack_fp16_byteplanes,
    validate_finite_highbytes,
    conservative_fp32_rounding_coeff,
    conservative_h_l1_upper,
    proofbits_argmax,
    dense_exact,
)


def bench_ms(fn, warmup=20, rep=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    vals = []
    for _ in range(rep):
        start.record(); fn(); end.record(); torch.cuda.synchronize()
        vals.append(start.elapsed_time(end))
    vals.sort()
    return {
        "median_ms": float(vals[len(vals)//2]),
        "p10_ms": float(vals[int(0.1*len(vals))]),
        "p90_ms": float(vals[int(0.9*len(vals))-1]),
    }


def traffic(V, D, low_rows):
    high_bytes = V * D
    low_bytes = int(low_rows) * D
    dense_bytes = 2 * V * D
    total = high_bytes + low_bytes
    return {
        "high_plane_bytes": int(high_bytes),
        "low_plane_bytes": int(low_bytes),
        "total_weight_bytes": int(total),
        "dense_fp16_weight_bytes": int(dense_bytes),
        "low_row_fraction": float(low_rows / V),
        "idealized_weight_byte_reduction": float(dense_bytes / total),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--prompt", default="Solve carefully: 137 * 29 =")
    ap.add_argument("--pilot-k", type=int, default=4)
    ap.add_argument("--fallback-fraction", type=float, default=1.0)
    ap.add_argument("--rounding-factor", type=int, default=4)
    ap.add_argument("--rep", type=int, default=100)
    ap.add_argument("--output", default="experiments/artifacts/proofbits_fp16_gpu_benchmark.json")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA GPU required"
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map=device
    )
    model.eval()
    x = tok(args.prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        h = model.model(**x, use_cache=False, return_dict=True).last_hidden_state[0,-1].float().contiguous()

    w16_cpu = model.lm_head.weight.detach().half().cpu().contiguous()
    high, low = pack_fp16_byteplanes(w16_cpu)
    validate_finite_highbytes(high)
    high = high.to(device); low = low.to(device)

    w16_gpu = w16_cpu.to(device)
    w32_gpu = w16_gpu.float()
    h16 = h.half()
    V, D = w16_cpu.shape
    conservative_coeff = conservative_fp32_rounding_coeff(D, factor=args.rounding_factor)
    h_l1_upper = conservative_h_l1_upper(h)

    @torch.no_grad()
    def torch_native_fp16(): return torch.mv(w16_gpu, h16)

    @torch.no_grad()
    def torch_fp32_reference_surrogate(): return torch.mv(w32_gpu, h)

    @torch.no_grad()
    def triton_dense(): return dense_exact(high, low, h)

    @torch.no_grad()
    def pb_raw():
        return proofbits_argmax(
            high, low, h,
            pilot_k=args.pilot_k,
            fallback_fraction=args.fallback_fraction,
            rounding_coeff=0.0,
            h_l1_upper=h_l1_upper,
        )

    @torch.no_grad()
    def pb_safe():
        return proofbits_argmax(
            high, low, h,
            pilot_k=args.pilot_k,
            fallback_fraction=args.fallback_fraction,
            rounding_coeff=conservative_coeff,
            h_l1_upper=h_l1_upper,
        )

    dense_ref = triton_dense(); ref_id = int(dense_ref.argmax())
    torch32_id = int(torch_fp32_reference_surrogate().argmax())
    torch16_id = int(torch_native_fp16().argmax())
    raw_id, raw_survivors, raw_low_rows = pb_raw()
    safe_id, safe_survivors, safe_low_rows = pb_safe()
    raw_id = int(raw_id); safe_id = int(safe_id)
    assert raw_id == ref_id, ("raw", raw_id, ref_id)
    assert safe_id == ref_id, ("safe", safe_id, ref_id)

    native_t = bench_ms(torch_native_fp16, rep=args.rep)
    fp32_t = bench_ms(torch_fp32_reference_surrogate, rep=args.rep)
    triton_t = bench_ms(triton_dense, rep=args.rep)
    raw_t = bench_ms(pb_raw, rep=args.rep)
    safe_t = bench_ms(pb_safe, rep=args.rep)

    report = {
        "kind": "proofbits_fp16_lutfree_upperonly_gpu_benchmark",
        "model": args.model,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "torch_version": torch.__version__,
        "vocab": V, "hidden_dim": D,
        "pilot_k": args.pilot_k,
        "fallback_fraction": args.fallback_fraction,
        "safe_rounding_factor": args.rounding_factor,
        "safe_rounding_coefficient": float(conservative_coeff),
        "safe_h_l1_upper": float(h_l1_upper),
        "reference_argmax": ref_id,
        "raw_exact_argmax": raw_id == ref_id,
        "safe_exact_argmax": safe_id == ref_id,
        "torch_fp32_surrogate_argmax_matches_reference": torch32_id == ref_id,
        "torch_native_fp16_argmax_matches_reference": torch16_id == ref_id,
        "raw": {
            "survivor_count": int(raw_survivors),
            "distinct_low_byte_rows_read": int(raw_low_rows),
            "traffic": traffic(V,D,raw_low_rows), "timing": raw_t,
        },
        "safe": {
            "survivor_count": int(safe_survivors),
            "distinct_low_byte_rows_read": int(safe_low_rows),
            "traffic": traffic(V,D,safe_low_rows), "timing": safe_t,
        },
        "baselines": {
            "torch_native_fp16": native_t,
            "torch_fp32_over_fp16_rounded_weights": fp32_t,
            "triton_dense_byteplane_fp32_accumulation": triton_t,
        },
        "speedups": {
            "raw_vs_native_fp16_median": float(native_t["median_ms"]/raw_t["median_ms"]),
            "safe_vs_native_fp16_median": float(native_t["median_ms"]/safe_t["median_ms"]),
            "raw_vs_triton_dense_median": float(triton_t["median_ms"]/raw_t["median_ms"]),
            "safe_vs_triton_dense_median": float(triton_t["median_ms"]/safe_t["median_ms"]),
        },
        "notes": [
            "The main upper pass is LUT-free: it reconstructs the extremal FP16 suffix directly from hidden/weight sign equality.",
            "The Triton dense byte-plane kernel is the matched correctness reference.",
            "Native PyTorch FP16 GEMV is the important performance baseline but may use different accumulation/output semantics; argmax equality is reported rather than assumed.",
            "Current ProofBits top-k/nonzero/unique orchestration and pilot reuse are not fused, so this is not the final optimized kernel.",
            "h_l1_upper is precomputed outside the timed call; production code must fuse/device-compute and account for it.",
            "Publication claims require Nsight/CUPTI DRAM counters and an implementation-specific rounding proof.",
        ],
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
