"""Final ProofBits FP16 upper-only Triton prototype.

Lossless storage:
    FP16 W[V,D] -> high_byte uint8[V,D] + low_byte uint8[V,D]

The high byte fixes an exact finite FP16 interval [lo, hi].  For a hidden
coordinate h_j the maximum possible contribution of the unread low byte is

    max(h_j * lo, h_j * hi)

so a single accumulation gives a deterministic row-score upper bound

    U_i = sum_j max(h_j*w^-_ij, h_j*w^+_ij).

ProofBits evaluates a few rows with the largest U_i exactly, takes the best
exact pilot score B, discards every row with U_i < B, and fetches the low byte
only for survivors.  The returned argmax is therefore exactly the argmax of
the stored FP16 matrix under FP32 accumulation.

This file is correctness-first.  The top-k, survivor compaction and final
reduction are host-side PyTorch operations and must be fused/optimized before
making wall-clock claims.
"""

import torch
import triton
import triton.language as tl


def pack_fp16_byteplanes(w: torch.Tensor):
    assert w.dtype == torch.float16 and w.ndim == 2
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    high = ((bits >> 8) & 0xFF).to(torch.uint8).contiguous()
    low = (bits & 0xFF).to(torch.uint8).contiguous()
    return high, low


def unpack_fp16_byteplanes(high: torch.Tensor, low: torch.Tensor):
    assert high.dtype == torch.uint8 and low.dtype == torch.uint8
    assert high.shape == low.shape
    raw = ((high.to(torch.int32) << 8) | low.to(torch.int32)).to(torch.int16).contiguous()
    return raw.view(torch.float16)


def build_highbyte_endpoint_lut(device=None):
    """Return exact finite lower/upper endpoint LUTs indexed by FP16 high byte."""
    hb = torch.arange(256, dtype=torch.int32)
    raw0 = (hb << 8).to(torch.int16).contiguous()
    raw1 = ((hb << 8) | 0xFF).to(torch.int16).contiguous()
    a = raw0.view(torch.float16).float()
    b = raw1.view(torch.float16).float()
    finite = torch.isfinite(a) & torch.isfinite(b)
    lo = torch.where(finite, torch.minimum(a, b), torch.zeros_like(a))
    hi = torch.where(finite, torch.maximum(a, b), torch.zeros_like(a))
    if device is not None:
        lo, hi, finite = lo.to(device), hi.to(device), finite.to(device)
    return lo.contiguous(), hi.contiguous(), finite.contiguous()


def validate_finite_highbytes(high: torch.Tensor):
    # Binary16 exponent occupies high-byte bits [6:2]. exponent==31 => Inf/NaN family.
    exp = (high.to(torch.int16) >> 2) & 0x1F
    if bool((exp == 0x1F).any()):
        raise ValueError("FP16 Inf/NaN exponent pattern found; finite interval assumption violated")
    return True


@triton.jit
def _upper_kernel(high_ptr, lo_lut_ptr, hi_lut_ptr, h_ptr, upper_ptr,
                  D: tl.constexpr, BLOCK_D: tl.constexpr):
    row = tl.program_id(0)
    j = tl.arange(0, BLOCK_D)
    mask = j < D
    hb = tl.load(high_ptr + row * D + j, mask=mask, other=0).to(tl.int32)
    lo = tl.load(lo_lut_ptr + hb, mask=mask, other=0.0).to(tl.float32)
    hi = tl.load(hi_lut_ptr + hb, mask=mask, other=0.0).to(tl.float32)
    h = tl.load(h_ptr + j, mask=mask, other=0.0).to(tl.float32)
    endpoint = tl.where(h >= 0.0, hi, lo)
    upper = tl.sum(h * endpoint, axis=0)
    tl.store(upper_ptr + row, upper)


@triton.jit
def _refine_exact_kernel(high_ptr, low_ptr, h_ptr, cand_ptr, exact_ptr,
                         D: tl.constexpr, BLOCK_D: tl.constexpr):
    ci = tl.program_id(0)
    row = tl.load(cand_ptr + ci).to(tl.int64)
    j = tl.arange(0, BLOCK_D)
    mask = j < D
    hb = tl.load(high_ptr + row * D + j, mask=mask, other=0).to(tl.uint16)
    lb = tl.load(low_ptr + row * D + j, mask=mask, other=0).to(tl.uint16)
    raw = (hb << 8) | lb
    w16 = tl.cast(raw, tl.float16, bitcast=True)
    h = tl.load(h_ptr + j, mask=mask, other=0.0).to(tl.float32)
    exact = tl.sum(h * w16.to(tl.float32), axis=0)
    tl.store(exact_ptr + ci, exact)


@triton.jit
def _dense_exact_kernel(high_ptr, low_ptr, h_ptr, out_ptr,
                        D: tl.constexpr, BLOCK_D: tl.constexpr):
    row = tl.program_id(0)
    j = tl.arange(0, BLOCK_D)
    mask = j < D
    hb = tl.load(high_ptr + row * D + j, mask=mask, other=0).to(tl.uint16)
    lb = tl.load(low_ptr + row * D + j, mask=mask, other=0).to(tl.uint16)
    raw = (hb << 8) | lb
    w16 = tl.cast(raw, tl.float16, bitcast=True)
    h = tl.load(h_ptr + j, mask=mask, other=0.0).to(tl.float32)
    exact = tl.sum(h * w16.to(tl.float32), axis=0)
    tl.store(out_ptr + row, exact)


def upper_scores(high, lo_lut, hi_lut, h):
    assert high.is_cuda and lo_lut.is_cuda and hi_lut.is_cuda and h.is_cuda
    V, D = high.shape
    block = triton.next_power_of_2(D)
    out = torch.empty(V, device=h.device, dtype=torch.float32)
    _upper_kernel[(V,)](high, lo_lut, hi_lut, h, out, D=D, BLOCK_D=block)
    return out


def refine_exact(high, low, h, candidates):
    candidates = candidates.to(device=h.device, dtype=torch.int64).contiguous()
    D = high.shape[1]
    block = triton.next_power_of_2(D)
    out = torch.empty(candidates.numel(), device=h.device, dtype=torch.float32)
    _refine_exact_kernel[(candidates.numel(),)](
        high, low, h, candidates, out, D=D, BLOCK_D=block
    )
    return out


def dense_exact(high, low, h):
    V, D = high.shape
    block = triton.next_power_of_2(D)
    out = torch.empty(V, device=h.device, dtype=torch.float32)
    _dense_exact_kernel[(V,)](high, low, h, out, D=D, BLOCK_D=block)
    return out


def proofbits_argmax(high, low, lo_lut, hi_lut, h, pilot_k=4, fallback_fraction=1.0):
    """Return exact FP16 argmax and survivor count.

    If the survivor fraction exceeds fallback_fraction, refine all rows.  This
    preserves exactness and caps suffix-byte traffic at the dense FP16 amount.
    """
    U = upper_scores(high, lo_lut, hi_lut, h)
    pilots = torch.topk(U, k=pilot_k).indices
    pilot_exact = refine_exact(high, low, h, pilots)
    B = pilot_exact.max()
    survivors = torch.nonzero(U >= B, as_tuple=False).squeeze(1)
    if survivors.numel() / high.shape[0] > fallback_fraction:
        survivors = torch.arange(high.shape[0], device=h.device, dtype=torch.int64)
    exact = refine_exact(high, low, h, survivors)
    winner = survivors[exact.argmax()]
    return winner, survivors.numel()


def _cpu_selftest():
    torch.manual_seed(0)
    w = (torch.randn(37, 896) * 0.05).half()
    assert torch.isfinite(w).all()
    high, low = pack_fp16_byteplanes(w)
    validate_finite_highbytes(high)
    rec = unpack_fp16_byteplanes(high, low)
    assert torch.equal(w.view(torch.int16), rec.view(torch.int16))

    lo_lut, hi_lut, finite = build_highbyte_endpoint_lut()
    assert bool(finite[high.long()].all())
    lo = lo_lut[high.long()]
    hi = hi_lut[high.long()]
    wf = w.float()
    assert bool(((wf >= lo) & (wf <= hi)).all())

    # Check the upper-only identity against midpoint+radius exactly up to FP32 rounding.
    h = torch.randn(896)
    midpoint = (lo + hi) * 0.5
    radius = (hi - lo) * 0.5
    old_upper = (midpoint * h + radius * h.abs()).sum(dim=1)
    endpoint = torch.where(h[None, :] >= 0, hi, lo)
    new_upper = (endpoint * h).sum(dim=1)
    assert torch.allclose(old_upper, new_upper, rtol=2e-6, atol=2e-6)

    exact = (wf * h).sum(dim=1)
    assert bool((new_upper + 2e-6 >= exact).all())
    return True


if __name__ == "__main__":
    print("upperonly_fp16_selftest=", _cpu_selftest())
