"""Final ProofBits FP16 upper-only Triton prototype.

Lossless storage:
    FP16 W[V,D] -> high_byte uint8[V,D] + low_byte uint8[V,D]

The high byte fixes an exact finite FP16 interval [lo, hi]. For a hidden
coordinate h_j, the maximum possible real-arithmetic contribution of the
unread low byte is

    max(h_j * lo, h_j * hi),

so a single accumulation gives the exact interval upper bound

    U_i = sum_j max(h_j*w^-_ij, h_j*w^+_ij).

For implementation-level certification, the same high-byte pass also reduces

    M_i = max_j max(|w^-_ij|, |w^+_ij|).

A caller-supplied finite-precision coefficient c_round then forms

    U_safe_i = Uhat_i + c_round * ||h||_1^upper * M_i.

The coefficient must be derived for the actual upper-bound reduction and the
dense reference's accumulation semantics. For example, the conservative CPU
stress test in this repository used c_round = 2*gamma_{4d}. The intended fused
one-dot GPU kernel should admit a tighter implementation-specific coefficient,
but this file deliberately does not hard-code an unverified theorem about a
vendor/compiler reduction order.

ProofBits evaluates a few rows with the largest safe upper bounds exactly,
takes the best exact pilot score B, discards every row with U_safe_i < B, and
fetches the low byte only for survivors. Pilot low-byte reads are included in
the reported low-byte row count via S union P.

This file is correctness-first. Top-k, survivor compaction and final reduction
are host-side PyTorch operations and must be fused/optimized before making
wall-clock claims.
"""

import math
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


def gamma(n: int, unit_roundoff: float = 2.0**-24) -> float:
    x = n * unit_roundoff
    if x >= 1.0:
        raise ValueError("gamma_n undefined because n*u >= 1")
    return x / (1.0 - x)


def conservative_fp32_rounding_coeff(d: int, factor: int = 4) -> float:
    """Conservative experimental coefficient used by the CPU safety kill-test.

    It is NOT asserted to be tight for Triton's reduction tree. Publication GPU
    code should derive a coefficient for the compiled reduction and matched
    dense reference, then pass that value explicitly.
    """
    return 2.0 * gamma(factor * d)


def conservative_h_l1_upper(h: torch.Tensor) -> float:
    """Upper-bound ||h||_1 from a FP32 sum using a sequential gamma envelope.

    This helper synchronizes if h is on CUDA and is intended for correctness
    validation, not the final fused fast path. A production kernel should
    compute a reduction-order-aware bound without a host sync.
    """
    assert h.ndim == 1
    shat = float(h.float().abs().sum().item())
    g = gamma(max(1, h.numel()))
    # If |fl(sum)-sum| <= gamma*sum, then sum <= fl(sum)/(1-gamma).
    return shat / (1.0 - g)


@triton.jit
def _upper_safe_kernel(high_ptr, lo_lut_ptr, hi_lut_ptr, h_ptr,
                       upper_raw_ptr, row_absmax_ptr, upper_safe_ptr,
                       h_l1_upper, rounding_coeff,
                       D: tl.constexpr, BLOCK_D: tl.constexpr):
    row = tl.program_id(0)
    j = tl.arange(0, BLOCK_D)
    mask = j < D
    hb = tl.load(high_ptr + row * D + j, mask=mask, other=0).to(tl.int32)
    lo = tl.load(lo_lut_ptr + hb, mask=mask, other=0.0).to(tl.float32)
    hi = tl.load(hi_lut_ptr + hb, mask=mask, other=0.0).to(tl.float32)
    h = tl.load(h_ptr + j, mask=mask, other=0.0).to(tl.float32)
    endpoint = tl.where(h >= 0.0, hi, lo)
    upper_raw = tl.sum(h * endpoint, axis=0)
    endpoint_absmax = tl.maximum(tl.abs(lo), tl.abs(hi))
    row_absmax = tl.max(tl.where(mask, endpoint_absmax, 0.0), axis=0)
    upper_safe = upper_raw + rounding_coeff * h_l1_upper * row_absmax
    tl.store(upper_raw_ptr + row, upper_raw)
    tl.store(row_absmax_ptr + row, row_absmax)
    tl.store(upper_safe_ptr + row, upper_safe)


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


def upper_scores_safe(high, lo_lut, hi_lut, h, *, rounding_coeff=0.0,
                      h_l1_upper=None):
    """Return (raw upper, row endpoint max, finite-precision-inflated upper)."""
    assert high.is_cuda and lo_lut.is_cuda and hi_lut.is_cuda and h.is_cuda
    V, D = high.shape
    block = triton.next_power_of_2(D)
    raw = torch.empty(V, device=h.device, dtype=torch.float32)
    rowmax = torch.empty_like(raw)
    safe = torch.empty_like(raw)
    if h_l1_upper is None:
        h_l1_upper = conservative_h_l1_upper(h)
    _upper_safe_kernel[(V,)](
        high, lo_lut, hi_lut, h, raw, rowmax, safe,
        float(h_l1_upper), float(rounding_coeff), D=D, BLOCK_D=block
    )
    return raw, rowmax, safe


def upper_scores(high, lo_lut, hi_lut, h):
    """Backward-compatible exact-real-arithmetic interval upper estimate."""
    raw, _, _ = upper_scores_safe(
        high, lo_lut, hi_lut, h, rounding_coeff=0.0
    )
    return raw


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


def proofbits_argmax(high, low, lo_lut, hi_lut, h, pilot_k=4,
                     fallback_fraction=1.0, rounding_coeff=0.0,
                     h_l1_upper=None):
    """Return exact-reference argmax plus survivor/read counts.

    Returns:
        winner, survivor_count, low_byte_rows_read

    `low_byte_rows_read` counts S union P, not just S, because pilot suffixes
    were already fetched before survivor refinement.
    """
    _, _, U = upper_scores_safe(
        high, lo_lut, hi_lut, h,
        rounding_coeff=rounding_coeff,
        h_l1_upper=h_l1_upper,
    )
    pilots = torch.topk(U, k=pilot_k).indices
    pilot_exact = refine_exact(high, low, h, pilots)
    B = pilot_exact.max()
    survivors = torch.nonzero(U >= B, as_tuple=False).squeeze(1)
    if survivors.numel() / high.shape[0] > fallback_fraction:
        survivors = torch.arange(high.shape[0], device=h.device, dtype=torch.int64)
    # Survivor exact evaluation may reread pilot suffixes in this correctness-first
    # implementation. Traffic accounting below counts each distinct row once;
    # a fused production implementation should cache/reuse pilot scores.
    exact = refine_exact(high, low, h, survivors)
    winner = survivors[exact.argmax()]
    low_rows = torch.unique(torch.cat([survivors, pilots])).numel()
    return winner, survivors.numel(), low_rows


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

    # Check upper-only identity against midpoint-radius up to FP32 reduction rounding.
    h = torch.randn(896)
    midpoint = (lo + hi) * 0.5
    radius = (hi - lo) * 0.5
    old_upper = (midpoint * h + radius * h.abs()).sum(dim=1)
    endpoint = torch.where(h[None, :] >= 0, hi, lo)
    new_upper = (endpoint * h).sum(dim=1)
    assert torch.allclose(old_upper, new_upper, rtol=2e-6, atol=2e-6)

    exact = (wf * h).sum(dim=1)
    assert bool((new_upper + 2e-6 >= exact).all())

    # Conservative finite-precision slack must be nonnegative and preserve the bound.
    rowmax = torch.maximum(lo.abs(), hi.abs()).amax(dim=1)
    coeff = conservative_fp32_rounding_coeff(h.numel(), factor=4)
    h1 = conservative_h_l1_upper(h)
    safe = new_upper + coeff * h1 * rowmax
    assert bool((safe >= new_upper).all())
    assert bool((safe >= exact).all())
    return True


if __name__ == "__main__":
    print("upperonly_fp16_safe_selftest=", _cpu_selftest())
