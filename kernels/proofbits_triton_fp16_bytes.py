"""ProofBits exact FP16 byte-plane Triton prototype.

Main storage transformation (lossless):
  FP16 [V,D] -> high_byte uint8 [V,D] + low_byte uint8 [V,D].

Stage 1 reads ONLY high_byte. A 256-entry lookup table maps each possible high byte
(sign + exponent + top-2 mantissa bits) to the midpoint and half-width of the exact
set of finite FP16 numbers obtained by varying the unread low byte.

For a hidden state h and vocabulary row i:
  c_i = sum_j h_j * midpoint[high_ij]
  e_i = sum_j |h_j| * radius[high_ij]
therefore exact FP16 score z_i lies in [c_i-e_i, c_i+e_i].

Stage 2 reads low_byte only for certified survivor rows and reconstructs the exact
FP16 weights by bitcasting the concatenated uint16 bit pattern.
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


def build_highbyte_interval_lut(device=None):
    """Returns midpoint/radius float32 LUTs of length 256.

    High-byte values whose exponent bits are all ones can encode Inf/NaN depending on
    low byte. They are marked non-finite. A normal finite LLM FP16 weight matrix must
    never contain those high-byte patterns, and packing validation rejects them.
    """
    hi = torch.arange(256, dtype=torch.int32)
    lo_bits = (hi << 8).to(torch.int16).contiguous()
    hi_bits = ((hi << 8) | 0xFF).to(torch.int16).contiguous()
    a = lo_bits.view(torch.float16).float()
    b = hi_bits.view(torch.float16).float()
    finite = torch.isfinite(a) & torch.isfinite(b)
    # Values for invalid patterns are irrelevant after finite-weight validation.
    mid = torch.where(finite, (a + b) * 0.5, torch.zeros_like(a))
    rad = torch.where(finite, (b - a).abs() * 0.5, torch.full_like(a, float('inf')))
    if device is not None:
        mid, rad = mid.to(device), rad.to(device)
    return mid.contiguous(), rad.contiguous(), finite


def validate_finite_highbytes(high: torch.Tensor):
    # exponent occupies high-byte bits [6:2]
    exp = (high.to(torch.int16) >> 2) & 0x1F
    if bool((exp == 0x1F).any()):
        raise ValueError('FP16 Inf/NaN exponent pattern found; ProofBits finite interval assumption violated')
    return True


@triton.jit
def _coarse_upper_kernel(high_ptr, mid_lut_ptr, rad_lut_ptr, h_ptr,
                         coarse_ptr, upper_ptr,
                         D: tl.constexpr, BLOCK_D: tl.constexpr):
    row = tl.program_id(0)
    j = tl.arange(0, BLOCK_D)
    mask = j < D
    hb = tl.load(high_ptr + row * D + j, mask=mask, other=0).to(tl.int32)
    # 256-entry LUT: ~2 KiB total for midpoint + radius, intended to remain cache-resident.
    mid = tl.load(mid_lut_ptr + hb, mask=mask, other=0.0).to(tl.float32)
    rad = tl.load(rad_lut_ptr + hb, mask=mask, other=0.0).to(tl.float32)
    h = tl.load(h_ptr + j, mask=mask, other=0.0).to(tl.float32)
    c = tl.sum(h * mid, axis=0)
    e = tl.sum(tl.abs(h) * rad, axis=0)
    tl.store(coarse_ptr + row, c)
    tl.store(upper_ptr + row, c + e)


@triton.jit
def _refine_exact_fp16_kernel(high_ptr, low_ptr, h_ptr, cand_ptr, exact_ptr,
                                D: tl.constexpr, BLOCK_D: tl.constexpr):
    ci = tl.program_id(0)
    row = tl.load(cand_ptr + ci).to(tl.int64)
    j = tl.arange(0, BLOCK_D)
    mask = j < D
    hi = tl.load(high_ptr + row * D + j, mask=mask, other=0).to(tl.uint16)
    lo = tl.load(low_ptr + row * D + j, mask=mask, other=0).to(tl.uint16)
    raw = (hi << 8) | lo
    # Exact bit reinterpretation, not numerical integer->float conversion.
    w16 = tl.cast(raw, tl.float16, bitcast=True)
    h = tl.load(h_ptr + j, mask=mask, other=0.0).to(tl.float32)
    acc = tl.sum(h * w16.to(tl.float32), axis=0)
    tl.store(exact_ptr + ci, acc)


@triton.jit
def _dense_exact_fp16_kernel(high_ptr, low_ptr, h_ptr, out_ptr,
                              D: tl.constexpr, BLOCK_D: tl.constexpr):
    row = tl.program_id(0)
    j = tl.arange(0, BLOCK_D)
    mask = j < D
    hi = tl.load(high_ptr + row * D + j, mask=mask, other=0).to(tl.uint16)
    lo = tl.load(low_ptr + row * D + j, mask=mask, other=0).to(tl.uint16)
    raw = (hi << 8) | lo
    w16 = tl.cast(raw, tl.float16, bitcast=True)
    h = tl.load(h_ptr + j, mask=mask, other=0.0).to(tl.float32)
    acc = tl.sum(h * w16.to(tl.float32), axis=0)
    tl.store(out_ptr + row, acc)


def coarse_upper(high, mid_lut, rad_lut, h):
    assert high.is_cuda and mid_lut.is_cuda and rad_lut.is_cuda and h.is_cuda
    V, D = high.shape
    block = triton.next_power_of_2(D)
    coarse = torch.empty(V, device=h.device, dtype=torch.float32)
    upper = torch.empty_like(coarse)
    _coarse_upper_kernel[(V,)](high, mid_lut, rad_lut, h, coarse, upper,
                               D=D, BLOCK_D=block)
    return coarse, upper


def refine_exact(high, low, h, candidates):
    candidates = candidates.to(device=h.device, dtype=torch.int64).contiguous()
    D = high.shape[1]
    block = triton.next_power_of_2(D)
    out = torch.empty(candidates.numel(), device=h.device, dtype=torch.float32)
    _refine_exact_fp16_kernel[(candidates.numel(),)](
        high, low, h, candidates, out, D=D, BLOCK_D=block)
    return out


def dense_exact(high, low, h):
    V, D = high.shape
    block = triton.next_power_of_2(D)
    out = torch.empty(V, device=h.device, dtype=torch.float32)
    _dense_exact_fp16_kernel[(V,)](high, low, h, out, D=D, BLOCK_D=block)
    return out


def proofbits_argmax(high, low, mid_lut, rad_lut, h, pilot_k=1):
    """Correctness-first orchestration. Candidate compaction/reduction is not fused yet."""
    coarse, upper = coarse_upper(high, mid_lut, rad_lut, h)
    pilots = torch.topk(coarse, k=pilot_k).indices
    pilot_exact = refine_exact(high, low, h, pilots)
    lower_bound = pilot_exact.max()
    candidates = torch.nonzero(upper >= lower_bound, as_tuple=False).squeeze(1)
    exact = refine_exact(high, low, h, candidates)
    winner = candidates[exact.argmax()]
    return winner, candidates.numel()


def _cpu_representation_selftest():
    w = (torch.randn(31, 896) * 0.03).half()
    assert torch.isfinite(w).all()
    high, low = pack_fp16_byteplanes(w)
    validate_finite_highbytes(high)
    rec = unpack_fp16_byteplanes(high, low)
    assert torch.equal(w.view(torch.int16), rec.view(torch.int16))
    mid, rad, finite = build_highbyte_interval_lut()
    # Every original value must lie in its high-byte interval.
    m = mid[high.long()]
    r = rad[high.long()]
    wf = w.float()
    assert bool(((wf >= m-r) & (wf <= m+r)).all())
    return True


if __name__ == '__main__':
    print('fp16_byteplane_selftest=', _cpu_representation_selftest())
