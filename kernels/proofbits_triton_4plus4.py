"""ProofBits 4+4 row-wise INT8 Triton prototype.

Storage per vocabulary row:
  base_hi:  two high nibbles packed per byte (D/2 bytes)
  resid_lo: two low nibbles packed per byte (D/2 bytes)
  scale:    one fp16/fp32 row scale

For offset code u=q+128, q in [-127,127]:
  q_mid4 = 16*(u>>4) + 7.5 - 128
  q_exact = 16*(u>>4) + (u&15) - 128
so the unread residual satisfies |q_exact-q_mid4| <= 7.5.
"""

import torch
import triton
import triton.language as tl


def pack_rowwise_int8_4plus4(q: torch.Tensor):
    assert q.dtype == torch.int8 and q.ndim == 2
    V, D = q.shape
    assert D % 2 == 0
    u = q.to(torch.int16) + 128
    hi = (u >> 4).to(torch.uint8)
    lo = (u & 15).to(torch.uint8)
    base = (hi[:, 0::2] | (hi[:, 1::2] << 4)).contiguous()
    resid = (lo[:, 0::2] | (lo[:, 1::2] << 4)).contiguous()
    return base, resid


def unpack_rowwise_int8_4plus4(base: torch.Tensor, resid: torch.Tensor):
    assert base.shape == resid.shape and base.ndim == 2
    V, B = base.shape
    hi0 = (base & 15).to(torch.int16)
    hi1 = (base >> 4).to(torch.int16)
    lo0 = (resid & 15).to(torch.int16)
    lo1 = (resid >> 4).to(torch.int16)
    q = torch.empty((V, B * 2), dtype=torch.int16, device=base.device)
    q[:, 0::2] = ((hi0 << 4) | lo0) - 128
    q[:, 1::2] = ((hi1 << 4) | lo1) - 128
    return q.to(torch.int8)


@triton.jit
def _coarse4_kernel(base_ptr, scale_ptr, h_ptr, coarse_ptr, upper_ptr,
                    h_l1, ROW_BYTES: tl.constexpr, BLOCK_BYTES: tl.constexpr):
    row = tl.program_id(0)
    b = tl.arange(0, BLOCK_BYTES)
    m = b < ROW_BYTES
    p = tl.load(base_ptr + row * ROW_BYTES + b, mask=m, other=0).to(tl.int32)
    hi0 = (p & 15).to(tl.float32)
    hi1 = ((p >> 4) & 15).to(tl.float32)
    d0 = 2 * b
    d1 = d0 + 1
    h0 = tl.load(h_ptr + d0, mask=m, other=0.0).to(tl.float32)
    h1 = tl.load(h_ptr + d1, mask=m, other=0.0).to(tl.float32)
    # midpoint: 16*hi + 7.5 - 128 = 16*hi - 120.5
    acc = tl.sum((16.0 * hi0 - 120.5) * h0 + (16.0 * hi1 - 120.5) * h1, axis=0)
    s = tl.load(scale_ptr + row).to(tl.float32)
    c = acc * s
    tl.store(coarse_ptr + row, c)
    tl.store(upper_ptr + row, c + 7.5 * s * h_l1)


@triton.jit
def _dense8_kernel(base_ptr, resid_ptr, scale_ptr, h_ptr, out_ptr,
                   ROW_BYTES: tl.constexpr, BLOCK_BYTES: tl.constexpr):
    row = tl.program_id(0)
    b = tl.arange(0, BLOCK_BYTES)
    m = b < ROW_BYTES
    ph = tl.load(base_ptr + row * ROW_BYTES + b, mask=m, other=0).to(tl.int32)
    pl = tl.load(resid_ptr + row * ROW_BYTES + b, mask=m, other=0).to(tl.int32)
    hi0, hi1 = ph & 15, (ph >> 4) & 15
    lo0, lo1 = pl & 15, (pl >> 4) & 15
    q0 = ((hi0 << 4) | lo0).to(tl.float32) - 128.0
    q1 = ((hi1 << 4) | lo1).to(tl.float32) - 128.0
    d0, d1 = 2 * b, 2 * b + 1
    h0 = tl.load(h_ptr + d0, mask=m, other=0.0).to(tl.float32)
    h1 = tl.load(h_ptr + d1, mask=m, other=0.0).to(tl.float32)
    acc = tl.sum(q0 * h0 + q1 * h1, axis=0)
    s = tl.load(scale_ptr + row).to(tl.float32)
    tl.store(out_ptr + row, acc * s)


@triton.jit
def _refine4_kernel(resid_ptr, scale_ptr, h_ptr, coarse_ptr, cand_ptr, exact_ptr,
                    ROW_BYTES: tl.constexpr, BLOCK_BYTES: tl.constexpr):
    ci = tl.program_id(0)
    row = tl.load(cand_ptr + ci).to(tl.int64)
    b = tl.arange(0, BLOCK_BYTES)
    m = b < ROW_BYTES
    p = tl.load(resid_ptr + row * ROW_BYTES + b, mask=m, other=0).to(tl.int32)
    lo0 = (p & 15).to(tl.float32)
    lo1 = ((p >> 4) & 15).to(tl.float32)
    d0, d1 = 2 * b, 2 * b + 1
    h0 = tl.load(h_ptr + d0, mask=m, other=0.0).to(tl.float32)
    h1 = tl.load(h_ptr + d1, mask=m, other=0.0).to(tl.float32)
    corr = tl.sum((lo0 - 7.5) * h0 + (lo1 - 7.5) * h1, axis=0)
    s = tl.load(scale_ptr + row).to(tl.float32)
    c = tl.load(coarse_ptr + row).to(tl.float32)
    tl.store(exact_ptr + ci, c + s * corr)


def coarse4(base, scale, h):
    assert base.is_cuda and scale.is_cuda and h.is_cuda
    V, row_bytes = base.shape
    block = triton.next_power_of_2(row_bytes)
    coarse = torch.empty(V, device=h.device, dtype=torch.float32)
    upper = torch.empty_like(coarse)
    h_l1 = float(h.float().abs().sum().item())
    _coarse4_kernel[(V,)](base, scale, h, coarse, upper, h_l1,
                          ROW_BYTES=row_bytes, BLOCK_BYTES=block)
    return coarse, upper


def dense8(base, resid, scale, h):
    V, row_bytes = base.shape
    block = triton.next_power_of_2(row_bytes)
    out = torch.empty(V, device=h.device, dtype=torch.float32)
    _dense8_kernel[(V,)](base, resid, scale, h, out,
                         ROW_BYTES=row_bytes, BLOCK_BYTES=block)
    return out


def refine4(resid, scale, h, coarse, candidates):
    candidates = candidates.to(device=h.device, dtype=torch.int64).contiguous()
    row_bytes = resid.shape[1]
    block = triton.next_power_of_2(row_bytes)
    out = torch.empty(candidates.numel(), device=h.device, dtype=torch.float32)
    _refine4_kernel[(candidates.numel(),)](resid, scale, h, coarse, candidates, out,
                                           ROW_BYTES=row_bytes, BLOCK_BYTES=block)
    return out


def proofbits_argmax(base, resid, scale, h, pilot_k=1):
    """Correctness-first host orchestration; compaction/reduction is not fused yet."""
    coarse, upper = coarse4(base, scale, h)
    pilot = torch.topk(coarse, k=pilot_k).indices
    pilot_exact = refine4(resid, scale, h, coarse, pilot)
    B = pilot_exact.max()
    candidates = torch.nonzero(upper >= B, as_tuple=False).squeeze(1)
    exact = refine4(resid, scale, h, coarse, candidates)
    return candidates[exact.argmax()], candidates.numel()


def _cpu_pack_selftest():
    q = torch.randint(-127, 128, (17, 896), dtype=torch.int16).to(torch.int8)
    base, resid = pack_rowwise_int8_4plus4(q)
    rec = unpack_rowwise_int8_4plus4(base, resid)
    assert torch.equal(q, rec)
    return True


if __name__ == '__main__':
    print('pack_selftest=', _cpu_pack_selftest())
