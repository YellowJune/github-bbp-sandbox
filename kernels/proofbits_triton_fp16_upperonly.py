"""Final ProofBits FP16 upper-only Triton prototype.

Lossless storage:
    FP16 W[V,D] -> high_byte uint8[V,D] + low_byte uint8[V,D]

The high byte fixes an exact finite FP16 interval.  The upper endpoint needed
for the score certificate can be reconstructed *without a LUT*.

For one coordinate, let s_w be the FP16 weight sign bit already present in the
high byte and s_h be the sign of h_j.  The extremal unread suffix is

    suffix = 0xFF  iff  s_w == s_h,
             0x00  otherwise.

Why: for a positive stored weight, increasing the unread mantissa increases the
value; for a negative stored weight it makes the value more negative.  The
score contribution h_j*w is maximized by the largest value when h_j>=0 and the
smallest value when h_j<0.  These cases reduce exactly to sign equality.

Thus the high-byte pass forms the exact interval endpoint bits directly,
bitcasts them to FP16, and performs one dot-product-like accumulation:

    U_i = sum_j max(h_j*w^-_ij, h_j*w^+_ij).

For implementation-level certification, the same pass also reduces

    M_i = max_j max(|w^-_ij|, |w^+_ij|).

No per-weight conversion is needed for M_i: among finite FP16 values whose low
byte is unread, maximum absolute value is obtained with suffix 0xFF, and
positive FP16 magnitude ordering follows the signless high-byte code.  The
kernel therefore max-reduces (high_byte & 0x7F), appends 0xFF once per row, and
bitcasts that scalar to obtain M_i.

A caller-supplied finite-precision coefficient c_round forms

    U_safe_i = Uhat_i + c_round * ||h||_1^upper * M_i.

The coefficient must be derived for the actual upper-bound reduction and the
matched dense reference's accumulation semantics.  The repository's
conservative CPU stress test used c_round = 2*gamma_{4d}; this file does not
pretend that constant is a tight theorem for an uninspected GPU reduction tree.

ProofBits exact-evaluates a few rows with largest U_safe, obtains lower bound B,
retains U_safe_i >= B, and fetches low bytes only for pilots/survivors.
Traffic accounting uses S union P.

Top-k/compaction/final reduction are still host-side in this correctness-first
prototype and must be fused before publication-grade wall-clock claims.
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
    """Reference-only endpoint LUT used by CPU self-tests, not by the GPU upper kernel."""
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
    return 2.0 * gamma(factor * d)


def conservative_h_l1_upper(h: torch.Tensor) -> float:
    assert h.ndim == 1
    shat = float(h.float().abs().sum().item())
    g = gamma(max(1, h.numel()))
    return shat / (1.0 - g)


@triton.jit
def _upper_safe_kernel(high_ptr, h_ptr,
                       upper_raw_ptr, row_absmax_ptr, upper_safe_ptr,
                       h_l1_upper, rounding_coeff,
                       D: tl.constexpr, BLOCK_D: tl.constexpr):
    row = tl.program_id(0)
    j = tl.arange(0, BLOCK_D)
    mask = j < D
    hb = tl.load(high_ptr + row * D + j, mask=mask, other=0).to(tl.uint16)
    h = tl.load(h_ptr + j, mask=mask, other=0.0).to(tl.float32)

    weight_sign = (hb >> 7) & 1
    hidden_sign = tl.where(h < 0.0, 1, 0).to(tl.uint16)
    same_sign = weight_sign == hidden_sign
    suffix = tl.where(same_sign, 255, 0).to(tl.uint16)
    endpoint_raw = (hb << 8) | suffix
    endpoint = tl.cast(endpoint_raw, tl.float16, bitcast=True).to(tl.float32)
    upper_raw = tl.sum(h * endpoint, axis=0)

    # max absolute possible endpoint in the row: strip sign, append suffix FF once.
    mag_hi = (hb & 0x7F).to(tl.int32)
    row_mag_hi = tl.max(tl.where(mask, mag_hi, 0), axis=0)
    row_absmax_raw = (row_mag_hi.to(tl.uint16) << 8) | 255
    row_absmax = tl.cast(row_absmax_raw, tl.float16, bitcast=True).to(tl.float32)

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


def upper_scores_safe(high, h, *, rounding_coeff=0.0, h_l1_upper=None):
    """Return (raw upper, row magnitude max, finite-precision-inflated upper)."""
    assert high.is_cuda and h.is_cuda
    V, D = high.shape
    block = triton.next_power_of_2(D)
    raw = torch.empty(V, device=h.device, dtype=torch.float32)
    rowmax = torch.empty_like(raw)
    safe = torch.empty_like(raw)
    if h_l1_upper is None:
        h_l1_upper = conservative_h_l1_upper(h)
    _upper_safe_kernel[(V,)](
        high, h, raw, rowmax, safe,
        float(h_l1_upper), float(rounding_coeff), D=D, BLOCK_D=block
    )
    return raw, rowmax, safe


def upper_scores(high, h):
    raw, _, _ = upper_scores_safe(high, h, rounding_coeff=0.0)
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


def proofbits_argmax(high, low, h, pilot_k=4, fallback_fraction=1.0,
                     rounding_coeff=0.0, h_l1_upper=None):
    _, _, U = upper_scores_safe(
        high, h, rounding_coeff=rounding_coeff, h_l1_upper=h_l1_upper
    )
    pilots = torch.topk(U, k=pilot_k).indices
    pilot_exact = refine_exact(high, low, h, pilots)
    B = pilot_exact.max()
    survivors = torch.nonzero(U >= B, as_tuple=False).squeeze(1)
    if survivors.numel() / high.shape[0] > fallback_fraction:
        survivors = torch.arange(high.shape[0], device=h.device, dtype=torch.int64)
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

    h = torch.randn(896)
    # Direct no-LUT endpoint reconstruction used by the Triton kernel.
    hb16 = high.to(torch.int16)
    wsign = (hb16 >> 7) & 1
    hsign = (h < 0).to(torch.int16)[None, :]
    suffix = torch.where(wsign == hsign, 255, 0).to(torch.int16)
    raw_endpoint = ((hb16 << 8) | suffix).contiguous()
    direct_endpoint = raw_endpoint.view(torch.float16).float()
    reference_endpoint = torch.where(h[None, :] >= 0, hi, lo)
    assert torch.equal(direct_endpoint, reference_endpoint)

    direct_upper = (h * direct_endpoint).sum(dim=1)
    midpoint = (lo + hi) * 0.5
    radius = (hi - lo) * 0.5
    midpoint_upper = (midpoint * h + radius * h.abs()).sum(dim=1)
    assert torch.allclose(direct_upper, midpoint_upper, rtol=2e-6, atol=2e-6)

    # Direct row magnitude code reduction must equal exact endpoint max magnitude.
    mag_hi = high.to(torch.int16) & 0x7F
    row_mag_hi = mag_hi.amax(dim=1)
    rowmax_raw = ((row_mag_hi << 8) | 0xFF).to(torch.int16).contiguous()
    direct_rowmax = rowmax_raw.view(torch.float16).float()
    reference_rowmax = torch.maximum(lo.abs(), hi.abs()).amax(dim=1)
    assert torch.equal(direct_rowmax, reference_rowmax)

    exact = (wf * h).sum(dim=1)
    assert bool((direct_upper + 2e-6 >= exact).all())
    coeff = conservative_fp32_rounding_coeff(h.numel(), factor=4)
    h1 = conservative_h_l1_upper(h)
    safe = direct_upper + coeff * h1 * direct_rowmax
    assert bool((safe >= direct_upper).all())
    assert bool((safe >= exact).all())
    return True


if __name__ == "__main__":
    print("upperonly_fp16_nolut_safe_selftest=", _cpu_selftest())
