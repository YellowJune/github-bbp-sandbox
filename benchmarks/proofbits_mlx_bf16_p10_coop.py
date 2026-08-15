import json
from pathlib import Path

import mlx.core as mx

import proofbits_mlx_bf16_p10_integrated as base

_original_make = base.make_kernels


def make_cooperative_kernels(D):
    dense, _old_upper, pilot, refine = _original_make(D)
    pwr = D * 10 // 32
    assert D % 32 == 0
    # Every 32 weights occupy exactly ten uint32 words in the 10-bit prefix
    # stream. Lanes 0..9 perform the only memory loads; simd_shuffle shares
    # those ten words with all 32 lanes. Thus the upper pass issues ~10 bits
    # of prefix payload per weight rather than one redundant uint32 load/lane.
    upper_src = f'''
        uint row = threadgroup_position_in_grid.x;
        uint lane = thread_index_in_simdgroup;
        ulong rowBase = (ulong)row * {pwr}ul;
        float acc = 0.0f;
        for (uint pair = 0u; pair < {D // 32}u; ++pair) {{
            ulong packBase = rowBase + (ulong)pair * 10ul;
            uint loaded = lane < 10u ? prefix[packBase + lane] : 0u;
            uint bit = lane * 10u;
            uint wi = bit >> 5;
            uint sh = bit & 31u;
            uint w0 = simd_shuffle(loaded, wi);
            uint w1 = simd_shuffle(loaded, min(wi + 1u, 9u));
            uint pre = w0 >> sh;
            if (sh > 22u) pre |= w1 << (32u - sh);
            pre &= 0x3FFu;
            uint rawBase = pre << 6u;
            uint weightSign = rawBase & 0x8000u;
            uint j = pair * 32u + lane;
            uint hiddenSign = hidden[j] < 0.0f ? 0x8000u : 0u;
            uint raw = rawBase | ((weightSign == hiddenSign) ? 0x3Fu : 0u);
            float endpoint = as_type<float>(raw << 16u);
            acc = fma(hidden[j], endpoint, acc);
        }}
        float total = simd_sum(acc);
        if (lane == 0) upper[row] = total;
    '''
    upper = mx.fast.metal_kernel(
        name='pb_bf16_p10_upper_cooperative',
        input_names=['prefix','hidden'],
        output_names=['upper'],
        source=upper_src,
    )
    return dense, upper, pilot, refine


base.make_kernels = make_cooperative_kernels

if __name__ == '__main__':
    base.main()
    p = Path('experiments/artifacts/proofbits_mlx_bf16_p10_integrated.json')
    if p.exists():
        d = json.loads(p.read_text())
        d['kind'] = 'proofbits_native_bf16_p10_simd_cooperative_integrated'
        d['upper_unpack'] = '10 uint32 loads per 32 weights, shared by simd_shuffle'
        p.write_text(json.dumps(d, indent=2))
