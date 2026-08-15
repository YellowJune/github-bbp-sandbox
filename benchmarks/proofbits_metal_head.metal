#include <metal_stdlib>
using namespace metal;

constant uint SG = 32;
constant uint ARG_CHUNK = 4096;
constant float NEG_INF_F = -3.402823466e+38f;

kernel void dense_fp16_row(
    device const ushort* full_bits [[buffer(0)]],
    device const float* h [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& D [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    float acc = 0.0f;
    const ulong base = (ulong)row * (ulong)D;
    for (uint j = lane; j < D; j += SG) {
        half w = as_type<half>(full_bits[base + j]);
        acc = fma(h[j], (float)w, acc);
    }
    float total = simd_sum(acc);
    if (lane == 0) out[row] = total;
}

kernel void proofbits_high_upper_row(
    device const uchar* high [[buffer(0)]],
    device const float* h [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& D [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    float acc = 0.0f;
    const ulong base = (ulong)row * (ulong)D;
    for (uint j = lane; j < D; j += SG) {
        uchar hb = high[base + j];
        ushort weightSign = (ushort)(hb & (uchar)0x80);
        ushort hiddenSign = (h[j] < 0.0f) ? (ushort)0x80 : (ushort)0x00;
        ushort suffix = (weightSign == hiddenSign) ? (ushort)0x00FF : (ushort)0x0000;
        ushort raw = ((ushort)hb << 8) | suffix;
        half endpoint = as_type<half>(raw);
        acc = fma(h[j], (float)endpoint, acc);
    }
    float total = simd_sum(acc);
    if (lane == 0) out[row] = total;
}

kernel void argmax_stage1(
    device const float* values [[buffer(0)]],
    device float* block_values [[buffer(1)]],
    device uint* block_indices [[buffer(2)]],
    constant uint& N [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint begin = group * ARG_CHUNK;
    uint end = min(begin + ARG_CHUNK, N);
    float best = NEG_INF_F;
    uint bestIdx = 0xffffffffu;
    for (uint j = begin + lane; j < end; j += SG) {
        float v = values[j];
        if (v > best || (v == best && j < bestIdx)) { best = v; bestIdx = j; }
    }
    for (ushort off = 16; off > 0; off >>= 1) {
        float ov = simd_shuffle_down(best, off);
        uint oi = simd_shuffle_down(bestIdx, off);
        if (lane < off && (ov > best || (ov == best && oi < bestIdx))) { best = ov; bestIdx = oi; }
    }
    if (lane == 0) { block_values[group] = best; block_indices[group] = bestIdx; }
}

kernel void argmax_final(
    device const float* block_values [[buffer(0)]],
    device const uint* block_indices [[buffer(1)]],
    device uint* out_index [[buffer(2)]],
    constant uint& Nblocks [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]])
{
    float best = NEG_INF_F;
    uint bestIdx = 0xffffffffu;
    for (uint j = lane; j < Nblocks; j += SG) {
        float v = block_values[j];
        uint idx = block_indices[j];
        if (v > best || (v == best && idx < bestIdx)) { best = v; bestIdx = idx; }
    }
    for (ushort off = 16; off > 0; off >>= 1) {
        float ov = simd_shuffle_down(best, off);
        uint oi = simd_shuffle_down(bestIdx, off);
        if (lane < off && (ov > best || (ov == best && oi < bestIdx))) { best = ov; bestIdx = oi; }
    }
    if (lane == 0) out_index[0] = bestIdx;
}

kernel void exact_pilot_row(
    device const uchar* high [[buffer(0)]],
    device const uchar* low [[buffer(1)]],
    device const float* h [[buffer(2)]],
    device const uint* pilot_index [[buffer(3)]],
    device float* out_B [[buffer(4)]],
    constant uint& D [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint row = pilot_index[0];
    ulong base = (ulong)row * (ulong)D;
    float acc = 0.0f;
    for (uint j = lane; j < D; j += SG) {
        ushort raw = ((ushort)high[base + j] << 8) | (ushort)low[base + j];
        acc = fma(h[j], (float)as_type<half>(raw), acc);
    }
    float total = simd_sum(acc);
    if (lane == 0) out_B[0] = total;
}

kernel void conditional_refine_row(
    device const uchar* high [[buffer(0)]],
    device const uchar* low [[buffer(1)]],
    device const float* h [[buffer(2)]],
    device const float* U [[buffer(3)]],
    device const float* Bptr [[buffer(4)]],
    device float* exact_or_neg_inf [[buffer(5)]],
    constant uint& D [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    float B = Bptr[0];
    if (U[row] < B) {
        if (lane == 0) exact_or_neg_inf[row] = NEG_INF_F;
        return;
    }
    ulong base = (ulong)row * (ulong)D;
    float acc = 0.0f;
    for (uint j = lane; j < D; j += SG) {
        ushort raw = ((ushort)high[base + j] << 8) | (ushort)low[base + j];
        acc = fma(h[j], (float)as_type<half>(raw), acc);
    }
    float total = simd_sum(acc);
    if (lane == 0) exact_or_neg_inf[row] = total;
}
