#include <metal_stdlib>
using namespace metal;

constant uint SG_H = 32;
constant uint ARG_CHUNK_H = 4096;
constant float NEG_INF_H = -3.402823466e+38f;

kernel void argmax_half_stage1(
    device const half* values [[buffer(0)]],
    device float* block_values [[buffer(1)]],
    device uint* block_indices [[buffer(2)]],
    constant uint& N [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint begin = group * ARG_CHUNK_H;
    uint end = min(begin + ARG_CHUNK_H, N);
    float best = NEG_INF_H;
    uint bestIdx = 0xffffffffu;
    for (uint j = begin + lane; j < end; j += SG_H) {
        float v = (float)values[j];
        if (v > best || (v == best && j < bestIdx)) { best = v; bestIdx = j; }
    }
    for (ushort off = 16; off > 0; off >>= 1) {
        float ov = simd_shuffle_down(best, off);
        uint oi = simd_shuffle_down(bestIdx, off);
        if (lane < off && (ov > best || (ov == best && oi < bestIdx))) { best = ov; bestIdx = oi; }
    }
    if (lane == 0) { block_values[group] = best; block_indices[group] = bestIdx; }
}

kernel void argmax_float_final_mps(
    device const float* block_values [[buffer(0)]],
    device const uint* block_indices [[buffer(1)]],
    device uint* out_index [[buffer(2)]],
    constant uint& Nblocks [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]])
{
    float best = NEG_INF_H;
    uint bestIdx = 0xffffffffu;
    for (uint j = lane; j < Nblocks; j += SG_H) {
        float v = block_values[j]; uint idx = block_indices[j];
        if (v > best || (v == best && idx < bestIdx)) { best = v; bestIdx = idx; }
    }
    for (ushort off = 16; off > 0; off >>= 1) {
        float ov = simd_shuffle_down(best, off);
        uint oi = simd_shuffle_down(bestIdx, off);
        if (lane < off && (ov > best || (ov == best && oi < bestIdx))) { best = ov; bestIdx = oi; }
    }
    if (lane == 0) out_index[0] = bestIdx;
}
