#include <metal_stdlib>
using namespace metal;

// One 32-lane SIMDgroup computes one vocabulary row. Qwen D=896 is exactly
// 28 values/lane; Gemma D=640 is 20 values/lane. Dense and ProofBits use the
// identical work decomposition so the stage comparison is matched.
constant uint SG = 32;

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
        ushort raw = full_bits[base + j];
        half w = as_type<half>(raw);
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
