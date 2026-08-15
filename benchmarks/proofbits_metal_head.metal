#include <metal_stdlib>
using namespace metal;

// One 128-thread threadgroup computes one vocabulary row.
// D is runtime to keep the same kernel usable for Qwen/Gemma follow-ups.
constant uint TG = 128;

kernel void dense_fp16_row(
    device const ushort* full_bits [[buffer(0)]],
    device const float* h [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& D [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    threadgroup float* scratch [[threadgroup(0)]])
{
    float acc = 0.0f;
    const ulong base = (ulong)row * (ulong)D;
    for (uint j = tid; j < D; j += TG) {
        ushort raw = full_bits[base + j];
        half w = as_type<half>(raw);
        acc = fma(h[j], (float)w, acc);
    }
    scratch[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = TG >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) out[row] = scratch[0];
}

kernel void proofbits_high_upper_row(
    device const uchar* high [[buffer(0)]],
    device const float* h [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& D [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    threadgroup float* scratch [[threadgroup(0)]])
{
    float acc = 0.0f;
    const ulong base = (ulong)row * (ulong)D;
    for (uint j = tid; j < D; j += TG) {
        uchar hb = high[base + j];
        ushort weightSign = (ushort)(hb & (uchar)0x80);
        ushort hiddenSign = (h[j] < 0.0f) ? (ushort)0x80 : (ushort)0x00;
        ushort suffix = (weightSign == hiddenSign) ? (ushort)0x00FF : (ushort)0x0000;
        ushort raw = ((ushort)hb << 8) | suffix;
        half endpoint = as_type<half>(raw);
        acc = fma(h[j], (float)endpoint, acc);
    }
    scratch[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = TG >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) out[row] = scratch[0];
}
