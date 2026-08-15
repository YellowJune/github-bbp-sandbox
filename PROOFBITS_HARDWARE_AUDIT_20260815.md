# ProofBits Hardware Audit — 2026-08-15

This note freezes the strongest hardware results and the negative results observed so far. It intentionally separates logical traffic, matched custom kernels, strong vendor-library baselines, and whole-model projections.

## Final main fast path

Lossless FP16 byte planes, upper-only, pilot=1:

1. read high byte for every output-head weight;
2. reconstruct the exact score-maximizing FP16 interval endpoint from hidden/weight sign bits;
3. compute certified upper score `U_i`;
4. `p = argmax(U)`;
5. read the low byte of pilot row `p`, obtaining exact threshold `B=z_p`;
6. refine only rows with `U_i >= B`;
7. return exact stored-FP16 argmax using matched tie policy.

The pilot need not be the winner. Correctness follows from `U_* >= z_* >= z_p = B`.

---

## CPU negative result

Actual Qwen2.5-0.5B head, V=151,936, D=896, eight real WikiText hidden states, cache-flushed AVX2/F16C microbenchmark.

LUT-free direct endpoint reconstruction:

| Threads | Dense FP16 | ProofBits | Speedup |
|---:|---:|---:|---:|
| 1 | 19.057 ms | 21.313 ms | 0.894x |
| 2 | 10.419 ms | 11.697 ms | 0.891x |
| 4 | 7.142 ms | 8.443 ms | 0.846x |

All decisions were exact; mean survivor count was 79.75.

Stage breakdown showed that the control path was not the main problem. The high-byte upper pass itself was slower than dense FP16 on this CPU:

- 1 thread: dense 15.777 ms vs high pass 22.600 ms (0.698x)
- 4 threads: dense 9.804 ms vs high pass 13.384 ms (0.733x)

**Conclusion:** half the logical weight bytes do not automatically imply a speedup. Endpoint reconstruction / FP16 conversion can dominate on CPUs without a favorable native execution path.

---

## Apple M1 matched custom Metal — Qwen stage

Actual Qwen2.5-0.5B head and real hidden states.

Matched one-SIMDgroup-per-row custom Metal kernels:

- dense FP16 row scoring: ~6.388 ms GPU median
- ProofBits high-byte upper scoring: ~3.379 ms GPU median
- stage speedup: **1.890x**
- synchronized wall speedup: ~1.869x

Logical weight bytes:

- dense: 272.27 MB
- high plane: 136.13 MB

Approximate effective rates were similar (~42.6 GB/s dense vs ~40.3 GB/s high), supporting the intended memory-traffic mechanism rather than an unrelated arithmetic shortcut.

---

## Full custom Metal decision-head — Qwen

Full GPU pipelines:

Dense:
`dense FP16 scores -> GPU argmax`

ProofBits:
`high upper -> argmax(U) -> exact pilot -> conditional low-byte refine -> GPU argmax`

Three real Qwen hidden states, 12 timed repetitions/query:

- exact: 3/3
- survivor rows: 10, 88, 289
- two of three pilots were *not* the true winner, yet final decision remained exact

Timing:

- dense custom GPU median: **6.933 ms**
- ProofBits GPU median: **4.051 ms**
- GPU speedup: **1.712x**
- synchronized wall speedup: **1.689x**

---

## Full custom Metal decision-head — Gemma 3 270M

Model mirror: `unsloth/gemma-3-270m`, V=262,144, D=640.

Three real hidden states, 12 repetitions/query:

- exact: 3/3
- survivors: 8, 126, 25

Timing:

- dense custom GPU median: **8.368 ms**
- ProofBits GPU median: **4.618 ms**
- GPU speedup: **1.812x**
- wall speedup: **1.763x**

This cross-model result weakens the hypothesis that the Qwen hardware result is an idiosyncrasy of one vocabulary/head geometry.

---

## Strong vendor-library baseline — direct Apple MPS

A PyTorch MPS `torch.mv` baseline was measured but is **not** used as the primary baseline because PyTorch emitted a runtime warning that this M=1 shape hits a known MPS matrix-multiplication issue and falls back to a generic Metal implementation.

The stronger baseline directly invokes Apple's `MPSMatrixVectorMultiplication` on FP16 weights, uses Apple's recommended matrix row stride, and appends a custom GPU FP16 argmax in the same command buffer.

### Qwen — same-VM counterbalanced

Four rounds on the same GitHub `macos-15` M1 VM, alternating execution order MPS->PB / PB->MPS.

Per-round GPU speedups:

1. 1.662x
2. 1.608x
3. 1.557x
4. 1.557x

Summary:

- **median GPU speedup: 1.583x**
- mean GPU speedup: 1.596x
- **median wall speedup: 1.572x**
- all ProofBits rounds exact

This is the strongest Qwen system result so far.

### Gemma — same-VM counterbalanced

Four same-M1 rounds against direct `MPSMatrixVectorMultiplication + GPU argmax`:

- **median GPU speedup: ~1.633x**
- **median wall speedup: ~1.619x**
- all ProofBits rounds exact

This is the strongest cross-model confirmation so far.

---

## Whole-token impact: current projection only

Qwen PyTorch-MPS body-only KV-cache decode (lm_head intentionally omitted) had steady-state median ~50.70 ms/token on the hosted M1 setup.

Combining that body measurement with same-hardware head medians gives only a component-level projection:

`(body + dense head) / (body + ProofBits head) ~= 1.041x`.

Thus Qwen's measured ~1.58x decision-head speedup projects to only ~4% whole-token improvement in that particular PyTorch-MPS body stack.

**This is not an integrated end-to-end ProofBits benchmark.** Framework/body overhead dominates the Qwen setup. The practical main target should be head-dominant compact / huge-vocabulary models and/or an integrated runtime such as MLX where the body and custom head share the same graph/runtime.

---

## ProofBits+ logical ceiling

Lossless high-byte compression is a secondary branch, not the current main system.

A 6-bit palette+escape representation covers ~98.2–98.6% of high bytes and gives ~6.11–6.14 logical bits/weight for the prefix plane across Qwen0.5, Gemma270M, and Qwen1.5B.

If suffix traffic remains negligible, the logical FP16/full-prefix ceiling becomes ~2.60–2.62x rather than 2x. No packed GPU decoder has been benchmarked, so do not turn this into a latency claim.

---

## Current claim boundary

Supported now:

> On an Apple M1 accelerator, a lossless decision-certified FP16 output head can skip almost all low-byte weight reads and achieve an exact decision-head speedup of roughly **1.58x (Qwen)** and **1.63x (Gemma)** versus a direct Apple MPS FP16 matrix-vector + GPU-argmax baseline in same-VM counterbalanced tests.

Not yet supported:

- integrated whole-transformer token/s speedup;
- NVIDIA CUDA/Triton latency or Nsight DRAM counters;
- universal speedup across hardware;
- full softmax/top-p efficiency comparable to argmax/top-k;
- large-batch throughput superiority;
- production-grade fused implementation.

## Highest-priority next experiment

Integrate the ProofBits custom Metal head directly into an MLX-LM autoregressive decode path. MLX supports custom Metal kernels and unified CPU/GPU memory, so it may permit the body hidden state to flow directly into ProofBits without the PyTorch/Swift component boundary. If successful, measure exact greedy token equality and actual tokens/s versus native MLX dense decoding.
