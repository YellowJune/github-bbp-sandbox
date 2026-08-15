# ProofBits Hardware Audit — 2026-08-15

This note freezes the strongest positive and negative hardware results observed so far. It separates logical traffic, matched custom kernels, vendor-library baselines, and integrated autoregressive decode. Later sections supersede earlier projections where explicitly stated.

## Final main fast path

Lossless FP16 byte planes, upper-only, pilot=1:

1. read the high byte for every output-head weight;
2. reconstruct the exact score-maximizing FP16 interval endpoint from hidden/weight sign bits;
3. compute a certified upper score `U_i` for every row;
4. choose `p = argmax(U)`;
5. read the low byte of pilot row `p`, obtaining exact threshold `B = z_p`;
6. refine only rows with `U_i >= B`;
7. return the exact stored-FP16 argmax under the specified accumulation/tie semantics.

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

Stage breakdown showed that control flow was not the main problem. The high-byte upper pass itself was slower than dense FP16 on this CPU:

- 1 thread: dense 15.777 ms vs high pass 22.600 ms (0.698x)
- 4 threads: dense 9.804 ms vs high pass 13.384 ms (0.733x)

**Conclusion:** reducing logical weight bytes does not guarantee speedup. Endpoint reconstruction / FP16 conversion can dominate on an unfavorable CPU execution path.

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

## Strong vendor-library decision-head baseline — direct Apple MPS

A PyTorch MPS `torch.mv` baseline was measured but is **not** used as the primary baseline because PyTorch emitted a runtime warning that this M=1 shape hits a known MPS matrix-multiplication issue and falls back to a generic Metal implementation.

The stronger baseline directly invokes Apple's `MPSMatrixVectorMultiplication` on FP16 weights, uses Apple's matrix row-stride requirements, and appends a custom GPU FP16 argmax in the same command buffer.

### Qwen — same-VM counterbalanced

Four rounds on the same GitHub `macos-15` M1 VM, alternating MPS->PB / PB->MPS.

Per-round GPU speedups:

1. 1.662x
2. 1.608x
3. 1.557x
4. 1.557x

Summary:

- **median GPU speedup: 1.583x**
- mean GPU speedup: 1.596x
- **median wall speedup: 1.572x**
- all ProofBits rounds exact relative to the matched reference

### Gemma — same-VM counterbalanced

Four same-M1 rounds against direct `MPSMatrixVectorMultiplication + GPU argmax`:

- **median GPU speedup: 1.633x**
- **median wall speedup: 1.619x**
- all ProofBits rounds exact relative to the matched reference

---

# Integrated MLX autoregressive decode

These experiments supersede the earlier component-only whole-token projection. The MLX-LM transformer body, KV cache, ProofBits head, token materialization, and token feedback all execute inside the timed autoregressive loop.

Both the matched dense baseline and ProofBits use the same lossless FP16 output-head storage. Conversion from the downloaded BF16 checkpoint to the test FP16 head representation and byte-plane construction occur outside timed generation.

## Qwen2.5-0.5B — integrated matched-reference decode

Model: `mlx-community/Qwen2.5-0.5B-Instruct-bf16`, V=151,936, D=896.

Four AB/BA counterbalanced trajectories, 48 generated tokens each:

| Round | Dense custom FP16 | ProofBits | Speedup | Sequence exact |
|---:|---:|---:|---:|:---:|
| 1 | 25.191 ms | 20.972 ms | 1.201x | yes |
| 2 | 25.243 ms | 20.920 ms | 1.207x | yes |
| 3 | 26.797 ms | 20.760 ms | 1.291x | yes |
| 4 | 23.651 ms | 21.149 ms | 1.118x | yes |

Summary:

- **192/192 generated tokens match exactly** between the matched dense FP16 reference and ProofBits.
- **median integrated speedup: 1.204x**
- median dense latency across rounds: ~25.217 ms/token
- median ProofBits latency across rounds: ~20.946 ms/token

A separate diagnostic trajectory measured mean survivor count ~44.15 / 151,936 (~0.0291%), but its timing is not used because the diagnostic adds an extra survivor-count reduction.

### Qwen multi-prompt robustness

Five prompt families (explanation, algebra, code, systems summary, creative writing), 32 tokens each, independent KV caches, alternating dense/PB order, no asymmetric diagnostics in either timed path:

- **160/160 tokens exact** under the matched FP16 reference.
- median speedup: **1.064x**
- mean speedup: **1.054x**
- range: **0.958x to 1.117x**
- one prompt was slower with ProofBits (~0.958x).

**Interpretation:** Qwen provides strong exactness evidence but not a universal latency win. The earlier single-prompt 1.204x integrated result is real for that trajectory, but should not be presented as a prompt-independent Qwen speedup. The defensible cross-prompt number is the more modest ~1.06x median, with an explicit negative prompt case.

---

## Gemma 3 270M — integrated matched-reference decode

Model: `mlx-community/gemma-3-270m-bf16`, V=262,144, D=640, explicit `lm_head.weight`.

Four AB/BA trajectories, 48 generated tokens each:

| Round | Dense custom FP16 | ProofBits | Speedup | Sequence exact |
|---:|---:|---:|---:|:---:|
| 1 | 15.577 ms | 12.770 ms | 1.220x | yes |
| 2 | 16.505 ms | 12.812 ms | 1.288x | yes |
| 3 | 19.897 ms | 14.833 ms | 1.341x | yes |
| 4 | 21.481 ms | 13.920 ms | 1.543x | yes |

Summary:

- **192/192 generated tokens exact** under the matched FP16 reference.
- **median integrated speedup: 1.315x**
- median dense latency across rounds: ~18.201 ms/token
- median ProofBits latency across rounds: ~13.366 ms/token

A separate diagnostic ProofBits trajectory found:

- mean survivors: **6.21 / 262,144 = 0.00237%**
- median survivors: **1.5 rows**

This is the strongest integrated matched-reference case so far and is consistent with the intended deployment target: compact models with very large vocabularies / output spaces where the decision head is a large fraction of decode cost.

---

## Integrated native-MLX FP16 serving reference

To avoid an unfair comparison, native MLX FP16 matvec and ProofBits were rerun in 4-round AB/BA tests with **no survivor diagnostics in either timed path**. Both use the same FP16 head storage and the same MLX KV-cache body.

### Qwen

Per-round native-MLX / ProofBits latency ratios:

- 1.145x
- 1.107x
- 1.109x
- 1.090x

Summary:

- **median speedup vs native MLX FP16: 1.108x**
- however, native MLX and ProofBits generated sequences differ in these tests because reduction/accumulation semantics differ.

Therefore this Qwen comparison is a **performance reference only**, not an exact-equivalent serving claim.

### Gemma

Per-round native-MLX / ProofBits ratios:

- 1.087x
- 1.130x
- 1.206x
- 1.185x

Summary:

- **median speedup vs native MLX FP16: 1.158x**
- all 4x48-token tested trajectories matched between native MLX FP16 and ProofBits.

This is currently the strongest practical serving result. The sequence equality is empirical for the tested trajectories; mathematical exactness is still defined relative to the specified matched FP16 accumulation/tie semantics, not arbitrary vendor reduction orders.

---

## Exactness semantics

The formal exactness claim is deliberately narrow and reproducible:

> ProofBits returns the same decision as evaluation of the same stored FP16 weights under the specified reference accumulation/reduction and tie policy.

Vendor libraries may use different reduction trees, fused operations, accumulator dtypes, or tie behavior. Therefore native-framework equality is measured empirically and never assumed. Qwen demonstrates that such differences can alter a generated trajectory; Gemma demonstrates a case where the tested native trajectory still matches.

---

## ProofBits+ logical ceiling

Lossless high-byte compression remains a secondary branch, not the current main system.

A 6-bit palette+escape representation covers ~98.2–98.6% of high bytes and gives ~6.11–6.14 logical bits/weight for the prefix plane across Qwen0.5, Gemma270M, and Qwen1.5B.

If suffix traffic remains negligible, the logical FP16/full-prefix ceiling becomes ~2.60–2.62x rather than 2x. No packed GPU decoder has been benchmarked, so this is **not** a latency claim.

---

# Current defensible claim boundary

## Supported now

1. **Exact matched-reference decision:** ProofBits can avoid almost all FP16 suffix reads while exactly reproducing the specified stored-FP16 reference argmax/top-k decision.
2. **Actual accelerator speedup:** on Apple M1, full decision-head speedups of ~1.58x (Qwen) and ~1.63x (Gemma) were measured against direct Apple MPS FP16 matrix-vector + GPU argmax in same-VM counterbalanced tests.
3. **Integrated autoregressive speedup:** in MLX-LM, matched-reference greedy decoding showed ~1.204x on one Qwen trajectory and ~1.315x on one Gemma trajectory, with all matched tokens exact.
4. **Cross-prompt Qwen exactness:** 160/160 tokens across five prompt families were exact; median integrated speedup was ~1.064x, but one prompt was slower (~0.958x).
5. **Native serving reference:** Gemma ProofBits was ~1.158x faster than native MLX FP16 in a fair no-diagnostic counterbalanced test and generated the same tested trajectories.
6. **Large-vocabulary behavior:** Gemma's 262k-vocabulary integrated diagnostic retained only ~6.21 rows on average for suffix refinement.

## Not supported

- universal speedup across prompts, models, hardware, or batch sizes;
- mathematical equivalence to arbitrary vendor-library reduction semantics;
- NVIDIA CUDA/Triton latency or Nsight DRAM-counter claims;
- efficient full-softmax/top-p comparable to greedy/top-k;
- large-batch throughput superiority;
- production-grade packed ProofBits+ latency gains.

## Highest-priority next experiments

1. **Gemma cross-prompt integrated replication** — strongest deployment case must be tested across prompt families, not one trajectory.
2. **Context-length sweep** — measure integrated speedup as KV/body cost rises, to quantify the Amdahl boundary directly.
3. **NVIDIA CUDA/Triton benchmark + Nsight DRAM counters** — test whether conditional byte fetching transfers to datacenter GPU memory systems.
4. **Longer trajectories / multiple seeds/prompts** — tighten confidence intervals and characterize rare slow cases.
5. **Optional ProofBits+ packed decoder** — only after the 8+8 implementation is fully characterized.
