# ProofBits Native-BF16 Audit — 2026-08-15

This note records the native-BF16 branch separately from the established FP16 8+8 implementation.

## Why byte-aligned 8+8 fails for BF16

BF16 uses 1 sign bit, 8 exponent bits, and 7 fraction bits. Splitting the raw 16-bit word after its first byte leaves one exponent bit in the unknown suffix. Therefore the interval represented by an 8-bit prefix can span an enormous numerical range even though the representation is lossless after refinement.

The direct 8+8 kill test confirms this failure mode.

### Qwen2.5-0.5B

Eight natural hidden states, original BF16 output-head storage:

- exactness after full refinement: 8/8;
- mean survivors at p=8: **151,936 / 151,936 = 100%**;
- ideal logical traffic reduction: **1.0×**.

The same hidden states with an FP16-rounded control retain only ~133.9 / 151,936 rows on average (~0.0881%), confirming that the failure is the BF16 bit layout rather than the decision certificate itself.

### Gemma prefix-sweep cross-check

The Gemma p=8 prefix test leaves ~99.96% of 262,144 rows alive, independently confirming that native BF16 8+8 is not a viable bandwidth-saving split.

**Decision: native BF16 8+8 is rejected.**

---

## Non-byte-aligned BF16 prefix sweep

Let p be the number of most-significant BF16 storage bits fetched on the first pass. For a survivor fraction f_p, the ideal densely-packed logical traffic is

\[
b(p)=p+(16-p)f_p\quad \text{bits/weight},
\]

with ideal traffic reduction

\[
R(p)=\frac{16}{b(p)}.
\]

This is a storage/traffic calculation, not yet a measured kernel speedup.

### Qwen2.5-0.5B

Two natural hidden states:

| Prefix p | Mean survivor fraction | Mean logical bits/weight | Ideal traffic reduction |
|---:|---:|---:|---:|
| 8 | 100% | 16.000 | 1.000× |
| 9 | 71.41% | 13.999 | 1.146× |
| **10** | **2.464%** | **10.148** | **1.577×** |
| 11 | 0.1241% | 11.006 | 1.454× |
| 12 | 0.00461% | 12.000 | 1.333× |
| 13 | 0.00132% | 13.000 | 1.231× |
| 14 | 0.00099% | 14.000 | 1.143× |

All tested decisions are exact after refinement.

### Gemma 3 270M

Two natural hidden states:

| Prefix p | Mean survivor fraction | Mean logical bits/weight | Ideal traffic reduction |
|---:|---:|---:|---:|
| 8 | 99.960% | 15.997 | 1.000× |
| 9 | 47.822% | 12.348 | 1.296× |
| **10** | **0.3677%** | **10.022** | **1.596×** |
| 11 | 0.01621% | 11.001 | 1.454× |
| 12 | 0.00153% | 12.000 | 1.333× |
| 13 | 0.000572% | 13.000 | 1.231× |
| 14 | 0.000381% | 14.000 | 1.143× |

All tested decisions are exact after refinement.

## Cross-model conclusion

Both Qwen and Gemma independently select **p=10** as the logical-traffic optimum among p=8..14:

- Qwen: ~**1.577×** ideal weight-traffic reduction;
- Gemma: ~**1.596×** ideal weight-traffic reduction.

p=11 is a different Pareto point: it sacrifices some prefix bandwidth reduction but reduces suffix survivors by roughly one to two orders of magnitude.

The primary native-BF16 candidate is therefore:

> **ProofBits-BF16 10+6:** densely pack the most-significant 10 storage bits as the certified prefix plane, and fetch the remaining 6 bits only for survivors.

For p=10 the prefix includes sign + the full BF16 exponent + the top fraction bit. Unlike 8+8, no exponent bit remains unknown, which explains the large collapse in interval width.

---

## Raw checkpoint-storage validation on MLX

For `mlx-community/gemma-3-270m-bf16`, MLX exposes the BF16 lm-head array through the Python buffer protocol:

- shape: 262,144 × 640;
- raw storage: **335,544,320 bytes**;
- raw 16-bit storage words can be read directly;
- decoding a raw BF16 word by placing it in the high 16 bits of an IEEE FP32 word reproduces the MLX BF16 value bit-exactly on the tested sample.

Therefore a packed 10+6 representation can be created directly from the checkpoint's BF16 storage without first changing its numerical values.

---

## Real packed 10+6 Metal implementation — lane-local unpack

A native-BF16 Metal implementation was built for `mlx-community/gemma-3-270m-bf16`.

Storage:

- original BF16 lm head: **335,544,320 bytes**;
- packed 10-bit prefix plane: **209,715,200 bytes**;
- packed 6-bit suffix plane: **125,829,120 bytes**;
- one-time pack setup: ~**2.99 s**, excluded from generation timing.

The matched dense reference reads the original BF16 head and accumulates row scores in FP32 with the same one-SIMDgroup-per-row reduction structure. ProofBits reconstructs exact BF16 values from packed 10+6 bits for the pilot/survivors.

Four AB/BA trajectories ×48 generated tokens:

- **192/192 tokens exact** under the matched BF16 reference semantics;
- median decision-head speedup: **1.183×**;
- median integrated autoregressive speedup: **1.102×**;
- mean integrated speedup: **1.099×**;
- integrated diagnostic mean survivors: **86.81 / 262,144 = 0.0331%**.

Per-round integrated speedups were approximately 1.012×, 1.125×, 1.181×, and 1.079×.

**Conclusion:** native checkpoint-BF16 ProofBits is not merely a logical traffic construction. The first packed implementation produces a modest but real matched-reference wall-clock gain while preserving the original BF16 values exactly.

The measured head gain is much smaller than the ideal ~1.596× traffic ceiling, indicating that packed bit extraction and address/control overhead are first-order costs.

---

## Falsified optimization: SIMD-cooperative 10-bit unpack

A second kernel attempted to force physical prefix loads closer to the ideal 10 bits/weight. For each 32-weight block, only lanes 0..9 loaded the ten packed uint32 words and distributed them to all 32 lanes with `simd_shuffle`.

This **failed decisively on Apple M1** despite preserving exactness:

- **192/192 tokens exact**;
- median decision-head speedup: **0.531×** (i.e. substantially slower than dense);
- median integrated speedup: **0.705×**;
- per-round integrated ratios: ~0.789×, 0.685×, 0.725×, 0.677×.

Survivor counts were unchanged, so the regression is an unpack/execution-path effect rather than a certificate-quality failure.

**Decision: reject the SIMD-cooperative M1 implementation.** Lower payload traffic is insufficient when shuffle/unpack cost dominates. The lane-local packed implementation remains the current positive native-BF16 kernel.

---

## Current claim boundary

Supported:
- BF16 8+8 is decisively unsuitable because the prefix cuts the exponent field;
- p=10 is the best logical-traffic point in the tested p=8..14 sweep on both Qwen and Gemma;
- exact decision recovery is retained in all prefix-sweep tests;
- raw MLX BF16 storage can be captured exactly for packing;
- a real packed BF16 10+6 Metal path preserves 192/192 matched generated decisions and yields ~**1.18× head / 1.10× integrated** speedup on the tested Gemma trajectory;
- a seemingly more bandwidth-efficient cooperative unpack can be slower, establishing that unpack architecture is part of the systems problem rather than an implementation footnote.

Not yet supported:
- superiority of the positive p=10 kernel over MLX's optimized native BF16 `lm_head` serving path;
- cross-prompt / multi-runner native-BF16 p=10 robustness;
- CUDA/Triton implementation of p=10;
- energy/DRAM-counter gains.

The immediate gate is a same-run AB/BA comparison between the positive lane-local p=10 kernel and native MLX BF16 `lm_head + argmax`. Sequence equality is measured empirically rather than assumed because vendor reduction semantics can differ.
