# ProofBits Full-Model FP16 Runtime Audit — 2026-08-15

This note supersedes the earlier mixed-precision `BF16 transformer body + FP16 output head` concern for the main Apple-M1 practical result.

## Experimental state

Source model: `mlx-community/gemma-3-270m-bf16`.

Before any warmup or timing, the entire MLX model is converted with `model.set_dtype(mx.float16)`. The benchmark explicitly verifies:

- runtime output-head dtype: `mlx.core.float16`;
- runtime transformer-body output dtype: `mlx.core.float16`.

Thus this is a **full-model FP16 runtime state**, not a BF16 body with only an FP16 test head.

The downloadable source checkpoint remains BF16; this experiment does **not** claim that the upstream repository publishes a distinct official FP16 checkpoint. It tests the actual deployed in-memory model after a full runtime conversion to FP16.

## Compared paths

All paths share the same converted FP16 model weights, transformer body, KV cache, prompts, and autoregressive token-feedback loop.

### Matched dense FP16 reference

- complete FP16 output head;
- one-SIMDgroup-per-row custom Metal scoring;
- FP32 row accumulation/reduction used by the formal ProofBits reference semantics;
- exact argmax.

### ProofBits FP16

- same stored FP16 output-head values;
- 8-bit high plane read for every row;
- certified interval upper scores;
- pilot = `argmax(U)`;
- exact one-row pilot score;
- low plane fetched only for rows with `U_i >= B`;
- final exact argmax under the matched reference accumulation/tie semantics.

### Native MLX FP16 reference

- native MLX FP16 output-head matvec + argmax;
- same full-FP16 transformer body and KV cache.

Native MLX equality is empirical rather than part of the theorem because MLX may use different reduction trees/accumulation semantics.

---

# Initial full-FP16 validation

Five prompt families ×32 generated tokens =160 paired decisions.

Results:

- matched dense vs ProofBits: **160/160 exact**;
- median integrated matched speedup: **1.280×**;
- mean matched speedup: **1.252×**;
- prompt range: **1.156×–1.301×**, all positive;
- native MLX FP16 vs ProofBits: **160/160 empirically identical generated tokens** in this initial suite;
- median native-MLX/PB latency ratio: **1.280×**;
- native comparison range: **1.200×–1.341×**.

This initial suite established that the earlier mixed-precision concern does not explain the FP16 speedup.

---

# Replicated 12-prompt full-FP16 stress suite

Three independently provisioned GitHub-hosted `macos-15` Apple-M1 runners were used.

Per runner:

- 12 heterogeneous prompts;
- 48 generated tokens per prompt;
- 576 paired matched-reference decisions;
- three serving methods: matched custom dense FP16, ProofBits FP16, native MLX FP16;
- method ordering rotated across prompts so each method appears first/second/third four times;
- no ProofBits-only diagnostic reductions in the timed paths.

Total matched decisions across the three independent runners:

\[
3\times 12\times 48 = \boxed{1728}.
\]

All **1728/1728** ProofBits decisions match the formal dense FP16 reference.

## Runner-level matched-reference results

| Independent M1 runner | Exact matched tokens | Median of prompt-median speedups | Pooled end-to-end speedup |
|---:|---:|---:|---:|
| 1 | 576/576 | **1.306×** | **1.220×** |
| 2 | 576/576 | **1.247×** | **1.169×** |
| 3 | 576/576 | **1.194×** | **1.046×** |

Across runners:

- matched exactness: **1728/1728**;
- median of runner prompt-medians: **1.247×**;
- mean of runner prompt-medians: **1.249×**;
- median pooled end-to-end speedup: **1.169×**;
- mean pooled speedup: **1.145×**;
- pooled speedup is positive on **3/3 independently provisioned runners**.

The third runner is materially noisier/slower for ProofBits in mean-latency tails, so the strongest defensible aggregate is the runner-median pooled gain (~1.17×), not the best single-run number.

## Native MLX FP16 serving reference

Runner-level median prompt ratios `native MLX latency / ProofBits latency` were approximately:

- runner 1: **1.179×**;
- runner 2: **1.153×**;
- runner 3: **1.172×**.

Across runners:

- median of runner prompt-medians: **1.172×**;
- mean: **1.168×**;
- pooled native/PB ratios: about **1.146× / 1.068× / 1.023×**;
- median pooled native/PB ratio: **1.068×**.

However, native MLX and ProofBits do **not** generate identical trajectories for every prompt. In each stress runner, some prompts diverge because the native reduction/accumulation semantics differ from the formal matched reference.

Therefore:

> Native MLX is a practical performance reference, not the formal exactness baseline.

The exactness theorem and 1728/1728 empirical validation refer to the explicitly specified matched FP16 reference semantics.

---

# Interpretation

The full-model FP16 result removes an important experimental confound:

> The earlier practical speedup was not merely an artifact of combining a BF16 transformer body with a specially converted FP16 output head.

When **both transformer body and output head run in FP16**, ProofBits still:

1. reproduces every tested matched-reference autoregressive decision exactly;
2. accelerates integrated greedy decoding on all three independent M1 runners in pooled latency;
3. remains competitive with, and on aggregate faster than, native MLX FP16 in the tested low-batch large-vocabulary workload.

The effect is smaller and noisier than the isolated decision-head speedup because the transformer body and runtime overhead are unchanged. This is consistent with the previously measured Amdahl decomposition where the large Gemma output head accounts for roughly half of decode latency.

---

# Current strongest Apple-M1 claim

A conservative practical statement is:

> **For a compact, 262k-vocabulary Gemma runtime converted fully to FP16, ProofBits exactly preserves a specified stored-FP16 greedy-decision semantics and yields a reproducible integrated decode-speed improvement across three independent Apple-M1 runners. Across 1,728 paired generated decisions, exactness is 100%; the median runner-level pooled end-to-end speedup is about 1.17×, while the median runner-level prompt speedup is about 1.25×.**

This is stronger than the earlier mixed-precision experiment because the entire inference runtime is FP16.

---

# Negative and boundary evidence that remains important

- Qwen cross-prompt improvement is much weaker and has a known slow prompt in the earlier suite.
- CPU AVX2/F16C execution is slower than dense FP16.
- native BF16 8+8 fails because the byte boundary cuts the exponent.
- packed native-BF16 10+6 beats a matched custom dense BF16 kernel modestly, but remains slower than native MLX BF16 on Apple M1.
- a seemingly more bandwidth-efficient SIMD-cooperative BF16 unpack is substantially slower, showing that physical load reduction alone is insufficient.
- full-softmax/top-p remains much less attractive than greedy/top-k.
- no NVIDIA CUDA/Triton + Nsight result has yet been established.

These negative results should remain in the paper because they define the actual systems regime in which conditional bit fetching is useful.

---

# Next critical hardware gate

The main unresolved external-validity question is now datacenter-GPU transfer:

1. CUDA/Triton implementation of the FP16 8+8 pilot=1 fast path;
2. same-GPU dense vs ProofBits counterbalancing;
3. hardware DRAM-byte counters (Nsight Compute / equivalent);
4. batch-size sweep;
5. output-head and integrated decode latency;
6. exact matched-reference decision audit.

Until that is run, Apple M1 is the strongest measured accelerator result and NVIDIA speedup must not be claimed.
