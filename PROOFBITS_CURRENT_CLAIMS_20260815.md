# ProofBits — Current Defensible Paper Claims (2026-08-15)

## One-sentence thesis

**ProofBits accelerates exact low-batch decision heads by reading only a decision-certifying prefix of each stored numerical weight and fetching the remaining bits only for rows that can still change the discrete output.**

The key distinction from approximate quantization / progressive precision is that suffix memory access is **conditionally eliminated by a proof that the final decision cannot change**, rather than tolerated as an accuracy tradeoff.

---

## Core FP16 algorithm

For FP16 stored weights, split each 16-bit word into high and low bytes.

1. Read the high byte for every output-head weight.
2. From the known high byte and the hidden-value sign, reconstruct the exact endpoint of the compatible FP16 interval that maximizes each product.
3. Sum those endpoints to obtain a certified row upper bound `U_i`.
4. Choose `p = argmax_i U_i`.
5. Read the low byte of only pilot row `p` and compute its exact score `B = z_p`.
6. Read low bytes only for rows satisfying `U_i >= B`.
7. Compute exact scores for those survivors and return the exact argmax/top-k under the specified matched accumulation and tie semantics.

The pilot is not required to be the winner. If `*` is the true winner,

\[
U_* \ge z_* \ge z_p = B,
\]

so the true winner can never be incorrectly pruned.

---

# Strongest practical result

## Full-model FP16 Gemma runtime on Apple M1

Source checkpoint: `mlx-community/gemma-3-270m-bf16`.

The entire MLX model is converted to FP16 **before warmup/timing** using `model.set_dtype(mx.float16)`. Runtime checks confirm:

- transformer-body output dtype = FP16;
- output-head dtype = FP16.

Thus the main result is no longer the earlier mixed `BF16 body + FP16 head` experiment.

### Replicated stress suite

- model output space: 262,144 vocabulary rows;
- 12 heterogeneous prompts;
- 48 generated tokens/prompt;
- 576 paired decisions/runner;
- 3 independently provisioned GitHub-hosted Apple-M1 runners;
- three-method execution order rotated across prompts;
- ProofBits-only diagnostics excluded from timing.

Formal matched-reference result:

\[
\boxed{1728/1728\text{ generated decisions exact}}
\]

Runner-level matched-reference results:

| Runner | Prompt-median speedup | Pooled end-to-end speedup |
|---:|---:|---:|
| 1 | 1.306× | 1.220× |
| 2 | 1.247× | 1.169× |
| 3 | 1.194× | 1.046× |

Aggregate:

- runner-median prompt speedup: **1.247×**;
- runner-mean prompt speedup: **1.249×**;
- runner-median pooled decode speedup: **1.169×**;
- runner-mean pooled decode speedup: **1.145×**;
- positive pooled end-to-end gain on **3/3 independent runners**.

The strong but conservative Apple-M1 headline is therefore approximately:

> **100% matched-reference decision fidelity across 1,728 paired autoregressive decisions with a median independent-runner pooled decode speedup of ~1.17× in a full-model FP16, 262k-vocabulary Gemma runtime.**

---

## Isolated decision-head evidence

On Apple M1, direct Apple `MPSMatrixVectorMultiplication FP16 + GPU argmax` same-VM counterbalanced baselines showed:

- Qwen2.5-0.5B: median GPU head speedup **~1.58×**;
- Gemma 3 270M: median GPU head speedup **~1.63×**.

Matched custom Metal decision heads showed up to roughly 1.7–1.8× on the tested Qwen/Gemma heads.

The integrated speedup is smaller, as expected, because transformer-body cost is unchanged.

---

## Amdahl/mechanism validation

A stabilized Gemma context sweep at prompt lengths 16, 64, 256, 512, and 1024 tokens found:

- all matched generated decisions exact;
- decision-head speedup roughly **1.71×–1.87×**;
- integrated speedup roughly **1.26×–1.39×** in that mixed-body experiment;
- dense decision head remained about **50%** of per-token latency through 1k context.

At 1024 tokens, the measured head fraction and head speedup produce an Amdahl prediction close to the measured total speedup. This supports the claim that the end-to-end gain comes from decision-head acceleration rather than unrelated runtime variation.

---

# Native optimized-runtime comparison

Native MLX FP16 is a **performance reference**, not the formal exactness reference, because its reduction tree/accumulation semantics can differ.

In the replicated full-FP16 stress suite, runner-level median prompt ratios `native MLX latency / ProofBits latency` were approximately:

- 1.179×;
- 1.153×;
- 1.172×.

Median across runners: **~1.172×**.

Pooled native/PB ratios were approximately 1.146×, 1.068×, and 1.023×.

Some native trajectories diverged from the matched FP16 reference, demonstrating why formal exactness must be tied to an explicitly specified reduction/tie semantics rather than an arbitrary vendor kernel.

---

# Cross-model / cross-prompt evidence

### Qwen2.5-0.5B

- exact certificate remains strong;
- 5-prompt integrated suite: 160/160 matched tokens exact;
- median speedup only ~1.06×;
- one prompt was slightly slower (~0.958×).

Therefore Qwen is **not** used to claim universal practical speedup.

### Gemma 3 270M

The large 262k output space is substantially more favorable.

Earlier mixed-body large stress suite:

- 12 prompts ×64 tokens ×3 independent M1 runners;
- **2304/2304 matched tokens exact**;
- all 36 runner×prompt median latency comparisons positive;
- median runner-level pooled speedup ~**1.273×**.

The newer full-model FP16 suite is the preferred main result because it removes the mixed-precision confound.

---

# Native BF16 branch: important negative/secondary result

A naive BF16 8+8 byte split fails because the first byte does not contain the full 8-bit exponent.

Observed:

- Qwen native BF16 p=8: essentially **100% survivor rate**;
- Gemma p=8: essentially **100% survivor rate**.

A non-byte-aligned prefix sweep finds p=10 to be the tested logical-traffic optimum:

- Qwen ideal logical traffic reduction ~**1.58×**;
- Gemma ~**1.60×**.

A real packed BF16 10+6 Metal implementation:

- preserves 192/192 matched generated decisions;
- beats a matched custom dense BF16 head modestly (~1.18× head, ~1.10× integrated);
- but is about **5% slower end-to-end than native MLX BF16** on the tested Gemma workload.

A SIMD-cooperative unpack intended to lower physical loads further was dramatically slower (~0.53× head), showing that decoder/unpack architecture is a first-order systems issue.

**Native BF16 is therefore not the current practical headline.**

---

# Explicit falsifications / boundaries

The paper must retain these negative findings:

1. **CPU:** ProofBits is slower on tested AVX2/F16C CPU kernels.
2. **Qwen cross-prompt:** not universally faster; one prompt loses.
3. **BF16 8+8:** fails because the byte boundary cuts the exponent.
4. **BF16 p10 vs native MLX:** current packed implementation loses to optimized native BF16.
5. **SIMD-cooperative p10 unpack:** lower apparent memory payload can still lose badly due shuffle/unpack overhead.
6. **Top-p/full softmax:** much weaker conditional-fetch savings than greedy/top-k.
7. **Large batch:** no universal throughput advantage claimed.
8. **NVIDIA:** no CUDA/Triton + Nsight wall-clock/DRAM-counter result has yet been measured.

These are not side notes; they define the regime where the mechanism is actually useful.

---

# Novelty-safe positioning

Do not frame the work merely as:

- progressive precision;
- approximate MIPS;
- bit-serial inference;
- quantized output heads;
- early-exit arithmetic.

The core novelty claim should be:

> **Decision-certified conditional bit fetching:** stored numerical suffix bits become conditionally unnecessary memory accesses once interval bounds prove they cannot alter the discrete decision.

The most important system implication is that **exactness need not imply reading every stored bit**.

---

# Remaining decisive external-validity gate

The largest remaining weakness is hardware diversity.

The next experiment should be a datacenter-GPU implementation of the simple FP16 8+8 pilot=1 path with:

- same-GPU dense vs ProofBits counterbalancing;
- Triton/CUDA fused implementation;
- batch sweep;
- head-only and integrated decode timing;
- Nsight DRAM-byte counters;
- formal matched-reference exactness audit.

Until this is measured, the paper should say **Apple-M1 accelerator results**, not general GPU speedup.
