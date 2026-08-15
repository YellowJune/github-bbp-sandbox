# ProofBits Result Snapshot — 2026-08-15

## Full-model FP16 MLX runtime — Gemma 3 270M

Source checkpoint: `mlx-community/gemma-3-270m-bf16`, then the *entire MLX model* is converted with `model.set_dtype(float16)` before warmup and timing. Verified runtime dtypes: output head `mlx.core.float16`; transformer body output `mlx.core.float16`.

Five prompt families × 32 generated tokens = 160 tokens per comparison.

### Matched exact FP16 reference vs ProofBits

- generated tokens matched exactly: **160/160**
- median integrated speedup: **1.2803×**
- mean integrated speedup: **1.2567×**
- per-prompt speedup range: **1.1557×–1.3007×**
- per-prompt speedups: **1.2803×, 1.2908×, 1.3007×, 1.2560×, 1.1557×**

### Native MLX FP16 lm_head vs ProofBits

This is an optimized-runtime comparison, not the formal exact-semantics reference. Sequence equality is measured empirically.

- empirical generated-token equality: **160/160**
- median native-MLX-over-ProofBits speedup: **1.2798×**
- mean speedup: **1.2630×**
- per-prompt range: **1.2001×–1.3413×**
- per-prompt speedups: **1.2001×, 1.3413×, 1.2088×, 1.2798×, 1.2851×**

So, on this fully FP16 autoregressive runtime, ProofBits is about **28% faster in median end-to-end token latency** than both the matched custom FP16 reference and the native MLX FP16 head path, while preserving all 160 generated tokens in the measured trajectories.

## Earlier Gemma FP16-head integrated replication

With a BF16 body plus FP16 test head, the 12-prompt × 64-token stress suite was run on three independent hosted Apple-M1 runners:

- **2,304/2,304 generated tokens exact** against the matched FP16 reference;
- runner pooled end-to-end speedups: **1.318× / 1.219× / 1.273×**;
- median across runners: **1.273×**;
- mean across runners: **1.270×**.

A stabilized context sweep from 16 to 1024 tokens retained exactness throughout. Head-only speedup was about **1.71×–1.87×**, while integrated token-latency speedup remained about **1.26×–1.39×**.

## Qwen2.5-0.5B FP16-head result

Five prompt families × 32 generated tokens:

- **160/160 tokens exact**;
- median integrated speedup: **1.064×**;
- mean: **1.054×**;
- range: **0.958×–1.117×**.

This is prompt-dependent and materially weaker than Gemma, so it should not be used as the headline system result.

## Native BF16 extension

### Byte-aligned 8+8

Rejected. On Qwen, all 151,936 rows survived the 8-bit prefix certificate; on Gemma approximately 99.96% survived. The BF16 byte boundary cuts the exponent field.

### Non-byte-aligned 10+6

Both Qwen and Gemma selected `p=10` as the best logical-traffic point among p=8..14:

- Qwen ideal logical traffic reduction: about **1.577×**;
- Gemma: about **1.596×**.

A real packed Gemma BF16 10+6 Metal implementation using lane-local unpack gave:

- **192/192 matched generated tokens exact**;
- matched-reference head speedup: **1.183×**;
- matched-reference integrated speedup: **1.102×**;
- mean survivor count: **86.81 / 262,144 = 0.0331%**.

However, against the optimized native MLX BF16 `lm_head + argmax` path, the same lane-local ProofBits kernel was slower:

- **192/192 empirical generated-token equality**;
- median total speed ratio native-MLX / ProofBits: **0.9519×** (ProofBits about 5% slower);
- median head speed ratio: **0.9110×** (ProofBits about 9.8% slower).

The SIMD-cooperative 10-bit unpack variant was also rejected: despite **192/192 exact**, it achieved only **0.531× head** and **0.705× integrated** speed ratios versus the matched dense BF16 reference.

## Current strongest practical result

The strongest result is therefore the **fully FP16 Gemma autoregressive runtime**:

> **160/160 matched generated tokens exact, median ~1.280× end-to-end speedup; and 160/160 empirical sequence equality with native MLX FP16 while retaining ~1.280× median end-to-end speedup.**

The broader independent-runner Gemma FP16-head stress suite independently supports a roughly **1.27×** integrated speedup with 2,304/2,304 exact generated tokens.