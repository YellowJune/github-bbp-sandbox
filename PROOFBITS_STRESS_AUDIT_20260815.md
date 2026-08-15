# ProofBits Large Integrated Stress Audit — 2026-08-15

This file freezes the larger cross-prompt, independent-runner Apple-M1 stress suite added after the main hardware audit.

## Experimental design

Model: `mlx-community/gemma-3-270m-bf16`

Matched decision semantics:
- both dense and ProofBits use the same output-head weights cast once to stored FP16 before timing;
- dense reference evaluates the complete stored-FP16 head with the custom matched accumulation path;
- ProofBits uses the exact high-byte certificate, pilot=1, and conditional low-byte refinement;
- both paths use the same MLX-LM transformer body, KV cache, token materialization, and token-feedback loop;
- output-head conversion / plane construction is outside timed generation;
- execution order alternates dense→ProofBits / ProofBits→dense across prompts;
- no ProofBits-only survivor diagnostics are evaluated in the timed path.

Workload per independent runner:
- 12 heterogeneous prompts;
- 64 generated tokens per prompt;
- 768 paired generated tokens / runner.

The prompt suite spans information theory, algebra, programming, databases, inference systems, networking, biology, statistics, creative writing, AI ethics, and causal experimental design.

Three independently provisioned GitHub-hosted `macos-15` Apple-M1 runners were used.

## Results

| Runner | Exact tokens | Median of prompt-median speedups | Minimum prompt-median speedup | Maximum prompt-median speedup | Pooled decode speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 768/768 | **1.319×** | 1.196× | 1.413× | **1.318×** |
| 2 | 768/768 | **1.236×** | 1.009× | 1.468× | **1.219×** |
| 3 | 768/768 | **1.261×** | 1.049× | 1.537× | **1.273×** |

Aggregate descriptive statistics:

- **2,304 / 2,304 paired generated tokens exact** under the matched stored-FP16 reference semantics;
- **36 / 36 runner × prompt pairs have median speedup > 1**;
- median of the three runner-level prompt-median speedups: **1.261×**;
- mean of the three runner-level prompt-median speedups: **1.272×**;
- median of the three pooled decode speedups: **1.273×**;
- mean of the three pooled decode speedups: **1.270×**;
- worst observed prompt-median ratio across all 36 pairs: **1.0087×**;
- best observed prompt-median ratio: **1.537×**.

The pooled ratio is computed as total dense decode wall time divided by total ProofBits decode wall time over all 12×64 generated tokens on each runner, rather than averaging per-token ratios.

## Tail-noise caveat

Although every prompt has a median latency win, a small number of runner/prompt pairs have mean-latency ratios slightly below 1 because of long-tail hosted-runner jitter (examples around 0.97–0.99×). Therefore the strongest robust statement is not “every individual timing statistic improves,” but:

> Across three independent hosted M1 runners and 12 heterogeneous prompts, all 2,304 matched generated decisions are exact; every prompt has a positive median latency improvement; and pooled end-to-end decode improves by roughly 1.22×–1.32× per runner, with a runner median of 1.273×.

## Relation to the smaller replication

An earlier 5-prompt ×32-token suite across four independent runners produced 640/640 exact tokens and a median-of-runner-median speedup of 1.253×. The present stress suite is larger and should be preferred for the Apple-M1 cross-prompt headline because it uses:

- more prompts (12 vs 5),
- longer trajectories (64 vs 32 generated tokens),
- 2,304 paired decisions in the new suite alone,
- three independently provisioned runners.

The earlier result remains useful as an independent replication at a different workload size.

## Context robustness already established separately

A stabilized Gemma context sweep at 16, 64, 256, 512, and 1,024 prompt tokens found:

- matched exactness at every context;
- decision-head speedup roughly **1.707×–1.872×**;
- integrated token speedup roughly **1.257×–1.390×**;
- dense head fraction near **50%** through 1k context.

At 1,024 tokens, the measured head fraction (~0.505) and head speedup (~1.725×) imply an Amdahl prediction near 1.269×, close to the observed 1.257× total speedup. This supports the interpretation that the integrated improvement is causally explained by decision-head acceleration.

## Defensible Apple-M1 practical claim after this suite

For a compact, very-large-vocabulary model where the output decision head is a first-order decode bottleneck, ProofBits can provide a reproducible end-to-end low-batch greedy-decode speedup while exactly preserving a specified stored-FP16 decision semantics.

The best current Apple-M1 practical evidence is:

- direct Apple-MPS decision-head baseline: ~**1.63×** Gemma head speedup;
- matched integrated stress suite: pooled runner-median ~**1.273×** end-to-end speedup;
- 2,304/2,304 exact paired decisions in the large stress suite;
- context sweep maintains ~**1.26×–1.39×** integrated gain through 1k prompt tokens.

## Still not established

- universal speedup across model families (Qwen has a known negative prompt case and much smaller cross-prompt gain);
- CPU speedup (the measured CPU path is slower);
- arbitrary native-library reduction equivalence;
- CUDA/Triton datacenter-GPU speedup or Nsight DRAM evidence;
- large-batch throughput superiority;
- efficient full top-p / softmax acceleration;
- packed ProofBits+ latency gains.

The next hardware gate is NVIDIA CUDA/Triton + hardware memory-traffic counters. A separate native-BF16 byte-plane kill test is also being run to determine whether the FP16 output-head test representation can be removed entirely for BF16 checkpoints.
