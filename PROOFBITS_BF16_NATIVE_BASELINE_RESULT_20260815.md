# ProofBits BF16 p10 vs Native MLX BF16 — 2026-08-15

This note closes the optimized-runtime gate for the current native-BF16 10+6 implementation.

## Setup

Model: `mlx-community/gemma-3-270m-bf16`

ProofBits path:
- original checkpoint BF16 values preserved exactly;
- densely packed 10-bit prefix + 6-bit suffix;
- positive lane-local packed extraction kernel;
- pilot=1 exact certificate;
- MLX transformer body and KV cache.

Native reference:
- MLX `model.lm_head` operating directly on BF16;
- native MLX argmax;
- same transformer body and KV cache.

Four AB/BA trajectories, 48 generated tokens each. Sequence equality is measured empirically because native MLX reduction semantics are not assumed identical to the formal matched reference.

## Result

All four 48-token native-MLX / ProofBits trajectories were identical: **192/192 generated tokens empirically equal**.

| Round | Native MLX total | ProofBits p10 total | Native/PB ratio | Native head | PB head | Native/PB head ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12.465 ms | 13.264 ms | 0.940× | 5.964 ms | 6.573 ms | 0.907× |
| 2 | 12.662 ms | 14.229 ms | 0.890× | 6.003 ms | 6.740 ms | 0.891× |
| 3 | 13.153 ms | 13.644 ms | 0.964× | 6.052 ms | 6.617 ms | 0.915× |
| 4 | 13.266 ms | 13.595 ms | 0.976× | 6.104 ms | 6.611 ms | 0.923× |

Summary:
- median native/PB integrated ratio: **0.952×**;
- mean native/PB integrated ratio: **0.942×**;
- median native/PB head ratio: **0.911×**.

Equivalently, the current ProofBits-BF16 p10 kernel is about **5% slower end-to-end** and about **10% slower in the head** than native MLX BF16 on this workload.

## Decision

The native-BF16 p10 branch remains scientifically useful:
- p=10 is the cross-model logical-traffic optimum among tested prefix widths;
- matched custom dense BF16 is accelerated (~1.18× head / ~1.10× integrated in the positive implementation);
- exact BF16 checkpoint semantics are preserved;
- the failed SIMD-cooperative variant shows unpack architecture is a genuine systems variable.

But **native-BF16 p10 is not a current optimized-runtime headline** on Apple M1 because native MLX BF16 remains faster.

The main practical systems result therefore remains the FP16 stored-head branch, where Gemma ProofBits beat direct Apple MPS and native MLX FP16 references and showed robust integrated speedups across prompts/runners.

A separate experiment now tests a full-model FP16 MLX runtime (body and head both FP16) to remove the earlier mixed-precision `BF16 body + FP16 decision head` concern without claiming an official FP16 Gemma checkpoint.
