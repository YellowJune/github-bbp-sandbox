# ProofBits Final Fast Path — Pilot-1 Upper-Only FP16

This note records the current main system design after falsifying several more complex alternatives.

## Main path

Stored FP16 output weights are losslessly laid out as two byte planes:

- `high[V,D]` = upper 8 bits of each FP16 word
- `low[V,D]` = lower 8 bits

No weight information is discarded and total weight storage remains 16 bits/weight.

For query hidden state `h`, the high byte defines an exact interval `[w^-_ij,w^+_ij]`. The high-byte kernel computes one certified upper score per vocabulary row:

\[
U_i=\sum_j \max(h_j w^-_{ij},h_j w^+_{ij}).
\]

The extremal suffix is reconstructible without a LUT:

\[
\mathrm{suffix}_{ij}=\begin{cases}
0xFF,&\operatorname{sign}(h_j)=\operatorname{sign}(w_{ij}),\\
0x00,&\text{otherwise}.
\end{cases}
\]

### Pilot-1 lower bound

1. `p = argmax_i U_i`.
2. Fetch low bytes for row `p` only and compute its exact stored-FP16 score `B=z_p`.
3. Survivors are
   \[
   S=\{i:U_i\ge B\}.
   \]
4. Fetch low bytes only for `S` (pilot is already in `S` because `U_p >= z_p=B`).
5. Exact-refine survivor scores and return the maximum using the matched dense tie policy.

Correctness does **not** require the pilot to be the true winner. The true winner `i*` satisfies

\[
U_{i^*}\ge z_{i^*}\ge z_p=B,
\]

so it always survives.

## Why pilot-1 is now the default

A 64-state comparison showed that pilot-1 preserves essentially the same idealized byte reduction as pilot-4 while replacing a global top-4 selection with a single argmax reduction.

### Qwen2.5-0.5B, V=151,936

| pilot k | exact | mean survivor rows | survivor fraction | idealized FP16 head-byte reduction |
|---:|---:|---:|---:|---:|
| 1 | 64/64 | 106.86 | 0.0703% | 1.99859x |
| 4 | 64/64 | 78.92 | 0.0519% | 1.99896x |

### Gemma-3-270M mirror, V=262,144

| pilot k | exact | mean survivor rows | survivor fraction | idealized FP16 head-byte reduction |
|---:|---:|---:|---:|---:|
| 1 | 64/64 | 147.56 | 0.0563% | 1.99887x |
| 4 | 64/64 | 135.95 | 0.0519% | 1.99896x |

The byte advantage of pilot-4 is negligible compared with its extra selection/control complexity.

## Falsified alternatives

### Pilot-free lower+upper

Using `B=max_i L_i` is exact and removes pilot refinement, but requires both lower and upper accumulations and leaves substantially more survivors:

- Qwen0.5: mean 1,429.8 survivors, 0.941%, 1.98135x idealized bytes.
- Gemma270M: mean 1,951.9 survivors, 0.745%, 1.98522x.

It is rejected as the main fast path because a single pilot row costs only `D` low-byte reads while a second full score-bound accumulation costs `V*D` arithmetic.

### Native E5M2 + row/block radius

FP16 high-byte raw codes are numerically identical to finite FP8 E5M2 codes, but compressing suffix uncertainty into row/block symmetric radii destroys certificate tightness. Qwen block-radius experiments retained 88.5–100% of vocabulary even with 64 blocks and were dense-or-worse after metadata. Rejected.

### Top-p as main path

Exact nucleus certification exists but current partition-function interval bounds require 24–71% of rows depending on temperature/p. Keep as theorem/generality result, not headline fast path.

## Intended GPU pipeline

The publication-oriented pipeline should be reduced to:

```text
lossless FP16 byte planes
    ↓
high-byte certified upper kernel
    ↓
argmax(U) reduction
    ↓
exact low-byte refinement of one pilot row
    ↓
threshold + compact U >= B
    ↓
sparse low-byte exact refinement
    ↓
final argmax / top-k
```

The high-byte pass reads approximately half the dense FP16 weight bytes. The pilot read is negligible. The threshold scan reads the `U[V]` score vector, not another weight plane.

## Claim boundary

Until a CUDA GPU benchmark and DRAM counters exist, do **not** convert the ~1.999x idealized weight-byte result into a latency claim. The current evidence supports exactness and conditional memory-access opportunity, not measured device speedup.
