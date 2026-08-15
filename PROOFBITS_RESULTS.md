# ProofBits Experimental Audit Index

Last consolidated: 2026-08-15

This file is an audit trail, not a paper draft. It records the strongest current
results, failed branches, exact reference semantics, and the one decisive
missing measurement (real GPU timing/DRAM counters).

## Locked main method

**ProofBits FP16 8+8, upper-only certificate**

Lossless storage:

- `high_byte[V,d]`: unconditional prefix plane,
- `low_byte[V,d]`: fetched only for pilots/survivors.

For each row, high-byte endpoints provide the exact finite interval
`[w^-_ij, w^+_ij]`. The one-accumulator bound is

\[
U_i=\sum_j\max(h_jw^-_{ij},h_jw^+_{ij}).
\]

Current default pilot count: `pilot_k=4`.

Actual low-byte rows read are `R = S union P`. Idealized FP16 weight-byte
reduction is therefore

\[
2/(1+|R|/V).
\]

Formal statements and floating-point caveats are in
`notes/proofbits_theory_v2.md`.

---

## Main exact upper-only cross-model result

Workflow: `.github/workflows/proofbits-fp16-upperonly.yml`
Run: `31860284074`

All measurements below use the FP16-rounded lm-head with FP32 accumulation as
the exact stored-format reference in the CPU experiment.

| Model | Domain | States | Exact | Mean low-byte rows (pilot=4) | p99 | Max | Idealized weight-byte reduction |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B-Instruct | AG News | 64 | 64/64 | 55.92 | 391.05 | 491 | 1.99926x |
| Qwen2.5-0.5B-Instruct | autoregressive math/code | 64 | 64/64 | 13.52 | 62.22 | 66 | 1.99982x |
| Qwen3-0.6B | AG News | 64 | 64/64 | 31.52 | 155.13 | 183 | 1.99959x |
| Qwen3-0.6B | autoregressive math/code | 64 | 64/64 | 21.83 | 97.33 | 118 | 1.99971x |
| SmolLM2-360M-Instruct | AG News | 64 | 64/64 | 78.55 | 187.74 | 237 | 1.99681x |
| SmolLM2-360M-Instruct | autoregressive math/code | 64 | 64/64 | 15.84 | 53.59 | 70 | 1.99936x |

Interpretation: the upper-only expression preserves the same interval bound as
the earlier midpoint+radius implementation while removing one accumulation
stream.

---

## Natural intermediate-position + exact decision-boundary stress

Workflow: `.github/workflows/proofbits-upperonly-final-stress.yml`
Run: `31860597698`
Job: `94953199400`

Model: Qwen2.5-0.5B-Instruct, `V=151,936`, `d=896`, `pilot_k=4`.

### Natural WikiText intermediate positions

- 256/256 exact.
- mean candidates: **40.48**.
- median: 29.
- p90: 83.
- p99: **455.25**.
- maximum: **612**.
- mean candidate fraction: `0.0002664`.
- idealized weight-byte reduction: **1.99947x**.
- minimum observed exact top1-top2 margin: **0.0022068**.

### Adversarial decision-boundary interpolation

Constructed 48 line segments between real hidden states with different winners,
then bisected to the first FP16 top-1 decision boundary.

- 48/48 exact.
- margins: `0` to about `1.907e-6`.
- mean candidates: **49.52**.
- median: 44.5.
- p90: 82.6.
- p99: 94.89.
- maximum: **101**.
- idealized weight-byte reduction: **1.99935x**.

Near-exact ties did not trigger a dense suffix collapse in this stress set.

---

## Conservative finite-precision-safe certificate stress

Workflow: `.github/workflows/proofbits-roundoff-safe.yml`
Run: `31860748548`
Job: `94953579806`

Model: Qwen2.5-0.5B-Instruct, `d=896`.

CPU test uses an intentionally loose safety envelope

\[
U_i^{safe}=\widehat U_i+2\gamma_{4d}\|h\|_1M_i,
\]

where `gamma_4d = 0.00021366869143188546` and
`M_i=max_j max(|w^-_ij|,|w^+_ij|)`.

### Natural 128 intermediate states

- all exact under safe certificate.
- raw mean candidates under same safe-selected pilots: 49.625.
- **safe mean candidates: 53.539**.
- safe median: 23.
- safe p90: 119.6.
- safe p99: 627.07.
- safe max: 663.
- safe mean candidate fraction: `0.000352379`.
- safe idealized weight-byte reduction: **1.99930x**.

### Boundary 32 states

- all exact under safe certificate.
- margin median: `9.5367e-7`; minimum: `0`.
- safe mean candidates: **52.06**.
- safe p99: 102.97.
- safe max: 107.
- safe idealized weight-byte reduction: **1.99931x**.

Important: this proves the *conservative CPU arithmetic model tested here* is
not an efficiency killer. The final GPU coefficient must be derived for the
actual compiled reduction/FMA order and the numerical semantics of the matched
dense reference.

---

## 1.5B scale-up falsification

Workflow: `.github/workflows/proofbits-upperonly-qwen15b.yml`
Run: `31860650094`
Job: `94953333730`

Model: Qwen2.5-1.5B-Instruct, `V=151,936`, `d=1536`, only 24+24 states.

| Domain | Exact | Mean candidates | p99 | Max | Idealized reduction |
|---|---:|---:|---:|---:|---:|
| AG News | 24/24 | 95.83 | 281.37 | 295 | 1.99874x |
| autoregressive math/code | 24/24 | 17.58 | 45.31 | 46 | 1.99977x |

This is positive scale-up evidence, not a large-model generalization claim.

---

## Exact top-k

Workflow: `.github/workflows/proofbits-fp16-topk.yml`
Run: `31860245597`
Job: `94952251968`

Qwen2.5-0.5B, 64 AG + 64 autoregressive states, tested `k=1,5,10,50`.

All settings:

- exact top-k set rate = 1.0,
- maximum top-k logit difference = 0.0.

Low-byte rows including pilots:

| k | AG mean | AG idealized reduction | autoregressive mean | autoregressive idealized reduction |
|---:|---:|---:|---:|---:|
| 1 | 50.86 | 1.99933x | 13.13 | 1.99983x |
| 5 | 181.38 | 1.99762x | 64.00 | 1.99916x |
| 10 | 312.00 | 1.99590x | 126.44 | 1.99834x |
| 50 | 1262.61 | 1.98352x | 753.78 | 1.99013x |

This supports exact greedy/top-k selection, not unrestricted full-softmax or
nucleus/top-p certification.

---

## Prefix ablations / representation generality

### FP16

Workflow: `.github/workflows/proofbits-fp16-prefix-sweep.yml`
Run: `31859882888`

Across Qwen2.5-0.5B, Qwen3-0.6B and SmolLM2-360M, an **8-bit prefix** was the
best tested idealized traffic point. A 9-bit prefix reduces survivors but the
unconditional extra bit/weight outweighs that benefit. This supports the simple
hardware layout `high byte -> certificate -> sparse low byte`.

### BF16

Workflow: `.github/workflows/proofbits-bf16-sweep.yml`
Run: `31860320837`
Job: `94952458058`

On Qwen2.5-0.5B, BF16 8-bit prefix is too loose because it leaves part of the
exponent unread. Best tested prefix was 10 bits:

- AG: 1.5886x idealized reduction,
- autoregressive: 1.5962x,
- exact in all tested states.

This supports representation-general theory but makes FP16 8+8 the cleaner
main system.

---

## INT8 historical branch — secondary only

INT8 experiments originally motivated progressive precision, but they are no
longer the main method because exact FP16 byte-prefix certification is cleaner
and avoids quantizer dependence.

Important results/failures:

- Qwen2.5 groupwise INT8 progressive reading achieved strong candidate collapse.
- Row-wise INT8 failed on SmolLM2 (roughly 90-99% candidates retained).
- SmolLM2 groupwise INT8 rescued the behavior, proving the row-wise failure was
  quantizer/outlier-sensitive rather than a failure of interval certification.

Keep INT8 only as an ablation/generalization section.

---

## Failed branches (must not be silently omitted)

1. **Low-rank/PCA exact MIPS**: even rank 128 required about 94.8% of vocabulary
   to be exact-refined; screening was not useful. Discarded.
2. **Naive one-sided 4+4 bit-plane interval**: about 51.4% of vocabulary needed
   rereading. Discarded.
3. **Residual-norm certificate sketches**: metadata cost plus loose L2 geometry
   gave only about 1.29x at the best tested AG point, worse than zero-metadata
   alternatives. Discarded.
4. **Row-wise INT8 universal claim**: falsified by SmolLM2. Discarded.
5. **BF16 8+8 universal byte-split claim**: falsified; BF16 needs a longer
   prefix. Discarded.
6. **Midpoint+radius main kernel**: not wrong, but superseded by the identical
   one-accumulator upper-only expression.

---

## Code / kernel status

### Final correctness-first kernel

`kernels/proofbits_triton_fp16_upperonly.py`

Contains:

- lossless FP16 high/low byte-plane packing,
- exact 256-entry endpoint LUT,
- one-accumulator high-byte upper pass,
- same-pass row endpoint max `M_i` reduction for finite-precision safety,
- caller-parameterized safe rounding inflation,
- exact low-byte survivor reconstruction by FP16 bitcast,
- pilot/survivor accounting using `S union P`,
- dense suffix fallback,
- static lossless/interval/identity/safety self-tests.

### GPU benchmark harness

`benchmarks/benchmark_proofbits_fp16_gpu.py`

Now benchmarks:

- native PyTorch FP16 GEMV,
- PyTorch FP32 over FP16-rounded weights,
- matched Triton dense byte-plane reference,
- raw upper-only ProofBits,
- conservative roundoff-safe ProofBits,
- `S union P` traffic,
- exact argmax checks.

Manual workflow:

`.github/workflows/proofbits-gpu-benchmark.yml`

requires labels:

`[self-hosted, linux, x64, gpu]`.

---

# Decisive missing measurement

**No real GPU ProofBits wall-clock or DRAM counter result has been obtained.**

Therefore the repository currently supports:

- a strong exactness/certification claim,
- strong cross-model and adversarial candidate-collapse evidence,
- an approximately 2x **idealized lm-head weight-byte traffic ceiling** in the
  tested FP16 settings,
- a concrete Triton implementation path.

It does **not** yet support:

- `2x lm-head latency`,
- a fixed whole-model decoding speedup,
- measured DRAM transaction reduction,
- a production-grade fused kernel claim.

The next publication-critical experiment is the self-hosted CUDA benchmark plus
Nsight Compute/CUPTI DRAM counters against a strong native FP16 lm-head baseline.
