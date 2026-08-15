# ProofBits Theory v3 Addendum — Exact Top-k and Nucleus Certification

This addendum extends `notes/proofbits_theory_v2.md`. The main efficient path remains FP16 8+8 upper-only certification for argmax/top-k. Nucleus/top-p is included as a correctness/generalization result, not as the main efficiency claim.

## 1. Exact top-k theorem

Let `P` be any pilot set with `|P| >= k`, and let `B_k` be the k-th largest exact FP16 score among pilots. With certified high-byte upper bounds `U_i >= z_i`, define

\[
S_k=\{i: U_i\ge B_k\}.
\]

Because pilots are a subset of all rows, their k-th score cannot exceed the global k-th score `z_(k)`:

\[
B_k\le z_{(k)}.
\]

Hence every true global top-k row satisfies

\[
U_i\ge z_i\ge z_{(k)}\ge B_k,
\]

and must survive. Exact low-byte refinement of `S_k union P`, followed by the dense tie policy, therefore reproduces the exact stored-FP16 top-k set and logits.

### Sampling corollary

Once exact top-k logits are recovered, any top-k sampling implementation using the same temperature, RNG stream, normalization over the selected k logits, and tie policy produces the same sampling distribution as dense FP16 top-k and can reproduce the same sampled tokens under a fixed RNG stream.

## 2. Exact top-p / nucleus certification

Unlike top-k, nucleus sampling depends on the normalization mass of every vocabulary row. Therefore upper bounds alone are insufficient. Let high-byte intervals yield exact score bounds

\[
L_i\le z_i\le U_i.
\]

After exact low-byte refinement of a row set `R`, define its exact exponentiated mass at temperature `T` using any common numerical shift `c`:

\[
E_R=\sum_{i\in R}\exp(z_i/T-c).
\]

For unrefined rows, define certified partition bounds

\[
Z_{\min}=E_R+\sum_{i\notin R}\exp(L_i/T-c),
\]

\[
Z_{\max}=E_R+\sum_{i\notin R}\exp(U_i/T-c).
\]

Suppose the first `k` exact refined rows have also been globally order-certified, i.e. no unrefined row has an upper bound exceeding the k-th refined score. Let

\[
A_j=\sum_{r=1}^{j}\exp(z_{(r)}/T-c).
\]

### Theorem — exact nucleus boundary

If

\[
\frac{A_{k-1}}{Z_{\min}} < p
\]

and

\[
\frac{A_k}{Z_{\max}} \ge p,
\]

then the dense FP16 nucleus boundary is exactly `k`, and the first `k` refined rows are exactly the dense nucleus set.

**Reason.** `Z_min` is the smallest possible denominator consistent with unread suffixes, so `A_(k-1)/Z_min` upper-bounds the possible cumulative probability before row k. `Z_max` is the largest possible denominator, so `A_k/Z_max` lower-bounds cumulative probability at row k. The two strict inequalities therefore certify that the threshold cannot be crossed before k and must be crossed by k. Global rank certification prevents an unrefined row from entering the prefix.

## 3. Current empirical implication

On Qwen2.5-0.5B, 48 WikiText hidden states, exact nucleus certification succeeded for every tested state at `(temperature, p)` in `{0.7,1.0} x {0.90,0.95}`. However, current 8+8 interval tail-mass bounds are frequently too loose for efficient top-p at temperature 1.0.

Observed mean low-byte refinement rows:

| Temperature | p | Mean refined rows | Mean fraction of 151,936 | Idealized head-byte reduction |
|---:|---:|---:|---:|---:|
| 0.7 | 0.90 | 36,500 | 24.0% | 1.613x |
| 0.7 | 0.95 | 47,629 | 31.3% | 1.523x |
| 1.0 | 0.90 | 94,228 | 62.0% | 1.234x |
| 1.0 | 0.95 | 108,339 | 71.3% | 1.168x |

These counts use a geometric refinement schedule and therefore upper-bound the minimal rows required by a more adaptive implementation.

### Claim discipline

The main systems claim should remain:

> exact decision-certified conditional low-byte fetching for argmax and top-k.

Nucleus certification is evidence that the interval framework generalizes to normalized decisions, but current top-p efficiency is not competitive enough to headline. A future sharper tail-partition bound or multi-prefix method may improve it.

## 4. Failure mode and fallback

The correctness failure mode of ProofBits is not approximate output: if the certificate is loose, more rows survive. With dense suffix fallback, the stored-FP16 result remains exact. The system can set a survivor threshold and switch to a dense low-byte pass before sparse-gather/control overhead dominates.

This distinction should be explicit in the paper: uncertainty costs performance, not model quality.
