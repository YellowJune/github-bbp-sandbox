# ProofBits Theory v2 — Decision-Certified Memory Access

This note supersedes the first draft for formal statements. It distinguishes
(i) exact real-arithmetic interval logic, (ii) the floating-point implementation
of the certificate, and (iii) actual bytes fetched by pilots and survivors.

## 1. Prefix intervals

Let the stored output-head row be \(W_i\in\mathbb F^d\) and the hidden state be
\(h\in\mathbb R^d\). After reading only a prefix of each stored representation,
the unread suffix induces an **exact value interval**

\[
W_{ij}\in[\ell_{ij},u_{ij}].
\]

No probability model is assumed.

For FP16 8+8 ProofBits, the prefix is exactly the high byte. It contains the
sign, all five exponent bits and the two most significant fraction bits. The
unread low byte enumerates the remaining eight fraction bits. A 256-entry LUT
*could* map a high byte to exact finite endpoints, but the final implementation
does not need to read such a LUT.

### Proposition 0 — LUT-free FP16 extremal suffix

Let \(s_w\in\{0,1\}\) be the weight sign bit contained in the high byte and
\(s_h=\mathbf 1[h_j<0]\). Among all 256 possible unread low bytes, the value
that maximizes the coordinate contribution \(h_jW_{ij}\) has

\[
\boxed{
\text{low-byte}=
\begin{cases}
\texttt{0xFF}, & s_w=s_h,\\
\texttt{0x00}, & s_w\ne s_h.
\end{cases}
}
\]

**Proof.** For positive FP16 numbers with fixed high byte, increasing the low
mantissa byte monotonically increases the represented value. For negative
numbers it monotonically decreases the represented numerical value (increases
its magnitude). If \(h_j\ge0\), maximize the weight value; if \(h_j<0\),
minimize it. The four sign cases reduce exactly to equality of \(s_w\) and
\(s_h\). For \(h_j=0\), either endpoint gives the same zero contribution.
\(\square\)

Thus the certificate endpoint can be constructed directly from the stored high
byte and the hidden-state sign, then bitcast as FP16. No per-weight endpoint
metadata or endpoint-LUT traffic is necessary.

For the finite-precision safety term below, define the maximum possible absolute
endpoint in row \(i\). For finite FP16, after stripping the sign bit, numerical
magnitude is monotone in the remaining exponent/fraction code. Since maximum
absolute completion always uses low byte `0xFF`, the implementation can reduce

\[
q_i=\max_j(\text{high}_{ij}\ \&\ \texttt{0x7F})
\]

as an integer and convert only the single row code

\[
(q_i\ll8)\,|\,\texttt{0xFF}
\]

to obtain \(M_i\). This avoids a second per-weight floating-point conversion.

## 2. One-accumulator upper bound

Define

\[
\boxed{U_i=\sum_j\max(h_j\ell_{ij},h_ju_{ij})}
\]

or equivalently

\[
U_i=\sum_j
\begin{cases}
h_ju_{ij},&h_j\ge0,\\
h_j\ell_{ij},&h_j<0.
\end{cases}
\]

### Proposition 1 — validity

For every completion of all unread suffixes,

\[
z_i=\sum_jh_jW_{ij}\le U_i.
\]

The proof is coordinatewise monotonicity under multiplication by the sign of
\(h_j\).

### Proposition 2 — same bound as midpoint-radius

With

\[
m_{ij}=(\ell_{ij}+u_{ij})/2,\qquad r_{ij}=(u_{ij}-\ell_{ij})/2,
\]

we have exactly

\[
\boxed{h_jm_{ij}+|h_j|r_{ij}=\max(h_j\ell_{ij},h_ju_{ij})}
\]

and hence

\[
\sum_jh_jm_{ij}+\sum_j|h_j|r_{ij}=U_i.
\]

Thus the upper-only form is not a looser approximation: it is the identical
interval bound expressed with one selected endpoint and one accumulation.

## 3. Exact argmax under exact arithmetic

Choose a nonempty pilot set \(P\), fetch its suffixes and calculate its exact
stored-format scores. Let

\[
B=\max_{p\in P}z_p,
\qquad
S=\{i:U_i\ge B\}.
\]

### Theorem 1 — no global maximizer is eliminated

Every global maximizer lies in \(S\). Indeed, if \(i^\star\) is a maximizer,
then \(z_{i^\star}\ge B\), while Proposition 1 gives
\(U_{i^\star}\ge z_{i^\star}\). Therefore

\[
U_{i^\star}\ge B.
\]

Reading exact suffixes for survivors and applying the dense tie-breaking rule
therefore reproduces the dense stored-format decision.

Pilot selection affects efficiency only. The current main implementation uses
the largest certified \(U_i\) values and empirically uses \(p=4\) pilots.

## 4. Exact top-k

For requested \(k\), choose at least \(k\) pilots and let \(B_k\) be the
\(k\)-th largest exact pilot score. Define

\[
S_k=\{i:U_i\ge B_k\}.
\]

Because the pilot set is a subset of the vocabulary,

\[
B_k\le z_{(k)},
\]

where \(z_{(k)}\) is the global k-th score. Every global top-k row satisfies

\[
U_i\ge z_i\ge z_{(k)}\ge B_k,
\]

so all global top-k rows are retained. Exact survivor evaluation recovers the
dense top-k logits and, with the same deterministic tie policy, the dense
selected top-k indices.

This certifies greedy and top-k selection. It does not alone certify arbitrary
full-softmax or nucleus/top-p sampling because those additionally depend on the
normalization/tail mass of all logits.

## 5. Correct byte accounting

Let \(S\) be the survivor set and \(P\) the pilot set. Pilot suffixes are
already fetched before pruning, so the rows whose low bytes are actually read
are

\[
\boxed{R=S\cup P.}
\]

Let

\[
f_R=|R|/V.
\]

Dense FP16 weight traffic is

\[
T_{dense}=16Vd\quad\text{bits}.
\]

FP16 8+8 ProofBits ideally reads

\[
T_{PB}=8Vd+8|R|d=8Vd(1+f_R).
\]

Therefore

\[
\boxed{\rho_{bytes}=\frac{2}{1+f_R}.}
\]

With a dense suffix fallback, suffix bytes are fetched at most once for all
rows, so weight-byte traffic is capped at

\[
T_{PB}\le16Vd.
\]

This is only a **weight-byte** guarantee. Control flow, top-k, compaction,
arithmetic, cache effects and synchronization can still make a naive kernel
slower in wall-clock time.

## 6. Floating-point certification

The previous sections are exact real-arithmetic statements. A practical GPU
implementation computes both the bound and the dense reference logits in
finite precision. A rounded-down computed bound must not be allowed to cause a
false elimination.

For a stated FP32-accumulated reference, define the standard unit roundoff

\[
u=2^{-24},
\qquad
\gamma_n=\frac{nu}{1-nu}.
\]

Let

\[
M_i=\max_j\max(|\ell_{ij}|,|u_{ij}|).
\]

Because

\[
\sum_j |h_jW_{ij}|\le \|h\|_1M_i,
\]

a conservative implementation can inflate the computed upper bound by a
rounding envelope proportional to

\[
\|h\|_1M_i.
\]

For example, if one implementation-specific error analysis bounds downward
error of \(\widehat U_i\) by \(\gamma_a\|h\|_1M_i\) and upward error of the
dense exact-score computation by \(\gamma_b\|h\|_1M_i\), then use

\[
\boxed{
U_i^{safe}=\widehat U_i+(\gamma_a+\gamma_b)\|h\|_1M_i.
}
\]

Rows can then be eliminated only when

\[
U_i^{safe}<B.
\]

The constants \(\gamma_a,\gamma_b\) are **implementation dependent**: they
must reflect the actual FMA/reduction order and numerical semantics of the GPU
kernel and the dense baseline. A loose sequential bound is safe but may retain
more rows; a fixed tree reduction can admit a tighter bound.

Importantly, \(M_i\) requires no second weight stream or per-weight LUT in the
final FP16 implementation: Proposition 0's signless high-byte magnitude code is
max-reduced during the same high-byte scan. The hidden-state quantity
\(\|h\|_1\) is computed once per query.

## 7. Representation-general form

Nothing in Theorems 1–2 requires FP16. The requirement is a prefix map

\[
\mathcal I_b(prefix)=[\ell,u]
\]

that exactly contains every suffix completion. The efficient prefix length
depends on the representation. Current experiments find FP16 particularly
clean because the 8-bit boundary already contains the full exponent. BF16 also
works with the same theory, but empirically requires a longer prefix for useful
intervals.

## 8. Claims allowed before GPU benchmarking

Mathematics + current CPU experiments support:

- lossless FP16 high/low byte-plane storage;
- exact interval derivation from the high byte;
- LUT-free extremal endpoint reconstruction from sign equality;
- deterministic argmax and top-k retention under the stated arithmetic model;
- identical upper-only and midpoint-radius interval bounds;
- conditional suffix necessity: eliminated rows' suffixes cannot change the
  certified decision;
- idealized weight-byte traffic \(2/(1+f_R)\);
- a finite-precision safety construction once the actual reduction semantics
  are specified.

They do **not** yet establish:

- 2x lm-head latency;
- any fixed end-to-end LLM speedup;
- measured DRAM traffic reduction on a GPU;
- equivalence to arbitrary vendor mixed-precision accumulation behavior.

Those are hardware claims and require a real GPU kernel benchmark plus DRAM
counters.
