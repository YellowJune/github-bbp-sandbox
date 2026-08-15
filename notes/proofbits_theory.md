# ProofBits: Decision-Certified Memory Access

## 1. Problem

Let a frozen output head contain finite stored weights

\[
W\in\mathbb F^{V\times d},\qquad z_i(h)=\sum_{j=1}^d h_jW_{ij},
\]

where \(\mathbb F\) is a finite numerical storage format such as IEEE FP16.
Conventional inference reads the complete representation of every \(W_{ij}\)
before computing \(\arg\max_i z_i\).

ProofBits instead partitions each stored representation into a prefix and a
suffix.  The prefix must identify an **exact numerical interval** containing
every value consistent with the unread suffix.

For row \(i\), coordinate \(j\), after reading the prefix define

\[
W_{ij}\in [\ell_{ij},u_{ij}].
\]

No distributional or probabilistic assumption is made.

---

## 2. Exact upper bound from a representation prefix

For a fixed hidden state \(h\), define

\[
\boxed{
U_i(h)=\sum_{j=1}^{d}
\max\{h_j\ell_{ij},h_ju_{ij}\}.
}
\]

Equivalently,

\[
U_i(h)=\sum_j
\begin{cases}
h_j u_{ij}, & h_j\ge 0,\\
h_j \ell_{ij}, & h_j<0.
\end{cases}
\]

### Proposition 1 — Prefix upper-bound validity

For every completion of every unread suffix,

\[
\boxed{z_i(h)\le U_i(h).}
\]

**Proof.**  For each coordinate, \(W_{ij}\in[\ell_{ij},u_{ij}]\).  If
\(h_j\ge0\), multiplication preserves order and
\(h_jW_{ij}\le h_ju_{ij}\).  If \(h_j<0\), multiplication reverses order and
\(h_jW_{ij}\le h_j\ell_{ij}\).  Sum over coordinates. \(\square\)

---

## 3. Upper-only equals midpoint-radius certification

Let

\[
m_{ij}=\frac{\ell_{ij}+u_{ij}}2,
\qquad
r_{ij}=\frac{u_{ij}-\ell_{ij}}2\ge0.
\]

The common midpoint-radius bound is

\[
c_i=\sum_jh_jm_{ij},
\qquad
e_i=\sum_j|h_j|r_{ij},
\qquad z_i\le c_i+e_i.
\]

### Proposition 2 — Exact bound identity

Coordinatewise,

\[
\boxed{
h_jm_{ij}+|h_j|r_{ij}
=\max\{h_j\ell_{ij},h_ju_{ij}\}.
}
\]

Therefore

\[
\boxed{c_i+e_i=U_i.}
\]

**Proof.**  If \(h_j\ge0\),
\(h_jm+|h_j|r=h_j(m+r)=h_ju\).  If \(h_j<0\),
\(h_jm+|h_j|r=h_j(m-r)=h_j\ell\). \(\square\)

**Systems consequence.**  Computing \(c_i\) and \(e_i\) separately performs
two accumulation streams.  The upper-only form computes the identical bound
with one sign-select and one accumulation stream.

---

## 4. Exact argmax certification

Read all prefixes and obtain \(U_i\).  Choose any nonempty pilot set
\(P\subseteq\{1,\dots,V\}\), read the full representations of pilots, and
compute their exact stored-format scores.  Let

\[
B=\max_{p\in P} z_p.
\]

Define survivors

\[
S=\{i:U_i\ge B\}.
\]

### Theorem 1 — Exact argmax preservation

\[
\boxed{
\arg\max_i z_i
\in
\arg\max_{i\in S}z_i.
}
\]

If the dense stored-format argmax is unique, the ProofBits result is identical.
Under ties, applying the same deterministic tie-breaking rule to the exact
survivor scores reproduces the dense decision provided all tied maximizers are
retained; every maximizer is retained by the proof below.

**Proof.**  Let \(i^\star\) be any global maximizer.  Because \(B\) is the
score of an actually existing row,

\[
z_{i^\star}\ge B.
\]

By Proposition 1,

\[
U_{i^\star}\ge z_{i^\star}\ge B,
\]

hence \(i^\star\in S\). \(\square\)

The pilot selection strategy affects efficiency only, not correctness.  In the
current FP16 implementation, pilots are rows with the largest \(U_i\).

---

## 5. Exact top-k certification

For desired \(k\), choose a pilot set \(P\) containing at least \(k\) rows,
read their exact scores, and let \(B_k\) be the \(k\)-th largest exact pilot
score.  Define

\[
S_k=\{i:U_i\ge B_k\}.
\]

### Theorem 2 — Exact top-k set preservation

Every row in the global exact top-k set lies in \(S_k\).  Consequently,
computing exact scores only for \(S_k\) is sufficient to recover the dense
stored-format top-k set and its exact logits.

**Proof.**  The pilot set is a subset of the vocabulary, so its \(k\)-th exact
score cannot exceed the global \(k\)-th score:

\[
B_k\le z_{(k)}.
\]

For every global top-k row \(i\),

\[
U_i\ge z_i\ge z_{(k)}\ge B_k,
\]

hence \(i\in S_k\). \(\square\)

This directly supports exact greedy decoding and exact top-k selection/logits.
It does **not**, by itself, certify unrestricted full-softmax probabilities or
arbitrary nucleus/top-p sampling, because those require controlling the
contribution of all excluded logits to the normalization/tail mass.

---

## 6. FP16 byte-plane specialization

IEEE binary16 consists of one sign bit, five exponent bits and ten fraction
bits.  The first byte of the stored 16-bit word fixes the sign, the complete
exponent, and the two most significant fraction bits.  Varying only the unread
low byte therefore enumerates exactly 256 possible completions within one
finite interval.

Store losslessly as

- `high_byte[V,d]`,
- `low_byte[V,d]`.

A 256-entry lookup table maps each finite high-byte value to exact interval
endpoints \((\ell,u)\).  No model-specific certificate metadata is needed.

The current main algorithm is therefore:

1. fetch all high bytes;
2. compute one \(U_i\) per vocabulary row;
3. select \(p\) high-\(U\) pilots (empirically \(p=4\) is a strong default);
4. fetch pilot low bytes and obtain exact \(B\);
5. retain \(S=\{i:U_i\ge B\}\);
6. fetch low bytes only for \(S\);
7. reconstruct exact FP16 weights and reduce exact survivor scores.

---

## 7. Memory traffic

Let \(f=|S|/V\) be the fraction of rows whose low bytes are fetched.  Ignoring
small index/control traffic, dense FP16 reads

\[
T_{dense}=16Vd\ \text{bits}.
\]

FP16 8+8 ProofBits reads

\[
T_{PB}=8Vd+8|S|d=8Vd(1+f).
\]

Hence the idealized weight-byte reduction is

\[
\boxed{
\rho_{bytes}=\frac{T_{dense}}{T_{PB}}
=\frac{2}{1+f}.
}
\]

As \(f\to0\), \(\rho_{bytes}\to2\).

### Theorem 3 — Dense-byte fallback cap

If implementation falls back to a dense low-byte pass whenever survivor
fraction exceeds a threshold, suffix traffic is capped at \(8Vd\).  Thus

\[
\boxed{T_{PB}\le16Vd=T_{dense}}
\]

for **weight bytes**.

This is not a wall-clock theorem: upper-bound arithmetic, reduction, top-k,
compaction, synchronization and cache behavior can still make a naive kernel
slower.  Real GPU timing and DRAM counters remain mandatory.

---

## 8. Representation-general form

ProofBits does not require FP16.  It requires only a storage prefix whose unread
suffix induces a computable exact interval.  For a format \(\mathbb F\) with a
chosen prefix length \(b\), define interval map

\[
\mathcal I_b(prefix)=[\ell,u].
\]

Propositions 1–2 and Theorems 1–3 then apply unchanged.  The efficient prefix
length is representation-dependent.  Current experiments show that FP16 is
especially hardware-friendly because an 8-bit byte boundary already includes
the entire exponent, whereas BF16 needs more than eight prefix bits for tight
intervals.

---

## 9. What is and is not claimed

Claimed by the mathematics:

- deterministic stored-format argmax certification;
- deterministic stored-format top-k certification;
- exact lossless FP16 byte-plane representation;
- suffix bytes of eliminated rows are unnecessary for the certified decision;
- upper-only and midpoint-radius produce the same interval upper bound.

Not established without GPU experiments:

- 2x lm-head latency;
- a particular whole-model decoding speedup;
- a particular DRAM transaction reduction on a real accelerator;
- equivalence to every vendor's native mixed-precision accumulation convention.

The current experimental reference is the stored FP16 lm-head evaluated with
FP32 accumulation.  A publication implementation must state and match the
chosen dense baseline's numerical semantics exactly.
