# ProofBits — Extreme-Value Geometry of Candidate Collapse

This note provides a stylized explanation, **not a theorem that trained LLM logits are iid Gaussian**.

## 1. Simplified model

Suppose a decision head has `V` exact scores

\[
z_i \overset{iid}{\sim} \mathcal N(0,\sigma^2)
\]

and a prefix certificate produces upper bounds with a roughly fixed additive uncertainty scale `delta` near the winning region:

\[
z_i\le U_i\lesssim z_i+\delta.
\]

A pilot threshold close to the exact maximum retains approximately rows whose exact scores lie within `delta` of the maximum:

\[
S_\delta \approx \{i:z_i\ge M_V-\delta\},\qquad M_V=\max_i z_i.
\]

For Gaussian extremes,

\[
M_V/\sigma\approx b_V\approx \sqrt{2\log V}
\]

up to the usual lower-order `log log V` correction.

## 2. Approximate near-maximum population

Write `a=delta/sigma`. Since `V * barPhi(b_V)` is order one, the expected number of scores above `sigma(b_V-a)` is approximately the Gaussian tail ratio

\[
E|S_\delta|\approx
\frac{\bar\Phi(b_V-a)}{\bar\Phi(b_V)}.
\]

Using the Mills-ratio asymptotic,

\[
\frac{\bar\Phi(b-a)}{\bar\Phi(b)}
\approx
\frac{b}{b-a}
\exp\left(ab-\frac{a^2}{2}\right).
\]

Therefore, for fixed normalized uncertainty `a`,

\[
\boxed{
E|S_\delta|
=\exp\big(O(\sqrt{\log V})\big)
=V^{o(1)}.
}
\]

So the *absolute* number of near-max competitors can grow while the *fraction*

\[
|S|/V
\]

shrinks strongly with vocabulary size. This is exactly the regime in which conditional suffix fetching becomes attractive: high-byte work remains dense, but expensive suffix traffic can be sublinear in the number of alternatives.

## 3. Connection to current evidence

The empirical evidence is consistent with, but does not prove, this stylized picture:

- Qwen natural queries retain tens to low hundreds of rows out of ~152k.
- Gemma retains tens to low hundreds out of ~262k.
- In one controlled experiment using nested random subsets of the **same trained Gemma head**, vocabulary size grew 32x (8,192 -> 262,144) while mean survivor count grew only ~3.9x (34.9 -> 136.0); a descriptive log-log fit gave exponent ~0.396.
- Query-geometry controls show the phenomenon is not exclusive to model-generated hidden states: even isotropic norm-matched Gaussian queries against the trained Qwen head retained only ~0.67% of vocabulary on average, while natural queries retained ~0.065%. Thus learned query/head alignment sharpens the effect by about one order of magnitude, but high-dimensional extreme-value competition appears to provide a substantial baseline effect.

## 4. What must NOT be claimed

Do not claim:

1. trained LLM logits are iid Gaussian;
2. `0.396` is a universal scaling exponent;
3. the simplified fixed-delta model exactly describes FP16 prefix intervals;
4. sublinear survivor growth alone implies GPU speedup.

The useful paper claim is narrower:

> Extreme-value concentration supplies a plausible mechanism for why exact decision certification can eliminate almost all suffix reads even as the number of alternatives becomes very large; learned model geometry further tightens the certificate empirically.

## 5. Stronger future theory target

A more realistic theorem should replace constant `delta` by row-dependent certificate errors

\[
e_i(h)=U_i(h)-z_i(h)\ge0
\]

and bound

\[
E\left[\sum_i 1\{z_i+e_i\ge B\}\right]
\]

under assumptions on the joint tail of `(z_i,e_i)` or on sub-Gaussian weight/query coordinates. The pilot-1 lower bound `B=z_{argmax U}` creates dependence and should be handled explicitly rather than approximated by `M_V-delta` in a final theorem.
