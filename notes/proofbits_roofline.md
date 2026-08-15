# ProofBits GPU Roofline / Deployment-Scope Note

This note is an analytical bound, **not a measured GPU benchmark**.

## 1. Batch-1 arithmetic intensity

For a single hidden state and an FP16 output head with `V*D` weights:

Dense FP16 GEMV approximately performs `2VD` FLOPs and reads `2VD` weight bytes, so its dominant weight arithmetic intensity is roughly

\[
I_{dense}\approx 1\ \mathrm{FLOP/byte}.
\]

The ProofBits upper-only high-byte pass performs one multiply-add-like contribution per weight while reading one high byte per weight:

\[
I_{PB,high}\approx 2\ \mathrm{FLOP/byte},
\]

plus small integer bit operations and output/reduction traffic. Candidate low-byte traffic is empirically <<1% of the full head in the main tested regimes.

NVIDIA's published H100 SXM figures are 67 TFLOP/s FP32 and 3.35 TB/s memory bandwidth, giving a nominal FP32 machine balance near

\[
67/3.35\approx20\ \mathrm{FLOP/byte}.
\]

Therefore both dense GEMV and the ProofBits high-byte pass are far below the nominal balance at batch 1 and should be bandwidth-dominated in an ideal coalesced implementation. This is **why** the ~2x weight-byte opportunity can plausibly translate into a substantial head-latency gain. It does not prove that it will: reduction, endpoint reconstruction, compaction, launch overhead, cache effects, and compiler codegen still matter.

Source used for the paper note: NVIDIA H100 product specification page (H100 SXM FP32=67 TFLOP/s, memory bandwidth=3.35 TB/s), checked 2026-08-15.

## 2. Why large batch is not the main target

For batch `B`, if the weight matrix is reused from memory/cache, idealized arithmetic intensity grows roughly with B. More importantly, dense FP16/BF16 output projection becomes a conventional GEMM and can exploit highly optimized Tensor Core paths.

ProofBits has a query-dependent extremal endpoint:

\[
U_i(h)=\sum_j\max(h_j w^-_{ij},h_j w^+_{ij}),
\]

so the chosen endpoint depends on each query's hidden-sign pattern. This prevents treating the high-byte certificate as one fixed ordinary weight matrix shared by every batch element without reformulating the computation.

Consequently the strongest deployment scope is:

- autoregressive decoding with batch 1 or small batches,
- latency-sensitive interactive serving,
- compact models where the huge-vocabulary output head is a first-order fraction of per-token cost,
- large-label / retrieval-style decision heads with similar memory-bound geometry.

Do **not** promise a universal speedup for large-batch throughput serving before GPU measurements.

## 3. Batch survivor-union evidence

The existing Qwen2.5-0.5B 256-query stress shows that low-byte row union itself remains sparse even at batch 128 (mean union 2.99% of vocabulary, idealized shared weight-byte ratio 1.942x). Thus the *suffix traffic* does not collapse quickly with batch. The unresolved problem at large batch is principally compute/control efficiency versus dense GEMM, not loss of certificate sparsity.

## 4. Hardware benchmark requirements

A publication-grade GPU result must report at minimum:

1. native dense FP16/BF16 output projection latency;
2. matched lossless byte-plane dense kernel latency;
3. ProofBits raw and finite-precision-safe latency;
4. batch sweep, especially B=1,2,4,8,16+;
5. HBM/DRAM bytes from Nsight Compute/CUPTI, not just logical traffic;
6. kernel launch count and occupancy;
7. survivor count / low-byte transaction count;
8. end-to-end decode latency on a head-dominant compact model such as a 262k-vocabulary Gemma-class model;
9. fallback frequency and latency tail (p50/p90/p99).

The decisive practical claim should be phrased only after these counters exist.
