# ProofBits — Latest Experimental Delta (2026-08-15)

This file records results newer than the main `PROOFBITS_RESULTS.md` audit. GPU wall-clock and DRAM-counter measurements remain missing and are not inferred from idealized byte traffic.

## Exact top-k — expanded 192-state WikiText test

Model: Qwen2.5-0.5B-Instruct, V=151,936. High byte read for all rows; exact low byte fetched for certified survivors and pilots.

All 192/192 states reproduced the exact dense FP16 top-k set for every k tested.

| k | Pilot rows | Mean refined rows | Median | p90 | p99 | Max | Mean fraction | Idealized head-byte reduction |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 34.02 | 15 | 60.9 | 334.55 | 363 | 0.0224% | 1.99955x |
| 5 | 20 | 146.90 | 71 | 277.1 | 1243.13 | 1349 | 0.0967% | 1.99807x |
| 10 | 40 | 279.80 | 147 | 516.4 | 2124.36 | 2341 | 0.1842% | 1.99632x |
| 50 | 200 | 1299.35 | 821 | 2600.6 | 8055.18 | 9315 | 0.8552% | 1.98304x |

Workflow run: 31862154632.

Interpretation: the efficient exact path is not restricted to greedy argmax. Exact top-k logits permit exact dense-equivalent top-k sampling under matched temperature/RNG semantics.

## Qwen2.5-1.5B scale-up

Model: Qwen2.5-1.5B-Instruct, V=151,936, d=1536, N=48 WikiText hidden states.

- exact: 48/48
- mean refined rows: 140.08
- median: 12
- p90: 283.5
- p99: 1301.15
- max: 1350
- mean refined fraction: 0.0922%
- idealized head-byte reduction: 1.99816x
- minimum sampled top1-top2 margin: 0.001299

Workflow run: 31862175921.

Interpretation: candidate collapse remains strong when hidden dimension/model size increases from the 0.5B test, but this is still compact-model evidence rather than a large-LLM universal claim.

## Exact top-p / nucleus certification

Model: Qwen2.5-0.5B-Instruct, N=48 WikiText hidden states.

Every tested state/settings pair was mathematically certified and reproduced the exact dense FP16 nucleus set. Efficiency, however, is substantially weaker because normalization couples all vocabulary logits.

| Temperature | p | Mean refined rows | Mean fraction | Idealized head-byte reduction |
|---:|---:|---:|---:|---:|
| 0.7 | 0.90 | 36,500 | 24.0% | 1.613x |
| 0.7 | 0.95 | 47,629 | 31.3% | 1.523x |
| 1.0 | 0.90 | 94,228 | 62.0% | 1.234x |
| 1.0 | 0.95 | 108,339 | 71.3% | 1.168x |

Workflow run: 31862281706.

Decision: **do not headline top-p.** Keep argmax/top-k as the main efficient system. Use top-p as a correctness/generalization theorem and an explicit negative/limitation result unless a substantially tighter tail-mass certificate is found.

## Current strongest scientific statement

The evidence now supports a narrow but strong principle:

> A lossless stored numerical representation can be treated as a progressively revealed proof object: for discrete decisions such as argmax/top-k, unread suffix bits need not be fetched once interval bounds certify that they cannot change the decision.

The strongest missing claim remains hardware-level performance. Near-2x numbers above are **idealized FP16 lm-head weight-byte reductions**, not measured GPU latency or whole-model decoding speedups.
