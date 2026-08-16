# CLIF HTTP MVP v0.2

A deliberately small, isolated proof-of-concept for **program-local online learning of security-flow-relevant mutation operators**.

## What changed from v0.1

v0.1 used a synthetic probabilistic environment. v0.2 sends real HTTP requests to a local `ThreadingHTTPServer`. Each request is processed by deterministic toy program logic, and instrumentation exposes only source/sink labels through `X-CLIF-Flow` metadata.

No external training set, pretrained model, or cross-program state reuse is used. A new UCB learner is initialized for each program/trial.

## Toy programs

- `users`: deliberate cross-user disclosure when a scope mutation crosses an ownership boundary.
- `reports`: deliberate ownership-filter bypass under a debug/page mutation family.
- `tokens`: deliberate role-confusion metadata disclosure for guest tokens.

All records are non-sensitive toy markers and the server binds to `127.0.0.1` on an ephemeral port.

## Run

```bash
python -m unittest -v clif_mvp.test_http_mvp
python clif_mvp/http_mvp.py --trials 40 --budget 80 --seed 20260816 \
  --out artifacts/clif_http_mvp_results.json
```

## MVP reward

For one mutation operator step, input distance is fixed to 1. A new forbidden source→sink signature receives reward 1.0; a repeated forbidden flow receives 0.2; otherwise 0. This is only a finite-difference analogue for the research MVP, not the final CLIF metric.

## Interpretation

This project tests one narrow claim only: **without prior training data, can an online learner infer which mutation families are more likely to produce security-relevant information-flow events in a previously unseen toy program?**
