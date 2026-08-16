# CLIF MVP — program-local online learner

This deliberately small MVP tests one claim only: a fuzzer can start with no training dataset, learn mutation-family value from executions of one target program, and discard that learned state when the target changes.

## Scope
- synthetic, local toy targets only
- no pretrained model and no external dataset
- UCB1-style online learner
- random mutation baseline
- security-flow observation is explicit instrumentation from the toy target
- `CSA = security_delta / input_distance`

## Run
```bash
python clif_mvp/mvp.py --trials 80 --budget 120 --out artifacts/clif_mvp_results.json
python -m unittest clif_mvp/test_mvp.py -v
```

The MVP is intentionally not evidence for a real-world vulnerability-discovery claim. It only validates the learning loop and experiment plumbing before replacing toy instrumentation with real local program telemetry.
