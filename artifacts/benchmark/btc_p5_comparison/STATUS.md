# CURRENT

The P5 aggregate: nine cells joined into one document — the parity proof, the
independent recomputation, and the per-fold deltas of each arm against the `ohlcv14`
control, for all three models.

**Derived evidence.** It declares `evidence_class: "derived"` in its own JSON,
`tools.freeze_evidence` refuses to hash it, and it is regenerated whenever the
aggregator improves. What pins it is `tests/test_p5_evidence.py`, which rebuilds it and
checks what it says.

**It is not the deciding artifact.** P5's rule reads exactly one cell pair —
`xgboost x ohlcv14_plus_mtf_v1` against `xgboost x ohlcv14` — and the record of that
decision is `benchmark/btc_p5_decision`. This document reports all six model-arm
comparisons because reporting them all is the commitment; none of the other five may
switch the answer.

**Adaptive research evidence.** Its four outer blocks had been read eight times before
P5 ran, so a positive result here would have needed confirmation these blocks cannot
supply. It is negative, which needs none.
