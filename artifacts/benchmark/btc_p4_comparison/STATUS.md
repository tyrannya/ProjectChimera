# CURRENT

The aggregate over all nine P4 Stage-1 cells, for the P4 derivatives-and-positioning
question: does causal `derivatives_v1`, alone or combined with OHLCV14, add usable
information beyond the OHLCV14 information set?

**It is not the deciding artifact.** The preregistered P4 continuation decision was made
by the Stage-1 screen in `benchmark/btc_p4_stage1`, which reads only the `xgboost`
combined and control arms, on the exploratory blocks that were available. That screen did
not pass, so P4 stopped at Stage 1: there was no Stage 2 and no re-fit, and the
single-use P4-HOLD holdout was retired **unread** — never opened, scored or evaluated —
because publishing the Stage-1 result is what would have made a later reuse of it
adaptive. This directory is the full nine-cell picture behind that decision, not a second
decision.

**Derived evidence.** Regenerated from the nine frozen
`btc_p4_<information_set>_<model>` cells by `nn.p2b_compare`, under the
`btc-usdt-1h-gen1` research contract, from the committed research snapshot in
`data/research/`. It carries no checksum of its own — the cells do, under
`btc_p4_stage1_SHA256SUMS.txt` — and what pins it instead is
`tests/test_p4_evidence.py`, which asserts its six fold counts, that it declares itself
derived under checkpoint P4, and its integrity counters directly. See
`artifacts/README.md`.

Every cell was checked to have answered the same research question and scored the same
outer rows; every persisted row index was checked to be the row the fold plan selected,
by SHA-256 over the exact `int64` bytes; every scored row's timestamp, label and realised
return was checked against the snapshot itself; and every reported trading and
classification number was recomputed from the persisted predictions.

Its evidence is negative: across three models and two derivatives arms, the best result
was 2 of 4 folds improved, against a predeclared bar of three. The two combinations that
reach 2 both show a positive mean net-return delta while improving only half the folds;
the mean is not the finding.

What this is evidence *for* is narrow, and stating it narrowly is the point: the
`derivatives_v1` design, at this horizon, in the current BTC 1h/6h cost-aware setup, did
not clear its own preregistered bar. It is not evidence that derivatives or positioning
information is useless in general.

This replaces neither `artifacts/diagnostics/btc_regimes_v4` nor
`artifacts/benchmark/btc_p2a_comparison`, `btc_p2b_comparison`, `btc_p2c_comparison` nor
`btc_p3_comparison` — it answers a different research question. Like P2b, P2c and P3, P4
reuses already-observed outer folds, so it is adaptive research evidence rather than
pristine out-of-sample evidence. See `artifacts/README.md`.
