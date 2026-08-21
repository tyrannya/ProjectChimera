# CURRENT

The authoritative result for the P2b information-set question: does causal market
structure add usable information beyond the OHLCV14 information set?

**Derived evidence.** Regenerated from the nine frozen
`btc_p2b_<information_set>_<model>` cells by `nn.p2b_compare`, under the
`btc-usdt-1h-gen1` research contract, from the committed research snapshot in
`data/research/`. It carries no checksum of its own — the cells do — and what pins it
instead is `tests/test_p2b_evidence.py`, which asserts its six fold counts, its
verdicts and its integrity counters directly. See `artifacts/README.md`.

Every cell was checked to have answered the same research question and scored the same
outer rows; every persisted row index was checked to be the row the fold plan selected,
by SHA-256 over the exact `int64` bytes; every scored row's timestamp, label and
realised return was checked against the snapshot itself; and every reported trading and
classification number was recomputed from the persisted predictions.

This replaces neither `artifacts/diagnostics/btc_regimes_v4` nor
`artifacts/benchmark/btc_p2a_comparison` — it answers a different research question.
P2b reuses already-observed outer folds, so it is adaptive research evidence rather
than pristine out-of-sample evidence. See `artifacts/README.md`.
