# CURRENT

The authoritative result for the P2c information-set question: does causal classical
chart structure add usable information beyond OHLCV14?

**Exploratory and adaptive.** By the time P2c ran, these same four outer blocks had been
read by v4, P2a, P2b, the P2b ablation and the P2b regime description. P2c can generate
hypotheses; it cannot confirm one. Its evidence is negative in any case.

**Derived evidence.** Regenerated from the nine frozen
`btc_p2c_<information_set>_<model>` cells by `nn.p2b_compare`, under the
`btc-usdt-1h-gen1` research contract, from the committed research snapshot. It carries
no checksum of its own — the cells do — and what pins it instead is
`tests/test_p2b_evidence.py`, which asserts its six fold counts, its verdicts and its
integrity counters directly. See `artifacts/README.md`.

Every cell was checked to have answered the same research question and scored the same
outer rows; every persisted row index was checked to be the row the fold plan selected,
by SHA-256 over the exact `int64` bytes; every scored row's timestamp, label and
realised return was checked against the snapshot; and every reported trading and
classification number was recomputed from the persisted predictions.

Answers a different question than `btc_p2b_comparison` and replaces neither it nor
`btc_p2a_comparison` nor `diagnostics/btc_regimes_v4`.
