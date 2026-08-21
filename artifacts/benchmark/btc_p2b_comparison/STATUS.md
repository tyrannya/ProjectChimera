# CURRENT

The authoritative result for the P2b information-set question: does causal market
structure add usable information beyond the OHLCV14 information set?

Joined from the nine frozen `btc_p2b_<information_set>_<model>` cells, under the
`btc-usdt-1h-gen1` research contract, from the committed research snapshot in
`data/research/`. Every cell was checked to have scored the same outer rows, every
reported number was recomputed from the persisted predictions, and every prediction
was checked against the snapshot itself.

This replaces neither `artifacts/diagnostics/btc_regimes_v4` nor
`artifacts/benchmark/btc_p2a_comparison` — it answers a different research question.
P2b reuses already-observed outer folds, so it is adaptive research evidence rather
than pristine out-of-sample evidence. See `artifacts/README.md`.
