# HISTORICAL

An independent re-run of the `ohlcv14_plus_smc_v1` x `xgboost` cell, kept as a determinism
check rather than as a result. It reproduces the canonical cell exactly: all four folds'
outer reports, all four selected thresholds, and all 18,939 per-sample predictions are
identical.

Read `benchmark/btc_p2b_ohlcv14_plus_smc_v1_xgboost` for the cell itself, and
`benchmark/btc_p2b_comparison` for the result.
