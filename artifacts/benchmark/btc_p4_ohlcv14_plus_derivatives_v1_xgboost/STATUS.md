# CURRENT

One P4 Stage-1 cell: the `xgboost` model on the `ohlcv14_plus_derivatives_v1` information set, over four temporal
outer folds, under the `btc-usdt-1h-gen1` research contract.

A source run for `benchmark/btc_p4_comparison`, which is where the aggregate over all nine
cells lives. This directory on its own is one arm of a three-arm comparison and says
nothing about the question by itself.

It is also one half of the pair the deciding Stage-1 screen read: `benchmark/btc_p4_stage1`
compares the `ohlcv14_plus_derivatives_v1` arm against the `ohlcv14` control on `xgboost`,
and names this artifact in its own `provenance`. That screen is a screening rule on
already-burned exploratory blocks, not a research result, and it did not pass.
