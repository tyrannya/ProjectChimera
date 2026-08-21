# P2b cell — xgboost on `smc_v1`

**Question:** does causal smc_v1, alone or combined with OHLCV14, add usable information beyond OHLCV14?

One information set, one model, 4 temporal folds.
Each row is a `64 x 39` window flattened to
`2496` values. Nothing is tuned.

**Research contract:** `btc-usdt-1h-gen1`, semantic identity
`sha256:dca6d0a891a257197a7c8aecec04fdba0a0b3009cfe93c5f3a397d458ab4a1de`.

**Sealed test block:** everything at or after `2025-08-27T23:00:00+00:00`. Not planned over, not fitted on,
not selected on, not scored. `sealed_test: false`.

**Feature spec:** smc `smc_v1` `sha256:3421312fc8d8687e...`

## Per-fold outer validation

| fold | outer period | samples | threshold | trades | net return | ann. Sharpe | max DD | exposure | macro F1 | dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.70 | 20 | +0.137677 | 2.1783 | 0.0473 | 0.0253 | 0.1741 | 0.4375 |
| 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.68 | 29 | -0.031267 | -0.5211 | 0.0838 | 0.0366 | 0.1350 | 0.4630 |
| 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.62 | 49 | -0.112438 | -1.6510 | 0.1293 | 0.0619 | 0.1273 | 0.4000 |
| 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.70 | 20 | +0.079851 | 1.6925 | 0.0306 | 0.0253 | 0.1131 | 0.5806 |

## Across the four temporal folds

- net return: mean +0.018456, std 0.111935, min -0.112438, median +0.024292, max +0.137677
- positive in **2 of 4** temporal folds

The statistical unit is the temporal period. These estimators are deterministic,
so repeating the run under another seed would copy this evidence, not add to it.
