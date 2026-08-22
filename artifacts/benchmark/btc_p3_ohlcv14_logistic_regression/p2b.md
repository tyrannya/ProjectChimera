# P3 cell — logistic_regression on `ohlcv14`

**Question:** does causal microstructure_v1, alone or combined with OHLCV14, add usable information beyond OHLCV14?

One information set, one model, 4 temporal folds.
Each row is a `64 x 14` window flattened to
`896` values. Nothing is tuned.

**Research contract:** `btc-usdt-1h-gen1`, semantic identity
`sha256:dca6d0a891a257197a7c8aecec04fdba0a0b3009cfe93c5f3a397d458ab4a1de`.

**Sealed test block:** everything at or after `2025-08-27T23:00:00+00:00`. Not planned over, not fitted on,
not selected on, not scored. `sealed_test: false`.

**Feature spec:** smc `smc_v1` `sha256:3421312fc8d8687e...`

## Per-fold outer validation

| fold | outer period | samples | threshold | trades | net return | ann. Sharpe | max DD | exposure | macro F1 | dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.62 | 30 | -0.019651 | -0.2449 | 0.0628 | 0.0379 | 0.1798 | 0.5294 |
| 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.66 | 9 | +0.055412 | 1.7241 | 0.0196 | 0.0114 | 0.1298 | 0.6471 |
| 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.56 | 72 | +0.158958 | 1.2689 | 0.1292 | 0.0909 | 0.1426 | 0.5098 |
| 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.62 | 31 | -0.070168 | -0.9047 | 0.1185 | 0.0391 | 0.1138 | 0.3913 |

## Across the four temporal folds

- net return: mean +0.031138, std 0.099615, min -0.070168, median +0.017881, max +0.158958
- positive in **2 of 4** temporal folds

The statistical unit is the temporal period. These estimators are deterministic,
so repeating the run under another seed would copy this evidence, not add to it.
