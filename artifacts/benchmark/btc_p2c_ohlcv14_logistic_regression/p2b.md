# P2c cell — logistic_regression on `ohlcv14`

**Question:** does causal chart_structure_v1, alone or combined with OHLCV14, add usable information beyond OHLCV14?

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
| 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.62 | 28 | -0.006932 | -0.0490 | 0.0588 | 0.0354 | 0.1808 | 0.5686 |
| 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.66 | 11 | +0.066156 | 1.9999 | 0.0196 | 0.0139 | 0.1305 | 0.6842 |
| 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.56 | 75 | +0.130769 | 1.0856 | 0.0891 | 0.0947 | 0.1428 | 0.5098 |
| 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.56 | 110 | -0.157307 | -1.0412 | 0.3166 | 0.1389 | 0.1441 | 0.5093 |

## Across the four temporal folds

- net return: mean +0.008171, std 0.123833, min -0.157307, median +0.029612, max +0.130769
- positive in **2 of 4** temporal folds

The statistical unit is the temporal period. These estimators are deterministic,
so repeating the run under another seed would copy this evidence, not add to it.
