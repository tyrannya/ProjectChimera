# P2b cell — xgboost on `ohlcv14`

**Question:** does causal smc_v1, alone or combined with OHLCV14, add usable information beyond OHLCV14?

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
| 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.64 | 80 | -0.101571 | -1.0454 | 0.1478 | 0.1010 | 0.1968 | 0.4405 |
| 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.72 | 11 | +0.032526 | 0.6611 | 0.0393 | 0.0139 | 0.1293 | 0.3913 |
| 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.70 | 4 | -0.008425 | -0.3705 | 0.0144 | 0.0051 | 0.1172 | 0.4286 |
| 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.68 | 16 | +0.105629 | 1.8526 | 0.0310 | 0.0202 | 0.1129 | 0.5926 |

## Across the four temporal folds

- net return: mean +0.007040, std 0.086419, min -0.101571, median +0.012050, max +0.105629
- positive in **2 of 4** temporal folds

The statistical unit is the temporal period. These estimators are deterministic,
so repeating the run under another seed would copy this evidence, not add to it.
