# P3 cell — lightgbm on `microstructure_v1`

**Question:** does causal microstructure_v1, alone or combined with OHLCV14, add usable information beyond OHLCV14?

One information set, one model, 4 temporal folds.
Each row is a `64 x 32` window flattened to
`2048` values. Nothing is tuned.

**Research contract:** `btc-usdt-1h-gen1`, semantic identity
`sha256:dca6d0a891a257197a7c8aecec04fdba0a0b3009cfe93c5f3a397d458ab4a1de`.

**Sealed test block:** everything at or after `2025-08-27T23:00:00+00:00`. Not planned over, not fitted on,
not selected on, not scored. `sealed_test: false`.

**Feature spec:** smc `smc_v1` `sha256:3421312fc8d8687e...`

## Per-fold outer validation

| fold | outer period | samples | threshold | trades | net return | ann. Sharpe | max DD | exposure | macro F1 | dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.64 | 66 | -0.114764 | -1.2623 | 0.1852 | 0.0833 | 0.1873 | 0.4490 |
| 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.66 | 7 | -0.027211 | -0.8594 | 0.0357 | 0.0088 | 0.1279 | 0.5714 |
| 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.60 | 43 | -0.151649 | -1.9341 | 0.1648 | 0.0543 | 0.1222 | 0.3469 |
| 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.60 | 38 | -0.023117 | -0.2638 | 0.1366 | 0.0480 | 0.1151 | 0.4231 |

## Across the four temporal folds

- net return: mean -0.079185, std 0.064192, min -0.151649, median -0.070988, max -0.023117
- positive in **0 of 4** temporal folds

The statistical unit is the temporal period. These estimators are deterministic,
so repeating the run under another seed would copy this evidence, not add to it.
