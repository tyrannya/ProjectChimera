# P2b cell — logistic_regression on `smc_v1`

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
| 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.84 | 119 | -0.128245 | -1.3208 | 0.2051 | 0.1503 | 0.1984 | 0.4041 |
| 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.88 | 22 | -0.027936 | -0.7273 | 0.0526 | 0.0278 | 0.1309 | 0.5000 |
| 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.82 | 32 | +0.027249 | 0.6866 | 0.0470 | 0.0404 | 0.1247 | 0.5098 |
| 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.74 | 152 | -0.187203 | -1.2390 | 0.2788 | 0.1919 | 0.1457 | 0.4440 |

## Across the four temporal folds

- net return: mean -0.079034, std 0.096660, min -0.187203, median -0.078090, max +0.027249
- positive in **1 of 4** temporal folds

The statistical unit is the temporal period. These estimators are deterministic,
so repeating the run under another seed would copy this evidence, not add to it.
