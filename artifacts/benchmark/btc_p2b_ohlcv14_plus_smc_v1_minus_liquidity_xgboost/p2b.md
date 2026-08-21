# P2b cell — xgboost on `ohlcv14_plus_smc_v1_minus_liquidity`

One information set, one model, 4 temporal folds.
Each row is a `64 x 47` window flattened to
`3008` values. Nothing is tuned.

**Research contract:** `btc-usdt-1h-gen1`, semantic identity
`sha256:dca6d0a891a257197a7c8aecec04fdba0a0b3009cfe93c5f3a397d458ab4a1de`.

**Sealed test block:** everything at or after `2025-08-27T23:00:00+00:00`. Not planned over, not fitted on,
not selected on, not scored. `sealed_test: false`.

**Feature spec:** smc `smc_v1` `sha256:3421312fc8d8687e...`

## Per-fold outer validation

| fold | outer period | samples | threshold | trades | net return | ann. Sharpe | max DD | exposure | macro F1 | dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.68 | 30 | -0.157832 | -1.8116 | 0.2303 | 0.0379 | 0.1766 | 0.4000 |
| 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.70 | 13 | -0.039686 | -0.9786 | 0.0571 | 0.0164 | 0.1311 | 0.5385 |
| 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.70 | 2 | -0.003071 | -0.1613 | 0.0038 | 0.0025 | 0.1164 | 0.3333 |
| 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.64 | 41 | +0.115909 | 1.4466 | 0.0903 | 0.0518 | 0.1194 | 0.5522 |

## Across the four temporal folds

- net return: mean -0.021170, std 0.112750, min -0.157832, median -0.021378, max +0.115909
- positive in **1 of 4** temporal folds

The statistical unit is the temporal period. These estimators are deterministic,
so repeating the run under another seed would copy this evidence, not add to it.
