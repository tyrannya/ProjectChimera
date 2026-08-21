# P2c cell — lightgbm on `ohlcv14`

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
| 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.70 | 30 | -0.046515 | -0.5360 | 0.1085 | 0.0379 | 0.1774 | 0.3433 |
| 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.60 | 96 | -0.189062 | -1.5579 | 0.1978 | 0.1212 | 0.1623 | 0.4530 |
| 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.66 | 21 | -0.025714 | -0.4007 | 0.0744 | 0.0265 | 0.1228 | 0.4737 |
| 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.62 | 59 | +0.256085 | 2.3691 | 0.0369 | 0.0745 | 0.1290 | 0.5909 |

## Across the four temporal folds

- net return: mean -0.001302, std 0.186317, min -0.189062, median -0.036115, max +0.256085
- positive in **1 of 4** temporal folds

The statistical unit is the temporal period. These estimators are deterministic,
so repeating the run under another seed would copy this evidence, not add to it.
