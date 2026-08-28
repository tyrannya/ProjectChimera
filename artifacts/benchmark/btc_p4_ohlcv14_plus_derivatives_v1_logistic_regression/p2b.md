# P4 cell — logistic_regression on `ohlcv14_plus_derivatives_v1`

**Question:** does causal derivatives_v1, alone or combined with OHLCV14, add usable information beyond OHLCV14?

One information set, one model, 4 temporal folds.
Each row is a `64 x 22` window flattened to
`1408` values. Nothing is tuned.

**Research contract:** `btc-usdt-1h-gen1`, semantic identity
`sha256:dca6d0a891a257197a7c8aecec04fdba0a0b3009cfe93c5f3a397d458ab4a1de`.

**Sealed test block:** everything at or after `2025-08-27T23:00:00+00:00`. Not planned over, not fitted on,
not selected on, not scored. `sealed_test: false`.

**Feature spec:** combined `sha256:2a33458e5d85caec...`

## Per-fold outer validation

| fold | outer period | samples | threshold | trades | net return | ann. Sharpe | max DD | exposure | macro F1 | dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | 4443 | 0.74 | 27 | +0.017871 | 0.3980 | 0.0404 | 0.0341 | 0.1806 | 0.5106 |
| 1 | 2023-09-24 → 2024-04-12 | 4656 | 0.68 | 50 | -0.115990 | -1.6152 | 0.1315 | 0.0631 | 0.1418 | 0.5000 |
| 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.72 | 7 | -0.073195 | -2.6186 | 0.0732 | 0.0088 | 0.1168 | 0.2000 |
| 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.62 | 100 | -0.128754 | -0.8468 | 0.2137 | 0.1263 | 0.1362 | 0.4444 |

## Across the four temporal folds

- net return: mean -0.075017, std 0.066327, min -0.128754, median -0.094592, max +0.017871
- positive in **1 of 4** temporal folds

The statistical unit is the temporal period. These estimators are deterministic,
so repeating the run under another seed would copy this evidence, not add to it.
