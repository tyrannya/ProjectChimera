# P5 cell — logistic_regression on `ohlcv14_plus_mtf_v1`

**Question:** does causal mtf_v1, alone or combined with OHLCV14, add usable information beyond OHLCV14?

One information set, one model, 4 temporal folds.
Each row is a `64 x 42` window flattened to
`2688` values. Nothing is tuned.

**Research contract:** `btc-usdt-1h-gen1`, semantic identity
`sha256:dca6d0a891a257197a7c8aecec04fdba0a0b3009cfe93c5f3a397d458ab4a1de`.

**Sealed test block:** everything at or after `2025-08-27T23:00:00+00:00`. Not planned over, not fitted on,
not selected on, not scored. `sealed_test: false`.

**Feature spec:** combined `sha256:2a33458e5d85caec...`

## Per-fold outer validation

| fold | outer period | samples | threshold | trades | net return | ann. Sharpe | max DD | exposure | macro F1 | dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.80 | 31 | +0.053752 | 0.9449 | 0.0623 | 0.0391 | 0.1761 | 0.4000 |
| 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.78 | 41 | -0.107694 | -1.4956 | 0.1104 | 0.0518 | 0.1349 | 0.3939 |
| 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.82 | 13 | -0.086772 | -1.4504 | 0.0964 | 0.0164 | 0.1188 | 0.4211 |
| 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.80 | 9 | -0.014788 | -0.2978 | 0.0592 | 0.0114 | 0.1088 | 0.4000 |

## Across the four temporal folds

- net return: mean -0.038876, std 0.073462, min -0.107694, median -0.050780, max +0.053752
- positive in **1 of 4** temporal folds

The statistical unit is the temporal period. These estimators are deterministic,
so repeating the run under another seed would copy this evidence, not add to it.
