# P5 cell — xgboost on `ohlcv14_plus_mtf_v1`

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
| 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.74 | 22 | -0.112678 | -2.3906 | 0.1162 | 0.0278 | 0.1750 | 0.4000 |
| 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.72 | 21 | -0.027115 | -0.3859 | 0.0633 | 0.0265 | 0.1334 | 0.4082 |
| 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.74 | 8 | +0.093658 | 1.8746 | 0.0182 | 0.0101 | 0.1189 | 0.7273 |
| 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.66 | 48 | +0.245659 | 1.9886 | 0.1405 | 0.0606 | 0.1268 | 0.6333 |

## Across the four temporal folds

- net return: mean +0.049881, std 0.155563, min -0.112678, median +0.033272, max +0.245659
- positive in **2 of 4** temporal folds

The statistical unit is the temporal period. These estimators are deterministic,
so repeating the run under another seed would copy this evidence, not add to it.
