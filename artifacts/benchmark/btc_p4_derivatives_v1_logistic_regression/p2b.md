# P4 cell — logistic_regression on `derivatives_v1`

**Question:** does causal derivatives_v1, alone or combined with OHLCV14, add usable information beyond OHLCV14?

One information set, one model, 4 temporal folds.
Each row is a `64 x 8` window flattened to
`512` values. Nothing is tuned.

**Research contract:** `btc-usdt-1h-gen1`, semantic identity
`sha256:dca6d0a891a257197a7c8aecec04fdba0a0b3009cfe93c5f3a397d458ab4a1de`.

**Sealed test block:** everything at or after `2025-08-27T23:00:00+00:00`. Not planned over, not fitted on,
not selected on, not scored. `sealed_test: false`.

**Feature spec:** combined `sha256:2a33458e5d85caec...`

## Per-fold outer validation

| fold | outer period | samples | threshold | trades | net return | ann. Sharpe | max DD | exposure | macro F1 | dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | 4443 | 0.68 | 16 | -0.027344 | -0.7850 | 0.0442 | 0.0202 | 0.1722 | 0.2917 |
| 1 | 2023-09-24 → 2024-04-12 | 4656 | 0.60 | 65 | -0.010160 | -0.0135 | 0.0786 | 0.0821 | 0.1427 | 0.4259 |
| 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.64 | 3 | -0.036235 | -2.2053 | 0.0362 | 0.0038 | 0.1160 | 0.0000 |
| 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.58 | 30 | +0.021658 | 0.3593 | 0.0863 | 0.0379 | 0.1153 | 0.5106 |

## Across the four temporal folds

- net return: mean -0.013020, std 0.025527, min -0.036235, median -0.018752, max +0.021658
- positive in **1 of 4** temporal folds

The statistical unit is the temporal period. These estimators are deterministic,
so repeating the run under another seed would copy this evidence, not add to it.
