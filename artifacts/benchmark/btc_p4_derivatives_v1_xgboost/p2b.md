# P4 cell — xgboost on `derivatives_v1`

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
| 0 | 2023-03-04 → 2023-09-24 | 4443 | 0.72 | 10 | -0.032045 | -1.0502 | 0.0346 | 0.0126 | 0.1725 | 0.4375 |
| 1 | 2023-09-24 → 2024-04-12 | 4656 | 0.60 | 82 | -0.041218 | -0.3421 | 0.1310 | 0.1035 | 0.1611 | 0.4434 |
| 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.60 | 19 | -0.155042 | -3.0588 | 0.1777 | 0.0240 | 0.1227 | 0.3774 |
| 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.62 | 11 | -0.002499 | -0.0866 | 0.0234 | 0.0139 | 0.1105 | 0.4762 |

## Across the four temporal folds

- net return: mean -0.057701, std 0.066964, min -0.155042, median -0.036631, max -0.002499
- positive in **0 of 4** temporal folds

The statistical unit is the temporal period. These estimators are deterministic,
so repeating the run under another seed would copy this evidence, not add to it.
