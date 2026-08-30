# P5 cell — xgboost on `ohlcv14`

**Question:** does causal mtf_v1, alone or combined with OHLCV14, add usable information beyond OHLCV14?

One information set, one model, 4 temporal folds.
Each row is a `64 x 14` window flattened to
`896` values. Nothing is tuned.

**Research contract:** `btc-usdt-1h-gen1`, semantic identity
`sha256:dca6d0a891a257197a7c8aecec04fdba0a0b3009cfe93c5f3a397d458ab4a1de`.

**Sealed test block:** everything at or after `2025-08-27T23:00:00+00:00`. Not planned over, not fitted on,
not selected on, not scored. `sealed_test: false`.

**Feature spec:** combined `sha256:2a33458e5d85caec...`

## Per-fold outer validation

| fold | outer period | samples | threshold | trades | net return | ann. Sharpe | max DD | exposure | macro F1 | dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.56 | 207 | -0.227758 | -1.7224 | 0.2480 | 0.2614 | 0.2409 | 0.3996 |
| 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.68 | 26 | +0.048244 | 0.7076 | 0.0713 | 0.0328 | 0.1354 | 0.4902 |
| 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.64 | 29 | +0.133502 | 1.8538 | 0.0224 | 0.0366 | 0.1263 | 0.5370 |
| 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.62 | 47 | +0.429306 | 3.6617 | 0.0304 | 0.0593 | 0.1320 | 0.6827 |

## Across the four temporal folds

- net return: mean +0.095824, std 0.270554, min -0.227758, median +0.090873, max +0.429306
- positive in **3 of 4** temporal folds

The statistical unit is the temporal period. These estimators are deterministic,
so repeating the run under another seed would copy this evidence, not add to it.
