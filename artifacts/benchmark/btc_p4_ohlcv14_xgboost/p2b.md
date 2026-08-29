# P4 cell — xgboost on `ohlcv14`

**Question:** does causal derivatives_v1, alone or combined with OHLCV14, add usable information beyond OHLCV14?

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
| 0 | 2023-03-04 → 2023-09-24 | 4443 | 0.78 | 14 | -0.020017 | -0.6351 | 0.0397 | 0.0177 | 0.1736 | 0.4286 |
| 1 | 2023-09-24 → 2024-04-12 | 4656 | 0.70 | 15 | +0.051490 | 0.8818 | 0.0351 | 0.0189 | 0.1307 | 0.5455 |
| 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.66 | 17 | -0.103661 | -2.0214 | 0.1162 | 0.0215 | 0.1191 | 0.3478 |
| 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.64 | 42 | -0.043012 | -0.3814 | 0.0921 | 0.0530 | 0.1201 | 0.4810 |

## Across the four temporal folds

- net return: mean -0.028800, std 0.064109, min -0.103661, median -0.031515, max +0.051490
- positive in **1 of 4** temporal folds

The statistical unit is the temporal period. These estimators are deterministic,
so repeating the run under another seed would copy this evidence, not add to it.
