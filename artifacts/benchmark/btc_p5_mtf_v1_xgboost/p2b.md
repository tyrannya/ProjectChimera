# P5 cell — xgboost on `mtf_v1`

**Question:** does causal mtf_v1, alone or combined with OHLCV14, add usable information beyond OHLCV14?

One information set, one model, 4 temporal folds.
Each row is a `64 x 28` window flattened to
`1792` values. Nothing is tuned.

**Research contract:** `btc-usdt-1h-gen1`, semantic identity
`sha256:dca6d0a891a257197a7c8aecec04fdba0a0b3009cfe93c5f3a397d458ab4a1de`.

**Sealed test block:** everything at or after `2025-08-27T23:00:00+00:00`. Not planned over, not fitted on,
not selected on, not scored. `sealed_test: false`.

**Feature spec:** combined `sha256:2a33458e5d85caec...`

## Per-fold outer validation

| fold | outer period | samples | threshold | trades | net return | ann. Sharpe | max DD | exposure | macro F1 | dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.64 | 89 | -0.217654 | -3.3841 | 0.2331 | 0.1124 | 0.1989 | 0.3320 |
| 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.60 | 112 | -0.169813 | -1.4995 | 0.2286 | 0.1414 | 0.1675 | 0.4228 |
| 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.64 | 35 | -0.039222 | -0.5224 | 0.0761 | 0.0442 | 0.1312 | 0.5600 |
| 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.60 | 51 | +0.008644 | 0.1775 | 0.1247 | 0.0644 | 0.1225 | 0.4190 |

## Across the four temporal folds

- net return: mean -0.104511, std 0.106665, min -0.217654, median -0.104517, max +0.008644
- positive in **1 of 4** temporal folds

The statistical unit is the temporal period. These estimators are deterministic,
so repeating the run under another seed would copy this evidence, not add to it.
