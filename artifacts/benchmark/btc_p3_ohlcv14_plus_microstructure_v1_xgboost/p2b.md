# P3 cell — xgboost on `ohlcv14_plus_microstructure_v1`

**Question:** does causal microstructure_v1, alone or combined with OHLCV14, add usable information beyond OHLCV14?

One information set, one model, 4 temporal folds.
Each row is a `64 x 46` window flattened to
`2944` values. Nothing is tuned.

**Research contract:** `btc-usdt-1h-gen1`, semantic identity
`sha256:dca6d0a891a257197a7c8aecec04fdba0a0b3009cfe93c5f3a397d458ab4a1de`.

**Sealed test block:** everything at or after `2025-08-27T23:00:00+00:00`. Not planned over, not fitted on,
not selected on, not scored. `sealed_test: false`.

**Feature spec:** smc `smc_v1` `sha256:3421312fc8d8687e...`

## Per-fold outer validation

| fold | outer period | samples | threshold | trades | net return | ann. Sharpe | max DD | exposure | macro F1 | dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.60 | 118 | -0.223011 | -2.2150 | 0.2289 | 0.1490 | 0.2062 | 0.4364 |
| 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.70 | 12 | +0.024126 | 0.5074 | 0.0440 | 0.0152 | 0.1297 | 0.4500 |
| 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.62 | 28 | +0.057675 | 1.0797 | 0.0442 | 0.0354 | 0.1228 | 0.4419 |
| 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.66 | 21 | +0.091029 | 1.5282 | 0.0190 | 0.0265 | 0.1146 | 0.6774 |

## Across the four temporal folds

- net return: mean -0.012545, std 0.142944, min -0.223011, median +0.040900, max +0.091029
- positive in **3 of 4** temporal folds

The statistical unit is the temporal period. These estimators are deterministic,
so repeating the run under another seed would copy this evidence, not add to it.
