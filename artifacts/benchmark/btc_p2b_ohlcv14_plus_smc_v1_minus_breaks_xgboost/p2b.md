# P2b cell — xgboost on `ohlcv14_plus_smc_v1_minus_breaks`

One information set, one model, 4 temporal folds.
Each row is a `64 x 47` window flattened to
`3008` values. Nothing is tuned.

**Research contract:** `btc-usdt-1h-gen1`, semantic identity
`sha256:dca6d0a891a257197a7c8aecec04fdba0a0b3009cfe93c5f3a397d458ab4a1de`.

**Sealed test block:** everything at or after `2025-08-27T23:00:00+00:00`. Not planned over, not fitted on,
not selected on, not scored. `sealed_test: false`.

**Feature spec:** smc `smc_v1` `sha256:3421312fc8d8687e...`

## Per-fold outer validation

| fold | outer period | samples | threshold | trades | net return | ann. Sharpe | max DD | exposure | macro F1 | dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.70 | 25 | -0.097744 | -1.1667 | 0.2115 | 0.0316 | 0.1748 | 0.4167 |
| 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.64 | 61 | -0.009035 | -0.0215 | 0.1130 | 0.0770 | 0.1464 | 0.5044 |
| 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.60 | 76 | +0.029593 | 0.3659 | 0.1042 | 0.0960 | 0.1394 | 0.4926 |
| 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.66 | 38 | +0.100973 | 1.2366 | 0.0572 | 0.0480 | 0.1184 | 0.5763 |

## Across the four temporal folds

- net return: mean +0.005947, std 0.082796, min -0.097744, median +0.010279, max +0.100973
- positive in **2 of 4** temporal folds

The statistical unit is the temporal period. These estimators are deterministic,
so repeating the run under another seed would copy this evidence, not add to it.
