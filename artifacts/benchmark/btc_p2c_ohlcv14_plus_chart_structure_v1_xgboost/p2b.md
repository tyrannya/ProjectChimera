# P2c cell — xgboost on `ohlcv14_plus_chart_structure_v1`

**Question:** does causal chart_structure_v1, alone or combined with OHLCV14, add usable information beyond OHLCV14?

One information set, one model, 4 temporal folds.
Each row is a `64 x 44` window flattened to
`2816` values. Nothing is tuned.

**Research contract:** `btc-usdt-1h-gen1`, semantic identity
`sha256:dca6d0a891a257197a7c8aecec04fdba0a0b3009cfe93c5f3a397d458ab4a1de`.

**Sealed test block:** everything at or after `2025-08-27T23:00:00+00:00`. Not planned over, not fitted on,
not selected on, not scored. `sealed_test: false`.

**Feature spec:** smc `smc_v1` `sha256:3421312fc8d8687e...`

## Per-fold outer validation

| fold | outer period | samples | threshold | trades | net return | ann. Sharpe | max DD | exposure | macro F1 | dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.76 | 8 | -0.026855 | -0.5807 | 0.0586 | 0.0101 | 0.1732 | 0.5714 |
| 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.70 | 27 | -0.024471 | -0.3752 | 0.0471 | 0.0341 | 0.1317 | 0.3659 |
| 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.54 | 260 | -0.388741 | -2.3466 | 0.4018 | 0.3283 | 0.2135 | 0.4561 |
| 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.60 | 121 | +0.040238 | 0.3973 | 0.1806 | 0.1528 | 0.1596 | 0.5298 |

## Across the four temporal folds

- net return: mean -0.099957, std 0.195015, min -0.388741, median -0.025663, max +0.040238
- positive in **1 of 4** temporal folds

The statistical unit is the temporal period. These estimators are deterministic,
so repeating the run under another seed would copy this evidence, not add to it.
