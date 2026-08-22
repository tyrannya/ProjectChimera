# P3 cell — logistic_regression on `ohlcv14_plus_microstructure_v1`

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
| 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.90 | 33 | -0.019030 | -0.3790 | 0.0554 | 0.0417 | 0.1766 | 0.4082 |
| 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.90 | 23 | -0.073999 | -2.0485 | 0.0940 | 0.0290 | 0.1305 | 0.3000 |
| 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.86 | 40 | +0.012511 | 0.2337 | 0.0880 | 0.0505 | 0.1235 | 0.4200 |
| 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.86 | 36 | +0.099168 | 1.2404 | 0.0486 | 0.0455 | 0.1158 | 0.5208 |

## Across the four temporal folds

- net return: mean +0.004663, std 0.072438, min -0.073999, median -0.003259, max +0.099168
- positive in **2 of 4** temporal folds

The statistical unit is the temporal period. These estimators are deterministic,
so repeating the run under another seed would copy this evidence, not add to it.
