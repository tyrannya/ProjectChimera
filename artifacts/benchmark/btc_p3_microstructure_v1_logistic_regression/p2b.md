# P3 cell — logistic_regression on `microstructure_v1`

**Question:** does causal microstructure_v1, alone or combined with OHLCV14, add usable information beyond OHLCV14?

One information set, one model, 4 temporal folds.
Each row is a `64 x 32` window flattened to
`2048` values. Nothing is tuned.

**Research contract:** `btc-usdt-1h-gen1`, semantic identity
`sha256:dca6d0a891a257197a7c8aecec04fdba0a0b3009cfe93c5f3a397d458ab4a1de`.

**Sealed test block:** everything at or after `2025-08-27T23:00:00+00:00`. Not planned over, not fitted on,
not selected on, not scored. `sealed_test: false`.

**Feature spec:** smc `smc_v1` `sha256:3421312fc8d8687e...`

## Per-fold outer validation

| fold | outer period | samples | threshold | trades | net return | ann. Sharpe | max DD | exposure | macro F1 | dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.82 | 90 | -0.128970 | -2.0391 | 0.1850 | 0.1136 | 0.1872 | 0.3884 |
| 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.88 | 27 | -0.084011 | -2.1015 | 0.1019 | 0.0341 | 0.1320 | 0.3542 |
| 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.88 | 28 | +0.112619 | 1.9783 | 0.0553 | 0.0354 | 0.1221 | 0.4722 |
| 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.84 | 39 | +0.043236 | 0.6849 | 0.0687 | 0.0492 | 0.1167 | 0.5273 |

## Across the four temporal folds

- net return: mean -0.014282, std 0.111696, min -0.128970, median -0.020388, max +0.112619
- positive in **2 of 4** temporal folds

The statistical unit is the temporal period. These estimators are deterministic,
so repeating the run under another seed would copy this evidence, not add to it.
