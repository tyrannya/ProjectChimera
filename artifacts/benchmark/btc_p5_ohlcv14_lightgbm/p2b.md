# P5 cell — lightgbm on `ohlcv14`

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
| 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.64 | 76 | -0.114274 | -1.0692 | 0.1799 | 0.0960 | 0.1941 | 0.3851 |
| 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.64 | 50 | -0.025507 | -0.2425 | 0.0845 | 0.0631 | 0.1449 | 0.4865 |
| 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.60 | 72 | -0.110713 | -0.9608 | 0.1509 | 0.0909 | 0.1377 | 0.4412 |
| 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.60 | 78 | +0.079661 | 0.7256 | 0.1352 | 0.0985 | 0.1368 | 0.5305 |

## Across the four temporal folds

- net return: mean -0.042708, std 0.091317, min -0.114274, median -0.068110, max +0.079661
- positive in **1 of 4** temporal folds

The statistical unit is the temporal period. These estimators are deterministic,
so repeating the run under another seed would copy this evidence, not add to it.
