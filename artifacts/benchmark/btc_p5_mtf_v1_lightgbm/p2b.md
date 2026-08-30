# P5 cell — lightgbm on `mtf_v1`

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
| 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.74 | 16 | -0.074332 | -2.3700 | 0.0743 | 0.0202 | 0.1708 | 0.2593 |
| 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.64 | 67 | -0.212689 | -2.5745 | 0.2141 | 0.0846 | 0.1435 | 0.3701 |
| 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.68 | 16 | +0.098904 | 1.7135 | 0.0241 | 0.0202 | 0.1202 | 0.3750 |
| 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.66 | 15 | +0.013021 | 0.2698 | 0.0846 | 0.0189 | 0.1131 | 0.5333 |

## Across the four temporal folds

- net return: mean -0.043774, std 0.132977, min -0.212689, median -0.030655, max +0.098904
- positive in **2 of 4** temporal folds

The statistical unit is the temporal period. These estimators are deterministic,
so repeating the run under another seed would copy this evidence, not add to it.
