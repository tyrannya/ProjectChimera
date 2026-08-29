# P5 cell — lightgbm on `ohlcv14_plus_mtf_v1`

**Question:** does causal mtf_v1, alone or combined with OHLCV14, add usable information beyond OHLCV14?

One information set, one model, 4 temporal folds.
Each row is a `64 x 42` window flattened to
`2688` values. Nothing is tuned.

**Research contract:** `btc-usdt-1h-gen1`, semantic identity
`sha256:dca6d0a891a257197a7c8aecec04fdba0a0b3009cfe93c5f3a397d458ab4a1de`.

**Sealed test block:** everything at or after `2025-08-27T23:00:00+00:00`. Not planned over, not fitted on,
not selected on, not scored. `sealed_test: false`.

**Feature spec:** combined `sha256:2a33458e5d85caec...`

## Per-fold outer validation

| fold | outer period | samples | threshold | trades | net return | ann. Sharpe | max DD | exposure | macro F1 | dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.66 | 135 | -0.238971 | -2.2943 | 0.3156 | 0.1705 | 0.2160 | 0.3415 |
| 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.76 | 12 | +0.004063 | 0.1443 | 0.0301 | 0.0152 | 0.1305 | 0.5417 |
| 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.74 | 7 | +0.068067 | 1.9719 | 0.0075 | 0.0088 | 0.1196 | 0.7692 |
| 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.70 | 27 | -0.045858 | -0.5411 | 0.0994 | 0.0341 | 0.1177 | 0.5769 |

## Across the four temporal folds

- net return: mean -0.053175, std 0.132350, min -0.238971, median -0.020898, max +0.068067
- positive in **2 of 4** temporal folds

The statistical unit is the temporal period. These estimators are deterministic,
so repeating the run under another seed would copy this evidence, not add to it.
