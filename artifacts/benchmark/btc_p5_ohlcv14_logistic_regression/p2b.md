# P5 cell — logistic_regression on `ohlcv14`

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
| 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.74 | 2 | +0.008960 | 0.9446 | 0.0022 | 0.0025 | 0.1683 | 0.5000 |
| 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.62 | 41 | +0.077488 | 0.9531 | 0.0645 | 0.0518 | 0.1393 | 0.5672 |
| 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.56 | 89 | +0.202977 | 1.4746 | 0.1154 | 0.1124 | 0.1452 | 0.5120 |
| 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.56 | 116 | -0.044097 | -0.1481 | 0.2770 | 0.1465 | 0.1471 | 0.5336 |

## Across the four temporal folds

- net return: mean +0.061332, std 0.106743, min -0.044097, median +0.043224, max +0.202977
- positive in **3 of 4** temporal folds

The statistical unit is the temporal period. These estimators are deterministic,
so repeating the run under another seed would copy this evidence, not add to it.
