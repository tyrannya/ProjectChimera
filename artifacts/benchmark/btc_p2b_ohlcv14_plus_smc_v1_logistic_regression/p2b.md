# P2b cell — logistic_regression on `ohlcv14_plus_smc_v1`

One information set, one model, 4 temporal folds.
Each row is a `64 x 53` window flattened to
`3392` values. Nothing is tuned.

**Research contract:** `btc-usdt-1h-gen1`, semantic identity
`sha256:dca6d0a891a257197a7c8aecec04fdba0a0b3009cfe93c5f3a397d458ab4a1de`.

**Sealed test block:** everything at or after `2025-08-27T23:00:00+00:00`. Not planned over, not fitted on,
not selected on, not scored. `sealed_test: false`.

**Feature spec:** smc `smc_v1` `sha256:3421312fc8d8687e...`

## Per-fold outer validation

| fold | outer period | samples | threshold | trades | net return | ann. Sharpe | max DD | exposure | macro F1 | dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.90 | 67 | -0.113659 | -1.2746 | 0.1837 | 0.0846 | 0.1848 | 0.4158 |
| 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.90 | 22 | +0.069826 | 1.0787 | 0.0526 | 0.0278 | 0.1321 | 0.5484 |
| 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.84 | 25 | -0.026093 | -0.3730 | 0.0474 | 0.0316 | 0.1228 | 0.5000 |
| 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.82 | 55 | -0.051878 | -0.5009 | 0.1134 | 0.0694 | 0.1183 | 0.4250 |

## Across the four temporal folds

- net return: mean -0.030451, std 0.076283, min -0.113659, median -0.038985, max +0.069826
- positive in **1 of 4** temporal folds

The statistical unit is the temporal period. These estimators are deterministic,
so repeating the run under another seed would copy this evidence, not add to it.
