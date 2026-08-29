# P4 cell — xgboost on `ohlcv14_plus_derivatives_v1`

**Question:** does causal derivatives_v1, alone or combined with OHLCV14, add usable information beyond OHLCV14?

One information set, one model, 4 temporal folds.
Each row is a `64 x 22` window flattened to
`1408` values. Nothing is tuned.

**Research contract:** `btc-usdt-1h-gen1`, semantic identity
`sha256:dca6d0a891a257197a7c8aecec04fdba0a0b3009cfe93c5f3a397d458ab4a1de`.

**Sealed test block:** everything at or after `2025-08-27T23:00:00+00:00`. Not planned over, not fitted on,
not selected on, not scored. `sealed_test: false`.

**Feature spec:** combined `sha256:2a33458e5d85caec...`

## Per-fold outer validation

| fold | outer period | samples | threshold | trades | net return | ann. Sharpe | max DD | exposure | macro F1 | dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | 4443 | 0.78 | 18 | -0.036194 | -1.2430 | 0.0362 | 0.0227 | 0.1723 | 0.2692 |
| 1 | 2023-09-24 → 2024-04-12 | 4656 | 0.70 | 20 | +0.005028 | 0.1357 | 0.0827 | 0.0253 | 0.1341 | 0.5385 |
| 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.62 | 72 | -0.196729 | -2.1866 | 0.2469 | 0.0909 | 0.1374 | 0.4326 |
| 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.68 | 15 | -0.019946 | -0.2622 | 0.0738 | 0.0189 | 0.1123 | 0.4412 |

## Across the four temporal folds

- net return: mean -0.061960, std 0.091431, min -0.196729, median -0.028070, max +0.005028
- positive in **1 of 4** temporal folds

The statistical unit is the temporal period. These estimators are deterministic,
so repeating the run under another seed would copy this evidence, not add to it.
