# P4 cell — logistic_regression on `ohlcv14`

**Question:** does causal derivatives_v1, alone or combined with OHLCV14, add usable information beyond OHLCV14?

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
| 0 | 2023-03-04 → 2023-09-24 | 4443 | 0.66 | 30 | +0.088301 | 1.3514 | 0.0399 | 0.0379 | 0.1805 | 0.4630 |
| 1 | 2023-09-24 → 2024-04-12 | 4656 | 0.66 | 22 | +0.030567 | 0.5632 | 0.0720 | 0.0278 | 0.1345 | 0.5946 |
| 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.60 | 48 | +0.164339 | 1.4740 | 0.0776 | 0.0606 | 0.1296 | 0.4419 |
| 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.58 | 89 | +0.247617 | 1.8337 | 0.1650 | 0.1124 | 0.1358 | 0.5030 |

## Across the four temporal folds

- net return: mean +0.132706, std 0.094180, min +0.030567, median +0.126320, max +0.247617
- positive in **4 of 4** temporal folds

The statistical unit is the temporal period. These estimators are deterministic,
so repeating the run under another seed would copy this evidence, not add to it.
