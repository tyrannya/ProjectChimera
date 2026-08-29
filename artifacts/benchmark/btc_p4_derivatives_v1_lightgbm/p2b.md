# P4 cell — lightgbm on `derivatives_v1`

**Question:** does causal derivatives_v1, alone or combined with OHLCV14, add usable information beyond OHLCV14?

One information set, one model, 4 temporal folds.
Each row is a `64 x 8` window flattened to
`512` values. Nothing is tuned.

**Research contract:** `btc-usdt-1h-gen1`, semantic identity
`sha256:dca6d0a891a257197a7c8aecec04fdba0a0b3009cfe93c5f3a397d458ab4a1de`.

**Sealed test block:** everything at or after `2025-08-27T23:00:00+00:00`. Not planned over, not fitted on,
not selected on, not scored. `sealed_test: false`.

**Feature spec:** combined `sha256:2a33458e5d85caec...`

## Per-fold outer validation

| fold | outer period | samples | threshold | trades | net return | ann. Sharpe | max DD | exposure | macro F1 | dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | 4443 | 0.62 | 96 | -0.158054 | -1.7801 | 0.1862 | 0.1212 | 0.2032 | 0.3767 |
| 1 | 2023-09-24 → 2024-04-12 | 4656 | 0.66 | 22 | +0.003675 | 0.1165 | 0.0536 | 0.0278 | 0.1354 | 0.4444 |
| 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.64 | 9 | -0.091735 | -2.0389 | 0.0917 | 0.0114 | 0.1178 | 0.2174 |
| 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.56 | 39 | -0.069995 | -1.0032 | 0.1167 | 0.0492 | 0.1156 | 0.3784 |

## Across the four temporal folds

- net return: mean -0.079027, std 0.066653, min -0.158054, median -0.080865, max +0.003675
- positive in **1 of 4** temporal folds

The statistical unit is the temporal period. These estimators are deterministic,
so repeating the run under another seed would copy this evidence, not add to it.
