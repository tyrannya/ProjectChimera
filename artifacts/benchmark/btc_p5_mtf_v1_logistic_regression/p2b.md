# P5 cell — logistic_regression on `mtf_v1`

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
| 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.82 | 7 | -0.009230 | -0.3384 | 0.0183 | 0.0088 | 0.1686 | 0.2222 |
| 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.64 | 119 | -0.337711 | -3.2340 | 0.3556 | 0.1503 | 0.1654 | 0.4354 |
| 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.76 | 16 | -0.192572 | -3.0734 | 0.2091 | 0.0202 | 0.1201 | 0.3929 |
| 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.74 | 11 | -0.067798 | -1.4347 | 0.0835 | 0.0139 | 0.1094 | 0.4118 |

## Across the four temporal folds

- net return: mean -0.151828, std 0.145611, min -0.337711, median -0.130185, max -0.009230
- positive in **0 of 4** temporal folds

The statistical unit is the temporal period. These estimators are deterministic,
so repeating the run under another seed would copy this evidence, not add to it.
