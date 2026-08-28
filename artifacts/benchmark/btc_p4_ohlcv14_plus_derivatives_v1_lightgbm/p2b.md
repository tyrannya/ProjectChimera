# P4 cell — lightgbm on `ohlcv14_plus_derivatives_v1`

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
| 0 | 2023-03-04 → 2023-09-24 | 4443 | 0.80 | 16 | -0.082431 | -1.7089 | 0.0827 | 0.0202 | 0.1738 | 0.5000 |
| 1 | 2023-09-24 → 2024-04-12 | 4656 | 0.64 | 73 | -0.032540 | -0.2072 | 0.1412 | 0.0922 | 0.1535 | 0.4839 |
| 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.60 | 106 | -0.189804 | -1.5762 | 0.2718 | 0.1338 | 0.1479 | 0.4493 |
| 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.66 | 28 | -0.066293 | -0.6912 | 0.1138 | 0.0354 | 0.1171 | 0.5000 |

## Across the four temporal folds

- net return: mean -0.092767, std 0.067949, min -0.189804, median -0.074362, max -0.032540
- positive in **0 of 4** temporal folds

The statistical unit is the temporal period. These estimators are deterministic,
so repeating the run under another seed would copy this evidence, not add to it.
