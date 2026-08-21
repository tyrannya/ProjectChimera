# P2b cell — lightgbm on `ohlcv14_plus_chart_structure_v1`

One information set, one model, 4 temporal folds.
Each row is a `64 x 44` window flattened to
`2816` values. Nothing is tuned.

**Research contract:** `btc-usdt-1h-gen1`, semantic identity
`sha256:dca6d0a891a257197a7c8aecec04fdba0a0b3009cfe93c5f3a397d458ab4a1de`.

**Sealed test block:** everything at or after `2025-08-27T23:00:00+00:00`. Not planned over, not fitted on,
not selected on, not scored. `sealed_test: false`.

**Feature spec:** smc `smc_v1` `sha256:3421312fc8d8687e...`

## Per-fold outer validation

| fold | outer period | samples | threshold | trades | net return | ann. Sharpe | max DD | exposure | macro F1 | dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.76 | 13 | +0.012508 | 0.3797 | 0.0354 | 0.0164 | 0.1727 | 0.3793 |
| 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.74 | 10 | -0.018675 | -0.4393 | 0.0403 | 0.0126 | 0.1289 | 0.3333 |
| 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.58 | 165 | -0.399846 | -3.3233 | 0.4212 | 0.2083 | 0.1742 | 0.4509 |
| 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.66 | 41 | +0.166344 | 1.6504 | 0.0633 | 0.0518 | 0.1243 | 0.5882 |

## Across the four temporal folds

- net return: mean -0.059917, std 0.240619, min -0.399846, median -0.003084, max +0.166344
- positive in **2 of 4** temporal folds

The statistical unit is the temporal period. These estimators are deterministic,
so repeating the run under another seed would copy this evidence, not add to it.
