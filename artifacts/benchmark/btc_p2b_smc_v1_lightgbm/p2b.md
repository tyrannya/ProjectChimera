# P2b cell — lightgbm on `smc_v1`

One information set, one model, 4 temporal folds.
Each row is a `64 x 39` window flattened to
`2496` values. Nothing is tuned.

**Research contract:** `btc-usdt-1h-gen1`, semantic identity
`sha256:dca6d0a891a257197a7c8aecec04fdba0a0b3009cfe93c5f3a397d458ab4a1de`.

**Sealed test block:** everything at or after `2025-08-27T23:00:00+00:00`. Not planned over, not fitted on,
not selected on, not scored. `sealed_test: false`.

**Feature spec:** smc `smc_v1` `sha256:3421312fc8d8687e...`

## Per-fold outer validation

| fold | outer period | samples | threshold | trades | net return | ann. Sharpe | max DD | exposure | macro F1 | dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.70 | 24 | +0.170439 | 2.7954 | 0.0378 | 0.0303 | 0.1766 | 0.5000 |
| 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.64 | 79 | -0.069390 | -0.5959 | 0.1460 | 0.0997 | 0.1475 | 0.4242 |
| 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.60 | 119 | -0.101892 | -0.8346 | 0.1492 | 0.1503 | 0.1510 | 0.5121 |
| 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.58 | 146 | +0.150413 | 1.1336 | 0.1421 | 0.1843 | 0.1559 | 0.5356 |

## Across the four temporal folds

- net return: mean +0.037393, std 0.142919, min -0.101892, median +0.040511, max +0.170439
- positive in **2 of 4** temporal folds

The statistical unit is the temporal period. These estimators are deterministic,
so repeating the run under another seed would copy this evidence, not add to it.
