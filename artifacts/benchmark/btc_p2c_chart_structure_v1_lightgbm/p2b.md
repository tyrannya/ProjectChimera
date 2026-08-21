# P2b cell — lightgbm on `chart_structure_v1`

One information set, one model, 4 temporal folds.
Each row is a `64 x 30` window flattened to
`1920` values. Nothing is tuned.

**Research contract:** `btc-usdt-1h-gen1`, semantic identity
`sha256:dca6d0a891a257197a7c8aecec04fdba0a0b3009cfe93c5f3a397d458ab4a1de`.

**Sealed test block:** everything at or after `2025-08-27T23:00:00+00:00`. Not planned over, not fitted on,
not selected on, not scored. `sealed_test: false`.

**Feature spec:** smc `smc_v1` `sha256:3421312fc8d8687e...`

## Per-fold outer validation

| fold | outer period | samples | threshold | trades | net return | ann. Sharpe | max DD | exposure | macro F1 | dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.78 | 13 | -0.065004 | -2.2576 | 0.0677 | 0.0164 | 0.1708 | 0.2222 |
| 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.76 | 8 | +0.031004 | 1.2785 | 0.0159 | 0.0101 | 0.1286 | 0.5000 |
| 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.64 | 90 | -0.147132 | -1.3290 | 0.2102 | 0.1136 | 0.1462 | 0.4759 |
| 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.68 | 28 | +0.054467 | 1.0052 | 0.0412 | 0.0354 | 0.1155 | 0.5455 |

## Across the four temporal folds

- net return: mean -0.031666, std 0.092719, min -0.147132, median -0.017000, max +0.054467
- positive in **2 of 4** temporal folds

The statistical unit is the temporal period. These estimators are deterministic,
so repeating the run under another seed would copy this evidence, not add to it.
