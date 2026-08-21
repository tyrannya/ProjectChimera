# P2b cell — logistic_regression on `chart_structure_v1`

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
| 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.86 | 56 | -0.184801 | -2.1988 | 0.1848 | 0.0707 | 0.1751 | 0.2020 |
| 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.84 | 19 | +0.014957 | 0.4147 | 0.0116 | 0.0240 | 0.1311 | 0.4839 |
| 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.76 | 44 | -0.093523 | -1.1207 | 0.1589 | 0.0556 | 0.1262 | 0.4375 |
| 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.72 | 58 | +0.055146 | 0.6758 | 0.0757 | 0.0732 | 0.1253 | 0.5253 |

## Across the four temporal folds

- net return: mean -0.052055, std 0.108511, min -0.184801, median -0.039283, max +0.055146
- positive in **2 of 4** temporal folds

The statistical unit is the temporal period. These estimators are deterministic,
so repeating the run under another seed would copy this evidence, not add to it.
