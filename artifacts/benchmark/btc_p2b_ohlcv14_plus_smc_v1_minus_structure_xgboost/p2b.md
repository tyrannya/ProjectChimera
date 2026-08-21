# P2b cell — xgboost on `ohlcv14_plus_smc_v1_minus_structure`

One information set, one model, 4 temporal folds.
Each row is a `64 x 45` window flattened to
`2880` values. Nothing is tuned.

**Research contract:** `btc-usdt-1h-gen1`, semantic identity
`sha256:dca6d0a891a257197a7c8aecec04fdba0a0b3009cfe93c5f3a397d458ab4a1de`.

**Sealed test block:** everything at or after `2025-08-27T23:00:00+00:00`. Not planned over, not fitted on,
not selected on, not scored. `sealed_test: false`.

**Feature spec:** smc `smc_v1` `sha256:3421312fc8d8687e...`

## Per-fold outer validation

| fold | outer period | samples | threshold | trades | net return | ann. Sharpe | max DD | exposure | macro F1 | dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.66 | 58 | -0.142926 | -1.7151 | 0.1601 | 0.0732 | 0.1850 | 0.3559 |
| 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.66 | 37 | +0.292569 | 3.4418 | 0.0561 | 0.0467 | 0.1429 | 0.7121 |
| 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.60 | 67 | -0.078103 | -0.7829 | 0.1403 | 0.0846 | 0.1369 | 0.4754 |
| 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.64 | 40 | +0.185582 | 1.8164 | 0.0833 | 0.0505 | 0.1180 | 0.4247 |

## Across the four temporal folds

- net return: mean +0.064281, std 0.208196, min -0.142926, median +0.053739, max +0.292569
- positive in **2 of 4** temporal folds

The statistical unit is the temporal period. These estimators are deterministic,
so repeating the run under another seed would copy this evidence, not add to it.
