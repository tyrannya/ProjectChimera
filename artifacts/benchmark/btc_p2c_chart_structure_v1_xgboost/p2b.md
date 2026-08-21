# P2b cell — xgboost on `chart_structure_v1`

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
| 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.76 | 18 | -0.077599 | -1.4756 | 0.0820 | 0.0227 | 0.1724 | 0.3333 |
| 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.70 | 20 | +0.022154 | 0.4549 | 0.0316 | 0.0253 | 0.1322 | 0.4000 |
| 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.66 | 67 | -0.087196 | -0.8012 | 0.1828 | 0.0846 | 0.1431 | 0.5500 |
| 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.70 | 26 | +0.092231 | 1.3430 | 0.0640 | 0.0328 | 0.1157 | 0.5581 |

## Across the four temporal folds

- net return: mean -0.012603, std 0.085609, min -0.087196, median -0.027723, max +0.092231
- positive in **2 of 4** temporal folds

The statistical unit is the temporal period. These estimators are deterministic,
so repeating the run under another seed would copy this evidence, not add to it.
