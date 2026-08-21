# P2b cell — xgboost on `ohlcv14_plus_smc_v1_minus_sweeps`

One information set, one model, 4 temporal folds.
Each row is a `64 x 47` window flattened to
`3008` values. Nothing is tuned.

**Research contract:** `btc-usdt-1h-gen1`, semantic identity
`sha256:dca6d0a891a257197a7c8aecec04fdba0a0b3009cfe93c5f3a397d458ab4a1de`.

**Sealed test block:** everything at or after `2025-08-27T23:00:00+00:00`. Not planned over, not fitted on,
not selected on, not scored. `sealed_test: false`.

**Feature spec:** smc `smc_v1` `sha256:3421312fc8d8687e...`

## Per-fold outer validation

| fold | outer period | samples | threshold | trades | net return | ann. Sharpe | max DD | exposure | macro F1 | dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.62 | 105 | -0.166073 | -1.4648 | 0.2394 | 0.1326 | 0.2080 | 0.4500 |
| 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.70 | 20 | +0.030598 | 0.5504 | 0.0592 | 0.0253 | 0.1307 | 0.3824 |
| 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.68 | 10 | +0.016611 | 0.4562 | 0.0502 | 0.0126 | 0.1191 | 0.4444 |
| 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.66 | 30 | +0.171714 | 2.4397 | 0.0798 | 0.0379 | 0.1166 | 0.5600 |

## Across the four temporal folds

- net return: mean +0.013213, std 0.138540, min -0.166073, median +0.023605, max +0.171714
- positive in **3 of 4** temporal folds

The statistical unit is the temporal period. These estimators are deterministic,
so repeating the run under another seed would copy this evidence, not add to it.
