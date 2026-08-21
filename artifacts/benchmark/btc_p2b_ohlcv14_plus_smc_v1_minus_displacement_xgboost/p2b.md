# P2b cell — xgboost on `ohlcv14_plus_smc_v1_minus_displacement`

**Question:** does causal smc_v1, alone or combined with OHLCV14, add usable information beyond OHLCV14?

One information set, one model, 4 temporal folds.
Each row is a `64 x 48` window flattened to
`3072` values. Nothing is tuned.

**Research contract:** `btc-usdt-1h-gen1`, semantic identity
`sha256:dca6d0a891a257197a7c8aecec04fdba0a0b3009cfe93c5f3a397d458ab4a1de`.

**Sealed test block:** everything at or after `2025-08-27T23:00:00+00:00`. Not planned over, not fitted on,
not selected on, not scored. `sealed_test: false`.

**Feature spec:** smc `smc_v1` `sha256:3421312fc8d8687e...`

## Per-fold outer validation

| fold | outer period | samples | threshold | trades | net return | ann. Sharpe | max DD | exposure | macro F1 | dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.70 | 17 | -0.012099 | -0.0751 | 0.1389 | 0.0215 | 0.1728 | 0.4783 |
| 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.68 | 24 | -0.072431 | -1.3824 | 0.0890 | 0.0303 | 0.1351 | 0.5106 |
| 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.62 | 51 | +0.077028 | 0.8911 | 0.0630 | 0.0644 | 0.1350 | 0.5824 |
| 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.64 | 51 | +0.141094 | 1.4391 | 0.1081 | 0.0644 | 0.1217 | 0.5000 |

## Across the four temporal folds

- net return: mean +0.033398, std 0.094467, min -0.072431, median +0.032465, max +0.141094
- positive in **2 of 4** temporal folds

The statistical unit is the temporal period. These estimators are deterministic,
so repeating the run under another seed would copy this evidence, not add to it.
