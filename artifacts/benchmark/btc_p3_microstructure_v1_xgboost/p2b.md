# P3 cell — xgboost on `microstructure_v1`

**Question:** does causal microstructure_v1, alone or combined with OHLCV14, add usable information beyond OHLCV14?

One information set, one model, 4 temporal folds.
Each row is a `64 x 32` window flattened to
`2048` values. Nothing is tuned.

**Research contract:** `btc-usdt-1h-gen1`, semantic identity
`sha256:dca6d0a891a257197a7c8aecec04fdba0a0b3009cfe93c5f3a397d458ab4a1de`.

**Sealed test block:** everything at or after `2025-08-27T23:00:00+00:00`. Not planned over, not fitted on,
not selected on, not scored. `sealed_test: false`.

**Feature spec:** smc `smc_v1` `sha256:3421312fc8d8687e...`

## Per-fold outer validation

| fold | outer period | samples | threshold | trades | net return | ann. Sharpe | max DD | exposure | macro F1 | dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.64 | 53 | -0.001580 | 0.0424 | 0.0630 | 0.0669 | 0.1802 | 0.4085 |
| 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.68 | 5 | +0.004570 | 0.2444 | 0.0094 | 0.0063 | 0.1273 | 0.6000 |
| 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.58 | 65 | -0.203475 | -2.3848 | 0.2274 | 0.0821 | 0.1278 | 0.4571 |
| 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.58 | 99 | +0.243765 | 1.9645 | 0.1000 | 0.1250 | 0.1282 | 0.5000 |

## Across the four temporal folds

- net return: mean +0.010820, std 0.182919, min -0.203475, median +0.001495, max +0.243765
- positive in **2 of 4** temporal folds

The statistical unit is the temporal period. These estimators are deterministic,
so repeating the run under another seed would copy this evidence, not add to it.
