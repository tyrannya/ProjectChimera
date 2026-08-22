# P3 cell — lightgbm on `ohlcv14_plus_microstructure_v1`

**Question:** does causal microstructure_v1, alone or combined with OHLCV14, add usable information beyond OHLCV14?

One information set, one model, 4 temporal folds.
Each row is a `64 x 46` window flattened to
`2944` values. Nothing is tuned.

**Research contract:** `btc-usdt-1h-gen1`, semantic identity
`sha256:dca6d0a891a257197a7c8aecec04fdba0a0b3009cfe93c5f3a397d458ab4a1de`.

**Sealed test block:** everything at or after `2025-08-27T23:00:00+00:00`. Not planned over, not fitted on,
not selected on, not scored. `sealed_test: false`.

**Feature spec:** smc `smc_v1` `sha256:3421312fc8d8687e...`

## Per-fold outer validation

| fold | outer period | samples | threshold | trades | net return | ann. Sharpe | max DD | exposure | macro F1 | dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.66 | 49 | -0.090621 | -1.1907 | 0.1443 | 0.0619 | 0.1827 | 0.4615 |
| 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.62 | 50 | +0.023591 | 0.3315 | 0.1135 | 0.0631 | 0.1444 | 0.4717 |
| 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.68 | 5 | +0.032538 | 1.0723 | 0.0205 | 0.0063 | 0.1177 | 0.5714 |
| 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.62 | 40 | +0.065474 | 0.8780 | 0.0955 | 0.0505 | 0.1171 | 0.5091 |

## Across the four temporal folds

- net return: mean +0.007746, std 0.068006, min -0.090621, median +0.028064, max +0.065474
- positive in **3 of 4** temporal folds

The statistical unit is the temporal period. These estimators are deterministic,
so repeating the run under another seed would copy this evidence, not add to it.
