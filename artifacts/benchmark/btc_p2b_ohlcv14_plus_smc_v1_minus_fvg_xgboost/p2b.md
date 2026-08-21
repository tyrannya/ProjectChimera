# P2b cell — xgboost on `ohlcv14_plus_smc_v1_minus_fvg`

**Question:** does causal smc_v1, alone or combined with OHLCV14, add usable information beyond OHLCV14?

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
| 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.68 | 22 | +0.027773 | 0.4461 | 0.1137 | 0.0278 | 0.1741 | 0.4054 |
| 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.72 | 12 | -0.112331 | -2.0932 | 0.1123 | 0.0152 | 0.1284 | 0.2727 |
| 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.60 | 95 | +0.083240 | 0.8027 | 0.1188 | 0.1199 | 0.1464 | 0.5266 |
| 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.62 | 73 | +0.127198 | 1.2305 | 0.0780 | 0.0922 | 0.1354 | 0.5782 |

## Across the four temporal folds

- net return: mean +0.031470, std 0.104142, min -0.112331, median +0.055507, max +0.127198
- positive in **3 of 4** temporal folds

The statistical unit is the temporal period. These estimators are deterministic,
so repeating the run under another seed would copy this evidence, not add to it.
