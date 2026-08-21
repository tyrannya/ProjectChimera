# P2b cell — xgboost on `ohlcv14_plus_smc_v1`

**Question:** does causal smc_v1, alone or combined with OHLCV14, add usable information beyond OHLCV14?

One information set, one model, 4 temporal folds.
Each row is a `64 x 53` window flattened to
`3392` values. Nothing is tuned.

**Research contract:** `btc-usdt-1h-gen1`, semantic identity
`sha256:dca6d0a891a257197a7c8aecec04fdba0a0b3009cfe93c5f3a397d458ab4a1de`.

**Sealed test block:** everything at or after `2025-08-27T23:00:00+00:00`. Not planned over, not fitted on,
not selected on, not scored. `sealed_test: false`.

**Feature spec:** smc `smc_v1` `sha256:3421312fc8d8687e...`

## Per-fold outer validation

| fold | outer period | samples | threshold | trades | net return | ann. Sharpe | max DD | exposure | macro F1 | dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.66 | 43 | -0.112009 | -1.3738 | 0.2010 | 0.0543 | 0.1794 | 0.3333 |
| 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.70 | 20 | -0.080644 | -1.5729 | 0.0865 | 0.0253 | 0.1310 | 0.4333 |
| 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.64 | 26 | +0.113964 | 1.7825 | 0.0493 | 0.0328 | 0.1259 | 0.5490 |
| 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.62 | 69 | +0.047844 | 0.5222 | 0.1539 | 0.0871 | 0.1313 | 0.5748 |

## Across the four temporal folds

- net return: mean -0.007711, std 0.106597, min -0.112009, median -0.016400, max +0.113964
- positive in **2 of 4** temporal folds

The statistical unit is the temporal period. These estimators are deterministic,
so repeating the run under another seed would copy this evidence, not add to it.
