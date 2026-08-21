# P2c cell — logistic_regression on `ohlcv14_plus_chart_structure_v1`

**Question:** does causal chart_structure_v1, alone or combined with OHLCV14, add usable information beyond OHLCV14?

One information set, one model, 4 temporal folds.
Each row is a `64 x 44` window flattened to
`2816` values. Nothing is tuned.

**Research contract:** `btc-usdt-1h-gen1`, semantic identity
`sha256:dca6d0a891a257197a7c8aecec04fdba0a0b3009cfe93c5f3a397d458ab4a1de`.

**Sealed test block:** everything at or after `2025-08-27T23:00:00+00:00`. Not planned over, not fitted on,
not selected on, not scored. `sealed_test: false`.

**Feature spec:** smc `smc_v1` `sha256:3421312fc8d8687e...`

## Per-fold outer validation

| fold | outer period | samples | threshold | trades | net return | ann. Sharpe | max DD | exposure | macro F1 | dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.86 | 73 | -0.153192 | -2.1305 | 0.1532 | 0.0922 | 0.1843 | 0.3256 |
| 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.88 | 14 | -0.007183 | -0.1944 | 0.0253 | 0.0177 | 0.1281 | 0.3529 |
| 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.70 | 136 | -0.114550 | -0.7033 | 0.3003 | 0.1717 | 0.1489 | 0.3909 |
| 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.68 | 146 | -0.169830 | -1.1029 | 0.2238 | 0.1843 | 0.1526 | 0.4821 |

## Across the four temporal folds

- net return: mean -0.111189, std 0.073102, min -0.169830, median -0.133871, max -0.007183
- positive in **0 of 4** temporal folds

The statistical unit is the temporal period. These estimators are deterministic,
so repeating the run under another seed would copy this evidence, not add to it.
