# P2b cell — lightgbm on `ohlcv14_plus_smc_v1`

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
| 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.64 | 55 | -0.220954 | -2.3223 | 0.2314 | 0.0694 | 0.1851 | 0.3832 |
| 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.60 | 150 | -0.171190 | -1.1753 | 0.1900 | 0.1894 | 0.1814 | 0.5060 |
| 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.66 | 20 | -0.028819 | -0.6251 | 0.0581 | 0.0253 | 0.1203 | 0.4000 |
| 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.66 | 36 | -0.000090 | 0.0727 | 0.0912 | 0.0455 | 0.1187 | 0.5000 |

## Across the four temporal folds

- net return: mean -0.105263, std 0.107449, min -0.220954, median -0.100005, max -0.000090
- positive in **0 of 4** temporal folds

The statistical unit is the temporal period. These estimators are deterministic,
so repeating the run under another seed would copy this evidence, not add to it.
