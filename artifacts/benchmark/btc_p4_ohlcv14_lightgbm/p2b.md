# P4 cell — lightgbm on `ohlcv14`

**Question:** does causal derivatives_v1, alone or combined with OHLCV14, add usable information beyond OHLCV14?

One information set, one model, 4 temporal folds.
Each row is a `64 x 14` window flattened to
`896` values. Nothing is tuned.

**Research contract:** `btc-usdt-1h-gen1`, semantic identity
`sha256:dca6d0a891a257197a7c8aecec04fdba0a0b3009cfe93c5f3a397d458ab4a1de`.

**Sealed test block:** everything at or after `2025-08-27T23:00:00+00:00`. Not planned over, not fitted on,
not selected on, not scored. `sealed_test: false`.

**Feature spec:** combined `sha256:2a33458e5d85caec...`

## Per-fold outer validation

| fold | outer period | samples | threshold | trades | net return | ann. Sharpe | max DD | exposure | macro F1 | dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | 4443 | 0.82 | 3 | -0.017623 | -0.3359 | 0.0468 | 0.0038 | 0.1699 | 0.2500 |
| 1 | 2023-09-24 → 2024-04-12 | 4656 | 0.54 | 280 | -0.111839 | -0.4898 | 0.3189 | 0.3535 | 0.2258 | 0.4501 |
| 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.56 | 170 | -0.542260 | -4.8983 | 0.5549 | 0.2146 | 0.1641 | 0.3737 |
| 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.66 | 28 | +0.185147 | 2.1849 | 0.0358 | 0.0354 | 0.1165 | 0.5490 |

## Across the four temporal folds

- net return: mean -0.121644, std 0.306570, min -0.542260, median -0.064731, max +0.185147
- positive in **1 of 4** temporal folds

The statistical unit is the temporal period. These estimators are deterministic,
so repeating the run under another seed would copy this evidence, not add to it.
