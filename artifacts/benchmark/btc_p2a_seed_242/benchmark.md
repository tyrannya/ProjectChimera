# P2a simple-model benchmark (seed 242)

Three untuned models — logistic regression, LightGBM, XGBoost — fitted on the
**same samples** the MTST Transformer is fitted on: each row is MTST's
`64 x 14` window flattened to
`896` values in a declared order. Nothing is tuned;
the configurations are predeclared constants.

**Research contract:** `btc-usdt-1h-gen1` (generation 1, directional-classification, binance BTC/USDT 1h),
semantic identity `sha256:dca6d0a891a257197a7c8aecec04fdba0a0b3009cfe93c5f3a397d458ab4a1de`.

**Research input:** `sha256:1c09218442cdf98cfe2f49ac521c70b2d7692717d7488e8105c972a3d2ac4740` over 48217 research-visible rows.

**Sealed test block:** everything at or after `2025-08-27T23:00:00+00:00`, rows 48217-56790. Not planned over, not
fitted on, not selected on, not scored. `sealed_test: false`.

## Fold geometry and selected thresholds

| fold | train rows | inner rows | outer rows | outer period | logistic_regression thr | lightgbm thr | xgboost thr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0-21697 | 21697-26518 | 26518-31339 | 2023-03-04 to 2023-09-24 | 0.52 | 0.70 | 0.64 |
| 1 | 0-26518 | 26518-31339 | 31339-36160 | 2023-09-24 to 2024-04-12 | 0.66 | 0.60 | 0.72 |
| 2 | 0-31339 | 31339-36160 | 36160-40981 | 2024-04-12 to 2024-10-30 | 0.56 | 0.66 | 0.70 |
| 3 | 0-36160 | 36160-40981 | 40981-45802 | 2024-10-30 to 2025-05-19 | 0.56 | 0.62 | 0.68 |

## Outer validation (the reported block)

`ann. Sharpe` is candle-level portfolio returns (equity unchanged while flat, marked to market while a position is open, both cost sides charged when they are paid), annualised by sqrt(candles_per_year) over elapsed wall-clock time, with a zero return for each calendar interval absent from the processed dataset. `n/a` means undefined, not zero.

| fold | model | trades | net return | ann. Sharpe | per-trade Sharpe | exposure | max DD | macro F1 | dir acc | coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | logistic_regression | 190 | -0.2924 | -2.26 | -0.12 | 0.2399 | 0.3146 | 0.2418 | 0.4786 | 0.0848 |
| 0 | lightgbm | 30 | -0.0465 | -0.54 | -0.05 | 0.0379 | 0.1085 | 0.1774 | 0.3433 | 0.0143 |
| 0 | xgboost | 80 | -0.1016 | -1.05 | -0.08 | 0.1010 | 0.1478 | 0.1968 | 0.4405 | 0.0359 |
| 0 | majority_baseline | 781 | -0.7562 | -5.89 | -0.16 | 0.9861 | 0.7842 | 0.1633 | 0.3244 | 1.0000 |
| 0 | momentum_baseline | 781 | -0.7828 | -6.39 | -0.17 | 0.9861 | 0.8337 | 0.2432 | 0.3018 | 0.9968 |
| 1 | logistic_regression | 9 | +0.0554 | 1.72 | 0.46 | 0.0114 | 0.0196 | 0.1298 | 0.6471 | 0.0036 |
| 1 | lightgbm | 96 | -0.1891 | -1.56 | -0.14 | 0.1212 | 0.1978 | 0.1623 | 0.4530 | 0.0492 |
| 1 | xgboost | 11 | +0.0325 | 0.66 | 0.15 | 0.0139 | 0.0393 | 0.1293 | 0.3913 | 0.0048 |
| 1 | majority_baseline | 792 | -0.4470 | -2.00 | -0.05 | 1.0000 | 0.4879 | 0.1995 | 0.4270 | 1.0000 |
| 1 | momentum_baseline | 792 | -0.7716 | -5.35 | -0.13 | 1.0000 | 0.7827 | 0.2726 | 0.3702 | 0.9981 |
| 2 | logistic_regression | 72 | +0.1590 | 1.27 | 0.11 | 0.0909 | 0.1292 | 0.1426 | 0.5098 | 0.0322 |
| 2 | lightgbm | 21 | -0.0257 | -0.40 | -0.06 | 0.0265 | 0.0744 | 0.1228 | 0.4737 | 0.0080 |
| 2 | xgboost | 4 | -0.0084 | -0.37 | -0.26 | 0.0051 | 0.0144 | 0.1172 | 0.4286 | 0.0015 |
| 2 | majority_baseline | 792 | -0.7732 | -5.24 | -0.14 | 1.0000 | 0.7862 | 0.1937 | 0.4095 | 1.0000 |
| 2 | momentum_baseline | 792 | -0.7792 | -5.35 | -0.15 | 1.0000 | 0.7907 | 0.2809 | 0.3748 | 0.9983 |
| 3 | logistic_regression | 111 | -0.1453 | -0.92 | -0.07 | 0.1402 | 0.3236 | 0.1442 | 0.5093 | 0.0455 |
| 3 | lightgbm | 59 | +0.2561 | 2.37 | 0.25 | 0.0745 | 0.0369 | 0.1290 | 0.5909 | 0.0231 |
| 3 | xgboost | 16 | +0.1056 | 1.85 | 0.39 | 0.0202 | 0.0310 | 0.1129 | 0.5926 | 0.0057 |
| 3 | majority_baseline | 792 | -0.6921 | -3.78 | -0.10 | 1.0000 | 0.7464 | 0.1989 | 0.4253 | 1.0000 |
| 3 | momentum_baseline | 792 | -0.8439 | -6.09 | -0.16 | 1.0000 | 0.8664 | 0.2844 | 0.3859 | 0.9989 |

### Across folds, outer validation only (mean ± std)

| model | net return | ann. Sharpe | exposure | max DD | macro F1 | dir acc | trades | positive folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | -0.0558 ± 0.2020 | -0.05 ± 1.87 | 0.1206 ± 0.0956 | 0.1968 ± 0.1482 | 0.1646 ± 0.0519 | 0.5362 ± 0.0754 | 95.5 ± 75.7 | 2/4 |
| lightgbm | -0.0013 ± 0.1863 | -0.03 ± 1.68 | 0.0650 ± 0.0427 | 0.1044 ± 0.0688 | 0.1479 ± 0.0262 | 0.4652 ± 0.1015 | 51.5 ± 33.8 | 1/4 |
| xgboost | 0.0070 ± 0.0864 | 0.27 ± 1.26 | 0.0350 ± 0.0444 | 0.0581 ± 0.0607 | 0.1391 ± 0.0391 | 0.4632 ± 0.0887 | 27.8 ± 35.2 | 2/4 |
| majority_baseline | -0.6671 ± 0.1508 | -4.23 ± 1.73 | 0.9965 ± 0.0069 | 0.7012 ± 0.1434 | 0.1888 ± 0.0172 | 0.3966 ± 0.0487 | 789.2 ± 5.5 | 0/4 |
| momentum_baseline | -0.7944 ± 0.0334 | -5.79 ± 0.53 | 0.9965 ± 0.0069 | 0.8184 ± 0.0391 | 0.2703 ± 0.0187 | 0.3582 ± 0.0382 | 789.2 ± 5.5 | 0/4 |

This is one seed. The comparison against the frozen MTST v4 evidence, the
economic references and the other seeds is `python -m nn.benchmark_compare`;
no verdict is stated here, because a single seed cannot support one.

P2a is an adaptive follow-up designed after prior outer-validation results were
seen. These outer folds are research evidence, not a pristine out-of-sample
test, and nothing here is a claim about live profitability. The sealed test
block remains unopened.
