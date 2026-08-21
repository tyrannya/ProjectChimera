# P2b — does causal market structure add information beyond OHLCV14?

Three information sets, three untuned models, four temporal outer folds, one
sample universe. `ohlcv14` is P2a's control re-run under this code path;
`smc_v1` is causal market structure alone (`docs/smc_v1.md`);
`ohlcv14_plus_smc_v1` is both.

**Research contract:** `btc-usdt-1h-gen1` (generation 1), semantic identity
`sha256:dca6d0a891a257197a7c8aecec04fdba0a0b3009cfe93c5f3a397d458ab4a1de`.

**Sealed test block:** everything at or after `2025-08-27T23:00:00+00:00`.
Not planned over, not fitted on, not selected on, not scored.
**Styx was not opened.**

**Statistical unit:** one temporal outer period per fold, four in total. These
estimators are deterministic given their inputs, so no seed replication appears
anywhere below:
a second seed would copy this evidence rather than add to it.

**Adaptive status:** P2b was designed after P2a's outer results had been seen. Its
outer blocks are adaptive research evidence, not a pristine out-of-sample test.

## Sample-universe parity

every cell scored the same outer rows; a difference between two cells can only be the information set or the model. Checked across all 9 cells:

- research contract and its hash
- snapshot identity and semantic hashes
- fold sizes and periods
- per-fold sample-index hashes from the alignment proof
- label horizon and costs
- threshold grid, objective and trade floor
- combined feature-spec hash
- majority and momentum baseline outer reports
- CASH and buy-and-hold economic references

## Independent recomputation

Every reported trading and classification number was rebuilt from the 9 cells'
persisted `outer_predictions.parquet` files: **36 cell-folds checked, 0 mismatches**.

- not recomputed: annualised_sharpe and candle_max_drawdown need the candle price path, which the prediction file does not carry

## Per-fold outer validation

### logistic_regression

| information set | fold | outer period | samples | thr | trades | exposure | turnover | gross | costs | **net** | max DD | ann. Sharpe | per-trade Sharpe | win rate | profit factor | macro F1 | dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ohlcv14` | 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.62 | 28 | 0.0354 | 56.0 | +0.052144 | 0.0560 | **-0.006932** | 0.0588 | -0.0490 | -0.0091 | 0.5357 | 0.9747 | 0.1808 | 0.5686 |
| `ohlcv14` | 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.66 | 11 | 0.0139 | 22.0 | +0.086935 | 0.0220 | **+0.066156** | 0.0196 | 1.9999 | 0.5019 | 0.6364 | 3.8474 | 0.1305 | 0.6842 |
| `ohlcv14` | 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.56 | 75 | 0.0947 | 150.0 | +0.284734 | 0.1500 | **+0.130769** | 0.0891 | 1.0856 | 0.1002 | 0.4800 | 1.3600 | 0.1428 | 0.5098 |
| `ohlcv14` | 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.56 | 110 | 0.1389 | 220.0 | +0.067557 | 0.2200 | **-0.157307** | 0.3166 | -1.0412 | -0.0754 | 0.4909 | 0.8183 | 0.1441 | 0.5093 |
| `smc_v1` | 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.84 | 119 | 0.1503 | 238.0 | +0.111805 | 0.2380 | **-0.128245** | 0.2051 | -1.3208 | -0.0771 | 0.4202 | 0.7553 | 0.1984 | 0.4041 |
| `smc_v1` | 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.88 | 22 | 0.0278 | 44.0 | +0.017362 | 0.0440 | **-0.027936** | 0.0526 | -0.7273 | -0.0959 | 0.3636 | 0.7536 | 0.1309 | 0.5000 |
| `smc_v1` | 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.82 | 32 | 0.0404 | 64.0 | +0.092196 | 0.0640 | **+0.027249** | 0.0470 | 0.6866 | 0.0961 | 0.4688 | 1.3257 | 0.1247 | 0.5098 |
| `smc_v1` | 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.74 | 152 | 0.1919 | 304.0 | +0.118235 | 0.3040 | **-0.187203** | 0.2788 | -1.2390 | -0.0729 | 0.4803 | 0.8044 | 0.1457 | 0.4440 |
| `ohlcv14_plus_smc_v1` | 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.90 | 67 | 0.0846 | 134.0 | +0.023097 | 0.1340 | **-0.113659** | 0.1837 | -1.2746 | -0.0961 | 0.4478 | 0.6819 | 0.1848 | 0.4158 |
| `ohlcv14_plus_smc_v1` | 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.90 | 22 | 0.0278 | 44.0 | +0.114072 | 0.0440 | **+0.069826** | 0.0526 | 1.0787 | 0.2070 | 0.5455 | 1.7951 | 0.1321 | 0.5484 |
| `ohlcv14_plus_smc_v1` | 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.84 | 25 | 0.0316 | 50.0 | +0.024655 | 0.0500 | **-0.026093** | 0.0474 | -0.3730 | -0.1069 | 0.4400 | 0.7445 | 0.1228 | 0.5000 |
| `ohlcv14_plus_smc_v1` | 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.82 | 55 | 0.0694 | 110.0 | +0.064037 | 0.1100 | **-0.051878** | 0.1134 | -0.5009 | -0.0508 | 0.4545 | 0.8568 | 0.1183 | 0.4250 |

### lightgbm

| information set | fold | outer period | samples | thr | trades | exposure | turnover | gross | costs | **net** | max DD | ann. Sharpe | per-trade Sharpe | win rate | profit factor | macro F1 | dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ohlcv14` | 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.70 | 30 | 0.0379 | 60.0 | +0.020906 | 0.0600 | **-0.046515** | 0.1085 | -0.5360 | -0.0533 | 0.3333 | 0.8317 | 0.1774 | 0.3433 |
| `ohlcv14` | 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.60 | 96 | 0.1212 | 192.0 | -0.007094 | 0.1920 | **-0.189062** | 0.1978 | -1.5579 | -0.1414 | 0.4688 | 0.6876 | 0.1623 | 0.4530 |
| `ohlcv14` | 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.66 | 21 | 0.0265 | 42.0 | +0.019137 | 0.0420 | **-0.025714** | 0.0744 | -0.4007 | -0.0607 | 0.4286 | 0.8336 | 0.1228 | 0.4737 |
| `ohlcv14` | 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.62 | 59 | 0.0745 | 118.0 | +0.353907 | 0.1180 | **+0.256085** | 0.0369 | 2.3691 | 0.2468 | 0.5593 | 2.1441 | 0.1290 | 0.5909 |
| `smc_v1` | 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.70 | 24 | 0.0303 | 48.0 | +0.211642 | 0.0480 | **+0.170439** | 0.0378 | 2.7954 | 0.2993 | 0.5833 | 3.2182 | 0.1766 | 0.5000 |
| `smc_v1` | 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.64 | 79 | 0.0997 | 158.0 | +0.093188 | 0.1580 | **-0.069390** | 0.1460 | -0.5959 | -0.0606 | 0.4051 | 0.8270 | 0.1475 | 0.4242 |
| `smc_v1` | 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.60 | 119 | 0.1503 | 238.0 | +0.138986 | 0.2380 | **-0.101892** | 0.1492 | -0.8346 | -0.0697 | 0.5126 | 0.8185 | 0.1510 | 0.5121 |
| `smc_v1` | 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.58 | 146 | 0.1843 | 292.0 | +0.447275 | 0.2920 | **+0.150413** | 0.1421 | 1.1336 | 0.0734 | 0.5137 | 1.2274 | 0.1559 | 0.5356 |
| `ohlcv14_plus_smc_v1` | 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.64 | 55 | 0.0694 | 110.0 | -0.123702 | 0.1100 | **-0.220954** | 0.2314 | -2.3223 | -0.1787 | 0.4545 | 0.5193 | 0.1851 | 0.3832 |
| `ohlcv14_plus_smc_v1` | 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.60 | 150 | 0.1894 | 300.0 | +0.127769 | 0.3000 | **-0.171190** | 0.1900 | -1.1753 | -0.0800 | 0.4867 | 0.7962 | 0.1814 | 0.5060 |
| `ohlcv14_plus_smc_v1` | 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.66 | 20 | 0.0253 | 40.0 | +0.011486 | 0.0400 | **-0.028819** | 0.0581 | -0.6251 | -0.1655 | 0.4000 | 0.6675 | 0.1203 | 0.4000 |
| `ohlcv14_plus_smc_v1` | 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.66 | 36 | 0.0455 | 72.0 | +0.074995 | 0.0720 | **-0.000090** | 0.0912 | 0.0727 | 0.0063 | 0.4722 | 1.0159 | 0.1187 | 0.5000 |

### xgboost

| information set | fold | outer period | samples | thr | trades | exposure | turnover | gross | costs | **net** | max DD | ann. Sharpe | per-trade Sharpe | win rate | profit factor | macro F1 | dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ohlcv14` | 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.64 | 80 | 0.1010 | 160.0 | +0.062295 | 0.1600 | **-0.101571** | 0.1478 | -1.0454 | -0.0802 | 0.4625 | 0.7578 | 0.1968 | 0.4405 |
| `ohlcv14` | 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.72 | 11 | 0.0139 | 22.0 | +0.056220 | 0.0220 | **+0.032526** | 0.0393 | 0.6611 | 0.1482 | 0.4545 | 1.5588 | 0.1293 | 0.3913 |
| `ohlcv14` | 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.70 | 4 | 0.0051 | 8.0 | -0.000356 | 0.0080 | **-0.008425** | 0.0144 | -0.3705 | -0.2624 | 0.5000 | 0.4235 | 0.1172 | 0.4286 |
| `ohlcv14` | 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.68 | 16 | 0.0202 | 32.0 | +0.134772 | 0.0320 | **+0.105629** | 0.0310 | 1.8526 | 0.3872 | 0.6875 | 3.2504 | 0.1129 | 0.5926 |
| `smc_v1` | 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.70 | 20 | 0.0253 | 40.0 | +0.174873 | 0.0400 | **+0.137677** | 0.0473 | 2.1783 | 0.2756 | 0.4500 | 3.1901 | 0.1741 | 0.4375 |
| `smc_v1` | 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.68 | 29 | 0.0366 | 58.0 | +0.028122 | 0.0580 | **-0.031267** | 0.0838 | -0.5211 | -0.0892 | 0.4828 | 0.7992 | 0.1350 | 0.4630 |
| `smc_v1` | 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.62 | 49 | 0.0619 | 98.0 | -0.017978 | 0.0980 | **-0.112438** | 0.1293 | -1.6510 | -0.2072 | 0.4490 | 0.5518 | 0.1273 | 0.4000 |
| `smc_v1` | 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.70 | 20 | 0.0253 | 40.0 | +0.118246 | 0.0400 | **+0.079851** | 0.0306 | 1.6925 | 0.3360 | 0.5500 | 2.5737 | 0.1131 | 0.5806 |
| `ohlcv14_plus_smc_v1` | 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.66 | 43 | 0.0543 | 86.0 | -0.027001 | 0.0860 | **-0.112009** | 0.2010 | -1.3738 | -0.1605 | 0.3023 | 0.5885 | 0.1794 | 0.3333 |
| `ohlcv14_plus_smc_v1` | 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.70 | 20 | 0.0253 | 40.0 | -0.042640 | 0.0400 | **-0.080644** | 0.0865 | -1.5729 | -0.3588 | 0.4000 | 0.4074 | 0.1310 | 0.4333 |
| `ohlcv14_plus_smc_v1` | 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.64 | 26 | 0.0328 | 52.0 | +0.163235 | 0.0520 | **+0.113964** | 0.0493 | 1.7825 | 0.2706 | 0.6923 | 2.3748 | 0.1259 | 0.5490 |
| `ohlcv14_plus_smc_v1` | 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.62 | 69 | 0.0871 | 138.0 | +0.191588 | 0.1380 | **+0.047844** | 0.1539 | 0.5222 | 0.0547 | 0.5362 | 1.1643 | 0.1313 | 0.5748 |

## Across the four temporal folds

| model | information set | net mean | net std | net min | net median | net max | positive folds | Sharpe mean | max DD mean | exposure mean | macro F1 mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | `ohlcv14` | +0.008171 | 0.123833 | -0.157307 | +0.029612 | +0.130769 | **2 of 4** | 0.4988 | 0.1210 | 0.0707 | 0.1495 |
| logistic_regression | `smc_v1` | -0.079034 | 0.096660 | -0.187203 | -0.078090 | +0.027249 | **1 of 4** | -0.6501 | 0.1459 | 0.1026 | 0.1499 |
| logistic_regression | `ohlcv14_plus_smc_v1` | -0.030451 | 0.076283 | -0.113659 | -0.038985 | +0.069826 | **1 of 4** | -0.2675 | 0.0993 | 0.0534 | 0.1395 |
| lightgbm | `ohlcv14` | -0.001302 | 0.186317 | -0.189062 | -0.036115 | +0.256085 | **1 of 4** | -0.0314 | 0.1044 | 0.0650 | 0.1479 |
| lightgbm | `smc_v1` | +0.037393 | 0.142919 | -0.101892 | +0.040511 | +0.170439 | **2 of 4** | 0.6246 | 0.1188 | 0.1162 | 0.1578 |
| lightgbm | `ohlcv14_plus_smc_v1` | -0.105263 | 0.107449 | -0.220954 | -0.100005 | -0.000090 | **0 of 4** | -1.0125 | 0.1427 | 0.0824 | 0.1514 |
| xgboost | `ohlcv14` | +0.007040 | 0.086419 | -0.101571 | +0.012050 | +0.105629 | **2 of 4** | 0.2745 | 0.0581 | 0.0350 | 0.1391 |
| xgboost | `smc_v1` | +0.018456 | 0.111935 | -0.112438 | +0.024292 | +0.137677 | **2 of 4** | 0.4247 | 0.0727 | 0.0373 | 0.1374 |
| xgboost | `ohlcv14_plus_smc_v1` | -0.007711 | 0.106597 | -0.112009 | -0.016400 | +0.113964 | **2 of 4** | -0.1605 | 0.1227 | 0.0499 | 0.1419 |

## Incremental value of market structure

Per model, per fold: the information set minus the `ohlcv14` control on the same rows.

### logistic_regression: `smc_v1` − `ohlcv14`

| fold | outer period | Δ net | Δ ann. Sharpe | Δ max DD (lower is better) | Δ exposure | Δ trades | Δ macro F1 | Δ dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | **-0.121313** | -1.2718 | +0.1463 | +0.1149 | +91 | +0.0176 | -0.1645 |
| 1 | 2023-09-24 → 2024-04-12 | **-0.094092** | -2.7272 | +0.0330 | +0.0139 | +11 | +0.0004 | -0.1842 |
| 2 | 2024-04-12 → 2024-10-30 | **-0.103520** | -0.3990 | -0.0421 | -0.0543 | -43 | -0.0181 | +0.0000 |
| 3 | 2024-10-30 → 2025-05-19 | **-0.029896** | -0.1978 | -0.0378 | +0.0530 | +42 | +0.0016 | -0.0653 |

Net return improved in **0 of 4** temporal folds (mean Δ -0.087205, min -0.121313, max -0.029896).

**Verdict:** no improvement in any temporal fold — negative evidence.

### logistic_regression: `ohlcv14_plus_smc_v1` − `ohlcv14`

| fold | outer period | Δ net | Δ ann. Sharpe | Δ max DD (lower is better) | Δ exposure | Δ trades | Δ macro F1 | Δ dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | **-0.106727** | -1.2256 | +0.1249 | +0.0492 | +39 | +0.0040 | -0.1528 |
| 1 | 2023-09-24 → 2024-04-12 | **+0.003670** | -0.9212 | +0.0330 | +0.0139 | +11 | +0.0016 | -0.1358 |
| 2 | 2024-04-12 → 2024-10-30 | **-0.156862** | -1.4586 | -0.0417 | -0.0631 | -50 | -0.0200 | -0.0098 |
| 3 | 2024-10-30 → 2025-05-19 | **+0.105429** | +0.5403 | -0.2032 | -0.0695 | -55 | -0.0258 | -0.0843 |

Net return improved in **2 of 4** temporal folds (mean Δ -0.038623, min -0.156862, max +0.105429).

**Verdict:** improvement in two of four temporal folds — regime-dependent, inconclusive.

### lightgbm: `smc_v1` − `ohlcv14`

| fold | outer period | Δ net | Δ ann. Sharpe | Δ max DD (lower is better) | Δ exposure | Δ trades | Δ macro F1 | Δ dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | **+0.216954** | +3.3314 | -0.0707 | -0.0076 | -6 | -0.0008 | +0.1567 |
| 1 | 2023-09-24 → 2024-04-12 | **+0.119672** | +0.9620 | -0.0518 | -0.0215 | -17 | -0.0148 | -0.0288 |
| 2 | 2024-04-12 → 2024-10-30 | **-0.076178** | -0.4339 | +0.0748 | +0.1238 | +98 | +0.0282 | +0.0384 |
| 3 | 2024-10-30 → 2025-05-19 | **-0.105672** | -1.2355 | +0.1052 | +0.1098 | +87 | +0.0269 | -0.0553 |

Net return improved in **2 of 4** temporal folds (mean Δ +0.038694, min -0.105672, max +0.216954).

**Verdict:** improvement in two of four temporal folds — regime-dependent, inconclusive.

### lightgbm: `ohlcv14_plus_smc_v1` − `ohlcv14`

| fold | outer period | Δ net | Δ ann. Sharpe | Δ max DD (lower is better) | Δ exposure | Δ trades | Δ macro F1 | Δ dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | **-0.174439** | -1.7863 | +0.1229 | +0.0315 | +25 | +0.0077 | +0.0399 |
| 1 | 2023-09-24 → 2024-04-12 | **+0.017872** | +0.3826 | -0.0078 | +0.0682 | +54 | +0.0191 | +0.0530 |
| 2 | 2024-04-12 → 2024-10-30 | **-0.003105** | -0.2244 | -0.0163 | -0.0012 | -1 | -0.0025 | -0.0737 |
| 3 | 2024-10-30 → 2025-05-19 | **-0.256175** | -2.2964 | +0.0543 | -0.0290 | -23 | -0.0103 | -0.0909 |

Net return improved in **1 of 4** temporal folds (mean Δ -0.103962, min -0.256175, max +0.017872).

**Verdict:** improvement in one of four temporal folds — weak evidence against.

### xgboost: `smc_v1` − `ohlcv14`

| fold | outer period | Δ net | Δ ann. Sharpe | Δ max DD (lower is better) | Δ exposure | Δ trades | Δ macro F1 | Δ dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | **+0.239248** | +3.2237 | -0.1005 | -0.0757 | -60 | -0.0227 | -0.0030 |
| 1 | 2023-09-24 → 2024-04-12 | **-0.063793** | -1.1822 | +0.0445 | +0.0227 | +18 | +0.0057 | +0.0717 |
| 2 | 2024-04-12 → 2024-10-30 | **-0.104013** | -1.2805 | +0.1149 | +0.0568 | +45 | +0.0101 | -0.0286 |
| 3 | 2024-10-30 → 2025-05-19 | **-0.025778** | -0.1601 | -0.0004 | +0.0051 | +4 | +0.0002 | -0.0120 |

Net return improved in **1 of 4** temporal folds (mean Δ +0.011416, min -0.104013, max +0.239248).

**Verdict:** improvement in one of four temporal folds — weak evidence against.

### xgboost: `ohlcv14_plus_smc_v1` − `ohlcv14`

| fold | outer period | Δ net | Δ ann. Sharpe | Δ max DD (lower is better) | Δ exposure | Δ trades | Δ macro F1 | Δ dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | **-0.010438** | -0.3284 | +0.0532 | -0.0467 | -37 | -0.0174 | -0.1072 |
| 1 | 2023-09-24 → 2024-04-12 | **-0.113170** | -2.2340 | +0.0472 | +0.0114 | +9 | +0.0017 | +0.0420 |
| 2 | 2024-04-12 → 2024-10-30 | **+0.122389** | +2.1530 | +0.0349 | +0.0277 | +22 | +0.0087 | +0.1204 |
| 3 | 2024-10-30 → 2025-05-19 | **-0.057785** | -1.3304 | +0.1229 | +0.0669 | +53 | +0.0184 | -0.0178 |

Net return improved in **1 of 4** temporal folds (mean Δ -0.014751, min -0.113170, max +0.122389).

**Verdict:** improvement in one of four temporal folds — weak evidence against.

## Long / short attribution

Additive decomposition of the realised trades a cell took. These are the two halves of
one reported result, not two standalone strategies: neither side was selected for, and
neither could have been traded on its own without the threshold that produced both.

| model | information set | long trades | long hit | long mean net | short trades | short hit | short mean net |
| --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | `ohlcv14` | 150 | 0.5133 | +0.000794 | 74 | 0.4730 | -0.001024 |
| logistic_regression | `smc_v1` | 188 | 0.4468 | -0.000957 | 137 | 0.4526 | -0.000952 |
| logistic_regression | `ohlcv14_plus_smc_v1` | 96 | 0.4688 | -0.000191 | 73 | 0.4521 | -0.001285 |
| lightgbm | `ohlcv14` | 135 | 0.5111 | +0.001070 | 71 | 0.3944 | -0.002388 |
| lightgbm | `smc_v1` | 279 | 0.4982 | +0.000629 | 89 | 0.4831 | -0.000228 |
| lightgbm | `ohlcv14_plus_smc_v1` | 203 | 0.4532 | -0.001175 | 58 | 0.5345 | -0.003327 |
| xgboost | `ohlcv14` | 83 | 0.4699 | +0.000589 | 28 | 0.5714 | -0.000641 |
| xgboost | `smc_v1` | 98 | 0.5204 | +0.001368 | 20 | 0.2500 | -0.003339 |
| xgboost | `ohlcv14_plus_smc_v1` | 119 | 0.4370 | -0.001112 | 39 | 0.6154 | +0.002603 |

## Economic references

CASH and buy-and-hold over the same outer windows. **Reference only.** No feature,
model or threshold in P2b was selected using them, and buy-and-hold is fully
exposed for the whole window while every cell above is exposed only while a
position is open — the two are not comparable as strategies.

| fold | outer period | CASH net | buy-and-hold net | buy-and-hold max DD |
| --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | +0.000000 | +0.182520 | 0.2085 |
| 1 | 2023-09-24 → 2024-04-12 | +0.000000 | +1.659405 | 0.2019 |
| 2 | 2024-04-12 → 2024-10-30 | +0.000000 | +0.103569 | 0.3057 |
| 3 | 2024-10-30 → 2025-05-19 | +0.000000 | +0.479817 | 0.3096 |
