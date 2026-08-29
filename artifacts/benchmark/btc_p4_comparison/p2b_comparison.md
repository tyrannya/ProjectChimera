# P4 — do `derivatives_v1` or `ohlcv14_plus_derivatives_v1` add information beyond `ohlcv14`?

**Research question:** does causal derivatives_v1, alone or combined with OHLCV14, add usable information beyond OHLCV14?

3 information sets, 3 untuned models, four temporal outer
folds, one sample universe. `ohlcv14` is the control, re-run under this code
path rather than copied. The other arms are:

- `derivatives_v1`
- `ohlcv14_plus_derivatives_v1`

**Research contract:** `btc-usdt-1h-gen1` (generation 1), semantic identity
`sha256:dca6d0a891a257197a7c8aecec04fdba0a0b3009cfe93c5f3a397d458ab4a1de`.

**Sealed test block:** everything at or after `2025-08-27T23:00:00+00:00`.
Not planned over, not fitted on, not selected on, not scored.
**Styx was not opened.**

**Statistical unit:** one temporal outer period per fold, four in total. These
estimators are deterministic given their inputs, so no seed replication appears
anywhere below:
a second seed would copy this evidence rather than add to it.

**Adaptive status:** adaptive on the four exploratory blocks; single-region and never-sealed on the holdout. P4 cannot produce confirmatory evidence..

## Sample-universe parity

every cell scored the same outer rows; a difference between two cells can only be the information set or the model. Checked across all 9 cells:

- research contract and its hash
- snapshot identity and semantic hashes
- trade-source identity, aggregation spec and microstructure spec, where a checkpoint has one
- the same absence of a trade source, where it has none
- derivatives-source identity, derivatives spec and the sample universe, where a checkpoint has one
- the same absence of a derivatives source, where it has none
- fold sizes and periods
- the research checkpoint each cell says it answers
- per-fold sample-index hashes from the alignment proof
- label horizon and costs
- threshold grid, objective and trade floor
- combined feature-spec hash
- majority and momentum baseline outer reports
- CASH and buy-and-hold economic references
- the numerical environment the cells were fitted under, where they record one

## Persisted rows are the planned rows

Each cell's persisted `row_index` sequence was compared against the outer sample
index its fold plan selected before anything was fitted — count, uniqueness,
strict order, first and last row, and a SHA-256 over the exact `int64` bytes:

- **9 cells, 36 folds, 167427 rows checked**
- missing folds: **0**
- unplanned folds: **0**
- non integer row index: **0**
- duplicate rows: **0**
- unsorted rows: **0**
- count mismatches: **0**
- sample index hash mismatches: **0**
- first last mismatches: **0**
- cross fold rows: **0**
- snapshot value mismatches: **0**

A wrong sample chosen consistently — every row's own timestamp, label and return
copied correctly from the snapshot — passes the anchoring check below and fails
this one.


## Independent recomputation

Every reported trading and classification number was rebuilt from the 9 cells'
persisted `outer_predictions.parquet` files: **36 cell-folds checked, 0 mismatches**.

- not recomputed: trading.annualised_sharpe, trading.candle_max_drawdown and trading.elapsed_intervals need the candle price path, which the prediction file does not carry
- not recomputed: trading.sharpe_basis, trading.per_trade_sharpe_reason and trading.annualised_sharpe_reason are prose, not measurements

## Per-fold outer validation

### logistic_regression

| information set | fold | outer period | samples | thr | trades | exposure | turnover | gross | costs | **net** | max DD | ann. Sharpe | per-trade Sharpe | win rate | profit factor | macro F1 | dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ohlcv14` | 0 | 2023-03-04 → 2023-09-24 | 4443 | 0.66 | 30 | 0.0379 | 60.0 | +0.147667 | 0.0600 | **+0.088301** | 0.0399 | 1.3514 | 0.2032 | 0.5000 | 2.0676 | 0.1805 | 0.4630 |
| `ohlcv14` | 1 | 2023-09-24 → 2024-04-12 | 4656 | 0.66 | 22 | 0.0278 | 44.0 | +0.076958 | 0.0440 | **+0.030567** | 0.0720 | 0.5632 | 0.0914 | 0.5000 | 1.3037 | 0.1345 | 0.5946 |
| `ohlcv14` | 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.60 | 48 | 0.0606 | 96.0 | +0.257915 | 0.0960 | **+0.164339** | 0.0776 | 1.4740 | 0.1661 | 0.4375 | 1.6249 | 0.1296 | 0.4419 |
| `ohlcv14` | 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.58 | 89 | 0.1124 | 178.0 | +0.412639 | 0.1780 | **+0.247617** | 0.1650 | 1.8337 | 0.1523 | 0.5393 | 1.4610 | 0.1358 | 0.5030 |
| `derivatives_v1` | 0 | 2023-03-04 → 2023-09-24 | 4443 | 0.68 | 16 | 0.0202 | 32.0 | +0.004751 | 0.0320 | **-0.027344** | 0.0442 | -0.7850 | -0.2200 | 0.2500 | 0.5346 | 0.1722 | 0.2917 |
| `derivatives_v1` | 1 | 2023-09-24 → 2024-04-12 | 4656 | 0.60 | 65 | 0.0821 | 130.0 | +0.129426 | 0.1300 | **-0.010160** | 0.0786 | -0.0135 | -0.0005 | 0.4308 | 0.9983 | 0.1427 | 0.4259 |
| `derivatives_v1` | 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.64 | 3 | 0.0038 | 6.0 | -0.030400 | 0.0060 | **-0.036235** | 0.0362 | -2.2053 | -0.7296 | 0.0000 | 0.0000 | 0.1160 | 0.0000 |
| `derivatives_v1` | 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.58 | 30 | 0.0379 | 60.0 | +0.085018 | 0.0600 | **+0.021658** | 0.0863 | 0.3593 | 0.0527 | 0.4667 | 1.1629 | 0.1153 | 0.5106 |
| `ohlcv14_plus_derivatives_v1` | 0 | 2023-03-04 → 2023-09-24 | 4443 | 0.74 | 27 | 0.0341 | 54.0 | +0.073053 | 0.0540 | **+0.017871** | 0.0404 | 0.3980 | 0.0693 | 0.5185 | 1.2166 | 0.1806 | 0.5106 |
| `ohlcv14_plus_derivatives_v1` | 1 | 2023-09-24 → 2024-04-12 | 4656 | 0.68 | 50 | 0.0631 | 100.0 | -0.018866 | 0.1000 | **-0.115990** | 0.1315 | -1.6152 | -0.1807 | 0.4400 | 0.6009 | 0.1418 | 0.5000 |
| `ohlcv14_plus_derivatives_v1` | 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.72 | 7 | 0.0088 | 14.0 | -0.060606 | 0.0140 | **-0.073195** | 0.0732 | -2.6186 | -0.5874 | 0.2857 | 0.2149 | 0.1168 | 0.2000 |
| `ohlcv14_plus_derivatives_v1` | 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.62 | 100 | 0.1263 | 200.0 | +0.078212 | 0.2000 | **-0.128754** | 0.2137 | -0.8468 | -0.0680 | 0.4900 | 0.8335 | 0.1362 | 0.4444 |

### lightgbm

| information set | fold | outer period | samples | thr | trades | exposure | turnover | gross | costs | **net** | max DD | ann. Sharpe | per-trade Sharpe | win rate | profit factor | macro F1 | dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ohlcv14` | 0 | 2023-03-04 → 2023-09-24 | 4443 | 0.82 | 3 | 0.0038 | 6.0 | -0.010264 | 0.0060 | **-0.017623** | 0.0468 | -0.3359 | -0.1424 | 0.3333 | 0.6530 | 0.1699 | 0.2500 |
| `ohlcv14` | 1 | 2023-09-24 → 2024-04-12 | 4656 | 0.54 | 280 | 0.3535 | 560.0 | +0.473971 | 0.5600 | **-0.111839** | 0.3189 | -0.4898 | -0.0200 | 0.5143 | 0.9420 | 0.2258 | 0.4501 |
| `ohlcv14` | 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.56 | 170 | 0.2146 | 340.0 | -0.420514 | 0.3400 | **-0.542260** | 0.5549 | -4.8983 | -0.2989 | 0.3765 | 0.4249 | 0.1641 | 0.3737 |
| `ohlcv14` | 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.66 | 28 | 0.0354 | 56.0 | +0.231380 | 0.0560 | **+0.185147** | 0.0358 | 2.1849 | 0.3235 | 0.5714 | 2.4599 | 0.1165 | 0.5490 |
| `derivatives_v1` | 0 | 2023-03-04 → 2023-09-24 | 4443 | 0.62 | 96 | 0.1212 | 192.0 | +0.025134 | 0.1920 | **-0.158054** | 0.1862 | -1.7801 | -0.1697 | 0.3854 | 0.5937 | 0.2032 | 0.3767 |
| `derivatives_v1` | 1 | 2023-09-24 → 2024-04-12 | 4656 | 0.66 | 22 | 0.0278 | 44.0 | +0.050001 | 0.0440 | **+0.003675** | 0.0536 | 0.1165 | 0.0182 | 0.5000 | 1.0537 | 0.1354 | 0.4444 |
| `derivatives_v1` | 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.64 | 9 | 0.0114 | 18.0 | -0.076999 | 0.0180 | **-0.091735** | 0.0917 | -2.0389 | -0.8015 | 0.1111 | 0.0139 | 0.1178 | 0.2174 |
| `derivatives_v1` | 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.56 | 39 | 0.0492 | 78.0 | +0.007738 | 0.0780 | **-0.069995** | 0.1167 | -1.0032 | -0.1662 | 0.4359 | 0.6428 | 0.1156 | 0.3784 |
| `ohlcv14_plus_derivatives_v1` | 0 | 2023-03-04 → 2023-09-24 | 4443 | 0.80 | 16 | 0.0202 | 32.0 | -0.052495 | 0.0320 | **-0.082431** | 0.0827 | -1.7089 | -0.4050 | 0.4375 | 0.2359 | 0.1738 | 0.5000 |
| `ohlcv14_plus_derivatives_v1` | 1 | 2023-09-24 → 2024-04-12 | 4656 | 0.64 | 73 | 0.0922 | 146.0 | +0.126180 | 0.1460 | **-0.032540** | 0.1412 | -0.2072 | -0.0140 | 0.4795 | 0.9563 | 0.1535 | 0.4839 |
| `ohlcv14_plus_derivatives_v1` | 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.60 | 106 | 0.1338 | 212.0 | +0.016792 | 0.2120 | **-0.189804** | 0.2718 | -1.5762 | -0.1093 | 0.4434 | 0.7228 | 0.1479 | 0.4493 |
| `ohlcv14_plus_derivatives_v1` | 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.66 | 28 | 0.0354 | 56.0 | -0.008353 | 0.0560 | **-0.066293** | 0.1138 | -0.6912 | -0.1311 | 0.4286 | 0.7290 | 0.1171 | 0.5000 |

### xgboost

| information set | fold | outer period | samples | thr | trades | exposure | turnover | gross | costs | **net** | max DD | ann. Sharpe | per-trade Sharpe | win rate | profit factor | macro F1 | dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ohlcv14` | 0 | 2023-03-04 → 2023-09-24 | 4443 | 0.78 | 14 | 0.0177 | 28.0 | +0.008585 | 0.0280 | **-0.020017** | 0.0397 | -0.6351 | -0.1252 | 0.4286 | 0.6759 | 0.1736 | 0.4286 |
| `ohlcv14` | 1 | 2023-09-24 → 2024-04-12 | 4656 | 0.70 | 15 | 0.0189 | 30.0 | +0.082688 | 0.0300 | **+0.051490** | 0.0351 | 0.8818 | 0.1882 | 0.5333 | 1.7717 | 0.1307 | 0.5455 |
| `ohlcv14` | 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.66 | 17 | 0.0215 | 34.0 | -0.072296 | 0.0340 | **-0.103661** | 0.1162 | -2.0214 | -0.3360 | 0.3529 | 0.4029 | 0.1191 | 0.3478 |
| `ohlcv14` | 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.64 | 42 | 0.0530 | 84.0 | +0.046976 | 0.0840 | **-0.043012** | 0.0921 | -0.3814 | -0.0478 | 0.4286 | 0.8728 | 0.1201 | 0.4810 |
| `derivatives_v1` | 0 | 2023-03-04 → 2023-09-24 | 4443 | 0.72 | 10 | 0.0126 | 20.0 | -0.012194 | 0.0200 | **-0.032045** | 0.0346 | -1.0502 | -0.3820 | 0.4000 | 0.2861 | 0.1725 | 0.4375 |
| `derivatives_v1` | 1 | 2023-09-24 → 2024-04-12 | 4656 | 0.60 | 82 | 0.1035 | 164.0 | +0.130520 | 0.1640 | **-0.041218** | 0.1310 | -0.3421 | -0.0281 | 0.4634 | 0.9227 | 0.1611 | 0.4434 |
| `derivatives_v1` | 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.60 | 19 | 0.0240 | 38.0 | -0.126861 | 0.0380 | **-0.155042** | 0.1777 | -3.0588 | -0.4920 | 0.3684 | 0.1928 | 0.1227 | 0.3774 |
| `derivatives_v1` | 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.62 | 11 | 0.0139 | 22.0 | +0.019794 | 0.0220 | **-0.002499** | 0.0234 | -0.0866 | -0.0261 | 0.4545 | 0.9416 | 0.1105 | 0.4762 |
| `ohlcv14_plus_derivatives_v1` | 0 | 2023-03-04 → 2023-09-24 | 4443 | 0.78 | 18 | 0.0227 | 36.0 | -0.000519 | 0.0360 | **-0.036194** | 0.0362 | -1.2430 | -0.3369 | 0.3889 | 0.4410 | 0.1723 | 0.2692 |
| `ohlcv14_plus_derivatives_v1` | 1 | 2023-09-24 → 2024-04-12 | 4656 | 0.70 | 20 | 0.0253 | 40.0 | +0.053201 | 0.0400 | **+0.005028** | 0.0827 | 0.1357 | 0.0222 | 0.6000 | 1.0754 | 0.1341 | 0.5385 |
| `ohlcv14_plus_derivatives_v1` | 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.62 | 72 | 0.0909 | 144.0 | -0.067667 | 0.1440 | **-0.196729** | 0.2469 | -2.1866 | -0.2080 | 0.4028 | 0.5605 | 0.1374 | 0.4326 |
| `ohlcv14_plus_derivatives_v1` | 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.68 | 15 | 0.0189 | 30.0 | +0.011397 | 0.0300 | **-0.019946** | 0.0738 | -0.2622 | -0.0838 | 0.4000 | 0.8043 | 0.1123 | 0.4412 |

## Across the four temporal folds

| model | information set | net mean | net std | net min | net median | net max | positive folds | Sharpe mean | max DD mean | exposure mean | macro F1 mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | `ohlcv14` | +0.132706 | 0.094180 | +0.030567 | +0.126320 | +0.247617 | **4 of 4** | 1.3056 | 0.0886 | 0.0597 | 0.1451 |
| logistic_regression | `derivatives_v1` | -0.013020 | 0.025527 | -0.036235 | -0.018752 | +0.021658 | **1 of 4** | -0.6611 | 0.0613 | 0.0360 | 0.1366 |
| logistic_regression | `ohlcv14_plus_derivatives_v1` | -0.075017 | 0.066327 | -0.128754 | -0.094592 | +0.017871 | **1 of 4** | -1.1706 | 0.1147 | 0.0581 | 0.1439 |
| lightgbm | `ohlcv14` | -0.121644 | 0.306570 | -0.542260 | -0.064731 | +0.185147 | **1 of 4** | -0.8848 | 0.2391 | 0.1518 | 0.1691 |
| lightgbm | `derivatives_v1` | -0.079027 | 0.066653 | -0.158054 | -0.080865 | +0.003675 | **1 of 4** | -1.1764 | 0.1120 | 0.0524 | 0.1430 |
| lightgbm | `ohlcv14_plus_derivatives_v1` | -0.092767 | 0.067949 | -0.189804 | -0.074362 | -0.032540 | **0 of 4** | -1.0459 | 0.1524 | 0.0704 | 0.1481 |
| xgboost | `ohlcv14` | -0.028800 | 0.064109 | -0.103661 | -0.031515 | +0.051490 | **1 of 4** | -0.5390 | 0.0708 | 0.0278 | 0.1359 |
| xgboost | `derivatives_v1` | -0.057701 | 0.066964 | -0.155042 | -0.036631 | -0.002499 | **0 of 4** | -1.1344 | 0.0917 | 0.0385 | 0.1417 |
| xgboost | `ohlcv14_plus_derivatives_v1` | -0.061960 | 0.091431 | -0.196729 | -0.028070 | +0.005028 | **1 of 4** | -0.8890 | 0.1099 | 0.0394 | 0.1390 |

## Incremental value of market structure

Per model, per fold: the information set minus the `ohlcv14` control on the same rows.

### logistic_regression: `derivatives_v1` − `ohlcv14`

| fold | outer period | Δ net | Δ ann. Sharpe | Δ max DD (lower is better) | Δ exposure | Δ trades | Δ macro F1 | Δ dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | **-0.115645** | -2.1364 | +0.0043 | -0.0177 | -14 | -0.0083 | -0.1713 |
| 1 | 2023-09-24 → 2024-04-12 | **-0.040727** | -0.5767 | +0.0066 | +0.0543 | +43 | +0.0082 | -0.1687 |
| 2 | 2024-04-12 → 2024-10-30 | **-0.200574** | -3.6793 | -0.0414 | -0.0568 | -45 | -0.0136 | -0.4419 |
| 3 | 2024-10-30 → 2025-05-19 | **-0.225959** | -1.4744 | -0.0787 | -0.0745 | -59 | -0.0205 | +0.0076 |

Net return improved in **0 of 4** temporal folds (mean Δ -0.145726, min -0.225959, max -0.040727).

**Verdict:** no improvement in any temporal fold — negative evidence.

### logistic_regression: `ohlcv14_plus_derivatives_v1` − `ohlcv14`

| fold | outer period | Δ net | Δ ann. Sharpe | Δ max DD (lower is better) | Δ exposure | Δ trades | Δ macro F1 | Δ dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | **-0.070430** | -0.9534 | +0.0005 | -0.0038 | -3 | +0.0001 | +0.0476 |
| 1 | 2023-09-24 → 2024-04-12 | **-0.146557** | -2.1784 | +0.0595 | +0.0353 | +28 | +0.0073 | -0.0946 |
| 2 | 2024-04-12 → 2024-10-30 | **-0.237534** | -4.0926 | -0.0044 | -0.0518 | -41 | -0.0128 | -0.2419 |
| 3 | 2024-10-30 → 2025-05-19 | **-0.376371** | -2.6805 | +0.0487 | +0.0139 | +11 | +0.0004 | -0.0586 |

Net return improved in **0 of 4** temporal folds (mean Δ -0.207723, min -0.376371, max -0.070430).

**Verdict:** no improvement in any temporal fold — negative evidence.

### lightgbm: `derivatives_v1` − `ohlcv14`

| fold | outer period | Δ net | Δ ann. Sharpe | Δ max DD (lower is better) | Δ exposure | Δ trades | Δ macro F1 | Δ dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | **-0.140431** | -1.4442 | +0.1394 | +0.1174 | +93 | +0.0333 | +0.1267 |
| 1 | 2023-09-24 → 2024-04-12 | **+0.115514** | +0.6063 | -0.2653 | -0.3257 | -258 | -0.0904 | -0.0057 |
| 2 | 2024-04-12 → 2024-10-30 | **+0.450525** | +2.8594 | -0.4632 | -0.2032 | -161 | -0.0463 | -0.1563 |
| 3 | 2024-10-30 → 2025-05-19 | **-0.255142** | -3.1881 | +0.0809 | +0.0138 | +11 | -0.0009 | -0.1706 |

Net return improved in **2 of 4** temporal folds (mean Δ +0.042617, min -0.255142, max +0.450525).

**Verdict:** improvement in two of four temporal folds — regime-dependent, inconclusive.

### lightgbm: `ohlcv14_plus_derivatives_v1` − `ohlcv14`

| fold | outer period | Δ net | Δ ann. Sharpe | Δ max DD (lower is better) | Δ exposure | Δ trades | Δ macro F1 | Δ dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | **-0.064808** | -1.3730 | +0.0359 | +0.0164 | +13 | +0.0039 | +0.2500 |
| 1 | 2023-09-24 → 2024-04-12 | **+0.079299** | +0.2826 | -0.1777 | -0.2613 | -207 | -0.0723 | +0.0338 |
| 2 | 2024-04-12 → 2024-10-30 | **+0.352456** | +3.3221 | -0.2831 | -0.0808 | -64 | -0.0162 | +0.0756 |
| 3 | 2024-10-30 → 2025-05-19 | **-0.251440** | -2.8761 | +0.0780 | +0.0000 | +0 | +0.0006 | -0.0490 |

Net return improved in **2 of 4** temporal folds (mean Δ +0.028877, min -0.251440, max +0.352456).

**Verdict:** improvement in two of four temporal folds — regime-dependent, inconclusive.

### xgboost: `derivatives_v1` − `ohlcv14`

| fold | outer period | Δ net | Δ ann. Sharpe | Δ max DD (lower is better) | Δ exposure | Δ trades | Δ macro F1 | Δ dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | **-0.012028** | -0.4151 | -0.0051 | -0.0051 | -4 | -0.0011 | +0.0089 |
| 1 | 2023-09-24 → 2024-04-12 | **-0.092708** | -1.2239 | +0.0959 | +0.0846 | +67 | +0.0304 | -0.1021 |
| 2 | 2024-04-12 → 2024-10-30 | **-0.051381** | -1.0374 | +0.0615 | +0.0025 | +2 | +0.0036 | +0.0296 |
| 3 | 2024-10-30 → 2025-05-19 | **+0.040513** | +0.2948 | -0.0687 | -0.0391 | -31 | -0.0096 | -0.0048 |

Net return improved in **1 of 4** temporal folds (mean Δ -0.028901, min -0.092708, max +0.040513).

**Verdict:** improvement in one of four temporal folds — weak evidence against.

### xgboost: `ohlcv14_plus_derivatives_v1` − `ohlcv14`

| fold | outer period | Δ net | Δ ann. Sharpe | Δ max DD (lower is better) | Δ exposure | Δ trades | Δ macro F1 | Δ dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | **-0.016177** | -0.6079 | -0.0035 | +0.0050 | +4 | -0.0013 | -0.1594 |
| 1 | 2023-09-24 → 2024-04-12 | **-0.046462** | -0.7461 | +0.0476 | +0.0064 | +5 | +0.0034 | -0.0070 |
| 2 | 2024-04-12 → 2024-10-30 | **-0.093068** | -0.1652 | +0.1307 | +0.0694 | +55 | +0.0183 | +0.0848 |
| 3 | 2024-10-30 → 2025-05-19 | **+0.023066** | +0.1192 | -0.0183 | -0.0341 | -27 | -0.0078 | -0.0398 |

Net return improved in **1 of 4** temporal folds (mean Δ -0.033160, min -0.093068, max +0.023066).

**Verdict:** improvement in one of four temporal folds — weak evidence against.

## Long / short attribution

Additive decomposition of the realised trades a cell took. These are the two halves of
one reported result, not two standalone strategies: neither side was selected for, and
neither could have been traded on its own without the threshold that produced both.

| model | information set | long trades | long hit | long mean net | short trades | short hit | short mean net |
| --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | `ohlcv14` | 98 | 0.4898 | +0.002390 | 91 | 0.5165 | +0.003110 |
| logistic_regression | `derivatives_v1` | 68 | 0.4706 | -0.000133 | 46 | 0.3043 | -0.000655 |
| logistic_regression | `ohlcv14_plus_derivatives_v1` | 97 | 0.4639 | +0.000224 | 87 | 0.4828 | -0.003654 |
| lightgbm | `ohlcv14` | 269 | 0.5167 | -0.000022 | 212 | 0.4057 | -0.003214 |
| lightgbm | `derivatives_v1` | 93 | 0.3763 | -0.000767 | 73 | 0.4247 | -0.003490 |
| lightgbm | `ohlcv14_plus_derivatives_v1` | 138 | 0.4565 | -0.000901 | 85 | 0.4471 | -0.002818 |
| xgboost | `ohlcv14` | 61 | 0.4426 | +0.000067 | 27 | 0.4074 | -0.004227 |
| xgboost | `derivatives_v1` | 65 | 0.4154 | +0.000258 | 57 | 0.4737 | -0.004377 |
| xgboost | `ohlcv14_plus_derivatives_v1` | 87 | 0.4828 | -0.000725 | 38 | 0.3158 | -0.005014 |

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
