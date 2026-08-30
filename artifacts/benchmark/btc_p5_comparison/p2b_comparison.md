# P5 — do `mtf_v1` or `ohlcv14_plus_mtf_v1` add information beyond `ohlcv14`?

**Research question:** does causal mtf_v1, alone or combined with OHLCV14, add usable information beyond OHLCV14?

3 information sets, 3 untuned models, four temporal outer
folds, one sample universe. `ohlcv14` is the control, re-run under this code
path rather than copied. The other arms are:

- `mtf_v1`
- `ohlcv14_plus_mtf_v1`

**Research contract:** `btc-usdt-1h-gen1` (generation 1), semantic identity
`sha256:dca6d0a891a257197a7c8aecec04fdba0a0b3009cfe93c5f3a397d458ab4a1de`.

**Sealed test block:** everything at or after `2025-08-27T23:00:00+00:00`.
Not planned over, not fitted on, not selected on, not scored.
**Styx was not opened.**

**Statistical unit:** one temporal outer period per fold, four in total. These
estimators are deterministic given their inputs, so no seed replication appears
anywhere below:
a second seed would copy this evidence rather than add to it.

**Adaptive status:** P5 was designed after P4's outer results had been seen, and by the time it runs these four outer blocks will have been read by v4, P2a, P2b, the P2b ablation, the P2b regime description, P2c, P3 and P4. Its feature definition, arms, models, folds and decision rule were fixed before its own outer numbers existed, and the axis was chosen because four handcrafted families on one clock had failed — so this is exploratory adaptive evidence: it generates hypotheses and cannot confirm one.

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

- **9 cells, 36 folds, 170451 rows checked**
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
| `ohlcv14` | 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.74 | 2 | 0.0025 | 4.0 | +0.012984 | 0.0040 | **+0.008960** | 0.0022 | 0.9446 | 0.4777 | 0.5000 | 5.1658 | 0.1683 | 0.5000 |
| `ohlcv14` | 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.62 | 41 | 0.0518 | 82.0 | +0.161179 | 0.0820 | **+0.077488** | 0.0645 | 0.9531 | 0.1285 | 0.5366 | 1.4372 | 0.1393 | 0.5672 |
| `ohlcv14` | 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.56 | 89 | 0.1124 | 178.0 | +0.377883 | 0.1780 | **+0.202977** | 0.1154 | 1.4746 | 0.1214 | 0.4607 | 1.4322 | 0.1452 | 0.5120 |
| `ohlcv14` | 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.56 | 116 | 0.1465 | 232.0 | +0.206449 | 0.2320 | **-0.044097** | 0.2770 | -0.1481 | -0.0120 | 0.5345 | 0.9687 | 0.1471 | 0.5336 |
| `mtf_v1` | 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.82 | 7 | 0.0088 | 14.0 | +0.005018 | 0.0140 | **-0.009230** | 0.0183 | -0.3384 | -0.1317 | 0.2857 | 0.6511 | 0.1686 | 0.2222 |
| `mtf_v1` | 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.64 | 119 | 0.1503 | 238.0 | -0.158416 | 0.2380 | **-0.337711** | 0.3556 | -3.2340 | -0.2118 | 0.3697 | 0.5309 | 0.1654 | 0.4354 |
| `mtf_v1` | 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.76 | 16 | 0.0202 | 32.0 | -0.176954 | 0.0320 | **-0.192572** | 0.2091 | -3.0734 | -0.6122 | 0.3125 | 0.1010 | 0.1201 | 0.3929 |
| `mtf_v1` | 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.74 | 11 | 0.0139 | 22.0 | -0.044950 | 0.0220 | **-0.067798** | 0.0835 | -1.4347 | -0.2465 | 0.3636 | 0.4663 | 0.1094 | 0.4118 |
| `ohlcv14_plus_mtf_v1` | 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.80 | 31 | 0.0391 | 62.0 | +0.120071 | 0.0620 | **+0.053752** | 0.0623 | 0.9449 | 0.0946 | 0.3548 | 1.4496 | 0.1761 | 0.4000 |
| `ohlcv14_plus_mtf_v1` | 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.78 | 41 | 0.0518 | 82.0 | -0.027343 | 0.0820 | **-0.107694** | 0.1104 | -1.4956 | -0.1800 | 0.3902 | 0.6010 | 0.1349 | 0.3939 |
| `ohlcv14_plus_mtf_v1` | 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.82 | 13 | 0.0164 | 26.0 | -0.061929 | 0.0260 | **-0.086772** | 0.0964 | -1.4504 | -0.3308 | 0.3077 | 0.3690 | 0.1188 | 0.4211 |
| `ohlcv14_plus_mtf_v1` | 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.80 | 9 | 0.0114 | 18.0 | +0.005041 | 0.0180 | **-0.014788** | 0.0592 | -0.2978 | -0.0651 | 0.4444 | 0.8177 | 0.1088 | 0.4000 |

### lightgbm

| information set | fold | outer period | samples | thr | trades | exposure | turnover | gross | costs | **net** | max DD | ann. Sharpe | per-trade Sharpe | win rate | profit factor | macro F1 | dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ohlcv14` | 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.64 | 76 | 0.0960 | 152.0 | +0.040503 | 0.1520 | **-0.114274** | 0.1799 | -1.0692 | -0.0912 | 0.4211 | 0.7390 | 0.1941 | 0.3851 |
| `ohlcv14` | 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.64 | 50 | 0.0631 | 100.0 | +0.078005 | 0.1000 | **-0.025507** | 0.0845 | -0.2425 | -0.0351 | 0.4800 | 0.9059 | 0.1449 | 0.4865 |
| `ohlcv14` | 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.60 | 72 | 0.0909 | 144.0 | +0.036667 | 0.1440 | **-0.110713** | 0.1509 | -0.9608 | -0.0892 | 0.5000 | 0.7784 | 0.1377 | 0.4412 |
| `ohlcv14` | 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.60 | 78 | 0.0985 | 156.0 | +0.243106 | 0.1560 | **+0.079661** | 0.1352 | 0.7256 | 0.0677 | 0.5000 | 1.1935 | 0.1368 | 0.5305 |
| `mtf_v1` | 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.74 | 16 | 0.0202 | 32.0 | -0.044481 | 0.0320 | **-0.074332** | 0.0743 | -2.3700 | -0.5512 | 0.3125 | 0.1530 | 0.1708 | 0.2593 |
| `mtf_v1` | 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.64 | 67 | 0.0846 | 134.0 | -0.097397 | 0.1340 | **-0.212689** | 0.2141 | -2.5745 | -0.2327 | 0.3731 | 0.5229 | 0.1435 | 0.3701 |
| `mtf_v1` | 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.68 | 16 | 0.0202 | 32.0 | +0.128803 | 0.0320 | **+0.098904** | 0.0241 | 1.7135 | 0.3500 | 0.4375 | 2.5364 | 0.1202 | 0.3750 |
| `mtf_v1` | 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.66 | 15 | 0.0189 | 30.0 | +0.046968 | 0.0300 | **+0.013021** | 0.0846 | 0.2698 | 0.0465 | 0.5333 | 1.1674 | 0.1131 | 0.5333 |
| `ohlcv14_plus_mtf_v1` | 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.66 | 135 | 0.1705 | 270.0 | +0.010354 | 0.2700 | **-0.238971** | 0.3156 | -2.2943 | -0.1362 | 0.3407 | 0.6124 | 0.2160 | 0.3415 |
| `ohlcv14_plus_mtf_v1` | 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.76 | 12 | 0.0152 | 24.0 | +0.028610 | 0.0240 | **+0.004063** | 0.0301 | 0.1443 | 0.0383 | 0.5833 | 1.1039 | 0.1305 | 0.5417 |
| `ohlcv14_plus_mtf_v1` | 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.74 | 7 | 0.0088 | 14.0 | +0.081022 | 0.0140 | **+0.068067** | 0.0075 | 1.9719 | 0.5596 | 0.7143 | 8.6988 | 0.1196 | 0.7692 |
| `ohlcv14_plus_mtf_v1` | 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.70 | 27 | 0.0341 | 54.0 | +0.009469 | 0.0540 | **-0.045858** | 0.0994 | -0.5411 | -0.1225 | 0.4815 | 0.7214 | 0.1177 | 0.5769 |

### xgboost

| information set | fold | outer period | samples | thr | trades | exposure | turnover | gross | costs | **net** | max DD | ann. Sharpe | per-trade Sharpe | win rate | profit factor | macro F1 | dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ohlcv14` | 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.56 | 207 | 0.2614 | 414.0 | +0.179737 | 0.4140 | **-0.227758** | 0.2480 | -1.7224 | -0.0742 | 0.4058 | 0.7630 | 0.2409 | 0.3996 |
| `ohlcv14` | 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.68 | 26 | 0.0328 | 52.0 | +0.102474 | 0.0520 | **+0.048244** | 0.0713 | 0.7076 | 0.1185 | 0.5385 | 1.3863 | 0.1354 | 0.4902 |
| `ohlcv14` | 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.64 | 29 | 0.0366 | 58.0 | +0.186747 | 0.0580 | **+0.133502** | 0.0224 | 1.8538 | 0.2916 | 0.6207 | 2.6792 | 0.1263 | 0.5370 |
| `ohlcv14` | 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.62 | 47 | 0.0593 | 94.0 | +0.458247 | 0.0940 | **+0.429306** | 0.0304 | 3.6617 | 0.4869 | 0.7234 | 4.5918 | 0.1320 | 0.6827 |
| `mtf_v1` | 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.64 | 89 | 0.1124 | 178.0 | -0.062956 | 0.1780 | **-0.217654** | 0.2331 | -3.3841 | -0.2792 | 0.3146 | 0.4436 | 0.1989 | 0.3320 |
| `mtf_v1` | 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.60 | 112 | 0.1414 | 224.0 | +0.047070 | 0.2240 | **-0.169813** | 0.2286 | -1.4995 | -0.1239 | 0.4107 | 0.7020 | 0.1675 | 0.4228 |
| `mtf_v1` | 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.64 | 35 | 0.0442 | 70.0 | +0.033170 | 0.0700 | **-0.039222** | 0.0761 | -0.5224 | -0.0773 | 0.4571 | 0.8191 | 0.1312 | 0.5600 |
| `mtf_v1` | 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.60 | 51 | 0.0644 | 102.0 | +0.117905 | 0.1020 | **+0.008644** | 0.1247 | 0.1775 | 0.0182 | 0.3725 | 1.0497 | 0.1225 | 0.4190 |
| `ohlcv14_plus_mtf_v1` | 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.74 | 22 | 0.0278 | 44.0 | -0.073002 | 0.0440 | **-0.112678** | 0.1162 | -2.3906 | -0.3703 | 0.4091 | 0.2564 | 0.1750 | 0.4000 |
| `ohlcv14_plus_mtf_v1` | 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.72 | 21 | 0.0265 | 42.0 | +0.017742 | 0.0420 | **-0.027115** | 0.0633 | -0.3859 | -0.0641 | 0.4762 | 0.8210 | 0.1334 | 0.4082 |
| `ohlcv14_plus_mtf_v1` | 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.74 | 8 | 0.0101 | 16.0 | +0.107951 | 0.0160 | **+0.093658** | 0.0182 | 1.8746 | 0.4850 | 0.7500 | 6.0520 | 0.1189 | 0.7273 |
| `ohlcv14_plus_mtf_v1` | 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.66 | 48 | 0.0606 | 96.0 | +0.331609 | 0.0960 | **+0.245659** | 0.1405 | 1.9886 | 0.1914 | 0.6458 | 1.7584 | 0.1268 | 0.6333 |

## Across the four temporal folds

| model | information set | net mean | net std | net min | net median | net max | positive folds | Sharpe mean | max DD mean | exposure mean | macro F1 mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | `ohlcv14` | +0.061332 | 0.106743 | -0.044097 | +0.043224 | +0.202977 | **3 of 4** | 0.8061 | 0.1148 | 0.0783 | 0.1500 |
| logistic_regression | `mtf_v1` | -0.151828 | 0.145611 | -0.337711 | -0.130185 | -0.009230 | **0 of 4** | -2.0201 | 0.1666 | 0.0483 | 0.1409 |
| logistic_regression | `ohlcv14_plus_mtf_v1` | -0.038876 | 0.073462 | -0.107694 | -0.050780 | +0.053752 | **1 of 4** | -0.5747 | 0.0821 | 0.0297 | 0.1346 |
| lightgbm | `ohlcv14` | -0.042708 | 0.091317 | -0.114274 | -0.068110 | +0.079661 | **1 of 4** | -0.3867 | 0.1376 | 0.0871 | 0.1534 |
| lightgbm | `mtf_v1` | -0.043774 | 0.132977 | -0.212689 | -0.030655 | +0.098904 | **2 of 4** | -0.7403 | 0.0993 | 0.0360 | 0.1369 |
| lightgbm | `ohlcv14_plus_mtf_v1` | -0.053175 | 0.132350 | -0.238971 | -0.020898 | +0.068067 | **2 of 4** | -0.1798 | 0.1132 | 0.0571 | 0.1459 |
| xgboost | `ohlcv14` | +0.095824 | 0.270554 | -0.227758 | +0.090873 | +0.429306 | **3 of 4** | 1.1252 | 0.0930 | 0.0975 | 0.1587 |
| xgboost | `mtf_v1` | -0.104511 | 0.106665 | -0.217654 | -0.104517 | +0.008644 | **1 of 4** | -1.3071 | 0.1656 | 0.0906 | 0.1550 |
| xgboost | `ohlcv14_plus_mtf_v1` | +0.049881 | 0.155563 | -0.112678 | +0.033272 | +0.245659 | **2 of 4** | 0.2717 | 0.0846 | 0.0312 | 0.1385 |

## Incremental value of market structure

Per model, per fold: the information set minus the `ohlcv14` control on the same rows.

### logistic_regression: `mtf_v1` − `ohlcv14`

| fold | outer period | Δ net | Δ ann. Sharpe | Δ max DD (lower is better) | Δ exposure | Δ trades | Δ macro F1 | Δ dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | **-0.018190** | -1.2830 | +0.0161 | +0.0063 | +5 | +0.0003 | -0.2778 |
| 1 | 2023-09-24 → 2024-04-12 | **-0.415199** | -4.1871 | +0.2911 | +0.0985 | +78 | +0.0261 | -0.1318 |
| 2 | 2024-04-12 → 2024-10-30 | **-0.395549** | -4.5480 | +0.0937 | -0.0922 | -73 | -0.0251 | -0.1191 |
| 3 | 2024-10-30 → 2025-05-19 | **-0.023701** | -1.2866 | -0.1935 | -0.1326 | -105 | -0.0377 | -0.1218 |

Net return improved in **0 of 4** temporal folds (mean Δ -0.213160, min -0.415199, max -0.018190).

**Verdict:** no improvement in any temporal fold — negative evidence.

### logistic_regression: `ohlcv14_plus_mtf_v1` − `ohlcv14`

| fold | outer period | Δ net | Δ ann. Sharpe | Δ max DD (lower is better) | Δ exposure | Δ trades | Δ macro F1 | Δ dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | **+0.044792** | +0.0003 | +0.0601 | +0.0366 | +29 | +0.0078 | -0.1000 |
| 1 | 2023-09-24 → 2024-04-12 | **-0.185182** | -2.4487 | +0.0459 | +0.0000 | +0 | -0.0044 | -0.1733 |
| 2 | 2024-04-12 → 2024-10-30 | **-0.289749** | -2.9250 | -0.0190 | -0.0960 | -76 | -0.0264 | -0.0909 |
| 3 | 2024-10-30 → 2025-05-19 | **+0.029309** | -0.1497 | -0.2178 | -0.1351 | -107 | -0.0383 | -0.1336 |

Net return improved in **2 of 4** temporal folds (mean Δ -0.100208, min -0.289749, max +0.044792).

**Verdict:** improvement in two of four temporal folds — regime-dependent, inconclusive.

### lightgbm: `mtf_v1` − `ohlcv14`

| fold | outer period | Δ net | Δ ann. Sharpe | Δ max DD (lower is better) | Δ exposure | Δ trades | Δ macro F1 | Δ dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | **+0.039942** | -1.3008 | -0.1056 | -0.0758 | -60 | -0.0233 | -0.1258 |
| 1 | 2023-09-24 → 2024-04-12 | **-0.187182** | -2.3320 | +0.1296 | +0.0215 | +17 | -0.0014 | -0.1164 |
| 2 | 2024-04-12 → 2024-10-30 | **+0.209617** | +2.6743 | -0.1268 | -0.0707 | -56 | -0.0175 | -0.0662 |
| 3 | 2024-10-30 → 2025-05-19 | **-0.066640** | -0.4558 | -0.0506 | -0.0796 | -63 | -0.0237 | +0.0028 |

Net return improved in **2 of 4** temporal folds (mean Δ -0.001066, min -0.187182, max +0.209617).

**Verdict:** improvement in two of four temporal folds — regime-dependent, inconclusive.

### lightgbm: `ohlcv14_plus_mtf_v1` − `ohlcv14`

| fold | outer period | Δ net | Δ ann. Sharpe | Δ max DD (lower is better) | Δ exposure | Δ trades | Δ macro F1 | Δ dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | **-0.124697** | -1.2251 | +0.1357 | +0.0745 | +59 | +0.0219 | -0.0436 |
| 1 | 2023-09-24 → 2024-04-12 | **+0.029570** | +0.3868 | -0.0544 | -0.0479 | -38 | -0.0144 | +0.0552 |
| 2 | 2024-04-12 → 2024-10-30 | **+0.178780** | +2.9327 | -0.1434 | -0.0821 | -65 | -0.0181 | +0.3280 |
| 3 | 2024-10-30 → 2025-05-19 | **-0.125519** | -1.2667 | -0.0358 | -0.0644 | -51 | -0.0191 | +0.0464 |

Net return improved in **2 of 4** temporal folds (mean Δ -0.010467, min -0.125519, max +0.178780).

**Verdict:** improvement in two of four temporal folds — regime-dependent, inconclusive.

### xgboost: `mtf_v1` − `ohlcv14`

| fold | outer period | Δ net | Δ ann. Sharpe | Δ max DD (lower is better) | Δ exposure | Δ trades | Δ macro F1 | Δ dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | **+0.010104** | -1.6617 | -0.0149 | -0.1490 | -118 | -0.0420 | -0.0676 |
| 1 | 2023-09-24 → 2024-04-12 | **-0.218057** | -2.2071 | +0.1573 | +0.1086 | +86 | +0.0321 | -0.0674 |
| 2 | 2024-04-12 → 2024-10-30 | **-0.172724** | -2.3762 | +0.0537 | +0.0076 | +6 | +0.0049 | +0.0230 |
| 3 | 2024-10-30 → 2025-05-19 | **-0.420662** | -3.4842 | +0.0943 | +0.0051 | +4 | -0.0095 | -0.2637 |

Net return improved in **1 of 4** temporal folds (mean Δ -0.200335, min -0.420662, max +0.010104).

**Verdict:** improvement in one of four temporal folds — weak evidence against.

### xgboost: `ohlcv14_plus_mtf_v1` − `ohlcv14`

| fold | outer period | Δ net | Δ ann. Sharpe | Δ max DD (lower is better) | Δ exposure | Δ trades | Δ macro F1 | Δ dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | **+0.115080** | -0.6682 | -0.1318 | -0.2336 | -185 | -0.0659 | +0.0004 |
| 1 | 2023-09-24 → 2024-04-12 | **-0.075359** | -1.0935 | -0.0080 | -0.0063 | -5 | -0.0020 | -0.0820 |
| 2 | 2024-04-12 → 2024-10-30 | **-0.039844** | +0.0208 | -0.0042 | -0.0265 | -21 | -0.0074 | +0.1903 |
| 3 | 2024-10-30 → 2025-05-19 | **-0.183647** | -1.6731 | +0.1101 | +0.0013 | +1 | -0.0052 | -0.0494 |

Net return improved in **1 of 4** temporal folds (mean Δ -0.045942, min -0.183647, max +0.115080).

**Verdict:** improvement in one of four temporal folds — weak evidence against.

## Long / short attribution

Additive decomposition of the realised trades a cell took. These are the two halves of
one reported result, not two standalone strategies: neither side was selected for, and
neither could have been traded on its own without the threshold that produced both.

| model | information set | long trades | long hit | long mean net | short trades | short hit | short mean net |
| --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | `ohlcv14` | 159 | 0.5094 | +0.001270 | 89 | 0.5056 | +0.000680 |
| logistic_regression | `mtf_v1` | 95 | 0.4105 | -0.002145 | 58 | 0.2759 | -0.008233 |
| logistic_regression | `ohlcv14_plus_mtf_v1` | 53 | 0.4340 | -0.000175 | 41 | 0.2927 | -0.003485 |
| lightgbm | `ohlcv14` | 194 | 0.4639 | -0.001023 | 82 | 0.5000 | +0.000546 |
| lightgbm | `mtf_v1` | 83 | 0.4337 | -0.000514 | 31 | 0.2903 | -0.004886 |
| lightgbm | `ohlcv14_plus_mtf_v1` | 147 | 0.3810 | -0.001108 | 34 | 0.4412 | -0.002047 |
| xgboost | `ohlcv14` | 220 | 0.4773 | +0.001292 | 89 | 0.5056 | +0.000282 |
| xgboost | `mtf_v1` | 201 | 0.3930 | -0.001001 | 86 | 0.3488 | -0.002763 |
| xgboost | `ohlcv14_plus_mtf_v1` | 73 | 0.5616 | +0.003177 | 26 | 0.5769 | -0.001753 |

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
