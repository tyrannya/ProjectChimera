# P2c — do `chart_structure_v1` or `ohlcv14_plus_chart_structure_v1` add information beyond `ohlcv14`?

**Research question:** does causal chart_structure_v1, alone or combined with OHLCV14, add usable information beyond OHLCV14?

3 information sets, 3 untuned models, four temporal outer
folds, one sample universe. `ohlcv14` is the control, re-run under this code
path rather than copied. The other arms are:

- `chart_structure_v1`
- `ohlcv14_plus_chart_structure_v1`

**Research contract:** `btc-usdt-1h-gen1` (generation 1), semantic identity
`sha256:dca6d0a891a257197a7c8aecec04fdba0a0b3009cfe93c5f3a397d458ab4a1de`.

**Sealed test block:** everything at or after `2025-08-27T23:00:00+00:00`.
Not planned over, not fitted on, not selected on, not scored.
**Styx was not opened.**

**Statistical unit:** one temporal outer period per fold, four in total. These
estimators are deterministic given their inputs, so no seed replication appears
anywhere below:
a second seed would copy this evidence rather than add to it.

**Adaptive status:** P2c was designed after P2b's outer results had been seen, and by the time it ran these four outer blocks had already been read by v4, P2a, P2b, the P2b ablation and the P2b regime description. Its constants were fixed before its own outer results were read, but the family was chosen because the previous one failed, so this is exploratory adaptive evidence: it generates hypotheses and cannot confirm one.

## Sample-universe parity

every cell scored the same outer rows; a difference between two cells can only be the information set or the model. Checked across all 9 cells:

- research contract and its hash
- snapshot identity and semantic hashes
- fold sizes and periods
- the research checkpoint each cell says it answers
- per-fold sample-index hashes from the alignment proof
- label horizon and costs
- threshold grid, objective and trade floor
- combined feature-spec hash
- majority and momentum baseline outer reports
- CASH and buy-and-hold economic references

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
| `ohlcv14` | 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.62 | 28 | 0.0354 | 56.0 | +0.052144 | 0.0560 | **-0.006932** | 0.0588 | -0.0490 | -0.0091 | 0.5357 | 0.9747 | 0.1808 | 0.5686 |
| `ohlcv14` | 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.66 | 11 | 0.0139 | 22.0 | +0.086935 | 0.0220 | **+0.066156** | 0.0196 | 1.9999 | 0.5019 | 0.6364 | 3.8474 | 0.1305 | 0.6842 |
| `ohlcv14` | 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.56 | 75 | 0.0947 | 150.0 | +0.284734 | 0.1500 | **+0.130769** | 0.0891 | 1.0856 | 0.1002 | 0.4800 | 1.3600 | 0.1428 | 0.5098 |
| `ohlcv14` | 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.56 | 110 | 0.1389 | 220.0 | +0.067557 | 0.2200 | **-0.157307** | 0.3166 | -1.0412 | -0.0754 | 0.4909 | 0.8183 | 0.1441 | 0.5093 |
| `chart_structure_v1` | 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.86 | 56 | 0.0707 | 112.0 | -0.086173 | 0.1120 | **-0.184801** | 0.1848 | -2.1988 | -0.2433 | 0.2321 | 0.4249 | 0.1751 | 0.2020 |
| `chart_structure_v1` | 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.84 | 19 | 0.0240 | 38.0 | +0.053330 | 0.0380 | **+0.014957** | 0.0116 | 0.4147 | 0.1105 | 0.4211 | 1.3512 | 0.1311 | 0.4839 |
| `chart_structure_v1` | 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.76 | 44 | 0.0556 | 88.0 | -0.005396 | 0.0880 | **-0.093523** | 0.1589 | -1.1207 | -0.1437 | 0.4091 | 0.6727 | 0.1262 | 0.4375 |
| `chart_structure_v1` | 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.72 | 58 | 0.0732 | 116.0 | +0.173832 | 0.1160 | **+0.055146** | 0.0757 | 0.6758 | 0.0828 | 0.4655 | 1.2353 | 0.1253 | 0.5253 |
| `ohlcv14_plus_chart_structure_v1` | 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.86 | 73 | 0.0922 | 146.0 | -0.014867 | 0.1460 | **-0.153192** | 0.1532 | -2.1305 | -0.1849 | 0.3425 | 0.5174 | 0.1843 | 0.3256 |
| `ohlcv14_plus_chart_structure_v1` | 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.88 | 14 | 0.0177 | 28.0 | +0.021547 | 0.0280 | **-0.007183** | 0.0253 | -0.1944 | -0.0427 | 0.4286 | 0.8769 | 0.1281 | 0.3529 |
| `ohlcv14_plus_chart_structure_v1` | 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.70 | 136 | 0.1717 | 272.0 | +0.170617 | 0.2720 | **-0.114550** | 0.3003 | -0.7033 | -0.0428 | 0.4118 | 0.8835 | 0.1489 | 0.3909 |
| `ohlcv14_plus_chart_structure_v1` | 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.68 | 146 | 0.1843 | 292.0 | +0.126100 | 0.2920 | **-0.169830** | 0.2238 | -1.1029 | -0.0684 | 0.4863 | 0.8288 | 0.1526 | 0.4821 |

### lightgbm

| information set | fold | outer period | samples | thr | trades | exposure | turnover | gross | costs | **net** | max DD | ann. Sharpe | per-trade Sharpe | win rate | profit factor | macro F1 | dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ohlcv14` | 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.70 | 30 | 0.0379 | 60.0 | +0.020906 | 0.0600 | **-0.046515** | 0.1085 | -0.5360 | -0.0533 | 0.3333 | 0.8317 | 0.1774 | 0.3433 |
| `ohlcv14` | 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.60 | 96 | 0.1212 | 192.0 | -0.007094 | 0.1920 | **-0.189062** | 0.1978 | -1.5579 | -0.1414 | 0.4688 | 0.6876 | 0.1623 | 0.4530 |
| `ohlcv14` | 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.66 | 21 | 0.0265 | 42.0 | +0.019137 | 0.0420 | **-0.025714** | 0.0744 | -0.4007 | -0.0607 | 0.4286 | 0.8336 | 0.1228 | 0.4737 |
| `ohlcv14` | 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.62 | 59 | 0.0745 | 118.0 | +0.353907 | 0.1180 | **+0.256085** | 0.0369 | 2.3691 | 0.2468 | 0.5593 | 2.1441 | 0.1290 | 0.5909 |
| `chart_structure_v1` | 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.78 | 13 | 0.0164 | 26.0 | -0.040776 | 0.0260 | **-0.065004** | 0.0677 | -2.2576 | -0.7796 | 0.1538 | 0.0672 | 0.1708 | 0.2222 |
| `chart_structure_v1` | 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.76 | 8 | 0.0101 | 16.0 | +0.046859 | 0.0160 | **+0.031004** | 0.0159 | 1.2785 | 0.4403 | 0.5000 | 2.9304 | 0.1286 | 0.5000 |
| `chart_structure_v1` | 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.64 | 90 | 0.1136 | 180.0 | +0.031938 | 0.1800 | **-0.147132** | 0.2102 | -1.3290 | -0.1056 | 0.4444 | 0.7230 | 0.1462 | 0.4759 |
| `chart_structure_v1` | 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.68 | 28 | 0.0354 | 56.0 | +0.111407 | 0.0560 | **+0.054467** | 0.0412 | 1.0052 | 0.1510 | 0.6071 | 1.4556 | 0.1155 | 0.5455 |
| `ohlcv14_plus_chart_structure_v1` | 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.76 | 13 | 0.0164 | 26.0 | +0.039511 | 0.0260 | **+0.012508** | 0.0354 | 0.3797 | 0.0774 | 0.4615 | 1.2382 | 0.1727 | 0.3793 |
| `ohlcv14_plus_chart_structure_v1` | 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.74 | 10 | 0.0126 | 20.0 | +0.002607 | 0.0200 | **-0.018675** | 0.0403 | -0.4393 | -0.0970 | 0.4000 | 0.7787 | 0.1289 | 0.3333 |
| `ohlcv14_plus_chart_structure_v1` | 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.58 | 165 | 0.2083 | 330.0 | -0.162596 | 0.3300 | **-0.399846** | 0.4212 | -3.3233 | -0.2075 | 0.4424 | 0.5565 | 0.1742 | 0.4509 |
| `ohlcv14_plus_chart_structure_v1` | 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.66 | 41 | 0.0518 | 82.0 | +0.245900 | 0.0820 | **+0.166344** | 0.0633 | 1.6504 | 0.1790 | 0.5366 | 1.7562 | 0.1243 | 0.5882 |

### xgboost

| information set | fold | outer period | samples | thr | trades | exposure | turnover | gross | costs | **net** | max DD | ann. Sharpe | per-trade Sharpe | win rate | profit factor | macro F1 | dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ohlcv14` | 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.64 | 80 | 0.1010 | 160.0 | +0.062295 | 0.1600 | **-0.101571** | 0.1478 | -1.0454 | -0.0802 | 0.4625 | 0.7578 | 0.1968 | 0.4405 |
| `ohlcv14` | 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.72 | 11 | 0.0139 | 22.0 | +0.056220 | 0.0220 | **+0.032526** | 0.0393 | 0.6611 | 0.1482 | 0.4545 | 1.5588 | 0.1293 | 0.3913 |
| `ohlcv14` | 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.70 | 4 | 0.0051 | 8.0 | -0.000356 | 0.0080 | **-0.008425** | 0.0144 | -0.3705 | -0.2624 | 0.5000 | 0.4235 | 0.1172 | 0.4286 |
| `ohlcv14` | 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.68 | 16 | 0.0202 | 32.0 | +0.134772 | 0.0320 | **+0.105629** | 0.0310 | 1.8526 | 0.3872 | 0.6875 | 3.2504 | 0.1129 | 0.5926 |
| `chart_structure_v1` | 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.76 | 18 | 0.0227 | 36.0 | -0.041986 | 0.0360 | **-0.077599** | 0.0820 | -1.4756 | -0.2491 | 0.3333 | 0.4185 | 0.1724 | 0.3333 |
| `chart_structure_v1` | 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.70 | 20 | 0.0253 | 40.0 | +0.064987 | 0.0400 | **+0.022154** | 0.0316 | 0.4549 | 0.0692 | 0.4000 | 1.1998 | 0.1322 | 0.4000 |
| `chart_structure_v1` | 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.66 | 67 | 0.0846 | 134.0 | +0.052323 | 0.1340 | **-0.087196** | 0.1828 | -0.8012 | -0.0726 | 0.5075 | 0.7984 | 0.1431 | 0.5500 |
| `chart_structure_v1` | 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.70 | 26 | 0.0328 | 52.0 | +0.144933 | 0.0520 | **+0.092231** | 0.0640 | 1.3430 | 0.1844 | 0.6538 | 1.9234 | 0.1157 | 0.5581 |
| `ohlcv14_plus_chart_structure_v1` | 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.76 | 8 | 0.0101 | 16.0 | -0.009831 | 0.0160 | **-0.026855** | 0.0586 | -0.5807 | -0.1648 | 0.3750 | 0.5913 | 0.1732 | 0.5714 |
| `ohlcv14_plus_chart_structure_v1` | 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.70 | 27 | 0.0341 | 54.0 | +0.031499 | 0.0540 | **-0.024471** | 0.0471 | -0.3752 | -0.0631 | 0.3704 | 0.8537 | 0.1317 | 0.3659 |
| `ohlcv14_plus_chart_structure_v1` | 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.54 | 260 | 0.3283 | 520.0 | +0.059263 | 0.5200 | **-0.388741** | 0.4018 | -2.3466 | -0.1145 | 0.4269 | 0.7203 | 0.2135 | 0.4561 |
| `ohlcv14_plus_chart_structure_v1` | 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.60 | 121 | 0.1528 | 242.0 | +0.298370 | 0.2420 | **+0.040238** | 0.1806 | 0.3973 | 0.0276 | 0.5124 | 1.0813 | 0.1596 | 0.5298 |

## Across the four temporal folds

| model | information set | net mean | net std | net min | net median | net max | positive folds | Sharpe mean | max DD mean | exposure mean | macro F1 mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | `ohlcv14` | +0.008171 | 0.123833 | -0.157307 | +0.029612 | +0.130769 | **2 of 4** | 0.4988 | 0.1210 | 0.0707 | 0.1495 |
| logistic_regression | `chart_structure_v1` | -0.052055 | 0.108511 | -0.184801 | -0.039283 | +0.055146 | **2 of 4** | -0.5573 | 0.1077 | 0.0559 | 0.1394 |
| logistic_regression | `ohlcv14_plus_chart_structure_v1` | -0.111189 | 0.073102 | -0.169830 | -0.133871 | -0.007183 | **0 of 4** | -1.0328 | 0.1757 | 0.1165 | 0.1535 |
| lightgbm | `ohlcv14` | -0.001302 | 0.186317 | -0.189062 | -0.036115 | +0.256085 | **1 of 4** | -0.0314 | 0.1044 | 0.0650 | 0.1479 |
| lightgbm | `chart_structure_v1` | -0.031666 | 0.092719 | -0.147132 | -0.017000 | +0.054467 | **2 of 4** | -0.3257 | 0.0838 | 0.0439 | 0.1403 |
| lightgbm | `ohlcv14_plus_chart_structure_v1` | -0.059917 | 0.240619 | -0.399846 | -0.003084 | +0.166344 | **2 of 4** | -0.4331 | 0.1401 | 0.0723 | 0.1500 |
| xgboost | `ohlcv14` | +0.007040 | 0.086419 | -0.101571 | +0.012050 | +0.105629 | **2 of 4** | 0.2745 | 0.0581 | 0.0350 | 0.1391 |
| xgboost | `chart_structure_v1` | -0.012603 | 0.085609 | -0.087196 | -0.027723 | +0.092231 | **2 of 4** | -0.1197 | 0.0901 | 0.0413 | 0.1409 |
| xgboost | `ohlcv14_plus_chart_structure_v1` | -0.099957 | 0.195015 | -0.388741 | -0.025663 | +0.040238 | **1 of 4** | -0.7263 | 0.1720 | 0.1313 | 0.1695 |

## Incremental value of market structure

Per model, per fold: the information set minus the `ohlcv14` control on the same rows.

### logistic_regression: `chart_structure_v1` − `ohlcv14`

| fold | outer period | Δ net | Δ ann. Sharpe | Δ max DD (lower is better) | Δ exposure | Δ trades | Δ macro F1 | Δ dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | **-0.177869** | -2.1498 | +0.1260 | +0.0353 | +28 | -0.0057 | -0.3666 |
| 1 | 2023-09-24 → 2024-04-12 | **-0.051199** | -1.5852 | -0.0080 | +0.0101 | +8 | +0.0006 | -0.2003 |
| 2 | 2024-04-12 → 2024-10-30 | **-0.224292** | -2.2063 | +0.0698 | -0.0391 | -31 | -0.0166 | -0.0723 |
| 3 | 2024-10-30 → 2025-05-19 | **+0.212453** | +1.7170 | -0.2409 | -0.0657 | -52 | -0.0188 | +0.0160 |

Net return improved in **1 of 4** temporal folds (mean Δ -0.060227, min -0.224292, max +0.212453).

**Verdict:** improvement in one of four temporal folds — weak evidence against.

### logistic_regression: `ohlcv14_plus_chart_structure_v1` − `ohlcv14`

| fold | outer period | Δ net | Δ ann. Sharpe | Δ max DD (lower is better) | Δ exposure | Δ trades | Δ macro F1 | Δ dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | **-0.146260** | -2.0815 | +0.0944 | +0.0568 | +45 | +0.0035 | -0.2430 |
| 1 | 2023-09-24 → 2024-04-12 | **-0.073339** | -2.1943 | +0.0057 | +0.0038 | +3 | -0.0024 | -0.3313 |
| 2 | 2024-04-12 → 2024-10-30 | **-0.245319** | -1.7889 | +0.2112 | +0.0770 | +61 | +0.0061 | -0.1189 |
| 3 | 2024-10-30 → 2025-05-19 | **-0.012523** | -0.0617 | -0.0928 | +0.0454 | +36 | +0.0085 | -0.0272 |

Net return improved in **0 of 4** temporal folds (mean Δ -0.119360, min -0.245319, max -0.012523).

**Verdict:** no improvement in any temporal fold — negative evidence.

### lightgbm: `chart_structure_v1` − `ohlcv14`

| fold | outer period | Δ net | Δ ann. Sharpe | Δ max DD (lower is better) | Δ exposure | Δ trades | Δ macro F1 | Δ dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | **-0.018489** | -1.7216 | -0.0408 | -0.0215 | -17 | -0.0066 | -0.1211 |
| 1 | 2023-09-24 → 2024-04-12 | **+0.220066** | +2.8364 | -0.1819 | -0.1111 | -88 | -0.0337 | +0.0470 |
| 2 | 2024-04-12 → 2024-10-30 | **-0.121418** | -0.9283 | +0.1358 | +0.0871 | +69 | +0.0234 | +0.0022 |
| 3 | 2024-10-30 → 2025-05-19 | **-0.201618** | -1.3639 | +0.0043 | -0.0391 | -31 | -0.0135 | -0.0454 |

Net return improved in **1 of 4** temporal folds (mean Δ -0.030365, min -0.201618, max +0.220066).

**Verdict:** improvement in one of four temporal folds — weak evidence against.

### lightgbm: `ohlcv14_plus_chart_structure_v1` − `ohlcv14`

| fold | outer period | Δ net | Δ ann. Sharpe | Δ max DD (lower is better) | Δ exposure | Δ trades | Δ macro F1 | Δ dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | **+0.059023** | +0.9157 | -0.0731 | -0.0215 | -17 | -0.0047 | +0.0360 |
| 1 | 2023-09-24 → 2024-04-12 | **+0.170387** | +1.1186 | -0.1575 | -0.1086 | -86 | -0.0334 | -0.1197 |
| 2 | 2024-04-12 → 2024-10-30 | **-0.374132** | -2.9226 | +0.3468 | +0.1818 | +144 | +0.0514 | -0.0228 |
| 3 | 2024-10-30 → 2025-05-19 | **-0.089741** | -0.7187 | +0.0264 | -0.0227 | -18 | -0.0047 | -0.0027 |

Net return improved in **2 of 4** temporal folds (mean Δ -0.058616, min -0.374132, max +0.170387).

**Verdict:** improvement in two of four temporal folds — regime-dependent, inconclusive.

### xgboost: `chart_structure_v1` − `ohlcv14`

| fold | outer period | Δ net | Δ ann. Sharpe | Δ max DD (lower is better) | Δ exposure | Δ trades | Δ macro F1 | Δ dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | **+0.023972** | -0.4302 | -0.0658 | -0.0783 | -62 | -0.0244 | -0.1072 |
| 1 | 2023-09-24 → 2024-04-12 | **-0.010372** | -0.2062 | -0.0077 | +0.0114 | +9 | +0.0029 | +0.0087 |
| 2 | 2024-04-12 → 2024-10-30 | **-0.078771** | -0.4307 | +0.1684 | +0.0795 | +63 | +0.0259 | +0.1214 |
| 3 | 2024-10-30 → 2025-05-19 | **-0.013398** | -0.5096 | +0.0330 | +0.0126 | +10 | +0.0028 | -0.0345 |

Net return improved in **1 of 4** temporal folds (mean Δ -0.019642, min -0.078771, max +0.023972).

**Verdict:** improvement in one of four temporal folds — weak evidence against.

### xgboost: `ohlcv14_plus_chart_structure_v1` − `ohlcv14`

| fold | outer period | Δ net | Δ ann. Sharpe | Δ max DD (lower is better) | Δ exposure | Δ trades | Δ macro F1 | Δ dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | **+0.074716** | +0.4647 | -0.0892 | -0.0909 | -72 | -0.0236 | +0.1309 |
| 1 | 2023-09-24 → 2024-04-12 | **-0.056997** | -1.0363 | +0.0078 | +0.0202 | +16 | +0.0024 | -0.0254 |
| 2 | 2024-04-12 → 2024-10-30 | **-0.380316** | -1.9761 | +0.3874 | +0.3232 | +256 | +0.0963 | +0.0275 |
| 3 | 2024-10-30 → 2025-05-19 | **-0.065391** | -1.4553 | +0.1496 | +0.1326 | +105 | +0.0467 | -0.0628 |

Net return improved in **1 of 4** temporal folds (mean Δ -0.106997, min -0.380316, max +0.074716).

**Verdict:** improvement in one of four temporal folds — weak evidence against.

## Long / short attribution

Additive decomposition of the realised trades a cell took. These are the two halves of
one reported result, not two standalone strategies: neither side was selected for, and
neither could have been traded on its own without the threshold that produced both.

| model | information set | long trades | long hit | long mean net | short trades | short hit | short mean net |
| --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | `ohlcv14` | 150 | 0.5133 | +0.000794 | 74 | 0.4730 | -0.001024 |
| logistic_regression | `chart_structure_v1` | 111 | 0.3604 | -0.000792 | 66 | 0.3939 | -0.001977 |
| logistic_regression | `ohlcv14_plus_chart_structure_v1` | 208 | 0.4327 | -0.000653 | 161 | 0.4224 | -0.001856 |
| lightgbm | `ohlcv14` | 135 | 0.5111 | +0.001070 | 71 | 0.3944 | -0.002388 |
| lightgbm | `chart_structure_v1` | 90 | 0.4222 | -0.001917 | 49 | 0.5102 | +0.000897 |
| lightgbm | `ohlcv14_plus_chart_structure_v1` | 146 | 0.4521 | -0.001480 | 83 | 0.4699 | -0.001404 |
| xgboost | `ohlcv14` | 83 | 0.4699 | +0.000589 | 28 | 0.5714 | -0.000641 |
| xgboost | `chart_structure_v1` | 88 | 0.4773 | -0.000429 | 43 | 0.5349 | -0.000093 |
| xgboost | `ohlcv14_plus_chart_structure_v1` | 249 | 0.4859 | -0.000411 | 167 | 0.3892 | -0.002098 |

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
