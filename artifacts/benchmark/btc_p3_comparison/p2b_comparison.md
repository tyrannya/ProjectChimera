# P3 — do `microstructure_v1` or `ohlcv14_plus_microstructure_v1` add information beyond `ohlcv14`?

**Research question:** does causal microstructure_v1, alone or combined with OHLCV14, add usable information beyond OHLCV14?

3 information sets, 3 untuned models, four temporal outer
folds, one sample universe. `ohlcv14` is the control, re-run under this code
path rather than copied. The other arms are:

- `microstructure_v1`
- `ohlcv14_plus_microstructure_v1`

**Research contract:** `btc-usdt-1h-gen1` (generation 1), semantic identity
`sha256:dca6d0a891a257197a7c8aecec04fdba0a0b3009cfe93c5f3a397d458ab4a1de`.

**Sealed test block:** everything at or after `2025-08-27T23:00:00+00:00`.
Not planned over, not fitted on, not selected on, not scored.
**Styx was not opened.**

**Statistical unit:** one temporal outer period per fold, four in total. These
estimators are deterministic given their inputs, so no seed replication appears
anywhere below:
a second seed would copy this evidence rather than add to it.

**Adaptive status:** P3 was designed after P2b's and P2c's outer results had been seen, and by the time it ran these four outer blocks had already been read by v4, P2a, P2b, the P2b ablation, the P2b regime description and P2c. Its constants were fixed before its own outer results were read, and its information source is new rather than another transformation of the same candles — but the blocks are the same blocks, so this is exploratory adaptive evidence: it generates hypotheses and cannot confirm one.

## Sample-universe parity

every cell scored the same outer rows; a difference between two cells can only be the information set or the model. Checked across all 9 cells:

- research contract and its hash
- snapshot identity and semantic hashes
- trade-source identity, aggregation spec and microstructure spec, where a checkpoint has one
- the same absence of a trade source, where it has none
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
| `ohlcv14` | 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.62 | 30 | 0.0379 | 60.0 | +0.043126 | 0.0600 | **-0.019651** | 0.0628 | -0.2449 | -0.0394 | 0.5000 | 0.8949 | 0.1798 | 0.5294 |
| `ohlcv14` | 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.66 | 9 | 0.0114 | 18.0 | +0.072782 | 0.0180 | **+0.055412** | 0.0196 | 1.7241 | 0.4631 | 0.5556 | 3.4022 | 0.1298 | 0.6471 |
| `ohlcv14` | 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.56 | 72 | 0.0909 | 144.0 | +0.306114 | 0.1440 | **+0.158958** | 0.1292 | 1.2689 | 0.1117 | 0.4861 | 1.4147 | 0.1426 | 0.5098 |
| `ohlcv14` | 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.62 | 31 | 0.0391 | 62.0 | -0.006708 | 0.0620 | **-0.070168** | 0.1185 | -0.9047 | -0.1364 | 0.4194 | 0.7207 | 0.1138 | 0.3913 |
| `microstructure_v1` | 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.82 | 90 | 0.1136 | 180.0 | +0.045062 | 0.1800 | **-0.128970** | 0.1850 | -2.0391 | -0.1819 | 0.3889 | 0.5975 | 0.1872 | 0.3884 |
| `microstructure_v1` | 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.88 | 27 | 0.0341 | 54.0 | -0.032671 | 0.0540 | **-0.084011** | 0.1019 | -2.1015 | -0.3795 | 0.3333 | 0.3298 | 0.1320 | 0.3542 |
| `microstructure_v1` | 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.88 | 28 | 0.0354 | 56.0 | +0.165741 | 0.0560 | **+0.112619** | 0.0553 | 1.9783 | 0.2702 | 0.4643 | 2.1135 | 0.1221 | 0.4722 |
| `microstructure_v1` | 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.84 | 39 | 0.0492 | 78.0 | +0.124393 | 0.0780 | **+0.043236** | 0.0687 | 0.6849 | 0.0817 | 0.5641 | 1.2457 | 0.1167 | 0.5273 |
| `ohlcv14_plus_microstructure_v1` | 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.90 | 33 | 0.0417 | 66.0 | +0.048676 | 0.0660 | **-0.019030** | 0.0554 | -0.3790 | -0.0485 | 0.5455 | 0.8674 | 0.1766 | 0.4082 |
| `ohlcv14_plus_microstructure_v1` | 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.90 | 23 | 0.0290 | 46.0 | -0.030084 | 0.0460 | **-0.073999** | 0.0940 | -2.0485 | -0.4259 | 0.3478 | 0.2870 | 0.1305 | 0.3000 |
| `ohlcv14_plus_microstructure_v1` | 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.86 | 40 | 0.0505 | 80.0 | +0.099617 | 0.0800 | **+0.012511** | 0.0880 | 0.2337 | 0.0253 | 0.4250 | 1.0817 | 0.1235 | 0.4200 |
| `ohlcv14_plus_microstructure_v1` | 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.86 | 36 | 0.0455 | 72.0 | +0.169939 | 0.0720 | **+0.099168** | 0.0486 | 1.2404 | 0.1989 | 0.5278 | 1.6608 | 0.1158 | 0.5208 |

### lightgbm

| information set | fold | outer period | samples | thr | trades | exposure | turnover | gross | costs | **net** | max DD | ann. Sharpe | per-trade Sharpe | win rate | profit factor | macro F1 | dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ohlcv14` | 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.70 | 30 | 0.0379 | 60.0 | +0.020906 | 0.0600 | **-0.046515** | 0.1085 | -0.5360 | -0.0533 | 0.3333 | 0.8317 | 0.1774 | 0.3433 |
| `ohlcv14` | 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.60 | 96 | 0.1212 | 192.0 | -0.007094 | 0.1920 | **-0.189062** | 0.1978 | -1.5579 | -0.1414 | 0.4688 | 0.6876 | 0.1623 | 0.4530 |
| `ohlcv14` | 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.66 | 21 | 0.0265 | 42.0 | +0.019137 | 0.0420 | **-0.025714** | 0.0744 | -0.4007 | -0.0607 | 0.4286 | 0.8336 | 0.1228 | 0.4737 |
| `ohlcv14` | 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.62 | 59 | 0.0745 | 118.0 | +0.353907 | 0.1180 | **+0.256085** | 0.0369 | 2.3691 | 0.2468 | 0.5593 | 2.1441 | 0.1290 | 0.5909 |
| `microstructure_v1` | 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.64 | 66 | 0.0833 | 132.0 | +0.018799 | 0.1320 | **-0.114764** | 0.1852 | -1.2623 | -0.1054 | 0.4545 | 0.7113 | 0.1873 | 0.4490 |
| `microstructure_v1` | 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.66 | 7 | 0.0088 | 14.0 | -0.013091 | 0.0140 | **-0.027211** | 0.0357 | -0.8594 | -0.3199 | 0.5714 | 0.4094 | 0.1279 | 0.5714 |
| `microstructure_v1` | 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.60 | 43 | 0.0543 | 86.0 | -0.075014 | 0.0860 | **-0.151649** | 0.1648 | -1.9341 | -0.3070 | 0.3488 | 0.4624 | 0.1222 | 0.3469 |
| `microstructure_v1` | 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.60 | 38 | 0.0480 | 76.0 | +0.056619 | 0.0760 | **-0.023117** | 0.1366 | -0.2638 | -0.0345 | 0.4211 | 0.9144 | 0.1151 | 0.4231 |
| `ohlcv14_plus_microstructure_v1` | 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.66 | 49 | 0.0619 | 98.0 | +0.009953 | 0.0980 | **-0.090621** | 0.1443 | -1.1907 | -0.1069 | 0.4490 | 0.7078 | 0.1827 | 0.4615 |
| `ohlcv14_plus_microstructure_v1` | 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.62 | 50 | 0.0631 | 100.0 | +0.129492 | 0.1000 | **+0.023591** | 0.1135 | 0.3315 | 0.0370 | 0.5000 | 1.1146 | 0.1444 | 0.4717 |
| `ohlcv14_plus_microstructure_v1` | 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.68 | 5 | 0.0063 | 10.0 | +0.042813 | 0.0100 | **+0.032538** | 0.0205 | 1.0723 | 0.3517 | 0.6000 | 2.5922 | 0.1177 | 0.5714 |
| `ohlcv14_plus_microstructure_v1` | 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.62 | 40 | 0.0505 | 80.0 | +0.147853 | 0.0800 | **+0.065474** | 0.0955 | 0.8780 | 0.1128 | 0.5000 | 1.3224 | 0.1171 | 0.5091 |

### xgboost

| information set | fold | outer period | samples | thr | trades | exposure | turnover | gross | costs | **net** | max DD | ann. Sharpe | per-trade Sharpe | win rate | profit factor | macro F1 | dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ohlcv14` | 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.64 | 80 | 0.1010 | 160.0 | +0.062295 | 0.1600 | **-0.101571** | 0.1478 | -1.0454 | -0.0802 | 0.4625 | 0.7578 | 0.1968 | 0.4405 |
| `ohlcv14` | 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.72 | 11 | 0.0139 | 22.0 | +0.056220 | 0.0220 | **+0.032526** | 0.0393 | 0.6611 | 0.1482 | 0.4545 | 1.5588 | 0.1293 | 0.3913 |
| `ohlcv14` | 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.70 | 4 | 0.0051 | 8.0 | -0.000356 | 0.0080 | **-0.008425** | 0.0144 | -0.3705 | -0.2624 | 0.5000 | 0.4235 | 0.1172 | 0.4286 |
| `ohlcv14` | 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.68 | 16 | 0.0202 | 32.0 | +0.134772 | 0.0320 | **+0.105629** | 0.0310 | 1.8526 | 0.3872 | 0.6875 | 3.2504 | 0.1129 | 0.5926 |
| `microstructure_v1` | 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.64 | 53 | 0.0669 | 106.0 | +0.110523 | 0.1060 | **-0.001580** | 0.0630 | 0.0424 | 0.0055 | 0.3585 | 1.0192 | 0.1802 | 0.4085 |
| `microstructure_v1` | 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.68 | 5 | 0.0063 | 10.0 | +0.014634 | 0.0100 | **+0.004570** | 0.0094 | 0.2444 | 0.1541 | 0.6000 | 1.4938 | 0.1273 | 0.6000 |
| `microstructure_v1` | 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.58 | 65 | 0.0821 | 130.0 | -0.089020 | 0.1300 | **-0.203475** | 0.2274 | -2.3848 | -0.2147 | 0.4308 | 0.5310 | 0.1278 | 0.4571 |
| `microstructure_v1` | 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.58 | 99 | 0.1250 | 198.0 | +0.429811 | 0.1980 | **+0.243765** | 0.1000 | 1.9645 | 0.1403 | 0.5051 | 1.5133 | 0.1282 | 0.5000 |
| `ohlcv14_plus_microstructure_v1` | 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.60 | 118 | 0.1490 | 236.0 | -0.005313 | 0.2360 | **-0.223011** | 0.2289 | -2.2150 | -0.1517 | 0.3983 | 0.6283 | 0.2062 | 0.4364 |
| `ohlcv14_plus_microstructure_v1` | 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.70 | 12 | 0.0152 | 24.0 | +0.050600 | 0.0240 | **+0.024126** | 0.0440 | 0.5074 | 0.0981 | 0.4167 | 1.3595 | 0.1297 | 0.4500 |
| `ohlcv14_plus_microstructure_v1` | 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.62 | 28 | 0.0354 | 56.0 | +0.113875 | 0.0560 | **+0.057675** | 0.0442 | 1.0797 | 0.1812 | 0.5000 | 1.5909 | 0.1228 | 0.4419 |
| `ohlcv14_plus_microstructure_v1` | 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.66 | 21 | 0.0265 | 42.0 | +0.130299 | 0.0420 | **+0.091029** | 0.0190 | 1.5282 | 0.4209 | 0.7619 | 2.9717 | 0.1146 | 0.6774 |

## Across the four temporal folds

| model | information set | net mean | net std | net min | net median | net max | positive folds | Sharpe mean | max DD mean | exposure mean | macro F1 mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | `ohlcv14` | +0.031138 | 0.099615 | -0.070168 | +0.017881 | +0.158958 | **2 of 4** | 0.4608 | 0.0825 | 0.0448 | 0.1415 |
| logistic_regression | `microstructure_v1` | -0.014282 | 0.111696 | -0.128970 | -0.020388 | +0.112619 | **2 of 4** | -0.3694 | 0.1027 | 0.0581 | 0.1395 |
| logistic_regression | `ohlcv14_plus_microstructure_v1` | +0.004663 | 0.072438 | -0.073999 | -0.003259 | +0.099168 | **2 of 4** | -0.2384 | 0.0715 | 0.0417 | 0.1366 |
| lightgbm | `ohlcv14` | -0.001302 | 0.186317 | -0.189062 | -0.036115 | +0.256085 | **1 of 4** | -0.0314 | 0.1044 | 0.0650 | 0.1479 |
| lightgbm | `microstructure_v1` | -0.079185 | 0.064192 | -0.151649 | -0.070988 | -0.023117 | **0 of 4** | -1.0799 | 0.1306 | 0.0486 | 0.1381 |
| lightgbm | `ohlcv14_plus_microstructure_v1` | +0.007746 | 0.068006 | -0.090621 | +0.028064 | +0.065474 | **3 of 4** | 0.2728 | 0.0935 | 0.0454 | 0.1405 |
| xgboost | `ohlcv14` | +0.007040 | 0.086419 | -0.101571 | +0.012050 | +0.105629 | **2 of 4** | 0.2745 | 0.0581 | 0.0350 | 0.1391 |
| xgboost | `microstructure_v1` | +0.010820 | 0.182919 | -0.203475 | +0.001495 | +0.243765 | **2 of 4** | -0.0334 | 0.0999 | 0.0701 | 0.1409 |
| xgboost | `ohlcv14_plus_microstructure_v1` | -0.012545 | 0.142944 | -0.223011 | +0.040900 | +0.091029 | **3 of 4** | 0.2251 | 0.0840 | 0.0565 | 0.1433 |

## Incremental value of market structure

Per model, per fold: the information set minus the `ohlcv14` control on the same rows.

### logistic_regression: `microstructure_v1` − `ohlcv14`

| fold | outer period | Δ net | Δ ann. Sharpe | Δ max DD (lower is better) | Δ exposure | Δ trades | Δ macro F1 | Δ dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | **-0.109319** | -1.7942 | +0.1222 | +0.0757 | +60 | +0.0074 | -0.1410 |
| 1 | 2023-09-24 → 2024-04-12 | **-0.139423** | -3.8256 | +0.0823 | +0.0227 | +18 | +0.0022 | -0.2929 |
| 2 | 2024-04-12 → 2024-10-30 | **-0.046339** | +0.7094 | -0.0739 | -0.0555 | -44 | -0.0205 | -0.0376 |
| 3 | 2024-10-30 → 2025-05-19 | **+0.113404** | +1.5896 | -0.0498 | +0.0101 | +8 | +0.0029 | +0.1360 |

Net return improved in **1 of 4** temporal folds (mean Δ -0.045419, min -0.139423, max +0.113404).

**Verdict:** improvement in one of four temporal folds — weak evidence against.

### logistic_regression: `ohlcv14_plus_microstructure_v1` − `ohlcv14`

| fold | outer period | Δ net | Δ ann. Sharpe | Δ max DD (lower is better) | Δ exposure | Δ trades | Δ macro F1 | Δ dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | **+0.000621** | -0.1341 | -0.0074 | +0.0038 | +3 | -0.0032 | -0.1212 |
| 1 | 2023-09-24 → 2024-04-12 | **-0.129411** | -3.7726 | +0.0744 | +0.0176 | +14 | +0.0007 | -0.3471 |
| 2 | 2024-04-12 → 2024-10-30 | **-0.146447** | -1.0352 | -0.0412 | -0.0404 | -32 | -0.0191 | -0.0898 |
| 3 | 2024-10-30 → 2025-05-19 | **+0.169336** | +2.1451 | -0.0699 | +0.0064 | +5 | +0.0020 | +0.1295 |

Net return improved in **2 of 4** temporal folds (mean Δ -0.026475, min -0.146447, max +0.169336).

**Verdict:** improvement in two of four temporal folds — regime-dependent, inconclusive.

### lightgbm: `microstructure_v1` − `ohlcv14`

| fold | outer period | Δ net | Δ ann. Sharpe | Δ max DD (lower is better) | Δ exposure | Δ trades | Δ macro F1 | Δ dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | **-0.068249** | -0.7263 | +0.0767 | +0.0454 | +36 | +0.0099 | +0.1057 |
| 1 | 2023-09-24 → 2024-04-12 | **+0.161851** | +0.6985 | -0.1621 | -0.1124 | -89 | -0.0344 | +0.1184 |
| 2 | 2024-04-12 → 2024-10-30 | **-0.125935** | -1.5334 | +0.0904 | +0.0278 | +22 | -0.0006 | -0.1268 |
| 3 | 2024-10-30 → 2025-05-19 | **-0.279202** | -2.6329 | +0.0997 | -0.0265 | -21 | -0.0139 | -0.1678 |

Net return improved in **1 of 4** temporal folds (mean Δ -0.077884, min -0.279202, max +0.161851).

**Verdict:** improvement in one of four temporal folds — weak evidence against.

### lightgbm: `ohlcv14_plus_microstructure_v1` − `ohlcv14`

| fold | outer period | Δ net | Δ ann. Sharpe | Δ max DD (lower is better) | Δ exposure | Δ trades | Δ macro F1 | Δ dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | **-0.044106** | -0.6547 | +0.0358 | +0.0240 | +19 | +0.0053 | +0.1182 |
| 1 | 2023-09-24 → 2024-04-12 | **+0.212653** | +1.8894 | -0.0843 | -0.0581 | -46 | -0.0179 | +0.0187 |
| 2 | 2024-04-12 → 2024-10-30 | **+0.058252** | +1.4730 | -0.0539 | -0.0202 | -16 | -0.0051 | +0.0977 |
| 3 | 2024-10-30 → 2025-05-19 | **-0.190611** | -1.4911 | +0.0586 | -0.0240 | -19 | -0.0119 | -0.0818 |

Net return improved in **2 of 4** temporal folds (mean Δ +0.009047, min -0.190611, max +0.212653).

**Verdict:** improvement in two of four temporal folds — regime-dependent, inconclusive.

### xgboost: `microstructure_v1` − `ohlcv14`

| fold | outer period | Δ net | Δ ann. Sharpe | Δ max DD (lower is better) | Δ exposure | Δ trades | Δ macro F1 | Δ dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | **+0.099991** | +1.0878 | -0.0848 | -0.0341 | -27 | -0.0166 | -0.0320 |
| 1 | 2023-09-24 → 2024-04-12 | **-0.027956** | -0.4167 | -0.0299 | -0.0076 | -6 | -0.0020 | +0.2087 |
| 2 | 2024-04-12 → 2024-10-30 | **-0.195050** | -2.0143 | +0.2130 | +0.0770 | +61 | +0.0106 | +0.0285 |
| 3 | 2024-10-30 → 2025-05-19 | **+0.138136** | +0.1119 | +0.0690 | +0.1048 | +83 | +0.0153 | -0.0926 |

Net return improved in **2 of 4** temporal folds (mean Δ +0.003780, min -0.195050, max +0.138136).

**Verdict:** improvement in two of four temporal folds — regime-dependent, inconclusive.

### xgboost: `ohlcv14_plus_microstructure_v1` − `ohlcv14`

| fold | outer period | Δ net | Δ ann. Sharpe | Δ max DD (lower is better) | Δ exposure | Δ trades | Δ macro F1 | Δ dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | **-0.121440** | -1.1696 | +0.0811 | +0.0480 | +38 | +0.0094 | -0.0041 |
| 1 | 2023-09-24 → 2024-04-12 | **-0.008400** | -0.1537 | +0.0047 | +0.0013 | +1 | +0.0004 | +0.0587 |
| 2 | 2024-04-12 → 2024-10-30 | **+0.066100** | +1.4502 | +0.0298 | +0.0303 | +24 | +0.0056 | +0.0133 |
| 3 | 2024-10-30 → 2025-05-19 | **-0.014600** | -0.3244 | -0.0120 | +0.0063 | +5 | +0.0017 | +0.0848 |

Net return improved in **1 of 4** temporal folds (mean Δ -0.019585, min -0.121440, max +0.066100).

**Verdict:** improvement in one of four temporal folds — weak evidence against.

## Long / short attribution

Additive decomposition of the realised trades a cell took. These are the two halves of
one reported result, not two standalone strategies: neither side was selected for, and
neither could have been traded on its own without the threshold that produced both.

| model | information set | long trades | long hit | long mean net | short trades | short hit | short mean net |
| --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | `ohlcv14` | 91 | 0.4835 | +0.000997 | 51 | 0.4706 | +0.000797 |
| logistic_regression | `microstructure_v1` | 112 | 0.4821 | +0.000382 | 72 | 0.3472 | -0.001504 |
| logistic_regression | `ohlcv14_plus_microstructure_v1` | 78 | 0.5256 | +0.001394 | 54 | 0.3889 | -0.001567 |
| lightgbm | `ohlcv14` | 135 | 0.5111 | +0.001070 | 71 | 0.3944 | -0.002388 |
| lightgbm | `microstructure_v1` | 82 | 0.4146 | -0.001018 | 72 | 0.4306 | -0.003295 |
| lightgbm | `ohlcv14_plus_microstructure_v1` | 90 | 0.5000 | +0.000884 | 54 | 0.4630 | -0.000694 |
| xgboost | `ohlcv14` | 83 | 0.4699 | +0.000589 | 28 | 0.5714 | -0.000641 |
| xgboost | `microstructure_v1` | 155 | 0.4387 | +0.000216 | 67 | 0.4776 | -0.000172 |
| xgboost | `ohlcv14_plus_microstructure_v1` | 135 | 0.4519 | -0.000294 | 44 | 0.4773 | -0.000656 |

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
