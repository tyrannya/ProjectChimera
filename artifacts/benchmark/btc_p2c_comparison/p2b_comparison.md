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

### lightgbm

| information set | fold | outer period | samples | thr | trades | exposure | turnover | gross | costs | **net** | max DD | ann. Sharpe | per-trade Sharpe | win rate | profit factor | macro F1 | dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ohlcv14` | 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.70 | 30 | 0.0379 | 60.0 | +0.020906 | 0.0600 | **-0.046515** | 0.1085 | -0.5360 | -0.0533 | 0.3333 | 0.8317 | 0.1774 | 0.3433 |
| `ohlcv14` | 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.60 | 96 | 0.1212 | 192.0 | -0.007094 | 0.1920 | **-0.189062** | 0.1978 | -1.5579 | -0.1414 | 0.4688 | 0.6876 | 0.1623 | 0.4530 |
| `ohlcv14` | 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.66 | 21 | 0.0265 | 42.0 | +0.019137 | 0.0420 | **-0.025714** | 0.0744 | -0.4007 | -0.0607 | 0.4286 | 0.8336 | 0.1228 | 0.4737 |
| `ohlcv14` | 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.62 | 59 | 0.0745 | 118.0 | +0.353907 | 0.1180 | **+0.256085** | 0.0369 | 2.3691 | 0.2468 | 0.5593 | 2.1441 | 0.1290 | 0.5909 |

### xgboost

| information set | fold | outer period | samples | thr | trades | exposure | turnover | gross | costs | **net** | max DD | ann. Sharpe | per-trade Sharpe | win rate | profit factor | macro F1 | dir. acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ohlcv14` | 0 | 2023-03-04 → 2023-09-24 | 4683 | 0.64 | 80 | 0.1010 | 160.0 | +0.062295 | 0.1600 | **-0.101571** | 0.1478 | -1.0454 | -0.0802 | 0.4625 | 0.7578 | 0.1968 | 0.4405 |
| `ohlcv14` | 1 | 2023-09-24 → 2024-04-12 | 4752 | 0.72 | 11 | 0.0139 | 22.0 | +0.056220 | 0.0220 | **+0.032526** | 0.0393 | 0.6611 | 0.1482 | 0.4545 | 1.5588 | 0.1293 | 0.3913 |
| `ohlcv14` | 2 | 2024-04-12 → 2024-10-30 | 4752 | 0.70 | 4 | 0.0051 | 8.0 | -0.000356 | 0.0080 | **-0.008425** | 0.0144 | -0.3705 | -0.2624 | 0.5000 | 0.4235 | 0.1172 | 0.4286 |
| `ohlcv14` | 3 | 2024-10-30 → 2025-05-19 | 4752 | 0.68 | 16 | 0.0202 | 32.0 | +0.134772 | 0.0320 | **+0.105629** | 0.0310 | 1.8526 | 0.3872 | 0.6875 | 3.2504 | 0.1129 | 0.5926 |

## Across the four temporal folds

| model | information set | net mean | net std | net min | net median | net max | positive folds | Sharpe mean | max DD mean | exposure mean | macro F1 mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | `ohlcv14` | +0.008171 | 0.123833 | -0.157307 | +0.029612 | +0.130769 | **2 of 4** | 0.4988 | 0.1210 | 0.0707 | 0.1495 |
| lightgbm | `ohlcv14` | -0.001302 | 0.186317 | -0.189062 | -0.036115 | +0.256085 | **1 of 4** | -0.0314 | 0.1044 | 0.0650 | 0.1479 |
| xgboost | `ohlcv14` | +0.007040 | 0.086419 | -0.101571 | +0.012050 | +0.105629 | **2 of 4** | 0.2745 | 0.0581 | 0.0350 | 0.1391 |

## Incremental value of market structure

Per model, per fold: the information set minus the `ohlcv14` control on the same rows.

## Long / short attribution

Additive decomposition of the realised trades a cell took. These are the two halves of
one reported result, not two standalone strategies: neither side was selected for, and
neither could have been traded on its own without the threshold that produced both.

| model | information set | long trades | long hit | long mean net | short trades | short hit | short mean net |
| --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | `ohlcv14` | 150 | 0.5133 | +0.000794 | 74 | 0.4730 | -0.001024 |
| lightgbm | `ohlcv14` | 135 | 0.5111 | +0.001070 | 71 | 0.3944 | -0.002388 |
| xgboost | `ohlcv14` | 83 | 0.4699 | +0.000589 | 28 | 0.5714 | -0.000641 |

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
