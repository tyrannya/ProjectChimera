# P2b robustness — leave-one-family-out ablation

Model: **xgboost**. Reference arm: `ohlcv14_plus_smc_v1`.

**Post-hoc and descriptive.** These comparisons were chosen after the canonical
P2b result was produced, on the same four outer blocks. They are a source of
hypotheses for a later research generation, not confirmation of anything, and no
model or threshold anywhere in P2b was selected using them.

**What a delta means.** `full − ablated` is what a family contributed *given the
other five and OHLCV14 were present*. A near-zero delta means the family added
nothing on top of the rest — which is not the same as the family being
uninformative, because another family may already carry the same information.

**Why sign consistency comes first.** Four temporal periods do not support a
significance claim, and their returns are serially dependent. How many of the four
agree on the sign is the honest summary.

## Marginal contribution by family

| family | columns | Δ net mean | Δ net min | Δ net max | folds agreeing | direction |
| --- | --- | --- | --- | --- | --- | --- |
| `structure` | 8 | -0.071992 | -0.373213 | +0.192067 | **2 of 4** | split |
| `liquidity` | 6 | +0.013459 | -0.068065 | +0.117035 | **2 of 4** | split |
| `breaks` | 6 | -0.013658 | -0.071609 | +0.084371 | **3 of 4** | negative |
| `sweeps` | 6 | -0.020924 | -0.123870 | +0.097353 | **2 of 4** | split |
| `displacement` | 5 | -0.041109 | -0.099910 | +0.036936 | **3 of 4** | negative |
| `fvg` | 8 | -0.039181 | -0.139782 | +0.031687 | **2 of 4** | split |

## Per fold

### `structure` — 8 columns removed

| fold | outer period | full net | ablated net | **Δ net** | Δ Sharpe | Δ max DD | Δ trades | Δ macro F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | -0.112009 | -0.142926 | **+0.030917** | +0.3413 | +0.0409 | -15 | -0.0056 |
| 1 | 2023-09-24 → 2024-04-12 | -0.080644 | +0.292569 | **-0.373213** | -5.0147 | +0.0304 | -17 | -0.0119 |
| 2 | 2024-04-12 → 2024-10-30 | +0.113964 | -0.078103 | **+0.192067** | +2.5654 | -0.0910 | -41 | -0.0110 |
| 3 | 2024-10-30 → 2025-05-19 | +0.047844 | +0.185582 | **-0.137738** | -1.2942 | +0.0706 | +29 | +0.0133 |

### `liquidity` — 6 columns removed

| fold | outer period | full net | ablated net | **Δ net** | Δ Sharpe | Δ max DD | Δ trades | Δ macro F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | -0.112009 | -0.157832 | **+0.045823** | +0.4378 | -0.0293 | +13 | +0.0028 |
| 1 | 2023-09-24 → 2024-04-12 | -0.080644 | -0.039686 | **-0.040958** | -0.5943 | +0.0294 | +7 | -0.0001 |
| 2 | 2024-04-12 → 2024-10-30 | +0.113964 | -0.003071 | **+0.117035** | +1.9438 | +0.0455 | +24 | +0.0095 |
| 3 | 2024-10-30 → 2025-05-19 | +0.047844 | +0.115909 | **-0.068065** | -0.9244 | +0.0636 | +28 | +0.0119 |

### `breaks` — 6 columns removed

| fold | outer period | full net | ablated net | **Δ net** | Δ Sharpe | Δ max DD | Δ trades | Δ macro F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | -0.112009 | -0.097744 | **-0.014265** | -0.2071 | -0.0105 | +18 | +0.0046 |
| 1 | 2023-09-24 → 2024-04-12 | -0.080644 | -0.009035 | **-0.071609** | -1.5514 | -0.0265 | -41 | -0.0154 |
| 2 | 2024-04-12 → 2024-10-30 | +0.113964 | +0.029593 | **+0.084371** | +1.4166 | -0.0549 | -50 | -0.0135 |
| 3 | 2024-10-30 → 2025-05-19 | +0.047844 | +0.100973 | **-0.053129** | -0.7144 | +0.0967 | +31 | +0.0129 |

### `sweeps` — 6 columns removed

| fold | outer period | full net | ablated net | **Δ net** | Δ Sharpe | Δ max DD | Δ trades | Δ macro F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | -0.112009 | -0.166073 | **+0.054064** | +0.0910 | -0.0384 | -62 | -0.0286 |
| 1 | 2023-09-24 → 2024-04-12 | -0.080644 | +0.030598 | **-0.111242** | -2.1233 | +0.0273 | +0 | +0.0003 |
| 2 | 2024-04-12 → 2024-10-30 | +0.113964 | +0.016611 | **+0.097353** | +1.3263 | -0.0009 | +16 | +0.0068 |
| 3 | 2024-10-30 → 2025-05-19 | +0.047844 | +0.171714 | **-0.123870** | -1.9175 | +0.0741 | +39 | +0.0147 |

### `displacement` — 5 columns removed

| fold | outer period | full net | ablated net | **Δ net** | Δ Sharpe | Δ max DD | Δ trades | Δ macro F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | -0.112009 | -0.012099 | **-0.099910** | -1.2987 | +0.0621 | +26 | +0.0066 |
| 1 | 2023-09-24 → 2024-04-12 | -0.080644 | -0.072431 | **-0.008213** | -0.1905 | -0.0025 | -4 | -0.0041 |
| 2 | 2024-04-12 → 2024-10-30 | +0.113964 | +0.077028 | **+0.036936** | +0.8914 | -0.0137 | -25 | -0.0091 |
| 3 | 2024-10-30 → 2025-05-19 | +0.047844 | +0.141094 | **-0.093250** | -0.9169 | +0.0458 | +18 | +0.0096 |

### `fvg` — 8 columns removed

| fold | outer period | full net | ablated net | **Δ net** | Δ Sharpe | Δ max DD | Δ trades | Δ macro F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | -0.112009 | +0.027773 | **-0.139782** | -1.8199 | +0.0873 | +21 | +0.0053 |
| 1 | 2023-09-24 → 2024-04-12 | -0.080644 | -0.112331 | **+0.031687** | +0.5203 | -0.0258 | +8 | +0.0026 |
| 2 | 2024-04-12 → 2024-10-30 | +0.113964 | +0.083240 | **+0.030724** | +0.9798 | -0.0695 | -69 | -0.0205 |
| 3 | 2024-10-30 → 2025-05-19 | +0.047844 | +0.127198 | **-0.079354** | -0.7083 | +0.0759 | -4 | -0.0041 |

## Importance share by family

Native `None` importance from the fitted trees, summed
across each feature's 64 timesteps and then across its family. Tree
importance describes what the fitted model used, not what the market rewards —
a column can dominate because it is finely resolved rather than informative.

| family | fold 0 | fold 1 | fold 2 | fold 3 | mean |
| --- | --- | --- | --- | --- | --- |
| `ohlcv14` | 0.4031 | 0.4072 | 0.4167 | 0.4074 | **0.4086** |
| `structure` | 0.2002 | 0.1990 | 0.1947 | 0.1963 | **0.1975** |
| `fvg` | 0.1113 | 0.1064 | 0.1113 | 0.1136 | **0.1107** |
| `liquidity` | 0.0882 | 0.0834 | 0.0822 | 0.0815 | **0.0838** |
| `sweeps` | 0.0788 | 0.0819 | 0.0759 | 0.0789 | **0.0789** |
| `displacement` | 0.0695 | 0.0690 | 0.0683 | 0.0679 | **0.0687** |
| `breaks` | 0.0490 | 0.0531 | 0.0510 | 0.0544 | **0.0519** |

### Feature stability across the four temporal folds

How often each column appeared in the fold's top 15, and at what rank. A column
in the top 15 of one fold and absent from the other three is describing that
period.

| feature | folds in top 15 | ranks |
| --- | --- | --- |
| `atr_norm` | 4 of 4 | 1, 1, 1, 1 |
| `ema_cross` | 4 of 4 | 4, 5, 3, 4 |
| `macd_hist_norm` | 4 of 4 | 8, 7, 7, 7 |
| `macd_norm` | 4 of 4 | 2, 2, 2, 2 |
| `realized_vol` | 4 of 4 | 3, 3, 4, 3 |
| `smc_bars_since_sweep_high` | 4 of 4 | 9, 8, 8, 9 |
| `smc_bars_since_sweep_low` | 4 of 4 | 7, 9, 10, 10 |
| `smc_eqh_dist_atr` | 4 of 4 | 6, 6, 6, 6 |
| `smc_eql_dist_atr` | 4 of 4 | 5, 4, 5, 5 |
| `smc_range_position` | 4 of 4 | 11, 10, 12, 12 |
| `smc_range_width_atr` | 4 of 4 | 14, 13, 11, 11 |
| `smc_bars_since_break` | 3 of 4 | 15, 11, 14, — |
| `smc_bear_fvg_dist_atr` | 3 of 4 | 10, 12, —, 14 |
| `smc_dist_swing_high_atr` | 3 of 4 | 12, 14, 15, — |
| `ema_slow_ratio` | 2 of 4 | —, —, 13, 13 |
| `hl_range` | 2 of 4 | —, —, 9, 8 |
| `smc_bull_fvg_dist_atr` | 2 of 4 | 13, 15, —, — |
| `rsi_centered` | 1 of 4 | —, —, —, 15 |
