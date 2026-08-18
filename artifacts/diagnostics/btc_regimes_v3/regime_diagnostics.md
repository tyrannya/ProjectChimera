# Walk-forward regime diagnostics

5 run(s), read from disk. Sealed test block starts at row 48217 and is not opened by this tool.

| run | path | seed | folds | integrity |
| --- | --- | --- | --- | --- |
| btc_nested_v3_seed_42 | `artifacts/walkforward/btc_nested_v3_seed_42/walkforward.json` | 42 | 4 | ok |
| btc_nested_v3_seed_142 | `artifacts/walkforward/btc_nested_v3_seed_142/walkforward.json` | 142 | 4 | ok |
| btc_nested_v3_seed_242 | `artifacts/walkforward/btc_nested_v3_seed_242/walkforward.json` | 242 | 4 | ok |
| btc_nested_v3_seed_342 | `artifacts/walkforward/btc_nested_v3_seed_342/walkforward.json` | 342 | 4 | ok |
| btc_nested_v3_seed_442 | `artifacts/walkforward/btc_nested_v3_seed_442/walkforward.json` | 442 | 4 | ok |

## Integrity

Every run: three regions per fold in order, outer blocks disjoint, no row at or beyond the sealed boundary, sealed test not evaluated, and the recorded across-fold summary reproduced from the per-fold reports.

## Fold geometry (identical across runs)

| fold | train | inner validation | outer validation |
| --- | --- | --- | --- |
| 0 | [0, 21697) | [21697, 26518) | [26518, 31339) |
| 1 | [0, 26518) | [26518, 31339) | [31339, 36160) |
| 2 | [0, 31339) | [31339, 36160) | [36160, 40981) |
| 3 | [0, 36160) | [36160, 40981) | [40981, 45802) |

## Outer validation across runs

Each run's across-fold mean, then the spread of those means. The spread is seed sensitivity of the whole procedure — retraining changes the answer by this much.

| model | metric | mean of run means | std | min | max |
| --- | --- | --- | --- | --- | --- |
| majority_baseline | macro_f1 | 0.18885 | 0 | 0.18885 | 0.18885 |
| majority_baseline | directional_accuracy | 0.39655 | 0 | 0.39655 | 0.39655 |
| majority_baseline | coverage | 1 | 0 | 1 | 1 |
| majority_baseline | calibration_error | 0.60345 | 0 | 0.60345 | 0.60345 |
| majority_baseline | n_trades | 789.25 | 0 | 789.25 | 789.25 |
| majority_baseline | net_return | -0.667118 | 0 | -0.667118 | -0.667118 |
| majority_baseline | sharpe | -4.28892 | 0 | -4.28892 | -4.28892 |
| majority_baseline | max_drawdown | 0.701175 | 0 | 0.701175 | 0.701175 |
| momentum_baseline | macro_f1 | 0.270275 | 0 | 0.270275 | 0.270275 |
| momentum_baseline | directional_accuracy | 0.358175 | 0 | 0.358175 | 0.358175 |
| momentum_baseline | coverage | 0.998025 | 0 | 0.998025 | 0.998025 |
| momentum_baseline | calibration_error | 0.642025 | 0 | 0.642025 | 0.642025 |
| momentum_baseline | n_trades | 789.25 | 0 | 789.25 | 789.25 |
| momentum_baseline | net_return | -0.794387 | 0 | -0.794387 | -0.794387 |
| momentum_baseline | sharpe | -5.8187 | 0 | -5.8187 | -5.8187 |
| momentum_baseline | max_drawdown | 0.818375 | 0 | 0.818375 | 0.818375 |
| mtst | macro_f1 | 0.160395 | 0.005304 | 0.156 | 0.168575 |
| mtst | directional_accuracy | 0.52015 | 0.022419 | 0.495025 | 0.546525 |
| mtst | coverage | 0.03599 | 0.006166 | 0.02975 | 0.045075 |
| mtst | calibration_error | 0.108695 | 0.011008 | 0.0895 | 0.11715 |
| mtst | n_trades | 72.2 | 6.05547 | 63.5 | 78.75 |
| mtst | net_return | -0.029359 | 0.021987 | -0.061914 | -0.009296 |
| mtst | sharpe | -0.781495 | 1.19076 | -2.08672 | 0.320725 |
| mtst | max_drawdown | 0.172605 | 0.021044 | 0.14125 | 0.19135 |

## Per-fold spread across runs (mtst, outer validation)

| fold | metric | mean | std | min | max |
| --- | --- | --- | --- | --- | --- |
| 0 | macro_f1 | 0.19278 | 0.017981 | 0.1778 | 0.2218 |
| 1 | macro_f1 | 0.161 | 0.013502 | 0.1399 | 0.1768 |
| 2 | macro_f1 | 0.15406 | 0.01379 | 0.133 | 0.1661 |
| 3 | macro_f1 | 0.13374 | 0.019773 | 0.1096 | 0.1601 |
| 0 | directional_accuracy | 0.5369 | 0.055163 | 0.4854 | 0.6047 |
| 1 | directional_accuracy | 0.56988 | 0.019131 | 0.5493 | 0.5959 |
| 2 | directional_accuracy | 0.47334 | 0.0148 | 0.4537 | 0.4881 |
| 3 | directional_accuracy | 0.50048 | 0.040378 | 0.445 | 0.5436 |
| 0 | coverage | 0.02432 | 0.020964 | 0.0077 | 0.0585 |
| 1 | coverage | 0.03702 | 0.014175 | 0.0149 | 0.0532 |
| 2 | coverage | 0.04904 | 0.017345 | 0.0227 | 0.0657 |
| 3 | coverage | 0.03358 | 0.024058 | 0.0029 | 0.0627 |
| 0 | calibration_error | 0.20594 | 0.029698 | 0.1633 | 0.2353 |
| 1 | calibration_error | 0.10498 | 0.012802 | 0.0908 | 0.1206 |
| 2 | calibration_error | 0.08316 | 0.023324 | 0.0572 | 0.1096 |
| 3 | calibration_error | 0.0407 | 0.023418 | 0.0169 | 0.0796 |
| 0 | n_trades | 41.4 | 23.5754 | 23 | 76 |
| 1 | n_trades | 70.2 | 22.3652 | 34 | 91 |
| 2 | n_trades | 102.8 | 30.2688 | 50 | 123 |
| 3 | n_trades | 74.4 | 47.6844 | 12 | 133 |
| 0 | net_return | -0.063989 | 0.10852 | -0.22219 | 0.073657 |
| 1 | net_return | 0.180662 | 0.199114 | -0.080952 | 0.387633 |
| 2 | net_return | -0.180108 | 0.048724 | -0.249143 | -0.124785 |
| 3 | net_return | -0.054002 | 0.114081 | -0.19528 | 0.064764 |
| 0 | sharpe | -1.62384 | 5.88604 | -6.8696 | 8.075 |
| 1 | sharpe | 4.84052 | 4.93116 | -2.3425 | 10.6563 |
| 2 | sharpe | -4.09394 | 1.75503 | -6.3561 | -1.8055 |
| 3 | sharpe | -2.24872 | 4.55905 | -7.8275 | 1.3333 |
| 0 | max_drawdown | 0.15356 | 0.095755 | 0.0757 | 0.311 |
| 1 | max_drawdown | 0.08996 | 0.031579 | 0.052 | 0.1376 |
| 2 | max_drawdown | 0.28122 | 0.045873 | 0.2304 | 0.3494 |
| 3 | max_drawdown | 0.16568 | 0.059263 | 0.0869 | 0.243 |

## Selection stability (chosen on inner validation)

| fold | threshold mean | threshold std | threshold range | epoch mean | epoch std |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.4120 | 0.0363 | 0.36–0.46 | 3.60 | 1.82 |
| 1 | 0.4560 | 0.0261 | 0.44–0.50 | 5.40 | 1.95 |
| 2 | 0.4440 | 0.0219 | 0.42–0.46 | 3.40 | 0.55 |
| 3 | 0.4960 | 0.0456 | 0.44–0.56 | 4.00 | 1.58 |

## Against the baselines

MTST beat both baselines on net return in 20/20 run-folds, and in a majority of folds in 5/5 runs.

| run | folds won |
| --- | --- |
| btc_nested_v3_seed_42 | 4/4 |
| btc_nested_v3_seed_142 | 4/4 |
| btc_nested_v3_seed_242 | 4/4 |
| btc_nested_v3_seed_342 | 4/4 |
| btc_nested_v3_seed_442 | 4/4 |

## Dataset / sealed-test status

- Processed dataset: `data/datasets/binance_BTC_USDT_1h.parquet`
- Raw OHLCV: `data/raw/binance/BTC_USDT_1h.parquet`
- Rows read: `[0, 48217)`. The frame is truncated at the sealed boundary on load, so no statistic below can have seen a sealed row.
- Model-facing statistics use the scored rows only (seq_len 64, horizon 6), selected through the same index logic the run used.
- Highest outer row used: 45801 (sealed test starts at 48217).
- Sealed test evaluated: **no**.

## Market regime statistics (outer blocks)

Identical across seeds by construction — the market in a fold does not
depend on which seed trained on it.

Computed over the rows the model was **scored** on — window warm-up,
label embargo and market-gap filtering applied — not over every row the
block spans. Both counts are shown.

| fold | block rows | scored rows | period | fut ret mean | fut ret std | fut ret abs | pos frac |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 4821 | 4683 | 2023-03-06 to 2023-09-24 | +0.000268 | 0.011185 | 0.006570 | 0.4931 |
| 1 | 4821 | 4752 | 2023-09-27 to 2024-04-12 | +0.001314 | 0.012499 | 0.008245 | 0.5400 |
| 2 | 4821 | 4752 | 2024-04-15 to 2024-10-30 | +0.000191 | 0.012558 | 0.008592 | 0.5141 |
| 3 | 4821 | 4752 | 2024-11-02 to 2025-05-19 | +0.000591 | 0.013780 | 0.009251 | 0.5221 |

### Feature regime

| fold | ema_cross mean | ema_cross >0 | atr_norm mean | atr_norm p90 | realized_vol mean | realized_vol p90 | hl_range mean | volume_z mean | volume_z p90abs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | +0.000210 | 0.4798 | 0.005576 | 0.008914 | 0.003764 | 0.006643 | 0.005584 | -0.010327 | 1.437431 |
| 1 | +0.001421 | 0.6252 | 0.007143 | 0.011048 | 0.004622 | 0.007321 | 0.007173 | +0.001536 | 1.528431 |
| 2 | +0.000114 | 0.5221 | 0.007130 | 0.010315 | 0.004762 | 0.007423 | 0.007120 | +0.014858 | 1.521130 |
| 3 | +0.000553 | 0.5589 | 0.007613 | 0.012402 | 0.005120 | 0.008408 | 0.007626 | +0.011161 | 1.528455 |

### Target distribution

| fold | SHORT n | SHORT frac | SHORT mean ret | HOLD n | HOLD frac | HOLD mean ret | LONG n | LONG frac | LONG mean ret |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1587 | 0.3389 | -0.008817 | 1577 | 0.3367 | -0.000017 | 1519 | 0.3244 | +0.010055 |
| 1 | 1613 | 0.3394 | -0.009871 | 1110 | 0.2336 | -0.000024 | 2029 | 0.4270 | +0.010938 |
| 2 | 1804 | 0.3796 | -0.010789 | 1002 | 0.2109 | +0.000011 | 1946 | 0.4095 | +0.010462 |
| 3 | 1820 | 0.3830 | -0.011054 | 911 | 0.1917 | -0.000022 | 2021 | 0.4253 | +0.011356 |

### Raw market behaviour (timestamp-aligned OHLCV)

A view of the market over the **whole block span**, independent of
which rows were scorable. Not directly comparable row-for-row with the
model-facing statistics above.

| fold | start close | end close | market return | mean hourly | std hourly | ann. vol | mean abs hourly | pos candles | max DD | gaps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 22359.09 | 26573.10 | +0.1885 | +0.000055 | 0.004522 | 0.4232 | 0.002582 | 0.5053 | 0.2084 | 1 |
| 1 | 26673.07 | 69888.00 | +1.6202 | +0.000213 | 0.005143 | 0.4814 | 0.003361 | 0.5204 | 0.2018 | 0 |
| 2 | 69270.01 | 72378.02 | +0.0449 | +0.000024 | 0.005402 | 0.5056 | 0.003463 | 0.5038 | 0.3054 | 0 |
| 3 | 71983.99 | 103226.29 | +0.4340 | +0.000091 | 0.005719 | 0.5353 | 0.003770 | 0.5086 | 0.3094 | 0 |

## Per-fold model stability across seeds

| fold | net return mean | std | median | positive seeds | dir acc mean±std | coverage mean±std | trades mean±std | threshold mean±std | threshold range | epoch mean±std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | -0.063989 | 0.108520 | -0.071335 | 1/5 | 0.5369±0.0552 | 0.0243±0.0210 | 41.4±23.6 | 0.4120±0.0363 | 0.36–0.46 | 3.60±1.82 |
| 1 | +0.180662 | 0.199114 | +0.172551 | 4/5 | 0.5699±0.0191 | 0.0370±0.0142 | 70.2±22.4 | 0.4560±0.0261 | 0.44–0.50 | 5.40±1.95 |
| 2 | -0.180108 | 0.048724 | -0.170316 | 0/5 | 0.4733±0.0148 | 0.0490±0.0173 | 102.8±30.3 | 0.4440±0.0219 | 0.42–0.46 | 3.40±0.55 |
| 3 | -0.054002 | 0.114081 | +0.007225 | 3/5 | 0.5005±0.0404 | 0.0336±0.0241 | 74.4±47.7 | 0.4960±0.0456 | 0.44–0.56 | 4.00±1.58 |

## Best vs worst regime — fold 1 (best) vs fold 2 (worst)

Selected from the data: highest and lowest mean MTST outer net return across
seeds. Ranked by difference relative to the larger magnitude. **These are
coincidences, not causes** — a row says the two folds differ, nothing more.

| rank | group | metric | best fold | worst fold | absolute diff | relative |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | model | mtst net return | +0.180662 | -0.180108 | +0.36077 | +1.9969 |
| 2 | market | BTC market return | +1.62017 | +0.044868 | +1.5753 | +0.9723 |
| 3 | trend | ema_cross mean | +0.00142135 | +0.00011408 | +0.00130727 | +0.9197 |
| 4 | volume | volume_z mean | +0.00153649 | +0.0148584 | -0.0133219 | -0.8966 |
| 5 | target | future_return mean | +0.00131394 | +0.00019062 | +0.00112332 | +0.8549 |
| 6 | market | market max drawdown | +0.201837 | +0.305383 | -0.103546 | -0.3391 |
| 7 | model | trades | +70.2 | +102.8 | -32.6 | -0.3171 |
| 8 | model | coverage | +0.03702 | +0.04904 | -0.01202 | -0.2451 |
| 9 | volume | volume_change mean | +0.151282 | +0.187974 | -0.0366921 | -0.1952 |
| 10 | model | directional accuracy | +0.56988 | +0.47334 | +0.09654 | +0.1694 |
| 11 | trend | ema_cross fraction positive | +0.62521 | +0.522096 | +0.103114 | +0.1649 |
| 12 | target | SHORT fraction | +0.339436 | +0.37963 | -0.0401936 | -0.1059 |
| 13 | target | HOLD fraction | +0.233586 | +0.210859 | +0.0227273 | +0.0973 |
| 14 | volatility | atr_norm p90 | +0.0110475 | +0.0103153 | +0.00073226 | +0.0663 |
| 15 | market | annualised volatility | +0.481374 | +0.505619 | -0.0242447 | -0.0480 |
| 16 | target | positive future_return fraction | +0.539983 | +0.514099 | +0.0258838 | +0.0479 |
| 17 | target | LONG fraction | +0.426978 | +0.409512 | +0.0174663 | +0.0409 |
| 18 | target | future_return mean abs | +0.00824543 | +0.00859157 | -0.00034614 | -0.0403 |
| 19 | market | positive candle fraction | +0.520431 | +0.503837 | +0.0165941 | +0.0319 |
| 20 | market | mean abs hourly return | +0.00336091 | +0.00346278 | -0.00010187 | -0.0294 |
| 21 | volatility | realized_vol mean | +0.00462179 | +0.00476181 | -0.00014002 | -0.0294 |
| 22 | model | selected threshold | +0.456 | +0.444 | +0.012 | +0.0263 |
| 23 | volatility | realized_vol p90 | +0.00732109 | +0.00742348 | -0.00010239 | -0.0138 |
| 24 | volatility | hl_range mean | +0.00717291 | +0.00712031 | +5.26e-05 | +0.0073 |
| 25 | volume | volume_z p90 abs | +1.52843 | +1.52113 | +0.00730063 | +0.0048 |
| 26 | target | future_return std | +0.0124986 | +0.0125579 | -5.932e-05 | -0.0047 |
| 27 | trend | ema_cross std | +0.00576335 | +0.00578853 | -2.518e-05 | -0.0043 |
| 28 | volatility | atr_norm mean | +0.00714286 | +0.00712967 | +1.319e-05 | +0.0018 |

## LONG vs SHORT attribution

Exact, from persisted outer predictions, using the same cost model as the
fold reports.

| side | trades | hit rate | mean net | median net | additive sum | coverage |
| --- | --- | --- | --- | --- | --- | --- |
| LONG | 991 | 0.5348 | +0.000328 | +0.000843 | +0.324939 | 0.0223 |
| SHORT | 453 | 0.4349 | -0.002175 | -0.001731 | -0.985301 | 0.0137 |

HOLD / no-trade samples: 91283 of 94695.

## Candidate hypotheses

Research questions suggested by the diagnostics, ranked by the size of the
difference behind them. **None is implemented, tuned or tested here.**

1. **Fold 2 behaves consistently worse than fold 1 across every seed observed, so it is worth characterising as a regime rather than dismissed as noise — then replicated on independent periods and assets before the characterisation is generalised.**
   Coincides with: across 5 seed(s) fold 1 spans [-0.080952, +0.387633] and fold 2 spans [-0.249143, -0.124785] with no overlap, and fold 2 is negative in all 5 observed seeds. Consistent within these seeds; one fold per regime is still a sample of one, and seed count is not a substitute for independent periods (signal strength 1.9969).

2. **The feature set may lack regime context: nothing in it states which regime the window belongs to, only what just happened within it.**
   Coincides with: trend orientation and volume behaviour differ between the folds while the feature contract is identical across every run. (signal strength 0.9197).

3. **LONG and SHORT may need separate decision thresholds rather than one symmetric cut.**
   Coincides with: per-direction attribution is available and can be read directly; the selected threshold varies by up to 0.046 across seeds on a single fold. (signal strength 0.2451).


## Limitations

- These are outer-validation blocks: nothing was fitted on them, which is
  what makes them comparable across seeds. They are still research blocks,
  re-run while the method was being built. Seed spread measures the stability
  of the research procedure, not out-of-sample performance, and this report is
  not a claim of profitability. The sealed test block remains unopened.
- Every difference reported above is a **coincidence in the data**, not a
  cause. One fold per regime is a sample of one; nothing here establishes that
  a market property produced a model outcome.
- Regime statistics describe the outer block only. The model was trained on
  everything before it, so a fold's difficulty also depends on how much its
  training window resembled it — which these numbers do not measure.
- Per-fold seed spread is not a confidence interval on performance.

