# P2b — what the four outer periods were, and where market structure helped

**Descriptive. Nothing here is fitted, and nothing here becomes a filter.**

Four temporal blocks is few enough that a per-fold result is a statement about four
specific stretches of market as much as about the information set. This describes
them from research-visible rows only and lays the deltas beside the description, so
a reader can judge which. Fitting a rule that trades market structure only in the
periods where it won — on the same four periods that revealed the win — would fit
four observations and report the fit as a result. That is not done here, and the
output is a hypothesis for a later generation to test on periods it has not seen.

## The four outer periods

| fold | period | rows | total return | ann. vol | high/low range | max DD | directionality | ATR% | autocorr | volume trend | LONG/SHORT | HOLD share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | 4821 | +0.1885 | 0.4278 | 0.6269 | 0.2084 | 0.0123 | 0.00560 | -0.0411 | -0.7196 | 0.9621 | 0.3350 |
| 1 | 2023-09-24 → 2024-04-12 | 4821 | +1.6202 | 0.4813 | 1.8386 | 0.2018 | 0.0552 | 0.00713 | -0.0166 | +0.2740 | 1.248 | 0.2348 |
| 2 | 2024-04-12 → 2024-10-30 | 4821 | +0.0449 | 0.5058 | 0.5025 | 0.3054 | 0.0030 | 0.00720 | -0.0229 | -0.1083 | 1.078 | 0.2099 |
| 3 | 2024-10-30 → 2025-05-19 | 4821 | +0.4340 | 0.5355 | 0.6397 | 0.3094 | 0.0187 | 0.00762 | -0.0248 | -0.2720 | 1.1038 | 0.1931 |

`directionality` is the efficiency ratio: the net move divided by the summed
absolute candle moves. 1.0 is a straight line; near 0 is a round trip. `chop` is
its complement.

## Where market structure helped, and where it hurt

### lightgbm

| fold | period | directionality | ann. vol | Δ net (`smc_v1` − `ohlcv14`) | Δ net (`combined` − `ohlcv14`) |
| --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | 0.0123 | 0.4278 | +0.216954 | -0.174439 |
| 1 | 2023-09-24 → 2024-04-12 | 0.0552 | 0.4813 | +0.119672 | +0.017872 |
| 2 | 2024-04-12 → 2024-10-30 | 0.0030 | 0.5058 | -0.076178 | -0.003105 |
| 3 | 2024-10-30 → 2025-05-19 | 0.0187 | 0.5355 | -0.105672 | -0.256175 |

### logistic_regression

| fold | period | directionality | ann. vol | Δ net (`smc_v1` − `ohlcv14`) | Δ net (`combined` − `ohlcv14`) |
| --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | 0.0123 | 0.4278 | -0.121313 | -0.106727 |
| 1 | 2023-09-24 → 2024-04-12 | 0.0552 | 0.4813 | -0.094092 | +0.003670 |
| 2 | 2024-04-12 → 2024-10-30 | 0.0030 | 0.5058 | -0.103520 | -0.156862 |
| 3 | 2024-10-30 → 2025-05-19 | 0.0187 | 0.5355 | -0.029896 | +0.105429 |

### xgboost

| fold | period | directionality | ann. vol | Δ net (`smc_v1` − `ohlcv14`) | Δ net (`combined` − `ohlcv14`) |
| --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 → 2023-09-24 | 0.0123 | 0.4278 | +0.239248 | -0.010438 |
| 1 | 2023-09-24 → 2024-04-12 | 0.0552 | 0.4813 | -0.063793 | -0.113170 |
| 2 | 2024-04-12 → 2024-10-30 | 0.0030 | 0.5058 | -0.104013 | +0.122389 |
| 3 | 2024-10-30 → 2025-05-19 | 0.0187 | 0.5355 | -0.025778 | -0.057785 |

## Observations

- fold 1 was the most directional period (efficiency ratio 0.0552) and fold 2 the choppiest (0.0030)
- fold 3 had the highest annualised realised volatility (0.5355), fold 0 the lowest (0.4278)
- lightgbm: `smc_v1` beat the control in 2 of 4 folds (folds [0, 1]), mean Δ net +0.038694
- lightgbm: `ohlcv14_plus_smc_v1` beat the control in 1 of 4 folds (folds [1]), mean Δ net -0.103962
- logistic_regression: `smc_v1` beat the control in 0 of 4 folds, mean Δ net -0.087205
- logistic_regression: `ohlcv14_plus_smc_v1` beat the control in 2 of 4 folds (folds [1, 3]), mean Δ net -0.038623
- xgboost: `smc_v1` beat the control in 1 of 4 folds (folds [0]), mean Δ net +0.011416
- xgboost: `ohlcv14_plus_smc_v1` beat the control in 1 of 4 folds (folds [2]), mean Δ net -0.014751

These are readings of the table above, not causal claims. With four periods, any
association between a market statistic and a delta is compatible with coincidence,
and the honest use of it is to state a hypothesis a later generation can test.
