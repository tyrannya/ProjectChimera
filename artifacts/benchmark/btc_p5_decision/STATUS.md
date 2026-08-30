# CURRENT

## P5 — the preregistered decision

**Outcome: `negative`.** 1 of 4 folds improved, against a bar of 3.

Deciding cell: `xgboost x ohlcv14_plus_mtf_v1` against `xgboost x ohlcv14`, outer-validation cost-aware net return at cost multiplier 1.0.

Preregistration: `sha256:dc4bd73a078a166e366381c2297bcdad0328c5e08da8def928e8e1f37f04ed8c`.

| fold | period | control | combined | delta | improved | trades |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | `2023-03-04 07:00:00+00:00` .. `2023-09-24 16:00:00+00:00` | `-0.227758` | `-0.112678` | `0.11508` | yes | 207/22 |
| 1 | `2023-09-24 17:00:00+00:00` .. `2024-04-12 13:00:00+00:00` | `0.048244` | `-0.027115` | `-0.075359` | no | 26/21 |
| 2 | `2024-04-12 14:00:00+00:00` .. `2024-10-30 10:00:00+00:00` | `0.133502` | `0.093658` | `-0.039844` | no | 29/8 ⚑ |
| 3 | `2024-10-30 11:00:00+00:00` .. `2025-05-19 07:00:00+00:00` | `0.429306` | `0.245659` | `-0.183647` | no | 47/48 |

Mean delta `-0.0459425`, worst fold `-0.183647`. **Both descriptive.** The preregistration makes them decisive in neither direction: they may not rescue a fold-count failure and may not veto a fold-count pass.

### Availability

44171 of 45802 rows eligible (0.96439); **4 of 4 folds available**, gate passed.

Recomputed here from the committed snapshot and checked against the universe digest every cell recorded, so these figures describe the rows the models were actually fitted on rather than a claim the cells made about themselves.

### What this is, and is not

P5 is negative. Do not tune mtf_v1's constants — they were predeclared, and searching them against these four outer blocks would convert a negative result into a fitted one. Do not create mtf_v2. The next research move changes axis

### Context, deciding nothing

| model | arm | improved | deltas | mean | role |
| --- | --- | --- | --- | --- | --- |
| `xgboost` | `mtf_v1` | 1 of 4 | `[0.010104, -0.218057, -0.172724, -0.420662]` | `-0.20033475` | secondary (descriptive; cannot switch the deciding cell) |
| `xgboost` | `ohlcv14_plus_mtf_v1` | 1 of 4 | `[0.11508, -0.075359, -0.039844, -0.183647]` | `-0.0459425` | primary |
| `logistic_regression` | `mtf_v1` | 0 of 4 | `[-0.01819, -0.415199, -0.395549, -0.023701]` | `-0.21315975` | secondary (descriptive; cannot switch the deciding cell) |
| `logistic_regression` | `ohlcv14_plus_mtf_v1` | 2 of 4 | `[0.044792, -0.185182, -0.289749, 0.029309]` | `-0.1002075` | secondary (descriptive; cannot switch the deciding cell) |
| `lightgbm` | `mtf_v1` | 2 of 4 | `[0.039942, -0.187182, 0.209617, -0.06664]` | `-0.00106575` | secondary (descriptive; cannot switch the deciding cell) |
| `lightgbm` | `ohlcv14_plus_mtf_v1` | 2 of 4 | `[-0.124697, 0.02957, 0.17878, -0.125519]` | `-0.0104665` | secondary (descriptive; cannot switch the deciding cell) |
