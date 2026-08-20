# P2a — do simple models beat MTST on MTST's own information?

Does model complexity add predictive or economic value when simple models receive exactly the same information as MTST?

Three untuned, predeclared models — logistic regression, LightGBM, XGBoost —
were fitted on the same samples MTST was fitted on: each row is MTST's
`64 x 14` window flattened to `896` values
(column j = timestep t * n_features + feature f, t oldest first). MTST was **not retrained**: its numbers are the
frozen v4 artifacts, read only.

**Research contract:** `btc-usdt-1h-gen1`, semantic identity `sha256:dca6d0a891a257197a7c8aecec04fdba0a0b3009cfe93c5f3a397d458ab4a1de`.

**Research input:** `sha256:1c09218442cdf98cfe2f49ac521c70b2d7692717d7488e8105c972a3d2ac4740`.

**Sealed test:** anchor `2025-08-27T23:00:00+00:00`, first sealed row 48217, `evaluated: false`.

## Outer validation across every seed and fold

`ann. Sharpe` and `per-trade Sharpe` are `n/a` where undefined, never zero.

| model | mean net return | ± std | positive folds | trades | exposure | max DD | ann. Sharpe | per-trade Sharpe | macro F1 | dir acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | -0.0558 | 0.1795 | 10/20 | 95.5 | 0.1206 | 0.1968 | -0.05 | 0.10 | 0.1646 | 0.5362 |
| lightgbm | -0.0013 | 0.1655 | 5/20 | 51.5 | 0.0650 | 0.1044 | -0.03 | -0.00 | 0.1479 | 0.4652 |
| xgboost | +0.0070 | 0.0768 | 10/20 | 27.8 | 0.0350 | 0.0581 | 0.27 | 0.05 | 0.1391 | 0.4632 |
| mtst | -0.0193 | 0.1790 | 9/20 | 69.4 | 0.0876 | 0.1682 | -0.06 | 0.02 | 0.1585 | 0.5294 |
| majority_baseline | -0.6671 | 0.1340 | 0/20 | 789.2 | 0.9965 | 0.7012 | -4.23 | -0.11 | 0.1888 | 0.3966 |
| momentum_baseline | -0.7944 | 0.0296 | 0/20 | 789.2 | 0.9965 | 0.8184 | -5.79 | -0.15 | 0.2703 | 0.3582 |

### Economic references (not models, not baselines)

| policy | mean net return over the same windows |
| --- | --- |
| CASH (never trade) | +0.0000 |
| buy-and-hold | +0.6063 |

### Dispersion

| model | seed dispersion of mean net return | fold dispersion | min fold | max fold |
| --- | --- | --- | --- | --- |
| logistic_regression | 0.0000 | 0.2020 | -0.2924 | +0.1590 |
| lightgbm | 0.0000 | 0.1863 | -0.1891 | +0.2561 |
| xgboost | 0.0000 | 0.0864 | -0.1016 | +0.1056 |
| mtst | 0.0258 | 0.1560 | -0.2491 | +0.3876 |
| majority_baseline | 0.0000 | 0.1508 | -0.7732 | -0.4470 |
| momentum_baseline | 0.0000 | 0.0334 | -0.8439 | -0.7716 |

### The checkpoint's questions

| model | beats MTST | beats majority | beats momentum | beats CASH | beats buy-and-hold | predictive-baseline improvement | economic alpha after costs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | no (8/20 folds) | yes | yes | no | 5/20 | yes | no |
| lightgbm | yes (12/20 folds) | yes | yes | no | 0/20 | yes | no |
| xgboost | yes (12/20 folds) | yes | yes | yes | 0/20 | yes | yes |

**Read the last two columns apart.** *Predictive-baseline improvement* means the
model learned more than a trivial rule — a floor test. *Economic alpha after
costs* means acting on it earned money after fees and slippage, above CASH. A
model can pass the first and fail the second, and this repository has already
published exactly that; no model is called profitable here for clearing majority
or momentum.

### Selected thresholds

| model | thresholds chosen across folds and seeds |
| --- | --- |
| logistic_regression | 0.52, 0.56, 0.66 |
| lightgbm | 0.60, 0.62, 0.66, 0.70 |
| xgboost | 0.64, 0.68, 0.70, 0.72 |
| mtst (frozen, not retrained) | 0.36, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.56 |

### Directional attribution (exact, from persisted predictions)

| model | LONG trades | LONG hit rate | LONG sum | SHORT trades | SHORT hit rate | SHORT sum |
| --- | --- | --- | --- | --- | --- | --- |
| lightgbm | 675 | 0.5111 | +0.7220 | 355 | 0.3944 | -0.8477 |
| logistic_regression | 1245 | 0.4900 | +0.5401 | 665 | 0.4211 | -1.7638 |
| mtst | 943 | 0.5345 | +0.3819 | 445 | 0.4382 | -0.8386 |
| xgboost | 415 | 0.4699 | +0.2444 | 140 | 0.5714 | -0.0898 |

### Integrity

* 80 outer report(s) recomputed from the persisted per-sample predictions and matched.
* MTST was not retrained; the frozen v4 artifacts were read only.
* Every run shares one research contract, one research-input fingerprint, one
  sealed anchor and one fold geometry, and the two families scored the same
  outer rows with identical statistical baselines — checked before aggregation.

P2a is an adaptive follow-up designed after prior outer-validation results were observed, and it reuses those outer folds. They are research evidence, not a pristine final out-of-sample test. The sealed test block is unopened and no number here is a claim about live profitability.
