# Nested walk-forward validation

Expanding training window. Each fold has three chronological regions:
train -> inner validation -> outer validation. The scaler and the weights
are fitted on train; early stopping and the decision threshold are chosen on
inner validation; the frozen model is measured once on outer validation.
**Only the outer block is reported below.** Outer blocks do not overlap, so
no row is reported as a result twice.

**Research contract:** `btc-usdt-1h-gen1` (generation 1, directional-classification, binance BTC/USDT 1h),
semantic identity `sha256:dca6d0a891a257197a7c8aecec04fdba0a0b3009cfe93c5f3a397d458ab4a1de`.

**Research input:** `sha256:1c09218442cdf98cfe2f49ac521c70b2d7692717d7488e8105c972a3d2ac4740` over 48217 research-visible rows,
2020-01-04 to 2025-08-27. The contract says what this generation was allowed
to see; this says what it saw. Two runs are repeated measurements of one procedure only if both match.

**Sealed test block:** everything at or after `2025-08-27T23:00:00+00:00` — an immutable wall-clock
anchor, which in this dataset is rows 48217-56790, 2025-08-27 to 2026-08-20. The row
range is metadata about this dataset; the contract is the boundary, and
appending later candles cannot move it. No fold below plans, trains on,
selects on, or scores a row at or after that boundary.

## Fold geometry

| fold | train rows | inner rows | outer rows | outer period | threshold | best epoch |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0-21697 | 21697-26518 | 26518-31339 | 2023-03-04 to 2023-09-24 | 0.46 | 2 |
| 1 | 0-26518 | 26518-31339 | 31339-36160 | 2023-09-24 to 2024-04-12 | 0.50 | 8 |
| 2 | 0-31339 | 31339-36160 | 36160-40981 | 2024-04-12 to 2024-10-30 | 0.46 | 4 |
| 3 | 0-36160 | 36160-40981 | 40981-45802 | 2024-10-30 to 2025-05-19 | 0.48 | 3 |

## Outer validation (the reported result)

### Statistical / rule baselines and the model

What these answer: **did the model learn more than a trivial rule?** They are a
floor, not a business case. Beating them says nothing about whether money was
made — that is the next section.

`ann. Sharpe` is candle-level portfolio returns (equity unchanged while flat, marked to market while a position is open, both cost sides charged when they are paid), annualised by sqrt(candles_per_year) over elapsed wall-clock time, with a zero return for each calendar interval absent from the processed dataset. `n/a` means undefined, not zero: a
portfolio that never moved has no Sharpe. Annualising magnifies whatever
happened in a short block, so compare folds against each other and against
the references below — a single fold's figure is not an expected annual result.

| fold | outer period | model | trades | net return | ann. Sharpe | per-trade Sharpe | exposure | max DD | macro F1 | dir acc | coverage | calib err |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 to 2023-09-24 | majority_baseline | 781 | -0.7562 | -5.89 | -0.16 | 0.9861 | 0.7842 | 0.1633 | 0.3244 | 1.0000 | 0.6756 |
| 0 | 2023-03-04 to 2023-09-24 | momentum_baseline | 781 | -0.7828 | -6.39 | -0.17 | 0.9861 | 0.8337 | 0.2432 | 0.3018 | 0.9968 | 0.6985 |
| 0 | 2023-03-04 to 2023-09-24 | mtst | 7 | +0.0465 | 1.23 | 0.48 | 0.0088 | 0.0253 | 0.1730 | 0.7857 | 0.0030 | 0.0975 |
| 1 | 2023-09-24 to 2024-04-12 | majority_baseline | 792 | -0.4470 | -2.00 | -0.05 | 1.0000 | 0.4879 | 0.1995 | 0.4270 | 1.0000 | 0.5730 |
| 1 | 2023-09-24 to 2024-04-12 | momentum_baseline | 792 | -0.7716 | -5.35 | -0.13 | 1.0000 | 0.7827 | 0.2726 | 0.3702 | 0.9981 | 0.6301 |
| 1 | 2023-09-24 to 2024-04-12 | mtst | 68 | +0.3876 | 2.90 | 0.28 | 0.0859 | 0.0520 | 0.1587 | 0.5528 | 0.0339 | 0.0967 |
| 2 | 2024-04-12 to 2024-10-30 | majority_baseline | 792 | -0.7732 | -5.24 | -0.14 | 1.0000 | 0.7862 | 0.1937 | 0.4095 | 1.0000 | 0.5905 |
| 2 | 2024-04-12 to 2024-10-30 | momentum_baseline | 792 | -0.7792 | -5.35 | -0.15 | 1.0000 | 0.7907 | 0.2809 | 0.3748 | 0.9983 | 0.6252 |
| 2 | 2024-04-12 to 2024-10-30 | mtst | 117 | -0.2061 | -1.36 | -0.11 | 0.1477 | 0.2978 | 0.1595 | 0.4881 | 0.0530 | 0.1058 |
| 3 | 2024-10-30 to 2025-05-19 | majority_baseline | 792 | -0.6921 | -3.78 | -0.10 | 1.0000 | 0.7464 | 0.1989 | 0.4253 | 1.0000 | 0.5747 |
| 3 | 2024-10-30 to 2025-05-19 | momentum_baseline | 792 | -0.8439 | -6.09 | -0.16 | 1.0000 | 0.8664 | 0.2844 | 0.3859 | 0.9989 | 0.6143 |
| 3 | 2024-10-30 to 2025-05-19 | mtst | 43 | -0.1555 | -1.46 | -0.20 | 0.0543 | 0.1712 | 0.1194 | 0.4795 | 0.0154 | 0.0392 |

### Economic references (not models, not baselines)

What these answer: **did the strategy make money, did it beat doing nothing,
and did it beat simply owning the asset?** CASH never trades and returns
exactly zero. Buy-and-hold buys at the first scored candle and sells at the
candle closing the last scored sample's horizon — the same window the model
traded in, as a *continuous market span* rather than the scored-sample set —
and pays **one** round trip for the whole hold, not one per horizon.

Their `ann. Sharpe` is built the same way as the models' above, from the same
cost model, so the columns are comparable.

| fold | policy | trades | gross return | net return | ann. Sharpe | candle max DD | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | CASH | 0 | +0.0000 | +0.0000 | n/a | 0.0000 | no position, no fees |
| 0 | buy-and-hold | 1 | +0.1845 | +0.1825 | n/a | 0.2085 | spans 85 missing candle(s); Sharpe withheld, drawdown is a lower bound |
| 1 | CASH | 0 | +0.0000 | +0.0000 | n/a | 0.0000 | no position, no fees |
| 1 | buy-and-hold | 1 | +1.6614 | +1.6594 | 3.96 | 0.2019 | 4757 candles held |
| 2 | CASH | 0 | +0.0000 | +0.0000 | n/a | 0.0000 | no position, no fees |
| 2 | buy-and-hold | 1 | +0.1056 | +0.1036 | 0.61 | 0.3057 | 4757 candles held |
| 3 | CASH | 0 | +0.0000 | +0.0000 | n/a | 0.0000 | no position, no fees |
| 3 | buy-and-hold | 1 | +0.4818 | +0.4798 | 1.61 | 0.3096 | 4757 candles held |

### Across folds, outer validation only (mean ± std)

| model | net return | ann. Sharpe | per-trade Sharpe | exposure | max DD | macro F1 | dir acc | coverage | calib err | trades | positive folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| majority_baseline | -0.6671 ± 0.1508 | -4.23 ± 1.73 | -0.11 ± 0.05 | 0.9965 ± 0.0069 | 0.7012 ± 0.1434 | 0.1888 ± 0.0172 | 0.3966 ± 0.0487 | 1.0000 ± 0.0000 | 0.6035 ± 0.0487 | 789.2 ± 5.5 | 0/4 |
| momentum_baseline | -0.7944 ± 0.0334 | -5.79 ± 0.53 | -0.15 ± 0.02 | 0.9965 ± 0.0069 | 0.8184 ± 0.0391 | 0.2703 ± 0.0187 | 0.3582 ± 0.0382 | 0.9980 ± 0.0009 | 0.6420 ± 0.0382 | 789.2 ± 5.5 | 0/4 |
| mtst | 0.0181 ± 0.2694 | 0.33 ± 2.12 | 0.11 ± 0.32 | 0.0742 ± 0.0583 | 0.1366 ± 0.1248 | 0.1527 ± 0.0231 | 0.5765 ± 0.1432 | 0.0263 ± 0.0219 | 0.0848 ± 0.0307 | 58.8 ± 46.2 | 2/4 |

**Verdict:** Model beat both statistical/rule baselines in 4/4 outer folds, and made money on outer validation (mean net return +0.0181), beating CASH in 2/4 folds; buy-and-hold returned +0.6063 on the same windows and was beaten in 0/4 folds. Beating majority-class and momentum is a floor test, not evidence of profitability.

These are outer-validation numbers from model development. Nothing was
fitted on them, which is what makes them worth reading — but the folds were
run repeatedly while the method was being built, so they are research
evidence, not an out-of-sample result and not a claim of profitability. The
sealed test block remains unopened.
