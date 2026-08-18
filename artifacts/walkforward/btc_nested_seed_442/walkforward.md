# Nested walk-forward validation

Expanding training window. Each fold has three chronological regions:
train -> inner validation -> outer validation. The scaler and the weights
are fitted on train; early stopping and the decision threshold are chosen on
inner validation; the frozen model is measured once on outer validation.
**Only the outer block is reported below.** Outer blocks do not overlap, so
no row is reported as a result twice.

**Sealed test block:** rows 48217-56726, 2025-08-27 to 2026-08-17. No fold below plans, trains
on, selects on, or scores a row at or after that boundary.

## Fold geometry

| fold | train rows | inner rows | outer rows | outer period | threshold | best epoch |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0-21697 | 21697-26518 | 26518-31339 | 2023-03-04 to 2023-09-24 | 0.42 | 2 |
| 1 | 0-26518 | 26518-31339 | 31339-36160 | 2023-09-24 to 2024-04-12 | 0.46 | 3 |
| 2 | 0-31339 | 31339-36160 | 36160-40981 | 2024-04-12 to 2024-10-30 | 0.46 | 3 |
| 3 | 0-36160 | 36160-40981 | 40981-45802 | 2024-10-30 to 2025-05-19 | 0.44 | 4 |

## Outer validation (the reported result)

| fold | outer period | model | trades | net return | Sharpe | max DD | macro F1 | dir acc | coverage | calib err |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-03-04 to 2023-09-24 | majority_baseline | 781 | -0.7544 | -5.82 | 0.7826 | 0.1633 | 0.3244 | 1.0000 | 0.1157 |
| 0 | 2023-03-04 to 2023-09-24 | momentum_baseline | 780 | -0.8199 | -7.21 | 0.8621 | 0.2432 | 0.3018 | 0.9968 | 0.2985 |
| 0 | 2023-03-04 to 2023-09-24 | mtst | 26 | -0.0134 | -0.83 | 0.0757 | 0.1778 | 0.5833 | 0.0077 | 0.1886 |
| 1 | 2023-09-24 to 2024-04-12 | majority_baseline | 0 | +0.0000 | 0.00 | 0.0000 | 0.1262 | 0.0000 | 0.0000 | 0.0029 |
| 1 | 2023-09-24 to 2024-04-12 | momentum_baseline | 792 | -0.7716 | -5.10 | 0.7827 | 0.2726 | 0.3702 | 0.9981 | 0.2301 |
| 1 | 2023-09-24 to 2024-04-12 | mtst | 34 | +0.0616 | 3.47 | 0.0988 | 0.1399 | 0.5493 | 0.0149 | 0.0908 |
| 2 | 2024-04-12 to 2024-10-30 | majority_baseline | 0 | +0.0000 | 0.00 | 0.0000 | 0.1161 | 0.0000 | 0.0000 | 0.0012 |
| 2 | 2024-04-12 to 2024-10-30 | momentum_baseline | 792 | -0.7792 | -5.55 | 0.7907 | 0.2809 | 0.3748 | 0.9983 | 0.2252 |
| 2 | 2024-04-12 to 2024-10-30 | mtst | 105 | -0.1502 | -3.03 | 0.2513 | 0.1476 | 0.4874 | 0.0419 | 0.1096 |
| 3 | 2024-10-30 to 2025-05-19 | majority_baseline | 0 | +0.0000 | 0.00 | 0.0000 | 0.1072 | 0.0000 | 0.0000 | 0.0147 |
| 3 | 2024-10-30 to 2025-05-19 | momentum_baseline | 792 | -0.8439 | -6.23 | 0.8664 | 0.2844 | 0.3859 | 0.9989 | 0.2143 |
| 3 | 2024-10-30 to 2025-05-19 | mtst | 133 | +0.0648 | 1.33 | 0.1392 | 0.1601 | 0.5436 | 0.0627 | 0.0796 |

## Across folds, outer validation only (mean ± std)

| model | net return | Sharpe | max DD | macro F1 | dir acc | coverage | calib err | trades | positive folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| majority_baseline | -0.1886 ± 0.3772 | -1.46 ± 2.91 | 0.1956 ± 0.3913 | 0.1282 ± 0.0247 | 0.0811 ± 0.1622 | 0.2500 ± 0.5000 | 0.0336 ± 0.0550 | 195.2 ± 390.5 | 0/4 |
| momentum_baseline | -0.8037 ± 0.0342 | -6.02 ± 0.92 | 0.8255 ± 0.0449 | 0.2703 ± 0.0187 | 0.3582 ± 0.0382 | 0.9980 ± 0.0009 | 0.2420 ± 0.0382 | 789.0 ± 6.0 | 0/4 |
| mtst | -0.0093 ± 0.1006 | 0.24 ± 2.80 | 0.1412 ± 0.0779 | 0.1563 ± 0.0165 | 0.5409 ± 0.0397 | 0.0318 ± 0.0253 | 0.1172 ± 0.0492 | 74.5 ± 52.7 | 2/4 |

**Verdict:** model beat both baselines in a majority of outer-validation folds (3/4 folds).

These are outer-validation numbers from model development. Nothing was
fitted on them, which is what makes them worth reading — but the folds were
run repeatedly while the method was being built, so they are research
evidence, not an out-of-sample result and not a claim of profitability. The
sealed test block remains unopened.
