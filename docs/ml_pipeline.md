# ML pipeline

Every modelling decision in ProjectChimera, and the reasoning behind it.

## The target

**The model predicts a cost-aware direction over a fixed horizon.**

For candle `t`:

```
future_return(t) = close[t + horizon] / close[t] - 1

LONG   if future_return >  cost_threshold
SHORT  if future_return < -cost_threshold
HOLD   otherwise
```

`cost_threshold = 2 × (fee_rate + slippage_rate)` — a round trip, since capturing
a move costs an entry and an exit, and each pays a fee and crosses some spread.

Defaults: `horizon = 6` candles, `fee_rate = 0.0005`, `slippage_rate = 0.0005`,
giving `cost_threshold = 0.002` (20 bps). All three are CLI flags on
`tools/build_features.py` and are recorded in the dataset and model metadata, so a
model can never be trained under one definition and served under another.

### Why not predict the price

The previous design regressed the absolute next close price and then entered on
`prediction > 0.6`. For BTC that condition reads `60000 > 0.6` — always true. The
entry rule was a constant.

Two separate failures were involved, and both are fixed by construction here:

1. **The units did not match.** A price was compared with a probability. Now the
   model emits class probabilities and the strategy compares them against a
   probability threshold, so both sides of the comparison mean the same thing.
2. **The target was non-stationary and unbounded.** A model fitted on 2020 price
   levels has no basis for 2024 levels. Every feature and the target are now
   scale-free.

### Why costs are in the label

A 0.05% move is not an opportunity if capturing it costs 0.2%. Labelling such
moves HOLD stops the model from learning to chase noise that loses money after
fees, and it means the offline "net return" figure is comparable with what the
strategy would actually have earned.

The side effect is that HOLD becomes the majority class. That is handled with
inverse-frequency class weights in the loss (`nn/train.py::class_weights`);
without them the cheapest way to reduce the loss is to always predict HOLD, and
the model converges to the majority baseline.

## Features

Fourteen features, all causal and all scale-free, defined in
`chimera/features.py` and emitted in a fixed order:

| Feature | Definition |
| --- | --- |
| `ret_1` | `close.pct_change()` |
| `log_ret_1` | `log(close / close.shift(1))` |
| `ret_close_open` | `close / open - 1` |
| `hl_range` | `(high - low) / close` |
| `ema_fast_ratio` | `close / EMA(12) - 1` |
| `ema_slow_ratio` | `close / EMA(26) - 1` |
| `ema_cross` | `EMA(12) / EMA(26) - 1` |
| `rsi_centered` | `RSI(14) / 100 - 0.5` |
| `macd_norm` | `MACD / close` |
| `macd_hist_norm` | `(MACD - signal) / close` |
| `atr_norm` | `ATR(14) / close` |
| `realized_vol` | rolling std of `log_ret_1` over 24 |
| `volume_change` | `volume.pct_change()` |
| `volume_z` | rolling z-score of volume over 24 |

Implemented in plain pandas rather than `ta` or `talib`, so the strategy
container and the training container run byte-identical code with no shared
native dependency.

Two properties are asserted in `tests/test_features.py`:

- **Causality** — computing features on a prefix of the series gives identical
  values for the overlapping rows. Any centred window, backfill or forward shift
  would break this. It is the whole no-look-ahead claim, tested directly.
- **Determinism** — the same input always produces the same output, in the same
  column order.

Warm-up rows (three time constants for the recursive indicators) are dropped by
`build_dataset` rather than filled, because filling a warm-up NaN with zero is
indistinguishable from a real zero-return candle.

## Splits and leakage

Splits are contiguous blocks of rows in time order. There is no shuffling
anywhere in the codebase.

A sample at row `i` uses feature rows `[i - seq_len + 1, i]` and a label derived
from the price at row `i + horizon`. For a split `[start, end)` a sample is only
emitted when:

```
start + seq_len - 1  ≤  i  ≤  end - 1 - horizon
```

The left bound keeps the input window inside the block. The right bound keeps the
*label* inside it — this is the subtle direction, because a training label near
the end of the training block would otherwise be computed from a price that falls
in validation. The embargo comes out of this index arithmetic rather than a fudge
factor, which is why it can be tested exactly
(`tests/test_dataset.py::test_no_sample_index_is_shared_between_splits`).

The scaler is fitted on training rows only and applied unchanged to validation
and test.

### What the test split is for

Exactly one thing: a single estimate of unseen performance, produced after every
fitted quantity is frozen. It is not used for hyperparameter tuning, threshold
selection, feature selection, early stopping, or the promotion decision. Those
all use validation.

The estimate is only worth having *once*. Every time a decision is made after
looking at a test number — a different learning rate, a wider model, one more
epoch — that number stops being an estimate of unseen performance and becomes a
second validation score, with none of the honesty and all of the confidence.
The workflow below exists to make that failure mode hard to fall into by
accident. See [The research workflow](#the-research-workflow).

## Baselines

A Transformer that does not beat "always predict the majority class" has learned
the class prior, not the market. `nn/baselines.py` provides:

- **`MajorityClassBaseline`** — the training set's class prior. Its accuracy *is*
  the class prior, which is precisely why raw accuracy is not a headline metric
  here: with HOLD dominant, a model can score 70% accuracy while never making a
  tradeable prediction.
- **`MomentumBaseline`** — a rule: follow the sign of `ema_cross` at the end of
  the window.

Both emit probability matrices in the same shape and class order as the model, so
all three are scored by identical code.

## Metrics

**Classification:** class distribution, predicted distribution, per-class
precision/recall/F1, macro F1, confusion matrix, expected calibration error,
directional accuracy (over called trades only — counting HOLDs would inflate it),
and coverage.

Calibration matters here more than usual, because the trading threshold *is* a
probability cut: an overconfident model trades far more than its stated threshold
implies.

**Trading:** net return, gross return, total costs, average trade, win rate,
profit factor, Sharpe, max drawdown, number of trades, exposure, turnover.

`trading_metrics` is a *signal evaluation*, not a backtest. It takes
non-overlapping trades in time order, holds each for `horizon` candles, and
charges the round-trip cost. Overlapping them would book the same price move
several times. Freqtrade's backtester remains the authority on execution; this
exists so model selection can optimise a cost-aware objective instead of accuracy.

Baselines are always reported next to the model, on both validation and test.

## Threshold selection

The decision threshold is chosen on **validation**, by maximising net return
after costs over a grid from 0.34 to 0.90, subject to producing at least 10
trades. A threshold that fires three times and gets lucky is not a threshold.

The chosen value is stored in the model metadata and returned by `/readyz` and
every `/predict` response, so the strategy uses the threshold that was actually
selected rather than a hard-coded constant.

`nn/evaluate.py::signals_from_proba` and `chimera/contracts.py::decide` implement
the same rule, and `tests/test_evaluate.py` asserts they agree on random inputs.
If they diverged, the strategy would trade one policy while the reports measured
another.

## The model

`MTST` — a small Transformer encoder over the feature window, classifying from
the final step.

Defaults: `d_model=64`, 2 layers, 4 heads, dropout 0.1 — roughly 10^5 parameters.
The previous 8-layer/128-dim/8-head stack had order 10^6 parameters, which for a
few thousand hourly candles is several parameters per training sample. Capacity
should be justified by data volume and by beating the baselines, not chosen for
impressiveness. All dimensions are CLI flags.

## Reproducibility

`nn/train.py::set_seed` seeds Python, NumPy and torch, enables cuDNN determinism
and sets `CUBLAS_WORKSPACE_CONFIG`. `tests/test_train_smoke.py` asserts that two
runs with the same seed produce identical model outputs and the same selected
threshold.

CPU is the default and fully supported. Mixed precision is enabled only when CUDA
is genuinely present (`torch.amp.autocast("cuda", enabled=use_amp)`), so a CPU run
never hits a CUDA-specific path. Ray Tune is optional: `--tune-trials` defaults to
0, which trains once.

## Versioning and promotion

```
artifacts/models/
    20260816T120000Z-a1b2c3/
        model.pt        # state_dict
        config.json     # MTSTConfig
        metadata.json   # features, scaler, threshold, target spec, dataset period
        report.json     # every metric, both splits, all three models
    current.json        # {"version": "..."} — what inference serves
```

`metadata.json` carries everything inference needs: feature order, sequence
length, scaler mean and std, decision threshold, target spec, feature spec, model
version, training date, dataset period, and the pair/exchange/timeframe it was
trained on. `load_model` refuses an artifact whose metadata and weights disagree.

**Finishing a training run does not make a model live.** `promote()` is the only
function that writes `current.json`, and `nn/train.py` calls it only when
`--promote` is passed *and* `check_gates` passes on the **validation** report:

| Gate | Default |
| --- | --- |
| Macro F1 exceeds the best baseline's by a margin | 0.01 |
| Minimum validation trades | 10 |
| Net return after costs is positive | required |
| Calibration error below | 0.25 |

The previous code called `mlflow.register_model` twice per run and set the `prod`
alias unconditionally on completion — two versions per run, both auto-promoted,
with no criterion at all.

### Promotion fails closed

Beyond the gates, `promote()` reads the artifact's own `report.json` and
requires positive evidence that the sealed test split was actually spent:

| Condition | Result |
| --- | --- |
| `research_only: false` **and** `test_evaluated: true` | may be promoted |
| `research_only: true` (a `--validation-only` run) | refused |
| `test_evaluated: false` | refused |
| either field missing, null, or not a boolean | refused |
| `report.json` missing or unparseable | refused |

Absence of a warning is not evidence. An artifact whose provenance cannot be
read is one whose out-of-sample performance is unknown, and "unknown" must not
resolve to "serve it". Research artifacts are still written and kept — they are
useful for inspection — they simply cannot become `current.json`.

## Backtesting a saved model is guarded against in-sample evaluation

`NNPredictorStrategy` can load one finished artifact and run it over any
historical range. Features are causal, so that is computationally sound — but it
is not automatically *statistically* out-of-sample. Backtesting a model trained
through 2026-01 over 2025-01 to 2026-06 scores it largely on data it was fitted
on, and the result looks excellent while meaning nothing.

Artifacts therefore record temporal provenance:

| Field | Meaning |
| --- | --- |
| `train_end` | last candle the **weights** saw |
| `validation_end` | last candle the early-stopping epoch and threshold saw |
| `training_cutoff` | `validation_end or train_end` — the real in-sample boundary |

The cutoff is `validation_end`, not `train_end`: early stopping and the decision
threshold were both fitted on validation, so data in between is in-sample too.

Before an offline backtest emits a single signal, the strategy checks that its
first *evaluated* candle is strictly after the cutoff, and raises
`InSampleBacktestError` otherwise:

```
ML backtest overlaps the model training period for BTC/USDT. Model
20260816T120000Z-a1b2c3 was fitted on data through 2025-06-01, but this
backtest starts evaluating at 2025-03-01. Use data after 2025-06-01 or run
nn.walkforward, which retrains per fold.
```

It raises rather than quietly holding over the overlapping stretch: a partially
in-sample backtest that still prints a summary is more dangerous than a failed
run, because the summary reads as a result. An artifact with no temporal
metadata is refused too — "cannot prove it is out-of-sample" is not "safe".

The check is on the first evaluated candle, not the first row: Freqtrade
prepends `startup_candle_count` rows to warm indicators up, and those are inputs
the strategy would equally have had live, not scored predictions.

**Walk-forward remains the recommended research method** — it retrains per fold,
so the question never arises.

## The research workflow

**Development, repeated as often as you like:**

```
train  ->  validation  /  walk-forward validation  ->  choose a candidate
```

**Once, and only after the research decisions are frozen:**

```
one sealed test evaluation
```

**Then:**

```
Freqtrade backtest  ->  dry-run
```

**Never:** tune repeatedly against the same test set. Each pass burns the only
out-of-sample estimate the project has, and the damage is invisible — the
numbers keep looking fine while meaning progressively less.

### During development: leave test sealed

```bash
# one configuration
python -m nn.train --dataset DATASET --validation-only --epochs 30

# a predeclared grid
python -m nn.experiment --dataset DATASET --seed 1 2 3 --lr 1e-4 3e-4 1e-3 --epochs 20

# expanding-window walk-forward
python -m nn.walkforward --dataset DATASET --folds 4 --epochs 20
```

`--validation-only` is research mode. It trains, early-stops, selects the
decision threshold and reports on validation, and then stops. The test split is
not scored, not printed, and — the part that matters — never windowed at all:
`nn/train.py` contains exactly one block that builds test windows, and research
mode skips it. The run says so on stdout, records `"test_evaluated": false` and
`"research_only": true` in `report.json`, and the artifact it writes can never
be promoted. `--promote` together with `--validation-only` is refused by the CLI,
and `registry.promote()` refuses a `research_only` artifact even when called
directly, so reaching past the CLI does not help.

`nn.experiment` and `nn.walkforward` have no research mode to forget to switch
on: neither one has any code path that windows test. All three entrypoints share
one core in `nn/train.py` — `prepare_research_windows` and `fit_and_validate` —
whose signatures take a training split and a validation split and nothing else.
The guarantee is a property of those signatures, and
`tests/test_research_workflow.py` asserts it by spying on `build_windows` and
checking the test split never appears.

### The experiment runner

`nn.experiment` runs the full cartesian product of the values passed on the
command line, over seven dimensions: `seed`, `lr`, `seq_len`, `d_model`,
`n_heads`, `num_layers`, `dropout`. Omitting a flag pins that dimension to its
`nn.train` default rather than searching it. The grid is enumerated and written
out before any training starts, so what was searched is on the record.

Configurations are ranked by a stated validation objective — `net_return`
(default, net of the configured round-trip costs), `sharpe` or `macro_f1` —
with macro F1 as the tie-break. Both baselines are refitted for every run and
reported next to it, because a ranking of models against each other says nothing
about whether any of them beat a rule.

A configuration that raises is recorded with its error and appears in both
output files and the console summary. It is never silently dropped: a grid that
quietly shrinks is how a search comes to be reported over settings it never
actually tried.

The grid is written to `experiment_plan.json` **before the first model trains**,
and that file is not rewritten afterwards — a plan that only appears once the
results are in is a description of what finished, not a predeclaration. The
manifest carries a `plan_hash` over the grid, the fixed parameters and the
dataset, which the results file repeats, so results can be tied back to the plan
they came from. Running a different grid into a directory that already holds a
plan is refused rather than allowed to overwrite the record.

Output: `artifacts/experiments/experiment_plan.json` (written first),
`experiments.json` and `experiments.csv`.

### Walk-forward validation

`nn.walkforward` asks the harder question — does the *procedure* keep working as
the market moves? The training window expands, and each fold is validated on the
block immediately after it:

```
fold 0: [--- train ---][ val ]
fold 1: [------ train ------][ val ]
fold 2: [--------- train --------][ val ]
```

**Folds are planned over the research region only** — rows `[0,
sealed_test_start)`, where the boundary comes from
`nn.dataset.sealed_test_start` under the same 70/15/15 contract `nn.train` uses.
`--train-frac` and `--val-frac` locate that boundary and do nothing else.

Fold geometry is configurable as fractions *of the research region*: `--folds`,
`--min-train-frac` (or `--min-train-size`), `--fold-val-frac` (or
`--fold-val-size`) and `--step`, which defaults to spreading the folds evenly
over the research region after the first training window. Asking for more rows
than the research region holds is an error — the sealed rows are never borrowed
to make up the difference.

Per fold, and asserted rather than intended:

- no planned row, training or validation, reaches the sealed boundary;
- the scaler is fitted on that fold's training rows only;
- early stopping and threshold selection use that fold's validation rows only;
- no input window or label horizon crosses a fold boundary, and none crosses a
  market-data gap — folds go through the same `build_windows` segment check as
  `nn.train`;
- validation begins at or after the row where training ends, so no fold can be
  influenced by rows a later fold validates on.

Reported per fold and aggregated as mean ± standard deviation across folds:
macro F1, directional accuracy, coverage, trade count, net return after costs,
Sharpe, max drawdown — for the model and both baselines. The summary also counts
how many folds the model beat both baselines in, because beating them in one
fold out of four is not evidence of anything.

**These are validation numbers.** Walk-forward is what makes it affordable to
iterate without spending the sealed estimate; it is not itself an out-of-sample
result.

Two earlier versions got this wrong, in ways worth recording because the second
looked fine:

1. The first scored an explicit per-fold **test** block, so every research
   iteration consumed the estimate outright.
2. Its replacement stopped naming any split "test", but still planned folds over
   the whole dataset. With the default geometry the last two validation windows
   landed inside the sealed block — on a 56,726-row dataset, 1,890 and 8,508
   rows past the boundary. The output read `test_evaluated: false` and was
   wrong: those were sealed rows labelled "validation".

Renaming a split does not unseal its rows. That is why the boundary is a row
index both modules compute the same way, and why the regression tests compare
row indices instead of split names — a name-based check is exactly what let the
second bug through.

Output: `artifacts/walkforward/walkforward.json` (which records
`sealed_test.start_row` and the sealed period) and `walkforward.md`.

### Spending the test split

When the research decisions are frozen — architecture, hyperparameters, feature
set, horizon — run training once without `--validation-only`:

```bash
python -m nn.train --dataset DATASET --epochs 30 --seq-len 64
```

That run scores test once, after the weights, scaler, threshold and
early-stopping epoch are all frozen, and it is the only run whose artifact can
be promoted. If the result disappoints, the honest options are to accept it or
to collect more data — not to adjust and re-run. A number produced by the second
attempt is not the same kind of number as one produced by the first.
