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

## Walk-forward

`nn.walkforward` repeats the whole procedure on rolling windows:

```
fold 0: [--- train ---][ val ][ test ]
fold 1:        [--- train ---][ val ][ test ]
fold 2:               [--- train ---][ val ][ test ]
```

Nothing carries between folds — each refits its scaler, retrains, and re-selects
its threshold. The summary reports how many folds the model beat both baselines
in, because beating them in one fold out of four is not evidence of anything.

Output: `artifacts/walkforward/walkforward.json` and `walkforward.md`.
