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
fees, and it puts the offline "net return" figure in the same units as a
cost-aware decision instead of a gross one.

**What that figure is not.** It is a signal evaluation charged a flat
round-trip, and it differs from what a live spot bot would have earned in three
ways worth naming rather than leaving to be discovered:

- **The cost is applied additively**, as `direction × future_return − 0.002`,
  not as a compounded `(1 + r)(1 − c)² − 1`. On a 20 bps round trip and hourly
  BTC moves the difference is a few parts in ten thousand of the trade return,
  which is immaterial against the effects being measured — but it is an
  approximation, and it is applied identically to every arm, so it cannot
  favour one.
- **The cost is a constant, not a model of the book.** `fee_rate` is a flat
  taker fee and `slippage_rate` a flat allowance; neither responds to size,
  volatility or time of day, and no funding, borrow or maker rebate exists in
  it at all. A strategy whose edge is smaller than the uncertainty in that
  constant has not been shown to have one.
- **The research evaluation takes SHORT trades that this repository's live
  strategies cannot.** `realised_trades` opens a `-1` position on a SHORT
  signal and nets `−future_return − cost`. Every shipped strategy sets
  `can_short = False` and every shipped config is spot, so on the live path a
  SHORT signal produces no entry at all. The research number is therefore an
  answer about the *information*, not a projection of a deployable spot
  strategy's return; `nn.regime.direction_attribution` exists to split the two
  sides so the LONG-only half can be read on its own. Freqtrade's backtester
  remains the authority on execution.

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

### Research contracts

A **research contract** is a committed, versioned JSON document under
`nn/research_contracts/` that declares one research generation: which market it
studies, and the immutable UTC instant at which its test data is sealed. The
first one, `nn/research_contracts/btc-usdt-1h-gen1.json`, is the current BTC
cycle:

```json
{
  "contract_id": "btc-usdt-1h-gen1",
  "research_generation": 1,
  "domain": "directional-classification",
  "scope": { "exchange": "binance", "pair": "BTC/USDT", "timeframe": "1h" },
  "sealed_test_start": "2025-08-27T23:00:00+00:00"
}
```

It exists because a single module constant is a boundary you *edit*, and editing
a sealed boundary after seeing results contaminates the holdout. A contract is a
boundary you *add*: a new research generation — a new instrument, timeframe,
exchange, domain, or simply a new cycle on the same market — is a new file, and
the one already committed is never re-pointed.

**Identity is semantic, and cryptographic.** `contract_hash` is SHA-256 over the
canonicalised research-defining content: the schema name, id, generation, domain,
exchange, pair, timeframe and sealed instant, with keys sorted, no insignificant
whitespace, the anchor normalised to UTC ISO-8601 and scope values
case-normalised. So reformatting the file, reordering its keys, writing the
anchor as `Z` instead of `+00:00`, recasing `Binance`, or rewriting the
`description` all leave the identity alone; changing the pair, timeframe,
generation, domain, id or sealed instant all change it. The human-readable
`contract_id` is a name, not an identity — an id can be reused while the content
behind it moves, which is exactly what the hash catches.

**Selection, never construction.** `nn.train`, `nn.experiment` and
`nn.walkforward` each take `--research-contract`, whose accepted values are
exactly the committed contract ids. There is no `--sealed-date`, no external
contract path, no inline contract JSON and no environment variable: the worst a
command line can do is pick a different committed generation, which it must then
record in every artifact it writes.

**Scope fails closed.** `ResearchContract.require_scope` refuses a dataset whose
exchange, pair or timeframe is not the one the contract describes — and refuses a
dataset that declares no identity at all, because a dataset that cannot say what
it is cannot be shown to be in scope. A sealed instant only means something for
the market it was declared for.

**Every new artifact records the contract.** `report.json`, `metadata.json`,
`experiment_plan.json`, `experiments.json` and `walkforward.json` all carry
`contract_id` and `contract_hash` (the latter two under
`sealed_test.research_contract`), so an artifact can name the exact generation
that produced it. `experiment_plan.json`'s `plan_hash` covers the contract, so
the same grid run under a different generation is a different experiment.

`tests/test_research_contracts.py` holds all of this.

### Research-data provenance

A contract says what a generation was *allowed* to see. It cannot say what it
*saw*. Two datasets can agree on exchange, pair, timeframe, row count, first and
last timestamp, feature contract and target spec while differing in one
historical candle — and every field any artifact records would be identical. The
raw candles and the built dataset are local runtime inputs and are not committed,
so "same metadata" was the strongest claim available.

So every new artifact also records a **research-input fingerprint**
(`nn/data_fingerprint.py`), under a top-level `research_input` key:

```json
"research_input": {
  "fingerprint_schema": "chimera.research-input/1",
  "research_input_hash": "…64 hex…",
  "full_table_hash": "…64 hex…",
  "research_rows": 48217,
  "total_rows": 56726,
  "research_start": "…", "research_end": "…",
  "columns": ["date", "close", "segment_id", "…features…", "future_return", "target"]
}
```

**It is semantic, not a hash of the file.** Nothing reads Parquet bytes. The
digest covers the *values* research reads, each normalised to a fixed width and
byte order: timestamps as UTC nanoseconds since the epoch, integers as
little-endian `int64`, floats as little-endian `float64` with `-0.0` folded onto
`0.0` and every NaN folded onto one. A canonical JSON header carries the
normalised market, the ordered feature names, the feature spec and the target
spec. So recompressing the file, storing timestamps at microsecond resolution,
reordering its columns, writing a pandas index into it, reformatting the sidecar,
recasing `Binance` or moving the file all leave the identity alone; one changed
price does not.

**Which rows.** `research_input_hash` covers rows `[0, sealed_start)` — exactly
the research-visible region under the selected contract. Appending candles after
the seal is the ordinary way this dataset grows and cannot change it, so growth
never makes two runs incomparable. A correction *before* the seal does change it,
and should: every row index in an artifact then addresses a different candle.

**Which columns.** `date`, `close`, `segment_id` (when present), the feature
columns in their recorded order, `future_return` and `target` — precisely what
`nn.train.ResearchData` reads. A column research never looks at does not change
the identity; a missing `segment_id` does, because gap handling is not a detail.

**`full_table_hash` is audit metadata, and only that.** It spans every row,
answering "is this the same file?", and it is never a comparability input — a run
whose dataset has since grown past the seal is still the same research input.
Computing it feeds sealed values into a one-way digest; no label is inspected, no
metric computed, and nothing branches on the result.

**Three identities, three reasons to change.** The contract is the research
question. The resolved row is where its instant lands in one table. The
fingerprint is what was in that table. A corrected candle before the seal moves
the last two and leaves the contract alone — which is exactly the case the
separation exists to represent.

**Comparability.** `nn.wf_diagnostics` refuses to aggregate runs whose
`research_input_hash` differs, checked alongside the contract hash and before the
row and geometry checks, because matching contracts, geometry and dates say the
runs measured the same *rows*, not that those rows held the same data. Handed a
`--dataset`, it recomputes the fingerprint from the research half of that file
and refuses a mismatch rather than silently re-indexing. Handed `--raw`, it
records a `raw_input` fingerprint of the candles at or before the last
research-visible timestamp, so the market statistics can be reproduced from any
file with that identity.

**History stays history.** Artifacts that predate fingerprints carry none, are
read exactly as before, and are never given one — reported as *no research-input
fingerprint (dataset metadata only)*, which is not an integrity fault. A block
that is *present* but untrustworthy — partial, from another schema, malformed, or
claiming a research region that is not the one the same file's
`sealed_test.start_row` describes — is a fault.

`tests/test_data_provenance.py` holds all of this.

### Where the sealed test block begins

**One immutable UTC timestamp**, carried by the selected research contract:

```
research: timestamp <  contract.sealed_test_start
sealed:   timestamp >= contract.sealed_test_start
```

`nn.dataset.resolve_sealed_boundary` turns that instant into a row index for one
particular dataset: the first row whose timestamp is at or after the anchor. The
anchor candle need not exist — if it is missing because of a real market-data gap
or a dropped feature row, the first surviving row after it is the sealed start,
and no candle is invented to make the timestamp present. The resolver fails
closed on malformed timestamps, on timestamps that are not strictly increasing
(unsorted or duplicated, either of which makes the partition ambiguous), and on a
dataset that lies entirely on one side of the anchor.

**The row index is not the contract.** On the canonical BTC 1h dataset the anchor
resolves to row 48,217 of 56,726, which is what the committed walk-forward
artifacts record. That number is a property of that dataset: appending candles
cannot change it, and a legitimate repair to history *before* the anchor could
change it without moving the seal by one second. Artifacts therefore record every
half — the research contract, its `anchor_timestamp`, and the `start_row` that
anchor resolved to — so a reader can tell which generation a run was produced
under. Two runs landing on the same row are not thereby comparable:
`nn.wf_diagnostics` compares contract identities, not row indices.

The boundary used to be `int(n_rows * (train_frac + val_frac))`. That number moved
forward every time the dataset grew, so timestamps that were sealed when one run
was planned were research data by the next, and the held-out estimate shrank
silently — nothing in any artifact recorded what the row had meant in wall-clock
time. Nothing may move the anchor now: not a CLI flag, an environment variable,
a dataset length, an experiment argument, or a walk-forward fraction. The only
thing a command line can change is *which committed contract* it runs under, and
that is recorded rather than hidden.

`--train-frac` and `--val-frac` survive on `nn.train` and `nn.experiment`, where
they allocate train against validation **inside** the research region
`[0, sealed_start)` — only their ratio is used, and they cannot move
`test.start`. Under the default 0.70/0.15 the training block is 70/85 of the
research region, which on the canonical dataset is exactly the row the previous
`int(n_rows * 0.70)` produced, so the default geometry is unchanged.

`tests/test_sealed_boundary.py` and `tests/test_research_contracts.py` hold all of
this, including the core regression:
appending 1, 100, 10,000 or a year of candles after the anchor leaves the
resolved boundary, the research timestamps and the fold geometry identical.

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
profit factor, two risk-adjusted statistics, max drawdown, number of trades,
exposure, turnover.

**The two risk-adjusted statistics are named for what they are**, because one
name was doing both jobs and neither honestly:

- `per_trade_sharpe` — mean net trade return over its standard deviation *across
  trades*. A per-trade quality ratio. **Not annualised**, and not comparable to a
  published Sharpe.
- `annualised_sharpe` — computed from an actual candle-by-candle portfolio return
  series: equity is unchanged while flat, and marked to market while a position
  is open, with each cost side charged when it is paid. That curve equals
  `cumprod(1 + trade_returns)` at every completed trade boundary, so it is the
  equity curve already being reported at candle resolution rather than a second
  return model. It is annualised by `sqrt(candles_per_year)` over **elapsed
  wall-clock time**: each calendar candle interval absent from the processed
  research dataset still counts as time that passed, and carries a zero
  portfolio return. That zero is exact rather than assumed — `build_windows`
  refuses any window or label crossing a segment boundary, so no scored position
  can be held across a discontinuity. Counting dataset rows instead would treat
  a six-hour discontinuity as a single hour and inflate the result.

  An absent interval is **not** necessarily a candle the exchange failed to
  publish. `nn/data_pipeline.py` also drops per-segment warm-up rows and rows
  whose features are undefined, then re-segments what survives by timestamp — so
  a discontinuity may be a genuine source-data gap *or* a candle deliberately
  absent from the feature dataset. The arithmetic is the same either way, and
  the reports do not claim to know which it was.

Both are `None` — rendered `n/a` — when undefined, never `0.0`: a portfolio that
never moved has no Sharpe, and a zero would read as a measurement. That rule
applies to the model, to the baselines and to CASH alike.

**`max_drawdown` was corrected at the same time.** It took its running peak from
the first completed trade rather than from starting capital, so a strategy under
water from its first trade understated the fall it actually took — a single trade
losing 10% reported a drawdown of zero, and two consecutive 10% losses reported
10% instead of 19%. The peak now starts at 1.0. `candle_max_drawdown` is the same
equity curve at candle resolution, so it also sees the drawdown suffered *inside*
an open position and is always the larger of the two.

The field keeps its name, because the old number was wrong rather than a
different measurement worth preserving. But a pre-correction value is not
comparable to a corrected one, so `nn.wf_diagnostics` skips it for legacy
artifacts and reports it separately from the fields those artifacts simply never
recorded — `skipped_because_redefined` versus `skipped_because_absent`.

The version this replaced annualised per-trade statistics by
`candles_per_year / horizon`: the number of trades the strategy *could* have taken
back to back, regardless of how many it took. On a signal with ~3% coverage that
overstated the figure several-fold, and it grew as the target horizon shrank, so
shortening the horizon "improved" it without trading any more often. Artifacts
produced before the change are detected and flagged rather than compared — see
[`artifacts/README.md`](../artifacts/README.md).

`trading_metrics` is a *signal evaluation*, not a backtest. It takes
non-overlapping trades in time order, holds each for `horizon` candles, and
charges the round-trip cost as a flat subtraction from the trade's return; it
takes SHORT trades, which the long-only spot strategies here do not. Overlapping
them would book the same price move several times. See *Why costs are in the
label* above for what the cost model is and is not. Freqtrade's backtester
remains the authority on execution; this exists so model selection can optimise
a cost-aware objective instead of accuracy.

Baselines are always reported next to the model, on both validation and test.

**Baselines are not economic references, and the reports keep them apart.**
Majority-class and momentum are statistical/rule floors: beating them says the
model learned more than a trivial rule, and nothing else. Whether acting on it
made money is a different question, answered by two reference policies that make
no predictions at all:

- **CASH** — never trade. Zero return, zero costs, zero drawdown. The line every
  strategy clears before any other column means anything.
- **buy-and-hold** — buy at the close of the first scored candle, hold, sell at
  the candle that closes the last scored sample's label horizon. That is exactly
  the window the model could trade in, taken as a *continuous market span* rather
  than the scored-sample set, and both counts are reported rather than equated.
  It pays **one** round trip for the whole hold, not one per horizon — charging
  per horizon would make it a different policy.

Buy-and-hold's `annualised_sharpe` is built from the same portfolio construction
and the same cost model as the model's, so the columns are comparable. The one
exception is stated in the report: no model trade ever spans a market-data gap,
so the strategy's portfolio is provably flat across one and padding it with zeros
invents nothing — but a *hold* is exposed to a price path nobody recorded, so
across a gap its Sharpe is withheld with a reason rather than published under a
name that promises comparability. Its return stays exact and its drawdown is
flagged as a lower bound.

These live in `nn/evaluate.py`, not `nn/baselines.py`: that module's contract is
"emit a probability matrix in the model's shape", and buy-and-hold is not a
per-sample decision rule. Encoding it as one would make it re-enter and pay the
round-trip cost every `horizon` candles, turning "hold the asset" into "churn the
asset".

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
train  ->  validation  /  nested walk-forward  ->  choose a candidate
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

# nested walk-forward (train -> inner validation -> outer validation)
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
whose signatures take a training split and a validation split and nothing else,
so nothing fitted can come from anywhere but those two. Evaluation of a frozen
model goes through `score_frozen_split`, which fits nothing at all; `nn.train`
points it at the sealed test split exactly once, and `nn.walkforward` points it
at each fold's outer validation block. The guarantee is a property of those
signatures, and `tests/test_research_workflow.py` asserts it by spying on
`build_windows` and checking the test split never appears.

### The experiment runner

`nn.experiment` runs the full cartesian product of the values passed on the
command line, over seven dimensions: `seed`, `lr`, `seq_len`, `d_model`,
`n_heads`, `num_layers`, `dropout`. Omitting a flag pins that dimension to its
`nn.train` default rather than searching it. The grid is enumerated and written
out before any training starts, so what was searched is on the record.

Configurations are ranked by a stated validation objective — `net_return`
(default, net of the configured round-trip costs), `annualised_sharpe` or
`macro_f1` —
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

### Nested walk-forward validation

`nn.walkforward` asks the harder question — does the *procedure* keep working as
the market moves? The training window expands, and each fold has **three**
chronological regions:

```
fold 0: [--- train ---][ inner ][ outer ]
fold 1: [------ train ------][ inner ][ outer ]
fold 2: [--------- train --------][ inner ][ outer ]
```

| region | what happens there |
| --- | --- |
| **train** | the scaler is fitted here; the model weights are fitted here |
| **inner validation** | early stopping, decision-threshold selection, any other model-selection quantity — never reported as fold performance |
| **outer validation** | the frozen model at the frozen threshold, measured once — the only block reported as the fold's result |

The third region is the point. An earlier version had two and used the second
one twice: it chose the early-stopping epoch and the decision threshold on the
validation block and then reported that same block as the fold's performance.
Both quantities were fitted on the data they were scored on, so the reported
numbers were optimistic by construction — a selection score presented as a
result. Nothing at all is fitted on the outer block: it reaches exactly one
function, `nn.train.score_frozen_split`, which takes an already-fitted scaler,
model, threshold and baselines and only transforms, windows and measures.

**Outer blocks never overlap.** They advance by `--step`, which defaults to the
outer block size, so consecutive outer blocks are back to back and no row is
reported as the result of two folds; a step smaller than the outer block is
refused rather than allowed to double-count. A fold's *inner* block may be a
previous fold's outer block, and a later fold may train on it — by then those
rows are history, which is exactly what walk-forward is meant to simulate.

**Folds are planned over the research region only** — the rows strictly before
`nn.dataset.SEALED_TEST_START_UTC`, resolved against the dataset's own timestamps
by `ResearchData.sealed_boundary()`, the same call `nn.train` and `nn.experiment`
make. Walk-forward has no `--train-frac`/`--val-frac`: they existed only to
locate the old moving boundary, and were removed rather than left looking as
though they still control the seal.

Fold geometry is configurable as fractions *of the research region*: `--folds`,
`--min-train-frac` (or `--min-train-size`), `--inner-val-frac` (or
`--inner-val-size`), `--outer-val-frac` (or `--outer-val-size`) and `--step`.
Asking for more rows than the research region holds is an error — the number of
folds is never silently reduced, and the sealed rows are never borrowed to make
up the difference.

The defaults (45% / 10% / 10%, step = outer size) give four folds on the
56,726-row BTC 1h dataset, in which the anchor resolves to row 48,217 — so the
research region is rows `[0, 48217)`. That row is metadata about *that* dataset;
appending candles does not move it, and a repair to history before the anchor
could move it without moving the seal:

| fold | train | inner validation | outer validation |
| --- | --- | --- | --- |
| 0 | 0–21697 | 21697–26518 | 26518–31339 |
| 1 | 0–26518 | 26518–31339 | 31339–36160 |
| 2 | 0–31339 | 31339–36160 | 36160–40981 |
| 3 | 0–36160 | 36160–40981 | 40981–45802 |

Per fold, and asserted rather than intended:

- no planned row — train, inner or outer — reaches the sealed boundary at
  48,217;
- the scaler is fitted on that fold's training rows only;
- early stopping and threshold selection use that fold's **inner** rows only,
  and `select_threshold` is never called on outer rows;
- the baselines that have state are fitted on training rows only;
- no input window or label horizon crosses a region boundary, and none crosses a
  market-data gap — every region goes through the same `build_windows` segment
  check as `nn.train`;
- outer blocks are disjoint, checked on the row indices actually evaluated.

Reported per fold on the **outer** block, and aggregated as mean ± standard
deviation across folds: macro F1, directional accuracy, coverage, calibration
error, trade count, net return after costs, both risk-adjusted statistics,
exposure, max drawdown — for the model and both baselines, with the CASH and
buy-and-hold references for the same window reported in their own section.

The summary counts how many folds the model beat both baselines in, because
beating them in one fold out of four is not evidence of anything, and it
aggregates outer validation only. It also counts how many folds it beat CASH and
buy-and-hold in, and the verdict states both results in one sentence. The earlier
version emitted "model beat both baselines in a majority of outer-validation
folds" on its own — a true sentence that read as evidence of profitability while
the model lost money in every fold it won.

**These are model-development numbers.** Nothing was fitted on the outer blocks,
which is what makes them worth reading — but the folds get re-run while the
method is being built, so they are research evidence, not an out-of-sample
result and not a claim of profitability. Walk-forward is what makes it
affordable to iterate without spending the sealed estimate; the sealed test
block remains unopened.

Three earlier versions got some part of this wrong, in ways worth recording
because the later two looked fine:

1. The first scored an explicit per-fold **test** block, so every research
   iteration consumed the estimate outright.
2. Its replacement stopped naming any split "test", but still planned folds over
   the whole dataset. With the default geometry the last two validation windows
   landed inside the sealed block — on a 56,726-row dataset, 1,890 and 8,508
   rows past the boundary. The output read `test_evaluated: false` and was
   wrong: those were sealed rows labelled "validation".
3. The third kept every row below the boundary but reported the block it had
   selected the threshold and the early-stopping epoch on, and its validation
   ranges overlapped between folds — so the same rows were counted more than
   once in the across-fold mean.

Renaming a split does not unseal its rows, and calling a block "validation" does
not make it an evaluation. That is why the boundary is a row index both modules
compute the same way, and why the regression tests compare row indices instead
of split names — a name-based check is exactly what let the second bug through
and would have passed against the third.

Output: `artifacts/walkforward/walkforward.json` (which records
`sealed_test.anchor_timestamp` and `sealed_test.start_row`, the sealed period,
`reported_block: "outer_validation"`, and per fold a `periods` object with all
three regions, a `selection` object and an `outer_validation` object) and
`walkforward.md`. Both halves of the boundary are recorded because an artifact
carrying only a row cannot say which seal it was produced under.

### Diagnostics across several walk-forward runs

`nn.wf_diagnostics` reads `walkforward.json` artifacts that already exist and
reports on them. It takes paths on the command line — artifact directories or
the JSON files themselves — so nothing has to be committed, and it never opens a
dataset or a checkpoint:

```bash
python -m nn.wf_diagnostics \
    artifacts/walkforward/run_a \
    artifacts/walkforward/run_b \
    --out artifacts/walkforward/diagnostics
```

It answers two questions a single run cannot answer about itself.

**Is each artifact sound?** The invariants walk-forward asserts while running are
re-checked against what landed on disk, on row indices rather than region names:

- the three regions of every fold in order, `train.end <= inner.start` and
  `inner.end <= outer.start`;
- outer blocks disjoint within the run — no row reported by two folds;
- no region ending at or beyond `sealed_test.start_row`;
- runs compared against each other agree on `sealed_test.anchor_timestamp`, so a
  timestamp-anchored run is never averaged together with one that records only a
  row index. The five committed artifacts predate the anchor, carry none, and are
  reported as row-only rather than relabelled;
- a recorded `research_input` block is whole, from a schema this tool reads, made
  of well-formed digests, and claims the same research region as the same file's
  `sealed_test.start_row`. Absent is not a fault — the committed artifacts
  predate fingerprints — but present and untrustworthy is;
- `test_evaluated` and `sealed_test.evaluated` both false, failing closed when
  either is absent rather than absent-means-fine;
- `reported_block` and `summary.aggregated_from` both `outer_validation`;
- **the recorded across-fold summary reproduced from the per-fold reports beside
  it**, so the headline is provably the folds' average and not an edit.

A pre-nested artifact — one `validation` block per fold, selected on and
reported — is refused by name rather than read as if it were an evaluation.

**How much of the result is the seed?** Runs that differ only in `--seed` give a
distribution, and the width is the useful number. The report shows each run's
across-fold mean and the spread of those means, the same metrics fold by fold
across runs, and the stability of the *selected* threshold and early-stopping
epoch — a fold whose threshold swings between seeds is a fold whose inner block
did not have a strong preference. It also counts how many run-folds MTST beat
both baselines in, because winning a majority in one run and losing it in the
next is not a result.

Runs whose geometry, sealed boundary, dataset or fold count differ are reported
as **not comparable** and no aggregate is produced: averaging metrics from
different blocks of market is a category error, not a seed study. So are runs
whose `research_input_hash` differs — checked before the row and geometry
checks, because a shared contract, a shared shape and a shared date range say
the runs measured the same *rows*, not that those rows held the same data. The same
applies when any run fails its audit — the problems are printed and the headline
is withheld. The command exits non-zero in both cases, so it can gate a
pipeline.

Output: the Markdown report on stdout, plus `wf_diagnostics.json` and
`wf_diagnostics.md` when `--out` is given.

**What it is not.** Seed spread measures the stability of the research
procedure. It is not an out-of-sample estimate and not a claim of
profitability — the outer blocks are still research blocks, re-run while the
method was being built. The sealed test block stays unopened, and the tool
refuses an artifact that says otherwise.

### Regime diagnostics: what differed about the market

`nn.wf_diagnostics` can say fold 2 earned more than fold 1. With `--dataset` it
can say *what was different about the market* in those two stretches.

```bash
python -m nn.wf_diagnostics \
    artifacts/walkforward/run_a artifacts/walkforward/run_b \
    --dataset data/datasets/binance_BTC_USDT_1h.parquet \
    --raw data/raw/binance/BTC_USDT_1h.parquet \
    --out artifacts/diagnostics/btc_regimes_v1
```

Both data flags are optional. Without `--dataset` the audit, seed stability and
per-fold model stability still run, so artifacts remain analysable with no
dataset at hand. `--raw` requires `--dataset`.

**Per outer block, from the processed dataset:** `future_return` mean, median,
std, mean absolute and sign fractions; `ema_cross` mean/median/std and sign
fractions; `ema_fast_ratio` and `ema_slow_ratio` mean and median; `atr_norm` and
`realized_vol` mean/median/p90; `hl_range`, `volume_change` mean and median;
`volume_z` mean, median and p90 of the absolute value; exact SHORT/HOLD/LONG
counts and fractions with mean and median `future_return` per class. Columns are
addressed by the feature names the artifact recorded, never by position.

**Computed over the rows the model was scored on, not the rows the block
spans.** An outer block of 4,821 rows does not yield 4,821 samples: the first
`seq_len - 1` cannot open a window, the last `horizon` cannot close a label, and
any candidate straddling a market-data gap is dropped. On the BTC artifacts that
is 4,683 scored rows in fold 0 against 4,821 block rows. Summarising the block
would describe a slightly different stretch of market than the metric being
explained, so rows are selected through `nn.dataset.sample_indices` — the same
function `build_windows` uses — with the artifact's own `config.seq_len` and
target horizon. Both counts appear in the report.

The reconstruction is **checked, not asserted**: each fold's artifact records how
many outer samples the run scored, and a reconstruction that does not reproduce
that number exactly stops the report. An artifact with no recorded `seq_len`
cannot be regime-analysed at all — guessing one would summarise rows the model
never saw.

**Per outer block, from raw OHLCV:** start and end close, market return, mean /
median / std candle return, annualised realised volatility, mean absolute candle
return, positive and negative candle fractions, and max drawdown. These are taken
over the **whole block span** — a view of the market independent of which rows
were scorable — and labelled as such, so they are not read as row-for-row
comparable with the model-facing statistics above. The report records a
`raw_input` fingerprint of the candles at or before the last research-visible
timestamp, so these numbers can be reproduced from any file with that identity
rather than only from the local path they were read through.

Four rules make those numbers trustworthy rather than merely available:

- **The sealed boundary is enforced by slicing.** `load_research_frame`
  truncates the dataset at `sealed_test_start` on load, so a sealed row is not
  in the frame at all and no off-by-one can reach one. Block ranges are checked
  against the boundary again on the way in.
- **Row indices only mean a candle against the dataset they were recorded on.**
  Every identity field the artifact and `DatasetMetadata` both carry is checked —
  rows, exchange, pair, timeframe, first and last timestamp, feature names,
  feature spec, target spec — and the metadata's span is cross-checked against
  the frame's own first and last timestamps, because a sidecar can be stale. A
  wrong file with a coincidentally matching row count is refused rather than
  silently reindexed.
- **Matching metadata is not matching data.** When the artifact records a
  research-input fingerprint, it is recomputed from the truncated frame and a
  mismatch is refused — the case where a dataset passes every field check above
  and still holds a different candle. Only the research half is verifiable here,
  because the frame is cut at the seal before the check can run. A run that
  recorded no fingerprint is still analysable; it simply cannot be verified, and
  that is reported rather than glossed.
- **Raw candles join on timestamps, never on position.** The processed dataset
  dropped warm-up and label rows, so processed row *i* is not raw row *i*. A
  missing timestamp, a duplicate, or a timeframe that disagrees is an error.
  Candle-to-candle returns are taken only between timestamps exactly one
  timeframe apart, and the count of skipped pairs is reported.

**Best vs worst.** The folds with the highest and lowest mean MTST outer net
return across seeds are chosen from the data — never named in the source — and
compared row by row, ranked by difference relative to the larger magnitude.
Every row is a coincidence, not a cause: one fold per regime is a sample of one.

**LONG / SHORT attribution.** Aggregate reports cannot answer "did the longs or
the shorts lose the money?", and it is not approximated from them. Runs write
`outer_predictions.parquet` beside the artifact — one row per outer sample with
`fold`, `seed`, `row_index`, `timestamp`, `true_target`, `future_return`,
`p_short`/`p_hold`/`p_long`, `selected_action` and `threshold`, outer rows only,
with a sealed row refused before the file is created. When present, the
diagnostics report exact per-direction trade counts, hit rates, mean/median net
return, `additive_trade_return_sum`, HOLD count and per-side coverage. The trades
come from `nn.evaluate.realised_trades`, the same generator `trading_metrics`
aggregates, so the two sides partition exactly the trades the fold reports and
there is no second cost model to drift. When absent, the report says so and
offers nothing in its place.

`additive_trade_return_sum` is named for what it is: the arithmetic total of one
side's per-trade returns. The reported `net_return` **compounds**
(`prod(1 + r) - 1`), so the two sides' sums do not add up to it and this is not a
decomposition of the fold's return — it compares the sides against each other and
nothing more.

**Candidate hypotheses.** Three research questions, generated from the
diagnostics and ranked by the size of the difference behind them. None is
implemented or tuned — a diagnostic that quietly starts fitting is how a
research tool becomes a source of overfitting.

Their wording follows the observed numbers rather than being fixed in the
source. Whether the best and worst folds separate is a property of the data: if
their per-seed ranges overlap, the report says the two cannot be told apart yet;
if they do not, it says the worst fold behaves consistently worse across every
seed observed — and in both cases it records the ranges, the seed count and the
positive-seed count that chose the wording, so the sentence can be checked
against the numbers. Neither branch generalises: one fold per regime is a sample
of one, and seed count is not a substitute for independent periods.

**Baselines are scored at their own declared threshold.** Every model used to be
scored at the threshold selected for MTST, which made the "floor" move with the
seed: `MajorityClassBaseline` emitted the empirical class prior, so a threshold
above the prior's largest entry silently suppressed it to HOLD. Both baselines
now emit one-hot actions — the vector *is* the decision, so no threshold can
change it — and are scored at `nn.baselines.BASELINE_DECISION_THRESHOLD`, a
declared constant fitted on nothing. On identical geometry and data their reports
are byte-identical across seeds, which is asserted. A run set that shows
deterministic-baseline spread predates this fix, and the report says so rather
than presenting it as a finding.

Output: `regime_diagnostics.json` and `regime_diagnostics.md` under `--out`,
with sections for the integrity audit, dataset/sealed status, fold geometry,
market regime statistics, seed stability, per-fold model stability, best vs
worst, attribution, hypotheses and limitations.

### P2a: does the model family matter, on identical information?

`nn.benchmark` answers one diagnostic question — **does model complexity add
predictive or economic value when simple models receive exactly the same
information as MTST?** — and `nn.benchmark_compare` scores the answer against the
frozen MTST evidence without retraining it:

```bash
python -m nn.benchmark --dataset DATASET --research-contract btc-usdt-1h-gen1 \
    --folds 4 --seq-len 64 --seed 42 --out artifacts/benchmark/btc_p2a_seed_42

python -m nn.benchmark_compare \
    --benchmark artifacts/benchmark/btc_p2a_seed_{42,142,242,342,442} \
    --mtst artifacts/walkforward/btc_nested_v4_seed_{42,142,242,342,442} \
    --dataset DATASET --out artifacts/benchmark/btc_p2a_comparison
```

Three untuned, predeclared models — logistic regression, LightGBM, XGBoost —
against MTST, the majority and momentum baselines, CASH and buy-and-hold. One
thing varies: the model family.

**The samples are MTST's samples, not a re-derivation of them.** The benchmark
plans folds with `nn.walkforward.plan_nested_folds`, builds train and inner
windows with `nn.train.prepare_research_windows`, and scores the outer block
through `nn.train.score_frozen_split` — the same three calls the MTST
walk-forward makes. The one addition is `nn.simple_models.flatten_windows`, which
turns the `(64, 14)` sequence into a `896`-wide row where column `t * 14 + f` is
feature `f` at timestep `t`, oldest first. A tabular model given only the latest
candle would be answering a different question, so it is not given one.

Everything else is held: the same train-only scaler, the same eligible rows, the
same gap handling, the same labels, horizon and cost model, the same threshold
grid and cost-aware selection objective on the inner block only, the same sealed
boundary. There is no hyperparameter search of any kind, no early stopping, and
no flag on the command line that can move a model's configuration.

`nn.benchmark_compare` **fails closed**. It refuses to aggregate runs that
disagree on the research contract, the contract hash, the research-input
fingerprint, the sealed anchor or its resolved row, the fold count or geometry,
the feature contract, the target/horizon/cost semantics or the sequence length —
and, beyond the metadata, it requires that both families scored the *same outer
rows* and that their shared statistical baselines produced *identical* numbers.
Those baselines are fitted on the fold's training labels and on nothing, so on
the same samples they are a constant: a discrepancy in them is a discrepancy in
the samples, which no agreement between metadata blocks could reveal.

The report keeps two verdicts apart and never collapses them. *Predictive-baseline
improvement* is "did it learn more than a trivial rule" — a floor test.
*Economic alpha after costs* is "did acting on it earn money after fees and
slippage, above CASH". A model can pass the first and fail the second, and this
repository has published exactly that.

P2a is an adaptive follow-up designed after prior outer-validation results were
observed, and it reuses those outer folds. They are research evidence, not a
pristine out-of-sample test, and no number it produces is a claim about live
profitability. The sealed test block is not planned over, fitted on, selected on
or scored.

The three libraries are their own optional dependency group — `pip install -e
".[benchmark]"` — because nothing in the inference container or any live path
imports them. `.[all]`, which CI installs, includes it.

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
