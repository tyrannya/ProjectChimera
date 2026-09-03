# ProjectChimera

A research and **dry-run** platform for machine-learning crypto trading, built on
[Freqtrade](https://www.freqtrade.io/) as the execution engine.

It exists to make one chain reproducible and safe to run end to end:

```
market data → validated dataset → features → leakage-safe training
   → versioned model → inference service → Freqtrade strategy
   → central risk controls → dry-run trading → metrics and alerts
```

> **Historical backtest performance does not guarantee future profitability.**
>
> Nothing here is a claim that this system makes money, and nothing in it has
> been tuned to make a backtest look good. The included synthetic data is for
> exercising the pipeline, not for measuring it. Trading cryptocurrency risks
> the loss of your capital.

**LIVE TRADING: DISABLED BY DEFAULT.** See [Live trading protection](#live-trading-protection).

> **Where the project is going, as of 2026-09-03.** The owner has adopted the
> Fable 5.1 replacement roadmap:
> [`docs/proposed_development_plan_post_fable_5_1_audit.md`](docs/proposed_development_plan_post_fable_5_1_audit.md),
> expanded in
> [`docs/proposed_demo_implementation_master_plan.md`](docs/proposed_demo_implementation_master_plan.md).
> The next deciding evidence is a preregistered six-month **prospective**
> dry-run campaign on data that has not yet occurred, on the Binance USD-M
> BTCUSDT perpetual with spot BTCUSDT as the hedge leg — not another reading of
> the four historical outer blocks, which are now closed to deciding use. The
> active stages are S0 → S6; S0 is done and the next task is the offline
> prospective recorder. `docs/current_development_plan.md` is **superseded as a
> roadmap** and kept for its closed-checkpoint findings and standing
> constraints. Nothing in the adopted plan authorises real money.

---

## What works today

| Component | Status | Notes |
| --- | --- | --- |
| Data download and validation | Working | UTC, deduplicated, gap-detected, OHLC-checked |
| Feature engineering | Working | 14 causal, scale-free features; shared by training and strategy |
| Cost-aware labelling | Working | SHORT/HOLD/LONG against a fee + slippage threshold |
| Chronological splits, nested walk-forward | Working | Leakage prevented by index arithmetic, asserted in tests |
| Training (`nn.train`) | Working | CPU-first, reproducible, baselines reported alongside |
| Model artifacts and gated promotion | Working | On-disk artifact is the source of truth |
| Inference service (`nn.infer_service`) | Working | FastAPI, schema-validated, `/livez` + `/readyz` |
| `NNPredictorStrategy` | Working | Fails closed to HOLD on any inference problem |
| `SwingSpot` | Working | Simple EMA/RSI, long-only spot |
| Central risk engine + kill switch | Working | On the entry path via `confirm_trade_entry` |
| Futures execution (`chimera.futures`) | Working, **dry-run only** | USD-M perpetuals, isolated 1x, LONG and SHORT. No live-order path exists and no credential is required |
| Prometheus + Grafana | Working | Every panel queries a metric this code exports |
| Trading-mode controller (`chimera.modes`) | Working, **selects nothing today** | SCALPING / DAY_TRADING / SWING / FLAT as states. Every mode is `NOT_ELIGIBLE` under the committed evidence, so it decides `FLAT` on every bar |
| Paper chain (`tools.paper_run`) | Working, **dry-run replay only** | Runs specialists → consensus → mode → Aegis → Hermes into the dry-run venue. A smoke, not sustained paper validation, and its report says so in fields |
| Telegram notifications | Working, optional | Absent credentials disable it silently |
| MLflow tracking | Optional | `--mlflow`; artifacts do not depend on it |
| Ray Tune | Optional | `--tune-trials N`; default 0 runs a single pass |

### Experimental / disabled

| Component | Why |
| --- | --- |
| `ScalpFutures` | Needs an order-book feed this repository does not have, and depth is not available historically. Emits no entries. |
| `ArbMM` | Freqtrade cannot execute a two-leg spread trade, and the second leg's data is absent. Emits no entries. |

Both are kept, unarmed, with the reasoning in their module docstrings. They are
not silently broken strategies pretending to work — that was the previous state,
and [`docs/engineering-audit.md`](docs/engineering-audit.md) records it.

---

## Install

Requires Python 3.11+. Docker is optional but recommended for the full stack.

```bash
git clone https://github.com/tyrannya/projectchimera.git
cd projectchimera

python -m venv .venv && source .venv/bin/activate
make setup          # installs .[all] and the pre-commit hooks

cp .env.example .env   # optional: only needed for real data or notifications
```

`pyproject.toml` is the single source of truth for dependencies. The extras map
onto the containers:

| Extra | For | Contains |
| --- | --- | --- |
| `.[trade]` | Freqtrade container | freqtrade |
| `.[ml]` | Inference and training | torch, fastapi, uvicorn, ccxt |
| `.[tracking]` | Optional | mlflow |
| `.[tune]` | Optional | ray[tune] |
| `.[dev]` | Development | pytest, black, flake8, mypy, pre-commit |
| `.[all]` | Everything | all of the above |

`requirements*.txt` are thin pointers at these extras so Docker can cache the
install layer; they never diverge from `pyproject.toml`.

## Run the tests

```bash
make test           # pytest
make lint           # pre-commit over every file
make check          # compileall + pytest + pre-commit + docker compose config
```

## Try it without a network

```bash
make smoke
```

This walks synthetic candles → features → a one-epoch model → artifact → the
inference service → a strategy decision → a risk decision, in under a minute on
CPU. It proves the plumbing, not profitability.

---

## The pipeline, step by step

### 1. Download data

```bash
make backfill EXCHANGE=binance PAIR=BTC/USDT TIMEFRAME=1h START=2023-01-01
```

Writes `data/raw/binance/BTC_USDT_1h.parquet` plus a `.meta.json` sidecar with the
validation report (duplicates removed, gaps found, missing candles counted). Gaps
are **reported, never filled** — forward-filling a gap invents market data that a
backtest would then trade on.

No exchange credentials are needed for public candle data.

To work offline instead:

```bash
make sample         # synthetic candles in data/raw/synthetic/
```

### 2. Build features and labels

```bash
make features EXCHANGE=binance PAIR=BTC/USDT TIMEFRAME=1h
```

Produces `data/datasets/binance_BTC_USDT_1h.parquet` with 14 features, a
`future_return` column and a `target` column, plus metadata recording exactly
which specs produced them.

### 3. Train

```bash
make train DATASET=data/datasets/binance_BTC_USDT_1h.parquet EPOCHS=30
```

or directly, for a short run:

```bash
python -m nn.train --dataset data/datasets/binance_BTC_USDT_1h.parquet \
    --epochs 2 --tune-trials 0
```

Prints validation and test tables with both baselines beside the model, and
writes `artifacts/models/<version>/` containing `model.pt`, `config.json`,
`metadata.json` and `report.json`.

Training **does not** promote a model. Pass `--promote` to make it live, and it
will still only be promoted if it clears the gates in `nn/registry.py`.

The command above scores the held-out test split, which is worth doing exactly
once. While you are still choosing a model, use research mode instead:

```bash
python -m nn.train --dataset DATASET --validation-only --epochs 30
```

It reports on validation and leaves the test split sealed. See
[the research workflow](docs/ml_pipeline.md#the-research-workflow).

The sealed block begins at one immutable UTC instant, and that instant belongs to
a **research contract**: a committed, versioned document under
`nn/research_contracts/` that says which market a research generation studies and
where its data is sealed. The first one is `btc-usdt-1h-gen1` — Binance BTC/USDT
1h, generation 1, sealed at `2025-08-27T23:00:00+00:00`.

Everything before the seal is research; everything at or after it is sealed.
Appending candles cannot move it, and no CLI flag, environment variable or dataset
length may either: `--research-contract` *selects* one of the committed contracts
and cannot define a new boundary, and there is no flag anywhere that takes a date.
A new research generation is a new contract file, never an edit to an existing
one — editing one changes its SHA-256 semantic identity, which every artifact
records and the diagnostics refuse to aggregate across. The row the instant
resolves to (48,217 on the canonical BTC dataset) is metadata about that dataset,
recorded beside the contract in every artifact, and is not the contract. See
[research contracts](docs/ml_pipeline.md#research-contracts) and
[where the sealed test block begins](docs/ml_pipeline.md#where-the-sealed-test-block-begins).

A contract says what a generation was *allowed* to see. It cannot say what it
*saw*: two datasets can agree on scope, row count, period, feature contract and
target spec and still differ in a historical candle. So every new artifact also
records a **research-input fingerprint** — a SHA-256 over the values research
reads, in the rows before the seal. It is semantic, not a hash of the Parquet
file: recompressing, restoring at a different resolution, reordering columns or
moving the file leaves it alone, while one changed price does not. Appending
candles after the seal cannot change it either, so the ordinary way this dataset
grows never makes two runs incomparable; a correction *before* the seal does
change it, and should. `nn.wf_diagnostics` refuses to average runs that read
different data even when their contract, geometry and dates all match, and
verifies the fingerprint against any dataset it is handed. See
[research-data provenance](docs/ml_pipeline.md#research-data-provenance).

### 4. Research: experiment grids and walk-forward validation

```bash
make experiment DATASET=data/datasets/binance_BTC_USDT_1h.parquet
make walkforward DATASET=data/datasets/binance_BTC_USDT_1h.parquet
```

`nn.experiment` runs a predeclared grid over seed, learning rate, sequence
length and model size, ranks the configurations by a stated validation
objective, and writes `artifacts/experiments/{experiments.json,experiments.csv}`.

`nn.walkforward` retrains from scratch on each expanding fold. Every fold has
three chronological regions — train, inner validation, outer validation. Early
stopping and the decision threshold are chosen on the inner block; the frozen
model is measured once on the outer block, and only outer results are reported
and aggregated as mean +/- std to
`artifacts/walkforward/{walkforward.json,walkforward.md}`. Outer blocks do not
overlap, so no row is reported as the result of two folds.

Neither one scores the test split. Only the plain `nn.train` run in step 3 does,
and only that run's artifact can be promoted.

Once several walk-forward runs exist — the same geometry with different
`--seed` values — `nn.wf_diagnostics` audits and compares them:

```bash
python -m nn.wf_diagnostics artifacts/walkforward/run_a artifacts/walkforward/run_b
```

It re-checks each artifact's leakage invariants on its own row indices, refuses
to aggregate runs that do not measure the same blocks, and reports how far the
outer-validation numbers move when only the seed changes.

Pass a dataset and it also explains what differed about the *market* between the
best and worst fold:

```bash
python -m nn.wf_diagnostics artifacts/walkforward/run_a artifacts/walkforward/run_b \
    --dataset data/datasets/binance_BTC_USDT_1h.parquet \
    --raw data/raw/binance/BTC_USDT_1h.parquet \
    --out artifacts/diagnostics/btc_regimes_v1
```

Both data flags are optional and both read local paths given at runtime; nothing
needs to be committed. The dataset is truncated at the sealed boundary on load,
raw candles join on exact timestamps rather than position, and the best and
worst folds are chosen from the data. Differences are reported as coincidences,
never causes.

Reports separate two questions that are easy to run together. **Statistical /
rule baselines** — majority-class and momentum — answer whether the model learned
more than a trivial rule. **Economic references** — CASH (never trade) and
buy-and-hold over the same window — answer whether it made money. A model can
beat both baselines in every fold and still lose to CASH, and the verdict says so
in one sentence rather than leaving the reader to notice.

Which artifact generation is authoritative is recorded in
[`artifacts/README.md`](artifacts/README.md), not inferred from directory names.
The current BTC baseline is `artifacts/diagnostics/btc_regimes_v4/`, and its
evidence is negative.

### 4a-note. Which checkpoints have evidence

Generated from the artifact tree, not written by hand: a checkpoint that
gains committed evidence makes this table — and the CI check behind it —
fail until every front-door document is reconciled with it.

**Two ceilings apply to everything below, and they are part of the record.**
First, the sealed **Styx** region (`2025-08-27T23:00:00+00:00`) and the
research-visible cutoff (`2025-05-19T08:00:00+00:00`) are 2025 market dates, but
this programme and its sealing machinery were authored in **August 2026**. The
seal genuinely protects against later in-repository adaptivity and accidental
reads; it does **not** make Styx prospective blind evidence, so any future Styx
result carries a **hindsight-era ceiling** and is described as a one-shot
historical evaluation. Second, the four outer blocks are **adaptive** and have
now been read by every checkpoint from `P2a` onward; no result on them is
confirmatory, and since 2026-09-03 they are closed to **deciding** use
altogether. See [`docs/current_development_plan.md`](docs/current_development_plan.md)
for the full post-audit disclosures.

**Two of the states below mean a checkpoint ended without ever producing a
number, and neither is a result.** `withdrawn` (`P8`) means the question became
moot and the checkpoint was never opened. `declined` (`P14`) means the design
was reviewed and refused before opening — `P14` was preregistered on the branch
`claude/p14-native-tradeflow-prereg`, no statistic of any kind was ever
computed, and PR #67 was closed without merging with the branch retained as
historical design evidence. Neither checkpoint is answered, negative, positive,
failed or inconclusive; there is no result for those words to describe, and CI
rejects a front-door document that says otherwise.

<!-- research-state:begin -->
<!--
  Generated by nn.research_state from the artifact tree; do not edit by hand.
  Regenerate with: python -m tools.verify_research_state --write
-->

| checkpoint | research question | state |
| --- | --- | --- |
| `v4` | `btc_ohlcv14_mtst_baseline` | **answered** |
| `P2a` | `btc_p2a_model_family_benchmark` | **answered** |
| `P2b` | `btc_p2b_information_set_benchmark` | **answered** |
| `P2c` | `btc_p2c_information_set_benchmark` | **answered** |
| `P3` | `btc_p3_information_set_benchmark` | **answered** |
| `P4` | `btc_p4_derivatives_positioning_benchmark` | **answered** |
| `P5` | `btc_p5_information_set_benchmark` | **answered** |
| `P6` | `btc_p6_multiclock_specialist_screen` | **answered** |
| `P6-EXT` | `btc_p6ext_swing_clock_specialist_screen` | **answered** |
| `P7` | `btc_p7_cross_timeframe_consensus` | **answered** |
| `P8` | `btc_p8_automatic_trading_mode_router` | **withdrawn** |
| `P13` | `btc_p13_structural_carry_feasibility` | **preregistered** |
| `P14` | `btc_p14_native_tradeflow_screen` | **declined** |

<!-- research-state:end -->

### 4b. Research: information sets (checkpoints P2b, P2c, P3, P4 and P5)

P2a asked whether the *model family* changes what can be extracted from the
frozen OHLCV14 feature set, and found that it barely does. That makes the next
question an information question: is there measurable, causal structure the
fourteen columns do not carry — in the price series, in a different source, or
on a different clock?

```bash
make p2b-btc && make p2b-compare      # market structure
make p2c-btc && make p2c-compare      # classical chart structure
make p3-btc  && make p3-compare       # trade-level microstructure
```

Three checkpoints, three information sets each, against the same control.
**P2b**: `ohlcv14` (14 columns), `smc_v1` (39 causal market-structure columns,
[`docs/smc_v1.md`](docs/smc_v1.md)) and `ohlcv14_plus_smc_v1` (53).
**P2c**: `ohlcv14`, `chart_structure_v1` (30 causal classical-pattern columns,
[`docs/chart_structure_v1.md`](docs/chart_structure_v1.md)) and
`ohlcv14_plus_chart_structure_v1` (44).
**P3**: `ohlcv14`, `microstructure_v1` (32 causal trade-flow columns,
[`docs/microstructure_v1.md`](docs/microstructure_v1.md)) and
`ohlcv14_plus_microstructure_v1` (46) — the first checkpoint whose *source* is
not the hourly candle, computed from Binance's public spot `aggTrades` archive
and committed as an hourly aggregate under `data/research/`. Each set goes to
the same three untuned models over the same four nested temporal folds.
Everything except the feature columns is held at the values P2a ran under: the
horizon, the costs, the fold geometry, the inner-only threshold rule, the model
configurations and the sealed boundary.

**All three answered no.** Across three models and two non-control arms, no
comparison in any of the three checkpoints improved on the OHLCV14 control in
more than two of four temporal folds, against a bar of three fixed before any
number was read. The evidence is in
[`artifacts/benchmark/btc_p2b_comparison/`](artifacts/benchmark/btc_p2b_comparison/),
[`btc_p2c_comparison/`](artifacts/benchmark/btc_p2c_comparison/) and
[`btc_p3_comparison/`](artifacts/benchmark/btc_p3_comparison/). What that does
and does not license is [`docs/research_roadmap.md`](docs/research_roadmap.md)'s
subject; the short version is that it is evidence against another hand-designed
transformation of the same hourly bars, not a proof about a space.

The fourth checkpoint, **P4**, has completed Stage 1 under its preregistered design. It tested whether causal derivatives positioning/carry information — realised funding, open interest and perpetual basis — added usable information beyond OHLCV14 in the unchanged BTC 1h/6h cost-aware setup.

The deciding XGBoost combined-vs-control comparison had three availability-qualified valid folds. One of three improved; mean net-return delta was `-0.038821333333333326` and the worst-fold delta was `-0.09306799999999998`. P4 therefore failed the preregistered continuation rule and ended as `screened_out`. This is negative evidence for this feature design and horizon, not a claim that derivatives information is generally useless.

There was no Stage 2 or re-fit. P4-HOLD was never opened, scored or evaluated and is now retired unread. Styx remains sealed. The nine primary Stage-1 cells are frozen under `artifacts/btc_p4_stage1_SHA256SUMS.txt`; the deciding screen is frozen under `artifacts/btc_p4_screen_SHA256SUMS.txt`. See [`docs/p4_preregistration.md`](docs/p4_preregistration.md) for the preregistered rules and [`docs/derivatives_v1.md`](docs/derivatives_v1.md) for the information set.

The fifth checkpoint, **P5**, is also **negative**. It changed exactly one axis — the *clock*. Every family before it was computed on the 1h bar; `mtf_v1` is `chimera.features.compute_features`, the same fourteen columns with the same window lengths, evaluated over fully closed 4h and 1d bars and aligned to each 1h row by the last bar that had closed. No new source: the bars are cut from the OHLCV history the control already reads.

The deciding `xgboost` comparison improved **1 of 4** temporal outer folds against a preregistered bar of 3 — fold deltas `+0.11508`, `-0.075359`, `-0.039844`, `-0.183647`, mean `-0.0459425`, worst `-0.183647`, the last two reported and decisive in neither direction. The availability gate passed with all four folds available, measured before any fit. This is evidence against *this* representation of higher-timeframe context at *this* horizon on *this* asset; it is not a proof that timeframe context is uninformative. The nine primary cells are frozen under `artifacts/btc_p5_SHA256SUMS.txt` and the decision record under `artifacts/btc_p5_decision_SHA256SUMS.txt`. See [`docs/p5_preregistration.md`](docs/p5_preregistration.md) and [`docs/mtf_v1.md`](docs/mtf_v1.md).

Five families have now failed on this design — three transformations of the 1h bar, one new source, one new clock — and the roadmap's conclusion is that the next research move changes axis rather than adding a sixth. That move is §4c: the clock itself, tested natively rather than supplied as context.


`--checkpoint` is a required input rather than something inferred from the arms,
because `ohlcv14` is the control of all five and cannot say which question a
cell answers. Every artifact records the checkpoint and the question it belongs to,
and `nn.p2b_compare` refuses to join cells that disagree about either.

Unlike the steps above, these run from the **committed research snapshot** under
`data/research/`, so a fresh clone reproduces them with no VPS, no private data
and no access to the sealed block. The runner verifies that snapshot itself —
all 23 checks, before a single model is fitted — so a snapshot whose manifest
has stopped describing its files produces a named rejection and zero fits,
whether it was launched through `make` or directly.

The comparison that matters is one model, one fold, one sample set, three column
sets — so the three arms have to be scored on *identical rows*. That is a
property of how the views are built (they share the spine's dates, labels,
returns and segment ids by object identity, and differ only in `features`), and
it is proved rather than asserted: the alignment evidence recomputes the sample
index per fold, and `nn.p2b_compare` refuses to aggregate cells whose baselines,
economic references or per-fold sample-index hashes disagree. It also rebuilds
every reported trading and classification number from the persisted predictions
before reporting anything.

The statistical unit is **four temporal periods**. There is no seed dimension:
these estimators are deterministic given their inputs, and logistic regression
takes no seed at all, so P2a's five seeds produced five identical copies of its
logistic evidence rather than five observations.

Two post-hoc analyses sit beside the canonical result and say so in their own
output — `make p2b-ablation MODEL=xgboost` (what each market-structure family
contributed given the others) and `make p2b-regimes` (what the four outer
periods actually were). Neither selects anything, and no regime filter is fitted
from either.

See [`docs/p2b_methodology.md`](docs/p2b_methodology.md) for the full design and
[`docs/research_reproduction.md`](docs/research_reproduction.md) for the exact
command sequence from a fresh clone.

### 4c. Research: the multi-clock architecture (checkpoints P6, P6-EXT and P7)

The five checkpoints above all varied the *columns* attached to one 1h bar. This
one varies the **clock**, which is the architecture the system was designed
around and which nothing before it had tested: independent native-timeframe
specialists, and a cross-timeframe consensus over them.

```bash
make verify-multiclock-snapshot       # the source, offline, before anything is fitted
make p6-btc && make p6-decide         # five specialists, one verdict per clock
make p7-btc && make p7-decide         # consensus in two trading modes
```

The data foundation is one canonical source — Binance's published spot
`BTCUSDT` 1m archive, every monthly object held to the SHA-256 Binance publishes
beside it — from which the 5m, 15m, 30m, 1h, 4h and 1d bars are cut on the fixed
UTC grid. A bar exists only if **every** constituent minute closed; incomplete
bars are dropped rather than forward-completed, and no bar may draw a minute from
after its own close. The source stops before `2025-05-19T08:00:00+00:00`, the
first instant of the retired `P4-HOLD` region, so no multi-clock row reaches it.
Design, derivation rules, gap table and the 1h parity investigation are in
[`docs/multiclock_v1.md`](docs/multiclock_v1.md).

**P6 is negative on all five clocks.** Same fourteen features, same cost model,
same four real-world periods, same three untuned families as P2a — only the bar
changes, and the label is six of *that clock's own* bars. Every clock cleared
exactly two of the four folds a gate fixed at three required, and `30m` and `1h`
additionally had a negative mean:

| clock | horizon | positive folds | mean outer net return | beats native momentum |
| --- | --- | ---: | ---: | ---: |
| `1m` | 6 minutes | 2 / 4 | `+0.030087` | 4 / 4 |
| `5m` | 30 minutes | 2 / 4 | `+0.0114415` | 4 / 4 |
| `15m` | 90 minutes | 2 / 4 | `+0.00926375` | 4 / 4 |
| `30m` | 3 hours | 2 / 4 | `-0.0091055` | 4 / 4 |
| `1h` | 6 hours | 2 / 4 | `-0.0267705` | 4 / 4 |

The momentum column is a floor, not a compliment. `MomentumBaseline` takes a
position on every bar it is allowed to, so at six native bars and a 20 bps round
trip it pays the cost model roughly a hundred times per fold and returns about
`-1.0`. "Beats native momentum 4 / 4" means "did not trade itself to death", and
both deciding documents say so; the gate's first condition is what bound.

Three clocks have a positive mean while improving two of four folds — the
count-the-folds rule, predeclared in P2b, firing for the third time. Two
secondary families would have passed on the fast clocks; the design named
XGBoost the deciding family before any fit existed, and the secondary results are
reported in full and decide nothing. **P6-EXT** applied the same design to the
two slow clocks a `SWING` mode needs and found neither `4h` nor `1d` viable —
both, unlike every fast clock, also lost to their own native momentum baseline in
half the folds.

**P7 is negative in both measured modes.** It fits nothing: it replays the frozen
P6 XGBoost cells' committed per-sample predictions, aligns each specialist to the
decision clock by the last bar that had closed, and requires a strict majority. A
specialist with no closed bar yet is unavailable, and unavailable is not a
`HOLD` — the whole consensus holds. Scalping (1m decision clock) improved on the
fold-wise best of its own constituents in 1 of 4 folds with mean delta
`-0.0265515`; day trading (5m) in 1 of 4 with mean delta `-0.034336`. Both
validity gates passed: each mode's own decision-clock specialist reproduced its
frozen P6 cell exactly, over the 1,161,875 and 232,285 decision rows of the four
folds. Read the trade counts before the deltas: the day-trading consensus took
6, 1, 6 and **0** trades across its four folds, so that mode's `-0.034336` rests
on thirteen realised trades. The preregistration's own low-trade flag, and where
it fires, are in the closure of
[`docs/p7_preregistration.md`](docs/p7_preregistration.md).

Each checkpoint's design was committed and pushed before its first fit, and each
document carries the SHA-256 of the machine-readable design it was run under:
[`docs/p6_preregistration.md`](docs/p6_preregistration.md),
[`docs/p6_extension_preregistration.md`](docs/p6_extension_preregistration.md),
[`docs/p7_preregistration.md`](docs/p7_preregistration.md). The closures are in
those same documents, below the design they close.

**Two limits the independent audit put on the record.** P6's fifteen primary
cells record `dirty: true` and a digest of an uncommitted tree, so while their
committed predictions and gate arithmetic replay exactly — the verdict is the
historical verdict — the *fit itself* cannot be reconstructed from a clean
checkout. P6-EXT and P7 pin their sources properly and do not inherit that. And
P6's negative is **deciding-family-specific**: logistic regression would have
cleared the screen on `1m`, `5m`, `15m` and LightGBM on `1m`, `5m`, but XGBoost
was named deciding before any fit existed and the design refused to shop for the
winner. Those cells are leads, not evidence, and re-declaring one of them the
winner on these same four burned blocks would not be a fresh result.

**What this licenses operationally: nothing new.** No clock is viable, so no
trading mode is eligible, so the mode controller
([`docs/trading_modes_v1.md`](docs/trading_modes_v1.md)) reports `FLAT` for
`specialist_not_viable` on every bar, and the committed paper smoke in
`artifacts/paper_smoke/` places zero orders for that reason. `make paper-smoke`
runs the whole chain into the dry-run venue; it is a smoke, and its report denies
being sustained paper validation, live, real money, or evidence about alpha in
fields rather than only in prose. **P8**, the automatic mode router, is
preregistered at [`docs/p8_preregistration.md`](docs/p8_preregistration.md) and
is **not opened**: its precondition is two eligible modes, and there are none.

### 5. Serve the model

```bash
make infer          # uvicorn on 127.0.0.1:3000
```

```bash
curl -s localhost:3000/livez
curl -s localhost:3000/readyz | jq
```

`POST /predict` takes raw (unscaled) features; the service applies the scaler
stored with the model:

```bash
curl -s -X POST localhost:3000/predict \
  -H 'Content-Type: application/json' \
  -d '{"pair":"BTC/USDT","timeframe":"1h","timestamp":"2026-08-16T12:00:00Z",
       "features":[[...], ...]}' | jq
```

```json
{
  "model_version": "20260816T120000Z-a1b2c3",
  "signal": "LONG",
  "probabilities": {"SHORT": 0.08, "HOLD": 0.21, "LONG": 0.71},
  "confidence": 0.71,
  "decision_threshold": 0.55,
  "served_at": "2026-08-16T12:00:01.234567Z"
}
```

Malformed bodies get `422`, a feature matrix of the wrong shape gets `400` with
the expected shape and feature order in the message, and an inference failure
gets `500` — never a fabricated score.

### 6. Dry-run trading

```bash
make dry-run EXCHANGE=binance STRATEGY=NNPredictorStrategy
```

This merges `conf/base.json` with `conf/binance.test.json`, resolves `${VAR}`
placeholders from the environment, runs the safety gate, and starts Freqtrade.
No orders reach the exchange.

### 7. Metrics

```bash
make docker-up
```

- Grafana — <http://localhost:3001> (admin/admin, change it)
- Prometheus — <http://localhost:9090>
- Inference — <http://localhost:3000/metrics>

Two dashboards are provisioned: **Chimera / Trading** (equity, PnL, drawdown,
exposure, open positions, rejected entries, kill-switch state) and **Chimera / ML
and System** (inference latency, errors, prediction and confidence
distributions, served model version, data staleness). A test asserts that every
panel and alert rule queries a metric this code actually exports.

---

## Live trading protection

ProjectChimera is dry-run only unless **two independent things** are true:

1. `ENABLE_LIVE_TRADING` is set to exactly `I_UNDERSTAND_THE_RISK`; **and**
2. the launcher is asked for live mode (`--mode live`, which selects
   `conf/<exchange>.live.json`).

Neither alone is enough. In particular:

- **Having exchange API keys does not enable live trading.**
- Asking for `--mode live` without the environment variable aborts the launch
  with exit code 2, before any config is written.
- Setting the environment variable without asking for live mode still runs dry-run.
- A config with no `dry_run` key at all is treated as dry-run — it fails closed.

**No committed config file is independently live-capable.** Every file in
`conf/` — including the `.live.json` profiles — keeps `dry_run: true`. The live
profiles mark themselves with `"chimera_live_intent": true`, and
`tools/run_bot.py` is the only thing that ever sets `dry_run: false`, writing it
to a private generated config after the gate passes. Cloning this repository and
pointing Freqtrade straight at any committed config cannot place a real order,
so the safety system does not depend on the user entering through the launcher.
A test asserts this over every file in `conf/`.

The gate lives in `chimera/safety.py` and is enforced by `tools/run_bot.py`
before Freqtrade starts. `tests/test_safety.py` and `tests/test_config_and_cli.py`
cover every path. CI never sets the variable and has no live-capable job.

**Two different claims live here, and collapsing them would be wrong.** The
section above is about the **legacy Freqtrade spot pathway**, which *is*
live-capable in principle and is deliberately double-gated. The
**`chimera.futures` chain** is a separate statement: it has **no authenticated
live order route at all** — no credentialed endpoint, no signing path, nothing
that reaches a venue — and `tests/test_futures_no_live_path.py` asserts it stays
unreachable. So "the futures chain cannot trade live" is true, and "no
live-capable code exists in this repository" is **not**. Nothing in the current
research evidence authorises enabling either path.

If you do enable live trading, you are choosing to risk real money, and the
risk limits in `conf/base.json` are defaults you should review rather than trust.

---

## Documentation

| Document | What it covers |
| --- | --- |
| [docs/architecture.md](docs/architecture.md) | Component boundaries and data flow |
| [docs/ml_pipeline.md](docs/ml_pipeline.md) | Features, the target, splits, metrics, promotion |
| [docs/risk_manager.md](docs/risk_manager.md) | Limits, sizing arithmetic, the kill switch |
| [docs/dry_run.md](docs/dry_run.md) | Running dry-run, and what is verified vs. not |
| [docs/engineering-audit.md](docs/engineering-audit.md) | What was broken before this rebuild |
| [docs/smc_v1.md](docs/smc_v1.md) | The causal market-structure information set: 39 exact definitions |
| [docs/chart_structure_v1.md](docs/chart_structure_v1.md) | The causal classical-pattern information set: 30 exact definitions |
| [docs/microstructure_v1.md](docs/microstructure_v1.md) | The causal trade-flow information set: 32 exact definitions (checkpoint P3) |
| [docs/p2b_methodology.md](docs/p2b_methodology.md) | Checkpoints P2b, P2c and P3: does any of those families add information beyond OHLCV14? |
| [docs/p4_preregistration.md](docs/p4_preregistration.md) | Checkpoint P4, preregistered before its data existed and closed after Stage 1 screened out: derivatives positioning and carry |
| [docs/derivatives_v1.md](docs/derivatives_v1.md) | P4's information set, as implemented: what §5 left open, and two consequences that tighten its gate |
| [docs/p5_preregistration.md](docs/p5_preregistration.md) | Checkpoint P5, preregistered before any P5 model was fitted and closed as negative: strictly causal higher-timeframe OHLCV context |
| [docs/mtf_v1.md](docs/mtf_v1.md) | P5's information set, as implemented: the OHLCV14 engine on a 4h and a daily clock, and why its join needed a different witness |
| [docs/multiclock_v1.md](docs/multiclock_v1.md) | The causal multi-clock source: one 1m archive, seven clocks, and what makes a bar exist |
| [docs/p6_preregistration.md](docs/p6_preregistration.md) | Checkpoint P6, preregistered before its first fit and closed as negative: independent native-timeframe specialists on five clocks |
| [docs/p6_extension_preregistration.md](docs/p6_extension_preregistration.md) | Checkpoint P6-EXT, P6's design on the 4h and 1d clocks a SWING mode needs; closed as negative |
| [docs/p7_preregistration.md](docs/p7_preregistration.md) | Checkpoint P7, preregistered after P6 closed and closed as negative: cross-timeframe consensus over the frozen specialists |
| [docs/trading_modes_v1.md](docs/trading_modes_v1.md) | SCALPING, DAY_TRADING, SWING and FLAT as operating states, their eligibility rule, and what may never select a mode |
| [docs/p8_preregistration.md](docs/p8_preregistration.md) | Checkpoint P8's committed design. P8 was **never opened** and is **withdrawn as moot**: no router exists and no P8 number exists |
| [docs/paper_operation_runbook.md](docs/paper_operation_runbook.md) | Running the dry-run paper chain, and the difference between a smoke and sustained paper validation |
| [docs/proposed_development_plan_post_fable_5_1_audit.md](docs/proposed_development_plan_post_fable_5_1_audit.md) | **The adopted roadmap (2026-09-03).** Stages S0 to S6, the prospective campaign, the finite research budget. Start here for direction |
| [docs/proposed_demo_implementation_master_plan.md](docs/proposed_demo_implementation_master_plan.md) | The adopted engineering plan: target architecture, the recorder, the demo runner, the 16-PR sequence, the test plan |
| [docs/fable_5_1_full_project_strategic_audit.md](docs/fable_5_1_full_project_strategic_audit.md) | The independent audit that produced them. A historical record, not edited after the fact |
| [docs/current_development_plan.md](docs/current_development_plan.md) | **Superseded as a roadmap.** Retained for the closed checkpoints' findings, the post-audit disclosures, and the standing constraints |
| [docs/futures_execution_v1.md](docs/futures_execution_v1.md) | Futures Execution v1: dry-run-only USD-M perpetuals, LONG and SHORT, and the risk boundary that does not move |
| [docs/futures_dry_run_validation.md](docs/futures_dry_run_validation.md) | The operational protocol Futures Execution v1 was validated against, frozen before it was evaluated |
| [docs/research_reproduction.md](docs/research_reproduction.md) | Reproducing the research from a fresh clone, without the sealed block |
| [docs/research_roadmap.md](docs/research_roadmap.md) | What has been asked, what was answered, what was never opened, and what is next |

## Repository layout

```
chimera/       Shared, dependency-light core: features, contracts, risk, safety,
               metrics, notifications, inference client. No torch, no freqtrade.
chimera/futures/  Dry-run USD-M perpetual execution: positions, order state
               machine, venue constraints, fees and funding, reconciliation.
nn/            Data pipeline, model, training, evaluation, walk-forward,
               artifact registry, inference service.
strategies/    Freqtrade strategies and the risk-aware base class.
tools/         CLI entrypoints: backfill, build_features, run_bot, smoke.
conf/          Freqtrade configs, Prometheus, Alertmanager, alert rules.
grafana/       Datasource and dashboard provisioning.
tests/         The test suite.
docs/          Architecture, ML pipeline, risk, dry-run, audit.
```

## License

MIT — see [LICENSE](LICENSE).
