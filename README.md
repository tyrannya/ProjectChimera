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
| Prometheus + Grafana | Working | Every panel queries a metric this code exports |
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

## Repository layout

```
chimera/       Shared, dependency-light core: features, contracts, risk, safety,
               metrics, notifications, inference client. No torch, no freqtrade.
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
