# Engineering audit — ProjectChimera

Baseline recorded before the rebuild, at commit `3a09dc9` on branch
`claude/chimera-engineering-rebuild-ua70ns`.

This is a working document: it records what was found, and what the rebuild
changes. It is not a full report.

## Baseline check results (before changes)

| Check | Result |
| --- | --- |
| `python -m compileall .` | **FAIL** — 2 files with syntax errors |
| `pytest` | **FAIL** — 3 collection errors, 0 tests executed |
| `pre-commit run --all-files` | not runnable — `pre-commit` not installed, and hooks fail on the syntax errors above |
| `docker compose config` | passes, but describes an unbuildable stack (see below) |

Environment at audit time had none of the project's declared runtime
dependencies installed (`pandas`, `torch`, `ccxt`, `ta`, `freqtrade`,
`bentoml`, `mlflow` all absent), which is itself a finding: `requirements.txt`
declares only `requests` and `python-dotenv`.

## Broken components

1. **`nn/train.py` does not parse.** `IndentationError`/`SyntaxError` at line
   190: a `try:` block was opened at line 185 and the rest of `__main__` was
   left at the outer indentation level. The training entrypoint has never been
   runnable in this state.
2. **`tools/telegram_notifier.py` does not parse.** Four unterminated string
   literals (lines 42, 96, 123, 127) — literal newlines inside single-quoted
   strings. Every module that imports it therefore fails: `nn/train.py`,
   `nn/infer_service.py`, `strategies/common/notifying_strategy.py` and, through
   it, *all four strategies*.
3. **`tests/test_risk_manager.py` imports `CommonRiskManager`**, which does not
   exist. The class is `RiskManager` and has a completely different method set
   (`update_equity`/`register_order`/`funding_guard` vs. the tested
   `check_drawdown`/`log_http_429`/`update_funding_and_pnl`). The test file and
   `docs/risk_manager.md` describe a class that was never written.
4. **`nn/train.py` imports `from model_def import MTST`** — a bare top-level
   import that only resolves when CWD is `nn/`, contradicting the documented
   `python nn/train.py` invocation.
5. **`nn/infer_service.py` loads the model at import time** via
   `bentoml.torchscript.load_model("nn_predictor:prod")`. With no model in the
   store the module cannot even be imported, so it cannot be tested.
6. **`nn/Dockerfile.nn_infer` serves `service:svc`** — no `service.py` exists;
   the module is `infer_service.py`. `nn/Dockerfile` is a second, divergent
   image for the same service that binds port 5000 while compose maps 3000.
7. **`tools/webhook_pause.py` requires Flask**, which is in no requirements
   file, and posts to `localhost:8080` from inside a container where nothing
   listens.
8. **Two imports name Freqtrade APIs that do not exist.** Verified against a
   clean `pip install freqtrade` (resolved to **2026.7**):
   - `strategies/common/risk_manager.py` does
     `from freqtrade.exceptions import TemporaryStopException`. That name is
     not in `freqtrade.exceptions`, which exports `ConfigurationError`,
     `DDosProtection`, `DependencyException`, `ExchangeError`,
     `FreqtradeException`, `InsufficientFundsError`, `InvalidOrderException`,
     `OperationalException`, `PricingError`, `RetryableOrderError`,
     `StrategyError`, `TemporaryError`. The risk manager cannot be imported at
     all against a real Freqtrade install.
   - `strategies/nn_predictor_strategy.py` does
     `from freqtrade.exchange.exchange_utils import run_async_function`. That
     symbol does not exist either, so the NN strategy cannot be loaded.

   Both invented names are exactly the failure mode of writing against a
   remembered API instead of the installed one. The existing tests hid this by
   stubbing `freqtrade` with `types.ModuleType` and defining
   `TemporaryStopException` themselves — the tests assert against a fictional
   dependency, so they would stay green no matter what Freqtrade actually
   exports.

## Architectural inconsistencies

1. **No feature contract between pipeline, training and inference.**
   `make_features()` emits `close, sma_20, ema_50, rsi, macd, return_1, volume`.
   `train.py::load_data()` reads `["close", "return_1", "ema_9", "ema_21",
   "volume"]`. Two of the five training columns are never produced. Training
   would `KeyError` on real pipeline output.
2. **No preprocessing artifacts at all.** No scaler, no feature order, no
   sequence length, no horizon persisted with the model. Inference guesses
   `(1, 100, 5)` in `readyz` from a comment.
3. **Inference contract is unusable.** The strategy does
   `GET http://nn_infer:3000/predict` with *no body* and reads `js["score"]`;
   the service exposes a BentoML `NumpyNdarray` POST endpoint returning a bare
   array. These two have never been able to talk to each other.
4. **ML output semantics do not match the trading decision.** The model
   regresses the *absolute next close price* (`y = df.iloc[i+1]["close"]`), and
   the strategy tests `nn_score > 0.6`. For BTC this is `~60000 > 0.6` — always
   true. The entry condition is a constant.
5. **Data storage is inconsistent with documentation.** `save_delta()` writes
   Delta Lake tables; README and `train.py` both say `.parquet`, and
   `pd.read_parquet` cannot read the directory `backfill.py` produces.
6. **`RiskManager` is decorative.** Nothing instantiates it. It is not imported
   by any strategy, config, or entrypoint. `funding_guard()` fires a blind
   `requests.post("http://localhost:8080/api/v1/stop")` with no timeout, no
   auth, and no error handling — that HTTP call is the *entire* kill switch.
7. **No metrics are exported anywhere.** Five Grafana dashboards query
   `equity`, `drawdown`, `infer_latency_ms`, `drift_score`, `nn_online`,
   `sharpe_live`, `retrain_trigger`, `trading_equity`, `strategy_pnl`,
   `total_pnl` — not one of these series is produced by any code in the repo.
   Prometheus scrapes `freqtrade_exporter` and `drift_watcher`, neither of
   which exists as a service.

## Security / safety issues

1. **`.env` is committed** (`d40f38e "Update .env"`). It currently holds only
   placeholders, but it is a tracked runtime config file and `.gitignore`
   claims to exclude it — the ignore rule is inert for an already-tracked file,
   so any real key written there would be committed on the next `git add`.
2. **Live trading is one CLI word away.** `./tools/start.sh binance live` loads
   `conf/binance.live.json` with `"dry_run": false` and passes no `--dry-run`.
   There is no confirmation, no guard, and no environment gate. Possession of
   an API key is sufficient.
3. **`.github/workflows/dryrun.yml` runs `./tools/start.sh binance test`** on a
   nightly schedule. That path is one argument away from `live`, and the
   workflow already has repository secrets in scope.
4. **Config secret interpolation does not work.** `conf/*.json` contain
   `"key": "${BINANCE_KEY}"`. Freqtrade does not expand `${...}` inside JSON
   config values, so the literal string `${BINANCE_KEY}` is passed as the API
   key. This fails closed today, but silently.
5. **Telegram token is read from a committed `.env`** and the notifier prints
   `token[:10]` to stdout in its `__main__` block.

## ML problems

1. **Target is the absolute future price**, an unbounded non-stationary
   quantity, trained with `HuberLoss` on unnormalised BTC prices (~10^4). No
   scaling is applied anywhere.
2. **No cost awareness.** Nothing in the target, loss, or decision rule
   references fees or slippage.
3. **`policy_gradient_reward` is not a gradient path** — `torch.sign` has zero
   gradient almost everywhere and `==` produces a non-differentiable bool. The
   term `(1 - policy_gradient_reward(...))` contributes exactly nothing to
   backprop; it is a constant added to the loss.
4. **8-layer, d_model=128, 8-head Transformer** is hardcoded and grossly
   oversized for the data volume; none of the dimensions are configurable.
5. **`GradScaler`/`autocast` are used unconditionally**, including on CPU,
   where `torch.cuda.amp` is deprecated and the scaler is a no-op that still
   warns.
6. **Ray Tune with `num_samples=100` is mandatory** — there is no way to run a
   short training pass. `tune.report(sharpe=...)` is also the removed
   pre-2.0 API.
7. **Validation split is `int(len(X) * 0.8)`** — chronological by luck, but
   there is no test set, no walk-forward, and no leakage test. Windows
   straddling the split boundary overlap between train and validation.
8. **Duplicate model registration.** `save_and_register` calls
   `mlflow.register_model` twice and sets the `prod` alias twice, creating two
   versions per run and promoting to production unconditionally on completion.

## Backtesting problems

1. **`NNPredictorStrategy.populate_indicators` makes a network call per
   dataframe**, then broadcasts a single scalar across every historical row —
   so a backtest applies today's prediction to 2023 candles. This is
   look-ahead by construction.
2. **Silent fallback changes the strategy's identity.** If the NN call fails,
   the strategy quietly becomes an EMA/RSI rule strategy. Entry and exit both
   test `nn_score` against `0.6`, so with the fallback `nn_score = 0` the exit
   condition `0 < 0.6` is permanently true.
3. **`ScalpFutures` sets `orderbook_delta = 0`** when absent — which is always,
   since nothing populates it. Both entry conditions require it to be non-zero,
   so the strategy can never enter. It is dead code presented as working.
4. **`ArbMM` sets `zscore = 0`** when `eth_btc`/`eth_usdt` are absent — always.
   Entry requires `|zscore| > 2`; it can never enter.
5. **`SwingSpot` emits `enter_short`/`exit_short`** but no config sets
   `trading_mode: futures` or `can_short`, so the short side is ignored.
6. **`conf/base.json` is not a valid Freqtrade config**: `symbols` is not a
   Freqtrade key (`exchange.pair_whitelist` is), `initial_state: "paused"` is
   not a valid value (`running`/`stopped`), and required keys
   (`stake_currency`, `stake_amount`, `max_open_trades`, `timeframe`,
   `entry_pricing`, `exit_pricing`) are all missing.
7. No fee or slippage assumption is stated anywhere, and no lookahead analysis
   is run.

## CI problems

1. **`build.yml` declares `name:` twice** — a duplicate mapping key, so the
   workflow file is invalid YAML for strict parsers and at minimum ambiguous.
2. **`train.yml` ends with a ` main` merge leftover**, targets a
   non-existent `[self-hosted, gpu]` runner, runs `python nn/train.py` with no
   `--features` (a required argument), and pushes to a literal
   `ghcr.io/<org>/nn_infer:latest`.
3. **`pytest.yml` installs only `pandas pytest`** — not `ta`, `ccxt`,
   `deltalake`, or anything else the tests import. The suite errors at
   collection.
4. **`dryrun.yml`** runs a live-capable script on a schedule (see safety), and
   `curl http://localhost:3000/livez` against a container whose image is never
   built by CI.
5. No lint, no format check, no config validation, no compile check.

## Duplicated / misleading artifacts

- `prometheus.yml` and `conf/prometheus.yml`; `alertmanager.yml` and
  `conf/alertmanager.yml` (compose only mounts the `conf/` copies).
- `LICENSE` and `MIT License.txt` — identical MIT text.
- `grafana/provisioning/dashboards/trading_pnl.json` and `Trading-PNL.json`;
  `nn_latency.json` and `NN-Health.json` overlap. None are wired into a Grafana
  provisioning config, and compose mounts no volume into the Grafana container.
- `README.md` contains four merge-conflict remnants (`=======`, ` main`) and
  documents three mutually contradictory ways to start the stack.
- Pervasive `# Added`, `# Changed`, `# Removed`, `# Duplicate import removed`
  comments left by previous automated edits.

## What this rebuild changes

Ordered by dependency, not by importance.

1. Repository health: fix both syntax errors, remove merge leftovers, make one
   coherent dependency story split into runtime / ML / dev extras.
2. Secrets: untrack `.env`, validate required environment at startup, keep
   secrets out of logs.
3. Safety: dry-run by default; live trading requires
   `ENABLE_LIVE_TRADING=I_UNDERSTAND_THE_RISK` *and* an explicit live config,
   enforced in code before Freqtrade starts. Remove the live-capable path from
   CI.
4. Data: rewrite the pipeline around a validated OHLCV schema (UTC, monotonic,
   deduplicated, gap-detected) and a deterministic, documented feature set
   written to Parquet with metadata.
5. ML target: replace absolute-price regression with cost-aware
   SHORT/HOLD/LONG classification over a configurable horizon.
6. Training: chronological splits with an embargo, persisted preprocessing
   artifacts, CPU-first, optional tuning, naive and rule-based baselines
   reported alongside the model.
7. Inference: a schema-validated `POST /predict` with an explicit response
   contract, `/livez` and `/readyz`.
8. Strategy: build the real feature window from the dataframe, call inference
   once per closed candle, cache by pair+timestamp+model version, and fail
   closed to HOLD.
9. Risk: a real risk engine that participates in the entry path, with a kill
   switch backed by local state rather than an HTTP call.
10. Observability: export the metrics the dashboards actually query; delete
    panels with no source.
11. Docker/CI/docs: buildable images, a CI graph that matches the code, and a
    README that matches the repository.
