# Dry-run operation

Dry-run (paper trading) is the intended and default mode. Freqtrade simulates
fills against live market data; no order reaches an exchange.

## Prerequisites

1. A trained, promoted model — or accept that the strategy will hold:

   ```bash
   make sample
   python -m tools.build_features \
       --candles data/raw/synthetic/SYNTH_USDT_1h.parquet \
       --out data/datasets/synth.parquet \
       --exchange synthetic --pair SYNTH/USDT --timeframe 1h
   python -m nn.train --dataset data/datasets/synth.parquet --epochs 5 --promote
   ```

2. The inference service running:

   ```bash
   make infer     # or: docker compose up -d nn_infer
   curl -s localhost:3000/readyz | jq
   ```

No exchange credentials are required. Dry-run works without them, and
`tools/run_bot.py` says so rather than failing.

## Start

```bash
make dry-run EXCHANGE=binance STRATEGY=NNPredictorStrategy
```

or the whole stack:

```bash
make docker-up
docker compose logs -f freqtrade
```

`tools/run_bot.py` merges `conf/base.json` with `conf/<exchange>.test.json`,
resolves `${VAR}` placeholders from the environment, runs the safety gate, writes
the merged config with owner-only permissions (it can contain API keys), and
starts Freqtrade.

## Confirming it is really dry-run

```bash
python -m tools.run_bot --exchange binance --mode test --dry-run-only-check
# {"dry_run": true, "strategy": "NNPredictorStrategy"}
```

The startup log states it explicitly:

```
INFO: Dry-run mode: config requests dry-run
```

Live mode logs `WARNING: LIVE TRADING ENABLED` instead. If you do not see that
warning, no real orders are being placed.

Attempting live without the acknowledgement fails before writing a config:

```bash
$ python -m tools.run_bot --exchange binance --mode live --dry-run-only-check
ERROR: config sets dry_run=false but ENABLE_LIVE_TRADING is not set to
       'I_UNDERSTAND_THE_RISK'. Refusing to start: ...
$ echo $?
2
```

## What to watch

| Signal | Where | Means |
| --- | --- | --- |
| `chimera_risk_halted` | Grafana / Trading | Kill switch engaged; no new entries |
| `chimera_rejected_entries_total` | Grafana / Trading | Entries the risk engine blocked, by reason |
| `chimera_drawdown` | Grafana / Trading | Distance below peak equity |
| `chimera_predictions_total` | Grafana / ML | If everything is HOLD, the model is not trading |
| `chimera_inference_errors_total` | Grafana / ML | Client/service disagreement or an outage |
| `chimera_data_staleness_seconds` | Grafana / ML | Above `max_data_staleness_s`, entries are blocked |

An all-HOLD prediction stream is normal on a fresh or under-trained model and is
the intended failure mode, not a bug.

## Failure behaviour

| Failure | Result |
| --- | --- |
| Inference service down | HOLD. No new ML entries. Existing trades keep their stoploss and ROI. |
| Timeout | HOLD, after one bounded retry. |
| Malformed response | HOLD, logged. |
| Feature contract mismatch | HOLD, logged as an error — the strategy's code and the model's expectations differ. |
| Too little history | HOLD until enough clean candles exist. |
| Risk halt | All new entries blocked, alert sent once, halt persists across restart. |

Nothing here converts the strategy into a different strategy. A rule-based
fallback exists but must be requested explicitly with
`"nn_fallback_strategy": true`.

## Stopping

```bash
docker compose down          # or Ctrl-C for a local run
```

To clear a persisted risk halt after investigating:

```bash
rm user_data/risk_state.json
```

## What has been verified, and what has not

Verified locally, without credentials or exchange access:

- every module compiles and imports;
- the full test suite passes against real Freqtrade, torch, pandas and FastAPI;
- `conf/*.json` validate against Freqtrade's own JSON schema;
- all four strategies load through Freqtrade's `StrategyResolver`, not just as
  Python modules;
- the live-trading gate refuses a live config and exits 2;
- `docker compose config` parses;
- the end-to-end smoke path runs: synthetic candles → features → training →
  artifact → reload → service → prediction → strategy interpretation → risk
  decision;
- strategies produce the expected signals for LONG/HOLD/SHORT and for every
  failure mode.

Verified in CI (the `Docker build smoke` job, which is skipped on pull-request
events and must be run via `workflow_dispatch`):

- both images build from a clean checkout — `Dockerfile` and
  `nn/Dockerfile.nn_infer`.

**Not** verified anywhere yet, and requiring your environment:

- a real dry-run session against a live exchange feed (needs outbound network to
  an exchange);
- running the built images — they compile and install, but no container has been
  started and exercised end to end;
- Grafana rendering the provisioned dashboards against live Prometheus data;
- Telegram delivery (needs a bot token);
- anything about profitability, on any data.

Before trusting a dry-run session, run it yourself for a period long enough to
produce trades, and read `chimera_rejected_entries_total` to understand why the
risk engine blocked what it blocked.
