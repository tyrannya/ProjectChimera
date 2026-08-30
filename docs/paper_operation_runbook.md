# Paper-operation runbook

How to run `Pythia → mode → Aegis → Hermes → futures dry-run venue → Argus` for
days at a time, what to watch, and what has to be built first.

**Sustained multi-day paper validation has NOT been performed.** Nothing in this
repository claims it has. What exists is `tools/paper_run.py` — the loop — and an
engineering smoke of it. This document is the procedure for the sustained run
that has not happened yet.

## 1. What exists today

| piece | state |
| --- | --- |
| the loop | `tools/paper_run.py`, tested |
| execution chain | `chimera.futures`, dry-run only, 16/16 protocol invariants in [`futures_dry_run_validation.md`](futures_dry_run_validation.md) |
| mode controller | `chimera.modes`, [`trading_modes_v1.md`](trading_modes_v1.md) |
| consensus | `chimera.consensus`, the function `P7` measured |
| telemetry | `chimera_mode_*` and `chimera_futures_*` |
| offline replay source | `ReplaySource`, frozen `P6` predictions |
| **live market-data source** | **missing — see §3** |
| **live specialist inference** | **missing — see §3** |

## 2. The engineering smoke

```
make paper-smoke          # or: python -m tools.paper_run --smoke --bars 500
```

Deterministic and offline. It walks closed decision bars in time order, runs the
consensus, asks the mode controller, and drives Hermes into the dry-run venue.

**Its expected result today is that it places no order at all.** `P6`, `P6-EXT`
and `P7` were negative, so no mode is eligible, the controller returns `FLAT` with
reason `specialist_not_viable` on every bar, and the correct number of orders is
zero. **That is the system working.** A smoke that started trading would mean
something had gone wrong with eligibility, not that the strategy had improved.

The report at `artifacts/paper_smoke/paper_run.json` carries
`claims.sustained_paper_validation: false`, `claims.live: false`,
`claims.alpha: false`, and a test asserts all three.

The order path itself is proven separately and does not depend on a mode being
eligible: `tests/test_paper_run.py` drives the same loop with a synthetic viable
specialist set and asserts positions open, reverse, and return to flat through the
venue.

## 3. What a sustained run needs that does not exist

1. **A live market-data feed of closed candles.** `LiveSource` is the seam; it
   raises today rather than pretending. It needs 1m candles from the exchange and
   `nn.multiclock.resample_from_minutes(..., boundary=None)` for the higher
   clocks — the same resampler the research used, with the research boundary
   disabled because "the future" is simply not in the frame yet.
2. **A specialist that serves predictions per clock.** `P6` fitted its
   specialists and measured them; `nn.benchmark` does not persist estimators, so
   there is nothing to serve. Building this means persisting a fitted model and
   its scaler per clock, and an inference path that reproduces the training-time
   feature construction exactly.
3. **An eligible mode.** Even with 1 and 2, `chimera.modes` will return `FLAT`
   until some checkpoint finds a viable specialist. That is deliberate.

Until all three exist, a "sustained paper run" would be a loop calling `FLAT` for
a week. Worth doing as a soak test of the plumbing; **not** worth describing as
paper validation of a strategy.

## 4. Running it for days

```
python -m tools.paper_run --smoke --bars 0 --out artifacts/paper_run_$(date -u +%Y%m%dT%H%M%SZ)
```

Run it under a supervisor that restarts on exit. State lives in the run's
`futures_state.json`; **point a restart at the same file**, or the executor
bootstraps from a venue-reported flat position and forgets the position it held.

### 4.1 Restart recovery

`FuturesExecutor.recover(reported)` is the only entry. On restart:

1. read the persisted state; an unreadable store adopts nothing and returns
   `None` — that is a stop, not a warning;
2. pass the venue's reported positions. Agreement resumes; disagreement marks the
   symbol **disputed**, moves every non-terminal order to
   `RECONCILIATION_REQUIRED`, and blocks the symbol until
   `resolve_reconciliation` is called with a note;
3. **agreement never clears a standing dispute.** Only an explicit resolution
   does.

### 4.2 Reconciliation

Reconciliation compares side, quantity and — when not flat — leverage and margin
mode. Under `ReconciliationPolicy.HALT` (the default) a mismatch stops the
symbol; under `FLATTEN` it also emergency-flattens. Watch
`chimera_futures_reconciliation_total{outcome="MISMATCH"}`: a non-zero value is
an incident, not a metric.

## 5. What to watch

| what | where | what "wrong" looks like |
| --- | --- | --- |
| LONG/SHORT balance | `chimera_futures_fills_total{side}` | one-sided over days, with a mode that can express both |
| Aegis vetoes | `chimera_futures_risk_vetoes_total{reason}`, `chimera_mode_risk_vetoes_total{mode,reason}` | a single reason dominating, especially `stale_data` or `stale_inference` |
| funding | `chimera_futures_funding_total{direction}` | funding outpacing gross return |
| partial fills | `chimera_futures_fills_total{kind="partial"}` | a rising share — the size is too large for the simulated book |
| slippage | `chimera_futures_slippage_bps` | drift above the 5 bps the fill model assumes |
| exposure | `chimera_futures_gross_exposure`, `_net_exposure` | gross above net with one mode active |
| turnover | `chimera_futures_turnover_total` | growth without matching fills |
| mode dwell | `chimera_mode_active_seconds_total{mode}` | churn between modes, or 100% in one |
| FLAT fraction | `chimera_mode_selected{mode="FLAT"}` | **1.0 is correct today** |
| latency | `chimera_futures_execution_latency_seconds` | tail beyond the decision cadence |
| data delay | `chimera_data_delay_seconds{pair}` | past `max_data_delay_s` (300s) — Aegis vetoes on it |
| telemetry continuity | scrape gaps | a gap is a missing observation, not a quiet period |

## 6. Emergency flatten

`FuturesExecutor.emergency_flatten(symbol, cause, reference_price)` is reachable
while halted **and** while disputed — that is the point. It records the cause
before acting and marks the symbol disputed if the position is not flat
afterwards. It bypasses the risk gate deliberately: reducing exposure is never
the thing to block.

Causes are a bounded enum (`FlattenCause`). Operator-initiated flatten is
`FlattenCause.OPERATOR`.

## 7. Zero live-order reachability

Asserted, not assumed, by `tests/test_futures_no_live_path.py` and by invariant
`I13` of the frozen dry-run protocol:

- `FuturesExecutionConfig(dry_run=False)` raises `LiveFuturesNotImplemented`,
  **and still raises with `ENABLE_LIVE_TRADING` set**;
- `chimera/futures/*.py` imports no network, credential or crypto module —
  `requests`, `ccxt`, `socket`, `ssl`, `hmac`, `hashlib` and others are all
  refused by an AST scan that enrols any new module in the package
  automatically;
- no source in the package contains an exchange host, an API-key token, or
  `os.environ`;
- `DryRunFuturesVenue` is the only class in the package whose name ends in
  `Venue`, and the only `__all__` entry containing "live" is
  `LiveFuturesNotImplemented`;
- a fresh interpreter with a scrubbed environment reports
  `{"live_var_set": false, "acknowledged": false, "empty_config_is_dry_run": true,
  "venues": ["DryRunFuturesVenue"]}`.

**No real money. No authenticated order route. No leverage above 1x.**

## 8. Before a sustained run is described as validation

All of these, and none of them is optional:

- §3's three missing pieces exist;
- some checkpoint has found a viable specialist, so a mode is eligible;
- the run covered days of wall-clock time across at least one funding cycle and
  at least one restart;
- reconciliation ran on restart and agreed;
- the invariants in §5 were watched throughout and the exceptions written down;
- the report says which of them it is evidence about — **operational invariants,
  never alpha**. Simulated PnL from a dry-run venue is not evidence about a
  strategy, and this repository has said so since Futures Execution v1.
