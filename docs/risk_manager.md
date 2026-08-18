# Risk engine

`chimera/risk.py` holds the limits, the sizing arithmetic and the kill switch.
`strategies/common/risk_manager.py` binds it to Freqtrade.

The previous `RiskManager` was never instantiated anywhere in the repository. It
imported an exception (`TemporaryStopException`) that Freqtrade does not export,
so it could not even be imported, and its only enforcement mechanism was an
unauthenticated `requests.post` to `localhost:8080`. This one is on the entry
path and is tested.

## Where it runs

Every entry goes through `confirm_trade_entry`, which Freqtrade calls before
placing the order. Returning `False` stops the order.

| Freqtrade callback | Risk engine action |
| --- | --- |
| `bot_loop_start` | `update_equity()` — recompute drawdown and daily loss, publish metrics |
| `custom_stake_amount` | `position_size()` — proposes risk-based sizing |
| `leverage` | clamps to `max_leverage` and records it for the gate |
| `confirm_trade_entry` | `evaluate_entry()` — **the gate**, on the real order |
| `order_filled` | `set_position_exposure()` / `close_position()` / `record_trade_result()` |

### The gate judges the actual order

`custom_stake_amount` only *proposes* a size. Freqtrade then runs
`get_valid_enter_price_and_stake`, which may raise the stake to the exchange
minimum, cap it, or round it, and finally computes
`amount = (stake / rate) * leverage` before calling `confirm_trade_entry`.

So the gate reconstructs what is actually being committed:

```
committed stake = amount * rate / leverage
```

and applies every limit to *that*, additionally rejecting the entry if it
exceeds what risk-based sizing would have allowed. An exchange minimum cannot
silently inflate a position past the risk envelope — if it would, the trade is
refused rather than taken at a size whose stated risk-per-trade is untrue.

Freqtrade does not pass leverage to `confirm_trade_entry`, so the `leverage()`
callback records it per pair. On spot it is always 1.0.

### Exposure bookkeeping

`order_filled` fires for **every** order reaching a closed state — entries,
partial fills, position adjustments and partial exits — and Freqtrade recomputes
`trade.stake_amount` as the trade's *total* stake each time. Exposure is
therefore **set**, not accumulated: `set_position_exposure(pair, stake)`. Adding
instead double-counted, reporting 400 of exposure for a 200 position that filled
in two parts.

State is keyed by pair because Freqtrade opens at most one trade per pair — it
removes pairs with an open trade from the candidate whitelist. Position
adjustments extend that one trade.

**Order-rate accounting** counts orders this gate *approves for submission*,
which is the rate the exchange actually sees. Rejected signals are not orders,
and counting fills instead would let a burst of unfilled orders through.

**Restart:** exposure is rebuilt from `order_filled` callbacks, so a restart
starts from an empty map until orders fill. The halt flag, by contrast, is
persisted — see the kill switch below.

The gate is a synchronous local check. No network call, no external service, and
nothing that can fail open.

## Position sizing

This is the part most often got wrong, so it is stated explicitly.

**`risk_per_trade_pct` is the fraction of equity lost if the stop is hit. It is
not the fraction of the wallet to spend.**

```
risk_capital  = equity × risk_per_trade_pct
stop_distance = |entry − stop| / entry
notional      = risk_capital / stop_distance
stake         = notional / leverage,  capped by max_position_pct × equity
```

With 10,000 equity, 1% risk per trade and a 5% stop:

```
risk_capital = 10,000 × 0.01 = 100
notional     = 100 / 0.05    = 2,000
```

A 5% adverse move on a 2,000 position loses exactly 100 — the 1% that was
intended. Buying "1% of the wallet" instead would be a 100 position, whose actual
risk is 5 — twenty times smaller than stated, which is a different strategy
wearing the same label.

`tests/test_risk_manager.py::test_sizing_risks_the_configured_fraction_not_the_wallet_fraction`
asserts both the size and the implied loss.

Sizing returns 0 when the stop distance is outside
`[min_stop_distance_pct, max_stop_distance_pct]`: too tight a stop implies an
unbounded position, too wide a stop means the trade's risk cannot be controlled.

Sizing deliberately does **not** shrink the stake to fit remaining exposure
headroom. Sizing answers "how large should this trade be?"; whether there is room
is `evaluate_entry`'s decision. If sizing clamped instead, a portfolio at 97% of
its exposure cap would quietly open a position 3% of the intended size — all of
the fees, none of the edge — and the exposure limit would never reject anything.

## Limits

Configured under `"risk"` in the Freqtrade config; unknown keys are ignored.

### Account

| Limit | Default | Effect when breached |
| --- | --- | --- |
| `max_drawdown_pct` | 0.15 | **Halt** |
| `max_daily_loss_pct` | 0.05 | **Halt** |
| `max_open_positions` | 3 | Reject entry |
| `max_total_exposure_pct` | 1.0 | Reject entry |
| `max_exposure_per_asset_pct` | 0.35 | Reject entry |

The daily-loss budget resets when the UTC date changes.

### Per trade

| Limit | Default |
| --- | --- |
| `risk_per_trade_pct` | 0.01 |
| `max_position_pct` | 0.25 |
| `max_leverage` | 1.0 (3.0 in code default) |
| `min_stop_distance_pct` | 0.005 |
| `max_stop_distance_pct` | 0.15 |

### Operational

| Limit | Default | Effect |
| --- | --- | --- |
| `max_orders_per_minute` | 10 | **Halt** — a runaway loop is a bug, not a busy day |
| `loss_streak_limit` | 3 | Start a cooldown |
| `cooldown_seconds` | 3600 | Reject entries while active |
| `max_data_delay_s` | 300 | Reject entry |
| `max_inference_staleness_s` | 300 | Reject entry |

The freshness guards are fed by `data_delay_seconds()` and
`inference_age_seconds()`, which `NNPredictorStrategy` implements from the last
candle and the last successful prediction.

**`max_data_delay_s` is a delay past the candle's close, not the candle's age.**
An OHLCV timestamp is the candle's *open* time, so a 1h candle that has only
just closed is already 3600s "old". Measuring age that way meant a 300s limit
rejected every single NN entry. The guard now computes
`now - (candle_open + timeframe)`, which is how Freqtrade itself measures candle
age in `IStrategy.ignore_expired_candle`, and clamps at zero so clock skew near
a boundary cannot read as staleness. Because it is a delay past close, one value
is meaningful on any timeframe.

### Futures

| Limit | Default | Effect |
| --- | --- | --- |
| `max_funding_rate` | 0.0005 | Reject entry (absolute value) |
| `min_liquidation_distance_pct` | 0.5 | Reject entry |
| `max_leverage` | 3.0 | Reject entry |

These only apply when the caller supplies `funding_rate` / `liquidation_price`.
The shipped configs are spot, so they are inert there.

## The kill switch

`halt(reason)` sets `RiskEngine.halted`, which is the **first** thing
`evaluate_entry` checks. A halted engine cannot approve an entry even if every
network path in the process is down.

Three properties:

- **Persistent.** The halt is written to `state_file` (default
  `user_data/risk_state.json`) and reloaded at startup, so restarting the bot
  does not clear it. `resume()` is the only way back, and it is an explicit
  operator action.
- **Fails closed.** An unreadable state file starts the engine halted rather than
  trading. Tested.
- **Alerts once.** `halt()` is idempotent — re-halting keeps the original reason
  and does not re-notify. Combined with `chimera/notify.py`'s deduplication, a
  persistent halt sends one message rather than one per bot loop.

Halt triggers: drawdown limit, daily loss limit, order-rate limit, non-positive
equity, and any explicit call.

## Rejection reasons and metrics

`evaluate_entry` returns a `RiskDecision` with a human-readable reason, which is
logged and collapsed to a low-cardinality label for the
`chimera_rejected_entries_total{reason=...}` counter. Free text is never used as
a Prometheus label — embedded numbers would create a new time series per
rejection.

Labels: `halted`, `cooldown`, `exchange_unhealthy`, `no_equity`, `stale_data`,
`stale_inference`, `max_positions`, `funding`, `leverage`, `liquidation`,
`sizing`, `total_exposure`, `asset_exposure`, `order_rate`, `other`.

## Using it directly

```python
from chimera.risk import RiskEngine, RiskLimits

engine = RiskEngine(
    RiskLimits(max_drawdown_pct=0.10, risk_per_trade_pct=0.005),
    state_path="user_data/risk_state.json",
)

engine.update_equity(10_000.0)

decision = engine.evaluate_entry(
    pair="BTC/USDT",
    equity=10_000.0,
    entry_price=60_000.0,
    stop_price=57_000.0,
    data_delay_s=30.0,
    inference_age_s=45.0,
)

if decision.allowed:
    place_order(stake=decision.stake)
else:
    log.info("blocked: %s", decision.reason)
```

Strategies get this for free by subclassing `RiskAwareStrategy`.
