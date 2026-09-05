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

The futures path reaches the same gate through
`chimera.futures.executor.FuturesExecutor._ask_aegis`, which asks about
exposure-*increasing* orders only, and the demo runner adds the operational
notes described under [Persisted state](#persisted-state).

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

**Restart:** the whole state is persisted, exposure included — see
[Persisted state](#persisted-state).

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
| `max_funding_rate` | 0.0005 | Reject entry (absolute value, when no side is given) |
| `max_funding_cost_rate` | 0.0005 | Reject entry (what the named side would pay) |
| `funding_adverse_streak_limit` | 3 | **Funding halt** — refuse increases |
| `min_liquidation_distance_pct` | 0.5 | Reject entry |
| `max_leverage` | 3.0 | Reject entry |

The two funding limits are separate keys on purpose. `max_funding_rate` is an
existing configurable key and its meaning is the sign-blind bound; giving that
key the side-aware meaning would silently change what any config that sets it
enforces, without anybody editing that config.

These only apply when the caller supplies `funding_rate` / `liquidation_price`.
The shipped configs are spot, so they are inert there.

**The funding sign.** The cost a position pays per settlement is
`sign(side) × rate`: a long pays a positive rate, a short pays a negative one.
That is the table `chimera/futures/accounting.py` states once for the whole
system, written there as a *cash flow* (`-sign(side) × notional × rate`, negative
when the position pays). A cost is the negation of that cash flow, so the two
are one statement read from opposite ends.

Without a side there is no way to tell a cost from a rebate, so the sign-blind
check bounds `|rate|`. With equal thresholds it rejects a superset of what the
side-aware check rejects, which is why a caller whose side cannot be read
(`FLAT`, or anything unrecognised) falls back to it rather than past it.

**The liquidation figure.** `min_liquidation_distance_pct` is judged against the
liquidation price of the position the order *would produce*, computed by
`margin_state` and handed over by the executor. A `liquidation_price` of `None`
means "there is nothing here to judge", which is true of a flat position and
false of one whose figure merely could not be computed — so the executor now
vetoes an exposure increase whose prospective position is non-flat and has no
liquidation price, with the reason `liquidation unknown` (metric label
`liquidation_unknown`). A genuinely flat prospective still passes `None`.

## The kill switch

`halt(reason)` sets `RiskEngine.halted`, which is the **first** thing
`evaluate_entry` checks. A halted engine cannot approve an entry even if every
network path in the process is down.

Three properties:

- **Persistent.** The halt is written to `state_file` (default
  `user_data/risk_state.json`) and reloaded at startup, so restarting the bot
  does not clear it. `resume()` is the only way back, and it is an explicit
  operator action. It also clears the kill-switch mirror and the funding halt —
  the funding halt because that is the only exit it has, since the streak's own
  remedy is to reduce and a flat position never settles again. It leaves the
  reconciliation disputes and the stale-feed mark alone: each already has a
  clearing path that works while the account is flat, and a blanket resume that
  silently forgot a disputed position would be the failure the dispute exists to
  prevent.
- **Fails closed.** An unreadable state file starts the engine halted rather than
  trading. Tested.
- **Alerts once.** `halt()` is idempotent — re-halting keeps the original reason
  and does not re-notify. Combined with `chimera/notify.py`'s deduplication, a
  persistent halt sends one message rather than one per bot loop.

Halt triggers: drawdown limit, daily loss limit, order-rate limit, non-positive
equity, the kill-switch file, and any explicit call.

### The kill-switch file

`check_kill_switch()` looks for the file named by the engine's
`kill_switch_path` and halts with the reason `kill_switch` when it is there. It
is a file rather than a config key so that a human with a shell can stop new
exposure in one command, without restarting anything.

**It has to be wired.** An engine constructed without a `kill_switch_path` has
no switch configured, and `check_kill_switch()` is then a no-op returning
`False` — "nothing is watching", not "the switch is off".
`DEFAULT_KILL_SWITCH_PATH` (`user_data/KILL_SWITCH`) is the path a deployment
passes, not a default the engine applies on its own, because that path is
*relative*: an engine that applied it would resolve it against whatever
directory the process started in. Every `RiskEngine` in the repository — the
Freqtrade strategy, the smoke and paper tools, and the generator behind the
frozen `artifacts/futures_dry_run_v1` — would then read an untracked file that
no committed input names, so an engaged switch on one host would change the
frozen protocol's output and fail much of the test suite. Evidence must be a
function of committed inputs, and a guard whose reach depends on the current
working directory is not a guard anyway.

The engine checks the switch once at construction, when it was given a path.
Re-reading it on every tick is the **runner's** contract; nothing in this
package drives that loop yet, and the demo runner is where it will live.

**The halt is level-triggered.** While the file is there, every check
re-asserts the halt. Halting only on the absent-to-present transition left one
state that traded through an engaged switch — a persisted mirror already set to
`true` beside `halted: false`, which is what hand-editing a halt out of the
state file produces — where the method answered "engaged" and the engine
approved entries anyway. `halt()` is idempotent, so re-asserting costs nothing.

**Absent and unexaminable are different answers.** Only `FileNotFoundError`
means absent. Any other `OSError` — a parent that is not a directory, a
permission denial, an I/O error on the mount — means the answer is unknown, and
an unknown kill switch is treated as an engaged one. `Path.exists()` is
deliberately not used: it turns several of those errors into a confident
`False`. The offending path and the errno go to the log; the halt *reason* names
the problem without them, because `snapshot()` hashes the reason and a host's
absolute path or an OS-specific errno string would make two hosts in the same
semantic state hash differently.

Removing the file clears the flag but **not** the halt it caused; `resume()`
does that, and if the file is still on disk the next check halts again.

## Persisted state

The whole decision-relevant state is written to `state_file` after every
mutation, under the schema string `chimera.risk-state/1`:

| field | why it is persisted |
| --- | --- |
| `equity`, `peak_equity` | the drawdown is measured from the peak, not from the restart |
| `day_start_equity`, `day` | the daily-loss budget belongs to the UTC day, not to the process |
| `daily_pnl` | the daily report is written from the file, not from a live engine. A wipeout is recorded before it is halted on, so the report for the day an account went to zero does not show a flat P&L |
| `open_positions` | the exposure limits are cumulative |
| `order_times` | the live 60-second rate window, pruned before every write |
| `consecutive_losses`, `cooldown_until` | a restart is not a way out of a cooldown |
| `halted`, `halt_reason` | unchanged |
| `kill_switch` | whether the file was there, or unexaminable, at the last check |
| `stale_feed_since` | restarting into a dead feed is not a fresh feed |
| `reconciliation_disputed` | a dispute a reboot forgets is a dispute that gets traded through |
| `funding_adverse_streak`, `funding_halt` | the streak is about settlements, not about uptime |
| `updated_at`, `schema` | when it was written, and what shape it is in |

The **drawdown is not persisted**. It is derived by `current_drawdown()` from
`peak_equity` and `equity`; a stored copy is a second authority that can
disagree with the two numbers it came from, and the disagreement would only ever
be discovered by a limit failing to fire.

Before this, only the halt survived a restart. A bot that stopped between its
equity peak and the fall that breached the drawdown limit came back measuring
that fall from the wrong peak — and approved the trade the limit exists to stop.

### Durability

The write goes to a temporary sibling, is flushed and `fsync`-ed, and is then
swapped in with `os.replace` — the pattern `FuturesStore.save` uses. A crash in
the middle of a write leaves the previous file intact rather than a truncated
one. A write that cannot land is logged, not raised: a halt that cannot be
written down is still a halt in memory, and letting the `OSError` unwind the
caller would turn a disk problem into a skipped guard.

`updated_at` is stamped from the engine's own clock, not the wall clock, so an
injected or replayed clock governs every timestamp the engine writes rather than
only the ones a decision reads. `update_equity()` takes the UTC day from the
same clock when the caller names no time, for the same reason: `day` and
`day_start_equity` are persisted and hashed into `snapshot()`, and two of the
engine's fields being governed by a clock that disagrees with the cooldown's and
the rate window's would be a divergence a replay could not reproduce.

### Reading a file that cannot be believed

| file | result |
| --- | --- |
| written by this build | loaded |
| absent (`FileNotFoundError`) | a cold start, on defaults |
| the legacy `{halted, halt_reason, updated_at}` | the halt is adopted; nothing else is invented |
| no schema, but other keys | **halt**, reason names the keys |
| unparseable, or not a JSON object | **halt**, reason names the parse error |
| a schema this build does not know | **halt**, reason names the schema |
| missing or mistyped field | **halt**, reason names the field |
| present but unexaminable (`ENOTDIR`, `EACCES`, `ELOOP`, an I/O error) | **halt**; the path goes to the log |

Nothing on that list is repaired with a default. A missing number means the file
does not say what the account was doing, and presenting that as "flat, no
drawdown, no cooldown" is the one reading that can lose money. The read decides
whether there is a file, rather than `Path.exists()` deciding first: `exists()`
answers `False` for a path it merely could not examine, so a state file on a
degraded mount was read as "there is no state file" and started the engine on
unhalted defaults — losing the halt, the peak equity, the cooldown, the order
window and any open dispute in one step.

**The unreadable file is preserved, not overwritten.** While a load has failed
closed the engine writes *nothing*, because every mutation persists and the next
tick would otherwise replace the one record of what the account was doing with
an all-defaults document — after which a single `resume()` leaves every restart
loading a confident "flat, no drawdown, no dispute" with nothing left to
contradict it. A restart therefore re-reads the same bad bytes and re-halts,
which is the intended behaviour and not a loop to route around.

`adopt_after_unreadable(note)` is the deliberate way through, mirroring
`FuturesStore.adopt_after_unreadable`. It takes a written reason, moves the file
aside to `<name>.corrupt` so a later investigation still has it, and starts
recording again from an empty state. It is **not** a resume: the engine stays
halted with the reason it failed closed on, so "I have preserved the evidence"
and "I accept trading from an empty state" are never the same keystroke.

## Runner and operator notes

The demo runner feeds Aegis four facts it cannot observe for itself. All four
are persisted, so none of them is forgotten by a restart.

| call | effect |
| --- | --- |
| `note_feed(last_minute_close_ns, now_ns)` | marks the feed stale when the delay is **above** `max_data_delay_s`; exactly at the limit is not stale, the same comparison `evaluate_entry` makes on `data_delay_s`. The mark records *when* it went stale, not the latest look, and a fresh minute clears it. A **negative** delay is stale too: the last complete minute cannot close after the present, so the clocks or the units disagree and the feed's age is unknown — clearing the mark there would let a skewed clock re-enable entries on data nobody has checked. |
| `note_reconciliation(symbol, disputed \| None)` | opens or closes a per-symbol dispute. A disputed symbol's increases are refused, because the size every limit would be applied to is not known. Only an explicit call with `None` clears it. |
| `note_funding_settlement(pair, side, rate)` | scores a settlement as paid or received by the position. Paying extends `funding_adverse_streak`; being paid resets it and lifts any funding halt; exactly zero leaves it alone. At `funding_adverse_streak_limit` the engine refuses increases and the runner must reduce. A side that is neither LONG nor SHORT raises `RiskViolation` rather than being guessed. |
| `adopt_after_unreadable(note)` | above. |
| `check_kill_switch()` | above. |

`snapshot()` returns the semantic state for a decision log to hash. It carries
what a decision depends on and nothing that merely describes the process: no
write time, no path, no host, no PID. `updated_at` therefore stays in the file
and out of the snapshot — it changes on every write, including writes that
changed no decision, and a hash that moved for that reason would report two
identical states as different.

## Reductions are never gated

`OrderPurpose.increases_exposure` is true only for `OPEN` and `INCREASE`. The
futures executor asks Aegis about those and nothing else, and
`emergency_flatten` passes `bypass_risk_gate=True`. So `REDUCE`, `CLOSE` and
`FLATTEN` remain possible while the engine is halted, kill-switched, in
cooldown, disputed, funding-halted or looking at a stale feed. A kill switch
that also blocked exits would be a trap rather than a brake, and
`tests/test_risk_new_rules.py` drives the real executor through every one of
those states to prove the exit still lands.

## Rejection reasons and metrics

`evaluate_entry` returns a `RiskDecision` with a human-readable reason, which is
logged and collapsed to a low-cardinality label for the
`chimera_rejected_entries_total{reason=...}` counter. Free text is never used as
a Prometheus label — embedded numbers would create a new time series per
rejection.

Labels: `halted`, `cooldown`, `exchange_unhealthy`, `no_equity`, `stale_data`,
`stale_inference`, `max_positions`, `funding`, `leverage`, `liquidation`,
`sizing`, `total_exposure`, `asset_exposure`, `order_rate`, `other`.

The futures path collapses the same reasons independently, in
`chimera.futures.executor._veto_label`, for `chimera_futures_risk_vetoes_total`.
It carries one label the spot table has no counterpart for,
`liquidation_unknown`, and it must be matched before `liquidation` because the
collapse is by prefix. The kill switch arrives there as `halted`, and a stale
feed as `stale_data`; the reconciliation-dispute and funding-halt reasons have
no label of their own yet and collapse to `other`, which is bounded and safe but
is a gap for the observability package to close.

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
