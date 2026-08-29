# Futures Execution v1 — Binance USD-M perpetuals, dry-run only

**This is engineering, not a research checkpoint.** It is not P5, it produces no
alpha evidence, and nothing it measures may select a model, a feature, a
threshold, a horizon or a target. Its simulated PnL is a property of the fill
model, not of a market.

It exists because of a gap that four negative research checkpoints did not close
and could not: every strategy this repository could promote is currently unable
to express SHORT exposure at all. `strategies/scalp_futures.py` has
`can_short = False` with a comment saying why — Freqtrade refuses to load a
`can_short` strategy against a spot config, and every config shipped here is spot.
So the *instrument* is missing, independently of whether any *signal* is worth
acting on.

P4's negative result does not bear on this. P4 asked whether derivatives data
**predicts**; this asks how a perpetual position is **held and closed safely**.
A negative answer to the first is not an answer to the second, and confusing them
would leave the execution layer unable to act on a positive answer whenever one
arrives.

---

## 1. Scope, and what is deliberately outside it

| in | out |
| --- | --- |
| Binance USD-M perpetual semantics | every other venue |
| isolated margin, exactly 1x | cross margin, any other leverage |
| LONG and SHORT as first-class sides | hedge mode (two sides at once) |
| MARKET orders, simulated | LIMIT, STOP, a resting-order book |
| deterministic simulated fills | a real fill, a real venue, a real order |
| fees, funding, realised PnL | tax lots, multi-currency collateral |
| reconciliation, flatten, recovery | an automatic mismatch resolution |

**There is no live-order path, and there is no configuration that enables one.**
`FuturesExecutionConfig(dry_run=False)` raises `LiveFuturesNotImplemented`; it
raises *even when* `ENABLE_LIVE_TRADING` is set to `chimera.safety`'s exact
acknowledgement token, because the spot gate does not unlock a path that has not
been written. The only venue class in the package is `DryRunFuturesVenue`, which
simulates fills inside this process.

That claim is asserted about the source rather than promised in prose:
`tests/test_futures_no_live_path.py` parses every module in `chimera/futures/`
with `ast` and fails if any of them imports `requests`, `urllib`, `httpx`,
`aiohttp`, `socket`, `ssl`, `ccxt`, `binance`, `websockets`, `hmac` or `hashlib`;
if any of them reads an environment variable; or if any credential-shaped
identifier or Binance endpoint appears anywhere in the package. The whole suite
runs with a scrubbed environment.

---

## 2. The risk boundary, which does not move

    Pythia signal  ->  Aegis decision  ->  Hermes execution

`chimera.risk.RiskEngine` remains the **sole** portfolio and risk authority.
`chimera.futures.executor.FuturesExecutor` is the Hermes half and reaches
`DryRunFuturesVenue.submit` through exactly one route, which passes through
`RiskEngine.evaluate_entry`. There is no branch that submits without it. That is
what "an Aegis veto must make execution impossible" has to mean to be worth
stating — not a check that could be skipped, but the only door.

**No second risk authority appears.** The executor refuses orders on two grounds,
both of them venue facts rather than opinions:

* a **venue constraint** — a size, price or notional Binance would itself reject
  (§4);
* a **reconciliation mismatch** — a state it cannot act on truthfully (§7).

Every discretionary limit — exposure, drawdown, leverage, funding, liquidation
distance, order rate, the kill switch — lives in `chimera.risk` and is reached
through `evaluate_entry`. Margin and liquidation are **computed** in
`chimera.futures.accounting` and **reported** to Aegis as a number; nothing in
this package holds a threshold or vetoes anything on them.

### 2.1 Reductions are not gated, and that is the point

`OrderPurpose.increases_exposure` splits the two cases. `OPEN` and `INCREASE` go
through Aegis. `REDUCE`, `CLOSE` and `FLATTEN` do not.

A halted account is exactly the account that most needs to be able to close. An
entry gate that also blocked exits would turn a kill switch into a trap: the risk
engine would halt on a drawdown breach and then hold the losing position it
halted over. `emergency_flatten` is deliberately reachable while the engine is
halted, while a reconciliation mismatch stands, and after a previous flatten.

### 2.2 What Aegis is handed

`_ask_aegis` computes the liquidation price of the position the order **would
create** — not of the position that exists, which for an opening order is flat
and has no liquidation price at all — and passes it as `liquidation_price`.
`RiskLimits.min_liquidation_distance_pct` then decides.

At the 1x this package is fixed to, the isolated liquidation distance is 0.996
for both a LONG and a SHORT (Binance's tier-1 maintenance margin rate is 0.004),
so the default 0.5 threshold passes. It would bite at 2x and refuse everything at
3x. Recording that here because it is a live property of the default, not a
coincidence: the guard is real, and 1x is where it happens to be slack.

---

## 3. Position semantics

`PositionSide` is `FLAT | LONG | SHORT`, and `Position.quantity` is a
**magnitude** — the invariants `side is FLAT <=> quantity == 0` and, for an open
position, `entry_price > 0` are both checked on construction. SHORT is never a
negative number, and a position that was never entered at a price cannot be
built: `(mark − 0) × quantity` reports the whole notional as profit, and
`liquidation_price` refuses the same object, so the two would disagree about
whether it can exist.

That is not stylistic. With a signed quantity, "reduce by 3" applied to a
position of 2 becomes a SHORT of 1, and no arithmetic can distinguish that from
an intended reversal. `Position.apply_fill` refuses any reducing fill larger than
the position it reduces, by name:

> a reducing fill of 3 against a LONG position of 2 would reverse it. Closing
> never flips a position; plan the open as its own order.

`plan_transition(current, target)` produces the orders, one row per supported
transition:

| transition | intents |
| --- | --- |
| flat → LONG / SHORT | one `OPEN` |
| increase LONG / SHORT | one `INCREASE` |
| reduce LONG / SHORT | one `REDUCE`, `reduce_only` |
| LONG / SHORT → flat | one `CLOSE`, `reduce_only` |
| LONG ↔ SHORT | one `CLOSE` **then** one `OPEN` |
| already at target | none |

The reversal row is why this returns a list. A single order of
`current.quantity + target.quantity` is what a signed implementation writes, and
it is one arithmetic slip from a close that overshoots into a new position. Two
orders cannot overshoot: the first is `reduce_only` and exactly the size of what
it closes. If the close fills and the open is vetoed, the account is flat, which
is a safe place to stop.

`plan_flatten` is separate from `plan_transition(target=flat)` because the
*purpose* differs, and the purpose is what reaches telemetry and the persisted
reason.

---

## 4. Venue constraints, and failing closed

`SymbolConstraints` holds tick size, step size, quantity and price precision,
minimum quantity, minimum notional, symbol status, the maintenance margin rate,
the fee rates, and which order types and position sides the venue supports.
There is exactly one copy: no layer below keeps its own tick size.

`SymbolConstraints.from_dict` validates and never repairs. It refuses:

* any required field absent — including `maintenance_margin_rate`, because a
  liquidation price cannot be estimated without it and an unestimable liquidation
  price is not something Aegis may be handed as a number;
* a non-numeric or non-positive increment;
* a fee rate or maintenance margin rate outside its fraction range;
* a `step_size` needing more decimals than `quantity_precision` allows, or the
  same for `tick_size` and `price_precision` — contradictory metadata is refused
  rather than reconciled, because either field could be the wrong one and picking
  is a guess;
* a `min_quantity` that is not a multiple of `step_size`, which would make the
  smallest placeable order unplaceable;
* an order-type set containing nothing this package can simulate.

Quantities are quantized **down** to the step, never to nearest: rounding up
would hand the venue an order larger than the one Aegis approved. An order whose
quantity rounds to zero is `REJECTED` with that reason rather than sent.

The one exemption is a venue fact and is labelled as such: `check_placeable`
skips the minimum-notional test for a `reduce_only` order, because Binance does,
for the obvious reason that a dust position would otherwise be unclosable.

---

## 5. Order state machine

`ALLOWED_TRANSITIONS` is a total table over `OrderState`:

    PLANNED     -> RISK_APPROVED | REJECTED | CANCELLED | FAILED
    RISK_APPROVED -> SUBMITTED | CANCELLED | FAILED
    SUBMITTED   -> ACKNOWLEDGED | REJECTED | FAILED | RECONCILIATION_REQUIRED
    ACKNOWLEDGED -> PARTIALLY_FILLED | FILLED | CANCELLED | REJECTED | FAILED
                    | RECONCILIATION_REQUIRED
    PARTIALLY_FILLED -> PARTIALLY_FILLED | FILLED | CANCELLED | FAILED
                    | RECONCILIATION_REQUIRED
    RECONCILIATION_REQUIRED -> CANCELLED | FAILED | FILLED   (explicit resolution only)
    FILLED | CANCELLED | REJECTED | FAILED -> (terminal)

A transition absent from the table raises `InvalidTransition` naming both states
and what was allowed. There is no "unknown transitions are permitted" branch,
because the failure it would hide — an order that fills after it was cancelled —
is exactly the one that duplicates exposure.

### 5.1 Idempotency

`OrderRecord.book_fill` is the one place `filled_quantity` moves, so
`filled_quantity <= intent.quantity` has somewhere to live: a venue that reports
more fills than the order carried is refused and flagged `over_delivered`, rather
than driving `remaining_quantity` negative — which any cancel-the-rest or resize
path downstream reads as a negative order size. It is the record-level analogue
of the reversal guard `Position.apply_fill` enforces.

`net_exposure` and `gross_exposure` refuse a position they have no price for.
Skipping one silently under-reports exposure, and by the most exactly when a
symbol's feed is broken — the moment a risk check reading the number most needs
it to be right.

`OrderEvent.event_id` is required and is the idempotency key.
`FuturesExecutor.apply_event` returns immediately for an id already in
`OrderRecord.applied_events`, having changed no state, no quantity, no fee, no
position and no ledger entry. `applied_events` is **persisted with the record**,
so a redelivered fill, a replayed journal and a restarted process all get the
same answer.

Funding is deduplicated the same way, by `FundingEvent.settlement_id`, in
`Ledger.book_funding` — a restart must not re-charge funding the account has
already paid.

---

## 6. Accounting

Three cash flows, kept apart because they are not one quantity with different
signs:

* **trading fees** — always a cost, always a positive magnitude in
  `Ledger.trading_fees`;
* **funding paid** and **funding received** — two fields and two counters. A
  strategy that pays 12 and receives 10 is not the same as one that pays 2 and
  receives nothing, even though both net to −2: the first is running a position
  the market charges it to hold;
* **realised PnL** — booked only when a position is reduced. Unrealised PnL is
  derived on demand from a mark price and is never accumulated.

The funding sign convention, written once so the two places that need it cannot
disagree:

| side | funding rate | the position holder |
| --- | --- | --- |
| LONG | positive | **pays** |
| LONG | negative | receives |
| SHORT | positive | **receives** |
| SHORT | negative | pays |

which is exactly `cash_flow = -sign(side) * notional * rate`. A flat position is
stated to be zero rather than reached by multiplying by a zero sign.

**Funding here is an execution cash flow, not information.** Nothing in
`chimera.futures.accounting` may be read by a feature, a label or a model. P4
asked whether funding predicts and answered no; this is about what funding costs.

### 6.1 Liquidation, without invented precision

    LONG :  entry * (1 - 1/leverage + maintenance_margin_rate)
    SHORT:  entry * (1 + 1/leverage - maintenance_margin_rate)

At 1x that puts a LONG's liquidation at `entry * 0.004` — a 99.6% adverse move.
That is the right answer, not an oversight: an isolated 1x position can lose
essentially all of its margin before it is liquidated.

What it deliberately does **not** model, each because it is unknowable from what
the package is given and inventing it would put precision into a figure Aegis
then treats as real: accrued funding, unrealised PnL from other positions (there
are none — isolated), the tiered maintenance-margin schedule above tier 1, and
the maintenance *amount* deduction. `margin_state` returns `None` for a flat
position rather than a zeroed record, because `liquidation_price=0` would read to
Aegis as "liquidation is 100% away" — a claim about a position that does not
exist.

---

## 7. Reconciliation

Three states are represented and compared, never merged:

* **intended** — the `TargetPosition` the strategy wants;
* **local** — `FuturesState.positions`, the persisted execution truth;
* **reported** — `DryRunFuturesVenue.reported_position`, the venue's own view.

`reconcile()` compares local against reported. On disagreement it does four
things and one thing it does not do:

1. logs at CRITICAL with both views;
2. increments `chimera_futures_reconciliation_total{outcome="MISMATCH"}`;
3. moves every open order for the symbol to `RECONCILIATION_REQUIRED`;
4. applies the configured `ReconciliationPolicy` — `HALT` (default) stops, or
   `FLATTEN` emergency-flattens the local position and *still* refuses to trade,
   because flattening resolves the exposure and not the disagreement;

and it **never replaces the local position with the reported one.** That would
make every mismatch invisible in exactly the situation where a human needs to see
it. `require_ready` then refuses to plan anything for that symbol, so the mismatch
is a stop rather than a warning.

The only exit is `resolve_reconciliation(symbol, adopted, note)`, which requires
a written reason. Automatic resolution is the thing the state exists to prevent.

A fill larger than an order's outstanding quantity takes the same route: the
order goes to `RECONCILIATION_REQUIRED` rather than being booked.

---

## 8. Restart and recovery

**An empty memory is not a flat account.** `FuturesStore.open` distinguishes
three cases a plain `dict.get` would flatten into one:

| outcome | meaning | what the executor does |
| --- | --- | --- |
| `LOADED` | a state file was read | act on it, after reconciling |
| `MISSING` | no state file exists | start **unbootstrapped**; refuse to plan |
| `UNREADABLE` | a state file exists and would not parse | start unbootstrapped, **leave the file exactly where it is**, and refuse to adopt anything |

The `UNREADABLE` case leaves the file untouched on purpose: overwriting the one
record of what the account was doing is how a recoverable incident becomes an
unrecoverable one. It also refuses `bootstrap`, and `recover()` returns without
adopting — because `recover({})` on a cold start is byte-for-byte the same call
whether the file was missing or corrupt, so an adoption there would undo the load
path's decision one line after it was taken, *while* overwriting the file.
`FuturesStore.adopt_after_unreadable(reported, note)` is the deliberate way
through: it takes a written reason, moves the unreadable file to `<name>.corrupt`
rather than over it, and only then adopts.

The parse guard catches `ArithmeticError` as well as `ValueError`, because the
likeliest corruption of all is a mangled number in a persisted field and
`Decimal("0.5O")` raises `decimal.InvalidOperation` — whose MRO reaches
`ArithmeticError` and never `ValueError`.

Writes are atomic — temp file, `fsync`, `os.replace` — so a crash midway leaves
the previous state rather than half of the new one. Every mutation is followed by
a write, which is what makes the five restart boundaries recoverable:

1. **pre-submission** — a `PLANNED` or `RISK_APPROVED` order never reached the
   venue, so recovery cancels it locally. Cancelling cannot orphan anything:
   `submit` is the only place the venue is told an order exists.
2. **acknowledged, no fill** — reconciled against what the venue reports now.
3. **partially filled** — the applied event ids are persisted, so local exposure
   is what actually filled: not zero, not the whole order.
4. **filled but local completion not persisted** — the fill's id is not in
   `applied_events`, so re-applying it completes the order exactly once.
5. **repeated recovery** — `recover()` is idempotent. It adopts a reported
   position only when there is no local state to contradict; where local state
   exists it *compares*.

---

## 9. Telemetry

The futures block in `chimera/metrics.py` extends the existing Prometheus
abstraction; it does not introduce a second one. Every label is a bounded enum —
a side, a purpose, a state, an outcome, a `FlattenCause` — and none carries a
free-text reason, an order id, a price or a quantity. An Aegis veto reaches
`chimera_futures_risk_vetoes_total` as a collapsed token (`halted`, `liquidation`,
`total_exposure`), never as the human sentence, because a label whose value set
grows with traffic is a new time series per event and Prometheus keeps them
forever. The one non-enum label is `symbol`, bounded by the configured whitelist
exactly as `chimera_data_delay_seconds{pair}` already is.

The position gauge is zeroed for every symbol it has ever published before the
current positions are written, because `FuturesState.set_position` removes a flat
position from the map — so a gauge published only for what is still there keeps
its last non-zero value forever after a close, and a panel reads an open position
on a flat account.

Emitted: signals by outcome, risk vetoes by reason, orders planned / submitted /
rejected, fills split into partial and full, slippage in bps, trading fees,
funding split into paid and received, turnover, position quantity by symbol and
side, gross and net exposure, realised and net PnL, drawdown of simulated net
PnL, reconciliation outcomes, invalid transitions by originating state,
emergency flattens by cause, recoveries by load outcome, and execution latency.

No secret can reach any of it: nothing in the package reads a credential, and
`tests/test_futures_telemetry.py` asserts that no metric name or help string
contains a token `chimera.safety` would consider secret-shaped.

---

## 10. What v1 does not do, and what would have to change

- **No wall-clock paper trading.** The validation in
  [`futures_dry_run_validation.md`](futures_dry_run_validation.md) is a
  deterministic replay. Sustained real-time paper operation against a live
  Binance USD-M feed is a separate operational requirement and is recorded there
  as one.
- **No live orders.** Adding them is a separate piece of work with its own
  review: a signed REST client, key handling, an order-id reconciliation strategy
  against a venue that outlives the process, and a rate limiter. None of it is a
  configuration flag away.
- **No LIMIT orders.** A resting-order book needs a queue model, and pretending
  to have one would put fake precision into every reported fill.
- **No hedge mode.** One position per symbol, one side at a time.
- **The fill model is a model.** `DeterministicFillModel` is adverse, partial-fill
  capable and reproducible. It is not a market. Any number it produces describes
  the model.
- **`RiskEngine` persists only its halt.** `peak_equity`, `day_start_equity`,
  the loss streak, the cooldown and the order-rate window all reset on restart —
  a pre-existing property of `chimera/risk.py`, unchanged here, and one a future
  session should address on the risk side rather than by growing a second store
  in this package.
