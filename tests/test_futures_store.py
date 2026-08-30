"""Persisted futures state, and the restart boundaries it exists for.

One claim runs through every test here: **an empty memory is not a flat
account.** A missing state file, an unparseable one and a loaded one are three
different situations, and the only one an executor may plan from is the third.
The first two leave it unbootstrapped, and the unreadable case additionally
leaves the file exactly where it was, because overwriting the only record of
what the account was doing is how a recoverable incident stops being one.

The second half of the file drives a real :class:`FuturesExecutor` against a
real :class:`DryRunFuturesVenue`, kills it at each of the five points a crash
can land on, and starts a second executor over ``FuturesStore.open`` of the same
path. What is under test there is that the file carried enough across the gap:
the order the venue never saw is cancelled, the partial fill that was booked
stays booked, and an event id already in ``applied_events`` still deduplicates
after the restart — which is what stops a redelivered fill booking the exposure
twice.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from chimera.futures import (
    STORE_SCHEMA,
    DeterministicFillModel,
    DryRunFuturesVenue,
    EventKind,
    FillPlan,
    FlattenCause,
    FuturesExecutor,
    FuturesStore,
    LoadOutcome,
    NotBootstrapped,
    OrderEvent,
    OrderIntent,
    OrderPurpose,
    OrderRecord,
    OrderSide,
    OrderState,
    Position,
    PositionSide,
    ReconciliationOutcome,
    ReconciliationRequired,
    StoreError,
    TargetPosition,
    load_constraint_source,
)
from chimera.risk import RiskEngine, RiskLimits

SYMBOL = "BTC/USDT:USDT"
#: On the 0.10 tick grid, so every simulated price below is exact.
REFERENCE = Decimal("60000")
#: REFERENCE plus the 5bp adverse slippage DeterministicFillModel puts on a BUY.
FILL_PRICE = Decimal("60030.00")
EQUITY = 100_000.0
#: Frozen so `record_flatten` writes a timestamp a test can assert on exactly.
FROZEN_CLOCK = 1_700_000_000.0
FROZEN_AT = "2023-11-14T22:13:20+00:00"
#: `_plan` derives this from the symbol and a per-executor counter.
FIRST_ORDER_ID = "BTCUSDTUSDT-000001"


class AcknowledgeOnlyFillModel:
    """A venue that accepts an order and then fills none of it.

    ``FillModel`` is the package's own seam for how a simulated order fills, so
    swapping it replaces nothing in the store or the executor. It is the only
    way to reach the "acknowledged, still working" restart boundary, which
    ``DeterministicFillModel`` never produces.
    """

    def plan(self, intent, reference_price, constraints) -> FillPlan:
        return FillPlan(fills=())


def state_path(tmp_path):
    """A state file inside a directory `save()` has to create for itself."""
    return tmp_path / "state" / "futures.json"


def risk_engine() -> RiskEngine:
    """An engine wide enough to approve the 0.5 BTC opens these tests need.

    The defaults veto almost every open, which is the right default and the
    wrong fixture: a restart test that never got a fill would pass for the wrong
    reason.
    """
    engine = RiskEngine(RiskLimits(max_position_pct=1.0, risk_per_trade_pct=0.5))
    engine.update_equity(EQUITY)
    return engine


def build_venue(fill_model=None) -> DryRunFuturesVenue:
    """A simulated exchange holding no position. Survives a restart, as one does."""
    return DryRunFuturesVenue(
        source=load_constraint_source(), fill_model=fill_model or DeterministicFillModel()
    )


def build_executor(path, venue) -> FuturesExecutor:
    """An executor over whatever is at ``path``, loaded the way a process would."""
    return FuturesExecutor(
        venue=venue,
        risk=risk_engine(),
        store=FuturesStore.open(path),
        clock=lambda: FROZEN_CLOCK,
    )


def open_long(executor, quantity: str = "0.5"):
    """Take the position from flat to LONG ``quantity`` through the whole path."""
    return executor.execute_target(
        TargetPosition(SYMBOL, PositionSide.LONG, Decimal(quantity)),
        REFERENCE,
        equity=EQUITY,
    )


def buy_intent(quantity: str) -> OrderIntent:
    """A well-formed opening order for the committed symbol."""
    return OrderIntent(
        symbol=SYMBOL,
        side=OrderSide.BUY,
        quantity=Decimal(quantity),
        purpose=OrderPurpose.OPEN,
        reduce_only=False,
        position_side=PositionSide.LONG,
    )


def strand_order(store, order_id: str, state: OrderState) -> OrderRecord:
    """Leave a pre-submission order on disk, as a crash before `submit` would.

    The executor persists the record at every step, so a process killed between
    the risk gate and ``venue.submit`` leaves exactly this: a PLANNED or
    RISK_APPROVED record the venue has never been told about.
    """
    record = OrderRecord(order_id=order_id, intent=buy_intent("0.100"))
    if state is OrderState.RISK_APPROVED:
        record.transition(OrderState.RISK_APPROVED)
    store.state.orders[order_id] = record
    store.save()
    return record


def submit_without_booking(executor, quantity: str, order_id: str):
    """Get the venue's events for a real SUBMITTED order, without applying them.

    ``FuturesExecutor._submit`` books every event in one go, so a test that has
    to stop halfway through — the whole point of the partial-fill and
    unpersisted-completion boundaries — has to hold the events itself. The
    record, the transitions, the venue and the fills are all the real ones.
    """
    intent = buy_intent(quantity)
    record = OrderRecord(order_id=order_id, intent=intent)
    executor.store.state.orders[order_id] = record
    record.transition(OrderState.RISK_APPROVED)
    record.transition(OrderState.SUBMITTED)
    executor.store.save()
    return executor.venue.submit(order_id, intent, REFERENCE)


def venue_view(venue) -> dict[str, Position]:
    """What a restarting process asks the exchange for before it does anything."""
    return {SYMBOL: venue.reported_position(SYMBOL)}


# --- an absent file is not a flat account ----------------------------------
def test_open_on_a_missing_file_reports_missing_and_holds_nothing(tmp_path):
    """Catches a store that treats "no file" as "no position".

    MISSING and LOADED-and-flat are indistinguishable once the outcome is
    dropped, and only one of them is safe to trade from.
    """
    path = state_path(tmp_path)
    store = FuturesStore.open(path)

    assert store.outcome is LoadOutcome.MISSING
    assert store.state.bootstrapped is False
    assert store.state.positions == {}
    assert store.state.orders == {}
    assert store.state.position(SYMBOL).side is PositionSide.FLAT
    assert not path.exists()


def test_an_executor_over_a_missing_state_file_refuses_to_plan(tmp_path):
    """Catches a restart that opens a second position on top of an unknown first.

    The refusal has to come before planning: an order recorded and then refused
    would still be an order this process invented from an empty memory.
    """
    path = state_path(tmp_path)
    executor = build_executor(path, build_venue())

    with pytest.raises(NotBootstrapped) as excinfo:
        open_long(executor)

    message = str(excinfo.value)
    assert "An empty state file is not a flat account" in message
    assert "call recover() with the venue's reported positions" in message
    assert executor.store.state.orders == {}
    assert executor.venue.reported_position(SYMBOL).side is PositionSide.FLAT
    assert not path.exists()


# --- an unreadable file is the worst case, and is treated as one -----------
def test_open_on_invalid_json_reports_unreadable_and_leaves_the_file_alone(tmp_path):
    """Catches a store that repairs a corrupt file by overwriting it.

    The bytes are the only record of what the account was doing; a truncated
    write is recoverable by hand, and a truncated write that has been replaced
    with `{}` is not.
    """
    path = state_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text('{"store_schema": "chimera.futures-execution-state/1", "posi')
    before = path.read_bytes()

    store = FuturesStore.open(path)

    assert store.outcome is LoadOutcome.UNREADABLE
    assert store.state.bootstrapped is False
    assert store.state.positions == {}
    assert path.read_bytes() == before


def test_open_on_a_foreign_schema_reports_unreadable_and_leaves_the_file_alone(tmp_path):
    """Catches a build that best-effort parses a file written by another build.

    Valid JSON is not a readable state file. A schema this build does not know
    may spell a quantity, a side or a fee differently, and guessing which is the
    same failure as guessing a tick size.
    """
    path = state_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(
        '{"store_schema": "chimera.futures-execution-state/999", "bootstrapped": true,'
        ' "positions": {}, "orders": {}, "ledger": {}, "flatten_reasons": []}\n'
    )
    before = path.read_bytes()

    store = FuturesStore.open(path)

    assert store.outcome is LoadOutcome.UNREADABLE
    assert store.state.bootstrapped is False
    assert store.state.positions == {}
    assert path.read_bytes() == before


def test_the_schema_a_readable_file_must_declare_is_the_one_that_is_written(tmp_path):
    """Guards the two literals above against a bump that only lands in one place."""
    path = state_path(tmp_path)
    store = FuturesStore.open(path)
    store.save()

    assert STORE_SCHEMA == "chimera.futures-execution-state/1"
    assert f'"store_schema": "{STORE_SCHEMA}"' in path.read_text()


# --- what survives a write --------------------------------------------------
def test_save_then_open_round_trips_every_field_with_decimals_intact(tmp_path):
    """Catches a field dropped from `to_dict`/`from_dict`, or turned into a float.

    A lost order, a lost fee or a position that comes back at 0.5000000001 is
    what the next restart would then treat as the account's truth.
    """
    path = state_path(tmp_path)
    executor = build_executor(path, build_venue())
    executor.recover({})
    open_long(executor)
    executor.store.record_flatten(SYMBOL, FlattenCause.OPERATOR.value, FROZEN_AT)

    reopened = FuturesStore.open(path)

    assert reopened.outcome is LoadOutcome.LOADED
    assert reopened.state.bootstrapped is True
    assert reopened.state.to_dict() == executor.store.state.to_dict()

    position = reopened.state.position(SYMBOL)
    assert isinstance(position.quantity, Decimal)
    assert position.side is PositionSide.LONG
    assert position.quantity == Decimal("0.500")
    assert position.entry_price == FILL_PRICE
    assert position.leverage == Decimal("1")
    assert position.margin_mode == "ISOLATED"

    ledger = reopened.state.ledger
    assert isinstance(ledger.trading_fees, Decimal)
    assert ledger.trading_fees == Decimal("15.00750000")
    assert ledger.turnover == Decimal("30015")
    assert ledger.realised_pnl == Decimal("0")
    assert ledger.applied_funding == []

    order = reopened.state.orders[FIRST_ORDER_ID]
    assert order.state is OrderState.FILLED
    assert order.filled_quantity == Decimal("0.500")
    assert order.average_price == FILL_PRICE
    assert order.fees == Decimal("15.00750000")
    assert order.intent.quantity == Decimal("0.500")
    assert order.applied_events == [f"{FIRST_ORDER_ID}:1", f"{FIRST_ORDER_ID}:2"]

    assert reopened.state.flatten_reasons == [
        {"symbol": SYMBOL, "reason": "OPERATOR", "at": FROZEN_AT}
    ]


def test_save_leaves_no_temporary_file_behind(tmp_path):
    """Catches an atomic write that renames the wrong path, or renames nothing.

    A ``.tmp`` sibling surviving the write means the next reader is one glob
    away from loading a half-written state as the real one.
    """
    path = state_path(tmp_path)
    executor = build_executor(path, build_venue())
    executor.recover({})
    open_long(executor)

    assert [p.name for p in sorted(path.parent.iterdir())] == [path.name]
    assert not (path.parent / (path.name + ".tmp")).exists()


# --- bootstrapping ----------------------------------------------------------
def test_bootstrap_adopts_the_reported_position_and_records_that_it_did(tmp_path):
    """Catches an adoption that is not persisted: the next restart would redo it."""
    path = state_path(tmp_path)
    store = FuturesStore.open(path)
    reported = Position(
        symbol=SYMBOL,
        side=PositionSide.SHORT,
        quantity=Decimal("0.250"),
        entry_price=Decimal("59000.00"),
    )

    store.bootstrap({SYMBOL: reported})

    assert store.state.bootstrapped is True
    assert store.state.position(SYMBOL) == reported

    reloaded = FuturesStore.open(path)
    assert reloaded.outcome is LoadOutcome.LOADED
    assert reloaded.state.bootstrapped is True
    assert reloaded.state.position(SYMBOL).quantity == Decimal("0.250")
    assert str(reloaded.state.position(SYMBOL).quantity) == "0.250"
    assert reloaded.state.position(SYMBOL).side is PositionSide.SHORT


def test_bootstrapping_twice_is_refused(tmp_path):
    """Catches a second adoption being treated as a harmless no-op.

    The second call carries a *newer* reported position, and quietly taking it
    is a reconciliation decision dressed up as start-up.
    """
    path = state_path(tmp_path)
    store = FuturesStore.open(path)
    store.bootstrap({})

    with pytest.raises(StoreError) as excinfo:
        store.bootstrap({SYMBOL: Position(symbol=SYMBOL)})

    assert "adopting a reported state twice is not a no-op" in str(excinfo.value)


def test_bootstrap_over_existing_local_positions_is_refused_as_reconciliation(tmp_path):
    """Catches the venue's view silently overwriting a local position.

    Resolving a disagreement in the venue's favour is exactly the thing the
    executor is supposed to refuse to trade through, so the store must not do it
    on the way up.
    """
    path = state_path(tmp_path)
    store = FuturesStore.open(path)
    store.state.set_position(
        Position(
            symbol=SYMBOL,
            side=PositionSide.LONG,
            quantity=Decimal("0.500"),
            entry_price=FILL_PRICE,
        )
    )

    with pytest.raises(StoreError) as excinfo:
        store.bootstrap({SYMBOL: Position(symbol=SYMBOL)})

    assert "That is reconciliation" in str(excinfo.value)
    assert store.state.bootstrapped is False
    assert store.state.position(SYMBOL).quantity == Decimal("0.500")


def test_bootstrap_refuses_a_reported_position_labelled_with_another_symbol(tmp_path):
    """Catches a mislabelled venue report being adopted under the wrong key."""
    store = FuturesStore.open(state_path(tmp_path))

    with pytest.raises(StoreError) as excinfo:
        store.bootstrap({SYMBOL: Position(symbol="ETH/USDT:USDT")})

    assert f"reported position for {SYMBOL} is labelled ETH/USDT:USDT" in str(excinfo.value)
    assert store.state.bootstrapped is False


# --- restart boundary 1: the order the venue never saw ---------------------
@pytest.mark.parametrize("stranded_state", [OrderState.PLANNED, OrderState.RISK_APPROVED])
def test_recovery_cancels_an_order_that_never_reached_the_venue(tmp_path, stranded_state):
    """Catches a pre-submission order being left open, or being submitted late.

    PLANNED and RISK_APPROVED are the two states the venue has provably not
    heard about, so cancelling them locally cannot orphan anything — and leaving
    them open would block the symbol behind an order that does not exist.
    """
    path = state_path(tmp_path)
    market = build_venue()
    first = build_executor(path, market)
    first.recover({})
    open_long(first)
    strand_order(first.store, "BTCUSDTUSDT-000009", stranded_state)

    second = build_executor(path, market)
    report = second.recover(venue_view(market))

    assert report.outcome is ReconciliationOutcome.AGREED
    stranded = second.store.state.orders["BTCUSDTUSDT-000009"]
    assert stranded.state is OrderState.CANCELLED
    assert stranded.reason == "not submitted before restart"
    assert stranded.filled_quantity == Decimal("0")
    # The venue holds only what the one real order filled: the stranded 0.100
    # never reached it, before the restart or after it.
    assert market.reported_position(SYMBOL).quantity == Decimal("0.500")
    assert second.position(SYMBOL).quantity == Decimal("0.500")
    assert second.store.state.orders[FIRST_ORDER_ID].state is OrderState.FILLED


# --- restart boundary 2: acknowledged, nothing filled ----------------------
def test_recovery_of_an_acknowledged_but_unfilled_order_leaves_the_account_flat(tmp_path):
    """Catches a recovery that books exposure for an order that filled nothing.

    An acknowledgement is the venue saying it has the order, not that it has
    traded, and the reconciliation that follows a restart has to agree with the
    venue that both sides are still flat.
    """
    path = state_path(tmp_path)
    market = build_venue(AcknowledgeOnlyFillModel())
    first = build_executor(path, market)
    first.recover({})
    records = open_long(first)

    assert records[0].state is OrderState.ACKNOWLEDGED
    assert records[0].filled_quantity == Decimal("0")

    second = build_executor(path, market)
    report = second.recover(venue_view(market))

    assert report.outcome is ReconciliationOutcome.AGREED
    assert second.store.state.positions == {}
    assert second.position(SYMBOL).side is PositionSide.FLAT
    assert second.position(SYMBOL).quantity == Decimal("0")
    assert market.reported_position(SYMBOL).side is PositionSide.FLAT
    assert second.store.state.orders[FIRST_ORDER_ID].state is OrderState.ACKNOWLEDGED
    assert second.ledger.turnover == Decimal("0")
    assert second.ledger.trading_fees == Decimal("0")


# --- restart boundary 3: partially filled ----------------------------------
def test_recovery_keeps_the_partial_fill_that_was_booked_and_no_more(tmp_path):
    """Catches a restart that rounds a partial fill to zero or to the full order.

    Zero would re-open exposure the account already has; the full order would
    book a fill that never happened. The recovered position has to be exactly
    what was persisted, and the disagreement with the venue has to be reported
    rather than silently resolved.
    """
    path = state_path(tmp_path)
    market = build_venue(DeterministicFillModel(max_fill_ratio=Decimal("0.5")))
    first = build_executor(path, market)
    first.recover({})
    events = submit_without_booking(first, "0.500", FIRST_ORDER_ID)

    assert [e.kind for e in events] == [
        EventKind.ACKNOWLEDGED,
        EventKind.PARTIAL_FILL,
        EventKind.FILL,
    ]
    # The process dies having booked the acknowledgement and the first fill.
    first.apply_event(FIRST_ORDER_ID, events[0], REFERENCE)
    first.apply_event(FIRST_ORDER_ID, events[1], REFERENCE)
    assert first.position(SYMBOL).quantity == Decimal("0.250")

    second = build_executor(path, market)
    report = second.recover(venue_view(market))

    assert second.position(SYMBOL).side is PositionSide.LONG
    assert second.position(SYMBOL).quantity == Decimal("0.250")
    assert second.position(SYMBOL).entry_price == FILL_PRICE
    booked = second.store.state.orders[FIRST_ORDER_ID].filled_quantity
    assert booked == Decimal("0.250")
    # Scale, not just value. "0.250" is what was persisted, and a filled quantity
    # reloaded through a float would come back as "0.25" while still comparing
    # equal to it — which is the one form of Decimal loss `==` cannot see.
    assert str(booked) == "0.250"
    # The venue filled all 0.500, so local and reported genuinely disagree, and
    # the store's copy is kept rather than being replaced by the venue's.
    assert report.outcome is ReconciliationOutcome.MISMATCH
    assert "local says LONG 0.250" in report.detail
    assert "the venue says LONG 0.500" in report.detail
    assert market.reported_position(SYMBOL).quantity == Decimal("0.500")
    assert (
        second.store.state.orders[FIRST_ORDER_ID].state is OrderState.RECONCILIATION_REQUIRED
    )


def test_a_dispute_outlives_both_the_restart_and_the_orders_that_caused_it(tmp_path):
    """Catches a disagreement that a restart, or a later agreement, forgets.

    The mark is on the *symbol* and it is persisted, so it still stops the next
    signal once every order involved is terminal and once local and reported
    have drifted back into agreement — neither of which is an explanation, and
    only an operator's explanation clears it.
    """
    path = state_path(tmp_path)
    market = build_venue()
    first = build_executor(path, market)
    first.recover({})
    events = submit_without_booking(first, "0.500", FIRST_ORDER_ID)
    first.apply_event(FIRST_ORDER_ID, events[0], REFERENCE)

    second = build_executor(path, market)
    report = second.recover(venue_view(market))
    assert report.outcome is ReconciliationOutcome.MISMATCH
    assert second.store.state.disputed == {SYMBOL: report.detail}

    # The fill is replayed, so the order goes terminal and the two sides agree
    # again; neither of those is a resolution.
    for event in events:
        second.apply_event(FIRST_ORDER_ID, event, REFERENCE)
    assert second.store.state.orders[FIRST_ORDER_ID].is_terminal is True

    third = build_executor(path, market)
    assert third.store.state.disputed == {SYMBOL: report.detail}
    assert third.recover(venue_view(market)).outcome is ReconciliationOutcome.AGREED
    assert third.store.state.disputed == {SYMBOL: report.detail}

    with pytest.raises(ReconciliationRequired) as excinfo:
        open_long(third)
    assert f"{SYMBOL} is disputed: local says FLAT 0" in str(excinfo.value)
    assert "Resolve it with resolve_reconciliation()" in str(excinfo.value)


# --- restart boundary 4: filled, completion not persisted ------------------
def test_events_replayed_after_a_restart_book_the_fill_exactly_once(tmp_path):
    """Catches a replay that doubles the position, or one that drops the fill.

    The crash lands between the venue's fill and the local booking of it, so the
    second process is handed every event again. The acknowledgement it already
    has must change nothing, and the fill it never saw must be booked in full.
    """
    path = state_path(tmp_path)
    market = build_venue()
    first = build_executor(path, market)
    first.recover({})
    events = submit_without_booking(first, "0.500", FIRST_ORDER_ID)

    assert [e.kind for e in events] == [EventKind.ACKNOWLEDGED, EventKind.FILL]
    # Only the acknowledgement is persisted; the process dies before the fill.
    first.apply_event(FIRST_ORDER_ID, events[0], REFERENCE)
    assert first.position(SYMBOL).side is PositionSide.FLAT

    second = build_executor(path, market)
    second.recover(venue_view(market))
    for event in events:
        second.apply_event(FIRST_ORDER_ID, event, REFERENCE)

    order = second.store.state.orders[FIRST_ORDER_ID]
    assert order.state is OrderState.FILLED
    assert order.filled_quantity == Decimal("0.500")
    assert order.applied_events == [f"{FIRST_ORDER_ID}:1", f"{FIRST_ORDER_ID}:2"]
    assert second.position(SYMBOL).quantity == Decimal("0.500")
    assert second.position(SYMBOL).entry_price == FILL_PRICE
    assert second.ledger.turnover == Decimal("30015")
    assert second.ledger.trading_fees == Decimal("15.00750000")
    assert second.position(SYMBOL).quantity == market.reported_position(SYMBOL).quantity


# --- restart boundary 5: recovering again ----------------------------------
def test_recovering_repeatedly_changes_nothing(tmp_path):
    """Catches a recovery whose second call differs from its first.

    Recovery is reached by an operator, a supervisor restart and a retry loop
    alike, so a second call that cancelled something, re-booked something or
    re-bootstrapped would be a sixth boundary nobody tested.
    """
    path = state_path(tmp_path)
    market = build_venue()
    first = build_executor(path, market)
    first.recover({})
    open_long(first)

    second = build_executor(path, market)
    second.recover(venue_view(market))
    snapshot = second.store.state.to_dict()
    on_disk = path.read_bytes()

    for _ in range(2):
        report = second.recover(venue_view(market))
        assert report.outcome is ReconciliationOutcome.AGREED
        assert second.store.state.to_dict() == snapshot
        assert path.read_bytes() == on_disk

    assert second.position(SYMBOL).quantity == Decimal("0.500")
    assert second.ledger.turnover == Decimal("30015")
    assert [o.state for o in second.store.state.orders.values()] == [OrderState.FILLED]


# --- what makes all of the above true --------------------------------------
def test_applied_event_ids_survive_the_restart_that_makes_redelivery_a_no_op(tmp_path):
    """Catches `applied_events` being dropped from the persisted record.

    Without it a venue that redelivers a fill after a restart books the exposure
    a second time — the one failure the idempotency key exists to prevent, and
    the one that only shows up across a process boundary.
    """
    path = state_path(tmp_path)
    market = build_venue()
    first = build_executor(path, market)
    first.recover({})
    open_long(first)
    fill_id = f"{FIRST_ORDER_ID}:2"
    assert first.store.state.orders[FIRST_ORDER_ID].applied_events[-1] == fill_id

    second = build_executor(path, market)
    second.recover(venue_view(market))
    assert second.store.state.orders[FIRST_ORDER_ID].applied_events == [
        f"{FIRST_ORDER_ID}:1",
        fill_id,
    ]

    # The venue redelivers the fill it already sent, with the same id.
    redelivered = OrderEvent(
        event_id=fill_id,
        kind=EventKind.FILL,
        quantity=Decimal("0.500"),
        price=FILL_PRICE,
        fee=Decimal("15.00750000"),
    )
    second.apply_event(FIRST_ORDER_ID, redelivered, REFERENCE)

    order = second.store.state.orders[FIRST_ORDER_ID]
    assert order.state is OrderState.FILLED
    assert order.filled_quantity == Decimal("0.500")
    assert order.fees == Decimal("15.00750000")
    assert order.applied_events == [f"{FIRST_ORDER_ID}:1", fill_id]
    assert second.position(SYMBOL).quantity == Decimal("0.500")
    assert second.ledger.turnover == Decimal("30015")
    assert second.ledger.trading_fees == Decimal("15.00750000")
