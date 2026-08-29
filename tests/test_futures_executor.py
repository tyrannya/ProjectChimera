"""The Pythia -> Aegis -> Hermes path in ``chimera.futures.executor``.

The claim this file exists to hold up is the one the module makes about itself:
there is no branch in which an order reaches the venue without
:meth:`chimera.risk.RiskEngine.evaluate_entry` having approved it. So the veto
tests do not merely check a return value — they hand the executor a venue whose
``submit`` raises, and let reaching it be the failure.

The other half is the mirror image: a *reduction* is deliberately not gated, and
:meth:`FuturesExecutor.emergency_flatten` is reachable while the account is
halted and while a reconciliation mismatch stands. A kill switch that also
blocked exits would be a trap, so those are tested as hard as the veto is.

Everything here runs in-process: an in-memory store, a dry-run venue, no
credentials, no environment and no network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

import pytest

from chimera.contracts import Signal
from chimera.futures import (
    ConstraintError,
    DeterministicFillModel,
    DryRunFuturesVenue,
    EventKind,
    FillPlan,
    FlattenCause,
    FundingEvent,
    FuturesError,
    FuturesExecutionConfig,
    FuturesExecutor,
    FuturesStore,
    InvalidTransition,
    LiveFuturesNotImplemented,
    NotBootstrapped,
    OrderEvent,
    OrderPurpose,
    OrderState,
    Position,
    PositionSide,
    ReconciliationOutcome,
    ReconciliationPolicy,
    ReconciliationRequired,
    TargetPosition,
    default_constraints_table,
    load_constraint_source,
)
from chimera.risk import RiskEngine, RiskLimits

SYMBOL = "BTC/USDT:USDT"
UNKNOWN_SYMBOL = "ETH/USDT:USDT"
PRICE = Decimal("60000")
EQUITY = 100_000.0

#: The committed BTCUSDT table with a lot size ten steps wide, so that a
#: quantity can be below ``min_quantity`` without also rounding to zero at the
#: step. With the real table the two refusals are the same order.
COARSE_LOT_TABLE = {SYMBOL: {**default_constraints_table()[SYMBOL], "min_quantity": "0.010"}}


def wide_limits(**overrides):
    """Limits loose enough that only the guard a test is about can veto."""
    return RiskLimits(**{"max_position_pct": 1.0, "risk_per_trade_pct": 0.5, **overrides})


def dry_run_venue(cls=DryRunFuturesVenue, *, fill_model=None, table=None):
    return cls(
        source=load_constraint_source(table),
        fill_model=fill_model or DeterministicFillModel(),
    )


def make_executor(*, venue=None, limits=None, config=None, fill_model=None, table=None):
    """A bootstrapped executor over a flat dry-run venue and an in-memory store."""
    risk = RiskEngine(limits if limits is not None else wide_limits())
    risk.update_equity(EQUITY)
    executor = FuturesExecutor(
        venue=venue or dry_run_venue(fill_model=fill_model, table=table),
        risk=risk,
        store=FuturesStore.open(None),
        config=config or FuturesExecutionConfig(),
    )
    executor.recover({})
    return executor


def move_to(executor, side, quantity, price=PRICE, **kwargs):
    """Ask for a target position, in the one form every test in this file wants."""
    target = TargetPosition(SYMBOL, side, Decimal(quantity))
    return executor.execute_target(target, price, equity=EQUITY, **kwargs)


@dataclass(frozen=True)
class HalfFillModel:
    """Fills half an order and stops, leaving it PARTIALLY_FILLED.

    A stand-in for the venue *simulator's* own injection point, not for anything
    under test. It is the only way to reach a position built by an order that is
    still open, which the over-fill and reconciliation tests both need.
    """

    def plan(self, intent, reference_price, constraints):
        half = constraints.quantize_quantity(intent.quantity / 2)
        return FillPlan(fills=((half, constraints.quantize_price(reference_price)),))


@dataclass
class ExplodingVenue(DryRunFuturesVenue):
    """A venue whose ``submit`` must be unreachable. Reaching it is the failure."""

    reached: list = field(default_factory=list)

    def submit(self, order_id, intent, reference_price):
        self.reached.append(order_id)
        raise AssertionError(
            f"submit() was reached for {order_id}: a vetoed order left the risk gate"
        )


@dataclass
class RecordingVenue(DryRunFuturesVenue):
    """Keeps every event it returned, so a test can redeliver the same objects."""

    delivered: list = field(default_factory=list)

    def submit(self, order_id, intent, reference_price):
        events = super().submit(order_id, intent, reference_price)
        self.delivered.append((order_id, list(events)))
        return events


# --- configuration ---------------------------------------------------------
def test_a_live_config_is_refused_and_says_there_is_no_live_path_to_enable():
    """Catches a config flag being read as a switch that turns live trading on."""
    with pytest.raises(LiveFuturesNotImplemented) as excinfo:
        FuturesExecutionConfig(dry_run=False)
    message = str(excinfo.value)
    assert "no live-order path" in message
    assert "no credential to supply and no endpoint to enable" in message


def test_leverage_other_than_one_is_refused_at_construction():
    """Catches a 3x config producing positions whose liquidation model is 1x."""
    with pytest.raises(FuturesError, match="leverage 3 is not 1"):
        FuturesExecutionConfig(leverage=Decimal("3"))


def test_cross_margin_is_refused_at_construction():
    """Catches cross margin being accepted by a model that only represents isolated."""
    with pytest.raises(FuturesError, match="is not ISOLATED"):
        FuturesExecutionConfig(margin_mode="CROSS")


def test_positions_the_executor_opens_are_isolated_at_one_times_leverage():
    """Catches a position recorded under a margin regime the config did not ask for."""
    executor = make_executor()
    move_to(executor, PositionSide.LONG, "0.5")

    position = executor.position(SYMBOL)
    assert position.leverage == executor.config.leverage == Decimal("1")
    assert position.margin_mode == executor.config.margin_mode == "ISOLATED"
    # The recorded regime is what the margin figures Aegis is handed are computed
    # from, so a position booked at another leverage is a different claim.
    assert executor.margin(SYMBOL, PRICE).initial_margin == Decimal("0.5") * PRICE


def test_an_unbootstrapped_executor_refuses_to_plan():
    """Catches an empty state file being read as a flat account."""
    executor = FuturesExecutor(
        venue=dry_run_venue(), risk=RiskEngine(wide_limits()), store=FuturesStore.open(None)
    )
    with pytest.raises(NotBootstrapped, match="An empty state file is not a flat account"):
        move_to(executor, PositionSide.LONG, "0.5")


# --- the signal path -------------------------------------------------------
def test_target_for_maps_every_signal_to_a_target_position():
    """HOLD means flat, not 'do nothing'; catches held exposure the model stopped asking for."""
    executor = make_executor()
    quantity = Decimal("0.5")

    assert executor.target_for(Signal.LONG, SYMBOL, quantity) == TargetPosition(
        SYMBOL, PositionSide.LONG, quantity
    )
    assert executor.target_for(Signal.SHORT, SYMBOL, quantity) == TargetPosition(
        SYMBOL, PositionSide.SHORT, quantity
    )
    hold = executor.target_for(Signal.HOLD, SYMBOL, quantity)
    assert hold == TargetPosition(SYMBOL, PositionSide.FLAT, Decimal("0"))


def test_the_long_lifecycle_opens_increases_reduces_and_closes():
    """Catches a leg planned with the wrong purpose, side or quantity."""
    executor = make_executor()

    (opened,) = move_to(executor, PositionSide.LONG, "0.5")
    assert opened.state is OrderState.FILLED
    assert opened.intent.purpose is OrderPurpose.OPEN
    assert opened.intent.reduce_only is False
    assert executor.position(SYMBOL).side is PositionSide.LONG
    assert executor.position(SYMBOL).quantity == Decimal("0.5")
    assert executor.position(SYMBOL).entry_price == Decimal("60030")

    (increased,) = move_to(executor, PositionSide.LONG, "0.8")
    assert increased.intent.purpose is OrderPurpose.INCREASE
    assert increased.intent.quantity == Decimal("0.3")
    assert executor.position(SYMBOL).quantity == Decimal("0.8")

    (reduced,) = move_to(executor, PositionSide.LONG, "0.3")
    assert reduced.intent.purpose is OrderPurpose.REDUCE
    assert reduced.intent.reduce_only is True
    assert reduced.intent.quantity == Decimal("0.5")
    assert executor.position(SYMBOL).quantity == Decimal("0.3")
    # A reduction realises PnL and does not re-price what is left.
    assert executor.ledger.realised_pnl == Decimal("-30")
    assert executor.position(SYMBOL).entry_price == Decimal("60030")

    (closed,) = move_to(executor, PositionSide.FLAT, "0")
    assert closed.intent.purpose is OrderPurpose.CLOSE
    assert closed.intent.quantity == Decimal("0.3")
    assert executor.position(SYMBOL).side is PositionSide.FLAT
    assert executor.position(SYMBOL).quantity == Decimal("0")


def test_the_short_lifecycle_opens_increases_reduces_and_closes():
    """The mirror of the LONG path; catches a side that only works one way round."""
    executor = make_executor()

    (opened,) = move_to(executor, PositionSide.SHORT, "0.5")
    assert opened.state is OrderState.FILLED
    assert opened.intent.purpose is OrderPurpose.OPEN
    assert executor.position(SYMBOL).side is PositionSide.SHORT
    assert executor.position(SYMBOL).quantity == Decimal("0.5")
    assert executor.position(SYMBOL).entry_price == Decimal("59970")

    (increased,) = move_to(executor, PositionSide.SHORT, "0.8")
    assert increased.intent.purpose is OrderPurpose.INCREASE
    assert increased.intent.quantity == Decimal("0.3")
    assert executor.position(SYMBOL).quantity == Decimal("0.8")
    assert executor.position(SYMBOL).side is PositionSide.SHORT

    (reduced,) = move_to(executor, PositionSide.SHORT, "0.3")
    assert reduced.intent.purpose is OrderPurpose.REDUCE
    assert reduced.intent.reduce_only is True
    assert reduced.intent.quantity == Decimal("0.5")
    assert executor.position(SYMBOL).quantity == Decimal("0.3")
    # A short loses when the buy-back price is above entry.
    assert executor.ledger.realised_pnl == Decimal("-30")

    (closed,) = move_to(executor, PositionSide.FLAT, "0")
    assert closed.intent.purpose is OrderPurpose.CLOSE
    assert closed.intent.quantity == Decimal("0.3")
    assert executor.position(SYMBOL).side is PositionSide.FLAT


def test_a_long_to_short_signal_closes_then_opens_and_lands_on_the_target_exactly():
    """Catches a reversal expressed as one order of current + target."""
    executor = make_executor()
    move_to(executor, PositionSide.LONG, "0.5")

    closed, opened = move_to(executor, PositionSide.SHORT, "0.4")
    assert (closed.state, opened.state) == (OrderState.FILLED, OrderState.FILLED)
    assert closed.intent.purpose is OrderPurpose.CLOSE
    assert closed.intent.quantity == Decimal("0.5")
    assert closed.intent.reduce_only is True
    assert opened.intent.purpose is OrderPurpose.OPEN
    assert opened.intent.quantity == Decimal("0.4")

    position = executor.position(SYMBOL)
    assert position.side is PositionSide.SHORT
    assert position.quantity == Decimal("0.4")  # the target, never current + target


def test_a_target_equal_to_the_current_position_plans_nothing():
    """Catches a repeated signal churning fees on an order that changes nothing."""
    executor = make_executor()
    move_to(executor, PositionSide.LONG, "0.5")
    fees = executor.ledger.trading_fees

    assert move_to(executor, PositionSide.LONG, "0.5") == []
    assert executor.ledger.trading_fees == fees


# --- the Aegis gate --------------------------------------------------------
def test_a_halted_risk_engine_makes_submission_impossible():
    """The kill switch. Catches any path from a vetoed order to the venue."""
    executor = make_executor(venue=dry_run_venue(ExplodingVenue))
    executor.risk.halt("operator pulled the kill switch")

    (record,) = move_to(executor, PositionSide.LONG, "0.5")

    assert record.state is OrderState.REJECTED
    assert record.reason == "halted: operator pulled the kill switch"
    assert executor.position(SYMBOL).side is PositionSide.FLAT
    assert executor.venue.reached == []


def test_an_over_exposure_veto_makes_submission_impossible():
    """The same guarantee for a limit rather than the halt: no order, no venue call."""
    executor = make_executor(
        venue=dry_run_venue(ExplodingVenue),
        limits=wide_limits(max_total_exposure_pct=0.001),
    )

    (record,) = move_to(executor, PositionSide.LONG, "0.5")

    assert record.state is OrderState.REJECTED
    assert record.reason.startswith("total exposure ")
    assert "would exceed 100.00" in record.reason
    assert executor.position(SYMBOL).side is PositionSide.FLAT
    assert executor.venue.reached == []


def test_the_default_risk_limits_veto_an_open_above_the_risk_envelope():
    """Catches the gate being widened by accident: stock limits refuse a 30k stake."""
    executor = make_executor(venue=dry_run_venue(ExplodingVenue), limits=RiskLimits())

    (record,) = move_to(executor, PositionSide.LONG, "0.5")

    assert record.state is OrderState.REJECTED
    assert record.reason.startswith("order stake 30000.00 exceeds the risk-based maximum")
    assert executor.venue.reached == []


def test_a_vetoed_open_after_a_close_abandons_the_rest_of_a_reversal():
    """A reversal whose open is refused must stop flat, not half reversed."""
    executor = make_executor()
    move_to(executor, PositionSide.LONG, "0.5")
    executor.risk.halt("halted between the two legs")

    closed, opened = move_to(executor, PositionSide.SHORT, "0.4")
    assert closed.state is OrderState.FILLED
    assert opened.state is OrderState.REJECTED
    assert executor.position(SYMBOL).side is PositionSide.FLAT


def test_a_reducing_order_is_not_gated_by_the_kill_switch():
    """A halted account must still be able to close. Catches a kill switch that traps."""
    executor = make_executor()
    move_to(executor, PositionSide.LONG, "0.5")
    executor.risk.halt("operator pulled the kill switch")

    (record,) = move_to(executor, PositionSide.FLAT, "0")

    assert record.intent.purpose is OrderPurpose.CLOSE
    assert record.state is OrderState.FILLED
    assert executor.position(SYMBOL).side is PositionSide.FLAT
    assert executor.position(SYMBOL).quantity == Decimal("0")


# --- emergency flatten -----------------------------------------------------
def test_emergency_flatten_reaches_zero_from_a_long():
    """Catches a flatten that overshoots into the opposite side."""
    executor = make_executor()
    move_to(executor, PositionSide.LONG, "0.5")

    record = executor.emergency_flatten(SYMBOL, FlattenCause.OPERATOR, PRICE)

    assert record.state is OrderState.FILLED
    assert record.intent.purpose is OrderPurpose.FLATTEN
    assert record.intent.reduce_only is True
    assert record.intent.quantity == Decimal("0.5")
    assert executor.position(SYMBOL).side is PositionSide.FLAT
    assert executor.position(SYMBOL).quantity == Decimal("0")


def test_emergency_flatten_reaches_zero_from_a_short():
    """The mirror case; a SHORT must flatten to FLAT and never to LONG."""
    executor = make_executor()
    move_to(executor, PositionSide.SHORT, "0.5")

    record = executor.emergency_flatten(SYMBOL, FlattenCause.SHUTDOWN, PRICE)

    assert record.state is OrderState.FILLED
    assert record.intent.quantity == Decimal("0.5")
    assert executor.position(SYMBOL).side is PositionSide.FLAT
    assert executor.position(SYMBOL).quantity == Decimal("0")


def test_emergency_flatten_reaches_zero_from_a_partially_filled_position():
    """Catches a flatten sized from the order rather than from the position."""
    executor = make_executor(fill_model=HalfFillModel())
    (partial,) = move_to(executor, PositionSide.LONG, "0.5")
    assert partial.state is OrderState.PARTIALLY_FILLED
    assert executor.position(SYMBOL).quantity == Decimal("0.25")

    executor.venue.fill_model = DeterministicFillModel()
    record = executor.emergency_flatten(SYMBOL, FlattenCause.DATA_LOSS, PRICE)

    assert record.intent.quantity == Decimal("0.25")
    assert record.state is OrderState.FILLED
    assert executor.position(SYMBOL).side is PositionSide.FLAT


def test_emergency_flatten_on_a_flat_position_records_the_reason_and_orders_nothing():
    """'We tried to flatten and there was nothing there' has to reach the log."""
    executor = make_executor()

    assert executor.emergency_flatten(SYMBOL, FlattenCause.SHUTDOWN, PRICE) is None
    assert executor.store.state.orders == {}
    assert executor.store.state.flatten_reasons[-1]["reason"] == "SHUTDOWN"
    assert executor.store.state.flatten_reasons[-1]["symbol"] == SYMBOL


def test_repeated_emergency_flatten_is_safe():
    """Catches a second flatten opening the opposite side of an already flat account."""
    executor = make_executor()
    move_to(executor, PositionSide.LONG, "0.5")

    assert executor.emergency_flatten(SYMBOL, FlattenCause.OPERATOR, PRICE) is not None
    assert executor.emergency_flatten(SYMBOL, FlattenCause.OPERATOR, PRICE) is None
    assert executor.emergency_flatten(SYMBOL, FlattenCause.OPERATOR, PRICE) is None

    assert executor.position(SYMBOL).side is PositionSide.FLAT
    assert len(executor.store.state.flatten_reasons) == 3


def test_emergency_flatten_works_while_the_risk_engine_is_halted():
    """The situation flatten exists for. Catches the exit gated behind the entry gate."""
    executor = make_executor()
    move_to(executor, PositionSide.LONG, "0.5")
    executor.risk.halt("drawdown breach")

    record = executor.emergency_flatten(SYMBOL, FlattenCause.RISK_HALT, PRICE)

    assert record.state is OrderState.FILLED
    assert executor.position(SYMBOL).side is PositionSide.FLAT


def test_emergency_flatten_works_while_a_reconciliation_mismatch_stands():
    """A disputed symbol refuses new targets but must still be reducible to zero."""
    executor = make_executor()
    move_to(executor, PositionSide.LONG, "0.5")
    executor.venue.apply_settlement(
        SYMBOL, Position(SYMBOL, PositionSide.LONG, Decimal("2"), Decimal("59000"))
    )
    assert executor.reconcile(SYMBOL).outcome is ReconciliationOutcome.MISMATCH

    with pytest.raises(ReconciliationRequired):
        move_to(executor, PositionSide.FLAT, "0")

    record = executor.emergency_flatten(SYMBOL, FlattenCause.RECONCILIATION_MISMATCH, PRICE)
    assert record.state is OrderState.FILLED
    assert executor.position(SYMBOL).side is PositionSide.FLAT


# --- venue refusals --------------------------------------------------------
def test_an_open_below_min_notional_is_rejected_rather_than_raised():
    """A venue refusal is an order outcome, not an exception the caller must catch."""
    executor = make_executor()

    (record,) = move_to(executor, PositionSide.LONG, "0.001")

    assert record.state is OrderState.REJECTED
    assert "below min_notional 100" in record.reason
    assert executor.position(SYMBOL).side is PositionSide.FLAT


def test_a_quantity_below_min_quantity_is_rejected_rather_than_raised():
    """Catches an order the venue would refuse being sent anyway."""
    executor = make_executor(table=COARSE_LOT_TABLE)

    (record,) = move_to(executor, PositionSide.LONG, "0.005")

    assert record.state is OrderState.REJECTED
    assert "below min_quantity 0.010" in record.reason
    assert executor.position(SYMBOL).side is PositionSide.FLAT


def test_a_quantity_that_rounds_to_zero_at_the_step_is_rejected():
    """Catches a dust order becoming a zero-quantity order at the venue."""
    executor = make_executor()

    (record,) = move_to(executor, PositionSide.LONG, "0.0004")

    assert record.state is OrderState.REJECTED
    assert record.reason == "quantity 0.0004 rounds to zero at step 0.001"
    assert executor.position(SYMBOL).side is PositionSide.FLAT


def test_a_symbol_with_no_venue_metadata_fails_closed():
    """Catches a missing tick or step size being defaulted into a guessed order size."""
    executor = make_executor()

    with pytest.raises(ConstraintError, match="no venue metadata for"):
        executor.execute_target(
            TargetPosition(UNKNOWN_SYMBOL, PositionSide.LONG, Decimal("0.5")),
            PRICE,
            equity=EQUITY,
        )


# --- events ----------------------------------------------------------------
def test_duplicate_events_do_not_duplicate_exposure():
    """A redelivered fill must change nothing at all: no quantity, no fee, no PnL."""
    executor = make_executor(venue=dry_run_venue(RecordingVenue))
    (opened,) = move_to(executor, PositionSide.LONG, "0.5")
    (reduced,) = move_to(executor, PositionSide.LONG, "0.25")

    position = executor.position(SYMBOL)
    assert position.quantity == Decimal("0.25")
    assert executor.ledger.turnover == Decimal("45007.5")
    assert executor.ledger.realised_pnl == Decimal("-15")
    assert executor.ledger.trading_fees == Decimal("22.50375")

    redelivered = 0
    for order_id, events in executor.venue.delivered:
        for event in events:
            executor.apply_event(order_id, event, PRICE)
            redelivered += 1
    assert redelivered == 4  # an acknowledgement and a fill for each of the two orders

    assert executor.position(SYMBOL) == position
    assert opened.filled_quantity == Decimal("0.5")
    assert opened.fees == Decimal("15.0075")
    assert reduced.filled_quantity == Decimal("0.25")
    assert reduced.fees == Decimal("7.49625")
    assert executor.ledger.turnover == Decimal("45007.5")
    assert executor.ledger.realised_pnl == Decimal("-15")
    assert executor.ledger.trading_fees == Decimal("22.50375")


def test_an_event_for_an_unknown_order_raises():
    """Catches a stray fill being booked against a position it does not belong to."""
    executor = make_executor()
    event = OrderEvent(
        event_id="stray-1",
        kind=EventKind.FILL,
        quantity=Decimal("0.5"),
        price=PRICE,
    )

    with pytest.raises(FuturesError, match="no such order 'BTCUSDTUSDT-999999'"):
        executor.apply_event("BTCUSDTUSDT-999999", event, PRICE)


def test_an_acknowledgement_after_a_fill_raises_invalid_transition():
    """An out-of-order event is raised, not clamped, logged or ignored."""
    executor = make_executor()
    (record,) = move_to(executor, PositionSide.LONG, "0.5")
    assert record.state is OrderState.FILLED

    late_ack = OrderEvent(event_id="late-ack", kind=EventKind.ACKNOWLEDGED)
    with pytest.raises(InvalidTransition, match="FILLED -> ACKNOWLEDGED"):
        executor.apply_event(record.order_id, late_ack, PRICE)

    assert record.state is OrderState.FILLED
    assert executor.position(SYMBOL).quantity == Decimal("0.5")


def test_a_fill_larger_than_the_outstanding_quantity_demands_reconciliation():
    """An over-fill is a disagreement about the account, not exposure to book."""
    executor = make_executor(fill_model=HalfFillModel())
    (record,) = move_to(executor, PositionSide.LONG, "0.5")
    assert record.remaining_quantity == Decimal("0.25")

    oversized = OrderEvent(
        event_id="oversized-fill",
        kind=EventKind.FILL,
        quantity=Decimal("0.5"),
        price=PRICE,
    )
    executor.apply_event(record.order_id, oversized, PRICE)

    assert record.state is OrderState.RECONCILIATION_REQUIRED
    assert "exceeds the 0.25 outstanding" in record.reason
    assert record.filled_quantity == Decimal("0.25")
    assert executor.position(SYMBOL).quantity == Decimal("0.25")


# --- reconciliation --------------------------------------------------------
def test_reconcile_agrees_when_local_and_the_venue_hold_the_same_position():
    """Catches a comparison that reports a mismatch against itself."""
    executor = make_executor()
    move_to(executor, PositionSide.LONG, "0.5")
    before = executor.position(SYMBOL)

    report = executor.reconcile(SYMBOL)

    assert report.outcome is ReconciliationOutcome.AGREED
    assert report.agrees is True
    assert report.local == before
    assert report.reported == before
    assert executor.position(SYMBOL) == before
    assert executor.store.state.disputed == {}


def test_a_mismatch_moves_open_orders_aside_and_never_overwrites_the_local_position():
    """The silent overwrite this package refuses: local must survive a disagreement."""
    executor = make_executor(fill_model=HalfFillModel())
    (order,) = move_to(executor, PositionSide.LONG, "0.5")
    local = executor.position(SYMBOL)
    assert local.quantity == Decimal("0.25")

    executor.venue.apply_settlement(
        SYMBOL, Position(SYMBOL, PositionSide.SHORT, Decimal("2"), Decimal("50000"))
    )
    report = executor.reconcile(SYMBOL)

    assert report.outcome is ReconciliationOutcome.MISMATCH
    assert report.detail == "local says LONG 0.25, the venue says SHORT 2"
    assert report.local == local
    assert report.reported.side is PositionSide.SHORT
    assert order.state is OrderState.RECONCILIATION_REQUIRED
    # The whole point: the local position is still the local one.
    assert executor.position(SYMBOL) == local
    assert executor.position(SYMBOL).side is PositionSide.LONG


def test_execute_target_refuses_while_a_mismatch_stands():
    """Catches a new order being sized against a position nobody can vouch for."""
    executor = make_executor()
    move_to(executor, PositionSide.LONG, "0.5")
    executor.venue.apply_settlement(
        SYMBOL, Position(SYMBOL, PositionSide.SHORT, Decimal("2"), Decimal("50000"))
    )
    executor.reconcile(SYMBOL)

    with pytest.raises(ReconciliationRequired, match="is disputed"):
        move_to(executor, PositionSide.LONG, "0.8")

    assert executor.position(SYMBOL).quantity == Decimal("0.5")


def test_resolve_reconciliation_clears_the_dispute_and_lets_trading_resume():
    """An operator's written decision is the only way out of a dispute."""
    executor = make_executor()
    move_to(executor, PositionSide.LONG, "0.5")
    adopted = Position(SYMBOL, PositionSide.SHORT, Decimal("2"), Decimal("50000"))
    executor.venue.apply_settlement(SYMBOL, adopted)
    executor.reconcile(SYMBOL)

    executor.resolve_reconciliation(SYMBOL, adopted, "operator confirmed the venue by hand")

    assert executor.store.state.disputed == {}
    assert executor.position(SYMBOL) == adopted

    (closed,) = move_to(executor, PositionSide.FLAT, "0")
    assert closed.state is OrderState.FILLED
    assert closed.intent.quantity == Decimal("2")
    assert executor.position(SYMBOL).side is PositionSide.FLAT


def test_resolving_a_mismatch_without_a_stated_reason_is_refused():
    """Automatic resolution is the thing the disputed state exists to prevent."""
    executor = make_executor()
    move_to(executor, PositionSide.LONG, "0.5")
    executor.venue.apply_settlement(
        SYMBOL, Position(SYMBOL, PositionSide.SHORT, Decimal("2"), Decimal("50000"))
    )
    executor.reconcile(SYMBOL)

    with pytest.raises(FuturesError, match="requires a stated reason"):
        executor.resolve_reconciliation(SYMBOL, executor.position(SYMBOL), "")

    assert SYMBOL in executor.store.state.disputed
    with pytest.raises(ReconciliationRequired):
        move_to(executor, PositionSide.LONG, "0.8")


def test_the_flatten_policy_flattens_locally_and_still_refuses_to_trade():
    """Flattening resolves the exposure, not the disagreement."""
    executor = make_executor(
        config=FuturesExecutionConfig(reconciliation_policy=ReconciliationPolicy.FLATTEN)
    )
    move_to(executor, PositionSide.LONG, "0.5")
    executor.venue.apply_settlement(
        SYMBOL, Position(SYMBOL, PositionSide.LONG, Decimal("2"), Decimal("59000"))
    )

    report = executor.reconcile(SYMBOL)

    assert report.outcome is ReconciliationOutcome.MISMATCH
    assert executor.position(SYMBOL).side is PositionSide.FLAT
    assert executor.store.state.flatten_reasons[-1]["reason"] == "RECONCILIATION_MISMATCH"
    with pytest.raises(ReconciliationRequired, match="is disputed"):
        move_to(executor, PositionSide.LONG, "0.5")


# --- funding ---------------------------------------------------------------
def test_funding_is_booked_once_and_a_repeat_settlement_returns_zero():
    """Catches a redelivered settlement charging the account twice."""
    executor = make_executor()
    move_to(executor, PositionSide.LONG, "0.5")
    event = FundingEvent(SYMBOL, Decimal("0.0001"), PRICE, "settlement-1")

    assert executor.settle_funding(event) == Decimal("-3")
    assert executor.ledger.funding_paid == Decimal("3")

    assert executor.settle_funding(event) == Decimal("0")
    assert executor.ledger.funding_paid == Decimal("3")


@pytest.mark.parametrize(
    "side, rate, expected",
    [
        (PositionSide.LONG, "0.0001", Decimal("-3")),
        (PositionSide.LONG, "-0.0001", Decimal("3")),
        (PositionSide.SHORT, "0.0001", Decimal("3")),
        (PositionSide.SHORT, "-0.0001", Decimal("-3")),
    ],
)
def test_the_funding_sign_follows_the_side_and_rate_table(side, rate, expected):
    """Longs pay a positive rate and shorts receive it. Catches a flipped sign."""
    executor = make_executor()
    move_to(executor, side, "0.5")

    flow = executor.settle_funding(FundingEvent(SYMBOL, Decimal(rate), PRICE, "settlement-1"))

    assert flow == expected


def test_funding_on_a_flat_position_is_zero():
    """A position that does not exist neither pays nor receives."""
    executor = make_executor()

    event = FundingEvent(SYMBOL, Decimal("0.0001"), PRICE, "settlement-1")
    assert executor.settle_funding(event) == Decimal("0")
    assert executor.ledger.funding_paid == Decimal("0")
    assert executor.ledger.funding_received == Decimal("0")


# --- margin ----------------------------------------------------------------
def test_margin_is_none_when_flat_and_reports_one_times_leverage_when_open():
    """A flat position has no liquidation price; a zeroed record would read as 100% away."""
    executor = make_executor()
    assert executor.margin(SYMBOL, PRICE) is None

    move_to(executor, PositionSide.LONG, "0.5")
    state = executor.margin(SYMBOL, PRICE)

    assert state.side is PositionSide.LONG
    assert state.leverage == Decimal("1")
    assert state.initial_margin == Decimal("30000")
    assert state.maintenance_margin == Decimal("120")
    assert state.liquidation_price == Decimal("240.12")


# --- regressions on two defects this suite found ---------------------------
# Both were real when the tests were written and both are fixed. They are kept
# because the fixes are one line each and the symptoms are silent: an Aegis
# approval describing a quantity nobody sent, and a restart that disputed only
# the alphabetically-first disagreeing symbol.
def test_aegis_is_asked_about_the_quantity_the_venue_will_actually_receive():
    """The approved stake and the sent order must describe the same quantity.

    0.5004 BTC is not on the 0.001 lot step; ``_plan`` rounds it down to 0.500,
    a 30000 stake. Aegis is nonetheless asked about 30024, so a ceiling drawn
    between the two refuses an order that would have been inside it. The
    direction is conservative — ``quantize_quantity`` rounds down, so this can
    only over-reject — but the approval, the recorded stake and the prospective
    liquidation figure all describe a quantity that was never sent.
    """
    ceiling = wide_limits(max_total_exposure_pct=0.3001)  # 30010 on 100k equity

    control = make_executor(limits=ceiling)
    (accepted,) = move_to(control, PositionSide.LONG, "0.500")
    assert accepted.state is OrderState.FILLED, "the control stake is inside the ceiling"

    executor = make_executor(limits=ceiling)
    (record,) = move_to(executor, PositionSide.LONG, "0.5004")

    assert record.intent.quantity == Decimal("0.500")
    assert record.state is OrderState.FILLED, record.reason


def test_recover_reconciles_every_symbol_not_only_the_first_disagreement():
    """A restart must dispute every symbol that disagrees, not the first one.

    ``recover`` sorts the symbols and stops at the first mismatch, so on a
    multi-symbol account only the alphabetically-first disagreeing symbol is
    marked disputed. The rest pass ``require_ready`` and would be traded against
    state nobody verified — the exact thing the disputed mark exists to prevent.
    """
    executor = FuturesExecutor(
        venue=dry_run_venue(), risk=RiskEngine(wide_limits()), store=FuturesStore.open(None)
    )
    # Adopted at bootstrap: there is no local state to contradict them.
    executor.recover(
        {
            SYMBOL: Position(SYMBOL, PositionSide.LONG, Decimal("0.5"), PRICE),
            UNKNOWN_SYMBOL: Position(UNKNOWN_SYMBOL, PositionSide.LONG, Decimal("1"), PRICE),
        }
    )

    # A restart against a venue that now reports flat: both symbols disagree.
    executor.recover({})

    assert set(executor.store.state.disputed) == {SYMBOL, UNKNOWN_SYMBOL}
