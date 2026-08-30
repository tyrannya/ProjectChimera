"""Eight defects the adversarial audit found, and the guards that closed them.

The audit ran six independent perspectives over the branch and verified every
finding adversarially before reporting it. What survived was one blocker and
seven majors, all in the execution layer, and every one of them silent: the
package did the wrong thing and said nothing.

They are kept together rather than folded into the per-module files because they
share a shape worth seeing at once. Six of the eight are a *guarantee stated in a
docstring that the code did not actually provide* — an id that survives a restart,
an idempotency key that is only recorded on success, a `reduce_only` flag the
venue never read, a flatten that reported itself done, a risk limit that read a
map nothing wrote. The remaining two are a default argument that meant the
opposite of the same default on the neighbouring method, and a guard that was
skipped on the one path it existed for.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from chimera.futures import (
    DeterministicFillModel,
    DryRunFuturesVenue,
    EventKind,
    FlattenCause,
    FuturesError,
    FuturesExecutionConfig,
    FuturesExecutor,
    FuturesStore,
    InvalidTransition,
    OrderEvent,
    OrderState,
    Position,
    PositionSide,
    ReconciliationOutcome,
    ReconciliationPolicy,
    TargetPosition,
    load_constraint_source,
)
from chimera.risk import RiskEngine, RiskLimits

SYMBOL = "BTC/USDT:USDT"
PRICE = Decimal("60000")


def limits(**overrides) -> RiskLimits:
    return RiskLimits(
        **{
            "max_position_pct": 1.0,
            "risk_per_trade_pct": 0.5,
            "max_total_exposure_pct": 10.0,
            "max_exposure_per_asset_pct": 10.0,
            **overrides,
        }
    )


def build(store: FuturesStore, *, risk_limits=None, fill_model=None, policy=None, venue=None):
    engine = RiskEngine(risk_limits if risk_limits is not None else limits())
    engine.update_equity(1_000_000.0)
    market = venue or DryRunFuturesVenue(
        source=load_constraint_source(), fill_model=fill_model or DeterministicFillModel()
    )
    config = (
        FuturesExecutionConfig(reconciliation_policy=policy)
        if policy is not None
        else FuturesExecutionConfig()
    )
    return (
        FuturesExecutor(venue=market, risk=engine, store=store, config=config),
        market,
        engine,
    )


def opened(executor, side=PositionSide.LONG, qty="0.5", price=PRICE):
    executor.execute_target(
        TargetPosition(SYMBOL, side, Decimal(qty)), price, equity=1_000_000.0
    )
    return executor.position(SYMBOL)


# --- 1. the blocker: order ids must survive a restart ------------------------


def test_a_restart_does_not_reuse_an_order_id_and_overwrite_the_persisted_record(tmp_path):
    """The one boundary the store exists to survive was the one that broke it.

    `_order_seq` started at zero in every process and `_plan` writes
    `store.state.orders[order_id]`, so the first order after a restart replaced
    the first order before it — its fills, its fees, its history, and its
    `applied_events`, which is the whole of the idempotency guarantee.
    """
    path = tmp_path / "state.json"
    first, venue, _ = build(FuturesStore.open(path))
    first.recover({})
    opened(first)
    before = dict(first.store.state.orders)
    assert len(before) == 1
    original_id = next(iter(before))

    second, _, _ = build(FuturesStore.open(path), venue=venue)
    second.recover({SYMBOL: venue.reported_position(SYMBOL)})
    second.execute_target(
        TargetPosition(SYMBOL, PositionSide.LONG, Decimal("0.8")), PRICE, equity=1_000_000.0
    )

    assert original_id in second.store.state.orders, "the pre-restart order was erased"
    survivor = second.store.state.orders[original_id]
    assert survivor.to_dict() == before[original_id].to_dict()
    assert len(second.store.state.orders) == 2, "the new order took the old one's id"


def test_the_counter_is_seeded_even_when_the_store_holds_many_orders(tmp_path):
    path = tmp_path / "state.json"
    first, venue, _ = build(FuturesStore.open(path))
    first.recover({})
    for quantity in ("0.1", "0.2", "0.3"):
        first.execute_target(
            TargetPosition(SYMBOL, PositionSide.LONG, Decimal(quantity)),
            PRICE,
            equity=1_000_000.0,
        )
    ids = set(first.store.state.orders)

    second, _, _ = build(FuturesStore.open(path), venue=venue)
    assert second._order_seq == len(ids)
    second.recover({SYMBOL: venue.reported_position(SYMBOL)})
    second.execute_target(TargetPosition.flat(SYMBOL), PRICE, equity=1_000_000.0)
    assert ids < set(second.store.state.orders)


# --- 2. Aegis's cumulative limits must not be inert --------------------------


def test_the_executor_reports_exposure_so_the_cumulative_limits_can_bite():
    """Three limits read `RiskState.open_positions`, and nothing was writing it.

    `evaluate_entry` still bounded each *individual* order, so the gap only
    showed on a position built in steps — which is exactly how a position gets
    built.
    """
    executor, _, risk = build(FuturesStore.open(None))
    executor.recover({})
    assert risk.state.open_positions == {}

    opened(executor, qty="0.5")
    assert risk.state.open_positions[SYMBOL] == pytest.approx(0.5 * 60030.0, rel=1e-9)

    executor.execute_target(TargetPosition.flat(SYMBOL), PRICE, equity=1_000_000.0)
    assert SYMBOL not in risk.state.open_positions


def test_a_position_built_in_steps_is_measured_as_a_whole():
    """The failure the dead limit hid: each leg was inside the envelope, the sum was not."""
    capped = limits(max_exposure_per_asset_pct=0.04)  # 40,000 on 1,000,000 equity
    executor, _, _ = build(FuturesStore.open(None), risk_limits=capped)
    executor.recover({})

    opened(executor, qty="0.5")  # ~30,000
    assert executor.position(SYMBOL).quantity == Decimal("0.5")

    executor.execute_target(
        TargetPosition(SYMBOL, PositionSide.LONG, Decimal("0.9")), PRICE, equity=1_000_000.0
    )
    assert executor.position(SYMBOL).quantity == Decimal(
        "0.5"
    ), "the second leg would take the position past the per-asset cap and must be vetoed"


# --- 3. nothing may be booked before the last thing that can raise -----------


def test_a_refused_transition_books_nothing_and_stays_refusable():
    """Booking the fill first left the exposure moved and the event id unrecorded.

    A redelivery of that same event was then not recognised as a duplicate and
    moved the position again — unboundedly.
    """
    executor, venue, _ = build(FuturesStore.open(None))
    executor.recover({})
    opened(executor, qty="0.5")
    order_id, record = next(iter(executor.store.state.orders.items()))
    assert record.state is OrderState.FILLED

    before = (executor.position(SYMBOL).to_dict(), executor.ledger.to_dict())
    late = OrderEvent(
        event_id="late-fill",
        kind=EventKind.FILL,
        quantity=Decimal("0.001"),
        price=PRICE,
        fee=Decimal("1"),
    )
    for _ in range(3):
        with pytest.raises((InvalidTransition, Exception)):
            executor.apply_event(order_id, late, PRICE)
        assert (executor.position(SYMBOL).to_dict(), executor.ledger.to_dict()) == before


# --- 4. the venue must honour reduce_only ------------------------------------


def test_a_reduce_only_order_cannot_open_a_position_at_the_venue():
    """`reduce_only` is the venue-level restatement of the local invariant.

    The only venue in the package did not model it, so whenever the two views
    differed a close opened the opposite exposure *there* — visible only to a
    later reconcile.
    """
    executor, venue, _ = build(FuturesStore.open(None))
    executor.recover({})
    opened(executor, qty="0.5")

    venue.apply_settlement(SYMBOL, Position(SYMBOL))  # the venue now holds nothing
    record = executor.emergency_flatten(SYMBOL, FlattenCause.OPERATOR, PRICE)

    assert venue.reported_position(
        SYMBOL
    ).is_flat, "a reduce-only flatten opened a SHORT at the venue"
    assert record is not None and record.state is OrderState.REJECTED


def test_a_reduce_only_order_larger_than_the_venue_holds_is_refused():
    executor, venue, _ = build(FuturesStore.open(None))
    executor.recover({})
    opened(executor, qty="0.5")
    venue.apply_settlement(SYMBOL, Position(SYMBOL, PositionSide.LONG, Decimal("0.2"), PRICE))
    executor.emergency_flatten(SYMBOL, FlattenCause.OPERATOR, PRICE)
    assert venue.reported_position(SYMBOL).quantity == Decimal("0.2")
    assert venue.reported_position(SYMBOL).side is PositionSide.LONG


# --- 5. a flatten that did not reach zero must not report success ------------


def test_a_flatten_that_could_not_reach_zero_disputes_the_symbol():
    """The order is planned from local and applied to the venue's own position.

    When the venue holds less, `Position.apply_fill` refuses, `_submit` turns
    that into FAILED and returns normally — so without this the position simply
    survived a flatten that logged itself done.
    """
    executor, venue, _ = build(FuturesStore.open(None))
    executor.recover({})
    opened(executor, qty="0.5")
    venue.apply_settlement(SYMBOL, Position(SYMBOL, PositionSide.LONG, Decimal("0.2"), PRICE))

    executor.emergency_flatten(SYMBOL, FlattenCause.RECONCILIATION_MISMATCH, PRICE)
    assert not executor.position(SYMBOL).is_flat
    assert SYMBOL in executor.store.state.disputed
    assert "did not reach zero" in executor.store.state.disputed[SYMBOL]


def test_the_flatten_policy_does_not_report_a_flatten_that_did_not_happen():
    """Worst under FLATTEN, whose whole trigger is that the quantities differ."""
    executor, venue, _ = build(FuturesStore.open(None), policy=ReconciliationPolicy.FLATTEN)
    executor.recover({})
    opened(executor, qty="0.5")
    venue.apply_settlement(SYMBOL, Position(SYMBOL, PositionSide.LONG, Decimal("0.2"), PRICE))
    report = executor.reconcile(SYMBOL, mark_price=PRICE)
    assert report.outcome is ReconciliationOutcome.MISMATCH
    assert not executor.position(SYMBOL).is_flat
    assert "did not reach zero" in executor.store.state.disputed[SYMBOL]


# --- 6. recover() must be told what the venue reported -----------------------


def test_recover_requires_the_reported_positions():
    """`reconcile(symbol)` reads None as 'ask the venue'; `recover()` read it as
    'the venue holds nothing'. The same word meant opposite things on one class,
    and the reading that costs money was the shorter spelling."""
    executor, _, _ = build(FuturesStore.open(None))
    with pytest.raises(TypeError):
        executor.recover()


def test_recover_with_an_explicit_empty_mapping_still_means_flat():
    executor, _, _ = build(FuturesStore.open(None))
    executor.recover({})
    assert executor.store.state.bootstrapped is True
    assert executor.position(SYMBOL).is_flat


# --- 7. flatten must refuse when the account is unknown ----------------------


def test_flatten_refuses_on_an_unbootstrapped_executor(tmp_path):
    """`FlattenCause.DATA_LOSS` exists for exactly the case that did nothing.

    `FuturesState.position` returns FLAT for a symbol it has never heard of, so
    an executor that had adopted nothing concluded the account was flat, recorded
    a flatten reason, and returned — while the venue still held the position.
    """
    path = tmp_path / "state.json"
    path.write_text("{not json")
    original = path.read_bytes()
    store = FuturesStore.open(path)
    executor, _, _ = build(store)
    executor.recover({})

    with pytest.raises(Exception) as excinfo:
        executor.emergency_flatten(SYMBOL, FlattenCause.DATA_LOSS, PRICE)
    assert type(excinfo.value).__name__ == "NotBootstrapped"
    assert store.state.flatten_reasons == [], "a refused flatten recorded a reason"
    assert path.read_bytes() == original, "the unreadable file was overwritten"


# --- 8. the margin regime must hold for the position, not only the config ----


def test_a_position_held_off_regime_stops_planning():
    """The config check is about what this process will open.

    A position adopted at bootstrap can carry another regime, and `margin_state`
    would then hand Aegis a liquidation price computed for leverage this package
    does not support.
    """
    executor, _, _ = build(FuturesStore.open(None))
    executor.recover(
        {
            SYMBOL: Position(
                SYMBOL, PositionSide.LONG, Decimal("0.5"), PRICE, leverage=Decimal("3")
            )
        }
    )
    with pytest.raises(FuturesError, match="3x"):
        executor.execute_target(
            TargetPosition(SYMBOL, PositionSide.LONG, Decimal("0.8")),
            PRICE,
            equity=1_000_000.0,
        )


# --- the smaller ones -------------------------------------------------------


def test_the_order_that_trips_the_rate_limit_is_itself_refused():
    """`record_order` can trip the halt, and the order that trips it is this one."""
    executor, venue, risk = build(
        FuturesStore.open(None), risk_limits=limits(max_orders_per_minute=1)
    )
    executor.recover({})
    opened(executor, qty="0.1")
    executor.execute_target(
        TargetPosition(SYMBOL, PositionSide.LONG, Decimal("0.2")), PRICE, equity=1_000_000.0
    )
    assert risk.halted
    assert executor.position(SYMBOL).quantity == Decimal(
        "0.1"
    ), "the order that breached the rate limit was still filled"


def test_reconciliation_notices_a_margin_regime_disagreement():
    executor, venue, _ = build(FuturesStore.open(None))
    executor.recover({})
    opened(executor, qty="0.5")
    local = executor.position(SYMBOL)
    venue.apply_settlement(
        SYMBOL,
        Position(SYMBOL, local.side, local.quantity, local.entry_price, leverage=Decimal("3")),
    )
    report = executor.reconcile(SYMBOL)
    assert report.outcome is ReconciliationOutcome.MISMATCH
    assert "3x" in report.detail


def test_quantize_quantity_never_rounds_up_at_any_precision():
    """A quantity longer than the default context divided to one ulp above an
    integer, and ROUND_DOWN then floored to the step above."""
    constraints = load_constraint_source().constraints(SYMBOL)
    for value in (
        "0.0019",
        "0.0015",
        "1.9999999999999999999999999999999999",
        "0.0009999999999999999999999999999999",
    ):
        quantity = Decimal(value)
        assert constraints.quantize_quantity(quantity) <= quantity, value
