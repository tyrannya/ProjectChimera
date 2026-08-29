"""Position, plan and order-lifecycle semantics in ``chimera.futures.domain``.

Two properties are what this module exists to make structural, and most of the
tests below exist to catch them regressing:

* a position's ``quantity`` is a magnitude with a side, never a signed number,
  so no arithmetic slip can turn an over-large reduction into a reversal;
* a reversal is *planned* as two orders, so no single order ever carries a
  quantity larger than the position it reduces.

The order state machine gets the same treatment: an illegal transition raises
rather than being clamped, logged or ignored.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from chimera.futures import (
    ALLOWED_TRANSITIONS,
    TERMINAL_STATES,
    EventKind,
    FuturesError,
    InvalidTransition,
    OrderEvent,
    OrderIntent,
    OrderPurpose,
    OrderRecord,
    OrderSide,
    OrderState,
    Position,
    PositionError,
    PositionSide,
    TargetPosition,
    can_transition,
    gross_exposure,
    net_exposure,
    plan_flatten,
    plan_transition,
)

SYMBOL = "BTC/USDT:USDT"
OTHER_SYMBOL = "ETH/USDT:USDT"


def position(side, quantity, entry_price="30000", symbol=SYMBOL, leverage="1"):
    """A Position from strings, so no test has to spell Decimal out five times."""
    return Position(
        symbol=symbol,
        side=side,
        quantity=Decimal(quantity),
        entry_price=Decimal(entry_price),
        leverage=Decimal(leverage),
    )


def flat(symbol=SYMBOL):
    return Position(symbol=symbol)


def wanted(side, quantity, symbol=SYMBOL):
    """The TargetPosition a strategy would ask for."""
    return TargetPosition(symbol=symbol, side=side, quantity=Decimal(quantity))


def an_intent(
    purpose=OrderPurpose.OPEN,
    side=OrderSide.BUY,
    quantity="0.5",
    reduce_only=False,
    position_side=PositionSide.LONG,
):
    return OrderIntent(
        symbol=SYMBOL,
        side=side,
        quantity=Decimal(quantity),
        purpose=purpose,
        reduce_only=reduce_only,
        position_side=position_side,
    )


def an_order(state=OrderState.PLANNED):
    return OrderRecord(order_id="ord-1", intent=an_intent(), state=state)


# --- position construction ------------------------------------------------
def test_a_negative_quantity_is_refused_because_short_is_a_side_not_a_sign():
    """Catches a signed-quantity representation sneaking back in.

    With a signed quantity a "reduce by 3" against a position of 2 becomes a
    SHORT of 1 and nothing downstream can tell that from an intended reversal.
    """
    with pytest.raises(PositionError, match="SHORT is a side"):
        Position(symbol=SYMBOL, side=PositionSide.LONG, quantity=Decimal("-1"))


def test_a_flat_position_carrying_a_quantity_is_refused():
    """side is FLAT <=> quantity == 0; exposure with no side is unbookable."""
    with pytest.raises(PositionError, match="side FLAT with quantity 0.5"):
        Position(symbol=SYMBOL, side=PositionSide.FLAT, quantity=Decimal("0.5"))


def test_a_non_flat_position_with_no_quantity_is_refused():
    """The other half of the invariant: a side with nothing behind it."""
    with pytest.raises(PositionError, match="side SHORT with quantity 0"):
        Position(symbol=SYMBOL, side=PositionSide.SHORT, quantity=Decimal("0"))


@pytest.mark.parametrize("leverage", ["0", "-2"])
def test_a_non_positive_leverage_is_refused(leverage):
    """Zero leverage divides by zero in the margin figures Aegis is handed."""
    with pytest.raises(PositionError, match=f"leverage {leverage} is not positive"):
        position(PositionSide.LONG, "1", leverage=leverage)


# --- apply_fill: opening --------------------------------------------------
def test_a_buy_into_a_flat_position_opens_a_long_at_the_fill_price():
    """A first fill sets the side and *is* the entry price; nothing is averaged."""
    opened, realised = flat().apply_fill(OrderSide.BUY, Decimal("0.5"), Decimal("30000"))
    assert opened.side is PositionSide.LONG
    assert opened.quantity == Decimal("0.5")
    assert opened.entry_price == Decimal("30000")
    assert realised == Decimal("0")


def test_a_sell_into_a_flat_position_opens_a_short_at_the_fill_price():
    """SHORT is opened by a SELL and is a side, not a negative LONG."""
    opened, realised = flat().apply_fill(OrderSide.SELL, Decimal("0.5"), Decimal("30000"))
    assert opened.side is PositionSide.SHORT
    assert opened.quantity == Decimal("0.5")
    assert opened.entry_price == Decimal("30000")
    assert realised == Decimal("0")


@pytest.mark.parametrize("quantity", ["0", "-1"])
def test_a_fill_of_no_quantity_is_refused(quantity):
    with pytest.raises(PositionError, match=f"fill quantity {quantity} is not positive"):
        flat().apply_fill(OrderSide.BUY, Decimal(quantity), Decimal("30000"))


def test_a_fill_at_no_price_is_refused():
    with pytest.raises(PositionError, match="fill price 0 is not positive"):
        flat().apply_fill(OrderSide.BUY, Decimal("1"), Decimal("0"))


# --- apply_fill: increasing -----------------------------------------------
def test_increasing_a_long_volume_weights_the_entry_price():
    """3 @ 30000 plus 1 @ 34000 is 4 @ 31000, not 4 @ 34000 and not 4 @ 32000.

    Overwriting the entry price with the newest fill, or averaging the two
    prices unweighted, both look right on a doubling and are wrong everywhere
    else — and the entry price is what every PnL and liquidation figure hangs on.
    """
    grown, realised = position(PositionSide.LONG, "3").apply_fill(
        OrderSide.BUY, Decimal("1"), Decimal("34000")
    )
    assert grown.side is PositionSide.LONG
    assert grown.quantity == Decimal("4")
    assert grown.entry_price == Decimal("31000")
    assert realised == Decimal("0"), "adding exposure realises nothing"


def test_increasing_a_short_volume_weights_the_entry_price():
    """The same weighting on the other side: 3 @ 30000 plus 1 @ 26000 is 4 @ 29000."""
    grown, realised = position(PositionSide.SHORT, "3").apply_fill(
        OrderSide.SELL, Decimal("1"), Decimal("26000")
    )
    assert grown.side is PositionSide.SHORT
    assert grown.quantity == Decimal("4")
    assert grown.entry_price == Decimal("29000")
    assert realised == Decimal("0")


# --- apply_fill: reducing -------------------------------------------------
def test_reducing_a_long_leaves_the_entry_price_alone_and_realises_the_gain():
    """Realising part of a position must not re-price the part still open."""
    reduced, realised = position(PositionSide.LONG, "2").apply_fill(
        OrderSide.SELL, Decimal("0.5"), Decimal("31000")
    )
    assert reduced.side is PositionSide.LONG
    assert reduced.quantity == Decimal("1.5")
    assert reduced.entry_price == Decimal("30000")
    assert realised == Decimal("500")


def test_reducing_a_short_realises_the_opposite_sign_of_the_same_move():
    """A price rise pays a LONG and costs a SHORT; one sign error flips the book."""
    reduced, realised = position(PositionSide.SHORT, "2").apply_fill(
        OrderSide.BUY, Decimal("0.5"), Decimal("31000")
    )
    assert reduced.side is PositionSide.SHORT
    assert reduced.quantity == Decimal("1.5")
    assert reduced.entry_price == Decimal("30000")
    assert realised == Decimal("-500")


def test_closing_a_position_exactly_returns_flat_with_no_quantity_or_entry_price():
    """A closed position keeps no residue: FLAT, zero, and no stale entry price."""
    closed, realised = position(PositionSide.LONG, "2").apply_fill(
        OrderSide.SELL, Decimal("2"), Decimal("30500")
    )
    assert closed.side is PositionSide.FLAT
    assert closed.quantity == Decimal("0")
    assert closed.entry_price == Decimal("0")
    assert closed.is_flat
    assert realised == Decimal("1000")


# --- apply_fill: the reversal guard ---------------------------------------
def test_a_reducing_fill_larger_than_a_long_refuses_to_reverse_it():
    """The load-bearing one: a close is a close, never an open on the far side.

    A venue reporting a reducing fill bigger than the position it reduces is a
    reconciliation problem. Booking it as a SHORT would silently open exposure
    nothing approved.
    """
    with pytest.raises(PositionError, match="would reverse it") as excinfo:
        position(PositionSide.LONG, "1").apply_fill(
            OrderSide.SELL, Decimal("2"), Decimal("30000")
        )
    assert "a reducing fill of 2 against a LONG position of 1" in str(excinfo.value)


def test_a_reducing_fill_larger_than_a_short_refuses_to_reverse_it():
    """Same guard on the other side, which a sign-based implementation would miss."""
    with pytest.raises(PositionError, match="would reverse it") as excinfo:
        position(PositionSide.SHORT, "1").apply_fill(
            OrderSide.BUY, Decimal("2"), Decimal("30000")
        )
    assert "a reducing fill of 2 against a SHORT position of 1" in str(excinfo.value)


# --- plan_transition: one test per row of the table -----------------------
def test_planning_flat_to_long_opens_with_a_single_buy():
    (intent,) = plan_transition(flat(), wanted(PositionSide.LONG, "0.5"))
    assert intent.purpose is OrderPurpose.OPEN
    assert intent.side is OrderSide.BUY
    assert intent.quantity == Decimal("0.5")
    assert intent.reduce_only is False
    assert intent.position_side is PositionSide.LONG


def test_planning_flat_to_short_opens_with_a_single_sell():
    (intent,) = plan_transition(flat(), wanted(PositionSide.SHORT, "0.5"))
    assert intent.purpose is OrderPurpose.OPEN
    assert intent.side is OrderSide.SELL
    assert intent.quantity == Decimal("0.5")
    assert intent.reduce_only is False
    assert intent.position_side is PositionSide.SHORT


def test_planning_a_larger_long_orders_only_the_difference():
    """Ordering the whole target rather than the delta would double the position."""
    (intent,) = plan_transition(
        position(PositionSide.LONG, "0.5"), wanted(PositionSide.LONG, "2")
    )
    assert intent.purpose is OrderPurpose.INCREASE
    assert intent.side is OrderSide.BUY
    assert intent.quantity == Decimal("1.5")
    assert intent.reduce_only is False
    assert intent.position_side is PositionSide.LONG


def test_planning_a_larger_short_orders_only_the_difference():
    (intent,) = plan_transition(
        position(PositionSide.SHORT, "0.5"), wanted(PositionSide.SHORT, "2")
    )
    assert intent.purpose is OrderPurpose.INCREASE
    assert intent.side is OrderSide.SELL
    assert intent.quantity == Decimal("1.5")
    assert intent.reduce_only is False
    assert intent.position_side is PositionSide.SHORT


def test_planning_a_smaller_long_sells_the_difference_reduce_only():
    """A reduction is a positive magnitude on the opposite side, and reduce-only."""
    (intent,) = plan_transition(
        position(PositionSide.LONG, "2"), wanted(PositionSide.LONG, "0.5")
    )
    assert intent.purpose is OrderPurpose.REDUCE
    assert intent.side is OrderSide.SELL
    assert intent.quantity == Decimal("1.5")
    assert intent.reduce_only is True
    assert intent.position_side is PositionSide.LONG


def test_planning_a_smaller_short_buys_the_difference_reduce_only():
    (intent,) = plan_transition(
        position(PositionSide.SHORT, "2"), wanted(PositionSide.SHORT, "0.5")
    )
    assert intent.purpose is OrderPurpose.REDUCE
    assert intent.side is OrderSide.BUY
    assert intent.quantity == Decimal("1.5")
    assert intent.reduce_only is True
    assert intent.position_side is PositionSide.SHORT


def test_planning_a_long_to_flat_closes_exactly_the_position_reduce_only():
    (intent,) = plan_transition(position(PositionSide.LONG, "2"), TargetPosition.flat(SYMBOL))
    assert intent.purpose is OrderPurpose.CLOSE
    assert intent.side is OrderSide.SELL
    assert intent.quantity == Decimal("2")
    assert intent.reduce_only is True
    assert intent.position_side is PositionSide.LONG


def test_planning_a_short_to_flat_closes_exactly_the_position_reduce_only():
    (intent,) = plan_transition(position(PositionSide.SHORT, "2"), TargetPosition.flat(SYMBOL))
    assert intent.purpose is OrderPurpose.CLOSE
    assert intent.side is OrderSide.BUY
    assert intent.quantity == Decimal("2")
    assert intent.reduce_only is True
    assert intent.position_side is PositionSide.SHORT


@pytest.mark.parametrize("side", [PositionSide.LONG, PositionSide.SHORT])
def test_planning_a_position_that_is_already_the_target_orders_nothing(side):
    """A no-op that emitted a zero-quantity order would be refused downstream."""
    assert plan_transition(position(side, "1.25"), wanted(side, "1.25")) == []


def test_planning_flat_to_flat_orders_nothing():
    assert plan_transition(flat(), TargetPosition.flat(SYMBOL)) == []


# --- plan_transition: the reversal --------------------------------------------
@pytest.mark.parametrize(
    "current_side, close_side, open_side",
    [
        (PositionSide.LONG, OrderSide.SELL, OrderSide.SELL),
        (PositionSide.SHORT, OrderSide.BUY, OrderSide.BUY),
    ],
)
def test_a_reversal_is_planned_as_a_close_then_an_open_and_never_as_one_order(
    current_side, close_side, open_side
):
    """The property that makes "a close cannot reverse" structural.

    Expressing LONG 2 -> SHORT 3 as one order of 5 is what every signed-quantity
    implementation does, and it is one slip away from a close that overshoots
    into a new position. No intent here may carry the combined 5.
    """
    current = position(current_side, "2")
    target = wanted(current_side.opposite, "3")

    close, open_ = plan_transition(current, target)

    assert close.purpose is OrderPurpose.CLOSE
    assert close.side is close_side
    assert close.quantity == Decimal("2")
    assert close.reduce_only is True
    assert close.position_side is current_side

    assert open_.purpose is OrderPurpose.OPEN
    assert open_.side is open_side
    assert open_.quantity == Decimal("3")
    assert open_.reduce_only is False
    assert open_.position_side is current_side.opposite

    combined = current.quantity + target.quantity
    assert all(i.quantity != combined for i in (close, open_))


def test_planning_one_symbol_against_another_symbols_target_is_refused():
    """Planning across symbols would size an order against the wrong position."""
    with pytest.raises(PositionError, match="cannot plan"):
        plan_transition(
            position(PositionSide.LONG, "1"), wanted(PositionSide.SHORT, "1", OTHER_SYMBOL)
        )


# --- plan_flatten -------------------------------------------------------------
@pytest.mark.parametrize(
    "side, order_side",
    [(PositionSide.LONG, OrderSide.SELL), (PositionSide.SHORT, OrderSide.BUY)],
)
def test_flattening_orders_exactly_the_position_reduce_only(side, order_side):
    """An emergency flatten reaches zero and, being exact and reduce-only, stops."""
    (intent,) = plan_flatten(position(side, "1.75"))
    assert intent.purpose is OrderPurpose.FLATTEN
    assert intent.side is order_side
    assert intent.quantity == Decimal("1.75")
    assert intent.reduce_only is True
    assert intent.position_side is side


def test_flattening_a_flat_position_orders_nothing():
    assert plan_flatten(flat()) == []


# --- order sides --------------------------------------------------------------
def test_order_sides_open_and_close_each_position_side_the_right_way_round():
    """A close that used the opening side would double the position, not end it."""
    assert OrderSide.opening(PositionSide.LONG) is OrderSide.BUY
    assert OrderSide.opening(PositionSide.SHORT) is OrderSide.SELL
    assert OrderSide.closing(PositionSide.LONG) is OrderSide.SELL
    assert OrderSide.closing(PositionSide.SHORT) is OrderSide.BUY


def test_no_order_side_opens_a_flat_position():
    """FLAT is the absence of a position, so there is nothing for an order to open."""
    with pytest.raises(PositionError, match="FLAT is not a side an order can open"):
        OrderSide.opening(PositionSide.FLAT)


# --- the state machine --------------------------------------------------------
#: The lifecycle as the module documents it, written out by hand. Compared
#: against ALLOWED_TRANSITIONS below, so widening the table without widening
#: this list fails rather than passing quietly.
LEGAL_TRANSITIONS = [
    (OrderState.PLANNED, OrderState.RISK_APPROVED),
    (OrderState.PLANNED, OrderState.REJECTED),
    (OrderState.PLANNED, OrderState.CANCELLED),
    (OrderState.PLANNED, OrderState.FAILED),
    (OrderState.RISK_APPROVED, OrderState.SUBMITTED),
    (OrderState.RISK_APPROVED, OrderState.CANCELLED),
    (OrderState.RISK_APPROVED, OrderState.FAILED),
    (OrderState.SUBMITTED, OrderState.ACKNOWLEDGED),
    (OrderState.SUBMITTED, OrderState.REJECTED),
    (OrderState.SUBMITTED, OrderState.FAILED),
    (OrderState.SUBMITTED, OrderState.RECONCILIATION_REQUIRED),
    (OrderState.ACKNOWLEDGED, OrderState.PARTIALLY_FILLED),
    (OrderState.ACKNOWLEDGED, OrderState.FILLED),
    (OrderState.ACKNOWLEDGED, OrderState.CANCELLED),
    (OrderState.ACKNOWLEDGED, OrderState.REJECTED),
    (OrderState.ACKNOWLEDGED, OrderState.FAILED),
    (OrderState.ACKNOWLEDGED, OrderState.RECONCILIATION_REQUIRED),
    (OrderState.PARTIALLY_FILLED, OrderState.PARTIALLY_FILLED),
    (OrderState.PARTIALLY_FILLED, OrderState.FILLED),
    (OrderState.PARTIALLY_FILLED, OrderState.CANCELLED),
    (OrderState.PARTIALLY_FILLED, OrderState.FAILED),
    (OrderState.PARTIALLY_FILLED, OrderState.RECONCILIATION_REQUIRED),
    (OrderState.RECONCILIATION_REQUIRED, OrderState.CANCELLED),
    (OrderState.RECONCILIATION_REQUIRED, OrderState.FAILED),
    (OrderState.RECONCILIATION_REQUIRED, OrderState.FILLED),
]

ILLEGAL_TRANSITIONS = [
    (OrderState.FILLED, OrderState.PARTIALLY_FILLED),
    (OrderState.CANCELLED, OrderState.FILLED),
    (OrderState.REJECTED, OrderState.ACKNOWLEDGED),
    (OrderState.PLANNED, OrderState.FILLED),
    (OrderState.PLANNED, OrderState.SUBMITTED),
    (OrderState.SUBMITTED, OrderState.PARTIALLY_FILLED),
    (OrderState.ACKNOWLEDGED, OrderState.SUBMITTED),
    (OrderState.FAILED, OrderState.CANCELLED),
    (OrderState.RECONCILIATION_REQUIRED, OrderState.PARTIALLY_FILLED),
]


def test_the_transition_table_is_total_over_every_order_state():
    """A state missing from the table makes can_transition raise KeyError instead."""
    assert set(ALLOWED_TRANSITIONS) == set(OrderState)


def test_every_state_is_reachable_from_the_state_an_order_starts_in():
    """The mirror of totality: a state no edge leads to is one no order can enter.

    Totality above catches a state missing from the table's *keys*. This catches
    the other half — a state that is a key, and a legal destination in nobody's
    value set, so the executor that tries to move an order into it can only ever
    get InvalidTransition.
    """
    reached = {OrderState.PLANNED}
    frontier = [OrderState.PLANNED]
    while frontier:
        for target in ALLOWED_TRANSITIONS[frontier.pop()]:
            if target not in reached:
                reached.add(target)
                frontier.append(target)
    assert reached == set(OrderState)


def test_the_transition_table_is_exactly_the_documented_lifecycle():
    """Catches the table being widened without anyone restating the lifecycle."""
    table = {(state, t) for state, targets in ALLOWED_TRANSITIONS.items() for t in targets}
    assert table == set(LEGAL_TRANSITIONS)


@pytest.mark.parametrize("source, destination", LEGAL_TRANSITIONS)
def test_a_legal_transition_moves_the_order_and_is_appended_to_its_history(
    source, destination
):
    """History is the audit trail; a move that did not append is invisible later."""
    assert can_transition(source, destination) is True
    order = an_order(state=source)
    order.transition(destination, "venue said so")
    assert order.state is destination
    assert order.history == [f"{source.value}->{destination.value}"]
    assert order.reason == "venue said so"


@pytest.mark.parametrize("source, destination", ILLEGAL_TRANSITIONS)
def test_an_illegal_transition_raises_and_names_both_states(source, destination):
    """Clamping or ignoring these is how an order fills after it was cancelled."""
    assert can_transition(source, destination) is False
    order = an_order(state=source)
    with pytest.raises(InvalidTransition) as excinfo:
        order.transition(destination)
    assert f"{source.value} -> {destination.value}" in str(excinfo.value)
    assert order.state is source, "a refused transition must not half-happen"
    assert order.history == []


def test_terminal_states_are_exactly_the_states_with_nowhere_to_go():
    """TERMINAL_STATES and the table are two statements of one fact; they must agree."""
    assert {s for s, targets in ALLOWED_TRANSITIONS.items() if not targets} == set(
        TERMINAL_STATES
    )


@pytest.mark.parametrize("state", sorted(TERMINAL_STATES, key=lambda s: s.value))
def test_an_order_in_a_terminal_state_refuses_every_further_transition(state):
    """Nothing follows a terminal state, including a repeat of the state itself."""
    for destination in OrderState:
        order = an_order(state=state)
        assert order.is_terminal
        with pytest.raises(InvalidTransition):
            order.transition(destination)


# --- order events -------------------------------------------------------------
def test_an_event_with_no_id_is_refused_because_it_cannot_be_deduplicated():
    """An event with no identity may be applied twice, booking exposure twice."""
    with pytest.raises(FuturesError, match="cannot be deduplicated"):
        OrderEvent(event_id="", kind=EventKind.FILL, quantity=Decimal("1"), price=Decimal("1"))


@pytest.mark.parametrize("kind", [EventKind.FILL, EventKind.PARTIAL_FILL])
def test_a_fill_event_that_fills_nothing_is_refused(kind):
    with pytest.raises(FuturesError, match=f"{kind.value} event ord-1:1 fills nothing"):
        OrderEvent(
            event_id="ord-1:1", kind=kind, quantity=Decimal("0"), price=Decimal("30000")
        )


@pytest.mark.parametrize("kind", [EventKind.FILL, EventKind.PARTIAL_FILL])
def test_a_fill_event_with_no_price_is_refused(kind):
    """A priced fill is the only kind that can be booked; zero is not a price."""
    with pytest.raises(FuturesError, match=f"{kind.value} event ord-1:1 has no price"):
        OrderEvent(event_id="ord-1:1", kind=kind, quantity=Decimal("1"), price=Decimal("0"))


def test_a_non_fill_event_may_carry_no_quantity_or_price():
    """The quantity and price checks apply to fills only, not to every event."""
    event = OrderEvent(event_id="ord-1:2", kind=EventKind.CANCELLED, reason="operator")
    assert event.quantity == Decimal("0")
    assert event.price == Decimal("0")
    assert event.reason == "operator"


# --- order intents ------------------------------------------------------------
@pytest.mark.parametrize(
    "purpose, reduce_only",
    [
        (OrderPurpose.OPEN, True),
        (OrderPurpose.INCREASE, True),
        (OrderPurpose.REDUCE, False),
        (OrderPurpose.CLOSE, False),
        (OrderPurpose.FLATTEN, False),
    ],
)
def test_an_intent_whose_purpose_and_reduce_only_disagree_is_refused(purpose, reduce_only):
    """reduce_only is the venue-side restatement of the local reversal guard.

    An OPEN sent reduce-only would be silently dropped by the venue; a CLOSE sent
    without it is a close that can reverse.
    """
    with pytest.raises(PositionError, match="disagree about whether this order adds"):
        an_intent(purpose=purpose, reduce_only=reduce_only)


def test_an_intent_for_no_quantity_is_refused():
    with pytest.raises(PositionError, match="order quantity 0 is not positive"):
        an_intent(quantity="0")


# --- persistence round trips --------------------------------------------------
def test_a_position_survives_a_dict_round_trip_with_its_decimals_intact():
    """Persisted state is JSON; a float round trip here would move an entry price."""
    original = Position(
        symbol=SYMBOL,
        side=PositionSide.SHORT,
        quantity=Decimal("0.125"),
        entry_price=Decimal("30123.45"),
        leverage=Decimal("1"),
    )
    revived = Position.from_dict(original.to_dict())
    assert revived == original
    assert revived.quantity == Decimal("0.125")
    assert str(revived.entry_price) == "30123.45"
    assert revived.side is PositionSide.SHORT
    assert revived.margin_mode == "ISOLATED"


def test_an_intent_survives_a_dict_round_trip_with_its_decimals_intact():
    original = an_intent(
        purpose=OrderPurpose.REDUCE,
        side=OrderSide.SELL,
        quantity="0.001",
        reduce_only=True,
        position_side=PositionSide.LONG,
    )
    revived = OrderIntent.from_dict(original.to_dict())
    assert revived == original
    assert str(revived.quantity) == "0.001"
    assert revived.reduce_only is True
    assert revived.purpose is OrderPurpose.REDUCE


def test_an_event_survives_a_dict_round_trip_with_its_decimals_intact():
    original = OrderEvent(
        event_id="ord-1:3",
        kind=EventKind.PARTIAL_FILL,
        quantity=Decimal("0.007"),
        price=Decimal("30250.10"),
        fee=Decimal("0.10587535"),
        reason="",
    )
    revived = OrderEvent.from_dict(original.to_dict())
    assert revived == original
    assert str(revived.quantity) == "0.007"
    assert str(revived.price) == "30250.10"
    assert str(revived.fee) == "0.10587535"


def test_an_order_record_survives_a_dict_round_trip_including_its_applied_events():
    """applied_events is the whole idempotency guarantee, so it has to persist."""
    original = an_order(state=OrderState.PARTIALLY_FILLED)
    original.filled_quantity = Decimal("0.25")
    original.average_price = Decimal("30250.10")
    original.fees = Decimal("3.78126250")
    original.applied_events = ["ord-1:1", "ord-1:2"]
    original.history = ["PLANNED->RISK_APPROVED", "RISK_APPROVED->SUBMITTED"]
    original.reason = "partially filled"

    revived = OrderRecord.from_dict(original.to_dict())

    assert revived == original
    assert revived.applied_events == ["ord-1:1", "ord-1:2"]
    assert str(revived.average_price) == "30250.10"
    assert revived.remaining_quantity == Decimal("0.25")
    assert revived.intent.quantity == Decimal("0.5")
    assert revived.is_terminal is False


# --- exposure -----------------------------------------------------------------
def test_gross_exposure_adds_both_sides_while_net_exposure_subtracts_the_short():
    """A SHORT that added to net exposure would hide a hedge as concentration."""
    positions = [
        position(PositionSide.LONG, "2"),
        position(PositionSide.SHORT, "10", entry_price="2000", symbol=OTHER_SYMBOL),
    ]
    prices = {SYMBOL: Decimal("31000"), OTHER_SYMBOL: Decimal("2100")}

    assert gross_exposure(positions, prices) == Decimal("83000")
    assert net_exposure(positions, prices) == Decimal("41000")


def test_a_flat_position_contributes_nothing_to_either_exposure():
    prices = {SYMBOL: Decimal("31000")}
    assert gross_exposure([flat()], prices) == Decimal("0")
    assert net_exposure([flat()], prices) == Decimal("0")
