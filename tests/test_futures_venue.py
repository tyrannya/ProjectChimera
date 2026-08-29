"""Venue constraints, and the dry-run simulator that stands in for the exchange.

Two claims are under test here and nothing else. First, that venue metadata is
*refused* rather than repaired: a missing, non-positive or self-contradictory
constraint stops order planning, because a plausible substituted tick size is how
an order ends up a different size than the risk engine approved. Second, that the
simulator is pessimistic and deterministic — adverse slippage, exact partial
fills, identical answers to identical questions — since a simulator that
sometimes improves the price is one whose expected cost is lower than a real
venue's.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from chimera.futures import (
    SUPPORTED_ORDER_TYPES,
    TRADABLE_STATUS,
    ConstraintError,
    DeterministicFillModel,
    DryRunFuturesVenue,
    EventKind,
    OrderIntent,
    OrderPurpose,
    OrderSide,
    PositionSide,
    StaticConstraintSource,
    SymbolConstraints,
    default_constraints_table,
    load_constraint_source,
)

SYMBOL = "BTC/USDT:USDT"
ZERO = Decimal("0")
#: On the 0.10 tick grid, so a fill price is exact and a placement check is
#: about the thing each test is actually about.
REFERENCE = Decimal("60000")

#: The fields :meth:`SymbolConstraints.from_dict` refuses to do without. Spelled
#: out rather than imported so the parametrisation reads as a checklist;
#: ``test_the_required_field_list_matches_the_modules_own`` keeps the two in step.
REQUIRED_FIELDS = (
    "symbol",
    "status",
    "tick_size",
    "step_size",
    "quantity_precision",
    "price_precision",
    "min_quantity",
    "min_notional",
    "maintenance_margin_rate",
    "taker_fee_rate",
    "maker_fee_rate",
)


def constraint_fields(**overrides):
    """The committed BTC/USDT:USDT metadata, shaped the way ``from_dict`` wants it."""
    fields = dict(default_constraints_table()[SYMBOL])
    fields["symbol"] = SYMBOL
    fields.update(overrides)
    return fields


def constraints(**overrides) -> SymbolConstraints:
    """Validated constraints for the committed symbol, with fields swapped out."""
    return SymbolConstraints.from_dict(constraint_fields(**overrides))


def venue(fill_model=None, **overrides) -> DryRunFuturesVenue:
    """A fresh simulated venue holding no position."""
    source = StaticConstraintSource.from_mapping({SYMBOL: constraint_fields(**overrides)})
    return DryRunFuturesVenue(source=source, fill_model=fill_model or DeterministicFillModel())


def open_intent(side: OrderSide, quantity: str = "0.500") -> OrderIntent:
    """A well-formed opening order. ``OrderIntent`` checks its own coherence."""
    return OrderIntent(
        symbol=SYMBOL,
        side=side,
        quantity=Decimal(quantity),
        purpose=OrderPurpose.OPEN,
        reduce_only=False,
        position_side=PositionSide.LONG if side is OrderSide.BUY else PositionSide.SHORT,
    )


# --- the committed metadata -----------------------------------------------
def test_the_default_constraints_table_yields_the_documented_binance_values():
    """The dry-run protocol runs against these numbers, so they are pinned here.

    Catches a silent edit to the committed BTCUSDT filters: every simulated fill
    price, fee and minimum in this package is derived from them.
    """
    limits = constraints()
    assert limits.symbol == SYMBOL
    assert limits.status == TRADABLE_STATUS == "TRADING"
    assert limits.tradable is True
    assert limits.tick_size == Decimal("0.10")
    assert limits.step_size == Decimal("0.001")
    assert limits.quantity_precision == 3
    assert limits.price_precision == 2
    assert limits.min_quantity == Decimal("0.001")
    assert limits.min_notional == Decimal("100")
    assert limits.maintenance_margin_rate == Decimal("0.004")
    assert limits.taker_fee_rate == Decimal("0.0005")
    assert limits.maker_fee_rate == Decimal("0.0002")
    assert limits.supported_order_types == frozenset({"MARKET"})
    assert limits.supports_reduce_only is True
    assert limits.supported_position_sides == frozenset({"LONG", "SHORT"})


def test_the_committed_table_is_exactly_one_symbol_and_it_validates():
    """A second symbol appearing here would trade under untested constraints."""
    source = load_constraint_source()
    assert sorted(source.table) == [SYMBOL]
    assert source.constraints(SYMBOL).min_notional == Decimal("100")


def test_the_required_field_list_matches_the_modules_own():
    """Guards the checklist above: a newly required field must gain a test."""
    from chimera.futures.venue import _REQUIRED_FIELDS

    assert REQUIRED_FIELDS == tuple(_REQUIRED_FIELDS)


# --- fail closed on the metadata itself ------------------------------------
@pytest.mark.parametrize("field", REQUIRED_FIELDS)
def test_every_required_field_is_refused_when_missing(field):
    """No field has a default. A guessed constraint sizes an order by guesswork."""
    data = constraint_fields()
    del data[field]
    with pytest.raises(ConstraintError) as excinfo:
        SymbolConstraints.from_dict(data)
    message = str(excinfo.value)
    assert f"missing ['{field}']" in message
    assert "sized by guesswork" in message


def test_a_non_numeric_tick_size_is_refused_rather_than_defaulted():
    """Metadata that cannot be read is not metadata that can be assumed."""
    with pytest.raises(ConstraintError) as excinfo:
        constraints(tick_size="0.1O")
    message = str(excinfo.value)
    assert "constraint 'tick_size' is '0.1O', which is not a number" in message


@pytest.mark.parametrize(
    "field, value",
    [
        ("tick_size", "nan"),
        ("tick_size", "Infinity"),
        ("step_size", "Infinity"),
        ("min_notional", "Infinity"),
    ],
)
def test_non_finite_metadata_is_refused_as_a_constraint_error(field, value):
    """NaN and infinity are not numbers a venue quotes, and must fail closed.

    They are worse than unreadable metadata: a NaN comparison signals rather than
    answers, and an infinite minimum silently makes every order unplaceable.
    """
    with pytest.raises(ConstraintError) as excinfo:
        constraints(**{field: value})
    message = str(excinfo.value)
    assert SYMBOL in message
    assert field in message


@pytest.mark.parametrize("tick_size", ["0", "-0.10"])
def test_a_non_positive_tick_size_is_refused(tick_size):
    """A zero or negative increment makes every price representable."""
    with pytest.raises(ConstraintError) as excinfo:
        constraints(tick_size=tick_size)
    assert "non-positive increment" in str(excinfo.value)


@pytest.mark.parametrize("step_size", ["0", "-0.001"])
def test_a_non_positive_step_size_is_refused(step_size):
    """Same refusal for quantities: without a step there is no quantity grid."""
    with pytest.raises(ConstraintError) as excinfo:
        constraints(step_size=step_size)
    assert "non-positive increment" in str(excinfo.value)


@pytest.mark.parametrize("min_quantity", ["0", "-0.001"])
def test_a_non_positive_min_quantity_is_refused(min_quantity):
    """A minimum of zero would let a zero-quantity order look placeable."""
    with pytest.raises(ConstraintError) as excinfo:
        constraints(min_quantity=min_quantity)
    assert f"min_quantity={min_quantity} is not positive" in str(excinfo.value)


def test_a_negative_min_notional_is_refused():
    """Zero is a legitimate minimum notional; below zero is a metadata error."""
    with pytest.raises(ConstraintError) as excinfo:
        constraints(min_notional="-1")
    assert "min_notional=-1 is negative" in str(excinfo.value)


@pytest.mark.parametrize("rate", ["0", "1", "-0.004"])
def test_a_maintenance_margin_rate_outside_the_open_unit_interval_is_refused(rate):
    """The liquidation estimate Aegis is handed is only defined for a fraction."""
    with pytest.raises(ConstraintError) as excinfo:
        constraints(maintenance_margin_rate=rate)
    assert f"maintenance_margin_rate={rate} is not a fraction in (0, 1)" in str(excinfo.value)


@pytest.mark.parametrize(
    "field, rate",
    [
        ("taker_fee_rate", "1"),
        ("taker_fee_rate", "-0.0005"),
        ("maker_fee_rate", "1"),
        ("maker_fee_rate", "-0.0002"),
    ],
)
def test_a_fee_rate_outside_zero_to_one_is_refused(field, rate):
    """A 100% fee is not a fee, and a negative one is a rebate path that is absent."""
    with pytest.raises(ConstraintError) as excinfo:
        constraints(**{field: rate})
    assert f"{field}={rate} is not a fraction in [0, 1)" in str(excinfo.value)


@pytest.mark.parametrize("field", ["quantity_precision", "price_precision"])
def test_a_negative_precision_is_refused(field):
    """A negative precision would quantize to tens, which no venue means."""
    with pytest.raises(ConstraintError) as excinfo:
        constraints(**{field: -1})
    assert "a precision field is negative" in str(excinfo.value)


def test_a_step_size_finer_than_the_quantity_precision_is_refused():
    """Contradictory metadata is refused, not reconciled: either field could be wrong."""
    with pytest.raises(ConstraintError) as excinfo:
        constraints(step_size="0.0001")
    assert "step_size=0.0001 needs more decimals than quantity_precision=3 allows" in str(
        excinfo.value
    )


def test_a_tick_size_finer_than_the_price_precision_is_refused():
    """The same contradiction on the price side, and the same refusal."""
    with pytest.raises(ConstraintError) as excinfo:
        constraints(tick_size="0.001")
    assert "tick_size=0.001 needs more decimals than price_precision=2 allows" in str(
        excinfo.value
    )


def test_a_min_quantity_off_the_step_grid_is_refused():
    """If the smallest allowed order is not on the grid, nothing is placeable."""
    with pytest.raises(ConstraintError) as excinfo:
        constraints(min_quantity="0.0015")
    message = str(excinfo.value)
    assert "min_quantity=0.0015 is not a multiple of step_size=0.001" in message
    assert "the smallest placeable order is not placeable" in message


def test_an_order_type_set_this_package_cannot_simulate_is_refused():
    """A LIMIT-only venue needs a resting-order model this package does not have."""
    with pytest.raises(ConstraintError) as excinfo:
        constraints(supported_order_types=["LIMIT"])
    message = str(excinfo.value)
    assert "the venue supports ['LIMIT']" in message
    assert "can only simulate ['MARKET']" in message


def test_order_types_are_narrowed_to_the_ones_this_package_simulates():
    """A venue offering more than MARKET is usable, but only its MARKET is kept."""
    limits = constraints(supported_order_types=["MARKET", "LIMIT"])
    assert limits.supported_order_types == frozenset({"MARKET"}) == SUPPORTED_ORDER_TYPES


# --- quantization -----------------------------------------------------------
def test_quantize_quantity_rounds_down_and_never_up():
    """Rounding up would submit an order larger than the risk engine approved."""
    limits = constraints()
    assert limits.quantize_quantity(Decimal("0.0019")) == Decimal("0.001")
    # Exactly half a step: rounds down, where quantize_price would round up.
    assert limits.quantize_quantity(Decimal("0.0015")) == Decimal("0.001")
    assert limits.quantize_quantity(Decimal("1.9999")) == Decimal("1.999")


def test_quantize_quantity_of_less_than_one_step_is_zero():
    """Dust rounds away entirely rather than up to the minimum placeable size."""
    limits = constraints()
    assert limits.quantize_quantity(Decimal("0.0009")) == ZERO
    assert limits.quantize_quantity(Decimal("-1")) == ZERO


def test_quantize_price_rounds_to_the_nearest_tick():
    """Prices round both ways: only the quantity is deliberately truncated."""
    limits = constraints()
    assert limits.quantize_price(Decimal("60000.04")) == Decimal("60000.00")
    assert limits.quantize_price(Decimal("60000.06")) == Decimal("60000.10")
    assert limits.quantize_price(Decimal("60000.05")) == Decimal("60000.10")


def test_quantize_price_refuses_a_non_positive_price():
    """A zero reference price is a missing price, not a cheap one."""
    with pytest.raises(ConstraintError) as excinfo:
        constraints().quantize_price(ZERO)
    assert "price 0 is not positive" in str(excinfo.value)


# --- placement checks -------------------------------------------------------
def test_check_placeable_refuses_a_symbol_that_is_not_trading():
    """A status off the tradable list and an untradable symbol are one answer."""
    with pytest.raises(ConstraintError) as excinfo:
        constraints(status="BREAK").check_placeable(
            Decimal("0.010"), Decimal("60000.00"), reduce_only=False
        )
    assert "status is 'BREAK', not 'TRADING'" in str(excinfo.value)


def test_check_placeable_refuses_a_quantity_below_the_minimum():
    """The quantity is on the step grid, so only the minimum can be refusing it."""
    with pytest.raises(ConstraintError) as excinfo:
        constraints(min_quantity="0.010").check_placeable(
            Decimal("0.005"), Decimal("60000.00"), reduce_only=False
        )
    assert "quantity 0.005 is below min_quantity 0.010" in str(excinfo.value)


def test_check_placeable_refuses_a_quantity_off_the_step_grid():
    """A quantity between two steps is one the venue would reject outright."""
    with pytest.raises(ConstraintError) as excinfo:
        constraints().check_placeable(
            Decimal("0.0015"), Decimal("60000.00"), reduce_only=False
        )
    assert "quantity 0.0015 is not a multiple of step_size 0.001" in str(excinfo.value)


def test_check_placeable_refuses_a_price_off_the_tick_grid():
    """Half a tick is not a price this venue quotes."""
    with pytest.raises(ConstraintError) as excinfo:
        constraints().check_placeable(Decimal("1.000"), Decimal("60000.05"), reduce_only=False)
    assert "price 60000.05 is not a multiple of tick_size 0.10" in str(excinfo.value)


def test_check_placeable_refuses_a_notional_below_the_minimum_when_opening():
    """60 USDT of exposure is below Binance's 100 minimum and cannot be opened."""
    with pytest.raises(ConstraintError) as excinfo:
        constraints().check_placeable(Decimal("0.001"), Decimal("60000.00"), reduce_only=False)
    assert "is below min_notional 100" in str(excinfo.value)


def test_check_placeable_exempts_a_reduce_only_order_from_the_minimum_notional():
    """Binance's own exemption: a dust position would otherwise be unclosable.

    This is a venue fact, so it lives here. The identical order refused by
    ``..._below_the_minimum_when_opening`` is allowed purely because it reduces.
    """
    limits = constraints()
    limits.check_placeable(Decimal("0.001"), Decimal("60000.00"), reduce_only=True)
    with pytest.raises(ConstraintError) as excinfo:
        limits.check_placeable(Decimal("0.001"), Decimal("60000.00"), reduce_only=False)
    assert "is below min_notional 100" in str(excinfo.value)


def test_check_placeable_refuses_reduce_only_where_the_venue_does_not_support_it():
    """Without reduce-only the venue cannot promise a close will not reverse."""
    with pytest.raises(ConstraintError) as excinfo:
        constraints(supports_reduce_only=False).check_placeable(
            Decimal("1.000"), Decimal("60000.00"), reduce_only=True
        )
    assert "does not accept reduce-only orders" in str(excinfo.value)


def test_an_unknown_symbol_is_refused_and_the_known_ones_are_named():
    """An unknown symbol is not a tradable one, and the message says what is."""
    with pytest.raises(ConstraintError) as excinfo:
        load_constraint_source().constraints("ETH/USDT:USDT")
    message = str(excinfo.value)
    assert "no venue metadata for 'ETH/USDT:USDT'" in message
    assert f"known symbols: ['{SYMBOL}']" in message


# --- the fill model ---------------------------------------------------------
def test_slippage_is_adverse_for_a_buy_and_for_a_sell():
    """A model that can improve the price costs less than a real venue does.

    Same reference, same 5 bps: the BUY fills strictly above it and the SELL
    strictly below it.
    """
    model = DeterministicFillModel()
    limits = constraints()
    bought = model.plan(open_intent(OrderSide.BUY), REFERENCE, limits)
    sold = model.plan(open_intent(OrderSide.SELL), REFERENCE, limits)

    assert bought.fills == ((Decimal("0.500"), Decimal("60030.00")),)
    assert sold.fills == ((Decimal("0.500"), Decimal("59970.00")),)
    assert bought.fills[0][1] > REFERENCE > sold.fills[0][1]


def test_the_fill_model_is_deterministic():
    """Identical inputs, identical fills — otherwise a replay is a sample."""
    model = DeterministicFillModel()
    limits = constraints()
    intent = open_intent(OrderSide.BUY)
    first = model.plan(intent, REFERENCE, limits)
    second = model.plan(intent, REFERENCE, limits)
    third = DeterministicFillModel().plan(intent, REFERENCE, limits)
    assert first == second == third


def test_a_capped_fill_ratio_splits_the_order_into_fills_that_sum_exactly():
    """Partial fills must not lose or invent quantity in the arithmetic."""
    model = DeterministicFillModel(max_fill_ratio=Decimal("0.4"))
    plan = model.plan(open_intent(OrderSide.BUY, "1.000"), REFERENCE, constraints())

    assert [quantity for quantity, _ in plan.fills] == [
        Decimal("0.400"),
        Decimal("0.400"),
        Decimal("0.200"),
    ]
    assert plan.filled_quantity == Decimal("1.000")
    assert plan.rejection == ""


def test_a_quantity_below_the_simulated_minimum_is_planned_as_a_rejection():
    """A rejected order fills nothing at all; it is not partially filled."""
    model = DeterministicFillModel(reject_below_quantity=Decimal("0.010"))
    plan = model.plan(open_intent(OrderSide.BUY, "0.005"), REFERENCE, constraints())
    assert plan.fills == ()
    assert plan.filled_quantity == ZERO
    assert plan.rejection == "below the venue's simulated minimum"


def test_the_fill_model_refuses_to_simulate_without_a_reference_price():
    """A zero mark is a missing mark, and a fill against it would be invented."""
    with pytest.raises(ConstraintError) as excinfo:
        DeterministicFillModel().plan(open_intent(OrderSide.BUY), ZERO, constraints())
    assert "no reference price to simulate a fill against" in str(excinfo.value)


# --- the dry-run venue ------------------------------------------------------
def test_submit_acknowledges_first_then_fills_and_moves_the_venue_position():
    """The venue's own position is what reconciliation compares against.

    Catches an acknowledgement emitted after a fill, and a reported position that
    drifts from the quantity actually filled.
    """
    market = venue()
    before = market.reported_position(SYMBOL)
    assert before.side is PositionSide.FLAT
    assert before.quantity == ZERO

    events = market.submit("ORD-1", open_intent(OrderSide.BUY, "0.500"), REFERENCE)
    assert [event.kind for event in events] == [EventKind.ACKNOWLEDGED, EventKind.FILL]
    assert events[1].quantity == Decimal("0.500")
    assert events[1].price == Decimal("60030.00")

    after = market.reported_position(SYMBOL)
    assert after.side is PositionSide.LONG
    assert after.quantity - before.quantity == Decimal("0.500")
    assert after.entry_price == Decimal("60030.00")


def test_submit_marks_every_fill_but_the_last_as_partial_with_unique_event_ids():
    """Duplicate ids would let one fill be booked twice, or two be booked once."""
    market = venue(DeterministicFillModel(max_fill_ratio=Decimal("0.4")))
    events = market.submit("ORD-1", open_intent(OrderSide.BUY, "1.000"), REFERENCE)

    assert [event.kind for event in events] == [
        EventKind.ACKNOWLEDGED,
        EventKind.PARTIAL_FILL,
        EventKind.PARTIAL_FILL,
        EventKind.FILL,
    ]
    assert [event.event_id for event in events] == [
        "ORD-1:1",
        "ORD-1:2",
        "ORD-1:3",
        "ORD-1:4",
    ]
    assert market.reported_position(SYMBOL).quantity == Decimal("1.000")

    # A resubmission under the *same* order id is where ids could collide, and a
    # collision would be swallowed by the executor's duplicate-event guard.
    again = market.submit("ORD-1", open_intent(OrderSide.BUY, "1.000"), REFERENCE)
    assert [event.event_id for event in again] == [
        "ORD-1:5",
        "ORD-1:6",
        "ORD-1:7",
        "ORD-1:8",
    ]


def test_submit_charges_the_taker_fee_on_every_fill():
    """Fees are per fill, not per order: a split order pays the same total."""
    limits = constraints()
    market = venue(DeterministicFillModel(max_fill_ratio=Decimal("0.4")))
    events = market.submit("ORD-1", open_intent(OrderSide.BUY, "1.000"), REFERENCE)
    fills = [e for e in events if e.kind in (EventKind.PARTIAL_FILL, EventKind.FILL)]

    for event in fills:
        assert event.fee == event.quantity * event.price * limits.taker_fee_rate
    assert [event.fee for event in fills] == [
        Decimal("12.006"),
        Decimal("12.006"),
        Decimal("6.003"),
    ]
    assert sum(event.fee for event in fills) == Decimal("30.015")


def test_submit_of_a_rejected_order_leaves_the_venue_position_untouched():
    """A rejection is acknowledged and then refused, and books no exposure."""
    market = venue(DeterministicFillModel(reject_below_quantity=Decimal("0.010")))
    events = market.submit("ORD-1", open_intent(OrderSide.BUY, "0.005"), REFERENCE)

    assert [event.kind for event in events] == [EventKind.ACKNOWLEDGED, EventKind.REJECTED]
    assert events[1].reason == "below the venue's simulated minimum"
    assert market.positions == {}
    assert market.reported_position(SYMBOL).side is PositionSide.FLAT
    assert market.reported_position(SYMBOL).quantity == ZERO


def test_submit_does_not_apply_the_per_order_minimums_to_each_fill_chunk():
    """Chunking is a simulator setting; it must not manufacture a venue rejection."""
    market = venue(DeterministicFillModel(max_fill_ratio=Decimal("0.5")))
    events = market.submit("ORD-1", open_intent(OrderSide.BUY, "0.002"), REFERENCE)

    assert [event.kind for event in events] == [
        EventKind.ACKNOWLEDGED,
        EventKind.PARTIAL_FILL,
        EventKind.FILL,
    ]
    assert market.reported_position(SYMBOL).quantity == Decimal("0.002")
