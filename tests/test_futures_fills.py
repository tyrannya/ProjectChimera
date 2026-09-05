"""Fills priced from a recorded book, and the venue's refusal of a side it cannot hold.

Two claims are under test. First, that :class:`RecordedQuoteFillModel` prices a
fill from the *book* and never from anything cheaper: a BUY crosses to the ask, a
SELL crosses to the bid, the configured slippage moves both further against the
order, and every way the inputs can be untrustworthy — no book, an old book, a
crossed book, a book that disagrees with the price the decision was made at —
ends in a refusal rather than in a price. Second, that
:meth:`DryRunFuturesVenue.submit` refuses a position side the symbol does not
support before any fill reaches its position, which is what stops the demo's
spot leg being sold short.

The numbers here are pinned rather than recomputed by the assertions. A test that
recomputes the implementation's arithmetic passes whatever that arithmetic is;
these are hand-traceable, and the trace is in the docstring of each test that has
one.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import pytest

from chimera.futures import (
    NO_FRESH_QUOTE,
    QUOTE_REFERENCE_DIVERGENCE,
    ConstraintError,
    DeterministicFillModel,
    DryRunFuturesVenue,
    EventKind,
    OrderIntent,
    OrderPurpose,
    OrderSide,
    PositionSide,
    RecordedQuoteFillModel,
    StaticConstraintSource,
    SymbolConstraints,
    TopOfBook,
    default_constraints_table,
)

PERP = "BTC/USDT:USDT"
SPOT = "BTC/USDT"
ZERO = Decimal("0")

#: The instant every book in this file is stamped with. Round, so an age is
#: readable as the offset it is.
INSTANT_NS = 1_700_000_000_000_000_000

#: The decision minute's price. The books below straddle it, so a fill price that
#: came from the reference rather than from the book is visible as such.
REFERENCE = Decimal("60000")


#: The spot leg's constraints, from section 5.3 of the demo plan. The plan fixes
#: the symbol, step size, minimum notional, both fee rates, the supported
#: position sides and the maintenance margin rate; the tick size, the precisions
#: and the minimum quantity are Binance's published BTCUSDT spot filters, needed
#: because ``SymbolConstraints.from_dict`` refuses metadata that is missing any
#: of them.
#:
#: Built here rather than added to ``default_constraints_table()`` deliberately:
#: that table is the committed constraint set the futures dry-run validation
#: protocol runs against, and it holds exactly one symbol — the perpetual. Adding
#: a spot symbol to it would put a second, differently-shaped market inside the
#: protocol's own constraint source, which is not what PR-07 is for.
SPOT_CONSTRAINTS = {
    "symbol": SPOT,
    "status": "TRADING",
    "tick_size": "0.01",
    "step_size": "0.00001",
    "quantity_precision": 5,
    "price_precision": 2,
    "min_quantity": "0.00001",
    "min_notional": "10",
    "maintenance_margin_rate": "0.004",
    "taker_fee_rate": "0.001",
    "maker_fee_rate": "0.001",
    "supported_order_types": ["MARKET"],
    "supports_reduce_only": True,
    "supported_position_sides": ["LONG"],
}


def perp_constraints(**overrides) -> SymbolConstraints:
    """The committed perpetual constraints, with fields swapped out."""
    fields = dict(default_constraints_table()[PERP])
    fields["symbol"] = PERP
    fields.update(overrides)
    return SymbolConstraints.from_dict(fields)


def spot_constraints(**overrides) -> SymbolConstraints:
    """The demo's spot leg: 1e-5 steps, 10 USDT minimum, LONG only."""
    return SymbolConstraints.from_dict({**SPOT_CONSTRAINTS, **overrides})


def book(
    bid: str = "59990.00",
    ask: str = "60010.00",
    *,
    instant_ns: int = INSTANT_NS,
    bid_qty: str = "3.5",
    ask_qty: str = "2.5",
) -> TopOfBook:
    """A well-formed snapshot straddling :data:`REFERENCE` by ten quote units."""
    return TopOfBook(
        instant_ns=instant_ns,
        bid=Decimal(bid),
        bid_qty=Decimal(bid_qty),
        ask=Decimal(ask),
        ask_qty=Decimal(ask_qty),
    )


def model(quote: TopOfBook | None = None, *, age_ns: int = 0, **overrides):
    """A fill model holding ``quote``, seen ``age_ns`` after the book was taken."""
    filler = RecordedQuoteFillModel(**overrides)
    if quote is not None:
        filler.set_quote(quote, quote.instant_ns + age_ns)
    return filler


def intent(
    side: OrderSide,
    quantity: str = "0.500",
    *,
    symbol: str = PERP,
    position_side: PositionSide | None = None,
) -> OrderIntent:
    """A well-formed opening order. ``OrderIntent`` checks its own coherence."""
    if position_side is None:
        position_side = PositionSide.LONG if side is OrderSide.BUY else PositionSide.SHORT
    return OrderIntent(
        symbol=symbol,
        side=side,
        quantity=Decimal(quantity),
        purpose=OrderPurpose.OPEN,
        reduce_only=False,
        position_side=position_side,
    )


def venue(fill_model=None, table=None) -> DryRunFuturesVenue:
    """A fresh simulated venue over the perpetual and the spot symbol."""
    fields = dict(default_constraints_table()[PERP])
    source = StaticConstraintSource.from_mapping(
        table or {PERP: fields, SPOT: dict(SPOT_CONSTRAINTS)}
    )
    return DryRunFuturesVenue(source=source, fill_model=fill_model or model(book()))


@dataclass
class RecordingFillModel:
    """A fill model that records being asked, so "never asked" is assertable."""

    asked: list = None  # type: ignore[assignment]

    def plan(self, intent, reference_price, constraints):
        if self.asked is None:
            self.asked = []
        self.asked.append(intent)
        return RecordedQuoteFillModel().plan(intent, reference_price, constraints)


# --- the snapshot itself ----------------------------------------------------
def test_a_well_formed_book_reports_its_mid_and_spread():
    """The two derived numbers, hand-traced.

    bid 59990 and ask 60010 give a mid of 60000 and a spread of 20, which is
    20 / 60000 * 10000 = 3.333... basis points.
    """
    quote = book()
    assert quote.mid() == Decimal("60000")
    assert quote.spread_bps() == pytest.approx(Decimal("3.3333333"), abs=1e-6)
    assert quote.instant_ns == INSTANT_NS


def test_a_crossed_book_is_refused_and_a_book_one_tick_wide_is_not():
    """A bid above an ask is two instants glued together, not a market.

    Two-sided on the exact boundary the rule draws: one tick of separation is a
    book, none at all is not.
    """
    narrow = book(bid="60009.90", ask="60010.00")
    assert narrow.mid() == Decimal("60009.95")

    with pytest.raises(ConstraintError) as excinfo:
        book(bid="60010.10", ask="60010.00")
    assert "is not below ask" in str(excinfo.value)


def test_a_locked_book_is_refused_because_it_could_not_have_stood():
    """A matching engine cannot leave a bid resting at the ask; they would trade."""
    with pytest.raises(ConstraintError) as excinfo:
        book(bid="60010.00", ask="60010.00")
    assert "bid 60010.00 is not below ask 60010.00" in str(excinfo.value)


@pytest.mark.parametrize(
    "field,value",
    [
        ("bid", "0"),
        ("bid", "-1"),
        ("ask", "0"),
        ("bid_qty", "0"),
        ("ask_qty", "-0.5"),
    ],
)
def test_a_non_positive_side_or_size_is_refused(field, value):
    """An empty or negative level is not a level an order could have crossed."""
    fields = {
        "instant_ns": INSTANT_NS,
        "bid": Decimal("59990"),
        "bid_qty": Decimal("3"),
        "ask": Decimal("60010"),
        "ask_qty": Decimal("2"),
        field: Decimal(value),
    }
    with pytest.raises(ConstraintError) as excinfo:
        TopOfBook(**fields)
    assert field in str(excinfo.value)


def test_a_price_that_arrived_as_a_float_is_refused_rather_than_coerced():
    """Catches a JSON reader handing the model a float and losing precision silently.

    A float also has no ``is_finite`` method, so without the type check the
    failure would surface from inside the arithmetic instead of here.
    """
    with pytest.raises(ConstraintError) as excinfo:
        TopOfBook(
            instant_ns=INSTANT_NS,
            bid=60000.0,  # type: ignore[arg-type]
            bid_qty=Decimal("1"),
            ask=Decimal("60010"),
            ask_qty=Decimal("1"),
        )
    assert "not a Decimal" in str(excinfo.value)


def test_a_nan_price_is_refused_as_a_constraint_error():
    """NaN signals rather than compares, so it has to be excluded by name.

    Without the finiteness check this raises ``InvalidOperation`` from the
    comparison — past every caller that catches ``ConstraintError``.
    """
    with pytest.raises(ConstraintError) as excinfo:
        TopOfBook(
            instant_ns=INSTANT_NS,
            bid=Decimal("NaN"),
            bid_qty=Decimal("1"),
            ask=Decimal("60010"),
            ask_qty=Decimal("1"),
        )
    assert "not a finite number" in str(excinfo.value)


def test_a_float_instant_is_refused_because_it_would_decide_the_freshness_bound():
    """A float clock is not a clock at nanosecond-epoch magnitude.

    ``1_700_000_000_120_000_001`` is one nanosecond past the default 120 s bound,
    so a book stamped with it must be refused as unfresh. As a float it is
    indistinguishable from the bound itself — the nearest representable value is
    256 ns away — so an unchecked float would have quietly turned a refusal into
    a fill. The first assertion is that arithmetic, so the test fails if the
    magnitudes ever stop making the point; the second is the refusal.
    """
    one_past_the_bound = INSTANT_NS + 120_000_000_001
    assert int(float(one_past_the_bound)) - INSTANT_NS == 120_000_000_000

    with pytest.raises(ConstraintError) as excinfo:
        TopOfBook(
            instant_ns=float(INSTANT_NS),  # type: ignore[arg-type]
            bid=Decimal("59990"),
            bid_qty=Decimal("1"),
            ask=Decimal("60010"),
            ask_qty=Decimal("1"),
        )
    assert "instant_ns" in str(excinfo.value)
    assert "not a whole number of nanoseconds" in str(excinfo.value)


# --- freshness --------------------------------------------------------------
def test_no_book_at_all_is_a_rejection_and_not_a_fill():
    """A model that was never handed a book must not invent one."""
    plan = RecordedQuoteFillModel().plan(intent(OrderSide.BUY), REFERENCE, perp_constraints())
    assert plan.fills == ()
    assert plan.filled_quantity == ZERO
    assert plan.rejection == NO_FRESH_QUOTE == "no_fresh_quote"


@pytest.mark.parametrize(
    "age_ns,fresh",
    [
        (0, True),
        (119_999_999_999, True),
        (120_000_000_000, True),
        (120_000_000_001, False),
        (600_000_000_000, False),
    ],
)
def test_the_freshness_bound_is_inclusive_to_the_nanosecond(age_ns, fresh):
    """Exactly ``max_quote_age_ns`` old is fresh; one nanosecond older is not.

    The default bound is 120 s, so the three cases around it are 119.999999999 s,
    120 s and 120.000000001 s.
    """
    plan = model(book(), age_ns=age_ns).plan(
        intent(OrderSide.BUY), REFERENCE, perp_constraints()
    )
    if fresh:
        assert plan.rejection == ""
        assert plan.filled_quantity == Decimal("0.500")
    else:
        assert plan.rejection == NO_FRESH_QUOTE
        assert plan.fills == ()


def test_a_book_dated_after_the_clock_is_not_fresh_either():
    """Catches "age <= max" being read as the whole test.

    A negative age means the clock went backwards or the wrong minute's snapshot
    was installed. Either way it is not evidence of the current market, and an
    unsigned reading of the bound would have filled against it.
    """
    plan = model(book(), age_ns=-1).plan(intent(OrderSide.BUY), REFERENCE, perp_constraints())
    assert plan.rejection == NO_FRESH_QUOTE
    assert plan.fills == ()


def test_set_quote_moves_the_book_and_the_clock_together():
    """A book installed without moving the clock would be stale by construction."""
    filler = RecordedQuoteFillModel()
    assert filler.quote is None and filler.now_ns == 0

    first = book()
    filler.set_quote(first, INSTANT_NS + 5)
    assert filler.quote == first and filler.now_ns == INSTANT_NS + 5

    second = book(bid="60100.00", ask="60110.00", instant_ns=INSTANT_NS + 60_000_000_000)
    filler.set_quote(second, second.instant_ns)
    assert filler.quote == second and filler.now_ns == second.instant_ns


def test_the_clock_ages_the_installed_book_without_a_new_one_arriving():
    """The caller's half of the freshness rule, pinned on both sides.

    ``now_ns`` moves only when the caller moves it, and :meth:`set_quote` always
    replaces the book as well, so a runner that installs a book only on the
    minutes one arrived can never age the one it already holds. Both halves are
    fixed here: a clock advanced past the bound with no new book must refuse the
    old one, and — the hazard, recorded rather than described — a clock left
    where it was keeps filling from the same book however many cycles pass.
    """
    filler = model(book())
    limits = perp_constraints()
    order = intent(OrderSide.BUY)

    # The clock is public precisely so a cycle with no book can move it.
    filler.now_ns = INSTANT_NS + 120_000_000_001
    stale = filler.plan(order, REFERENCE, limits)
    assert stale.rejection == NO_FRESH_QUOTE
    assert stale.fills == ()

    filler.now_ns = INSTANT_NS + 120_000_000_000
    assert filler.plan(order, REFERENCE, limits).rejection == ""

    # And the hazard: a frozen clock is a book that never ages.
    frozen = model(book())
    prices = [frozen.plan(order, REFERENCE, limits).fills[0][1] for _ in range(5)]
    assert prices == [Decimal("60022.00")] * 5


# --- which side of the book, and in which direction -------------------------
def test_a_buy_fills_at_the_ask_and_a_sell_at_the_bid_with_no_slippage():
    """The price comes from the touch, not from the reference or the mid.

    With slippage switched off the two fills are the recorded ask and the
    recorded bid exactly. The reference is 60000 and the mid is 60000 too, so a
    price derived from either would be 60000 and neither assertion below would
    hold.
    """
    limits = perp_constraints()
    quote = book()
    bought = model(quote, slippage_bps=ZERO).plan(intent(OrderSide.BUY), REFERENCE, limits)
    sold = model(quote, slippage_bps=ZERO).plan(intent(OrderSide.SELL), REFERENCE, limits)

    assert bought.fills == ((Decimal("0.500"), Decimal("60010.00")),)
    assert sold.fills == ((Decimal("0.500"), Decimal("59990.00")),)
    assert bought.fills[0][1] == quote.ask
    assert sold.fills[0][1] == quote.bid


def test_slippage_is_adverse_on_both_sides_and_on_top_of_the_crossing():
    """Hand-traced at the default 2 bps.

    BUY: 60010 * 1.0002 = 60022.002, quantized to the 0.10 tick as 60022.00 —
    above the ask it crossed. SELL: 59990 * 0.9998 = 59978.002, quantized as
    59978.00 — below the bid. Both are further from the mid than the touch was,
    which is what "on top of the spread" means.
    """
    limits = perp_constraints()
    quote = book()
    bought = model(quote).plan(intent(OrderSide.BUY), REFERENCE, limits)
    sold = model(quote).plan(intent(OrderSide.SELL), REFERENCE, limits)

    assert bought.fills == ((Decimal("0.500"), Decimal("60022.00")),)
    assert sold.fills == ((Decimal("0.500"), Decimal("59978.00")),)
    assert bought.fills[0][1] > quote.ask > quote.mid()
    assert sold.fills[0][1] < quote.bid < quote.mid()


def test_more_slippage_is_strictly_worse_on_both_sides():
    """Catches the sign of the adjustment being right only by coincidence."""
    limits = perp_constraints()
    quote = book()
    cheap_buy = model(quote, slippage_bps=Decimal("2")).plan(
        intent(OrderSide.BUY), REFERENCE, limits
    )
    dear_buy = model(quote, slippage_bps=Decimal("20")).plan(
        intent(OrderSide.BUY), REFERENCE, limits
    )
    good_sell = model(quote, slippage_bps=Decimal("2")).plan(
        intent(OrderSide.SELL), REFERENCE, limits
    )
    bad_sell = model(quote, slippage_bps=Decimal("20")).plan(
        intent(OrderSide.SELL), REFERENCE, limits
    )

    assert dear_buy.fills[0][1] > cheap_buy.fills[0][1]
    assert bad_sell.fills[0][1] < good_sell.fills[0][1]


def test_a_negative_slippage_is_refused_rather_than_improving_the_fill():
    """A model that can beat the book costs less than the venue it stands for."""
    filler = model(book(), slippage_bps=Decimal("-2"))
    with pytest.raises(ConstraintError) as excinfo:
        filler.plan(intent(OrderSide.BUY), REFERENCE, perp_constraints())
    assert "negative" in str(excinfo.value)

    filler.slippage_bps = ZERO
    assert filler.plan(intent(OrderSide.BUY), REFERENCE, perp_constraints()).rejection == ""


# --- the venue's price grid -------------------------------------------------
def test_the_fill_price_is_quantized_to_the_venues_tick():
    """60000.05 * 1.0002 = 60012.05001, which is not a price this venue quotes.

    On the 0.10 tick grid it rounds to 60012.10 — up, away from the order, and
    still above the ask it crossed.
    """
    plan = model(book(bid="59990.00", ask="60000.05")).plan(
        intent(OrderSide.BUY), REFERENCE, perp_constraints()
    )
    assert plan.fills == ((Decimal("0.500"), Decimal("60012.10")),)
    assert plan.fills[0][1] % perp_constraints().tick_size == ZERO


def test_a_quantized_fill_better_than_the_touch_is_refused_rather_than_booked():
    """Rounding to the nearest tick can move a price down; that fill is refused.

    With an ask of 60000.04 on a 0.10 grid and no slippage, the nearest tick is
    60000.00 — a BUY filled *better* than the ask it was supposed to cross. A
    fill better than the book is the one direction this package will not
    simulate, so it raises instead of quietly handing back the cheaper price.
    """
    with pytest.raises(ConstraintError) as excinfo:
        model(book(bid="59990.00", ask="60000.04"), slippage_bps=ZERO).plan(
            intent(OrderSide.BUY), REFERENCE, perp_constraints()
        )
    assert "below the ask it crossed" in str(excinfo.value)

    with pytest.raises(ConstraintError) as excinfo:
        model(book(bid="59999.96", ask="60010.00"), slippage_bps=ZERO).plan(
            intent(OrderSide.SELL), REFERENCE, perp_constraints()
        )
    assert "above the bid it crossed" in str(excinfo.value)


def test_the_same_book_prices_correctly_on_the_grid_that_does_match_it():
    """The other half of the pair above: 0.01 ticks accept what 0.10 ticks refuse."""
    plan = model(book(bid="59990.00", ask="60000.04"), slippage_bps=ZERO).plan(
        intent(OrderSide.BUY, "0.10000", symbol=SPOT), REFERENCE, spot_constraints()
    )
    assert plan.fills == ((Decimal("0.10000"), Decimal("60000.04")),)


def test_the_touch_guard_is_not_a_cross_symbol_wiring_check():
    """The documented limit of the guard above, pinned so nobody relies on more.

    A snapshot carries no symbol, so nothing here can compare the book against
    ``constraints.symbol``; pairing one venue with one book is the caller's job.
    Two mis-wired combinations therefore price silently. A perpetual book on the
    spot leg's finer 0.01 grid is never caught, because every 0.10-grid price is
    also on the 0.01 grid. And a spot book on the coarser 0.10 grid is caught
    only when the rounding goes *down*: 60000.04 raises above, while 60000.06
    rounds up to 60000.10 and fills.
    """
    perp_book_on_spot_grid = model(book(), slippage_bps=ZERO).plan(
        intent(OrderSide.BUY, "0.10000", symbol=SPOT), REFERENCE, spot_constraints()
    )
    assert perp_book_on_spot_grid.fills == ((Decimal("0.10000"), Decimal("60010.00")),)

    rounds_away_from_the_order = model(
        book(bid="59990.00", ask="60000.06"), slippage_bps=ZERO
    ).plan(intent(OrderSide.BUY), REFERENCE, perp_constraints())
    assert rounds_away_from_the_order.fills == ((Decimal("0.500"), Decimal("60000.10")),)


# --- the divergence guard ---------------------------------------------------
@pytest.mark.parametrize(
    "ask,bid,accepted",
    [
        ("60299.90", "60299.80", True),
        ("60300.00", "60299.90", True),
        ("60300.10", "60300.00", False),
    ],
)
def test_a_buy_diverging_from_the_reference_is_refused_only_past_the_boundary(
    ask, bid, accepted
):
    """50 bps of 60000 is exactly 300, so 60300.00 is the last accepted fill.

    "Exceeds" is strict: a fill exactly at the tolerance is filled, and the next
    tick past it is not. Slippage is off so the fill price is the ask itself and
    the boundary is readable.
    """
    plan = model(book(bid=bid, ask=ask), slippage_bps=ZERO).plan(
        intent(OrderSide.BUY), REFERENCE, perp_constraints()
    )
    if accepted:
        assert plan.rejection == ""
        assert plan.fills[0][1] == Decimal(ask)
    else:
        assert plan.rejection == QUOTE_REFERENCE_DIVERGENCE == "quote_reference_divergence"
        assert plan.fills == ()


@pytest.mark.parametrize(
    "bid,ask,accepted",
    [
        ("59700.10", "59700.20", True),
        ("59700.00", "59700.10", True),
        ("59699.90", "59700.00", False),
    ],
)
def test_a_sell_diverging_below_the_reference_is_guarded_the_same_way(bid, ask, accepted):
    """The guard is on the absolute deviation, so it is symmetric.

    A book 300 below the reference is exactly at the tolerance; 300.10 below is
    past it.
    """
    plan = model(book(bid=bid, ask=ask), slippage_bps=ZERO).plan(
        intent(OrderSide.SELL), REFERENCE, perp_constraints()
    )
    if accepted:
        assert plan.rejection == ""
        assert plan.fills[0][1] == Decimal(bid)
    else:
        assert plan.rejection == QUOTE_REFERENCE_DIVERGENCE
        assert plan.fills == ()


def test_the_divergence_guard_measures_the_fill_price_and_not_the_touch():
    """Slippage is part of what is booked, so it is part of what is checked.

    The ask is 60299.90, which is 299.90 from the reference and inside the 50 bps
    tolerance. Two basis points of slippage put the fill at 60312.00 — 312 away,
    which is not. A guard applied to the touch would have let this through.
    """
    quote = book(bid="60299.80", ask="60299.90")
    at_the_touch = model(quote, slippage_bps=ZERO).plan(
        intent(OrderSide.BUY), REFERENCE, perp_constraints()
    )
    assert at_the_touch.rejection == ""
    assert at_the_touch.fills[0][1] == Decimal("60299.90")

    plan = model(quote).plan(intent(OrderSide.BUY), REFERENCE, perp_constraints())
    assert plan.fills == ()
    assert plan.rejection == QUOTE_REFERENCE_DIVERGENCE


def test_against_the_books_own_mid_the_guard_is_a_spread_ceiling_and_nothing_more():
    """What the guard does and does not catch under section 5.2's own wiring.

    Section 5.2 has the runner pass ``reference_price = quote.mid()``. The fill
    price and the reference are then two numbers from the same snapshot, so the
    deviation is exactly half the quoted spread plus the configured slippage:
    the rule is a ceiling on how wide a book this model will cross, and it cannot
    see a disagreement between two series because it is only ever handed one.

    Both consequences are pinned. A tight book that is wrong by half — 30000.05
    mid where the market was at 60000 — is accepted against its own mid and fills
    at 30006.10, so a stale or corrupt book is *not* what this guard catches. The
    same book measured against an independent 60000 reference is refused, which
    is the check section 5.2 describes and which the runner only gets by passing
    a reference from a different series, such as the recorded mark close.
    """
    wrong_but_tight = book(bid="30000.00", ask="30000.10")
    filler = model(wrong_but_tight)
    limits = perp_constraints()

    against_itself = filler.plan(intent(OrderSide.BUY), wrong_but_tight.mid(), limits)
    assert against_itself.rejection == ""
    assert against_itself.fills == ((Decimal("0.500"), Decimal("30006.10")),)

    against_another_series = filler.plan(intent(OrderSide.BUY), REFERENCE, limits)
    assert against_another_series.rejection == QUOTE_REFERENCE_DIVERGENCE
    assert against_another_series.fills == ()

    # And the converse: a legitimately wide book is what the mid-referenced guard
    # actually refuses, so the reason string reports the spread, not a stale feed.
    wide = book(bid="59000.00", ask="61000.00")
    plan = model(wide).plan(intent(OrderSide.BUY), wide.mid(), limits)
    assert plan.rejection == QUOTE_REFERENCE_DIVERGENCE


def test_a_wider_tolerance_accepts_the_same_book_the_default_refuses():
    """The guard is the configured number, not a hard-coded one."""
    quote = book(bid="60300.00", ask="60300.10")
    strict = model(quote, slippage_bps=ZERO)
    lenient = model(quote, slippage_bps=ZERO, max_reference_deviation_bps=Decimal("51"))

    assert strict.plan(intent(OrderSide.BUY), REFERENCE, perp_constraints()).rejection == (
        QUOTE_REFERENCE_DIVERGENCE
    )
    assert lenient.plan(intent(OrderSide.BUY), REFERENCE, perp_constraints()).rejection == ""


# --- the reference price itself ---------------------------------------------
@pytest.mark.parametrize("reference", [Decimal("0"), Decimal("-60000"), Decimal("NaN")])
def test_a_reference_price_that_is_not_a_positive_number_raises(reference):
    """A missing mark is missing, not zero, and a fill against it is invented.

    This raises rather than rejecting: a rejection reason is a statement about
    the market, and a caller that lost its reference price has a defect that
    would otherwise be filed as one.
    """
    with pytest.raises(ConstraintError) as excinfo:
        model(book()).plan(intent(OrderSide.BUY), reference, perp_constraints())
    assert "reference_price" in str(excinfo.value)


def test_the_reference_price_is_checked_before_the_book_is_looked_at():
    """Precedence, pinned: a caller defect must not be reported as a stale market.

    With no book installed *and* no reference price, the answer is the exception,
    not the ``no_fresh_quote`` rejection.
    """
    with pytest.raises(ConstraintError):
        RecordedQuoteFillModel().plan(intent(OrderSide.BUY), ZERO, perp_constraints())


# --- quantity chunking ------------------------------------------------------
def test_a_whole_order_arrives_as_one_fill_unless_faults_are_injected():
    """The v1 default: no partial fill is reachable without ``max_fill_ratio``."""
    plan = model(book()).plan(intent(OrderSide.BUY, "12.345"), REFERENCE, perp_constraints())
    assert len(plan.fills) == 1
    assert plan.filled_quantity == Decimal("12.345")


def test_a_capped_fill_ratio_splits_the_order_into_fills_that_sum_exactly():
    """Fault injection, and the arithmetic must neither lose nor invent quantity."""
    plan = model(book(), max_fill_ratio=Decimal("0.4")).plan(
        intent(OrderSide.BUY, "1.000"), REFERENCE, perp_constraints()
    )
    assert [quantity for quantity, _ in plan.fills] == [
        Decimal("0.400"),
        Decimal("0.400"),
        Decimal("0.200"),
    ]
    assert plan.filled_quantity == Decimal("1.000")
    assert {price for _, price in plan.fills} == {Decimal("60022.00")}
    assert plan.rejection == ""


def test_a_trailing_sliver_smaller_than_a_step_is_carried_into_the_fill_before_it():
    """A remainder off the step grid is not a fill the venue would accept.

    0.0021 at half an order per fill gives a 0.001 chunk: 0.001 then 0.0011,
    because leaving 0.0001 behind would leave a fill that is not a multiple of
    the 0.001 step.
    """
    plan = model(book(), max_fill_ratio=Decimal("0.5")).plan(
        intent(OrderSide.BUY, "0.0021"), REFERENCE, perp_constraints()
    )
    assert [quantity for quantity, _ in plan.fills] == [Decimal("0.001"), Decimal("0.0011")]
    assert plan.filled_quantity == Decimal("0.0021")


def test_a_chunk_that_quantizes_away_becomes_the_whole_order():
    """A ratio below one step must not loop forever on a zero-sized fill."""
    plan = model(book(), max_fill_ratio=Decimal("0.0001")).plan(
        intent(OrderSide.BUY, "0.500"), REFERENCE, perp_constraints()
    )
    assert plan.fills == ((Decimal("0.500"), Decimal("60022.00")),)


@pytest.mark.parametrize("quantity", ["1.000", "0.0021", "0.500", "7.777"])
@pytest.mark.parametrize("ratio", ["1", "0.5", "0.4", "0.3333"])
def test_the_two_fill_models_split_an_order_identically(quantity, ratio):
    """The chunking rule is shared by contract, so it is asserted rather than assumed.

    A scenario replayed under the deterministic model and then under this one has
    to produce the same event sequence, or a difference in duplicate-event
    handling would be indistinguishable from a difference in chunking. Only the
    quantities are compared: the prices are meant to differ, and that difference
    is this module's whole point.
    """
    limits = perp_constraints()
    recorded = model(book(), max_fill_ratio=Decimal(ratio)).plan(
        intent(OrderSide.BUY, quantity), REFERENCE, limits
    )
    deterministic = DeterministicFillModel(max_fill_ratio=Decimal(ratio)).plan(
        intent(OrderSide.BUY, quantity), REFERENCE, limits
    )
    assert [q for q, _ in recorded.fills] == [q for q, _ in deterministic.fills]
    assert recorded.filled_quantity == deterministic.filled_quantity == Decimal(quantity)


@pytest.mark.parametrize("ratio", ["0", "-0.5", "1.5"])
def test_a_fill_ratio_outside_the_unit_interval_is_refused(ratio):
    """Zero would fill nothing forever and more than one would fill more than asked."""
    with pytest.raises(ConstraintError) as excinfo:
        model(book(), max_fill_ratio=Decimal(ratio)).plan(
            intent(OrderSide.BUY), REFERENCE, perp_constraints()
        )
    assert "max_fill_ratio" in str(excinfo.value)


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_quote_age_ns", -1),
        ("max_quote_age_ns", 1.5),
        ("max_reference_deviation_bps", Decimal("-1")),
        ("slippage_bps", Decimal("10000")),
        ("slippage_bps", Decimal("NaN")),
        ("max_fill_ratio", Decimal("NaN")),
        ("now_ns", "1700000000000000000"),
        ("now_ns", 1.7e18),
        ("now_ns", True),
    ],
)
def test_a_setting_outside_its_own_domain_is_refused_at_plan_time(field, value):
    """Checked in ``plan``, because the fields are mutable and public.

    A constructor check would be bypassed by writing the field afterwards, which
    is exactly how a model gets reconfigured between two minutes of a run.

    Every case here must reach the caller as a ``ConstraintError`` and not as
    whatever the arithmetic would have raised on its own: a NaN ratio signals
    ``decimal.InvalidOperation`` from the range comparison and a string clock
    raises ``TypeError`` from the subtraction, and the executor catches neither,
    so either one takes a run down instead of recording one failed order.
    """
    filler = model(book())
    setattr(filler, field, value)
    with pytest.raises(ConstraintError) as excinfo:
        filler.plan(intent(OrderSide.BUY), REFERENCE, perp_constraints())
    assert field in str(excinfo.value)


# --- determinism ------------------------------------------------------------
def test_identical_inputs_produce_an_identical_plan():
    """Replay is a check only while the model answers the same question the same way."""
    limits = perp_constraints()
    quote = book()
    order = intent(OrderSide.BUY, "1.000")

    first = model(quote, max_fill_ratio=Decimal("0.4"))
    second = model(quote, max_fill_ratio=Decimal("0.4"))
    plans = [first.plan(order, REFERENCE, limits) for _ in range(3)]
    plans.append(second.plan(order, REFERENCE, limits))

    assert len({repr(plan) for plan in plans}) == 1
    assert plans[0] == plans[-1]


# --- through the venue ------------------------------------------------------
def test_the_venue_charges_the_taker_rate_on_every_fill_and_never_a_maker_rebate():
    """Hand-traced fees, and the maker rate is a different number on purpose.

    One BUY of 0.500 at 60022.00 is 30011.00 of notional. At the perpetual's
    0.0005 taker rate that is 15.0055; at the 0.0002 maker rate it would be
    6.0022, and a rebate would be negative. The fee is computed by the venue, not
    by the fill model: the plan carries prices and quantities only.
    """
    limits = perp_constraints()
    market = venue()
    events = market.submit("ORD-1", intent(OrderSide.BUY), REFERENCE)
    fills = [e for e in events if e.kind in (EventKind.PARTIAL_FILL, EventKind.FILL)]

    assert [event.kind for event in events] == [EventKind.ACKNOWLEDGED, EventKind.FILL]
    assert fills[0].price == Decimal("60022.00")
    assert fills[0].quantity == Decimal("0.500")
    assert fills[0].fee == Decimal("15.0055")
    assert fills[0].fee == fills[0].quantity * fills[0].price * limits.taker_fee_rate
    assert fills[0].fee != fills[0].quantity * fills[0].price * limits.maker_fee_rate
    assert fills[0].fee > ZERO


def test_a_split_order_pays_the_taker_rate_on_each_fill_and_nothing_extra():
    """Fees are per fill; the same order split three ways pays the same total.

    0.400 and 0.400 and 0.200 at 60022.00: 12.0044, 12.0044 and 6.0022, summing
    to 30.011 — which is 1.000 * 60022.00 * 0.0005.
    """
    market = venue(fill_model=model(book(), max_fill_ratio=Decimal("0.4")))
    events = market.submit("ORD-1", intent(OrderSide.BUY, "1.000"), REFERENCE)
    fills = [e for e in events if e.kind in (EventKind.PARTIAL_FILL, EventKind.FILL)]

    assert [event.fee for event in fills] == [
        Decimal("12.0044"),
        Decimal("12.0044"),
        Decimal("6.0022"),
    ]
    assert sum(event.fee for event in fills) == Decimal("30.011")
    assert market.reported_position(PERP).quantity == Decimal("1.000")


def test_a_sell_pays_the_same_rate_on_the_bid_side():
    """0.500 at 59978.00 is 29989.00 of notional and 14.9945 of taker fee."""
    market = venue()
    events = market.submit("ORD-1", intent(OrderSide.SELL), REFERENCE)
    fill = events[-1]

    assert fill.kind is EventKind.FILL
    assert fill.price == Decimal("59978.00")
    assert fill.fee == Decimal("14.9945")
    assert market.reported_position(PERP).side is PositionSide.SHORT


def test_a_stale_book_reaches_the_caller_as_an_acknowledged_rejection():
    """The rejection reason is what the runner's telemetry records, so it is pinned."""
    market = venue(fill_model=model(book(), age_ns=120_000_000_001))
    events = market.submit("ORD-1", intent(OrderSide.BUY), REFERENCE)

    assert [event.kind for event in events] == [EventKind.ACKNOWLEDGED, EventKind.REJECTED]
    assert events[1].reason == "no_fresh_quote"
    assert market.positions == {}
    assert market.reported_position(PERP).side is PositionSide.FLAT


def test_a_divergent_book_reaches_the_caller_as_an_acknowledged_rejection():
    """The other rejection this model can produce, with its own reason string."""
    market = venue(fill_model=model(book(bid="60300.00", ask="60300.10"), slippage_bps=ZERO))
    events = market.submit("ORD-1", intent(OrderSide.BUY), REFERENCE)

    assert [event.kind for event in events] == [EventKind.ACKNOWLEDGED, EventKind.REJECTED]
    assert events[1].reason == "quote_reference_divergence"
    assert market.positions == {}


# --- the side check ---------------------------------------------------------
def test_a_supported_side_fills_and_moves_the_venues_position():
    """The half of the side check that has to keep working.

    The perpetual supports both sides, so neither a LONG nor a SHORT is refused
    for its side; without this the rejection tests below would pass on a venue
    that refused everything.
    """
    market = venue()
    long_events = market.submit("ORD-1", intent(OrderSide.BUY, "0.500"), REFERENCE)
    assert [event.kind for event in long_events] == [EventKind.ACKNOWLEDGED, EventKind.FILL]
    assert market.reported_position(PERP).side is PositionSide.LONG

    other = venue()
    short_events = other.submit("ORD-2", intent(OrderSide.SELL, "0.500"), REFERENCE)
    assert [event.kind for event in short_events] == [EventKind.ACKNOWLEDGED, EventKind.FILL]
    assert other.reported_position(PERP).side is PositionSide.SHORT


def test_a_spot_short_is_rejected_and_books_no_exposure():
    """The reason the check exists: the demo's spot leg is LONG-only.

    A SELL that opens a SHORT in a symbol that cannot be sold short must not
    reach the venue's position, because the venue's position is what
    reconciliation compares local state against — the exposure would have been
    real there and only a later reconcile would have found it.
    """
    market = venue()
    events = market.submit(
        "ORD-1",
        intent(OrderSide.SELL, "0.10000", symbol=SPOT, position_side=PositionSide.SHORT),
        REFERENCE,
    )

    assert [event.kind for event in events] == [EventKind.ACKNOWLEDGED, EventKind.REJECTED]
    assert "does not hold a SHORT position in BTC/USDT" in events[1].reason
    assert "['LONG']" in events[1].reason
    assert market.positions == {}
    assert market.reported_position(SPOT).side is PositionSide.FLAT
    assert market.reported_position(SPOT).quantity == ZERO


def test_a_spot_long_is_accepted_by_the_same_venue_that_refused_the_short():
    """Two-sided against the test above: the symbol is LONG-only, not untradable.

    Hand-traced: 0.10000 at an ask of 60010.00 plus 2 bps is 60022.002, which on
    the spot leg's 0.01 tick is 60022.00. The notional is 6002.20 and the spot
    taker rate is 0.001, so the fee is 6.0022.
    """
    market = venue()
    events = market.submit(
        "ORD-1",
        intent(OrderSide.BUY, "0.10000", symbol=SPOT, position_side=PositionSide.LONG),
        REFERENCE,
    )

    assert [event.kind for event in events] == [EventKind.ACKNOWLEDGED, EventKind.FILL]
    assert events[1].price == Decimal("60022.00")
    assert events[1].quantity == Decimal("0.10000")
    assert events[1].fee == Decimal("6.0022")
    assert market.reported_position(SPOT).side is PositionSide.LONG
    assert market.reported_position(SPOT).quantity == Decimal("0.10000")


def test_a_perpetual_restricted_to_one_side_refuses_the_other():
    """The check reads the constraints, not a hard-coded list of spot symbols."""
    fields = {**default_constraints_table()[PERP], "supported_position_sides": ["SHORT"]}
    table = {PERP: fields}
    market = venue(table=table)

    refused = market.submit("ORD-1", intent(OrderSide.BUY), REFERENCE)
    assert [event.kind for event in refused] == [EventKind.ACKNOWLEDGED, EventKind.REJECTED]
    assert "does not hold a LONG position" in refused[1].reason

    allowed = market.submit("ORD-2", intent(OrderSide.SELL), REFERENCE)
    assert [event.kind for event in allowed] == [EventKind.ACKNOWLEDGED, EventKind.FILL]


def test_the_side_check_runs_before_the_fill_model_is_asked_for_a_price():
    """An order the venue will never accept must not be priced at all.

    Otherwise a spot SHORT submitted on a stale book would be reported as
    ``no_fresh_quote`` — a market condition an operator would wait out — instead
    of as the structural refusal it is.
    """
    recording = RecordingFillModel()
    market = venue(fill_model=recording)
    events = market.submit(
        "ORD-1",
        intent(OrderSide.SELL, "0.10000", symbol=SPOT, position_side=PositionSide.SHORT),
        REFERENCE,
    )

    assert events[1].kind is EventKind.REJECTED
    assert recording.asked is None

    market.submit(
        "ORD-2",
        intent(OrderSide.BUY, "0.10000", symbol=SPOT, position_side=PositionSide.LONG),
        REFERENCE,
    )
    assert [order.symbol for order in recording.asked] == [SPOT]


def test_a_fill_model_that_raises_leaves_the_venues_event_sequence_untouched():
    """The side check must not change when the venue allocates an event id.

    ``RecordedQuoteFillModel.plan`` raises on inputs the previous fill model
    never raised on, so "a submit whose fill model raised consumes no id" stops
    being an exotic path and becomes an ordinary one. The acknowledgement is
    therefore still built *after* ``plan`` returns, and the refusal branch builds
    its own pair: an event log captured before this check existed replays to the
    same ids after it.
    """
    market = venue(fill_model=model(book(), max_fill_ratio=Decimal("-1")))
    assert market._sequence == 0

    with pytest.raises(ConstraintError):
        market.submit("ORD-1", intent(OrderSide.BUY), REFERENCE)
    assert market._sequence == 0

    refused = market.submit(
        "ORD-2",
        intent(OrderSide.SELL, "0.10000", symbol=SPOT, position_side=PositionSide.SHORT),
        REFERENCE,
    )
    assert [event.event_id for event in refused] == ["ORD-2:1", "ORD-2:2"]
