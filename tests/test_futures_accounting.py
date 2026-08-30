"""Futures cash flows and margin: the sign table, the two funding counters, margin.

Every number in this file is written out rather than recomputed from the
implementation. A funding sign that flips, a paid/received pair that quietly
collapses into a net, or a liquidation price that drifts is a bug that arithmetic
mirroring the code under test cannot see.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from chimera.futures import (
    AccountingError,
    FundingEvent,
    Ledger,
    MarginState,
    Position,
    PositionSide,
    SymbolConstraints,
    funding_cash_flow,
    liquidation_price,
    load_constraint_source,
    margin_state,
    unrealised_pnl,
)
from chimera.risk import RiskLimits

SYMBOL = "BTC/USDT:USDT"
OTHER_SYMBOL = "ETH/USDT:USDT"

#: Binance USD-M tier-1 maintenance margin rate, the value the committed
#: constraint table publishes for BTCUSDT.
MMR = Decimal("0.004")


def btc_constraints() -> SymbolConstraints:
    """The committed BTC/USDT:USDT venue metadata. No network, no credentials."""
    return load_constraint_source().constraints(SYMBOL)


def open_position(
    side: PositionSide,
    quantity: str = "0.5",
    entry_price: str = "60000",
    leverage: str = "1",
) -> Position:
    return Position(
        symbol=SYMBOL,
        side=side,
        quantity=Decimal(quantity),
        entry_price=Decimal(entry_price),
        leverage=Decimal(leverage),
    )


def funding(
    rate: str,
    mark_price: str = "60000",
    settlement_id: str = "settlement-1",
    symbol: str = SYMBOL,
) -> FundingEvent:
    return FundingEvent(
        symbol=symbol,
        rate=Decimal(rate),
        mark_price=Decimal(mark_price),
        settlement_id=settlement_id,
    )


# --- the funding sign table -------------------------------------------------
# 0.5 BTC marked at 60000 is a notional of 30000, so a rate of 0.0004 is a flow
# of exactly 12 quote units. The four tests below differ only in the side and in
# the sign of the rate, which is the whole of the table.


def test_a_long_pays_funding_when_the_rate_is_positive():
    """Longs pay shorts. A positive rate on a LONG must be a NEGATIVE cash flow.

    Catches the sign error that turns the single largest recurring cost of a
    held long into income.
    """
    flow = funding_cash_flow(open_position(PositionSide.LONG), funding("0.0004"))
    assert flow == Decimal("-12")


def test_a_long_receives_funding_when_the_rate_is_negative():
    """A negative rate reverses who pays: the LONG is credited, not charged."""
    flow = funding_cash_flow(open_position(PositionSide.LONG), funding("-0.0004"))
    assert flow == Decimal("12")


def test_a_short_receives_funding_when_the_rate_is_positive():
    """The mirror of the LONG case. Catches a table that ignores the side."""
    flow = funding_cash_flow(open_position(PositionSide.SHORT), funding("0.0004"))
    assert flow == Decimal("12")


def test_a_short_pays_funding_when_the_rate_is_negative():
    """A negative rate charges the SHORT. The fourth and last cell of the table."""
    flow = funding_cash_flow(open_position(PositionSide.SHORT), funding("-0.0004"))
    assert flow == Decimal("-12")


@pytest.mark.parametrize("rate", ["0", "0.0004", "-0.0004", "0.75", "-0.75"])
def test_a_flat_position_neither_pays_nor_receives_at_any_rate(rate):
    """Exactly zero, for every rate, because there is nothing to charge funding on.

    This pins the public contract, not the ``is_flat`` early return that
    implements it: ``Position.__post_init__`` forces ``quantity == 0`` whenever
    the side is FLAT, so the notional is zero and any sign is multiplied away.
    The rates include +/-0.75 so that a rounding slip would still be visible.

    The early return runs *after* the symbol check, which is what the two tests
    below pin: being flat makes the amount zero, it does not make a foreign
    settlement applicable.
    """
    assert funding_cash_flow(Position(symbol=SYMBOL), funding(rate)) == Decimal("0")


def test_funding_for_another_symbol_cannot_be_applied_to_this_position():
    """A BTC settlement must never be charged against an ETH position, or vice versa.

    Catches a mismatch being silently priced at the wrong symbol's mark.
    """
    with pytest.raises(AccountingError, match="cannot be applied to a"):
        funding_cash_flow(
            open_position(PositionSide.LONG), funding("0.0004", symbol=OTHER_SYMBOL)
        )


def test_funding_for_another_symbol_is_refused_even_when_the_position_is_flat():
    """The mismatch guard must not depend on whether the position happens to be open.

    "Funding for X cannot be applied to a Y position" is a statement about the
    two symbols. Being flat makes the *amount* zero; it does not make an ETH
    settlement a thing that belongs to a BTC position.
    """
    with pytest.raises(AccountingError, match="cannot be applied to a"):
        funding_cash_flow(Position(symbol=SYMBOL), funding("0.0004", symbol=OTHER_SYMBOL))


def test_a_foreign_settlement_booked_while_flat_does_not_poison_the_dedup_list():
    """The dangerous consequence: a swallowed mismatch is remembered as applied.

    ``book_funding`` appends the id whatever ``funding_cash_flow`` returned, so
    a foreign settlement absorbed while flat lands in ``applied_funding``. If
    the same id is later delivered against an open position the idempotency
    check fires first and the mismatch is never raised at all — it is silently
    deduplicated away. Settlement ids are venue-scoped and one Ledger spans
    every symbol, so a venue numbering settlements per-symbol ("1", "2", ...)
    can cross-suppress genuine funding between symbols this way.
    """
    ledger = Ledger()
    foreign = funding("0.0004", symbol=OTHER_SYMBOL, settlement_id="1")

    with pytest.raises(AccountingError, match="cannot be applied to a"):
        ledger.book_funding(Position(symbol=SYMBOL), foreign)

    assert ledger.applied_funding == []
    # And the mismatch is still raised, not deduplicated, against an open position.
    with pytest.raises(AccountingError, match="cannot be applied to a"):
        ledger.book_funding(open_position(PositionSide.LONG), foreign)


# --- funding events ---------------------------------------------------------
def test_a_funding_event_without_a_settlement_id_is_refused():
    """No id means no deduplication, and a redelivered settlement charged twice."""
    with pytest.raises(AccountingError, match="cannot be deduplicated"):
        FundingEvent(
            symbol=SYMBOL,
            rate=Decimal("0.0004"),
            mark_price=Decimal("60000"),
            settlement_id="",
        )


@pytest.mark.parametrize("mark_price", ["0", "-1", "-60000"])
def test_a_funding_event_with_a_non_positive_mark_price_is_refused(mark_price):
    """A funding flow is a fraction of a notional, and a notional needs a real price."""
    with pytest.raises(AccountingError, match="funding mark price"):
        FundingEvent(
            symbol=SYMBOL,
            rate=Decimal("0.0004"),
            mark_price=Decimal(mark_price),
            settlement_id="settlement-1",
        )


# --- the ledger's two funding counters --------------------------------------
def test_book_funding_records_paid_and_received_separately_without_netting():
    """Pay 12, receive 10, and the ledger must show 12 and 10 — never a single 2.

    Both counters are non-negative magnitudes. Collapsing them loses the fact
    that the account is being charged to hold what it holds, which is the one
    thing an operator reading the funding line wants to know.
    """
    ledger = Ledger()
    position = open_position(PositionSide.LONG)

    paid = ledger.book_funding(position, funding("0.0004", settlement_id="paid-12"))
    received = ledger.book_funding(
        position, funding("-0.0004", mark_price="50000", settlement_id="received-10")
    )

    assert paid == Decimal("-12")
    assert received == Decimal("10")
    # A negative flow is stored as a positive magnitude under funding_paid.
    assert ledger.funding_paid == Decimal("12")
    assert ledger.funding_received == Decimal("10")


def test_net_funding_is_received_minus_paid():
    """Paying 12 and receiving 10 nets to -2: the reader nets, the ledger does not."""
    ledger = Ledger(funding_paid=Decimal("12"), funding_received=Decimal("10"))
    assert ledger.net_funding == Decimal("-2")


def test_book_funding_is_idempotent_by_settlement_id():
    """A redelivered settlement changes nothing at all and reports a zero flow.

    Catches the restart or webhook replay that charges the same 8-hour funding
    twice. Every field is compared, not just the funding ones.
    """
    ledger = Ledger()
    position = open_position(PositionSide.LONG)
    event = funding("0.0004", settlement_id="settlement-8h")

    first = ledger.book_funding(position, event)
    snapshot = ledger.to_dict()

    second = ledger.book_funding(position, event)

    assert first == Decimal("-12")
    assert second == Decimal("0")
    assert ledger.to_dict() == snapshot
    assert ledger.funding_paid == Decimal("12")
    assert ledger.applied_funding == ["settlement-8h"]


def test_net_pnl_is_realised_less_fees_plus_net_funding():
    """Realised 100, fees 7, received 10, paid 12 gives exactly 91.

    Catches fees added instead of subtracted, and net funding dropped from the
    headline number entirely.
    """
    ledger = Ledger()
    ledger.book_realised(Decimal("100"))
    ledger.book_fee(Decimal("7"))
    ledger.funding_received = Decimal("10")
    ledger.funding_paid = Decimal("12")

    assert ledger.net_funding == Decimal("-2")
    assert ledger.net_pnl == Decimal("91")


def test_book_fee_refuses_a_negative_fee():
    """There is no maker-rebate path here, so a negative fee is a sign error upstream."""
    ledger = Ledger()
    with pytest.raises(AccountingError, match="is a rebate"):
        ledger.book_fee(Decimal("-0.01"))
    assert ledger.trading_fees == Decimal("0")


def test_book_turnover_refuses_a_negative_notional():
    """Turnover is an absolute quantity; a negative one would shrink the total traded."""
    ledger = Ledger()
    with pytest.raises(AccountingError, match=r"turnover .* is negative"):
        ledger.book_turnover(Decimal("-0.01"))
    assert ledger.turnover == Decimal("0")


def test_ledger_round_trips_through_to_dict_and_from_dict_with_applied_funding():
    """Applied settlement ids must survive a restart, or funding is charged twice.

    Catches a persisted ledger that keeps the money but drops the idempotency
    keys, which is the exact shape of a double-charge after a crash.
    """
    ledger = Ledger()
    position = open_position(PositionSide.LONG)
    ledger.book_funding(position, funding("0.0004", settlement_id="paid-12"))
    ledger.book_funding(
        position, funding("-0.0004", mark_price="50000", settlement_id="received-10")
    )
    ledger.book_fee(Decimal("7"))
    ledger.book_turnover(Decimal("30000"))
    ledger.book_realised(Decimal("100"))

    revived = Ledger.from_dict(ledger.to_dict())

    assert revived == ledger
    assert revived.applied_funding == ["paid-12", "received-10"]
    assert revived.funding_paid == Decimal("12")
    assert revived.funding_received == Decimal("10")
    assert revived.trading_fees == Decimal("7")
    assert revived.turnover == Decimal("30000")
    assert revived.realised_pnl == Decimal("100")
    assert revived.net_pnl == Decimal("91")
    # And the revived ledger still refuses the settlements it already booked.
    assert revived.book_funding(position, funding("0.0004", settlement_id="paid-12")) == (
        Decimal("0")
    )


# --- unrealised PnL ---------------------------------------------------------
def test_unrealised_pnl_of_a_long_follows_the_mark_above_and_below_entry():
    """0.5 BTC long from 60000 is +500 at 61000 and -500 at 59000."""
    position = open_position(PositionSide.LONG)
    assert unrealised_pnl(position, Decimal("61000")) == Decimal("500")
    assert unrealised_pnl(position, Decimal("59000")) == Decimal("-500")


def test_unrealised_pnl_of_a_short_mirrors_the_long():
    """The same marks with the sign reversed: -500 above entry, +500 below.

    Catches a mark-to-market that ignores the side and reports a short rallying
    into profit.
    """
    position = open_position(PositionSide.SHORT)
    assert unrealised_pnl(position, Decimal("61000")) == Decimal("-500")
    assert unrealised_pnl(position, Decimal("59000")) == Decimal("500")


def test_unrealised_pnl_of_a_flat_position_is_exactly_zero():
    """A flat position has no entry price to mark against, so the answer is 0, not entry."""
    assert unrealised_pnl(Position(symbol=SYMBOL), Decimal("60000")) == Decimal("0")


@pytest.mark.parametrize("mark_price", ["0", "-1"])
def test_unrealised_pnl_refuses_a_non_positive_mark_price(mark_price):
    """A zero mark would report the whole position as a total loss rather than raise."""
    with pytest.raises(AccountingError, match=r"mark price .* is not positive"):
        unrealised_pnl(open_position(PositionSide.LONG), Decimal(mark_price))


# --- liquidation price ------------------------------------------------------
def test_liquidation_price_of_a_long_at_one_times_leverage_is_entry_times_mmr():
    """LONG at 1x liquidates at entry * mmr = 60000 * 0.004 = 240, not at zero.

    An isolated 1x long really can lose almost all of its margin first; a
    formula that forgot the maintenance rate would put this at exactly 0.
    """
    price = liquidation_price(PositionSide.LONG, Decimal("60000"), Decimal("1"), MMR)
    assert price == Decimal("240")


def test_liquidation_price_of_a_short_at_one_times_leverage_is_entry_times_two_minus_mmr():
    """SHORT at 1x liquidates at entry * (2 - mmr) = 60000 * 1.996 = 119760."""
    price = liquidation_price(PositionSide.SHORT, Decimal("60000"), Decimal("1"), MMR)
    assert price == Decimal("119760")


def test_liquidation_price_at_two_times_leverage_halves_the_distance_to_entry():
    """A second point on the curve: 30240 for a LONG, 89760 for a SHORT.

    One leverage cannot distinguish `1 - 1/leverage` from a constant, so the
    2x point is what pins the leverage term itself.
    """
    long_price = liquidation_price(PositionSide.LONG, Decimal("60000"), Decimal("2"), MMR)
    short_price = liquidation_price(PositionSide.SHORT, Decimal("60000"), Decimal("2"), MMR)
    assert long_price == Decimal("30240")
    assert short_price == Decimal("89760")


def test_liquidation_price_refuses_a_flat_side():
    """A position that does not exist has no liquidation price, not a price of zero."""
    with pytest.raises(AccountingError, match="no liquidation price"):
        liquidation_price(PositionSide.FLAT, Decimal("60000"), Decimal("1"), MMR)


@pytest.mark.parametrize("entry_price", ["0", "-60000"])
def test_liquidation_price_refuses_a_non_positive_entry_price(entry_price):
    """Every term scales off the entry, so a zero entry would report 0 for both sides."""
    with pytest.raises(AccountingError, match=r"entry price .* is not positive"):
        liquidation_price(PositionSide.LONG, Decimal(entry_price), Decimal("1"), MMR)


@pytest.mark.parametrize("leverage", ["0", "-1"])
def test_liquidation_price_refuses_a_non_positive_leverage(leverage):
    """1/leverage would divide by zero or flip the formula's sign."""
    with pytest.raises(AccountingError, match=r"leverage .* is not positive"):
        liquidation_price(PositionSide.LONG, Decimal("60000"), Decimal(leverage), MMR)


@pytest.mark.parametrize("rate", ["0", "1", "1.5", "-0.004", "4"])
def test_liquidation_price_refuses_a_maintenance_margin_rate_outside_zero_to_one(rate):
    """A rate given as a percent (4) rather than a fraction (0.04) must not be used.

    Catches the units mistake that would move a liquidation price by two orders
    of magnitude and hand Aegis a number it treats as real.
    """
    with pytest.raises(AccountingError, match=r"not a fraction in \(0, 1\)"):
        liquidation_price(PositionSide.LONG, Decimal("60000"), Decimal("1"), Decimal(rate))


# --- margin state -----------------------------------------------------------
def test_margin_state_of_a_flat_position_is_none_rather_than_a_zeroed_record():
    """None, not a record of zeros.

    A zeroed MarginState would report liquidation_price=0 and therefore a
    liquidation_distance of 100%, which reads to Aegis as "maximally safe" —
    a confident claim about a position that does not exist. `is None` is the
    only answer that cannot be misread.
    """
    assert margin_state(Position(symbol=SYMBOL), Decimal("60000"), btc_constraints()) is None


def test_margin_state_of_an_open_long_reports_exact_margins_and_liquidation():
    """0.5 BTC long at 60000, 1x, marked at 60000: 30000 / 120 / 240 / 0.996."""
    state = margin_state(open_position(PositionSide.LONG), Decimal("60000"), btc_constraints())
    assert isinstance(state, MarginState)
    assert state.symbol == SYMBOL
    assert state.side is PositionSide.LONG
    assert state.leverage == Decimal("1")
    assert state.initial_margin == Decimal("30000")
    assert state.maintenance_margin == Decimal("120")
    assert state.liquidation_price == Decimal("240")
    assert state.liquidation_distance == Decimal("0.996")


def test_margin_state_of_an_open_short_reports_exact_margins_and_liquidation():
    """The same position short: identical margins, liquidation above at 119760.

    Catches a margin calculation that signs the notional and reports a negative
    initial margin for a short.
    """
    state = margin_state(
        open_position(PositionSide.SHORT), Decimal("60000"), btc_constraints()
    )
    assert state is not None
    assert state.side is PositionSide.SHORT
    assert state.initial_margin == Decimal("30000")
    assert state.maintenance_margin == Decimal("120")
    assert state.liquidation_price == Decimal("119760")
    assert state.liquidation_distance == Decimal("0.996")


def test_margin_state_divides_the_initial_margin_by_the_positions_leverage():
    """At 2x the same notional commits half the margin and liquidates at 30240.

    Catches an initial margin hard-coded to the full notional, which would
    understate how little equity actually stands behind a levered position.
    """
    state = margin_state(
        open_position(PositionSide.LONG, leverage="2"), Decimal("60000"), btc_constraints()
    )
    assert state is not None
    assert state.leverage == Decimal("2")
    assert state.initial_margin == Decimal("15000")
    assert state.maintenance_margin == Decimal("120")
    assert state.liquidation_price == Decimal("30240")
    assert state.liquidation_distance == Decimal("0.496")


def test_margin_state_measures_the_liquidation_distance_from_the_mark_not_the_entry():
    """Marked at 48000 the same long is 0.995 from liquidation, not 0.996.

    Catches a distance computed against the entry price, which would never move
    as the market approached liquidation.
    """
    state = margin_state(open_position(PositionSide.LONG), Decimal("48000"), btc_constraints())
    assert state is not None
    assert state.liquidation_price == Decimal("240")
    assert state.initial_margin == Decimal("24000")
    assert state.maintenance_margin == Decimal("96")
    assert state.liquidation_distance == Decimal("0.995")


@pytest.mark.parametrize("mark_price", ["0", "-60000"])
def test_margin_state_refuses_a_non_positive_mark_price(mark_price):
    """Without a real mark there is no notional and no distance to report."""
    with pytest.raises(AccountingError, match=r"mark price .* is not positive"):
        margin_state(open_position(PositionSide.LONG), Decimal(mark_price), btc_constraints())


def test_liquidation_distance_is_the_same_0_996_for_both_sides_at_one_times_leverage():
    """LONG and SHORT are equidistant from liquidation at 1x, and both clear Aegis.

    This is the number `RiskLimits.min_liquidation_distance_pct` is compared
    against. At 1x it is 0.996 for either side and passes the 0.5 default; at 2x
    it is 0.496 for either side and the same limit bites. Pinning both points
    keeps the v1 "isolated, 1x" scope and the risk limit consistent — a change
    that made 1x fail, or 2x pass, would be a change to what Aegis permits.
    """
    constraints = btc_constraints()
    mark = Decimal("60000")
    minimum = Decimal(str(RiskLimits().min_liquidation_distance_pct))

    at_1x = [
        margin_state(open_position(side, leverage="1"), mark, constraints)
        for side in (PositionSide.LONG, PositionSide.SHORT)
    ]
    at_2x = [
        margin_state(open_position(side, leverage="2"), mark, constraints)
        for side in (PositionSide.LONG, PositionSide.SHORT)
    ]

    assert [s.liquidation_distance for s in at_1x] == [Decimal("0.996"), Decimal("0.996")]
    assert [s.liquidation_distance for s in at_2x] == [Decimal("0.496"), Decimal("0.496")]
    assert minimum == Decimal("0.5")
    assert all(s.liquidation_distance > minimum for s in at_1x)
    assert all(s.liquidation_distance < minimum for s in at_2x)
