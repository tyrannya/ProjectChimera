"""Hand-traced accounting witnesses for the P13 carry engine, and the mutants they catch.

Every expected number in this file was computed by hand before the code was run,
from round prices chosen so the arithmetic is checkable on paper. That is the
point of the exercise: a test whose expected value came out of the function it is
testing proves only that the function is deterministic.

The second half is the adversarial half. For each plausible one-line accounting
bug — leverage applied twice, a fee on quantity instead of notional, a reversed
funding sign, basis points read as percent, an omitted close fee, one leg's
capital used as the denominator, a settlement applied twice — there is a test
that FAILS if the bug is present. A suite that stays green under those injections
is not testing the accounting.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from chimera.futures.accounting import FundingEvent, funding_cash_flow
from chimera.futures.domain import Position, PositionSide
from nn.p13_carry import (
    Allocation,
    CarryError,
    Costs,
    FundingSettlement,
    Quote,
    Venue,
    apply_funding,
    close_carry,
    evaluate_block,
    hedge_quantity,
    is_liquidated,
    open_carry,
)

D = Decimal
ZERO = D("0")

VENUE = Venue(
    step_size=D("0.001"),
    min_notional=D("5"),
    maintenance_margin_rate=D("0.004"),
)
CAPITAL = Allocation(total_capital=D("1000"), spot=D("500"), perp=D("500"))
FREE = Costs(spot_fee=ZERO, spot_slippage=ZERO, perp_fee=ZERO, perp_slippage=ZERO)
#: The frozen P13 rates, as fractions of notional.
REAL = Costs(
    spot_fee=D("0.001"),
    spot_slippage=D("0.0005"),
    perp_fee=D("0.0005"),
    perp_slippage=D("0.0005"),
)


def quote(instant: int, spot: str, perp: str, mark: str | None = None) -> Quote:
    return Quote(
        instant_ns=instant,
        spot=D(spot),
        perp=D(perp),
        mark=D(mark) if mark is not None else None,
    )


# ---------------------------------------------------------------------------
# Witness A — the basis identity, traced by hand at zero cost
# ---------------------------------------------------------------------------
#
#   capital 1000, split 500/500, no frictions
#   entry: spot 100, perp 101   -> basis +1
#   exit : spot 200, perp 200   -> basis  0
#
#   Q = step_floor(min(500/100, 500/101)) = step_floor(4.950495...) = 4.950
#   spot notional 495.00   perp notional 499.95 = margin
#   free cash 1000 - 495 - 499.95 = 5.05
#   exit: spot 990.00; perp realised (101-200)*4.95 = -490.05; perp back 9.90
#   final 5.05 + 990 + 9.90 = 1004.95  ->  net +4.95
#   identity Q*(basis_in - basis_out) = 4.95 * 1 = 4.95
# ---------------------------------------------------------------------------


def test_witness_a_price_pnl_is_exactly_the_basis_move():
    entry, exit_ = quote(0, "100", "101"), quote(3_600_000_000_000, "200", "200")
    position = open_carry(entry, CAPITAL, FREE, VENUE)

    assert position.quantity == D("4.950")
    assert position.perp_margin == D("499.95")
    assert position.free_cash == D("5.05")

    final = close_carry(position, exit_, FREE)
    assert final == D("1004.95")
    assert final - CAPITAL.total_capital == D("4.95")
    # The whole of it is basis convergence, and the price doubling is absent.
    assert final - CAPITAL.total_capital == D("4.950") * (entry.basis - exit_.basis)


def test_price_pnl_is_invariant_to_the_path_between_the_same_basis_endpoints():
    """A 10x rally and a 90% crash pay the same, given the same entry/exit basis."""
    entry = quote(0, "100", "101")
    outcomes = []
    for spot, perp in (("200", "200"), ("1000", "1000"), ("10", "10")):
        position = open_carry(entry, CAPITAL, FREE, VENUE)
        outcomes.append(close_carry(position, quote(1, spot, perp), FREE))
    assert outcomes[0] == outcomes[1] == outcomes[2] == D("1004.95")


# ---------------------------------------------------------------------------
# Witness B — funding, traced by hand
# ---------------------------------------------------------------------------


def test_witness_b_short_receives_when_funding_is_positive():
    position = open_carry(quote(0, "100", "101"), CAPITAL, FREE, VENUE)
    cash_before = position.free_cash

    flow = apply_funding(position, FundingSettlement(1, D("0.0001"), D("100")))

    # 4.950 BTC x 100 mark x 0.0001 = 0.0495, received by the SHORT.
    assert flow == D("0.0495")
    assert position.funding_received == D("0.0495")
    assert position.funding_paid == ZERO
    assert position.free_cash == cash_before + D("0.0495")


def test_short_pays_when_funding_is_negative():
    position = open_carry(quote(0, "100", "101"), CAPITAL, FREE, VENUE)
    flow = apply_funding(position, FundingSettlement(1, D("-0.0001"), D("100")))
    assert flow == D("-0.0495")
    assert position.funding_paid == D("0.0495")
    assert position.funding_received == ZERO


def test_funding_sign_agrees_with_the_repositorys_own_convention():
    """Cross-checked against chimera.futures, not merely restated here.

    Two independent implementations agreeing is evidence; one implementation
    agreeing with a comment about itself is not.
    """
    position = open_carry(quote(0, "100", "101"), CAPITAL, FREE, VENUE)
    for rate in ("0.0001", "-0.0001", "0.00075", "0"):
        mine = -Decimal(PositionSide.SHORT.sign) * position.quantity * D("100") * D(rate)
        theirs = funding_cash_flow(
            Position(
                symbol="BTCUSDT",
                side=PositionSide.SHORT,
                quantity=position.quantity,
                entry_price=position.perp_entry,
                leverage=D("1"),
            ),
            FundingEvent(
                symbol="BTCUSDT", rate=D(rate), mark_price=D("100"), settlement_id=rate
            ),
        )
        assert mine == theirs, f"sign convention diverged at rate {rate}"


def test_a_redelivered_settlement_is_charged_once():
    position = open_carry(quote(0, "100", "101"), CAPITAL, FREE, VENUE)
    settlement = FundingSettlement(1, D("0.0001"), D("100"))
    first = apply_funding(position, settlement)
    second = apply_funding(position, FundingSettlement(1, D("0.0001"), D("100")))
    assert first == D("0.0495")
    assert second == ZERO
    assert position.funding_received == D("0.0495")


def test_funding_uses_the_mark_price_not_the_entry_price():
    """The notional base is the mark AT SETTLEMENT, which moves."""
    position = open_carry(quote(0, "100", "101"), CAPITAL, FREE, VENUE)
    flow = apply_funding(position, FundingSettlement(1, D("0.0001"), D("200")))
    assert flow == D("4.950") * D("200") * D("0.0001")
    assert flow != D("4.950") * D("101") * D("0.0001")


# ---------------------------------------------------------------------------
# Witness C — frictions, traced by hand
# ---------------------------------------------------------------------------
#
#   flat prices 100/100 in and out, so every penny of the result is friction
#   Q = step_floor(min(500/100.15, 500/100.10)) = step_floor(4.99251...) = 4.992
#   notional 499.20 per leg
#   entry: spot 0.4992 + 0.2496 ; perp 0.2496 + 0.2496   = 1.2480
#   exit : the same                                       = 1.2480
#   net = -2.4960
# ---------------------------------------------------------------------------


def test_witness_c_flat_prices_lose_exactly_the_round_trip_friction():
    entry, exit_ = quote(0, "100", "100"), quote(1, "100", "100")
    position = open_carry(entry, CAPITAL, REAL, VENUE)
    assert position.quantity == D("4.992")

    final = close_carry(position, exit_, REAL)
    net = final - CAPITAL.total_capital

    assert position.fees + position.slippage == D("2.4960")
    assert net == D("-2.4960")
    assert net == -(position.fees + position.slippage)


def test_the_quantity_is_the_minimum_over_both_legs_not_the_spot_leg_alone():
    """In contango the perpetual leg binds, and sizing from spot would over-commit."""
    entry = quote(0, "100", "102")  # +200 bps basis
    q = hedge_quantity(entry, CAPITAL, REAL, VENUE)

    spot_only = CAPITAL.spot / (entry.spot * (D(1) + REAL.spot_fee + REAL.spot_slippage))
    assert q < spot_only

    # And the position it produces actually fits inside both allocations.
    position = open_carry(entry, CAPITAL, REAL, VENUE)
    assert position.free_cash >= ZERO
    assert q * entry.perp <= CAPITAL.perp


@pytest.mark.parametrize("basis_bps", ["-200", "-10", "0", "10", "100", "500"])
def test_a_position_opens_across_the_whole_basis_range(basis_bps):
    """The sizing rule must not refuse to open in exactly the contango regimes
    a carry position exists to harvest. Sizing from the spot allocation alone
    refused everything above about +5 bps."""
    perp = D("100") * (D(1) + D(basis_bps) / D("10000"))
    position = open_carry(quote(0, "100", str(perp)), CAPITAL, REAL, VENUE)
    assert position.quantity > ZERO
    assert position.free_cash >= ZERO


def test_gross_leverage_never_exceeds_one_times_capital():
    for basis_bps in ("-500", "0", "500"):
        entry = quote(0, "100", str(D("100") * (D(1) + D(basis_bps) / D("10000"))))
        position = open_carry(entry, CAPITAL, REAL, VENUE)
        gross = position.quantity * entry.spot + position.quantity * entry.perp
        assert gross <= CAPITAL.total_capital


def test_entry_equity_is_capital_minus_entry_frictions_and_nothing_else():
    """The equity line must not double-count what free cash already reflects."""
    entry = quote(0, "100", "100")
    position = open_carry(entry, CAPITAL, REAL, VENUE)
    assert position.equity(entry) == CAPITAL.total_capital - (
        position.fees + position.slippage
    )


# ---------------------------------------------------------------------------
# Adversarial controls: each asserts a specific one-line bug would be caught
# ---------------------------------------------------------------------------


def test_double_applied_leverage_would_be_caught():
    """Asserted against what open_carry ACTUALLY posts as margin.

    The earlier form hand-constructed the levered object and so passed under a
    real `perp_margin = perp_notional / 2` bug in open_carry. Margin at 1x IS the
    entry notional; anything less is leverage the capital did not authorise, and
    equity alone cannot see it — the freed margin reappears as free cash — which
    is why margin and free cash are both pinned.
    """
    entry = quote(0, "100", "100")
    position = open_carry(entry, CAPITAL, REAL, VENUE)

    assert position.perp_margin == position.quantity * entry.perp_fill
    assert position.perp_margin / position.leverage == position.quantity * entry.perp_fill

    spot_out = (
        position.quantity * entry.spot_fill * (D(1) + REAL.spot_fee + REAL.spot_slippage)
    )
    perp_out = position.perp_margin + position.quantity * entry.perp_fill * (
        REAL.perp_fee + REAL.perp_slippage
    )
    assert position.free_cash == CAPITAL.total_capital - spot_out - perp_out


def test_a_fee_charged_on_quantity_instead_of_notional_is_off_by_the_price():
    """Asserted against what open_carry ACTUALLY booked, not against arithmetic
    on this test's own locals — the earlier form passed under a real
    fee-on-quantity bug because it never read position.fees."""
    position = open_carry(quote(0, "100", "100"), CAPITAL, REAL, VENUE)
    notional = position.quantity * D("100")

    expected = notional * REAL.spot_fee + notional * REAL.perp_fee
    assert position.fees == expected

    on_quantity = position.quantity * (REAL.spot_fee + REAL.perp_fee)
    assert position.fees != on_quantity
    assert position.fees == on_quantity * D("100")  # off by exactly the price


def test_a_reversed_funding_sign_flips_paid_and_received():
    position = open_carry(quote(0, "100", "100"), CAPITAL, FREE, VENUE)
    apply_funding(position, FundingSettlement(1, D("0.0001"), D("100")))
    assert position.funding_received > ZERO and position.funding_paid == ZERO

    reversed_flow = -(
        -Decimal(PositionSide.SHORT.sign) * position.quantity * D("100") * D("0.0001")
    )
    assert reversed_flow < ZERO  # a LONG's flow, which a SHORT must never book


def test_basis_points_read_as_percent_cannot_reach_the_accounting():
    """0.05% funding is 0.0005, not 0.05. Were the hundredfold form to reach the
    ledger it would multiply the payoff term by a hundred, so the guard refuses
    it at construction rather than letting the arithmetic scale."""
    position = open_carry(quote(0, "100", "100"), CAPITAL, FREE, VENUE)
    as_fraction = apply_funding(position, FundingSettlement(1, D("0.0005"), D("100")))
    assert as_fraction == position.quantity * D("100") * D("0.0005")

    with pytest.raises(CarryError):
        FundingSettlement(1, D("0.05"), D("100"))


@pytest.mark.parametrize("bad", ["1", "1.5", "-0.001", "100"])
def test_a_cost_rate_outside_zero_to_one_is_refused(bad):
    """A rate handed over in percent or bps lands here instead of silently scaling."""
    with pytest.raises(CarryError):
        Costs(spot_fee=D(bad), spot_slippage=ZERO, perp_fee=ZERO, perp_slippage=ZERO)


def test_omitting_one_legs_close_fee_would_flatter_every_block():
    entry, exit_ = quote(0, "100", "100"), quote(1, "100", "100")
    position = open_carry(entry, CAPITAL, REAL, VENUE)
    close_carry(position, exit_, REAL)
    full = position.fees + position.slippage

    entry_only = open_carry(entry, CAPITAL, REAL, VENUE)
    assert full > entry_only.fees + entry_only.slippage
    # Exactly double: at flat prices the entry and exit frictions are equal, so
    # dropping either leg's close fee would show up as a missing quarter.
    assert full == (entry_only.fees + entry_only.slippage) * D("2")


def test_the_denominator_is_total_capital_not_one_leg():
    entry, exit_ = quote(0, "100", "101"), quote(1, "100", "100")
    result = evaluate_block(
        "witness", [entry, exit_], [], CAPITAL, FREE, VENUE, min_settlements=1
    )
    assert result.net_return == result.net_pnl / CAPITAL.total_capital
    # Quoting on the spot leg alone would double the apparent performance.
    assert result.net_return != result.net_pnl / CAPITAL.spot


def test_a_non_positive_price_is_refused():
    with pytest.raises(CarryError):
        Quote(instant_ns=0, spot=ZERO, perp=D("100"))
    with pytest.raises(CarryError):
        Quote(instant_ns=0, spot=D("100"), perp=D("-1"))


def test_allocations_exceeding_capital_are_refused_as_arithmetic_leverage():
    with pytest.raises(CarryError):
        Allocation(total_capital=D("1000"), spot=D("700"), perp=D("700"))


def test_quantity_is_floored_to_the_step_never_rounded_up():
    fine = Venue(step_size=D("1"), min_notional=D("5"), maintenance_margin_rate=D("0.004"))
    q = hedge_quantity(quote(0, "100", "100"), CAPITAL, FREE, VENUE)
    coarse = hedge_quantity(quote(0, "100", "100"), CAPITAL, FREE, fine)
    assert coarse == D("5")  # 5.0 exactly at zero cost
    assert q <= D("5")


# ---------------------------------------------------------------------------
# Causality
# ---------------------------------------------------------------------------


def test_a_settlement_after_the_close_is_not_this_positions_cash_flow():
    quotes = [quote(0, "100", "100"), quote(100, "100", "100")]
    inside = FundingSettlement(50, D("0.001"), D("100"))
    after = FundingSettlement(500, D("0.001"), D("100"))

    with_after = evaluate_block(
        "b", quotes, [inside, after], CAPITAL, FREE, VENUE, min_settlements=1
    )
    without = evaluate_block("b", quotes, [inside], CAPITAL, FREE, VENUE, min_settlements=1)

    assert with_after.settlements == 1
    assert with_after.funding_received == without.funding_received
    assert with_after.net_pnl == without.net_pnl


def test_a_settlement_before_the_open_is_not_this_positions_cash_flow():
    quotes = [quote(100, "100", "100"), quote(200, "100", "100")]
    before = FundingSettlement(1, D("0.001"), D("100"))
    result = evaluate_block("b", quotes, [before], CAPITAL, FREE, VENUE, min_settlements=1)
    assert result.settlements == 0
    assert result.funding_received == ZERO


def test_shifting_funding_one_settlement_into_the_future_changes_the_result():
    """The positive control §6 requires: if a shift were undetectable, the
    engine would not be reading settlement instants at all."""
    quotes = [quote(0, "100", "100"), quote(100, "100", "100")]
    honest = [FundingSettlement(50, D("0.001"), D("100"))]
    shifted = [FundingSettlement(150, D("0.001"), D("100"))]

    a = evaluate_block("b", quotes, honest, CAPITAL, FREE, VENUE, min_settlements=1)
    b = evaluate_block("b", quotes, shifted, CAPITAL, FREE, VENUE, min_settlements=1)
    assert a.settlements == 1 and b.settlements == 0
    assert a.net_pnl != b.net_pnl


# ---------------------------------------------------------------------------
# Liquidation
# ---------------------------------------------------------------------------


def test_an_isolated_short_at_1x_liquidates_only_on_a_near_doubling():
    position = open_carry(quote(0, "100", "100"), CAPITAL, FREE, VENUE)
    assert not is_liquidated(position, quote(1, "150", "150"), VENUE, isolated=True)
    assert not is_liquidated(position, quote(1, "199", "199"), VENUE, isolated=True)
    assert is_liquidated(position, quote(1, "200", "200"), VENUE, isolated=True)


def test_the_portfolio_model_survives_a_doubling_the_isolated_model_does_not():
    """The two models disagree exactly where the modelling choice matters, which
    is why S4 reports the isolated case rather than the design asserting it away."""
    position = open_carry(quote(0, "100", "100"), CAPITAL, FREE, VENUE)
    doubled = quote(1, "200", "200")
    assert is_liquidated(position, doubled, VENUE, isolated=True)
    assert not is_liquidated(position, doubled, VENUE, isolated=False)


def test_an_isolated_liquidation_forfeits_margin_but_not_the_spot_leg():
    quotes = [quote(0, "100", "100"), quote(1, "200", "200")]
    result = evaluate_block(
        "b", quotes, [], CAPITAL, FREE, VENUE, min_settlements=1, isolated=True
    )
    assert result.liquidated
    # The spot leg roughly doubled while the short lost its margin, so this is a
    # hedge failure rather than a total loss. Asserting the sign, not a guess.
    assert result.net_pnl > -CAPITAL.total_capital


def test_a_funding_rate_in_percent_is_refused_as_a_unit_error():
    """0.05% is 0.0005. Handed over as 0.05 it is a hundredfold error on the
    one term the checkpoint exists to measure, so it must refuse, not scale."""
    FundingSettlement(1, D("0.0005"), D("100"))  # a plausible 8-hourly rate
    with pytest.raises(CarryError):
        FundingSettlement(1, D("0.05"), D("100"))
    with pytest.raises(CarryError):
        FundingSettlement(1, D("-0.05"), D("100"))


def test_a_non_positive_funding_mark_is_refused():
    with pytest.raises(CarryError):
        FundingSettlement(1, D("0.0001"), ZERO)


def test_orders_fill_at_the_candle_open_not_its_close():
    """Filling at the close of a candle labelled t executes at a price revealed
    an hour later — at both ends, which is the whole of the price PnL."""
    entry = Quote(
        instant_ns=0, spot=D("110"), perp=D("110"), spot_open=D("100"), perp_open=D("100")
    )
    position = open_carry(entry, CAPITAL, FREE, VENUE)
    assert position.spot_entry == D("100")
    assert position.perp_entry == D("100")
    # Sizing, too, is done against what an order actually pays.
    assert position.quantity == D("5.000")


def test_the_basis_identity_holds_on_transacted_prices_when_open_differs_from_close():
    entry = Quote(
        instant_ns=0, spot=D("111"), perp=D("113"), spot_open=D("100"), perp_open=D("101")
    )
    exit_ = Quote(
        instant_ns=1, spot=D("300"), perp=D("300"), spot_open=D("200"), perp_open=D("200")
    )
    result = evaluate_block("b", [entry, exit_], [], CAPITAL, FREE, VENUE, min_settlements=1)
    assert result.basis_entry == D("1")  # 101 - 100, the fill basis
    assert result.basis_exit == ZERO  # 200 - 200
    assert result.net_pnl == result.basis_pnl


def test_the_isolated_balance_feels_its_own_funding():
    """Routing S4's funding to free cash would make the strict bound lenient."""
    position = open_carry(quote(0, "100", "100"), CAPITAL, FREE, VENUE)
    # A short paying funding settlement after settlement must erode the walled-off
    # balance. Q=5 at mark 100 pays 5.00 per settlement against a 500.00 margin and
    # a 2.00 maintenance requirement, so 100 settlements exhaust it exactly.
    for i in range(1, 101):
        apply_funding(position, FundingSettlement(i, D("-0.01"), D("100")))
    assert position.funding_paid == D("500.00")
    assert position.perp_funding < ZERO
    assert position.funding_paid > ZERO
    # The portfolio is still solvent; the isolated balance is the one under strain.
    flat = quote(999, "100", "100")
    assert not is_liquidated(position, flat, VENUE, isolated=False)
    assert is_liquidated(position, flat, VENUE, isolated=True)


# ---------------------------------------------------------------------------
# The acquisition plan: networkless, boundary-respecting, and refusing
# ---------------------------------------------------------------------------


def test_the_plan_is_networkless_and_covers_every_frozen_source():
    from tools.acquire_p13_sources import plan_payload

    payload = plan_payload("BTCUSDT")
    assert set(payload["objects_by_field"]) == {
        "spot_price",
        "perpetual_price",
        "mark_price",
        "funding_settlement",
    }
    # Every source family covers the same months, so none is silently short.
    counts = set(payload["objects_by_field"].values())
    assert len(counts) == 1


def test_the_plan_never_names_an_object_beginning_at_or_after_the_boundary():
    from nn import p13_preregistration as prereg
    from tools.acquire_p13_sources import plan_objects

    boundary = prereg.DATA_BOUNDARY["span_end_exclusive"][:7]  # YYYY-MM
    for obj in plan_objects("BTCUSDT"):
        assert obj.period <= boundary, f"{obj.path} begins at or after the boundary"


def test_the_plan_uses_binances_own_path_grammar():
    from tools.acquire_p13_sources import plan_objects

    by_field: dict[str, object] = {}
    for o in plan_objects("BTCUSDT"):
        by_field.setdefault(o.field, o)  # the first month of each family
    assert by_field["spot_price"].path.startswith("data/spot/monthly/klines/BTCUSDT/1h/")
    assert by_field["perpetual_price"].path.startswith(
        "data/futures/um/monthly/klines/BTCUSDT/1h/"
    )
    assert by_field["mark_price"].path.startswith(
        "data/futures/um/monthly/markPriceKlines/BTCUSDT/1h/"
    )
    # The fundingRate family carries no interval segment, per get_path.
    assert by_field["funding_settlement"].path == (
        "data/futures/um/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-2020-01.zip"
    )


def test_every_object_has_a_published_checksum_companion():
    from tools.acquire_p13_sources import plan_objects

    objects = plan_objects("BTCUSDT")
    # Every family, not just the first — slicing the head covered spot_price
    # alone, so removing the funding family's companion left the suite green.
    assert len({o.field for o in objects}) == 4
    for obj in objects:
        assert obj.checksum_url == obj.url + ".CHECKSUM"


def test_a_refusal_record_says_what_was_not_done():
    """NOT EVALUABLE is a research outcome and needs evidence like any other."""
    from tools.acquire_p13_sources import refusal_record

    probes = [
        {
            "field": "spot_price",
            "url": "https://data.binance.vision/x",
            "reachable": False,
            "status": None,
            "error": "blocked",
        }
    ]
    record = refusal_record(probes, "BTCUSDT", "unreachable")
    assert record["outcome"] == "NOT EVALUABLE"
    assert record["hosts_refused"] == ["data.binance.vision"]
    assert "not a negative economic result" in record["what_this_does_not_mean"]
    assert any("different venue" in item for item in record["what_was_not_done"])
    assert any("Styx" in item for item in record["what_was_not_done"])


def test_liquidation_is_tested_against_the_intra_bar_high_when_available():
    """An hourly grid cannot resolve within-bar action, so testing the close
    alone would miss a touch the position genuinely took."""
    position = open_carry(quote(0, "100", "100"), CAPITAL, FREE, VENUE)
    # Closes well below the isolated threshold, but the bar's mark high reaches it.
    bar = Quote(instant_ns=1, spot=D("120"), perp=D("120"), mark=D("120"), mark_high=D("205"))
    assert bar.liquidation_touch == D("205")
    assert bar.liquidation_touch_is_high
    assert is_liquidated(position, bar, VENUE, isolated=True)

    # Without the high, the check falls back to the close and records that.
    closes_only = Quote(instant_ns=1, spot=D("120"), perp=D("120"), mark=D("120"))
    assert closes_only.liquidation_touch == D("120")
    assert not closes_only.liquidation_touch_is_high
    assert not is_liquidated(position, closes_only, VENUE, isolated=True)
