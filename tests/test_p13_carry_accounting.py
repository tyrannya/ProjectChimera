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

from decimal import Decimal, InvalidOperation

import pytest

from chimera.futures.accounting import FundingEvent, funding_cash_flow
from chimera.futures.domain import Position, PositionSide
from nn.p13_carry import (
    NOMINAL_BAR_NS,
    NOT_DETERMINABLE,
    RESEARCH_BOUNDARY_NS,
    TOUCH_MARK_CLOSE,
    TOUCH_MARK_HIGH,
    TOUCH_SOURCES,
    TOUCH_SPOT_CLOSE,
    Allocation,
    CarryError,
    Costs,
    FundingSettlement,
    LiquidationTouchProvenance,
    Quote,
    Venue,
    apply_funding,
    close_carry,
    evaluate_block,
    hedge_quantity,
    is_liquidated,
    open_carry,
    unclosed_block_result,
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
    """A flat synthetic witness: open EQUALS close on both legs, said out loud.

    `Quote` no longer falls back to the close when an execution open is missing —
    that fallback let a production row without an open execute at a price revealed
    an hour later without anyone deciding to. A witness written in a flat world is
    entitled to open == close, so it states both rather than inheriting one, and
    every hand-traced number below is unchanged by the difference.
    """
    return Quote(
        instant_ns=instant,
        spot=D(spot),
        perp=D(perp),
        mark=D(mark) if mark is not None else None,
        spot_open=D(spot),
        perp_open=D(perp),
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
    # Three bars, not two: the trigger is the middle one, so the forced close has
    # the FOLLOWING bar the frozen design fills at. Its open is 200 — the same
    # price the two-bar version of this witness used to fill at — so the claim and
    # the arithmetic below are unchanged; only the causality is now legal. A
    # trigger on the LAST bar has no permitted fill at all, and that case is
    # amendment A1's, tested separately.
    quotes = [quote(0, "100", "100"), quote(1, "200", "200"), quote(2, "200", "200")]
    result = evaluate_block(
        "b", quotes, [], CAPITAL, FREE, VENUE, min_settlements=1, isolated=True
    )
    assert result.liquidated and not result.unclosed
    assert result.liquidation_instant_ns == 1
    assert result.forced_close_instant_ns == 2
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


# ---------------------------------------------------------------------------
# Liquidation causality — the trigger and the fill are different instants
# ---------------------------------------------------------------------------
#
# The frozen rule (MARGIN_AND_LIQUIDATION.forced_close_price) is that a
# liquidation-forced close fills at the OPEN OF THE FOLLOWING BAR. The engine
# used to fill at the TRIGGER bar's own open — a price stamped an hour before the
# intra-bar high that caused the liquidation, so the fill preceded its own cause.
#
# One witness, hand-traced, at zero cost and Q = 5.000 from 1,000 of capital:
#
#   t0 open 100/100      -> entry, fill basis 0
#   t1 open 150/158      -> TRIGGER (mark high spikes), fill basis 8
#   t2 open 300/301      -> the following bar, fill basis 1
#
# Correct:  net = Q x (basis_in - basis_out) = 5 x (0 - 1) = -5
# Acausal:  net = Q x (basis_in - basis_out) = 5 x (0 - 8) = -40
# ---------------------------------------------------------------------------

HOUR = 3_600_000_000_000

#: A mark high absurdly far above the closes. It is a synthetic control of the
#: LIQUIDATION PREDICATE and nothing else: the primary portfolio model is
#: price-invariant at equal quantity, so no ordinary price path can fire it, and
#: the question under test is which bar's open the forced close then executes at.
PREDICATE_SPIKE = "60000"


def _liquidation_witness(final_bar_is_the_trigger: bool) -> list[Quote]:
    trigger = Quote(
        instant_ns=HOUR,
        spot=D("200"),
        perp=D("200"),
        mark_high=D(PREDICATE_SPIKE),
        spot_open=D("150"),
        perp_open=D("158"),
    )
    quotes = [quote(0, "100", "100"), trigger]
    if not final_bar_is_the_trigger:
        quotes.append(
            Quote(
                instant_ns=2 * HOUR,
                spot=D("400"),
                perp=D("400"),
                spot_open=D("300"),
                perp_open=D("301"),
            )
        )
    return quotes


def test_a_forced_close_fills_at_the_following_bars_open_not_the_trigger_bars():
    """The defect the external audit proved, and the number that separates them."""
    quotes = _liquidation_witness(final_bar_is_the_trigger=False)
    result = evaluate_block("b", quotes, [], CAPITAL, FREE, VENUE, min_settlements=0)

    assert result.liquidated and not result.unclosed
    assert result.liquidation_instant_ns == HOUR
    assert result.forced_close_instant_ns == 2 * HOUR
    # The economic close, not merely the bookkeeping: the exit basis is the
    # FOLLOWING bar's 301 - 300, not the trigger bar's 158 - 150.
    assert result.basis_exit == D("1")
    assert result.net_pnl == D("-5")
    assert result.net_pnl == result.basis_pnl

    # And what the acausal fill would have produced, stated so the test cannot
    # pass by accident: the trigger bar's own fill basis is 8, worth -40.
    assert quotes[1].fill_basis == D("8")
    acausal = (result.basis_entry - quotes[1].fill_basis) * result.quantity
    assert acausal == D("-40")
    assert result.net_pnl != acausal


def test_no_forced_close_ever_fills_before_its_own_trigger():
    """The invariant behind the witness, asserted as an invariant."""
    for isolated in (False, True):
        result = evaluate_block(
            "b",
            _liquidation_witness(final_bar_is_the_trigger=False),
            [],
            CAPITAL,
            FREE,
            VENUE,
            min_settlements=0,
            isolated=isolated,
        )
        assert result.liquidated
        assert result.forced_close_instant_ns > result.liquidation_instant_ns


def test_a_settlement_after_the_liquidation_trigger_is_not_this_positions_cash_flow():
    """The position stopped existing at the trigger, an hour before the fill.

    A settlement between the trigger and the forced-close bar belongs to a
    position that no longer existed, and crediting it would be a gift bought with
    the extra hour the causal fill rule introduced.
    """
    quotes = _liquidation_witness(final_bar_is_the_trigger=False)
    after_trigger = FundingSettlement(2 * HOUR, D("0.001"), D("100"))
    result = evaluate_block(
        "b", quotes, [after_trigger], CAPITAL, FREE, VENUE, min_settlements=0
    )
    assert result.liquidated
    assert result.settlements == 0
    assert result.funding_received == ZERO

    # At the trigger instant itself it IS applied: the position was held through
    # that accrual window. This is the frozen boundary tie rule, at the one
    # instant the repair moved things around.
    at_trigger = FundingSettlement(HOUR, D("0.001"), D("100"))
    tied = evaluate_block("b", quotes, [at_trigger], CAPITAL, FREE, VENUE, min_settlements=0)
    assert tied.settlements == 1
    assert tied.funding_received > ZERO


# ---------------------------------------------------------------------------
# Amendment A1 — and what correct holding-window attribution does to it
# ---------------------------------------------------------------------------
#
# A1 is unchanged, still hashed, and still in force. What changed is which of its
# two causes can reach the evaluator.
#
# The held intra-bar windows are bars 0 .. N-1, so a liquidation trigger always
# has a successor and the LIQUIDATION route into A1 is unreachable from
# evaluate_block. That route was reachable only while the loop also tested bar N
# — a bar the position had already closed at the OPEN of — so the case A1 was
# written against was in part an artefact of the off-by-one this branch repairs.
#
# The surviving cause is the one A1 says it "applies equally to": the block that
# has no valid exit instant at or before its last hour
# (POSITION_LIFECYCLE.close_instant). That belongs to the block runner, which
# does not exist. So A1's encoding is tested directly, through the function both
# causes are meant to call, rather than through a branch nothing can reach.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("isolated", [False, True], ids=["portfolio", "isolated"])
@pytest.mark.parametrize("length", [2, 3, 4, 5])
def test_no_liquidation_trigger_can_leave_a_block_unclosed(length, isolated):
    """The reachability claim, executed over every shape rather than asserted.

    A trigger on bar k needs bar k+1 to fill at. Under bars-0..N-1 attribution
    k <= N-1 always, so the fill always exists — for every block length, every
    position of the touch, and both margin models. If this ever fails, the
    holding window has drifted back over the exit bar.
    """
    for spike_at in range(length):
        quotes = []
        for index in range(length):
            quotes.append(
                Quote(
                    instant_ns=index * HOUR,
                    spot=D("100"),
                    perp=D("100"),
                    spot_open=D("100"),
                    perp_open=D("100"),
                    mark_high=D(PREDICATE_SPIKE) if index == spike_at else None,
                )
            )
        result = evaluate_block(
            "b", quotes, [], CAPITAL, FREE, VENUE, min_settlements=0, isolated=isolated
        )
        assert not result.unclosed, f"length={length} spike_at={spike_at} reached A1"
        if result.liquidated:
            assert result.forced_close_instant_ns is not None
            assert result.forced_close_instant_ns > result.liquidation_instant_ns


def test_a_touch_on_the_old_final_bar_no_longer_triggers_at_all():
    """The specific witness the A1 liquidation route used to rest on.

    Two quotes, the touch on the second: the position closed at that bar's OPEN,
    so its post-open high happens to a position that no longer exists. It must
    close normally, not liquidate and not go UNCLOSED.
    """
    result = evaluate_block(
        "b",
        _liquidation_witness(final_bar_is_the_trigger=True),
        [],
        CAPITAL,
        FREE,
        VENUE,
        min_settlements=0,
    )
    assert result.opened
    assert not result.liquidated and not result.unclosed
    assert result.reason == "closed at block end"
    assert result.held_bars == 1
    # It closed at the final bar's OPEN — 150 spot against 158 perp, basis 8 —
    # so by hand net = Q x (basis_in - basis_out) = 5 x (0 - 8) = -40.
    assert result.basis_exit == D("8")
    assert result.net_pnl == D("-40")


def test_the_a1_encoding_reports_incurred_facts_and_not_determinable_economics():
    """A1's rule, exercised through the definition both of its causes call.

    Hand-supplied facts, so nothing here is read back out of the evaluator: what
    is under test is that the encoding reports what WAS incurred and refuses to
    put a number on what was not.
    """
    result = unclosed_block_result(
        "2025",
        "UNCLOSED: no valid exit instant at or before the block's last hour",
        settlements=812,
        quantity=D("70.123"),
        basis_entry=D("12.5"),
        funding_received=D("9000"),
        funding_paid=D("400"),
        fees=D("1500"),
        slippage=D("500"),
        max_adverse_excursion_pnl=D("-2500"),
        total_capital=D("1000000"),
        thin_sample=False,
    )

    assert result.opened and result.unclosed
    assert result.forced_close_instant_ns is None
    # The facts A1 says are reported "as the facts they are".
    assert result.settlements == 812
    assert result.quantity == D("70.123")
    assert result.basis_entry == D("12.5")
    assert result.net_funding == D("8600")
    assert result.fees == D("1500")
    assert result.slippage == D("500")
    assert result.max_adverse_excursion_pnl == D("-2500")
    # And the fraction the frozen design names, on the same base as the return.
    assert result.max_adverse_excursion == D("-0.0025")
    # And the close-dependent quantities, which nobody measured.
    for value in (result.net_return, result.net_pnl, result.basis_pnl, result.basis_exit):
        assert value.is_nan()
        assert value != ZERO


def test_an_unclosed_blocks_return_is_not_determinable_rather_than_zero():
    """A gate that forgets the UNCLOSED flag must crash, not average in a zero.

    Zero is the flattering answer: it would enter G2's mean as a harmless block
    and G3's worst-block test as a block that lost nothing, which is exactly the
    treatment VIABILITY_GATE.liquidated_blocks refuses for every other liquidated
    block.
    """
    result = unclosed_block_result(
        "2025",
        "UNCLOSED",
        settlements=0,
        quantity=D("5"),
        basis_entry=ZERO,
        funding_received=ZERO,
        funding_paid=ZERO,
        fees=ZERO,
        slippage=ZERO,
        max_adverse_excursion_pnl=ZERO,
        total_capital=D("1000"),
        thin_sample=True,
    )
    with pytest.raises(InvalidOperation):
        _ = result.net_return > ZERO
    with pytest.raises(InvalidOperation):
        _ = result.net_return < D("-0.02")


# ---------------------------------------------------------------------------
# Funding settlement ties, at the two instants that decide them
# ---------------------------------------------------------------------------


def test_a_settlement_exactly_at_the_open_instant_is_not_applied():
    """`open_instant < settlement_instant`, strictly. Every block opens at
    00:00:00 on 1 January, which is itself a settlement instant, so this tie
    fires six times out of six."""
    quotes = [quote(0, "100", "100"), quote(HOUR, "100", "100")]
    at_open = FundingSettlement(0, D("0.001"), D("100"))
    result = evaluate_block("b", quotes, [at_open], CAPITAL, FREE, VENUE, min_settlements=0)
    assert result.settlements == 0
    assert result.funding_received == ZERO


def test_a_settlement_exactly_at_the_close_instant_is_applied():
    """`settlement_instant <= close_instant`, inclusive. The position was held
    through that accrual window."""
    quotes = [quote(0, "100", "100"), quote(HOUR, "100", "100")]
    at_close = FundingSettlement(HOUR, D("0.001"), D("100"))
    result = evaluate_block("b", quotes, [at_close], CAPITAL, FREE, VENUE, min_settlements=0)
    assert result.settlements == 1
    # Q = 5.000 at a mark of 100 is 500 of notional; a +0.001 rate pays the short
    # 0.50, and nothing else moved.
    assert result.funding_received == D("0.500")
    assert result.net_pnl == D("0.500")


def test_a_delayed_settlement_does_not_become_visible_at_an_earlier_quote():
    """Its cash must follow its instant, and the equity path must say so.

    The observable is a liquidation at the bar BEFORE the settlement: the flow is
    not this position's, so neither the reported funding nor the maximum adverse
    excursion may contain it.
    """
    quotes = _liquidation_witness(final_bar_is_the_trigger=False)
    later = FundingSettlement(2 * HOUR, D("-0.01"), D("100"))
    earlier = FundingSettlement(HOUR, D("-0.01"), D("100"))

    delayed = evaluate_block("b", quotes, [later], CAPITAL, FREE, VENUE, min_settlements=0)
    on_time = evaluate_block("b", quotes, [earlier], CAPITAL, FREE, VENUE, min_settlements=0)

    assert delayed.settlements == 0 and delayed.funding_paid == ZERO
    assert on_time.settlements == 1 and on_time.funding_paid == D("5.000")
    # The equity path itself, not merely the totals. Both blocks liquidate and
    # both lose the same 5 of basis at the forced close, so the ONLY thing that
    # can separate their excursions is the 5 of funding — and it separates them
    # exactly, which is what makes this a measurement of the payment rather than
    # of anything else.
    assert delayed.net_pnl == D("-5") and delayed.max_adverse_excursion_pnl == D("-5")
    assert on_time.net_pnl == D("-10") and on_time.max_adverse_excursion_pnl == D("-10")
    assert on_time.max_adverse_excursion_pnl - delayed.max_adverse_excursion_pnl == D("-5")
    # The same facts on the block return's own base: of 1,000 of capital.
    assert delayed.max_adverse_excursion == D("-0.005")
    assert on_time.max_adverse_excursion == D("-0.01")
    assert delayed.net_pnl != on_time.net_pnl


# ---------------------------------------------------------------------------
# The execution open is required, and its absence is never a licence to fill
# at the close
# ---------------------------------------------------------------------------


def test_a_quote_without_an_execution_open_refuses_to_produce_a_fill():
    closes_only = Quote(instant_ns=0, spot=D("100"), perp=D("101"))
    assert not closes_only.has_execution_opens
    # The close is still available as a MARK — that is what it is for.
    assert closes_only.basis == D("1")
    with pytest.raises(CarryError, match="no spot open"):
        _ = closes_only.spot_fill
    with pytest.raises(CarryError, match="no perpetual open"):
        _ = closes_only.perp_fill


@pytest.mark.parametrize("missing", ["spot_open", "perp_open"])
def test_a_missing_execution_open_fails_the_block_closed_rather_than_softly(missing):
    """It must REFUSE, not report a block that merely failed to open.

    "Not opened" is a reported reason that leaves a block out of G1 and G2 — a
    filter. A row without the field the design executes against is invalid data,
    and the difference between refusing and quietly excluding is the difference
    between a loader bug that stops the run and one that reshapes the sample.
    """
    fields = {"spot_open": D("100"), "perp_open": D("100")}
    del fields[missing]
    lame = Quote(instant_ns=HOUR, spot=D("100"), perp=D("100"), **fields)
    quotes = [quote(0, "100", "100"), lame]
    with pytest.raises(CarryError, match="execution open"):
        evaluate_block("b", quotes, [], CAPITAL, FREE, VENUE, min_settlements=0)


# ---------------------------------------------------------------------------
# The research boundary, asserted by the production evaluator itself
# ---------------------------------------------------------------------------
#
# Deleting evaluate_block's boundary block must make these fail. They call the
# evaluator, not the preregistration constant and not the acquisition planner:
# the external audit removed that enforcement and the whole suite stayed green.
# ---------------------------------------------------------------------------

LAST_LEGAL_NS = RESEARCH_BOUNDARY_NS - 1


def test_the_last_instant_strictly_before_the_boundary_is_accepted():
    """The positive half, which is what makes the four refusals below mean `<`
    rather than `<=`."""
    quotes = [
        quote(RESEARCH_BOUNDARY_NS - HOUR, "100", "100"),
        quote(LAST_LEGAL_NS, "100", "100"),
    ]
    settlement = FundingSettlement(LAST_LEGAL_NS, D("0.001"), D("100"))
    result = evaluate_block("b", quotes, [settlement], CAPITAL, FREE, VENUE, min_settlements=0)
    assert result.opened
    assert result.settlements == 1


@pytest.mark.parametrize(
    "instant",
    [RESEARCH_BOUNDARY_NS, RESEARCH_BOUNDARY_NS + 1],
    ids=["exactly_at_the_boundary", "strictly_after_the_boundary"],
)
def test_a_quote_at_or_after_the_research_boundary_is_refused(instant):
    quotes = [quote(RESEARCH_BOUNDARY_NS - HOUR, "100", "100"), quote(instant, "100", "100")]
    with pytest.raises(CarryError, match="research boundary"):
        evaluate_block("b", quotes, [], CAPITAL, FREE, VENUE, min_settlements=0)


@pytest.mark.parametrize(
    "instant",
    [RESEARCH_BOUNDARY_NS, RESEARCH_BOUNDARY_NS + 1],
    ids=["exactly_at_the_boundary", "strictly_after_the_boundary"],
)
def test_a_funding_settlement_at_or_after_the_research_boundary_is_refused(instant):
    quotes = [
        quote(RESEARCH_BOUNDARY_NS - HOUR, "100", "100"),
        quote(LAST_LEGAL_NS, "100", "100"),
    ]
    settlement = FundingSettlement(instant, D("0.001"), D("100"))
    with pytest.raises(CarryError, match="research boundary"):
        evaluate_block("b", quotes, [settlement], CAPITAL, FREE, VENUE, min_settlements=0)


def test_the_boundary_the_evaluator_asserts_is_the_one_the_design_froze():
    """Read from the preregistration, never a second literal to drift from it."""
    from datetime import datetime

    from nn.p13_preregistration import DATA_BOUNDARY

    frozen = datetime.fromisoformat(DATA_BOUNDARY["span_end_exclusive"])
    assert RESEARCH_BOUNDARY_NS == int(frozen.timestamp() * 1_000_000_000)


# ---------------------------------------------------------------------------
# R1 — the holding window is bars 0 .. N-1, and nothing else
# ---------------------------------------------------------------------------
#
# A position opened at bar 0's OPEN is exposed to the REMAINDER OF BAR 0. A
# position closed at bar N's OPEN is exposed to NOTHING after that open. The
# evaluator once ran its liquidation and excursion loop over `quotes[1:]`, which
# is wrong at BOTH ends at once: it hid the entry bar's intra-bar action from the
# liquidation model and simultaneously tested the exit bar's post-open window
# against a position that had already closed.
#
# Every number below is hand-traced at zero cost, Q = 5.000 out of 1,000 of
# capital, so nothing is read back out of the function under test:
#
#   Q = step_floor(min(500/100, 500/100)) = 5.000
#   spot notional 500, perp notional 500 = margin, free cash 0
#   equity_t = 0 + 5 x spot_close_t + 500 + (100 - perp_close_t) x 5
# ---------------------------------------------------------------------------


def bar(
    instant: int,
    spot_open: str,
    perp_open: str,
    spot_close: str | None = None,
    perp_close: str | None = None,
    mark: str | None = None,
    mark_high: str | None = None,
) -> Quote:
    """A witness bar whose OPEN and CLOSE are stated separately and on purpose.

    The fills come from the opens; the marks, the basis series and the excursion
    come from the closes. Keeping them distinct is what lets one bar carry an
    ordinary fill and a catastrophic close, which is the shape every window
    question below turns on.
    """
    return Quote(
        instant_ns=instant,
        spot=D(spot_close if spot_close is not None else spot_open),
        perp=D(perp_close if perp_close is not None else perp_open),
        spot_open=D(spot_open),
        perp_open=D(perp_open),
        mark=D(mark) if mark is not None else None,
        mark_high=D(mark_high) if mark_high is not None else None,
    )


def _run(quotes, settlements=(), isolated=False, venue=VENUE):
    return evaluate_block(
        "b",
        quotes,
        list(settlements),
        CAPITAL,
        FREE,
        venue,
        min_settlements=0,
        isolated=isolated,
    )


# --- Witness A: the entry bar --------------------------------------------- #


@pytest.mark.parametrize("isolated", [False, True], ids=["portfolio", "isolated"])
def test_witness_a_a_touch_during_the_remainder_of_the_entry_bar_liquidates(isolated):
    """The position held through bar 0. Its high must reach the liquidation model.

    Portfolio, by hand: the maintenance requirement is Q x touch x mmr =
    5 x 60,000 x 0.004 = 1,200 against an equity of 1,000, so the bar liquidates.
    Isolated: the touch is far past entry x (2 - mmr) = 199.6. Under the old
    `quotes[1:]` window neither fired, because bar 0 was never tested at all.

    The forced close then fills at bar 1's OPEN — 100 spot against 103 perp,
    basis 3 — so net = Q x (basis_in - basis_out) = 5 x (0 - 3) = -15.
    """
    quotes = [
        bar(0, "100", "100", mark_high=PREDICATE_SPIKE),
        bar(HOUR, "100", "103"),
        bar(2 * HOUR, "100", "100"),
    ]
    result = _run(quotes, isolated=isolated)

    assert result.liquidated, "a touch inside the entry bar was invisible to the model"
    assert not result.unclosed
    assert result.liquidation_instant_ns == 0
    assert result.forced_close_instant_ns == HOUR
    assert result.held_bars == 1
    assert result.basis_exit == D("3")
    if not isolated:
        assert result.net_pnl == D("-15")
        assert result.net_pnl == result.basis_pnl


def test_witness_b_a_touch_on_a_middle_held_bar_still_liquidates():
    """The control. Without it a window that tested NOTHING would look repaired."""
    quotes = [
        bar(0, "100", "100"),
        bar(HOUR, "100", "100", mark_high=PREDICATE_SPIKE),
        bar(2 * HOUR, "100", "102"),
    ]
    result = _run(quotes)
    assert result.liquidated and not result.unclosed
    assert result.liquidation_instant_ns == HOUR
    assert result.forced_close_instant_ns == 2 * HOUR
    assert result.held_bars == 2
    # Filled at bar 2's open, basis 2: net = 5 x (0 - 2) = -10.
    assert result.net_pnl == D("-10")


# --- Witness C: the normal exit bar ---------------------------------------- #


@pytest.mark.parametrize("isolated", [False, True], ids=["portfolio", "isolated"])
def test_witness_c_the_exit_bars_post_open_window_is_not_tested(isolated):
    """The position closed at bar 2's OPEN. Bar 2's high is not its business.

    The same spike that liquidates on bars 0 and 1 must do nothing here, and the
    block must close normally at that bar's open — basis 2, so net = -10.
    """
    quotes = [
        bar(0, "100", "100"),
        bar(HOUR, "100", "100"),
        bar(2 * HOUR, "100", "102", mark_high=PREDICATE_SPIKE),
    ]
    result = _run(quotes, isolated=isolated)

    assert not result.liquidated, "a bar closed at the open of was still tested"
    assert not result.unclosed
    assert result.liquidation_instant_ns is None
    assert result.forced_close_instant_ns is None
    assert result.reason == "closed at block end"
    assert result.held_bars == 2
    assert result.net_pnl == D("-10")


def test_the_same_touch_liquidates_on_a_held_bar_and_not_on_the_exit_bar():
    """The two-sided control: one spike, two positions, opposite answers.

    Neither half means much alone — a model that never liquidates passes the
    second, one that always liquidates passes the first. Together they pin the
    window's far edge to exactly one bar.
    """
    held = _run(
        [
            bar(0, "100", "100"),
            bar(HOUR, "100", "100", mark_high=PREDICATE_SPIKE),
            bar(2 * HOUR, "100", "100"),
        ]
    )
    after_exit = _run(
        [
            bar(0, "100", "100"),
            bar(HOUR, "100", "100"),
            bar(2 * HOUR, "100", "100", mark_high=PREDICATE_SPIKE),
        ]
    )
    assert held.liquidated and not after_exit.liquidated


# --- Witness D: the excursion window --------------------------------------- #


def test_witness_d_movement_after_the_exit_cannot_reach_the_excursion():
    """Bar 2 opens flat and closes catastrophically. The position left at the open.

    By hand, had that close been marked: equity = 0 + 5 x 100 + 500 +
    (100 - 200) x 5 = 500, an excursion of -500. It must not appear, and the
    realised result must be exactly zero, because the exit FILL basis is zero.
    """
    quotes = [
        bar(0, "100", "100"),
        bar(HOUR, "100", "100"),
        bar(2 * HOUR, "100", "100", spot_close="100", perp_close="200"),
    ]
    result = _run(quotes)
    assert result.max_adverse_excursion_pnl == ZERO
    assert result.max_adverse_excursion == ZERO
    assert result.max_adverse_excursion_pnl != D("-500")
    assert result.net_pnl == ZERO


def test_the_entry_bars_own_close_does_reach_the_excursion():
    """The mirror image, and the reason the previous test is not vacuous.

    The identical -500 basis excursion, moved onto bar 0 — a bar the position
    held through — must be recorded. A window that simply dropped both ends would
    pass the test above and fail this one.
    """
    quotes = [
        bar(0, "100", "100", spot_close="100", perp_close="200"),
        bar(HOUR, "100", "100"),
        bar(2 * HOUR, "100", "100"),
    ]
    result = _run(quotes)
    assert result.max_adverse_excursion_pnl == D("-500")
    # -500 of 1,000 of capital. G3's floor is -0.02 on this same base, so an
    # excursion reported in quote units would read as 25,000x its true depth.
    assert result.max_adverse_excursion == D("-0.5")
    assert result.net_pnl == ZERO


def test_the_excursion_stops_at_the_liquidation_trigger():
    """A liquidated position stops accruing marks at the trigger, not at block end."""
    quotes = [
        bar(0, "100", "100"),
        bar(HOUR, "100", "100", mark_high=PREDICATE_SPIKE),
        bar(2 * HOUR, "100", "100", spot_close="100", perp_close="900"),
    ]
    result = _run(quotes)
    assert result.liquidated
    assert result.max_adverse_excursion == ZERO


# ---------------------------------------------------------------------------
# R2 — an economic quantity nobody measured is not a zero
# ---------------------------------------------------------------------------
#
# VIABILITY_GATE.excluded_blocks leaves an unopened block out of G1 and G2, so a
# correct gate never reads these fields. The repair is about the gate that
# FORGETS: a finite zero enters G2's mean as an ordinary block and G3's
# worst-block test as a block that lost nothing. No gate rule changes here — only
# whether the flattering failure mode is available at all.
# ---------------------------------------------------------------------------

#: Nothing can be bought with 500 of allocation at a 10,000 minimum notional.
CANNOT_OPEN = Venue(
    step_size=D("0.001"), min_notional=D("10000"), maintenance_margin_rate=D("0.004")
)

#: Every field whose value is an ECONOMIC MEASUREMENT of a position. With no
#: position there is no measurement, so none of them may be a number.
UNMEASURED_FIELDS = (
    "basis_entry",
    "basis_exit",
    "basis_pnl",
    "net_pnl",
    "net_return",
    "max_adverse_excursion_pnl",
    "max_adverse_excursion",
)

#: Every field that is a STRUCTURAL FACT about a block with no position rather
#: than a measurement of one. These stay numeric on purpose: no fill happened, so
#: no fee was charged; no position existed, so no funding reached it. Losing this
#: half would be the opposite error — refusing to state what IS known.
STRUCTURAL_FIELDS = (
    "quantity",
    "funding_received",
    "funding_paid",
    "fees",
    "slippage",
    "rebalance_cost",
)


def _not_opened() -> "object":
    return _run([bar(0, "100", "100"), bar(HOUR, "100", "100")], venue=CANNOT_OPEN)


def _too_few_quotes() -> "object":
    return _run([bar(0, "100", "100")])


@pytest.mark.parametrize(
    "make", [_not_opened, _too_few_quotes], ids=["not_opened", "one_quote"]
)
@pytest.mark.parametrize("name", UNMEASURED_FIELDS)
def test_an_unmeasured_economic_field_is_not_a_finite_zero(make, name):
    value = getattr(make(), name)
    assert value.is_nan(), f"{name} is a number for a block that never opened"
    assert value != ZERO


@pytest.mark.parametrize(
    "make", [_not_opened, _too_few_quotes], ids=["not_opened", "one_quote"]
)
@pytest.mark.parametrize("name", STRUCTURAL_FIELDS)
def test_a_structurally_known_zero_stays_a_number(make, name):
    """The other direction. NaN everywhere would be its own kind of dishonest."""
    value = getattr(make(), name)
    assert not value.is_nan()
    assert value == ZERO


def test_the_not_opened_flags_stay_readable_without_touching_the_economics():
    """Checking `opened is False` must remain the cheap, obvious thing to do."""
    result = _not_opened()
    assert result.opened is False
    assert result.liquidated is False
    assert result.unclosed is False
    assert result.thin_sample is True
    assert result.settlements == 0
    assert result.held_bars == 0
    assert "not opened" in result.reason


def test_the_unmeasured_value_is_the_modules_own_declared_sentinel():
    """Not merely "some NaN": the one NOT_DETERMINABLE names.

    A Decimal NaN is never equal to itself, so what can be asserted is the
    representation — which is enough to catch a signalling NaN, a negative one, or
    a float sneaking in where a Decimal belongs.
    """
    value = _not_opened().net_return
    assert isinstance(value, type(NOT_DETERMINABLE))
    assert str(value) == str(NOT_DETERMINABLE) == "NaN"


def test_ordering_an_unmeasured_return_fails_closed():
    """G1's `> 0` and G3's `>= -0.02` must raise rather than answer."""
    result = _not_opened()
    for other in (ZERO, D("-0.02"), D("0.0025")):
        with pytest.raises(InvalidOperation):
            _ = result.net_return > other
        with pytest.raises(InvalidOperation):
            _ = result.net_return < other


def test_an_accidental_mean_cannot_silently_absorb_an_unmeasured_return():
    """G2 averages block returns. A forgotten exclusion must poison the mean.

    Two real blocks at -0.01 and +0.03 average to +0.01 — a passing-looking
    number. Letting a block that never opened in as a zero would drag it to
    +0.00667 and still look like an answer. It must be NaN, and any decision
    taken on it must raise.
    """
    unmeasured = _not_opened().net_return
    honest = (D("-0.01") + D("0.03")) / 2
    assert honest == D("0.01")

    contaminated = (D("-0.01") + D("0.03") + unmeasured) / 3
    assert contaminated.is_nan()
    assert not contaminated == D("0.00666666666666666666666666667")
    with pytest.raises(InvalidOperation):
        _ = contaminated > ZERO


def test_summing_unmeasured_returns_does_not_produce_a_total():
    """The aggregate half: a total built over a NaN is a NaN, not a subtotal."""
    unmeasured = _not_opened().net_return
    assert sum([D("0.01"), unmeasured, D("0.02")], ZERO).is_nan()


# ---------------------------------------------------------------------------
# R3 — which series the liquidation check actually used, persisted
# ---------------------------------------------------------------------------
#
# MARGIN_AND_LIQUIDATION requires the check to record whether it used the hourly
# mark HIGH or a weaker fallback, "so the check is never quietly weaker than it
# claims to be". Quote knew; BlockResult threw it away at the end of the loop, so
# a block checked entirely against spot closes was byte-identical to one checked
# against real mark highs.
#
# Three sources, not two: MARK_PRICE_FALLBACK substitutes the SPOT close for the
# mark series itself, per archive object, so a mark close and a spot close are
# different fallbacks of different strength.
# ---------------------------------------------------------------------------


def _three_bars(**kwargs) -> list[Quote]:
    return [bar(i * HOUR, "100", "100", **kwargs) for i in range(3)]


def test_a_block_checked_entirely_against_mark_highs_records_high_coverage():
    result = _run(_three_bars(mark="100", mark_high="100"))
    provenance = result.liquidation_touch_provenance
    assert provenance.as_dict() == {
        TOUCH_MARK_HIGH: 2,
        TOUCH_MARK_CLOSE: 0,
        TOUCH_SPOT_CLOSE: 0,
    }
    assert provenance.all_mark_high
    assert not provenance.used_a_weaker_fallback
    # Two held bars, two tests. A count that drifted from the window would show.
    assert provenance.tested == result.held_bars == 2


def test_a_block_that_fell_back_to_the_mark_close_records_that_fallback():
    result = _run(_three_bars(mark="100"))
    provenance = result.liquidation_touch_provenance
    assert provenance.as_dict() == {
        TOUCH_MARK_HIGH: 0,
        TOUCH_MARK_CLOSE: 2,
        TOUCH_SPOT_CLOSE: 0,
    }
    assert provenance.used_a_weaker_fallback
    assert not provenance.all_mark_high


def test_a_block_that_fell_back_to_the_spot_close_records_the_weaker_provenance():
    """MARK_PRICE_FALLBACK's case, and it must not be filed as a mark close."""
    result = _run(_three_bars())
    provenance = result.liquidation_touch_provenance
    assert provenance.as_dict() == {
        TOUCH_MARK_HIGH: 0,
        TOUCH_MARK_CLOSE: 0,
        TOUCH_SPOT_CLOSE: 2,
    }
    assert provenance.used_a_weaker_fallback
    assert not provenance.all_mark_high


def test_changing_high_availability_changes_the_recorded_provenance():
    """The discriminating claim: the record follows the data, not the code path."""
    strong = _run(_three_bars(mark="100", mark_high="100")).liquidation_touch_provenance
    middle = _run(_three_bars(mark="100")).liquidation_touch_provenance
    weak = _run(_three_bars()).liquidation_touch_provenance
    assert len({tuple(p.as_dict().items()) for p in (strong, middle, weak)}) == 3


def test_a_block_that_mixes_sources_reports_every_one_of_them():
    """Partial availability is the LIKELY case: the fallback is per archive object."""
    quotes = [
        bar(0, "100", "100", mark="100", mark_high="100"),
        bar(HOUR, "100", "100", mark="100"),
        bar(2 * HOUR, "100", "100"),
    ]
    provenance = _run(quotes).liquidation_touch_provenance
    assert provenance.as_dict() == {
        TOUCH_MARK_HIGH: 1,
        TOUCH_MARK_CLOSE: 1,
        TOUCH_SPOT_CLOSE: 0,
    }
    assert not provenance.all_mark_high
    assert provenance.used_a_weaker_fallback


def test_the_artifact_facing_record_cannot_claim_all_high_coverage_after_a_fallback():
    """One weak bar in a hundred strong ones still forfeits the claim."""
    quotes = [bar(i * HOUR, "100", "100", mark="100", mark_high="100") for i in range(20)]
    quotes[7] = bar(7 * HOUR, "100", "100", mark="100")
    provenance = _run(quotes).liquidation_touch_provenance
    assert provenance.mark_high == 18 and provenance.mark_close == 1
    assert not provenance.all_mark_high
    assert sum(provenance.as_dict().values()) == provenance.tested


def test_a_block_that_never_opened_does_not_claim_high_coverage_vacuously():
    """No test ran, so no coverage was established. `all` over nothing is not True here."""
    provenance = _not_opened().liquidation_touch_provenance
    assert provenance.tested == 0
    assert not provenance.all_mark_high
    assert not provenance.used_a_weaker_fallback


def test_the_provenance_of_a_liquidated_block_counts_only_the_bars_tested():
    """It stops at the trigger, like the rest of the holding window."""
    quotes = [
        bar(0, "100", "100", mark="100", mark_high="100"),
        bar(HOUR, "100", "100", mark="100", mark_high=PREDICATE_SPIKE),
        bar(2 * HOUR, "100", "100"),
        bar(3 * HOUR, "100", "100"),
    ]
    result = _run(quotes)
    assert result.liquidated and result.held_bars == 2
    assert result.liquidation_touch_provenance.mark_high == 2
    assert result.liquidation_touch_provenance.tested == 2


def test_the_touch_source_names_agree_with_the_value_the_touch_takes():
    """The name and the number must come from the same branch, or the record lies."""
    high = bar(0, "100", "100", spot_close="110", mark="120", mark_high="130")
    assert (
        high.liquidation_touch == D("130") and high.liquidation_touch_source == TOUCH_MARK_HIGH
    )
    close = bar(0, "100", "100", spot_close="110", mark="120")
    assert (
        close.liquidation_touch == D("120")
        and close.liquidation_touch_source == TOUCH_MARK_CLOSE
    )
    spot = bar(0, "100", "100", spot_close="110")
    assert (
        spot.liquidation_touch == D("110")
        and spot.liquidation_touch_source == TOUCH_SPOT_CLOSE
    )
    assert set(TOUCH_SOURCES) == {TOUCH_MARK_HIGH, TOUCH_MARK_CLOSE, TOUCH_SPOT_CLOSE}


def test_a_negative_touch_count_is_refused():
    with pytest.raises(CarryError, match="negative"):
        LiquidationTouchProvenance(mark_high=-1)


# ---------------------------------------------------------------------------
# R4 — causality is read from the instants, never from the caller's list order
# ---------------------------------------------------------------------------


def test_a_sorted_unique_quote_series_is_accepted():
    """The positive control, without which the four refusals below prove nothing."""
    result = _run([bar(i * HOUR, "100", "100") for i in range(4)])
    assert result.opened and result.held_bars == 3


@pytest.mark.parametrize(
    "instants,ids",
    [
        ([0, HOUR, HOUR, 2 * HOUR], "duplicate_timestamp"),
        ([2 * HOUR, HOUR, 0], "fully_descending"),
        ([0, 2 * HOUR, HOUR, 3 * HOUR], "one_inversion_inside_a_sorted_series"),
        ([0, 0], "duplicate_at_the_open"),
    ],
    ids=lambda value: value if isinstance(value, str) else "",
)
def test_a_quote_series_that_is_not_strictly_increasing_is_refused(instants, ids):
    with pytest.raises(CarryError, match="strictly increasing"):
        _run([bar(i, "100", "100") for i in instants])


def test_the_chronology_check_runs_before_any_economic_state_exists():
    """It must REFUSE, not report a block. A malformed series is invalid data.

    Proved by making the series one that would otherwise have opened and traded:
    if the check ran after `open_carry`, a BlockResult would come back instead.
    """
    good = _run([bar(0, "100", "100"), bar(HOUR, "100", "102")])
    assert good.opened and good.net_pnl == D("-10")
    with pytest.raises(CarryError, match="strictly increasing"):
        _run([bar(HOUR, "100", "102"), bar(0, "100", "100")])


def test_an_out_of_order_series_is_refused_rather_than_sorted():
    """Sorting it would be the flattering repair: it would silently invent an
    entry and an exit the caller never handed over, at prices belonging to
    different bars."""
    ascending = _run([bar(0, "100", "100"), bar(HOUR, "100", "105")])
    assert ascending.basis_exit == D("5")
    with pytest.raises(CarryError):
        _run([bar(HOUR, "100", "105"), bar(0, "100", "100")])


# --- Funding: order-independent by construction, and deduplicated ---------- #


def _settled_series() -> list[Quote]:
    return [bar(i * HOUR, "100", "100") for i in range(3)]


def test_funding_settlements_are_order_independent_because_production_sorts_them():
    """Not a new rule — the existing one, executed.

    `evaluate_block` sorts the due settlements by instant and keys them by it, so
    the caller's list order cannot reach the result. Q = 5 at a mark of 100 is 500
    of notional; a +0.001 rate pays the short 0.50, twice.
    """
    first = FundingSettlement(HOUR, D("0.001"), D("100"))
    second = FundingSettlement(2 * HOUR, D("0.001"), D("100"))
    forward = _run(_settled_series(), [first, second])
    backward = _run(_settled_series(), [second, first])

    assert forward.settlements == backward.settlements == 2
    assert forward.funding_received == backward.funding_received == D("1.000")
    assert forward.net_pnl == backward.net_pnl == D("1.000")


def test_a_duplicated_settlement_row_changes_nothing():
    """FUNDING_SEMANTICS.application: "deduplicated by settlement instant"."""
    once = FundingSettlement(HOUR, D("0.001"), D("100"))
    plain = _run(_settled_series(), [once])
    repeated = _run(_settled_series(), [once, once, once])
    assert plain.settlements == repeated.settlements == 1
    assert plain.funding_received == repeated.funding_received == D("0.500")


def test_a_settlement_at_the_close_instant_survives_the_corrected_window():
    """The tie rule fires at an instant the held-bar loop no longer visits.

    Bars 0..N-1 are the held windows, so the loop stops before the close instant.
    A settlement AT the close is still this position's cash flow — the position
    held through that accrual window — and must be applied after the loop rather
    than dropped with the bar.
    """
    at_close = FundingSettlement(2 * HOUR, D("0.001"), D("100"))
    result = _run(_settled_series(), [at_close])
    assert result.settlements == 1
    assert result.funding_received == D("0.500")
    assert result.net_pnl == D("0.500")


# ---------------------------------------------------------------------------
# Gaps — a multi-hour jump must not look like an ordinary +1h transition
# ---------------------------------------------------------------------------
#
# "Following bar" means the next VALID EXECUTABLE OBSERVATION, which is the
# frozen text's own resolution rather than a choice made here:
# MARGIN_AND_LIQUIDATION.forced_close_price calls the next open "the first price
# an operator could actually have transacted at", and all three lifecycle
# instants in POSITION_LIFECYCLE resolve an invalid grid point by moving to the
# first VALID one at or after it. An operator cannot transact at a hole.
#
# So the rule needs no amendment. What it needs is for the hole to be visible.
# ---------------------------------------------------------------------------


def test_a_contiguous_hourly_block_records_no_gap_and_a_one_bar_step():
    """The two fields say different things, and the names now say which.

    Zero gaps, and a largest ADJACENT SPACING of one nominal bar. The spacing is
    not a gap, which is why it is no longer called one: a non-zero "max gap"
    beside a gap count of zero would be a contradiction sitting in the evidence.
    """
    result = _run([bar(i * HOUR, "100", "100") for i in range(5)])
    assert result.quote_gap_count == 0
    assert result.max_quote_step_ns == NOMINAL_BAR_NS


def test_a_hole_in_the_grid_is_detected_and_its_duration_recorded():
    quotes = [bar(0, "100", "100"), bar(HOUR, "100", "100"), bar(6 * HOUR, "100", "100")]
    result = _run(quotes)
    assert result.quote_gap_count == 1
    assert result.max_quote_step_ns == 5 * HOUR
    assert result.max_quote_step_ns > NOMINAL_BAR_NS


def test_a_multi_hour_jump_does_not_look_identical_to_a_normal_transition():
    """The whole requirement, as one comparison of two otherwise identical blocks."""
    contiguous = _run(
        [bar(0, "100", "100"), bar(HOUR, "100", "100"), bar(2 * HOUR, "100", "102")]
    )
    gapped = _run([bar(0, "100", "100"), bar(HOUR, "100", "100"), bar(9 * HOUR, "100", "102")])
    # Same economics, by construction — the fills and marks are the same prices.
    assert contiguous.net_pnl == gapped.net_pnl == D("-10")
    # And distinguishable anyway, which is the point.
    assert (contiguous.quote_gap_count, contiguous.max_quote_step_ns) != (
        gapped.quote_gap_count,
        gapped.max_quote_step_ns,
    )
    assert gapped.quote_gap_count == 1 and gapped.max_quote_step_ns == 8 * HOUR


def test_a_forced_close_that_fills_across_a_gap_records_how_far_it_reached():
    """The fill is still the next VALID observation. How far away it was is a fact.

    Trigger on bar 0; the next executable observation is four hours later, so the
    forced close is four hours after its trigger rather than one — visible in the
    record instead of implied by two instants a reader has to subtract.
    """
    quotes = [
        bar(0, "100", "100", mark_high=PREDICATE_SPIKE),
        bar(4 * HOUR, "100", "103"),
        bar(5 * HOUR, "100", "100"),
    ]
    result = _run(quotes)
    assert result.liquidated
    assert result.liquidation_instant_ns == 0
    assert result.forced_close_instant_ns == 4 * HOUR
    assert result.forced_close_gap_ns == 4 * HOUR
    assert result.forced_close_gap_ns > NOMINAL_BAR_NS
    # The rule itself is unchanged: it filled at that bar's OPEN, basis 3.
    assert result.basis_exit == D("3")
    assert result.net_pnl == D("-15")


def test_a_forced_close_on_a_contiguous_grid_reaches_exactly_one_bar():
    quotes = [
        bar(0, "100", "100", mark_high=PREDICATE_SPIKE),
        bar(HOUR, "100", "103"),
        bar(2 * HOUR, "100", "100"),
    ]
    result = _run(quotes)
    assert result.forced_close_gap_ns == NOMINAL_BAR_NS


def test_a_block_with_no_forced_close_has_no_forced_close_gap():
    assert _run([bar(0, "100", "100"), bar(HOUR, "100", "100")]).forced_close_gap_ns is None


def test_the_nominal_bar_is_the_hourly_grid_the_design_froze():
    """Read off the frozen sources rather than chosen here."""
    from nn.p13_preregistration import DATA_SOURCES

    assert NOMINAL_BAR_NS == 3600 * 1_000_000_000
    klines = [s for s in DATA_SOURCES if s["field"].endswith("price")]
    assert len(klines) == 3
    for source in klines:
        assert "/1h/" in source["archive"]
        assert "open + 1h" in source["timestamp_semantics"]


# ---------------------------------------------------------------------------
# The excursion's unit, which the frozen design fixes and the code did not
# ---------------------------------------------------------------------------
#
# VIABILITY_GATE.maximum_adverse_excursion: "the most negative value of
# (equity_t - total_starting_capital) over the holding period, AS A FRACTION OF
# TOTAL CAPITAL — the same base as the block return, so the two are comparable."
#
# It was reported in quote units. At the design's 1,000,000 USDT that is a
# millionfold misreading of a field a reader is invited to compare against G3's
# -0.02 floor. The frozen text settles the unit, so reporting the fraction
# conforms to the preregistration rather than amending it.
# ---------------------------------------------------------------------------


def test_the_excursion_is_reported_on_the_same_base_as_the_block_return():
    """The comparability the frozen sentence asks for, asserted as a ratio."""
    quotes = [
        bar(0, "100", "100", spot_close="100", perp_close="140"),
        bar(HOUR, "100", "100"),
        bar(2 * HOUR, "100", "102"),
    ]
    result = _run(quotes)
    # By hand: equity at bar 0's close = 0 + 5x100 + 500 + (100-140)x5 = 800,
    # an excursion of -200 against 1,000 of capital, so -0.2 as a fraction.
    assert result.max_adverse_excursion_pnl == D("-200")
    assert result.max_adverse_excursion == D("-0.2")
    # And the pair holds the same relationship net_pnl / net_return does — stated
    # as hand-derived literals on both halves rather than by dividing one of them
    # by the capital base, which would only restate the production line.
    assert result.net_pnl == D("-10") and result.net_return == D("-0.01")


def test_the_excursion_fraction_is_scale_free():
    """The property that makes it comparable at all: 1,000 or 1,000,000, same number.

    CAPITAL_CONTRACT.returns_are_fractions says the scale is not a parameter and
    any scale yields the same fractional result. An excursion in quote units does
    not have that property, which is why it could not be compared to -0.02.
    """
    quotes = [
        bar(0, "100", "100", spot_close="100", perp_close="140"),
        bar(HOUR, "100", "100"),
        bar(2 * HOUR, "100", "100"),
    ]
    small = _run(quotes)
    big = evaluate_block(
        "b",
        quotes,
        [],
        Allocation(total_capital=D("1000000"), spot=D("500000"), perp=D("500000")),
        FREE,
        VENUE,
        min_settlements=0,
    )
    assert small.max_adverse_excursion_pnl != big.max_adverse_excursion_pnl
    assert small.max_adverse_excursion == big.max_adverse_excursion == D("-0.2")


# ---------------------------------------------------------------------------
# The excursion covers the WHOLE holding period, ends included
# ---------------------------------------------------------------------------
#
# VIABILITY_GATE.maximum_adverse_excursion: "the most negative value of
# (equity_t - total_starting_capital) OVER THE HOLDING PERIOD". The holding
# period starts at the entry instant and ends at the close instant, and
# VIABILITY_GATE.block_net_pnl defines the block return as "equity AT THE
# BLOCK'S CLOSE minus total_starting_capital". So the close is one of the points
# the excursion ranges over, and `max_adverse_excursion <= net_return` is not an
# observation about these witnesses — it is an identity the definition forces.
#
# Sampling only the held bars' CLOSES broke it at both ends: the entry instant
# was never marked (so an opened block could report an excursion of exactly zero
# while having already paid its entry frictions), and the close instant was never
# marked (so a funding payment settling exactly at the close — which the frozen
# boundary tie rule DOES apply — reached the block return without ever reaching
# the drawdown).
#
# These witnesses run at REAL frictions on purpose. Every other excursion test in
# this file uses FREE, and in a zero-cost world entry equity equals capital
# exactly, so the whole defect class is invisible to them by construction.
# ---------------------------------------------------------------------------


def _run_real(quotes, settlements=(), isolated=False):
    return evaluate_block(
        "b",
        quotes,
        list(settlements),
        CAPITAL,
        REAL,
        VENUE,
        min_settlements=0,
        isolated=isolated,
    )


# Hand-traced at the frozen rates, capital 1,000 split 500/500, price 100 flat:
#   Q = step_floor(min(500/100.15, 500/100.10)) = step_floor(4.99251...) = 4.992
#   notional per leg      4.992 x 100 = 499.20
#   entry frictions       499.20 x (0.001 + 0.0005) spot  = 0.7488
#                       + 499.20 x (0.0005 + 0.0005) perp = 0.4992   -> 1.2480
#   exit frictions        the same again                             -> 1.2480
#   equity at entry       1000 - 1.2480 = 998.7520
#   equity at close       1000 - 2.4960 = 997.5040
ENTRY_FRICTION = D("1.2480")
ROUND_TRIP = D("2.4960")


def test_a_flat_block_at_real_frictions_has_a_hand_traced_excursion():
    result = _run_real([bar(0, "100", "100"), bar(HOUR, "100", "100")])
    assert result.quantity == D("4.992")
    assert result.fees + result.slippage == ROUND_TRIP
    assert result.net_pnl == -ROUND_TRIP
    # The deepest point IS the close here, because nothing else moved.
    assert result.max_adverse_excursion_pnl == -ROUND_TRIP
    assert result.max_adverse_excursion == -ROUND_TRIP / CAPITAL.total_capital


def test_an_opened_block_cannot_report_a_zero_excursion_once_frictions_are_charged():
    """Equity at the entry instant is capital minus the two legs' entry frictions.

    That is the equity invariant asserted elsewhere in this file, so an opened
    block's excursion is strictly negative whenever any friction is charged, and
    a reported zero is not a shallow drawdown — it is a missing measurement.
    """
    result = _run_real([bar(0, "100", "100"), bar(HOUR, "100", "100")])
    assert result.opened
    assert result.max_adverse_excursion_pnl < ZERO
    assert result.max_adverse_excursion_pnl <= -ENTRY_FRICTION


def test_a_settlement_at_the_close_instant_reaches_the_excursion():
    """It reaches the block return, so it must reach the drawdown too.

    The frozen boundary tie rule applies a settlement whose instant EQUALS the
    close. The held-bar loop stops one bar earlier, so this payment lands after
    the last bar sample — and before this repair it was charged to the result
    while leaving the reported drawdown untouched.
    """
    # 4.992 BTC at a mark of 100 is 499.20 of notional; a -0.01 rate makes the
    # SHORT pay 4.9920, on top of the 2.4960 round trip.
    at_close = FundingSettlement(HOUR, D("-0.01"), D("100"))
    result = _run_real([bar(0, "100", "100"), bar(HOUR, "100", "100")], [at_close])
    assert result.settlements == 1
    assert result.funding_paid == D("4.99200")
    assert result.net_pnl == -(ROUND_TRIP + D("4.99200"))
    assert result.max_adverse_excursion_pnl == result.net_pnl

    # The negative control: the same payment one nanosecond later is not this
    # position's, and then it is in neither number.
    after = FundingSettlement(HOUR + 1, D("-0.01"), D("100"))
    clean = _run_real([bar(0, "100", "100"), bar(HOUR, "100", "100")], [after])
    assert clean.settlements == 0
    assert clean.net_pnl == -ROUND_TRIP
    assert clean.max_adverse_excursion_pnl == -ROUND_TRIP


@pytest.mark.parametrize("isolated", [False, True], ids=["portfolio", "isolated"])
def test_the_excursion_is_never_shallower_than_the_block_return(isolated):
    """The identity the frozen definition forces, over every shape and both models.

    Not a property of these prices: the close is inside the holding period, so a
    block cannot finish worse than its own worst point. A violation means the
    excursion is not ranging over the period the design says it ranges over.
    """
    paths = [
        ["100", "100", "100"],
        ["100", "120", "90"],
        ["100", "90", "140"],
        ["100", "101", "99"],
        ["100", "100"],
        ["100", "105", "95", "115", "85"],
    ]
    for path in paths:
        for skew in ("100", "103", "97"):
            quotes = [
                bar(i * HOUR, price, skew if i == 0 else price) for i, price in enumerate(path)
            ]
            result = _run_real(quotes, isolated=isolated)
            if not result.opened or result.unclosed:
                continue
            assert result.max_adverse_excursion <= result.net_return, (
                f"path={path} skew={skew}: drawdown {result.max_adverse_excursion} is "
                f"shallower than the realised return {result.net_return}"
            )


# ---------------------------------------------------------------------------
# Two funding rows at one instant that disagree
# ---------------------------------------------------------------------------
#
# FUNDING_SEMANTICS.application deduplicates "a redelivered or duplicated archive
# row" — a row delivered TWICE. Two rows at one instant carrying DIFFERENT rates
# are not that; they are exactly what POSITION_LIFECYCLE.validity_definition
# calls invalid: "no duplicate row makes the instant ambiguous. Anything else is
# invalid and fails closed."
#
# Collapsing them by last-writer-wins let the CALLER'S LIST ORDER choose which
# rate the payoff variable took — the same defect R4 removed from the quote path.
# ---------------------------------------------------------------------------


def test_two_settlements_at_one_instant_that_disagree_are_refused():
    positive = FundingSettlement(HOUR, D("0.001"), D("100"))
    negative = FundingSettlement(HOUR, D("-0.001"), D("100"))
    for order in ([positive, negative], [negative, positive]):
        with pytest.raises(CarryError, match="ambiguous"):
            _run(_settled_series(), order)


def test_settlements_at_one_instant_that_disagree_only_on_the_mark_are_refused():
    """The notional base is as decision-relevant as the rate."""
    one = FundingSettlement(HOUR, D("0.001"), D("100"))
    other = FundingSettlement(HOUR, D("0.001"), D("200"))
    with pytest.raises(CarryError, match="ambiguous"):
        _run(_settled_series(), [one, other])


def test_an_identical_redelivered_row_is_still_silently_deduplicated():
    """The frozen sentence this must NOT break: a duplicate row changes nothing."""
    once = FundingSettlement(HOUR, D("0.001"), D("100"))
    twin = FundingSettlement(HOUR, D("0.001"), D("100"))
    plain = _run(_settled_series(), [once])
    doubled = _run(_settled_series(), [once, twin, once])
    assert plain.settlements == doubled.settlements == 1
    assert plain.funding_received == doubled.funding_received == D("0.500")
    assert plain.net_pnl == doubled.net_pnl


def test_the_entry_instant_is_the_excursions_first_sample():
    """A block that finishes UP still had a worst point, and it is the entry.

    Hand-traced. Entry fills 100 spot / 102 perp, exit fills 100/100, and the
    entry bar closes at a basis of 0 so the short is already ahead by the time
    that bar ends:

      Q = step_floor(min(500/100.15, 500/(102 x 1.001))) = step_floor(4.8971...)
        = 4.897
      spot notional 489.70,  perp notional 4.897 x 102 = 499.494 = margin
      entry frictions  489.70 x 0.0015 + 499.494 x 0.0010 = 1.234044
      equity at entry  1000 - 1.234044        -> excursion -1.234044
      equity at bar 0's close  +8.559956      (the short is up 2 x 4.897)
      basis PnL 4.897 x (2 - 0) = 9.794, exit frictions 1.224250
      net PnL  9.794 - 1.234044 - 1.224250 = 7.335706  -> finishes UP

    Every sample except the entry is positive, so the entry instant is the only
    thing standing between this block and a reported drawdown of zero — which is
    what a floor of zero used to report, for a position that had already paid
    1.234044 of frictions the moment it opened.
    """
    quotes = [
        bar(0, "100", "102", spot_close="100", perp_close="100"),
        bar(HOUR, "100", "100"),
    ]
    result = _run_real(quotes)

    assert result.quantity == D("4.897")
    assert result.net_pnl == D("7.335706")
    assert result.net_pnl > ZERO
    assert result.max_adverse_excursion_pnl == D("-1.234044")
    assert result.max_adverse_excursion_pnl == -(result.fees + result.slippage - D("1.224250"))
    assert result.max_adverse_excursion == D("-1.234044") / CAPITAL.total_capital
    # The claim stated as the thing that would break: a zero floor reports 0 here.
    assert result.max_adverse_excursion_pnl != ZERO
