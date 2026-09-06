"""``chimera.carry.accounting`` equals ``nn.p13_carry``, on synthetic worlds only.

Section 6.9 of the adopted master plan reuses P13's arithmetic by PORT rather
than by import: :mod:`nn.p13_carry` pulls ``pandas`` in through
``nn.p13_preregistration``, and section 16 forbids the demo runtime from
importing historical research code. A port with no test is just a second
implementation waiting to drift, so this module asserts term-by-term equality.

**This file is the only place in PR-08 that may import ``nn.p13_carry``**, and
it may do so only because a test is not the runtime. What it evaluates is
arithmetic on numbers chosen by the test author -- flat worlds, hand-traced
witnesses, seeded pseudo-random Decimals. **No P13 historical market data is
read here, no historical series is opened, and no economic result of the P13
checkpoint is computed.** P13's economics remain unrun.
"""

from __future__ import annotations

import random
from decimal import Decimal

import pytest

from chimera.carry import accounting as carry
from chimera.futures.accounting import FundingEvent, funding_cash_flow
from chimera.futures.domain import Position, PositionSide
from nn import p13_carry as p13

D = Decimal
ZERO = D("0")

# The frozen P13 fixtures, restated so a drift in either module is visible here.
VENUE_KWARGS = dict(
    step_size=D("0.001"), min_notional=D("5"), maintenance_margin_rate=D("0.004")
)
CAPITAL_KWARGS = dict(total_capital=D("1000"), spot=D("500"), perp=D("500"))
FREE_KWARGS = dict(spot_fee=ZERO, spot_slippage=ZERO, perp_fee=ZERO, perp_slippage=ZERO)
REAL_KWARGS = dict(
    spot_fee=D("0.001"),
    spot_slippage=D("0.0005"),
    perp_fee=D("0.0005"),
    perp_slippage=D("0.0005"),
)

POSITION_FIELDS = (
    "quantity",
    "spot_entry",
    "perp_entry",
    "perp_margin",
    "free_cash",
    "leverage",
    "fees",
    "slippage",
    "funding_received",
    "funding_paid",
    "perp_funding",
    "settled",
)


def _pair(module, kind: str, **kwargs):
    return getattr(module, kind)(**kwargs)


def both(kind: str, **kwargs):
    """The same construction in both modules."""
    return _pair(carry, kind, **kwargs), _pair(p13, kind, **kwargs)


def quotes(
    instant: int, spot: str, perp: str, mark: str | None = None, high: str | None = None
):
    """A flat witness quote in both modules: open EQUALS close, stated out loud."""
    kwargs = dict(
        instant_ns=instant,
        spot=D(spot),
        perp=D(perp),
        mark=D(mark if mark is not None else spot),
        spot_open=D(spot),
        perp_open=D(perp),
    )
    if high is not None:
        kwargs["mark_high"] = D(high)
    return carry.Quote(**kwargs), p13.Quote(**kwargs)


def assert_positions_equal(mine, theirs, label: str = "") -> None:
    for name in POSITION_FIELDS:
        assert getattr(mine, name) == getattr(theirs, name), f"{label}{name}"
    assert mine.net_funding == theirs.net_funding, f"{label}net_funding"


# ---------------------------------------------------------------------------
# The constants and the refusals
# ---------------------------------------------------------------------------


def test_the_ported_constants_are_the_frozen_ones():
    assert carry.ZERO == p13.ZERO
    assert carry.ONE == p13.ONE
    assert (
        carry.FundingSettlement.MAX_PLAUSIBLE_RATE == p13.FundingSettlement.MAX_PLAUSIBLE_RATE
    )
    assert carry.TOUCH_MARK_HIGH == p13.TOUCH_MARK_HIGH
    assert carry.TOUCH_MARK_CLOSE == p13.TOUCH_MARK_CLOSE


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(spot_fee=D("-0.001"), spot_slippage=ZERO, perp_fee=ZERO, perp_slippage=ZERO),
        dict(spot_fee=D("1"), spot_slippage=ZERO, perp_fee=ZERO, perp_slippage=ZERO),
        dict(spot_fee=0.001, spot_slippage=ZERO, perp_fee=ZERO, perp_slippage=ZERO),
    ],
)
def test_costs_refuse_the_same_inputs(kwargs):
    with pytest.raises(carry.CarryError):
        carry.Costs(**kwargs)
    with pytest.raises(p13.CarryError):
        p13.Costs(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(step_size=ZERO, min_notional=D("5"), maintenance_margin_rate=D("0.004")),
        dict(step_size=D("0.001"), min_notional=D("-1"), maintenance_margin_rate=D("0.004")),
        dict(step_size=D("0.001"), min_notional=D("5"), maintenance_margin_rate=D("1")),
    ],
)
def test_venue_refuses_the_same_inputs(kwargs):
    with pytest.raises(carry.CarryError):
        carry.Venue(**kwargs)
    with pytest.raises(p13.CarryError):
        p13.Venue(**kwargs)


def test_allocation_refuses_arithmetic_leverage_identically():
    bad = dict(total_capital=D("1000"), spot=D("600"), perp=D("500"))
    with pytest.raises(carry.CarryError):
        carry.Allocation(**bad)
    with pytest.raises(p13.CarryError):
        p13.Allocation(**bad)


def test_a_percent_shaped_funding_rate_is_refused_by_both():
    """MAX_PLAUSIBLE_RATE is a unit detector; it must refuse, not clip."""
    bad = dict(instant_ns=1, rate=D("0.5"), mark_price=D("100"))
    with pytest.raises(carry.CarryError):
        carry.FundingSettlement(**bad)
    with pytest.raises(p13.CarryError):
        p13.FundingSettlement(**bad)


def test_a_quote_without_an_execution_open_refuses_in_both():
    mine = carry.Quote(instant_ns=0, spot=D("100"), perp=D("101"), mark=D("100"))
    theirs = p13.Quote(instant_ns=0, spot=D("100"), perp=D("101"), mark=D("100"))
    for quote in (mine, theirs):
        assert not quote.has_execution_opens
    with pytest.raises(carry.CarryError):
        mine.spot_fill
    with pytest.raises(p13.CarryError):
        theirs.spot_fill


# ---------------------------------------------------------------------------
# step_floor, sizing, opening
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    ["4.9504950495", "0.0009", "0.001", "123.4567", "0", "1000.0005"],
)
def test_step_floor_agrees(raw):
    mine, theirs = both("Venue", **VENUE_KWARGS)
    assert mine.step_floor(D(raw)) == theirs.step_floor(D(raw))


@pytest.mark.parametrize(
    "spot,perp",
    [("100", "101"), ("100", "100"), ("100", "102"), ("30000", "30030"), ("100", "99")],
)
def test_hedge_quantity_agrees(spot, perp):
    entry_mine, entry_theirs = quotes(0, spot, perp)
    venue_mine, venue_theirs = both("Venue", **VENUE_KWARGS)
    alloc_mine, alloc_theirs = both("Allocation", **CAPITAL_KWARGS)
    for costs_kwargs in (FREE_KWARGS, REAL_KWARGS):
        costs_mine, costs_theirs = both("Costs", **costs_kwargs)
        assert carry.hedge_quantity(
            entry_mine, alloc_mine, costs_mine, venue_mine
        ) == p13.hedge_quantity(entry_theirs, alloc_theirs, costs_theirs, venue_theirs)


@pytest.mark.parametrize("costs_kwargs", [FREE_KWARGS, REAL_KWARGS])
@pytest.mark.parametrize("spot,perp", [("100", "101"), ("100", "100"), ("30000", "30030")])
def test_open_carry_agrees_field_by_field(costs_kwargs, spot, perp):
    entry_mine, entry_theirs = quotes(0, spot, perp)
    venue_mine, venue_theirs = both("Venue", **VENUE_KWARGS)
    alloc_mine, alloc_theirs = both("Allocation", **CAPITAL_KWARGS)
    costs_mine, costs_theirs = both("Costs", **costs_kwargs)
    assert_positions_equal(
        carry.open_carry(entry_mine, alloc_mine, costs_mine, venue_mine),
        p13.open_carry(entry_theirs, alloc_theirs, costs_theirs, venue_theirs),
    )


def test_open_carry_refuses_the_same_impossible_sizes():
    """Below min_notional, and a step that rounds the quantity away."""
    entry_mine, entry_theirs = quotes(0, "100", "101")
    costs_mine, costs_theirs = both("Costs", **FREE_KWARGS)
    alloc_mine, alloc_theirs = both("Allocation", **CAPITAL_KWARGS)

    big_step = dict(
        step_size=D("1000"), min_notional=D("5"), maintenance_margin_rate=D("0.004")
    )
    with pytest.raises(carry.CarryError):
        carry.open_carry(entry_mine, alloc_mine, costs_mine, carry.Venue(**big_step))
    with pytest.raises(p13.CarryError):
        p13.open_carry(entry_theirs, alloc_theirs, costs_theirs, p13.Venue(**big_step))

    huge_min = dict(
        step_size=D("0.001"), min_notional=D("100000"), maintenance_margin_rate=D("0.004")
    )
    with pytest.raises(carry.CarryError):
        carry.open_carry(entry_mine, alloc_mine, costs_mine, carry.Venue(**huge_min))
    with pytest.raises(p13.CarryError):
        p13.open_carry(entry_theirs, alloc_theirs, costs_theirs, p13.Venue(**huge_min))


# ---------------------------------------------------------------------------
# Funding: flow, accumulators, dedup, and the A10 sign
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rate", ["0.0001", "-0.0001", "0.01", "-0.01", "0"])
def test_apply_funding_agrees_on_flow_and_accumulators(rate):
    entry_mine, entry_theirs = quotes(0, "100", "101")
    venue_mine, venue_theirs = both("Venue", **VENUE_KWARGS)
    alloc_mine, alloc_theirs = both("Allocation", **CAPITAL_KWARGS)
    costs_mine, costs_theirs = both("Costs", **FREE_KWARGS)
    mine = carry.open_carry(entry_mine, alloc_mine, costs_mine, venue_mine)
    theirs = p13.open_carry(entry_theirs, alloc_theirs, costs_theirs, venue_theirs)

    kwargs = dict(instant_ns=1, rate=D(rate), mark_price=D("100"))
    flow_mine = carry.apply_funding(mine, carry.FundingSettlement(**kwargs))
    flow_theirs = p13.apply_funding(theirs, p13.FundingSettlement(**kwargs))
    assert flow_mine == flow_theirs
    assert_positions_equal(mine, theirs)

    # And a repeat of the same settlement id books nothing, in both.
    assert carry.apply_funding(mine, carry.FundingSettlement(**kwargs)) == ZERO
    assert p13.apply_funding(theirs, p13.FundingSettlement(**kwargs)) == ZERO
    assert_positions_equal(mine, theirs, "after duplicate: ")


@pytest.mark.parametrize("rate", ["0.0001", "-0.0001", "0.0003"])
def test_the_funding_flow_is_the_repository_wide_convention(rate):
    """Parity is not enough: the flow must equal `chimera.futures.accounting`."""
    entry, _ = quotes(0, "100", "101")
    venue = carry.Venue(**VENUE_KWARGS)
    position = carry.open_carry(
        entry, carry.Allocation(**CAPITAL_KWARGS), carry.Costs(**FREE_KWARGS), venue
    )
    settlement = carry.FundingSettlement(instant_ns=7, rate=D(rate), mark_price=D("100"))
    flow = carry.apply_funding(position, settlement)

    # The same settlement expressed in the executor's own domain objects.
    leg = Position(
        symbol="BTC/USDT:USDT",
        side=PositionSide.SHORT,
        quantity=position.quantity,
        entry_price=position.perp_entry,
    )
    event = FundingEvent(
        symbol="BTC/USDT:USDT",
        rate=settlement.rate,
        mark_price=settlement.mark_price,
        settlement_id="s7",
    )
    assert flow == funding_cash_flow(leg, event)


@pytest.mark.parametrize(
    "side,rate,pays",
    [
        (PositionSide.LONG, D("0.0001"), True),
        (PositionSide.LONG, D("-0.0001"), False),
        (PositionSide.SHORT, D("0.0001"), False),
        (PositionSide.SHORT, D("-0.0001"), True),
    ],
)
def test_amendment_a10_cost_rate_has_the_four_intended_cases(side, rate, pays):
    """A10: the paid cost rate is ``sign(side) * rate``, not ``-sign(side) * rate``.

    Asserted against the cash-flow convention rather than restated, so the two
    cannot drift: a negative cash flow means the position paid.
    """
    cost_rate = Decimal(side.sign) * rate
    assert (cost_rate > ZERO) is pays

    leg = Position(symbol="BTC/USDT:USDT", side=side, quantity=D("10"), entry_price=D("100"))
    event = FundingEvent(
        symbol="BTC/USDT:USDT", rate=rate, mark_price=D("100"), settlement_id="a10"
    )
    notional = leg.notional(event.mark_price)
    flow = funding_cash_flow(leg, event)
    assert (flow < ZERO) is pays
    assert cost_rate == -flow / notional


# ---------------------------------------------------------------------------
# Liquidation, both models
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("isolated", [True, False])
@pytest.mark.parametrize(
    "spot,perp,mark,high",
    [
        ("100", "101", "100", None),
        ("100", "101", "100", "150"),
        ("100", "101", "100", "205"),
        ("100", "101", "260", None),
        ("50", "50", "50", "60"),
    ],
)
def test_is_liquidated_agrees_in_both_models(isolated, spot, perp, mark, high):
    entry_mine, entry_theirs = quotes(0, "100", "101")
    venue_mine, venue_theirs = both("Venue", **VENUE_KWARGS)
    alloc_mine, alloc_theirs = both("Allocation", **CAPITAL_KWARGS)
    costs_mine, costs_theirs = both("Costs", **FREE_KWARGS)
    mine = carry.open_carry(entry_mine, alloc_mine, costs_mine, venue_mine)
    theirs = p13.open_carry(entry_theirs, alloc_theirs, costs_theirs, venue_theirs)

    bar_mine, bar_theirs = quotes(1, spot, perp, mark, high)
    assert carry.is_liquidated(mine, bar_mine, venue_mine, isolated) == p13.is_liquidated(
        theirs, bar_theirs, venue_theirs, isolated
    )
    assert bar_mine.liquidation_touch == bar_theirs.liquidation_touch
    assert bar_mine.liquidation_touch_source == bar_theirs.liquidation_touch_source
    assert bar_mine.liquidation_touch_is_high == bar_theirs.liquidation_touch_is_high


def test_a_bar_with_no_mark_series_refuses_the_liquidation_test_in_both():
    mine = carry.Quote(
        instant_ns=1, spot=D("100"), perp=D("101"), spot_open=D("100"), perp_open=D("101")
    )
    theirs = p13.Quote(
        instant_ns=1, spot=D("100"), perp=D("101"), spot_open=D("100"), perp_open=D("101")
    )
    with pytest.raises(carry.CarryError):
        mine.liquidation_touch
    with pytest.raises(p13.CarryError):
        theirs.liquidation_touch


# ---------------------------------------------------------------------------
# Closing, equity, basis
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("forfeit", [False, True])
@pytest.mark.parametrize("costs_kwargs", [FREE_KWARGS, REAL_KWARGS])
@pytest.mark.parametrize("exit_spot,exit_perp", [("200", "200"), ("10", "10"), ("100", "100")])
def test_close_carry_agrees(forfeit, costs_kwargs, exit_spot, exit_perp):
    entry_mine, entry_theirs = quotes(0, "100", "101")
    exit_mine, exit_theirs = quotes(3_600_000_000_000, exit_spot, exit_perp)
    venue_mine, venue_theirs = both("Venue", **VENUE_KWARGS)
    alloc_mine, alloc_theirs = both("Allocation", **CAPITAL_KWARGS)
    costs_mine, costs_theirs = both("Costs", **costs_kwargs)
    mine = carry.open_carry(entry_mine, alloc_mine, costs_mine, venue_mine)
    theirs = p13.open_carry(entry_theirs, alloc_theirs, costs_theirs, venue_theirs)

    assert carry.close_carry(mine, exit_mine, costs_mine, forfeit) == p13.close_carry(
        theirs, exit_theirs, costs_theirs, forfeit
    )
    assert_positions_equal(mine, theirs)


@pytest.mark.parametrize("spot,perp", [("100", "101"), ("200", "199"), ("30000", "30030")])
def test_equity_and_basis_agree(spot, perp):
    entry_mine, entry_theirs = quotes(0, "100", "101")
    venue_mine, venue_theirs = both("Venue", **VENUE_KWARGS)
    alloc_mine, alloc_theirs = both("Allocation", **CAPITAL_KWARGS)
    costs_mine, costs_theirs = both("Costs", **REAL_KWARGS)
    mine = carry.open_carry(entry_mine, alloc_mine, costs_mine, venue_mine)
    theirs = p13.open_carry(entry_theirs, alloc_theirs, costs_theirs, venue_theirs)

    bar_mine, bar_theirs = quotes(1, spot, perp)
    assert mine.equity(bar_mine) == theirs.equity(bar_theirs)
    assert mine.equity_at(D(spot), D(perp)) == theirs.equity_at(D(spot), D(perp))
    assert mine.unrealised_perp(D(perp)) == theirs.unrealised_perp(D(perp))
    assert bar_mine.basis == bar_theirs.basis
    assert bar_mine.fill_basis == bar_theirs.fill_basis


# ---------------------------------------------------------------------------
# The three hand-traced witnesses, reproduced against the PORT
# ---------------------------------------------------------------------------


def test_witness_a_price_pnl_is_exactly_the_basis_move():
    """Q x (basis_in - basis_out), with the price doubling absent."""
    entry, _ = quotes(0, "100", "101")
    exit_, _ = quotes(3_600_000_000_000, "200", "200")
    venue = carry.Venue(**VENUE_KWARGS)
    capital = carry.Allocation(**CAPITAL_KWARGS)
    free = carry.Costs(**FREE_KWARGS)

    position = carry.open_carry(entry, capital, free, venue)
    assert position.quantity == D("4.950")
    assert position.perp_margin == D("499.95")
    assert position.free_cash == D("5.05")

    final = carry.close_carry(position, exit_, free)
    assert final == D("1004.95")
    assert final - capital.total_capital == D("4.95")
    assert final - capital.total_capital == D("4.950") * (entry.basis - exit_.basis)


def test_witness_b_short_receives_when_funding_is_positive():
    """4.950 BTC x 100 mark x 0.0001 = 0.0495, received by the SHORT."""
    entry, _ = quotes(0, "100", "101")
    position = carry.open_carry(
        entry,
        carry.Allocation(**CAPITAL_KWARGS),
        carry.Costs(**FREE_KWARGS),
        carry.Venue(**VENUE_KWARGS),
    )
    cash_before = position.free_cash

    flow = carry.apply_funding(
        position, carry.FundingSettlement(instant_ns=1, rate=D("0.0001"), mark_price=D("100"))
    )
    assert flow == D("0.0495")
    assert position.funding_received == D("0.0495")
    assert position.funding_paid == ZERO
    assert position.free_cash == cash_before + D("0.0495")


def test_witness_c_flat_prices_lose_exactly_the_round_trip_friction():
    """Flat in and out, so every penny of the result is friction: -2.4960."""
    entry, _ = quotes(0, "100", "100")
    exit_, _ = quotes(1, "100", "100")
    capital = carry.Allocation(**CAPITAL_KWARGS)
    real = carry.Costs(**REAL_KWARGS)

    position = carry.open_carry(entry, capital, real, carry.Venue(**VENUE_KWARGS))
    assert position.quantity == D("4.992")

    final = carry.close_carry(position, exit_, real)
    net = final - capital.total_capital
    assert position.fees + position.slippage == D("2.4960")
    assert net == D("-2.4960")
    assert net == -(position.fees + position.slippage)


# ---------------------------------------------------------------------------
# Randomised parity, fixed seed. The seed is part of the test.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [11, 23, 47, 101, 997])
def test_randomised_worlds_agree_through_a_whole_life_cycle(seed):
    """Open, settle several fundings, mark, test liquidation, close -- both modules."""
    rng = random.Random(seed)
    venue_mine, venue_theirs = both("Venue", **VENUE_KWARGS)
    alloc_mine, alloc_theirs = both("Allocation", **CAPITAL_KWARGS)
    costs_mine, costs_theirs = both("Costs", **REAL_KWARGS)

    spot = D(rng.randrange(5_000, 60_000))
    basis = D(rng.randrange(-50, 200)) / D("10")
    entry_mine, entry_theirs = quotes(0, str(spot), str(spot + basis))
    mine = carry.open_carry(entry_mine, alloc_mine, costs_mine, venue_mine)
    theirs = p13.open_carry(entry_theirs, alloc_theirs, costs_theirs, venue_theirs)
    assert_positions_equal(mine, theirs, f"seed {seed} open: ")

    for index in range(rng.randrange(1, 6)):
        rate = D(rng.randrange(-30, 31)) / D("100000")
        mark = spot + D(rng.randrange(-200, 200)) / D("10")
        kwargs = dict(instant_ns=index + 1, rate=rate, mark_price=mark)
        assert carry.apply_funding(
            mine, carry.FundingSettlement(**kwargs)
        ) == p13.apply_funding(theirs, p13.FundingSettlement(**kwargs))
        assert_positions_equal(mine, theirs, f"seed {seed} funding {index}: ")

    move = D(rng.randrange(-4_000, 4_000)) / D("10")
    bar_mine, bar_theirs = quotes(
        9, str(spot + move), str(spot + move + basis), str(spot + move), str(spot + move + 25)
    )
    assert mine.equity(bar_mine) == theirs.equity(bar_theirs)
    for isolated in (True, False):
        assert carry.is_liquidated(mine, bar_mine, venue_mine, isolated) == p13.is_liquidated(
            theirs, bar_theirs, venue_theirs, isolated
        )

    assert carry.close_carry(mine, bar_mine, costs_mine) == p13.close_carry(
        theirs, bar_theirs, costs_theirs
    )
    assert_positions_equal(mine, theirs, f"seed {seed} close: ")


def test_the_runtime_package_does_not_import_the_historical_engine():
    """The port exists so `chimera/carry/` never reaches into `nn/`."""
    import ast
    from pathlib import Path

    package = Path(carry.__file__).parent
    offenders = []
    for path in sorted(package.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if name == "nn" or name.startswith("nn."):
                    offenders.append(f"{path.name}:{node.lineno}: {name}")
    assert not offenders, (
        "chimera/carry/ must not import historical research code (section 16 tier "
        f"table; nn.p13_carry pulls pandas through nn.p13_preregistration): {offenders}"
    )


def test_only_the_factory_constructs_a_venue():
    """Section 7.6 step 3, asserted for the half of it PR-08 owns.

    The architectural proof that no position increase bypasses Aegis rests on
    the demo runner constructing venues *only* inside the hedged position's
    factory. The full AST scan over `chimera/demo/` is PR-10's
    `tests/test_demo_no_live_path.py`; this is the `chimera/carry/` half, so the
    constraint is enforced from the moment the package exists rather than from
    the moment the scan arrives.
    """
    import ast
    from pathlib import Path

    package = Path(carry.__file__).parent
    offenders = []
    for path in sorted(package.rglob("*.py")):
        if path.name == "factory.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            module = ""
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
            elif isinstance(node, ast.Import):
                module = ",".join(a.name for a in node.names)
            if "chimera.futures.venue" in module:
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, (
        "only chimera/carry/factory.py may import chimera.futures.venue; " f"found {offenders}"
    )


def test_nothing_in_the_carry_package_opens_a_socket_or_reads_a_credential():
    """No network and no credentials anywhere on the carry path."""
    import ast
    from pathlib import Path

    forbidden_modules = {"socket", "ssl", "http", "urllib", "requests", "websockets", "ccxt"}
    offenders = []
    package = Path(carry.__file__).parent
    for path in sorted(package.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if name.split(".")[0] in forbidden_modules:
                    offenders.append(f"{path.name}:{node.lineno}: {name}")
        for marker in ("os.environ", "getenv", "API_KEY", "SECRET"):
            assert marker not in source, f"{path.name} touches {marker}"
    assert not offenders, f"the carry package must not import a network module: {offenders}"
