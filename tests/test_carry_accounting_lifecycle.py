"""Carry accounting across a whole lifecycle, against an oracle that is not the ledger.

Every expectation here is built from the two EXECUTORS' own accumulators and
from the frictions the position measured before the ledger saw them. That is
deliberate and it is the lesson of the rounds this file was written after: the
reduction tests that passed while the accounting was wrong computed the exit
frictions from `ledger.fees` -- the accumulator that was not moving -- so the
identity was exact against itself.

Synthetic worlds only. No market data is read and no economic claim follows from
any number here.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from decimal import Decimal as D

import pytest

from chimera.carry.factory import build_hedged_position
from chimera.carry.hedge import (
    PERP,
    SPOT,
    HedgedPosition,
    HedgeState,
    HedgeTarget,
    PerpSettlement,
)
from chimera.carry.ledger import CarryLedger, LedgerError
from chimera.futures.executor import FlattenCause
from chimera.futures.fills import RecordedQuoteFillModel, TopOfBook
from chimera.risk import RiskEngine, RiskLimits

CAPITAL = D("1000000")
EQUITY = 1_000_000.0
#: The declared Decimal rounding rule for the randomized lifecycle, and nowhere
#: else. See `World.assert_cash_is_right`. Deliberately far tighter than
#: `chimera.carry.ledger.IDENTITY_TOLERANCE` (0.01): a bound loose enough to
#: hide a real accounting error would prove nothing.
DECIMAL_ARTIFACT = D("1e-18")
START_NS = 1_700_000_000_000_000_000
MINUTE_NS = 60_000_000_000


@dataclass(frozen=True)
class Minute:
    """The minimum a carry position needs. Satisfies `CarryMarketState`."""

    minute_ns: int
    complete: bool
    spot_close: D
    perp_close: D
    mark: D
    mark_high: D | None = None


class Recorder:
    """Every friction the position COMPUTES, captured before the ledger sees it.

    Slippage is the one accounting term no executor accumulates, so it is the
    one an oracle cannot recover from the stores. Wrapping `_frictions` -- where
    a fee and a slippage first exist -- is what makes "nothing was dropped" an
    assertion about the code rather than about the accumulator under test.
    """

    def __init__(self, position: HedgedPosition) -> None:
        original = HedgedPosition._frictions
        self.slippage = D("0")
        self.fees = D("0")

        def wrapper(records, reference):
            fee, slip = original(records, reference)
            self.fees += fee
            self.slippage += slip
            return fee, slip

        position._frictions = wrapper


@dataclass
class World:
    position: HedgedPosition
    recorder: Recorder

    def minute(
        self, index: int, *, spot: str = "30000", perp: str = "30030", size: str = "500"
    ) -> Minute:
        """A minute whose closes ARE the books' mids.

        The reference the fills are checked against is the close, so a fixture
        whose book wanders away from its own close has its orders refused for a
        reason the test did not intend. Anchoring the two together means a
        refusal in one of these tests is always the one it staged.
        """
        spot_mid, perp_mid = D(spot), D(perp)
        at = Minute(
            minute_ns=START_NS + index * MINUTE_NS,
            complete=True,
            spot_close=spot_mid,
            perp_close=perp_mid,
            mark=perp_mid,
        )
        for leg, mid in ((SPOT, spot_mid), (PERP, perp_mid)):
            self.position.fill_models[leg].set_quote(
                TopOfBook(
                    instant_ns=at.minute_ns,
                    bid=mid - D("0.5"),
                    bid_qty=D(size),
                    ask=mid + D("0.5"),
                    ask_qty=D(size),
                ),
                at.minute_ns,
            )
        return at

    def target(self, quantity: str, at: Minute):
        """The production path: plan the rule's target, then apply it."""
        position = self.position
        return position.apply(position.plan(HedgeTarget(D(quantity)), at), at, equity=EQUITY)

    # -- the oracle ------------------------------------------------------
    def cash_from_the_legs(self) -> D:
        """``free_cash`` as the two EXECUTORS describe it. Section 6.6's identity.

        ``capital`` less what each leg's inventory cost at that leg's own VWAP,
        less cumulative fees, plus cumulative realised PnL and funding. Every
        term comes from an executor; not one is read from the carry ledger.

        **Slippage is not a term here.** It used to be, subtracted from both
        sides, which made the one accumulator with no independent source agree
        with itself: booking it once, twice or not at all left this oracle
        satisfied. It is not a cash flow -- the crossing it measures is already
        inside the VWAPs the first two terms use -- so removing it from the
        oracle is what makes the oracle independent, and it is asserted
        separately, exactly, in `assert_cash_is_right`.
        """
        position = self.position
        spot, perp = position.leg(SPOT), position.leg(PERP)
        return (
            CAPITAL
            - spot.quantity * spot.entry_price
            - perp.quantity * perp.entry_price
            - (position.spot.ledger.trading_fees + position.perp.ledger.trading_fees)
            + (position.spot.ledger.realised_pnl + position.perp.ledger.realised_pnl)
            + (position.spot.ledger.net_funding + position.perp.ledger.net_funding)
        )

    def assert_cash_is_right(self, where: str, *, tolerance: D = D("0")) -> None:
        """Exact by default. ``tolerance`` is a declared Decimal rule, never a fudge.

        Every deterministic witness in this file asserts EXACT equality and gets
        it. The one place a tolerance is passed is the randomized lifecycle,
        where a leg's VWAP can be a repeating decimal: `Position.apply_fill`
        averages with a division, so `quantity * entry_price` fills the default
        28-digit context, and `free_cash` -- six integer digits wide -- cannot
        then hold the difference exactly. The artifact is bounded and does not
        accumulate: measured over 1600 randomized steps it stays at 3e-22, four
        orders of magnitude inside the bound asserted there and eighteen inside
        `IDENTITY_TOLERANCE`.
        """
        state = self.position.ledger.state
        gap = state.free_cash - self.cash_from_the_legs()
        assert abs(gap) <= tolerance, (
            f"{where}: free_cash {state.free_cash} but the legs say "
            f"{self.cash_from_the_legs()} (gap {gap})"
        )
        assert state.slippage == self.recorder.slippage, (
            f"{where}: the ledger booked {state.slippage} of slippage and "
            f"{self.recorder.slippage} was measured"
        )
        assert state.spot_principal == (
            self.position.leg(SPOT).quantity * self.position.leg(SPOT).entry_price
        ), f"{where}: the spot principal is not what the spot leg holds"
        assert state.perp_margin == (
            self.position.leg(PERP).quantity * self.position.leg(PERP).entry_price
        ), f"{where}: the perpetual margin is not what the perpetual leg holds"


def make_world(tmp_path) -> World:
    risk = RiskEngine(RiskLimits(max_position_pct=1.0, risk_per_trade_pct=0.5))
    risk.update_equity(EQUITY)
    position = build_hedged_position(
        risk=risk, capital=CAPITAL, state_dir=tmp_path, fill_model=RecordedQuoteFillModel()
    )
    position.spot.recover({})
    position.perp.recover({})
    return World(position, Recorder(position))


@pytest.fixture
def world(tmp_path) -> World:
    return make_world(tmp_path)


# ---------------------------------------------------------------------------
# The transitions that had no branch
# ---------------------------------------------------------------------------


def test_a_hedge_completed_by_the_correction_path_books_its_entry(world):
    """The entry booked when the second leg fills a MINUTE LATE, not a cycle late.

    `apply` sends the perpetual first and stops at the first leg that does not
    fill, so a refused spot leg leaves PARTIAL. The next minute plans only the
    missing leg -- and an entry trigger that needed both legs' frictions in one
    call never fired. The position reached HEDGED with an empty carry ledger:
    `free_cash` still held the money it had never spent, so equity read the
    whole notional as profit and `risk.update_equity` was given it.
    """
    at = world.minute(0)
    world.position.fill_models[SPOT].max_reference_deviation_bps = D("0")
    world.target("0.500", at)
    assert world.position.state is HedgeState.PARTIAL
    assert world.position.leg(PERP).quantity == D("0.500")
    assert world.position.leg(SPOT).is_flat
    world.assert_cash_is_right("one leg filled")

    at = world.minute(1)
    world.position.fill_models[SPOT].max_reference_deviation_bps = D("50")
    world.position.correct(at, equity=EQUITY)

    assert world.position.state is HedgeState.HEDGED
    assert world.position.imbalance() == D("0")
    state = world.position.ledger.state
    assert state.quantity == D("0.500"), "the hedge is held and the ledger says nothing"
    assert state.spot_entry is not None and state.perp_entry is not None
    assert state.entry_basis == state.perp_entry - state.spot_entry
    world.assert_cash_is_right("hedge completed by correction")

    # And the close that follows returns what the entry took.
    at = world.minute(2, spot="30100", perp="30125")
    world.target("0", at)
    assert world.position.state is HedgeState.FLAT
    assert world.position.ledger.state.quantity == D("0")
    world.assert_cash_is_right("close after a corrected entry")


def test_a_rebalance_up_books_the_increment_it_bought(world):
    """The rule re-sizes from equity every minute, so an increase is ordinary.

    Nothing booked one. The legs went to the larger quantity while the ledger
    kept the smaller, so `free_cash` never paid for the increment and
    `mark_to_market` counted inventory the campaign had been given for nothing:
    equity above capital on a minute with no price move at all.
    """
    at = world.minute(0)
    world.target("0.500", at)
    world.assert_cash_is_right("entry")

    at = world.minute(1)
    world.target("0.800", at)

    state = world.position.ledger.state
    assert world.position.leg(SPOT).quantity == D("0.800")
    assert state.quantity == D("0.800"), "the ledger did not follow the legs up"
    world.assert_cash_is_right("increased")

    # The increment is the increase's own fill, not a re-priced whole position.
    spot, perp = world.position.leg(SPOT), world.position.leg(PERP)
    mark = world.position.mark_to_market(at)
    assert mark.equity < CAPITAL, "an increase with no price move cannot create equity"
    # Section 6.6's equity line, recomputed here from the legs rather than read
    # from `CarryMark`: free cash, the spot inventory at the close, the margin
    # posted for what the PERPETUAL leg holds, and that same leg's unrealised.
    assert mark.equity == (
        state.free_cash
        + spot.quantity * at.spot_close
        + perp.quantity * perp.entry_price
        + perp.quantity * (perp.entry_price - at.perp_close)
    )

    at = world.minute(2)
    world.target("0", at)
    world.assert_cash_is_right("closed after an increase")
    assert world.position.ledger.state.quantity == D("0")


def test_the_entry_basis_follows_the_vwap_through_an_increase(world):
    """An increase moves both legs' VWAPs, so the entry basis has to move with them.

    Section 6.5's identity is asserted against the legs' marked PnL every
    minute. An entry basis left at the first fill's would disagree with the legs
    by the whole size of the increment.
    """
    world.target("0.400", world.minute(0))
    world.target("0.900", world.minute(1, spot="30500", perp="30560"))

    spot, perp = world.position.leg(SPOT), world.position.leg(PERP)
    state = world.position.ledger.state
    assert state.spot_entry == spot.entry_price
    assert state.perp_entry == perp.entry_price
    assert state.entry_basis == perp.entry_price - spot.entry_price

    at = world.minute(2, spot="30700", perp="30740")
    residual = world.position.ledger.check_identity(
        spot_close=at.spot_close, perp_close=at.perp_close
    )
    assert residual == D("0")
    assert world.position.ledger.disputed is None


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------


def test_a_partial_reduction_releases_margin_pro_rata(world):
    world.target("0.500", world.minute(0))
    world.assert_cash_is_right("entry")
    margin_at_entry = world.position.ledger.state.perp_margin

    world.target("0.200", world.minute(1, spot="30100", perp="30125"))

    state = world.position.ledger.state
    assert state.quantity == D("0.200")
    assert state.perp_margin == margin_at_entry * D("0.200") / D("0.500")
    world.assert_cash_is_right("partial reduction")


def test_many_partial_reductions_then_a_close_leave_no_residue(world):
    """Section 6.6's identity after every step, and exactly zero at the end.

    The margin release used to be a division by the surviving quantity, which
    leaves a residue whenever the division is inexact. Levels do not divide.
    """
    world.target("0.700", world.minute(0))
    for index, quantity in enumerate(("0.510", "0.330", "0.170", "0.030"), start=1):
        world.target(quantity, world.minute(index, spot="30000", perp="30030"))
        world.assert_cash_is_right(f"after reducing to {quantity}")
        assert world.position.ledger.state.quantity == D(quantity)

    world.target("0", world.minute(9))
    state = world.position.ledger.state
    assert state.quantity == D("0")
    assert state.perp_margin == D("0"), "a residue outlived the position"
    assert state.spot_principal == D("0")
    assert state.spot_entry is None and state.entry_basis is None
    world.assert_cash_is_right("final close")


def test_partial_reductions_reach_the_same_cash_as_one_close(tmp_path):
    """Invariant 3, as a two-sided control rather than an assertion about one run.

    Both worlds see identical prices, so the only difference between them is how
    many round trips the exit took. The frictions of those extra trips are real
    and are the whole of the difference; everything else must agree exactly.
    """
    stepwise = make_world(tmp_path / "stepwise")
    one_shot = make_world(tmp_path / "one_shot")

    for world, path in ((stepwise, ("0.400", "0.250", "0.100", "0")), (one_shot, ("0",))):
        world.target("0.500", world.minute(0))
        for index, quantity in enumerate(path, start=1):
            world.target(quantity, world.minute(index))
        world.assert_cash_is_right("end of lifecycle")

    a, b = stepwise.position.ledger.state, one_shot.position.ledger.state
    assert a.quantity == b.quantity == D("0")
    assert a.perp_margin == b.perp_margin == D("0")
    # Same prices throughout, so the extra round trips cost nothing beyond their
    # own frictions -- and at this fixture's prices they cost nothing at all,
    # because every fill is at the same touch.
    assert a.realised == b.realised
    assert a.free_cash - a.slippage * 0 == b.free_cash - b.slippage * 0
    assert (a.free_cash + a.fees + a.slippage) == (b.free_cash + b.fees + b.slippage)


def test_reconciling_twice_with_no_new_economics_books_nothing(world):
    """Invariant 4. The second call must compute a difference of zero, not repeat one."""
    world.target("0.500", world.minute(0))
    at = world.minute(1, spot="30100", perp="30125")
    world.target("0", at)

    state = world.position.ledger.state
    before = (state.free_cash, state.fees, state.slippage, state.realised, state.quantity)
    world.position._reconcile_ledger()
    world.position._reconcile_ledger({SPOT: (D("0"), D("0")), PERP: (D("0"), D("0"))})
    after = (state.free_cash, state.fees, state.slippage, state.realised, state.quantity)
    assert before == after
    world.assert_cash_is_right("after two idle reconciliations")


# ---------------------------------------------------------------------------
# Asymmetry
# ---------------------------------------------------------------------------


def test_an_asymmetric_close_books_each_leg_and_disputes_the_pair(world):
    """One leg closed and one did not, so neither is guessed at.

    The closed leg really did return its margin and really did pay its exit fee
    and slippage; the leg still holding really does still hold its principal.
    Refusing to book any of it discarded that cycle's slippage, which no
    executor accumulates and nothing can recover afterwards.
    """
    world.target("0.500", world.minute(0))
    spot_principal = world.position.ledger.state.spot_principal
    slippage_before = world.recorder.slippage

    at = world.minute(1, spot="30100", perp="30125")
    world.position.fill_models[SPOT].max_reference_deviation_bps = D("0")
    world.position.emergency_reduce(FlattenCause.RISK_HALT, at)

    state = world.position.ledger.state
    assert world.position.leg(PERP).is_flat and not world.position.leg(SPOT).is_flat
    assert state.perp_margin == D("0"), "the perpetual closed and kept its margin"
    assert state.spot_principal == spot_principal, "the spot leg is still held"
    assert world.recorder.slippage > slippage_before, "the exit was free; nothing is proved"
    world.assert_cash_is_right("asymmetric close")
    assert "asymmetric_close" in (state.disputed or "")
    assert world.position.is_disputed

    at = world.minute(2, spot="30200", perp="30215")
    world.position.fill_models[SPOT].max_reference_deviation_bps = D("50")
    world.position.emergency_reduce(FlattenCause.RISK_HALT, at)

    assert world.position.leg(SPOT).is_flat and world.position.leg(PERP).is_flat
    assert state.spot_principal == D("0") and state.perp_margin == D("0")
    world.assert_cash_is_right("second call completes the close")
    # Both legs' totals arrived, however many calls it took.
    legs = world.position
    assert state.realised == legs.spot.ledger.realised_pnl + legs.perp.ledger.realised_pnl
    assert state.fees == legs.spot.ledger.trading_fees + legs.perp.ledger.trading_fees
    assert state.disputed is not None, "a completed close is not a cleared dispute"


# ---------------------------------------------------------------------------
# Funding
# ---------------------------------------------------------------------------


def _settle(world: World, at: Minute, rate: str, index: int) -> D:
    settlement = PerpSettlement(
        symbol=world.position.config.perp_symbol,
        rate=D(rate),
        mark_price=at.perp_close,
        settlement_id=str(1_000 + index),
        instant_ns=at.minute_ns,
    )
    return world.position.settle_funding(
        settlement,
        open_instant_ns=world.position.ledger.state.open_instant_ns or 0,
        now_ns=at.minute_ns,
    )


@pytest.mark.parametrize(
    "rate, direction",
    [("0.0001", "received"), ("-0.0001", "paid")],
)
def test_funding_before_a_close_survives_the_close(world, rate, direction):
    """A10's sign, and then a close that must not reclassify or lose the flow.

    The perpetual leg is SHORT, so `-sign(SHORT) x notional x rate` means a
    POSITIVE rate is received. Both directions are witnessed because a close
    that netted funding into realised PnL would look right in one of them.
    """
    world.target("0.500", world.minute(0))
    at = world.minute(2)
    flow = _settle(world, at, rate, index=1)
    assert flow != D("0"), "the settlement fell outside the window; nothing is proved"
    if direction == "received":
        assert flow > D("0")
    else:
        assert flow < D("0")

    state = world.position.ledger.state
    received, paid = state.funding_received, state.funding_paid
    world.assert_cash_is_right("after funding")

    world.target("0", world.minute(3, spot="30100", perp="30125"))

    assert state.funding_received == received, "the close moved funding_received"
    assert state.funding_paid == paid, "the close moved funding_paid"
    assert state.net_funding == flow
    world.assert_cash_is_right("close after funding")


def test_funding_then_a_partial_then_the_final_close(world):
    """Funding, a reduction that leaves the position open, and the close after it."""
    world.target("0.600", world.minute(0))
    flow = _settle(world, world.minute(2), "0.0001", index=2)
    assert flow > D("0")

    world.target("0.250", world.minute(3, spot="30050", perp="30085"))
    state = world.position.ledger.state
    assert state.net_funding == flow, "a reduction re-classified booked funding"
    world.assert_cash_is_right("partial after funding")

    world.target("0", world.minute(4, spot="30090", perp="30120"))
    assert state.net_funding == flow
    assert state.quantity == D("0")
    world.assert_cash_is_right("final close after funding")


def test_a_settlement_in_the_flat_gap_cannot_reach_the_next_position(world):
    """`open_instant` is cleared on flat, so the window cannot span two positions."""
    world.target("0.500", world.minute(0))
    world.target("0", world.minute(1))
    assert world.position.ledger.state.open_instant_ns is None

    gap = world.minute(2)
    world.target("0.300", world.minute(3, spot="30400", perp="30440"))
    opened = world.position.ledger.state.open_instant_ns
    assert opened == world.minute(3, spot="30400", perp="30440").minute_ns + MINUTE_NS

    flow = world.position.settle_funding(
        PerpSettlement(
            symbol=world.position.config.perp_symbol,
            rate=D("0.0001"),
            mark_price=gap.perp_close,
            settlement_id="gap",
            instant_ns=gap.minute_ns,
        ),
        open_instant_ns=opened,
        now_ns=gap.minute_ns + 10 * MINUTE_NS,
    )
    assert flow == D("0"), "a settlement from while nothing was held was charged"


# ---------------------------------------------------------------------------
# Reopen, and the restart windows
# ---------------------------------------------------------------------------


def test_reopening_after_flat_starts_a_fresh_lifecycle(world):
    world.target("0.500", world.minute(0))
    first_entry = world.position.ledger.state.spot_entry
    world.target("0", world.minute(1))

    state = world.position.ledger.state
    assert state.open_instant_ns is None
    assert state.spot_entry is None and state.perp_entry is None
    assert state.entry_basis is None and state.current_basis is None

    at = world.minute(2, spot="30500", perp="30545")
    world.target("0.300", at)

    assert state.quantity == D("0.300")
    assert state.spot_entry is not None and state.spot_entry != first_entry
    assert state.perp_margin == D("0.300") * world.position.leg(PERP).entry_price
    assert state.open_instant_ns == at.minute_ns + MINUTE_NS
    assert state.entry_basis == state.perp_entry - state.spot_entry
    world.assert_cash_is_right("reopened")


def test_a_crash_between_the_stores_and_the_ledger_disputes_on_restart(tmp_path):
    """The window `save` order creates, and the one a `held > 0` guard could not see.

    The executors persist inside `execute_target`; this ledger persists later.
    So a crash between them leaves the legs FLAT and the ledger holding the
    whole position -- and `held` is then zero, which is exactly when the old
    comparison declined to run. `reconstruct` returned READY and the campaign
    carried on short by a principal and a margin it had already been paid back.
    """
    world = make_world(tmp_path)
    world.target("0.500", world.minute(0))
    world.position.ledger.save(now_ns=START_NS)
    stale_bytes = world.position.ledger.path.read_bytes()

    world.target("0", world.minute(1, spot="30100", perp="30125"))
    assert world.position.leg(SPOT).is_flat and world.position.leg(PERP).is_flat
    # The crash: the stores are on disk, the ledger's own save never happened.
    world.position.ledger.path.write_bytes(stale_bytes)

    restarted = make_world(tmp_path)
    restarted.position.ledger = CarryLedger.open(
        tmp_path / "carry_ledger.json", capital=CAPITAL
    )
    assert restarted.position.ledger.state.quantity == D("0.500")

    outcome = restarted.position.reconstruct()
    assert outcome.state is HedgeState.DISPUTED
    assert "ledger_store_mismatch" in (restarted.position.ledger.disputed or "")
    assert restarted.position.risk.state.halted


def test_a_crash_during_an_increase_disputes_on_the_principal(tmp_path):
    """The quantities can agree while the principals do not.

    A torn write during an increase moves both legs and the hedged quantity by
    the same step, so a comparison of quantities alone cannot see it. The
    principals are what `free_cash` moved by, so they are what is compared.
    """
    world = make_world(tmp_path)
    world.target("0.500", world.minute(0))
    world.position.ledger.save(now_ns=START_NS)
    ledger_path = world.position.ledger.path

    # A ledger that reached the same quantity at a different cost: the increase
    # that raised the principal never reached the file.
    import json

    payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    payload["spot_principal"] = str(D(payload["spot_principal"]) - D("100"))
    ledger_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", "utf-8")

    restarted = make_world(tmp_path)
    restarted.position.ledger = CarryLedger.open(ledger_path, capital=CAPITAL)
    outcome = restarted.position.reconstruct()
    assert outcome.state is HedgeState.DISPUTED
    assert "ledger_store_mismatch" in (restarted.position.ledger.disputed or "")


def test_a_ledger_written_without_the_principal_field_restates_it(tmp_path):
    """The additive field's fallback is a restatement of the old build, not a guess."""
    world = make_world(tmp_path)
    world.target("0.500", world.minute(0))
    world.position.ledger.save(now_ns=START_NS)
    path = world.position.ledger.path

    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = D(payload.pop("spot_principal"))
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", "utf-8")

    reloaded = CarryLedger.open(path, capital=CAPITAL)
    assert reloaded.state.spot_principal == expected
    assert reloaded.state.spot_principal == reloaded.state.quantity * reloaded.state.spot_entry


def test_an_executor_ledger_that_went_backwards_disputes_rather_than_crediting(world):
    """Fees never fall, so a total below what was booked is a reset, not a rebate."""
    world.target("0.500", world.minute(0))
    booked = world.position.ledger.state.spot.fees
    assert booked > D("0")
    cash_before = world.position.ledger.state.free_cash

    # What `FuturesStore.adopt_after_unreadable` does: a fresh state, and with
    # it a ledger whose accumulators start again from zero.
    world.position.spot.store.state.ledger.trading_fees = D("0")
    world.position._reconcile_ledger()

    assert "spot_ledger_regressed" in (world.position.ledger.disputed or "")
    assert (
        world.position.ledger.state.free_cash == cash_before
    ), "the campaign credited itself its own fee history"


def test_a_leg_that_raises_after_another_filled_still_books_the_filled_one(world):
    """`execute_target` raises per leg, and the second leg can raise after the first filled.

    Fees and realised PnL would be recovered by the next level comparison. This
    cycle's slippage would not: nothing accumulates it, so a cycle that measures
    it and then throws it away has lost it for good.
    """
    world.target("0.500", world.minute(0))
    at = world.minute(1, spot="30100", perp="30125")

    from chimera.futures.executor import FuturesError

    original = world.position.spot.execute_target

    def explode(*args, **kwargs):
        raise FuturesError("the spot leg's store went away mid-cycle")

    world.position.spot.execute_target = explode
    with pytest.raises(FuturesError):
        world.target("0", at)
    world.position.spot.execute_target = original

    # The perpetual filled and was booked; the spot leg never moved.
    assert world.position.leg(PERP).is_flat
    assert not world.position.leg(SPOT).is_flat
    world.assert_cash_is_right("after a mid-cycle raise")


# ---------------------------------------------------------------------------
# Property-style: any lifecycle at all
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [1, 2, 3, 5, 8, 13, 21])
def test_any_sequence_of_targets_keeps_the_cash_identity(tmp_path, seed):
    """Random targets, random prices, random refusals -- one identity throughout.

    The point is not any particular sequence. It is that no ordering of
    increases, decreases, closes, reopens and refused legs can move `free_cash`
    away from what the two executors say it should be.
    """
    world = make_world(tmp_path)
    rng = random.Random(seed)
    quantities = ["0", "0.100", "0.250", "0.400", "0.600", "0.750"]

    for index in range(1, 26):
        spot = 30000 + rng.randrange(-400, 400)
        at = world.minute(index, spot=str(spot), perp=str(spot + rng.randrange(5, 60)))
        if rng.random() < 0.25:
            leg = rng.choice((SPOT, PERP))
            world.position.fill_models[leg].max_reference_deviation_bps = D("0")
        world.target(rng.choice(quantities), at)
        for leg in (SPOT, PERP):
            world.position.fill_models[leg].max_reference_deviation_bps = D("50")

        world.assert_cash_is_right(f"seed {seed}, step {index}", tolerance=DECIMAL_ARTIFACT)
        state = world.position.ledger.state
        assert state.quantity == min(
            world.position.leg(SPOT).quantity, world.position.leg(PERP).quantity
        )
        assert state.spot_principal >= D("0") and state.perp_margin >= D("0")
        if world.position.state is HedgeState.HEDGED:
            assert world.position.imbalance() == D("0")


def test_a_reduction_can_never_return_more_than_was_booked(world):
    """Invariant 2, stated where it is enforced: a level cannot fall below zero."""
    world.target("0.500", world.minute(0))
    spot = world.position.leg(SPOT)
    with pytest.raises(LedgerError):
        world.position.ledger.book_position(
            spot_quantity=D("-0.100"),
            spot_entry=spot.entry_price,
            perp_quantity=D("0"),
            perp_entry=D("0"),
        )
    with pytest.raises(LedgerError):
        world.position.ledger.book_position(
            spot_quantity=D("0.100"),
            spot_entry=D("0"),
            perp_quantity=D("0.100"),
            perp_entry=D("30000"),
        )


def test_a_ledger_whose_quantity_disagrees_disputes_even_with_zero_principals(tmp_path):
    """The quantity comparison, on the one state the principal comparison cannot see.

    `book_position` writes the quantity and the two principals together, so a
    ledger this build wrote cannot hold a quantity beside zero principals. A
    hand-repaired one can -- and hand repair is what the runbook tells an
    operator to do for the disputes that have no command -- so the comparison
    that catches it has to be its own, not a side effect of the principal check.
    """
    import json

    world = make_world(tmp_path)
    world.target("0.500", world.minute(0))
    world.target("0", world.minute(1))
    world.position.ledger.save(now_ns=START_NS)
    path = world.position.ledger.path

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["spot_principal"] == "0" or D(payload["spot_principal"]) == D("0")
    payload["quantity"] = "0.500"  # a quantity nothing paid for
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", "utf-8")

    restarted = make_world(tmp_path)
    restarted.position.ledger = CarryLedger.open(path, capital=CAPITAL)
    outcome = restarted.position.reconstruct()
    assert outcome.state is HedgeState.DISPUTED
    assert "ledger_store_mismatch" in (restarted.position.ledger.disputed or "")


def test_an_imbalanced_position_marks_each_leg_on_what_that_leg_holds(world):
    """Section 6.6's equity line, on the state where the two quantities differ.

    `perp_margin` is the margin posted for what the PERPETUAL leg holds, so the
    unrealised term beside it has to be measured on that same quantity. Marking
    it on `min(spot, perp)` reported a margin the position held and a PnL it did
    not, and the two disagreed by exactly the imbalance -- in the one state an
    equity reading most needs to be right, because a one-legged position is the
    state Aegis is being asked to halt on.
    """
    at = world.minute(0)
    world.position.fill_models[SPOT].max_reference_deviation_bps = D("0")
    world.target("0.500", at)
    assert world.position.state is HedgeState.PARTIAL
    perp = world.position.leg(PERP)
    assert perp.quantity == D("0.500") and world.position.leg(SPOT).is_flat

    later = world.minute(1, spot="30400", perp="30450")
    mark = world.position.mark_to_market(later)
    state = world.position.ledger.state

    # The oracle is the legs, recomputed here: cash, no spot inventory, the
    # margin posted for 0.500 of perpetual, and 0.500 of perpetual unrealised.
    assert mark.equity == (
        state.free_cash
        + D("0")
        + perp.quantity * perp.entry_price
        + perp.quantity * (perp.entry_price - later.perp_close)
    )
    assert mark.perp_pnl == perp.quantity * (perp.entry_price - later.perp_close)
    assert mark.perp_pnl != D("0"), "the price did not move; nothing is proved"
    # The reported hedged quantity is still the hedged one, and it is zero here.
    assert mark.quantity == D("0")


def test_a_spot_heavy_imbalance_marks_the_spot_leg_on_what_it_holds(world):
    """The other half of the per-leg mark, and the direction the fixtures skip.

    Every imbalance an ordinary campaign reaches is perpetual-heavy, because
    `apply` sends the perpetual first and stops at the first leg that does not
    fill. So a witness staged only that way asserts the spot term as the literal
    zero, and a mark that valued the spot inventory at `min(spot, perp)` instead
    of at what the spot leg holds would pass it. Here the spot leg is the one
    holding more, which is the state a refused CLOSE leaves.
    """
    world.target("0.500", world.minute(0))
    at = world.minute(1, spot="30300", perp="30360")
    world.position.fill_models[SPOT].max_reference_deviation_bps = D("0")
    world.position.emergency_reduce(FlattenCause.RISK_HALT, at)

    spot, perp = world.position.leg(SPOT), world.position.leg(PERP)
    assert spot.quantity > perp.quantity, "this is not the spot-heavy state"
    assert perp.is_flat

    later = world.minute(2, spot="30500", perp="30545")
    mark = world.position.mark_to_market(later)
    state = world.position.ledger.state

    assert mark.spot_pnl == spot.quantity * (later.spot_close - spot.entry_price)
    assert mark.spot_pnl != D("0"), "the price did not move; nothing is proved"
    assert mark.equity == (
        state.free_cash
        + spot.quantity * later.spot_close
        + perp.quantity * perp.entry_price
        + perp.quantity * (perp.entry_price - later.perp_close)
    )
    # And the whole spot inventory is valued, not the hedged quantity, which is
    # zero here: an equity that marked `min(spot, perp)` would lose all of it.
    assert mark.equity - state.free_cash == spot.quantity * later.spot_close


def test_a_partial_reduction_that_leaves_the_legs_unequal_also_disputes(world):
    """`asymmetric_close` is about the two legs disagreeing, not about a zero.

    Both other witnesses drive a leg to exactly flat. Section 6.8 compares the
    two legs' quantities; a reduction whose second leg is refused leaves them
    unequal at two non-zero quantities, and the ledger has the same single
    quantity problem there.
    """
    world.target("0.500", world.minute(0))
    at = world.minute(1, spot="30100", perp="30150")
    world.position.fill_models[SPOT].max_reference_deviation_bps = D("0")
    world.target("0.300", at)

    spot, perp = world.position.leg(SPOT), world.position.leg(PERP)
    assert spot.quantity == D("0.500") and perp.quantity == D(
        "0.300"
    ), f"the staged state is spot={spot.quantity} perp={perp.quantity}"
    assert "asymmetric_close" in (world.position.ledger.state.disputed or "")
    world.assert_cash_is_right("partial asymmetric reduction")


def test_the_perpetual_leg_regressing_disputes_too(world):
    """The guard is per leg, and only the spot half was witnessed."""
    world.target("0.500", world.minute(0))
    cash_before = world.position.ledger.state.free_cash
    assert world.position.ledger.state.perp.fees > D("0")

    world.position.perp.store.state.ledger.trading_fees = D("0")
    world.position._reconcile_ledger()

    assert "perp_ledger_regressed" in (world.position.ledger.disputed or "")
    assert world.position.ledger.state.free_cash == cash_before


def test_reconstruct_disputes_on_a_perp_margin_that_disagrees(tmp_path):
    """The principal comparison is two-armed; only the spot arm was witnessed."""
    import json

    world = make_world(tmp_path)
    world.target("0.500", world.minute(0))
    world.position.ledger.save(now_ns=START_NS)
    path = world.position.ledger.path

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["perp_margin"] = str(D(payload["perp_margin"]) - D("250"))
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", "utf-8")

    restarted = make_world(tmp_path)
    restarted.position.ledger = CarryLedger.open(path, capital=CAPITAL)
    outcome = restarted.position.reconstruct()
    assert outcome.state is HedgeState.DISPUTED
    assert "ledger_store_mismatch" in (restarted.position.ledger.disputed or "")
    assert "perp" in (restarted.position.ledger.disputed or "")


def test_slippage_is_recorded_and_never_spent(world):
    """The correction that made the oracle independent, asserted directly.

    `RecordedQuoteFillModel` crosses to the recorded touch and applies the
    configured slippage on top, so the executor's VWAP IS the slipped price and
    the inventory, the margin and the realised PnL are all valued at it.
    Deducting the measured slippage from cash as well charged the crossing
    twice, and the error grew with turnover and reached `risk.update_equity`.
    """
    world.target("0.500", world.minute(0))
    state = world.position.ledger.state
    spot, perp = world.position.leg(SPOT), world.position.leg(PERP)

    assert state.slippage > D("0"), "no spread was crossed; nothing is proved"
    # The slippage the ledger recorded IS the gap between the fills and the
    # decision closes -- i.e. money already inside the prices below.
    assert state.slippage == (
        (spot.entry_price - D("30000")) * spot.quantity
        + (D("30030") - perp.entry_price) * perp.quantity
    )
    fees = world.position.spot.ledger.trading_fees + world.position.perp.ledger.trading_fees
    committed = spot.quantity * spot.entry_price + perp.quantity * perp.entry_price + fees
    assert state.free_cash == CAPITAL - committed, "cash moved by something not transacted"

    world.target("0", world.minute(1, spot="30100", perp="30140"))
    fees = world.position.spot.ledger.trading_fees + world.position.perp.ledger.trading_fees
    realised = (
        world.position.spot.ledger.realised_pnl + world.position.perp.ledger.realised_pnl
    )
    assert state.free_cash == CAPITAL - fees + realised
    assert state.slippage > D("0"), "the round trip recorded no slippage at all"
