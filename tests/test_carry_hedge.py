"""``HedgedPosition``: entry, correction, dispute, restart, and the imbalance.

Synthetic worlds only. No market data is read, and no economic result is
computed: every number here was chosen by the test author.

The invariant asserted everywhere: **whenever the state is HEDGED, the two legs
hold the same BTC quantity.**
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from decimal import Decimal as D

import pytest

from chimera.carry.factory import PERP_SYMBOL, SPOT_SYMBOL, build_hedged_position
from chimera.carry.hedge import HedgedPosition, HedgeState, HedgeTarget, PERP, SPOT
from chimera.futures.executor import FlattenCause
from chimera.futures.fills import RecordedQuoteFillModel, TopOfBook
from chimera.risk import RiskEngine, RiskLimits

CAPITAL = D("1000000")
EQUITY = 1_000_000.0
START_NS = 1_700_000_000_000_000_000
MINUTE_NS = 60_000_000_000


@dataclass(frozen=True)
class Minute:
    """The minimum a carry position needs to know. Satisfies `CarryMarketState`."""

    minute_ns: int
    complete: bool
    spot_close: D
    perp_close: D
    mark: D


def minute(index: int = 0, *, spot: str = "30000", perp: str = "30030", complete: bool = True):
    return Minute(
        minute_ns=START_NS + index * MINUTE_NS,
        complete=complete,
        spot_close=D(spot),
        perp_close=D(perp),
        mark=D(perp),
    )


class BothLegModels:
    """Every leg's fill model, driven together.

    The tests in this file are fill-mechanics tests: they install ONE book and
    read the spread, the slippage and the resulting entry basis straight off it,
    so giving both legs that same book is the point rather than an oversight.
    ``-13.00`` below is the cost of crossing one spread twice, not a basis.

    Production does NOT share a model. ``build_hedged_position`` builds one per
    leg and ``HedgedPosition.install_quote`` gives each its own side of the
    market; the witness that the spot leg fills inside the SPOT book lives in
    ``tests/test_demo_runner.py``, where there are two books to tell apart.
    """

    def __init__(self, models):
        self._models = tuple(models)
        assert self._models, "a position with no fill models cannot be quoted"

    def set_quote(self, quote, now_ns):
        for model in self._models:
            model.set_quote(quote, now_ns)

    @property
    def now_ns(self):
        return self._models[0].now_ns

    @now_ns.setter
    def now_ns(self, value):
        for model in self._models:
            model.now_ns = value


def book(model: BothLegModels, at: Minute, *, bid="30000", ask="30001", size="50"):
    model.set_quote(
        TopOfBook(
            instant_ns=at.minute_ns,
            bid=D(bid),
            bid_qty=D(size),
            ask=D(ask),
            ask_qty=D(size),
        ),
        at.minute_ns,
    )


@pytest.fixture
def position(tmp_path):
    risk = RiskEngine(RiskLimits(max_position_pct=1.0, risk_per_trade_pct=0.5))
    risk.update_equity(EQUITY)
    model = RecordedQuoteFillModel()
    hedged = build_hedged_position(
        risk=risk, capital=CAPITAL, state_dir=tmp_path, fill_model=model
    )
    hedged.spot.recover({})
    hedged.perp.recover({})
    hedged.model = BothLegModels(hedged.fill_models.values())  # type: ignore[attr-defined]
    return hedged


def open_hedge(position: HedgedPosition, quantity: str = "0.500", at: Minute | None = None):
    bar = at or minute(0)
    book(position.model, bar)  # type: ignore[attr-defined]
    intents = position.plan(HedgeTarget(D(quantity)), bar)
    return position.apply(intents, bar, equity=EQUITY), bar


def assert_hedged_is_balanced(position: HedgedPosition) -> None:
    """The invariant, asserted wherever the state could have changed."""
    if position.state is HedgeState.HEDGED:
        assert position.imbalance() == D("0"), "HEDGED with a non-zero imbalance"


# ---------------------------------------------------------------------------
# Entry sequence
# ---------------------------------------------------------------------------


def test_the_entry_sends_the_perpetual_first_and_the_spot_second(position):
    bar = minute(0)
    book(position.model, bar)
    intents = position.plan(HedgeTarget(D("0.500")), bar)
    assert [i.leg for i in intents] == [PERP, SPOT]
    assert intents[0].symbol == PERP_SYMBOL
    assert intents[1].symbol == SPOT_SYMBOL


def test_both_legs_filling_reaches_hedged_with_zero_imbalance(position):
    outcome, _ = open_hedge(position)
    assert outcome.state is HedgeState.HEDGED
    assert position.state is HedgeState.HEDGED
    assert position.imbalance() == D("0.000")
    assert position.leg(SPOT).quantity == position.leg(PERP).quantity == D("0.500")
    assert_hedged_is_balanced(position)


def test_a_flat_target_plans_nothing_when_already_flat(position):
    bar = minute(0)
    book(position.model, bar)
    assert position.plan(HedgeTarget(D("0")), bar) == []
    assert position.state is HedgeState.FLAT


def test_an_incomplete_minute_plans_nothing(position):
    """Never trade on a minute the feed could not complete."""
    bar = minute(0, complete=False)
    book(position.model, bar)
    assert position.plan(HedgeTarget(D("0.500")), bar) == []


def test_a_negative_target_is_refused():
    with pytest.raises(Exception):
        HedgeTarget(D("-1"))


# ---------------------------------------------------------------------------
# Partial-leg failure and correction
# ---------------------------------------------------------------------------


def test_a_stale_book_leaves_the_position_flat_rather_than_one_sided(position):
    """The first leg is refused, so the second is never sent."""
    bar = minute(0)
    book(position.model, bar)
    # Age the clock far past the freshness bound without installing a new book.
    position.model.now_ns = bar.minute_ns + 10 * MINUTE_NS
    outcome = position.apply(position.plan(HedgeTarget(D("0.500")), bar), bar, equity=EQUITY)
    assert outcome.unfilled == (PERP,)
    assert position.state is HedgeState.FLAT
    assert position.imbalance() == D("0")


def test_a_second_leg_failure_leaves_PARTIAL_with_a_real_imbalance(position):
    """The perp fills, the spot is refused: exactly the case section 6.3 names."""
    bar = minute(0)
    book(position.model, bar)
    intents = position.plan(HedgeTarget(D("0.500")), bar)

    # Send the perpetual only, then refuse the spot by ageing the book.
    outcome = position.apply(intents[:1], bar, equity=EQUITY)
    assert outcome.state is HedgeState.PARTIAL
    assert position.leg(PERP).quantity == D("0.500")
    assert position.leg(SPOT).quantity == D("0")
    assert position.imbalance() == D("-0.500")


def test_correction_retries_and_then_flattens_with_the_right_cause(position):
    """After max_correction_minutes the filled leg is reduced, not left hanging."""
    bar = minute(0)
    book(position.model, bar)
    position.apply(position.plan(HedgeTarget(D("0.500")), bar)[:1], bar, equity=EQUITY)
    assert position.state is HedgeState.PARTIAL

    # Every retry happens against a stale book, so none of them fills.
    for index in range(1, position.config.max_correction_minutes + 1):
        later = minute(index)
        position.model.now_ns = later.minute_ns + 10 * MINUTE_NS
        position.correct(later, equity=EQUITY)

    final = minute(position.config.max_correction_minutes + 1)
    book(position.model, final)
    outcome = position.correct(final, equity=EQUITY)

    assert position.state is HedgeState.FLAT
    assert outcome.detail.startswith("hedge correction timed out")
    assert position.imbalance() == D("0")
    reasons = [r["reason"] for r in position.perp.store.state.flatten_reasons]
    assert FlattenCause.HEDGE_CORRECTION.value in reasons
    assert FlattenCause.DATA_LOSS.value not in reasons


def test_correction_is_a_no_op_when_the_position_is_not_partial(position):
    open_hedge(position)
    outcome = position.correct(minute(1), equity=EQUITY)
    assert outcome.detail == "not partial"
    assert position.state is HedgeState.HEDGED


# ---------------------------------------------------------------------------
# Emergency reduction
# ---------------------------------------------------------------------------


def test_emergency_reduce_flattens_the_perpetual_before_the_spot(position):
    open_hedge(position)
    bar = minute(1)
    book(position.model, bar)
    outcome = position.emergency_reduce(FlattenCause.RISK_HALT, bar)

    assert position.state is HedgeState.FLAT
    assert outcome.detail.endswith(FlattenCause.RISK_HALT.value)
    assert position.leg(SPOT).is_flat and position.leg(PERP).is_flat
    for store in (position.perp.store, position.spot.store):
        assert store.state.flatten_reasons[-1]["reason"] == FlattenCause.RISK_HALT.value


def test_a_reduction_is_still_planned_while_halted(position):
    """Reductions remain possible when Aegis has halted; increases do not."""
    open_hedge(position)
    position.risk.halt("synthetic")
    bar = minute(1)
    book(position.model, bar)

    assert position.plan(HedgeTarget(D("0.800")), bar) == []  # increase: refused
    assert position.plan(HedgeTarget(D("0.200")), bar) != []  # reduction: allowed
    assert position.plan(HedgeTarget(D("0")), bar) != []  # flatten: allowed


def test_no_increase_is_planned_while_disputed(position):
    open_hedge(position)
    position.dispute("synthetic dispute")
    bar = minute(1)
    book(position.model, bar)

    assert position.state is HedgeState.DISPUTED
    assert position.risk.state.halted
    assert position.plan(HedgeTarget(D("0.900")), bar) == []
    assert position.plan(HedgeTarget(D("0")), bar) != []


# ---------------------------------------------------------------------------
# Funding accrual window
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Settlement:
    instant_ns: int


@pytest.mark.parametrize(
    "settlement_ns,charged",
    [
        (0, False),  # exactly the open instant: NOT charged
        (1, True),  # just after the open
        (500, True),  # inside
        (1000, True),  # exactly now: charged
        (1001, False),  # after now
    ],
)
def test_both_funding_tie_boundaries_are_pinned(position, monkeypatch, settlement_ns, charged):
    """The window is ``open_instant < settlement <= now``, both ends pinned."""
    booked: list[int] = []
    monkeypatch.setattr(position.perp, "settle_funding", lambda event: D("1"))
    monkeypatch.setattr(
        position.ledger, "book_funding", lambda instant, flow: booked.append(instant) or flow
    )
    position.settle_funding(Settlement(settlement_ns), open_instant_ns=0, now_ns=1000)
    assert bool(booked) is charged


def test_a_settlement_is_never_booked_twice(position):
    open_hedge(position)
    flows = []
    for _ in range(2):
        flows.append(position.ledger.book_funding(42, D("5")))
    assert flows == [D("5"), D("0")]
    assert position.ledger.state.funding_received == D("5")


# ---------------------------------------------------------------------------
# Marking and the identity
# ---------------------------------------------------------------------------


def test_marking_a_hedged_position_reports_the_basis_and_updates_worst_equity(position):
    open_hedge(position)
    mark = position.mark_to_market(minute(1, spot="30100", perp="30130"))
    assert mark.basis == D("30")
    assert mark.quantity == D("0.500")
    assert position.ledger.state.worst_equity == mark.equity
    assert position.state is not HedgeState.DISPUTED


def test_a_broken_identity_disputes_and_halts(position):
    open_hedge(position)
    position.ledger.state.perp_entry = D("40000")  # corrupt one leg's story
    position.mark_to_market(minute(1))
    assert position.state is HedgeState.DISPUTED
    assert position.risk.state.halted


# ---------------------------------------------------------------------------
# Restart
# ---------------------------------------------------------------------------


def test_reconstruct_recovers_a_hedged_position(position):
    open_hedge(position)
    position.ledger.state.quantity = D("0.500")
    position.ledger.save()

    outcome = position.reconstruct()
    assert outcome.state is HedgeState.HEDGED
    assert position.imbalance() == D("0.000")
    assert_hedged_is_balanced(position)


def test_reconstruct_disputes_when_the_ledger_disagrees_with_the_stores(position):
    open_hedge(position)
    position.ledger.state.quantity = D("0.250")  # the ledger says something else
    outcome = position.reconstruct()
    assert position.state is HedgeState.DISPUTED
    assert "ledger_store_mismatch" in (position.ledger.disputed or "")
    assert outcome.detail == "ledger disagrees with the stores"


def test_reconstruct_disputes_on_a_stale_leg(position):
    open_hedge(position)
    position.ledger.state.quantity = D("0.500")
    position.ledger.note_leg_mark(SPOT, START_NS)
    position.ledger.note_leg_mark(PERP, START_NS - 10 * MINUTE_NS)
    outcome = position.reconstruct()
    assert position.state is HedgeState.DISPUTED
    assert "stale_leg" in (position.ledger.disputed or "")
    assert "stale leg" in outcome.detail


def test_reconstruct_disputes_on_a_corrupt_ledger(tmp_path):
    risk = RiskEngine(RiskLimits(max_position_pct=1.0, risk_per_trade_pct=0.5))
    risk.update_equity(EQUITY)
    (tmp_path / "carry_ledger.json").write_text("{ not json", encoding="utf-8")
    hedged = build_hedged_position(risk=risk, capital=CAPITAL, state_dir=tmp_path)
    hedged.spot.recover({})
    hedged.perp.recover({})

    outcome = hedged.reconstruct()
    assert hedged.state is HedgeState.DISPUTED
    assert (hedged.ledger.disputed or "").startswith("ledger_unreadable")
    assert outcome.state is HedgeState.DISPUTED


def test_a_dispute_survives_and_requires_an_operator_note(position):
    open_hedge(position)
    position.dispute("synthetic")
    with pytest.raises(Exception, match="requires an operator note"):
        position.ledger.resolve("")
    position.ledger.resolve("inspected both legs; the perp store was behind by one fill")
    assert position.ledger.disputed is None
    assert position.ledger.state.resolutions[-1]["note"].startswith("inspected both legs")


# ---------------------------------------------------------------------------
# Property / invariant test with a fixed seed. The seed is part of the test.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [3, 17, 64, 128, 511])
def test_hedged_always_means_zero_imbalance_under_random_fill_and_fault_sequences(
    tmp_path, seed
):
    """Random targets, random stale-book faults: HEDGED never lies about balance."""
    rng = random.Random(seed)
    risk = RiskEngine(RiskLimits(max_position_pct=1.0, risk_per_trade_pct=0.5))
    risk.update_equity(EQUITY)
    model = RecordedQuoteFillModel()
    hedged = build_hedged_position(
        risk=risk, capital=CAPITAL, state_dir=tmp_path / f"s{seed}", fill_model=model
    )
    hedged.spot.recover({})
    hedged.perp.recover({})
    model = BothLegModels(hedged.fill_models.values())

    seen = set()
    for index in range(40):
        bar = minute(index, spot=str(30000 + rng.randrange(-200, 200)))
        drop_quote = rng.random() < 0.25
        if drop_quote:
            model.now_ns = bar.minute_ns + 10 * MINUTE_NS  # age it out, install nothing
        else:
            book(model, bar)

        if hedged.state is HedgeState.PARTIAL:
            hedged.correct(bar, equity=EQUITY)
        else:
            quantity = D(rng.choice(["0", "0.100", "0.250", "0.500"]))
            hedged.apply(hedged.plan(HedgeTarget(quantity), bar), bar, equity=EQUITY)

        seen.add(hedged.state)
        # The invariant, on every single step.
        assert_hedged_is_balanced(hedged)
        if hedged.state is HedgeState.FLAT:
            assert hedged.leg(SPOT).is_flat and hedged.leg(PERP).is_flat

    # The run has to have actually exercised more than one state to mean anything.
    assert len(seen) >= 2, f"seed {seed} only ever reached {seen}"


# ---------------------------------------------------------------------------
# The acceptance example, hand-calculated independently and pinned here.
# ---------------------------------------------------------------------------


def test_the_hand_traced_long_spot_short_perp_example(position):
    """Every number below was computed by hand from the adopted constants.

    They are the oracle: the implementation is asserted against them, not the
    other way round. Derivation, with the venue's committed values --
    spot fee 0.001, perp fee 0.0005, slippage 2 bps, spot tick 0.01, perp tick
    0.10, common step 0.001:

        book            bid 30000.00 / ask 30001.00
        spot LONG fills at the ask crossed by 2 bps:
                        30001 x 1.0002 = 30007.0002 -> 30007.00
        perp SHORT fills at the bid crossed by 2 bps:
                        30000 x 0.9998 = 29994.00   -> 29994.00
        Q               0.500 BTC
        spot notional   0.500 x 30007.00 = 15003.50
        perp notional   0.500 x 29994.00 = 14997.00   (= margin at 1x)
        spot fee        15003.50 x 0.001   = 15.0035
        perp fee        14997.00 x 0.0005  =  7.4985
        entry basis     29994.00 - 30007.00 = -13.00

        one settlement at mark 30010.00, rate +0.0001:
        notional        0.500 x 30010.00 = 15005.00
        flow            -sign(SHORT) x 15005.00 x 0.0001 = +1.5005  (SHORT receives)

        mark at spot 31000.00 / perp 31005.00:
        basis now       31005.00 - 31000.00 = 5.00
        spot PnL        0.500 x (31000.00 - 30007.00) =  496.50
        perp PnL        0.500 x (29994.00 - 31005.00) = -505.50
        legs total                                     =  -9.00
        Q x (basis_in - basis_now) = 0.500 x (-13.00 - 5.00) = -9.00
        identity residual                              =   0.00

        net PnL = basis + funding - fees
                = -9.00 + 1.5005 - 22.5020 = -30.0015
    """
    bar = minute(0)
    book(position.model, bar)
    outcome = position.apply(position.plan(HedgeTarget(D("0.500")), bar), bar, equity=EQUITY)
    assert outcome.state is HedgeState.HEDGED

    # -- entry ---------------------------------------------------------------
    assert position.leg(SPOT).entry_price == D("30007.00")
    assert position.leg(PERP).entry_price == D("29994.00")
    assert position.leg(SPOT).quantity == position.leg(PERP).quantity == D("0.500")
    assert position.ledger.state.entry_basis == D("-13.00")

    ledger = position.ledger.state
    assert ledger.spot.fees == D("15.00350000")
    assert ledger.perp.fees == D("7.498500000")
    assert ledger.fees == D("22.50200000")

    # -- one funding settlement, received by the SHORT -----------------------
    assert position.ledger.book_funding(1, D("1.500500000")) == D("1.500500000")
    assert ledger.funding_received == D("1.500500000")
    assert ledger.funding_paid == D("0")

    # -- mark ----------------------------------------------------------------
    marked = position.mark_to_market(minute(1, spot="31000.00", perp="31005.00"))
    assert marked.basis == D("5.00")
    assert marked.spot_pnl == D("496.50000")
    assert marked.perp_pnl == D("-505.50000")
    assert marked.spot_pnl + marked.perp_pnl == D("-9.00000")
    assert marked.identity_residual == D("0.00000")
    assert position.state is HedgeState.HEDGED

    # -- the net, reached two independent ways -------------------------------
    from_components = (
        D("0.500") * (ledger.entry_basis - marked.basis)
        + ledger.net_funding
        - ledger.fees
        - ledger.slippage
    )
    assert marked.equity - ledger.capital == from_components
    assert from_components == D("-30.00150000") - ledger.slippage


def test_a_correction_flatten_returns_the_cash_of_a_booked_entry(position):
    """The correction path books a reduction too, not only the emergency one.

    `flatten_for_correction` is reachable from a HEDGED position: a rebalance
    that leaves one leg behind goes PARTIAL, and an exhausted correction
    flattens. Before `book_reduction` existed neither flatten path told the carry
    ledger anything, so `free_cash` kept the whole entry debit and every later
    equity reading was wrong by roughly the position's notional.

    The oracle is the ENTRY record plus the legs' own realised PnL -- neither of
    them written by the code under test.
    """
    open_hedge(position, "0.500")
    ledger = position.ledger.state
    assert ledger.quantity == D("0.500") and ledger.spot_entry is not None
    principal = ledger.quantity * ledger.spot_entry
    margin = ledger.perp_margin
    cash_before = ledger.free_cash
    frictions_before = ledger.fees + ledger.slippage
    realised_before = ledger.realised

    later = minute(1)
    book(position.model, later)
    position.flatten_for_correction(later)

    assert position.state is HedgeState.FLAT
    assert ledger.quantity == D("0") and ledger.perp_margin == D("0")
    assert ledger.spot_entry is None and ledger.entry_basis is None

    returned = ledger.free_cash - cash_before
    exit_frictions = (ledger.fees + ledger.slippage) - frictions_before
    assert (
        returned == principal + margin + (ledger.realised - realised_before) - exit_frictions
    )

    # The realised half, cross-checked against the executors rather than against
    # the ledger that is under test.
    legs_realised = position.spot.ledger.realised_pnl + position.perp.ledger.realised_pnl
    assert ledger.realised == legs_realised
