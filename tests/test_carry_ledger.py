"""``CarryLedger``: persistence, fail-closed loading, funding, and the identity.

Every test here is synthetic arithmetic. No market data is read.
"""

from __future__ import annotations

import json
from decimal import Decimal as D

import pytest

from chimera.carry.ledger import (
    IDENTITY_TOLERANCE,
    LEDGER_SCHEMA,
    CarryLedger,
    LedgerError,
    LoadOutcome,
)

CAPITAL = D("1000000")


def fresh(tmp_path, capital: D = CAPITAL) -> CarryLedger:
    return CarryLedger.open(tmp_path / "carry_ledger.json", capital=capital)


def opened(tmp_path) -> CarryLedger:
    """A ledger with one hedge booked: 0.5 BTC, spot 30000, perp 30030."""
    ledger = fresh(tmp_path)
    ledger.book_position(
        spot_quantity=D("0.5"),
        spot_entry=D("30000"),
        perp_quantity=D("0.5"),
        perp_entry=D("30030"),
    )
    ledger.book_costs(leg="spot", fee=D("15"), slippage=D("3"))
    ledger.book_costs(leg="perp", fee=D("7.5075"), slippage=D("3"))
    return ledger


def test_a_missing_file_starts_a_ledger_against_the_configured_capital(tmp_path):
    ledger = fresh(tmp_path)
    assert ledger.outcome is LoadOutcome.MISSING
    assert ledger.state.capital == CAPITAL
    assert ledger.state.free_cash == CAPITAL
    assert ledger.disputed is None


def test_the_round_trip_is_exact(tmp_path):
    ledger = opened(tmp_path)
    ledger.book_funding(1, D("-1.5"))
    ledger.mark(spot_close=D("30010"), perp_close=D("30040"), equity=D("999980"))
    ledger.save(now_ns=1_700_000_000_000_000_000)

    reloaded = CarryLedger.open(tmp_path / "carry_ledger.json", capital=CAPITAL)
    assert reloaded.outcome is LoadOutcome.LOADED
    assert reloaded.state.to_dict() == ledger.state.to_dict()
    assert reloaded.state.quantity == D("0.5")
    assert reloaded.state.funding_paid == D("1.5")
    assert reloaded.state.settled == [1]


def test_the_persisted_file_declares_the_schema(tmp_path):
    ledger = opened(tmp_path)
    ledger.save()
    payload = json.loads((tmp_path / "carry_ledger.json").read_text(encoding="utf-8"))
    assert payload["schema"] == LEDGER_SCHEMA


def test_the_write_is_atomic_and_leaves_no_temporary_behind(tmp_path):
    ledger = opened(tmp_path)
    ledger.save()
    assert (tmp_path / "carry_ledger.json").is_file()
    assert not list(tmp_path.glob("*.tmp"))


def test_a_write_failure_is_raised_rather_than_swallowed(tmp_path, monkeypatch):
    ledger = opened(tmp_path)

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("chimera.carry.ledger.os.replace", boom)
    with pytest.raises(LedgerError, match="could not persist"):
        ledger.save()


@pytest.mark.parametrize(
    "body",
    [
        "{ not json",
        json.dumps({"schema": "chimera.carry-ledger/999", "capital": "1000000"}),
        json.dumps({"capital": "1000000"}),
        json.dumps({"schema": LEDGER_SCHEMA, "capital": "not-a-number"}),
        json.dumps({"schema": LEDGER_SCHEMA, "capital": "1000000", "settled": "nope"}),
        json.dumps({"schema": LEDGER_SCHEMA, "capital": "1000000", "settled": ["x"]}),
    ],
)
def test_a_damaged_ledger_disputes_and_is_never_silently_reset(tmp_path, body):
    path = tmp_path / "carry_ledger.json"
    path.write_text(body, encoding="utf-8")
    before = path.read_bytes()

    ledger = CarryLedger.open(path, capital=CAPITAL)

    assert ledger.outcome is LoadOutcome.UNREADABLE
    assert ledger.disputed is not None
    assert ledger.disputed.startswith("ledger_unreadable")
    assert path.read_bytes() == before


def test_a_capital_mismatch_disputes(tmp_path):
    ledger = opened(tmp_path)
    ledger.save()
    reloaded = CarryLedger.open(tmp_path / "carry_ledger.json", capital=D("5"))
    assert reloaded.outcome is LoadOutcome.LOADED
    assert reloaded.disputed is not None
    assert "ledger_capital_mismatch" in reloaded.disputed


def test_resolution_requires_a_note_and_records_it(tmp_path):
    ledger = opened(tmp_path)
    ledger.dispute("identity_violation: synthetic")

    for empty in ("", "   "):
        with pytest.raises(LedgerError, match="requires an operator note"):
            ledger.resolve(empty)
    assert ledger.disputed is not None

    ledger.resolve("checked both stores by hand; perp fill was double-counted")
    assert ledger.disputed is None
    assert len(ledger.state.resolutions) == 1
    assert ledger.state.resolutions[0]["cleared"].startswith("identity_violation")
    assert "double-counted" in ledger.state.resolutions[0]["note"]


def test_resolving_nothing_is_refused(tmp_path):
    ledger = opened(tmp_path)
    with pytest.raises(LedgerError, match="no dispute"):
        ledger.resolve("nothing was wrong")


def test_the_first_dispute_reason_is_kept(tmp_path):
    """A later symptom must not overwrite the first cause."""
    ledger = opened(tmp_path)
    ledger.dispute("first")
    ledger.dispute("second")
    assert ledger.disputed == "first"


def test_a_dispute_must_state_a_reason(tmp_path):
    with pytest.raises(LedgerError, match="must state a reason"):
        opened(tmp_path).dispute("")


def test_received_and_paid_are_never_netted(tmp_path):
    ledger = opened(tmp_path)
    ledger.book_funding(1, D("10"))
    ledger.book_funding(2, D("-4"))
    assert ledger.state.funding_received == D("10")
    assert ledger.state.funding_paid == D("4")
    assert ledger.state.net_funding == D("6")


def test_a_settlement_is_booked_once(tmp_path):
    ledger = opened(tmp_path)
    cash = ledger.state.free_cash
    assert ledger.book_funding(7, D("10")) == D("10")
    assert ledger.book_funding(7, D("10")) == D("0")
    assert ledger.state.free_cash == cash + D("10")
    assert ledger.state.settled == [7]


def test_the_dedup_key_survives_a_restart(tmp_path):
    """A crash between the venue settling and the ledger saving must not double-book."""
    ledger = opened(tmp_path)
    ledger.book_funding(7, D("10"))
    ledger.save()

    reloaded = CarryLedger.open(tmp_path / "carry_ledger.json", capital=CAPITAL)
    assert reloaded.book_funding(7, D("10")) == D("0")
    assert reloaded.state.funding_received == D("10")


def test_a_zero_flow_is_recorded_as_settled_without_moving_either_accumulator(tmp_path):
    ledger = opened(tmp_path)
    assert ledger.book_funding(3, D("0")) == D("0")
    assert ledger.state.settled == [3]
    assert ledger.state.funding_received == D("0")
    assert ledger.state.funding_paid == D("0")


def test_costs_are_attributed_per_leg(tmp_path):
    ledger = opened(tmp_path)
    assert ledger.state.spot.fees == D("15")
    assert ledger.state.perp.fees == D("7.5075")
    assert ledger.state.fees == D("22.5075")
    assert ledger.state.slippage == D("6")


def test_rebalance_cost_is_recorded_separately_and_also_in_the_leg(tmp_path):
    ledger = opened(tmp_path)
    ledger.book_costs(leg="spot", fee=D("0.5"), slippage=D("0.25"), rebalance=True)
    assert ledger.state.rebalance_cost == D("0.75")
    assert ledger.state.spot.fees == D("15.5")


def test_an_unknown_leg_is_refused(tmp_path):
    with pytest.raises(LedgerError, match="unknown leg"):
        opened(tmp_path).book_costs(leg="futures", fee=D("1"))


def test_worst_equity_only_ever_falls(tmp_path):
    ledger = opened(tmp_path)
    ledger.mark(spot_close=D("30000"), perp_close=D("30030"), equity=D("100"))
    ledger.mark(spot_close=D("30000"), perp_close=D("30030"), equity=D("40"))
    ledger.mark(spot_close=D("30000"), perp_close=D("30030"), equity=D("90"))
    assert ledger.state.worst_equity == D("40")
    assert ledger.state.last_equity == D("90")


def test_the_identity_holds_exactly_when_both_legs_move_together(tmp_path):
    """Q x (basis_in - basis_now) equals the two legs' marked PnL."""
    ledger = opened(tmp_path)
    residual = ledger.check_identity(spot_close=D("31000"), perp_close=D("31030"))
    assert residual == D("0")
    assert ledger.disputed is None
    assert ledger.state.identity_gap == D("0")


def test_the_identity_holds_when_the_basis_converges(tmp_path):
    ledger = opened(tmp_path)
    # Entry basis 30, now 10: the position has earned Q x 20.
    residual = ledger.check_identity(spot_close=D("31000"), perp_close=D("31010"))
    assert residual == D("0")
    assert ledger.disputed is None


def test_an_identity_violation_disputes(tmp_path):
    """A leg that is wrong by more than the tolerance is not averaged away."""
    ledger = opened(tmp_path)
    ledger.state.perp_entry = D("30100")
    residual = ledger.check_identity(spot_close=D("30000"), perp_close=D("30030"))
    assert abs(residual) > IDENTITY_TOLERANCE
    assert ledger.disputed is not None
    assert ledger.disputed.startswith("identity_violation")


def test_a_residual_inside_the_tolerance_does_not_dispute(tmp_path):
    """Two-sided: the tolerance exists because fee rounding is real."""
    ledger = opened(tmp_path)
    ledger.state.perp_entry = D("30030") + IDENTITY_TOLERANCE / D("4") / ledger.state.quantity
    residual = ledger.check_identity(spot_close=D("30000"), perp_close=D("30030"))
    assert D("0") < abs(residual) <= IDENTITY_TOLERANCE
    assert ledger.disputed is None


def test_the_identity_is_silent_before_an_entry(tmp_path):
    ledger = fresh(tmp_path)
    assert ledger.check_identity(spot_close=D("30000"), perp_close=D("30030")) == D("0")
    assert ledger.state.identity_gap is None
    assert ledger.disputed is None


def test_no_stale_leg_before_both_have_been_observed(tmp_path):
    ledger = opened(tmp_path)
    assert ledger.stale_leg(tolerance_ns=60_000_000_000) is None
    ledger.note_leg_mark("spot", 1_000)
    assert ledger.stale_leg(tolerance_ns=60_000_000_000) is None


@pytest.mark.parametrize(
    "spot_ns,perp_ns,expected",
    [
        (0, 0, None),
        (60_000_000_000, 0, None),
        (0, 60_000_000_000, None),
        (60_000_000_001, 0, "perp"),
        (0, 60_000_000_001, "spot"),
    ],
)
def test_the_stale_leg_boundary_is_pinned_on_both_sides(tmp_path, spot_ns, perp_ns, expected):
    """Exactly the tolerance apart is fresh; one nanosecond more is not."""
    ledger = opened(tmp_path)
    ledger.note_leg_mark("spot", spot_ns)
    ledger.note_leg_mark("perp", perp_ns)
    assert ledger.stale_leg(tolerance_ns=60_000_000_000) == expected


def test_the_observation_instants_persist(tmp_path):
    ledger = opened(tmp_path)
    ledger.note_leg_mark("spot", 5)
    ledger.note_leg_mark("perp", 9)
    ledger.save()
    reloaded = CarryLedger.open(tmp_path / "carry_ledger.json", capital=CAPITAL)
    assert reloaded.state.marked_at_ns == {"spot": 5, "perp": 9}


def test_an_in_memory_ledger_persists_nothing(tmp_path):
    ledger = CarryLedger.open(None, capital=CAPITAL)
    ledger.save()
    assert not list(tmp_path.iterdir())


# ---------------------------------------------------------------------------
# Amendment A12: measured slippage is evidence, not a second cash debit
# ---------------------------------------------------------------------------
def test_amendment_a12_slippage_is_accumulated_and_not_spent(tmp_path):
    """A12's rule, stated as arithmetic: fees move cash, slippage does not.

    Written out from `docs/amendment_a12_recorded_quote_slippage.md` the way
    `test_amendment_a10s_four_cases_land_in_the_column_a10_names` is written out
    from A10, so the adopted decision has an executable witness and not only a
    docstring arguing for it.
    """
    ledger = fresh(tmp_path)
    before = ledger.state.free_cash

    ledger.book_costs(leg="spot", fee=D("0"), slippage=D("42"))
    assert ledger.state.slippage == D("42"), "slippage must still be accumulated"
    assert ledger.state.free_cash == before, "A12: slippage must not move free_cash"

    ledger.book_costs(leg="spot", fee=D("10"), slippage=D("0"))
    assert ledger.state.free_cash == before - D("10"), "fees are still a cash debit"


def test_amendment_a12_hand_traced_against_an_executed_vwap(tmp_path):
    """The reason A12 is right, traced by hand on one LONG spot leg.

    A BUY of 1 BTC with a recorded ask of 30000 and 2 bps of configured
    slippage fills at `30000 * 1.0002 = 30006`. The 6 quote units of slippage
    are INSIDE that price, so the principal the ledger books is 30006 and the
    cash it moves is 30006 -- not 30006 + 6. The measured slippage is reported
    beside the cash line, which is what section 6.5 asks for.

    The two-sided control is the second half: booking at the UN-slipped 30000
    and charging the 6 separately reaches the same free cash. That is the
    disjoint-terms model `chimera.carry.accounting` uses and the one section
    6.6's line was written for, and it is exactly why doing BOTH double-charges.
    """
    ask, slip_rate, qty = D("30000"), D("0.0002"), D("1")
    executed = ask * (1 + slip_rate)
    measured_slippage = (executed - ask) * qty
    fee = executed * qty * D("0.001")
    assert executed == D("30006.0000") and measured_slippage == D("6.0000")

    (tmp_path / "a").mkdir()
    priced_in = CarryLedger.open((tmp_path / "a") / "l.json", capital=CAPITAL)
    priced_in.book_position(
        spot_quantity=qty, spot_entry=executed, perp_quantity=D("0"), perp_entry=D("0")
    )
    priced_in.book_costs(leg="spot", fee=fee, slippage=measured_slippage)

    # The disjoint-terms model: un-slipped notional, slippage charged as cash.
    (tmp_path / "b").mkdir()
    disjoint = CarryLedger.open((tmp_path / "b") / "l.json", capital=CAPITAL)
    disjoint.book_position(
        spot_quantity=qty, spot_entry=ask, perp_quantity=D("0"), perp_entry=D("0")
    )
    disjoint.book_costs(leg="spot", fee=fee + measured_slippage, slippage=D("0"))

    assert (
        priced_in.state.free_cash == disjoint.state.free_cash
    ), "the two models must agree on cash; only one of them may be applied"
    assert priced_in.state.free_cash == CAPITAL - executed * qty - fee
    # ...and only the A12 model still has the slippage available as evidence.
    assert priced_in.state.slippage == measured_slippage
    assert disjoint.state.slippage == D("0")

    # What the double debit would have cost: charged twice, cash is short by
    # exactly the measured slippage, and it grows with every fill.
    doubled = priced_in.state.free_cash - measured_slippage
    assert doubled != priced_in.state.free_cash
