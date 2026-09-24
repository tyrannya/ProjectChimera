"""R1-k: every Aegis limit enforces something, and the detector that keeps it so.

> **R1-k Unreachable Aegis rules:** the funding-cost entry veto made reachable
> (pass `funding_rate` to `execute_target`) or removed; loss-streak and cooldown
> limits either driven by a real `record_trade_result` caller or removed from
> the config schema; each wiring/removal has a breach-refused and non-breach-
> allowed test; the unsafe-default detector (a configured limit that no code
> path enforces) added to CI.

Three limits were configured, mapped into Aegis, and enforced nowhere, and
`chimera/demo/risk_wiring.py` said so in prose:

* ``max_funding_cost_rate`` -- ``hedge.py`` called ``execute_target`` without a
  ``funding_rate``, so ``evaluate_entry`` skipped the funding branch entirely;
* ``loss_streak_limit`` and ``cooldown_seconds`` -- nothing anywhere called
  ``record_trade_result``, so ``consecutive_losses`` never left zero, the
  cooldown was never opened, and the gate that vetoes while it is open could
  never close.

All three are now wired, each with the two-sided witness R1-k asks for. The last
section is the detector: a limit that no code path enforces fails the suite
rather than sitting in a config file looking like protection.

ENGINEERING only. Nothing here asserts an economic quantity.
"""

from __future__ import annotations

import ast
from dataclasses import fields
from decimal import Decimal
from pathlib import Path

import pytest

from chimera.carry.hedge import PERP, SPOT, HedgedPosition
from chimera.risk import RiskEngine, RiskLimits

REPO = Path(__file__).resolve().parents[1]


def _engine(**limits) -> RiskEngine:
    engine = RiskEngine(RiskLimits(**limits))
    engine.update_equity(100_000.0)
    return engine


def _entry(engine: RiskEngine, **kwargs):
    """One exposure-increasing order, through the gate every order goes through."""
    defaults = dict(
        pair="BTC/USDT:USDT",
        equity=100_000.0,
        entry_price=100_000.0,
        stop_price=95_000.0,
        proposed_stake=1_000.0,
    )
    defaults.update(kwargs)
    return engine.evaluate_entry(**defaults)


# ---------------------------------------------------------------------------
# max_funding_cost_rate
# ---------------------------------------------------------------------------
def test_the_funding_cost_veto_refuses_a_breach():
    """Breach refused: a SHORT that would PAY more than the ceiling."""
    engine = _engine(max_funding_cost_rate=0.0005)

    decision = _entry(engine, funding_rate=-0.001, position_side="short")

    assert decision.allowed is False
    assert "funding" in decision.reason


def test_the_funding_cost_veto_allows_a_non_breach():
    """Non-breach allowed, and the same rate on the other side is a REBATE.

    The sign convention is the whole content of the rule: a SHORT receives a
    positive rate and pays a negative one. A veto that judged the magnitude
    would refuse exactly the settlements the position is being paid for.
    """
    engine = _engine(max_funding_cost_rate=0.0005)

    assert _entry(engine, funding_rate=0.001, position_side="short").allowed is True
    assert _entry(engine, funding_rate=-0.0001, position_side="short").allowed is True


def test_the_perpetual_leg_now_reaches_the_veto_with_a_rate():
    """The reachability half: the demo path passes a rate where it never did.

    The rule was never unimplemented -- it was unreachable, because every call
    arrived with ``funding_rate=None`` and the branch was skipped. This asserts
    the wiring rather than the rule.
    """

    class _State:
        funding_rate_last = Decimal("-0.001")

    assert HedgedPosition._funding_rate_for(PERP, _State()) == pytest.approx(-0.001)


def test_the_spot_leg_is_given_no_rate_at_all():
    """The two-sided half, and it is not an omission.

    Spot inventory pays and receives no funding. Aegis reads the rate
    side-awarely, so handing one to the LONG spot leg would veto spot entries
    whenever the perpetual rate was positive -- a veto on a cost that leg does
    not pay.
    """

    class _State:
        funding_rate_last = Decimal("-0.001")

    assert HedgedPosition._funding_rate_for(SPOT, _State()) is None


def test_a_minute_with_no_published_rate_skips_the_branch():
    """No rate is no veto. A fabricated one would judge on nobody's number."""

    class _State:
        funding_rate_last = None

    assert HedgedPosition._funding_rate_for(PERP, _State()) is None


# ---------------------------------------------------------------------------
# loss_streak_limit and cooldown_seconds
# ---------------------------------------------------------------------------
def test_a_loss_streak_opens_a_cooldown_and_the_cooldown_refuses_entries():
    """Breach refused: the Nth consecutive loss closes the gate."""
    clock = {"now": 1_000.0}
    engine = RiskEngine(
        RiskLimits(loss_streak_limit=3, cooldown_seconds=3_600.0),
        clock=lambda: clock["now"],
    )
    engine.update_equity(100_000.0)

    engine.record_trade_result(-10.0)
    engine.record_trade_result(-10.0)
    assert _entry(engine).allowed is True, "two losses is not yet a streak"

    engine.record_trade_result(-10.0)

    decision = _entry(engine)
    assert decision.allowed is False
    assert "cooldown" in decision.reason


def test_a_win_breaks_the_streak_and_leaves_the_gate_open():
    """Non-breach allowed: the counter is consecutive, not cumulative."""
    clock = {"now": 1_000.0}
    engine = RiskEngine(
        RiskLimits(loss_streak_limit=3, cooldown_seconds=3_600.0),
        clock=lambda: clock["now"],
    )
    engine.update_equity(100_000.0)

    engine.record_trade_result(-10.0)
    engine.record_trade_result(-10.0)
    engine.record_trade_result(+1.0)
    engine.record_trade_result(-10.0)

    assert _entry(engine).allowed is True


def test_the_cooldown_expires_rather_than_latching():
    """The other side of the gate: it is a cooldown, not a halt."""
    clock = {"now": 1_000.0}
    engine = RiskEngine(
        RiskLimits(loss_streak_limit=1, cooldown_seconds=600.0),
        clock=lambda: clock["now"],
    )
    engine.update_equity(100_000.0)

    engine.record_trade_result(-10.0)
    assert _entry(engine).allowed is False

    clock["now"] += 601.0
    assert _entry(engine).allowed is True


def test_a_closed_round_trip_reports_its_result_and_an_open_one_does_not():
    """The reachability half, on the ledger that produces the number.

    ``note_cycle`` states a result only on the call that CLOSES a position, and
    only when the opening equity was recorded. Everything else is ``None``, and
    ``None`` is not zero: a zero is a trade that did not lose.
    """
    from chimera.carry.ledger import CarryLedger

    ledger = CarryLedger.open(None, capital=Decimal("1000000"))

    assert ledger.note_cycle(flat=False, equity=Decimal("1000")) is None
    assert ledger.note_cycle(flat=False, equity=Decimal("1010")) is None, "still open"

    assert ledger.note_cycle(flat=True, equity=Decimal("990")) == Decimal("-10")
    assert ledger.note_cycle(flat=True, equity=Decimal("990")) is None, "already closed"


def test_a_close_whose_open_was_never_recorded_reports_nothing():
    """The fail-safe half: an unmeasured cycle resets no streak."""
    from chimera.carry.ledger import CarryLedger

    ledger = CarryLedger.open(None, capital=Decimal("1000000"))

    assert ledger.note_cycle(flat=True, equity=Decimal("990")) is None


def test_a_close_by_any_path_clears_the_cycle_baseline(tmp_path):
    """A round trip closed by a path other than ``apply`` must not leak forward.

    ``apply`` is one of four ways a position flattens --
    ``flatten_for_correction``, ``emergency_reduce`` and ``reconstruct`` are the
    others. Measuring the cycle from ``apply`` alone would leave the opening
    equity of a position closed by any of them standing, and the NEXT round trip
    would be measured from it: a profit reported as a loss, or the reverse,
    driving the very gates this file is about.

    The bookkeeping therefore lives in ``mark_to_market``, which runs every
    minute and sees every close however it happened. This asserts that the
    baseline is taken and cleared by the MARK, on a ledger, so the property does
    not depend on which method flattened the position.
    """
    from chimera.carry.ledger import CarryLedger

    ledger = CarryLedger.open(None, capital=Decimal("1000000"))

    assert ledger.note_cycle(flat=False, equity=Decimal("1000")) is None
    # The position is flattened by some other path; the next mark sees it flat.
    assert ledger.note_cycle(flat=True, equity=Decimal("980")) == Decimal("-20")
    # And the cycle after it is measured from ITS own open, not from the first.
    assert ledger.note_cycle(flat=False, equity=Decimal("980")) is None
    assert ledger.note_cycle(flat=True, equity=Decimal("990")) == Decimal("10")


def test_the_cycle_is_measured_after_execution_not_before(tmp_path):
    """The closing legs' fees and slippage are part of what the trade cost.

    An equity read before execution -- which is what ``apply`` is handed --
    contains the OPENING legs' frictions and not the closing ones, so a
    marginally losing round trip reports as a win and resets a loss streak. The
    mark runs after execution, so this is a property of where the call sits;
    asserted here as the ordering the runner relies on.
    """
    runner = (REPO / "chimera" / "demo" / "runner.py").read_text(encoding="utf-8")

    apply_at = runner.index("outcome = self.position.apply(")
    mark_at = runner.index("mark = self.position.mark_to_market(state)", apply_at)
    report_at = runner.index("self.risk.record_trade_result(", mark_at)
    assert apply_at < mark_at < report_at


# ---------------------------------------------------------------------------
# the detector
# ---------------------------------------------------------------------------
#: Every ``RiskLimits`` field, and where the enforcement that reads it lives.
#: This is the R1-k detector's declaration, and it is deliberately a statement
#: about ENFORCEMENT rather than about configuration: a limit can be present in
#: a config file, mapped into the engine and read by an expression, and still
#: protect nothing, which is exactly what happened to the three above.
#:
#: A field that enforces nothing belongs in :data:`NOT_ENFORCED` with the reason.
ENFORCED_BY: dict[str, str] = {
    "max_drawdown_pct": "update_equity halts on the drawdown from peak",
    "max_daily_loss_pct": "update_equity halts on the loss from the day's start",
    "max_open_positions": "evaluate_entry vetoes a new pair beyond the count",
    "max_total_exposure_pct": "evaluate_entry vetoes on projected total exposure",
    "max_exposure_per_asset_pct": "evaluate_entry vetoes on this pair's exposure",
    "max_position_pct": "position_size caps one order's stake",
    "max_leverage": "evaluate_entry vetoes above it; position_size divides by it",
    "max_orders_per_minute": "record_order halts above the rolling-window count",
    "loss_streak_limit": "record_trade_result opens a cooldown at the Nth loss (R1-k)",
    "cooldown_seconds": "the length of that cooldown; evaluate_entry vetoes until it ends",
    "max_data_delay_s": (
        "note_feed marks the feed stale past the delay and evaluate_entry vetoes on "
        "the mark; the runner passes no data_delay_s, so the mark IS the demo's "
        "enforcement rather than the direct comparison beside it"
    ),
    "max_funding_cost_rate": "evaluate_entry's side-aware funding veto (R1-k)",
    "funding_adverse_streak_limit": "note_funding_settlement raises funding_halt",
    "min_liquidation_distance_pct": "evaluate_entry vetoes a too-near liquidation",
}

#: Fields whose enforcement EXISTS in the engine and is not reached from the
#: demo path, each with what is missing. This is the category R1-k is about, and
#: keeping it -- rather than emptying it and declaring victory -- is the point:
#: a limit here is protection the campaign does not currently have, said out
#: loud, and a limit that moves out of it has to move with a witness.
UNREACHABLE_ON_DEMO_PATH: dict[str, str] = {
    "max_funding_rate": (
        "the sign-blind ceiling, taken only when a caller names NO position side. "
        "Since R1-k a rate does arrive, but every demo caller names a side, so "
        "evaluate_entry always takes the side-aware max_funding_cost_rate branch. "
        "Kept in step with that ceiling so the superset relation holds for the day "
        "a caller does not name one."
    ),
    "max_inference_staleness_s": (
        "there is no inference on the carry path to be stale. No caller passes an "
        "inference_age_s because no model produces one; the limit becomes reachable "
        "when a learned candidate does, which is R8's runtime and not this one."
    ),
}

#: Fields that enforce nothing, each with the reason. Naming one here is what
#: makes it a decision instead of an oversight -- and moving a field into this
#: map is a visible change in review rather than a silent default.
NOT_ENFORCED: dict[str, str] = {
    "risk_per_trade_pct": (
        "there is no stop on the demo path for it to be about; a carry hedge has "
        "no stop-loss. chimera/demo/risk_wiring.py records the invariant that "
        "replaces it."
    ),
    "min_stop_distance_pct": "the sizing band's lower edge, not a gate",
    "max_stop_distance_pct": "the sizing band's upper edge, not a gate",
}


def test_every_limit_is_classified():
    """Exhaustive, so a new limit cannot arrive unclassified.

    The same shape as ``risk_wiring.check_exhaustive``: a field added to
    ``RiskLimits`` and to a config file, but never to either map, fails here
    rather than shipping as protection nobody wired.
    """
    maps = (ENFORCED_BY, UNREACHABLE_ON_DEMO_PATH, NOT_ENFORCED)
    declared = set().union(*maps)
    actual = {field.name for field in fields(RiskLimits)}

    counted = sum(len(entry) for entry in maps)
    assert counted == len(declared), "a limit appears in more than one map"
    assert actual - declared == set(), "unclassified limit(s)"
    assert declared - actual == set(), "classified limit(s) that no longer exist"


def test_every_enforced_limit_is_actually_read_by_the_engine():
    """The static half of the detector: a claimed enforcement that reads nothing.

    Walks ``chimera/risk.py``'s AST for attribute reads of each limit -- on
    ``lim``, ``self.limits`` or any other holder -- rather than grepping, so a
    name that appears only in a docstring or a comment does not count as
    enforcement.
    """
    tree = ast.parse((REPO / "chimera" / "risk.py").read_text(encoding="utf-8"))
    read = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load)
    }

    claimed = set(ENFORCED_BY) | set(UNREACHABLE_ON_DEMO_PATH)
    unread = sorted(name for name in claimed if name not in read)
    assert unread == [], f"claimed to be enforced but never read: {unread}"


def test_the_three_limits_r1k_wired_have_a_reachable_caller():
    """The reachability half, and the defect R1-k names.

    A limit can be read by an expression the production path never reaches --
    which is exactly what ``max_funding_cost_rate`` was, sitting inside a branch
    guarded by a ``funding_rate`` nobody passed. The static check above cannot
    see that, so the caller is asserted directly from the source of the modules
    that must do the calling.
    """
    hedge = (REPO / "chimera" / "carry" / "hedge.py").read_text(encoding="utf-8")
    runner = (REPO / "chimera" / "demo" / "runner.py").read_text(encoding="utf-8")

    assert "funding_rate=self._funding_rate_for(" in hedge, "the perp leg passes no rate"
    assert "self.risk.record_trade_result(" in runner, "no caller drives the loss streak"
