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

from chimera.carry.hedge import PERP, SPOT, HedgeState, HedgedPosition
from chimera.carry.ledger import CarryLedger
from chimera.risk import RiskEngine, RiskLimits

REPO = Path(__file__).resolve().parents[1]
MINUTE_MS = 60_000


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
# max_funding_cost_rate: the rule
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


# ---------------------------------------------------------------------------
# max_funding_cost_rate: the reachability, and WHICH rate
# ---------------------------------------------------------------------------
class _State:
    """A minute carrying both rates, deliberately different numbers."""

    funding_rate_next = Decimal("-0.001")  # the settlement AHEAD
    funding_rate_last = Decimal("+0.004")  # the settlement already charged


def test_the_perpetual_leg_now_reaches_the_veto_with_a_rate():
    """The reachability half: the demo path passes a rate where it never did.

    The rule was never unimplemented -- it was unreachable, because every call
    arrived with ``funding_rate=None`` and the branch was skipped. This asserts
    the wiring rather than the rule.
    """
    assert HedgedPosition._funding_rate_for(PERP, _State()) == pytest.approx(-0.001)


def test_the_veto_judges_the_rate_ahead_and_not_the_one_already_charged():
    """The two numbers under confusingly close names, held apart.

    ``MarketState`` carries both. ``funding_rate_next`` is the venue's ``r`` from
    the mark-price stream: the rate standing for the settlement AHEAD, which is
    what Aegis's refusal is about -- it says the rate "would cost this position X
    per settlement". ``funding_rate_last`` is the last REALISED rate, read off
    the recorded settlement rows, and vetoing a new entry on it would be a veto
    on a cost somebody else already paid.

    The two are given OPPOSITE SIGNS here, so a wiring that reached for the
    nearer-looking name is not merely imprecise but inverts the verdict: at
    -0.001 a SHORT pays and is refused, at +0.004 a SHORT is paid and allowed.
    On the fixture's own minutes, where funding is flat, no test could tell.
    """
    rate = HedgedPosition._funding_rate_for(PERP, _State())

    assert rate == pytest.approx(-0.001)
    assert rate != pytest.approx(float(_State.funding_rate_last))

    engine = _engine(max_funding_cost_rate=0.0005)
    assert _entry(engine, funding_rate=rate, position_side="short").allowed is False
    assert (
        _entry(
            engine,
            funding_rate=float(_State.funding_rate_last),
            position_side="short",
        ).allowed
        is True
    ), "the realised rate would have allowed exactly what the forward rate refuses"


def test_the_spot_leg_is_given_no_rate_at_all():
    """The two-sided half, and it is not an omission.

    Spot inventory pays and receives no funding. Aegis reads the rate
    side-awarely, so handing one to the LONG spot leg would veto spot entries
    whenever the perpetual rate was positive -- a veto on a cost that leg does
    not pay.
    """
    assert HedgedPosition._funding_rate_for(SPOT, _State()) is None


def test_a_minute_with_no_published_rate_skips_the_branch():
    """No rate is no veto. A fabricated one would judge on nobody's number."""

    class _NoRate:
        funding_rate_next = None
        funding_rate_last = Decimal("-0.001")

    assert HedgedPosition._funding_rate_for(PERP, _NoRate()) is None


def test_the_feed_reads_the_forward_rate_off_the_perpetual_row(tmp_path):
    """Where ``funding_rate_next`` comes from, asserted against the real files.

    The recorder's COLUMN is called ``funding_rate_last`` and holds the mark
    stream's ``r`` -- the last rate PUBLISHED, which is the next one to be
    charged. ``MarketState`` names it for what it is about instead, and the two
    fields must not be crossed: this asserts the state's forward rate against the
    perpetual row and its realised rate against the settlement rows.
    """
    from tests.demo_harness import build

    harness = build(tmp_path)
    minute = harness.first_minute_ms()
    state = harness.runner.cursor.state_for(minute, now_ns=harness.runner.clock.now_ns)
    row = harness.runner.cursor.record("um", minute)

    assert row is not None
    assert state.funding_rate_next == row.decimal("funding_rate_last")


# ---------------------------------------------------------------------------
# loss_streak_limit and cooldown_seconds: the rule
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


# ---------------------------------------------------------------------------
# loss_streak_limit and cooldown_seconds: the number, and who reports it
# ---------------------------------------------------------------------------
def test_a_closed_round_trip_reports_its_result_and_an_open_one_does_not():
    """``note_cycle`` states a result only on the call that CLOSES a position.

    And only when the opening equity was recorded. Everything else is ``None``,
    and ``None`` is not zero: a zero is a trade that did not lose.
    """
    ledger = CarryLedger.open(None, capital=Decimal("1000000"))

    assert ledger.note_cycle(flat=False, equity=Decimal("1000")) is None
    assert ledger.note_cycle(flat=False, equity=Decimal("1010")) is None, "still open"

    assert ledger.note_cycle(flat=True, equity=Decimal("990")) == Decimal("-10")
    assert ledger.note_cycle(flat=True, equity=Decimal("990")) is None, "already closed"


def test_a_close_whose_open_was_never_recorded_reports_nothing():
    """The fail-safe half: an unmeasured cycle resets no streak."""
    ledger = CarryLedger.open(None, capital=Decimal("1000000"))

    assert ledger.note_cycle(flat=True, equity=Decimal("990")) is None


def test_the_baseline_survives_a_restart(tmp_path):
    """``equity_at_open`` is persisted, so a cycle is not remeasured mid-flight.

    Without it a restart while a position is open would measure the round trip
    from the restart rather than from the open, and report a result the trade
    never had.
    """
    path = tmp_path / "carry_ledger.json"
    ledger = CarryLedger.open(path, capital=Decimal("1000000"))
    ledger.note_cycle(flat=False, equity=Decimal("1000"))
    ledger.save()

    reloaded = CarryLedger.open(path, capital=Decimal("1000000"))

    assert reloaded.state.equity_at_open == Decimal("1000")
    assert reloaded.note_cycle(flat=True, equity=Decimal("1007")) == Decimal("7")


def test_a_file_written_before_the_field_existed_still_loads(tmp_path):
    """Additive, like ``open_instant_ns``: absent reads as none, same schema id.

    And the absence means what it says -- an unknown baseline -- so the first
    close after such a load reports nothing rather than measuring from a zero.
    """
    import json

    path = tmp_path / "carry_ledger.json"
    CarryLedger.open(path, capital=Decimal("1000000")).save()
    body = json.loads(path.read_text(encoding="utf-8"))
    del body["equity_at_open"]
    path.write_text(json.dumps(body), encoding="utf-8")

    reloaded = CarryLedger.open(path, capital=Decimal("1000000"))

    assert reloaded.state.equity_at_open is None
    assert reloaded.note_cycle(flat=True, equity=Decimal("990")) is None


def test_the_mark_and_not_apply_is_what_sees_a_close(tmp_path):
    """Why the bookkeeping sits in ``mark_to_market``.

    ``apply`` is one of four paths that can flatten a position --
    ``flatten_for_correction``, ``emergency_reduce`` and ``reconstruct`` are the
    others -- and ``equity_at_open`` is persisted, so a result taken only from
    ``apply`` would leave the baseline of a position closed by any of them
    standing and measure the NEXT round trip from it. A profit could then be
    reported as a loss, driving the very gates R1-k just made reachable.
    """
    from tests.demo_harness import build

    harness = build(tmp_path)
    first = harness.first_minute_ms()
    minute = first
    for index in range(6):
        minute = first + index * MINUTE_MS
        harness.tick(minute)
        if harness.runner.position.state is HedgeState.HEDGED:
            break
    assert harness.runner.position.state is HedgeState.HEDGED
    assert harness.runner.position.ledger.state.equity_at_open is not None

    # Flattened by a path that is NOT `apply`.
    state = harness.runner.cursor.state_for(minute, now_ns=harness.runner.clock.now_ns)
    harness.runner.position.install_quote(state)
    from chimera.futures.executor import FlattenCause

    harness.runner.position.emergency_reduce(FlattenCause.RISK_HALT, state)
    mark = harness.runner.position.mark_to_market(state)

    assert harness.runner.position.state is HedgeState.FLAT
    assert mark.cycle_result is not None, "the close was not seen"
    assert (
        harness.runner.position.ledger.state.equity_at_open is None
    ), "baseline left standing"


def test_the_cycle_result_stays_out_of_the_decision_record():
    """It is a message to Aegis about a finished trade, not this minute's mark.

    In the record it would make one minute's evidence carry another's outcome.
    """
    from chimera.carry.hedge import CarryMark

    mark = CarryMark(
        quantity=Decimal("1"),
        spot_close=Decimal("1"),
        perp_close=Decimal("1"),
        basis=Decimal("0"),
        spot_pnl=Decimal("0"),
        perp_pnl=Decimal("0"),
        equity=Decimal("1"),
        identity_residual=Decimal("0"),
        cycle_result=Decimal("-5"),
    )

    assert "cycle_result" not in mark.to_dict()


def test_aegis_is_told_the_result_and_only_when_there_is_one():
    """The runner's single writer, both sides.

    A mark carrying a result reports it; a mark carrying none says nothing,
    rather than reporting a zero that would reset a streak.
    """

    class _Risk:
        def __init__(self) -> None:
            self.equities: list[float] = []
            self.results: list[float] = []

        def update_equity(self, value: float) -> None:
            self.equities.append(value)

        def record_trade_result(self, value: float) -> None:
            self.results.append(value)

    class _Mark:
        def __init__(self, equity, result):
            self.equity = equity
            self.cycle_result = result

    from chimera.demo.runner import DemoRunner

    class _Stand:
        def __init__(self) -> None:
            self.risk = _Risk()

    # The real method, on a stand-in holding only what it reads. `DemoRunner`
    # exposes `risk` as a read-only property, so the alternative would be a
    # whole runner, and this test is about the writer and nothing else.
    runner = _Stand()
    DemoRunner._tell_aegis(runner, _Mark(Decimal("10"), Decimal("-3")), report_cycle=True)
    DemoRunner._tell_aegis(runner, _Mark(Decimal("11"), None), report_cycle=True)
    # The intervention paths: a result present, and deliberately not reported.
    DemoRunner._tell_aegis(runner, _Mark(Decimal("12"), Decimal("-9")))

    assert runner.risk.equities == [10.0, 11.0, 12.0]
    assert runner.risk.results == [-3.0], "a mark with no result, or no leave to report, spoke"


def test_the_report_follows_the_ledger_save_at_every_call_site():
    """Ordering, asserted from the source because no single test reaches all six.

    The mark that produces a result also CLEARS the ledger's baseline for it, so
    a crash between the two must lose the report rather than repeat it: a
    cooldown opened on a trade that happened once is a halt built on a fiction.
    Every ``_tell_aegis`` call therefore sits below its branch's ``_save_ledger``.
    """
    source = (REPO / "chimera" / "demo" / "runner.py").read_text(encoding="utf-8")
    lines = source.splitlines()

    calls = [i for i, line in enumerate(lines) if "self._tell_aegis(" in line]
    definition = [i for i, line in enumerate(lines) if "def _tell_aegis(" in line]
    assert definition, "the single writer is gone"
    calls = [i for i in calls if i > definition[0] + 40]
    assert len(calls) >= 5, f"expected every mark site to report, found {len(calls)}"

    for index in calls:
        window = "\n".join(lines[max(0, index - 25) : index])
        assert (
            "_save_ledger()" in window
        ), f"the report at line {index + 1} does not follow a ledger save"


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
        "since R1-f the service's READY gate (DemoRunner.check_feed) reads it and "
        "halts the feed as FEED_STALLED; that gate is the demo's enforcement, and "
        "note_feed's mark is no longer fed by the demo at all"
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


def test_only_the_ordinary_tick_reports_a_cycle(tmp_path):
    """The load-bearing half: which path may tell Aegis a trade finished.

    ``consecutive_losses`` is a HASHED risk field. If a result were reported by
    whichever path happened to mark the position, a campaign that crashed and was
    repaired would reach a different ``risk.state_hash`` from the uninterrupted
    one at the same equity -- and the convergence of those two is precisely what
    R1-i's torn-settlement resolution establishes. Reporting only from the tick,
    the path a replay also runs, is what keeps the counter replayable.

    Asserted from the source because the alternative is to run every one of the
    six mark sites, and what the rule is about is which of them carries the flag.
    """
    source = (REPO / "chimera" / "demo" / "runner.py").read_text(encoding="utf-8")

    reporting = source.count("_tell_aegis(mark, report_cycle=True)")
    assert reporting == 1, f"exactly one path may report a cycle, found {reporting}"

    plain = source.count("self._tell_aegis(")
    assert plain - reporting >= 4, "the intervention paths no longer go through the writer"

    assert (
        "if report_cycle and mark.cycle_result is not None:" in source
    ), "the gate is gone; every marking path would report again"


def test_an_intervention_clears_the_baseline_without_reporting(tmp_path):
    """Both halves of what an operator flatten does, and does not, do.

    It must CLEAR the cycle baseline -- otherwise the next round trip is measured
    from a position that is long gone, which is the leak that made a profit
    report as a loss -- and it must tell Aegis nothing, because an operator's
    intervention is not the strategy losing money.
    """
    from chimera.futures.executor import FlattenCause

    from tests.demo_harness import build

    harness = build(tmp_path)
    first = harness.first_minute_ms()
    minute = first
    for index in range(6):
        minute = first + index * MINUTE_MS
        harness.tick(minute)
        if harness.runner.position.state is HedgeState.HEDGED:
            break
    assert harness.runner.position.state is HedgeState.HEDGED
    assert harness.runner.position.ledger.state.equity_at_open is not None
    before = harness.runner.risk.state.consecutive_losses

    state = harness.runner.cursor.state_for(minute, now_ns=harness.runner.clock.now_ns)
    harness.runner.position.install_quote(state)
    harness.runner.position.emergency_reduce(FlattenCause.RISK_HALT, state)
    mark = harness.runner.position.mark_to_market(state)

    assert harness.runner.position.state is HedgeState.FLAT
    assert mark.cycle_result is not None, "the close was not seen at all"
    assert (
        harness.runner.position.ledger.state.equity_at_open is None
    ), "baseline left standing"
    assert (
        harness.runner.risk.state.consecutive_losses == before
    ), "an intervention moved the strategy's loss streak"


def test_a_reported_loss_still_opens_the_cooldown_at_the_committed_limit():
    """And the gate still closes, on section 7.4's own ``loss_streak_limit``.

    Narrowing WHICH path reports does not narrow what a report does: three
    losses from the tick meet the committed limit and the next entry is refused.
    """
    clock = {"now": 1_000.0}
    engine = RiskEngine(
        RiskLimits(loss_streak_limit=3, cooldown_seconds=3_600.0),
        clock=lambda: clock["now"],
    )
    engine.update_equity(1_000_000.0)

    class _Stand:
        def __init__(self, risk) -> None:
            self.risk = risk

    class _Mark:
        def __init__(self, equity, result):
            self.equity = equity
            self.cycle_result = result

    from chimera.demo.runner import DemoRunner

    runner = _Stand(engine)
    for _ in range(3):
        DemoRunner._tell_aegis(
            runner, _Mark(Decimal("999000"), Decimal("-40")), report_cycle=True
        )

    assert engine.state.consecutive_losses == 3
    assert _entry(engine).allowed is False
    assert "cooldown" in _entry(engine).reason


def test_the_rate_the_veto_reads_is_in_the_hashed_inputs(tmp_path):
    """A decided input that the record omits is an input hashed away.

    The rule is the one stated beside ``mark_high`` in ``MarketState.to_dict``:
    a field belongs in the hashed inputs because it DECIDES something. Since
    R1-k this rate can veto an entry, so a decision minute that left it out
    would hash a set of inputs it did not actually decide on -- and two minutes
    differing only in the rate that refused one of them would hash alike.
    """
    from tests.demo_harness import build

    harness = build(tmp_path)
    minute = harness.first_minute_ms()
    state = harness.runner.cursor.state_for(minute, now_ns=harness.runner.clock.now_ns)

    inputs = state.canonical()

    assert "funding_rate_next" in inputs
    # And still beside, not instead of, the realised rate.
    assert "funding_rate_last" in inputs

    # The consequence, not just the key: two minutes differing only in the rate
    # that could refuse one of them must not hash alike.
    from dataclasses import replace

    from chimera.demo.runner import _inputs_hash

    moved = replace(state, funding_rate_next=(state.funding_rate_next or Decimal("0")) - 1)
    assert _inputs_hash(moved) != _inputs_hash(state)
