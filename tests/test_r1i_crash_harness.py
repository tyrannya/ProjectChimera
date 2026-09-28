"""R1-i: the state-machine crash harness, in CI, including a disk-full fault.

Every scenario below is one real transition of the demo runner -- a tick that
opens, the hourly reconciliation, a funding settlement, a liquidation touch, a
correction step and a correction flatten, an operator flatten, resume and
resolve, and a plain restart -- started from a valid persisted state. For every
persistence step the transition makes (``tests/r1i_crash.py`` lists them and
proves the list complete), the transition is killed there, in both variants,
and then:

1. a fresh runner is built from disk and started (a new process);
2. the documented operator paths are applied where it halts (``operate``);
3. the campaign continues to the same later minute as an uninterrupted
   reference run of the same transition;
4. the result must be the reference's, exactly -- positions, cash, fees,
   realised PnL, funding, settlements, open instant, both equities -- with a
   decision log that verifies and decides no minute twice;
5. a second restart must converge: it finds nothing new to recover.

A fault point whose recovery REFUSES is an explicit dispute. It is accepted
only where the refusal names a fact that genuinely no file holds, and each such
point is listed in the test that expects it; everything else must recover.

The disk-full sweep replaces the kill with ``OSError(ENOSPC)`` at each step and
requires the process-failure contract: no further write but the emergency
record, a :class:`~chimera.persistence.PersistenceFailure` out of the runner,
the trace in the reserve, and a restart that halts on it and recovers through
``resolve --emergency``.
"""

from __future__ import annotations

import errno
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Callable

import pytest

from chimera.demo import emergency
from chimera.demo.runner import DemoRunner, RunnerState
from chimera.persistence import PersistenceFailure

from demo_harness import DAY, build
from r1i_crash import (
    PRIMITIVES,
    Faults,
    Killed,
    World,
    check_log,
    consistency,
    copy_state,
    economics,
    excluded_minutes,
    instrumented,
    kill_at_the_note,
    operate,
    records,
    state_bytes,
)

D = Decimal
NOTE = "crash harness"


# ---------------------------------------------------------------------------
# the scenarios
# ---------------------------------------------------------------------------
@dataclass
class Scenario:
    name: str
    #: Builds the valid pre-transition state under ``state_dir``.
    prepare: Callable[[World, Path], None]
    #: The transition itself, on a fresh process's runner.
    transition: Callable[[DemoRunner], None]
    #: The minute (index) the campaign is continued to afterwards.
    continue_to: int
    #: The halt an uninterrupted run of this transition ends in, by reason
    #: prefix; recovery clears disputes and resumes any OTHER halt.
    ends_halted: str | None = None
    #: The operator command the transition is, which an operator whose command
    #: died runs again.
    command: str | None = None
    #: The world (recorder root) the scenario runs in.
    world: str = "plain"
    #: Exact recovery required (the R1-i blocker remediation's resolve sweeps):
    #: no kill point may pass by excluding a minute, Aegis's funding streak is
    #: compared with the reference's as each left it before its operator
    #: `resume`, every OPERATOR request must be completed or reported, and no
    #: settlement may be booked twice in the log.
    exact: bool = False
    #: The one exclusion an exact scenario may record, and only where it is
    #: true: a torn-funding re-book killed after its ledger save and before its
    #: completion record leaves the ledger's funding total ahead of every record
    #: the log holds, which section 9.3's triage reports as LOG_BEHIND_STATE and
    #: labels the resolve's pending minute (the index here). The economics must
    #: still equal the reference's exactly.
    rebook_log_behind_minute: int | None = None


def _process(world: World, state_dir: Path, minutes: int) -> None:
    """Decide minutes 0 .. minutes-1 in one process and shut down cleanly."""
    runner = world.runner(state_dir)
    runner.start()
    base = runner.cursor.next_minute_ms()
    runner.catch_up(now_ms=base + (minutes - 1) * 60_000)
    runner.shutdown("prepared")


def _tick_to(index: int) -> Callable[[DemoRunner], None]:
    def transition(runner: DemoRunner) -> None:
        runner.start()
        first = runner.cursor.next_minute_ms()
        target = runner.cursor.last_minute_processed
        minute = first if target is None else target + 60_000
        assert minute == first, "the transition decides exactly the next minute"
        runner.tick(minute)
        assert (minute - _BASE_MS[0]) // 60_000 == index

    return transition


_BASE_MS: list[int] = []
#: A fixture's own deliberate edit of the cash, by scenario (the touch drains it).
CASH_OFFSET: dict[str, Decimal] = {}


def _flatten(runner: DemoRunner) -> None:
    runner.start()
    runner.flatten(NOTE)


def _resume(runner: DemoRunner) -> None:
    runner.start()
    runner.resume(NOTE)


def _resolve_ledger(runner: DemoRunner) -> None:
    runner.start()
    runner.resolve_ledger("ledger_store_mismatch", NOTE)


def _resolve_emergency(runner: DemoRunner) -> None:
    runner.start()
    runner.resolve_emergency(NOTE)


def _restart(runner: DemoRunner) -> None:
    runner.start()
    runner.shutdown("restart transition")


def _prepare_minutes(count: int) -> Callable[[World, Path], None]:
    def prepare(world: World, state_dir: Path) -> None:
        _process(world, state_dir, count)

    return prepare


def _prepare_halted(world: World, state_dir: Path) -> None:
    """Ten minutes, then an ordinary kill-switch halt, then the switch removed."""
    _process(world, state_dir, 10)
    (state_dir / "KILL_SWITCH").write_text("drill\n", encoding="utf-8")
    runner = world.runner(state_dir)
    assert runner.start() is RunnerState.HALT
    (state_dir / "KILL_SWITCH").unlink()


def _prepare_ledger_mismatch(world: World, state_dir: Path) -> None:
    """The ORIGINAL crash window with no clearing path: the opening tick killed
    after the perpetual store's save, before the ledger's."""
    faults = Faults(kill_at=_step_before(world, "carry_ledger.json"), when="after")
    runner = world.runner(state_dir)
    with instrumented(faults), pytest.raises(Killed):
        _tick_to(0)(runner)
    halted = world.runner(state_dir)
    assert halted.start() is RunnerState.HALT
    assert (halted.position.ledger.disputed or "").startswith("ledger_store_mismatch")


def _step_before(world: World, role: str) -> int:
    """The step just before the opening tick's first write of ``role``: for the
    carry ledger, both legs' stores are saved and the ledger is not."""
    scratch = world.root.parent / "_step_probe"
    copy_state(world.root.parent / "_empty", scratch)
    faults = Faults()
    with instrumented(faults):
        _tick_to(0)(world.runner(scratch))
    return next(step.index for step in faults.steps if step.role == role) - 1


def _prepare_tripped(world: World, state_dir: Path) -> None:
    """Five minutes, then a disk-full failure on the next tick's Aegis write."""
    _process(world, state_dir, 5)
    runner = world.runner(state_dir)
    faults = Faults(fail_at=1, fail_errno=errno.ENOSPC)
    runner.start()
    with instrumented(faults) as watched, pytest.raises(PersistenceFailure) as failure:
        # The failure is injected at the first write of the next tick.
        watched.fail_at = len(watched.steps) + 1
        runner.tick(runner.cursor.last_minute_processed + 60_000)
    assert runner.record_persistence_failure(failure.value, command="run")


def _prepare_touch(world: World, state_dir: Path) -> None:
    """Three minutes, then the cash drained to just under section 6.7's line
    for minute 3 -- the funding-erosion mechanism, applied to the file."""
    _process(world, state_dir, 3)
    runner = world.runner(state_dir)
    runner.start()
    state = runner.cursor.state_for(runner.cursor.last_minute_processed + 60_000)
    position = runner.position
    adverse = state.mark_high or state.mark
    maintenance = (
        position.leg("perp").quantity * adverse * position.config.maintenance_margin_rate
    )
    equity = position.equity_at(state)
    drained = equity - maintenance + D("1")
    CASH_OFFSET["touch"] = -drained
    position.ledger.state.free_cash -= drained
    position.ledger.save()
    runner.shutdown("prepared")


def _prepare_partial(minutes: int) -> Callable[[World, Path], None]:
    def prepare(world: World, state_dir: Path) -> None:
        _process(world, state_dir, minutes)
        runner = world.runner(state_dir)
        runner.start()
        assert runner.position.state.value == "PARTIAL", runner.position.state
        runner.shutdown("prepared")

    return prepare


SCENARIOS = [
    Scenario("open", lambda w, s: None, _tick_to(0), 5),
    Scenario("reconcile", _prepare_minutes(60), _tick_to(60), 64),
    Scenario("funding", _prepare_minutes(479), _tick_to(479), 483),
    Scenario("flatten", _prepare_minutes(10), _flatten, 14, command="flatten"),
    Scenario("resume", _prepare_halted, _resume, 14, command="resume"),
    Scenario(
        "resolve_ledger",
        _prepare_ledger_mismatch,
        _resolve_ledger,
        4,
        command="resolve-ledger",
    ),
    Scenario(
        "resolve_emergency",
        _prepare_tripped,
        _resolve_emergency,
        9,
        command="resolve-emergency",
    ),
    Scenario("restart", _prepare_minutes(10), _restart, 13),
    Scenario("touch", _prepare_touch, _tick_to(3), 3, ends_halted="liquidation_touch"),
]

#: A funding settlement the carry position PAYS, in its own world.
PAID_RATE = "-0.0003"
FUNDING_PAID = Scenario("funding_paid", _prepare_minutes(479), _tick_to(479), 483)


# ---------------------------------------------------------------------------
# the R1-i blocker remediation: every operator resolve, killed at every step
# ---------------------------------------------------------------------------
#: The second settlement of the ``two_paid`` world (fixture settlements at
#: hours 1 and 2, both paid): its minute's close is 02:00.
SECOND_SETTLEMENT = 119
#: Aegis's funding guard after the second settlement in an UNINTERRUPTED run,
#: by scenario: the value every recovered run must hold before its resume.
CLEAN_STREAK: dict[str, tuple[int, bool]] = {}


def _resolve_selector(selector: str) -> Callable[[DemoRunner], None]:
    def transition(runner: DemoRunner) -> None:
        runner.start()
        if selector.startswith("--ledger "):
            runner.resolve_ledger(selector.split(" ", 1)[1], NOTE)
        elif selector == "--symbol":
            runner.resolve(runner.position.config.perp_symbol, NOTE)
        else:
            assert selector == "--equity"
            runner.resolve_equity(NOTE)

    return transition


def _prepare_torn(origin: str, when: str) -> Callable[[World, Path], None]:
    """The second settlement torn: killed ``before`` Aegis noted it (MISSED) or
    ``after`` its note persisted (COUNTED), with the log's newest risk
    statement a restated hash (``plain``) or an operator RESUME (``resumed``)."""

    def prepare(world: World, state_dir: Path) -> None:
        _process(world, state_dir, SECOND_SETTLEMENT)
        if origin == "resumed":
            (state_dir / "KILL_SWITCH").write_text("drill\n", encoding="utf-8")
            halted = world.runner(state_dir)
            assert halted.start() is RunnerState.HALT
            halted.shutdown("halted")
            (state_dir / "KILL_SWITCH").unlink()
            operator = world.runner(state_dir)
            operator.start()
            operator.resume(NOTE)
            operator.shutdown("resumed")
        name = f"resolve_funding_{'counted' if when == 'after' else 'missed'}_{origin}"
        clean = copy_state(state_dir, state_dir.parent / f"clean_{name}")
        runner = world.runner(clean)
        runner.start()
        runner.tick(_BASE_MS[0] + SECOND_SETTLEMENT * 60_000)
        assert runner.state is not RunnerState.HALT, runner.halt_reason
        CLEAN_STREAK[name] = (
            runner.risk.state.funding_adverse_streak,
            runner.risk.state.funding_halt,
        )
        runner.shutdown("clean")
        faults = Faults()
        runner = world.runner(state_dir)
        with instrumented(faults), kill_at_the_note(faults, when), pytest.raises(Killed):
            runner.start()
            runner.tick(_BASE_MS[0] + SECOND_SETTLEMENT * 60_000)
        halted = world.runner(state_dir)
        assert halted.start() is RunnerState.HALT
        assert (halted.position.ledger.disputed or "").startswith("funding_booking_torn")
        halted.shutdown("torn")

    return prepare


def _prepare_symbol_dispute(world: World, state_dir: Path) -> None:
    """A REAL reconciliation mismatch: the dry-run venue reports a different
    perpetual quantity at one hourly reconciliation (minute 60)."""
    _process(world, state_dir, 60)
    runner = world.runner(state_dir)
    runner.start()
    perp = runner.position.perp
    symbol = runner.position.config.perp_symbol
    real = perp.venue.reported_position

    def once(sym):
        held = real(sym)
        if sym != symbol:
            return held
        perp.venue.reported_position = real
        return type(held)(
            symbol=held.symbol,
            side=held.side,
            quantity=held.quantity + D("0.001"),
            entry_price=held.entry_price,
            leverage=held.leverage,
            margin_mode=held.margin_mode,
        )

    perp.venue.reported_position = once
    runner.tick(_BASE_MS[0] + 60 * 60_000)
    assert runner.state is RunnerState.HALT
    runner.shutdown("halted on the dispute")
    assert f"--symbol {symbol}" in [s for s, _ in _started(world, state_dir)]


def _prepare_equity_dispute(world: World, state_dir: Path) -> None:
    """R1-b's window: a tick killed between the ledger's save and Aegis's
    ``update_equity``, so the two persisted equities disagree. On the first
    settlement's minute (59), whose booking moves the equity (the fixture's
    price path alone does not, this early)."""
    _process(world, state_dir, 59)
    runner = world.runner(state_dir)
    runner.start()
    faults = Faults()

    def killed(*_args, **_kwargs):
        faults.killed = True
        raise Killed("killed before update_equity")

    runner.risk.update_equity = killed  # type: ignore[method-assign]
    with instrumented(faults), pytest.raises(Killed):
        runner.tick(_BASE_MS[0] + 59 * 60_000)
    assert "--equity" in [s for s, _ in _started(world, state_dir)]


def _prepare_stale_leg(world: World, state_dir: Path) -> None:
    """A stale leg: one leg last observed five minutes before the other."""
    _process(world, state_dir, 10)
    ledger = world.runner(state_dir)
    ledger.start()
    carry = ledger.position.ledger
    marks = dict(carry.state.marked_at_ns)
    carry.state.marked_at_ns["perp"] = marks["spot"] - 5 * 60_000_000_000
    carry.save()
    ledger.shutdown("stale")
    assert "--ledger stale_leg" in [s for s, _ in _started(world, state_dir)]


def _prepare_asymmetric_close(world: World, state_dir: Path) -> None:
    """A close only the perpetual leg makes (the spot fill is refused), then the
    runbook's `flatten`: the legs are level and ``asymmetric_close`` stands."""
    from chimera.demo.rules import HedgeTarget, RuleDecision, RuleRegistry

    class Closer:
        rule_id = "R_closer"
        version = "1.0.0"
        actionable = True

        def rule_hash(self):
            return "sha256:" + "e" * 64

        def params_hash(self):
            return "sha256:" + "f" * 64

        def evaluate(self, state, portfolio):
            return RuleDecision(
                rule_id=self.rule_id,
                rule_hash=self.rule_hash(),
                target=HedgeTarget(D("0")),
                reason="synthetic close",
                inputs_hash="sha256:" + "a" * 64,
                params_hash=self.params_hash(),
            )

    _process(world, state_dir, 2)
    runner = world.runner(state_dir)
    runner.start()
    runner.rules = RuleRegistry([Closer()])
    runner.position.fill_models["spot"].max_reference_deviation_bps = D("0")
    runner.tick(_BASE_MS[0] + 2 * 60_000)
    assert runner.state is RunnerState.HALT
    runner.shutdown("asymmetric")
    flattening = world.runner(state_dir)
    flattening.start()
    assert flattening.position.imbalance()
    flattening.flatten(NOTE)
    flattening.shutdown("flattened")
    again = world.runner(state_dir)
    again.start()
    assert not again.position.imbalance()
    assert (again.position.ledger.disputed or "").startswith("asymmetric_close")
    again.shutdown("level")


def _started(world: World, state_dir: Path) -> list[tuple[str, str]]:
    runner = world.runner(state_dir)
    runner.start()
    disputes = runner.standing_disputes()
    runner.shutdown("looked")
    return disputes


RESOLVE_SCENARIOS = [
    *(
        Scenario(
            f"resolve_funding_{'counted' if when == 'after' else 'missed'}_{origin}",
            _prepare_torn(origin, when),
            _resolve_selector("--ledger funding_booking_torn"),
            SECOND_SETTLEMENT + 4,
            command="resolve-ledger",
            world="two_paid",
            exact=True,
            rebook_log_behind_minute=SECOND_SETTLEMENT,
        )
        for origin in ("resumed", "plain")
        for when in ("after", "before")
    ),
    Scenario(
        "resolve_symbol",
        _prepare_symbol_dispute,
        _resolve_selector("--symbol"),
        64,
        command="resolve",
        exact=True,
    ),
    Scenario(
        "resolve_equity",
        _prepare_equity_dispute,
        _resolve_selector("--equity"),
        63,
        command="resolve-equity",
        world="two_paid",
        exact=True,
    ),
    Scenario(
        "resolve_stale_leg",
        _prepare_stale_leg,
        _resolve_selector("--ledger stale_leg"),
        14,
        command="resolve-ledger",
        exact=True,
    ),
    Scenario(
        "resolve_asymmetric_close",
        _prepare_asymmetric_close,
        _resolve_selector("--ledger asymmetric_close"),
        6,
        command="resolve-ledger",
        exact=True,
    ),
]

#: Worlds whose spot leg refuses every fill: a PARTIAL that corrects, and one
#: that times out and is flattened (section 6.3).
REFUSING = {"max_reference_deviation_bps": "0"}
CORRECTION_SCENARIOS = [
    Scenario("correction_step", _prepare_partial(1), _tick_to(1), 6),
    Scenario("correction_timeout", _prepare_partial(4), _tick_to(4), 8),
]


# ---------------------------------------------------------------------------
# the machinery
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def worlds(tmp_path_factory):
    """One recorder root per world, written once; state is copied per run."""
    made = {}
    for name, spot in (
        ("plain", None),
        ("refusing", REFUSING),
        ("paying", None),
        ("two_paid", None),
    ):
        base = tmp_path_factory.mktemp(f"world_{name}")
        harness = build(base, start=False)
        if name == "paying":
            # The 08:00 settlement is PAID by the short perpetual: the one
            # direction that moves Aegis's funding streak (a rebate on a
            # streak already at zero moves nothing).
            harness.feed.write_settlements([DAY], rates={(DAY, 8): PAID_RATE})
        if name == "two_paid":
            # Two paid settlements, at hours 1 and 2 (fixture data): room for
            # a halt and a RESUME between them, so the log's newest risk
            # statement before the second is one that restates no hash.
            harness.feed.write_settlements(
                [DAY], hours=(1, 2), rates={(DAY, 1): PAID_RATE, (DAY, 2): PAID_RATE}
            )
        world = World(root=harness.root)
        world.spot_model = spot  # type: ignore[attr-defined]
        (base / "_empty").mkdir()
        _BASE_MS[:] = [harness.first_minute_ms()]
        made[name] = world
    return made


def _patch_world(world: World) -> World:
    spot = getattr(world, "spot_model", None)
    if spot is None:
        return world
    original = world.runner

    def runner(state_dir: Path) -> DemoRunner:
        built = original(state_dir)
        factory = built._position_factory

        def refusing(risk, clock):
            position = factory(risk, clock)
            for key, value in spot.items():
                setattr(position.fill_models["spot"], key, D(value))
            return position

        built._position_factory = refusing
        return built

    world.runner = runner  # type: ignore[method-assign]
    return world


def _base(world: World, scenario: Scenario, tmp: Path) -> Path:
    base = tmp / f"base_{scenario.name}"
    base.mkdir()
    scenario.prepare(world, base)
    return base


def _finish(world: World, state_dir: Path, scenario: Scenario):
    """Continue the campaign to the scenario's minute, then restart once more."""
    runner = world.runner(state_dir)
    state = runner.start()
    if not scenario.ends_halted:
        assert state in (RunnerState.READY, RunnerState.FEED_STALLED), runner.halt_reason
    if state in (RunnerState.READY, RunnerState.FEED_STALLED):
        runner.catch_up(now_ms=_BASE_MS[0] + scenario.continue_to * 60_000)
    result = economics(runner)
    problems = consistency(runner, cash_offset=CASH_OFFSET.get(scenario.name, D("0")))
    assert not problems, problems
    final_state = runner.state
    runner.shutdown("finished")
    before = len(records(state_dir))
    again = world.runner(state_dir)
    again_state = again.start()
    kinds = [r["kind"] for r in records(state_dir)[before:]]
    assert "RECOVERY" not in kinds, "a second restart found something new to recover"
    assert economics(again) == result, "a second restart does not converge"
    assert again_state is final_state or scenario.ends_halted
    again.shutdown("converged")
    return result, final_state


def _operate(world: World, state_dir: Path, scenario: Scenario, since: int = 0):
    return operate(
        world,
        state_dir,
        retry=scenario.command,
        since=since,
        keep_halt=scenario.ends_halted,
    )


def _reference(world: World, scenario: Scenario, base: Path, tmp: Path):
    state_dir = copy_state(base, tmp / f"reference_{scenario.name}")
    faults = Faults()
    with instrumented(faults):
        scenario.transition(world.runner(state_dir))
    assert not faults.uninventoried
    operated = None
    if not scenario.ends_halted and scenario.command is not None:
        operated = operate(world, state_dir)
        assert not operated.refused, operated.refused
    result, final = _finish(world, state_dir, scenario)
    if scenario.exact and operated is not None and operated.streak_before_resume is not None:
        # The reference's own streak, as its resolve left it, before its resume
        # (a resolve that leaves nothing halted -- `--equity` -- has none, and
        # its final streak is compared as it is).
        REFERENCE_STREAK[scenario.name] = operated.streak_before_resume[:2]
    if scenario.exact and scenario.name in CLEAN_STREAK:
        # A torn settlement's uninterrupted resolve must leave what an untorn
        # settlement leaves: the reference is itself held to the clean run.
        assert REFERENCE_STREAK.get(scenario.name) == CLEAN_STREAK[scenario.name], (
            f"the uninterrupted resolve leaves {REFERENCE_STREAK.get(scenario.name)}; an "
            f"untorn settlement leaves {CLEAN_STREAK[scenario.name]}"
        )
    return faults.steps, result, final


#: Aegis's funding guard as each exact scenario's reference left it before
#: its operator resume.
REFERENCE_STREAK: dict[str, tuple[int, bool]] = {}


#: Fault points whose recovery excluded a minute (reported, not hidden).
EXCLUDED: list[str] = []


def _crash_points(steps) -> list[tuple[int, str]]:
    return [(step.index, when) for step in steps for when in ("mid", "after")]


def _run_crash(world, scenario, base, tmp, index, when):
    state_dir = copy_state(base, tmp / f"{scenario.name}_{when}_{index}")
    since = len(records(state_dir))
    faults = Faults(kill_at=index, when=when)
    with instrumented(faults):
        try:
            scenario.transition(world.runner(state_dir))
        except Killed:
            pass
    assert faults.killed, f"step {index} was never reached"
    assert not faults.uninventoried
    operated = _operate(world, state_dir, scenario, since)
    if operated.refused:
        return state_dir, None, operated
    result, final = _finish(world, state_dir, scenario)
    return state_dir, (result, final), operated


def _differences(
    reference: dict, result: dict, operated, reference_streak: tuple[int, bool] | None = None
) -> list[str]:
    """Where a recovered run's economics differ from the uninterrupted one's.

    Aegis's funding streak is compared as the recovery LEFT it: an operator
    ``resume`` clears it by design (``RiskEngine.resume``), so when the resume
    came after every settlement was booked the streak compared is the one just
    before it -- which is what shows a torn settlement counted twice, or not at
    all. A resume before a settlement is booked resets nothing the reference
    holds, and the final streak is compared as it is.
    """
    reference, result = dict(reference), dict(result)
    problems = []
    before = operated.streak_before_resume
    if reference_streak is not None:
        # Both runs resumed after their resolve: each is compared as it stood
        # just before its resume, which is what shows a settlement counted twice.
        if before is None or before[:2] != reference_streak:
            problems.append(
                f"Aegis's funding streak before the resume is "
                f"{None if before is None else before[:2]} where the reference holds "
                f"{reference_streak}"
            )
    elif before is not None and before[2] == len(reference["settled"]):
        expected = reference.pop("funding_streak")
        result.pop("funding_streak")
        if before[:2] != expected:
            problems.append(
                f"Aegis's funding streak before the resume is {before[:2]}"
                f" where the reference holds {expected}"
            )
    diff = {k: (reference[k], result[k]) for k in reference if reference[k] != result[k]}
    if diff:
        problems.append(f"economics differ from the reference: {diff}")
    return problems


def _sweep(world: World, scenario: Scenario, tmp: Path):
    base = _base(world, scenario, tmp)
    steps, reference, final = _reference(world, scenario, base, tmp)
    assert steps, "the transition made no persistence step"
    failures = []
    since = len(records(base))
    for index, when in _crash_points(steps):
        try:
            state_dir, outcome, operated = _run_crash(world, scenario, base, tmp, index, when)
        except AssertionError as exc:
            failures.append(f"{scenario.name}: kill {when} step {index}: {exc}")
            continue
        label = f"{scenario.name}: kill {when} step {index} ({steps[index - 1].primitive}:{steps[index - 1].role})"
        problems = check_log(state_dir)
        if outcome is None:
            failures.append(f"{label}: recovery refused: {operated.refused}")
            continue
        result, crash_final = outcome
        if scenario.exact:
            failures.extend(
                f"{label}: {p}"
                for p in _exact(
                    state_dir, since, _permitted_exclusion(scenario, steps, index, when)
                )
            )
            failures.extend(
                f"{label}: {p}"
                for p in _differences(
                    reference, result, operated, REFERENCE_STREAK.get(scenario.name)
                )
            )
        elif excluded_minutes(state_dir, since):
            # Section 9.3's deliberate exclusion: a minute a crash left half
            # recorded is excluded and not decided again, so the next minute
            # sizes from the mark before it. Consistent (checked in `_finish`)
            # and recorded; not required to equal the uninterrupted run.
            EXCLUDED.append(label)
        else:
            failures.extend(f"{label}: {p}" for p in _differences(reference, result, operated))
        if crash_final is not final:
            failures.append(f"{label}: ends {crash_final} where the reference ends {final}")
        failures.extend(f"{label}: {problem}" for problem in problems)
    return steps, failures


def _permitted_exclusion(scenario: Scenario, steps, index: int, when: str) -> str | None:
    """The one minute an exact scenario's kill point may see excluded, or None.

    Only a torn-funding re-book's window: killed after the resolve's own ledger
    save (the first ``carry_ledger.save`` after its ``requested`` record, which
    is the first log append after ``start()``'s ``runner.save_state``) and
    before its completion (the transition's last log append).
    """
    if scenario.rebook_log_behind_minute is None:
        return None
    kinds = [step.primitive for step in steps]
    started = kinds.index("runner.save_state") + 1
    requested = kinds.index("decision_log.append", started) + 1
    saved = kinds.index("carry_ledger.save", requested) + 1
    completion = len(kinds) - kinds[::-1].index("decision_log.append")
    if (index == saved and when == "after") or saved < index < completion:
        minute_ms = _BASE_MS[0] + scenario.rebook_log_behind_minute * 60_000
        return datetime.fromtimestamp(minute_ms / 1000, tz=timezone.utc).isoformat()
    return None


def _exact(state_dir: Path, since: int, permitted: str | None = None) -> list[str]:
    """What an exact scenario requires beyond the reference's economics."""
    problems: list[str] = []
    for record in records(state_dir)[since:]:
        recovery = record.get("recovery")
        if not isinstance(recovery, dict) or not recovery.get("evidence_excluded_minute"):
            continue
        minute = recovery["evidence_excluded_minute"]
        if permitted is None or minute != permitted or recovery["cause"] != "LOG_BEHIND_STATE":
            problems.append(
                f"recovered only by excluding {minute} ({recovery['cause']}); exact "
                f"recovery is required (permitted here: {permitted or 'none'})"
            )
    requested: set[int] = set()
    closed: set[int] = set()
    booked: dict[int, int] = {}
    for record in records(state_dir):
        operator = record.get("operator")
        if isinstance(operator, dict):
            if operator.get("phase") == "requested":
                requested.add(int(record["seq"]))
            elif isinstance(operator.get("request_seq"), int):
                if operator["request_seq"] not in requested:
                    problems.append(f"seq {record['seq']} completes an unknown request")
                closed.add(int(operator["request_seq"]))
            rebook = operator.get("rebook")
            if (
                operator.get("phase") == "completed"
                and operator.get("kind") == "funding_booking_torn"
                and isinstance(rebook, dict)
            ):
                instant = int(rebook["instant_ns"])
                booked[instant] = booked.get(instant, 0) + 1
        recovery = record.get("recovery")
        if isinstance(recovery, dict) and recovery.get("cause") == "OPERATOR_INCOMPLETE":
            closed.add(int(recovery["request_seq"]))
        funding = record.get("funding")
        if record.get("kind") == "FUNDING" and isinstance(funding, dict):
            instant = int(funding["settlement_instant_ns"])
            booked[instant] = booked.get(instant, 0) + 1
    if requested - closed:
        problems.append(
            f"OPERATOR requests neither completed nor reported: {requested - closed}"
        )
    twice = {instant: n for instant, n in booked.items() if n > 1}
    if twice:
        problems.append(f"settlements booked more than once in the log: {twice}")
    return problems


# ---------------------------------------------------------------------------
# the harness validates itself
# ---------------------------------------------------------------------------
def test_the_inventory_names_every_writer_the_runner_reaches(worlds, tmp_path):
    """Every primitive in the inventory is exercised by some scenario, and no
    scenario makes a durable write outside it (checked inside every sweep)."""
    seen: set[str] = set()
    for scenario in SCENARIOS:
        base = _base(worlds["plain"], scenario, tmp_path)
        faults = Faults()
        state_dir = copy_state(base, tmp_path / f"inv_{scenario.name}")
        with instrumented(faults):
            scenario.transition(worlds["plain"].runner(state_dir))
        assert not faults.uninventoried, scenario.name
        seen.update(step.primitive for step in faults.steps)
    # `decision_log.recover_tail` (a torn tail) and `emergency.trip` (a failure)
    # are reached by their own tests below; everything else by a transition.
    assert seen >= set(PRIMITIVES) - {"decision_log.recover_tail", "emergency.trip"}, (
        set(PRIMITIVES) - seen
    )


def test_an_unlisted_writer_is_caught_by_the_harness(worlds, tmp_path):
    """Negative control: remove one primitive from the inventory and the fsync
    guard reports the writes it made as uninventoried."""
    import r1i_crash

    saved = dict(r1i_crash.PRIMITIVES)
    try:
        del r1i_crash.PRIMITIVES["carry_ledger.save"]
        state_dir = tmp_path / "unlisted"
        state_dir.mkdir()
        faults = Faults()
        with instrumented(faults), pytest.raises(Exception):
            _tick_to(0)(worlds["plain"].runner(state_dir))
        assert faults.uninventoried
    finally:
        r1i_crash.PRIMITIVES.clear()
        r1i_crash.PRIMITIVES.update(saved)


def test_a_no_op_fault_changes_nothing(worlds, tmp_path):
    """Control: instrumented but never killed, the transition writes exactly the
    bytes an uninstrumented one writes."""
    world = worlds["plain"]
    plain = tmp_path / "plain"
    watched = tmp_path / "watched"
    plain.mkdir()
    watched.mkdir()
    _tick_to(0)(world.runner(plain))
    with instrumented(Faults()):
        _tick_to(0)(world.runner(watched))
    assert state_bytes(plain) == state_bytes(watched)


# ---------------------------------------------------------------------------
# kill at every persistence step, recover, assert consistency
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
def test_every_kill_point_recovers_to_the_reference(worlds, tmp_path, scenario):
    steps, failures = _sweep(worlds["plain"], scenario, tmp_path)
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("scenario", CORRECTION_SCENARIOS, ids=lambda s: s.name)
def test_every_kill_point_of_a_correction_recovers(worlds, tmp_path, scenario):
    world = _patch_world(worlds["refusing"])
    steps, failures = _sweep(world, scenario, tmp_path)
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("scenario", RESOLVE_SCENARIOS, ids=lambda s: s.name)
def test_every_kill_point_of_every_resolve_recovers_exactly(worlds, tmp_path, scenario):
    """The R1-i blocker remediation's sweep: each successful operator resolve,
    killed at every persistence step it makes (``start()``'s included), in both
    variants, then restarted and driven through the documented operator paths
    -- the interrupted resolve run again -- must reach its uninterrupted run
    exactly: no excluded minute, the same economics, Aegis's funding streak as
    the uninterrupted resolve (and, for a torn settlement, an untorn one) left
    it, every OPERATOR request completed or reported, no settlement booked
    twice, and a second restart that finds nothing new."""
    steps, failures = _sweep(worlds[scenario.world], scenario, tmp_path)
    primitives = {step.primitive for step in steps}
    assert {"decision_log.append", "runner.save_state"} <= primitives, primitives
    if scenario.command == "resolve-ledger" and scenario.name != "resolve_stale_leg":
        assert "carry_ledger.save" in primitives, primitives
    assert not failures, "\n".join(failures)


def test_every_kill_point_of_a_paid_settlement_recovers(worlds, tmp_path):
    """The funding window with a PAID settlement: a kill after Aegis counted it
    and before the ledger booked it is proved by R1-c's ``funding`` window, and
    ``resolve --ledger funding_booking_torn`` books it without counting it
    again (the reference comparison includes Aegis's funding streak)."""
    steps, failures = _sweep(worlds["paying"], FUNDING_PAID, tmp_path)
    assert not failures, "\n".join(failures)


# ---------------------------------------------------------------------------
# disk full at every persistence step
# ---------------------------------------------------------------------------
DISK_SCENARIOS = [
    s for s in SCENARIOS if s.name in ("open", "funding", "flatten", "resume", "touch")
]


@pytest.mark.parametrize("scenario", DISK_SCENARIOS + [FUNDING_PAID], ids=lambda s: s.name)
def test_disk_full_at_every_step_is_a_process_failure_with_a_trace(worlds, tmp_path, scenario):
    world = worlds["paying" if scenario is FUNDING_PAID else "plain"]
    base = _base(world, scenario, tmp_path)
    steps, reference, final = _reference(world, scenario, base, tmp_path)
    failures = []
    since = len(records(base))
    for step in steps:
        state_dir = copy_state(base, tmp_path / f"enospc_{scenario.name}_{step.index}")
        faults = Faults(fail_at=step.index, fail_errno=errno.ENOSPC)
        runner = world.runner(state_dir)
        caught = None
        with instrumented(faults):
            try:
                scenario.transition(runner)
            except PersistenceFailure as exc:
                caught = exc
                # The CLI's handler: the reserve, and nothing else.
                written = runner.record_persistence_failure(exc, command=scenario.name)
        label = f"{scenario.name}: ENOSPC at step {step.index} ({step.primitive}:{step.role})"
        if caught is None:
            failures.append(f"{label}: no PersistenceFailure reached the caller")
            continue
        if caught.errno != "ENOSPC":
            failures.append(f"{label}: failure names {caught.errno}")
        if faults.writes_after_failure:
            failures.append(f"{label}: wrote after the failure: {faults.writes_after_failure}")
        if step.primitive == "emergency.arm":
            # The reserve itself could not be created: nothing has been
            # processed, so there is nothing to trace -- and the next start,
            # with space again, arms it and proceeds.
            left = set(state_bytes(state_dir)) - set(state_bytes(base))
            if written or left - {emergency.EMERGENCY_NAME + ".tmp"}:
                failures.append(
                    f"{label}: a failed arm left {sorted(left)} (written={written})"
                )
            continue
        if not written:
            failures.append(f"{label}: the emergency record was not written")
            continue
        trace = emergency.read_trace(state_dir)
        if trace.state is not emergency.EmergencyState.TRIPPED:
            failures.append(f"{label}: trace is {trace.state}")
        restarted = world.runner(state_dir)
        if restarted.start() is not RunnerState.HALT or "persistence_failure" not in (
            restarted.halt_reason or ""
        ):
            failures.append(f"{label}: the restart did not halt on the trace")
        restarted.shutdown("seen")
        operated = _operate(world, state_dir, scenario, since)
        if operated.refused:
            failures.append(f"{label}: recovery refused: {operated.refused}")
            continue
        if not operated.actions or operated.actions[0] != "resolve --emergency":
            failures.append(f"{label}: the trace was not the first thing cleared: {operated}")
        result, crash_final = _finish(world, state_dir, scenario)
        failures.extend(f"{label}: {p}" for p in _differences(reference, result, operated))
        failures.extend(f"{label}: {p}" for p in check_log(state_dir))
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize(
    "code", [errno.EACCES, errno.EIO, errno.EROFS], ids=["EACCES", "EIO", "EROFS"]
)
@pytest.mark.parametrize("where", ["fsync", "replace"])
def test_any_persistence_failure_not_only_enospc(worlds, tmp_path, code, where):
    """The contract is per failure, not per errno: permission denied, an I/O
    error and a read-only filesystem, at the fsync or at the rename."""
    world = worlds["plain"]
    base = _base(world, SCENARIOS[1], tmp_path)
    faults = Faults()
    with instrumented(faults):
        SCENARIOS[1].transition(world.runner(copy_state(base, tmp_path / "count")))
    wanted = next(
        s.index
        for s in faults.steps
        if (where == "fsync" or s.primitive != "decision_log.append")
        and s.primitive != "emergency.arm"
    )
    state_dir = copy_state(base, tmp_path / "failing")
    runner = world.runner(state_dir)
    failing = Faults(fail_at=wanted, fail_errno=code, fail_in=where)
    with instrumented(failing), pytest.raises(PersistenceFailure) as failure:
        SCENARIOS[1].transition(runner)
    assert failure.value.errno == errno.errorcode[code]
    assert not failing.writes_after_failure
    assert runner.record_persistence_failure(failure.value)


# ---------------------------------------------------------------------------
# the harness against a real process death
# ---------------------------------------------------------------------------
_CHILD = r"""
import os, sys
sys.path.insert(0, {tests!r})
from pathlib import Path
import r1i_crash
from r1i_crash import Faults, World, instrumented
from test_r1i_crash_harness import _tick_to, _BASE_MS

_BASE_MS[:] = [{base_ms}]
world = World(root=Path({root!r}))
faults = Faults(kill_at={index}, when={when!r})
original = r1i_crash.Killed

class Exit(BaseException):
    pass

# A real death: no unwinding, no finally, no atexit.
def die(*_args, **_kwargs):
    os._exit(137)

r1i_crash.Killed = type("Die", (BaseException,), {{"__init__": die}})
with instrumented(faults):
    _tick_to(0)(world.runner(Path({state!r})))
os._exit(0)
"""


@pytest.mark.parametrize("when", ["mid", "after"])
def test_an_in_process_kill_leaves_the_bytes_a_real_process_death_leaves(
    worlds, tmp_path, when
):
    """The harness's equivalence claim, checked: a subprocess that dies with
    ``os._exit`` at a step leaves byte-identical state to the in-process kill
    at the same step."""
    world = worlds["plain"]
    counted = Faults()
    probe = tmp_path / "probe"
    probe.mkdir()
    with instrumented(counted):
        _tick_to(0)(world.runner(probe))
    tests = str(Path(__file__).resolve().parent)
    for index in (1, len(counted.steps) // 2, len(counted.steps)):
        in_process = tmp_path / f"in_{when}_{index}"
        in_process.mkdir()
        faults = Faults(kill_at=index, when=when)
        with instrumented(faults):
            with pytest.raises(Killed):
                _tick_to(0)(world.runner(in_process))
        child_dir = tmp_path / f"child_{when}_{index}"
        child_dir.mkdir()
        script = _CHILD.format(
            tests=tests,
            base_ms=_BASE_MS[0],
            root=str(world.root),
            index=index,
            when=when,
            state=str(child_dir),
        )
        completed = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=300
        )
        assert completed.returncode == 137, completed.stderr[-2000:]
        assert state_bytes(child_dir) == state_bytes(in_process), f"step {index} {when}"
