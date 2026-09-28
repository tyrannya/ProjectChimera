"""R1-i's state-machine crash harness: kill at every persistence step, recover,
assert consistency.

The roadmap's words are "the state-machine crash harness (kill at every
persistence step, recover, assert consistency) added to CI, including a
disk-full fault". This module is the harness; the tests that drive it are
``tests/test_r1i_crash_harness.py``.

**The inventory.** :data:`PRIMITIVES` names every function through which the
demo runner makes a write durable. The harness does not trust that list: while
a transition runs, ``os.fsync`` is watched, and an fsync made OUTSIDE every
listed primitive fails the run as an uninventoried persistence step. So a
writer added later without being listed here is found by the harness itself,
not by a reviewer.

**What a step is.** One call of a listed primitive that performed at least one
``fsync``. A call that wrote nothing (a sealed Aegis refusing to persist, a
ledger that may not speak) is not a step: killing "after" it is killing after
the step before.

**Why an in-process kill is a process death at the durability boundary.** Each
fault point raises :class:`Killed`, a ``BaseException`` that every ``except
Exception`` on the demo path lets through, either at the first ``fsync`` inside
step *k* (``mid``: the bytes are written and not yet durable or renamed into
place) or as step *k* returns (``after``: it is durable). From that instant the
harness refuses EVERY further write -- any primitive entry and any ``fsync``
raises :class:`Killed` again -- so ``finally`` blocks unwinding through the
runner cannot write anything a dead process could not have written. What is on
disk is exactly what a process killed at that instant leaves; the page cache
survives a process death, so a written-and-flushed but unsynced line is on disk
too, which the ``mid`` variant reproduces. ``tests/test_r1i_crash_harness.py``
checks this equivalence against a real subprocess killed with ``os._exit`` at the
same step, byte for byte.

**Recovery.** A fresh runner is built from disk -- a new process -- and started.
If it halts, :func:`operate` plays the operator the runbook describes: one
command per process, each through ``start()``; a standing dispute is cleared by
its own ``resolve`` and only then is a halt resumed. A ``resolve`` that REFUSES
ends the operator's work: the state is an explicit dispute with its evidence,
which the caller judges.
"""

from __future__ import annotations

import contextlib
import errno as _errno
import os
import re
import shutil
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

from chimera.carry.factory import build_hedged_position
from chimera.carry.ledger import CarryLedger
from chimera.demo import emergency as emergency_module
from chimera.demo import runner as runner_module
from chimera.demo.decision_log import DecisionLog, read_records, verify_log
from chimera.demo.inspection import inspect_demo_state
from chimera.demo.risk_wiring import build_risk_engine
from chimera.demo.rules import RuleRegistry
from chimera.demo.rules_carry import CarryParams, CarryRule
from chimera.demo.runner import DemoRunner, RunnerError, RunnerState
from chimera.demo.telemetry import NullTelemetry
from chimera.futures.store import FuturesStore
from chimera.recorder.contract import load_recorder_contract
from chimera.risk import RiskEngine

from demo_harness import CAPITAL, campaign_config

#: Every durable writer the demo runner reaches, by name. See the module
#: docstring: the harness also PROVES this list complete for each transition.
PRIMITIVES: Mapping[str, tuple[Any, str]] = {
    "futures_store.save": (FuturesStore, "save"),
    "carry_ledger.save": (CarryLedger, "save"),
    "risk.persist": (RiskEngine, "_persist"),
    "decision_log.append": (DecisionLog, "append"),
    "decision_log.close": (DecisionLog, "close"),
    "decision_log.recover_tail": (runner_module, "recover_tail"),
    "runner.save_state": (DemoRunner, "save_state"),
    "emergency.arm": (emergency_module, "arm"),
    "emergency.trip": (emergency_module, "trip"),
    "emergency.rearm": (emergency_module, "rearm"),
}


class Killed(BaseException):
    """The process died here. Never caught by the code under test."""


class UninventoriedWrite(AssertionError):
    """An ``fsync`` outside every listed primitive: a persistence step the
    harness does not know about."""


@dataclass
class Step:
    index: int
    primitive: str
    role: str


@dataclass
class Faults:
    """What to do to the persistence steps of one transition.

    ``kill_at`` is the 1-based step to kill at and ``when`` is ``"mid"`` (at its
    first fsync) or ``"after"`` (as it returns). ``fail_at`` instead makes that
    step's first fsync raise ``OSError(fail_errno)`` -- a disk-full or other
    persistence failure -- and records every write attempted afterwards, which
    must be none but the emergency record's.
    """

    kill_at: int | None = None
    when: str = "after"
    fail_at: int | None = None
    fail_errno: int = _errno.ENOSPC
    fail_in: str = "fsync"
    steps: list[Step] = field(default_factory=list)
    killed: bool = False
    failed: bool = False
    writes_after_failure: list[str] = field(default_factory=list)
    uninventoried: list[str] = field(default_factory=list)
    _depth: int = 0
    _call_fsyncs: list[int] = field(default_factory=list)
    _call_names: list[tuple[str, str]] = field(default_factory=list)


def _role(primitive: str, target: Any, args: tuple[Any, ...]) -> str:
    """The file a primitive call writes, by name (never a host path)."""
    if primitive in ("futures_store.save", "carry_ledger.save"):
        path = getattr(target, "path", None)
        return path.name if path is not None else "-"
    if primitive == "risk.persist":
        path = getattr(target, "_state_path", None)
        return path.name if path is not None else "-"
    if primitive.startswith("decision_log"):
        return "decision_log"
    if primitive == "runner.save_state":
        return "runner_state.json"
    return emergency_module.EMERGENCY_NAME


@contextlib.contextmanager
def instrumented(faults: Faults) -> Iterator[Faults]:
    """Patch every primitive and ``os.fsync`` for the duration of one transition."""
    originals: dict[str, Any] = {}
    real_fsync = os.fsync
    real_replace = os.replace
    real_unlink = os.unlink

    def fsync(fd: int) -> None:
        if faults.killed:
            raise Killed("write after death")
        if faults._depth == 0:
            faults.uninventoried.append("fsync outside every inventoried primitive")
            raise UninventoriedWrite("an fsync outside every inventoried primitive")
        step_no = len(faults.steps) + 1
        first_in_call = faults._call_fsyncs[-1] == 0
        faults._call_fsyncs[-1] += 1
        if first_in_call:
            name, role = faults._call_names[-1]
            if faults.kill_at == step_no and faults.when == "mid":
                faults.killed = True
                raise Killed(f"killed mid step {step_no}")
            if faults.fail_at == step_no and faults.fail_in == "fsync" and not faults.failed:
                faults.failed = True
                raise OSError(faults.fail_errno, os.strerror(faults.fail_errno))
        real_fsync(fd)

    def unlink(path: Any, *args: Any, **kwargs: Any) -> None:
        # A dead process removes nothing: the atomic writers' own clean-up of
        # their temporary runs in an `except BaseException` handler, which an
        # in-process kill would otherwise reach and a real death never does.
        if faults.killed:
            raise Killed("unlink after death")
        real_unlink(path, *args, **kwargs)

    def replace(src: Any, dst: Any) -> None:
        if faults.killed:
            raise Killed("write after death")
        step_no = len(faults.steps) + 1
        if faults.fail_at == step_no and faults.fail_in == "replace" and not faults.failed:
            faults.failed = True
            raise OSError(faults.fail_errno, os.strerror(faults.fail_errno))
        real_replace(src, dst)

    def wrap(name: str, owner: Any, attribute: str) -> Callable[..., Any]:
        original = getattr(owner, attribute)
        originals[name] = (owner, attribute, original)
        is_method = isinstance(owner, type)

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if faults.killed:
                raise Killed(f"{name} after death")
            target = args[0] if is_method and args else None
            role = _role(name, target, args)
            if faults.failed and name != "emergency.trip":
                faults.writes_after_failure.append(f"{name}:{role}")
            faults._depth += 1
            faults._call_fsyncs.append(0)
            faults._call_names.append((name, role))
            try:
                result = original(*args, **kwargs)
            finally:
                faults._depth -= 1
                wrote = faults._call_fsyncs.pop()
                faults._call_names.pop()
            if wrote and faults._depth == 0:
                faults.steps.append(Step(len(faults.steps) + 1, name, role))
                if faults.kill_at == len(faults.steps) and faults.when == "after":
                    faults.killed = True
                    raise Killed(f"killed after step {len(faults.steps)}")
            elif wrote and faults._depth > 0:
                # A primitive inside a primitive (the ledger's directory fsync
                # is inside its own save): the outer call is the step.
                faults._call_fsyncs[-1] += wrote
            return result

        return wrapped

    try:
        for name, (owner, attribute) in PRIMITIVES.items():
            setattr(owner, attribute, wrap(name, owner, attribute))
        os.fsync = fsync
        os.replace = replace
        os.unlink = unlink
        yield faults
    finally:
        os.fsync = real_fsync
        os.replace = real_replace
        os.unlink = real_unlink
        for owner, attribute, original in originals.values():
            setattr(owner, attribute, original)


# ---------------------------------------------------------------------------
# a runner from disk: a new process
# ---------------------------------------------------------------------------
@dataclass
class World:
    """One recorder root (read-only, shared) and the config overrides."""

    root: Path
    limits: Mapping[str, Any] | None = None

    def config(self, state_dir: Path):
        return campaign_config(state_dir, limits=self.limits)

    def runner(self, state_dir: Path) -> DemoRunner:
        """A fresh process's runner over ``state_dir``: nothing carried in memory."""
        cfg = self.config(state_dir)
        contract = load_recorder_contract("btcusdt-prospective-gen3")
        rules = RuleRegistry([CarryRule(CarryParams.from_config(cfg.rule_params("R1_carry")))])

        def risk_factory(clock: Any) -> Any:
            return build_risk_engine(cfg, capital=CAPITAL, state_dir=state_dir, clock=clock)

        def position_factory(risk: Any, clock: Any) -> Any:
            return build_hedged_position(
                risk=risk, capital=CAPITAL, state_dir=state_dir, clock=clock
            )

        return DemoRunner(
            cfg,
            self.root,
            contract=contract,
            inspection_factory=lambda: inspect_demo_state(
                cfg, capital=CAPITAL, state_dir=state_dir
            ),
            risk_factory=risk_factory,
            position_factory=position_factory,
            rules=rules,
            capital=CAPITAL,
            software={"revision": "synthetic", "dirty": False, "python": "3.11"},
            telemetry=NullTelemetry(),
        )


def copy_state(source: Path, destination: Path) -> Path:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    lock = destination / "runner.lock"
    if lock.exists():
        lock.unlink()
    return destination


# ---------------------------------------------------------------------------
# the operator, and the consistency checks
# ---------------------------------------------------------------------------
@dataclass
class Operated:
    state: RunnerState
    actions: list[str]
    refused: str = ""
    #: Aegis's funding streak just before the operator's first ``resume``,
    #: which clears it by design (``RiskEngine.resume``): what the recovery
    #: left, before the operator's own documented reset -- with how many
    #: settlements the ledger had booked by then.
    streak_before_resume: tuple[int, bool, int] | None = None


NOTE = "crash harness: the operator checked the files the runbook names"


def operate(
    world: World,
    state_dir: Path,
    *,
    limit: int = 12,
    retry: str | None = None,
    since: int = 0,
    keep_halt: str | None = None,
) -> Operated:
    """Drive a halted campaign back through the documented clearing paths.

    One command per process, each through ``start()``, in the runbook's order:

    1. a persistence failure in the emergency record is acknowledged
       (`resolve --emergency`) before anything it left behind is judged;
    2. the command the failed process was running is run again when the log
       holds no completion of it after ``since`` records (``retry``: an operator
       whose command died or exited 4 sees no success), and so is any command a
       restart reported unfinished (OPERATOR_INCOMPLETE);
    3. each standing dispute is cleared by its own `resolve`;
    4. the halt is resumed -- unless its reason starts with ``keep_halt``, the
       halt the transition itself is meant to end in.

    A refusal stops here and is reported.
    """
    actions: list[str] = []
    streak: tuple[int, bool, int] | None = None
    handled: set[int] = set()
    for _ in range(limit):
        runner = world.runner(state_dir)
        state = runner.start()
        try:
            if runner.emergency_trace.needs_attention:
                actions.append("resolve --emergency")
                runner.resolve_emergency(NOTE)
                runner.shutdown("operator command done")
                continue
            if retry is not None and not completed(state_dir, retry, since):
                actions.append(f"again: {retry}")
                command, retry = retry, None
                rerun(runner, command)
                runner.shutdown("operator command done")
                continue
            retry = None
            unfinished = unfinished_command(state_dir, handled)
            if unfinished is not None:
                seq, command = unfinished
                handled.add(seq)
                actions.append(f"again: {command}")
                rerun(runner, command)
                runner.shutdown("operator command done")
                continue
            if state in (RunnerState.READY, RunnerState.FEED_STALLED):
                return Operated(state, actions, streak_before_resume=streak)
            disputes = runner.standing_disputes()
            if disputes:
                selector = disputes[0][0]
                actions.append(f"resolve {selector}")
                if selector == "--ledger asymmetric_close" and runner.position.imbalance():
                    # The runbook: one leg did not close, so close it.
                    actions[-1] = "flatten (asymmetric_close)"
                    runner.flatten(NOTE)
                elif selector.startswith("--ledger "):
                    runner.resolve_ledger(selector.split(" ", 1)[1], NOTE)
                elif selector.startswith("--symbol "):
                    runner.resolve(selector.split(" ", 1)[1], NOTE)
                elif selector == "--equity":
                    runner.resolve_equity(NOTE)
                else:
                    runner.resolve_risk_state(NOTE)
            elif keep_halt is not None and (runner.halt_reason or "").startswith(keep_halt):
                runner.shutdown("halted, as intended")
                return Operated(state, actions, streak_before_resume=streak)
            else:
                if streak is None:
                    streak = (
                        runner.risk.state.funding_adverse_streak,
                        runner.risk.state.funding_halt,
                        len(runner.position.ledger.state.settled),
                    )
                actions.append("resume")
                runner.resume(NOTE)
        except RunnerError as exc:
            return Operated(
                runner.state, actions, refused=str(exc), streak_before_resume=streak
            )
        runner.shutdown("operator command done")
    return Operated(
        RunnerState.HALT, actions, refused="the operator gave up", streak_before_resume=streak
    )


def completed(state_dir: Path, command: str, since: int) -> bool:
    """Whether a completion record of ``command`` exists after ``since`` records."""
    for record in records(state_dir)[since:]:
        operator = record.get("operator")
        if (
            isinstance(operator, Mapping)
            and operator.get("phase") == "completed"
            and operator.get("command") == command
        ):
            return True
    return False


def unfinished_command(state_dir: Path, handled: set[int]) -> tuple[int, str] | None:
    """The newest OPERATOR_INCOMPLETE finding not yet acted on, as (seq, command)."""
    found: tuple[int, str] | None = None
    for record in records(state_dir):
        recovery = record.get("recovery")
        if (
            isinstance(recovery, Mapping)
            and recovery.get("cause") == "OPERATOR_INCOMPLETE"
            and record["seq"] not in handled
        ):
            found = (int(record["seq"]), str(recovery.get("command")))
    return found


def rerun(runner: DemoRunner, command: str) -> None:
    """Run an unfinished operator command again, on a started runner."""
    if command == "flatten":
        runner.flatten(NOTE)
    elif command == "resume":
        if runner.state is RunnerState.HALT:
            runner.resume(NOTE)
    elif command == "resolve-emergency":
        if runner.emergency_trace.needs_attention:
            runner.resolve_emergency(NOTE)
    elif command == "resolve-ledger":
        disputes = [s for s, _ in runner.standing_disputes() if s.startswith("--ledger ")]
        if disputes:
            runner.resolve_ledger(disputes[0].split(" ", 1)[1], NOTE)
    elif command == "resolve":
        disputes = [s for s, _ in runner.standing_disputes() if s.startswith("--symbol ")]
        if disputes:
            runner.resolve(disputes[0].split(" ", 1)[1], NOTE)
    elif command == "resolve-equity":
        if any(s == "--equity" for s, _ in runner.standing_disputes()):
            runner.resolve_equity(NOTE)
    else:  # pragma: no cover - a command the harness does not know
        raise AssertionError(f"unknown unfinished command {command!r}")


def economics(runner: DemoRunner) -> dict[str, Any]:
    """What the campaign holds, exactly, for comparing two histories.

    The ledger's slippage is left out on purpose: it is the one quantity no
    executor accumulates, so a re-booked crash window cannot restore the lost
    cycle's measurement, and `resolve --ledger` says so rather than guessing.
    """
    position = runner.position
    state = position.ledger.state

    def leg(name: str) -> tuple[str, str, str]:
        held = position.leg(name)
        return held.side.value, str(held.quantity), str(held.entry_price)

    return {
        "spot": leg("spot"),
        "perp": leg("perp"),
        "spot_fees": str(position.spot.ledger.trading_fees),
        "perp_fees": str(position.perp.ledger.trading_fees),
        "spot_realised": str(position.spot.ledger.realised_pnl),
        "perp_realised": str(position.perp.ledger.realised_pnl),
        "perp_funding": (
            str(position.perp.ledger.funding_paid),
            str(position.perp.ledger.funding_received),
        ),
        "free_cash": str(state.free_cash),
        "spot_principal": str(state.spot_principal),
        "perp_margin": str(state.perp_margin),
        "carry_fees": str(state.fees),
        "carry_realised": str(state.realised),
        "carry_funding": (str(state.funding_paid), str(state.funding_received)),
        "settled": sorted(state.settled),
        "open_instant_ns": state.open_instant_ns,
        "equity": str(state.last_equity),
        "risk_equity": repr(runner.risk.state.equity),
        "open_positions": dict(sorted(runner.risk.state.open_positions.items())),
        # Aegis's funding guard: a torn settlement re-booked must be counted
        # exactly once.
        "funding_streak": (
            runner.risk.state.funding_adverse_streak,
            runner.risk.state.funding_halt,
        ),
    }


def consistency(runner: DemoRunner, *, cash_offset: Decimal = Decimal("0")) -> list[str]:
    """Internal consistency, independent of any reference run.

    * free cash is exactly what the executors' own records imply: capital, less
      each leg's level (quantity x VWAP), less both legs' fees, plus both legs'
      realised PnL and the perpetual's net funding;
    * the ledger's levels and hedged quantity are the legs';
    * Aegis's equity is the ledger's (R1-b's comparison) -- except after a
      liquidation touch, whose flatten R1-b documents as not handing Aegis the
      post-flatten equity (the dispute surfaces on resume, `resolve --equity`);
    * Aegis's exposure per symbol is what the store implies.

    ``cash_offset`` is a fixture's own deliberate edit of the cash, if any.
    """
    from chimera.demo.inspection import store_exposure

    problems: list[str] = []
    position = runner.position
    state = position.ledger.state
    spot, perp = position.leg("spot"), position.leg("perp")
    implied_cash = (
        CAPITAL
        + cash_offset
        - spot.quantity * spot.entry_price
        - perp.quantity * perp.entry_price
        - (position.spot.ledger.trading_fees + position.perp.ledger.trading_fees)
        + (position.spot.ledger.realised_pnl + position.perp.ledger.realised_pnl)
        + (position.spot.ledger.net_funding + position.perp.ledger.net_funding)
    )
    if state.free_cash != implied_cash:
        problems.append(f"free cash {state.free_cash} is not the legs' {implied_cash}")
    if state.spot_principal != spot.quantity * spot.entry_price:
        problems.append("the ledger's spot principal is not the spot leg's level")
    if state.perp_margin != perp.quantity * perp.entry_price:
        problems.append("the ledger's perp margin is not the perp leg's level")
    if state.quantity != min(spot.quantity, perp.quantity):
        problems.append("the ledger's hedged quantity is not the legs'")
    touched = runner.risk.state.halt_reason.startswith("liquidation_touch")
    if (
        not touched
        and state.last_equity is not None
        and float(state.last_equity) != runner.risk.state.equity
    ):
        problems.append(
            f"Aegis's equity {runner.risk.state.equity!r} is not the ledger's {state.last_equity}"
        )
    for executor, symbol in (
        (position.spot, position.config.spot_symbol),
        (position.perp, position.config.perp_symbol),
    ):
        implied = store_exposure(executor.store, symbol)
        if runner.risk.state.open_positions.get(symbol) != implied:
            problems.append(
                f"Aegis's exposure for {symbol} is {runner.risk.state.open_positions.get(symbol)} "
                f"and the store implies {implied}"
            )
    return problems


def excluded_minutes(state_dir: Path, since: int = 0) -> list[str]:
    """Minutes a RECOVERY record excluded from the evidence after ``since``."""
    out: list[str] = []
    for record in records(state_dir)[since:]:
        recovery = record.get("recovery")
        if isinstance(recovery, Mapping) and recovery.get("evidence_excluded_minute"):
            out.append(str(recovery["evidence_excluded_minute"]))
    return out


def check_log(state_dir: Path) -> list[str]:
    """The decision log verifies and never decides a minute twice."""
    problems: list[str] = []
    verification = verify_log(state_dir / "decision_log")
    if not verification.ok:
        problems.append(f"log does not verify: {verification.summary()}")
    decided: dict[str, int] = {}
    for path in sorted((state_dir / "decision_log").glob("*.ndjson")):
        for record in read_records(path):
            if record.get("kind") == "DECISION":
                decided[record["minute"]] = decided.get(record["minute"], 0) + 1
    twice = sorted(minute for minute, count in decided.items() if count > 1)
    if twice:
        problems.append(f"minutes decided twice: {twice}")
    return problems


def records(state_dir: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in sorted((state_dir / "decision_log").glob("*.ndjson")):
        out.extend(read_records(path))
    return out


def state_bytes(state_dir: Path) -> dict[str, bytes]:
    """Every persisted file, for byte comparison. The lock is not state.

    The recorder's atomic writer names its temporary ``<name>.<random>.tmp``
    (R1-h), so a temporary a death left behind is compared by content under one
    canonical name rather than by its random one.
    """
    return {
        _RANDOM_TMP.sub(".RANDOM.tmp", str(path.relative_to(state_dir))): path.read_bytes()
        for path in sorted(state_dir.rglob("*"))
        if path.is_file() and path.name != "runner.lock"
    }


_RANDOM_TMP = re.compile(r"\.[0-9a-f]{8,}\.tmp$")


def minute_ms(base_ms: int, index: int) -> int:
    return base_ms + index * 60_000


D = Decimal
