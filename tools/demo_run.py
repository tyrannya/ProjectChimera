"""Section 8.3's operator CLI for the demo runner.

Exactly six subcommands, as the plan lists them:

    run --config PATH [--replay FROM TO | --once] [--allow-dirty] [--metrics-port PORT]
    status
    flatten --note TEXT
    resume --note TEXT
    resolve (--symbol S | --equity) --note TEXT
    report --day D

"Every operator command writes a decision-log record with `kind = OPERATOR`."

`resolve` clears one dispute, and which one is named rather than guessed:
``--symbol`` is a leg's reconciliation dispute, ``--equity`` the one a restart
raises when the persisted risk state and the carry ledger state different
equities (R1-b). One or the other, never both and never neither, so the command
cannot be run in the hope that it clears whatever it finds. Six subcommands
still, as section 8.3 lists them.

``--note`` is REQUIRED on `flatten`, `resume` and `resolve`, and a note that is
empty or only whitespace is refused. That matches
`FuturesExecutor.resolve_reconciliation`, which already refuses one: an operator
action with no stated reason is an unexplained change to a position, and the
decision log is the only place that reason will ever exist.

**`run` is the service (R1-d).** Without ``--replay`` or ``--once`` it starts the
runner ONCE and then loops: catch up every closed minute, wait for the next
minute close plus section 8.1's READY grace, repeat -- until SIGTERM or SIGINT,
which is honoured only between minutes, so a minute's PERSISTENCE always
completes. It exits 0 on that request, 3 on HALT (so a supervisor's
``RestartPreventExitStatus=3`` never turns a halt into an automatic resume), and
with a traceback on anything it did not expect. ``--replay`` and ``--once`` are
the bounded forms, for replays and diagnostics, and behave as `run` always did.

**Two clocks, and this file is where they are kept apart (R1-e).** The DECISION
clock is the runner's `RunnerClock`, advanced only by recorded instants; it is
the only clock Aegis and both executors are ever built with, and the factories
below cannot be called without it. The OPERATIONAL clock is the host's wall time,
for scheduling the service's wake-ups, the heartbeat and the operator's age
gauges, and never for a decision. It enters the process in exactly one place --
`main`'s ``operational_clock`` and ``sleep`` -- and from there reaches the
service loop and the telemetry as the same one clock; nothing further in reads
the host's time for itself.

`run`, `flatten`, `resume` and `resolve` each hold an exclusive lock on the state
directory while they run. A service keeps the decision log's chain head in
memory, so a second writer beside it would fork the chain; the second one is
refused instead (exit 2). `status` and `report` only read and take no lock.

This tool builds a dry-run venue through `chimera.carry.factory` -- the one
place section 7.6 permits -- and reads the recorder's files read-only. It reads
no credential, and the only socket it can open is the Prometheus scrape endpoint
`run --metrics-port` asks for: a listening port that serves counters, reachable
from nothing the runner reads back.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import signal
import sys
import time
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

from chimera.carry.factory import build_hedged_position
from chimera.demo.config import ConfigProfile, DemoConfig, parse_demo_config
from chimera.demo.inspection import inspect_demo_state
from chimera.demo.risk_wiring import build_risk_engine
from chimera.demo.rules import RuleRegistry
from chimera.demo.rules_carry import CarryParams, CarryRule
from chimera.demo.rules_shadow import DailyMomentumRule, FrozenLogisticRule, ShadowParams
from chimera.demo.runner import _STATE_NAMES, DemoRunner, RunnerError, RunnerState
from chimera.demo.telemetry import NullTelemetry, RunnerTelemetry
from chimera.futures.fills import RecordedQuoteFillModel
from chimera.recorder.contract import load_recorder_contract
from chimera.recorder.events import NS_PER_SECOND

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_REFUSED = 2
EXIT_HALTED = 3

#: R1-d's heartbeat cadence while the service waits for the next minute close.
HEARTBEAT_SECONDS = 30.0
#: The longest single sleep. A stop request is noticed within this, because the
#: signal handler only sets a flag (see `_install_stop_handlers`).
WAIT_SLICE_SECONDS = 1.0
#: The commands that write under the state directory, and therefore lock it.
WRITING_COMMANDS: frozenset[str] = frozenset({"run", "flatten", "resume", "resolve"})
#: The lock file, under the state directory.
LOCK_NAME = "runner.lock"


def _require_note(value: str, command: str) -> str:
    note = (value or "").strip()
    if not note:
        raise SystemExit(
            f"{command} requires a non-empty --note. An operator action with no stated "
            "reason is an unexplained change, and this note is the only record of why"
        )
    return note


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="demo_run",
        description="Operator commands for the demo runner (section 8.3).",
    )
    parser.add_argument("--config", type=Path, help="campaign configuration JSON")
    parser.add_argument("--root", type=Path, help="the recorder's storage root")
    parser.add_argument(
        "--profile",
        default=ConfigProfile.CAMPAIGN.value,
        choices=[p.value for p in ConfigProfile],
        help="which profile the configuration must declare",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser(
        "run", help="run the service: every closed minute, until SIGTERM or SIGINT"
    )
    run.add_argument("--replay", nargs=2, metavar=("FROM", "TO"), type=int)
    run.add_argument(
        "--once",
        action="store_true",
        help="one bounded catch-up pass, then exit (diagnostics; not the service)",
    )
    run.add_argument(
        "--allow-dirty",
        action="store_true",
        help="soak runs only: skip the source-identity check",
    )
    run.add_argument("--max-minutes", type=int, default=None)
    run.add_argument(
        "--metrics-port",
        type=int,
        default=None,
        help="serve Prometheus metrics on this port (default: do not serve)",
    )

    sub.add_parser("status", help="print the runner's state as JSON")

    flatten = sub.add_parser("flatten", help="reduce the position to flat")
    flatten.add_argument("--note", required=True)

    resume = sub.add_parser("resume", help="leave HALT")
    resume.add_argument("--note", required=True)

    resolve = sub.add_parser("resolve", help="clear one dispute")
    which = resolve.add_mutually_exclusive_group(required=True)
    which.add_argument("--symbol", help="a leg whose reconciliation is disputed")
    which.add_argument(
        "--equity",
        action="store_true",
        help="the restart equity dispute between risk.json and carry_ledger.json",
    )
    resolve.add_argument("--note", required=True)

    report = sub.add_parser("report", help="summarise one day's decision log")
    report.add_argument("--day", required=True)
    return parser


def _config(args: argparse.Namespace) -> DemoConfig:
    if args.config is None or args.root is None:
        raise SystemExit("--config and --root are required")
    payload = json.loads(Path(args.config).read_text(encoding="utf-8"))
    return parse_demo_config(payload, expected_profile=ConfigProfile(args.profile))


def _load(
    args: argparse.Namespace, *, operational_clock: Callable[[], float] | None = None
) -> DemoRunner:
    """Build the runner. ``operational_clock`` is required by every command but
    `status`, which emits nothing and so reads no clock at all."""
    config = _config(args)

    contract = load_recorder_contract("btcusdt-prospective-gen3")
    state_dir = Path(config.runner_setting("state_dir"))
    capital = Decimal("1000000")
    rules = RuleRegistry([CarryRule(CarryParams.from_config(config.rule_params("R1_carry")))])
    for rule_id, cls in (
        ("R2_frozen_logistic", FrozenLogisticRule),
        ("R3_daily_momentum", DailyMomentumRule),
    ):
        params = config.rule_params(rule_id)
        if params:
            rules.register(cls(ShadowParams.from_config(rule_id, params)))

    # Section 7.4's limits reach Aegis through `chimera.demo.risk_wiring` and
    # nowhere else. Building `RiskLimits` here is what let a campaign be hashed
    # under one risk regime and enforced under another; the mapping is one
    # audited function now, and `tests/test_demo_risk_wiring.py` holds this file
    # to using it.
    def _risk_factory(clock: Any) -> Any:
        return build_risk_engine(config, capital=capital, state_dir=state_dir, clock=clock)

    def _position_factory(risk: Any, clock: Any) -> Any:
        return build_hedged_position(
            risk=risk,
            capital=capital,
            state_dir=state_dir,
            fill_model=RecordedQuoteFillModel(),
            clock=clock,
        )

    def _inspection_factory():
        return inspect_demo_state(config, capital=capital, state_dir=state_dir)

    # `status` is an inspection, not a process-liveness event. In particular it
    # must not move Prometheus state merely by being read.
    command = getattr(args, "command", None)
    telemetry: Any
    if command == "status":
        telemetry = NullTelemetry()
    elif operational_clock is None:
        raise ValueError(
            f"{command} needs the operational clock for its telemetry; `main` composes "
            "it, and the runner is not allowed to pick a wall clock of its own"
        )
    else:
        clock = operational_clock
        telemetry = RunnerTelemetry(
            state_dir=state_dir,
            states=_STATE_NAMES,
            rules=rules.ids,
            wall_ns=lambda: int(clock() * NS_PER_SECOND),
        )

    return DemoRunner(
        config,
        args.root,
        contract=contract,
        inspection_factory=_inspection_factory,
        risk_factory=_risk_factory,
        position_factory=_position_factory,
        rules=rules,
        capital=capital,
        software=_software(),
        telemetry=telemetry,
    )


#: The checkout whose source `_software` identifies. `tools/` sits directly
#: under the repository root, so the root is this file's grandparent -- the same
#: derivation `nn/p2b.py` uses for the same call.
CHECKOUT_ROOT = Path(__file__).resolve().parent.parent


def _software(root: Path | None = None) -> dict[str, Any]:
    """The revision block section 9.1 stamps into every record.

    ``root`` is the checkout to identify and defaults to this one. It is a
    parameter because the only honest test of the CAMPAIGN self-check is
    against a *real* checkout that is genuinely clean and then genuinely dirty,
    and a test cannot dirty the checkout it is itself running from.

    The block is built from fail-closed values that a COMPLETE identification
    replaces, and never the other way round. `source_identity` REQUIRES its
    ``root`` and returns a MAPPING: calling it bare raised `TypeError` on every
    run, and reading the mapping with `getattr` then yielded ""/""/False -- so
    once the first half was fixed alone, a genuinely dirty tree would have
    reported itself CLEAN and a campaign would have run on source nobody could
    reconstruct. Both halves are load-bearing, which is why the keys below are
    read by key and why an incomplete identity does not reach the caller.
    """
    revision = ""
    digest = ""
    dirty = True
    try:
        from nn.source_identity import source_identity

        identity = source_identity(CHECKOUT_ROOT if root is None else Path(root))
        found_revision = identity.get("revision")
        found_digest = identity.get("source_digest")
        found_dirty = identity.get("dirty")
        # `revision` and `dirty` are `None` when git could not be asked, and
        # `bool(None)` is False: falling through on a partial answer would claim
        # a clean tree on the strength of a question nobody answered.
        if found_revision and found_digest and found_dirty is not None:
            revision = str(found_revision)
            digest = str(found_digest)
            dirty = bool(found_dirty)
    except Exception:
        # A runner that cannot establish its own revision says so rather than
        # claiming one; SELF_CHECK then refuses a campaign on the dirty flag.
        pass
    return {
        "revision": revision,
        "source_digest": digest,
        "dirty": dirty,
        "python": sys.version.split()[0],
    }


class StateDirectoryBusy(RuntimeError):
    """Another process holds the state directory's writer lock."""


@contextlib.contextmanager
def _single_writer(state_dir: Path) -> Iterator[None]:
    """Hold an exclusive, non-blocking lock on ``state_dir`` for the command.

    Advisory, and the OS drops it when the holder dies -- SIGKILL included -- so
    a crash never leaves a lock an operator has to delete by hand. Taken BEFORE
    the runner is built: a snapshot read while another writer was still active
    would be stale the moment that writer appended again.
    """
    state_dir.mkdir(parents=True, exist_ok=True)
    handle = open(state_dir / LOCK_NAME, "a+b")
    locked = False
    try:
        try:
            if sys.platform == "win32":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise StateDirectoryBusy(
                f"another demo_run process holds {state_dir / LOCK_NAME}: the service is "
                "probably running. Stop it first (systemctl stop chimera-demo); two "
                "writers on one state directory would fork the decision log's chain"
            ) from exc
        locked = True
        yield
    finally:
        if locked and sys.platform == "win32":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        handle.close()


def _next_wake(now_s: float, grace_s: float) -> float:
    """The next minute close plus ``grace_s``, strictly after ``now_s``.

    Absolute, so the schedule does not drift with how long a pass took. After an
    overrun it is simply the next such instant: the minutes missed meanwhile are
    not separate wakes, because one `catch_up` takes every pending minute.
    """
    return (math.floor((now_s - grace_s) / 60.0) + 1) * 60.0 + grace_s


def _install_stop_handlers(stop: dict[str, int]) -> dict[int, Any]:
    """SIGTERM and SIGINT only RECORD a stop request; the loop acts on it.

    The handler writes one dict entry and returns. It does no I/O, takes no lock
    and changes no runner state, so wherever the signal lands -- a PERSISTENCE
    write included -- that code runs on to completion. A `threading.Event` is
    deliberately not used: `set()` from a handler that interrupted the same
    Event's `wait()` can block on the lock that `wait()` still holds.
    """

    def request(signum: int, _frame: Any) -> None:
        stop["signal"] = signum

    previous: dict[int, Any] = {}
    for name in ("SIGTERM", "SIGINT"):
        number = getattr(signal, name)
        previous[number] = signal.signal(number, request)
    return previous


def _daemon(
    runner: DemoRunner,
    stop: dict[str, int],
    *,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
) -> int:
    """R1-d's continuous READY loop, over a runner `start()` has already started.

    Each pass is `catch_up`, the existing unit of work, and nothing else: no
    restart check runs again, because none of them is repeated in-process. The
    wait between passes ends at the next minute close plus section 8.1's READY
    grace, sleeps in slices of at most `WAIT_SLICE_SECONDS`, and beats the
    heartbeat at most `HEARTBEAT_SECONDS` apart -- on this thread, so a wedged
    loop stops beating.

    ``clock`` and ``sleep`` are the OPERATIONAL clock (R1-e): host wall seconds
    and the matching wait, injected by `main`, with no default here so that this
    loop cannot quietly read the host for itself. They decide WHEN a pass runs
    and when the heartbeat beats, never WHAT a pass decides: every decision
    instant is still the runner's own recorded clock, and nothing they return
    reaches a record.

    R1-f's READY gate runs first in every pass, on this same clock, whether or
    not a minute arrived: `check_feed` compares it with the instant the
    recorder's heartbeat vouches for the required kline stream up to, not the
    newest normalized minute (R1-g kept it so; `DemoRunner.check_feed` says
    why). A stale feed turns the pass into a stall tick, which checks the held
    position on the last recorded minute and decides nothing; a fresh one
    catches up -- every pending minute, since R1-g, up to any whose funding
    settlement has not been recorded yet. While the feed is fresh the wait
    also ends at the instant it would turn stale plus the same grace, so a dead
    stream or recorder is declared within ``max_data_delay_s +
    ready_grace_seconds`` of the last instant vouched for, for any limit, not
    only one that falls on a minute.
    """
    grace = float(runner.config.runner_setting("ready_grace_seconds"))

    def requested() -> bool:
        return "signal" in stop

    next_beat = clock()
    while not requested():
        stale_after_ns = runner.check_feed(int(clock() * NS_PER_SECOND))
        if runner.state is RunnerState.FEED_STALLED:
            runner.stall_tick()
        else:
            runner.catch_up(stop=requested)
        if runner.state is RunnerState.HALT:
            reason = runner.halt_reason
            outcome = runner.shutdown(f"halted: {reason}")
            print(
                json.dumps(
                    {"state": "HALT", "reason": reason, "last_record": outcome.record_hash}
                )
            )
            return EXIT_HALTED
        wake = _next_wake(clock(), grace)
        if stale_after_ns is not None and runner.state is not RunnerState.FEED_STALLED:
            wake = min(wake, stale_after_ns / NS_PER_SECOND + grace)
        while not requested():
            now = clock()
            if now >= wake:
                break
            if now >= next_beat:
                runner.telemetry.on_heartbeat()
                next_beat = now + HEARTBEAT_SECONDS
            sleep(min(WAIT_SLICE_SECONDS, wake - now, next_beat - now))

    name = signal.Signals(stop["signal"]).name
    logger.info("stop requested by %s; the minute in hand has finished", name)
    outcome = runner.shutdown(f"stopped by {name}")
    print(json.dumps({"state": runner.state.value, "last_record": outcome.record_hash}))
    return EXIT_OK


def main(
    argv: Sequence[str] | None = None,
    *,
    operational_clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """The process's entry point, and the one place the host's clock enters it.

    ``operational_clock`` and ``sleep`` default to the host's because this IS
    the composition boundary: an operational clock is host wall time by
    definition. Everything below receives them as arguments (R1-e) and none of
    it may read the host for itself; `tests/test_r1e_clock_hardening.py` holds
    every other ``time`` reference in this file, and every one in the demo
    package, to zero. They never reach the decision clock: the runner's Aegis
    and executors are built on its own `RunnerClock` instead.
    """
    args = build_parser().parse_args(argv)

    if args.command == "report":
        return _report(args)

    if args.command not in WRITING_COMMANDS:
        return _command(args, operational_clock=operational_clock, sleep=sleep)
    try:
        with _single_writer(Path(_config(args).runner_setting("state_dir"))):
            return _command(args, operational_clock=operational_clock, sleep=sleep)
    except StateDirectoryBusy as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_REFUSED


def _command(
    args: argparse.Namespace,
    *,
    operational_clock: Callable[[], float],
    sleep: Callable[[float], None],
) -> int:
    runner = _load(args, operational_clock=operational_clock)

    if args.command == "run":
        if args.metrics_port is not None:
            # Imported and called here rather than in `chimera/demo`, because
            # `serve_metrics` opens a listening socket and the whole demo
            # package is held to opening none (tests/test_demo_no_live_path.py,
            # and PR-10's "the runner never opens a socket"). The endpoint is
            # therefore the CLI's, exactly as it is in tools/recorder.py, and
            # nothing listens unless an operator asks for it.
            from chimera.metrics import serve_metrics

            serve_metrics(args.metrics_port)

        # Once per process, for the process's whole life: the endpoint above,
        # the handlers below and `start()` -- which is where the restart checks
        # (R1-b's equity reconciliation, R1-c's continuity verdict, section 9.3's
        # triage) run. The service loop only ever repeats `catch_up`.
        service = not (args.once or args.replay)
        stop: dict[str, int] = {}
        previous = _install_stop_handlers(stop) if service else {}
        try:
            state = runner.start(allow_dirty=args.allow_dirty)
            if state.value == "HALT":
                print(json.dumps({"state": "HALT", "reason": runner.halt_reason}))
                return EXIT_HALTED
            if service:
                return _daemon(runner, stop, clock=operational_clock, sleep=sleep)
            if args.replay:
                runner.replay(args.replay[0], args.replay[1])
            else:
                runner.catch_up()
            outcome = runner.shutdown("cli run complete")
            print(
                json.dumps({"state": runner.state.value, "last_record": outcome.record_hash})
            )
            return EXIT_OK
        finally:
            for number, handler in previous.items():
                signal.signal(number, handler)

    if args.command == "status":
        print(json.dumps(_status(runner), indent=2, sort_keys=True))
        return EXIT_OK

    if args.command == "flatten":
        note = _require_note(args.note, "flatten")
        runner.start(allow_dirty=True)
        outcome = runner.flatten(note)
        print(json.dumps({"flattened": True, "record": outcome.record_hash}))
        return EXIT_OK

    if args.command == "resume":
        note = _require_note(args.note, "resume")
        try:
            outcome = runner.resume(note)
        except RunnerError as exc:
            print(str(exc), file=sys.stderr)
            return EXIT_REFUSED
        print(json.dumps({"resumed": True, "record": outcome.record_hash}))
        return EXIT_OK

    if args.command == "resolve":
        note = _require_note(args.note, "resolve")
        if args.equity:
            # Started first, unlike the `--symbol` arm below. `resolve_equity`
            # reads Aegis, and Aegis does not exist until `start()` builds it --
            # which is also where the restart reconciliation runs and raises the
            # halt this settles, so there is nothing to settle before it. The
            # `--symbol` arm is left exactly as it was: making THAT one reachable
            # from a fresh process is canonical R1-i's item ("`resume` and
            # `resolve` succeed from the CLI via `start()` with log-tail clock
            # seeding"), and changing it here would be taking that decision.
            #
            # Deliberately without `allow_dirty`: a dirty tree makes `start()`
            # refuse and halt, but `RiskEngine.halt` keeps the FIRST reason, so
            # the dispute survives and can still be settled. An operator is never
            # locked out of a recovery path by the state of the working tree.
            try:
                runner.start()
                outcome = runner.resolve_equity(note)
            except RunnerError as exc:
                print(str(exc), file=sys.stderr)
                return EXIT_REFUSED
            print(
                json.dumps(
                    {
                        "resolved": "equity",
                        "state": runner.state.value,
                        "record": outcome.record_hash,
                    }
                )
            )
            return EXIT_OK
        try:
            outcome = runner.resolve(args.symbol, note)
        except RunnerError as exc:
            print(str(exc), file=sys.stderr)
            return EXIT_REFUSED
        print(json.dumps({"resolved": args.symbol, "record": outcome.record_hash}))
        return EXIT_OK

    raise SystemExit(f"unknown command {args.command!r}")  # pragma: no cover


def _status(runner: DemoRunner) -> dict[str, Any]:
    inspection = runner.inspection
    ledger = inspection.ledger.state
    return {
        "campaign_id": runner.config.campaign_id,
        "profile": runner.config.profile.value,
        "protocol_frozen": runner.config.protocol_frozen,
        "runner_state": runner.state.value,
        "halt_reason": runner.halt_reason,
        "hedge_state": inspection.hedge_state.value,
        "imbalance": str(inspection.imbalance),
        "last_minute_processed": runner.cursor.last_minute_processed,
        "last_record_hash": runner.last_record_hash,
        "risk_halted": inspection.risk_state.halted,
        # Canonical R1-c's load-time verdict on `risk.json` against the decision
        # log. Reported by the one command that never calls `start()`, for the
        # same reason `ledger_complaint` is: an operator whose campaign will not
        # start has to be able to see WHY without starting it.
        "risk_continuity": runner.risk_continuity.status_block(),
        # The one read-only inspection command must not report a ledger the guard
        # has refused as if it were the campaign's. `_ledger_regression` is
        # decided in `__init__`, so it is available here without `start()` --
        # which matters, because this command never calls `start()`.
        "ledger_may_speak": runner._ledger_may_speak(),
        "ledger_complaint": (
            None if runner._ledger_may_speak() else runner._ledger_state_complaint()
        ),
        "ledger": {
            "fees": str(ledger.fees),
            "funding_received": str(ledger.funding_received),
            "funding_paid": str(ledger.funding_paid),
            "equity": str(ledger.last_equity) if ledger.last_equity is not None else None,
        },
        "disputed": inspection.ledger.disputed,
    }


def _report(args: argparse.Namespace) -> int:
    """Summarise one day's decision log. Counting only; PR-12 owns the report."""
    if args.config is None:
        raise SystemExit("--config is required to locate the state directory")
    payload = json.loads(Path(args.config).read_text(encoding="utf-8"))
    config = parse_demo_config(payload, expected_profile=ConfigProfile(args.profile))
    path = Path(config.runner_setting("state_dir")) / "decision_log" / f"{args.day}.ndjson"
    if not path.is_file():
        print(json.dumps({"day": args.day, "records": 0, "note": "no log for that day"}))
        return EXIT_OK
    kinds: dict[str, int] = {}
    total = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        total += 1
        kind = json.loads(line).get("kind", "?")
        kinds[kind] = kinds.get(kind, 0) + 1
    print(json.dumps({"day": args.day, "records": total, "kinds": kinds}, sort_keys=True))
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
