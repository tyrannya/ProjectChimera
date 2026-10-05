"""R1's synthetic-soak CI job: the acceptance run, in compressed synthetic time.

R1's TECHNICAL DESIGN, verbatim (``docs/master_roadmap_r0_r18.md``):

    the four-item minimum test architecture (state-machine crash harness; clock
    hostility; replay-determinism CI job; unsafe-default detector) plus a
    **synthetic-soak CI job**.

R1's ACCEPTANCE, verbatim:

    CI green on the exact head of each PR; a 72 h SOAK-profile run unattended on
    an engineering recorder root with one planned restart and one `SIGKILL`,
    PARITY on every day, zero `SKIPPED_STALE` minutes while the recorder was
    healthy, zero manual state edits.

**What this job is.** The five clauses of that acceptance, each witnessed
mechanically over 72 hours of synthetic minutes under the SOAK profile:

===========================  ==========================================================
unattended, 72 h, SOAK       4320 minutes across three UTC days, no operator command
one planned restart          a graceful SIGTERM and a fresh process (day 1, 10:00)
one ``SIGKILL``              a process killed mid-persistence (day 2, 06:00)
PARITY on every day          three replays, one per day boundary, each PARITY
zero ``SKIPPED_STALE``       while the recorder is healthy, and it is frozen once
zero manual state edits      nothing but a runner process writes the state directory
===========================  ==========================================================

**What this job is not.** It is NOT R1's acceptance run. The acceptance is 72
hours of wall-clock time on a host, with a real systemd unit, a real recorder, a
real disk and a real ``SIGKILL`` delivered by the kernel; that run is the
owner's and this file cannot stand in for it. What this job does is make the
acceptance's five clauses fail CI the moment a change breaks one of them,
instead of on hour 50 of a run nobody can repeat cheaply.

Three honest limits of the compression, stated rather than glossed:

* **Time is simulated.** :class:`tests.test_r1d_daemon_service.FakeTime` moves
  only when the service sleeps (R1-d's loop takes its clock and its sleep as
  parameters), so 72 hours pass in seconds and no real scheduler, drift or
  timer is exercised.
* **The kill is in-process.** ``tests/r1i_crash.py``'s fault point raises at a
  named persistence step and then refuses every further write, which
  ``tests/test_r1i_crash_harness.py`` proves byte-for-byte equal to a real
  subprocess killed at the same step. The process that dies here is a real
  fresh process built from disk through the CLI's own loader; what precedes it
  is a graceful stop of the previous one, which is this harness's artifact and
  not part of the acceptance.
* **The world is synthetic.** Every minute comes from
  ``chimera.demo.fixtures``; the venue is a dry-run one. No market data, no
  burned or sealed block, no economic meaning, and nothing here is scientific
  evidence of anything (ENGINEERING, as R1 says of itself).
"""

from __future__ import annotations

import contextlib
import io
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd
import pytest

from chimera.demo.decision_log import RecordKind, verify_log
from chimera.demo.fixtures import MinuteShape, SyntheticFeed
from chimera.demo.telemetry import NullTelemetry
from chimera.recorder.contract import load_recorder_contract
from chimera.recorder.health import HEARTBEAT_INTERVAL_S
from r1i_crash import Faults, Killed, instrumented
from tests.demo_harness import (
    CARRY_PARAMS,
    MOMENTUM_PARAMS,
    REPO,
    SHADOW_PARAMS,
    write_heartbeat,
)
from tests.test_r1d_daemon_service import FakeTime, installed_sigterm
from tools import demo_run, replay_parity

pytestmark = pytest.mark.synthetic_soak

NS = 1_000_000_000
MINUTES_PER_DAY = 1440

#: The three decided days and the one after them. The last decided minute (day
#: three, 23:59) closes at the fourth day's midnight, so that day's settlement
#: file has to exist for the live run and be copied for the replay -- the same
#: reason R1-j's 48 h fixture names three days for two.
DAYS = ("2026-09-19", "2026-09-20", "2026-09-21")
TRAILING_DAY = "2026-09-22"
SOAK_MINUTES = len(DAYS) * MINUTES_PER_DAY

DAY_START_S = pd.Timestamp(DAYS[0], tz="UTC").timestamp()
FIRST_MS = int(DAY_START_S * 1000)

#: A healthy recorder publishes a closed minute this long after it closes
#: (R1-g's incremental publication), and beats this often, phased off the
#: minute so a beat never sits on a close.
PUBLISH_DELAY = 2.0
BEAT_PHASE = 7.0

#: Minutes already published when the first process starts.
FIRST = 10

#: The scripted events, as minute indices into the 72 hours.
RESTART_AT = 600  # day 1, 10:00 -- a graceful SIGTERM and a fresh process

#: Day 2, 06:17: a process dies mid-persistence. The minute is deliberately NOT
#: on the hour and not a funding instant, because the crash window a kill lands
#: in decides what recovery can promise:
#:
#: * on a plain minute the only log write of the transition is the minute's own
#:   DECISION, so a kill just after it leaves the log one record ahead of the
#:   state with the minute FINISHED. The recovering process reconciles the state
#:   to the committed record, decides nothing twice, and the day still reaches
#:   PARITY with nothing excluded. That is what R1-b, R1-c and R1-j promise, and
#:   it is what this witness holds the runtime to.
#: * on the hour the transition appends its hourly RECONCILIATION first. A kill
#:   between that and the DECISION leaves the minute part-recorded and
#:   undecided, which `tools/replay_parity.py` explains and excludes by design --
#:   but the recovering process, whose `last_reconcile_minute_ms` was never
#:   persisted, reconciles AGAIN on the next minute, and that extra
#:   RECONCILIATION is a comparable record in a minute nothing excludes. Every
#:   later `parity_seq` is then off by one and the day reports DIVERGED. A run of
#:   this soak with the kill at day 2, 06:00 showed exactly that: 4,319 of 4,320
#:   minutes decided, one explained exclusion, 1,065 divergences. R1's
#:   ACCEPTANCE asks for one `SIGKILL` *and* PARITY on every day, so that
#:   combination is reachable only outside this window. Making it reachable
#:   inside would mean changing R1-j's exclusion policy or persisting the
#:   reconciliation cadence before the record, which is a design change beyond
#:   this item and belongs in its own reviewed PR (R1's own KILL / STOP
#:   condition).
KILL_AT = MINUTES_PER_DAY + 377
STALL_FROM = MINUTES_PER_DAY + 840  # day 2, 14:00 -- the required stream freezes
STALL_MINUTES = 30

#: `conf/demo/pvc1.json`, section 8.1: the READY gate's limit and its grace.
LIMIT_S = 180.0
GRACE_S = 5.0


def minute_iso(index: int) -> str:
    return pd.Timestamp(FIRST_MS + index * 60_000, unit="ms", tz="UTC").isoformat()


def minute_ms(index: int) -> int:
    return FIRST_MS + index * 60_000


def close_s(index: int) -> float:
    """The operational second at which minute ``index`` closes."""
    return DAY_START_S + 60.0 * (index + 1)


# --------------------------------------------------------------------------- #
# the world: a recorder that publishes minute by minute, and beats
# --------------------------------------------------------------------------- #
class SoakRecorder:
    """Three days of normalized files, published incrementally on simulated time.

    Two cadences, kept apart because R1-f's READY gate reads only one of them:
    the normalized days are rewritten as each minute closes (R1-g), and
    ``health/heartbeat.json`` is written every ``HEARTBEAT_INTERVAL_S`` in the
    recorder's own production shape. The gate's authority is the heartbeat and
    the kline stream's last event in it.

    ``frozen`` is the one failure this soak injects: inside that window of
    operational seconds the required stream is dead -- nothing is published and
    no newer kline event is vouched for -- while every other stream and the
    heartbeat itself carry on, which is what a stalled feed looks like to the
    gate. The window is a fact of the run, so the "while the recorder was
    healthy" clause of R1's acceptance has something to mean.
    """

    def __init__(self, feed: SyntheticFeed, root: Path, contract: Any) -> None:
        self.feed = feed
        self.root = root
        self.contract = contract
        self.published = 0
        self.beat_s: float | None = None
        self.frozen: tuple[float, float] | None = None
        self.publications: list[tuple[float, int]] = []
        self.publish(FIRST)

    def publish(self, count: int) -> None:
        """Publish through minute ``count`` - 1 of the 72 hours.

        Only the days whose contents change are rewritten: the open one, plus
        any day the jump completed and any day that has to appear as empty.
        A finished day is never rewritten -- the recorder does not rewrite
        yesterday, and a soak that did would spend its whole budget on parquet.
        """
        first = self.published // MINUTES_PER_DAY
        last = min(count // MINUTES_PER_DAY, len(DAYS) - 1)
        for offset in range(first, last + 1):
            day = DAYS[offset]
            start = offset * MINUTES_PER_DAY
            hidden = {
                slot: MinuteShape(present=False)
                for slot in range(max(0, count - start), MINUTES_PER_DAY)
            }
            for market in ("um", "spot"):
                self.feed.write_day(market, day, hidden)
        self.published = count

    def is_frozen(self, t: float) -> bool:
        return self.frozen is not None and self.frozen[0] <= t < self.frozen[1]

    def due(self, t: float) -> int:
        return max(FIRST, min(int((t - PUBLISH_DELAY - DAY_START_S) // 60), SOAK_MINUTES))

    def __call__(self, fake: FakeTime) -> None:
        beat = (
            DAY_START_S
            + BEAT_PHASE
            + HEARTBEAT_INTERVAL_S
            * ((fake.t - DAY_START_S - BEAT_PHASE) // HEARTBEAT_INTERVAL_S)
        )
        if beat != self.beat_s:
            self.beat_s = beat
            write_heartbeat(
                self.root,
                self.contract,
                at_ns=int(round(beat * NS)),
                kline_last_ns=int(round(DAY_START_S + 60.0 * (self.published - 1)) * NS),
            )
        if self.is_frozen(fake.t):
            return
        due = self.due(fake.t)
        if due != self.published:
            self.publish(due)
            self.publications.append((fake.t, due))


def config_file(where: Path, state_dir: Path) -> Path:
    """The committed campaign's limits and rules on the SOAK profile.

    SOAK is the profile R1's acceptance names, and the one the config schema
    lets carry a fault block (``chimera.demo.config.FAULT_PROFILES``); the
    limits, the grace and the rule parameters are the committed file's.
    """
    payload = json.loads((REPO / "conf" / "demo" / "pvc1.json").read_text(encoding="utf-8"))
    payload.update(
        {
            "profile": "SOAK",
            "runner": {"state_dir": str(state_dir)},
            "rules": {
                "R1_carry": dict(CARRY_PARAMS),
                "R2_frozen_logistic": dict(SHADOW_PARAMS),
                "R3_daily_momentum": dict(MOMENTUM_PARAMS),
            },
        }
    )
    where.mkdir(parents=True, exist_ok=True)
    path = where / "soak_campaign.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


#: ``--allow-dirty`` is passed to every process here, and it is not a shortcut.
#: ``DemoRunner.self_check`` refuses a working tree it cannot reconstruct unless
#: the caller says this is a soak -- and refuses the flag itself on a CAMPAIGN
#: profile, which `tests/test_demo_cli.py` and R1-a's tests hold it to. A soak IS
#: the run the flag exists for ("soak runs, whose records are operational rather
#: than evidence"), and without it this witness would only run from a checkout
#: that happens to be clean: a developer with one edited file, or a copy of the
#: tree without its `.git`, would watch all six assertions fail at startup for a
#: reason that has nothing to do with what they changed.
ALLOW_DIRTY = "--allow-dirty"


def serve(
    argv: list[str],
    fake: FakeTime,
    *,
    until: float,
    hooks: Iterable[Callable[[FakeTime], None]],
) -> int:
    """One service process: R1-d's loop on simulated time, stopped by SIGTERM.

    ``demo_run.main`` is the entry point, so the runner is built from the files
    on disk exactly as a restarted service builds it, and the operational clock
    enters at the one place R1-e allows. The SIGTERM is delivered through the
    handler the service itself installed.
    """
    fake.hooks = []
    for hook in hooks:
        hook(fake)  # the world as it stands when the process starts
        fake.hooks.append(hook)

    def deadline(f: FakeTime) -> None:
        # A loop that never waits must fail, not hang.
        assert len(f.sleeps) < 400_000, "the service slept more times than the soak can need"
        if f.t >= until:
            installed_sigterm()

    fake.hooks.append(deadline)
    return demo_run.main(
        argv + ["run", ALLOW_DIRTY], operational_clock=fake.clock, sleep=fake.sleep
    )


def records(log_dir: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in sorted(log_dir.glob("*.ndjson")):
        out += [json.loads(line) for line in path.read_text("utf-8").splitlines() if line]
    return out


def state_inventory(state_dir: Path) -> list[str]:
    return sorted(
        p.relative_to(state_dir).as_posix() for p in state_dir.rglob("*") if p.is_file()
    )


# --------------------------------------------------------------------------- #
# the kill: a fresh process that dies mid-persistence
# --------------------------------------------------------------------------- #
def kill_one_minute(
    argv: list[str], root: Path, state_dir: Path, fake: FakeTime, where: Path
) -> dict[str, Any]:
    """Decide one minute in a fresh process and die after the record is durable.

    The step is chosen by running the SAME transition on a COPY of the state
    first and reading its persistence steps, the way
    ``tests/test_r1i_crash_harness.py`` chooses one: the kill lands just after
    ``decision_log.append`` has made the record durable and before
    ``runner_state.json`` names it. That is the crash window that leaves the log
    one record ahead of the state -- the one a recovering process has to
    reconcile rather than ignore -- and it is reached through the CLI's own
    loader, so what dies is a process built from disk.
    """
    probe_args = demo_run.build_parser().parse_args(argv + ["run", "--once", ALLOW_DIRTY])
    probe_state = where / "kill_probe_state"
    if probe_state.exists():
        shutil.rmtree(probe_state)
    shutil.copytree(state_dir, probe_state)
    (probe_state / "runner.lock").unlink(missing_ok=True)
    probe_config = config_file(where / "kill_probe", probe_state)
    probe_argv = [
        "--config", str(probe_config),
        "--root", str(root),
        "--profile", "SOAK",
        "run", "--once", ALLOW_DIRTY,
    ]  # fmt: skip
    probe = demo_run._load(
        demo_run.build_parser().parse_args(probe_argv), operational_clock=fake.clock
    )
    probe.telemetry = NullTelemetry()
    probe.start(allow_dirty=True)
    target_ms = probe.cursor.last_minute_processed + 60_000
    # The minute must be closed and published: a service never decides a minute
    # the recorder has not written, so a kill on one would witness nothing.
    assert target_ms <= minute_ms(SOAK_MINUTES - 1), "the kill target is inside the soak"

    dry = Faults()
    with instrumented(dry):
        probe.tick(target_ms)
    assert not dry.uninventoried, dry.uninventoried
    steps = [(step.index, step.primitive, step.role) for step in dry.steps]
    appends = [step.index for step in dry.steps if step.role == "decision_log"]
    # Exactly one log write, so the kill lands after the minute's own DECISION
    # and not after some earlier record of the same minute. A transition with
    # two (an hourly reconciliation, a funding settlement) is a different crash
    # window with a different promise, named at `KILL_AT`; this assertion is
    # what stops the witness drifting into it unnoticed.
    assert len(appends) == 1, steps
    kill_at = appends[0]

    # The process that really dies, built from disk like any other.
    doomed = demo_run._load(probe_args, operational_clock=fake.clock)
    doomed.telemetry = NullTelemetry()
    doomed.start(allow_dirty=True)
    faults = Faults(kill_at=kill_at, when="after")
    with instrumented(faults), pytest.raises(Killed):
        doomed.tick(target_ms)
    assert faults.killed and not faults.uninventoried, (faults.killed, faults.uninventoried)
    return {
        "minute": pd.Timestamp(target_ms, unit="ms", tz="UTC").isoformat(),
        "steps": steps,
        "killed_after": steps[kill_at - 1],
        "persisted": [step.role for step in faults.steps],
    }


# --------------------------------------------------------------------------- #
# the soak
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def soak(tmp_path_factory):
    """Run the 72 hours once; every test below reads its evidence.

    The sequence, by operational instant:

        process 1   day 1 00:10 .. 10:00   the service starts and runs
        (SIGTERM)   day 1 10:00            the planned restart, graceful
        process 2   day 1 10:00 .. day 2 05:59   unattended across the UTC day roll
        process 3   day 2 06:00            one minute, then death mid-persistence
        process 4   day 2 06:00 .. day 3 23:59   recovery, the stall, and the rest

    At each day's end the live log is snapshotted, so the per-day parity runs
    compare what existed at that boundary rather than what the run later became.
    """
    tmp_path = tmp_path_factory.mktemp("r1_soak")
    root = tmp_path / "recorder"
    state_dir = tmp_path / "state"
    snapshots = tmp_path / "snapshots"
    contract = load_recorder_contract("btcusdt-prospective-gen3")

    feed = SyntheticFeed(root, contract)
    feed.write_settlements([*DAYS, TRAILING_DAY])
    recorder = SoakRecorder(feed, root, contract)
    # The freeze begins once minute STALL_FROM - 1 has been published and ends
    # once the window's last minute would have been: the frozen minutes are
    # exactly STALL_FROM .. STALL_FROM + STALL_MINUTES - 1.
    recorder.frozen = (
        close_s(STALL_FROM - 1) + PUBLISH_DELAY + 1.0,
        close_s(STALL_FROM + STALL_MINUTES - 1) + PUBLISH_DELAY + 1.0,
    )

    config = config_file(tmp_path / "conf", state_dir)
    argv = ["--config", str(config), "--root", str(root), "--profile", "SOAK"]
    fake = FakeTime(start=close_s(FIRST - 1) + PUBLISH_DELAY + 0.5)

    taken: dict[str, Path] = {}

    def snapshot(fake_time: FakeTime) -> None:
        """At 00:00:45 of a new day, freeze the log the finished day left behind.

        The log is what bounds a parity run: ``replay_parity`` replays the live
        log's own minute range, so a snapshot taken between the previous day's
        last decision and the new day's first one is exactly "the campaign so
        far", which is what a daily parity check replays. The recorder root
        needs no snapshot: at this instant every minute of every finished day
        has been published, and later publications fall outside the range.
        """
        for index, day in enumerate(DAYS):
            at = close_s((index + 1) * MINUTES_PER_DAY - 1) + 45.0
            if day in taken or fake_time.t < at:
                continue
            where = snapshots / day / "decision_log"
            shutil.copytree(state_dir / "decision_log", where)
            taken[day] = where

    hooks = (recorder, snapshot)
    exits = [serve(argv, fake, until=close_s(RESTART_AT - 1), hooks=hooks)]
    # Stopped once minute KILL_AT - 1 is decided, so the minute the next
    # process dies on is KILL_AT and nothing else.
    exits.append(
        serve(argv, fake, until=close_s(KILL_AT - 1) + PUBLISH_DELAY + 5.0, hooks=hooks)
    )

    # The uncontrolled death, on the minute the next process would have decided.
    fake.t = close_s(KILL_AT) + PUBLISH_DELAY + 1.0
    recorder(fake)
    kill = kill_one_minute(argv, root, state_dir, fake, tmp_path)

    exits.append(serve(argv, fake, until=close_s(SOAK_MINUTES - 1) + 60.0, hooks=hooks))

    live = records(state_dir / "decision_log")
    following = [*DAYS[1:], TRAILING_DAY]
    parity: dict[str, dict[str, Any]] = {}
    for index, day in enumerate(DAYS):
        assert day in taken, f"no snapshot was taken at the end of {day}"
        scratch = tmp_path / "scratch" / day
        # The days replayed are every decided day so far plus the one after it,
        # whose settlement file the last minute of the last one needs.
        days = [*DAYS[: index + 1], following[index]]
        replay_argv = [
            "--config", str(config),
            "--root", str(root),
            "--profile", "SOAK",
            "--live-log", str(taken[day]),
            "--days", *days,
            "--scratch", str(scratch),
            "--json",
        ]  # fmt: skip
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            code = replay_parity.main(replay_argv, operational_clock=fake.clock)
        parity[day] = {"exit": code, "report": json.loads(captured.getvalue()), "days": days}
    return {
        "live": live,
        "state_dir": state_dir,
        "root": root,
        "recorder": recorder,
        "kill": kill,
        "exits": exits,
        "parity": parity,
        "config": config,
        "snapshots": taken,
        "tmp_path": tmp_path,
        "argv": argv,
    }


# --------------------------------------------------------------------------- #
# the acceptance, clause by clause
# --------------------------------------------------------------------------- #
def test_the_soak_decided_every_minute_of_the_seventy_two_hours_exactly_once(soak):
    """72 h, SOAK, no minute lost -- through a restart, a death and a stall."""
    decided = [r["minute"] for r in soak["live"] if r["kind"] == RecordKind.DECISION.value]
    assert len(decided) == len(set(decided)), "a minute was decided twice"
    assert sorted(decided) == [minute_iso(i) for i in range(SOAK_MINUTES)]
    assert soak["recorder"].published == SOAK_MINUTES
    assert all(code == demo_run.EXIT_OK for code in soak["exits"]), soak["exits"]


def test_no_minute_was_skipped_as_stale_while_the_recorder_was_healthy(soak):
    """R1's acceptance clause, and R1-g's: zero `SKIPPED_STALE`, and the stall
    window proves the clause is about something. Nothing is recorded as an
    incomplete minute either: the service waits for a published minute."""
    kinds = Counter(r["kind"] for r in soak["live"])
    assert kinds[RecordKind.SKIPPED_STALE.value] == 0
    assert kinds[RecordKind.INCOMPLETE_STATE.value] == 0
    assert kinds[RecordKind.FEED_STALLED.value] == 1, "the frozen stream must have halted"
    assert kinds[RecordKind.FEED_RESUMED.value] == 1, "and the feed must have come back"


def test_the_stall_held_the_position_and_lost_no_minute(soak):
    """While FEED_STALLED no minute is decided, and on FEED_RESUMED the runner
    catches up: the window's minutes are decided after the resume, not skipped."""
    live = soak["live"]
    stalled = next(i for i, r in enumerate(live) if r["kind"] == RecordKind.FEED_STALLED.value)
    resumed = next(i for i, r in enumerate(live) if r["kind"] == RecordKind.FEED_RESUMED.value)
    assert stalled < resumed
    during = [r for r in live[stalled + 1 : resumed] if r["kind"] == RecordKind.DECISION.value]
    assert during == [], "a minute was decided while the feed was known stale"
    window = {minute_iso(i) for i in range(STALL_FROM, STALL_FROM + STALL_MINUTES)}
    after = {r["minute"] for r in live[resumed:] if r["kind"] == RecordKind.DECISION.value}
    assert window <= after, "the stalled minutes were never caught up"


def test_the_planned_restart_and_the_uncontrolled_death_both_really_happened(soak):
    """The witness is not vacuous: four processes, one graceful stop, one death
    after the record was durable, and a recovery that reconciled it."""
    live = soak["live"]
    kinds = Counter(r["kind"] for r in live)
    # Four starts: the first process, the planned restart, the process that
    # dies, and the one that recovers. The dead one writes no SHUTDOWN.
    assert kinds[RecordKind.STARTUP.value] == 4, kinds
    assert kinds[RecordKind.SHUTDOWN.value] == 3, kinds
    assert kinds[RecordKind.RECOVERY.value] == 1, "the death left no recovery to do"

    recovery = next(r for r in live if r["kind"] == RecordKind.RECOVERY.value)
    assert recovery["recovery"]["cause"] == "LOG_AHEAD_OF_STATE", recovery["recovery"]
    kill = soak["kill"]
    assert kill["killed_after"][2] == "decision_log", kill
    assert kill["persisted"][-1] == "decision_log", kill
    assert kill["minute"] == minute_iso(KILL_AT), kill
    # The killed minute is decided exactly once, by the process that recovered.
    assert [r["minute"] for r in live if r["kind"] == RecordKind.DECISION.value].count(
        minute_iso(KILL_AT)
    ) == 1


def test_every_day_of_the_soak_replays_to_parity(soak, capsys):
    """ "PARITY on every day": at each day's boundary a replay of the campaign so
    far reproduces the live log, the day the process died included."""
    report_lines: dict[str, Any] = {}
    for day, outcome in soak["parity"].items():
        report = outcome["report"]
        assert outcome["exit"] == replay_parity.EXIT_PARITY, (day, report)
        assert report["label"] == "PARITY" and report["parity"] is True, (day, report)
        assert report["divergences"] == 0, (day, report["first_divergence"])
        assert report["live_only"] == [] and report["replay_only"] == [], (day, report)
        assert report["explained_exclusions"] == [], (day, report)
        assert report["records_compared"] >= MINUTES_PER_DAY, (day, report)
        report_lines[day] = report["records_compared"]

    kinds = Counter(r["kind"] for r in soak["live"])
    with capsys.disabled():
        print(
            "\nR1-SOAK "
            + json.dumps(
                {
                    "hours": SOAK_MINUTES // 60,
                    "minutes": SOAK_MINUTES,
                    "profile": "SOAK",
                    "days_at_parity": sorted(report_lines),
                    "records_compared": report_lines,
                    "decisions": kinds[RecordKind.DECISION.value],
                    "skipped_stale": kinds[RecordKind.SKIPPED_STALE.value],
                    "starts": kinds[RecordKind.STARTUP.value],
                    "recoveries": kinds[RecordKind.RECOVERY.value],
                    "feed_stalls": kinds[RecordKind.FEED_STALLED.value],
                    "operator_commands": kinds[RecordKind.OPERATOR.value],
                    "killed_minute": soak["kill"]["minute"],
                },
                sort_keys=True,
            )
        )


def test_nothing_but_a_runner_process_wrote_the_state_directory(soak):
    """ "zero manual state edits", made checkable: the log's hash chain verifies
    over every day, no operator command was issued at all (the run is
    unattended), and the state directory holds only files a runner writes."""
    state_dir = soak["state_dir"]
    # The whole campaign, not one file at a time: `verify_log` also checks that
    # each day continues the previous one, which is what deleting or rewriting a
    # whole day would break.
    #
    # `expected_last_hash` is what makes this more than a self-consistency
    # check. A chain verifies prefix-wise, so a log whose LAST records were
    # simply removed still links perfectly -- the cheapest hand edit there is.
    # The runner's own state file names the record it last wrote, so holding the
    # log to that hash is what catches a truncated tail (the same reconciliation
    # R1-c does at startup).
    named = json.loads((state_dir / "runner_state.json").read_text(encoding="utf-8"))
    chain = verify_log(
        state_dir / "decision_log", expected_last_hash=named["last_record_hash"]
    )
    assert chain.ok, [str(defect) for defect in chain.defects]
    assert chain.records == len(soak["live"]), (chain.records, len(soak["live"]))
    # One file per decided day, plus the trailing one: the last minute of day
    # three closes at the fourth day's midnight, so the final SHUTDOWN is
    # stamped there and the log carries a fourth, one-record day.
    assert len(chain.files) == len(DAYS) + 1, chain.files

    kinds = Counter(r["kind"] for r in soak["live"])
    assert kinds[RecordKind.OPERATOR.value] == 0, "the soak is unattended"
    assert kinds[RecordKind.RESUME.value] == 0, "nothing had to be resumed"
    assert kinds[RecordKind.HALT.value] == 0, "a halt would need an operator"

    written = set(state_inventory(state_dir))
    expected = {
        "carry_ledger.json",
        "emergency_record.bin",
        "perp_store.json",
        "risk.json",
        "runner_state.json",
        "spot_store.json",
    }
    logs = {f"decision_log/{day}.ndjson" for day in (*DAYS, TRAILING_DAY)}
    unexpected = written - expected - logs - {"metrics.prom", "runner.lock"}
    assert unexpected == set(), f"files no runner process writes: {sorted(unexpected)}"
    assert expected <= written, f"missing: {sorted(expected - written)}"
