"""R1-d: the demo runner as a continuous service.

The adopted item, verbatim (`docs/master_roadmap_r0_r18.md`, R1-d): "a
continuous READY loop (sleep to the next expected minute close + grace),
graceful SIGTERM deferred past PERSISTENCE, heartbeat every 30 s, persistent
metrics endpoint; the systemd unit becomes a real long-running service;
`RunnerDown` becomes a liveness alert; process-per-pass supervision is removed."

Engineering evidence only. Every minute below is synthetic
(`chimera.demo.fixtures`), the venue is a dry-run one, and nothing here reads a
research artifact.

Time is simulated. `_daemon` takes its wall clock and its sleep as parameters,
and :class:`FakeTime` advances a number instead of waiting, so a test of "wait for
the next minute close" takes milliseconds and a test of "beat at least every 30 s"
can span an hour. Callbacks run inside the fake sleep, at chosen simulated
instants, to do what the world does while a real service waits: the recorder
publishes more minutes, an operator touches the kill switch, systemd sends
SIGTERM.
"""

from __future__ import annotations

import configparser
import functools
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import pytest
import yaml

from chimera import metrics
from chimera.demo import feed as feed_module
from chimera.demo.feed import PERP_MARKET, FeedCursor
from chimera.demo.fixtures import MinuteShape
from chimera.demo.runner import _STATE_NAMES, RunnerState
from chimera.demo.telemetry import RunnerTelemetry
from tests.demo_harness import DAY, NEXT_DAY, campaign_config
from tests.demo_harness import build as _build
from tests.test_demo_cli import written_config
from tools import demo_run

REPO = Path(__file__).resolve().parents[1]
UNIT = REPO / "deploy" / "systemd" / "chimera-demo.service"
MINUTE_MS = 60_000
#: An arbitrary wall instant, deliberately NOT on a minute boundary. The wall
#: clock and the recorded minutes are different clocks; nothing ties them.
T0 = 1_790_000_017.25
GRACE = 5.0  # conf/demo/pvc1.json and section 8.1: `ready_grace_seconds`
#: An ABSOLUTE tolerance for simulated instants. `pytest.approx`'s default is
#: relative, and at T0's magnitude relative means roughly 1800 seconds.
EPS = 1e-3


#: R1-f. Every service here runs an operational clock that has nothing to do
#: with the recorded day -- T0 is two days after it -- and under R1-f's READY gate
#: that is a stale feed. These tests are about the loop's schedule, its signals
#: and its heartbeat, not about staleness, so every harness in this module holds
#: `max_data_delay_s` out of reach (about 31.7 years). Staleness has its own
#: two-sided tests, and the same service properties during a stall, in
#: `tests/test_r1f_real_staleness.py`.
STALENESS_OUT_OF_REACH = {"max_data_delay_s": 1e9}


def build(tmp_path: Path, **kwargs: Any):
    """`tests.demo_harness.build`, with the READY gate's limit out of reach."""
    kwargs.setdefault(
        "config", campaign_config(tmp_path / "state", limits=STALENESS_OUT_OF_REACH)
    )
    return _build(tmp_path, **kwargs)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
class FakeTime:
    """A wall clock that moves only when `sleep` is called, or a test moves it."""

    def __init__(self, start: float = T0) -> None:
        self.t = start
        self.sleeps: list[float] = []
        self.hooks: list[Callable[["FakeTime"], None]] = []

    def clock(self) -> float:
        return self.t

    def wall_ns(self) -> int:
        return int(round(self.t * 1e9))

    def sleep(self, seconds: float) -> None:
        assert seconds > 0, f"a non-positive sleep ({seconds}) is a busy loop"
        self.sleeps.append(seconds)
        self.t += seconds
        for hook in list(self.hooks):
            hook(self)


def present_minutes(count: int) -> dict[str, dict[int, MinuteShape]]:
    """Shapes for a day whose first ``count`` minutes exist, both markets."""
    absent = {slot: MinuteShape(present=False) for slot in range(count, 1440)}
    return {f"um:{DAY}": dict(absent), f"spot:{DAY}": dict(absent)}


def publish(harness, count: int) -> None:
    """The recorder rewriting today's parquet with ``count`` minutes in it."""
    harness.feed.write_days([DAY], shapes=present_minutes(count))


def day_start_ms() -> int:
    return int(pd.Timestamp(DAY, tz="UTC").timestamp() * 1000)


def records(state_dir: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in sorted((state_dir / "decision_log").glob("*.ndjson")):
        out += [json.loads(line) for line in path.read_text("utf-8").splitlines() if line]
    return out


def kinds(state_dir: Path) -> list[str]:
    return [r["kind"] for r in records(state_dir)]


def runner_state(state_dir: Path) -> dict[str, Any]:
    return json.loads((state_dir / "runner_state.json").read_text("utf-8"))


def count_calls(obj: Any, name: str, log: list[float], fake: FakeTime) -> None:
    """Wrap ``obj.name`` so every call records the simulated instant it began."""
    original = getattr(obj, name)

    @functools.wraps(original)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        log.append(fake.t)
        return original(*args, **kwargs)

    setattr(obj, name, wrapped)


def cli_argv(tmp_path: Path, harness) -> list[str]:
    config = written_config(tmp_path, harness)
    return ["--config", str(config), "--root", str(harness.root), "--profile", "TEST"]


def patch_daemon(monkeypatch, fake: FakeTime) -> None:
    """Drive `main`'s service loop on simulated time.

    Injected at `main`, the one place the operational clock enters the process
    (R1-e), so the schedule and the telemetry both run on ``fake``.
    """
    original = demo_run.main
    monkeypatch.setattr(
        demo_run,
        "main",
        functools.partial(original, operational_clock=fake.clock, sleep=fake.sleep),
    )


def installed_sigterm(_fake: FakeTime | None = None) -> None:
    """Deliver SIGTERM through the handler `main` really installed."""
    handler = signal.getsignal(signal.SIGTERM)
    assert callable(handler), "the service installed no SIGTERM handler"
    handler(signal.SIGTERM, None)


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


# --------------------------------------------------------------------------- #
# FeedCursor: a long-lived cursor sees what the recorder publishes later
# --------------------------------------------------------------------------- #
def test_a_long_lived_cursor_sees_minutes_added_to_a_day_it_already_read(tmp_path):
    """The same day rewritten with more minutes becomes visible, no restart."""
    harness = build(tmp_path, shapes=present_minutes(10), start=False)
    cursor = FeedCursor(harness.root, harness.runner.contract)
    first = day_start_ms()
    assert cursor.latest_minute_ms() == first + 9 * MINUTE_MS
    assert cursor.record(PERP_MARKET, first + 15 * MINUTE_MS) is None

    publish(harness, 20)

    assert cursor.latest_minute_ms() == first + 19 * MINUTE_MS
    assert cursor.record(PERP_MARKET, first + 15 * MINUTE_MS) is not None


def test_a_long_lived_cursor_sees_a_day_created_after_its_absence_was_observed(tmp_path):
    """The ABSENCE of a day's file must not be cached for the cursor's life.

    That is the transition a service meets every UTC midnight: the new day's
    parquet does not exist when the cursor first asks for it.
    """
    harness = build(tmp_path, days=(DAY,), start=False)
    cursor = FeedCursor(harness.root, harness.runner.contract)
    next_first = day_start_ms() + 1440 * MINUTE_MS
    assert cursor.record(PERP_MARKET, next_first) is None, "NEXT_DAY is not written yet"
    last_of_day = cursor.latest_minute_ms()

    harness.feed.write_days([NEXT_DAY])

    assert cursor.record(PERP_MARKET, next_first) is not None
    assert cursor.latest_minute_ms() == next_first + 1439 * MINUTE_MS != last_of_day


def test_an_unchanged_day_is_not_read_again(tmp_path, monkeypatch):
    """The control: the cache still caches. Only a changed file is re-read."""
    harness = build(tmp_path, shapes=present_minutes(10), start=False)
    cursor = FeedCursor(harness.root, harness.runner.contract)
    reads: list[Any] = []
    real = pd.read_parquet

    def counting(path, *args, **kwargs):
        reads.append(Path(path))
        return real(path, *args, **kwargs)

    monkeypatch.setattr(feed_module.pd, "read_parquet", counting)
    minute = day_start_ms() + 3 * MINUTE_MS
    for _ in range(5):
        assert cursor.record(PERP_MARKET, minute) is not None
    assert len(reads) == 1

    publish(harness, 11)
    assert cursor.record(PERP_MARKET, minute) is not None
    assert len(reads) == 2


# --------------------------------------------------------------------------- #
# scheduling: the next expected minute close plus grace
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("now", "expected"),
    [
        (600.0, 605.0),  # exactly on a close: this close's grace is still ahead
        (603.0, 605.0),  # started just after a close, inside its grace
        (604.999, 605.0),
        (605.0, 665.0),  # exactly at close + grace: strictly after, so the next
        (606.0, 665.0),  # just after close + grace
        (659.9, 665.0),  # just before the next close
        (845.0, 905.0),  # a pass that overran several closes wakes ONCE, next slot
    ],
)
def test_the_wake_is_the_next_minute_close_plus_grace(now, expected):
    assert demo_run._next_wake(now, GRACE) == pytest.approx(expected)
    assert demo_run._next_wake(now, GRACE) > now


def test_the_grace_is_the_configured_ready_grace(tmp_path):
    """No new timing constant: section 8.1's READY grace, from the config."""
    harness = build(tmp_path, shapes=present_minutes(5), start=False)
    assert harness.runner.config.runner_setting("ready_grace_seconds") == GRACE


# --------------------------------------------------------------------------- #
# A. the continuous READY loop
# --------------------------------------------------------------------------- #
def test_one_process_runs_many_passes_and_each_decides_what_was_published(tmp_path):
    """Three passes in one process. Each sees minutes published while it waited.

    Also the R1-b/R1-c guard: `start()` ran once, so there is exactly one
    STARTUP record and no RECOVERY -- a pass is not an in-process restart.
    """
    harness = build(tmp_path, shapes=present_minutes(10))
    runner, fake, stop = harness.runner, FakeTime(), {}
    passes: list[float] = []
    count_calls(runner, "catch_up", passes, fake)
    cursors: list[int | None] = []
    #: What the world does during the wait after the n-th pass.
    publishes = {1: 14, 2: 19}

    def hook(_fake: FakeTime) -> None:
        n = len(passes)
        if len(cursors) >= n:
            return  # this wait has had its turn
        cursors.append(runner.cursor.last_minute_processed)
        if n in publishes:
            publish(harness, publishes[n])
        else:
            stop["signal"] = int(signal.SIGTERM)

    fake.hooks.append(hook)

    assert (
        demo_run._daemon(runner, stop, clock=fake.clock, sleep=fake.sleep) == demo_run.EXIT_OK
    )

    first = day_start_ms()
    assert cursors == [first + 9 * MINUTE_MS, first + 13 * MINUTE_MS, first + 18 * MINUTE_MS]
    assert len(passes) == 3
    seen = kinds(harness.state_dir)
    assert seen.count("STARTUP") == 1
    assert "RECOVERY" not in seen
    assert "HALT" not in seen
    assert seen[-1] == "SHUTDOWN"


def test_the_service_does_not_exit_after_its_first_pass(tmp_path):
    """The one-pass shape, refused: with no stop request the loop keeps going."""
    harness = build(tmp_path, shapes=present_minutes(5))
    runner, fake, stop = harness.runner, FakeTime(), {}
    passes: list[float] = []
    count_calls(runner, "catch_up", passes, fake)

    def hook(f: FakeTime) -> None:
        if f.t >= T0 + 30 * 60:  # half an hour of simulated waiting
            stop["signal"] = int(signal.SIGTERM)

    fake.hooks.append(hook)
    demo_run._daemon(runner, stop, clock=fake.clock, sleep=fake.sleep)
    assert len(passes) >= 30


# --------------------------------------------------------------------------- #
# K / L. no busy loop, and passes land on close + grace
# --------------------------------------------------------------------------- #
def test_with_nothing_new_the_service_sleeps_to_each_close_plus_grace(tmp_path):
    """Ten simulated minutes, nothing published: ten more passes, never a spin.

    Fails if the loop called the pass again without sleeping (the fake sleep
    refuses a non-positive duration and every pass is timed), and if a pass ran
    anywhere but a minute close plus the grace.
    """
    harness = build(tmp_path, shapes=present_minutes(5))
    runner, fake, stop = harness.runner, FakeTime(), {}
    passes: list[float] = []
    original = runner.catch_up

    def guarded(*args: Any, **kwargs: Any) -> Any:
        # A spinning loop never reaches the fake sleep, so no hook could stop
        # it: refuse the second pass at the same instant instead of hanging.
        assert not passes or fake.t > passes[-1], "a pass ran again without waiting"
        passes.append(fake.t)
        return original(*args, **kwargs)

    runner.catch_up = guarded
    fake.hooks.append(
        lambda f: stop.update(signal=int(signal.SIGTERM)) if f.t >= T0 + 600 else None
    )

    demo_run._daemon(runner, stop, clock=fake.clock, sleep=fake.sleep)

    assert passes[0] == T0
    later = passes[1:]
    assert len(later) == 10
    assert all((t - GRACE) % 60 == pytest.approx(0, abs=EPS) for t in later), later
    assert all(b - a == pytest.approx(60, abs=EPS) for a, b in zip(later, later[1:]))
    assert max(fake.sleeps) <= demo_run.WAIT_SLICE_SECONDS
    assert sum(fake.sleeps) == pytest.approx(fake.t - T0)


def test_a_pass_that_overruns_a_close_wakes_at_the_next_slot_not_in_a_burst(tmp_path):
    """Work that takes 150 s is followed by ONE pass at the next close + grace."""
    harness = build(tmp_path, shapes=present_minutes(5))
    runner, fake, stop = harness.runner, FakeTime(), {}
    passes: list[float] = []
    original = runner.catch_up

    def slow(*args, **kwargs):
        passes.append(fake.t)
        result = original(*args, **kwargs)
        if len(passes) == 2:
            fake.t += 150.0
        return result

    runner.catch_up = slow
    fake.hooks.append(lambda f: stop.update(signal=15) if len(passes) >= 4 else None)
    demo_run._daemon(runner, stop, clock=fake.clock, sleep=fake.sleep)

    second_end = passes[1] + 150.0
    assert passes[2] == pytest.approx(demo_run._next_wake(second_end, GRACE), abs=EPS)
    assert passes[3] - passes[2] == pytest.approx(60, abs=EPS)


# --------------------------------------------------------------------------- #
# C / D. SIGTERM
# --------------------------------------------------------------------------- #
def test_sigterm_while_waiting_stops_promptly_and_decides_nothing_more(
    tmp_path, monkeypatch, capsys
):
    """Idle: exit 0 within one wait slice, SHUTDOWN last, no new decision."""
    harness = build(tmp_path, shapes=present_minutes(8), start=False)
    fake = FakeTime()
    patch_daemon(monkeypatch, fake)
    marks: dict[str, Any] = {}

    def hook(f: FakeTime) -> None:
        if "at" not in marks and f.t >= T0 + 20:
            marks["at"] = f.t
            marks["records"] = len(records(harness.state_dir))
            installed_sigterm()

    fake.hooks.append(hook)
    argv = cli_argv(tmp_path, harness)
    assert demo_run.main(argv + ["run", "--allow-dirty"]) == demo_run.EXIT_OK

    assert fake.t - marks["at"] <= demo_run.WAIT_SLICE_SECONDS
    after = records(harness.state_dir)[marks["records"] :]
    assert [r["kind"] for r in after] == ["SHUTDOWN"]
    assert after[0]["operator"]["note"] == "stopped by SIGTERM"
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["state"] == "SHUTDOWN"


def _fire_in_persistence(monkeypatch, deliver: Callable[[], None]) -> dict[str, Any]:
    """Deliver a stop request the first time any runner enters PERSISTENCE."""
    seen: dict[str, Any] = {}
    original = RunnerTelemetry.on_state

    def on_state(self, state: str) -> None:
        original(self, state)
        if state == RunnerState.PERSISTENCE.value and "fired" not in seen:
            seen["fired"] = True
            deliver()

    monkeypatch.setattr(RunnerTelemetry, "on_state", on_state)
    return seen


def _assert_the_minute_in_hand_completed(state_dir: Path) -> None:
    log = records(state_dir)
    tail = [r["kind"] for r in log[-2:]]
    assert tail == ["DECISION", "SHUTDOWN"], tail
    assert "HALT" not in [r["kind"] for r in log], "SIGTERM is not a halt reason"
    decided = log[-2]["minute"]
    persisted = runner_state(state_dir)
    assert persisted["last_record_hash"] == log[-1]["record_hash"]
    minute_ms = persisted["last_minute_processed"]
    assert pd.Timestamp(minute_ms, unit="ms", tz="UTC").strftime("%Y-%m-%dT%H:%M") in decided


def test_sigterm_during_persistence_lets_the_minute_complete_then_stops(tmp_path, monkeypatch):
    """The deferral, against the runner's own PERSISTENCE state.

    The signal arrives after `tick` has entered PERSISTENCE and before any of
    its writes. The minute's ledger, DECISION record and runner state are all
    written; the service then stops before the NEXT minute (the fixture leaves
    two more pending, so stopping between minutes is visible too); nothing is
    recorded as a halt; and a fresh process starts cleanly from it with no
    RECOVERY -- which is what a torn persistence would have produced.
    """
    harness = build(tmp_path, shapes=present_minutes(10), start=False)
    fake = FakeTime()
    patch_daemon(monkeypatch, fake)
    seen = _fire_in_persistence(monkeypatch, installed_sigterm)
    argv = cli_argv(tmp_path, harness)

    assert demo_run.main(argv + ["run", "--allow-dirty"]) == demo_run.EXIT_OK
    assert seen.get("fired")
    _assert_the_minute_in_hand_completed(harness.state_dir)
    first = day_start_ms()
    # 7 SKIPPED_STALE, then the first of the three decidable minutes, and stop.
    assert runner_state(harness.state_dir)["last_minute_processed"] == first + 7 * MINUTE_MS
    assert fake.sleeps == [], "a stop requested mid-pass does not wait for the next close"

    before = len(records(harness.state_dir))
    monkeypatch.undo()
    assert demo_run.main(argv + ["run", "--once", "--allow-dirty"]) == demo_run.EXIT_OK
    restarted = [r["kind"] for r in records(harness.state_dir)[before:]]
    assert restarted[0] == "STARTUP"
    assert "RECOVERY" not in restarted and "HALT" not in restarted


@pytest.mark.skipif(sys.platform == "win32", reason="os.kill(SIGTERM) terminates on Windows")
def test_a_real_sigterm_during_persistence_is_deferred(tmp_path, monkeypatch):
    """The same property with a real signal, delivered by the kernel."""
    harness = build(tmp_path, shapes=present_minutes(10), start=False)
    fake = FakeTime()
    patch_daemon(monkeypatch, fake)
    seen = _fire_in_persistence(monkeypatch, lambda: os.kill(os.getpid(), signal.SIGTERM))
    argv = cli_argv(tmp_path, harness)

    assert demo_run.main(argv + ["run", "--allow-dirty"]) == demo_run.EXIT_OK
    assert seen.get("fired")
    _assert_the_minute_in_hand_completed(harness.state_dir)


def test_the_service_restores_the_signal_handlers_it_replaced(tmp_path, monkeypatch):
    harness = build(tmp_path, shapes=present_minutes(5), start=False)
    fake = FakeTime()
    patch_daemon(monkeypatch, fake)
    fake.hooks.append(lambda f: installed_sigterm())
    before = (signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT))
    demo_run.main(cli_argv(tmp_path, harness) + ["run", "--allow-dirty"])
    assert (signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)) == before


def test_the_bounded_forms_install_no_handler(tmp_path, monkeypatch):
    """`--once` behaves as `run` always did, default signal disposition included."""
    harness = build(tmp_path, shapes=present_minutes(5), start=False)
    installed: list[Any] = []
    monkeypatch.setattr(demo_run, "_install_stop_handlers", lambda stop: installed.append(1))
    argv = cli_argv(tmp_path, harness)
    assert demo_run.main(argv + ["run", "--once", "--allow-dirty"]) == demo_run.EXIT_OK
    assert installed == []


# --------------------------------------------------------------------------- #
# E / F. heartbeat every 30 s: waiting, and during in-flight work
# --------------------------------------------------------------------------- #
@pytest.fixture
def beats(monkeypatch) -> list[float]:
    """Every value written to `chimera_demo_heartbeat_timestamp`, in order."""
    written: list[float] = []
    monkeypatch.setattr(metrics.DEMO_HEARTBEAT, "set", written.append)
    return written


def _telemetry_on(fake: FakeTime, harness_dir: Path) -> RunnerTelemetry:
    return RunnerTelemetry(
        state_dir=harness_dir, states=_STATE_NAMES, rules=("R1_carry",), wall_ns=fake.wall_ns
    )


def _max_gap(values: list[float]) -> float:
    return max(b - a for a, b in zip(values, values[1:]))


def test_the_heartbeat_beats_at_most_30s_apart_while_the_service_waits(tmp_path, beats):
    fake = FakeTime()
    harness = build(
        tmp_path, shapes=present_minutes(5), telemetry=_telemetry_on(fake, tmp_path)
    )
    stop: dict[str, int] = {}
    fake.hooks.append(lambda f: stop.update(signal=15) if f.t >= T0 + 5 * 60 else None)
    beats.clear()  # start() beat at T0; the wait is what is measured

    demo_run._daemon(harness.runner, stop, clock=fake.clock, sleep=fake.sleep)

    # Absolute tolerances throughout: at T0's magnitude pytest.approx's default
    # RELATIVE tolerance is about half an hour, which would pass anything.
    waiting = [b for b in beats if b >= T0 - EPS]
    assert waiting[0] == pytest.approx(T0, abs=EPS), "the wait beats as it begins"
    assert len({round(b) for b in waiting}) >= 5 * 60 / demo_run.HEARTBEAT_SECONDS
    assert _max_gap(waiting) <= demo_run.HEARTBEAT_SECONDS + EPS
    assert waiting[-1] >= fake.t - demo_run.HEARTBEAT_SECONDS - EPS
    # And it stops with the service: no thread goes on beating after the loop.
    count = len(beats)
    fake.t += 3600
    time.sleep(0.05)
    assert len(beats) == count


def test_the_heartbeat_stays_within_30s_through_a_pass_that_works_for_many_minutes(
    tmp_path, beats
):
    """Correction 2's witness: progress, not only waiting, keeps it fresh.

    One pass over 200 published minutes -- 197 SKIPPED_STALE and three
    decisions -- where every durable write costs 12 simulated seconds, so the
    pass alone spans about forty simulated minutes. Every gap between beats stays
    within the cadence, because each processed minute passes a main-thread
    checkpoint: a state change inside `tick`, or the committed record of a
    minute that is not decided. Remove the record-time beat and the backlog runs
    for half an hour with no beat at all.
    """
    fake = FakeTime()
    harness = build(
        tmp_path, shapes=present_minutes(200), telemetry=_telemetry_on(fake, tmp_path)
    )
    runner, stop = harness.runner, {}
    original = runner.save_state

    def slow_save() -> None:
        fake.t += 12.0
        original()

    runner.save_state = slow_save
    fake.hooks.append(lambda f: stop.update(signal=15) if f.t >= T0 + 30 * 60 else None)
    beats.clear()

    demo_run._daemon(runner, stop, clock=fake.clock, sleep=fake.sleep)

    assert fake.t - T0 > 30 * 60, "the pass alone worked for half an hour of simulated time"
    assert fake.sleeps == [] or fake.sleeps[0] <= demo_run.WAIT_SLICE_SECONDS
    assert kinds(harness.state_dir).count("SKIPPED_STALE") == 197
    assert _max_gap([T0, *beats]) <= demo_run.HEARTBEAT_SECONDS + EPS


# --------------------------------------------------------------------------- #
# G. the metrics endpoint: started once, alive through the wait
# --------------------------------------------------------------------------- #
def test_the_metrics_endpoint_is_started_once_for_the_process(tmp_path, monkeypatch):
    harness = build(tmp_path, shapes=present_minutes(5), start=False)
    fake = FakeTime()
    patch_daemon(monkeypatch, fake)
    started: list[int] = []
    monkeypatch.setattr(metrics, "serve_metrics", started.append)
    passes = {"n": 0}
    original = demo_run.DemoRunner.catch_up

    def counting(self, *args, **kwargs):
        passes["n"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(demo_run.DemoRunner, "catch_up", counting)
    fake.hooks.append(lambda f: installed_sigterm() if passes["n"] >= 3 else None)

    argv = cli_argv(tmp_path, harness)
    assert demo_run.main(argv + ["run", "--allow-dirty", "--metrics-port", "9"]) == 0
    assert passes["n"] == 3
    assert started == [9]


def test_the_metrics_endpoint_answers_during_the_wait_between_passes(tmp_path, monkeypatch):
    """A real listening endpoint, scraped while the service waits, twice."""
    harness = build(tmp_path, shapes=present_minutes(5), start=False)
    fake = FakeTime()
    patch_daemon(monkeypatch, fake)
    port = free_port()
    scrapes: list[str] = []
    passes = {"n": 0}
    original = demo_run.DemoRunner.catch_up

    def counting(self, *args, **kwargs):
        passes["n"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(demo_run.DemoRunner, "catch_up", counting)

    def hook(f: FakeTime) -> None:
        if passes["n"] in (2, 3) and len(scrapes) < passes["n"] - 1:
            url = f"http://127.0.0.1:{port}/metrics"
            scrapes.append(urllib.request.urlopen(url, timeout=5).read().decode())
        if passes["n"] >= 3 and len(scrapes) == 2:
            installed_sigterm()

    fake.hooks.append(hook)
    argv = cli_argv(tmp_path, harness)
    assert demo_run.main(argv + ["run", "--allow-dirty", "--metrics-port", str(port)]) == 0
    assert len(scrapes) == 2
    for body in scrapes:
        assert "chimera_demo_heartbeat_timestamp" in body
        assert 'chimera_demo_state{state="READY"} 1.0' in body


# --------------------------------------------------------------------------- #
# J. exit semantics: a halt stops the process, a failure is not swallowed
# --------------------------------------------------------------------------- #
def test_a_halt_during_a_pass_exits_3_and_does_not_keep_looping(tmp_path, capsys):
    """`RestartPreventExitStatus=3` depends on this: a halted service exits 3."""
    harness = build(tmp_path, shapes=present_minutes(8))
    runner, fake, stop = harness.runner, FakeTime(), {}
    passes: list[float] = []
    count_calls(runner, "catch_up", passes, fake)

    def pull_the_switch(f: FakeTime) -> None:
        if len(passes) == 1 and not (harness.state_dir / "KILL_SWITCH").exists():
            (harness.state_dir / "KILL_SWITCH").write_text("", encoding="utf-8")
            publish(harness, 9)

    fake.hooks.append(pull_the_switch)
    assert demo_run._daemon(runner, stop, clock=fake.clock, sleep=fake.sleep) == 3
    assert len(passes) == 2
    assert kinds(harness.state_dir)[-2:] == ["HALT", "SHUTDOWN"]
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["state"] == "HALT" and out["reason"] == "kill_switch"


def test_a_start_into_halt_still_exits_3_without_entering_the_loop(tmp_path, monkeypatch):
    harness = build(tmp_path, shapes=present_minutes(5), start=False)
    (harness.state_dir).mkdir(parents=True, exist_ok=True)
    (harness.state_dir / "KILL_SWITCH").write_text("", encoding="utf-8")
    entered: list[Any] = []
    monkeypatch.setattr(demo_run, "_daemon", lambda *a, **k: entered.append(1) or 0)
    argv = cli_argv(tmp_path, harness)
    assert demo_run.main(argv + ["run", "--allow-dirty"]) == demo_run.EXIT_HALTED
    assert entered == []


def test_an_unexpected_failure_in_a_pass_propagates_rather_than_looking_healthy(tmp_path):
    harness = build(tmp_path, shapes=present_minutes(5))
    runner, fake = harness.runner, FakeTime()
    calls = {"n": 0}
    original = runner.catch_up

    def failing(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("disk went away")
        return original(*args, **kwargs)

    runner.catch_up = failing
    with pytest.raises(OSError, match="disk went away"):
        demo_run._daemon(runner, {}, clock=fake.clock, sleep=fake.sleep)
    assert "SHUTDOWN" not in kinds(harness.state_dir), "a crash is not a clean stop"


# --------------------------------------------------------------------------- #
# the single-writer lock
# --------------------------------------------------------------------------- #
def _state_bytes(state_dir: Path) -> dict[str, bytes]:
    return {
        p.relative_to(state_dir).as_posix(): p.read_bytes()
        for p in sorted(state_dir.rglob("*"))
        if p.is_file() and p.name != demo_run.LOCK_NAME
    }


def test_a_second_writer_is_refused_while_the_service_runs(tmp_path, monkeypatch, capsys):
    """`flatten` beside a live service would fork the chain; it is refused.

    `status` still answers, because it only reads. After the service stops, the
    lock is free again (the control).
    """
    harness = build(tmp_path, shapes=present_minutes(8), start=False)
    fake = FakeTime()
    patch_daemon(monkeypatch, fake)
    argv = cli_argv(tmp_path, harness)
    seen: dict[str, Any] = {}

    def hook(f: FakeTime) -> None:
        if seen:
            return
        before = _state_bytes(harness.state_dir)
        seen["flatten"] = demo_run.main(argv + ["flatten", "--note", "beside the service"])
        seen["unchanged"] = _state_bytes(harness.state_dir) == before
        seen["status"] = demo_run.main(argv + ["status"])
        installed_sigterm()

    fake.hooks.append(hook)
    assert demo_run.main(argv + ["run", "--allow-dirty"]) == demo_run.EXIT_OK
    assert seen["flatten"] == demo_run.EXIT_REFUSED
    assert seen["unchanged"]
    assert seen["status"] == demo_run.EXIT_OK
    assert "runner.lock" in capsys.readouterr().err

    with demo_run._single_writer(harness.state_dir):
        pass


def test_the_lock_refuses_a_second_holder_and_frees_on_release(tmp_path):
    state_dir = tmp_path / "state"
    with demo_run._single_writer(state_dir):
        with pytest.raises(demo_run.StateDirectoryBusy):
            with demo_run._single_writer(state_dir):
                pass
    with demo_run._single_writer(state_dir):
        pass


@pytest.mark.skipif(sys.platform == "win32", reason="the service's signals are POSIX")
def test_the_service_process_stays_up_refuses_a_second_writer_and_stops_on_sigterm(tmp_path):
    """The operational witness: a real process, a real lock, a real SIGTERM."""
    harness = build(tmp_path, shapes=present_minutes(8), start=False)
    argv = cli_argv(tmp_path, harness)
    base = [sys.executable, "-m", "tools.demo_run", *argv]
    service = subprocess.Popen(
        [*base, "run", "--allow-dirty"],
        cwd=REPO,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        last = day_start_ms() + 7 * MINUTE_MS
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            path = harness.state_dir / "runner_state.json"
            if (
                path.is_file()
                and runner_state(harness.state_dir)["last_minute_processed"] == last
            ):
                break
            assert service.poll() is None, service.communicate()
            time.sleep(0.2)
        else:
            pytest.fail("the service never caught up")
        time.sleep(1.0)
        assert service.poll() is None, "the service exited after its first pass"

        second = subprocess.run(
            [*base, "flatten", "--note", "beside the service"],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert second.returncode == demo_run.EXIT_REFUSED, second.stderr
        assert "runner.lock" in second.stderr

        service.send_signal(signal.SIGTERM)
        out, err = service.communicate(timeout=30)
    finally:
        if service.poll() is None:
            service.kill()
    assert service.returncode == demo_run.EXIT_OK, err
    assert kinds(harness.state_dir)[-1] == "SHUTDOWN"


# --------------------------------------------------------------------------- #
# I. systemd and compose: a long-running service, not a scheduler
# --------------------------------------------------------------------------- #
def _unit() -> configparser.ConfigParser:
    parser = configparser.ConfigParser(strict=False, interpolation=None)
    parser.optionxform = str  # type: ignore[assignment,method-assign]
    parser.read_string(UNIT.read_text(encoding="utf-8"))
    return parser


def test_the_unit_is_a_long_running_service_and_restarts_only_on_failure():
    unit = _unit()
    service = unit["Service"]
    assert service["Type"] == "simple"
    assert service["Restart"] == "on-failure", "exit is no longer the scheduler"
    assert service["RestartPreventExitStatus"] == "3", "a halt must not auto-restart"
    assert service["KillSignal"] == "SIGTERM"
    assert int(service["TimeoutStopSec"]) >= 60
    exec_start = " ".join(service["ExecStart"].split())
    assert exec_start.endswith("run --metrics-port 9103")
    assert "--once" not in exec_start and "--replay" not in exec_start
    # A crash loop ends; it does not retry for ever.
    assert int(unit["Unit"]["StartLimitBurst"]) >= 1
    assert unit["Unit"]["StartLimitIntervalSec"] not in ("0", "infinity")


def test_the_unit_no_longer_describes_a_bounded_pass():
    text = UNIT.read_text(encoding="utf-8")
    assert "ONE BOUNDED CATCH-UP PASS" not in text
    directives = [line.strip() for line in text.splitlines() if not line.startswith("#")]
    assert "Restart=always" not in directives
    assert "cannot be read as liveness" not in text


@pytest.mark.skipif(shutil.which("systemd-analyze") is None, reason="needs systemd-analyze")
def test_systemd_accepts_the_unit(tmp_path):
    """`systemd-analyze verify` on a copy whose ExecStart exists on this host."""
    text = UNIT.read_text(encoding="utf-8")
    lines, skipping = [], False
    for line in text.splitlines():
        if line.startswith("ExecStart="):
            lines.append("ExecStart=/bin/true")
            skipping = line.endswith("\\")
            continue
        if skipping:
            skipping = line.endswith("\\")
            continue
        lines.append(line)
    copy = tmp_path / "chimera-demo.service"
    copy.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = subprocess.run(
        ["systemd-analyze", "verify", str(copy)], capture_output=True, text=True, timeout=60
    )
    problems = [
        line
        for line in (result.stdout + result.stderr).splitlines()
        if "chimera-demo.service" in line and "Unknown" in line
    ]
    assert not problems, result.stdout + result.stderr


def test_compose_gives_the_service_the_same_time_to_stop():
    compose = yaml.safe_load((REPO / "docker-compose.yml").read_text(encoding="utf-8"))
    demo = compose["services"]["demo"]
    assert demo["stop_grace_period"] == "120s"
    assert "--once" not in demo["command"] and "--replay" not in demo["command"]


# --------------------------------------------------------------------------- #
# H. RunnerDown is liveness, and the documents say so
# --------------------------------------------------------------------------- #
def test_runner_down_fires_for_a_dead_process_as_well_as_a_wedged_one():
    rules = yaml.safe_load((REPO / "conf" / "alerts_demo.yml").read_text(encoding="utf-8"))
    (rule,) = [
        r for g in rules["groups"] for r in g["rules"] if r.get("alert") == "RunnerDown"
    ]
    expr = rule["expr"]
    # Wedged: scraped, but the heartbeat stopped moving.
    assert "time() - chimera_demo_heartbeat_timestamp > 120" in expr
    # Dead: a failed scrape marks the series stale, so the line above returns
    # nothing at all; only an absence test can fire then.
    assert "absent_over_time(chimera_demo_heartbeat_timestamp[2m])" in expr
    assert rule["for"] == "0m"
    assert demo_run.HEARTBEAT_SECONDS * 4 <= 120


def test_the_runbook_no_longer_says_the_runner_is_a_bounded_pass():
    text = (REPO / "docs" / "demo_runbook.md").read_text(encoding="utf-8")
    assert "is one bounded catch-up pass, not a daemon" not in text
    assert "cannot be read as liveness" not in text
    assert "shows the unit inactive between passes" not in text
