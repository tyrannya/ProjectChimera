"""R1-e: the decision clock, the operational clock, and the wall between them.

The adopted item, verbatim (`docs/master_roadmap_r0_r18.md`, R1-e): "RiskEngine
and executors refuse construction without an injected clock under CAMPAIGN/SOAK
profiles; no `time.time` default reachable from the demo path; the daemon's
operational clock (used only for staleness, heartbeats and the stall tick) is
itself injected and distinct from the decision clock; a clock-hostility test."

The two clocks:

* DECISION -- `RunnerClock`, advanced only by recorded instants. The only clock
  Aegis and both executors are built with; it may decide things.
* OPERATIONAL -- host wall time, composed once in `tools/demo_run.main` and handed
  to the service loop and the telemetry. It schedules and it reports ages and
  heartbeats; it may not decide anything. (R1-f's staleness check and stall tick
  will read it too; neither exists yet and nothing here builds them.)

Engineering evidence only: every minute is synthetic (`chimera.demo.fixtures`),
the venue is a dry-run one, and nothing here reads a research artifact.
"""

from __future__ import annotations

import ast
import inspect
import json
import signal
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import pytest

from chimera import metrics
from chimera.carry.factory import PERP_SYMBOL, SPOT_SYMBOL, build_hedged_position
from chimera.demo.config import ConfigProfile, parse_demo_config
from chimera.demo.risk_wiring import RiskWiringError, build_risk_engine
from chimera.demo.runner import _STATE_NAMES, DemoRunner, RunnerState
from chimera.demo.telemetry import RunnerTelemetry
from chimera.futures.executor import FlattenCause, FuturesExecutor
from chimera.risk import RiskEngine
from tests.demo_harness import CAPITAL, DAY, REPO, build, campaign_config
from tests.test_demo_cli import written_config
from tests.test_r1d_daemon_service import (
    EPS,
    FakeTime,
    installed_sigterm,
    present_minutes,
    publish,
)
from tools import demo_run

#: The builtin, captured before any test here rebinds `time.time`.
HOST_TIME = time.time
PROFILES = ("CAMPAIGN", "SOAK", "TEST")
NS = 1_000_000_000
DAY_START_NS = pd.Timestamp(DAY, tz="UTC").value
#: Two host clocks no honest decision could survive: 2100-01-01 and 1971-02-05.
HOST_FUTURE = 4_102_444_800.0
HOST_PAST = 400 * 86_400.0
#: Two operational clocks three hundred million seconds (about 9.5 years) apart.
WALL_A = 1_790_000_017.25
WALL_B = WALL_A + 300_000_000.0
#: More passes than either service test drives; reaching it means a busy loop.
MAX_PASSES = 20
DEMO_PACKAGES = (REPO / "chimera" / "demo", REPO / "chimera" / "carry")
DEMO_CLI = REPO / "tools" / "demo_run.py"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def forbidden_decision_time() -> float:
    raise AssertionError("a demo-path component fell back to its host-time default")


def hostile_host(mp: pytest.MonkeyPatch, seconds: float) -> None:
    """Make the host's clock hostile, by both routes a fallback could take.

    ``time.time()`` looked up when it is called -- a loop or an emitter reading
    the host for itself -- answers ``seconds``. A DEFAULT argument, though, was
    bound when its class was defined, so rebinding ``time.time`` never reaches
    ``RiskEngine``'s or ``FuturesExecutor``'s own ``clock=time.time``: those are
    replaced on the constructors with a clock that RAISES, because on the demo
    path reaching one is the defect itself. The swap must find each fallback,
    or the hostility would be a no-op that proves nothing.
    """
    mp.setattr(time, "time", lambda: seconds)
    mp.setattr(time, "time_ns", lambda: int(seconds * NS))
    for init in (RiskEngine.__init__, FuturesExecutor.__init__):
        defaults = init.__defaults__ or ()
        assert defaults.count(HOST_TIME) == 1, f"{init.__qualname__}: its fallback moved"
        mp.setattr(
            init,
            "__defaults__",
            tuple(forbidden_decision_time if d is HOST_TIME else d for d in defaults),
        )


def config_for(profile: str, state_dir: Path):
    """A NATIVE configuration of ``profile``.

    CAMPAIGN is the committed `conf/demo/pvc1.json` itself, which parses as a
    CAMPAIGN; it carries no rules, which is all the factories need.
    """
    if profile == "CAMPAIGN":
        payload = json.loads((REPO / "conf" / "demo" / "pvc1.json").read_text("utf-8"))
        payload["runner"] = {"state_dir": str(state_dir)}
        return parse_demo_config(payload, expected_profile=ConfigProfile.CAMPAIGN)
    return campaign_config(state_dir, profile=profile)


def runner_on(tmp_path: Path, profile: str, **kwargs: Any):
    """A harness runner whose configuration is on ``profile``, not yet started.

    CAMPAIGN is forced the way `test_demo_runner` and `test_demo_cli` already
    force it: the committed CAMPAIGN configuration cannot carry rules while its
    `protocol_hash` is null, and that is a separate R1 item. Forced BEFORE
    `start()`, so the factories are handed a CAMPAIGN configuration.
    """
    config = campaign_config(
        tmp_path / "state", profile="TEST" if profile == "CAMPAIGN" else profile
    )
    harness = build(tmp_path, config=config, start=False, **kwargs)
    if profile == "CAMPAIGN":
        object.__setattr__(harness.runner.config, "profile", ConfigProfile.CAMPAIGN)
    assert harness.runner.config.profile.value == profile
    return harness


def legs(runner: DemoRunner):
    return (
        ("spot", runner.position.spot, SPOT_SYMBOL),
        ("perp", runner.position.perp, PERP_SYMBOL),
    )


def iso(seconds: float) -> str:
    return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# A. the required injection, and all three consumers really use it
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("profile", PROFILES)
def test_aegis_and_both_executors_run_on_the_decision_clock_under_a_hostile_host(
    tmp_path, profile
):
    """What each consumer PERSISTS is the recorded instant, not the host's.

    The host says 2100 and every fallback raises. Aegis's order window, its
    `risk.json` stamp and each executor's flatten stamp must all be the
    runner's recorded instant -- which is what an injection that reached all
    three looks like from the outside.
    """
    with pytest.MonkeyPatch.context() as mp:
        hostile_host(mp, HOST_FUTURE)
        harness = runner_on(tmp_path, profile)
        assert harness.runner.start() is RunnerState.READY
        runner = harness.runner
        instant_ns = runner.clock.now_ns + 7 * 60 * NS + 123_456_789
        runner.clock.observe(instant_ns)

        runner.risk.record_order()
        for name, executor, symbol in legs(runner):
            executor.emergency_flatten(symbol, FlattenCause.RISK_HALT, Decimal("1"))
            assert executor.store.state.flatten_reasons[-1]["at"] == iso(instant_ns / NS), name

    assert runner.risk.state.order_times[-1] == pytest.approx(instant_ns / NS)
    persisted = json.loads((harness.state_dir / "risk.json").read_text("utf-8"))
    assert persisted["updated_at"] == iso(instant_ns / NS)
    assert persisted["day"] == DAY


def test_the_production_loader_builds_aegis_and_both_executors_on_the_decision_clock(
    tmp_path, monkeypatch
):
    """The witness above, but with the runner built by `tools.demo_run._load`.

    The test above composes its runner from the harness's own factories, so it
    cannot see `_load`'s: a `_load` that handed its operational clock to Aegis,
    or to both executors, passed every committed test (R1-e independent review,
    F2). Here the production loader is given an operational clock in 2036, the
    host says 2100 and every class fallback raises. What Aegis and both executors
    persist must still be the recorded instant, while the heartbeat -- the
    operational clock's legitimate consumer -- reads 2036, so the operational
    clock was reachable and simply not used to decide.
    """
    import argparse

    hostile_host(monkeypatch, HOST_FUTURE)
    harness = runner_on(tmp_path, "SOAK")
    args = argparse.Namespace(
        config=written_config(tmp_path, harness),
        root=harness.root,
        profile="SOAK",
        command="run",
    )
    beats: list[float] = []
    monkeypatch.setattr(metrics.DEMO_HEARTBEAT, "set", beats.append)

    runner = demo_run._load(args, operational_clock=lambda: WALL_B)
    assert runner.start(allow_dirty=True) is RunnerState.READY
    instant_ns = runner.clock.now_ns + 7 * 60 * NS + 123_456_789
    runner.clock.observe(instant_ns)
    runner.risk.record_order()
    for name, executor, symbol in legs(runner):
        executor.emergency_flatten(symbol, FlattenCause.RISK_HALT, Decimal("1"))
        assert executor.store.state.flatten_reasons[-1]["at"] == iso(instant_ns / NS), name

    assert runner.risk.state.order_times[-1] == pytest.approx(instant_ns / NS)
    persisted = json.loads((harness.state_dir / "risk.json").read_text("utf-8"))
    assert persisted["updated_at"] == iso(instant_ns / NS)
    assert persisted["day"] == DAY
    assert beats, "the operational clock never reached its consumer"
    assert all(abs(beat - WALL_B) <= EPS for beat in beats), beats


# --------------------------------------------------------------------------- #
# B. a missing decision clock is refused, on every profile, before any write
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("profile", PROFILES)
def test_the_risk_engine_is_refused_without_a_decision_clock(tmp_path, profile):
    state_dir = tmp_path / "state"
    config = config_for(profile, state_dir)
    assert config.profile.value == profile

    with pytest.raises(TypeError, match="clock"):
        build_risk_engine(config, capital=CAPITAL, state_dir=state_dir)  # type: ignore[call-arg]
    with pytest.raises(RiskWiringError, match=profile):
        build_risk_engine(config, capital=CAPITAL, state_dir=state_dir, clock=None)  # type: ignore[arg-type]
    assert not (state_dir / "risk.json").exists(), "a refused build wrote the risk state"

    # The control: the same call with the clock builds, and seeds, on that clock.
    engine = build_risk_engine(config, capital=CAPITAL, state_dir=state_dir, clock=lambda: 1e9)
    assert engine.state.equity == pytest.approx(float(CAPITAL))
    assert json.loads((state_dir / "risk.json").read_text("utf-8"))["updated_at"] == iso(1e9)


def test_the_hedged_position_is_refused_without_a_decision_clock(tmp_path):
    state_dir = tmp_path / "state"
    risk = build_risk_engine(
        config_for("SOAK", state_dir), capital=CAPITAL, state_dir=state_dir, clock=lambda: 1e9
    )
    with pytest.raises(TypeError, match="clock"):
        build_hedged_position(risk=risk, capital=CAPITAL, state_dir=state_dir)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="decision clock"):
        build_hedged_position(risk=risk, capital=CAPITAL, state_dir=state_dir, clock=None)  # type: ignore[arg-type]
    assert not (state_dir / "spot_store.json").exists()
    assert not (state_dir / "perp_store.json").exists()


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("which", ["risk", "position"])
def test_a_runner_whose_factory_drops_the_clock_does_not_start(tmp_path, profile, which):
    """The realistic regression: a factory in `_load` loses its ``clock=clock``.

    The governed path must refuse at `start()` rather than build the engine or
    the executors on host time.
    """
    harness = runner_on(tmp_path, profile)
    runner, cfg, state_dir = harness.runner, harness.runner.config, harness.state_dir
    if which == "risk":
        runner._risk_factory = lambda clock: build_risk_engine(  # type: ignore[call-arg]
            cfg, capital=CAPITAL, state_dir=state_dir
        )
    else:
        runner._position_factory = lambda risk, clock: build_hedged_position(  # type: ignore[call-arg]
            risk=risk, capital=CAPITAL, state_dir=state_dir
        )
    with pytest.raises(TypeError, match="clock"):
        runner.start()
    assert not (state_dir / "spot_store.json").exists()
    if which == "risk":
        assert not (state_dir / "risk.json").exists()


# --------------------------------------------------------------------------- #
# C. hostility: two services, two hostile hosts, two operational clocks
# --------------------------------------------------------------------------- #
def _service(where: Path, profile: str, *, host: float, wall: float, lurch: float) -> dict:
    """One continuous service over the same recorded day, on hostile time.

    The world is driven by PASS COUNT, never by time: the recorder publishes 14
    and then 19 minutes after the first and second passes and SIGTERM follows
    the third, so both arms see the same recorded input whatever their clocks
    do. After the first pass the operational clock ``lurch``-es -- forward or
    backward -- as a host clock stepped by an operator or by NTP would.
    """
    beats: list[float] = []
    ages: list[float] = []
    with pytest.MonkeyPatch.context() as mp:
        hostile_host(mp, host)
        mp.setattr(metrics.DEMO_HEARTBEAT, "set", beats.append)
        mp.setattr(metrics.DEMO_LAST_MINUTE_AGE, "set", ages.append)
        fake = FakeTime(wall)
        telemetry = RunnerTelemetry(
            state_dir=where / "state",
            states=_STATE_NAMES,
            rules=("R1_carry", "R2_frozen_logistic", "R3_daily_momentum"),
            wall_ns=fake.wall_ns,
        )
        harness = runner_on(where, profile, shapes=present_minutes(10), telemetry=telemetry)
        assert harness.runner.start() is RunnerState.READY
        stop: dict[str, int] = {}
        passes: list[float] = []
        catch_up = harness.runner.catch_up

        def counted(*args: Any, **kwargs: Any) -> Any:
            # A loop that never waits (one reading a host frozen PAST its wake)
            # never reaches the fake sleep, so no hook could stop it: fail here.
            assert len(passes) < MAX_PASSES, "the service passed again without waiting"
            passes.append(fake.t)
            return catch_up(*args, **kwargs)

        harness.runner.catch_up = counted  # type: ignore[method-assign]
        acted: set[int] = set()

        def world(f: FakeTime) -> None:
            if len(f.sleeps) > 20_000:  # a loop that never wakes must not hang the suite
                stop["signal"] = int(signal.SIGTERM)
            n = len(passes)
            if n in acted:
                return
            acted.add(n)
            if n == 1:
                publish(harness, 14)
                f.t += lurch
            elif n == 2:
                publish(harness, 19)
            else:
                stop["signal"] = int(signal.SIGTERM)

        fake.hooks.append(world)
        code = demo_run._daemon(harness.runner, stop, clock=fake.clock, sleep=fake.sleep)

    state = {
        p.relative_to(harness.state_dir).as_posix(): p.read_bytes()
        for p in sorted(harness.state_dir.rglob("*"))
        if p.is_file()
    }
    return {
        "code": code,
        "passes": len(passes),
        "state": state,
        "beats": beats,
        "ages": ages,
        "wall": (wall, fake.t),
        "sleeps": fake.sleeps,
    }


def _records(state: dict[str, bytes]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name in sorted(state):
        if name.startswith("decision_log/"):
            out += [json.loads(line) for line in state[name].decode().splitlines() if line]
    return out


@pytest.mark.parametrize("profile", ["CAMPAIGN", "SOAK"])
def test_hostile_host_and_operational_clocks_cannot_move_a_decision(tmp_path, profile):
    """Same recorded input, radically different time, identical evidence.

    Arm A: host in 2100, operational clock at `WALL_A`, leaping a day forward.
    Arm B: host in 1971, operational clock 300 million seconds later, stepping
    90 s BACKWARDS. Every fallback to a class default raises in both.

    Every persisted byte -- decision log, `risk.json`, both stores, the ledger,
    the runner state -- must be identical. And the operational consumers must
    visibly have run on each arm's OWN operational clock, so the equality
    cannot be the trivial one of a clock nobody read.
    """
    a = _service(tmp_path / "a", profile, host=HOST_FUTURE, wall=WALL_A, lurch=86_400.0)
    b = _service(tmp_path / "b", profile, host=HOST_PAST, wall=WALL_B, lurch=-90.0)

    for arm in (a, b):
        assert arm["code"] == demo_run.EXIT_OK and arm["passes"] == 3, arm["passes"]
        assert arm["sleeps"], "the service never waited on its operational clock"

    # The decision side: byte-identical, and on the RECORDED day.
    assert sorted(a["state"]) == sorted(b["state"])
    assert a["state"] == b["state"]
    records = _records(a["state"])
    kinds = [r["kind"] for r in records]
    assert kinds.count("DECISION") >= 3 and kinds[-1] == "SHUTDOWN", kinds
    for record in records:
        assert DAY_START_NS <= record["runner_now_ns"] <= DAY_START_NS + 86_400 * NS
    spot = json.loads(a["state"]["spot_store.json"])
    perp = json.loads(a["state"]["perp_store.json"])
    assert spot["orders"] and perp["orders"], "the executors were never exercised"
    assert json.loads(a["state"]["risk.json"])["updated_at"].startswith(DAY)

    # The operational side: each arm's heartbeats and ages ran on its own clock.
    for arm in (a, b):
        low, high = arm["wall"]
        low = min(low, high)  # arm B's clock stepped backwards
        assert arm["beats"], "no heartbeat was stamped"
        assert all(low - 90.0 - EPS <= beat <= max(arm["wall"]) + EPS for beat in arm["beats"])
    assert min(b["beats"]) - max(a["beats"]) > 250_000_000
    assert b["ages"][0] - a["ages"][0] == pytest.approx(WALL_B - WALL_A, abs=1e-3)


def test_main_hands_one_operational_clock_to_the_schedule_and_the_telemetry(
    tmp_path, monkeypatch
):
    """The composition boundary itself, driven through `main` on SOAK.

    `main` is given an operational clock and the host is hostile. Every
    heartbeat the CLI-built telemetry stamps must come from that clock, and the
    service's waits must be taken on it too: one clock, reaching both. A `_load`
    that built its emitter on `time.time_ns`, or a loop that read `time.time`,
    would stamp the host's 1971 instead.
    """
    hostile_host(monkeypatch, HOST_PAST)
    beats: list[float] = []
    monkeypatch.setattr(metrics.DEMO_HEARTBEAT, "set", beats.append)
    harness = runner_on(tmp_path, "SOAK", shapes=present_minutes(8))
    argv = ["--config", str(written_config(tmp_path, harness)), "--root", str(harness.root)]
    fake = FakeTime(WALL_B)
    passes = {"n": 0}
    original = DemoRunner.catch_up

    def counting(self, *args, **kwargs):
        assert passes["n"] < MAX_PASSES, "the service passed again without waiting"
        passes["n"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(DemoRunner, "catch_up", counting)
    # The sleep bound turns a loop that never wakes -- one reading a frozen host
    # instead of `fake` -- into a failure below rather than a hung suite.
    fake.hooks.append(
        lambda f: installed_sigterm() if passes["n"] >= 2 or len(f.sleeps) > 20_000 else None
    )
    # The harness above built its own emitter on the (hostile) host clock just to
    # write the fixture -- and it duly beat 1971. Only the CLI's beats count.
    beats.clear()

    code = demo_run.main(
        argv + ["--profile", "SOAK", "run", "--allow-dirty"],
        operational_clock=fake.clock,
        sleep=fake.sleep,
    )

    assert code == demo_run.EXIT_OK and passes["n"] == 2
    assert fake.sleeps, "the schedule did not run on the injected clock"
    assert beats and all(WALL_B - EPS <= beat <= fake.t + EPS for beat in beats), beats


def test_status_needs_no_clock_and_every_other_command_refuses_to_run_without_one(tmp_path):
    import argparse

    harness = runner_on(tmp_path, "TEST")
    config = written_config(tmp_path, harness)
    args = argparse.Namespace(config=config, root=harness.root, profile="TEST")
    args.command = "status"
    assert demo_run._load(args).state is RunnerState.STARTUP
    args.command = "run"
    with pytest.raises(ValueError, match="operational clock"):
        demo_run._load(args)


# --------------------------------------------------------------------------- #
# D. the structural guards, each shown to catch a planted regression
# --------------------------------------------------------------------------- #
HOST_CLOCK_ATTRIBUTES = frozenset(
    {"time", "time_ns", "monotonic", "monotonic_ns", "perf_counter", "perf_counter_ns"}
)
CONSTRUCTORS_NEEDING_A_CLOCK = frozenset({"RiskEngine", "FuturesExecutor"})


def host_clock_reads(source: str) -> list[str]:
    """Every way ``source`` could read the host's clock, by line.

    Not a grep: ``datetime.fromtimestamp(x)`` is a conversion of a value the
    caller already has and is not a read, while ``time.time`` passed as a
    DEFAULT is a read waiting to happen and is.
    """
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import) and any(a.name == "time" for a in node.names):
            found.append(f"{node.lineno}: import time")
        elif isinstance(node, ast.ImportFrom) and node.module == "time":
            found.append(f"{node.lineno}: from time import ...")
        elif isinstance(node, ast.Attribute):
            base = node.value
            name = base.id if isinstance(base, ast.Name) else getattr(base, "attr", None)
            if name == "time" and node.attr in HOST_CLOCK_ATTRIBUTES:
                found.append(f"{node.lineno}: time.{node.attr}")
            elif name in ("datetime", "date") and node.attr in ("now", "utcnow", "today"):
                found.append(f"{node.lineno}: {name}.{node.attr}")
    return found


def unclocked_constructions(source: str) -> list[str]:
    """Every `RiskEngine(...)` / `FuturesExecutor(...)` not given ``clock=``.

    A ``**kwargs`` splat is refused as well: it is exactly how the pre-R1-e
    factories made the clock optional, and it hides from this check whether
    ``clock`` is in it.
    """
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name not in CONSTRUCTORS_NEEDING_A_CLOCK:
            continue
        keywords = {k.arg for k in node.keywords}
        if None in keywords or "clock" not in keywords:
            found.append(f"{node.lineno}: {name}(...) without an explicit clock=")
    return found


def demo_path_sources() -> dict[str, str]:
    files = [p for package in DEMO_PACKAGES for p in sorted(package.rglob("*.py"))]
    return {p.relative_to(REPO).as_posix(): p.read_text("utf-8") for p in [*files, DEMO_CLI]}


def test_the_demo_packages_read_no_host_clock_at_all():
    """`chimera/demo` and `chimera/carry` take every instant they use as an argument."""
    offenders = {
        name: found
        for name, source in demo_path_sources().items()
        if name != DEMO_CLI.relative_to(REPO).as_posix()
        and (found := host_clock_reads(source))
    }
    assert offenders == {}


def test_the_cli_reads_the_host_clock_only_as_mains_operational_defaults():
    """The one sanctioned entry: `main`'s ``operational_clock`` and ``sleep``."""
    tree = ast.parse(DEMO_CLI.read_text("utf-8"))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    sanctioned = {(d.lineno, d.col_offset) for d in main.args.kw_defaults if d is not None}
    reads = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "time"
            and (node.lineno, node.col_offset) not in sanctioned
        ):
            reads.append(f"{node.lineno}: time.{node.attr}")
    assert reads == []
    assert len(sanctioned) == 2
    signature = inspect.signature(demo_run.main).parameters
    assert signature["operational_clock"].default is HOST_TIME
    assert signature["sleep"].default is time.sleep


def test_every_demo_path_risk_engine_and_executor_is_given_its_clock():
    offenders = {
        name: found
        for name, source in demo_path_sources().items()
        if (found := unclocked_constructions(source))
    }
    assert offenders == {}
    # Not vacuous: the factories, the inspection and the R1-c replay are all here.
    constructed = sum(
        source.count("RiskEngine(") + source.count("FuturesExecutor(")
        for source in demo_path_sources().values()
    )
    assert constructed >= 5


CLOCK_PARAMETERS: tuple[tuple[Callable[..., Any], str], ...] = (
    (build_risk_engine, "clock"),
    (build_hedged_position, "clock"),
    (demo_run._daemon, "clock"),
    (demo_run._daemon, "sleep"),
    (RunnerTelemetry.__init__, "wall_ns"),
    (DemoRunner.__init__, "telemetry"),
)


@pytest.mark.parametrize(
    "function, parameter",
    CLOCK_PARAMETERS,
    ids=[f"{f.__qualname__}:{p}" for f, p in CLOCK_PARAMETERS],
)
def test_no_clock_parameter_on_the_demo_path_has_a_default(function, parameter):
    assert inspect.signature(function).parameters[parameter].default is inspect.Parameter.empty


@pytest.mark.parametrize(
    "planted",
    [
        "import time\nx = time.time()\n",
        "import time\ndef f(clock=time.time):\n    return clock()\n",
        "from time import time_ns\n",
        "import time as t\n",
        "from datetime import datetime\nx = datetime.now()\n",
        "import datetime\nx = datetime.datetime.utcnow()\n",
        "import time\ndef loop(stop):\n    time.sleep(1)\n",
    ],
)
def test_the_host_clock_guard_catches_a_planted_read(planted):
    assert host_clock_reads(planted)


def test_the_host_clock_guard_passes_a_conversion_that_reads_nothing():
    legit = "from datetime import datetime, timezone\nx = datetime.fromtimestamp(1.0, timezone.utc)\n"
    assert host_clock_reads(legit) == []


@pytest.mark.parametrize(
    "planted",
    [
        "RiskEngine(limits, state_path=p)\n",
        "FuturesExecutor(venue=v, risk=r, store=s, config=c)\n",
        "FuturesExecutor(venue=v, risk=r, store=s, **kwargs)\n",
        "chimera.risk.RiskEngine(limits, clock=c, **extra)\n",
    ],
)
def test_the_construction_guard_catches_a_planted_fallback(planted):
    assert unclocked_constructions(planted)


def test_the_construction_guard_passes_an_explicit_clock():
    assert (
        unclocked_constructions("FuturesExecutor(venue=v, risk=r, store=s, clock=c)\n") == []
    )
