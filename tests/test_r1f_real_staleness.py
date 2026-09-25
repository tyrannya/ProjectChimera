"""R1-f: real staleness detection, in the service's READY gate.

The adopted item, verbatim (`docs/master_roadmap_r0_r18.md`, R1-f): "the age is
computed as the injected operational wall clock minus the newest available minute
close (or as the recorder heartbeat age), never as the decision clock compared
with itself (today the comparison is identically zero by construction); compared
against `max_data_delay_s` in the runner's READY gate, outside the per-order
decision path; a stale feed produces a recorded `FEED_STALLED` halt (positions
held; funding and liquidation checks continue on the last valid state via a stall
tick driven by the operational clock), automatic continuation when data resumes;
two-sided test: a frozen recorder with advancing wall time halts within threshold
+ grace, a healthy feed never halts; `max_data_delay_s` is enforced here or
removed from the schema."

The invariant every test below leans on: the OPERATIONAL clock decides WHETHER
there is fresh data to process; the DECISION clock (`RunnerClock`, recorded
instants) decides WHAT is decided about it. The operational clock is therefore
allowed to cause a FEED_STALLED or FEED_RESUMED record, and never allowed to
put a value of its own into one.

Time is simulated with R1-d's `FakeTime`, and the recorder with `Recorder`
below: it publishes each minute a couple of seconds after that minute closes,
by the operational clock, until a test freezes it. That is the healthy feed the
roadmap names -- a recorder that publishes the last closed minute promptly, which
is R1-g's cadence and not the 300 s one the recorder has today (see
`test_todays_300s_producer_stalls_between_publications_and_decides_the_same`).

Engineering evidence only: every minute is synthetic (`chimera.demo.fixtures`),
the venue is a dry-run one, and nothing here reads a research artifact.
"""

from __future__ import annotations

import ast
import json
import signal
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import pytest

from chimera import metrics
from chimera.carry.hedge import HedgeState
from chimera.demo.config import ConfigProfile
from chimera.demo.decision_log import LOG_DIR_NAME, verify_log
from chimera.demo.runner import _STATE_NAMES, RunnerError, RunnerState
from chimera.demo.telemetry import RunnerTelemetry
from tests.demo_harness import DAY, REPO, build, campaign_config
from tests.test_demo_cli import written_config
from tests.test_r1d_daemon_service import (
    EPS,
    FakeTime,
    installed_sigterm,
    present_minutes,
    publish,
)
from tests.test_r1e_clock_hardening import HOST_FUTURE, HOST_PAST, WALL_B, hostile_host
from tools import demo_run

NS = 1_000_000_000
MINUTE_MS = 60_000
MINUTE_NS = 60 * NS
DAY_START_S = pd.Timestamp(DAY, tz="UTC").timestamp()
#: The committed campaign's limit and grace (`conf/demo/pvc1.json`, section 8.1).
LIMIT = 180.0
GRACE = 5.0
#: Minutes present when a service starts, and when the frozen recorder stops.
FIRST = 10
FROZEN = 20
#: A healthy recorder publishes a minute this long after it closes.
PUBLISH_DELAY = 2.0
#: Just after minute FIRST-1 was published, so the service starts on a fresh feed.
T_START = DAY_START_S + FIRST * 60 + PUBLISH_DELAY + 0.5
#: The operational clocks the hostility tests use: 2036 and ten years later.
WALL_2036 = WALL_B
WALL_2046 = WALL_B + 10 * 365 * 86_400.0
#: A limit no clock here comes near: for services that must not stall.
OUT_OF_REACH = 1e9
PROFILES = ("CAMPAIGN", "SOAK", "TEST")


# --------------------------------------------------------------------------- #
# the world
# --------------------------------------------------------------------------- #
def close_s(index: int) -> float:
    """Operational seconds at which minute ``index`` of `DAY` closes."""
    return DAY_START_S + 60.0 * (index + 1)


def close_ns(minute_ms: int) -> int:
    return int(minute_ms) * 1_000_000 + MINUTE_NS


def minute_ms(index: int) -> int:
    return int(DAY_START_S * 1000) + index * MINUTE_MS


class Recorder:
    """The recorder, driven by the operational clock inside `FakeTime.sleep`.

    It publishes every minute ``delay`` seconds after it closes -- in whole
    batches of ``batch`` minutes when asked to behave like today's producer --
    and publishes nothing past ``frozen_at`` minutes while that is set.
    """

    def __init__(self, harness, *, count: int = FIRST, delay: float = PUBLISH_DELAY):
        self.harness = harness
        self.count = count
        self.delay = delay
        self.batch = 1
        self.frozen_at: int | None = None

    def due(self, t: float) -> int:
        n = int((t - self.delay - DAY_START_S) // 60)
        n = n // self.batch * self.batch
        if self.frozen_at is not None:
            n = min(n, self.frozen_at)
        return max(min(n, 1440), self.count)

    def __call__(self, fake: FakeTime) -> None:
        n = self.due(fake.t)
        if n != self.count:
            publish(self.harness, n)
            self.count = n


def harness_for(
    where: Path,
    profile: str = "SOAK",
    *,
    limit: float | None = None,
    runner: dict[str, Any] | None = None,
    count: int = FIRST,
    telemetry: Any | None = None,
    start: bool = True,
):
    """A runner on ``profile`` over `DAY`'s first ``count`` minutes.

    CAMPAIGN is forced the way `tests/test_r1e_clock_hardening.py` forces it (the
    committed CAMPAIGN configuration cannot carry rules while its protocol is
    not frozen, which is a separate R1 item), BEFORE `start()`.
    """
    limits = {} if limit is None else {"max_data_delay_s": limit}
    config = campaign_config(
        where / "state",
        profile="TEST" if profile == "CAMPAIGN" else profile,
        limits=limits,
        runner=runner,
    )
    harness = build(
        where, config=config, shapes=present_minutes(count), start=False, telemetry=telemetry
    )
    if profile == "CAMPAIGN":
        object.__setattr__(harness.runner.config, "profile", ConfigProfile.CAMPAIGN)
    assert harness.runner.config.profile.value == profile
    if start:
        assert harness.runner.start() is RunnerState.READY
    return harness


def telemetry_on(fake: FakeTime, where: Path) -> RunnerTelemetry:
    return RunnerTelemetry(
        state_dir=where / "state",
        states=_STATE_NAMES,
        rules=("R1_carry", "R2_frozen_logistic", "R3_daily_momentum"),
        wall_ns=fake.wall_ns,
    )


def watch(runner, name: str, fake: FakeTime) -> list[tuple[float, RunnerState, RunnerState]]:
    """Wrap ``runner.name``: every call's simulated instant and the state around it."""
    seen: list[tuple[float, RunnerState, RunnerState]] = []
    original = getattr(runner, name)

    def watched(*args: Any, **kwargs: Any) -> Any:
        before = runner.state
        result = original(*args, **kwargs)
        seen.append((fake.t, before, runner.state))
        return result

    setattr(runner, name, watched)
    return seen


def serve(
    harness,
    fake: FakeTime,
    *,
    until: float,
    hooks: tuple[Callable[[FakeTime], None], ...] = (),
) -> int:
    """R1-d's service loop over ``harness`` until the operational clock reaches ``until``."""
    stop: dict[str, int] = {}
    for hook in hooks:
        fake.hooks.append(hook)

    def deadline(f: FakeTime) -> None:
        if f.t >= until or len(f.sleeps) > 200_000:  # a loop that never waits fails, not hangs
            stop["signal"] = int(signal.SIGTERM)

    fake.hooks.append(deadline)
    return demo_run._daemon(harness.runner, stop, clock=fake.clock, sleep=fake.sleep)


def kinds_of(harness) -> list[str]:
    return [r["kind"] for r in harness.records()]


def after(records: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    """Every record after the first one of ``kind``."""
    index = next(i for i, r in enumerate(records) if r["kind"] == kind)
    return records[index + 1 :]


def legs(runner) -> dict[str, str]:
    return {
        "hedge_state": runner.position.state.value,
        "spot": str(runner.position.leg("spot").quantity),
        "perp": str(runner.position.leg("perp").quantity),
    }


def order_counts(runner) -> tuple[int, int]:
    return (
        len(runner.position.spot.store.state.orders),
        len(runner.position.perp.store.state.orders),
    )


def state_bytes(state_dir: Path) -> dict[str, bytes]:
    return {
        p.relative_to(state_dir).as_posix(): p.read_bytes()
        for p in sorted(state_dir.rglob("*"))
        if p.is_file()
    }


# --------------------------------------------------------------------------- #
# the limit, and the vacuous check it replaces
# --------------------------------------------------------------------------- #
def test_the_gate_reads_the_committed_limit_through_the_one_wiring(tmp_path):
    committed = json.loads((REPO / "conf" / "demo" / "pvc1.json").read_text("utf-8"))
    assert committed["limits"]["max_data_delay_s"] == LIMIT
    assert committed.get("runner", {}).get("ready_grace_seconds", GRACE) == GRACE
    harness = harness_for(tmp_path)
    assert harness.runner.risk.limits.max_data_delay_s == LIMIT


def test_nothing_on_the_demo_path_calls_note_feed_any_more():
    """The old check compared a minute's close with a decision clock `tick` had
    just set to that close -- zero by construction. It is gone, so the READY
    gate is the one authority on staleness and no second, differently-clocked
    one can disagree with it."""
    sources = [
        *sorted((REPO / "chimera" / "demo").rglob("*.py")),
        *sorted((REPO / "chimera" / "carry").rglob("*.py")),
        REPO / "tools" / "demo_run.py",
    ]
    calls = [
        f"{path.name}:{node.lineno}"
        for path in sources
        for node in ast.walk(ast.parse(path.read_text("utf-8")))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "note_feed"
    ]
    assert calls == []


# --------------------------------------------------------------------------- #
# G. the threshold, exactly
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("profile", PROFILES)
def test_the_threshold_is_an_age_above_the_limit_measured_from_the_close(tmp_path, profile):
    """Below, at, and above ``max_data_delay_s``, to the nanosecond.

    The age is measured from the newest minute's CLOSE (its open plus a minute):
    at exactly ``close + limit`` the feed is fresh, and one nanosecond later it
    is stale. An age measured from the open would already be a minute over.
    """
    harness = harness_for(tmp_path, profile)
    runner = harness.runner
    newest = runner.cursor.latest_minute_ms()
    assert newest == minute_ms(FIRST - 1)
    close = close_ns(newest)
    limit = int(LIMIT * NS)
    count = len(harness.records())

    assert runner.check_feed(close + limit - 1) == close + limit
    assert runner.state is RunnerState.READY
    assert runner.check_feed(close + limit) == close + limit
    assert runner.state is RunnerState.READY
    assert len(harness.records()) == count, "a fresh feed wrote a record"

    runner.check_feed(close + limit + 1)
    assert runner.state is RunnerState.FEED_STALLED
    assert kinds_of(harness)[count:] == ["FEED_STALLED"]

    runner.check_feed(close + limit + 10**15)
    assert kinds_of(harness)[count:] == ["FEED_STALLED"], "a stall is recorded once"

    runner.check_feed(close + limit)
    assert runner.state is RunnerState.READY
    assert kinds_of(harness)[count:] == ["FEED_STALLED", "FEED_RESUMED"]
    runner.check_feed(close)
    assert runner.state is RunnerState.READY


@pytest.mark.parametrize("profile", PROFILES)
def test_a_close_ahead_of_the_operational_clock_is_stale_not_fresh(tmp_path, profile):
    """`RiskEngine.note_feed`'s rule: a negative age means the clocks disagree
    and the feed's age is unknown, so it is not read as fresh."""
    harness = harness_for(tmp_path, profile)
    close = close_ns(harness.runner.cursor.latest_minute_ms())
    harness.runner.check_feed(close - 1)
    assert harness.runner.state is RunnerState.FEED_STALLED


def test_a_feed_with_no_minute_is_stale(tmp_path):
    """Nothing published is not fresh, and no instant is invented for it."""
    harness = harness_for(tmp_path)
    for parquet in (harness.root).rglob("*.parquet"):
        parquet.unlink()
    assert harness.runner.cursor.latest_minute_ms() is None
    assert harness.runner.check_feed(int(T_START * NS)) is None
    assert harness.runner.state is RunnerState.FEED_STALLED
    assert harness.records()[-1]["feed"]["newest_minute"] is None


def test_a_stall_record_carries_recorded_facts_and_no_operational_value(tmp_path):
    harness = harness_for(tmp_path)
    runner = harness.runner
    newest = runner.cursor.latest_minute_ms()
    decision_now = runner.clock.now_ns
    runner.check_feed(int(WALL_2036 * NS))
    record = harness.records()[-1]

    assert record["kind"] == "FEED_STALLED"
    assert record["feed"] == {
        "newest_minute": pd.Timestamp(newest, unit="ms", tz="UTC").isoformat(),
        "max_data_delay_s": LIMIT,
    }
    assert record["runner_now_ns"] == decision_now, "the decision clock, not the wall"
    assert record["position_after"]["hedge_state"] == runner.position.state.value
    assert record["veto_or_rejection"]["label"] == "feed_stalled"
    assert str(int(WALL_2036)) not in json.dumps(record)


# --------------------------------------------------------------------------- #
# A. a frozen recorder halts within threshold + grace, and holds the position
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "profile, limit",
    [("CAMPAIGN", None), ("SOAK", None), ("TEST", None), ("SOAK", 150.0)],
    ids=["CAMPAIGN-180", "SOAK-180", "TEST-180", "SOAK-150"],
)
def test_a_frozen_recorder_stalls_within_the_limit_plus_grace_and_holds(
    tmp_path, profile, limit
):
    """The recorder publishes minutes 0..FROZEN-1 on time and then nothing more,
    while the operational clock runs on for half an hour.

    150 s is not a whole number of minutes, so the wake at each minute close
    alone would declare the stall up to a minute late; the gate's own wake is
    what makes "within threshold + grace" true for any limit.
    """
    limit_s = LIMIT if limit is None else limit
    fake = FakeTime(T_START)
    harness = harness_for(
        tmp_path, profile, limit=limit, telemetry=telemetry_on(fake, tmp_path)
    )
    runner = harness.runner
    recorder = Recorder(harness)
    recorder.frozen_at = FROZEN
    gate = watch(runner, "check_feed", fake)
    stalls = watch(runner, "stall_tick", fake)
    at_stall: dict[str, Any] = {}

    def snapshot(_f: FakeTime) -> None:
        if runner.state is RunnerState.FEED_STALLED and not at_stall:
            at_stall.update(
                clock=runner.clock.now_ns,
                cursor=runner.cursor.last_minute_processed,
                legs=legs(runner),
                orders=order_counts(runner),
            )

    last_close = close_s(FROZEN - 1)
    code = serve(
        harness, fake, until=last_close + limit_s + 30 * 60, hooks=(recorder, snapshot)
    )
    assert code == demo_run.EXIT_OK

    # When: the first gate to see the stale feed, and it is within limit + grace.
    stalled_at = next(t for t, before, now in gate if now is RunnerState.FEED_STALLED)
    assert limit_s < stalled_at - last_close <= limit_s + GRACE + EPS
    assert stalled_at - last_close == pytest.approx(limit_s + GRACE, abs=EPS)
    fresh = [t for t, _before, now in gate if t < stalled_at]
    assert fresh and all(t - last_close <= limit_s for t in fresh[-1:])

    # What: one FEED_STALLED record and nothing decided after it.
    records = harness.records()
    kinds = [r["kind"] for r in records]
    assert kinds.count("FEED_STALLED") == 1 and "FEED_RESUMED" not in kinds
    stalled = records[kinds.index("FEED_STALLED")]
    assert [r["kind"] for r in after(records, "FEED_STALLED")] == ["SHUTDOWN"]
    assert (
        stalled["feed"]["newest_minute"]
        == pd.Timestamp(minute_ms(FROZEN - 1), unit="ms", tz="UTC").isoformat()
    )
    assert "HALT" not in kinds and not runner.risk.state.halted

    # The decision clock did not move with the wall, nothing was processed or
    # ordered, and the position the stall found is the position it kept.
    assert runner.cursor.last_minute_processed == minute_ms(FROZEN - 1) == at_stall["cursor"]
    assert runner.clock.now_ns == at_stall["clock"] == stalled["runner_now_ns"]
    assert records[-1]["runner_now_ns"] == stalled["runner_now_ns"]
    assert at_stall["legs"]["hedge_state"] == HedgeState.HEDGED.value, "nothing to hold"
    assert legs(runner) == at_stall["legs"]
    assert stalled["position_after"]["spot_qty"] == at_stall["legs"]["spot"]
    assert order_counts(runner) == at_stall["orders"]

    # The stall tick ran on every wake of the stall, on the minute schedule.
    ticks = [t for t, _b, _n in stalls]
    assert len(ticks) >= 29
    assert all((t - GRACE) % 60 == pytest.approx(0, abs=EPS) for t in ticks[1:]), ticks[:5]


# --------------------------------------------------------------------------- #
# B. a healthy feed never halts
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("profile", PROFILES)
def test_a_healthy_feed_never_stalls_and_every_minute_is_decided(tmp_path, profile):
    fake = FakeTime(T_START)
    harness = harness_for(tmp_path, profile, telemetry=telemetry_on(fake, tmp_path))
    recorder = Recorder(harness)
    gate = watch(harness.runner, "check_feed", fake)

    # Stopped a second after a close, before that minute is published, so the
    # last pass has seen every minute the recorder has published.
    until = close_s(FIRST + 40) + 1.0
    assert serve(harness, fake, until=until, hooks=(recorder,)) == demo_run.EXIT_OK
    assert recorder.count == FIRST + 40

    kinds = kinds_of(harness)
    assert "FEED_STALLED" not in kinds and "FEED_RESUMED" not in kinds and "HALT" not in kinds
    assert len(gate) >= 40, "the gate did not run on every pass"
    assert harness.runner.cursor.last_minute_processed == minute_ms(recorder.count - 1)
    decided = [r["minute"] for r in harness.records() if r["kind"] == "DECISION"]
    # 7 catch-up minutes were too old at start (section 2.2); every later minute,
    # one per wake, is decided exactly once.
    assert len(decided) == len(set(decided)) == recorder.count - (FIRST - 3)


# --------------------------------------------------------------------------- #
# C. automatic continuation, and only for the stall
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("catch_up", [60, None], ids=["every-pending-minute", "committed-cap"])
def test_the_feed_resumes_by_itself_and_no_minute_is_lost_or_repeated(tmp_path, catch_up):
    """Freeze, stall, publish again. No operator, no RESUME, nothing lost.

    With a catch-up window wide enough for the outage every pending minute is
    decided, the first one included. With the committed one (3) the older ones
    are recorded as SKIPPED_STALE, exactly once each -- that cap is section 2.2's
    and R1-g's to replace, not this gate's.
    """
    runner_settings = None if catch_up is None else {"max_catchup_minutes": catch_up}
    fake = FakeTime(T_START)
    harness = harness_for(
        tmp_path, runner=runner_settings, telemetry=telemetry_on(fake, tmp_path)
    )
    runner = harness.runner
    recorder = Recorder(harness)
    recorder.frozen_at = FROZEN
    stall_close = close_s(FROZEN - 1)

    def thaw(f: FakeTime) -> None:
        if f.t >= stall_close + LIMIT + GRACE + 150:
            recorder.frozen_at = None

    assert (
        serve(harness, fake, until=stall_close + 20 * 60, hooks=(thaw, recorder))
        == demo_run.EXIT_OK
    )

    records = harness.records()
    kinds = [r["kind"] for r in records]
    assert kinds.count("FEED_STALLED") == 1 and kinds.count("FEED_RESUMED") == 1
    assert kinds.index("FEED_STALLED") < kinds.index("FEED_RESUMED")
    assert not {"RESUME", "OPERATOR", "HALT"} & set(kinds)
    assert not runner.risk.state.halted

    stalled = records[kinds.index("FEED_STALLED")]
    resumed = records[kinds.index("FEED_RESUMED")]
    assert resumed["runner_now_ns"] == stalled["runner_now_ns"], "the wall moved no clock"
    assert resumed["veto_or_rejection"] is None

    # Every minute after the frozen one is accounted for exactly once, in order.
    accounted = [
        r["minute"]
        for r in after(records, "FEED_RESUMED")
        if r["kind"] in ("DECISION", "SKIPPED_STALE")
    ]
    expected = [
        pd.Timestamp(minute_ms(i), unit="ms", tz="UTC").isoformat()
        for i in range(FROZEN, recorder.count)
    ]
    assert accounted == expected
    decided = [r["minute"] for r in records if r["kind"] == "DECISION"]
    assert len(decided) == len(set(decided))
    if catch_up is not None:
        first = next(r for r in after(records, "FEED_RESUMED") if r["kind"] == "DECISION")
        assert first["minute"] == expected[0]
        assert "SKIPPED_STALE" not in [r["kind"] for r in after(records, "FEED_RESUMED")]
    assert runner.cursor.last_minute_processed == minute_ms(recorder.count - 1)


@pytest.mark.parametrize("decided", [True, False], ids=["after-a-minute", "before-any"])
def test_a_fresh_feed_never_clears_a_halt(tmp_path, decided):
    """The kill switch, engaged during a stall, halts on the stall tick -- with
    or without a processed minute to check a position on; the feed then comes
    back, and the halt stands. The gate moves only between READY and
    FEED_STALLED."""
    harness = harness_for(tmp_path)
    runner = harness.runner
    if decided:
        runner.catch_up()
    close = close_ns(runner.cursor.latest_minute_ms())
    runner.check_feed(close + int(LIMIT * NS) + 1)
    assert runner.state is RunnerState.FEED_STALLED

    switch = Path(runner.risk._kill_switch_path)
    switch.parent.mkdir(parents=True, exist_ok=True)
    switch.write_text("drill", encoding="utf-8")
    runner.stall_tick()
    assert runner.state is RunnerState.HALT and runner.risk.state.halted
    switch.unlink()

    count = len(harness.records())
    publish(harness, FIRST + 5)
    runner.check_feed(close_ns(runner.cursor.latest_minute_ms()) + NS)
    assert runner.state is RunnerState.HALT
    assert runner.risk.state.halted and runner.halt_reason == "kill_switch"
    assert len(harness.records()) == count, "a fresh feed wrote over a halt"
    assert runner.catch_up() == []


# --------------------------------------------------------------------------- #
# D / E. the stall tick: the held position is checked, never traded, and
# checking it again changes nothing
# --------------------------------------------------------------------------- #
LATE = 66  # minutes decided before the stall: past the 01:00 settlement instant


def stalled_on_a_held_position(tmp_path):
    harness = harness_for(tmp_path, count=LATE + 5, start=True)
    harness.run(LATE)
    runner = harness.runner
    assert runner.position.state is HedgeState.HEDGED
    runner.check_feed(int(WALL_2036 * NS))
    assert runner.state is RunnerState.FEED_STALLED
    return harness


def test_the_stall_tick_checks_the_last_recorded_minute_and_trades_nothing(
    tmp_path, monkeypatch
):
    harness = stalled_on_a_held_position(tmp_path)
    runner = harness.runner
    last = runner.cursor.last_minute_processed
    assert last == minute_ms(LATE - 1)

    checked: list[int] = []
    touched = runner.position.liquidation_touched

    def spy(state, **kwargs):
        checked.append(int(state.minute_ns))
        return touched(state, **kwargs)

    monkeypatch.setattr(runner.position, "liquidation_touched", spy)
    evaluated: list[str] = []
    for rule in runner.rules:
        monkeypatch.setattr(
            rule, "evaluate", lambda *a, _id=rule.rule_id, **k: evaluated.append(_id)
        )

    clock, held, orders = runner.clock.now_ns, legs(runner), order_counts(runner)
    count = len(harness.records())
    for _ in range(25):
        assert runner.stall_tick() is None

    assert checked == [last * 1_000_000] * 25, "the liquidation check ran on the last minute"
    assert evaluated == [], "a rule was evaluated during a stall"
    assert runner.clock.now_ns == clock and runner.cursor.last_minute_processed == last
    assert legs(runner) == held and order_counts(runner) == orders
    assert len(harness.records()) == count
    assert runner.state is RunnerState.FEED_STALLED and not runner.risk.state.halted
    with pytest.raises(RunnerError, match="FEED_STALLED"):
        runner.tick(last + MINUTE_MS)
    assert runner.catch_up() == [] and runner.run_minutes([last + MINUTE_MS]) == []


def test_a_late_settlement_is_booked_once_and_a_future_one_never(tmp_path):
    """Funding "continues on the last valid state": the window is the last
    minute's, so a row that arrives late for an instant the feed has already
    passed is booked -- once, however many stall ticks follow -- and a row for
    an instant the feed has not reached is left for the minute that reaches it."""
    harness = stalled_on_a_held_position(tmp_path)
    runner = harness.runner
    ledger_path = harness.state_dir / "carry_ledger.json"
    one = int(pd.Timestamp(f"{DAY}T01:00", tz="UTC").value)
    two = int(pd.Timestamp(f"{DAY}T02:00", tz="UTC").value)
    assert close_ns(runner.cursor.last_minute_processed) > one
    assert one not in runner.position.ledger.state.settled

    harness.feed.write_settlements([DAY], hours=(0, 1, 2, 8, 16))
    runner.stall_tick()
    funding = [r for r in harness.records() if r["kind"] == "FUNDING"]
    assert [r["funding"]["settlement_instant_ns"] for r in funding] == [one]
    assert (
        funding[0]["minute"]
        == pd.Timestamp(runner.cursor.last_minute_processed, unit="ms", tz="UTC").isoformat()
    )
    assert one in runner.position.ledger.state.settled
    assert two not in runner.position.ledger.state.settled
    # Aegis was handed the equity the settlement moved, before the record, so
    # the record's hash is the state on disk and the two equities agree.
    assert runner.risk.state.equity == pytest.approx(
        float(runner.position.ledger.state.last_equity), abs=1e-6
    )

    frozen = state_bytes(harness.state_dir)
    for _ in range(20):
        runner.stall_tick()
    assert state_bytes(harness.state_dir) == frozen, "a repeated stall tick changed state"
    assert [r["kind"] for r in harness.records()].count("FUNDING") == 1
    assert ledger_path.is_file()
    assert verify_log(harness.state_dir / LOG_DIR_NAME).ok

    # And a clean restart after it neither disputes the equity (R1-b) nor the
    # risk state's continuity (R1-c): it comes back stalled, and nothing else.
    config = runner.config
    runner.shutdown("restart after a stall-tick settlement")
    resumed = build(tmp_path, config=config, shapes=present_minutes(LATE + 5), start=False)
    resumed.feed.write_settlements([DAY], hours=(0, 1, 2, 8, 16))
    assert resumed.runner.start() is RunnerState.FEED_STALLED
    assert "RECOVERY" not in kinds_of(resumed) and not resumed.runner.risk.state.halted


def test_a_real_touch_during_a_stall_flattens_as_a_liquidation_not_as_a_stall(
    tmp_path, monkeypatch
):
    """The other side of "positions held": the check is live. A touch on the last
    recorded mark flattens and halts exactly as a tick's would, and says so."""
    harness = stalled_on_a_held_position(tmp_path)
    runner = harness.runner
    monkeypatch.setattr(runner.position, "liquidation_touched", lambda *a, **k: True)
    runner.stall_tick()
    kinds = kinds_of(harness)
    assert kinds[-2:] == ["LIQUIDATION_TOUCH", "HALT"]
    assert runner.halt_reason.startswith("liquidation_touch")
    assert (
        runner.position.leg("spot").quantity == 0 and runner.position.leg("perp").quantity == 0
    )


# --------------------------------------------------------------------------- #
# H. a restart during a stall
# --------------------------------------------------------------------------- #
def test_a_restart_during_a_stall_stays_stalled_until_the_feed_is_fresh(tmp_path):
    harness = harness_for(tmp_path)
    config = harness.runner.config
    harness.runner.catch_up()
    assert harness.runner.cursor.last_minute_processed == minute_ms(FIRST - 1)
    stale = int(WALL_2036 * NS)
    harness.runner.check_feed(stale)
    harness.runner.shutdown("planned restart during a stall")

    resumed = build(tmp_path, config=config, shapes=present_minutes(FIRST), start=False)
    count = len(resumed.records())
    runner = resumed.runner
    assert runner.start() is RunnerState.FEED_STALLED, "a restart is not news about the feed"
    assert kinds_of(resumed)[count:] == ["STARTUP"]
    assert runner.catch_up() == []
    runner.check_feed(stale)
    assert kinds_of(resumed)[count:] == ["STARTUP"], "the open stall was declared again"

    publish(resumed, FIRST + 2)
    runner.check_feed(close_ns(minute_ms(FIRST + 1)) + NS)
    assert runner.state is RunnerState.READY
    assert kinds_of(resumed)[count:] == ["STARTUP", "FEED_RESUMED"]
    decided = [o.kind.value for o in runner.catch_up()]
    assert decided == ["DECISION", "DECISION"]


def test_a_crash_between_the_stall_record_and_the_state_file_recovers_stalled(tmp_path):
    """LOG_AHEAD_OF_STATE with a FEED_STALLED tail: nothing is excluded (no
    minute was left half-decided) and the restarted runner is still stalled."""
    harness = harness_for(tmp_path)
    config = harness.runner.config
    state_file = harness.state_dir / "runner_state.json"
    before = state_file.read_bytes()
    harness.runner.check_feed(int(WALL_2036 * NS))
    state_file.write_bytes(before)  # the crash: the record committed, the state file not

    resumed = build(tmp_path, config=config, shapes=present_minutes(FIRST), start=False)
    assert resumed.runner.start() is RunnerState.FEED_STALLED
    recovery = [r for r in resumed.records() if r["kind"] == "RECOVERY"][-1]["recovery"]
    assert recovery["cause"] == "LOG_AHEAD_OF_STATE"
    assert recovery["minute_finished"] is True
    assert recovery["evidence_excluded_minute"] is None


def test_a_crash_after_a_stall_ticks_late_funding_record_excludes_no_minute(tmp_path):
    """A FUNDING record for the last processed minute AFTER its DECISION is the
    stall tick's shape, not a half-decided minute: the minute stays in the
    evidence, and the settlement is not booked a second time."""
    harness = stalled_on_a_held_position(tmp_path)
    config = harness.runner.config
    last = harness.runner.cursor.last_minute_processed
    state_file = harness.state_dir / "runner_state.json"
    before = state_file.read_bytes()
    harness.feed.write_settlements([DAY], hours=(0, 1, 8, 16))
    harness.runner.stall_tick()
    assert kinds_of(harness)[-1] == "FUNDING"
    state_file.write_bytes(before)

    resumed = build(tmp_path, config=config, shapes=present_minutes(LATE + 5), start=False)
    resumed.feed.write_settlements([DAY], hours=(0, 1, 8, 16))
    assert resumed.runner.start() is RunnerState.FEED_STALLED
    recovery = [r for r in resumed.records() if r["kind"] == "RECOVERY"][-1]["recovery"]
    assert recovery["cause"] == "LOG_AHEAD_OF_STATE"
    assert recovery["minute_finished"] is True
    assert recovery["evidence_excluded_minute"] is None
    assert resumed.runner.cursor.last_minute_processed == last
    resumed.runner.stall_tick()
    assert kinds_of(resumed).count("FUNDING") == 1


def test_the_service_restarted_mid_stall_does_not_declare_it_twice(tmp_path, monkeypatch):
    """Through `main`, as systemd would run it: a stall, SIGTERM, a restart into
    the same stale feed, SIGTERM, and a restart once the feed is fresh."""
    harness = harness_for(tmp_path, start=False)
    argv = [
        "--config",
        str(written_config(tmp_path, harness)),
        "--root",
        str(harness.root),
        "--profile",
        "SOAK",
    ]

    def run(fake: FakeTime, passes: int) -> int:
        seen = {"n": 0}
        original = demo_run.DemoRunner.check_feed

        def counted(self, now_ns):
            seen["n"] += 1
            return original(self, now_ns)

        monkeypatch.setattr(demo_run.DemoRunner, "check_feed", counted)
        # The sleep bound makes a service that never reaches its gate fail
        # here, on the assertions below, instead of never stopping.
        fake.hooks.append(
            lambda f: (
                installed_sigterm() if seen["n"] >= passes or len(f.sleeps) > 20_000 else None
            )
        )
        return demo_run.main(
            argv + ["run", "--allow-dirty"], operational_clock=fake.clock, sleep=fake.sleep
        )

    stale = close_s(FIRST - 1) + LIMIT + 60
    assert run(FakeTime(stale), 3) == demo_run.EXIT_OK
    assert run(FakeTime(stale + 600), 3) == demo_run.EXIT_OK
    kinds = kinds_of(harness)
    assert kinds.count("FEED_STALLED") == 1 and kinds.count("STARTUP") == 2
    assert "DECISION" not in kinds[kinds.index("FEED_STALLED") :]

    publish(harness, FIRST + 12)
    assert run(FakeTime(close_s(FIRST + 11) + PUBLISH_DELAY), 2) == demo_run.EXIT_OK
    kinds = kinds_of(harness)
    assert kinds.count("FEED_STALLED") == 1 and kinds.count("FEED_RESUMED") == 1
    assert "DECISION" in kinds[kinds.index("FEED_RESUMED") :]


# --------------------------------------------------------------------------- #
# R1-d during a stall: signals, heartbeat, gauges
# --------------------------------------------------------------------------- #
def test_sigterm_during_a_stall_tick_stops_cleanly(tmp_path):
    fake = FakeTime(T_START)
    harness = harness_for(tmp_path, telemetry=telemetry_on(fake, tmp_path))
    runner = harness.runner
    recorder = Recorder(harness)
    recorder.frozen_at = FIRST
    stop: dict[str, int] = {}
    original = runner.stall_tick

    def interrupted() -> Any:
        stop["signal"] = int(signal.SIGTERM)  # arrives while the stall tick runs
        return original()

    runner.stall_tick = interrupted  # type: ignore[method-assign]
    fake.hooks.append(recorder)
    # A service that never stalls never calls the stall tick: stop it anyway,
    # so that failing is an assertion below and not a loop that never ends.
    fake.hooks.append(
        lambda f: stop.update(signal=int(signal.SIGTERM)) if len(f.sleeps) > 20_000 else None
    )
    code = demo_run._daemon(runner, stop, clock=fake.clock, sleep=fake.sleep)

    assert code == demo_run.EXIT_OK
    kinds = kinds_of(harness)
    assert kinds[-2:] == ["FEED_STALLED", "SHUTDOWN"] and "HALT" not in kinds
    assert harness.records()[-1]["operator"]["note"] == "stopped by SIGTERM"


def test_the_heartbeat_and_the_ages_keep_moving_through_a_stall(tmp_path, monkeypatch):
    """A stalled service is alive (heartbeat) and its feed is visibly old (the
    ages keep rising instead of freezing at their last healthy reading)."""
    beats: list[float] = []
    ages: list[float] = []
    monkeypatch.setattr(metrics.DEMO_HEARTBEAT, "set", beats.append)
    monkeypatch.setattr(metrics.DEMO_LAST_MINUTE_AGE, "set", ages.append)
    fake = FakeTime(T_START)
    harness = harness_for(tmp_path, telemetry=telemetry_on(fake, tmp_path))
    recorder = Recorder(harness)
    recorder.frozen_at = FIRST
    beats.clear()
    ages.clear()

    serve(harness, fake, until=T_START + 20 * 60, hooks=(recorder,))

    assert harness.runner.state is RunnerState.SHUTDOWN
    assert "FEED_STALLED" in kinds_of(harness)
    assert all(b - a <= 30.0 + EPS for a, b in zip(beats, beats[1:])), "a gap in the beat"
    assert ages[-1] >= 19 * 60, "the age froze during the stall"
    assert ages[-1] == pytest.approx(fake.t - close_s(FIRST - 1), abs=31.0)


def test_a_process_restarted_into_a_stall_reports_the_real_age(tmp_path, monkeypatch):
    """A fresh process has read no minute, so its age gauge would read Prometheus's
    default, zero, for the whole stall. The stall tick reports the minute it holds
    on, without counting it as attempted, and the heartbeat then ages it."""
    harness = harness_for(tmp_path)
    config = harness.runner.config
    harness.runner.catch_up()
    harness.runner.check_feed(int(WALL_2036 * NS))
    harness.runner.shutdown("restart into a stall")

    ages: list[float] = []
    ticks: list[int] = []
    monkeypatch.setattr(metrics.DEMO_LAST_MINUTE_AGE, "set", ages.append)
    monkeypatch.setattr(metrics.DEMO_TICKS, "inc", lambda *a: ticks.append(1))
    fake = FakeTime(WALL_2036)
    resumed = build(
        tmp_path,
        config=config,
        shapes=present_minutes(FIRST),
        start=False,
        telemetry=telemetry_on(fake, tmp_path),
    )
    assert resumed.runner.start() is RunnerState.FEED_STALLED
    resumed.runner.stall_tick()
    fake.t += 30
    resumed.runner.telemetry.on_heartbeat()
    assert ages and ages[-1] == pytest.approx(fake.t - close_s(FIRST - 1), abs=EPS)
    assert ticks == [], "the stall tick counted a minute it did not attempt"


# --------------------------------------------------------------------------- #
# F. hostility: the operational clock decides WHETHER, never WHAT
# --------------------------------------------------------------------------- #
def test_fresh_and_stale_are_decided_by_the_operational_clock_not_the_host(tmp_path):
    """Both arms: host in 2100, every host-time fallback raising. One arm's
    operational clock tracks the recorder; the other's is in 2036. Only the
    operational clock differs, and it alone decides whether anything is decided."""
    with pytest.MonkeyPatch.context() as mp:
        hostile_host(mp, HOST_FUTURE)
        fake = FakeTime(T_START)
        fresh = harness_for(
            tmp_path / "fresh", telemetry=telemetry_on(fake, tmp_path / "fresh")
        )
        recorder = Recorder(fresh)
        assert serve(fresh, fake, until=T_START + 10 * 60, hooks=(recorder,)) == 0

        late = FakeTime(WALL_2036)
        stale = harness_for(
            tmp_path / "stale", telemetry=telemetry_on(late, tmp_path / "stale")
        )
        assert serve(stale, late, until=WALL_2036 + 10 * 60) == 0

    fresh_kinds, stale_kinds = kinds_of(fresh), kinds_of(stale)
    assert "FEED_STALLED" not in fresh_kinds and fresh_kinds.count("DECISION") >= 10
    assert stale_kinds == ["STARTUP", "FEED_STALLED", "SHUTDOWN"]
    day_start_ns = int(DAY_START_S * NS)
    for record in fresh.records() + stale.records():
        assert day_start_ns <= record["runner_now_ns"] <= day_start_ns + 86_400 * NS


@pytest.mark.parametrize("profile", ["CAMPAIGN", "SOAK"])
def test_hostile_hosts_and_different_operational_clocks_write_identical_evidence(
    tmp_path, profile
):
    """R1-e's byte-identity, now under the committed limit and through a stall.

    Arm A: host in 2100, the recorder 2 s late by an operational clock that
    tracks it. Arm B: host in 1971, the recorder 90 s late -- still fresh, and
    every wake of B sees a different age from A's. Both decide the same minutes,
    both stall after the same minute, and every persisted byte is identical: the
    stall records hold no operational value, or the arms would differ there.
    """
    arms = []
    for name, host, delay in (("a", HOST_FUTURE, PUBLISH_DELAY), ("b", HOST_PAST, 90.0)):
        where = tmp_path / name
        with pytest.MonkeyPatch.context() as mp:
            hostile_host(mp, host)
            start = DAY_START_S + FIRST * 60 + delay + 0.5
            fake = FakeTime(start)
            harness = harness_for(where, profile, telemetry=telemetry_on(fake, where))
            recorder = Recorder(harness, delay=delay)
            recorder.frozen_at = FROZEN
            code = serve(
                harness,
                fake,
                until=close_s(FROZEN - 1) + LIMIT + GRACE + 5 * 60,
                hooks=(recorder,),
            )
        assert code == demo_run.EXIT_OK
        arms.append(state_bytes(harness.state_dir))
        kinds = kinds_of(harness)
        assert kinds.count("FEED_STALLED") == 1 and kinds[-1] == "SHUTDOWN"
        assert kinds.count("DECISION") == FROZEN - (FIRST - 3)

    assert sorted(arms[0]) == sorted(arms[1])
    assert arms[0] == arms[1]


def test_two_stale_services_a_decade_apart_write_identical_stall_evidence(tmp_path):
    """Operational 2036 under a 2100 host, and 2046 under a 1971 one."""
    arms = []
    for name, host, wall in (("a", HOST_FUTURE, WALL_2036), ("b", HOST_PAST, WALL_2046)):
        where = tmp_path / name
        with pytest.MonkeyPatch.context() as mp:
            hostile_host(mp, host)
            fake = FakeTime(wall)
            harness = harness_for(where, telemetry=telemetry_on(fake, where))
            assert serve(harness, fake, until=wall + 5 * 60) == demo_run.EXIT_OK
        arms.append(state_bytes(harness.state_dir))
        assert kinds_of(harness) == ["STARTUP", "FEED_STALLED", "SHUTDOWN"]
    assert arms[0] == arms[1]


# --------------------------------------------------------------------------- #
# the producer as it is today
# --------------------------------------------------------------------------- #
def _cadence_run(where: Path, limit: float | None) -> list[dict[str, Any]]:
    fake = FakeTime(T_START)
    harness = harness_for(where, limit=limit, telemetry=telemetry_on(fake, where))
    recorder = Recorder(harness)
    recorder.batch = 5  # today's recorder: the open day re-rendered every 300 s
    assert serve(harness, fake, until=T_START + 40 * 60, hooks=(recorder,)) == 0
    return harness.records()


def test_todays_300s_producer_stalls_between_publications_and_decides_the_same(tmp_path):
    """The dependency on R1-g, stated as a test rather than a promise.

    Today's recorder re-renders the open day at most every 300 s, which is longer
    than the committed 180 s limit, so between two publications the newest close
    really is older than the limit and the gate says so: FEED_STALLED and
    FEED_RESUMED pairs, on a recorder whose streams are healthy. R1-f does not
    make the recorder faster (R1-g does). What this shows is that those stalls
    change no decision: every non-stall record is the one a run with the limit
    out of reach writes, apart from the chain fields the extra records shift and
    the configuration hash the different limit changes.
    """
    gated = _cadence_run(tmp_path / "gated", None)
    ungated = _cadence_run(tmp_path / "ungated", OUT_OF_REACH)

    kinds = [r["kind"] for r in gated]
    assert kinds.count("FEED_STALLED") >= 3
    assert kinds.count("FEED_STALLED") - kinds.count("FEED_RESUMED") in (0, 1)
    assert not {"FEED_STALLED", "FEED_RESUMED"} & {r["kind"] for r in ungated}

    volatile = {"seq", "prev_hash", "record_hash", "config_hash"}

    def comparable(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {k: v for k, v in r.items() if k not in volatile}
            for r in records
            if r["kind"] not in ("FEED_STALLED", "FEED_RESUMED", "STARTUP")
        ]

    assert comparable(gated) == comparable(ungated)
    assert [r["kind"] for r in gated].count("DECISION") >= 20
