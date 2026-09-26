"""R1-g: the recorder publishes each minute within seconds, the runner decides
every one of them, and a funding minute waits for its settlement.

Master roadmap R1-g: "the recorder publishes the last closed minute
incrementally (seconds, not the 300 s full-day cadence); the runner decides
every closed minute since its last decided minute (no three-minute cap) ...; a
minute whose close coincides with a funding instant is decidable only once its
settlement row exists in the recorder output (or the recorder has marked the
instant as no-settlement) -- a late settlement row defers the decision rather
than skipping the booking; two-sided test: live and replay book the same
settlement in the same minute; acceptance: zero SKIPPED_STALE minutes while the
recorder is healthy."

Three layers, each tested against its own two-sided control:

* the recorder -- a minute is published once it has SETTLED (every stream folded
  into its row has passed its close, or a cap has), by a publisher that renders
  when a minute settles, serialised with every other render and freeze;
* the recorder's funding observation -- the windows of successful, complete,
  fully recorded polls, and nothing else;
* the runner -- no cap, a funding deferral that leaves nothing behind, and a
  day caught mid-rewrite treated as "not yet" rather than as a halt.

Engineering evidence only: synthetic minutes, a dry-run venue, no research
artifact read.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from chimera.demo.decision_log import iso_minute
from chimera.demo.feed import SETTLEMENT_QUERY_ALLOWANCE_MS
from chimera.demo.fixtures import MinuteShape
from chimera.demo.runner import RunnerState
from chimera.recorder.contract import load_recorder_contract
from chimera.recorder.events import (
    NS_PER_MILLISECOND,
    SPOT_BOOK_TICKER,
    SPOT_KLINE_1M,
    UM_FUNDING,
    UM_KLINE_1M,
    UM_MARK_PRICE,
)
from chimera.recorder.normalize import (
    FUNDING_OBSERVATION_SCHEMA,
    read_funding_observation,
)
from chimera.recorder.rest import FUNDING_POLL_DELAY_S, MAX_FUNDING_LIMIT, RestPoller
from chimera.recorder.service import (
    NORMALIZE_INTERVAL_S,
    PUBLISH_INTERVAL_S,
    PUBLISH_SETTLE_CAP_S,
    RecorderService,
)
from chimera.recorder.sink import RecorderSinkError
from tests.demo_harness import DAY, build
from tests.recorder_synthetic import DAY as RDAY
from tests.recorder_synthetic import NEXT_DAY as RNEXT
from tests.recorder_synthetic import (
    book_event,
    funding_rest_row,
    kline_event,
    mark_event,
    minute_ms,
)
from tests.test_r1f_real_staleness import TODAY_BATCH, healthy_run, kinds_of
from tests.test_recorder_rest_fake import FakeResponse, FakeSession
from tests.test_recorder_service import ScriptedClient
from tools.replay_parity import compare_logs

CONTRACT = load_recorder_contract()
NS = 1_000_000_000
MINUTE = 60_000


# =========================================================================== #
# the recorder: publication
# =========================================================================== #
class Wall:
    """The recorder's injected wall clock, moved by the test."""

    def __init__(self, ms: int) -> None:
        self.ns = ms * NS_PER_MILLISECOND

    def __call__(self) -> int:
        return self.ns

    def at(self, ms: int) -> None:
        self.ns = ms * NS_PER_MILLISECOND


def recorder(tmp_path: Path, wall: Wall, answers=(), **options) -> RecorderService:
    """A recovered service with no network: fake REST, events fed by hand."""
    poller = RestPoller(
        session=FakeSession(list(answers)), sleep=lambda s: None, min_interval_s=0.0
    )
    service = RecorderService(
        CONTRACT, tmp_path, poller=poller, clients=[], wall_ns=wall, gapfill=False, **options
    )
    service.recover()
    return service


def rows(service: RecorderService, market: str, day: str = RDAY) -> pd.DataFrame:
    path = service.normalizer.parquet_path(market, day)
    if not path.exists():
        return pd.DataFrame({"minute_open_ms": []})
    return pd.read_parquet(path)


def published(service: RecorderService, market: str, day: str = RDAY) -> list[int]:
    return [int(m) for m in rows(service, market, day)["minute_open_ms"]]


def minute_zero(service: RecorderService, *, spot_book: bool = True) -> None:
    """Every stream's observations inside minute 0, and nothing at or after its close."""
    opened = minute_ms(0)
    for event in (
        kline_event(opened),
        mark_event(opened + 5_000, mark="60050.00"),
        mark_event(opened + 55_000, mark="60060.00"),
        book_event(1, event_ms=opened + 50_000),
        kline_event(opened, stream=SPOT_KLINE_1M),
    ):
        service._record(event)
    if spot_book:
        service._record(
            book_event(
                2,
                stream=SPOT_BOOK_TICKER,
                event_ms=None,
                receipt_wall_ns=(opened + 40_000) * NS_PER_MILLISECOND,
            )
        )


def past_the_close(service: RecorderService, *, spot_book: bool = True) -> None:
    """The first observation of every stream after minute 0 closed: minute 1 forming."""
    close = minute_ms(1)
    for event in (
        kline_event(close, closed=False),
        mark_event(close + 1_000),
        book_event(3, event_ms=close + 500),
        kline_event(close, stream=SPOT_KLINE_1M, closed=False),
    ):
        service._record(event)
    if spot_book:
        service._record(
            book_event(
                4,
                stream=SPOT_BOOK_TICKER,
                event_ms=None,
                receipt_wall_ns=(close + 700) * NS_PER_MILLISECOND,
            )
        )


def test_the_publisher_runs_on_a_seconds_cadence_beside_the_300s_maintenance():
    assert PUBLISH_INTERVAL_S <= 1.0 and PUBLISH_SETTLE_CAP_S <= 10.0
    assert NORMALIZE_INTERVAL_S == 300.0, "maintenance keeps its own cadence"


def test_a_minute_is_published_only_once_every_stream_has_passed_its_close(tmp_path):
    """A closed kline is not a settled minute: its last mark can still be on the way.

    Minute 0's closed candle is in, and so is every other stream's data inside
    it -- but no stream has yet said anything at or after the close. A mark
    stamped 59 s into the minute then arrives late (another socket). Only when
    every stream has passed the close is the minute published, and the row it is
    published with is the one that includes the late mark: the row a runner
    reads is the row the finished file will hold.
    """
    wall = Wall(minute_ms(1) + 2_000)  # two seconds after the close: the cap is far off
    service = recorder(tmp_path, wall)
    minute_zero(service)
    asyncio.run(service._publish())
    assert published(service, "um") == [] and published(service, "spot") == []

    service._record(mark_event(minute_ms(0) + 59_000, mark="61000.00"))
    asyncio.run(service._publish())
    assert published(service, "um") == []

    past_the_close(service)
    asyncio.run(service._publish())
    assert published(service, "um") == [minute_ms(0)]
    assert published(service, "spot") == [minute_ms(0)]
    assert float(rows(service, "um")["mark_close"].iloc[0]) == 61000.0
    # Minute 1's partial candle is never a published minute.
    assert minute_ms(1) not in published(service, "um")


def test_a_quiet_stream_holds_a_minute_back_only_until_the_cap(tmp_path):
    """The two-sided control: spot's book never speaks again. The minute is not
    held for ever -- it is published once the recorder's clock is the cap past
    its close, with the silent stream's column empty, as it always was."""
    wall = Wall(minute_ms(1) + 2_000)
    service = recorder(tmp_path, wall)
    minute_zero(service, spot_book=False)
    past_the_close(service, spot_book=False)
    asyncio.run(service._publish())
    assert published(service, "um") == []

    wall.at(minute_ms(1) + int(PUBLISH_SETTLE_CAP_S * 1000) - 1)
    asyncio.run(service._publish())
    assert published(service, "um") == []

    wall.at(minute_ms(1) + int(PUBLISH_SETTLE_CAP_S * 1000))
    asyncio.run(service._publish())
    assert published(service, "um") == [minute_ms(0)]
    assert published(service, "spot") == [minute_ms(0)]
    assert not bool(rows(service, "spot")["book_present"].iloc[0])


def test_the_running_service_publishes_without_waiting_for_maintenance(tmp_path):
    """End to end: with maintenance five minutes away, the publisher alone makes
    the delivered minutes visible while the service is still running.

    The recorder's clock runs twenty times faster than the test's, starting just
    after minute 1 closed, so the cap settles minute 1 within half a second."""
    began = time.monotonic()
    start_ns = (minute_ms(2) + 5_000) * NS_PER_MILLISECOND

    def wall() -> int:
        return start_ns + int((time.monotonic() - began) * 20 * NS)

    poller = RestPoller(session=FakeSession([]), sleep=lambda s: None, min_interval_s=0.0)
    service = RecorderService(
        CONTRACT,
        tmp_path,
        poller=poller,
        clients=[],
        wall_ns=wall,
        gapfill=False,
        heartbeat_interval_s=0.05,
        sync_interval_s=0.05,
        normalize_interval_s=NORMALIZE_INTERVAL_S,
        publish_interval_s=0.05,
        premium_index_interval_s=3600.0,
        funding_catchup_interval_s=3600.0,
    )
    events = [kline_event(minute_ms(0)), kline_event(minute_ms(1))]
    service.clients = (
        ScriptedClient("um-test", (UM_KLINE_1M, UM_MARK_PRICE), service._record, events),
    )
    seen: dict[str, Any] = {}

    async def scenario():
        stop = asyncio.Event()

        async def look():
            await asyncio.sleep(0.5)
            seen["during"] = published(service, "um")
            stop.set()

        await asyncio.gather(service.run(stop), look())

    asyncio.run(scenario())
    assert seen["during"] == [minute_ms(0), minute_ms(1)]


def test_renders_and_freezes_never_overlap(tmp_path):
    """The publisher, the maintenance pass and the freeze share one lock.

    Two renders and a freeze started together, each slowed down inside: at no
    instant do two of them run. Without the lock they advance the same
    incremental cache from under each other, and a freeze moves files a render
    is reading.
    """
    wall = Wall(minute_ms(12 * 60))
    service = recorder(tmp_path, wall)
    service._record(kline_event(minute_ms(0)))
    busy = {"now": 0, "most": 0}
    guard = threading.Lock()

    def occupied(work):
        def run(*args, **kwargs):
            with guard:
                busy["now"] += 1
                busy["most"] = max(busy["most"], busy["now"])
            time.sleep(0.05)
            try:
                return work(*args, **kwargs)
            finally:
                with guard:
                    busy["now"] -= 1

        return run

    service.incremental.build_day = occupied(service.incremental.build_day)
    for sink in service.sinks.values():
        sink.freeze_day = occupied(lambda *a, **k: None)
    service._frozen_after = {RDAY: 0.0}
    threads = [
        threading.Thread(target=service._render, args=([RDAY],)),
        threading.Thread(target=service._render, args=([RDAY],)),
        threading.Thread(target=service._freeze_due),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert busy["most"] == 1


def test_one_pass_is_one_horizon_with_the_perpetual_last(tmp_path):
    """Every market of a pass gets the same horizon, and the perpetual -- the
    market the runner walks -- is written last, so a minute the perpetual shows
    is always already in the spot file. The wall moves during the pass to make a
    per-market horizon visible."""
    wall = Wall(minute_ms(12 * 60))
    service = recorder(tmp_path, wall)
    calls: list[tuple[str, int]] = []
    original = service.incremental.build_day

    def spy(market, day, **kwargs):
        calls.append((market, kwargs["through_ms"]))
        wall.ns += 90 * NS
        return original(market, day, **kwargs)

    service.incremental.build_day = spy
    service._render([RDAY])
    assert [market for market, _ in calls] == ["spot", "um"]
    assert len({through for _, through in calls}) == 1


def test_the_horizon_never_moves_back(tmp_path):
    """A clock stepped back does not unpublish a minute a reader may have taken."""
    wall = Wall(minute_ms(30))
    service = recorder(tmp_path, wall)
    for index in range(20):
        service._record(kline_event(minute_ms(index)))
    service._render([RDAY])
    before = published(service, "um")
    assert before == [minute_ms(i) for i in range(20)]
    wall.at(minute_ms(5))
    service._render([RDAY])
    assert published(service, "um") == before


def test_the_last_minute_of_a_day_is_published_across_midnight(tmp_path):
    """Minutes that settle together on both sides of midnight are published in
    both days by one pass: 23:59 lives in the day that has just ended."""
    last = minute_ms(1439)
    wall = Wall(minute_ms(0, day=RNEXT))
    service = recorder(tmp_path, wall)
    for opened in (last, minute_ms(0, day=RNEXT), minute_ms(1, day=RNEXT)):
        service._record(kline_event(opened))
        service._record(kline_event(opened, stream=SPOT_KLINE_1M))
    service._published_through_ms = last  # 23:58 was the newest minute published
    wall.at(minute_ms(2, day=RNEXT) + int(PUBLISH_SETTLE_CAP_S * 1000))
    asyncio.run(service._publish())
    assert published(service, "um", RDAY)[-1:] == [last]
    assert published(service, "um", RNEXT) == [
        minute_ms(0, day=RNEXT),
        minute_ms(1, day=RNEXT),
    ]


# =========================================================================== #
# the recorder: the funding observation
# =========================================================================== #
def observed(service: RecorderService) -> tuple[int, int] | None:
    return read_funding_observation(service.normalizer.funding_observation_path("um"), "um")


def settlement_instants(service: RecorderService) -> list[int]:
    path = service.normalizer.settlements_path("um")
    if not path.exists():
        return []
    return [
        json.loads(line)["funding_time_ms"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


NOW = minute_ms(8 * 60 + 10)  # 08:10, ten minutes after the 08:00 settlement
LOOKBACK_START = NOW - 3 * 24 * 60 * MINUTE


def test_a_complete_poll_observes_exactly_its_query_window(tmp_path):
    """The bound is what was ASKED -- the request's own endTime -- never a row's
    instant: a settlement instant is what is being asked about, not the moment
    the recorder knew it. The rows are on disk before the observation says so."""
    answer = FakeResponse(
        payload=[funding_rest_row(minute_ms(0)), funding_rest_row(minute_ms(480))]
    )
    service = recorder(tmp_path, Wall(NOW), answers=[answer])
    assert asyncio.run(service.poll_funding()) == 2
    assert observed(service) == (LOOKBACK_START, NOW)
    assert settlement_instants(service) == [minute_ms(0), minute_ms(480)]
    document = json.loads(
        service.normalizer.funding_observation_path("um").read_text(encoding="utf-8")
    )
    assert document["schema"] == FUNDING_OBSERVATION_SCHEMA


def test_an_empty_answer_observes_its_window_and_invents_no_settlement(tmp_path):
    service = recorder(tmp_path, Wall(NOW), answers=[FakeResponse(payload=[])])
    assert asyncio.run(service.poll_funding()) == 0
    assert observed(service) == (LOOKBACK_START, NOW)
    assert settlement_instants(service) == []


@pytest.mark.parametrize(
    "answer",
    [
        FakeResponse(status_code=400, payload={"code": -1, "msg": "no"}),
        FakeResponse(payload=[funding_rest_row(minute_ms(0))] * MAX_FUNDING_LIMIT),
        FakeResponse(payload=[funding_rest_row(minute_ms(0)), {"symbol": "BTCUSDT"}]),
    ],
    ids=["failed", "cut-off-at-the-limit", "a-row-that-is-not-a-settlement"],
)
def test_a_poll_that_is_not_complete_observes_nothing(tmp_path, answer):
    service = recorder(tmp_path, Wall(NOW), answers=[answer])
    asyncio.run(service.poll_funding())
    assert observed(service) is None


def test_a_halted_funding_stream_observes_nothing(tmp_path):
    """Rows the recorder refused to store were not recorded, whatever came back."""
    service = recorder(tmp_path, Wall(NOW), answers=[FakeResponse(payload=[])])
    service.health.stream(UM_FUNDING).halt("synthetic storage failure", now_ns=NOW * 1_000_000)
    asyncio.run(service.poll_funding())
    assert observed(service) is None


def test_the_observation_extends_only_across_windows_that_touch(tmp_path):
    """One interval, never one with a hole in it. A later window that overlaps
    extends it; one that starts after it ended replaces it, so the instants in
    between are covered by nothing."""
    service = recorder(tmp_path, Wall(NOW))
    service._observe_funding(1_000, 5_000)
    service._observe_funding(4_000, 9_000)
    assert observed(service) == (1_000, 9_000)
    service._observe_funding(9_001, 12_000)
    assert observed(service) == (1_000, 12_000), "adjacent windows touch"
    service._observe_funding(20_000, 30_000)
    assert observed(service) == (20_000, 30_000), "a gap is never bridged"
    service._observe_funding(1_000, 2_000)
    assert observed(service) == (20_000, 30_000), "an older window says nothing new"


def test_an_observation_that_cannot_be_written_does_not_stop_the_recorder(
    tmp_path, monkeypatch
):
    """The atomic writer raises RecorderSinkError, a RuntimeError, not OSError.
    Escaping the funding task would take the whole service down over this file."""
    from chimera.recorder import service as service_module

    service = recorder(
        tmp_path, Wall(NOW), answers=[FakeResponse(payload=[funding_rest_row(minute_ms(0))])]
    )
    written = service_module.write_json_atomic

    def refuse(path, document):
        if Path(path).name == "observed.json":
            raise RecorderSinkError("disk full")
        return written(path, document)

    monkeypatch.setattr(service_module, "write_json_atomic", refuse)
    assert asyncio.run(service.poll_funding()) == 1
    assert observed(service) is None
    assert settlement_instants(service) == [minute_ms(0)]
    assert any("funding observation not written" in e for e in service.health.errors)


# =========================================================================== #
# the runner: every minute, in order, and a stop in the middle of them
# =========================================================================== #
def m(index: int) -> int:
    return int(pd.Timestamp(DAY, tz="UTC").timestamp() * 1000) + index * MINUTE


def iso(index: int) -> str:
    return iso_minute(m(index) * 1_000_000)


def minutes_of(harness, kind: str) -> list[str]:
    return [r["minute"] for r in harness.records() if r["kind"] == kind]


def test_a_long_backlog_is_decided_whole_in_order_with_no_skip(tmp_path):
    harness = build(tmp_path)
    outcomes = harness.runner.catch_up(now_ms=m(149))
    assert [o.minute_ms for o in outcomes] == [m(i) for i in range(150)]
    assert minutes_of(harness, "DECISION") == [iso(i) for i in range(150)]
    assert "SKIPPED_STALE" not in kinds_of(harness)


def test_a_stop_in_the_middle_of_a_backlog_resumes_at_the_next_minute(tmp_path):
    """R1-d's stop is asked between minutes. Without a cap a backlog can be a day
    long, so the stop must bound it: the minute in hand completes, the cursor is
    that minute, and the next start takes the one after it -- nothing skipped,
    nothing twice."""
    harness = build(tmp_path)
    asked = {"n": 0}

    def stop() -> bool:
        asked["n"] += 1
        return asked["n"] > 37

    outcomes = harness.runner.catch_up(now_ms=m(99), stop=stop)
    assert len(outcomes) == 37
    assert harness.runner.cursor.last_minute_processed == m(36)
    persisted = json.loads(
        (harness.state_dir / "runner_state.json").read_text(encoding="utf-8")
    )
    assert persisted["last_minute_processed"] == m(36)
    config = harness.runner.config
    harness.runner.shutdown("stop requested")

    again = build(tmp_path, config=config, start=False)
    again.runner.start()
    assert again.runner.cursor.next_minute_ms() == m(37)
    again.runner.catch_up(now_ms=m(99))
    assert minutes_of(again, "DECISION") == [iso(i) for i in range(100)]
    assert "SKIPPED_STALE" not in kinds_of(again)


def test_a_healthy_per_minute_recorder_has_every_minute_decided_and_replays(tmp_path):
    """R1-g's acceptance witness on R1-f's service harness: the recorder
    publishing every minute two seconds after it closes, for two operational
    hours, under the committed configuration and the heartbeat gate. Every
    minute is decided, none is skipped, the feed never stalls, and a replay of
    the finished files reaches PARITY."""
    live, recorder_, _ = healthy_run(tmp_path / "live", batch=1)
    kinds = kinds_of(live)
    assert "SKIPPED_STALE" not in kinds and "FEED_STALLED" not in kinds
    decided = minutes_of(live, "DECISION")
    assert decided == [iso(i) for i in range(recorder_.count)] and len(decided) >= 120

    from tests.test_r1f_real_staleness import harness_for, minute_ms as r1f_minute

    replay = harness_for(tmp_path / "replay", count=recorder_.count)
    replay.runner.replay(r1f_minute(0), r1f_minute(recorder_.count - 1))
    replay.runner.shutdown("replay")
    report = compare_logs(live.records(), replay.records())
    assert report.ok and report.label == "PARITY", report.to_dict()


def test_todays_300s_batches_still_never_stall_and_now_skip_nothing(tmp_path):
    """R1-f's integration witness under R1-g: the slowest healthy cadence the gate
    must tolerate still raises no stall, and now every minute is decided."""
    live, recorder_, _ = healthy_run(tmp_path, batch=TODAY_BATCH)
    kinds = kinds_of(live)
    assert "FEED_STALLED" not in kinds and "SKIPPED_STALE" not in kinds
    assert minutes_of(live, "DECISION") == [iso(i) for i in range(recorder_.count)]


# =========================================================================== #
# the runner: the funding minute
# =========================================================================== #
#: 07:30 to 08:59: the carry position is open within two minutes, and held over
#: the 08:00 settlement, whose minute is 07:59 (it closes at 08:00).
WINDOW = range(450, 540)
DUE = 479
INSTANT = m(480)
LAST = 489


def funding_world(where: Path, *, hours=(0, 8, 16), config=None, start=True):
    absent = {s: MinuteShape(present=False) for s in range(1440) if s not in WINDOW}
    harness = build(
        where,
        shapes={f"um:{DAY}": absent, f"spot:{DAY}": absent},
        config=config,
        start=False,
    )
    harness.feed.write_settlements([DAY], hours=hours)
    if start:
        harness.runner.start()
    return harness


def fundings(harness) -> list[tuple[str, int]]:
    return [
        (r["minute"], r["funding"]["settlement_instant_ns"] // 1_000_000)
        for r in harness.records()
        if r["kind"] == "FUNDING"
    ]


def state_bytes(state_dir: Path) -> dict[str, bytes]:
    return {
        p.relative_to(state_dir).as_posix(): p.read_bytes()
        for p in sorted(state_dir.rglob("*"))
        if p.is_file()
    }


def test_a_settlement_already_recorded_is_booked_in_its_own_minute_once(tmp_path):
    harness = funding_world(tmp_path)
    outcomes = harness.runner.catch_up(now_ms=m(LAST))
    assert all(o.kind is not None for o in outcomes)
    assert fundings(harness) == [(iso(DUE), INSTANT)]
    assert minutes_of(harness, "DECISION") == [iso(i) for i in range(WINDOW.start, LAST + 1)]


def test_a_late_settlement_defers_its_minute_and_is_booked_there_once(tmp_path):
    """The row is not there when the minute is reached: the minute waits, writes
    nothing, moves nothing -- not the cursor, not the clock, not a byte of state
    -- however many times it is tried. The row arrives: the SAME minute is
    decided and books it, once, and the minutes after it follow."""
    harness = funding_world(tmp_path, hours=(0, 16))
    runner = harness.runner
    outcomes = runner.catch_up(now_ms=m(LAST))
    assert outcomes[-1].minute_ms == m(DUE) and outcomes[-1].kind is None
    assert "funding instant" in outcomes[-1].detail
    assert runner.cursor.last_minute_processed == m(DUE - 1)
    assert iso(DUE) not in [r["minute"] for r in harness.records()]
    assert runner.state is RunnerState.READY

    clock, frozen = runner.clock.now_ns, state_bytes(harness.state_dir)
    for _ in range(5):
        again = runner.catch_up(now_ms=m(LAST))
        assert [(o.minute_ms, o.kind) for o in again] == [(m(DUE), None)]
    assert runner.clock.now_ns == clock, "a deferral observed the clock"
    assert state_bytes(harness.state_dir) == frozen, "a deferral wrote something"

    harness.feed.write_settlements([DAY], hours=(0, 8, 16))
    runner.catch_up(now_ms=m(LAST))
    assert fundings(harness) == [(iso(DUE), INSTANT)]
    assert minutes_of(harness, "DECISION") == [iso(i) for i in range(WINDOW.start, LAST + 1)]
    runner.catch_up(now_ms=m(LAST))
    assert len(fundings(harness)) == 1


def test_run_minutes_and_replay_stop_at_a_deferral_too(tmp_path):
    """The path a replay drives. Stepping past the waiting minute would decide
    later minutes ahead of it and drop it from the log."""
    harness = funding_world(tmp_path, hours=(0, 16))
    runner = harness.runner
    outcomes = runner.run_minutes([m(i) for i in range(WINDOW.start, LAST + 1)])
    assert outcomes[-1].minute_ms == m(DUE) and outcomes[-1].kind is None
    assert runner.cursor.last_minute_processed == m(DUE - 1)
    assert runner.replay(m(DUE), m(LAST))[-1].kind is None
    assert runner.cursor.last_minute_processed == m(DUE - 1)
    assert [r["minute"] for r in harness.records()][-1] == iso(DUE - 1)


def test_a_restart_while_deferred_takes_the_deferred_minute_first(tmp_path):
    harness = funding_world(tmp_path, hours=(0, 16))
    config = harness.runner.config
    harness.runner.catch_up(now_ms=m(LAST))
    harness.runner.shutdown("restart while deferred")

    again = funding_world(tmp_path, hours=(0, 16), config=config)
    first = again.runner.catch_up(now_ms=m(LAST))
    assert [(o.minute_ms, o.kind) for o in first] == [(m(DUE), None)]
    again.feed.write_settlements([DAY], hours=(0, 8, 16))
    again.runner.catch_up(now_ms=m(LAST))
    assert fundings(again) == [(iso(DUE), INSTANT)]
    decided = minutes_of(again, "DECISION")
    assert decided == [iso(i) for i in range(WINDOW.start, LAST + 1)]


def test_live_and_replay_book_the_late_settlement_in_the_same_minute(tmp_path):
    """The roadmap's two-sided test. Live: the minute is reached before its row,
    defers, and books it when it lands. Replay: the finished files, every row
    already there. Same settlement, same minute, and PARITY over the whole log.

    The control removes the deferral: the live run then books the settlement a
    minute LATER than the replay does, and section 10's comparison says so."""

    def live(where: Path, *, defer: bool):
        harness = funding_world(where, hours=(0, 16))
        if not defer:
            harness.runner.cursor.funding_pending = lambda minute: None
        harness.runner.catch_up(now_ms=m(DUE + 1))
        harness.feed.write_settlements([DAY], hours=(0, 8, 16))
        harness.runner.catch_up(now_ms=m(LAST))
        harness.runner.shutdown("live")
        return harness

    replay = funding_world(tmp_path / "replay")
    replay.runner.replay(m(WINDOW.start), m(LAST))
    replay.runner.shutdown("replay")
    assert fundings(replay) == [(iso(DUE), INSTANT)]

    deferred = live(tmp_path / "deferred", defer=True)
    assert fundings(deferred) == fundings(replay)
    report = compare_logs(deferred.records(), replay.records())
    assert report.ok and report.label == "PARITY", report.to_dict()

    racing = live(tmp_path / "racing", defer=False)
    assert fundings(racing) == [(iso(DUE + 2), INSTANT)]
    assert not compare_logs(racing.records(), replay.records()).ok


def test_an_observed_instant_with_no_row_is_decided_and_books_nothing(tmp_path):
    """The no-settlement arm: the recorder asked, late enough, and there was no
    row. The minute is decided; nothing is booked, and nothing is invented."""
    harness = funding_world(tmp_path, hours=(0, 16))
    harness.feed.write_funding_observation(
        INSTANT - 3 * 24 * 60 * MINUTE, INSTANT + SETTLEMENT_QUERY_ALLOWANCE_MS
    )
    outcomes = harness.runner.catch_up(now_ms=m(LAST))
    assert all(o.kind is not None for o in outcomes)
    assert fundings(harness) == []
    assert INSTANT * 1_000_000 not in harness.runner.position.ledger.state.settled
    assert minutes_of(harness, "DECISION") == [iso(i) for i in range(WINDOW.start, LAST + 1)]


@pytest.mark.parametrize(
    "observation",
    [
        None,
        (INSTANT - MINUTE, INSTANT + SETTLEMENT_QUERY_ALLOWANCE_MS - 1),
        (INSTANT + 1, INSTANT + 10 * MINUTE),
        "garbage",
        "another-market",
    ],
    ids=[
        "absent",
        "asked-too-soon",
        "window-starts-after-the-instant",
        "unreadable",
        "not-um",
    ],
)
def test_no_observation_short_of_covering_the_instant_counts_as_no_settlement(
    tmp_path, observation
):
    """Absence is never read as "none": each of these is a recorder that has not
    shown it asked late enough, and the minute keeps waiting for its row."""
    harness = funding_world(tmp_path, hours=(0, 16))
    path = harness.feed.normalizer.funding_observation_path("um")
    if isinstance(observation, tuple):
        harness.feed.write_funding_observation(*observation)
    elif observation == "garbage":
        path.write_text("{ not json", encoding="utf-8")
    elif observation == "another-market":
        harness.feed.write_funding_observation(INSTANT - MINUTE, INSTANT + 10 * MINUTE)
        document = json.loads(path.read_text(encoding="utf-8"))
        path.write_text(json.dumps({**document, "market": "spot"}), encoding="utf-8")
    outcomes = harness.runner.catch_up(now_ms=m(LAST))
    assert outcomes[-1].minute_ms == m(DUE) and outcomes[-1].kind is None


def test_the_query_allowance_is_inside_the_recorders_own_poll_delay():
    assert 0 < SETTLEMENT_QUERY_ALLOWANCE_MS < FUNDING_POLL_DELAY_S * 1000


def test_a_venue_row_stamped_off_the_schedule_still_answers_it(tmp_path):
    """A settlement row a few milliseconds after the announced instant is still
    that instant's answer: the minute proceeds, and the row is booked in the
    window it falls in -- the next minute's, in live and replay alike."""
    harness = funding_world(tmp_path, hours=(0, 16))
    path = harness.feed.normalizer.settlements_path("um")
    lines = path.read_text(encoding="utf-8").splitlines()
    row = {**json.loads(lines[0]), "funding_time_ms": INSTANT + 3}
    path.write_text(
        "\n".join([lines[0], json.dumps(row), *lines[1:]]) + "\n", encoding="utf-8"
    )
    outcomes = harness.runner.catch_up(now_ms=m(LAST))
    assert all(o.kind is not None for o in outcomes)
    assert fundings(harness) == [(iso(DUE + 1), INSTANT + 3)]


# =========================================================================== #
# the runner: a day caught mid-rewrite
# =========================================================================== #
def torn(path: Path) -> bytes:
    """Cut a file to half its length, as a reader inside an in-place rewrite sees it."""
    whole = path.read_bytes()
    path.write_bytes(whole[: len(whole) // 2])
    return whole


def test_a_day_caught_mid_rewrite_is_waited_for_not_halted_on(tmp_path, caplog):
    harness = build(tmp_path)
    runner = harness.runner
    runner.catch_up(now_ms=m(9))
    spot = harness.feed.normalizer.parquet_path("spot", DAY)
    whole = torn(spot)

    with caplog.at_level(logging.WARNING):
        outcomes = runner.catch_up(now_ms=m(19))
    assert [(o.minute_ms, o.kind) for o in outcomes] == [(m(10), None)]
    assert runner.state is RunnerState.READY and runner.cursor.last_minute_processed == m(9)
    assert "deferred" in caplog.text
    assert iso(10) not in [r["minute"] for r in harness.records()]

    spot.write_bytes(whole)
    runner.catch_up(now_ms=m(19))
    assert minutes_of(harness, "DECISION") == [iso(i) for i in range(20)]
    assert "HALT" not in kinds_of(harness)


def test_a_day_that_stays_unreadable_is_still_refused(tmp_path):
    """The other side: the same broken file, unchanged, a second time is not
    being rewritten -- it is broken, and the runner halts on it as it always
    did, rather than wait for ever."""
    harness = build(tmp_path)
    runner = harness.runner
    runner.catch_up(now_ms=m(9))
    torn(harness.feed.normalizer.parquet_path("um", DAY))
    first = runner.catch_up(now_ms=m(19))
    assert [(o.minute_ms, o.kind) for o in first] == [(m(10), None)]
    second = runner.catch_up(now_ms=m(19))
    assert runner.state is RunnerState.HALT
    assert "has not changed since it last failed" in runner.halt_reason
    assert [o.kind.value for o in second] == ["HALT"]
