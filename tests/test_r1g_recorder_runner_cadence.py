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

Two layers, each tested against its own two-sided control:

* the recorder -- a minute is published once it has SETTLED (every stream folded
  into its row has passed its close, or a cap has), by a publisher that renders
  when a minute settles, serialised with every other render and freeze;
* the runner -- no cap, a funding deferral that leaves nothing behind and ends
  only when the instant's own settlement row exists, and a day caught
  mid-rewrite treated as "not yet" rather than as a halt.

The roadmap's second arm ("marked as no-settlement") is not implemented: no
authoritative source bounds how late the venue may publish a row, so nothing the
recorder can observe proves a settlement did not happen (owner decision after
the independent review of PR #108, finding F1).

Engineering evidence only: synthetic minutes, a dry-run venue, no research
artifact read.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from chimera import metrics
from chimera.carry.hedge import HedgeState
from chimera.demo.decision_log import iso_minute
from chimera.demo.fixtures import MinuteShape
from chimera.demo.runner import RunnerState
from chimera.recorder.contract import load_recorder_contract
from chimera.recorder.events import (
    NS_PER_MILLISECOND,
    SPOT_BOOK_TICKER,
    SPOT_KLINE_1M,
    UM_KLINE_1M,
    UM_MARK_PRICE,
)
from chimera.recorder.rest import RestPoller
from chimera.recorder.service import (
    NORMALIZE_INTERVAL_S,
    PUBLISH_INTERVAL_S,
    PUBLISH_SETTLE_CAP_S,
    RecorderService,
)
from tests.demo_harness import DAY, build, write_heartbeat
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
    # Nor does any other render path -- the maintenance pass here, shutdown and
    # recovery alike -- publish it early: the horizon is applied inside the render.
    asyncio.run(service._maintain())
    assert published(service, "um") == [] and published(service, "spot") == []

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


def capped(tmp_path: Path) -> tuple[RecorderService, Wall]:
    """Minute 0 published by the cap, with spot's book silent through it."""
    wall = Wall(minute_ms(1) + 2_000)
    service = recorder(tmp_path, wall)
    minute_zero(service, spot_book=False)
    past_the_close(service, spot_book=False)
    wall.at(minute_ms(1) + int(PUBLISH_SETTLE_CAP_S * 1000))
    asyncio.run(service._publish())
    assert published(service, "spot") == [minute_ms(0)]
    assert not bool(rows(service, "spot")["book_present"].iloc[0])
    return service, wall


def test_a_row_published_by_the_cap_is_not_final(tmp_path):
    """F3, and the limit it states. The cap publishes a minute some stream has
    not passed, so an event stamped inside it can still arrive, and the next
    render changes the row -- the reviewer's case: a mark stamped 58 s into the
    minute, delivered after the cap published it. A runner that decided the
    first row disagrees with a replay of the finished file; nothing in R1-g
    makes a capped row final (reconciling a day that changed is R1-h's)."""
    service, _ = capped(tmp_path)
    assert float(rows(service, "um")["mark_close"].iloc[0]) == 60060.0
    service._record(mark_event(minute_ms(0) + 58_000, mark="61234.00"))
    service._render([RDAY])
    assert published(service, "um") == [minute_ms(0)]
    assert float(rows(service, "um")["mark_close"].iloc[0]) == 61234.0


def test_a_late_event_inside_a_published_minute_republishes_at_the_next_check(tmp_path):
    """F4. A closed candle that arrives after the cap published its minute
    without it is rendered at the publisher's next check -- a second later --
    not when the horizon next crosses a close, up to a minute later. The
    control: an event for the minute still forming renders nothing."""
    wall = Wall(minute_ms(1) + 2_000)
    service = recorder(tmp_path, wall)
    opened = minute_ms(0)
    for event in (
        mark_event(opened + 5_000),
        book_event(1, event_ms=opened + 50_000),
        kline_event(opened, stream=SPOT_KLINE_1M),
        book_event(
            2,
            stream=SPOT_BOOK_TICKER,
            event_ms=None,
            receipt_wall_ns=(opened + 40_000) * 1_000_000,
        ),
    ):
        service._record(event)
    past_the_close(service)
    wall.at(minute_ms(1) + int(PUBLISH_SETTLE_CAP_S * 1000))
    asyncio.run(service._publish())
    assert published(service, "um") == [] and published(service, "spot") == [minute_ms(0)]

    renders: list[list[str]] = []
    render = service._render

    def spy(days):
        renders.append(list(days))
        return render(days)

    service._render = spy
    wall.at(minute_ms(1) + 12_000)  # the horizon floor is still minute 1's open
    service._record(kline_event(minute_ms(1), closed=False))
    asyncio.run(service._publish())
    assert renders == [], "an event for the forming minute rendered"

    service._record(kline_event(opened))  # minute 0's closed candle, 12 s late
    asyncio.run(service._publish())
    assert len(renders) == 1
    assert published(service, "um") == [minute_ms(0)]
    asyncio.run(service._publish())
    assert len(renders) == 1, "rendered again with nothing new"


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
# the recorder: a funding poll records rows and nothing else
# =========================================================================== #
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


def test_a_funding_poll_writes_the_rows_it_was_given_and_no_verdict_on_absence(tmp_path):
    """F1: nothing the recorder writes may stand for "no settlement". A poll
    records the rows that came back; an empty answer leaves nothing beside the
    settlements for a runner to read as an absence."""
    answer = FakeResponse(payload=[funding_rest_row(minute_ms(0))])
    service = recorder(tmp_path, Wall(NOW), answers=[answer, FakeResponse(payload=[])])
    assert asyncio.run(service.poll_funding()) == 1
    assert asyncio.run(service.poll_funding()) == 0
    assert settlement_instants(service) == [minute_ms(0)]
    funding_dir = service.normalizer.settlements_path("um").parent
    assert sorted(p.name for p in funding_dir.iterdir()) == [
        "settlements.ndjson",
        "settlements.sha256",
    ]


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


def window_through(last: int) -> dict[str, dict[int, MinuteShape]]:
    """The window's minutes up to ``last``, as a recorder that has got that far."""
    absent = {
        s: MinuteShape(present=False) for s in range(1440) if not WINDOW.start <= s <= last
    }
    return {f"um:{DAY}": absent, f"spot:{DAY}": absent}


def publish_through(harness, last: int) -> None:
    """The recorder publishes the window up to minute ``last``."""
    harness.feed.write_days([DAY], shapes=window_through(last))


def funding_world(
    where: Path, *, hours=(0, 8, 16), config=None, start=True, through=WINDOW.stop - 1
):
    harness = build(where, shapes=window_through(through), config=config, start=False)
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
    harness = funding_world(tmp_path, hours=(0, 16), through=DUE + 1)
    runner = harness.runner
    outcomes = runner.catch_up()
    assert outcomes[-1].minute_ms == m(DUE) and outcomes[-1].kind is None
    assert "funding instant" in outcomes[-1].detail
    assert runner.cursor.last_minute_processed == m(DUE - 1)
    assert iso(DUE) not in [r["minute"] for r in harness.records()]
    assert runner.state is RunnerState.READY

    clock, frozen = runner.clock.now_ns, state_bytes(harness.state_dir)
    for _ in range(5):
        again = runner.catch_up()
        assert [(o.minute_ms, o.kind) for o in again] == [(m(DUE), None)]
    assert runner.clock.now_ns == clock, "a deferral observed the clock"
    assert state_bytes(harness.state_dir) == frozen, "a deferral wrote something"

    harness.feed.write_settlements([DAY], hours=(0, 8, 16))
    publish_through(harness, LAST)
    runner.catch_up()
    assert fundings(harness) == [(iso(DUE), INSTANT)]
    assert minutes_of(harness, "DECISION") == [iso(i) for i in range(WINDOW.start, LAST + 1)]
    runner.catch_up()
    assert len(fundings(harness)) == 1


def test_run_minutes_and_replay_stop_at_a_deferral_too(tmp_path):
    """The path a replay drives. Stepping past the waiting minute would decide
    later minutes ahead of it and drop it from the log."""
    harness = funding_world(tmp_path, hours=(0, 16), through=DUE + 1)
    runner = harness.runner
    outcomes = runner.run_minutes([m(i) for i in range(WINDOW.start, LAST + 1)])
    assert outcomes[-1].minute_ms == m(DUE) and outcomes[-1].kind is None
    assert runner.cursor.last_minute_processed == m(DUE - 1)
    assert runner.replay(m(DUE), m(LAST))[-1].kind is None
    assert runner.cursor.last_minute_processed == m(DUE - 1)
    assert [r["minute"] for r in harness.records()][-1] == iso(DUE - 1)


@pytest.mark.parametrize("stop", ["shutdown", "crash"])
def test_a_restart_while_deferred_takes_the_deferred_minute_first(tmp_path, stop):
    """Clean or not. A crash leaves the files exactly as the deferral left them,
    so they must already say the deferred minute is the next one -- a shutdown
    would re-save the cursor and hide a deferral that had persisted a wrong one."""
    harness = funding_world(tmp_path, hours=(0, 16), through=DUE)
    config = harness.runner.config
    harness.runner.catch_up()
    if stop == "shutdown":
        harness.runner.shutdown("restart while deferred")
    before = len(harness.records())

    again = funding_world(tmp_path, hours=(0, 16), config=config, through=DUE + 1)
    restarted = [r["kind"] for r in again.records()[before:]]
    assert "RECOVERY" not in restarted and "HALT" not in restarted, restarted
    first = again.runner.catch_up()
    assert [(o.minute_ms, o.kind) for o in first] == [(m(DUE), None)]
    again.feed.write_settlements([DAY], hours=(0, 8, 16))
    publish_through(again, LAST)
    again.runner.catch_up()
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
        harness = funding_world(where, hours=(0, 16), through=DUE + 1)
        if not defer:
            harness.runner.cursor.funding_pending = lambda minute: None
        harness.runner.catch_up()
        harness.feed.write_settlements([DAY], hours=(0, 8, 16))
        publish_through(harness, LAST)
        harness.runner.catch_up()
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


def deferral_gauges() -> tuple[float, float] | None:
    """R1-g's two deferral gauges, or None where prometheus_client is absent."""
    if not metrics.PROMETHEUS_AVAILABLE:
        return None
    return (
        metrics.DEMO_FUNDING_DEFERRED._value.get(),
        metrics.DEMO_FUNDING_DEFERRED_INSTANT._value.get(),
    )


def legacy_observation(harness, *, through_ms: int) -> None:
    """The funding observation this PR's first head wrote and read, covering the
    instant with far more than its retired 30 s allowance: a successful poll
    long after the instant that came back without the instant's row."""
    path = harness.feed.normalizer.settlements_path("um").with_name("observed.json")
    path.write_text(
        json.dumps(
            {
                "schema": "chimera.recorder-funding-observation/1",
                "market": "um",
                "observed_from_ms": INSTANT - 3 * 24 * 60 * MINUTE,
                "observed_through_ms": through_ms,
            }
        ),
        encoding="utf-8",
    )


def test_a_row_later_than_any_poll_threshold_defers_and_is_booked_in_its_minute(tmp_path):
    """F1, the independent reviewer's arm. The due minute appears; a poll made
    well after the instant came back with no row; the recorder publishes minute
    after minute past the instant -- past the retired 30 s allowance, past a
    60 s one, to the edge of the declared bound -- and the minute still waits.
    The row lands late: the SAME minute is decided and books it, once, and a
    replay of the finished files books it in that minute too. No threshold on
    how long ago the recorder asked, or how far it has published, can stand for
    "no settlement": the only answer to an instant is its row."""
    harness = funding_world(tmp_path / "live", hours=(0, 16), through=DUE)
    runner = harness.runner
    legacy_observation(harness, through_ms=INSTANT + 10 * MINUTE)
    assert runner.catch_up()[-1].kind is None  # the due minute, as the instant passes
    for through in (DUE + 1, DUE + 2, DUE + 3):  # the instant +60, +120, +180 s
        publish_through(harness, through)
        outcomes = runner.catch_up()
        assert [(o.minute_ms, o.kind) for o in outcomes] == [(m(DUE), None)], through
        assert runner.cursor.last_minute_processed == m(DUE - 1)
        assert fundings(harness) == []
        assert deferral_gauges() in (None, (1.0, INSTANT / 1000))

    harness.feed.write_settlements([DAY], hours=(0, 8, 16))
    runner.catch_up()
    assert fundings(harness) == [(iso(DUE), INSTANT)]
    gauges = deferral_gauges()
    assert gauges is None or (gauges[0] == 0.0 and gauges[1] != gauges[1]), gauges
    assert minutes_of(harness, "DECISION") == [iso(i) for i in range(WINDOW.start, DUE + 4)]
    publish_through(harness, LAST)
    runner.catch_up()
    runner.shutdown("live")
    assert fundings(harness) == [(iso(DUE), INSTANT)]

    replay = funding_world(tmp_path / "replay")
    replay.runner.replay(m(WINDOW.start), m(LAST))
    replay.runner.shutdown("replay")
    assert fundings(replay) == fundings(harness)
    report = compare_logs(harness.records(), replay.records())
    assert report.ok and report.label == "PARITY", report.to_dict()


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
# the runner: the held position stays safe while a minute is deferred (F2)
# =========================================================================== #
def held_over_the_instant(where: Path, *, through: int = DUE, config=None, start=True):
    """A carry position held into the 08:00 instant, whose row has not come; the
    runner has reached the due minute and is waiting on it."""
    harness = funding_world(where, hours=(0, 16), through=through, config=config, start=start)
    if start:
        assert harness.runner.catch_up()[-1].kind is None
        assert harness.runner.position.state is HedgeState.HEDGED
    return harness


def legs(runner) -> tuple:
    return runner.position.leg("spot").quantity, runner.position.leg("perp").quantity


def no_rule_may_run(runner, monkeypatch) -> list[str]:
    evaluated: list[str] = []
    for rule in runner.rules:
        monkeypatch.setattr(
            rule, "evaluate", lambda *a, _id=rule.rule_id, **k: evaluated.append(_id)
        )
    return evaluated


def spike_mark(harness, index: int) -> None:
    """The recorded mark high of minute ``index``, far past any liquidation level."""
    path = harness.feed.normalizer.parquet_path("um", DAY)
    frame = pd.read_parquet(path)
    frame.loc[frame["minute_open_ms"] == m(index), "mark_high"] = 1_000_000_000.0
    frame.to_parquet(path, index=False)


def fresh_heartbeat(harness, index: int) -> int:
    """The recorder vouches for the kline stream up to minute ``index``'s close."""
    at = (m(index) + MINUTE) * 1_000_000 + NS
    write_heartbeat(harness.root, harness.runner.contract, at_ns=at)
    return at + NS


def test_the_kill_switch_is_honoured_while_a_minute_is_deferred(tmp_path, monkeypatch):
    """F2, A. The switch is engaged while the funding minute waits: the next
    pass halts on it, before anything else, and decides nothing."""
    harness = held_over_the_instant(tmp_path)
    runner = harness.runner
    evaluated = no_rule_may_run(runner, monkeypatch)
    for _ in range(3):
        assert runner.catch_up()[-1].kind is None and runner.state is RunnerState.READY
    switch = Path(runner.risk._kill_switch_path)
    switch.parent.mkdir(parents=True, exist_ok=True)
    switch.write_text("drill", encoding="utf-8")
    publish_through(harness, DUE + 1)
    outcomes = runner.catch_up()
    assert [(o.minute_ms, o.kind.value) for o in outcomes] == [(m(DUE), "HALT")]
    assert runner.state is RunnerState.HALT and runner.halt_reason == "kill_switch"
    assert runner.cursor.last_minute_processed == m(DUE - 1)
    assert minutes_of(harness, "DECISION")[-1] == iso(DUE - 1)
    assert evaluated == [] and fundings(harness) == []


def test_a_touch_on_a_newer_recorded_mark_is_acted_on_while_deferred(tmp_path, monkeypatch):
    """F2, B. The recorder publishes past the waiting minute, and a newer
    minute's mark crosses the liquidation level. The touch is not left for the
    settlement row: it flattens and halts as a tick's would, and is recorded on
    the last decided minute, naming the minute that touched. The minutes before
    it did not touch, and no rule ran. A restart does not touch again."""
    harness = held_over_the_instant(tmp_path)
    runner = harness.runner
    evaluated = no_rule_may_run(runner, monkeypatch)
    publish_through(harness, DUE + 1)
    assert runner.catch_up()[-1].kind is None and runner.state is RunnerState.READY
    count = len(harness.records())

    publish_through(harness, DUE + 2)
    spike_mark(harness, DUE + 2)
    outcomes = runner.catch_up()
    assert [o.kind.value for o in outcomes] == ["LIQUIDATION_TOUCH"]
    assert kinds_of(harness)[count:] == ["LIQUIDATION_TOUCH", "HALT"]
    touch = harness.records()[count]
    assert touch["minute"] == iso(DUE - 1)
    touched = pd.Timestamp(m(DUE + 2), unit="ms", tz="UTC").isoformat()
    assert f"on the recorded minute {touched}" in touch["veto_or_rejection"]["detail"]
    assert runner.halt_reason.startswith("liquidation_touch")
    assert runner.state is RunnerState.HALT
    assert legs(runner) == (0, 0)
    assert runner.cursor.last_minute_processed == m(DUE - 1)
    assert evaluated == [] and fundings(harness) == []
    after = len(harness.records())
    assert runner.catch_up() == [], "a pass after the touch acted again"
    assert len(harness.records()) == after

    config = runner.config
    runner.shutdown("after the touch")
    again = held_over_the_instant(tmp_path, through=DUE + 2, config=config, start=False)
    spike_mark(again, DUE + 2)
    assert again.runner.start() is RunnerState.HALT
    assert again.runner.catch_up() == []
    assert kinds_of(again).count("LIQUIDATION_TOUCH") == 1


def test_a_row_that_lands_during_the_safety_pass_is_booked_by_its_own_minute(
    tmp_path, monkeypatch
):
    """F2: the safety pass books nothing. The recorder writes whenever it likes,
    so the row can land after the gate looked and while the pass runs; it is
    then booked by the due minute on the next pass -- never by the safety pass,
    whose window would reach it from the position's open instant and stamp it on
    whatever minute the pass was looking at."""
    harness = held_over_the_instant(tmp_path, through=DUE + 2)
    runner = harness.runner
    scan = runner._touch_while_deferred

    def and_the_row_lands(*args, **kwargs):
        harness.feed.write_settlements([DAY], hours=(0, 8, 16))
        return scan(*args, **kwargs)

    monkeypatch.setattr(runner, "_touch_while_deferred", and_the_row_lands)
    assert [(o.minute_ms, o.kind) for o in runner.catch_up()] == [(m(DUE), None)]
    assert fundings(harness) == []
    monkeypatch.setattr(runner, "_touch_while_deferred", scan)
    runner.catch_up()
    assert fundings(harness) == [(iso(DUE), INSTANT)]


def test_a_deferral_that_outlasts_max_data_delay_s_halts_by_name_with_the_position_held(
    tmp_path, monkeypatch
):
    """F2, C. The heartbeat stays fresh and the recorder keeps publishing; the
    row never comes. Within the committed ``max_data_delay_s`` the minute waits,
    READY, visibly. Once the recorder has published past the instant by more
    than that, the runner halts ``funding_unresolved_timeout``: the position is
    held, the cursor is where it was, nothing is booked, and the feed is not
    called stale. A replay of the files the row never reached halts at the same
    minute for the same reason."""
    harness = held_over_the_instant(tmp_path)
    runner = harness.runner
    assert float(runner.risk.limits.max_data_delay_s) == 180.0
    evaluated = no_rule_may_run(runner, monkeypatch)
    held = legs(runner)
    for through in (DUE + 1, DUE + 2, DUE + 3):  # the instant +60 .. +180 s
        publish_through(harness, through)
        runner.check_feed(fresh_heartbeat(harness, through))
        assert [(o.minute_ms, o.kind) for o in runner.catch_up()] == [(m(DUE), None)]
        assert runner.state is RunnerState.READY
        assert deferral_gauges() in (None, (1.0, INSTANT / 1000))
    count = len(harness.records())

    publish_through(harness, DUE + 4)  # +240 s
    runner.check_feed(fresh_heartbeat(harness, DUE + 4))
    outcomes = runner.catch_up()
    assert [(o.minute_ms, o.kind.value) for o in outcomes] == [(m(DUE), "HALT")]
    assert runner.halt_reason.startswith("funding_unresolved_timeout")
    assert kinds_of(harness)[count:] == ["HALT"]
    halt = harness.records()[-1]
    assert halt["minute"] == iso(DUE - 1)
    assert "FEED_STALLED" not in kinds_of(harness)
    assert legs(runner) == held and runner.risk.state.halted
    assert runner.cursor.last_minute_processed == m(DUE - 1)
    assert evaluated == [] and fundings(harness) == []

    replay = funding_world(tmp_path / "replay", hours=(0, 16))
    replay.runner.replay(m(WINDOW.start), m(LAST))
    assert replay.runner.halt_reason == runner.halt_reason
    assert replay.records()[-1]["kind"] == "HALT"
    assert replay.records()[-1]["minute"] == halt["minute"]


@pytest.mark.parametrize("stop", ["shutdown", "crash"])
def test_a_restart_does_not_grant_a_fresh_wait(tmp_path, stop):
    """F2, E. The bound is measured on the recorded files, not from when this
    process first saw the deferral: a runner restarted after the recorder has
    published past the bound halts on its first pass, having booked nothing and
    skipped nothing."""
    harness = held_over_the_instant(tmp_path, through=DUE + 2)
    config = harness.runner.config
    if stop == "shutdown":
        harness.runner.shutdown("restart while deferred")

    again = held_over_the_instant(tmp_path, through=DUE + 4, config=config, start=False)
    assert again.runner.start() is RunnerState.READY
    outcomes = again.runner.catch_up()
    assert [(o.minute_ms, o.kind.value) for o in outcomes] == [(m(DUE), "HALT")]
    assert again.runner.halt_reason.startswith("funding_unresolved_timeout")
    assert again.runner.cursor.last_minute_processed == m(DUE - 1)
    assert fundings(again) == []


# =========================================================================== #
# the runner: a settlement owed across a safety flatten (N2)
# =========================================================================== #
#: The short perpetual RECEIVES a positive rate and PAYS a negative one.
RECEIVE, PAY = "0.0001", "-0.0003"
TOUCH = DUE + 2


def settle_at(harness, rate: str) -> None:
    """The 08:00 row lands, at ``rate``."""
    harness.feed.write_settlements([DAY], rates={(DAY, 8): rate})


def economics(runner) -> dict[str, Any]:
    """Every accumulator a settlement moves, on both ledgers and in Aegis."""
    ledger = runner.position.ledger.state
    perp = runner.position.perp.ledger
    return {
        "funding_paid": ledger.funding_paid,
        "funding_received": ledger.funding_received,
        "free_cash": ledger.free_cash,
        "last_equity": ledger.last_equity,
        "settled": sorted(ledger.settled),
        "fees": ledger.fees,
        "slippage": ledger.slippage,
        "realised": ledger.realised,
        "perp_ledger": (
            perp.funding_paid,
            perp.funding_received,
            sorted(perp.applied_funding),
        ),
        "funding_streak": runner.risk.state.funding_adverse_streak,
    }


def liquidated_while_deferred(where: Path):
    """N2's live run, up to the row: the position is held into the 08:00 instant,
    its row has not come, the due minute waits, and a newer mark touches. The
    safety pass flattens and halts, with the settlement still unknown."""
    harness = held_over_the_instant(where, through=TOUCH)
    spike_mark(harness, TOUCH)
    assert [o.kind.value for o in harness.runner.catch_up()] == ["LIQUIDATION_TOUCH"]
    assert legs(harness.runner) == (0, 0) and fundings(harness) == []
    return harness


def restarted(where: Path, config, rate: str | None):
    """A new process over the same files, the row landed at ``rate`` (or not)."""
    again = held_over_the_instant(where, through=TOUCH, config=config, start=False)
    spike_mark(again, TOUCH)
    if rate is not None:
        settle_at(again, rate)
    assert again.runner.start() is RunnerState.HALT
    return again


def replayed(where: Path, rate: str):
    """The finished files, every row already there: the due minute books the
    settlement, and the newer mark then touches as a tick's own."""
    replay = funding_world(where, through=TOUCH, start=False)
    settle_at(replay, rate)
    spike_mark(replay, TOUCH)
    replay.runner.start()
    replay.runner.replay(m(WINDOW.start), m(TOUCH))
    assert replay.runner.halt_reason.startswith("liquidation_touch")
    assert legs(replay.runner) == (0, 0)
    return replay


@pytest.mark.parametrize("rate", [RECEIVE, PAY])
def test_a_settlement_owed_across_a_safety_flatten_is_booked_once_in_its_own_minute(
    tmp_path, rate
):
    """N2-A/N2-H. The safety pass flattens before the row exists, and the row
    lands afterwards. The next start books it -- once, on the exposure held
    across the instant, in the due minute -- and leaves the runner halted and
    flat, with no rule run. Its economics are then the replay's to the unit, in
    either direction of the cash flow."""
    live = liquidated_while_deferred(tmp_path / "live")
    config = live.runner.config
    live.runner.shutdown("halted by the touch")

    again = restarted(tmp_path / "live", config, rate)
    replay = replayed(tmp_path / "replay", rate)
    assert fundings(again) == fundings(replay) == [(iso(DUE), INSTANT)]
    assert economics(again.runner) == economics(replay.runner)
    booked = [r for r in again.records() if r["kind"] == "FUNDING"][0]["funding"]
    expected = [r for r in replay.records() if r["kind"] == "FUNDING"][0]["funding"]
    for field_name in ("side", "quantity", "rate", "mark_price", "notional", "cash_flow"):
        assert booked[field_name] == expected[field_name], field_name
    assert booked["direction"] == ("received" if rate == RECEIVE else "paid")

    assert again.runner.state is RunnerState.HALT and legs(again.runner) == (0, 0)
    assert again.runner.cursor.last_minute_processed == m(DUE - 1)
    assert "DECISION" not in kinds_of(again)[kinds_of(again).index("LIQUIDATION_TOUCH") :]
    assert owed_on_disk(again) == []


def test_a_touch_on_the_first_pass_over_the_due_minute_still_owes_the_settlement(tmp_path):
    """The deferral and the touch in ONE pass: the runner reaches the due minute
    for the first time -- a backlog, a restart -- with a newer mark already past
    the liquidation level. What is owed has to be written before that pass
    flattens, or there is no leg left to write it from."""
    harness = funding_world(tmp_path / "live", hours=(0, 16), through=TOUCH)
    spike_mark(harness, TOUCH)
    assert harness.runner.catch_up()[-1].kind.value == "LIQUIDATION_TOUCH"
    assert legs(harness.runner) == (0, 0) and len(owed_on_disk(harness)) == 1
    config = harness.runner.config
    harness.runner.shutdown("halted by the touch")

    again = restarted(tmp_path / "live", config, PAY)
    replay = replayed(tmp_path / "replay", PAY)
    assert fundings(again) == fundings(replay) == [(iso(DUE), INSTANT)]
    assert economics(again.runner) == economics(replay.runner)


def owed_on_disk(harness) -> list[dict[str, Any]]:
    """What the persisted carry ledger says is owed, read off the file itself."""
    path = harness.state_dir / "carry_ledger.json"
    return json.loads(path.read_text(encoding="utf-8"))["funding_owed"]


def funding_cash_flows(harness) -> list[str]:
    return [r["funding"]["cash_flow"] for r in harness.records() if r["kind"] == "FUNDING"]


def test_the_owed_instant_is_durable_from_the_deferral_and_resolved_by_its_booking(tmp_path):
    """N2-F, and the witness for when the fact is written: the leg is held into
    the instant, the row is absent, the minute defers -- and the carry ledger ON
    DISK already says which instant was crossed, on which leg, in which window,
    with nothing booked. The row lands: the leg is still held, so the open gate
    releases the entry (N2R-1) and the due minute books the row once, the
    ordinary way. Live and replay then agree to the unit, and section 10's
    comparison is PARITY."""
    harness = funding_world(tmp_path / "live", hours=(0, 16), through=DUE + 1)
    runner = harness.runner
    assert runner.catch_up()[-1].kind is None
    perp = runner.position.leg("perp")
    assert owed_on_disk(harness) == [
        {
            "instant_ns": INSTANT * 1_000_000,
            "open_instant_ns": runner.position.ledger.state.open_instant_ns,
            "perp": runner.position.perp.position(perp.symbol).to_dict(),
        }
    ]
    assert perp.side.value == "SHORT" and perp.quantity > 0
    assert INSTANT * 1_000_000 not in runner.position.ledger.state.settled
    assert runner.position.ledger.state.funding_received == 0 and fundings(harness) == []

    harness.feed.write_settlements([DAY], hours=(0, 8, 16))
    publish_through(harness, LAST)
    runner.catch_up()
    runner.shutdown("live")
    assert fundings(harness) == [(iso(DUE), INSTANT)] and owed_on_disk(harness) == []

    replay = funding_world(tmp_path / "replay")
    replay.runner.replay(m(WINDOW.start), m(LAST))
    replay.runner.shutdown("replay")
    assert economics(harness.runner) == economics(replay.runner)
    report = compare_logs(harness.records(), replay.records())
    assert report.ok and report.label == "PARITY", report.to_dict()


def test_a_position_flat_across_the_instant_owes_nothing(tmp_path):
    """N2-E, the control. The carry rule stays out (its basis floor is out of
    reach), so nothing is held into 08:00. The due minute still waits for the
    row -- the gate is about the instant, not the position -- but no owed entry
    is written for it, and none is booked when the row lands."""
    harness = funding_world(tmp_path, hours=(0, 16), through=DUE + 1, start=False)
    carry = next(rule for rule in harness.runner.rules if rule.rule_id == "R1_carry")
    carry.params = replace(carry.params, min_basis=Decimal("1e12"))
    harness.runner.start()
    assert harness.runner.catch_up()[-1].kind is None
    assert legs(harness.runner) == (0, 0)
    assert owed_on_disk(harness) == []
    harness.feed.write_settlements([DAY], hours=(0, 8, 16))
    publish_through(harness, LAST)
    harness.runner.catch_up()
    assert fundings(harness) == [] and owed_on_disk(harness) == []
    assert minutes_of(harness, "DECISION")[-1] == iso(LAST)


def test_an_owed_settlement_survives_a_restart_before_its_row(tmp_path):
    """N2-B, with crash windows A and C. Flattened and halted with the row still
    missing; a start then finds nothing to book, and books nothing, but keeps
    what is owed. The row lands; the next start books it once, flat and halted,
    and no later start -- clean or after a crash -- books it again (E)."""
    live = liquidated_while_deferred(tmp_path / "live")
    config = live.runner.config
    owed = owed_on_disk(live)
    assert [entry["instant_ns"] for entry in owed] == [INSTANT * 1_000_000]
    live.runner.shutdown("halted by the touch")

    early = restarted(tmp_path / "live", config, None)
    assert fundings(early) == [] and owed_on_disk(early) == owed
    early.runner.shutdown("the row is not there yet")

    booked = restarted(tmp_path / "live", config, RECEIVE)  # and then a crash: no shutdown
    assert fundings(booked) == [(iso(DUE), INSTANT)] and owed_on_disk(booked) == []
    after = economics(booked.runner)

    for _ in range(2):
        again = restarted(tmp_path / "live", config, RECEIVE)
        assert fundings(again) == [(iso(DUE), INSTANT)]
        assert economics(again.runner) == after
        assert legs(again.runner) == (0, 0)
        again.runner.shutdown("nothing left to book")
    assert economics(replayed(tmp_path / "replay", RECEIVE).runner) == after


def test_a_crash_between_the_booking_and_its_record_books_nothing_twice(tmp_path, monkeypatch):
    """Crash window D. The late booking is persisted -- both ledgers, the owed
    entry gone in the same carry-ledger save -- and the process dies before its
    FUNDING record. The next start recovers (section 9.3; the dying start's own
    STARTUP record is the one ahead of the state file, so that is the cause
    named) and books nothing again: the economics are the replay's, once."""
    live = liquidated_while_deferred(tmp_path / "live")
    config = live.runner.config
    live.runner.shutdown("halted by the touch")

    dying = held_over_the_instant(tmp_path / "live", through=TOUCH, config=config, start=False)
    spike_mark(dying, TOUCH)
    settle_at(dying, RECEIVE)
    append = dying.runner._append

    def crash_at_the_record(kind, *args, **kwargs):
        if kind.value == "FUNDING":
            raise RuntimeError("killed before the FUNDING record")
        return append(kind, *args, **kwargs)

    monkeypatch.setattr(dying.runner, "_append", crash_at_the_record)
    with pytest.raises(RuntimeError, match="killed before"):
        dying.runner.start()
    assert owed_on_disk(dying) == []

    again = restarted(tmp_path / "live", config, RECEIVE)
    assert fundings(again) == []
    recovery = [r for r in again.records() if r["kind"] == "RECOVERY"]
    assert [r["recovery"]["cause"] for r in recovery] == ["LOG_AHEAD_OF_STATE"]
    assert economics(again.runner) == economics(replayed(tmp_path / "replay", RECEIVE).runner)


def test_a_touch_recorded_before_the_flatten_persisted_still_owes_the_settlement(
    tmp_path, monkeypatch
):
    """Crash window B. The process dies after the LIQUIDATION_TOUCH record and
    before the flatten: the stores still hold the position and Aegis is halted.
    The row lands; the start books the settlement once, on the leg held across
    the instant -- still held, never reopened, never flattened by this path --
    at the flow the replay books in the due minute."""
    harness = held_over_the_instant(tmp_path / "live", through=TOUCH)
    spike_mark(harness, TOUCH)
    held = legs(harness.runner)

    def killed(*args, **kwargs):
        raise RuntimeError("killed before the flatten")

    monkeypatch.setattr(harness.runner.position, "emergency_reduce", killed)
    with pytest.raises(RuntimeError, match="killed before"):
        harness.runner.catch_up()
    assert kinds_of(harness)[-1] == "LIQUIDATION_TOUCH"
    config = harness.runner.config

    for _ in range(2):
        again = restarted(tmp_path / "live", config, RECEIVE)
        assert legs(again.runner) == held
        assert fundings(again) == [(iso(DUE), INSTANT)]
        again.runner.shutdown("still halted")
    replay = replayed(tmp_path / "replay", RECEIVE)
    assert funding_cash_flows(again) == funding_cash_flows(replay)


def test_a_row_that_lands_during_a_touching_safety_pass_is_owed_not_lost(
    tmp_path, monkeypatch
):
    """N2-D. The row lands while the pass is scanning, and the pass touches. The
    touch is acted on at once and the pass books nothing (its window is not the
    due minute's); the settlement is then booked once, in the due minute, on the
    leg held across the instant -- and matches the replay."""
    harness = held_over_the_instant(tmp_path / "live", through=TOUCH)
    spike_mark(harness, TOUCH)
    runner = harness.runner
    scan = runner._touch_while_deferred

    def and_the_row_lands(*args, **kwargs):
        settle_at(harness, RECEIVE)
        return scan(*args, **kwargs)

    monkeypatch.setattr(runner, "_touch_while_deferred", and_the_row_lands)
    assert [o.kind.value for o in runner.catch_up()] == ["LIQUIDATION_TOUCH"]
    assert fundings(harness) == [] and legs(runner) == (0, 0)
    config = runner.config
    runner.shutdown("halted by the touch")

    again = restarted(tmp_path / "live", config, RECEIVE)
    replay = replayed(tmp_path / "replay", RECEIVE)
    assert fundings(again) == fundings(replay) == [(iso(DUE), INSTANT)]
    assert economics(again.runner) == economics(replay.runner)


def test_a_kill_switch_while_deferred_still_owes_the_settlement(tmp_path):
    """N2-G. The switch halts the waiting runner with the position held, as it
    always did; an operator then flattens. The row lands afterwards, and the
    start books it once, on the leg held across the instant, at the replay's
    flow -- then stays halted and flat, and never books it again."""
    harness = held_over_the_instant(tmp_path / "live")
    runner = harness.runner
    switch = Path(runner.risk._kill_switch_path)
    switch.parent.mkdir(parents=True, exist_ok=True)
    switch.write_text("drill", encoding="utf-8")
    assert runner.catch_up()[-1].kind.value == "HALT"
    assert runner.halt_reason == "kill_switch" and legs(runner) != (0, 0)
    runner.flatten("drill: close the held carry")
    assert legs(runner) == (0, 0) and len(owed_on_disk(harness)) == 1
    config = runner.config
    runner.shutdown("halted by the switch")

    for _ in range(2):
        again = held_over_the_instant(tmp_path / "live", config=config, start=False)
        settle_at(again, PAY)
        assert again.runner.start() is RunnerState.HALT
        assert fundings(again) == [(iso(DUE), INSTANT)] and legs(again.runner) == (0, 0)
        again.runner.shutdown("still halted")

    replay = funding_world(tmp_path / "replay", start=False)
    settle_at(replay, PAY)
    replay.runner.start()
    replay.runner.replay(m(WINDOW.start), m(DUE))
    assert funding_cash_flows(again) == funding_cash_flows(replay)
    assert again.runner.position.ledger.state.funding_paid > 0


# =========================================================================== #
# the runner: what is owed never overrides the ordinary path (N2R-1)
# =========================================================================== #
#: A venue row a few milliseconds after the instant: the next minute's window.
OFF_SCHEDULE_MS = 3


def settle_off_schedule(harness, rate: str, offset_ms: int) -> None:
    """The 08:00 row lands, at ``rate``, stamped ``offset_ms`` after the instant."""
    path = harness.feed.write_settlements([DAY], rates={(DAY, 8): rate})
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        if row["funding_time_ms"] == INSTANT:
            row["funding_time_ms"] = INSTANT + offset_ms
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def basis_collapses_from(harness, index: int) -> None:
    """Spot 25 higher from minute ``index`` on: the basis falls under R1_carry's
    floor, and the rule exits the carry in that minute. Rewritten by every
    publish, so it is applied again after one."""
    path = harness.feed.normalizer.parquet_path("spot", DAY)
    frame = pd.read_parquet(path)
    later = frame["minute_open_ms"] >= m(index)
    for column in (
        "kline_open",
        "kline_high",
        "kline_low",
        "kline_close",
        "book_bid",
        "book_ask",
    ):
        frame.loc[later, column] += 25.0
    frame.to_parquet(path, index=False)


@pytest.mark.parametrize("rate", [RECEIVE, PAY])
@pytest.mark.parametrize(
    ("offset_ms", "exits", "booked"),
    [
        (OFF_SCHEDULE_MS, True, []),
        (0, True, [(iso(DUE), INSTANT)]),
        (OFF_SCHEDULE_MS, False, [(iso(DUE + 1), INSTANT + OFF_SCHEDULE_MS)]),
    ],
    ids=["off-schedule-row-and-exit", "exact-row-and-exit", "off-schedule-row-held"],
)
def test_an_owed_instant_on_a_leg_still_held_is_left_to_the_ordinary_path(
    tmp_path, rate, offset_ms, exits, booked
):
    """N2R-1. The due minute waits with the carry held, and what is owed is
    recorded. The row lands -- no crash, no flatten, no halt -- so the ordinary
    path is intact, and it answers the row exactly as it did before anything was
    owed: in the window the row falls in, on the leg held there. A row 3 ms
    after the instant belongs to 08:00's window; the rule exits in 07:59, so
    08:00 holds nothing and nothing is booked, in live as in the replay. The
    recorded pre-instant leg must not book it. Controls: the row at the instant
    itself (07:59 books it, before the exit) and the late row with no exit
    (08:00 books it on the leg still held)."""
    live = funding_world(tmp_path / "live", hours=(0, 16), through=DUE + 1)
    if exits:
        basis_collapses_from(live, DUE)
    assert live.runner.catch_up()[-1].kind is None
    assert [entry["instant_ns"] for entry in owed_on_disk(live)] == [INSTANT * 1_000_000]
    settle_off_schedule(live, rate, offset_ms)
    publish_through(live, LAST)
    if exits:
        basis_collapses_from(live, DUE)
    live.runner.catch_up()
    live.runner.shutdown("live")

    replay = funding_world(tmp_path / "replay", hours=(0, 16), start=False)
    settle_off_schedule(replay, rate, offset_ms)
    if exits:
        basis_collapses_from(replay, DUE)
    replay.runner.start()
    replay.runner.replay(m(WINDOW.start), m(LAST))
    replay.runner.shutdown("replay")

    assert fundings(live) == fundings(replay) == booked
    assert funding_cash_flows(live) == funding_cash_flows(replay)
    assert economics(live.runner) == economics(replay.runner)
    assert owed_on_disk(live) == []
    assert (legs(live.runner) == (0, 0)) is exits
    assert minutes_of(live, "DECISION")[-1] == iso(LAST)
    report = compare_logs(live.records(), replay.records())
    assert report.ok and report.label == "PARITY", report.to_dict()


@pytest.mark.parametrize("rate", [RECEIVE, PAY])
def test_an_off_schedule_row_owed_across_a_safety_flatten_is_booked_on_the_owed_leg(
    tmp_path, rate
):
    """N2R-1's other side. The safety pass flattened while the due minute waited,
    so the leg the late row is charged on no longer exists: the recorded one
    still answers it -- once, in its own window, 08:00 -- exactly as the replay
    books it there before its own tick meets the touch."""
    live = liquidated_while_deferred(tmp_path / "live")
    config = live.runner.config
    live.runner.shutdown("halted by the touch")

    again = held_over_the_instant(tmp_path / "live", through=TOUCH, config=config, start=False)
    spike_mark(again, TOUCH)
    settle_off_schedule(again, rate, OFF_SCHEDULE_MS)
    assert again.runner.start() is RunnerState.HALT

    replay = funding_world(tmp_path / "replay", through=TOUCH, start=False)
    settle_off_schedule(replay, rate, OFF_SCHEDULE_MS)
    spike_mark(replay, TOUCH)
    replay.runner.start()
    replay.runner.replay(m(WINDOW.start), m(TOUCH))
    assert replay.runner.halt_reason.startswith("liquidation_touch")

    moment = [(iso(DUE + 1), INSTANT + OFF_SCHEDULE_MS)]
    assert fundings(again) == fundings(replay) == moment
    assert economics(again.runner) == economics(replay.runner)
    assert owed_on_disk(again) == [] and legs(again.runner) == (0, 0)


@pytest.mark.parametrize("rate", [RECEIVE, PAY])
def test_an_owed_instant_whose_leg_was_flattened_while_waiting_survives_the_open_gate(
    tmp_path, rate
):
    """N2R-1's boundary on the ordinary path. An operator flattens while the due
    minute waits -- no halt, so the next tick is an ordinary one. When the row
    lands and the gate opens, the leg recorded is no longer held: something
    other than the ordinary path took it, so the entry is kept and answers the
    row, once, in the due minute, on the leg held across the instant -- the
    replay's flow."""
    harness = held_over_the_instant(tmp_path / "live")
    runner = harness.runner
    runner.flatten("drill: close the carry while the funding minute waits")
    assert runner.state is RunnerState.READY and legs(runner) == (0, 0)
    assert len(owed_on_disk(harness)) == 1
    settle_at(harness, rate)
    publish_through(harness, DUE + 1)
    runner.catch_up()
    assert fundings(harness) == [(iso(DUE), INSTANT)] and owed_on_disk(harness) == []

    replay = funding_world(tmp_path / "replay", hours=(0, 16), start=False)
    settle_at(replay, rate)
    replay.runner.start()
    replay.runner.replay(m(WINDOW.start), m(DUE))
    assert fundings(replay) == [(iso(DUE), INSTANT)]
    assert funding_cash_flows(harness) == funding_cash_flows(replay)


def engage_kill_switch(runner) -> None:
    switch = Path(runner.risk._kill_switch_path)
    switch.parent.mkdir(parents=True, exist_ok=True)
    switch.write_text("drill", encoding="utf-8")


def test_a_halt_before_the_hand_over_keeps_the_entry_for_a_resumed_minute(tmp_path):
    """RR-2's resume side. The row lands, then the switch is engaged: the due
    minute's gate opens and the tick halts on the switch before its safety pass
    is through, so nothing is handed over and the entry stays on disk. A halted
    start with the leg still held books nothing from it; the resumed minute
    books the row once, the ordinary way, and resolves the entry, at the
    replay's flow."""
    harness = held_over_the_instant(tmp_path / "live")
    runner = harness.runner
    settle_at(harness, RECEIVE)
    engage_kill_switch(runner)
    assert runner.catch_up()[-1].kind.value == "HALT" and runner.halt_reason == "kill_switch"
    assert len(owed_on_disk(harness)) == 1 and fundings(harness) == []
    config = runner.config
    runner.shutdown("halted by the switch")

    again = held_over_the_instant(tmp_path / "live", config=config, start=False)
    settle_at(again, RECEIVE)
    assert again.runner.start() is RunnerState.HALT
    assert fundings(again) == [] and legs(again.runner) != (0, 0)
    Path(again.runner.risk._kill_switch_path).unlink()
    again.runner.resume("drill over: the switch is off")
    again.runner.catch_up()
    assert fundings(again) == [(iso(DUE), INSTANT)] and owed_on_disk(again) == []

    replay = funding_world(tmp_path / "replay", hours=(0, 16), start=False)
    settle_at(replay, RECEIVE)
    replay.runner.start()
    replay.runner.replay(m(WINDOW.start), m(DUE))
    assert funding_cash_flows(again) == funding_cash_flows(replay)


def test_a_kill_switch_on_the_first_pass_over_the_due_minute_still_owes_the_settlement(
    tmp_path,
):
    """Control B on the first pass. The switch is already engaged when the
    runner first reaches the due minute, so the pass that halts is the one that
    records what is owed, and it records it first. An operator flatten follows;
    the row lands; the start books it once, on the leg held across the instant,
    at the replay's flow."""
    harness = funding_world(tmp_path / "live", hours=(0, 16), through=DUE)
    runner = harness.runner
    runner.catch_up(now_ms=m(DUE - 1))
    assert runner.cursor.last_minute_processed == m(DUE - 1) and legs(runner) != (0, 0)
    engage_kill_switch(runner)
    assert runner.catch_up()[-1].kind.value == "HALT" and runner.halt_reason == "kill_switch"
    assert [entry["instant_ns"] for entry in owed_on_disk(harness)] == [INSTANT * 1_000_000]
    runner.flatten("drill: close the held carry")
    config = runner.config
    runner.shutdown("halted by the switch")

    again = held_over_the_instant(tmp_path / "live", config=config, start=False)
    settle_at(again, PAY)
    assert again.runner.start() is RunnerState.HALT
    assert fundings(again) == [(iso(DUE), INSTANT)] and legs(again.runner) == (0, 0)
    assert owed_on_disk(again) == []

    replay = funding_world(tmp_path / "replay", start=False)
    settle_at(replay, PAY)
    replay.runner.start()
    replay.runner.replay(m(WINDOW.start), m(DUE))
    assert funding_cash_flows(again) == funding_cash_flows(replay)


# =========================================================================== #
# the runner: an owed entry answers only the exposure it held (RR-1, RR-2)
# =========================================================================== #
def touched_while_deferred(where: Path, touch: int):
    """The carry is held into 08:00, the due minute waits for its row, and the
    recorded mark of minute ``touch`` touches: the safety pass flattens and halts.
    Returns the config the restarts need."""
    harness = held_over_the_instant(where, through=touch)
    spike_mark(harness, touch)
    assert [o.kind.value for o in harness.runner.catch_up()] == ["LIQUIDATION_TOUCH"]
    assert legs(harness.runner) == (0, 0) and fundings(harness) == []
    assert len(owed_on_disk(harness)) == 1
    harness.runner.shutdown("halted by the touch")
    return harness.runner.config


def restarted_touched(where: Path, config, touch: int, rate: str, offset_ms: int):
    """A new process over the live files, the 08:00 row landed ``offset_ms`` after
    the instant and the recorder two minutes further on."""
    again = held_over_the_instant(where, through=DUE + 2, config=config, start=False)
    spike_mark(again, touch)
    settle_off_schedule(again, rate, offset_ms)
    assert again.runner.start() is RunnerState.HALT
    return again


def replayed_touched(where: Path, touch: int, rate: str, offset_ms: int):
    """The finished files: the row is there from the start, and minute ``touch``'s
    own tick liquidates, after booking what its window holds."""
    replay = funding_world(where, through=DUE + 2, start=False)
    settle_off_schedule(replay, rate, offset_ms)
    spike_mark(replay, touch)
    replay.runner.start()
    replay.runner.replay(m(WINDOW.start), m(DUE + 2))
    assert replay.runner.halt_reason.startswith("liquidation_touch")
    assert legs(replay.runner) == (0, 0)
    return replay


@pytest.mark.parametrize("rate", [RECEIVE, PAY])
@pytest.mark.parametrize(
    ("touch", "offset_ms", "booked"),
    [
        (DUE, OFF_SCHEDULE_MS, []),
        (DUE, 0, [(iso(DUE), INSTANT)]),
        (DUE + 1, OFF_SCHEDULE_MS, [(iso(DUE + 1), INSTANT + OFF_SCHEDULE_MS)]),
        (DUE + 2, OFF_SCHEDULE_MS, [(iso(DUE + 1), INSTANT + OFF_SCHEDULE_MS)]),
    ],
    ids=[
        "off-schedule-row-touch-in-due-minute",
        "exact-row-touch-in-due-minute",
        "off-schedule-row-touch-at-0800",
        "off-schedule-row-touch-at-0801",
    ],
)
def test_a_touch_ends_the_exposure_an_owed_entry_may_answer(
    tmp_path, rate, touch, offset_ms, booked
):
    """RR-1. A touch flattens at the close of the minute that touched -- in live,
    where the safety pass meets it while the due minute waits, as in the replay,
    whose own tick meets it. The row/window contract charges a settlement on the
    exposure held at the row's own stamp, so the entry answers only a row stamped
    at or before that close. A row 3 ms after the instant, with the touch in the
    due minute itself (07:59, closing at the instant), falls after the flatten:
    nothing is booked, live as in the replay, and the entry is resolved rather
    than left behind. Controls: the row at the instant with the same touch (07:59
    books it before the flatten), and the same late row with the touch at 08:00
    or 08:01 (08:00 books it, on the leg still held)."""
    config = touched_while_deferred(tmp_path / "live", touch)
    again = restarted_touched(tmp_path / "live", config, touch, rate, offset_ms)
    replay = replayed_touched(tmp_path / "replay", touch, rate, offset_ms)

    assert fundings(again) == fundings(replay) == booked
    assert funding_cash_flows(again) == funding_cash_flows(replay)
    assert economics(again.runner) == economics(replay.runner)
    assert owed_on_disk(again) == [] and legs(again.runner) == (0, 0)
    assert again.runner.state is RunnerState.HALT
    after = economics(again.runner)
    again.runner.shutdown("still halted")

    later = restarted_touched(tmp_path / "live", config, touch, rate, offset_ms)
    assert fundings(later) == booked and economics(later.runner) == after


@pytest.mark.parametrize("rate", [RECEIVE, PAY])
@pytest.mark.parametrize(
    ("offset_ms", "booked"),
    [
        (0, [(iso(DUE), INSTANT)]),
        (OFF_SCHEDULE_MS, [(iso(DUE + 1), INSTANT + OFF_SCHEDULE_MS)]),
    ],
    ids=["exact-row", "off-schedule-row"],
)
def test_a_kill_switch_on_the_pass_that_opens_the_gate_leaves_the_settlement_owed(
    tmp_path, rate, offset_ms, booked
):
    """RR-2. The due minute waits with the carry held; the row lands; the switch
    is engaged before the next pass. That pass opens the funding gate and halts
    on the switch before anything is booked. The ordinary path never got the
    row, so the entry must still be on disk: a start while the leg is still held
    books nothing and keeps it, an operator flatten follows, and the next start
    books the settlement once, on the leg held across the instant, at the
    replay's flow. No later start books it again."""
    harness = held_over_the_instant(tmp_path / "live")
    runner = harness.runner
    settle_off_schedule(harness, rate, offset_ms)
    publish_through(harness, DUE + 1)
    engage_kill_switch(runner)
    assert runner.catch_up()[-1].kind.value == "HALT" and runner.halt_reason == "kill_switch"
    assert fundings(harness) == [] and len(owed_on_disk(harness)) == 1
    config = runner.config
    runner.shutdown("halted by the switch")

    def started():
        again = held_over_the_instant(
            tmp_path / "live", through=DUE + 1, config=config, start=False
        )
        settle_off_schedule(again, rate, offset_ms)
        assert again.runner.start() is RunnerState.HALT
        return again

    held = started()
    assert fundings(held) == [] and legs(held.runner) != (0, 0)
    assert len(owed_on_disk(held)) == 1
    held.runner.flatten("drill: close the held carry")
    held.runner.shutdown("flattened")

    for _ in range(2):
        again = started()
        assert fundings(again) == booked and owed_on_disk(again) == []
        assert legs(again.runner) == (0, 0)
        again.runner.shutdown("still halted")

    replay = funding_world(tmp_path / "replay", start=False)
    settle_off_schedule(replay, rate, offset_ms)
    replay.runner.start()
    replay.runner.replay(m(WINDOW.start), m(DUE + 1))
    assert fundings(replay) == booked
    assert funding_cash_flows(again) == funding_cash_flows(replay)
    assert (again.runner.position.ledger.state.funding_paid > 0) is (rate == PAY)


@pytest.mark.parametrize(
    ("offset_ms", "booked"),
    [(OFF_SCHEDULE_MS, []), (0, [(iso(DUE), INSTANT)])],
    ids=["off-schedule-row", "exact-row"],
)
def test_the_touch_bound_is_on_disk_before_the_flatten(
    tmp_path, monkeypatch, offset_ms, booked
):
    """RR-1, crash window B. The process dies after the touch in the due minute
    and before the flatten: the stores still hold the position. The bound was
    saved with the touch, so the start that finds the row books exactly what the
    replay books -- nothing for the row after the touch's close, the row at the
    instant once -- and resolves the entry either way."""
    harness = held_over_the_instant(tmp_path / "live")
    spike_mark(harness, DUE)

    def killed(*args, **kwargs):
        raise RuntimeError("killed before the flatten")

    monkeypatch.setattr(harness.runner.position, "emergency_reduce", killed)
    with pytest.raises(RuntimeError, match="killed before"):
        harness.runner.catch_up()
    assert [entry["held_until_ns"] for entry in owed_on_disk(harness)] == [INSTANT * 1_000_000]
    config = harness.runner.config

    again = restarted_touched(tmp_path / "live", config, DUE, PAY, offset_ms)
    replay = replayed_touched(tmp_path / "replay", DUE, PAY, offset_ms)
    assert fundings(again) == fundings(replay) == booked
    assert funding_cash_flows(again) == funding_cash_flows(replay)
    assert owed_on_disk(again) == [] and legs(again.runner) != (0, 0)


@pytest.mark.parametrize(
    ("offset_ms", "booked"),
    [(OFF_SCHEDULE_MS, []), (0, [(iso(DUE), INSTANT)])],
    ids=["off-schedule-row", "exact-row"],
)
def test_a_touch_on_the_pass_that_opens_the_gate_is_the_replays_touch(
    tmp_path, offset_ms, booked
):
    """RR-1 on the ordinary path. The due minute waited with the carry held; the
    row lands, and the due minute's own mark touches. The pass that opens the
    gate books what the minute's window holds, hands the entry over, and meets
    the touch exactly as the replay's tick does: the row at the instant is
    booked once, the row 3 ms later never, and no entry is left for a halted
    start to answer."""
    live = held_over_the_instant(tmp_path / "live")
    settle_off_schedule(live, PAY, offset_ms)
    publish_through(live, DUE + 2)
    spike_mark(live, DUE)
    assert live.runner.catch_up()[-1].kind.value == "LIQUIDATION_TOUCH"
    assert owed_on_disk(live) == [] and legs(live.runner) == (0, 0)
    config = live.runner.config
    live.runner.shutdown("halted by the touch")

    again = restarted_touched(tmp_path / "live", config, DUE, PAY, offset_ms)
    replay = replayed_touched(tmp_path / "replay", DUE, PAY, offset_ms)
    assert fundings(again) == fundings(replay) == booked
    assert economics(again.runner) == economics(replay.runner)


class Killed(BaseException):
    """The process dies here: nothing in the runner catches it."""


def exit_world(where: Path, config=None, *, start: bool = True):
    """The carry held into 08:00, its row landed 3 ms late, and the basis
    collapsing from the due minute, so the rule exits there."""
    harness = funding_world(where, hours=(0, 16), through=LAST, config=config, start=False)
    settle_off_schedule(harness, PAY, OFF_SCHEDULE_MS)
    basis_collapses_from(harness, DUE)
    if start:
        harness.runner.start()
    return harness


@pytest.mark.parametrize("crash", ["before-the-hand-over", "after-the-hand-over"])
def test_a_crash_around_the_hand_over_neither_loses_nor_revives_the_entry(
    tmp_path, monkeypatch, crash
):
    """RR-2, F. The due minute waited with the carry held and the entry was
    written; the row lands 3 ms late and the rule exits in the due minute. The
    process dies on the pass that opens the gate: before the hand-over (the
    entry is still on disk) or after it was persisted (the entry is gone and
    stays gone). Either way the next process decides the due minute again and
    ends where the uninterrupted run and the replay end: nothing booked, no
    entry left."""
    live = funding_world(tmp_path / "live", hours=(0, 16), through=DUE)
    basis_collapses_from(live, DUE)
    assert live.runner.catch_up()[-1].kind is None and len(owed_on_disk(live)) == 1
    settle_off_schedule(live, PAY, OFF_SCHEDULE_MS)
    publish_through(live, LAST)
    basis_collapses_from(live, DUE)
    runner = live.runner
    if crash == "before-the-hand-over":

        def dies(*args, **kwargs):
            raise Killed

        monkeypatch.setattr(runner.position, "release_funding_owed", dies)
    else:
        rule = next(iter(runner.rules))

        def dies(*args, **kwargs):
            raise Killed

        monkeypatch.setattr(rule, "evaluate", dies)
    with pytest.raises(Killed):
        runner.catch_up()
    assert fundings(live) == []
    assert len(owed_on_disk(live)) == (1 if crash == "before-the-hand-over" else 0)
    config = runner.config

    again = exit_world(tmp_path / "live", config)
    again.runner.catch_up()
    assert owed_on_disk(again) == []
    again.runner.shutdown("live")
    replay = exit_world(tmp_path / "replay")
    replay.runner.replay(m(WINDOW.start), m(LAST))
    replay.runner.shutdown("replay")
    assert fundings(again) == fundings(replay) == []
    assert economics(again.runner) == economics(replay.runner)
    assert minutes_of(again, "DECISION")[-1] == iso(LAST)

    later = exit_world(tmp_path / "live", config)
    later.runner.catch_up()
    assert fundings(later) == [] and owed_on_disk(later) == []


@pytest.mark.parametrize("rate", [RECEIVE, PAY])
def test_a_halted_start_leaves_a_held_entry_to_the_resumed_minute(tmp_path, rate):
    """RR-3. The switch halts the waiting runner with the carry still held; the
    row lands 3 ms late. A halted start books nothing from the entry -- nothing
    has taken that exposure -- and keeps it. The operator resumes, the due
    minute is decided, and the rule exits there: the row falls in 08:00, which
    holds nothing, so nothing is booked, live as in the replay. Booking the
    recorded leg at the halted start would have charged an exposure the
    resumed minute gave up."""
    harness = held_over_the_instant(tmp_path / "live")
    runner = harness.runner
    engage_kill_switch(runner)
    assert runner.catch_up()[-1].kind.value == "HALT" and legs(runner) != (0, 0)
    config = runner.config
    runner.shutdown("halted by the switch")

    again = held_over_the_instant(tmp_path / "live", through=LAST, config=config, start=False)
    settle_off_schedule(again, rate, OFF_SCHEDULE_MS)
    basis_collapses_from(again, DUE)
    assert again.runner.start() is RunnerState.HALT
    assert fundings(again) == [] and len(owed_on_disk(again)) == 1
    Path(again.runner.risk._kill_switch_path).unlink()
    again.runner.resume("drill over: the switch is off")
    again.runner.catch_up()
    again.runner.shutdown("live")

    replay = funding_world(tmp_path / "replay", hours=(0, 16), start=False)
    settle_off_schedule(replay, rate, OFF_SCHEDULE_MS)
    basis_collapses_from(replay, DUE)
    replay.runner.start()
    replay.runner.replay(m(WINDOW.start), m(LAST))
    replay.runner.shutdown("replay")
    assert fundings(again) == fundings(replay) == []
    assert economics(again.runner) == economics(replay.runner)
    assert owed_on_disk(again) == [] and legs(again.runner) == (0, 0)


def stale_spot_world(where: Path, *, through: int = DUE):
    """The carry held into 08:00, with no spot book recorded for the three
    minutes before the due one, so a flatten there has no spot quote to fill
    against."""
    shapes = window_through(through)
    shapes[f"spot:{DAY}"] = {
        **shapes[f"spot:{DAY}"],
        **{i: MinuteShape(book=False) for i in range(DUE - 3, DUE)},
    }
    return shapes


def test_an_entry_whose_perpetual_leg_alone_was_flattened_is_not_handed_over(tmp_path):
    """RR-4. The operator flattens while the due minute waits, and the flatten
    meets a stale spot quote: the perpetual leg closes, the spot leg and the
    funding window stay, and the position is disputed. The window alone says
    the recorded exposure is still held; the perpetual leg -- what a settlement
    is charged on -- says it is not. So when the row lands 3 ms late and the due
    minute's safety pass is through, the entry is NOT handed over: it stays on
    disk as the record of what the settlement is owed on, and the dispute halt
    that follows books nothing from it."""
    harness = build(tmp_path, shapes=stale_spot_world(tmp_path), start=False)
    harness.feed.write_settlements([DAY], hours=(0, 16))
    runner = harness.runner
    runner.start()
    assert runner.catch_up()[-1].kind is None and len(owed_on_disk(harness)) == 1
    opened = runner.position.ledger.state.open_instant_ns
    runner.flatten("drill: close the carry while the funding minute waits")
    spot, perp = legs(runner)
    assert perp == 0 and spot != 0 and runner.state is RunnerState.READY
    assert runner.position.ledger.state.open_instant_ns == opened
    owed = owed_on_disk(harness)

    settle_off_schedule(harness, PAY, OFF_SCHEDULE_MS)
    harness.feed.write_days([DAY], shapes=stale_spot_world(tmp_path, through=DUE + 2))
    assert [o.kind.value for o in runner.catch_up()] == ["HALT"]
    assert runner.halt_reason.startswith("identity_violation")
    assert owed_on_disk(harness) == owed and fundings(harness) == []


def rule_raises_from(harness, index: int) -> None:
    """The carry rule raises on minute ``index`` and after, in whatever process
    evaluates it: a defect the replay of the same files meets too."""
    carry = next(rule for rule in harness.runner.rules if rule.rule_id == "R1_carry")
    evaluate = carry.evaluate

    def raises(state, portfolio):
        if state.minute_ns >= m(index) * 1_000_000:
            raise RuntimeError("drill: the rule fails")
        return evaluate(state, portfolio)

    carry.evaluate = raises


def test_a_halt_after_the_hand_over_leaves_nothing_owed_on_disk(tmp_path):
    """RR-2, the hand-over's persistence. The row lands 3 ms late, the due
    minute's pass hands the entry over, and the rule then fails: a halt the
    replay meets in the same minute, so the replay never reaches 08:00 and books
    nothing. The process dies without a shutdown. The hand-over is already on
    disk, so the operator's flatten and the next start book nothing either --
    an entry left behind would book the late row on the recorded leg."""
    live = funding_world(tmp_path / "live", hours=(0, 16), through=DUE)
    assert live.runner.catch_up()[-1].kind is None and len(owed_on_disk(live)) == 1
    settle_off_schedule(live, PAY, OFF_SCHEDULE_MS)
    publish_through(live, LAST)
    rule_raises_from(live, DUE)
    assert live.runner.catch_up()[-1].kind.value == "HALT"
    assert live.runner.halt_reason.startswith("rule_exception")
    assert owed_on_disk(live) == []
    config = live.runner.config  # and the process dies: no shutdown

    def started():
        again = funding_world(
            tmp_path / "live", hours=(0, 16), through=LAST, config=config, start=False
        )
        settle_off_schedule(again, PAY, OFF_SCHEDULE_MS)
        assert again.runner.start() is RunnerState.HALT
        return again

    held = started()
    held.runner.flatten("drill: close the carry after the rule failed")
    held.runner.shutdown("flattened")
    again = started()
    assert fundings(again) == [] and owed_on_disk(again) == []

    replay = funding_world(tmp_path / "replay", hours=(0, 16), start=False)
    settle_off_schedule(replay, PAY, OFF_SCHEDULE_MS)
    rule_raises_from(replay, DUE)
    replay.runner.start()
    replay.runner.replay(m(WINDOW.start), m(LAST))
    assert replay.runner.halt_reason.startswith("rule_exception")
    assert fundings(replay) == []


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


PREV = (pd.Timestamp(DAY) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def two_days(tmp_path: Path):
    """The day before carries only its last minute: the gate reads it for every
    minute of the day after (an instant announced there), `state_for` never."""
    earlier = {s: MinuteShape(present=False) for s in range(1439)}
    harness = build(tmp_path, days=(PREV, DAY), shapes={f"um:{PREV}": earlier})
    harness.runner.catch_up(now_ms=m(9))
    assert harness.runner.cursor.last_minute_processed == m(9)
    return harness


@pytest.mark.parametrize("then", ["restored", "unchanged"])
def test_the_funding_gate_waits_on_a_day_before_mid_rewrite_and_halts_on_a_broken_one(
    tmp_path, then
):
    """F5. The day before, caught mid-rewrite, is "not yet": the minute waits, and
    proceeds once the file is whole. The same file failing again unchanged is
    broken, and the gate halts on it -- it may not be swallowed, because nothing
    after the gate reads that day, so the minute would be decided with its
    funding gate silently off."""
    harness = two_days(tmp_path)
    runner = harness.runner
    whole = torn(harness.feed.normalizer.parquet_path("um", PREV))
    first = runner.catch_up(now_ms=m(19))
    assert [(o.minute_ms, o.kind) for o in first] == [(m(10), None)]
    if then == "restored":
        harness.feed.normalizer.parquet_path("um", PREV).write_bytes(whole)
        runner.catch_up(now_ms=m(19))
        assert runner.cursor.last_minute_processed == m(19)
        assert "HALT" not in kinds_of(harness)
    else:
        second = runner.catch_up(now_ms=m(19))
        assert runner.state is RunnerState.HALT
        assert runner.halt_reason.startswith("feed_unreadable")
        assert "has not changed since it last failed" in runner.halt_reason
        assert [(o.minute_ms, o.kind.value) for o in second] == [(m(10), "HALT")]
        assert runner.cursor.last_minute_processed == m(9)


@pytest.mark.parametrize("fault", ["disagreeing-settlement-rows", "a-defect-in-the-gate"])
def test_anything_but_not_yet_from_the_funding_gate_halts(tmp_path, fault):
    """F5's other two: a settlements file that contradicts itself, and a gate
    that raises something no reader meant. Neither is a reason to wait, and
    neither may pass the minute ungated."""
    harness = build(tmp_path)
    runner = harness.runner
    runner.catch_up(now_ms=m(9))
    if fault == "disagreeing-settlement-rows":
        harness.feed.write_settlements(
            [DAY], duplicate=[(DAY, 8)], rates={(DAY, 16): "0.0002"}
        )
        path = harness.feed.normalizer.settlements_path("um")
        lines = path.read_text(encoding="utf-8").splitlines()
        row = json.loads(lines[1])
        lines[2] = json.dumps({**row, "funding_rate": "0.0009"}, sort_keys=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:

        def defect(*args, **kwargs):
            raise RuntimeError("a defect in the gate")

        runner.cursor._scheduled_through = defect
    outcomes = runner.catch_up(now_ms=m(19))
    assert [(o.minute_ms, o.kind.value) for o in outcomes] == [(m(10), "HALT")]
    assert runner.state is RunnerState.HALT
    assert runner.halt_reason.startswith("feed_unreadable")
    assert runner.cursor.last_minute_processed == m(9)


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
