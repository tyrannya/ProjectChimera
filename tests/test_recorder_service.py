"""The recorder service: startup recovery, the loops, shutdown, and what it reports.

The websocket client and the REST poller are tested against fakes in their own
files. This one is about the layer that owns *when*: what happens on a start
that follows a crash, what a reconnect leads to, what the heartbeat says, what a
failing task does to the process, and what is left on disk when it stops.

**Everything here runs offline.** The websocket clients are replaced by scripted
stand-ins and the REST poller by a fake session, so the tests exercise the
service's own control flow rather than a network. Two of them do start a real
local websocket server, because "a reconnect is followed by a gap-fill" is a
statement about two subsystems agreeing and a stand-in for one of them would
make it vacuous.

**The invariants under test, stated once.** A gap-filled minute is a recorded
observation and never a repair. A missing minute stays missing. A day that has
been frozen is not written to again. The prospective boundary is never set by
anything that runs. And a task that dies takes the service down with it rather
than leaving a process that reports itself healthy while recording nothing.
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from chimera.metrics import RECORDER_METRIC_NAMES
from chimera.recorder.contract import load_recorder_contract
from chimera.recorder.events import (
    NS_PER_MILLISECOND,
    SPOT_KLINE_1M,
    UM_FUNDING,
    UM_KLINE_1M,
    UM_MARK_PRICE,
    EventSource,
    day_start_ns,
    utc_day,
)
from chimera.recorder.health import (
    HEARTBEAT_INTERVAL_S,
    HEARTBEAT_SCHEMA,
    RecorderHealthError,
    heartbeat_path,
    initial_health,
    read_status,
)
from chimera.recorder.rest import RestPoller
from chimera.recorder.service import (
    MAX_GAPFILL_MINUTES,
    RecorderService,
    RecorderServiceError,
    build_service,
)
from chimera.recorder.sink import RawSink, read_raw_events
from tests.recorder_synthetic import (
    DAY,
    funding_rest_row,
    kline_event,
    kline_rest_row,
    kline_ws_frame,
    minute_ms,
    premium_index_row,
)
from tests.test_recorder_rest_fake import FakeResponse, FakeSession

CONTRACT = load_recorder_contract()

#: A wall clock pinned inside the synthetic day, so "today" is ``DAY`` and every
#: path a test asserts on is the one the fixtures write to.
DAY_NS = day_start_ns(DAY)
NOON_NS = DAY_NS + 12 * 3_600_000 * NS_PER_MILLISECOND


def frozen_clock(ns: int = NOON_NS):
    return lambda: ns


class ScriptedClient:
    """A stand-in for :class:`StreamClient` that delivers a list of events.

    It has the attributes the service reads — ``name``, ``stream_ids``,
    ``counters``, ``skew``, ``connected`` — and a ``run`` that delivers its
    script and then waits for the stop event, which is what a healthy client
    does between frames.
    """

    def __init__(self, name, stream_ids, on_event, events=(), *, fail=None):
        from chimera.recorder.streams import SkewMeter, StreamCounters

        self.name = name
        self.stream_ids = tuple(stream_ids)
        self.on_event = on_event
        self.events = list(events)
        self.fail = fail
        self.counters = StreamCounters()
        self.skew = SkewMeter()
        self.connected = False
        self.started = asyncio.Event()

    async def run(self, stop):
        self.connected = True
        self.started.set()
        if self.fail is not None:
            raise self.fail
        for event in self.events:
            self.on_event(event)
            self.counters.events += 1
        try:
            await stop.wait()
        finally:
            self.connected = False


def service_for(tmp_path, *, events=(), answers=(), clients=None, **options):
    """A service writing under ``tmp_path``, with the network replaced by fakes."""
    options.setdefault("wall_ns", frozen_clock())
    options.setdefault("gapfill", False)
    options.setdefault("heartbeat_interval_s", 0.05)
    options.setdefault("sync_interval_s", 0.05)
    options.setdefault("normalize_interval_s", 0.05)
    options.setdefault("premium_index_interval_s", 3600.0)
    options.setdefault("funding_catchup_interval_s", 3600.0)
    poller = RestPoller(
        session=FakeSession(list(answers)), sleep=lambda s: None, min_interval_s=0.0
    )
    service = RecorderService(CONTRACT, tmp_path, poller=poller, clients=[], **options)
    if clients is None:
        clients = [
            ScriptedClient("um-test", (UM_KLINE_1M, UM_MARK_PRICE), service._record, events)
        ]
    service.clients = tuple(clients)
    return service


async def run_briefly(service, *, seconds: float = 0.3):
    """Run the service and stop it, with a ceiling so a hang fails the test."""
    stop = asyncio.Event()

    async def stopper():
        await asyncio.sleep(seconds)
        stop.set()

    result, _ = await asyncio.gather(service.run(stop), stopper())
    return result


# --- A. startup recovery --------------------------------------------------------
def test_a_cold_start_normalizes_the_open_day_and_reports_what_it_found(tmp_path):
    service = service_for(tmp_path)
    recovery = service.recover()
    assert set(recovery.tails) == set(CONTRACT.streams)
    assert recovery.normalized == (f"spot/{DAY}", f"um/{DAY}")
    assert service.normalizer.parquet_path("um", DAY).exists()
    document = json.loads(service.normalizer.meta_path("um", DAY).read_text(encoding="utf-8"))
    assert document["rows"] == 0
    assert len(document["missing"]) == 1440, "an empty day is 1440 missing minutes"


def test_a_restart_reads_back_the_last_canonical_instant_from_disk(tmp_path):
    """Step 2 of section 4.3, and the input to the gap-fill that follows it."""
    with RawSink(tmp_path, UM_KLINE_1M, contract=CONTRACT) as sink:
        for index in (0, 1, 2):
            sink.append(kline_event(minute_ms(index)))
        sink.sync()

    service = service_for(tmp_path)
    recovery = service.recover()
    assert recovery.last_canonical_ns[UM_KLINE_1M] == minute_ms(2) * NS_PER_MILLISECOND
    assert recovery.last_canonical_ns[SPOT_KLINE_1M] is None
    assert (
        service.health.stream(UM_KLINE_1M).last_event_ns == minute_ms(2) * NS_PER_MILLISECOND
    )


def test_a_torn_tail_is_repaired_before_anything_else_reads_the_file(tmp_path):
    """Step 1: a crash mid-write leaves half a record, and it is preserved."""
    sink = RawSink(tmp_path, UM_KLINE_1M, contract=CONTRACT)
    sink.append(kline_event(minute_ms(0)))
    sink.sync()
    sink.close()
    path = sink.events_path(DAY)
    with path.open("ab") as handle:
        handle.write(b'{"schema": "chimera.recorder-raw-event/1", "stream": "um.k')

    service = service_for(tmp_path)
    recovery = service.recover()
    assert recovery.tails[UM_KLINE_1M] == 1
    assert any("torn" in note for note in recovery.notes)
    assert path.with_name(path.name + ".truncated").exists(), "the torn bytes are kept"
    assert len(read_raw_events(tmp_path, UM_KLINE_1M, DAY)) == 1


def test_a_frozen_day_is_not_reopened_by_a_restart(tmp_path):
    """A frozen raw day is finalised; recovery must read past it, not into it."""
    with RawSink(tmp_path, UM_KLINE_1M, contract=CONTRACT) as sink:
        sink.append(kline_event(minute_ms(0)))
        sink.sync()
        sink.freeze_day(DAY)
    manifest = json.loads(
        (tmp_path / "raw" / UM_KLINE_1M / DAY / "manifest.json").read_text(encoding="utf-8")
    )
    service = service_for(tmp_path)
    recovery = service.recover()
    assert recovery.tails[UM_KLINE_1M] == 0, "a frozen day is not recovered"
    assert recovery.last_canonical_ns[UM_KLINE_1M] == manifest["last_canonical_ns"]


# --- B. gap fill ----------------------------------------------------------------
def test_a_gap_fill_records_rest_rows_through_the_same_sink_and_labels_them(tmp_path):
    """Section 4.2: gap-filled minutes are observations, with their source said."""
    last_closed = (NOON_NS // NS_PER_MILLISECOND) // 60_000 * 60_000 - 60_000
    rows = [kline_rest_row(last_closed - 60_000), kline_rest_row(last_closed)]
    service = service_for(tmp_path, answers=[FakeResponse(payload=rows)])
    service.recover()

    filled = asyncio.run(service.fill_kline_gap("um"))
    assert filled == 2
    stored = read_raw_events(tmp_path, UM_KLINE_1M, DAY)
    assert [event.source for event in stored] == [EventSource.REST_GAPFILL] * 2
    assert [event.minute_open_ms for event in stored] == [last_closed - 60_000, last_closed]
    assert service.health.stream(UM_KLINE_1M).gapfill_rows == 2


def test_a_gap_fill_never_asks_for_the_minute_that_is_still_forming(tmp_path):
    """A REST row for the current minute would be a partial candle wearing a
    finished candle's shape, because ``rest_payload`` marks every row closed."""
    service = service_for(tmp_path, answers=[FakeResponse(payload=[])])
    service.recover()
    asyncio.run(service.fill_kline_gap("um"))
    params = service.poller.session.params[0]
    current_open = (NOON_NS // NS_PER_MILLISECOND) // 60_000 * 60_000
    assert params["endTime"] == current_open - 60_000
    assert params["endTime"] < current_open


def test_a_gap_fill_starts_after_the_last_minute_this_host_already_holds(tmp_path):
    with RawSink(tmp_path, UM_KLINE_1M, contract=CONTRACT) as sink:
        sink.append(kline_event(minute_ms(10)))
        sink.sync()
    service = service_for(tmp_path, answers=[FakeResponse(payload=[])])
    service.recover()
    asyncio.run(service.fill_kline_gap("um"))
    assert service.poller.session.params[0]["startTime"] == minute_ms(11)


def test_a_gap_fill_reaches_back_no_further_than_the_horizon(tmp_path):
    """Beyond a day it is not gap-filling, it is backfilling, and PR-06 owns that."""
    service = service_for(tmp_path, answers=[FakeResponse(payload=[])])
    service.recover()
    asyncio.run(service.fill_kline_gap("um"))
    start = service.poller.session.params[0]["startTime"]
    end = service.poller.session.params[0]["endTime"]
    assert (end - start) // 60_000 <= MAX_GAPFILL_MINUTES


def test_a_gap_fill_that_fails_is_noted_and_does_not_stop_the_recorder(tmp_path):
    service = service_for(tmp_path, answers=[FakeResponse(500, text="down")])
    service.poller.max_attempts = 1
    service.recover()
    assert asyncio.run(service.fill_kline_gap("um")) == 0
    assert any("gap-fill failed" in note for note in service.health.errors)


def test_a_minute_neither_source_produced_stays_missing(tmp_path):
    """The invariant the whole recorder is built around, asserted end to end."""
    last_closed = (NOON_NS // NS_PER_MILLISECOND) // 60_000 * 60_000 - 60_000
    service = service_for(
        tmp_path, answers=[FakeResponse(payload=[kline_rest_row(last_closed)])]
    )
    service.recover()
    asyncio.run(service.fill_kline_gap("um"))
    service._normalize("um", DAY)
    document = json.loads(service.normalizer.meta_path("um", DAY).read_text(encoding="utf-8"))
    assert document["rows"] == 1
    assert len(document["missing"]) == 1439
    assert last_closed - 60_000 in document["missing"], "the minute before was not invented"


# --- C. funding and the premium index -------------------------------------------
def test_funding_settlements_are_recorded_and_the_settlements_file_is_rebuilt(tmp_path):
    rows = [
        funding_rest_row(DAY_NS // NS_PER_MILLISECOND + hour * 3_600_000) for hour in (0, 8)
    ]
    service = service_for(tmp_path, answers=[FakeResponse(payload=rows)])
    service.recover()
    assert asyncio.run(service.poll_funding()) == 2
    settlements = service.normalizer.settlements_path("um")
    lines = [line for line in settlements.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 2
    assert service.normalizer.settlements_digest_path("um").exists()


def test_the_funding_poller_records_what_came_back_and_checks_no_count(tmp_path):
    """A day with four settlements, or one, is recorded as it happened.

    Establishing what the venue *scheduled* is PR-06's, from the archive. Nothing
    here compares the number of rows against the eight-hour cadence.
    """
    base = DAY_NS // NS_PER_MILLISECOND
    rows = [funding_rest_row(base + hour * 3_600_000) for hour in (0, 4, 8, 12)]
    service = service_for(tmp_path, answers=[FakeResponse(payload=rows)])
    service.recover()
    assert asyncio.run(service.poll_funding()) == 4
    assert service.health.errors == (), "four settlements in a day is not an error"


def test_the_premium_index_is_recorded_on_the_mark_stream_and_never_as_a_settlement(tmp_path):
    service = service_for(
        tmp_path,
        answers=[FakeResponse(payload=premium_index_row(NOON_NS // NS_PER_MILLISECOND))],
    )
    service.recover()
    assert asyncio.run(service.poll_premium_index()) is True
    marks = read_raw_events(tmp_path, UM_MARK_PRICE, DAY)
    assert len(marks) == 1
    assert marks[0].source is EventSource.REST_POLL
    assert marks[0].stream == UM_MARK_PRICE
    assert (
        not service.sinks[UM_FUNDING].events_path(DAY).exists()
    ), "the rate in effect is not a settlement and must not be written as one"


def test_a_premium_index_answer_missing_a_value_is_refused_rather_than_recorded(tmp_path):
    row = premium_index_row(NOON_NS // NS_PER_MILLISECOND)
    del row["indexPrice"]
    service = service_for(tmp_path, answers=[FakeResponse(payload=row)])
    service.recover()
    assert asyncio.run(service.poll_premium_index()) is False
    assert any("premiumIndex" in note for note in service.health.errors)


# --- D. the run, and how it stops -----------------------------------------------
def test_a_run_records_what_the_streams_deliver_and_stops_cleanly(tmp_path):
    events = [kline_event(minute_ms(index)) for index in range(3)]
    service = service_for(tmp_path, events=events)
    result = asyncio.run(run_briefly(service))
    assert result.events == 3
    assert result.write_errors == 0
    assert result.heartbeats >= 2, "the heartbeat kept beating while it ran"
    assert result.seconds >= 0.0
    assert service.sinks[UM_KLINE_1M].open_day is None, "the sinks were closed"


def test_the_run_returns_after_the_stop_event_and_leaves_no_task_running(tmp_path):
    async def scenario():
        service = service_for(tmp_path)
        stop = asyncio.Event()
        task = asyncio.create_task(service.run(stop))
        await asyncio.sleep(0.15)
        stop.set()
        await asyncio.wait_for(task, timeout=5.0)
        await asyncio.sleep(0.05)
        return [
            repr(t)
            for t in asyncio.all_tasks()
            if t is not asyncio.current_task() and "chimera" in repr(t.get_coro())
        ]

    assert asyncio.run(scenario()) == []


def test_a_task_that_fails_takes_the_service_down_rather_than_looking_healthy(tmp_path):
    """The alternative is a process that reports itself up and records nothing."""

    def make(service):
        return [
            ScriptedClient(
                "um-test", (UM_KLINE_1M,), service._record, fail=RuntimeError("boom")
            )
        ]

    service = service_for(tmp_path)
    service.clients = tuple(make(service))
    with pytest.raises(RecorderServiceError, match="a task failed"):
        asyncio.run(run_briefly(service, seconds=2.0))
    assert any("boom" in note for note in service.health.errors)
    document = json.loads(heartbeat_path(tmp_path).read_text(encoding="utf-8"))
    assert all(stream["connected"] is False for stream in document["streams"])


def test_shutdown_normalizes_the_open_day_and_writes_a_last_heartbeat(tmp_path):
    events = [kline_event(minute_ms(index)) for index in range(2)]
    service = service_for(tmp_path, events=events)
    asyncio.run(run_briefly(service))
    document = json.loads(service.normalizer.meta_path("um", DAY).read_text(encoding="utf-8"))
    assert document["rows"] == 2
    heartbeat = json.loads(heartbeat_path(tmp_path).read_text(encoding="utf-8"))
    assert heartbeat["schema"] == HEARTBEAT_SCHEMA
    assert all(stream["connected"] is False for stream in heartbeat["streams"])


# --- E. reconnect, then gap fill, over a real socket -----------------------------
def test_a_reconnect_is_followed_by_a_rest_gap_fill_of_the_minutes_it_missed(tmp_path):
    """Two subsystems agreeing, so neither is faked.

    A real local websocket server delivers one closed minute and drops the
    connection. The service's own gap-fill then fetches the minute that fell in
    the gap from the fake REST endpoint, and both land in the same raw file
    under their own source labels.
    """
    from tests.test_recorder_streams_fake_server import FakeVenue

    async def scenario():
        from chimera.recorder.streams import Backoff, StreamClient, subscriptions_for

        last_closed = (NOON_NS // NS_PER_MILLISECOND) // 60_000 * 60_000 - 60_000
        pushed = kline_ws_frame(last_closed - 60_000)
        gap_row = kline_rest_row(last_closed)
        async with FakeVenue([[pushed], []]) as venue:
            service = service_for(tmp_path, answers=[FakeResponse(payload=[gap_row])])
            subscriptions = [
                s for s in subscriptions_for(CONTRACT, "um") if s.stream_id == UM_KLINE_1M
            ]
            client = StreamClient(
                venue.url,
                subscriptions,
                service._record,
                backoff=Backoff(initial=0.01, maximum=0.02, jitter=0.0),
                name="um-ws",
                ping_interval=None,
            )
            service.clients = (client,)
            stop = asyncio.Event()
            task = asyncio.create_task(service.run(stop))
            while client.counters.reconnects < 1:
                await asyncio.sleep(0.01)
            await service.fill_kline_gap("um")
            stop.set()
            await asyncio.wait_for(task, timeout=5.0)
            return last_closed

    last_closed = asyncio.run(scenario())
    stored = read_raw_events(tmp_path, UM_KLINE_1M, DAY)
    by_minute = {event.minute_open_ms: event.source for event in stored}
    assert by_minute[last_closed - 60_000] is EventSource.WEBSOCKET
    assert by_minute[last_closed] is EventSource.REST_GAPFILL


# --- F. health, metrics and the heartbeat ---------------------------------------
def test_the_heartbeat_is_replaced_atomically_and_never_read_half_written(tmp_path):
    """A supervisor reading during a rewrite sees the previous document."""
    service = service_for(tmp_path)
    service.recover()
    first = service.heartbeat.write(service.health, now_ns=NOON_NS)
    body = heartbeat_path(tmp_path).read_bytes()
    assert json.loads(body)["heartbeat_ns"] == first["heartbeat_ns"]

    second = service.heartbeat.write(service.health, now_ns=NOON_NS + 30 * 10**9)
    assert second["heartbeat_ns"] > first["heartbeat_ns"]
    assert list(heartbeat_path(tmp_path).parent.iterdir()) == [
        heartbeat_path(tmp_path)
    ], "an atomic write leaves no temporary file behind"


def test_the_heartbeat_carries_provenance_and_says_what_the_data_is(tmp_path):
    service = service_for(tmp_path, source_revision="abc1234")
    service.recover()
    document = service.heartbeat.write(service.health, now_ns=NOON_NS)
    assert document["contract_id"] == CONTRACT.contract_id
    assert document["contract_hash"] == CONTRACT.contract_hash
    assert document["prospective_from"] is None
    assert (
        document["evidence_class"] == "engineering"
    ), "until a boundary is committed, everything recorded is engineering data"
    assert document["source_revision"] == "abc1234"
    assert {stream["stream"] for stream in document["streams"]} == set(CONTRACT.streams)


def test_the_heartbeat_carries_no_price_and_no_economic_quantity(tmp_path):
    """Scanned over the field names and the values, with the stream ids excluded.

    ``um.markPrice`` is the *name of a stream*, and a heartbeat that lists the
    streams it is recording has to say it. What must not appear is a field
    reporting a price, a return or a flow — so the scan is over every key, and
    over every value that is not one of the contract's own stream ids.
    """
    service = service_for(tmp_path)
    service.recover()
    document = service.heartbeat.write(service.health, now_ns=NOON_NS)
    forbidden = ("price", "return", "pnl", "profit", "basis", "funding", "equity", "alpha")

    def walk(node, path="$"):
        if isinstance(node, dict):
            for key, value in node.items():
                assert not any(token in key.lower() for token in forbidden), (
                    f"the heartbeat has a field {path}.{key}, which reports an economic "
                    "quantity the recorder does not compute"
                )
                walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")
        elif isinstance(node, str) and node not in CONTRACT.streams:
            assert not any(
                token in node.lower() for token in forbidden
            ), f"the heartbeat value at {path} names an economic quantity: {node!r}"

    walk(document)
    assert any(
        stream["stream"] == UM_MARK_PRICE for stream in document["streams"]
    ), "the scan is not vacuous: a stream id containing 'Price' is present and allowed"


def test_a_stream_that_has_seen_nothing_has_no_age_rather_than_an_age_of_zero(tmp_path):
    health = initial_health(CONTRACT)
    assert health.stream(UM_KLINE_1M).age_seconds(NOON_NS) is None
    health.stream(UM_KLINE_1M).last_event_ns = NOON_NS - 5 * 10**9
    assert health.stream(UM_KLINE_1M).age_seconds(NOON_NS) == pytest.approx(5.0)


def test_the_section_4_8_metric_names_are_all_present_and_none_is_economic():
    required = {
        "chimera_recorder_up",
        "chimera_recorder_events_total",
        "chimera_recorder_last_event_age_seconds",
        "chimera_recorder_reconnects_total",
        "chimera_recorder_duplicates_total",
        "chimera_recorder_late_total",
        "chimera_recorder_gapfill_rows_total",
        "chimera_recorder_missing_minutes_total",
        "chimera_recorder_clock_skew_ms",
        "chimera_recorder_disk_free_bytes",
        "chimera_recorder_write_errors_total",
        "chimera_recorder_heartbeat_timestamp",
    }
    assert set(RECORDER_METRIC_NAMES) == required, "section 4.8's list, exactly"
    assert len(RECORDER_METRIC_NAMES) == len(set(RECORDER_METRIC_NAMES))
    for name in RECORDER_METRIC_NAMES:
        assert name.startswith("chimera_recorder_")
        for token in ("price", "return", "pnl", "profit", "basis", "equity", "alpha"):
            assert token not in name, f"{name} reports an economic quantity"


def test_the_heartbeat_cadence_is_the_adopted_thirty_seconds():
    assert HEARTBEAT_INTERVAL_S == 30.0


def test_an_unreadable_heartbeat_is_refused_rather_than_read_as_an_empty_one(tmp_path):
    path = heartbeat_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(RecorderHealthError, match="not readable"):
        read_status(CONTRACT, tmp_path, now_ns=NOON_NS)


# --- G. status is read-only -----------------------------------------------------
def test_status_reports_what_is_on_disk_and_writes_nothing(tmp_path):
    service = service_for(tmp_path, events=[kline_event(minute_ms(0))])
    asyncio.run(run_briefly(service))
    before = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))
    stamps = {path: path.stat().st_mtime_ns for path in tmp_path.rglob("*") if path.is_file()}

    report = read_status(CONTRACT, tmp_path, now_ns=NOON_NS)
    after = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))
    assert before == after, "status created or removed a file"
    assert all(
        path.stat().st_mtime_ns == stamp for path, stamp in stamps.items()
    ), "status modified a file"
    assert report["contract_hash"] == CONTRACT.contract_hash
    assert report["evidence_class"] == "engineering"
    assert report["heartbeat"]["schema"] == HEARTBEAT_SCHEMA
    um = next(market for market in report["markets"] if market["market"] == "um")
    assert um["last_day"] == DAY and um["rows"] == 1
    assert um["missing_minutes"] == 1439


def test_status_on_an_empty_root_is_a_report_and_not_an_error(tmp_path):
    report = read_status(CONTRACT, tmp_path / "nothing", now_ns=NOON_NS)
    assert report["exists"] is False
    assert report["heartbeat"] is None
    assert report["settlements_rows"] == 0
    assert [stream["days"] for stream in report["streams"]] == [[]] * len(CONTRACT.streams)


# --- H. the prospective boundary is not this PR's to move -----------------------
def test_nothing_the_service_does_sets_the_prospective_boundary(tmp_path):
    """Recording is PR-05's. Deciding that a recording is evidence is not.

    The committed contract carries ``prospective_from: null``, a full run leaves
    it null, and the file on disk is unchanged — the boundary is written once, by
    a reviewed commit, and nothing that runs may do it.
    """
    from chimera.recorder.contract import CONTRACTS_DIR, GEN3_CONTRACT_ID

    committed = CONTRACTS_DIR / f"{GEN3_CONTRACT_ID}.json"
    before = committed.read_bytes()
    service = service_for(tmp_path, events=[kline_event(minute_ms(0))])
    asyncio.run(run_briefly(service))
    assert service.contract.prospective_from is None
    assert service.contract.activated is False
    assert committed.read_bytes() == before, "a run rewrote the committed contract"
    assert service.health.evidence_class == "engineering"


def test_the_storage_root_carries_a_copy_of_the_contract_it_was_recorded_under(tmp_path):
    """Section 4.3's layout: a directory of recordings says what it is.

    It is one of the three things ``.gitignore`` re-includes under
    ``data/prospective/``, so the provenance of a recording can be committed
    while the recording itself cannot.
    """
    service = service_for(tmp_path)
    path = service.bind_contract()
    assert path == tmp_path / "contract" / f"{CONTRACT.contract_id}.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["contract_hash"] == CONTRACT.contract_hash
    assert document["contract"]["prospective_from"] is None
    assert document["storage_layout_version"] == CONTRACT.storage_layout_version


def test_binding_the_same_contract_again_is_idempotent(tmp_path):
    service = service_for(tmp_path)
    first = service.bind_contract().read_bytes()
    service.recover()
    assert service.bind_contract().read_bytes() == first


def test_a_root_recorded_under_another_contract_is_refused(tmp_path):
    """The contract's own version policy: one campaign never mixes contracts."""
    service = service_for(tmp_path)
    path = service.bind_contract()
    document = json.loads(path.read_text(encoding="utf-8"))
    document["contract_hash"] = "0" * 64
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(RecorderServiceError, match="never mixes"):
        service.bind_contract()
    with pytest.raises(RecorderServiceError, match="never mixes"):
        service.recover()


def test_an_unreadable_contract_copy_stops_the_recorder_rather_than_being_replaced(tmp_path):
    service = service_for(tmp_path)
    path = service.bind_contract()
    path.write_text("{ truncated", encoding="utf-8")
    with pytest.raises(RecorderServiceError, match="not readable"):
        service.bind_contract()


def test_the_storage_root_is_the_contracts_own_and_lands_under_data(tmp_path):
    service = build_service(CONTRACT, tmp_path, gapfill=False)
    assert service.root == tmp_path / "prospective" / "gen3"
    assert (
        service.root.relative_to(tmp_path).parts[0] == "prospective"
    ), ".gitignore excludes data/prospective/**; a root outside it could be committed"


def test_the_service_declines_a_stream_the_contract_does_not_name(tmp_path):
    service = service_for(tmp_path)
    service.recover()
    event = kline_event(minute_ms(0))
    object.__setattr__(event, "stream", "um.trades")
    service._record(event)
    assert service.health.stream("um.trades").write_errors == 1
    assert any("no sink" in note for note in service.health.errors)


def test_the_open_day_follows_the_clock_and_not_the_local_timezone(tmp_path):
    service = service_for(tmp_path, wall_ns=frozen_clock(DAY_NS + 60 * 10**9))
    service.recover()
    assert service.health.open_day == utc_day(DAY_NS)
    assert service.health.open_day == DAY


def test_a_write_error_is_counted_and_surfaced_rather_than_swallowed(tmp_path):
    service = service_for(tmp_path)
    service.recover()

    class Refusing:
        stream = UM_KLINE_1M

        def append(self, event):
            raise OSError("disk full")

    service.sinks[UM_KLINE_1M] = Refusing()
    service._record(kline_event(minute_ms(0)))
    assert service.health.stream(UM_KLINE_1M).write_errors == 1
    assert service.health.write_errors == 1
    assert any("write failed" in note for note in service.health.errors)
    document = service.heartbeat.write(service.health, now_ns=NOON_NS)
    assert document["write_errors"] == 1, "the acceptance criterion reads this field"


def test_the_service_time_is_the_injected_clock_so_a_test_can_pin_a_day(tmp_path):
    """A guard on the fixtures above: if the service read the real clock, every
    assertion about ``DAY`` in this file would be about today instead."""
    service = service_for(tmp_path)
    assert service._wall_ns() == NOON_NS
    assert abs(time.time_ns() - NOON_NS) > 10**9, "the pinned day is not now"
