"""R1-h: atomic publication, the silence watchdog, and the recorder's lifecycle log.

The roadmap item, clause by clause: temp-file + fsync + rename for every parquet
and manifest write; an in-process silence watchdog that reconnects on stream
silence; persisted ``recorder.up`` / ``recorder.down`` records so a dead recorder
leaves evidence; gen3's contract hash unchanged by any of it.

Every property below is tested from both sides: the case that must hold and the
case that must fail, so a test cannot pass by the mechanism being absent.
Offline throughout: the websocket is an in-memory stand-in and the REST poller
a fake session.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import threading
import time
from pathlib import Path

import pandas as pd
import pytest

import chimera.recorder.health as health_module
import chimera.recorder.sink as sink_module
from chimera.recorder.contract import (
    GEN3_CONTRACT_ID,
    canonical_material,
    load_recorder_contract,
)
from chimera.recorder.events import (
    SPOT_BOOK_TICKER,
    SPOT_KLINE_1M,
    UM_BOOK_TICKER,
    UM_FUNDING,
    UM_KLINE_1M,
    UM_MARK_PRICE,
)
from chimera.recorder.health import (
    LIFECYCLE_DOWN,
    LIFECYCLE_DOWN_MISSING,
    LIFECYCLE_UP,
    LifecycleLog,
    RecorderHealthError,
    lifecycle_path,
    read_heartbeat,
)
from chimera.recorder.normalize import MinuteNormalizer
from chimera.recorder.service import RecorderServiceError
from chimera.recorder.sink import RawSink, RecorderSinkError, write_bytes_atomic
from chimera.recorder.streams import (
    PERIODIC_STREAMS,
    SILENCE_TIMEOUT_S,
    WEBSOCKET_STREAMS,
    Backoff,
    RecorderStreamError,
    StreamClient,
    clients_for,
    subscriptions_for,
)
from tests.recorder_synthetic import (
    DAY,
    book_ws_frame,
    funding_event,
    kline_event,
    kline_ws_frame,
    mark_ws_frame,
    minute_ms,
)
from tests.test_recorder_service import ScriptedClient, run_briefly, service_for

CONTRACT = load_recorder_contract()


# =========================================================================== #
# A. atomic publication
# =========================================================================== #
def temporaries(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.tmp"))


def test_a_publication_replaces_the_whole_file_and_leaves_no_temporary(tmp_path):
    target = tmp_path / "day.bin"
    write_bytes_atomic(target, b"old")
    write_bytes_atomic(target, b"the whole new body")
    assert target.read_bytes() == b"the whole new body"
    assert temporaries(tmp_path) == []


@pytest.mark.parametrize(
    "fault", [OSError("disk full"), RuntimeError("a bug"), KeyboardInterrupt()]
)
def test_a_failed_write_keeps_the_old_file_and_removes_its_temporary(
    tmp_path, monkeypatch, fault
):
    """OSError becomes RecorderSinkError; anything else propagates as itself. Both
    leave the destination byte-identical and no temporary behind."""
    target = tmp_path / "day.bin"
    write_bytes_atomic(target, b"previous publication")

    def fail(fd):
        raise fault

    monkeypatch.setattr(sink_module.os, "fsync", fail)
    expected = RecorderSinkError if isinstance(fault, OSError) else type(fault)
    with pytest.raises(expected):
        write_bytes_atomic(target, b"half of the next one")
    assert target.read_bytes() == b"previous publication"
    assert temporaries(tmp_path) == []


def test_the_body_is_fsynced_in_a_temporary_beside_the_target_before_the_rename(
    tmp_path, monkeypatch
):
    target = tmp_path / "day.bin"
    calls: list[tuple] = []
    real_fsync, real_replace = sink_module.os.fsync, sink_module.os.replace

    def fsync(fd):
        calls.append(("fsync",))
        real_fsync(fd)

    def replace(source, destination):
        # At the rename the destination has not been touched: nothing wrote it in place.
        calls.append(("replace", Path(source), Path(destination), Path(source).read_bytes()))
        real_replace(source, destination)

    monkeypatch.setattr(sink_module.os, "fsync", fsync)
    monkeypatch.setattr(sink_module.os, "replace", replace)
    write_bytes_atomic(target, b"body")
    assert [call[0] for call in calls] == ["fsync", "replace"], "fsync before the rename"
    _, source, destination, body = calls[1]
    assert destination == target and body == b"body"
    assert source.parent == target.parent, "same directory, so the rename is atomic"
    assert source.name.startswith("day.bin.") and source.name.endswith(".tmp")


def test_two_writers_of_one_destination_never_share_a_temporary(tmp_path, monkeypatch):
    """A second publication lands between the first's fsync and its rename -- a
    second process on the same root. Each publishes its own complete body; with
    one shared ``<name>.tmp`` the second would consume the first's temporary."""
    target = tmp_path / "day.bin"
    real_replace = sink_module.os.replace
    nested = {"done": False}

    def replace(source, destination):
        if not nested["done"]:
            nested["done"] = True
            write_bytes_atomic(target, b"second writer")
            assert target.read_bytes() == b"second writer"
        real_replace(source, destination)

    monkeypatch.setattr(sink_module.os, "replace", replace)
    write_bytes_atomic(target, b"first writer")
    assert target.read_bytes() == b"first writer", "the last rename wins, whole"
    assert temporaries(tmp_path) == []


def refusing(times: int):
    state = {"calls": 0}
    real = sink_module.os.replace

    def replace(source, destination):
        state["calls"] += 1
        if state["calls"] <= times:
            raise PermissionError(13, "the file is open in another process")
        real(source, destination)

    return replace, state


def test_a_reader_holding_the_file_on_windows_is_waited_out(tmp_path, monkeypatch):
    target = tmp_path / "day.bin"
    write_bytes_atomic(target, b"old")
    replace, state = refusing(2)
    monkeypatch.setattr(sink_module, "_RETRY_SHARING_VIOLATIONS", True)
    monkeypatch.setattr(sink_module.time, "sleep", lambda s: None)
    monkeypatch.setattr(sink_module.os, "replace", replace)
    write_bytes_atomic(target, b"new")
    assert target.read_bytes() == b"new" and state["calls"] == 3


def test_a_permission_error_elsewhere_is_raised_at_once(tmp_path, monkeypatch):
    target = tmp_path / "day.bin"
    write_bytes_atomic(target, b"old")
    replace, state = refusing(1)
    monkeypatch.setattr(sink_module, "_RETRY_SHARING_VIOLATIONS", False)
    monkeypatch.setattr(sink_module.os, "replace", replace)
    with pytest.raises(RecorderSinkError):
        write_bytes_atomic(target, b"new")
    assert state["calls"] == 1 and target.read_bytes() == b"old"
    assert temporaries(tmp_path) == []


def test_a_reader_that_never_lets_go_fails_the_write_bounded_and_clean(tmp_path, monkeypatch):
    target = tmp_path / "day.bin"
    write_bytes_atomic(target, b"old")
    replace, state = refusing(10**6)
    monkeypatch.setattr(sink_module, "_RETRY_SHARING_VIOLATIONS", True)
    monkeypatch.setattr(sink_module.time, "sleep", lambda s: None)
    monkeypatch.setattr(sink_module.os, "replace", replace)
    with pytest.raises(RecorderSinkError):
        write_bytes_atomic(target, b"new")
    assert state["calls"] == len(sink_module._SHARING_RETRY_DELAYS_S) + 1
    assert target.read_bytes() == b"old" and temporaries(tmp_path) == []


def test_a_publication_over_a_file_a_reader_has_open_succeeds(tmp_path):
    """The real platform, not a stub: on Windows the rename is refused while the
    handle is open and succeeds once it closes; on POSIX it succeeds at once."""
    target = tmp_path / "day.bin"
    write_bytes_atomic(target, b"old")
    reader = open(target, "rb")
    errors: list[BaseException] = []

    def publish():
        try:
            write_bytes_atomic(target, b"new")
        except BaseException as exc:  # pragma: no cover - the failure being tested for
            errors.append(exc)

    writer = threading.Thread(target=publish)
    writer.start()
    time.sleep(0.1)
    assert reader.read() == b"old", "the reader's open file is the old publication, whole"
    reader.close()
    writer.join(5)
    assert errors == [] and target.read_bytes() == b"new"


# --- the normalized day -------------------------------------------------------
def raw_klines(root: Path, minutes) -> None:
    with RawSink(root, UM_KLINE_1M, contract=CONTRACT) as sink:
        for index in minutes:
            sink.append(kline_event(minute_ms(index)))
        sink.sync()


def test_the_normalized_day_is_published_not_rewritten_in_place(tmp_path, monkeypatch):
    raw_klines(tmp_path, range(3))
    normalizer = MinuteNormalizer(tmp_path, CONTRACT)
    first = normalizer.build_day("um", DAY)
    parquet = normalizer.parquet_path("um", DAY)
    assert first.parquet_sha256 == hashlib.sha256(parquet.read_bytes()).hexdigest()
    previous = parquet.read_bytes()

    raw_klines(tmp_path, range(3, 6))
    seen: list[tuple[Path, int]] = []
    real_replace = sink_module.os.replace

    def replace(source, destination):
        if Path(destination) == parquet:
            # The instant before publication: still the whole previous day.
            assert parquet.read_bytes() == previous
            seen.append((Path(destination), len(pd.read_parquet(source))))
        real_replace(source, destination)

    monkeypatch.setattr(sink_module.os, "replace", replace)
    second = normalizer.build_day("um", DAY)
    assert seen == [(parquet, second.rows)], "the parquet went through the atomic rename"
    assert second.parquet_sha256 == hashlib.sha256(parquet.read_bytes()).hexdigest()


def test_a_failed_republication_leaves_the_previous_day_readable(tmp_path, monkeypatch):
    raw_klines(tmp_path, range(3))
    normalizer = MinuteNormalizer(tmp_path, CONTRACT)
    normalizer.build_day("um", DAY)
    parquet, meta = normalizer.parquet_path("um", DAY), normalizer.meta_path("um", DAY)
    before = parquet.read_bytes(), meta.read_bytes()

    raw_klines(tmp_path, range(3, 6))

    def refuse(source, destination):
        raise OSError("the rename never happened")

    monkeypatch.setattr(sink_module.os, "replace", refuse)
    with pytest.raises(RecorderSinkError):
        normalizer.build_day("um", DAY)
    assert (parquet.read_bytes(), meta.read_bytes()) == before
    assert len(pd.read_parquet(parquet)) == 3
    assert temporaries(parquet.parent) == []


def test_a_reader_never_sees_a_partial_day_while_both_render_paths_republish(tmp_path):
    """The publisher and the maintenance pass render the open day from two
    threads (R1-g serialises them with one lock); the runner reads it from a
    third. Every read is a whole day. Paced like a real reader, not a busy loop,
    because on Windows a reader holding the file continuously is exactly the
    case the bounded retry is allowed to fail.

    On Windows an OPEN that lands while a rename is replacing the file can be
    refused (``PermissionError``, errno 13). That is not a partial day -- nothing
    was read -- and the runner's own reader already treats a failed read as
    "not yet" (`FeedCursor._day` raises `FeedNotReady` and the minute waits).
    So there, and only there, a refused open is retried, exactly as
    `sink._replace` retries the writer's side of the same contention. Anything
    else -- a torn parquet, a short count, a refusal off Windows -- still fails."""
    service = service_for(tmp_path)
    service.recover()
    for index in range(30):
        service._record(kline_event(minute_ms(index)))
    service._sync()
    parquet = service.normalizer.parquet_path("um", DAY)
    service._render([DAY])
    expected = len(pd.read_parquet(parquet))

    failures: list[BaseException] = []
    counts: set[int] = set()
    done = threading.Event()

    def render():
        try:
            for _ in range(15):
                service._render([DAY])
        except BaseException as exc:  # pragma: no cover - the failure being tested for
            failures.append(exc)

    def read():
        while not done.is_set():
            try:
                counts.add(len(pd.read_parquet(parquet)))
            except PermissionError as exc:
                if not sink_module._RETRY_SHARING_VIOLATIONS:
                    failures.append(exc)
            except BaseException as exc:  # pragma: no cover - a partial day
                failures.append(exc)
            time.sleep(0.02)

    reader = threading.Thread(target=read)
    reader.start()
    writers = [threading.Thread(target=render) for _ in range(2)]
    for writer in writers:
        writer.start()
    for writer in writers:
        writer.join(120)
    done.set()
    reader.join(10)
    assert failures == []
    assert counts == {expected}


APPEND_ONLY = {"events.ndjson", "events.late.ndjson", "lifecycle.ndjson"}
#: Written once by a freeze, never replaced, and refused by every reader while
#: the plain file beside it still exists: see `RawSink.freeze_day`.
FROZEN_ONCE = {"events.ndjson.gz"}


def test_every_file_the_recorder_publishes_arrives_by_rename(tmp_path, monkeypatch):
    """The publication inventory, enforced: after a run, a settlements rebuild
    and both freezes, every file under the root either arrived by an atomic
    rename or is one of the named append-only / freeze-once files."""
    replaced: set[Path] = set()
    real_replace = sink_module.os.replace

    def replace(source, destination):
        replaced.add(Path(destination).resolve())
        real_replace(source, destination)

    monkeypatch.setattr(sink_module.os, "replace", replace)
    events = [kline_event(minute_ms(i)) for i in range(3)] + [funding_event(minute_ms(0))]
    service = service_for(
        tmp_path,
        clients=[],
    )
    service.clients = (
        ScriptedClient("um-test", (UM_KLINE_1M, UM_FUNDING), service._record, events),
    )
    asyncio.run(run_briefly(service, seconds=0.3))
    service.normalizer.build_settlements("um")
    for stream in (UM_KLINE_1M, UM_FUNDING):
        RawSink(tmp_path, stream, contract=CONTRACT).freeze_day(DAY)
    service.normalizer.freeze_day("um", DAY)

    published = {path.resolve() for path in tmp_path.rglob("*") if path.is_file()}
    names = {path.name for path in published}
    for required in (
        "manifest.json",
        f"{DAY}.parquet",
        f"{DAY}.meta.json",
        f"{DAY}.sha256",
        "settlements.ndjson",
        "heartbeat.json",
    ):
        assert required in names, f"the inventory run never wrote {required}"
    strays = {
        path.relative_to(tmp_path.resolve()).as_posix()
        for path in published - replaced
        if path.name not in APPEND_ONLY | FROZEN_ONCE
    }
    assert strays == set(), f"published without an atomic rename: {sorted(strays)}"
    assert [path for path in published if path.name.endswith(".tmp")] == []


# =========================================================================== #
# B. the silence watchdog
# =========================================================================== #
class FakeSocket:
    def __init__(self) -> None:
        self.frames: asyncio.Queue = asyncio.Queue()
        self.sent: list[str] = []

    async def send(self, body) -> None:
        self.sent.append(body)

    async def recv(self):
        return await self.frames.get()


class FakeConnect:
    """``connect(url, **options)`` for a StreamClient: one fresh socket per
    session, fed by ``feed(socket)`` for as long as the session lasts."""

    def __init__(self, feed=None) -> None:
        self.feed = feed
        self.sessions: list[FakeSocket] = []

    def __call__(self, url, **options):
        return self._session()

    @contextlib.asynccontextmanager
    async def _session(self):
        socket = FakeSocket()
        self.sessions.append(socket)
        feeder = asyncio.ensure_future(self.feed(socket)) if self.feed else None
        try:
            yield socket
        finally:
            if feeder is not None:
                feeder.cancel()
                await asyncio.gather(feeder, return_exceptions=True)


def every(interval: float, *makers):
    """A feed pushing one frame from each maker every ``interval`` seconds."""

    async def feed(socket: FakeSocket) -> None:
        tick = 0
        while True:
            for make in makers:
                frame = make(tick)
                socket.frames.put_nowait(
                    frame if isinstance(frame, str) else json.dumps(frame)
                )
            tick += 1
            await asyncio.sleep(interval)

    return feed


def mark(tick):
    return mark_ws_frame(minute_ms(0) + tick)


def um_kline(tick):
    return kline_ws_frame(minute_ms(0), closed=False, event_ms=minute_ms(0) + tick)


def spot_kline(tick):
    return kline_ws_frame(minute_ms(0), closed=False, event_ms=minute_ms(0) + tick)


def book(tick):
    return book_ws_frame(tick + 1, event_ms=minute_ms(0) + tick)


def spot_book(tick):
    return book_ws_frame(tick + 1, event_ms=None)


def subs(market, *streams):
    return tuple(s for s in subscriptions_for(CONTRACT, market) if s.stream_id in streams)


def drive(client: StreamClient, seconds: float) -> None:
    async def scenario():
        stop = asyncio.Event()
        task = asyncio.create_task(client.run(stop))
        await asyncio.sleep(seconds)
        stop.set()
        await asyncio.wait_for(task, 5)
        await asyncio.sleep(0)
        assert [t for t in asyncio.all_tasks() if t is not asyncio.current_task()] == []

    asyncio.run(scenario())


def client(
    subscriptions, feed=None, *, silence=0.15, **options
) -> tuple[StreamClient, FakeConnect]:
    connect = FakeConnect(feed)
    return (
        StreamClient(
            "ws://fake",
            subscriptions,
            lambda event: None,
            connect=connect,
            backoff=Backoff(initial=0.01, maximum=0.02, factor=2.0, jitter=0.0),
            silence_timeout_s=silence,
            name="test",
            **options,
        ),
        connect,
    )


UM_MARKET = (UM_KLINE_1M, UM_MARK_PRICE)


def test_a_connected_socket_whose_periodic_stream_stops_is_reconnected_and_told_apart():
    silent, connect = client(subs("um", *UM_MARKET))
    drive(silent, 0.6)
    assert silent.counters.silent_reconnects >= 2
    assert (
        silent.counters.reconnects >= silent.counters.silent_reconnects
    ), "a silent reconnect is also a reconnect"
    assert len(connect.sessions) >= 3
    # the control: the same socket with its mark flowing is never ended for silence
    healthy, connect = client(subs("um", *UM_MARKET), every(0.02, mark, um_kline))
    drive(healthy, 0.6)
    assert healthy.counters.silent_reconnects == 0 and len(connect.sessions) == 1
    assert healthy.counters.events > 0


NOT_A_MARK = [
    lambda t: json.dumps({"result": None, "id": 1}),  # a subscribe ack
    lambda t: "not json",
    lambda t: "[1, 2]",
    lambda t: {"e": "aggTrade", "s": "BTCUSDT"},  # a kind this client never subscribed to
    lambda t: {k: v for k, v in mark_ws_frame(minute_ms(0)).items() if k != "p"},  # malformed
    lambda t: {**mark_ws_frame(minute_ms(0)), "s": "ETHUSDT"},  # another instrument
    um_kline,  # a real, delivered event -- of a stream that is not watched
]


def test_only_a_delivered_event_of_a_watched_stream_resets_its_clock():
    """Every frame here arrives, and none is a delivered mark: the clock never
    moves, whatever the traffic. The klines are delivered, and prove the socket
    is up -- which is what `ping` already knew -- not that the mark is."""
    flooded, _ = client(subs("um", *UM_MARKET), every(0.01, *NOT_A_MARK))
    drive(flooded, 0.5)
    assert flooded.counters.frames > 100 and flooded.counters.events > 0
    assert flooded.counters.silent_reconnects >= 1


def test_an_event_driven_stream_going_quiet_is_not_a_failure():
    """um.kline_1m pushes "(if existing)": silent klines beside a flowing mark
    are a quiet market as far as the protocol says, and are left alone."""
    quiet_klines, connect = client(subs("um", *UM_MARKET), every(0.02, mark))
    drive(quiet_klines, 0.5)
    assert quiet_klines.counters.silent_reconnects == 0 and len(connect.sessions) == 1


def test_the_book_only_socket_has_no_watchdog():
    public, connect = client(subs("um", UM_BOOK_TICKER))
    assert public.watched == ()
    drive(public, 0.5)
    assert public.counters.silent_reconnects == 0 and len(connect.sessions) == 1


def test_the_spot_socket_is_watched_on_its_kline_and_not_its_book():
    spot = subs("spot", SPOT_KLINE_1M, SPOT_BOOK_TICKER)
    books_only, _ = client(spot, every(0.01, spot_book))
    drive(books_only, 0.5)
    assert books_only.counters.events > 0 and books_only.counters.silent_reconnects >= 1
    klines_only, connect = client(spot, every(0.02, spot_kline))
    drive(klines_only, 0.5)
    assert klines_only.counters.silent_reconnects == 0 and len(connect.sessions) == 1


def test_stop_ends_a_silent_wait_at_once_and_counts_nothing():
    waiting, _ = client(subs("um", *UM_MARKET), silence=30.0)
    began = time.monotonic()
    drive(waiting, 0.1)
    assert time.monotonic() - began < 2.0
    assert waiting.counters.silent_reconnects == 0


def test_the_session_deadline_is_an_ordinary_reconnect_not_a_silent_one():
    rotating, connect = client(
        subs("um", *UM_MARKET), every(0.02, mark), silence=30.0, session_seconds=0.1
    )
    drive(rotating, 0.5)
    assert rotating.counters.reconnects >= 2 and len(connect.sessions) >= 3
    assert rotating.counters.silent_reconnects == 0


@pytest.mark.parametrize("value", [0, -1.0, float("nan")])
def test_a_silence_budget_that_is_not_positive_is_refused(value):
    with pytest.raises(RecorderStreamError, match="silence_timeout_s"):
        client(subs("um", *UM_MARKET), silence=value)


def test_the_watched_streams_are_exactly_the_ones_the_venue_pushes_on_a_period():
    assert PERIODIC_STREAMS == {UM_MARK_PRICE, SPOT_KLINE_1M}
    assert PERIODIC_STREAMS <= set(WEBSOCKET_STREAMS)
    assert UM_FUNDING not in PERIODIC_STREAMS, "funding has no socket"
    assert not PERIODIC_STREAMS & {UM_BOOK_TICKER, SPOT_BOOK_TICKER, UM_KLINE_1M}
    watched = {c.name: set(c.watched) for c in clients_for(CONTRACT, lambda event: None)}
    assert watched == {
        "um-market-ws": {UM_MARK_PRICE},
        "um-public-ws": set(),
        "spot-ws": {SPOT_KLINE_1M},
    }
    assert SILENCE_TIMEOUT_S == 90.0


def test_the_heartbeat_tells_a_silent_reconnect_from_an_ordinary_one(tmp_path):
    service = service_for(tmp_path)
    scripted = service.clients[0]
    scripted.counters.reconnects, scripted.counters.silent_reconnects = 5, 2
    service._beat()
    streams = {s["stream"]: s for s in read_heartbeat(tmp_path)["streams"]}
    assert (
        streams[UM_MARK_PRICE]["reconnects"],
        streams[UM_MARK_PRICE]["silent_reconnects"],
    ) == (5, 2)


# =========================================================================== #
# C. the lifecycle log
# =========================================================================== #
def records(root: Path) -> list[dict]:
    return [
        json.loads(line) for line in lifecycle_path(root).read_bytes().splitlines() if line
    ]


def events(root: Path) -> list[str]:
    return [record["event"] for record in records(root)]


def test_a_clean_run_writes_up_then_down_and_a_second_run_finds_nothing_missing(tmp_path):
    for _ in range(2):
        asyncio.run(run_briefly(service_for(tmp_path), seconds=0.2))
    assert events(tmp_path) == [LIFECYCLE_UP, LIFECYCLE_DOWN] * 2
    up, down = records(tmp_path)[:2]
    assert up["contract_hash"] == CONTRACT.contract_hash
    assert (down["reason"], down["shutdown"]) == ("stopped", "complete")
    assert "shutdown_errors" not in down and "error" not in down


FABRICATED_CAUSES = ("crash", "kill", "power", "network", "host", "oom", "signal")


def test_a_run_that_left_no_down_is_named_by_the_next_start_without_a_cause(tmp_path):
    log = LifecycleLog(tmp_path, wall_ns=lambda: 1_000)
    log.up()
    LifecycleLog(tmp_path, wall_ns=lambda: 2_000).up()
    assert events(tmp_path) == [LIFECYCLE_UP, LIFECYCLE_DOWN_MISSING, LIFECYCLE_UP]
    missing = records(tmp_path)[1]
    assert (missing["last_event"], missing["last_wall_ns"]) == (LIFECYCLE_UP, 1_000)
    assert missing["last_line_unreadable"] is False
    text = json.dumps(missing).lower()
    assert not [word for word in FABRICATED_CAUSES if word in text]
    # the control: a run that did write its down leaves nothing to name
    LifecycleLog(tmp_path).down("stopped")
    LifecycleLog(tmp_path).up()
    assert events(tmp_path)[-2:] == [LIFECYCLE_DOWN, LIFECYCLE_UP]


def test_a_torn_last_line_is_kept_and_never_swallows_the_next_record(tmp_path):
    log = LifecycleLog(tmp_path)
    log.up()
    fragment = b'{"schema":"chimera.recorder-lifecycle/1","event":"recorder.do'
    with lifecycle_path(tmp_path).open("ab") as handle:
        handle.write(fragment)
    log.up()
    lines = lifecycle_path(tmp_path).read_bytes().split(b"\n")
    assert lines[1] == fragment, "the fragment is preserved as its own line, unaltered"
    after = [json.loads(line) for line in lines[2:] if line]
    assert [record["event"] for record in after] == [LIFECYCLE_DOWN_MISSING, LIFECYCLE_UP]
    assert after[0]["last_line_unreadable"] is True and "last_event" not in after[0]


def test_every_record_is_fsynced_before_the_call_returns(tmp_path, monkeypatch):
    on_disk_at_fsync: list[bytes] = []
    real_fsync = health_module.os.fsync

    def fsync(fd):
        real_fsync(fd)
        on_disk_at_fsync.append(lifecycle_path(tmp_path).read_bytes())

    monkeypatch.setattr(health_module.os, "fsync", fsync)
    log = LifecycleLog(tmp_path)
    log.up()
    log.down("stopped")
    assert len(on_disk_at_fsync) == 2
    assert on_disk_at_fsync[0].count(b"\n") == 1 and on_disk_at_fsync[1].count(b"\n") == 2


def test_up_is_on_disk_before_recovery_and_a_failed_recovery_says_so(tmp_path, monkeypatch):
    service = service_for(tmp_path)
    at_recovery: list[list[str]] = []

    def recover():
        at_recovery.append(events(tmp_path))
        raise RecorderSinkError("recovery could not read a tail")

    monkeypatch.setattr(service, "recover", recover)
    with pytest.raises(RecorderSinkError):
        asyncio.run(run_briefly(service))
    assert at_recovery == [[LIFECYCLE_UP]]
    down = records(tmp_path)[-1]
    assert (down["event"], down["reason"], down["shutdown"]) == (
        LIFECYCLE_DOWN,
        "startup_failed",
        "not_run",
    )
    assert "recovery could not read a tail" in down["error"]


def test_down_is_written_only_after_every_sink_is_closed(tmp_path, monkeypatch):
    service = service_for(tmp_path)
    at_close: list[str] = []
    for sink in service.sinks.values():
        real_close = sink.close

        def close(real_close=real_close):
            at_close.append(events(tmp_path)[-1])
            real_close()

        monkeypatch.setattr(sink, "close", close)
    asyncio.run(run_briefly(service, seconds=0.2))
    assert len(at_close) == len(service.sinks)
    assert set(at_close) == {LIFECYCLE_UP}, "no down existed while a sink was still open"
    assert events(tmp_path)[-1] == LIFECYCLE_DOWN


def test_a_close_that_fails_is_a_shutdown_with_errors_not_a_clean_one(tmp_path, monkeypatch):
    service = service_for(tmp_path)

    def close():
        raise RecorderSinkError("the disk went away")

    monkeypatch.setattr(service.sinks[UM_KLINE_1M], "close", close)
    asyncio.run(run_briefly(service, seconds=0.2))
    down = records(tmp_path)[-1]
    assert (down["reason"], down["shutdown"]) == ("stopped", "errors")
    assert any("the disk went away" in note for note in down["shutdown_errors"])


def test_a_shutdown_that_raises_is_recorded_as_failed_and_still_raises(tmp_path, monkeypatch):
    service = service_for(tmp_path)

    def shutdown():
        raise RuntimeError("shutdown blew up")

    monkeypatch.setattr(service, "_shutdown", shutdown)
    with pytest.raises(RuntimeError, match="shutdown blew up"):
        asyncio.run(run_briefly(service, seconds=0.2))
    down = records(tmp_path)[-1]
    assert (down["event"], down["shutdown"]) == (LIFECYCLE_DOWN, "failed")
    assert "shutdown blew up" in down["shutdown_errors"][-1]


def test_a_failed_task_and_an_interrupted_run_each_say_which(tmp_path):
    failing = service_for(tmp_path / "failed")
    failing.clients = (
        ScriptedClient(
            "um-test", (UM_KLINE_1M,), failing._record, fail=RuntimeError("socket bug")
        ),
    )
    with pytest.raises(RecorderServiceError):
        asyncio.run(run_briefly(failing))
    down = records(tmp_path / "failed")[-1]
    assert down["reason"] == "task_failed" and "socket bug" in down["error"]

    async def cancelled(service):
        task = asyncio.create_task(service.run(asyncio.Event()))
        await asyncio.sleep(0.2)
        task.cancel()
        await task

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(cancelled(service_for(tmp_path / "cancelled")))
    down = records(tmp_path / "cancelled")[-1]
    assert down["reason"] == "interrupted" and "CancelledError" in down["error"]


def test_a_lifecycle_log_that_cannot_be_written_is_reported_and_recording_goes_on(tmp_path):
    lifecycle_path(tmp_path).mkdir(parents=True)  # a directory where the file should be
    with pytest.raises(RecorderHealthError):
        LifecycleLog(tmp_path).up()
    service = service_for(tmp_path, events=[kline_event(minute_ms(0))])
    result = asyncio.run(run_briefly(service, seconds=0.2))
    assert result.events == 1, "the market data was still recorded"
    assert any(f"{LIFECYCLE_UP} not recorded" in note for note in result.errors)
    assert any(f"{LIFECYCLE_DOWN} not recorded" in note for note in result.errors)
    assert any(
        f"{LIFECYCLE_UP} not recorded" in note for note in read_heartbeat(tmp_path)["errors"]
    )


LIFECYCLE_FIELDS = {
    "schema",
    "event",
    "wall_ns",
    "wall_utc",
    "pid",
    "contract_id",
    "contract_hash",
    "source_revision",
    "reason",
    "error",
    "shutdown",
    "shutdown_errors",
    "last_event",
    "last_wall_ns",
    "last_line_unreadable",
    "detail",
}


def test_the_lifecycle_log_is_operational_evidence_and_never_market_evidence(tmp_path):
    """It carries no market field, and nothing that builds a recorded value, a
    digest or a decision reads it: the same raw normalizes to the same bytes with
    and without a log beside it."""
    for root in (tmp_path / "with", tmp_path / "without"):
        raw_klines(root, range(5))
    log = LifecycleLog(tmp_path / "with")
    log.up()
    log.down("stopped")
    LifecycleLog(tmp_path / "with").up()
    assert {key for record in records(tmp_path / "with") for key in record} <= LIFECYCLE_FIELDS
    built = {
        name: MinuteNormalizer(tmp_path / name, CONTRACT).build_day("um", DAY)
        for name in ("with", "without")
    }
    assert built["with"].digest == built["without"].digest
    assert built["with"].parquet_sha256 == built["without"].parquet_sha256

    # Nothing but its writer names it: no normalizer, sink, feed, runner, report
    # or tool can reach the file except by these identifiers.
    repository = Path(__file__).resolve().parents[1]
    names = (
        "lifecycle.ndjson",
        "LIFECYCLE_FILE",
        "lifecycle_path",
        "LifecycleLog",
        "recorder.up",
        "recorder.down",
    )
    readers = sorted(
        path.relative_to(repository).as_posix()
        for folder in ("chimera", "nn", "tools")
        for path in (repository / folder).rglob("*.py")
        if any(name in path.read_text(encoding="utf-8") for name in names)
    )
    assert readers == ["chimera/recorder/health.py", "chimera/recorder/service.py"]


# =========================================================================== #
# D. gen3 is unchanged
# =========================================================================== #
#: gen3's identity on main at cb51dcb, before R1-h, from `canonical_material`.
GEN3_HASH = "55268a0fe805f937824e1ae45ee3a32e6db00f0714910b31c1cc3266d63cde7c"


def test_gen3s_contract_hash_is_unchanged_by_r1h():
    gen3 = load_recorder_contract(GEN3_CONTRACT_ID)
    assert gen3.contract_hash == GEN3_HASH
    assert hashlib.sha256(canonical_material(gen3).encode("utf-8")).hexdigest() == GEN3_HASH
