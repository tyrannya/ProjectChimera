"""R1-h: atomic parquet publication, the silence watchdog, and recorder evidence.

Three claims, each with its two-sided control:

* a normalised day is PUBLISHED, so a reader never sees a partial rewrite;
* a connected socket that stops delivering frames is reconnected rather than
  believed;
* a recorder that dies leaves a record saying it was running, because the
  heartbeat -- which is replaced in place -- cannot say so.

ENGINEERING only. Nothing here reads or asserts an economic quantity, and
nothing touches a contract hash, a preregistration or a scientific artifact.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from chimera.recorder.contract import load_recorder_contract
from chimera.recorder.health import (
    LIFECYCLE_DOWN,
    LIFECYCLE_SCHEMA,
    LIFECYCLE_UNCLEAN,
    LIFECYCLE_UP,
    LifecycleLog,
    lifecycle_path,
)
from chimera.recorder.sink import RecorderSinkError, publish_atomically
from chimera.recorder.streams import SILENCE_TIMEOUT_S, StreamClient, Subscription
from chimera.recorder.streams import StreamKind


# ---------------------------------------------------------------------------
# atomic publication
# ---------------------------------------------------------------------------
def test_a_failed_write_leaves_the_previous_file_intact(tmp_path):
    """The defect side: a reader must never be handed half a rewrite.

    ``to_parquet`` writing in place truncates the destination first, so a crash
    -- or a reader looking at the wrong moment -- finds a file that is valid and
    short, or not a container at all. Publication has no such window.
    """
    target = tmp_path / "day.parquet"
    target.write_bytes(b"the previous day")

    def explode(temporary: Path) -> None:
        temporary.write_bytes(b"half of the new day")
        raise OSError("disk full")

    with pytest.raises(RecorderSinkError, match="could not publish"):
        publish_atomically(target, explode)

    assert target.read_bytes() == b"the previous day"
    assert list(tmp_path.glob("*.tmp")) == [], "a failed publish left its temporary behind"


def test_a_successful_publish_replaces_the_file_whole(tmp_path):
    """The control: the new bytes appear, and the temporary does not survive."""
    target = tmp_path / "day.parquet"
    target.write_bytes(b"the previous day")

    publish_atomically(target, lambda temporary: temporary.write_bytes(b"the new day"))

    assert target.read_bytes() == b"the new day"
    assert list(tmp_path.glob("*.tmp")) == []


def test_a_writer_that_raises_something_else_also_cleans_up(tmp_path):
    """Not only ``OSError``: any failure must leave no half-published file."""
    target = tmp_path / "day.parquet"

    def explode(temporary: Path) -> None:
        temporary.write_bytes(b"partial")
        raise ValueError("the frame was not what the writer expected")

    with pytest.raises(ValueError):
        publish_atomically(target, explode)

    assert not target.exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_the_normalizer_publishes_rather_than_writes_in_place(tmp_path):
    """The property that matters, asserted on the normalizer's own output.

    A day is rebuilt repeatedly while the runner reads it. Each rebuild must
    land whole: the file's inode changes, and no ``.tmp`` is left in the
    market's directory for a reader to trip over.
    """
    from chimera.demo.fixtures import SyntheticFeed
    from chimera.recorder.normalize import MinuteNormalizer

    contract = load_recorder_contract("btcusdt-prospective-gen3")
    root = tmp_path / "recorder"
    feed = SyntheticFeed(root, contract)
    feed.write_days(["2026-09-19"])

    normalizer = MinuteNormalizer(root, contract)
    parquet = normalizer.parquet_path("um", "2026-09-19")
    assert parquet.is_file()
    before = parquet.stat().st_ino

    normalizer.build_day("um", "2026-09-19")

    assert parquet.stat().st_ino != before, "the day was rewritten in place"
    assert list(parquet.parent.glob("*.tmp")) == []


# ---------------------------------------------------------------------------
# the silence watchdog
# ---------------------------------------------------------------------------
class _SilentSocket:
    """A socket that is open, healthy and delivers nothing. The failure case.

    It is what a ping cannot distinguish from a quiet market: the connection is
    fine, the peer is there, and no data arrives.
    """

    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send(self, message: str) -> None:
        self.sent.append(message)

    async def recv(self) -> str:
        await asyncio.sleep(3600)
        raise AssertionError("unreachable")

    async def __aenter__(self) -> "_SilentSocket":
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False


class _TalkingSocket(_SilentSocket):
    """The control: a socket that keeps delivering, slowly but steadily."""

    def __init__(self, frames: int) -> None:
        super().__init__()
        self._left = frames

    async def recv(self) -> str | None:
        if self._left <= 0:
            return None
        self._left -= 1
        await asyncio.sleep(0.001)
        return json.dumps({"e": "not-a-kind-this-client-subscribed-to"})


def _client(socket, *, silence_timeout_s: float, mono) -> StreamClient:
    return StreamClient(
        "wss://example.invalid/ws",
        [
            Subscription(
                stream_id="um.bookTicker",
                venue_name="btcusdt@bookTicker",
                kind=StreamKind.BOOK_TICKER,
            )
        ],
        lambda event: None,
        connect=lambda *args, **kwargs: socket,
        mono_ns=mono,
        silence_timeout_s=silence_timeout_s,
        name="test-ws",
    )


def test_a_connected_socket_that_stops_delivering_is_reconnected():
    """The defect side: silence ends the session instead of being believed."""
    clock = {"ns": 0}

    def mono() -> int:
        # Every read of the clock advances it by a second, so the watchdog's
        # budget is exhausted by the loop itself rather than by real waiting.
        clock["ns"] += 1_000_000_000
        return clock["ns"]

    socket = _SilentSocket()
    client = _client(socket, silence_timeout_s=5.0, mono=mono)
    stop = asyncio.Event()

    async def drive() -> None:
        session = asyncio.ensure_future(client._session(stop))
        await asyncio.wait_for(session, timeout=5)

    asyncio.run(drive())

    assert client.counters.silent_reconnects == 1
    assert client.counters.frames == 0
    assert socket.sent, "the client never subscribed, so it proved nothing about silence"


def test_a_socket_that_keeps_delivering_is_left_alone():
    """The control: a stream that speaks is never cut by the watchdog."""
    clock = {"ns": 0}

    def mono() -> int:
        clock["ns"] += 1_000_000_000
        return clock["ns"]

    socket = _TalkingSocket(frames=20)
    client = _client(socket, silence_timeout_s=5.0, mono=mono)
    stop = asyncio.Event()

    asyncio.run(asyncio.wait_for(client._session(stop), timeout=5))

    assert client.counters.silent_reconnects == 0
    assert client.counters.frames == 20


def test_the_silence_budget_is_longer_than_any_stream_it_watches():
    """The constant's own claim, kept honest.

    Every websocket client this contract builds carries ``kline_1m``, which the
    venue pushes every second or two. ``um.funding``, the one genuinely sparse
    stream, has no websocket source and is polled instead. So the budget must be
    comfortably longer than a minute and must not be so long that a dead socket
    survives a whole normalisation cycle.
    """
    assert 60.0 < SILENCE_TIMEOUT_S <= 300.0


def test_a_non_positive_silence_budget_is_refused():
    """Zero would not disable the watchdog; it would cut every session at once."""
    from chimera.recorder.streams import RecorderStreamError

    with pytest.raises(RecorderStreamError, match="silence_timeout_s must be positive"):
        _client(_SilentSocket(), silence_timeout_s=0.0, mono=lambda: 0)


# ---------------------------------------------------------------------------
# recorder evidence
# ---------------------------------------------------------------------------
def _events(root: Path) -> list[str]:
    """Every well-formed event in the file, oldest first.

    A malformed line is skipped rather than raising, because one of the tests
    below deliberately writes a torn one: a reader of this file has to survive
    exactly the interruption the file exists to record.
    """
    events: list[str] = []
    for line in lifecycle_path(root).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line)["event"])
        except ValueError:
            continue
    return events


def test_a_clean_run_leaves_an_up_and_a_down(tmp_path):
    """The control: a recorder that stopped on purpose says so."""
    log = LifecycleLog(tmp_path)
    log.up(source_revision="abc123")
    log.down("stopped")

    assert _events(tmp_path) == [LIFECYCLE_UP, LIFECYCLE_DOWN]
    first = json.loads(lifecycle_path(tmp_path).read_text(encoding="utf-8").splitlines()[0])
    assert first["schema"] == LIFECYCLE_SCHEMA
    assert first["source_revision"] == "abc123"
    assert first["wall_utc"].endswith("+00:00")


def test_a_killed_run_is_named_by_the_next_start(tmp_path):
    """The defect side, and the whole reason the file is appended.

    A process that is killed writes no ``down``. The heartbeat left behind says
    only that the recorder was alive at some past instant, which a reader cannot
    tell from a stale copy. The next start reads the tail, finds an unmatched
    ``up``, and records that the previous stop was unwitnessed -- as its own
    line, because it is a fact about a different process.
    """
    LifecycleLog(tmp_path).up()  # and then the host dies

    LifecycleLog(tmp_path).up()

    assert _events(tmp_path) == [LIFECYCLE_UP, LIFECYCLE_UNCLEAN, LIFECYCLE_UP]
    unclean = json.loads(lifecycle_path(tmp_path).read_text(encoding="utf-8").splitlines()[1])
    assert unclean["previous_up_ns"] > 0
    assert "asserts nothing about its cause" in unclean["detail"]


def test_a_clean_stop_is_not_reported_as_unclean(tmp_path):
    """The two-sided half: a matched ``down`` produces no unclean line."""
    log = LifecycleLog(tmp_path)
    log.up()
    log.down("stopped")

    LifecycleLog(tmp_path).up()

    assert _events(tmp_path) == [LIFECYCLE_UP, LIFECYCLE_DOWN, LIFECYCLE_UP]


def test_a_torn_last_line_does_not_hide_the_history(tmp_path):
    """The tail is exactly what a kill can truncate, so it may not raise.

    The earlier lines are still evidence. Refusing the file because its last
    bytes are missing would discard the history in order to protect a guess --
    and the torn line is itself a symptom of the event being recorded.
    """
    log = LifecycleLog(tmp_path)
    log.up()
    with open(lifecycle_path(tmp_path), "a", encoding="utf-8") as handle:
        handle.write('{"event": "recorder.do')

    assert log.last_event()["event"] == LIFECYCLE_UP
    LifecycleLog(tmp_path).up()
    assert _events(tmp_path)[-2:] == [LIFECYCLE_UNCLEAN, LIFECYCLE_UP]


def test_a_record_after_a_torn_line_is_not_swallowed_by_it(tmp_path):
    """The defect a torn tail caused, and the reason the writer terminates one.

    Appending straight onto a half-written line concatenates the two into one
    unparseable line, destroying the NEW record as well as the old. The record
    lost that way is the worst one available: the ``unclean_stop`` a start
    writes on finding an unmatched ``up`` -- which is exactly what the torn line
    is a symptom of. So the fragment is closed off first, stays on disk as its
    own unreadable line, and every record after it is well formed.
    """
    log = LifecycleLog(tmp_path)
    log.up()
    with open(lifecycle_path(tmp_path), "a", encoding="utf-8") as handle:
        handle.write('{"event": "recorder.do')

    LifecycleLog(tmp_path).up()

    lines = lifecycle_path(tmp_path).read_text(encoding="utf-8").splitlines()
    # The fragment is still there, alone, and unparseable.
    assert lines[1] == '{"event": "recorder.do'
    with pytest.raises(ValueError):
        json.loads(lines[1])
    # And both records written after it survived.
    assert _events(tmp_path) == [LIFECYCLE_UP, LIFECYCLE_UNCLEAN, LIFECYCLE_UP]


def test_an_unwritable_log_never_stops_the_recorder(tmp_path):
    """Evidence about the recorder is not part of the recording.

    A full disk must not stop the recorder starting, and must not raise out of a
    shutdown path and lose the clean stop it was recording. The heartbeat's
    ``disk_free_bytes`` is what surfaces the underlying fault.
    """
    blocked = tmp_path / "health"
    blocked.write_text("not a directory", encoding="utf-8")

    log = LifecycleLog(tmp_path)
    assert log.up() is None
    assert log.down("stopped") is None
    assert log.last_event() is None


def test_the_service_records_its_own_start_and_stop(tmp_path):
    """End to end, through the service rather than the log on its own."""
    from chimera.recorder.service import RecorderService

    contract = load_recorder_contract("btcusdt-prospective-gen3")
    root = tmp_path / "recorder"
    root.mkdir(parents=True, exist_ok=True)
    service = RecorderService(contract, root, clients=(), gapfill=False)

    stop = asyncio.Event()
    stop.set()
    asyncio.run(service.run(stop))

    assert _events(root) == [LIFECYCLE_UP, LIFECYCLE_DOWN]
    up = json.loads(lifecycle_path(root).read_text(encoding="utf-8").splitlines()[0])
    assert up["contract_hash"] == contract.contract_hash
    assert up["contract_id"] == contract.contract_id
