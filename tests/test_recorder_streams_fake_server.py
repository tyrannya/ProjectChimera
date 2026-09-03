"""The websocket client, against a local fake server rather than against Binance.

Every test here starts a real websocket server on ``127.0.0.1`` and talks to it
over a real socket. Nothing leaves the machine, and no test depends on Binance
being up, on the market being open, or on a frame arriving within some number of
seconds. That matters twice over: it makes the suite deterministic, and it makes
it possible to test the things a real exchange will not do on request — close the
connection mid-session, deliver the same frame twice, deliver update ids
backwards, send a frame that is not JSON.

**The clock is injected, so a 24-hour rule is tested in milliseconds.** Section
4.2 requires a proactive reconnect at 23 h 50 m, before the exchange's own
24-hour close. A test that waited for that would not be a test. The client takes
``mono_ns`` as an argument, so the test moves the clock forward by a day and
asserts what the client does about it.

**What is deliberately *not* asserted here.** Whether a frame is stored, whether
it is a duplicate on disk, and whether a minute is complete. Those are the
sink's and the normalizer's, tested in their own files. This file's subject is
the connection: what it subscribes to, what it does when it breaks, what it
counts, and how quickly it can be stopped.
"""

from __future__ import annotations

import asyncio
import json
import socket
import time

import pytest
from websockets.asyncio.server import serve

from chimera.recorder.contract import load_recorder_contract
from chimera.recorder.events import (
    SPOT_BOOK_TICKER,
    UM_BOOK_TICKER,
    UM_KLINE_1M,
    UM_MARK_PRICE,
)
from chimera.recorder.streams import (
    Backoff,
    PROACTIVE_RECONNECT_S,
    RecorderStreamError,
    StreamClient,
    Subscription,
    StreamKind,
    subscriptions_for,
)
from chimera.recorder.sink import AppendOutcome, RawSink
from tests.recorder_synthetic import book_ws_frame, kline_ws_frame, mark_ws_frame, minute_ms

CONTRACT = load_recorder_contract()
OPEN_MS = minute_ms(0)
UM_SUBSCRIPTIONS = subscriptions_for(CONTRACT, "um")

#: Short enough that a test finishes, long enough that the ordering is real.
TEST_BACKOFF = dict(initial=0.01, maximum=0.02, factor=2.0, jitter=0.0)


class FakeVenue:
    """A local websocket server that replays scripted frames.

    Deliberately not a Binance simulator. Each connection is handed the next
    script in the list, so a test says "first connection: two frames then close;
    second connection: one frame" and reads back what the client did about it.
    """

    def __init__(self, scripts, *, close_after=True):
        self.scripts = [list(script) for script in scripts]
        self.close_after = close_after
        self.connections = 0
        self.subscriptions: list[dict] = []
        self.delivered = asyncio.Event()
        self._server = None
        self.port = 0

    async def __aenter__(self):
        self._server = await serve(self._handle, "127.0.0.1", 0).__aenter__()
        self.port = self._server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *exc):
        await self._server.__aexit__(*exc)

    @property
    def url(self) -> str:
        return f"ws://127.0.0.1:{self.port}"

    async def _handle(self, websocket):
        index = self.connections
        self.connections += 1
        try:
            request = json.loads(await websocket.recv())
            self.subscriptions.append(request)
            await websocket.send(json.dumps({"result": None, "id": request.get("id")}))
        except Exception:  # pragma: no cover - a client that closed while subscribing
            return
        script = self.scripts[index] if index < len(self.scripts) else []
        for frame in script:
            body = frame if isinstance(frame, str) else json.dumps(frame)
            await websocket.send(body)
        self.delivered.set()
        if self.close_after:
            await websocket.close()
            return
        try:
            await websocket.wait_closed()
        except Exception:  # pragma: no cover
            return


class Clock:
    """A monotonic clock a test can move."""

    def __init__(self, ns: int = 0) -> None:
        self.ns = ns

    def __call__(self) -> int:
        return self.ns


def collector():
    """An ``on_event`` that keeps what it was given."""
    events = []
    return events, events.append


def client_for(venue, on_event, **options) -> StreamClient:
    options.setdefault("backoff", Backoff(**TEST_BACKOFF))
    options.setdefault("open_timeout", 5.0)
    options.setdefault("ping_interval", None)
    options.setdefault("ping_timeout", None)
    return StreamClient(venue.url, UM_SUBSCRIPTIONS, on_event, name="um-test", **options)


async def run_until(client, stop, *, deadline: float = 5.0):
    """Run the client, with a hard ceiling so a broken test fails rather than hangs."""
    task = asyncio.create_task(client.run(stop))
    try:
        await asyncio.wait_for(task, timeout=deadline)
    except asyncio.TimeoutError:  # pragma: no cover - only on a genuine hang
        task.cancel()
        raise


# --- A. the ordinary case -------------------------------------------------------
def test_frames_arrive_are_parsed_and_are_delivered_as_raw_events():
    async def scenario():
        frames = [
            kline_ws_frame(OPEN_MS, closed=True),
            mark_ws_frame(OPEN_MS + 1_000),
            book_ws_frame(11, event_ms=OPEN_MS + 2_000),
        ]
        async with FakeVenue([frames], close_after=False) as venue:
            events, on_event = collector()
            client = client_for(venue, on_event)
            stop = asyncio.Event()

            async def stop_when_done():
                await venue.delivered.wait()
                await asyncio.sleep(0.05)
                stop.set()

            await asyncio.gather(run_until(client, stop), stop_when_done())
            return events, client, venue

    events, client, venue = asyncio.run(scenario())
    assert [event.stream for event in events] == [UM_KLINE_1M, UM_MARK_PRICE, UM_BOOK_TICKER]
    assert client.counters.frames >= 4, "the subscribe acknowledgement is a frame too"
    assert client.counters.events == 3
    assert client.counters.ignored_frames == 1, "the acknowledgement is ignored, not parsed"
    # The contract normalises its stream list at parse time, so the subscription
    # order is the contract's and is the same on every host and every restart.
    assert venue.subscriptions[0] == {
        "method": "SUBSCRIBE",
        "params": ["btcusdt@bookTicker", "btcusdt@kline_1m", "btcusdt@markPrice@1s"],
        "id": 1,
    }


def test_the_canonical_stamp_comes_from_the_exchange_and_the_receipt_from_this_host():
    async def scenario():
        async with FakeVenue([[kline_ws_frame(OPEN_MS)]]) as venue:
            events, on_event = collector()
            client = client_for(venue, on_event, wall_ns=lambda: 1_700_000_000_000_000_000)
            stop = asyncio.Event()

            async def stop_when_done():
                await venue.delivered.wait()
                await asyncio.sleep(0.05)
                stop.set()

            await asyncio.gather(run_until(client, stop), stop_when_done())
            return events, client

    events, client = asyncio.run(scenario())
    assert events[0].canonical_ns == OPEN_MS * 1_000_000, "the minute key, not the clock"
    assert events[0].receipt_wall_ns == 1_700_000_000_000_000_000
    assert events[0].receipt_mono_ns >= 0
    assert client.skew.samples == 1, "skew is measured where the exchange published an E"
    assert client.skew.median_ms() is not None


def test_a_frame_for_another_instrument_is_not_recorded_as_this_one():
    async def scenario():
        async with FakeVenue([[kline_ws_frame(OPEN_MS, symbol="ETHUSDT")]]) as venue:
            events, on_event = collector()
            client = client_for(venue, on_event)
            stop = asyncio.Event()

            async def stop_when_done():
                await venue.delivered.wait()
                await asyncio.sleep(0.05)
                stop.set()

            await asyncio.gather(run_until(client, stop), stop_when_done())
            return events, client

    events, client = asyncio.run(scenario())
    assert events == [], "a BTCUSDT client must not record an ETHUSDT candle"
    assert client.counters.ignored_frames >= 1


# --- B. reconnect ---------------------------------------------------------------
def test_a_closed_connection_is_reconnected_and_counted():
    async def scenario():
        scripts = [[kline_ws_frame(OPEN_MS)], [kline_ws_frame(OPEN_MS + 60_000)]]
        async with FakeVenue(scripts) as venue:
            events, on_event = collector()
            client = client_for(venue, on_event)
            stop = asyncio.Event()

            async def stop_when_two():
                while len(events) < 2:
                    await asyncio.sleep(0.01)
                stop.set()

            await asyncio.gather(run_until(client, stop), stop_when_two())
            return events, client, venue

    events, client, venue = asyncio.run(scenario())
    assert len(events) == 2, "the second frame arrived on the second connection"
    assert venue.connections >= 2
    assert client.counters.reconnects >= 1
    assert len(venue.subscriptions) >= 2, "every connection subscribes again"


def test_a_connection_that_cannot_be_made_is_retried_and_counted():
    """A refused connection is a network fact, not a bug: back off and try again."""

    async def scenario():
        # Bind and immediately release a port so nothing is listening on it.
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()

        events, on_event = collector()
        client = StreamClient(
            f"ws://127.0.0.1:{port}",
            UM_SUBSCRIPTIONS,
            on_event,
            backoff=Backoff(**TEST_BACKOFF),
            name="um-refused",
            open_timeout=0.5,
        )
        stop = asyncio.Event()

        async def stop_after_two_attempts():
            while client.counters.reconnects < 2:
                await asyncio.sleep(0.01)
            stop.set()

        await asyncio.gather(run_until(client, stop, deadline=10.0), stop_after_two_attempts())
        return events, client

    events, client = asyncio.run(scenario())
    assert events == []
    assert client.counters.reconnects >= 2
    assert client.connected is False


def test_the_backoff_grows_from_one_second_to_sixty_and_resets_on_success():
    """Section 4.2's policy, as arithmetic rather than as a wait."""
    backoff = Backoff(jitter=0.0)
    delays = [backoff.next_delay() for _ in range(10)]
    assert delays[:4] == [1.0, 2.0, 4.0, 8.0]
    assert max(delays) == 60.0, "the ceiling is 60 s"
    assert delays == sorted(delays), "it grows and never shrinks"
    backoff.reset()
    assert backoff.next_delay() == 1.0, "a successful connection starts the sequence again"


def test_the_backoff_jitter_never_pushes_a_delay_past_the_ceiling():
    backoff = Backoff(jitter=0.5)
    for _ in range(20):
        assert backoff.next_delay(rand=lambda: 1.0) <= 60.0


def test_a_backoff_that_shrinks_or_starts_at_zero_is_refused():
    with pytest.raises(RecorderStreamError, match="backoff must grow"):
        Backoff(initial=0.0)
    with pytest.raises(RecorderStreamError, match="backoff must grow"):
        Backoff(initial=10.0, maximum=1.0)
    with pytest.raises(RecorderStreamError, match="jitter"):
        Backoff(jitter=1.0)


def test_the_client_waits_between_attempts_rather_than_spinning():
    async def scenario():
        async with FakeVenue([[], [], []]) as venue:
            events, on_event = collector()
            client = client_for(
                venue, on_event, backoff=Backoff(initial=0.2, maximum=0.2, jitter=0.0)
            )
            stop = asyncio.Event()

            async def stop_after_two():
                while client.counters.reconnects < 2:
                    await asyncio.sleep(0.01)
                stop.set()

            began = time.monotonic()
            await asyncio.gather(run_until(client, stop, deadline=10.0), stop_after_two())
            return time.monotonic() - began

    elapsed = asyncio.run(scenario())
    assert elapsed >= 0.2, "two attempts with a 0.2 s backoff cannot happen instantly"


# --- C. the proactive close before the exchange's 24-hour limit -----------------
def test_the_client_closes_the_session_itself_before_the_venue_would():
    """Section 4.2's 23 h 50 m rule, driven by an injected clock.

    The exchange closes a connection after 24 hours. Being closed *by* the venue
    mid-frame is a disconnect; closing first is a rotation. The client is given a
    clock the test controls, so what would take a day takes a millisecond.
    """

    async def scenario():
        scripts = [[kline_ws_frame(OPEN_MS)], [kline_ws_frame(OPEN_MS + 60_000)]]
        async with FakeVenue(scripts, close_after=False) as venue:
            clock = Clock()
            events = []

            def on_event(event):
                events.append(event)
                # The moment the first frame lands, a day has "passed".
                clock.ns += 24 * 3600 * 1_000_000_000

            client = client_for(venue, on_event, mono_ns=clock, session_seconds=100.0)
            stop = asyncio.Event()

            async def stop_when_two():
                while len(events) < 2:
                    await asyncio.sleep(0.01)
                stop.set()

            await asyncio.gather(run_until(client, stop), stop_when_two())
            return events, client, venue

    events, client, venue = asyncio.run(scenario())
    assert len(events) == 2
    assert venue.connections >= 2, "the client opened a second connection on its own"
    assert client.counters.reconnects >= 1


def test_the_proactive_deadline_is_ten_minutes_short_of_a_day():
    assert PROACTIVE_RECONNECT_S == 23 * 3600 + 50 * 60
    assert PROACTIVE_RECONNECT_S < 24 * 3600, "it must fire before the exchange's own close"
    assert 24 * 3600 - PROACTIVE_RECONNECT_S == 600


# --- D. duplicates and ordering -------------------------------------------------
def test_a_re_delivered_kline_frame_is_passed_on_and_the_sink_calls_it_a_duplicate():
    """The division of labour, asserted on both sides of it.

    The client does not deduplicate klines: a partial frame and a closed frame
    for one minute are different observations, and deciding which describes a
    finished minute is the normalizer's. What makes a re-delivery harmless is the
    sink's key, which is a function of the payload.
    """

    async def scenario():
        frame = kline_ws_frame(OPEN_MS)
        async with FakeVenue([[frame, frame]]) as venue:
            events, on_event = collector()
            client = client_for(venue, on_event)
            stop = asyncio.Event()

            async def stop_when_done():
                await venue.delivered.wait()
                await asyncio.sleep(0.05)
                stop.set()

            await asyncio.gather(run_until(client, stop), stop_when_done())
            return events

    events = asyncio.run(scenario())
    assert len(events) == 2, "the client passes on what the exchange sent"
    assert events[0].dedup_key == events[1].dedup_key


def test_the_sink_recognises_the_re_delivered_frame(tmp_path):
    """The other half of the previous test, on disk."""

    async def scenario():
        frame = kline_ws_frame(OPEN_MS)
        async with FakeVenue([[frame, frame]]) as venue:
            events, on_event = collector()
            client = client_for(venue, on_event)
            stop = asyncio.Event()

            async def stop_when_done():
                await venue.delivered.wait()
                await asyncio.sleep(0.05)
                stop.set()

            await asyncio.gather(run_until(client, stop), stop_when_done())
            return events

    events = asyncio.run(scenario())
    with RawSink(tmp_path, UM_KLINE_1M, contract=CONTRACT) as sink:
        outcomes = [sink.append(event).outcome for event in events]
    assert outcomes == [AppendOutcome.ACCEPTED, AppendOutcome.DUPLICATE]


def test_a_book_update_whose_id_does_not_advance_is_counted_and_dropped():
    """Section 4.1's rule for ``bookTicker``: ordered by ``u``, and only by ``u``."""

    async def scenario():
        frames = [
            book_ws_frame(10, event_ms=OPEN_MS),
            book_ws_frame(9, event_ms=OPEN_MS + 1),
            book_ws_frame(10, event_ms=OPEN_MS + 2),
            book_ws_frame(11, event_ms=OPEN_MS + 3),
        ]
        async with FakeVenue([frames]) as venue:
            events, on_event = collector()
            client = client_for(venue, on_event)
            stop = asyncio.Event()

            async def stop_when_done():
                await venue.delivered.wait()
                await asyncio.sleep(0.05)
                stop.set()

            await asyncio.gather(run_until(client, stop), stop_when_done())
            return events, client

    events, client = asyncio.run(scenario())
    books = [event for event in events if event.stream == UM_BOOK_TICKER]
    assert [event.dedup_key for event in books] == ["book:10", "book:11"]
    assert client.counters.out_of_order == 2, "one older, one repeated"


def test_no_timestamp_is_manufactured_for_a_stream_that_publishes_none():
    """A spot book client stamps with the receipt clock and says that it did."""

    async def scenario():
        subscription = Subscription(
            stream_id=SPOT_BOOK_TICKER,
            venue_name="btcusdt@bookTicker",
            kind=StreamKind.BOOK_TICKER,
        )
        async with FakeVenue([[book_ws_frame(5, event_ms=None)]]) as venue:
            events, on_event = collector()
            client = StreamClient(
                venue.url,
                [subscription],
                on_event,
                backoff=Backoff(**TEST_BACKOFF),
                name="spot-test",
                wall_ns=lambda: 1_700_000_000_123_456_789,
                ping_interval=None,
            )
            stop = asyncio.Event()

            async def stop_when_done():
                await venue.delivered.wait()
                await asyncio.sleep(0.05)
                stop.set()

            await asyncio.gather(run_until(client, stop), stop_when_done())
            return events, client

    events, client = asyncio.run(scenario())
    assert len(events) == 1
    assert events[0].canonical_ns == 1_700_000_000_123_456_789
    assert events[0].time_basis.value == "RECEIPT"
    assert client.skew.samples == 0, "there is no exchange time to compare against"


# --- E. malformed input ---------------------------------------------------------
def test_a_frame_that_is_not_json_is_counted_and_the_connection_survives():
    async def scenario():
        frames = ["this is not json", kline_ws_frame(OPEN_MS)]
        async with FakeVenue([frames]) as venue:
            events, on_event = collector()
            client = client_for(venue, on_event)
            stop = asyncio.Event()

            async def stop_when_done():
                await venue.delivered.wait()
                await asyncio.sleep(0.05)
                stop.set()

            await asyncio.gather(run_until(client, stop), stop_when_done())
            return events, client

    events, client = asyncio.run(scenario())
    assert client.counters.decode_errors == 1
    assert len(events) == 1, "the good frame after the bad one still arrived"


def test_a_frame_missing_a_published_field_is_dropped_rather_than_defaulted():
    async def scenario():
        broken = kline_ws_frame(OPEN_MS)
        del broken["k"]["o"]
        async with FakeVenue([[broken, kline_ws_frame(OPEN_MS + 60_000)]]) as venue:
            events, on_event = collector()
            client = client_for(venue, on_event)
            stop = asyncio.Event()

            async def stop_when_done():
                await venue.delivered.wait()
                await asyncio.sleep(0.05)
                stop.set()

            await asyncio.gather(run_until(client, stop), stop_when_done())
            return events, client

    events, client = asyncio.run(scenario())
    assert client.counters.decode_errors == 1
    assert [event.minute_open_ms for event in events] == [OPEN_MS + 60_000]


def test_a_frame_that_is_a_json_array_is_ignored_rather_than_parsed():
    async def scenario():
        async with FakeVenue([["[1, 2, 3]", kline_ws_frame(OPEN_MS)]]) as venue:
            events, on_event = collector()
            client = client_for(venue, on_event)
            stop = asyncio.Event()

            async def stop_when_done():
                await venue.delivered.wait()
                await asyncio.sleep(0.05)
                stop.set()

            await asyncio.gather(run_until(client, stop), stop_when_done())
            return events, client

    events, client = asyncio.run(scenario())
    assert client.counters.decode_errors == 1 and len(events) == 1


# --- F. stopping ----------------------------------------------------------------
def test_setting_the_stop_event_ends_the_run_without_waiting_for_a_backoff():
    """A service that took a minute to stop would be a service that gets killed."""

    async def scenario():
        async with FakeVenue([[]], close_after=False) as venue:
            events, on_event = collector()
            client = client_for(venue, on_event, backoff=Backoff(initial=60.0, maximum=60.0))
            stop = asyncio.Event()

            async def stop_soon():
                await venue.delivered.wait()
                stop.set()

            began = time.monotonic()
            await asyncio.gather(run_until(client, stop, deadline=5.0), stop_soon())
            return time.monotonic() - began

    elapsed = asyncio.run(scenario())
    assert elapsed < 5.0, "the stop must not wait out the 60 s backoff"


def test_cancelling_the_run_task_propagates_and_leaves_nothing_behind():
    async def scenario():
        async with FakeVenue([[]], close_after=False) as venue:
            events, on_event = collector()
            client = client_for(venue, on_event)
            stop = asyncio.Event()
            task = asyncio.create_task(client.run(stop))
            await venue.delivered.wait()
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await asyncio.sleep(0.05)
            # The fake server's own accept loop belongs to the fixture and is
            # torn down with it; what must be gone is everything the client made.
            return [
                repr(t)
                for t in asyncio.all_tasks()
                if t is not asyncio.current_task() and "chimera" in repr(t.get_coro())
            ], client.connected

    leftovers, connected = asyncio.run(scenario())
    assert leftovers == [], f"cancellation left {leftovers} running"
    assert connected is False, "a cancelled client must not still report a connection"


# --- G. construction refusals ---------------------------------------------------
def test_a_client_with_nothing_to_subscribe_to_is_refused():
    with pytest.raises(RecorderStreamError, match="no subscriptions"):
        StreamClient("ws://127.0.0.1:1", [], lambda event: None)


def test_a_client_that_subscribes_to_one_stream_twice_is_refused():
    with pytest.raises(RecorderStreamError, match="twice"):
        StreamClient(
            "ws://127.0.0.1:1",
            [UM_SUBSCRIPTIONS[0], UM_SUBSCRIPTIONS[0]],
            lambda event: None,
        )


def test_a_subscription_needs_a_recorder_stream_id_and_a_venue_name():
    with pytest.raises(RecorderStreamError, match="stream_id"):
        Subscription(stream_id="kline", venue_name="btcusdt@kline_1m", kind=StreamKind.KLINE)
    with pytest.raises(RecorderStreamError, match="venue_name"):
        Subscription(stream_id="um.kline_1m", venue_name="", kind=StreamKind.KLINE)
    with pytest.raises(RecorderStreamError, match="StreamKind"):
        Subscription(stream_id="um.kline_1m", venue_name="x", kind="KLINE")  # type: ignore[arg-type]
