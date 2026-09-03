"""Binance public market-data websockets, and the reconnect policy around them.

This is the first module in :mod:`chimera.recorder` that opens a socket, and it
is deliberately the *only* kind of thing it does. It connects to a public
market-data endpoint, subscribes to named streams, turns each frame into the
:class:`~chimera.recorder.events.RawEvent` the offline core already knows how to
store, and hands it to a callback. It does not write files, it does not
normalize, it does not decide anything about coverage, and it holds no
credential — every endpoint below is public and unauthenticated, and there is no
version of this recorder that needs a key.

**What is canonical, and what is merely observed.** Section 4.1 of the adopted
plan fixes the timestamp discipline and this module implements it literally:

* a kline is stamped by its own open time ``k.t``;
* a mark-price update is stamped by the exchange's event time ``E``;
* a USD-M book update is stamped by ``E``;
* a *spot* book update has no exchange event time at all, so it is stamped with
  the local receipt wall clock and :class:`~chimera.recorder.events.TimeBasis`
  records that it was — see :meth:`BookTickerEvent.to_raw_event`;
* every frame also carries the local receipt wall clock and the local monotonic
  clock, separately, because one of them is comparable with the exchange and the
  other is the only one that is monotonic.

Clock skew is *measured* here (``receipt_wall_ms - E``, rolling median) and
published as a metric. It never adjusts a canonical timestamp. A recorder that
silently corrected the exchange's clock would be publishing its own opinion as
market data.

**Ordering.** ``bookTicker`` is ordered by the exchange's update id ``u``, and a
frame whose ``u`` is not greater than the last one kept is a duplicate or an out
-of-order delivery: it is counted and dropped, which is what section 4.1 says.
Klines are not dropped on arrival — a partial frame and a closed frame for the
same minute are different observations, and the normalizer, not this module,
decides which one describes a finished minute.

**Back-pressure is deliberate.** ``on_event`` is a plain synchronous callable
and the read loop awaits nothing between receiving a frame and delivering it.
Buffering frames in an unbounded queue would turn a slow disk into silent memory
growth and then into lost observations; delivering them synchronously means a
slow sink slows the socket, the exchange's own buffer fills, and the resulting
disconnect is visible in ``reconnects`` rather than invisible in RAM.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

from websockets.asyncio.client import connect as websocket_connect
from websockets.exceptions import WebSocketException

from chimera.recorder.contract import RecorderContract
from chimera.recorder.events import (
    NS_PER_MILLISECOND,
    SPOT_BOOK_TICKER,
    SPOT_KLINE_1M,
    UM_BOOK_TICKER,
    UM_KLINE_1M,
    UM_MARK_PRICE,
    BookTickerEvent,
    EventSource,
    KlineEvent,
    MarkPriceEvent,
    RawEvent,
    RecorderEventError,
)

logger = logging.getLogger(__name__)

#: The public market-data websocket bases, exactly as section 4.1 names them.
#: Both are unauthenticated: no key is sent, and neither host has a signed
#: endpoint this module could reach even if one existed.
UM_WS_BASE = "wss://fstream.binance.com/ws"
SPOT_WS_BASE = "wss://stream.binance.com:9443/ws"

#: The venue's own stream-name suffixes. A recorder stream id (``um.kline_1m``)
#: is this repository's name for a thing; these are Binance's, and the two are
#: kept apart on purpose so that a rename on either side is a visible edit.
KLINE_1M_SUFFIX = "kline_1m"
MARK_PRICE_SUFFIX = "markPrice@1s"
BOOK_TICKER_SUFFIX = "bookTicker"

#: The ``e`` field Binance puts on a raw frame, mapped to what it is. Spot's
#: ``bookTicker`` is the documented exception: it carries no ``e`` at all, which
#: is why :func:`frame_kind` falls back to the field shape for it.
EVENT_TYPE_KLINE = "kline"
EVENT_TYPE_MARK_PRICE = "markPriceUpdate"
EVENT_TYPE_BOOK_TICKER = "bookTicker"

#: The subscribe request Binance's raw-stream endpoint expects.
SUBSCRIBE_METHOD = "SUBSCRIBE"

#: Reconnect policy, section 4.2. The exchange closes a connection after 24
#: hours, so the client closes it first, at 23 h 50 m, and reconnects on its own
#: terms rather than in the middle of a frame.
DEFAULT_BACKOFF_INITIAL_S = 1.0
DEFAULT_BACKOFF_MAX_S = 60.0
DEFAULT_BACKOFF_FACTOR = 2.0
PROACTIVE_RECONNECT_S = 23 * 3600 + 50 * 60

#: How many skew samples the rolling median is taken over. Bounded so that a
#: process running for months does not accumulate one sample per frame.
SKEW_WINDOW = 512


class RecorderStreamError(RuntimeError):
    """A stream cannot be constructed, or a frame cannot be honestly recorded."""


class StreamKind(str, Enum):
    """Which parser a frame belongs to. A bounded label, not a free string."""

    KLINE = "KLINE"
    MARK_PRICE = "MARK_PRICE"
    BOOK_TICKER = "BOOK_TICKER"


@dataclass(frozen=True)
class Subscription:
    """One venue stream, and what this repository calls the thing it carries.

    Section 12.3 sketches ``StreamClient(url, stream_names, on_event, backoff)``.
    A bare name cannot say which parser to use or which recorder stream id to
    record under — ``btcusdt@kline_1m`` is ``um.kline_1m`` on one host and
    ``spot.kline_1m`` on the other — so the client takes these instead and
    exposes :attr:`StreamClient.stream_names` for the name list the sketch meant.
    """

    stream_id: str
    venue_name: str
    kind: StreamKind

    def __post_init__(self) -> None:
        if not self.stream_id or "." not in self.stream_id:
            raise RecorderStreamError(
                f"stream_id must be a <market>.<stream> id, got {self.stream_id!r}"
            )
        if not self.venue_name or self.venue_name != self.venue_name.strip():
            raise RecorderStreamError(f"venue_name {self.venue_name!r} is not a stream name")
        if not isinstance(self.kind, StreamKind):
            raise RecorderStreamError(f"kind must be a StreamKind, got {self.kind!r}")


#: Recorder stream id -> (venue suffix, parser kind). The four websocket streams
#: of the gen3 contract; ``um.funding`` is absent because funding is published
#: over REST and not pushed, which is :mod:`chimera.recorder.rest`'s job.
WEBSOCKET_STREAMS: Mapping[str, tuple[str, StreamKind]] = {
    UM_KLINE_1M: (KLINE_1M_SUFFIX, StreamKind.KLINE),
    UM_MARK_PRICE: (MARK_PRICE_SUFFIX, StreamKind.MARK_PRICE),
    UM_BOOK_TICKER: (BOOK_TICKER_SUFFIX, StreamKind.BOOK_TICKER),
    SPOT_KLINE_1M: (KLINE_1M_SUFFIX, StreamKind.KLINE),
    SPOT_BOOK_TICKER: (BOOK_TICKER_SUFFIX, StreamKind.BOOK_TICKER),
}


def venue_stream_name(symbol: str, suffix: str) -> str:
    """``btcusdt@kline_1m``: the venue's lowercase ``<symbol>@<suffix>`` form."""
    if not symbol or not symbol.strip():
        raise RecorderStreamError("a venue stream name needs a symbol")
    return f"{symbol.strip().lower()}@{suffix}"


def subscriptions_for(contract: RecorderContract, market: str) -> tuple[Subscription, ...]:
    """The websocket subscriptions one market of ``contract`` asks for.

    Derived from the contract rather than from a list kept beside it: a stream
    the contract does not declare is not subscribed to, and a stream it declares
    that has no websocket source — ``um.funding`` — is left to the REST poller
    instead of being silently dropped.
    """
    symbol = contract.market(market).symbol
    subscriptions: list[Subscription] = []
    for stream_id in contract.streams_for(market):
        entry = WEBSOCKET_STREAMS.get(stream_id)
        if entry is None:
            continue
        suffix, kind = entry
        subscriptions.append(
            Subscription(
                stream_id=stream_id,
                venue_name=venue_stream_name(symbol, suffix),
                kind=kind,
            )
        )
    return tuple(subscriptions)


def frame_kind(frame: Mapping[str, Any]) -> StreamKind | None:
    """What kind of market-data frame this is, or ``None`` if it is not one.

    Binance's raw-stream frames carry an event type in ``e`` — except spot's
    ``bookTicker``, which is documented as carrying an update id and the four
    book fields and nothing else. So the event type is used where there is one
    and the field shape is used where there is not, and a subscribe
    acknowledgement (``{"result": null, "id": 1}``) matches neither and is not
    market data.
    """
    event_type = frame.get("e")
    if isinstance(event_type, str):
        if event_type == EVENT_TYPE_KLINE:
            return StreamKind.KLINE
        if event_type == EVENT_TYPE_MARK_PRICE:
            return StreamKind.MARK_PRICE
        if event_type == EVENT_TYPE_BOOK_TICKER:
            return StreamKind.BOOK_TICKER
        return None
    if {"u", "b", "B", "a", "A"} <= set(frame):
        return StreamKind.BOOK_TICKER
    return None


@dataclass
class Backoff:
    """Exponential backoff from 1 s to 60 s, with jitter, per section 4.2.

    The jitter is bounded above by the delay itself and is there so that two
    clients that lost the same connection do not retry in lockstep. It never
    lengthens a delay past :attr:`maximum`.
    """

    initial: float = DEFAULT_BACKOFF_INITIAL_S
    maximum: float = DEFAULT_BACKOFF_MAX_S
    factor: float = DEFAULT_BACKOFF_FACTOR
    jitter: float = 0.1
    _attempt: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.initial <= 0 or self.maximum < self.initial or self.factor < 1:
            raise RecorderStreamError(
                f"backoff must grow from a positive initial delay to a larger maximum; got "
                f"initial={self.initial}, maximum={self.maximum}, factor={self.factor}"
            )
        if not 0 <= self.jitter < 1:
            raise RecorderStreamError(f"jitter must be in [0, 1), got {self.jitter}")

    @property
    def attempt(self) -> int:
        """Consecutive failures since the last :meth:`reset`."""
        return self._attempt

    def reset(self) -> None:
        """Called when a connection succeeds: the next failure waits 1 s again."""
        self._attempt = 0

    def next_delay(self, *, rand: Callable[[], float] = random.random) -> float:
        """The delay before the next attempt, and advance the sequence."""
        delay = min(self.maximum, self.initial * (self.factor**self._attempt))
        self._attempt += 1
        if self.jitter:
            delay = min(self.maximum, delay * (1.0 + self.jitter * rand()))
        return delay


@dataclass
class StreamCounters:
    """What one client has seen. Read by health; never read by a decision."""

    frames: int = 0
    events: int = 0
    reconnects: int = 0
    decode_errors: int = 0
    out_of_order: int = 0
    ignored_frames: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "frames": self.frames,
            "events": self.events,
            "reconnects": self.reconnects,
            "decode_errors": self.decode_errors,
            "out_of_order": self.out_of_order,
            "ignored_frames": self.ignored_frames,
        }


class SkewMeter:
    """A bounded rolling median of ``receipt_wall_ms - E``.

    Reported, never applied. The median rather than the mean because one frame
    delivered after a garbage collection pause should not move the number a
    human is asked to read.
    """

    def __init__(self, window: int = SKEW_WINDOW) -> None:
        if window < 1:
            raise RecorderStreamError(f"skew window must be >= 1, got {window}")
        self._samples: deque[float] = deque(maxlen=window)

    def observe(self, skew_ms: float) -> None:
        self._samples.append(float(skew_ms))

    @property
    def samples(self) -> int:
        return len(self._samples)

    def median_ms(self) -> float | None:
        """The rolling median, or ``None`` while nothing has been measured."""
        if not self._samples:
            return None
        ordered = sorted(self._samples)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[middle]
        return (ordered[middle - 1] + ordered[middle]) / 2.0


class StreamClient:
    """One websocket connection to one public market-data host.

    ``run(stop)`` is the whole lifecycle: connect, subscribe, read until the
    connection ends or ``stop`` is set, then back off and do it again. It
    returns only when ``stop`` is set or an error it must not swallow is raised.
    """

    def __init__(
        self,
        url: str,
        subscriptions: Sequence[Subscription],
        on_event: Callable[[RawEvent], None],
        *,
        backoff: Backoff | None = None,
        connect: Callable[..., Any] | None = None,
        wall_ns: Callable[[], int] = time.time_ns,
        mono_ns: Callable[[], int] = time.monotonic_ns,
        session_seconds: float = PROACTIVE_RECONNECT_S,
        name: str = "stream",
        open_timeout: float = 15.0,
        ping_interval: float = 20.0,
        ping_timeout: float = 20.0,
    ) -> None:
        if not url:
            raise RecorderStreamError("a stream client needs a url")
        if not subscriptions:
            raise RecorderStreamError(
                f"{name} was given no subscriptions; a client with nothing to subscribe to "
                "would connect, sit silent and look healthy"
            )
        seen: set[str] = set()
        for subscription in subscriptions:
            if subscription.stream_id in seen:
                raise RecorderStreamError(
                    f"{name} subscribes to {subscription.stream_id!r} twice"
                )
            seen.add(subscription.stream_id)
        if session_seconds <= 0:
            raise RecorderStreamError(
                f"session_seconds must be positive, got {session_seconds}"
            )
        self.url = url
        self.name = name
        self.subscriptions = tuple(subscriptions)
        self._by_kind: dict[StreamKind, list[Subscription]] = {}
        for subscription in self.subscriptions:
            self._by_kind.setdefault(subscription.kind, []).append(subscription)
        self._on_event = on_event
        self.backoff = backoff or Backoff()
        self._connect = connect or websocket_connect
        self._wall_ns = wall_ns
        self._mono_ns = mono_ns
        self._session_seconds = float(session_seconds)
        self._open_timeout = open_timeout
        self._ping_interval = ping_interval
        self._ping_timeout = ping_timeout
        self.counters = StreamCounters()
        self.skew = SkewMeter()
        self._last_update_id: dict[str, int] = {}
        self._last_event_ns: dict[str, int] = {}
        self._connected = False
        self._subscribe_id = 0

    # --- observable state -------------------------------------------------
    @property
    def stream_names(self) -> tuple[str, ...]:
        """The venue stream names this client subscribes to."""
        return tuple(subscription.venue_name for subscription in self.subscriptions)

    @property
    def stream_ids(self) -> tuple[str, ...]:
        """The recorder stream ids this client produces events for."""
        return tuple(subscription.stream_id for subscription in self.subscriptions)

    @property
    def connected(self) -> bool:
        """Whether a connection is open right now."""
        return self._connected

    def last_event_ns(self, stream_id: str) -> int | None:
        """Canonical time of the last event delivered for one stream."""
        return self._last_event_ns.get(stream_id)

    def subscribe_payload(self) -> dict[str, Any]:
        """The exact SUBSCRIBE message sent after every connection."""
        self._subscribe_id += 1
        return {
            "method": SUBSCRIBE_METHOD,
            "params": list(self.stream_names),
            "id": self._subscribe_id,
        }

    # --- the loop ---------------------------------------------------------
    async def run(self, stop: asyncio.Event) -> None:
        """Connect, subscribe and read until ``stop`` is set.

        Every failure that is a property of a network — a refused connection, a
        closed socket, a handshake that timed out — is a reconnect, counted and
        logged. Everything else propagates: a bug in a parser must not be
        retried forever behind a backoff while the recorder reports itself up.
        """
        first = True
        while not stop.is_set():
            if not first:
                delay = self.backoff.next_delay()
                logger.info("%s reconnecting in %.1fs", self.name, delay)
                if await _sleep_or_stop(stop, delay):
                    return
                if stop.is_set():
                    return
            first = False
            try:
                await self._session(stop)
            except asyncio.CancelledError:
                raise
            except (OSError, WebSocketException, asyncio.TimeoutError) as exc:
                self.counters.reconnects += 1
                self._connected = False
                logger.warning("%s connection ended: %s", self.name, exc)
                continue
            if stop.is_set():
                return
            # A clean end of session — the proactive close, or the peer going
            # away without an error — is still a reconnect for counting.
            self.counters.reconnects += 1

    async def _session(self, stop: asyncio.Event) -> None:
        """One connection, from handshake to close."""
        async with self._connect(
            self.url,
            open_timeout=self._open_timeout,
            ping_interval=self._ping_interval,
            ping_timeout=self._ping_timeout,
        ) as socket:
            self._connected = True
            self.backoff.reset()
            deadline = self._mono_ns() + int(self._session_seconds * 1e9)
            try:
                await socket.send(json.dumps(self.subscribe_payload()))
                logger.info("%s subscribed to %s", self.name, list(self.stream_names))
                await self._read(socket, stop, deadline)
            finally:
                self._connected = False

    async def _read(self, socket: Any, stop: asyncio.Event, deadline: int) -> None:
        """Read frames until the session deadline, ``stop``, or the peer."""
        stop_task = asyncio.ensure_future(stop.wait())
        try:
            while not stop.is_set():
                remaining = (deadline - self._mono_ns()) / 1e9
                if remaining <= 0:
                    logger.info("%s closing before the exchange's 24h limit", self.name)
                    return
                receive = asyncio.ensure_future(_recv(socket))
                done, _ = await asyncio.wait(
                    {receive, stop_task},
                    timeout=remaining,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if receive not in done:
                    receive.cancel()
                    # Either ``stop`` fired or the session deadline expired; the
                    # loop head decides which, and both end the session.
                    if stop.is_set():
                        return
                    continue
                message = receive.result()
                if message is None:
                    return
                self._handle(message)
        finally:
            stop_task.cancel()

    # --- one frame --------------------------------------------------------
    def _handle(self, message: str | bytes) -> None:
        """Decode one frame and deliver whatever honest record it produces."""
        self.counters.frames += 1
        receipt_wall_ns = self._wall_ns()
        receipt_mono_ns = self._mono_ns()
        try:
            decoded = json.loads(message)
        except (TypeError, ValueError) as exc:
            self.counters.decode_errors += 1
            logger.warning("%s dropped a frame that is not JSON: %s", self.name, exc)
            return
        if not isinstance(decoded, Mapping):
            self.counters.decode_errors += 1
            logger.warning("%s dropped a frame that is not an object", self.name)
            return
        # A combined-stream endpoint wraps the payload; the raw endpoint this
        # client uses does not. Unwrapping when the wrapper is there costs one
        # branch and makes the client correct against both.
        payload = decoded.get("data") if isinstance(decoded.get("data"), Mapping) else decoded
        kind = frame_kind(payload)
        if kind is None:
            self.counters.ignored_frames += 1
            return
        subscription = self._subscription_for(kind, payload)
        if subscription is None:
            self.counters.ignored_frames += 1
            logger.warning("%s received a %s frame it did not subscribe to", self.name, kind)
            return
        try:
            event = self._to_event(
                subscription,
                payload,
                receipt_wall_ns=receipt_wall_ns,
                receipt_mono_ns=receipt_mono_ns,
            )
        except RecorderEventError as exc:
            self.counters.decode_errors += 1
            logger.warning("%s dropped a malformed %s frame: %s", self.name, kind, exc)
            return
        if event is None:
            return
        self.counters.events += 1
        self._last_event_ns[subscription.stream_id] = event.canonical_ns
        self._on_event(event)

    def _subscription_for(
        self, kind: StreamKind, payload: Mapping[str, Any]
    ) -> Subscription | None:
        """Which subscription a frame belongs to.

        One client talks to one host and one symbol, so a kind identifies a
        subscription uniquely here. The symbol is still checked where the frame
        carries one, because a frame for another instrument arriving on this
        connection is a fact worth refusing rather than recording.
        """
        candidates = self._by_kind.get(kind)
        if not candidates:
            return None
        subscription = candidates[0]
        symbol = payload.get("s")
        if isinstance(symbol, str) and symbol:
            expected = subscription.venue_name.split("@", 1)[0].upper()
            if symbol.upper() != expected:
                return None
        return subscription

    def _to_event(
        self,
        subscription: Subscription,
        payload: Mapping[str, Any],
        *,
        receipt_wall_ns: int,
        receipt_mono_ns: int,
    ) -> RawEvent | None:
        """Parse one frame, or say why it is not recorded.

        Returns ``None`` for an observation that section 4.1 says is counted
        rather than stored — a book update whose ``u`` is not greater than the
        last one kept.
        """
        stream = subscription.stream_id
        if subscription.kind is StreamKind.KLINE:
            parsed = KlineEvent.from_ws(payload, stream=stream)
            self._observe_skew(parsed.event_ms, receipt_wall_ns)
            return parsed.to_raw_event(
                payload,
                receipt_wall_ns=receipt_wall_ns,
                receipt_mono_ns=receipt_mono_ns,
                source=EventSource.WEBSOCKET,
            )
        if subscription.kind is StreamKind.MARK_PRICE:
            mark = MarkPriceEvent.from_ws(payload, stream=stream)
            self._observe_skew(mark.event_ms, receipt_wall_ns)
            return mark.to_raw_event(
                payload,
                receipt_wall_ns=receipt_wall_ns,
                receipt_mono_ns=receipt_mono_ns,
            )
        book = BookTickerEvent.from_ws(payload, stream=stream)
        last = self._last_update_id.get(stream)
        if last is not None and book.update_id <= last:
            self.counters.out_of_order += 1
            return None
        self._last_update_id[stream] = book.update_id
        self._observe_skew(book.event_ms, receipt_wall_ns)
        return book.to_raw_event(
            payload,
            receipt_wall_ns=receipt_wall_ns,
            receipt_mono_ns=receipt_mono_ns,
        )

    def _observe_skew(self, event_ms: int | None, receipt_wall_ns: int) -> None:
        """Measure ``receipt_wall_ms - E`` where the exchange published an ``E``."""
        if event_ms is None:
            return
        self.skew.observe(receipt_wall_ns / NS_PER_MILLISECOND - event_ms)


async def _recv(socket: Any) -> str | bytes | None:
    """One message, or ``None`` when the peer has gone away."""
    try:
        return await socket.recv()
    except WebSocketException:
        return None


async def _sleep_or_stop(stop: asyncio.Event, delay: float) -> bool:
    """Wait ``delay`` seconds, or return early. ``True`` when ``stop`` fired.

    A backoff implemented with a bare ``asyncio.sleep`` makes a service that
    takes up to a minute to shut down, which in practice means a service that
    gets killed instead of stopped.
    """
    try:
        await asyncio.wait_for(stop.wait(), timeout=delay)
    except asyncio.TimeoutError:
        return False
    return True


def clients_for(
    contract: RecorderContract,
    on_event: Callable[[RawEvent], None],
    *,
    um_url: str = UM_WS_BASE,
    spot_url: str = SPOT_WS_BASE,
    **options: Any,
) -> tuple[StreamClient, ...]:
    """One client per market the contract declares websocket streams for."""
    urls = {"um": um_url, "spot": spot_url}
    clients: list[StreamClient] = []
    for market in contract.market_keys():
        subscriptions = subscriptions_for(contract, market)
        if not subscriptions:
            continue
        url = urls.get(market)
        if url is None:
            raise RecorderStreamError(
                f"contract {contract.label} declares websocket streams for market {market!r} "
                f"and this build knows a websocket host for {sorted(urls)} only"
            )
        clients.append(
            StreamClient(url, subscriptions, on_event, name=f"{market}-ws", **options)
        )
    return tuple(clients)
