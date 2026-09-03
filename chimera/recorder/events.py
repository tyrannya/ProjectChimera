"""What the recorder stores, one observation at a time.

Two kinds of thing live here, and keeping them apart is the point of the module.

:class:`RawEvent` is the **generic** record: a stream id, an integer canonical
timestamp, the local receipt timestamps, where the observation came from, a
deduplication key, and an opaque payload. It knows nothing about Binance, about
websockets or about HTTP, and it must stay that way — the network clients that
produce these events are a later package, and no response object, socket or
session may ever reach the stored form.

The parsers below (:class:`KlineEvent`, :class:`MarkPriceEvent`,
:class:`BookTickerEvent`, :class:`FundingSettlement`) are the **specific** half:
pure functions from an already-decoded JSON mapping — or from a REST row that
has already been decoded into one — to a typed record and to a
:class:`RawEvent`. They take the payload; they never fetch it. A parser is
handed the receipt timestamps rather than reading a clock, so the whole module
is a pure function of its inputs and a test can pin every byte it produces.

**Canonical time versus receipt time.** These are different facts and the record
never conflates them:

``canonical_ns``
    when the *exchange* says the observation happened, in integer nanoseconds
    since the UTC epoch. It is what every day boundary, minute key and ordering
    in this package is computed from.

``receipt_wall_ns`` / ``receipt_mono_ns``
    when *this host* saw it: the wall clock, which can jump, and the monotonic
    clock, which cannot and is therefore the ordering of last resort. Neither
    ever alters a canonical timestamp.

``time_basis``
    which of the two ``canonical_ns`` actually came from.
    :attr:`TimeBasis.EXCHANGE` for every stream that publishes an event time.
    :attr:`TimeBasis.RECEIPT` for the one that does not — Binance's spot
    ``bookTicker`` carries an update id and no timestamp — where the local
    receipt time is used for minute assignment. Without this field a local clock
    reading would sit in ``canonical_ns`` indistinguishable from an exchange
    one, and the whole distinction above would be a comment rather than a
    property of the data.

**Deduplication is derivable, never remembered.** :attr:`RawEvent.dedup_key` is
a pure function of the stream and the payload, so two processes, or the same
process after a restart, agree about what "the same observation" means without
sharing any state.

The key identifies an *observation*, not the thing observed, and the difference
is the whole design. For a kline and for a funding settlement it is the
exchange's own key — the minute, or the settlement instant — **plus a digest of
the payload**, so a re-delivered identical frame is a duplicate while a
genuinely different reading of the same minute or the same settlement is not.
That is what keeps a partial kline frame from being eaten by the closed one for
the same minute, keeps a websocket kline and a REST gap-fill of the same minute
both on disk where the reconciliation can compare them, and keeps two
disagreeing readings of one settlement both on disk instead of silently
resolving them in favour of whichever arrived first. Identifying the *settlement*
rather than the observation is the normalizer's job, and it refuses a
disagreement rather than choosing.

``bookTicker`` is the exception and is keyed on the exchange's update id ``u``
alone, because ``u`` is a monotonic sequence the exchange guarantees and the
recorder's rule for that stream is stated in terms of it: an update whose ``u``
is not greater than the last kept is a repeat or an out-of-order frame, and is
counted rather than stored.

This module opens no socket, makes no request and reads no clock.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

#: Names the shape of one line in a raw event file. Bumped when the on-disk
#: record changes in a way an older reader would misinterpret; a line from a
#: schema this build does not know is refused, not best-effort parsed.
RAW_EVENT_SCHEMA = "chimera.recorder-raw-event/1"

NS_PER_MICROSECOND = 1_000
NS_PER_MILLISECOND = 1_000_000
NS_PER_SECOND = 1_000_000_000
NS_PER_MINUTE = 60 * NS_PER_SECOND
NS_PER_DAY = 24 * 60 * NS_PER_MINUTE
MS_PER_MINUTE = 60_000

#: Minutes in a UTC day. Constant by construction: UTC has no daylight saving,
#: and POSIX time has no leap seconds, so a day is 1440 minutes for ever.
MINUTES_PER_DAY = 1440

#: The widest integer nanosecond instant this package accepts, and the reason
#: for the bound: a payload carrying seconds where milliseconds were expected,
#: or nanoseconds where milliseconds were expected, produces an instant
#: thousands of years away from now, and it is far better to refuse it than to
#: create a directory named ``57892-11-05``.
MIN_CANONICAL_NS = 0
MAX_CANONICAL_NS = 4_102_444_800 * NS_PER_SECOND  # 2100-01-01T00:00:00Z

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)

#: Binance USD-M perpetual streams, as the gen3 contract names them.
UM_KLINE_1M = "um.kline_1m"
UM_MARK_PRICE = "um.markPrice"
UM_BOOK_TICKER = "um.bookTicker"
UM_FUNDING = "um.funding"

#: Binance spot streams.
SPOT_KLINE_1M = "spot.kline_1m"
SPOT_BOOK_TICKER = "spot.bookTicker"

KLINE_STREAMS = frozenset({UM_KLINE_1M, SPOT_KLINE_1M})
BOOK_TICKER_STREAMS = frozenset({UM_BOOK_TICKER, SPOT_BOOK_TICKER})


class RecorderEventError(ValueError):
    """An observation cannot be turned into an honest record."""


class TimeBasis(str, Enum):
    """Where :attr:`RawEvent.canonical_ns` came from. A bounded, persisted label."""

    #: The exchange published an event time and it is what was stored.
    EXCHANGE = "EXCHANGE"
    #: The payload carries no exchange time, so the local receipt wall clock is
    #: used for minute assignment. Binance's spot ``bookTicker`` is the only
    #: stream in the gen3 contract that is in this state.
    RECEIPT = "RECEIPT"


class EventSource(str, Enum):
    """How the observation reached this host. A bounded, persisted label."""

    #: A push frame from a market-data websocket.
    WEBSOCKET = "WEBSOCKET"
    #: A REST row fetched to fill a gap a disconnect may have caused.
    REST_GAPFILL = "REST_GAPFILL"
    #: A REST row fetched on a schedule, which is how funding settlements and
    #: the premium index are published rather than pushed.
    REST_POLL = "REST_POLL"


# --- time -------------------------------------------------------------------
def require_canonical_ns(value: Any, *, field: str = "canonical_ns") -> int:
    """An integer UTC nanosecond instant, or an explanation of why it is not one.

    ``bool`` is rejected explicitly because it is an ``int`` in Python and
    ``True`` would otherwise be the instant one nanosecond after the epoch.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise RecorderEventError(
            f"{field} must be an integer nanosecond instant, got "
            f"{type(value).__name__} {value!r}"
        )
    if not MIN_CANONICAL_NS <= value <= MAX_CANONICAL_NS:
        raise RecorderEventError(
            f"{field} {value} is outside [{MIN_CANONICAL_NS}, {MAX_CANONICAL_NS}]. An "
            "instant that far from now is a unit mistake — seconds or milliseconds where "
            "nanoseconds were expected — and guessing the unit is how two files end up "
            "describing different centuries"
        )
    return value


def utc_day(canonical_ns: int) -> str:
    """The UTC day an instant belongs to, as ``YYYY-MM-DD``.

    Integer arithmetic on nanoseconds, then a date. No locale, no timezone
    database, no floating point, and the same answer on every platform.
    """
    day_number = require_canonical_ns(canonical_ns) // NS_PER_DAY
    return (_EPOCH + timedelta(days=day_number)).strftime("%Y-%m-%d")


def day_start_ns(day: str) -> int:
    """The first instant of a ``YYYY-MM-DD`` UTC day, in nanoseconds."""
    try:
        parsed = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise RecorderEventError(f"{day!r} is not a YYYY-MM-DD UTC day: {exc}") from exc
    return int((parsed - _EPOCH).total_seconds()) * NS_PER_SECOND


def minute_open_ms(canonical_ns: int) -> int:
    """The minute an instant falls in, keyed by that minute's open in UTC ms.

    Half-open by construction: the minute ``[t, t + 60000)`` owns ``t`` and does
    not own ``t + 60000``. An event landing exactly on a boundary belongs to the
    minute that is opening, never to the one that just closed.
    """
    milliseconds = require_canonical_ns(canonical_ns) // NS_PER_MILLISECOND
    return (milliseconds // MS_PER_MINUTE) * MS_PER_MINUTE


def iso_utc(canonical_ns: int) -> str:
    """An instant as a UTC ISO-8601 string, truncated to microseconds.

    For manifests and logs. The authoritative value is always the integer
    nanoseconds recorded beside it; this is the readable form, and the
    truncation is named here so nobody mistakes it for the record.
    """
    instant = _EPOCH + timedelta(microseconds=require_canonical_ns(canonical_ns) // 1_000)
    return instant.isoformat()


# --- the generic record -----------------------------------------------------
def canonical_json(payload: Any) -> str:
    """The one serialization this package uses: sorted, tight, UTF-8, no NaN.

    ``allow_nan=False`` matters. Python's default emits bare ``NaN`` and
    ``Infinity``, which are not JSON, so a payload carrying one would be written
    to disk as a line no conforming reader can parse. Refusing it at the point of
    construction turns a corrupt file into an explicit failure.
    """
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise RecorderEventError(
            f"payload cannot be serialised deterministically: {exc}. A raw record holds "
            "decoded JSON and nothing else — no response object, no bytes, no NaN"
        ) from exc


def payload_digest(payload: Mapping[str, Any]) -> str:
    """SHA-256 over :func:`canonical_json` of a payload, as 64 hex digits."""
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RawEvent:
    """One observation, exactly as it will be written to the raw file.

    Frozen, validated at construction, and serialisable to a single canonical
    line. Everything that could vary between two hosts observing the same frame
    — dictionary order, float formatting, text encoding, path separators — is
    normalised away, so byte equality of two records means the two observations
    were the same.
    """

    stream: str
    canonical_ns: int
    time_basis: TimeBasis
    receipt_wall_ns: int
    receipt_mono_ns: int
    source: EventSource
    dedup_key: str
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.stream, str) or "." not in self.stream:
            raise RecorderEventError(
                f"stream must be a <market>.<stream> id, got {self.stream!r}"
            )
        if not isinstance(self.time_basis, TimeBasis):
            raise RecorderEventError(
                f"time_basis must be a TimeBasis, got {self.time_basis!r}"
            )
        if not isinstance(self.source, EventSource):
            raise RecorderEventError(f"source must be an EventSource, got {self.source!r}")
        require_canonical_ns(self.canonical_ns)
        require_canonical_ns(self.receipt_wall_ns, field="receipt_wall_ns")
        if isinstance(self.receipt_mono_ns, bool) or not isinstance(self.receipt_mono_ns, int):
            raise RecorderEventError(
                f"receipt_mono_ns must be an integer, got {self.receipt_mono_ns!r}"
            )
        if self.receipt_mono_ns < 0:
            raise RecorderEventError(f"receipt_mono_ns {self.receipt_mono_ns} is negative")
        if not isinstance(self.dedup_key, str) or not self.dedup_key:
            raise RecorderEventError("dedup_key must be a non-empty string")
        if len(self.dedup_key) > 256 or not self.dedup_key.isprintable():
            raise RecorderEventError(
                f"dedup_key {self.dedup_key!r} must be at most 256 printable characters; it "
                "is compared, logged and read back out of a one-line record"
            )
        if not isinstance(self.payload, Mapping):
            raise RecorderEventError(
                f"payload must be a mapping, got {type(self.payload).__name__}"
            )
        if self.time_basis is TimeBasis.RECEIPT and self.canonical_ns != self.receipt_wall_ns:
            raise RecorderEventError(
                "time_basis RECEIPT says canonical_ns is the local receipt wall clock, so "
                f"the two must be equal; got {self.canonical_ns} and {self.receipt_wall_ns}. "
                "A basis that does not describe the value beside it is worse than no basis"
            )
        # Serialising here is what makes "malformed payload fails explicitly"
        # true at construction rather than at the moment of the write, when the
        # file is already open and a stream is already running.
        canonical_json(dict(self.payload))
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))

    @property
    def day(self) -> str:
        """The UTC day this event belongs to, by canonical time."""
        return utc_day(self.canonical_ns)

    @property
    def minute_open_ms(self) -> int:
        """The minute key this event falls in."""
        return minute_open_ms(self.canonical_ns)

    def to_record(self) -> dict[str, Any]:
        """The mapping one raw line holds."""
        return {
            "schema": RAW_EVENT_SCHEMA,
            "stream": self.stream,
            "canonical_ns": self.canonical_ns,
            "time_basis": self.time_basis.value,
            "receipt_wall_ns": self.receipt_wall_ns,
            "receipt_mono_ns": self.receipt_mono_ns,
            "source": self.source.value,
            "dedup_key": self.dedup_key,
            "payload": dict(self.payload),
        }

    def canonical_line(self) -> bytes:
        """The exact bytes appended to the raw file, newline included.

        Always ``\\n``, never ``\\r\\n``: the file is opened in binary append
        mode precisely so that a record written on Windows and one written on
        Linux are the same bytes.
        """
        return canonical_json(self.to_record()).encode("utf-8") + b"\n"

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "RawEvent":
        """Rebuild an event from a parsed raw line, refusing anything unclear."""
        if not isinstance(record, Mapping):
            raise RecorderEventError("a raw record must be a JSON object")
        schema = record.get("schema")
        if schema != RAW_EVENT_SCHEMA:
            raise RecorderEventError(
                f"raw record schema is {schema!r}, not {RAW_EVENT_SCHEMA!r}"
            )
        expected = {
            "schema",
            "stream",
            "canonical_ns",
            "time_basis",
            "receipt_wall_ns",
            "receipt_mono_ns",
            "source",
            "dedup_key",
            "payload",
        }
        unknown = sorted(set(record) - expected)
        missing = sorted(expected - set(record))
        if unknown or missing:
            raise RecorderEventError(
                f"a raw record carries exactly {sorted(expected)}: unexpected {unknown}, "
                f"missing {missing}"
            )
        try:
            time_basis = TimeBasis(record["time_basis"])
            source = EventSource(record["source"])
        except ValueError as exc:
            raise RecorderEventError(f"raw record carries an unknown label: {exc}") from exc
        return cls(
            stream=record["stream"],
            canonical_ns=record["canonical_ns"],
            time_basis=time_basis,
            receipt_wall_ns=record["receipt_wall_ns"],
            receipt_mono_ns=record["receipt_mono_ns"],
            source=source,
            dedup_key=record["dedup_key"],
            payload=record["payload"],
        )

    @classmethod
    def from_line(cls, line: bytes | str) -> "RawEvent":
        """Rebuild an event from one raw line. A malformed line raises."""
        text = line.decode("utf-8") if isinstance(line, bytes) else line
        try:
            record = json.loads(text)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RecorderEventError(f"raw line is not readable JSON: {exc}") from exc
        return cls.from_record(record)


# --- payload readers --------------------------------------------------------
def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RecorderEventError(f"{where} must be an object, got {type(value).__name__}")
    return value


def _int_field(payload: Mapping[str, Any], key: str, where: str) -> int:
    if key not in payload:
        raise RecorderEventError(f"{where} is missing the integer field {key!r}")
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise RecorderEventError(
            f"{where}.{key} must be an integer, got {type(value).__name__} {value!r}"
        )
    return value


def _decimal_field(payload: Mapping[str, Any], key: str, where: str) -> str:
    """One published decimal quantity, kept as the exchange's own string.

    Binance publishes prices and sizes as decimal strings. They are stored
    verbatim: the raw layer is what the exchange said, and any conversion to a
    machine number is a decision the normalizer makes and records, not one the
    recorder makes silently on the way in.
    """
    if key not in payload:
        raise RecorderEventError(f"{where} is missing the decimal field {key!r}")
    value = payload[key]
    if not isinstance(value, str) or not value.strip():
        raise RecorderEventError(
            f"{where}.{key} must be a non-empty decimal string as the exchange publishes "
            f"it, got {type(value).__name__} {value!r}"
        )
    return value.strip()


def _bool_field(payload: Mapping[str, Any], key: str, where: str) -> bool:
    if key not in payload:
        raise RecorderEventError(f"{where} is missing the boolean field {key!r}")
    value = payload[key]
    if not isinstance(value, bool):
        raise RecorderEventError(f"{where}.{key} must be a boolean, got {value!r}")
    return value


def _optional_text(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise RecorderEventError(f"{key} must be a non-empty string when present")
    return value.strip()


#: The Binance kline object's field names, in the order the REST row publishes
#: them. Identical to :data:`nn.p13_sources.KLINE_COLUMNS` read as a mapping
#: onto the websocket object, which is why a REST row can be adapted into the
#: websocket shape without dropping a published value.
REST_KLINE_FIELDS: tuple[str, ...] = (
    "t",  # open time
    "o",  # open
    "h",  # high
    "l",  # low
    "c",  # close
    "v",  # base volume
    "T",  # close time
    "q",  # quote volume
    "n",  # number of trades
    "V",  # taker buy base volume
    "Q",  # taker buy quote volume
    "B",  # the field Binance documents as "ignore"
)


@dataclass(frozen=True)
class KlineEvent:
    """One 1m candle frame, closed or still forming.

    ``closed`` is the whole of the difference between an observation the
    normalizer may use and one it may not: only a frame the exchange marked
    ``x == true`` describes a minute that has finished printing. Partial frames
    are kept in the raw file because they are what the exchange said, and are
    ignored by the normalizer because they are not a candle.
    """

    stream: str
    open_ms: int
    close_ms: int
    closed: bool
    open: str
    high: str
    low: str
    close: str
    volume: str
    trades: int
    taker_buy_base: str
    taker_buy_quote: str
    event_ms: int | None

    @property
    def canonical_ns(self) -> int:
        """The minute key, in nanoseconds: a candle is stamped by its open."""
        return self.open_ms * NS_PER_MILLISECOND

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any], *, stream: str) -> "KlineEvent":
        """Read a kline out of the stored payload shape.

        The shape is Binance's own websocket frame: a ``k`` object carrying the
        candle and, on a pushed frame, a top-level ``E`` event time. A REST row
        adapted by :meth:`from_rest` has the same ``k`` object and no ``E``.
        """
        frame = _mapping(payload, f"{stream} payload")
        candle = _mapping(frame.get("k"), f"{stream} payload.k")
        where = f"{stream} payload.k"
        event_ms = None if "E" not in frame else _int_field(frame, "E", f"{stream} payload")
        return cls(
            stream=stream,
            open_ms=_int_field(candle, "t", where),
            close_ms=_int_field(candle, "T", where),
            closed=_bool_field(candle, "x", where),
            open=_decimal_field(candle, "o", where),
            high=_decimal_field(candle, "h", where),
            low=_decimal_field(candle, "l", where),
            close=_decimal_field(candle, "c", where),
            volume=_decimal_field(candle, "v", where),
            trades=_int_field(candle, "n", where),
            taker_buy_base=_decimal_field(candle, "V", where),
            taker_buy_quote=_decimal_field(candle, "Q", where),
            event_ms=event_ms,
        )

    @classmethod
    def from_ws(cls, frame: Mapping[str, Any], *, stream: str) -> "KlineEvent":
        """Read a kline out of a decoded websocket frame. An alias for clarity."""
        return cls.from_payload(frame, stream=stream)

    @classmethod
    def rest_payload(cls, row: Sequence[Any]) -> dict[str, Any]:
        """Adapt one REST kline row into the stored payload shape.

        Binance's REST row is a twelve-element array whose columns map exactly
        onto the websocket candle object, so nothing published is dropped: the
        adapter renames, and ``x`` is set because a row the REST endpoint
        returns for a past minute is by definition closed.
        """
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
            raise RecorderEventError(
                f"a REST kline row must be a sequence of {len(REST_KLINE_FIELDS)} fields, "
                f"got {type(row).__name__}"
            )
        if len(row) != len(REST_KLINE_FIELDS):
            raise RecorderEventError(
                f"a REST kline row has {len(REST_KLINE_FIELDS)} fields "
                f"{list(REST_KLINE_FIELDS)}, got {len(row)}"
            )
        candle: dict[str, Any] = dict(zip(REST_KLINE_FIELDS, row))
        candle["x"] = True
        return {"k": candle}

    @classmethod
    def from_rest(cls, row: Sequence[Any], *, stream: str) -> "KlineEvent":
        """Read a kline out of one REST row."""
        return cls.from_payload(cls.rest_payload(row), stream=stream)

    def to_raw_event(
        self,
        payload: Mapping[str, Any],
        *,
        receipt_wall_ns: int,
        receipt_mono_ns: int,
        source: EventSource,
    ) -> RawEvent:
        """The raw record for this observation, storing ``payload`` verbatim."""
        return RawEvent(
            stream=self.stream,
            canonical_ns=self.canonical_ns,
            time_basis=TimeBasis.EXCHANGE,
            receipt_wall_ns=receipt_wall_ns,
            receipt_mono_ns=receipt_mono_ns,
            source=source,
            dedup_key=kline_dedup_key(self, payload),
            payload=payload,
        )


def kline_dedup_key(event: KlineEvent, payload: Mapping[str, Any]) -> str:
    """``kline:<open_ms>:<close_ms>:<C|O>:<digest16>``.

    The open and close times alone would make every partial frame of a minute a
    duplicate of the first one, and would make a websocket close and a REST
    gap-fill of the same minute indistinguishable. The closed flag separates the
    first case and the payload digest separates the second, which is what leaves
    both records on disk for the reconciliation to compare. The digest is the
    first sixteen hex digits of the payload's SHA-256: long enough that two
    different frames in one day colliding is not a thing that happens, short
    enough that a raw line stays readable.

    It is a function of the stream and the payload alone, so a restart, a second
    process or a re-read of the file all agree about what "the same observation"
    means without sharing any state.
    """
    flag = "C" if event.closed else "O"
    return f"kline:{event.open_ms}:{event.close_ms}:{flag}:{payload_digest(payload)[:16]}"


@dataclass(frozen=True)
class MarkPriceEvent:
    """One mark-price update: mark, index, estimated settlement and funding state.

    The index price and the per-minute mark high are derived from this stream
    rather than subscribed separately, which is why every field the exchange
    publishes on it is kept.
    """

    stream: str
    event_ms: int
    mark: str
    index: str
    estimated_settle: str | None
    funding_rate: str
    next_funding_ms: int

    @property
    def canonical_ns(self) -> int:
        return self.event_ms * NS_PER_MILLISECOND

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any], *, stream: str = UM_MARK_PRICE
    ) -> "MarkPriceEvent":
        frame = _mapping(payload, f"{stream} payload")
        where = f"{stream} payload"
        settle = frame.get("P")
        if settle is not None and (not isinstance(settle, str) or not settle.strip()):
            raise RecorderEventError(f"{where}.P must be a decimal string when present")
        return cls(
            stream=stream,
            event_ms=_int_field(frame, "E", where),
            mark=_decimal_field(frame, "p", where),
            index=_decimal_field(frame, "i", where),
            estimated_settle=None if settle is None else settle.strip(),
            funding_rate=_decimal_field(frame, "r", where),
            next_funding_ms=_int_field(frame, "T", where),
        )

    @classmethod
    def from_ws(
        cls, frame: Mapping[str, Any], *, stream: str = UM_MARK_PRICE
    ) -> "MarkPriceEvent":
        return cls.from_payload(frame, stream=stream)

    def to_raw_event(
        self,
        payload: Mapping[str, Any],
        *,
        receipt_wall_ns: int,
        receipt_mono_ns: int,
        source: EventSource = EventSource.WEBSOCKET,
    ) -> RawEvent:
        return RawEvent(
            stream=self.stream,
            canonical_ns=self.canonical_ns,
            time_basis=TimeBasis.EXCHANGE,
            receipt_wall_ns=receipt_wall_ns,
            receipt_mono_ns=receipt_mono_ns,
            source=source,
            dedup_key=f"mark:{self.event_ms}:{payload_digest(payload)[:16]}",
            payload=payload,
        )


@dataclass(frozen=True)
class BookTickerEvent:
    """One best bid / best ask update.

    ``event_ms`` is ``None`` on Binance spot, which publishes an update id and no
    timestamp. That absence is carried through to :class:`TimeBasis`: a spot
    book event is stamped with the local receipt wall clock and *says so*, and a
    perpetual book event is stamped with the exchange's ``E`` and says that.
    """

    stream: str
    update_id: int
    bid: str
    bid_qty: str
    ask: str
    ask_qty: str
    event_ms: int | None
    transaction_ms: int | None

    @property
    def has_exchange_time(self) -> bool:
        return self.event_ms is not None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any], *, stream: str) -> "BookTickerEvent":
        frame = _mapping(payload, f"{stream} payload")
        where = f"{stream} payload"
        return cls(
            stream=stream,
            update_id=_int_field(frame, "u", where),
            bid=_decimal_field(frame, "b", where),
            bid_qty=_decimal_field(frame, "B", where),
            ask=_decimal_field(frame, "a", where),
            ask_qty=_decimal_field(frame, "A", where),
            event_ms=None if "E" not in frame else _int_field(frame, "E", where),
            transaction_ms=None if "T" not in frame else _int_field(frame, "T", where),
        )

    @classmethod
    def from_ws(cls, frame: Mapping[str, Any], *, stream: str) -> "BookTickerEvent":
        return cls.from_payload(frame, stream=stream)

    def to_raw_event(
        self,
        payload: Mapping[str, Any],
        *,
        receipt_wall_ns: int,
        receipt_mono_ns: int,
        source: EventSource = EventSource.WEBSOCKET,
    ) -> RawEvent:
        """The raw record, stamped with exchange time where there is one.

        Where there is not, ``canonical_ns`` is the receipt wall clock and
        ``time_basis`` is :attr:`TimeBasis.RECEIPT`. Nothing downstream has to
        guess which happened.
        """
        if self.event_ms is None:
            canonical_ns = require_canonical_ns(receipt_wall_ns, field="receipt_wall_ns")
            time_basis = TimeBasis.RECEIPT
        else:
            canonical_ns = self.event_ms * NS_PER_MILLISECOND
            time_basis = TimeBasis.EXCHANGE
        return RawEvent(
            stream=self.stream,
            canonical_ns=canonical_ns,
            time_basis=time_basis,
            receipt_wall_ns=receipt_wall_ns,
            receipt_mono_ns=receipt_mono_ns,
            source=source,
            dedup_key=f"book:{self.update_id}",
            payload=payload,
        )


@dataclass(frozen=True)
class FundingSettlement:
    """One realised funding settlement, exactly as the exchange published it.

    :attr:`settlement_id` is ``fundingTime``: a settlement happens once, at a
    scheduled instant, and the normalized settlements file holds one record per
    instant. The *raw* deduplication key is deliberately not the same thing —
    see :meth:`to_raw_event` — because two readings of one settlement that
    disagree are a finding the reconciliation has to be able to see, and a key
    that collapsed them would delete the evidence of the disagreement.
    """

    stream: str
    symbol: str
    funding_time_ms: int
    funding_rate: str
    mark_price: str | None
    rate_type: str | None

    @property
    def canonical_ns(self) -> int:
        return self.funding_time_ms * NS_PER_MILLISECOND

    @property
    def settlement_id(self) -> int:
        return self.funding_time_ms

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any], *, stream: str = UM_FUNDING
    ) -> "FundingSettlement":
        frame = _mapping(payload, f"{stream} payload")
        where = f"{stream} payload"
        mark = frame.get("markPrice")
        if mark is not None and (not isinstance(mark, str) or not mark.strip()):
            raise RecorderEventError(
                f"{where}.markPrice must be a decimal string when present"
            )
        symbol = _optional_text(frame, "symbol")
        if symbol is None:
            raise RecorderEventError(f"{where} is missing the string field 'symbol'")
        return cls(
            stream=stream,
            symbol=symbol.upper(),
            funding_time_ms=_int_field(frame, "fundingTime", where),
            funding_rate=_decimal_field(frame, "fundingRate", where),
            mark_price=None if mark is None else mark.strip(),
            rate_type=_optional_text(frame, "rateType"),
        )

    @classmethod
    def from_rest(
        cls, row: Mapping[str, Any], *, stream: str = UM_FUNDING
    ) -> "FundingSettlement":
        """Read a settlement out of one REST ``fundingRate`` row."""
        return cls.from_payload(row, stream=stream)

    def to_raw_event(
        self,
        payload: Mapping[str, Any],
        *,
        receipt_wall_ns: int,
        receipt_mono_ns: int,
        source: EventSource = EventSource.REST_POLL,
    ) -> RawEvent:
        """The raw record, keyed by the settlement instant *and* the payload.

        Re-fetching a window that already contains this settlement produces the
        same bytes and therefore the same key, so a poller can be as generous as
        it likes with its overlap. A settlement whose published rate or mark
        differs from one already recorded is a different observation, is stored,
        and stops the settlements rebuild until somebody looks at it.
        """
        return RawEvent(
            stream=self.stream,
            canonical_ns=self.canonical_ns,
            time_basis=TimeBasis.EXCHANGE,
            receipt_wall_ns=receipt_wall_ns,
            receipt_mono_ns=receipt_mono_ns,
            source=source,
            dedup_key=f"funding:{self.funding_time_ms}:{payload_digest(payload)[:16]}",
            payload=payload,
        )

    def to_settlement_record(self, *, receipt_wall_ns: int) -> dict[str, Any]:
        """The normalized settlements line: what was settled, and when it was seen."""
        return {
            "funding_time_ms": self.funding_time_ms,
            "funding_time_utc": iso_utc(self.canonical_ns),
            "funding_rate": self.funding_rate,
            "mark_price": self.mark_price,
            "rate_type": self.rate_type,
            "symbol": self.symbol,
            "receipt_wall_ns": receipt_wall_ns,
        }


def sort_events(events: Iterable[RawEvent]) -> list[RawEvent]:
    """Events in the one order every derived file is computed from.

    Canonical time first, then the order they were written in — which is the
    order they are read in, because the raw file is append-only. Sorting by a
    key that is a function of the file rather than of the process makes the
    normalizer produce the same output whether the exchange delivered the frames
    in order or not.
    """
    indexed = sorted(enumerate(events), key=lambda pair: (pair[1].canonical_ns, pair[0]))
    return [event for _, event in indexed]
