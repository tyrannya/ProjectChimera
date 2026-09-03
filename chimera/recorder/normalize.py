"""One row per minute the exchange actually printed, and nothing else.

The normalizer turns the raw event files into the shape everything downstream
reads: a daily table with one row per *closed* minute, plus a metadata document
saying what the day holds and what it does not. It is structural work. It
computes no return, no signal, no funding profit, no basis, no PnL and no
statistic of any kind; every number in the output was published by the exchange
and is carried across unchanged.

**Missing data is missing.** This is the whole discipline of the module and it
is worth stating as a list of things that never happen here: no forward fill, no
backward fill, no interpolation, no invented candle, no minute inferred from a
neighbour, no mark or index or book value substituted from another stream, no
gap quietly closed. A minute for which the exchange never published a closed
kline has **no row**, is named in ``missing``, and is summarised in ``gaps``. A
minute that has a kline but no book has a row whose ``book_present`` is false
and whose book columns are null. The reader can always tell the difference
between "zero" and "not there", because one is a number and the other is null.

**Deterministic by construction.** Events are sorted by canonical time with the
file order as the tie-break, so a day rebuilt from the same raw files is the
same table whether the exchange delivered the frames in order or not, and
whether a late file exists or not. Column names, order and kinds are fixed by
:data:`MARKET_COLUMNS`; the values are written as fixed-width little-endian
bytes for the digest.

**What the identity is.** :func:`digest` is a value-level digest in the style of
:func:`nn.multiclock.candle_digest` and :func:`nn.data_fingerprint`: a domain
prefix, a canonical header naming the market and the columns, the row count, and
then for each column a null mask followed by its values. Two Parquet files
written by different pyarrow versions differ byte for byte while carrying
identical minutes, so the *container* is deliberately not the identity — its
SHA-256 is recorded beside the digest as audit metadata, and the digest is what
a re-encoding cannot move.

This module opens no socket, makes no request and reads no clock.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from chimera.recorder.contract import STORAGE_LAYOUT_VERSION, RecorderContract
from chimera.recorder.events import (
    MINUTES_PER_DAY,
    MS_PER_MINUTE,
    NS_PER_MILLISECOND,
    BookTickerEvent,
    FundingSettlement,
    KlineEvent,
    MarkPriceEvent,
    RawEvent,
    canonical_json,
    day_start_ns,
    iso_utc,
    sort_events,
)
from chimera.recorder.sink import (
    EVENTS_FILE,
    GZIP_SUFFIX,
    LATE_FILE,
    RAW_DIRECTORY,
    RecorderSinkError,
    available_days,
    read_raw_events,
    require_day,
    write_bytes_atomic,
    write_json_atomic,
)

#: Names the meaning of a normalized day: which columns exist, what they hold,
#: and how :func:`digest` reduces them to an identity. Hashed with the values,
#: so a future change to the schema is a change of identity rather than a silent
#: reinterpretation of the digests already written into committed metadata.
NORMALIZED_SCHEMA = "chimera.recorder-minutes/1"

#: Names the shape of a normalized day's metadata document.
NORMALIZED_META_SCHEMA = "chimera.recorder-normalized-day/1"

#: Prefixed into every digest so it can never collide with one taken over some
#: other repository object that happens to hash the same bytes.
NORMALIZED_DIGEST_DOMAIN = b"chimera.recorder-minutes/1"

NORMALIZED_DIRECTORY = "normalized"
FUNDING_DIRECTORY = "funding"
SETTLEMENTS_FILE = "settlements.ndjson"
SETTLEMENTS_DIGEST_FILE = "settlements.sha256"

#: The one clock the recorder normalizes to. Every other clock the project uses
#: is cut from a minute source by :mod:`nn.multiclock`, and there is exactly one
#: source of truth for what a minute is.
CLOCK = "1m"


class RecorderNormalizeError(RuntimeError):
    """A day cannot be normalized into something honest."""


@dataclass(frozen=True)
class ColumnSpec:
    """One column of a normalized day: its name, its width, and whether it may be null."""

    name: str
    kind: str
    nullable: bool

    @property
    def dtype(self) -> str:
        return {"i8": "Int64", "f8": "Float64", "b1": "boolean", "str": "string"}[self.kind]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "nullable": self.nullable}


_KLINE_COLUMNS: tuple[ColumnSpec, ...] = (
    ColumnSpec("minute_open_ms", "i8", False),
    ColumnSpec("kline_close_ms", "i8", False),
    ColumnSpec("kline_source", "str", False),
    ColumnSpec("kline_open", "f8", False),
    ColumnSpec("kline_high", "f8", False),
    ColumnSpec("kline_low", "f8", False),
    ColumnSpec("kline_close", "f8", False),
    ColumnSpec("kline_volume", "f8", False),
    ColumnSpec("kline_trades", "i8", False),
    ColumnSpec("kline_taker_buy_base", "f8", False),
    ColumnSpec("kline_taker_buy_quote", "f8", False),
)

_MARK_COLUMNS: tuple[ColumnSpec, ...] = (
    ColumnSpec("mark_present", "b1", False),
    ColumnSpec("mark_open", "f8", True),
    ColumnSpec("mark_high", "f8", True),
    ColumnSpec("mark_low", "f8", True),
    ColumnSpec("mark_close", "f8", True),
    ColumnSpec("index_open", "f8", True),
    ColumnSpec("index_high", "f8", True),
    ColumnSpec("index_low", "f8", True),
    ColumnSpec("index_close", "f8", True),
    ColumnSpec("mark_events", "i8", True),
    ColumnSpec("funding_rate_last", "f8", True),
    ColumnSpec("next_funding_time_ms", "i8", True),
)

_BOOK_COLUMNS: tuple[ColumnSpec, ...] = (
    ColumnSpec("book_present", "b1", False),
    ColumnSpec("book_bid", "f8", True),
    ColumnSpec("book_bid_qty", "f8", True),
    ColumnSpec("book_ask", "f8", True),
    ColumnSpec("book_ask_qty", "f8", True),
    ColumnSpec("book_update_id", "i8", True),
    ColumnSpec("book_canonical_ms", "i8", True),
    # Which clock ``book_canonical_ms`` came from. Binance's perpetual
    # bookTicker publishes an event time and its value here is ``EXCHANGE``;
    # the spot one publishes an update id and no timestamp at all, so the
    # minute it is assigned to comes from this host's receipt clock and the
    # column says ``RECEIPT``. Carrying the basis into the normalized layer is
    # what stops a later quote-age calculation from treating the two alike.
    ColumnSpec("book_time_basis", "str", True),
)

#: The columns of a normalized day, per market, in the order they are written
#: and hashed. Fixed here rather than derived from the data: a schema that grew
#: a column because one day happened to contain one would make two days of the
#: same market incomparable.
MARKET_COLUMNS: Mapping[str, tuple[ColumnSpec, ...]] = {
    "um": _KLINE_COLUMNS + _MARK_COLUMNS + _BOOK_COLUMNS,
    "spot": _KLINE_COLUMNS + _BOOK_COLUMNS,
}


def columns_for(market: str) -> tuple[ColumnSpec, ...]:
    """The column specification of one market. An unknown market is refused."""
    try:
        return MARKET_COLUMNS[market]
    except KeyError:
        raise RecorderNormalizeError(
            f"no normalized schema for market {market!r}; this build normalizes "
            f"{sorted(MARKET_COLUMNS)}"
        ) from None


@dataclass(frozen=True)
class KlineMinute:
    """The closed candle a minute is built around."""

    source: str
    close_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    trades: int
    taker_buy_base: float
    taker_buy_quote: float


@dataclass(frozen=True)
class MarkMinute:
    """What the mark-price stream said during one minute."""

    open: float
    high: float
    low: float
    close: float
    index_open: float
    index_high: float
    index_low: float
    index_close: float
    events: int
    funding_rate: float
    next_funding_ms: int


@dataclass(frozen=True)
class BookMinute:
    """The last top of book seen before the minute closed."""

    bid: float
    bid_qty: float
    ask: float
    ask_qty: float
    update_id: int
    canonical_ms: int
    time_basis: str


@dataclass(frozen=True)
class MinuteRecord:
    """One normalized minute.

    A record exists only when the exchange published a closed kline for the
    minute, which is exactly the definition the coverage gate counts against.
    ``mark`` and ``book`` are ``None`` when that stream had nothing in the
    minute, and the absence travels into the table as a false ``*_present`` flag
    and null columns rather than as a substituted value.
    """

    minute_open_ms: int
    kline: KlineMinute
    mark: MarkMinute | None = None
    book: BookMinute | None = None

    def to_row(self) -> dict[str, Any]:
        """Every column this record can fill, whether or not a market wants it.

        The union rather than one market's selection, so that adding a market is
        an entry in :data:`MARKET_COLUMNS` and nothing else: a branch on the
        market's name here would silently omit the columns a new market declared
        and leave :func:`minute_frame` to build them out of nothing.
        """
        mark = self.mark
        book = self.book
        return {
            "minute_open_ms": self.minute_open_ms,
            "kline_close_ms": self.kline.close_ms,
            "kline_source": self.kline.source,
            "kline_open": self.kline.open,
            "kline_high": self.kline.high,
            "kline_low": self.kline.low,
            "kline_close": self.kline.close,
            "kline_volume": self.kline.volume,
            "kline_trades": self.kline.trades,
            "kline_taker_buy_base": self.kline.taker_buy_base,
            "kline_taker_buy_quote": self.kline.taker_buy_quote,
            "mark_present": mark is not None,
            "mark_open": None if mark is None else mark.open,
            "mark_high": None if mark is None else mark.high,
            "mark_low": None if mark is None else mark.low,
            "mark_close": None if mark is None else mark.close,
            "index_open": None if mark is None else mark.index_open,
            "index_high": None if mark is None else mark.index_high,
            "index_low": None if mark is None else mark.index_low,
            "index_close": None if mark is None else mark.index_close,
            "mark_events": None if mark is None else mark.events,
            "funding_rate_last": None if mark is None else mark.funding_rate,
            "next_funding_time_ms": None if mark is None else mark.next_funding_ms,
            "book_present": book is not None,
            "book_bid": None if book is None else book.bid,
            "book_bid_qty": None if book is None else book.bid_qty,
            "book_ask": None if book is None else book.ask,
            "book_ask_qty": None if book is None else book.ask_qty,
            "book_update_id": None if book is None else book.update_id,
            "book_canonical_ms": None if book is None else book.canonical_ms,
            "book_time_basis": None if book is None else book.time_basis,
        }


#: Every column a :class:`MinuteRecord` can fill: the union of both markets'
#: schemas, derived from a record rather than restated beside it, so the two
#: cannot drift apart.
ROW_COLUMNS: frozenset[str] = frozenset(
    MinuteRecord(
        minute_open_ms=0,
        kline=KlineMinute(
            source="",
            close_ms=0,
            open=0.0,
            high=0.0,
            low=0.0,
            close=0.0,
            volume=0.0,
            trades=0,
            taker_buy_base=0.0,
            taker_buy_quote=0.0,
        ),
    ).to_row()
)


@dataclass(frozen=True)
class DayReport:
    """What building one normalized day produced."""

    market: str
    day: str
    parquet_path: Path
    meta_path: Path
    rows: int
    missing: tuple[int, ...]
    conflicts: tuple[int, ...]
    digest: str
    parquet_sha256: str


@dataclass(frozen=True)
class SettlementsReport:
    """What rebuilding a market's funding settlements produced."""

    market: str
    path: Path
    digest_path: Path
    rows: int
    first_funding_time_ms: int | None
    last_funding_time_ms: int | None
    sha256: str


# --- value conversion -------------------------------------------------------
def _number(text: str, *, field: str, minute: int) -> float:
    """One published decimal string as a float64.

    Binance publishes BTCUSDT prices with two decimals and sizes with eight;
    both round-trip through float64 exactly, which is why
    :data:`nn.multiclock.PARITY_TOLERANCE` treats agreement between two correct
    readings as identity rather than proximity. A value that will not parse, or
    that is not finite, stops the day rather than becoming a NaN nobody notices.
    """
    try:
        value = float(text)
    except (TypeError, ValueError) as exc:
        raise RecorderNormalizeError(
            f"minute {minute}: {field} {text!r} is not a number the exchange could have "
            f"published: {exc}"
        ) from exc
    if not np.isfinite(value):
        raise RecorderNormalizeError(f"minute {minute}: {field} {text!r} is not finite")
    return value


# --- aggregation ------------------------------------------------------------
def _kline_minutes(
    events: Sequence[RawEvent], *, stream: str, day: str
) -> tuple[dict[int, KlineMinute], list[int], dict[str, int]]:
    """The closed candle of every minute, the conflicting minutes, and the tallies.

    The last closed frame of a minute wins, which is the exchange's own rule for
    a stream that republishes a forming candle every second. Two closed frames
    that disagree on any published field are recorded as a conflict — a
    websocket close and a REST gap-fill of the same minute are both kept on disk
    for exactly this reason — and the last one still wins, because a normalizer
    that dropped the minute would turn a disagreement into a gap.
    """
    chosen: dict[int, KlineMinute] = {}
    material: dict[int, tuple[str, ...]] = {}
    conflicts: set[int] = set()
    counts = {"records": 0, "closed": 0, "partial": 0}
    for event in events:
        counts["records"] += 1
        candle = KlineEvent.from_payload(event.payload, stream=stream)
        if candle.open_ms * NS_PER_MILLISECOND != event.canonical_ns:
            raise RecorderNormalizeError(
                f"{stream} record for {day} stamps canonical_ns {event.canonical_ns} on a "
                f"candle whose open is {candle.open_ms}ms. A candle is stamped by its open, "
                "and a record where the two disagree cannot be placed in a minute"
            )
        if not candle.closed:
            counts["partial"] += 1
            continue
        counts["closed"] += 1
        minute = candle.open_ms
        published = (
            candle.open,
            candle.high,
            candle.low,
            candle.close,
            candle.volume,
            str(candle.trades),
            candle.taker_buy_base,
            candle.taker_buy_quote,
        )
        if minute in material and material[minute] != published:
            conflicts.add(minute)
        material[minute] = published
        chosen[minute] = KlineMinute(
            source=event.source.value,
            close_ms=candle.close_ms,
            open=_number(candle.open, field="kline_open", minute=minute),
            high=_number(candle.high, field="kline_high", minute=minute),
            low=_number(candle.low, field="kline_low", minute=minute),
            close=_number(candle.close, field="kline_close", minute=minute),
            volume=_number(candle.volume, field="kline_volume", minute=minute),
            trades=candle.trades,
            taker_buy_base=_number(
                candle.taker_buy_base, field="kline_taker_buy_base", minute=minute
            ),
            taker_buy_quote=_number(
                candle.taker_buy_quote, field="kline_taker_buy_quote", minute=minute
            ),
        )
    return chosen, sorted(conflicts), counts


def _mark_minutes(
    events: Sequence[RawEvent], *, stream: str
) -> tuple[dict[int, MarkMinute], dict[str, int]]:
    """The per-minute mark and index aggregate, and how many events built it.

    Open is the first event of the minute and close is the last, in the one
    order :func:`chimera.recorder.events.sort_events` defines; high and low are
    the extremes actually observed. A minute with no mark event has no entry,
    and the row it belongs to reports ``mark_present`` false rather than
    borrowing the previous minute's mark.
    """
    opens: dict[int, tuple[float, float, float]] = {}
    closes: dict[int, tuple[float, float, float, int]] = {}
    highs: dict[int, tuple[float, float]] = {}
    lows: dict[int, tuple[float, float]] = {}
    counts: dict[int, int] = {}
    tallies = {"records": 0}
    for event in events:
        tallies["records"] += 1
        update = MarkPriceEvent.from_payload(event.payload, stream=stream)
        minute = event.minute_open_ms
        mark = _number(update.mark, field="mark", minute=minute)
        index = _number(update.index, field="index", minute=minute)
        rate = _number(update.funding_rate, field="funding_rate", minute=minute)
        if minute not in opens:
            opens[minute] = (mark, index, rate)
            highs[minute] = (mark, index)
            lows[minute] = (mark, index)
            counts[minute] = 0
        highs[minute] = (max(highs[minute][0], mark), max(highs[minute][1], index))
        lows[minute] = (min(lows[minute][0], mark), min(lows[minute][1], index))
        closes[minute] = (mark, index, rate, update.next_funding_ms)
        counts[minute] += 1
    minutes = {
        minute: MarkMinute(
            open=opens[minute][0],
            high=highs[minute][0],
            low=lows[minute][0],
            close=closes[minute][0],
            index_open=opens[minute][1],
            index_high=highs[minute][1],
            index_low=lows[minute][1],
            index_close=closes[minute][1],
            events=counts[minute],
            funding_rate=closes[minute][2],
            next_funding_ms=closes[minute][3],
        )
        for minute in opens
    }
    return minutes, tallies


def _book_minutes(
    events: Sequence[RawEvent], *, stream: str
) -> tuple[dict[int, BookMinute], dict[str, int]]:
    """The last top of book of each minute, and how many updates it saw."""
    minutes: dict[int, BookMinute] = {}
    tallies = {"records": 0}
    for event in events:
        tallies["records"] += 1
        quote = BookTickerEvent.from_payload(event.payload, stream=stream)
        minute = event.minute_open_ms
        minutes[minute] = BookMinute(
            bid=_number(quote.bid, field="book_bid", minute=minute),
            bid_qty=_number(quote.bid_qty, field="book_bid_qty", minute=minute),
            ask=_number(quote.ask, field="book_ask", minute=minute),
            ask_qty=_number(quote.ask_qty, field="book_ask_qty", minute=minute),
            update_id=quote.update_id,
            canonical_ms=event.canonical_ns // NS_PER_MILLISECOND,
            time_basis=event.time_basis.value,
        )
    return minutes, tallies


def build_minutes(
    *,
    market: str,
    day: str,
    klines: Sequence[RawEvent],
    kline_stream: str,
    marks: Sequence[RawEvent] = (),
    mark_stream: str | None = None,
    books: Sequence[RawEvent] = (),
    book_stream: str | None = None,
) -> tuple[list[MinuteRecord], list[int], list[int], dict[str, dict[str, int]]]:
    """Aggregate one day of raw events into minutes, gaps and per-stream tallies.

    Pure: takes events, returns records. Nothing here reads a file, and nothing
    here writes one, which is what lets a test build a day out of a list.
    """
    columns_for(market)
    require_day(day)
    start_ms = day_start_ns(day) // NS_PER_MILLISECOND
    end_ms = start_ms + MINUTES_PER_DAY * MS_PER_MINUTE

    candles, conflicts, kline_counts = _kline_minutes(
        sort_events(klines), stream=kline_stream, day=day
    )
    outside = sorted(m for m in candles if not start_ms <= m < end_ms)
    if outside:
        raise RecorderNormalizeError(
            f"{kline_stream} minutes {outside[:3]} are outside the UTC day {day}. A day is "
            "built from its own raw directory, and a record in the wrong one is a defect to "
            "look at rather than to normalize"
        )

    mark_minutes: dict[int, MarkMinute] = {}
    mark_counts = {"records": 0}
    if mark_stream is not None:
        mark_minutes, mark_counts = _mark_minutes(sort_events(marks), stream=mark_stream)
    book_minutes: dict[int, BookMinute] = {}
    book_counts = {"records": 0}
    if book_stream is not None:
        book_minutes, book_counts = _book_minutes(sort_events(books), stream=book_stream)

    records = [
        MinuteRecord(
            minute_open_ms=minute,
            kline=candles[minute],
            mark=mark_minutes.get(minute),
            book=book_minutes.get(minute),
        )
        for minute in sorted(candles)
    ]
    present = set(candles)
    missing = [m for m in range(start_ms, end_ms, MS_PER_MINUTE) if m not in present]
    tallies: dict[str, dict[str, int]] = {kline_stream: kline_counts}
    if mark_stream is not None:
        tallies[mark_stream] = mark_counts
    if book_stream is not None:
        tallies[book_stream] = book_counts
    return records, missing, conflicts, tallies


def minute_frame(records: Iterable[MinuteRecord], *, market: str) -> pd.DataFrame:
    """The normalized table for one market, with fixed columns and fixed dtypes.

    Built from an explicit column list rather than from whatever keys the rows
    happen to carry, and typed with pandas' nullable dtypes so that a missing
    mark is ``pd.NA`` and never ``NaN`` — a value that is absent and a value that
    is not a number are different facts and the table keeps them apart.
    """
    specs = columns_for(market)
    rows = [record.to_row() for record in records]
    unknown = sorted({spec.name for spec in specs} - ROW_COLUMNS)
    if unknown:
        raise RecorderNormalizeError(
            f"the {market} schema declares {unknown}, which a MinuteRecord cannot fill. A "
            "column built out of nothing would be a value the exchange never published"
        )
    frame = pd.DataFrame(rows, columns=[spec.name for spec in specs])
    for spec in specs:
        frame[spec.name] = pd.array(frame[spec.name].to_numpy(dtype=object), dtype=spec.dtype)
    return frame


def gaps_of(missing: Sequence[int]) -> list[dict[str, Any]]:
    """Missing minutes as spans rather than as a count.

    The same shape :func:`nn.multiclock.minute_gaps` reports, for the same
    reason: a reader comparing a day's row count against 1440 needs to see
    *where* the difference is, and a run of six hundred consecutive missing
    minutes is a different incident from six hundred scattered ones.
    """
    spans: list[dict[str, Any]] = []
    for minute in sorted(missing):
        if spans and spans[-1]["_next"] == minute:
            spans[-1]["missing_minutes"] += 1
            spans[-1]["_next"] = minute + MS_PER_MINUTE
            spans[-1]["last_missing_ms"] = minute
            continue
        spans.append(
            {
                "first_missing_ms": minute,
                "last_missing_ms": minute,
                "missing_minutes": 1,
                "_next": minute + MS_PER_MINUTE,
            }
        )
    for span in spans:
        span.pop("_next")
        span["first_missing_utc"] = iso_utc(span["first_missing_ms"] * NS_PER_MILLISECOND)
        span["last_missing_utc"] = iso_utc(span["last_missing_ms"] * NS_PER_MILLISECOND)
    return spans


# --- identity ---------------------------------------------------------------
def _canonical_column(column: pd.Series, spec: ColumnSpec, mask: np.ndarray) -> bytes:
    """One column in its fixed-width little-endian canonical form.

    The null mask is hashed before this, so the placeholder a null takes here
    carries no information and cannot be confused with a real value. Explicit
    ``<`` byte order rather than native: a digest that changed on a big-endian
    machine would make the metadata a claim about the reader's CPU.
    """
    if spec.kind == "i8":
        values = pd.array(column, dtype="Int64").to_numpy(dtype="int64", na_value=0)
        return values.astype("<i8").tobytes()
    if spec.kind == "f8":
        values = pd.array(column, dtype="Float64").to_numpy(dtype="float64", na_value=0.0)
        live = values[~mask]
        if live.size and not np.isfinite(live).all():
            raise RecorderNormalizeError(
                f"column {spec.name!r} holds a non-finite value that is not null. A price "
                "the exchange published is finite; a NaN here would be a computation this "
                "module does not perform"
            )
        # -0.0 and 0.0 are the same published quantity and must hash alike.
        values = np.where(values == 0, 0.0, values)
        return values.astype("<f8").tobytes()
    if spec.kind == "b1":
        values = pd.array(column, dtype="boolean").to_numpy(dtype="bool", na_value=False)
        return values.astype("<u1").tobytes()
    if spec.kind == "str":
        parts: list[bytes] = []
        for null, value in zip(mask, column.to_numpy(dtype=object)):
            if null:
                parts.append(b"\x00")
                continue
            encoded = str(value).encode("utf-8")
            parts.append(b"\x01" + len(encoded).to_bytes(4, "little") + encoded)
        return b"".join(parts)
    raise RecorderNormalizeError(f"unknown column kind {spec.kind!r}")  # pragma: no cover


def digest(frame: pd.DataFrame, *, market: str) -> str:
    """A value-level digest of a normalized day. The identity of its minutes.

    Independent of the file it is stored in: compression, row-group layout,
    key/value metadata, the pyarrow version and the path on disk all leave it
    alone, and changing one published number does not.
    """
    specs = columns_for(market)
    names = [spec.name for spec in specs]
    if list(frame.columns) != names:
        raise RecorderNormalizeError(
            f"a {market} day has columns {names}, got {list(frame.columns)}. The schema is "
            "fixed so that two days of the same market are comparable"
        )
    header = {
        "schema": NORMALIZED_SCHEMA,
        "market": market,
        "columns": [spec.to_dict() for spec in specs],
    }
    running = hashlib.sha256()
    running.update(NORMALIZED_DIGEST_DOMAIN)
    running.update(canonical_json(header).encode("utf-8"))
    running.update(str(len(frame)).encode("utf-8"))
    for spec in specs:
        column = frame[spec.name]
        mask = column.isna().to_numpy(dtype=bool)
        if not spec.nullable and bool(mask.any()):
            raise RecorderNormalizeError(
                f"column {spec.name!r} is declared non-null and holds a null. A minute with "
                "no closed kline has no row at all; it never has a row of nulls"
            )
        running.update(spec.name.encode("utf-8"))
        running.update(mask.astype("<u1").tobytes())
        running.update(_canonical_column(column, spec, mask))
    return running.hexdigest()


def meta(
    frame: pd.DataFrame,
    *,
    market: str,
    day: str,
    contract: RecorderContract,
    missing: Sequence[int],
    conflicts: Sequence[int],
    streams: Mapping[str, Mapping[str, int]],
    parquet_path: str,
    parquet_sha256: str,
    source_paths: Sequence[str] = (),
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """What a normalized day says about itself.

    ``rows + len(missing) == expected_minutes`` is an invariant of this document
    and is asserted here rather than left to a reader: the two numbers are the
    whole of the claim that nothing was invented and nothing was dropped.
    """
    rows = int(len(frame))
    if rows + len(missing) != MINUTES_PER_DAY:
        raise RecorderNormalizeError(
            f"{market} {day} has {rows} rows and {len(missing)} missing minutes, which is "
            f"not {MINUTES_PER_DAY}. Either a minute was counted twice or one is unaccounted "
            "for, and both are defects in the normalizer rather than facts about the day"
        )
    minutes = frame["minute_open_ms"]
    return {
        "meta_schema": NORMALIZED_META_SCHEMA,
        "normalized_schema": NORMALIZED_SCHEMA,
        "storage_layout_version": STORAGE_LAYOUT_VERSION,
        "market": market,
        "clock": CLOCK,
        "day": day,
        "contract": contract.provenance(),
        "columns": [spec.to_dict() for spec in columns_for(market)],
        "expected_minutes": MINUTES_PER_DAY,
        "rows": rows,
        "missing_minutes": len(missing),
        "missing": [int(value) for value in missing],
        "gaps": gaps_of(missing),
        "conflicting_minutes": [int(value) for value in conflicts],
        "streams": {name: dict(counts) for name, counts in sorted(streams.items())},
        "first_minute_open_ms": None if rows == 0 else int(minutes.iloc[0]),
        "last_minute_open_ms": None if rows == 0 else int(minutes.iloc[-1]),
        "digest": digest(frame, market=market),
        "parquet_path": parquet_path,
        "parquet_sha256": parquet_sha256,
        "source_paths": list(source_paths),
        "provenance": None if provenance is None else dict(provenance),
        "note": (
            "digest is the identity of the minutes and reproduces from any host; "
            "parquet_sha256 identifies the stored container, which depends on the writer. "
            "A minute absent from this table was never published as a closed kline and is "
            "listed in missing; it is never filled, interpolated or inferred."
        ),
    }


# --- the normalizer ---------------------------------------------------------
class MinuteNormalizer:
    """Builds normalized days and funding settlements from one storage root."""

    def __init__(self, root: str | Path, contract: RecorderContract) -> None:
        self.root = Path(root)
        self.contract = contract

    # --- paths ------------------------------------------------------------
    def market_dir(self, market: str) -> Path:
        columns_for(market)
        self.contract.market(market)
        return self.root / NORMALIZED_DIRECTORY / market / CLOCK

    def parquet_path(self, market: str, day: str) -> Path:
        return self.market_dir(market) / f"{require_day(day)}.parquet"

    def meta_path(self, market: str, day: str) -> Path:
        return self.market_dir(market) / f"{require_day(day)}.meta.json"

    def sha256_path(self, market: str, day: str) -> Path:
        return self.market_dir(market) / f"{require_day(day)}.sha256"

    def settlements_path(self, market: str) -> Path:
        self.contract.market(market)
        return self.root / FUNDING_DIRECTORY / market / SETTLEMENTS_FILE

    def settlements_digest_path(self, market: str) -> Path:
        return self.settlements_path(market).with_name(SETTLEMENTS_DIGEST_FILE)

    def is_frozen(self, market: str, day: str) -> bool:
        """Whether the day has a ``.sha256`` and is therefore immutable."""
        return self.sha256_path(market, day).exists()

    # --- streams ----------------------------------------------------------
    def _stream(self, market: str, suffix: str) -> str | None:
        name = f"{market}.{suffix}"
        return name if name in self.contract.streams else None

    # --- building ---------------------------------------------------------
    def build_day(
        self, market: str, day: str, *, provenance: Mapping[str, Any] | None = None
    ) -> DayReport:
        """Normalize one UTC day of one market from its raw files.

        Deterministic and idempotent while the day is open: rebuilding it from
        the same raw files produces the same table, the same digest and the same
        metadata, which is what lets the recorder re-derive the current day on
        every restart. Once the day is frozen it is refused, because a frozen
        day is evidence and a correction is a new file with a note rather than a
        quiet overwrite.
        """
        require_day(day)
        columns_for(market)
        self.contract.market(market)
        if self.is_frozen(market, day):
            raise RecorderNormalizeError(
                f"{self.sha256_path(market, day)} exists, so {market} {day} is frozen. A "
                "frozen day is never rewritten; a correction is a new file, and the "
                "reconciliation report says which version was used"
            )

        kline_stream = self._stream(market, "kline_1m")
        if kline_stream is None:
            raise RecorderNormalizeError(
                f"recorder contract {self.contract.label} declares no {market}.kline_1m "
                "stream, and the minute grid is built from closed klines"
            )
        mark_stream = self._stream(market, "markPrice")
        book_stream = self._stream(market, "bookTicker")

        sources: list[str] = []
        klines = self._read(kline_stream, day, sources)
        marks = self._read(mark_stream, day, sources) if mark_stream else []
        books = self._read(book_stream, day, sources) if book_stream else []

        records, missing, conflicts, tallies = build_minutes(
            market=market,
            day=day,
            klines=klines,
            kline_stream=kline_stream,
            marks=marks,
            mark_stream=mark_stream,
            books=books,
            book_stream=book_stream,
        )
        frame = minute_frame(records, market=market)

        parquet = self.parquet_path(market, day)
        parquet.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(parquet, index=False, compression="zstd", compression_level=19)
        parquet_sha = hashlib.sha256(parquet.read_bytes()).hexdigest()

        document = meta(
            frame,
            market=market,
            day=day,
            contract=self.contract,
            missing=missing,
            conflicts=conflicts,
            streams=tallies,
            parquet_path=parquet.relative_to(self.root).as_posix(),
            parquet_sha256=parquet_sha,
            source_paths=sorted(sources),
            provenance=provenance,
        )
        write_json_atomic(self.meta_path(market, day), document)
        return DayReport(
            market=market,
            day=day,
            parquet_path=parquet,
            meta_path=self.meta_path(market, day),
            rows=int(len(frame)),
            missing=tuple(int(value) for value in missing),
            conflicts=tuple(int(value) for value in conflicts),
            digest=str(document["digest"]),
            parquet_sha256=parquet_sha,
        )

    def freeze_day(self, market: str, day: str) -> Path:
        """Write the day's ``.sha256`` and make it immutable.

        The file holds the digest of the Parquet container in ``sha256sum``
        format, which is what an operator can check with the tool of the same
        name. The identity of the minutes is the value digest in the metadata
        beside it, and freezing does not change either.
        """
        target = self.sha256_path(market, day)
        if target.exists():
            raise RecorderNormalizeError(f"{target} already exists; a frozen day stays frozen")
        parquet = self.parquet_path(market, day)
        if not parquet.exists():
            raise RecorderNormalizeError(
                f"no normalized day at {parquet}; freezing a day that was never built would "
                "assert an identity for a file that does not exist"
            )
        line = f"{hashlib.sha256(parquet.read_bytes()).hexdigest()}  {parquet.name}\n"
        write_bytes_atomic(target, line.encode("utf-8"))
        return target

    def build_settlements(self, market: str) -> SettlementsReport:
        """Rebuild a market's funding settlements from every recorded day.

        Append-only in the sense that matters: a settlement, once published, has
        one value for ever, so the file is rewritten atomically from the raw
        records and two records for the same settlement instant that disagree
        stop the rebuild instead of being resolved. Nothing here computes a
        funding flow, a cost or a return — the rate and the mark are carried
        across exactly as the exchange published them.
        """
        stream = self._stream(market, "funding")
        if stream is None:
            raise RecorderNormalizeError(
                f"recorder contract {self.contract.label} declares no {market}.funding stream"
            )
        settlements: dict[int, FundingSettlement] = {}
        receipts: dict[int, int] = {}
        for day in available_days(self.root, stream):
            for event in read_raw_events(self.root, stream, day):
                record = FundingSettlement.from_payload(event.payload, stream=stream)
                existing = settlements.get(record.settlement_id)
                if existing is not None and existing != record:
                    raise RecorderNormalizeError(
                        f"two {stream} records for settlement {record.settlement_id} "
                        f"disagree: {existing} versus {record}. A settlement is published "
                        "once and is final; a disagreement is a finding for the "
                        "reconciliation, never something to resolve here"
                    )
                if existing is None:
                    settlements[record.settlement_id] = record
                    receipts[record.settlement_id] = event.receipt_wall_ns

        lines = [
            canonical_json(
                settlements[key].to_settlement_record(receipt_wall_ns=receipts[key])
            ).encode("utf-8")
            + b"\n"
            for key in sorted(settlements)
        ]
        body = b"".join(lines)
        path = self.settlements_path(market)
        write_bytes_atomic(path, body)

        checksum = hashlib.sha256(body).hexdigest()
        digest_path = self.settlements_digest_path(market)
        write_bytes_atomic(digest_path, f"{checksum}  {SETTLEMENTS_FILE}\n".encode("utf-8"))
        keys = sorted(settlements)
        return SettlementsReport(
            market=market,
            path=path,
            digest_path=digest_path,
            rows=len(keys),
            first_funding_time_ms=keys[0] if keys else None,
            last_funding_time_ms=keys[-1] if keys else None,
            sha256=checksum,
        )

    # --- internals --------------------------------------------------------
    def _read(self, stream: str, day: str, sources: list[str]) -> list[RawEvent]:
        directory = self.root / RAW_DIRECTORY / stream / day
        for name in (EVENTS_FILE, EVENTS_FILE + GZIP_SUFFIX, LATE_FILE):
            candidate = directory / name
            if candidate.exists():
                sources.append(candidate.relative_to(self.root).as_posix())
        try:
            return read_raw_events(self.root, stream, day)
        except RecorderSinkError as exc:
            raise RecorderNormalizeError(str(exc)) from exc
