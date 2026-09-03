"""Normalizing the open day without re-reading it, and why that is still exact.

:meth:`MinuteNormalizer.build_day` re-reads every raw event of a day to produce
its minutes. That is the right shape for a day that is finished and the wrong
shape for a day that is still filling: the perpetual's book stream carries
hundreds of updates a second, so by evening a rebuild reads tens of millions of
records, and a recorder that re-derives its open day every few minutes — or
restarts and has to catch up — spends its time reading its own past. Measured on
a real 44-minute recording, one rebuild of the perpetual took 61.8 s and startup
recovery took 120.1 s, against an adopted expectation of under a minute.

This module folds the same day incrementally, from a cursor, and produces the
same values.

**Why the same values, and not merely similar ones.** The full path sorts each
stream's events by canonical time with file order as the tie-break, then folds
left. Every one of its three aggregations is an extremum or a count under that
order:

* a minute's **kline** is the *last* closed frame — the maximum under the key;
* a minute's **mark** takes its open from the *first* event and its close from
  the *last* — the minimum and the maximum — with high, low and the event count
  order-independent by construction;
* a minute's **book** is the *last* update — the maximum again.

Extrema and counts are commutative and associative, so folding the events in
arrival order while comparing on the sort key gives the identical answer to
sorting first and folding after. The kline conflict flag is the one clause where
that is not obvious: the full path compares each closed frame against the
previous one in sorted order, which is equivalent to "this minute's closed
frames are not all identical", and that is order-independent too. Both claims are
tested against the full path rather than argued, on ordered, out-of-order,
equal-timestamp, duplicated, late and conflicting material.

**The order key.** ``(canonical_ns, file_rank, line)``. ``file_rank`` is 0 for a
day's own events file and 1 for its late file, which is the order
:func:`chimera.recorder.sink.read_raw_events` reads them in, and ``line`` is the
record's position within its file. Sorting by that triple is order-isomorphic to
sorting by ``(canonical_ns, position in the concatenated read)``, which is what
:func:`chimera.recorder.events.sort_events` does.

**The cache is not evidence.** It is an engineering artefact, rebuildable from
the raw files at any time, and it is deliberately outside everything that
identifies a recording: it is not in the contract, not in the value digest, not
in a day's metadata, not in a day manifest, and not in anything a reconciliation
or a scientific report reads. Deleting it costs time and nothing else. The raw
NDJSON remains the only authority, and the durability order is always

    durable raw  ->  cursor and aggregate  ->  normalized output

so a cursor can only ever name material the raw files already hold. A cursor
that names more than the file contains — a truncated tail, a frozen day, a
schema from another build — is refused, and the day is rebuilt the slow way
rather than rendered from a state nobody can vouch for.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from chimera.recorder.contract import RecorderContract
from chimera.recorder.events import (
    MINUTES_PER_DAY,
    MS_PER_MINUTE,
    NS_PER_MILLISECOND,
    BookTickerEvent,
    KlineEvent,
    MarkPriceEvent,
    RawEvent,
    RecorderEventError,
    day_start_ns,
)
from chimera.recorder.normalize import (
    BookMinute,
    DayReport,
    KlineMinute,
    MarkMinute,
    MinuteNormalizer,
    MinuteRecord,
    RecorderNormalizeError,
    _number,
    columns_for,
)
from chimera.recorder.sink import (
    EVENTS_FILE,
    GZIP_SUFFIX,
    LATE_FILE,
    RAW_DIRECTORY,
    require_day,
    require_stream_id,
    write_json_atomic,
)

logger = logging.getLogger(__name__)

#: The cache's own schema. Its own, and explicitly not the normalized day's:
#: a reader must be able to tell an engineering cache from a recorded value at a
#: glance, and a cache written by another build must be refused rather than
#: interpreted.
CACHE_SCHEMA = "chimera.recorder-normalize-cache/1"

#: Where it lives under the storage root. A sibling of the recorded data and not
#: part of it; ``.gitignore`` excludes everything here that is not a manifest.
CACHE_DIRECTORY = "cache"
CACHE_SUBDIRECTORY = "normalize"

#: The two files a day's events are read from, in the order
#: :func:`read_raw_events` reads them, and the rank each contributes to the sort
#: key. A record in the late file sorts after every record in the events file
#: that shares its canonical time, exactly as concatenating the two does.
MAIN_RANK = 0
LATE_RANK = 1


class NormalizeCacheError(RecorderNormalizeError):
    """The cache cannot be used, and the caller must rebuild instead."""


@dataclass
class CacheStatus:
    """What happened the last time a day was rendered. Reported, never persisted."""

    market: str
    day: str
    resumed: bool = False
    rebuilt: bool = False
    reason: str = ""
    replayed_records: int = 0
    replayed_bytes: int = 0
    cached_records: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "market": self.market,
            "day": self.day,
            "resumed": self.resumed,
            "rebuilt": self.rebuilt,
            "reason": self.reason,
            "replayed_records": self.replayed_records,
            "replayed_bytes": self.replayed_bytes,
            "cached_records": self.cached_records,
        }


@dataclass
class _Cursor:
    """How much of one raw file has been folded in already."""

    offset: int = 0
    lines: int = 0
    variant: str = EVENTS_FILE

    def to_dict(self) -> dict[str, Any]:
        return {"offset": self.offset, "lines": self.lines, "variant": self.variant}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "_Cursor":
        return cls(
            offset=int(payload["offset"]),
            lines=int(payload["lines"]),
            variant=str(payload["variant"]),
        )


@dataclass
class _KlineState:
    """One minute's closed candle, its conflict flag, and which frame won."""

    key: tuple[int, int, int]
    source: str
    close_ms: int
    values: tuple[float, float, float, float, float, int, float, float]
    published: tuple[str, ...]
    conflict: bool = False

    def to_minute(self) -> KlineMinute:
        return KlineMinute(
            source=self.source,
            close_ms=self.close_ms,
            open=self.values[0],
            high=self.values[1],
            low=self.values[2],
            close=self.values[3],
            volume=self.values[4],
            trades=self.values[5],
            taker_buy_base=self.values[6],
            taker_buy_quote=self.values[7],
        )


@dataclass
class _MarkState:
    """One minute's mark and index aggregate, as extrema under the sort key."""

    first_key: tuple[int, int, int]
    last_key: tuple[int, int, int]
    open: tuple[float, float, float]
    close: tuple[float, float, float, int]
    high: tuple[float, float]
    low: tuple[float, float]
    events: int

    def to_minute(self) -> MarkMinute:
        return MarkMinute(
            open=self.open[0],
            high=self.high[0],
            low=self.low[0],
            close=self.close[0],
            index_open=self.open[1],
            index_high=self.high[1],
            index_low=self.low[1],
            index_close=self.close[1],
            events=self.events,
            funding_rate=self.close[2],
            next_funding_ms=self.close[3],
        )


@dataclass
class _BookState:
    """One minute's last top of book, and which update it was."""

    key: tuple[int, int, int]
    values: tuple[float, float, float, float]
    update_id: int
    canonical_ms: int
    time_basis: str

    def to_minute(self) -> BookMinute:
        return BookMinute(
            bid=self.values[0],
            bid_qty=self.values[1],
            ask=self.values[2],
            ask_qty=self.values[3],
            update_id=self.update_id,
            canonical_ms=self.canonical_ms,
            time_basis=self.time_basis,
        )


@dataclass
class DayState:
    """Everything the fold knows about one market-day.

    Rebuildable from the raw files in full at any time. Nothing here is a
    recorded value: it is a memo of a computation over recorded values.
    """

    market: str
    day: str
    contract_hash: str
    cursors: dict[str, _Cursor] = field(default_factory=dict)
    klines: dict[int, _KlineState] = field(default_factory=dict)
    marks: dict[int, _MarkState] = field(default_factory=dict)
    books: dict[int, _BookState] = field(default_factory=dict)
    tallies: dict[str, dict[str, int]] = field(default_factory=dict)
    #: The latest canonical instant folded so far, per stream. The recorder's
    #: recovery needs it to know where to resume a gap-fill from, and computing
    #: it by re-reading the day is what made recovery O(the day). It is a
    #: maximum over events already parsed here, so it costs nothing extra and is
    #: exactly what a full re-read would have found.
    last_canonical: dict[str, int] = field(default_factory=dict)

    def cursor(self, key: str) -> _Cursor:
        cursor = self.cursors.get(key)
        if cursor is None:
            cursor = _Cursor()
            self.cursors[key] = cursor
        return cursor

    def observe(self, stream: str, canonical_ns: int) -> None:
        """Remember the latest instant this stream has reached."""
        seen = self.last_canonical.get(stream)
        if seen is None or canonical_ns > seen:
            self.last_canonical[stream] = canonical_ns

    def tally(self, stream: str) -> dict[str, int]:
        counts = self.tallies.get(stream)
        if counts is None:
            counts = {"records": 0}
            self.tallies[stream] = counts
        return counts

    @property
    def records(self) -> int:
        return sum(counts.get("records", 0) for counts in self.tallies.values())


# --- folding ----------------------------------------------------------------
def _fold_kline(state: DayState, event: RawEvent, key, *, stream: str, day: str) -> None:
    candle = KlineEvent.from_payload(event.payload, stream=stream)
    if candle.open_ms * NS_PER_MILLISECOND != event.canonical_ns:
        raise RecorderNormalizeError(
            f"{stream} record for {day} stamps canonical_ns {event.canonical_ns} on a "
            f"candle whose open is {candle.open_ms}ms. A candle is stamped by its open, "
            "and a record where the two disagree cannot be placed in a minute"
        )
    counts = state.tally(stream)
    counts["records"] = counts.get("records", 0) + 1
    counts.setdefault("closed", 0)
    counts.setdefault("partial", 0)
    if not candle.closed:
        counts["partial"] += 1
        return
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
    values = (
        _number(candle.open, field="kline_open", minute=minute),
        _number(candle.high, field="kline_high", minute=minute),
        _number(candle.low, field="kline_low", minute=minute),
        _number(candle.close, field="kline_close", minute=minute),
        _number(candle.volume, field="kline_volume", minute=minute),
        candle.trades,
        _number(candle.taker_buy_base, field="kline_taker_buy_base", minute=minute),
        _number(candle.taker_buy_quote, field="kline_taker_buy_quote", minute=minute),
    )
    existing = state.klines.get(minute)
    if existing is None:
        state.klines[minute] = _KlineState(
            key=key,
            source=event.source.value,
            close_ms=candle.close_ms,
            values=values,
            published=published,
        )
        return
    # "More than one distinct published tuple" — equivalent to the full path's
    # comparison against the previous frame in sorted order, and unlike it,
    # independent of the order the frames are folded in.
    if published != existing.published:
        existing.conflict = True
    if key > existing.key:
        existing.key = key
        existing.source = event.source.value
        existing.close_ms = candle.close_ms
        existing.values = values


def _fold_mark(state: DayState, event: RawEvent, key, *, stream: str) -> None:
    update = MarkPriceEvent.from_payload(event.payload, stream=stream)
    counts = state.tally(stream)
    counts["records"] = counts.get("records", 0) + 1
    minute = event.minute_open_ms
    mark = _number(update.mark, field="mark", minute=minute)
    index = _number(update.index, field="index", minute=minute)
    rate = _number(update.funding_rate, field="funding_rate", minute=minute)
    existing = state.marks.get(minute)
    if existing is None:
        state.marks[minute] = _MarkState(
            first_key=key,
            last_key=key,
            open=(mark, index, rate),
            close=(mark, index, rate, update.next_funding_ms),
            high=(mark, index),
            low=(mark, index),
            events=1,
        )
        return
    existing.events += 1
    existing.high = (max(existing.high[0], mark), max(existing.high[1], index))
    existing.low = (min(existing.low[0], mark), min(existing.low[1], index))
    if key < existing.first_key:
        existing.first_key = key
        existing.open = (mark, index, rate)
    if key > existing.last_key:
        existing.last_key = key
        existing.close = (mark, index, rate, update.next_funding_ms)


def _fold_book(state: DayState, event: RawEvent, key, *, stream: str) -> None:
    quote = BookTickerEvent.from_payload(event.payload, stream=stream)
    counts = state.tally(stream)
    counts["records"] = counts.get("records", 0) + 1
    minute = event.minute_open_ms
    existing = state.books.get(minute)
    if existing is not None and key <= existing.key:
        return
    values = (
        _number(quote.bid, field="book_bid", minute=minute),
        _number(quote.bid_qty, field="book_bid_qty", minute=minute),
        _number(quote.ask, field="book_ask", minute=minute),
        _number(quote.ask_qty, field="book_ask_qty", minute=minute),
    )
    state.books[minute] = _BookState(
        key=key,
        values=values,
        update_id=quote.update_id,
        canonical_ms=event.canonical_ns // NS_PER_MILLISECOND,
        time_basis=event.time_basis.value,
    )


# --- reading the tail --------------------------------------------------------
def _complete_lines(path: Path, offset: int) -> tuple[list[bytes], int]:
    """Every whole record after ``offset``, and how many bytes they occupy.

    A trailing partial line is left unconsumed: the writer is appending and the
    record is not finished, so it is not yet material. That is also what makes a
    crash between an append and a cursor update safe — the next pass reads the
    record exactly once, from the offset that never claimed it.
    """
    with path.open("rb") as handle:
        handle.seek(offset)
        data = handle.read()
    if not data:
        return [], 0
    end = data.rfind(b"\n")
    if end == -1:
        return [], 0
    consumed = end + 1
    return data[:consumed].split(b"\n")[:-1], consumed


def _advance(
    state: DayState, path: Path, cursor_key: str, rank: int, folder, *, stream: str
) -> tuple[int, int]:
    """Fold every unread record of one file into the state. Returns (records, bytes)."""
    cursor = state.cursor(cursor_key)
    if not path.exists():
        return 0, 0
    size = path.stat().st_size
    if size < cursor.offset:
        raise NormalizeCacheError(
            f"{path} is {size} bytes and the cache has already folded {cursor.offset}. A raw "
            "file that got shorter is a truncated tail or a different file; either way the "
            "cache cannot vouch for what it holds"
        )
    lines, consumed = _complete_lines(path, cursor.offset)
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            event = RawEvent.from_line(line)
        except RecorderEventError as exc:
            raise NormalizeCacheError(f"{path} line {cursor.lines + index}: {exc}") from exc
        folder(event, (event.canonical_ns, rank, cursor.lines + index))
        state.observe(stream, event.canonical_ns)
    cursor.offset += consumed
    cursor.lines += len(lines)
    return len(lines), consumed


# --- persistence -------------------------------------------------------------
def _key_list(key: Iterable[int]) -> list[int]:
    return [int(part) for part in key]


def state_to_document(state: DayState) -> dict[str, Any]:
    """The cache exactly as it is written. Plain JSON, no float surprises.

    Python renders a float as the shortest string that reads back as the same
    float64, so a value written here and read back is the same number, which is
    what lets the digest of a rendered day match the digest of a rebuilt one.
    """
    return {
        "cache_schema": CACHE_SCHEMA,
        "note": (
            "Engineering cache. Rebuildable from the raw files, not evidence, not part of "
            "any contract, digest, manifest or report. Safe to delete; deleting it costs "
            "time and nothing else."
        ),
        "market": state.market,
        "day": state.day,
        "contract_hash": state.contract_hash,
        "cursors": {name: cursor.to_dict() for name, cursor in sorted(state.cursors.items())},
        "tallies": {name: dict(counts) for name, counts in sorted(state.tallies.items())},
        "last_canonical": {
            name: int(value) for name, value in sorted(state.last_canonical.items())
        },
        "klines": {
            str(minute): {
                "key": _key_list(entry.key),
                "source": entry.source,
                "close_ms": entry.close_ms,
                "values": list(entry.values),
                "published": list(entry.published),
                "conflict": entry.conflict,
            }
            for minute, entry in sorted(state.klines.items())
        },
        "marks": {
            str(minute): {
                "first_key": _key_list(entry.first_key),
                "last_key": _key_list(entry.last_key),
                "open": list(entry.open),
                "close": list(entry.close),
                "high": list(entry.high),
                "low": list(entry.low),
                "events": entry.events,
            }
            for minute, entry in sorted(state.marks.items())
        },
        "books": {
            str(minute): {
                "key": _key_list(entry.key),
                "values": list(entry.values),
                "update_id": entry.update_id,
                "canonical_ms": entry.canonical_ms,
                "time_basis": entry.time_basis,
            }
            for minute, entry in sorted(state.books.items())
        },
    }


def state_from_document(
    document: Mapping[str, Any], *, market: str, day: str, hash_: str
) -> DayState:
    """Read a cache back, refusing anything this build cannot vouch for."""
    if not isinstance(document, Mapping):
        raise NormalizeCacheError("the cache file does not hold an object")
    schema = document.get("cache_schema")
    if schema != CACHE_SCHEMA:
        raise NormalizeCacheError(
            f"the cache declares schema {schema!r}; this build writes {CACHE_SCHEMA!r}"
        )
    if document.get("market") != market or document.get("day") != day:
        raise NormalizeCacheError(
            f"the cache is for {document.get('market')!r} {document.get('day')!r}, not "
            f"{market!r} {day!r}"
        )
    if document.get("contract_hash") != hash_:
        raise NormalizeCacheError(
            "the cache was folded under contract hash "
            f"{document.get('contract_hash')!r} and this recorder carries {hash_!r}"
        )
    try:
        state = DayState(
            market=market,
            day=day,
            contract_hash=hash_,
            cursors={
                name: _Cursor.from_dict(payload)
                for name, payload in dict(document["cursors"]).items()
            },
            tallies={
                name: {key: int(value) for key, value in dict(counts).items()}
                for name, counts in dict(document["tallies"]).items()
            },
            last_canonical={
                name: int(value)
                for name, value in dict(document.get("last_canonical", {})).items()
            },
            klines={
                int(minute): _KlineState(
                    key=tuple(int(part) for part in entry["key"]),
                    source=str(entry["source"]),
                    close_ms=int(entry["close_ms"]),
                    values=tuple(entry["values"]),
                    published=tuple(str(part) for part in entry["published"]),
                    conflict=bool(entry["conflict"]),
                )
                for minute, entry in dict(document["klines"]).items()
            },
            marks={
                int(minute): _MarkState(
                    first_key=tuple(int(part) for part in entry["first_key"]),
                    last_key=tuple(int(part) for part in entry["last_key"]),
                    open=tuple(entry["open"]),
                    close=tuple(entry["close"][:3]) + (int(entry["close"][3]),),
                    high=tuple(entry["high"]),
                    low=tuple(entry["low"]),
                    events=int(entry["events"]),
                )
                for minute, entry in dict(document["marks"]).items()
            },
            books={
                int(minute): _BookState(
                    key=tuple(int(part) for part in entry["key"]),
                    values=tuple(entry["values"]),
                    update_id=int(entry["update_id"]),
                    canonical_ms=int(entry["canonical_ms"]),
                    time_basis=str(entry["time_basis"]),
                )
                for minute, entry in dict(document["books"]).items()
            },
        )
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise NormalizeCacheError(f"the cache is not readable as one: {exc!r}") from exc
    # The kline trades count is an integer in the full path and must stay one.
    for entry in state.klines.values():
        values = list(entry.values)
        values[5] = int(values[5])
        entry.values = tuple(values)
    return state


class IncrementalNormalizer:
    """Renders the open day from a cursor, and falls back to the slow truth.

    :meth:`build_day` produces exactly what
    :meth:`MinuteNormalizer.build_day` produces — the same rows, the same
    missing list, the same conflicts, the same tallies, the same digest — and
    the full path is kept as the oracle the parity tests compare against.
    """

    def __init__(
        self,
        root: str | Path,
        contract: RecorderContract,
        *,
        normalizer: MinuteNormalizer | None = None,
    ) -> None:
        self.root = Path(root)
        self.contract = contract
        self.normalizer = normalizer or MinuteNormalizer(self.root, contract)
        self.status: dict[tuple[str, str], CacheStatus] = {}

    # --- paths ------------------------------------------------------------
    def cache_path(self, market: str, day: str) -> Path:
        columns_for(market)
        self.contract.market(market)
        return (
            self.root
            / CACHE_DIRECTORY
            / CACHE_SUBDIRECTORY
            / market
            / f"{require_day(day)}.json"
        )

    def _raw_paths(self, stream: str, day: str) -> tuple[Path, Path, Path]:
        directory = self.root / RAW_DIRECTORY / require_stream_id(stream) / require_day(day)
        return (
            directory / EVENTS_FILE,
            directory / (EVENTS_FILE + GZIP_SUFFIX),
            directory / LATE_FILE,
        )

    def _streams(self, market: str) -> dict[str, str | None]:
        def named(suffix: str) -> str | None:
            name = f"{market}.{suffix}"
            return name if name in self.contract.streams else None

        return {
            "kline": named("kline_1m"),
            "mark": named("markPrice"),
            "book": named("bookTicker"),
        }

    # --- the fold ---------------------------------------------------------
    def load_state(self, market: str, day: str) -> tuple[DayState, CacheStatus]:
        """The cache for one day, or a fresh state and the reason it is fresh."""
        status = CacheStatus(market=market, day=day)
        path = self.cache_path(market, day)
        fresh = DayState(market=market, day=day, contract_hash=self.contract.contract_hash)
        if not path.exists():
            status.reason = "no cache; folding the day from the start"
            return fresh, status
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            state = state_from_document(
                document, market=market, day=day, hash_=self.contract.contract_hash
            )
        except (OSError, ValueError, NormalizeCacheError) as exc:
            status.rebuilt = True
            status.reason = f"cache unusable, folding from the start instead: {exc}"
            logger.warning("%s %s: %s", market, day, status.reason)
            return fresh, status
        status.resumed = True
        status.cached_records = state.records
        status.reason = "resumed from cache"
        return state, status

    def update(self, market: str, day: str) -> tuple[DayState, CacheStatus]:
        """Fold every raw record this day has gained since the cache was written."""
        state, status = self.load_state(market, day)
        streams = self._streams(market)
        if streams["kline"] is None:
            raise RecorderNormalizeError(
                f"recorder contract {self.contract.label} declares no {market}.kline_1m "
                "stream, and the minute grid is built from closed klines"
            )
        for kind, stream in streams.items():
            if stream is None:
                continue
            plain, packed, late = self._raw_paths(stream, day)
            if packed.exists():
                # A frozen day. Its events live in the compressed file now, so a
                # byte cursor into the plain one means nothing, and a day that is
                # finished is exactly the case the authoritative path is for.
                raise NormalizeCacheError(
                    f"{stream} {day} is frozen; a finished day is rebuilt from the "
                    "compressed file by the authoritative path"
                )
            counts = state.tally(stream)
            if kind == "kline":
                counts.setdefault("closed", 0)
                counts.setdefault("partial", 0)

            def folder(event, key, _kind=kind, _stream=stream):
                if _kind == "kline":
                    _fold_kline(state, event, key, stream=_stream, day=day)
                elif _kind == "mark":
                    _fold_mark(state, event, key, stream=_stream)
                else:
                    _fold_book(state, event, key, stream=_stream)

            for path, rank, name in (
                (plain, MAIN_RANK, "main"),
                (late, LATE_RANK, "late"),
            ):
                records, consumed = _advance(
                    state, path, f"{stream}:{name}", rank, folder, stream=stream
                )
                status.replayed_records += records
                status.replayed_bytes += consumed
        self.status[(market, day)] = status
        return state, status

    def peek_last_canonical(self, market: str, day: str) -> dict[str, int]:
        """The latest instant per stream, from the cache plus the unfolded tail.

        The recorder's recovery needs this to resume a gap-fill, and reading it
        by parsing the whole day is what made a restart cost minutes. The cache
        already holds the maximum over everything it has folded, so only the
        bytes no cursor has claimed have to be looked at — a few seconds of
        arrivals on a normal restart rather than a day of them.

        Exact, not an approximation: a maximum over the folded part and a
        maximum over the rest is the maximum over the whole. Returns ``{}`` when
        there is no usable cache, and the caller falls back to reading the day.
        """
        try:
            state, status = self.load_state(market, day)
        except NormalizeCacheError:
            return {}
        if status.rebuilt or not status.resumed:
            return {}
        latest = dict(state.last_canonical)
        for stream in self._streams(market).values():
            if stream is None:
                continue
            plain, packed, late = self._raw_paths(stream, day)
            if packed.exists():
                return {}
            for (
                path,
                name,
            ) in ((plain, "main"), (late, "late")):
                if not path.exists():
                    continue
                cursor = state.cursor(f"{stream}:{name}")
                if path.stat().st_size < cursor.offset:
                    return {}
                lines, _ = _complete_lines(path, cursor.offset)
                for line in lines:
                    if not line.strip():
                        continue
                    try:
                        instant = RawEvent.from_line(line).canonical_ns
                    except RecorderEventError:
                        return {}
                    if instant > latest.get(stream, -1):
                        latest[stream] = instant
        return latest

    def render(self, state: DayState) -> tuple[list[MinuteRecord], list[int], list[int], dict]:
        """The same four values :func:`build_minutes` returns, from the state."""
        day = state.day
        start_ms = day_start_ns(day) // NS_PER_MILLISECOND
        end_ms = start_ms + MINUTES_PER_DAY * MS_PER_MINUTE
        streams = self._streams(state.market)
        outside = sorted(m for m in state.klines if not start_ms <= m < end_ms)
        if outside:
            raise RecorderNormalizeError(
                f"{streams['kline']} minutes {outside[:3]} are outside the UTC day {day}. A "
                "day is built from its own raw directory, and a record in the wrong one is a "
                "defect to look at rather than to normalize"
            )
        records = [
            MinuteRecord(
                minute_open_ms=minute,
                kline=state.klines[minute].to_minute(),
                mark=(state.marks[minute].to_minute() if minute in state.marks else None),
                book=(state.books[minute].to_minute() if minute in state.books else None),
            )
            for minute in sorted(state.klines)
        ]
        present = set(state.klines)
        missing = [m for m in range(start_ms, end_ms, MS_PER_MINUTE) if m not in present]
        conflicts = sorted(m for m, entry in state.klines.items() if entry.conflict)
        tallies: dict[str, dict[str, int]] = {}
        for kind, stream in streams.items():
            if stream is None:
                continue
            counts = dict(state.tallies.get(stream, {"records": 0}))
            if kind == "kline":
                counts.setdefault("closed", 0)
                counts.setdefault("partial", 0)
                counts = {
                    "records": counts["records"],
                    "closed": counts["closed"],
                    "partial": counts["partial"],
                }
            else:
                counts = {"records": counts.get("records", 0)}
            tallies[stream] = counts
        return records, missing, conflicts, tallies

    def save(self, state: DayState) -> Path:
        """Persist the cache atomically, after the raw it describes is durable."""
        path = self.cache_path(state.market, state.day)
        write_json_atomic(path, state_to_document(state))
        return path

    def drop(self, market: str, day: str) -> None:
        """Forget the cache for one day. The raw files are untouched."""
        path = self.cache_path(market, day)
        if path.exists():
            path.unlink()

    # --- the public shape -------------------------------------------------
    def build_day(
        self, market: str, day: str, *, provenance: Mapping[str, Any] | None = None
    ) -> DayReport:
        """Normalize one day incrementally, or fall back to the authoritative path.

        Falls back — whole, never partially — when the cache cannot be vouched
        for: a different schema, another contract, a shorter raw file than the
        cursor claims, a day frozen since the cache was written, a record that
        will not parse. The slow path is the same code every other caller uses,
        so a fallback costs time and changes no value.
        """
        try:
            state, status = self.update(market, day)
        except NormalizeCacheError as exc:
            logger.warning(
                "%s %s: %s; rebuilding from raw with the authoritative path", market, day, exc
            )
            self.drop(market, day)
            self.status[(market, day)] = CacheStatus(
                market=market, day=day, rebuilt=True, reason=str(exc)
            )
            return self.normalizer.build_day(market, day, provenance=provenance)
        report = self.normalizer.write_day(
            market, day, self.render(state), provenance=provenance
        )
        self.save(state)
        return report
