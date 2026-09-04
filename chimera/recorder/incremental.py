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
NDJSON remains the only authority.

**Which is also why it does not get to be believed.** Being rebuildable is half
of "not evidence"; the other half is that its aggregates never become the source
of a normalized value. A file whose schema, market, day and contract hash all
still look right can still hold a book price, a mark close, a kline value, a
tally or a resume instant that nobody folded, and rendering it would quietly
make the cache the authority the raw files are supposed to be. So every cache is
written sealed with :func:`cache_digest` — a domain-separated SHA-256 over the
whole document — and the seal is verified before a single aggregate is read. It
is a checksum and not a signature: it makes an altered cache detectable, and it
does not pretend to stop someone who can rewrite the file and recompute the
hash. It does not have to. A cache that fails the seal, that is missing it, or
whose shapes are not the shapes this build writes is discarded and folded again
from the raw — never repaired, never half believed.

**The order things become durable.** Raw first, then the checkpoint, then the
derived output:

    durable raw  ->  verified cursor and aggregate  ->  normalized output

:meth:`IncrementalNormalizer.build_day` writes in that order, so each crash
window has exactly one recovery. Before the cache: the raw tail is folded again,
once, because the cursor never claimed it. After the cache and before the
output: the day is rendered again from the checkpoint without re-reading a byte
of the prefix. After the output: the three already agree. A cursor can therefore
only ever name material the raw files already hold, and one that names more than
the file contains — a truncated tail, a frozen day, a schema from another build
— is refused, and the day is rebuilt the slow way rather than rendered from a
state nobody can vouch for.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from chimera.recorder.contract import RecorderContract
from chimera.recorder.events import (
    MINUTES_PER_DAY,
    MS_PER_MINUTE,
    NS_PER_MILLISECOND,
    BookTickerEvent,
    EventSource,
    KlineEvent,
    MarkPriceEvent,
    RawEvent,
    RecorderEventError,
    TimeBasis,
    canonical_json,
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
    RecorderSinkError,
    require_day,
    require_stream_id,
    write_json_atomic,
)

logger = logging.getLogger(__name__)

#: The cache's own schema. Its own, and explicitly not the normalized day's:
#: a reader must be able to tell an engineering cache from a recorded value at a
#: glance, and a cache written by another build must be refused rather than
#: interpreted.
#:
#: ``/2`` because ``/1`` carried no integrity digest, so its aggregates could be
#: edited and still read back as if they had been folded. A ``/1`` file is not
#: migrated: trusting its values is the one thing the bump exists to stop, so it
#: is stale engineering state and its day is folded again from the raw.
CACHE_SCHEMA = "chimera.recorder-normalize-cache/2"

#: Prefixed into the cache's integrity digest so it can never collide with a
#: digest taken over some other repository object, and carrying the schema so a
#: document written under one cannot verify under another.
CACHE_DIGEST_DOMAIN = b"chimera.recorder-normalize-cache/2"

#: Where the seal lives in the document. Engineering metadata, and only that: it
#: identifies a memo of a computation and never a recorded value, so it belongs
#: to no contract, no contract hash, no value digest, no day metadata, no
#: manifest and no report.
CACHE_DIGEST_FIELD = "cache_digest"

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


def cache_digest(document: Mapping[str, Any]) -> str:
    """The cache's seal: SHA-256 over everything in the document except itself.

    A domain prefix, then the canonical JSON of the document with
    :data:`CACHE_DIGEST_FIELD` removed — so schema, market, day, contract hash,
    every cursor, every tally, every resume instant and every kline, mark and
    book aggregate is covered, and changing any one of them changes the seal.

    A checksum, not a signature. It makes an edited cache detectable; it does not
    pretend to withstand someone who can rewrite the file and recompute the hash.
    It does not have to: the raw NDJSON is the authority, and a cache that fails
    this check is thrown away and folded again from it.
    """
    payload = {key: value for key, value in document.items() if key != CACHE_DIGEST_FIELD}
    running = hashlib.sha256()
    running.update(CACHE_DIGEST_DOMAIN)
    running.update(canonical_json(payload).encode("utf-8"))
    return running.hexdigest()


def state_to_document(state: DayState) -> dict[str, Any]:
    """The cache exactly as it is written, sealed. Plain JSON, no float surprises.

    Python renders a float as the shortest string that reads back as the same
    float64, so a value written here and read back is the same number, which is
    what lets the digest of a rendered day match the digest of a rebuilt one —
    and what lets the seal taken here verify against the file it is read from.
    """
    document: dict[str, Any] = {
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
    document[CACHE_DIGEST_FIELD] = cache_digest(document)
    return document


# --- reading a cache back ----------------------------------------------------
# A cache is trusted whole or refused whole, so every one of these raises
# NormalizeCacheError with a sentence in it rather than letting a short tuple, a
# string where a number belongs or a negative offset leave this module as a bare
# IndexError, TypeError or OSError for RecorderService to meet.

#: The characters a SHA-256 hex digest is made of, and the only ones a seal may
#: hold. ``hexdigest`` is lowercase, so a mixed-case field did not come from one.
_HEX = frozenset("0123456789abcdef")

#: Exactly the keys each part of the document carries. Exactly: a key this build
#: does not write is as much a sign of a document it did not write as a missing
#: one, and a validation with a hole in it is not one.
_DOCUMENT_KEYS = frozenset(
    {
        "cache_schema",
        CACHE_DIGEST_FIELD,
        "note",
        "market",
        "day",
        "contract_hash",
        "cursors",
        "tallies",
        "last_canonical",
        "klines",
        "marks",
        "books",
    }
)
_CURSOR_KEYS = frozenset({"offset", "lines", "variant"})
_KLINE_KEYS = frozenset({"key", "source", "close_ms", "values", "published", "conflict"})
_MARK_KEYS = frozenset({"first_key", "last_key", "open", "close", "high", "low", "events"})
_BOOK_KEYS = frozenset({"key", "values", "update_id", "canonical_ms", "time_basis"})

#: The raw files a cursor can be reading. Both, because a cursor names one of
#: the two files :func:`read_raw_events` concatenates and nothing else.
_CURSOR_VARIANTS = (EVENTS_FILE, LATE_FILE)


def _mapping(value: Any, what: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise NormalizeCacheError(f"{what} is {type(value).__name__}, not an object")
    return value


def _keys(payload: Mapping[str, Any], expected: frozenset[str], what: str) -> None:
    present = frozenset(payload)
    if present != expected:
        missing = sorted(expected - present)
        extra = sorted(present - expected)
        raise NormalizeCacheError(
            f"{what} is missing {missing} and carries {extra} this build does not write"
        )


def _int(value: Any, what: str, *, minimum: int | None = None) -> int:
    # ``bool`` is an ``int`` in Python and is not one here: a cache that holds
    # True where a count belongs is a cache that was written by something else.
    if isinstance(value, bool) or not isinstance(value, int):
        raise NormalizeCacheError(f"{what} is {value!r}, not an integer")
    if minimum is not None and value < minimum:
        raise NormalizeCacheError(f"{what} is {value}; the least it can be is {minimum}")
    return value


def _float(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NormalizeCacheError(f"{what} is {value!r}, not a number")
    number = float(value)
    if not math.isfinite(number):
        raise NormalizeCacheError(f"{what} is {value!r}, which is not a finite number")
    return number


def _text(value: Any, what: str) -> str:
    if not isinstance(value, str):
        raise NormalizeCacheError(f"{what} is {value!r}, not a string")
    return value


def _bool(value: Any, what: str) -> bool:
    if not isinstance(value, bool):
        raise NormalizeCacheError(f"{what} is {value!r}, not a boolean")
    return value


def _label(value: Any, kind: type[EventSource] | type[TimeBasis], what: str) -> str:
    text = _text(value, what)
    try:
        kind(text)
    except ValueError as exc:
        raise NormalizeCacheError(
            f"{what} is {text!r}, and the labels this build writes are "
            f"{[member.value for member in kind]}"
        ) from exc
    return text


def _components(value: Any, arity: int, what: str) -> list[Any]:
    if not isinstance(value, list) or len(value) != arity:
        raise NormalizeCacheError(f"{what} is {value!r}; this build writes {arity} components")
    return value


def _numbers(value: Any, arity: int, what: str) -> tuple[float, ...]:
    return tuple(
        _float(part, f"{what}[{index}]")
        for index, part in enumerate(_components(value, arity, what))
    )


def _order_key(value: Any, what: str) -> tuple[int, int, int]:
    parts = _components(value, 3, what)
    first, second, third = (
        _int(part, f"{what}[{index}]", minimum=0) for index, part in enumerate(parts)
    )
    return first, second, third


def _minute(key: Any, what: str) -> int:
    """A minute key: an integer, and deliberately not asserted to be aligned.

    Neither this module nor the full path floors a kline into its minute — both
    take the open the exchange published — so a multiple-of-60000 check here
    could refuse a cache the oracle would happily accept, and the price of a
    refusal is the whole-day rebuild this module exists to avoid. Nothing is lost
    by leaving it out: the seal already covers the key against editing, which is
    what this validation is for.
    """
    text = _text(key, what)
    try:
        minute = int(text)
    except ValueError as exc:
        raise NormalizeCacheError(f"{what} {text!r} is not an integer minute") from exc
    if minute < 0:
        raise NormalizeCacheError(f"{what} {minute} is before the epoch")
    return minute


def _verify_seal(document: Mapping[str, Any]) -> None:
    """Refuse a document whose contents are not the contents it was sealed over."""
    claimed = document.get(CACHE_DIGEST_FIELD)
    if claimed is None:
        raise NormalizeCacheError(
            f"the cache carries no {CACHE_DIGEST_FIELD}. Every {CACHE_SCHEMA} cache is "
            "written with one, and aggregates nothing vouches for are not read"
        )
    if not isinstance(claimed, str) or len(claimed) != 64 or not _HEX.issuperset(claimed):
        raise NormalizeCacheError(
            f"the cache's {CACHE_DIGEST_FIELD} is {claimed!r}, which is not a SHA-256"
        )
    try:
        actual = cache_digest(document)
    except RecorderEventError as exc:
        raise NormalizeCacheError(
            "the cache holds something that cannot be digested, so nothing in it can be "
            f"vouched for: {exc}"
        ) from exc
    if actual != claimed:
        raise NormalizeCacheError(
            f"the cache is sealed {claimed} and its contents digest to {actual}. Its "
            "aggregates are not the ones it was written with, and raw is what they are "
            "rebuilt from"
        )


def _cursor_from(payload: Any, what: str) -> _Cursor:
    entry = _mapping(payload, what)
    _keys(entry, _CURSOR_KEYS, what)
    variant = _text(entry["variant"], f"{what}.variant")
    if variant not in _CURSOR_VARIANTS:
        raise NormalizeCacheError(
            f"{what}.variant is {variant!r}; a cursor reads {EVENTS_FILE} or {LATE_FILE}"
        )
    return _Cursor(
        offset=_int(entry["offset"], f"{what}.offset", minimum=0),
        lines=_int(entry["lines"], f"{what}.lines", minimum=0),
        variant=variant,
    )


def _kline_from(payload: Any, what: str) -> _KlineState:
    entry = _mapping(payload, what)
    _keys(entry, _KLINE_KEYS, what)
    values = _components(entry["values"], 8, f"{what}.values")
    published = _components(entry["published"], 8, f"{what}.published")
    return _KlineState(
        key=_order_key(entry["key"], f"{what}.key"),
        source=_label(entry["source"], EventSource, f"{what}.source"),
        close_ms=_int(entry["close_ms"], f"{what}.close_ms", minimum=0),
        values=(
            _float(values[0], f"{what}.values[0]"),
            _float(values[1], f"{what}.values[1]"),
            _float(values[2], f"{what}.values[2]"),
            _float(values[3], f"{what}.values[3]"),
            _float(values[4], f"{what}.values[4]"),
            # The trade count is an integer in the full path and stays one here.
            _int(values[5], f"{what}.values[5]", minimum=0),
            _float(values[6], f"{what}.values[6]"),
            _float(values[7], f"{what}.values[7]"),
        ),
        published=tuple(
            _text(part, f"{what}.published[{index}]") for index, part in enumerate(published)
        ),
        conflict=_bool(entry["conflict"], f"{what}.conflict"),
    )


def _mark_from(payload: Any, what: str) -> _MarkState:
    entry = _mapping(payload, what)
    _keys(entry, _MARK_KEYS, what)
    close = _components(entry["close"], 4, f"{what}.close")
    open_ = _numbers(entry["open"], 3, f"{what}.open")
    high = _numbers(entry["high"], 2, f"{what}.high")
    low = _numbers(entry["low"], 2, f"{what}.low")
    return _MarkState(
        first_key=_order_key(entry["first_key"], f"{what}.first_key"),
        last_key=_order_key(entry["last_key"], f"{what}.last_key"),
        open=(open_[0], open_[1], open_[2]),
        close=(
            _float(close[0], f"{what}.close[0]"),
            _float(close[1], f"{what}.close[1]"),
            _float(close[2], f"{what}.close[2]"),
            _int(close[3], f"{what}.close[3]"),
        ),
        high=(high[0], high[1]),
        low=(low[0], low[1]),
        # At least one: a mark aggregate exists because an event created it, and
        # a minute claiming none would publish a count no reading supports.
        events=_int(entry["events"], f"{what}.events", minimum=1),
    )


def _book_from(payload: Any, what: str) -> _BookState:
    entry = _mapping(payload, what)
    _keys(entry, _BOOK_KEYS, what)
    values = _numbers(entry["values"], 4, f"{what}.values")
    return _BookState(
        key=_order_key(entry["key"], f"{what}.key"),
        values=(values[0], values[1], values[2], values[3]),
        update_id=_int(entry["update_id"], f"{what}.update_id"),
        canonical_ms=_int(entry["canonical_ms"], f"{what}.canonical_ms", minimum=0),
        time_basis=_label(entry["time_basis"], TimeBasis, f"{what}.time_basis"),
    )


def _tally_from(payload: Any, what: str) -> dict[str, int]:
    counts = _mapping(payload, what)
    return {
        str(name): _int(value, f"{what}[{name!r}]", minimum=0)
        for name, value in counts.items()
    }


def state_from_document(
    document: Mapping[str, Any], *, market: str, day: str, hash_: str
) -> DayState:
    """Read a cache back, refusing anything this build cannot vouch for.

    In this order: is it an object at all, is it this build's schema, do its
    contents digest to the seal it carries, is it this market's day under this
    contract, and is every value the shape this build writes. The seal comes
    before any aggregate is looked at, because looking at one first would
    already be trusting it.
    """
    if not isinstance(document, Mapping):
        raise NormalizeCacheError("the cache file does not hold an object")
    schema = document.get("cache_schema")
    if schema != CACHE_SCHEMA:
        raise NormalizeCacheError(
            f"the cache declares schema {schema!r}; this build writes {CACHE_SCHEMA!r}"
        )
    _verify_seal(document)
    _keys(document, _DOCUMENT_KEYS, "the cache")
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
    _text(document["note"], "the cache's note")
    return DayState(
        market=market,
        day=day,
        contract_hash=hash_,
        cursors={
            str(name): _cursor_from(payload, f"cursor {name!r}")
            for name, payload in _mapping(document["cursors"], "the cache's cursors").items()
        },
        tallies={
            str(name): _tally_from(counts, f"the {name!r} tallies")
            for name, counts in _mapping(document["tallies"], "the cache's tallies").items()
        },
        last_canonical={
            str(name): _int(value, f"last_canonical[{name!r}]", minimum=0)
            for name, value in _mapping(
                document["last_canonical"], "the cache's last_canonical"
            ).items()
        },
        klines={
            _minute(minute, "a kline minute key"): _kline_from(entry, f"kline minute {minute}")
            for minute, entry in _mapping(document["klines"], "the cache's klines").items()
        },
        marks={
            _minute(minute, "a mark minute key"): _mark_from(entry, f"mark minute {minute}")
            for minute, entry in _mapping(document["marks"], "the cache's marks").items()
        },
        books={
            _minute(minute, "a book minute key"): _book_from(entry, f"book minute {minute}")
            for minute, entry in _mapping(document["books"], "the cache's books").items()
        },
    )


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
        """Persist the cache atomically and verified, after its raw is durable.

        Sealed by :func:`state_to_document`, written through the same temp-file
        ``fsync``-and-rename the raw files use, and then read back through the
        same refusal every later start reads it through. A checkpoint the
        crash protocol relies on and nobody has looked at is not a checkpoint: a
        torn write, a disk that lied or a document that will not verify is found
        here, while the raw tail behind it can still simply be folded again,
        rather than on the next start. A file that fails is removed, not
        repaired.
        """
        path = self.cache_path(state.market, state.day)
        write_json_atomic(path, state_to_document(state))
        try:
            state_from_document(
                json.loads(path.read_text(encoding="utf-8")),
                market=state.market,
                day=state.day,
                hash_=state.contract_hash,
            )
        except (OSError, ValueError, NormalizeCacheError) as exc:
            self.drop(state.market, state.day)
            raise NormalizeCacheError(
                f"the cache written to {path} does not read back as the state it was "
                f"written from: {exc}. It has been removed, and the day will be folded "
                "again from the raw"
            ) from exc
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
        for: a different schema, a broken seal, an aggregate that is not the
        shape this build writes, another contract, a shorter raw file than the
        cursor claims, a day frozen since the cache was written, a record that
        will not parse. The slow path is the same code every other caller uses,
        so a fallback costs time and changes no value.

        The two durable writes happen in the order the module's crash protocol
        states: the raw is already fsynced, the verified checkpoint goes down
        next, and the derived output after it.
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
        rendered = self.render(state)
        try:
            self.save(state)
        except (NormalizeCacheError, RecorderSinkError) as exc:
            # A checkpoint that cannot be written must not stop a day from being
            # normalized. The cache is a memo, the raw it summarises is still
            # there, and no cache at all is a state the next pass already knows
            # how to recover from — it folds the day again.
            logger.warning("%s %s: the cache was not saved: %s", market, day, exc)
            status.reason = f"{status.reason}; cache not saved: {exc}"
        return self.normalizer.write_day(market, day, rendered, provenance=provenance)
