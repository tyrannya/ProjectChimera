"""Reading the recorder's normalized minutes, one minute at a time.

Section 2.2's D -> E edge and section 12.3's `feed.py` row. The recorder is a
separate process; this module treats its files as **read-only input** (section
8.2) and never writes into the recorder's tree.

Three things live here:

`MinuteRecord`
    One minute of one market, exactly as the normalized day stored it, together
    with the digest of that single row. The digest is
    :func:`chimera.recorder.normalize.digest` applied to the one-row frame --
    the frozen function, not a new hashing convention -- so section 9.1's
    "digests of the exact MinuteRecords read" is answered by the recorder's own
    definition of minute identity.

`MarketState`
    The frozen snapshot a rule is allowed to see: perpetual and spot OHLCV, top
    of book for both, mark and index at the minute close, the last funding
    settlement, the next funding time, and the feed ages. A state with any
    required field missing is `complete == False`, and section 2.2 says what
    follows: "rules do not evaluate; the log records INCOMPLETE_STATE".

`FeedCursor`
    Walks minutes forward from a persisted position. It never reads a minute
    twice and never invents one.

**Nothing here reads a clock.** Every instant comes from the files or from the
caller, so a replay of the same files produces the same states -- which is what
section 10's byte-for-byte parity comparison rests on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import pandas as pd

from chimera.carry.hedge import PerpSettlement
from chimera.recorder.contract import RecorderContract
from chimera.recorder.events import UM_KLINE_1M
from chimera.recorder.normalize import (
    MinuteNormalizer,
    columns_for,
    digest,
    read_funding_observation,
)

__all__ = [
    "FeedCursor",
    "FeedError",
    "FeedNotReady",
    "MarketState",
    "MinuteRecord",
    "fresh_through_ns",
    "plain_json",
    "settlement_from_row",
    "MINUTE_NS",
    "PERP_MARKET",
    "REQUIRED_STREAM",
    "SETTLEMENT_INSTANT_FIELD",
    "SETTLEMENT_QUERY_ALLOWANCE_MS",
    "SPOT_MARKET",
]

#: The two market keys the campaign's contract declares. Named rather than
#: discovered so a contract that grew a third market fails loudly here instead
#: of silently changing what a decision was made from.
PERP_MARKET = "um"
SPOT_MARKET = "spot"

MINUTE_NS = 60_000_000_000
_MS_TO_NS = 1_000_000

#: The settlement instant's field name in the recorder's own settlements file,
#: read off :meth:`chimera.recorder.events.FundingSettlement.to_settlement_record`
#: rather than guessed.
#:
#: It is named here because getting it wrong is silent. Until PR-10R this module
#: read ``settlement_ms``, a key the recorder has never written: every row
#: therefore defaulted to instant ``0``, ``settlements()`` "sorted" them all
#: equal, and ``last_settlement_at_or_before`` answered every minute with the
#: last row of the file -- so a decision's ``funding_last`` input was whatever
#: settlement happened to be last in the file rather than the one in force. The
#: synthetic fixture wrote the same wrong key, which is why no test saw it.
SETTLEMENT_INSTANT_FIELD = "funding_time_ms"


class FeedError(RuntimeError):
    """The feed cannot answer, and guessing would be worse than stopping."""


class FeedNotReady(FeedError):
    """The feed cannot answer YET: ask again after the recorder has written more.

    R1-g. Raised for a normalized day that failed to read while it was changing
    on disk. The recorder rewrites a day's parquet in place, not by atomic
    replacement (that is R1-h's), so a read that lands inside a rewrite sees a
    truncated file; with minutes published within seconds and a backlog taken
    whole, a long catch-up is almost certain to land in one. The runner waits on
    this -- it decides nothing from a file it could not read -- and a file that
    fails again unchanged is a :class:`FeedError` and a halt, as it always was.
    """


#: R1-g: how long after a scheduled funding instant a funding query must have
#: been made before its silence may count as "no settlement". The recorder's
#: own scheduled poll comes `chimera.recorder.rest.FUNDING_POLL_DELAY_S` (60 s)
#: after each instant, because a settlement's row is not published at the
#: instant itself; half of that lets the scheduled poll clear it even when its
#: timer fires a little early, and keeps a query made at the instant -- when the
#: row cannot exist yet -- from ever counting. A literal rather than an import:
#: this package imports nothing that can reach a network.
SETTLEMENT_QUERY_ALLOWANCE_MS = 30_000


#: The one recorder stream R1-f's READY gate reads liveness from: the
#: perpetual's klines, the market `FeedCursor.latest_minute_ms` walks. No other
#: stream -- however busy -- can stand in for it.
REQUIRED_STREAM = UM_KLINE_1M


def fresh_through_ns(heartbeat: Mapping[str, Any] | None) -> int | None:
    """The instant up to which the recorder's heartbeat vouches for the feed.

    ``min(heartbeat_ns, last_event_ns of REQUIRED_STREAM)``, or None when the
    heartbeat cannot vouch at all: no document, no such stream, a stream that is
    not ``up`` (the recorder's own ``connected and not halted``), or a missing
    stamp. The min is what keeps the two facts from masking each other: a
    process still beating cannot make a dead stream fresh, and a stream's last
    event cannot make a heartbeat file that stopped being rewritten fresh.

    A kline's ``last_event_ns`` is its minute's OPEN (the recorder stamps a
    candle by its open, partial frames included), used as-is: a lower bound on
    when the exchange last spoke. Its close would lie ahead of a forming candle.

    Pure: the caller subtracts this from its own operational clock, so an age
    keeps growing after the heartbeat stops rather than freezing at the age the
    last heartbeat reported.
    """
    if heartbeat is None:
        return None
    beat = heartbeat.get("heartbeat_ns")
    entry = next(
        (
            s
            for s in heartbeat.get("streams") or ()
            if isinstance(s, Mapping) and s.get("stream") == REQUIRED_STREAM
        ),
        None,
    )
    if type(beat) is not int or entry is None or entry.get("up") is not True:
        return None
    last = entry.get("last_event_ns")
    if type(last) is not int:
        return None
    return min(beat, last)


def _decimal(value: Any) -> Decimal | None:
    """A stored number as a Decimal, via ``str`` so no binary float leaks in."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except Exception:  # pragma: no cover - defensive
        return None


def plain_json(value: Any) -> Any:
    """Coerce a value read out of a parquet frame into plain JSON types.

    A number read from a normalized day arrives as a numpy or pandas scalar,
    which ``json.dumps`` refuses and which
    :func:`chimera.demo.decision_log.validate_payload` rejects by name. Section
    9.2 requires a record to serialize canonically and ``inputs_hash`` is taken
    over exactly these mappings, so an unconverted scalar does not merely look
    untidy -- it raises at the moment a decision is being written down.

    Exposed rather than kept private because the runner puts the same book
    blocks into a record. Two coercions would be two chances to disagree about
    what a `Decimal` renders as, and the two are compared byte for byte by
    section 10's parity check.
    """
    if value is None or value is pd.NA:
        return None
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Mapping):
        return {k: plain_json(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [plain_json(v) for v in value]
    if isinstance(value, (bool, int, float, str)):
        return value
    item = getattr(value, "item", None)
    if callable(item):  # numpy / pandas scalar
        try:
            return plain_json(item())
        except (ValueError, TypeError):  # pragma: no cover - defensive
            pass
    return str(value)


@dataclass(frozen=True)
class MinuteRecord:
    """One minute of one market, plus the identity of that minute."""

    market: str
    minute_open_ms: int
    row: Mapping[str, Any]
    row_digest: str

    @property
    def minute_ns(self) -> int:
        return self.minute_open_ms * _MS_TO_NS

    def value(self, column: str) -> Any:
        return self.row.get(column)

    def decimal(self, column: str) -> Decimal | None:
        return _decimal(self.row.get(column))

    @property
    def has_book(self) -> bool:
        return bool(self.row.get("book_present"))


@dataclass(frozen=True)
class MarketState:
    """What a rule may see about one minute. Frozen, and complete or not.

    Satisfies `chimera.carry.hedge.CarryMarketState` structurally, which is how
    PR-08 types its market snapshot without importing this module: section 13
    makes PR-10 depend on PR-08, so the import may only ever run that way.
    """

    minute_ns: int
    minute: str
    complete: bool
    missing: tuple[str, ...] = ()

    spot_close: Decimal | None = None
    perp_close: Decimal | None = None
    mark: Decimal | None = None
    #: The minute's mark HIGH, which section 6.7's liquidation test is written
    #: against: ``equity < Q * mark_high * maintenance_margin_rate``. The
    #: recorder has always stored it -- ``columns_for("um")`` carries
    #: ``mark_open/high/low/close`` -- and dropping it here made the demo's
    #: liquidation check read the CLOSE, which is never above the high, so the
    #: threshold sat strictly below the adopted one and the deviation was in the
    #: unsafe direction.
    mark_high: Decimal | None = None
    index: Decimal | None = None

    spot_ohlcv: Mapping[str, Decimal | None] = field(default_factory=dict)
    perp_ohlcv: Mapping[str, Decimal | None] = field(default_factory=dict)
    book_spot: Mapping[str, Any] = field(default_factory=dict)
    book_perp: Mapping[str, Any] = field(default_factory=dict)

    funding_rate_last: Decimal | None = None
    next_funding_time_ms: int | None = None

    perp_digest: str = ""
    spot_digest: str = ""
    feed_age_ns: Mapping[str, int] = field(default_factory=dict)

    @property
    def basis(self) -> Decimal | None:
        if self.perp_close is None or self.spot_close is None:
            return None
        return self.perp_close - self.spot_close

    def canonical(self) -> dict[str, Any]:
        """The shape section 9.1's ``inputs_hash`` is taken over.

        Only what a decision may depend on. Ordering and rendering follow
        section 9.2, so the same minute produces the same bytes in live and in
        replay.
        """

        render = plain_json

        return {
            "minute": self.minute,
            "complete": self.complete,
            "spot_close": render(self.spot_close),
            "perp_close": render(self.perp_close),
            "mark": render(self.mark),
            # In the hashed inputs because it DECIDES something: section 6.7's
            # liquidation touch is evaluated against it, so a decision minute
            # that omitted it would hash away one of its own inputs.
            "mark_high": render(self.mark_high),
            "index": render(self.index),
            "book_spot": render(dict(self.book_spot)),
            "book_perp": render(dict(self.book_perp)),
            "funding_rate_last": render(self.funding_rate_last),
            "next_funding_time_ms": self.next_funding_time_ms,
            "um_minute_digest": self.perp_digest,
            "spot_minute_digest": self.spot_digest,
        }


def _file_stamp(path: Path) -> tuple[int, int] | None:
    """``(st_size, st_mtime_ns)``, or ``None`` for a file that is not there."""
    try:
        stat = path.stat()
    except OSError:
        return None
    return (stat.st_size, stat.st_mtime_ns)


def _settlement_ms(row: Mapping[str, Any]) -> int:
    """One settlement row's instant, in milliseconds. Refuses a row without one.

    A row whose instant cannot be read is not sorted to zero and not skipped: a
    settlements file the runner cannot place in time is a funding input it cannot
    use, and reading past it would leave the campaign charging some settlements
    and not others with nothing recording which.
    """
    value = row.get(SETTLEMENT_INSTANT_FIELD)
    if value is None:
        raise FeedError(
            f"a funding settlement row has no {SETTLEMENT_INSTANT_FIELD!r}: {row!r}. "
            "That is the field the recorder writes for the settlement instant"
        )
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise FeedError(
            f"funding settlement {SETTLEMENT_INSTANT_FIELD}={value!r} is not an integer "
            "number of milliseconds"
        ) from exc


def settlement_from_row(
    row: Mapping[str, Any], *, position_symbol: str, venue_symbol: str
) -> PerpSettlement:
    """One recorded settlement row, translated for the perpetual leg to book.

    Three translations happen here and each of them is a place a silent wrong
    answer was available:

    **The symbol.** The row carries the VENUE's spelling (``BTCUSDT``) and the
    position is keyed by the executor's (``BTC/USDT:USDT``).
    :func:`chimera.futures.accounting.funding_cash_flow` looks the position up by
    the event's symbol, so an untranslated row finds no position, reads as flat
    and books zero -- a settlement the log would report as settled and the ledger
    would show as free. The row's own spelling is checked against the contract's
    market symbol first, so a settlements file for another instrument is refused
    rather than relabelled.

    **The instant.** ``funding_time_ms`` is milliseconds; the carry package's
    window and dedup key are nanoseconds.

    **The mark price.** The recorder records it when the venue publishes it and
    writes ``null`` when it does not (amendment A4: the funding archive publishes
    no settlement mark price, and the recorded one is never reconstructed from
    mark-price candles). A settlement without one cannot be given a notional, so
    it is refused here. Substituting this minute's mark would be exactly the
    "synthesize funding from markPrice" that A4 forbids.
    """
    published = str(row.get("symbol", "")).upper()
    if published != venue_symbol.upper():
        raise FeedError(
            f"funding settlement names symbol {published!r}, and this campaign's "
            f"contract records {venue_symbol.upper()!r}. A settlement for another "
            "instrument is refused rather than booked against this position"
        )
    instant_ms = _settlement_ms(row)
    rate = _decimal(row.get("funding_rate"))
    if rate is None:
        raise FeedError(
            f"funding settlement at {instant_ms} has no funding_rate; a settlement "
            "without a realised rate cannot be booked"
        )
    mark = _decimal(row.get("mark_price"))
    if mark is None:
        raise FeedError(
            f"funding settlement at {instant_ms} has no mark_price. The notional is "
            "the quantity at the settlement's own mark, and this minute's mark is a "
            "different number recorded for a different instant"
        )
    return PerpSettlement(
        symbol=position_symbol,
        rate=rate,
        mark_price=mark,
        # Section 5.3: the settlement id IS the settlement instant in
        # milliseconds. The futures ledger deduplicates on it and persists it, so
        # it may never become anything else.
        settlement_id=str(instant_ms),
        instant_ns=instant_ms * _MS_TO_NS,
    )


def _row_digest(frame: pd.DataFrame, index: int, market: str) -> str:
    """The recorder's own digest, applied to the single row at ``index``.

    :func:`chimera.recorder.normalize.digest` validates the column set and then
    hashes a header, the row count and each column's values, so it is defined
    for a frame of any length. Applying it to a one-row slice is therefore the
    recorder's existing definition of minute identity narrowed to one minute --
    section 9.1 asks for "digests of the exact MinuteRecords read", and this
    answers with the frozen function rather than a convention invented here.

    Returned bare (64 hex, no ``sha256:`` prefix), which is what
    ``chimera.demo.decision_log`` expects of a field whose name ends in
    ``_digest``.
    """
    return digest(frame.iloc[index : index + 1].reset_index(drop=True), market=market)


class _Day:
    """One market's normalized day, read once and indexed by minute."""

    __slots__ = ("market", "day", "frame", "_by_minute", "meta")

    def __init__(self, market: str, day: str, frame: pd.DataFrame, meta: Mapping[str, Any]):
        self.market = market
        self.day = day
        self.frame = frame
        self.meta = meta
        minutes = frame["minute_open_ms"].tolist()
        self._by_minute = {int(m): i for i, m in enumerate(minutes)}

    def record(self, minute_open_ms: int) -> MinuteRecord | None:
        index = self._by_minute.get(int(minute_open_ms))
        if index is None:
            return None
        row = {k: v for k, v in self.frame.iloc[index].items()}
        return MinuteRecord(
            market=self.market,
            minute_open_ms=int(minute_open_ms),
            row=row,
            row_digest=_row_digest(self.frame, index, self.market),
        )


class FeedCursor:
    """Walks the recorder's normalized minutes forward, once each.

    ``FeedCursor(root, contract, state)`` is section 12.3's signature. ``state``
    is the runner's persisted cursor -- section 2.2 names its field
    ``last_minute_processed`` -- so a restart resumes at the next unprocessed
    minute rather than replaying one or skipping one.
    """

    def __init__(
        self,
        root: str | Path,
        contract: RecorderContract,
        state: Mapping[str, Any] | None = None,
    ) -> None:
        self.root = Path(root)
        self.contract = contract
        self._normalizer = MinuteNormalizer(self.root, contract)
        #: Each day as last read, keyed by the stamps of its two files. See `_day`.
        self._days: dict[tuple[str, str], tuple[tuple[Any, Any], _Day | None]] = {}
        self._settlements: list[Mapping[str, Any]] | None = None
        #: ``(size, mtime_ns)`` of the settlements file when it was last read.
        self._settlements_stamp: tuple[int, int] | None = None
        #: The stamps of a day that last failed to read. See `_day`.
        self._unreadable: dict[tuple[str, str], tuple[Any, Any]] = {}
        state = dict(state or {})
        last = state.get("last_minute_processed")
        self._last_minute_ms: int | None = int(last) if last is not None else None

    # --- position ---------------------------------------------------------
    @property
    def last_minute_processed(self) -> int | None:
        return self._last_minute_ms

    def to_dict(self) -> dict[str, Any]:
        return {"last_minute_processed": self._last_minute_ms}

    def mark_processed(self, minute_open_ms: int) -> None:
        """Advance the cursor. Monotone: an earlier minute never moves it back."""
        minute_open_ms = int(minute_open_ms)
        if self._last_minute_ms is None or minute_open_ms > self._last_minute_ms:
            self._last_minute_ms = minute_open_ms

    # --- files ------------------------------------------------------------
    @staticmethod
    def day_of(minute_open_ms: int) -> str:
        return pd.Timestamp(int(minute_open_ms), unit="ms", tz="UTC").strftime("%Y-%m-%d")

    def _day(self, market: str, day: str) -> _Day | None:
        """One market-day as the recorder has it NOW, re-read only when it changed.

        Cached against the ``(st_size, st_mtime_ns)`` of the parquet and of its
        metadata, the same rule :meth:`settlements` uses and for the same reason:
        it used to be cached for the life of the cursor, and the cursor lives as
        long as the runner. A process that outlives one catch-up pass (R1-d's
        daemon) would then never see a minute the recorder published after the
        first read -- neither a minute added to a day it had already read, nor a
        day whose file did not yet exist, because the ABSENCE was cached too. It
        would look alive and decide nothing.

        The stamp decides only WHEN a file is re-read, never what a minute says:
        a minute's row is the file's row, so a replay over the finished files
        reads the same rows. An unchanged file is not read again.

        A parquet that fails to read raises :class:`FeedNotReady` the first time
        (R1-g: it is most likely mid-rewrite), and :class:`FeedError` if it
        fails again with the stamps unchanged -- nothing is rewriting it, so it
        is broken rather than busy. Nothing from a failed read is cached.
        """
        key = (market, day)
        parquet = self._normalizer.parquet_path(market, day)
        meta_path = self._normalizer.meta_path(market, day)
        stamp = (_file_stamp(parquet), _file_stamp(meta_path))
        cached = self._days.get(key)
        if cached is not None and cached[0] == stamp:
            return cached[1]
        loaded: _Day | None = None
        if parquet.is_file():
            try:
                frame = pd.read_parquet(parquet)
            except Exception as exc:
                if self._unreadable.get(key) == stamp:
                    raise FeedError(
                        f"{parquet} cannot be read and has not changed since it last "
                        f"failed: {exc}"
                    ) from exc
                self._unreadable[key] = stamp
                raise FeedNotReady(
                    f"{parquet} could not be read, most likely mid-rewrite: {exc}"
                ) from exc
            self._unreadable.pop(key, None)
            names = [spec.name for spec in columns_for(market)]
            if list(frame.columns) != names:
                raise FeedError(
                    f"{parquet} has columns {list(frame.columns)}, expected {names}. "
                    "A day whose schema is not the recorder's cannot be read as one"
                )
            meta: Mapping[str, Any] = {}
            if meta_path.is_file():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            loaded = _Day(market, day, frame, meta)
        self._days[key] = (stamp, loaded)
        return loaded

    def record(self, market: str, minute_open_ms: int) -> MinuteRecord | None:
        day = self._day(market, self.day_of(minute_open_ms))
        return None if day is None else day.record(minute_open_ms)

    # --- funding ----------------------------------------------------------
    def settlements(self) -> Sequence[Mapping[str, Any]]:
        """Every recorded funding settlement, oldest first.

        Re-read whenever the file has changed, and cached in between. It used to
        be read exactly once for the life of the cursor -- and the cursor lives
        as long as the runner -- so every settlement the recorder wrote after
        that first read was invisible to the process. On the funding path that
        is not a stale number: a settlement missed on the minute it belongs to is
        booked by a later process at a later minute, so the live log and a replay
        of the same files disagree about which minute it fell in, which is a
        parity divergence with no cause visible in either log.

        Staleness is judged on ``(st_size, st_mtime_ns)``, and that is not a
        decision input: it decides only WHEN the file is re-read. Which
        settlements are booked stays section 6.9's window plus the persisted
        dedup, so a replay over the finished file books exactly the same set at
        exactly the same minutes.

        Two rows for one settlement instant that DISAGREE are refused. The
        recorder already refuses to write them (a settlement is published once),
        and every other unusable row on this path is refused rather than
        resolved by file order; a self-contradicting one would otherwise be
        booked as whichever copy sorted first.
        """
        path = self._normalizer.settlements_path(PERP_MARKET)
        try:
            stat = path.stat()
            stamp: tuple[int, int] | None = (stat.st_size, stat.st_mtime_ns)
        except OSError:
            stamp = None
        if self._settlements is not None and stamp == self._settlements_stamp:
            return self._settlements

        rows: list[Mapping[str, Any]] = []
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        rows.sort(key=lambda r: _settlement_ms(r))
        seen: dict[int, Mapping[str, Any]] = {}
        for row in rows:
            instant = _settlement_ms(row)
            first = seen.get(instant)
            if first is not None and dict(first) != dict(row):
                raise FeedError(
                    f"two funding settlement rows for instant {instant} disagree: "
                    f"{first!r} versus {row!r}. A settlement is published once, so the "
                    "runner refuses the file rather than booking whichever copy sorted "
                    "first"
                )
            seen.setdefault(instant, row)
        self._settlements = rows
        self._settlements_stamp = stamp
        return self._settlements

    def last_settlement_at_or_before(self, minute_open_ms: int) -> Mapping[str, Any] | None:
        seen = None
        for row in self.settlements():
            if _settlement_ms(row) <= int(minute_open_ms):
                seen = row
            else:
                break
        return seen

    def funding_observation(self) -> tuple[int, int] | None:
        """The recorder's funding observation ``(from_ms, through_ms)``, or None.

        The query windows the recorder's successful, complete funding polls
        covered -- what it ASKED, not what happened. See
        :func:`chimera.recorder.normalize.funding_observation_document`.
        """
        return read_funding_observation(
            self._normalizer.funding_observation_path(PERP_MARKET), PERP_MARKET
        )

    def funding_pending(self, minute_open_ms: int) -> int | None:
        """The funding instant this minute must wait for, or None (R1-g).

        A minute depends on every settlement at or before its close: it books
        those in its window, and the next one reads the last of them as its
        ``funding_rate_last``. A row that lands after the minute was decided
        would therefore be booked at a LATER minute live, and at this one by a
        replay of the finished files. So a minute waits while the latest funding
        instant the venue scheduled at or before its close is unresolved.

        *Scheduled*: the perpetual's recorded ``next_funding_time_ms`` -- the
        venue's own announcement -- on this minute's row and every earlier one
        of this day and the day before, so a minute whose own row cannot say
        (its mark missing, or the row itself) still sees the instant an earlier
        row announced.

        *Resolved*, by recorder facts only: the instant's own settlement row
        exists -- stamped within a minute after it, so a venue that stamps its
        row a few milliseconds off the schedule still answers it, and a LATER
        settlement never does -- or the recorder's funding observation covers
        the instant with `SETTLEMENT_QUERY_ALLOWANCE_MS` to spare: it asked late
        enough that a row would have been there, and none was. An absent,
        unreadable, too-early or gapped observation resolves nothing: absence of
        a row is never read as absence of a settlement.
        """
        close_ms = int(minute_open_ms) + 60_000
        scheduled = self._scheduled_through(int(minute_open_ms), close_ms)
        if scheduled is None:
            return None
        if any(
            scheduled <= _settlement_ms(row) < scheduled + 60_000 for row in self.settlements()
        ):
            return None
        observed = self.funding_observation()
        if (
            observed is not None
            and observed[0] <= scheduled
            and scheduled + SETTLEMENT_QUERY_ALLOWANCE_MS <= observed[1]
        ):
            return None
        return scheduled

    def _scheduled_through(self, minute_open_ms: int, close_ms: int) -> int | None:
        """The latest instant announced on perpetual rows up to this minute, <= close."""
        latest: int | None = None
        for back in (0, 1):
            day = self._day(PERP_MARKET, self.day_of(minute_open_ms - back * 86_400_000))
            if day is None:
                continue
            frame = day.frame
            announced = frame.loc[
                frame["minute_open_ms"] <= minute_open_ms, "next_funding_time_ms"
            ].dropna()
            announced = announced[announced <= close_ms]
            if len(announced):
                value = int(announced.max())
                latest = value if latest is None else max(latest, value)
        return latest

    # --- walking ----------------------------------------------------------
    def next_minute_ms(self, *, now_ms: int | None = None) -> int | None:
        """The next minute to process, or None if there is nothing new.

        The next minute is one minute after the cursor. With no cursor yet, it
        is the earliest minute the perpetual's days actually carry -- a campaign
        starts where the evidence starts, never at an invented instant.
        """
        if self._last_minute_ms is not None:
            candidate = self._last_minute_ms + 60_000
        else:
            candidate = self._earliest_minute_ms()
            if candidate is None:
                return None
        if now_ms is not None and candidate > int(now_ms):
            return None
        return candidate

    def latest_minute_ms(self) -> int | None:
        """The newest minute the perpetual's days actually carry, or None.

        The mirror of :meth:`_earliest_minute_ms`, and read off the day frames
        rather than by probing minute by minute: catch-up asks for it once per
        pending minute, and a linear probe made that quadratic.
        """
        market_dir = self._normalizer.market_dir(PERP_MARKET)
        if not market_dir.is_dir():
            return None
        for parquet in sorted(market_dir.glob("*.parquet"), reverse=True):
            day = self._day(PERP_MARKET, parquet.stem)
            if day is not None and len(day.frame):
                return int(day.frame["minute_open_ms"].iloc[-1])
        return None

    def _earliest_minute_ms(self) -> int | None:
        market_dir = self._normalizer.market_dir(PERP_MARKET)
        if not market_dir.is_dir():
            return None
        for parquet in sorted(market_dir.glob("*.parquet")):
            day = self._day(PERP_MARKET, parquet.stem)
            if day is not None and len(day.frame):
                return int(day.frame["minute_open_ms"].iloc[0])
        return None

    def state_for(self, minute_open_ms: int, *, now_ns: int | None = None) -> MarketState:
        """Build the `MarketState` for one minute. Missing input is not fatal.

        A minute the recorder never captured, or captured without a book, yields
        `complete == False` and the names of what was missing. Section 2.2 is
        explicit that this is a state the runner logs and steps over, not an
        error it raises: an incomplete minute is evidence about the feed.
        """
        minute_ns = int(minute_open_ms) * _MS_TO_NS
        minute_iso = pd.Timestamp(int(minute_open_ms), unit="ms", tz="UTC").isoformat()
        perp = self.record(PERP_MARKET, minute_open_ms)
        spot = self.record(SPOT_MARKET, minute_open_ms)

        missing: list[str] = []
        if perp is None:
            missing.append("um_minute")
        if spot is None:
            missing.append("spot_minute")
        if perp is not None and not perp.has_book:
            missing.append("um_book")
        if spot is not None and not spot.has_book:
            missing.append("spot_book")
        if perp is not None and not bool(perp.value("mark_present")):
            missing.append("um_mark")

        def ohlcv(record: MinuteRecord | None) -> dict[str, Decimal | None]:
            if record is None:
                return {}
            return {
                key: record.decimal(f"kline_{key}")
                for key in ("open", "high", "low", "close", "volume")
            }

        def book(record: MinuteRecord | None) -> dict[str, Any]:
            if record is None or not record.has_book:
                return {}
            return {
                "bid": record.decimal("book_bid"),
                "bid_qty": record.decimal("book_bid_qty"),
                "ask": record.decimal("book_ask"),
                "ask_qty": record.decimal("book_ask_qty"),
                "update_id": record.value("book_update_id"),
                "canonical_ms": record.value("book_canonical_ms"),
                "time_basis": record.value("book_time_basis"),
            }

        ages: dict[str, int] = {}
        if now_ns is not None:
            for name, record in (("um", perp), ("spot", spot)):
                if record is not None:
                    ages[name] = max(0, int(now_ns) - (record.minute_ns + MINUTE_NS))

        settlement = self.last_settlement_at_or_before(minute_open_ms)
        return MarketState(
            minute_ns=minute_ns,
            minute=minute_iso,
            complete=not missing,
            missing=tuple(missing),
            spot_close=spot.decimal("kline_close") if spot else None,
            perp_close=perp.decimal("kline_close") if perp else None,
            mark=perp.decimal("mark_close") if perp else None,
            mark_high=perp.decimal("mark_high") if perp else None,
            index=perp.decimal("index_close") if perp else None,
            spot_ohlcv=ohlcv(spot),
            perp_ohlcv=ohlcv(perp),
            book_spot=book(spot),
            book_perp=book(perp),
            funding_rate_last=(
                _decimal(settlement.get("funding_rate")) if settlement else None
            ),
            next_funding_time_ms=(
                int(perp.value("next_funding_time_ms"))
                if perp is not None and perp.value("next_funding_time_ms") is not None
                else None
            ),
            perp_digest=perp.row_digest if perp else "",
            spot_digest=spot.row_digest if spot else "",
            feed_age_ns=ages,
        )

    def walk(self, *, now_ms: int | None = None) -> Iterator[MarketState]:
        """Yield states forward from the cursor. The caller marks each processed."""
        while True:
            minute = self.next_minute_ms(now_ms=now_ms)
            if minute is None:
                return
            state = self.state_for(minute)
            if (
                self.record(PERP_MARKET, minute) is None
                and self.record(SPOT_MARKET, minute) is None
            ):
                # Neither market has this minute at all. Only step over it once a
                # LATER minute exists, so a cursor sitting at the live edge waits
                # for the recorder rather than racing past an unwritten minute.
                if not self._anything_after(minute):
                    return
            yield state

    def _anything_after(self, minute_open_ms: int) -> bool:
        for market in (PERP_MARKET, SPOT_MARKET):
            market_dir = self._normalizer.market_dir(market)
            if not market_dir.is_dir():
                continue
            for parquet in sorted(market_dir.glob("*.parquet")):
                day = self._day(market, parquet.stem)
                if day is None or not len(day.frame):
                    continue
                if int(day.frame["minute_open_ms"].iloc[-1]) > int(minute_open_ms):
                    return True
        return False
