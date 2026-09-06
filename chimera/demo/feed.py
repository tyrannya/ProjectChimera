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
from chimera.recorder.normalize import (
    MinuteNormalizer,
    columns_for,
    digest,
)

__all__ = [
    "FeedCursor",
    "FeedError",
    "MarketState",
    "MinuteRecord",
    "plain_json",
    "settlement_from_row",
    "MINUTE_NS",
    "PERP_MARKET",
    "SETTLEMENT_INSTANT_FIELD",
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
            "index": render(self.index),
            "book_spot": render(dict(self.book_spot)),
            "book_perp": render(dict(self.book_perp)),
            "funding_rate_last": render(self.funding_rate_last),
            "next_funding_time_ms": self.next_funding_time_ms,
            "um_minute_digest": self.perp_digest,
            "spot_minute_digest": self.spot_digest,
        }


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
        self._days: dict[tuple[str, str], _Day | None] = {}
        self._settlements: list[Mapping[str, Any]] | None = None
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
        key = (market, day)
        if key in self._days:
            return self._days[key]
        parquet = self._normalizer.parquet_path(market, day)
        meta_path = self._normalizer.meta_path(market, day)
        loaded: _Day | None = None
        if parquet.is_file():
            frame = pd.read_parquet(parquet)
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
        self._days[key] = loaded
        return loaded

    def record(self, market: str, minute_open_ms: int) -> MinuteRecord | None:
        day = self._day(market, self.day_of(minute_open_ms))
        return None if day is None else day.record(minute_open_ms)

    # --- funding ----------------------------------------------------------
    def settlements(self) -> Sequence[Mapping[str, Any]]:
        """Every recorded funding settlement, oldest first. Read once."""
        if self._settlements is None:
            path = self._normalizer.settlements_path(PERP_MARKET)
            rows: list[Mapping[str, Any]] = []
            if path.is_file():
                for line in path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            rows.sort(key=lambda r: _settlement_ms(r))
            self._settlements = rows
        return self._settlements

    def last_settlement_at_or_before(self, minute_open_ms: int) -> Mapping[str, Any] | None:
        seen = None
        for row in self.settlements():
            if _settlement_ms(row) <= int(minute_open_ms):
                seen = row
            else:
                break
        return seen

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
