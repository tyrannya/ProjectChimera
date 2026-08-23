"""Hourly sufficient statistics of individual trades: P3's research source.

Research checkpoints v4, P2a, P2b and P2c all read the same five numbers per
hour. P3 asks whether *individual executions* carry information those five
cannot, so it needs a genuinely different source: Binance's public
``aggTrades`` archive for BTCUSDT, which records one row per aggressed order
rather than one row per hour.

**Why the committed source is an hourly aggregate and not the trades.**
2020-01-01 to the last hour research may reach is on the order of a billion
aggTrades. That is not committable, not reviewable, and not something CI can
re-read. So the acquisition step streams the raw archives once and writes the
*sufficient statistics* every ``microstructure_v1`` feature is defined over —
counts, sums, per-second dispersion, quarter buckets, a fixed trade-size
histogram and a per-hour volume-at-price histogram. The raw archives are
identified by name, byte count and SHA-256 in the manifest, so the derivation
is auditable by anyone who can fetch them; what this repository commits is the
summary, and it says so rather than calling itself the trades.

The word *sufficient* is a claim with a test behind it: every feature in
``docs/microstructure_v1.md`` is a function of these columns alone, and
:mod:`nn.microstructure` cannot read anything else because nothing else is
there. A feature that needed a trade-level quantity this table does not carry
would have to be added here first, which is the point — the aggregation is
predeclared, and a later feature cannot quietly widen what the source saw.

**Aggressor semantics.** Binance's ``is_buyer_maker`` says the *buyer* sat on
the book, so the taker was the seller and the execution was an aggressive
**sell**. ``false`` is an aggressive **buy**. Everything this module calls
``buy_`` is taker-buy under that reading, which is the only reading the field
supports; it is recorded in the manifest rather than assumed by the reader.

**Epochs are normalised before anything else happens.** Binance spot archives
carried ``transact_time`` in *milliseconds* historically and switched to
*microseconds* for data from 2025-01-01 onward. A month read in the wrong unit
lands 55 years from where it belongs, and a fingerprint taken over it would be
a confident identity for the wrong hours. :func:`resolve_epoch_unit` therefore
decides the unit *per archive* by requiring the whole archive to fall inside
the calendar period the archive is named for, and fails closed when no
supported unit does. Every timestamp downstream of that is int64 UTC
nanoseconds — one canonical representation, fingerprinted as such.

**Styx.** The aggregator is constructed with the sealed instant and refuses a
trade at or after it, and refuses to open an hour bucket that could contain
one. Both guards are unconditional and neither prints the row it rejected.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd

#: Names the meaning of an hourly trade-aggregate table. Hashed with the data,
#: so a future change to *what* is aggregated is a change of identity rather
#: than a silent reinterpretation of a digest already recorded in an artifact.
TRADE_AGGREGATE_SCHEMA = "chimera.trade-aggregate/1"

#: Nanoseconds in the bar this table aggregates to. P3 predicts on the existing
#: BTC/USDT 1h research problem, so the bucket is the research timeframe and is
#: not a parameter: a different bucket is a different alignment to the spine.
HOUR_NS = 3_600_000_000_000
SECOND_NS = 1_000_000_000

#: Seconds per bucket, and quarters per bucket. Both are consequences of the
#: hour above rather than independent settings.
SECONDS_PER_HOUR = 3600
QUARTERS = 4

#: Trade-size histogram edges, in base asset (BTC), half a decade apart.
#:
#: Written as literals rather than computed as ``10 ** arange(-4, 1.5, 0.5)``
#: on purpose. ``pow`` is not bit-identical across libm implementations, and an
#: edge that differs in its last bit moves a trade sitting exactly on it into
#: the neighbouring bin — which would make the semantic fingerprint of the same
#: archives depend on the machine that read them.
#:
#: Absolute rather than price-adjusted, and deliberately so: BTC was near
#: $7,000 at the start of the research region and near $100,000 at the end, so
#: a fixed BTC-denominated grid drifts in dollar terms across the sample. Every
#: feature built from this histogram is a *share* measured against a trailing
#: window of the histogram itself (docs/microstructure_v1.md §C), so the drift
#: is normalised away by the feature rather than by the bin.
SIZE_BIN_EDGES: tuple[float, ...] = (
    0.0001,
    0.00031622776601683794,
    0.001,
    0.0031622776601683794,
    0.01,
    0.031622776601683794,
    0.1,
    0.31622776601683794,
    1.0,
    3.1622776601683795,
    10.0,
)
#: One more bin than edges: everything at or below the first edge, one bin per
#: interval, and everything above the last.
SIZE_BINS = len(SIZE_BIN_EDGES) + 1

#: Equal-width price bins spanning one hour's own traded range. Eight is enough
#: to separate a single dominant node from a flat distribution and few enough
#: that a bin still holds meaningful volume in a quiet hour.
PRICE_BINS = 8

#: Epoch units a Binance archive is allowed to be in, largest first. Seconds are
#: absent deliberately: no Binance trade archive uses them, and admitting a unit
#: no source produces would only widen what :func:`resolve_epoch_unit` can be
#: talked into.
EPOCH_UNITS: dict[str, int] = {"ms": 1_000_000, "us": 1_000, "ns": 1}


class TradeAggregationError(ValueError):
    """The trade stream cannot be aggregated into an honest hourly table."""


class StyxBreach(TradeAggregationError):
    """A trade at or after the sealed instant reached the aggregator.

    Its own class because it is the one failure whose *content* must never be
    reported. The message counts rows and names the instant; it never carries a
    price, a quantity, a trade id or a timestamp from the offending rows.
    """


def _quarter_columns(prefix: str) -> tuple[str, ...]:
    return tuple(f"q{i}_{prefix}" for i in range(QUARTERS))


def _indexed_columns(prefix: str, count: int) -> tuple[str, ...]:
    return tuple(f"{prefix}_{i}" for i in range(count))


#: Every column of the aggregate table, in the order the digest reads them, with
#: the canonical kind :func:`nn.data_fingerprint._canonical_chunk` normalises to.
#:
#: Counts are ``i8`` and money/size are ``f8``. That split is not cosmetic: a
#: count is exact and a float sum is not, so recording a count as a float would
#: make two readers of the same archive able to disagree in the last bit about
#: something that has an exact answer.
AGGREGATE_COLUMN_KINDS: tuple[tuple[str, str], ...] = (
    ("date", "t8"),
    ("n_trades", "i8"),
    ("buy_trades", "i8"),
    ("qty", "f8"),
    ("buy_qty", "f8"),
    ("notional", "f8"),
    ("buy_notional", "f8"),
    ("first_price", "f8"),
    ("last_price", "f8"),
    ("high_price", "f8"),
    ("low_price", "f8"),
    ("active_seconds", "i8"),
    ("second_count_sq_sum", "i8"),
    ("max_second_trades", "i8"),
    ("max_empty_run_seconds", "i8"),
    *((name, "i8") for name in _quarter_columns("trades")),
    *((name, "f8") for name in _quarter_columns("qty")),
    *((name, "f8") for name in _quarter_columns("buy_qty")),
    *((name, "i8") for name in _indexed_columns("size_count", SIZE_BINS)),
    *((name, "f8") for name in _indexed_columns("size_qty", SIZE_BINS)),
    *((name, "f8") for name in _indexed_columns("price_qty", PRICE_BINS)),
)

#: The column names alone, in the same order.
AGGREGATE_COLUMNS: tuple[str, ...] = tuple(name for name, _ in AGGREGATE_COLUMN_KINDS)


def aggregation_spec() -> dict[str, Any]:
    """What this module computes, as data, for the manifest and every artifact.

    Recorded rather than described because it is part of what a P3 number
    *means*: two runs that bucketed trades differently, split the aggressor the
    other way round, or used another size grid are not two measurements of one
    thing. :func:`aggregation_spec_hash` reduces it to one identity a cell can
    carry and a comparison can refuse to join across.
    """
    return {
        "aggregate_schema": TRADE_AGGREGATE_SCHEMA,
        "bucket_ns": HOUR_NS,
        "bucket_convention": (
            "row 'date' is the hour's OPEN instant; the row aggregates every trade "
            "with timestamp in [date, date + 1h), which is exactly the trades that "
            "produced the OHLCV candle carrying the same 'date'"
        ),
        "timezone": "UTC",
        "epoch_representation": "int64 nanoseconds since 1970-01-01T00:00:00Z",
        "aggressor_rule": (
            "is_buyer_maker=false is an aggressive BUY (the taker bought); "
            "is_buyer_maker=true is an aggressive SELL"
        ),
        "price_dtype": "IEEE-754 float64 parsed from the archive's decimal text",
        "qty_dtype": "IEEE-754 float64 parsed from the archive's decimal text",
        "identity_column": "agg_trade_id",
        "ordering_rule": (
            "strictly increasing agg_trade_id; timestamps non-decreasing but not "
            "unique, since many aggressed orders share a millisecond"
        ),
        "duplicate_rule": "an agg_trade_id at or below the previous one is a duplicate",
        "size_bin_edges": list(SIZE_BIN_EDGES),
        "size_bins": SIZE_BINS,
        "price_bins": PRICE_BINS,
        "price_bin_rule": (
            f"{PRICE_BINS} equal-width bins spanning [low_price, high_price] of the "
            "hour itself; a degenerate hour with one traded price puts all of its "
            "quantity in bin 0"
        ),
        "quarters": QUARTERS,
        "seconds_per_bucket": SECONDS_PER_HOUR,
        "columns": [list(entry) for entry in AGGREGATE_COLUMN_KINDS],
    }


def aggregation_spec_hash() -> str:
    """SHA-256 over :func:`aggregation_spec`, canonically encoded."""
    material = json.dumps(
        aggregation_spec(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


#: The epoch-nanosecond range ``pandas`` can represent as a timestamp. Values
#: outside it are a real input this module has to describe rather than a case it
#: is entitled to crash on: an archive whose timestamps are already nanoseconds,
#: read as milliseconds, is off by a factor of a million.
_NS_MIN = -(2**63) + 1
_NS_MAX = 2**63 - 1


def _describe_instant(value: int) -> str:
    """Render an epoch-nanosecond value, or say why it is not an instant.

    :func:`resolve_epoch_unit` reports what each candidate unit *would* have
    made of the archive, and one of those candidates is always wrong by three or
    six orders of magnitude — that is the point of the report. Formatting it
    with ``pd.Timestamp`` therefore raised ``OutOfBoundsDatetime`` from inside
    the error path, so a malformed or mixed-unit archive surfaced as a pandas
    bounds error about a value the caller never supplied, rather than as this
    module's own refusal to read it.
    """
    if not _NS_MIN <= value <= _NS_MAX:
        return f"{value} ns (not a representable instant)"
    return str(pd.Timestamp(value, unit="ns", tz="UTC"))


def scale_to_nanoseconds(raw: np.ndarray, unit: str) -> np.ndarray:
    """Convert an archive's raw epoch column to int64 UTC nanoseconds.

    Fails closed on overflow rather than letting NumPy wrap. ``raw * scale`` on
    an ``int64`` array is silent modular arithmetic: nanosecond values read as
    milliseconds wrap to arbitrary instants, some of which land back inside the
    archive's own period and would pass the range check downstream.
    """
    try:
        scale = EPOCH_UNITS[unit]
    except KeyError as exc:
        raise TradeAggregationError(
            f"{unit!r} is not a supported epoch unit; expected one of "
            f"{sorted(EPOCH_UNITS)}"
        ) from exc
    values = np.asarray(raw, dtype=np.int64)
    if scale > 1 and len(values):
        limit = _NS_MAX // scale
        if int(values.max()) > limit or int(values.min()) < -limit:
            raise TradeAggregationError(
                f"an epoch value in this archive does not fit in int64 nanoseconds "
                f"under unit {unit!r} (raw span {int(values.min())}..{int(values.max())}). "
                "The archive is not in the unit it was resolved to, or it mixes units."
            )
    return values * scale


def resolve_epoch_unit(
    raw_min: int, raw_max: int, *, period_start: pd.Timestamp, period_end: pd.Timestamp
) -> str:
    """Decide which epoch unit an archive's timestamps are in, or fail closed.

    Binance spot archives carry ``transact_time`` in milliseconds up to the end
    of 2024 and in microseconds from 2025-01-01. Guessing by magnitude alone is
    the obvious approach and the wrong one: it works until a source adds a unit,
    and it cannot tell a correctly-read archive from one whose values happen to
    land in the right decade.

    So the archive's own calendar period decides. A daily archive named
    ``…-2025-05-19.zip`` covers one UTC day; a monthly one covers one UTC month.
    The unit is the one that places *every* timestamp inside that period. If two
    units both fit, or none does, the archive is refused rather than read under
    a guess — an ambiguous period is a bug in the caller's bounds, not a
    coin-flip the exporter is entitled to make.
    """
    if raw_min > raw_max:
        raise TradeAggregationError(
            f"archive timestamp range is empty or inverted ({raw_min} > {raw_max})"
        )
    lo = int(pd.Timestamp(period_start).value)
    hi = int(pd.Timestamp(period_end).value)
    fits = [
        unit
        for unit, scale in EPOCH_UNITS.items()
        if lo <= raw_min * scale and raw_max * scale < hi
    ]
    if len(fits) == 1:
        return fits[0]
    low_stamp = pd.Timestamp(lo, unit="ns", tz="UTC")
    high_stamp = pd.Timestamp(hi, unit="ns", tz="UTC")
    window = f"[{low_stamp}, {high_stamp})"
    if not fits:
        raise TradeAggregationError(
            f"no supported epoch unit places this archive inside {window}: its raw "
            f"timestamps span {raw_min}..{raw_max}, which is "
            + ", ".join(
                f"{unit} -> {_describe_instant(raw_min * scale)}"
                f"..{_describe_instant(raw_max * scale)}"
                for unit, scale in EPOCH_UNITS.items()
            )
            + ". Refusing to read it under a guessed unit."
        )
    raise TradeAggregationError(
        f"epoch unit is ambiguous for this archive: {fits} all place it inside "
        f"{window}. Narrow the declared period rather than picking one."
    )


@dataclass
class StreamStats:
    """What the aggregator saw, for the manifest's claims about the source.

    Counters rather than samples: a duplicate is reported as a count and never
    as the row that caused it, for the same reason the seal check withholds
    its rows.
    """

    trades: int = 0
    duplicate_ids: int = 0
    timestamp_regressions: int = 0
    invalid_rows: int = 0
    nonpositive_price: int = 0
    nonpositive_qty: int = 0
    first_timestamp_ns: int | None = None
    last_timestamp_ns: int | None = None
    first_trade_id: int | None = None
    last_trade_id: int | None = None
    buy_trades: int = 0
    archives: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        def stamp(value: int | None) -> str | None:
            if value is None:
                return None
            return pd.Timestamp(value, unit="ns", tz="UTC").isoformat()

        return {
            "trades": self.trades,
            "buy_trades": self.buy_trades,
            "duplicate_ids": self.duplicate_ids,
            "timestamp_regressions": self.timestamp_regressions,
            "invalid_rows": self.invalid_rows,
            "nonpositive_price": self.nonpositive_price,
            "nonpositive_qty": self.nonpositive_qty,
            "first_timestamp": stamp(self.first_timestamp_ns),
            "last_timestamp": stamp(self.last_timestamp_ns),
            "first_trade_id": self.first_trade_id,
            "last_trade_id": self.last_trade_id,
            "strictly_sorted": self.duplicate_ids == 0 and self.timestamp_regressions == 0,
        }


class HourlyTradeAggregator:
    """Fold an ascending trade stream into one row per hour that had a trade.

    Trades arrive in chunks, ascending by ``agg_trade_id`` — the order the
    archives are written in. The aggregator holds *one* hour's trades at a time
    and emits its row when a trade for a later hour arrives, so peak memory is
    one hour of BTCUSDT rather than one month of it.

    Buffering the hour rather than accumulating incrementally is deliberate.
    The volume-at-price histogram spans ``[low, high]`` of the hour, which is
    not known until the hour is complete, and a two-pass statistic written as a
    one-pass approximation is how a "volume profile" quietly stops being one.
    """

    def __init__(self, *, styx_ns: int) -> None:
        if styx_ns % HOUR_NS != 0:
            raise TradeAggregationError(
                "the sealed instant must fall on an hour boundary for the last "
                "aggregatable hour to be well defined"
            )
        self._styx_ns = int(styx_ns)
        self._hour: int | None = None
        self._chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        self._rows: list[dict[str, Any]] = []
        self.stats = StreamStats()

    def _styx_instant(self) -> str:
        return pd.Timestamp(self._styx_ns, unit="ns", tz="UTC").isoformat()

    # -- ingestion ---------------------------------------------------------- #
    def add_chunk(
        self,
        timestamps_ns: np.ndarray,
        prices: np.ndarray,
        quantities: np.ndarray,
        buyer_maker: np.ndarray,
        trade_ids: np.ndarray,
    ) -> None:
        """Admit one ascending chunk of trades, validating before aggregating.

        Every rejection here is a case where aggregating on would produce a
        number that looks like a measurement. A non-positive price makes the
        notional and the VWAP meaningless; a regressed trade id means the
        stream is not the single ordered sequence the duplicate rule assumes;
        and a timestamp at or after the seal is refused before it can reach a
        bucket, without being described.
        """
        ts = np.asarray(timestamps_ns, dtype=np.int64)
        price = np.asarray(prices, dtype=np.float64)
        qty = np.asarray(quantities, dtype=np.float64)
        maker = np.asarray(buyer_maker, dtype=bool)
        ids = np.asarray(trade_ids, dtype=np.int64)
        if not (len(ts) == len(price) == len(qty) == len(maker) == len(ids)):
            raise TradeAggregationError("trade chunk columns have different lengths")
        if len(ts) == 0:
            return

        breaches = int((ts >= self._styx_ns).sum())
        if breaches:
            raise StyxBreach(
                f"{breaches} of {len(ts)} trades in this chunk are at or after the "
                f"sealed instant {self._styx_instant()}; "
                "the rows themselves are withheld. Bound the acquisition before the "
                "seal rather than filtering after it."
            )

        bad_price = ~np.isfinite(price) | (price <= 0.0)
        bad_qty = ~np.isfinite(qty) | (qty <= 0.0)
        self.stats.nonpositive_price += int(bad_price.sum())
        self.stats.nonpositive_qty += int(bad_qty.sum())
        if bad_price.any() or bad_qty.any():
            raise TradeAggregationError(
                f"{int((bad_price | bad_qty).sum())} trade(s) carry a non-positive or "
                "non-finite price or quantity; the hour's VWAP and notional would be "
                "meaningless and the rows are refused rather than dropped"
            )

        if not np.all(np.diff(ids) > 0):
            self.stats.duplicate_ids += int((np.diff(ids) <= 0).sum())
        if self.stats.last_trade_id is not None and ids[0] <= self.stats.last_trade_id:
            self.stats.duplicate_ids += 1
        if self.stats.duplicate_ids:
            raise TradeAggregationError(
                f"{self.stats.duplicate_ids} trade id(s) do not strictly increase. "
                f"{aggregation_spec()['duplicate_rule']}, so the stream is either "
                "duplicated or out of order and cannot be aggregated as one sequence."
            )

        if not np.all(np.diff(ts) >= 0):
            self.stats.timestamp_regressions += int((np.diff(ts) < 0).sum())
        if self.stats.last_timestamp_ns is not None and ts[0] < self.stats.last_timestamp_ns:
            self.stats.timestamp_regressions += 1
        if self.stats.timestamp_regressions:
            raise TradeAggregationError(
                f"{self.stats.timestamp_regressions} timestamp regression(s): the stream "
                "must be non-decreasing in time for one-hour-at-a-time folding to see "
                "every trade of an hour before it emits that hour"
            )

        self.stats.trades += len(ts)
        self.stats.buy_trades += int((~maker).sum())
        if self.stats.first_timestamp_ns is None:
            self.stats.first_timestamp_ns = int(ts[0])
            self.stats.first_trade_id = int(ids[0])
        self.stats.last_timestamp_ns = int(ts[-1])
        self.stats.last_trade_id = int(ids[-1])

        # The bucket's own *instant*, not its index. `aggregate_hour` subtracts
        # this from every timestamp to get the offset inside the hour, so an
        # index here would make the offsets absolute epoch nanoseconds and ask
        # `bincount` for a per-second array covering all of history.
        hours = (ts // HOUR_NS) * HOUR_NS
        # `hours` is non-decreasing, so its change points split the chunk into
        # runs that each belong to exactly one bucket.
        cuts = np.flatnonzero(np.diff(hours)) + 1
        for start, stop in zip(np.concatenate(([0], cuts)), np.concatenate((cuts, [len(ts)]))):
            hour = int(hours[start])
            if self._hour is not None and hour != self._hour:
                self._flush()
            self._hour = hour
            self._chunks.append(
                (ts[start:stop], price[start:stop], qty[start:stop], maker[start:stop])
            )

    def finish(self) -> pd.DataFrame:
        """Emit the last hour and return the table, one row per non-empty hour."""
        self._flush()
        if not self._rows:
            raise TradeAggregationError("no trades were aggregated; the table would be empty")
        frame = pd.DataFrame(self._rows, columns=list(AGGREGATE_COLUMNS))
        frame["date"] = pd.to_datetime(frame["date"], unit="ns", utc=True)
        for name, kind in AGGREGATE_COLUMN_KINDS:
            if kind == "i8":
                frame[name] = frame[name].astype("int64")
            elif kind == "f8":
                frame[name] = frame[name].astype("float64")
        return frame

    # -- aggregation -------------------------------------------------------- #
    def _flush(self) -> None:
        if self._hour is None or not self._chunks:
            return
        ts = np.concatenate([c[0] for c in self._chunks])
        price = np.concatenate([c[1] for c in self._chunks])
        qty = np.concatenate([c[2] for c in self._chunks])
        maker = np.concatenate([c[3] for c in self._chunks])
        self._rows.append(aggregate_hour(self._hour, ts, price, qty, maker))
        self._chunks = []


def aggregate_hour(
    hour_ns: int,
    timestamps_ns: np.ndarray,
    prices: np.ndarray,
    quantities: np.ndarray,
    buyer_maker: np.ndarray,
) -> dict[str, Any]:
    """Every statistic one hour contributes, from that hour's trades alone.

    Split out of the aggregator so the definition of a row can be tested
    directly against hand-built trades, without a stream, an archive or a seal.
    Order within the hour affects exactly two outputs — ``first_price`` and
    ``last_price`` — and nothing else here reads the sequence, which is what
    makes the order-invariance claim in ``docs/microstructure_v1.md`` §4 a
    property of the code rather than an intention.
    """
    ts = np.asarray(timestamps_ns, dtype=np.int64)
    price = np.asarray(prices, dtype=np.float64)
    qty = np.asarray(quantities, dtype=np.float64)
    maker = np.asarray(buyer_maker, dtype=bool)
    taker_buy = ~maker
    notional = price * qty

    offsets = ts - hour_ns
    if len(offsets) and (offsets.min() < 0 or offsets.max() >= HOUR_NS):
        raise TradeAggregationError(
            f"a trade offset of {int(offsets.min())}..{int(offsets.max())} ns is outside "
            f"the hour beginning {pd.Timestamp(hour_ns, unit='ns', tz='UTC').isoformat()}. "
            "Every statistic below indexes seconds and quarters from that instant, so an "
            "out-of-bucket trade would size a per-second array from an epoch offset."
        )
    seconds = (offsets // SECOND_NS).astype(np.int64)
    per_second = np.bincount(seconds, minlength=SECONDS_PER_HOUR)
    quarter = (offsets // (HOUR_NS // QUARTERS)).astype(np.int64)

    size_bin = np.searchsorted(np.asarray(SIZE_BIN_EDGES), qty, side="left")
    low, high = float(price.min()), float(price.max())
    span = high - low
    if span > 0.0:
        # `minimum` keeps a trade at exactly `high` in the last bin rather than
        # in a phantom bin past the end.
        price_bin = np.minimum((price - low) / span * PRICE_BINS, PRICE_BINS - 1).astype(
            np.int64
        )
    else:
        price_bin = np.zeros(len(price), dtype=np.int64)

    row: dict[str, Any] = {
        "date": int(hour_ns),
        "n_trades": int(len(ts)),
        "buy_trades": int(taker_buy.sum()),
        "qty": float(qty.sum()),
        "buy_qty": float(qty[taker_buy].sum()),
        "notional": float(notional.sum()),
        "buy_notional": float(notional[taker_buy].sum()),
        "first_price": float(price[0]),
        "last_price": float(price[-1]),
        "high_price": high,
        "low_price": low,
        "active_seconds": int((per_second > 0).sum()),
        "second_count_sq_sum": int((per_second.astype(np.int64) ** 2).sum()),
        "max_second_trades": int(per_second.max()),
        "max_empty_run_seconds": _longest_zero_run(per_second),
    }
    for q in range(QUARTERS):
        in_q = quarter == q
        row[f"q{q}_trades"] = int(in_q.sum())
        row[f"q{q}_qty"] = float(qty[in_q].sum())
        row[f"q{q}_buy_qty"] = float(qty[in_q & taker_buy].sum())
    counts = np.bincount(size_bin, minlength=SIZE_BINS)
    sums = np.bincount(size_bin, weights=qty, minlength=SIZE_BINS)
    for b in range(SIZE_BINS):
        row[f"size_count_{b}"] = int(counts[b])
        row[f"size_qty_{b}"] = float(sums[b])
    price_qty = np.bincount(price_bin, weights=qty, minlength=PRICE_BINS)
    for b in range(PRICE_BINS):
        row[f"price_qty_{b}"] = float(price_qty[b])
    return row


def _longest_zero_run(counts: np.ndarray) -> int:
    """Longest run of consecutive silent seconds inside the hour.

    A gap measure the per-second variance cannot express: two hours can have
    the same dispersion while one traded evenly and the other stopped for
    twenty minutes.
    """
    silent = counts == 0
    if not silent.any():
        return 0
    # Run lengths from the positions where the silence flips, which is cheaper
    # and clearer than a Python loop over 3600 slots.
    padded = np.concatenate(([False], silent, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return int((edges[1::2] - edges[::2]).max())


def hourly_grid(frame: pd.DataFrame) -> pd.DatetimeIndex:
    """Every hour between the table's first and last, inclusive."""
    dates = pd.DatetimeIndex(pd.to_datetime(frame["date"], utc=True))
    return pd.date_range(dates[0], dates[-1], freq="1h", tz="UTC")


def coverage_report(frame: pd.DataFrame) -> dict[str, Any]:
    """Which hours of the table's own span have a row, and which do not.

    Missing hours are hours in which BTCUSDT recorded no trade at all — venue
    downtime, or a hole in the archive. Either way they are a fact about the
    source and are counted and listed rather than filled: a fabricated zero-trade
    hour is indistinguishable from a real one once it is in the table, and
    ``microstructure_v1`` treats a missing hour as a segment boundary.
    """
    dates = pd.DatetimeIndex(pd.to_datetime(frame["date"], utc=True))
    grid = hourly_grid(frame)
    missing = grid.difference(dates)
    intervals: list[dict[str, Any]] = []
    for stamp in missing:
        if intervals and pd.Timestamp(intervals[-1]["end"]) + pd.Timedelta("1h") == stamp:
            intervals[-1]["end"] = stamp.isoformat()
            intervals[-1]["hours"] += 1
        else:
            intervals.append(
                {"start": stamp.isoformat(), "end": stamp.isoformat(), "hours": 1}
            )
    for interval in intervals:
        interval["end"] = pd.Timestamp(interval["end"]).isoformat()
    return {
        "first_hour": dates[0].isoformat(),
        "last_hour": dates[-1].isoformat(),
        "hours_in_span": int(len(grid)),
        "hours_present": int(len(dates)),
        "hours_missing": int(len(missing)),
        "missing_intervals": intervals,
    }


def iter_aggregate_columns() -> Iterator[tuple[str, str]]:
    """The (name, kind) pairs, for a caller that wants them without the tuple."""
    yield from AGGREGATE_COLUMN_KINDS


def check_table(frame: pd.DataFrame, *, styx: pd.Timestamp) -> None:
    """Refuse an aggregate table that cannot be an honest hourly summary.

    Called by the exporter before it writes and by the verifier after it reads,
    so a table that was correct when written and has been edited since is
    refused on the same grounds it would have been refused originally.
    """
    missing = [name for name in AGGREGATE_COLUMNS if name not in frame.columns]
    if missing:
        raise TradeAggregationError(f"aggregate table is missing column(s) {missing}")
    dates = pd.DatetimeIndex(pd.to_datetime(frame["date"], utc=True))
    if len(dates) == 0:
        raise TradeAggregationError("aggregate table is empty")
    if not dates.is_monotonic_increasing or dates.has_duplicates:
        raise TradeAggregationError(
            "aggregate hours must be strictly increasing; a repeated or out-of-order "
            "hour means two rows claim the same bucket"
        )
    if (dates.asi8 % HOUR_NS != 0).any():
        raise TradeAggregationError("every aggregate row must fall on an hour boundary")
    breaches = int((dates >= pd.Timestamp(styx)).sum())
    if breaches:
        raise StyxBreach(
            f"{breaches} of {len(dates)} aggregate hours are at or after the sealed "
            f"instant {pd.Timestamp(styx).isoformat()}; the rows themselves are withheld"
        )
    counts = frame["n_trades"].to_numpy(dtype=np.int64)
    if (counts <= 0).any():
        raise TradeAggregationError(
            f"{int((counts <= 0).sum())} row(s) claim an hour with no trade. An hour with "
            "no trade has no row; a zero-trade row is a fabricated bucket"
        )
    if (frame["buy_trades"].to_numpy(dtype=np.int64) > counts).any():
        raise TradeAggregationError("an hour reports more taker buys than trades")
    for total, part in (("qty", "buy_qty"), ("notional", "buy_notional")):
        whole = frame[total].to_numpy(dtype=np.float64)
        piece = frame[part].to_numpy(dtype=np.float64)
        if (piece < -0.0).any() or (piece > whole * (1 + 1e-9) + 1e-9).any():
            raise TradeAggregationError(f"{part} is not a share of {total} on every row")
    for name in ("qty", "notional", "first_price", "last_price", "high_price", "low_price"):
        values = frame[name].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or (values <= 0.0).any():
            raise TradeAggregationError(f"{name} must be finite and positive on every row")
    high = frame["high_price"].to_numpy(dtype=np.float64)
    low = frame["low_price"].to_numpy(dtype=np.float64)
    if (high < low).any():
        raise TradeAggregationError("an hour reports a high below its low")
    for name in ("first_price", "last_price"):
        values = frame[name].to_numpy(dtype=np.float64)
        if (values < low * (1 - 1e-12)).any() or (values > high * (1 + 1e-12)).any():
            raise TradeAggregationError(f"{name} falls outside the hour's traded range")
    seconds = frame["active_seconds"].to_numpy(dtype=np.int64)
    if (seconds < 1).any() or (seconds > SECONDS_PER_HOUR).any():
        raise TradeAggregationError(
            f"active_seconds must be in [1, {SECONDS_PER_HOUR}] on every row"
        )
    quarters = np.column_stack(
        [frame[name].to_numpy(dtype=np.int64) for name in _quarter_columns("trades")]
    )
    if not np.array_equal(quarters.sum(axis=1), counts):
        raise TradeAggregationError("the quarter trade counts do not sum to n_trades")
    size_counts = np.column_stack(
        [
            frame[name].to_numpy(dtype=np.int64)
            for name in _indexed_columns("size_count", SIZE_BINS)
        ]
    )
    if not np.array_equal(size_counts.sum(axis=1), counts):
        raise TradeAggregationError("the size histogram does not sum to n_trades")


def sequence_from_records(records: Sequence[Sequence[Any]]) -> tuple[np.ndarray, ...]:
    """Columns from ``(ts_ns, price, qty, buyer_maker, trade_id)`` tuples.

    A convenience for tests and for the exporter's row-by-row fallback, kept
    here so both build the arrays the same way.
    """
    if not records:
        raise TradeAggregationError("no trade records given")
    array = list(zip(*records))
    return (
        np.asarray(array[0], dtype=np.int64),
        np.asarray(array[1], dtype=np.float64),
        np.asarray(array[2], dtype=np.float64),
        np.asarray(array[3], dtype=bool),
        np.asarray(array[4], dtype=np.int64),
    )
