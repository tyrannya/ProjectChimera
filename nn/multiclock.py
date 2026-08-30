"""Causal candles on five clocks, cut from one minute-resolution source.

Every checkpoint from v4 to P5 observed the market on **one** clock. The bar was
one hour, the label looked six bars ahead, and the only thing that ever changed
was which columns were attached to that bar. P5 came closest to changing it and
did not: `mtf_v1` read 4h and 1d bars as *context columns on a 1h row*, so the
decision was still taken hourly by a model fitted on hourly samples.

This module is what a genuinely different clock needs. It takes one canonical
1m OHLCV history and cuts 5m, 15m, 30m and 1h bars from it on the fixed UTC
grid, under rules chosen so that a bar can only exist if the market has already
finished printing it:

* **Strict UTC boundaries.** A bar opens at a multiple of its own period
  measured from the epoch, never at an offset derived from where the data
  happens to start. ``resample`` does not accept an origin argument, because an
  origin is exactly the knob that would let two runs disagree about which
  minutes belong to which bar.
* **Full constituent counts, or the bar does not exist.** A 5m bar is five 1m
  candles. Four is not a 5m bar with a caveat, it is an artifact of an exchange
  outage, and a research process that accepts it has quietly agreed to trade a
  price that was never printed. Incomplete bars are *dropped*, never
  forward-filled, forward-completed, or padded from the next period.
* **No membership from the future.** The last constituent of a bar closes
  strictly before the bar's own period ends. This is asserted rather than
  assumed, because it is the one property whose violation is invisible in a
  plot and fatal in a backtest.
* **A hard research-visible boundary.** :data:`RESEARCH_VISIBLE_END` is the
  first instant of ``P4-HOLD``. No row at or after it may enter feature
  construction, model selection, fitting or scoring on any clock. The boundary
  is enforced on *constituent minutes*, not on bar-open timestamps, so a bar
  that opens before the boundary but would need a minute at or after it is
  unavailable rather than short.

The 1h clock is deliberately re-derived here rather than read from
``btc_usdt_1h_gen1_raw_pre_styx.parquet``. That gives the programme a value-level
parity check between two independently published Binance series — see
:func:`parity_against` — instead of an assumption that they agree.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from nn.data_pipeline import OHLCV_COLUMNS, timeframe_to_minutes
from nn.p4_holdout import holdout_first_instant
from nn.research_contract import load_contract

#: The contract whose seal every clock in this generation inherits. Any gen-2
#: contract would answer the same, because a change of clock may not move the
#: sealed instant; the base clock is named because it is the one the others are
#: cut from.
BASE_CONTRACT_ID = "btc-usdt-1m-gen2"

#: The instant ``P4-HOLD`` begins. Research may read rows strictly before it and
#: no others. It is not a tuning parameter and it does not move: the region
#: ``[45802, 48211)`` was retired unread by
#: ``data/research/p4_holdout_ledger.json`` and stays that way, so a minute-clock
#: checkpoint that reached into it would be spending a region the ledger says is
#: available to nobody.
#:
#: **Resolved from the ledger, never restated.** A second copy of a boundary is
#: the defect this repository already closed for the sealed instant: two
#: constants that agree today are two constants that can disagree tomorrow, and
#: the one in the module is the one nobody re-reads.
RESEARCH_VISIBLE_END = pd.Timestamp(holdout_first_instant()).tz_convert("UTC")

#: The sealed instant, resolved from the committed contract for the same reason.
#: Nothing here may reach it; the research-visible boundary above is three months
#: earlier and binds first.
STYX_START = pd.Timestamp(load_contract(BASE_CONTRACT_ID).sealed_test_start).tz_convert("UTC")

#: The clock the source is published on and everything else is cut from.
BASE_CLOCK = "1m"

#: The five clocks P6 screens. Frozen: P6 reports a verdict for each of them and
#: may not drop one that disappoints.
CLOCKS: tuple[str, ...] = ("1m", "5m", "15m", "30m", "1h")

#: The two slow clocks a genuine SWING mode needs. They are **not** part of P6 —
#: they are a separately preregistered extension, because P6's design was frozen
#: over the five above before any P6 number existed.
SWING_CLOCKS: tuple[str, ...] = ("4h", "1d")

#: Every clock the architecture supports.
ALL_CLOCKS: tuple[str, ...] = CLOCKS + SWING_CLOCKS

#: Columns a candle frame carries, in order. ``date`` is the bar's **open**
#: time, matching ``btc_usdt_1h_gen1_raw_pre_styx.parquet`` and every reader in
#: this repository.
CANDLE_COLUMNS: tuple[str, ...] = ("date", *OHLCV_COLUMNS)


class MulticlockError(ValueError):
    """A candle series cannot be trusted, and saying so beats resampling it."""


def constituent_count(timeframe: str) -> int:
    """How many 1m candles a fully closed ``timeframe`` bar is made of."""
    minutes = timeframe_to_minutes(timeframe)
    if minutes < 1:
        raise MulticlockError(f"{timeframe!r} is shorter than the 1m source")
    return minutes


def assert_minute_grid(frame: pd.DataFrame, *, what: str = "the 1m source") -> pd.Series:
    """Refuse a minute series that is not a clean, unique, ordered UTC grid.

    Four separate failures are checked, because each of them corrupts a
    different downstream guarantee and a single "looks wrong" message would not
    say which. Duplicate timestamps in particular are checked *before* the
    completeness rule below can be applied at all: sixty rows in an hour is only
    evidence of a complete hour if the sixty are sixty distinct minutes.
    """
    missing = [column for column in CANDLE_COLUMNS if column not in frame.columns]
    if missing:
        raise MulticlockError(f"{what} is missing column(s) {missing}")
    if frame.empty:
        raise MulticlockError(f"{what} is empty")

    dates = pd.to_datetime(frame["date"], utc=True)
    # Whole minutes to the nanosecond. `second`/`microsecond` alone leave a
    # sub-microsecond offset undetected, and a timestamp 1 ns off its minute
    # floors into the right bucket while making the grid a lie — so the check is
    # against the epoch remainder, which has no such blind spot.
    minute = pd.Timedelta(minutes=1).value
    if (dates.astype("int64") % minute).ne(0).any():
        raise MulticlockError(f"{what} carries timestamps that are not whole minutes")
    # A NaN price is not a candle. Counting one as a constituent would let an
    # incomplete bar reach the full-constituent rule and be emitted as complete,
    # with `first` and `last` silently taken from whichever minute did have data.
    values = frame.loc[:, list(OHLCV_COLUMNS)]
    unusable = int((~np.isfinite(values.to_numpy(dtype=np.float64))).sum())
    if unusable:
        raise MulticlockError(
            f"{what} carries {unusable} non-finite OHLCV value(s); a NaN is not a price "
            "and must not be counted as a constituent"
        )
    # A candle whose high is below its own open, or whose price is not positive,
    # is not a candle either. `high` and `low` become the max and min of a
    # resampled bar, so one broken row silently widens the bar it lands in and
    # nothing downstream can tell that from a real wick.
    prices = values.loc[:, ["open", "high", "low", "close"]].to_numpy(dtype=np.float64)
    nonpositive = int((prices <= 0.0).sum())
    if nonpositive:
        raise MulticlockError(
            f"{what} carries {nonpositive} non-positive price(s); a candle at or below "
            "zero is not a BTCUSDT candle"
        )
    body_high = np.maximum(values["open"].to_numpy(float), values["close"].to_numpy(float))
    body_low = np.minimum(values["open"].to_numpy(float), values["close"].to_numpy(float))
    broken = int(
        (values["high"].to_numpy(float) < body_high).sum()
        + (values["low"].to_numpy(float) > body_low).sum()
    )
    if broken:
        raise MulticlockError(
            f"{what} carries {broken} candle(s) whose high or low does not contain its "
            "own open and close"
        )
    if (values["volume"].to_numpy(float) < 0.0).any():
        raise MulticlockError(f"{what} carries negative volume")

    duplicates = int(dates.duplicated().sum())
    if duplicates:
        raise MulticlockError(f"{what} carries {duplicates} duplicate timestamp(s)")
    if not dates.is_monotonic_increasing:
        raise MulticlockError(f"{what} is not in increasing timestamp order")
    return dates


def assert_research_visible(
    dates: pd.Series, *, what: str, boundary: pd.Timestamp = RESEARCH_VISIBLE_END
) -> None:
    """Refuse any row at or after the research-visible boundary.

    Reports the count and never a value: a guard that printed the offending
    candle would publish the region it exists to keep closed.
    """
    breaches = int((dates >= boundary).sum())
    if breaches:
        raise MulticlockError(
            f"{what} carries {breaches} row(s) at or after the research-visible "
            f"boundary {boundary.isoformat()}. That instant is the start of "
            "P4-HOLD, which was retired unread; no checkpoint may reach it."
        )


def resample_from_minutes(
    minutes: pd.DataFrame,
    timeframe: str,
    *,
    boundary: pd.Timestamp | None = RESEARCH_VISIBLE_END,
) -> pd.DataFrame:
    """Cut ``timeframe`` bars from 1m candles, keeping only fully closed ones.

    The aggregation itself is the obvious one — first open, max high, min low,
    last close, summed volume. What matters is everything around it:

    * bars are bucketed by flooring to the period on the epoch-anchored UTC
      grid, so the boundaries are a property of the calendar and not of the
      data;
    * a bucket survives only with exactly :func:`constituent_count` distinct
      constituents, which (given the uniqueness and ordering already asserted)
      means every minute of the period is present;
    * the first constituent must be the bar's own open minute and the last must
      close strictly inside the period, which is checked here rather than
      trusted.

    ``boundary`` is the first instant research may not read. It defaults to
    :data:`RESEARCH_VISIBLE_END` so that every research path is guarded without
    having to remember to ask; passing ``None`` disables it for the live
    resampling the trading-mode controller does, where "the future" is simply
    not in the frame yet and the guard would be a category error.
    """
    dates = assert_minute_grid(minutes)
    if boundary is not None:
        assert_research_visible(dates, what="the 1m source", boundary=boundary)

    frame = minutes.loc[:, list(CANDLE_COLUMNS)].reset_index(drop=True)
    frame["date"] = dates.reset_index(drop=True)
    if timeframe == BASE_CLOCK:
        return frame

    expected = constituent_count(timeframe)
    period = pd.Timedelta(minutes=expected)
    frame["_bucket"] = frame["date"].dt.floor(period)

    grouped = frame.groupby("_bucket", sort=True)
    bars = grouped.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        _members=("close", "size"),
        _first_minute=("date", "min"),
        _last_minute=("date", "max"),
    ).reset_index()

    complete = bars["_members"].to_numpy() == expected
    bars = bars.loc[complete].reset_index(drop=True)
    if bars.empty:
        raise MulticlockError(
            f"no fully closed {timeframe} bar survives: the 1m source covers "
            f"{len(frame)} minute(s) and not one complete period"
        )

    # Membership, asserted rather than assumed. `_first_minute` proves the bar
    # opens on its own boundary; `_last_minute` proves nothing from the next
    # period was folded in. A resampler that silently shifted the grid by one
    # minute would still produce full-looking buckets and would fail here.
    if not np.array_equal(bars["_first_minute"].to_numpy(), bars["_bucket"].to_numpy()):
        raise MulticlockError(
            f"{timeframe} bars do not open on their own UTC boundary; the grid is shifted"
        )
    overrun = bars["_last_minute"] >= bars["_bucket"] + period
    if bool(overrun.any()):
        raise MulticlockError(
            f"{int(overrun.sum())} {timeframe} bar(s) contain a 1m candle from the "
            "following period, which is a look-ahead join rather than a resample"
        )

    # A bar is available only once the market has finished printing it. Its last
    # constituent minute opens at `_bucket + period - 1m`, so the bar closes at
    # `_bucket + period`; requiring that to be at most the boundary is the same
    # statement as requiring every constituent to lie strictly before it, which
    # the completeness rule has already delivered. Kept as an explicit check
    # because the two are only the same while the completeness rule holds.
    if boundary is not None:
        late = (bars["_bucket"] + period) > boundary
        if bool(late.any()):
            raise MulticlockError(
                f"{int(late.sum())} {timeframe} bar(s) would close after the "
                f"research-visible boundary {boundary.isoformat()}"
            )

    out = bars.rename(columns={"_bucket": "date"}).loc[:, list(CANDLE_COLUMNS)]
    return out.reset_index(drop=True)


def bar_availability(minutes: pd.DataFrame, timeframe: str) -> dict[str, Any]:
    """What the strict rules dropped, counted so a reader need not re-derive it."""
    dates = assert_minute_grid(minutes)
    expected = constituent_count(timeframe)
    buckets = dates.dt.floor(pd.Timedelta(minutes=expected))
    members = buckets.value_counts()
    return {
        "timeframe": timeframe,
        "constituent_minutes": expected,
        "buckets_touched": int(len(members)),
        "complete_bars": int((members == expected).sum()),
        "incomplete_bars_dropped": int((members != expected).sum()),
    }


def minute_gaps(minutes: pd.DataFrame) -> list[dict[str, Any]]:
    """Every discontinuity in the 1m source, as spans rather than as a count.

    The gaps are a property of the exchange, not a defect in the acquisition:
    Binance's published 1m archive is missing the minutes it never printed. They
    are enumerated because every derived clock's completeness rule is a function
    of them, and a reader comparing two clocks' bar counts needs to see why.
    """
    dates = assert_minute_grid(minutes)
    step = dates.diff()
    breaks = step.ne(pd.Timedelta(minutes=1)) & step.notna()
    spans = []
    for position in np.flatnonzero(breaks.to_numpy()):
        before = dates.iloc[position - 1]
        after = dates.iloc[position]
        spans.append(
            {
                "after": before.isoformat(),
                "before": after.isoformat(),
                "missing_minutes": int((after - before) / pd.Timedelta(minutes=1)) - 1,
            }
        )
    return spans


@dataclass(frozen=True)
class ParityResult:
    """A value-level comparison of two candle series over their shared bars."""

    timeframe: str
    tolerance: float
    overlapping_bars: int
    only_in_left: int
    only_in_right: int
    mismatching_bars: int
    max_relative_difference: dict[str, float]
    exact_columns: tuple[str, ...]
    mismatching_timestamps: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "timeframe": self.timeframe,
            "tolerance": self.tolerance,
            "overlapping_bars": self.overlapping_bars,
            "only_in_left": self.only_in_left,
            "only_in_right": self.only_in_right,
            "mismatching_bars": self.mismatching_bars,
            "max_relative_difference": dict(self.max_relative_difference),
            "exact_columns": list(self.exact_columns),
            "mismatching_timestamps": list(self.mismatching_timestamps),
        }


#: How close two independently published Binance series must be to count as the
#: same value. Binance publishes prices as decimal strings with two decimal
#: places for BTCUSDT and volumes with eight; both round-trip through float64
#: exactly, so agreement between two correct readings of the same bar is
#: *identity*, not proximity. The tolerance is therefore set at the width of a
#: float64 summation-order difference rather than at anything economic: a
#: discrepancy larger than this is a real disagreement between the two sources
#: and must be explained, never absorbed.
PARITY_TOLERANCE = 1e-9


def parity_against(
    derived: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    timeframe: str,
    tolerance: float = PARITY_TOLERANCE,
) -> ParityResult:
    """Compare a derived clock against an independently published series.

    Reports rather than raises. Whether a given level of disagreement is
    acceptable is a research judgement that belongs in a preregistration and in
    a manifest, not in a comparison function — and a function that raised would
    make the disagreement impossible to describe.
    """
    left = derived.loc[:, list(CANDLE_COLUMNS)].copy()
    right = reference.loc[:, list(CANDLE_COLUMNS)].copy()
    left["date"] = pd.to_datetime(left["date"], utc=True)
    right["date"] = pd.to_datetime(right["date"], utc=True)

    joined = left.merge(right, on="date", how="outer", suffixes=("_derived", "_reference"))
    both = joined.dropna(subset=["close_derived", "close_reference"])

    mismatched = np.zeros(len(both), dtype=bool)
    worst: dict[str, float] = {}
    exact: list[str] = []
    for column in OHLCV_COLUMNS:
        a = both[f"{column}_derived"].to_numpy(dtype=np.float64)
        b = both[f"{column}_reference"].to_numpy(dtype=np.float64)
        relative = np.abs(a - b) / np.maximum(np.abs(b), 1.0)
        # A non-finite difference is a disagreement, not an absence of one:
        # `nan > tolerance` is False, so a NaN on either side would otherwise
        # read as agreement and be written into the manifest as one.
        relative = np.where(np.isfinite(relative), relative, np.inf)
        worst[column] = float(relative.max()) if len(relative) else 0.0
        if worst[column] == 0.0:
            exact.append(column)
        mismatched |= relative > tolerance

    timestamps = tuple(
        pd.to_datetime(value, utc=True).isoformat()
        for value in both.loc[mismatched, "date"].tolist()
    )
    return ParityResult(
        timeframe=timeframe,
        tolerance=tolerance,
        overlapping_bars=int(len(both)),
        only_in_left=int(joined["close_reference"].isna().sum()),
        only_in_right=int(joined["close_derived"].isna().sum()),
        mismatching_bars=int(mismatched.sum()),
        max_relative_difference=worst,
        exact_columns=tuple(exact),
        mismatching_timestamps=timestamps,
    )


#: Prefixed into every candle digest so that a digest can never be confused with
#: one taken over some other repository object that happens to hash the same
#: bytes, and so that a change to this definition is a visible version bump.
CANDLE_DIGEST_DOMAIN = b"chimera.multiclock-candles/1"


def candle_digest(frame: pd.DataFrame) -> str:
    """A value-level digest of a candle frame, independent of file encoding.

    Two Parquet files written by different library versions differ byte for byte
    while carrying identical candles. A research process cares about the candles,
    so identity is defined over the values: open times as int64 UTC nanoseconds
    and the five numbers as IEEE-754 float64, both little-endian and in row
    order, so the digest is a function of the numbers and of nothing else.

    Explicit ``<`` byte order rather than native: a digest that changed on a
    big-endian machine would make the manifest a claim about the reader's CPU.
    """
    digest = hashlib.sha256()
    digest.update(CANDLE_DIGEST_DOMAIN)
    digest.update(str(len(frame)).encode())
    dates = pd.to_datetime(frame["date"], utc=True)
    digest.update(dates.to_numpy(dtype="datetime64[ns]").astype("<i8").tobytes())
    for name in OHLCV_COLUMNS:
        digest.update(name.encode())
        digest.update(frame[name].to_numpy(dtype=np.float64).astype("<f8").tobytes())
    return digest.hexdigest()


def clocks_from_minutes(
    minutes: pd.DataFrame,
    timeframes: Iterable[str] = CLOCKS,
    *,
    boundary: pd.Timestamp | None = RESEARCH_VISIBLE_END,
) -> dict[str, pd.DataFrame]:
    """Every requested clock, cut from one source under one set of rules."""
    return {
        timeframe: resample_from_minutes(minutes, timeframe, boundary=boundary)
        for timeframe in timeframes
    }


def describe_clock(frame: pd.DataFrame, timeframe: str) -> dict[str, Any]:
    """What a manifest records about one derived clock."""
    dates = pd.to_datetime(frame["date"], utc=True)
    return {
        "timeframe": timeframe,
        "constituent_minutes": constituent_count(timeframe),
        "rows": int(len(frame)),
        "start": dates.iloc[0].isoformat(),
        "end": dates.iloc[-1].isoformat(),
        "digest": candle_digest(frame),
    }


def assert_manifest_clock(frame: pd.DataFrame, record: Mapping[str, Any]) -> None:
    """Hold a derived clock to what the manifest says it is.

    The digest is recomputed rather than read. A manifest is a claim about a
    file; this is the only line that turns the claim into a fact.
    """
    timeframe = str(record["timeframe"])
    actual = describe_clock(frame, timeframe)
    for key in ("rows", "start", "end", "digest"):
        if actual[key] != record[key]:
            raise MulticlockError(
                f"derived {timeframe} clock disagrees with the manifest on {key}: "
                f"recomputed {actual[key]!r}, manifest says {record[key]!r}"
            )
