"""``mtf_v1``: the OHLCV14 engine on a 4h and a daily clock, causally aligned.

P5's information set, implemented exactly as ``docs/p5_preregistration.md``
specifies it. Nothing here chooses anything: the timeframes, the bar grid, the
completeness rule, the feature engine, the warm-up and the alignment rule are all
values in :mod:`nn.p5_preregistration`, and the column names come from there
rather than being spelled again — an arm whose columns were named somewhere other
than the preregistration would be answering a question nobody registered.

Three properties this module has to have, and the shape of the code that gives
each one:

*a bar that has not closed is invisible.* The as-of index is
``searchsorted(close_times, t, side="right") - 1``, so the bar selected for the
row at ``t`` is the last one whose **close time** is at or before ``t``. There is
no branch in which a bar contributes to a row inside its own window.

*a partial bar is not a bar.* :func:`higher_timeframe_bars` counts the 1h candles
that landed in each grid bucket and keeps only the buckets that are full. The
committed history has 15 gaps; a bar built from three of its four hours is not a
4h bar, and forward-filling one would put an unobserved hour into every row that
reads it.

*a row with no fresh context is ineligible, not served a stale one.* If a dropped
bar leaves a hole, the nearest complete bar is more than one bar old and the row
is excluded from the sample universe rather than given eight-hour-old context
while its neighbours get fresh context.

The join is the step where a value could land on the wrong row, and here it fails
differently from every other family in this repository: a higher-timeframe column
is **piecewise constant** across the rows inside one bar, so the ±1-row shift
control that catches a bad ``smc_body_ratio`` or ``ms_qty_imbalance`` join matches
almost everywhere and proves nothing. :func:`mtf_join_evidence` therefore checks
the shift at the **boundary rows** — where the as-of bar index changes, and where
a shift is detectable — and reports how many such rows there were, so the
coverage does not have to be taken on trust.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd

from chimera.features import FeatureSpec, compute_features, feature_columns
from nn.p5_preregistration import (
    ALIGNMENT,
    TIMEFRAMES,
    WARMUP_BARS,
    mtf_columns,
    preregistration_hash,
)

#: Bumped when the columns or their semantics change. A run under a different
#: version is a different feature family and may not share an arm name.
MTF_SPEC_VERSION = 1

#: The two families an ablation could remove, partitioning the 28 columns. Fixed
#: here rather than derived from whichever clock turned out to matter: choosing
#: the groups after seeing the result would make an ablation a search.
MTF_FEATURE_FAMILIES: dict[str, tuple[str, ...]] = {
    f"mtf_{timeframe}": tuple(
        column
        for column in mtf_columns()
        if column.startswith(TIMEFRAMES[timeframe]["prefix"])
    )
    for timeframe in TIMEFRAMES
}

#: The value written into an ineligible row. Arbitrary and unreachable: the
#: sample universe excludes those rows from every split of every fold, so no
#: model is fitted on one and no prediction is scored at one. It is a definite
#: number rather than NaN because a NaN in a feature matrix is the kind of thing
#: that survives until something far away divides by it.
INELIGIBLE_FILL = 0.0


class MtfError(ValueError):
    """The higher-timeframe family cannot be built from what it was given."""


def mtf_feature_columns() -> list[str]:
    """The 28 column names, in the order the family emits them."""
    return mtf_columns()


@dataclass(frozen=True)
class MtfSpec:
    """Every constant ``mtf_v1`` reads. All of them come from the preregistration."""

    #: In the order the columns are emitted.
    timeframes: tuple[str, ...] = tuple(TIMEFRAMES)
    #: Bars discarded at the head of each higher-timeframe series. The same
    #: number, and the same reason, the 1h research spine was built with.
    warmup_bars: int = WARMUP_BARS
    #: How many bars stale the as-of context may be. One means "the immediately
    #: preceding complete bar, and nothing older".
    staleness_bound_bars: int = int(ALIGNMENT["staleness_bound_bars"])
    #: The OHLCV14 window lengths, unchanged, measured in bars of the higher
    #: timeframe rather than in hours.
    feature_spec: FeatureSpec = field(default_factory=FeatureSpec)

    def __post_init__(self) -> None:
        unknown = [t for t in self.timeframes if t not in TIMEFRAMES]
        if unknown:
            raise MtfError(f"{unknown} are not preregistered timeframes: {sorted(TIMEFRAMES)}")
        if self.warmup_bars < 0 or self.staleness_bound_bars < 1:
            raise MtfError("warm-up must be non-negative and the staleness bound at least 1")

    def hours(self, timeframe: str) -> int:
        return int(TIMEFRAMES[timeframe]["hours"])

    def prefix(self, timeframe: str) -> str:
        return str(TIMEFRAMES[timeframe]["prefix"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": MTF_SPEC_VERSION,
            "timeframes": list(self.timeframes),
            "hours": {t: self.hours(t) for t in self.timeframes},
            "warmup_bars": self.warmup_bars,
            "staleness_bound_bars": self.staleness_bound_bars,
            "feature_spec": self.feature_spec.to_dict(),
            "feature_names": mtf_feature_columns(),
            "families": {k: list(v) for k, v in MTF_FEATURE_FAMILIES.items()},
            "preregistration_hash": preregistration_hash(),
        }

    def spec_hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()


#: The epoch instant the UTC grid is cut from. Bars are `epoch_hours // width`,
#: so a 4h bar opens at 00/04/08/12/16/20 UTC and a daily bar at 00 UTC.
_EPOCH = pd.Timestamp(0, tz="UTC")


def epoch_hours(dates: pd.Series) -> pd.Series:
    """Whole hours since the epoch, independent of the column's time resolution.

    Deliberately not ``astype("int64")``: the committed snapshot stores
    ``datetime64[ms, UTC]`` and the canonical dataset stores nanoseconds, so a
    raw integer cast silently changes units — and a division by 3.6e12 then puts
    every candle in bucket zero and every bar in one incomplete bucket. Dividing
    two pandas durations has no unit to get wrong.
    """
    return (pd.to_datetime(dates, utc=True) - _EPOCH) // pd.Timedelta(hours=1)


def higher_timeframe_bars(candles: pd.DataFrame, hours: int) -> pd.DataFrame:
    """Complete bars of ``hours`` width, on the fixed UTC grid.

    Buckets are cut by integer division of the epoch hour, which puts a 4h bar's
    open at 00/04/08/12/16/20 UTC and a daily bar's at 00 UTC — the grid the venue
    publishes, and the only one a live system could reproduce without knowing when
    it happened to start.

    A bucket becomes a bar only if all ``hours`` of its 1h candles are present.
    The returned frame carries ``bar_start``, ``close_time`` and the five OHLCV
    columns, indexed 0..n-1 over the *complete* bars, and it is that index the
    warm-up and the as-of join are expressed in.
    """
    required = ("date", "open", "high", "low", "close", "volume")
    missing = [c for c in required if c not in candles.columns]
    if missing:
        raise MtfError(f"the candle frame is missing {missing}")
    if hours <= 0:
        raise MtfError(f"a bar of {hours} hours is not a bar")

    frame = candles.loc[:, list(required)].copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    if not frame["date"].is_monotonic_increasing or not frame["date"].is_unique:
        raise MtfError("candles must be sorted and free of duplicate timestamps")

    frame["bucket"] = epoch_hours(frame["date"]) // hours

    grouped = frame.groupby("bucket", sort=True)
    bars = pd.DataFrame(
        {
            "open": grouped["open"].first(),
            "high": grouped["high"].max(),
            "low": grouped["low"].min(),
            "close": grouped["close"].last(),
            "volume": grouped["volume"].sum(),
            "candles": grouped["close"].size(),
        }
    )
    complete = bars.loc[bars["candles"] == hours].copy()
    starts = pd.to_datetime(complete.index.to_numpy() * hours, unit="h", utc=True)
    complete.insert(0, "bar_start", starts)
    complete.insert(1, "close_time", starts + pd.Timedelta(hours=hours))
    return complete.reset_index(drop=True)


@dataclass(frozen=True)
class TimeframeContext:
    """One higher timeframe, aligned to the rows it will be read at."""

    timeframe: str
    hours: int
    #: Feature values at each candle row, ``(n_rows, 14)``. Rows whose context is
    #: unusable hold :data:`INELIGIBLE_FILL` and are excluded by ``eligible``.
    values: np.ndarray
    #: Which candle rows have a usable, fresh, warmed-up context.
    eligible: np.ndarray
    #: The complete-bar index each candle row reads, ``-1`` where there is none.
    as_of: np.ndarray
    #: How many hours past its close the as-of bar is, at each row.
    staleness_hours: np.ndarray
    bars: pd.DataFrame
    grid_bars: int
    first_usable_bar: int


def _timeframe_context(
    candles: pd.DataFrame, spec: MtfSpec, timeframe: str
) -> TimeframeContext:
    hours = spec.hours(timeframe)
    dates = pd.to_datetime(candles["date"], utc=True)
    bars = higher_timeframe_bars(candles, hours)

    buckets = epoch_hours(dates) // hours
    grid_bars = int(buckets.iloc[-1] - buckets.iloc[0] + 1)

    if len(bars) <= spec.warmup_bars:
        raise MtfError(
            f"{timeframe}: {len(bars)} complete bars is not more than the {spec.warmup_bars}-"
            "bar warm-up, so no row could ever have a usable context"
        )

    features = compute_features(
        bars[["open", "high", "low", "close", "volume"]], spec.feature_spec
    )
    finite = np.isfinite(features.to_numpy(dtype=np.float64)).all(axis=1)
    usable = finite.copy()
    usable[: spec.warmup_bars] = False
    first_usable = int(np.argmax(usable)) if usable.any() else -1
    if first_usable < 0:
        raise MtfError(f"{timeframe}: no bar survives the warm-up and the finiteness check")

    # The as-of index: the last bar whose CLOSE is at or before the row. `right`
    # so a row landing exactly on a close *does* see that bar — the boundary case
    # the leakage battery's L3 pins.
    closes = bars["close_time"].to_numpy()
    as_of = np.searchsorted(closes, dates.to_numpy(), side="right") - 1

    have = as_of >= 0
    safe = np.where(have, as_of, 0)
    staleness = np.full(len(dates), np.nan)
    delta = (dates.to_numpy() - closes[safe]) / np.timedelta64(1, "h")
    staleness[have] = delta[have]

    fresh = have & (staleness < hours * spec.staleness_bound_bars)
    eligible = fresh & usable[safe]

    values = np.full((len(dates), len(feature_columns())), INELIGIBLE_FILL, dtype=np.float64)
    matrix = features.to_numpy(dtype=np.float64)
    values[eligible] = matrix[safe[eligible]]

    return TimeframeContext(
        timeframe=timeframe,
        hours=hours,
        values=values,
        eligible=eligible,
        as_of=as_of,
        staleness_hours=staleness,
        bars=bars,
        grid_bars=grid_bars,
        first_usable_bar=first_usable,
    )


@dataclass(frozen=True)
class MtfContext:
    """``mtf_v1`` over one candle history, with the mask and the join witness."""

    spec: MtfSpec
    #: ``(n_rows, 28)`` in :func:`mtf_feature_columns` order.
    values: np.ndarray
    #: One flag per candle row: whether every timeframe has a usable context.
    eligible: np.ndarray
    per_timeframe: dict[str, TimeframeContext]
    evidence: dict[str, Any]

    def column(self, name: str) -> np.ndarray:
        return self.values[:, mtf_feature_columns().index(name)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_version": MTF_SPEC_VERSION,
            "spec_hash": self.spec.spec_hash(),
            "rows": int(len(self.eligible)),
            "rows_eligible": int(np.count_nonzero(self.eligible)),
            "feature_names": mtf_feature_columns(),
            "timeframes": {
                name: {
                    "hours": context.hours,
                    "grid_bars": context.grid_bars,
                    "complete_bars": int(len(context.bars)),
                    "dropped_bars": int(context.grid_bars - len(context.bars)),
                    "first_usable_bar": context.first_usable_bar,
                    "first_usable_close": str(
                        context.bars["close_time"].iloc[context.first_usable_bar]
                    ),
                    "rows_eligible": int(np.count_nonzero(context.eligible)),
                    "max_staleness_hours": float(
                        np.nanmax(context.staleness_hours[context.eligible])
                    ),
                }
                for name, context in self.per_timeframe.items()
            },
            "evidence": self.evidence,
        }


def build_mtf_context(candles: pd.DataFrame, spec: MtfSpec | None = None) -> MtfContext:
    """``mtf_v1`` over ``candles``, aligned row-for-row to them.

    ``candles`` is the 1h history the research spine was derived from, already
    truncated at the last row the spine uses. The truncation is the caller's
    (:func:`nn.information_sets.build_information_set_views` does it before any
    family is computed), and L10 of the leakage battery checks that it makes no
    difference — which is what makes causality structural here rather than a
    property of today's implementation.
    """
    spec = spec or MtfSpec()
    contexts = {t: _timeframe_context(candles, spec, t) for t in spec.timeframes}

    eligible = np.ones(len(candles), dtype=bool)
    for context in contexts.values():
        eligible &= context.eligible

    values = np.concatenate([contexts[t].values for t in spec.timeframes], axis=1)
    # A row is either eligible with a real context on every clock, or ineligible
    # and filled. Zeroing here rather than trusting the per-timeframe masks means
    # a row excluded on one clock cannot keep a value from the other.
    values[~eligible] = INELIGIBLE_FILL

    context = MtfContext(
        spec=spec,
        values=values,
        eligible=eligible,
        per_timeframe=contexts,
        evidence={},
    )
    return MtfContext(
        spec=spec,
        values=values,
        eligible=eligible,
        per_timeframe=contexts,
        evidence=mtf_join_evidence(candles, context),
    )


def mtf_join_evidence(candles: pd.DataFrame, context: MtfContext) -> dict[str, Any]:
    """Re-derive the join independently, and score the shift that would break it.

    Two witnesses per timeframe, chosen so they fail differently:

    * ``ret_1`` is rebuilt **positionally**, from the complete-bar close series at
      the as-of indices the join selected. It is local to two adjacent bars, has
      no window and no state, and cannot agree under a shift.
    * ``atr_norm`` is rebuilt **by close timestamp**, through a lookup keyed on
      the bar's own close time rather than on a row number. A positional check
      alone would still pass if ``as_of`` and the re-derivation were shifted
      together.

    The shift control is evaluated at the **boundary rows** — the rows where the
    as-of bar index changes — because a higher-timeframe column is piecewise
    constant inside a bar and a whole-column ±1 shift matches almost everywhere.
    The count of boundary rows is reported so the coverage is visible.
    """
    dates = pd.to_datetime(candles["date"], utc=True)
    per_timeframe: dict[str, Any] = {}

    for name, tf in context.per_timeframe.items():
        prefix = context.spec.prefix(name)
        bars = tf.bars
        closes = bars["close"].to_numpy(dtype=np.float64)
        # The COMBINED mask, not this timeframe's own: `build_mtf_context` zeroes
        # a row that any clock found ineligible, so a row eligible on 4h and not
        # on 1d holds the fill value and comparing it against a real return would
        # report a join failure that is not one.
        eligible = context.eligible
        as_of = tf.as_of

        # Witness 1: ret_1 from the bar closes, positionally.
        expected_ret = np.full(len(bars), np.nan)
        expected_ret[1:] = closes[1:] / closes[:-1] - 1.0
        joined_ret = context.column(f"{prefix}ret_1")
        if not np.allclose(
            joined_ret[eligible], expected_ret[as_of[eligible]], rtol=0.0, atol=1e-12
        ):
            worst = int(
                np.argmax(np.abs(joined_ret[eligible] - expected_ret[as_of[eligible]]))
            )
            raise MtfError(
                f"the {name} join is wrong: {prefix}ret_1 at eligible row {worst} does not "
                "match the bar the as-of index selected. A value is sitting on the wrong row."
            )

        # Witness 2: atr_norm by close timestamp rather than by row number.
        features = compute_features(
            bars[["open", "high", "low", "close", "volume"]], context.spec.feature_spec
        )
        by_close = pd.Series(
            features["atr_norm"].to_numpy(dtype=np.float64),
            index=pd.DatetimeIndex(bars["close_time"]),
        )
        looked_up = by_close.reindex(
            pd.DatetimeIndex(bars["close_time"].to_numpy()[as_of[eligible]])
        ).to_numpy()
        if not np.allclose(
            context.column(f"{prefix}atr_norm")[eligible], looked_up, rtol=0.0, atol=1e-12
        ):
            raise MtfError(
                f"the {name} join is wrong: {prefix}atr_norm does not match the bar with "
                "the close time the as-of index points at"
            )

        # The shift control, at the rows where it can bite.
        changes = np.flatnonzero(np.diff(as_of, prepend=as_of[0] - 1) != 0)
        boundary = changes[eligible[changes]]
        boundary = boundary[(as_of[boundary] > 0) & (as_of[boundary] < len(bars) - 1)]

        def shifted_matches(offset: int) -> bool:
            if len(boundary) == 0:
                return True
            return bool(
                np.allclose(
                    joined_ret[boundary],
                    expected_ret[as_of[boundary] + offset],
                    rtol=0.0,
                    atol=1e-12,
                    equal_nan=True,
                )
            )

        # Causality, stated as a measurement rather than as a comment.
        close_times = bars["close_time"].to_numpy()
        row_times = dates.to_numpy()
        not_yet_closed = int(
            np.count_nonzero(close_times[as_of[eligible]] > row_times[eligible])
        )

        per_timeframe[name] = {
            "recomputed": [
                f"{prefix}ret_1 from the complete-bar closes, positionally",
                f"{prefix}atr_norm from the complete bars, by close timestamp",
            ],
            "rows": int(len(as_of)),
            "rows_eligible_on_this_clock": int(np.count_nonzero(tf.eligible)),
            "rows_eligible_overall": int(np.count_nonzero(eligible)),
            "boundary_rows_checked": int(len(boundary)),
            "matches": True,
            "matches_under_plus_one_bar_shift": shifted_matches(1),
            "matches_under_minus_one_bar_shift": shifted_matches(-1),
            "rows_reading_a_bar_that_had_not_closed": not_yet_closed,
            "max_staleness_hours": float(np.nanmax(tf.staleness_hours[tf.eligible])),
            "staleness_bound_hours": tf.hours * context.spec.staleness_bound_bars,
            "as_of_is_monotone": bool(np.all(np.diff(as_of) >= 0)),
            "first_eligible_row": int(np.argmax(eligible)),
            "grid_bars": tf.grid_bars,
            "complete_bars": int(len(bars)),
            "dropped_bars": int(tf.grid_bars - len(bars)),
        }

        if not_yet_closed:
            raise MtfError(
                f"{name}: {not_yet_closed} eligible row(s) read a bar that had not closed "
                "at the row's own timestamp"
            )
        if (
            per_timeframe[name]["matches_under_plus_one_bar_shift"]
            or per_timeframe[name]["matches_under_minus_one_bar_shift"]
        ):
            raise MtfError(
                f"the {name} join check cannot tell a correct join from a shifted one on "
                "this data, so it is not evidence and must not be reported as though it "
                "were"
            )
        if not per_timeframe[name]["as_of_is_monotone"]:
            raise MtfError(f"{name}: the as-of index is not monotone; a row reads backwards")

    return {
        "spec_hash": context.spec.spec_hash(),
        "preregistration_hash": preregistration_hash(),
        "rows": int(len(context.eligible)),
        "rows_eligible": int(np.count_nonzero(context.eligible)),
        "eligible_fraction": round(float(context.eligible.mean()), 6),
        "per_timeframe": per_timeframe,
    }


def mtf_spec_identity(spec: MtfSpec, context: Mapping[str, Any]) -> dict[str, Any]:
    """What a P5 cell records about the family it was built from.

    A block of its own rather than a widening of ``feature_spec_identity``: a new
    source or a new family gets a new block, because widening the combined hash
    would give the same earlier research a different identity.
    """
    return {
        "spec_version": MTF_SPEC_VERSION,
        "spec": spec.to_dict(),
        "spec_hash": spec.spec_hash(),
        "preregistration_hash": preregistration_hash(),
        "columns": mtf_feature_columns(),
        "families": {k: list(v) for k, v in MTF_FEATURE_FAMILIES.items()},
        "universe": dict(context),
        "combined_mtf_hash": hashlib.sha256(
            json.dumps(
                {"spec": spec.spec_hash(), "universe": dict(context)},
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode()
        ).hexdigest(),
    }
