"""``derivatives_v1``: the eight causal positioning-and-carry columns of P4.

The specification is ``docs/p4_preregistration.md`` §5 and its machine-readable
twin :mod:`nn.p4_preregistration`. **This module reads that specification rather
than restating it**: the column names, their order, their windows and their clips
are taken from ``FEATURES`` at import, so a run cannot compute a window the
preregistration does not declare, and moving a constant moves both the
preregistration hash and this module's spec hash at once.

**Everything here is causal by construction, on the conservative rule of §4.**
A funding settlement at instant ``T`` is visible to row ``t`` only when
``T <= t`` — the candle's *open*, one full hour before the row decides. An
open-interest snapshot is visible only when ``create_time <= t`` and it is at
most an hour old. The perpetual and spot closes are the candle at ``t``, which
is exactly the boundary every OHLCV14 column already sits on. Nothing reads a
predicted funding rate, at any horizon.

**A row whose feature is undefined leaves the sample universe; it does not take
a default.** This is the one structural difference from ``smc_v1``,
``chart_structure_v1`` and ``microstructure_v1``, which all guarantee a finite
value on every row and fill from a declared default. P4's control is re-run on
the *intersection* (§6.2), so an undefined derivatives row has to leave the
universe for **every** arm rather than be imputed for one — and that is only
honest if the engine reports which rows are defined instead of hiding the
question behind a fill value. :func:`compute_derivatives_features` therefore
returns a ``defined`` mask beside the columns, and the columns carry a declared
zero where it is false.

**Trailing windows stop at a segment boundary.** Segments are the maximal runs of
consecutive hours in the candle history, exactly as ``smc_v1`` resets its state
at a market-data gap, which is §6.2's fourth universe condition.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from nn.derivatives_sources import (
    DERIVATIVES_COLUMNS,
    HOUR_NS,
    staleness_bound_ns,
)
from nn.p4_preregistration import FEATURES, WARMUP_HOURS

#: Names the meaning of this feature family. Recorded in every P4 artifact, so a
#: later `derivatives_v2` cannot be mistaken for this one in a comparison.
DERIVATIVES_SPEC_VERSION = "derivatives_v1"

#: The eight columns of §5, in that exact order, read from the preregistration.
DERIVATIVES_FEATURE_COLUMNS: tuple[str, ...] = tuple(feature["name"] for feature in FEATURES)

#: Family name -> its columns: the three fields of §3. A partition, asserted at
#: import, and derived from the preregistration's own `family` key rather than
#: from whatever grouping turned out to be convenient.
DERIVATIVES_FEATURE_FAMILIES: dict[str, tuple[str, ...]] = {
    family: tuple(f["name"] for f in FEATURES if f["family"] == family)
    for family in dict.fromkeys(f["family"] for f in FEATURES)
}

_covered = [c for group in DERIVATIVES_FEATURE_FAMILIES.values() for c in group]
assert sorted(_covered) == sorted(
    DERIVATIVES_FEATURE_COLUMNS
), "the derivatives families do not partition the column list"
assert len(DERIVATIVES_FEATURE_COLUMNS) == 8, "derivatives_v1 is eight columns, no more"

#: Window and clip per column, read from the preregistration once.
_WINDOW: dict[str, int | None] = {f["name"]: f["window"] for f in FEATURES}
_CLIP: dict[str, list[float]] = {f["name"]: list(f["clip"]) for f in FEATURES}


def derivatives_feature_columns() -> list[str]:
    """The eight columns of §5, in order. A copy, so a caller cannot reorder them."""
    return list(DERIVATIVES_FEATURE_COLUMNS)


class DerivativesError(ValueError):
    """The derivatives source cannot produce an honest ``derivatives_v1`` row."""


@dataclass(frozen=True)
class DerivativesSpec:
    """The constants ``derivatives_v1`` is computed under, and the readings it took.

    The windows and clips are **not** fields here: they are values in
    :data:`nn.p4_preregistration.FEATURES` and are hashed from there, so this
    class cannot be constructed with a window §5 does not declare. What it does
    carry is the small number of constructions §5's prose leaves open, recorded
    as values so that the reading is part of the spec hash rather than a habit
    of the implementation:

    ``trailing_window_includes_current_row``
        §5 says "the mean of the last 168h" and "the last 30 visible
        settlements", and names ``drv_funding_last`` — the most recent visible
        settlement — as the numerator of ``drv_funding_z``. "The last N" therefore
        includes the row's own observation, which is available to it. Recorded
        because this repository's other families use the opposite convention
        (:func:`nn.microstructure._trailing` is strictly prior) and a silent
        disagreement between two families is exactly the kind of thing that is
        discovered from a number rather than from a diff.

    ``funding_z_window_uses_clipped_rates``
        False. §5 standardises ``drv_funding_last`` against "the last 30 visible
        **settlements**", which are the archive's rates, not this column's
        clipped copy. For BTCUSDT the distinction cannot bind: Binance caps the
        instrument's funding rate at ±0.75%, strictly inside the ±1% clip, and
        ``tests/test_derivatives.py`` asserts that the two readings agree
        wherever the clip does not bite.

    ``basis_z_window_uses_clipped_basis``
        True. §5 standardises ``drv_basis`` — the column, clip and all — against
        "the mean of the last 168h" of the same quantity, so numerator and
        window are one series.

    ``min_window_observations``
        One. A trailing mean over the last 168 hours is defined when the window
        holds an observation; §3.0a's own arithmetic ("one missing day removes
        every hour of that UTC day") is only true if a punctured window still
        produces a value for the hours around it. A row whose *own* observation
        is missing or stale leaves the universe regardless, which is the rule
        that does the work.
    """

    trailing_window_includes_current_row: bool = True
    funding_z_window_uses_clipped_rates: bool = False
    basis_z_window_uses_clipped_basis: bool = True
    min_window_observations: int = 1
    eps: float = 1e-12
    warmup_hours: int = WARMUP_HOURS

    def to_dict(self) -> dict[str, Any]:
        return {
            "trailing_window_includes_current_row": self.trailing_window_includes_current_row,
            "funding_z_window_uses_clipped_rates": self.funding_z_window_uses_clipped_rates,
            "basis_z_window_uses_clipped_basis": self.basis_z_window_uses_clipped_basis,
            "min_window_observations": self.min_window_observations,
            "eps": self.eps,
            "warmup_hours": self.warmup_hours,
            "max_staleness_hours": {
                field: staleness_bound_ns(field) // HOUR_NS
                for field in ("funding_rate", "open_interest", "perpetual_price")
            },
        }

    def spec_hash(self) -> str:
        """Identity of the definition, including the §5 constants it reads.

        The windows and clips are inside the material even though they are not
        fields, because they are what the columns *are*: a spec hash that moved
        only when this class changed would call two different feature families
        by one name the moment §5 was edited.
        """
        material = {
            "version": DERIVATIVES_SPEC_VERSION,
            "constants": self.to_dict(),
            "columns": list(DERIVATIVES_FEATURE_COLUMNS),
            "families": {k: list(v) for k, v in DERIVATIVES_FEATURE_FAMILIES.items()},
            "windows": {name: _WINDOW[name] for name in DERIVATIVES_FEATURE_COLUMNS},
            "clips": {name: _CLIP[name] for name in DERIVATIVES_FEATURE_COLUMNS},
        }
        return hashlib.sha256(
            json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def _clip(values: np.ndarray, name: str) -> np.ndarray:
    lo, hi = _CLIP[name]
    return np.clip(values, lo, hi)


def segment_ids_from_hours(dates: pd.Series) -> np.ndarray:
    """Maximal runs of consecutive hours, numbered in order.

    The same rule :mod:`nn.data_pipeline` gives the research spine — a break of
    more than one bar starts a new segment — applied to the candle history the
    derivatives observations are aligned to. §6.2's fourth universe condition is
    enforced against these ids, so they must be the spine's own gap structure and
    not a second opinion about it.
    """
    stamps = pd.DatetimeIndex(pd.to_datetime(dates, utc=True))
    if len(stamps) == 0:
        raise DerivativesError("an empty hourly table has no segments")
    if not stamps.is_monotonic_increasing or stamps.has_duplicates:
        raise DerivativesError(
            "hours must be strictly increasing before they can be segmented"
        )
    step = stamps.to_series().diff()
    new = (step != pd.Timedelta(hours=1)).to_numpy(dtype=bool, copy=True)
    new[0] = True
    return np.cumsum(new) - 1


def _segment_bounds(segments: np.ndarray) -> list[tuple[int, int]]:
    """``[start, end)`` of each segment, in order."""
    starts = np.flatnonzero(np.concatenate(([True], segments[1:] != segments[:-1])))
    ends = np.concatenate((starts[1:], [len(segments)]))
    return [(int(a), int(b)) for a, b in zip(starts, ends)]


def _trailing_sum_and_count(
    values: np.ndarray, present: np.ndarray, window: int, lo: int, hi: int
) -> tuple[np.ndarray, np.ndarray]:
    """Sum and count of the present observations in the last ``window`` rows.

    Inclusive of the row itself (:class:`DerivativesSpec`), computed with prefix
    sums inside one segment ``[lo, hi)`` so the window can never reach across a
    market-data gap.
    """
    span = hi - lo
    contributed = np.where(present[lo:hi], values[lo:hi], 0.0)
    counted = present[lo:hi].astype(np.float64)
    csum = np.concatenate(([0.0], np.cumsum(contributed)))
    ccount = np.concatenate(([0.0], np.cumsum(counted)))
    end = np.arange(1, span + 1)
    start = np.maximum(end - window, 0)
    return csum[end] - csum[start], ccount[end] - ccount[start]


def _trailing_mean_std(
    values: np.ndarray, present: np.ndarray, window: int, lo: int, hi: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean, population std and observation count of a trailing window."""
    total, count = _trailing_sum_and_count(values, present, window, lo, hi)
    sq_total, _ = _trailing_sum_and_count(np.square(values), present, window, lo, hi)
    safe = np.maximum(count, 1.0)
    mean = total / safe
    variance = np.maximum(sq_total / safe - np.square(mean), 0.0)
    return mean, np.sqrt(variance), count


def _lagged(values: np.ndarray, present: np.ndarray, lag: int, lo: int, hi: int):
    """The observation ``lag`` rows earlier, inside one segment, and whether it exists."""
    span = hi - lo
    out = np.zeros(span, dtype=np.float64)
    ok = np.zeros(span, dtype=bool)
    if span > lag:
        out[lag:] = values[lo : hi - lag]
        ok[lag:] = present[lo : hi - lag]
    return out, ok


def compute_derivatives_features(
    hourly: pd.DataFrame,
    spot_close: np.ndarray,
    spec: DerivativesSpec | None = None,
    *,
    segment_ids: np.ndarray | None = None,
) -> pd.DataFrame:
    """The eight columns of §5, one row per hour, plus the ``defined`` mask.

    ``hourly`` is the committed derivatives source: one row per hour of the
    acquisition window, in the schema :mod:`nn.derivatives_sources` defines, with
    each field's availability and age already resolved against §3.4's bound.
    ``spot_close`` is the committed candle history's close on the same hours —
    the fourth source of §3, which is not acquired because this repository
    already holds it.

    ``segment_ids`` is §6.2's fourth condition, and it is an argument rather than
    something derived here because the research spine's segments are **stricter**
    than the candle history's hour gaps: the dataset builder drops an indicator
    warm-up after every gap, so a two-hour hole in the candles becomes a
    three-day hole in the spine. Windows must reset at the spine's boundaries,
    not at the candles' — a window that reached back over the dropped warm-up
    would bridge a boundary the fold geometry believes is there. Passing ``None``
    falls back to the frame's own hour gaps, which is correct only when the frame
    *is* one contiguous run.

    The returned frame carries the eight columns and a boolean ``defined``. A
    row is defined when every one of the eight is, which is §6.2's second
    condition; the caller adds the other three. Undefined rows carry ``0.0``,
    which is a declared placeholder and never a value: it is unreachable, because
    the universe mask removes those rows from every arm before anything is
    windowed.
    """
    spec = spec or DerivativesSpec()
    frame = hourly.reset_index(drop=True)
    missing = [name for name in DERIVATIVES_COLUMNS if name not in frame.columns]
    if missing:
        raise DerivativesError(
            f"the derivatives source is missing column(s) {missing}; "
            f"{DERIVATIVES_SPEC_VERSION} is defined over the schema in "
            "nn.derivatives_sources and cannot be computed from anything else"
        )
    close = np.asarray(spot_close, dtype=np.float64)
    if len(close) != len(frame):
        raise DerivativesError(
            f"the derivatives source has {len(frame)} hours and the candle history "
            f"{len(close)}. The basis is the perpetual close over the spot close at the "
            "same instant, so the two must be the same hours in the same order."
        )
    if not np.isfinite(close).all() or (close <= 0.0).any():
        raise DerivativesError(
            "the candle history carries a non-positive or non-finite close, which the "
            "basis and the price divergence both divide by"
        )

    if segment_ids is None:
        segments = segment_ids_from_hours(frame["date"])
    else:
        segments = np.asarray(segment_ids, dtype=np.int64)
        if len(segments) != len(frame):
            raise DerivativesError(
                f"{len(segments)} segment id(s) for {len(frame)} hours; a window that "
                "reset on the wrong row would bridge a boundary silently"
            )
        # Still checked, because passing ids does not license a frame with a hole
        # in it: "the last 168h" means 168 rows only while the rows are hours.
        segment_ids_from_hours(frame["date"])
    n = len(frame)

    funding_ok = frame["funding_available"].to_numpy(dtype=np.int64) == 1
    funding_last = frame["funding_last_rate"].to_numpy(dtype=np.float64)
    funding_age = frame["funding_age_ns"].to_numpy(dtype=np.int64)
    settled = frame["funding_settled"].to_numpy(dtype=np.int64) == 1
    settled_rate = frame["funding_settled_rate"].to_numpy(dtype=np.float64)

    oi_ok = frame["oi_available"].to_numpy(dtype=np.int64) == 1
    oi_contracts = frame["oi_contracts"].to_numpy(dtype=np.float64)
    oi_notional = frame["oi_notional"].to_numpy(dtype=np.float64)
    oi_age = frame["oi_age_ns"].to_numpy(dtype=np.int64)

    perp_ok = frame["perp_available"].to_numpy(dtype=np.int64) == 1
    perp_close = frame["perp_close"].to_numpy(dtype=np.float64)
    perp_age = frame["perp_age_ns"].to_numpy(dtype=np.int64)

    # §3.4 again, in the engine rather than only in the exporter. The universe is
    # decided here, so "no observation it depends on is stale beyond its bound"
    # has to be checked by the thing that decides it — an exporter that got the
    # cap wrong must not be able to widen the universe by writing a larger age.
    funding_ok &= funding_age >= 0
    funding_ok &= funding_age <= staleness_bound_ns("funding_rate")
    oi_ok &= oi_age >= 0
    oi_ok &= oi_age <= staleness_bound_ns("open_interest")
    perp_ok &= perp_age >= 0
    perp_ok &= perp_age <= staleness_bound_ns("perpetual_price")
    oi_ok &= np.isfinite(oi_contracts) & (oi_contracts > 0.0)
    oi_ok &= np.isfinite(oi_notional) & (oi_notional > 0.0)
    perp_ok &= np.isfinite(perp_close) & (perp_close > 0.0)
    funding_ok &= np.isfinite(funding_last)

    out = {name: np.zeros(n, dtype=np.float64) for name in DERIVATIVES_FEATURE_COLUMNS}
    ok = {name: np.zeros(n, dtype=bool) for name in DERIVATIVES_FEATURE_COLUMNS}
    #: Hours of in-segment history behind each row, for §6.2's warm-up condition.
    history = np.zeros(n, dtype=np.int64)

    basis_raw = np.zeros(n, dtype=np.float64)
    basis_raw[perp_ok] = perp_close[perp_ok] / close[perp_ok] - 1.0
    basis = _clip(basis_raw, "drv_basis")

    for lo, hi in _segment_bounds(segments):
        span = hi - lo
        history[lo:hi] = np.arange(span, dtype=np.int64)

        # --- funding -------------------------------------------------------- #
        # The settlement series, in visibility order inside this segment. §5's
        # windows count settlements rather than hours, and a settlement that
        # became visible before this segment began is not in it: §6.2's fourth
        # condition forbids a window bridging a spine segment boundary, and the
        # funding window is 30 settlements = the 240-hour warm-up exactly.
        seg_settled = settled[lo:hi]
        seg_rate = settled_rate[lo:hi]
        rates = seg_rate[seg_settled]
        # `k[i]` = how many settlements are visible at row `lo + i`.
        k = np.cumsum(seg_settled.astype(np.int64))
        prefix = np.concatenate(([0.0], np.cumsum(rates)))
        prefix_sq = np.concatenate(([0.0], np.cumsum(np.square(rates))))

        seg_funding_ok = funding_ok[lo:hi]
        last_clipped = _clip(funding_last[lo:hi], "drv_funding_last")
        out["drv_funding_last"][lo:hi] = np.where(seg_funding_ok, last_clipped, 0.0)
        ok["drv_funding_last"][lo:hi] = seg_funding_ok

        for name, stat in (("drv_funding_sum_9", "sum"), ("drv_funding_z", "z")):
            window = int(_WINDOW[name])
            enough = seg_funding_ok & (k >= window)
            start = np.maximum(k - window, 0)
            total = prefix[k] - prefix[start]
            if stat == "sum":
                out[name][lo:hi] = np.where(enough, _clip(total, name), 0.0)
            else:
                mean = total / float(window)
                sq = (prefix_sq[k] - prefix_sq[start]) / float(window)
                std = np.sqrt(np.maximum(sq - np.square(mean), 0.0))
                z = (last_clipped - mean) / (std + spec.eps)
                out[name][lo:hi] = np.where(enough, _clip(z, name), 0.0)
            ok[name][lo:hi] = enough

        # --- open interest --------------------------------------------------- #
        seg_oi_ok = oi_ok[lo:hi]
        lag = int(_WINDOW["drv_oi_log_change_24h"])
        prior_oi, prior_oi_ok = _lagged(oi_contracts, oi_ok, lag, lo, hi)
        pair_ok = seg_oi_ok & prior_oi_ok
        ratio = np.ones(span, dtype=np.float64)
        np.divide(oi_contracts[lo:hi], prior_oi, out=ratio, where=pair_ok)
        oi_log_change = np.where(pair_ok, np.log(np.where(pair_ok, ratio, 1.0)), 0.0)
        out["drv_oi_log_change_24h"][lo:hi] = _clip(oi_log_change, "drv_oi_log_change_24h")
        ok["drv_oi_log_change_24h"][lo:hi] = pair_ok

        prior_close, _ = _lagged(close, np.ones(n, dtype=bool), lag, lo, hi)
        close_ok = np.zeros(span, dtype=bool)
        close_ok[lag:] = True
        price_log_change = np.zeros(span, dtype=np.float64)
        np.log(
            np.divide(close[lo:hi], prior_close, out=np.ones(span), where=close_ok),
            out=price_log_change,
            where=close_ok,
        )
        divergence_ok = pair_ok & close_ok
        out["drv_oi_price_divergence"][lo:hi] = np.where(
            divergence_ok,
            _clip(oi_log_change - price_log_change, "drv_oi_price_divergence"),
            0.0,
        )
        ok["drv_oi_price_divergence"][lo:hi] = divergence_ok

        window = int(_WINDOW["drv_oi_notional_ratio"])
        mean_notional, _, count = _trailing_mean_std(oi_notional, oi_ok, window, lo, hi)
        ratio_ok = seg_oi_ok & (count >= spec.min_window_observations)
        safe_mean = np.where(mean_notional > 0.0, mean_notional, 1.0)
        out["drv_oi_notional_ratio"][lo:hi] = np.where(
            ratio_ok, _clip(oi_notional[lo:hi] / safe_mean, "drv_oi_notional_ratio"), 0.0
        )
        ok["drv_oi_notional_ratio"][lo:hi] = ratio_ok & (mean_notional > 0.0)

        # --- basis ------------------------------------------------------------ #
        seg_perp_ok = perp_ok[lo:hi]
        out["drv_basis"][lo:hi] = np.where(seg_perp_ok, basis[lo:hi], 0.0)
        ok["drv_basis"][lo:hi] = seg_perp_ok

        window = int(_WINDOW["drv_basis_z"])
        source = basis if spec.basis_z_window_uses_clipped_basis else basis_raw
        mean_basis, std_basis, count = _trailing_mean_std(source, perp_ok, window, lo, hi)
        z_ok = seg_perp_ok & (count >= spec.min_window_observations)
        z = (source[lo:hi] - mean_basis) / (std_basis + spec.eps)
        out["drv_basis_z"][lo:hi] = np.where(z_ok, _clip(z, "drv_basis_z"), 0.0)
        ok["drv_basis_z"][lo:hi] = z_ok

    defined = np.ones(n, dtype=bool)
    for name in DERIVATIVES_FEATURE_COLUMNS:
        defined &= ok[name]
    # §6.2's second condition names the 240-hour warm-up explicitly rather than
    # leaving it implied by the longest window, so it is checked explicitly.
    defined &= history >= spec.warmup_hours

    result = pd.DataFrame({"date": pd.to_datetime(frame["date"], utc=True)})
    for name in DERIVATIVES_FEATURE_COLUMNS:
        values = np.where(defined, out[name], 0.0)
        if not np.isfinite(values).all():
            bad = int(np.flatnonzero(~np.isfinite(values))[0])
            raise DerivativesError(
                f"{name} is non-finite at hour {bad} despite being marked defined; "
                "derivatives_v1 carries a finite value on every row it claims"
            )
        result[name] = values
    result["defined"] = defined
    result["segment_id"] = segments
    for name in DERIVATIVES_FEATURE_COLUMNS:
        result[f"{name}__defined"] = ok[name]
    result["warmup_hours_behind"] = history
    return result


def undefined_reasons(features: pd.DataFrame) -> dict[str, int]:
    """How many rows each column, and the warm-up, left undefined.

    Reported per block beside the availability gate, because "2% of the rows did
    not survive" is only actionable next to which condition removed them.
    """
    reasons = {
        name: int((~features[f"{name}__defined"].to_numpy(dtype=bool)).sum())
        for name in DERIVATIVES_FEATURE_COLUMNS
    }
    reasons["warmup_hours"] = int(
        (features["warmup_hours_behind"].to_numpy(dtype=np.int64) < WARMUP_HOURS).sum()
    )
    return reasons
