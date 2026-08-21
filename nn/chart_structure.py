"""The causal chart-structure information set of ``docs/chart_structure_v1.md``.

That document, not this module, is the definition. Every constant, formula and
firing rule here is a transcription of it, and a disagreement between the two is
a bug in one of them rather than a licence to reinterpret either.

**Vectorised where a column is a function of one trailing window, a forward pass
where it is not.** Rows 1-18 and 29-30 read only ``I_s(t)``, ``I_l(t)`` or
``I_l^-(t)``, so they are built from :func:`_trailing_view`, whose row ``t`` is
literally the window ``[t - w + 1, t]`` and nothing else. Append invariance is
then arithmetic rather than a property a test has to hope for: appending a candle
cannot change a window that does not contain it. The two mechanisms that carry
state — the pending breakout of §3.6 and the two newest confirmed pivots of §3.7
— cannot be written that way, because what row ``t`` reports depends on which
earlier rows fired. They run in one forward loop, in §8's declared order, and
that loop only ever reads state written strictly before ``t``.

**Left truncation is padding, not ``min_periods``.** ``_trailing_view`` pads the
head of the segment with NaN and every aggregate over it is nan-aware, so a
window that would begin before the segment is short by exactly the bars that do
not exist. The alternative spellings — a centred window, a ``shift(-k)``, a
backward fill — all produce a frame whose historical rows change when a new
candle arrives, which is what makes a backtest unfalsifiable.

**ATR and pivots are imported, not reimplemented.** ``_true_range``/``_wilder``
come from :mod:`chimera.features` so the denominator these features scale against
is byte-identical to the one ``atr_norm`` uses in the OHLCV14 set, and
``_raw_pivots`` comes from :mod:`nn.smc` so this family sees the identical swing
set §1.3 promises. A second implementation agreeing to within a rounding error
would make every P2b comparison between the families depend on which of the two
produced the number.

**No column is ever NaN or infinite, on any row, including row 0.**
``nn.data_pipeline.build_dataset`` drops rows with a NaN feature, so a NaN here
would evaluate the information sets on different samples and confound P2b with
the sample universe rather than the information. Where state does not exist yet
the value is §5's declared default, and the two availability flags of §4 F carry
the "not yet" — they exist precisely because ``0.0`` is a legal signed distance.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from chimera.features import _true_range, _wilder
from nn.data_pipeline import _contiguous_segment_ids
from nn.smc import _raw_pivots

CHART_SPEC_VERSION: str = "chart_structure_v1"

#: Family name -> its columns, the six families of §4 in order. The P2b ablation
#: drops one family at a time, so the grouping is part of the contract too.
CHART_FEATURE_FAMILIES: dict[str, tuple[str, ...]] = {
    "range": (
        "chart_window_fill",
        "chart_range_width_atr",
        "chart_range_pos",
        "chart_range_pos_fast",
        "chart_touch_high_frac",
        "chart_touch_low_frac",
    ),
    "compression": (
        "chart_high_slope_atr",
        "chart_low_slope_atr",
        "chart_convergence_atr",
        "chart_slope_asymmetry",
    ),
    "channel": (
        "chart_trend_slope_atr",
        "chart_trend_r2",
        "chart_channel_disp_atr",
        "chart_channel_pos",
    ),
    "volatility": (
        "chart_range_ratio",
        "chart_tr_ratio",
    ),
    "breakout": (
        "chart_breakout_up_atr",
        "chart_breakout_dn_atr",
        "chart_failed_breakout_up_atr",
        "chart_failed_breakout_dn_atr",
    ),
    "patterns": (
        "chart_dt_valid",
        "chart_dt_offset_atr",
        "chart_dt_trough_atr",
        "chart_dt_neckline_atr",
        "chart_db_valid",
        "chart_db_offset_atr",
        "chart_db_peak_atr",
        "chart_db_neckline_atr",
        "chart_pole_atr",
        "chart_impulse_atr",
    ),
}

#: The 30 columns of §4, in that exact order. Never reordered: the feature-spec
#: hash and every persisted prediction record depend on it. Derived from the
#: families so the partition of §7 cannot drift out of agreement with them.
CHART_FEATURE_NAMES: tuple[str, ...] = tuple(
    name for family in CHART_FEATURE_FAMILIES.values() for name in family
)

_REQUIRED_COLUMNS = ("open", "high", "low", "close")


@dataclass(frozen=True)
class ChartSpec:
    """The predeclared constants of §1.3.

    Frozen because they were fixed before any P2b outer-validation result was
    observed and tuning them against one is forbidden; a mutable spec would make
    "the constants were not searched" a claim about discipline rather than about
    the object recorded in the artifact.
    """

    atr_period: int = 14
    win_short: int = 20
    win_long: int = 60
    touch_atr: float = 0.25
    pivot_left: int = 3
    pivot_right: int = 3

    def to_dict(self) -> dict[str, int | float]:
        # asdict, not a float cast: the canonical JSON must read "win_long":60,
        # and a cast to satisfy a dict[str, float] annotation would change the
        # hash. That is smc_v1 §9(d), recorded so it is not repeated here.
        return asdict(self)

    def spec_hash(self) -> str:
        """sha256 over canonical JSON of version + constants + feature names."""
        payload = json.dumps(
            {
                "version": CHART_SPEC_VERSION,
                "constants": self.to_dict(),
                "feature_names": list(CHART_FEATURE_NAMES),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()


def chart_feature_columns() -> list[str]:
    """The feature names, in the order :func:`compute_chart_features` emits them."""
    return list(CHART_FEATURE_NAMES)


def _trailing_view(values: np.ndarray, window: int) -> np.ndarray:
    """The windows ``I_n(t)`` of §1.2 as an ``(n, window)`` array, NaN before row 0.

    Row ``t`` holds ``values[t - window + 1 … t]``, with the entries that fall
    before the segment start padded with NaN, so a nan-aware reduction over
    ``axis=1`` is exactly the left-truncated trailing aggregate: never centred,
    never reaching past ``t``, and short only where the history really is.
    """
    padded = np.full(len(values) + window - 1, np.nan, dtype="float64")
    padded[window - 1 :] = values
    return sliding_window_view(padded, window)


def _rolling_slope(
    view: np.ndarray, xc: np.ndarray, sxx: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """§3.1's least-squares slope per window, and the window-centred series.

    ``xc`` is the centred regressor ``x - xbar`` and ``sxx = w(w^2-1)/12`` its own
    sum of squares — a function of the effective length alone, zero only at
    ``w == 1``, which is the single case §5 declares to be ``0.0``. The centred
    series is returned rather than recomputed because §3.4's residuals are built
    from it.
    """
    yc = view - np.nanmean(view, axis=1)[:, None]
    num = np.nansum(yc * xc, axis=1)
    slope = np.zeros(len(sxx), dtype="float64")
    np.divide(num, sxx, out=slope, where=sxx > 0.0)
    return slope, yc


def compute_chart_features(df: pd.DataFrame, spec: ChartSpec | None = None) -> pd.DataFrame:
    """Chart-structure features for ONE contiguous segment.

    ``df`` needs ``open``/``high``/``low``/``close`` (``volume`` is accepted and
    unused). The result has ``df``'s index and exactly ``CHART_FEATURE_NAMES`` as
    columns, all float64, and contains no NaN and no infinity.
    """
    spec = spec or ChartSpec()
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"OHLC frame is missing columns: {missing}")

    high = df["high"].astype("float64")
    low = df["low"].astype("float64")
    close = df["close"].astype("float64")

    true_range = _true_range(pd.DataFrame({"high": high, "low": low, "close": close}))
    atr = _wilder(true_range, spec.atr_period)
    h = high.to_numpy()
    lo_ = low.to_numpy()
    c = close.to_numpy()
    tr = true_range.to_numpy()
    # A flat market drives ATR to zero; the floor keeps every division finite
    # without pretending the range was larger than it was.
    atr_den = np.maximum(atr.to_numpy(), 1e-12 * c)
    n = len(df)

    long_view_h = _trailing_view(h, spec.win_long)
    long_view_l = _trailing_view(lo_, spec.win_long)
    long_view_c = _trailing_view(c, spec.win_long)
    w_l = np.minimum(np.arange(1.0, n + 1.0), float(spec.win_long))

    # §3.2, the range box. t is inside its own window, so L_l <= c[t] <= H_l and
    # the position needs no clip.
    h_l = np.nanmax(long_view_h, axis=1)
    l_l = np.nanmin(long_view_l, axis=1)
    h_s = np.nanmax(_trailing_view(h, spec.win_short), axis=1)
    l_s = np.nanmin(_trailing_view(lo_, spec.win_short), axis=1)
    width_l = h_l - l_l
    width_s = h_s - l_s

    range_pos = np.full(n, 0.5, dtype="float64")
    np.divide(c - l_l, width_l, out=range_pos, where=width_l > 0.0)
    range_pos_fast = np.full(n, 0.5, dtype="float64")
    np.divide(c - l_s, width_s, out=range_pos_fast, where=width_s > 0.0)

    # One scale for the whole window (§3.2): a per-bar tolerance would make the
    # count depend on how volatility moved inside the window.
    tolerance = spec.touch_atr * atr_den
    touch_high = np.count_nonzero(long_view_h >= (h_l - tolerance)[:, None], axis=1)
    touch_low = np.count_nonzero(long_view_l <= (l_l + tolerance)[:, None], axis=1)

    # §3.1: x runs over [-(w-1), 0] with the window's own index as the regressor.
    x = np.arange(spec.win_long, dtype="float64") - (spec.win_long - 1.0)
    xbar = -(w_l - 1.0) / 2.0
    sxx = w_l * (w_l * w_l - 1.0) / 12.0
    xc = x[None, :] - xbar[:, None]

    high_slope = _rolling_slope(long_view_h, xc, sxx)[0] / atr_den
    low_slope = _rolling_slope(long_view_l, xc, sxx)[0] / atr_den
    slope_sum = np.abs(high_slope) + np.abs(low_slope)
    asymmetry = np.zeros(n, dtype="float64")
    np.divide(high_slope + low_slope, slope_sum, out=asymmetry, where=slope_sum > 0.0)

    # §3.4, the trend channel: one fit, reused for slope, dispersion, fit quality
    # and the standardised current residual.
    trend_slope, cc = _rolling_slope(long_view_c, xc, sxx)
    resid = cc - trend_slope[:, None] * xc
    rss = np.nansum(resid * resid, axis=1)
    sst = np.nansum(cc * cc, axis=1)
    channel_disp = np.sqrt(rss / w_l)
    unexplained = np.zeros(n, dtype="float64")
    np.divide(rss, sst, out=unexplained, where=sst > 0.0)
    # The clip absorbs float noise only; it cannot move a value further (§3.4).
    trend_r2 = np.where(sst > 0.0, 1.0 - unexplained, 0.0).clip(0.0, 1.0)
    channel_pos = np.zeros(n, dtype="float64")
    # `sigma == 0` in §3.4 means "the fit is perfect", and a computed sqrt
    # almost never lands on zero when it does: on a flat window the residuals
    # come out around 1e-14, the exact-zero guard misses, and the ratio of two
    # rounding errors is a number of order one. That would hand the model a
    # measurement of nothing, bounded by sqrt(w-1) and indistinguishable from a
    # real reading. Nine rows of the committed pre-Styx history do exactly this
    # under an exact guard. The threshold is relative to ATR because sigma is a
    # price-scale quantity and 1e-9 ATR is not a channel.
    # Two floors, not one. `1e-9 * atr_den` is the ATR-scale test — but `atr_den`
    # itself floors at `1e-12 * c`, so on a stretch where ATR has decayed toward
    # that floor the composed threshold reaches `1e-21 * c`, roughly five orders
    # BELOW the ULP of `c`. The guard then cannot fire in the one regime it
    # exists for: 216 of 300 frozen candles after a real BTC prefix still emitted
    # a nonzero standardised residual under the ATR-only test. The absolute floor
    # sits ~4 orders above float noise and ~8 below the smallest real
    # `channel_disp` on the committed history, so it closes the hole without
    # touching a single existing row.
    degenerate = channel_disp > np.maximum(1e-9 * atr_den, 1e-12 * c)
    np.divide(resid[:, -1], channel_disp, out=channel_pos, where=degenerate)

    # §3.5. Neither ratio is ATR-normalised: both are ratios of like to like.
    range_ratio = np.ones(n, dtype="float64")
    np.divide(width_s, width_l, out=range_ratio, where=width_l > 0.0)
    tr_mean_l = np.nanmean(_trailing_view(tr, spec.win_long), axis=1)
    tr_mean_s = np.nanmean(_trailing_view(tr, spec.win_short), axis=1)
    tr_ratio = np.ones(n, dtype="float64")
    np.divide(tr_mean_s, tr_mean_l, out=tr_ratio, where=tr_mean_l > 0.0)

    # §3.6 step 2: I_l^-(t) excludes t, and it is exactly I_l(t-1). Slot 0 is the
    # undefined prior window and is never read — rows 17/18 are 0.0 there per §5
    # and no pending break can open on a segment's first candle.
    prior_high = np.concatenate(([h[0]], h_l[:-1]))
    prior_low = np.concatenate(([lo_[0]], l_l[:-1]))
    breakout_up = np.zeros(n, dtype="float64")
    breakout_dn = np.zeros(n, dtype="float64")
    breakout_up[1:] = (c[1:] - prior_high[1:]) / atr_den[1:]
    breakout_dn[1:] = (prior_low[1:] - c[1:]) / atr_den[1:]

    # §3.8, clamped at the segment start rather than truncated: the displacement
    # over the bars that exist is honest, a NaN would not be.
    back_short = np.maximum(np.arange(n) - spec.win_short, 0)
    back_long = np.maximum(np.arange(n) - spec.win_long, 0)
    pole = (c[back_short] - c[back_long]) / atr_den
    impulse = (c - c[back_short]) / atr_den

    is_swing_high, is_swing_low = _raw_pivots(h, lo_, spec.pivot_left, spec.pivot_right)

    failed_up = np.zeros(n, dtype="float64")
    failed_dn = np.zeros(n, dtype="float64")
    dt_valid = np.zeros(n, dtype="float64")
    dt_offset = np.zeros(n, dtype="float64")
    dt_trough = np.zeros(n, dtype="float64")
    dt_neckline = np.zeros(n, dtype="float64")
    db_valid = np.zeros(n, dtype="float64")
    db_offset = np.zeros(n, dtype="float64")
    db_peak = np.zeros(n, dtype="float64")
    db_neckline = np.zeros(n, dtype="float64")

    last_sh_i = last_sl_i = -1
    has_dt = has_db = False
    dt_off = dt_tro = dt_low = 0.0
    db_off = db_pk = db_high = 0.0
    pending_up_i = pending_dn_i = -1
    pending_up_level = pending_dn_level = 0.0

    for t in range(n):
        den = atr_den[t]
        c_t = c[t]

        # 1. absorb the pivot confirmed by this candle: it becomes i2 / j2 at
        #    once, because this is the first row on which it exists (§2(b), §8).
        pivot = t - spec.pivot_right
        if pivot >= 0:
            if is_swing_high[pivot]:
                if last_sh_i >= 0:
                    has_dt = True
                    dt_low = lo_[last_sh_i : pivot + 1].min()
                    dt_off = h[pivot] - h[last_sh_i]
                    dt_tro = min(h[last_sh_i], h[pivot]) - dt_low
                last_sh_i = pivot
            if is_swing_low[pivot]:
                if last_sl_i >= 0:
                    has_db = True
                    db_high = h[last_sl_i : pivot + 1].max()
                    db_off = lo_[pivot] - lo_[last_sl_i]
                    db_pk = db_high - max(lo_[last_sl_i], lo_[pivot])
                last_sl_i = pivot

        # 3. expire: the level that defined the break has rolled out of the
        #    window every other feature can see, so it stops being geometry.
        if pending_up_i >= 0 and t - pending_up_i > spec.win_long:
            pending_up_i = -1
        if pending_dn_i >= 0 and t - pending_dn_i > spec.win_long:
            pending_dn_i = -1

        # 4. resolve against the level R recorded when the break opened — not
        #    against prior_high(t), which is a level the break never happened at.
        if pending_up_i >= 0 and c_t < pending_up_level:
            failed_up[t] = (pending_up_level - c_t) / den
            pending_up_i = -1
        if pending_dn_i >= 0 and c_t > pending_dn_level:
            failed_dn[t] = (c_t - pending_dn_level) / den
            pending_dn_i = -1

        # 5. open or replace, after step 4 so one candle can resolve an up-break
        #    and open a down-break. Replace, never stack: the question is about
        #    the level price is currently beyond.
        if t > 0:
            if c_t > prior_high[t]:
                pending_up_i, pending_up_level = t, prior_high[t]
            if c_t < prior_low[t]:
                pending_dn_i, pending_dn_level = t, prior_low[t]

        # 7. the state-dependent half of the row (§3.7). "Else 0" is per quantity:
        #    the side that has confirmed twice reports even if the other has not.
        if has_dt:
            dt_valid[t] = 1.0
            dt_offset[t] = dt_off / den
            dt_trough[t] = dt_tro / den
            dt_neckline[t] = (c_t - dt_low) / den
        if has_db:
            db_valid[t] = 1.0
            db_offset[t] = db_off / den
            db_peak[t] = db_pk / den
            db_neckline[t] = (db_high - c_t) / den

    out = np.column_stack(
        (
            w_l / spec.win_long,
            width_l / atr_den,
            range_pos,
            range_pos_fast,
            touch_high / w_l,
            touch_low / w_l,
            high_slope,
            low_slope,
            low_slope - high_slope,
            asymmetry,
            trend_slope / atr_den,
            trend_r2,
            channel_disp / atr_den,
            channel_pos,
            range_ratio,
            tr_ratio,
            breakout_up,
            breakout_dn,
            failed_up,
            failed_dn,
            dt_valid,
            dt_offset,
            dt_trough,
            dt_neckline,
            db_valid,
            db_offset,
            db_peak,
            db_neckline,
            pole,
            impulse,
        )
    )
    return pd.DataFrame(out, index=df.index, columns=list(CHART_FEATURE_NAMES))


def compute_chart_features_segmented(
    candles: pd.DataFrame, timeframe: str, spec: ChartSpec | None = None
) -> pd.DataFrame:
    """Split ``candles`` on market-data gaps, compute per segment, concatenate.

    Segments come from :func:`nn.data_pipeline._contiguous_segment_ids`, so the
    boundaries are the ones the processed dataset already used. State never
    crosses a boundary.
    """
    segment_ids = _contiguous_segment_ids(candles["date"], timeframe)
    blocks = [
        compute_chart_features(candles[segment_ids == segment_id], spec)
        for segment_id in segment_ids.unique()
    ]
    return pd.concat(blocks)
