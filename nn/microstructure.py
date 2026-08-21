"""`microstructure_v1`: 32 causal trade-flow columns. Spec: docs/microstructure_v1.md.

P3 asks whether individual executions carry information the hourly candle does
not. This module is the answer's input half: it turns the hourly sufficient
statistics :mod:`nn.trade_aggregates` exports into the 32 columns
``docs/microstructure_v1.md`` §3 defines, and nothing else in this repository
computes a microstructure feature.

**Everything here is causal by construction.** A row reads its own aggregate row
and the rows strictly before it in the same segment. There is no whole-history
statistic anywhere: the one quantile the family uses — the "large trade"
threshold of §C — is taken over the *previous week's* pooled size histogram, and
:func:`_large_trade_bin` is written so that the future is not merely unused but
unreachable, because the array it reads has already been shifted.

**Every column is finite on every row.** ``docs/microstructure_v1.md`` §5 says
why that matters and not just that it is true: a ``NaN`` would be dropped by the
dataset builder, every later row would shift, and the fold boundaries — which are
row indices — would land on different candles for this arm than for the OHLCV14
control. The two arms would then be compared over different market periods and
nothing would raise.

**Trailing windows stop at a gap.** An hour in which BTCUSDT recorded no trade
has no row, and a mean taken across a two-day hole would compare this week
against last month. Segments are the maximal runs of consecutive hours, exactly
as ``smc_v1`` resets its state at a market-data gap, and a row without a full
in-segment window takes the declared default from §3 — always the value meaning
"indistinguishable from typical".
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from nn.trade_aggregates import (
    PRICE_BINS,
    QUARTERS,
    SECONDS_PER_HOUR,
    SIZE_BIN_EDGES,
    SIZE_BINS,
)

#: Names the meaning of this feature family. Recorded in every P3 artifact, so a
#: later `microstructure_v2` cannot be mistaken for this one in a comparison.
MICROSTRUCTURE_SPEC_VERSION = "microstructure_v1"

#: The 32 columns of docs/microstructure_v1.md §3, in that exact order.
#:
#: Never reordered, never renamed, never extended inside `v1`: the feature-spec
#: hash, the flattened column names in every persisted prediction record and the
#: arm check in `nn.p2b_compare.load_cell` all depend on this list being what it
#: was when the cells ran.
MICROSTRUCTURE_FEATURE_COLUMNS: tuple[str, ...] = (
    # A. intensity
    "ms_log_trade_count",
    "ms_trade_count_ratio_24",
    "ms_trade_count_ratio_168",
    "ms_trade_count_accel",
    "ms_trades_per_active_second",
    # B. aggressive-flow imbalance
    "ms_qty_imbalance",
    "ms_notional_imbalance",
    "ms_count_imbalance",
    "ms_notional_imbalance_z24",
    # C. trade size
    "ms_mean_size_ratio_168",
    "ms_median_size_ratio_168",
    "ms_large_trade_count_share",
    "ms_large_trade_qty_share",
    # D. burstiness
    "ms_second_dispersion",
    "ms_active_second_share",
    "ms_max_second_share",
    "ms_max_silence_share",
    # E. displacement per volume
    "ms_displacement_per_notional",
    "ms_range_per_notional",
    "ms_signed_flow_return",
    # F. absorption-like
    "ms_absorption_like",
    "ms_volume_without_range",
    "ms_unconverted_pressure",
    # G. intra-hour distribution
    "ms_q2_trade_share",
    "ms_q3_trade_share",
    "ms_q4_trade_share",
    "ms_late_early_qty_imbalance",
    "ms_q4_flow_imbalance",
    # H. volume-at-price approximations
    "ms_price_hhi",
    "ms_price_entropy",
    "ms_close_from_vwap",
    "ms_close_from_node",
)

#: Family name -> its columns, the eight groups of §3. A partition, asserted at
#: import. Declared here rather than derived from whatever turned out to matter,
#: for the reason `nn.information_sets.InformationSet` gives: groups chosen after
#: seeing a result turn a diagnostic into a search.
MICROSTRUCTURE_FEATURE_FAMILIES: dict[str, tuple[str, ...]] = {
    "intensity": MICROSTRUCTURE_FEATURE_COLUMNS[0:5],
    "flow_imbalance": MICROSTRUCTURE_FEATURE_COLUMNS[5:9],
    "trade_size": MICROSTRUCTURE_FEATURE_COLUMNS[9:13],
    "burstiness": MICROSTRUCTURE_FEATURE_COLUMNS[13:17],
    "displacement": MICROSTRUCTURE_FEATURE_COLUMNS[17:20],
    "absorption_like": MICROSTRUCTURE_FEATURE_COLUMNS[20:23],
    "intra_hour": MICROSTRUCTURE_FEATURE_COLUMNS[23:28],
    "volume_at_price": MICROSTRUCTURE_FEATURE_COLUMNS[28:32],
}

_covered = [c for group in MICROSTRUCTURE_FEATURE_FAMILIES.values() for c in group]
assert sorted(_covered) == sorted(
    MICROSTRUCTURE_FEATURE_COLUMNS
), "the microstructure families do not partition the column list"


def microstructure_feature_columns() -> list[str]:
    """The 32 columns of §3, in order. A copy, so a caller cannot reorder them."""
    return list(MICROSTRUCTURE_FEATURE_COLUMNS)


@dataclass(frozen=True)
class MicrostructureSpec:
    """The predeclared constants of ``docs/microstructure_v1.md`` §1.6.

    Frozen and hashed for the same reason :class:`nn.smc.SMCSpec` is: these
    numbers were fixed before any P3 fold was scored, and a run under different
    ones is a different information set wearing this one's name. The hash travels
    in every artifact so a reader can tell the two apart without the source.
    """

    w_short: int = 24
    w_long: int = 168
    large_trade_quantile: float = 0.99
    ratio_clip: float = 10.0
    wide_clip: float = 20.0
    z_clip: float = 5.0
    eps: float = 1e-12

    def to_dict(self) -> dict[str, Any]:
        return {
            "w_short": self.w_short,
            "w_long": self.w_long,
            "large_trade_quantile": self.large_trade_quantile,
            "ratio_clip": self.ratio_clip,
            "wide_clip": self.wide_clip,
            "z_clip": self.z_clip,
            "eps": self.eps,
            "seconds_per_hour": SECONDS_PER_HOUR,
            "quarters": QUARTERS,
            "price_bins": PRICE_BINS,
            "size_bins": SIZE_BINS,
            "size_bin_edges": list(SIZE_BIN_EDGES),
        }

    def spec_hash(self) -> str:
        material = {
            "version": MICROSTRUCTURE_SPEC_VERSION,
            "constants": self.to_dict(),
            "columns": list(MICROSTRUCTURE_FEATURE_COLUMNS),
            "families": {k: list(v) for k, v in MICROSTRUCTURE_FEATURE_FAMILIES.items()},
        }
        return hashlib.sha256(
            json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def with_windows(self, *, w_short: int, w_long: int) -> "MicrostructureSpec":
        """A spec with shorter windows, for tests that cannot afford 168 rows.

        Deliberately the only mutator, and deliberately not reachable from any
        entrypoint: the committed constants are what P3 runs under, and a CLI
        that could change a window would be a way to search one.
        """
        return replace(self, w_short=w_short, w_long=w_long)


#: Nominal edges of the open outermost size bins, one decade beyond the outermost
#: real edge. A declared convention for interpolating a median inside an
#: unbounded bin, not a measurement — see docs/microstructure_v1.md §C.
_BIN_LO = np.array([SIZE_BIN_EDGES[0] / 10.0, *SIZE_BIN_EDGES], dtype=np.float64)
_BIN_HI = np.array([*SIZE_BIN_EDGES, SIZE_BIN_EDGES[-1] * 10.0], dtype=np.float64)


class MicrostructureError(ValueError):
    """The aggregate table cannot produce an honest microstructure row."""


def _segments(dates: pd.Series) -> pd.Series:
    """Maximal runs of consecutive hours, numbered in order.

    Computed here rather than imported from :mod:`nn.data_pipeline` because that
    helper is defined over a candle table's timeframe string, and this table's
    bucket is fixed at one hour by :mod:`nn.trade_aggregates`. The rule is the
    same one: a break of more than one bar starts a new segment.
    """
    stamps = pd.DatetimeIndex(pd.to_datetime(dates, utc=True))
    if len(stamps) == 0:
        raise MicrostructureError("an empty aggregate table has no segments")
    if not stamps.is_monotonic_increasing or stamps.has_duplicates:
        raise MicrostructureError(
            "aggregate hours must be strictly increasing before they can be segmented"
        )
    # `copy=True` because pandas 3 hands back a read-only view here, and the
    # first row has to be marked as starting a segment.
    step = stamps.to_series().diff()
    new = (step != pd.Timedelta(hours=1)).to_numpy(dtype=bool, copy=True)
    new[0] = True
    return pd.Series(np.cumsum(new) - 1, dtype=np.int64)


def _trailing(values: pd.Series, segments: pd.Series, window: int, how: str) -> pd.Series:
    """``how`` of the ``window`` rows strictly before each row, within its segment.

    ``min_periods=window`` and then ``shift(1)``, in that order and both inside
    the group. A partially filled window is left undefined rather than averaged
    over whatever happens to be there, because a feature whose value depended on
    how much history the snapshot began with would break the append-invariance
    property exactly, and the caller substitutes a declared default instead.
    """

    def roll(series: pd.Series) -> pd.Series:
        rolled = series.rolling(window, min_periods=window)
        computed = rolled.std(ddof=0) if how == "std" else getattr(rolled, how)()
        return computed.shift(1)

    # `reindex` rather than trusting the concatenation order: segments are
    # contiguous today, so `apply` happens to return the rows in their original
    # order, and a feature engine should not depend on "happens to".
    rolled = values.groupby(segments, group_keys=False).apply(roll)
    return rolled.reindex(values.index)


def _histogram_median(counts: np.ndarray) -> np.ndarray:
    """Histogram-implied median trade size per row, log-interpolated in its bin.

    Not the median of the trades — the median of the distribution the histogram
    describes, under a log-uniform assumption inside the bin the halfway point
    lands in. ``docs/microstructure_v1.md`` §C says so, and the value is only
    ever used as a ratio to the same statistic over a trailing window, where the
    interpolation convention largely cancels.

    Rows whose counts sum to zero — a trailing window that is not full — return
    ``0.0``, and the caller's default handles them.
    """
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum(axis=1)
    out = np.zeros(len(counts), dtype=np.float64)
    live = total > 0
    if not live.any():
        return out
    cum = np.cumsum(counts[live], axis=1)
    target = 0.5 * total[live]
    # First bin whose cumulative count reaches the halfway point. `argmax` on a
    # boolean row returns the first True, and the last column is always True
    # because the cumulative sum ends at the total.
    idx = np.argmax(cum >= target[:, None], axis=1)
    rows = np.arange(len(idx))
    before = np.where(idx > 0, cum[rows, np.maximum(idx - 1, 0)], 0.0)
    width = counts[live][rows, idx]
    frac = np.where(width > 0, (target - before) / np.maximum(width, 1.0), 0.0)
    lo, hi = _BIN_LO[idx], _BIN_HI[idx]
    out[live] = np.exp(np.log(lo) + np.clip(frac, 0.0, 1.0) * (np.log(hi) - np.log(lo)))
    return out


def _large_trade_bin(trailing_counts: np.ndarray, quantile: float) -> np.ndarray:
    """The smallest size bin whose trailing cumulative count share reaches ``quantile``.

    ``trailing_counts`` is already the *shifted* pooled histogram of the previous
    ``w_long`` hours, so the future is unreachable here rather than merely
    unused. A row whose window is not full has a zero histogram and gets
    :data:`SIZE_BINS` — a bin index past the end, which makes "strictly above it"
    the empty set and the §C12–13 shares exactly their declared default of zero.
    """
    counts = np.asarray(trailing_counts, dtype=np.float64)
    total = counts.sum(axis=1)
    out = np.full(len(counts), SIZE_BINS, dtype=np.int64)
    live = total > 0
    if not live.any():
        return out
    share = np.cumsum(counts[live], axis=1) / total[live][:, None]
    out[live] = np.argmax(share >= quantile, axis=1)
    return out


def compute_microstructure_features(
    aggregates: pd.DataFrame, spec: MicrostructureSpec | None = None
) -> pd.DataFrame:
    """The 32 columns of §3, one row per aggregate row, in the source's own order.

    ``aggregates`` is the hourly table :mod:`nn.trade_aggregates` defines, read
    from the committed trade snapshot. It is not re-validated here — the runner
    verifies the whole snapshot before this is reached, and re-deriving a weaker
    version of those checks would be a second definition of what a valid source
    is — but the columns this reads must be present, and their absence is named
    rather than raised as a ``KeyError`` from inside an expression.
    """
    spec = spec or MicrostructureSpec()
    frame = aggregates.reset_index(drop=True)
    needed = [
        "date",
        "n_trades",
        "buy_trades",
        "qty",
        "buy_qty",
        "notional",
        "buy_notional",
        "first_price",
        "last_price",
        "high_price",
        "low_price",
        "active_seconds",
        "second_count_sq_sum",
        "max_second_trades",
        "max_empty_run_seconds",
    ]
    missing = [name for name in needed if name not in frame.columns]
    if missing:
        raise MicrostructureError(
            f"the trade aggregate table is missing column(s) {missing}; "
            f"{MICROSTRUCTURE_SPEC_VERSION} is defined over the schema in "
            "nn.trade_aggregates and cannot be computed from anything else"
        )

    eps = spec.eps
    segments = _segments(frame["date"])

    n = frame["n_trades"].to_numpy(dtype=np.float64)
    nb = frame["buy_trades"].to_numpy(dtype=np.float64)
    qty = frame["qty"].to_numpy(dtype=np.float64)
    buy_qty = frame["buy_qty"].to_numpy(dtype=np.float64)
    notional = frame["notional"].to_numpy(dtype=np.float64)
    buy_notional = frame["buy_notional"].to_numpy(dtype=np.float64)
    p0 = frame["first_price"].to_numpy(dtype=np.float64)
    p1 = frame["last_price"].to_numpy(dtype=np.float64)
    ph = frame["high_price"].to_numpy(dtype=np.float64)
    pl = frame["low_price"].to_numpy(dtype=np.float64)
    active = frame["active_seconds"].to_numpy(dtype=np.float64)
    sq_sum = frame["second_count_sq_sum"].to_numpy(dtype=np.float64)
    max_second = frame["max_second_trades"].to_numpy(dtype=np.float64)
    max_silence = frame["max_empty_run_seconds"].to_numpy(dtype=np.float64)

    vwap = notional / np.maximum(qty, eps)
    imbalance = (2.0 * buy_notional - notional) / np.maximum(notional, eps)
    displacement = (p1 - p0) / np.maximum(p0, eps)
    span = (ph - pl) / np.maximum(vwap, eps)
    mean_size = qty / np.maximum(n, 1.0)

    def trailing(values: np.ndarray, window: int, how: str = "mean") -> np.ndarray:
        return _trailing(pd.Series(values), segments, window, how).to_numpy(dtype=np.float64)

    mean_n_24 = trailing(n, spec.w_short)
    mean_n_168 = trailing(n, spec.w_long)
    mean_i_24 = trailing(imbalance, spec.w_short)
    std_i_24 = trailing(imbalance, spec.w_short, "std")
    mean_size_168 = trailing(mean_size, spec.w_long)
    mean_absr_168 = trailing(np.abs(displacement), spec.w_long)
    mean_notional_168 = trailing(notional, spec.w_long)
    mean_span_168 = trailing(span, spec.w_long)

    def ratio(numer: np.ndarray, denom: np.ndarray, default: float, clip: float) -> np.ndarray:
        """``numer / denom``, clipped, with ``default`` where ``denom`` is undefined."""
        safe = np.where(np.isfinite(denom), denom, 1.0)
        value = np.clip(numer / np.maximum(safe, eps), 0.0, clip)
        return np.where(np.isfinite(denom), value, default)

    # -- normalised intermediates of §3 ------------------------------------- #
    d = ratio(np.abs(displacement), mean_absr_168, 1.0, spec.wide_clip)
    w = ratio(notional, mean_notional_168, 1.0, spec.wide_clip)
    g = ratio(span, mean_span_168, 1.0, spec.wide_clip)
    known_absr = np.isfinite(mean_absr_168)
    absr_scale = np.where(known_absr, np.maximum(mean_absr_168, eps), 1.0)

    columns: dict[str, np.ndarray] = {}

    # -- A. intensity ------------------------------------------------------- #
    columns["ms_log_trade_count"] = np.log1p(n)
    columns["ms_trade_count_ratio_24"] = ratio(n, mean_n_24, 1.0, spec.ratio_clip)
    columns["ms_trade_count_ratio_168"] = ratio(n, mean_n_168, 1.0, spec.ratio_clip)
    accel = np.clip(
        np.where(np.isfinite(mean_n_24) & np.isfinite(mean_n_168), mean_n_24, 0.0)
        / np.where(np.isfinite(mean_n_168), np.maximum(mean_n_168, eps), 1.0)
        - 1.0,
        -1.0,
        5.0,
    )
    columns["ms_trade_count_accel"] = np.where(
        np.isfinite(mean_n_24) & np.isfinite(mean_n_168), accel, 0.0
    )
    columns["ms_trades_per_active_second"] = np.log1p(n / np.maximum(active, 1.0))

    # -- B. aggressive-flow imbalance --------------------------------------- #
    columns["ms_qty_imbalance"] = (2.0 * buy_qty - qty) / np.maximum(qty, eps)
    columns["ms_notional_imbalance"] = imbalance
    columns["ms_count_imbalance"] = (2.0 * nb - n) / np.maximum(n, 1.0)
    known_z = np.isfinite(mean_i_24) & np.isfinite(std_i_24)
    z = np.clip(
        (imbalance - np.where(known_z, mean_i_24, 0.0))
        / np.where(known_z, np.maximum(std_i_24, eps), 1.0),
        -spec.z_clip,
        spec.z_clip,
    )
    columns["ms_notional_imbalance_z24"] = np.where(known_z, z, 0.0)

    # -- C. trade size ------------------------------------------------------ #
    size_counts = frame[[f"size_count_{b}" for b in range(SIZE_BINS)]].to_numpy(
        dtype=np.float64
    )
    size_qty = frame[[f"size_qty_{b}" for b in range(SIZE_BINS)]].to_numpy(dtype=np.float64)
    trailing_counts = np.column_stack(
        [trailing(size_counts[:, b], spec.w_long, "sum") for b in range(SIZE_BINS)]
    )
    pooled = np.where(np.isfinite(trailing_counts), trailing_counts, 0.0)

    columns["ms_mean_size_ratio_168"] = ratio(mean_size, mean_size_168, 1.0, spec.ratio_clip)
    median_now = _histogram_median(size_counts)
    median_past = _histogram_median(pooled)
    columns["ms_median_size_ratio_168"] = np.where(
        median_past > 0.0,
        np.clip(median_now / np.maximum(median_past, eps), 0.0, spec.ratio_clip),
        1.0,
    )
    threshold_bin = _large_trade_bin(pooled, spec.large_trade_quantile)
    above = np.arange(SIZE_BINS)[None, :] > threshold_bin[:, None]
    columns["ms_large_trade_count_share"] = (size_counts * above).sum(axis=1) / np.maximum(
        n, 1.0
    )
    columns["ms_large_trade_qty_share"] = (size_qty * above).sum(axis=1) / np.maximum(qty, eps)

    # -- D. burstiness ------------------------------------------------------ #
    # `S2/n - n/S` is >= 0 by Cauchy-Schwarz (see the spec), so `log1p` is
    # always defined; `maximum(..., 0)` guards the last bit of float error
    # rather than a real negative.
    fano = np.maximum(sq_sum / np.maximum(n, 1.0) - n / SECONDS_PER_HOUR, 0.0)
    columns["ms_second_dispersion"] = np.log1p(fano)
    columns["ms_active_second_share"] = active / SECONDS_PER_HOUR
    columns["ms_max_second_share"] = max_second / np.maximum(n, 1.0)
    columns["ms_max_silence_share"] = max_silence / SECONDS_PER_HOUR

    # -- E. displacement per volume ----------------------------------------- #
    columns["ms_displacement_per_notional"] = np.clip(
        d / np.maximum(w, eps), 0.0, spec.wide_clip
    )
    columns["ms_range_per_notional"] = np.clip(g / np.maximum(w, eps), 0.0, spec.wide_clip)
    columns["ms_signed_flow_return"] = np.where(
        known_absr, np.clip(imbalance * displacement / absr_scale, -10.0, 10.0), 0.0
    )

    # -- F. absorption-like ------------------------------------------------- #
    columns["ms_absorption_like"] = np.abs(imbalance) / (1.0 + d)
    columns["ms_volume_without_range"] = np.clip(w / (1.0 + g), 0.0, spec.wide_clip)
    columns["ms_unconverted_pressure"] = imbalance - np.where(
        known_absr, np.clip(displacement / absr_scale, -1.0, 1.0), 0.0
    )

    # -- G. intra-hour distribution ----------------------------------------- #
    quarter_trades = frame[[f"q{k}_trades" for k in range(QUARTERS)]].to_numpy(
        dtype=np.float64
    )
    quarter_qty = frame[[f"q{k}_qty" for k in range(QUARTERS)]].to_numpy(dtype=np.float64)
    quarter_buy = frame[[f"q{k}_buy_qty" for k in range(QUARTERS)]].to_numpy(dtype=np.float64)
    quarter_shares = (
        (1, "ms_q2_trade_share"),
        (2, "ms_q3_trade_share"),
        (3, "ms_q4_trade_share"),
    )
    for position, name in quarter_shares:
        columns[name] = quarter_trades[:, position] / np.maximum(n, 1.0)
    columns["ms_late_early_qty_imbalance"] = (
        quarter_qty[:, 2] + quarter_qty[:, 3] - quarter_qty[:, 0] - quarter_qty[:, 1]
    ) / np.maximum(qty, eps)
    last_qty = quarter_qty[:, 3]
    columns["ms_q4_flow_imbalance"] = np.where(
        last_qty > 0.0,
        (2.0 * quarter_buy[:, 3] - last_qty) / np.maximum(last_qty, eps),
        0.0,
    )

    # -- H. volume-at-price approximations ---------------------------------- #
    price_qty = frame[[f"price_qty_{j}" for j in range(PRICE_BINS)]].to_numpy(dtype=np.float64)
    shares = price_qty / np.maximum(price_qty.sum(axis=1, keepdims=True), eps)
    columns["ms_price_hhi"] = (shares**2).sum(axis=1)
    # `0 ln 0 := 0`, spelled with `where` rather than a nan-aware sum so no NaN
    # is ever created in the first place.
    logs = np.where(shares > 0.0, np.log(np.where(shares > 0.0, shares, 1.0)), 0.0)
    columns["ms_price_entropy"] = -(shares * logs).sum(axis=1) / np.log(PRICE_BINS)
    price_span = np.maximum(ph - pl, eps * vwap)
    columns["ms_close_from_vwap"] = np.clip((p1 - vwap) / price_span, -1.0, 1.0)
    node = pl + (np.argmax(price_qty, axis=1) + 0.5) * (ph - pl) / PRICE_BINS
    columns["ms_close_from_node"] = np.clip((p1 - node) / price_span, -1.0, 1.0)

    out = pd.DataFrame(
        {name: columns[name] for name in MICROSTRUCTURE_FEATURE_COLUMNS},
        index=frame.index,
    )
    if not np.isfinite(out.to_numpy(dtype=np.float64)).all():
        bad = np.argwhere(~np.isfinite(out.to_numpy(dtype=np.float64)))
        column = MICROSTRUCTURE_FEATURE_COLUMNS[int(bad[0][1])]
        raise MicrostructureError(
            f"{len(bad)} non-finite microstructure value(s), first in column {column!r} at "
            f"row {int(bad[0][0])}. docs/microstructure_v1.md §5 guarantees a finite value "
            "on every row; a NaN here would silently change which rows the information "
            "sets are compared on"
        )
    return out


def segment_ids(aggregates: pd.DataFrame) -> np.ndarray:
    """The segment each aggregate row belongs to, for callers checking gap rules."""
    return _segments(aggregates.reset_index(drop=True)["date"]).to_numpy(dtype=np.int64)
