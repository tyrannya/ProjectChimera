"""Executable pin for ``docs/chart_structure_v1.md``, written from the document alone.

Every expected number below was derived by hand from the formulas in the spec
and never from ``nn.chart_structure``: a test whose assertions were read back
out of the implementation agrees with that implementation by construction and
therefore cannot catch a bug in it. That independence is the only reason this
file has any value, so the fixtures are hand-built candle sequences small
enough to compute on paper rather than recorded engine output.

The one piece of the spec restated here is the §1.1 ATR, because a tolerance
such as ``TOUCH_ATR * atr_den[t]`` cannot be checked without it. Most fixtures
are built from candles that are exactly 1.0 tall and always contain the
previous close, which forces ``TR[t] == 1.0`` on every row and therefore
``ATR == 1.0`` exactly, so the surrounding arithmetic stays readable;
``test_fixture_atr_is_one_by_construction`` guards that property. The
compression fixtures cannot have a constant true range — a converging box is a
shrinking box — so those three restate ``atr_den`` instead.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from nn.chart_structure import (
    CHART_FEATURE_FAMILIES,
    CHART_FEATURE_NAMES,
    CHART_SPEC_VERSION,
    ChartSpec,
    chart_feature_columns,
    compute_chart_features,
    compute_chart_features_segmented,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_CANDLES = REPO_ROOT / "data" / "research" / "btc_usdt_1h_gen1_raw_pre_styx.parquet"
STYX = pd.Timestamp("2025-08-27T23:00:00+00:00")

START = pd.Timestamp("2024-01-01T00:00:00+00:00")
HOUR = pd.Timedelta(hours=1)

# §1.3, restated so the arithmetic below can be read without the document open.
WIN_SHORT = 20
WIN_LONG = 60
TOUCH_ATR = 0.25
PIVOT_RIGHT = 3

Candle = tuple[float, float, float, float]


# --------------------------------------------------------------------------- #
# fixture construction
# --------------------------------------------------------------------------- #
def _frame(
    rows: Sequence[Candle], dates: Sequence[pd.Timestamp] | None = None
) -> pd.DataFrame:
    """OHLCV frame with a real 1h UTC ``date`` column, one row per ``(o, h, l, c)``."""
    if dates is None:
        dates = [START + i * HOUR for i in range(len(rows))]
    frame = pd.DataFrame(
        {
            "date": pd.DatetimeIndex(dates, tz="UTC"),
            "open": [r[0] for r in rows],
            "high": [r[1] for r in rows],
            "low": [r[2] for r in rows],
            "close": [r[3] for r in rows],
            "volume": [10.0] * len(rows),
        }
    )
    top = frame[["open", "close"]].max(axis=1)
    bottom = frame[["open", "close"]].min(axis=1)
    assert (frame["high"] >= top).all() and (frame["low"] <= bottom).all()
    return frame


def _mid_candles(centres: Sequence[float]) -> list[Candle]:
    """Candles 1.0 tall centred on ``centres``, each opening at the previous close.

    With ``|centre[t] - centre[t-1]| <= 0.5`` the previous close always sits
    inside ``[l[t], h[t]]``, so ``TR[t] = h[t] - l[t] = 1.0`` on every row.
    """
    rows: list[Candle] = []
    previous = centres[0]
    for centre in centres:
        rows.append((previous, centre + 0.5, centre - 0.5, centre))
        previous = centre
    return rows


def _atr_den(df: pd.DataFrame) -> np.ndarray:
    """``atr_den`` of §1.1, restated so ATR-normalised values can be checked."""
    prev_close = df["close"].shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.ewm(alpha=1.0 / 14, adjust=False).mean()
    return np.maximum(atr.to_numpy(), 1e-12 * df["close"].to_numpy())


def _random_walk(rows: int, seed: int) -> pd.DataFrame:
    """A deterministic series rich enough to exercise every family."""
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.4, rows))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + np.abs(rng.normal(0.0, 0.3, rows))
    low = np.minimum(open_, close) - np.abs(rng.normal(0.0, 0.3, rows))
    return _frame(list(zip(open_, high, low, close)))


# --- the fixtures themselves ------------------------------------------------ #
# G1: a clean horizontal box. Centres cycle [100.0, 100.5, 101.0, 100.5], so
# highs cycle [100.5, 101.0, 101.5, 101.0] and lows [99.5, 100.0, 100.5, 100.0].
# 60 and 20 are both whole multiples of the period, so a full window holds each
# phase equally often and the box is exactly [99.5, 101.5].
_G1_CYCLE = [100.0, 100.5, 101.0, 100.5]
G1_RANGE = _frame(_mid_candles([_G1_CYCLE[i % 4] for i in range(72)]))

# G2: flat highs at 102.0, lows rising exactly 0.02 per bar. The low slope is
# therefore exactly 0.02 price/bar and the high slope exactly 0.
G2_ASCENDING = _frame(
    [
        (
            101.0 if i == 0 else 101.0 + 0.01 * (i - 1),
            102.0,
            100.0 + 0.02 * i,
            101.0 + 0.01 * i,
        )
        for i in range(80)
    ]
)

# G3: the mirror -- flat lows at 100.0, highs falling exactly 0.02 per bar.
G3_DESCENDING = _frame(
    [
        (
            101.0 if i == 0 else 101.0 - 0.01 * (i - 1),
            102.0 - 0.02 * i,
            100.0,
            101.0 - 0.01 * i,
        )
        for i in range(80)
    ]
)

# G4: both boundaries converge at 0.01 per bar onto a close pinned at 101.0.
# The flat close also exercises three of §5's declared degenerate values.
G4_SYMMETRIC = _frame([(101.0, 103.0 - 0.01 * i, 99.0 + 0.01 * i, 101.0) for i in range(80)])

# G5: a clean linear ramp, 0.1 per bar, in 1.0-tall candles. Every close lies
# exactly on the least-squares line, so the residuals vanish.
G5_RAMP = _frame(_mid_candles([100.0 + 0.1 * i for i in range(80)]))

# G6/G7: 2.0-tall and 0.2-tall candles, all closing at 100.0, so TR is exactly
# the candle height. 60 wide then 20 narrow contracts; the reverse expands.
_WIDE = (100.0, 101.0, 99.0, 100.0)
_NARROW = (100.0, 100.1, 99.9, 100.0)
G6_CONTRACTION = _frame([_WIDE] * 60 + [_NARROW] * 20)
G7_EXPANSION = _frame([_NARROW] * 60 + [_WIDE] * 20)

# G8: 30 candles alternating between centres 100.0 and 100.5 (so prior_high is
# 101.0 from row 2 onward and no close ever reaches it), then row 30 closes at
# 101.5, clear of it. Rows 31-32 stay above 101.0 so nothing is reclaimed.
_G8_BASE = _mid_candles([100.0 if i % 2 == 0 else 100.5 for i in range(30)])
G8_BREAK_UP = _frame(
    _G8_BASE
    + [
        (100.5, 101.5, 100.5, 101.5),
        (101.5, 102.0, 101.0, 101.5),
        (101.5, 102.0, 101.0, 101.5),
    ]
)

# G9: the mirror -- prior_low is 99.0, row 30 closes at 98.5.
_G9_BASE = _mid_candles([100.0 if i % 2 == 0 else 99.5 for i in range(30)])
G9_BREAK_DN = _frame(
    _G9_BASE
    + [
        (99.5, 99.5, 98.5, 98.5),
        (98.5, 99.0, 98.0, 98.5),
        (98.5, 99.0, 98.0, 98.5),
    ]
)

# G10: G8's break at row 30 (R = prior_high(30) = 101.0), then row 31 closes at
# 100.6, back inside. Row 32 stays inside, so the reclaim fires exactly once.
G10_FAILED_UP = _frame(
    _G8_BASE
    + [
        (100.5, 101.5, 100.5, 101.5),
        (101.5, 101.5, 100.5, 100.6),
        (100.6, 101.1, 100.1, 100.6),
    ]
)

# G11: the mirror -- R = prior_low(30) = 99.0, row 31 closes back above at 99.4.
G11_FAILED_DN = _frame(
    _G9_BASE
    + [
        (99.5, 99.5, 98.5, 98.5),
        (98.5, 99.5, 98.5, 99.4),
        (99.4, 99.9, 98.9, 99.4),
    ]
)

# G12: two comparable tops at indices 4 (h = 101.5) and 10 (h = 101.4) with a
# trough at index 7 (l = 99.0) between them. The second confirms at row 13.
# Price never closes below the trough, so a definition that waited for the
# neckline break would never fire anywhere in this fixture.
# fmt: off
_G12_CENTRES = [
    99.0, 99.5, 100.0, 100.5, 101.0, 100.5, 100.0, 99.5,
    100.0, 100.5, 100.9, 100.5, 100.0, 99.8, 99.9, 100.0,
]
# fmt: on
G12_DOUBLE_TOP = _frame(_mid_candles(_G12_CENTRES))

# G13: the mirror -- bottoms at 4 (l = 98.5) and 10 (l = 98.6), peak at 7
# (h = 101.0). The second bottom confirms at row 13.
# fmt: off
_G13_CENTRES = [
    101.0, 100.5, 100.0, 99.5, 99.0, 99.5, 100.0, 100.5,
    100.0, 99.5, 99.1, 99.5, 100.0, 100.2, 100.1, 100.0,
]
# fmt: on
G13_DOUBLE_BOTTOM = _frame(_mid_candles(_G13_CENTRES))

# G14: a 20-point pole over rows 0-40, then a 20-bar flag oscillating in a
# 1.2-wide box. Candle heights never change, which is why §3.5 needs both
# ratios: the box contracts to 0.059 while the true-range ratio stays at 1.0.
_G14_CENTRES = [100.0 + 0.5 * i for i in range(41)] + [
    120.0 if (i - 41) % 2 == 0 else 119.8 for i in range(41, 61)
]
G14_FLAG = _frame(_mid_candles(_G14_CENTRES))

# G15: G12's double top plus two candles that open a pending up-break at row 17
# (c = 102.0 > prior_high = 101.5), then five missing hours, then eight
# identical candles. If any state bridged the hole the post-gap rows would show
# it: the pending break would resolve on the first one (its close of 100.6 is
# below R = 101.5), chart_dt_valid and chart_db_valid would still read 1, the
# box would be [98.5, 102.0] rather than [100.3, 100.9], and chart_pole_atr
# would measure against a close of 99.0 instead of 100.6.
_G15_SEGMENT_ONE = _mid_candles(_G12_CENTRES) + [
    (100.0, 101.0, 100.0, 101.0),
    (101.0, 102.0, 101.0, 102.0),
]
_G15_SEGMENT_TWO: list[Candle] = [(100.6, 100.9, 100.3, 100.6)] * 8
_G15_DATES = [START + i * HOUR for i in range(len(_G15_SEGMENT_ONE))] + [
    START + (len(_G15_SEGMENT_ONE) + 5 + i) * HOUR for i in range(len(_G15_SEGMENT_TWO))
]
G15_GAP = _frame(_G15_SEGMENT_ONE + _G15_SEGMENT_TWO, dates=_G15_DATES)
G15_SEGMENT_ONE_ONLY = _frame(_G15_SEGMENT_ONE)

# G17: eighty identical candles. Every close is the same float, so in exact
# arithmetic the least-squares fit is the constant line, every residual is 0
# and sigma is 0 -- which is the condition §3.4 and §4 row 14 attach a declared
# value of 0.0 to. atr_den is exactly 0.6, the candle height.
G17_FLAT_CLOSE = _frame([(100.6, 100.9, 100.3, 100.6)] * 80)

# G16: three candles, small enough that the whole least-squares apparatus can
# be written out longhand. Closes 100.0, 100.5, 100.25.
G16_TINY = _frame(_mid_candles([100.0, 100.5, 100.25]))


@pytest.fixture(scope="module")
def real_candles() -> pd.DataFrame:
    """The committed pre-Styx BTC research candles."""
    frame = pd.read_parquet(REAL_CANDLES).reset_index(drop=True)
    assert pd.to_datetime(frame["date"], utc=True).max() < STYX
    return frame


@pytest.fixture(scope="module")
def real_contiguous(real_candles: pd.DataFrame) -> pd.DataFrame:
    """A gap-free block of the real candles, for the one-segment entry point."""
    block = real_candles.iloc[4289:8010].reset_index(drop=True)
    assert (pd.to_datetime(block["date"], utc=True).diff().iloc[1:] == HOUR).all()
    return block


# --------------------------------------------------------------------------- #
# fixture self-checks
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "frame",
    [
        G1_RANGE,
        G5_RAMP,
        G8_BREAK_UP,
        G9_BREAK_DN,
        G10_FAILED_UP,
        G11_FAILED_DN,
        G12_DOUBLE_TOP,
        G13_DOUBLE_BOTTOM,
        G14_FLAG,
        G16_TINY,
    ],
    ids=["g1", "g5", "g8", "g9", "g10", "g11", "g12", "g13", "g14", "g16"],
)
def test_fixture_atr_is_one_by_construction(frame: pd.DataFrame) -> None:
    assert _atr_den(frame) == pytest.approx(np.ones(len(frame)), abs=1e-12)


def test_contraction_fixture_true_range_is_the_candle_height() -> None:
    # TR[t] = max(h - l, |h - c[t-1]|, |l - c[t-1]|) with every close at 100.0.
    assert _atr_den(G6_CONTRACTION)[:60] == pytest.approx(np.full(60, 2.0), abs=1e-12)
    assert _atr_den(G7_EXPANSION)[:60] == pytest.approx(np.full(60, 0.2), abs=1e-12)


# --------------------------------------------------------------------------- #
# 1. the horizontal range box, §3.2 and §4 A
# --------------------------------------------------------------------------- #
def test_clean_horizontal_range_width_position_and_touch_counts() -> None:
    out = compute_chart_features(G1_RANGE)

    # Row 62: I_l = [3, 62] is 60 bars, exactly 15 of each of the four phases;
    # I_s = [43, 62] is 20 bars, exactly 5 of each. Box = [99.5, 101.5].
    assert out["chart_window_fill"].iloc[62] == pytest.approx(1.0)
    assert out["chart_range_width_atr"].iloc[62] == pytest.approx(2.0)
    # c[62] = centre[62 % 4 == 2] = 101.0 -> (101.0 - 99.5) / 2.0.
    assert out["chart_range_pos"].iloc[62] == pytest.approx(0.75)
    assert out["chart_range_pos_fast"].iloc[62] == pytest.approx(0.75)
    # Only h = 101.5 clears 101.5 - 0.25 * 1.0; only l = 99.5 clears 99.5 + 0.25.
    assert out["chart_touch_high_frac"].iloc[62] == pytest.approx(15.0 / 60.0)
    assert out["chart_touch_low_frac"].iloc[62] == pytest.approx(15.0 / 60.0)
    # I_s and I_l see the same extremes, so the box has not contracted at all.
    assert out["chart_range_ratio"].iloc[62] == pytest.approx(1.0)
    assert out["chart_tr_ratio"].iloc[62] == pytest.approx(1.0)

    # Row 9: the window truncates left to I_l = [0, 9], which holds three bars
    # of phase 0, three of phase 1, two of phase 2 and two of phase 3.
    assert out["chart_window_fill"].iloc[9] == pytest.approx(10.0 / 60.0)
    assert out["chart_range_width_atr"].iloc[9] == pytest.approx(2.0)
    assert out["chart_range_pos"].iloc[9] == pytest.approx(0.5)  # c[9] = 100.5
    assert out["chart_touch_high_frac"].iloc[9] == pytest.approx(2.0 / 10.0)
    assert out["chart_touch_low_frac"].iloc[9] == pytest.approx(3.0 / 10.0)

    # The box is inclusive of t, so the position never needs a clip (§3.2).
    assert (out["chart_range_pos"] >= 0.0).all() and (out["chart_range_pos"] <= 1.0).all()
    assert (out["chart_touch_high_frac"] > 0.0).all()
    assert (out["chart_touch_low_frac"] <= 1.0).all()


# --------------------------------------------------------------------------- #
# 2-4. compression, §3.3
# --------------------------------------------------------------------------- #
def test_ascending_compression_rising_lows_against_flat_highs() -> None:
    out = compute_chart_features(G2_ASCENDING)
    den = _atr_den(G2_ASCENDING)[79]

    # Highs are the constant 102.0, so slope(h, I_l) is exactly 0; lows are
    # exactly linear at 0.02 per bar, so slope(l, I_l) is exactly 0.02.
    assert out["chart_high_slope_atr"].iloc[79] == pytest.approx(0.0, abs=1e-9)
    assert out["chart_low_slope_atr"].iloc[79] == pytest.approx(0.02 / den)
    assert out["chart_low_slope_atr"].iloc[79] > 0.0
    # §3.3: conv = s_lo - s_hi, positive iff the boundaries converge.
    assert out["chart_convergence_atr"].iloc[79] == pytest.approx(0.02 / den)
    assert out["chart_convergence_atr"].iloc[79] > 0.0
    # §3.3's table: flat highs, rising lows reads asym = +1.
    assert out["chart_slope_asymmetry"].iloc[79] == pytest.approx(1.0)


def test_descending_compression_falling_highs_against_flat_lows() -> None:
    out = compute_chart_features(G3_DESCENDING)
    den = _atr_den(G3_DESCENDING)[79]

    assert out["chart_low_slope_atr"].iloc[79] == pytest.approx(0.0, abs=1e-9)
    assert out["chart_high_slope_atr"].iloc[79] == pytest.approx(-0.02 / den)
    assert out["chart_high_slope_atr"].iloc[79] < 0.0
    # conv = 0 - (-0.02 / den): falling highs converge onto flat lows.
    assert out["chart_convergence_atr"].iloc[79] == pytest.approx(0.02 / den)
    assert out["chart_convergence_atr"].iloc[79] > 0.0
    assert out["chart_slope_asymmetry"].iloc[79] == pytest.approx(-1.0)


def test_symmetric_compression_converges_from_both_sides() -> None:
    out = compute_chart_features(G4_SYMMETRIC)
    den = _atr_den(G4_SYMMETRIC)[79]

    assert out["chart_high_slope_atr"].iloc[79] == pytest.approx(-0.01 / den)
    assert out["chart_low_slope_atr"].iloc[79] == pytest.approx(0.01 / den)
    assert out["chart_convergence_atr"].iloc[79] == pytest.approx(0.02 / den)
    # Equal and opposite: the signed axis of §3.3 reads exactly 0.
    assert out["chart_slope_asymmetry"].iloc[79] == pytest.approx(0.0, abs=1e-9)

    # Every close is 101.0, so sum (c - cbar)^2 == 0 and sigma == 0: three of
    # §5's declared degenerate values, all on the same row.
    assert out["chart_trend_slope_atr"].iloc[79] == pytest.approx(0.0, abs=1e-12)
    assert out["chart_channel_disp_atr"].iloc[79] == pytest.approx(0.0, abs=1e-12)
    assert out["chart_trend_r2"].iloc[79] == 0.0
    assert out["chart_channel_pos"].iloc[79] == 0.0


# --------------------------------------------------------------------------- #
# 5. the trend channel, §3.4
# --------------------------------------------------------------------------- #
def test_trend_channel_on_a_clean_linear_ramp() -> None:
    out = compute_chart_features(G5_RAMP)

    # Closes rise exactly 0.1 per bar against ATR == 1.0, and every close lies
    # on the fitted line, so the residuals -- and sigma -- vanish and R2 = 1.
    assert out["chart_trend_slope_atr"].iloc[70] == pytest.approx(0.1)
    assert out["chart_channel_disp_atr"].iloc[70] == pytest.approx(0.0, abs=1e-9)
    assert out["chart_trend_r2"].iloc[70] == pytest.approx(1.0, abs=1e-9)
    assert out["chart_trend_r2"].iloc[70] <= 1.0
    # sigma is zero to float noise, so chan_pos is only pinned by its §3.4
    # bound; the exact-zero case is asserted on G4, where sigma is exactly 0.
    assert abs(out["chart_channel_pos"].iloc[70]) <= math.sqrt(WIN_LONG - 1)

    # Highs and lows are the closes shifted by +-0.5, so both boundary slopes
    # are 0.1: a parallel channel, not a triangle.
    assert out["chart_high_slope_atr"].iloc[70] == pytest.approx(0.1)
    assert out["chart_low_slope_atr"].iloc[70] == pytest.approx(0.1)
    assert out["chart_convergence_atr"].iloc[70] == pytest.approx(0.0, abs=1e-9)
    assert out["chart_slope_asymmetry"].iloc[70] == pytest.approx(1.0)

    # I_l = [11, 70]: H_l = c[70] + 0.5 = 107.5, L_l = c[11] - 0.5 = 100.6.
    assert out["chart_range_width_atr"].iloc[70] == pytest.approx(6.9)
    assert out["chart_range_pos"].iloc[70] == pytest.approx(6.4 / 6.9)
    # I_s = [51, 70]: H_s = 107.5, L_s = c[51] - 0.5 = 104.6.
    assert out["chart_range_ratio"].iloc[70] == pytest.approx(2.9 / 6.9)
    # Only c[u] >= 106.75 reaches within 0.25 ATR of the high, i.e. u >= 68.
    assert out["chart_touch_high_frac"].iloc[70] == pytest.approx(3.0 / 60.0)
    assert out["chart_touch_low_frac"].iloc[70] == pytest.approx(3.0 / 60.0)


def test_small_window_channel_arithmetic_longhand() -> None:
    out = compute_chart_features(G16_TINY)

    # Row 2: w = 3, x = [-2, -1, 0], c = [100.0, 100.5, 100.25], ATR = 1.0.
    # xbar = -1, cbar = 100.25, sum (x - xbar)(c - cbar) = 0.25, sum (x - xbar)^2 = 2
    # -> b = 0.125, a = cbar - b * xbar = 100.375, r = [-0.125, 0.25, -0.125].
    assert out["chart_trend_slope_atr"].iloc[2] == pytest.approx(0.125)
    # sigma = sqrt(mean r^2) = sqrt(0.09375 / 3) = sqrt(0.03125).
    assert out["chart_channel_disp_atr"].iloc[2] == pytest.approx(math.sqrt(0.03125))
    # R2 = 1 - 0.09375 / 0.125.
    assert out["chart_trend_r2"].iloc[2] == pytest.approx(0.25)
    # chan_pos = r[2] / sigma = -0.125 / sqrt(0.03125).
    assert out["chart_channel_pos"].iloc[2] == pytest.approx(-1.0 / math.sqrt(2.0))

    # Highs and lows are the closes shifted by +-0.5, so both slopes are 0.125.
    assert out["chart_high_slope_atr"].iloc[2] == pytest.approx(0.125)
    assert out["chart_low_slope_atr"].iloc[2] == pytest.approx(0.125)
    # Box [99.5, 101.0]; touch tolerance 0.25 admits h = 101.0 and h = 100.75,
    # and l = 99.5 and l = 99.75.
    assert out["chart_range_width_atr"].iloc[2] == pytest.approx(1.5)
    assert out["chart_range_pos"].iloc[2] == pytest.approx(0.5)
    assert out["chart_touch_high_frac"].iloc[2] == pytest.approx(2.0 / 3.0)
    assert out["chart_touch_low_frac"].iloc[2] == pytest.approx(2.0 / 3.0)


def test_flat_close_window_has_a_zero_standardised_residual() -> None:
    """§3.4's degenerate branch, on the window shape that actually triggers it.

    A window whose closes are all identical fits the constant line exactly, so
    ``r[u] == 0`` for every ``u``, ``sigma(t) == 0``, and §4 row 14 declares
    ``chart_channel_pos`` to be ``0.0`` -- the close sits *on* the fitted line,
    which is the least extreme position a residual can take.
    """
    out = compute_chart_features(G17_FLAT_CLOSE)

    # The non-channel half of the row, all of it forced by identical candles.
    for row in (0, 1, 19, 30, 59, 60, 79):
        assert out["chart_range_width_atr"].iloc[row] == pytest.approx(1.0)
        assert out["chart_range_pos"].iloc[row] == pytest.approx(0.5)
        assert out["chart_range_ratio"].iloc[row] == pytest.approx(1.0)
        assert out["chart_tr_ratio"].iloc[row] == pytest.approx(1.0)
        assert out["chart_slope_asymmetry"].iloc[row] == 0.0
        # sum (c - cbar)^2 == 0, so §5 declares the fit quality 0.0.
        assert out["chart_trend_r2"].iloc[row] == 0.0
        assert out["chart_trend_slope_atr"].iloc[row] == pytest.approx(0.0, abs=1e-9)

    # Asserted as a pair so a failure reports both halves of the branch: the
    # dispersion that should be zero and the position that is declared zero.
    for row in (0, 1, 19, 30, 59, 60, 79):
        assert (
            out["chart_channel_pos"].iloc[row],
            out["chart_channel_disp_atr"].iloc[row] < 1e-9,
        ) == (0.0, True), f"row {row}"

    assert (out["chart_channel_pos"] == 0.0).all()


# --------------------------------------------------------------------------- #
# 6. volatility contraction and expansion, §3.5
# --------------------------------------------------------------------------- #
def test_volatility_contraction_moves_both_ratios_below_one() -> None:
    out = compute_chart_features(G6_CONTRACTION)

    # Row 79: I_s = [60, 79] is 20 narrow bars (box 0.2 wide, TR 0.2 each);
    # I_l = [20, 79] is 40 wide bars and 20 narrow ones (box 2.0 wide).
    assert out["chart_range_ratio"].iloc[79] == pytest.approx(0.2 / 2.0)
    assert out["chart_range_ratio"].iloc[79] < 1.0
    # mean TR over I_l = (40 * 2.0 + 20 * 0.2) / 60 = 1.4.
    assert out["chart_tr_ratio"].iloc[79] == pytest.approx(0.2 / 1.4)
    assert out["chart_tr_ratio"].iloc[79] < 1.0
    # Both are exactly 1.0 while w_s == w_l, which §3.5 calls the correct
    # reading: no contraction is measurable yet.
    assert out["chart_range_ratio"].iloc[0] == pytest.approx(1.0)
    assert out["chart_tr_ratio"].iloc[0] == pytest.approx(1.0)
    assert out["chart_range_ratio"].iloc[:20].to_numpy() == pytest.approx(np.ones(20))


def test_volatility_expansion_moves_the_true_range_ratio_above_one() -> None:
    out = compute_chart_features(G7_EXPANSION)

    # Row 79: I_s = [60, 79] is 20 wide bars, I_l = [20, 79] is 40 narrow and
    # 20 wide -- the extremes of the two windows coincide, the bar sizes do not.
    assert out["chart_range_ratio"].iloc[79] == pytest.approx(1.0)
    # mean TR over I_l = (40 * 0.2 + 20 * 2.0) / 60 = 0.8.
    assert out["chart_tr_ratio"].iloc[79] == pytest.approx(2.0 / 0.8)
    assert out["chart_tr_ratio"].iloc[79] > 1.0
    # §3.5 bounds tr_ratio by WIN_LONG / WIN_SHORT = 3.
    assert (out["chart_tr_ratio"] <= WIN_LONG / WIN_SHORT).all()
    assert (out["chart_range_ratio"] <= 1.0).all()


# --------------------------------------------------------------------------- #
# 7-8. breakout, §3.6
# --------------------------------------------------------------------------- #
def test_upside_breakout_is_measured_against_the_prior_window() -> None:
    out = compute_chart_features(G8_BREAK_UP)

    # prior_high(30) = max h over [0, 29] = 101.0; c[30] = 101.5, ATR = 1.0.
    assert out["chart_breakout_up_atr"].iloc[30] == pytest.approx(0.5)
    # Nothing before it closes above its own prior high.
    assert (out["chart_breakout_up_atr"].iloc[:30] <= 0.0).all()
    assert out["chart_breakout_up_atr"].iloc[0] == 0.0  # §5: empty prior window
    # prior_low(30) = 99.5, so the down side reads -2.0 on the same row.
    assert out["chart_breakout_dn_atr"].iloc[30] == pytest.approx(-2.0)
    # The break is never reclaimed: rows 31-32 close at 101.5, above R = 101.0.
    assert (out["chart_failed_breakout_up_atr"] == 0.0).all()
    assert (out["chart_failed_breakout_dn_atr"] == 0.0).all()


def test_downside_breakout_is_measured_against_the_prior_window() -> None:
    out = compute_chart_features(G9_BREAK_DN)

    # prior_low(30) = min l over [0, 29] = 99.0; c[30] = 98.5, ATR = 1.0.
    assert out["chart_breakout_dn_atr"].iloc[30] == pytest.approx(0.5)
    assert (out["chart_breakout_dn_atr"].iloc[:30] <= 0.0).all()
    assert out["chart_breakout_dn_atr"].iloc[0] == 0.0
    # prior_high(30) = 100.5.
    assert out["chart_breakout_up_atr"].iloc[30] == pytest.approx(-2.0)
    assert (out["chart_failed_breakout_dn_atr"] == 0.0).all()
    assert (out["chart_failed_breakout_up_atr"] == 0.0).all()


# --------------------------------------------------------------------------- #
# 9-10. failed breakout / reclaim, §3.6 and §2(c)
# --------------------------------------------------------------------------- #
def test_failed_upside_breakout_is_stamped_on_the_reclaim_candle() -> None:
    out = compute_chart_features(G10_FAILED_UP)

    # Row 30 opens the pending break, recording R = prior_high(30) = 101.0.
    assert out["chart_breakout_up_atr"].iloc[30] == pytest.approx(0.5)
    assert out["chart_failed_breakout_up_atr"].iloc[30] == 0.0

    # Row 31 closes at 100.6, back inside: (101.0 - 100.6) / 1.0.
    assert out["chart_failed_breakout_up_atr"].iloc[31] == pytest.approx(0.4)
    assert out["chart_failed_breakout_up_atr"].sum() == pytest.approx(0.4)
    assert out["chart_failed_breakout_up_atr"].iloc[32] == 0.0

    # §2(c): the reclaim candle is not itself a breakout. prior_high(31) is
    # max h over [0, 30] = 101.5, so the plain column reads -0.9 there.
    assert out["chart_breakout_up_atr"].iloc[31] == pytest.approx(-0.9)
    assert out["chart_breakout_up_atr"].iloc[31] < 0.0
    # Step 4 precedes step 5, but nothing opens here: c[31] = 100.6 is inside
    # both prior extremes -- prior_low(31) = 99.5 -- so no down-break fires.
    assert out["chart_breakout_dn_atr"].iloc[31] == pytest.approx(-1.1)
    assert (out["chart_failed_breakout_dn_atr"] == 0.0).all()


def test_failed_downside_breakout_is_stamped_on_the_reclaim_candle() -> None:
    out = compute_chart_features(G11_FAILED_DN)

    # Row 30 opens the pending break with R = prior_low(30) = 99.0.
    assert out["chart_breakout_dn_atr"].iloc[30] == pytest.approx(0.5)
    assert out["chart_failed_breakout_dn_atr"].iloc[30] == 0.0

    # Row 31 closes at 99.4, back above R: (99.4 - 99.0) / 1.0.
    assert out["chart_failed_breakout_dn_atr"].iloc[31] == pytest.approx(0.4)
    assert out["chart_failed_breakout_dn_atr"].sum() == pytest.approx(0.4)
    assert out["chart_failed_breakout_dn_atr"].iloc[32] == 0.0

    # prior_low(31) = min l over [0, 30] = 98.5 and prior_high(31) = 100.5.
    assert out["chart_breakout_dn_atr"].iloc[31] == pytest.approx(-0.9)
    assert out["chart_breakout_dn_atr"].iloc[31] < 0.0
    assert out["chart_breakout_up_atr"].iloc[31] == pytest.approx(-1.1)
    assert (out["chart_failed_breakout_up_atr"] == 0.0).all()


# --------------------------------------------------------------------------- #
# 11-12. double-top / double-bottom geometry, §3.7
# --------------------------------------------------------------------------- #
def test_double_top_geometry_fires_on_the_second_pivots_confirmation() -> None:
    out = compute_chart_features(G12_DOUBLE_TOP)

    # Swing highs at i1 = 4 (h = 101.5) and i2 = 10 (h = 101.4). §2(b) makes
    # the second knowable only at 10 + PIVOT_RIGHT = 13, and until then only
    # one confirmed swing high exists, so the flag and rows 22-24 stay at 0.
    assert (out["chart_dt_valid"].iloc[:13] == 0.0).all()
    for row in (10, 11, 12):
        assert out["chart_dt_offset_atr"].iloc[row] == 0.0
        assert out["chart_dt_trough_atr"].iloc[row] == 0.0
        assert out["chart_dt_neckline_atr"].iloc[row] == 0.0

    assert out["chart_dt_valid"].iloc[13] == 1.0
    # (h[10] - h[4]) / 1.0 = 101.4 - 101.5.
    assert out["chart_dt_offset_atr"].iloc[13] == pytest.approx(-0.1)
    # m = min l over [4, 10] = l[7] = 99.0; min(h[4], h[10]) = 101.4.
    assert out["chart_dt_trough_atr"].iloc[13] == pytest.approx(2.4)
    # (c[13] - m) / 1.0 = 99.8 - 99.0.
    assert out["chart_dt_neckline_atr"].iloc[13] == pytest.approx(0.8)
    assert out["chart_dt_neckline_atr"].iloc[14] == pytest.approx(0.9)
    assert out["chart_dt_neckline_atr"].iloc[15] == pytest.approx(1.0)

    # §2(c) and §9(a): the structure is reported without ever waiting for the
    # neckline break. Price closes above m on every single row of the fixture,
    # so a definition that required the break would report nothing at all.
    assert (out["chart_dt_neckline_atr"].iloc[13:] > 0.0).all()
    assert (out["chart_dt_valid"].iloc[13:] == 1.0).all()

    # Only one swing low confirms here (i = 7), so the mirror side stays off --
    # §4's "else 0 is per quantity, not gated on the flag".
    assert (out["chart_db_valid"] == 0.0).all()
    assert (out["chart_db_offset_atr"] == 0.0).all()
    assert (out["chart_db_peak_atr"] == 0.0).all()
    assert (out["chart_db_neckline_atr"] == 0.0).all()


def test_double_bottom_geometry_fires_on_the_second_pivots_confirmation() -> None:
    out = compute_chart_features(G13_DOUBLE_BOTTOM)

    # Swing lows at j1 = 4 (l = 98.5) and j2 = 10 (l = 98.6), confirmed at 13.
    assert (out["chart_db_valid"].iloc[:13] == 0.0).all()
    for row in (10, 11, 12):
        assert out["chart_db_offset_atr"].iloc[row] == 0.0
        assert out["chart_db_peak_atr"].iloc[row] == 0.0
        assert out["chart_db_neckline_atr"].iloc[row] == 0.0

    assert out["chart_db_valid"].iloc[13] == 1.0
    # (l[10] - l[4]) / 1.0 = 98.6 - 98.5.
    assert out["chart_db_offset_atr"].iloc[13] == pytest.approx(0.1)
    # M = max h over [4, 10] = h[7] = 101.0; max(l[4], l[10]) = 98.6.
    assert out["chart_db_peak_atr"].iloc[13] == pytest.approx(2.4)
    # (M - c[13]) / 1.0 = 101.0 - 100.2.
    assert out["chart_db_neckline_atr"].iloc[13] == pytest.approx(0.8)
    assert out["chart_db_neckline_atr"].iloc[14] == pytest.approx(0.9)
    assert out["chart_db_neckline_atr"].iloc[15] == pytest.approx(1.0)

    assert (out["chart_db_neckline_atr"].iloc[13:] > 0.0).all()
    assert (out["chart_db_valid"].iloc[13:] == 1.0).all()

    assert (out["chart_dt_valid"] == 0.0).all()
    assert (out["chart_dt_offset_atr"] == 0.0).all()
    assert (out["chart_dt_trough_atr"] == 0.0).all()
    assert (out["chart_dt_neckline_atr"] == 0.0).all()


# --------------------------------------------------------------------------- #
# 13. impulse, pole and the flag, §3.8
# --------------------------------------------------------------------------- #
def test_pole_then_compression_reads_as_three_measurements() -> None:
    out = compute_chart_features(G14_FLAG)

    # Row 60: c_at(60) = c[0] = 100.0, c_at(20) = c[40] = 120.0, c[60] = 119.8.
    assert out["chart_pole_atr"].iloc[60] == pytest.approx(20.0)
    assert out["chart_impulse_atr"].iloc[60] == pytest.approx(-0.2)

    # The flag itself: I_s = [41, 60] spans [119.3, 120.5] against an I_l =
    # [1, 60] box of [100.0, 120.5].
    assert out["chart_range_width_atr"].iloc[60] == pytest.approx(20.5)
    assert out["chart_range_ratio"].iloc[60] == pytest.approx(1.2 / 20.5)
    # Every candle is the same height, so §3.5's other ratio sees nothing --
    # which is exactly the non-redundancy the section claims.
    assert out["chart_tr_ratio"].iloc[60] == pytest.approx(1.0)

    # §5: below WIN_SHORT and WIN_LONG the lookbacks clamp to the segment's
    # first close, so the pole is 0 and the impulse carries the whole move.
    assert out["chart_pole_atr"].iloc[10] == 0.0
    assert out["chart_impulse_atr"].iloc[10] == pytest.approx(5.0)
    # Row 30: c_at(20) = c[10] = 105.0, c_at(60) = c[0] = 100.0.
    assert out["chart_pole_atr"].iloc[30] == pytest.approx(5.0)
    assert out["chart_impulse_atr"].iloc[30] == pytest.approx(10.0)


# --------------------------------------------------------------------------- #
# 14. gap reset, §1 and §7
# --------------------------------------------------------------------------- #
def test_state_does_not_bridge_a_missing_candle_gap() -> None:
    out = compute_chart_features_segmented(G15_GAP, timeframe="1h")
    assert len(out) == len(G15_GAP)

    n_one = len(_G15_SEGMENT_ONE)
    # Pre-gap the segment carries live state on both sides: two confirmed swing
    # highs, two confirmed swing lows, and a pending up-break opened at row 17
    # against R = prior_high(17) = 101.5.
    assert out["chart_dt_valid"].iloc[17] == 1.0
    assert out["chart_db_valid"].iloc[17] == 1.0
    assert out["chart_breakout_up_atr"].iloc[17] == pytest.approx(0.5)
    assert out["chart_window_fill"].iloc[17] == pytest.approx(18.0 / 60.0)

    # Post-gap every candle is (100.6, 100.9, 100.3, 100.6), so TR -- and
    # therefore atr_den -- is exactly 0.6 from the segment's own row 0.
    after = out.iloc[n_one:].reset_index(drop=True)
    assert len(after) == len(_G15_SEGMENT_TWO)
    assert _atr_den(_frame(_G15_SEGMENT_TWO)) == pytest.approx(
        np.full(len(_G15_SEGMENT_TWO), 0.6), abs=1e-12
    )

    for k in range(len(_G15_SEGMENT_TWO)):
        row = after.iloc[k]
        # A: the window restarts, so the fill counts this segment's rows only.
        assert row["chart_window_fill"] == pytest.approx((k + 1) / 60.0)
        # Box = [100.3, 100.9], width 0.6, i.e. exactly 1.0 ATR.
        assert row["chart_range_width_atr"] == pytest.approx(1.0)
        assert row["chart_range_pos"] == pytest.approx(0.5)
        assert row["chart_range_pos_fast"] == pytest.approx(0.5)
        # Every bar is within 0.25 * 0.6 = 0.15 of both edges.
        assert row["chart_touch_high_frac"] == pytest.approx(1.0)
        assert row["chart_touch_low_frac"] == pytest.approx(1.0)
        # B and C: identical candles, so every slope and residual is zero.
        assert row["chart_high_slope_atr"] == 0.0
        assert row["chart_low_slope_atr"] == 0.0
        assert row["chart_convergence_atr"] == 0.0
        assert row["chart_slope_asymmetry"] == 0.0
        assert row["chart_trend_slope_atr"] == 0.0
        assert row["chart_trend_r2"] == 0.0
        # sigma is a sqrt of a mean of squares, so "zero" is asserted to a
        # tolerance here exactly as it is on G4 and G5; in exact arithmetic
        # every residual is 0. chart_channel_pos has no such latitude: §4 row
        # 14 declares it 0.0 outright whenever sigma is 0. See
        # test_flat_close_window_has_a_zero_standardised_residual.
        assert row["chart_channel_disp_atr"] == pytest.approx(0.0, abs=1e-9)
        assert row["chart_channel_pos"] == 0.0
        # D: I_s and I_l coincide inside this short segment.
        assert row["chart_range_ratio"] == pytest.approx(1.0)
        assert row["chart_tr_ratio"] == pytest.approx(1.0)
        # E: the prior window is empty on row 0 and holds only this segment's
        # own identical candles afterwards: (100.6 - 100.9) / 0.6 = -0.5.
        expected_break = 0.0 if k == 0 else -0.5
        assert row["chart_breakout_up_atr"] == pytest.approx(expected_break)
        assert row["chart_breakout_dn_atr"] == pytest.approx(expected_break)
        # The pre-gap pending break must not resolve here; if it had bridged,
        # row 0 would carry (101.5 - 100.6) / 0.6 = 1.5.
        assert row["chart_failed_breakout_up_atr"] == 0.0
        assert row["chart_failed_breakout_dn_atr"] == 0.0
        # F: no pivot can be confirmed inside eight identical candles.
        assert row["chart_dt_valid"] == 0.0
        assert row["chart_dt_offset_atr"] == 0.0
        assert row["chart_dt_trough_atr"] == 0.0
        assert row["chart_dt_neckline_atr"] == 0.0
        assert row["chart_db_valid"] == 0.0
        assert row["chart_db_offset_atr"] == 0.0
        assert row["chart_db_peak_atr"] == 0.0
        assert row["chart_db_neckline_atr"] == 0.0
        # §3.8's lookbacks clamp to this segment's first close, not to 99.0.
        assert row["chart_pole_atr"] == 0.0
        assert row["chart_impulse_atr"] == 0.0

    # And the pre-gap rows are exactly what the first segment produces alone.
    assert_frame_equal(out.iloc[:n_one], compute_chart_features(G15_SEGMENT_ONE_ONLY))


# --------------------------------------------------------------------------- #
# 15. append invariance, §2
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n,k", [(50, 1), (80, 3), (120, 10), (200, 50), (310, 90)])
def test_append_invariance_synthetic(n: int, k: int) -> None:
    raw = _random_walk(rows=420, seed=7)
    long = compute_chart_features(raw.iloc[: n + k])
    short = compute_chart_features(raw.iloc[:n])
    assert_frame_equal(long.iloc[:n], short)


@pytest.mark.parametrize("n,k", [(60, 5), (150, 20), (240, 100)])
def test_append_invariance_synthetic_segmented(n: int, k: int) -> None:
    raw = _random_walk(rows=420, seed=11)
    # Punch two market-data holes so the appended slice must re-segment too.
    gapped = raw.drop(index=[137, 138, 139, 264]).reset_index(drop=True)
    long = compute_chart_features_segmented(gapped.iloc[: n + k], timeframe="1h")
    short = compute_chart_features_segmented(gapped.iloc[:n], timeframe="1h")
    assert_frame_equal(long.iloc[:n], short)


@pytest.mark.parametrize("n,k", [(1000, 1), (1800, 7), (2500, 600)])
def test_append_invariance_real(real_contiguous: pd.DataFrame, n: int, k: int) -> None:
    long = compute_chart_features(real_contiguous.iloc[: n + k])
    short = compute_chart_features(real_contiguous.iloc[:n])
    assert_frame_equal(long.iloc[:n], short)


@pytest.mark.parametrize("n,k", [(1200, 1), (2000, 300), (3000, 50)])
def test_append_invariance_real_segmented(real_candles: pd.DataFrame, n: int, k: int) -> None:
    long = compute_chart_features_segmented(real_candles.iloc[: n + k], timeframe="1h")
    short = compute_chart_features_segmented(real_candles.iloc[:n], timeframe="1h")
    assert_frame_equal(long.iloc[:n], short)


# --------------------------------------------------------------------------- #
# 16. the no-NaN guarantee on real candles, §5
# --------------------------------------------------------------------------- #
def test_no_nan_no_inf_and_exact_column_order_on_real_candles(
    real_candles: pd.DataFrame,
) -> None:
    out = compute_chart_features_segmented(real_candles, timeframe="1h")

    assert list(out.columns) == list(CHART_FEATURE_NAMES)
    assert len(out) == len(real_candles)
    assert (out.dtypes == np.float64).all()
    assert np.isfinite(out.to_numpy()).all()
    assert not out.isna().to_numpy().any()

    # The bounds §5 claims are bounds by construction, not clips.
    assert (out["chart_window_fill"] > 0.0).all()
    assert (out["chart_window_fill"] <= 1.0).all()
    assert (out["chart_range_pos"].between(0.0, 1.0)).all()
    assert (out["chart_range_pos_fast"].between(0.0, 1.0)).all()
    assert (out["chart_touch_high_frac"].between(0.0, 1.0, inclusive="right")).all()
    assert (out["chart_touch_low_frac"].between(0.0, 1.0, inclusive="right")).all()
    assert (out["chart_slope_asymmetry"].between(-1.0, 1.0)).all()
    assert (out["chart_trend_r2"].between(0.0, 1.0)).all()
    assert (out["chart_range_ratio"].between(0.0, 1.0)).all()
    assert (out["chart_tr_ratio"].between(0.0, WIN_LONG / WIN_SHORT)).all()
    assert (out["chart_channel_pos"].abs() <= math.sqrt(WIN_LONG - 1)).all()
    assert (out["chart_dt_valid"].isin([0.0, 1.0])).all()
    assert (out["chart_db_valid"].isin([0.0, 1.0])).all()
    assert (out["chart_dt_trough_atr"] >= 0.0).all()
    assert (out["chart_db_peak_atr"] >= 0.0).all()
    assert (out["chart_failed_breakout_up_atr"] >= 0.0).all()
    assert (out["chart_failed_breakout_dn_atr"] >= 0.0).all()


def test_no_nan_no_inf_on_a_single_real_segment(real_contiguous: pd.DataFrame) -> None:
    out = compute_chart_features(real_contiguous)

    assert list(out.columns) == list(CHART_FEATURE_NAMES)
    assert out.index.equals(real_contiguous.index)
    assert np.isfinite(out.to_numpy()).all()


# --------------------------------------------------------------------------- #
# 17. the public contract, §4 and §7
# --------------------------------------------------------------------------- #
def test_feature_names_are_the_thirty_columns_of_the_spec() -> None:
    assert CHART_SPEC_VERSION == "chart_structure_v1"
    assert CHART_FEATURE_NAMES == (
        "chart_window_fill",
        "chart_range_width_atr",
        "chart_range_pos",
        "chart_range_pos_fast",
        "chart_touch_high_frac",
        "chart_touch_low_frac",
        "chart_high_slope_atr",
        "chart_low_slope_atr",
        "chart_convergence_atr",
        "chart_slope_asymmetry",
        "chart_trend_slope_atr",
        "chart_trend_r2",
        "chart_channel_disp_atr",
        "chart_channel_pos",
        "chart_range_ratio",
        "chart_tr_ratio",
        "chart_breakout_up_atr",
        "chart_breakout_dn_atr",
        "chart_failed_breakout_up_atr",
        "chart_failed_breakout_dn_atr",
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
    )
    assert chart_feature_columns() == list(CHART_FEATURE_NAMES)


def test_feature_families_partition_the_feature_names() -> None:
    families = CHART_FEATURE_FAMILIES
    assert len(families) == 6
    assert set(families) == {
        "range",
        "compression",
        "channel",
        "volatility",
        "breakout",
        "patterns",
    }

    union: set[str] = set()
    total = 0
    for name, columns in families.items():
        assert len(set(columns)) == len(columns), name
        assert union.isdisjoint(columns), name
        union.update(columns)
        total += len(columns)

    assert total == 30
    assert len(CHART_FEATURE_NAMES) == 30
    assert union == set(CHART_FEATURE_NAMES)
    # §4's six families are 6, 4, 4, 2, 4 and 10 columns wide.
    assert {name: len(cols) for name, cols in families.items()} == {
        "range": 6,
        "compression": 4,
        "channel": 4,
        "volatility": 2,
        "breakout": 4,
        "patterns": 10,
    }
    assert tuple(families["volatility"]) == ("chart_range_ratio", "chart_tr_ratio")
    assert tuple(families["breakout"]) == (
        "chart_breakout_up_atr",
        "chart_breakout_dn_atr",
        "chart_failed_breakout_up_atr",
        "chart_failed_breakout_dn_atr",
    )


# --------------------------------------------------------------------------- #
# 18. the spec hash, §7
# --------------------------------------------------------------------------- #
def test_spec_hash_is_stable_and_changes_with_every_constant() -> None:
    spec = ChartSpec()
    assert spec.spec_hash() == spec.spec_hash() == ChartSpec().spec_hash()
    assert len(spec.spec_hash()) == 64
    assert set(spec.spec_hash()) <= set("0123456789abcdef")

    # §7 pins int-valued constants as ints: casting them to float to satisfy a
    # dict[str, float] annotation would change the canonical JSON and the hash.
    assert spec.to_dict() == {
        "atr_period": 14,
        "win_short": 20,
        "win_long": 60,
        "touch_atr": 0.25,
        "pivot_left": 3,
        "pivot_right": 3,
    }
    for field in ("atr_period", "win_short", "win_long", "pivot_left", "pivot_right"):
        assert isinstance(spec.to_dict()[field], int), field
        assert not isinstance(spec.to_dict()[field], bool), field

    # §7's reference value for the default spec.
    assert (
        spec.spec_hash() == "0f62f35d87cd92abe439959976de288d72a8adbf98eab804b05e594c131e946b"
    )

    changed = {
        "atr_period": 20,
        "win_short": 25,
        "win_long": 90,
        "touch_atr": 0.30,
        "pivot_left": 4,
        "pivot_right": 5,
    }
    for field, value in changed.items():
        other = dataclasses.replace(spec, **{field: value})
        assert other.spec_hash() != spec.spec_hash(), field


# --------------------------------------------------------------------------- #
# 19. warm-up, §5 at its hardest point
# --------------------------------------------------------------------------- #
def test_warmup_rows_are_finite_and_carry_the_declared_degenerate_values() -> None:
    out = compute_chart_features(G1_RANGE)

    # Row 0: w = 1. §5 declares 0.0 for rows 7-14 and 17-18; and because
    # ATR[0] = TR[0] = h[0] - l[0], the box is exactly 1.0 ATR wide with both
    # ratios at 1.0 and both touch fractions at 1.0 (the extreme bar counts).
    assert np.isfinite(out.iloc[0].to_numpy()).all()
    assert out["chart_window_fill"].iloc[0] == pytest.approx(1.0 / 60.0)
    assert out["chart_range_width_atr"].iloc[0] == pytest.approx(1.0)
    assert out["chart_range_pos"].iloc[0] == pytest.approx(0.5)
    assert out["chart_range_pos_fast"].iloc[0] == pytest.approx(0.5)
    assert out["chart_touch_high_frac"].iloc[0] == pytest.approx(1.0)
    assert out["chart_touch_low_frac"].iloc[0] == pytest.approx(1.0)
    for column in CHART_FEATURE_NAMES[6:14]:
        assert out[column].iloc[0] == 0.0, column
    assert out["chart_range_ratio"].iloc[0] == pytest.approx(1.0)
    assert out["chart_tr_ratio"].iloc[0] == pytest.approx(1.0)
    assert out["chart_breakout_up_atr"].iloc[0] == 0.0
    assert out["chart_breakout_dn_atr"].iloc[0] == 0.0
    assert out["chart_pole_atr"].iloc[0] == 0.0
    assert out["chart_impulse_atr"].iloc[0] == 0.0

    # Row 1: w = 2. Box [99.5, 101.0]; slope over x = [-1, 0] is 0.5 on both
    # boundaries and on the closes, and a two-point fit has zero residuals.
    assert out["chart_window_fill"].iloc[1] == pytest.approx(2.0 / 60.0)
    assert out["chart_range_width_atr"].iloc[1] == pytest.approx(1.5)
    assert out["chart_range_pos"].iloc[1] == pytest.approx(1.0 / 1.5)
    assert out["chart_touch_high_frac"].iloc[1] == pytest.approx(0.5)
    assert out["chart_touch_low_frac"].iloc[1] == pytest.approx(0.5)
    assert out["chart_high_slope_atr"].iloc[1] == pytest.approx(0.5)
    assert out["chart_low_slope_atr"].iloc[1] == pytest.approx(0.5)
    assert out["chart_convergence_atr"].iloc[1] == pytest.approx(0.0, abs=1e-12)
    assert out["chart_slope_asymmetry"].iloc[1] == pytest.approx(1.0)
    assert out["chart_trend_slope_atr"].iloc[1] == pytest.approx(0.5)
    assert out["chart_trend_r2"].iloc[1] == pytest.approx(1.0)
    assert out["chart_channel_disp_atr"].iloc[1] == pytest.approx(0.0, abs=1e-12)
    assert out["chart_channel_pos"].iloc[1] == 0.0
    assert out["chart_range_ratio"].iloc[1] == pytest.approx(1.0)
    assert out["chart_tr_ratio"].iloc[1] == pytest.approx(1.0)
    # prior_high(1) = h[0] = 100.5 = c[1]; prior_low(1) = l[0] = 99.5.
    assert out["chart_breakout_up_atr"].iloc[1] == pytest.approx(0.0, abs=1e-12)
    assert out["chart_breakout_dn_atr"].iloc[1] == pytest.approx(-1.0)
    assert out["chart_pole_atr"].iloc[1] == 0.0
    assert out["chart_impulse_atr"].iloc[1] == pytest.approx(0.5)

    # Row 2: box [99.5, 101.5], one bar within 0.25 of each edge.
    assert out["chart_range_width_atr"].iloc[2] == pytest.approx(2.0)
    assert out["chart_range_pos"].iloc[2] == pytest.approx(0.75)
    assert out["chart_touch_high_frac"].iloc[2] == pytest.approx(1.0 / 3.0)
    assert out["chart_touch_low_frac"].iloc[2] == pytest.approx(1.0 / 3.0)
    assert out["chart_impulse_atr"].iloc[2] == pytest.approx(1.0)

    # The whole truncated stretch: every column finite, the fill exact.
    warmup = out.iloc[:WIN_LONG]
    assert np.isfinite(warmup.to_numpy()).all()
    assert warmup["chart_window_fill"].to_numpy() == pytest.approx(
        np.arange(1, WIN_LONG + 1) / WIN_LONG
    )


def test_warmup_availability_flags_are_zero_until_two_pivots_can_confirm(
    real_contiguous: pd.DataFrame,
) -> None:
    out = compute_chart_features(real_contiguous.iloc[:40])

    # §3.7 forces i2 >= i1 + 4 and i1 >= PIVOT_LEFT = 3, and §2(b) delays each
    # by PIVOT_RIGHT, so no segment can hold two confirmed swing points before
    # row 3 + 4 + PIVOT_RIGHT = 10, whatever the candles look like.
    head = out.iloc[: 3 + 4 + PIVOT_RIGHT]
    assert (head["chart_dt_valid"] == 0.0).all()
    assert (head["chart_db_valid"] == 0.0).all()
    for column in CHART_FEATURE_NAMES[20:28]:
        assert (head[column] == 0.0).all(), column

    # And nothing anywhere in the truncated-window stretch is NaN or infinite.
    assert np.isfinite(out.to_numpy()).all()
    assert out["chart_window_fill"].to_numpy() == pytest.approx(np.arange(1, 41) / 60.0)
    assert out["chart_range_width_atr"].iloc[0] == pytest.approx(1.0)
    assert out["chart_range_ratio"].iloc[0] == pytest.approx(1.0)
    assert out["chart_tr_ratio"].iloc[0] == pytest.approx(1.0)
    assert out["chart_touch_high_frac"].iloc[0] == pytest.approx(1.0)
    assert out["chart_touch_low_frac"].iloc[0] == pytest.approx(1.0)
    for column in CHART_FEATURE_NAMES[6:14]:
        assert out[column].iloc[0] == 0.0, column
