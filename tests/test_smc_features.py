"""Executable pin for ``docs/smc_v1.md``, written from the document alone.

Every expected number below was derived by hand from the formulas in the spec
and never from ``nn.smc``: a test whose assertions were read back out of the
implementation agrees with that implementation by construction and therefore
cannot catch a bug in it. That independence is the only reason this file has
any value, so the fixtures are hand-built candle sequences small enough to
compute on paper rather than recorded engine output.

The one piece of the spec restated here is the §1.1 ATR, because a threshold
such as ``BREAK_ATR * atr_den[t]`` cannot be checked without it. Most fixtures
are built from candles that are exactly 1.0 tall and always contain the
previous close, which forces ``TR[t] == 1.0`` on every row and therefore
``ATR == 1.0`` exactly, so the surrounding arithmetic stays readable;
``test_fixture_atr_is_one_by_construction`` guards that property.
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

from nn.smc import (
    SMC_FEATURE_FAMILIES,
    SMC_FEATURE_NAMES,
    SMC_SPEC_VERSION,
    SMCSpec,
    compute_smc_features,
    compute_smc_features_segmented,
    smc_feature_columns,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_CANDLES = REPO_ROOT / "data" / "research" / "btc_usdt_1h_gen1_raw_pre_styx.parquet"
STYX = pd.Timestamp("2025-08-27T23:00:00+00:00")

START = pd.Timestamp("2024-01-01T00:00:00+00:00")
HOUR = pd.Timedelta(hours=1)

LOG1P = {n: math.log1p(n) for n in range(0, 12)}

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
    """``atr_den`` of §1.1, restated so threshold arithmetic can be checked."""
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
    """A deterministic series rich enough to fire every event family."""
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.4, rows))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + np.abs(rng.normal(0.0, 0.3, rows))
    low = np.minimum(open_, close) - np.abs(rng.normal(0.0, 0.3, rows))
    return _frame(list(zip(open_, high, low, close)))


# --- the fixtures themselves ------------------------------------------------ #
# F1: a flat top of three equal highs at indices 4, 5, 6. Strict-left /
# non-strict-right makes index 4 the single pivot; it is confirmed at row 7.
F1_SWING_HIGH = _frame(
    [
        (99.5, 100.0, 99.0, 99.5),
        (99.5, 100.2, 99.2, 99.7),
        (99.7, 100.4, 99.4, 99.9),
        (99.9, 100.6, 99.6, 100.1),
        (100.1, 101.0, 100.0, 100.5),
        (100.5, 101.0, 100.0, 100.5),
        (100.5, 101.0, 100.0, 100.5),
        (100.5, 100.8, 99.8, 100.3),
        (100.3, 100.6, 99.6, 100.1),
        (100.1, 100.4, 99.4, 99.9),
        (99.9, 100.2, 99.2, 99.7),
        (99.7, 100.0, 99.0, 99.5),
    ]
)

# F2: the mirror image -- a flat bottom of three equal lows at 4, 5, 6.
F2_SWING_LOW = _frame(
    [
        (100.5, 101.0, 100.0, 100.5),
        (100.5, 100.8, 99.8, 100.3),
        (100.3, 100.6, 99.6, 100.1),
        (100.1, 100.4, 99.4, 99.9),
        (99.9, 100.0, 99.0, 99.5),
        (99.5, 100.0, 99.0, 99.5),
        (99.5, 100.0, 99.0, 99.5),
        (99.5, 100.2, 99.2, 99.7),
        (99.7, 100.4, 99.4, 99.9),
        (99.9, 100.6, 99.6, 100.1),
        (100.1, 100.8, 99.8, 100.3),
        (100.3, 101.0, 100.0, 100.5),
    ]
)

# F3: swing low at index 4 (l = 97.9, confirmed at 7) then swing high at index
# 10 (h = 101.3, confirmed at 13). Nothing else fires anywhere in it.
F3_DELAY = _frame(
    _mid_candles(
        [
            100.0,
            99.6,
            99.2,
            98.8,
            98.4,
            98.8,
            99.2,
            99.6,
            100.0,
            100.4,
            100.8,
            100.4,
            100.0,
            99.6,
            99.2,
            98.8,
            98.4,
        ]
    )
)

# F4: swing high at 4 (h = 102.1, confirmed at 7), price walks away, then one
# outsized bullish candle at 11 closes clear of the level with dir still 0.
F4_BOS_BULL = _frame(
    _mid_candles([100.0, 100.4, 100.8, 101.2, 101.6, 101.2, 100.8, 100.4, 100.0, 99.6, 99.2])
    + [(99.2, 102.7, 99.1, 102.6), (102.6, 103.0, 102.0, 102.5), (102.5, 102.9, 101.9, 102.4)]
)

# F5: mirror of F4 -- swing low at 4 (l = 97.9), bearish break at 11.
F5_BOS_BEAR = _frame(
    _mid_candles([100.0, 99.6, 99.2, 98.8, 98.4, 98.8, 99.2, 99.6, 100.0, 100.4, 100.8])
    + [(100.8, 100.9, 97.3, 97.4), (97.4, 98.0, 97.0, 97.5), (97.5, 98.1, 97.1, 97.6)]
)

# F6: swing low at 4 (97.9) and swing high at 10 (101.3); bearish break at 14
# sets dir = -1, so the bullish break at 15 is a CHOCH rather than a BOS.
F6_CHOCH_BULL = _frame(
    _mid_candles(
        [
            100.0,
            99.6,
            99.2,
            98.8,
            98.4,
            98.8,
            99.2,
            99.6,
            100.0,
            100.4,
            100.8,
            100.4,
            100.0,
            99.6,
        ]
    )
    + [(99.6, 99.7, 97.4, 97.5), (97.5, 101.9, 97.4, 101.8)]
)

# F7: mirror of F6 -- swing high at 4 (102.1), swing low at 10 (98.7), bullish
# break at 14 sets dir = +1, bearish break at 15 is the CHOCH.
F7_CHOCH_BEAR = _frame(
    _mid_candles(
        [
            100.0,
            100.4,
            100.8,
            101.2,
            101.6,
            101.2,
            100.8,
            100.4,
            100.0,
            99.6,
            99.2,
            99.6,
            100.0,
            100.4,
        ]
    )
    + [(100.4, 102.6, 100.3, 102.5), (102.5, 102.6, 98.1, 98.2)]
)

# F8: swing highs at 4 (h = 101.0) and 11 (h = 100.9); the second confirms at
# row 14 and |100.9 - 101.0| = 0.1 <= 0.25 * 1.0, so equal highs fire there.
F8_EQUAL_HIGHS = _frame(
    _mid_candles(
        [
            99.6,
            100.0,
            100.2,
            100.4,
            100.5,
            100.1,
            99.7,
            99.3,
            99.5,
            99.7,
            99.9,
            100.4,
            100.0,
            99.7,
            99.4,
            99.5,
            99.6,
        ]
    )
)

# F9: mirror -- swing lows at 4 (l = 99.0) and 11 (l = 99.1), equal lows at 14.
F9_EQUAL_LOWS = _frame(
    _mid_candles(
        [
            100.4,
            100.0,
            99.8,
            99.6,
            99.5,
            99.9,
            100.3,
            100.7,
            100.5,
            100.3,
            100.1,
            99.6,
            100.0,
            100.3,
            100.6,
            100.5,
            100.4,
        ]
    )
)

# F10: swing high at 4 (102.1, confirmed at 7). Row 11 wicks to 102.6 but
# closes at 101.5; row 13 closes at 102.8 and is the real break.
F10_SWEEP_HIGH = _frame(
    _mid_candles([100.0, 100.4, 100.8, 101.2, 101.6, 101.2, 100.8, 100.4, 100.8, 101.2, 101.6])
    + [
        (101.6, 102.6, 101.4, 101.5),
        (101.5, 102.0, 101.0, 101.5),
        (101.5, 102.9, 101.4, 102.8),
    ]
)

# F11: mirror -- swing low at 4 (97.9), sweep at 11, genuine break at 13.
F11_SWEEP_LOW = _frame(
    _mid_candles([100.0, 99.6, 99.2, 98.8, 98.4, 98.8, 99.2, 99.6, 99.2, 98.8, 98.4])
    + [(98.4, 98.6, 97.4, 98.5), (98.5, 99.0, 98.0, 98.5), (98.5, 98.6, 97.1, 97.2)]
)

# F12: the F10 sweep at 11 (candle low 101.4), then closes 101.6, 101.1, 101.0.
# The first close below 101.4 is row 13.
F12_RECLAIM_HIGH = _frame(
    _mid_candles([100.0, 100.4, 100.8, 101.2, 101.6, 101.2, 100.8, 100.4, 100.8, 101.2, 101.6])
    + [
        (101.6, 102.6, 101.4, 101.5),
        (101.5, 101.9, 101.4, 101.6),
        (101.6, 101.7, 101.0, 101.1),
        (101.1, 101.4, 100.8, 101.0),
    ]
)

# F13: mirror -- sweep candle high 98.6, closes 98.4, 98.9, 98.8.
F13_RECLAIM_LOW = _frame(
    _mid_candles([100.0, 99.6, 99.2, 98.8, 98.4, 98.8, 99.2, 99.6, 99.2, 98.8, 98.4])
    + [
        (98.4, 98.6, 97.4, 98.5),
        (98.5, 98.6, 98.1, 98.4),
        (98.4, 99.0, 98.3, 98.9),
        (98.9, 99.2, 98.6, 98.8),
    ]
)

# Ten identical candles: TR = 1.0 on every row, and equal highs and equal lows
# mean strict-left comparison never holds, so the base carries no pivots.
_FLAT_BASE = [(100.0, 100.5, 99.5, 100.0)] * 10

# F14: row 10 satisfies both displacement tests, row 11 only the magnitude one
# (ratio 0.375) and row 12 only the ratio one (body 0.3 against ATR 1.199).
F14_DISPLACEMENT_BULL = _frame(
    _FLAT_BASE
    + [
        (100.0, 101.6, 99.9, 101.5),
        (101.5, 103.5, 99.5, 103.0),
        (103.0, 103.35, 102.95, 103.3),
    ]
)

# F15: the same three candles reflected.
F15_DISPLACEMENT_BEAR = _frame(
    _FLAT_BASE
    + [
        (100.0, 100.1, 98.4, 98.5),
        (98.5, 100.5, 96.5, 97.0),
        (97.0, 97.05, 96.65, 96.7),
    ]
)

# F16: bullish gap at 12 (l = 101.2 over h[10] = 100.8, size 0.4); row 13's gap
# is exactly 0.0 and row 15's is 0.03 against a 0.0511 minimum; row 16 trades
# down to 100.5 and fills the gap completely.
F16_FVG_BULL = _frame(
    _FLAT_BASE
    + [
        (100.0, 100.8, 99.8, 100.6),
        (100.6, 101.4, 100.4, 101.3),
        (101.3, 102.2, 101.2, 102.0),
        (102.0, 102.4, 101.4, 101.8),
        (101.8, 102.3, 101.3, 101.9),
        (102.5, 103.2, 102.43, 103.0),
        (103.0, 103.1, 100.5, 101.6),
    ]
)

# F17: the mirror -- bearish gap at 12 (h = 98.8 under l[10] = 99.2), filled at
# 16 by a candle whose high reaches 99.5.
F17_FVG_BEAR = _frame(
    _FLAT_BASE
    + [
        (100.0, 100.2, 99.2, 99.4),
        (99.4, 99.6, 98.6, 98.7),
        (98.7, 98.8, 97.8, 98.0),
        (98.0, 98.6, 97.6, 98.2),
        (98.2, 98.7, 97.7, 98.1),
        (97.5, 97.57, 96.8, 97.0),
        (97.0, 99.5, 96.9, 98.4),
    ]
)

# F18: F1's confirmed swing high plus a bullish FVG at row 13, then five
# missing hours, then eight identical candles that carry no structure of their
# own. Every post-gap feature is therefore zero unless state bridged the hole.
_F18_SEGMENT_ONE = [
    (99.5, 100.0, 99.0, 99.5),
    (99.5, 100.2, 99.2, 99.7),
    (99.7, 100.4, 99.4, 99.9),
    (99.9, 100.6, 99.6, 100.1),
    (100.1, 101.0, 100.0, 100.5),
    (100.5, 101.0, 100.0, 100.5),
    (100.5, 101.0, 100.0, 100.5),
    (100.5, 100.8, 99.8, 100.3),
    (100.3, 100.6, 99.6, 100.1),
    (100.1, 100.4, 99.4, 99.9),
    (99.9, 100.2, 99.2, 99.7),
    (99.7, 100.0, 99.0, 99.5),
    (99.5, 100.4, 99.4, 100.3),
    (100.3, 101.0, 100.2, 100.9),
]
_F18_SEGMENT_TWO = [(100.6, 100.9, 100.3, 100.6)] * 8
_F18_DATES = [START + i * HOUR for i in range(len(_F18_SEGMENT_ONE))] + [
    START + (len(_F18_SEGMENT_ONE) + 5 + i) * HOUR for i in range(len(_F18_SEGMENT_TWO))
]
F18_GAP = _frame(_F18_SEGMENT_ONE + _F18_SEGMENT_TWO, dates=_F18_DATES)
F18_SEGMENT_ONE_ONLY = _frame(_F18_SEGMENT_ONE)


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
    [F1_SWING_HIGH, F2_SWING_LOW, F3_DELAY, F8_EQUAL_HIGHS, F9_EQUAL_LOWS],
    ids=["f1", "f2", "f3", "f8", "f9"],
)
def test_fixture_atr_is_one_by_construction(frame: pd.DataFrame) -> None:
    assert _atr_den(frame) == pytest.approx(np.ones(len(frame)), abs=1e-12)


# --------------------------------------------------------------------------- #
# 1-2. pivots and the plateau tie-break
# --------------------------------------------------------------------------- #
def test_swing_high_plateau_confirms_the_earliest_candle() -> None:
    out = compute_smc_features(F1_SWING_HIGH)

    # h[4] = h[5] = h[6] = 101.0 with h[3] = 100.6 < 101.0: strict left and
    # non-strict right make index 4 the pivot and indices 5 and 6 not pivots.
    assert (out["smc_dist_swing_high_atr"].iloc[:7] == 0.0).all()
    assert (out["smc_age_swing_high"].iloc[:7] == 0.0).all()

    # Row 7 = 4 + PIVOT_RIGHT. (101.0 - 100.3) / 1.0 and log1p(7 - 4).
    assert out["smc_dist_swing_high_atr"].iloc[7] == pytest.approx(0.7)
    assert out["smc_age_swing_high"].iloc[7] == pytest.approx(LOG1P[3])
    assert out["smc_dist_swing_high_atr"].iloc[8] == pytest.approx(0.9)
    assert out["smc_age_swing_high"].iloc[8] == pytest.approx(LOG1P[4])
    # Still index 4 six bars later: a non-strict left comparison would have
    # confirmed index 6 at row 9 and reported log1p(4) here instead.
    assert out["smc_dist_swing_high_atr"].iloc[10] == pytest.approx(1.3)
    assert out["smc_age_swing_high"].iloc[10] == pytest.approx(LOG1P[6])

    # No swing low anywhere in the fixture.
    assert (out["smc_structure_valid"] == 0.0).all()
    assert (out["smc_dist_swing_low_atr"] == 0.0).all()
    assert (out["smc_age_swing_low"] == 0.0).all()
    assert (out["smc_range_width_atr"] == 0.0).all()
    assert (out["smc_range_position"] == 0.5).all()


def test_swing_low_plateau_confirms_the_earliest_candle() -> None:
    out = compute_smc_features(F2_SWING_LOW)

    # l[4] = l[5] = l[6] = 99.0, l[3] = 99.4 > 99.0: index 4 is the pivot.
    assert (out["smc_dist_swing_low_atr"].iloc[:7] == 0.0).all()
    assert (out["smc_age_swing_low"].iloc[:7] == 0.0).all()

    # Row 7: (99.7 - 99.0) / 1.0, age log1p(3).
    assert out["smc_dist_swing_low_atr"].iloc[7] == pytest.approx(0.7)
    assert out["smc_age_swing_low"].iloc[7] == pytest.approx(LOG1P[3])
    assert out["smc_dist_swing_low_atr"].iloc[8] == pytest.approx(0.9)
    assert out["smc_age_swing_low"].iloc[8] == pytest.approx(LOG1P[4])
    assert out["smc_dist_swing_low_atr"].iloc[10] == pytest.approx(1.3)
    assert out["smc_age_swing_low"].iloc[10] == pytest.approx(LOG1P[6])

    assert (out["smc_structure_valid"] == 0.0).all()
    assert (out["smc_dist_swing_high_atr"] == 0.0).all()
    assert (out["smc_age_swing_high"] == 0.0).all()


# --------------------------------------------------------------------------- #
# 3. the delayed confirmation
# --------------------------------------------------------------------------- #
def test_swing_high_is_invisible_until_pivot_right_candles_have_closed() -> None:
    out = compute_smc_features(F3_DELAY)

    # The swing low at index 4 (l = 97.9) confirms first, at row 7.
    assert (out["smc_dist_swing_low_atr"].iloc[:7] == 0.0).all()
    assert out["smc_dist_swing_low_atr"].iloc[7] == pytest.approx(1.7)  # 99.6 - 97.9
    assert out["smc_age_swing_low"].iloc[7] == pytest.approx(LOG1P[3])

    # The swing high sits at index 10 (h = 101.3). Rows 10, 11 and 12 cannot
    # know it: h[11], h[12], h[13] are part of the predicate that defines it.
    for row in (10, 11, 12):
        assert out["smc_structure_valid"].iloc[row] == 0.0
        assert out["smc_dist_swing_high_atr"].iloc[row] == 0.0
        assert out["smc_age_swing_high"].iloc[row] == 0.0
        assert out["smc_range_width_atr"].iloc[row] == 0.0
        assert out["smc_range_position"].iloc[row] == 0.5

    # Row 13 = 10 + PIVOT_RIGHT is the first row that may report it.
    assert out["smc_structure_valid"].iloc[13] == 1.0
    assert out["smc_dist_swing_high_atr"].iloc[13] == pytest.approx(1.7)  # 101.3 - 99.6
    assert out["smc_age_swing_high"].iloc[13] == pytest.approx(LOG1P[3])

    # Range geometry, once both sides exist: 101.3 - 97.9 = 3.4 wide.
    assert out["smc_range_width_atr"].iloc[14] == pytest.approx(3.4)
    assert out["smc_range_position"].iloc[14] == pytest.approx(
        1.3 / 3.4
    )  # (99.2 - 97.9) / 3.4
    assert (out["smc_structure_direction"] == 0.0).all()


# --------------------------------------------------------------------------- #
# 4-7. breaks
# --------------------------------------------------------------------------- #
def test_bullish_bos() -> None:
    out = compute_smc_features(F4_BOS_BULL)
    # TR[11] = 102.7 - 99.1 = 3.6, so ATR[11] = (13 * 1.0 + 3.6) / 14.
    atr = 16.6 / 14.0
    assert _atr_den(F4_BOS_BULL)[11] == pytest.approx(atr, rel=1e-12)

    # active_sh = h[4] = 102.1 from row 7; c[11] = 102.6 > 102.1 + 0.10 * ATR
    # = 102.2186, and dir was 0, so this is a BOS and not a CHOCH.
    assert (out["smc_bos_bull"].iloc[:11] == 0.0).all()
    assert out["smc_dist_swing_high_atr"].iloc[10] == pytest.approx(2.9)  # 102.1 - 99.2
    assert out["smc_bos_bull"].iloc[11] == 1.0
    assert out["smc_choch_bull"].iloc[11] == 0.0
    assert out["smc_bos_bear"].iloc[11] == 0.0
    assert out["smc_sweep_high"].iloc[11] == 0.0
    assert out["smc_structure_direction"].iloc[11] == 1.0
    assert out["smc_break_magnitude_atr"].iloc[11] == pytest.approx(0.5 / atr)
    assert out["smc_bars_since_break"].iloc[11] == 0.0
    assert out["smc_bars_since_break"].iloc[12] == pytest.approx(LOG1P[1])
    assert out["smc_structure_direction"].iloc[12] == 1.0
    assert (out["smc_break_magnitude_atr"].iloc[:11] == 0.0).all()


def test_bearish_bos() -> None:
    out = compute_smc_features(F5_BOS_BEAR)
    atr = 16.6 / 14.0  # TR[11] = 100.9 - 97.3 = 3.6
    assert _atr_den(F5_BOS_BEAR)[11] == pytest.approx(atr, rel=1e-12)

    # active_sl = l[4] = 97.9; c[11] = 97.4 < 97.9 - 0.10 * ATR = 97.7814.
    assert (out["smc_bos_bear"].iloc[:11] == 0.0).all()
    assert out["smc_bos_bear"].iloc[11] == 1.0
    assert out["smc_choch_bear"].iloc[11] == 0.0
    assert out["smc_bos_bull"].iloc[11] == 0.0
    assert out["smc_sweep_low"].iloc[11] == 0.0
    assert out["smc_structure_direction"].iloc[11] == -1.0
    assert out["smc_break_magnitude_atr"].iloc[11] == pytest.approx(-0.5 / atr)
    assert out["smc_bars_since_break"].iloc[11] == 0.0
    assert out["smc_bars_since_break"].iloc[12] == pytest.approx(LOG1P[1])


def test_bullish_choch_after_a_bearish_break() -> None:
    out = compute_smc_features(F6_CHOCH_BULL)
    atr14 = 15.3 / 14.0  # TR[14] = 99.7 - 97.4 = 2.3
    atr15 = (13.0 * atr14 + 4.5) / 14.0  # TR[15] = 101.9 - 97.4 = 4.5
    den = _atr_den(F6_CHOCH_BULL)
    assert den[14] == pytest.approx(atr14, rel=1e-12)
    assert den[15] == pytest.approx(atr15, rel=1e-12)

    # Row 14 breaks the swing low at 97.9 with dir == 0 -> BOS, dir := -1.
    assert out["smc_bos_bear"].iloc[14] == 1.0
    assert out["smc_choch_bear"].iloc[14] == 0.0
    assert out["smc_structure_direction"].iloc[14] == -1.0
    assert out["smc_break_magnitude_atr"].iloc[14] == pytest.approx(-0.4 / atr14)

    # Row 15 closes at 101.8, clear of the swing high at 101.3 + 0.10 * ATR
    # = 101.4336. dir was -1, so the same break is a CHOCH, not a BOS.
    assert out["smc_choch_bull"].iloc[15] == 1.0
    assert out["smc_bos_bull"].iloc[15] == 0.0
    assert out["smc_structure_direction"].iloc[15] == 1.0
    assert out["smc_break_magnitude_atr"].iloc[15] == pytest.approx(0.5 / atr15)
    assert out["smc_bars_since_break"].iloc[15] == 0.0


def test_bearish_choch_after_a_bullish_break() -> None:
    out = compute_smc_features(F7_CHOCH_BEAR)
    atr14 = 15.3 / 14.0  # TR[14] = 102.6 - 100.3 = 2.3
    atr15 = (13.0 * atr14 + 4.5) / 14.0  # TR[15] = 102.6 - 98.1 = 4.5
    den = _atr_den(F7_CHOCH_BEAR)
    assert den[14] == pytest.approx(atr14, rel=1e-12)
    assert den[15] == pytest.approx(atr15, rel=1e-12)

    # Row 14 breaks the swing high at 102.1 with dir == 0 -> BOS, dir := +1.
    assert out["smc_bos_bull"].iloc[14] == 1.0
    assert out["smc_choch_bull"].iloc[14] == 0.0
    assert out["smc_structure_direction"].iloc[14] == 1.0
    assert out["smc_break_magnitude_atr"].iloc[14] == pytest.approx(0.4 / atr14)

    # Row 15 closes at 98.2 below the swing low 98.7 - 0.10 * ATR = 98.5664.
    assert out["smc_choch_bear"].iloc[15] == 1.0
    assert out["smc_bos_bear"].iloc[15] == 0.0
    assert out["smc_structure_direction"].iloc[15] == -1.0
    assert out["smc_break_magnitude_atr"].iloc[15] == pytest.approx(-0.5 / atr15)


# --------------------------------------------------------------------------- #
# 8-9. equal highs and equal lows
# --------------------------------------------------------------------------- #
def test_equal_highs_rest_at_the_higher_of_the_pair() -> None:
    out = compute_smc_features(F8_EQUAL_HIGHS)

    # Swing high at 11 (h = 100.9) confirms at row 14 against prev_sh = 101.0:
    # |100.9 - 101.0| = 0.1 <= EQUAL_ATR * 1.0 = 0.25.
    assert out["smc_equal_high"].iloc[14] == 1.0
    assert out["smc_equal_high"].sum() == 1.0
    assert (out["smc_eqh_active"].iloc[:14] == 0.0).all()
    assert (out["smc_eqh_dist_atr"].iloc[:14] == 0.0).all()

    # The level is max(100.9, 101.0) = 101.0, so the distances below are 0.1
    # larger than they would be if the engine kept h[i] instead.
    assert out["smc_eqh_active"].iloc[14] == 1.0
    assert out["smc_eqh_dist_atr"].iloc[14] == pytest.approx(1.6)  # 101.0 - 99.4
    assert out["smc_eqh_dist_atr"].iloc[15] == pytest.approx(1.5)  # 101.0 - 99.5
    assert out["smc_eqh_dist_atr"].iloc[16] == pytest.approx(1.4)  # 101.0 - 99.6
    assert (out["smc_equal_low"] == 0.0).all()


def test_equal_lows_rest_at_the_lower_of_the_pair() -> None:
    out = compute_smc_features(F9_EQUAL_LOWS)

    # Swing low at 11 (l = 99.1) confirms at 14 against prev_sl = 99.0.
    assert out["smc_equal_low"].iloc[14] == 1.0
    assert out["smc_equal_low"].sum() == 1.0
    assert (out["smc_eql_active"].iloc[:14] == 0.0).all()
    assert (out["smc_eql_dist_atr"].iloc[:14] == 0.0).all()

    # Level = min(99.1, 99.0) = 99.0.
    assert out["smc_eql_active"].iloc[14] == 1.0
    assert out["smc_eql_dist_atr"].iloc[14] == pytest.approx(1.6)  # 100.6 - 99.0
    assert out["smc_eql_dist_atr"].iloc[15] == pytest.approx(1.5)  # 100.5 - 99.0
    assert out["smc_eql_dist_atr"].iloc[16] == pytest.approx(1.4)  # 100.4 - 99.0
    assert (out["smc_equal_high"] == 0.0).all()


# --------------------------------------------------------------------------- #
# 10-11. sweeps
# --------------------------------------------------------------------------- #
def test_high_side_sweep_is_not_a_break_and_leaves_the_level_active() -> None:
    out = compute_smc_features(F10_SWEEP_HIGH)
    atr11 = 14.2 / 14.0  # TR[11] = 102.6 - 101.4 = 1.2
    atr13 = (13.0 * ((13.0 * atr11 + 1.0) / 14.0) + 1.5) / 14.0  # TR[12] = 1.0, TR[13] = 1.5
    den = _atr_den(F10_SWEEP_HIGH)
    assert den[11] == pytest.approx(atr11, rel=1e-12)
    assert den[13] == pytest.approx(atr13, rel=1e-12)

    # active_sh = 102.1. h[11] = 102.6 > 102.1 + 0.05 * ATR = 102.1507 while
    # c[11] = 101.5 <= 102.1, and 101.5 is far below the break threshold
    # 102.1 + 0.10 * ATR = 102.2014.
    assert out["smc_sweep_high"].iloc[11] == 1.0
    assert out["smc_sweep_high"].sum() == 1.0
    assert out["smc_bos_bull"].iloc[11] == 0.0
    assert out["smc_choch_bull"].iloc[11] == 0.0
    assert out["smc_break_magnitude_atr"].iloc[11] == 0.0
    assert out["smc_bars_since_sweep_high"].iloc[11] == 0.0
    assert out["smc_bars_since_sweep_high"].iloc[12] == pytest.approx(LOG1P[1])
    assert out["smc_bars_since_sweep_high"].iloc[13] == pytest.approx(LOG1P[2])

    # The probed level survives the sweep: row 13 still breaks 102.1.
    assert out["smc_bos_bull"].iloc[13] == 1.0
    assert out["smc_break_magnitude_atr"].iloc[13] == pytest.approx(0.7 / atr13)
    assert (out["smc_sweep_high_reclaim"] == 0.0).all()


def test_low_side_sweep_is_not_a_break_and_leaves_the_level_active() -> None:
    out = compute_smc_features(F11_SWEEP_LOW)
    atr11 = 14.2 / 14.0  # TR[11] = 98.6 - 97.4 = 1.2
    atr13 = (13.0 * ((13.0 * atr11 + 1.0) / 14.0) + 1.5) / 14.0
    den = _atr_den(F11_SWEEP_LOW)
    assert den[11] == pytest.approx(atr11, rel=1e-12)
    assert den[13] == pytest.approx(atr13, rel=1e-12)

    # active_sl = 97.9. l[11] = 97.4 < 97.9 - 0.05 * ATR = 97.8493 while
    # c[11] = 98.5 >= 97.9; the break threshold is 97.9 - 0.10 * ATR = 97.7986.
    assert out["smc_sweep_low"].iloc[11] == 1.0
    assert out["smc_sweep_low"].sum() == 1.0
    assert out["smc_bos_bear"].iloc[11] == 0.0
    assert out["smc_choch_bear"].iloc[11] == 0.0
    assert out["smc_break_magnitude_atr"].iloc[11] == 0.0
    assert out["smc_bars_since_sweep_low"].iloc[11] == 0.0
    assert out["smc_bars_since_sweep_low"].iloc[12] == pytest.approx(LOG1P[1])
    assert out["smc_bars_since_sweep_low"].iloc[13] == pytest.approx(LOG1P[2])

    assert out["smc_bos_bear"].iloc[13] == 1.0
    assert out["smc_break_magnitude_atr"].iloc[13] == pytest.approx(-0.7 / atr13)
    assert (out["smc_sweep_low_reclaim"] == 0.0).all()


# --------------------------------------------------------------------------- #
# 12-13. reclaims
# --------------------------------------------------------------------------- #
def test_high_sweep_reclaim_fires_on_the_first_close_below_the_sweep_low() -> None:
    out = compute_smc_features(F12_RECLAIM_HIGH)

    # Sweep at row 11, whose candle low is 101.4. Closes: 101.6 (row 12, still
    # inside the candle), 101.1 (row 13, the first below it), 101.0 (row 14).
    assert out["smc_sweep_high"].iloc[11] == 1.0
    assert out["smc_sweep_high_reclaim"].iloc[11] == 0.0
    assert out["smc_sweep_high_reclaim"].iloc[12] == 0.0
    assert out["smc_sweep_high_reclaim"].iloc[13] == 1.0
    assert out["smc_sweep_high_reclaim"].iloc[14] == 0.0
    assert out["smc_sweep_high_reclaim"].sum() == 1.0
    assert out["smc_bars_since_sweep_high"].iloc[13] == pytest.approx(LOG1P[2])


def test_low_sweep_reclaim_fires_on_the_first_close_above_the_sweep_high() -> None:
    out = compute_smc_features(F13_RECLAIM_LOW)

    # Sweep at row 11, candle high 98.6. Closes: 98.4, 98.9 (first above), 98.8.
    assert out["smc_sweep_low"].iloc[11] == 1.0
    assert out["smc_sweep_low_reclaim"].iloc[11] == 0.0
    assert out["smc_sweep_low_reclaim"].iloc[12] == 0.0
    assert out["smc_sweep_low_reclaim"].iloc[13] == 1.0
    assert out["smc_sweep_low_reclaim"].iloc[14] == 0.0
    assert out["smc_sweep_low_reclaim"].sum() == 1.0
    assert out["smc_bars_since_sweep_low"].iloc[13] == pytest.approx(LOG1P[2])


# --------------------------------------------------------------------------- #
# 14-15. displacement
# --------------------------------------------------------------------------- #
def test_bullish_displacement_requires_both_magnitude_and_ratio() -> None:
    out = compute_smc_features(F14_DISPLACEMENT_BULL)
    atr10 = 14.7 / 14.0  # TR[10] = 101.6 - 99.9 = 1.7
    atr11 = (13.0 * atr10 + 4.0) / 14.0  # TR[11] = 103.5 - 99.5 = 4.0
    atr12 = (13.0 * atr11 + 0.4) / 14.0  # TR[12] = 103.35 - 102.95 = 0.4
    den = _atr_den(F14_DISPLACEMENT_BULL)
    assert den[10:13] == pytest.approx([atr10, atr11, atr12], rel=1e-12)

    # Row 10: body 1.5 >= 1.00 * 1.05 and 1.5 / 1.7 = 0.882 >= 0.60.
    assert out["smc_displacement_bull"].iloc[10] == 1.0
    assert out["smc_displacement_bull"].sum() == 1.0
    assert (out["smc_displacement_bear"] == 0.0).all()
    assert out["smc_body_signed_atr"].iloc[10] == pytest.approx(1.5 / atr10)
    assert out["smc_body_abs_atr"].iloc[10] == pytest.approx(1.5 / atr10)
    assert out["smc_body_ratio"].iloc[10] == pytest.approx(1.5 / 1.7)

    # Row 11 clears the magnitude test (1.5 / 1.2607 = 1.19 >= 1.00) and fails
    # only the ratio test (1.5 / 4.0 = 0.375 < 0.60).
    assert out["smc_body_abs_atr"].iloc[11] == pytest.approx(1.5 / atr11)
    assert out["smc_body_abs_atr"].iloc[11] > 1.0
    assert out["smc_body_ratio"].iloc[11] == pytest.approx(0.375)
    assert out["smc_displacement_bull"].iloc[11] == 0.0

    # Row 12 clears the ratio test (0.3 / 0.4 = 0.75) and fails the magnitude
    # one (0.3 / 1.1992 = 0.25 < 1.00).
    assert out["smc_body_ratio"].iloc[12] == pytest.approx(0.75)
    assert out["smc_body_abs_atr"].iloc[12] == pytest.approx(0.3 / atr12)
    assert out["smc_body_abs_atr"].iloc[12] < 1.0
    assert out["smc_displacement_bull"].iloc[12] == 0.0


def test_bearish_displacement_requires_both_magnitude_and_ratio() -> None:
    out = compute_smc_features(F15_DISPLACEMENT_BEAR)
    atr10 = 14.7 / 14.0  # TR[10] = 100.1 - 98.4 = 1.7
    atr11 = (13.0 * atr10 + 4.0) / 14.0  # TR[11] = 100.5 - 96.5 = 4.0
    atr12 = (13.0 * atr11 + 0.4) / 14.0  # TR[12] = 97.05 - 96.65 = 0.4
    den = _atr_den(F15_DISPLACEMENT_BEAR)
    assert den[10:13] == pytest.approx([atr10, atr11, atr12], rel=1e-12)

    assert out["smc_displacement_bear"].iloc[10] == 1.0
    assert out["smc_displacement_bear"].sum() == 1.0
    assert (out["smc_displacement_bull"] == 0.0).all()
    assert out["smc_body_signed_atr"].iloc[10] == pytest.approx(-1.5 / atr10)
    assert out["smc_body_abs_atr"].iloc[10] == pytest.approx(1.5 / atr10)
    assert out["smc_body_ratio"].iloc[10] == pytest.approx(1.5 / 1.7)

    assert out["smc_body_abs_atr"].iloc[11] == pytest.approx(1.5 / atr11)
    assert out["smc_body_ratio"].iloc[11] == pytest.approx(0.375)
    assert out["smc_displacement_bear"].iloc[11] == 0.0

    assert out["smc_body_ratio"].iloc[12] == pytest.approx(0.75)
    assert out["smc_body_abs_atr"].iloc[12] == pytest.approx(0.3 / atr12)
    assert out["smc_displacement_bear"].iloc[12] == 0.0


# --------------------------------------------------------------------------- #
# 16-17. fair-value gaps
# --------------------------------------------------------------------------- #
def test_bullish_fvg_size_minimum_and_fill() -> None:
    out = compute_smc_features(F16_FVG_BULL)
    atr15 = 14.3 / 14.0  # TR[15] = |103.2 - 101.9| = 1.3
    den = _atr_den(F16_FVG_BULL)
    assert den[12] == pytest.approx(1.0, rel=1e-12)
    assert den[15] == pytest.approx(atr15, rel=1e-12)

    # Row 12: l[12] = 101.2 > h[10] = 100.8, size 0.4 >= 0.05 * 1.0.
    assert out["smc_fvg_bull"].iloc[12] == 1.0
    assert out["smc_fvg_bull"].sum() == 1.0
    assert (out["smc_fvg_bear"] == 0.0).all()
    assert out["smc_fvg_bull_size_atr"].iloc[12] == pytest.approx(0.4)
    assert (out["smc_fvg_bull_size_atr"].drop(index=12) == 0.0).all()

    # The gap [100.8, 101.2] is active from its own candle onward.
    assert out["smc_bull_fvg_active"].iloc[12] == 1.0
    assert out["smc_bull_fvg_dist_atr"].iloc[12] == pytest.approx(0.8)  # 102.0 - 101.2
    assert out["smc_bull_fvg_dist_atr"].iloc[13] == pytest.approx(0.6)  # 101.8 - 101.2
    assert out["smc_bull_fvg_dist_atr"].iloc[14] == pytest.approx(0.7)  # 101.9 - 101.2
    assert out["smc_bull_fvg_dist_atr"].iloc[15] == pytest.approx(1.8 / atr15)

    # Row 13's imbalance is exactly 0.0 wide (l[13] = h[11] = 101.4) and row
    # 15's is 0.03 against a minimum of 0.05 * 1.0214 = 0.0511: neither fires.
    assert out["smc_fvg_bull"].iloc[13] == 0.0
    assert out["smc_fvg_bull"].iloc[15] == 0.0

    # Row 16 trades to 100.5, at or below the 100.8 floor: fully filled, and
    # the fill is processed before the row is emitted. Were it still active the
    # distance would be (101.6 - 101.2) / 1.1342 = 0.353, not 0.
    assert out["smc_bull_fvg_active"].iloc[16] == 0.0
    assert out["smc_bull_fvg_dist_atr"].iloc[16] == 0.0


def test_bearish_fvg_size_minimum_and_fill() -> None:
    out = compute_smc_features(F17_FVG_BEAR)
    atr15 = 14.3 / 14.0  # TR[15] = |96.8 - 98.1| = 1.3
    den = _atr_den(F17_FVG_BEAR)
    assert den[12] == pytest.approx(1.0, rel=1e-12)
    assert den[15] == pytest.approx(atr15, rel=1e-12)

    # Row 12: h[12] = 98.8 < l[10] = 99.2, size 0.4 >= 0.05 * 1.0.
    assert out["smc_fvg_bear"].iloc[12] == 1.0
    assert out["smc_fvg_bear"].sum() == 1.0
    assert (out["smc_fvg_bull"] == 0.0).all()
    assert out["smc_fvg_bear_size_atr"].iloc[12] == pytest.approx(0.4)
    assert (out["smc_fvg_bear_size_atr"].drop(index=12) == 0.0).all()

    # Gap [98.8, 99.2]; the reported distance is lo - c[t].
    assert out["smc_bear_fvg_active"].iloc[12] == 1.0
    assert out["smc_bear_fvg_dist_atr"].iloc[12] == pytest.approx(0.8)  # 98.8 - 98.0
    assert out["smc_bear_fvg_dist_atr"].iloc[13] == pytest.approx(0.6)  # 98.8 - 98.2
    assert out["smc_bear_fvg_dist_atr"].iloc[14] == pytest.approx(0.7)  # 98.8 - 98.1
    assert out["smc_bear_fvg_dist_atr"].iloc[15] == pytest.approx(1.8 / atr15)

    # Row 13's imbalance is 0.0 wide (h[13] = l[11] = 98.6), row 15's is 0.03.
    assert out["smc_fvg_bear"].iloc[13] == 0.0
    assert out["smc_fvg_bear"].iloc[15] == 0.0

    # Row 16 reaches 99.5, at or above the 99.2 ceiling: fully filled.
    assert out["smc_bear_fvg_active"].iloc[16] == 0.0
    assert out["smc_bear_fvg_dist_atr"].iloc[16] == 0.0


# --------------------------------------------------------------------------- #
# 18. gap reset
# --------------------------------------------------------------------------- #
def test_state_does_not_bridge_a_missing_candle_gap() -> None:
    out = compute_smc_features_segmented(F18_GAP, timeframe="1h")
    assert len(out) == len(F18_GAP)

    # Pre-gap the segment carries live state: a swing high confirmed at row 7
    # (h[4] = 101.0) and a bullish FVG created at row 13 (l[13] = 100.2 over
    # h[11] = 100.0). ATR[13] = (13 * 1.0 + 0.8) / 14 because TR[13] = 0.8.
    atr13 = 13.8 / 14.0
    assert _atr_den(F18_SEGMENT_ONE_ONLY)[13] == pytest.approx(atr13, rel=1e-12)
    assert out["smc_dist_swing_high_atr"].iloc[13] == pytest.approx(0.1 / atr13)
    assert out["smc_age_swing_high"].iloc[13] == pytest.approx(LOG1P[9])
    assert out["smc_fvg_bull"].iloc[13] == 1.0
    assert out["smc_fvg_bull_size_atr"].iloc[13] == pytest.approx(0.2 / atr13)
    assert out["smc_bull_fvg_active"].iloc[13] == 1.0
    assert out["smc_bull_fvg_dist_atr"].iloc[13] == pytest.approx(0.7 / atr13)

    # The post-gap candles never fill that gap (their low is 100.3 > 100.0) and
    # close at 100.6, above its 100.2 ceiling and below the 101.0 swing high,
    # so bridged state would show up as non-zero on every one of them.
    after = out.iloc[len(_F18_SEGMENT_ONE) :]
    assert len(after) == len(_F18_SEGMENT_TWO)
    for column in (
        "smc_structure_valid",
        "smc_dist_swing_high_atr",
        "smc_age_swing_high",
        "smc_dist_swing_low_atr",
        "smc_age_swing_low",
        "smc_structure_direction",
        "smc_range_width_atr",
        "smc_bull_fvg_active",
        "smc_bull_fvg_dist_atr",
        "smc_fvg_bull",
        "smc_eqh_active",
        "smc_eqh_dist_atr",
        "smc_bos_bull",
        "smc_bos_bear",
        "smc_choch_bull",
        "smc_choch_bear",
        "smc_break_magnitude_atr",
        "smc_bars_since_break",
        "smc_sweep_high",
        "smc_sweep_low",
        "smc_bars_since_sweep_high",
        "smc_bars_since_sweep_low",
    ):
        assert (after[column] == 0.0).all(), column
    assert (after["smc_range_position"] == 0.5).all()

    # And the pre-gap rows are exactly what the first segment produces alone.
    assert_frame_equal(
        out.iloc[: len(_F18_SEGMENT_ONE)],
        compute_smc_features(F18_SEGMENT_ONE_ONLY),
    )


# --------------------------------------------------------------------------- #
# 19. append invariance
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n,k", [(50, 1), (80, 3), (120, 10), (200, 50), (310, 90)])
def test_append_invariance_synthetic(n: int, k: int) -> None:
    raw = _random_walk(rows=420, seed=7)
    long = compute_smc_features(raw.iloc[: n + k])
    short = compute_smc_features(raw.iloc[:n])
    assert_frame_equal(long.iloc[:n], short)


@pytest.mark.parametrize("n,k", [(60, 5), (150, 20), (240, 100)])
def test_append_invariance_synthetic_segmented(n: int, k: int) -> None:
    raw = _random_walk(rows=420, seed=11)
    # Punch two market-data holes so the appended slice must re-segment too.
    gapped = raw.drop(index=[137, 138, 139, 264]).reset_index(drop=True)
    long = compute_smc_features_segmented(gapped.iloc[: n + k], timeframe="1h")
    short = compute_smc_features_segmented(gapped.iloc[:n], timeframe="1h")
    assert_frame_equal(long.iloc[:n], short)


@pytest.mark.parametrize("n,k", [(1000, 1), (1800, 7), (2500, 600)])
def test_append_invariance_real(real_contiguous: pd.DataFrame, n: int, k: int) -> None:
    long = compute_smc_features(real_contiguous.iloc[: n + k])
    short = compute_smc_features(real_contiguous.iloc[:n])
    assert_frame_equal(long.iloc[:n], short)


@pytest.mark.parametrize("n,k", [(1200, 1), (2000, 300), (3000, 50)])
def test_append_invariance_real_segmented(real_candles: pd.DataFrame, n: int, k: int) -> None:
    long = compute_smc_features_segmented(real_candles.iloc[: n + k], timeframe="1h")
    short = compute_smc_features_segmented(real_candles.iloc[:n], timeframe="1h")
    assert_frame_equal(long.iloc[:n], short)


# --------------------------------------------------------------------------- #
# 20-22. the public contract
# --------------------------------------------------------------------------- #
def test_no_nan_no_inf_and_exact_column_order_on_real_candles(
    real_candles: pd.DataFrame,
) -> None:
    out = compute_smc_features_segmented(real_candles, timeframe="1h")

    assert list(out.columns) == list(SMC_FEATURE_NAMES)
    assert len(out) == len(real_candles)
    assert (out.dtypes == np.float64).all()
    assert np.isfinite(out.to_numpy()).all()
    assert not out.isna().to_numpy().any()


def test_no_nan_no_inf_on_a_single_real_segment(real_contiguous: pd.DataFrame) -> None:
    out = compute_smc_features(real_contiguous)

    assert list(out.columns) == list(SMC_FEATURE_NAMES)
    assert out.index.equals(real_contiguous.index)
    assert np.isfinite(out.to_numpy()).all()


def test_feature_names_are_the_thirty_nine_columns_of_the_spec() -> None:
    assert SMC_SPEC_VERSION == "smc_v1"
    assert SMC_FEATURE_NAMES == (
        "smc_structure_valid",
        "smc_dist_swing_high_atr",
        "smc_dist_swing_low_atr",
        "smc_age_swing_high",
        "smc_age_swing_low",
        "smc_structure_direction",
        "smc_range_width_atr",
        "smc_range_position",
        "smc_equal_high",
        "smc_equal_low",
        "smc_eqh_active",
        "smc_eql_active",
        "smc_eqh_dist_atr",
        "smc_eql_dist_atr",
        "smc_bos_bull",
        "smc_bos_bear",
        "smc_choch_bull",
        "smc_choch_bear",
        "smc_break_magnitude_atr",
        "smc_bars_since_break",
        "smc_sweep_high",
        "smc_sweep_low",
        "smc_sweep_high_reclaim",
        "smc_sweep_low_reclaim",
        "smc_bars_since_sweep_high",
        "smc_bars_since_sweep_low",
        "smc_body_signed_atr",
        "smc_body_abs_atr",
        "smc_body_ratio",
        "smc_displacement_bull",
        "smc_displacement_bear",
        "smc_fvg_bull",
        "smc_fvg_bear",
        "smc_fvg_bull_size_atr",
        "smc_fvg_bear_size_atr",
        "smc_bull_fvg_active",
        "smc_bear_fvg_active",
        "smc_bull_fvg_dist_atr",
        "smc_bear_fvg_dist_atr",
    )
    assert smc_feature_columns() == list(SMC_FEATURE_NAMES)


def test_feature_families_partition_the_feature_names() -> None:
    families = SMC_FEATURE_FAMILIES
    assert len(families) == 6

    union: set[str] = set()
    total = 0
    for name, columns in families.items():
        assert len(set(columns)) == len(columns), name
        assert union.isdisjoint(columns), name
        union.update(columns)
        total += len(columns)

    assert total == 39
    assert len(SMC_FEATURE_NAMES) == 39
    assert union == set(SMC_FEATURE_NAMES)
    # §4's six families are 8, 6, 6, 6, 5 and 8 columns wide.
    assert sorted(len(columns) for columns in families.values()) == [5, 6, 6, 6, 8, 8]


def test_spec_hash_is_stable_and_changes_with_every_constant() -> None:
    spec = SMCSpec()
    assert spec.spec_hash() == spec.spec_hash() == SMCSpec().spec_hash()
    assert len(spec.spec_hash()) == 64
    assert set(spec.spec_hash()) <= set("0123456789abcdef")

    assert spec.to_dict() == {
        "atr_period": 14,
        "pivot_left": 3,
        "pivot_right": 3,
        "break_atr": 0.10,
        "sweep_atr": 0.05,
        "equal_atr": 0.25,
        "displacement_body_atr": 1.00,
        "displacement_body_ratio": 0.60,
        "fvg_min_atr": 0.05,
    }

    changed = {
        "atr_period": 20,
        "pivot_left": 4,
        "pivot_right": 5,
        "break_atr": 0.2,
        "sweep_atr": 0.1,
        "equal_atr": 0.3,
        "displacement_body_atr": 1.5,
        "displacement_body_ratio": 0.7,
        "fvg_min_atr": 0.02,
    }
    for field, value in changed.items():
        other = dataclasses.replace(spec, **{field: value})
        assert other.spec_hash() != spec.spec_hash(), field
