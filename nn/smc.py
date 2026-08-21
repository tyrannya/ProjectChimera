"""The causal market-structure information set of ``docs/smc_v1.md``.

That document, not this module, is the definition. Every constant, formula and
firing rule here is a transcription of it, and a disagreement between the two is
a bug in one of them rather than a licence to reinterpret either.

**One forward pass, because append invariance must hold by construction.** A
pivot at index ``i`` is a function of ``h[i-3 … i+3]``, so it cannot be known
until candle ``i+3`` has closed. Detecting pivots with a centred window and
shifting the result, or filling structure state backwards from the candle that
confirmed it, produces a frame whose historical rows change when a new candle
arrives — which is exactly what makes a backtest unfalsifiable, because the
model is then scored on levels that were drawn after the move. Row ``t`` here
first absorbs the pivot at ``t - pivot_right`` and then emits its own features,
so a written value can never be revised by data that had not happened yet.
``F(candles[:N+K])[:N] == F(candles[:N])`` is a consequence of the control flow,
not a property a test has to hope for.

**The intra-row order is declared.** Several events can land on one candle, and
which state they see is part of the answer, so the loop fixes the order once:

1. absorb the pivot confirmed at this row (§3.2), creating any equal level (§3.4);
2. remove equal levels this candle's high/low traded through (§3.4);
3. remove fair-value gaps this candle filled (§3.7);
4. breaks — bullish evaluated before bearish, the declared tie-break of §3.3;
5. sweeps (§3.5), which deliberately leave ``active_sh``/``active_sl`` standing;
6. reclaims, so a fresh sweep on the same side replaces the pending one rather
   than resolving it on its own candle (§3.5);
7. create the fair-value gap ending at this row (§3.7);
8. emit the row.

Steps 4-6 run in the order §3 defines them, which decides the two cases the spec
leaves implicit: a break cancels a pending reclaim before that reclaim can fire
on the same candle, and a level newly confirmed at step 1 is available to be
broken or swept immediately.

**ATR comes from :mod:`chimera.features`.** ``_true_range``/``_wilder`` are
imported rather than reimplemented so that the denominator scaling these
features against is byte-identical to the one ``atr_norm`` uses in the OHLCV14
set. A second implementation that agreed to within a rounding error would make
every SMC/OHLCV14 comparison in P2b depend on which of the two produced it.

**No column is ever NaN, on any row, including row 0.**
``nn.data_pipeline.build_dataset`` drops rows with a NaN feature, so a NaN here
would evaluate the two information sets on different samples and confound P2b
with the sample universe. Where state does not exist yet the value is the
declared default and the paired availability flag carries the "not yet" — the
flags exist precisely because ``0.0`` is a legal distance.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from math import log1p

import numpy as np
import pandas as pd

from chimera.features import _true_range, _wilder
from nn.data_pipeline import _contiguous_segment_ids

SMC_SPEC_VERSION: str = "smc_v1"

#: Family name -> its columns, the six families of §4 in order. The P2b ablation
#: drops one family at a time, so the grouping is part of the contract too.
SMC_FEATURE_FAMILIES: dict[str, tuple[str, ...]] = {
    "structure": (
        "smc_structure_valid",
        "smc_dist_swing_high_atr",
        "smc_dist_swing_low_atr",
        "smc_age_swing_high",
        "smc_age_swing_low",
        "smc_structure_direction",
        "smc_range_width_atr",
        "smc_range_position",
    ),
    "liquidity": (
        "smc_equal_high",
        "smc_equal_low",
        "smc_eqh_active",
        "smc_eql_active",
        "smc_eqh_dist_atr",
        "smc_eql_dist_atr",
    ),
    "breaks": (
        "smc_bos_bull",
        "smc_bos_bear",
        "smc_choch_bull",
        "smc_choch_bear",
        "smc_break_magnitude_atr",
        "smc_bars_since_break",
    ),
    "sweeps": (
        "smc_sweep_high",
        "smc_sweep_low",
        "smc_sweep_high_reclaim",
        "smc_sweep_low_reclaim",
        "smc_bars_since_sweep_high",
        "smc_bars_since_sweep_low",
    ),
    "displacement": (
        "smc_body_signed_atr",
        "smc_body_abs_atr",
        "smc_body_ratio",
        "smc_displacement_bull",
        "smc_displacement_bear",
    ),
    "fvg": (
        "smc_fvg_bull",
        "smc_fvg_bear",
        "smc_fvg_bull_size_atr",
        "smc_fvg_bear_size_atr",
        "smc_bull_fvg_active",
        "smc_bear_fvg_active",
        "smc_bull_fvg_dist_atr",
        "smc_bear_fvg_dist_atr",
    ),
}

#: The 39 columns of §4, in that exact order. Never reordered: the feature-spec
#: hash and every persisted prediction record depend on it.
SMC_FEATURE_NAMES: tuple[str, ...] = tuple(
    name for family in SMC_FEATURE_FAMILIES.values() for name in family
)

_REQUIRED_COLUMNS = ("open", "high", "low", "close")


@dataclass(frozen=True)
class SMCSpec:
    """The predeclared constants of §1.2.

    Frozen because they were fixed before any P2b outer-validation result was
    observed and tuning them against one is forbidden; a mutable spec would make
    "the constants were not searched" a claim about discipline rather than about
    the object recorded in the artifact.
    """

    atr_period: int = 14
    pivot_left: int = 3
    pivot_right: int = 3
    break_atr: float = 0.10
    sweep_atr: float = 0.05
    equal_atr: float = 0.25
    displacement_body_atr: float = 1.00
    displacement_body_ratio: float = 0.60
    fvg_min_atr: float = 0.05

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    def spec_hash(self) -> str:
        """sha256 over canonical JSON of version + constants + feature names."""
        payload = json.dumps(
            {
                "version": SMC_SPEC_VERSION,
                "constants": self.to_dict(),
                "feature_names": list(SMC_FEATURE_NAMES),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()


def smc_feature_columns() -> list[str]:
    """The feature names, in the order :func:`compute_smc_features` emits them."""
    return list(SMC_FEATURE_NAMES)


def _raw_pivots(
    high: np.ndarray, low: np.ndarray, left: int, right: int
) -> tuple[np.ndarray, np.ndarray]:
    """The geometric predicates of §3.1, before any confirmation delay.

    Strict on the left and non-strict on the right, which is the tie-break: on a
    plateau of equal highs exactly one candle — the earliest — is the pivot.
    Vectorised because the predicate is local; the confirmation delay that makes
    it usable is applied by the forward pass, never here.
    """
    n = len(high)
    is_high = np.zeros(n, dtype=bool)
    is_low = np.zeros(n, dtype=bool)
    if n <= left + right:
        return is_high, is_low

    end = n - right
    candidate_high = np.ones(end - left, dtype=bool)
    candidate_low = np.ones(end - left, dtype=bool)
    for k in range(1, left + 1):
        candidate_high &= high[left:end] > high[left - k : end - k]
        candidate_low &= low[left:end] < low[left - k : end - k]
    for k in range(1, right + 1):
        candidate_high &= high[left:end] >= high[left + k : end + k]
        candidate_low &= low[left:end] <= low[left + k : end + k]
    is_high[left:end] = candidate_high
    is_low[left:end] = candidate_low
    return is_high, is_low


def compute_smc_features(df: pd.DataFrame, spec: SMCSpec | None = None) -> pd.DataFrame:
    """SMC features for ONE contiguous segment.

    ``df`` needs ``open``/``high``/``low``/``close`` (``volume`` is accepted and
    unused). The result has ``df``'s index and exactly ``SMC_FEATURE_NAMES`` as
    columns, all float64, and contains no NaN and no infinity.
    """
    spec = spec or SMCSpec()
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"OHLC frame is missing columns: {missing}")

    open_ = df["open"].astype("float64")
    high = df["high"].astype("float64")
    low = df["low"].astype("float64")
    close = df["close"].astype("float64")

    atr = _wilder(
        _true_range(pd.DataFrame({"high": high, "low": low, "close": close})),
        spec.atr_period,
    )
    o = open_.to_numpy()
    h = high.to_numpy()
    lo_ = low.to_numpy()
    c = close.to_numpy()
    # A flat market drives ATR to zero; the floor keeps every division finite
    # without pretending the range was larger than it was.
    atr_den = np.maximum(atr.to_numpy(), 1e-12 * c)
    n = len(df)

    # §3.6 is purely local to the candle, so it is computed in one shot.
    body = c - o
    hl = h - lo_
    body_ratio = np.zeros(n, dtype="float64")
    np.divide(np.abs(body), hl, out=body_ratio, where=hl > 0.0)
    body_signed_atr = body / atr_den
    displaces = (np.abs(body) >= spec.displacement_body_atr * atr_den) & (
        body_ratio >= spec.displacement_body_ratio
    )
    displacement_bull = displaces & (body > 0.0)
    displacement_bear = displaces & (body < 0.0)

    is_swing_high, is_swing_low = _raw_pivots(h, lo_, spec.pivot_left, spec.pivot_right)

    last_sh = last_sl = 0.0
    last_sh_i = last_sl_i = -1
    active_sh: float | None = None
    active_sl: float | None = None
    direction = 0
    eqh_levels: list[float] = []
    eql_levels: list[float] = []
    bull_gaps: list[tuple[float, float]] = []
    bear_gaps: list[tuple[float, float]] = []
    last_break_i = -1
    last_sweep_high_i = last_sweep_low_i = -1
    pending_high_sweep = -1
    pending_low_sweep = -1

    out = np.zeros((n, len(SMC_FEATURE_NAMES)), dtype="float64")

    for t in range(n):
        den = atr_den[t]
        c_t = c[t]
        equal_high = equal_low = 0.0
        bos_bull = bos_bear = choch_bull = choch_bear = 0.0
        break_magnitude = 0.0
        sweep_high = sweep_low = 0.0
        sweep_high_reclaim = sweep_low_reclaim = 0.0
        fvg_bull = fvg_bear = 0.0
        fvg_bull_size = fvg_bear_size = 0.0

        # 1. absorb the pivot confirmed by this candle (§2, §3.2, §3.4)
        pivot = t - spec.pivot_right
        if pivot >= 0:
            if is_swing_high[pivot]:
                price = h[pivot]
                if last_sh_i >= 0 and abs(price - last_sh) <= spec.equal_atr * den:
                    equal_high = 1.0
                    eqh_levels.append(max(price, last_sh))
                last_sh, last_sh_i = price, pivot
                active_sh = price
            if is_swing_low[pivot]:
                price = lo_[pivot]
                if last_sl_i >= 0 and abs(price - last_sl) <= spec.equal_atr * den:
                    equal_low = 1.0
                    eql_levels.append(min(price, last_sl))
                last_sl, last_sl_i = price, pivot
                active_sl = price

        # 2. liquidity taken by this candle is gone before the row reports it
        if eqh_levels:
            eqh_levels = [level for level in eqh_levels if h[t] <= level]
        if eql_levels:
            eql_levels = [level for level in eql_levels if lo_[t] >= level]

        # 3. gaps this candle filled, likewise (§3.7)
        if bull_gaps:
            bull_gaps = [gap for gap in bull_gaps if lo_[t] > gap[0]]
        if bear_gaps:
            bear_gaps = [gap for gap in bear_gaps if h[t] < gap[1]]

        # 4. breaks on the close, bullish first (§3.3)
        if active_sh is not None and c_t > active_sh + spec.break_atr * den:
            if direction >= 0:
                bos_bull = 1.0
            else:
                choch_bull = 1.0
            break_magnitude = (c_t - active_sh) / den
            direction = 1
            active_sh = None
            last_break_i = t
            pending_high_sweep = -1
        if active_sl is not None and c_t < active_sl - spec.break_atr * den:
            if direction <= 0:
                bos_bear = 1.0
            else:
                choch_bear = 1.0
            break_magnitude = (c_t - active_sl) / den
            direction = -1
            active_sl = None
            last_break_i = t
            pending_low_sweep = -1

        # 5. sweeps, which leave the probed level standing (§3.5)
        if (
            active_sh is not None
            and h[t] > active_sh + spec.sweep_atr * den
            and c_t <= active_sh
        ):
            sweep_high = 1.0
            last_sweep_high_i = t
            pending_high_sweep = t
        if (
            active_sl is not None
            and lo_[t] < active_sl - spec.sweep_atr * den
            and c_t >= active_sl
        ):
            sweep_low = 1.0
            last_sweep_low_i = t
            pending_low_sweep = t

        # 6. reclaims: the close gives back the entire sweep candle (§3.5)
        if (
            pending_high_sweep >= 0
            and pending_high_sweep < t
            and c_t < lo_[pending_high_sweep]
        ):
            sweep_high_reclaim = 1.0
            pending_high_sweep = -1
        if pending_low_sweep >= 0 and pending_low_sweep < t and c_t > h[pending_low_sweep]:
            sweep_low_reclaim = 1.0
            pending_low_sweep = -1

        # 7. the three-candle imbalance ending here (§3.7)
        if t >= 2:
            gap_up = lo_[t] - h[t - 2]
            if gap_up > 0.0 and gap_up >= spec.fvg_min_atr * den:
                fvg_bull = 1.0
                fvg_bull_size = gap_up / den
                bull_gaps.append((h[t - 2], lo_[t]))
            gap_down = lo_[t - 2] - h[t]
            if gap_down > 0.0 and gap_down >= spec.fvg_min_atr * den:
                fvg_bear = 1.0
                fvg_bear_size = gap_down / den
                bear_gaps.append((h[t], lo_[t - 2]))

        # 8. emit
        has_sh = last_sh_i >= 0
        has_sl = last_sl_i >= 0
        width = last_sh - last_sl if has_sh and has_sl else 0.0
        nearest_eqh = min((level for level in eqh_levels if level >= c_t), default=None)
        nearest_eql = max((level for level in eql_levels if level <= c_t), default=None)
        nearest_bull_gap = max((gap[1] for gap in bull_gaps if gap[1] <= c_t), default=None)
        nearest_bear_gap = min((gap[0] for gap in bear_gaps if gap[0] >= c_t), default=None)

        out[t] = (
            1.0 if has_sh and has_sl else 0.0,
            (last_sh - c_t) / den if has_sh else 0.0,
            (c_t - last_sl) / den if has_sl else 0.0,
            log1p(t - last_sh_i) if has_sh else 0.0,
            log1p(t - last_sl_i) if has_sl else 0.0,
            float(direction),
            width / den if has_sh and has_sl else 0.0,
            min(max((c_t - last_sl) / width, -2.0), 3.0) if width > 0.0 else 0.5,
            equal_high,
            equal_low,
            1.0 if nearest_eqh is not None else 0.0,
            1.0 if nearest_eql is not None else 0.0,
            (nearest_eqh - c_t) / den if nearest_eqh is not None else 0.0,
            (c_t - nearest_eql) / den if nearest_eql is not None else 0.0,
            bos_bull,
            bos_bear,
            choch_bull,
            choch_bear,
            break_magnitude,
            log1p(t - last_break_i) if last_break_i >= 0 else 0.0,
            sweep_high,
            sweep_low,
            sweep_high_reclaim,
            sweep_low_reclaim,
            log1p(t - last_sweep_high_i) if last_sweep_high_i >= 0 else 0.0,
            log1p(t - last_sweep_low_i) if last_sweep_low_i >= 0 else 0.0,
            body_signed_atr[t],
            abs(body_signed_atr[t]),
            body_ratio[t],
            float(displacement_bull[t]),
            float(displacement_bear[t]),
            fvg_bull,
            fvg_bear,
            fvg_bull_size,
            fvg_bear_size,
            1.0 if nearest_bull_gap is not None else 0.0,
            1.0 if nearest_bear_gap is not None else 0.0,
            (c_t - nearest_bull_gap) / den if nearest_bull_gap is not None else 0.0,
            (nearest_bear_gap - c_t) / den if nearest_bear_gap is not None else 0.0,
        )

    return pd.DataFrame(out, index=df.index, columns=list(SMC_FEATURE_NAMES))


def compute_smc_features_segmented(
    candles: pd.DataFrame, timeframe: str, spec: SMCSpec | None = None
) -> pd.DataFrame:
    """Split ``candles`` on market-data gaps, compute per segment, concatenate.

    Segments come from :func:`nn.data_pipeline._contiguous_segment_ids`, so the
    boundaries are the ones the processed dataset already used. State never
    crosses a boundary.
    """
    segment_ids = _contiguous_segment_ids(candles["date"], timeframe)
    blocks = [
        compute_smc_features(candles[segment_ids == segment_id], spec)
        for segment_id in segment_ids.unique()
    ]
    return pd.concat(blocks)
