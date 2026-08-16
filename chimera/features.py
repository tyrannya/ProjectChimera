"""Deterministic, causal feature engineering.

This module is the single source of truth for what a "feature vector" is. The
training pipeline and the Freqtrade strategy both call :func:`compute_features`,
so a model can never be served a differently-shaped or differently-ordered
input than it was trained on.

Two properties are enforced by tests in ``tests/test_features.py``:

*causality*
    the value of any feature at row ``t`` depends only on rows ``<= t``. Every
    primitive used here (``shift``, ``rolling``, ``ewm``, ``pct_change``) is
    backward-looking; nothing centred or forward-filled from the future is
    allowed.

*determinism*
    the same input frame always produces byte-identical output, and the column
    order is fixed by :func:`feature_columns`.

All features are scale-free (ratios, returns, normalised ranges) rather than
raw prices. A model trained on absolute BTC price levels cannot generalise to a
price range it has never seen; ratios keep the input distribution stationary.

Implemented with plain pandas rather than ``ta``/``talib`` on purpose: the
strategy container and the training container then run byte-identical code with
no shared native dependency.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class FeatureSpec:
    """Window lengths for the feature set.

    Persisted alongside the model so inference can rebuild the exact same
    columns. Defaults are deliberately conservative and few; adding indicators
    without a reason costs warm-up rows and adds correlated noise.
    """

    ema_fast: int = 12
    ema_slow: int = 26
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    atr_period: int = 14
    vol_window: int = 24
    volume_window: int = 24

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FeatureSpec":
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: int(v) for k, v in data.items() if k in fields})

    @property
    def warmup(self) -> int:
        """Rows to discard at the start of a series before features are valid.

        Wilder-smoothed indicators (RSI, ATR) and EMAs never become exactly
        NaN-free-and-correct at a fixed row, they converge; three time
        constants is the usual rule of thumb for the transient to decay below
        5%. Rolling windows are exact after ``window`` rows.
        """
        return max(
            3 * self.ema_slow,
            3 * self.rsi_period,
            3 * self.atr_period,
            self.macd_slow + self.macd_signal,
            self.vol_window,
            self.volume_window,
        )


#: Fixed column order. Never reorder this list: model metadata records it and
#: inference rejects any request whose width disagrees.
_FEATURE_NAMES = (
    "ret_1",
    "log_ret_1",
    "ret_close_open",
    "hl_range",
    "ema_fast_ratio",
    "ema_slow_ratio",
    "ema_cross",
    "rsi_centered",
    "macd_norm",
    "macd_hist_norm",
    "atr_norm",
    "realized_vol",
    "volume_change",
    "volume_z",
)


def feature_columns() -> list[str]:
    """The feature names, in the order :func:`compute_features` emits them."""
    return list(_FEATURE_NAMES)


def n_features() -> int:
    return len(_FEATURE_NAMES)


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _wilder(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing, the recursive average used by RSI and ATR."""
    return series.ewm(alpha=1.0 / period, adjust=False).mean()


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = _wilder(gain, period)
    avg_loss = _wilder(loss, period)
    # avg_loss == 0 means an unbroken run of gains -> RSI 100. Guard the divide
    # rather than letting it produce inf.
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.where(avg_loss != 0.0, 100.0).where(avg_gain != 0.0, 0.0)


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def compute_features(df: pd.DataFrame, spec: FeatureSpec | None = None) -> pd.DataFrame:
    """Build the feature frame for an OHLCV frame.

    The result has the same index as ``df`` and the columns of
    :func:`feature_columns`. Early rows contain NaN while indicators warm up;
    callers decide whether to drop them (training) or slice past them
    (inference). Nothing is filled in, because silently filling a warm-up NaN
    with zero is indistinguishable from a real zero-return candle.
    """
    spec = spec or FeatureSpec()
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"OHLCV frame is missing columns: {missing}")

    close = df["close"].astype("float64")
    high = df["high"].astype("float64")
    low = df["low"].astype("float64")
    open_ = df["open"].astype("float64")
    volume = df["volume"].astype("float64")

    out = pd.DataFrame(index=df.index)

    out["ret_1"] = close.pct_change()
    out["log_ret_1"] = np.log(close / close.shift(1))
    out["ret_close_open"] = close / open_ - 1.0
    out["hl_range"] = (high - low) / close

    ema_fast = _ema(close, spec.ema_fast)
    ema_slow = _ema(close, spec.ema_slow)
    out["ema_fast_ratio"] = close / ema_fast - 1.0
    out["ema_slow_ratio"] = close / ema_slow - 1.0
    out["ema_cross"] = ema_fast / ema_slow - 1.0

    out["rsi_centered"] = _rsi(close, spec.rsi_period) / 100.0 - 0.5

    macd = _ema(close, spec.macd_fast) - _ema(close, spec.macd_slow)
    macd_signal = _ema(macd, spec.macd_signal)
    out["macd_norm"] = macd / close
    out["macd_hist_norm"] = (macd - macd_signal) / close

    out["atr_norm"] = _wilder(_true_range(df), spec.atr_period) / close

    out["realized_vol"] = out["log_ret_1"].rolling(spec.vol_window).std()

    out["volume_change"] = volume.pct_change()
    vol_mean = volume.rolling(spec.volume_window).mean()
    vol_std = volume.rolling(spec.volume_window).std()
    out["volume_z"] = (volume - vol_mean) / vol_std.replace(0.0, np.nan)

    # Division by a zero price/volume yields +-inf, which would poison the
    # scaler and the model. Treat it as missing.
    out = out.replace([np.inf, -np.inf], np.nan)
    return out[list(_FEATURE_NAMES)]
