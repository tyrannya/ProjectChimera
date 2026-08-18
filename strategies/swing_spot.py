"""EMA/RSI swing strategy for spot markets.

Long-only. The previous version emitted ``enter_short``/``exit_short`` columns,
but every config in this repository trades spot, where Freqtrade ignores them —
so the short side looked implemented and did nothing. If you want the short
side, run this on a futures config with ``can_short`` and re-test it; do not
assume the columns alone are enough.

Indicators are computed with pandas rather than talib so the strategy imports
in any environment, and so its warm-up behaviour matches
:mod:`chimera.features`.
"""

from __future__ import annotations

import pandas as pd

from strategies.common.risk_manager import RiskAwareStrategy


class SwingSpot(RiskAwareStrategy):
    timeframe = "4h"
    minimal_roi = {"0": 0.03}
    stoploss = -0.05
    startup_candle_count: int = 200
    can_short = False

    ema_fast_period = 21
    ema_slow_period = 55
    rsi_period = 14

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        close = dataframe["close"]
        dataframe["ema_fast"] = close.ewm(span=self.ema_fast_period, adjust=False).mean()
        dataframe["ema_slow"] = close.ewm(span=self.ema_slow_period, adjust=False).mean()

        delta = close.diff()
        gain = delta.clip(lower=0.0).ewm(alpha=1 / self.rsi_period, adjust=False).mean()
        loss = (-delta).clip(lower=0.0).ewm(alpha=1 / self.rsi_period, adjust=False).mean()
        rs = gain / loss.replace(0.0, pd.NA)
        dataframe["rsi"] = (100.0 - 100.0 / (1.0 + rs)).fillna(50.0).astype("float64")
        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe["enter_long"] = 0
        # Uptrend plus a pullback: the fast EMA leads, but RSI has dipped.
        entry = (dataframe["ema_fast"] > dataframe["ema_slow"]) & (dataframe["rsi"] < 40)
        dataframe.loc[entry, "enter_long"] = 1
        dataframe.loc[entry, "enter_tag"] = "swing_pullback"
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe["exit_long"] = 0
        exit_signal = (dataframe["rsi"] > 70) | (dataframe["ema_fast"] < dataframe["ema_slow"])
        dataframe.loc[exit_signal, "exit_long"] = 1
        return dataframe
