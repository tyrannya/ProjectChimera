"""EXPERIMENTAL — DISABLED. Statistical arbitrage that Freqtrade cannot execute.

Do not enable this strategy. Two independent problems, either of which is fatal.

**1. The data does not exist.** The z-score is computed from ``eth_btc`` and
``eth_usdt`` columns. Freqtrade hands ``populate_indicators`` the candles for
*one* pair; those two columns are never present. The previous version fell back
to ``dataframe["zscore"] = 0``, and the entry condition ``abs(zscore) > 2``
could then never be true — a strategy that looked implemented and could not
trade. Pulling the second leg in would mean an informative pair, which is
possible, but see the next point before bothering.

**2. Freqtrade cannot execute a pairs trade.** A spread trade is one position
long ETH/BTC and one short ETH/USDT, opened and closed together, sized against
each other. Freqtrade manages independent single-pair trades with independent
stops and exits. Expressing "these two positions are one trade and must be
unwound together" is not something the engine models. A single-leg
approximation is not this strategy, it is a naked directional bet triggered by
a spread signal — with a completely different risk profile.

Delivering this properly means either an execution layer Freqtrade does not
provide, or a different engine. This repository has one execution engine on
purpose, so the strategy stays disabled and the reason stays written down.
"""

from __future__ import annotations

import logging

import pandas as pd

from strategies.common.risk_manager import RiskAwareStrategy

logger = logging.getLogger(__name__)


class ArbMM(RiskAwareStrategy):
    """Spread/z-score mean reversion. Emits no entries."""

    #: Guard flag. See the module docstring for what it would take to clear it.
    DISABLED = True

    timeframe = "1m"
    minimal_roi = {"0": 0.003}
    stoploss = -0.01
    startup_candle_count: int = 200
    can_short = False

    zscore_window = 100
    entry_zscore = 2.0
    exit_zscore = 0.5

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # No silent zero fallback: without both legs there is no spread, and
        # NaN says so honestly.
        if {"eth_btc", "eth_usdt"} <= set(dataframe.columns):
            spread = dataframe["eth_btc"] - dataframe["eth_usdt"]
            rolling = spread.rolling(self.zscore_window)
            dataframe["zscore"] = (spread - rolling.mean()) / rolling.std()
        else:
            dataframe["zscore"] = float("nan")
        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe["enter_long"] = 0
        logger.warning(
            "ArbMM is disabled: Freqtrade cannot execute a two-leg spread trade, "
            "and the second leg's data is not available. No entries generated."
        )
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe["exit_long"] = 0
        return dataframe
