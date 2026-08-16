"""EXPERIMENTAL — DISABLED. Requires order-book data this repository does not have.

Do not enable this strategy. It is kept, unarmed, because the *idea* is sound
and someone may want to finish it; it is disabled because it cannot work as
written and pretending otherwise is worse than deleting it.

Why it is disabled
------------------
Both entry conditions require ``orderbook_delta``, a measure of bid/ask volume
imbalance. Nothing in ProjectChimera produces it. The previous version papered
over that with::

    if "orderbook_delta" not in dataframe.columns:
        dataframe["orderbook_delta"] = 0

which makes every entry condition (`> 0` / `< 0`) permanently false. The
strategy could never open a trade, and the constant column made that look like
a working strategy that simply had no signals.

Worse, order-book depth is not available historically from any exchange this
repository uses. A backtest of this strategy is not merely unimplemented, it is
not *possible* with the available data — any implementation would be
backtesting synthetic numbers.

To finish it you would need to:

1. add a live order-book feed (``ccxt.watch_order_book`` or an exchange
   websocket) writing depth snapshots to storage;
2. accept that backtests can only cover the period since you started recording;
3. re-derive the entry thresholds on that recorded data;
4. remove :attr:`DISABLED` and the guard in ``populate_entry_trend``.
"""

from __future__ import annotations

import logging

import pandas as pd

from strategies.common.risk_manager import RiskAwareStrategy

logger = logging.getLogger(__name__)


class ScalpFutures(RiskAwareStrategy):
    """Order-book imbalance scalper. Emits no entries until real depth data exists."""

    #: Guard flag. While True, ``populate_entry_trend`` returns no entries.
    DISABLED = True

    timeframe = "1m"
    minimal_roi = {"0": 0.01}
    stoploss = -0.01
    startup_candle_count: int = 60
    can_short = True

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        close = dataframe["close"]
        dataframe["ema_fast"] = close.ewm(span=9, adjust=False).mean()
        dataframe["ema_slow"] = close.ewm(span=21, adjust=False).mean()
        # Deliberately NOT defaulted to zero. Its absence must stay visible.
        if "orderbook_delta" not in dataframe.columns:
            dataframe["orderbook_delta"] = float("nan")
        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0

        if self.DISABLED or dataframe["orderbook_delta"].isna().all():
            logger.warning(
                "ScalpFutures is disabled: it needs an order-book feed that this "
                "deployment does not provide. No entries will be generated."
            )
            return dataframe

        delta = dataframe["orderbook_delta"]
        trend_up = dataframe["ema_fast"] > dataframe["ema_slow"]
        dataframe.loc[trend_up & (delta > 0), "enter_long"] = 1
        dataframe.loc[~trend_up & (delta < 0), "enter_short"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        if self.DISABLED:
            return dataframe
        trend_up = dataframe["ema_fast"] > dataframe["ema_slow"]
        dataframe.loc[~trend_up, "exit_long"] = 1
        dataframe.loc[trend_up, "exit_short"] = 1
        return dataframe
