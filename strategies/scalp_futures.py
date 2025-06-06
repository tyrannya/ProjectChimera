# from freqtrade.strategy import IStrategy, merge_informative_pair # Replaced by NotifyingStrategy
from strategies.common.notifying_strategy import NotifyingStrategy # Added
from freqtrade.strategy import merge_informative_pair # Keep for merge_informative_pair if used directly
import pandas as pd


class ScalpFutures(NotifyingStrategy): # Changed from IStrategy
    timeframe = "1m"
    minimal_roi = {"0": 0.01}
    stoploss = -0.01
    position_adjustment_enable = True

    def populate_indicators(
        self, dataframe: pd.DataFrame, metadata: dict
    ) -> pd.DataFrame:
        dataframe["ema9"] = dataframe["close"].ewm(span=9, adjust=False).mean()
        dataframe["ema21"] = dataframe["close"].ewm(span=21, adjust=False).mean()
        # orderbook_delta — предполагаем, что интеграция через custom indicator
        if "orderbook_delta" not in dataframe.columns:
            dataframe["orderbook_delta"] = 0
        return dataframe

    def populate_entry_trend(
        self, dataframe: pd.DataFrame, metadata: dict
    ) -> pd.DataFrame:
        dataframe["enter_long"] = (dataframe["ema9"] > dataframe["ema21"]) & (
            dataframe["orderbook_delta"] > 0
        )
        dataframe["enter_short"] = (dataframe["ema9"] < dataframe["ema21"]) & (
            dataframe["orderbook_delta"] < 0
        )
        return dataframe

    def populate_exit_trend(
        self, dataframe: pd.DataFrame, metadata: dict
    ) -> pd.DataFrame:
        dataframe["exit_long"] = (dataframe["ema9"] < dataframe["ema21"]) | (
            dataframe["orderbook_delta"] < 0
        )
        dataframe["exit_short"] = (dataframe["ema9"] > dataframe["ema21"]) | (
            dataframe["orderbook_delta"] > 0
        )
        return dataframe
