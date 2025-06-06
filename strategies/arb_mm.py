# from freqtrade.strategy import IStrategy # Replaced by NotifyingStrategy
from strategies.common.notifying_strategy import NotifyingStrategy # Added
import pandas as pd
import numpy as np


class ArbMM(NotifyingStrategy): # Changed from IStrategy
    timeframe = "1m"
    minimal_roi = {"0": 0.003}
    stoploss = -0.01

    def populate_indicators(
        self, dataframe: pd.DataFrame, metadata: dict
    ) -> pd.DataFrame:
        # ETH/BTC и ETH/USDT спред, z-score
        if {"eth_btc", "eth_usdt"} <= set(dataframe.columns):
            spread = dataframe["eth_btc"] - dataframe["eth_usdt"]
            mean = spread.rolling(window=100).mean()
            std = spread.rolling(window=100).std()
            dataframe["zscore"] = (spread - mean) / std
        else:
            dataframe["zscore"] = 0
        return dataframe

    def populate_entry_trend(
        self, dataframe: pd.DataFrame, metadata: dict
    ) -> pd.DataFrame:
        dataframe["enter_long"] = dataframe["zscore"].abs() > 2
        return dataframe

    def populate_exit_trend(
        self, dataframe: pd.DataFrame, metadata: dict
    ) -> pd.DataFrame:
        dataframe["exit_long"] = dataframe["zscore"].abs() < 0.5
        return dataframe
