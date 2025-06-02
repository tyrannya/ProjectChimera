from freqtrade.strategy import IStrategy, merge_informative_pair
import pandas as pd
import talib.abstract as ta

class SwingSpot(IStrategy):
    timeframe = "4h"
    minimal_roi = {"0": 0.03}
    stoploss = -0.05

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['ema21'] = ta.EMA(dataframe, timeperiod=21)
        dataframe['ema55'] = ta.EMA(dataframe, timeperiod=55)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['enter_long'] = (
            (dataframe['ema21'] > dataframe['ema55']) &
            (dataframe['rsi'] < 30)
        )
        dataframe['enter_short'] = (
            (dataframe['ema21'] < dataframe['ema55']) &
            (dataframe['rsi'] > 70)
        )
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['exit_long'] = (dataframe['rsi'] > 50)
        dataframe['exit_short'] = (dataframe['rsi'] < 50)
        return dataframe
