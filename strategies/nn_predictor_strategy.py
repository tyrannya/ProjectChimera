from freqtrade.strategy import IStrategy
import pandas as pd
import asyncio
import aiohttp
import talib.abstract as ta

class NNPredictorStrategy(IStrategy):
    timeframe = "4h"
    minimal_roi = {"0": 0.03}
    stoploss = -0.05

    async def get_nn_score(self, df: pd.DataFrame) -> float:
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get("http://nn_infer:3000/predict", timeout=2) as resp:
                    js = await resp.json()
                    return js.get('score', 0)
            except Exception:
                return None

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        score = None
        try:
            score = asyncio.run(self.get_nn_score(dataframe))
        except Exception:
            pass
        if score is not None:
            dataframe['nn_score'] = score
        else:
            # fallback к swing_spot: ema21, ema55, rsi
            dataframe['ema21'] = dataframe['close'].ewm(span=21, adjust=False).mean()
            dataframe['ema55'] = dataframe['close'].ewm(span=55, adjust=False).mean()
            dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
            dataframe['nn_score'] = 0
        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['enter_long'] = (
            (dataframe['nn_score'] > 0.6) |
            (
                (dataframe['ema21'] > dataframe['ema55']) &
                (dataframe.get('rsi', 100) < 30)
            )
        )
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['exit_long'] = (
            (dataframe['nn_score'] < 0.6) |
            (
                (dataframe['ema21'] < dataframe['ema55']) &
                (dataframe.get('rsi', 0) > 50)
            )
        )
        return dataframe
