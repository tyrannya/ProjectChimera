import types
import sys
import pandas as pd

# Stub talib module before importing strategy
talib_module = types.ModuleType("talib")
talib_abstract = types.ModuleType("talib.abstract")
talib_module.abstract = talib_abstract
sys.modules.setdefault("talib", talib_module)
sys.modules.setdefault("talib.abstract", talib_abstract)

# Stub freqtrade modules
freqtrade_exceptions = types.ModuleType("freqtrade.exceptions")
class TemporaryStopException(Exception):
    pass
freqtrade_exceptions.TemporaryStopException = TemporaryStopException

freqtrade_strategy = types.ModuleType("freqtrade.strategy")
class IStrategy:
    pass

def merge_informative_pair(*args, **kwargs):
    return None

freqtrade_strategy.IStrategy = IStrategy
freqtrade_strategy.merge_informative_pair = merge_informative_pair

freqtrade_module = types.ModuleType("freqtrade")
freqtrade_module.exceptions = freqtrade_exceptions
freqtrade_module.strategy = freqtrade_strategy
sys.modules.setdefault("freqtrade", freqtrade_module)
sys.modules.setdefault("freqtrade.exceptions", freqtrade_exceptions)
sys.modules.setdefault("freqtrade.strategy", freqtrade_strategy)

from strategies import swing_spot


def test_populate_indicators(monkeypatch):
    df = pd.DataFrame({'close': [1, 2, 3, 4, 5]})

    def ema(df, timeperiod=21):
        return df['close'].rolling(window=timeperiod, min_periods=1).mean()

    def rsi(df, timeperiod=14):
        return pd.Series([50] * len(df))

    monkeypatch.setattr(swing_spot.ta, 'EMA', ema, raising=False)
    monkeypatch.setattr(swing_spot.ta, 'RSI', rsi, raising=False)

    strat = swing_spot.SwingSpot()
    out = strat.populate_indicators(df.copy(), {})

    assert {'ema21', 'ema55', 'rsi'} <= set(out.columns)
    assert out['ema21'].iloc[-1] == df['close'].rolling(window=21, min_periods=1).mean().iloc[-1]
