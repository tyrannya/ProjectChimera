import types
import sys

# Stub freqtrade exceptions
freqtrade_exceptions = types.ModuleType("freqtrade.exceptions")
class TemporaryStopException(Exception):
    pass
freqtrade_exceptions.TemporaryStopException = TemporaryStopException

freqtrade_module = types.ModuleType("freqtrade")
freqtrade_module.exceptions = freqtrade_exceptions
sys.modules.setdefault("freqtrade", freqtrade_module)
sys.modules.setdefault("freqtrade.exceptions", freqtrade_exceptions)

from strategies.common.risk_manager import CommonRiskManager


def test_drawdown_triggered():
    rm = CommonRiskManager()
    equity = [100, 91]
    try:
        rm.check_drawdown(equity)
        raised = False
    except TemporaryStopException:
        raised = True
    assert raised, "Expected drawdown trigger"


def test_drawdown_ok():
    rm = CommonRiskManager()
    equity = [100, 94]
    rm.check_drawdown(equity)  # should not raise


def test_rate_limit_trigger():
    rm = CommonRiskManager()
    for t in range(5):
        if t < 4:
            rm.log_http_429(t)
        else:
            try:
                rm.log_http_429(t)
                raised = False
            except TemporaryStopException:
                raised = True
    assert raised, "Expected rate limit trigger"


def test_rate_limit_window():
    rm = CommonRiskManager()
    for t in range(4):
        rm.log_http_429(t)
    rm.log_http_429(61)
    assert len(rm.http_429_errors) == 4
