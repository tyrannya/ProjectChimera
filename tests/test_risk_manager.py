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
=======
import sys
import types
from pathlib import Path
import pytest

# Create a minimal stub for freqtrade.exceptions before importing the module
ft_module = types.ModuleType("freqtrade")
exceptions = types.ModuleType("freqtrade.exceptions")

class TemporaryStopException(Exception):
    pass

exceptions.TemporaryStopException = TemporaryStopException
ft_module.exceptions = exceptions
sys.modules.setdefault("freqtrade", ft_module)
sys.modules.setdefault("freqtrade.exceptions", exceptions)

# Ensure project root is on sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
 main

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
=======
def test_check_drawdown_triggers_exception():
    rm = CommonRiskManager()
    with pytest.raises(TemporaryStopException):
        rm.check_drawdown([100, 90])  # 10% drawdown


def test_log_http_429_triggers_after_five_events():
    rm = CommonRiskManager()
    with pytest.raises(TemporaryStopException):
        for _ in range(5):
            rm.log_http_429(100)


def test_funding_guard_triggers_on_high_cost():
    rm = CommonRiskManager()
    rm.update_funding_and_pnl(funding_cost=40, realized_pnl=100)
    with pytest.raises(TemporaryStopException):
        rm.funding_guard()
 main
