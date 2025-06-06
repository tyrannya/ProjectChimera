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

from strategies.common.risk_manager import CommonRiskManager


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
