"""The live-trading gate.

If any test in this file starts failing, assume live trading can be reached by
accident until proven otherwise.
"""

from __future__ import annotations

import pytest

from chimera.safety import (
    LIVE_TRADING_ACK,
    LIVE_TRADING_ENV_VAR,
    LiveTradingBlocked,
    enforce_dry_run,
    is_secret_name,
    live_trading_acknowledged,
    redact,
    require_env,
    resolve_trading_mode,
)


@pytest.fixture(autouse=True)
def _clear_ack(monkeypatch):
    """Every test starts with no acknowledgement in the environment."""
    monkeypatch.delenv(LIVE_TRADING_ENV_VAR, raising=False)


# --- the gate -------------------------------------------------------------
def test_dry_run_config_stays_dry_run():
    assert resolve_trading_mode({"dry_run": True}).dry_run is True


def test_missing_dry_run_key_fails_closed():
    """A truncated or unexpected config must not be read as permission to trade."""
    assert resolve_trading_mode({}).dry_run is True


def test_live_config_without_the_ack_is_blocked():
    with pytest.raises(LiveTradingBlocked, match=LIVE_TRADING_ENV_VAR):
        resolve_trading_mode({"dry_run": False})


def test_live_config_with_the_ack_is_allowed(monkeypatch):
    monkeypatch.setenv(LIVE_TRADING_ENV_VAR, LIVE_TRADING_ACK)
    mode = resolve_trading_mode({"dry_run": False})
    assert mode.live is True and mode.dry_run is False


@pytest.mark.parametrize(
    "value",
    ["", "yes", "true", "1", "I_UNDERSTAND", "i_understand_the_risk", " " + LIVE_TRADING_ACK],
)
def test_only_the_exact_token_unlocks_live_trading(monkeypatch, value):
    monkeypatch.setenv(LIVE_TRADING_ENV_VAR, value)
    assert not live_trading_acknowledged()
    with pytest.raises(LiveTradingBlocked):
        resolve_trading_mode({"dry_run": False})


def test_api_keys_alone_never_enable_live_trading(monkeypatch):
    """Possession of credentials is not consent."""
    monkeypatch.setenv("BINANCE_KEY", "real-looking-key")
    monkeypatch.setenv("BINANCE_SECRET", "real-looking-secret")
    with pytest.raises(LiveTradingBlocked):
        resolve_trading_mode({"dry_run": False, "exchange": {"key": "real-looking-key"}})


def test_ack_alone_does_not_make_a_dry_run_config_live(monkeypatch):
    """Both conditions are required; the ack does not override the config."""
    monkeypatch.setenv(LIVE_TRADING_ENV_VAR, LIVE_TRADING_ACK)
    assert resolve_trading_mode({"dry_run": True}).dry_run is True


def test_enforce_dry_run_rewrites_the_config():
    config = {"dry_run": True, "exchange": {"name": "binance"}}
    assert enforce_dry_run(config)["dry_run"] is True


def test_enforce_dry_run_raises_rather_than_downgrading_silently():
    with pytest.raises(LiveTradingBlocked):
        enforce_dry_run({"dry_run": False})


# --- secrets ---------------------------------------------------------------
@pytest.mark.parametrize(
    "name", ["BINANCE_KEY", "BYBIT_SECRET", "TELEGRAM_BOT_TOKEN", "DB_PASSWORD", "my_apikey"]
)
def test_secret_names_are_recognised(name):
    assert is_secret_name(name)


@pytest.mark.parametrize("name", ["EXCHANGE", "TIMEFRAME", "LOG_LEVEL"])
def test_ordinary_names_are_not_secret(name):
    assert not is_secret_name(name)


def test_redact_hides_secret_values():
    assert redact("BINANCE_KEY", "abcdef123456") == "<set>"
    assert redact("BINANCE_KEY", "") == "<empty>"
    assert redact("BINANCE_KEY", None) == "<unset>"


def test_redact_passes_through_non_secrets():
    assert redact("TIMEFRAME", "1h") == "1h"


# --- environment validation -------------------------------------------------
def test_require_env_returns_present_values(monkeypatch):
    monkeypatch.setenv("CHIMERA_TEST_A", "1")
    monkeypatch.setenv("CHIMERA_TEST_B", "2")
    assert require_env(["CHIMERA_TEST_A", "CHIMERA_TEST_B"]) == {
        "CHIMERA_TEST_A": "1",
        "CHIMERA_TEST_B": "2",
    }


def test_require_env_names_what_is_missing(monkeypatch):
    monkeypatch.delenv("CHIMERA_TEST_MISSING", raising=False)
    with pytest.raises(RuntimeError, match="CHIMERA_TEST_MISSING"):
        require_env(["CHIMERA_TEST_MISSING"])


def test_require_env_error_never_contains_a_secret_value(monkeypatch):
    monkeypatch.setenv("PRESENT_KEY", "super-secret-value")
    monkeypatch.delenv("ABSENT_KEY", raising=False)
    with pytest.raises(RuntimeError) as excinfo:
        require_env(["PRESENT_KEY", "ABSENT_KEY"])
    assert "super-secret-value" not in str(excinfo.value)


def test_empty_string_counts_as_missing(monkeypatch):
    monkeypatch.setenv("CHIMERA_TEST_EMPTY", "")
    with pytest.raises(RuntimeError, match="CHIMERA_TEST_EMPTY"):
        require_env(["CHIMERA_TEST_EMPTY"])
