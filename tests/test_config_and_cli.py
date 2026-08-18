"""Configuration validity and the launcher's safety behaviour.

The configs are checked against Freqtrade's *own* JSON schema, not a
hand-written idea of what a config should look like. The previous
``conf/base.json`` used keys (``symbols``, ``initial_state: "paused"``) that no
version of Freqtrade has ever accepted, and nothing caught it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from freqtrade.configuration.config_validation import validate_config_schema

from chimera.safety import LIVE_TRADING_ACK, LIVE_TRADING_ENV_VAR, LiveTradingBlocked
from strategies.nn_predictor_strategy import NNPredictorStrategy
from tools.run_bot import deep_merge, load_config, main, resolve_placeholders

CONF_DIR = Path(__file__).resolve().parents[1] / "conf"
EXCHANGES = ["binance", "bybit", "okx"]


@pytest.fixture(autouse=True)
def _no_ack(monkeypatch):
    monkeypatch.delenv(LIVE_TRADING_ENV_VAR, raising=False)


def validated(exchange: str, mode: str) -> dict:
    """Merge a config the way run_bot does and run Freqtrade's schema on it."""
    config = load_config(exchange, mode)
    config["strategy"] = "NNPredictorStrategy"
    config = resolve_placeholders(config, [])
    config["runmode"] = "dry_run"
    # StrategyResolver copies these off the strategy onto the config before
    # Freqtrade validates, so the configs themselves must not declare them.
    config["stoploss"] = NNPredictorStrategy.stoploss
    config["minimal_roi"] = NNPredictorStrategy.minimal_roi
    config["timeframe"] = NNPredictorStrategy.timeframe
    validate_config_schema(config)
    return config


# --- config files ---------------------------------------------------------
@pytest.mark.parametrize("path", sorted(CONF_DIR.glob("*.json")))
def test_every_config_is_valid_json(path):
    json.loads(path.read_text())


@pytest.mark.parametrize("exchange", EXCHANGES)
@pytest.mark.parametrize("mode", ["test", "live"])
def test_merged_configs_satisfy_freqtrades_schema(exchange, mode):
    validated(exchange, mode)


@pytest.mark.parametrize("exchange", EXCHANGES)
def test_test_configs_are_dry_run(exchange):
    assert validated(exchange, "test")["dry_run"] is True


@pytest.mark.parametrize("exchange", EXCHANGES)
def test_live_profiles_declare_intent_without_being_live(exchange):
    """The live profile must be recognisable to the launcher yet inert on disk.

    Previously these files carried dry_run=false, which meant the whole safety
    system depended on the user entering through tools/run_bot.py.
    """
    config = load_config(exchange, "live")
    assert config["dry_run"] is True
    assert config["chimera_live_intent"] is True


def test_no_config_contains_a_literal_credential():
    """Only ``${VAR}`` placeholders, never a value."""
    for path in CONF_DIR.glob("*.json"):
        exchange = json.loads(path.read_text()).get("exchange", {})
        for field in ("key", "secret", "password", "uid"):
            value = exchange.get(field)
            if value:
                assert value.startswith("${") and value.endswith(
                    "}"
                ), f"{path.name} has a literal {field}"


def test_base_config_does_not_enable_the_rest_api():
    """The REST API can start, stop and force-enter trades."""
    base = json.loads((CONF_DIR / "base.json").read_text())
    assert base.get("api_server", {}).get("enabled", False) is False


def test_base_config_does_not_force_entry():
    assert json.loads((CONF_DIR / "base.json").read_text())["force_entry_enable"] is False


def test_base_config_states_a_fee():
    """Backtests without an explicit fee silently assume the exchange default."""
    assert json.loads((CONF_DIR / "base.json").read_text())["fee"] > 0


def test_base_config_carries_risk_limits():
    risk = json.loads((CONF_DIR / "base.json").read_text())["risk"]
    for key in ("max_drawdown_pct", "risk_per_trade_pct", "max_open_positions"):
        assert key in risk


# --- merging and placeholders -----------------------------------------------
def test_deep_merge_is_recursive_and_non_destructive():
    base = {"a": 1, "exchange": {"name": "x", "pair_whitelist": ["A"]}}
    override = {"exchange": {"name": "y"}}
    merged = deep_merge(base, override)
    assert merged["exchange"] == {"name": "y", "pair_whitelist": ["A"]}
    assert base["exchange"]["name"] == "x", "inputs must not be mutated"


def test_placeholders_resolve_from_the_environment(monkeypatch):
    monkeypatch.setenv("CHIMERA_TEST_KEY", "resolved-value")
    resolved = resolve_placeholders({"exchange": {"key": "${CHIMERA_TEST_KEY}"}}, [])
    assert resolved["exchange"]["key"] == "resolved-value"


def test_unset_placeholders_are_reported_not_passed_through(monkeypatch):
    """Freqtrade does not expand ``${...}``; passing the literal string to an
    exchange as an API key is what the old configs actually did."""
    monkeypatch.delenv("CHIMERA_ABSENT", raising=False)
    missing: list[str] = []
    resolved = resolve_placeholders({"key": "${CHIMERA_ABSENT}"}, missing)
    assert missing == ["CHIMERA_ABSENT"]
    assert resolved["key"] == ""


def test_non_placeholder_strings_are_untouched():
    assert resolve_placeholders({"name": "binance", "x": "$NOT{}"}, [])["name"] == "binance"


# --- the launcher ---------------------------------------------------------------
def test_launcher_refuses_live_without_the_acknowledgement(tmp_path):
    config_out = tmp_path / "merged.json"
    exit_code = main(
        [
            "--exchange",
            "binance",
            "--mode",
            "live",
            "--dry-run-only-check",
            "--config-out",
            str(config_out),
        ]
    )
    assert exit_code == 2
    assert not config_out.exists(), "a blocked launch must not leave a live config behind"


def test_launcher_allows_dry_run_without_credentials(tmp_path, capsys):
    config_out = tmp_path / "merged.json"
    exit_code = main(
        [
            "--exchange",
            "binance",
            "--mode",
            "test",
            "--dry-run-only-check",
            "--config-out",
            str(config_out),
        ]
    )
    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["dry_run"] is True
    assert json.loads(config_out.read_text())["dry_run"] is True


def test_launcher_writes_the_merged_config_privately(tmp_path):
    """It contains API keys; it must not be world-readable."""
    config_out = tmp_path / "merged.json"
    main(
        [
            "--exchange",
            "binance",
            "--mode",
            "test",
            "--dry-run-only-check",
            "--config-out",
            str(config_out),
        ]
    )
    assert config_out.stat().st_mode & 0o077 == 0


def test_launcher_goes_live_only_with_both_the_ack_and_a_live_config(monkeypatch, tmp_path):
    monkeypatch.setenv(LIVE_TRADING_ENV_VAR, LIVE_TRADING_ACK)
    monkeypatch.setenv("BINANCE_KEY", "k")
    monkeypatch.setenv("BINANCE_SECRET", "s")
    config_out = tmp_path / "merged.json"
    assert (
        main(
            [
                "--exchange",
                "binance",
                "--mode",
                "live",
                "--dry-run-only-check",
                "--config-out",
                str(config_out),
            ]
        )
        == 0
    )
    assert json.loads(config_out.read_text())["dry_run"] is False


def test_live_without_credentials_is_refused_even_with_the_ack(monkeypatch, tmp_path):
    monkeypatch.setenv(LIVE_TRADING_ENV_VAR, LIVE_TRADING_ACK)
    monkeypatch.delenv("BINANCE_KEY", raising=False)
    monkeypatch.delenv("BINANCE_SECRET", raising=False)
    assert (
        main(
            [
                "--exchange",
                "binance",
                "--mode",
                "live",
                "--dry-run-only-check",
                "--config-out",
                str(tmp_path / "m.json"),
            ]
        )
        == 2
    )


def test_the_ack_does_not_make_a_test_config_live(monkeypatch, tmp_path):
    monkeypatch.setenv(LIVE_TRADING_ENV_VAR, LIVE_TRADING_ACK)
    config_out = tmp_path / "merged.json"
    main(
        [
            "--exchange",
            "binance",
            "--mode",
            "test",
            "--dry-run-only-check",
            "--config-out",
            str(config_out),
        ]
    )
    assert json.loads(config_out.read_text())["dry_run"] is True


def test_load_config_rejects_an_unknown_pair_of_names():
    with pytest.raises(SystemExit, match="config not found"):
        load_config("binance", "nonexistent")


def test_enforce_dry_run_is_what_blocks_the_launcher(monkeypatch):
    """Documents where the gate lives, so a refactor cannot quietly drop it."""
    from chimera.safety import enforce_dry_run

    with pytest.raises(LiveTradingBlocked):
        enforce_dry_run(load_config("binance", "live"), request_live=True)


def test_a_hand_edited_live_config_still_needs_the_acknowledgement():
    """Defence in depth: no committed file sets dry_run=false, but if one ever
    reaches the gate it must not sail through on the config alone."""
    from chimera.safety import enforce_dry_run

    with pytest.raises(LiveTradingBlocked):
        enforce_dry_run({"dry_run": False})


# --- repository hygiene ---------------------------------------------------------
def test_env_file_is_not_committed():
    repo = Path(__file__).resolve().parents[1]
    assert not (repo / ".env").exists(), ".env must never be in the repository"


def test_env_example_contains_no_values():
    repo = Path(__file__).resolve().parents[1]
    for line in (repo / ".env.example").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        assert value == "", f"{name} in .env.example must have an empty value"


def test_no_merge_conflict_markers_remain():
    repo = Path(__file__).resolve().parents[1]
    patterns = ("<" * 7, "=" * 7, ">" * 7)
    for path in repo.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        if path.suffix not in {
            ".py",
            ".md",
            ".json",
            ".yml",
            ".yaml",
            ".txt",
            ".cfg",
            ".toml",
        }:
            continue
        for number, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
            # Only a line that *starts* with a marker is a real conflict
            # remnant. docs/engineering-audit.md quotes these sequences inline
            # while describing the ones this rebuild removed.
            for pattern in patterns:
                assert not line.startswith(pattern), f"merge marker at {path}:{number}"
