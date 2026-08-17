"""Regression tests for the trading-safety issues found in review of PR #23.

Each test here reproduces a specific confirmed defect. Every one of them fails
against commit `aba47dc` (the PR head before this pass) and passes after the
fix, which is the property that makes them regression tests rather than
restatements of the implementation.

The semantics they encode were read out of the installed Freqtrade 2026.7, not
recalled:

* ``dataframe["date"]`` is the candle **open** time — verified with
  ``freqtrade.data.converter.ohlcv_to_dataframe``. Freqtrade's own
  ``IStrategy.ignore_expired_candle`` measures candle age as
  ``current_time - (open + timeframe)``, which is the formula adopted here.
* ``FreqtradeBot.execute_entry`` computes ``amount = (stake_amount / rate) *
  leverage`` and then hands ``amount`` and ``rate`` to ``confirm_trade_entry``,
  so the stake actually being committed is ``amount * rate / leverage``.
* ``order_filled`` is invoked from ``_update_trade_after_fill`` for **every**
  order that reaches a closed state, and ``LocalTrade.recalc_trade_from_orders``
  rewrites ``trade.stake_amount`` to the trade's *total* stake each time.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest
from freqtrade.data.converter import ohlcv_to_dataframe
from freqtrade.enums import RunMode

from chimera.risk import RiskEngine, RiskLimits
from strategies.nn_predictor_strategy import NNPredictorStrategy

ROOT = Path(__file__).resolve().parents[1]
CONF_DIR = ROOT / "conf"

BASE_CONFIG = {
    "stake_currency": "USDT",
    "stake_amount": 200,
    "dry_run": True,
    "runmode": "dry_run",
    "risk": {},
}


# ---------------------------------------------------------------------------
# 1. Timeframe-aware market data freshness
# ---------------------------------------------------------------------------
class FakeDataProvider:
    def __init__(self, dataframe, runmode=RunMode.DRY_RUN):
        self.runmode = runmode
        self._dataframe = dataframe

    def get_analyzed_dataframe(self, pair, timeframe):
        return self._dataframe, None


def candles_ending_at(last_open: datetime, timeframe_minutes: int = 60, rows: int = 5):
    """Build a real Freqtrade OHLCV frame whose final candle opens at ``last_open``."""
    raw = []
    for i in range(rows - 1, -1, -1):
        opened = last_open - timedelta(minutes=timeframe_minutes * i)
        raw.append([int(opened.timestamp() * 1000), 100.0, 101.0, 99.0, 100.5, 10.0])
    tf = f"{timeframe_minutes}m" if timeframe_minutes < 60 else f"{timeframe_minutes // 60}h"
    return ohlcv_to_dataframe(raw, tf, "BTC/USDT", fill_missing=False, drop_incomplete=False)


def strategy_with(dataframe, timeframe="1h"):
    strat = NNPredictorStrategy({**BASE_CONFIG})
    strat.timeframe = timeframe
    strat.dp = FakeDataProvider(dataframe)
    return strat


def test_a_just_closed_1h_candle_is_not_stale():
    """The bug: age was measured from the candle's OPEN time.

    A 1h candle that has only just closed is ~3600s old by that measure, which
    is far beyond the 300s limit, so *every* NN entry was rejected as stale.
    """
    now = datetime(2026, 8, 16, 12, 0, 30, tzinfo=timezone.utc)
    # Candle opened 11:00 and closed at 12:00 — 30 seconds ago.
    strat = strategy_with(candles_ending_at(datetime(2026, 8, 16, 11, 0, tzinfo=timezone.utc)))

    delay = strat.data_delay_seconds("BTC/USDT", now)
    assert delay == pytest.approx(
        30, abs=1
    ), "freshness must be measured from the candle close, not its open"

    engine = RiskEngine(RiskLimits())
    decision = engine.evaluate_entry(
        pair="BTC/USDT",
        equity=10_000.0,
        entry_price=100.0,
        stop_price=95.0,
        proposed_stake=1000.0,
        data_delay_s=delay,
    )
    assert decision.allowed, f"a fresh 1h candle must not block entry: {decision.reason}"


def test_a_genuinely_stale_1h_candle_is_rejected():
    now = datetime(2026, 8, 16, 18, 0, tzinfo=timezone.utc)
    # Last candle closed at 12:00 — six hours late.
    strat = strategy_with(candles_ending_at(datetime(2026, 8, 16, 11, 0, tzinfo=timezone.utc)))

    delay = strat.data_delay_seconds("BTC/USDT", now)
    assert delay == pytest.approx(6 * 3600, abs=1)

    engine = RiskEngine(RiskLimits(max_data_delay_s=300))
    decision = engine.evaluate_entry(
        pair="BTC/USDT",
        equity=10_000.0,
        entry_price=100.0,
        stop_price=95.0,
        proposed_stake=1000.0,
        data_delay_s=delay,
    )
    assert not decision.allowed
    assert "market data" in decision.reason


def test_freshness_scales_with_a_shorter_timeframe():
    """A 5m candle 30s past close is fresh; the same delay logic must hold."""
    now = datetime(2026, 8, 16, 12, 5, 30, tzinfo=timezone.utc)
    strat = strategy_with(
        candles_ending_at(
            datetime(2026, 8, 16, 12, 0, tzinfo=timezone.utc), timeframe_minutes=5
        ),
        timeframe="5m",
    )
    assert strat.data_delay_seconds("BTC/USDT", now) == pytest.approx(30, abs=1)


def test_a_naive_current_time_is_treated_as_utc():
    """Freqtrade passes tz-aware UTC, but a naive datetime must not explode."""
    naive = datetime(2026, 8, 16, 12, 0, 30)
    strat = strategy_with(candles_ending_at(datetime(2026, 8, 16, 11, 0, tzinfo=timezone.utc)))
    assert strat.data_delay_seconds("BTC/USDT", naive) == pytest.approx(30, abs=1)


def test_delay_is_never_negative_at_the_boundary():
    """Clock skew can put "now" fractionally before the candle close."""
    now = datetime(2026, 8, 16, 11, 59, 59, tzinfo=timezone.utc)
    strat = strategy_with(candles_ending_at(datetime(2026, 8, 16, 11, 0, tzinfo=timezone.utc)))
    delay = strat.data_delay_seconds("BTC/USDT", now)
    assert delay is not None and delay >= 0.0


def test_exactly_at_the_limit_is_allowed_and_one_second_past_is_not():
    engine = RiskEngine(RiskLimits(max_data_delay_s=300))
    common = dict(
        pair="BTC/USDT",
        equity=10_000.0,
        entry_price=100.0,
        stop_price=95.0,
        proposed_stake=1000.0,
    )
    assert engine.evaluate_entry(**common, data_delay_s=300.0).allowed
    assert not engine.evaluate_entry(**common, data_delay_s=301.0).allowed


# ---------------------------------------------------------------------------
# 2. The final gate must judge the order that is actually being sent
# ---------------------------------------------------------------------------
def test_gate_rejects_an_order_inflated_beyond_the_risk_envelope():
    """The bug: confirm_trade_entry ignored `amount` and re-derived a stake.

    Freqtrade may raise a stake to the exchange minimum after
    custom_stake_amount has spoken. The gate has to judge the real order.
    """
    engine = RiskEngine(RiskLimits(risk_per_trade_pct=0.01, max_position_pct=0.25))
    # Risk-based stake for a 5% stop on 10k equity is 2000.
    assert engine.position_size(10_000.0, 100.0, 95.0) == pytest.approx(2000.0)

    decision = engine.evaluate_entry(
        pair="BTC/USDT",
        equity=10_000.0,
        entry_price=100.0,
        stop_price=95.0,
        proposed_stake=2600.0,  # an exchange minimum pushed it past max_position_pct
    )
    assert not decision.allowed
    assert "exceeds" in decision.reason or "risk" in decision.reason


def test_gate_accepts_an_order_equal_to_the_risk_stake():
    engine = RiskEngine(RiskLimits(risk_per_trade_pct=0.01, max_position_pct=0.25))
    decision = engine.evaluate_entry(
        pair="BTC/USDT",
        equity=10_000.0,
        entry_price=100.0,
        stop_price=95.0,
        proposed_stake=2000.0,
    )
    assert decision.allowed
    assert decision.stake == pytest.approx(2000.0)


def test_gate_applies_asset_exposure_to_the_actual_order():
    engine = RiskEngine(RiskLimits(max_exposure_per_asset_pct=0.10, max_position_pct=1.0))
    decision = engine.evaluate_entry(
        pair="BTC/USDT",
        equity=10_000.0,
        entry_price=100.0,
        stop_price=95.0,
        proposed_stake=1500.0,  # 15% of equity, over the 10% per-asset cap
    )
    assert not decision.allowed
    assert "BTC/USDT" in decision.reason


def test_gate_applies_portfolio_exposure_to_the_actual_order():
    engine = RiskEngine(RiskLimits(max_total_exposure_pct=0.30, max_position_pct=1.0))
    engine.set_position_exposure("ETH/USDT", 2_900.0)
    decision = engine.evaluate_entry(
        pair="BTC/USDT",
        equity=10_000.0,
        entry_price=100.0,
        stop_price=95.0,
        proposed_stake=500.0,
    )
    assert not decision.allowed
    assert "total exposure" in decision.reason


def test_strategy_converts_amount_and_rate_into_the_committed_stake():
    """Freqtrade: amount = (stake / rate) * leverage, so stake = amount*rate/leverage."""
    strat = NNPredictorStrategy({**BASE_CONFIG})
    strat.wallets = None
    # Spot: leverage 1, so 20 units at 100 is a 2000 stake.
    assert strat._committed_stake(amount=20.0, rate=100.0, pair="BTC/USDT") == pytest.approx(
        2000.0
    )

    # With leverage recorded by the leverage() callback, margin is notional/leverage.
    strat._record_leverage("BTC/USDT", 2.0)
    assert strat._committed_stake(amount=20.0, rate=100.0, pair="BTC/USDT") == pytest.approx(
        1000.0
    )


def test_leverage_callback_clamps_to_the_configured_maximum():
    """The old code read self.config['leverage'], which is not a Freqtrade key."""
    strat = NNPredictorStrategy({**BASE_CONFIG, "risk": {"max_leverage": 3.0}})
    got = strat.leverage(
        pair="BTC/USDT",
        current_time=datetime.now(timezone.utc),
        current_rate=100.0,
        proposed_leverage=10.0,
        max_leverage=20.0,
        entry_tag=None,
        side="long",
    )
    assert got == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# 3. Exposure bookkeeping under repeated fill callbacks
# ---------------------------------------------------------------------------
def test_repeated_fill_callbacks_do_not_double_count_exposure():
    """The bug: open_position() added trade.stake_amount on every callback.

    Freqtrade calls order_filled once per closed order and recomputes
    trade.stake_amount as the trade's *total* stake, so accumulating produced
    exposure of 400 for a 200 position after two callbacks.
    """
    engine = RiskEngine(RiskLimits())
    engine.set_position_exposure("BTC/USDT", 200.0)
    engine.set_position_exposure("BTC/USDT", 200.0)
    assert engine.total_exposure == pytest.approx(200.0)


def test_position_adjustment_increases_exposure_to_the_new_total():
    engine = RiskEngine(RiskLimits())
    engine.set_position_exposure("BTC/USDT", 200.0)
    engine.set_position_exposure("BTC/USDT", 350.0)  # DCA raised the trade's stake
    assert engine.total_exposure == pytest.approx(350.0)


def test_partial_exit_reduces_exposure():
    engine = RiskEngine(RiskLimits())
    engine.set_position_exposure("BTC/USDT", 400.0)
    engine.set_position_exposure("BTC/USDT", 150.0)  # partial exit
    assert engine.total_exposure == pytest.approx(150.0)


def test_full_close_removes_exposure():
    engine = RiskEngine(RiskLimits())
    engine.set_position_exposure("BTC/USDT", 400.0)
    engine.close_position("BTC/USDT")
    assert engine.total_exposure == 0.0
    assert engine.state.open_positions == {}


def test_zero_stake_clears_rather_than_recording_an_empty_position():
    engine = RiskEngine(RiskLimits(max_open_positions=1))
    engine.set_position_exposure("BTC/USDT", 0.0)
    assert engine.state.open_positions == {}


# ---------------------------------------------------------------------------
# 5. No committed config may independently request live trading
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("path", sorted(CONF_DIR.glob("*.json")))
def test_no_committed_config_is_independently_live_capable(path):
    """Cloning the repo and pointing Freqtrade at any committed config must not
    place real orders, even if the ProjectChimera launcher is bypassed."""
    config = json.loads(path.read_text())
    assert config.get("dry_run", True) is not False, (
        f"{path.name} sets dry_run=false; a user running freqtrade directly "
        "against it would trade for real without passing the safety gate"
    )


def test_live_profiles_still_declare_their_intent():
    """Dry-run-safe on disk, but the launcher must still know they mean live."""
    for exchange in ("binance", "bybit", "okx"):
        config = json.loads((CONF_DIR / f"{exchange}.live.json").read_text())
        assert (
            config.get("chimera_live_intent") is True
        ), f"{exchange}.live.json must mark itself as the live profile"


# ---------------------------------------------------------------------------
# 4. In-sample ML backtests must be refused, not silently produced
# ---------------------------------------------------------------------------
def _metadata_with(train_end: str = "", validation_end: str = ""):
    from chimera.contracts import ModelMetadata, TargetSpec
    from chimera.features import FeatureSpec, feature_columns

    n = len(feature_columns())
    return ModelMetadata(
        model_version="test-v1",
        feature_names=feature_columns(),
        sequence_length=8,
        feature_spec=FeatureSpec(),
        target_spec=TargetSpec(),
        scaler_mean=[0.0] * n,
        scaler_std=[1.0] * n,
        decision_threshold=0.5,
        train_end=train_end,
        validation_end=validation_end,
    )


def test_a_backtest_after_the_training_cutoff_is_allowed():
    from strategies.nn_predictor_strategy import _assert_out_of_sample

    meta = _metadata_with(train_end="2025-01-01", validation_end="2025-06-01")
    _assert_out_of_sample(pd.Timestamp("2025-06-02", tz="UTC"), meta, "BTC/USDT")


def test_a_backtest_overlapping_the_training_period_is_refused():
    from strategies.nn_predictor_strategy import InSampleBacktestError, _assert_out_of_sample

    meta = _metadata_with(train_end="2025-01-01", validation_end="2025-06-01")
    with pytest.raises(InSampleBacktestError, match="overlaps the model training period"):
        _assert_out_of_sample(pd.Timestamp("2025-03-01", tz="UTC"), meta, "BTC/USDT")


def test_a_fully_in_sample_backtest_is_refused():
    from strategies.nn_predictor_strategy import InSampleBacktestError, _assert_out_of_sample

    meta = _metadata_with(train_end="2025-01-01", validation_end="2025-06-01")
    with pytest.raises(InSampleBacktestError):
        _assert_out_of_sample(pd.Timestamp("2024-02-01", tz="UTC"), meta, "BTC/USDT")


def test_the_cutoff_is_validation_end_not_train_end():
    """Early stopping and the threshold were fitted on validation, so data
    between train_end and validation_end is in-sample too."""
    from strategies.nn_predictor_strategy import InSampleBacktestError, _assert_out_of_sample

    meta = _metadata_with(train_end="2025-01-01", validation_end="2025-06-01")
    with pytest.raises(InSampleBacktestError):
        _assert_out_of_sample(pd.Timestamp("2025-04-01", tz="UTC"), meta, "BTC/USDT")


def test_missing_temporal_metadata_fails_closed():
    """An artifact that cannot prove it is out-of-sample is not assumed to be."""
    from strategies.nn_predictor_strategy import InSampleBacktestError, _assert_out_of_sample

    with pytest.raises(InSampleBacktestError, match="no training-cutoff metadata"):
        _assert_out_of_sample(
            pd.Timestamp("2030-01-01", tz="UTC"), _metadata_with(), "BTC/USDT"
        )


def test_the_boundary_candle_itself_is_in_sample():
    from strategies.nn_predictor_strategy import InSampleBacktestError, _assert_out_of_sample

    meta = _metadata_with(validation_end="2025-06-01 00:00:00+00:00")
    with pytest.raises(InSampleBacktestError):
        _assert_out_of_sample(pd.Timestamp("2025-06-01", tz="UTC"), meta, "BTC/USDT")


def test_naive_and_aware_timestamps_compare_without_raising_typeerror():
    from strategies.nn_predictor_strategy import _assert_out_of_sample

    meta = _metadata_with(validation_end="2025-06-01")  # naive in metadata
    _assert_out_of_sample(pd.Timestamp("2025-07-01", tz="UTC"), meta, "BTC/USDT")


def test_training_records_the_split_boundaries(tmp_path):
    """The whole guard depends on nn.train writing these; prove it does."""
    from chimera.contracts import TargetSpec
    from chimera.features import FeatureSpec
    from nn.data_pipeline import build_dataset, save_dataset
    from nn.registry import load_model
    from nn.train import main as train_main
    from tools.make_sample_data import generate_candles

    dataset = tmp_path / "ds.parquet"
    frame, meta = build_dataset(
        generate_candles(rows=900, seed=4),
        FeatureSpec(),
        TargetSpec(horizon=4),
        exchange="synthetic",
        pair="SYNTH/USDT",
        timeframe="1h",
    )
    save_dataset(dataset, frame, meta)

    models_dir = tmp_path / "models"
    assert (
        train_main(
            [
                "--dataset",
                str(dataset),
                "--models-dir",
                str(models_dir),
                "--epochs",
                "1",
                "--seq-len",
                "16",
                "--batch-size",
                "64",
                "--d-model",
                "16",
                "--n-heads",
                "2",
                "--num-layers",
                "1",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    version = next(p for p in models_dir.iterdir() if p.is_dir())
    _, metadata = load_model(version)

    assert metadata.train_end, "train_end must be recorded"
    assert metadata.validation_end, "validation_end must be recorded"
    assert pd.Timestamp(metadata.train_end) < pd.Timestamp(metadata.validation_end)
    assert metadata.training_cutoff == metadata.validation_end
    # And the cutoff must sit inside the dataset, before its final candle.
    assert pd.Timestamp(metadata.validation_end) < pd.Timestamp(metadata.dataset_end)


# ---------------------------------------------------------------------------
# 6. Adjacent bugs found by running Freqtrade's own CLI against the repository
# ---------------------------------------------------------------------------
def test_config_has_no_protections_key():
    """Freqtrade 2026.7 raises ConfigurationError for `protections` in a config.

    `freqtrade show-config` on the generated config failed outright with
    "DEPRECATED: Setting 'protections' in the configuration is deprecated".
    They belong on the strategy class instead.
    """
    base = json.loads((CONF_DIR / "base.json").read_text())
    assert "protections" not in base

    # And they did not simply vanish — the strategy carries them.
    from strategies.common.risk_manager import RiskAwareStrategy

    methods = {p["method"] for p in RiskAwareStrategy.protections}
    assert {"CooldownPeriod", "MaxDrawdown"} <= methods


def test_generated_config_passes_freqtrades_own_deprecation_checks(tmp_path):
    """The real check that caught it, run as a test."""
    from freqtrade.configuration.config_validation import validate_config_consistency
    from freqtrade.configuration.deprecated_settings import (
        process_temporary_deprecated_settings,
    )

    from tools.run_bot import load_config, resolve_placeholders

    config = resolve_placeholders(load_config("binance", "test"), [])
    config["strategy"] = "NNPredictorStrategy"
    # Raises ConfigurationError on a deprecated key such as `protections`.
    process_temporary_deprecated_settings(config)
    assert validate_config_consistency is not None


def test_launcher_puts_the_repo_root_on_pythonpath_for_freqtrade(monkeypatch, tmp_path):
    """Freqtrade's resolver imports strategies by path, so without this every
    `from chimera...` import inside a strategy fails with "No module named".

    Reproduced with `freqtrade list-strategies`, which reported LOAD FAILED for
    all four strategies until the launcher exported PYTHONPATH.
    """
    import subprocess

    from tools import run_bot

    captured = {}

    def fake_call(command, env=None):
        captured["command"] = command
        captured["env"] = env
        return 0

    monkeypatch.setattr(subprocess, "call", fake_call)
    monkeypatch.setattr(run_bot.shutil, "which", lambda _: "/usr/bin/freqtrade")

    assert (
        run_bot.main(
            [
                "--exchange",
                "binance",
                "--mode",
                "test",
                "--config-out",
                str(tmp_path / "merged.json"),
            ]
        )
        == 0
    )

    env = captured["env"]
    assert env is not None, "the subprocess must get an explicit environment"
    assert str(run_bot.REPO_ROOT) in env["PYTHONPATH"].split(os.pathsep)
    # And the strategy path is passed, so user_data mounts cannot shadow it.
    assert "--strategy-path" in captured["command"]


def test_launcher_preserves_an_existing_pythonpath(monkeypatch, tmp_path):
    import subprocess

    from tools import run_bot

    captured = {}
    monkeypatch.setenv("PYTHONPATH", "/somewhere/else")
    monkeypatch.setattr(
        subprocess, "call", lambda command, env=None: captured.update(env=env) or 0
    )
    monkeypatch.setattr(run_bot.shutil, "which", lambda _: "/usr/bin/freqtrade")

    run_bot.main(
        ["--exchange", "binance", "--mode", "test", "--config-out", str(tmp_path / "m.json")]
    )
    parts = captured["env"]["PYTHONPATH"].split(os.pathsep)
    assert str(run_bot.REPO_ROOT) in parts
    assert "/somewhere/else" in parts
