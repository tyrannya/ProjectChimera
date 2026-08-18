"""Strategy behaviour against the real Freqtrade IStrategy interface.

No module stubbing: these import the installed Freqtrade. If a signature ever
drifts, these tests break — which is the point.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest
from freqtrade.enums import RunMode

from chimera.inference_client import Prediction
from strategies.arb_mm import ArbMM
from strategies.nn_predictor_strategy import NNPredictorStrategy
from strategies.scalp_futures import ScalpFutures
from strategies.swing_spot import SwingSpot

BASE_CONFIG = {
    "stake_currency": "USDT",
    "stake_amount": 100,
    "dry_run": True,
    "runmode": "dry_run",
    "risk": {},
}


class FakeDataProvider:
    def __init__(self, runmode=RunMode.DRY_RUN, dataframe=None):
        self.runmode = runmode
        self._dataframe = dataframe

    def get_analyzed_dataframe(self, pair, timeframe):
        return (self._dataframe if self._dataframe is not None else pd.DataFrame()), None


@pytest.fixture
def strategy(candles):
    strat = NNPredictorStrategy({**BASE_CONFIG, "nn_infer_url": "http://test:3000"})
    strat.dp = FakeDataProvider(RunMode.DRY_RUN, candles)
    return strat


def stub_client(strategy, *, info=True, prediction=None, seq_len=32):
    """Point the strategy's client at canned answers instead of a socket."""
    from chimera.inference_client import ServerInfo

    from chimera.features import feature_columns

    server_info = (
        ServerInfo(
            model_version="v1",
            sequence_length=seq_len,
            n_features=len(feature_columns()),
            decision_threshold=0.5,
            feature_names=tuple(feature_columns()),
            feature_spec={},
        )
        if info
        else None
    )
    strategy.client.describe = lambda force=False: server_info
    strategy.client.predict = lambda *a, **k: prediction
    return strategy


def make_prediction(signal="LONG", confidence=0.7):
    probabilities = {"SHORT": 0.15, "HOLD": 0.15, "LONG": 0.70}
    if signal == "SHORT":
        probabilities = {"SHORT": 0.70, "HOLD": 0.15, "LONG": 0.15}
    elif signal == "HOLD":
        probabilities = {"SHORT": 0.15, "HOLD": 0.70, "LONG": 0.15}
    return Prediction(
        model_version="v1",
        signal=signal,
        probabilities=probabilities,
        confidence=confidence,
        decision_threshold=0.5,
        received_at=time.time(),
    )


# --- signal interpretation ------------------------------------------------
def test_long_prediction_produces_a_long_entry(strategy, candles):
    stub_client(strategy, prediction=make_prediction("LONG"))
    df = strategy.populate_entry_trend(
        strategy.populate_indicators(candles.copy(), {"pair": "BTC/USDT"}), {}
    )
    assert df["enter_long"].iloc[-1] == 1
    assert (
        df["enter_long"].iloc[:-1].sum() == 0
    ), "only the last candle may carry a live signal"


def test_hold_prediction_produces_no_entry(strategy, candles):
    stub_client(strategy, prediction=make_prediction("HOLD"))
    df = strategy.populate_entry_trend(
        strategy.populate_indicators(candles.copy(), {"pair": "BTC/USDT"}), {}
    )
    assert df["enter_long"].sum() == 0


def test_short_prediction_exits_a_long(strategy, candles):
    stub_client(strategy, prediction=make_prediction("SHORT"))
    analyzed = strategy.populate_indicators(candles.copy(), {"pair": "BTC/USDT"})
    entries = strategy.populate_entry_trend(analyzed.copy(), {})
    exits = strategy.populate_exit_trend(analyzed.copy(), {})
    assert entries["enter_long"].sum() == 0
    assert exits["exit_long"].iloc[-1] == 1


def test_short_produces_no_short_entry_when_shorting_is_off(strategy, candles):
    stub_client(strategy, prediction=make_prediction("SHORT"))
    df = strategy.populate_entry_trend(
        strategy.populate_indicators(candles.copy(), {"pair": "BTC/USDT"}), {}
    )
    assert df["enter_short"].sum() == 0


def test_short_produces_a_short_entry_when_shorting_is_on(strategy, candles):
    strategy.can_short = True
    stub_client(strategy, prediction=make_prediction("SHORT"))
    df = strategy.populate_entry_trend(
        strategy.populate_indicators(candles.copy(), {"pair": "BTC/USDT"}), {}
    )
    assert df["enter_short"].iloc[-1] == 1


# --- fail closed --------------------------------------------------------------
def test_unavailable_service_produces_no_entries(strategy, candles):
    stub_client(strategy, info=False)
    df = strategy.populate_entry_trend(
        strategy.populate_indicators(candles.copy(), {"pair": "BTC/USDT"}), {}
    )
    assert df["enter_long"].sum() == 0
    assert df["enter_short"].sum() == 0


def test_no_prediction_produces_no_entries(strategy, candles):
    stub_client(strategy, prediction=None)
    df = strategy.populate_entry_trend(
        strategy.populate_indicators(candles.copy(), {"pair": "BTC/USDT"}), {}
    )
    assert df["enter_long"].sum() == 0


def test_failure_does_not_turn_it_into_a_rule_strategy(strategy, candles):
    """The old fallback silently became an EMA/RSI strategy on any error."""
    stub_client(strategy, info=False)
    df = strategy.populate_indicators(candles.copy(), {"pair": "BTC/USDT"})
    assert (df["nn_signal"] == 0).all()
    assert "fb_ema_fast" not in df.columns


def test_the_rule_fallback_must_be_asked_for(candles):
    strat = NNPredictorStrategy({**BASE_CONFIG, "nn_fallback_strategy": True})
    strat.dp = FakeDataProvider(RunMode.DRY_RUN, candles)
    stub_client(strat, info=False)
    df = strat.populate_indicators(candles.copy(), {"pair": "BTC/USDT"})
    assert "fb_ema_fast" in df.columns


def test_a_feature_contract_mismatch_holds(strategy, candles):
    from chimera.inference_client import ServerInfo

    strategy.client.describe = lambda force=False: ServerInfo(
        model_version="v1",
        sequence_length=32,
        n_features=3,
        decision_threshold=0.5,
        feature_names=("something", "completely", "different"),
        feature_spec={},
    )
    strategy.client.predict = lambda *a, **k: make_prediction("LONG")
    df = strategy.populate_indicators(candles.copy(), {"pair": "BTC/USDT"})
    assert (df["nn_signal"] == 0).all()


def test_insufficient_history_holds(strategy, small_candles):
    stub_client(strategy, prediction=make_prediction("LONG"), seq_len=5000)
    df = strategy.populate_indicators(small_candles.copy(), {"pair": "BTC/USDT"})
    assert (df["nn_signal"] == 0).all()


def test_an_empty_dataframe_is_handled(strategy):
    empty = pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    df = strategy.populate_entry_trend(
        strategy.populate_indicators(empty, {"pair": "BTC/USDT"}), {}
    )
    assert len(df) == 0


# --- no stale prediction reuse ---------------------------------------------------
def test_predictions_are_not_reused_across_candles(strategy, candles):
    """Each candle asks again; the cache keys on the candle timestamp."""
    seen = []

    def record(pair, timeframe, timestamp, features):
        seen.append(timestamp)
        return make_prediction("LONG")

    stub_client(strategy)
    strategy.client.predict = record

    strategy.populate_indicators(candles.iloc[:-1].copy(), {"pair": "BTC/USDT"})
    strategy.populate_indicators(candles.copy(), {"pair": "BTC/USDT"})
    assert len(seen) == 2
    assert seen[0] != seen[1]


def test_inference_age_tracks_the_last_successful_prediction(strategy, candles):
    stub_client(strategy, prediction=make_prediction("LONG"))
    assert strategy.inference_age_seconds("BTC/USDT") is None
    strategy.populate_indicators(candles.copy(), {"pair": "BTC/USDT"})
    age = strategy.inference_age_seconds("BTC/USDT")
    assert age is not None and age < 5


def test_data_delay_is_reported_for_the_risk_engine(strategy, candles):
    """Delay is measured past the candle's close, so a 1h candle sampled 90
    minutes after it opened is 30 minutes late, not 90."""
    now = pd.Timestamp(candles["date"].iloc[-1]).to_pydatetime() + timedelta(minutes=90)
    delay = strategy.data_delay_seconds("BTC/USDT", now)
    assert delay == pytest.approx(1800, abs=1)


# --- risk integration ---------------------------------------------------------------
def test_confirm_trade_entry_blocks_when_halted(strategy):
    strategy.wallets = None
    strategy.risk.update_equity(10_000.0)
    strategy.risk.halt("test halt")
    assert (
        strategy.confirm_trade_entry(
            "BTC/USDT", "limit", 1.0, 100.0, "gtc", datetime.now(timezone.utc), None, "long"
        )
        is False
    )


def test_stop_price_follows_the_configured_stoploss(strategy):
    assert strategy._stop_price(100.0, "long") == pytest.approx(95.0)
    assert strategy._stop_price(100.0, "short") == pytest.approx(105.0)


# --- rule-based strategies -------------------------------------------------------------
def test_swing_spot_emits_signals(candles):
    strat = SwingSpot(BASE_CONFIG)
    analyzed = strat.populate_indicators(candles.copy(), {"pair": "BTC/USDT"})
    assert {"ema_fast", "ema_slow", "rsi"} <= set(analyzed.columns)
    assert analyzed["rsi"].between(0, 100).all()

    entries = strat.populate_entry_trend(analyzed.copy(), {})
    exits = strat.populate_exit_trend(analyzed.copy(), {})
    assert set(entries["enter_long"].unique()) <= {0, 1}
    assert set(exits["exit_long"].unique()) <= {0, 1}


def test_swing_spot_indicators_are_causal(candles):
    strat = SwingSpot(BASE_CONFIG)
    full = strat.populate_indicators(candles.copy(), {"pair": "BTC/USDT"})
    prefix = strat.populate_indicators(candles.iloc[:500].copy(), {"pair": "BTC/USDT"})
    pd.testing.assert_series_equal(prefix["ema_fast"], full["ema_fast"].iloc[:500])


def test_swing_spot_is_long_only(candles):
    strat = SwingSpot(BASE_CONFIG)
    analyzed = strat.populate_indicators(candles.copy(), {"pair": "BTC/USDT"})
    assert "enter_short" not in strat.populate_entry_trend(analyzed, {}).columns


@pytest.mark.parametrize("cls", [ScalpFutures, ArbMM])
def test_disabled_strategies_generate_no_entries(cls, candles):
    """They are kept for documentation, not for trading."""
    strat = cls(BASE_CONFIG)
    assert strat.DISABLED is True
    analyzed = strat.populate_indicators(candles.copy(), {"pair": "BTC/USDT"})
    entries = strat.populate_entry_trend(analyzed, {})
    assert entries["enter_long"].sum() == 0
    if "enter_short" in entries:
        assert entries["enter_short"].sum() == 0


def test_scalp_futures_does_not_fake_orderbook_data(candles):
    """A zero default made the missing feed invisible; NaN keeps it visible."""
    strat = ScalpFutures(BASE_CONFIG)
    analyzed = strat.populate_indicators(candles.copy(), {"pair": "BTC/USDT"})
    assert analyzed["orderbook_delta"].isna().all()


def test_arb_mm_does_not_fake_a_spread(candles):
    strat = ArbMM(BASE_CONFIG)
    analyzed = strat.populate_indicators(candles.copy(), {"pair": "ETH/USDT"})
    assert analyzed["zscore"].isna().all()


# --- real Freqtrade strategy resolution -------------------------------------
@pytest.mark.parametrize("name", ["NNPredictorStrategy", "SwingSpot", "ScalpFutures", "ArbMM"])
def test_strategies_load_through_freqtrades_own_resolver(name):
    """The real integration point: Freqtrade must be able to load each class.

    Catches things importing the module cannot — for example a `can_short`
    strategy against a spot config, which Freqtrade refuses outright.
    """
    from pathlib import Path

    from freqtrade.resolvers import StrategyResolver

    from tools.run_bot import load_config, resolve_placeholders

    root = Path(__file__).resolve().parents[1]
    config = resolve_placeholders(load_config("binance", "test"), [])
    config.update(
        strategy=name,
        runmode="dry_run",
        user_data_dir=root / "user_data",
        strategy_path=str(root / "strategies"),
        datadir=root / "user_data" / "data",
    )

    strategy = StrategyResolver.load_strategy(config)
    assert strategy.risk is not None, "every strategy must carry the risk engine"
    assert strategy.stoploss < 0


def test_base_config_does_not_override_strategy_timeframes():
    """Freqtrade's config wins over the strategy attribute, so a `timeframe` in
    base.json would silently force every strategy onto the same one."""
    import json
    from pathlib import Path

    base = json.loads((Path(__file__).resolve().parents[1] / "conf" / "base.json").read_text())
    assert "timeframe" not in base
