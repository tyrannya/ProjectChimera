"""Risk engine invariants.

Replaces a suite that imported a ``CommonRiskManager`` class which never
existed, against a hand-stubbed ``freqtrade`` module that defined an exception
the real library does not export. Everything here exercises the real code.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from chimera.risk import RiskEngine, RiskLimits


@pytest.fixture
def engine():
    return RiskEngine(RiskLimits())


def entry(engine, **overrides):
    kwargs = {
        "pair": "BTC/USDT",
        "equity": 10_000.0,
        "entry_price": 100.0,
        "stop_price": 95.0,
        **overrides,
    }
    return engine.evaluate_entry(**kwargs)


# --- position sizing ------------------------------------------------------
def test_sizing_risks_the_configured_fraction_not_the_wallet_fraction():
    """1% risk over a 5% stop is a 2000 notional on 10k, not 100.

    The distinction the original code got wrong: risking 1% of equity means
    losing 1% *if the stop is hit*, which requires a position 20x larger than
    1% of the wallet when the stop is 5% away.
    """
    engine = RiskEngine(RiskLimits(risk_per_trade_pct=0.01, max_position_pct=1.0))
    stake = engine.position_size(equity=10_000.0, entry_price=100.0, stop_price=95.0)
    assert stake == pytest.approx(2000.0)

    # And the loss it implies really is 1% of equity.
    assert stake * 0.05 == pytest.approx(10_000.0 * 0.01)


def test_sizing_is_capped_by_max_position_pct():
    engine = RiskEngine(RiskLimits(risk_per_trade_pct=0.01, max_position_pct=0.10))
    assert engine.position_size(10_000.0, 100.0, 95.0) == pytest.approx(1000.0)


def test_sizing_scales_inversely_with_stop_distance():
    engine = RiskEngine(RiskLimits(max_position_pct=1.0))
    tight = engine.position_size(10_000.0, 100.0, 99.0)  # 1% stop
    wide = engine.position_size(10_000.0, 100.0, 90.0)  # 10% stop
    assert tight == pytest.approx(10 * wide)


def test_sizing_refuses_stops_outside_the_allowed_band():
    engine = RiskEngine(RiskLimits(min_stop_distance_pct=0.005, max_stop_distance_pct=0.15))
    assert engine.position_size(10_000.0, 100.0, 99.9) == 0.0  # 0.1%, too tight
    assert engine.position_size(10_000.0, 100.0, 70.0) == 0.0  # 30%, too wide


def test_sizing_divides_by_leverage_and_caps_it():
    engine = RiskEngine(RiskLimits(max_position_pct=1.0, max_leverage=3.0))
    unlevered = engine.position_size(10_000.0, 100.0, 95.0, leverage=1.0)
    levered = engine.position_size(10_000.0, 100.0, 95.0, leverage=2.0)
    assert levered == pytest.approx(unlevered / 2)
    # Leverage above the cap is clamped, not honoured.
    assert engine.position_size(10_000.0, 100.0, 95.0, leverage=50.0) == pytest.approx(
        unlevered / 3
    )


# --- account guards -------------------------------------------------------
def test_drawdown_limit_halts_trading():
    # Daily loss is relaxed so this test isolates the drawdown guard.
    engine = RiskEngine(RiskLimits(max_drawdown_pct=0.10, max_daily_loss_pct=0.99))
    engine.update_equity(10_000.0)
    engine.update_equity(9_500.0)
    assert not engine.halted

    engine.update_equity(8_900.0)  # 11% off the peak
    assert engine.halted
    assert "drawdown" in engine.state.halt_reason
    assert not entry(engine, equity=8_900.0).allowed


def test_daily_loss_limit_halts_trading():
    engine = RiskEngine(RiskLimits(max_daily_loss_pct=0.05, max_drawdown_pct=0.99))
    day = datetime(2026, 1, 1, tzinfo=timezone.utc)
    engine.update_equity(10_000.0, now=day)
    engine.update_equity(9_400.0, now=day + timedelta(hours=3))
    assert engine.halted
    assert "daily loss" in engine.state.halt_reason


def test_daily_loss_budget_resets_on_a_new_day():
    engine = RiskEngine(RiskLimits(max_daily_loss_pct=0.05, max_drawdown_pct=0.99))
    day_one = datetime(2026, 1, 1, tzinfo=timezone.utc)
    engine.update_equity(10_000.0, now=day_one)
    engine.update_equity(9_600.0, now=day_one + timedelta(hours=6))  # -4%, allowed
    assert not engine.halted

    day_two = day_one + timedelta(days=1)
    engine.update_equity(9_600.0, now=day_two)
    engine.update_equity(9_300.0, now=day_two + timedelta(hours=1))  # -3.1% of today
    assert not engine.halted


def test_non_positive_equity_halts():
    engine = RiskEngine()
    engine.update_equity(0.0)
    assert engine.halted


# --- exposure and position count ------------------------------------------
def test_max_open_positions_is_enforced():
    engine = RiskEngine(RiskLimits(max_open_positions=2))
    engine.open_position("BTC/USDT", 100.0)
    engine.open_position("ETH/USDT", 100.0)
    decision = entry(engine, pair="SOL/USDT")
    assert not decision.allowed
    assert "max open positions" in decision.reason


def test_existing_pair_may_still_be_added_to_at_the_position_cap():
    engine = RiskEngine(RiskLimits(max_open_positions=1))
    engine.open_position("BTC/USDT", 100.0)
    assert entry(engine, pair="BTC/USDT").allowed


def test_total_exposure_cap_is_enforced():
    engine = RiskEngine(RiskLimits(max_total_exposure_pct=0.30, max_open_positions=10))
    engine.open_position("ETH/USDT", 2_900.0)
    decision = entry(engine)
    assert not decision.allowed
    assert "exposure" in decision.reason


def test_per_asset_exposure_cap_is_enforced():
    engine = RiskEngine(RiskLimits(max_exposure_per_asset_pct=0.10, max_position_pct=1.0))
    decision = entry(engine)  # would size 2000 = 20% of equity
    assert not decision.allowed
    assert "exposure in BTC/USDT" in decision.reason


# --- operational guards ----------------------------------------------------
def test_order_rate_limit_halts():
    engine = RiskEngine(RiskLimits(max_orders_per_minute=3))
    for _ in range(3):
        engine.record_order()
    assert not engine.halted
    engine.record_order()
    assert engine.halted
    assert "order rate" in engine.state.halt_reason


def test_order_rate_window_rolls_forward():
    clock = [1000.0]
    engine = RiskEngine(RiskLimits(max_orders_per_minute=2), clock=lambda: clock[0])
    engine.record_order()
    engine.record_order()
    clock[0] += 61.0  # the earlier orders age out of the 60s window
    engine.record_order()
    assert not engine.halted


def test_loss_streak_triggers_a_cooldown():
    clock = [1000.0]
    engine = RiskEngine(
        RiskLimits(loss_streak_limit=3, cooldown_seconds=600), clock=lambda: clock[0]
    )
    for _ in range(3):
        engine.record_trade_result(-10.0)

    decision = entry(engine)
    assert not decision.allowed
    assert "cooldown" in decision.reason

    clock[0] += 601.0
    assert entry(engine).allowed


def test_a_win_resets_the_loss_streak():
    engine = RiskEngine(RiskLimits(loss_streak_limit=3))
    engine.record_trade_result(-10.0)
    engine.record_trade_result(-10.0)
    engine.record_trade_result(+5.0)
    engine.record_trade_result(-10.0)
    assert entry(engine).allowed


def test_stale_market_data_blocks_entry():
    engine = RiskEngine(RiskLimits(max_data_staleness_s=300))
    assert not entry(engine, data_age_s=600).allowed
    assert entry(engine, data_age_s=10).allowed


def test_stale_inference_blocks_entry():
    engine = RiskEngine(RiskLimits(max_inference_staleness_s=300))
    decision = entry(engine, inference_age_s=900)
    assert not decision.allowed
    assert "inference stale" in decision.reason


def test_unhealthy_exchange_blocks_entry(engine):
    assert not entry(engine, exchange_healthy=False).allowed


# --- futures guards ---------------------------------------------------------
def test_excessive_funding_rate_blocks_entry():
    engine = RiskEngine(RiskLimits(max_funding_rate=0.0005))
    assert not entry(engine, funding_rate=0.002).allowed
    assert not entry(engine, funding_rate=-0.002).allowed
    assert entry(engine, funding_rate=0.0001).allowed


def test_close_liquidation_price_blocks_entry():
    engine = RiskEngine(RiskLimits(min_liquidation_distance_pct=0.5))
    assert not entry(engine, liquidation_price=80.0).allowed  # 20% away
    assert entry(engine, liquidation_price=40.0).allowed  # 60% away


def test_leverage_above_the_cap_blocks_entry():
    engine = RiskEngine(RiskLimits(max_leverage=3.0))
    assert not entry(engine, leverage=10.0).allowed


# --- kill switch -------------------------------------------------------------
def test_halt_blocks_every_entry(engine):
    engine.update_equity(10_000.0)
    assert entry(engine).allowed
    engine.halt("manual")
    decision = entry(engine)
    assert not decision.allowed
    assert decision.stake == 0.0
    assert "halted" in decision.reason


def test_halt_is_idempotent_so_alerts_do_not_storm(engine):
    engine.halt("first reason")
    engine.halt("second reason")
    assert engine.state.halt_reason == "first reason"


def test_halt_persists_across_a_restart(tmp_path):
    state = tmp_path / "risk.json"
    RiskEngine(state_path=state).halt("drawdown breach")

    revived = RiskEngine(state_path=state)
    assert revived.halted
    assert revived.state.halt_reason == "drawdown breach"
    assert not entry(revived).allowed


def test_resume_clears_the_persisted_halt(tmp_path):
    state = tmp_path / "risk.json"
    engine = RiskEngine(state_path=state)
    engine.halt("temporary")
    engine.resume()
    assert not RiskEngine(state_path=state).halted


def test_unreadable_state_file_fails_closed(tmp_path):
    state = tmp_path / "risk.json"
    state.write_text("{not json")
    engine = RiskEngine(state_path=state)
    assert engine.halted, "a corrupt risk state must not silently allow trading"


# --- decisions ----------------------------------------------------------------
def test_allowed_entry_reports_the_stake(engine):
    decision = entry(engine)
    assert decision.allowed and bool(decision) is True
    assert decision.stake > 0


def test_zero_equity_blocks_entry(engine):
    assert not entry(engine, equity=0.0).allowed


def test_limits_roundtrip_through_a_dict():
    limits = RiskLimits(max_drawdown_pct=0.2, max_open_positions=7)
    assert RiskLimits.from_dict(limits.to_dict()) == limits


def test_limits_ignore_unknown_config_keys():
    limits = RiskLimits.from_dict({"max_open_positions": 5, "not_a_limit": 1})
    assert limits.max_open_positions == 5
