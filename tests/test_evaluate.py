"""Metrics, threshold selection and the model/decision agreement check."""

from __future__ import annotations

import numpy as np
import pytest

from chimera.contracts import HOLD_IDX, LONG_IDX, SHORT_IDX, TargetSpec, decide
from nn.baselines import MajorityClassBaseline, MomentumBaseline
from nn import evaluate as ev
from nn.evaluate import (
    classification_metrics,
    compare,
    confusion_matrix,
    evaluate,
    expected_calibration_error,
    realised_trades,
    select_threshold,
    signals_from_proba,
    trading_metrics,
)


# --- signals ---------------------------------------------------------------
def test_signals_from_proba_matches_decide():
    """The vectorised offline rule and the per-request live rule must agree.

    If they diverge, the strategy trades one policy while the reports measure
    another.
    """
    rng = np.random.default_rng(0)
    raw = rng.random((300, 3))
    proba = raw / raw.sum(axis=1, keepdims=True)

    for threshold in (0.34, 0.4, 0.5, 0.6, 0.8):
        vectorised = signals_from_proba(proba, threshold)
        for i, row in enumerate(proba):
            expected = decide({"SHORT": row[0], "HOLD": row[1], "LONG": row[2]}, threshold)
            assert vectorised[i] == {"SHORT": 0, "HOLD": 1, "LONG": 2}[expected.value]


def test_a_high_threshold_suppresses_all_trades():
    proba = np.array([[0.4, 0.2, 0.4], [0.5, 0.1, 0.4]])
    assert (signals_from_proba(proba, 0.99) == HOLD_IDX).all()


# --- classification --------------------------------------------------------
def test_confusion_matrix_rows_are_truth():
    y_true = np.array([0, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 2])
    matrix = confusion_matrix(y_true, y_pred)
    assert matrix[0, 0] == 1 and matrix[0, 1] == 1
    assert matrix[1, 1] == 1 and matrix[2, 2] == 1
    assert matrix.sum() == 4


def test_perfect_predictions_score_perfectly():
    y_true = np.array([SHORT_IDX, HOLD_IDX, LONG_IDX] * 10)
    proba = np.eye(3)[y_true]
    report = classification_metrics(proba, y_true, threshold=0.5)
    assert report["macro_f1"] == pytest.approx(1.0)
    assert report["directional_accuracy"] == pytest.approx(1.0)
    assert report["accuracy"] == pytest.approx(1.0)


def test_directional_accuracy_ignores_holds():
    """Counting HOLD calls as correct directional predictions would inflate it."""
    y_true = np.array([LONG_IDX, SHORT_IDX, HOLD_IDX, HOLD_IDX])
    proba = np.array([[0.1, 0.1, 0.8], [0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.8, 0.1]])
    report = classification_metrics(proba, y_true, threshold=0.5)
    assert report["directional_accuracy"] == pytest.approx(1.0)
    assert report["coverage"] == pytest.approx(0.5)


def test_coverage_reports_how_often_a_trade_is_called():
    y_true = np.zeros(10, dtype=int)
    proba = np.tile([0.34, 0.33, 0.33], (10, 1))
    assert classification_metrics(proba, y_true, threshold=0.9)["coverage"] == 0.0


def test_report_includes_the_true_class_distribution():
    y_true = np.array([SHORT_IDX] * 5 + [HOLD_IDX] * 3 + [LONG_IDX] * 2)
    proba = np.tile([0.34, 0.33, 0.33], (10, 1))
    report = classification_metrics(proba, y_true, threshold=0.5)
    assert report["class_distribution"] == {"SHORT": 5, "HOLD": 3, "LONG": 2}


def test_calibration_error_is_zero_for_a_perfectly_calibrated_model():
    proba = np.tile([0.0, 0.0, 1.0], (100, 1))
    y_true = np.full(100, LONG_IDX)
    assert expected_calibration_error(proba, y_true) == pytest.approx(0.0, abs=1e-9)


def test_calibration_error_catches_overconfidence():
    """Claims 100% confidence, is right half the time."""
    proba = np.tile([0.0, 0.0, 1.0], (100, 1))
    y_true = np.array([LONG_IDX] * 50 + [SHORT_IDX] * 50)
    assert expected_calibration_error(proba, y_true) == pytest.approx(0.5, abs=0.01)


# --- trading ----------------------------------------------------------------
def test_costs_are_charged_on_every_trade():
    spec = TargetSpec(horizon=1, fee_rate=0.001, slippage_rate=0.001)  # 0.4% round trip
    signals = np.array([LONG_IDX, LONG_IDX, LONG_IDX])
    future_return = np.array([0.01, 0.01, 0.01])
    report = trading_metrics(signals, future_return, spec)
    assert report["n_trades"] == 3
    assert report["avg_trade"] == pytest.approx(0.01 - 0.004)
    assert report["total_costs"] == pytest.approx(3 * 0.004)


def test_a_move_smaller_than_costs_is_a_losing_trade():
    spec = TargetSpec(horizon=1, fee_rate=0.001, slippage_rate=0.001)
    report = trading_metrics(np.array([LONG_IDX]), np.array([0.002]), spec)
    assert report["net_return"] < 0


def test_shorts_profit_from_falling_prices():
    spec = TargetSpec(horizon=1, fee_rate=0.0, slippage_rate=0.0)
    report = trading_metrics(np.array([SHORT_IDX]), np.array([-0.05]), spec)
    assert report["net_return"] == pytest.approx(0.05)


def test_trades_do_not_overlap():
    """Holding for `horizon` candles means the next signal is skipped.

    Counting overlapping trades would book the same price move several times.
    """
    spec = TargetSpec(horizon=3, fee_rate=0.0, slippage_rate=0.0)
    signals = np.full(9, LONG_IDX)
    report = trading_metrics(signals, np.full(9, 0.01), spec)
    assert report["n_trades"] == 3


def test_holds_produce_no_trades():
    spec = TargetSpec(horizon=1)
    report = trading_metrics(np.full(10, HOLD_IDX), np.full(10, 0.05), spec)
    assert report["n_trades"] == 0
    assert report["net_return"] == 0.0
    assert report["exposure"] == 0.0


def test_max_drawdown_is_measured_on_the_equity_curve():
    spec = TargetSpec(horizon=1, fee_rate=0.0, slippage_rate=0.0)
    signals = np.array([LONG_IDX, LONG_IDX, LONG_IDX])
    report = trading_metrics(signals, np.array([0.10, -0.20, 0.05]), spec)
    assert report["max_drawdown"] == pytest.approx(0.20, abs=1e-6)


def test_win_rate_and_profit_factor():
    spec = TargetSpec(horizon=1, fee_rate=0.0, slippage_rate=0.0)
    signals = np.full(4, LONG_IDX)
    report = trading_metrics(signals, np.array([0.02, 0.02, -0.01, -0.01]), spec)
    assert report["win_rate"] == pytest.approx(0.5)
    assert report["profit_factor"] == pytest.approx(2.0)


def test_report_contains_every_required_trading_metric():
    spec = TargetSpec(horizon=1)
    report = trading_metrics(np.array([LONG_IDX]), np.array([0.05]), spec)
    for key in (
        "net_return",
        "total_costs",
        "annualised_sharpe",
        "per_trade_sharpe",
        "sharpe_basis",
        "candle_max_drawdown",
        "max_drawdown",
        "win_rate",
        "profit_factor",
        "n_trades",
        "exposure",
        "avg_trade",
        "turnover",
    ):
        assert key in report


# --- threshold selection -------------------------------------------------------
def test_threshold_is_chosen_on_the_data_it_is_given():
    rng = np.random.default_rng(1)
    n = 400
    # Confident rows are right, unconfident rows are noise: a good threshold
    # should sit high enough to keep only the confident ones.
    proba = np.full((n, 3), 0.33)
    future_return = rng.normal(0, 0.01, n)
    confident = np.arange(0, n, 2)
    proba[confident] = 0.05
    proba[confident, 2] = 0.90
    future_return[confident] = 0.05

    threshold, report = select_threshold(proba, future_return, TargetSpec(horizon=1))
    assert 0.34 <= threshold <= 0.9
    assert report["n_trades"] >= 10
    assert report["net_return"] > 0


def test_threshold_selection_respects_the_minimum_trade_count():
    proba = np.tile([0.05, 0.05, 0.90], (200, 1))
    future_return = np.full(200, 0.05)
    _, report = select_threshold(proba, future_return, TargetSpec(horizon=1), min_trades=50)
    assert report["n_trades"] >= 50


def test_threshold_selection_falls_back_rather_than_crashing():
    proba = np.tile([0.34, 0.33, 0.33], (5, 1))
    threshold, report = select_threshold(
        proba, np.zeros(5), TargetSpec(horizon=1), min_trades=1000
    )
    assert isinstance(threshold, float)
    assert "n_trades" in report


# --- baselines -------------------------------------------------------------------
def test_majority_baseline_learns_the_class_prior():
    y = np.array([HOLD_IDX] * 70 + [LONG_IDX] * 20 + [SHORT_IDX] * 10)
    baseline = MajorityClassBaseline().fit(y)

    # The prior is kept for inspection...
    assert baseline.prior[HOLD_IDX] == pytest.approx(0.7)
    assert baseline.prior[LONG_IDX] == pytest.approx(0.2)
    # ...but the emitted action is a hard one-hot on the majority class.
    assert baseline.majority == HOLD_IDX
    proba = baseline.predict_proba(np.zeros((5, 1)))
    assert proba.shape == (5, 3)
    assert proba[0].tolist() == [0.0, 1.0, 0.0]


def test_majority_baseline_ties_break_deterministically():
    """Two runs on the same training rows must not disagree about the floor."""
    y = np.array([SHORT_IDX] * 30 + [HOLD_IDX] * 30 + [LONG_IDX] * 30)
    first = MajorityClassBaseline().fit(y).majority
    second = MajorityClassBaseline().fit(y[::-1].copy()).majority
    assert first == second == SHORT_IDX


def test_the_majority_baseline_action_does_not_depend_on_the_threshold():
    """Regression: the floor must not move when the model's threshold moves.

    The baseline used to emit the empirical class prior, so a threshold above
    the prior's largest entry silently suppressed it to HOLD — making a rule
    with no fitted parameters vary across seed-only reruns, because the
    threshold it was scored at came from the model.
    """
    for majority, y in (
        (HOLD_IDX, [HOLD_IDX] * 70 + [LONG_IDX] * 20 + [SHORT_IDX] * 10),
        (LONG_IDX, [LONG_IDX] * 45 + [SHORT_IDX] * 40 + [HOLD_IDX] * 15),
        (SHORT_IDX, [SHORT_IDX] * 50 + [HOLD_IDX] * 30 + [LONG_IDX] * 20),
    ):
        baseline = MajorityClassBaseline().fit(np.array(y))
        proba = baseline.predict_proba(np.zeros((6, 1)))
        for threshold in (0.0, 0.34, 0.5, 0.7, 0.9, 1.0):
            actions = set(signals_from_proba(proba, threshold).tolist())
            assert actions == {majority}, (majority, threshold, actions)


def test_momentum_baseline_follows_the_trend_feature():
    X = np.zeros((3, 4, 2))
    X[0, -1, 0] = 0.05  # up
    X[1, -1, 0] = -0.05  # down
    X[2, -1, 0] = 0.0  # flat
    proba = MomentumBaseline(feature_index=0).predict_proba(X)
    assert proba.argmax(axis=1).tolist() == [LONG_IDX, SHORT_IDX, HOLD_IDX]
    assert proba.sum(axis=1).tolist() == [1.0, 1.0, 1.0]


def test_the_momentum_baseline_action_does_not_depend_on_the_threshold():
    """The same regression, for the other rule: it had a fabricated confidence."""
    X = np.zeros((3, 4, 2))
    X[0, -1, 0] = 0.05
    X[1, -1, 0] = -0.05
    proba = MomentumBaseline(feature_index=0).predict_proba(X)
    for threshold in (0.0, 0.34, 0.5, 0.7, 0.9, 1.0):
        assert signals_from_proba(proba, threshold).tolist() == [
            LONG_IDX,
            SHORT_IDX,
            HOLD_IDX,
        ], threshold


def test_momentum_baseline_rejects_2d_input():
    with pytest.raises(ValueError):
        MomentumBaseline(feature_index=0).predict_proba(np.zeros((3, 2)))


# --- reporting ---------------------------------------------------------------------
def test_evaluate_returns_both_report_halves():
    y = np.array([LONG_IDX] * 20)
    proba = np.tile([0.05, 0.15, 0.80], (20, 1))
    report = evaluate(proba, y, np.full(20, 0.03), TargetSpec(horizon=1), 0.5)
    assert "classification" in report and "trading" in report


def test_compare_renders_every_model_as_a_row():
    y = np.array([LONG_IDX] * 20)
    proba = np.tile([0.05, 0.15, 0.80], (20, 1))
    report = evaluate(proba, y, np.full(20, 0.03), TargetSpec(horizon=1), 0.5)
    table = compare({"baseline": report, "mtst": report})
    assert "| baseline |" in table and "| mtst |" in table


# --- portfolio equity: the two reconciliation invariants -----------------------
#
# `annualised_sharpe` is only worth reading if the series it comes from is the
# equity curve already being reported. These tests are that proof, and they are
# closed-form: they assert the identity, not that the number looks plausible.
def _market(n: int, seed: int = 0, *, gap_after: int | None = None) -> ev.MarketContext:
    """A price path and hourly timestamps, optionally with a missing stretch."""
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
    offsets = np.arange(n)
    if gap_after is not None:
        # Everything after `gap_after` happens a day later than its row implies:
        # 24 candles the exchange never published.
        offsets = offsets + 24 * (offsets > gap_after)
    dates = np.datetime64("2023-01-01T00:00") + offsets * np.timedelta64(1, "h")
    return ev.MarketContext(close=close, dates=dates, candles_per_year=24 * 365)


def _future_return(close: np.ndarray, horizon: int) -> np.ndarray:
    return close[horizon:] / close[:-horizon] - 1.0


def test_one_long_trade_reconciles_with_its_realised_trade_return():
    spec = TargetSpec(horizon=4, fee_rate=0.001, slippage_rate=0.001)
    market = _market(40, seed=1)
    future_return = _future_return(market.close, spec.horizon)
    signals = np.full(len(future_return), HOLD_IDX)
    signals[5] = LONG_IDX

    _, _, returns = realised_trades(signals, future_return, spec)
    equity = ev.portfolio_equity(market, [(5, 9, 1.0)], spec.cost_threshold / 2, 5, 9)

    # Start-to-exit portfolio return is the trade's net return, exactly.
    assert equity[-1] - 1.0 == pytest.approx(returns[0], abs=1e-12)


def test_one_short_trade_reconciles_too():
    """The direction that a compounded construction would silently re-lever."""
    spec = TargetSpec(horizon=4, fee_rate=0.001, slippage_rate=0.001)
    market = _market(40, seed=2)
    future_return = _future_return(market.close, spec.horizon)
    signals = np.full(len(future_return), HOLD_IDX)
    signals[5] = SHORT_IDX

    _, _, returns = realised_trades(signals, future_return, spec)
    equity = ev.portfolio_equity(market, [(5, 9, -1.0)], spec.cost_threshold / 2, 5, 9)
    assert equity[-1] - 1.0 == pytest.approx(returns[0], abs=1e-12)


def test_sequential_trades_compound_exactly_like_the_trade_equity_curve():
    """The invariant that a `1 + cumsum(pnl)` construction would fail.

    Across several trades the candle-level curve has to reproduce
    ``cumprod(1 + trade_returns)`` — the curve ``net_return`` and
    ``max_drawdown`` already come from — at every completed trade boundary. If
    it does not, the Sharpe taken from it is measuring a different portfolio
    than the numbers printed beside it.
    """
    spec = TargetSpec(horizon=3, fee_rate=0.002, slippage_rate=0.002)  # 0.8% round trip
    market = _market(80, seed=3)
    future_return = _future_return(market.close, spec.horizon)
    signals = np.full(len(future_return), HOLD_IDX)
    for row in (2, 10, 25, 40):
        signals[row] = LONG_IDX
    for row in (16, 50):
        signals[row] = SHORT_IDX

    entries, directions, returns = realised_trades(signals, future_return, spec)
    assert len(returns) >= 4, "the fixture must exercise compounding"

    positions = [
        (int(e), int(e) + spec.horizon, float(d)) for e, d in zip(entries, directions)
    ]
    start, end = int(entries[0]), int(entries[-1]) + spec.horizon
    equity = ev.portfolio_equity(market, positions, spec.cost_threshold / 2, start, end)
    expected = np.cumprod(1.0 + returns)

    for trade, entry in enumerate(entries):
        exit_row = int(entry) + spec.horizon
        assert equity[exit_row - start] == pytest.approx(expected[trade], abs=1e-12)
    assert equity[-1] == pytest.approx(expected[-1], abs=1e-12)


def test_costs_are_charged_once_per_side_and_nowhere_else():
    """Entry side at entry, exit side at exit. A flat price path isolates them."""
    spec = TargetSpec(horizon=5, fee_rate=0.001, slippage_rate=0.001)
    per_side = spec.cost_threshold / 2
    close = np.full(30, 100.0)
    dates = np.datetime64("2023-01-01T00:00") + np.arange(30) * np.timedelta64(1, "h")
    market = ev.MarketContext(close=close, dates=dates, candles_per_year=24 * 365)

    equity = ev.portfolio_equity(market, [(4, 9, 1.0)], per_side, 4, 9)
    # Price never moves, so every change in equity is a cost.
    assert equity[0] == pytest.approx(1.0 - per_side, abs=1e-12)  # entry candle
    assert equity[1:5] == pytest.approx(1.0 - per_side, abs=1e-12)  # held, no move
    assert equity[5] == pytest.approx(1.0 - 2 * per_side, abs=1e-12)  # exit candle
    # Total charged over the trade is exactly one round trip.
    assert 1.0 - equity[-1] == pytest.approx(spec.cost_threshold, abs=1e-12)


def test_flat_candles_leave_the_portfolio_unchanged():
    spec = TargetSpec(horizon=2, fee_rate=0.001, slippage_rate=0.001)
    market = _market(40, seed=4)
    future_return = _future_return(market.close, spec.horizon)
    signals = np.full(len(future_return), HOLD_IDX)
    signals[10] = LONG_IDX

    equity = ev.portfolio_equity(market, [(10, 12, 1.0)], spec.cost_threshold / 2, 0, 20)
    returns = ev.portfolio_returns(equity)
    # Held over candles 10..12; everything else is flat and earns exactly zero.
    held = np.zeros(len(returns), dtype=bool)
    held[10:13] = True
    assert (returns[~held] == 0.0).all()
    assert (returns[held] != 0.0).any()


def test_annualised_sharpe_is_the_closed_form_with_no_horizon_term():
    """The whole point of the correction, asserted as an identity.

    If the reported number equals mean/std of the portfolio series times
    ``sqrt(candles_per_year)``, then no ``horizon`` term can be hiding in the
    annualisation — which is exactly what the previous formula had.
    """
    spec = TargetSpec(horizon=3, fee_rate=0.001, slippage_rate=0.001)
    market = _market(120, seed=5)
    future_return = _future_return(market.close, spec.horizon)
    signals = np.full(len(future_return), HOLD_IDX)
    for row in (5, 30, 60, 90):
        signals[row] = LONG_IDX

    report = trading_metrics(signals, future_return, spec, market=market)
    entries, directions, _ = realised_trades(signals, future_return, spec)
    positions = [
        (int(e), int(e) + spec.horizon, float(d)) for e, d in zip(entries, directions)
    ]
    start, end = 0, len(future_return) - 1 + spec.horizon
    equity = ev.portfolio_equity(market, positions, spec.cost_threshold / 2, start, end)
    returns = ev.portfolio_returns(equity)

    expected = returns.mean() / returns.std(ddof=1) * np.sqrt(market.candles_per_year)
    assert report["annualised_sharpe"] == pytest.approx(expected, abs=5e-5)
    assert report["sharpe_basis"] == ev.SHARPE_BASIS


def test_zero_padding_matches_materialising_the_zeros():
    """The sums shortcut and an explicitly padded array must agree."""
    returns = np.array([0.01, -0.02, 0.0, 0.03, -0.01])
    padded = np.concatenate([returns, np.zeros(20)])
    shortcut, _ = ev.annualised_sharpe(returns, len(padded), 24 * 365)
    direct = padded.mean() / padded.std(ddof=1) * np.sqrt(24 * 365)
    assert shortcut == pytest.approx(direct, abs=1e-12)


def test_per_trade_sharpe_is_not_annualised():
    spec = TargetSpec(horizon=1, fee_rate=0.0, slippage_rate=0.0)
    trade_returns = np.array([0.02, -0.01, 0.03, -0.02])
    signals = np.full(4, LONG_IDX)
    report = trading_metrics(signals, trade_returns, spec)
    expected = trade_returns.mean() / trade_returns.std(ddof=1)
    assert report["per_trade_sharpe"] == pytest.approx(expected, abs=1e-4)


def test_idle_capital_deflates_the_annualised_sharpe():
    """The same trades, judged over a longer idle stretch, score lower.

    This is the correction, stated as a controlled experiment: both arms take
    the *identical* trades, so their per-trade statistics are identical and the
    old formula gives them the same number. Only the amount of time capital sat
    in cash differs. A portfolio that earned those trades over ten times as long
    is a worse portfolio, and only the new statistic says so.
    """
    spec = TargetSpec(horizon=2, fee_rate=0.0005, slippage_rate=0.0005)
    market = _market(600, seed=6)
    future_return = _future_return(market.close, spec.horizon)
    entries = (2, 8, 14, 20, 26)

    def report_over(scored_rows: int) -> dict:
        signals = np.full(scored_rows, HOLD_IDX)
        for row in entries:
            signals[row] = LONG_IDX
        return trading_metrics(signals, future_return[:scored_rows], spec, market=market)

    busy = report_over(40)
    idle = report_over(400)

    # Identical trades: the per-trade view cannot tell the two apart, and neither
    # could the formula this replaced, which scaled exactly that number.
    assert busy["n_trades"] == idle["n_trades"] == len(entries)
    assert busy["net_return"] == pytest.approx(idle["net_return"], abs=1e-12)
    assert busy["per_trade_sharpe"] == pytest.approx(idle["per_trade_sharpe"], abs=1e-12)

    # The portfolio view does: capital was deployed a tenth as often.
    assert idle["exposure"] < busy["exposure"] / 5
    assert abs(idle["annualised_sharpe"]) < abs(busy["annualised_sharpe"]) / 2


def test_the_replaced_formula_would_have_reported_a_far_larger_number():
    """What the correction is worth, on a signal shaped like the real one.

    The old statistic was ``per_trade_sharpe * sqrt(candles_per_year / horizon)``
    — a fixed x38 at the project's 6-candle horizon on 1h candles, applied
    however rarely the strategy traded. This fixture trades on ~2% of candles,
    close to MTST's measured coverage, and the two numbers are not close.

    A horizon term cannot be hiding in the replacement: the closed form asserted
    above is ``mean/std * sqrt(candles_per_year)``, and ``horizon`` appears
    nowhere in it.
    """
    spec = TargetSpec(horizon=6, fee_rate=0.0005, slippage_rate=0.0005)
    market = _market(2100, seed=12)
    future_return = _future_return(market.close, spec.horizon)
    signals = np.full(2000, HOLD_IDX)
    signals[np.arange(5, 2000, 200)] = LONG_IDX

    report = trading_metrics(signals, future_return[:2000], spec, market=market)
    # Deployed 3% of the time, which is roughly MTST's measured coverage.
    assert report["exposure"] == pytest.approx(0.03, abs=0.005)

    old_value = report["per_trade_sharpe"] * np.sqrt((24 * 365) / spec.horizon)
    assert abs(report["annualised_sharpe"]) < abs(old_value) / 3

    # And the gap is not arbitrary: including the idle candles scales the figure
    # by roughly the square root of the fraction of time capital was at work.
    # That factor is what the old formula left out.
    ratio = abs(old_value / report["annualised_sharpe"])
    assert ratio == pytest.approx(1 / np.sqrt(report["exposure"]), rel=0.25)


def test_undefined_risk_statistics_are_none_not_zero():
    """A portfolio that never moved has no Sharpe. Reporting 0.0 would lie.

    The same rule has to hold for the model as for CASH, or an identical
    economic situation reads two different ways depending on the row's name.
    """
    spec = TargetSpec(horizon=1)
    market = _market(20, seed=7)
    report = trading_metrics(np.full(10, HOLD_IDX), np.full(10, 0.05), spec, market=market)
    assert report["n_trades"] == 0
    assert report["annualised_sharpe"] is None
    assert report["per_trade_sharpe"] is None
    assert report["annualised_sharpe_reason"]
    assert report["per_trade_sharpe_reason"]

    cash = ev.cash_reference(spec)
    assert cash["annualised_sharpe"] is None and cash["per_trade_sharpe"] is None


def test_a_close_series_that_stops_short_of_the_exit_raises():
    """Truncating would price the exit at the wrong candle, silently."""
    spec = TargetSpec(horizon=6, fee_rate=0.001, slippage_rate=0.001)
    market = _market(12, seed=8)
    with pytest.raises(ValueError, match="close series"):
        ev.portfolio_equity(market, [(4, 20, 1.0)], spec.cost_threshold / 2, 4, 20)


def test_no_report_carries_the_ambiguous_sharpe_field():
    """Regression: the old name must not come back as a convenience alias.

    Its semantics changed; keeping the name would make every committed artifact
    look comparable to a new run when it is not.
    """
    spec = TargetSpec(horizon=1)
    market = _market(20, seed=9)
    report = trading_metrics(np.array([LONG_IDX]), np.array([0.05]), spec, market=market)
    assert "sharpe" not in report
    assert "annualised_sharpe" in report and "per_trade_sharpe" in report


# --- economic references -------------------------------------------------------
def test_cash_pays_none_of_the_strategys_costs():
    """CASH is the zero line. It must not inherit the model's cost model.

    A no-trade policy that quietly paid trading costs would make every losing
    strategy look better than doing nothing.
    """
    expensive = TargetSpec(horizon=6, fee_rate=0.01, slippage_rate=0.01)  # 4% round trip
    cash = ev.cash_reference(expensive)
    assert cash["n_trades"] == 0
    assert cash["net_return"] == 0.0
    assert cash["gross_return"] == 0.0
    assert cash["total_costs"] == 0.0
    assert cash["max_drawdown"] == 0.0
    assert cash["exposure"] == 0.0
    assert cash["cost_model_applied"] is False


def test_buy_and_hold_is_priced_from_the_evaluation_windows_own_candles():
    """Entry at the first scored candle, exit at the last sample's label candle."""
    spec = TargetSpec(horizon=6, fee_rate=0.0005, slippage_rate=0.0005)
    market = _market(200, seed=13)
    row_index = np.arange(20, 150)

    hold = ev.buy_and_hold_reference(market, spec, row_index)
    assert hold["entry_row"] == 20
    assert hold["exit_row"] == 149 + spec.horizon
    expected = market.close[155] / market.close[20] - 1.0
    assert hold["gross_return"] == pytest.approx(expected, abs=1e-6)
    assert hold["net_return"] == pytest.approx(expected - spec.cost_threshold, abs=1e-6)


def test_buy_and_hold_pays_one_round_trip_not_one_per_horizon():
    """The distinction that separates holding an asset from churning it.

    Over a long window, charging the round trip each horizon would cost orders
    of magnitude more than holding — a different policy wearing the same name.
    """
    spec = TargetSpec(horizon=6, fee_rate=0.0005, slippage_rate=0.0005)
    market = _market(2100, seed=14)
    row_index = np.arange(0, 2000)

    hold = ev.buy_and_hold_reference(market, spec, row_index)
    assert hold["total_costs"] == pytest.approx(spec.cost_threshold, abs=1e-12)
    assert hold["n_trades"] == 1

    per_horizon = (hold["exit_row"] - hold["entry_row"]) / spec.horizon * spec.cost_threshold
    assert per_horizon > 100 * hold["total_costs"]
    assert hold["gross_return"] - hold["net_return"] == pytest.approx(
        spec.cost_threshold, abs=1e-6
    )


def test_buy_and_hold_and_the_model_report_the_same_sharpe_basis():
    """Two rows in one table must mean the same thing by the same name."""
    spec = TargetSpec(horizon=4, fee_rate=0.0005, slippage_rate=0.0005)
    market = _market(300, seed=15)
    future_return = _future_return(market.close, spec.horizon)
    signals = np.full(250, HOLD_IDX)
    signals[np.arange(3, 250, 20)] = LONG_IDX
    row_index = np.arange(250)

    model = trading_metrics(signals, future_return[:250], spec, market=market)
    hold = ev.buy_and_hold_reference(market, spec, row_index)

    assert model["sharpe_basis"] == ev.SHARPE_BASIS
    assert hold["sharpe_basis"] == ev.SHARPE_BASIS
    assert hold["annualised_sharpe"] is not None


def test_buy_and_hold_sharpe_is_net_of_its_cost():
    """Not a gross market Sharpe dressed up as a comparable one."""
    spec = TargetSpec(horizon=4, fee_rate=0.02, slippage_rate=0.02)  # 8% round trip
    market = _market(200, seed=16)
    row_index = np.arange(150)

    charged = ev.buy_and_hold_reference(market, spec, row_index)
    free = ev.buy_and_hold_reference(
        market, TargetSpec(horizon=4, fee_rate=0.0, slippage_rate=0.0), row_index
    )
    assert charged["annualised_sharpe"] != free["annualised_sharpe"]
    assert charged["net_return"] < free["net_return"]


def test_buy_and_hold_withholds_its_sharpe_across_a_market_data_gap():
    """A hold is exposed to a price path nobody recorded; the strategy is not.

    Rather than treat a 24-hour outage as one ordinary candle and publish the
    result under a name that promises comparability, the statistic is withheld.
    Return stays exact — start-to-end is well defined even when the path is not
    — and the drawdown is flagged as a lower bound.
    """
    spec = TargetSpec(horizon=4, fee_rate=0.0005, slippage_rate=0.0005)
    market = _market(200, seed=17, gap_after=80)
    row_index = np.arange(150)

    hold = ev.buy_and_hold_reference(market, spec, row_index)
    assert hold["spans_market_data_gap"] is True
    assert hold["missing_candles"] == 24
    assert hold["annualised_sharpe"] is None
    assert "invent" in hold["annualised_sharpe_reason"]
    assert hold["drawdown_is_lower_bound"] is True
    # Still exact, and still reported.
    expected = market.close[153] / market.close[0] - 1.0
    assert hold["gross_return"] == pytest.approx(expected, abs=1e-6)
    assert hold["net_return"] == pytest.approx(expected - spec.cost_threshold, abs=1e-6)


def test_a_gap_is_annualised_from_elapsed_time_not_row_count():
    """The strategy's padding: 24 unpublished candles are 24 idle intervals.

    The portfolio is provably flat across a gap — no trade may span one — so a
    zero is the true return there rather than an assumption, and counting rows
    instead of hours would treat a day-long outage as a single hour.
    """
    market = _market(200, seed=18, gap_after=80)
    contiguous = _market(200, seed=18)
    assert market.elapsed_intervals(0, 150) == contiguous.elapsed_intervals(0, 150) + 24
    assert market.missing_candles(0, 150) == 24
    # With no gap, elapsed intervals and row count agree exactly.
    assert contiguous.elapsed_intervals(0, 150) == 151
    assert contiguous.missing_candles(0, 150) == 0


def test_economic_references_bundle_both_policies():
    spec = TargetSpec(horizon=4)
    market = _market(200, seed=19)
    references = ev.economic_references(market, spec, np.arange(150))
    assert set(references) >= {"cash", "buy_and_hold"}
    assert references["buy_and_hold"]["available"] is True
    assert references["cash"]["net_return"] == 0.0


def test_buy_and_hold_is_unavailable_rather_than_invented_without_prices():
    spec = TargetSpec(horizon=4)
    hold = ev.buy_and_hold_reference(None, spec, np.arange(10))
    assert hold["available"] is False
    assert hold["reason"]
    assert "net_return" not in hold


def test_baseline_reports_do_not_move_with_the_models_threshold():
    """The floor must not shift when the model's fitted threshold shifts.

    Extended from the action-level check to the whole trading report: a
    one-hot rule's decisions are threshold-independent, so every number derived
    from them has to be too, including the new risk-adjusted statistics.
    """
    spec = TargetSpec(horizon=3, fee_rate=0.0005, slippage_rate=0.0005)
    market = _market(200, seed=20)
    future_return = _future_return(market.close, spec.horizon)
    y = np.array([HOLD_IDX] * 70 + [LONG_IDX] * 20 + [SHORT_IDX] * 10)
    baseline = MajorityClassBaseline().fit(y)
    proba = baseline.predict_proba(np.zeros((150, 1)))

    reports = [
        trading_metrics(
            signals_from_proba(proba, threshold), future_return[:150], spec, market=market
        )
        for threshold in (0.0, 0.34, 0.5, 0.7, 0.9, 1.0)
    ]
    assert all(report == reports[0] for report in reports)
