"""Metrics, threshold selection and the model/decision agreement check."""

from __future__ import annotations

import numpy as np
import pytest

from chimera.contracts import HOLD_IDX, LONG_IDX, SHORT_IDX, TargetSpec, decide
from nn.baselines import MajorityClassBaseline, MomentumBaseline
from nn.evaluate import (
    classification_metrics,
    compare,
    confusion_matrix,
    evaluate,
    expected_calibration_error,
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
        "sharpe",
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
    proba = baseline.predict_proba(np.zeros((5, 1)))
    assert proba.shape == (5, 3)
    assert proba[0, HOLD_IDX] == pytest.approx(0.7)
    assert proba[0, LONG_IDX] == pytest.approx(0.2)


def test_momentum_baseline_follows_the_trend_feature():
    X = np.zeros((3, 4, 2))
    X[0, -1, 0] = 0.05  # up
    X[1, -1, 0] = -0.05  # down
    X[2, -1, 0] = 0.0  # flat
    proba = MomentumBaseline(feature_index=0).predict_proba(X)
    assert proba.argmax(axis=1).tolist() == [LONG_IDX, SHORT_IDX, HOLD_IDX]


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
