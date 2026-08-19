import json

import numpy as np
import pandas as pd
import pytest

from chimera.contracts import LONG_IDX, SHORT_IDX, TargetSpec
from nn import evaluate as ev
from nn import regime
from nn.regime import RegimeDataError, direction_attribution, load_predictions


def _prediction_frame(*, seed: int = 42) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fold": [0, 0],
            "seed": [seed, seed],
            "row_index": [10, 11],
            "timestamp": pd.to_datetime(
                ["2024-01-01T10:00:00Z", "2024-01-01T11:00:00Z"], utc=True
            ),
            "true_target": [LONG_IDX, LONG_IDX],
            "future_return": [0.01, 0.02],
            "p_short": [0.05, 0.05],
            "p_hold": [0.05, 0.05],
            "p_long": [0.90, 0.90],
            "selected_action": [LONG_IDX, LONG_IDX],
            "threshold": [0.50, 0.50],
        }
    )


def _artifact(*, declared: bool = True, frame: pd.DataFrame | None = None) -> dict:
    frame = _prediction_frame() if frame is None else frame
    spec = TargetSpec()
    ordered = frame.sort_values("row_index")
    report = ev.evaluate(
        ordered[["p_short", "p_hold", "p_long"]].to_numpy(dtype=np.float64),
        ordered["true_target"].to_numpy(dtype=np.int64),
        ordered["future_return"].to_numpy(dtype=np.float64),
        spec,
        0.50,
        row_index=ordered["row_index"].to_numpy(dtype=np.int64),
    )
    payload = {
        "dataset": {"target_spec": spec.to_dict(), "timeframe": "1h"},
        "config": {"seed": 42},
        "sealed_test": {"start_row": 100, "evaluated": False},
        "test_evaluated": False,
        "folds": [
            {
                "fold": 0,
                "samples": {"outer_validation": 2},
                "periods": {"outer_validation": {"row_range": [10, 20]}},
                "selection": {"threshold": 0.50},
                "outer_validation": {"mtst": report},
            }
        ],
    }
    if declared:
        payload["outer_predictions"] = "outer_predictions.parquet"
    return payload


def _write_run(tmp_path, frame: pd.DataFrame, artifact: dict):
    run = tmp_path / "run_a"
    run.mkdir(parents=True)
    path = run / "outer_predictions.parquet"
    frame.to_parquet(path, index=False)
    (run / "walkforward.json").write_text(json.dumps(artifact) + "\n")
    return path


def test_realised_trades_uses_dataset_rows_across_gap():
    spec = TargetSpec(horizon=6, fee_rate=0.0, slippage_rate=0.0)
    signals = np.array([LONG_IDX, LONG_IDX, LONG_IDX])
    future_return = np.array([0.01, 0.02, 0.03])

    old_positions, _, _ = ev.realised_trades(signals, future_return, spec)
    gap_positions, _, _ = ev.realised_trades(
        signals,
        future_return,
        spec,
        row_index=np.array([100, 101, 200]),
    )

    assert old_positions.tolist() == [0]
    assert gap_positions.tolist() == [0, 2]


def test_evaluate_forwards_row_index_to_trading_metrics():
    spec = TargetSpec(horizon=6, fee_rate=0.0, slippage_rate=0.0)
    proba = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )
    report = ev.evaluate(
        proba,
        np.array([LONG_IDX, LONG_IDX, LONG_IDX]),
        np.array([0.01, 0.02, 0.03]),
        spec,
        0.5,
        row_index=np.array([100, 101, 200]),
    )
    assert report["trading"]["n_trades"] == 2
    # Two six-candle trades over the represented row span [100, 200]. The
    # compressed three-sample array must not make this look 100% exposed.
    assert report["trading"]["exposure"] == pytest.approx(0.1188)


def test_load_predictions_requires_artifact_declaration_and_identity(tmp_path):
    frame = _prediction_frame()
    path = _write_run(tmp_path, frame, _artifact(frame=frame))
    loaded = load_predictions(path, sealed_test_start=100)
    assert loaded["_run_name"].unique().tolist() == [str(path.parent.resolve())]


def test_loaded_run_identity_is_unique_even_when_basenames_match(tmp_path):
    frame = _prediction_frame()
    first = _write_run(tmp_path / "experiment_a", frame, _artifact(frame=frame))
    second = _write_run(tmp_path / "experiment_b", frame, _artifact(frame=frame))

    first_id = load_predictions(first, sealed_test_start=100)["_run_name"].iloc[0]
    second_id = load_predictions(second, sealed_test_start=100)["_run_name"].iloc[0]
    assert first.parent.name == second.parent.name == "run_a"
    assert first_id != second_id


def test_load_predictions_refuses_undeclared_file(tmp_path):
    frame = _prediction_frame()
    path = _write_run(tmp_path, frame, _artifact(declared=False, frame=frame))
    with pytest.raises(RegimeDataError, match="declares outer_predictions"):
        load_predictions(path, sealed_test_start=100)


def test_load_predictions_refuses_seed_mismatch(tmp_path):
    frame = _prediction_frame(seed=99)
    path = _write_run(tmp_path, frame, _artifact(frame=frame))
    with pytest.raises(RegimeDataError, match="seed values"):
        load_predictions(path, sealed_test_start=100)


def test_load_predictions_refuses_non_integral_fold_ids(tmp_path):
    frame = _prediction_frame()
    frame["fold"] = [0.1, 0.9]
    path = _write_run(tmp_path, frame, _artifact())
    with pytest.raises(RegimeDataError, match="non-integral fold"):
        load_predictions(path, sealed_test_start=100)


def test_load_predictions_refuses_content_that_does_not_reproduce_report(tmp_path):
    original = _prediction_frame()
    stale = original.copy()
    stale[["p_short", "p_long"]] = stale[["p_long", "p_short"]].to_numpy()
    stale["selected_action"] = SHORT_IDX
    path = _write_run(tmp_path, stale, _artifact(frame=original))
    with pytest.raises(RegimeDataError, match="does not reproduce"):
        load_predictions(path, sealed_test_start=100)


def test_direction_attribution_keeps_same_seed_runs_independent():
    spec = TargetSpec(horizon=6, fee_rate=0.0, slippage_rate=0.0)
    first = _prediction_frame().iloc[[0]].copy()
    second = _prediction_frame().iloc[[0]].copy()
    first["_run_name"] = "/tmp/experiment-a/output"
    second["_run_name"] = "/tmp/experiment-b/output"

    report = direction_attribution(pd.concat([first, second], ignore_index=True), spec)
    assert report["long"]["trades"] == 2
    assert report["short"]["trades"] == 0


# --- prediction binding validates schema before excluding values ---------------
#
# The candle-level metrics cannot be *recomputed* from a per-sample predictions
# file — they need prices past the last scored row. Their presence and shape
# still can be checked, and must be: dropping them unconditionally would let a
# current artifact that lost one bind cleanly against its parquet.
def _current_report() -> dict:
    return {
        "classification": {"macro_f1": 0.2},
        "trading": {
            "n_trades": 3,
            "net_return": -0.01,
            "max_drawdown": 0.05,
            "annualised_sharpe": -0.5,
            "annualised_sharpe_reason": "",
            "sharpe_basis": "candle-level portfolio returns ...",
            "candle_max_drawdown": 0.07,
            "elapsed_intervals": 240,
        },
    }


def test_a_current_report_validates_and_keeps_its_reproducible_fields():
    report = _current_report()
    assert regime.validate_report_schema(report, "test") == "current"
    comparable = regime._comparable_report(report)
    # Values that cannot be reproduced are dropped...
    for field in regime.NON_REPRODUCIBLE_TRADING_METRICS:
        assert field not in comparable["trading"]
    # ...and everything else is still compared.
    assert comparable["trading"]["net_return"] == -0.01
    assert comparable["trading"]["max_drawdown"] == 0.05


def test_an_undefined_sharpe_is_a_valid_current_report():
    """`None` is a real state, not a malformed field."""
    report = _current_report()
    report["trading"]["annualised_sharpe"] = None
    report["trading"]["candle_max_drawdown"] = None
    assert regime.validate_report_schema(report, "test") == "current"


def test_a_current_report_missing_a_candle_level_field_fails_closed():
    """The regression: it must not bind by having the gap dropped."""
    report = _current_report()
    del report["trading"]["candle_max_drawdown"]
    with pytest.raises(regime.RegimeDataError, match="refusing to bind around the gap"):
        regime.validate_report_schema(report, "test")


def test_a_malformed_candle_level_field_fails_closed():
    report = _current_report()
    report["trading"]["elapsed_intervals"] = "many"
    with pytest.raises(regime.RegimeDataError, match="elapsed_intervals"):
        regime.validate_report_schema(report, "test")


def test_a_pre_correction_report_is_recognised_rather_than_refused():
    report = _current_report()
    for field in regime.NON_REPRODUCIBLE_TRADING_METRICS:
        report["trading"].pop(field)
    report["trading"]["sharpe"] = 18.22
    assert regime.validate_report_schema(report, "test") == "legacy_pre_metric_correction"


def test_a_half_renamed_report_fails_closed():
    """Both field generations at once: semantics unknown."""
    report = _current_report()
    report["trading"]["sharpe"] = 18.22
    with pytest.raises(regime.RegimeDataError, match="neither the old metric schema"):
        regime.validate_report_schema(report, "test")


def test_a_report_without_a_trading_section_fails_closed():
    with pytest.raises(regime.RegimeDataError, match="no trading report"):
        regime.validate_report_schema({"classification": {}}, "test")


# --- a genuine pre-correction run must stay auditable --------------------------
#
# The v2 and v3 run directories persisted `outer_predictions.parquet` beside a
# `walkforward.json` written by the *old* evaluator. After the metric
# correction, recomputing from those predictions produces the new schema:
# no `sharpe`, a new `per_trade_sharpe`, and a `max_drawdown` measured from
# starting capital. Comparing that against the recorded report field-for-field
# can never match — which would make every one of those runs unreadable exactly
# when we need to re-audit them.
#
# So the recomputed report is recast into the old schema, with both changed
# metrics rebuilt under their own historical formulas, and still compared.
def _legacy_prediction_frame() -> pd.DataFrame:
    """Enough trades that the legacy Sharpe has dispersion to measure."""
    rows = [10, 20, 30, 40]
    return pd.DataFrame(
        {
            "fold": [0] * len(rows),
            "seed": [42] * len(rows),
            "row_index": rows,
            "timestamp": pd.to_datetime(
                [f"2024-01-01T{10 + i:02d}:00:00Z" for i in range(len(rows))], utc=True
            ),
            "true_target": [SHORT_IDX, LONG_IDX, LONG_IDX, SHORT_IDX],
            # The first trade loses, so the pre-fix drawdown (peak from the
            # first completed trade) and the corrected one (peak from starting
            # capital) genuinely differ — otherwise the test below is vacuous.
            "future_return": [-0.04, 0.03, 0.01, -0.02],
            "p_short": [0.05, 0.05, 0.05, 0.05],
            "p_hold": [0.05, 0.05, 0.05, 0.05],
            "p_long": [0.90, 0.90, 0.90, 0.90],
            "selected_action": [LONG_IDX] * len(rows),
            "threshold": [0.50] * len(rows),
        }
    )


def _legacy_artifact(frame: pd.DataFrame) -> dict:
    """A walkforward.json in the exact pre-correction schema.

    Its trading report is built by taking the current evaluator's output and
    putting it back into the old shape: `sharpe` from the trade-capacity
    formula, `max_drawdown` from the pre-fix peak, and no candle-level fields.
    That is what those runs actually contain.
    """
    spec = TargetSpec()
    ordered = frame.sort_values("row_index")
    proba = ordered[["p_short", "p_hold", "p_long"]].to_numpy(dtype=np.float64)
    future_return = ordered["future_return"].to_numpy(dtype=np.float64)
    row_index = ordered["row_index"].to_numpy(dtype=np.int64)

    report = ev.evaluate(
        proba,
        ordered["true_target"].to_numpy(dtype=np.int64),
        future_return,
        spec,
        0.50,
        row_index=row_index,
    )
    _, _, trade_returns = ev.realised_trades(
        ev.signals_from_proba(proba, 0.50), future_return, spec, row_index=row_index
    )
    assert len(trade_returns) >= 2, "the fixture must produce a measurable legacy Sharpe"

    trading = dict(report["trading"])
    for field in (*regime.NON_REPRODUCIBLE_TRADING_METRICS, "per_trade_sharpe"):
        trading.pop(field, None)
    trading["sharpe"] = round(regime.legacy_sharpe(trade_returns, spec.horizon, 24 * 365), 4)
    trading["max_drawdown"] = round(regime.legacy_max_drawdown(trade_returns), 4)
    report = {**report, "trading": trading}

    return {
        "dataset": {"target_spec": spec.to_dict(), "timeframe": "1h"},
        "config": {"seed": 42},
        "sealed_test": {"start_row": 100, "evaluated": False},
        "test_evaluated": False,
        "outer_predictions": "outer_predictions.parquet",
        "folds": [
            {
                "fold": 0,
                "samples": {"outer_validation": len(frame)},
                "periods": {"outer_validation": {"row_range": [10, 50]}},
                "selection": {"threshold": 0.50},
                "outer_validation": {"mtst": report},
            }
        ],
    }


def test_a_genuine_legacy_run_still_binds_to_its_predictions(tmp_path):
    """The regression: this is what the v2/v3 directories look like."""
    frame = _legacy_prediction_frame()
    artifact = _legacy_artifact(frame)
    path = _write_run(tmp_path, frame, artifact)

    recorded = artifact["folds"][0]["outer_validation"]["mtst"]["trading"]
    assert "sharpe" in recorded and "annualised_sharpe" not in recorded
    assert (
        regime.validate_report_schema(
            artifact["folds"][0]["outer_validation"]["mtst"], "fixture"
        )
        == "legacy_pre_metric_correction"
    )

    loaded = load_predictions(path, sealed_test_start=100)
    assert len(loaded) == len(frame)


def test_a_legacy_run_with_a_tampered_net_return_still_fails_to_bind(tmp_path):
    """A reproducible field is still compared exactly."""
    frame = _legacy_prediction_frame()
    artifact = _legacy_artifact(frame)
    artifact["folds"][0]["outer_validation"]["mtst"]["trading"]["net_return"] += 0.05
    path = _write_run(tmp_path, frame, artifact)

    with pytest.raises(RegimeDataError, match="does not reproduce"):
        load_predictions(path, sealed_test_start=100)


def test_a_tampered_legacy_sharpe_is_caught(tmp_path):
    """Rebuilt under its old formula, so it is checked rather than dropped."""
    frame = _legacy_prediction_frame()
    artifact = _legacy_artifact(frame)
    artifact["folds"][0]["outer_validation"]["mtst"]["trading"]["sharpe"] = 18.22
    path = _write_run(tmp_path, frame, artifact)

    with pytest.raises(RegimeDataError, match="does not reproduce"):
        load_predictions(path, sealed_test_start=100)


def test_a_tampered_legacy_max_drawdown_is_caught(tmp_path):
    """Likewise: the old semantics are recomputed, not excluded."""
    frame = _legacy_prediction_frame()
    artifact = _legacy_artifact(frame)
    artifact["folds"][0]["outer_validation"]["mtst"]["trading"]["max_drawdown"] = 0.999
    path = _write_run(tmp_path, frame, artifact)

    with pytest.raises(RegimeDataError, match="does not reproduce"):
        load_predictions(path, sealed_test_start=100)


def test_a_legacy_artifact_carrying_the_corrected_max_drawdown_fails(tmp_path):
    """The two generations' values differ, and the check notices.

    Guards against the reverse mistake: a legacy artifact whose drawdown was
    quietly rewritten with the new semantics is not the run it claims to be.
    """
    frame = _legacy_prediction_frame()
    artifact = _legacy_artifact(frame)
    trading = artifact["folds"][0]["outer_validation"]["mtst"]["trading"]

    spec = TargetSpec()
    ordered = frame.sort_values("row_index")
    _, _, trade_returns = ev.realised_trades(
        ev.signals_from_proba(
            ordered[["p_short", "p_hold", "p_long"]].to_numpy(dtype=np.float64), 0.50
        ),
        ordered["future_return"].to_numpy(dtype=np.float64),
        spec,
        row_index=ordered["row_index"].to_numpy(dtype=np.int64),
    )
    corrected = round(ev.max_drawdown_of(np.cumprod(1.0 + trade_returns)), 4)
    assert corrected != trading["max_drawdown"], (
        "the fixture must make the two drawdown definitions differ, or this test "
        "proves nothing"
    )
    trading["max_drawdown"] = corrected
    path = _write_run(tmp_path, frame, artifact)

    with pytest.raises(RegimeDataError, match="does not reproduce"):
        load_predictions(path, sealed_test_start=100)


def test_a_current_run_still_binds_to_its_predictions(tmp_path):
    """The corrected path is unaffected by the legacy branch."""
    frame = _prediction_frame()
    path = _write_run(tmp_path, frame, _artifact(frame=frame))
    assert len(load_predictions(path, sealed_test_start=100)) == len(frame)
