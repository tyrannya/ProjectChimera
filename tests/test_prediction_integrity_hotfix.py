import json

import numpy as np
import pandas as pd
import pytest

from chimera.contracts import LONG_IDX, SHORT_IDX, TargetSpec
from nn import evaluate as ev
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
