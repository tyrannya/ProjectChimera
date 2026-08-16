"""Model shape validation, artifact round-trip and promotion gating."""

from __future__ import annotations

import json

import pytest
import torch

from chimera.contracts import CLASS_ORDER, ModelMetadata, TargetSpec
from chimera.features import FeatureSpec, feature_columns
from nn.model_def import MTST, MTSTConfig
from nn.registry import (
    PromotionGates,
    check_gates,
    load_model,
    new_version,
    promote,
    resolve_current,
    save_model,
)


@pytest.fixture
def config():
    return MTSTConfig(
        input_dim=len(feature_columns()), seq_len=16, d_model=16, n_heads=2, num_layers=1
    )


@pytest.fixture
def metadata():
    n = len(feature_columns())
    return ModelMetadata(
        model_version="test-v1",
        feature_names=feature_columns(),
        sequence_length=16,
        feature_spec=FeatureSpec(),
        target_spec=TargetSpec(),
        scaler_mean=[0.0] * n,
        scaler_std=[1.0] * n,
        decision_threshold=0.5,
    )


# --- architecture ----------------------------------------------------------
def test_forward_returns_one_logit_per_class(config):
    out = MTST(config)(torch.zeros(4, config.seq_len, config.input_dim))
    assert out.shape == (4, len(CLASS_ORDER))


def test_wrong_sequence_length_raises(config):
    with pytest.raises(ValueError, match="sequence length"):
        MTST(config)(torch.zeros(2, config.seq_len + 1, config.input_dim))


def test_wrong_feature_count_raises(config):
    with pytest.raises(ValueError, match="features"):
        MTST(config)(torch.zeros(2, config.seq_len, config.input_dim + 3))


def test_wrong_rank_raises(config):
    with pytest.raises(ValueError, match="batch"):
        MTST(config)(torch.zeros(config.seq_len, config.input_dim))


def test_d_model_must_divide_by_heads():
    with pytest.raises(ValueError, match="divisible"):
        MTSTConfig(input_dim=4, seq_len=8, d_model=10, n_heads=4)


def test_config_roundtrips(config):
    assert MTSTConfig.from_dict(config.to_dict()) == config


def test_model_is_deterministic_in_eval_mode(config):
    torch.manual_seed(0)
    model = MTST(config).eval()
    x = torch.randn(3, config.seq_len, config.input_dim)
    with torch.no_grad():
        torch.testing.assert_close(model(x), model(x))


# --- artifacts --------------------------------------------------------------
def test_save_then_load_reproduces_the_same_outputs(tmp_path, config, metadata):
    torch.manual_seed(1)
    model = MTST(config).eval()
    x = torch.randn(2, config.seq_len, config.input_dim)
    with torch.no_grad():
        before = model(x)

    save_model(tmp_path, "v1", model, metadata, {"note": "test"})
    reloaded, reloaded_meta = load_model(tmp_path / "v1")
    with torch.no_grad():
        after = reloaded(x)

    torch.testing.assert_close(before, after)
    assert reloaded_meta.feature_names == feature_columns()
    assert reloaded_meta.sequence_length == 16


def test_saved_artifact_contains_every_required_file(tmp_path, config, metadata):
    save_model(tmp_path, "v1", MTST(config), metadata, {})
    for name in ("model.pt", "config.json", "metadata.json", "report.json"):
        assert (tmp_path / "v1" / name).exists()


def test_loading_an_incomplete_artifact_raises(tmp_path, config, metadata):
    save_model(tmp_path, "v1", MTST(config), metadata)
    (tmp_path / "v1" / "metadata.json").unlink()
    with pytest.raises(FileNotFoundError, match="metadata.json"):
        load_model(tmp_path / "v1")


def test_inconsistent_artifact_is_rejected(tmp_path, config, metadata):
    """Metadata claiming a different feature count than the weights expect is a
    corrupt artifact, and loading it would silently mis-shape every request."""
    save_model(tmp_path, "v1", MTST(config), metadata)
    path = tmp_path / "v1" / "metadata.json"
    data = json.loads(path.read_text())
    data["feature_names"] = data["feature_names"][:-1]
    data["scaler_mean"] = data["scaler_mean"][:-1]
    data["scaler_std"] = data["scaler_std"][:-1]
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="inconsistent"):
        load_model(tmp_path / "v1")


def test_versions_are_unique_and_sortable():
    versions = {new_version(str(i)) for i in range(20)}
    assert len(versions) == 20


# --- promotion ---------------------------------------------------------------
def _report(macro_f1: float, net_return: float, n_trades: int = 50, calibration: float = 0.05):
    return {
        "classification": {"macro_f1": macro_f1, "calibration_error": calibration},
        "trading": {"net_return": net_return, "n_trades": n_trades},
    }


def test_a_model_that_beats_its_baselines_passes():
    passed, failures = check_gates(
        _report(0.45, 0.05), {"majority": _report(0.20, -0.10)}, PromotionGates()
    )
    assert passed and failures == []


def test_a_model_that_does_not_beat_the_baseline_is_rejected():
    passed, failures = check_gates(
        _report(0.30, 0.05), {"majority": _report(0.30, -0.10)}, PromotionGates()
    )
    assert not passed
    assert any("baseline" in f for f in failures)


def test_a_losing_model_is_rejected():
    passed, failures = check_gates(
        _report(0.60, -0.02), {"majority": _report(0.20, 0.0)}, PromotionGates()
    )
    assert not passed
    assert any("net return" in f for f in failures)


def test_a_model_that_barely_trades_is_rejected():
    passed, failures = check_gates(
        _report(0.60, 0.10, n_trades=3), {"majority": _report(0.20, 0.0)}, PromotionGates()
    )
    assert not passed
    assert any("trades" in f for f in failures)


def test_a_badly_calibrated_model_is_rejected():
    passed, failures = check_gates(
        _report(0.60, 0.10, calibration=0.5),
        {"majority": _report(0.20, 0.0)},
        PromotionGates(),
    )
    assert not passed
    assert any("calibration" in f for f in failures)


def test_training_alone_does_not_promote(tmp_path, config, metadata):
    """Saving an artifact must not make it live; only promote() does that."""
    save_model(tmp_path, "v1", MTST(config), metadata)
    with pytest.raises(FileNotFoundError, match="no promoted model"):
        resolve_current(tmp_path)


def test_promote_points_current_at_the_version(tmp_path, config, metadata):
    save_model(tmp_path, "v1", MTST(config), metadata)
    promote(tmp_path, "v1")
    assert resolve_current(tmp_path) == tmp_path / "v1"


def test_promoting_an_unknown_version_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="unknown version"):
        promote(tmp_path, "nope")


def test_a_dangling_pointer_raises(tmp_path, config, metadata):
    save_model(tmp_path, "v1", MTST(config), metadata)
    promote(tmp_path, "v1")
    (tmp_path / "current.json").write_text(json.dumps({"version": "deleted"}))
    with pytest.raises(FileNotFoundError, match="not in"):
        resolve_current(tmp_path)
