"""Signal semantics and model metadata."""

from __future__ import annotations

import pytest

from chimera.contracts import (
    CLASS_ORDER,
    HOLD_IDX,
    LONG_IDX,
    SHORT_IDX,
    ModelMetadata,
    Signal,
    TargetSpec,
    decide,
)
from chimera.features import FeatureSpec, feature_columns


def test_class_order_is_short_hold_long():
    """Frozen by contract: the model head, the wire format, the metadata and
    the evaluation code all index probabilities this way."""
    assert [c.value for c in CLASS_ORDER] == ["SHORT", "HOLD", "LONG"]
    assert (SHORT_IDX, HOLD_IDX, LONG_IDX) == (0, 1, 2)


def test_cost_threshold_is_a_round_trip():
    spec = TargetSpec(fee_rate=0.001, slippage_rate=0.0005)
    assert spec.cost_threshold == pytest.approx(2 * 0.0015)


def test_target_spec_roundtrips():
    spec = TargetSpec(horizon=12, fee_rate=0.002, slippage_rate=0.001)
    assert TargetSpec.from_dict(spec.to_dict()) == spec


# --- decide ---------------------------------------------------------------
def test_confident_long_is_a_long():
    assert decide({"SHORT": 0.05, "HOLD": 0.20, "LONG": 0.75}, 0.6) is Signal.LONG


def test_confident_short_is_a_short():
    assert decide({"SHORT": 0.80, "HOLD": 0.15, "LONG": 0.05}, 0.6) is Signal.SHORT


def test_winning_class_below_the_threshold_is_a_hold():
    """A model that merely prefers LONG does not get to trade."""
    assert decide({"SHORT": 0.20, "HOLD": 0.35, "LONG": 0.45}, 0.6) is Signal.HOLD


def test_hold_majority_is_a_hold():
    assert decide({"SHORT": 0.10, "HOLD": 0.80, "LONG": 0.10}, 0.6) is Signal.HOLD


def test_threshold_is_inclusive():
    assert decide({"SHORT": 0.1, "HOLD": 0.3, "LONG": 0.6}, 0.6) is Signal.LONG


def test_a_tie_between_long_and_short_is_a_hold():
    assert decide({"SHORT": 0.45, "HOLD": 0.10, "LONG": 0.45}, 0.4) is Signal.HOLD


def test_missing_keys_degrade_to_hold():
    """Fail closed rather than raising inside a trading loop."""
    assert decide({}, 0.5) is Signal.HOLD
    assert decide({"HOLD": 1.0}, 0.5) is Signal.HOLD


def test_a_high_threshold_suppresses_everything():
    assert decide({"SHORT": 0.33, "HOLD": 0.33, "LONG": 0.34}, 0.99) is Signal.HOLD


# --- metadata --------------------------------------------------------------
def _metadata(**overrides) -> ModelMetadata:
    defaults = dict(
        model_version="v1",
        feature_names=feature_columns(),
        sequence_length=64,
        feature_spec=FeatureSpec(),
        target_spec=TargetSpec(),
        scaler_mean=[0.0] * len(feature_columns()),
        scaler_std=[1.0] * len(feature_columns()),
        decision_threshold=0.55,
    )
    defaults.update(overrides)
    return ModelMetadata(**defaults)


def test_metadata_roundtrips_through_json_shapes():
    original = _metadata(pair="BTC/USDT", timeframe="1h", exchange="binance")
    restored = ModelMetadata.from_dict(original.to_dict())
    assert restored.feature_names == original.feature_names
    assert restored.sequence_length == original.sequence_length
    assert restored.feature_spec == original.feature_spec
    assert restored.target_spec == original.target_spec
    assert restored.decision_threshold == original.decision_threshold
    assert restored.pair == "BTC/USDT"


def test_metadata_records_everything_inference_needs():
    data = _metadata().to_dict()
    for key in (
        "feature_names",
        "sequence_length",
        "scaler_mean",
        "scaler_std",
        "decision_threshold",
        "target_spec",
        "feature_spec",
        "model_version",
    ):
        assert key in data, f"metadata must persist {key}"


def test_n_features_follows_the_feature_list():
    assert _metadata().n_features == len(feature_columns())
