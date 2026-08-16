"""Inference HTTP contract.

Every test runs against a real (tiny) model artifact through FastAPI's
TestClient, so the request schema, the status codes and the response shape are
all exercised as the strategy will meet them.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from chimera.contracts import CLASS_ORDER, ModelMetadata, TargetSpec
from chimera.features import FeatureSpec, feature_columns
from nn.model_def import MTST, MTSTConfig
from nn.registry import promote, save_model

SEQ_LEN = 12
N_FEATURES = len(feature_columns())


@pytest.fixture
def models_dir(tmp_path):
    """A promoted, loadable model artifact."""
    torch.manual_seed(0)
    config = MTSTConfig(
        input_dim=N_FEATURES, seq_len=SEQ_LEN, d_model=16, n_heads=2, num_layers=1
    )
    metadata = ModelMetadata(
        model_version="test-v1",
        feature_names=feature_columns(),
        sequence_length=SEQ_LEN,
        feature_spec=FeatureSpec(),
        target_spec=TargetSpec(),
        scaler_mean=[0.0] * N_FEATURES,
        scaler_std=[1.0] * N_FEATURES,
        decision_threshold=0.4,
        pair="BTC/USDT",
        timeframe="1h",
    )
    save_model(tmp_path, "test-v1", MTST(config).eval(), metadata)
    promote(tmp_path, "test-v1")
    return tmp_path


@pytest.fixture
def client(models_dir, monkeypatch):
    monkeypatch.setenv("CHIMERA_MODELS_DIR", str(models_dir))
    monkeypatch.delenv("CHIMERA_MODEL_DIR", raising=False)
    # Import inside the fixture so the module-level holder is fresh per test run.
    from nn import infer_service

    infer_service.holder.__init__()
    with TestClient(infer_service.app) as test_client:
        yield test_client


def valid_body(rows: int = SEQ_LEN, cols: int = N_FEATURES) -> dict:
    rng = np.random.default_rng(3)
    return {
        "pair": "BTC/USDT",
        "timeframe": "1h",
        "timestamp": "2026-08-16T12:00:00Z",
        "features": rng.normal(0, 0.01, (rows, cols)).tolist(),
    }


# --- health ---------------------------------------------------------------
def test_livez_is_up(client):
    response = client.get("/livez")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_readyz_reports_the_model_contract(client):
    payload = client.get("/readyz").json()
    assert payload["status"] == "ok"
    assert payload["model_version"] == "test-v1"
    assert payload["sequence_length"] == SEQ_LEN
    assert payload["n_features"] == N_FEATURES
    assert payload["feature_names"] == feature_columns()
    assert payload["decision_threshold"] == 0.4


def test_readyz_is_503_without_a_model(tmp_path, monkeypatch):
    monkeypatch.setenv("CHIMERA_MODELS_DIR", str(tmp_path))
    monkeypatch.delenv("CHIMERA_MODEL_DIR", raising=False)
    from nn import infer_service

    infer_service.holder.__init__()
    with TestClient(infer_service.app) as client:
        assert client.get("/livez").status_code == 200, "livez must not depend on a model"
        assert client.get("/readyz").status_code == 503
        assert client.post("/predict", json=valid_body()).status_code == 503


# --- happy path -------------------------------------------------------------
def test_valid_request_returns_the_documented_shape(client):
    response = client.post("/predict", json=valid_body())
    assert response.status_code == 200

    payload = response.json()
    assert set(payload) == {
        "model_version",
        "signal",
        "probabilities",
        "confidence",
        "decision_threshold",
        "served_at",
    }
    assert payload["signal"] in [c.value for c in CLASS_ORDER]
    assert set(payload["probabilities"]) == {"SHORT", "HOLD", "LONG"}
    assert sum(payload["probabilities"].values()) == pytest.approx(1.0, abs=1e-4)
    assert payload["confidence"] == pytest.approx(max(payload["probabilities"].values()))
    assert payload["model_version"] == "test-v1"


def test_the_service_applies_its_own_scaler(client, models_dir):
    """Clients send raw features; a different scaler must change the answer.

    This is what stops the strategy and the model from disagreeing about
    normalisation without anyone noticing.
    """
    body = valid_body()
    first = client.post("/predict", json=body).json()["probabilities"]

    path = models_dir / "test-v1" / "metadata.json"
    data = json.loads(path.read_text())
    data["scaler_mean"] = [5.0] * N_FEATURES
    path.write_text(json.dumps(data))

    from nn import infer_service

    infer_service.holder.load(models_dir / "test-v1")
    second = client.post("/predict", json=body).json()["probabilities"]
    assert first != second


def test_identical_requests_are_deterministic(client):
    body = valid_body()
    assert (
        client.post("/predict", json=body).json()["probabilities"]
        == client.post("/predict", json=body).json()["probabilities"]
    )


# --- rejection ----------------------------------------------------------------
def test_too_few_rows_is_a_400(client):
    response = client.post("/predict", json=valid_body(rows=SEQ_LEN - 1))
    assert response.status_code == 400
    assert "shape" in response.json()["detail"]


def test_too_many_rows_is_a_400(client):
    assert client.post("/predict", json=valid_body(rows=SEQ_LEN + 1)).status_code == 400


def test_wrong_feature_count_is_a_400(client):
    response = client.post("/predict", json=valid_body(cols=N_FEATURES - 2))
    assert response.status_code == 400
    # The error tells the caller what the model actually wants.
    assert "ret_1" in response.json()["detail"]


def test_missing_fields_are_a_422(client):
    assert client.post("/predict", json={"pair": "BTC/USDT"}).status_code == 422


def test_unknown_fields_are_a_422(client):
    body = valid_body()
    body["surprise"] = 1
    assert client.post("/predict", json=body).status_code == 422


def test_ragged_features_are_a_422(client):
    body = valid_body()
    body["features"][0] = body["features"][0][:-1]
    assert client.post("/predict", json=body).status_code == 422


def test_non_finite_features_are_rejected(client):
    """NaN in, no answer out — it would propagate silently through the model.

    Sent as a raw body because ``NaN`` is not legal JSON and most client
    encoders refuse to emit it; Python's ``json.loads`` accepts it on the way
    in, so the server has to be the one that says no.
    """
    body = valid_body()
    body["features"][0][0] = float("nan")
    raw = json.dumps(body)  # emits the non-standard NaN literal
    assert "NaN" in raw
    response = client.post(
        "/predict", content=raw, headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422


def test_empty_features_are_a_422(client):
    body = valid_body()
    body["features"] = []
    assert client.post("/predict", json=body).status_code == 422


def test_garbage_body_is_a_422(client):
    assert client.post("/predict", json={"nonsense": True}).status_code == 422


def test_inference_failure_is_a_500_with_no_fabricated_score(client, monkeypatch):
    from nn import infer_service

    monkeypatch.setattr(
        infer_service.holder,
        "infer",
        lambda *_: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    response = client.post("/predict", json=valid_body())
    assert response.status_code == 500
    assert "signal" not in response.json()
    assert "probabilities" not in response.json()


def test_metrics_endpoint_exposes_the_series(client):
    client.post("/predict", json=valid_body())
    body = client.get("/metrics").text
    assert "chimera_inference_requests_total" in body
    assert "chimera_predictions_total" in body
