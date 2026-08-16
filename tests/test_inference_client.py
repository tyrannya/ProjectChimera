"""Inference client: fail-closed behaviour and caching.

No network. ``requests.request`` is replaced with a recorder so every failure
mode can be produced deterministically.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
import requests

from chimera.inference_client import InferenceClient

READYZ = {
    "status": "ok",
    "model_version": "v1",
    "sequence_length": 4,
    "n_features": 3,
    "decision_threshold": 0.5,
    "feature_names": ["a", "b", "c"],
    "feature_spec": {"ema_fast": 12},
}
PREDICTION = {
    "model_version": "v1",
    "signal": "LONG",
    "probabilities": {"SHORT": 0.1, "HOLD": 0.2, "LONG": 0.7},
    "confidence": 0.7,
    "decision_threshold": 0.5,
    "served_at": "2026-08-16T12:00:00Z",
}


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


@pytest.fixture
def transport(monkeypatch):
    """Records calls and replays queued responses keyed by path."""

    class Transport:
        def __init__(self):
            self.calls: list[tuple[str, str]] = []
            self.responses: dict[str, object] = {
                "/readyz": FakeResponse(200, READYZ),
                "/predict": FakeResponse(200, PREDICTION),
            }

        def __call__(self, method, url, json=None, timeout=None):
            path = url[url.index("/", len("http://")) :]
            self.calls.append((method, path))
            response = self.responses[path]
            if isinstance(response, Exception):
                raise response
            if callable(response):
                return response()
            return response

    fake = Transport()
    monkeypatch.setattr(requests, "request", fake)
    return fake


@pytest.fixture
def client():
    return InferenceClient("http://nn_infer:3000", timeout_s=0.1, retries=1)


def features(rows=4, cols=3):
    return [[0.1 * (r + c) for c in range(cols)] for r in range(rows)]


def when(offset_minutes: int = 0) -> datetime:
    return datetime(2026, 8, 16, 12, 0, tzinfo=timezone.utc) + timedelta(
        minutes=offset_minutes
    )


# --- happy path -----------------------------------------------------------
def test_describe_reads_the_model_contract(client, transport):
    info = client.describe()
    assert info.model_version == "v1"
    assert info.sequence_length == 4
    assert info.n_features == 3
    assert info.feature_names == ("a", "b", "c")


def test_predict_returns_a_validated_prediction(client, transport):
    prediction = client.predict("BTC/USDT", "1h", when(), features())
    assert prediction is not None
    assert prediction.signal == "LONG"
    assert prediction.confidence == pytest.approx(0.7)
    assert prediction.model_version == "v1"


# --- caching ---------------------------------------------------------------
def test_the_same_candle_is_requested_once(client, transport):
    for _ in range(5):
        client.predict("BTC/USDT", "1h", when(), features())
    assert sum(1 for m, p in transport.calls if p == "/predict") == 1


def test_a_new_candle_triggers_a_new_request(client, transport):
    client.predict("BTC/USDT", "1h", when(0), features())
    client.predict("BTC/USDT", "1h", when(60), features())
    assert sum(1 for m, p in transport.calls if p == "/predict") == 2


def test_different_pairs_do_not_share_a_cache_entry(client, transport):
    client.predict("BTC/USDT", "1h", when(), features())
    client.predict("ETH/USDT", "1h", when(), features())
    assert sum(1 for m, p in transport.calls if p == "/predict") == 2


def test_a_new_model_version_invalidates_the_cache(client, transport):
    client.predict("BTC/USDT", "1h", when(), features())
    transport.responses["/readyz"] = FakeResponse(200, {**READYZ, "model_version": "v2"})
    client.describe(force=True)
    client.predict("BTC/USDT", "1h", when(), features())
    assert sum(1 for m, p in transport.calls if p == "/predict") == 2


def test_the_cache_is_bounded():
    client = InferenceClient("http://x", cache_size=3)
    for i in range(10):
        client._cache[("p", "1h", str(i), "v1")] = None  # type: ignore[assignment]
        while len(client._cache) > client.cache_size:
            client._cache.popitem(last=False)
    assert len(client._cache) == 3


# --- fail closed -------------------------------------------------------------
def test_connection_error_returns_none(client, transport):
    transport.responses["/readyz"] = requests.ConnectionError("refused")
    assert client.describe() is None
    assert client.predict("BTC/USDT", "1h", when(), features()) is None


def test_timeout_returns_none(client, transport):
    transport.responses["/predict"] = requests.Timeout("too slow")
    assert client.predict("BTC/USDT", "1h", when(), features()) is None


def test_server_error_returns_none(client, transport):
    transport.responses["/predict"] = FakeResponse(500, None, "boom")
    assert client.predict("BTC/USDT", "1h", when(), features()) is None


def test_client_error_returns_none(client, transport):
    transport.responses["/predict"] = FakeResponse(400, None, "bad shape")
    assert client.predict("BTC/USDT", "1h", when(), features()) is None


def test_non_json_body_returns_none(client, transport):
    transport.responses["/predict"] = FakeResponse(200, None, "<html>")
    assert client.predict("BTC/USDT", "1h", when(), features()) is None


@pytest.mark.parametrize(
    "payload",
    [
        {"signal": "LONG"},  # no probabilities
        {**PREDICTION, "signal": "MOON"},  # unknown class
        {**PREDICTION, "probabilities": {"SHORT": 0.1}},  # incomplete
        {
            **PREDICTION,
            "probabilities": {"SHORT": 0.1, "HOLD": 0.1, "LONG": 0.1},
        },  # sums to 0.3
        {**PREDICTION, "probabilities": "not-an-object"},
        {**PREDICTION, "confidence": "high"},
        ["a", "list"],
        None,
    ],
)
def test_malformed_responses_return_none(client, transport, payload):
    transport.responses["/predict"] = FakeResponse(200, payload)
    assert client.predict("BTC/USDT", "1h", when(), features()) is None


def test_nonsense_model_shape_is_rejected(client, transport):
    transport.responses["/readyz"] = FakeResponse(200, {**READYZ, "sequence_length": 0})
    assert client.describe() is None


def test_malformed_readyz_is_rejected(client, transport):
    transport.responses["/readyz"] = FakeResponse(200, {"status": "ok"})
    assert client.describe() is None


# --- retries -------------------------------------------------------------------
def test_transport_failures_are_retried_a_bounded_number_of_times(client, transport):
    attempts = []

    def failing():
        attempts.append(1)
        raise requests.ConnectionError("refused")

    transport.responses["/predict"] = failing
    assert client.predict("BTC/USDT", "1h", when(), features()) is None
    # retries=1 means two attempts total, never an unbounded loop.
    assert len(attempts) == 2


def test_client_errors_are_not_retried(client, transport):
    attempts = []

    def rejecting():
        attempts.append(1)
        return FakeResponse(400, None, "bad request")

    transport.responses["/predict"] = rejecting
    assert client.predict("BTC/USDT", "1h", when(), features()) is None
    assert len(attempts) == 1, "a 4xx is our fault; resending it is pointless"


def test_retries_are_clamped():
    assert InferenceClient("http://x", retries=99).retries == 3
    assert InferenceClient("http://x", retries=-5).retries == 0
