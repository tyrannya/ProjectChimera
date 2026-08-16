"""HTTP inference service.

    uvicorn nn.infer_service:app --host 0.0.0.0 --port 3000

Contract
--------

``POST /predict``::

    {
      "pair": "BTC/USDT",
      "timeframe": "1h",
      "timestamp": "2026-08-16T12:00:00Z",
      "features": [[...], ...]          # (sequence_length, n_features), RAW
    }

    -> {
      "model_version": "20260816T120000Z-a1b2c3",
      "signal": "LONG",
      "probabilities": {"SHORT": 0.08, "HOLD": 0.21, "LONG": 0.71},
      "confidence": 0.71,
      "decision_threshold": 0.55,
      "served_at": "2026-08-16T12:00:00.123456+00:00"
    }

**The client sends raw, unscaled features.** Standardisation uses the scaler
stored in the model's metadata and is applied here, so the strategy cannot
apply the wrong scaler — or forget to.

Status codes are meaningful: ``422`` for a body that does not parse, ``400``
for a well-formed body whose matrix disagrees with the model's declared shape,
``503`` when no model is loaded, ``500`` for an inference failure. A failure
never returns a fabricated score.

Why FastAPI and not BentoML
---------------------------
The previous service loaded its model at import time from the BentoML store,
which made it impossible to import — and therefore to test — without a
populated store, and added a third model-versioning system alongside MLflow and
the on-disk artifacts. FastAPI and Pydantic are already in Freqtrade's own
dependency tree, give schema validation and correct status codes for free, and
let the whole contract be exercised by ``TestClient`` with a tiny model.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from chimera import metrics
from chimera.contracts import CLASS_ORDER, ModelMetadata, Signal, decide
from chimera.notify import TelegramNotifier
from nn.registry import DEFAULT_MODELS_DIR, load_model, resolve_current

logger = logging.getLogger(__name__)

MAX_SEQUENCE_ROWS = 4096


class PredictRequest(BaseModel):
    """One prediction request. Extra fields are rejected, not ignored."""

    model_config = {"extra": "forbid"}

    pair: str = Field(min_length=1, max_length=32)
    timeframe: str = Field(min_length=2, max_length=8)
    timestamp: datetime
    features: list[list[float]] = Field(min_length=1)

    @field_validator("features")
    @classmethod
    def _rectangular_and_finite(cls, value: list[list[float]]) -> list[list[float]]:
        if len(value) > MAX_SEQUENCE_ROWS:
            raise ValueError(f"at most {MAX_SEQUENCE_ROWS} rows allowed")
        width = len(value[0])
        if width == 0:
            raise ValueError("feature rows must not be empty")
        for i, row in enumerate(value):
            if len(row) != width:
                raise ValueError(f"row {i} has {len(row)} values, expected {width}")
        array = np.asarray(value, dtype=np.float64)
        if not np.isfinite(array).all():
            raise ValueError("features must be finite (no NaN or inf)")
        return value


class PredictResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_version: str
    signal: Signal
    probabilities: dict[str, float]
    confidence: float
    decision_threshold: float
    served_at: datetime


class ModelHolder:
    """Owns the loaded model. Kept off module scope so tests can inject one."""

    def __init__(self) -> None:
        self.model: Any = None
        self.metadata: ModelMetadata | None = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    @property
    def ready(self) -> bool:
        return self.model is not None and self.metadata is not None

    def load(self, model_dir: str | Path) -> None:
        model, metadata = load_model(model_dir)
        self.model = model
        self.metadata = metadata
        self._mean = np.asarray(metadata.scaler_mean, dtype=np.float64)
        self._std = np.asarray(metadata.scaler_std, dtype=np.float64)
        if self._mean.shape != (metadata.n_features,) or self._std.shape != (
            metadata.n_features,
        ):
            raise ValueError("scaler parameters do not match the feature count")
        metrics.MODEL_INFO.labels(version=metadata.model_version).set(1)
        logger.info(
            "Loaded model %s: %d features, sequence length %d, threshold %.2f",
            metadata.model_version,
            metadata.n_features,
            metadata.sequence_length,
            metadata.decision_threshold,
        )

    def infer(self, features: np.ndarray) -> np.ndarray:
        """Scale, run the model, return probabilities in ``CLASS_ORDER``."""
        assert self.metadata is not None and self._mean is not None
        scaled = ((features - self._mean) / self._std).astype(np.float32)
        with torch.no_grad():
            logits = self.model(torch.from_numpy(scaled[None, ...]))
            proba = torch.softmax(logits.float(), dim=-1)[0].numpy()
        return proba.astype(np.float64)


holder = ModelHolder()
notifier = TelegramNotifier()


def _model_dir() -> Path:
    """Where to load from: an explicit override, else the promoted model."""
    override = os.environ.get("CHIMERA_MODEL_DIR")
    if override:
        return Path(override)
    return resolve_current(os.environ.get("CHIMERA_MODELS_DIR", str(DEFAULT_MODELS_DIR)))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model at startup, but do not crash the process if it is absent.

    A service that cannot load a model still starts and answers ``/livez``;
    ``/readyz`` reports it as not ready, so an orchestrator can distinguish
    "starting" from "broken" instead of watching a container crash-loop.
    """
    metrics.SERVICE_UP.labels(component="inference").set(0)
    try:
        holder.load(_model_dir())
        metrics.SERVICE_UP.labels(component="inference").set(1)
        notifier.send_event(
            "Inference service started", f"model {holder.metadata.model_version}"
        )
    except Exception as exc:  # noqa: BLE001 - reported through /readyz
        logger.error("Could not load a model at startup: %s", exc)
        notifier.send_error("Inference service started without a model", str(exc))
    yield
    metrics.SERVICE_UP.labels(component="inference").set(0)
    notifier.send_event("Inference service stopping")


app = FastAPI(title="ProjectChimera inference", version="1.0.0", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    """Return a clean 422 that does not echo the rejected input back.

    FastAPI's default handler includes the offending value under ``input``.
    That breaks twice: a body containing the non-standard JSON literal ``NaN``
    parses fine on the way in but cannot be re-serialised on the way out, so
    the service answers a malformed request with an unhandled encoder crash
    instead of a 422; and echoing a whole feature matrix back makes error
    responses enormous. Only the location and the message are returned.
    """
    return JSONResponse(
        status_code=422,
        content={
            "detail": [
                {"loc": [str(part) for part in err.get("loc", ())], "msg": err.get("msg", "")}
                for err in exc.errors()
            ]
        },
    )


@app.get("/livez")
async def livez() -> dict[str, str]:
    """Liveness: the process is running. Says nothing about the model."""
    return {"status": "ok"}


@app.get("/readyz")
async def readyz(response: Response) -> dict[str, Any]:
    """Readiness: model loaded, metadata present, and a dummy inference passes.

    Deliberately silent — no notification is sent from here. A health endpoint
    is polled continuously, and alerting from it produces a message storm the
    moment anything wobbles. Startup and shutdown alert; probes do not.
    """
    if not holder.ready:
        response.status_code = 503
        return {"status": "not ready", "reason": "no model loaded"}

    metadata = holder.metadata
    assert metadata is not None
    try:
        dummy = np.zeros((metadata.sequence_length, metadata.n_features), dtype=np.float64)
        proba = holder.infer(dummy)
        if proba.shape != (len(CLASS_ORDER),) or not np.isfinite(proba).all():
            raise ValueError(f"dummy inference returned {proba!r}")
    except Exception as exc:  # noqa: BLE001
        logger.error("Readiness check failed: %s", exc)
        response.status_code = 503
        return {"status": "not ready", "reason": f"dummy inference failed: {exc}"}

    return {
        "status": "ok",
        "model_version": metadata.model_version,
        "sequence_length": metadata.sequence_length,
        "n_features": metadata.n_features,
        "decision_threshold": metadata.decision_threshold,
        # The client builds its feature window from these, so a strategy can
        # never send features computed with different windows than the model
        # was trained on.
        "feature_names": list(metadata.feature_names),
        "feature_spec": metadata.feature_spec.to_dict(),
    }


@app.get("/metrics")
async def prometheus_metrics() -> Response:
    if not metrics.PROMETHEUS_AVAILABLE:
        raise HTTPException(status_code=501, detail="prometheus_client is not installed")
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    start = time.monotonic()

    if not holder.ready:
        metrics.mark_inference_failure("no_model")
        raise HTTPException(status_code=503, detail="no model loaded")

    metadata = holder.metadata
    assert metadata is not None

    features = np.asarray(request.features, dtype=np.float64)
    expected = (metadata.sequence_length, metadata.n_features)
    if features.shape != expected:
        metrics.mark_inference_failure("bad_shape")
        raise HTTPException(
            status_code=400,
            detail=(
                f"features must have shape {expected} "
                f"(sequence_length, n_features), got {features.shape}. "
                f"Expected feature order: {metadata.feature_names}"
            ),
        )

    try:
        proba = holder.infer(features)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Inference failed for %s", request.pair)
        metrics.mark_inference_failure("inference_error")
        notifier.send_error("Inference failure", f"{type(exc).__name__}: {exc}")
        # No fabricated score: the caller must be able to tell failure from a
        # confident HOLD, because those two mean different things to the risk layer.
        raise HTTPException(status_code=500, detail="inference failed") from exc

    probabilities = {cls.value: round(float(p), 6) for cls, p in zip(CLASS_ORDER, proba)}
    signal = decide(probabilities, metadata.decision_threshold)
    confidence = float(max(probabilities.values()))

    metrics.mark_inference_success(signal.value, confidence, time.monotonic() - start)

    return PredictResponse(
        model_version=metadata.model_version,
        signal=signal,
        probabilities=probabilities,
        confidence=confidence,
        decision_threshold=metadata.decision_threshold,
        served_at=datetime.now(timezone.utc),
    )
