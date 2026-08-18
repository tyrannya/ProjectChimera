"""Client for the inference service, used by the Freqtrade strategy.

Two rules govern everything here:

*Fail closed.* Every failure mode — connection refused, timeout, HTTP error,
malformed body, missing field, unknown signal — returns ``None``. The caller
treats ``None`` as HOLD. The client never invents a probability, and never
substitutes a different decision procedure for the one it could not reach.

*One call per candle.* Predictions are cached on
``(pair, timeframe, candle timestamp, model version)``. Freqtrade re-analyses
the whole dataframe on every bot iteration, so without this the service would
be hit several times per candle for an answer that cannot have changed.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Sequence

logger = logging.getLogger(__name__)

VALID_SIGNALS = ("SHORT", "HOLD", "LONG")


@dataclass(frozen=True)
class ServerInfo:
    """What ``/readyz`` reports about the currently served model.

    Carries the feature contract as well as the shape, so the caller can build
    exactly the features the served model was trained on and refuse to guess
    when its own code disagrees.
    """

    model_version: str
    sequence_length: int
    n_features: int
    decision_threshold: float
    feature_names: tuple[str, ...] = ()
    feature_spec: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class Prediction:
    """A validated prediction. Constructing one means the response was sound."""

    model_version: str
    signal: str
    probabilities: dict[str, float]
    confidence: float
    decision_threshold: float
    received_at: float


class InferenceClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = 2.0,
        retries: int = 1,
        cache_size: int = 256,
        describe_ttl_s: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        # Bounded on purpose. An unbounded retry loop against a service that is
        # down turns one stalled candle into a stalled bot.
        self.retries = max(0, min(retries, 3))
        self.cache_size = cache_size
        self.describe_ttl_s = describe_ttl_s
        self._cache: OrderedDict[tuple, Prediction] = OrderedDict()
        self._info: ServerInfo | None = None
        self._info_at: float = 0.0

    # ------------------------------------------------------------------
    def describe(self, force: bool = False) -> ServerInfo | None:
        """Ask ``/readyz`` for the served model's shape. ``None`` if unavailable.

        The strategy builds its feature window from this, so the window can
        never be a different length than the model expects.
        """
        now = time.monotonic()
        if not force and self._info is not None and now - self._info_at < self.describe_ttl_s:
            return self._info

        payload = self._get("/readyz")
        if payload is None:
            self._info = None
            return None
        try:
            info = ServerInfo(
                model_version=str(payload["model_version"]),
                sequence_length=int(payload["sequence_length"]),
                n_features=int(payload["n_features"]),
                decision_threshold=float(payload["decision_threshold"]),
                feature_names=tuple(str(n) for n in payload.get("feature_names", ())),
                feature_spec={
                    str(k): int(v) for k, v in dict(payload.get("feature_spec", {})).items()
                },
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("Malformed /readyz response: %s", exc)
            self._info = None
            return None

        if info.sequence_length < 1 or info.n_features < 1:
            logger.warning("Nonsensical model shape from /readyz: %s", info)
            self._info = None
            return None

        self._info = info
        self._info_at = now
        return info

    def predict(
        self,
        pair: str,
        timeframe: str,
        timestamp: datetime,
        features: Sequence[Sequence[float]],
    ) -> Prediction | None:
        """Predict for one candle. ``None`` means "no usable answer" — hold."""
        info = self.describe()
        if info is None:
            # Without a readiness answer we do not know which model would reply,
            # so the prediction could not be cached correctly or attributed to a
            # version. A service that cannot say it is ready is not usable.
            logger.warning("Inference service is not ready; not predicting for %s", pair)
            return None
        key = (pair, timeframe, timestamp.isoformat(), info.model_version)

        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached

        payload = self._post(
            "/predict",
            {
                "pair": pair,
                "timeframe": timeframe,
                "timestamp": timestamp.isoformat(),
                "features": [[float(v) for v in row] for row in features],
            },
        )
        if payload is None:
            return None

        prediction = self._parse(payload)
        if prediction is None:
            return None

        self._cache[key] = prediction
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return prediction

    # ------------------------------------------------------------------
    @staticmethod
    def _parse(payload: Any) -> Prediction | None:
        """Validate a response body. Anything unexpected yields ``None``."""
        if not isinstance(payload, dict):
            logger.warning("Inference response is not an object")
            return None
        try:
            signal = str(payload["signal"])
            if signal not in VALID_SIGNALS:
                logger.warning("Unknown signal in response: %r", signal)
                return None

            raw = payload["probabilities"]
            if not isinstance(raw, dict):
                logger.warning("probabilities is not an object")
                return None
            probabilities = {name: float(raw[name]) for name in VALID_SIGNALS}

            total = sum(probabilities.values())
            if not 0.98 <= total <= 1.02:
                logger.warning("probabilities sum to %.4f, not 1", total)
                return None

            return Prediction(
                model_version=str(payload["model_version"]),
                signal=signal,
                probabilities=probabilities,
                confidence=float(payload["confidence"]),
                decision_threshold=float(payload["decision_threshold"]),
                received_at=time.time(),
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("Malformed inference response: %s", exc)
            return None

    def _get(self, path: str) -> Any | None:
        return self._request("GET", path, None)

    def _post(self, path: str, body: dict[str, Any]) -> Any | None:
        return self._request("POST", path, body)

    def _request(self, method: str, path: str, body: dict[str, Any] | None) -> Any | None:
        import requests

        url = f"{self.base_url}{path}"
        for attempt in range(self.retries + 1):
            try:
                response = requests.request(method, url, json=body, timeout=self.timeout_s)
            except requests.RequestException as exc:
                # Transport failures are the only thing worth retrying.
                if attempt < self.retries:
                    logger.debug("%s %s failed (%s), retrying", method, path, exc)
                    continue
                logger.warning("%s %s failed: %s", method, path, type(exc).__name__)
                return None

            if response.status_code >= 500:
                if attempt < self.retries:
                    continue
                logger.warning("%s %s returned %d", method, path, response.status_code)
                return None
            if response.status_code >= 400:
                # A 4xx is our fault; retrying sends the same bad request again.
                logger.warning(
                    "%s %s rejected with %d: %s",
                    method,
                    path,
                    response.status_code,
                    response.text[:300],
                )
                return None

            try:
                return response.json()
            except ValueError:
                logger.warning("%s %s returned a non-JSON body", method, path)
                return None
        return None
