"""Prometheus metrics.

Every series defined here is actually written by code in this repository, and
the Grafana dashboards query only series defined here. If you add a panel, add
the metric first.

``prometheus_client`` is optional: when it is not installed the module degrades
to no-op stubs so that importing a strategy never fails because a monitoring
dependency is missing.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

try:  # pragma: no cover - exercised by whichever branch the env provides
    from prometheus_client import (
        REGISTRY,
        Counter,
        Gauge,
        Histogram,
        start_http_server,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:  # pragma: no cover
    PROMETHEUS_AVAILABLE = False

    class _Stub:
        """No-op stand-in with the subset of the client API we use."""

        def __init__(self, *_: Any, **__: Any) -> None:
            pass

        def labels(self, *_: Any, **__: Any) -> "_Stub":
            return self

        def inc(self, *_: Any, **__: Any) -> None:
            pass

        def set(self, *_: Any, **__: Any) -> None:
            pass

        def observe(self, *_: Any, **__: Any) -> None:
            pass

    Counter = Gauge = Histogram = _Stub  # type: ignore[assignment,misc]
    REGISTRY = None  # type: ignore[assignment]

    def start_http_server(*_: Any, **__: Any) -> None:  # type: ignore[misc]
        logger.warning("prometheus_client not installed; metrics server not started")


_PREFIX = "chimera"

# --- trading ------------------------------------------------------------
EQUITY = Gauge(f"{_PREFIX}_equity", "Account equity in stake currency")
PNL_TOTAL = Gauge(f"{_PREFIX}_pnl_total", "Cumulative realised PnL in stake currency")
DRAWDOWN = Gauge(f"{_PREFIX}_drawdown", "Current drawdown from peak equity (fraction)")
OPEN_POSITIONS = Gauge(f"{_PREFIX}_open_positions", "Number of open positions")
EXPOSURE = Gauge(f"{_PREFIX}_exposure", "Total exposure in stake currency")
TRADES_TOTAL = Counter(f"{_PREFIX}_trades_total", "Closed trades", ["result"])
REJECTED_ENTRIES = Counter(
    f"{_PREFIX}_rejected_entries_total",
    "Entry signals rejected before reaching the exchange",
    ["reason"],
)
RISK_HALTED = Gauge(f"{_PREFIX}_risk_halted", "1 when the risk kill switch is engaged")

# --- ML -----------------------------------------------------------------
INFERENCE_LATENCY = Histogram(
    f"{_PREFIX}_inference_latency_seconds",
    "Inference request latency",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
INFERENCE_REQUESTS = Counter(f"{_PREFIX}_inference_requests_total", "Inference requests")
INFERENCE_ERRORS = Counter(
    f"{_PREFIX}_inference_errors_total", "Failed inference requests", ["kind"]
)
PREDICTIONS = Counter(f"{_PREFIX}_predictions_total", "Predicted signals", ["signal"])
CONFIDENCE = Histogram(
    f"{_PREFIX}_prediction_confidence",
    "Winning-class probability of each prediction",
    buckets=(0.34, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
)
MODEL_INFO = Gauge(
    f"{_PREFIX}_model_info", "1 for the currently served model version", ["version"]
)
LAST_INFERENCE_TS = Gauge(
    f"{_PREFIX}_last_successful_inference_timestamp",
    "Unix timestamp of the last successful inference",
)

# --- futures execution (dry-run) ----------------------------------------
# Every label below is a bounded enum from chimera.futures: a side, a purpose, a
# state, an outcome. None of them carries a free-text reason, an order id, a
# price or a quantity — a label whose value set grows with traffic is a new time
# series per event, and Prometheus keeps them forever. The one exception is
# `symbol`, which is bounded by the configured whitelist in the same way
# DATA_DELAY's `pair` already is.
FUT_SIGNALS = Counter(
    f"{_PREFIX}_futures_signals_total",
    "Futures signals received, by what happened to them",
    ["outcome"],
)
FUT_RISK_VETOES = Counter(
    f"{_PREFIX}_futures_risk_vetoes_total",
    "Futures orders the risk engine refused, by collapsed reason",
    ["reason"],
)
FUT_ORDERS_PLANNED = Counter(
    f"{_PREFIX}_futures_orders_planned_total",
    "Futures orders planned, before the risk gate",
    ["side", "purpose"],
)
FUT_ORDERS_SUBMITTED = Counter(
    f"{_PREFIX}_futures_orders_submitted_total",
    "Futures orders submitted to the dry-run simulator",
    ["side", "purpose"],
)
FUT_ORDERS_REJECTED = Counter(
    f"{_PREFIX}_futures_orders_rejected_total",
    "Futures orders the venue or its constraints refused",
    ["reason"],
)
FUT_FILLS = Counter(
    f"{_PREFIX}_futures_fills_total",
    "Simulated futures fills, partial and final counted separately",
    ["side", "kind"],
)
FUT_SLIPPAGE_BPS = Histogram(
    f"{_PREFIX}_futures_slippage_bps",
    "Adverse distance from the reference price of each simulated fill, in bps",
    buckets=(0.0, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0),
)
FUT_TRADING_FEES = Counter(
    f"{_PREFIX}_futures_trading_fees_total", "Simulated futures trading fees paid"
)
FUT_FUNDING = Counter(
    f"{_PREFIX}_futures_funding_total",
    "Simulated funding, paid and received kept apart",
    ["direction"],
)
FUT_TURNOVER = Counter(
    f"{_PREFIX}_futures_turnover_total", "Notional traded by the futures executor"
)
FUT_POSITION_QUANTITY = Gauge(
    f"{_PREFIX}_futures_position_quantity",
    "Open futures position size, as a magnitude",
    ["symbol", "side"],
)
FUT_GROSS_EXPOSURE = Gauge(
    f"{_PREFIX}_futures_gross_exposure", "Absolute futures notional across positions"
)
FUT_NET_EXPOSURE = Gauge(
    f"{_PREFIX}_futures_net_exposure", "Signed futures notional: LONG adds, SHORT subtracts"
)
FUT_REALISED_PNL = Gauge(
    f"{_PREFIX}_futures_realised_pnl", "Cumulative simulated realised futures PnL"
)
FUT_NET_PNL = Gauge(
    f"{_PREFIX}_futures_net_pnl", "Realised PnL after trading fees and net funding"
)
FUT_DRAWDOWN = Gauge(
    f"{_PREFIX}_futures_drawdown", "Drawdown of simulated net PnL from its own peak"
)
FUT_RECONCILIATION = Counter(
    f"{_PREFIX}_futures_reconciliation_total",
    "Reconciliation attempts, by whether local and reported agreed",
    ["outcome"],
)
FUT_INVALID_TRANSITIONS = Counter(
    f"{_PREFIX}_futures_invalid_transitions_total",
    "Order state transitions refused by the state machine",
    ["from_state"],
)
FUT_EMERGENCY_FLATTEN = Counter(
    f"{_PREFIX}_futures_emergency_flatten_total",
    "Emergency flattens, by their declared cause",
    ["cause"],
)
FUT_RECOVERY = Counter(
    f"{_PREFIX}_futures_recovery_total",
    "Restart recoveries, by how the persisted state was found",
    ["outcome"],
)
FUT_EXECUTION_LATENCY = Histogram(
    f"{_PREFIX}_futures_execution_latency_seconds",
    "Wall time from signal received to the last simulated event applied",
    buckets=(0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0),
)


# --- system -------------------------------------------------------------
SERVICE_UP = Gauge(f"{_PREFIX}_service_up", "1 when the component is ready", ["component"])
API_FAILURES = Counter(
    f"{_PREFIX}_api_failures_total", "Outbound API call failures", ["component"]
)
DATA_DELAY = Gauge(
    f"{_PREFIX}_data_delay_seconds",
    "Seconds the newest candle is late past its close (0 when on time)",
    ["pair"],
)


def mark_inference_success(signal: str, confidence: float, latency_s: float) -> None:
    """Record one successful prediction across the ML metric family."""
    INFERENCE_REQUESTS.inc()
    INFERENCE_LATENCY.observe(latency_s)
    PREDICTIONS.labels(signal=signal).inc()
    CONFIDENCE.observe(confidence)
    LAST_INFERENCE_TS.set(time.time())


def mark_inference_failure(kind: str) -> None:
    INFERENCE_REQUESTS.inc()
    INFERENCE_ERRORS.labels(kind=kind).inc()


def serve_metrics(port: int) -> None:
    """Expose /metrics on ``port``. Safe to call when the client is missing."""
    start_http_server(port)
    logger.info("Prometheus metrics served on port %d", port)
