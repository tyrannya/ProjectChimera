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
# `symbol` on FUT_POSITION_QUANTITY, and unlike DATA_DELAY's `pair` there is no
# configured whitelist behind it. An *order* is bounded: it can only be planned
# for a symbol the venue's constraint table has, because
# StaticConstraintSource.constraints refuses an unknown one. The gauge is wider,
# because the executor publishes it from `store.state.positions`, which can also
# hold a symbol adopted at bootstrap from the venue's reported positions or
# restored from a state file — neither is checked against the constraint table.
# What keeps it finite is that a value appears only where a position exists, so
# the set is the symbols the account has held, not one series per event.
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


# --- trading modes ------------------------------------------------------
#
# Every label here is a bounded enum from `chimera.modes`: a TradingMode value
# or a ReasonCode value. Neither grows with traffic, and neither carries a
# free-text reason, an order id, a price or a quantity — the same rule the
# futures block above states. The one thing deliberately *absent* is any
# per-mode return: a mode metric that reported how well a mode had been doing
# would be the profit-based selection input the scaffold exists to not have.
# Exposure, turnover, fees, funding and drawdown are already published by the
# futures family, unlabelled, and stay that way for exactly one active mode.
MODE_SELECTED = Gauge(
    f"{_PREFIX}_mode_selected",
    "1 for the currently active trading mode, 0 for every other",
    ["mode"],
)
MODE_ELIGIBLE = Gauge(
    f"{_PREFIX}_mode_eligible",
    "1 when a mode's specialists are all screened and viable, 0 otherwise",
    ["mode"],
)
MODE_DECISIONS = Counter(
    f"{_PREFIX}_mode_decisions_total",
    "Mode decisions taken, by resulting mode and reason",
    ["mode", "reason"],
)
MODE_TRANSITIONS = Counter(
    f"{_PREFIX}_mode_transitions_total",
    "Mode changes, by origin and destination",
    ["from_mode", "to_mode"],
)
MODE_TRANSITION_FLATTENS = Counter(
    f"{_PREFIX}_mode_transition_flattens_total",
    "Mode changes that required flattening an inherited position first",
    ["from_mode", "to_mode"],
)
MODE_CONSENSUS_STATE = Counter(
    f"{_PREFIX}_mode_consensus_state_total",
    "Consensus outcomes inside an eligible mode, by mode and reason",
    ["mode", "reason"],
)
MODE_RISK_VETOES = Counter(
    f"{_PREFIX}_mode_risk_vetoes_total",
    "Aegis vetoes attributed to the mode that was active",
    ["mode", "reason"],
)


# --- prospective recorder -----------------------------------------------
#
# Section 4.8 of the adopted demo plan, in full. The only label anywhere in this
# family is `stream`, whose value set is the six stream ids the gen3 recorder
# contract declares — bounded by a committed file, not by traffic. There is no
# price here, no return, no funding flow and no basis: every series below counts
# observations, connections, files and clocks, which is the whole of what a
# recorder knows. A series reporting how a recorded price had *moved* would be
# an economic quantity computed by the recorder, and the recorder computes none.
RECORDER_UP = Gauge(
    f"{_PREFIX}_recorder_up",
    "1 while a recorder stream is connected and receiving, 0 otherwise",
    ["stream"],
)
RECORDER_EVENTS = Counter(
    f"{_PREFIX}_recorder_events_total",
    "Observations accepted into the raw sink, by stream",
    ["stream"],
)
RECORDER_LAST_EVENT_AGE = Gauge(
    f"{_PREFIX}_recorder_last_event_age_seconds",
    "Now minus the canonical time of the last observation, by stream",
    ["stream"],
)
RECORDER_RECONNECTS = Counter(
    f"{_PREFIX}_recorder_reconnects_total",
    "Websocket reconnects, by stream",
    ["stream"],
)
RECORDER_DUPLICATES = Counter(
    f"{_PREFIX}_recorder_duplicates_total",
    "Re-delivered observations the sink recognised and did not store twice, by stream",
    ["stream"],
)
RECORDER_LATE = Counter(
    f"{_PREFIX}_recorder_late_total",
    "Observations whose canonical day had already closed, by stream",
    ["stream"],
)
RECORDER_GAPFILL_ROWS = Counter(
    f"{_PREFIX}_recorder_gapfill_rows_total",
    "Closed klines fetched over REST after a disconnect, by stream",
    ["stream"],
)
RECORDER_MISSING_MINUTES = Gauge(
    f"{_PREFIX}_recorder_missing_minutes_total",
    "Minutes of the current UTC day with no closed kline in this recorder, by stream",
    ["stream"],
)
RECORDER_CLOCK_SKEW = Gauge(
    f"{_PREFIX}_recorder_clock_skew_ms",
    "Rolling median of receipt wall time minus exchange event time, in milliseconds",
)
RECORDER_DISK_FREE = Gauge(
    f"{_PREFIX}_recorder_disk_free_bytes",
    "Free bytes on the filesystem holding the recorder's storage root",
)
RECORDER_WRITE_ERRORS = Counter(
    f"{_PREFIX}_recorder_write_errors_total",
    "Failed writes, by stream",
    ["stream"],
)
RECORDER_HEARTBEAT = Gauge(
    f"{_PREFIX}_recorder_heartbeat_timestamp",
    "Unix timestamp of the last heartbeat the recorder wrote",
)

#: The twelve series section 4.8 requires, by name. Pinned here so that a rename
#: is caught by a test rather than by a blank dashboard panel.
RECORDER_METRIC_NAMES: tuple[str, ...] = (
    f"{_PREFIX}_recorder_up",
    f"{_PREFIX}_recorder_events_total",
    f"{_PREFIX}_recorder_last_event_age_seconds",
    f"{_PREFIX}_recorder_reconnects_total",
    f"{_PREFIX}_recorder_duplicates_total",
    f"{_PREFIX}_recorder_late_total",
    f"{_PREFIX}_recorder_gapfill_rows_total",
    f"{_PREFIX}_recorder_missing_minutes_total",
    f"{_PREFIX}_recorder_clock_skew_ms",
    f"{_PREFIX}_recorder_disk_free_bytes",
    f"{_PREFIX}_recorder_write_errors_total",
    f"{_PREFIX}_recorder_heartbeat_timestamp",
)


def mark_mode_decision(decision: Any) -> None:
    """Record one mode decision across the mode family.

    Takes a `chimera.modes.ModeDecision`. Typed loosely on purpose: this module
    is imported by `chimera.risk` and by the futures package, and importing
    `chimera.modes` here to name the type would make the metrics module depend
    on the layer that reports to it.
    """
    mode = decision.mode.value
    reason = decision.reason.value
    MODE_DECISIONS.labels(mode=mode, reason=reason).inc()
    MODE_CONSENSUS_STATE.labels(mode=mode, reason=reason).inc()

    # Both gauges are set across *every* mode, not only the ones that qualify.
    # A gauge that is only ever set to 1 never comes back down: the mode that was
    # eligible an hour ago would still read eligible, and two modes would read
    # selected at once. The label set is the TradingMode enum, so writing all of
    # them costs nothing and bounds nothing differently.
    eligible = {candidate.value for candidate in decision.eligible_modes}
    for candidate in type(decision.mode):
        MODE_ELIGIBLE.labels(mode=candidate.value).set(1 if candidate.value in eligible else 0)
        MODE_SELECTED.labels(mode=candidate.value).set(1 if candidate.value == mode else 0)


def mark_mode_transition(plan: Any) -> None:
    """Record one mode change, and whether it had to unwind a position."""
    labels = {"from_mode": plan.from_mode.value, "to_mode": plan.to_mode.value}
    if plan.from_mode is plan.to_mode:
        return
    MODE_TRANSITIONS.labels(**labels).inc()
    if plan.must_flatten:
        MODE_TRANSITION_FLATTENS.labels(**labels).inc()


def mark_mode_veto(mode: str, reason: str) -> None:
    """One Aegis veto, attributed to the mode that was active when it fired.

    `chimera_futures_risk_vetoes_total` already counts vetoes by reason; this is
    the same event split by mode, which is what says whether one mode is being
    refused far more often than the others. Both labels are bounded enums —
    a TradingMode value and a RiskEngine reason — and neither carries an order,
    a price or a free-text string.
    """
    MODE_RISK_VETOES.labels(mode=mode, reason=reason).inc()


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
