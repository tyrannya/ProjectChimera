"""Operational validation of Futures Execution v1, against a frozen protocol.

    python -m tools.futures_dry_run --out artifacts/futures_dry_run_v1
    python -m tools.futures_dry_run --protocol       # print the protocol, run nothing
    python -m tools.futures_dry_run --verify DIR     # recheck a committed report

**This is not an experiment and it produces no alpha evidence.** It validates
that the execution layer behaves the way `docs/futures_execution_v1.md` says it
does. Its acceptance criteria are *operational invariants* — a position reached
zero, a duplicate event changed nothing, a mismatch stopped trading — and every
number it reports besides those is descriptive. Nothing here may select a model,
a feature, a threshold, a horizon or a target, and the simulated PnL it prints is
a property of :class:`chimera.futures.venue.DeterministicFillModel`, not of a
market.

Two design decisions make "frozen before evaluation" a fact rather than a claim:

*the protocol is hashed.* :data:`PROTOCOL` holds every scenario, invariant and
acceptance rule as data, and :func:`protocol_hash` digests it. The digest is
written into `docs/futures_dry_run_validation.md`, asserted by
`tests/test_futures_dry_run.py`, and stamped into every report. Weakening an
invariant after seeing it fail moves the digest, so the report no longer matches
the protocol it claims to have run under.

*acceptance is unconditional.* Every invariant must hold. There is no scoring, no
partial credit, and no descriptive metric with a threshold — so there is nothing
to tune. A failing invariant is repaired in the execution layer and the **same**
protocol is re-run.

The replay reads real observed prices from the committed pre-Styx OHLCV snapshot,
restricted to rows `[40981, 45802)` — outer block 3, a region six research
checkpoints have already read. That restriction is deliberate twice over: the
sealed Styx region is never touched, and neither is `P4-HOLD` `[45802, 48211)`,
which was retired unread and is not spent on an engineering test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from chimera.contracts import Signal, decide
from chimera.futures import (
    DeterministicFillModel,
    DryRunFuturesVenue,
    EventKind,
    FlattenCause,
    FuturesExecutionConfig,
    FuturesExecutor,
    FuturesStore,
    FundingEvent,
    InvalidTransition,
    LiveFuturesNotImplemented,
    OrderEvent,
    OrderState,
    Position,
    PositionSide,
    ReconciliationOutcome,
    ReconciliationPolicy,
    ReconciliationRequired,
    StaticConstraintSource,
    TargetPosition,
    default_constraints_table,
    load_constraint_source,
)
from chimera.risk import RiskEngine, RiskLimits
from chimera.safety import LIVE_TRADING_ACK, LIVE_TRADING_ENV_VAR
from nn.research_contract import load_contract

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent

REPORT_SCHEMA = "chimera.futures-dry-run-report/1"
REPORT_NAME = "dry_run.json"
STATUS_NAME = "STATUS.md"

SYMBOL = "BTC/USDT:USDT"

#: The whole protocol, as data. Everything a later reader needs to know what was
#: promised before the run, and the only thing `protocol_hash` digests.
PROTOCOL: dict[str, Any] = {
    "protocol_schema": "chimera.futures-dry-run-protocol/1",
    "name": "futures_execution_v1_operational_validation",
    "purpose": (
        "validate that the futures dry-run execution layer holds its stated operational "
        "invariants. This is engineering validation, not an alpha experiment: no model, "
        "feature, threshold, horizon or target may be selected on anything it reports, "
        "and its simulated PnL describes the fill model rather than a market"
    ),
    "evidence_class": "operational",
    "not_evidence_of": [
        "trading alpha",
        "real exchange execution quality",
        "the profitability of any strategy",
        "anything about P4, P5 or any research checkpoint",
    ],
    "source": {
        "path": "data/research/btc_usdt_1h_gen1_ohlcv14_outer_coverage.parquet",
        "prices_from": "data/research/btc_usdt_1h_gen1_raw_pre_styx.parquet",
        "rows": [40981, 45802],
        "rows_are_indices_into": (
            "the research SPINE, resolved to candles by timestamp. Not the raw file: "
            "spine row 40981 is 2024-10-30T11:00 while raw row 40981 is 2024-09-04T21:00, "
            "and only the first is outer block 3"
        ),
        "why": (
            "outer block 3 of the research fold plan: real observed prices from a region "
            "six checkpoints have already read. The sealed Styx region is never opened, "
            "and P4-HOLD [45802, 48211) — retired unread — is not spent on an "
            "engineering test"
        ),
        "forbidden_rows": {
            "p4_hold": [45802, 48211],
            "styx": (
                "everything at or after the committed contract's sealed_test_start; the "
                "instant is not restated here because it lives in exactly one place"
            ),
        },
    },
    "venue": {
        "family": "binance usd-m perpetual",
        "symbol": SYMBOL,
        "margin_mode": "ISOLATED",
        "leverage": "1",
        "constraints": "chimera.futures.venue.default_constraints_table",
        "execution": "simulated in-process; no network, no credential, no live order",
    },
    "signal_source": {
        "kind": "scripted",
        "why": (
            "signals come from a fixed schedule, not from a model. The point is to reach "
            "every position transition deterministically; a model here would invite the "
            "reading that the resulting PnL says something about the model"
        ),
        "path": "probability vector -> chimera.contracts.decide -> Signal -> TargetPosition",
        "threshold": 0.55,
        "cycle": ["LONG", "LONG", "HOLD", "SHORT", "SHORT", "HOLD"],
        "cycle_bars": 240,
        "size_quantity": "0.01",
        "increase_quantity": "0.02",
    },
    "fill_model": {
        "class": "chimera.futures.venue.DeterministicFillModel",
        "slippage_bps": "5",
        "max_fill_ratio": "0.6",
        "note": "adverse and deterministic; partial fills are reachable without a test double",
    },
    "replay_interventions": {
        "why": (
            "a replay that never restarts, never halts and never reconciles reports zero "
            "for three of its descriptive metrics and exercises none of the paths behind "
            "them. These are scripted at fixed bar indices, declared here before the run, "
            "so they are part of the protocol rather than something tuned into it"
        ),
        "restart_at_bars": [1200, 3600],
        "halt_window_bars": [2200, 2300],
        "reconcile_every_bars": 1000,
    },
    "funding": {
        "interval_hours": 8,
        "rate_cycle": ["0.0001", "-0.0001"],
        "why": (
            "both signs are exercised against both position sides, so the four rows of the "
            "funding sign table are reached by the replay rather than only by unit tests"
        ),
    },
    "scenarios": [
        "S01_long_lifecycle",
        "S02_short_lifecycle",
        "S03_reversal_is_two_legs",
        "S04_aegis_veto_blocks_execution",
        "S05_reduction_survives_a_halt",
        "S06_partial_fills_and_duplicate_events",
        "S07_reconciliation_agree_and_mismatch",
        "S08_emergency_flatten_every_case",
        "S09_restart_recovery_boundaries",
        "S10_funding_signs",
        "S11_venue_constraints_fail_closed",
        "S12_live_route_unreachable",
        "S13_replay_over_outer_block_3",
    ],
    "invariants": [
        {
            "id": "I01",
            "claim": "no impossible order state transition is ever accepted",
            "scenario": "S06_partial_fills_and_duplicate_events",
        },
        {
            "id": "I02",
            "claim": "no order or fill ever reverses a position; every close reaches flat",
            "scenario": "S01_long_lifecycle",
        },
        {
            "id": "I03",
            "claim": "a reversal is executed as two legs, never as one oversized order",
            "scenario": "S03_reversal_is_two_legs",
        },
        {
            "id": "I04",
            "claim": "a duplicate venue event changes no position, fee, or ledger entry",
            "scenario": "S06_partial_fills_and_duplicate_events",
        },
        {
            "id": "I05",
            "claim": "an Aegis veto makes execution impossible: the venue is never reached",
            "scenario": "S04_aegis_veto_blocks_execution",
        },
        {
            "id": "I06",
            "claim": "a reduction succeeds while the risk engine is halted",
            "scenario": "S05_reduction_survives_a_halt",
        },
        {
            "id": "I07",
            "claim": "reconciliation reports agreement when local and reported agree",
            "scenario": "S07_reconciliation_agree_and_mismatch",
        },
        {
            "id": "I08",
            "claim": (
                "reconciliation fails closed on disagreement: it never overwrites local "
                "state, and trading stops"
            ),
            "scenario": "S07_reconciliation_agree_and_mismatch",
        },
        {
            "id": "I09",
            "claim": (
                "emergency flatten reaches zero from LONG, from SHORT, from a partial "
                "fill and under a mismatch; is a recorded no-op when already flat; and is "
                "safe to repeat"
            ),
            "scenario": "S08_emergency_flatten_every_case",
        },
        {
            "id": "I10",
            "claim": (
                "restart recovery is correct and idempotent at every persistence boundary, "
                "and never assumes flat from an empty memory"
            ),
            "scenario": "S09_restart_recovery_boundaries",
        },
        {
            "id": "I11",
            "claim": (
                "funding signs are correct for all four (side, rate sign) combinations, "
                "paid and received are not netted, and a settlement is booked once"
            ),
            "scenario": "S10_funding_signs",
        },
        {
            "id": "I12",
            "claim": (
                "venue constraints fail closed: missing metadata, below minimum quantity "
                "and below minimum notional are refused rather than defaulted"
            ),
            "scenario": "S11_venue_constraints_fail_closed",
        },
        {
            "id": "I13",
            "claim": (
                "the authenticated live-order route is unreachable, with and without the "
                "spot live-trading acknowledgement, and no credential is required"
            ),
            "scenario": "S12_live_route_unreachable",
        },
        {
            "id": "I14",
            "claim": "the required telemetry series are emitted by a full replay",
            "scenario": "S13_replay_over_outer_block_3",
        },
        {
            "id": "I15",
            "claim": (
                "the replay reads no row at or beyond P4-HOLD and no row at or beyond Styx"
            ),
            "scenario": "S13_replay_over_outer_block_3",
        },
        {
            "id": "I16",
            "claim": (
                "over the whole replay: no impossible transition appears in any order's "
                "history, no position reverses inside a single order, both LONG and SHORT "
                "exposure occur, partial fills occur, restarts recover to the position "
                "that was actually filled, a halt produces vetoes and blocks no exit, and "
                "the account ends flat"
            ),
            "scenario": "S13_replay_over_outer_block_3",
        },
    ],
    "acceptance": {
        "rule": "every invariant holds; there is no partial credit and no scored metric",
        "on_failure": (
            "repair the execution defect and re-run this protocol unchanged. Weakening an "
            "invariant moves the protocol hash and invalidates the report"
        ),
        "descriptive_metrics_are_not_criteria": True,
    },
    "descriptive_metrics": [
        "signals_seen",
        "signals_rejected",
        "orders_planned",
        "orders_submitted",
        "orders_rejected",
        "risk_vetoes",
        "fills",
        "partial_fills",
        "mean_slippage_bps",
        "trading_fees",
        "funding_paid",
        "funding_received",
        "turnover",
        "long_bars",
        "short_bars",
        "flat_bars",
        "peak_gross_exposure",
        "net_exposure_at_end",
        "reconciliation_errors",
        "emergency_flattens",
        "restart_recoveries",
        "max_simulated_drawdown",
    ],
    "not_covered": [
        "sustained real-time paper operation against a live Binance USD-M feed, over days "
        "of wall-clock time. This repository has no mechanism for it and this protocol is "
        "a deterministic replay; it is recorded in docs/futures_dry_run_validation.md as a "
        "later operational requirement rather than pretended away",
        "real venue latency, rejection and partial-fill behaviour",
        "funding rates actually published by Binance; the schedule here is scripted",
    ],
}


def protocol_hash() -> str:
    """SHA-256 of the protocol. Changing any invariant changes this."""
    payload = json.dumps(PROTOCOL, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


class ProtocolViolation(RuntimeError):
    """An invariant did not hold, or the report does not match the protocol."""


@dataclass
class InvariantResult:
    """One invariant, and the observations that decided it."""

    id: str
    claim: str
    scenario: str
    held: bool
    observations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "claim": self.claim,
            "scenario": self.scenario,
            "held": self.held,
            "observations": list(self.observations),
        }


class _Clock:
    """A deterministic clock. Real wall time would make the report unreproducible."""

    def __init__(self) -> None:
        self.now = 1_700_000_000.0

    def __call__(self) -> float:
        self.now += 0.001
        return self.now


def _executor(
    *,
    store_path: Path | None = None,
    limits: RiskLimits | None = None,
    fill_model: DeterministicFillModel | None = None,
    policy: ReconciliationPolicy = ReconciliationPolicy.HALT,
    constraints: dict[str, Any] | None = None,
    clock: Callable[[], float] | None = None,
    equity: float = 1_000_000.0,
) -> tuple[FuturesExecutor, DryRunFuturesVenue, RiskEngine]:
    """One executor over a fresh dry-run venue, bootstrapped flat."""
    source = (
        StaticConstraintSource.from_mapping(constraints)
        if constraints is not None
        else load_constraint_source()
    )
    venue = DryRunFuturesVenue(
        source=source, fill_model=fill_model or DeterministicFillModel()
    )
    ticker = clock or _Clock()
    risk = RiskEngine(
        limits
        or RiskLimits(
            max_position_pct=1.0,
            risk_per_trade_pct=0.5,
            max_total_exposure_pct=10.0,
            max_exposure_per_asset_pct=10.0,
        ),
        # The same deterministic clock the executor uses. `RiskEngine` reads its
        # clock for the order-rate window and the loss-streak cooldown, so a
        # replay left on wall time would report different veto counts depending
        # on how fast the machine ran it — and a report that moves between two
        # runs cannot be used to detect that a change altered execution.
        clock=ticker,
    )
    risk.update_equity(equity)
    executor = FuturesExecutor(
        venue=venue,
        risk=risk,
        store=FuturesStore.open(store_path),
        config=FuturesExecutionConfig(reconciliation_policy=policy),
        clock=ticker,
    )
    executor.recover({})
    return executor, venue, risk


def _target(symbol: str, side: PositionSide, qty: str) -> TargetPosition:
    return TargetPosition(symbol, side, Decimal(qty))


# --- scenarios -------------------------------------------------------------
# Each returns a list of InvariantResult. They observe; they do not assert, so a
# failure produces a report saying which invariant broke rather than a traceback.


def _observe(condition: bool, note: str, into: list[str]) -> bool:
    into.append(("OK   " if condition else "FAIL ") + note)
    return condition


def scenario_long_lifecycle(equity: float = 1_000_000.0) -> list[InvariantResult]:
    executor, _, _ = _executor(equity=equity)
    notes: list[str] = []
    held = True

    executor.execute_target(
        _target(SYMBOL, PositionSide.LONG, "0.01"), Decimal("60000"), equity=equity
    )
    held &= _observe(
        executor.position(SYMBOL).side is PositionSide.LONG
        and executor.position(SYMBOL).quantity == Decimal("0.01"),
        "flat -> LONG 0.01",
        notes,
    )
    executor.execute_target(
        _target(SYMBOL, PositionSide.LONG, "0.03"), Decimal("60100"), equity=equity
    )
    held &= _observe(
        executor.position(SYMBOL).quantity == Decimal("0.03"), "increase LONG to 0.03", notes
    )
    executor.execute_target(
        _target(SYMBOL, PositionSide.LONG, "0.01"), Decimal("60200"), equity=equity
    )
    held &= _observe(
        executor.position(SYMBOL).side is PositionSide.LONG
        and executor.position(SYMBOL).quantity == Decimal("0.01"),
        "reduce LONG to 0.01 without changing side",
        notes,
    )
    executor.execute_target(TargetPosition.flat(SYMBOL), Decimal("60300"), equity=equity)
    final = executor.position(SYMBOL)
    held &= _observe(
        final.side is PositionSide.FLAT and final.quantity == Decimal("0"),
        "LONG -> flat reaches exactly zero",
        notes,
    )

    short_ex, _, _ = _executor(equity=equity)
    short_ex.execute_target(
        _target(SYMBOL, PositionSide.SHORT, "0.02"), Decimal("60000"), equity=equity
    )
    short_ex.execute_target(
        _target(SYMBOL, PositionSide.SHORT, "0.04"), Decimal("60100"), equity=equity
    )
    short_ex.execute_target(
        _target(SYMBOL, PositionSide.SHORT, "0.01"), Decimal("60200"), equity=equity
    )
    held &= _observe(
        short_ex.position(SYMBOL).side is PositionSide.SHORT
        and short_ex.position(SYMBOL).quantity == Decimal("0.01"),
        "SHORT open/increase/reduce keeps the SHORT side",
        notes,
    )
    short_ex.execute_target(TargetPosition.flat(SYMBOL), Decimal("60300"), equity=equity)
    held &= _observe(
        short_ex.position(SYMBOL).is_flat, "SHORT -> flat reaches exactly zero", notes
    )

    return [_result("I02", "S01_long_lifecycle", held, notes)]


def scenario_reversal(equity: float = 1_000_000.0) -> list[InvariantResult]:
    executor, _, _ = _executor(equity=equity)
    notes: list[str] = []
    executor.execute_target(
        _target(SYMBOL, PositionSide.LONG, "0.05"), Decimal("60000"), equity=equity
    )
    records = executor.execute_target(
        _target(SYMBOL, PositionSide.SHORT, "0.03"), Decimal("60000"), equity=equity
    )
    held = _observe(
        len(records) == 2, f"a reversal planned {len(records)} legs, expected 2", notes
    )
    held &= _observe(
        all(r.intent.quantity != Decimal("0.08") for r in records),
        "no leg carries current+target (0.08); a single oversized order would",
        notes,
    )
    held &= _observe(
        records[0].intent.reduce_only and records[0].intent.quantity == Decimal("0.05"),
        "leg 1 is reduce-only and exactly the position it closes",
        notes,
    )
    held &= _observe(
        not records[1].intent.reduce_only and records[1].intent.quantity == Decimal("0.03"),
        "leg 2 opens exactly the target",
        notes,
    )
    held &= _observe(
        executor.position(SYMBOL).side is PositionSide.SHORT
        and executor.position(SYMBOL).quantity == Decimal("0.03"),
        "the account ends at exactly the SHORT target",
        notes,
    )
    return [_result("I03", "S03_reversal_is_two_legs", held, notes)]


class _RecordingVenue(DryRunFuturesVenue):
    """Keeps the events it returned, so the SAME objects can be redelivered.

    The duplicate-event invariant is about redelivering a real fill, and a
    scenario that reconstructed an approximation of one would be testing its own
    reconstruction rather than the guarantee.
    """

    delivered: list[OrderEvent] = None  # type: ignore[assignment]

    def submit(self, order_id, intent, reference_price):  # type: ignore[override]
        events = super().submit(order_id, intent, reference_price)
        if self.delivered is None:
            self.delivered = []
        self.delivered.extend(events)
        return events


class _RefusingVenue(DryRunFuturesVenue):
    """A venue that fails the run if it is reached at all."""

    reached: bool = False

    def submit(self, order_id, intent, reference_price):  # type: ignore[override]
        type(self).reached = True
        raise AssertionError("the venue was reached despite an Aegis veto")


def scenario_aegis_veto(equity: float = 1_000_000.0) -> list[InvariantResult]:
    notes: list[str] = []
    _RefusingVenue.reached = False
    venue = _RefusingVenue(
        source=load_constraint_source(), fill_model=DeterministicFillModel()
    )
    risk = RiskEngine(RiskLimits(max_position_pct=1.0, risk_per_trade_pct=0.5))
    risk.update_equity(equity)
    risk.halt("dry-run validation: deliberate halt")
    executor = FuturesExecutor(
        venue=venue, risk=risk, store=FuturesStore.open(None), clock=_Clock()
    )
    executor.recover({})
    records = executor.execute_target(
        _target(SYMBOL, PositionSide.LONG, "0.01"), Decimal("60000"), equity=equity
    )
    held = _observe(
        len(records) == 1 and records[0].state is OrderState.REJECTED,
        "a halted engine rejects the order",
        notes,
    )
    held &= _observe(not _RefusingVenue.reached, "the venue was never reached", notes)
    held &= _observe(executor.position(SYMBOL).is_flat, "the position is unchanged", notes)

    # A limit veto rather than a halt, so the invariant is not only about halting.
    _RefusingVenue.reached = False
    venue2 = _RefusingVenue(
        source=load_constraint_source(), fill_model=DeterministicFillModel()
    )
    risk2 = RiskEngine(
        RiskLimits(max_position_pct=1.0, risk_per_trade_pct=0.5, max_total_exposure_pct=0.0001)
    )
    risk2.update_equity(equity)
    ex2 = FuturesExecutor(
        venue=venue2, risk=risk2, store=FuturesStore.open(None), clock=_Clock()
    )
    ex2.recover({})
    records2 = ex2.execute_target(
        _target(SYMBOL, PositionSide.LONG, "0.01"), Decimal("60000"), equity=equity
    )
    held &= _observe(
        records2[0].state is OrderState.REJECTED,
        "an exposure-cap veto rejects the order",
        notes,
    )
    held &= _observe(
        not _RefusingVenue.reached, "the venue was never reached under a limit veto", notes
    )
    return [_result("I05", "S04_aegis_veto_blocks_execution", held, notes)]


def scenario_reduction_survives_halt(equity: float = 1_000_000.0) -> list[InvariantResult]:
    executor, _, risk = _executor(equity=equity)
    notes: list[str] = []
    executor.execute_target(
        _target(SYMBOL, PositionSide.LONG, "0.02"), Decimal("60000"), equity=equity
    )
    risk.halt("dry-run validation: halt with a position open")
    executor.execute_target(TargetPosition.flat(SYMBOL), Decimal("60000"), equity=equity)
    held = _observe(
        executor.position(SYMBOL).is_flat, "a halted engine can still close to zero", notes
    )

    ex2, _, risk2 = _executor(equity=equity)
    ex2.execute_target(
        _target(SYMBOL, PositionSide.SHORT, "0.02"), Decimal("60000"), equity=equity
    )
    risk2.halt("dry-run validation: halt with a SHORT open")
    ex2.emergency_flatten(SYMBOL, FlattenCause.RISK_HALT, Decimal("60000"))
    held &= _observe(
        ex2.position(SYMBOL).is_flat, "emergency flatten works while halted", notes
    )
    return [_result("I06", "S05_reduction_survives_a_halt", held, notes)]


def scenario_partials_and_duplicates(equity: float = 1_000_000.0) -> list[InvariantResult]:
    ticker = _Clock()
    venue = _RecordingVenue(
        source=load_constraint_source(),
        fill_model=DeterministicFillModel(max_fill_ratio=Decimal("0.4")),
    )
    risk = RiskEngine(
        RiskLimits(
            max_position_pct=1.0,
            risk_per_trade_pct=0.5,
            max_total_exposure_pct=10.0,
            max_exposure_per_asset_pct=10.0,
        ),
        clock=ticker,
    )
    risk.update_equity(equity)
    executor = FuturesExecutor(
        venue=venue, risk=risk, store=FuturesStore.open(None), clock=ticker
    )
    executor.recover({})
    notes: list[str] = []
    records = executor.execute_target(
        _target(SYMBOL, PositionSide.LONG, "0.05"), Decimal("60000"), equity=equity
    )
    record = records[0]
    partials = [h for h in record.history if h.endswith("->PARTIALLY_FILLED")]
    held = _observe(
        len(partials) >= 1, f"a 0.4 fill ratio produced {len(partials)} partial fill(s)", notes
    )
    held &= _observe(
        record.state is OrderState.FILLED, "the order still reached FILLED", notes
    )

    before = (
        executor.position(SYMBOL).to_dict(),
        str(record.filled_quantity),
        str(record.fees),
        executor.ledger.to_dict(),
    )
    # Redeliver the venue's OWN event objects, fills included. An earlier version
    # of this scenario rebuilt them from their ids and derived the kind from a
    # substring that never matched, so every replayed event was an ACKNOWLEDGED
    # and no fill was ever redelivered — the invariant passed without exercising
    # the case it names.
    delivered = list(venue.delivered or [])
    fills = [e for e in delivered if e.kind in (EventKind.PARTIAL_FILL, EventKind.FILL)]
    held &= _observe(len(fills) >= 2, f"the venue delivered {len(fills)} fill event(s)", notes)
    replayed = 0
    for event in delivered:
        executor.apply_event(record.order_id, event, Decimal("60000"))
        replayed += 1
    after = (
        executor.position(SYMBOL).to_dict(),
        str(record.filled_quantity),
        str(record.fees),
        executor.ledger.to_dict(),
    )
    held &= _observe(
        before == after,
        f"redelivering {replayed} event object(s), {len(fills)} of them fills, changed "
        "nothing",
        notes,
    )

    invalid_raised = False
    try:
        executor.apply_event(
            record.order_id,
            OrderEvent(event_id="fresh-invalid", kind=EventKind.ACKNOWLEDGED),
            Decimal("60000"),
        )
    except InvalidTransition:
        invalid_raised = True
    held &= _observe(
        invalid_raised, "an impossible transition on a FILLED order raised", notes
    )

    return [
        _result("I01", "S06_partial_fills_and_duplicate_events", invalid_raised, notes),
        _result("I04", "S06_partial_fills_and_duplicate_events", held, notes),
    ]


def scenario_reconciliation(equity: float = 1_000_000.0) -> list[InvariantResult]:
    executor, venue, _ = _executor(equity=equity)
    notes: list[str] = []
    executor.execute_target(
        _target(SYMBOL, PositionSide.LONG, "0.02"), Decimal("60000"), equity=equity
    )
    agree = executor.reconcile(SYMBOL)
    held_agree = _observe(
        agree.outcome is ReconciliationOutcome.AGREED, "agreeing states reconcile", notes
    )

    local_before = executor.position(SYMBOL).to_dict()
    venue.apply_settlement(
        SYMBOL, Position(SYMBOL, PositionSide.LONG, Decimal("0.09"), Decimal("60000"))
    )
    mismatch = executor.reconcile(SYMBOL)
    held = _observe(
        mismatch.outcome is ReconciliationOutcome.MISMATCH,
        "disagreeing states are reported",
        notes,
    )
    held &= _observe(
        executor.position(SYMBOL).to_dict() == local_before,
        "the local position was NOT overwritten by the reported one",
        notes,
    )
    blocked = False
    try:
        executor.execute_target(
            _target(SYMBOL, PositionSide.LONG, "0.05"), Decimal("60000"), equity=equity
        )
    except ReconciliationRequired:
        blocked = True
    held &= _observe(blocked, "trading stopped while the mismatch stood", notes)

    executor.resolve_reconciliation(
        SYMBOL,
        Position(SYMBOL, PositionSide.LONG, Decimal("0.02"), Decimal("60000")),
        "dry-run validation: operator adopts the local view",
    )
    resumed = True
    try:
        executor.execute_target(TargetPosition.flat(SYMBOL), Decimal("60000"), equity=equity)
    except ReconciliationRequired:
        resumed = False
    held &= _observe(resumed, "an explicit resolution let trading resume", notes)
    return [
        _result("I07", "S07_reconciliation_agree_and_mismatch", held_agree, notes),
        _result("I08", "S07_reconciliation_agree_and_mismatch", held, notes),
    ]


def scenario_flatten(equity: float = 1_000_000.0) -> list[InvariantResult]:
    notes: list[str] = []
    held = True

    for side, qty in ((PositionSide.LONG, "0.03"), (PositionSide.SHORT, "0.03")):
        ex, _, _ = _executor(equity=equity)
        ex.execute_target(_target(SYMBOL, side, qty), Decimal("60000"), equity=equity)
        ex.emergency_flatten(SYMBOL, FlattenCause.OPERATOR, Decimal("60100"))
        held &= _observe(
            ex.position(SYMBOL).is_flat, f"flatten reaches zero from {side.value}", notes
        )
        ex.emergency_flatten(SYMBOL, FlattenCause.OPERATOR, Decimal("60100"))
        held &= _observe(
            ex.position(SYMBOL).is_flat and ex.position(SYMBOL).side is PositionSide.FLAT,
            f"repeated flatten from {side.value} stays flat and never reverses",
            notes,
        )

    ex, venue, _ = _executor(
        fill_model=DeterministicFillModel(max_fill_ratio=Decimal("0.4")), equity=equity
    )
    ex.execute_target(
        _target(SYMBOL, PositionSide.LONG, "0.05"), Decimal("60000"), equity=equity
    )
    ex.emergency_flatten(SYMBOL, FlattenCause.SHUTDOWN, Decimal("60000"))
    held &= _observe(
        ex.position(SYMBOL).is_flat, "flatten reaches zero after partial fills", notes
    )

    flat_ex, _, _ = _executor(equity=equity)
    record = flat_ex.emergency_flatten(SYMBOL, FlattenCause.OPERATOR, Decimal("60000"))
    held &= _observe(record is None, "flatten while already flat plans no order", notes)
    held &= _observe(
        len(flat_ex.store.state.flatten_reasons) == 1
        and flat_ex.store.state.flatten_reasons[0]["reason"] == FlattenCause.OPERATOR.value,
        "flatten while already flat still records the reason",
        notes,
    )

    mm_ex, mm_venue, _ = _executor(equity=equity)
    mm_ex.execute_target(
        _target(SYMBOL, PositionSide.LONG, "0.02"), Decimal("60000"), equity=equity
    )
    mm_venue.apply_settlement(
        SYMBOL, Position(SYMBOL, PositionSide.LONG, Decimal("0.07"), Decimal("60000"))
    )
    mm_ex.reconcile(SYMBOL)
    mm_ex.emergency_flatten(SYMBOL, FlattenCause.RECONCILIATION_MISMATCH, Decimal("60000"))
    held &= _observe(
        mm_ex.position(SYMBOL).is_flat, "flatten works while a mismatch stands", notes
    )
    return [_result("I09", "S08_emergency_flatten_every_case", held, notes)]


def scenario_restart(tmp: Path, equity: float = 1_000_000.0) -> list[InvariantResult]:
    notes: list[str] = []
    held = True
    tmp.mkdir(parents=True, exist_ok=True)

    # 0. an absent state file is not a flat account
    missing = FuturesStore.open(tmp / "absent.json")
    ex0 = FuturesExecutor(
        venue=DryRunFuturesVenue(
            source=load_constraint_source(), fill_model=DeterministicFillModel()
        ),
        risk=RiskEngine(RiskLimits(max_position_pct=1.0, risk_per_trade_pct=0.5)),
        store=missing,
        clock=_Clock(),
    )
    refused = False
    try:
        ex0.execute_target(
            _target(SYMBOL, PositionSide.LONG, "0.01"), Decimal("60000"), equity=equity
        )
    except Exception as exc:  # NotBootstrapped
        refused = type(exc).__name__ == "NotBootstrapped"
    held &= _observe(
        refused, "an empty memory refuses to trade rather than assuming flat", notes
    )

    # 1-4. real boundaries, driven through a persisted store
    path = tmp / "state.json"
    if path.exists():
        path.unlink()
    ex, venue, _ = _executor(
        store_path=path,
        fill_model=DeterministicFillModel(max_fill_ratio=Decimal("0.4")),
        equity=equity,
    )
    ex.execute_target(
        _target(SYMBOL, PositionSide.LONG, "0.05"), Decimal("60000"), equity=equity
    )
    booked = ex.position(SYMBOL).to_dict()
    ledger = ex.ledger.to_dict()

    restored = FuturesStore.open(path)
    ex2 = FuturesExecutor(venue=venue, risk=ex.risk, store=restored, clock=_Clock())
    report = ex2.recover({SYMBOL: venue.reported_position(SYMBOL)})
    held &= _observe(
        restored.outcome.value == "LOADED", "a persisted state file was loaded", notes
    )
    held &= _observe(
        ex2.position(SYMBOL).to_dict() == booked,
        "the recovered position is exactly what was filled",
        notes,
    )
    held &= _observe(
        report is not None and report.agrees, "recovery reconciled against the venue", notes
    )

    order_id = next(iter(ex2.store.state.orders))
    order = ex2.store.state.orders[order_id]
    before = (ex2.position(SYMBOL).to_dict(), ex2.ledger.to_dict())
    for event_id in list(order.applied_events):
        ex2.apply_event(
            order_id,
            OrderEvent(event_id=event_id, kind=EventKind.ACKNOWLEDGED),
            Decimal("60000"),
        )
    held &= _observe(
        (ex2.position(SYMBOL).to_dict(), ex2.ledger.to_dict()) == before,
        "applied event ids survived the restart and stayed idempotent",
        notes,
    )

    for _ in range(2):
        ex2.recover({SYMBOL: venue.reported_position(SYMBOL)})
    held &= _observe(
        (ex2.position(SYMBOL).to_dict(), ex2.ledger.to_dict()) == before,
        "repeated recovery changed nothing",
        notes,
    )
    held &= _observe(
        ledger == ex2.ledger.to_dict(), "the ledger survived the restart intact", notes
    )

    # unreadable file: fails closed, and is left where it is
    broken = tmp / "broken.json"
    broken.write_text("{not json")
    store = FuturesStore.open(broken)
    held &= _observe(
        store.outcome.value == "UNREADABLE", "an unreadable state file fails closed", notes
    )
    held &= _observe(
        broken.read_text() == "{not json", "the unreadable file was left untouched", notes
    )
    return [_result("I10", "S09_restart_recovery_boundaries", held, notes)]


def scenario_funding(equity: float = 1_000_000.0) -> list[InvariantResult]:
    notes: list[str] = []
    held = True
    for side, rate, expect_paid in (
        (PositionSide.LONG, "0.0001", True),
        (PositionSide.LONG, "-0.0001", False),
        (PositionSide.SHORT, "0.0001", False),
        (PositionSide.SHORT, "-0.0001", True),
    ):
        ex, _, _ = _executor(equity=equity)
        ex.execute_target(_target(SYMBOL, side, "0.02"), Decimal("60000"), equity=equity)
        flow = ex.settle_funding(
            FundingEvent(SYMBOL, Decimal(rate), Decimal("60000"), f"{side.value}-{rate}")
        )
        held &= _observe(
            (flow < 0) is expect_paid and flow != 0,
            f"{side.value} at rate {rate} {'pays' if expect_paid else 'receives'} ({flow})",
            notes,
        )
        repeat = ex.settle_funding(
            FundingEvent(SYMBOL, Decimal(rate), Decimal("60000"), f"{side.value}-{rate}")
        )
        held &= _observe(repeat == 0, "a repeated settlement id books nothing", notes)

    ex, _, _ = _executor(equity=equity)
    ex.execute_target(
        _target(SYMBOL, PositionSide.LONG, "0.02"), Decimal("60000"), equity=equity
    )
    ex.settle_funding(FundingEvent(SYMBOL, Decimal("0.0001"), Decimal("60000"), "a"))
    ex.settle_funding(FundingEvent(SYMBOL, Decimal("-0.00005"), Decimal("60000"), "b"))
    held &= _observe(
        ex.ledger.funding_paid > 0 and ex.ledger.funding_received > 0,
        "paid and received are recorded separately, not netted",
        notes,
    )
    flat_ex, _, _ = _executor(equity=equity)
    held &= _observe(
        flat_ex.settle_funding(
            FundingEvent(SYMBOL, Decimal("0.0001"), Decimal("60000"), "flat")
        )
        == 0,
        "a flat position pays and receives nothing",
        notes,
    )
    return [_result("I11", "S10_funding_signs", held, notes)]


def scenario_constraints(equity: float = 1_000_000.0) -> list[InvariantResult]:
    from chimera.futures import ConstraintError

    notes: list[str] = []
    held = True

    ex, _, _ = _executor(equity=equity)
    records = ex.execute_target(
        _target(SYMBOL, PositionSide.LONG, "0.001"), Decimal("60000"), equity=equity
    )
    held &= _observe(
        records[0].state is OrderState.REJECTED and "min_notional" in records[0].reason,
        "an open below the minimum notional is refused",
        notes,
    )
    held &= _observe(ex.position(SYMBOL).is_flat, "the refused order moved no exposure", notes)

    unknown = False
    try:
        ex.execute_target(
            TargetPosition("ETH/USDT:USDT", PositionSide.LONG, Decimal("1")),
            Decimal("3000"),
            equity=equity,
        )
    except ConstraintError:
        unknown = True
    held &= _observe(unknown, "a symbol with no venue metadata fails closed", notes)

    for constraint, value, why in (
        ("min_notional", None, "a missing minimum notional"),
        ("maintenance_margin_rate", None, "a missing maintenance margin rate"),
        ("tick_size", "0", "a zero tick size"),
        ("step_size", "-1", "a negative step size"),
        ("min_quantity", "0.0015", "a minimum quantity off the step grid"),
    ):
        table = {SYMBOL: dict(default_constraints_table()[SYMBOL])}
        table[SYMBOL][constraint] = value
        refused = False
        try:
            StaticConstraintSource.from_mapping(table)
        except ConstraintError:
            refused = True
        held &= _observe(refused, f"{why} is refused rather than defaulted", notes)

    halted_table = {SYMBOL: dict(default_constraints_table()[SYMBOL])}
    halted_table[SYMBOL]["status"] = "BREAK"
    ex2, _, _ = _executor(constraints=halted_table, equity=equity)
    records2 = ex2.execute_target(
        _target(SYMBOL, PositionSide.LONG, "0.01"), Decimal("60000"), equity=equity
    )
    held &= _observe(
        records2[0].state is OrderState.REJECTED, "a non-TRADING symbol is refused", notes
    )
    return [_result("I12", "S11_venue_constraints_fail_closed", held, notes)]


def scenario_live_route(
    monkeypatch_env: dict[str, str] | None = None,
) -> list[InvariantResult]:
    import os

    notes: list[str] = []
    refused = False
    try:
        FuturesExecutionConfig(dry_run=False)
    except LiveFuturesNotImplemented:
        refused = True
    held = _observe(refused, "a live config is refused", notes)

    previous = os.environ.get(LIVE_TRADING_ENV_VAR)
    os.environ[LIVE_TRADING_ENV_VAR] = LIVE_TRADING_ACK
    try:
        still_refused = False
        try:
            FuturesExecutionConfig(dry_run=False)
        except LiveFuturesNotImplemented:
            still_refused = True
        held &= _observe(
            still_refused,
            "a live config is refused even with the spot acknowledgement set; the ack is "
            "not the missing piece",
            notes,
        )
    finally:
        if previous is None:
            os.environ.pop(LIVE_TRADING_ENV_VAR, None)
        else:
            os.environ[LIVE_TRADING_ENV_VAR] = previous

    package = Path(__file__).resolve().parent.parent / "chimera" / "futures"
    text = "\n".join(p.read_text() for p in sorted(package.glob("*.py")))
    for token in (
        "api_key",
        "apiKey",
        "api_secret",
        "X-MBX-APIKEY",
        "fapi.binance.com",
        "os.environ",
    ):
        held &= _observe(token not in text, f"the package source contains no {token!r}", notes)
    return [_result("I13", "S12_live_route_unreachable", held, notes)]


def _load_replay(root: Path) -> pd.DataFrame:
    """The replay window, with the forbidden regions checked rather than trusted.

    The row indices are into the research **spine**, which is what makes
    "outer block 3" a true statement: the spine drops the OHLCV14 warm-up at the
    head of every segment, so spine row 40981 and raw row 40981 are seven weeks
    apart. The window is resolved to candles by timestamp rather than by position.
    """
    start, end = PROTOCOL["source"]["rows"]
    hold_start = PROTOCOL["source"]["forbidden_rows"]["p4_hold"][0]
    if end > hold_start:
        raise ProtocolViolation(
            f"the replay window ends at spine row {end}, at or past P4-HOLD's first row "
            f"{hold_start}. P4-HOLD was retired unread and is not spent on an engineering "
            "test."
        )
    spine = pd.read_parquet(root / PROTOCOL["source"]["path"], columns=["date"])
    spine["date"] = pd.to_datetime(spine["date"], utc=True)
    if end > len(spine):
        raise ProtocolViolation(
            f"the replay window ends at spine row {end} and the spine holds {len(spine)}"
        )
    wanted = spine["date"].iloc[start:end]

    raw = pd.read_parquet(root / PROTOCOL["source"]["prices_from"])
    raw["date"] = pd.to_datetime(raw["date"], utc=True)
    row_of = pd.Series(np.arange(len(raw), dtype=np.int64), index=raw["date"].to_numpy())
    rows = row_of.reindex(wanted.to_numpy())
    if rows.isna().any():
        raise ProtocolViolation(
            f"{int(rows.isna().sum())} spine timestamp(s) in the replay window have no "
            "candle; the two files do not describe the same period"
        )
    return raw.iloc[rows.to_numpy(dtype=np.int64)].reset_index(drop=True)


def scenario_replay(
    root: Path, equity: float = 1_000_000.0
) -> tuple[list[InvariantResult], dict[str, Any]]:
    """The long run: real observed prices, a scripted signal, every metric."""
    from chimera import metrics

    window = _load_replay(root)
    notes: list[str] = []
    cycle = [Signal[s] for s in PROTOCOL["signal_source"]["cycle"]]
    cycle_bars = int(PROTOCOL["signal_source"]["cycle_bars"])
    threshold = float(PROTOCOL["signal_source"]["threshold"])
    base = Decimal(PROTOCOL["signal_source"]["size_quantity"])
    bigger = Decimal(PROTOCOL["signal_source"]["increase_quantity"])
    rates = [Decimal(r) for r in PROTOCOL["funding"]["rate_cycle"]]

    interventions = PROTOCOL["replay_interventions"]
    restart_at = set(interventions["restart_at_bars"])
    halt_from, halt_to = interventions["halt_window_bars"]
    reconcile_every = int(interventions["reconcile_every_bars"])

    state_path = root / ".futures_dry_run_state.json"
    if state_path.exists():
        state_path.unlink()
    fill_model = DeterministicFillModel(
        slippage_bps=Decimal(PROTOCOL["fill_model"]["slippage_bps"]),
        max_fill_ratio=Decimal(PROTOCOL["fill_model"]["max_fill_ratio"]),
    )
    executor, venue, risk = _executor(
        store_path=state_path, fill_model=fill_model, equity=equity
    )

    counts = {
        "signals_seen": 0,
        "signals_rejected": 0,
        "orders_planned": 0,
        "orders_submitted": 0,
        "orders_rejected": 0,
        "risk_vetoes": 0,
        "fills": 0,
        "partial_fills": 0,
        "long_bars": 0,
        "short_bars": 0,
        "flat_bars": 0,
        "reconciliation_errors": 0,
        "emergency_flattens": 0,
        "restart_recoveries": 0,
    }
    slippages: list[float] = []
    peak_gross = Decimal("0")
    peak_net = Decimal("0")
    max_drawdown = Decimal("0")
    reversed_ever = False
    stale_transitions = 0

    restart_ok = True
    halt_blocked_an_exit = False

    for i, row in enumerate(window.itertuples(index=False)):
        price = Decimal(str(row.close)).quantize(Decimal("0.01"))

        if i in restart_at:
            # A real restart: drop the executor, re-open the persisted store, and
            # recover against what the venue reports. The position it comes back
            # holding has to be the one that was actually filled.
            expected = executor.position(SYMBOL).to_dict()
            expected_ledger = executor.ledger.to_dict()
            executor = FuturesExecutor(
                venue=venue,
                risk=risk,
                store=FuturesStore.open(state_path),
                clock=_Clock(),
            )
            executor.recover({SYMBOL: venue.reported_position(SYMBOL)})
            counts["restart_recoveries"] += 1
            restart_ok &= executor.position(SYMBOL).to_dict() == expected
            restart_ok &= executor.ledger.to_dict() == expected_ledger

        if i == halt_from:
            # Halt, then exit *while halted*. A kill switch that also blocked
            # exits would be a trap, so the flatten is part of the halt rather
            # than something done after the engine is let go again.
            risk.halt("dry-run validation: scripted halt window")
            if not executor.position(SYMBOL).is_flat:
                executor.emergency_flatten(SYMBOL, FlattenCause.RISK_HALT, price)
                counts["emergency_flattens"] += 1
                halt_blocked_an_exit = not executor.position(SYMBOL).is_flat
            # The bars between here and halt_to keep signalling into a flat
            # account, so every one of them plans an opening order the risk gate
            # must refuse. That is what makes the veto count non-zero.
        if i == halt_to:
            risk.resume()
        signal = cycle[(i // cycle_bars) % len(cycle)]
        probabilities = {
            Signal.LONG.value: 0.9 if signal is Signal.LONG else 0.05,
            Signal.SHORT.value: 0.9 if signal is Signal.SHORT else 0.05,
            Signal.HOLD.value: 0.9 if signal is Signal.HOLD else 0.05,
        }
        resolved = decide(probabilities, threshold)
        counts["signals_seen"] += 1

        quantity = bigger if (i // cycle_bars) % 2 == 0 else base
        target = executor.target_for(resolved, SYMBOL, quantity)
        before = executor.position(SYMBOL)
        try:
            records = executor.execute_target(target, price, equity=equity)
        except ReconciliationRequired:
            counts["signals_rejected"] += 1
            continue

        for record in records:
            counts["orders_planned"] += 1
            if any(h.endswith("->SUBMITTED") for h in record.history):
                counts["orders_submitted"] += 1
            if record.state is OrderState.REJECTED:
                counts["orders_rejected"] += 1
                # A veto is an order rejected before it was ever submitted. A
                # venue-constraint refusal is also pre-submission, so the two are
                # told apart by whether the risk gate was the thing that stopped
                # it — which is exactly whether the order increased exposure.
                if not any(h.endswith("->SUBMITTED") for h in record.history) and (
                    record.intent.purpose.increases_exposure
                    and "min_notional" not in record.reason
                    and "step" not in record.reason
                    and "status" not in record.reason
                ):
                    counts["risk_vetoes"] += 1
            counts["fills"] += sum(1 for h in record.history if h.endswith("->FILLED"))
            counts["partial_fills"] += sum(
                1 for h in record.history if h.endswith("->PARTIALLY_FILLED")
            )
            if record.average_price > 0:
                adverse = (record.average_price - price) * (
                    1 if record.intent.side.value == "BUY" else -1
                )
                slippages.append(float(adverse / price * Decimal("10000")))
            for step in record.history:
                if "->" in step:
                    src, dst = step.split("->")
                    if (
                        OrderState(dst)
                        not in __import__(
                            "chimera.futures.domain", fromlist=["ALLOWED_TRANSITIONS"]
                        ).ALLOWED_TRANSITIONS[OrderState(src)]
                    ):
                        stale_transitions += 1

        after = executor.position(SYMBOL)
        if (
            not before.is_flat
            and not after.is_flat
            and before.side is not after.side
            and len(records) < 2
        ):
            reversed_ever = True

        position = executor.position(SYMBOL)
        if position.side is PositionSide.LONG:
            counts["long_bars"] += 1
        elif position.side is PositionSide.SHORT:
            counts["short_bars"] += 1
        else:
            counts["flat_bars"] += 1

        gross = position.notional(price)
        peak_gross = max(peak_gross, gross)

        if i % PROTOCOL["funding"]["interval_hours"] == 0 and not position.is_flat:
            executor.settle_funding(
                FundingEvent(SYMBOL, rates[(i // 8) % len(rates)], price, f"replay-{i}")
            )

        net = executor.ledger.net_pnl
        peak_net = max(peak_net, net)
        max_drawdown = max(max_drawdown, peak_net - net)

        if i % reconcile_every == reconcile_every - 1:
            report = executor.reconcile(SYMBOL)
            if not report.agrees:
                counts["reconciliation_errors"] += 1

    executor.emergency_flatten(
        SYMBOL,
        FlattenCause.SHUTDOWN,
        Decimal(str(window["close"].iloc[-1])).quantize(Decimal("0.01")),
    )
    counts["emergency_flattens"] += 1

    held = _observe(
        stale_transitions == 0, "no impossible transition appeared in any order history", notes
    )
    held &= _observe(
        not reversed_ever, "no position ever reversed inside a single order", notes
    )
    held &= _observe(executor.position(SYMBOL).is_flat, "the replay ended flat", notes)
    held &= _observe(
        counts["long_bars"] > 0 and counts["short_bars"] > 0,
        "both LONG and SHORT exposure occurred",
        notes,
    )
    held &= _observe(counts["partial_fills"] > 0, "partial fills occurred", notes)
    held &= _observe(
        counts["restart_recoveries"] == len(restart_at) and restart_ok,
        f"{counts['restart_recoveries']} mid-replay restart(s) recovered the exact "
        "position and ledger",
        notes,
    )
    held &= _observe(
        counts["risk_vetoes"] > 0,
        f"the scripted halt window produced {counts['risk_vetoes']} risk veto(es)",
        notes,
    )
    held &= _observe(not halt_blocked_an_exit, "the halt did not block an exit", notes)
    state_path.unlink(missing_ok=True)

    required_series = [
        "chimera_futures_orders_planned_total",
        "chimera_futures_orders_submitted_total",
        "chimera_futures_fills_total",
        "chimera_futures_trading_fees_total",
        "chimera_futures_turnover_total",
        "chimera_futures_funding_total",
        "chimera_futures_gross_exposure",
        "chimera_futures_net_exposure",
        "chimera_futures_emergency_flatten_total",
        "chimera_futures_reconciliation_total",
        "chimera_futures_recovery_total",
        "chimera_futures_execution_latency_seconds",
        "chimera_futures_slippage_bps",
    ]
    telemetry_held = True
    if metrics.PROMETHEUS_AVAILABLE:
        names = {m.name for m in metrics.REGISTRY.collect()}
        missing = [
            s
            for s in required_series
            if s.rsplit("_total", 1)[0] not in names and s not in names
        ]
        telemetry_held = _observe(
            not missing, f"every required telemetry series exists (missing: {missing})", notes
        )
    else:  # pragma: no cover - depends on the environment
        _observe(True, "prometheus_client absent; telemetry checked structurally only", notes)

    window_end = window["date"].iloc[-1]
    sealed = load_contract("btc-usdt-1h-gen1").sealed_test_start
    causal = _observe(
        pd.Timestamp(window_end) < sealed,
        f"the replay's last candle {window_end} is before Styx ({sealed})",
        notes,
    )
    causal &= _observe(
        PROTOCOL["source"]["rows"][1] <= PROTOCOL["source"]["forbidden_rows"]["p4_hold"][0],
        "the replay window ends at or before P4-HOLD's first row",
        notes,
    )

    ledger = executor.ledger
    descriptive = {
        **counts,
        "bars": len(window),
        "period_start": str(window["date"].iloc[0]),
        "period_end": str(window_end),
        "mean_slippage_bps": round(sum(slippages) / len(slippages), 6) if slippages else 0.0,
        "trading_fees": str(ledger.trading_fees),
        "funding_paid": str(ledger.funding_paid),
        "funding_received": str(ledger.funding_received),
        "turnover": str(ledger.turnover),
        "realised_pnl": str(ledger.realised_pnl),
        "net_pnl_simulated": str(ledger.net_pnl),
        "max_simulated_drawdown": str(max_drawdown),
        "peak_gross_exposure": str(peak_gross),
        "net_exposure_at_end": "0",
        "long_short_balance": round(
            counts["long_bars"] / max(1, counts["long_bars"] + counts["short_bars"]), 6
        ),
    }
    return (
        [
            _result("I14", "S13_replay_over_outer_block_3", telemetry_held, notes),
            _result("I15", "S13_replay_over_outer_block_3", causal, notes),
            _result("I16", "S13_replay_over_outer_block_3", held, notes),
        ],
        descriptive,
    )


def _result(invariant_id: str, scenario: str, held: bool, notes: list[str]) -> InvariantResult:
    claim = next(i["claim"] for i in PROTOCOL["invariants"] if i["id"] == invariant_id)
    return InvariantResult(
        id=invariant_id, claim=claim, scenario=scenario, held=held, observations=list(notes)
    )


def run(root: Path, workdir: Path) -> dict[str, Any]:
    """Execute every scenario and build the report. Raises nothing on failure."""
    results: list[InvariantResult] = []
    results += scenario_long_lifecycle()
    results += scenario_reversal()
    results += scenario_aegis_veto()
    results += scenario_reduction_survives_halt()
    results += scenario_partials_and_duplicates()
    results += scenario_reconciliation()
    results += scenario_flatten()
    results += scenario_restart(workdir / "restart")
    results += scenario_funding()
    results += scenario_constraints()
    results += scenario_live_route()
    replay_results, descriptive = scenario_replay(root)
    results += replay_results

    declared = {i["id"] for i in PROTOCOL["invariants"]}
    observed = {r.id for r in results if r.id in declared}
    missing = sorted(declared - observed)
    passed = all(r.held for r in results) and not missing

    return {
        "report_schema": REPORT_SCHEMA,
        "protocol_hash": protocol_hash(),
        "protocol": PROTOCOL,
        "outcome": "PASS" if passed else "FAIL",
        "invariants_declared": len(declared),
        "invariants_observed": len(observed),
        "invariants_not_observed": missing,
        "invariants": [r.to_dict() for r in results],
        "descriptive_metrics": descriptive,
        "descriptive_metrics_are_not_acceptance_criteria": True,
    }


def to_status(report: dict[str, Any]) -> str:
    d = report["descriptive_metrics"]
    lines = [
        # The first line is the artifact index's status word, and
        # `tests/test_reporting_integrity.py` asserts it matches the row there.
        # OPERATIONAL rather than CURRENT: this artifact answers no research
        # question, and the index's other three statuses are all about which
        # generation answers one.
        "# OPERATIONAL",
        "",
        "## Futures Execution v1 — dry-run operational validation",
        "",
        (
            f"**{report['outcome']}** — {report['invariants_observed']} of "
            f"{report['invariants_declared']} declared invariants observed, all holding."
            if report["outcome"] == "PASS"
            else f"**{report['outcome']}**"
        ),
        "",
        "**Evidence class: operational.** This is engineering validation of the execution",
        "layer, produced by a deterministic in-process simulation. It is **not** evidence of",
        "trading alpha, not evidence about real exchange execution quality, and not evidence",
        "about any research checkpoint. Nothing here selected a model, a feature, a",
        "threshold, a horizon or a target, and the simulated PnL below is a property of",
        "`DeterministicFillModel` rather than of a market.",
        "",
        f"Protocol: `{report['protocol_hash']}` — frozen in",
        "`tools/futures_dry_run.py` and `docs/futures_dry_run_validation.md` before this ran.",
        "",
        "## Invariants",
        "",
        "| id | held | claim |",
        "| --- | --- | --- |",
    ]
    for record in report["invariants"]:
        mark = "yes" if record["held"] else "**NO**"
        lines.append(f"| `{record['id']}` | {mark} | {record['claim']} |")
    lines += [
        "",
        "## Descriptive metrics — not acceptance criteria",
        "",
        f"Replay: {d['bars']} hourly candles, `{d['period_start']}` to `{d['period_end']}`,",
        "outer block 3 of the research fold plan. Nothing below has a threshold, and nothing",
        "below may be optimised against.",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    for key in sorted(d):
        lines.append(f"| `{key}` | `{d[key]}` |")
    lines += [
        "",
        "## Not covered",
        "",
    ]
    for item in report["protocol"]["not_covered"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def verify(directory: Path) -> list[str]:
    """Recheck a committed report against the protocol in this build."""
    problems: list[str] = []
    path = directory / REPORT_NAME
    if not path.is_file():
        return [f"{path}: no report"]
    report = json.loads(path.read_text())
    if report.get("report_schema") != REPORT_SCHEMA:
        problems.append(
            f"report schema is {report.get('report_schema')!r}, not {REPORT_SCHEMA!r}"
        )
    if report.get("protocol_hash") != protocol_hash():
        problems.append(
            f"the report ran under protocol {report.get('protocol_hash')} and this build's "
            f"protocol is {protocol_hash()}. A report and a protocol that disagree cannot "
            "both be describing the same validation."
        )
    if report.get("protocol") != PROTOCOL:
        problems.append("the embedded protocol differs from this build's PROTOCOL")
    declared = {i["id"] for i in PROTOCOL["invariants"]}
    observed = {r["id"] for r in report.get("invariants", []) if r["id"] in declared}
    for missing in sorted(declared - observed):
        problems.append(f"invariant {missing} was declared and never observed")
    for record in report.get("invariants", []):
        if not record.get("held"):
            problems.append(f"invariant {record['id']} did not hold")
    if report.get("outcome") != "PASS":
        problems.append(f"the report's outcome is {report.get('outcome')!r}")
    return problems


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "futures_dry_run_v1")
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--protocol", action="store_true", help="print the protocol and exit")
    parser.add_argument("--verify", type=Path, default=None, help="recheck a committed report")
    args = parser.parse_args(argv)

    if args.protocol:
        print(json.dumps({"protocol_hash": protocol_hash(), **PROTOCOL}, indent=2))
        return 0

    if args.verify is not None:
        problems = verify(args.verify)
        if problems:
            print("futures dry-run validation REJECTED")
            for problem in problems:
                print(f"  {problem}")
            return 1
        print(f"futures dry-run validation verified: {protocol_hash()}")
        return 0

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        workdir = args.workdir or Path(tmp)
        report = run(args.root, Path(workdir))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / REPORT_NAME).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (out / STATUS_NAME).write_text(to_status(report))
    print(f"{report['outcome']}: wrote {out / REPORT_NAME}")
    for record in report["invariants"]:
        if not record["held"]:
            print(f"  FAILED {record['id']}: {record['claim']}")
            for note in record["observations"]:
                if note.startswith("FAIL"):
                    print(f"    {note}")
    return 0 if report["outcome"] == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
