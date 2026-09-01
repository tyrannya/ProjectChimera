"""The shape of the evidence a governed P13 run will one day leave behind.

Deterministic structures only. **This module writes nothing by default, and it
refuses to write anywhere a reader could mistake for governed economic evidence.**
The acquisition and the historical run are later chronology steps; what exists now
is the schema they will fill, exercised against synthetic fixtures.

**The design identity travels with the numbers.** Every evidence object carries
``ACTIVE_DESIGN`` and the active preregistration hash, recomputed from
:mod:`nn.p13_preregistration` at call time rather than pasted. And
:func:`assert_governing_hash` refuses any hash the module itself lists as
SUPERSEDED, so a run cannot claim the original design, P13-A1 or the first
committed P13-A2 as the design it was governed by. That check is cheap and the
failure it prevents is not: an artifact quoting a retired hash is an artifact
whose rules a reader would reconstruct wrongly.

**NOT EVALUABLE is a first-class outcome here, not a missing field.** A terminated
screen produces evidence with the refusal, the state, the instant and the missing
sources — and NO block results, because A2R1 requires that numbers computed before
the refusal never be reported as a partial answer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Sequence

from nn.p13_blocks import BlockRun, ScreenRun, SourceInsufficiency
from nn.p13_carry import BlockResult
from nn.p13_gate import GateResult
from nn.p13_preregistration import (
    ACTIVE_DESIGN,
    EVIDENCE_CEILING,
    SUPERSEDED_HASHES,
    preregistration_hash,
)
from nn.p13_sources import ObjectProvenance
from nn.p13_stress import StressResults

__all__ = [
    "EvidenceError",
    "SCHEMA",
    "FROZEN_ARTIFACT_ROOTS",
    "active_design_identity",
    "assert_governing_hash",
    "block_events",
    "ScreenEvidence",
    "write_evidence",
]

SCHEMA = "chimera.p13-screen-evidence/1"

#: Paths a governed run will one day own. Nothing this module writes may land
#: inside them, because a synthetic fixture sitting where economic evidence
#: belongs is indistinguishable from economic evidence to everything except the
#: person who put it there.
FROZEN_ARTIFACT_ROOTS: tuple[str, ...] = (
    "artifacts/benchmark/btc_p13_carry",
    "artifacts/benchmark/btc_p13_decision",
)


class EvidenceError(RuntimeError):
    """Evidence cannot be assembled or written under the frozen rules."""


def active_design_identity() -> dict[str, Any]:
    """Who governs this run, recomputed rather than quoted."""
    return {
        "checkpoint": "P13",
        "active_design": ACTIVE_DESIGN,
        "preregistration_hash": preregistration_hash(),
        "evidence_ceiling": EVIDENCE_CEILING,
        "superseded_hashes": [entry["hash"] for entry in SUPERSEDED_HASHES],
    }


def assert_governing_hash(candidate: str) -> None:
    """Refuse a hash the preregistration itself marks retired.

    ``SUPERSEDED_HASHES`` lives inside the hashed payload precisely so this check
    is possible from the module alone. A run quoting a retired hash is quoting a
    design whose rules differ from the ones it actually ran under.
    """
    retired = {entry["hash"]: entry for entry in SUPERSEDED_HASHES}
    if candidate in retired:
        entry = retired[candidate]
        raise EvidenceError(
            f"{candidate} is the SUPERSEDED hash of {entry['design']}, retired by "
            f"{entry['superseded_by']}. It is historical provenance only and must not "
            f"govern new economic evidence. The active design is {ACTIVE_DESIGN}, "
            f"{preregistration_hash()}."
        )
    if candidate != preregistration_hash():
        raise EvidenceError(
            f"{candidate} is neither the active preregistration hash "
            f"({preregistration_hash()}) nor any hash this design records as superseded"
        )


def block_events(run: BlockRun) -> tuple[dict[str, Any], ...]:
    """One block's event ledger, in causal order.

    Event-level rather than summary because ``ARTIFACT_POLICY`` asks for it: a
    block total can be reproduced from events, and events cannot be reconstructed
    from a total.
    """
    if not run.quotes:
        return (
            {
                "event": "not_opened",
                "block": run.block.label,
                "reason": run.result.reason,
            },
        )
    events: list[dict[str, Any]] = [
        {
            "event": "open",
            "block": run.block.label,
            "instant_ns": run.quotes[0].instant_ns,
            "calendar_start_ns": run.block.start_ns,
            "delayed": run.opening.delayed,
            "skipped_instants": run.opening.skipped_instants,
            "quantity": str(run.result.quantity),
            "basis_entry": str(run.result.basis_entry),
        }
    ]
    for settlement in run.settlements:
        events.append(
            {
                "event": "funding_settlement",
                "block": run.block.label,
                "instant_ns": settlement.instant_ns,
                "rate": str(settlement.rate),
                "notional_base": str(settlement.mark_price),
            }
        )
    if run.result.liquidated:
        events.append(
            {
                "event": "liquidation_trigger",
                "block": run.block.label,
                "instant_ns": run.result.liquidation_instant_ns,
                "forced_close_instant_ns": run.result.forced_close_instant_ns,
                "gap_ns": run.result.forced_close_gap_ns,
            }
        )
    events.append(
        {
            "event": "unclosed" if run.result.unclosed else "close",
            "block": run.block.label,
            "instant_ns": run.quotes[-1].instant_ns,
            "basis_exit": str(run.result.basis_exit),
            "net_return": str(run.result.net_return),
            "held_bars": run.result.held_bars,
        }
    )
    return tuple(events)


def _block_dict(result: BlockResult) -> dict[str, Any]:
    """One block, with every frozen per-block report field present.

    ``str`` rather than ``float`` throughout: a Decimal rendered through binary
    floating point is a different number, and the whole engine exists because the
    difference is the size of the answer.
    """
    return {
        "block": result.label,
        "opened": result.opened,
        "reason": result.reason,
        "settlements": result.settlements,
        "quantity_btc": str(result.quantity),
        "basis_entry": str(result.basis_entry),
        "basis_exit": str(result.basis_exit),
        "basis_pnl_quote": str(result.basis_pnl),
        "funding_received_quote": str(result.funding_received),
        "funding_paid_quote": str(result.funding_paid),
        "fees_quote": str(result.fees),
        "slippage_quote": str(result.slippage),
        "rebalance_cost_quote": str(result.rebalance_cost),
        "net_pnl_quote": str(result.net_pnl),
        "net_return_fraction": str(result.net_return),
        "max_adverse_excursion_pnl_quote": str(result.max_adverse_excursion_pnl),
        "max_adverse_excursion_fraction": str(result.max_adverse_excursion),
        "liquidated": result.liquidated,
        "liquidation_instant_ns": result.liquidation_instant_ns,
        "forced_close_instant_ns": result.forced_close_instant_ns,
        "unclosed": result.unclosed,
        "thin_sample": result.thin_sample,
        "held_bars": result.held_bars,
        "quote_gap_count": result.quote_gap_count,
        "max_quote_step_ns": result.max_quote_step_ns,
        "liquidation_touch_provenance": result.liquidation_touch_provenance.as_dict(),
    }


@dataclass(frozen=True)
class ScreenEvidence:
    """Everything a governed run must leave behind, assembled deterministically."""

    screen: ScreenRun
    gate: GateResult | None = None
    stresses: StressResults | None = None
    sources: tuple[ObjectProvenance, ...] = ()
    leave_one_out: tuple[tuple[str, Decimal], ...] = ()
    offset_partition: tuple[BlockResult, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        terminal: SourceInsufficiency | None = self.screen.terminal
        payload: dict[str, Any] = {
            "schema": SCHEMA,
            "design": active_design_identity(),
            "sources": [provenance.as_dict() for provenance in self.sources],
            "delayed_opens": [opening.as_dict() for opening in self.screen.openings],
            "evaluable": self.screen.evaluable,
            "notes": list(self.notes),
        }
        if terminal is not None:
            payload["terminal_refusal"] = terminal.as_dict()
            payload["result_state"] = terminal.result_state
            # Explicitly empty, and explicitly explained. A reader must be able to
            # tell "no blocks were reported" from "no blocks were computed".
            payload["blocks"] = []
            payload["gate"] = None
            payload["stresses"] = None
            payload["partial_results_withheld"] = (
                "A2R1 requires that any block economics computed before the terminal "
                "refusal are NOT a result: not written as primary evidence, not reported "
                "as a partial answer, and not admitted to G1-G6. None are reported here."
            )
            payload["not_evaluable_is_not_not_viable"] = (
                "NOT EVALUABLE is source insufficiency for a required risk quantity. It is "
                "not an economic finding and must never be cited as a negative result."
            )
            return payload
        payload["terminal_refusal"] = None
        payload["blocks"] = [_block_dict(run.result) for run in self.screen.blocks]
        payload["events"] = [
            event for run in self.screen.blocks for event in block_events(run)
        ]
        payload["gate"] = None if self.gate is None else self.gate.as_dict()
        payload["result_state"] = None if self.gate is None else self.gate.result_state
        payload["stresses"] = None if self.stresses is None else self.stresses.as_dict()
        payload["diagnostics"] = {
            "D2_leave_one_out_means": [
                {"omitted": label, "mean": str(mean)} for label, mean in self.leave_one_out
            ],
            "D3_offset_partition": [_block_dict(result) for result in self.offset_partition],
        }
        payload["funding_notional_fallback"] = {
            "settlements_on_substituted_base": sum(
                run.mark_substituted_settlements for run in self.screen.blocks
            ),
            "authorised_for": "the FUNDING NOTIONAL BASE only, never for liquidation",
        }
        return payload

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, sort_keys=True)


def write_evidence(evidence: ScreenEvidence, path: str | Path) -> Path:
    """Write evidence to a NON-frozen path, or refuse.

    The refusal is the point. A synthetic fixture written under
    ``artifacts/benchmark/btc_p13_carry/`` or ``btc_p13_decision/`` would sit
    exactly where a governed result belongs, and the tripwire in
    ``tests/test_p13_preregistration.py`` watches the second of those precisely
    because a file appearing there means the design is no longer unrun.
    """
    target = Path(path)
    normalised = target.as_posix()
    for frozen in FROZEN_ARTIFACT_ROOTS:
        if frozen in normalised:
            raise EvidenceError(
                f"refusing to write to {target}: {frozen} is a frozen primary artifact path. "
                "Nothing written by this module is a governed economic result, and putting "
                "one there would make it indistinguishable from one."
            )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(evidence.to_json(), encoding="utf-8")
    return target


def summarise(runs: Iterable[BlockRun]) -> Sequence[dict[str, Any]]:
    """A compact per-block view, for a human reading a test failure."""
    return [
        {
            "block": run.block.label,
            "opened": run.result.opened,
            "opened_at_ns": run.opening.opened_at_ns,
            "held_bars": run.result.held_bars,
            "net_return": str(run.result.net_return),
        }
        for run in runs
    ]
