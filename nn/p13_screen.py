"""The offline P13 screen: aligned sources in, evidence out, no network anywhere.

This is the top of the runtime the frozen design describes. It runs the six
calendar blocks, the two gated stresses, the two diagnostic stresses, the three
payoff-side diagnostics and the gate, and assembles the evidence.

**What it deliberately does not do.** It does not fetch anything, does not know a
URL, and takes :class:`~nn.p13_alignment.AlignedSources` that a caller has already
built from LOCAL bytes. Acquisition is a later chronology step. Running this
against real Binance history is a later chronology step again, and neither has
happened.

**Order of operations, and why it is this order.** Blocks first, because A2's
terminal branch lives there and must be able to stop everything. Stresses next,
over the series the base screen already assembled, so no stress can move an
opening instant. The gate last, and only if the screen is evaluable —
:func:`~nn.p13_gate.evaluate_gate` refuses a terminated screen rather than
trusting this function to have checked.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Sequence

from nn.p13_alignment import AlignedSources
from nn.p13_blocks import (
    BlockError,
    CalendarBlock,
    ScreenRun,
    SourceInsufficiency,
    calendar_blocks,
    run_screen,
)
from nn.p13_carry import Allocation, Costs, Venue
from nn.p13_evidence import ScreenEvidence
from nn.p13_gate import GateResult, StressInputs, evaluate_gate
from nn.p13_preregistration import (
    CAPITAL_CONTRACT,
    COST_MODEL,
    MIN_SETTLEMENTS_PER_BLOCK,
    VENUE_CONSTRAINTS,
)
from nn.p13_sources import RESEARCH_BOUNDARY_NS
from nn.p13_stress import StressResults, leave_one_out_means, run_stresses

__all__ = [
    "frozen_allocation",
    "frozen_costs",
    "frozen_venue",
    "offset_partition",
    "ScreenOutcome",
    "run_offline_screen",
]


def frozen_allocation() -> Allocation:
    """``CAPITAL_CONTRACT``, read rather than restated."""
    return Allocation(
        total_capital=Decimal(CAPITAL_CONTRACT["total_starting_capital"]),
        spot=Decimal(CAPITAL_CONTRACT["spot_allocation"]),
        perp=Decimal(CAPITAL_CONTRACT["perp_margin_allocation"]),
    )


def frozen_costs() -> Costs:
    """``COST_MODEL``, read rather than restated.

    The engine takes costs as an injected value precisely so it cannot invent one;
    this is the single place the frozen numbers enter the runtime, so a run using
    different frictions would have to change the preregistration.
    """
    return Costs(
        spot_fee=Decimal(COST_MODEL["spot_entry_fee_rate"]),
        spot_slippage=Decimal(COST_MODEL["spot_slippage_rate"]),
        perp_fee=Decimal(COST_MODEL["perp_entry_fee_rate"]),
        perp_slippage=Decimal(COST_MODEL["perp_slippage_rate"]),
    )


def frozen_venue() -> Venue:
    """``VENUE_CONSTRAINTS``, with the COARSER step size the frozen text names."""
    perpetual = VENUE_CONSTRAINTS["perpetual"]
    return Venue(
        step_size=Decimal(VENUE_CONSTRAINTS["effective_step_size"].split(",")[0]),
        min_notional=Decimal(perpetual["min_notional"]),
        maintenance_margin_rate=Decimal(perpetual["tier_1_maintenance_margin_rate"]),
    )


def offset_partition() -> tuple[CalendarBlock, ...]:
    """D3: "six blocks offset by six months, truncated at the research boundary".

    Generated literally from that sentence — six one-year blocks whose starts are
    the frozen partition's starts plus six months, each clipped at the boundary.
    The final one begins after the boundary and is therefore empty; it is still
    generated, and reported as a block that could not be opened, because silently
    emitting five blocks from a definition that says six would be the diagnostic
    quietly reshaping itself.

    D3 is a DIAGNOSTIC. ``PAYOFF_DIAGNOSTIC_DISCIPLINE`` and the frozen
    ``role: diagnostic only`` keep it outside the gate, so nothing here can move a
    verdict.
    """
    offset: list[CalendarBlock] = []
    for block in calendar_blocks():
        start = datetime.fromtimestamp(block.start_ns / 1_000_000_000, tz=timezone.utc)
        shifted = datetime(
            start.year + (start.month + 6 > 12),
            (start.month + 6 - 1) % 12 + 1,
            1,
            tzinfo=timezone.utc,
        )
        end = datetime(shifted.year + 1, shifted.month, 1, tzinfo=timezone.utc)
        start_ns = int(shifted.timestamp() * 1_000_000_000)
        end_ns = min(int(end.timestamp() * 1_000_000_000), RESEARCH_BOUNDARY_NS)
        offset.append(
            CalendarBlock(
                label=f"{shifted.year}-{shifted.month:02d}-offset",
                start_ns=start_ns,
                end_exclusive_ns=max(end_ns, start_ns),
            )
        )
    return tuple(offset)


@dataclass(frozen=True)
class ScreenOutcome:
    """Everything one offline screen produced, evaluable or not."""

    screen: ScreenRun
    evidence: ScreenEvidence
    gate: GateResult | None
    stresses: StressResults | None

    @property
    def result_state(self) -> str | None:
        if self.screen.terminal is not None:
            return self.screen.terminal.result_state
        return None if self.gate is None else self.gate.result_state


def run_offline_screen(
    aligned: AlignedSources,
    *,
    allocation: Allocation | None = None,
    costs: Costs | None = None,
    venue: Venue | None = None,
    min_settlements: int = MIN_SETTLEMENTS_PER_BLOCK,
    blocks: Sequence[CalendarBlock] | None = None,
) -> ScreenOutcome:
    """Run the whole screen offline, against sources the caller already loaded."""
    allocation = allocation or frozen_allocation()
    costs = costs or frozen_costs()
    venue = venue or frozen_venue()

    # The bases are DISCARDED here on purpose. Each settlement already carries
    # whether MARK_PRICE_FALLBACK substituted its notional base, and the count of
    # settlements charged on a substituted base is taken PER BLOCK by the
    # evaluator that knows which settlements each block actually applied. This
    # function used to compute one screen-wide total and hand the same number to
    # every block, which the evidence layer then SUMMED — reporting a single
    # substituted settlement six times over.
    settlements, _bases = aligned.settlements(aligned.funding)

    screen = run_screen(
        aligned,
        allocation=allocation,
        costs=costs,
        venue=venue,
        min_settlements=min_settlements,
        settlements=settlements,
        blocks=blocks,
    )
    if not screen.evaluable:
        # Terminal. No stresses, no diagnostics, no gate, and no blocks — every
        # one of which would be a number computed under a screen that A2 has
        # already said produced no result.
        return ScreenOutcome(
            screen=screen,
            evidence=ScreenEvidence(screen=screen, sources=aligned.provenance),
            gate=None,
            stresses=None,
        )

    stresses = run_stresses(
        screen.blocks,
        allocation=allocation,
        costs=costs,
        venue=venue,
        min_settlements=min_settlements,
    )
    gate = evaluate_gate(
        [run.result for run in screen.blocks],
        StressInputs(s1=stresses.s1, s3=stresses.s3),
        terminal=screen.terminal,
    )
    d3 = _offset_partition_results(
        aligned,
        allocation=allocation,
        costs=costs,
        venue=venue,
        min_settlements=min_settlements,
        settlements=settlements,
    )
    evidence = ScreenEvidence(
        screen=screen,
        gate=gate,
        stresses=stresses,
        sources=aligned.provenance,
        leave_one_out=leave_one_out_means([run.result for run in screen.blocks]),
        offset_partition=d3,
    )
    return ScreenOutcome(screen=screen, evidence=evidence, gate=gate, stresses=stresses)


def _offset_partition_results(
    aligned: AlignedSources,
    *,
    allocation: Allocation,
    costs: Costs,
    venue: Venue,
    min_settlements: int,
    settlements: Sequence,
) -> tuple:
    """D3, run over the offset blocks and never allowed to move the verdict.

    A diagnostic that terminated the governed screen would let a partition nobody
    gates on decide the result, so a source insufficiency met only here is caught
    and reported as an empty diagnostic rather than propagated. It cannot arise
    from sources that satisfied the frozen partition — the offset blocks span a
    subset of the same hours — so the catch is defensive, and it is recorded
    rather than swallowed.
    """
    try:
        offset = run_screen(
            aligned,
            allocation=allocation,
            costs=costs,
            venue=venue,
            min_settlements=min_settlements,
            settlements=settlements,
            blocks=offset_partition(),
        )
    except (SourceInsufficiency, BlockError):
        return ()
    if not offset.evaluable:
        return ()
    return tuple(run.result for run in offset.blocks)
