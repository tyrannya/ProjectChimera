"""S1-S4 and D1-D3, exactly as ``STRESS_CASES`` and ``PAYOFF_SIDE_DIAGNOSTICS`` froze them.

Every variant here re-evaluates a block's ALREADY-ASSEMBLED quote series. None of
them re-runs the opening search or rebuilds the held window, and that is a
correctness property rather than an optimisation: a stress is a perturbation of
COSTS or of the BASIS, never a statement about which hours the archives published.
A variant that re-derived the opening instant could open somewhere else and report
the difference as a friction effect.

**The gate/diagnostic split is enforced by types, not by discipline.** ``G5`` gates
on S1 and S3 only; ``S2`` and ``S4`` are ``role: diagnostic only, outside the
gate`` and :class:`~nn.p13_gate.StressInputs` has no field to put them in. A
future refactor that wanted to gate on S4 would have to change the gate's
signature, which is exactly the kind of change a reviewer notices.

**Nothing here is a tuning surface.** The S1 factor, the S3 magnitude and the D1
factor are read from the frozen design or written as the single literal the frozen
design names, and ``STRESS_DISCIPLINE`` forbids revising any of them after seeing a
result.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Sequence

from nn.p13_blocks import BlockRun
from nn.p13_carry import (
    Allocation,
    BlockResult,
    Costs,
    FundingSettlement,
    Quote,
    Venue,
    evaluate_block,
)

__all__ = [
    "S1_FRICTION_FACTOR",
    "S3_BASIS_SHIFT",
    "D1_FUNDING_FACTOR",
    "higher_friction_costs",
    "adverse_basis_quotes",
    "delayed_hedge_quotes",
    "halved_funding",
    "rerun",
    "StressResults",
    "run_stresses",
    "leave_one_out_means",
]

#: S1: "every fee and slippage rate in COST_MODEL doubled".
S1_FRICTION_FACTOR = Decimal("2")

#: S3: "a fixed 10 bps of spot at each end". Ten basis points is 0.0010 as a
#: fraction, and the frozen text pins the arithmetic that follows from it — "a
#: total charge of 20 bps of spot notional, about 0.0010 of total capital" — which
#: is the check that this constant is the intended one rather than 0.010.
S3_BASIS_SHIFT = Decimal("0.0010")

#: D1: "every realised funding rate multiplied by 0.5".
D1_FUNDING_FACTOR = Decimal("0.5")


def higher_friction_costs(costs: Costs) -> Costs:
    """S1. Uses the engine's own ``scaled``, so one definition of doubling exists."""
    return costs.scaled(S1_FRICTION_FACTOR)


def adverse_basis_quotes(quotes: Sequence[Quote]) -> tuple[Quote, ...]:
    """S3: move the basis AGAINST the position by 10 bps of spot at each end.

    The frozen definition names both the magnitude and the DIRECTION, because
    "worsened" is not a direction: the ENTRY basis is REDUCED and the EXIT basis is
    INCREASED. Since the hedged price PnL is ``Q x (basis_in - basis_out)``, both
    moves subtract.

    Applied to the PERPETUAL leg's fill, because ``fill_basis`` is
    ``perp_fill - spot_fill`` and the shift is denominated in spot: lowering the
    perpetual open at entry lowers the entry basis by exactly ``0.0010 x spot``,
    and raising it at exit raises the exit basis by the same. The spot leg is left
    alone so the shift cannot leak into the capital the position sized itself on.

    Only the two FILL bars are touched. The bars in between are marks, and moving
    a mark would perturb the liquidation test and the excursion, which is a
    different stress from the one that was frozen.
    """
    if len(quotes) < 2:
        return tuple(quotes)
    first, last = quotes[0], quotes[-1]
    entry = replace(
        first, perp_open=_positive(first.perp_fill - S3_BASIS_SHIFT * first.spot_fill)
    )
    exit_ = replace(last, perp_open=last.perp_fill + S3_BASIS_SHIFT * last.spot_fill)
    return (entry, *quotes[1:-1], exit_)


def _positive(value: Decimal) -> Decimal:
    """A stressed price must still be a price.

    A 10 bp shift cannot make a BTC price non-positive at any level this
    checkpoint sees, so reaching this guard means the inputs were not what the
    design assumes — and a stress that silently produced a non-positive fill would
    be refused by :class:`~nn.p13_carry.Quote` anyway, one layer later and with a
    less useful message.
    """
    if value <= 0:
        raise ValueError(f"the S3 basis shift produced a non-positive fill price {value}")
    return value


def delayed_hedge_quotes(quotes: Sequence[Quote], *, delay_spot: bool) -> tuple[Quote, ...]:
    """S2: one leg opened a bar late and closed a bar early, leaving directional ends.

    The DELAYED leg enters at bar 1's open instead of bar 0's and exits at bar
    N-1's open instead of bar N's; the other leg keeps both. The position is
    therefore genuinely directional for one bar at each end, which is the whole
    point of the diagnostic.

    ``BASIS_DEFINITION.identity_scope`` records that the telescoping basis identity
    does NOT hold here, so no caller may assert it against an S2 result.
    """
    if len(quotes) < 4:
        raise ValueError("S2 needs at least four bars to delay a leg at both ends")
    first, second = quotes[0], quotes[1]
    penultimate, last = quotes[-2], quotes[-1]
    if delay_spot:
        entry = replace(first, spot_open=second.spot_fill)
        exit_ = replace(last, spot_open=penultimate.spot_fill)
    else:
        entry = replace(first, perp_open=second.perp_fill)
        exit_ = replace(last, perp_open=penultimate.perp_fill)
    return (entry, *quotes[1:-1], exit_)


def halved_funding(
    settlements: Sequence[FundingSettlement],
) -> tuple[FundingSettlement, ...]:
    """D1: every realised rate multiplied by 0.5, on the same instants and bases."""
    return tuple(
        FundingSettlement(
            instant_ns=settlement.instant_ns,
            rate=settlement.rate * D1_FUNDING_FACTOR,
            mark_price=settlement.mark_price,
        )
        for settlement in settlements
    )


def rerun(
    run: BlockRun,
    *,
    allocation: Allocation,
    costs: Costs,
    venue: Venue,
    min_settlements: int,
    quotes: Sequence[Quote] | None = None,
    settlements: Sequence[FundingSettlement] | None = None,
    isolated: bool = False,
) -> BlockResult:
    """Re-evaluate one already-assembled block under a variant.

    A block that never opened has no assembled series to perturb, and a variant of
    "not opened" is still "not opened" — returned unchanged rather than
    reconstructed, so a stress cannot resurrect a block the construction refused.
    """
    if not run.quotes:
        return run.result
    return evaluate_block(
        run.block.label,
        tuple(quotes) if quotes is not None else run.quotes,
        tuple(settlements) if settlements is not None else run.settlements,
        allocation,
        costs,
        venue,
        min_settlements,
        isolated=isolated,
    )


@dataclass(frozen=True)
class StressResults:
    """Every predeclared stress and diagnostic, per block, in one place.

    S1 and S3 are the two the gate reads. S2 and S4 sit beside them and are never
    handed to it.
    """

    s0: tuple[BlockResult, ...]
    s1: tuple[BlockResult, ...]
    s2: tuple[BlockResult, ...]
    s3: tuple[BlockResult, ...]
    s4: tuple[BlockResult, ...]
    d1: tuple[BlockResult, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            name: [
                {
                    "block": result.label,
                    "opened": result.opened,
                    "net_return": str(result.net_return),
                    "liquidated": result.liquidated,
                }
                for result in getattr(self, name)
            ]
            for name in ("s0", "s1", "s2", "s3", "s4", "d1")
        }


def run_stresses(
    runs: Sequence[BlockRun],
    *,
    allocation: Allocation,
    costs: Costs,
    venue: Venue,
    min_settlements: int,
) -> StressResults:
    """Every stress and D1, over the blocks the base screen already assembled."""
    s0 = tuple(run.result for run in runs)
    s1 = tuple(
        rerun(
            run,
            allocation=allocation,
            costs=higher_friction_costs(costs),
            venue=venue,
            min_settlements=min_settlements,
        )
        for run in runs
    )
    s3 = tuple(
        rerun(
            run,
            allocation=allocation,
            costs=costs,
            venue=venue,
            min_settlements=min_settlements,
            quotes=adverse_basis_quotes(run.quotes) if run.quotes else None,
        )
        for run in runs
    )
    s4 = tuple(
        rerun(
            run,
            allocation=allocation,
            costs=costs,
            venue=venue,
            min_settlements=min_settlements,
            isolated=True,
        )
        for run in runs
    )
    d1 = tuple(
        rerun(
            run,
            allocation=allocation,
            costs=costs,
            venue=venue,
            min_settlements=min_settlements,
            settlements=halved_funding(run.settlements),
        )
        for run in runs
    )
    s2 = tuple(
        _worse_of_both_delays(
            run,
            allocation=allocation,
            costs=costs,
            venue=venue,
            min_settlements=min_settlements,
        )
        for run in runs
    )
    return StressResults(s0=s0, s1=s1, s2=s2, s3=s3, s4=s4, d1=d1)


def _worse_of_both_delays(
    run: BlockRun,
    *,
    allocation: Allocation,
    costs: Costs,
    venue: Venue,
    min_settlements: int,
) -> BlockResult:
    """S2 evaluated BOTH WAYS, reporting the worse.

    The frozen definition insists on it: "A one-sided delay is not a stress: in a
    rising sample, hedging late is a benefit, and reporting only that ordering
    would dress a windfall as a robustness check."
    """
    if len(run.quotes) < 4:
        return run.result
    outcomes = [
        rerun(
            run,
            allocation=allocation,
            costs=costs,
            venue=venue,
            min_settlements=min_settlements,
            quotes=delayed_hedge_quotes(run.quotes, delay_spot=delay_spot),
        )
        for delay_spot in (True, False)
    ]
    opened = [outcome for outcome in outcomes if outcome.opened and not outcome.unclosed]
    if len(opened) < len(outcomes):
        # A delay that cannot be evaluated is reported as itself rather than
        # silently replaced by the orderings that could be.
        return next(outcome for outcome in outcomes if outcome not in opened)
    return min(opened, key=lambda outcome: outcome.net_return)


def leave_one_out_means(results: Sequence[BlockResult]) -> tuple[tuple[str, Decimal], ...]:
    """D2: the mean recomputed once per omitted block, reported as the full set.

    Only INCLUDED blocks contribute, for the same reason G2 reads only included
    blocks: a block that never opened has no return to omit or to average.
    """
    included = [result for result in results if result.opened and not result.unclosed]
    if len(included) < 2:
        return ()
    out: list[tuple[str, Decimal]] = []
    for omitted in included:
        rest = [result.net_return for result in included if result.label != omitted.label]
        out.append((omitted.label, sum(rest, Decimal("0")) / Decimal(len(rest))))
    return tuple(out)
