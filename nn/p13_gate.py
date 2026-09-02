"""The viability gate, and the state machine that keeps four verdicts apart.

``VIABILITY_GATE`` froze G1-G6 before any P13 number existed. This module is that
text as arithmetic and nothing else: every threshold is READ from
:mod:`nn.p13_preregistration` rather than written here, so a threshold cannot be
moved by editing this file — moving one requires moving the preregistration hash,
which is the whole point of freezing it.

**Four states, and the two that must never be confused.**

``NOT YET RUN``
    No screen has been run. The frozen ``CURRENT_RESULT_STATE``, and deliberately
    NOT a member of ``RESULT_STATES``.
``NOT EVALUABLE``
    A required quantity could not be obtained. **Source insufficiency, not
    economics.** This module cannot produce it, and that is structural: it is
    raised by :mod:`nn.p13_blocks` before a gate exists, and :func:`evaluate_gate`
    REFUSES to run on a terminated screen rather than computing something to put
    beside it.
``INVALID``
    Too little was measured to decide — an UNCLOSED block under amendment A1, or
    fewer than the frozen minimum of included blocks.
``VIABLE`` / ``NOT VIABLE``
    The only two states that are claims about carry.

**NaN is load-bearing.** A block that never opened, and the close-dependent
fields of an UNCLOSED one, carry :data:`~nn.p13_carry.NOT_DETERMINABLE` rather
than zero. Every comparison below therefore runs only on blocks that have been
established as INCLUDED first; a rule that forgot would raise
:class:`decimal.InvalidOperation` instead of averaging in a number nobody
measured, which is the failure mode the representation was chosen for.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Sequence

from nn.p13_carry import BlockResult
from nn.p13_preregistration import (
    BREADTH_OF,
    BREADTH_REQUIRED,
    MIN_INCLUDED_BLOCKS,
    MIN_MEAN_NET_RETURN,
    MIN_SETTLEMENTS_PER_BLOCK,
    RESULT_STATES,
    WORST_BLOCK_FLOOR,
)

__all__ = [
    "GateError",
    "VIABLE",
    "NOT_VIABLE",
    "INVALID",
    "NOT_EVALUABLE",
    "NOT_YET_RUN",
    "GATE_CONDITIONS",
    "ConditionResult",
    "GateResult",
    "StressInputs",
    "evaluate_gate",
]


class GateError(RuntimeError):
    """The gate was asked to decide something it is not entitled to decide."""


VIABLE, NOT_VIABLE, INVALID, NOT_EVALUABLE = RESULT_STATES
NOT_YET_RUN = "P13 ALWAYS-ON ANNUAL SPOT/PERP CARRY: NOT YET RUN"

GATE_CONDITIONS: tuple[str, ...] = ("G1", "G2", "G3", "G4", "G5", "G6")

#: The frozen thresholds, parsed once. They are strings in the preregistration so
#: that a decimal literal cannot be rounded into the design by a float; parsing
#: them here keeps the arithmetic exact and the source of truth singular.
_MIN_MEAN = Decimal(MIN_MEAN_NET_RETURN)
_WORST_FLOOR = Decimal(WORST_BLOCK_FLOOR)


@dataclass(frozen=True)
class ConditionResult:
    """One gate condition, its verdict, and the number it was decided on."""

    name: str
    passed: bool
    detail: str

    def as_dict(self) -> dict[str, object]:
        return {"condition": self.name, "passed": self.passed, "detail": self.detail}


@dataclass(frozen=True)
class StressInputs:
    """The two stressed screens ``G5`` gates on, alongside the base one.

    S1 and S3 only. S2 and S4 are ``role: diagnostic only, outside the gate`` in
    the frozen text and are not accepted here at all — a diagnostic that could be
    passed to the gate is a diagnostic one refactor away from gating.
    """

    s1: Sequence[BlockResult]
    s3: Sequence[BlockResult]


@dataclass(frozen=True)
class GateResult:
    """The verdict, and every intermediate a reader would need to check it."""

    result_state: str
    conditions: tuple[ConditionResult, ...]
    included_blocks: tuple[str, ...]
    excluded_blocks: tuple[str, ...]
    positive_blocks: int
    mean_net_return: Decimal | None
    worst_net_return: Decimal | None

    @property
    def passed(self) -> bool:
        return self.result_state == VIABLE

    def as_dict(self) -> dict[str, object]:
        return {
            "result_state": self.result_state,
            "conditions": [condition.as_dict() for condition in self.conditions],
            "included_blocks": list(self.included_blocks),
            "excluded_blocks": list(self.excluded_blocks),
            "positive_blocks": self.positive_blocks,
            "mean_net_return": _text(self.mean_net_return),
            "worst_net_return": _text(self.worst_net_return),
            "breadth_required": BREADTH_REQUIRED,
            "breadth_of": BREADTH_OF,
            "min_mean_net_return": MIN_MEAN_NET_RETURN,
            "worst_block_floor": WORST_BLOCK_FLOOR,
            "min_settlements_per_block": MIN_SETTLEMENTS_PER_BLOCK,
            "min_included_blocks": MIN_INCLUDED_BLOCKS,
        }


def _text(value: Decimal | None) -> str | None:
    return None if value is None else str(value)


def _included(blocks: Sequence[BlockResult]) -> tuple[list[BlockResult], list[BlockResult]]:
    """Split into blocks the gate may read and blocks ``excluded_blocks`` removes.

    A block is INCLUDED when it opened. ``VIABILITY_GATE.liquidated_blocks`` is
    explicit that a liquidated block is "counted as included blocks, at their
    realised negative return. Not excluded", so liquidation is not a reason to
    drop one — only failure to OPEN is.
    """
    included = [block for block in blocks if block.opened]
    excluded = [block for block in blocks if not block.opened]
    return included, excluded


def _breadth_and_mean(blocks: Sequence[BlockResult]) -> tuple[int, Decimal, Decimal]:
    returns = [block.net_return for block in blocks]
    positive = sum(1 for value in returns if value > 0)
    total = sum(returns, Decimal("0"))
    return positive, total / Decimal(len(returns)), min(returns)


def evaluate_gate(
    blocks: Sequence[BlockResult],
    stresses: StressInputs,
    *,
    terminal: object | None = None,
) -> GateResult:
    """Decide the screen, or refuse to.

    ``terminal`` is the source-insufficiency refusal from :mod:`nn.p13_blocks`,
    passed in so this function can REFUSE rather than be trusted not to be called.
    A2 requires that a terminal NOT EVALUABLE bypass gate computation entirely,
    and the only way to require that of a caller is to make the alternative an
    exception.
    """
    if terminal is not None:
        raise GateError(
            "the screen terminated NOT EVALUABLE on source insufficiency, so there is "
            "nothing to gate. A2 makes that outcome terminal and screen-wide, and any "
            "block economics computed before it fired are NOT a result: they do not enter "
            "G1-G6 and are not reported as a partial answer. NOT EVALUABLE is not NOT "
            "VIABLE and must never be presented as one."
        )
    if not blocks:
        raise GateError("no block results were supplied to the gate")

    included, excluded = _included(blocks)
    included_labels = tuple(block.label for block in included)
    excluded_labels = tuple(block.label for block in excluded)

    # Amendment A1 first, because an UNCLOSED block has no determinable return and
    # every condition below would be reading a NaN.
    unclosed = [block.label for block in included if block.unclosed]
    if unclosed:
        return GateResult(
            result_state=INVALID,
            conditions=(
                ConditionResult(
                    "A1",
                    False,
                    f"block(s) {unclosed} opened and could not be closed, so their "
                    "close-dependent economics are NOT DETERMINABLE. Amendment A1 "
                    "terminates the screen INVALID rather than excluding them.",
                ),
            ),
            included_blocks=included_labels,
            excluded_blocks=excluded_labels,
            positive_blocks=0,
            mean_net_return=None,
            worst_net_return=None,
        )

    if len(included) < MIN_INCLUDED_BLOCKS:
        return GateResult(
            result_state=INVALID,
            conditions=(
                ConditionResult(
                    "G4",
                    False,
                    f"{len(included)} included block(s), fewer than the frozen minimum of "
                    f"{MIN_INCLUDED_BLOCKS}. VIABILITY_GATE.excluded_blocks makes this "
                    "INVALID rather than PASS or FAIL: too little was measured to decide.",
                ),
            ),
            included_blocks=included_labels,
            excluded_blocks=excluded_labels,
            positive_blocks=0,
            mean_net_return=None,
            worst_net_return=None,
        )

    positive, mean, worst = _breadth_and_mean(included)
    conditions: list[ConditionResult] = []

    # G1 — strictly positive in at least BREADTH_REQUIRED blocks. Strict, and a
    # block of exactly zero does NOT count: VIABILITY_GATE.tie_handling says so.
    conditions.append(
        ConditionResult(
            "G1",
            positive >= BREADTH_REQUIRED,
            f"{positive} of {len(included)} included blocks strictly positive; "
            f"{BREADTH_REQUIRED} of {BREADTH_OF} required",
        )
    )

    # G2 — strictly positive mean. Exactly zero FAILS.
    conditions.append(
        ConditionResult("G2", mean > 0, f"mean net block return {mean} must exceed 0")
    )

    # G3 — the worst block at or above the floor. Exactly -0.02 PASSES: an
    # inclusive bound, per tie_handling.
    conditions.append(
        ConditionResult(
            "G3",
            worst >= _WORST_FLOOR,
            f"worst block net return {worst} must be at least {_WORST_FLOOR}",
        )
    )

    # G4 — settlements per included block, inclusive at exactly 200, and the
    # minimum number of included blocks.
    thin = [block.label for block in included if block.settlements < MIN_SETTLEMENTS_PER_BLOCK]
    conditions.append(
        ConditionResult(
            "G4",
            not thin and len(included) >= MIN_INCLUDED_BLOCKS,
            f"{len(included)} included blocks (min {MIN_INCLUDED_BLOCKS}); blocks below "
            f"{MIN_SETTLEMENTS_PER_BLOCK} settlements: {thin or 'none'}",
        )
    )

    # G5 — S1 AND S3 must EACH satisfy G1 and G2, and G3 must hold under S1 as
    # well as under S0. Read literally: G3 is NOT required of S3.
    conditions.append(_g5(stresses))

    # G6 — the mean must EXCEED the frozen minimum effect size. Exactly 0.0025
    # FAILS.
    conditions.append(
        ConditionResult(
            "G6",
            mean > _MIN_MEAN,
            f"mean net block return {mean} must exceed the frozen floor {_MIN_MEAN}",
        )
    )

    passed = all(condition.passed for condition in conditions)
    return GateResult(
        result_state=VIABLE if passed else NOT_VIABLE,
        conditions=tuple(conditions),
        included_blocks=included_labels,
        excluded_blocks=excluded_labels,
        positive_blocks=positive,
        mean_net_return=mean,
        worst_net_return=worst,
    )


def _g5(stresses: StressInputs) -> ConditionResult:
    """S1 and S3 each under G1 and G2, plus G3 under S1.

    Scoped exactly as the frozen sentence scopes it. Requiring G3 of S3 as well
    would be a stricter gate than the one that was frozen, and a gate nobody
    preregistered is not the gate this screen is entitled to apply — in either
    direction.
    """
    problems: list[str] = []
    for name, results in (("S1", stresses.s1), ("S3", stresses.s3)):
        included, _ = _included(results)
        if len(included) < MIN_INCLUDED_BLOCKS:
            problems.append(f"{name}: only {len(included)} included blocks")
            continue
        if any(block.unclosed for block in included):
            problems.append(f"{name}: an UNCLOSED block leaves no determinable return")
            continue
        positive, mean, worst = _breadth_and_mean(included)
        if positive < BREADTH_REQUIRED:
            problems.append(f"{name}: G1 fails, {positive} positive blocks")
        if not mean > 0:
            problems.append(f"{name}: G2 fails, mean {mean}")
        if name == "S1" and not worst >= _WORST_FLOOR:
            problems.append(f"S1: G3 fails, worst block {worst}")
    return ConditionResult(
        "G5",
        not problems,
        (
            "; ".join(problems)
            if problems
            else "S1 and S3 both satisfy G1 and G2; G3 holds " "under S1"
        ),
    )
