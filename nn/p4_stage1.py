"""P4 stage 1: the exploratory screen, its decision rule, and the interlock.

``docs/p4_preregistration.md`` §8.2 runs three arms against three models over the
four burned outer blocks, reports all nine cells, and lets **one** comparison
decide whether the single-use holdout is opened. This module is that run and
that decision, and it is deliberately also the thing that refuses to start one.

**Nothing here fits a model until a commit says it may.** ``--checkpoint P4`` is
a registered checkpoint — §13's checklist requires it — and registration is not
permission. :func:`assert_fit_authorised` reads
``data/research/p4_stage1_authorisation.json``, which ships as
``not_authorised``, and refuses unless it says otherwise *and* the caller passes
a separate confirmation at the command line. Two independent gates, one of which
is a reviewable diff with a name and a reason on it, because "we implemented the
runner and then it ran" is exactly how a preregistered checkpoint stops being
preregistered.

**The primary comparison is the only one that decides.** Nine cells are produced
and all nine are reported. :func:`stage_one_report` builds the continuation
report from the XGBoost combined-vs-control deltas at the base cost and nothing
else, and it stamps ``decided_by`` so that
:func:`nn.p4_holdout.assert_primary_decision` can refuse a report any other cell
produced. A secondary model that improves while the primary does not is that
finding; it is not a near miss, and there is no code path by which it reaches
P4-HOLD.

**Cost sensitivity is reported, never consulted.** §7.2 fixes 1.0x, 1.5x and
2.0x and says the headline is always 1.0x. :func:`cost_sensitivity` computes all
three for every arm and marks every non-base multiplier ``decides: false``;
:func:`stage_one_report` reads the base row by construction and
``assert_primary_decision`` refuses a report that says otherwise. There is no
outcome in which a worse result at the base cost is redeemed elsewhere.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from chimera.contracts import TargetSpec
from nn import evaluate as ev
from nn.p4_holdout import HoldoutError, assert_primary_decision, assert_stage_one_fold_plan
from nn.p4_preregistration import (
    ARMS,
    COST_SENSITIVITY_MULTIPLIERS,
    EXPLORATORY_OUTER_BLOCKS,
    IMPROVED_RULE,
    MIN_OUTER_TRADES,
    PRIMARY_COMPARISON,
    PRIMARY_MODEL,
    SECONDARY_MODELS,
    STAGE_1_CONTINUATION,
    preregistration_hash,
)

#: Where the interlock lives, relative to the repository root.
AUTHORISATION_PATH = Path("data/research/p4_stage1_authorisation.json")

#: The interlock's schema. A file under another schema is a different document
#: and is refused rather than read with defaults.
AUTHORISATION_SCHEMA = "chimera.p4-stage1-authorisation/1"

AUTHORISED = "authorised"
NOT_AUTHORISED = "not_authorised"

#: The base cost multiplier: the only one the decision rule reads.
BASE_COST_MULTIPLIER = 1.0

#: The full stage-1 matrix, in report order. Three arms, three models, nine
#: cells, all of them reported — hiding six of them would be the multiplicity
#: problem, reporting them is not (§11).
STAGE_1_MODELS: tuple[str, ...] = (PRIMARY_MODEL, *SECONDARY_MODELS)
STAGE_1_CELLS: tuple[tuple[str, str], ...] = tuple(
    (arm, model) for arm in ARMS for model in STAGE_1_MODELS
)


class Stage1Error(RuntimeError):
    """Stage 1 cannot be run, or reported, the way it is being asked for."""


class Stage1Interlock(Stage1Error):
    """No P4 model may be fitted yet, and this says exactly what is missing."""


def _root(root: Path | None) -> Path:
    return Path(root) if root is not None else Path(__file__).resolve().parent.parent


def read_authorisation(root: Path | None = None) -> dict[str, Any]:
    """The committed interlock, checked for schema. Never a default."""
    path = _root(root) / AUTHORISATION_PATH
    if not path.is_file():
        raise Stage1Interlock(
            f"no P4 stage-1 authorisation at {path}. A missing interlock is not an open "
            "one: the file is what records that a human decided this checkpoint may "
            "begin fitting, and under whose name."
        )
    payload = json.loads(path.read_text())
    if payload.get("authorisation_schema") != AUTHORISATION_SCHEMA:
        raise Stage1Interlock(
            f"{path} declares schema {payload.get('authorisation_schema')!r}, not "
            f"{AUTHORISATION_SCHEMA!r}. A file under another schema is a different "
            "document and does not authorise anything here."
        )
    if payload.get("state") not in {AUTHORISED, NOT_AUTHORISED}:
        raise Stage1Interlock(f"{path} is in unknown state {payload.get('state')!r}")
    return payload


def assert_fit_authorised(
    *, confirm: bool, availability: Mapping[str, Any] | None = None, root: Path | None = None
) -> dict[str, Any]:
    """The only door to a P4 model fit. Refuses unless every precondition holds.

    Four preconditions, and none of them is a judgement call:

    1. the committed authorisation says ``authorised``;
    2. it was granted under *this* preregistration hash, so an authorisation
       given for one design does not carry over to an edited one;
    3. the caller passed the separate command-line confirmation, so a Make
       target or a script cannot start a fit by existing;
    4. the availability gate passed — an intersection that cannot support the
       design makes P4 ``not_evaluable`` (§9.0), and stage 1 is not run on the
       blocks that happened to survive.

    Returns the record a cell must carry, so an artifact says which
    authorisation it was produced under rather than that one existed.
    """
    payload = read_authorisation(root)
    if payload["state"] != AUTHORISED:
        raise Stage1Interlock(
            f"P4 stage 1 is {payload['state']}. The machinery is implemented and its "
            "inputs are frozen; beginning to fit is a separate, deliberate act. Set "
            f"state to {AUTHORISED!r} in {AUTHORISATION_PATH} — with a name and a reason "
            "— and commit it, then re-run with the command-line confirmation."
        )
    declared = payload.get("preregistration_hash")
    if declared != preregistration_hash():
        raise Stage1Interlock(
            f"the authorisation was granted under preregistration {declared!r} and this "
            f"checkout is {preregistration_hash()!r}. Permission to run one design is "
            "not permission to run another."
        )
    if not confirm:
        raise Stage1Interlock(
            "the authorisation file permits a P4 fit and this run did not confirm it. "
            "Pass --authorise-fit. The second gate exists so that a target, a script or "
            "a stale shell command cannot begin the checkpoint on its own."
        )
    if availability is None or not availability.get("gate_passed"):
        raise Stage1Interlock(
            "the availability gate has not passed, so stage 1 does not run. Insufficient "
            "coverage is not_evaluable / insufficient_coverage (§9.0) — a fact about "
            "what public archives contain — and it is not a licence to screen on the "
            "blocks that did survive."
        )
    return {
        "state": payload["state"],
        "authorised_by": payload.get("authorised_by"),
        "authorised_at": payload.get("authorised_at"),
        "reason": payload.get("reason"),
        "preregistration_hash": declared,
        "confirmed_at_the_command_line": True,
        "availability_gate_passed": True,
    }


# --------------------------------------------------------------------------- #
# cost sensitivity
# --------------------------------------------------------------------------- #
def scaled_target_spec(target_spec: TargetSpec, multiplier: float) -> TargetSpec:
    """``target_spec`` with the round-trip cost multiplied, and nothing else moved.

    The horizon is unchanged and the fee/slippage split is scaled together, so
    the sensitivity charges more per trade without becoming a different label,
    a different holding period or a different experiment. The *labels* are not
    recomputed: §7.2's sensitivity is a sensitivity of the net return to the
    cost charged, not a re-run of the checkpoint under another target.
    """
    if multiplier <= 0:
        raise Stage1Error(f"a cost multiplier of {multiplier} is not a cost")
    return replace(
        target_spec,
        fee_rate=target_spec.fee_rate * multiplier,
        slippage_rate=target_spec.slippage_rate * multiplier,
    )


def net_return_at_cost(
    signals: np.ndarray,
    future_return: np.ndarray,
    target_spec: TargetSpec,
    multiplier: float,
    *,
    row_index: np.ndarray | None = None,
) -> dict[str, Any]:
    """One arm's outer net return with the round-trip cost scaled by ``multiplier``.

    Goes through :func:`nn.evaluate.realised_trades` rather than adjusting a
    number afterwards, so the sensitivity and the headline share one definition
    of what a trade is and what it costs. The trades themselves are identical at
    every multiplier — the cost does not enter the signal or the non-overlap
    rule — which is what makes this a sensitivity rather than a second run.
    """
    spec = scaled_target_spec(target_spec, multiplier)
    _, _, returns = ev.realised_trades(signals, future_return, spec, row_index=row_index)
    net = float(np.prod(1.0 + returns) - 1.0) if len(returns) else 0.0
    return {
        "cost_multiplier": float(multiplier),
        "cost_threshold": round(spec.cost_threshold, 6),
        "n_trades": int(len(returns)),
        "net_return": round(net, 6),
        "decides": float(multiplier) == BASE_COST_MULTIPLIER,
    }


def cost_sensitivity(
    signals: np.ndarray,
    future_return: np.ndarray,
    target_spec: TargetSpec,
    *,
    row_index: np.ndarray | None = None,
) -> dict[str, Any]:
    """§7.2's three multipliers, for one arm on one block. All reported, one decides."""
    rows = [
        net_return_at_cost(
            signals, future_return, target_spec, multiplier, row_index=row_index
        )
        for multiplier in COST_SENSITIVITY_MULTIPLIERS
    ]
    deciding = [row for row in rows if row["decides"]]
    if len(deciding) != 1:
        raise Stage1Error(
            f"{len(deciding)} of the preregistered cost multipliers "
            f"{list(COST_SENSITIVITY_MULTIPLIERS)} claim to decide; exactly one — the "
            "base cost — may."
        )
    return {
        "multipliers": rows,
        "headline": deciding[0],
        "rule": (
            "the headline is always the base cost. The other multipliers are reported "
            "for every arm whatever they show, and there is no outcome in which a worse "
            "result at 1.0x is redeemed by a better one at another multiplier"
        ),
    }


# --------------------------------------------------------------------------- #
# the screen
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BlockResult:
    """What one available outer block produced for the deciding comparison."""

    block: tuple[int, int]
    control_net_return: float
    combined_net_return: float
    control_trades: int
    combined_trades: int

    @property
    def delta(self) -> float:
        return self.combined_net_return - self.control_net_return

    @property
    def valid(self) -> bool:
        """§8.4: *each* compared arm must realise at least ten outer trades."""
        return min(self.control_trades, self.combined_trades) >= MIN_OUTER_TRADES

    def to_dict(self) -> dict[str, Any]:
        # Nothing here is rounded, and that is the point. §8.2 makes "improved"
        # a strict inequality against zero, so a delta rounded to six places
        # would let the *reporting* decide the screen: a positive delta smaller
        # than 5e-7 would be published as 0.0 and counted as the null. Rounding
        # belongs in the rendering of a number, never in the number a rule reads.
        return {
            "block": list(self.block),
            "control_net_return": self.control_net_return,
            "combined_net_return": self.combined_net_return,
            "control_trades": self.control_trades,
            "combined_trades": self.combined_trades,
            "delta": self.delta,
            "valid": self.valid,
            "improved": self.valid and self.delta > 0.0,
            "min_outer_trades": MIN_OUTER_TRADES,
        }


def stage_one_report(
    blocks: Sequence[BlockResult],
    *,
    availability: Mapping[str, Any],
    universe: Mapping[str, Any],
    provenance: Mapping[str, Any],
    frozen_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The one document that can open P4-HOLD, and the only shape that can.

    ``blocks`` carries the *deciding* comparison — combined against control, on
    the primary model, at the base cost — for each outer block that was
    **available** under §8.0. A block that failed the availability rule is not
    passed here at all: §8.0 says it is excluded from the fold count entirely and
    never included at a discount.

    Invalid folds (§8.4) are kept in the report and excluded from both the
    numerator and the denominator of the screen, which is what
    :func:`nn.p4_holdout.stage_one_gate` does with them. Reporting them is not
    optional: a screen that quietly dropped its thin folds would be a different
    rule from the one preregistered.
    """
    available_blocks = {
        tuple(entry["block"])
        for entry in availability["exploratory_blocks"]
        if entry["available"]
    }
    for result in blocks:
        if tuple(result.block) not in available_blocks:
            raise Stage1Error(
                f"block {list(result.block)} is reported as a fold of the screen but the "
                "availability gate did not mark it available. §8.0 excludes an "
                "unavailable block from the fold count entirely; it is never included "
                "at a discount."
            )
    report = {
        "checkpoint": "P4",
        "stage": 1,
        "preregistration_hash": preregistration_hash(),
        "decided_by": {
            "model": PRIMARY_MODEL,
            "comparison": list(PRIMARY_COMPARISON),
            "cost_multiplier": BASE_COST_MULTIPLIER,
        },
        "secondary_models": list(SECONDARY_MODELS),
        "secondary_models_role": (
            "run and reported in full over all nine cells; descriptive only. A secondary "
            "model that improves while the primary does not is reported as exactly that "
            "and cannot change the answer or open P4-HOLD"
        ),
        "availability": dict(availability),
        "universe": dict(universe),
        "provenance": dict(provenance),
        "min_outer_trades": MIN_OUTER_TRADES,
        "improved_rule": dict(IMPROVED_RULE),
        "continuation_rule": dict(STAGE_1_CONTINUATION),
        "evidence_class": "exploratory screen on burned blocks; not a research result",
        "folds": [result.to_dict() for result in blocks],
    }
    if frozen_evidence is not None:
        report["frozen_evidence"] = dict(frozen_evidence)
    # Built and then checked by the same function the holdout door uses, so a
    # report this module produced cannot be one that door would refuse for a
    # reason this module did not think of.
    assert_primary_decision(report)
    return report


def screen(report: Mapping[str, Any]) -> dict[str, Any]:
    """Apply §8.2's continuation rule, through the holdout module's own gate.

    Deliberately a thin wrapper: the rule that decides whether to continue and
    the rule that opens the region must be one implementation, or a run could
    report "continued" while the door disagreed.
    """
    from nn.p4_holdout import stage_one_gate

    assert_primary_decision(report)
    gate = stage_one_gate(report)
    return {
        **gate,
        "outcome": "continue_to_stage_2" if gate["passed"] else "screened_out",
        "classification": (
            "exploratory; a pass is a decision to spend one holdout evaluation and is "
            "not a finding"
            if gate["passed"]
            else "screened_out: negative for this design at this horizon, on burned "
            "blocks. The holdout is not opened and there is no re-fit"
        ),
    }


def assert_stage_one_geometry(folds: Sequence[Any]) -> dict[str, Any]:
    """The fold plan is the preregistered one, and it stops before P4-HOLD.

    Two checks in one place because they fail the same way if either is skipped:
    a plan that reached row 45802 would let the screen read the region it screens
    in order to decide whether to open, and a plan that did not match §8.2's four
    blocks would answer a question with the right name and the wrong geometry.
    """
    bound = assert_stage_one_fold_plan(folds, what="the P4 stage-1 fold plan")
    planned = [tuple(block) for block in bound["outer_blocks"]]
    expected = [tuple(block) for block in EXPLORATORY_OUTER_BLOCKS]
    if planned != expected:
        raise Stage1Error(
            f"the fold plan's outer blocks are {planned}, not the preregistered "
            f"{expected}. Stage 1 screens on the four burned blocks and on nothing else."
        )
    return {**bound, "matches_preregistered_blocks": True}


def cells_to_run() -> list[dict[str, Any]]:
    """The nine cells of §10, in report order, with which one decides.

    A list rather than a loop body, so ``--plan`` can show the whole matrix and a
    test can assert it is nine and that exactly one of them is primary.
    """
    return [
        {
            "information_set": arm,
            "model": model,
            "decides": model == PRIMARY_MODEL
            and set(PRIMARY_COMPARISON) == {arm, PRIMARY_COMPARISON[1]}
            and arm == PRIMARY_COMPARISON[0],
            "role": "primary" if model == PRIMARY_MODEL else "secondary (descriptive)",
        }
        for arm, model in STAGE_1_CELLS
    ]


def describe(availability: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """What stage 1 would run, and what is stopping it. No fit, no network.

    The answer to "is P4 running yet?" in one object: the matrix, the decision
    rule, the interlock's state, and the reason a fit would be refused right now.
    """
    try:
        authorisation = read_authorisation()
        state = authorisation["state"]
        granted_for = authorisation.get("preregistration_hash")
    except Stage1Interlock as exc:
        state, granted_for = f"unreadable: {exc}", None
    try:
        assert_fit_authorised(confirm=True, availability=availability)
        blocked_by = None
    except (Stage1Interlock, HoldoutError) as exc:
        blocked_by = str(exc)
    return {
        "checkpoint": "P4",
        "preregistration_hash": preregistration_hash(),
        "cells": cells_to_run(),
        "cell_count": len(STAGE_1_CELLS),
        "primary_model": PRIMARY_MODEL,
        "secondary_models": list(SECONDARY_MODELS),
        "primary_comparison": list(PRIMARY_COMPARISON),
        "cost_multipliers": list(COST_SENSITIVITY_MULTIPLIERS),
        "base_cost_multiplier": BASE_COST_MULTIPLIER,
        "outer_blocks": [list(block) for block in EXPLORATORY_OUTER_BLOCKS],
        "continuation_rule": dict(STAGE_1_CONTINUATION),
        "interlock": {
            "path": str(AUTHORISATION_PATH),
            "state": state,
            "granted_for_preregistration_hash": granted_for,
            "requires_command_line_confirmation": True,
        },
        "fit_would_be_refused_because": blocked_by,
        "fits_run": 0,
    }
