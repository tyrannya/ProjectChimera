"""The mechanical guard on P4-HOLD, rows ``[45802, 48211)``.

``docs/p4_preregistration.md`` promises three things about the one region this
programme has never scored: stage 1 cannot reach it, it is opened only after a
frozen stage-1 pass, and it is spent at most once by at most one checkpoint,
ever. A promise in a document is a convention. This module is the mechanism.

**Everything here is outcome-blind.** It reads row indices, a ledger, and the
counters a stage-1 report declares about *the burned exploratory blocks* — the
fold count, the trade counts, the deltas that stage 1 published anyway. It never
reads a price, a label, a return or a prediction from inside the holdout, and no
function here returns holdout data of any kind. Deciding whether the holdout may
be opened must not require having looked at it, which is the whole difficulty
of a one-shot holdout and the reason the check is arithmetic on a gate rather
than a judgement about a result.

**Retirement is not a technicality.** If P4 never spends the region — stage 1
failed, the availability gate failed, the checkpoint was abandoned — the rows
are still retired. By then P4's stage-1 numbers are published, and a later
checkpoint that retunes against them and presents these same rows as a fresh
holdout would be using an adaptive region while calling it independent. The
ledger records the retirement so that the next checkpoint has to argue with a
file rather than with a memory.

Spending the holdout does not make P4 confirmatory. Its maximum label is
*single-region supported*, and nothing in this module changes that.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from nn.p4_preregistration import (
    AVAILABILITY_GATE,
    HOLDOUT_ROWS,
    HOLDOUT_SPEND_POLICY,
    IMPROVED_RULE,
    MIN_OUTER_TRADES,
    RESEARCH_ROWS,
    STAGE_1_CONTINUATION,
    STAGE_1_MAX_ROW_EXCLUSIVE,
    TARGET,
    preregistration_hash,
)

#: Where the ledger lives, relative to the repository root.
LEDGER_PATH = Path("data/research/p4_holdout_ledger.json")

#: The ledger's schema name. A ledger under another schema is a different
#: document and is refused rather than read with defaults.
LEDGER_SCHEMA = "chimera.p4-holdout-ledger/1"

#: The three states the region can be in. There is no path back to `unspent`.
UNSPENT = "unspent"
SPENT = "spent"
RETIRED = "retired"


class HoldoutError(RuntimeError):
    """The P4-HOLD region may not be reached the way it is being reached."""


def _root(root: Path | None) -> Path:
    return Path(root) if root is not None else Path(__file__).resolve().parent.parent


def read_ledger(root: Path | None = None) -> dict[str, Any]:
    """The committed ledger, checked for schema and for the region it names."""
    path = _root(root) / LEDGER_PATH
    if not path.is_file():
        raise HoldoutError(
            f"no P4-HOLD ledger at {path}. The region's state is what says whether it "
            "may be spent; a missing ledger is not an unspent one."
        )
    payload = json.loads(path.read_text())
    if payload.get("ledger_schema") != LEDGER_SCHEMA:
        raise HoldoutError(
            f"{path} declares schema {payload.get('ledger_schema')!r}, not "
            f"{LEDGER_SCHEMA!r}. A ledger under another schema is a different document."
        )
    if list(payload.get("region", [])) != list(HOLDOUT_ROWS):
        raise HoldoutError(
            f"{path} governs rows {payload.get('region')}, but P4-HOLD is "
            f"{list(HOLDOUT_ROWS)}. A ledger for another region governs nothing here."
        )
    if payload.get("state") not in {UNSPENT, SPENT, RETIRED}:
        raise HoldoutError(f"{path} is in unknown state {payload.get('state')!r}")
    return payload


def assert_stage_one_rows(rows: Sequence[int], *, what: str) -> None:
    """Refuse a stage-1 row set that reaches into the holdout.

    ``rows`` is anything expressed in canonical processed-dataset row numbers —
    a fold's outer range, a snapshot's row range, a plan's last row. Stage 1 may
    read rows ``[0, 45802)`` and nothing else.
    """
    reaching = sorted({int(row) for row in rows if int(row) >= STAGE_1_MAX_ROW_EXCLUSIVE})
    if reaching:
        raise HoldoutError(
            f"{what} reaches row {reaching[0]}, which is at or past "
            f"{STAGE_1_MAX_ROW_EXCLUSIVE}. Stage 1 is the exploratory screen and runs "
            f"on the burned blocks only; rows {list(HOLDOUT_ROWS)} are P4-HOLD and are "
            "opened once, under assert_holdout_release, and never by a screen."
        )


def assert_stage_one_bound(end_exclusive: int, *, what: str) -> None:
    """Refuse a stage-1 row *range* whose exclusive end passes the holdout.

    Separate from :func:`assert_stage_one_rows` because the two take different
    things and confusing them is an off-by-one in the direction that matters:
    ``[0, 45802)`` is the whole legal stage-1 space and its end bound is exactly
    the first holdout row, while a stage-1 *index* of 45802 is already inside
    the region.
    """
    if int(end_exclusive) > STAGE_1_MAX_ROW_EXCLUSIVE:
        raise HoldoutError(
            f"{what} ends at row {int(end_exclusive)} exclusive, past "
            f"{STAGE_1_MAX_ROW_EXCLUSIVE}. Stage 1 is the exploratory screen and runs "
            f"on the burned blocks only; rows {list(HOLDOUT_ROWS)} are P4-HOLD and are "
            "opened once, under assert_holdout_release, and never by a screen."
        )


def assert_stage_one_snapshot(manifest_path: Path) -> dict[str, Any]:
    """Refuse a snapshot manifest that a stage-1 run must not be handed.

    The committed research snapshot holds rows ``[0, 45802)`` and therefore
    cannot reach the holdout — but that is a fact about today's file, and a
    later export cut one block longer would silently hand stage 1 the region it
    is screening in order to decide whether to open. This turns the fact into a
    precondition.
    """
    manifest_path = Path(manifest_path)
    payload = json.loads(manifest_path.read_text())
    processed = payload.get("processed_outer_coverage") or {}
    row_range = list(processed.get("row_range") or [0, int(processed.get("rows", 0))])
    assert_stage_one_bound(row_range[1], what=f"the snapshot at {manifest_path}")
    return {
        "manifest": str(manifest_path),
        "row_range": row_range,
        "stage_1_max_row_exclusive": STAGE_1_MAX_ROW_EXCLUSIVE,
    }


def stage_one_gate(report: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the preregistered stage-1 continuation rule to a stage-1 report.

    ``report`` is what a stage-1 run publishes about the *burned* blocks: one
    entry per block with its net-return delta and each arm's outer trade count.
    Nothing here is about the holdout.

    Returns the verdict and every reason behind it, so a refusal names which
    condition failed rather than saying no.
    """
    folds = list(report.get("folds") or [])
    improved_strictly = IMPROVED_RULE["improved_when"] == "delta > 0"
    valid, invalid = [], []
    for fold in folds:
        trades = [int(fold["control_trades"]), int(fold["combined_trades"])]
        (valid if min(trades) >= MIN_OUTER_TRADES else invalid).append(fold)

    deltas = [float(fold["delta"]) for fold in valid]
    improved = [d for d in deltas if (d > 0.0 if improved_strictly else d >= 0.0)]
    mean_delta = sum(deltas) / len(deltas) if deltas else None
    worst = min(deltas) if deltas else None

    reasons: list[str] = []
    if len(valid) < STAGE_1_CONTINUATION["valid_folds_required"]:
        reasons.append(
            f"{len(valid)} valid fold(s); "
            f"{STAGE_1_CONTINUATION['valid_folds_required']} required "
            f"(a fold is valid when both arms realise >= {MIN_OUTER_TRADES} outer trades)"
        )
    if len(improved) < STAGE_1_CONTINUATION["improved_folds_required"]:
        reasons.append(
            f"{len(improved)} improved fold(s) of {len(valid)} valid; "
            f"{STAGE_1_CONTINUATION['improved_folds_required']} required "
            "(improved means delta > 0; a zero delta is not an improvement)"
        )
    if mean_delta is None or mean_delta <= STAGE_1_CONTINUATION["mean_delta_above"]:
        reasons.append(f"mean delta over valid folds is {mean_delta}; must be > 0")
    if worst is None or worst < STAGE_1_CONTINUATION["worst_fold_delta_at_least"]:
        reasons.append(
            f"worst valid fold delta is {worst}; must be at least "
            f"{STAGE_1_CONTINUATION['worst_fold_delta_at_least']}"
        )

    return {
        "passed": not reasons,
        "reasons": reasons,
        "valid_folds": len(valid),
        "invalid_folds": len(invalid),
        "improved_folds": len(improved),
        "mean_delta": mean_delta,
        "worst_fold_delta": worst,
        "rule": dict(STAGE_1_CONTINUATION),
    }


def assert_holdout_release(
    checkpoint: str, stage_one_report: Mapping[str, Any], *, root: Path | None = None
) -> dict[str, Any]:
    """The only door to P4-HOLD. Refuses unless every precondition holds.

    Four preconditions, and all four are checked without reading a holdout row:

    1. the ledger says the region is ``unspent``;
    2. the stage-1 report was produced under *this* preregistration hash, so a
       pass computed under an edited rule cannot open it;
    3. the availability gate passed;
    4. the stage-1 continuation rule passes on that report.

    Returns the release record a caller must write to the ledger. It does not
    write it: spending the region is :func:`record_spend`, and separating the
    two means an exporter cannot half-spend it by crashing.
    """
    ledger = read_ledger(root)
    if ledger["state"] != UNSPENT:
        raise HoldoutError(
            f"P4-HOLD is {ledger['state']} — {ledger.get('reason') or 'no reason recorded'} "
            f"(by {ledger.get('checkpoint') or 'no checkpoint'}). "
            f"{HOLDOUT_SPEND_POLICY['evaluations_permitted']} evaluation by "
            f"{HOLDOUT_SPEND_POLICY['checkpoints_permitted']} checkpoint is permitted, "
            "ever. A second reading of these rows is not a fresh holdout."
        )

    declared = stage_one_report.get("preregistration_hash")
    if declared != preregistration_hash():
        raise HoldoutError(
            f"the stage-1 report was produced under preregistration {declared!r} and "
            f"this checkout is {preregistration_hash()!r}. A pass computed under a "
            "different rule does not open this region."
        )

    availability = stage_one_report.get("availability") or {}
    if not availability.get("gate_passed"):
        raise HoldoutError(
            "the stage-1 report does not record a passing availability gate "
            f"({AVAILABILITY_GATE['requires_exploratory_blocks_available']} exploratory "
            "blocks and P4-HOLD available under the block rule). Insufficient coverage "
            "is not_evaluable, not a licence to evaluate what did survive."
        )

    gate = stage_one_gate(stage_one_report)
    if not gate["passed"]:
        raise HoldoutError(
            "stage 1 did not pass, so P4-HOLD is not opened: " + "; ".join(gate["reasons"])
        )

    return {
        "checkpoint": checkpoint,
        "region": list(HOLDOUT_ROWS),
        "preregistration_hash": preregistration_hash(),
        "stage_one": {k: v for k, v in gate.items() if k != "rule"},
        "evaluation_rows": list(HOLDOUT_ROWS),
        "label_ceiling": HOLDOUT_SPEND_POLICY["does_not_upgrade"],
    }


def _write(ledger: dict[str, Any], root: Path | None) -> dict[str, Any]:
    path = _root(root) / LEDGER_PATH
    path.write_text(json.dumps(ledger, indent=2) + "\n")
    return ledger


def record_spend(release: Mapping[str, Any], *, root: Path | None = None) -> dict[str, Any]:
    """Mark the region spent. Refuses if it already is."""
    ledger = read_ledger(root)
    if ledger["state"] != UNSPENT:
        raise HoldoutError(f"P4-HOLD is already {ledger['state']}; it cannot be spent again")
    ledger.update(
        {
            "state": SPENT,
            "checkpoint": release["checkpoint"],
            "reason": "evaluated once under the preregistered one-shot rule",
            "release": dict(release),
        }
    )
    return _write(ledger, root)


def record_retirement(reason: str, *, root: Path | None = None) -> dict[str, Any]:
    """Retire the region without spending it. Refuses if it is already spent.

    Called when P4 ends without opening the holdout. The rows stay unread and
    stop being available as *independent* evidence, because P4's stage-1 numbers
    are published by then and a later design that reacts to them has made these
    rows adaptive whether or not anyone scored them.
    """
    ledger = read_ledger(root)
    if ledger["state"] == SPENT:
        raise HoldoutError("P4-HOLD is spent; retirement is for a region that was not")
    ledger.update({"state": RETIRED, "reason": reason})
    return _write(ledger, root)


def styx_untouched() -> dict[str, Any]:
    """What this module does *not* reach, asserted rather than promised."""
    return {
        "holdout_last_row_exclusive": HOLDOUT_ROWS[1],
        "research_rows": RESEARCH_ROWS,
        "horizon": TARGET["horizon"],
        "label_of_last_holdout_row_closes_at": HOLDOUT_ROWS[1] - 1 + TARGET["horizon"],
        "sealed_from_row": RESEARCH_ROWS,
        "reaches_sealed_rows": (HOLDOUT_ROWS[1] - 1 + TARGET["horizon"]) >= RESEARCH_ROWS,
    }
