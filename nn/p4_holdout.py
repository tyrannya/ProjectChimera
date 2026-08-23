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

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from nn.p4_preregistration import (
    AVAILABILITY_GATE,
    COST_SENSITIVITY_MULTIPLIERS,
    HOLDOUT_ROWS,
    HOLDOUT_SPEND_POLICY,
    IMPROVED_RULE,
    MIN_OUTER_TRADES,
    PRIMARY_COMPARISON,
    PRIMARY_MODEL,
    RESEARCH_ROWS,
    SECONDARY_MODELS,
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


def holdout_first_instant(root: Path | None = None) -> Any:
    """The first UTC instant inside P4-HOLD, from the ledger's declared span.

    A *timestamp*, not a row — because the derivatives source is an hourly table
    and its bound has to be checked in the space it is expressed in. Nothing
    inside the region is read to produce it: the span is a property of where the
    region starts, it is published in ``docs/p4_preregistration.md`` and in the
    ledger, and :func:`check_holdout_boundary` proves the ledger's copy agrees
    with the committed snapshot rather than trusting it.
    """
    from datetime import datetime

    ledger = read_ledger(root)
    span = str(ledger.get("region_span") or "")
    first = span.split("..")[0].strip()
    if not first:
        raise HoldoutError(
            "the P4-HOLD ledger declares no region_span, so the instant stage 1's "
            "sources must stop before is unknown. Refusing to guess it."
        )
    return datetime.fromisoformat(first)


def check_holdout_boundary(manifest_path: Path) -> dict[str, Any]:
    """Prove the ledger's declared holdout instant is the committed spine's own next hour.

    The ledger is a file, and a file can be edited. This recomputes the boundary
    from the committed research snapshot — the last stage-1 hour, plus one — and
    refuses if the two disagree. Reading the snapshot's *end timestamp* is not
    reading the holdout: the snapshot stops at row 45802 and physically does not
    contain it.
    """
    import pandas as pd

    payload = json.loads(Path(manifest_path).read_text())
    processed = payload["processed_outer_coverage"]
    spine_end = pd.Timestamp(processed["end"]).tz_convert("UTC")
    expected = spine_end + pd.Timedelta(hours=1)
    declared = pd.Timestamp(holdout_first_instant()).tz_convert("UTC")
    if declared != expected:
        raise HoldoutError(
            f"the ledger says P4-HOLD begins at {declared.isoformat()} but the committed "
            f"snapshot's last stage-1 hour is {spine_end.isoformat()}, so the region "
            f"begins at {expected.isoformat()}. One of the two is wrong about where the "
            "screen stops and neither may guess."
        )
    return {
        "p4_hold_first_instant": declared.isoformat(),
        "stage_1_last_instant": spine_end.isoformat(),
        "derived_from": str(manifest_path),
    }


def assert_stage_one_instants(dates: Any, *, what: str, root: Path | None = None) -> None:
    """Refuse a stage-1 *table* that reaches P4-HOLD's first instant.

    The row guards above speak in canonical dataset rows, which the derivatives
    source does not have: it is an hourly grid that starts before row 0 and is
    joined to the spine by timestamp. Without this, the one file P4 adds to the
    research inputs would be the one input no holdout guard could see.
    """
    import pandas as pd

    stamps = pd.DatetimeIndex(pd.to_datetime(dates, utc=True))
    if len(stamps) == 0:
        return
    boundary = pd.Timestamp(holdout_first_instant(root)).tz_convert("UTC")
    reaching = stamps[stamps >= boundary]
    if len(reaching):
        raise HoldoutError(
            f"{what} reaches {reaching[0].isoformat()}, at or past P4-HOLD's first "
            f"instant {boundary.isoformat()} ({len(reaching)} hour(s) in total). Stage "
            "1's inputs must structurally not contain the region stage 1 decides "
            "whether to open."
        )


def assert_stage_one_fold_plan(folds: Iterable[Any], *, what: str) -> dict[str, Any]:
    """Refuse a fold plan whose blocks reach the holdout.

    Takes the plan rather than the data, because the plan is what decides which
    rows are read: a snapshot bounded correctly and a plan bounded wrongly is a
    run that fails on an index error somewhere far from the reason.
    """
    blocks = []
    for plan in folds:
        outer = getattr(plan, "outer", plan)
        if hasattr(outer, "start") and hasattr(outer, "end"):
            start, end = int(outer.start), int(outer.end)
        else:
            start, end = int(outer[0]), int(outer[1])
        assert_stage_one_bound(end, what=f"{what} outer block [{start}, {end})")
        blocks.append([start, end])
    return {"outer_blocks": blocks, "stage_1_max_row_exclusive": STAGE_1_MAX_ROW_EXCLUSIVE}


def assert_primary_decision(report: Mapping[str, Any]) -> dict[str, Any]:
    """Refuse a stage-1 report that any cell other than the primary one decided.

    ``docs/p4_preregistration.md`` §10 runs three models over three arms and
    reports all nine, and lets exactly one comparison decide: XGBoost, combined
    against control, at the base cost. The two secondary families are described
    as unable to change the answer — this is the sentence that makes that true
    rather than promised, and it is checked on the way *into* the holdout door
    rather than trusted from the runner that wrote the report.

    A cost multiplier other than 1.0 is refused for the same reason. §7.2 says
    the headline is always 1.0x and that there is no outcome in which a worse
    result at the base cost is redeemed by a better one elsewhere; a release
    computed against 1.5x would be exactly that outcome.
    """
    decided = report.get("decided_by") or {}
    model = decided.get("model")
    if model != PRIMARY_MODEL:
        raise HoldoutError(
            f"the stage-1 report says it was decided by {model!r}. Only {PRIMARY_MODEL!r} "
            f"decides: {list(SECONDARY_MODELS)} are secondary, are reported in full, and "
            "cannot open this region. A secondary model that improved while the primary "
            "did not is that finding, not a near miss."
        )
    comparison = list(decided.get("comparison") or [])
    if comparison != list(PRIMARY_COMPARISON):
        raise HoldoutError(
            f"the stage-1 report says it was decided by the comparison {comparison}, not "
            f"{list(PRIMARY_COMPARISON)}. One comparison decides; every other pairing is "
            "reported and is not a second route to the holdout."
        )
    multiplier = decided.get("cost_multiplier")
    if multiplier is None or float(multiplier) != 1.0:
        raise HoldoutError(
            f"the stage-1 report says it was decided at {multiplier}x the round-trip "
            f"cost. The base cost decides; {list(COST_SENSITIVITY_MULTIPLIERS)} are all "
            "reported for every arm and none of them is a rescue."
        )
    return {
        "model": model,
        "comparison": comparison,
        "cost_multiplier": float(multiplier),
        "secondary_models_cannot_decide": list(SECONDARY_MODELS),
    }


def assert_frozen_stage_one(
    report: Mapping[str, Any], *, root: Path | None = None
) -> dict[str, Any]:
    """Refuse a stage-1 report that is not frozen evidence on disk.

    ``assert_holdout_release`` took a mapping, and a mapping is something a
    caller constructs. The preregistration says the holdout opens on a *frozen*
    stage-1 pass, so the report has to be a file whose digest a checksum manifest
    already recorded — and the mapping being checked has to be that file's
    content, not a copy of it with better numbers.

    Three things, none of them a judgement call: the manifest verifies, it covers
    the report, and the report on disk parses to exactly what was passed in.
    """
    from tools import freeze_evidence

    binding = report.get("frozen_evidence") or {}
    manifest = binding.get("manifest")
    report_path = binding.get("report_path")
    if not manifest or not report_path:
        raise HoldoutError(
            "the stage-1 report declares no frozen_evidence {manifest, report_path}. "
            "The holdout opens on frozen stage-1 evidence, and a report that is not on "
            "disk behind a checksum is a number someone is holding, not a result."
        )
    tree = _root(root)
    manifest_file = tree / manifest
    if not manifest_file.is_file():
        raise HoldoutError(f"no frozen-evidence manifest at {manifest_file}")
    problems = freeze_evidence.check(manifest_file, root=tree)
    if problems:
        raise HoldoutError(
            f"the frozen-evidence manifest {manifest} does not verify: {problems[:4]}"
        )
    # `(sha256, path)`, in that order: unpacking it the other way round makes
    # `covered` a set of digests, and every report is then "not covered".
    covered = {path for _digest, path in freeze_evidence.manifest_entries(manifest_file)}
    if report_path not in covered:
        raise HoldoutError(
            f"{report_path} is not covered by {manifest}. The stage-1 numbers that open "
            "the holdout must be the frozen ones."
        )
    on_disk = json.loads((tree / report_path).read_text())
    if on_disk != dict(report):
        raise HoldoutError(
            f"the stage-1 report passed to the holdout gate is not what {report_path} "
            "holds. The frozen file is the evidence; anything else is a copy of it that "
            "someone has edited."
        )
    return {
        "manifest": manifest,
        "report_path": report_path,
        "report_sha256": hashlib.sha256((tree / report_path).read_bytes()).hexdigest(),
        "covered_files": len(covered),
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

    Six preconditions, and all six are checked without reading a holdout row:

    1. the ledger says the region is ``unspent``;
    2. the stage-1 report is **frozen evidence on disk**, behind a checksum
       manifest that still verifies and that covers it
       (:func:`assert_frozen_stage_one`);
    3. the report was decided by the *primary* cell — XGBoost, combined against
       control, at the base cost — so no secondary model and no cost multiplier
       is a second route here (:func:`assert_primary_decision`);
    4. the report was produced under *this* preregistration hash, so a pass
       computed under an edited rule cannot open it;
    5. the availability gate passed;
    6. the stage-1 continuation rule passes on that report.

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

    frozen = assert_frozen_stage_one(stage_one_report, root=root)
    decision = assert_primary_decision(stage_one_report)

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
        "frozen_evidence": frozen,
        "decided_by": decision,
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
