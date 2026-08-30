"""P5's decision, applied mechanically to the frozen cells.

    python -m nn.p5_decision --runs artifacts/benchmark/btc_p5_* --out DIR

P2b, P2c and P3 left their "3 of 4" bar in a dictionary of English verdict
strings and in prose. Nothing raised on any verdict, and nothing wrote down what
the checkpoint had concluded — a reader had to count the folds themselves and
trust that the count they got was the one the rule meant. P4 fixed that for its
own two-stage screen (:mod:`nn.p4_stage1`) and this does the same for P5, in one
place, so that the answer is an artifact rather than a paragraph.

What it does, in order, and it refuses rather than continuing at every step:

1. reads the nine cells and checks they are P5's, under one preregistration hash
   and one ``mtf_v1`` spec hash;
2. recomputes the sample universe from the committed snapshot and checks its
   digest against the one every cell recorded — so the availability figures below
   describe the rows the models were actually fitted on;
3. evaluates the preregistered block-availability rule on each fold's inner and
   outer block, and the gate that requires all four folds;
4. applies the decision rule to exactly one cell pair —
   ``xgboost x ohlcv14_plus_mtf_v1`` against ``xgboost x ohlcv14`` — counting
   folds whose net-return delta is strictly positive;
5. writes the record, including the mean and the worst fold, both labelled
   descriptive, and the per-fold trade counts, labelled a flag that changes no
   denominator.

Nothing here can produce a different answer from the one the preregistration
specifies, because every number it compares against is read from
:mod:`nn.p5_preregistration` rather than written again.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nn.information_sets import P5_INFORMATION_SETS
from nn.mtf import MtfSpec, build_mtf_context
from nn.p2b import ARTIFACT_NAME, DEFAULT_MANIFEST, load_snapshot, plan_from_manifest
from nn.p5_preregistration import (
    AVAILABILITY_GATE,
    BLOCK_AVAILABILITY_RULE,
    COMBINED,
    CONTROL,
    DECISION_RULE,
    IMPROVED_RULE,
    MODELS,
    PRIMARY_COMPARISON,
    PRIMARY_MODEL,
    STOPPING_RULE,
    TRADE_COUNT_DIAGNOSTIC,
    preregistration_hash,
)

logger = logging.getLogger(__name__)

DECISION_NAME = "decision.json"
STATUS_NAME = "STATUS.md"

#: Not the literal `"derived"`, so `tools.freeze_evidence` will hash it. The
#: decision record is what the checkpoint answered, and P4 froze its screen for
#: the same reason.
EVIDENCE_CLASS = "the preregistered decision, applied to frozen cells; the P5 outcome"

OUTCOME_SUPPORTIVE = "supportive_adaptive"
OUTCOME_NEGATIVE = "negative"
OUTCOME_NOT_EVALUABLE = "not_evaluable"


class DecisionError(SystemExit):
    """The cells cannot be decided on, and saying so beats deciding anyway."""


def load_cells(run_dirs: list[Path]) -> dict[tuple[str, str], dict[str, Any]]:
    """Every P5 cell, keyed by ``(model, information_set)``, or a refusal."""
    cells: dict[tuple[str, str], dict[str, Any]] = {}
    for directory in run_dirs:
        path = Path(directory) / ARTIFACT_NAME
        if not path.is_file():
            raise DecisionError(f"{directory}: no {ARTIFACT_NAME}")
        payload = json.loads(path.read_text())
        if payload.get("checkpoint") != "P5":
            raise DecisionError(
                f"{directory} is a {payload.get('checkpoint')!r} cell. A decision built "
                "from another checkpoint's numbers would be about neither."
            )
        key = (payload["model"], payload["information_set"])
        if key in cells:
            raise DecisionError(f"two cells for {key}; exactly one may decide")
        payload["_dir"] = str(directory)
        cells[key] = payload
    return cells


def check_cells_agree(cells: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    """Nine cells, one design. Anything else is not one experiment."""
    expected = {(model, arm) for model in MODELS for arm in P5_INFORMATION_SETS}
    missing = sorted(expected - set(cells))
    extra = sorted(set(cells) - expected)
    if missing:
        raise DecisionError(f"missing cells: {missing}")
    if extra:
        raise DecisionError(f"unexpected cells: {extra}")

    def unique(getter, what: str) -> Any:
        seen = {json.dumps(getter(c), sort_keys=True, default=str) for c in cells.values()}
        if len(seen) != 1:
            raise DecisionError(
                f"the cells disagree about {what}, so they are not one experiment: "
                f"{sorted(seen)[:2]}"
            )
        return getter(next(iter(cells.values())))

    prereg = unique(lambda c: c["mtf_spec"]["preregistration_hash"], "the preregistration")
    if prereg != preregistration_hash():
        raise DecisionError(
            f"the cells were produced under preregistration {prereg} and this build "
            f"computes {preregistration_hash()}. Two preregistration hashes are two "
            "designs; they must not share a decision."
        )
    return {
        "preregistration_hash": prereg,
        "mtf_spec_hash": unique(lambda c: c["mtf_spec"]["spec_hash"], "the mtf_v1 spec"),
        "universe_sha256": unique(
            lambda c: c["sample_universe"]["universe_sha256"], "the universe"
        ),
        "target": unique(lambda c: c["target"], "the target"),
        "contract_hash": unique(lambda c: c["contract"]["contract_hash"], "the contract"),
        "code_revision": unique(lambda c: c["code"], "the code revision"),
        "folds": unique(lambda c: c["config"]["folds"], "the fold count"),
        "seed": unique(lambda c: c["config"]["seed"], "the seed"),
    }


def recompute_availability(manifest_path: Path, universe_sha256: str) -> dict[str, Any]:
    """The block-availability rule, on the universe the cells were fitted on.

    Recomputed rather than read out of a cell: a figure the cells asserted about
    themselves would be a claim, and this is a check. The digest binds the two.
    """
    spine, _, raw, manifest = load_snapshot(manifest_path)
    raw = raw.reset_index(drop=True).copy()
    raw["date"] = pd.to_datetime(raw["date"], utc=True)
    dates = pd.to_datetime(spine["date"], utc=True)
    row_of = pd.Series(np.arange(len(raw), dtype=np.int64), index=raw["date"].to_numpy())
    rows = row_of.reindex(dates.to_numpy()).to_numpy(dtype=np.int64)
    context = build_mtf_context(
        raw.iloc[: int(rows[-1]) + 1].reset_index(drop=True), MtfSpec()
    )
    eligible = np.asarray(context.eligible, dtype=bool)[rows]

    from nn.information_sets import _universe_hash

    digest = _universe_hash(eligible)
    if digest != universe_sha256:
        raise DecisionError(
            f"the recomputed sample universe digests to {digest} and the cells recorded "
            f"{universe_sha256}. The availability below would describe different rows "
            "from the ones the models were fitted on."
        )

    folds, _ = plan_from_manifest(manifest, len(spine))
    minimum = float(BLOCK_AVAILABILITY_RULE["min_eligible_row_fraction"])
    max_run = int(BLOCK_AVAILABILITY_RULE["max_contiguous_ineligible_hours"])

    def longest_false_run(mask: np.ndarray) -> int:
        longest = run = 0
        for value in mask:
            run = 0 if value else run + 1
            longest = max(longest, run)
        return longest

    blocks: list[dict[str, Any]] = []
    for index, plan in enumerate(folds):
        record: dict[str, Any] = {"fold": index, "available": True, "reasons": []}
        for label, split in (
            ("inner_validation", plan.inner),
            ("outer_validation", plan.outer),
        ):
            mask = eligible[split.start : split.end]
            fraction = float(mask.mean())
            outage = longest_false_run(mask)
            record[label] = {
                "rows": int(len(mask)),
                "rows_eligible": int(mask.sum()),
                "eligible_fraction": round(fraction, 6),
                "max_contiguous_ineligible_hours": outage,
            }
            if fraction < minimum:
                record["available"] = False
                record["reasons"].append(
                    f"{label}: {mask.sum()}/{len(mask)} rows eligible ({fraction:.4f}); "
                    f"at least {minimum} is required"
                )
            if outage > max_run:
                record["available"] = False
                record["reasons"].append(
                    f"{label}: the longest contiguous run of ineligible rows is {outage}h; "
                    f"at most {max_run}h is permitted"
                )
        train = eligible[plan.train.start : plan.train.end]
        record["train"] = {
            "rows": int(len(train)),
            "rows_eligible": int(train.sum()),
            "eligible_fraction": round(float(train.mean()), 6),
            "gating": False,
        }
        blocks.append(record)

    available = [b for b in blocks if b["available"]]
    return {
        "rule": dict(BLOCK_AVAILABILITY_RULE),
        "gate": dict(AVAILABILITY_GATE),
        "universe_sha256": digest,
        "rows": int(len(eligible)),
        "rows_eligible": int(eligible.sum()),
        "eligible_fraction": round(float(eligible.mean()), 6),
        "blocks": blocks,
        "folds_available": len(available),
        "gate_passed": len(available) >= int(AVAILABILITY_GATE["folds_required"]),
    }


def _fold_returns(cell: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Per-fold outer-validation trading figures, read the way the cell writes them.

    ``outer_validation`` is keyed by model name — the cell reports its own model
    alongside the majority and momentum baselines and the economic references —
    so the model is looked up rather than assumed, and a cell whose block does not
    carry its own model is refused rather than silently read from a baseline.
    """
    model = cell["model"]
    out: dict[int, dict[str, Any]] = {}
    for record in cell["folds"]:
        block = record["outer_validation"]
        if model not in block:
            raise DecisionError(
                f"{cell['_dir']} fold {record['fold']}: the outer block reports "
                f"{sorted(block)} and not {model!r}"
            )
        trading = block[model]["trading"]
        periods = record["periods"]["outer_validation"]
        out[int(record["fold"])] = {
            "net_return": float(trading["net_return"]),
            "n_trades": int(trading["n_trades"]),
            "period_start": str(periods["start"]),
            "period_end": str(periods["end"]),
        }
    return out


def decide(
    cells: dict[tuple[str, str], dict[str, Any]], availability: dict[str, Any]
) -> dict[str, Any]:
    """The preregistered rule, applied to exactly one cell pair."""
    combined_arm, control_arm = PRIMARY_COMPARISON
    combined = _fold_returns(cells[(PRIMARY_MODEL, combined_arm)])
    control = _fold_returns(cells[(PRIMARY_MODEL, control_arm)])
    if sorted(combined) != sorted(control):
        raise DecisionError("the deciding cells report different folds")

    floor = int(TRADE_COUNT_DIAGNOSTIC["flag_below_outer_trades"])
    per_fold = []
    for fold in sorted(combined):
        delta = round(combined[fold]["net_return"] - control[fold]["net_return"], 6)
        per_fold.append(
            {
                "fold": fold,
                "period_start": combined[fold]["period_start"],
                "period_end": combined[fold]["period_end"],
                "control_net_return": control[fold]["net_return"],
                "combined_net_return": combined[fold]["net_return"],
                "delta": delta,
                "improved": delta > 0.0,
                "control_trades": control[fold]["n_trades"],
                "combined_trades": combined[fold]["n_trades"],
                "thin_trades_flag": min(control[fold]["n_trades"], combined[fold]["n_trades"])
                < floor,
            }
        )

    improved = sum(1 for row in per_fold if row["improved"])
    required = int(DECISION_RULE["improved_folds_required"])
    deltas = [row["delta"] for row in per_fold]

    if not availability["gate_passed"]:
        outcome = OUTCOME_NOT_EVALUABLE
        passed = False
    else:
        passed = improved >= required
        outcome = OUTCOME_SUPPORTIVE if passed else OUTCOME_NEGATIVE

    return {
        "decided_by": {"model": PRIMARY_MODEL, "comparison": list(PRIMARY_COMPARISON)},
        "rule": dict(DECISION_RULE),
        "improved_rule": dict(IMPROVED_RULE),
        "folds": per_fold,
        "improved_folds": improved,
        "total_folds": len(per_fold),
        "required_folds": required,
        "passed": passed,
        "outcome": outcome,
        "descriptive": {
            "mean_delta": round(float(np.mean(deltas)), 12) if deltas else None,
            "worst_fold_delta": round(min(deltas), 12) if deltas else None,
            "best_fold_delta": round(max(deltas), 12) if deltas else None,
            "note": (
                "reported for completeness and decisive in neither direction: the mean may "
                "not rescue a fold-count failure and may not veto a fold-count pass"
            ),
        },
        "trade_count_diagnostic": {
            **dict(TRADE_COUNT_DIAGNOSTIC),
            "folds_flagged": [row["fold"] for row in per_fold if row["thin_trades_flag"]],
        },
        "interpretation": STOPPING_RULE[
            {
                OUTCOME_SUPPORTIVE: "on_pass",
                OUTCOME_NEGATIVE: "on_fail",
                OUTCOME_NOT_EVALUABLE: "on_not_evaluable",
            }[outcome]
        ],
    }


def secondary_context(cells: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    """Every non-deciding comparison, reported in full and deciding nothing."""
    rows = []
    for model in MODELS:
        control = _fold_returns(cells[(model, CONTROL)])
        for arm in P5_INFORMATION_SETS:
            if arm == CONTROL:
                continue
            arm_folds = _fold_returns(cells[(model, arm)])
            deltas = [
                round(arm_folds[f]["net_return"] - control[f]["net_return"], 6)
                for f in sorted(control)
            ]
            rows.append(
                {
                    "model": model,
                    "information_set": arm,
                    "deltas": deltas,
                    "improved_folds": sum(1 for d in deltas if d > 0.0),
                    "total_folds": len(deltas),
                    "mean_delta": round(float(np.mean(deltas)), 12),
                    "decides": model == PRIMARY_MODEL and arm == COMBINED,
                    "role": (
                        "primary"
                        if model == PRIMARY_MODEL and arm == COMBINED
                        else "secondary (descriptive; cannot switch the deciding cell)"
                    ),
                }
            )
    return rows


def to_markdown(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    availability = payload["availability"]
    status = {
        OUTCOME_SUPPORTIVE: "CURRENT",
        OUTCOME_NEGATIVE: "CURRENT",
        OUTCOME_NOT_EVALUABLE: "CURRENT",
    }[decision["outcome"]]
    lines = [
        f"# {status}",
        "",
        "## P5 — the preregistered decision",
        "",
        f"**Outcome: `{decision['outcome']}`.** "
        f"{decision['improved_folds']} of {decision['total_folds']} folds improved, "
        f"against a bar of {decision['required_folds']}.",
        "",
        "Deciding cell: `"
        + decision["decided_by"]["model"]
        + " x "
        + decision["decided_by"]["comparison"][0]
        + "` against `"
        + decision["decided_by"]["model"]
        + " x "
        + decision["decided_by"]["comparison"][1]
        + "`, outer-validation cost-aware net return at cost multiplier 1.0.",
        "",
        f"Preregistration: `{payload['identity']['preregistration_hash']}`.",
        "",
        "| fold | period | control | combined | delta | improved | trades |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in decision["folds"]:
        flag = " ⚑" if row["thin_trades_flag"] else ""
        lines.append(
            f"| {row['fold']} | `{row['period_start']}` .. `{row['period_end']}` | "
            f"`{row['control_net_return']}` | `{row['combined_net_return']}` | "
            f"`{row['delta']}` | {'yes' if row['improved'] else 'no'} | "
            f"{row['control_trades']}/{row['combined_trades']}{flag} |"
        )
    lines += [
        "",
        f"Mean delta `{decision['descriptive']['mean_delta']}`, worst fold "
        f"`{decision['descriptive']['worst_fold_delta']}`. **Both descriptive.** "
        "The preregistration makes them decisive in neither direction: they may not "
        "rescue a fold-count failure and may not veto a fold-count pass.",
        "",
        "### Availability",
        "",
        f"{availability['rows_eligible']} of {availability['rows']} rows eligible "
        f"({availability['eligible_fraction']}); "
        f"**{availability['folds_available']} of {len(availability['blocks'])} folds "
        f"available**, gate "
        f"{'passed' if availability['gate_passed'] else 'FAILED'}.",
        "",
        "Recomputed here from the committed snapshot and checked against the universe "
        "digest every cell recorded, so these figures describe the rows the models were "
        "actually fitted on rather than a claim the cells made about themselves.",
        "",
        "### What this is, and is not",
        "",
        decision["interpretation"],
        "",
        "### Context, deciding nothing",
        "",
        "| model | arm | improved | deltas | mean | role |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["context"]:
        lines.append(
            f"| `{row['model']}` | `{row['information_set']}` | "
            f"{row['improved_folds']} of {row['total_folds']} | `{row['deltas']}` | "
            f"`{row['mean_delta']}` | {row['role']} |"
        )
    lines.append("")
    return "\n".join(lines)


def build(run_dirs: list[Path], manifest_path: Path) -> dict[str, Any]:
    cells = load_cells(run_dirs)
    identity = check_cells_agree(cells)
    availability = recompute_availability(manifest_path, identity["universe_sha256"])
    decision = decide(cells, availability)
    return {
        "checkpoint": "P5",
        "evidence_class": EVIDENCE_CLASS,
        "identity": identity,
        "cells": sorted(cell["_dir"] for cell in cells.values()),
        "availability": availability,
        "decision": decision,
        "context": secondary_context(cells),
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = build_argparser().parse_args(argv)
    payload = build(list(args.runs), args.manifest)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / DECISION_NAME).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (out / STATUS_NAME).write_text(to_markdown(payload))
    decision = payload["decision"]
    logger.warning(
        "P5 %s: %d of %d folds improved against a bar of %d. Mean delta %s, worst fold %s "
        "(both descriptive).",
        decision["outcome"],
        decision["improved_folds"],
        decision["total_folds"],
        decision["required_folds"],
        decision["descriptive"]["mean_delta"],
        decision["descriptive"]["worst_fold_delta"],
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
