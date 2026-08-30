"""P6's gate, applied mechanically to the frozen cells.

    python -m nn.p6_decision --runs artifacts/benchmark/btc_p6_* --out DIR

P2b, P2c and P3 left their bar in a dictionary of English verdict strings and in
prose. P4 fixed that for its own screen and P5 for its own comparison; this does
the same for P6, in one place, so that the answer is an artifact rather than a
paragraph.

What it does, in order, refusing rather than continuing at every step:

1. reads the fifteen cells and checks they are P6's, under one preregistration
   hash and one 1m source digest;
2. checks every clock reports the same four outer periods as the preregistration
   froze — the property that makes five clocks one experiment;
3. applies the gate to the **XGBoost** cell of each clock, evaluating all three
   preregistered conditions and requiring their conjunction;
4. reports the other two families in full, deciding nothing;
5. writes five verdicts and no summary row.

Nothing here can produce a different answer from the one the preregistration
specifies, because every number it compares against is read from
:mod:`nn.p6_preregistration` rather than written again.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nn.p6 import ARTIFACT_NAME
from nn.p6_preregistration import (
    CHECKPOINT,
    CLOCKS,
    EVIDENCE_CEILING,
    FOLD_PERIODS,
    HORIZONS,
    MODELS,
    PRIMARY_MODEL,
    QUESTION,
    STOPPING_RULE,
    VIABILITY_GATE,
    preregistration_hash,
)

logger = logging.getLogger(__name__)

DECISION_NAME = "decision.json"
STATUS_NAME = "STATUS.md"

#: Not the literal "derived", so `tools.freeze_evidence` will hash it. The
#: decision record is what the checkpoint answered.
EVIDENCE_CLASS = "the preregistered gate, applied to frozen cells; the P6 outcome"

OUTCOME_SUPPORTIVE = "supportive_adaptive"
OUTCOME_NEGATIVE = "negative"

VERDICT_VIABLE = "viable_on_adaptive_research_folds"
VERDICT_NOT_VIABLE = "not_viable"


class DecisionError(SystemExit):
    """The cells cannot be decided on, and saying so beats deciding anyway."""


def load_cells(run_dirs: list[Path]) -> dict[tuple[str, str], dict[str, Any]]:
    """Every P6 cell, keyed by (clock, model), with its directory recorded."""
    cells: dict[tuple[str, str], dict[str, Any]] = {}
    for directory in sorted(run_dirs):
        artifact = directory / ARTIFACT_NAME
        if not artifact.is_file():
            continue
        payload = json.loads(artifact.read_text())
        if payload.get("checkpoint") != CHECKPOINT:
            raise DecisionError(
                f"{directory} reports checkpoint {payload.get('checkpoint')!r}; this "
                f"decides {CHECKPOINT} and refuses to read another checkpoint's cells"
            )
        payload["_dir"] = str(directory)
        cells[(payload["clock"], payload["model"])] = payload

    missing = [
        (clock, model) for clock in CLOCKS for model in MODELS if (clock, model) not in cells
    ]
    if missing:
        raise DecisionError(
            f"{len(missing)} of {len(CLOCKS) * len(MODELS)} cells are absent: {missing}. "
            "P6 reports every clock; deciding on a subset is the winner-shopping this "
            "design forbids."
        )
    return cells


def check_cells_agree(cells: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    """One design, one source, one set of periods — checked, not assumed."""
    hashes = {cell["preregistration_hash"] for cell in cells.values()}
    if hashes != {preregistration_hash()}:
        raise DecisionError(
            f"the cells were produced under {sorted(hashes)} and this repository's "
            f"preregistration is {preregistration_hash()}. A cell fitted under an edited "
            "design is a different object and may not be decided with these."
        )
    minutes = {cell["source"]["minutes_digest"] for cell in cells.values()}
    if len(minutes) != 1:
        raise DecisionError(f"the cells read {len(minutes)} different 1m sources: {minutes}")

    frozen = [(row["outer_start"], row["outer_end"]) for row in FOLD_PERIODS]
    for (clock, model), cell in sorted(cells.items()):
        periods = [
            (
                record["periods"]["outer_validation"]["start"],
                record["periods"]["outer_validation"]["end"],
            )
            for record in cell["folds"]
        ]
        if len(periods) != len(frozen):
            raise DecisionError(
                f"{clock} x {model} reports {len(periods)} folds and P6 froze {len(frozen)}"
            )
        # The cell records the first and last bar *open* of its outer block. A
        # 1m block's last open is 59 minutes short of a 1h block's, so the check
        # is containment inside the frozen half-open window rather than equality
        # — the periods are the same real-world window on every clock, and only
        # the granularity of the last bar differs.
        #
        # Parsed, never compared as text. The cells render timestamps the way
        # pandas does ("2023-03-04 07:00:00+00:00") and the frozen periods are
        # ISO-8601 ("2023-03-04T07:00:00+00:00"); a string comparison across
        # those two spellings orders ' ' before 'T' and silently rejects a
        # correct block.
        for position, (start, end) in enumerate(periods):
            low, high = (pd.Timestamp(value) for value in frozen[position])
            first, last = pd.Timestamp(start), pd.Timestamp(end)
            if not (low <= first and last < high):
                raise DecisionError(
                    f"{clock} x {model} fold {position} reports outer {start} .. {end}, "
                    f"which is not inside the frozen period {low} .. {high}"
                )
    return {
        "preregistration_hash": preregistration_hash(),
        "minutes_digest": minutes.pop(),
        "cells": len(cells),
        "clock_digests": {
            clock: cells[(clock, PRIMARY_MODEL)]["source"]["clock_digest"] for clock in CLOCKS
        },
    }


def _fold_rows(cell: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-fold outer figures for a cell's own model and its momentum floor.

    The model is looked up by name rather than assumed, and a cell whose outer
    block does not carry its own model is refused rather than silently read from
    a baseline.
    """
    model = cell["model"]
    rows = []
    for record in cell["folds"]:
        block = record["outer_validation"]
        for name in (model, "momentum_baseline"):
            if name not in block:
                raise DecisionError(
                    f"{cell['_dir']} fold {record['fold']}: the outer block reports "
                    f"{sorted(block)} and not {name!r}"
                )
        model_net = float(block[model]["trading"]["net_return"])
        momentum_net = float(block["momentum_baseline"]["trading"]["net_return"])
        periods = record["periods"]["outer_validation"]
        rows.append(
            {
                "fold": int(record["fold"]),
                "period_start": str(periods["start"]),
                "period_end": str(periods["end"]),
                "net_return": model_net,
                "momentum_net_return": momentum_net,
                "beats_momentum": model_net > momentum_net,
                "positive": model_net > 0.0,
                "n_trades": int(block[model]["trading"]["n_trades"]),
                "turnover": float(block[model]["trading"]["turnover"]),
                "threshold": record["model"]["selection"]["threshold"],
                "buy_and_hold": float(
                    block["economic_references"]["buy_and_hold"]["net_return"]
                ),
            }
        )
    return rows


def verdict_for(cell: dict[str, Any]) -> dict[str, Any]:
    """The three preregistered conditions, evaluated and conjoined."""
    rows = _fold_rows(cell)
    returns = [row["net_return"] for row in rows]
    positive = sum(1 for row in rows if row["positive"])
    beats = sum(1 for row in rows if row["beats_momentum"])
    mean = float(np.mean(returns))

    required_positive = int(VIABILITY_GATE["positive_folds_required"])
    required_beats = int(VIABILITY_GATE["beats_momentum_folds_required"])
    conditions = {
        "positive_folds": {
            "required": required_positive,
            "observed": positive,
            "passed": positive >= required_positive,
        },
        "mean_outer_net_return": {
            "required": "> 0",
            "observed": round(mean, 12),
            "passed": mean > 0.0,
        },
        "beats_native_momentum_folds": {
            "required": required_beats,
            "observed": beats,
            "passed": beats >= required_beats,
        },
    }
    viable = all(item["passed"] for item in conditions.values())
    return {
        "clock": cell["clock"],
        "model": cell["model"],
        "horizon": HORIZONS[cell["clock"]],
        "folds": rows,
        "conditions": conditions,
        "viable": viable,
        "verdict": VERDICT_VIABLE if viable else VERDICT_NOT_VIABLE,
        "descriptive": {
            "worst_fold": round(min(returns), 12),
            "best_fold": round(max(returns), 12),
            "total_outer_trades": sum(row["n_trades"] for row in rows),
            "note": (
                "worst and best folds are reported for completeness and decide nothing; "
                "the mean is one of the three gate conditions and does decide"
            ),
        },
    }


def secondary_context(cells: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    """The two non-deciding families, reported in full and deciding nothing."""
    rows = []
    for clock in CLOCKS:
        for model in MODELS:
            if model == PRIMARY_MODEL:
                continue
            verdict = verdict_for(cells[(clock, model)])
            rows.append(
                {
                    "clock": clock,
                    "model": model,
                    "would_have_passed": verdict["viable"],
                    "positive_folds": verdict["conditions"]["positive_folds"]["observed"],
                    "mean_outer_net_return": verdict["conditions"]["mean_outer_net_return"][
                        "observed"
                    ],
                    "beats_momentum_folds": verdict["conditions"][
                        "beats_native_momentum_folds"
                    ]["observed"],
                }
            )
    return rows


def decide(cells: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    """Five verdicts, one per clock, and no summary row."""
    verdicts = [verdict_for(cells[(clock, PRIMARY_MODEL)]) for clock in CLOCKS]
    viable = [row["clock"] for row in verdicts if row["viable"]]
    outcome = OUTCOME_SUPPORTIVE if viable else OUTCOME_NEGATIVE
    return {
        "checkpoint": CHECKPOINT,
        "question": QUESTION,
        "evidence_class": EVIDENCE_CLASS,
        "evidence_ceiling": EVIDENCE_CEILING,
        "preregistration_hash": preregistration_hash(),
        "decided_by": PRIMARY_MODEL,
        "gate": dict(VIABILITY_GATE),
        "clocks": verdicts,
        "viable_clocks": viable,
        "outcome": outcome,
        "answer_is": (
            "the five per-clock verdicts above. `outcome` says only whether any clock "
            "cleared the absolute gate; it is not a checkpoint-level score and there is "
            "deliberately no best-clock row."
        ),
        "secondary_context": secondary_context(cells),
        "secondary_context_note": (
            "reported for every clock and decisive for none. A clock whose verdict is "
            "not viable stays not viable if another family would have passed."
        ),
        "interpretation": STOPPING_RULE["on_pass" if viable else "on_all_fail"],
    }


def to_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# {payload['checkpoint']} — decision",
        "",
        f"**{payload['evidence_class']}**",
        "",
        f"Preregistration `{payload['preregistration_hash']}`, decided by "
        f"`{payload['decided_by']}`.",
        "",
        "| clock | horizon | positive folds | mean net return | beats momentum | verdict |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["clocks"]:
        conditions = row["conditions"]
        lines.append(
            f"| `{row['clock']}` | {row['horizon']} | "
            f"{conditions['positive_folds']['observed']} of 4 | "
            f"{conditions['mean_outer_net_return']['observed']} | "
            f"{conditions['beats_native_momentum_folds']['observed']} of 4 | "
            f"**{row['verdict']}** |"
        )
    lines += [
        "",
        f"Viable clocks: {payload['viable_clocks'] or 'none'}.",
        "",
        payload["answer_is"],
        "",
        payload["interpretation"],
        "",
        f"Evidence ceiling: {payload['evidence_ceiling']}",
    ]
    return "\n".join(lines) + "\n"


def build(run_dirs: list[Path]) -> dict[str, Any]:
    cells = load_cells(run_dirs)
    identity = check_cells_agree(cells)
    payload = decide(cells)
    payload["identity"] = identity
    return payload


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_argparser().parse_args(argv)
    payload = build(list(args.runs))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / DECISION_NAME).write_text(json.dumps(payload, indent=2) + "\n")
    (args.out / STATUS_NAME).write_text(to_markdown(payload))
    for row in payload["clocks"]:
        logger.info("%4s: %s", row["clock"], row["verdict"])
    logger.info("P6 outcome: %s", payload["outcome"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
