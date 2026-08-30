"""P7's decision, applied mechanically to the frozen mode evidence.

    python -m nn.p7_decision --runs artifacts/benchmark/btc_p7_* --out DIR

Reads the two mode artifacts, checks they are P7's and were produced under one
preregistration over one set of P6 specialists, confirms each mode's validity
gate passed, and applies the preregistered rule to each mode **separately**.

Two verdicts. There is no combined score and no best-mode row: §7.1 of the
preregistration lists "scalping supportive only" and "day trading supportive
only" as answers in their own right, and collapsing them into one would be the
selection the design forbids.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from nn.p6_preregistration import preregistration_hash as p6_hash
from nn.p7 import ARTIFACT_NAME
from nn.p7_preregistration import (
    CHECKPOINT,
    DECISION_RULE,
    EVIDENCE_CEILING,
    MODES,
    OUTCOME_INDEPENDENCE,
    QUESTION,
    STOPPING_RULE,
    VALIDITY_GATE,
    preregistration_hash,
)

logger = logging.getLogger(__name__)

DECISION_NAME = "decision.json"
STATUS_NAME = "STATUS.md"

EVIDENCE_CLASS = "the preregistered decision, applied to frozen mode evidence; the P7 outcome"

VERDICT_SUPPORTIVE = "supportive_adaptive"
VERDICT_NEGATIVE = "negative"
VERDICT_INVALID = "invalid"


class DecisionError(SystemExit):
    """The mode evidence cannot be decided on, and saying so beats deciding anyway."""


def load_modes(run_dirs: list[Path]) -> dict[str, dict[str, Any]]:
    """Both mode artifacts, keyed by mode, with neither missing."""
    found: dict[str, dict[str, Any]] = {}
    for directory in sorted(run_dirs):
        artifact = directory / ARTIFACT_NAME
        if not artifact.is_file():
            continue
        payload = json.loads(artifact.read_text())
        if payload.get("checkpoint") != CHECKPOINT:
            raise DecisionError(
                f"{directory} reports checkpoint {payload.get('checkpoint')!r}; this decides "
                f"{CHECKPOINT}"
            )
        payload["_dir"] = str(directory)
        found[payload["mode"]["mode"]] = payload

    expected = [item["mode"] for item in MODES]
    missing = [name for name in expected if name not in found]
    if missing:
        raise DecisionError(
            f"mode evidence is absent for {missing}. P7 reports both modes; deciding on one "
            "is the selection this design forbids."
        )
    return found


def check_modes_agree(modes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """One design, one set of specialists, and both validity gates passed."""
    hashes = {payload["preregistration_hash"] for payload in modes.values()}
    if hashes != {preregistration_hash()}:
        raise DecisionError(
            f"the mode evidence was produced under {sorted(hashes)} and this repository's "
            f"preregistration is {preregistration_hash()}"
        )
    p6 = {
        payload["specialist_source"]["p6_preregistration_hash"] for payload in modes.values()
    }
    if p6 != {p6_hash()}:
        raise DecisionError(
            f"the modes replayed specialists from {sorted(p6)} and P6's preregistration is "
            f"{p6_hash()}; they are not the specialists P6 published"
        )
    for name, payload in sorted(modes.items()):
        gate = payload.get("validity_gate") or {}
        if not gate.get("passed"):
            raise DecisionError(
                f"{name}: the validity gate did not pass, so its own decision-clock "
                "specialist is not being reproduced and no delta from it means anything"
            )
    return {
        "preregistration_hash": preregistration_hash(),
        "p6_preregistration_hash": p6_hash(),
        "validity_gate": VALIDITY_GATE["check"],
        "modes": sorted(modes),
    }


def verdict_for(payload: dict[str, Any]) -> dict[str, Any]:
    """The two preregistered conditions, evaluated and conjoined, for one mode."""
    folds = payload["folds"]
    deltas = [float(record["delta"]) for record in folds]
    improved = sum(1 for value in deltas if value > 0.0)
    mean = float(np.mean(deltas))
    required = int(DECISION_RULE["improved_folds_required"])

    conditions = {
        "improved_folds": {
            "required": required,
            "observed": improved,
            "passed": improved >= required,
        },
        "mean_fold_delta": {
            "required": "> 0",
            "observed": round(mean, 12),
            "passed": mean > 0.0,
        },
    }
    supportive = all(item["passed"] for item in conditions.values())
    return {
        "mode": payload["mode"]["mode"],
        "decision_clock": payload["mode"]["decision_clock"],
        "specialists": list(payload["consensus_rule"]["specialists"]),
        "agreement_required": payload["consensus_rule"]["agreement_required"],
        "veto_specialist": payload["consensus_rule"]["veto_specialist"],
        "folds": [
            {
                "fold": record["fold"],
                "period_start": record["period_start"],
                "period_end": record["period_end"],
                "consensus_net_return": record["consensus"]["net_return"],
                "best_constituent": record["best_constituent"]["clock"],
                "best_constituent_net_return": record["best_constituent"]["net_return"],
                "delta": record["delta"],
                "improved": record["delta"] > 0.0,
                "consensus_trades": record["consensus"]["n_trades"],
                "consensus_turnover": record["consensus"]["turnover"],
                "consensus_hold_rows": record["consensus"]["signal_counts"]["HOLD"],
                "decision_rows": record["decision_rows"],
            }
            for record in folds
        ],
        "conditions": conditions,
        "verdict": VERDICT_SUPPORTIVE if supportive else VERDICT_NEGATIVE,
        "descriptive": {
            "worst_fold_delta": round(min(deltas), 12),
            "best_fold_delta": round(max(deltas), 12),
            "note": "worst and best folds are reported for completeness and decide nothing",
        },
    }


def decide(modes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    verdicts = [verdict_for(modes[item["mode"]]) for item in MODES]
    supportive = [row["mode"] for row in verdicts if row["verdict"] == VERDICT_SUPPORTIVE]
    if len(supportive) == len(verdicts):
        outcome = "both modes supportive"
    elif supportive:
        outcome = f"{supportive[0].lower().replace('_', ' ')} supportive only"
    else:
        outcome = "neither supportive"
    return {
        "checkpoint": CHECKPOINT,
        "question": QUESTION,
        "evidence_class": EVIDENCE_CLASS,
        "evidence_ceiling": EVIDENCE_CEILING,
        "preregistration_hash": preregistration_hash(),
        "rule": dict(DECISION_RULE),
        "modes": verdicts,
        "supportive_modes": supportive,
        "outcome": outcome,
        "outcome_independence": OUTCOME_INDEPENDENCE,
        "interpretation": STOPPING_RULE["on_supportive" if supportive else "on_negative"],
    }


def to_markdown(payload: dict[str, Any]) -> str:
    lines = [
        # `# CURRENT` on the first line, because this file is also the
        # directory's STATUS marker and `artifacts/README.md`'s index is
        # checked against it. The heading below says which checkpoint.
        "# CURRENT",
        "",
        f"## {payload['checkpoint']} — decision",
        "",
        f"**{payload['evidence_class']}**",
        "",
        f"Preregistration `{payload['preregistration_hash']}`.",
        "",
        "| mode | decision clock | improved folds | mean delta | verdict |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in payload["modes"]:
        conditions = row["conditions"]
        lines.append(
            f"| `{row['mode']}` | `{row['decision_clock']}` | "
            f"{conditions['improved_folds']['observed']} of 4 | "
            f"{conditions['mean_fold_delta']['observed']} | **{row['verdict']}** |"
        )
    lines += ["", "Per-fold deltas against the fold-wise best constituent:", ""]
    for row in payload["modes"]:
        lines += [
            f"### `{row['mode']}` — {row['agreement_required']} of "
            f"{len(row['specialists'])}, `{row['veto_specialist']}` vetoes",
            "",
            "| fold | consensus | best constituent | delta | trades |",
            "| --- | --- | --- | --- | --- |",
        ]
        for record in row["folds"]:
            lines.append(
                f"| {record['fold']} | {record['consensus_net_return']} | "
                f"`{record['best_constituent']}` {record['best_constituent_net_return']} | "
                f"**{record['delta']}** | {record['consensus_trades']} |"
            )
        lines.append("")
    lines += [
        f"Supportive modes: {payload['supportive_modes'] or 'none'}. "
        f"Outcome: **{payload['outcome']}**.",
        "",
        payload["outcome_independence"],
        "",
        payload["interpretation"],
        "",
        f"Evidence ceiling: {payload['evidence_ceiling']}",
    ]
    return "\n".join(lines) + "\n"


def build(run_dirs: list[Path]) -> dict[str, Any]:
    modes = load_modes(run_dirs)
    identity = check_modes_agree(modes)
    payload = decide(modes)
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
    for row in payload["modes"]:
        logger.info("%-12s %s", row["mode"], row["verdict"])
    logger.info("P7 outcome: %s", payload["outcome"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
