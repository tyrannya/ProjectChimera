"""P2b: read the nine cells, recompute them, and compare information sets.

    python -m nn.p2b_compare --runs artifacts/benchmark/btc_p2b_* \
        --out artifacts/benchmark/btc_p2b_comparison

Each :mod:`nn.p2b` run covers one information set and one model. This joins
them into the answer P2b was asked for, and does three things on the way that a
plain aggregator would not:

*it refuses to aggregate cells that did not score the same rows*
    Every cell records its own baselines, its own economic references and a
    hash of its sample indices per fold. Those are properties of the *rows*, not
    of the model, so on identical data they are one value across all nine cells.
    Checking them is a direct, data-level proof of the parity claim — the claim
    survives being split across nine processes precisely because nothing but the
    data could make these agree.

*it recomputes, from the persisted predictions, every number that can be*
    ``outer_predictions.parquet`` holds the probability, the selected action,
    the threshold and the realised future return of every scored sample. The
    trading and classification metrics are recomputed from those columns through
    :mod:`nn.evaluate` and checked against what the cell reported. A report that
    disagrees with its own predictions is a report about nothing, and there is
    no way to notice that by reading it.

*it counts temporal periods, not runs*
    The statistical unit is one of four outer blocks. These estimators are
    deterministic given their inputs, so repeating a cell under another seed
    would copy the evidence rather than add to it, and "15 of 20 wins" from five
    seeds over four periods would be a four-period result wearing a larger
    number. Only ``n of 4`` appears anywhere in the output.

The verdict thresholds are stated before the numbers are read: improvement in
3 or 4 of 4 folds is evidence worth continuing on, 2 of 4 is regime-dependent
and inconclusive, 1 or 0 is weak-to-negative. They are predeclared so that the
result cannot pick its own bar.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from chimera.contracts import TargetSpec
from nn import evaluate as ev
from nn.data_pipeline import load_dataset
from nn.information_sets import CHECKPOINTS, Checkpoint, OHLCV14
from nn.p2b import ARTIFACT_NAME, PREDICTIONS_NAME
from nn.regime import direction_attribution
from nn.simple_models import SIMPLE_MODEL_NAMES
from nn.walkforward import REFERENCES_KEY
from tools.freeze_evidence import DERIVED

logger = logging.getLogger(__name__)

COMPARISON_JSON = "p2b_comparison.json"
COMPARISON_MD = "p2b_comparison.md"

#: The control every delta is taken against.
CONTROL = OHLCV14

#: Reported per fold, per cell. ``(section, key)`` into an outer-validation report.
FOLD_METRICS: dict[str, tuple[str, str]] = {
    "n_trades": ("trading", "n_trades"),
    "exposure": ("trading", "exposure"),
    "turnover": ("trading", "turnover"),
    "gross_return": ("trading", "gross_return"),
    "total_costs": ("trading", "total_costs"),
    "net_return": ("trading", "net_return"),
    "max_drawdown": ("trading", "max_drawdown"),
    "annualised_sharpe": ("trading", "annualised_sharpe"),
    "per_trade_sharpe": ("trading", "per_trade_sharpe"),
    "win_rate": ("trading", "win_rate"),
    "profit_factor": ("trading", "profit_factor"),
    "macro_f1": ("classification", "macro_f1"),
    "directional_accuracy": ("classification", "directional_accuracy"),
    "accuracy": ("classification", "accuracy"),
    "coverage": ("classification", "coverage"),
}

#: The deltas P2b is actually about, in report order.
DELTA_METRICS = (
    "net_return",
    "annualised_sharpe",
    "max_drawdown",
    "exposure",
    "n_trades",
    "macro_f1",
    "directional_accuracy",
)

VERDICTS = {
    4: "consistent improvement across all four temporal folds — evidence worth continuing on",
    3: "improvement in three of four temporal folds — evidence worth continuing on",
    2: "improvement in two of four temporal folds — regime-dependent, inconclusive",
    1: "improvement in one of four temporal folds — weak evidence against",
    0: "no improvement in any temporal fold — negative evidence",
}


class ComparisonError(SystemExit):
    """The cells cannot be compared, and saying so beats averaging them anyway."""


def load_cell(run_dir: Path) -> dict[str, Any]:
    """One cell: its artifact, its predictions, and where they came from."""
    artifact_path = run_dir / ARTIFACT_NAME
    if not artifact_path.is_file():
        raise ComparisonError(f"{run_dir} has no {ARTIFACT_NAME}; it is not a P2b cell")
    payload = json.loads(artifact_path.read_text())
    # Which checkpoint this is comes from the cell, and the cells must then agree
    # — see `checkpoint_of`. What is refused here is a cell that names no
    # checkpoint this runner knows how to ask about, because there is nothing
    # for the others to agree *with*.
    named = payload.get("checkpoint")
    if named not in CHECKPOINTS:
        raise ComparisonError(
            f"{artifact_path} is checkpoint {named!r}; this joins cells of "
            f"{sorted(CHECKPOINTS)}"
        )
    declared = payload.get("outer_predictions")
    if declared != PREDICTIONS_NAME:
        raise ComparisonError(
            f"{artifact_path} declares its predictions as {declared!r}; refusing to pair "
            "it with a file it does not name"
        )
    predictions = pd.read_parquet(run_dir / PREDICTIONS_NAME)
    return {
        "dir": run_dir,
        "information_set": payload["information_set"],
        "model": payload["model"],
        "payload": payload,
        "predictions": predictions,
    }


def checkpoint_of(cells: Sequence[dict[str, Any]]) -> Checkpoint:
    """The one research question these cells answer, or a refusal.

    A comparison is an answer to a question, and cells that asked different
    questions have no common answer to give. Before this existed the checkpoint
    was a constant in :mod:`nn.p2b`, so both checkpoints wrote ``"P2b"`` and
    every cell trivially agreed: nine P2c artifacts and their comparison all
    identified as P2b market-structure research, and a glob wide enough to catch
    both checkpoints' cells would have averaged twelve arms of two different
    families into one table without a word.

    Three things are required, not one. The cells must name the same checkpoint;
    that name must be a checkpoint this repository declares; and every arm
    present must belong to it — so a P2b cell relabelled ``"P2c"`` by hand is
    still refused, because ``smc_v1`` is not one of P2c's arms.
    """
    named = sorted({c["payload"]["checkpoint"] for c in cells})
    if len(named) != 1:
        by_checkpoint = {
            name: sorted(
                f"{c['information_set']} x {c['model']}"
                for c in cells
                if c["payload"]["checkpoint"] == name
            )
            for name in named
        }
        raise ComparisonError(
            "these cells answer different research questions and cannot be joined into "
            f"one comparison: {json.dumps(by_checkpoint, indent=2)}"
        )
    checkpoint = CHECKPOINTS[named[0]]

    foreign = sorted(
        {c["information_set"] for c in cells if not checkpoint.accepts(c["information_set"])}
    )
    if foreign:
        raise ComparisonError(
            f"{foreign} are not arms of {checkpoint.name}, whose arms are "
            f"{list(checkpoint.arms)}. A cell that carries the right checkpoint label and "
            "the wrong columns is the one corruption a label check alone cannot see."
        )
    if not any(c["information_set"] == checkpoint.control for c in cells):
        raise ComparisonError(
            f"no {checkpoint.control!r} cell present; every delta below is taken against "
            "the control and there is nothing to take one against"
        )
    stated = sorted({c["payload"].get("question") for c in cells})
    if stated != [checkpoint.question]:
        raise ComparisonError(
            f"{checkpoint.name} asks {checkpoint.question!r} but its cells state "
            f"{stated!r}. The question a cell prints and the checkpoint it claims must "
            "be the same statement, or one of the two is decoration."
        )
    return checkpoint


def check_cells_agree(cells: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Prove the cells scored the same rows before any of them is aggregated.

    Everything checked here is a function of the data and the fold geometry
    alone. A model cannot move any of it, so a disagreement means two cells were
    not looking at the same rows — and every delta below would then be measuring
    the difference between two sample universes rather than between two
    information sets.
    """
    reference = cells[0]
    ref = reference["payload"]

    def described(cell: dict[str, Any]) -> str:
        return f"{cell['information_set']} x {cell['model']}"

    for cell in cells[1:]:
        payload = cell["payload"]
        mine = (payload.get("code") or {}).get("source_digest")
        theirs = (ref.get("code") or {}).get("source_digest")
        if mine != theirs:
            raise ComparisonError(
                f"{described(cell)} ran source digest {mine} and "
                f"{described(reference)} ran {theirs}. Every other identity a cell "
                "records describes the data or the definitions, so two revisions of "
                "the runner would otherwise join without a word. The digest covers "
                "every repository module the process imported, so a documentation "
                "commit does not split a batch and an uncommitted edit does not hide."
            )
        for key in ("contract", "sizes", "target", "threshold_selection", "snapshot"):
            if payload[key] != ref[key]:
                raise ComparisonError(
                    f"{described(cell)} and {described(reference)} disagree on {key!r}"
                )
        if (
            payload["feature_spec"]["combined_spec_hash"]
            != ref["feature_spec"]["combined_spec_hash"]
        ):
            raise ComparisonError(f"{described(cell)} used a different feature spec")
        # Length first: `zip` truncates, so two cells with a different number of
        # folds would compare only the folds they share and pass.
        if len(payload["alignment"]["folds"]) != len(ref["alignment"]["folds"]) or len(
            payload["folds"]
        ) != len(ref["folds"]):
            raise ComparisonError(
                f"{described(cell)} and {described(reference)} report a different "
                "number of folds"
            )
        for fold, (a, b) in enumerate(
            zip(payload["alignment"]["folds"], ref["alignment"]["folds"])
        ):
            if a != b:
                raise ComparisonError(
                    f"{described(cell)} fold {fold} scored a different sample index than "
                    f"{described(reference)}"
                )
        for fold, (a, b) in enumerate(zip(payload["folds"], ref["folds"])):
            if a["samples"] != b["samples"] or a["periods"] != b["periods"]:
                raise ComparisonError(
                    f"{described(cell)} fold {fold} has different samples or periods"
                )
            for name in ("majority_baseline", "momentum_baseline", REFERENCES_KEY):
                if a["outer_validation"][name] != b["outer_validation"][name]:
                    raise ComparisonError(
                        f"{described(cell)} fold {fold} reports a different {name} than "
                        f"{described(reference)}, so the two did not score the same rows"
                    )
    return {
        "cells": [described(cell) for cell in cells],
        "identical_across_cells": [
            "research contract and its hash",
            "snapshot identity and semantic hashes",
            "fold sizes and periods",
            "the research checkpoint each cell says it answers",
            "per-fold sample-index hashes from the alignment proof",
            "label horizon and costs",
            "threshold grid, objective and trade floor",
            "combined feature-spec hash",
            "majority and momentum baseline outer reports",
            "CASH and buy-and-hold economic references",
        ],
        "code": {
            "source_digest": (cells[0]["payload"].get("code") or {}).get("source_digest"),
            "revisions": sorted(
                {
                    (c["payload"].get("code") or {}).get("revision")
                    for c in cells
                    if (c["payload"].get("code") or {}).get("revision")
                }
            ),
            "note": (
                "one source digest across every cell. The revision list may hold more "
                "than one entry when documentation was committed between cells; that "
                "moves HEAD without changing a line any cell executed"
            ),
        },
        "conclusion": (
            "every cell scored the same outer rows; a difference between two cells can "
            "only be the information set or the model"
        ),
    }


def anchor_to_snapshot(cell: dict[str, Any], spine: pd.DataFrame) -> dict[str, Any]:
    """Check a cell's predictions against the research data itself.

    :func:`recompute_cell` rebuilds a cell's metrics from its own persisted
    ``future_return`` and ``true_target``, so a cell whose label array was
    mis-joined reproduces its own wrong numbers exactly and reports no
    mismatch. This is the check that closes that: the persisted timestamp,
    label and realised return of every scored row must equal the snapshot's, at
    the row index the cell says it scored.

    Cheap, and it is the only place the evidence is compared against something
    the run did not produce.
    """
    predictions = cell["predictions"]
    rows = predictions["row_index"].to_numpy(dtype=np.int64)
    problems: list[str] = []
    if rows.max() >= len(spine):
        problems.append(f"row {int(rows.max())} is past the snapshot's {len(spine)} rows")
        return {"checked": 0, "problems": problems}

    for column, source in (
        ("future_return", "future_return"),
        ("true_target", "target"),
    ):
        persisted = predictions[column].to_numpy()
        expected = spine[source].to_numpy()[rows]
        if not np.allclose(persisted.astype(np.float64), expected.astype(np.float64)):
            worst = int(np.argmax(np.abs(persisted.astype(float) - expected.astype(float))))
            problems.append(
                f"{column} at research row {int(rows[worst])} is {persisted[worst]!r} but "
                f"the snapshot holds {expected[worst]!r}"
            )
    persisted_dates = pd.to_datetime(pd.Index(predictions["timestamp"].to_numpy()), utc=True)
    expected_dates = pd.to_datetime(pd.Index(spine["date"].to_numpy()[rows]), utc=True)
    if not persisted_dates.equals(expected_dates):
        problems.append("persisted timestamps do not match the snapshot at those rows")

    return {
        "checked": int(len(rows)),
        "against": "the committed research snapshot, at each prediction's own row index",
        "columns": ["timestamp", "true_target", "future_return"],
        "problems": problems,
    }


def recompute_cell(cell: dict[str, Any]) -> list[dict[str, Any]]:
    """Rebuild each fold's reported metrics from the persisted predictions.

    The market context a Sharpe needs is not in the prediction file — it is the
    candle-level price path — so the annualised Sharpe and the candle drawdown
    are deliberately *not* recomputed here. Everything that depends only on the
    realised trades is, which covers the numbers P2b actually compares.
    """
    payload = cell["payload"]
    spec = TargetSpec.from_dict(payload["target"])
    predictions = cell["predictions"]
    model = cell["model"]

    findings: list[dict[str, Any]] = []
    for record in payload["folds"]:
        fold = record["fold"]
        # Sorted by row index, because the trade walk in `realised_trades` is
        # sequential: this makes the check one about the persisted values rather
        # than about the order they happen to sit in the file.
        rows = predictions[predictions["fold"] == fold].sort_values("row_index")
        reported = record["outer_validation"][model]
        threshold = record["model"]["selection"]["threshold"]

        proba = rows[["p_short", "p_hold", "p_long"]].to_numpy(dtype=np.float64)
        actions = rows["selected_action"].to_numpy(dtype=np.int64)
        future_return = rows["future_return"].to_numpy(dtype=np.float64)
        row_index = rows["row_index"].to_numpy(dtype=np.int64)
        y_true = rows["true_target"].to_numpy(dtype=np.int64)

        mismatches: list[str] = []
        if len(rows) != reported["classification"]["n_samples"]:
            mismatches.append(
                f"{len(rows)} persisted rows vs "
                f"{reported['classification']['n_samples']} reported"
            )
        if not np.array_equal(ev.signals_from_proba(proba, threshold), actions):
            mismatches.append(
                "persisted selected_action is not the action the threshold implies"
            )
        if not np.allclose(rows["threshold"].to_numpy(dtype=np.float64), threshold):
            mismatches.append(
                "persisted threshold column disagrees with the selected threshold"
            )

        trading = ev.trading_metrics(actions, future_return, spec, row_index=row_index)
        classification = ev.classification_metrics(proba, y_true, threshold)
        for key in (
            "n_trades",
            "net_return",
            "gross_return",
            "total_costs",
            "win_rate",
            "profit_factor",
            "exposure",
            "turnover",
            "max_drawdown",
            "per_trade_sharpe",
            "avg_trade",
            "cost_per_trade",
        ):
            got, want = trading[key], reported["trading"][key]
            if got is None or want is None:
                if got is not want:
                    mismatches.append(f"trading.{key}: recomputed {got!r}, reported {want!r}")
            elif not np.isclose(float(got), float(want), rtol=1e-9, atol=1e-9):
                mismatches.append(f"trading.{key}: recomputed {got}, reported {want}")
        for key in (
            "macro_f1",
            "accuracy",
            "directional_accuracy",
            "coverage",
            "calibration_error",
        ):
            if not np.isclose(classification[key], reported["classification"][key], atol=1e-9):
                mismatches.append(
                    f"classification.{key}: recomputed {classification[key]}, "
                    f"reported {reported['classification'][key]}"
                )
        for key in (
            "class_distribution",
            "predicted_distribution",
            "per_class",
            "confusion_matrix",
            "n_samples",
            "threshold",
        ):
            if classification[key] != reported["classification"][key]:
                mismatches.append(f"classification.{key} disagrees with the predictions")
        findings.append(
            {
                "fold": fold,
                "samples": len(rows),
                "recomputed_from": PREDICTIONS_NAME,
                "recomputed_keys": {
                    "trading": [
                        "n_trades",
                        "net_return",
                        "gross_return",
                        "total_costs",
                        "win_rate",
                        "profit_factor",
                        "exposure",
                        "turnover",
                        "max_drawdown",
                        "per_trade_sharpe",
                        "avg_trade",
                        "cost_per_trade",
                    ],
                    "classification": [
                        "macro_f1",
                        "accuracy",
                        "directional_accuracy",
                        "coverage",
                        "calibration_error",
                        "class_distribution",
                        "predicted_distribution",
                        "per_class",
                        "confusion_matrix",
                        "n_samples",
                        "threshold",
                    ],
                },
                # Named, not omitted. A recomputation that covers ten of eighteen
                # keys and reports "0 mismatches" invites a reader to believe it
                # covered all eighteen.
                "not_recomputed": [
                    "trading.annualised_sharpe, trading.candle_max_drawdown and "
                    "trading.elapsed_intervals need the candle price path, which the "
                    "prediction file does not carry",
                    "trading.sharpe_basis, trading.per_trade_sharpe_reason and "
                    "trading.annualised_sharpe_reason are prose, not measurements",
                ],
                "mismatches": mismatches,
            }
        )
    return findings


def fold_row(record: dict[str, Any], model: str) -> dict[str, Any]:
    """Everything reported for one cell on one fold, flat."""
    outer = record["outer_validation"][model]
    row = {
        "fold": record["fold"],
        "period_start": record["periods"]["outer_validation"]["start"],
        "period_end": record["periods"]["outer_validation"]["end"],
        "samples": record["samples"]["outer_validation"],
        "threshold": record["model"]["selection"]["threshold"],
        "class_distribution": outer["classification"]["class_distribution"],
        "predicted_distribution": outer["classification"]["predicted_distribution"],
    }
    for name, (section, key) in FOLD_METRICS.items():
        row[name] = outer[section][key]
    return row


def spread_of(values: Iterable[Any]) -> dict[str, Any]:
    """Mean, std, min, median, max over the folds where the metric is defined."""
    defined = [float(v) for v in values if v is not None]
    if not defined:
        return {
            "mean": None,
            "std": None,
            "min": None,
            "median": None,
            "max": None,
            "defined_folds": 0,
        }
    return {
        "mean": round(statistics.fmean(defined), 6),
        "std": round(statistics.stdev(defined), 6) if len(defined) > 1 else 0.0,
        "min": round(min(defined), 6),
        "median": round(statistics.median(defined), 6),
        "max": round(max(defined), 6),
        "defined_folds": len(defined),
    }


def build_matrix(cells: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Per model, per information set: the four folds and their aggregate."""
    matrix: dict[str, Any] = {}
    for cell in cells:
        model, set_name = cell["model"], cell["information_set"]
        folds = [fold_row(r, model) for r in cell["payload"]["folds"]]
        aggregate = {
            metric: spread_of(row[metric] for row in folds) for metric in FOLD_METRICS
        }
        aggregate["positive_net_return_folds"] = sum(1 for r in folds if r["net_return"] > 0)
        aggregate["total_folds"] = len(folds)
        matrix.setdefault(model, {})[set_name] = {
            "folds": folds,
            "aggregate": aggregate,
            "thresholds": [row["threshold"] for row in folds],
            "long_short_attribution": direction_attribution(
                cell["predictions"], TargetSpec.from_dict(cell["payload"]["target"])
            ),
        }
    return matrix


def build_deltas(matrix: dict[str, Any]) -> dict[str, Any]:
    """SMC minus control, and combined minus control, per model and per fold.

    A delta is only meaningful because the two cells scored the same rows, which
    :func:`check_cells_agree` has already proved by the time this runs.
    """
    deltas: dict[str, Any] = {}
    for model, by_set in matrix.items():
        if CONTROL not in by_set:
            continue
        control = {row["fold"]: row for row in by_set[CONTROL]["folds"]}
        for set_name, entry in by_set.items():
            if set_name == CONTROL:
                continue
            per_fold = []
            for row in entry["folds"]:
                base = control[row["fold"]]
                per_fold.append(
                    {
                        "fold": row["fold"],
                        "period_start": row["period_start"],
                        "period_end": row["period_end"],
                        **{
                            metric: (
                                None
                                if row[metric] is None or base[metric] is None
                                else round(float(row[metric]) - float(base[metric]), 6)
                            )
                            for metric in DELTA_METRICS
                        },
                    }
                )
            improved = sum(
                1 for d in per_fold if d["net_return"] is not None and d["net_return"] > 0
            )
            deltas.setdefault(model, {})[set_name] = {
                "vs": CONTROL,
                "per_fold": per_fold,
                "aggregate": {
                    metric: spread_of(d[metric] for d in per_fold) for metric in DELTA_METRICS
                },
                "net_return_improved_folds": improved,
                "total_folds": len(per_fold),
                "verdict": VERDICTS[improved],
            }
    return deltas


def to_markdown(payload: dict[str, Any]) -> str:
    matrix, deltas = payload["matrix"], payload["deltas"]
    contract = payload["contract"]
    # Derived from the data, never from a checkpoint's arm list. Reading the
    # arms from P2B_INFORMATION_SETS meant a P2c run — whose arms are
    # chart_structure_v1 and ohlcv14_plus_chart_structure_v1 — rendered a
    # document titled P2b containing only its control, with an empty
    # incremental-value section: 12 of 36 cell-folds, and the answer nowhere on
    # the page. The JSON was complete throughout, which is exactly why nobody
    # would have caught it by checking that the run succeeded.
    present = list(dict.fromkeys(k for by in matrix.values() for k in by))
    sets = [CONTROL] + [s for s in present if s != CONTROL]
    models = [m for m in SIMPLE_MODEL_NAMES if m in matrix]

    named = " or ".join(f"`{s}`" for s in sets if s != CONTROL) or "further columns"
    lines = [
        f"# {payload['checkpoint']} — do {named} add information beyond `{CONTROL}`?",
        "",
        f"**Research question:** {payload['question']}",
        "",
        f"{len(sets)} information sets, {len(models)} untuned models, four temporal outer",
        f"folds, one sample universe. `{CONTROL}` is the control, re-run under this code",
        "path rather than copied. The other arms are:",
        "",
    ]
    lines += [f"- `{s}`" for s in sets if s != CONTROL]
    lines += [
        "",
        f"**Research contract:** `{contract['contract_id']}` "
        f"(generation {contract['research_generation']}), semantic identity",
        f"`sha256:{contract['contract_hash']}`.",
        "",
        f"**Sealed test block:** everything at or after `{contract['sealed_test_start']}`.",
        "Not planned over, not fitted on, not selected on, not scored.",
        "**Styx was not opened.**",
        "",
        "**Statistical unit:** one temporal outer period per fold, four in total. These",
        "estimators are deterministic given their inputs, so no seed replication appears",
        "anywhere below:",
        "a second seed would copy this evidence rather than add to it.",
        "",
        f"**Adaptive status:** {payload['adaptive_status']}.",
        "",
        "## Sample-universe parity",
        "",
        payload["parity"]["conclusion"] + ". Checked across all "
        f"{len(payload['parity']['cells'])} cells:",
        "",
    ]
    lines += [f"- {item}" for item in payload["parity"]["identical_across_cells"]]

    recompute = payload["independent_recompute"]
    lines += [
        "",
        "## Independent recomputation",
        "",
        "Every reported trading and classification number was rebuilt from the "
        f"{recompute['cells_checked']} cells'",
        f"persisted `{PREDICTIONS_NAME}` files: "
        f"**{recompute['folds_checked']} cell-folds checked, "
        f"{recompute['mismatches']} mismatches**.",
        "",
    ]
    for note in recompute["not_recomputed"]:
        lines.append(f"- not recomputed: {note}")

    lines += ["", "## Per-fold outer validation", ""]
    for model in models:
        lines += [
            f"### {model}",
            "",
            "| information set | fold | outer period | samples | thr | trades | exposure | "
            "turnover | gross | costs | **net** | max DD | ann. Sharpe | per-trade Sharpe | "
            "win rate | profit factor | macro F1 | dir. acc |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- "
            "| --- | --- | --- | --- | --- |",
        ]
        for set_name in sets:
            entry = matrix[model].get(set_name)
            if entry is None:
                continue
            for row in entry["folds"]:
                lines.append(
                    f"| `{set_name}` | {row['fold']} | {row['period_start'][:10]} → "
                    f"{row['period_end'][:10]} | {row['samples']} | {row['threshold']:.2f} "
                    f"| {row['n_trades']} | {row['exposure']:.4f} | {row['turnover']:.1f} "
                    f"| {row['gross_return']:+.6f} | {row['total_costs']:.4f} "
                    f"| **{row['net_return']:+.6f}** | {row['max_drawdown']:.4f} "
                    f"| {ev.number(row['annualised_sharpe'], '.4f')} "
                    f"| {ev.number(row['per_trade_sharpe'], '.4f')} "
                    f"| {row['win_rate']:.4f} | {row['profit_factor']:.4f} "
                    f"| {row['macro_f1']:.4f} | {row['directional_accuracy']:.4f} |"
                )
        lines.append("")

    lines += [
        "## Across the four temporal folds",
        "",
        "| model | information set | net mean | net std | net min | net median | net max "
        "| positive folds | Sharpe mean | max DD mean | exposure mean | macro F1 mean |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for model in models:
        for set_name in sets:
            entry = matrix[model].get(set_name)
            if entry is None:
                continue
            agg = entry["aggregate"]
            net = agg["net_return"]
            lines.append(
                f"| {model} | `{set_name}` | {net['mean']:+.6f} | {net['std']:.6f} "
                f"| {net['min']:+.6f} | {net['median']:+.6f} | {net['max']:+.6f} "
                f"| **{agg['positive_net_return_folds']} of {agg['total_folds']}** "
                f"| {ev.number(agg['annualised_sharpe']['mean'], '.4f')} "
                f"| {agg['max_drawdown']['mean']:.4f} | {agg['exposure']['mean']:.4f} "
                f"| {agg['macro_f1']['mean']:.4f} |"
            )

    lines += [
        "",
        "## Incremental value of market structure",
        "",
        "Per model, per fold: the information set minus the `ohlcv14` control on the "
        "same rows.",
        "",
    ]
    for model in models:
        for set_name in deltas.get(model, {}):
            entry = deltas[model][set_name]
            if entry is None:
                continue
            lines += [
                f"### {model}: `{set_name}` − `{CONTROL}`",
                "",
                "| fold | outer period | Δ net | Δ ann. Sharpe | Δ max DD (lower is "
                "better) | Δ exposure | Δ trades | Δ macro F1 | Δ dir. acc |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
            for d in entry["per_fold"]:
                lines.append(
                    f"| {d['fold']} | {d['period_start'][:10]} → {d['period_end'][:10]} "
                    f"| **{d['net_return']:+.6f}** "
                    f"| {ev.number(d['annualised_sharpe'], '+.4f')} "
                    f"| {d['max_drawdown']:+.4f} | {d['exposure']:+.4f} "
                    f"| {d['n_trades']:+.0f} | {d['macro_f1']:+.4f} "
                    f"| {d['directional_accuracy']:+.4f} |"
                )
            agg = entry["aggregate"]["net_return"]
            lines += [
                "",
                f"Net return improved in **{entry['net_return_improved_folds']} of "
                f"{entry['total_folds']}** temporal folds "
                f"(mean Δ {agg['mean']:+.6f}, min {agg['min']:+.6f}, max {agg['max']:+.6f}).",
                "",
                f"**Verdict:** {entry['verdict']}.",
                "",
            ]

    lines += [
        "## Long / short attribution",
        "",
        "Additive decomposition of the realised trades a cell took. These are the two "
        "halves of",
        "one reported result, not two standalone strategies: neither side was selected "
        "for, and",
        "neither could have been traded on its own without the threshold that produced "
        "both.",
        "",
        "| model | information set | long trades | long hit | long mean net | "
        "short trades | short hit | short mean net |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for model in models:
        for set_name in sets:
            entry = matrix[model].get(set_name)
            if entry is None:
                continue
            att = entry["long_short_attribution"]
            long_, short = att["long"], att["short"]
            lines.append(
                f"| {model} | `{set_name}` | {long_['trades']} | {long_['hit_rate']:.4f} "
                f"| {long_['mean_net_return']:+.6f} | {short['trades']} "
                f"| {short['hit_rate']:.4f} | {short['mean_net_return']:+.6f} |"
            )

    refs = payload["economic_references"]
    lines += [
        "",
        "## Economic references",
        "",
        "CASH and buy-and-hold over the same outer windows. **Reference only.** No "
        "feature,",
        "model or threshold in P2b was selected using them, and buy-and-hold is fully",
        "exposed for the whole window while every cell above is exposed only while a",
        "position is open — the two are not comparable as strategies.",
        "",
        "| fold | outer period | CASH net | buy-and-hold net | buy-and-hold max DD |",
        "| --- | --- | --- | --- | --- |",
    ]
    for ref in refs:
        lines.append(
            f"| {ref['fold']} | {ref['period_start'][:10]} → {ref['period_end'][:10]} "
            f"| {ref['cash_net_return']:+.6f} | {ref['buy_and_hold_net_return']:+.6f} "
            f"| {ev.number(ref['buy_and_hold_max_drawdown'], '.4f')} |"
        )
    lines.append("")
    return "\n".join(lines)


def economic_reference_rows(cell: dict[str, Any]) -> list[dict[str, Any]]:
    """CASH and buy-and-hold per fold — one value, since every cell agrees."""
    rows = []
    for record in cell["payload"]["folds"]:
        refs = record["outer_validation"][REFERENCES_KEY]
        cash = refs.get("cash", {})
        hold = refs.get("buy_and_hold", {})
        rows.append(
            {
                "fold": record["fold"],
                "period_start": record["periods"]["outer_validation"]["start"],
                "period_end": record["periods"]["outer_validation"]["end"],
                "cash_net_return": cash.get("net_return", 0.0),
                "buy_and_hold_net_return": hold.get("net_return"),
                # `buy_and_hold_reference` emits `candle_max_drawdown`, not
                # `max_drawdown`. Reading the wrong key printed "n/a" in this column
                # of every comparison ever produced — telling a reader the number was
                # never computed, in the one table whose whole purpose is comparing a
                # strategy's drawdown to the asset's.
                "buy_and_hold_max_drawdown": hold.get("candle_max_drawdown"),
                "buy_and_hold_exposure": hold.get("exposure"),
            }
        )
    return rows


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = build_argparser().parse_args(argv)

    cells = [load_cell(Path(d)) for d in sorted(args.runs)]
    if not cells:
        raise ComparisonError("no cells to compare")
    seen = {(c["information_set"], c["model"]) for c in cells}
    if len(seen) != len(cells):
        raise ComparisonError("two cells cover the same information set and model")
    checkpoint = checkpoint_of(cells)
    logger.info("comparing %d %s cells: %s", len(cells), checkpoint.name, sorted(seen))

    parity = check_cells_agree(cells)

    manifest_path = Path(cells[0]["payload"]["snapshot"]["manifest"])
    manifest = json.loads(manifest_path.read_text())
    root = manifest_path.resolve().parent.parent.parent
    spine, _ = load_dataset(root / manifest["processed_outer_coverage"]["path"])
    anchored = {
        f"{c['information_set']}::{c['model']}": anchor_to_snapshot(c, spine) for c in cells
    }
    adrift = [
        {"cell": name, **result} for name, result in anchored.items() if result["problems"]
    ]
    if adrift:
        raise ComparisonError(
            "a cell's persisted predictions disagree with the research snapshot:\n"
            + json.dumps(adrift, indent=2)
        )

    recomputed = {f"{c['information_set']}::{c['model']}": recompute_cell(c) for c in cells}
    mismatches = [
        {"cell": name, **finding}
        for name, findings in recomputed.items()
        for finding in findings
        if finding["mismatches"]
    ]
    if mismatches:
        raise ComparisonError(
            "a cell's report disagrees with its own persisted predictions:\n"
            + json.dumps(mismatches, indent=2)
        )

    matrix = build_matrix(cells)
    deltas = build_deltas(matrix)

    payload = {
        "checkpoint": checkpoint.name,
        "question": checkpoint.question,
        "evidence_class": DERIVED,
        "derived_from": [str(c["dir"]) for c in cells],
        "control": checkpoint.control,
        "contract": cells[0]["payload"]["contract"],
        "snapshot": cells[0]["payload"]["snapshot"],
        "feature_spec": cells[0]["payload"]["feature_spec"],
        "sizes": cells[0]["payload"]["sizes"],
        "target": cells[0]["payload"]["target"],
        "threshold_selection": cells[0]["payload"]["threshold_selection"],
        "alignment": cells[0]["payload"]["alignment"],
        "sealed_test": False,
        "statistical_unit": (
            "one temporal outer period per fold, four in total; no seed replication, "
            "because these estimators are deterministic given their inputs"
        ),
        "folds_are_not_independent": (
            "the geometry is P2a's, and in it fold k's inner-validation block is "
            "exactly fold k-1's reported outer block, while folds 2 and 3 train on "
            "earlier folds' outer blocks. No fold touches its own outer rows before "
            "scoring them, so no fold's own number is contaminated — but a regime "
            "spanning a fold boundary can move one fold's result and the next fold's "
            "selected threshold together. 'Three of four' is therefore fewer than "
            "four independent draws, and the verdict rule should be read that way"
        ),
        "adaptive_status": checkpoint.adaptive_status,
        "verdict_rule": VERDICTS,
        "parity": parity,
        "snapshot_anchoring": {
            "cells_checked": len(cells),
            "rows_checked": sum(r["checked"] for r in anchored.values()),
            "problems": len(adrift),
            "note": (
                "every scored row's timestamp, label and realised return was compared "
                "against the committed snapshot at the row index the cell recorded — the "
                "one check made against data the run did not produce"
            ),
        },
        "independent_recompute": {
            "cells_checked": len(cells),
            "folds_checked": sum(len(v) for v in recomputed.values()),
            "mismatches": len(mismatches),
            "recomputed_keys": recomputed[next(iter(recomputed))][0]["recomputed_keys"],
            "not_recomputed": recomputed[next(iter(recomputed))][0]["not_recomputed"],
            "per_cell": recomputed,
        },
        "economic_references": economic_reference_rows(cells[0]),
        "matrix": matrix,
        "deltas": deltas,
        "cells": [
            {
                "information_set": c["information_set"],
                "model": c["model"],
                "dir": str(c["dir"]),
                "flattened_input_dim": c["payload"]["information_parity"][
                    "flattened_input_dim"
                ],
                "n_features": c["payload"]["information_parity"]["n_features"],
            }
            for c in cells
        ],
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / COMPARISON_JSON).write_text(json.dumps(payload, indent=2, default=str) + "\n")
    (out_dir / COMPARISON_MD).write_text(to_markdown(payload))
    logger.info("wrote %s and %s", out_dir / COMPARISON_JSON, out_dir / COMPARISON_MD)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
