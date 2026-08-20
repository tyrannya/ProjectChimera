"""P2a: the untuned simple-model benchmark, on MTST's own samples.

    python -m nn.benchmark --dataset data/datasets/binance_BTC_USDT_1h.parquet \
        --seed 42 --out artifacts/benchmark/btc_p2a_seed_42

**The question.** Does model complexity add predictive or economic value when
simple models receive exactly the same information as the MTST Transformer?
This is a diagnostic, not a search for a profitable configuration: one thing
changes — the model family — and everything else is held at the values the
frozen BTC v4 evidence was produced under.

**What "the same information" means here.** Not "the same features": *the same
samples*. This module does not build a dataset, a window or a split of its own.
It plans folds with :func:`nn.walkforward.plan_nested_folds`, materialises train
and inner-validation windows with :func:`nn.train.prepare_research_windows`, and
scores the outer block through :func:`nn.train.score_frozen_split` — the exact
three functions the MTST walk-forward runs through. The only thing it adds is
:func:`nn.simple_models.flatten_windows`, which turns MTST's
``(seq_len, n_features)`` sequence into one ``seq_len * n_features`` row in a
declared order. Eligible rows, targets, sequence boundaries, gap handling,
feature order, horizon, labels, costs, embargo geometry and the sealed boundary
are therefore not *matched* — they are the same code producing the same arrays,
which is what makes the comparison in :mod:`nn.benchmark_compare` mean anything.

**Fold roles are MTST's fold roles.** Training fits the scaler and the model.
Inner validation selects the decision threshold, through
:func:`nn.evaluate.select_threshold` at its declared grid and objective — the
same rule, not a variant tuned for simple models. Outer validation is frozen
evaluation evidence: no fit, no calibration, no early stopping, no model
selection. The estimators here have no early stopping to disable, and
:func:`nn.simple_models.fit_simple_model` takes no validation array at all.

**Nothing is tuned.** The three configurations are predeclared constants in
:mod:`nn.simple_models`. There is no search, no per-fold variation beyond the
seed, and no flag on this command line that can change a model's
hyperparameters.

**These are research folds.** P2a was designed after prior outer-validation
results had been seen, so its outer blocks are adaptive research evidence and
not a pristine out-of-sample test. The sealed block is not planned over, fitted
on, selected on or scored, and remains unopened.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from chimera.contracts import CLASS_ORDER
from nn import evaluate as ev
from nn.baselines import MajorityClassBaseline, MomentumBaseline
from nn.dataset import Split
from nn.research_contract import (
    ContractScopeError,
    ResearchContract,
    add_contract_argument,
    load_contract,
)
from nn.simple_models import (
    FLATTEN_ORDER,
    SIMPLE_MODEL_NAMES,
    SIMPLE_MODELS,
    SimpleModelSpec,
    fit_simple_model,
    flattened_column_names,
)
from nn.train import (
    ResearchData,
    RunConfig,
    load_research_data,
    prepare_research_windows,
    resolve_device,
    score_frozen_split,
)
from nn.walkforward import (
    REFERENCES_KEY,
    SUMMARY_METRICS,
    FoldPlan,
    plan_nested_folds,
    resolve_sizes,
    spread,
)

logger = logging.getLogger(__name__)

#: What this checkpoint is, recorded in every artifact it writes.
CHECKPOINT = "P2a"

#: The file this module writes. Deliberately not ``walkforward.json``: a P2a run
#: is not an MTST walk-forward run, and a tool that reads one must not silently
#: accept the other.
ARTIFACT_NAME = "benchmark.json"
MARKDOWN_NAME = "benchmark.md"

#: Per-sample outer predictions, beside the artifact. Same columns as
#: :data:`nn.regime.PREDICTION_COLUMNS` plus ``model``, so the repository's
#: existing prediction readers and the directional attribution in
#: :func:`nn.regime.direction_attribution` work on them unchanged.
PREDICTIONS_NAME = "outer_predictions.parquet"

#: The statistical/rule floors, refitted per fold exactly as MTST refits them.
#:
#: They are not new work: the majority rule is fitted on the fold's training
#: labels and the momentum rule is fitted on nothing, so on identical geometry
#: and identical data their outer numbers are a constant that the frozen MTST
#: artifacts already recorded. :mod:`nn.benchmark_compare` requires the two to
#: agree exactly before it aggregates anything, which turns them into a direct,
#: data-level proof that both families scored the same rows.
BASELINE_NAMES = ("majority_baseline", "momentum_baseline")

#: Everything reported per fold, in table order.
MODELS = SIMPLE_MODEL_NAMES + BASELINE_NAMES

#: How the decision threshold is chosen, recorded so a reader need not infer it.
THRESHOLD_OBJECTIVE = (
    "maximise net return after round-trip costs on the inner-validation block, "
    "subject to at least min_trades realised trades; falls back to the most "
    "permissive grid point when no candidate clears the floor"
)


def threshold_grid() -> np.ndarray:
    """The candidate thresholds, materialised so the artifact can record them.

    :func:`nn.evaluate.select_threshold` builds this same grid when none is
    passed. It is built here and passed in explicitly for one reason: a grid
    that only exists inside the selector cannot be written down, and a
    threshold nobody can see the candidates for is not auditable.
    """
    return np.round(np.arange(0.34, 0.91, 0.02), 4)


def fit_baselines(data: ResearchData, y_train: np.ndarray) -> dict[str, Any]:
    """The two rule baselines, fitted the way :func:`nn.train.fit_and_validate` fits them."""
    return {
        "majority_baseline": MajorityClassBaseline().fit(y_train),
        "momentum_baseline": MomentumBaseline(
            feature_index=data.feature_names.index("ema_cross")
        ),
    }


def run_fold(
    fold: int,
    data: ResearchData,
    plan: FoldPlan,
    run: RunConfig,
    specs: tuple[SimpleModelSpec, ...],
    *,
    min_trades: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fit on train, select the threshold on inner, measure once on outer.

    One call per fold covers every model in ``specs``, because they must all see
    the same arrays: ``prepared`` is built once and handed to each of them, so a
    difference between two models in this record cannot be a difference in the
    samples they were given.
    """
    prepared = prepare_research_windows(data, plan.train, plan.inner, run.seq_len)
    baselines = fit_baselines(data, prepared.y_train)
    # Required by the shared frozen-scoring signature and unused on this path:
    # the probabilities come from `proba_of`, not from a torch module.
    device = resolve_device("cpu")
    grid = threshold_grid()
    inner_return = data.future_return[prepared.idx_val]

    outer_reports: dict[str, Any] = {}
    per_model: dict[str, Any] = {}
    frames: list[pd.DataFrame] = []
    reference_outer: dict[str, Any] | None = None

    for spec in specs:
        # Per-fold seed, derived exactly as nn.walkforward derives MTST's, so a
        # lucky initialisation cannot flatter a whole run and a given --seed
        # still reproduces.
        seed = run.seed + fold
        logger.info("fold %d: fitting %s (seed %d)", fold, spec.name, seed)
        fitted = fit_simple_model(spec, prepared.X_train, prepared.y_train, seed=seed)

        inner_proba = fitted.predict_proba(prepared.X_val)
        threshold, threshold_report = ev.select_threshold(
            inner_proba,
            inner_return,
            data.target_spec,
            grid=grid,
            min_trades=min_trades,
            row_index=prepared.idx_val,
        )

        # The model and its threshold are frozen from here on. `score_frozen_split`
        # fits nothing; `plan.outer` reaches this call and no other.
        scored = score_frozen_split(
            data,
            prepared.scaler,
            plan.outer,
            run.seq_len,
            baselines=baselines,
            threshold=threshold,
            device=device,
            proba_of=fitted.predict_proba,
            model_name=spec.name,
        )
        outer_reports[spec.name] = scored.reports[spec.name]

        # Every model scored the same outer block with the same frozen baselines,
        # so the baseline reports and the economic references must be one value,
        # not three. Checked rather than assumed: if they ever differ, the models
        # were not given the same rows and nothing below this line is comparable.
        current = {name: scored.reports[name] for name in BASELINE_NAMES}
        current[REFERENCES_KEY] = scored.economic_references
        if reference_outer is None:
            reference_outer = current
        elif current != reference_outer:
            raise AssertionError(
                f"fold {fold}: scoring {spec.name} produced different baseline or "
                "economic-reference numbers than the previous model on the same outer "
                "block, so the models did not see the same rows"
            )

        per_model[spec.name] = {
            "provenance": fitted.provenance(),
            "selection": {
                "block": "inner_validation",
                "threshold": threshold,
                "objective": THRESHOLD_OBJECTIVE,
                "threshold_grid": [float(value) for value in grid],
                "min_trades": min_trades,
                # The selection score, kept because it is what the threshold was
                # chosen on — and deliberately outside `outer_validation`, which
                # is the only block reported as a result.
                "inner_validation_trading": threshold_report,
            },
        }
        frames.append(outer_predictions(spec.name, fold, seed, data, scored, threshold))

    assert reference_outer is not None
    outer_reports.update(reference_outer)

    return {
        "fold": fold,
        "run_seed": run.seed,
        "fold_seed": run.seed + fold,
        "samples": {
            "train": len(prepared.X_train),
            "inner_validation": len(prepared.X_val),
            "outer_validation": len(frames[0]),
        },
        "periods": {
            "train": data.period(plan.train),
            "inner_validation": data.period(plan.inner),
            "outer_validation": data.period(plan.outer),
        },
        "models": per_model,
        "outer_validation": outer_reports,
    }, pd.concat(frames, ignore_index=True)


def outer_predictions(
    model: str,
    fold: int,
    seed: int,
    data: ResearchData,
    scored: Any,
    threshold: float,
) -> pd.DataFrame:
    """One row per outer sample: what was predicted, and what happened.

    Same columns as ``nn.walkforward``'s prediction file, plus ``model``, so the
    two are read by the same code and compared row for row. ``selected_action``
    comes from :func:`nn.evaluate.signals_from_proba` — the function the reports
    themselves use — so the persisted action is the action that was scored.
    """
    actions = ev.signals_from_proba(scored.proba, threshold)
    short_idx, hold_idx, long_idx = range(len(CLASS_ORDER))
    return pd.DataFrame(
        {
            "model": np.repeat(model, len(scored.idx)),
            "fold": np.full(len(scored.idx), fold, dtype=np.int64),
            "seed": np.full(len(scored.idx), seed, dtype=np.int64),
            "row_index": scored.idx.astype(np.int64),
            "timestamp": data.dates[scored.idx],
            "true_target": scored.y.astype(np.int64),
            "future_return": data.future_return[scored.idx],
            "p_short": scored.proba[:, short_idx],
            "p_hold": scored.proba[:, hold_idx],
            "p_long": scored.proba[:, long_idx],
            "selected_action": actions.astype(np.int64),
            "threshold": np.full(len(scored.idx), float(threshold)),
        }
    )


def write_predictions(out_dir: Path, frames: list[pd.DataFrame], boundary: int) -> Path:
    """Write the per-sample outer predictions, refusing to persist a sealed row.

    The outer blocks are below the boundary by construction, so this should
    never fire. It is here for the same reason the identical guard in
    ``nn.walkforward`` is: this file is the one thing the benchmark writes that
    carries per-row market outcomes, and a silent off-by-one would put sealed
    rows on disk under a research filename.
    """
    predictions = pd.concat(frames, ignore_index=True)
    highest = int(predictions["row_index"].max())
    if highest >= boundary:
        raise AssertionError(
            f"refusing to write outer predictions: row {highest} is at or beyond the "
            f"sealed test block starting at {boundary}"
        )
    path = out_dir / PREDICTIONS_NAME
    predictions.to_parquet(path, index=False)
    return path


def summarise(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean and standard deviation of each outer metric across folds, per model.

    Reads ``outer_validation`` and nothing else, for the reason
    :func:`nn.walkforward.summarise` reads only that block: the inner block
    chose the threshold, so aggregating it would report a selection score as a
    result.
    """
    summary: dict[str, Any] = {
        "folds": len(results),
        "aggregated_from": "outer_validation",
        "per_model": {},
    }
    for name in MODELS:
        stats: dict[str, Any] = {}
        for metric, (section, key) in SUMMARY_METRICS.items():
            raw = [r["outer_validation"][name][section][key] for r in results]
            stats[metric] = spread([None if v is None else float(v) for v in raw])
        stats["positive_net_return_folds"] = sum(
            1 for v in stats["net_return"]["values"] if v > 0
        )
        summary["per_model"][name] = stats

    summary["thresholds"] = {
        name: [r["models"][name]["selection"]["threshold"] for r in results]
        for name in SIMPLE_MODEL_NAMES
    }
    return summary


def parity_record(data: ResearchData, seq_len: int) -> dict[str, Any]:
    """The information-parity claim, written down so it can be checked.

    Everything a later reader needs to verify that a P2a row is MTST's own
    window: the ordered feature names, the sequence length, the flattened width,
    the flattening rule, every flattened column name in order, and the name of
    the code that built the samples.
    """
    columns = flattened_column_names(data.feature_names, seq_len)
    return {
        "seq_len": seq_len,
        "n_features": len(data.feature_names),
        "feature_names": list(data.feature_names),
        "flattened_input_dim": len(columns),
        "flatten_order": FLATTEN_ORDER,
        "flattened_columns": columns,
        "sample_construction": (
            "nn.dataset.build_windows, reached through nn.train.prepare_research_windows "
            "(train and inner validation) and nn.train.score_frozen_split (outer "
            "validation) — the same calls nn.walkforward makes for MTST"
        ),
        "scaling": (
            "nn.dataset.StandardScaler fitted on the fold's training rows only and "
            "applied unchanged to inner and outer validation, exactly as for MTST. The "
            "tree models do not require scaling — a positive per-feature affine map "
            "cannot change which rows fall on which side of a split — but they are given "
            "the same scaled array so that no model in this benchmark sees a different "
            "number from the one MTST saw"
        ),
        "feature_selection": "none",
        "fitted_on_validation": "nothing",
    }


def to_markdown(
    payload: dict[str, Any], contract: ResearchContract, sealed_period: dict[str, Any]
) -> str:
    """A short per-run summary. The cross-seed comparison is a separate tool."""
    parity = payload["information_parity"]
    summary = payload["summary"]
    lines = [
        f"# P2a simple-model benchmark (seed {payload['config']['seed']})",
        "",
        "Three untuned models — logistic regression, LightGBM, XGBoost — fitted on the",
        "**same samples** the MTST Transformer is fitted on: each row is MTST's",
        f"`{parity['seq_len']} x {parity['n_features']}` window flattened to",
        f"`{parity['flattened_input_dim']}` values in a declared order. Nothing is tuned;",
        "the configurations are predeclared constants.",
        "",
        f"**Research contract:** `{contract.contract_id}` "
        f"(generation {contract.research_generation}, {contract.domain}, "
        f"{contract.scope.exchange} {contract.scope.pair} {contract.scope.timeframe}),",
        f"semantic identity `sha256:{contract.contract_hash}`.",
        "",
        f"**Research input:** `sha256:{payload['research_input']['research_input_hash']}` "
        f"over {payload['research_input']['research_rows']} research-visible rows.",
        "",
        f"**Sealed test block:** everything at or after "
        f"`{payload['sealed_test']['anchor_timestamp']}`, rows "
        f"{sealed_period['row_range'][0]}-{sealed_period['row_range'][1]}. Not planned "
        "over, not",
        "fitted on, not selected on, not scored. `sealed_test: false`.",
        "",
        "## Fold geometry and selected thresholds",
        "",
        "| fold | train rows | inner rows | outer rows | outer period | "
        + " | ".join(f"{name} thr" for name in SIMPLE_MODEL_NAMES)
        + " |",
        "| --- | --- | --- | --- | --- | "
        + " | ".join("---" for _ in SIMPLE_MODEL_NAMES)
        + " |",
    ]
    for result in payload["folds"]:
        periods = result["periods"]
        outer = periods["outer_validation"]
        thresholds = " | ".join(
            f"{result['models'][name]['selection']['threshold']:.2f}"
            for name in SIMPLE_MODEL_NAMES
        )
        lines.append(
            f"| {result['fold']} | {periods['train']['row_range'][0]}-"
            f"{periods['train']['row_range'][1]} | "
            f"{periods['inner_validation']['row_range'][0]}-"
            f"{periods['inner_validation']['row_range'][1]} | "
            f"{outer['row_range'][0]}-{outer['row_range'][1]} | "
            f"{outer['start'][:10]} to {outer['end'][:10]} | {thresholds} |"
        )

    lines += [
        "",
        "## Outer validation (the reported block)",
        "",
        f"`ann. Sharpe` is {ev.SHARPE_BASIS}. `n/a` means undefined, not zero.",
        "",
        "| fold | model | trades | net return | ann. Sharpe | per-trade Sharpe | "
        "exposure | max DD | macro F1 | dir acc | coverage |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for result in payload["folds"]:
        for name in MODELS:
            report = result["outer_validation"][name]
            trading, classification = report["trading"], report["classification"]
            lines.append(
                f"| {result['fold']} | {name} | {trading['n_trades']} | "
                f"{trading['net_return']:+.4f} | "
                f"{ev.number(trading['annualised_sharpe'], '.2f')} | "
                f"{ev.number(trading['per_trade_sharpe'], '.2f')} | "
                f"{trading['exposure']:.4f} | {trading['max_drawdown']:.4f} | "
                f"{classification['macro_f1']:.4f} | "
                f"{classification['directional_accuracy']:.4f} | "
                f"{classification['coverage']:.4f} |"
            )

    lines += [
        "",
        "### Across folds, outer validation only (mean ± std)",
        "",
        "| model | net return | ann. Sharpe | exposure | max DD | macro F1 | dir acc | "
        "trades | positive folds |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name in MODELS:
        stats = summary["per_model"][name]

        def cell(metric: str, places: int = 4, stats: dict[str, Any] = stats) -> str:
            value = stats[metric]
            if value["mean"] is None:
                return f"n/a (0/{summary['folds']})"
            body = f"{value['mean']:.{places}f} ± {value['std']:.{places}f}"
            missing = summary["folds"] - value["defined_folds"]
            return (
                body
                if not missing
                else f"{body} ({value['defined_folds']}/{summary['folds']})"
            )

        lines.append(
            f"| {name} | {cell('net_return')} | {cell('annualised_sharpe', 2)} | "
            f"{cell('exposure')} | {cell('max_drawdown')} | {cell('macro_f1')} | "
            f"{cell('directional_accuracy')} | {cell('n_trades', 1)} | "
            f"{stats['positive_net_return_folds']}/{summary['folds']} |"
        )

    lines += [
        "",
        "This is one seed. The comparison against the frozen MTST v4 evidence, the",
        "economic references and the other seeds is `python -m nn.benchmark_compare`;",
        "no verdict is stated here, because a single seed cannot support one.",
        "",
        "P2a is an adaptive follow-up designed after prior outer-validation results were",
        "seen. These outer folds are research evidence, not a pristine out-of-sample",
        "test, and nothing here is a claim about live profitability. The sealed test",
        "block remains unopened.",
        "",
    ]
    return "\n".join(lines)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="P2a: untuned simple models on MTST's own samples."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", default="artifacts/benchmark")
    add_contract_argument(parser)
    parser.add_argument("--folds", type=int, default=4)
    # Fold geometry, with the same names, meanings and defaults as
    # nn.walkforward's. They are a fraction of the *research* region, and they
    # are here so a P2a run can be planned onto the geometry a frozen MTST run
    # already used — not so the geometry can be explored.
    parser.add_argument("--min-train-frac", type=float, default=0.45)
    parser.add_argument("--min-train-size", type=int, default=None)
    parser.add_argument("--inner-val-frac", type=float, default=0.10)
    parser.add_argument("--inner-val-size", type=int, default=None)
    parser.add_argument("--outer-val-frac", type=float, default=0.10)
    parser.add_argument("--outer-val-size", type=int, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-trades",
        type=int,
        default=10,
        help=(
            "Trade floor for threshold selection on inner validation. The default is "
            "nn.evaluate.select_threshold's own, which is what MTST used."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(SIMPLE_MODEL_NAMES),
        choices=list(SIMPLE_MODEL_NAMES),
        help="Subset of the predeclared benchmark to run. For smoke runs only.",
    )
    # There is deliberately no flag here for any model hyperparameter, and no
    # flag for a search. The configurations are predeclared constants in
    # nn.simple_models; a command line that could move them would make this a
    # search wearing a benchmark's name.
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = build_argparser().parse_args(argv)

    data = load_research_data(args.dataset)
    contract = load_contract(args.research_contract)
    try:
        sealed = data.sealed_boundary(contract)
    except ContractScopeError as exc:
        raise SystemExit(str(exc)) from exc
    except ValueError as exc:
        raise SystemExit(f"cannot resolve the sealed test boundary: {exc}") from exc

    boundary = sealed.start_row
    fingerprint = data.input_fingerprint(sealed)
    sealed_split = Split("sealed_test", boundary, data.n_rows)
    min_train, inner_size, outer_size, step = resolve_sizes(args, boundary)
    plans = plan_nested_folds(boundary, args.folds, min_train, inner_size, outer_size, step)
    specs = tuple(spec for spec in SIMPLE_MODELS if spec.name in set(args.models))

    logger.warning(
        "P2a benchmark under research contract %s. %d nested folds over rows [0, %d) of "
        "%d. Models: %s, all untuned and predeclared. Outer blocks: %s. The sealed test "
        "block starts at the immutable anchor %s, row %d, and is NOT planned over, "
        "fitted on, selected on, or evaluated.",
        contract.label,
        len(plans),
        boundary,
        data.n_rows,
        ", ".join(spec.name for spec in specs),
        ", ".join(f"[{p.outer.start}, {p.outer.end})" for p in plans),
        sealed.anchor.isoformat(),
        boundary,
    )

    run = RunConfig(seed=args.seed, seq_len=args.seq_len)
    results = []
    frames = []
    for i, plan in enumerate(plans):
        logger.info("--- fold %d/%d ---", i + 1, len(plans))
        record, fold_frame = run_fold(i, data, plan, run, specs, min_trades=args.min_trades)
        results.append(record)
        frames.append(fold_frame)

    payload = {
        "checkpoint": CHECKPOINT,
        "purpose": (
            "Does model complexity add predictive or economic value when simple models "
            "receive exactly the same information as MTST? Model family is the only "
            "thing that varies."
        ),
        "dataset": data.ds_meta.to_dict(),
        "dataset_path": args.dataset,
        "research_input": fingerprint.to_dict(),
        "config": vars(args),
        "sizes": {
            "min_train": min_train,
            "inner_val_size": inner_size,
            "outer_val_size": outer_size,
            "step": step,
        },
        "sealed_test": {
            **sealed.to_dict(),
            "period": data.period(sealed_split),
            "evaluated": False,
        },
        "research_rows": boundary,
        "test_evaluated": False,
        "reported_block": "outer_validation",
        "outer_predictions": PREDICTIONS_NAME,
        "information_parity": parity_record(data, args.seq_len),
        "target": data.ds_meta.target_spec,
        "threshold_selection": {
            "block": "inner_validation",
            "objective": THRESHOLD_OBJECTIVE,
            "grid": [float(v) for v in threshold_grid()],
            "min_trades": args.min_trades,
            "applied_to_outer_validation": "the already-selected value, unchanged",
        },
        "tuning": {
            "search": "none",
            "hyperparameter_optimisation": None,
            "early_stopping": False,
            "outer_validation_used_for": "evaluation only",
        },
        "models": [spec.name for spec in specs],
        "folds": results,
        "summary": summarise(results),
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = write_predictions(out_dir, frames, boundary)
    logger.info(
        "Wrote %d outer predictions to %s",
        sum(len(frame) for frame in frames),
        predictions_path,
    )
    (out_dir / ARTIFACT_NAME).write_text(json.dumps(payload, indent=2, default=str) + "\n")
    markdown = to_markdown(payload, contract, data.period(sealed_split))
    (out_dir / MARKDOWN_NAME).write_text(markdown)
    print(markdown)
    logger.info("Wrote results to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
