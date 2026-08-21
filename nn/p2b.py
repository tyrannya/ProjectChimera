"""P2b: does causal market structure add information beyond OHLCV14?

    python -m nn.p2b --information-set ohlcv14_plus_smc_v1 --model xgboost \
        --out artifacts/benchmark/btc_p2b_ohlcv14_plus_smc_v1_xgboost

**The question.** P2a answered "does the model family change what can be
extracted from OHLCV14?" and found that it barely does: the best of four
families managed a mean outer net return near zero over four temporal periods.
That result makes the next question an *information* question, not an
architecture one. P2b changes the feature columns and holds everything else —
the models, the folds, the labels, the costs, the threshold rule, the sealed
boundary — at the values P2a ran under.

**Three arms.** ``ohlcv14`` is P2a's control, re-run here rather than copied so
the comparison is between two live numbers produced by one code path.
``smc_v1`` is causal market structure alone (``docs/smc_v1.md``).
``ohlcv14_plus_smc_v1`` is both. The comparison that matters is one model, one
fold, one sample set, three column sets.

**One cell per run.** A run covers one information set and one model across all
four folds, because the deterministic estimators here are single-threaded and
nine independent cells finish far sooner on four cores than one process would.
Splitting the work cannot change it: nothing is shared between cells except the
input data, and :mod:`nn.p2b_compare` refuses to aggregate cells whose baselines,
economic references or sample-index hashes disagree — which is a direct,
data-level proof that every cell scored the same rows.

**Where the rows come from.** The committed research snapshot under
``data/research/``, not a locally built dataset. The snapshot is a *prefix* of
the canonical processed dataset, truncated at the last row any outer block can
reach, so it cannot resolve the sealed boundary itself — by design, since it
contains no sealed row to resolve against. The fold geometry therefore comes
from the manifest's record of the canonical research region, and every planned
row is asserted to fall inside the snapshot before anything is fitted.

**These are research folds.** P2b was designed after P2a's outer results had
been seen, so its outer blocks are adaptive research evidence and not a pristine
out-of-sample test. The sealed block is not planned over, fitted on, selected on
or scored, and remains unopened.
"""

from __future__ import annotations

import argparse
import json
import logging
import hashlib
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info, threadpool_limits

from nn import evaluate as ev
from nn.benchmark import (
    BASELINE_NAMES,
    MARKDOWN_NAME,
    THRESHOLD_OBJECTIVE,
    fit_baselines,
    outer_predictions,
    threshold_grid,
)
from nn.data_pipeline import load_dataset
from nn.information_sets import (
    OHLCV14,
    P2B_INFORMATION_SETS,
    ablation_set_names,
    AlignedResearchSamples,
    build_information_set_views,
    feature_spec_identity,
)
from nn.research_contract import ContractScopeError, ResearchContract, load_contract
from nn.simple_models import (
    FLATTEN_ORDER,
    SIMPLE_MODEL_NAMES,
    SIMPLE_MODELS,
    SimpleModelSpec,
    fit_simple_model,
    flattened_column_names,
)
from nn.train import (
    BASELINE_DECISION_THRESHOLD,
    ResearchData,
    prepare_research_windows,
    resolve_device,
    score_frozen_split,
)
from nn.walkforward import FoldPlan, REFERENCES_KEY, SUMMARY_METRICS, plan_nested_folds, spread

logger = logging.getLogger(__name__)

CHECKPOINT = "P2b"

#: Deliberately not ``benchmark.json``: a P2b cell is not a P2a run, and a tool
#: that reads one must not silently accept the other.
ARTIFACT_NAME = "p2b.json"
PREDICTIONS_NAME = "outer_predictions.parquet"

#: Whose columns the rule floors and economic references are computed from.
CONTROL_SET = OHLCV14

DEFAULT_MANIFEST = Path("data/research/btc_usdt_1h_gen1_snapshot_manifest.json")

#: The geometry P2a ran under. Fractions of the *canonical* research region, not
#: of the snapshot: the snapshot is a truncated prefix, and taking fractions of
#: its length would silently plan a different set of folds.
MIN_TRAIN_FRAC = 0.45
INNER_VAL_FRAC = 0.10
OUTER_VAL_FRAC = 0.10
FOLDS = 4
SEQ_LEN = 64
SEED = 42
MIN_TRADES = 10


#: Native threads allowed while a model is fitted.
#:
#: Not a performance setting. ``LogisticRegression``'s lbfgs solver reduces
#: through multi-threaded BLAS, and a floating-point reduction whose order
#: depends on the thread count is not reproducible: measured on this dataset,
#: fold 0's outer probabilities move by up to 0.025 between a one-thread and a
#: four-thread fit, which is enough to select a different threshold off the grid
#: and report a materially different net return. LightGBM and XGBoost are
#: already pinned to ``n_jobs=1`` in :mod:`nn.simple_models` for the same
#: reason; this closes the remaining hole, and the observed pool state is
#: recorded in every artifact so the claim can be checked rather than trusted.
FIT_THREADS = 1


def code_revision() -> dict[str, Any]:
    """Which revision of this code produced the artifact.

    Every other identity a cell records — the contract hash, the snapshot
    hashes, the feature-spec hash, the library versions — describes the *data*
    and the *definitions*. None of them changes when the runner does, so nine
    cells built by nine different revisions of this module join into one
    comparison without a word. That is not hypothetical: two revisions of
    `build_information_set_views` produced identical `alignment["folds"]` blocks
    while building a different number of views.

    Absent when the tree is not a git checkout, which is a fact about the run
    and is recorded as such rather than guessed at.
    """
    root = Path(__file__).resolve().parent.parent

    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", "-C", str(root), *args],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):  # pragma: no cover
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    head = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")

    # The digest, not the revision, is what `nn.p2b_compare` compares. A commit
    # that touches only documentation moves HEAD without changing a line any
    # cell executes, and refusing to join cells across it would split a batch
    # for no reason; a dirty working tree moves no revision at all while
    # changing everything. Hashing the repository source each process actually
    # imported answers the question both of those get wrong — what code ran.
    sources: dict[str, str] = {}
    for name, module in sorted(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if not path:
            continue
        resolved = Path(path).resolve()
        # `is_file()` guards against a module whose `__file__` is a bare name —
        # a few libraries set one — which resolves against the working directory
        # and lands inside the repository without existing there.
        if root in resolved.parents and resolved.suffix == ".py" and resolved.is_file():
            sources[str(resolved.relative_to(root))] = hashlib.sha256(
                resolved.read_bytes()
            ).hexdigest()
    digest = hashlib.sha256(
        json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    return {
        "revision": head,
        "dirty": None if status is None else bool(status.strip()),
        "source_digest": digest,
        "source_files": len(sources),
        "note": (
            "source_digest covers every repository module this process imported and is "
            "what nn.p2b_compare requires cells to agree on; revision and dirty are "
            "recorded beside it because neither one alone says what code ran"
        ),
    }


def threadpool_record() -> dict[str, Any]:
    """The native thread pools actually in force, for the artifact."""
    return {
        "limit_applied": FIT_THREADS,
        "reason": (
            "BLAS reductions are order-dependent, so a multi-threaded lbfgs fit is not "
            "reproducible across machines with different core counts"
        ),
        "pools": [
            {
                "user_api": pool.get("user_api"),
                "internal_api": pool.get("internal_api"),
                "num_threads": pool.get("num_threads"),
                "version": pool.get("version"),
            }
            for pool in threadpool_info()
        ],
    }


class SnapshotError(SystemExit):
    """The research snapshot cannot support the run that was asked for."""


def load_snapshot(
    manifest_path: Path,
) -> tuple[pd.DataFrame, Any, pd.DataFrame, dict[str, Any]]:
    """Read the committed research snapshot: processed spine, raw candles, manifest.

    Integrity of the files themselves is `tools.verify_research_snapshot`'s job
    and is not duplicated here. What this does check is the one thing specific to
    *running* on the snapshot: that the manifest says it contains no sealed row,
    because every guarantee below rests on the fold plan being expressed in the
    canonical dataset's row numbers while the data present stops short of Styx.
    """
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("contains_styx") is not False:
        raise SnapshotError(
            f"{manifest_path} does not declare contains_styx=false; refusing to run "
            "research against a snapshot that will not say it is Styx-free"
        )
    if manifest.get("safety_checks", {}).get("styx_rows_exported") != 0:
        raise SnapshotError(f"{manifest_path} declares exported sealed rows; refusing to run")

    root = manifest_path.resolve().parent.parent.parent
    spine_path = root / manifest["processed_outer_coverage"]["path"]
    raw_path = root / manifest["raw_pre_styx"]["path"]
    spine, ds_meta = load_dataset(spine_path)
    raw = pd.read_parquet(raw_path)
    return spine, ds_meta, raw, manifest


def plan_from_manifest(
    manifest: dict[str, Any], rows_available: int
) -> tuple[list[FoldPlan], dict[str, int]]:
    """The four folds, planned over the canonical research region.

    ``research_rows`` is the number of rows the canonical dataset had before the
    sealed instant — 48,217 — and is what the fractions are taken of. The
    snapshot holds only the 45,802-row prefix the outer blocks reach, so the
    plan is checked against what is actually present rather than assumed to fit.
    """
    reference = manifest["canonical_reference"]
    research_rows = int(reference["research_rows"])
    sizes = {
        "research_rows": research_rows,
        "min_train": int(research_rows * MIN_TRAIN_FRAC),
        "inner_val_size": int(research_rows * INNER_VAL_FRAC),
        "outer_val_size": int(research_rows * OUTER_VAL_FRAC),
    }
    sizes["step"] = sizes["outer_val_size"]
    folds = plan_nested_folds(
        research_rows,
        FOLDS,
        sizes["min_train"],
        sizes["inner_val_size"],
        sizes["outer_val_size"],
        sizes["step"],
    )
    last_row = folds[-1].outer.end
    if last_row > rows_available:
        raise SnapshotError(
            f"the fold plan reaches row {last_row} but the snapshot holds only "
            f"{rows_available} rows. The snapshot was exported for a different fold "
            "geometry; re-export it rather than shrinking the folds to fit."
        )
    declared = int(reference["max_outer_end_row"])
    if last_row != declared:
        raise SnapshotError(
            f"the fold plan's last outer row is {last_row} but the manifest declares "
            f"max_outer_end_row={declared}. The snapshot and this runner disagree about "
            "which rows research may reach; one of them is wrong and neither may guess."
        )
    return folds, sizes


def aggregate_importance(
    fitted: Any, feature_names: Sequence[str], seq_len: int
) -> dict[str, Any]:
    """Fold a tree model's per-column importance back onto the feature it came from.

    A flattened row holds one column per (timestep, feature) pair, so a raw
    importance vector is 3,392 numbers about 53 features. Summing over the
    ``seq_len`` timesteps — the exact inverse of the ``t * n_features + f``
    layout :data:`nn.simple_models.FLATTEN_ORDER` declares — turns it back into
    one number per feature, which is the granularity the family question is
    asked at.

    Returns ``{}`` for an estimator with no importance to give, rather than
    inventing one. Both libraries' native units are recorded because they are
    not the same quantity: a split count and a summed gain rank features
    differently, and a reader who is not told which one they are looking at
    cannot know which.
    """
    estimator = fitted.estimator
    raw = getattr(estimator, "feature_importances_", None)
    if raw is None:
        return {}
    n_features = len(feature_names)
    values = np.asarray(raw, dtype=np.float64)
    if values.size != seq_len * n_features:
        return {}
    per_feature = values.reshape(seq_len, n_features).sum(axis=0)
    total = float(per_feature.sum())
    share = per_feature / total if total > 0 else np.zeros_like(per_feature)
    return {
        "importance_type": getattr(estimator, "importance_type", "unknown"),
        "aggregated_over": (
            f"summed across the {seq_len} timesteps of each feature, inverting "
            "column j = timestep t * n_features + feature f"
        ),
        "total": round(total, 8),
        "per_feature": {
            name: round(float(v), 8) for name, v in zip(feature_names, per_feature)
        },
        "per_feature_share": {
            name: round(float(v), 8) for name, v in zip(feature_names, share)
        },
    }


def score_baselines(
    control: ResearchData,
    prepared: Any,
    outer: Any,
    seq_len: int,
    baselines: dict[str, Any],
    device: Any,
) -> Any:
    """The rule floors and the economic references, on the control view's columns.

    Goes through :func:`nn.train.score_frozen_split` rather than scoring the
    baselines by hand, so the floors are built by the same windowing, market
    context and report construction the model's own number is. ``proba_of`` is
    required by that signature and is handed the momentum baseline at the
    baseline threshold, which recomputes ``momentum_baseline`` with identical
    arguments and overwrites it with itself — the cheapest way to satisfy the
    signature without introducing a second scoring path.
    """
    momentum = baselines["momentum_baseline"]
    return score_frozen_split(
        control,
        prepared.scaler,
        outer,
        seq_len,
        baselines=baselines,
        threshold=BASELINE_DECISION_THRESHOLD,
        device=device,
        proba_of=momentum.predict_proba,
        model_name="momentum_baseline",
    )


def run_cell(
    aligned: AlignedResearchSamples,
    set_name: str,
    spec: SimpleModelSpec,
    folds: Sequence[FoldPlan],
    *,
    seed: int,
    seq_len: int,
    min_trades: int,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Fit on train, select the threshold on inner, measure once on outer, per fold.

    One information set, one model, four folds. The scaler, the windows and the
    frozen scoring all come from :mod:`nn.train` — the same three functions P2a
    and the MTST walk-forward go through — so a difference between two cells in
    this report cannot be a difference in how the samples were built.
    """
    data: ResearchData = aligned.views[set_name]
    # The rule floors are scored on the control view, never on the view under
    # test. `MomentumBaseline` reads `ema_cross`, which the `smc_v1` arm does not
    # contain — but that is the smaller half of the reason. The larger one is
    # that a floor which changed with the information set would stop being a
    # floor: the baselines and the economic references are the data-level proof
    # that nine separately-run cells scored the same rows, and that proof only
    # works while every cell computes them from the same columns.
    control: ResearchData = aligned.views[CONTROL_SET]
    device = resolve_device("cpu")
    grid = threshold_grid()

    records: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []

    # Build one throwaway estimator first. `threadpool_limits` can only limit
    # pools that are already loaded when it is entered, and `spec.build` imports
    # the model's library lazily — so entering the limit and *then* fitting left
    # scipy's OpenBLAS and scikit-learn's libgomp at four threads on fold 0 and
    # pinned on folds 1-3, because by then the import had happened. Fold 0 ran
    # under a thread configuration no other fold used, and the pool snapshot
    # recorded in the artifact named neither library.
    spec.build(seed)

    with threadpool_limits(limits=FIT_THREADS):
        threads = threadpool_record()
        for fold, plan in enumerate(folds):
            fold_seed = seed + fold
            prepared = prepare_research_windows(data, plan.train, plan.inner, seq_len)
            control_prepared = prepare_research_windows(
                control, plan.train, plan.inner, seq_len
            )
            baselines = fit_baselines(control, control_prepared.y_train)
            inner_return = data.future_return[prepared.idx_val]

            logger.info(
                "fold %d: fitting %s on %s (%d x %d = %d inputs, seed %d)",
                fold,
                spec.name,
                set_name,
                seq_len,
                len(data.feature_names),
                seq_len * len(data.feature_names),
                fold_seed,
            )
            fitted = fit_simple_model(spec, prepared.X_train, prepared.y_train, seed=fold_seed)
            inner_proba = fitted.predict_proba(prepared.X_val)
            threshold, threshold_report = ev.select_threshold(
                inner_proba,
                inner_return,
                data.target_spec,
                grid=grid,
                min_trades=min_trades,
                row_index=prepared.idx_val,
            )

            # Frozen from here. `score_frozen_split` fits nothing, and `plan.outer`
            # reaches this call and no other in the whole module.
            scored = score_frozen_split(
                data,
                prepared.scaler,
                plan.outer,
                seq_len,
                baselines={},
                threshold=threshold,
                device=device,
                proba_of=fitted.predict_proba,
                model_name=spec.name,
            )
            floors = score_baselines(
                control, control_prepared, plan.outer, seq_len, baselines, device
            )

            outer = {spec.name: scored.reports[spec.name]}
            outer.update({name: floors.reports[name] for name in BASELINE_NAMES})
            outer[REFERENCES_KEY] = floors.economic_references
            if not np.array_equal(floors.idx, scored.idx):
                raise AssertionError(
                    f"fold {fold}: the rule floors scored {len(floors.idx)} rows and "
                    f"{spec.name} scored {len(scored.idx)}; the control view and the "
                    f"{set_name} view are not selecting the same samples"
                )

            records.append(
                {
                    "fold": fold,
                    "run_seed": seed,
                    "fold_seed": fold_seed,
                    "samples": {
                        "train": len(prepared.X_train),
                        "inner_validation": len(prepared.X_val),
                        "outer_validation": len(scored.idx),
                    },
                    "periods": {
                        "train": data.period(plan.train),
                        "inner_validation": data.period(plan.inner),
                        "outer_validation": data.period(plan.outer),
                    },
                    "model": {
                        "provenance": {**fitted.provenance(), "threading": threads},
                        "importance": aggregate_importance(
                            fitted, data.feature_names, seq_len
                        ),
                        "selection": {
                            "block": "inner_validation",
                            "threshold": threshold,
                            "objective": THRESHOLD_OBJECTIVE,
                            "threshold_grid": [float(v) for v in grid],
                            "min_trades": min_trades,
                            "inner_validation_trading": threshold_report,
                        },
                    },
                    "outer_validation": outer,
                }
            )
            frames.append(
                outer_predictions(spec.name, fold, fold_seed, data, scored, threshold)
            )

    predictions = pd.concat(frames, ignore_index=True)
    predictions.insert(0, "information_set", set_name)
    return records, predictions


def fold_spread(values: list[float | None]) -> dict[str, Any]:
    """:func:`nn.walkforward.spread` plus the order statistics P2b reports.

    Extends rather than replaces it: `mean`, `std`, `values` and `defined_folds`
    keep the meaning every other artifact in this repository gives them —
    including that a ``None`` is an undefined measurement and not a zero — and
    min, max and median are added because four folds are few enough that the
    range matters more than the standard deviation.
    """
    stats = spread(values)
    defined = [v for v in values if v is not None]
    stats["min"] = round(min(defined), 6) if defined else None
    stats["max"] = round(max(defined), 6) if defined else None
    stats["median"] = round(statistics.median(defined), 6) if defined else None
    return stats


def summarise(records: list[dict[str, Any]], model: str) -> dict[str, Any]:
    """Spread of each outer metric across the four temporal folds.

    Reads ``outer_validation`` only, for the reason
    :func:`nn.benchmark.summarise` reads only that block: the inner block chose
    the threshold, so aggregating it would report a selection score as a result.

    ``positive_net_return_folds`` is out of four **temporal periods**. There is
    no seed dimension in P2b and there is not meant to be one: these estimators
    are deterministic given their inputs, so a second seed would produce a
    second copy of the same evidence, not a second observation.
    """
    summary: dict[str, Any] = {
        "folds": len(records),
        "aggregated_from": "outer_validation",
        "statistical_unit": "one temporal outer period per fold; no seed replication",
        "per_model": {},
    }
    for name in (model, *BASELINE_NAMES):
        stats: dict[str, Any] = {}
        for metric, (section, key) in SUMMARY_METRICS.items():
            raw = [r["outer_validation"][name][section][key] for r in records]
            stats[metric] = fold_spread([None if v is None else float(v) for v in raw])
        stats["positive_net_return_folds"] = sum(
            1 for v in stats["net_return"]["values"] if v > 0
        )
        summary["per_model"][name] = stats
    summary["thresholds"] = [r["model"]["selection"]["threshold"] for r in records]
    return summary


def parity_record(data: ResearchData, seq_len: int) -> dict[str, Any]:
    """How a row was built, written down so the parity claim can be checked."""
    columns = flattened_column_names(data.feature_names, seq_len)
    return {
        "seq_len": seq_len,
        "n_features": len(data.feature_names),
        "feature_names": list(data.feature_names),
        "flattened_input_dim": len(columns),
        "flatten_order": FLATTEN_ORDER,
        "sample_construction": (
            "nn.dataset.build_windows, reached through nn.train.prepare_research_windows "
            "(train and inner validation) and nn.train.score_frozen_split (outer "
            "validation) — the same calls nn.benchmark makes for P2a"
        ),
        "scaling": (
            "nn.dataset.StandardScaler fitted on the fold's training rows only and "
            "applied unchanged to inner and outer validation"
        ),
        "feature_selection": "none",
        "fitted_on_validation": "nothing",
    }


def write_predictions(out_dir: Path, predictions: pd.DataFrame, boundary: int) -> Path:
    """Persist per-sample outer predictions, refusing to write a sealed row."""
    highest = int(predictions["row_index"].max())
    if highest >= boundary:
        raise AssertionError(
            f"refusing to write outer predictions: row {highest} is at or beyond the "
            f"sealed test block starting at {boundary}"
        )
    path = out_dir / PREDICTIONS_NAME
    predictions.to_parquet(path, index=False)
    return path


def to_markdown(payload: dict[str, Any]) -> str:
    """A short per-cell summary. The cross-cell comparison is a separate tool."""
    parity = payload["information_parity"]
    model = payload["model"]
    lines = [
        f"# P2b cell — {model} on `{payload['information_set']}`",
        "",
        f"One information set, one model, {payload['summary']['folds']} temporal folds.",
        f"Each row is a `{parity['seq_len']} x {parity['n_features']}` window flattened to",
        f"`{parity['flattened_input_dim']}` values. Nothing is tuned.",
        "",
        f"**Research contract:** `{payload['contract']['contract_id']}`, semantic identity",
        f"`sha256:{payload['contract']['contract_hash']}`.",
        "",
        f"**Sealed test block:** everything at or after "
        f"`{payload['contract']['sealed_test_start']}`. Not planned over, not fitted on,",
        "not selected on, not scored. `sealed_test: false`.",
        "",
        f"**Feature spec:** smc `{payload['feature_spec']['smc_spec_version']}` "
        f"`sha256:{payload['feature_spec']['smc_spec_hash'][:16]}...`",
        "",
        "## Per-fold outer validation",
        "",
        "| fold | outer period | samples | threshold | trades | net return | ann. Sharpe "
        "| max DD | exposure | macro F1 | dir. acc |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for record in payload["folds"]:
        outer = record["outer_validation"][model]
        trade, cls = outer["trading"], outer["classification"]
        period = record["periods"]["outer_validation"]
        lines.append(
            f"| {record['fold']} | {period['start'][:10]} → {period['end'][:10]} "
            f"| {record['samples']['outer_validation']} "
            f"| {record['model']['selection']['threshold']:.2f} "
            f"| {trade['n_trades']} | {trade['net_return']:+.6f} "
            f"| {ev.number(trade['annualised_sharpe'], '.4f')} "
            f"| {trade['max_drawdown']:.4f} | {trade['exposure']:.4f} "
            f"| {cls['macro_f1']:.4f} | {cls['directional_accuracy']:.4f} |"
        )
    stats = payload["summary"]["per_model"][model]
    net = stats["net_return"]
    lines += [
        "",
        "## Across the four temporal folds",
        "",
        f"- net return: mean {net['mean']:+.6f}, std {ev.number(net['std'], '.6f')}, "
        f"min {net['min']:+.6f}, median {net['median']:+.6f}, max {net['max']:+.6f}",
        f"- positive in **{stats['positive_net_return_folds']} of "
        f"{payload['summary']['folds']}** temporal folds",
        "",
        "The statistical unit is the temporal period. These estimators are deterministic,",
        "so repeating the run under another seed would copy this evidence, not add to it.",
        "",
    ]
    return "\n".join(lines)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--information-set",
        # The six ablation arms are accepted here too. They are not a fourth
        # information set: each is the combined arm with one declared family
        # removed, and what they produce is post-hoc diagnostic evidence rather
        # than canonical P2b evidence.
        choices=list(P2B_INFORMATION_SETS) + ablation_set_names(),
        required=True,
    )
    parser.add_argument("--model", choices=list(SIMPLE_MODEL_NAMES), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--research-contract", default="btc-usdt-1h-gen1")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--min-trades", type=int, default=MIN_TRADES)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = build_argparser().parse_args(argv)

    spine, ds_meta, raw, manifest = load_snapshot(args.manifest)
    contract: ResearchContract = load_contract(args.research_contract)
    try:
        contract.require_scope(
            exchange=ds_meta.exchange,
            pair=ds_meta.pair,
            timeframe=ds_meta.timeframe,
            source="the research snapshot",
        )
    except ContractScopeError as exc:
        raise SystemExit(str(exc)) from exc
    if manifest["contract"]["contract_hash"] != contract.contract_hash:
        raise SnapshotError(
            f"the snapshot was exported under contract hash "
            f"{manifest['contract']['contract_hash']} but {args.research_contract} hashes "
            f"to {contract.contract_hash}"
        )

    folds, sizes = plan_from_manifest(manifest, len(spine))
    names = tuple(dict.fromkeys((CONTROL_SET, args.information_set)))
    aligned = build_information_set_views(spine, ds_meta, raw, names=names)
    alignment = aligned.prove_alignment(folds, args.seq_len)

    spec = next(s for s in SIMPLE_MODELS if s.name == args.model)
    logger.warning(
        "P2b %s x %s. %d folds over rows [0, %d) of the canonical research region; the "
        "snapshot holds rows [0, %d). Outer blocks: %s. The sealed block starts at %s "
        "and is NOT planned over, trained on, selected on, or evaluated.",
        args.information_set,
        args.model,
        len(folds),
        sizes["research_rows"],
        len(spine),
        ", ".join(f"[{p.outer.start}, {p.outer.end})" for p in folds),
        contract.sealed_test_start.isoformat(),
    )

    records, predictions = run_cell(
        aligned,
        args.information_set,
        spec,
        folds,
        seed=args.seed,
        seq_len=args.seq_len,
        min_trades=args.min_trades,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = write_predictions(out_dir, predictions, sizes["research_rows"])

    payload = {
        "checkpoint": CHECKPOINT,
        "purpose": (
            "does causal market structure add information beyond OHLCV14? One cell: one "
            "information set, one model, four temporal folds"
        ),
        "information_set": args.information_set,
        "model": args.model,
        "dataset": ds_meta.to_dict(),
        "snapshot": {
            "manifest": str(args.manifest),
            "snapshot_schema": manifest["snapshot_schema"],
            "raw_semantic_hash": manifest["raw_pre_styx"]["semantic_hash"],
            "processed_semantic_prefix_hash": manifest["processed_outer_coverage"][
                "semantic_prefix_hash"
            ],
            "rows": len(spine),
            "contains_styx": manifest["contains_styx"],
        },
        "contract": {
            "contract_id": contract.contract_id,
            "contract_hash": contract.contract_hash,
            "research_generation": contract.research_generation,
            "sealed_test_start": contract.sealed_test_start.isoformat(),
        },
        "feature_spec": feature_spec_identity(aligned.smc_spec, ds_meta),
        "code": code_revision(),
        "config": {
            "seed": args.seed,
            "seq_len": args.seq_len,
            "min_trades": args.min_trades,
            "folds": FOLDS,
        },
        "sizes": sizes,
        "sealed_test": False,
        "reported_block": "outer_validation",
        "outer_predictions": PREDICTIONS_NAME,
        "alignment": alignment,
        "information_parity": parity_record(aligned.views[args.information_set], args.seq_len),
        "target": ds_meta.target_spec,
        "threshold_selection": {
            "block": "inner_validation",
            "objective": THRESHOLD_OBJECTIVE,
            "grid": [float(v) for v in threshold_grid()],
            "min_trades": args.min_trades,
            "applied_to_outer_validation": "the already-selected value, unchanged",
        },
        "tuning": "none (predeclared fixed configurations, reused from P2a)",
        "folds": records,
        "summary": summarise(records, args.model),
    }
    (out_dir / ARTIFACT_NAME).write_text(json.dumps(payload, indent=2, default=str) + "\n")
    (out_dir / MARKDOWN_NAME.replace("benchmark", "p2b")).write_text(to_markdown(payload))
    logger.info(
        "Wrote %d outer predictions to %s and the artifact to %s",
        len(predictions),
        predictions_path,
        out_dir / ARTIFACT_NAME,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
