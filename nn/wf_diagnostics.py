"""Diagnostics across completed walk-forward artifact directories.

    python -m nn.wf_diagnostics artifacts/walkforward/btc_nested_v1 \
        artifacts/walkforward/btc_nested_seed_142

Reads ``walkforward.json`` files that ``nn.walkforward`` has already written and
answers two questions the individual runs cannot answer about themselves:

1. **Is each artifact internally sound?** The invariants walk-forward asserts
   while running are re-checked here against what actually landed on disk — the
   three regions in order, outer blocks disjoint, nothing at or beyond
   ``sealed_test_start``, the sealed block unopened, and the recorded summary
   reproducible from the folds it claims to summarise. An artifact that has been
   hand-edited, truncated, or produced by the pre-nested code fails here rather
   than being quietly averaged into a headline.

2. **How much of the result is the seed?** One run of four folds gives four
   numbers. Several runs that differ only in ``--seed`` give a distribution, and
   the useful question is how wide it is. The report shows each run's
   across-fold mean, the spread of those means across runs, the per-fold spread
   across runs, and how stable the *selected* threshold and early-stopping epoch
   were — a fold whose threshold jumps around between seeds is a fold whose
   inner block is not deciding much.

3. **What was different about the market?** Given ``--dataset`` (and optionally
   ``--raw``), it opens the processed dataset the artifacts' row indices address
   and summarises each outer block's regime — returns, volatility, trend
   orientation, volume behaviour and target mix — then ranks what differs
   between the best and worst fold. Everything model-facing is computed over the
   rows the model was actually **scored** on, not the rows the block spans:
   ``nn.regime`` selects them through the same index logic the run used.

Everything is read from paths given on the command line. No artifact, dataset or
model is expected to live in the repository. Without ``--dataset`` nothing opens
a dataset at all, so artifacts stay analysable on their own; with it, the
dataset is truncated at the sealed boundary before a single statistic is taken,
and a file that disagrees with the artifact's recorded identity is refused
rather than reindexed. No checkpoint is ever loaded.

**Reading the output.** These are outer-validation numbers — blocks nothing was
fitted on, which is what makes them worth comparing. Seed spread measures the
stability of the research procedure, not out-of-sample performance, and it is
not a claim of profitability. Differences between folds are coincidences in the
data, never causes. The sealed test block stays unopened; this tool refuses to
read an artifact that says otherwise.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from chimera.contracts import CLASS_ORDER
from nn.regime import (
    RegimeDataError,
    block_statistics,
    direction_attribution,
    load_predictions,
    load_raw_ohlcv,
    load_research_frame,
    raw_block_statistics,
)
from nn.walkforward import MODELS, PREDICTIONS_NAME, REFERENCES_KEY, SUMMARY_METRICS

logger = logging.getLogger(__name__)

#: The file `nn.walkforward` writes, looked up when a directory is given.
ARTIFACT_NAME = "walkforward.json"

#: What this tool writes into ``--out``.
REPORT_JSON = "regime_diagnostics.json"
REPORT_MD = "regime_diagnostics.md"

#: Dataset identity fields that must match for runs to be one seed study.
#:
#: A different feature contract, target spec, horizon or cost model makes two
#: runs measurements of different systems, however similar the numbers look.
#: `created_at` and `class_balance` are deliberately absent: the first differs
#: for every build, the second is a summary of the same rows.
IDENTITY_FIELDS = (
    "rows",
    "pair",
    "timeframe",
    "exchange",
    "start",
    "end",
    "feature_names",
    "feature_spec",
    "target_spec",
)

#: The three regions of a fold, in the order they must appear in time.
REGIONS = ("train", "inner_validation", "outer_validation")

#: Target class names, in the dataset's own order.
_CLASS_LABELS = [c.value for c in CLASS_ORDER]

#: Baselines whose outer reports cannot legitimately vary across seed-only runs.
#:
#: The majority rule is fitted on training rows and the momentum rule is fitted
#: on nothing, so on identical geometry and data their numbers are a constant.
#: Spread here is therefore not a finding about the market — it is evidence that
#: the artifacts predate the fix that stopped baselines being scored at the
#: model's own selected threshold.
DETERMINISTIC_BASELINES = ("majority_baseline", "momentum_baseline")

#: Metrics that only artifacts produced after the metric correction carry.
#:
#: Both are computed from a candle-level portfolio return series, which earlier
#: runs did not build.
CURRENT_ONLY_METRICS = ("annualised_sharpe", "per_trade_sharpe")

#: Metrics present in both generations but **computed differently**.
#:
#: ``max_drawdown`` used to take its running peak from the first completed
#: trade rather than from starting capital, so a strategy that was under water
#: from its first trade understated its drawdown — a single 10% loss reported
#: zero. The field name is unchanged because the old number was simply wrong,
#: not a different measurement worth keeping; but a pre-correction value is not
#: comparable to a corrected one, so it is skipped for legacy artifacts exactly
#: as the missing fields are, and the report says which kind of skip each was.
REDEFINED_METRICS = ("max_drawdown",)

#: The field the correction removed, and what it used to mean.
#:
#: ``sharpe`` annualised per-trade statistics by ``candles_per_year / horizon``
#: — the number of trades the strategy *could* have taken back to back — so on
#: a signal that trades rarely it was overstated several-fold, and it grew as
#: the target horizon shrank. Its presence dates an artifact; its values are not
#: comparable to anything this tool now reports.
LEGACY_TRADING_KEY = "sharpe"

#: Metric schemas this tool knows how to read.
SCHEMA_CURRENT = "current"
SCHEMA_LEGACY = "legacy_pre_metric_correction"


def classify_schema(run: "RunArtifact") -> tuple[str, list[str]]:
    """Which metric schema an artifact carries, or why it is not readable.

    Returns ``(schema, problems)``. **Tolerance here is narrow on purpose.** The
    obvious implementation — aggregate whichever metrics happen to be present
    everywhere — would let a *current* artifact that lost a field silently pass
    by dropping it, which is the failure this whole module exists to catch. So
    only two shapes are readable:

    * every required metric present and no ``sharpe`` — current, read in full;
    * ``sharpe`` present, **both** :data:`CURRENT_ONLY_METRICS` absent, and every
      other required metric present — a positively identified pre-correction
      artifact, read with exactly those two metrics skipped and a warning.

    Anything else — a missing metric with no legacy fingerprint, a mixture of
    old and new fields, a partial rename — is a problem, not a degraded read.
    """
    required = set(SUMMARY_METRICS)
    optional = set(CURRENT_ONLY_METRICS)
    problems: list[str] = []
    schemas: set[str] = set()

    for position, fold in enumerate(run.folds):
        label = fold.get("fold", position)
        for model in MODELS:
            report = (fold.get("outer_validation") or {}).get(model)
            if not isinstance(report, dict):
                problems.append(f"fold {label} has no {model} report")
                continue
            present = {
                metric
                for metric, (section, key) in SUMMARY_METRICS.items()
                if isinstance(report.get(section), dict) and key in report[section]
            }
            legacy = LEGACY_TRADING_KEY in (report.get("trading") or {})
            missing = required - present

            if not missing and not legacy:
                schemas.add(SCHEMA_CURRENT)
            elif legacy and missing == optional:
                schemas.add(SCHEMA_LEGACY)
            elif legacy:
                problems.append(
                    f"fold {label} {model} carries the pre-correction "
                    f"{LEGACY_TRADING_KEY!r} field alongside {sorted(present & optional)} "
                    f"and is missing {sorted(missing - optional)}: this is neither the "
                    "old schema nor the new one, so its metric semantics are unknown"
                )
            else:
                problems.append(
                    f"fold {label} {model} is missing {sorted(missing)} and carries no "
                    f"{LEGACY_TRADING_KEY!r} field to date it as pre-correction. A "
                    "current artifact must carry every reported metric; refusing to "
                    "aggregate around the gap"
                )

    if len(schemas) > 1:
        problems.append(
            f"folds disagree about the metric schema ({sorted(schemas)}): some carry "
            "the pre-correction fields and some the corrected ones, so no single set "
            "of semantics describes this artifact"
        )
    if problems:
        return "", problems
    return (schemas.pop() if schemas else SCHEMA_CURRENT), []


def readable_metrics(schema: str) -> tuple[str, ...]:
    """The metrics that may be aggregated under a given schema.

    A pre-correction artifact loses two kinds of metric: the ones it never
    recorded (:data:`CURRENT_ONLY_METRICS`) and the ones it recorded under
    different semantics (:data:`REDEFINED_METRICS`). Both are unusable, for
    different reasons, and the report distinguishes them.
    """
    if schema == SCHEMA_LEGACY:
        skipped = set(CURRENT_ONLY_METRICS) | set(REDEFINED_METRICS)
        return tuple(m for m in SUMMARY_METRICS if m not in skipped)
    return tuple(SUMMARY_METRICS)


@dataclass(frozen=True)
class RunArtifact:
    """One ``walkforward.json``, loaded and given a name to report under."""

    name: str
    path: Path
    seed: Any
    sealed_test_start: int
    dataset: dict[str, Any]
    folds: list[dict[str, Any]]
    summary: dict[str, Any]
    raw: dict[str, Any]

    @property
    def n_folds(self) -> int:
        return len(self.folds)

    @property
    def geometry(self) -> list[tuple[int, int]]:
        """Every region's row range, flattened, as the run's comparable shape."""
        return [self.rows(fold, region) for fold in self.folds for region in REGIONS]

    @staticmethod
    def rows(fold: dict[str, Any], region: str) -> tuple[int, int]:
        start, end = fold["periods"][region]["row_range"]
        return int(start), int(end)

    def metric(self, fold: dict[str, Any], model: str, metric: str) -> float | None:
        """One metric from one fold, or ``None`` where it is undefined.

        Undefined is a real state for the risk-adjusted statistics — a portfolio
        that never moved has no Sharpe — and it is carried through rather than
        coerced, so no aggregate averages a zero that was never measured.
        """
        section, key = SUMMARY_METRICS[metric]
        value = fold["outer_validation"][model][section][key]
        return None if value is None else float(value)


def load_run(path: str | Path) -> RunArtifact:
    """Load one artifact from a directory or a ``walkforward.json`` path.

    Fails with a message naming the file rather than returning a partial run:
    a diagnostics tool that skips what it cannot read produces a summary of
    whatever happened to parse, which is the failure mode it exists to catch.
    """
    given = Path(path)
    # A path ending in .json is the artifact; anything else is the directory it
    # lives in. Deciding by suffix rather than by is_dir() means a path that does
    # not exist still reports the file it was looking for.
    points_at_file = given.suffix == ".json"
    artifact = given if points_at_file else given / ARTIFACT_NAME
    if not artifact.is_file():
        raise SystemExit(
            f"no walk-forward artifact at {artifact}. Give the directory "
            f"nn.walkforward wrote (containing {ARTIFACT_NAME}) or the JSON file itself."
        )

    try:
        payload = json.loads(artifact.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{artifact} is not readable JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"{artifact} is not an object")

    folds = payload.get("folds")
    if not isinstance(folds, list) or not folds:
        raise SystemExit(f"{artifact} has no folds to report on")

    for position, fold in enumerate(folds):
        if not isinstance(fold, dict):
            raise SystemExit(f"{artifact} fold at position {position} is not an object")
        if "outer_validation" not in fold:
            legacy = (
                " It carries a single 'validation' block per fold: this is a "
                "pre-nested artifact, whose reported block was also the block the "
                "threshold and early-stopping epoch were chosen on. Re-run "
                "nn.walkforward to produce a comparable result."
                if "validation" in fold
                else ""
            )
            raise SystemExit(
                f"{artifact} fold {fold.get('fold', position)} has no 'outer_validation' "
                f"block.{legacy}"
            )
        missing = [region for region in REGIONS if region not in fold.get("periods", {})]
        if missing:
            raise SystemExit(
                f"{artifact} fold {fold.get('fold', position)} is missing the "
                f"{', '.join(missing)} period(s)"
            )

    sealed = payload.get("sealed_test")
    if not isinstance(sealed, dict) or "start_row" not in sealed:
        raise SystemExit(f"{artifact} has no sealed_test.start_row to check folds against")

    return RunArtifact(
        name=(given.parent if points_at_file else given).name or str(given),
        path=artifact,
        seed=payload.get("config", {}).get("seed"),
        sealed_test_start=int(sealed["start_row"]),
        dataset=payload.get("dataset", {}) or {},
        folds=folds,
        summary=payload.get("summary", {}) or {},
        raw=payload,
    )


def audit_run(run: RunArtifact) -> list[str]:
    """Re-check one artifact's leakage and consistency invariants on its own rows.

    Returns the problems found, empty when the artifact is sound. The checks are
    on row indices rather than region names, for the same reason walk-forward's
    own tests are: a block named ``outer_validation`` is not evidence that it sits
    where an outer block belongs.
    """
    problems: list[str] = []
    boundary = run.sealed_test_start

    # Fail closed on the sealing flags: absent is not "false".
    if run.raw.get("test_evaluated") is not False:
        problems.append(f"test_evaluated is {run.raw.get('test_evaluated')!r}, expected false")
    if run.raw.get("sealed_test", {}).get("evaluated") is not False:
        problems.append(
            f"sealed_test.evaluated is "
            f"{run.raw.get('sealed_test', {}).get('evaluated')!r}, expected false"
        )
    if run.raw.get("reported_block") != "outer_validation":
        problems.append(
            f"reported_block is {run.raw.get('reported_block')!r}, expected "
            "'outer_validation'"
        )
    if run.summary.get("aggregated_from") != "outer_validation":
        problems.append(
            f"summary.aggregated_from is {run.summary.get('aggregated_from')!r}, "
            "expected 'outer_validation'"
        )
    if run.summary.get("folds") not in (None, run.n_folds):
        problems.append(
            f"summary.folds is {run.summary.get('folds')} but the artifact carries "
            f"{run.n_folds} folds"
        )

    reported_by: dict[int, tuple[int, Any]] = {}
    for position, fold in enumerate(run.folds):
        label = fold.get("fold", position)
        train, inner, outer = (RunArtifact.rows(fold, region) for region in REGIONS)

        for region, (start, end) in zip(REGIONS, (train, inner, outer)):
            if end <= start:
                problems.append(f"fold {label} {region} is empty: rows [{start}, {end})")
            if end > boundary:
                problems.append(
                    f"fold {label} {region} ends at row {end}, at or beyond the sealed "
                    f"test block which starts at {boundary}"
                )
        if train[1] > inner[0]:
            problems.append(
                f"fold {label} train ends at {train[1]} but inner validation starts at "
                f"{inner[0]}: training overlaps the block it is selected on"
            )
        if inner[1] > outer[0]:
            problems.append(
                f"fold {label} inner validation ends at {inner[1]} but outer validation "
                f"starts at {outer[0]}: the reported block was selected on"
            )

        # Identity is the fold's position, not its recorded label: two folds
        # labelled the same would otherwise hide an overlap from this check.
        for row in range(*outer):
            earlier = reported_by.setdefault(row, (position, label))
            if earlier[0] != position:
                problems.append(
                    f"fold {label} outer validation overlaps fold {earlier[1]}: row "
                    f"{row} is reported by both"
                )
                break

    problems.extend(classify_schema(run)[1])
    problems.extend(_audit_summary_matches_folds(run))
    return problems


def _audit_summary_matches_folds(run: RunArtifact) -> list[str]:
    """The recorded across-fold means must be reproducible from the folds.

    Proves the headline numbers come from the per-fold reports in the same file,
    rather than from an edit somewhere between the run and the reader.
    """
    problems: list[str] = []
    per_model = run.summary.get("per_model")
    if not isinstance(per_model, dict):
        return problems

    schema, unreadable = classify_schema(run)
    if unreadable:
        # The schema check has already reported why; re-deriving means from an
        # artifact whose fields are not understood would add noise, not evidence.
        return problems

    for model in MODELS:
        recorded_model = per_model.get(model)
        if not isinstance(recorded_model, dict):
            continue
        for metric in readable_metrics(schema):
            recorded = recorded_model.get(metric, {}).get("mean")
            if recorded is None:
                continue
            try:
                values = [run.metric(fold, model, metric) for fold in run.folds]
            except (KeyError, TypeError, ValueError):
                problems.append(
                    f"summary reports {model}.{metric} but the folds do not carry it"
                )
                continue
            mean = _mean_defined(values)
            if mean is None:
                problems.append(
                    f"summary {model}.{metric} mean is {recorded}, but no fold defines it"
                )
                continue
            recomputed = round(mean, 6)
            if abs(recomputed - float(recorded)) > 1e-6:
                problems.append(
                    f"summary {model}.{metric} mean is {recorded}, but the folds average "
                    f"{recomputed}"
                )
    return problems


def compare_runs(runs: list[RunArtifact]) -> list[str]:
    """Check the runs measure the same blocks of the same dataset.

    Averaging metrics from runs whose outer blocks are different rows is not a
    seed study, it is a category error, so a mismatch is reported and the
    aggregate is withheld rather than computed anyway.
    """
    problems: list[str] = []
    reference = runs[0]
    reference_schema = classify_schema(reference)[0]
    for run in runs[1:]:
        schema = classify_schema(run)[0]
        if schema and reference_schema and schema != reference_schema:
            problems.append(
                f"{run.name} carries the {schema!r} metric schema but {reference.name} "
                f"carries {reference_schema!r}. The two annualise risk-adjusted "
                "statistics differently, so averaging them would compare unlike "
                "definitions under one name"
            )
        if run.sealed_test_start != reference.sealed_test_start:
            problems.append(
                f"{run.name} has sealed_test_start {run.sealed_test_start}, "
                f"{reference.name} has {reference.sealed_test_start}"
            )
        if run.n_folds != reference.n_folds:
            problems.append(
                f"{run.name} has {run.n_folds} folds, {reference.name} has "
                f"{reference.n_folds}"
            )
        elif run.geometry != reference.geometry:
            differing = [
                f"fold {fold} {region}"
                for fold in range(run.n_folds)
                for position, region in enumerate(REGIONS)
                if run.geometry[fold * len(REGIONS) + position]
                != reference.geometry[fold * len(REGIONS) + position]
            ]
            problems.append(
                f"{run.name} does not share {reference.name}'s fold geometry "
                f"({', '.join(differing)} differ)"
            )
        for key in IDENTITY_FIELDS:
            if run.dataset.get(key) != reference.dataset.get(key):
                problems.append(
                    f"{run.name} dataset {key} is {run.dataset.get(key)!r}, "
                    f"{reference.name} has {reference.dataset.get(key)!r}"
                )
    return problems


def _spread(values: list[float | None]) -> dict[str, Any]:
    """Spread over the observations where the metric is defined.

    A risk-adjusted statistic is undefined when the portfolio never moved. That
    is reported as ``None`` rather than ``0.0`` so it cannot be read as a
    measurement, which means every aggregation has to carry it through instead
    of averaging in a zero that was never observed.
    """
    defined = [v for v in values if v is not None]
    if not defined:
        return {"mean": None, "std": None, "min": None, "max": None, "defined": 0}
    return {
        "mean": round(statistics.fmean(defined), 6),
        # Sample standard deviation needs two observations; one run has none.
        "std": round(statistics.stdev(defined), 6) if len(defined) > 1 else 0.0,
        "min": round(min(defined), 6),
        "max": round(max(defined), 6),
        "defined": len(defined),
    }


def aggregate(runs: list[RunArtifact]) -> dict[str, Any]:
    """Seed stability: how far each outer-validation number moves across runs.

    ``compare_runs`` has already refused a mixed set by the time this is called,
    so one schema describes every run. Under the pre-correction schema exactly
    the two candle-level statistics are skipped and named in ``skipped_metrics``
    — never dropped quietly, and never any other field.
    """
    names = [run.name for run in runs]
    n_folds = runs[0].n_folds
    schema = classify_schema(runs[0])[0]
    readable = readable_metrics(schema)

    per_model: dict[str, Any] = {}
    for model in MODELS:
        metrics: dict[str, Any] = {}
        for metric in readable:
            per_run = {
                run.name: [run.metric(fold, model, metric) for fold in run.folds]
                for run in runs
            }
            run_means = [_mean_defined(per_run[name]) for name in names]
            metrics[metric] = {
                # Each run's across-fold mean, then the spread of those means.
                "per_run_mean": {
                    name: None if mean is None else round(mean, 6)
                    for name, mean in zip(names, run_means)
                },
                "across_runs": _spread(run_means),
                # And the same metric fold by fold, across runs.
                "per_fold": [
                    {
                        "fold": fold,
                        "values": {
                            name: None if (v := per_run[name][fold]) is None else round(v, 6)
                            for name in names
                        },
                        **_spread([per_run[name][fold] for name in names]),
                    }
                    for fold in range(n_folds)
                ],
            }
        per_model[model] = metrics

    return {
        "runs": names,
        "folds": n_folds,
        "metric_schema": schema,
        "skipped_metrics": sorted(set(SUMMARY_METRICS) - set(readable)),
        # Why each was skipped, because "absent" and "recorded under different
        # semantics" are different problems with different remedies.
        "skipped_because_absent": sorted(set(CURRENT_ONLY_METRICS) - set(readable)),
        "skipped_because_redefined": sorted(set(REDEFINED_METRICS) - set(readable)),
        "per_model": per_model,
        "selection": _aggregate_selection(runs, names, n_folds),
        "baseline_comparison": _compare_to_baselines(runs, n_folds),
        "economic_comparison": _compare_to_references(runs),
    }


def _mean_defined(values: list[float | None]) -> float | None:
    """Mean over the folds where the metric is defined; ``None`` when none are."""
    defined = [v for v in values if v is not None]
    return statistics.fmean(defined) if defined else None


def _aggregate_selection(
    runs: list[RunArtifact], names: list[str], n_folds: int
) -> list[dict[str, Any]]:
    """Threshold and early-stopping epoch per fold, across runs.

    Chosen on the inner block, so they are selection diagnostics rather than
    results: a fold whose threshold swings between seeds is a fold whose inner
    block did not have a strong preference.
    """
    selection = []
    for fold in range(n_folds):
        entry: dict[str, Any] = {"fold": fold}
        for field in ("threshold", "best_epoch"):
            values = [float(run.folds[fold]["selection"][field]) for run in runs]
            entry[field] = {
                "values": {name: value for name, value in zip(names, values)},
                **_spread(values),
            }
        selection.append(entry)
    return selection


def _compare_to_baselines(runs: list[RunArtifact], n_folds: int) -> dict[str, Any]:
    """How often MTST beat both baselines, counted over every run and fold.

    A model that wins in one run's majority of folds and loses in the next has
    not been shown to work; counting run-folds is what makes that visible.
    """
    per_run: dict[str, int] = {}
    for run in runs:
        wins = 0
        for fold in run.folds:
            reports = fold["outer_validation"]
            mtst = float(reports["mtst"]["trading"]["net_return"])
            best_baseline = max(
                float(reports[name]["trading"]["net_return"])
                for name in MODELS
                if name != "mtst"
            )
            wins += mtst > best_baseline
        per_run[run.name] = wins

    total = sum(per_run.values())
    majority = sum(1 for wins in per_run.values() if wins * 2 > n_folds)
    return {
        "run_folds": len(runs) * n_folds,
        "mtst_beat_both_baselines": total,
        "per_run": per_run,
        "runs_with_majority": majority,
        "runs": len(runs),
    }


def _compare_to_references(runs: list[RunArtifact]) -> dict[str, Any]:
    """Did it make money, beat doing nothing, and beat owning the asset?

    Counted over every run-fold, from the economic references each fold
    recorded. Separate from :func:`_compare_to_baselines` because they answer
    different questions and one has repeatedly been read as answering the other:
    the model can beat majority-class and momentum in every fold while losing
    money in every fold, and both facts belong in the report.

    Pre-correction artifacts do not carry references. That is reported as
    unavailable rather than inferred — a net return sign is not a substitute for
    a buy-and-hold comparison.
    """
    run_folds = 0
    beat_cash = 0
    beat_hold = 0
    with_references = 0
    net_returns: list[float] = []
    hold_returns: list[float] = []

    for run in runs:
        for fold in run.folds:
            run_folds += 1
            reports = fold["outer_validation"]
            net = float(reports["mtst"]["trading"]["net_return"])
            net_returns.append(net)
            beat_cash += net > 0
            hold = (reports.get(REFERENCES_KEY) or {}).get("buy_and_hold") or {}
            if not hold.get("available"):
                continue
            with_references += 1
            hold_returns.append(float(hold["net_return"]))
            beat_hold += net > float(hold["net_return"])

    return {
        "run_folds": run_folds,
        "run_folds_with_references": with_references,
        "mean_net_return": round(statistics.fmean(net_returns), 6) if net_returns else None,
        "mean_buy_and_hold_net_return": (
            round(statistics.fmean(hold_returns), 6) if hold_returns else None
        ),
        "beat_cash_run_folds": beat_cash,
        "beat_buy_and_hold_run_folds": beat_hold,
    }


# --- per-fold model stability, and which fold is best and worst ---------------
def fold_model_stability(runs: list[RunArtifact]) -> list[dict[str, Any]]:
    """Per fold, how the model behaved across seeds.

    The market in a fold is the same whichever seed ran; only the model changes.
    Putting the two side by side is what makes "fold 2 is a hard regime"
    separable from "fold 2 got an unlucky initialisation".
    """
    stability = []
    for fold in range(runs[0].n_folds):
        net_returns = [run.metric(run.folds[fold], "mtst", "net_return") for run in runs]
        entry: dict[str, Any] = {
            "fold": fold,
            "net_return": {
                **_spread(net_returns),
                "median": round(statistics.median(net_returns), 6),
                "positive_seeds": sum(1 for value in net_returns if value > 0),
                "seeds": len(net_returns),
                "values": {run.name: round(value, 6) for run, value in zip(runs, net_returns)},
            },
        }
        for metric in ("directional_accuracy", "coverage", "n_trades"):
            entry[metric] = _spread(
                [run.metric(run.folds[fold], "mtst", metric) for run in runs]
            )
        for field in ("threshold", "best_epoch"):
            entry[field] = _spread(
                [float(run.folds[fold]["selection"][field]) for run in runs]
            )
        stability.append(entry)
    return stability


def best_and_worst(stability: list[dict[str, Any]]) -> tuple[int, int]:
    """The folds with the highest and lowest mean MTST outer net return.

    Chosen from the data every time. Naming a fold in the source would make the
    comparison a description of one dataset rather than a diagnostic.
    """
    ranked = sorted(stability, key=lambda entry: entry["net_return"]["mean"])
    return ranked[-1]["fold"], ranked[0]["fold"]


# --- dataset-backed regime statistics ----------------------------------------
def regime_report(
    runs: list[RunArtifact],
    research: Any,
    raw: Any = None,
) -> list[dict[str, Any]]:
    """Market statistics for every unique outer block.

    The runs share a geometry by the time this is called — ``compare_runs`` has
    already refused them otherwise — so the unique outer blocks are the first
    run's, and the statistics belong to the fold rather than to any seed.
    """
    seq_len = _seq_len(runs[0])
    blocks: list[dict[str, Any]] = []
    for fold in runs[0].folds:
        start, end = RunArtifact.rows(fold, "outer_validation")
        entry = block_statistics(research, start, end, seq_len)
        entry["fold"] = fold.get("fold")

        # The run recorded how many samples it scored. The reconstruction above
        # must reproduce that number exactly, or these statistics describe a
        # different set of rows than the metrics they are being used to explain
        # — which is the whole failure this reconstruction exists to avoid.
        recorded = (fold.get("samples") or {}).get("outer_validation")
        if recorded is not None and entry["scored_rows"] != recorded:
            raise RegimeDataError(
                f"fold {entry['fold']}: the run scored {recorded} outer samples but "
                f"{entry['scored_rows']} rows reconstruct from seq_len={seq_len} and "
                f"horizon={research.target_spec.horizon}. The dataset and the artifact "
                "disagree about which rows were evaluated; refusing to report."
            )
        if raw is not None:
            # Deliberately the whole block, not the scored rows: this is a view
            # of the market over the fold's span, independent of what the model
            # could be scored on. Labelled as such wherever it is reported.
            timestamps = research.block(start, end)["date"]
            entry["market"] = raw_block_statistics(raw, timestamps, research.timeframe)
        blocks.append(entry)
    return blocks


def _seq_len(run: RunArtifact) -> int:
    """The sequence length the run was produced with.

    Recorded in the artifact's config. Without it the scored rows cannot be
    reconstructed, and guessing one would silently summarise a different set of
    rows than the model saw — so its absence is an error, not a default.
    """
    seq_len = (run.raw.get("config") or {}).get("seq_len")
    if not isinstance(seq_len, int) or seq_len < 1:
        raise RegimeDataError(
            f"{run.path} does not record config.seq_len, so the rows the model was "
            "scored on cannot be reconstructed"
        )
    return seq_len


def _relative(best: float, worst: float) -> float | None:
    """Difference normalised by the larger magnitude: bounded, and sign-safe.

    A plain percentage change explodes when the denominator is near zero, which
    is exactly where several of these metrics live (mean net return, mean
    ema_cross). This stays finite and still ranks a doubling above a nudge.
    """
    scale = max(abs(best), abs(worst))
    return round((best - worst) / scale, 6) if scale > 1e-12 else None


def compare_best_worst(
    regime: list[dict[str, Any]] | None,
    stability: list[dict[str, Any]],
    best: int,
    worst: int,
) -> list[dict[str, Any]]:
    """Rank what differs most between the best and worst fold.

    Descriptive only. A row here says two numbers differ, and the ordering says
    by how much relative to their own scale — neither says one produced the
    other.
    """
    by_fold = {entry["fold"]: entry for entry in stability}
    rows: list[tuple[str, str, float, float]] = [
        (
            "model",
            "mtst net return",
            by_fold[best]["net_return"]["mean"],
            by_fold[worst]["net_return"]["mean"],
        ),
        (
            "model",
            "directional accuracy",
            by_fold[best]["directional_accuracy"]["mean"],
            by_fold[worst]["directional_accuracy"]["mean"],
        ),
        (
            "model",
            "coverage",
            by_fold[best]["coverage"]["mean"],
            by_fold[worst]["coverage"]["mean"],
        ),
        (
            "model",
            "trades",
            by_fold[best]["n_trades"]["mean"],
            by_fold[worst]["n_trades"]["mean"],
        ),
        (
            "model",
            "selected threshold",
            by_fold[best]["threshold"]["mean"],
            by_fold[worst]["threshold"]["mean"],
        ),
    ]

    if regime is not None:
        blocks = {entry["fold"]: entry for entry in regime}
        best_block, worst_block = blocks[best], blocks[worst]

        def add(group: str, label: str, reader) -> None:
            rows.append((group, label, reader(best_block), reader(worst_block)))

        add("target", "future_return mean", lambda b: b["future_return"]["mean"])
        add("target", "future_return std", lambda b: b["future_return"]["std"])
        add("target", "future_return mean abs", lambda b: b["future_return"]["mean_abs"])
        add(
            "target",
            "positive future_return fraction",
            lambda b: b["future_return"]["fraction_positive"],
        )
        for name in _CLASS_LABELS:
            add(
                "target",
                f"{name} fraction",
                lambda b, name=name: b["target_distribution"][name]["fraction"],
            )
        add("volatility", "atr_norm mean", lambda b: b["features"]["atr_norm"]["mean"])
        add("volatility", "atr_norm p90", lambda b: b["features"]["atr_norm"]["p90"])
        add(
            "volatility",
            "realized_vol mean",
            lambda b: b["features"]["realized_vol"]["mean"],
        )
        add("volatility", "realized_vol p90", lambda b: b["features"]["realized_vol"]["p90"])
        add("volatility", "hl_range mean", lambda b: b["features"]["hl_range"]["mean"])
        add("trend", "ema_cross mean", lambda b: b["features"]["ema_cross"]["mean"])
        add(
            "trend",
            "ema_cross fraction positive",
            lambda b: b["features"]["ema_cross"]["fraction_positive"],
        )
        add("trend", "ema_cross std", lambda b: b["features"]["ema_cross"]["std"])
        add("volume", "volume_z mean", lambda b: b["features"]["volume_z"]["mean"])
        add("volume", "volume_z p90 abs", lambda b: b["features"]["volume_z"]["p90_abs"])
        add(
            "volume",
            "volume_change mean",
            lambda b: b["features"]["volume_change"]["mean"],
        )
        if "market" in best_block and "market" in worst_block:
            add("market", "BTC market return", lambda b: b["market"]["market_return"])
            add(
                "market",
                "annualised volatility",
                lambda b: b["market"]["annualised_volatility"],
            )
            add(
                "market",
                "mean abs hourly return",
                lambda b: b["market"]["mean_abs_candle_return"],
            )
            add(
                "market",
                "positive candle fraction",
                lambda b: b["market"]["positive_candle_fraction"],
            )
            add("market", "market max drawdown", lambda b: b["market"]["max_drawdown"])

    comparison = [
        {
            "group": group,
            "metric": label,
            "best_fold": round(float(best_value), 8),
            "worst_fold": round(float(worst_value), 8),
            "absolute_difference": round(float(best_value) - float(worst_value), 8),
            "relative_difference": _relative(float(best_value), float(worst_value)),
        }
        for group, label, best_value, worst_value in rows
    ]
    comparison.sort(key=lambda row: abs(row["relative_difference"] or 0.0), reverse=True)
    return comparison


# --- LONG / SHORT attribution -------------------------------------------------
def attribution_report(runs: list[RunArtifact], target_spec: Any = None) -> dict[str, Any]:
    """Exact per-direction attribution, or the reason it cannot be computed.

    Never approximated. The aggregate reports in ``walkforward.json`` do not
    carry per-sample predictions, and a direction split inferred from them would
    be a guess wearing the costume of a measurement.
    """
    available = {run.name: (run.path.parent / PREDICTIONS_NAME) for run in runs}
    present = {name: path for name, path in available.items() if path.is_file()}
    if not present:
        return {
            "available": False,
            "reason": (
                "directional trade attribution unavailable for this legacy artifact "
                "because outer sample predictions were not persisted"
            ),
            "expected_file": PREDICTIONS_NAME,
        }
    if target_spec is None:
        return {
            "available": False,
            "reason": (
                "outer sample predictions were found, but attribution needs the "
                "dataset's target spec for its cost model: re-run with --dataset"
            ),
            "expected_file": PREDICTIONS_NAME,
        }

    missing = sorted(set(available) - set(present))
    per_run: dict[str, Any] = {}
    frames = []
    for name, path in present.items():
        predictions = load_predictions(
            path, sealed_test_start=next(r for r in runs if r.name == name).sealed_test_start
        )
        frames.append(predictions)
        per_run[name] = direction_attribution(predictions, target_spec)

    combined = pd.concat(frames, ignore_index=True)
    return {
        "available": True,
        "runs_with_predictions": sorted(present),
        "runs_without_predictions": missing,
        "overall": direction_attribution(combined, target_spec),
        "per_run": per_run,
        "per_fold": {
            int(fold): direction_attribution(group, target_spec)
            for fold, group in combined.groupby("fold", sort=True)
        },
    }


# --- candidate hypotheses ------------------------------------------------------
def _separability_candidate(
    stability: list[dict[str, Any]],
    best: int,
    worst: int,
    largest: Any,
) -> dict[str, Any]:
    """Whether the two folds separate, stated from the observed distributions.

    Deliberately not fixed wording. Whether the per-fold spread across seeds
    swamps the difference between folds is a property of the numbers in front of
    us, and asserting either answer in the source would put a claim in the report
    that the data had never been asked about.
    """
    by_fold = {entry["fold"]: entry["net_return"] for entry in stability}
    best_net, worst_net = by_fold[best], by_fold[worst]
    seeds = worst_net["seeds"]
    ranges = (
        f"across {seeds} seed(s) fold {best} spans "
        f"[{best_net['min']:+.6f}, {best_net['max']:+.6f}] and fold {worst} spans "
        f"[{worst_net['min']:+.6f}, {worst_net['max']:+.6f}]"
    )
    overlapping = worst_net["max"] >= best_net["min"]

    if overlapping:
        hypothesis = (
            f"Fold {worst} may be a harder regime, or may be an unlucky draw — the two "
            f"cannot be told apart from these runs. Compare it against fold {best} on "
            "more seeds before treating the across-fold mean as a single number."
        )
        evidence = (
            f"{ranges}, and the ranges overlap, so the per-fold seed spread is "
            "comparable to the difference between the folds"
        )
    else:
        consistency = (
            f"fold {worst} is negative in all {seeds} observed seeds"
            if worst_net["positive_seeds"] == 0
            else f"fold {worst} is positive in only "
            f"{worst_net['positive_seeds']}/{seeds} observed seeds"
        )
        hypothesis = (
            f"Fold {worst} behaves consistently worse than fold {best} across every "
            "seed observed, so it is worth characterising as a regime rather than "
            "dismissed as noise — then replicated on independent periods and assets "
            "before the characterisation is generalised."
        )
        evidence = (
            f"{ranges} with no overlap, and {consistency}. Consistent within these "
            "seeds; one fold per regime is still a sample of one, and seed count is "
            "not a substitute for independent periods"
        )

    return {
        "hypothesis": hypothesis,
        "evidence": evidence,
        "strength": largest("mtst net return", "directional accuracy"),
        # Recorded so a reader can check the wording against the numbers that
        # chose it, rather than trusting the sentence.
        "observed": {
            "best_fold": best,
            "worst_fold": worst,
            "seeds": seeds,
            "best_range": [best_net["min"], best_net["max"]],
            "worst_range": [worst_net["min"], worst_net["max"]],
            "ranges_overlap": bool(overlapping),
            "worst_positive_seeds": worst_net["positive_seeds"],
        },
    }


def candidate_hypotheses(
    comparison: list[dict[str, Any]],
    stability: list[dict[str, Any]],
    attribution: dict[str, Any],
    best: int,
    worst: int,
    horizon: int | None,
) -> list[dict[str, Any]]:
    """Three ranked research questions, generated from the diagnostics.

    Each is a hypothesis to *test*, not a finding, and each is ranked by the
    size of the difference that suggested it. None of them is implemented or
    tuned here; a diagnostic that quietly starts fitting is how a research tool
    becomes a source of overfitting.
    """
    strength = {row["metric"]: abs(row["relative_difference"] or 0.0) for row in comparison}

    def largest(*metrics: str) -> float:
        return max((strength.get(metric, 0.0) for metric in metrics), default=0.0)

    horizon_label = f"{horizon}-candle" if horizon else "fixed-horizon"
    threshold_spread = max(entry["threshold"]["std"] for entry in stability)

    candidates = [
        {
            "hypothesis": (
                f"The {horizon_label} target may behave differently across volatility "
                "regimes; a volatility-normalised target threshold is worth testing."
            ),
            "evidence": (
                "realised volatility and ATR differ most between the best and worst "
                "fold, and the label is defined against a fixed cost threshold "
                "regardless of how far price typically moves in that window."
            ),
            "strength": largest(
                "annualised volatility", "realized_vol mean", "atr_norm mean", "atr_norm p90"
            ),
        },
        {
            "hypothesis": (
                "The class prior shifts with the regime; the cost-threshold labelling "
                "may need to be regime-aware before class balance is interpretable."
            ),
            "evidence": (
                "the SHORT/HOLD/LONG mix and the future-return distribution differ "
                "between the two folds, so the model faces a different problem in each."
            ),
            "strength": largest(
                "SHORT fraction",
                "LONG fraction",
                "HOLD fraction",
                "future_return std",
                "future_return mean abs",
            ),
        },
        {
            "hypothesis": (
                "The feature set may lack regime context: nothing in it states which "
                "regime the window belongs to, only what just happened within it."
            ),
            "evidence": (
                "trend orientation and volume behaviour differ between the folds while "
                "the feature contract is identical across every run."
            ),
            "strength": largest(
                "ema_cross mean",
                "ema_cross fraction positive",
                "volume_z mean",
                "volume_z p90 abs",
            ),
        },
        {
            "hypothesis": (
                "LONG and SHORT may need separate decision thresholds rather than one "
                "symmetric cut."
            ),
            "evidence": (
                (
                    "per-direction attribution is available and can be read directly"
                    if attribution.get("available")
                    else "per-direction attribution is not available for these runs, so "
                    "this cannot yet be checked — it needs a run that persists outer "
                    "predictions"
                )
                + f"; the selected threshold varies by up to {threshold_spread:.3f} "
                "across seeds on a single fold."
            ),
            "strength": max(threshold_spread, largest("selected threshold", "coverage")),
        },
        _separability_candidate(stability, best, worst, largest),
    ]
    candidates.sort(key=lambda candidate: candidate["strength"], reverse=True)
    return [
        {"rank": rank, **candidate, "strength": round(candidate["strength"], 6)}
        for rank, candidate in enumerate(candidates[:3], start=1)
    ]


def _moved(std: float | None) -> bool:
    """Whether a recorded spread is evidence that a number moved.

    ``None`` is not. :func:`_spread` reports ``std`` as ``None`` when a metric is
    defined in no observation at all, and a baseline that never trades has no
    Sharpe to move in any run. Reading that as non-zero spread — which
    ``None != 0.0`` did — states that artifacts predate the baseline scoring fix
    on the strength of a measurement nobody ever took.

    Undefined is absence of evidence, not evidence of drift. A measured zero
    stays a measured zero, and a measured non-zero spread is still drift.
    """
    return std is not None and std != 0.0


def deterministic_baseline_drift(summary: dict[str, Any]) -> list[str]:
    """Deterministic baselines whose reported numbers moved across seeds.

    Impossible under the current scoring rules, so a non-empty result dates the
    artifacts rather than describing them: they were produced when every model —
    baselines included — was scored at the threshold selected for the *model*, so
    a rule with no fitted parameters inherited the model's seed dependence.

    Only *measured* spread counts — see :func:`_moved`.
    """
    drifted = []
    for name in DETERMINISTIC_BASELINES:
        stats = summary.get("per_model", {}).get(name, {})
        moved = any(
            _moved(metric.get("across_runs", {}).get("std"))
            or any(_moved(fold.get("std")) for fold in metric.get("per_fold", []))
            for metric in stats.values()
            if isinstance(metric, dict)
        )
        if moved:
            drifted.append(name)
    return drifted


def to_markdown(
    runs: list[RunArtifact],
    audits: dict[str, list[str]],
    mismatches: list[str],
    summary: dict[str, Any] | None,
    analysis: dict[str, Any] | None = None,
) -> str:
    reference = runs[0]
    lines = [
        "# Walk-forward regime diagnostics",
        "",
        f"{len(runs)} run(s), read from disk. Sealed test block starts at row "
        f"{reference.sealed_test_start} and is not opened by this tool.",
        "",
        "| run | path | seed | folds | integrity |",
        "| --- | --- | --- | --- | --- |",
    ]
    for run in runs:
        problems = audits[run.name]
        state = "ok" if not problems else f"**{len(problems)} problem(s)**"
        lines.append(f"| {run.name} | `{run.path}` | {run.seed} | {run.n_folds} | {state} |")

    failing = {name: problems for name, problems in audits.items() if problems}
    if failing:
        lines += ["", "## Integrity problems", ""]
        for name, problems in failing.items():
            lines.append(f"**{name}**")
            lines += [f"- {problem}" for problem in problems]
            lines.append("")
    else:
        lines += [
            "",
            "## Integrity",
            "",
            "Every run: three regions per fold in order, outer blocks disjoint, no row "
            "at or beyond the sealed boundary, sealed test not evaluated, and the "
            "recorded across-fold summary reproduced from the per-fold reports.",
        ]

    if mismatches:
        lines += ["", "## Runs are not comparable", ""]
        lines += [f"- {mismatch}" for mismatch in mismatches]
        lines += [
            "",
            "No cross-run aggregate is reported: these runs do not measure the same "
            "rows, so averaging them would compare different blocks of market.",
            "",
        ]
        return "\n".join(lines)

    lines += [
        "",
        "## Fold geometry (identical across runs)",
        "",
        "| fold | train | inner validation | outer validation |",
        "| --- | --- | --- | --- |",
    ]
    for fold in reference.folds:
        ranges = [RunArtifact.rows(fold, region) for region in REGIONS]
        cells = " | ".join(f"[{start}, {end})" for start, end in ranges)
        lines.append(f"| {fold.get('fold')} | {cells} |")

    if summary is None:
        return "\n".join(lines + [""])

    lines += [
        "",
        "## Outer validation across runs",
        "",
        "Each run's across-fold mean, then the spread of those means. The spread is "
        "seed sensitivity of the whole procedure — retraining changes the answer by "
        "this much.",
        "",
    ]

    drifted = deterministic_baseline_drift(summary)
    if drifted:
        lines += [
            f"> **These artifacts predate the baseline scoring fix.** "
            f"{', '.join(drifted)} {'show' if len(drifted) > 1 else 'shows'} "
            "non-zero spread across seeds, which is "
            "impossible for a rule with no fitted parameters on identical geometry "
            "and data. They were produced when every model was scored at the "
            "threshold selected for the *model*, so the baselines inherited its seed "
            "dependence. Their columns below are historical; re-running "
            "`nn.walkforward` regenerates them as the constants they should be. The "
            "MTST columns are unaffected.",
            "",
        ]

    if summary.get("metric_schema") == SCHEMA_LEGACY:
        lines += [
            "> **These artifacts predate the risk-metric correction.** They carry a "
            f"`{LEGACY_TRADING_KEY}` field that annualised per-trade statistics by "
            "`candles_per_year / horizon` — the number of trades the strategy *could* "
            "have taken back to back, not the number it took. On a signal that trades "
            "rarely that figure is overstated several-fold, and it grew as the target "
            "horizon shrank. It is **not** reported here and **not** comparable to "
            f"anything below; {', '.join(CURRENT_ONLY_METRICS)} are absent from these "
            "runs and are skipped. **`max_drawdown` is skipped for a different "
            "reason**: these runs recorded it, but measured the running peak from "
            "the first completed trade instead of from starting capital, so a "
            "strategy under water from its first trade understated it — a single "
            "10% loss recorded zero. Its values here are not comparable to a "
            "corrected run's. Every other metric is unaffected. Re-run "
            "`nn.walkforward` to produce an artifact with the corrected statistics.",
            "",
        ]

    lines += [
        "| model | metric | mean of run means | std | min | max |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for model in MODELS:
        for metric in summary["per_model"][model]:
            across = summary["per_model"][model][metric]["across_runs"]
            if across["mean"] is None:
                lines.append(f"| {model} | {metric} | n/a | n/a | n/a | n/a |")
                continue
            lines.append(
                f"| {model} | {metric} | {across['mean']:.6g} | {across['std']:.6g} | "
                f"{across['min']:.6g} | {across['max']:.6g} |"
            )

    lines += [
        "",
        "## Per-fold spread across runs (mtst, outer validation)",
        "",
        "| fold | metric | mean | std | min | max |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for metric in summary["per_model"]["mtst"]:
        for entry in summary["per_model"]["mtst"][metric]["per_fold"]:
            if entry["mean"] is None:
                lines.append(f"| {entry['fold']} | {metric} | n/a | n/a | n/a | n/a |")
                continue
            lines.append(
                f"| {entry['fold']} | {metric} | {entry['mean']:.6g} | "
                f"{entry['std']:.6g} | {entry['min']:.6g} | {entry['max']:.6g} |"
            )

    lines += [
        "",
        "## Selection stability (chosen on inner validation)",
        "",
        "| fold | threshold mean | threshold std | threshold range | epoch mean | "
        "epoch std |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for entry in summary["selection"]:
        threshold, epoch = entry["threshold"], entry["best_epoch"]
        lines.append(
            f"| {entry['fold']} | {threshold['mean']:.4f} | {threshold['std']:.4f} | "
            f"{threshold['min']:.2f}–{threshold['max']:.2f} | {epoch['mean']:.2f} | "
            f"{epoch['std']:.2f} |"
        )

    comparison = summary["baseline_comparison"]
    lines += [
        "",
        "## Against the statistical / rule baselines",
        "",
        "A floor test: did the model learn more than always predicting the majority "
        "class, or than following the trend? **Passing it is not evidence of "
        "profitability** — see the next section, which is the one that answers that.",
        "",
        f"MTST beat both baselines on net return in "
        f"{comparison['mtst_beat_both_baselines']}/{comparison['run_folds']} run-folds, "
        f"and in a majority of folds in {comparison['runs_with_majority']}/"
        f"{comparison['runs']} runs.",
        "",
        "| run | folds won |",
        "| --- | --- |",
    ]
    for name, wins in comparison["per_run"].items():
        lines.append(f"| {name} | {wins}/{summary['folds']} |")

    lines += _economic_markdown(summary)

    if analysis:
        lines += _analysis_markdown(analysis, reference)

    lines += [
        "",
        "## Limitations",
        "",
        "- These are outer-validation blocks: nothing was fitted on them, which is",
        "  what makes them comparable across seeds. They are still research blocks,",
        "  re-run while the method was being built. Seed spread measures the stability",
        "  of the research procedure, not out-of-sample performance, and this report is",
        "  not a claim of profitability. The sealed test block remains unopened.",
        "- Every difference reported above is a **coincidence in the data**, not a",
        "  cause. One fold per regime is a sample of one; nothing here establishes that",
        "  a market property produced a model outcome.",
        "- Regime statistics describe the outer block only. The model was trained on",
        "  everything before it, so a fold's difficulty also depends on how much its",
        "  training window resembled it — which these numbers do not measure.",
        "- Per-fold seed spread is not a confidence interval on performance.",
        "",
    ]
    return "\n".join(lines)


def _economic_markdown(summary: dict[str, Any]) -> list[str]:
    """Against doing nothing, and against owning the asset.

    The section the baseline comparison above cannot stand in for. A model can
    beat majority-class and momentum in every run-fold while losing money in
    every one of them — that has happened in this repository's own artifacts —
    so the two questions get two sections and the losing answer is not allowed
    to hide behind the winning one.
    """
    economic = summary.get("economic_comparison") or {}
    run_folds = economic.get("run_folds") or 0
    lines = ["", "## Against doing nothing", ""]

    mean_net = economic.get("mean_net_return")
    if mean_net is None:
        return lines + ["No net return was recorded for these runs.", ""]

    made_money = mean_net > 0
    lines += [
        f"**CASH** — never trade, no fees — returns exactly 0. MTST's mean outer net "
        f"return across all run-folds is **{mean_net:+.6f}**, so it "
        f"{'made money' if made_money else '**lost money**'} and beat doing nothing in "
        f"{economic['beat_cash_run_folds']}/{run_folds} run-folds.",
        "",
    ]

    with_references = economic.get("run_folds_with_references") or 0
    if not with_references:
        lines += [
            "**Buy-and-hold** is not available for these runs: they were produced "
            "before economic references were recorded, and it cannot be reconstructed "
            "from the aggregates. Nothing is inferred in its place — a net return sign "
            "says whether the strategy beat cash, not whether it beat owning the asset.",
            "",
        ]
        return lines

    hold = economic["mean_buy_and_hold_net_return"]
    lines += [
        f"**Buy-and-hold** over the same windows, one round trip charged once, returned "
        f"a mean **{hold:+.6f}**. MTST beat it in "
        f"{economic['beat_buy_and_hold_run_folds']}/{with_references} run-folds where "
        "the reference was recorded.",
        "",
    ]
    return lines


def _analysis_markdown(analysis: dict[str, Any], reference: RunArtifact) -> list[str]:
    """The dataset-backed sections: regimes, stability, best vs worst, attribution."""
    lines: list[str] = []
    regime = analysis.get("regime")
    stability = analysis["fold_stability"]
    best, worst = analysis["best_fold"], analysis["worst_fold"]

    lines += [
        "",
        "## Dataset / sealed-test status",
        "",
        (
            f"- Processed dataset: `{analysis['dataset']}`"
            if analysis.get("dataset")
            else "- Processed dataset: not given (`--dataset`), so no regime statistics below."
        ),
    ]
    if analysis.get("dataset"):
        lines += [
            (
                f"- Raw OHLCV: `{analysis['raw']}`"
                if analysis.get("raw")
                else "- Raw OHLCV: not given (`--raw`), so market-level statistics are absent."
            ),
            f"- Rows read: `[0, {reference.sealed_test_start})`. The frame is truncated at "
            "the sealed boundary on load, so no statistic below can have seen a sealed row.",
            f"- Model-facing statistics use the scored rows only "
            f"(seq_len {analysis['seq_len']}, horizon {analysis['horizon']}), selected "
            "through the same index logic the run used.",
            f"- Highest outer row used: {analysis['highest_outer_row']} "
            f"(sealed test starts at {reference.sealed_test_start}).",
            "- Sealed test evaluated: **no**.",
        ]

    if regime:
        lines += [
            "",
            "## Market regime statistics (outer blocks)",
            "",
            "Identical across seeds by construction — the market in a fold does not",
            "depend on which seed trained on it.",
            "",
            "Computed over the rows the model was **scored** on — window warm-up,",
            "label embargo and market-gap filtering applied — not over every row the",
            "block spans. Both counts are shown.",
            "",
            "| fold | block rows | scored rows | period | fut ret mean | fut ret std "
            "| fut ret abs | pos frac |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for block in regime:
            returns = block["future_return"]
            lines.append(
                f"| {block['fold']} | {block['block_rows']} | {block['scored_rows']} | "
                f"{block['period']['start'][:10]} to {block['period']['end'][:10]} | "
                f"{returns['mean']:+.6f} | {returns['std']:.6f} | "
                f"{returns['mean_abs']:.6f} | {returns['fraction_positive']:.4f} |"
            )

        lines += [
            "",
            "### Feature regime",
            "",
            "| fold | ema_cross mean | ema_cross >0 | atr_norm mean | atr_norm p90 | "
            "realized_vol mean | realized_vol p90 | hl_range mean | volume_z mean | "
            "volume_z p90abs |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for block in regime:
            f = block["features"]
            lines.append(
                f"| {block['fold']} | {f['ema_cross']['mean']:+.6f} | "
                f"{f['ema_cross']['fraction_positive']:.4f} | {f['atr_norm']['mean']:.6f} | "
                f"{f['atr_norm']['p90']:.6f} | {f['realized_vol']['mean']:.6f} | "
                f"{f['realized_vol']['p90']:.6f} | {f['hl_range']['mean']:.6f} | "
                f"{f['volume_z']['mean']:+.6f} | {f['volume_z']['p90_abs']:.6f} |"
            )

        lines += [
            "",
            "### Target distribution",
            "",
            "| fold | "
            + " | ".join(f"{name} n | {name} frac | {name} mean ret" for name in _CLASS_LABELS)
            + " |",
            "| --- |" + " --- |" * (3 * len(_CLASS_LABELS)),
        ]
        for block in regime:
            cells = []
            for name in _CLASS_LABELS:
                entry = block["target_distribution"][name]
                cells += [
                    str(entry["count"]),
                    f"{entry['fraction']:.4f}",
                    f"{entry['mean_future_return']:+.6f}",
                ]
            lines.append(f"| {block['fold']} | " + " | ".join(cells) + " |")

        if any("market" in block for block in regime):
            lines += [
                "",
                "### Raw market behaviour (timestamp-aligned OHLCV)",
                "",
                "A view of the market over the **whole block span**, independent of",
                "which rows were scorable. Not directly comparable row-for-row with the",
                "model-facing statistics above.",
                "",
                "| fold | start close | end close | market return | mean hourly | "
                "std hourly | ann. vol | mean abs hourly | pos candles | max DD | gaps |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
            for block in regime:
                m = block.get("market")
                if not m:
                    continue
                lines.append(
                    f"| {block['fold']} | {m['start_close']:.2f} | {m['end_close']:.2f} | "
                    f"{m['market_return']:+.4f} | {m['mean_candle_return']:+.6f} | "
                    f"{m['candle_return_std']:.6f} | {m['annualised_volatility']:.4f} | "
                    f"{m['mean_abs_candle_return']:.6f} | "
                    f"{m['positive_candle_fraction']:.4f} | {m['max_drawdown']:.4f} | "
                    f"{m['gap_pairs_skipped']} |"
                )

    lines += [
        "",
        "## Per-fold model stability across seeds",
        "",
        "| fold | net return mean | std | median | positive seeds | dir acc mean±std | "
        "coverage mean±std | trades mean±std | threshold mean±std | threshold range | "
        "epoch mean±std |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for entry in stability:
        net = entry["net_return"]
        lines.append(
            f"| {entry['fold']} | {net['mean']:+.6f} | {net['std']:.6f} | "
            f"{net['median']:+.6f} | {net['positive_seeds']}/{net['seeds']} | "
            f"{entry['directional_accuracy']['mean']:.4f}±"
            f"{entry['directional_accuracy']['std']:.4f} | "
            f"{entry['coverage']['mean']:.4f}±{entry['coverage']['std']:.4f} | "
            f"{entry['n_trades']['mean']:.1f}±{entry['n_trades']['std']:.1f} | "
            f"{entry['threshold']['mean']:.4f}±{entry['threshold']['std']:.4f} | "
            f"{entry['threshold']['min']:.2f}–{entry['threshold']['max']:.2f} | "
            f"{entry['best_epoch']['mean']:.2f}±{entry['best_epoch']['std']:.2f} |"
        )

    lines += [
        "",
        f"## Best vs worst regime — fold {best} (best) vs fold {worst} (worst)",
        "",
        "Selected from the data: highest and lowest mean MTST outer net return across",
        "seeds. Ranked by difference relative to the larger magnitude. **These are",
        "coincidences, not causes** — a row says the two folds differ, nothing more.",
        "",
        "| rank | group | metric | best fold | worst fold | absolute diff | relative |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for rank, row in enumerate(analysis["best_vs_worst"], start=1):
        relative = (
            "n/a"
            if row["relative_difference"] is None
            else (f"{row['relative_difference']:+.4f}")
        )
        lines.append(
            f"| {rank} | {row['group']} | {row['metric']} | {row['best_fold']:+.6g} | "
            f"{row['worst_fold']:+.6g} | {row['absolute_difference']:+.6g} | {relative} |"
        )

    lines += ["", "## LONG vs SHORT attribution", ""]
    attribution = analysis["attribution"]
    if not attribution.get("available"):
        lines += [
            f"**Unavailable.** {attribution['reason']}.",
            "",
            f"Nothing is approximated in its place: the aggregate reports do not carry "
            f"per-sample predictions, and a direction split inferred from them would be a "
            f"guess. Runs produced after this change write `{attribution['expected_file']}` "
            "beside the artifact, and this section fills in automatically when present.",
        ]
    else:
        overall = attribution["overall"]
        lines += [
            "Exact, from persisted outer predictions, using the same cost model as the",
            "fold reports.",
            "",
            "| side | trades | hit rate | mean net | median net | additive sum | "
            "coverage |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for side, coverage_key in (("long", "long_coverage"), ("short", "short_coverage")):
            entry = overall[side]
            lines.append(
                f"| {side.upper()} | {entry['trades']} | {entry['hit_rate']:.4f} | "
                f"{entry['mean_net_return']:+.6f} | {entry['median_net_return']:+.6f} | "
                f"{entry['additive_trade_return_sum']:+.6f} | "
                f"{overall[coverage_key]:.4f} |"
            )
        lines.append(
            f"\nHOLD / no-trade samples: {overall['hold_samples']} of {overall['samples']}."
        )
        if attribution.get("runs_without_predictions"):
            lines.append(
                "\nRuns without a predictions file (excluded from the split): "
                + ", ".join(attribution["runs_without_predictions"])
                + "."
            )

    lines += [
        "",
        "## Candidate hypotheses",
        "",
        "Research questions suggested by the diagnostics, ranked by the size of the",
        "difference behind them. **None is implemented, tuned or tested here.**",
        "",
    ]
    for candidate in analysis["hypotheses"]:
        lines += [
            f"{candidate['rank']}. **{candidate['hypothesis']}**",
            f"   Coincides with: {candidate['evidence']} "
            f"(signal strength {candidate['strength']:.4f}).",
            "",
        ]
    return lines


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit and compare completed walk-forward runs. Takes artifact "
            "directories (or walkforward.json paths) from the local filesystem."
        )
    )
    parser.add_argument(
        "runs",
        nargs="+",
        help=(
            "Artifact directories written by nn.walkforward, each containing "
            f"{ARTIFACT_NAME}. A path to the JSON file itself also works."
        ),
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Directory to write regime_diagnostics.json and regime_diagnostics.md "
            "into, e.g. artifacts/diagnostics/btc_regimes_v1."
        ),
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help=(
            "Processed dataset the artifacts' row indices address. Enables the "
            "regime statistics. Read only up to the sealed boundary."
        ),
    )
    parser.add_argument(
        "--raw",
        default=None,
        help=(
            "Raw OHLCV parquet, joined to the outer blocks on exact timestamps. "
            "Enables market-level statistics. Requires --dataset."
        ),
    )
    return parser


def analyse(
    runs: list[RunArtifact],
    dataset: str | None,
    raw_path: str | None,
) -> dict[str, Any]:
    """Fold stability, best/worst, regimes when a dataset is given, attribution.

    The dataset is optional; everything that does not need it is computed either
    way, so the five existing runs stay analysable without one.
    """
    reference = runs[0]
    stability = fold_model_stability(runs)
    best, worst = best_and_worst(stability)
    highest_outer = max(
        RunArtifact.rows(fold, "outer_validation")[1] - 1 for fold in reference.folds
    )

    research = None
    regime = None
    seq_len = None
    if dataset:
        seq_len = _seq_len(reference)
        research = load_research_frame(
            dataset,
            sealed_test_start=reference.sealed_test_start,
            identity=reference.dataset,
        )
        raw = load_raw_ohlcv(raw_path) if raw_path else None
        regime = regime_report(runs, research, raw)

    attribution = attribution_report(runs, research.target_spec if research else None)
    comparison = compare_best_worst(regime, stability, best, worst)
    horizon = (reference.dataset.get("target_spec") or {}).get("horizon")

    return {
        "dataset": dataset,
        "raw": raw_path,
        "seq_len": seq_len,
        "horizon": horizon,
        "sealed_test_start": reference.sealed_test_start,
        "sealed_test_evaluated": False,
        "highest_outer_row": highest_outer,
        "fold_stability": stability,
        "best_fold": best,
        "worst_fold": worst,
        "regime": regime,
        "best_vs_worst": comparison,
        "attribution": attribution,
        "hypotheses": candidate_hypotheses(
            comparison, stability, attribution, best, worst, horizon
        ),
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = build_argparser().parse_args(argv)

    runs = [load_run(path) for path in args.runs]
    duplicates = {run.name for run in runs if [r.name for r in runs].count(run.name) > 1}
    if duplicates:
        raise SystemExit(
            f"two runs would be reported under the same name: {sorted(duplicates)}. "
            "Give directories with distinct names."
        )

    audits = {run.name: audit_run(run) for run in runs}
    mismatches = compare_runs(runs) if len(runs) > 1 else []
    # An unsound or incomparable set of runs gets its problems reported and no
    # aggregate: a headline computed over them would be the thing to distrust.
    comparable = not mismatches and not any(audits.values())
    summary = aggregate(runs) if comparable else None

    if args.raw and not args.dataset:
        raise SystemExit(
            "--raw needs --dataset: raw candles are matched to the outer blocks by the "
            "timestamps in the processed dataset, so there is nothing to align without it."
        )
    analysis = None
    if comparable:
        try:
            analysis = analyse(runs, args.dataset, args.raw)
        except RegimeDataError as exc:
            raise SystemExit(str(exc)) from exc

    markdown = to_markdown(runs, audits, mismatches, summary, analysis)
    print(markdown)

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / REPORT_JSON).write_text(
            json.dumps(
                {
                    "runs": [
                        {
                            "name": run.name,
                            "path": str(run.path),
                            "seed": run.seed,
                            "folds": run.n_folds,
                            "sealed_test_start": run.sealed_test_start,
                            "integrity_problems": audits[run.name],
                        }
                        for run in runs
                    ],
                    "comparability_problems": mismatches,
                    "metric_schema": (summary or {}).get("metric_schema"),
                    "skipped_metrics": (summary or {}).get("skipped_metrics", []),
                    "sealed_test_evaluated": False,
                    "aggregated_from": "outer_validation",
                    "legacy_baseline_scoring": (
                        deterministic_baseline_drift(summary) if summary else []
                    ),
                    "summary": summary,
                    "analysis": analysis,
                },
                indent=2,
                default=str,
            )
            + "\n"
        )
        (out_dir / REPORT_MD).write_text(markdown + "\n")
        logger.info("Wrote diagnostics to %s", out_dir)

    problems = sum(len(found) for found in audits.values()) + len(mismatches)
    if problems:
        logger.error("%d problem(s) found across %d run(s)", problems, len(runs))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
