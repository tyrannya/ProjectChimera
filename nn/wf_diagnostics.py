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

Everything is read from paths given on the command line. No artifact, dataset or
model is expected to live in the repository, and nothing here loads a dataset or
a checkpoint: the inputs are JSON reports, the output is a report about them.

**Reading the output.** These are outer-validation numbers — blocks nothing was
fitted on, which is what makes them worth comparing. Seed spread measures the
stability of the research procedure, not out-of-sample performance, and it is
not a claim of profitability. The sealed test block stays unopened; this tool
refuses to read an artifact that says otherwise.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nn.walkforward import MODELS, SUMMARY_METRICS

logger = logging.getLogger(__name__)

#: The file `nn.walkforward` writes, looked up when a directory is given.
ARTIFACT_NAME = "walkforward.json"

#: The three regions of a fold, in the order they must appear in time.
REGIONS = ("train", "inner_validation", "outer_validation")


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

    def metric(self, fold: dict[str, Any], model: str, metric: str) -> float:
        section, key = SUMMARY_METRICS[metric]
        return float(fold["outer_validation"][model][section][key])


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

    for model in MODELS:
        recorded_model = per_model.get(model)
        if not isinstance(recorded_model, dict):
            continue
        for metric in SUMMARY_METRICS:
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
            recomputed = round(statistics.fmean(values), 6)
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
    for run in runs[1:]:
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
        for key in ("rows", "pair", "timeframe"):
            if run.dataset.get(key) != reference.dataset.get(key):
                problems.append(
                    f"{run.name} dataset {key} is {run.dataset.get(key)!r}, "
                    f"{reference.name} has {reference.dataset.get(key)!r}"
                )
    return problems


def _spread(values: list[float]) -> dict[str, float]:
    return {
        "mean": round(statistics.fmean(values), 6),
        # Sample standard deviation needs two observations; one run has none.
        "std": round(statistics.stdev(values), 6) if len(values) > 1 else 0.0,
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def aggregate(runs: list[RunArtifact]) -> dict[str, Any]:
    """Seed stability: how far each outer-validation number moves across runs."""
    names = [run.name for run in runs]
    n_folds = runs[0].n_folds

    per_model: dict[str, Any] = {}
    for model in MODELS:
        metrics: dict[str, Any] = {}
        for metric in SUMMARY_METRICS:
            per_run = {
                run.name: [run.metric(fold, model, metric) for fold in run.folds]
                for run in runs
            }
            run_means = [statistics.fmean(per_run[name]) for name in names]
            metrics[metric] = {
                # Each run's across-fold mean, then the spread of those means.
                "per_run_mean": {name: round(mean, 6) for name, mean in zip(names, run_means)},
                "across_runs": _spread(run_means),
                # And the same metric fold by fold, across runs.
                "per_fold": [
                    {
                        "fold": fold,
                        "values": {name: round(per_run[name][fold], 6) for name in names},
                        **_spread([per_run[name][fold] for name in names]),
                    }
                    for fold in range(n_folds)
                ],
            }
        per_model[model] = metrics

    return {
        "runs": names,
        "folds": n_folds,
        "per_model": per_model,
        "selection": _aggregate_selection(runs, names, n_folds),
        "baseline_comparison": _compare_to_baselines(runs, n_folds),
    }


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


def to_markdown(
    runs: list[RunArtifact],
    audits: dict[str, list[str]],
    mismatches: list[str],
    summary: dict[str, Any] | None,
) -> str:
    reference = runs[0]
    lines = [
        "# Walk-forward diagnostics",
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
        "| model | metric | mean of run means | std | min | max |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for model in MODELS:
        for metric in SUMMARY_METRICS:
            across = summary["per_model"][model][metric]["across_runs"]
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
    for metric in SUMMARY_METRICS:
        for entry in summary["per_model"]["mtst"][metric]["per_fold"]:
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
        "## Against the baselines",
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

    lines += [
        "",
        "## Reading this",
        "",
        "These are outer-validation blocks: nothing was fitted on them, which is what",
        "makes them comparable across seeds. The spread here measures how stable the",
        "research procedure is, not out-of-sample performance — the folds were run",
        "repeatedly during model development. It is not a claim of profitability, and",
        "the sealed test block remains unopened.",
        "",
    ]
    return "\n".join(lines)


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
        help="Directory to write wf_diagnostics.json and wf_diagnostics.md into.",
    )
    return parser


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

    markdown = to_markdown(runs, audits, mismatches, summary)
    print(markdown)

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "wf_diagnostics.json").write_text(
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
                    "sealed_test_evaluated": False,
                    "aggregated_from": "outer_validation",
                    "summary": summary,
                },
                indent=2,
                default=str,
            )
        )
        (out_dir / "wf_diagnostics.md").write_text(markdown)
        logger.info("Wrote diagnostics to %s", out_dir)

    problems = sum(len(found) for found in audits.values()) + len(mismatches)
    if problems:
        logger.error("%d problem(s) found across %d run(s)", problems, len(runs))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
