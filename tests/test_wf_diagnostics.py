"""Safeguards for the walk-forward diagnostics CLI.

The tool reads ``walkforward.json`` artifacts from paths given at runtime, so
these tests build artifacts on disk rather than reaching for any real run. Two
kinds of fixture, deliberately:

* **synthetic artifacts** built by :func:`write_run`, whose row geometry comes
  from the production planner and whose summary comes from the production
  aggregator. That keeps the fixture honest while letting a test break exactly
  one thing and check the diagnostics notices;
* **a real artifact**, produced by running ``nn.walkforward`` end to end on
  generated candles. That is the compatibility check: it is the only way to know
  the reader still matches what the writer emits, and no hand-written fixture
  can stand in for it.

No test here needs a committed dataset, a committed artifact, or a checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from chimera.contracts import CLASS_ORDER, TargetSpec
from chimera.features import FeatureSpec
from nn import evaluate as ev
from nn import regime, walkforward, wf_diagnostics
from nn.data_pipeline import build_dataset, save_dataset
from nn.dataset import sealed_test_start
from tools.make_sample_data import generate_candles

TINY = [
    "--epochs",
    "1",
    "--seq-len",
    "16",
    "--d-model",
    "16",
    "--n-heads",
    "2",
    "--num-layers",
    "1",
    "--device",
    "cpu",
]

N_ROWS = 1200
FOLDS = 3


# --- synthetic artifacts ------------------------------------------------------
def plan_for(n_rows: int = N_ROWS, folds: int = FOLDS):
    """Fold geometry straight from the production planner."""
    boundary = sealed_test_start(n_rows, 0.70, 0.15)
    args = walkforward.build_argparser().parse_args(["--dataset", "x", "--folds", str(folds)])
    sizes = walkforward.resolve_sizes(args, boundary)
    return boundary, walkforward.plan_nested_folds(boundary, folds, *sizes)


def period(split) -> dict[str, Any]:
    """A period block in the shape ``ResearchData.period`` produces."""
    return {
        "start": f"2023-01-01T{split.start % 24:02d}:00:00+00:00",
        "end": f"2023-01-02T{(split.end - 1) % 24:02d}:00:00+00:00",
        "rows": len(split),
        "row_range": [split.start, split.end],
    }


def report(net_return: float, macro_f1: float = 0.4) -> dict[str, Any]:
    """One model's evaluation report, carrying every aggregated metric."""
    return {
        "classification": {
            "n_samples": 83,
            "macro_f1": macro_f1,
            "accuracy": 0.5,
            "directional_accuracy": 0.55,
            "coverage": 0.75,
            "calibration_error": 0.1,
        },
        "trading": {
            "n_trades": 20,
            "net_return": net_return,
            "sharpe": net_return * 10,
            "max_drawdown": 0.05,
        },
    }


def write_run(
    directory: Path,
    *,
    seed: int = 42,
    n_rows: int = N_ROWS,
    folds: int = FOLDS,
    returns: dict[str, list[float]] | None = None,
    thresholds: list[float] | None = None,
    epochs: list[int] | None = None,
    dataset: dict[str, Any] | None = None,
) -> Path:
    """Write a well-formed artifact, then let a caller break one thing.

    ``returns`` maps model name to one net return per fold; the defaults make
    MTST win every fold, so a test about the baseline comparison has to say so
    explicitly rather than inherit it.
    """
    boundary, plans = plan_for(n_rows, folds)
    returns = returns or {
        "majority_baseline": [-0.01] * folds,
        "momentum_baseline": [0.0] * folds,
        "mtst": [0.1 * (i + 1) for i in range(folds)],
    }
    thresholds = thresholds or [0.4] * folds
    epochs = epochs or [2] * folds

    fold_payloads = []
    for index, plan in enumerate(plans):
        fold_payloads.append(
            {
                "fold": index,
                "seed": seed + index,
                "samples": {"train": 400, "inner_validation": 83, "outer_validation": 83},
                "periods": {
                    "train": period(plan.train),
                    "inner_validation": period(plan.inner),
                    "outer_validation": period(plan.outer),
                },
                "selection": {
                    "best_epoch": epochs[index],
                    "threshold": thresholds[index],
                    "inner_validation_loss": 1.0,
                },
                "outer_validation": {
                    model: report(values[index]) for model, values in returns.items()
                },
            }
        )

    directory.mkdir(parents=True, exist_ok=True)
    (directory / wf_diagnostics.ARTIFACT_NAME).write_text(
        json.dumps(
            {
                "dataset": dataset or {"rows": n_rows, "pair": "BTC/USDT", "timeframe": "1h"},
                "config": {"seed": seed, "folds": folds},
                "sealed_test": {
                    "start_row": boundary,
                    "period": {"row_range": [boundary, n_rows], "rows": n_rows - boundary},
                    "evaluated": False,
                },
                "research_rows": boundary,
                "test_evaluated": False,
                "reported_block": "outer_validation",
                "folds": fold_payloads,
                # The production aggregator, so the fixture's summary is exactly
                # what a real run would have recorded for these folds.
                "summary": walkforward.summarise(fold_payloads),
            },
            indent=2,
        )
    )
    return directory


def edit(directory: Path, mutate) -> Path:
    """Apply ``mutate`` to an artifact already on disk."""
    artifact = directory / wf_diagnostics.ARTIFACT_NAME
    payload = json.loads(artifact.read_text())
    mutate(payload)
    artifact.write_text(json.dumps(payload, indent=2))
    return directory


def problems_for(directory: Path) -> list[str]:
    return wf_diagnostics.audit_run(wf_diagnostics.load_run(directory))


# --- 1. loading ---------------------------------------------------------------
def test_a_directory_and_the_json_file_both_load(tmp_path):
    directory = write_run(tmp_path / "btc_nested_v1")
    from_dir = wf_diagnostics.load_run(directory)
    from_file = wf_diagnostics.load_run(directory / wf_diagnostics.ARTIFACT_NAME)

    assert from_dir.folds == from_file.folds
    # Either way the run is named after the directory it came from.
    assert from_dir.name == from_file.name == "btc_nested_v1"
    assert from_dir.seed == 42
    assert from_dir.n_folds == FOLDS


def test_a_missing_artifact_names_the_path_it_looked_for(tmp_path):
    with pytest.raises(SystemExit, match=r"no walk-forward artifact at .*walkforward\.json"):
        wf_diagnostics.load_run(tmp_path / "not_a_run")


def test_unreadable_json_is_refused(tmp_path):
    directory = write_run(tmp_path / "run")
    (directory / wf_diagnostics.ARTIFACT_NAME).write_text('{"folds": [')
    with pytest.raises(SystemExit, match="not readable JSON"):
        wf_diagnostics.load_run(directory)


def test_an_artifact_with_no_folds_is_refused(tmp_path):
    directory = edit(write_run(tmp_path / "run"), lambda p: p.update(folds=[]))
    with pytest.raises(SystemExit, match="has no folds"):
        wf_diagnostics.load_run(directory)


def test_a_pre_nested_artifact_is_refused_and_says_why(tmp_path):
    """The two-region schema must not be read as if it were an evaluation.

    Its single ``validation`` block was both selected on and reported, so
    averaging it beside nested runs would quietly reintroduce the optimism the
    nested design removed.
    """

    def to_legacy(payload):
        for fold in payload["folds"]:
            fold["validation"] = fold.pop("outer_validation")
            fold["periods"]["validation"] = fold["periods"].pop("outer_validation")

    directory = edit(write_run(tmp_path / "old_run"), to_legacy)
    with pytest.raises(SystemExit, match="pre-nested artifact"):
        wf_diagnostics.load_run(directory)


def test_a_fold_missing_a_region_is_refused(tmp_path):
    directory = edit(
        write_run(tmp_path / "run"),
        lambda p: p["folds"][1]["periods"].pop("inner_validation"),
    )
    with pytest.raises(SystemExit, match="missing the inner_validation period"):
        wf_diagnostics.load_run(directory)


def test_an_artifact_without_a_sealed_boundary_is_refused(tmp_path):
    directory = edit(write_run(tmp_path / "run"), lambda p: p.pop("sealed_test"))
    with pytest.raises(SystemExit, match="no sealed_test.start_row"):
        wf_diagnostics.load_run(directory)


# --- 2. the integrity audit ---------------------------------------------------
def test_a_sound_artifact_audits_clean(tmp_path):
    assert problems_for(write_run(tmp_path / "run")) == []


def test_overlapping_outer_blocks_are_caught(tmp_path):
    """Two folds reporting the same row would count it twice in the mean."""

    def overlap(payload):
        first = payload["folds"][0]["periods"]["outer_validation"]["row_range"]
        second = payload["folds"][1]["periods"]["outer_validation"]
        second["row_range"] = [first[0], first[1]]

    problems = problems_for(edit(write_run(tmp_path / "run"), overlap))
    assert any("overlaps fold 0" in problem for problem in problems)


def test_a_reported_block_that_was_selected_on_is_caught(tmp_path):
    """Inner running past the start of outer is the leak this design removed."""

    def bleed(payload):
        periods = payload["folds"][0]["periods"]
        outer_start = periods["outer_validation"]["row_range"][0]
        periods["inner_validation"]["row_range"][1] = outer_start + 10

    problems = problems_for(edit(write_run(tmp_path / "run"), bleed))
    assert any("the reported block was selected on" in problem for problem in problems)


def test_training_overlapping_the_inner_block_is_caught(tmp_path):
    def bleed(payload):
        periods = payload["folds"][0]["periods"]
        periods["train"]["row_range"][1] = periods["inner_validation"]["row_range"][0] + 5

    problems = problems_for(edit(write_run(tmp_path / "run"), bleed))
    assert any("training overlaps the block it is selected on" in p for p in problems)


@pytest.mark.parametrize("region", wf_diagnostics.REGIONS)
def test_any_region_reaching_the_sealed_block_is_caught(tmp_path, region):
    """Asserted on rows for all three regions, not just the reported one."""
    boundary = sealed_test_start(N_ROWS, 0.70, 0.15)

    def cross(payload):
        payload["folds"][-1]["periods"][region]["row_range"][1] = boundary + 1

    problems = problems_for(edit(write_run(tmp_path / "run"), cross))
    assert any(
        f"{region} ends at row {boundary + 1}" in problem and "sealed" in problem
        for problem in problems
    )


@pytest.mark.parametrize(
    "mutate, expected",
    [
        pytest.param(
            lambda p: p.update(test_evaluated=True), "test_evaluated", id="test-evaluated"
        ),
        pytest.param(
            lambda p: p.pop("test_evaluated"), "test_evaluated", id="test-flag-missing"
        ),
        pytest.param(
            lambda p: p["sealed_test"].update(evaluated=True),
            "sealed_test.evaluated",
            id="sealed-evaluated",
        ),
        pytest.param(
            lambda p: p.update(reported_block="inner_validation"),
            "reported_block",
            id="wrong-reported-block",
        ),
        pytest.param(
            lambda p: p["summary"].update(aggregated_from="inner_validation"),
            "summary.aggregated_from",
            id="wrong-aggregation-source",
        ),
        pytest.param(
            lambda p: p["summary"].update(folds=99), "summary.folds", id="fold-count"
        ),
    ],
)
def test_the_sealing_and_provenance_flags_fail_closed(tmp_path, mutate, expected):
    problems = problems_for(edit(write_run(tmp_path / "run"), mutate))
    assert any(expected in problem for problem in problems)


def test_a_summary_that_does_not_follow_from_the_folds_is_caught(tmp_path):
    """The headline has to be reproducible from the per-fold reports beside it."""

    def rewrite_headline(payload):
        payload["summary"]["per_model"]["mtst"]["net_return"]["mean"] = 0.9

    problems = problems_for(edit(write_run(tmp_path / "run"), rewrite_headline))
    assert any(
        "summary mtst.net_return mean is 0.9" in problem and "folds average" in problem
        for problem in problems
    )


def test_editing_a_fold_out_from_under_the_summary_is_caught(tmp_path):
    """The same check from the other side: the folds changed, the headline did not."""

    def rewrite_fold(payload):
        payload["folds"][0]["outer_validation"]["mtst"]["trading"]["net_return"] = 5.0

    problems = problems_for(edit(write_run(tmp_path / "run"), rewrite_fold))
    assert any("summary mtst.net_return" in problem for problem in problems)


# --- 3. comparing runs --------------------------------------------------------
def test_runs_with_the_same_geometry_compare_clean(tmp_path):
    runs = [
        wf_diagnostics.load_run(write_run(tmp_path / "a", seed=42)),
        wf_diagnostics.load_run(write_run(tmp_path / "b", seed=142)),
    ]
    assert wf_diagnostics.compare_runs(runs) == []


def test_runs_whose_outer_blocks_are_different_rows_are_not_comparable(tmp_path):
    def shorten_fold_1(payload):
        payload["folds"][1]["periods"]["outer_validation"]["row_range"][1] = 700

    runs = [
        wf_diagnostics.load_run(write_run(tmp_path / "a")),
        wf_diagnostics.load_run(edit(write_run(tmp_path / "b"), shorten_fold_1)),
    ]
    problems = wf_diagnostics.compare_runs(runs)
    assert any(
        "does not share" in problem and "fold 1 outer_validation" in problem
        for problem in problems
    )


def test_runs_with_different_fold_counts_are_not_comparable(tmp_path):
    runs = [
        wf_diagnostics.load_run(write_run(tmp_path / "a", folds=3)),
        wf_diagnostics.load_run(write_run(tmp_path / "b", folds=2)),
    ]
    assert any("folds" in problem for problem in wf_diagnostics.compare_runs(runs))


def test_runs_on_a_different_dataset_are_not_comparable(tmp_path):
    runs = [
        wf_diagnostics.load_run(write_run(tmp_path / "a")),
        wf_diagnostics.load_run(
            edit(write_run(tmp_path / "b"), lambda p: p["dataset"].update(pair="ETH/USDT"))
        ),
    ]
    assert any("dataset pair" in problem for problem in wf_diagnostics.compare_runs(runs))


def test_a_different_sealed_boundary_is_not_comparable(tmp_path):
    runs = [
        wf_diagnostics.load_run(write_run(tmp_path / "a")),
        wf_diagnostics.load_run(
            edit(write_run(tmp_path / "b"), lambda p: p["sealed_test"].update(start_row=999))
        ),
    ]
    assert any("sealed_test_start" in p for p in wf_diagnostics.compare_runs(runs))


# --- 4. the seed-stability aggregate ------------------------------------------
@pytest.fixture
def two_runs(tmp_path):
    """Two runs differing only in seed, with controlled per-fold net returns."""
    a = write_run(
        tmp_path / "btc_nested_v1",
        seed=42,
        returns={
            "majority_baseline": [-0.01, -0.01, -0.01],
            # Momentum wins fold 2 against the first run's MTST but not the second's.
            "momentum_baseline": [0.0, 0.0, 0.35],
            "mtst": [0.1, 0.2, 0.3],
        },
        thresholds=[0.40, 0.40, 0.50],
        epochs=[2, 3, 4],
    )
    b = write_run(
        tmp_path / "btc_nested_seed_142",
        seed=142,
        returns={
            "majority_baseline": [-0.01, -0.01, -0.01],
            "momentum_baseline": [0.0, 0.0, 0.35],
            "mtst": [0.2, 0.3, 0.4],
        },
        thresholds=[0.40, 0.42, 0.80],
        epochs=[2, 3, 6],
    )
    return [wf_diagnostics.load_run(a), wf_diagnostics.load_run(b)]


def test_the_aggregate_reports_each_run_mean_and_the_spread_of_those_means(two_runs):
    summary = wf_diagnostics.aggregate(two_runs)
    net_return = summary["per_model"]["mtst"]["net_return"]

    assert summary["runs"] == ["btc_nested_v1", "btc_nested_seed_142"]
    assert summary["folds"] == 3
    assert net_return["per_run_mean"] == {
        "btc_nested_v1": pytest.approx(0.2),
        "btc_nested_seed_142": pytest.approx(0.3),
    }
    across = net_return["across_runs"]
    assert across["mean"] == pytest.approx(0.25)
    assert across["std"] == pytest.approx(0.0707107, abs=1e-6)
    assert (across["min"], across["max"]) == (pytest.approx(0.2), pytest.approx(0.3))


def test_the_aggregate_reports_each_fold_across_runs(two_runs):
    """Which fold is stable and which is a seed lottery."""
    per_fold = wf_diagnostics.aggregate(two_runs)["per_model"]["mtst"]["net_return"][
        "per_fold"
    ]

    assert [entry["fold"] for entry in per_fold] == [0, 1, 2]
    assert per_fold[0]["values"] == {
        "btc_nested_v1": pytest.approx(0.1),
        "btc_nested_seed_142": pytest.approx(0.2),
    }
    assert per_fold[0]["mean"] == pytest.approx(0.15)
    assert per_fold[2]["mean"] == pytest.approx(0.35)
    for entry in per_fold:
        assert entry["std"] == pytest.approx(0.0707107, abs=1e-6)


def test_every_aggregated_metric_is_one_walkforward_already_summarises(two_runs):
    """One list of metrics, so the two modules cannot drift apart."""
    summary = wf_diagnostics.aggregate(two_runs)
    for model in walkforward.MODELS:
        assert set(summary["per_model"][model]) == set(walkforward.SUMMARY_METRICS)


def test_selection_stability_reports_threshold_and_epoch_per_fold(two_runs):
    selection = wf_diagnostics.aggregate(two_runs)["selection"]

    assert [entry["fold"] for entry in selection] == [0, 1, 2]
    # Fold 0 picked the same threshold in both runs; fold 2 did not.
    assert selection[0]["threshold"]["std"] == pytest.approx(0.0)
    assert selection[2]["threshold"]["std"] > 0.2
    assert selection[2]["threshold"]["values"] == {
        "btc_nested_v1": pytest.approx(0.5),
        "btc_nested_seed_142": pytest.approx(0.8),
    }
    assert selection[2]["best_epoch"]["mean"] == pytest.approx(5.0)


def test_the_baseline_comparison_counts_every_run_fold(two_runs):
    """MTST loses fold 2 in the first run and wins it in the second."""
    comparison = wf_diagnostics.aggregate(two_runs)["baseline_comparison"]

    assert comparison["run_folds"] == 6
    assert comparison["per_run"] == {"btc_nested_v1": 2, "btc_nested_seed_142": 3}
    assert comparison["mtst_beat_both_baselines"] == 5
    assert comparison["runs_with_majority"] == 2


def test_a_single_run_still_aggregates_with_zero_spread(tmp_path):
    runs = [wf_diagnostics.load_run(write_run(tmp_path / "only"))]
    across = wf_diagnostics.aggregate(runs)["per_model"]["mtst"]["net_return"]["across_runs"]
    assert across["std"] == 0.0
    assert across["mean"] == across["min"] == across["max"]


# --- 5. the CLI ---------------------------------------------------------------
def test_the_cli_audits_and_aggregates_and_writes_both_files(tmp_path, capsys):
    a = write_run(tmp_path / "btc_nested_v1", seed=42)
    b = write_run(tmp_path / "btc_nested_seed_142", seed=142)
    out = tmp_path / "diag"

    assert wf_diagnostics.main([str(a), str(b), "--out", str(out)]) == 0

    printed = capsys.readouterr().out
    assert "# Walk-forward regime diagnostics" in printed
    assert "btc_nested_seed_142" in printed

    payload = json.loads((out / wf_diagnostics.REPORT_JSON).read_text())
    assert payload["comparability_problems"] == []
    assert payload["sealed_test_evaluated"] is False
    assert payload["aggregated_from"] == "outer_validation"
    assert [run["integrity_problems"] for run in payload["runs"]] == [[], []]
    assert payload["summary"]["runs"] == ["btc_nested_v1", "btc_nested_seed_142"]
    # The file on disk is the report that was printed, not a second rendering.
    assert (out / wf_diagnostics.REPORT_MD).read_text().strip() == printed.strip()


def test_the_cli_exits_nonzero_and_withholds_the_aggregate_on_a_broken_run(tmp_path, capsys):
    """A headline over an artifact that failed its audit is the thing to distrust."""
    good = write_run(tmp_path / "good", seed=42)
    bad = edit(write_run(tmp_path / "bad", seed=142), lambda p: p.update(test_evaluated=True))
    out = tmp_path / "diag"

    assert wf_diagnostics.main([str(good), str(bad), "--out", str(out)]) == 1

    printed = capsys.readouterr().out
    assert "## Integrity problems" in printed
    assert "Outer validation across runs" not in printed
    assert json.loads((out / wf_diagnostics.REPORT_JSON).read_text())["summary"] is None


def test_the_cli_withholds_the_aggregate_when_runs_are_not_comparable(tmp_path, capsys):
    a = write_run(tmp_path / "a", folds=3)
    b = write_run(tmp_path / "b", folds=2)

    assert wf_diagnostics.main([str(a), str(b)]) == 1
    printed = capsys.readouterr().out
    assert "## Runs are not comparable" in printed
    assert "averaging them would compare different blocks" in printed


def test_two_runs_with_the_same_directory_name_are_refused(tmp_path):
    a = write_run(tmp_path / "one" / "btc_nested_v1")
    b = write_run(tmp_path / "two" / "btc_nested_v1")
    with pytest.raises(SystemExit, match="same name"):
        wf_diagnostics.main([str(a), str(b)])


def test_the_report_says_what_the_numbers_are_not(tmp_path, capsys):
    """The tool must not let a seed study read as an out-of-sample result."""
    wf_diagnostics.main([str(write_run(tmp_path / "run"))])
    printed = capsys.readouterr().out

    assert "## Limitations" in printed
    assert "not out-of-sample performance" in printed
    assert "not a claim of profitability" in printed
    assert "sealed test block remains unopened" in printed
    assert "nothing was fitted on them" in printed
    assert "coincidence in the data" in printed


def test_the_report_shows_the_geometry_the_runs_share(tmp_path, capsys):
    _, plans = plan_for()
    wf_diagnostics.main([str(write_run(tmp_path / "run"))])
    printed = capsys.readouterr().out

    for plan in plans:
        assert f"[{plan.outer.start}, {plan.outer.end})" in printed
        assert f"[{plan.inner.start}, {plan.inner.end})" in printed


# --- 6. compatibility with what nn.walkforward actually writes -----------------
@pytest.fixture(scope="module")
def real_runs(tmp_path_factory):
    """Two genuine walk-forward runs, differing only in --seed.

    Slow next to the synthetic fixtures and worth it: it is the only check that
    the reader still matches the writer. If the artifact schema changes, this
    fails and the hand-written fixtures above do not.
    """
    workspace = tmp_path_factory.mktemp("real_wf")
    dataset = workspace / "ds.parquet"
    frame, metadata = build_dataset(
        generate_candles(rows=1200, seed=17),
        FeatureSpec(),
        TargetSpec(horizon=4),
        exchange="synthetic",
        pair="SYNTH/USDT",
        timeframe="1h",
    )
    save_dataset(dataset, frame, metadata)

    directories = []
    for seed in (42, 142):
        out = workspace / f"synthetic_nested_seed_{seed}"
        assert (
            walkforward.main(
                [
                    "--dataset",
                    str(dataset),
                    "--out",
                    str(out),
                    "--folds",
                    "2",
                    "--seed",
                    str(seed),
                    *TINY,
                ]
            )
            == 0
        )
        directories.append(out)
    return directories


def test_a_real_walkforward_artifact_loads_and_audits_clean(real_runs):
    for directory in real_runs:
        run = wf_diagnostics.load_run(directory)
        assert run.n_folds == 2
        assert wf_diagnostics.audit_run(run) == [], f"{run.name} failed its audit"


def test_real_runs_that_differ_only_in_seed_are_comparable(real_runs):
    runs = [wf_diagnostics.load_run(directory) for directory in real_runs]
    assert [run.seed for run in runs] == [42, 142]
    assert wf_diagnostics.compare_runs(runs) == []


def test_the_cli_reports_on_real_artifacts(real_runs, tmp_path, capsys):
    out = tmp_path / "diag"
    assert wf_diagnostics.main([str(d) for d in real_runs] + ["--out", str(out)]) == 0

    printed = capsys.readouterr().out
    assert "Outer validation across runs" in printed
    assert "Selection stability" in printed

    payload = json.loads((out / wf_diagnostics.REPORT_JSON).read_text())
    assert payload["summary"]["folds"] == 2
    assert set(payload["summary"]["per_model"]) == set(walkforward.MODELS)
    for run in payload["runs"]:
        assert run["integrity_problems"] == []


def test_the_aggregate_matches_the_summaries_the_runs_recorded(real_runs):
    """Cross-check: each run's own across-fold mean is what the tool reports.

    ``nn.walkforward`` already computes a per-run mean; the diagnostics recompute
    it from the folds. They must agree, or one of the two is reading the wrong
    block.
    """
    runs = [wf_diagnostics.load_run(directory) for directory in real_runs]
    summary = wf_diagnostics.aggregate(runs)

    for run in runs:
        for model in walkforward.MODELS:
            for metric in walkforward.SUMMARY_METRICS:
                recorded = run.summary["per_model"][model][metric]["mean"]
                reported = summary["per_model"][model][metric]["per_run_mean"][run.name]
                assert reported == pytest.approx(recorded, abs=1e-6)


# --- 7. dataset-backed regime diagnostics -------------------------------------
#
# The processed dataset and the raw candles are built here, per test, from
# generated candles. Nothing reads a committed dataset, and the assertions are
# on exact numbers computed independently of the code under test.

RAW_ROWS = 900


def build_dataset_pair(
    directory: Path, *, rows: int = RAW_ROWS, seed: int = 5, horizon: int = 6
):
    """A processed dataset and the raw candles it was built from."""
    directory.mkdir(parents=True, exist_ok=True)
    candles = generate_candles(rows=rows, seed=seed)
    raw_path = directory / "raw.parquet"
    candles.to_parquet(raw_path, index=False)

    frame, meta = build_dataset(
        candles,
        FeatureSpec(),
        TargetSpec(horizon=horizon),
        exchange="synthetic",
        pair="SYNTH/USDT",
        timeframe="1h",
    )
    processed = directory / "processed.parquet"
    save_dataset(processed, frame, meta)
    return processed, raw_path, frame, meta


@pytest.fixture(scope="module")
def dataset_pair(tmp_path_factory):
    return build_dataset_pair(tmp_path_factory.mktemp("regime_data"))


def artifact_for(directory: Path, frame, meta, *, seed: int = 42, folds: int = 3) -> Path:
    """A run whose geometry is planned over the real processed dataset's rows."""
    return write_run(
        directory,
        seed=seed,
        n_rows=len(frame),
        folds=folds,
        dataset={
            "rows": len(frame),
            "pair": meta.pair,
            "timeframe": meta.timeframe,
            "exchange": meta.exchange,
            "start": meta.start,
            "end": meta.end,
            "feature_names": list(meta.feature_names),
            "feature_spec": dict(meta.feature_spec),
            "target_spec": dict(meta.target_spec),
        },
    )


def research_for(dataset_pair, frame):
    processed, _, _, _ = dataset_pair
    return regime.load_research_frame(
        processed, sealed_test_start=sealed_test_start(len(frame), 0.70, 0.15)
    )


def test_the_research_frame_stops_at_the_sealed_boundary(dataset_pair):
    """Sealed rows are absent from the object, not merely unread."""
    processed, _, frame, _ = dataset_pair
    boundary = sealed_test_start(len(frame), 0.70, 0.15)
    research = regime.load_research_frame(processed, sealed_test_start=boundary)

    assert len(research.frame) == boundary
    assert boundary < len(frame), "the fixture must actually have sealed rows to withhold"
    # The last row in the frame is the last research row, by index and by date.
    assert research.frame["date"].iloc[-1] == pd.to_datetime(
        frame["date"].iloc[boundary - 1], utc=True
    )


def test_a_block_at_or_past_the_boundary_is_refused(dataset_pair):
    processed, _, frame, _ = dataset_pair
    boundary = sealed_test_start(len(frame), 0.70, 0.15)
    research = regime.load_research_frame(processed, sealed_test_start=boundary)

    with pytest.raises(regime.RegimeDataError, match="beyond the sealed"):
        research.block(boundary - 10, boundary + 1)
    # Right up to the boundary is fine; one past it is not.
    assert len(research.block(boundary - 10, boundary)) == 10


def test_block_statistics_use_only_the_rows_of_that_block(dataset_pair):
    """Every statistic recomputed independently from the same slice."""
    processed, _, frame, _ = dataset_pair
    boundary = sealed_test_start(len(frame), 0.70, 0.15)
    research = regime.load_research_frame(processed, sealed_test_start=boundary)

    start, end = 400, 520
    stats = regime.block_statistics(research, start, end)
    block = frame.iloc[start:end]
    future = block["future_return"].to_numpy(dtype=float)

    assert stats["row_range"] == [start, end]
    assert stats["rows"] == end - start
    assert stats["future_return"]["mean"] == pytest.approx(float(future.mean()), abs=1e-8)
    assert stats["future_return"]["median"] == pytest.approx(
        float(np.median(future)), abs=1e-8
    )
    assert stats["future_return"]["std"] == pytest.approx(float(future.std(ddof=1)), abs=1e-8)
    assert stats["future_return"]["mean_abs"] == pytest.approx(
        float(np.abs(future).mean()), abs=1e-8
    )
    assert stats["future_return"]["fraction_positive"] == pytest.approx(
        float((future > 0).mean()), abs=1e-8
    )
    assert stats["future_return"]["fraction_negative"] == pytest.approx(
        float((future < 0).mean()), abs=1e-8
    )


def test_feature_statistics_are_exact_including_the_percentiles(dataset_pair):
    processed, _, frame, _ = dataset_pair
    boundary = sealed_test_start(len(frame), 0.70, 0.15)
    research = regime.load_research_frame(processed, sealed_test_start=boundary)

    start, end = 300, 480
    stats = regime.block_statistics(research, start, end)["features"]
    block = frame.iloc[start:end]

    for name, wanted in regime.FEATURE_STATS.items():
        values = block[name].to_numpy(dtype=float)
        assert set(stats[name]) == set(wanted), f"{name} reported the wrong statistics"
        if "mean" in wanted:
            assert stats[name]["mean"] == pytest.approx(float(values.mean()), abs=1e-8)
        if "median" in wanted:
            assert stats[name]["median"] == pytest.approx(float(np.median(values)), abs=1e-8)
        if "std" in wanted:
            assert stats[name]["std"] == pytest.approx(float(values.std(ddof=1)), abs=1e-8)
        if "p90" in wanted:
            assert stats[name]["p90"] == pytest.approx(
                float(np.percentile(values, 90)), abs=1e-8
            )
        if "p90_abs" in wanted:
            assert stats[name]["p90_abs"] == pytest.approx(
                float(np.percentile(np.abs(values), 90)), abs=1e-8
            )
        if "fraction_positive" in wanted:
            assert stats[name]["fraction_positive"] == pytest.approx(
                float((values > 0).mean()), abs=1e-8
            )


def test_the_target_distribution_is_exact(dataset_pair):
    processed, _, frame, _ = dataset_pair
    boundary = sealed_test_start(len(frame), 0.70, 0.15)
    research = regime.load_research_frame(processed, sealed_test_start=boundary)

    start, end = 250, 500
    distribution = regime.block_statistics(research, start, end)["target_distribution"]
    block = frame.iloc[start:end]

    assert set(distribution) == {c.value for c in CLASS_ORDER}
    assert sum(entry["count"] for entry in distribution.values()) == end - start
    assert sum(entry["fraction"] for entry in distribution.values()) == pytest.approx(1.0)

    for index, klass in enumerate(CLASS_ORDER):
        selected = block[block["target"] == index]
        entry = distribution[klass.value]
        assert entry["count"] == len(selected)
        assert entry["fraction"] == pytest.approx(len(selected) / (end - start), abs=1e-8)
        if len(selected):
            returns = selected["future_return"].to_numpy(dtype=float)
            assert entry["mean_future_return"] == pytest.approx(
                float(returns.mean()), abs=1e-8
            )
            assert entry["median_future_return"] == pytest.approx(
                float(np.median(returns)), abs=1e-8
            )


def test_a_dataset_that_does_not_match_the_artifact_is_refused(dataset_pair, tmp_path):
    """Row indices only mean a candle against the dataset they were recorded on."""
    processed, _, frame, meta = dataset_pair
    boundary = sealed_test_start(len(frame), 0.70, 0.15)

    with pytest.raises(regime.RegimeDataError, match="[Rr]ow indices would address"):
        regime.load_research_frame(
            processed, sealed_test_start=boundary, expected_rows=len(frame) + 1
        )
    with pytest.raises(regime.RegimeDataError, match="feature contract differs"):
        regime.load_research_frame(
            processed,
            sealed_test_start=boundary,
            expected_features=["not", "these", "columns"],
        )
    with pytest.raises(regime.RegimeDataError, match="target spec differs"):
        regime.load_research_frame(
            processed,
            sealed_test_start=boundary,
            expected_target_spec={**dict(meta.target_spec), "horizon": 99},
        )
    # The matching contract is accepted, so the test above can fail.
    regime.load_research_frame(
        processed,
        sealed_test_start=boundary,
        expected_rows=len(frame),
        expected_features=list(meta.feature_names),
        expected_target_spec=dict(meta.target_spec),
    )


# --- 8. raw OHLCV alignment ---------------------------------------------------
def test_raw_candles_are_matched_on_timestamps_not_position(dataset_pair):
    """The processed dataset dropped warm-up rows, so the offset is real."""
    processed, raw_path, frame, _ = dataset_pair
    boundary = sealed_test_start(len(frame), 0.70, 0.15)
    research = regime.load_research_frame(processed, sealed_test_start=boundary)
    raw = regime.load_raw_ohlcv(raw_path)

    start, end = 200, 260
    timestamps = research.block(start, end)["date"]
    aligned = regime.align_raw(raw, timestamps, "1h")

    assert list(aligned.index) == list(pd.to_datetime(timestamps, utc=True))
    # Positional alignment would have picked different candles entirely.
    positional = raw.iloc[start:end]
    assert list(positional.index) != list(
        aligned.index
    ), "the fixture must have a warm-up offset, or this test proves nothing"


def test_a_missing_raw_timestamp_fails_closed(dataset_pair, tmp_path):
    processed, raw_path, frame, _ = dataset_pair
    boundary = sealed_test_start(len(frame), 0.70, 0.15)
    research = regime.load_research_frame(processed, sealed_test_start=boundary)
    timestamps = research.block(200, 260)["date"]

    raw = pd.read_parquet(raw_path)
    holed = raw[raw["date"] != timestamps.iloc[5]]
    path = tmp_path / "holed.parquet"
    holed.to_parquet(path, index=False)

    with pytest.raises(regime.RegimeDataError, match="no raw candle"):
        regime.align_raw(regime.load_raw_ohlcv(path), timestamps, "1h")


def test_duplicate_raw_timestamps_fail_closed(dataset_pair, tmp_path):
    _, raw_path, _, _ = dataset_pair
    raw = pd.read_parquet(raw_path)
    path = tmp_path / "duplicated.parquet"
    pd.concat([raw, raw.iloc[[7]]], ignore_index=True).to_parquet(path, index=False)

    with pytest.raises(regime.RegimeDataError, match="duplicate timestamps"):
        regime.load_raw_ohlcv(path)


def test_an_incompatible_timeframe_fails_closed(dataset_pair):
    processed, raw_path, frame, _ = dataset_pair
    boundary = sealed_test_start(len(frame), 0.70, 0.15)
    research = regime.load_research_frame(processed, sealed_test_start=boundary)
    timestamps = research.block(200, 240)["date"]

    with pytest.raises(regime.RegimeDataError, match="spaced .* apart"):
        regime.align_raw(regime.load_raw_ohlcv(raw_path), timestamps, "4h")


def test_raw_without_ohlcv_columns_fails_closed(tmp_path):
    path = tmp_path / "not_ohlcv.parquet"
    pd.DataFrame({"date": pd.date_range("2023-01-01", periods=3, tz="UTC")}).to_parquet(
        path, index=False
    )
    with pytest.raises(regime.RegimeDataError, match="missing OHLCV column"):
        regime.load_raw_ohlcv(path)


def test_raw_block_statistics_are_exact(dataset_pair):
    processed, raw_path, frame, _ = dataset_pair
    boundary = sealed_test_start(len(frame), 0.70, 0.15)
    research = regime.load_research_frame(processed, sealed_test_start=boundary)
    raw = regime.load_raw_ohlcv(raw_path)

    start, end = 300, 400
    timestamps = research.block(start, end)["date"]
    stats = regime.raw_block_statistics(raw, timestamps, "1h")

    candles = raw.loc[pd.to_datetime(timestamps, utc=True).to_numpy()]
    close = candles["close"].to_numpy(dtype=float)
    returns = close[1:] / close[:-1] - 1.0

    assert stats["candles"] == end - start
    assert stats["start_close"] == pytest.approx(float(close[0]), abs=1e-8)
    assert stats["end_close"] == pytest.approx(float(close[-1]), abs=1e-8)
    assert stats["market_return"] == pytest.approx(float(close[-1] / close[0] - 1), abs=1e-8)
    assert stats["mean_candle_return"] == pytest.approx(float(returns.mean()), abs=1e-8)
    assert stats["candle_return_std"] == pytest.approx(float(returns.std(ddof=1)), abs=1e-8)
    assert stats["annualised_volatility"] == pytest.approx(
        float(returns.std(ddof=1) * np.sqrt(24 * 365)), abs=1e-6
    )
    assert stats["mean_abs_candle_return"] == pytest.approx(
        float(np.abs(returns).mean()), abs=1e-8
    )
    peak = np.maximum.accumulate(close)
    assert stats["max_drawdown"] == pytest.approx(float((1 - close / peak).max()), abs=1e-8)
    assert stats["positive_candle_fraction"] + stats["negative_candle_fraction"] <= 1.0


def test_raw_statistics_skip_returns_across_a_market_gap(dataset_pair, tmp_path):
    """A missing exchange candle is not a price move."""
    processed, raw_path, frame, _ = dataset_pair
    boundary = sealed_test_start(len(frame), 0.70, 0.15)
    research = regime.load_research_frame(processed, sealed_test_start=boundary)
    timestamps = research.block(300, 360)["date"]

    raw = regime.load_raw_ohlcv(raw_path)
    # Drop a candle that is *not* in the block, but sits inside its span, so the
    # block's own timestamps still all resolve while one pair is non-adjacent.
    gapped = raw.drop(index=timestamps.iloc[10])
    remaining = timestamps[timestamps != timestamps.iloc[10]]

    stats = regime.raw_block_statistics(gapped, remaining, "1h")
    assert stats["gap_pairs_skipped"] == 1
    assert stats["candles"] == len(remaining)


# --- 9. best/worst selection, seed-invariant market, identity -----------------
def stability_runs(tmp_path, net_by_fold):
    """Runs sharing geometry, whose per-fold MTST returns are dictated per seed."""
    runs = []
    for index, (name, values) in enumerate(net_by_fold.items()):
        directory = write_run(
            tmp_path / name,
            seed=42 + 100 * index,
            folds=len(values),
            returns={
                "majority_baseline": [-0.01] * len(values),
                "momentum_baseline": [0.0] * len(values),
                "mtst": values,
            },
        )
        runs.append(wf_diagnostics.load_run(directory))
    return runs


def test_best_and_worst_folds_come_from_the_data(tmp_path):
    """Fold 1 wins and fold 2 loses only because the numbers say so."""
    runs = stability_runs(
        tmp_path,
        {"run_a": [0.10, 0.90, -0.30], "run_b": [0.20, 0.70, -0.10]},
    )
    stability = wf_diagnostics.fold_model_stability(runs)
    assert wf_diagnostics.best_and_worst(stability) == (1, 2)

    # Reverse the ordering and the answer follows the data, not a constant.
    reversed_runs = stability_runs(
        tmp_path / "reversed",
        {"run_a": [0.90, 0.10, -0.30], "run_b": [0.70, 0.20, -0.10]},
    )
    assert wf_diagnostics.best_and_worst(
        wf_diagnostics.fold_model_stability(reversed_runs)
    ) == (0, 2)


def test_fold_stability_reports_every_requested_quantity(tmp_path):
    runs = stability_runs(tmp_path, {"a": [0.1, -0.2, 0.3], "b": [0.3, 0.2, 0.5]})
    stability = wf_diagnostics.fold_model_stability(runs)

    assert [entry["fold"] for entry in stability] == [0, 1, 2]
    fold_0 = stability[0]
    assert fold_0["net_return"]["mean"] == pytest.approx(0.2)
    assert fold_0["net_return"]["median"] == pytest.approx(0.2)
    assert fold_0["net_return"]["positive_seeds"] == 2
    assert fold_0["net_return"]["seeds"] == 2
    # Fold 1 has one negative seed out of two.
    assert stability[1]["net_return"]["positive_seeds"] == 1
    for field in ("directional_accuracy", "coverage", "n_trades", "threshold", "best_epoch"):
        assert {"mean", "std", "min", "max"} <= set(fold_0[field])


def test_market_statistics_are_identical_across_seeds(dataset_pair, tmp_path):
    """The market in a fold does not depend on which seed trained on it."""
    processed, raw_path, frame, meta = dataset_pair
    runs = [
        wf_diagnostics.load_run(artifact_for(tmp_path / name, frame, meta, seed=seed))
        for name, seed in (("seed_42", 42), ("seed_142", 142), ("seed_242", 242))
    ]
    assert wf_diagnostics.compare_runs(runs) == []

    research = regime.load_research_frame(
        processed, sealed_test_start=runs[0].sealed_test_start
    )
    raw = regime.load_raw_ohlcv(raw_path)
    reports = [wf_diagnostics.regime_report([run], research, raw) for run in runs]

    assert reports[0] == reports[1] == reports[2]
    assert all("market" in block for block in reports[0])


def test_a_differing_target_spec_blocks_aggregation(dataset_pair, tmp_path):
    processed, _, frame, meta = dataset_pair
    good = wf_diagnostics.load_run(artifact_for(tmp_path / "good", frame, meta))
    other = artifact_for(tmp_path / "other", frame, meta, seed=142)
    edit(other, lambda p: p["dataset"]["target_spec"].update(horizon=12))

    problems = wf_diagnostics.compare_runs([good, wf_diagnostics.load_run(other)])
    assert any("target_spec" in problem for problem in problems)


def test_a_differing_feature_contract_blocks_aggregation(dataset_pair, tmp_path):
    processed, _, frame, meta = dataset_pair
    good = wf_diagnostics.load_run(artifact_for(tmp_path / "good", frame, meta))
    other = artifact_for(tmp_path / "other", frame, meta, seed=142)
    edit(other, lambda p: p["dataset"]["feature_names"].append("an_extra_feature"))

    problems = wf_diagnostics.compare_runs([good, wf_diagnostics.load_run(other)])
    assert any("feature_names" in problem for problem in problems)


def test_seed_only_differences_do_not_block_aggregation(dataset_pair, tmp_path):
    """The whole point is comparing seeds; the seed itself must not disqualify them."""
    _, _, frame, meta = dataset_pair
    runs = [
        wf_diagnostics.load_run(artifact_for(tmp_path / f"s{seed}", frame, meta, seed=seed))
        for seed in (42, 142, 242, 342, 442)
    ]
    assert {run.seed for run in runs} == {42, 142, 242, 342, 442}
    assert wf_diagnostics.compare_runs(runs) == []


# --- 10. LONG / SHORT attribution ---------------------------------------------
def prediction_frame(**overrides) -> pd.DataFrame:
    """Six samples with hand-chosen actions, so the attribution is checkable by hand."""
    short_idx, hold_idx, long_idx = range(len(CLASS_ORDER))
    data = {
        "fold": [0] * 6,
        "seed": [42] * 6,
        "row_index": [100, 101, 102, 103, 104, 105],
        "timestamp": pd.date_range("2023-01-01", periods=6, freq="h", tz="UTC"),
        "true_target": [long_idx, hold_idx, short_idx, hold_idx, long_idx, hold_idx],
        "future_return": [0.05, 0.01, -0.04, 0.0, -0.02, 0.03],
        "p_short": [0.1, 0.2, 0.8, 0.2, 0.2, 0.1],
        "p_hold": [0.1, 0.6, 0.1, 0.6, 0.1, 0.1],
        "p_long": [0.8, 0.2, 0.1, 0.2, 0.7, 0.8],
        "selected_action": [long_idx, hold_idx, short_idx, hold_idx, long_idx, long_idx],
        "threshold": [0.6] * 6,
    }
    data.update(overrides)
    return pd.DataFrame(data)


def test_attribution_is_exact_and_uses_the_shared_cost_model():
    """Hand-computed: horizon 1 means every non-HOLD sample is its own trade."""
    spec = TargetSpec(horizon=1)
    cost = spec.cost_threshold
    result = regime.direction_attribution(prediction_frame(), spec)

    # LONG at rows 100 (+0.05), 104 (-0.02), 105 (+0.03); SHORT at 102 (-0.04 -> +0.04).
    long_returns = [0.05 - cost, -0.02 - cost, 0.03 - cost]
    short_returns = [0.04 - cost]

    assert result["long"]["trades"] == 3
    assert result["short"]["trades"] == 1
    assert result["hold_samples"] == 2
    assert result["long"]["mean_net_return"] == pytest.approx(np.mean(long_returns))
    assert result["long"]["median_net_return"] == pytest.approx(np.median(long_returns))
    assert result["long"]["cumulative_contribution"] == pytest.approx(sum(long_returns))
    assert result["long"]["hit_rate"] == pytest.approx(2 / 3, abs=1e-6)
    assert result["short"]["cumulative_contribution"] == pytest.approx(sum(short_returns))
    assert result["short"]["hit_rate"] == pytest.approx(1.0)
    assert result["long_coverage"] == pytest.approx(3 / 6, abs=1e-6)
    assert result["short_coverage"] == pytest.approx(1 / 6, abs=1e-6)


def test_attribution_totals_match_the_shared_trade_generator():
    """The split is a partition of exactly the trades nn.evaluate would take."""
    spec = TargetSpec(horizon=2)
    predictions = prediction_frame()
    result = regime.direction_attribution(predictions, spec)

    _, directions, returns = ev.realised_trades(
        predictions["selected_action"].to_numpy(np.int64),
        predictions["future_return"].to_numpy(float),
        spec,
    )
    assert result["long"]["trades"] + result["short"]["trades"] == len(returns)
    assert result["long"]["cumulative_contribution"] + result["short"][
        "cumulative_contribution"
    ] == pytest.approx(float(returns.sum()), abs=1e-8)
    assert result["long"]["trades"] == int((directions > 0).sum())


def test_attribution_generates_trades_per_run_not_across_runs():
    """One model walking one block: a second seed must not suppress the first's trade."""
    spec = TargetSpec(horizon=6)
    one_seed = prediction_frame()
    two_seeds = pd.concat([one_seed, prediction_frame(seed=[142] * 6)], ignore_index=True)

    single = regime.direction_attribution(one_seed, spec)
    doubled = regime.direction_attribution(two_seeds, spec)
    assert doubled["long"]["trades"] == 2 * single["long"]["trades"]
    assert doubled["short"]["trades"] == 2 * single["short"]["trades"]


def test_a_prediction_file_reaching_the_sealed_block_is_refused(tmp_path):
    path = tmp_path / "outer_predictions.parquet"
    prediction_frame(row_index=[100, 101, 102, 103, 104, 999]).to_parquet(path, index=False)

    with pytest.raises(regime.RegimeDataError, match="at or beyond the sealed"):
        regime.load_predictions(path, sealed_test_start=500)
    # Below the boundary it loads, so the guard above is doing the work.
    assert len(regime.load_predictions(path, sealed_test_start=1000)) == 6


def test_a_malformed_prediction_file_fails_clearly(tmp_path):
    path = tmp_path / "outer_predictions.parquet"
    prediction_frame().drop(columns=["p_long", "threshold"]).to_parquet(path, index=False)
    with pytest.raises(regime.RegimeDataError, match="missing prediction column"):
        regime.load_predictions(path, sealed_test_start=1000)

    empty = tmp_path / "empty.parquet"
    prediction_frame().iloc[:0].to_parquet(empty, index=False)
    with pytest.raises(regime.RegimeDataError, match="no predictions"):
        regime.load_predictions(empty, sealed_test_start=1000)


def test_absent_predictions_are_reported_not_approximated(tmp_path):
    runs = [wf_diagnostics.load_run(write_run(tmp_path / "legacy"))]
    report = wf_diagnostics.attribution_report(runs, TargetSpec())

    assert report["available"] is False
    assert "outer sample predictions were not persisted" in report["reason"]
    assert report["expected_file"] == walkforward.PREDICTIONS_NAME
    # No numbers are offered in place of the measurement.
    assert "long" not in report and "short" not in report


# --- 11. the regime mode end to end -------------------------------------------
@pytest.fixture(scope="module")
def real_regime_runs(tmp_path_factory):
    """Real walk-forward runs over a real processed dataset, plus its raw candles.

    Three seeds, so per-fold spread across seeds is a real number rather than a
    placeholder, and every artifact is written by ``nn.walkforward`` itself.
    """
    workspace = tmp_path_factory.mktemp("regime_e2e")
    processed, raw_path, frame, _ = build_dataset_pair(workspace, rows=1600, seed=13)

    directories = []
    for seed in (42, 142, 242):
        out = workspace / f"nested_seed_{seed}"
        assert (
            walkforward.main(
                [
                    "--dataset",
                    str(processed),
                    "--out",
                    str(out),
                    "--folds",
                    "3",
                    "--seed",
                    str(seed),
                    *TINY,
                ]
            )
            == 0
        )
        directories.append(out)
    return {
        "runs": directories,
        "dataset": processed,
        "raw": raw_path,
        "rows": len(frame),
        "boundary": sealed_test_start(len(frame), 0.70, 0.15),
    }


def test_walkforward_persists_outer_predictions_below_the_boundary(real_regime_runs):
    """Every persisted row is an outer row, and none is sealed."""
    for directory in real_regime_runs["runs"]:
        predictions = pd.read_parquet(directory / walkforward.PREDICTIONS_NAME)
        assert list(predictions.columns) == list(regime.PREDICTION_COLUMNS)
        assert predictions["row_index"].max() < real_regime_runs["boundary"]

        artifact = json.loads((directory / wf_diagnostics.ARTIFACT_NAME).read_text())
        assert artifact["outer_predictions"] == walkforward.PREDICTIONS_NAME
        # Each persisted row falls inside its own fold's outer block.
        for fold in artifact["folds"]:
            start, end = fold["periods"]["outer_validation"]["row_range"]
            rows = predictions[predictions["fold"] == fold["fold"]]["row_index"]
            assert rows.min() >= start and rows.max() < end


def test_persisted_probabilities_agree_with_the_persisted_action(real_regime_runs):
    """selected_action is the action the reports scored, not a second reading."""
    predictions = pd.read_parquet(real_regime_runs["runs"][0] / walkforward.PREDICTIONS_NAME)
    proba = predictions[["p_short", "p_hold", "p_long"]].to_numpy(dtype=float)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    for threshold, group in predictions.groupby("threshold"):
        expected = ev.signals_from_proba(
            group[["p_short", "p_hold", "p_long"]].to_numpy(dtype=float), float(threshold)
        )
        np.testing.assert_array_equal(group["selected_action"].to_numpy(), expected)


def test_the_regime_cli_reports_every_requested_section(real_regime_runs, tmp_path, capsys):
    out = tmp_path / "btc_regimes_v1"
    exit_code = wf_diagnostics.main(
        [str(d) for d in real_regime_runs["runs"]]
        + [
            "--dataset",
            str(real_regime_runs["dataset"]),
            "--raw",
            str(real_regime_runs["raw"]),
            "--out",
            str(out),
        ]
    )
    assert exit_code == 0

    printed = capsys.readouterr().out
    for heading in (
        "## Integrity",
        "## Dataset / sealed-test status",
        "## Fold geometry",
        "## Market regime statistics",
        "## Outer validation across runs",
        "## Per-fold model stability across seeds",
        "## Best vs worst regime",
        "## LONG vs SHORT attribution",
        "## Candidate hypotheses",
        "## Limitations",
    ):
        assert heading in printed, f"{heading} missing from the report"

    payload = json.loads((out / wf_diagnostics.REPORT_JSON).read_text())
    analysis = payload["analysis"]
    assert payload["sealed_test_evaluated"] is False
    assert analysis["sealed_test_evaluated"] is False
    assert analysis["highest_outer_row"] < real_regime_runs["boundary"]
    assert len(analysis["regime"]) == 3
    assert analysis["best_fold"] != analysis["worst_fold"]
    assert len(analysis["hypotheses"]) == 3
    assert [h["rank"] for h in analysis["hypotheses"]] == [1, 2, 3]
    assert analysis["attribution"]["available"] is True
    assert (
        (out / wf_diagnostics.REPORT_MD)
        .read_text()
        .startswith("# Walk-forward regime diagnostics")
    )


def test_the_regime_cli_never_reads_a_row_at_or_beyond_the_boundary(
    real_regime_runs, tmp_path, monkeypatch
):
    """Spied on the block reads themselves, asserted on row indices."""
    boundary = real_regime_runs["boundary"]
    seen: list[tuple[int, int]] = []
    original = regime.ResearchFrame.block

    def spy(self, start, end):
        seen.append((start, end))
        return original(self, start, end)

    monkeypatch.setattr(regime.ResearchFrame, "block", spy)
    assert (
        wf_diagnostics.main(
            [str(d) for d in real_regime_runs["runs"]]
            + [
                "--dataset",
                str(real_regime_runs["dataset"]),
                "--raw",
                str(real_regime_runs["raw"]),
                "--out",
                str(tmp_path / "diag"),
            ]
        )
        == 0
    )

    assert seen, "the spy must have fired"
    assert max(end for _, end in seen) <= boundary
    # And the frame handed to those reads physically stops at the boundary.
    research = regime.load_research_frame(
        real_regime_runs["dataset"], sealed_test_start=boundary
    )
    assert len(research.frame) == boundary


def test_the_regime_statistics_match_a_direct_computation(real_regime_runs, tmp_path):
    """The reported block statistics equal the same statistics taken by hand."""
    out = tmp_path / "diag"
    wf_diagnostics.main(
        [str(d) for d in real_regime_runs["runs"]]
        + ["--dataset", str(real_regime_runs["dataset"]), "--out", str(out)]
    )
    analysis = json.loads((out / wf_diagnostics.REPORT_JSON).read_text())["analysis"]

    frame = pd.read_parquet(real_regime_runs["dataset"])
    for block in analysis["regime"]:
        start, end = block["row_range"]
        assert end <= real_regime_runs["boundary"]
        rows = frame.iloc[start:end]
        assert block["rows"] == len(rows)
        assert block["future_return"]["mean"] == pytest.approx(
            float(rows["future_return"].mean()), abs=1e-8
        )
        assert block["features"]["atr_norm"]["p90"] == pytest.approx(
            float(np.percentile(rows["atr_norm"], 90)), abs=1e-8
        )


def test_the_attribution_is_exact_on_real_persisted_predictions(real_regime_runs, tmp_path):
    """Recomputed straight from the parquet files, independent of the report."""
    out = tmp_path / "diag"
    wf_diagnostics.main(
        [str(d) for d in real_regime_runs["runs"]]
        + ["--dataset", str(real_regime_runs["dataset"]), "--out", str(out)]
    )
    attribution = json.loads((out / wf_diagnostics.REPORT_JSON).read_text())["analysis"][
        "attribution"
    ]
    assert attribution["available"] is True

    combined = pd.concat(
        [pd.read_parquet(d / walkforward.PREDICTIONS_NAME) for d in real_regime_runs["runs"]],
        ignore_index=True,
    )
    spec = TargetSpec.from_dict(
        json.loads((real_regime_runs["runs"][0] / wf_diagnostics.ARTIFACT_NAME).read_text())[
            "dataset"
        ]["target_spec"]
    )
    expected = regime.direction_attribution(combined, spec)
    assert attribution["overall"] == expected
    assert attribution["overall"]["long"]["trades"] > 0


def test_raw_is_refused_without_a_dataset(real_regime_runs):
    with pytest.raises(SystemExit, match="--raw needs --dataset"):
        wf_diagnostics.main(
            [str(d) for d in real_regime_runs["runs"]]
            + ["--raw", str(real_regime_runs["raw"])]
        )


def test_a_mismatched_dataset_is_refused_by_the_cli(real_regime_runs, tmp_path):
    """Pointing at a differently sized dataset must not silently reindex."""
    other, _, _, _ = build_dataset_pair(tmp_path / "other", rows=1000, seed=99)
    with pytest.raises(SystemExit, match="[Rr]ow indices would address"):
        wf_diagnostics.main(
            [str(d) for d in real_regime_runs["runs"]] + ["--dataset", str(other)]
        )


def test_without_a_dataset_the_report_still_audits_and_says_what_is_missing(
    real_regime_runs, tmp_path, capsys
):
    """The five existing runs stay analysable with no dataset at hand."""
    assert (
        wf_diagnostics.main(
            [str(d) for d in real_regime_runs["runs"]] + ["--out", str(tmp_path / "diag")]
        )
        == 0
    )
    printed = capsys.readouterr().out

    assert "## Per-fold model stability across seeds" in printed
    assert "## Best vs worst regime" in printed
    assert "no regime statistics below" in printed
    assert "## Market regime statistics" not in printed
