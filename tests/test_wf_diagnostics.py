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

import pytest

from chimera.contracts import TargetSpec
from chimera.features import FeatureSpec
from nn import walkforward, wf_diagnostics
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
                "dataset": {"rows": n_rows, "pair": "BTC/USDT", "timeframe": "1h"},
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
    assert "# Walk-forward diagnostics" in printed
    assert "btc_nested_seed_142" in printed

    payload = json.loads((out / "wf_diagnostics.json").read_text())
    assert payload["comparability_problems"] == []
    assert payload["sealed_test_evaluated"] is False
    assert payload["aggregated_from"] == "outer_validation"
    assert [run["integrity_problems"] for run in payload["runs"]] == [[], []]
    assert payload["summary"]["runs"] == ["btc_nested_v1", "btc_nested_seed_142"]
    # The file on disk is the report that was printed, not a second rendering.
    assert (out / "wf_diagnostics.md").read_text().strip() == printed.strip()


def test_the_cli_exits_nonzero_and_withholds_the_aggregate_on_a_broken_run(tmp_path, capsys):
    """A headline over an artifact that failed its audit is the thing to distrust."""
    good = write_run(tmp_path / "good", seed=42)
    bad = edit(write_run(tmp_path / "bad", seed=142), lambda p: p.update(test_evaluated=True))
    out = tmp_path / "diag"

    assert wf_diagnostics.main([str(good), str(bad), "--out", str(out)]) == 1

    printed = capsys.readouterr().out
    assert "## Integrity problems" in printed
    assert "Outer validation across runs" not in printed
    assert json.loads((out / "wf_diagnostics.json").read_text())["summary"] is None


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

    assert "not out-of-sample performance" in printed
    assert "not a claim of profitability" in printed
    assert "sealed test block remains unopened" in printed
    assert "nothing was fitted on them" in printed


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

    payload = json.loads((out / "wf_diagnostics.json").read_text())
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
