"""Safeguards for the research workflow.

The workflow is only worth anything if the sealed test split stays sealed, so
these tests assert that property directly rather than trusting the entrypoints
to be well behaved. The technique throughout is a spy on ``nn.train`` internals:
if ``build_windows`` is never called with the test split, no test row can have
reached a metric, because windowing is the only way to get one there.

A guard test that cannot fail is worse than none, so the "test was not
windowed" assertions are paired with a control that runs the same entrypoint
*without* research mode and checks the test split IS windowed there.
"""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from chimera.contracts import TargetSpec
from chimera.features import FeatureSpec
from nn import experiment, train, walkforward
from nn.data_pipeline import build_dataset, save_dataset
from nn.dataset import Split, chronological_split
from nn.registry import promote
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


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    path = tmp_path_factory.mktemp("research") / "ds.parquet"
    frame, metadata = build_dataset(
        generate_candles(rows=1200, seed=17),
        FeatureSpec(),
        TargetSpec(horizon=4),
        exchange="synthetic",
        pair="SYNTH/USDT",
        timeframe="1h",
    )
    save_dataset(path, frame, metadata)
    return path


@pytest.fixture
def windowed_splits(monkeypatch):
    """Record the name of every split that gets windowed, anywhere."""
    names: list[str] = []
    original = train.build_windows

    def spy(features, targets, split, seq_len, horizon, **kwargs):
        names.append(split.name)
        return original(features, targets, split, seq_len, horizon, **kwargs)

    monkeypatch.setattr(train, "build_windows", spy)
    return names


def only_version(models_dir):
    return next(p for p in models_dir.iterdir() if p.is_dir())


# --- 1. validation-only mode leaves test alone --------------------------------
def test_validation_only_never_windows_the_test_split(dataset, tmp_path, windowed_splits):
    models_dir = tmp_path / "models"
    assert (
        train.main(
            ["--dataset", str(dataset), "--models-dir", str(models_dir), "--validation-only"]
            + TINY
        )
        == 0
    )
    assert "test" not in windowed_splits
    assert set(windowed_splits) == {"train", "validation"}


def test_a_normal_run_does_window_the_test_split(dataset, tmp_path, windowed_splits):
    """The control for the test above: without --validation-only, test IS scored.

    Without this, the guard tests would pass just as happily against an
    entrypoint that had stopped windowing anything at all.
    """
    models_dir = tmp_path / "models"
    train.main(["--dataset", str(dataset), "--models-dir", str(models_dir)] + TINY)
    assert "test" in windowed_splits


def test_validation_only_report_records_that_test_was_not_evaluated(dataset, tmp_path):
    models_dir = tmp_path / "models"
    train.main(
        ["--dataset", str(dataset), "--models-dir", str(models_dir), "--validation-only"]
        + TINY
    )
    report = json.loads((only_version(models_dir) / "report.json").read_text())

    assert report["test"] is None
    assert report["test_evaluated"] is False
    assert report["research_only"] is True
    assert report["promotion"]["eligible"] is False
    # Validation is still fully reported — sealing test costs nothing else.
    assert set(report["validation"]) == {"majority_baseline", "momentum_baseline", "mtst"}


def test_validation_only_says_out_loud_that_test_is_sealed(dataset, tmp_path, capsys):
    train.main(
        [
            "--dataset",
            str(dataset),
            "--models-dir",
            str(tmp_path / "models"),
            "--validation-only",
        ]
        + TINY
    )
    assert "SEALED" in capsys.readouterr().out


def test_promote_cannot_be_combined_with_validation_only(dataset, tmp_path):
    with pytest.raises(SystemExit, match="--promote cannot be combined"):
        train.main(
            [
                "--dataset",
                str(dataset),
                "--models-dir",
                str(tmp_path / "models"),
                "--validation-only",
                "--promote",
            ]
            + TINY
        )


def test_a_research_artifact_cannot_be_promoted_by_hand(dataset, tmp_path):
    """Reaching past the CLI must not get a research model in front of traffic."""
    models_dir = tmp_path / "models"
    train.main(
        ["--dataset", str(dataset), "--models-dir", str(models_dir), "--validation-only"]
        + TINY
    )
    with pytest.raises(ValueError, match="validation-only research run"):
        promote(models_dir, only_version(models_dir).name)


# --- 2. the fitted quantities come from the right rows ------------------------
def test_the_threshold_is_selected_only_from_validation_rows(dataset, tmp_path, monkeypatch):
    data = train.load_research_data(dataset)
    plan = chronological_split(data.n_rows, 0.70, 0.15)
    expected_idx = train.prepare_research_windows(
        data, plan.train, plan.validation, 16
    ).idx_val

    seen: list[np.ndarray] = []
    original = train.ev.select_threshold

    def spy(proba, future_return, target_spec, **kwargs):
        seen.append(np.asarray(future_return))
        return original(proba, future_return, target_spec, **kwargs)

    monkeypatch.setattr(train.ev, "select_threshold", spy)
    train.main(["--dataset", str(dataset), "--models-dir", str(tmp_path / "m")] + TINY)

    assert len(seen) == 1, "the threshold must be selected exactly once"
    np.testing.assert_allclose(seen[0], data.future_return[expected_idx])
    assert expected_idx.min() >= plan.validation.start
    assert expected_idx.max() < plan.validation.end


def test_the_scaler_is_fitted_only_on_training_rows(dataset, tmp_path, monkeypatch):
    data = train.load_research_data(dataset)
    plan = chronological_split(data.n_rows, 0.70, 0.15)

    fitted: list[np.ndarray] = []
    original = train.StandardScaler.fit

    def spy(self, X):
        fitted.append(np.asarray(X))
        return original(self, X)

    monkeypatch.setattr(train.StandardScaler, "fit", spy)
    train.main(["--dataset", str(dataset), "--models-dir", str(tmp_path / "m")] + TINY)

    assert len(fitted) == 1, "the scaler must be fitted once"
    np.testing.assert_allclose(fitted[0], data.features[plan.train.start : plan.train.end])


def test_prepare_research_windows_refuses_validation_before_training(dataset):
    data = train.load_research_data(dataset)
    with pytest.raises(ValueError, match="validation must begin at or after"):
        train.prepare_research_windows(
            data, Split("train", 0, 600), Split("validation", 400, 700), 16
        )


def test_research_windows_stay_inside_their_split_and_segment(dataset):
    """Neither the input window nor the label horizon may leave the fold."""
    data = train.load_research_data(dataset)
    train_split, val_split = Split("train", 0, 800), Split("validation", 800, 1000)
    seq_len, horizon = 16, data.target_spec.horizon

    prepared = train.prepare_research_windows(data, train_split, val_split, seq_len)
    idx = prepared.idx_val
    assert idx.min() - (seq_len - 1) >= val_split.start, "window reaches into training"
    assert idx.max() + horizon < val_split.end, "label reaches past the fold"


def test_research_windows_respect_market_data_gaps():
    """Regression: walk-forward used to skip segment_ids and bridge outages.

    Every research entrypoint now windows through ``prepare_research_windows``,
    so the gap safety that ``nn.train`` enforces applies to all of them.
    """
    rng = np.random.default_rng(3)
    n = 400
    segments = np.array([0] * 200 + [1] * 200, dtype=np.int64)
    data = train.ResearchData(
        ds_meta=train.DatasetMetadata(timeframe="1h"),
        feature_names=["ema_cross", "b"],
        feature_spec=FeatureSpec(),
        target_spec=TargetSpec(horizon=4),
        features=rng.normal(size=(n, 2)),
        targets=rng.integers(0, 3, size=n),
        future_return=rng.normal(size=n) * 0.01,
        segment_ids=segments,
        dates=np.arange(n),
        candles_per_year=24 * 365,
    )
    prepared = train.prepare_research_windows(
        data, Split("train", 0, 250), Split("validation", 250, 400), seq_len=8
    )
    for i in prepared.idx_val:
        assert segments[i - 7] == segments[i], "window bridges a market-data gap"
        assert segments[i + 4] == segments[i], "label bridges a market-data gap"


# --- 3. the experiment runner -------------------------------------------------
def experiment_args(dataset, out, extra=()):
    return [
        "--dataset",
        str(dataset),
        "--out",
        str(out),
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
        *extra,
    ]


def test_the_experiment_runner_never_scores_test(dataset, tmp_path, windowed_splits):
    out = tmp_path / "exp"
    assert experiment.main(experiment_args(dataset, out, ["--seed", "1", "2"])) == 0

    assert set(windowed_splits) == {"train", "validation"}, "the spy must have fired"
    summary = json.loads((out / "experiments.json").read_text())
    assert summary["test_evaluated"] is False
    assert summary["n_runs"] == 2
    for run in summary["runs"]:
        assert "test" not in run


def test_the_experiment_runner_keeps_failed_runs_visible(dataset, tmp_path):
    """A configuration that raises must be reported, not silently dropped.

    ``d_model=6`` with ``n_heads=4`` is rejected by ``MTSTConfig``; the run
    beside it is valid, so exactly one of the two must fail.
    """
    out = tmp_path / "exp"
    assert (
        experiment.main(
            experiment_args(dataset, out, ["--d-model", "6", "16", "--n-heads", "4"])
        )
        == 0
    )
    summary = json.loads((out / "experiments.json").read_text())

    assert summary["n_runs"] == 2
    assert summary["n_failed"] == 1
    failed = [r for r in summary["runs"] if r["status"] == "failed"]
    assert "divisible" in failed[0]["error"]

    rows = list(csv.DictReader((out / "experiments.csv").open()))
    assert len(rows) == 2, "the CSV must carry the failure too"
    assert {r["status"] for r in rows} == {"ok", "failed"}


def test_the_experiment_runner_ranks_by_the_declared_objective(dataset, tmp_path):
    out = tmp_path / "exp"
    experiment.main(
        experiment_args(dataset, out, ["--seed", "1", "2", "3", "--objective", "macro_f1"])
    )
    summary = json.loads((out / "experiments.json").read_text())

    assert summary["objective"] == "macro_f1"
    scores = [r["objective"] for r in summary["runs"] if r["status"] == "ok"]
    assert scores == sorted(scores, reverse=True)
    assert summary["best"]["objective"] == scores[0]
    # Baselines travel with every run, so a ranking can be read honestly.
    for run in summary["runs"]:
        assert set(run["validation"]) == {"majority_baseline", "momentum_baseline", "mtst"}


def test_every_grid_dimension_the_docs_promise_is_wired_up(dataset, tmp_path):
    parsed = experiment.build_argparser().parse_args(
        experiment_args(
            dataset, tmp_path, ["--dropout", "0.1", "0.2", "--num-layers", "1", "2"]
        )
    )
    grid = experiment.build_grid(
        {name: getattr(parsed, name) for name in experiment.GRID_DIMENSIONS},
        train.RunConfig(),
    )
    assert len(grid) == 4
    assert {(c.dropout, c.num_layers) for c in grid} == {
        (0.1, 1),
        (0.1, 2),
        (0.2, 1),
        (0.2, 2),
    }


# --- 4. walk-forward ----------------------------------------------------------
def test_expanding_folds_are_chronological_and_grow_forward():
    folds = walkforward.plan_expanding_folds(
        n_rows=1000, folds=4, min_train=400, val_size=100, step=125
    )
    assert len(folds) == 4

    previous_train_end = 0
    for train_split, val_split in folds:
        # Validation is strictly after training, always.
        assert val_split.start >= train_split.end
        assert val_split.end > val_split.start
        # Expanding: training always starts at the beginning and only grows.
        assert train_split.start == 0
        assert train_split.end > previous_train_end
        previous_train_end = train_split.end

    # The forbidden direction: no fold may train on rows a later fold validates,
    # and no fold's validation rows may appear in an earlier fold's training.
    for i, (_, val_split) in enumerate(folds):
        for j, (train_split, _) in enumerate(folds):
            if j <= i:
                assert train_split.end <= val_split.start


def test_expanding_folds_refuse_a_dataset_that_is_too_short():
    with pytest.raises(ValueError, match=r"need 999 rows .*the dataset has 500"):
        walkforward.plan_expanding_folds(
            n_rows=500, folds=4, min_train=400, val_size=200, step=133
        )


def test_default_fold_sizes_fit_the_dataset(dataset):
    """The defaults must produce a runnable plan, not an immediate error."""
    data = train.load_research_data(dataset)
    args = walkforward.build_argparser().parse_args(["--dataset", str(dataset)])
    min_train, val_size, step = walkforward.resolve_sizes(args, data.n_rows)
    folds = walkforward.plan_expanding_folds(
        data.n_rows, args.folds, min_train, val_size, step
    )
    assert len(folds) == args.folds
    assert folds[-1][1].end <= data.n_rows


def test_walkforward_never_scores_test(dataset, tmp_path, windowed_splits):
    out = tmp_path / "wf"
    assert (
        walkforward.main(
            [
                "--dataset",
                str(dataset),
                "--out",
                str(out),
                "--folds",
                "2",
                *TINY,
            ]
        )
        == 0
    )
    assert set(windowed_splits) == {"train", "validation"}, "the spy must have fired"

    results = json.loads((out / "walkforward.json").read_text())
    assert results["test_evaluated"] is False
    for fold in results["folds"]:
        assert "test" not in fold
        assert set(fold["validation"]) == {"majority_baseline", "momentum_baseline", "mtst"}


def test_walkforward_records_the_time_boundaries_of_every_fold(dataset, tmp_path):
    out = tmp_path / "wf"
    walkforward.main(["--dataset", str(dataset), "--out", str(out), "--folds", "2", *TINY])
    folds = json.loads((out / "walkforward.json").read_text())["folds"]

    for fold in folds:
        train_period, val_period = fold["periods"]["train"], fold["periods"]["validation"]
        for period in (train_period, val_period):
            assert period["start"] and period["end"] and period["rows"] > 0
        assert train_period["end"] <= val_period["start"], "validation precedes training"
    assert folds[1]["periods"]["train"]["rows"] > folds[0]["periods"]["train"]["rows"]


def test_walkforward_aggregates_mean_and_std_across_folds(dataset, tmp_path):
    out = tmp_path / "wf"
    walkforward.main(["--dataset", str(dataset), "--out", str(out), "--folds", "2", *TINY])
    results = json.loads((out / "walkforward.json").read_text())
    summary = results["summary"]

    assert summary["folds"] == 2
    for name in walkforward.MODELS:
        stats = summary["per_model"][name]
        assert set(walkforward.SUMMARY_METRICS) <= set(stats)
        for metric in walkforward.SUMMARY_METRICS:
            assert {"mean", "std", "values"} == set(stats[metric])
            assert len(stats[metric]["values"]) == 2
    assert "macro F1" in (out / "walkforward.md").read_text()


# --- 5. the whole chain -------------------------------------------------------
def test_the_research_chain_runs_end_to_end(dataset, tmp_path, windowed_splits):
    """dataset -> validation-only train -> experiment -> walk-forward.

    One test for the workflow the documentation describes, asserting the thing
    that ties it together: none of the three steps touches the test split.
    """
    assert (
        train.main(
            [
                "--dataset",
                str(dataset),
                "--models-dir",
                str(tmp_path / "models"),
                "--validation-only",
            ]
            + TINY
        )
        == 0
    )
    assert (
        experiment.main(experiment_args(dataset, tmp_path / "exp", ["--seed", "1", "2"])) == 0
    )
    assert (
        walkforward.main(
            ["--dataset", str(dataset), "--out", str(tmp_path / "wf"), "--folds", "2", *TINY]
        )
        == 0
    )

    assert set(windowed_splits) == {"train", "validation"}, "the spy must have fired"
    assert (tmp_path / "exp" / "experiments.csv").exists()
    assert (tmp_path / "wf" / "walkforward.md").exists()
