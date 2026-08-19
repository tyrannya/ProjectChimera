"""Safeguards for the research workflow.

Two properties this suite exists to hold, both asserted directly rather than
trusted to well-behaved entrypoints:

1. **The sealed test split stays sealed.** The technique is a spy on
   ``nn.train`` internals: if ``build_windows`` is never called with the test
   split, no test row can have reached a metric, because windowing is the only
   way to get one there.
2. **Walk-forward reports a block it never fitted anything on.** Every fold has
   three chronological regions — train, inner validation, outer validation. The
   scaler and weights come from train; early stopping and the decision threshold
   come from inner; only outer is reported. Section 4b spies on every fitting
   step of a real run and checks which rows each one saw.

A guard test that cannot fail is worse than none, so the "test was not
windowed" assertions are paired with a control that runs the same entrypoint
*without* research mode and checks the test split IS windowed there. For the
same reason the assertions about fold geometry are on row indices: a check that
a block is *named* ``outer_validation`` would have passed against the two-region
design this suite was written to rule out.
"""

from __future__ import annotations

import csv
import json
from typing import Any

import numpy as np
import pytest
import torch

from chimera.contracts import ModelMetadata, TargetSpec
from chimera.features import FeatureSpec
from nn import experiment, train, walkforward
from nn.data_pipeline import build_dataset, save_dataset
from nn.dataset import Split, chronological_split, sealed_test_start, window_indices
from nn.model_def import MTST, MTSTConfig
from nn.registry import promote, resolve_current, save_model
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
        close=100 * np.cumprod(1 + rng.normal(size=n) * 0.01),
        segment_ids=segments,
        # Real timestamps, because reporting annualises from elapsed calendar
        # time rather than from a row count. The second segment starts a day
        # late, which is the gap this fixture exists to exercise.
        dates=(
            np.datetime64("2023-01-01T00:00")
            + (np.arange(n) + 24 * (segments == 1)) * np.timedelta64(1, "h")
        ),
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


# --- 4. walk-forward: three chronological regions per fold --------------------
#
# TRAIN -> INNER VALIDATION -> OUTER VALIDATION. The scaler and the weights are
# fitted on train; early stopping and the decision threshold are chosen on
# inner; the frozen model is measured once on outer, and only outer becomes the
# fold's reported result.
#
# The previous design had two regions and used the second one twice — it chose
# the threshold and the early-stopping epoch on the validation block and then
# reported that same block as the fold's performance. The tests below assert the
# separation on row indices, because "the split is called outer_validation" is
# exactly the kind of claim that was true of the old leaky geometry too.

#: Rows in the BTC 1h dataset the default geometry is sized against.
BTC_ROWS = 56726


def plan_defaults(boundary, folds=4):
    """The plan the CLI defaults produce for a research region of ``boundary`` rows."""
    args = walkforward.build_argparser().parse_args(["--dataset", "x", "--folds", str(folds)])
    sizes = walkforward.resolve_sizes(args, boundary)
    return walkforward.plan_nested_folds(boundary, args.folds, *sizes)


def test_nested_folds_are_chronological_and_grow_forward():
    folds = walkforward.plan_nested_folds(
        boundary=1000, folds=4, min_train=400, inner_size=100, outer_size=100, step=100
    )
    assert len(folds) == 4

    previous_train_end = 0
    for plan in folds:
        # Expanding: training always starts at the beginning and only grows.
        assert plan.train.start == 0
        assert plan.train.end > previous_train_end
        previous_train_end = plan.train.end
        # train -> inner -> outer, in that order, with no inversion.
        assert plan.train.end <= plan.inner.start
        assert plan.inner.end <= plan.outer.start
        assert len(plan.train) > 0 and len(plan.inner) > 0 and len(plan.outer) > 0


def test_outer_validation_blocks_never_overlap():
    """No row may be reported as the result of two folds."""
    for step in (100, 137, 250):
        folds = walkforward.plan_nested_folds(
            boundary=2000, folds=4, min_train=800, inner_size=100, outer_size=100, step=step
        )
        scored: set[int] = set()
        for plan in folds:
            rows = set(range(plan.outer.start, plan.outer.end))
            assert not (rows & scored), f"an outer row is reported twice at step={step}"
            scored |= rows


def test_the_default_step_makes_outer_blocks_contiguous():
    """Back to back, so the reported blocks tile one stretch without overlapping."""
    folds = plan_defaults(sealed_test_start(BTC_ROWS, 0.70, 0.15))
    for previous, plan in zip(folds, folds[1:]):
        assert plan.outer.start == previous.outer.end


def test_a_step_below_the_outer_block_is_refused():
    """Overlapping outer blocks are an error, not a silently double-counted row."""
    with pytest.raises(ValueError, match="consecutive outer blocks would overlap"):
        walkforward.plan_nested_folds(
            boundary=2000, folds=3, min_train=800, inner_size=100, outer_size=200, step=150
        )


def test_no_fold_fits_on_a_row_that_it_or_a_later_fold_reports():
    """The direction that matters: nothing fitted may see a row still to be scored.

    The opposite direction is allowed on purpose — a later fold may *train* on an
    earlier fold's outer block, because by then those rows are history.
    """
    folds = plan_defaults(sealed_test_start(BTC_ROWS, 0.70, 0.15))
    for k, plan in enumerate(folds):
        for later in folds[k:]:
            assert plan.train.end <= later.outer.start
            assert plan.inner.end <= later.outer.start


def test_nested_folds_refuse_a_research_region_that_is_too_short():
    with pytest.raises(
        ValueError, match=r"need 1300 rows .*only 500 rows lie before the sealed"
    ):
        walkforward.plan_nested_folds(
            boundary=500, folds=4, min_train=400, inner_size=100, outer_size=200, step=200
        )


def test_the_planner_does_not_quietly_return_fewer_folds_than_asked_for():
    """Failing loudly beats a summary that says four folds and aggregates two."""
    boundary = sealed_test_start(BTC_ROWS, 0.70, 0.15)
    assert len(plan_defaults(boundary, folds=4)) == 4
    with pytest.raises(ValueError, match="lie before the sealed test block"):
        plan_defaults(boundary, folds=40)


def test_default_fold_sizes_fit_the_research_region(dataset):
    """The defaults must produce a runnable plan, not an immediate error.

    Sized against the research region, never the dataset: the difference is the
    sealed block, and using the larger number is the bug this suite exists for.
    """
    data = train.load_research_data(dataset)
    args = walkforward.build_argparser().parse_args(["--dataset", str(dataset)])
    boundary = sealed_test_start(data.n_rows, args.train_frac, args.val_frac)
    folds = walkforward.plan_nested_folds(
        boundary, args.folds, *walkforward.resolve_sizes(args, boundary)
    )

    assert len(folds) == args.folds
    assert folds[-1].outer.end <= boundary < data.n_rows


def test_the_default_geometry_for_the_btc_research_region():
    """The exact 4-fold plan for the 56,726-row BTC dataset, pinned row by row.

    The sealed block starts at row 48,217; every number below is under it, and
    the four outer blocks are disjoint.
    """
    boundary = sealed_test_start(BTC_ROWS, 0.70, 0.15)
    assert boundary == 48217

    args = walkforward.build_argparser().parse_args(["--dataset", "x"])
    assert args.folds == 4
    min_train, inner_size, outer_size, step = walkforward.resolve_sizes(args, boundary)
    assert (min_train, inner_size, outer_size, step) == (21697, 4821, 4821, 4821)

    folds = walkforward.plan_nested_folds(boundary, 4, min_train, inner_size, outer_size, step)
    assert [
        (p.train.start, p.train.end, p.inner.start, p.inner.end, p.outer.start, p.outer.end)
        for p in folds
    ] == [
        (0, 21697, 21697, 26518, 26518, 31339),
        (0, 26518, 26518, 31339, 31339, 36160),
        (0, 31339, 31339, 36160, 36160, 40981),
        (0, 36160, 36160, 40981, 40981, 45802),
    ]
    assert folds[-1].outer.end <= boundary < BTC_ROWS


# --- 4b. what each block is allowed to touch, spied on a real run -------------
#
# One walk-forward run, every fitting step instrumented, asserted from several
# angles. The record is built once because the assertions are about a single
# run's behaviour, not about six independent runs.

WF_FOLDS = 2


@pytest.fixture(scope="module")
def spied_run(dataset, tmp_path_factory):
    """Run walk-forward once, recording every fit and every windowed split."""
    out = tmp_path_factory.mktemp("wf_spied")
    record: dict[str, Any] = {
        "scaler_fits": [],
        "threshold_returns": [],
        "windowed": [],
        "early_stopping": [],
        "frozen_splits": [],
        "majority_fits": [],
        "prepared": [],
    }

    with pytest.MonkeyPatch.context() as mp:
        originals = {
            "scaler": train.StandardScaler.fit,
            "threshold": train.ev.select_threshold,
            "windows": train.build_windows,
            "train_model": train.train_model,
            "frozen": train.score_frozen_split,
            "majority": train.MajorityClassBaseline.fit,
            "prepare": train.prepare_research_windows,
        }

        def scaler_spy(self, X):
            record["scaler_fits"].append(np.asarray(X).copy())
            return originals["scaler"](self, X)

        def threshold_spy(proba, future_return, target_spec, **kwargs):
            record["threshold_returns"].append(np.asarray(future_return).copy())
            return originals["threshold"](proba, future_return, target_spec, **kwargs)

        def windows_spy(features, targets, split, seq_len, horizon, **kwargs):
            record["windowed"].append((split.name, split.start, split.end, len(features)))
            return originals["windows"](features, targets, split, seq_len, horizon, **kwargs)

        def train_model_spy(config, X_train, y_train, X_val, y_val, **kwargs):
            record["early_stopping"].append((X_train, y_train, X_val, y_val))
            return originals["train_model"](config, X_train, y_train, X_val, y_val, **kwargs)

        def frozen_spy(data_arg, scaler, split, seq_len, **kwargs):
            scored = originals["frozen"](data_arg, scaler, split, seq_len, **kwargs)
            record["frozen_splits"].append((split.name, split.start, split.end, scored.idx))
            return scored

        def majority_spy(self, y):
            record["majority_fits"].append(np.asarray(y).copy())
            return originals["majority"](self, y)

        def prepare_spy(data_arg, train_split, val_split, seq_len):
            prepared = originals["prepare"](data_arg, train_split, val_split, seq_len)
            record["prepared"].append(prepared)
            return prepared

        mp.setattr(train.StandardScaler, "fit", scaler_spy)
        mp.setattr(train.ev, "select_threshold", threshold_spy)
        mp.setattr(train, "build_windows", windows_spy)
        mp.setattr(train, "train_model", train_model_spy)
        mp.setattr(train, "score_frozen_split", frozen_spy)
        mp.setattr(walkforward, "score_frozen_split", frozen_spy)
        mp.setattr(train.MajorityClassBaseline, "fit", majority_spy)
        mp.setattr(train, "prepare_research_windows", prepare_spy)
        mp.setattr(walkforward, "prepare_research_windows", prepare_spy)

        assert (
            walkforward.main(
                ["--dataset", str(dataset), "--out", str(out), "--folds", str(WF_FOLDS), *TINY]
            )
            == 0
        )

    record["out"] = out
    record["data"] = train.load_research_data(dataset)
    record["results"] = json.loads((out / "walkforward.json").read_text())
    record["boundary"] = sealed_test_start(record["data"].n_rows, 0.70, 0.15)
    record["plan"] = plan_defaults(record["boundary"], folds=WF_FOLDS)
    return record


def test_the_scaler_is_fitted_on_train_rows_only_in_every_fold(spied_run):
    features = spied_run["data"].features
    fits = spied_run["scaler_fits"]

    assert len(fits) == WF_FOLDS, "one scaler per fold, fitted once"
    for fitted, plan in zip(fits, spied_run["plan"]):
        np.testing.assert_allclose(fitted, features[plan.train.start : plan.train.end])
        assert len(fitted) == len(plan.train)


def test_early_stopping_sees_train_and_inner_validation_only(spied_run):
    """The arrays handed to the training loop are the fold's own train/inner windows."""
    assert len(spied_run["early_stopping"]) == WF_FOLDS
    for (X_train, y_train, X_val, y_val), prepared in zip(
        spied_run["early_stopping"], spied_run["prepared"]
    ):
        assert X_train is prepared.X_train and y_train is prepared.y_train
        assert X_val is prepared.X_val and y_val is prepared.y_val
        # prepared.validation is the INNER block, by construction in run_fold.
        assert prepared.validation.name == "inner_validation"


def test_the_threshold_is_selected_on_inner_validation_only(spied_run):
    """One selection per fold, over exactly the inner block's rows."""
    data, seen = spied_run["data"], spied_run["threshold_returns"]
    assert len(seen) == WF_FOLDS, "the threshold must be selected once per fold"

    for returns, prepared, plan in zip(seen, spied_run["prepared"], spied_run["plan"]):
        idx = prepared.idx_val
        np.testing.assert_allclose(returns, data.future_return[idx])
        assert idx.min() >= plan.inner.start
        assert idx.max() + data.target_spec.horizon < plan.inner.end
        # And never the block the fold is reported on.
        assert idx.max() < plan.outer.start


def test_no_fold_reports_a_row_it_selected_its_own_threshold_on(spied_run):
    """The negative of the test above, stated as disjoint sets of rows.

    The threshold is a fitted parameter. Choosing it on the block that is then
    reported is the exact optimism this design removes, so the check is that
    within a fold, the rows the selection call saw and the rows scored as that
    fold's result do not intersect — not merely that the call count looks right.

    The check is deliberately per fold. Across folds the sets *do* meet: with the
    default step, fold k's inner block is fold k-1's outer block. That is the
    intended geometry — by the time fold k runs, those rows are history, and
    choosing a threshold on history is what walk-forward is for. What must never
    happen is a fold using its own reported rows, or any fold using rows a
    *later* fold will report; the latter is asserted in
    ``test_no_fold_fits_on_a_row_that_it_or_a_later_fold_reports``.
    """
    assert len(spied_run["threshold_returns"]) == WF_FOLDS
    assert len(spied_run["prepared"]) == len(spied_run["frozen_splits"]) == WF_FOLDS

    for prepared, (_, _, _, idx_outer) in zip(
        spied_run["prepared"], spied_run["frozen_splits"]
    ):
        selected = {int(i) for i in prepared.idx_val}
        scored = {int(i) for i in idx_outer}
        assert selected and scored
        assert not (selected & scored), "a fold reported a row it tuned its threshold on"
        assert max(selected) < min(scored), "selection must precede the reported block"


def test_no_window_or_label_crosses_a_region_boundary(spied_run):
    """Requirement stated as index arithmetic, for all three regions of each fold.

    A window that starts before its block would read rows the block does not own;
    a label that ends after it would read a price from the next block. Both bounds
    come from ``window_indices``, so they are checked there rather than inferred
    from a sample count.
    """
    seq_len, horizon = 16, spied_run["data"].target_spec.horizon
    for plan in spied_run["plan"]:
        for split in (plan.train, plan.inner, plan.outer):
            idx = window_indices(split, seq_len, horizon)
            assert idx.size > 0, f"{split.name} is too short to hold a sample"
            assert idx.min() - (seq_len - 1) >= split.start, "window leaves its block"
            assert idx.max() + horizon <= split.end - 1, "label leaves its block"
        assert plan.train.end <= plan.inner.start
        assert plan.inner.end <= plan.outer.start
        assert plan.outer.end <= spied_run["boundary"]


def test_frozen_scoring_respects_market_data_gaps():
    """The outer block goes through the same segment check as train and inner.

    ``score_frozen_split`` is the only path the outer block takes, so the gap
    safety ``nn.train`` enforces has to hold there too: a window may not bridge a
    missing exchange candle just because it is being scored rather than trained
    on.
    """
    rng = np.random.default_rng(5)
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
        close=100 * np.cumprod(1 + rng.normal(size=n) * 0.01),
        segment_ids=segments,
        # Real timestamps, because reporting annualises from elapsed calendar
        # time rather than from a row count. The second segment starts a day
        # late, which is the gap this fixture exists to exercise.
        dates=(
            np.datetime64("2023-01-01T00:00")
            + (np.arange(n) + 24 * (segments == 1)) * np.timedelta64(1, "h")
        ),
        candles_per_year=24 * 365,
    )
    outer = Split("outer_validation", 150, 400)
    scaler = train.StandardScaler().fit(data.features[:150])
    model = MTST(MTSTConfig(input_dim=2, seq_len=8, d_model=8, n_heads=2, num_layers=1))

    scored = train.score_frozen_split(
        data,
        scaler,
        outer,
        8,
        model=model,
        baselines={},
        threshold=0.5,
        device=torch.device("cpu"),
    )
    reports, idx = scored.reports, scored.idx

    assert set(reports) == {"mtst"}
    assert idx.min() - 7 >= outer.start and idx.max() + 4 <= outer.end - 1
    for i in idx:
        assert segments[i - 7] == segments[i], "window bridges a market-data gap"
        assert segments[i + 4] == segments[i], "label bridges a market-data gap"
    # The filter must actually have dropped the candidates that straddle row 200.
    assert len(idx) < len(window_indices(outer, 8, 4))


def test_the_baselines_are_fitted_on_train_rows_only(spied_run):
    """The majority baseline has state; it learns the prior from train windows."""
    fits = spied_run["majority_fits"]
    assert len(fits) == WF_FOLDS
    for y, prepared in zip(fits, spied_run["prepared"]):
        np.testing.assert_array_equal(y, prepared.y_train)


def test_outer_validation_reaches_only_the_frozen_evaluation(spied_run):
    """It is windowed exactly once per fold, and only by score_frozen_split."""
    frozen = spied_run["frozen_splits"]
    assert len(frozen) == WF_FOLDS
    for (name, start, end, _), plan in zip(frozen, spied_run["plan"]):
        assert (name, start, end) == ("outer_validation", plan.outer.start, plan.outer.end)

    outer_windowings = [w for w in spied_run["windowed"] if w[0] == "outer_validation"]
    assert len(outer_windowings) == WF_FOLDS


def test_walkforward_windows_three_regions_and_never_the_sealed_block(spied_run):
    windowed = spied_run["windowed"]
    assert {name for name, *_ in windowed} == {
        "train",
        "inner_validation",
        "outer_validation",
    }
    assert "test" not in {name for name, *_ in windowed}
    # Asserted on rows, not names: nothing windowed reaches the sealed boundary,
    # and the array a window is cut from does not even contain a sealed row.
    for _, _, end, n_features_rows in windowed:
        assert end <= spied_run["boundary"]
        assert n_features_rows <= spied_run["boundary"]


def test_outer_validation_samples_stay_inside_their_own_block(spied_run):
    """Window and label horizon alike, asserted on the indices actually scored."""
    horizon = spied_run["data"].target_spec.horizon
    seq_len = 16  # TINY
    for (_, start, end, idx), plan in zip(spied_run["frozen_splits"], spied_run["plan"]):
        assert (start, end) == (plan.outer.start, plan.outer.end)
        assert idx.min() - (seq_len - 1) >= plan.outer.start, "window reaches into inner"
        assert idx.max() + horizon < plan.outer.end, "label reaches past the fold"
        assert idx.max() + horizon < spied_run["boundary"], "label reaches sealed rows"


def test_no_row_is_scored_as_outer_validation_in_two_folds(spied_run):
    """End to end on the rows actually evaluated, not on the plan."""
    scored: set[int] = set()
    horizon = spied_run["data"].target_spec.horizon
    for _, _, _, idx in spied_run["frozen_splits"]:
        rows = set(int(i) for i in idx)
        assert not (rows & scored), "a row was evaluated as outer validation twice"
        scored |= rows
        assert max(rows) + horizon < spied_run["boundary"]


def test_the_output_schema_names_all_three_regions(spied_run):
    results = spied_run["results"]

    assert results["test_evaluated"] is False
    assert results["reported_block"] == "outer_validation"
    assert results["sealed_test"]["evaluated"] is False
    assert results["sealed_test"]["start_row"] == spied_run["boundary"]

    for fold, plan in zip(results["folds"], spied_run["plan"]):
        assert set(fold["periods"]) == {"train", "inner_validation", "outer_validation"}
        assert set(fold["selection"]) == {
            "best_epoch",
            "threshold",
            "inner_validation_loss",
        }
        # The models, plus the economic references for the same window. The
        # references are not models — nothing in them predicts anything — but
        # they describe this block and travel with it.
        assert set(fold["outer_validation"]) == {
            *walkforward.MODELS,
            walkforward.REFERENCES_KEY,
        }
        references = fold["outer_validation"][walkforward.REFERENCES_KEY]
        assert set(references) >= {"cash", "buy_and_hold"}
        # The ambiguous single "validation" key is gone: a reader cannot mistake
        # the selection block for the reported one.
        assert "validation" not in fold
        assert "test" not in fold

        assert fold["periods"]["train"]["row_range"] == [plan.train.start, plan.train.end]
        assert fold["periods"]["inner_validation"]["row_range"] == [
            plan.inner.start,
            plan.inner.end,
        ]
        assert fold["periods"]["outer_validation"]["row_range"] == [
            plan.outer.start,
            plan.outer.end,
        ]
        for period in fold["periods"].values():
            assert period["start"] and period["end"] and period["rows"] > 0
            assert period["row_range"][1] <= spied_run["boundary"]

    train_rows = [f["periods"]["train"]["rows"] for f in results["folds"]]
    assert train_rows == sorted(train_rows) and train_rows[-1] > train_rows[0]


def test_the_json_regions_are_ordered_train_inner_outer(spied_run):
    for fold in spied_run["results"]["folds"]:
        periods = fold["periods"]
        assert periods["train"]["row_range"][1] <= periods["inner_validation"]["row_range"][0]
        assert (
            periods["inner_validation"]["row_range"][1]
            <= periods["outer_validation"]["row_range"][0]
        )


def test_the_summary_aggregates_outer_validation_only(spied_run):
    summary = spied_run["results"]["summary"]

    assert summary["folds"] == WF_FOLDS
    assert summary["aggregated_from"] == "outer_validation"
    for name in walkforward.MODELS:
        stats = summary["per_model"][name]
        assert set(walkforward.SUMMARY_METRICS) <= set(stats)
        for metric, (section, key) in walkforward.SUMMARY_METRICS.items():
            assert {"mean", "std", "values", "defined_folds"} == set(stats[metric])
            # `None` survives aggregation as `None`: a risk-adjusted statistic
            # is undefined when the portfolio never moved, and rounding a
            # fabricated zero into the record would hide that.
            assert stats[metric]["values"] == [
                (
                    None
                    if (raw := f["outer_validation"][name][section][key]) is None
                    else round(float(raw), 6)
                )
                for f in spied_run["results"]["folds"]
            ]
    assert "outer validation" in (spied_run["out"] / "walkforward.md").read_text().lower()


def test_the_markdown_reports_outer_validation_and_claims_nothing_more(spied_run):
    markdown = (spied_run["out"] / "walkforward.md").read_text()

    assert "Outer validation (the reported result)" in markdown
    assert "calib err" in markdown
    assert "sealed test block remains unopened" in markdown
    assert "not an out-of-sample result" in markdown


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

    assert set(windowed_splits) == {
        "train",
        "validation",
        "inner_validation",
        "outer_validation",
    }, "the spy must have fired"
    assert "test" not in windowed_splits
    assert (tmp_path / "exp" / "experiments.csv").exists()
    assert (tmp_path / "wf" / "walkforward.md").exists()


# --- 6. the sealed boundary, asserted on row indices ---------------------------
#
# A name-based check is exactly what let a real leak through: walk-forward
# planned folds over the whole dataset, so its last validation windows sat
# inside the sealed test block. The rows were test rows wearing the label
# "validation", and every name-based check passed.
#
# Everything below therefore compares row indices against
# nn.dataset.sealed_test_start and never looks at a split's name.


def test_sealed_test_start_agrees_with_the_chronological_split():
    """One boundary, computed one way, for nn.train and walk-forward alike."""
    for n_rows in (1000, 1918, 5001, 56726, 123457):
        assert sealed_test_start(n_rows, 0.70, 0.15) == (
            chronological_split(n_rows, 0.70, 0.15).test.start
        )


def test_nested_folds_never_plan_a_row_at_or_beyond_the_boundary():
    """The regression for the leak, in the geometry that produced it.

    56,726 rows with the default fold settings used to put fold 2's validation
    1,890 rows into the sealed block and fold 3's almost exactly on top of it.
    Planning over the boundary instead of the dataset length is the fix, so the
    assertion is on row indices — for all three regions, not just the reported
    one.
    """
    boundary = sealed_test_start(BTC_ROWS, 0.70, 0.15)
    folds = plan_defaults(boundary)

    assert len(folds) == 4
    for plan in folds:
        assert plan.train.end <= boundary
        assert plan.inner.end <= boundary
        assert plan.outer.end <= boundary
        assert plan.outer.start < boundary


def test_planning_over_the_dataset_length_would_cross_the_boundary():
    """Proves the test above can fail: the old geometry really did leak.

    This reproduces what the previous implementation computed — sizes derived
    from ``n_rows`` rather than from the research region — and asserts the plan
    crosses the boundary. If a future change reintroduced it, the test above
    turns red rather than both quietly agreeing.
    """
    boundary = sealed_test_start(BTC_ROWS, 0.70, 0.15)
    min_train = int(BTC_ROWS * 0.45)
    inner_size = outer_size = int(BTC_ROWS * 0.10)

    leaky = walkforward.plan_nested_folds(
        BTC_ROWS, 4, min_train, inner_size, outer_size, outer_size
    )
    assert any(plan.outer.end > boundary for plan in leaky)


def test_the_planner_refuses_to_borrow_from_the_sealed_block():
    boundary = sealed_test_start(1000, 0.70, 0.15)  # 850
    with pytest.raises(ValueError, match="lie before the sealed test block"):
        walkforward.plan_nested_folds(
            boundary, 4, min_train=500, inner_size=100, outer_size=100, step=100
        )


def test_walkforward_row_indices_stay_below_the_boundary(dataset, tmp_path, monkeypatch):
    """End to end: no row a fold trains on, selects on or reports is sealed.

    The label horizon is included on both validation blocks, because a sample
    whose window sits below the boundary can still read its label from above it.
    """
    data = train.load_research_data(dataset)
    boundary = sealed_test_start(data.n_rows, 0.70, 0.15)
    horizon = data.target_spec.horizon

    seen: list[int] = []
    original_prepare = train.prepare_research_windows
    original_frozen = train.score_frozen_split

    def prepare_spy(data_arg, train_split, val_split, seq_len):
        prepared = original_prepare(data_arg, train_split, val_split, seq_len)
        seen.append(int(prepared.idx_val.max()) + horizon)
        seen.append(int(train_split.end) - 1)
        return prepared

    def frozen_spy(data_arg, scaler, split, seq_len, **kwargs):
        scored = original_frozen(data_arg, scaler, split, seq_len, **kwargs)
        seen.append(int(scored.idx.max()) + horizon)
        return scored

    monkeypatch.setattr(train, "prepare_research_windows", prepare_spy)
    monkeypatch.setattr(walkforward, "prepare_research_windows", prepare_spy)
    monkeypatch.setattr(train, "score_frozen_split", frozen_spy)
    monkeypatch.setattr(walkforward, "score_frozen_split", frozen_spy)
    walkforward.main(
        ["--dataset", str(dataset), "--out", str(tmp_path / "wf"), "--folds", "3", *TINY]
    )

    assert seen, "the spy must have fired"
    assert max(seen) < boundary, f"a fold reached row {max(seen)}, sealed starts at {boundary}"

    results = json.loads((tmp_path / "wf" / "walkforward.json").read_text())
    assert results["sealed_test"]["start_row"] == boundary
    assert results["sealed_test"]["evaluated"] is False
    assert results["sealed_test"]["period"]["rows"] == data.n_rows - boundary
    for fold in results["folds"]:
        for region in ("train", "inner_validation", "outer_validation"):
            assert fold["periods"][region]["row_range"][1] <= boundary


def test_experiment_validation_indices_stay_below_the_boundary(dataset, tmp_path):
    data = train.load_research_data(dataset)
    boundary = sealed_test_start(data.n_rows, 0.70, 0.15)
    plan = chronological_split(data.n_rows, 0.70, 0.15)

    prepared = train.prepare_research_windows(data, plan.train, plan.validation, 16)
    assert int(prepared.idx_val.max()) + data.target_spec.horizon < boundary

    out = tmp_path / "exp"
    experiment.main(experiment_args(dataset, out))
    summary = json.loads((out / "experiments.json").read_text())
    assert summary["sealed_test"]["start_row"] == boundary
    assert summary["periods"]["validation"]["row_range"][1] <= boundary


def test_prepared_windows_never_contain_a_row_from_a_later_fold(dataset):
    """The scaled array itself stops at the fold boundary, so later rows are
    not merely unused — they are absent."""
    data = train.load_research_data(dataset)
    val_split = Split("validation", 600, 800)
    prepared = train.prepare_research_windows(data, Split("train", 0, 600), val_split, 16)

    # Reconstruct the last row any sample touches: the newest label row.
    assert int(prepared.idx_val.max()) + data.target_spec.horizon <= val_split.end - 1


# --- 7. promotion fails closed -------------------------------------------------
def build_artifact(tmp_path, report):
    """A minimal on-disk artifact whose report.json is whatever we pass."""
    models_dir = tmp_path / "models"
    config = MTSTConfig(input_dim=4, seq_len=8)
    metadata = ModelMetadata(
        model_version="v1",
        feature_names=["a", "b", "c", "d"],
        sequence_length=8,
        feature_spec=FeatureSpec(),
        target_spec=TargetSpec(),
        scaler_mean=[0.0] * 4,
        scaler_std=[1.0] * 4,
        decision_threshold=0.5,
    )
    save_model(models_dir, "v1", MTST(config), metadata, report)
    return models_dir


def test_promotion_refuses_an_artifact_with_no_report(tmp_path):
    models_dir = build_artifact(tmp_path, None)
    assert not (models_dir / "v1" / "report.json").exists()
    with pytest.raises(ValueError, match="no report.json"):
        promote(models_dir, "v1")


def test_promotion_refuses_a_malformed_report(tmp_path):
    models_dir = build_artifact(tmp_path, {"research_only": False, "test_evaluated": True})
    (models_dir / "v1" / "report.json").write_text('{"research_only": fal')
    with pytest.raises(ValueError, match="not readable JSON"):
        promote(models_dir, "v1")


def test_promotion_refuses_a_report_that_is_not_an_object(tmp_path):
    models_dir = build_artifact(tmp_path, {"research_only": False, "test_evaluated": True})
    (models_dir / "v1" / "report.json").write_text("[1, 2, 3]")
    with pytest.raises(ValueError, match="not an object"):
        promote(models_dir, "v1")


@pytest.mark.parametrize(
    "report",
    [
        pytest.param({}, id="both-fields-absent"),
        pytest.param({"test_evaluated": True}, id="research_only-absent"),
        pytest.param({"research_only": False}, id="test_evaluated-absent"),
        pytest.param({"research_only": True, "test_evaluated": True}, id="research_only-true"),
        pytest.param(
            {"research_only": False, "test_evaluated": False}, id="test_evaluated-false"
        ),
        pytest.param(
            {"research_only": "false", "test_evaluated": "true"}, id="strings-not-booleans"
        ),
        pytest.param({"research_only": None, "test_evaluated": None}, id="explicit-nulls"),
    ],
)
def test_promotion_refuses_anything_short_of_positive_evidence(tmp_path, report):
    models_dir = build_artifact(tmp_path, report)
    with pytest.raises(ValueError, match="cannot promote v1"):
        promote(models_dir, "v1")
    assert not (models_dir / "current.json").exists()


def test_promotion_allows_a_sealed_test_artifact(tmp_path):
    """The normal path: gates already checked by the caller, test actually scored."""
    models_dir = build_artifact(tmp_path, {"research_only": False, "test_evaluated": True})
    promote(models_dir, "v1")
    assert resolve_current(models_dir) == models_dir / "v1"


def test_the_normal_train_promote_path_still_works(dataset, tmp_path, monkeypatch):
    """nn.train --promote must still be able to promote a passing model.

    The gates are stubbed to pass so the test exercises promotion plumbing
    rather than whether a one-epoch model on synthetic candles is any good.
    """
    monkeypatch.setattr(train, "check_gates", lambda *a, **k: (True, []))
    models_dir = tmp_path / "models"
    assert (
        train.main(
            ["--dataset", str(dataset), "--models-dir", str(models_dir), "--promote"] + TINY
        )
        == 0
    )
    promoted = resolve_current(models_dir)
    report = json.loads((promoted / "report.json").read_text())
    assert report["research_only"] is False
    assert report["test_evaluated"] is True


# --- 8. the experiment manifest is written before training ---------------------
def test_the_experiment_plan_exists_before_the_first_model_trains(
    dataset, tmp_path, monkeypatch
):
    """A grid declared only once the results are in is not predeclared."""
    out = tmp_path / "exp"
    seen_at_first_train: dict[str, Any] = {}
    original = experiment.fit_and_validate

    def spy(*args, **kwargs):
        seen_at_first_train.setdefault(
            "plan", json.loads((out / experiment.PLAN_FILE).read_text())
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(experiment, "fit_and_validate", spy)
    experiment.main(experiment_args(dataset, out, ["--seed", "1", "2"]))

    plan = seen_at_first_train["plan"]
    assert plan["n_runs"] == 2, "the whole grid must be on disk before training starts"
    assert [r["config"]["seed"] for r in plan["runs"]] == [1, 2]
    assert plan["sealed_test"]["evaluated"] is False


def test_the_results_carry_the_hash_of_the_plan_they_came_from(dataset, tmp_path):
    out = tmp_path / "exp"
    experiment.main(experiment_args(dataset, out, ["--seed", "1", "2"]))
    plan = json.loads((out / experiment.PLAN_FILE).read_text())
    results = json.loads((out / "experiments.json").read_text())

    assert results["plan_hash"] == plan["plan_hash"]
    assert results["plan_file"] == experiment.PLAN_FILE
    assert len(plan["plan_hash"]) == 16


def test_the_plan_hash_changes_with_the_grid(dataset, tmp_path):
    hashes = set()
    for extra in (["--seed", "1"], ["--seed", "1", "2"], ["--lr", "1e-3"]):
        out = tmp_path / f"exp{len(hashes)}"
        experiment.main(experiment_args(dataset, out, extra))
        hashes.add(json.loads((out / experiment.PLAN_FILE).read_text())["plan_hash"])
    assert len(hashes) == 3


def test_a_different_plan_will_not_overwrite_an_existing_one(dataset, tmp_path):
    out = tmp_path / "exp"
    experiment.main(experiment_args(dataset, out, ["--seed", "1"]))
    with pytest.raises(SystemExit, match="already holds a different experiment plan"):
        experiment.main(experiment_args(dataset, out, ["--seed", "2"]))


def test_research_artifacts_are_still_written(dataset, tmp_path):
    """Validation-only runs stay inspectable; they are just not promotable."""
    models_dir = tmp_path / "models"
    train.main(
        ["--dataset", str(dataset), "--models-dir", str(models_dir), "--validation-only"]
        + TINY
    )
    version = only_version(models_dir)
    for name in ("model.pt", "config.json", "metadata.json", "report.json"):
        assert (version / name).exists(), f"{name} must survive a research run"
    with pytest.raises(ValueError):
        promote(models_dir, version.name)
