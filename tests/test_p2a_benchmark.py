"""Safeguards for the P2a simple-model benchmark.

P2a's whole claim is that three untuned tabular models were given **exactly the
information MTST was given**, on the same folds, under the same research
generation, with the same threshold rule — so that any difference between the
families is a difference of model family and nothing else. A claim of that shape
is worth only as much as the checks behind it, so this suite asserts it directly
rather than trusting the entrypoint:

1. **Sample identity.** The rows P2a fits on are MTST's own windows, flattened.
   Asserted against a hand-built extraction, against
   :func:`nn.dataset.sample_indices`, and across a market-data gap.
2. **Fitting discipline.** Spies on every fitting step of a real run record which
   rows each one saw: the scaler sees training rows, the estimators see training
   rows, threshold selection sees inner rows, and nothing sees an outer or a
   sealed row.
3. **Artifacts.** The persisted outer predictions reproduce the reported metrics
   and the reported directional attribution exactly, and a rerun at one seed
   reproduces the first byte for byte.
4. **Fail-closed comparison.** A mismatched contract, fingerprint, fold geometry,
   sealed anchor, baseline or scored row set withholds the aggregate instead of
   producing one.
5. **The frozen evidence is untouched.** The committed BTC v4 artifacts still
   hash to their manifest, and still load.

Assertions about fold structure are on row indices, never on split names: a block
*named* ``outer_validation`` is not evidence that it sits where an outer block
belongs, and this repository has shipped that exact bug before.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from chimera.contracts import CLASS_ORDER, TargetSpec
from chimera.features import FeatureSpec
from nn import benchmark, benchmark_compare, evaluate as ev, regime, simple_models
from nn import train as train_module
from nn.data_pipeline import build_dataset, save_dataset
from nn.dataset import Split, sample_indices
from nn.research_contract import SYNTHETIC_CONTRACT_ID
from nn.simple_models import SIMPLE_MODEL_NAMES
from tools.make_sample_data import generate_candles

ROOT = Path(__file__).resolve().parents[1]

#: A fold plan small enough to fit three untuned models in seconds, and large
#: enough that every block holds many windows.
FOLDS = 2
SEQ_LEN = 8
HORIZON = 4

CLI = [
    "--research-contract",
    SYNTHETIC_CONTRACT_ID,
    "--folds",
    str(FOLDS),
    "--seq-len",
    str(SEQ_LEN),
]


@pytest.fixture(scope="module")
def dataset(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("p2a") / "ds.parquet"
    frame, metadata = build_dataset(
        generate_candles(rows=1500, seed=17),
        FeatureSpec(),
        TargetSpec(horizon=HORIZON),
        exchange="synthetic",
        pair="SYNTH/USDT",
        timeframe="1h",
    )
    save_dataset(path, frame, metadata)
    return path


@pytest.fixture(scope="module")
def data(dataset):
    return train_module.load_research_data(dataset)


@pytest.fixture(scope="module")
def run_dir(dataset, tmp_path_factory) -> Path:
    """One real benchmark run, shared by the artifact assertions."""
    out = tmp_path_factory.mktemp("p2a_run") / "seed_42"
    assert benchmark.main(["--dataset", str(dataset), "--out", str(out), *CLI]) == 0
    return out


@pytest.fixture(scope="module")
def artifact(run_dir) -> dict[str, Any]:
    return json.loads((run_dir / benchmark.ARTIFACT_NAME).read_text())


@pytest.fixture(scope="module")
def predictions(run_dir) -> pd.DataFrame:
    return pd.read_parquet(run_dir / benchmark.PREDICTIONS_NAME)


# --- 1. exact sample and flattening identity ----------------------------------
def test_flattened_row_is_the_mtst_window_element_for_element(data):
    """The flattened row is MTST's ``seq_len x n_features`` input, in the declared order.

    Built by hand from the scaled feature matrix — not by calling the function
    under test with different arguments — so this fails if the ordering claim
    ever stops being true.
    """
    split = Split("train", 0, 400)
    prepared = train_module.prepare_research_windows(
        data, split, Split("inner_validation", 400, 500), SEQ_LEN
    )
    flat = simple_models.flatten_windows(prepared.X_train)
    n_features = len(data.feature_names)

    assert flat.shape == (len(prepared.X_train), SEQ_LEN * n_features)
    for sample in (0, 1, len(flat) // 2, len(flat) - 1):
        for step in range(SEQ_LEN):
            for feature in range(n_features):
                assert (
                    flat[sample, step * n_features + feature]
                    == prepared.X_train[sample, step, feature]
                ), f"sample {sample} step {step} feature {feature} is in the wrong column"


def test_flatten_order_is_deterministic_and_named(data):
    """Repeated flattening agrees, and every column has the name the artifact records."""
    X = np.arange(2 * SEQ_LEN * len(data.feature_names), dtype=np.float32).reshape(
        2, SEQ_LEN, len(data.feature_names)
    )
    first = simple_models.flatten_windows(X)
    assert np.array_equal(first, simple_models.flatten_windows(X.copy()))
    # C order, spelled out: this is the contract FLATTEN_ORDER states.
    assert np.array_equal(first, X.reshape(2, -1))

    names = simple_models.flattened_column_names(data.feature_names, SEQ_LEN)
    assert len(names) == SEQ_LEN * len(data.feature_names)
    assert len(set(names)) == len(names)
    # Oldest timestep first; the decision row is offset zero and comes last.
    assert names[0] == f"t-{SEQ_LEN - 1}__{data.feature_names[0]}"
    assert names[-1] == f"t-0__{data.feature_names[-1]}"


def test_the_declared_btc_geometry_flattens_to_896(data):
    """The checkpoint's own numbers: a 64 x 14 MTST window is a 896-wide P2a row.

    The rest of this suite runs at a small ``seq_len`` for speed. This one pins
    the geometry P2a is actually declared at, on the real 14-feature contract,
    against a deterministic fixture — so a change to the flattening rule cannot
    slip through on the strength of a small case alone.
    """
    seq_len, n_features = 64, len(data.feature_names)
    assert n_features == 14
    rng = np.random.default_rng(0)
    X = rng.standard_normal((3, seq_len, n_features)).astype(np.float32)

    flat = simple_models.flatten_windows(X)
    assert flat.shape == (3, 896)

    names = simple_models.flattened_column_names(data.feature_names, seq_len)
    assert len(names) == 896
    for sample in range(3):
        for step in range(seq_len):
            for feature, feature_name in enumerate(data.feature_names):
                column = step * n_features + feature
                assert names[column] == f"t-{seq_len - 1 - step}__{feature_name}"
                assert flat[sample, column] == X[sample, step, feature]


def test_the_predeclared_configurations_are_the_declared_ones():
    """The fixed configurations are a contract, not a default to be revised.

    Written out here rather than referenced, so an edit to
    :mod:`nn.simple_models` after a BTC result has been seen fails this test
    instead of quietly redefining what P2a measured.
    """
    configs = {spec.name: spec.config(7) for spec in simple_models.SIMPLE_MODELS}
    assert configs["logistic_regression"] == {
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 2000,
        "class_weight": None,
    }
    assert configs["lightgbm"] == {
        "objective": "multiclass",
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "n_jobs": 1,
        "deterministic": True,
        "force_row_wise": True,
        "verbose": -1,
        "random_state": 7,
    }
    assert configs["xgboost"] == {
        "objective": "multi:softprob",
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.05,
        "min_child_weight": 1,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "eval_metric": "mlogloss",
        "device": "cpu",
        "n_jobs": 1,
        "random_state": 7,
    }
    # The seed is derived from the run/fold seed, never hard-coded.
    assert simple_models.LIGHTGBM.config(11)["random_state"] == 11
    assert simple_models.XGBOOST.config(11)["random_state"] == 11
    # lbfgs is deterministic and takes no seed; saying so beats inventing one.
    assert simple_models.LOGISTIC_REGRESSION.seeded == ""


def test_benchmark_samples_are_the_windowing_modules_own_rows(data):
    """Eligible rows come from :func:`nn.dataset.sample_indices`, not a second rule."""
    inner = Split("inner_validation", 400, 520)
    prepared = train_module.prepare_research_windows(
        data, Split("train", 0, 400), inner, SEQ_LEN
    )
    expected = sample_indices(inner, SEQ_LEN, HORIZON, segment_ids=data.segment_ids)
    assert np.array_equal(prepared.idx_val, expected)


def test_no_feature_window_reaches_a_future_row(data):
    """Every window ends on its own decision row, and every label stays inside the block."""
    split = Split("inner_validation", 400, 520)
    idx = sample_indices(split, SEQ_LEN, HORIZON, segment_ids=data.segment_ids)
    assert idx.size
    # The window is [i - seq_len + 1, i]: its newest row IS the decision row, so
    # nothing after i can be inside it.
    assert (idx - SEQ_LEN + 1 >= split.start).all()
    assert (idx + HORIZON <= split.end - 1).all()
    # Chronology: strictly increasing, never reordered or shuffled.
    assert (np.diff(idx) > 0).all()


def test_gaps_are_handled_identically_for_both_families(data):
    """A window straddling a market-data gap is dropped for P2a exactly as for MTST.

    The gap is imposed on the segment ids rather than on the dates, because that
    is the representation both families' windowing actually reads.
    """
    split = Split("inner_validation", 400, 520)
    gapped = np.zeros(data.n_rows, dtype=np.int64)
    gapped[450:] = 1
    with_gap = sample_indices(split, SEQ_LEN, HORIZON, segment_ids=gapped)
    without = sample_indices(split, SEQ_LEN, HORIZON, segment_ids=None)

    dropped = sorted(set(without.tolist()) - set(with_gap.tolist()))
    assert dropped, "the fixture must actually drop windows, or it proves nothing"
    # Exactly the rows whose input window or embargoed label crosses row 450.
    assert dropped == [
        row for row in without.tolist() if row - SEQ_LEN + 1 < 450 <= row + HORIZON
    ]

    # And the flattened rows are built from that same surviving set.
    scaled = np.zeros((data.n_rows, len(data.feature_names)), dtype=np.float32)
    from nn.dataset import build_windows

    X, _, idx = build_windows(
        scaled, data.targets, split, SEQ_LEN, HORIZON, segment_ids=gapped
    )
    assert np.array_equal(idx, with_gap)
    assert simple_models.flatten_windows(X).shape[0] == len(with_gap)


def test_probability_columns_are_placed_by_class_not_by_position():
    """An estimator that never saw a class still emits it, at probability zero."""
    aligned = simple_models.align_class_columns(
        np.array([[0.3, 0.7], [0.9, 0.1]]), np.array([0, 2])
    )
    assert aligned.shape == (2, len(CLASS_ORDER))
    assert np.allclose(aligned[:, 0], [0.3, 0.9])
    assert np.allclose(aligned[:, 1], [0.0, 0.0])
    assert np.allclose(aligned[:, 2], [0.7, 0.1])


# --- 2. nothing is fitted on validation, ever ---------------------------------
def test_only_training_rows_reach_the_scaler_and_the_estimators(
    dataset, data, tmp_path, monkeypatch
):
    """Spy on every fit in a real run and check the rows each one saw.

    The estimators are watched through :func:`nn.simple_models.fit_simple_model`,
    which is the only door a model can be fitted through, and the threshold
    through :func:`nn.evaluate.select_threshold`, which is the only door a
    threshold can be selected through.
    """
    seen: dict[str, list[Any]] = {"scaler": [], "fit_rows": [], "threshold_rows": []}
    original_scaler = train_module.StandardScaler.fit
    original_fit = benchmark.fit_simple_model
    original_threshold = benchmark.ev.select_threshold

    def scaler_spy(self, X):
        seen["scaler"].append(len(X))
        return original_scaler(self, X)

    def fit_spy(spec, X_train, y_train, *, seed):
        seen["fit_rows"].append((spec.name, X_train.shape, int(seed)))
        return original_fit(spec, X_train, y_train, seed=seed)

    def threshold_spy(proba, future_return, target_spec, **kwargs):
        seen["threshold_rows"].append(np.asarray(kwargs["row_index"]).copy())
        return original_threshold(proba, future_return, target_spec, **kwargs)

    monkeypatch.setattr(train_module.StandardScaler, "fit", scaler_spy)
    monkeypatch.setattr(benchmark, "fit_simple_model", fit_spy)
    monkeypatch.setattr(benchmark.ev, "select_threshold", threshold_spy)

    out = tmp_path / "spied"
    assert benchmark.main(["--dataset", str(dataset), "--out", str(out), *CLI]) == 0

    payload = json.loads((out / benchmark.ARTIFACT_NAME).read_text())
    boundary = payload["sealed_test"]["start_row"]
    plans = {
        int(fold["fold"]): {
            region: tuple(fold["periods"][region]["row_range"])
            for region in ("train", "inner_validation", "outer_validation")
        }
        for fold in payload["folds"]
    }

    # One scaler fit per fold, each on exactly that fold's training rows.
    assert seen["scaler"] == [
        plans[k]["train"][1] - plans[k]["train"][0] for k in sorted(plans)
    ]

    # Every estimator was fitted on the training window count, three per fold.
    assert len(seen["fit_rows"]) == FOLDS * len(SIMPLE_MODEL_NAMES)
    for fold_index, fold in enumerate(payload["folds"]):
        for name, shape, seed in seen["fit_rows"][
            fold_index * len(SIMPLE_MODEL_NAMES) : (fold_index + 1) * len(SIMPLE_MODEL_NAMES)
        ]:
            assert shape == (fold["samples"]["train"], SEQ_LEN, len(data.feature_names))
            assert seed == fold["fold_seed"]

    # Threshold selection saw inner rows only — never an outer row, never a
    # sealed one. Asserted on row indices, not on the name of the block.
    assert len(seen["threshold_rows"]) == FOLDS * len(SIMPLE_MODEL_NAMES)
    for fold_index, rows in enumerate(seen["threshold_rows"]):
        inner_start, inner_end = plans[fold_index // len(SIMPLE_MODEL_NAMES)][
            "inner_validation"
        ]
        assert rows.min() >= inner_start
        assert rows.max() + HORIZON <= inner_end - 1
        assert rows.max() < boundary


def test_fitting_takes_no_validation_argument():
    """There is no parameter an early-stopping or calibration set could arrive through."""
    import inspect

    parameters = set(inspect.signature(simple_models.fit_simple_model).parameters)
    assert parameters == {"spec", "X_train", "y_train", "seed"}


def test_no_sealed_row_is_fitted_selected_or_scored(artifact, predictions):
    """Every row the run touched lies strictly below the sealed boundary."""
    boundary = artifact["sealed_test"]["start_row"]
    assert artifact["test_evaluated"] is False
    assert artifact["sealed_test"]["evaluated"] is False
    for fold in artifact["folds"]:
        for region in ("train", "inner_validation", "outer_validation"):
            assert fold["periods"][region]["row_range"][1] <= boundary
    assert int(predictions["row_index"].max()) < boundary


def test_writing_a_sealed_prediction_is_refused(tmp_path):
    """The guard on the one file that carries per-row market outcomes actually fires."""
    frame = pd.DataFrame({"row_index": [10, 11, 12]})
    with pytest.raises(AssertionError, match="sealed test block"):
        benchmark.write_predictions(tmp_path, [frame], boundary=12)


def test_outer_validation_is_scored_but_never_fitted(artifact):
    """Outer numbers exist; no selection field is recorded against that block."""
    for fold in artifact["folds"]:
        for model in SIMPLE_MODEL_NAMES:
            selection = fold["models"][model]["selection"]
            assert selection["block"] == "inner_validation"
            assert fold["outer_validation"][model]["trading"]["n_trades"] >= 0
    assert artifact["tuning"] == {
        "search": "none",
        "hyperparameter_optimisation": None,
        "early_stopping": False,
        "outer_validation_used_for": "evaluation only",
    }


def test_threshold_rule_is_mtsts_own(artifact):
    """The grid, the objective and the trade floor are the ones MTST already used."""
    expected = [float(v) for v in benchmark.threshold_grid()]
    assert artifact["threshold_selection"]["grid"] == expected
    assert artifact["threshold_selection"]["min_trades"] == 10
    for fold in artifact["folds"]:
        for model in SIMPLE_MODEL_NAMES:
            selection = fold["models"][model]["selection"]
            assert selection["threshold_grid"] == expected
            assert selection["threshold"] in expected
            # The outer block applies the already-selected value, unchanged.
            assert (
                fold["outer_validation"][model]["classification"]["threshold"]
                == selection["threshold"]
            )


# --- 3. the artifacts say what happened ---------------------------------------
def test_provenance_records_everything_needed_to_rebuild_the_fit(artifact, data):
    parity = artifact["information_parity"]
    assert parity["seq_len"] == SEQ_LEN
    assert parity["feature_names"] == data.feature_names
    assert parity["flattened_input_dim"] == SEQ_LEN * len(data.feature_names)
    assert parity["flattened_columns"] == simple_models.flattened_column_names(
        data.feature_names, SEQ_LEN
    )
    assert parity["feature_selection"] == "none"

    for fold in artifact["folds"]:
        for model in SIMPLE_MODEL_NAMES:
            provenance = fold["models"][model]["provenance"]
            assert provenance["library_version"]
            assert provenance["early_stopping"] is False
            assert provenance["hyperparameter_search"].startswith("none")
            assert provenance["flattened_input_dim"] == parity["flattened_input_dim"]
            assert provenance["seed"] == fold["fold_seed"]
            spec = next(s for s in simple_models.SIMPLE_MODELS if s.name == model)
            assert provenance["config"] == spec.config(fold["fold_seed"])


def test_persisted_predictions_reproduce_the_reported_metrics(run_dir, artifact, predictions):
    """Every reported outer number is rebuilt from the rows beside it.

    The candle-level statistics are excluded by name rather than approximated:
    they need the price path through the last trade's exit candle, which lies
    past the last scored row and is not in a per-sample file.
    """
    target_spec = TargetSpec.from_dict(artifact["dataset"]["target_spec"])
    checked = 0
    for fold in artifact["folds"]:
        for model in SIMPLE_MODEL_NAMES:
            part = predictions[
                (predictions["fold"] == fold["fold"]) & (predictions["model"] == model)
            ]
            assert len(part) == fold["samples"]["outer_validation"]
            threshold = float(part["threshold"].iloc[0])
            rebuilt = benchmark_compare._report_from_predictions(part, target_spec, threshold)
            assert benchmark_compare._comparable(rebuilt) == benchmark_compare._comparable(
                fold["outer_validation"][model]
            )
            checked += 1
    assert checked == FOLDS * len(SIMPLE_MODEL_NAMES)


def test_persisted_actions_follow_from_the_persisted_probabilities(predictions, artifact):
    """``selected_action`` is the action that was scored, not a second reading of it."""
    proba = predictions[["p_short", "p_hold", "p_long"]].to_numpy(dtype=np.float64)
    thresholds = predictions["threshold"].to_numpy(dtype=np.float64)
    for threshold in np.unique(thresholds):
        mask = thresholds == threshold
        assert np.array_equal(
            ev.signals_from_proba(proba[mask], float(threshold)),
            predictions["selected_action"].to_numpy(dtype=np.int64)[mask],
        )


def test_directional_attribution_is_reproducible_from_the_predictions(predictions, artifact):
    """LONG/SHORT attribution replays the persisted actions through the one cost model."""
    target_spec = TargetSpec.from_dict(artifact["dataset"]["target_spec"])
    for model in SIMPLE_MODEL_NAMES:
        part = predictions[predictions["model"] == model]
        split = regime.direction_attribution(part, target_spec)
        assert split["samples"] == len(part)
        assert split["long"]["trades"] + split["short"]["trades"] > 0
        # The two sides account for every trade the reports counted.
        reported = sum(
            fold["outer_validation"][model]["trading"]["n_trades"]
            for fold in artifact["folds"]
        )
        assert split["long"]["trades"] + split["short"]["trades"] == reported


def test_a_rerun_at_one_seed_reproduces_the_first(dataset, run_dir, tmp_path):
    """Deterministic where the libraries permit it: same seed, same rows, same numbers."""
    again = tmp_path / "again"
    assert benchmark.main(["--dataset", str(dataset), "--out", str(again), *CLI]) == 0
    first = pd.read_parquet(run_dir / benchmark.PREDICTIONS_NAME)
    second = pd.read_parquet(again / benchmark.PREDICTIONS_NAME)
    pd.testing.assert_frame_equal(first, second)

    original = json.loads((run_dir / benchmark.ARTIFACT_NAME).read_text())
    repeat = json.loads((again / benchmark.ARTIFACT_NAME).read_text())
    for payload in (original, repeat):
        payload["config"]["out"] = ""
    assert original == repeat


# --- 4. the comparison fails closed -------------------------------------------
@pytest.fixture(scope="module")
def mtst_run(dataset, tmp_path_factory) -> Path:
    """A real MTST walk-forward run on the same dataset and the same geometry.

    Trained for a single epoch on a tiny model: what matters here is not its
    accuracy but that it is a genuine artifact of the MTST path, planned by the
    same planner over the same rows.
    """
    from nn import walkforward

    out = tmp_path_factory.mktemp("mtst_run") / "seed_42"
    assert (
        walkforward.main(
            [
                "--dataset",
                str(dataset),
                "--out",
                str(out),
                "--epochs",
                "1",
                "--d-model",
                "16",
                "--n-heads",
                "2",
                "--num-layers",
                "1",
                "--device",
                "cpu",
                *CLI,
            ]
        )
        == 0
    )
    return out


def test_the_two_families_scored_the_same_rows_with_the_same_baselines(run_dir, mtst_run):
    """The parity claim, checked on real artifacts rather than asserted in prose."""
    runs = [
        benchmark_compare.load_run(run_dir, kind="p2a"),
        benchmark_compare.load_run(mtst_run, kind="mtst"),
    ]
    assert benchmark_compare.comparability_problems(runs) == []

    p2a = pd.read_parquet(run_dir / benchmark.PREDICTIONS_NAME)
    mtst = pd.read_parquet(mtst_run / "outer_predictions.parquet")
    for fold in sorted(mtst["fold"].unique()):
        mine = np.unique(p2a[p2a["fold"] == fold]["row_index"].to_numpy(dtype=np.int64))
        theirs = np.sort(mtst[mtst["fold"] == fold]["row_index"].to_numpy(dtype=np.int64))
        assert np.array_equal(mine, theirs)


def test_the_comparison_runs_end_to_end(run_dir, mtst_run, dataset, tmp_path):
    out = tmp_path / "comparison"
    assert (
        benchmark_compare.main(
            [
                "--benchmark",
                str(run_dir),
                "--mtst",
                str(mtst_run),
                "--dataset",
                str(dataset),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    report = json.loads((out / benchmark_compare.REPORT_JSON).read_text())
    assert report["mtst_retrained"] is False
    assert report["sealed_test"]["evaluated"] is False
    assert report["recomputation"]["problems"] == []
    for model in SIMPLE_MODEL_NAMES:
        answer = report["answers"][model]
        # The two verdicts are always both present and never collapsed into one.
        assert "predictive_baseline_improvement" in answer
        # And the economic one states the literal fact it computes. The old name,
        # `economic_alpha_after_costs`, was a boolean for `mean(net_return) > 0`
        # over four dependent folds; a reader is entitled to read a field with
        # that name as "alpha was established", and this evidence cannot carry it.
        assert "positive_mean_net_return_after_costs" in answer
        assert "economic_alpha_after_costs" not in answer
        # The point estimate never appears without what qualifies it.
        evidence = answer["mean_net_return_evidence"]
        assert evidence["temporal_folds"] >= 1
        assert "min_outer_trades" in evidence
    markdown = (out / benchmark_compare.REPORT_MD).read_text()
    assert "research evidence" in markdown
    assert "sealed test block is unopened" in markdown


def corrupted(run_dir: Path, tmp_path: Path, mutate) -> Path:
    """A copy of a P2a run with one field of its artifact changed."""
    target = tmp_path / run_dir.name
    target.mkdir(parents=True, exist_ok=True)
    payload = json.loads((run_dir / benchmark.ARTIFACT_NAME).read_text())
    mutate(payload)
    (target / benchmark.ARTIFACT_NAME).write_text(json.dumps(payload))
    return target


@pytest.mark.parametrize(
    "field, mutate",
    [
        (
            "contract hash",
            lambda p: p["sealed_test"]["research_contract"].__setitem__(
                "contract_hash", "0" * 64
            ),
        ),
        (
            "contract id",
            lambda p: p["sealed_test"]["research_contract"].__setitem__(
                "contract_id", "other-gen2"
            ),
        ),
        (
            "research input fingerprint",
            lambda p: p["research_input"].__setitem__("research_input_hash", "f" * 64),
        ),
        (
            "sealed anchor",
            lambda p: p["sealed_test"].__setitem__(
                "anchor_timestamp", "2030-01-01T00:00:00+00:00"
            ),
        ),
        ("sealed row", lambda p: p["sealed_test"].__setitem__("start_row", 1)),
        (
            "fold geometry",
            lambda p: p["folds"][0]["periods"]["outer_validation"].__setitem__(
                "row_range", [0, 5]
            ),
        ),
        ("sequence length", lambda p: p["config"].__setitem__("seq_len", 128)),
        (
            "feature contract",
            lambda p: p["dataset"].__setitem__("feature_names", ["only_one"]),
        ),
        (
            "target semantics",
            lambda p: p["dataset"].__setitem__(
                "target_spec", {"horizon": 99, "cost_threshold": 0.5}
            ),
        ),
        (
            "statistical baselines",
            lambda p: p["folds"][0]["outer_validation"]["majority_baseline"][
                "trading"
            ].__setitem__("net_return", 1.234),
        ),
    ],
)
def test_an_incompatible_run_is_never_aggregated(run_dir, mtst_run, tmp_path, field, mutate):
    """Each way two runs can fail to be one experiment withholds the aggregate."""
    broken = corrupted(run_dir, tmp_path / field.replace(" ", "_"), mutate)
    runs = [
        benchmark_compare.load_run(broken, kind="p2a"),
        benchmark_compare.load_run(mtst_run, kind="mtst"),
    ]
    assert benchmark_compare.comparability_problems(runs), f"{field} mismatch was accepted"

    with pytest.raises(SystemExit, match="refusing to aggregate"):
        benchmark_compare.main(
            [
                "--benchmark",
                str(broken),
                "--mtst",
                str(mtst_run),
                "--out",
                str(tmp_path / "out"),
            ]
        )


def test_a_run_that_opened_the_seal_is_refused(run_dir, mtst_run, tmp_path):
    """P2a compares research folds; a run claiming a sealed-test result is not one."""
    broken = corrupted(
        run_dir, tmp_path / "sealed", lambda p: p.__setitem__("test_evaluated", True)
    )
    runs = [
        benchmark_compare.load_run(broken, kind="p2a"),
        benchmark_compare.load_run(mtst_run, kind="mtst"),
    ]
    problems = benchmark_compare.comparability_problems(runs)
    assert any("sealed test block as evaluated" in problem for problem in problems)


def test_a_benchmark_run_carrying_its_own_mtst_is_refused(run_dir, mtst_run, tmp_path):
    """The frozen evidence is the only MTST in this comparison."""

    def mutate(payload):
        for fold in payload["folds"]:
            fold["outer_validation"]["mtst"] = fold["outer_validation"]["lightgbm"]

    broken = corrupted(run_dir, tmp_path / "retrained", mutate)
    runs = [
        benchmark_compare.load_run(broken, kind="p2a"),
        benchmark_compare.load_run(mtst_run, kind="mtst"),
    ]
    problems = benchmark_compare.comparability_problems(runs)
    assert any("must not be regenerated" in problem for problem in problems)


def test_a_walkforward_artifact_is_not_accepted_as_a_benchmark_run(mtst_run):
    with pytest.raises(SystemExit):
        benchmark_compare.load_run(mtst_run / "walkforward.json", kind="p2a")


def test_differing_scored_rows_are_caught_even_when_the_metadata_agrees(
    run_dir, mtst_run, tmp_path
):
    """Metadata can agree while the arrays differ; the row indices cannot."""
    target = tmp_path / "rows"
    target.mkdir()
    (target / benchmark.ARTIFACT_NAME).write_text(
        (run_dir / benchmark.ARTIFACT_NAME).read_text()
    )
    frame = pd.read_parquet(run_dir / benchmark.PREDICTIONS_NAME)
    frame = frame.iloc[1:].reset_index(drop=True)
    frame.to_parquet(target / benchmark.PREDICTIONS_NAME, index=False)

    runs = [
        benchmark_compare.load_run(target, kind="p2a"),
        benchmark_compare.load_run(mtst_run, kind="mtst"),
    ]
    problems = benchmark_compare.comparability_problems(runs)
    assert any("same samples" in problem for problem in problems)


# --- 5. the frozen BTC v4 evidence is untouched -------------------------------
def test_frozen_v4_artifacts_still_match_their_manifest():
    """Every SHA-covered file hashes to the value the freeze recorded.

    P2a reads this evidence and must never write it. The manifest is the
    repository's own statement of what was frozen, so it is checked here rather
    than restated.
    """
    manifest = (ROOT / "artifacts" / "btc_v4_SHA256SUMS.txt").read_text().splitlines()
    entries = [line.split(maxsplit=1) for line in manifest if line.strip()]
    assert len(entries) == 17
    for expected, name in entries:
        path = ROOT / name.strip()
        assert path.is_file(), f"{name} is missing from the frozen evidence"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name


def test_the_frozen_v4_runs_still_load_and_describe_one_generation():
    """Historical artifact compatibility: the committed evidence stays readable."""
    directories = sorted((ROOT / "artifacts" / "walkforward").glob("btc_nested_v4_seed_*"))
    assert len(directories) == 5
    runs = [benchmark_compare.load_run(path, kind="mtst") for path in directories]
    assert benchmark_compare.comparability_problems(runs) == []
    for run in runs:
        assert run.contract_id == "btc-usdt-1h-gen1"
        assert (
            run.contract_hash
            == "dca6d0a891a257197a7c8aecec04fdba0a0b3009cfe93c5f3a397d458ab4a1de"
        )
        assert (
            run.research_input_hash
            == "1c09218442cdf98cfe2f49ac521c70b2d7692717d7488e8105c972a3d2ac4740"
        )
        assert run.sealed_anchor == "2025-08-27T23:00:00+00:00"
        assert run.sealed_start_row == 48217
        assert run.seq_len == 64
        assert len(run.folds) == 4
        assert run.payload["test_evaluated"] is False


def test_a_p2a_run_cannot_be_aggregated_with_the_frozen_btc_evidence_by_accident(
    run_dir, tmp_path
):
    """A synthetic P2a run and the BTC v4 evidence are different experiments.

    The fail-closed check is what stops a benchmark run on one market being
    averaged into another's frozen numbers because both happen to have four
    folds and a sealed row.
    """
    frozen = ROOT / "artifacts" / "walkforward" / "btc_nested_v4_seed_42"
    runs = [
        benchmark_compare.load_run(run_dir, kind="p2a"),
        benchmark_compare.load_run(frozen, kind="mtst"),
    ]
    problems = benchmark_compare.comparability_problems(runs)
    assert problems
    with pytest.raises(SystemExit, match="refusing to aggregate"):
        benchmark_compare.main(
            [
                "--benchmark",
                str(run_dir),
                "--mtst",
                str(frozen),
                "--out",
                str(tmp_path / "never"),
            ]
        )
    assert not (tmp_path / "never").exists()
