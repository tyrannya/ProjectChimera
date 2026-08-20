"""The sealed-test boundary is an instant, not a row.

The boundary used to be ``int(n_rows * (train_frac + val_frac))``. That number
moves forward every time a candle is appended, so a timestamp that was sealed
when one run was planned was research data by the next, and the held-out estimate
quietly shrank as the dataset grew. It was never visible in an artifact, because
every artifact recorded the row it had computed and nothing about what that row
meant in wall-clock time.

This suite holds the replacement contract::

    research: timestamp <  SEALED_TEST_START_UTC
    sealed:   timestamp >= SEALED_TEST_START_UTC

and the property that follows from it and is the whole point of the change:
**appending rows after the anchor cannot move the seal.** Everything here is
functional — resolved boundaries, split plans, fold geometry and persisted
prediction timestamps — rather than assertions about the source text, because a
grep for ``n_rows *`` would have passed against the very implementation this
replaces.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chimera.contracts import TargetSpec
from chimera.features import FeatureSpec
from nn import experiment, train, walkforward
from nn.data_pipeline import build_dataset, save_dataset
from nn.dataset import chronological_split, resolve_sealed_boundary
from nn.research_contract import (
    DEFAULT_CONTRACT_ID,
    SYNTHETIC_CONTRACT_ID,
    load_contract,
)
from tools.make_sample_data import generate_candles

REPO = Path(__file__).resolve().parents[1]

#: The committed contract of the current BTC research generation, and the
#: instant it seals at. Read from the registry rather than restated, so this
#: suite tests the contract the entrypoints actually resolve.
CONTRACT = load_contract(DEFAULT_CONTRACT_ID)
SEALED_TEST_START_UTC = CONTRACT.sealed_test_start

#: The contract the synthetic fixtures below belong to. It seals at the same
#: instant, so every assertion about the boundary still reads against
#: :data:`SEALED_TEST_START_UTC`; what differs is its scope, which is what lets
#: the entrypoints run against synthetic candles at all.
SYNTHETIC = load_contract(SYNTHETIC_CONTRACT_ID)
UNDER_SYNTHETIC = ["--research-contract", SYNTHETIC_CONTRACT_ID]

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


def hours(count: int, *, end_before_anchor: int = 0) -> pd.DatetimeIndex:
    """``count`` hourly stamps ending ``end_before_anchor`` hours before the anchor."""
    last = SEALED_TEST_START_UTC - pd.Timedelta(hours=end_before_anchor)
    return pd.date_range(end=last, periods=count, freq="1h")


def straddling(before: int, after: int) -> pd.DatetimeIndex:
    """``before`` hourly stamps below the anchor, then the anchor and ``after`` more."""
    return hours(before + after + 1, end_before_anchor=-after)


# --- A. resolution -----------------------------------------------------------
def test_the_first_row_at_or_after_the_anchor_is_the_sealed_start():
    dates = straddling(before=100, after=50)
    boundary = resolve_sealed_boundary(dates, contract=CONTRACT)

    assert boundary.anchor == SEALED_TEST_START_UTC
    assert boundary.start_row == 100
    assert (dates[: boundary.start_row] < SEALED_TEST_START_UTC).all()
    assert (dates[boundary.start_row :] >= SEALED_TEST_START_UTC).all()


def test_a_timestamp_exactly_on_the_anchor_is_sealed():
    """The partition is half-open the sealed way: ``>= anchor`` is withheld."""
    dates = straddling(before=100, after=50)
    boundary = resolve_sealed_boundary(dates, contract=CONTRACT)

    assert dates[boundary.start_row] == SEALED_TEST_START_UTC
    assert dates[boundary.start_row - 1] < SEALED_TEST_START_UTC


def test_the_anchor_candle_does_not_have_to_exist():
    """A market-data gap over the anchor seals the first surviving row after it.

    No candle is invented to make the timestamp present; the boundary is defined
    by the comparison, not by a lookup.
    """
    dates = straddling(before=100, after=50).delete(100)  # the anchor row itself

    boundary = resolve_sealed_boundary(dates, contract=CONTRACT)
    assert boundary.anchor == SEALED_TEST_START_UTC
    assert boundary.start_row == 100
    assert dates[100] == SEALED_TEST_START_UTC + pd.Timedelta(hours=1)
    assert dates[99] < SEALED_TEST_START_UTC


def test_naive_timestamps_are_read_as_utc():
    """The datasets in this repository store UTC; a naive column is not a second
    time zone to guess at."""
    aware = straddling(before=40, after=10)
    naive = aware.tz_localize(None).to_numpy()

    assert resolve_sealed_boundary(naive, contract=CONTRACT) == resolve_sealed_boundary(
        aware, contract=CONTRACT
    )


def test_the_anchor_is_the_committed_contract():
    assert SEALED_TEST_START_UTC == pd.Timestamp("2025-08-27T23:00:00+00:00")
    assert SEALED_TEST_START_UTC.isoformat() == "2025-08-27T23:00:00+00:00"


# --- B. appending future data does not move the seal -------------------------
#
# The core regression for this change.


@pytest.mark.parametrize("appended", [0, 1, 100, 10_000])
def test_appending_rows_after_the_anchor_never_moves_the_seal(appended):
    original = straddling(before=5_000, after=200)
    grown = original.append(
        pd.date_range(start=original[-1] + pd.Timedelta(hours=1), periods=appended, freq="1h")
    )

    base = resolve_sealed_boundary(original, contract=CONTRACT)
    after = resolve_sealed_boundary(grown, contract=CONTRACT)

    assert after.anchor == base.anchor == SEALED_TEST_START_UTC
    # Unchanged pre-anchor history means an unchanged resolved row, too.
    assert after.start_row == base.start_row
    # And the research region holds exactly the same timestamps, not merely the
    # same number of them.
    assert list(grown[: after.start_row]) == list(original[: base.start_row])


def test_a_year_of_appended_candles_cannot_unseal_a_sealed_timestamp():
    """The question this change exists to answer, asked directly."""
    original = straddling(before=5_000, after=200)
    grown = original.append(
        pd.date_range(start=original[-1] + pd.Timedelta(hours=1), periods=24 * 365, freq="1h")
    )

    was_sealed = set(
        original[resolve_sealed_boundary(original, contract=CONTRACT).start_row :]
    )
    now_research = set(grown[: resolve_sealed_boundary(grown, contract=CONTRACT).start_row])

    assert not (was_sealed & now_research)


def test_appending_grows_only_the_sealed_block():
    original = straddling(before=1_000, after=100)
    grown = original.append(
        pd.date_range(start=original[-1] + pd.Timedelta(hours=1), periods=500, freq="1h")
    )

    base, after = resolve_sealed_boundary(
        original, contract=CONTRACT
    ), resolve_sealed_boundary(grown, contract=CONTRACT)
    base_plan = chronological_split(len(original), base.start_row)
    after_plan = chronological_split(len(grown), after.start_row)

    # Train and validation are byte-identical; only the sealed block is longer.
    assert base_plan.train == after_plan.train
    assert base_plan.validation == after_plan.validation
    assert after_plan.test.start == base_plan.test.start
    assert len(after_plan.test) == len(base_plan.test) + 500


# --- C. the row index is not the contract ------------------------------------
def test_inserting_a_historical_row_moves_the_row_but_not_the_seal():
    """A repair to history before the anchor may move ``start_row``. It must not
    move which *timestamps* are research and which are sealed.

    This is the distinction the whole change rests on: the row index is metadata
    about one dataset, and timestamp membership is the contract.
    """
    original = straddling(before=200, after=50)
    inserted = original[0] - pd.Timedelta(minutes=30)
    repaired = original.insert(0, inserted).sort_values()

    base = resolve_sealed_boundary(original, contract=CONTRACT)
    after = resolve_sealed_boundary(repaired, contract=CONTRACT)

    # The anchor is untouched, and the row moved by exactly the one row added.
    assert after.anchor == base.anchor == SEALED_TEST_START_UTC
    assert after.start_row == base.start_row + 1

    # Every timestamp that existed before keeps exactly its old classification.
    was_research = set(original[: base.start_row])
    was_sealed = set(original[base.start_row :])
    now_research = set(repaired[: after.start_row])
    now_sealed = set(repaired[after.start_row :])

    assert was_research <= now_research
    assert was_sealed == now_sealed
    assert not (was_research & now_sealed)

    # The newly inserted pre-anchor row is research, and the first original
    # timestamp at or after the anchor is still sealed.
    assert inserted in now_research
    assert original[base.start_row] in now_sealed


# --- F. bad data fails closed ------------------------------------------------
def test_unsorted_timestamps_are_refused():
    dates = straddling(before=50, after=20).to_numpy()
    dates[[10, 40]] = dates[[40, 10]]
    with pytest.raises(ValueError, match="not sorted ascending"):
        resolve_sealed_boundary(dates, contract=CONTRACT)


def test_duplicate_timestamps_are_refused():
    dates = straddling(before=50, after=20)
    with pytest.raises(ValueError, match="duplicate timestamp"):
        resolve_sealed_boundary(dates.insert(10, dates[10]).sort_values(), contract=CONTRACT)


def test_a_dataset_entirely_before_the_anchor_is_refused():
    """No sealed block to withhold, so there is nothing to hold out."""
    with pytest.raises(ValueError, match="no sealed test block to withhold"):
        resolve_sealed_boundary(hours(500, end_before_anchor=1), contract=CONTRACT)


def test_a_dataset_entirely_at_or_after_the_anchor_is_refused():
    """No research region, so there is nothing to train on."""
    dates = pd.date_range(start=SEALED_TEST_START_UTC, periods=500, freq="1h")
    with pytest.raises(ValueError, match="no research region"):
        resolve_sealed_boundary(dates, contract=CONTRACT)


def test_missing_timestamps_are_refused():
    dates = straddling(before=50, after=20).to_numpy()
    dates[7] = np.datetime64("NaT")
    with pytest.raises(ValueError, match="missing or unparseable"):
        resolve_sealed_boundary(dates, contract=CONTRACT)


def test_malformed_timestamps_are_refused():
    with pytest.raises(ValueError, match="could not be parsed|missing or unparseable"):
        resolve_sealed_boundary(
            np.array(["2025-08-27T23:00:00Z", "not a date"]), contract=CONTRACT
        )


def test_an_empty_date_column_is_refused():
    with pytest.raises(ValueError, match="empty date column"):
        resolve_sealed_boundary(pd.DatetimeIndex([], tz="UTC"), contract=CONTRACT)


# --- the canonical dataset -----------------------------------------------------
#
# data/datasets/binance_BTC_USDT_1h.parquet is not committed, so compatibility is
# checked against the metadata the five committed walk-forward artifacts recorded
# for it. Timestamps and row counts only: no feature, target, return or
# prediction from the sealed block is read here or anywhere else in this suite.

ALL_RUNS = sorted((REPO / "artifacts" / "walkforward").glob("*/walkforward.json"))
HISTORICAL_RUNS = [p for p in ALL_RUNS if "_v4_" not in p.parent.name]
V4_RUNS = [p for p in ALL_RUNS if "_v4_" in p.parent.name]


def test_the_committed_artifacts_are_present_to_check_against():
    assert len(HISTORICAL_RUNS) == 5


@pytest.mark.parametrize("artifact", HISTORICAL_RUNS, ids=lambda p: p.parent.name)
def test_the_anchor_names_the_instant_the_current_generation_sealed_at(artifact):
    """The anchor is not a new decision: it is the seal already in force.

    Each committed run recorded ``start_row: 48217`` on a 56,726-row dataset and,
    beside it, the wall-clock instant that row began at. The anchor is that
    instant, so the boundary this change makes immutable is exactly the boundary
    the current research generation was produced under — the row number is
    evidence of that, not the contract.
    """
    payload = json.loads(artifact.read_text())
    sealed = payload["sealed_test"]

    assert payload["dataset"]["rows"] == 56726
    assert sealed["start_row"] == 48217
    assert pd.Timestamp(sealed["period"]["start"]) == SEALED_TEST_START_UTC


@pytest.mark.parametrize("artifact", HISTORICAL_RUNS, ids=lambda p: p.parent.name)
def test_historical_artifacts_are_not_relabelled_as_timestamp_anchored(artifact):
    """They predate the contract. Saying otherwise would manufacture provenance."""
    sealed = json.loads(artifact.read_text())["sealed_test"]
    assert "anchor_timestamp" not in sealed


def test_the_committed_v4_artifacts_are_present_to_check_against():
    assert len(V4_RUNS) == 5


@pytest.mark.parametrize("artifact", V4_RUNS, ids=lambda p: p.parent.name)
def test_the_v4_anchor_names_the_instant_the_current_generation_sealed_at(artifact):
    """v4 appended rows after the same seal; the anchor instant did not move."""
    payload = json.loads(artifact.read_text())
    sealed = payload["sealed_test"]

    assert sealed["start_row"] == 48217
    assert pd.Timestamp(sealed["anchor_timestamp"]) == SEALED_TEST_START_UTC
    assert pd.Timestamp(sealed["period"]["start"]) == SEALED_TEST_START_UTC


# --- D/E. the entrypoints share one seal, and appending does not move it ------


def dataset_at(path: Path, *, rows: int, appended: int = 0, seed: int = 17) -> Path:
    """A processed dataset straddling the anchor, optionally with rows appended.

    The appended candles continue the same series *after* its last row, which is
    already past the anchor — so they are sealed rows and nothing else.
    """
    candles = generate_candles(rows=rows + appended, seed=seed)
    if appended:
        # Keep the pre-anchor history byte-identical to the un-appended dataset
        # by generating the shorter series and extending it, rather than
        # generating a longer one that would start earlier.
        base = generate_candles(rows=rows, seed=seed)
        tail = candles.iloc[len(candles) - appended :].copy()
        tail["date"] = pd.date_range(
            start=base["date"].iloc[-1] + pd.Timedelta(hours=1), periods=appended, freq="1h"
        )
        candles = pd.concat([base, tail], ignore_index=True)

    frame, metadata = build_dataset(
        candles,
        FeatureSpec(),
        TargetSpec(horizon=4),
        exchange="synthetic",
        pair="SYNTH/USDT",
        timeframe="1h",
    )
    save_dataset(path, frame, metadata)
    return path


@pytest.fixture(scope="module")
def grown_pair(tmp_path_factory):
    """The same dataset, and the same dataset with 400 later candles appended."""
    directory = tmp_path_factory.mktemp("append_invariance")
    base = dataset_at(directory / "base.parquet", rows=1200)
    grown = dataset_at(directory / "grown.parquet", rows=1200, appended=400)
    return train.load_research_data(base), train.load_research_data(grown), base, grown


def test_the_fixture_really_does_append_only_sealed_rows(grown_pair):
    """A control: without this, every assertion below could pass vacuously."""
    base, grown, _, _ = grown_pair
    assert grown.n_rows > base.n_rows
    appended = pd.to_datetime(pd.Index(grown.dates[base.n_rows :]), utc=True)
    assert (appended >= SEALED_TEST_START_UTC).all()
    # And the shared prefix is untouched.
    assert list(pd.to_datetime(pd.Index(grown.dates[: base.n_rows]), utc=True)) == list(
        pd.to_datetime(pd.Index(base.dates), utc=True)
    )


def test_appending_does_not_move_the_split_plan(grown_pair):
    """D: plan.test.start names the same instant, and the pre-seal geometry holds."""
    base, grown, _, _ = grown_pair
    base_boundary, grown_boundary = base.sealed_boundary(SYNTHETIC), grown.sealed_boundary(
        SYNTHETIC
    )

    base_plan = chronological_split(base.n_rows, base_boundary.start_row)
    grown_plan = chronological_split(grown.n_rows, grown_boundary.start_row)

    assert base_boundary.anchor == grown_boundary.anchor == SEALED_TEST_START_UTC
    assert base.dates[base_plan.test.start] == grown.dates[grown_plan.test.start]
    assert base_plan.train == grown_plan.train
    assert base_plan.validation == grown_plan.validation


def test_appending_exposes_no_new_timestamp_to_train_or_validation(grown_pair):
    base, grown, _, _ = grown_pair
    seen = set()
    for data in (base, grown):
        plan = chronological_split(data.n_rows, data.sealed_boundary(SYNTHETIC).start_row)
        stamps = pd.to_datetime(pd.Index(data.dates), utc=True)
        research = set(stamps[: plan.validation.end])
        assert (pd.DatetimeIndex(sorted(research)) < SEALED_TEST_START_UTC).all()
        seen.add(frozenset(research))
    assert len(seen) == 1, "the research region must be the same set of timestamps"


def test_appending_does_not_change_the_walk_forward_plan(grown_pair):
    """E: research_rows and nested fold geometry are unchanged."""
    base, grown, _, _ = grown_pair
    args = walkforward.build_argparser().parse_args(["--dataset", "x", "--folds", "2"])

    geometries = []
    for data in (base, grown):
        boundary = data.sealed_boundary(SYNTHETIC).start_row
        folds = walkforward.plan_nested_folds(
            boundary, args.folds, *walkforward.resolve_sizes(args, boundary)
        )
        geometries.append(
            (
                boundary,
                [
                    (p.train.end, p.inner.start, p.inner.end, p.outer.start, p.outer.end)
                    for p in folds
                ],
                # The newest instant any fold may reach.
                pd.Timestamp(data.dates[folds[-1].outer.end - 1]),
            )
        )
    assert geometries[0] == geometries[1]
    assert geometries[0][2] < SEALED_TEST_START_UTC


def test_the_three_entrypoints_resolve_the_same_seal(grown_pair):
    """nn.train, nn.experiment and nn.walkforward cannot disagree about the seal.

    All three go through ``ResearchData.sealed_boundary`` and none of them takes
    an anchor from anywhere else, so this is one number by construction — but the
    previous design was one number by construction too, and it moved.
    """
    _, grown, _, _ = grown_pair
    boundary = grown.sealed_boundary(SYNTHETIC)
    stamps = pd.to_datetime(pd.Index(grown.dates), utc=True)

    assert boundary.anchor == SEALED_TEST_START_UTC
    assert stamps[boundary.start_row] >= SEALED_TEST_START_UTC
    assert stamps[boundary.start_row - 1] < SEALED_TEST_START_UTC


@pytest.mark.parametrize(
    ("train_frac", "val_frac"), [("0.70", "0.15"), ("0.80", "0.10"), ("0.50", "0.40")]
)
def test_the_split_fractions_cannot_move_the_seal(grown_pair, train_frac, val_frac):
    """Whatever the weights say, the test block starts at the resolved boundary.

    Under the old contract these three pairs put the boundary at three different
    rows — the weights *were* the seal. Now they only divide the research region
    between train and validation.
    """
    _, grown, _, grown_path = grown_pair
    boundary = grown.sealed_boundary(SYNTHETIC).start_row

    args = train.build_argparser().parse_args(
        ["--dataset", str(grown_path), "--train-frac", train_frac, "--val-frac", val_frac]
    )
    plan = chronological_split(grown.n_rows, boundary, args.train_frac, args.val_frac)

    assert plan.test.start == boundary
    assert plan.validation.end == boundary
    assert plan.test.end == grown.n_rows


def test_walk_forward_has_no_flag_that_looks_like_it_moves_the_seal():
    """They existed only to locate the moving boundary. An old command line must
    fail loudly rather than appear to still control the seal."""
    parser = walkforward.build_argparser()
    flags = {action.option_strings and action.option_strings[0] for action in parser._actions}
    assert "--train-frac" not in flags
    assert "--val-frac" not in flags

    with pytest.raises(SystemExit):
        parser.parse_args(["--dataset", "x", "--train-frac", "0.5"])


# --- the entrypoints, run for real -------------------------------------------


def test_the_experiment_manifest_records_the_immutable_anchor(grown_pair, tmp_path):
    _, grown, _, grown_path = grown_pair
    out = tmp_path / "exp"
    assert (
        experiment.main(
            ["--dataset", str(grown_path), "--out", str(out), *UNDER_SYNTHETIC, *TINY]
        )
        == 0
    )

    manifest = json.loads((out / "experiment_plan.json").read_text())
    assert manifest["sealed_test"]["anchor_timestamp"] == SEALED_TEST_START_UTC.isoformat()
    assert manifest["sealed_test"]["start_row"] == grown.sealed_boundary(SYNTHETIC).start_row
    assert manifest["sealed_test"]["evaluated"] is False
    assert json.loads((out / "experiments.json").read_text())["test_evaluated"] is False


def test_the_sealed_contract_is_part_of_the_plan_hash(grown_pair, tmp_path):
    """A plan made under another seal cannot share this plan's identity."""
    _, grown, _, grown_path = grown_pair
    args = experiment.build_argparser().parse_args(
        ["--dataset", str(grown_path), *UNDER_SYNTHETIC]
    )
    configs = experiment.build_grid(
        {name: [getattr(experiment.RunConfig(), name)] for name in experiment.GRID_DIMENSIONS},
        experiment.RunConfig(),
    )
    boundary = grown.sealed_boundary(SYNTHETIC)
    plan = chronological_split(grown.n_rows, boundary.start_row)

    moved = replace(boundary, start_row=boundary.start_row - 1)
    real = experiment.build_plan(
        args, configs, grown, plan, boundary, grown.input_fingerprint(boundary), []
    )
    elsewhere = experiment.build_plan(
        args, configs, grown, plan, moved, grown.input_fingerprint(moved), []
    )
    assert real["plan_hash"] != elsewhere["plan_hash"]


def test_training_records_both_halves_of_the_boundary(grown_pair, tmp_path):
    _, grown, _, grown_path = grown_pair
    models_dir = tmp_path / "models"
    assert (
        train.main(
            [
                "--dataset",
                str(grown_path),
                "--models-dir",
                str(models_dir),
                "--validation-only",
                *UNDER_SYNTHETIC,
                *TINY,
            ]
        )
        == 0
    )

    version = next(p for p in models_dir.iterdir() if p.is_dir())
    report = json.loads((version / "report.json").read_text())

    assert report["sealed_test"]["anchor_timestamp"] == SEALED_TEST_START_UTC.isoformat()
    assert report["sealed_test"]["start_row"] == grown.sealed_boundary(SYNTHETIC).start_row
    assert report["sealed_test"]["evaluated"] is False
    assert report["test_evaluated"] is False
    assert pd.Timestamp(report["periods"]["validation"]["end"]) < SEALED_TEST_START_UTC


def test_every_persisted_outer_prediction_is_before_the_anchor(grown_pair, tmp_path):
    """E, end to end: the one file walk-forward writes with per-row outcomes."""
    _, grown, _, grown_path = grown_pair
    out = tmp_path / "wf"
    assert (
        walkforward.main(
            [
                "--dataset",
                str(grown_path),
                "--out",
                str(out),
                "--folds",
                "2",
                *UNDER_SYNTHETIC,
                *TINY,
            ]
        )
        == 0
    )

    predictions = pd.read_parquet(out / walkforward.PREDICTIONS_NAME)
    stamps = pd.to_datetime(predictions["timestamp"], utc=True)
    assert len(stamps) > 0, "the run must have produced predictions to check"
    assert (stamps < SEALED_TEST_START_UTC).all()

    results = json.loads((out / "walkforward.json").read_text())
    boundary = grown.sealed_boundary(SYNTHETIC)
    assert results["sealed_test"]["anchor_timestamp"] == SEALED_TEST_START_UTC.isoformat()
    assert results["sealed_test"]["start_row"] == boundary.start_row
    assert results["sealed_test"]["evaluated"] is False
    assert results["test_evaluated"] is False
    assert results["research_rows"] == boundary.start_row
    assert int(predictions["row_index"].max()) < boundary.start_row
