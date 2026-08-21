"""Leakage properties that structure guarantees today, asserted here instead.

Four properties below hold because of how the code is *shaped*: the threshold is
chosen inside :func:`nn.p2b.run_cell` from ``prepared.idx_val``,
:func:`nn.train.prepare_research_windows` slices ``[:val_split.end]`` before it
windows anything, :func:`nn.dataset.sample_indices` derives its bounds from the
split it is handed, and :func:`nn.walkforward.plan_nested_folds` refuses a
geometry that does not fit. Nothing asserted any of them, and every one of them
fails *silently*: a threshold fitted on the block it is reported on still
produces a plausible net return, a scaler that saw one extra row still
converges, a window that overlaps its own label still trains. The artifact looks
identical either way, which is the failure this repository exists to prevent.

So each is restated as a test, in the terms the leakage would actually take:

A. **Threshold selection never sees an outer row.** The decision threshold is a
   fitted parameter chosen on the inner block. If the rows it is fitted on ever
   intersect the block the fold is *reported* on, the reported net return is a
   selection score wearing the name of a result. This is the assertion in this
   file that matters most.
B. **Nothing fitted reaches past the fold's visible bound.** Asserted causally
   rather than by inspection: every row from the bound onwards is replaced with
   NaN and the run must come back bit-identical to the clean one. Checking index
   arithmetic alone would pass over a scaler that read the forbidden rows and
   averaged them into a mean that still looks reasonable.
C. **A window's inputs never overlap its own label row.** The label for row
   ``i`` is derived from ``close[i + horizon]``
   (:class:`chimera.contracts.TargetSpec`), so a window reaching row ``i + 1``
   would hold part of its own answer. Asserted against the materialised tensor,
   not against the index arithmetic that produced it.
D. **The fold plan cannot reach the sealed block.** A geometry one row too long
   must raise rather than borrow that row from the sealed side of Styx.

The synthetic dataset is ``tests/test_p2b.py``'s — same generator, same seed,
same two folds — so a difference between the two files is a difference in what
is asserted, not in what was run.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from chimera.contracts import TargetSpec
from chimera.features import FeatureSpec
from nn import evaluate as ev
from nn import p2b
from nn.data_pipeline import build_dataset
from nn.dataset import Split, StandardScaler, build_windows, sample_indices
from nn.information_sets import OHLCV14, build_information_set_views
from nn.simple_models import LOGISTIC_REGRESSION
from nn.train import ResearchData, prepare_research_windows, resolve_device, score_frozen_split
from nn.walkforward import FoldPlan, plan_nested_folds
from tools.make_sample_data import generate_candles

ROOT = Path(__file__).resolve().parents[1]
REAL_MANIFEST = ROOT / "data" / "research" / "btc_usdt_1h_gen1_snapshot_manifest.json"

SEQ_LEN = 6
HORIZON = 4

#: The two folds ``tests/test_p2b.py`` runs, in the shape ``plan_nested_folds``
#: emits. Deliberately the same numbers: these tests and that file must be
#: talking about one run.
FOLDS = (
    FoldPlan(
        train=Split("train", 0, 300),
        inner=Split("inner_validation", 300, 400),
        outer=Split("outer_validation", 400, 500),
    ),
    FoldPlan(
        train=Split("train", 0, 400),
        inner=Split("inner_validation", 400, 500),
        outer=Split("outer_validation", 500, 600),
    ),
)

#: Low enough that the grid's permissive end fires on this small synthetic
#: sample, so the scoring path is exercised with real trades rather than none.
PROBE_THRESHOLD = 0.34


@pytest.fixture(scope="module")
def raw_candles():
    """Deterministic synthetic OHLCV, dated well before any sealed instant."""
    return generate_candles(rows=700, seed=11, start="2021-01-01")


@pytest.fixture(scope="module")
def spine(raw_candles):
    return build_dataset(
        raw_candles,
        FeatureSpec(),
        TargetSpec(horizon=HORIZON),
        exchange="synthetic",
        pair="SYNTH/USDT",
        timeframe="1h",
    )


@pytest.fixture(scope="module")
def aligned(spine, raw_candles):
    """The control view alone: leakage is not a property of the column set."""
    frame, ds_meta = spine
    return build_information_set_views(frame, ds_meta, raw_candles, names=(OHLCV14,))


@pytest.fixture(scope="module")
def real_manifest() -> dict[str, Any]:
    return json.loads(REAL_MANIFEST.read_text())


def poison_from(data: ResearchData, first_unreadable_row: int) -> ResearchData:
    """A copy of ``data`` whose rows from ``first_unreadable_row`` on are junk.

    NaN in the floats and an impossible class in the labels, so that reading any
    one of those rows changes the answer instead of nudging it. A result that
    survives this unchanged did not read them.
    """
    features = data.features.copy()
    features[first_unreadable_row:] = np.nan
    future_return = data.future_return.copy()
    future_return[first_unreadable_row:] = np.nan
    close = data.close.copy()
    close[first_unreadable_row:] = np.nan
    targets = data.targets.copy()
    targets[first_unreadable_row:] = -7
    return dataclasses.replace(
        data,
        features=features,
        future_return=future_return,
        close=close,
        targets=targets,
    )


def probe_proba(X: np.ndarray) -> np.ndarray:
    """Stand-in for a fitted model: probabilities that depend on every input.

    A constant would defeat the poisoning above. The point of this stand-in is
    that a value from a row the split may not read reaches the output *if* it is
    read, so the comparison against the clean run can detect it.
    """
    tilt = np.tanh(X.reshape(len(X), -1).mean(axis=1))
    return np.column_stack([0.4 - 0.2 * tilt, np.full(len(X), 0.2), 0.4 + 0.2 * tilt])


# --------------------------------------------------------------------------- #
# A. threshold selection never sees an outer row
# --------------------------------------------------------------------------- #
def test_threshold_selection_never_sees_a_row_outside_the_inner_block(aligned, monkeypatch):
    """Every row the threshold is fitted on lies strictly inside that fold's
    inner block — window inputs, scored row and label row alike.

    If this fails, the net returns reported for a fold were produced at a
    threshold chosen on the rows they are reported on.
    """
    seen: list[np.ndarray] = []
    real = ev.select_threshold

    def spy(*args, **kwargs):
        seen.append(np.asarray(kwargs["row_index"]))
        return real(*args, **kwargs)

    monkeypatch.setattr(ev, "select_threshold", spy)
    p2b.run_cell(
        aligned,
        OHLCV14,
        LOGISTIC_REGRESSION,
        FOLDS,
        seed=42,
        seq_len=SEQ_LEN,
        min_trades=3,
    )

    assert len(seen) == len(FOLDS)
    for plan, idx in zip(FOLDS, seen):
        assert idx.size > 0
        # The rows scored while choosing the threshold.
        assert idx.min() >= plan.inner.start
        assert idx.max() < plan.inner.end
        # The oldest input row of the oldest window.
        assert idx.min() - SEQ_LEN + 1 >= plan.inner.start
        # The newest label row of the newest window: the threshold's objective
        # is a net return, and the return at row i is realised at i + horizon.
        assert idx.max() + HORIZON < plan.inner.end
        # No outer row under any of those three readings.
        assert idx.max() + HORIZON < plan.outer.start


# --------------------------------------------------------------------------- #
# B. nothing fitted sees a row at or beyond the fold's visible bound
# --------------------------------------------------------------------------- #
def test_the_scaler_is_fitted_on_the_training_block_and_nothing_else(aligned):
    view = aligned.views[OHLCV14]
    for plan in FOLDS:
        prepared = prepare_research_windows(view, plan.train, plan.inner, SEQ_LEN)
        reference = StandardScaler().fit(view.features[plan.train.start : plan.train.end])
        assert np.array_equal(prepared.scaler.mean, reference.mean)
        assert np.array_equal(prepared.scaler.std, reference.std)

        # The equality is not vacuous: a scaler that also saw the inner block
        # would land somewhere else on this data.
        wider = StandardScaler().fit(view.features[plan.train.start : plan.inner.end])
        assert not np.allclose(prepared.scaler.mean, wider.mean)


def test_train_and_inner_windows_read_no_row_at_or_beyond_the_inner_bound(aligned):
    """Poison every row from ``plan.inner.end`` on; the fold must not notice."""
    view = aligned.views[OHLCV14]
    for plan in FOLDS:
        clean = prepare_research_windows(view, plan.train, plan.inner, SEQ_LEN)
        poisoned = prepare_research_windows(
            poison_from(view, plan.inner.end), plan.train, plan.inner, SEQ_LEN
        )
        assert np.array_equal(poisoned.scaler.mean, clean.scaler.mean)
        assert np.array_equal(poisoned.scaler.std, clean.scaler.std)
        assert np.array_equal(poisoned.X_train, clean.X_train)
        assert np.array_equal(poisoned.y_train, clean.y_train)
        assert np.array_equal(poisoned.X_val, clean.X_val)
        assert np.array_equal(poisoned.y_val, clean.y_val)
        assert np.array_equal(poisoned.idx_val, clean.idx_val)

        idx_train = sample_indices(plan.train, SEQ_LEN, HORIZON, segment_ids=view.segment_ids)
        assert len(idx_train) == len(clean.X_train)
        for idx, split in ((idx_train, plan.train), (clean.idx_val, plan.inner)):
            assert idx.size > 0
            assert idx.min() - SEQ_LEN + 1 >= split.start
            assert idx.max() + HORIZON < plan.inner.end


def test_the_outer_scoring_path_reads_no_row_at_or_beyond_the_outer_bound(aligned):
    """Same poisoning, at ``plan.outer.end``, through the frozen scorer."""
    view = aligned.views[OHLCV14]
    device = resolve_device("cpu")
    for plan in FOLDS:
        prepared = prepare_research_windows(view, plan.train, plan.inner, SEQ_LEN)

        def score(data: ResearchData):
            return score_frozen_split(
                data,
                prepared.scaler,
                plan.outer,
                SEQ_LEN,
                baselines={},
                threshold=PROBE_THRESHOLD,
                device=device,
                proba_of=probe_proba,
                model_name="probe",
            )

        clean = score(view)
        poisoned = score(poison_from(view, plan.outer.end))
        assert np.array_equal(poisoned.idx, clean.idx)
        assert np.array_equal(poisoned.proba, clean.proba)
        assert np.array_equal(poisoned.y, clean.y)
        assert poisoned.reports == clean.reports
        assert poisoned.economic_references == clean.economic_references

        # Non-vacuous: the report is of a strategy that actually traded, so the
        # return path and the price path were both exercised.
        assert clean.reports["probe"]["trading"]["n_trades"] > 0
        assert clean.idx.size > 0
        assert clean.idx.min() - SEQ_LEN + 1 >= plan.outer.start
        assert clean.idx.max() + HORIZON < plan.outer.end


# --------------------------------------------------------------------------- #
# C. a window's input rows never overlap its own label row
# --------------------------------------------------------------------------- #
def test_a_windows_inputs_never_overlap_its_own_label_row(aligned):
    view = aligned.views[OHLCV14]
    n_rows = aligned.n_rows
    split = Split("probe", 0, n_rows)
    # A market-data gap the synthetic history does not have, so the same-segment
    # clause below is a claim about behaviour rather than about a constant array.
    segments = view.segment_ids.copy()
    segments[n_rows // 2 :] += 1
    # Every value in a row is that row's own index, so the materialised tensor
    # says which rows it was built from instead of being taken on trust.
    marker = np.arange(n_rows, dtype=np.float64).reshape(-1, 1)

    for seq_len in (1, 2, 6, 16, 64):
        for horizon in (1, 2, 4, 6):
            idx = sample_indices(split, seq_len, horizon, segment_ids=segments)
            assert idx.size > 0
            # The gap really drops samples; without that, everything the
            # segment assertions below check would hold for the wrong reason.
            assert idx.size < sample_indices(split, seq_len, horizon).size

            X, y, built = build_windows(
                marker, marker.ravel(), split, seq_len, horizon, segment_ids=segments
            )
            assert np.array_equal(built, idx)

            input_rows = X[:, :, 0].astype(np.int64)
            label_rows = idx + horizon
            # The window is exactly [i - seq_len + 1, i] ...
            assert np.array_equal(input_rows, idx[:, None] + np.arange(-seq_len + 1, 1))
            # ... so its newest row precedes the label row: the two are disjoint.
            assert (input_rows.max(axis=1) < label_rows).all()
            assert np.array_equal(y, idx)
            # Both ends inside the block.
            assert input_rows.min() >= split.start
            assert label_rows.max() <= split.end - 1
            # And inside one contiguous segment, inputs and label alike.
            assert (segments[input_rows] == segments[idx][:, None]).all()
            assert np.array_equal(segments[label_rows], segments[idx])


# --------------------------------------------------------------------------- #
# D. the fold plan cannot reach the sealed block
# --------------------------------------------------------------------------- #
def test_a_fold_plan_one_row_too_long_is_refused_rather_than_shortened():
    """(folds, min_train, inner_size, outer_size, step), including P2b's own."""
    geometries = (
        (4, 21697, 4821, 4821, 4821),
        (1, 50, 10, 10, 10),
        (2, 300, 100, 100, 100),
        (3, 1000, 250, 250, 300),
    )
    for folds, min_train, inner_size, outer_size, step in geometries:
        needed = min_train + step * (folds - 1) + inner_size + outer_size

        # The largest plan that fits: the last outer block ends exactly at the
        # boundary, so the last row it scores is boundary - 1.
        plans = plan_nested_folds(needed, folds, min_train, inner_size, outer_size, step)
        assert len(plans) == folds
        assert plans[-1].outer.end == needed
        assert max(p.outer.end for p in plans) - 1 < needed

        # One row short, and the plan must not exist at all.
        with pytest.raises(ValueError, match="lie before the sealed test block"):
            plan_nested_folds(needed - 1, folds, min_train, inner_size, outer_size, step)


def test_the_committed_manifest_geometry_lands_exactly_on_45802(real_manifest):
    """The snapshot holds 45,802 rows because that is where the plan stops.

    ``tests/test_p2b.py`` asserts the fold rows themselves; what is asserted
    here is the relationship between the plan's furthest reach, the rows that
    were exported, and the research region — the three numbers that have to
    agree for "research never touched a sealed row" to mean anything.
    """
    folds, sizes = p2b.plan_from_manifest(real_manifest, 45802)
    reference = real_manifest["canonical_reference"]

    assert folds[-1].outer.end == 45802
    assert reference["max_outer_end_row"] == 45802
    assert real_manifest["processed_outer_coverage"]["rows"] == 45802
    # And 45,802 is well inside the research region, which itself stops at Styx.
    assert sizes["research_rows"] == reference["research_rows"] == 48217
    assert folds[-1].outer.end + reference["outer_to_styx_margin_rows"] == 48217

    # A snapshot one row shorter cannot carry this plan, and is refused rather
    # than having the plan shrunk to fit what happens to be present.
    with pytest.raises(p2b.SnapshotError, match="holds only 45801"):
        p2b.plan_from_manifest(real_manifest, 45801)
