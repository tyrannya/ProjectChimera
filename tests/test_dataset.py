"""Splitting, windowing and scaling — the leakage tests.

Everything here exists to make one claim checkable: no fitted quantity ever
sees data from the block it is later scored on.
"""

from __future__ import annotations

import numpy as np
import pytest

from nn.dataset import (
    Split,
    StandardScaler,
    build_windows,
    chronological_split,
    window_indices,
)


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(500, 4))
    targets = rng.integers(0, 3, size=500)
    return features, targets


# --- splitting -----------------------------------------------------------
def test_splits_are_contiguous_and_ordered():
    plan = chronological_split(1000, 0.7, 0.15)
    assert (plan.train.start, plan.train.end) == (0, 700)
    assert (plan.validation.start, plan.validation.end) == (700, 850)
    assert (plan.test.start, plan.test.end) == (850, 1000)


def test_splits_cover_every_row_exactly_once():
    plan = chronological_split(997, 0.6, 0.2)
    covered = []
    for split in plan:
        covered.extend(range(split.start, split.end))
    assert covered == list(range(997))


def test_splits_are_deterministic():
    assert chronological_split(1000).to_dict() == chronological_split(1000).to_dict()


@pytest.mark.parametrize(
    ("train", "val"), [(0.0, 0.2), (0.9, 0.2), (-0.1, 0.2), (0.5, 0.6), (0.5, 0.0)]
)
def test_invalid_fractions_are_rejected(train, val):
    with pytest.raises(ValueError):
        chronological_split(1000, train, val)


# --- windowing -----------------------------------------------------------
def test_window_stays_inside_its_split():
    split = Split("validation", 100, 200)
    idx = window_indices(split, seq_len=10, horizon=5)
    # Earliest sample needs 9 prior rows, so it starts at 109.
    assert idx[0] == 109
    # Latest sample's label reads row i+5, which must stay below 200.
    assert idx[-1] == 194


def test_no_sample_index_is_shared_between_splits(toy):
    features, targets = toy
    plan = chronological_split(len(features))
    seq_len, horizon = 20, 5

    used: dict[str, set[int]] = {}
    for split in plan:
        _, _, idx = build_windows(features, targets, split, seq_len, horizon)
        # Every row this split's samples touch: the window and the label row.
        touched: set[int] = set()
        for i in idx:
            touched.update(range(i - seq_len + 1, i + horizon + 1))
        used[split.name] = touched

    assert not used["train"] & used["validation"]
    assert not used["validation"] & used["test"]
    assert not used["train"] & used["test"]


def test_training_labels_never_read_validation_prices(toy):
    """The subtle direction of leakage: a label at the end of train looks
    ``horizon`` rows ahead, which would land inside validation if unbounded."""
    features, targets = toy
    plan = chronological_split(len(features))
    horizon = 7
    _, _, idx = build_windows(features, targets, plan.train, 20, horizon)
    assert (idx + horizon).max() < plan.validation.start


def test_window_contents_are_the_preceding_rows(toy):
    features, targets = toy
    split = Split("train", 0, 100)
    X, y, idx = build_windows(features, targets, split, seq_len=8, horizon=3)
    for k, i in enumerate(idx[:5]):
        np.testing.assert_allclose(X[k], features[i - 7 : i + 1])
        assert y[k] == targets[i]


def test_too_short_a_split_yields_no_samples(toy):
    features, targets = toy
    X, y, idx = build_windows(features, targets, Split("test", 0, 10), 20, 5)
    assert len(X) == 0 and len(y) == 0 and len(idx) == 0
    assert X.shape[1:] == (20, 4)


def test_build_windows_rejects_mismatched_lengths(toy):
    features, targets = toy
    with pytest.raises(ValueError, match="same length"):
        build_windows(features, targets[:-1], Split("train", 0, 100), 5, 1)


# --- scaling -------------------------------------------------------------
def test_scaler_standardises_the_data_it_was_fitted_on(toy):
    features, _ = toy
    scaler = StandardScaler().fit(features)
    scaled = scaler.transform(features)
    np.testing.assert_allclose(scaled.mean(axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(scaled.std(axis=0), 1.0, atol=1e-5)


def test_scaler_fitted_on_train_does_not_use_validation_statistics(toy):
    """Fitting on the training block only must give different parameters than
    fitting on everything — otherwise the split is not doing anything."""
    features, _ = toy
    plan = chronological_split(len(features))

    train_only = StandardScaler().fit(features[plan.train.start : plan.train.end])
    everything = StandardScaler().fit(features)
    assert not np.allclose(train_only.mean, everything.mean)


def test_scaler_survives_a_constant_feature():
    X = np.column_stack([np.ones(50), np.arange(50, dtype=float)])
    scaled = StandardScaler().fit_transform(X)
    assert np.isfinite(scaled).all()
    assert np.allclose(scaled[:, 0], 0.0)


def test_unfitted_scaler_refuses_to_transform(toy):
    features, _ = toy
    with pytest.raises(RuntimeError, match="not been fitted"):
        StandardScaler().transform(features)


def test_scaler_roundtrips_through_metadata_lists(toy):
    features, _ = toy
    original = StandardScaler().fit(features)
    restored = StandardScaler(original.mean.tolist(), original.std.tolist())
    np.testing.assert_allclose(original.transform(features), restored.transform(features))
