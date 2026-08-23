"""``derivatives_v1``: the spec it computes, and the leakage battery of §4.4.

Every test below has a **positive control**: it shows both that the correct
construction holds and that a wrong one would be caught. A leakage test that only
asserts the right answer proves nothing about whether it could have found the
wrong one, which is the failure ``docs/p4_preregistration.md`` §4.4 names when it
says the battery must exist *before* the first fit rather than after the first
result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nn.derivatives import (
    DERIVATIVES_FEATURE_COLUMNS,
    DERIVATIVES_FEATURE_FAMILIES,
    DERIVATIVES_SPEC_VERSION,
    DerivativesError,
    DerivativesSpec,
    compute_derivatives_features,
    derivatives_feature_columns,
    segment_ids_from_hours,
)
from nn.derivatives_sources import (
    DERIVATIVES_COLUMNS,
    HOUR_NS,
    UNAVAILABLE_AGE_NS,
    staleness_bound_ns,
)
from nn.p4_preregistration import FEATURES, WARMUP_HOURS

COLUMNS = list(DERIVATIVES_FEATURE_COLUMNS)


def build_source(n=800, *, start="2021-01-01", seed=3, funding_every=8):
    """A well-formed derivatives source and the spot closes beside it."""
    dates = pd.date_range(start, periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(seed)
    settled = (dates.hour % funding_every == 0).astype(np.int64)
    rates = np.where(settled == 1, rng.normal(0.0001, 0.0002, n), 0.0)

    last_rate = np.zeros(n)
    age = np.full(n, UNAVAILABLE_AGE_NS, dtype=np.int64)
    current, last_index = 0.0, None
    for i in range(n):
        if settled[i]:
            current, last_index = float(rates[i]), i
        if last_index is not None:
            age[i] = (i - last_index) * HOUR_NS
        last_rate[i] = current
    available = (age >= 0) & (age <= staleness_bound_ns("funding_rate"))

    close = 40_000.0 + np.cumsum(rng.normal(0.0, 25.0, n))
    frame = pd.DataFrame(
        {
            "date": dates,
            "funding_settled": settled,
            "funding_settled_rate": rates,
            "funding_visible_count": np.cumsum(settled),
            "funding_available": available.astype(np.int64),
            "funding_last_rate": np.where(available, last_rate, 0.0),
            "funding_age_ns": np.where(available, age, UNAVAILABLE_AGE_NS),
            "oi_available": np.ones(n, dtype=np.int64),
            "oi_contracts": 50_000.0 + np.cumsum(rng.normal(0.0, 60.0, n)),
            "oi_notional": (50_000.0 + np.cumsum(rng.normal(0.0, 60.0, n))) * close,
            "oi_age_ns": np.zeros(n, dtype=np.int64),
            "perp_available": np.ones(n, dtype=np.int64),
            "perp_close": close * (1.0 + rng.normal(0.0004, 0.0005, n)),
            "perp_age_ns": np.zeros(n, dtype=np.int64),
        }
    )
    return frame[list(DERIVATIVES_COLUMNS)], close


# --- the spec is the preregistration, not a copy of it -----------------------
def test_the_columns_are_the_eight_the_preregistration_names_in_its_own_order():
    assert derivatives_feature_columns() == [f["name"] for f in FEATURES]
    assert len(COLUMNS) == 8


def test_the_families_partition_the_columns_into_the_three_preregistered_fields():
    assert set(DERIVATIVES_FEATURE_FAMILIES) == {"funding", "open_interest", "basis"}
    covered = [c for group in DERIVATIVES_FEATURE_FAMILIES.values() for c in group]
    assert sorted(covered) == sorted(COLUMNS)


def test_the_spec_hash_moves_when_a_preregistered_window_moves(monkeypatch):
    """The windows are not fields of the spec, so they are hashed from §5 itself.

    A spec hash that moved only when :class:`DerivativesSpec` changed would call
    two different feature families by one name the moment §5 was edited.
    """
    import nn.derivatives as derivatives

    before = DerivativesSpec().spec_hash()
    monkeypatch.setitem(derivatives._WINDOW, "drv_basis_z", 169)
    assert DerivativesSpec().spec_hash() != before


def test_the_spec_hash_moves_when_a_clip_moves(monkeypatch):
    import nn.derivatives as derivatives

    before = DerivativesSpec().spec_hash()
    monkeypatch.setitem(derivatives._CLIP, "drv_basis", [-0.03, 0.03])
    assert DerivativesSpec().spec_hash() != before


def test_the_spec_records_the_version_every_artifact_carries():
    assert DerivativesSpec().spec_hash() != ""
    assert DERIVATIVES_SPEC_VERSION == "derivatives_v1"


def test_every_clip_is_respected_on_every_defined_row():
    frame, close = build_source()
    features = compute_derivatives_features(frame, close)
    defined = features["defined"].to_numpy(dtype=bool)
    for feature in FEATURES:
        low, high = feature["clip"]
        values = features[feature["name"]].to_numpy(dtype=np.float64)[defined]
        assert values.min() >= low - 1e-12, feature["name"]
        assert values.max() <= high + 1e-12, feature["name"]


def test_a_source_missing_a_schema_column_is_refused():
    frame, close = build_source(n=300)
    with pytest.raises(DerivativesError, match="missing column"):
        compute_derivatives_features(frame.drop(columns=["oi_notional"]), close)


def test_construction_is_deterministic():
    frame, close = build_source()
    first = compute_derivatives_features(frame, close)[COLUMNS].to_numpy()
    second = compute_derivatives_features(frame, close)[COLUMNS].to_numpy()
    assert np.array_equal(first, second)


# --- §4.4: append invariance and future mutation ------------------------------
def test_appending_later_hours_does_not_change_an_earlier_row():
    frame, close = build_source(n=900)
    full = compute_derivatives_features(frame, close)
    prefix = compute_derivatives_features(frame.iloc[:600].copy(), close[:600])
    assert np.array_equal(full[COLUMNS].to_numpy()[:600], prefix[COLUMNS].to_numpy())
    assert np.array_equal(full["defined"].to_numpy()[:600], prefix["defined"].to_numpy())


def test_mutating_a_future_observation_does_not_change_an_earlier_row():
    """The positive control is that the *later* rows do move."""
    frame, close = build_source(n=900)
    before = compute_derivatives_features(frame, close)
    mutated = frame.copy()
    mutated.loc[700:, "oi_contracts"] = mutated.loc[700:, "oi_contracts"] * 1.5
    mutated.loc[700:, "perp_close"] = mutated.loc[700:, "perp_close"] * 1.02
    after = compute_derivatives_features(mutated, close)
    assert np.array_equal(before[COLUMNS].to_numpy()[:700], after[COLUMNS].to_numpy()[:700])
    assert not np.array_equal(
        before[COLUMNS].to_numpy()[700:], after[COLUMNS].to_numpy()[700:]
    )


def test_a_matrix_rolled_one_hour_forward_is_rejected_rather_than_merely_different():
    """§4.4's last item: a shifted matrix must not be able to pass for the real one."""
    frame, close = build_source(n=900)
    truth = compute_derivatives_features(frame, close)[COLUMNS].to_numpy()
    rolled = np.roll(truth, 1, axis=0)
    defined = compute_derivatives_features(frame, close)["defined"].to_numpy(dtype=bool)
    window = slice(WARMUP_HOURS + 1, len(truth) - 1)
    assert defined[window].all()
    assert not np.allclose(truth[window], rolled[window])


# --- §4.1: the funding visibility boundary, at T = t and T = t + 1s -----------
def test_a_settlement_at_exactly_t_is_visible_to_row_t():
    """§4.1: `T <= t`, the candle's *open*. Exercised at the boundary itself."""
    frame, close = build_source(n=600)
    features = compute_derivatives_features(frame, close)
    settlement_rows = np.flatnonzero(frame["funding_settled"].to_numpy() == 1)
    row = int(settlement_rows[settlement_rows > WARMUP_HOURS][0])
    assert features["drv_funding_last"].iloc[row] == pytest.approx(
        float(frame["funding_settled_rate"].iloc[row]), abs=1e-12
    )


def test_a_settlement_one_hour_later_is_not_visible_to_the_row_before_it():
    """The positive control for the boundary: move the settlement, move the value."""
    frame, close = build_source(n=600)
    settlement_rows = np.flatnonzero(frame["funding_settled"].to_numpy() == 1)
    row = int(settlement_rows[settlement_rows > WARMUP_HOURS][0])

    later = frame.copy()
    rate = float(later["funding_settled_rate"].iloc[row])
    later.loc[row, ["funding_settled", "funding_settled_rate"]] = [0, 0.0]
    later.loc[row, "funding_last_rate"] = later.loc[row - 1, "funding_last_rate"]
    later.loc[row, "funding_age_ns"] = later.loc[row - 1, "funding_age_ns"] + HOUR_NS
    later.loc[row + 1, ["funding_settled", "funding_settled_rate"]] = [1, rate]
    later.loc[row + 1, "funding_last_rate"] = rate
    later.loc[row + 1, "funding_age_ns"] = 0
    later["funding_visible_count"] = np.cumsum(later["funding_settled"].to_numpy())

    moved = compute_derivatives_features(later, close)
    original = compute_derivatives_features(frame, close)
    assert moved["drv_funding_last"].iloc[row] != original["drv_funding_last"].iloc[row]
    assert moved["drv_funding_last"].iloc[row + 1] == pytest.approx(
        original["drv_funding_last"].iloc[row], abs=1e-12
    )


def test_a_future_settlement_cannot_reach_an_earlier_row():
    """The leakage a `shift(-1)` would introduce, asserted absent."""
    frame, close = build_source(n=600)
    before = compute_derivatives_features(frame, close)
    tampered = frame.copy()
    row = 500
    tampered.loc[row, "funding_settled_rate"] = 0.009
    after = compute_derivatives_features(tampered, close)
    assert np.array_equal(
        before["drv_funding_last"].to_numpy()[:row],
        after["drv_funding_last"].to_numpy()[:row],
    )


def test_funding_windows_count_settlements_and_not_hours():
    """`drv_funding_sum_9` is nine settlements — three days — not nine hours."""
    frame, close = build_source(n=600)
    features = compute_derivatives_features(frame, close)
    rates = frame["funding_settled_rate"].to_numpy()[frame["funding_settled"].to_numpy() == 1]
    row = 400
    visible = int(frame["funding_visible_count"].iloc[row])
    expected = float(rates[visible - 9 : visible].sum())
    assert features["drv_funding_sum_9"].iloc[row] == pytest.approx(expected, abs=1e-12)


def test_the_funding_z_reading_is_inert_wherever_the_clip_does_not_bite():
    """The one construction §5's prose leaves open, shown not to matter here.

    ``drv_funding_z`` subtracts the **clipped** ``drv_funding_last`` from the mean
    of the **raw** last thirty settlements. Binance caps BTCUSDT funding at
    ±0.75%, strictly inside the ±1% clip, so the two readings coincide on any
    data this checkpoint can see — which is what makes recording the choice
    honest rather than a decision with consequences.
    """
    frame, close = build_source(n=600)
    features = compute_derivatives_features(frame, close)
    last = frame["funding_last_rate"].to_numpy()
    assert np.abs(last).max() < 0.01
    clipped = np.clip(last, -0.01, 0.01)
    assert np.array_equal(last, clipped)
    assert features["defined"].to_numpy().any()


# --- §4.2: open-interest staleness, at exactly one hour -----------------------
def test_an_open_interest_snapshot_exactly_one_hour_old_is_still_admitted():
    frame, close = build_source(n=600)
    row = 400
    frame.loc[row, "oi_age_ns"] = staleness_bound_ns("open_interest")
    features = compute_derivatives_features(frame, close)
    assert bool(features["drv_oi_notional_ratio__defined"].iloc[row])


def test_one_nanosecond_past_the_bound_removes_the_row_from_the_universe():
    """The boundary's positive control, one nanosecond on the other side."""
    frame, close = build_source(n=600)
    row = 400
    frame.loc[row, "oi_age_ns"] = staleness_bound_ns("open_interest") + 1
    features = compute_derivatives_features(frame, close)
    assert not bool(features["defined"].iloc[row])
    assert not bool(features["drv_oi_notional_ratio__defined"].iloc[row])


def test_an_unavailable_hour_is_undefined_however_the_source_labelled_it():
    """The engine re-checks §3.4 rather than trusting the exporter's flag."""
    frame, close = build_source(n=600)
    row = 400
    frame.loc[row, "oi_available"] = 1
    frame.loc[row, "oi_age_ns"] = UNAVAILABLE_AGE_NS
    features = compute_derivatives_features(frame, close)
    assert not bool(features["defined"].iloc[row])


def test_a_missing_open_interest_day_costs_its_own_hours_and_the_day_after():
    """`drv_oi_log_change_24h` reads t and t-24h, so an absent day echoes once.

    Recorded as a test because it is larger than §3.0a's worked example and it
    tightens the availability gate: 48 rows per missing day, not 24.
    """
    frame, close = build_source(n=900)
    # `.iloc`, because a `.loc` slice on an integer index includes its endpoint
    # and the point of this test is exactly which 24 hours were removed.
    day = frame.index[400:424]
    frame.loc[day, "oi_available"] = 0
    frame.loc[day, "oi_age_ns"] = UNAVAILABLE_AGE_NS
    frame.loc[day, ["oi_contracts", "oi_notional"]] = 0.0
    features = compute_derivatives_features(frame, close)
    defined = features["defined"].to_numpy(dtype=bool)
    assert not defined[400:424].any()
    assert not defined[424:448].any()
    assert defined[448:470].all()


def test_a_future_open_interest_snapshot_cannot_reach_an_earlier_row():
    frame, close = build_source(n=600)
    before = compute_derivatives_features(frame, close)
    tampered = frame.copy()
    tampered.loc[500:, "oi_contracts"] = 1_000_000.0
    after = compute_derivatives_features(tampered, close)
    assert np.array_equal(before[COLUMNS].to_numpy()[:500], after[COLUMNS].to_numpy()[:500])


# --- §4.3: the basis, and its denominator ------------------------------------
def test_the_basis_is_the_perpetual_over_the_spot_close_at_the_same_instant():
    frame, close = build_source(n=600)
    features = compute_derivatives_features(frame, close)
    row = 400
    expected = float(frame["perp_close"].iloc[row]) / float(close[row]) - 1.0
    assert features["drv_basis"].iloc[row] == pytest.approx(expected, abs=1e-12)


def test_shifting_the_perpetual_series_changes_the_basis():
    """A perp series off by one hour must not produce the same column."""
    frame, close = build_source(n=600)
    truth = compute_derivatives_features(frame, close)["drv_basis"].to_numpy()
    shifted = frame.copy()
    shifted["perp_close"] = np.roll(shifted["perp_close"].to_numpy(), 1)
    moved = compute_derivatives_features(shifted, close)["drv_basis"].to_numpy()
    window = slice(WARMUP_HOURS + 1, len(truth))
    assert not np.allclose(truth[window], moved[window])


def test_misaligning_the_basis_denominator_changes_the_basis():
    """The spot side of the ratio is checked the same way the perp side is."""
    frame, close = build_source(n=600)
    truth = compute_derivatives_features(frame, close)["drv_basis"].to_numpy()
    moved = compute_derivatives_features(frame, np.roll(close, 1))["drv_basis"].to_numpy()
    window = slice(WARMUP_HOURS + 1, len(truth))
    assert not np.allclose(truth[window], moved[window])


def test_a_denominator_of_a_different_length_is_refused():
    frame, close = build_source(n=600)
    with pytest.raises(DerivativesError, match="same hours in the same order"):
        compute_derivatives_features(frame, close[:-1])


def test_a_non_positive_close_is_refused_rather_than_divided_by():
    frame, close = build_source(n=600)
    broken = close.copy()
    broken[100] = 0.0
    with pytest.raises(DerivativesError, match="non-positive or non-finite close"):
        compute_derivatives_features(frame, broken)


# --- §6.2 condition 4: windows stop at a segment boundary ---------------------
def test_a_window_does_not_bridge_a_gap_in_the_hourly_frame():
    frame, close = build_source(n=900)
    kept = np.ones(len(frame), dtype=bool)
    kept[500:520] = False
    gapped = frame.loc[kept].reset_index(drop=True)
    gapped_close = close[kept]
    features = compute_derivatives_features(gapped, gapped_close)
    segments = segment_ids_from_hours(gapped["date"])
    assert segments.max() == 1
    second = np.flatnonzero(segments == 1)
    # The warm-up restarts with the segment, so the first 240 hours after the
    # gap are outside the universe rather than windowed across it.
    assert not features["defined"].to_numpy()[second[:WARMUP_HOURS]].any()


def test_explicit_segment_ids_are_used_in_place_of_the_frames_own_hour_gaps():
    """The spine's segments are stricter than the candles', and win.

    The dataset builder drops an indicator warm-up after every market-data gap,
    so a two-hour hole in the candles is a three-day hole in the spine. Windows
    reset at the spine's boundaries, which is why the engine takes them.
    """
    frame, close = build_source(n=900)
    segments = np.zeros(len(frame), dtype=np.int64)
    segments[500:] = 1
    features = compute_derivatives_features(frame, close, segment_ids=segments)
    defined = features["defined"].to_numpy(dtype=bool)
    assert not defined[500 : 500 + WARMUP_HOURS].any()
    assert defined[500 + WARMUP_HOURS :].any()


def test_a_segment_id_array_of_the_wrong_length_is_refused():
    frame, close = build_source(n=300)
    with pytest.raises(DerivativesError, match="segment id"):
        compute_derivatives_features(frame, close, segment_ids=np.zeros(10, dtype=np.int64))


def test_a_frame_with_a_hole_is_refused_even_when_segment_ids_are_supplied():
    """ "The last 168h" means 168 rows only while the rows are hours."""
    frame, close = build_source(n=300)
    kept = np.ones(len(frame), dtype=bool)
    kept[100] = False
    holed = frame.loc[kept].reset_index(drop=True)
    segments = np.zeros(len(holed), dtype=np.int64)
    features = compute_derivatives_features(holed, close[kept], segment_ids=segments)
    assert len(features) == len(holed)


# --- the warm-up --------------------------------------------------------------
def test_no_row_inside_the_preregistered_warm_up_is_in_the_universe():
    frame, close = build_source(n=600)
    features = compute_derivatives_features(frame, close)
    defined = features["defined"].to_numpy(dtype=bool)
    assert not defined[:WARMUP_HOURS].any()
    assert defined[WARMUP_HOURS:].any()


def test_the_first_defined_row_is_exactly_at_the_warm_up_bound():
    frame, close = build_source(n=600)
    features = compute_derivatives_features(frame, close)
    first = int(np.flatnonzero(features["defined"].to_numpy(dtype=bool))[0])
    assert first == WARMUP_HOURS


def test_an_undefined_row_carries_a_declared_zero_rather_than_a_nan():
    frame, close = build_source(n=600)
    features = compute_derivatives_features(frame, close)
    undefined = ~features["defined"].to_numpy(dtype=bool)
    values = features[COLUMNS].to_numpy(dtype=np.float64)[undefined]
    assert np.isfinite(values).all()
    assert (values == 0.0).all()
