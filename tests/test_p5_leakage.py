"""P5's leakage battery, and the positive control for every item in it.

`docs/p5_preregistration.md` §4.4 declares ten things that must be shown before
any P5 cell is scored, and declares that each one carries a positive control:
a test that deliberately introduces the leak and asserts the check catches it.
That second half is the load-bearing one. **A check that has never failed is not
evidence** — it is a line of code that has always been true, and the difference
between those two is exactly the difference between an assertion and a proof.

So each section below is a pair: the property, and the mutation that breaks it.
Where the mutation is cheap the test performs it directly; where it would mean
rebuilding the family it constructs the leaked variant and asserts the same check
rejects it.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from chimera.features import FeatureSpec, compute_features, feature_columns
from nn.information_sets import (
    MTF_V1,
    OHLCV14,
    OHLCV14_PLUS_MTF,
    build_information_set_views,
)
from nn.mtf import (
    INELIGIBLE_FILL,
    MTF_FEATURE_FAMILIES,
    MtfError,
    MtfSpec,
    build_mtf_context,
    epoch_hours,
    higher_timeframe_bars,
    mtf_feature_columns,
)
from nn.p2b import DEFAULT_MANIFEST, load_snapshot, plan_from_manifest
from nn.p5_preregistration import LEAKAGE_BATTERY, WARMUP_BARS
from nn.research_contract import load_contract

SEQ_LEN = 64


@pytest.fixture(scope="module")
def snapshot():
    spine, ds_meta, raw, manifest = load_snapshot(DEFAULT_MANIFEST)
    return spine, ds_meta, raw, manifest


@pytest.fixture(scope="module")
def truncated(snapshot):
    """The raw history as `build_information_set_views` hands it to the family."""
    spine, _, raw, _ = snapshot
    raw = raw.reset_index(drop=True).copy()
    raw["date"] = pd.to_datetime(raw["date"], utc=True)
    dates = pd.to_datetime(spine["date"], utc=True)
    row_of = pd.Series(np.arange(len(raw), dtype=np.int64), index=raw["date"].to_numpy())
    rows = row_of.reindex(dates.to_numpy()).to_numpy(dtype=np.int64)
    return raw.iloc[: int(rows[-1]) + 1].reset_index(drop=True), rows


@pytest.fixture(scope="module")
def context(truncated):
    return build_mtf_context(truncated[0])


def test_the_battery_is_the_one_the_preregistration_declared():
    """Ten items, and this file is what discharges them."""
    assert [item["id"] for item in LEAKAGE_BATTERY] == [f"L{n}" for n in range(1, 11)]


# --- L1 / L2: no bar that has not closed ------------------------------------


@pytest.mark.parametrize("timeframe", ["4h", "1d"])
def test_no_row_reads_a_bar_that_had_not_closed(context, truncated, timeframe):
    """L1, L2. The four- and twenty-four-hour look-ahead this design could have."""
    raw, _ = truncated
    tf = context.per_timeframe[timeframe]
    rows = pd.to_datetime(raw["date"], utc=True).to_numpy()
    closes = tf.bars["close_time"].to_numpy()
    eligible = context.eligible
    assert (closes[tf.as_of[eligible]] <= rows[eligible]).all()
    assert tf.as_of[eligible].min() >= WARMUP_BARS


@pytest.mark.parametrize("timeframe", ["4h", "1d"])
def test_advancing_the_as_of_index_by_one_bar_is_caught(context, truncated, timeframe):
    """L1, L2 positive control. Shifting to the NEXT bar must fail the same check."""
    raw, _ = truncated
    tf = context.per_timeframe[timeframe]
    rows = pd.to_datetime(raw["date"], utc=True).to_numpy()
    closes = tf.bars["close_time"].to_numpy()
    eligible = context.eligible & (tf.as_of < len(tf.bars) - 1)
    leaked = tf.as_of + 1
    assert not (closes[leaked[eligible]] <= rows[eligible]).all(), (
        "reading the next bar would still have passed the causality check, so the "
        "check is not evidence"
    )


# --- L3: the boundary is exact ---------------------------------------------


@pytest.mark.parametrize("timeframe", ["4h", "1d"])
def test_a_row_landing_exactly_on_a_close_sees_that_bar_and_not_the_next(
    context, truncated, timeframe
):
    """L3. `side="right"` is what makes this true; `side="left"` would lose a bar."""
    raw, _ = truncated
    tf = context.per_timeframe[timeframe]
    rows = pd.to_datetime(raw["date"], utc=True).to_numpy()
    closes = tf.bars["close_time"].to_numpy()

    exact = np.flatnonzero(np.isin(rows, closes))
    assert len(exact) > 1000, f"only {len(exact)} rows land on a close; too few to check"
    for row in exact[:: max(1, len(exact) // 200)]:
        selected = tf.as_of[row]
        assert closes[selected] == rows[row], "the bar closing at t must be the one read"
        if selected + 1 < len(closes):
            assert closes[selected + 1] > rows[row]


@pytest.mark.parametrize("timeframe", ["4h", "1d"])
def test_shifting_the_close_convention_by_one_bar_width_moves_the_boundary(
    context, truncated, timeframe
):
    """L3 positive control. If the close time were the bar's OPEN, rows would move."""
    raw, _ = truncated
    tf = context.per_timeframe[timeframe]
    rows = pd.to_datetime(raw["date"], utc=True).to_numpy()
    honest = tf.bars["close_time"].to_numpy()
    leaked = tf.bars["bar_start"].to_numpy()  # the bar becomes visible as it opens
    assert not np.array_equal(
        np.searchsorted(honest, rows, side="right"),
        np.searchsorted(leaked, rows, side="right"),
    )


# --- L4: nothing reaches Styx or P4-HOLD ------------------------------------


def test_no_bar_is_built_from_an_hour_at_or_after_styx(context, truncated):
    """L4. The instant comes from the contract; this file does not restate it."""
    raw, _ = truncated
    sealed = load_contract("btc-usdt-1h-gen1").sealed_test_start
    assert pd.to_datetime(raw["date"], utc=True).max() < sealed
    for tf in context.per_timeframe.values():
        assert tf.bars["close_time"].max() <= pd.Timestamp(sealed)


def test_no_bar_is_built_from_an_hour_inside_p4_hold(snapshot, truncated):
    """L4. P4-HOLD is rows [45802, 48211) and was retired unread."""
    from nn.p4_preregistration import HOLDOUT_ROWS

    spine, _, raw, _ = snapshot
    raw_all = raw.reset_index(drop=True)
    hold_start_row = int(HOLDOUT_ROWS[0])
    assert hold_start_row == len(spine)
    truncated_raw, rows = truncated
    assert len(truncated_raw) == int(rows[-1]) + 1
    first_hold_timestamp = pd.to_datetime(raw_all["date"], utc=True).iloc[int(rows[-1]) + 1]
    assert pd.to_datetime(truncated_raw["date"], utc=True).max() < first_hold_timestamp


def test_extending_the_source_past_the_truncation_point_is_visible(snapshot, truncated):
    """L4 positive control. An untruncated history yields bars past the spine's end."""
    _, _, raw, _ = snapshot
    truncated_raw, _ = truncated
    full = build_mtf_context(raw.reset_index(drop=True))
    honest = build_mtf_context(truncated_raw)
    for timeframe in ("4h", "1d"):
        assert (
            full.per_timeframe[timeframe].bars["close_time"].max()
            > honest.per_timeframe[timeframe].bars["close_time"].max()
        )


# --- L10: causality is structural, not inherited ----------------------------


def test_truncating_the_history_changes_no_value_on_any_spine_row(snapshot, truncated):
    """L10. The strongest of the ten: the family cannot see past the spine's end.

    Computed on the full committed history and on the history truncated at the
    spine's last row, the values on every spine row are identical — so the
    truncation `build_information_set_views` performs is a belt to the engine's
    own braces rather than the thing making it causal.
    """
    _, _, raw, _ = snapshot
    truncated_raw, rows = truncated
    full = build_mtf_context(raw.reset_index(drop=True))
    honest = build_mtf_context(truncated_raw)
    np.testing.assert_array_equal(honest.values[rows], full.values[rows])
    np.testing.assert_array_equal(honest.eligible[rows], full.eligible[rows])


def test_a_centred_window_would_disagree_and_the_check_would_say_so(truncated):
    """L10 positive control. A non-causal engine fails the truncation check."""
    truncated_raw, rows = truncated
    bars = higher_timeframe_bars(truncated_raw, 4)

    def centred(frame: pd.DataFrame) -> np.ndarray:
        # A deliberately non-causal statistic: a centred rolling mean of close.
        return (
            frame["close"].rolling(9, center=True, min_periods=1).mean().to_numpy(np.float64)
        )

    short = higher_timeframe_bars(truncated_raw.iloc[: len(truncated_raw) - 200], 4)
    shared = min(len(bars), len(short))
    assert not np.allclose(centred(bars)[:shared], centred(short)[:shared]), (
        "a centred window agreed across two truncations, so this control proves "
        "nothing about the real check"
    )


# --- L5 / L6 / L7: the labels, the control and the row identity --------------


def test_every_arm_shares_the_spines_labels_and_scores_the_same_rows(snapshot):
    """L5, L7. `prove_alignment` raises on a view that does not, and it passes."""
    spine, ds_meta, raw, manifest = snapshot
    aligned = build_information_set_views(
        spine,
        ds_meta,
        raw,
        names=(OHLCV14, MTF_V1, OHLCV14_PLUS_MTF),
        mtf_spec=MtfSpec(),
    )
    evidence = aligned.prove_alignment(plan_from_manifest(manifest, len(spine))[0], SEQ_LEN)
    assert evidence["information_sets"] == [OHLCV14, MTF_V1, OHLCV14_PLUS_MTF]
    assert evidence["sample_universe"]["applied_to"] == [
        OHLCV14,
        MTF_V1,
        OHLCV14_PLUS_MTF,
    ]
    assert len(evidence["folds"]) == 4
    control = aligned.views[OHLCV14]
    for name in (MTF_V1, OHLCV14_PLUS_MTF):
        view = aligned.views[name]
        assert view.targets is control.targets
        assert view.future_return is control.future_return
        assert view.dates is control.dates
        assert view.segment_ids is control.segment_ids
        assert view.target_spec == control.target_spec


def test_the_control_arm_is_the_spines_own_columns(snapshot):
    """L6. A control that drifted would make every delta a comparison of two changes."""
    spine, ds_meta, raw, _ = snapshot
    aligned = build_information_set_views(
        spine, ds_meta, raw, names=(OHLCV14, MTF_V1), mtf_spec=MtfSpec()
    )
    control = aligned.views[OHLCV14]
    assert control.feature_names == feature_columns()
    for position, name in enumerate(feature_columns()):
        np.testing.assert_array_equal(
            control.features[:, position], spine[name].to_numpy(dtype=np.float64)
        )


def test_perturbing_one_control_column_is_caught(snapshot):
    """L6 positive control. `prove_alignment` re-derives every column it holds."""
    spine, ds_meta, raw, manifest = snapshot
    aligned = build_information_set_views(
        spine, ds_meta, raw, names=(OHLCV14, MTF_V1), mtf_spec=MtfSpec()
    )
    aligned.views[OHLCV14].features[10, 0] += 1.0
    with pytest.raises(AssertionError, match="does not match the column"):
        aligned.prove_alignment(plan_from_manifest(manifest, len(spine))[0], SEQ_LEN)


# --- L8: nothing is forward-filled from a future close ----------------------


@pytest.mark.parametrize("timeframe", ["4h", "1d"])
def test_the_as_of_index_only_ever_moves_forward(context, timeframe):
    """L8. A non-monotone index is a row reading a bar an earlier row could not."""
    as_of = context.per_timeframe[timeframe].as_of
    assert (np.diff(as_of) >= 0).all()


@pytest.mark.parametrize("timeframe", ["4h", "1d"])
def test_the_shift_control_is_evaluated_where_a_shift_is_detectable(context, timeframe):
    """L8. The whole-column control is degenerate here; this records the coverage."""
    evidence = context.evidence["per_timeframe"][timeframe]
    assert evidence["matches"] is True
    assert evidence["matches_under_plus_one_bar_shift"] is False
    assert evidence["matches_under_minus_one_bar_shift"] is False
    assert evidence["boundary_rows_checked"] > 1000
    assert evidence["rows_reading_a_bar_that_had_not_closed"] == 0


def test_a_whole_column_shift_would_have_passed_the_naive_control(context):
    """L8, and why the boundary control exists at all.

    A higher-timeframe column is piecewise constant inside a bar, so rolling it by
    one *row* agrees on the overwhelming majority of rows. This asserts that
    directly, because it is the reason the shift control had to be moved to the
    boundary rows rather than left where the other families keep it.
    """
    column = context.column("mtf_1d_atr_norm")
    rolled = np.roll(column, 1)
    agreement = np.mean(np.isclose(column[1:], rolled[1:], rtol=0.0, atol=1e-12))
    assert agreement > 0.95, (
        f"a one-row shift agreed on only {agreement:.3f} of rows, so the naive control "
        "might have been adequate after all and this reasoning needs revisiting"
    )


# --- L9: a deliberately future-shifted family is detected -------------------


@pytest.mark.parametrize("timeframe", ["4h", "1d"])
def test_a_family_built_from_the_next_bar_fails_the_causal_checks(
    context, truncated, timeframe
):
    """L9. The leak this whole design exists to prevent, constructed and caught."""
    raw, _ = truncated
    tf = context.per_timeframe[timeframe]
    rows = pd.to_datetime(raw["date"], utc=True).to_numpy()
    closes = tf.bars["close_time"].to_numpy()
    leaked = np.minimum(tf.as_of + 1, len(closes) - 1)
    eligible = context.eligible

    reading_the_future = int(np.count_nonzero(closes[leaked[eligible]] > rows[eligible]))
    assert reading_the_future > 0, "the leaked variant reads nothing it should not"

    honest = int(np.count_nonzero(closes[tf.as_of[eligible]] > rows[eligible]))
    assert honest == 0


def test_a_bar_including_the_current_hour_is_a_different_family(truncated):
    """L9. Reading the bar in progress is the other shape of the same leak."""
    raw, _ = truncated
    bars = higher_timeframe_bars(raw, 4)
    dates = pd.to_datetime(raw["date"], utc=True)
    buckets = epoch_hours(dates) // 4
    in_progress = np.searchsorted(
        (bars["bar_start"].pipe(epoch_hours) // 4).to_numpy(), buckets.to_numpy()
    )
    honest = np.searchsorted(bars["close_time"].to_numpy(), dates.to_numpy(), side="right") - 1
    # The in-progress index is the honest one plus one, wherever the bar exists.
    differing = np.count_nonzero(in_progress != honest)
    assert differing > len(raw) * 0.5, (
        "reading the in-progress bar would have selected the same bar as reading the "
        "last closed one, which cannot be right"
    )


# --- the family is what it says it is --------------------------------------


def test_the_family_is_the_ohlcv14_engine_on_a_wider_bar(truncated):
    """No second implementation: the engine is the one the control is built from."""
    raw, _ = truncated
    spec = MtfSpec()
    context = build_mtf_context(raw, spec)
    for timeframe, prefix in (("4h", "mtf_4h_"), ("1d", "mtf_1d_")):
        tf = context.per_timeframe[timeframe]
        direct = compute_features(
            tf.bars[["open", "high", "low", "close", "volume"]], spec.feature_spec
        )
        eligible = context.eligible
        for name in feature_columns():
            np.testing.assert_allclose(
                context.column(f"{prefix}{name}")[eligible],
                direct[name].to_numpy(dtype=np.float64)[tf.as_of[eligible]],
                rtol=0.0,
                atol=0.0,
            )


def test_a_partial_bar_is_dropped_rather_than_completed():
    """The rule that makes 'fully closed' mean something on a punctured feed."""
    hours = pd.date_range("2021-01-01", periods=12, freq="1h", tz="UTC")
    frame = pd.DataFrame(
        {
            "date": hours,
            "open": np.arange(12, dtype=float) + 1,
            "high": np.arange(12, dtype=float) + 2,
            "low": np.arange(12, dtype=float),
            "close": np.arange(12, dtype=float) + 1,
            "volume": np.ones(12),
        }
    )
    assert len(higher_timeframe_bars(frame, 4)) == 3

    punctured = frame.drop(index=5).reset_index(drop=True)  # 05:00 removed
    bars = higher_timeframe_bars(punctured, 4)
    assert len(bars) == 2
    assert pd.Timestamp("2021-01-01 04:00", tz="UTC") not in set(bars["bar_start"])


def test_a_stale_context_makes_a_row_ineligible(context):
    """Never served eight-hour-old 4h context while a neighbour gets fresh context."""
    for timeframe, tf in context.per_timeframe.items():
        fresh = tf.staleness_hours[tf.eligible]
        assert np.nanmax(fresh) < tf.hours, timeframe


def test_an_ineligible_row_holds_the_declared_fill_and_is_never_scored(snapshot):
    """The fill is arbitrary because the universe makes it unreachable."""
    spine, ds_meta, raw, manifest = snapshot
    aligned = build_information_set_views(
        spine, ds_meta, raw, names=(OHLCV14, MTF_V1), mtf_spec=MtfSpec()
    )
    eligible = aligned.eligible
    assert eligible is not None
    ineligible = np.flatnonzero(~eligible)
    assert len(ineligible) == 1631
    assert ineligible.min() == 0 and ineligible.max() == 1630
    view = aligned.views[MTF_V1]
    assert (view.features[ineligible] == INELIGIBLE_FILL).all()

    # Every block of every fold starts past the last ineligible row. Written as
    # one unconditional assertion over all three blocks: the earlier form carried
    # an `or block != "train"` that made it vacuously true for the inner and
    # outer blocks, which are the two that actually decide anything.
    folds, _ = plan_from_manifest(manifest, len(spine))
    evidence = aligned.prove_alignment(folds, SEQ_LEN)
    assert len(evidence["folds"]) == 4
    for fold in evidence["folds"]:
        for block in ("train", "inner_validation", "outer_validation"):
            assert fold[block]["first_row"] > ineligible.max(), (fold["fold"], block)


def test_the_fill_value_cannot_reach_a_fitted_model(snapshot):
    """The claim the fill rests on, checked rather than asserted.

    Every sample index in every block of every fold must be an eligible row, and
    so must every row of its input window — which is what `sample_indices` uses
    the universe for.
    """
    from nn.dataset import sample_indices

    spine, ds_meta, raw, manifest = snapshot
    aligned = build_information_set_views(
        spine, ds_meta, raw, names=(OHLCV14, MTF_V1), mtf_spec=MtfSpec()
    )
    folds, _ = plan_from_manifest(manifest, len(spine))
    horizon = ds_meta.target_spec["horizon"]
    view = aligned.views[MTF_V1]
    for plan in folds:
        for split in (plan.train, plan.inner, plan.outer):
            idx = sample_indices(
                split,
                SEQ_LEN,
                horizon,
                segment_ids=view.segment_ids,
                eligible=aligned.eligible,
            )
            assert len(idx) > 0
            assert aligned.eligible[idx].all()
            windows = idx[:, None] + np.arange(-SEQ_LEN + 1, 1)[None, :]
            assert aligned.eligible[windows].all()


def test_a_run_that_names_a_p5_arm_without_the_spec_is_refused(snapshot):
    """Fail closed: an empty higher-timeframe matrix is not an acceptable answer."""
    spine, ds_meta, raw, _ = snapshot
    with pytest.raises(ValueError, match="no mtf spec"):
        build_information_set_views(spine, ds_meta, raw, names=(OHLCV14, MTF_V1))


def test_two_sample_universes_at_once_are_refused(snapshot):
    """The arms would be scored on an intersection nobody preregistered."""

    class _FakeUniverse:
        eligible = None

    spine, ds_meta, raw, _ = snapshot
    fake = _FakeUniverse()
    fake.eligible = np.ones(len(spine), dtype=bool)
    fake.feature_names = ()
    with pytest.raises(ValueError):
        build_information_set_views(
            spine,
            ds_meta,
            raw,
            names=(OHLCV14, MTF_V1),
            mtf_spec=MtfSpec(),
            derivatives=fake,
        )


def test_the_families_partition_the_columns():
    covered = [c for group in MTF_FEATURE_FAMILIES.values() for c in group]
    assert sorted(covered) == sorted(mtf_feature_columns())
    assert set(MTF_FEATURE_FAMILIES) == {"mtf_4h", "mtf_1d"}


def test_the_spec_hash_moves_when_a_constant_moves():
    base = MtfSpec()
    assert base.spec_hash() == MtfSpec().spec_hash()
    assert replace(base, warmup_bars=77).spec_hash() != base.spec_hash()
    assert (
        replace(base, feature_spec=FeatureSpec(rsi_period=15)).spec_hash() != base.spec_hash()
    )


def test_an_unknown_timeframe_is_refused():
    with pytest.raises(MtfError, match="not preregistered"):
        MtfSpec(timeframes=("2h",))
