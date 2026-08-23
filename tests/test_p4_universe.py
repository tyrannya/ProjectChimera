"""P4's common sample universe and the availability gate, on the committed spine.

These run against the *real* research spine, because everything §6.2 and §8.0
decide is a property of the committed geometry: which rows the four burned blocks
cover, where the spine's segment boundaries fall, and how a missing archive day
lands on them. The derivatives source is synthetic — see ``tests/conftest.py``
for why it has to be — but the rows it is joined to are the rows P4 will run on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nn.derivatives import DerivativesSpec
from nn.derivatives_sources import UNAVAILABLE_AGE_NS
from nn.p4_preregistration import (
    AVAILABILITY_GATE,
    BLOCK_AVAILABILITY_RULE,
    EXPLORATORY_OUTER_BLOCKS,
    HOLDOUT_ROWS,
    WARMUP_HOURS,
    preregistration_hash,
)
from nn.p4_universe import (
    UniverseError,
    availability_gate,
    block_availability,
    build_universe,
    evaluate_blocks,
    holdout_coverage_from_archive_days,
    universe_identity,
)

MIN_FRACTION = float(BLOCK_AVAILABILITY_RULE["min_surviving_row_fraction"])
MAX_OUTAGE = int(BLOCK_AVAILABILITY_RULE["max_contiguous_missing_hours"])


# --- the universe is one object, applied to every arm ------------------------
def test_the_universe_covers_every_spine_row_exactly_once(p4_universe, p4_spine):
    assert p4_universe.rows_total == len(p4_spine["spine"])
    assert len(p4_universe.eligible) == len(p4_spine["spine"])
    assert p4_universe.features.shape == (len(p4_spine["spine"]), 8)


def test_no_row_before_the_open_interest_archive_begins_is_in_the_universe(p4_universe):
    """§3.6: a later first day narrows the universe rather than being worked around.

    The binding condition at the start of the open-interest history is not the
    240-hour warm-up — the spine's first segment began nine months earlier, so
    funding and basis are long since warm — but ``drv_oi_log_change_24h``, which
    reads the snapshot at ``t`` *and* at ``t − 24h``. The first eligible row is
    therefore exactly one day after the archive's first day.
    """
    archive_start = pd.Timestamp("2020-09-01T00:00:00Z")
    first = pd.Timestamp(p4_universe.first_eligible_instant)
    assert first == archive_start + pd.Timedelta(hours=24)


def test_the_universe_stops_at_the_spines_last_row(p4_universe, p4_spine):
    dates = pd.to_datetime(p4_spine["spine"]["date"], utc=True)
    assert p4_universe.last_eligible_row is not None
    assert p4_universe.last_eligible_row < HOLDOUT_ROWS[0]
    assert pd.Timestamp(p4_universe.last_eligible_instant) <= dates.iloc[-1]


def test_the_universe_hash_moves_when_a_row_leaves_it(p4_universe):
    from nn.p4_universe import _mask_hash

    other = p4_universe.eligible.copy()
    other[p4_universe.last_eligible_row] = False
    assert _mask_hash(other) != p4_universe.universe_hash


def test_the_universe_identity_binds_the_preregistration_and_the_gate(p4_universe):
    identity = universe_identity(p4_universe, {"gate_passed": True})
    assert identity["preregistration_hash"] == preregistration_hash()
    assert identity["universe_hash"] == p4_universe.universe_hash
    assert identity["derivatives_spec_hash"] == DerivativesSpec().spec_hash()
    other = universe_identity(p4_universe, {"gate_passed": False})
    assert other["identity"] != identity["identity"]


def test_the_join_evidence_would_not_survive_a_one_row_shift(p4_universe):
    evidence = p4_universe.join_evidence
    assert evidence["matches"] is True
    assert evidence["matches_under_plus_one_shift"] is False
    assert evidence["matches_under_minus_one_shift"] is False


def test_the_universe_records_which_condition_removed_what(p4_universe):
    excluded = p4_universe.excluded_by
    assert set(excluded["per_column"]) == set(p4_universe.feature_names)
    assert (
        excluded["derivatives_undefined"] == p4_universe.rows_total - p4_universe.rows_eligible
    )


# --- §8.0, on the committed geometry ------------------------------------------
def test_the_four_preregistered_blocks_are_the_ones_evaluated(p4_universe):
    labels = [entry["block"] for entry in evaluate_blocks(p4_universe)]
    assert labels == [list(block) for block in EXPLORATORY_OUTER_BLOCKS]


def test_a_spine_segment_boundary_inside_a_block_makes_it_unavailable(p4_universe):
    """A finding about the committed geometry, not about any data.

    The dataset builder drops an indicator warm-up after every market-data gap,
    so the spine's segments restart at row 26998 — inside the first outer block.
    ``derivatives_v1``'s binding window is 30 funding settlements and §6.2's
    fourth condition forbids a window bridging a segment boundary, so the 240
    rows after that restart leave the universe. 240 hours is a contiguous outage
    and §8.0 permits at most 48, so block 0 is unavailable *before any archive is
    fetched*. Blocks 1-3 contain no segment boundary and are clean.
    """
    blocks = evaluate_blocks(p4_universe)
    assert blocks[0]["available"] is False
    assert blocks[0]["max_contiguous_missing_hours"] >= WARMUP_HOURS
    assert [entry["available"] for entry in blocks[1:]] == [True, True, True]


def test_the_gate_needs_the_holdout_as_well_as_the_blocks(p4_universe):
    blocks = evaluate_blocks(p4_universe)
    without = availability_gate(blocks, None)
    assert without["gate_passed"] is False
    assert any("P4-HOLD" in reason for reason in without["reasons"])
    with_holdout = availability_gate(blocks, {"available": True})
    assert with_holdout["gate_passed"] is True


def test_a_gate_failure_is_not_evaluable_and_is_not_negative(p4_universe):
    """§9.0: nothing was measured, so nothing may be reported as evidence."""
    failed = availability_gate(evaluate_blocks(p4_universe), {"available": False})
    assert failed["gate_passed"] is False
    assert failed["outcome"]["label"] == "not_evaluable"
    assert failed["outcome"]["reason_code"] == "insufficient_coverage"
    assert failed["outcome"]["classification"] == "not a research result"
    assert "not a negative" in failed["note"] or "not evidence" in failed["note"]


def test_a_passing_gate_records_no_not_evaluable_outcome(p4_universe):
    passed = availability_gate(evaluate_blocks(p4_universe), {"available": True})
    assert passed["outcome"] is None


def test_fewer_than_two_available_blocks_fails_the_gate():
    blocks = [
        {"block": list(b), "available": index == 0, "reasons": []}
        for index, b in enumerate(EXPLORATORY_OUTER_BLOCKS)
    ]
    gate = availability_gate(blocks, {"available": True})
    assert gate["gate_passed"] is False
    assert (
        str(AVAILABILITY_GATE["requires_exploratory_blocks_available"]) in gate["reasons"][0]
    )


# --- the block rule, exercised directly --------------------------------------
def _mask(rows, excluded=()):
    mask = np.ones(rows, dtype=bool)
    for start, end in excluded:
        mask[start:end] = False
    return mask


def _scatter(rows, lost):
    """``lost`` single excluded rows, spread so no two of them are adjacent."""
    step = max(rows // (lost + 1), 2)
    return _mask(rows, [(i * step, i * step + 1) for i in range(lost)])


def test_a_block_just_above_the_coverage_floor_is_available():
    rows = 4821
    lost = int(rows * (1 - MIN_FRACTION)) - 1
    entry = block_availability(_scatter(rows, lost), (0, rows), label="synthetic")
    assert entry["available"] is True
    assert entry["surviving_fraction"] >= MIN_FRACTION


def test_a_block_just_below_the_coverage_floor_is_unavailable():
    rows = 4821
    lost = int(rows * (1 - MIN_FRACTION)) + 2
    entry = block_availability(_scatter(rows, lost), (0, rows), label="synthetic")
    assert entry["available"] is False
    assert "rows survive" in entry["reasons"][0]


def test_an_outage_of_exactly_forty_eight_hours_is_permitted():
    mask = _mask(4821, [(1000, 1000 + MAX_OUTAGE)])
    entry = block_availability(mask, (0, 4821), label="synthetic")
    assert entry["max_contiguous_missing_hours"] == MAX_OUTAGE
    assert entry["available"] is True


def test_an_outage_of_forty_nine_hours_is_not():
    """A longer unbroken outage removes a market episode rather than a sample."""
    mask = _mask(4821, [(1000, 1000 + MAX_OUTAGE + 1)])
    entry = block_availability(mask, (0, 4821), label="synthetic")
    assert entry["max_contiguous_missing_hours"] == MAX_OUTAGE + 1
    assert entry["available"] is False
    assert "contiguous" in entry["reasons"][0]


def test_a_block_the_mask_does_not_fully_cover_is_refused_rather_than_discounted():
    with pytest.raises(UniverseError, match="cannot be judged available at a discount"):
        block_availability(np.ones(100, dtype=bool), (0, 4821), label="synthetic")


# --- a missing archive day, on the real geometry -----------------------------
def test_one_missing_metrics_day_removes_forty_eight_rows_from_the_universe(
    p4_spine, p4_hourly
):
    """§3.0a's arithmetic, plus the echo `drv_oi_log_change_24h` adds to it."""
    from tests.conftest import synthetic_derivatives_hourly

    baseline = build_universe(p4_spine["spine"], p4_hourly, p4_spine["raw"])
    punctured = synthetic_derivatives_hourly(
        pd.to_datetime(p4_hourly["date"], utc=True).iloc[0],
        pd.to_datetime(p4_hourly["date"], utc=True).iloc[-1] + pd.Timedelta(hours=1),
        p4_spine["raw"],
        missing_days=["2024-06-05"],
    )
    holed = build_universe(p4_spine["spine"], punctured, p4_spine["raw"])
    lost = baseline.rows_eligible - holed.rows_eligible
    assert lost == 48


def test_a_three_day_outage_makes_its_block_unavailable(p4_spine, p4_hourly):
    from tests.conftest import synthetic_derivatives_hourly

    dates = pd.to_datetime(p4_hourly["date"], utc=True)
    punctured = synthetic_derivatives_hourly(
        dates.iloc[0],
        dates.iloc[-1] + pd.Timedelta(hours=1),
        p4_spine["raw"],
        missing_days=["2024-06-05", "2024-06-06", "2024-06-07"],
    )
    universe = build_universe(p4_spine["spine"], punctured, p4_spine["raw"])
    blocks = evaluate_blocks(universe)
    hit = next(b for b in blocks if b["block"] == [36160, 40981])
    assert hit["max_contiguous_missing_hours"] > MAX_OUTAGE
    assert hit["available"] is False


def test_removing_open_interest_entirely_empties_the_universe(p4_spine, p4_hourly):
    blanked = p4_hourly.copy()
    blanked["oi_available"] = 0
    blanked["oi_age_ns"] = UNAVAILABLE_AGE_NS
    blanked[["oi_contracts", "oi_notional"]] = 0.0
    universe = build_universe(p4_spine["spine"], blanked, p4_spine["raw"])
    assert universe.rows_eligible == 0
    gate = availability_gate(evaluate_blocks(universe), {"available": True})
    assert gate["gate_passed"] is False
    assert gate["outcome"]["reason_code"] == "insufficient_coverage"


# --- P4-HOLD's own coverage, from metadata alone ------------------------------
def test_the_holdout_rule_is_the_same_rule_and_reads_no_row():
    days = pd.date_range("2025-05-19", "2025-08-26", freq="D")
    published = {day.date().isoformat(): True for day in days}
    entry = holdout_coverage_from_archive_days(published)
    assert entry["available"] is True
    assert entry["block"] == list(HOLDOUT_ROWS)
    assert entry["basis"] == "archive-day metadata only; no P4-HOLD row was read"
    assert entry["rule"] == dict(BLOCK_AVAILABILITY_RULE)


def test_a_three_day_hole_in_the_holdouts_archive_makes_it_unavailable():
    days = pd.date_range("2025-05-19", "2025-08-26", freq="D")
    published = {day.date().isoformat(): True for day in days}
    for absent in ("2025-06-01", "2025-06-02", "2025-06-03"):
        published[absent] = False
    entry = holdout_coverage_from_archive_days(published)
    assert entry["available"] is False
    assert entry["max_contiguous_missing_hours"] > MAX_OUTAGE


def test_unknown_holdout_coverage_is_not_available_coverage():
    with pytest.raises(UniverseError, match="not an available one"):
        holdout_coverage_from_archive_days({})


# --- fail-closed joins --------------------------------------------------------
def test_a_derivatives_source_that_does_not_cover_the_spine_is_refused(p4_spine, p4_hourly):
    truncated = p4_hourly.iloc[:-500].reset_index(drop=True)
    with pytest.raises(UniverseError):
        build_universe(p4_spine["spine"], truncated, p4_spine["raw"])


def test_a_spine_without_segment_ids_is_refused(p4_spine, p4_hourly):
    spine = p4_spine["spine"].drop(columns=["segment_id"])
    with pytest.raises(UniverseError, match="segment_id"):
        build_universe(spine, p4_hourly, p4_spine["raw"])


def test_a_source_with_duplicate_hours_is_refused(p4_spine, p4_hourly):
    doubled = pd.concat([p4_hourly, p4_hourly.iloc[-1:]], ignore_index=True)
    with pytest.raises(UniverseError, match="strictly increasing"):
        build_universe(p4_spine["spine"], doubled, p4_spine["raw"])
