"""P4's commitments, pinned so that spending one is a diff rather than an edit.

A preregistration that lives only in prose can be revised by the commit that
reports the result. These tests hold the document and the module to each other,
hold both to the properties the design claims — one deciding comparison, a
holdout that cannot reach a sealed label, an outcome table with no escape
hatch — and assert that nothing about P4 has been *started*.

The point is not that any single number here is provably right. It is that
moving one has to be visible.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nn.information_sets import CHECKPOINTS
from nn.p4_preregistration import (
    ARMS,
    COMBINED,
    CONTROL,
    COST_SENSITIVITY_MULTIPLIERS,
    DERIVATIVES_V1,
    DATA_SOURCES,
    DEGREES_OF_FREEDOM,
    EXPLORATORY_OUTER_BLOCKS,
    FEATURES,
    FEATURE_NAMES,
    HOLDOUT_ROWS,
    MIN_OUTER_TRADES,
    NULL_PROBABILITY_THREE_OF_FOUR,
    PRIMARY_COMPARISON,
    PRIMARY_MODEL,
    RESEARCH_ROWS,
    SECONDARY_MODELS,
    STAGE_1_CONTINUATION,
    STAGE_2_OUTCOMES,
    TARGET,
    WARMUP_HOURS,
    preregistration,
    preregistration_hash,
)
from nn.walkforward import plan_nested_folds

ROOT = Path(__file__).resolve().parent.parent
DOCUMENT = ROOT / "docs" / "p4_preregistration.md"


# --- P4 is preregistered, and nothing about it has been started --------------
def test_p4_is_not_a_runnable_checkpoint():
    """`--checkpoint P4` must be refused, not answered with empty columns.

    A registered checkpoint with no engine behind it is the shape of a run that
    starts before its design is finished.
    """
    assert "P4" not in CHECKPOINTS


def test_no_p4_evidence_exists():
    assert not (ROOT / "artifacts" / "benchmark" / "btc_p4_comparison").exists()
    assert not list((ROOT / "artifacts" / "benchmark").glob("btc_p4_*"))


def test_no_p4_data_has_been_acquired():
    research = ROOT / "data" / "research"
    assert not list(research.glob("*funding*"))
    assert not list(research.glob("*open_interest*"))
    assert not list(research.glob("*p4*"))


def test_no_derivatives_engine_exists_yet():
    """The features are specified. They are not implemented, and must not be."""
    assert not (ROOT / "nn" / "derivatives.py").exists()
    for name in FEATURE_NAMES:
        assert not (ROOT / "nn" / "information_sets.py").read_text().count(name)


# --- the document and the module say the same thing --------------------------
def test_the_document_exists_and_records_the_hash():
    text = DOCUMENT.read_text()
    assert f"sha256:{preregistration_hash()}" in text


def test_every_feature_appears_in_the_document_with_its_window_and_clip():
    text = DOCUMENT.read_text()
    for feature in FEATURES:
        assert f"`{feature['name']}`" in text, feature["name"]
        low, high = feature["clip"]
        assert f"[{low:g}, {high:g}]" in text, feature["name"]
        if feature["window"] is not None:
            assert str(feature["window"]) in text


def test_the_document_states_the_decision_numbers():
    text = DOCUMENT.read_text()
    assert str(MIN_OUTER_TRADES) in text
    assert "5/16" in text
    assert f"{HOLDOUT_ROWS[0]}, {HOLDOUT_ROWS[1]})" in text
    assert PRIMARY_MODEL in text


def test_every_data_source_names_a_venue_a_column_and_a_publication_rule():
    for source in DATA_SOURCES:
        assert source["venue"]
        assert source["timestamp_column"]
        assert source["timestamp_semantics"]
        assert source["publication"]
        assert source["primary"]


def test_the_preregistration_serialises_and_hashes_stably():
    first = preregistration_hash()
    json.dumps(preregistration())  # must be JSON-serialisable at all
    assert first == preregistration_hash()
    assert len(first) == 64


# --- the design's own claims -------------------------------------------------
def test_three_arms_and_one_deciding_comparison():
    assert ARMS == (CONTROL, DERIVATIVES_V1, COMBINED)
    assert PRIMARY_COMPARISON == (COMBINED, CONTROL)
    assert PRIMARY_MODEL not in SECONDARY_MODELS
    assert len(SECONDARY_MODELS) == 2


def test_the_feature_family_is_small_and_every_column_is_bounded():
    assert len(FEATURES) == 8
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)
    for feature in FEATURES:
        low, high = feature["clip"]
        assert low < high
        assert feature["definition"]
        assert feature["visible_from"]
        assert feature["family"] in {"funding", "open_interest", "basis"}


def test_the_warmup_covers_every_declared_window():
    """A window longer than the warm-up would silently shorten the universe."""
    hours = []
    for feature in FEATURES:
        if feature["window"] is None:
            continue
        # Funding windows are counted in 8-hourly settlements, the rest in hours.
        hours.append(feature["window"] * (8 if feature["family"] == "funding" else 1))
    assert WARMUP_HOURS == max(hours)


def test_the_target_is_the_one_every_earlier_checkpoint_used():
    assert TARGET["horizon"] == 6
    assert TARGET["timeframe"] == "1h"
    assert TARGET["cost_threshold"] == 2 * (TARGET["fee_rate"] + TARGET["slippage_rate"])
    assert TARGET["cost_threshold"] == 0.002


def test_the_cost_sensitivity_starts_at_the_unchanged_model():
    assert COST_SENSITIVITY_MULTIPLIERS[0] == 1.0
    assert all(m >= 1.0 for m in COST_SENSITIVITY_MULTIPLIERS)


def test_the_holdout_label_cannot_reach_a_sealed_close():
    """`48211 + 6 == 48217`: the arithmetic the region's end comes from."""
    start, end = HOLDOUT_ROWS
    assert end + TARGET["horizon"] == RESEARCH_ROWS
    assert start == EXPLORATORY_OUTER_BLOCKS[-1][1]


def test_the_holdout_does_not_overlap_any_block_that_has_been_read():
    start, _ = HOLDOUT_ROWS
    for block_start, block_end in EXPLORATORY_OUTER_BLOCKS:
        assert block_end <= start


def test_the_exploratory_blocks_are_the_geometry_every_earlier_checkpoint_used():
    """Not restated by hand: re-planned from the committed fold parameters."""
    folds = plan_nested_folds(
        RESEARCH_ROWS,
        4,
        int(RESEARCH_ROWS * 0.45),
        int(RESEARCH_ROWS * 0.10),
        int(RESEARCH_ROWS * 0.10),
        int(RESEARCH_ROWS * 0.10),
    )
    assert tuple((f.outer.start, f.outer.end) for f in folds) == EXPLORATORY_OUTER_BLOCKS


def test_the_committed_snapshot_cannot_reach_the_holdout():
    """The structural guarantee, checked against the committed manifest."""
    manifest = json.loads(
        (ROOT / "data" / "research" / "btc_usdt_1h_gen1_snapshot_manifest.json").read_text()
    )
    assert manifest["processed_outer_coverage"]["rows"] == HOLDOUT_ROWS[0]


def test_the_screen_records_the_null_it_is_weak_against():
    assert NULL_PROBABILITY_THREE_OF_FOUR == 5 / 16
    assert STAGE_1_CONTINUATION["screen_false_positive_rate_under_coin_null"] == 5 / 16


def test_stage_one_needs_more_than_a_fold_count():
    """Three conditions, not one: P2b's positive-mean-one-fold case and its mirror."""
    assert STAGE_1_CONTINUATION["improved_folds_required"] == 3
    assert STAGE_1_CONTINUATION["valid_folds_required"] == 3
    assert STAGE_1_CONTINUATION["mean_delta_above"] == 0.0
    assert STAGE_1_CONTINUATION["worst_fold_delta_at_least"] < 0


def test_the_trade_floor_is_the_repositorys_own_enough_trades_number():
    """Refusing to *select* on four trades and *reporting* four is incoherent."""
    from nn.p2b import MIN_TRADES

    assert MIN_OUTER_TRADES == MIN_TRADES


def test_the_trade_floor_bites_once_and_leaves_the_screen_satisfiable():
    """Both halves matter, and they pull against each other.

    A floor that never fires is decoration; a floor that invalidates three of
    four folds of the control makes the checkpoint unanswerable by construction.
    Ten fires on P3's four-trade fold and leaves exactly the three stage 1 needs.
    """
    payload = json.loads(
        (ROOT / "artifacts" / "benchmark" / "btc_p3_ohlcv14_xgboost" / "p2b.json").read_text()
    )
    trades = [
        record["outer_validation"]["xgboost"]["trading"]["n_trades"]
        for record in payload["folds"]
    ]
    combined = json.loads(
        (
            ROOT
            / "artifacts"
            / "benchmark"
            / "btc_p3_ohlcv14_plus_microstructure_v1_xgboost"
            / "p2b.json"
        ).read_text()
    )
    other = [
        record["outer_validation"]["xgboost"]["trading"]["n_trades"]
        for record in combined["folds"]
    ]
    assert sorted(trades) == [4, 11, 16, 80]
    assert sorted(other) == [12, 21, 28, 118]

    # A fold is valid for the comparison only if *both* arms clear the floor.
    valid = [min(a, b) >= MIN_OUTER_TRADES for a, b in zip(trades, other)]
    assert valid.count(True) == 3
    assert valid.count(False) == 1
    # And the rejected alternative would have left too few to screen on.
    assert [min(a, b) >= 20 for a, b in zip(trades, other)].count(True) == 1


# --- there is no escape hatch ------------------------------------------------
def test_every_outcome_ends_the_checkpoint():
    labels = [outcome["label"] for outcome in STAGE_2_OUTCOMES]
    assert labels == ["negative", "inconclusive", "supported"]
    for outcome in STAGE_2_OUTCOMES:
        assert outcome["when"] and outcome["classification"] and outcome["then"]


def test_no_outcome_licenses_tuning_and_retrying():
    joined = " ".join(outcome["then"] for outcome in STAGE_2_OUTCOMES).lower()
    for escape in ("tune", "retry", "try again", "adjust the window", "re-search"):
        assert escape not in joined


def test_the_best_available_label_is_not_confirmatory():
    supported = next(o for o in STAGE_2_OUTCOMES if o["label"] == "supported")
    assert "NOT confirmatory" in supported["classification"]
    assert "new research generation" in supported["then"]


def test_styx_is_not_a_tiebreaker_anywhere_in_the_design():
    text = DOCUMENT.read_text()
    assert "not** opened" in text or "**not** opened" in text
    assert preregistration()["styx"].startswith("not opened")
    for outcome in STAGE_2_OUTCOMES:
        assert "styx" not in outcome["when"].lower()


def test_the_degrees_of_freedom_are_inventoried_and_each_one_is_closed():
    assert len(DEGREES_OF_FREEDOM) >= 12
    for entry in DEGREES_OF_FREEDOM:
        assert entry["choice"] and entry["constrained_by"]


@pytest.mark.parametrize(
    "forbidden",
    [
        "feature-window shopping",
        "model shopping",
        "horizon shopping",
        "cost shopping",
        "arm deletion",
        "regime deletion",
    ],
)
def test_the_document_forbids_each_named_post_hoc_move(forbidden):
    assert forbidden in DOCUMENT.read_text()
