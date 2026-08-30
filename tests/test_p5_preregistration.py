"""P5's commitments, pinned so that changing one is a diff rather than an edit.

A preregistration that lives only in prose can be revised by the commit that
reports the result. These tests hold ``docs/p5_preregistration.md`` and
``nn.p5_preregistration`` to each other, hold both to the properties the design
claims — one axis changing, one deciding comparison, a bar that cannot be
rescued by a mean — and assert that nothing about P5 has been *started*.

The point is not that any single number here is provably right. It is that moving
one has to be visible.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chimera.contracts import TargetSpec
from chimera.features import FeatureSpec, feature_columns
from nn.p5_preregistration import (
    ALIGNMENT,
    ARMS,
    AVAILABILITY_GATE,
    BLOCK_AVAILABILITY_RULE,
    CHECKPOINT,
    COMBINED,
    CONTIGUITY_POLICY,
    CONTROL,
    DECISION_RULE,
    ELIGIBILITY_CONDITIONS,
    EVIDENCE_CEILING,
    FAMILY,
    FEATURE_ENGINE,
    FORBIDDEN_AFTER_RESULTS,
    HELD_FIXED,
    IMPROVED_RULE,
    LEAKAGE_BATTERY,
    MEASURED_AVAILABILITY,
    MODELS,
    P4_HOLD_UNAVAILABILITY,
    PRIMARY_COMPARISON,
    PRIMARY_MODEL,
    SECONDARY_MODELS,
    STOPPING_RULE,
    STYX_PROHIBITION,
    TIMEFRAMES,
    TRADE_COUNT_DIAGNOSTIC,
    WARMUP_BARS,
    mtf_columns,
    payload,
    preregistration_hash,
)

ROOT = Path(__file__).resolve().parent.parent
DOCUMENT = ROOT / "docs" / "p5_preregistration.md"


@pytest.fixture(scope="module")
def document() -> str:
    return DOCUMENT.read_text()


# --- the document and the module say the same thing -------------------------


def test_the_document_exists_and_records_the_hash(document):
    """A document naming a hash nobody checks will eventually name the wrong one."""
    assert preregistration_hash() in document


def test_the_hash_is_stable_under_reserialisation():
    assert preregistration_hash() == preregistration_hash()
    assert preregistration_hash().startswith("sha256:")


@pytest.mark.parametrize(
    "key",
    [
        "arms",
        "primary_model",
        "primary_comparison",
        "improved_rule",
        "decision_rule",
        "timeframes",
        "bar_construction",
        "feature_engine",
        "warmup_bars",
        "alignment",
        "contiguity_policy",
        "eligibility_conditions",
        "block_availability_rule",
        "availability_gate",
        "held_fixed",
        "stopping_rule",
        "leakage_battery",
    ],
)
def test_every_result_critical_mechanic_is_in_the_hashed_payload(key):
    """A constant outside the payload could be edited without moving the hash."""
    assert key in payload()


def test_moving_any_of_them_moves_the_hash(monkeypatch):
    """The hash is only a commitment if it is sensitive to what it commits."""
    before = preregistration_hash()
    monkeypatch.setattr(
        "nn.p5_preregistration.DECISION_RULE",
        {**DECISION_RULE, "improved_folds_required": 2},
    )
    assert preregistration_hash() != before


def test_the_document_states_the_decision_numbers(document):
    assert "3 of 4" in document
    assert "at least 3 of 4 folds improved" in document.lower()
    assert "5/16 = 0.3125" in document
    assert "delta > 0`, strictly" in document or "strictly greater than zero" in document


def test_the_document_and_the_module_agree_on_the_deciding_cell(document):
    assert f"`{PRIMARY_MODEL} × {COMBINED}`" in document
    assert f"`{PRIMARY_MODEL} × {CONTROL}`" in document
    assert tuple(PRIMARY_COMPARISON) == (COMBINED, CONTROL)


def test_the_document_states_the_measured_availability(document):
    """The measurement is what lets the denominator be fixed before the test."""
    assert "44,171" in document
    assert "45,802" in document
    assert "1,631" in document
    assert "0.9644" in document
    assert "4 of 4" in document


# --- exactly one axis changes ----------------------------------------------


def test_the_three_arms_are_the_control_the_family_and_their_union():
    assert ARMS == (CONTROL, FAMILY, COMBINED)
    assert CONTROL == "ohlcv14"
    assert FAMILY == "mtf_v1"
    assert COMBINED == "ohlcv14_plus_mtf_v1"


def test_the_target_and_costs_are_the_ones_every_earlier_checkpoint_used():
    """Moving the horizon replaces the control with an unmeasured object."""
    spec = TargetSpec()
    assert HELD_FIXED["label_horizon_candles"] == spec.horizon == 6
    assert HELD_FIXED["cost_threshold"] == spec.cost_threshold == 0.002
    assert HELD_FIXED["round_trip_cost"] == 0.002
    assert "0.0005" in HELD_FIXED["target"]


def test_the_asset_the_clock_and_the_folds_are_unchanged():
    assert HELD_FIXED["exchange"] == "binance"
    assert HELD_FIXED["pair"] == "BTC/USDT"
    assert HELD_FIXED["base_timeframe"] == "1h"
    assert HELD_FIXED["folds"] == 4
    assert HELD_FIXED["seed"] == 42
    assert HELD_FIXED["seq_len"] == 64
    assert HELD_FIXED["control"] == CONTROL


def test_p5_acquires_no_new_source():
    """The property that makes 'P5 adds no new way to reach Styx' true."""
    assert "no new source" in HELD_FIXED["source"]


# --- the family is exactly the OHLCV14 engine on two other clocks -----------


def test_the_family_is_the_ohlcv14_engine_and_says_so():
    assert FEATURE_ENGINE["function"] == "chimera.features.compute_features"
    assert FEATURE_ENGINE["columns_per_timeframe"] == len(feature_columns()) == 14
    assert FEATURE_ENGINE["total_columns"] == 28


def test_the_columns_are_the_fourteen_names_on_each_of_two_clocks():
    columns = mtf_columns()
    assert len(columns) == 28 == len(set(columns))
    for timeframe in ("4h", "1d"):
        prefix = TIMEFRAMES[timeframe]["prefix"]
        assert [c[len(prefix) :] for c in columns if c.startswith(prefix)] == feature_columns()


def test_the_windows_are_not_rescaled_and_the_document_says_why(document):
    """Rescaling would make the arm a smoothed copy of the control."""
    assert FEATURE_ENGINE["windows_are_measured_in"].startswith("bars")
    assert "smoothed copy of the control" in FEATURE_ENGINE["why_not_rescaled"]
    assert "not rescaled" in document


def test_the_warmup_is_the_repositorys_own_warmup():
    """78 is `FeatureSpec.warmup`, the same number the 1h spine was built with."""
    assert WARMUP_BARS == FeatureSpec().warmup == 78


def test_a_partial_bar_is_not_a_bar():
    from nn.p5_preregistration import BAR_CONSTRUCTION

    assert "every constituent 1h candle must be present" in BAR_CONSTRUCTION["completeness"]
    assert BAR_CONSTRUCTION["incomplete_bar_policy"].startswith("dropped entirely")


def test_the_grid_is_fixed_utc_and_not_a_rolling_window(document):
    assert TIMEFRAMES["4h"]["hours"] == 4
    assert TIMEFRAMES["1d"]["hours"] == 24
    for timeframe in TIMEFRAMES.values():
        assert timeframe["grid"].startswith("UTC")
    assert "A fixed grid and not a rolling window" in document


# --- causality --------------------------------------------------------------


def test_alignment_requires_a_closed_bar():
    assert "close time is <= t" in ALIGNMENT["rule"]
    assert ALIGNMENT["strictly_causal"].startswith("a bar that has not closed")


def test_a_stale_context_makes_a_row_ineligible_rather_than_being_served():
    assert ALIGNMENT["staleness_bound_bars"] == 1
    assert "INELIGIBLE" in ALIGNMENT["staleness_rule"]


def test_the_contiguity_decision_records_the_measurement_that_settled_it():
    """It changes which folds exist, so the reason may not be reconstructed later."""
    assert CONTIGUITY_POLICY["measured_dropped_bars"] == {"4h": [20, 11792], "1d": [16, 1966]}
    assert "CHANGES WHICH FOLDS EXIST" in CONTIGUITY_POLICY["why_rejected"]
    assert "0.667" in CONTIGUITY_POLICY["why_rejected"]
    assert "0.964" in CONTIGUITY_POLICY["why_rejected"]


def test_the_leakage_battery_gives_every_check_a_positive_control():
    """A check that has never failed is not evidence."""
    assert len(LEAKAGE_BATTERY) == 10
    ids = [item["id"] for item in LEAKAGE_BATTERY]
    assert ids == [f"L{n}" for n in range(1, 11)]
    for item in LEAKAGE_BATTERY:
        assert item["must_show"].strip()
        assert item["positive_control"].strip()


def test_the_battery_covers_the_three_things_that_could_leak():
    text = " ".join(item["must_show"] for item in LEAKAGE_BATTERY).lower()
    assert "not-yet-closed 4h bar" in text
    assert "the same for 1d" in text
    assert "forward-filled" in text
    assert "styx" in text
    assert "p4-hold" in text
    assert "labels are unchanged" in text


# --- the decision rule ------------------------------------------------------


def test_one_deciding_model_and_two_that_cannot_switch_it():
    assert PRIMARY_MODEL == "xgboost"
    assert set(SECONDARY_MODELS) == {"logistic_regression", "lightgbm"}
    assert set(MODELS) == {PRIMARY_MODEL, *SECONDARY_MODELS}
    assert len(MODELS) == 3


def test_improved_is_a_strict_inequality():
    """Adding nothing is the null, not a win, and zero is reachable in practice."""
    assert IMPROVED_RULE["improved_when"] == "delta > 0"
    assert IMPROVED_RULE["zero_is_improved"] is False


def test_the_bar_is_three_of_four_and_nothing_else_decides():
    assert DECISION_RULE["folds"] == 4
    assert DECISION_RULE["improved_folds_required"] == 3
    assert DECISION_RULE["decided_by"] == {
        "model": PRIMARY_MODEL,
        "comparison": list(PRIMARY_COMPARISON),
    }
    assert DECISION_RULE["cost_multiplier"] == 1.0


def test_the_mean_can_neither_rescue_nor_veto_the_fold_count():
    """P2b had two arms with a positive mean while improving one and two folds."""
    for key in ("mean_delta", "worst_fold_delta"):
        assert DECISION_RULE[key].startswith("descriptive")
    assert "may not rescue a fold-count failure" in DECISION_RULE["mean_delta"]
    assert "may not veto a pass" in DECISION_RULE["mean_delta"]


def test_the_trade_count_is_a_flag_and_never_a_denominator():
    """A denominator that can move is a denominator someone can move."""
    assert TRADE_COUNT_DIAGNOSTIC["flag_below_outer_trades"] == 10
    assert TRADE_COUNT_DIAGNOSTIC["effect_on_the_denominator"] == "none"
    assert TRADE_COUNT_DIAGNOSTIC["effect_on_the_decision"] == "none"


def test_the_screen_records_the_null_it_is_weak_against():
    assert "5/16 = 0.3125" in DECISION_RULE["false_positive_rate_under_coin_null"]


# --- availability -----------------------------------------------------------


def test_the_availability_rule_is_p4s_shape_and_p4s_numbers():
    from nn.p4_preregistration import BLOCK_AVAILABILITY_RULE as P4_RULE

    assert BLOCK_AVAILABILITY_RULE["min_eligible_row_fraction"] == 0.98
    assert BLOCK_AVAILABILITY_RULE["max_contiguous_ineligible_hours"] == 48
    assert P4_RULE["min_surviving_row_fraction"] == 0.98
    assert P4_RULE["max_contiguous_missing_hours"] == 48


def test_the_availability_rule_also_covers_the_inner_block():
    """The threshold is selected there; P4 had no inner-block condition."""
    assert "inner-validation" in BLOCK_AVAILABILITY_RULE["applies_to"]
    assert "outer-validation" in BLOCK_AVAILABILITY_RULE["applies_to"]
    assert BLOCK_AVAILABILITY_RULE["training_blocks"].startswith("reported, never gating")


def test_all_four_folds_or_nothing():
    """A denominator free to move after results are seen is what this forecloses."""
    assert AVAILABILITY_GATE == {
        "folds_required": 4,
        "of": 4,
        "on_failure": "not_evaluable",
        "on_failure_means": AVAILABILITY_GATE["on_failure_means"],
    }
    assert "does NOT re-derive a bar" in AVAILABILITY_GATE["on_failure_means"]
    assert "does not drop a fold" in AVAILABILITY_GATE["on_failure_means"]


def test_the_availability_measurement_was_taken_before_any_fit():
    assert MEASURED_AVAILABILITY["measured_before_any_fit"] is True
    assert MEASURED_AVAILABILITY["spine_rows"] == 45802
    assert MEASURED_AVAILABILITY["eligible_rows"] == 44171
    assert MEASURED_AVAILABILITY["ineligible_rows"] == 1631
    assert MEASURED_AVAILABILITY["ineligible_span"] == [0, 1630]
    assert MEASURED_AVAILABILITY["folds_available"] == 4
    assert MEASURED_AVAILABILITY["per_fold_inner_eligible_fraction"] == [1.0, 1.0, 1.0, 1.0]
    assert MEASURED_AVAILABILITY["per_fold_outer_eligible_fraction"] == [1.0, 1.0, 1.0, 1.0]


def test_the_eligibility_conditions_are_stated_and_finite():
    assert len(ELIGIBILITY_CONDITIONS) == 5
    text = " ".join(ELIGIBILITY_CONDITIONS)
    assert "as-of complete 4h bar" in text
    assert "as-of complete 1d bar" in text
    assert "WARMUP_BARS" in text
    assert "staleness bound" in text
    assert "finite" in text


# --- the boundaries that do not move ---------------------------------------


def test_the_evidence_ceiling_forbids_reading_a_pass_as_confirmation():
    assert "cannot confirm anything" in EVIDENCE_CEILING
    assert "eight prior readings" in EVIDENCE_CEILING
    assert STOPPING_RULE["on_pass"].startswith("P5 is supportive exploratory ADAPTIVE")
    assert "does not open Styx" in STOPPING_RULE["on_pass"]


def test_a_negative_p5_may_not_be_answered_with_mtf_v2():
    assert "Do not tune mtf_v1" in STOPPING_RULE["on_fail"]
    assert "Do not create mtf_v2" in STOPPING_RULE["on_fail"]
    assert "changes axis" in STOPPING_RULE["on_fail"]


def test_a_gate_failure_is_not_written_up_as_a_negative():
    assert STOPPING_RULE["on_not_evaluable"].startswith("P5 is reported invalid")
    assert "is not a negative result" in STOPPING_RULE["on_not_evaluable"]


def test_p5_has_one_stage_and_no_holdout():
    assert STOPPING_RULE["stages"] == 1
    assert STOPPING_RULE["holdout"] is None


def test_p4_hold_is_named_as_unavailable_rather_than_merely_unused():
    assert "retired unread" in P4_HOLD_UNAVAILABILITY
    assert "checkpoint: null" in P4_HOLD_UNAVAILABILITY
    assert "one-way" in P4_HOLD_UNAVAILABILITY


def test_the_styx_prohibition_does_not_restate_the_sealed_instant():
    """One source of truth: `tests/test_research_contracts.py` enforces the rest."""
    assert "sealed_test_start" in STYX_PROHIBITION
    assert "2025-08-27" not in STYX_PROHIBITION


def test_adding_a_success_criterion_after_a_result_is_forbidden_by_name():
    forbidden = " ".join(FORBIDDEN_AFTER_RESULTS)
    assert "addition of any further success criterion" in forbidden
    assert "deciding model" in forbidden
    assert "the fold plan or the number of folds" in forbidden
    assert "re-running a valid cell because its number is disappointing" in forbidden


# --- P5 has run, and the design above did not move -------------------------
#
# Until P5 ran, the two tests below were tripwires asserting that nothing had
# been started — "P5 has evidence now" should be a diff somebody wrote, not a
# directory listing that quietly changed. It is that diff. They now assert the
# other half of the same property: that the evidence exists, and that it was
# produced under *this* preregistration rather than an edited one.


def test_p5_produced_exactly_the_nine_cells_its_arms_and_models_require():
    """Three arms times three models. Not eight, and not ten."""
    benchmark = ROOT / "artifacts" / "benchmark"
    expected = {f"btc_p5_{arm}_{model}" for arm in ARMS for model in MODELS}
    found = {p.name for p in benchmark.glob("btc_p5_*") if p.is_dir()}
    assert expected <= found, f"missing cells: {sorted(expected - found)}"
    assert found - expected == {"btc_p5_comparison", "btc_p5_decision"}, sorted(
        found - expected
    )


def test_every_cell_records_this_preregistration_and_not_another():
    """A cell produced under an edited design is a different object.

    This is what the hash is for, and it is checked against the artifacts rather
    than only against the module — an edit to the preregistration after the fact
    would move the hash here and leave nine cells carrying the old one.
    """
    benchmark = ROOT / "artifacts" / "benchmark"
    for arm in ARMS:
        for model in MODELS:
            cell = json.loads((benchmark / f"btc_p5_{arm}_{model}" / "p2b.json").read_text())
            assert (
                cell["mtf_spec"]["preregistration_hash"] == preregistration_hash()
            ), f"btc_p5_{arm}_{model} was produced under a different preregistration"


def test_the_checkpoint_is_answered_by_its_own_evidence():
    from nn.research_state import checkpoint_states

    assert checkpoint_states(ROOT)["P5"] == "answered"


def test_the_document_is_a_front_door_document():
    from nn.research_state import FRONT_DOOR_DOCUMENTS

    assert "docs/p5_preregistration.md" in FRONT_DOOR_DOCUMENTS


def test_the_checkpoint_name_is_p5():
    assert CHECKPOINT == "P5"


def test_the_payload_serialises_to_json():
    """The hash is taken over a JSON dump; a value that will not serialise breaks it."""
    assert json.loads(json.dumps(payload(), sort_keys=True))
