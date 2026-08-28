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

import pandas as pd
import pytest

from nn.derivatives_sources import KLINE_COLUMNS
from nn.information_sets import CHECKPOINTS
from nn.p4_preregistration import (
    ARMS,
    AVAILABILITY_GATE,
    BLOCK_AVAILABILITY_RULE,
    EVIDENCE_CLASSIFICATION,
    FUNDING_CSV_COLUMN_POLICY,
    HOLDOUT_SPEND_POLICY,
    IMPROVED_RULE,
    NOT_EVALUABLE_OUTCOME,
    STAGE_1_MAX_ROW_EXCLUSIVE,
    STAGE_2_EVALUATION_ROWS,
    STAGE_2_SELECTION_ROWS,
    STAGE_2_TRAIN_ROWS,
    COMBINED,
    CONTROL,
    COST_SENSITIVITY_MULTIPLIERS,
    DERIVATIVES_V1,
    DATA_SOURCES,
    DEGREES_OF_FREEDOM,
    EXPLORATORY_OUTER_BLOCKS,
    FEATURES,
    FUNDING_ARCHIVE_INCEPTION_POLICY,
    FEATURE_NAMES,
    PERPETUAL_KLINE_ARCHIVE_INCEPTION_POLICY,
    HOLDOUT_ROWS,
    MIN_OUTER_TRADES,
    NULL_PROBABILITY_THREE_OF_FOUR,
    OPEN_INTEREST_DUPLICATE_POLICY,
    OPEN_INTEREST_OBSERVATION_VALIDITY_POLICY,
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

#: The hash the original preregistration carried, before amendment A1 (§3.4a) put
#: `open_interest_duplicate_policy` into the payload. Historical provenance: no P4
#: cell, fit or outcome was produced under it, and the tests below assert both that
#: the document still records it and that nothing may be run under it.
SUPERSEDED_HASH = "68ba94f49099c90772cc29d9ed6ea0cb1c4fb3b49a457924e9c3ca9f9af865a4"

#: The hash the design carried between amendment A1 and amendment A2 (§3.4b), before
#: `funding_archive_inception_policy` entered the payload. Historical provenance on the
#: same terms: no P4 cell, fit or outcome was produced under it either.
SUPERSEDED_HASH_A1 = "e0c9a7aadd69abd8c6b81abe6d570545dbbf638884740d8d78dab8df27f783a5"

#: The hash the design carried between amendment A2 and amendment A3 (§3.4c), before
#: `perpetual_kline_archive_inception_policy` entered the payload. Historical provenance
#: on the same terms: no P4 cell, fit or outcome was produced under it either.
SUPERSEDED_HASH_A2 = "db0eb78012bae3495c3722f0d3abc37b528f2465c0019b9ff304efcec1febb05"

#: The hash the design carried between amendment A3 and amendment A4 (§3.4d), before
#: `open_interest_observation_validity_policy` entered the payload. Historical
#: provenance on the same terms: no P4 cell, fit or outcome was produced under it either.
SUPERSEDED_HASH_A3 = "dffec07e3f5004847d5ad7960b6b16d9cf86a146f336841882c3cf6ed69c5f45"


# --- P4 is implemented, registered, and has not been run ---------------------
def test_p4_is_a_registered_checkpoint_with_its_three_preregistered_arms():
    """§13's checklist requires the registration; it is not permission to run.

    This assertion used to be its opposite: `P4` was absent from `CHECKPOINTS`
    so that `--checkpoint P4` was refused by there being no engine. The engine
    exists now, and an absence is a weaker guarantee than a gate — it stops
    working the moment someone adds the arms. What stops a fit is
    `nn.p4_stage1`'s interlock, which the tests below assert is closed.
    """
    assert "P4" in CHECKPOINTS
    assert CHECKPOINTS["P4"].arms == ARMS
    assert CHECKPOINTS["P4"].control == CONTROL
    assert CHECKPOINTS["P4"].family == DERIVATIVES_V1


def test_p4_authorisation_still_requires_runtime_gates():
    """The committed authorisation does not bypass CLI confirmation or availability."""
    from nn.p4_stage1 import Stage1Interlock, assert_fit_authorised, read_authorisation

    assert read_authorisation()["state"] == "authorised"
    with pytest.raises(Stage1Interlock):
        assert_fit_authorised(confirm=False, availability={"gate_passed": True})
    with pytest.raises(Stage1Interlock):
        assert_fit_authorised(confirm=True, availability={"gate_passed": False})


def test_no_p4_evidence_exists():
    assert not (ROOT / "artifacts" / "benchmark" / "btc_p4_comparison").exists()
    assert not list((ROOT / "artifacts" / "benchmark").glob("btc_p4_*"))


def test_no_p4_market_data_has_been_acquired():
    research = ROOT / "data" / "research"
    assert not list(research.glob("*funding*"))
    assert not list(research.glob("*open_interest*"))
    assert not list(research.glob("*metrics*"))
    assert not list(research.glob("*basis*"))
    assert not list(research.glob("*perp*"))
    # Nothing P4 touches is a table. The two P4 files are both *states* rather
    # than observations: whether the holdout region has been spent, and whether
    # anyone has authorised a stage-1 fit. Neither carries an observation of any
    # market at all.
    #
    # This list is a tripwire, deliberately. Acquiring the derivatives source
    # adds a P4 *table*, and the session that acquires it has to come here and
    # say so — which is the point: "P4 has data now" should be a diff somebody
    # wrote, not a directory listing that quietly changed.
    #
    # `p4_holdout_coverage.json` is permitted and is not an acquisition. It is
    # the third *state* file: which of P4-HOLD's own archive days the source
    # publishes, established by `tools.export_derivatives_snapshot --probe` from
    # one HEAD request per day. It carries an existence flag per day and no
    # observation of any market — the same class of file as the two above, and
    # the reason it is named here rather than matched by a wildcard.
    permitted = {
        "p4_holdout_ledger.json",
        "p4_holdout_coverage.json",
        "p4_stage1_authorisation.json",
    }
    p4_files = sorted(path.name for path in research.glob("*p4*"))
    assert set(p4_files) <= permitted, sorted(set(p4_files) - permitted)
    assert "p4_holdout_ledger.json" in p4_files
    assert "p4_stage1_authorisation.json" in p4_files
    authorisation = json.loads((research / "p4_stage1_authorisation.json").read_text())
    assert authorisation["state"] == "authorised"
    ledger = json.loads((research / "p4_holdout_ledger.json").read_text())
    assert set(ledger) == {
        "ledger_schema",
        "region",
        "region_span",
        "state",
        "checkpoint",
        "reason",
        "evaluations_permitted",
        "checkpoints_permitted",
        "retired_if_unspent",
        "label_ceiling",
        "note",
    }


def test_the_derivatives_engine_computes_the_preregistered_columns_and_no_others():
    """The engine reads §5 rather than restating it, so drift is impossible.

    This test used to assert that ``nn/derivatives.py`` did not exist. It does
    now, and the guarantee it replaces the absence with is stronger: the column
    list, the windows and the clips are *read from* the preregistration at
    import, so a ninth column or a moved window is a change to the hashed
    payload rather than a change the engine could make on its own.
    """
    from nn.derivatives import DERIVATIVES_FEATURE_COLUMNS, DerivativesSpec

    assert list(DERIVATIVES_FEATURE_COLUMNS) == list(FEATURE_NAMES)
    assert len(DERIVATIVES_FEATURE_COLUMNS) == 8
    material = DerivativesSpec().spec_hash()
    assert isinstance(material, str) and len(material) == 64


def test_the_three_p4_arms_are_built_from_those_columns_and_nothing_else():
    from nn.derivatives import DERIVATIVES_FEATURE_COLUMNS
    from nn.information_sets import information_set

    control = information_set(CONTROL)
    family = information_set(DERIVATIVES_V1)
    combined = information_set(COMBINED)
    assert family.columns == tuple(DERIVATIVES_FEATURE_COLUMNS)
    assert combined.columns == control.columns + family.columns
    assert len(combined.columns) == len(control.columns) + 8


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


def test_the_document_and_the_module_agree_on_the_mechanics():
    """A doc section and a hashed value that drift apart is the old failure.

    Every number below decides something, and each one appears in both places;
    the test exists so that changing one of them in prose alone fails here
    rather than being discovered when a run disagrees with its own design.
    """
    text = DOCUMENT.read_text()
    rule = BLOCK_AVAILABILITY_RULE
    assert f"{rule['min_surviving_row_fraction']:.0%}" in text
    assert f"**{rule['max_contiguous_missing_hours']} hours**" in text
    assert "2020-09-01" in text
    assert "/daily/metrics/" in text
    assert "`delta > 0`, strictly" in text
    assert "not_evaluable" in text and "insufficient_coverage" in text
    for start, end in (STAGE_2_TRAIN_ROWS, STAGE_2_SELECTION_ROWS, STAGE_2_EVALUATION_ROWS):
        assert f"[{start}, {end})" in text


def test_the_document_states_the_mechanical_guard_rather_than_only_promising():
    text = DOCUMENT.read_text()
    assert "nn/p4_holdout.py" in text
    assert "p4_holdout_ledger.json" in text
    assert "assert_holdout_release" in text
    assert "retired" in text


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


# --- the science did not move ------------------------------------------------
def test_the_feature_family_is_exactly_what_was_preregistered():
    """Literal, so that a later edit to a window is a failing test.

    Every other test here checks a property — eight columns, bounded clips,
    named families. This one checks the values, because the whole point of a
    preregistration is that these particular numbers were fixed before the data
    existed, and a property test would pass a wholesale replacement.
    """
    assert [(f["name"], f["window"], tuple(f["clip"])) for f in FEATURES] == [
        ("drv_funding_last", None, (-0.01, 0.01)),
        ("drv_funding_sum_9", 9, (-0.09, 0.09)),
        ("drv_funding_z", 30, (-5.0, 5.0)),
        ("drv_oi_log_change_24h", 24, (-1.0, 1.0)),
        ("drv_oi_notional_ratio", 168, (0.0, 10.0)),
        ("drv_oi_price_divergence", 24, (-1.0, 1.0)),
        ("drv_basis", None, (-0.02, 0.02)),
        ("drv_basis_z", 168, (-5.0, 5.0)),
    ]


def test_the_model_target_and_costs_are_exactly_what_was_preregistered():
    assert PRIMARY_MODEL == "xgboost"
    assert SECONDARY_MODELS == ("logistic_regression", "lightgbm")
    assert TARGET["timeframe"] == "1h" and TARGET["horizon"] == 6
    assert TARGET["fee_rate"] == 0.0005 and TARGET["slippage_rate"] == 0.0005
    assert TARGET["cost_threshold"] == 0.002
    assert COST_SENSITIVITY_MULTIPLIERS == (1.0, 1.5, 2.0)
    assert MIN_OUTER_TRADES == 10
    assert STAGE_1_CONTINUATION["valid_folds_required"] == 3
    assert STAGE_1_CONTINUATION["improved_folds_required"] == 3
    assert STAGE_1_CONTINUATION["worst_fold_delta_at_least"] == -0.02


# --- the sources name something that exists ----------------------------------
def test_the_open_interest_source_is_the_daily_archive():
    """The first version named a monthly metrics path Binance does not publish.

    A preregistration that commits to a nonexistent URL commits to nothing, and
    the correction had to happen here — before any probe and before any P4
    number — because after either one it is a source chosen against data.
    """
    oi = next(s for s in DATA_SOURCES if s["field"] == "open_interest")
    assert "/daily/metrics/" in oi["primary"]
    assert "/monthly/metrics/" not in oi["primary"]
    assert "{day}" in oi["primary"]
    assert oi["archive_granularity"].startswith("daily")
    assert oi["earliest_intended_availability"] == "2020-09-01"
    assert "288" in oi["missing_day_behaviour"]
    assert "fail closed" in oi["coverage_failure_behaviour"]


def test_the_rest_endpoint_is_diagnostic_and_may_not_stand_in():
    oi = next(s for s in DATA_SOURCES if s["field"] == "open_interest")
    assert "openInterestHist" in oi["fallback"]
    assert "30 days" in oi["fallback"]
    assert "NEVER silently stand in" in oi["fallback"]


def test_every_named_source_is_a_daily_or_monthly_path_that_carries_its_period():
    """A template with no period placeholder cannot address an archive."""
    for source in DATA_SOURCES:
        if not source["primary"].startswith("https://data.binance.vision"):
            continue
        assert "{year}" in source["primary"] and "{month}" in source["primary"]


def test_the_funding_column_mapping_is_an_allow_list_fixed_before_any_archive():
    """So that a schema discovered later cannot become a choice made later."""
    policy = FUNDING_CSV_COLUMN_POLICY
    assert policy["canonical_fields"] == ["settlement_instant", "realised_funding_rate"]
    assert len(policy["allowed_header_maps"]) >= 2
    for layout in policy["allowed_header_maps"]:
        assert layout["settlement_instant"] in layout["columns"]
        assert layout["realised_funding_rate"] in layout["columns"]
    assert "refuse" in policy["on_unrecognised_layout"]
    assert "do not infer" in policy["on_unrecognised_layout"].lower()
    assert "moves the preregistration hash" in policy["on_unrecognised_layout"]


def test_the_headerless_case_is_bounded_rather_than_positional_guessing():
    layout = FUNDING_CSV_COLUMN_POLICY["headerless_positional_layout"]
    assert layout["columns"] == 2
    assert "calendar period" in layout["condition"]


# --- the decision mechanics are inside the hash ------------------------------
@pytest.mark.parametrize(
    "key",
    [
        "stage_2_train_rows",
        "stage_2_selection_rows",
        "stage_2_evaluation_rows",
        "stage_1_max_row_exclusive",
        "availability_gate",
        "block_availability_rule",
        "universe_conditions",
        "holdout_spend_policy",
        "improved_rule",
        "evidence_classification",
        "not_evaluable_outcome",
        "funding_csv_column_policy",
        "funding_archive_inception_policy",
        "perpetual_kline_archive_inception_policy",
        "open_interest_observation_validity_policy",
    ],
)
def test_every_result_critical_mechanic_is_in_the_hashed_payload(key):
    """Prose can be edited by the commit that reports the result. This cannot."""
    assert key in preregistration()


def test_moving_any_of_them_moves_the_hash(monkeypatch):
    """The property the previous test is only half of.

    Being *in* the payload is not enough — the payload has to be what the hash
    is taken over. Each mechanic is perturbed in turn and the hash must move.
    """
    import nn.p4_preregistration as prereg

    before = preregistration_hash()
    for name, value in (
        ("STAGE_2_TRAIN_ROWS", (0, 40000)),
        ("STAGE_2_SELECTION_ROWS", (40000, 45802)),
        ("STAGE_2_EVALUATION_ROWS", (45802, 48000)),
        ("STAGE_1_MAX_ROW_EXCLUSIVE", 46000),
        ("MIN_OUTER_TRADES", 11),
        ("AVAILABILITY_GATE", {"requires_exploratory_blocks_available": 1}),
        ("BLOCK_AVAILABILITY_RULE", {"min_surviving_row_fraction": 0.5}),
        ("IMPROVED_RULE", {"improved_when": "delta >= 0"}),
        ("HOLDOUT_SPEND_POLICY", {"evaluations_permitted": 2}),
        ("EVIDENCE_CLASSIFICATION", {"maximum_label": "confirmatory"}),
        ("FUNDING_CSV_COLUMN_POLICY", {"canonical_fields": []}),
        ("OPEN_INTEREST_DUPLICATE_POLICY", {"grouping_key": "other"}),
        ("FUNDING_ARCHIVE_INCEPTION_POLICY", {"first_protocol_month": "2021-06"}),
        ("PERPETUAL_KLINE_ARCHIVE_INCEPTION_POLICY", {"first_protocol_month": "2021-06"}),
        (
            "OPEN_INTEREST_OBSERVATION_VALIDITY_POLICY",
            {"validity_rule": "any published row is an observation"},
        ),
    ):
        with monkeypatch.context() as patch:
            patch.setattr(prereg, name, value)
            assert prereg.preregistration_hash() != before, name
    assert preregistration_hash() == before


def test_the_stage_two_geometry_is_contiguous_and_ends_at_the_holdout():
    assert STAGE_2_TRAIN_ROWS[0] == 0
    assert STAGE_2_TRAIN_ROWS[1] == STAGE_2_SELECTION_ROWS[0]
    assert STAGE_2_SELECTION_ROWS[1] == HOLDOUT_ROWS[0]
    assert tuple(STAGE_2_EVALUATION_ROWS) == tuple(HOLDOUT_ROWS)
    assert STAGE_2_SELECTION_ROWS == EXPLORATORY_OUTER_BLOCKS[-1]


def test_stage_one_cannot_reach_the_stage_two_evaluation_region():
    assert STAGE_1_MAX_ROW_EXCLUSIVE == HOLDOUT_ROWS[0]


# --- the ambiguities are gone ------------------------------------------------
def test_improved_is_a_strict_inequality():
    assert IMPROVED_RULE["improved_when"] == "delta > 0"
    assert IMPROVED_RULE["zero_is_improved"] is False
    assert STAGE_1_CONTINUATION["improved_rule"]["zero_is_improved"] is False


def test_blocks_in_full_has_an_operational_rule_for_a_punctured_feed():
    rule = BLOCK_AVAILABILITY_RULE
    assert 0 < rule["min_surviving_row_fraction"] < 1
    assert rule["max_contiguous_missing_hours"] >= 24
    assert "P4-HOLD" in rule["applies_to"]
    assert AVAILABILITY_GATE["block_rule"] == "BLOCK_AVAILABILITY_RULE"


def test_one_missing_day_is_survivable_and_a_missing_week_is_not():
    """The rule read against the mechanism that produces the gaps.

    A missing daily OI archive removes 24 hours. An outer block is 4,821 rows,
    so the fraction tolerates about four such days; the contiguity bound
    tolerates two in a row and no more.
    """
    block_rows = EXPLORATORY_OUTER_BLOCKS[0][1] - EXPLORATORY_OUTER_BLOCKS[0][0]
    tolerated_hours = block_rows * (1 - BLOCK_AVAILABILITY_RULE["min_surviving_row_fraction"])
    assert 24 <= tolerated_hours < 24 * 7
    assert BLOCK_AVAILABILITY_RULE["max_contiguous_missing_hours"] == 48


def test_insufficient_coverage_is_not_a_negative_research_result():
    outcome = NOT_EVALUABLE_OUTCOME
    assert outcome["label"] == "not_evaluable"
    assert outcome["reason_code"] == "insufficient_coverage"
    assert outcome["classification"] == "not a research result"
    assert "not_evaluable" in EVIDENCE_CLASSIFICATION
    assert "NOT negative evidence" in EVIDENCE_CLASSIFICATION["not_evaluable"]
    assert "do not report P4 as negative" in outcome["then"]
    # And it is not one of the stage-2 outcomes: nothing was measured.
    assert outcome["label"] not in {o["label"] for o in STAGE_2_OUTCOMES}


# --- the holdout is spendable once, ever -------------------------------------
def test_the_spend_policy_is_one_evaluation_by_one_checkpoint():
    policy = HOLDOUT_SPEND_POLICY
    assert policy["evaluations_permitted"] == 1
    assert policy["checkpoints_permitted"] == 1
    assert policy["retired_if_unspent"] is True
    assert policy["region"] == list(HOLDOUT_ROWS)
    assert policy["enforced_by"] == "nn.p4_holdout"


def test_retirement_forecloses_the_p4b_reuse_argument():
    note = HOLDOUT_SPEND_POLICY["note"]
    assert "P4b" in note and "P5" in note
    assert "fresh holdout" in note
    assert "does not forbid describing" in note


def test_spending_the_holdout_does_not_upgrade_the_label():
    assert "never confirmatory" in HOLDOUT_SPEND_POLICY["does_not_upgrade"]
    assert EVIDENCE_CLASSIFICATION["maximum_label"].endswith("never confirmatory")


def test_the_committed_ledger_matches_the_preregistered_policy():
    ledger = json.loads((ROOT / "data" / "research" / "p4_holdout_ledger.json").read_text())
    assert ledger["region"] == HOLDOUT_SPEND_POLICY["region"]
    assert ledger["evaluations_permitted"] == HOLDOUT_SPEND_POLICY["evaluations_permitted"]
    assert ledger["checkpoints_permitted"] == HOLDOUT_SPEND_POLICY["checkpoints_permitted"]
    assert ledger["retired_if_unspent"] == HOLDOUT_SPEND_POLICY["retired_if_unspent"]
    assert ledger["state"] == "unspent"


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


# --- amendment A1: the duplicate policy, and the two things it must not do ----
def test_the_duplicate_policy_is_inside_the_hashed_preregistration():
    """In the payload, not only in a module namespace beside it."""
    assert preregistration()["open_interest_duplicate_policy"] == dict(
        OPEN_INTEREST_DUPLICATE_POLICY
    )


def test_the_duplicate_policy_is_scoped_to_open_interest_and_refuses_conflicts():
    policy = OPEN_INTEREST_DUPLICATE_POLICY
    assert policy["scope"]["field"] == "open_interest"
    assert policy["scope"]["applies_to"] == ["open_interest"]
    assert sorted(policy["scope"]["does_not_apply_to"]) == ["funding_rate", "perpetual_price"]
    assert policy["grouping_key"] == "create_time"
    for forbidden in ("first", "last", "average", "infer"):
        assert forbidden in policy["on_conflict"].lower(), forbidden
    assert set(policy["provenance_required"]) == {
        "rows_read",
        "observations_retained",
        "exact_duplicate_rows_collapsed",
        "duplicate_instants",
    }


def test_the_policy_records_that_it_is_an_amendment_and_what_it_replaced():
    """The history is data, not a story told about the data.

    A rule adopted after looking at a source is a different kind of object from
    one fixed before, and the payload has to say which this is — otherwise the
    hash certifies a commitment whose provenance it cannot express.
    """
    policy = OPEN_INTEREST_DUPLICATE_POLICY
    assert policy["amendment"] == "A1"
    assert "before any P4 model fit" in policy["amendment_status"]
    assert "ANY source" in policy["supersedes"]
    assert "2020-09-01" in policy["adopted_because"]
    assert "2020-09-02" in policy["adopted_because"]
    assert "no claim is made" in policy["adopted_because"].lower()


def test_the_source_spec_reads_the_policy_rather_than_restating_it():
    """The anti-drift property, asserted as identity rather than as similarity."""
    from nn.derivatives_sources import source_spec

    assert source_spec()["duplicate_rule"] == dict(OPEN_INTEREST_DUPLICATE_POLICY)


def test_editing_the_policy_moves_both_hashes_together(monkeypatch):
    """`preregistration says X, source_spec says Y` must be unconstructible.

    Both identities are recomputed under a perturbed policy. If either failed to
    move, a later edit could leave the design and the acquisition disagreeing
    while the suite stayed green — which is the exact failure this amendment is
    supposed to make impossible.
    """
    import nn.p4_preregistration as prereg
    from nn.derivatives_sources import source_spec
    from tools.export_derivatives_snapshot import source_spec_hash

    before_prereg = preregistration_hash()
    before_source = source_spec_hash()
    with monkeypatch.context() as patch:
        patch.setattr(
            prereg,
            "OPEN_INTEREST_DUPLICATE_POLICY",
            {**OPEN_INTEREST_DUPLICATE_POLICY, "grouping_key": "something_else"},
        )
        assert prereg.preregistration_hash() != before_prereg
        assert source_spec_hash() != before_source
        assert source_spec()["duplicate_rule"]["grouping_key"] == "something_else"
    assert preregistration_hash() == before_prereg
    assert source_spec_hash() == before_source


def test_the_acquisition_groups_on_the_key_the_policy_names(monkeypatch, tmp_path):
    """The reader reads the policy; it does not carry its own copy of the key."""
    import zipfile

    import pandas as pd

    import nn.p4_preregistration as prereg
    from nn.derivatives_sources import DerivativesSourceError, metrics_archive
    from tools.export_derivatives_snapshot import read_metrics

    archive = metrics_archive(pd.Timestamp("2020-09-01", tz="UTC"))
    path = tmp_path / archive.name
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(
            archive.name.removesuffix(".zip") + ".csv",
            "create_time,sum_open_interest,sum_open_interest_value\n"
            "2020-09-01 00:00:00,1.0,2.0\n",
        )
    assert len(read_metrics(path, archive)[0]) == 1

    with monkeypatch.context() as patch:
        patch.setattr(
            prereg,
            "OPEN_INTEREST_DUPLICATE_POLICY",
            {**OPEN_INTEREST_DUPLICATE_POLICY, "grouping_key": "sum_open_interest"},
        )
        # Grouping on a column that is not an instant must fail in the parser,
        # which proves the key came from the policy and not from a literal.
        with pytest.raises(DerivativesSourceError):
            read_metrics(path, archive)


def test_the_document_records_the_amendment_and_preserves_the_superseded_hash():
    text = DOCUMENT.read_text()
    superseded = SUPERSEDED_HASH
    assert "3.4a" in text
    assert "Source-protocol amendment A1" in text
    assert "576" in text and "288" in text
    assert "2020-09-01" in text and "2020-09-02" in text
    # The active hash is the amended one, and the old one survives as history
    # rather than being quietly overwritten.
    assert f"sha256:{preregistration_hash()}" in text
    assert superseded in text
    assert superseded != preregistration_hash()
    assert "Superseded hash" in text
    # And it is named inside the amendment itself, not only in the header: §3.4a
    # is where the change is explained, so it is where the value it replaced has
    # to be readable.
    section = text.split("### 3.4a", 1)[1].split("### 3.5", 1)[0]
    assert superseded in section
    # The document must not claim the amendment predates the source inspection.
    assert "Amendment A1" in text.split("---", 1)[0]


def test_the_document_and_the_policy_agree_on_what_is_out_of_scope():
    """Markdown wraps prose, so the section is compared with runs of whitespace
    collapsed — otherwise the assertion tests the line width rather than the claim."""
    text = DOCUMENT.read_text()
    section = " ".join(text.split("### 3.4a", 1)[1].split("### 3.5", 1)[0].split())
    assert "no normalisation of any kind" in section
    assert "reject the acquisition" in section.lower()
    assert "funding" in section and "perpetual klines" in section
    assert OPEN_INTEREST_DUPLICATE_POLICY["grouping_key"] in section


def test_the_stage_one_authorisation_carries_only_the_active_hash_and_is_currently_authorised():
    """The later Stage 1 authorisation remains bound to the current design.

    And it names exactly one design. The superseded hash is provenance, and its
    home is the preregistration document — an interlock that carried two hashes
    would invite the question of which one it authorises.
    """
    path = ROOT / "data" / "research" / "p4_stage1_authorisation.json"
    text = path.read_text()
    payload = json.loads(text)
    assert payload["authorisation_schema"] == "chimera.p4-stage1-authorisation/1"
    assert payload["state"] == "authorised"
    assert payload["preregistration_hash"] == preregistration_hash()
    assert payload["authorised_by"] and payload["authorised_at"]
    assert SUPERSEDED_HASH not in text, "only the active hash belongs in the interlock"


def test_no_p4_result_can_be_produced_under_the_superseded_hash(tmp_path):
    """A design that no longer exists cannot authorise a fit under the new one."""
    from nn.p4_stage1 import Stage1Interlock, assert_fit_authorised

    superseded = SUPERSEDED_HASH
    research = tmp_path / "data" / "research"
    research.mkdir(parents=True)
    (research / "p4_stage1_authorisation.json").write_text(
        json.dumps(
            {
                "authorisation_schema": "chimera.p4-stage1-authorisation/1",
                "state": "authorised",
                "preregistration_hash": superseded,
                "authorised_by": "test",
                "authorised_at": "1970-01-01T00:00:00Z",
                "reason": "test",
            }
        )
    )
    with pytest.raises(Stage1Interlock, match="not permission to run another"):
        assert_fit_authorised(confirm=True, availability={"gate_passed": True}, root=tmp_path)


# --- amendment A2: where the funding archive begins, and what that is not -----
def test_the_inception_policy_is_inside_the_hashed_preregistration():
    """In the payload, not only in a module namespace beside it."""
    assert preregistration()["funding_archive_inception_policy"] == dict(
        FUNDING_ARCHIVE_INCEPTION_POLICY
    )


def test_the_inception_policy_is_scoped_to_funding_and_names_one_month():
    policy = FUNDING_ARCHIVE_INCEPTION_POLICY
    assert policy["scope"]["field"] == "funding_rate"
    assert policy["scope"]["applies_to"] == ["funding_rate"]
    assert sorted(policy["scope"]["does_not_apply_to"]) == ["open_interest", "perpetual_price"]
    assert (
        "monthly" in policy["scope"]["archive"] and "fundingRate" in policy["scope"]["archive"]
    )
    assert policy["first_protocol_month"] == "2020-01"
    assert policy["first_protocol_instant"].startswith("2020-01-01T00:00:00")
    assert set(policy["provenance_required"]) == {
        "generic_requested_from",
        "source_inception_month",
        "effective_from",
        "months_clamped",
    }


def test_the_inception_policy_records_that_it_is_an_amendment_and_what_it_replaced():
    """The history is data, not a story told about the data — as with A1."""
    policy = FUNDING_ARCHIVE_INCEPTION_POLICY
    assert policy["amendment"] == "A2"
    assert "before any P4 model fit" in policy["amendment_status"]
    assert "continuous" in policy["supersedes"]
    assert "2019-12" in policy["adopted_because"]
    assert "404" in policy["adopted_because"] and "200" in policy["adopted_because"]


def test_the_observed_evidence_is_status_codes_and_not_a_conclusion():
    """What was measured, month by month, with the boundary falling where it fell."""
    months = {
        entry["month"]: entry
        for entry in FUNDING_ARCHIVE_INCEPTION_POLICY["observed_evidence"]["months"]
    }
    for month in ("2019-09", "2019-10", "2019-11", "2019-12"):
        assert months[month]["status"] == 404 and months[month]["published"] is False
    for month in ("2020-01", "2020-02"):
        assert months[month]["status"] == 200 and months[month]["published"] is True
    # The first *required* missing archive is before the observed start of the
    # sequence, which is the whole reason A2 is a boundary and not a gap rule.
    assert max(m for m, e in months.items() if not e["published"]) < min(
        m for m, e in months.items() if e["published"]
    )
    evidence = FUNDING_ARCHIVE_INCEPTION_POLICY["observed_evidence"]
    assert evidence["first_observed_row"].startswith("2020-01-01T00:00:00")
    # The layout observed is one §3.0b already allowed; A2 changes no column rule.
    assert evidence["observed_layout"] in {
        entry["layout"] for entry in FUNDING_CSV_COLUMN_POLICY["allowed_header_maps"]
    }


def test_the_claim_is_about_the_archive_and_not_about_the_market():
    """The generalisation A2 explicitly refuses to make."""
    limit = FUNDING_ARCHIVE_INCEPTION_POLICY["generalisation_limit"]
    assert "six months were checked" in limit.lower()
    assert "monthly fundingRate archive" in limit
    assert "any kind" in limit


def test_a_pre_inception_month_is_outside_the_source_and_not_an_internal_gap():
    policy = FUNDING_ARCHIVE_INCEPTION_POLICY
    assert "not an internal continuity gap" in policy["pre_inception_behaviour"].lower()
    assert "not counted as a missing month" in policy["pre_inception_behaviour"]
    assert "max(generic requested start" in policy["acquisition_start_rule"]


def test_nothing_may_stand_in_for_the_pre_inception_region():
    substitution = FUNDING_ARCHIVE_INCEPTION_POLICY["no_substitution"]
    assert substitution.startswith("never")
    for forbidden in ("synthetic", "REST", "interpolated", "backwards"):
        assert forbidden in substitution, forbidden


def test_continuity_after_inception_is_unchanged_and_fail_closed():
    """A2 is a boundary, not a relaxation of §3.4 after it."""
    after = FUNDING_ARCHIVE_INCEPTION_POLICY["post_inception_continuity"]
    assert "mandatory" in after and "stops the acquisition" in after
    for forbidden in ("skipped", "interpolated", "forward-invented", "replaced"):
        assert forbidden in after, forbidden


def test_the_amendment_changes_no_feature_window_clip_or_bound():
    """The scientific degrees of freedom A2 must leave exactly where they were."""
    windows = {feature["name"]: feature["window"] for feature in FEATURES}
    clips = {feature["name"]: feature["clip"] for feature in FEATURES}
    assert windows["drv_funding_z"] == 30
    assert windows["drv_funding_sum_9"] == 9
    assert windows["drv_funding_last"] is None
    assert clips["drv_funding_z"] == [-5.0, 5.0]
    assert clips["drv_funding_sum_9"] == [-0.09, 0.09]
    assert clips["drv_funding_last"] == [-0.01, 0.01]
    assert WARMUP_HOURS == 240
    assert TARGET["horizon"] == 6 and TARGET["timeframe"] == "1h"
    unchanged = FUNDING_ARCHIVE_INCEPTION_POLICY["windows_unchanged"]
    assert "30-settlement" in unchanged and "9-settlement" in unchanged


def test_the_consequence_is_reported_rather_than_repaired():
    """Losing early rows is the intended outcome, and the payload says so."""
    consequence = FUNDING_ARCHIVE_INCEPTION_POLICY["downstream_consequence"]
    assert "outside the common sample universe" in consequence
    assert "EVERY arm" in consequence
    assert "reported, not repaired" in consequence


def test_editing_the_inception_policy_moves_both_hashes_together(monkeypatch):
    """`preregistration says 2020-01, acquisition asks for 2019-12` must be
    unconstructible — the same anti-drift property A1 has, for A2's rule."""
    import nn.p4_preregistration as prereg
    from nn.derivatives_sources import source_spec
    from tools.export_derivatives_snapshot import source_spec_hash

    before_prereg = preregistration_hash()
    before_source = source_spec_hash()
    with monkeypatch.context() as patch:
        patch.setattr(
            prereg,
            "FUNDING_ARCHIVE_INCEPTION_POLICY",
            {**FUNDING_ARCHIVE_INCEPTION_POLICY, "first_protocol_month": "2021-06"},
        )
        assert prereg.preregistration_hash() != before_prereg
        assert source_spec_hash() != before_source
        assert source_spec()["funding_inception_rule"]["first_protocol_month"] == "2021-06"
    assert preregistration_hash() == before_prereg
    assert source_spec_hash() == before_source


def test_the_acquisition_carries_no_second_copy_of_the_inception_month():
    """Checked with the parser, not with a grep, for the reason the REST test gives.

    The month appears in prose in these modules — a comment explaining the
    binding, a docstring naming what was asked for — and banning the characters
    would ban the documentation. What must not exist is a *value*: a string
    constant outside a docstring that a later edit could leave disagreeing with
    the hashed policy while the suite stayed green.
    """
    import ast

    month = FUNDING_ARCHIVE_INCEPTION_POLICY["first_protocol_month"]
    for name in ("nn/derivatives_sources.py", "tools/export_derivatives_snapshot.py"):
        tree = ast.parse((ROOT / name).read_text())
        docstrings = {
            id(node.body[0].value)
            for node in ast.walk(tree)
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef))
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        }
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in docstrings
            ):
                assert month not in node.value, f"{name}:{node.lineno}"


def test_the_document_records_the_amendment_and_both_superseded_hashes():
    text = DOCUMENT.read_text()
    assert "3.4b" in text
    assert "Source-protocol amendment A2" in text
    # The measured status codes, stated factually rather than summarised.
    for month in ("2019-09", "2019-10", "2019-11", "2019-12", "2020-01", "2020-02"):
        assert month in text, month
    assert "404" in text and "200" in text
    section = " ".join(text.split("### 3.4b", 1)[1].split("### 3.5", 1)[0].split())
    # The active hash, and the two it replaced, all inside the amendment itself.
    assert f"sha256:{preregistration_hash()}" in section
    assert SUPERSEDED_HASH in section
    assert SUPERSEDED_HASH_A1 in section
    assert SUPERSEDED_HASH_A1 != preregistration_hash()
    # And the amendment's own claims, in prose, matching the payload.
    assert "not an internal continuity gap" in section
    assert "max(generic requested start, 2020-01)" in section
    assert "stops the acquisition" in section
    assert "outside the common sample universe" in section
    # Ordering is preserved rather than rewritten: A1 came first, and the
    # original protocol is not retold as if 2020-01 had always been known.
    assert text.index("### 3.4a") < text.index("### 3.4b")
    assert "Amendment A2" in text.split("---", 1)[0]
    assert "the first version of this preregistration" not in section.lower()


def test_the_document_does_not_generalise_past_what_was_observed():
    text = " ".join(
        DOCUMENT.read_text().split("### 3.4b", 1)[1].split("### 3.5", 1)[0].split()
    )
    assert "Six months were checked" in text
    assert "monthly `fundingRate` archive" in text
    assert "Nothing here asserts that no BTCUSDT funding data of any kind existed" in text


def test_the_stage_one_authorisation_is_rebound_to_the_active_design_and_is_currently_authorised():
    """The later Stage 1 authorisation remains bound to the A2-updated active design."""
    path = ROOT / "data" / "research" / "p4_stage1_authorisation.json"
    text = path.read_text()
    payload = json.loads(text)
    assert payload["authorisation_schema"] == "chimera.p4-stage1-authorisation/1"
    assert payload["state"] == "authorised"
    assert payload["preregistration_hash"] == preregistration_hash()
    assert payload["authorised_by"] and payload["authorised_at"]
    assert payload["reason"]
    for superseded in (SUPERSEDED_HASH, SUPERSEDED_HASH_A1):
        assert superseded not in text, "only the active hash belongs in the interlock"


def test_no_p4_result_can_be_produced_under_the_pre_a2_hash(tmp_path):
    """An authorisation granted for the design A2 replaced authorises nothing."""
    from nn.p4_stage1 import Stage1Interlock, assert_fit_authorised

    research = tmp_path / "data" / "research"
    research.mkdir(parents=True)
    (research / "p4_stage1_authorisation.json").write_text(
        json.dumps(
            {
                "authorisation_schema": "chimera.p4-stage1-authorisation/1",
                "state": "authorised",
                "preregistration_hash": SUPERSEDED_HASH_A1,
                "authorised_by": "test",
                "authorised_at": "1970-01-01T00:00:00Z",
                "reason": "test",
            }
        )
    )
    with pytest.raises(Stage1Interlock, match="not permission to run another"):
        assert_fit_authorised(confirm=True, availability={"gate_passed": True}, root=tmp_path)


# --- amendment A3: where the perpetual kline archive begins -------------------
def test_the_perpetual_inception_policy_is_inside_the_hashed_preregistration():
    """In the payload, not only in a module namespace beside it."""
    assert preregistration()["perpetual_kline_archive_inception_policy"] == dict(
        PERPETUAL_KLINE_ARCHIVE_INCEPTION_POLICY
    )


def test_the_perpetual_inception_policy_is_scoped_to_klines_and_names_one_month():
    policy = PERPETUAL_KLINE_ARCHIVE_INCEPTION_POLICY
    assert policy["scope"]["field"] == "perpetual_price"
    assert policy["scope"]["applies_to"] == ["perpetual_price"]
    assert sorted(policy["scope"]["does_not_apply_to"]) == ["funding_rate", "open_interest"]
    archive = policy["scope"]["archive"]
    assert "MONTHLY" in archive and "1h" in archive and "kline" in archive
    assert policy["first_protocol_month"] == "2020-01"
    assert policy["first_protocol_instant"].startswith("2020-01-01T00:00:00")
    assert set(policy["provenance_required"]) == {
        "generic_requested_from",
        "source_inception_month",
        "effective_from",
        "months_clamped",
        "not_an_internal_gap",
        "amendment",
    }


def test_there_is_exactly_one_authoritative_copy_of_the_perpetual_inception_month():
    """The month is a value in one hashed object, and everything else reads it.

    Checked as identity rather than as agreement: the planner, the provenance and
    the source spec must all resolve to the *same* object, so an edit cannot leave
    one of them behind.
    """
    from nn.derivatives_sources import perpetual_inception, source_spec

    policy = PERPETUAL_KLINE_ARCHIVE_INCEPTION_POLICY
    assert source_spec()["perpetual_inception_rule"] == dict(policy)
    assert perpetual_inception() == pd.Timestamp(
        f"{policy['first_protocol_month']}-01", tz="UTC"
    )


def test_the_perpetual_policy_records_that_it_is_an_amendment_and_what_it_replaced():
    policy = PERPETUAL_KLINE_ARCHIVE_INCEPTION_POLICY
    assert policy["amendment"] == "A3"
    assert "before any P4 model fit" in policy["amendment_status"]
    assert "continuous" in policy["supersedes"]
    assert "A2 left this source explicitly untouched" in policy["supersedes"]
    assert "2019-12" in policy["adopted_because"]
    assert "404" in policy["adopted_because"] and "200" in policy["adopted_because"]


def test_a3_is_a_separate_object_from_a2_and_not_an_extension_of_it():
    """Two archives, measured separately, each with its own rule.

    They name the same month today. That must be an observation about two sources
    rather than one rule about a venue, so the objects stay distinct and moving
    one month moves exactly one plan.
    """
    a2 = FUNDING_ARCHIVE_INCEPTION_POLICY
    a3 = PERPETUAL_KLINE_ARCHIVE_INCEPTION_POLICY
    assert a2 is not a3 and dict(a2) != dict(a3)
    assert a2["amendment"] == "A2" and a3["amendment"] == "A3"
    assert a2["scope"]["field"] != a3["scope"]["field"]
    assert "perpetual_price" in a2["scope"]["does_not_apply_to"]
    assert "funding_rate" in a3["scope"]["does_not_apply_to"]
    assert "not an extension" in a3["relationship_to_a2"]
    # Independence, demonstrated: moving A3's month leaves A2's plan alone.
    import nn.p4_preregistration as prereg
    from nn.derivatives_sources import funding_inception, perpetual_inception

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            prereg,
            "PERPETUAL_KLINE_ARCHIVE_INCEPTION_POLICY",
            {**a3, "first_protocol_month": "2021-06"},
        )
        assert perpetual_inception() == pd.Timestamp("2021-06-01", tz="UTC")
        assert funding_inception() == pd.Timestamp("2020-01-01", tz="UTC")


def test_the_observed_perpetual_evidence_is_status_codes_and_not_a_conclusion():
    evidence = PERPETUAL_KLINE_ARCHIVE_INCEPTION_POLICY["observed_evidence"]
    months = {entry["month"]: entry for entry in evidence["months"]}
    for month in ("2019-09", "2019-10", "2019-11", "2019-12"):
        assert months[month]["status"] == 404 and months[month]["published"] is False
    for month in ("2020-01", "2020-02"):
        assert months[month]["status"] == 200 and months[month]["published"] is True
    # The first *required* missing archive is before the observed start of the
    # sequence, which is why A3 is a boundary and not a gap rule.
    assert max(m for m, e in months.items() if not e["published"]) < min(
        m for m, e in months.items() if e["published"]
    )
    assert evidence["first_observed_row"].startswith("2020-01-01T00:00:00")
    # The first row of the downloaded January archive, as an open-time in ms.
    assert evidence["first_observed_open_time_ms"] == 1577836800000
    assert pd.Timestamp(
        evidence["first_observed_open_time_ms"], unit="ms", tz="UTC"
    ) == pd.Timestamp("2020-01-01T00:00:00+00:00")
    # The layout observed is the one already allowed; A3 changes no column rule.
    assert evidence["observed_columns"] == len(KLINE_COLUMNS) == 12


def test_the_perpetual_claim_is_about_the_archive_and_not_about_the_market():
    """The generalisation A3 explicitly refuses to make."""
    limit = PERPETUAL_KLINE_ARCHIVE_INCEPTION_POLICY["generalisation_limit"]
    assert "six months were checked" in limit.lower()
    assert "monthly 1h kline archive" in limit
    assert "any kind" in limit


def test_a_pre_inception_perpetual_month_is_outside_the_source_and_not_a_gap():
    policy = PERPETUAL_KLINE_ARCHIVE_INCEPTION_POLICY
    assert "not an internal continuity gap" in policy["pre_inception_behaviour"].lower()
    assert "not counted as a missing month" in policy["pre_inception_behaviour"]
    assert "max(generic requested start" in policy["acquisition_start_rule"]


def test_nothing_may_stand_in_for_the_pre_inception_perpetual_region():
    substitution = PERPETUAL_KLINE_ARCHIVE_INCEPTION_POLICY["no_substitution"]
    assert substitution.startswith("never")
    for forbidden in ("synthetic", "REST", "interpolated", "backwards", "spot"):
        assert forbidden in substitution, forbidden
    # The spot leg is called out separately, because it is the one substitution
    # that would look plausible: it is already in the repository and already in
    # the basis formula, as the DENOMINATOR.
    spot = PERPETUAL_KLINE_ARCHIVE_INCEPTION_POLICY["no_spot_substitution"]
    assert "DENOMINATOR" in spot and "identically zero" in spot


def test_perpetual_continuity_after_inception_is_unchanged_and_fail_closed():
    after = PERPETUAL_KLINE_ARCHIVE_INCEPTION_POLICY["post_inception_continuity"]
    assert "mandatory" in after and "stops the acquisition" in after
    for forbidden in (
        "skipped",
        "interpolated",
        "forward-filled",
        "backward-filled",
        "replaced",
    ):
        assert forbidden in after, forbidden


def test_a3_changes_no_basis_feature_window_or_clip():
    """The scientific degrees of freedom A3 must leave exactly where they were."""
    windows = {feature["name"]: feature["window"] for feature in FEATURES}
    clips = {feature["name"]: feature["clip"] for feature in FEATURES}
    definitions = {feature["name"]: feature["definition"] for feature in FEATURES}
    assert windows["drv_basis"] is None
    assert windows["drv_basis_z"] == 168
    assert clips["drv_basis"] == [-0.02, 0.02]
    assert clips["drv_basis_z"] == [-5.0, 5.0]
    assert definitions["drv_basis"] == ("perpetual close at t over spot close at t, minus one")
    assert definitions["drv_basis_z"] == (
        "(drv_basis - mean of the last 168h) / (std of the same 168h + 1e-12)"
    )
    assert WARMUP_HOURS == 240
    assert TARGET["horizon"] == 6 and TARGET["timeframe"] == "1h"
    unchanged = PERPETUAL_KLINE_ARCHIVE_INCEPTION_POLICY["windows_unchanged"]
    assert "drv_basis" in unchanged and "168-hour z-score" in unchanged


def test_basis_features_are_available_only_where_their_perpetual_input_is():
    consequence = PERPETUAL_KLINE_ARCHIVE_INCEPTION_POLICY["downstream_consequence"]
    assert "only where their required perpetual-price input is actually present" in (
        consequence
    )
    assert "outside the common sample universe" in consequence
    assert "EVERY arm" in consequence
    assert "reported, not repaired" in consequence


def test_editing_the_perpetual_inception_policy_moves_both_hashes_together(monkeypatch):
    """`preregistration says 2020-01, acquisition asks for 2019-12` must be
    unconstructible — the anti-drift property A1 and A2 have, for A3's rule."""
    import nn.p4_preregistration as prereg
    from nn.derivatives_sources import source_spec
    from tools.export_derivatives_snapshot import source_spec_hash

    before_prereg = preregistration_hash()
    before_source = source_spec_hash()
    with monkeypatch.context() as patch:
        patch.setattr(
            prereg,
            "PERPETUAL_KLINE_ARCHIVE_INCEPTION_POLICY",
            {**PERPETUAL_KLINE_ARCHIVE_INCEPTION_POLICY, "first_protocol_month": "2021-06"},
        )
        assert prereg.preregistration_hash() != before_prereg
        assert source_spec_hash() != before_source
        assert source_spec()["perpetual_inception_rule"]["first_protocol_month"] == "2021-06"
    assert preregistration_hash() == before_prereg
    assert source_spec_hash() == before_source


def test_the_document_records_amendment_a3_and_all_three_superseded_hashes():
    text = DOCUMENT.read_text()
    assert "3.4c" in text
    assert "Source-protocol amendment A3" in text
    section = " ".join(text.split("### 3.4c", 1)[1].split("### 3.5", 1)[0].split())
    # The measured status codes, stated factually rather than summarised.
    for month in ("2019-09", "2019-10", "2019-11", "2019-12", "2020-01", "2020-02"):
        assert month in section, month
    assert "404" in section and "200" in section
    assert "2020-01-01T00:00:00Z" in section
    assert "12-field kline layout" in section
    # The active hash, and the three it replaced, all inside the amendment itself.
    assert f"sha256:{preregistration_hash()}" in section
    for superseded in (SUPERSEDED_HASH, SUPERSEDED_HASH_A1, SUPERSEDED_HASH_A2):
        assert superseded in section, superseded
        assert superseded != preregistration_hash()
    # And the amendment's own claims, in prose, matching the payload.
    assert "outside the published archive source and is not an internal continuity gap" in (
        section.lower()
    )
    assert "max(generic requested start, 2020-01)" in section
    assert "stops the acquisition" in section
    assert "BTCUSDT-1h-2019-12.zip" in section and "BTCUSDT-1h-2020-01.zip" in section
    assert "outside the common sample universe" in section
    # Ordering is preserved rather than rewritten: A1, then A2, then A3, and the
    # original protocol is not retold as if 2020-01 had always been known.
    assert text.index("### 3.4a") < text.index("### 3.4b") < text.index("### 3.4c")
    assert text.index("### 3.4c") < text.index("### 3.5 Fail-closed")
    assert "Amendment\nA3" in text.split("---", 1)[0]
    assert "the first version of this preregistration" not in section.lower()


def test_the_document_does_not_generalise_past_the_observed_perpetual_months():
    section = " ".join(
        DOCUMENT.read_text().split("### 3.4c", 1)[1].split("### 3.5", 1)[0].split()
    )
    assert "Six months were checked" in section
    assert "monthly 1h kline archive" in section
    assert (
        "Nothing here asserts that no BTCUSDT perpetual-price data of any kind existed"
        in section
    )
    # And it says why 2019-12 was asked for, rather than implying it was a mistake.
    assert "generic warm-up/source window begins" in section


def test_the_document_says_the_basis_definitions_do_not_change():
    section = " ".join(
        DOCUMENT.read_text().split("### 3.4c", 1)[1].split("### 3.5", 1)[0].split()
    )
    assert "The basis feature definitions do not change" in section
    assert "`[-0.02, 0.02]`" in section and "168-hour z-score" in section
    assert "the spot candle history does not stand in for the missing perpetual leg" in (
        section
    )


def test_the_stage_one_authorisation_carries_the_a3_hash_and_is_currently_authorised():
    """The later Stage 1 authorisation remains bound to the A3-updated active design."""
    path = ROOT / "data" / "research" / "p4_stage1_authorisation.json"
    text = path.read_text()
    payload = json.loads(text)
    assert payload["authorisation_schema"] == "chimera.p4-stage1-authorisation/1"
    assert payload["state"] == "authorised"
    assert payload["preregistration_hash"] == preregistration_hash()
    assert payload["authorised_by"] and payload["authorised_at"]
    assert payload["reason"]
    for superseded in (SUPERSEDED_HASH, SUPERSEDED_HASH_A1, SUPERSEDED_HASH_A2):
        assert superseded not in text, "only the active hash belongs in the interlock"


def test_no_p4_result_can_be_produced_under_the_pre_a3_hash(tmp_path):
    """An authorisation granted for the design A3 replaced authorises nothing."""
    from nn.p4_stage1 import Stage1Interlock, assert_fit_authorised

    research = tmp_path / "data" / "research"
    research.mkdir(parents=True)
    (research / "p4_stage1_authorisation.json").write_text(
        json.dumps(
            {
                "authorisation_schema": "chimera.p4-stage1-authorisation/1",
                "state": "authorised",
                "preregistration_hash": SUPERSEDED_HASH_A2,
                "authorised_by": "test",
                "authorised_at": "1970-01-01T00:00:00Z",
                "reason": "test",
            }
        )
    )
    with pytest.raises(Stage1Interlock, match="not permission to run another"):
        assert_fit_authorised(confirm=True, availability={"gate_passed": True}, root=tmp_path)


# --- amendment A4: which source rows are open-interest observations at all -----
def test_the_observation_validity_policy_is_inside_the_hashed_preregistration():
    """In the payload, not only in a module namespace beside it."""
    assert preregistration()["open_interest_observation_validity_policy"] == dict(
        OPEN_INTEREST_OBSERVATION_VALIDITY_POLICY
    )


def test_the_validity_policy_is_scoped_to_open_interest_and_names_both_metrics():
    policy = OPEN_INTEREST_OBSERVATION_VALIDITY_POLICY
    assert policy["scope"]["field"] == "open_interest"
    assert policy["scope"]["applies_to"] == ["open_interest"]
    assert sorted(policy["scope"]["does_not_apply_to"]) == ["funding_rate", "perpetual_price"]
    assert policy["scope"]["consumed_fields"] == [
        "sum_open_interest",
        "sum_open_interest_value",
    ]
    assert "DAILY metrics" in policy["scope"]["archive"]
    assert "sum_open_interest > 0" in policy["validity_rule"]
    assert "sum_open_interest_value > 0" in policy["validity_rule"]
    assert "if and only if" in policy["validity_rule"]
    assert set(policy["provenance_required"]) == {
        "logical_observations",
        "valid_positive_observations",
        "invalid_zero_observations",
        "invalid_both_zero_observations",
        "invalid_zero_contracts_only",
        "invalid_zero_notional_only",
        "negative_observations",
        "nonfinite_observations",
    }


def test_the_validity_policy_records_that_it_is_an_amendment_and_what_it_replaced():
    policy = OPEN_INTEREST_OBSERVATION_VALIDITY_POLICY
    assert policy["amendment"] == "A4"
    assert "before any P4 model fit" in policy["amendment_status"]
    assert "every metrics row surviving schema validation was an open-interest " in (
        policy["supersedes"]
    )
    assert "1722" in policy["adopted_because"]
    assert "open interest is non-positive on an available hour" in policy["adopted_because"]


def test_a4_runs_after_a1_and_does_not_replace_it():
    """Two rules about the same archive, in a fixed order, neither subsuming the other.

    A1 decides which *published rows* are one logical row; A4 decides which of
    those logical rows are *observations*. If A4 ran first it would discard zero
    rows before A1 could see that two of them disagreed, and A1's accounting would
    stop describing what the archive published.
    """
    a1, a4 = OPEN_INTEREST_DUPLICATE_POLICY, OPEN_INTEREST_OBSERVATION_VALIDITY_POLICY
    assert a1["amendment"] == "A1" and a4["amendment"] == "A4"
    assert a1 != a4
    assert "A1 exact-duplicate normalisation" in a4["applies_after"]
    assert "A1 still decides which published rows are one logical row" in a4["other_sources"]
    # A1 is untouched by A4: same scope, same key, same acceptance, same refusal.
    assert a1["grouping_key"] == "create_time"
    assert "reject the acquisition" in a1["on_conflict"]


def test_the_observed_evidence_is_counted_rows_and_not_a_conclusion():
    """What the scan found, as numbers, with no cause attached to any of them."""
    evidence = OPEN_INTEREST_OBSERVATION_VALIDITY_POLICY["observed_evidence"]
    run = evidence["max_consecutive_paired_zero_run"]
    assert run["observations"] == 102
    assert run["observation_spacing_minutes"] == 5
    assert run["minutes"] == 510 == run["observations"] * run["observation_spacing_minutes"]
    assert run["hours"] == 8.5

    one_sided = evidence["one_sided_zero_rows"]
    assert one_sided["found"] == 12
    assert one_sided["date"] == "2023-04-10"
    assert one_sided["all_on_one_date"] is True
    assert len(one_sided["instants_utc"]) == 12
    assert all(stamp.startswith("2023-04-10T") for stamp in one_sided["instants_utc"])

    assert evidence["scan_days"] == 1722
    assert evidence["scan_errors"] == 0
    assert evidence["negative_consumed_values_observed"] == 0
    assert "2021-05-22" in evidence["paired_zero_rows_exist"]
    assert "not isolated to one date" in evidence["paired_zero_rows_exist"]


def test_the_amendment_does_not_generalise_past_what_the_scan_measured():
    policy = OPEN_INTEREST_OBSERVATION_VALIDITY_POLICY
    limit = policy["generalisation_limit"]
    assert "that is all that is claimed" in limit
    assert "Nothing here asserts a cause" in limit
    # The two claims it must not make, checked as claims it does not make.
    assert "no claim is made" in policy["no_sentinel_claim"].lower()
    assert "No official source saying so" in policy["no_sentinel_claim"]
    assert "does not infer the true economic open interest" in policy["no_economic_inference"]


def test_a_zero_row_is_not_a_valid_zero_state_and_is_not_repaired():
    policy = OPEN_INTEREST_OBSERVATION_VALIDITY_POLICY
    assert "not a valid zero open-interest state" in policy["exactly_zero_is_invalid"]
    handling = policy["invalid_row_handling"]
    assert "remains a real published source row, counted in provenance and accounting" in (
        handling
    )
    for forbidden in ("interpolated", "averaged", "REST endpoint", "future observation"):
        assert any(forbidden in entry for entry in handling), forbidden
    assert any("does NOT update the hourly last-observation reducer" in e for e in handling)
    assert any("does NOT enter the causal OI observation sequence" in e for e in handling)


def test_a_negative_or_non_finite_value_is_a_hard_failure_and_not_a_zero_row():
    policy = OPEN_INTEREST_OBSERVATION_VALIDITY_POLICY
    assert "HARD FAIL" in policy["on_negative_or_nonfinite"]
    assert "never classified as an ordinary zero-invalid observation" in (
        policy["on_negative_or_nonfinite"]
    )
    assert "unparseable" in policy["on_negative_or_nonfinite"]


def test_a4_leaves_the_staleness_bound_and_the_missing_day_rule_alone():
    from nn.p4_preregistration import MAX_STALENESS_HOURS

    policy = OPEN_INTEREST_OBSERVATION_VALIDITY_POLICY
    assert MAX_STALENESS_HOURS["open_interest"] == 1
    assert "1 hour in MAX_STALENESS_HOURS" in policy["staleness_unchanged"]
    assert "UNAVAILABLE" in policy["staleness_unchanged"]
    assert "PROSPECTIVELY" in policy["later_valid_observation"]
    assert "never applied backwards" in policy["later_valid_observation"]
    assert "NOT thereby a §3.0a missing day" in policy["missing_day_relationship"]
    assert "404" in policy["missing_day_relationship"]


def test_a4_changes_no_feature_window_clip_or_gate():
    policy = OPEN_INTEREST_OBSERVATION_VALIDITY_POLICY
    unchanged = policy["windows_unchanged"]
    for name in ("drv_oi_log_change_24h", "drv_oi_notional_ratio", "drv_oi_price_divergence"):
        assert name in unchanged
    assert "No epsilon is added to open interest" in unchanged
    assert "no zero is replaced by a tiny positive value" in unchanged
    for word in ("window", "clip", "target", "horizon", "cost model", "gate"):
        assert word in unchanged, word
    # And the values themselves are the ones the feature spec already carried.
    assert {f["name"]: f["window"] for f in FEATURES if f["family"] == "open_interest"} == {
        "drv_oi_log_change_24h": 24,
        "drv_oi_notional_ratio": 168,
        "drv_oi_price_divergence": 24,
    }


def test_the_accounting_identity_is_stated_rather_than_left_to_the_reader():
    identity = OPEN_INTEREST_OBSERVATION_VALIDITY_POLICY["accounting_identity"]
    assert "logical_observations (after A1 duplicate collapse) ==" in identity
    assert "valid_positive_observations + invalid_zero_observations" in identity
    assert "disjoint" in identity


def test_editing_the_validity_policy_moves_both_hashes_together(monkeypatch):
    """The anti-drift property A1, A2 and A3 have, for A4's rule.

    `the preregistration says both metrics must be positive and the acquisition
    admits a zero` must not be a constructible checkout.
    """
    import nn.p4_preregistration as prereg
    from nn.derivatives_sources import source_spec
    from tools.export_derivatives_snapshot import source_spec_hash

    before_prereg = preregistration_hash()
    before_source = source_spec_hash()
    with monkeypatch.context() as patch:
        patch.setattr(
            prereg,
            "OPEN_INTEREST_OBSERVATION_VALIDITY_POLICY",
            {
                **OPEN_INTEREST_OBSERVATION_VALIDITY_POLICY,
                "validity_rule": "any published row is an observation",
            },
        )
        assert prereg.preregistration_hash() != before_prereg
        assert source_spec_hash() != before_source
        assert (
            source_spec()["open_interest_validity_rule"]["validity_rule"]
            == "any published row is an observation"
        )
    assert preregistration_hash() == before_prereg
    assert source_spec_hash() == before_source


def test_the_acquisition_reads_the_policy_rather_than_restating_it():
    """One authoritative copy, checked as identity rather than as agreement."""
    from nn.derivatives_sources import source_spec

    assert source_spec()["open_interest_validity_rule"] == dict(
        OPEN_INTEREST_OBSERVATION_VALIDITY_POLICY
    )


def test_the_document_records_amendment_a4_and_all_four_superseded_hashes():
    text = DOCUMENT.read_text()
    assert "3.4d" in text
    assert "Source-protocol amendment A4" in text
    section = " ".join(text.split("### 3.4d", 1)[1].split("### 3.5", 1)[0].split())
    # The measured facts, stated as facts rather than summarised into a cause.
    assert "1722" in section
    assert "open interest is non-positive on an available hour" in section
    assert "102" in section and "510 minutes" in section and "8.5 hours" in section
    assert "2021-05-22" in section
    assert "exactly 12" in section and "2023-04-10" in section
    assert "zero scan errors" in section
    assert "no negative" in section
    # The active hash, and the four it replaced, all inside the amendment itself.
    assert f"sha256:{preregistration_hash()}" in section
    for superseded in (
        SUPERSEDED_HASH,
        SUPERSEDED_HASH_A1,
        SUPERSEDED_HASH_A2,
        SUPERSEDED_HASH_A3,
    ):
        assert superseded in section, superseded
        assert superseded != preregistration_hash()
    # And the amendment's own claims, in prose, matching the payload.
    assert "if and only if" in section
    assert "HARD FAILURE" in section
    assert "prospectively only" in section
    assert "outside\nthe common sample universe".replace("\n", " ") in section
    assert "does not infer the economic open interest" in section.lower()
    assert "no official source saying so is known to this repository" in section.lower()
    # Ordering is preserved rather than rewritten: A1, A2, A3, then A4.
    assert (
        text.index("### 3.4a")
        < text.index("### 3.4b")
        < text.index("### 3.4c")
        < text.index("### 3.4d")
        < text.index("### 3.5 Fail-closed")
    )
    assert "Amendment A4" in text.split("---", 1)[0]


def test_the_document_says_the_open_interest_definitions_do_not_change():
    section = " ".join(
        DOCUMENT.read_text().split("### 3.4d", 1)[1].split("### 3.5", 1)[0].split()
    )
    assert "The open-interest feature definitions do not change" in section
    assert "`[-1.0, 1.0]`" in section and "`[0.0, 10.0]`" in section
    assert "No epsilon is added to open" in section
    assert "missing-day rule is unamended" in section


def test_the_stage_one_authorisation_carries_the_a4_hash_and_is_currently_authorised():
    """The later Stage 1 authorisation remains bound to the A4-updated active design."""
    path = ROOT / "data" / "research" / "p4_stage1_authorisation.json"
    text = path.read_text()
    payload = json.loads(text)
    assert payload["authorisation_schema"] == "chimera.p4-stage1-authorisation/1"
    assert payload["state"] == "authorised"
    assert payload["preregistration_hash"] == preregistration_hash()
    assert payload["authorised_by"] and payload["authorised_at"]
    assert payload["reason"]
    for superseded in (
        SUPERSEDED_HASH,
        SUPERSEDED_HASH_A1,
        SUPERSEDED_HASH_A2,
        SUPERSEDED_HASH_A3,
    ):
        assert superseded not in text, "only the active hash belongs in the interlock"


def test_no_p4_result_can_be_produced_under_the_pre_a4_hash(tmp_path):
    """An authorisation granted for the design A4 replaced authorises nothing."""
    from nn.p4_stage1 import Stage1Interlock, assert_fit_authorised

    research = tmp_path / "data" / "research"
    research.mkdir(parents=True)
    (research / "p4_stage1_authorisation.json").write_text(
        json.dumps(
            {
                "authorisation_schema": "chimera.p4-stage1-authorisation/1",
                "state": "authorised",
                "preregistration_hash": SUPERSEDED_HASH_A3,
                "authorised_by": "test",
                "authorised_at": "1970-01-01T00:00:00Z",
                "reason": "test",
            }
        )
    )
    with pytest.raises(Stage1Interlock, match="not permission to run another"):
        assert_fit_authorised(confirm=True, availability={"gate_passed": True}, root=tmp_path)
