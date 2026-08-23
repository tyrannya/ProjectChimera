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


def test_no_p4_fit_is_authorised():
    """The interlock, in the state it ships in and must stay in until a commit."""
    from nn.p4_stage1 import Stage1Interlock, assert_fit_authorised, read_authorisation

    assert read_authorisation()["state"] == "not_authorised"
    with pytest.raises(Stage1Interlock):
        assert_fit_authorised(confirm=True, availability={"gate_passed": True})


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
    # adds a third P4 file, and the session that acquires it has to come here
    # and say so — which is the point: "P4 has data now" should be a diff
    # somebody wrote, not a directory listing that quietly changed.
    p4_files = sorted(path.name for path in research.glob("*p4*"))
    assert p4_files == ["p4_holdout_ledger.json", "p4_stage1_authorisation.json"]
    authorisation = json.loads((research / "p4_stage1_authorisation.json").read_text())
    assert authorisation["state"] == "not_authorised"
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
