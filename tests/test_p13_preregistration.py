"""P13's design, pinned — and the tripwire proving it was committed before a result.

Two jobs. The first is the tripwire: assert that no P13 economic result artifact
exists, so a preregistration commit that silently carried one would fail the
suite rather than being taken on trust. The guarantee "the design was frozen
before the answer" is only worth what the check behind it is worth, and P6's
dirty-tree provenance is this repository's own evidence of what happens when the
check is prose.

The second is pinning. Every constant a P13 verdict depends on is asserted here
by value, so moving one after a result is a test failure and not an edit.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nn import p13_preregistration as prereg

ROOT = Path(__file__).resolve().parents[1]

#: The frozen design. Recomputed by the module and compared here; changing any
#: decision-relevant constant moves it, which is the point.
EXPECTED_HASH = prereg.preregistration_hash()


# ---------------------------------------------------------------------------
# The tripwire
# ---------------------------------------------------------------------------


def test_no_p13_result_artifact_exists():
    """The checkpoint's evidence aggregate must be absent at preregistration time."""
    decision = ROOT / "artifacts" / "benchmark" / "btc_p13_decision" / "decision.json"
    assert not decision.exists(), (
        f"{decision} exists, so this is no longer a preregistration-before-result commit. "
        "A design frozen after its answer is not a preregistration."
    )


def test_the_declared_result_state_says_nothing_has_been_run():
    assert prereg.CURRENT_RESULT_STATE.endswith("NOT YET RUN")
    assert prereg.CURRENT_RESULT_STATE not in prereg.RESULT_STATES


def test_the_hash_is_stable_across_calls():
    assert prereg.preregistration_hash() == EXPECTED_HASH
    assert EXPECTED_HASH.startswith("sha256:")
    assert len(EXPECTED_HASH) == len("sha256:") + 64


def test_describe_carries_the_hash_and_the_whole_payload():
    described = prereg.describe()
    assert described["preregistration_hash"] == EXPECTED_HASH
    for key in prereg.payload():
        assert key in described


# ---------------------------------------------------------------------------
# Boundaries and prohibitions
# ---------------------------------------------------------------------------


def test_the_research_boundary_is_the_retired_p4_hold_instant():
    assert prereg.RESEARCH_BOUNDARY_EXCLUSIVE == "2025-05-19T08:00:00+00:00"
    assert prereg.DATA_BOUNDARY["span_end_exclusive"] == "2025-05-19T08:00:00+00:00"


def test_the_boundary_is_stricter_than_the_styx_seal():
    """P4-HOLD begins before Styx, so it is the binding rule, not Styx."""
    assert prereg.STYX_SEALED_INSTANT == "2025-08-27T23:00:00+00:00"
    assert prereg.RESEARCH_BOUNDARY_EXCLUSIVE < prereg.STYX_SEALED_INSTANT


def test_the_span_starts_where_the_sources_actually_begin():
    assert prereg.DATA_BOUNDARY["span_start_inclusive"] == "2020-01-01T00:00:00+00:00"


def test_p4_hold_and_styx_are_declared_unread():
    assert "NOT READ" in prereg.DATA_BOUNDARY["p4_hold"]
    assert "NOT READ" in prereg.DATA_BOUNDARY["styx"]


@pytest.mark.parametrize(
    "phrase",
    [
        "no real money",
        "no leverage above 1x",
        "no P4-HOLD read",
        "no Styx read",
        "no manufactured historical holdout",
        "Aegis remains the central risk authority",
    ],
)
def test_the_safety_prohibitions_are_present(phrase):
    assert phrase in prereg.SAFETY_PROHIBITIONS


def test_leverage_is_exactly_one_and_never_more():
    assert prereg.CAPITAL_CONTRACT["leverage"].startswith("exactly 1x")
    assert "1x gross in every model" in prereg.MARGIN_AND_LIQUIDATION["leverage"]


# ---------------------------------------------------------------------------
# The capital contract
# ---------------------------------------------------------------------------


def test_the_capital_denominator_is_frozen_and_is_a_real_scale():
    contract = prereg.CAPITAL_CONTRACT
    assert contract["total_starting_capital"] == "1000000"
    assert contract["capital_units"] == "USDT"
    assert contract["spot_allocation"] == "500000"
    assert contract["perp_margin_allocation"] == "500000"


def test_the_allocations_sum_to_the_total_capital():
    contract = prereg.CAPITAL_CONTRACT
    total = int(contract["total_starting_capital"])
    assert int(contract["spot_allocation"]) + int(contract["perp_margin_allocation"]) == total


def test_the_return_denominator_is_both_legs_not_one():
    denominator = prereg.CAPITAL_CONTRACT["return_denominator"]
    assert "TOTAL committed capital" in denominator
    assert "Never one leg" in denominator


def test_the_quantity_rule_is_the_minimum_over_both_legs():
    """Sizing from the spot allocation alone refused to open above ~5 bps basis."""
    rule = prereg.CAPITAL_CONTRACT["quantity_rule"]
    assert "min(Q_spot_bound, Q_perp_bound)" in rule
    assert "COARSER" in rule


# ---------------------------------------------------------------------------
# Costs
# ---------------------------------------------------------------------------


def test_the_cost_rates_are_the_frozen_venue_taker_rates():
    costs = prereg.COST_MODEL
    assert costs["spot_entry_fee_rate"] == "0.001"
    assert costs["spot_exit_fee_rate"] == "0.001"
    assert costs["perp_entry_fee_rate"] == "0.0005"
    assert costs["perp_exit_fee_rate"] == "0.0005"
    assert costs["spot_slippage_rate"] == "0.0005"
    assert costs["perp_slippage_rate"] == "0.0005"


def test_fees_are_charged_on_notional_never_on_quantity():
    assert "NOTIONAL" in prereg.COST_MODEL["fee_basis"]
    assert "Never quantity alone" in prereg.COST_MODEL["fee_basis"]


def test_costs_may_never_be_lowered_after_a_result():
    assert "never_lowered" in prereg.COST_MODEL
    assert "moving any cost, fee, slippage or friction" in prereg.FORBIDDEN_AFTER_RESULTS


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def test_every_gate_threshold_is_pinned():
    assert prereg.BREADTH_REQUIRED == 4
    assert prereg.BREADTH_OF == 6
    assert prereg.MIN_SETTLEMENTS_PER_BLOCK == 200
    assert prereg.MIN_INCLUDED_BLOCKS == 5
    assert prereg.WORST_BLOCK_FLOOR == "-0.02"
    assert prereg.MIN_MEAN_NET_RETURN == "0.0025"


def test_the_gate_is_a_conjunction_of_all_six_conditions():
    gate = prereg.VIABILITY_GATE
    for condition in (
        "G1_breadth",
        "G2_central_tendency",
        "G3_downside",
        "G4_sample",
        "G5_stress",
        "G6_minimum_effect_size",
    ):
        assert condition in gate["conditions"], f"{condition} is missing from the gate"
    assert gate["conjunction"].startswith("ALL of G1, G2, G3, G4, G5 and G6")


def test_the_gate_demands_a_minimum_effect_size_not_merely_a_positive_mean():
    """Without G6 the gate could emit VIABLE on a few basis points a year."""
    assert (
        prereg.MIN_MEAN_NET_RETURN
        == prereg.COST_MODEL["per_block_round_trip_cost_of_total_capital"].split()[1]
    )


def test_the_breadth_requirement_is_not_a_copy_of_p6s_three_of_four():
    assert (prereg.BREADTH_REQUIRED, prereg.BREADTH_OF) != (3, 4)


def test_the_operating_characteristic_is_disclosed():
    """A gate that admits ~34% of coin flips on G1 alone must say so."""
    assert "34%" in prereg.VIABILITY_GATE["operating_characteristic"]
    assert "does not create" in prereg.VIABILITY_GATE["operating_characteristic"]


def test_ties_are_resolved_explicitly_in_every_direction():
    ties = prereg.VIABILITY_GATE["tie_handling"]
    assert ties["block_return_exactly_zero"].startswith("NOT positive")
    assert ties["mean_exactly_zero"].startswith("FAILS")
    assert ties["mean_exactly_0.0025"].startswith("FAILS")
    assert ties["worst_block_exactly_minus_0.02"].startswith("PASSES")
    assert ties["settlements_exactly_200"].startswith("PASSES")


# ---------------------------------------------------------------------------
# Causality, execution and stresses
# ---------------------------------------------------------------------------


def test_the_fill_price_is_the_candle_open_not_the_close():
    """Filling at the close of a candle labelled t is a one-hour lookahead."""
    policy = prereg.EXECUTION_PRICE_POLICY
    assert "OPEN of the candle" in policy["fill_price"]
    assert "never an execution price" in policy["close_is_never_a_fill"]


def test_the_predicted_funding_rate_is_never_read():
    assert "NEVER READ" in prereg.FUNDING_CAUSALITY["predicted_rate"]


def test_the_strategy_takes_no_funding_signal():
    assert prereg.FUNDING_CAUSALITY["does_the_strategy_use_funding_as_a_signal"].startswith(
        "NO"
    )


def test_the_remaining_hindsight_parameter_is_disclosed():
    """The leg direction fixes the sign of the funding payoff and is era-informed."""
    disclosure = prereg.FUNDING_CAUSALITY["the_one_parameter_that_remains"]
    assert "LEG DIRECTION" in disclosure
    assert "hindsight-informed" in disclosure


def test_the_settlement_boundary_tie_rule_is_explicit():
    rule = prereg.FUNDING_SEMANTICS["boundary_tie_rule"]
    assert "open_instant < settlement_instant <= close_instant" in rule


def test_the_exit_rule_is_causally_implementable():
    close = prereg.POSITION_LIFECYCLE["close_instant"]
    assert "FIRST valid instant AT OR AFTER" in close


def test_the_basis_identity_is_scoped_away_from_the_delayed_hedge_stress():
    assert "does NOT hold under S2" in prereg.BASIS_DEFINITION["identity_scope"]


def test_all_five_stress_cases_are_declared():
    assert [s["id"] for s in prereg.STRESS_CASES] == ["S0", "S1", "S2", "S3", "S4"]


def test_the_gated_stresses_include_the_basis_stress():
    """S1 alone is the stress this construction is least sensitive to."""
    assert "S3" in prereg.VIABILITY_GATE["conditions"]["G5_stress"]
    s3 = next(s for s in prereg.STRESS_CASES if s["id"] == "S3")
    assert "IN THE GATE" in s3["role"]


def test_the_isolated_margin_case_is_measured_as_a_stress_not_assumed_away():
    s4 = next(s for s in prereg.STRESS_CASES if s["id"] == "S4")
    assert "isolated" in s4["name"]
    assert "OUTSIDE the viability gate" in s4["role"]


def test_the_margin_model_choice_is_disclosed_as_data_informed():
    disclosure = prereg.MARGIN_AND_LIQUIDATION["disclosure_of_how_this_choice_was_made"]
    assert "HONESTLY DISCLOSED" in disclosure
    assert "2020, 2021, 2023 and 2024" in disclosure


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


def test_all_four_sources_are_binance_and_named():
    fields = {s["field"] for s in prereg.DATA_SOURCES}
    assert fields == {"spot_price", "perpetual_price", "funding_settlement", "mark_price"}
    for source in prereg.DATA_SOURCES:
        assert source["venue"] == "binance"
        assert source["symbol"] == "BTCUSDT"


def test_the_source_facts_are_pinned_to_first_party_binance_evidence():
    evidence = prereg.FIRST_PARTY_SOURCE_EVIDENCE
    assert evidence["repository"] == "https://github.com/binance/binance-public-data"
    assert len(evidence["commit"]) == 40
    assert "sha256" in evidence["checksum"]["algorithm"]


def test_the_archives_are_not_claimed_immutable():
    """Binance publishes a changelog of archive revisions, including a kline one."""
    assert prereg.ARCHIVE_REVISION_POLICY["archives_may_be_revised"] is True
    assert "2022-08-08" in prereg.ARCHIVE_REVISION_POLICY["known_revisions_in_scope"]


def test_the_spot_timestamp_unit_change_is_handled_rather_than_assumed():
    policy = prereg.TIMESTAMP_UNIT_POLICY
    assert "microseconds" in policy["spot_unit_change"]
    assert "resolve_epoch_unit" in policy["rule"]
    assert policy["fail_closed"].startswith("an archive whose unit cannot be resolved")


def test_the_mark_price_fallback_is_triggered_by_availability_only():
    fallback = prereg.MARK_PRICE_FALLBACK
    # The trigger is a publication fact about the archive, not anything a run computes.
    assert "unpublished" in fallback["trigger"]
    assert "acquisition probe" in fallback["trigger"]
    assert "Availability only" in fallback["never_triggered_by"]
    assert "any economic observation" in fallback["never_triggered_by"]
    assert "NEVER used" in fallback["forbidden_alternative"]


# ---------------------------------------------------------------------------
# Scope of the eventual claim
# ---------------------------------------------------------------------------


def test_the_result_labels_name_the_construction_they_screen():
    for state in prereg.RESULT_STATES:
        assert state.startswith("P13 ALWAYS-ON ANNUAL SPOT/PERP CARRY:")


def test_the_scope_of_a_negative_is_stated_before_the_result_exists():
    scope = prereg.WHAT_A_NEGATIVE_WOULD_AND_WOULD_NOT_MEAN
    assert "would NOT say that funding/basis carry is" in scope


def test_a_positive_result_authorises_nothing():
    on_viable = prereg.STOPPING_RULE["on_viable"]
    assert "STOP the alpha search" in on_viable
    assert "no real money" in on_viable.lower() or "authorises no real money" in on_viable


def test_the_evidence_ceiling_is_exploratory_and_the_hindsight_era_is_disclosed():
    assert prereg.EVIDENCE_CEILING.startswith("EXPLORATORY / ADAPTIVE HISTORICAL")
    assert "not blind to the era" in prereg.HINDSIGHT_DISCLOSURE


def test_a_dirty_source_tree_is_refused_for_primary_generation():
    assert "clean, committed source tree" in prereg.PROVENANCE_REQUIREMENT
    assert "P6" in prereg.PROVENANCE_REQUIREMENT


# ---------------------------------------------------------------------------
# Repairs from the adversarial design critique, pinned so they cannot regress
# ---------------------------------------------------------------------------


def test_only_one_margin_model_is_declared_primary():
    """Two hashed dictionaries once froze incompatible margin models."""
    assert prereg.CAPITAL_CONTRACT["margin_mode"].startswith("PORTFOLIO (cross)")
    assert prereg.MARGIN_AND_LIQUIDATION["primary_model"].startswith("PORTFOLIO (cross)")
    assert "isolated=False" in prereg.CAPITAL_CONTRACT["evaluator_flag"]


def test_the_funding_rate_unit_is_declared_and_fails_closed():
    """The one rate the payoff is made of had no unit statement."""
    semantics = prereg.FUNDING_SEMANTICS
    assert "DECIMAL FRACTION" in semantics["rate_unit"]
    assert "NOT a percent" in semantics["rate_unit"]
    assert "NEVER multiplied by" in semantics["rate_unit"]
    assert "0.01" in semantics["unit_fail_closed"]
    assert "REFUSED" in semantics["unit_fail_closed"]


def test_the_block_return_numerator_includes_entry_frictions():
    """equity_close - entry_equity would hide ~12.5 bps and flip G1's sign test."""
    numerator = prereg.VIABILITY_GATE["block_net_pnl"]
    assert "MINUS total_starting_capital" in numerator
    assert "INSIDE the numerator" in numerator


def test_the_horizon_disparity_between_blocks_is_declared():
    convention = prereg.VIABILITY_GATE["horizon_convention"]
    assert "NOT annualised" in convention
    assert "4.6 months" in convention


def test_the_venue_constants_are_frozen_in_the_hashed_design():
    """mmr, step size and min notional sit on the decision path."""
    constraints = prereg.VENUE_CONSTRAINTS
    assert constraints["perpetual"]["step_size"] == "0.001"
    assert constraints["perpetual"]["min_notional"] == "100"
    assert constraints["perpetual"]["tier_1_maintenance_margin_rate"] == "0.004"
    assert constraints["effective_step_size"].startswith("0.001")
    assert "venue_constraints" in prereg.payload()


def test_the_current_era_venue_table_limitation_is_stated():
    assert "CURRENT-ERA" in prereg.VENUE_CONSTRAINTS["era_limitation"]


def test_the_margin_disclosure_claims_only_what_the_check_established():
    """The price check shows liquidation occurs; it does not show the sign of returns."""
    disclosure = prereg.MARGIN_AND_LIQUIDATION["disclosure_of_how_this_choice_was_made"]
    assert "does NOT establish the SIGN" in disclosure
    assert "declines to predict" in disclosure


def test_the_margin_sufficiency_rule_is_an_assertion_not_a_live_branch():
    rule = prereg.CAPITAL_CONTRACT["margin_sufficiency_rule"]
    assert rule.startswith("an ASSERTION, not a branch")
    # ...and the gate no longer names it as a way a block gets excluded.
    assert "margin sufficiency" not in prereg.VIABILITY_GATE["excluded_blocks"]


def test_each_price_series_has_a_declared_role():
    roles = prereg.BASIS_DEFINITION["which_series_plays_which_role"]
    assert "MARK-TO-MARKET of each leg uses the CLOSE" in roles
    assert "funding notional and the liquidation test" in roles


def test_the_zero_delta_claims_are_scoped_away_from_the_delayed_hedge():
    assert "under S0, S1 and S3" in prereg.CAPITAL_CONTRACT["residual_delta"]
    assert "under S0, S1" in prereg.CONSTRUCTION


def test_the_isolated_stress_has_per_block_reporting_fields():
    fields = prereg.TEMPORAL_PARTITION["per_block_report_fields"]
    assert "S4 isolated net return" in fields
    assert "S4 isolated liquidation flag" in fields


def test_primary_evidence_is_event_level_and_separately_manifested():
    """A totals JSON cannot be audited, only believed."""
    policy = prereg.ARTIFACT_POLICY
    assert "one row per event" in policy["primary_evidence"]
    assert "RECOMPUTABLE" in policy["primary_evidence"]
    assert "its own SHA-256 manifest" in policy["primary_evidence_has_its_own_manifest"]
    assert "not_frozen_only_as_a_summary" in policy


def test_the_decision_aggregate_path_is_the_one_the_tripwire_watches():
    assert prereg.ARTIFACT_POLICY["decision_aggregate"] == (
        "artifacts/benchmark/btc_p13_decision/decision.json"
    )
    assert prereg.ARTIFACT_POLICY["decision_aggregate"] in prereg.TRIPWIRE


def test_the_portfolio_models_operational_assumption_is_stated():
    """Plain USD-M cross margin does not collateralise from spot BTC holdings."""
    stated = prereg.MARGIN_AND_LIQUIDATION["what_the_portfolio_model_assumes_operationally"]
    assert "Portfolio Margin" in stated
    assert "does NOT automatically collateralise from spot" in stated


# ---------------------------------------------------------------------------
# Repairs from the falsification and governance lenses
# ---------------------------------------------------------------------------


def test_the_sealed_instant_is_resolved_from_the_contract_not_restated():
    """tests/test_research_contracts.py forbids a second literal copy under nn/."""
    from nn.research_contract import load_contract

    assert prereg.STYX_SEALED_INSTANT == (
        load_contract("btc-usdt-1h-gen1").sealed_test_start.isoformat()
    )
    source = (ROOT / "nn" / "p13_preregistration.py").read_text()
    assert "2025-08-27T23:00:00" not in source


def test_the_committed_spot_snapshot_is_described_as_pre_styx_not_pre_boundary():
    """It carries ~2,415 rows of the retired P4-HOLD region."""
    spot = next(s for s in prereg.DATA_SOURCES if s["field"] == "spot_price")
    stated = spot["committed_alternative_is_pre_STYX_not_pre_boundary"]
    assert "2025-08-27T22:00:00+00:00" in stated
    assert "P4-HOLD" in stated
    assert "TRUNCATING READ" in spot["truncating_read_carve_out"]


def test_the_truncating_carve_out_does_not_extend_to_freshly_acquired_archives():
    carve = next(s for s in prereg.DATA_SOURCES if s["field"] == "spot_price")[
        "truncating_read_carve_out"
    ]
    assert "freshly acquired archive is still refused" in carve


def test_the_isolated_stress_debits_funding_from_the_isolated_balance():
    """Routing S4's funding to free cash makes the strict case systematically lenient."""
    s4 = next(s for s in prereg.STRESS_CASES if s["id"] == "S4")
    assert "ISOLATED MARGIN BALANCE" in s4["definition"]
    assert "cumulative_perp_funding" in s4["definition"]


def test_the_gated_stresses_are_named_consistently_everywhere():
    s1 = next(s for s in prereg.STRESS_CASES if s["id"] == "S1")
    s3 = next(s for s in prereg.STRESS_CASES if s["id"] == "S3")
    assert "IN THE GATE" in s1["role"] and "IN THE GATE" in s3["role"]
    assert "Only S1 and S3" in prereg.STRESS_DISCIPLINE
    for s in prereg.STRESS_CASES:
        if s["id"] in ("S2", "S4"):
            assert "outside the gate" in s["role"] or "OUTSIDE the viability gate" in s["role"]


def test_the_adverse_basis_stress_has_an_explicit_direction():
    s3 = next(s for s in prereg.STRESS_CASES if s["id"] == "S3")
    assert "REDUCED by 10 bps" in s3["definition"]
    assert "INCREASED by 10 bps" in s3["definition"]


def test_the_delayed_hedge_stress_is_two_sided():
    """A one-sided delay is a benefit in a rising sample, not a stress."""
    s2 = next(s for s in prereg.STRESS_CASES if s["id"] == "S2")
    assert "BOTH WAYS" in s2["definition"]
    assert "WORSE of the two" in s2["definition"]


def test_the_payoff_side_diagnostics_are_predeclared_and_never_gate():
    ids = [d["id"] for d in prereg.PAYOFF_SIDE_DIAGNOSTICS]
    assert ids == ["D1", "D2", "D3"]
    assert "never gated on" in prereg.PAYOFF_DIAGNOSTIC_DISCIPLINE
    # ...and none of them appears as a gate condition.
    for condition in prereg.VIABILITY_GATE["conditions"].values():
        for diagnostic in ids:
            assert f"{diagnostic} " not in condition


def test_the_mark_price_fallback_is_per_object_not_all_or_nothing():
    assert "PER ARCHIVE OBJECT" in prereg.MARK_PRICE_FALLBACK["trigger"]
    assert "per block" in prereg.MARK_PRICE_FALLBACK["reporting_granularity"]


def test_the_partition_does_not_claim_its_interior_boundaries_were_unchosen():
    stated = prereg.TEMPORAL_PARTITION["what_was_and_was_not_chosen"]
    assert "ARE a choice" in stated
    assert "would be false" in stated


def test_the_maintenance_margin_approximation_and_its_direction_are_disclosed():
    stated = prereg.MARGIN_AND_LIQUIDATION["maintenance_margin_rate_is_an_approximation"]
    assert "ABOVE Binance's tier-1" in stated
    assert "LATER liquidation" in stated


def test_the_primary_liquidation_test_does_not_claim_to_call_liquidation_price():
    check = prereg.MARGIN_AND_LIQUIDATION["liquidation_check"]
    assert "does NOT call liquidation_price" in check
    assert "FOLLOWING bar" in prereg.MARGIN_AND_LIQUIDATION["forced_close_price"]


def test_the_exit_search_is_bounded_by_the_block_end():
    assert "bounded by the BLOCK END" in prereg.POSITION_LIFECYCLE["close_instant"]


def test_the_gate_states_that_its_hurdle_is_zero_yield_cash():
    assert "ZERO" in prereg.VIABILITY_GATE["the_hurdle_is_zero_yield_cash"]
    assert (
        "not an economic hurdle rate" in prereg.VIABILITY_GATE["the_hurdle_is_zero_yield_cash"]
    )


def test_the_human_document_quotes_the_hash_the_module_computes():
    """The prose half and the machine half must not drift apart silently."""
    doc = (ROOT / "docs" / "p13_preregistration.md").read_text()
    assert EXPECTED_HASH in doc, (
        "docs/p13_preregistration.md does not carry the current preregistration hash. "
        "Regenerate it rather than editing the module and leaving the prose behind."
    )


def test_the_boundary_straddling_month_has_a_stated_rule():
    """The boundary falls mid-month, so the last archive month spans it."""
    rule = prereg.DATA_BOUNDARY["the_boundary_straddling_month"]
    assert "TRUNCATED AT LOAD" in rule
    assert "WHOLE published object" in rule
    assert "exactly two things" in rule
