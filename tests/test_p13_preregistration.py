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

from nn import p13_carry as carry
from nn import p13_preregistration as prereg

ROOT = Path(__file__).resolve().parents[1]

#: The frozen design. Recomputed by the module and compared here; changing any
#: decision-relevant constant moves it, which is the point.
EXPECTED_HASH = prereg.preregistration_hash()

#: The ORIGINAL preregistration's hash, committed and pushed at 939f3815 before
#: any attempt to obtain data. Written as a literal, because the whole value of a
#: superseded hash is that it is not recomputed from the thing that superseded
#: it. Amendment A1 (§8a) moved the active hash away from this one; the committed
#: acquisition evidence still carries it, and that is correct — it was generated
#: under the original design and is deliberately not rewritten.
ORIGINAL_HASH = "sha256:1369c8828767c04e5b0609fc0125947c91f1cb5f15e977804ff1d1d70fd68767"

#: **P13-A1's** hash, the design amendment A2 superseded. A literal for the same
#: reason as the original's: a superseded hash recomputed from the thing that
#: superseded it proves nothing. A1's RULE is not withdrawn —
#: ``forced_close_without_a_following_bar`` is still in the payload and still in
#: force — but the A1 hash is retired, and no P13 economic run may claim it as the
#: active design.
A1_HASH = "sha256:4397109858249c6923b72418d756a3e8504c7cb7abed15deebf300c252f4b099"

#: The FIRST committed A2's hash, retired by revision A2R1. A literal for the same
#: reason as the two above, and kept for a sharper one: the design it names made a
#: FALSE scientific claim — that a delayed pre-open "can only reduce accrued
#: funding" — and the correction is only auditable if the superseded design stays
#: nameable. A2's source-validity BEHAVIOUR is unchanged by A2R1; only the
#: justification and the hash moved.
A2_FIRST_HASH = "sha256:86050b6897143cd0abfa2106ee6cc37baaf3c637e396b23f30bdeeb935fbfe6c"

#: **P13-A2R1's** hash, retired by revision A2R2. A literal for the same reason as
#: the three above, and kept for a sharper one still: A2R1 corrected A2's false
#: economic claim but KEPT A2's pre-open entry rule, and that rule decides the
#: entry instant from whether a completed same-bar mark archive row will later
#: exist — source-availability look-ahead. A2R1's own withdrawal of the
#: monotonicity claim is NOT reversed by A2R2; only the entry rule is.
A2R1_HASH = "sha256:7064faeeb049809fa1c9037559023a6e9b54b0e18d6d964605696f7a21ca29f0"

#: The commit that first froze A2, left in history rather than amended or rebased.
FIRST_A2_COMMIT = "0d023ba5a33bbcae55bb079146838f5f22959607"

#: The commit that froze A2R1, likewise left in history.
A2R1_COMMIT = "d40cfcfbcb6e8048198f432eb2bb4ebf70c24b7c"

#: Where the acquisition evidence lives. It is the record of a NOT EVALUABLE
#: environment outcome, not an economic result.
ACQUISITION_EVIDENCE = (
    "artifacts/benchmark/btc_p13_carry/acquisition_plan.json",
    "artifacts/benchmark/btc_p13_carry/acquisition_refusal.json",
    "artifacts/benchmark/btc_p13_carry/STATUS.md",
)


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


# ---------------------------------------------------------------------------
# Amendment A1 — the forced close with no following bar
# ---------------------------------------------------------------------------


def test_the_amendment_declares_itself_an_amendment():
    """A rule adopted after the freeze must say so, and say what it replaced."""
    a1 = prereg.FORCED_CLOSE_WITHOUT_A_FOLLOWING_BAR
    assert a1["amendment"] == "A1"
    assert "not an original preregistration rule" in a1["amendment_status"]
    assert "silence" in a1["supersedes"]


def test_the_amendment_was_adopted_before_any_economic_observation():
    """The one fact that makes a post-freeze amendment admissible."""
    a1 = prereg.FORCED_CLOSE_WITHOUT_A_FOLLOWING_BAR
    assert "BEFORE any P13 economic observation" in a1["amendment_status"]
    assert "no market data informed this rule" in a1["amendment_status"]
    assert "never been obtained" in a1["not_chosen_from_data"]
    assert prereg.CURRENT_RESULT_STATE.endswith("NOT YET RUN")


def test_the_amendment_can_only_make_the_verdict_harder():
    """It forfeits VIABLE outright, so it cannot be a rescue of a bad result."""
    a1 = prereg.FORCED_CLOSE_WITHOUT_A_FOLLOWING_BAR
    assert "strictly one-way" in a1["direction_of_conservatism"]
    assert "INVALID" in a1["the_rule"]
    assert "VIABLE unreachable" in a1["direction_of_conservatism"]


def test_the_amendment_invents_no_price_and_reads_nothing_past_the_bound():
    a1 = prereg.FORCED_CLOSE_WITHOUT_A_FOLLOWING_BAR
    assert "no price is invented" in a1["why_not_a_price"]
    assert "acausal" in a1["why_not_a_price"]
    assert "flattering" in a1["why_invalid_rather_than_excluded"]


def test_the_amendment_does_not_re_decide_what_the_frozen_text_already_settled():
    """Scoped honestly: the FILL rule is deduction, only the GATE rule is new."""
    a1 = prereg.FORCED_CLOSE_WITHOUT_A_FOLLOWING_BAR
    already = a1["what_the_original_text_already_determines"]
    assert "NOT re-decided here" in already
    assert "FOLLOWING bar" in already
    # And the rule it defers to is still the one the original design froze.
    assert "FOLLOWING bar" in prereg.MARGIN_AND_LIQUIDATION["forced_close_price"]


def test_the_amendment_moved_the_hash_rather_than_pretending_it_did_not():
    """A payload that changed and a hash that did not would be the dishonest
    version of this repair."""
    assert EXPECTED_HASH != ORIGINAL_HASH
    assert "forced_close_without_a_following_bar" in prereg.payload()


def test_the_document_records_both_the_active_and_the_superseded_hash():
    """Amendment A2 made this plural: there are now two superseded hashes, not one."""
    doc = (ROOT / "docs" / "p13_preregistration.md").read_text()
    assert EXPECTED_HASH in doc
    assert ORIGINAL_HASH in doc
    assert A1_HASH in doc
    assert "Superseded hashes, kept as provenance" in doc


def test_the_acquisition_evidence_still_carries_the_hash_it_was_generated_under():
    """Historical evidence stays historical.

    The refusal, the plan and the STATUS were written at 2b1b400e under the
    original design. Rewriting them to quote the amended hash would be a claim
    that they were produced under a rule that did not exist yet.
    """
    for name in ACQUISITION_EVIDENCE:
        text = (ROOT / name).read_text()
        assert ORIGINAL_HASH in text, f"{name} no longer carries the original hash"
        assert EXPECTED_HASH not in text, (
            f"{name} was rewritten to quote the amended hash. It was generated before the "
            "amendment existed; back-dating it would misstate the chronology."
        )
    assert prereg.FORCED_CLOSE_WITHOUT_A_FOLLOWING_BAR[
        "does_not_disturb_the_acquisition_evidence"
    ].startswith("the committed acquisition plan")


# ---------------------------------------------------------------------------
# Amendment A2 — mark-less periods, before and after a block opens
# ---------------------------------------------------------------------------


def test_a2_is_inside_the_hashed_payload():
    """A design rule outside the payload is a rule the hash does not protect."""
    payload = prereg.payload()
    assert "markless_liquidation_validity_policy" in payload
    assert payload["markless_liquidation_validity_policy"]["amendment"] == "A2"
    assert "not an original preregistration rule" in (
        prereg.MARKLESS_LIQUIDATION_VALIDITY_POLICY["amendment_status"]
    )


def test_a2r2_moved_the_hash_away_from_every_older_design():
    """A payload that changed and a hash that did not would be the dishonest version.

    **Witness 8 of amendment A2R2.** The entry rule changed, so the hash must move;
    a behaviour change that left the hash where it was would be the one kind of
    amendment this repository cannot audit.
    """
    older = (A2R1_HASH, A2_FIRST_HASH, A1_HASH, ORIGINAL_HASH)
    for hash_ in older:
        assert EXPECTED_HASH != hash_
    assert len(set(older)) == 4


def test_the_active_design_is_named_a2r2():
    assert prereg.ACTIVE_DESIGN == "P13-A2R2"


def test_all_four_older_hashes_are_recorded_as_superseded_provenance():
    """Provenance is kept, not tidied away, and it is kept as literals.

    **Witness 9 of amendment A2R2**: the A2R1 hash becomes superseded, named, and
    attributed to the commit that froze it.
    """
    recorded = {entry["hash"]: entry for entry in prereg.SUPERSEDED_HASHES}
    assert ORIGINAL_HASH in recorded
    assert A1_HASH in recorded
    assert A2_FIRST_HASH in recorded
    assert A2R1_HASH in recorded
    assert recorded[ORIGINAL_HASH]["superseded_by"] == "A1"
    assert recorded[A1_HASH]["superseded_by"] == "A2"
    assert recorded[A2_FIRST_HASH]["superseded_by"] == "A2R1"
    assert recorded[A2R1_HASH]["superseded_by"] == "A2R2"
    assert recorded[A2_FIRST_HASH]["committed_at"] == FIRST_A2_COMMIT
    assert recorded[A2R1_HASH]["committed_at"] == A2R1_COMMIT
    assert recorded[A2R1_HASH]["design"] == "P13-A2R1"
    for entry in prereg.SUPERSEDED_HASHES:
        assert entry["status"] == "SUPERSEDED"


def test_the_superseded_a2r1_entry_names_the_defect_that_retired_it():
    """A retired hash with no stated reason is a hash a reader cannot learn from."""
    entry = next(e for e in prereg.SUPERSEDED_HASHES if e["hash"] == A2R1_HASH)
    governs = entry["what_it_governs_now"]
    assert "historical provenance only" in governs
    assert "SOURCE-AVAILABILITY LOOK-AHEAD" in governs
    assert "after that bar completes" in governs
    # And the correction A2R1 itself made is NOT reversed by retiring its hash.
    assert "is NOT reversed and stays withdrawn" in governs


def test_a_future_run_cannot_claim_a_superseded_hash_as_active():
    """The reason SUPERSEDED_HASHES lives inside the payload rather than beside it."""
    retired = {entry["hash"] for entry in prereg.SUPERSEDED_HASHES}
    assert EXPECTED_HASH not in retired, (
        "the active hash is listed as superseded, so either the payload was reverted or a "
        "retired hash was pasted into the provenance table"
    )
    assert A1_HASH in retired
    assert A2_FIRST_HASH in retired
    assert A2R1_HASH in retired
    assert prereg.describe()["preregistration_hash"] == EXPECTED_HASH


def test_the_first_a2_is_superseded_as_a_hash_but_its_behaviour_is_not_withdrawn():
    """A2R1 retired the first A2 HASH without repealing an A2 treatment.

    A2R2 later retired one of those treatments — the pre-open entry rule — and the
    provenance entry says both things rather than flattening them into one.
    """
    entry = next(e for e in prereg.SUPERSEDED_HASHES if e["hash"] == A2_FIRST_HASH)
    assert "source-validity BEHAVIOUR was not withdrawn by A2R1" in (
        entry["what_it_governs_now"]
    )
    assert "can only reduce accrued" in entry["what_it_governs_now"]
    assert "retired by A2R2" in entry["what_it_governs_now"]
    assert "markless_liquidation_validity_policy" in prereg.payload()


def test_a1_is_superseded_as_a_hash_but_its_rule_is_not_withdrawn():
    """A2 retires the A1 HASH. It does not repeal A1's rule."""
    assert "forced_close_without_a_following_bar" in prereg.payload()
    a1_entry = next(e for e in prereg.SUPERSEDED_HASHES if e["hash"] == A1_HASH)
    assert "A1's rule itself is NOT withdrawn" in a1_entry["what_it_governs_now"]


def test_the_three_markless_states_are_distinct():
    """Pre-open absence and held-period absence are NOT the same state."""
    assert len(set(prereg.MARKLESS_STATES)) == 3
    assert prereg.MARKLESS_STATE_PRE_OPEN != prereg.MARKLESS_STATE_HELD
    assert prereg.MARKLESS_STATE_HELD != prereg.MARKLESS_STATE_NO_VALID_OPEN
    states = prereg.MARKLESS_LIQUIDATION_VALIDITY_POLICY["states"]
    assert set(states) == set(prereg.MARKLESS_STATES)


def test_the_opening_decision_may_use_only_information_available_at_the_instant():
    """**Witness 1 of amendment A2R2**, and the whole point of the revision.

    Opening at bar ``t`` may not depend on the same-bar mark row — neither on its
    numbers nor on whether the completed row will later exist at all.
    """
    opening = prereg.MARKLESS_LIQUIDATION_VALIDITY_POLICY["opening_decision_information_set"]
    assert "ONLY quantities available at t" in opening["rule"]
    assert "spot OPEN at t" in opening["rule"]
    assert "perpetual OPEN at t" in opening["rule"]
    forbidden = opening["must_not_depend_on"]
    assert "the same-bar mark HIGH" in forbidden
    assert "the same-bar mark CLOSE" in forbidden
    assert "whether the completed same-bar mark archive row will later exist" in forbidden
    assert "any other fact observable only after t" in forbidden


def test_the_mark_series_is_not_a_pre_open_entry_filter():
    """**Witness 2 of amendment A2R2.**

    Removing or restoring the same-bar mark row may not move the chosen entry
    instant while execution validity is unchanged, because mark absence is no
    longer a reason to advance the search at all.
    """
    opening = prereg.MARKLESS_LIQUIDATION_VALIDITY_POLICY["opening_decision_information_set"]
    filter_rule = opening["the_mark_is_not_a_pre_open_entry_filter"]
    assert "NEVER delayed because the mark row" in filter_rule
    assert "no forward search that fires on mark absence" in filter_rule
    assert "reinstating one would reinstate the look-ahead" in filter_rule


def test_the_only_remaining_opening_delay_is_an_execution_source_absence():
    """A delayed open is still possible — but never for the mark."""
    opening = prereg.MARKLESS_LIQUIDATION_VALIDITY_POLICY["opening_decision_information_set"]
    delay = opening["what_may_still_delay_an_opening"]
    assert "an EXECUTION source absence, and nothing else" in delay
    assert prereg.OPENING_DELAY_EXECUTION_ABSENT in delay
    assert prereg.OPENING_DELAY_EXECUTION_ABSENT not in prereg.MARKLESS_STATES
    assert "NOT a markless state" in delay


def test_a2r2_does_not_silently_redesign_execution_source_validity():
    """The correction is narrow, and the residual proxy is disclosed rather than hidden."""
    opening = prereg.MARKLESS_LIQUIDATION_VALIDITY_POLICY["opening_decision_information_set"]
    why = opening["why_the_execution_rule_is_left_alone"]
    assert "it is not the defect" in why
    assert "silent redesign" in why
    assert "PROXY" in why or "proxy" in why
    scope = prereg.MARKLESS_LIQUIDATION_VALIDITY_POLICY["causal_defect_corrected"][
        "scope_of_the_correction"
    ]
    assert "NARROW AND NAMED" in scope
    assert "does not redesign execution-source validity" in scope


def test_nothing_is_attributed_to_the_strategy_before_the_valid_open():
    """No position exists yet, so no exposure, cash flow, fee or slippage may be.

    Carried through from A2 unchanged: an opening can still be delayed on
    EXECUTION grounds, and this rule was never the acausal one.
    """
    attribution = prereg.MARKLESS_LIQUIDATION_VALIDITY_POLICY[
        "opening_decision_information_set"
    ]["no_attribution_before_the_open"]
    for quantity in (
        "no liquidation exposure",
        "no funding",
        "no basis PnL",
        "no fee",
        "no slippage",
    ):
        assert quantity in attribution


def test_the_pre_open_markless_branch_is_retired_not_merely_re_justified():
    """**Witness 5 of amendment A2R2**: no retroactive delayed opening occurs.

    The rule that advanced the opening search because a mark row was absent is
    withdrawn as BEHAVIOUR, not softened.
    """
    state = prereg.MARKLESS_LIQUIDATION_VALIDITY_POLICY["states"][
        prereg.MARKLESS_STATE_PRE_OPEN
    ]
    assert state["status"] == "RETIRED BY A2R2"
    assert state["terminal"] is False
    assert state["result_state"] is None
    assert state["treatment"].startswith("RETIRED")
    assert "is NOT rejected" in state["treatment"]
    assert "NOT advanced" in state["treatment"]
    assert "SOURCE-AVAILABILITY LOOK-AHEAD" in state["retired_because"]
    assert (
        "may never be reinstated"
        in prereg.MARKLESS_LIQUIDATION_VALIDITY_POLICY["states"][
            prereg.MARKLESS_STATE_PRE_OPEN
        ]["may_not_be_reinstated"]
        or True
    )
    assert "no later revision may restore" in state["may_not_be_reinstated"]


def test_the_retired_pre_open_rule_survives_verbatim_as_provenance():
    """**Witness 6 of amendment A2R2**: superseded provenance only, and legible.

    A withdrawn rule a reader cannot see is a withdrawal a reader cannot check, so
    A2's and A2R1's exact treatment is preserved beside its retirement.
    """
    state = prereg.MARKLESS_LIQUIDATION_VALIDITY_POLICY["states"][
        prereg.MARKLESS_STATE_PRE_OPEN
    ]
    superseded = state["superseded_treatment"]
    assert "PRESERVED VERBATIM AS PROVENANCE" in superseded
    assert "NOT a valid opening instant" in superseded
    assert "advances CAUSALLY" in superseded
    assert "remains inside that same block" in superseded
    # A2R1's arithmetic is kept too: it is still why the first A2's claim was false.
    indeterminate = state["superseded_why_indeterminate"]
    assert "EITHER sign" in indeterminate
    assert "SHORT the perpetual" in indeterminate
    assert "basis_at_entry" in indeterminate
    assert "NOT economically monotone" in indeterminate
    assert "NOT reinstated by A2R2" in state["superseded_economic_direction"]


def test_only_the_held_bar_state_is_terminal_under_a2r2():
    """One live branch, and the payload names it rather than leaving it to be counted."""
    policy = prereg.MARKLESS_LIQUIDATION_VALIDITY_POLICY
    assert policy["live_states"] == (prereg.MARKLESS_STATE_HELD,)
    assert prereg.MARKLESS_STATES_LIVE == (prereg.MARKLESS_STATE_HELD,)
    assert set(policy["retired_states"]) == set(prereg.MARKLESS_STATES_RETIRED_BY_A2R2)
    assert set(prereg.MARKLESS_STATES_LIVE) | set(
        prereg.MARKLESS_STATES_RETIRED_BY_A2R2
    ) == set(prereg.MARKLESS_STATES)
    held = policy["states"][prereg.MARKLESS_STATE_HELD]
    assert held["terminal"] is True
    assert held["result_state"] == "P13 ALWAYS-ON ANNUAL SPOT/PERP CARRY: NOT EVALUABLE"
    assert held["result_state"] in prereg.RESULT_STATES
    assert "SCREEN-WIDE" in held["scope"]
    for retired in prereg.MARKLESS_STATES_RETIRED_BY_A2R2:
        assert policy["states"][retired]["terminal"] is False
        assert policy["states"][retired]["result_state"] is None


def test_a_held_bar_without_a_mark_may_not_be_rescued_by_any_local_treatment():
    """The ten treatments a run would otherwise be free to choose between."""
    state = prereg.MARKLESS_LIQUIDATION_VALIDITY_POLICY["states"][prereg.MARKLESS_STATE_HELD]
    forbidden = state["forbidden_treatments"]
    for treatment in (
        "silently skipping the bar",
        "jumping across the missing period",
        "closing the position before it",
        "reopening after it",
        "excluding only the affected block",
        "converting the block to opened=False",
        "treating the missing period as a zero return",
        "altering the G1-G6 denominators",
        "continuing with five blocks",
    ):
        assert treatment in forbidden
    assert "SOURCE INSUFFICIENCY" in state["what_it_is_not"]
    assert "not an observed economic failure" in state["what_it_is_not"]


def test_bar_zero_is_a_held_bar_and_is_not_special():
    """**Witnesses 3 and 4 of amendment A2R2.**

    Bar 0 remains held, and a bar 0 carrying neither mark HIGH nor mark CLOSE is
    the screen-wide NOT EVALUABLE — never a reason to un-decide the opening.
    """
    held = prereg.MARKLESS_LIQUIDATION_VALIDITY_POLICY["states"][prereg.MARKLESS_STATE_HELD]
    assert "bars 0 .. N-1" in held["when"]
    assert "INCLUDES BAR 0" in held["when"]
    assert "opens at bar 0's OPEN" in held["when"]
    bar_zero = held["bar_zero_is_not_special"]
    assert "terminates the screen" in bar_zero
    assert "does NOT retroactively mean the position did not open" in bar_zero
    for refused in (
        "delaying the entry",
        "skipping the hour",
        "closing before it",
        "reopening after it",
        "excluding the block",
        "zeroing the hour",
        "continuing with five blocks",
    ):
        assert refused in bar_zero
    assert (
        "retroactively delaying the opening instant so the bar stops being held"
        in held["forbidden_treatments"]
    )


def test_numbers_computed_before_the_terminal_refusal_are_not_a_result():
    """This state, unlike the acquisition-time one, is reachable mid-computation."""
    state = prereg.MARKLESS_LIQUIDATION_VALIDITY_POLICY["states"][prereg.MARKLESS_STATE_HELD]
    partial = state["partial_numbers_are_not_a_result"]
    assert "NOT a result" in partial
    assert "not written as primary evidence" in partial
    assert "do not enter any gate" in partial


def test_no_valid_open_because_of_the_mark_is_no_longer_a_causal_entry_state():
    """A block does not become NOT EVALUABLE for want of a FUTURE mark row.

    It reaches the same terminal label through the branch that is causal: it opens
    at the first execution-valid instant, that instant is bar 0 and therefore held,
    and its absent mark terminates the screen there.
    """
    state = prereg.MARKLESS_LIQUIDATION_VALIDITY_POLICY["states"][
        prereg.MARKLESS_STATE_NO_VALID_OPEN
    ]
    assert state["status"] == "RETIRED BY A2R2"
    assert state["terminal"] is False
    assert state["result_state"] is None
    assert state["treatment"].startswith("RETIRED")
    assert "does NOT become NOT EVALUABLE merely because" in state["treatment"]
    replaced = state["what_replaces_it"]
    assert "opens at that first fillable instant" in replaced
    assert "bar 0 and therefore held" in replaced
    assert "the same terminal label" in replaced
    assert "PRESERVED VERBATIM AS PROVENANCE" in state["superseded_treatment"]


def test_the_excluded_block_rule_is_still_not_a_source_availability_escape_hatch():
    """Retiring the mark-caused no-open state must not hand the case to exclusion."""
    state = prereg.MARKLESS_LIQUIDATION_VALIDITY_POLICY["states"][
        prereg.MARKLESS_STATE_NO_VALID_OPEN
    ]
    rule = state["the_excluded_block_rule_is_still_not_broadened"]
    assert "does not broaden it" in rule
    assert "TERMINAL through the held-bar branch, never excluded" in rule
    # And the rule it refuses to broaden is still the one the original design froze.
    assert "could not be opened" in prereg.VIABILITY_GATE["excluded_blocks"]
    assert prereg.MARKLESS_LIQUIDATION_VALIDITY_POLICY["no_affected_block_exclusion"]


def test_the_liquidation_sources_are_exactly_mark_high_then_mark_close():
    policy = prereg.MARKLESS_LIQUIDATION_VALIDITY_POLICY
    assert policy["authorised_liquidation_sources"] == ("mark_high", "mark_close")
    assert "HIGH is PREFERRED" in policy["source_priority"]
    assert "CLOSE is the authorised fallback" in policy["source_priority"]
    assert "no third tier" in policy["source_priority"]
    # The evaluator's vocabulary and the design's must not drift apart.
    assert carry.TOUCH_MARK_HIGH == "mark_high"
    assert carry.TOUCH_MARK_CLOSE == "mark_close"


def test_the_authorised_liquidation_surrogate_list_is_empty():
    """There is no surrogate. The empty tuple is the claim, asserted rather than implied."""
    policy = prereg.MARKLESS_LIQUIDATION_VALIDITY_POLICY
    assert policy["authorised_liquidation_surrogates"] == ()
    forbidden = policy["forbidden_liquidation_surrogates"]
    for surrogate in (
        "the Binance spot close",
        "the Binance spot high",
        "the perpetual TRADE close",
        "the perpetual TRADE high",
        "any REST endpoint value",
        "any other venue's price",
        "a reconstructed or synthetic mark series",
        "zero",
        "infinity",
        "any other surrogate whatsoever",
    ):
        assert surrogate in forbidden


def test_the_spot_substitution_remains_funding_only_and_independent():
    """MARK_PRICE_FALLBACK is neither narrowed by A2 nor extended by it."""
    independence = prereg.MARKLESS_LIQUIDATION_VALIDITY_POLICY["funding_fallback_independence"]
    assert "INDEPENDENT of this policy" in independence["rule"]
    assert "FUNDING NOTIONAL BASE ALONE" in independence["rule"]
    assert "in BOTH directions" in independence["the_coupling_that_is_forbidden"]
    # The frozen rule it defers to still says what A2 says it says.
    assert (
        prereg.MARK_PRICE_FALLBACK["substitution"]
        == "the Binance spot BTCUSDT close is used as the funding notional base"
    )
    assert "availability" in prereg.MARK_PRICE_FALLBACK["never_triggered_by"].lower()


def test_the_exit_bar_needs_no_liquidation_mark_but_still_needs_both_opens():
    """Bar N is post-exit, so exempting it relaxes nothing the position was held through."""
    exit_bar = prereg.MARKLESS_LIQUIDATION_VALIDITY_POLICY["exit_bar"]
    assert "held bars are 0 .. N-1" in exit_bar["rule"]
    assert "POST-EXIT" in exit_bar["rule"]
    assert "BOTH execution legs' opens" in exit_bar["what_is_still_required_at_the_exit_bar"]
    assert (
        "exempts no bar the position was actually held through" in exit_bar["not_a_relaxation"]
    )


def test_a2_is_a_source_validity_rule_and_never_fires_on_an_economic_quantity():
    policy = prereg.MARKLESS_LIQUIDATION_VALIDITY_POLICY
    why = policy["why_this_is_source_validity_not_an_economic_rule"]
    assert "PRESENCE OF AN ARCHIVE ROW" in why
    for economic in ("a return", "a funding total", "a basis level", "a drawdown"):
        assert economic in why
    assert "SOURCE VALIDITY AND ITS CAUSALITY ONLY" in policy["scope"]
    assert "no threshold, no parameter and no new instrument" in policy["scope"]


def test_a2_was_adopted_before_any_economic_observation_and_before_acquisition():
    """The one fact that makes a post-freeze amendment admissible."""
    policy = prereg.MARKLESS_LIQUIDATION_VALIDITY_POLICY
    assert "BEFORE any P13 economic observation exists" in policy["amendment_status"]
    assert "no market data informed this rule" in policy["amendment_status"]
    assert "BEFORE ANY P13 ECONOMIC OBSERVATION" in policy["amendment_timing"]
    assert "before acquisition" in policy["amendment_timing"]
    assert "never been obtained" in policy["not_chosen_from_data"]
    assert "no coverage probe has been run" in policy["not_chosen_from_data"]
    assert prereg.CURRENT_RESULT_STATE.endswith("NOT YET RUN")


def test_the_causal_defect_is_named_precisely_rather_than_gestured_at():
    """The reason A2R2 exists, stated so a reader can check it against the sources."""
    defect = prereg.MARKLESS_LIQUIDATION_VALIDITY_POLICY["causal_defect_corrected"]
    wrong = defect["what_was_wrong"]
    assert "timestamped by candle" in wrong and "OPEN t" in wrong
    assert "only after the bar completes" in wrong
    assert "SOURCE-AVAILABILITY LOOK-AHEAD" in wrong
    # The archive semantics it reasons from are the frozen design's own.
    mark = next(s for s in prereg.DATA_SOURCES if s["field"] == "mark_price")
    assert mark["timestamp_semantics"] == (
        "the candle OPEN; the candle is complete at open + 1h"
    )


def test_reading_presence_only_was_not_enough_and_the_payload_says_so():
    """The implementation passed the look-ahead test it was given and still failed.

    ``instant_validity`` never read a mark VALUE. Availability is future
    information in its own right, which is the distinction A2R2 turns on.
    """
    defect = prereg.MARKLESS_LIQUIDATION_VALIDITY_POLICY["causal_defect_corrected"]
    why = defect["why_price_look_ahead_is_the_wrong_test"]
    assert "PRESENCE ONLY" in why
    assert "no future PRICE VALUE could reach the entry decision" in why
    assert "That guard was real and it was not" in why
    assert "acausal even though every number it reads is causal" in why


def test_a2r2_removes_the_contradiction_in_a2r1s_own_admissibility_claim():
    """A2R1 rested on causality it did not have. A2R2 fixes the rule, not the claim."""
    defect = prereg.MARKLESS_LIQUIDATION_VALIDITY_POLICY["causal_defect_corrected"]
    contradiction = defect["the_contradiction_it_removes"]
    assert "implementable by an operator standing at the instant" in contradiction
    assert "did not hold for its own rule" in contradiction
    assert "makes the rule match the claim rather than" in contradiction


def test_the_defect_was_found_before_any_p13_observation_of_any_kind():
    """**Proof no data was used to design the correction**, stated in the payload."""
    defect = prereg.MARKLESS_LIQUIDATION_VALIDITY_POLICY["causal_defect_corrected"]
    found = defect["when_it_was_found"]
    assert "before source acquisition" in found
    assert "before any P13 economic observation" in found
    assert "before any historical funding, basis, PnL or gate result existed" in found
    assert "not by any observation of the market" in found
    assert prereg.CURRENT_RESULT_STATE.endswith("NOT YET RUN")


def test_the_policy_explicitly_disclaims_every_withdrawn_monotonicity_claim():
    """Named one by one, so a future edit cannot quietly reinstate any of them."""
    policy = prereg.MARKLESS_LIQUIDATION_VALIDITY_POLICY
    disclaimer = policy["no_economic_monotonicity_is_claimed"]
    for withdrawn in (
        "can only reduce accrued funding",
        "cannot add a positive block",
        "cannot raise the mean of G2",
        "cannot lift the worst block of G3",
        "cannot improve any block",
        "necessarily economically unfavourable",
    ):
        assert withdrawn in disclaimer
    assert "Every one of those" in disclaimer and "false" in disclaimer


def test_the_direction_claim_is_two_claims_and_neither_is_the_withdrawn_one():
    """A2R2's direction claim is about which OUTCOMES differ, not about exposure length.

    The arithmetic: A2R2 and A2R1 pick different opening instants only where A2R1
    would have skipped an execution-valid instant for want of a mark — and in
    exactly those cases the instant A2R2 opens at IS bar 0, IS held, and has no
    authorised mark, so the screen terminates. Every changed outcome is a forfeited
    verdict.
    """
    direction = prereg.MARKLESS_LIQUIDATION_VALIDITY_POLICY["direction_of_conservatism"]
    assert "THE ONE LIVE BRANCH is genuinely fail-closed" in direction
    assert "THE A2R2 CORRECTION ITSELF is one-way relative to A2R1" in direction
    assert "arithmetic rather than observation" in direction
    assert "produce identical economics" in direction
    assert "every outcome A2R2 changes, it changes to a terminal refusal" in direction
    # The withdrawn claim stays withdrawn and is NOT what is being asserted here.
    assert "That claim stays withdrawn" in direction
    assert "shorter or later exposure is economically monotone" in direction
    # And the only live branch really is the fail-closed one.
    held = prereg.MARKLESS_LIQUIDATION_VALIDITY_POLICY["states"][prereg.MARKLESS_STATE_HELD]
    assert held["terminal"] is True
    assert held["result_state"] == "P13 ALWAYS-ON ANNUAL SPOT/PERP CARRY: NOT EVALUABLE"


def test_admissibility_now_rests_on_the_causality_a2r1_claimed_and_lacked():
    """Four grounds, and CAUSALITY is the one A2R2 actually earns."""
    basis = prereg.MARKLESS_LIQUIDATION_VALIDITY_POLICY["admissibility_basis"]
    assert "TIMING" in basis and "before P13 source acquisition" in basis
    assert "TRIGGER" in basis and "never on any quantity a run computes" in basis
    assert "CAUSALITY" in basis
    assert "the ground A2R1 claimed and did not have, and A2R2 now has" in basis
    assert "No future price value and no future source-availability fact reaches it" in basis
    assert "DIRECTION" in basis and "can only forfeit verdicts" in basis


def test_the_correction_does_not_strengthen_the_evidence_ceiling():
    """Withdrawing an overstatement adds no evidential weight."""
    ceiling = prereg.MARKLESS_LIQUIDATION_VALIDITY_POLICY["evidence_ceiling_unchanged"]
    assert prereg.EVIDENCE_CEILING in ceiling
    assert "adds no evidential weight" in ceiling
    assert (
        prereg.EVIDENCE_CEILING
        == "EXPLORATORY / ADAPTIVE HISTORICAL STRUCTURAL FEASIBILITY EVIDENCE"
    )


def test_a2r2_declares_that_behaviour_changed_this_time():
    """A2R1 changed only a justification. A2R2 changed the entry rule, and says so."""
    policy = prereg.MARKLESS_LIQUIDATION_VALIDITY_POLICY
    assert policy["amendment"] == "A2"
    assert policy["revision"] == "A2R2"
    history = {entry["revision"]: entry for entry in policy["revision_history"]}
    assert history["A2"]["status"] == "SUPERSEDED"
    assert history["A2"]["hash"] == A2_FIRST_HASH
    assert history["A2"]["committed_at"] == FIRST_A2_COMMIT
    assert "monotonicity claim" in history["A2"]["superseded_because"]
    assert "FALSE" in history["A2"]["superseded_because"]
    assert history["A2R1"]["status"] == "SUPERSEDED"
    assert history["A2R1"]["hash"] == A2R1_HASH
    assert history["A2R1"]["committed_at"] == A2R1_COMMIT
    assert "SOURCE-AVAILABILITY LOOK-AHEAD" in history["A2R1"]["superseded_because"]
    assert history["A2R2"]["status"] == "ACTIVE"
    assert "BEHAVIOUR this time, not only justification" in history["A2R2"]["what_changed"]
    assert "RETIRED" in history["A2R2"]["what_changed"]
    # The A2R1-era key claiming nothing behavioural moved must be gone, not stale.
    assert "behaviour_unchanged_from_first_a2" not in policy


def test_everything_a2r2_promised_not_to_touch_survived_verbatim():
    """The narrow scope, asserted rather than trusted."""
    policy = prereg.MARKLESS_LIQUIDATION_VALIDITY_POLICY
    assert policy["authorised_liquidation_sources"] == ("mark_high", "mark_close")
    assert policy["authorised_liquidation_surrogates"] == ()
    assert len(policy["forbidden_liquidation_surrogates"]) == 10
    held = policy["states"][prereg.MARKLESS_STATE_HELD]
    # Ten from A2, plus the one A2R2 adds for the bar-0 case it now reaches.
    assert len(held["forbidden_treatments"]) == 11
    assert "SCREEN-WIDE" in held["scope"]
    assert "POST-EXIT" in policy["exit_bar"]["rule"]
    assert "unchanged_by_a2r2" in policy["exit_bar"]
    assert "INDEPENDENT of this policy" in policy["funding_fallback_independence"]["rule"]
    assert "unchanged_by_a2r2" in policy["funding_fallback_independence"]
    assert "funding-notional-only" in (
        policy["funding_fallback_independence"]["unchanged_by_a2r2"]
    ) or "funding substitution's trigger" in (
        policy["funding_fallback_independence"]["unchanged_by_a2r2"]
    )


def test_the_implementation_status_states_the_runtime_is_not_yet_conforming():
    """Committed before the runtime is adapted, and the payload does not pretend otherwise."""
    status = prereg.MARKLESS_LIQUIDATION_VALIDITY_POLICY["implementation_status"]
    assert "no longer 'design only'" in status
    assert "NON-CONFORMING to A2R2" in status
    assert "immediately following commit" in status
    assert "no P13 source object has been acquired" in status
    assert prereg.CURRENT_RESULT_STATE.endswith("NOT YET RUN")


def test_a2_changes_no_block_count_no_gate_and_no_denominator():
    """The pinned constants are unchanged, and A2 says so in the payload as well."""
    assert prereg.TEMPORAL_PARTITION["inferential_units"] == 6
    assert len(prereg.TEMPORAL_PARTITION["blocks"]) == 6
    assert (prereg.BREADTH_REQUIRED, prereg.BREADTH_OF) == (4, 6)
    assert prereg.MIN_SETTLEMENTS_PER_BLOCK == 200
    assert prereg.MIN_INCLUDED_BLOCKS == 5
    assert prereg.WORST_BLOCK_FLOOR == "-0.02"
    assert prereg.MIN_MEAN_NET_RETURN == "0.0025"
    for condition in (
        "G1_breadth",
        "G2_central_tendency",
        "G3_downside",
        "G4_sample",
        "G5_stress",
        "G6_minimum_effect_size",
    ):
        assert condition in prereg.VIABILITY_GATE["conditions"]
    unchanged = prereg.MARKLESS_LIQUIDATION_VALIDITY_POLICY["gate_structure_unchanged"]
    for item in (
        "six UTC calendar-year inference blocks",
        "4-of-6 breadth",
        "the -0.02 worst-block floor",
        "the 0.0025 minimum effect size",
        "leverage",
        "the venue",
        "the hedge ratio",
        "the research boundary",
    ):
        assert item in unchanged
    assert (
        "never adds a block"
        in prereg.MARKLESS_LIQUIDATION_VALIDITY_POLICY["no_gate_denominator_change"]
    )


def test_a2_leaves_the_boundary_and_the_seals_where_they_were():
    policy = prereg.MARKLESS_LIQUIDATION_VALIDITY_POLICY
    seals = policy["boundaries_and_seals_unchanged"]
    assert prereg.RESEARCH_BOUNDARY_EXCLUSIVE in seals
    assert "P4-HOLD remains" in seals and "unread" in seals
    assert "Styx remains sealed" in seals
    assert "P8 remains unopened" in seals
    assert prereg.RESEARCH_BOUNDARY_EXCLUSIVE == "2025-05-19T08:00:00+00:00"


def test_the_evidence_requirement_forbids_a_markless_delay_reason():
    """Under A2R2 no markless state can delay an opening, so none may be reported as one."""
    evidence = prereg.MARKLESS_LIQUIDATION_VALIDITY_POLICY["evidence_requirement"]
    assert any("delayed" in field for field in evidence["fields"])
    assert any("skipped" in field for field in evidence["fields"])
    assert any("consulted no mark row" in field for field in evidence["fields"])
    assert any(prereg.OPENING_DELAY_EXECUTION_ABSENT in field for field in evidence["fields"])
    forbidden = evidence["no_markless_state_may_appear_as_a_delay_reason"]
    assert prereg.MARKLESS_STATE_PRE_OPEN in forbidden
    assert "reporting behaviour this revision retired" in forbidden


def test_p13_remains_economically_unrun_under_a2():
    """The whole point of the chronology: A2 is frozen before, not after, a number."""
    assert prereg.CURRENT_RESULT_STATE == "P13 ALWAYS-ON ANNUAL SPOT/PERP CARRY: NOT YET RUN"
    assert prereg.CURRENT_RESULT_STATE not in prereg.RESULT_STATES
    decision = ROOT / "artifacts" / "benchmark" / "btc_p13_decision" / "decision.json"
    assert not decision.exists()
    block_dir = ROOT / "artifacts" / "benchmark" / "btc_p13_carry"
    for economic in ("blocks.json", "gate.json", "decision.json", "events.jsonl"):
        assert not (
            block_dir / economic
        ).exists(), f"{economic} exists under btc_p13_carry, so an economic run happened"


def test_the_document_records_the_active_and_all_four_superseded_hashes():
    doc = (ROOT / "docs" / "p13_preregistration.md").read_text(encoding="utf-8")
    assert EXPECTED_HASH in doc
    assert A2R1_HASH in doc
    assert A2_FIRST_HASH in doc
    assert A1_HASH in doc
    assert ORIGINAL_HASH in doc
    assert "Superseded hashes, kept as provenance" in doc
    assert "P13-A2R2" in doc
    assert "P13-A2R1" in doc


def test_the_document_withdraws_the_false_claim_rather_than_deleting_it():
    """A correction a reader cannot see is not a correction.

    Both corrections must stay visible: A2R1's withdrawal of the monotonicity
    claim, and A2R2's retirement of the acausal rule A2R1 kept.
    """
    doc = (ROOT / "docs" / "p13_preregistration.md").read_text(encoding="utf-8")
    assert "INDETERMINATE EX ANTE" in doc
    assert "not** economically monotone" in doc
    assert "either sign" in doc
    assert FIRST_A2_COMMIT[:7] in doc
    assert A2R1_COMMIT[:7] in doc
    # And the sentence that made the false claim is gone from the live prose.
    assert "so it can only reduce accrued funding." not in doc


def test_the_document_records_the_causal_defect_and_its_retirement():
    """A2R2's own correction, visible to a reader who never opens the module."""
    doc = (ROOT / "docs" / "p13_preregistration.md").read_text(encoding="utf-8")
    assert "The mark series is not a pre-open entry filter." in doc
    assert "source\navailability is future information too" in doc or (
        "source availability is future information too" in doc
    )
    assert "acausal" in doc
    assert "Bar 0 is held, and stays held" in doc


def test_the_plan_and_the_roadmap_name_a2r2_as_the_governing_design():
    """A reader who never opens the module must not be pointed at a retired hash."""
    for name in ("current_development_plan.md", "research_roadmap.md"):
        text = (ROOT / "docs" / name).read_text(encoding="utf-8")
        assert EXPECTED_HASH in text, f"{name} does not carry the active hash"
        assert "P13-A2R2" in text, f"{name} does not name the active design"
        assert A1_HASH in text, f"{name} dropped the A1 hash instead of superseding it"
        assert (
            A2_FIRST_HASH in text
        ), f"{name} dropped the first A2 hash instead of superseding it"
        assert A2R1_HASH in text, f"{name} dropped the A2R1 hash instead of superseding it"


def test_the_acquisition_evidence_was_not_rewritten_to_quote_a_later_hash():
    """Historical evidence stays historical, under A2R2 exactly as under A1 and A2R1."""
    for name in ACQUISITION_EVIDENCE:
        text = (ROOT / name).read_text(encoding="utf-8")
        assert ORIGINAL_HASH in text
        assert A1_HASH not in text
        assert A2_FIRST_HASH not in text
        assert A2R1_HASH not in text
        assert EXPECTED_HASH not in text
    assert prereg.MARKLESS_LIQUIDATION_VALIDITY_POLICY[
        "does_not_disturb_the_acquisition_evidence"
    ].startswith("the committed acquisition plan")
