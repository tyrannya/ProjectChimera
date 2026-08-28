"""P4 stage 1: the interlock, the screen, and the two things that cannot decide.

``docs/p4_preregistration.md`` §8.2 and §10 run nine cells and let one decide.
These tests are about the *mechanism* that makes that true — the secondary models
and the cost multipliers being unable to reach the holdout — and about the
interlock that keeps the whole matrix from running before someone says it may.

Nothing here fits a model. The interlock is exercised by being refused, which is
the state it must be in until a commit changes it.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from chimera.contracts import TargetSpec
from nn.p4_holdout import HoldoutError, assert_primary_decision
from nn.p4_preregistration import (
    ARMS,
    COST_SENSITIVITY_MULTIPLIERS,
    EXPLORATORY_OUTER_BLOCKS,
    MIN_OUTER_TRADES,
    PRIMARY_COMPARISON,
    PRIMARY_MODEL,
    SECONDARY_MODELS,
    preregistration_hash,
)
from nn.p4_stage1 import (
    AUTHORISATION_PATH,
    AUTHORISED,
    BASE_COST_MULTIPLIER,
    STAGE_1_CELLS,
    BlockResult,
    Stage1Error,
    Stage1Interlock,
    assert_fit_authorised,
    assert_stage_one_geometry,
    cells_to_run,
    cost_sensitivity,
    describe,
    net_return_at_cost,
    read_authorisation,
    scaled_target_spec,
    screen,
    stage_one_report,
)
from nn.dataset import Split
from nn.walkforward import plan_nested_folds

PASSING_AVAILABILITY = {
    "gate_passed": True,
    "exploratory_blocks": [
        {"block": list(block), "available": True} for block in EXPLORATORY_OUTER_BLOCKS
    ],
}


def _blocks(deltas, *, trades=30):
    """One deciding-comparison result per preregistered outer block."""
    out = []
    for block, delta in zip(EXPLORATORY_OUTER_BLOCKS, deltas):
        control, combined = (trades, trades) if isinstance(trades, int) else trades[len(out)]
        out.append(
            BlockResult(
                block=tuple(block),
                control_net_return=0.0,
                combined_net_return=delta,
                control_trades=control,
                combined_trades=combined,
            )
        )
    return out


def _report(deltas, *, trades=30, availability=None):
    return stage_one_report(
        _blocks(deltas, trades=trades),
        availability=availability or PASSING_AVAILABILITY,
        universe={"universe_hash": "0" * 64},
        provenance={"derivatives_spec_hash": "1" * 64},
    )


# --- the interlock ------------------------------------------------------------
def test_the_committed_interlock_authorises_only_the_active_design():
    payload = read_authorisation()
    assert payload["state"] == "authorised"
    assert payload["preregistration_hash"] == preregistration_hash()
    assert payload["authorised_by"]
    assert payload["authorised_at"]
    assert payload["reason"]


def test_committed_authorisation_still_requires_runtime_gates():
    with pytest.raises(Stage1Interlock, match="did not confirm"):
        assert_fit_authorised(confirm=False, availability=PASSING_AVAILABILITY)
    with pytest.raises(Stage1Interlock, match="not_evaluable"):
        assert_fit_authorised(confirm=True, availability={"gate_passed": False})


def test_command_line_confirmation_cannot_open_a_closed_interlock(tmp_path):
    root = tmp_path / "tree"
    (root / "data" / "research").mkdir(parents=True)
    payload = json.loads(AUTHORISATION_PATH.read_text())
    payload.update(
        {
            "state": "not_authorised",
            "authorised_by": None,
            "authorised_at": None,
            "reason": None,
        }
    )
    (root / AUTHORISATION_PATH).write_text(json.dumps(payload, indent=2))
    with pytest.raises(Stage1Interlock, match="not_authorised"):
        assert_fit_authorised(confirm=True, availability=PASSING_AVAILABILITY, root=root)


def _authorise(tmp_path, **overrides):
    root = tmp_path / "tree"
    (root / "data" / "research").mkdir(parents=True)
    payload = json.loads((AUTHORISATION_PATH).read_text())
    payload.update({"state": AUTHORISED, "authorised_by": "a human", **overrides})
    (root / AUTHORISATION_PATH).write_text(json.dumps(payload, indent=2))
    return root


def test_the_committed_file_alone_does_not_authorise_a_fit_either(tmp_path):
    root = _authorise(tmp_path)
    with pytest.raises(Stage1Interlock, match="did not confirm"):
        assert_fit_authorised(confirm=False, availability=PASSING_AVAILABILITY, root=root)


def test_both_gates_together_authorise_a_fit(tmp_path):
    root = _authorise(tmp_path)
    record = assert_fit_authorised(confirm=True, availability=PASSING_AVAILABILITY, root=root)
    assert record["state"] == AUTHORISED
    assert record["authorised_by"] == "a human"
    assert record["availability_gate_passed"] is True


def test_an_authorisation_granted_under_another_preregistration_does_not_carry_over(tmp_path):
    root = _authorise(tmp_path, preregistration_hash="0" * 64)
    with pytest.raises(Stage1Interlock, match="Permission to run one design"):
        assert_fit_authorised(confirm=True, availability=PASSING_AVAILABILITY, root=root)


def test_a_failed_availability_gate_stops_the_fit_even_when_authorised(tmp_path):
    """§9.0: insufficient coverage is not a licence to screen what did survive."""
    root = _authorise(tmp_path)
    with pytest.raises(Stage1Interlock, match="not_evaluable"):
        assert_fit_authorised(confirm=True, availability={"gate_passed": False}, root=root)


def test_an_absent_availability_gate_is_not_a_passing_one(tmp_path):
    root = _authorise(tmp_path)
    with pytest.raises(Stage1Interlock):
        assert_fit_authorised(confirm=True, availability=None, root=root)


def test_a_missing_interlock_file_is_not_an_open_one(tmp_path):
    with pytest.raises(Stage1Interlock, match="missing interlock is not an open one"):
        assert_fit_authorised(confirm=True, availability=PASSING_AVAILABILITY, root=tmp_path)


def test_an_interlock_under_another_schema_does_not_authorise_anything(tmp_path):
    root = tmp_path / "tree"
    (root / "data" / "research").mkdir(parents=True)
    (root / AUTHORISATION_PATH).write_text(
        json.dumps({"authorisation_schema": "other/1", "state": AUTHORISED})
    )
    with pytest.raises(Stage1Interlock, match="declares schema"):
        assert_fit_authorised(confirm=True, availability=PASSING_AVAILABILITY, root=root)


def test_describe_reports_authorisation_without_running_a_fit():
    described = describe(PASSING_AVAILABILITY)
    assert described["fits_run"] == 0
    assert described["interlock"]["state"] == "authorised"
    assert described["fit_would_be_refused_because"] is None


# --- the matrix ---------------------------------------------------------------
def test_the_matrix_is_three_arms_by_three_models():
    assert len(STAGE_1_CELLS) == 9
    assert {arm for arm, _ in STAGE_1_CELLS} == set(ARMS)
    assert {model for _, model in STAGE_1_CELLS} == {PRIMARY_MODEL, *SECONDARY_MODELS}


def test_exactly_one_of_the_nine_cells_decides():
    deciding = [cell for cell in cells_to_run() if cell["decides"]]
    assert len(deciding) == 1
    assert deciding[0]["information_set"] == PRIMARY_COMPARISON[0]
    assert deciding[0]["model"] == PRIMARY_MODEL


def test_every_secondary_cell_is_labelled_descriptive():
    for cell in cells_to_run():
        if cell["model"] != PRIMARY_MODEL:
            assert cell["decides"] is False
            assert "descriptive" in cell["role"]


def test_the_fold_plan_is_the_preregistered_geometry_and_stops_before_the_holdout():
    folds = plan_nested_folds(48217, 4, int(48217 * 0.45), 4821, 4821, 4821)
    bound = assert_stage_one_geometry(folds)
    assert bound["matches_preregistered_blocks"] is True
    assert bound["outer_blocks"] == [list(block) for block in EXPLORATORY_OUTER_BLOCKS]


def test_a_fold_plan_one_row_past_the_holdout_is_refused():
    """The one-row overrun, at the level that decides which rows are read."""
    folds = list(plan_nested_folds(48217, 4, int(48217 * 0.45), 4821, 4821, 4821))
    last = folds[-1]
    folds[-1] = replace(
        last, outer=Split(name=last.outer.name, start=last.outer.start, end=45803)
    )
    with pytest.raises(HoldoutError, match="P4-HOLD"):
        assert_stage_one_geometry(folds)


# --- the screen ---------------------------------------------------------------
def test_a_passing_screen_continues_to_stage_two():
    verdict = screen(_report([0.01, 0.02, 0.015, 0.005]))
    assert verdict["passed"] is True
    assert verdict["outcome"] == "continue_to_stage_2"
    assert "not a finding" in verdict["classification"]


def test_a_zero_delta_is_not_an_improvement():
    """§8.2, settled before it happened rather than after."""
    verdict = screen(_report([0.0, 0.0, 0.0, 0.02]))
    assert verdict["improved_folds"] == 1
    assert verdict["passed"] is False


def test_the_smallest_positive_delta_is_an_improvement():
    verdict = screen(_report([1e-12, 1e-12, 1e-12, 1e-12]))
    assert verdict["improved_folds"] == 4


def test_a_thin_fold_is_excluded_from_both_the_numerator_and_the_denominator():
    """§8.4: *each* compared arm must realise ten outer trades."""
    trades = [(30, 30), (30, 30), (MIN_OUTER_TRADES - 1, 30), (30, 30)]
    report = _report([0.01, 0.02, 5.0, 0.015], trades=trades)
    verdict = screen(report)
    assert verdict["valid_folds"] == 3
    assert verdict["invalid_folds"] == 1
    # The invalid fold's enormous delta reaches neither the count nor the mean.
    assert verdict["improved_folds"] == 3
    assert verdict["mean_delta"] == pytest.approx((0.01 + 0.02 + 0.015) / 3)
    assert verdict["passed"] is True


def test_an_invalid_fold_is_still_reported():
    trades = [(30, 30), (30, 30), (2, 30), (30, 30)]
    report = _report([0.01, 0.02, 0.03, 0.015], trades=trades)
    assert len(report["folds"]) == 4
    assert [fold["valid"] for fold in report["folds"]] == [True, True, False, True]


def test_two_valid_folds_cannot_pass_however_good_they_are():
    trades = [(30, 30), (30, 30), (1, 1), (1, 1)]
    verdict = screen(_report([1.0, 1.0, 0.0, 0.0], trades=trades))
    assert verdict["passed"] is False
    assert "valid fold" in verdict["reasons"][0]


def test_a_positive_mean_carried_by_one_fold_does_not_pass():
    """The composite screen, not a fold count: §8.2's condition 3 bites here."""
    verdict = screen(_report([0.5, -0.01, -0.01, -0.01]))
    assert verdict["improved_folds"] == 1
    assert verdict["mean_delta"] > 0
    assert verdict["passed"] is False


def test_three_wins_and_one_catastrophe_does_not_pass():
    verdict = screen(_report([0.01, 0.01, 0.01, -0.5]))
    assert verdict["improved_folds"] == 3
    assert verdict["passed"] is False
    assert any("worst valid fold" in reason for reason in verdict["reasons"])


def test_a_block_the_gate_called_unavailable_may_not_be_a_fold_of_the_screen():
    """§8.0: an unavailable block is excluded entirely, never at a discount."""
    availability = {
        "gate_passed": True,
        "exploratory_blocks": [
            {"block": list(block), "available": index > 0}
            for index, block in enumerate(EXPLORATORY_OUTER_BLOCKS)
        ],
    }
    with pytest.raises(Stage1Error, match="never included at a discount"):
        _report([0.01, 0.02, 0.015, 0.005], availability=availability)


# --- the two things that cannot decide ----------------------------------------
def test_a_report_decided_by_a_secondary_model_is_refused():
    report = _report([0.01, 0.02, 0.015, 0.005])
    report["decided_by"]["model"] = SECONDARY_MODELS[0]
    with pytest.raises(HoldoutError, match="cannot open this region"):
        assert_primary_decision(report)


@pytest.mark.parametrize("model", SECONDARY_MODELS)
def test_every_secondary_model_is_refused_by_name(model):
    report = _report([0.01, 0.02, 0.015, 0.005])
    report["decided_by"]["model"] = model
    with pytest.raises(HoldoutError):
        assert_primary_decision(report)


def test_a_report_decided_by_another_comparison_is_refused():
    report = _report([0.01, 0.02, 0.015, 0.005])
    report["decided_by"]["comparison"] = ["derivatives_v1", "ohlcv14"]
    with pytest.raises(HoldoutError, match="One comparison decides"):
        assert_primary_decision(report)


@pytest.mark.parametrize("multiplier", [m for m in COST_SENSITIVITY_MULTIPLIERS if m != 1.0])
def test_a_report_decided_at_another_cost_multiplier_is_refused(multiplier):
    """§7.2: there is no outcome in which a worse base-cost result is redeemed."""
    report = _report([0.01, 0.02, 0.015, 0.005])
    report["decided_by"]["cost_multiplier"] = multiplier
    with pytest.raises(HoldoutError, match="base cost decides"):
        assert_primary_decision(report)


def test_a_report_that_names_no_deciding_cell_is_refused():
    report = _report([0.01, 0.02, 0.015, 0.005])
    del report["decided_by"]
    with pytest.raises(HoldoutError):
        assert_primary_decision(report)


def test_a_report_this_module_builds_is_one_the_holdout_door_accepts():
    report = _report([0.01, 0.02, 0.015, 0.005])
    decision = assert_primary_decision(report)
    assert decision["model"] == PRIMARY_MODEL
    assert decision["cost_multiplier"] == BASE_COST_MULTIPLIER


# --- cost sensitivity ---------------------------------------------------------
def _signals(n=200, every=20):
    signals = np.ones(n, dtype=np.int64)  # HOLD
    signals[::every] = 2  # LONG
    return signals


def test_scaling_the_cost_moves_neither_the_horizon_nor_the_split():
    spec = TargetSpec(horizon=6, fee_rate=0.0005, slippage_rate=0.0005)
    scaled = scaled_target_spec(spec, 2.0)
    assert scaled.horizon == spec.horizon
    assert scaled.cost_threshold == pytest.approx(2.0 * spec.cost_threshold)


def test_a_higher_cost_takes_the_same_trades_and_nets_less():
    spec = TargetSpec(horizon=6)
    signals = _signals()
    returns = np.full(len(signals), 0.01)
    base = net_return_at_cost(signals, returns, spec, 1.0)
    doubled = net_return_at_cost(signals, returns, spec, 2.0)
    assert base["n_trades"] == doubled["n_trades"] > 0
    assert doubled["net_return"] < base["net_return"]


def test_only_the_base_multiplier_is_marked_as_deciding():
    spec = TargetSpec(horizon=6)
    result = cost_sensitivity(_signals(), np.full(200, 0.01), spec)
    assert [row["cost_multiplier"] for row in result["multipliers"]] == list(
        COST_SENSITIVITY_MULTIPLIERS
    )
    assert [row["decides"] for row in result["multipliers"]].count(True) == 1
    assert result["headline"]["cost_multiplier"] == BASE_COST_MULTIPLIER


def test_every_preregistered_multiplier_is_reported():
    spec = TargetSpec(horizon=6)
    result = cost_sensitivity(_signals(), np.full(200, 0.01), spec)
    assert len(result["multipliers"]) == len(COST_SENSITIVITY_MULTIPLIERS)


def test_a_cost_multiplier_of_zero_is_not_a_cost():
    with pytest.raises(Stage1Error, match="is not a cost"):
        scaled_target_spec(TargetSpec(), 0.0)
