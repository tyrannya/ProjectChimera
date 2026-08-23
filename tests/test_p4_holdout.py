"""The one region nothing has scored, and the mechanism that keeps it that way.

Every test here is outcome-blind. None of them reads a price, a label, a return
or a prediction from rows ``[45802, 48211)``; the stage-1 reports they build are
synthetic and describe the *burned* exploratory blocks, which is the only thing
the release gate is allowed to look at. A guard that had to see the holdout to
decide whether the holdout may be seen would not be a guard.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nn import p4_holdout
from nn.p4_holdout import (
    LEDGER_SCHEMA,
    assert_stage_one_bound,
    RETIRED,
    SPENT,
    UNSPENT,
    HoldoutError,
    assert_holdout_release,
    assert_stage_one_rows,
    assert_stage_one_snapshot,
    read_ledger,
    record_retirement,
    record_spend,
    stage_one_gate,
    styx_untouched,
)
from nn.p4_preregistration import (
    HOLDOUT_ROWS,
    MIN_OUTER_TRADES,
    PRIMARY_COMPARISON,
    PRIMARY_MODEL,
    RESEARCH_ROWS,
    SECONDARY_MODELS,
    STAGE_1_MAX_ROW_EXCLUSIVE,
    preregistration_hash,
)

ROOT = Path(__file__).resolve().parent.parent


def _report(**overrides):
    """A stage-1 report that passes, describing the burned blocks only.

    Shaped like the one :func:`nn.p4_stage1.stage_one_report` writes, because the
    release gate now checks the shape as well as the numbers: which cell decided,
    and whether the report is frozen evidence on disk.
    """
    report = {
        "preregistration_hash": preregistration_hash(),
        "decided_by": {
            "model": PRIMARY_MODEL,
            "comparison": list(PRIMARY_COMPARISON),
            "cost_multiplier": 1.0,
        },
        "availability": {"gate_passed": True},
        "folds": [
            {
                "block": [26518, 31339],
                "delta": 0.01,
                "control_trades": 30,
                "combined_trades": 33,
            },
            {
                "block": [31339, 36160],
                "delta": 0.02,
                "control_trades": 25,
                "combined_trades": 28,
            },
            {
                "block": [36160, 40981],
                "delta": -0.01,
                "control_trades": 40,
                "combined_trades": 44,
            },
            {
                "block": [40981, 45802],
                "delta": 0.015,
                "control_trades": 18,
                "combined_trades": 20,
            },
        ],
    }
    report.update(overrides)
    return report


def _freeze(root, report):
    """Write ``report`` as frozen stage-1 evidence, the way the gate requires it.

    The report a caller hands the release gate must be a file a checksum manifest
    already covers, so the fixture has to produce one: a directory declaring
    itself primary evidence, and a manifest over it.
    """
    from tools import freeze_evidence

    cell = root / "artifacts" / "benchmark" / "btc_p4_stage1"
    cell.mkdir(parents=True, exist_ok=True)
    report = dict(report)
    report["frozen_evidence"] = {
        "manifest": "artifacts/btc_p4_stage1_SHA256SUMS.txt",
        "report_path": "artifacts/benchmark/btc_p4_stage1/stage1.json",
    }
    (cell / "stage1.json").write_text(json.dumps(report, indent=2) + "\n")
    manifest = root / "artifacts" / "btc_p4_stage1_SHA256SUMS.txt"
    manifest.write_text(
        f"{freeze_evidence.digest(cell / 'stage1.json')}  "
        "artifacts/benchmark/btc_p4_stage1/stage1.json\n"
    )
    return report


@pytest.fixture
def tree(tmp_path):
    """A repository root holding only the ledger, so writes are throwaway."""
    (tmp_path / "data" / "research").mkdir(parents=True)
    (tmp_path / p4_holdout.LEDGER_PATH).write_text((ROOT / p4_holdout.LEDGER_PATH).read_text())
    return tmp_path


# --- the committed ledger ----------------------------------------------------
def test_the_committed_ledger_says_the_region_is_unspent():
    ledger = read_ledger(ROOT)
    assert ledger["ledger_schema"] == LEDGER_SCHEMA
    assert ledger["region"] == list(HOLDOUT_ROWS)
    assert ledger["state"] == UNSPENT
    assert ledger["evaluations_permitted"] == 1
    assert ledger["checkpoints_permitted"] == 1
    assert ledger["retired_if_unspent"] is True


def test_a_missing_ledger_is_not_an_unspent_one(tmp_path):
    with pytest.raises(HoldoutError, match="no P4-HOLD ledger"):
        read_ledger(tmp_path)


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda p: p.__setitem__("ledger_schema", "other/1"), "schema"),
        (lambda p: p.__setitem__("region", [0, 10]), "governs rows"),
        (lambda p: p.__setitem__("state", "maybe"), "unknown state"),
    ],
)
def test_a_ledger_that_governs_something_else_is_refused(tree, mutate, match):
    payload = json.loads((tree / p4_holdout.LEDGER_PATH).read_text())
    mutate(payload)
    (tree / p4_holdout.LEDGER_PATH).write_text(json.dumps(payload))
    with pytest.raises(HoldoutError, match=match):
        read_ledger(tree)


# --- stage 1 cannot reach the region ----------------------------------------
def test_stage_one_rows_stop_at_the_holdout():
    assert_stage_one_rows([0, 45801, STAGE_1_MAX_ROW_EXCLUSIVE - 1], what="a fold plan")
    with pytest.raises(HoldoutError, match="P4-HOLD"):
        assert_stage_one_rows([STAGE_1_MAX_ROW_EXCLUSIVE], what="a fold plan")
    with pytest.raises(HoldoutError, match="P4-HOLD"):
        assert_stage_one_rows([0, 46000], what="a fold plan")


def test_the_exclusive_bound_and_the_row_index_are_not_confused():
    """[0, 45802) is the whole legal space; index 45802 is already inside."""
    assert_stage_one_bound(STAGE_1_MAX_ROW_EXCLUSIVE, what="a range")
    with pytest.raises(HoldoutError, match="P4-HOLD"):
        assert_stage_one_bound(STAGE_1_MAX_ROW_EXCLUSIVE + 1, what="a range")
    with pytest.raises(HoldoutError, match="P4-HOLD"):
        assert_stage_one_rows([STAGE_1_MAX_ROW_EXCLUSIVE], what="an index set")


def test_the_committed_snapshot_is_a_legal_stage_one_input():
    checked = assert_stage_one_snapshot(
        ROOT / "data" / "research" / "btc_usdt_1h_gen1_snapshot_manifest.json"
    )
    assert checked["row_range"] == [0, STAGE_1_MAX_ROW_EXCLUSIVE]


def test_a_snapshot_cut_one_block_longer_is_refused(tmp_path):
    """The fact this turns into a precondition.

    Today's snapshot stops at 45802 and therefore cannot reach the holdout. A
    later export cut longer would hand the screen the rows it is screening in
    order to decide whether to open, and nothing would have said so.
    """
    source = ROOT / "data" / "research" / "btc_usdt_1h_gen1_snapshot_manifest.json"
    payload = json.loads(source.read_text())
    payload["processed_outer_coverage"]["rows"] = HOLDOUT_ROWS[1]
    payload["processed_outer_coverage"]["row_range"] = [0, HOLDOUT_ROWS[1]]
    longer = tmp_path / "longer_manifest.json"
    longer.write_text(json.dumps(payload))
    with pytest.raises(HoldoutError, match="P4-HOLD"):
        assert_stage_one_snapshot(longer)


# --- the release gate --------------------------------------------------------
def test_a_passing_report_opens_the_region(tree):
    release = assert_holdout_release("P4", _freeze(tree, _report()), root=tree)
    assert release["region"] == list(HOLDOUT_ROWS)
    assert release["preregistration_hash"] == preregistration_hash()
    assert "never confirmatory" in release["label_ceiling"]


def test_a_report_under_another_preregistration_does_not_open_it(tree):
    with pytest.raises(HoldoutError, match="preregistration"):
        assert_holdout_release(
            "P4", _freeze(tree, _report(preregistration_hash="0" * 64)), root=tree
        )


def test_a_failed_availability_gate_does_not_open_it(tree):
    report = _report()
    report["availability"] = {"gate_passed": False}
    with pytest.raises(HoldoutError, match="availability gate"):
        assert_holdout_release("P4", _freeze(tree, report), root=tree)


def test_a_report_with_no_availability_block_does_not_open_it(tree):
    report = _report()
    report.pop("availability")
    with pytest.raises(HoldoutError, match="availability gate"):
        assert_holdout_release("P4", _freeze(tree, report), root=tree)


def test_a_screen_that_did_not_pass_does_not_open_it(tree):
    report = _report()
    report["folds"][0]["delta"] = -0.005
    report["folds"][1]["delta"] = -0.005
    with pytest.raises(HoldoutError, match="stage 1 did not pass"):
        assert_holdout_release("P4", _freeze(tree, report), root=tree)


def test_thin_folds_do_not_count_toward_the_screen(tree):
    report = _report()
    for fold in report["folds"][:2]:
        fold["control_trades"] = MIN_OUTER_TRADES - 1
    with pytest.raises(HoldoutError, match="valid fold"):
        assert_holdout_release("P4", _freeze(tree, report), root=tree)


def test_a_report_that_is_not_frozen_evidence_does_not_open_it(tree):
    """§9.3: the holdout opens on a *frozen* stage-1 pass, not on a dict."""
    with pytest.raises(HoldoutError, match="declares no frozen_evidence"):
        assert_holdout_release("P4", _report(), root=tree)


def test_a_report_edited_after_it_was_frozen_does_not_open_it(tree):
    """The frozen file is the evidence; a copy of it with better numbers is not."""
    frozen = _freeze(tree, _report())
    edited = json.loads(json.dumps(frozen))
    edited["folds"][2]["delta"] = 0.5
    with pytest.raises(HoldoutError, match="is not what"):
        assert_holdout_release("P4", edited, root=tree)


def test_a_report_its_manifest_does_not_cover_does_not_open_it(tree):
    frozen = _freeze(tree, _report())
    (tree / "artifacts" / "btc_p4_stage1_SHA256SUMS.txt").write_text(
        "0" * 64 + "  artifacts/benchmark/btc_p4_stage1/other.json\n"
    )
    with pytest.raises(HoldoutError, match="does not verify|is not covered"):
        assert_holdout_release("P4", frozen, root=tree)


def test_a_secondary_models_success_does_not_open_it(tree):
    """§10: the two secondary families are reported in full and decide nothing."""
    report = _report()
    report["decided_by"]["model"] = SECONDARY_MODELS[0]
    with pytest.raises(HoldoutError, match="cannot open this region"):
        assert_holdout_release("P4", _freeze(tree, report), root=tree)


def test_a_near_miss_at_a_higher_cost_multiplier_does_not_open_it(tree):
    """§7.2: the base cost decides, and no other multiplier is a rescue."""
    report = _report()
    report["decided_by"]["cost_multiplier"] = 1.5
    with pytest.raises(HoldoutError, match="base cost decides"):
        assert_holdout_release("P4", _freeze(tree, report), root=tree)


def test_a_report_from_the_standalone_arms_comparison_does_not_open_it(tree):
    report = _report()
    report["decided_by"]["comparison"] = ["derivatives_v1", "ohlcv14"]
    with pytest.raises(HoldoutError, match="One comparison decides"):
        assert_holdout_release("P4", _freeze(tree, report), root=tree)


# --- improved means delta > 0 ------------------------------------------------
def test_a_zero_delta_is_not_an_improvement():
    """The tie that had to be settled before it happened rather than after."""
    report = _report()
    for fold in report["folds"]:
        fold["delta"] = 0.0
    gate = stage_one_gate(report)
    assert gate["improved_folds"] == 0
    assert gate["passed"] is False
    assert any("delta > 0" in reason for reason in gate["reasons"])


def test_the_smallest_positive_delta_is_an_improvement():
    report = _report()
    for fold in report["folds"]:
        fold["delta"] = 1e-9
    gate = stage_one_gate(report)
    assert gate["improved_folds"] == 4


def test_a_positive_mean_carried_by_one_fold_does_not_pass(tree):
    """P2b's shape, and the reason the screen is three conditions and not one."""
    report = _report()
    deltas = [0.30, -0.01, -0.01, -0.01]
    for fold, delta in zip(report["folds"], deltas):
        fold["delta"] = delta
    gate = stage_one_gate(report)
    assert gate["mean_delta"] > 0
    assert gate["passed"] is False


def test_one_catastrophic_fold_fails_the_screen_despite_three_wins(tree):
    report = _report()
    deltas = [0.01, 0.01, 0.01, -0.5]
    for fold, delta in zip(report["folds"], deltas):
        fold["delta"] = delta
    gate = stage_one_gate(report)
    assert gate["improved_folds"] == 3
    assert gate["passed"] is False
    assert any("worst valid fold" in reason for reason in gate["reasons"])


# --- once, by one checkpoint, ever -------------------------------------------
def test_the_region_can_be_spent_exactly_once(tree):
    release = assert_holdout_release("P4", _freeze(tree, _report()), root=tree)
    ledger = record_spend(release, root=tree)
    assert ledger["state"] == SPENT
    assert ledger["checkpoint"] == "P4"

    with pytest.raises(HoldoutError, match="spent"):
        assert_holdout_release("P4", _freeze(tree, _report()), root=tree)
    with pytest.raises(HoldoutError, match="spent"):
        record_spend(release, root=tree)


def test_a_second_checkpoint_cannot_spend_it_either(tree):
    record_spend(assert_holdout_release("P4", _freeze(tree, _report()), root=tree), root=tree)
    with pytest.raises(HoldoutError, match="spent"):
        assert_holdout_release("P4b", _freeze(tree, _report()), root=tree)


def test_an_unspent_region_is_retired_anyway(tree):
    """The rule a later checkpoint would otherwise argue its way around.

    P4's stage-1 numbers are published whether or not the holdout is opened, so
    a design that retunes against them has made these rows adaptive. Retirement
    records that, and it closes the region to a P4b claiming a fresh holdout.
    """
    ledger = record_retirement("P4 stopped at stage 1", root=tree)
    assert ledger["state"] == RETIRED
    with pytest.raises(HoldoutError, match="retired"):
        assert_holdout_release("P4b", _freeze(tree, _report()), root=tree)


def test_a_spent_region_cannot_be_retired_into_reuse(tree):
    record_spend(assert_holdout_release("P4", _freeze(tree, _report()), root=tree), root=tree)
    with pytest.raises(HoldoutError, match="spent"):
        record_retirement("trying to reset", root=tree)


# --- and Styx is somewhere else ---------------------------------------------
def test_the_holdout_never_reaches_a_sealed_row():
    facts = styx_untouched()
    assert facts["reaches_sealed_rows"] is False
    assert facts["label_of_last_holdout_row_closes_at"] == RESEARCH_ROWS - 1
    assert HOLDOUT_ROWS[1] + facts["horizon"] == RESEARCH_ROWS
