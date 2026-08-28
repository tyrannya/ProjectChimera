from __future__ import annotations

import json
from pathlib import Path

import pytest

from nn.p4_holdout import RETIRED, read_ledger, stage_one_gate, styx_untouched
from tools import freeze_evidence

ROOT = Path(__file__).resolve().parent.parent
BENCHMARK = ROOT / "artifacts" / "benchmark"


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def test_p4_primary_cells_are_frozen():
    manifest = ROOT / "artifacts" / "btc_p4_stage1_SHA256SUMS.txt"
    entries = freeze_evidence.manifest_entries(manifest)

    assert len(entries) == 27
    assert freeze_evidence.check(manifest) == []

    directories = {Path(name).parent.name for _, name in entries}
    assert len(directories) == 9
    assert "btc_p4_comparison" not in directories
    assert "btc_p4_stage1" not in directories


def test_p4_official_screen_is_frozen():
    manifest = ROOT / "artifacts" / "btc_p4_screen_SHA256SUMS.txt"
    entries = freeze_evidence.manifest_entries(manifest)

    assert freeze_evidence.check(manifest) == []
    assert entries == [
        (
            entries[0][0],
            "artifacts/benchmark/btc_p4_stage1/stage1.json",
        )
    ]

    report = _json(BENCHMARK / "btc_p4_stage1" / "stage1.json")
    assert report["frozen_evidence"] == {
        "manifest": "artifacts/btc_p4_screen_SHA256SUMS.txt",
        "report_path": "artifacts/benchmark/btc_p4_stage1/stage1.json",
    }
    assert report["evidence_class"] == (
        "exploratory screen on burned blocks; not a research result"
    )


def test_p4_screen_used_only_the_three_available_blocks():
    report = _json(BENCHMARK / "btc_p4_stage1" / "stage1.json")

    availability = {
        tuple(block["block"]): block["available"]
        for block in report["availability"]["exploratory_blocks"]
    }
    assert availability == {
        (26518, 31339): False,
        (31339, 36160): True,
        (36160, 40981): True,
        (40981, 45802): True,
    }

    assert [fold["block"] for fold in report["folds"]] == [
        [31339, 36160],
        [36160, 40981],
        [40981, 45802],
    ]


def test_p4_stage1_screened_out_under_the_preregistered_rule():
    report = _json(BENCHMARK / "btc_p4_stage1" / "stage1.json")

    assert report["decided_by"] == {
        "model": "xgboost",
        "comparison": ["ohlcv14_plus_derivatives_v1", "ohlcv14"],
        "cost_multiplier": 1.0,
    }

    folds = report["folds"]

    assert [fold["control_trades"] for fold in folds] == [15, 17, 42]
    assert [fold["combined_trades"] for fold in folds] == [20, 72, 15]
    assert [fold["valid"] for fold in folds] == [True, True, True]
    assert [fold["improved"] for fold in folds] == [False, False, True]

    assert [fold["delta"] for fold in folds] == pytest.approx(
        [
            -0.046462,
            -0.09306799999999998,
            0.023066000000000003,
        ],
        abs=1e-15,
    )

    gate = stage_one_gate(report)
    assert gate["passed"] is False
    assert gate["valid_folds"] == 3
    assert gate["invalid_folds"] == 0
    assert gate["improved_folds"] == 1
    assert gate["mean_delta"] == pytest.approx(
        -0.038821333333333326, abs=1e-15
    )
    assert gate["worst_fold_delta"] == pytest.approx(
        -0.09306799999999998, abs=1e-15
    )

    screen = report["screen"]
    assert screen["passed"] is False
    assert screen["outcome"] == "screened_out"
    assert screen["valid_folds"] == 3
    assert screen["improved_folds"] == 1
    assert screen["mean_delta"] == pytest.approx(
        gate["mean_delta"], abs=1e-15
    )
    assert screen["worst_fold_delta"] == pytest.approx(
        gate["worst_fold_delta"], abs=1e-15
    )


def test_p4_holdout_is_retired_without_a_checkpoint():
    ledger = read_ledger(ROOT)

    assert ledger["state"] == RETIRED
    assert ledger["checkpoint"] is None
    assert ledger["retired_if_unspent"] is True
    assert "Stage 1 screened out" in ledger["reason"]
    assert "not opened, scored, or evaluated" in ledger["reason"]


def test_p4_closure_did_not_reach_styx():
    facts = styx_untouched()

    assert facts["sealed_from_row"] == 48217
    assert facts["reaches_sealed_rows"] is False


def test_p4_generic_comparison_remains_derived():
    comparison = _json(
        BENCHMARK / "btc_p4_comparison" / "p2b_comparison.json"
    )

    assert comparison["checkpoint"] == "P4"
    assert comparison["evidence_class"] == "derived"
