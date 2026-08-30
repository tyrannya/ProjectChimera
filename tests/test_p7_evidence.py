"""P7's committed evidence, checked against the design it was produced under.

The load-bearing test is
:func:`test_every_verdict_recomputes_from_the_frozen_p6_predictions`. It reads no
delta, no summary and no verdict: it re-derives each mode's four fold deltas from
the frozen P6 prediction files — realigning, re-deciding through
`chimera.consensus.decide`, re-scoring through `nn.evaluate.trading_metrics`,
and recomputing the fold-wise best constituent — then applies the two
preregistered conditions itself and asserts the answer is the published one.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from chimera.contracts import TargetSpec
from nn.p6_preregistration import HORIZON_BARS
from nn.p6_preregistration import preregistration_hash as p6_hash
from nn.p7 import (
    ARTIFACT_NAME,
    aligned_actions,
    consensus_signals,
    constituent_signals,
    load_specialist,
    out_name,
    rule_for,
    score,
)
from nn.p7_decision import DECISION_NAME, VERDICT_NEGATIVE, VERDICT_SUPPORTIVE
from nn.p7_preregistration import COSTS, DECISION_RULE, MODES, preregistration_hash

REPO = Path(__file__).resolve().parents[1]
BENCHMARK = REPO / "artifacts" / "benchmark"
DECISION_DIR = BENCHMARK / "btc_p7_decision"
MANIFEST = REPO / "artifacts" / "btc_p7_SHA256SUMS.txt"

MODE_IDS = [design["mode"] for design in MODES]


@pytest.fixture(scope="module")
def modes() -> dict:
    return {
        design["mode"]: json.loads((BENCHMARK / out_name(design) / ARTIFACT_NAME).read_text())
        for design in MODES
    }


@pytest.fixture(scope="module")
def decision() -> dict:
    return json.loads((DECISION_DIR / DECISION_NAME).read_text())


# --------------------------------------------------------------------------- #
# A. the evidence is what it claims to be
# --------------------------------------------------------------------------- #


def test_both_modes_are_committed(modes):
    assert set(modes) == set(MODE_IDS)


@pytest.mark.parametrize("name", MODE_IDS)
def test_each_mode_declares_the_frozen_design(modes, name):
    payload = modes[name]
    assert payload["checkpoint"] == "P7"
    assert payload["preregistration_hash"] == preregistration_hash()
    assert payload["specialist_source"]["p6_preregistration_hash"] == p6_hash()
    assert payload["specialist_source"]["model"] == "xgboost"
    assert payload["target"]["horizon"] == HORIZON_BARS
    assert payload["target"]["fee_rate"] == COSTS["fee_rate"]
    assert payload["target"]["slippage_rate"] == COSTS["slippage_rate"]
    assert payload["consensus_rule"]["decided_by"] == "chimera.consensus.decide"
    assert len(payload["folds"]) == 4


@pytest.mark.parametrize("name", MODE_IDS)
def test_the_validity_gate_passed_on_every_mode(modes, name):
    gate = modes[name]["validity_gate"]
    assert gate["passed"] is True
    assert gate["decision_clock"] == modes[name]["mode"]["decision_clock"]
    assert gate["identity_rows"] > 0


@pytest.mark.parametrize("name", MODE_IDS)
def test_no_specialist_was_refitted(modes, name):
    """P7's evidence names the frozen cells and carries no fit of its own."""
    payload = modes[name]
    assert "seed" not in payload
    assert payload["specialist_source"]["cells"].startswith("artifacts/benchmark/btc_p6_")
    for record in payload["folds"]:
        assert set(record["constituents"]) == set(payload["consensus_rule"]["specialists"])


def test_the_p6_specialists_are_still_frozen():
    """P7 replays cells that P6's manifest still vouches for."""
    from tools.freeze_evidence import check

    assert check(REPO / "artifacts" / "btc_p6_SHA256SUMS.txt") == []


# --------------------------------------------------------------------------- #
# B. the verdicts, recomputed from the frozen predictions
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("design", MODES, ids=MODE_IDS)
def test_every_verdict_recomputes_from_the_frozen_p6_predictions(decision, design):
    rule = rule_for(design)
    spec = TargetSpec(
        horizon=HORIZON_BARS,
        fee_rate=COSTS["fee_rate"],
        slippage_rate=COSTS["slippage_rate"],
    )
    specialists = {clock: load_specialist(clock) for clock in rule.specialists}
    decision_all = specialists[rule.decision_clock]

    deltas: list[float] = []
    for fold in sorted(decision_all["fold"].unique()):
        rows = decision_all.loc[decision_all["fold"] == fold].reset_index(drop=True)
        actions = {
            clock: aligned_actions(
                rows,
                frame.loc[frame["fold"] == fold].reset_index(drop=True),
                rule.decision_clock,
                clock,
            )
            for clock, frame in specialists.items()
        }
        consensus = score(consensus_signals(actions, rule), rows, spec)["net_return"]
        best = max(
            score(constituent_signals(values), rows, spec)["net_return"]
            for values in actions.values()
        )
        deltas.append(round(consensus - best, 6))

    improved = sum(1 for value in deltas if value > 0.0)
    mean = float(np.mean(deltas))
    supportive = improved >= DECISION_RULE["improved_folds_required"] and mean > 0.0

    published = next(row for row in decision["modes"] if row["mode"] == design["mode"])
    assert [record["delta"] for record in published["folds"]] == deltas
    assert published["conditions"]["improved_folds"]["observed"] == improved
    assert published["conditions"]["mean_fold_delta"]["observed"] == pytest.approx(
        mean, abs=1e-9
    )
    assert published["verdict"] == (VERDICT_SUPPORTIVE if supportive else VERDICT_NEGATIVE)


def test_the_decision_reports_both_modes_separately(decision):
    assert [row["mode"] for row in decision["modes"]] == MODE_IDS
    assert decision["preregistration_hash"] == preregistration_hash()
    assert "best_mode" not in decision
    supportive = decision["supportive_modes"]
    if not supportive:
        assert decision["outcome"] == "neither supportive"
    elif len(supportive) == len(MODE_IDS):
        assert decision["outcome"] == "both modes supportive"
    else:
        assert decision["outcome"].endswith("supportive only")


def test_the_rule_applied_is_the_preregistered_one(decision):
    assert decision["rule"]["improved_folds_required"] == 3
    assert decision["rule"]["total_folds"] == 4
    assert decision["rule"]["conjunction"] == "both"
    for row in decision["modes"]:
        assert set(row["conditions"]) == {"improved_folds", "mean_fold_delta"}


@pytest.mark.parametrize("name", MODE_IDS)
def test_the_consensus_never_traded_a_row_it_could_not_see(modes, name):
    """Unavailable specialists cost only block heads, as preregistered."""
    payload = modes[name]
    for record in payload["folds"]:
        own = payload["mode"]["decision_clock"]
        assert record["unavailable_rows"][own] == 0
        for clock, count in record["unavailable_rows"].items():
            assert count <= 14, f"{clock} unavailable on {count} rows, more than a block head"


# --------------------------------------------------------------------------- #
# C. the manifest
# --------------------------------------------------------------------------- #


def test_the_primary_evidence_is_frozen_and_still_hashes():
    from tools.freeze_evidence import check

    assert MANIFEST.is_file()
    assert check(MANIFEST) == []


def test_the_manifest_covers_every_mode_artifact():
    covered = {
        line.split(maxsplit=1)[1].strip()
        for line in MANIFEST.read_text().splitlines()
        if line.strip()
    }
    for design in MODES:
        directory = (BENCHMARK / out_name(design)).relative_to(REPO)
        assert f"{directory}/{ARTIFACT_NAME}" in covered
