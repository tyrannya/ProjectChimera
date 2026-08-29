"""P5's answer, pinned to the numbers it actually produced.

The frozen cells are covered by `artifacts/btc_p5_SHA256SUMS.txt` and by the
coverage machinery in `tests/test_p2b_evidence.py`. What a checksum cannot say is
*what the checkpoint concluded*, so this file pins that: the four deciding fold
deltas, the improved-fold count, the outcome, and the classification of the
evidence — and it rebuilds the derived comparison and the derived decision and
checks they still say the same thing.

The failure this prevents is not a corrupted file. It is a later reader, or a
later aggregator, quietly producing a different answer from the same evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nn.p5_decision import (
    OUTCOME_NEGATIVE,
    build,
    load_cells,
    secondary_context,
)
from nn.p2b import DEFAULT_MANIFEST
from nn.p5_preregistration import (
    COMBINED,
    CONTROL,
    DECISION_RULE,
    PRIMARY_COMPARISON,
    PRIMARY_MODEL,
    preregistration_hash,
)

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "artifacts" / "benchmark"
DECISION = BENCH / "btc_p5_decision" / "decision.json"
COMPARISON = BENCH / "btc_p5_comparison" / "p2b_comparison.json"

#: What P5 answered, as numbers rather than as prose. Four folds, one improved.
DECIDING_DELTAS = [0.11508, -0.075359, -0.039844, -0.183647]
IMPROVED_FOLDS = 1
REQUIRED_FOLDS = 3
MEAN_DELTA = -0.0459425
WORST_FOLD_DELTA = -0.183647


@pytest.fixture(scope="module")
def decision() -> dict:
    return json.loads(DECISION.read_text())


@pytest.fixture(scope="module")
def comparison() -> dict:
    return json.loads(COMPARISON.read_text())


def test_the_deciding_fold_deltas_are_what_p5_produced(decision):
    """The four numbers the checkpoint turns on."""
    assert [row["delta"] for row in decision["decision"]["folds"]] == DECIDING_DELTAS


def test_one_of_four_folds_improved_against_a_bar_of_three(decision):
    record = decision["decision"]
    assert record["improved_folds"] == IMPROVED_FOLDS
    assert record["total_folds"] == 4
    assert (
        record["required_folds"] == REQUIRED_FOLDS == DECISION_RULE["improved_folds_required"]
    )
    assert record["passed"] is False


def test_p5_is_negative(decision):
    assert decision["decision"]["outcome"] == OUTCOME_NEGATIVE
    assert "Do not tune mtf_v1" in decision["decision"]["interpretation"]
    assert "Do not create mtf_v2" in decision["decision"]["interpretation"]


def test_the_deciding_cell_is_the_preregistered_one(decision):
    assert decision["decision"]["decided_by"] == {
        "model": PRIMARY_MODEL,
        "comparison": list(PRIMARY_COMPARISON),
    }
    assert PRIMARY_MODEL == "xgboost"
    assert tuple(PRIMARY_COMPARISON) == (COMBINED, CONTROL)


def test_the_mean_and_the_worst_fold_are_recorded_and_labelled_descriptive(decision):
    """They are reported in full, and they decided nothing in either direction."""
    descriptive = decision["decision"]["descriptive"]
    assert descriptive["mean_delta"] == MEAN_DELTA
    assert descriptive["worst_fold_delta"] == WORST_FOLD_DELTA
    assert "may not rescue" in descriptive["note"]
    assert "may not veto" in descriptive["note"]


def test_the_mean_did_not_decide_anything(decision):
    """A mean of −0.046 and a count of 1 of 4 agree here, and that is a coincidence.

    P2b had two arms with a *positive* mean while improving one and two of four
    folds. The count-the-folds rule is what decided P5, and this asserts the
    record says so rather than leaving a reader to assume the mean carried it.
    """
    record = decision["decision"]
    assert record["rule"]["statistic"].startswith("number of outer folds")
    assert record["rule"]["mean_delta"].startswith("descriptive")


def test_no_secondary_model_reached_the_bar_either(decision):
    """Stated as a fact about this run, not as a rule — the rule is that it could not.

    Logistic regression and LightGBM are context. Had one of them reached 3 of 4
    it would still not have switched the answer, and recording that none did keeps
    a reader from wondering whether the deciding model was chosen after the fact.
    """
    for row in decision["context"]:
        assert row["improved_folds"] < REQUIRED_FOLDS, row
        assert row["decides"] == (
            row["model"] == PRIMARY_MODEL and row["information_set"] == COMBINED
        )


def test_all_four_folds_were_available_and_the_gate_passed(decision):
    availability = decision["availability"]
    assert availability["folds_available"] == 4
    assert availability["gate_passed"] is True
    assert availability["rows_eligible"] == 44171
    assert availability["rows"] == 45802
    for block in availability["blocks"]:
        assert block["available"] is True
        assert block["inner_validation"]["eligible_fraction"] == 1.0
        assert block["outer_validation"]["eligible_fraction"] == 1.0


def test_the_thin_trade_flag_fired_and_changed_nothing(decision):
    """Fold 2's combined arm took 8 outer trades, below the 10 the flag names.

    The preregistration declared this a flag that changes no denominator, and the
    record has to show both halves: that it fired, and that the arithmetic above
    is over four folds regardless.
    """
    record = decision["decision"]
    assert record["trade_count_diagnostic"]["folds_flagged"] == [2]
    assert record["trade_count_diagnostic"]["effect_on_the_denominator"] == "none"
    assert record["total_folds"] == 4


def test_the_cells_were_produced_under_this_preregistration(decision):
    assert decision["identity"]["preregistration_hash"] == preregistration_hash()


def test_the_decision_recomputes_to_the_same_answer():
    """Rebuilt from the frozen cells, the rule gives the same result.

    Slow but load-bearing: it re-reads all nine cells, recomputes the sample
    universe from the snapshot and checks its digest, and re-applies the rule.
    A decision record that could not be reproduced from its own inputs would be a
    claim rather than a derivation.
    """
    rebuilt = build(_cell_dirs(), DEFAULT_MANIFEST)
    committed = json.loads(DECISION.read_text())
    assert rebuilt["decision"] == committed["decision"]
    assert rebuilt["availability"] == committed["availability"]
    assert rebuilt["identity"] == committed["identity"]


def _cell_dirs() -> list[Path]:
    from nn.information_sets import P5_INFORMATION_SETS
    from nn.simple_models import SIMPLE_MODEL_NAMES

    return [
        BENCH / f"btc_p5_{arm}_{model}"
        for arm in P5_INFORMATION_SETS
        for model in SIMPLE_MODEL_NAMES
    ]


def test_the_comparison_agrees_with_the_decision(comparison):
    """Two independent paths to the same four deltas.

    `nn.p2b_compare` computes them for every model-arm pair as part of a general
    aggregate; `nn.p5_decision` computes the deciding pair on its own and applies
    the rule. They read the same cells and must not disagree.
    """
    entry = comparison["deltas"][PRIMARY_MODEL][COMBINED]
    assert entry["vs"] == CONTROL
    assert [row["net_return"] for row in entry["per_fold"]] == DECIDING_DELTAS
    assert entry["net_return_improved_folds"] == IMPROVED_FOLDS
    assert entry["total_folds"] == 4
    assert "one of four" in entry["verdict"]


def test_the_comparison_is_derived_and_the_decision_is_not(comparison, decision):
    """Which of the two a checksum is allowed to cover."""
    assert comparison["evidence_class"] == "derived"
    assert decision["evidence_class"] != "derived"
    assert "the P5 outcome" in decision["evidence_class"]


def test_the_evidence_is_labelled_adaptive_rather_than_confirmatory(comparison):
    status = comparison["adaptive_status"]
    assert "adaptive" in status
    assert "cannot confirm" in status


def test_every_secondary_context_row_is_reproducible():
    """The six model-arm comparisons, recomputed from the cells."""
    rows = secondary_context(load_cells(_cell_dirs()))
    committed = json.loads(DECISION.read_text())["context"]
    assert rows == committed
    assert len(rows) == 6


def test_p4_hold_was_not_touched_by_p5():
    """P5 spent nothing that P4 retired."""
    ledger = json.loads((ROOT / "data" / "research" / "p4_holdout_ledger.json").read_text())
    assert ledger["state"] == "retired"
    assert ledger["checkpoint"] is None


def test_no_p5_cell_records_a_sealed_or_holdout_row(decision):
    """P5 adds no source, so it adds no new way to reach either region."""
    for directory in _cell_dirs():
        cell = json.loads((directory / "p2b.json").read_text())
        assert cell["sealed_test"] is False
        assert cell["snapshot"]["contains_styx"] is False
        assert cell["snapshot"]["rows"] == 45802
