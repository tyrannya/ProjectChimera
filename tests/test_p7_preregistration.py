"""P7's preregistration, held to the things it claims about itself.

As with P6, two kinds of test: the design is internally coherent and matches the
document beside it, and a **tripwire** asserting that while this file is the
whole of P7, no P7 artifact exists. Registration is not permission.

One test here is specific to P7 and load-bearing:
:func:`test_p7_was_registered_after_p6_closed` checks the chronology the whole
checkpoint depends on — P7 names P6's preregistration hash and P6's evidence is
committed, so a P7 designed before P6 closed could not have been written.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from nn.p6_preregistration import preregistration_hash as p6_hash
from nn.p7_preregistration import (
    ALIGNMENT,
    CHECKPOINT,
    CONSENSUS_VERSION,
    COSTS,
    DAY_TRADING,
    DECISION_RULE,
    FORBIDDEN_AFTER_RESULTS,
    LEAKAGE_BATTERY,
    MEASURED_AVAILABILITY,
    MODES,
    SCALPING,
    SPECIALIST_SOURCE,
    UNAVAILABILITY_RULE,
    VALIDITY_GATE,
    describe,
    mode,
    payload,
    preregistration_hash,
)

REPO = Path(__file__).resolve().parents[1]
DOCUMENT = REPO / "docs" / "p7_preregistration.md"

FROZEN_HASH = "sha256:b79365ce0c6a8d1464b2420e589fd502cf3520daba775c2da5430497be12cb50"


def _flat(text: str) -> str:
    """Markdown prose as one comparable line: no wrapping, emphasis, quoting or code."""
    stripped = text.replace("`", "").replace("*", "").replace(">", " ")
    return re.sub(r"\s+", " ", stripped).lower()


# --------------------------------------------------------------------------- #
# A. chronology
# --------------------------------------------------------------------------- #


def test_p7_was_registered_after_p6_closed():
    """P7 stands on P6's frozen evidence, and says which P6 it stands on."""
    assert SPECIALIST_SOURCE["p6_preregistration_hash"] == p6_hash()
    assert (REPO / "artifacts" / "btc_p6_SHA256SUMS.txt").is_file()
    decision = REPO / "artifacts" / "benchmark" / "btc_p6_decision" / "decision.json"
    assert decision.is_file(), "P7 may not be registered before P6's decision exists"
    p6 = json.loads(decision.read_text())
    assert p6["preregistration_hash"] == p6_hash()
    # P6 was negative, and P7 says so rather than quietly depending on it.
    assert p6["outcome"] == "negative"


def test_p7_reads_frozen_predictions_and_refits_nothing():
    assert SPECIALIST_SOURCE["model"] == "xgboost"
    assert SPECIALIST_SOURCE["column"] == "selected_action"
    flat = _flat(payload()["refit_prohibition"])
    for phrase in ("no specialist is refitted", "no threshold is re-selected"):
        assert phrase in flat


def test_the_specialist_sets_are_architectural_not_chosen():
    flat = _flat(payload()["specialist_set_is_architectural"])
    assert "not by p6's results" in flat


# --------------------------------------------------------------------------- #
# B. the design
# --------------------------------------------------------------------------- #


def test_the_two_modes_are_frozen():
    assert [item["mode"] for item in MODES] == ["SCALPING", "DAY_TRADING"]
    assert SCALPING["decision_clock"] == "1m"
    assert SCALPING["specialists"] == ["1m", "5m", "15m"]
    assert SCALPING["veto_specialist"] == "15m"
    assert (SCALPING["agreement_required"], SCALPING["of"]) == (2, 3)
    assert DAY_TRADING["decision_clock"] == "5m"
    assert DAY_TRADING["specialists"] == ["5m", "15m", "30m", "1h"]
    assert DAY_TRADING["veto_specialist"] == "1h"
    assert (DAY_TRADING["agreement_required"], DAY_TRADING["of"]) == (3, 4)
    assert CONSENSUS_VERSION == "consensus_v1"


@pytest.mark.parametrize("item", MODES, ids=[m["mode"] for m in MODES])
def test_each_mode_votes_on_its_own_specialists_and_vetoes_with_its_slowest(item):
    assert item["veto_specialist"] in item["specialists"]
    assert item["veto_specialist"] == item["specialists"][-1]
    assert item["of"] == len(item["specialists"])
    # A majority, never a unanimity and never a minority.
    assert item["of"] // 2 < item["agreement_required"] <= item["of"]
    assert item["decision_clock"] == item["specialists"][0]
    assert item["horizon_bars"] == 6


def test_mode_lookup_refuses_a_mode_nobody_registered():
    assert mode("SCALPING") is SCALPING
    with pytest.raises(KeyError, match="registered no mode"):
        mode("SWING")


def test_costs_are_unchanged_from_p6():
    from nn.p6_preregistration import COSTS as P6_COSTS

    for key in ("fee_rate", "slippage_rate", "cost_threshold"):
        assert COSTS[key] == P6_COSTS[key]


def test_alignment_is_close_referenced_and_right_sided():
    assert "close" in ALIGNMENT["reference_instant"]
    assert "side='right'" in ALIGNMENT["rule"]
    assert "searchsorted" in ALIGNMENT["rule"]
    assert "maps row t to row t exactly" in ALIGNMENT["own_clock_is_the_identity"]


def test_an_unavailable_specialist_holds_rather_than_voting_partially():
    assert UNAVAILABILITY_RULE["consensus"] == "HOLD"
    assert UNAVAILABILITY_RULE["constituent_replay"].startswith("HOLD")


def test_the_measured_availability_loses_only_block_heads():
    for name in ("scalping", "day_trading"):
        record = MEASURED_AVAILABILITY[name]
        assert record["worst_fold_unavailable"] <= 14
        # The mode's own decision clock is never unavailable: it is the identity.
        own = mode(name.upper())["decision_clock"]
        assert record["unavailable_rows"][own] == 0
        # A slower specialist loses at least as many rows as a faster one.
        losses = [
            record["unavailable_rows"][clock] for clock in mode(name.upper())["specialists"]
        ]
        assert losses == sorted(losses)


def test_the_decision_rule_has_exactly_two_conditions_and_both_are_required():
    assert DECISION_RULE["conjunction"] == "both"
    assert len(DECISION_RULE["conditions"]) == 2
    assert DECISION_RULE["improved_folds_required"] == 3
    assert DECISION_RULE["total_folds"] == 4
    assert DECISION_RULE["per_mode"] is True


def test_the_validity_gate_precedes_every_verdict():
    assert "VALIDITY_GATE" in ALIGNMENT["own_clock_is_the_identity"]
    assert "reproduces the frozen" in VALIDITY_GATE["check"]
    assert VALIDITY_GATE["on_failure"].startswith("P7 is invalid")


def test_the_hash_is_frozen():
    assert preregistration_hash() == FROZEN_HASH
    assert describe()["preregistration_hash"] == FROZEN_HASH


@pytest.mark.parametrize("key", sorted(payload()))
def test_every_payload_key_is_populated(key):
    assert payload()[key] not in (None, "", [], {})


def test_the_battery_covers_every_property_the_document_lists():
    ids = [item["id"] for item in LEAKAGE_BATTERY]
    assert ids == [f"C{n}" for n in range(1, len(ids) + 1)]
    assert len(ids) == 10
    text = DOCUMENT.read_text()
    for item in LEAKAGE_BATTERY:
        assert f"| {item['id']} |" in text


# --------------------------------------------------------------------------- #
# C. document and twin are one design
# --------------------------------------------------------------------------- #


def test_the_document_publishes_the_same_hash():
    text = DOCUMENT.read_text()
    assert FROZEN_HASH in text
    assert f"# {CHECKPOINT} — preregistration" in text


def test_the_document_states_both_consensus_rules_verbatim():
    text = _flat(DOCUMENT.read_text())
    assert "at least 2 of the 3 specialists are actively long" in text
    assert "the 15m specialist is not actively short" in text
    assert "at least 3 of the 4 specialists are actively long" in text
    assert "the 1h specialist is not actively short" in text
    assert "otherwise hold" in text


def test_every_forbidden_item_appears_in_the_document():
    section = _flat(DOCUMENT.read_text().split("## 8. Forbidden")[1].split("## 9.")[0])
    for item in FORBIDDEN_AFTER_RESULTS:
        assert _flat(item) in section, f"the document does not forbid: {item}"


def test_the_document_has_no_unresolved_placeholder():
    assert not re.search(r"\bTODO\b|\bTBD\b|XXX", DOCUMENT.read_text())


# --------------------------------------------------------------------------- #
# D. tripwire
# --------------------------------------------------------------------------- #

#: Flipped in the commit that produces P7 evidence.
P7_EVIDENCE_EXPECTED = True


def test_no_p7_artifact_exists_before_the_evidence_commit():
    if P7_EVIDENCE_EXPECTED:
        pytest.skip("P7 evidence is expected; its own tests check it")
    found = sorted(p.name for p in (REPO / "artifacts" / "benchmark").glob("btc_p7*"))
    assert (
        found == []
    ), f"P7 artifacts exist while the preregistration is the whole of P7: {found}"


def test_the_research_state_reports_p7_as_preregistered_or_answered():
    from nn.research_state import checkpoint_states

    state = checkpoint_states(REPO)["P7"]
    assert state in {"preregistered", "answered"}
    if not P7_EVIDENCE_EXPECTED:
        assert state == "preregistered"
