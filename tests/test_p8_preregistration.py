"""P8's design, and the one fact about P8 that is currently true: it is not opened.

Most of this file is the usual preregistration coherence check. The tests that
matter are the ones asserting **P8 has not been opened**: no router module, no P8
artifact, no P8 number anywhere, and a research state that says `unrun`,
`preregistered` or `withdrawn` and never `answered`. It is `withdrawn` since
2026-09-03: the precondition cannot be met, so the checkpoint was closed without
ever being opened, which is a non-result and not an answer.

A preregistration is a commitment, and the cheapest way for one to rot is for
somebody to quietly satisfy it. These tests are what makes "NOT OPENED" a
checkable claim rather than a sentence.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from nn.p8_preregistration import (
    CHECKPOINT,
    CONTRACT,
    CURRENT_RESULT_STATE,
    DECISION_RULE,
    FORBIDDEN_AFTER_RESULTS,
    FORBIDDEN_INPUTS,
    OUTCOME,
    PERMITTED_INPUTS,
    RESULT_STATES,
    SAFETY_PROHIBITIONS,
    TURNOVER_GUARD,
    describe,
    payload,
    preregistration_hash,
)

REPO = Path(__file__).resolve().parents[1]
DOCUMENT = REPO / "docs" / "p8_preregistration.md"

FROZEN_HASH = "sha256:abbb76258980d557eb601855ea24834295ca54f74b037f3ef4926233faaa01dd"


def _flat(text: str) -> str:
    stripped = text.replace("`", "").replace("*", "").replace(">", " ")
    return re.sub(r"\s+", " ", stripped).lower()


# --------------------------------------------------------------------------- #
# A. P8 is not opened
# --------------------------------------------------------------------------- #


def test_the_outcome_is_not_opened():
    assert OUTCOME == "NOT OPENED"
    assert CURRENT_RESULT_STATE == "P8 AUTO MODE ROUTER: NOT OPENED"
    assert CURRENT_RESULT_STATE in RESULT_STATES
    assert len(RESULT_STATES) == 4


def test_no_router_has_been_implemented():
    """A design, not a component. Nothing imports or provides one."""
    assert not (REPO / "nn" / "p8.py").exists()
    assert not (REPO / "nn" / "p8_decision.py").exists()
    assert not (REPO / "chimera" / "router.py").exists()
    assert not (REPO / "chimera" / "auto_router.py").exists()


def test_no_p8_artifact_exists():
    found = sorted(p.name for p in (REPO / "artifacts").rglob("*p8*"))
    assert found == [], f"P8 artifacts exist while P8 is not opened: {found}"


def test_the_research_state_never_calls_p8_answered():
    """P8 produced nothing, so no state implying a result may be reachable.

    `withdrawn` joined the vocabulary on 2026-09-03, when P8 was withdrawn as
    moot: its precondition needs two eligible modes, and producing a second one
    would mean refitting clocks P6 and P6-EXT screened out, which section 11
    forbids. That is still not an answer — it is a checkpoint closed without
    ever being opened — so the assertion this test was written for is unchanged.
    """
    from nn.research_state import ANSWERED, WITHDRAWN, checkpoint_states

    state = checkpoint_states(REPO)["P8"]
    assert state != ANSWERED
    assert state in {"unrun", "preregistered", WITHDRAWN}
    # And the fact S0 recorded, so that quietly un-withdrawing it fails here.
    assert state == WITHDRAWN


def test_there_is_no_auto_mode_to_route_into():
    from chimera.modes import TradingMode

    assert "AUTO" not in {mode.value for mode in TradingMode}


def test_the_precondition_is_recorded_and_currently_unmet():
    """P8 opens when two modes are eligible. None is."""
    import json

    from chimera.modes import TradingMode, evaluate_eligibility
    from chimera.modes import SpecialistStatus

    status = {}
    for directory in ("btc_p6_decision", "btc_p6ext_decision"):
        decision = json.loads(
            (REPO / "artifacts" / "benchmark" / directory / "decision.json").read_text()
        )
        for row in decision["clocks"]:
            status[row["clock"]] = SpecialistStatus(row["clock"], True, bool(row["viable"]))
    eligible = [
        mode
        for mode, row in evaluate_eligibility(status).items()
        if row.eligible and mode is not TradingMode.FLAT
    ]
    assert eligible == [], "P8's precondition is met; it may be opened deliberately"
    assert "at least two modes are eligible" in _flat(payload()["not_opened_because"])


# --------------------------------------------------------------------------- #
# B. the design
# --------------------------------------------------------------------------- #


def test_the_router_may_never_select_on_realised_performance():
    flat = " | ".join(FORBIDDEN_INPUTS).lower()
    for phrase in ("pnl", "outer-fold performance", "backtest rank", "post-hoc best"):
        assert phrase in flat
    # And none of the permitted inputs is a realised outcome.
    permitted = " | ".join(PERMITTED_INPUTS).lower()
    for phrase in ("pnl", "backtest", "realised return", "sharpe"):
        assert phrase not in permitted


def test_the_contract_names_every_field_the_router_must_emit():
    assert CONTRACT == (
        "mode",
        "mode_confidence",
        "eligible_modes",
        "reason_code",
        "expected_cost",
        "expected_edge",
        "consensus_state",
    )


def test_the_decision_rule_has_three_conditions_including_the_turnover_guard():
    assert DECISION_RULE["conjunction"] == "all three"
    assert len(DECISION_RULE["conditions"]) == 3
    assert DECISION_RULE["improved_folds_required"] == 3
    assert DECISION_RULE["total_folds"] == 4
    assert "turnover guard" in DECISION_RULE["conditions"][2]


def test_the_turnover_guard_is_a_frozen_number():
    assert TURNOVER_GUARD["multiple"] == 1.25
    assert "may not be moved afterwards" in TURNOVER_GUARD["frozen"]
    assert "1.25x" in TURNOVER_GUARD["rule"]


def test_the_safety_prohibitions_cover_the_ways_a_router_could_widen_risk():
    flat = " | ".join(SAFETY_PROHIBITIONS).lower()
    for phrase in (
        "leverage",
        "instrument family",
        "margin borrowing",
        "exchange",
        "different coin",
        "contradictory modes",
        "emergency flatten",
        "reconciliation dispute",
        "aegis veto",
    ):
        assert phrase in flat


def test_flat_is_declared_a_success_not_a_failure():
    assert "first-class successful outcome" in payload()["flat_is_success"]


def test_the_architecture_puts_aegis_after_the_router():
    architecture = payload()["architecture"]
    assert architecture.index("AUTO Mode Router") < architecture.index("Aegis")
    assert architecture.index("Aegis") < architecture.index("Hermes")


def test_the_hash_is_frozen_and_the_document_publishes_it():
    assert preregistration_hash() == FROZEN_HASH
    assert describe()["preregistration_hash"] == FROZEN_HASH
    assert FROZEN_HASH in DOCUMENT.read_text()


@pytest.mark.parametrize("key", sorted(payload()))
def test_every_payload_key_is_populated(key):
    assert payload()[key] not in (None, "", [], {})


def test_the_document_says_not_opened_where_a_reader_will_see_it():
    text = _flat(DOCUMENT.read_text())
    assert f"# {CHECKPOINT.lower()} — preregistration" in text
    assert "p8 is not opened" in text
    assert "the current state is not opened" in text
    assert "no p8 fit has been made" in text


def test_every_forbidden_item_appears_in_the_document():
    section = _flat(DOCUMENT.read_text().split("## 11. Forbidden")[1])
    for item in FORBIDDEN_AFTER_RESULTS:
        assert _flat(item) in section, f"the document does not forbid: {item}"
