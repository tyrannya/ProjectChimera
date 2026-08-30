"""The futures dry-run protocol, and the parts of it that must not drift.

Two failures these tests exist to catch, both of which would otherwise be silent:

*the protocol and its evidence coming apart.* A committed report embeds the
protocol hash it ran under. If someone weakens an invariant and re-runs, the hash
moves and the old report no longer verifies — which is the point. If someone
weakens an invariant and does *not* re-run, the committed report stops matching
the build, which :func:`tools.futures_dry_run.verify` reports. Either way the
disagreement surfaces; these tests are what make sure it surfaces in CI.

*the documented hash going stale.* ``docs/futures_dry_run_validation.md`` prints
the hash at the top. A doc that names a hash nobody checks is a doc that will
eventually name the wrong one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import futures_dry_run as dry_run

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = ROOT / "artifacts" / "futures_dry_run_v1"
PROTOCOL_DOC = ROOT / "docs" / "futures_dry_run_validation.md"


def test_the_protocol_hash_is_stable_under_reserialisation():
    """A hash that changed between two calls could not pin anything."""
    assert dry_run.protocol_hash() == dry_run.protocol_hash()
    assert dry_run.protocol_hash().startswith("sha256:")


def test_the_documented_protocol_hash_matches_the_code():
    """The doc names the hash. A stale one is worse than none: it reads as checked."""
    text = PROTOCOL_DOC.read_text()
    assert dry_run.protocol_hash() in text, (
        "docs/futures_dry_run_validation.md does not name the current protocol hash "
        f"{dry_run.protocol_hash()}. Regenerate the header when the protocol changes."
    )


def test_every_declared_invariant_has_a_claim_and_a_scenario():
    ids = [i["id"] for i in dry_run.PROTOCOL["invariants"]]
    assert len(ids) == len(set(ids)), "two invariants share an id"
    scenarios = set(dry_run.PROTOCOL["scenarios"])
    for invariant in dry_run.PROTOCOL["invariants"]:
        assert invariant["claim"].strip(), f"{invariant['id']} has no claim"
        assert invariant["scenario"] in scenarios, (
            f"{invariant['id']} names scenario {invariant['scenario']!r}, which is not "
            f"declared: {sorted(scenarios)}"
        )


def test_acceptance_has_no_threshold_on_any_measured_quantity():
    """There must be nothing to tune. A scored metric is a metric someone optimises."""
    acceptance = dry_run.PROTOCOL["acceptance"]
    assert acceptance["descriptive_metrics_are_not_criteria"] is True
    blob = json.dumps(acceptance)
    for word in ("at least", "above", "below", "threshold", ">=", "<="):
        assert word not in blob, (
            f"the acceptance rule mentions {word!r}. Acceptance is every invariant "
            "holding; a numeric bar would be a thing to tune towards."
        )


def test_the_replay_window_cannot_reach_p4_hold_or_styx():
    """P4-HOLD was retired unread and is not spent on an engineering test."""
    start, end = dry_run.PROTOCOL["source"]["rows"]
    hold_start, hold_end = dry_run.PROTOCOL["source"]["forbidden_rows"]["p4_hold"]
    assert (
        end <= hold_start
    ), f"the replay window [{start}, {end}) reaches P4-HOLD [{hold_start}, {hold_end})"
    window = dry_run._load_replay(ROOT)
    assert len(window) == end - start
    assert str(window["date"].iloc[-1]) < "2025-08-27 23:00:00+00:00"


def test_a_window_that_reached_p4_hold_is_refused(monkeypatch):
    """The rule is code, not care: moving the window into P4-HOLD must raise."""
    source = dict(dry_run.PROTOCOL["source"])
    source["rows"] = [40981, 46000]
    monkeypatch.setitem(dry_run.PROTOCOL, "source", source)
    with pytest.raises(dry_run.ProtocolViolation, match="P4-HOLD"):
        dry_run._load_replay(ROOT)


def test_the_committed_report_verifies_against_this_build():
    problems = dry_run.verify(EVIDENCE)
    assert problems == [], "\n".join(problems)


def test_the_committed_report_passed_every_declared_invariant():
    report = json.loads((EVIDENCE / dry_run.REPORT_NAME).read_text())
    assert report["outcome"] == "PASS"
    declared = {i["id"] for i in dry_run.PROTOCOL["invariants"]}
    observed = {r["id"] for r in report["invariants"]}
    assert declared <= observed, f"never observed: {sorted(declared - observed)}"
    assert all(r["held"] for r in report["invariants"])


def test_a_report_from_another_protocol_is_rejected(tmp_path):
    """The mechanism that makes 'frozen before evaluation' a fact rather than a claim."""
    report = json.loads((EVIDENCE / dry_run.REPORT_NAME).read_text())
    report["protocol_hash"] = "sha256:" + "0" * 64
    (tmp_path / dry_run.REPORT_NAME).write_text(json.dumps(report))
    problems = dry_run.verify(tmp_path)
    assert any("protocol" in p for p in problems)


def test_a_report_with_a_failed_invariant_is_rejected(tmp_path):
    report = json.loads((EVIDENCE / dry_run.REPORT_NAME).read_text())
    report["invariants"][0]["held"] = False
    (tmp_path / dry_run.REPORT_NAME).write_text(json.dumps(report))
    problems = dry_run.verify(tmp_path)
    assert any("did not hold" in p for p in problems)


def test_a_report_missing_an_invariant_is_rejected(tmp_path):
    """A protocol that gains an invariant must invalidate the reports predating it."""
    report = json.loads((EVIDENCE / dry_run.REPORT_NAME).read_text())
    dropped = report["invariants"].pop(0)["id"]
    (tmp_path / dry_run.REPORT_NAME).write_text(json.dumps(report))
    problems = dry_run.verify(tmp_path)
    assert any(dropped in p and "never observed" in p for p in problems)


def test_the_status_page_leads_with_what_the_evidence_is_not():
    """A reader who stops after the first screen must not think this is alpha evidence."""
    status = (EVIDENCE / dry_run.STATUS_NAME).read_text()
    assert status.startswith("# OPERATIONAL"), (
        "the first line is the artifact index's status word, and "
        "tests/test_reporting_integrity.py asserts it matches the row there"
    )
    head = status[:1200]
    assert "not** evidence of" in head or "not evidence of" in head
    assert "trading alpha" in head
    assert "operational" in head.lower()


def test_the_status_page_marks_every_metric_as_descriptive():
    status = (EVIDENCE / dry_run.STATUS_NAME).read_text()
    assert "not acceptance criteria" in status
    assert "may be optimised against" in status


def test_the_protocol_states_what_it_does_not_cover():
    """Sustained wall-clock paper trading is a later requirement, not a silent gap."""
    covered = " ".join(dry_run.PROTOCOL["not_covered"]).lower()
    assert "real-time paper" in covered
    assert "wall-clock" in covered
    assert "real-time paper" in PROTOCOL_DOC.read_text().lower()


def test_the_protocol_refuses_to_be_read_as_research_evidence():
    assert dry_run.PROTOCOL["evidence_class"] == "operational"
    not_evidence = " ".join(dry_run.PROTOCOL["not_evidence_of"]).lower()
    for claim in ("trading alpha", "exchange execution quality", "p5"):
        assert claim in not_evidence


def test_running_the_protocol_twice_produces_the_same_report(tmp_path):
    """Determinism is what makes this a check rather than a sample.

    Slow but load-bearing: a replay whose numbers moved between two runs could not
    be used to detect that a change had altered execution behaviour.
    """
    first = dry_run.run(ROOT, tmp_path / "a")
    second = dry_run.run(ROOT, tmp_path / "b")
    assert first == second
    assert first["outcome"] == "PASS"
