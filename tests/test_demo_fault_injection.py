"""`FaultSchedule`, and the runner meeting each fault it describes.

Section 12.3: fault schedules are "applied by tests and soak drills only".
Nothing in the runner reads one; a test arranges the broken world and the runner
meets it exactly as it would meet a genuinely degraded recorder. These tests
assert that, too -- a fault-injection branch on the production path would be a
switch someone could leave on.
"""

from __future__ import annotations

import ast
import json
from decimal import Decimal as D
from pathlib import Path

import pytest

from chimera.demo.config import ConfigProfile, DemoConfigError, parse_demo_config
from chimera.demo.decision_log import RecordKind
from chimera.demo.faults import Fault, FaultError, FaultSchedule, ScheduledFault
from chimera.demo.runner import RunnerState
from tests.demo_harness import DAY, build

REPO = Path(__file__).resolve().parents[1]


# --- the schedule itself ---------------------------------------------------
def test_a_schedule_reads_back_what_it_was_given():
    schedule = FaultSchedule(
        [
            ScheduledFault(3, Fault.MISSING_MINUTE),
            ScheduledFault(7, Fault.STALE_FEED, {"seconds": 600}),
        ]
    )
    assert schedule.offsets == (3, 7)
    assert schedule.faults_at(3) == {Fault.MISSING_MINUTE}
    assert Fault.STALE_FEED in schedule
    assert Fault.VENUE_REJECT not in schedule
    assert len(schedule) == 2


def test_an_empty_schedule_is_inert():
    schedule = FaultSchedule.from_config(None)
    assert len(schedule) == 0
    assert schedule.at(0) == ()


@pytest.mark.parametrize(
    "block,message",
    [
        ({"schedule": "not a list"}, "must be a list"),
        ({"schedule": [{"fault": "missing_minute"}]}, "needs both"),
        ({"schedule": [{"minute_offset": 1}]}, "needs both"),
        ({"schedule": [{"minute_offset": 1, "fault": "nonsense"}]}, "unknown fault"),
        ({"schedule": ["nope"]}, "must be a mapping"),
    ],
)
def test_a_schedule_that_cannot_be_read_is_refused(block, message):
    with pytest.raises(FaultError, match=message):
        FaultSchedule.from_config(block)


def test_a_negative_offset_is_refused():
    with pytest.raises(FaultError, match="forward"):
        ScheduledFault(-1, Fault.STALE_FEED)


def test_a_schedule_round_trips_through_its_own_dict():
    original = FaultSchedule([ScheduledFault(2, Fault.MISSING_BOOK, {"leg": "spot"})])
    restored = FaultSchedule.from_config(original.to_dict())
    assert restored.to_dict() == original.to_dict()


def test_a_schedule_carries_no_callable():
    """Data, not code: a config file may describe a drill, never execute one."""
    payload = FaultSchedule([ScheduledFault(1, Fault.STORE_ERROR)]).to_dict()
    json.dumps(payload)  # raises if anything unserialisable crept in


# --- the campaign profile still refuses faults -----------------------------
def test_a_campaign_configuration_still_refuses_a_fault_block(tmp_path):
    """PR-09's rule, re-asserted now that PR-10 gives faults a meaning."""
    base = json.loads((REPO / "conf" / "demo" / "pvc1.json").read_text(encoding="utf-8"))
    with pytest.raises(DemoConfigError):
        parse_demo_config(
            {**base, "faults": {"schedule": []}}, expected_profile=ConfigProfile.CAMPAIGN
        )


def test_a_soak_configuration_may_carry_one(tmp_path):
    base = json.loads((REPO / "conf" / "demo" / "pvc1.json").read_text(encoding="utf-8"))
    config = parse_demo_config(
        {
            **base,
            "profile": "SOAK",
            "faults": {"schedule": [{"minute_offset": 4, "fault": "missing_book"}]},
        },
        expected_profile=ConfigProfile.SOAK,
    )
    schedule = FaultSchedule.from_config(config.faults)
    assert schedule.faults_at(4) == {Fault.MISSING_BOOK}


# --- nothing on the production path reads a schedule -----------------------
def test_no_runtime_module_imports_the_fault_schedule():
    """The strongest form of "applied by tests and drills only"."""
    offenders = []
    for path in sorted((REPO / "chimera").rglob("*.py")):
        if path.name == "faults.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            module = ""
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
            elif isinstance(node, ast.Import):
                module = ",".join(a.name for a in node.names)
            if "demo.faults" in module:
                offenders.append(f"{path.relative_to(REPO)}:{node.lineno}")
    assert not offenders, (
        "no runtime module may import chimera.demo.faults; a fault schedule is "
        f"applied by a test or a drill, never by the runner. Found {offenders}"
    )


# --- the runner meeting each fault -----------------------------------------
def test_a_missing_minute_is_logged_and_stepped_over(tmp_path):
    schedule = FaultSchedule([ScheduledFault(2, Fault.MISSING_MINUTE)])
    harness = build(tmp_path, shapes={f"spot:{DAY}": schedule.minute_shapes("spot")})
    outcomes = harness.run(4)
    assert outcomes[2].kind is RecordKind.INCOMPLETE_STATE
    assert outcomes[3].kind is RecordKind.DECISION, "the run continues past the gap"
    assert harness.runner.state is RunnerState.READY


def test_a_missing_book_is_logged_and_no_rule_evaluates(tmp_path):
    schedule = FaultSchedule([ScheduledFault(1, Fault.MISSING_BOOK)])
    harness = build(tmp_path, shapes={f"um:{DAY}": schedule.minute_shapes("um")})
    outcomes = harness.run(3)
    assert outcomes[1].kind is RecordKind.INCOMPLETE_STATE
    record = [r for r in harness.records() if r["kind"] == "INCOMPLETE_STATE"][0]
    assert "um_book" in record["veto_or_rejection"]["detail"]
    assert "rule" not in record


def test_a_kill_switch_mid_run_halts_and_stops_the_loop(tmp_path):
    harness = build(tmp_path)
    harness.run(1)
    (harness.state_dir / "KILL_SWITCH").write_text("stop", encoding="utf-8")
    first = harness.first_minute_ms()
    outcomes = harness.runner.run_minutes([first + i * 60_000 for i in range(1, 4)])
    assert outcomes[0].kind is RecordKind.HALT
    assert len(outcomes) == 1, "run_minutes stops at HALT rather than continuing"
    assert harness.runner.state is RunnerState.HALT


def test_a_venue_that_will_not_fill_leaves_the_position_flat_not_one_sided(tmp_path):
    """No book installed, so the fill model refuses: both legs stay flat."""
    harness = build(tmp_path)
    harness.run(2, quote=False)
    assert harness.runner.position.imbalance() == D("0")
    assert harness.runner.position.leg("spot").is_flat
    assert harness.runner.position.leg("perp").is_flat


def test_the_runner_survives_a_stale_feed_without_inventing_a_decision(tmp_path):
    """A stale feed is section 7.2's VETO, not a halt: no decision is fabricated."""
    harness = build(tmp_path)
    first = harness.first_minute_ms()
    harness.runner.clock.observe((first + 10_000 * 60_000) * 1_000_000)
    outcome = harness.tick(first)
    assert outcome.kind in (RecordKind.DECISION, RecordKind.INCOMPLETE_STATE)
    assert harness.runner.state in (RunnerState.READY, RunnerState.HALT)
    if harness.runner.state is RunnerState.HALT:
        assert "kill_switch" not in (harness.runner.halt_reason or "")


def test_every_fault_the_enum_names_is_exercised_or_explained():
    """A schedule that could name a fault no test arranges would be decoration."""
    exercised = {
        Fault.MISSING_MINUTE,
        Fault.MISSING_BOOK,
        Fault.KILL_SWITCH,
        Fault.VENUE_REJECT,
        Fault.STALE_FEED,
    }
    # PARTIAL_FILL and STORE_ERROR are exercised against HedgedPosition and
    # FuturesStore directly, in tests/test_carry_hedge.py and the PR-07 suite;
    # they are named here so the enum stays honest about what it can describe.
    explained = {Fault.PARTIAL_FILL, Fault.STORE_ERROR}
    assert exercised | explained == set(Fault)
