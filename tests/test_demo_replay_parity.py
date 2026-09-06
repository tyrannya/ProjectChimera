"""Section 10's parity, on the synthetic fixture with faults.

Everything here is invented by `chimera.demo.fixtures`. Parity is a statement
about REPRODUCIBILITY, not about profit: it says that a replay of the recorded
minutes decides what the live run decided, which is what makes a decision log
evidence rather than an anecdote.
"""

from __future__ import annotations


import pytest

from chimera.demo.faults import Fault, FaultSchedule, ScheduledFault
from tests.demo_harness import DAY, NEXT_DAY, build
from tools.replay_parity import (
    ENVIRONMENT_PARITY,
    MUST_MATCH,
    OPERATIONAL_KINDS,
    ParityReport,
    compare_logs,
    copy_range,
    read_log,
)

MINUTES = 120


def run_campaign(where, *, shapes=None, minutes: int = MINUTES):
    """One campaign over synthetic days, returning its harness and records."""
    harness = build(where, shapes=shapes)
    first = harness.first_minute_ms()
    for index in range(minutes):
        harness.tick(first + index * 60_000)
    harness.runner.shutdown("done")
    return harness


@pytest.fixture
def faulted_shapes():
    schedule = FaultSchedule(
        [
            ScheduledFault(11, Fault.MISSING_MINUTE),
            ScheduledFault(47, Fault.MISSING_BOOK),
            ScheduledFault(88, Fault.MISSING_MINUTE),
        ]
    )
    return {f"spot:{DAY}": schedule.minute_shapes("spot")}


# ---------------------------------------------------------------------------
# the acceptance criterion
# ---------------------------------------------------------------------------
def test_a_replay_of_the_synthetic_fixture_with_faults_is_in_parity(tmp_path, faulted_shapes):
    """§14 row 11's acceptance: "parity on the synthetic fixture with faults"."""
    live = run_campaign(tmp_path / "live", shapes=faulted_shapes)
    replay = run_campaign(tmp_path / "replay", shapes=faulted_shapes)

    report = compare_logs(live.records(), replay.records())

    assert report.ok, (
        report.divergences[0].render() if report.divergences else report.to_dict()
    )
    assert report.compared >= MINUTES - 5, "the comparison must not be vacuous"
    assert report.label == "PARITY"
    assert not report.live_only and not report.replay_only


def test_the_faults_were_actually_met(tmp_path, faulted_shapes):
    """Otherwise the parity above would be over a run that met no fault at all."""
    live = run_campaign(tmp_path / "live", shapes=faulted_shapes)
    kinds = [r["kind"] for r in live.records()]
    assert kinds.count("INCOMPLETE_STATE") == 3


def test_parity_holds_without_faults_too(tmp_path):
    live = run_campaign(tmp_path / "a")
    replay = run_campaign(tmp_path / "b")
    assert compare_logs(live.records(), replay.records()).ok


# ---------------------------------------------------------------------------
# the comparison itself, two-sided
# ---------------------------------------------------------------------------
def test_a_changed_must_match_field_is_caught(tmp_path, faulted_shapes):
    """The negative control. A comparison that cannot fail proves nothing."""
    live = run_campaign(tmp_path / "live", shapes=faulted_shapes)
    records = live.records()
    tampered = [dict(r) for r in records]
    decision = next(r for r in tampered if r["kind"] == "DECISION")
    decision["signal"] = {**decision["signal"], "target_qty": "999.999"}

    report = compare_logs(records, tampered)
    assert not report.ok
    assert report.divergences[0].field_name == "signal"
    assert "999.999" in report.divergences[0].render()


#: Section 10's must-match list, written out INDEPENDENTLY of the tool's own
#: constant. Parametrising over `MUST_MATCH` made this vacuous in exactly the
#: direction that matters: deleting a field from the tool's list deleted its
#: test case too, so the suite stayed green while the comparison silently
#: stopped checking it. A mutation witness found that; this literal is the fix.
SECTION_10_MUST_MATCH: tuple[str, ...] = (
    "kind",
    "minute",
    "seq",
    "runner_now_ns",
    "inputs",
    "rule",
    "signal",
    "requested_action",
    "risk",
    "execution",
    "position_after",
    "ledger_effect",
    "veto_or_rejection",
)


def test_the_tools_must_match_list_is_exactly_section_10s():
    """The list itself, pinned. Removing a field from the tool now fails here."""
    assert tuple(sorted(MUST_MATCH)) == tuple(sorted(SECTION_10_MUST_MATCH))


def test_the_tools_operational_kinds_are_exactly_section_10s():
    """The set itself, pinned, for the same reason."""
    assert OPERATIONAL_KINDS == frozenset(
        {"STARTUP", "SHUTDOWN", "RECOVERY", "HALT", "RESUME"}
    )


@pytest.mark.parametrize(
    "field_name", [f for f in SECTION_10_MUST_MATCH if f not in ("kind", "minute")]
)
def test_every_must_match_field_is_actually_compared(tmp_path, field_name):
    """Each field in section 10's list, perturbed one at a time.

    A must-match list that named a field the comparison never read would be a
    promise the tool does not keep, and nothing else would notice.
    """
    live = run_campaign(tmp_path / "live", minutes=6)
    records = live.records()
    tampered = [dict(r) for r in records]
    target = next(r for r in tampered if r["kind"] == "DECISION")
    target[field_name] = "PERTURBED"

    report = compare_logs(records, tampered)
    assert not report.ok, f"{field_name} is in MUST_MATCH but is not compared"
    assert any(d.field_name == field_name for d in report.divergences)


def test_a_decision_dropped_from_the_replay_is_still_a_failure(tmp_path):
    """The negative control for the operational-kinds rule: only those are forgiven."""
    live = run_campaign(tmp_path / "live", minutes=8)
    records = live.records()
    fewer = [r for r in records if r["kind"] != "DECISION"] + [
        r for r in records if r["kind"] == "DECISION"
    ][1:]

    assert not compare_logs(records, fewer).ok, "a missing DECISION is a parity failure"


def test_a_missing_record_is_reported_rather_than_ignored(tmp_path):
    live = run_campaign(tmp_path / "live", minutes=8)
    records = live.records()
    shortened = [r for r in records if r["kind"] != "DECISION"] + [
        r for r in records if r["kind"] == "DECISION"
    ][:-1]

    report = compare_logs(records, shortened)
    assert not report.ok
    assert report.live_only, "a decision the replay never made must be named"


def test_an_extra_record_is_reported_rather_than_ignored(tmp_path):
    live = run_campaign(tmp_path / "live", minutes=8)
    records = live.records()
    extra = [dict(r) for r in records]
    ghost = dict(next(r for r in extra if r["kind"] == "DECISION"))
    ghost["minute"] = "2099-01-01T00:00:00+00:00"
    extra.append(ghost)

    report = compare_logs(records, extra)
    assert not report.ok
    assert report.replay_only


# ---------------------------------------------------------------------------
# what section 10 says may differ
# ---------------------------------------------------------------------------
def test_operational_kinds_are_aligned_rather_than_compared(tmp_path):
    """ "The replay may have fewer restarts."" """
    live = run_campaign(tmp_path / "live", minutes=8)
    records = live.records()
    # The kind is NAMED, not taken from the constant under test: building the
    # fixture from `OPERATIONAL_KINDS` made this vacuous, because emptying the
    # set left nothing dropped and the comparison trivially passed.
    without_startup = [r for r in records if r["kind"] != "STARTUP"]
    assert len(without_startup) < len(records), "the fixture must actually drop one"

    report = compare_logs(records, without_startup)
    assert report.ok, "dropping a STARTUP record must not be a parity failure"
    assert report.compared > 0


def test_a_different_environment_is_labelled_rather_than_passed(tmp_path):
    """Section 10: reported separately, never quietly counted as a pass."""
    live = run_campaign(tmp_path / "live", minutes=6)
    records = live.records()
    other = [dict(r) for r in records]
    for record in other:
        record["software"] = {**record["software"], "python": "9.9.9"}

    report = compare_logs(records, other)
    assert report.label == ENVIRONMENT_PARITY
    assert "9.9.9" in report.environment_note
    assert report.ok, "the environment label is not itself a divergence"


def test_the_same_environment_is_not_labelled(tmp_path):
    """The negative control for the label."""
    live = run_campaign(tmp_path / "live", minutes=6)
    report = compare_logs(live.records(), [dict(r) for r in live.records()])
    assert report.label == "PARITY"
    assert report.environment_note == ""


def test_runner_now_ns_is_compared_with_tolerance_zero(tmp_path):
    """Section 10's wording, taken literally.

    The field sits in the plan's "may differ" column, but the text says it "is
    compared with tolerance zero because it is derived from recorded receipt
    stamps, not the wall clock". Tolerance zero is exact comparison, and
    treating it as free would drop the one field that proves the clock is not
    the wall clock.
    """
    assert "runner_now_ns" in MUST_MATCH
    live = run_campaign(tmp_path / "live", minutes=6)
    records = live.records()
    shifted = [dict(r) for r in records]
    decision = next(r for r in shifted if r["kind"] == "DECISION")
    decision["runner_now_ns"] = decision["runner_now_ns"] + 1

    report = compare_logs(records, shifted)
    assert not report.ok
    assert any(d.field_name == "runner_now_ns" for d in report.divergences)


# ---------------------------------------------------------------------------
# the scratch copy
# ---------------------------------------------------------------------------
def test_the_scratch_copy_carries_the_days_asked_for_and_no_others(tmp_path):
    """Two days written, ONE asked for.

    An earlier version globbed for a day that never existed upstream, so a copy
    that took everything still passed it. The unasked-for day has to be present
    in the source for its absence in the scratch copy to mean anything.
    """
    harness = build(tmp_path / "live", days=(DAY, NEXT_DAY))
    source = harness.root / "normalized" / "um" / "1m"
    assert (source / f"{NEXT_DAY}.parquet").is_file(), "the fixture must write both days"

    scratch = tmp_path / "scratch"
    copy_range(harness.root, scratch, [DAY])

    copied = scratch / "normalized" / "um" / "1m"
    assert (copied / f"{DAY}.parquet").is_file()
    assert (scratch / "normalized" / "spot" / "1m" / f"{DAY}.parquet").is_file()
    assert (scratch / "funding" / "um" / "settlements.ndjson").is_file()
    assert not (copied / f"{NEXT_DAY}.parquet").exists(), (
        "a day the caller did not ask for must not be copied: the range is what "
        "a parity run is about"
    )


def test_the_scratch_copy_leaves_the_recorders_tree_untouched(tmp_path):
    """A parity run may never write near the recorder's own files."""
    harness = build(tmp_path / "live", days=(DAY,))
    before = {
        p.relative_to(harness.root).as_posix(): p.stat().st_mtime_ns
        for p in harness.root.rglob("*")
        if p.is_file()
    }
    copy_range(harness.root, tmp_path / "scratch", [DAY])
    after = {
        p.relative_to(harness.root).as_posix(): p.stat().st_mtime_ns
        for p in harness.root.rglob("*")
        if p.is_file()
    }
    assert before == after


def test_read_log_filters_by_day(tmp_path):
    live = run_campaign(tmp_path / "live", minutes=4)
    directory = live.state_dir / "decision_log"
    assert read_log(directory, [DAY])
    assert read_log(directory, ["2099-01-01"]) == []


def test_an_empty_report_is_not_a_pass():
    """A parity run over nothing must not look like a parity run that passed."""
    report = ParityReport()
    assert report.ok, "an empty comparison is vacuously ok at the type level"
    assert report.compared == 0, "which is why the CLI refuses when nothing was read"
