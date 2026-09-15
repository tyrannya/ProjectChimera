"""R1-c (AEG-4): does the persisted risk state continue the log beside it?

`RiskEngine` already fails closed on a `risk.json` it cannot READ. What it had no
opinion about at all was a `risk.json` that is simply not there: absence loaded as
`RiskStateLoad.MISSING`, which `seed_or_reconcile_equity` correctly reads as a
genuine first start -- so on a campaign that had already halted, decided and
persisted, deleting one file cleared the halt, reset the peak equity, the day, the
cooldown, the order window, the funding streak and every open reconciliation
dispute, and left nothing anywhere saying so. That is the audit's AEG-4, and
`test_the_defect_deleting_risk_json_used_to_clear_a_halt_silently` reproduces it
here against the same files R1-c refuses.

Every test is two-sided. A refusal is paired with the healthy campaign that must
NOT be refused, because "it halts" is satisfied just as well by a build that halts
on everything; and the direction of a disagreement is paired both ways, because a
guard that fails closed on the ordinary crash window would break the runner's
recovery rather than tighten it.

**Fixtures come from production writers.** The campaigns are driven through
`DemoRunner.tick`, `_halt`, `flatten`, `resume` and `shutdown`, and the states are
whatever those leave on disk. Two tests deliberately damage a file -- one replaces
`risk.json` with a directory and one restores an EARLIER copy of it -- and each
says so where it does it: neither state is reachable by writing a file, which is
the whole point of refusing them.

Nothing here is scientific evidence. The prices come from `chimera.demo.fixtures`
and every quantity below is the test author's own arithmetic.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

import chimera.demo.risk_continuity as risk_continuity
from chimera.demo.decision_log import RecordKind
from chimera.demo.risk_continuity import (
    RISK_CONTINUITY_PREFIX,
    ContinuityOutcome,
    RiskContinuity,
    continuity_already_recorded,
    evaluate_risk_continuity,
    is_risk_continuity_halt,
    risk_state_hash,
)
from chimera.demo.risk_wiring import EQUITY_DISPUTE_PREFIX
from chimera.demo.runner import RecoveryCause, RunnerState, _risk_hash
from chimera.risk import RiskEngine, RiskLimits, RiskState, RiskStateLoad
from tests.demo_harness import CAPITAL, DAY, NEXT_DAY, build

#: R1-b's day-boundary anchor, reused for the same reason it was chosen there:
#: the synthetic day is 1440 minutes and deciding from here puts UTC midnight
#: inside a short run. `update_equity` moves `day_start_equity` when the date
#: rolls, and that field IS inside `risk.state_hash` (`day` itself is not), so
#: crossing the boundary is the cheapest production-driven way to make the
#: persisted risk state genuinely different from the one it held ten minutes ago.
BOUNDARY_START = 1435
MINUTES_BEFORE_THE_BOUNDARY = 4
MINUTES_AFTER_THE_BOUNDARY = 6


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def records(state_dir: Path) -> list[dict[str, Any]]:
    """Every committed record of the log, in chain order."""
    out: list[dict[str, Any]] = []
    for path in sorted((Path(state_dir) / "decision_log").glob("*.ndjson")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
    return out


def continuity_records(state_dir: Path) -> list[dict[str, Any]]:
    """The RECOVERY records R1-c wrote, and only those."""
    return [
        record
        for record in records(state_dir)
        if record.get("kind") == RecordKind.RECOVERY.value
        and (record.get("recovery") or {}).get("cause")
        == RecoveryCause.RISK_STATE_DISCONTINUITY.value
    ]


def newest_hash_witness(state_dir: Path) -> tuple[int, str]:
    """The newest ``(seq, risk.state_hash)`` the log carries."""
    carried = [
        (int(record["seq"]), str((record.get("risk") or {}).get("state_hash", "")))
        for record in records(state_dir)
        if (record.get("risk") or {}).get("state_hash")
    ]
    assert carried, "the fixture wrote no record carrying a risk.state_hash"
    return carried[-1]


def loaded_state(state_dir: Path) -> tuple[RiskStateLoad, RiskState]:
    """Read ``risk.json`` exactly as the runner's read-only inspection does."""
    engine = RiskEngine(RiskLimits(), state_path=Path(state_dir) / "risk.json")
    return engine.load_outcome, engine.state


def verdict_for(state_dir: Path) -> RiskContinuity:
    """The verdict a fresh process would take on what is on disk right now."""
    load, state = loaded_state(state_dir)
    return evaluate_risk_continuity(state_dir, load=load, state=state)


def campaign(where: Path, *, days: tuple[str, ...] = (DAY,), minutes: int = 6):
    """A campaign that really ran, through the production tick loop."""
    harness = build(where, days=days)
    harness.run(minutes)
    assert harness.runner.state is not RunnerState.HALT, harness.runner.halt_reason
    return harness


def across_the_boundary(where: Path):
    """A campaign whose risk state moved, and a copy of it from before it did.

    Returns ``(harness, earlier_copy)``. The copy is taken while the campaign is
    running, from the file the campaign itself wrote, and is restored later by a
    test that needs a `risk.json` the log has already moved past. Copying is the
    one thing here that no production path does; every byte in it was written by
    one.
    """
    harness = build(where, days=(DAY, NEXT_DAY))
    first = harness.first_minute_ms() + BOUNDARY_START * 60_000
    harness.run(MINUTES_BEFORE_THE_BOUNDARY, start=first)

    earlier = where / "risk.json.before-the-boundary"
    shutil.copyfile(harness.state_dir / "risk.json", earlier)

    harness.run(
        MINUTES_AFTER_THE_BOUNDARY,
        start=first + MINUTES_BEFORE_THE_BOUNDARY * 60_000,
    )
    harness.runner.shutdown("clean stop after the boundary")
    return harness, earlier


# ---------------------------------------------------------------------------
# 0. the identity the whole comparison rests on
# ---------------------------------------------------------------------------
def test_the_hash_in_the_log_and_the_hash_of_the_file_are_one_function(tmp_path):
    """`risk.state_hash` and R1-c's reading of `risk.json` must not be two things.

    The comparison is worthless if the log records one function of the state and
    the continuity check computes another: they would drift on the first field
    added to `RiskState`, and the drift would show up as a campaign that cannot
    restart rather than as a failure here.
    """
    harness = campaign(tmp_path)
    harness.runner.shutdown("clean stop")

    load, state = loaded_state(harness.state_dir)
    assert load is RiskStateLoad.LOADED
    assert risk_state_hash(state) == _risk_hash(harness.runner.risk)
    assert risk_state_hash(state) == newest_hash_witness(harness.state_dir)[1]


# ---------------------------------------------------------------------------
# 1. a genuine first start is not a dispute
# ---------------------------------------------------------------------------
def test_a_pristine_first_start_is_not_a_continuity_dispute(tmp_path):
    """Requirement A. No log, no `risk.json`: there is nothing to continue.

    The direction matters more than the assertion. A continuity check that
    refused this would refuse every deployment there has ever been, and would do
    it on the one campaign that is definitionally correct.
    """
    harness = build(tmp_path, days=(DAY,), start=False)

    verdict = harness.runner.risk_continuity
    assert verdict.outcome is ContinuityOutcome.FIRST_START
    assert not verdict.disputed
    assert not verdict.has_history

    assert harness.runner.start() is RunnerState.READY
    assert not harness.runner.risk.state.halted
    assert continuity_records(harness.state_dir) == []


def test_a_first_start_that_wrote_its_risk_state_and_nothing_else_still_starts(tmp_path):
    """The ordering a first start really has, and why an empty log is not history.

    `build_risk_engine` seeds and persists `risk.json` BEFORE `start()` appends
    the STARTUP record, so a process that dies in between leaves a risk state
    beside an empty log. That is the ordinary shape of an interrupted first
    start, and reading it as a discontinuity would wedge a campaign that had
    never decided anything.
    """
    harness = build(tmp_path, days=(DAY,), start=False)
    harness.runner.start()
    state_dir = harness.state_dir
    config = harness.runner.config

    # Exactly the crash described above: `build_risk_engine` had persisted the
    # risk state and neither the first record nor the runner-state file existed
    # yet, so both are removed and the risk state is kept.
    assert (state_dir / "risk.json").is_file()
    # The open day file is this process's, not the crashed one's; a dead process
    # holds no handle and Windows refuses to unlink one that is held.
    if harness.runner._log is not None:
        harness.runner._log.close()
    shutil.rmtree(state_dir / "decision_log")
    (state_dir / "runner_state.json").unlink()

    resumed = build(tmp_path, days=(DAY,), config=config, start=False)
    assert resumed.runner.risk_continuity.outcome is ContinuityOutcome.FIRST_START
    assert resumed.runner.start() is RunnerState.READY


# ---------------------------------------------------------------------------
# 2. AEG-4 itself: a missing risk state after history exists
# ---------------------------------------------------------------------------
def _halted_campaign(tmp_path: Path):
    """A campaign that decided some minutes and then halted, through `_halt`."""
    harness = campaign(tmp_path)
    harness.runner._halt("a deliberate halt, so the log carries one")
    assert harness.runner.risk.state.halted
    return harness


def test_deleting_risk_json_after_a_campaign_has_run_is_refused(tmp_path):
    """Requirement B, and the audit's AEG-4 in one line: absence is not a default."""
    harness = _halted_campaign(tmp_path)
    config = harness.runner.config
    state_dir = harness.state_dir
    (state_dir / "risk.json").unlink()

    resumed = build(tmp_path, days=(DAY,), config=config, start=False)
    assert resumed.runner.risk_continuity.outcome is ContinuityOutcome.RISK_STATE_MISSING
    assert resumed.runner.risk_continuity.disputed

    assert resumed.runner.start() is RunnerState.HALT
    assert is_risk_continuity_halt(resumed.runner.halt_reason or "")
    assert is_risk_continuity_halt(resumed.runner.risk.state.halt_reason)


def test_the_defect_deleting_risk_json_used_to_clear_a_halt_silently(monkeypatch, tmp_path):
    """The other side: the pre-R1-c behaviour, reproduced against the same files.

    Without the check the campaign does not merely fail to notice. It reaches
    READY, unhalted, with equity re-seeded to the configured capital and a fresh
    `risk.json` written over the absence -- and the only record of the halt it
    dropped is the HALT record it is now running past.
    """
    harness = _halted_campaign(tmp_path)
    config = harness.runner.config
    (harness.state_dir / "risk.json").unlink()

    def first_start_always(state_dir, *, load, state):
        return RiskContinuity(outcome=ContinuityOutcome.FIRST_START, reason="", load=load)

    monkeypatch.setattr(risk_continuity, "evaluate_risk_continuity", first_start_always)
    monkeypatch.setattr("chimera.demo.runner.evaluate_risk_continuity", first_start_always)

    resumed = build(tmp_path, days=(DAY,), config=config, start=False)
    assert resumed.runner.start() is RunnerState.READY, "the defect must be reproducible"
    assert not resumed.runner.risk.state.halted
    assert resumed.runner.risk.state.equity == pytest.approx(float(CAPITAL))


def test_a_refused_missing_risk_state_does_not_reach_the_position(tmp_path):
    """Requirement G: a refusal changes nothing it did not have to change.

    `start()` refuses before `reconstruct()`, which adopts what the stores hold
    and can bootstrap one that has never been written. A campaign whose risk
    state is in dispute may not have its execution state moved on the way to
    saying so.
    """
    harness = _halted_campaign(tmp_path)
    config, state_dir = harness.runner.config, harness.state_dir
    (state_dir / "risk.json").unlink()
    stores_before = {
        name: (state_dir / name).read_bytes()
        for name in ("spot_store.json", "perp_store.json")
    }
    ledger_before = (state_dir / "carry_ledger.json").read_bytes()

    resumed = build(tmp_path, days=(DAY,), config=config, start=False)
    assert resumed.runner.start() is RunnerState.HALT

    for name, before in stores_before.items():
        assert (state_dir / name).read_bytes() == before
    assert (state_dir / "carry_ledger.json").read_bytes() == ledger_before


# ---------------------------------------------------------------------------
# 3 and 4. a risk state that exists and cannot be believed
# ---------------------------------------------------------------------------
def test_a_malformed_risk_json_with_history_is_a_continuity_dispute(tmp_path):
    """Requirement C. The bytes are left exactly as they were found."""
    harness = campaign(tmp_path)
    config, state_dir = harness.runner.config, harness.state_dir
    harness.runner.shutdown("clean stop")

    damaged = b"{ this was a risk state until something truncated it"
    (state_dir / "risk.json").write_bytes(damaged)

    resumed = build(tmp_path, days=(DAY,), config=config, start=False)
    verdict = resumed.runner.risk_continuity
    assert verdict.outcome is ContinuityOutcome.RISK_STATE_UNREADABLE
    assert resumed.runner.start() is RunnerState.HALT
    assert is_risk_continuity_halt(resumed.runner.halt_reason or "")
    assert (state_dir / "risk.json").read_bytes() == damaged, (
        "the suspect file is the only record of what the account was doing and "
        "must survive the refusal"
    )


def test_a_risk_path_that_cannot_be_examined_is_refused_rather_than_read_as_absent(
    tmp_path,
):
    """Requirement 4, at the runner. "Could not examine" must not become "missing".

    A directory where the file should be is the cross-platform way to reach it:
    the read raises an `OSError` that is not `FileNotFoundError` on every
    platform, so the load is UNREADABLE and never MISSING. The Windows mapping
    that makes a BLOCKED path look absent is exercised one test below.
    """
    harness = campaign(tmp_path)
    config, state_dir = harness.runner.config, harness.state_dir
    harness.runner.shutdown("clean stop")

    (state_dir / "risk.json").unlink()
    (state_dir / "risk.json").mkdir()

    resumed = build(tmp_path, days=(DAY,), config=config, start=False)
    verdict = resumed.runner.risk_continuity
    assert verdict.load is RiskStateLoad.UNREADABLE, "MISSING here would be the defect"
    assert verdict.outcome is ContinuityOutcome.RISK_STATE_UNREADABLE
    assert resumed.runner.start() is RunnerState.HALT
    assert (state_dir / "risk.json").is_dir()


def test_the_windows_not_found_mapping_reaches_r1c_as_unreadable(tmp_path, monkeypatch):
    """The other half of requirement 4, on the read that decides it.

    Windows reports a path blocked by a non-directory ancestor as a plain
    "not found", the same exception it uses for a file that is simply not there.
    `RiskEngine._load_state` asks the ancestors rather than the errno; this
    pins that its answer -- UNREADABLE, not MISSING -- is the one R1-c then
    refuses on, so the two halves cannot drift apart.
    """
    (tmp_path / "blocked").write_text("not a directory", encoding="utf-8")
    state = tmp_path / "blocked" / "risk.json"
    real_read_text = Path.read_text

    def windows_like(self, *args, **kwargs):
        if self == state:
            raise FileNotFoundError(3, "The system cannot find the path specified")
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", windows_like)
    engine = RiskEngine(RiskLimits(), state_path=state)
    assert engine.load_outcome is RiskStateLoad.UNREADABLE

    harness = campaign(tmp_path / "campaign")
    harness.runner.shutdown("clean stop")
    verdict = evaluate_risk_continuity(
        harness.state_dir, load=engine.load_outcome, state=engine.state
    )
    assert verdict.outcome is ContinuityOutcome.RISK_STATE_UNREADABLE
    assert verdict.disputed


def test_a_pre_schema_risk_state_cannot_continue_a_campaign_with_an_account(tmp_path):
    """The legacy document carries a halt and no account, so it continues nothing."""
    harness = campaign(tmp_path)
    config, state_dir = harness.runner.config, harness.state_dir
    harness.runner.shutdown("clean stop")

    (state_dir / "risk.json").write_text(
        json.dumps({"halted": False, "halt_reason": "", "updated_at": "2026-09-19"}),
        encoding="utf-8",
    )

    resumed = build(tmp_path, days=(DAY,), config=config, start=False)
    assert resumed.runner.risk_continuity.load is RiskStateLoad.LEGACY
    assert (
        resumed.runner.risk_continuity.outcome is ContinuityOutcome.RISK_STATE_WITHOUT_ACCOUNT
    )
    assert resumed.runner.start() is RunnerState.HALT


def test_a_first_start_onto_a_damaged_risk_state_is_not_an_r1c_dispute(tmp_path):
    """The control for the three above: with no history there is nothing to continue.

    `RiskEngine` still fails closed on the file itself -- that is its own guard
    and predates R1-c -- but the refusal is the engine's, not a continuity
    dispute, and the difference is what an operator reads.
    """
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True)
    (state_dir / "risk.json").write_bytes(b"{ not json")

    verdict = verdict_for(state_dir)
    assert verdict.load is RiskStateLoad.UNREADABLE
    assert verdict.outcome is ContinuityOutcome.FIRST_START
    assert not verdict.disputed


# ---------------------------------------------------------------------------
# 5. the healthy restart, which must stay operable
# ---------------------------------------------------------------------------
def test_a_clean_restart_continues_the_log_and_runs_again(tmp_path):
    """Requirement 5, and the single most load-bearing negative control here."""
    harness = campaign(tmp_path)
    config = harness.runner.config
    harness.runner.shutdown("clean stop")
    seq, witness = newest_hash_witness(harness.state_dir)

    resumed = build(tmp_path, days=(DAY,), config=config, start=False)
    verdict = resumed.runner.risk_continuity
    assert verdict.outcome is ContinuityOutcome.CONTINUOUS
    assert verdict.has_history
    assert verdict.observed_state_hash == witness
    assert verdict.hash_witness is not None and verdict.hash_witness.seq == seq

    assert resumed.runner.start() is RunnerState.READY
    minute = resumed.runner.cursor.next_minute_ms()
    assert minute is not None
    resumed.tick(minute)
    assert resumed.runner.state is not RunnerState.HALT, resumed.runner.halt_reason
    assert continuity_records(resumed.state_dir) == []


# ---------------------------------------------------------------------------
# 6, 7 and 8. which record is the witness, and which way the two disagree
# ---------------------------------------------------------------------------
def test_a_restart_after_a_halt_is_not_a_dispute(tmp_path):
    """Requirement 8, the side that a naive comparison gets wrong.

    `halted` and `halt_reason` are INSIDE `risk.state_hash`, so a halted campaign
    restores a state that legitimately hashes differently from the newest
    hash-bearing record -- the HALT record itself carries no hash. Comparing
    against "the log's last record that has one" would therefore dispute every
    healthy restart after a halt, which is most restarts an operator ever makes.
    """
    harness = _halted_campaign(tmp_path)
    config, state_dir = harness.runner.config, harness.state_dir

    _, witness = newest_hash_witness(state_dir)
    load, state = loaded_state(state_dir)
    assert (
        risk_state_hash(state) != witness
    ), "the halt must really have moved the hash, or this test proves nothing"

    resumed = build(tmp_path, days=(DAY,), config=config, start=False)
    verdict = resumed.runner.risk_continuity
    assert verdict.outcome is ContinuityOutcome.CONTINUOUS
    assert verdict.hash_witness is not None
    assert verdict.hash_witness.kind == RecordKind.HALT.value
    assert continuity_records(state_dir) == []


def test_a_risk_state_that_lost_its_halt_is_refused(tmp_path):
    """The AEG-4 harm reached WITHOUT deleting the file, and the halt witness.

    A halt is written to `risk.json` before the record that reports it, and only
    `resume` and `resolve-equity` ever clear one -- each of which writes a record
    of its own. So a HALT that nothing follows and a `risk.json` that is not
    halted cannot both be true of one campaign.

    The file is edited here, through `RiskState.to_dict` -- the engine's own
    writer -- because no code path produces this state and that is exactly the
    claim under test.
    """
    harness = _halted_campaign(tmp_path)
    config, state_dir = harness.runner.config, harness.state_dir

    load, state = loaded_state(state_dir)
    assert load is RiskStateLoad.LOADED and state.halted
    state.halted, state.halt_reason = False, ""
    (state_dir / "risk.json").write_text(
        json.dumps(state.to_dict(updated_at="2026-09-19T23:59:30+00:00"), indent=2),
        encoding="utf-8",
    )

    resumed = build(tmp_path, days=(DAY,), config=config, start=False)
    verdict = resumed.runner.risk_continuity
    assert verdict.outcome is ContinuityOutcome.HALT_NOT_HELD
    assert verdict.halt_witness is not None
    assert verdict.halt_witness.kind == RecordKind.HALT.value
    assert resumed.runner.start() is RunnerState.HALT
    assert is_risk_continuity_halt(resumed.runner.halt_reason or "")


def test_a_resumed_campaign_is_not_asked_to_hold_the_halt_it_cleared(tmp_path):
    """The other side of the halt witness: a RESUME is what clears it.

    Without this the guard above would refuse every campaign that was ever
    resumed, which is the same failure as refusing every restart after a halt
    and would make `resume` a one-way door.
    """
    harness = _halted_campaign(tmp_path)
    harness.runner.resume("checked both legs; the cause is gone")
    config = harness.runner.config
    assert not harness.runner.risk.state.halted

    resumed = build(tmp_path, days=(DAY,), config=config, start=False)
    verdict = resumed.runner.risk_continuity
    assert verdict.outcome is ContinuityOutcome.CONTINUOUS
    assert verdict.halt_witness is not None
    assert verdict.halt_witness.kind == RecordKind.RESUME.value
    assert resumed.runner.start() is not RunnerState.HALT


def test_an_operator_flatten_leaves_nothing_for_the_hash_to_be_compared_with(tmp_path):
    """The opaque witness, stated as a test rather than left as an assumption.

    An OPERATOR record moves the risk state and carries no hash, so after one
    there is nothing in the log for the loaded state to be compared against and
    R1-c asserts only that a state EXISTS. The residual is deliberate and is
    recorded here so that closing it -- by making those records carry a
    `risk.state_hash` -- is a visible change rather than a silent one.
    """
    harness = _halted_campaign(tmp_path)
    harness.runner.flatten("operator: reduce to flat during the incident")
    config = harness.runner.config

    resumed = build(tmp_path, days=(DAY,), config=config, start=False)
    verdict = resumed.runner.risk_continuity
    assert verdict.outcome is ContinuityOutcome.CONTINUOUS
    assert verdict.hash_witness is not None
    assert verdict.hash_witness.kind == RecordKind.OPERATOR.value
    assert verdict.hash_witness.state_hash == ""


def test_a_risk_state_restored_from_an_older_copy_is_refused(tmp_path):
    """Requirements 6 and 7: an older valid state does not pass for being valid.

    The copy is byte-for-byte what the campaign itself wrote four minutes before
    the UTC day rolled, so it parses, it loads LOADED, and every field in it is
    one a real campaign held. What makes it refusable is that the log records
    that state at an EARLIER witness and not at its newest: the campaign has
    already moved past it, and no crash moves a persisted state backwards.
    """
    harness, earlier = across_the_boundary(tmp_path)
    config, state_dir = harness.runner.config, harness.state_dir
    seq, newest = newest_hash_witness(state_dir)

    shutil.copyfile(earlier, state_dir / "risk.json")
    load, state = loaded_state(state_dir)
    assert load is RiskStateLoad.LOADED, "the stale copy must be a VALID risk state"
    assert (
        risk_state_hash(state) != newest
    ), "the day boundary must really have moved the state, or the fixture is vacuous"

    resumed = build(tmp_path, days=(DAY, NEXT_DAY), config=config, start=False)
    verdict = resumed.runner.risk_continuity
    assert verdict.outcome is ContinuityOutcome.STATE_HASH_REGRESSED
    assert verdict.matched_witness is not None
    assert verdict.matched_witness.seq < seq
    assert resumed.runner.start() is RunnerState.HALT
    assert is_risk_continuity_halt(resumed.runner.halt_reason or "")


def test_operational_records_after_the_witness_do_not_erase_the_check(tmp_path):
    """Requirement 9. The scan walks PAST the kinds that cannot have moved the state.

    The log here ends on a SHUTDOWN -- an operational record that carries no
    `risk.state_hash` -- and the stale state is still caught, because the witness
    is chosen by kind rather than by position. A check that gave up when the
    physically last record carried no hash would let any discontinuity through
    behind one clean stop.
    """
    harness, earlier = across_the_boundary(tmp_path)
    config, state_dir = harness.runner.config, harness.state_dir

    written = records(state_dir)
    assert written[-1]["kind"] == RecordKind.SHUTDOWN.value
    assert "risk" not in written[-1]

    shutil.copyfile(earlier, state_dir / "risk.json")
    resumed = build(tmp_path, days=(DAY, NEXT_DAY), config=config, start=False)
    verdict = resumed.runner.risk_continuity
    assert verdict.outcome is ContinuityOutcome.STATE_HASH_REGRESSED
    assert verdict.hash_witness is not None
    assert verdict.hash_witness.kind == RecordKind.DECISION.value


def test_a_risk_state_no_record_quotes_is_the_ordinary_crash_and_not_a_refusal(tmp_path):
    """The direction that must NOT fail closed, and the crash that produces it.

    `tick` persists Aegis and then commits the DECISION that quotes it, so a
    process killed between the two leaves a risk state no witness records. That
    is section 9.3's `LOG_BEHIND_STATE` -- the same direction the legs and the
    carry ledger are already recovered in -- and refusing it would turn an
    ordinary SIGKILL into a terminal halt.
    """

    class _SimulatedKill(RuntimeError):
        pass

    harness = build(tmp_path, days=(DAY, NEXT_DAY))
    first = harness.first_minute_ms() + BOUNDARY_START * 60_000
    harness.run(MINUTES_BEFORE_THE_BOUNDARY, start=first)
    config, state_dir = harness.runner.config, harness.state_dir
    real_append = harness.runner._append

    def killed(kind, *args, **kwargs):
        if kind is RecordKind.DECISION:
            raise _SimulatedKill()
        return real_append(kind, *args, **kwargs)

    harness.runner._append = killed
    with pytest.raises(_SimulatedKill):
        # The day roll moves `day_start_equity`, which is inside the hash, so the
        # state this minute persists really is one no record can quote.
        harness.tick(first + MINUTES_BEFORE_THE_BOUNDARY * 60_000)

    _, newest = newest_hash_witness(state_dir)
    load, state = loaded_state(state_dir)
    assert risk_state_hash(state) != newest, "the crash window must really be open"

    verdict = verdict_for(state_dir)
    assert verdict.outcome is ContinuityOutcome.STATE_AHEAD_OF_LOG
    assert not verdict.disputed
    assert verdict.matched_witness is None

    resumed = build(tmp_path, days=(DAY, NEXT_DAY), config=config, start=False)
    assert resumed.runner.start() is not RunnerState.HALT, resumed.runner.halt_reason
    written = [
        record
        for record in records(state_dir)
        if record.get("kind") == RecordKind.RECOVERY.value
    ]
    assert written, "the crash must still produce a RECOVERY"
    assert written[-1]["recovery"]["cause"] == RecoveryCause.LOG_BEHIND_STATE.value
    assert continuity_records(state_dir) == []


# ---------------------------------------------------------------------------
# the RECOVERY record R1-c owes the log
# ---------------------------------------------------------------------------
def test_the_refusal_writes_one_recovery_record_naming_what_it_compared(tmp_path):
    """Requirement F: the record is the evidence, so it names both sides."""
    harness = _halted_campaign(tmp_path)
    config, state_dir = harness.runner.config, harness.state_dir
    seq, witness = newest_hash_witness(state_dir)
    (state_dir / "risk.json").unlink()

    resumed = build(tmp_path, days=(DAY,), config=config, start=False)
    resumed.runner.start()

    written = continuity_records(state_dir)
    assert len(written) == 1
    recovery = written[0]["recovery"]
    block = recovery["risk_continuity"]
    assert block["outcome"] == ContinuityOutcome.RISK_STATE_MISSING.value
    assert block["risk_state_load"] == RiskStateLoad.MISSING.value
    assert block["halt_witness_kind"] == RecordKind.HALT.value
    assert "observed_state_hash" not in block, "there was no state to have an identity"
    assert (
        recovery["evidence_excluded_minute"] is None
    ), "the campaign decided nothing, so there is no minute to exclude"
    assert recovery["truncated_bytes"] == 0
    assert "records_verified" not in recovery


def test_the_file_a_refused_start_leaves_behind_is_a_halted_placeholder(tmp_path):
    """A disclosed consequence, pinned so that it is a decision and not a surprise.

    `start()` has to BUILD Aegis before it can halt it, and `build_risk_engine`
    seeds a first start's equity and persists it -- so by the time a MISSING
    verdict can be acted on, a `risk.json` exists again. What it holds is the
    configured capital and the refusal, which is a placeholder and not the
    campaign's risk state; the durable evidence that the file was missing is the
    RECOVERY record above. An operator restores their backup over this file
    (docs/demo_runbook.md, section 4).

    Suppressing the write means changing `seed_or_reconcile_equity`, which is
    R1-b's merged code, and R1-c does not touch it.
    """
    harness = _halted_campaign(tmp_path)
    config, state_dir = harness.runner.config, harness.state_dir
    (state_dir / "risk.json").unlink()

    resumed = build(tmp_path, days=(DAY,), config=config, start=False)
    assert resumed.runner.start() is RunnerState.HALT

    persisted = json.loads((state_dir / "risk.json").read_text(encoding="utf-8"))
    assert persisted["halted"] is True
    assert is_risk_continuity_halt(persisted["halt_reason"])
    assert persisted["equity"] == pytest.approx(float(CAPITAL))


def test_a_second_start_does_not_write_a_second_recovery_record(tmp_path):
    """Requirement F again. One event, one record, however often it is restarted.

    A refused campaign is restarted -- by an operator, by a supervisor, by a
    systemd restart loop -- and the discontinuity is still there each time. The
    verdict also changes shape between the two starts, which is the subtler half:
    the first refusal seeds and persists a `risk.json`, so a second start no
    longer finds one MISSING. It finds one halted on the refusal, and reports
    that same refusal rather than a differently-named one.
    """
    harness = _halted_campaign(tmp_path)
    config, state_dir = harness.runner.config, harness.state_dir
    (state_dir / "risk.json").unlink()

    first = build(tmp_path, days=(DAY,), config=config, start=False)
    assert first.runner.start() is RunnerState.HALT
    first_reason = first.runner.risk.state.halt_reason

    second = build(tmp_path, days=(DAY,), config=config, start=False)
    verdict = second.runner.risk_continuity
    assert verdict.outcome is ContinuityOutcome.ALREADY_DISPUTED
    assert verdict.reason == first_reason
    assert second.runner.start() is RunnerState.HALT
    assert second.runner.halt_reason == first_reason

    assert len(continuity_records(state_dir)) == 1


def test_the_fingerprint_is_what_decides_a_repeat(tmp_path):
    """The dedup key is a function of the verdict, not of when it was taken."""
    harness = _halted_campaign(tmp_path)
    config, state_dir = harness.runner.config, harness.state_dir
    (state_dir / "risk.json").unlink()

    before = verdict_for(state_dir)
    assert not continuity_already_recorded(state_dir, before.fingerprint)

    resumed = build(tmp_path, days=(DAY,), config=config, start=False)
    resumed.runner.start()
    assert continuity_already_recorded(state_dir, before.fingerprint)
    assert not continuity_already_recorded(state_dir, "sha256:" + "0" * 64)


# ---------------------------------------------------------------------------
# 10. R1-b is not disturbed
# ---------------------------------------------------------------------------
def test_the_r1b_equity_dispute_still_reaches_the_operator(tmp_path):
    """Requirement 10. R1-c must not stand in front of R1-b's dispute.

    The crash R1-b names -- a process killed between the ledger write and
    `update_equity` -- leaves the risk state exactly where the last record says
    it is, so continuity is intact and the halt an operator meets is still the
    equity dispute, with the clearing path R1-b gave it.
    """

    class _SimulatedKill(RuntimeError):
        pass

    harness = build(tmp_path, days=(DAY, NEXT_DAY))
    first = harness.first_minute_ms() + BOUNDARY_START * 60_000
    harness.run(MINUTES_BEFORE_THE_BOUNDARY, start=first)
    config, state_dir = harness.runner.config, harness.state_dir

    def killed(*args, **kwargs):
        raise _SimulatedKill()

    harness.runner.risk.update_equity = killed
    with pytest.raises(_SimulatedKill):
        harness.tick(first + MINUTES_BEFORE_THE_BOUNDARY * 60_000)

    verdict = verdict_for(state_dir)
    assert (
        verdict.outcome is ContinuityOutcome.CONTINUOUS
    ), "the risk state did not move in this window, so R1-c has nothing to say"

    resumed = build(tmp_path, days=(DAY, NEXT_DAY), config=config, start=False)
    assert resumed.runner.start() is RunnerState.HALT
    assert resumed.runner.risk.state.halt_reason.startswith(EQUITY_DISPUTE_PREFIX)


def test_a_flatten_then_a_restart_still_runs(tmp_path):
    """Requirement 10, the other R1-b witness: no new refusal on that path."""
    harness = campaign(tmp_path, minutes=12)
    harness.runner.flatten("operator: reduce to flat for maintenance")
    config = harness.runner.config

    resumed = build(tmp_path, days=(DAY,), config=config, start=False)
    assert resumed.runner.risk_continuity.outcome is ContinuityOutcome.CONTINUOUS
    assert resumed.runner.start() is not RunnerState.HALT, resumed.runner.halt_reason


# ---------------------------------------------------------------------------
# the operator's read
# ---------------------------------------------------------------------------
def test_status_reports_the_verdict_without_starting_the_campaign(tmp_path):
    """`status` never calls `start()`, and a refused campaign is what it is run on."""
    from tools.demo_run import _status

    harness = _halted_campaign(tmp_path)
    config, state_dir = harness.runner.config, harness.state_dir
    (state_dir / "risk.json").unlink()

    resumed = build(tmp_path, days=(DAY,), config=config, start=False)
    reported = _status(resumed.runner)["risk_continuity"]

    assert reported["outcome"] == ContinuityOutcome.RISK_STATE_MISSING.value
    assert reported["disputed"] is True
    assert reported["reason"].startswith(RISK_CONTINUITY_PREFIX)
    assert resumed.runner.state is RunnerState.STARTUP, "status must not have started it"
