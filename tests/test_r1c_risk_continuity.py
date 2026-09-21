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
`DemoRunner.tick`, `_halt`, `flatten`, `resume`, `shutdown` and, where an
operator's path is the point, the real `tools/demo_run.main`; the states are
whatever those leave on disk. What the tests then do TO those files -- delete
`risk.json`, restore an EARLIER copy of it, forge or tear a log line, kill a
startup at one statement -- is the damage under test, and each says so where it
does it: none of those states is reachable by the runner writing a file, which
is the whole point of refusing them.

Nothing here is scientific evidence. The prices come from `chimera.demo.fixtures`
and every quantity below is the test author's own arithmetic.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

import chimera.demo.risk_continuity as risk_continuity
from chimera.carry.hedge import HedgeState
from chimera.demo.decision_log import EVIDENCE_KINDS, RecordKind
from chimera.demo.risk_continuity import (
    RISK_CONTINUITY_PREFIX,
    RISK_HASH_EXCLUDED,
    ContinuityOutcome,
    RiskContinuity,
    continuity_already_recorded,
    evaluate_risk_continuity,
    halt_free_state_hash,
    is_risk_continuity_halt,
    risk_state_hash,
)
from chimera.demo.risk_wiring import EQUITY_DISPUTE_PREFIX, ledger_equity
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
    """The RECOVERY records R1-c wrote -- refusals AND settlements -- and only those."""
    return [
        record
        for record in records(state_dir)
        if record.get("kind") == RecordKind.RECOVERY.value
        and (record.get("recovery") or {}).get("cause")
        == RecoveryCause.RISK_STATE_DISCONTINUITY.value
    ]


def refusal_records(state_dir: Path) -> list[dict[str, Any]]:
    """The continuity RECOVERY records that OPENED a refusal."""
    return [
        record
        for record in continuity_records(state_dir)
        if record["recovery"]["risk_continuity"]["outcome"]
        != ContinuityOutcome.CONTINUOUS.value
    ]


def settlement_records(state_dir: Path) -> list[dict[str, Any]]:
    """The continuity RECOVERY records that SETTLED one."""
    return [
        record
        for record in continuity_records(state_dir)
        if record["recovery"]["risk_continuity"]["outcome"]
        == ContinuityOutcome.CONTINUOUS.value
    ]


def decisions_after(state_dir: Path, seq: int) -> list[dict[str, Any]]:
    """Every record carrying a `risk.state_hash` written after ``seq``."""
    return [
        record
        for record in records(state_dir)
        if int(record["seq"]) > seq and (record.get("risk") or {}).get("state_hash")
    ]


def last_seq(state_dir: Path) -> int:
    return max(int(record["seq"]) for record in records(state_dir))


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


def _evidence_count(state_dir: Path) -> int:
    """How many records section 9.4 scores. A refused startup must add none."""
    scored = {kind.value for kind in EVIDENCE_KINDS}
    return sum(1 for record in records(state_dir) if record.get("kind") in scored)


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
    # And it seeds and persists exactly as it always did: holding a DISPUTED
    # file as evidence must not become holding every file.
    load, seeded = loaded_state(harness.state_dir)
    assert load is RiskStateLoad.LOADED
    assert seeded.equity == pytest.approx(float(CAPITAL))


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

    What makes it pass is NOT that the scan gives up at the HALT. That is the
    defect the independent review measured, and it let an older halted state
    through a newer HALT history for the same reason. The scan walks past the
    hashless HALT to the DECISION behind it, and what matches that DECISION is
    the state's identity with the halt set aside -- `RiskEngine.halt` sets
    `halted` and `halt_reason` and nothing else, so that identity is exactly the
    one this campaign last recorded.
    """
    harness = _halted_campaign(tmp_path)
    config, state_dir = harness.runner.config, harness.state_dir

    seq, witness = newest_hash_witness(state_dir)
    load, state = loaded_state(state_dir)
    assert (
        risk_state_hash(state) != witness
    ), "the halt must really have moved the hash, or this test proves nothing"

    resumed = build(tmp_path, days=(DAY,), config=config, start=False)
    verdict = resumed.runner.risk_continuity
    assert verdict.outcome is ContinuityOutcome.CONTINUOUS
    # The witness is the hash-bearing record BEHIND the halt, not the halt.
    assert verdict.hash_witness is not None
    assert verdict.hash_witness.kind == RecordKind.DECISION.value
    assert verdict.hash_witness.seq == seq
    # And it is the halt-normalised identity that matches it, which is the whole
    # of why this restart is not a dispute.
    assert verdict.unhalted_state_hash == witness
    assert verdict.observed_state_hash != witness
    assert verdict.halt_witness is not None
    assert verdict.halt_witness.kind == RecordKind.HALT.value
    assert continuity_records(state_dir) == []

    assert resumed.runner.start() is RunnerState.HALT
    assert resumed.runner.halt_reason == state.halt_reason
    assert continuity_records(state_dir) == [], "a held halt is not a discontinuity"


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


def test_an_operator_flatten_moves_the_state_to_one_no_witness_records(tmp_path):
    """The residual after an OPERATOR command, stated exactly rather than widely.

    `flatten` hands the flattened equity to `update_equity` and writes an
    OPERATOR record that carries no hash, so the state it leaves is one no
    witness records and R1-c cannot say the file IS that state. What it can still
    say -- and what the independent review found it was not saying -- is that the
    file is not a state the campaign has already moved PAST: the scan walks past
    the OPERATOR record to the DECISION behind it, and
    `test_a_risk_state_restored_from_an_older_copy_survives_an_operator_tail`
    is the other side of this one.

    The residual is that an identity no witness records is accepted, which is the
    same direction the crash window is recovered in. Closing it means making
    OPERATOR records carry a `risk.state_hash`, which is a change to what the
    runner WRITES rather than to what it checks.
    """
    harness = _halted_campaign(tmp_path)
    harness.runner.flatten("operator: reduce to flat during the incident")
    config, state_dir = harness.runner.config, harness.state_dir

    resumed = build(tmp_path, days=(DAY,), config=config, start=False)
    verdict = resumed.runner.risk_continuity
    assert verdict.outcome is ContinuityOutcome.CONTINUOUS
    # The witness is a real one now, and the flatten really did move the state
    # off it -- so this is the "no witness records it" case and not a match.
    assert verdict.hash_witness is not None
    assert verdict.hash_witness.state_hash == newest_hash_witness(state_dir)[1]
    assert verdict.observed_state_hash != verdict.hash_witness.state_hash
    assert verdict.unhalted_state_hash != verdict.hash_witness.state_hash
    assert verdict.matched_witness is None
    # And it is NOT reported as the crash window, because a record in the tail
    # says the state was moved rather than lost.
    assert verdict.outcome is not ContinuityOutcome.STATE_AHEAD_OF_LOG
    assert continuity_records(state_dir) == []


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


def test_a_refused_start_leaves_the_missing_risk_state_missing(tmp_path):
    """The evidence is the absence, and a refusal writes nothing in its place.

    `start()` has to BUILD Aegis before it can halt it, and `build_risk_engine`
    seeds a first start's equity. On a dispute that engine is built with
    ``persist=False``: the seed, the constructor's kill-switch check and the
    refusal's own halt all happen in memory, and none of them reaches the file.
    An earlier revision let them through and left a placeholder holding the
    configured capital where the absence had been -- which a crash, a failed
    append or a forged log could then leave standing with nothing saying what it
    was (the independent review's second blocking finding).
    """
    harness = _halted_campaign(tmp_path)
    config, state_dir = harness.runner.config, harness.state_dir
    (state_dir / "risk.json").unlink()

    resumed = build(tmp_path, days=(DAY,), config=config, start=False)
    assert resumed.runner.start() is RunnerState.HALT
    # Halted where it matters -- in the engine the runner enforces with.
    assert resumed.runner.risk.state.halted
    assert is_risk_continuity_halt(resumed.runner.risk.state.halt_reason)

    assert not (state_dir / "risk.json").exists()
    assert not (state_dir / "risk.json.tmp").exists()


def test_a_second_start_does_not_write_a_second_recovery_record(tmp_path):
    """Requirement F again. One event, one record, however often it is restarted.

    A refused campaign is restarted -- by an operator, by a supervisor, by a
    systemd restart loop -- and the discontinuity is still there each time. It is
    there UNCHANGED: the first refusal wrote nothing to the file, and everything
    it wrote to the log is skipped as the work of a refused process, so the
    second start takes the same verdict with the same fingerprint and the dedup
    has nothing to reconcile.
    """
    harness = _halted_campaign(tmp_path)
    config, state_dir = harness.runner.config, harness.state_dir
    (state_dir / "risk.json").unlink()

    first = build(tmp_path, days=(DAY,), config=config, start=False)
    assert first.runner.start() is RunnerState.HALT
    first_verdict = first.runner.risk_continuity

    second = build(tmp_path, days=(DAY,), config=config, start=False)
    verdict = second.runner.risk_continuity
    assert verdict.outcome is ContinuityOutcome.RISK_STATE_MISSING
    assert verdict.fingerprint == first_verdict.fingerprint
    assert verdict.reason == first_verdict.reason
    assert second.runner.start() is RunnerState.HALT
    assert second.runner.halt_reason == first_verdict.reason

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


#: R1-b's settlement anchor, reused because it is the cheapest production-driven
#: way to move the EQUITY rather than only the day's starting equity: the carry
#: hedge is delta-neutral, so the mark moves equity only at a funding settlement,
#: and the minute whose close is the 08:00 boundary is this one.
MINUTE_BEFORE_A_SETTLEMENT = 479


def test_resolve_equity_refuses_while_a_continuity_dispute_stands(tmp_path):
    """The one command that could clear an R1-c refusal by answering something else.

    A `risk.json` restored from an older copy usually disagrees with the ledger
    too, so `seed_or_reconcile_equity` halts Aegis on `equity_dispute:` while the
    engine is being built -- and `RiskEngine.halt` keeps the FIRST reason, so the
    continuity refusal never reaches the file. Without this guard
    `resolve --equity` would settle the equity dispute, clear the halt, and write
    an OPERATOR record that becomes the log's newest risk witness: the next start
    would find nothing left to compare and the discontinuity would be gone.

    The copy is taken one minute before the 08:00 funding settlement, which is
    what makes the two equities differ at all.
    """
    from chimera.demo.runner import RunnerError

    harness = build(tmp_path, days=(DAY,))
    first = harness.first_minute_ms()
    harness.run(MINUTE_BEFORE_A_SETTLEMENT, start=first)
    earlier = tmp_path / "risk.json.before-the-settlement"
    shutil.copyfile(harness.state_dir / "risk.json", earlier)
    harness.run(2, start=first + MINUTE_BEFORE_A_SETTLEMENT * 60_000)
    harness.runner.shutdown("clean stop after the settlement")
    config, state_dir = harness.runner.config, harness.state_dir

    shutil.copyfile(earlier, state_dir / "risk.json")
    resumed = build(tmp_path, days=(DAY,), config=config, start=False)
    assert resumed.runner.risk_continuity.outcome is ContinuityOutcome.STATE_HASH_REGRESSED
    assert resumed.runner.start() is RunnerState.HALT
    assert resumed.runner.risk.state.halt_reason.startswith(EQUITY_DISPUTE_PREFIX), (
        "the fixture must really reach the overlap: Aegis holding the equity "
        "dispute while the runner is refused on continuity"
    )

    with pytest.raises(RunnerError, match="does not continue the decision log"):
        resumed.runner.resolve_equity("the ledger is the campaign's accounting")

    # Nothing moved: the halt Aegis holds is the one it held before the refusal.
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


# ---------------------------------------------------------------------------
# 11. the documented recovery, end to end. The independent review's blocker 1A:
#     R1-c's own refusal HALT read back as campaign evidence condemned the very
#     backup the runbook tells an operator to restore, and each retry appended
#     another RECOVERY record and replaced the correct file with another halted
#     placeholder. No previous revision of this PR contained this test.
# ---------------------------------------------------------------------------
def backup_of(state_dir: Path, where: Path) -> bytes:
    """The bytes the campaign itself last wrote to `risk.json`, kept aside.

    Copying is the one thing here no production path does; every byte in the
    copy was written by one. The bytes are returned as well as stored so a test
    can assert that what it restored is what the campaign wrote.
    """
    payload = (Path(state_dir) / "risk.json").read_bytes()
    where.write_bytes(payload)
    return payload


def test_restoring_the_byte_perfect_backup_after_a_refusal_starts_the_campaign(tmp_path):
    """Refuse -> restore -> start, with no hand-editing anywhere in it.

    This is the whole of section 4's recovery and it used to be unreachable. The
    refusal halts, `_halt` writes a HALT record, and reading that record back as
    ordinary campaign history made the log's newest halt witness a HALT that
    nothing cleared -- so the restored state, which is legitimately NOT halted,
    was refused as HALT_NOT_HELD. Each retry appended another RECOVERY and left
    another halted placeholder in place of the correct file, and the only escape
    was the hand-editing the runbook forbids.
    """
    harness = campaign(tmp_path)
    harness.runner.shutdown("clean stop")
    config, state_dir = harness.runner.config, harness.state_dir
    saved = backup_of(state_dir, tmp_path / "risk.json.backup")

    _, witness = newest_hash_witness(state_dir)
    load, state = loaded_state(state_dir)
    assert load is RiskStateLoad.LOADED
    assert risk_state_hash(state) == witness, (
        "the backup must be the state the log's newest witness records, which is "
        "what section 4 means by a copy at least as recent as that record"
    )

    (state_dir / "risk.json").unlink()
    refused = build(tmp_path, days=(DAY,), config=config, start=False)
    assert refused.runner.start() is RunnerState.HALT
    assert is_risk_continuity_halt(refused.runner.halt_reason or "")
    assert len(refusal_records(state_dir)) == 1
    refusal_seq = int(refusal_records(state_dir)[0]["seq"])
    # The refusal left the absence exactly as it found it.
    assert not (state_dir / "risk.json").exists()

    (state_dir / "risk.json").write_bytes(saved)
    restored = build(tmp_path, days=(DAY,), config=config, start=False)
    assert restored.runner.risk_continuity.outcome is ContinuityOutcome.CONTINUOUS
    assert restored.runner.risk_continuity.settles_seq == refusal_seq
    assert restored.runner.start() is RunnerState.READY, restored.runner.halt_reason

    # The correct backup is not condemned, not overwritten and not refused
    # again; the restore is written down once, as the settlement it is.
    assert (state_dir / "risk.json").read_bytes() == saved
    assert len(refusal_records(state_dir)) == 1
    settled = settlement_records(state_dir)
    assert len(settled) == 1
    assert settled[0]["recovery"]["risk_continuity"]["settles_seq"] == refusal_seq

    minute = restored.runner.cursor.next_minute_ms()
    assert minute is not None
    restored.tick(minute)
    assert restored.runner.state is not RunnerState.HALT, restored.runner.halt_reason


def test_repeated_restore_attempts_do_not_grow_one_discontinuity_into_many(tmp_path):
    """One physical event, however many times an operator or a supervisor tries.

    Two sequences, because they failed differently. Repeated REFUSED starts --
    a systemd restart loop against an unrestored file -- must write one record;
    and repeated RESTORE attempts must not each append another and replace the
    file again.
    """
    harness = campaign(tmp_path)
    harness.runner.shutdown("clean stop")
    config, state_dir = harness.runner.config, harness.state_dir
    saved = backup_of(state_dir, tmp_path / "risk.json.backup")
    (state_dir / "risk.json").unlink()

    for _ in range(4):
        attempt = build(tmp_path, days=(DAY,), config=config, start=False)
        assert attempt.runner.start() is RunnerState.HALT
        assert is_risk_continuity_halt(attempt.runner.halt_reason or "")
    assert len(continuity_records(state_dir)) == 1

    for _ in range(3):
        (state_dir / "risk.json").write_bytes(saved)
        attempt = build(tmp_path, days=(DAY,), config=config, start=False)
        assert attempt.runner.start() is RunnerState.READY, attempt.runner.halt_reason
        assert (state_dir / "risk.json").read_bytes() == saved
    # One refusal and the one settlement that closed it; the later starts found
    # nothing open and wrote nothing.
    assert len(refusal_records(state_dir)) == 1
    assert len(settlement_records(state_dir)) == 1


def test_a_second_genuinely_new_discontinuity_gets_its_own_record(tmp_path):
    """The other side of the dedup: one event per event, not one event ever.

    The first is settled the way one is settled -- the right file is back and the
    campaign decides another minute, which is the hash-bearing record that says
    so -- and only then is a second discontinuity created.
    """
    harness = campaign(tmp_path)
    harness.runner.shutdown("clean stop")
    config, state_dir = harness.runner.config, harness.state_dir
    saved = backup_of(state_dir, tmp_path / "risk.json.backup")

    (state_dir / "risk.json").unlink()
    first = build(tmp_path, days=(DAY,), config=config, start=False)
    assert first.runner.start() is RunnerState.HALT
    assert len(continuity_records(state_dir)) == 1

    (state_dir / "risk.json").write_bytes(saved)
    settled = build(tmp_path, days=(DAY,), config=config, start=False)
    assert settled.runner.start() is RunnerState.READY
    minute = settled.runner.cursor.next_minute_ms()
    assert minute is not None
    settled.tick(minute)
    settled.runner.shutdown("clean stop after the restore")

    (state_dir / "risk.json").unlink()
    second = build(tmp_path, days=(DAY,), config=config, start=False)
    assert second.runner.start() is RunnerState.HALT
    written = refusal_records(state_dir)
    assert len(written) == 2
    assert len(settlement_records(state_dir)) == 1
    assert (
        written[0]["recovery"]["risk_continuity"]["fingerprint_hash"]
        != written[1]["recovery"]["risk_continuity"]["fingerprint_hash"]
    ), "two different events must not share one fingerprint"


# ---------------------------------------------------------------------------
# 12. the review's blocker 1B: the continuity dispute must outlive the process
#     that detected it, even when R1-b's halt reason won first-reason-wins and
#     the continuity reason never reached `risk.json` at all.
# ---------------------------------------------------------------------------
def test_the_continuity_dispute_survives_a_restart_it_never_reached_risk_json_in(
    tmp_path,
):
    """Process #1 refuses; process #2 must refuse too, and for the same event.

    The overlap is natural rather than arranged: an older `risk.json` restored
    beside a ledger the campaign has since marked. `seed_or_reconcile_equity`
    halts Aegis on `equity_dispute:` while the engine is still being built, and
    `RiskEngine.halt` keeps the FIRST reason -- so the continuity refusal is in
    the runner and in the log and NOT in Aegis's reason. The independent review
    measured the dispute evaporating on the next start, `resolve --equity`
    succeeding on a rolled-back state, and the campaign reaching READY on a risk
    state whose `day_start_equity` was a whole day stale.

    What carries the dispute across the restart now is the file itself: neither
    halt reached it, so process #2 reads the same stale bytes and reaches the
    same verdict on its own.
    """
    from chimera.demo.runner import RunnerError

    harness, earlier = across_the_boundary(tmp_path)
    config, state_dir = harness.runner.config, harness.state_dir
    current = backup_of(state_dir, tmp_path / "risk.json.current")

    shutil.copyfile(earlier, state_dir / "risk.json")
    stale = (state_dir / "risk.json").read_bytes()
    assert stale != current, "the restored copy must really be an older one"
    _, stale_state = loaded_state(state_dir)
    accounted = ledger_equity(state_dir, capital=CAPITAL)
    assert accounted is not None and float(accounted) != stale_state.equity, (
        "the ledger must be genuinely newer than the restored risk state, or the "
        "R1-b overlap this test is about does not exist"
    )

    first = build(tmp_path, days=(DAY, NEXT_DAY), config=config, start=False)
    assert first.runner.risk_continuity.outcome is ContinuityOutcome.STATE_HASH_REGRESSED
    assert first.runner.start() is RunnerState.HALT
    assert is_risk_continuity_halt(first.runner.halt_reason or "")
    assert first.runner.risk.state.halt_reason.startswith(EQUITY_DISPUTE_PREFIX), (
        "first-reason-wins must really have put the OTHER reason in Aegis, or this "
        "test proves nothing about durability"
    )
    with pytest.raises(RunnerError, match="does not continue the decision log"):
        first.runner.resolve_equity("the ledger is the campaign's accounting")
    assert (
        state_dir / "risk.json"
    ).read_bytes() == stale, (
        "neither halt may reach the file: it is the evidence the next start reads"
    )

    # A fresh process, reading only what is on disk.
    second = build(tmp_path, days=(DAY, NEXT_DAY), config=config, start=False)
    assert second.runner is not first.runner
    verdict = second.runner.risk_continuity
    assert verdict.outcome is ContinuityOutcome.STATE_HASH_REGRESSED
    assert verdict.disputed
    assert verdict.fingerprint == first.runner.risk_continuity.fingerprint
    assert second.runner.start() is RunnerState.HALT
    with pytest.raises(RunnerError, match="does not continue the decision log"):
        second.runner.resolve_equity("the ledger is the campaign's accounting")
    assert len(continuity_records(state_dir)) == 1

    # A third, because a supervisor restarts more than once.
    third = build(tmp_path, days=(DAY, NEXT_DAY), config=config, start=False)
    assert third.runner.risk_continuity.disputed
    assert third.runner.start() is RunnerState.HALT
    assert len(continuity_records(state_dir)) == 1

    assert (state_dir / "risk.json").read_bytes() == stale

    # And the supported way out still works: put the right file back.
    (state_dir / "risk.json").write_bytes(current)
    healed = build(tmp_path, days=(DAY, NEXT_DAY), config=config, start=False)
    assert healed.runner.risk_continuity.outcome is ContinuityOutcome.CONTINUOUS
    assert healed.runner.start() is RunnerState.READY, healed.runner.halt_reason
    assert len(settlement_records(state_dir)) == 1


def test_resume_cannot_settle_a_continuity_refusal_either(tmp_path):
    """The other command that could clear the halt by answering nothing.

    `resume` clears Aegis's halt and writes a RESUME record, after which the
    campaign may tick -- and a tick's DECISION record is the one thing that
    settles an open continuity dispute. So a refusal an operator could resume out
    of is one they could erase by pressing on, which is the shape `resolve
    --equity` is already refused for.
    """
    from chimera.demo.runner import RunnerError

    harness = campaign(tmp_path)
    harness.runner.shutdown("clean stop")
    config, state_dir = harness.runner.config, harness.state_dir
    (state_dir / "risk.json").unlink()

    refused = build(tmp_path, days=(DAY,), config=config, start=False)
    assert refused.runner.start() is RunnerState.HALT
    with pytest.raises(RunnerError, match="does not continue the decision log"):
        refused.runner.resume("operator: I looked at it")
    assert refused.runner.state is RunnerState.HALT
    assert refused.runner.risk.state.halted


def test_flatten_still_reduces_exposure_during_a_continuity_dispute(tmp_path):
    """And must not clear it. Reducing while the risk state is disputed is the point.

    R1-b's boundary, restated as a control here because the two guards above sit
    next to it: `flatten` is the one operator action a refused campaign still
    needs, and the OPERATOR record it writes carries no `risk.state_hash`, so it
    cannot settle the dispute the way a decided minute does.
    """
    harness = campaign(tmp_path, minutes=12)
    assert (
        harness.runner.position.state is not HedgeState.FLAT
    ), "the fixture must hold something for a flatten to reduce"
    harness.runner.shutdown("clean stop")
    config, state_dir = harness.runner.config, harness.state_dir
    (state_dir / "risk.json").unlink()

    refused = build(tmp_path, days=(DAY,), config=config, start=False)
    assert refused.runner.start() is RunnerState.HALT
    refused.runner.flatten("operator: reduce to flat while the risk state is disputed")
    assert refused.runner.position.state is HedgeState.FLAT
    assert not (
        state_dir / "risk.json"
    ).exists(), "the flatten moved Aegis's equity and exposures, in memory only"

    after = build(tmp_path, days=(DAY,), config=config, start=False)
    assert after.runner.risk_continuity.disputed, "a flatten may not clear the dispute"
    assert after.runner.start() is RunnerState.HALT
    assert len(continuity_records(state_dir)) == 1


# ---------------------------------------------------------------------------
# 13. the review's blocker 2: a hashless record may not switch the comparison
#     off. Two-sided for each of the three tails.
# ---------------------------------------------------------------------------
def _stale_beside(tmp_path: Path):
    """A campaign past a day boundary, and the `risk.json` it held before it."""
    harness, earlier = across_the_boundary(tmp_path)
    return harness, earlier


def _restore_and_read(state_dir: Path, earlier: Path) -> RiskContinuity:
    current = risk_state_hash(loaded_state(state_dir)[1])
    shutil.copyfile(earlier, Path(state_dir) / "risk.json")
    stale = risk_state_hash(loaded_state(state_dir)[1])
    assert stale != current, "the restored copy must really be an older state"
    return verdict_for(state_dir)


def test_a_risk_state_restored_from_an_older_copy_survives_a_resume_tail(tmp_path):
    """Blocker 2A. A RESUME record carries no hash and must not end the scan."""
    harness, earlier = _stale_beside(tmp_path)
    state_dir = harness.state_dir
    harness.runner._halt("a deliberate halt, so RESUME has something to clear")
    harness.runner.resume("operator: checked, resuming")
    assert records(state_dir)[-1]["kind"] == RecordKind.RESUME.value

    verdict = _restore_and_read(state_dir, earlier)
    assert verdict.outcome is ContinuityOutcome.STATE_HASH_REGRESSED
    assert verdict.disputed
    assert verdict.hash_witness is not None
    assert verdict.hash_witness.state_hash, "the scan must have walked past the RESUME"
    assert verdict.matched_witness is not None


def test_a_healthy_restart_after_a_resume_is_not_a_dispute(tmp_path):
    """The other side. `resume` is an ordinary operator action and must stay one."""
    harness = campaign(tmp_path)
    config, state_dir = harness.runner.config, harness.state_dir
    harness.runner._halt("a deliberate halt, so RESUME has something to clear")
    harness.runner.resume("operator: checked, resuming")
    assert not harness.runner.risk.state.halted

    resumed = build(tmp_path, days=(DAY,), config=config, start=False)
    assert resumed.runner.risk_continuity.outcome is ContinuityOutcome.CONTINUOUS
    assert resumed.runner.start() is RunnerState.READY, resumed.runner.halt_reason
    assert continuity_records(state_dir) == []


def test_a_risk_state_restored_from_an_older_copy_survives_an_operator_tail(tmp_path):
    """Blocker 2B. An OPERATOR/flatten record carries no hash either."""
    harness, earlier = _stale_beside(tmp_path)
    state_dir = harness.state_dir
    harness.runner.flatten("operator: reduce to flat")
    assert records(state_dir)[-1]["kind"] == RecordKind.OPERATOR.value

    verdict = _restore_and_read(state_dir, earlier)
    assert verdict.outcome is ContinuityOutcome.STATE_HASH_REGRESSED
    assert verdict.disputed
    assert verdict.hash_witness is not None
    assert verdict.hash_witness.state_hash, "the scan must have walked past the OPERATOR"


def test_an_older_halted_state_does_not_pass_a_newer_halt_for_being_halted_too(tmp_path):
    """Blocker 2C and 2D. Two halted states are not one halted state.

    Both files say `halted`, so the halt witness is satisfied by either and only
    the identity can tell them apart -- and a halted state's own identity is in
    no record, because a HALT carries no hash. What the log DOES record is the
    identity the state had before the halt was written into it, which is where
    the two differ: the older one's is a DECISION the campaign has already moved
    past.
    """
    harness = build(tmp_path, days=(DAY, NEXT_DAY))
    first = harness.first_minute_ms() + BOUNDARY_START * 60_000
    harness.run(MINUTES_BEFORE_THE_BOUNDARY, start=first)
    harness.runner._halt("the first halt")
    state_dir = harness.state_dir
    earlier = tmp_path / "risk.json.first-halt"
    older = backup_of(state_dir, earlier)

    harness.runner.resume("operator: resuming from the first halt")
    harness.run(MINUTES_AFTER_THE_BOUNDARY, start=first + MINUTES_BEFORE_THE_BOUNDARY * 60_000)
    harness.runner._halt("the second halt")

    _, current_state = loaded_state(state_dir)
    (state_dir / "risk.json").write_bytes(older)
    _, older_state = loaded_state(state_dir)
    assert current_state.halted and older_state.halted, (
        "both states must be halted, or `halted == True` is not what is being " "tested"
    )
    assert risk_state_hash(current_state) != risk_state_hash(older_state)
    assert older_state.day_start_equity != current_state.day_start_equity, (
        "the two must differ in a field that governs a limit, not only in the " "halt reason"
    )

    verdict = verdict_for(state_dir)
    assert verdict.outcome is ContinuityOutcome.STATE_HASH_REGRESSED
    assert verdict.disputed
    assert verdict.unhalted_state_hash, "the halt-normalised identity is what found it"
    assert verdict.matched_witness is not None
    assert verdict.hash_witness is not None
    assert verdict.matched_witness.seq < verdict.hash_witness.seq


def test_a_flatten_after_a_halt_does_not_excuse_a_state_that_lost_it(tmp_path):
    """`flatten` moves the risk state and clears no halt, and both matter here.

    The OPERATOR record is the newest halt-relevant record in the log, so a scan
    that treated every OPERATOR command as halt-clearing -- or that added
    `flatten` to the ones that are -- would stop asking whether the loaded state
    still holds the halt. Nothing else would catch it: the flatten moved the
    equity to a state no witness records, so the identity comparison has nothing
    to say and only the halt witness does.
    """
    harness = _halted_campaign(tmp_path)
    config, state_dir = harness.runner.config, harness.state_dir
    harness.runner.flatten("operator: reduce to flat during the incident")
    assert records(state_dir)[-1]["kind"] == RecordKind.OPERATOR.value

    load, state = loaded_state(state_dir)
    assert state.halted, "the fixture must still be halted before it is edited"
    document = state.to_dict(updated_at="2026-09-19T00:00:00+00:00")
    document["halted"] = False
    document["halt_reason"] = ""
    (state_dir / "risk.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    verdict = verdict_for(state_dir)
    assert verdict.outcome is ContinuityOutcome.HALT_NOT_HELD
    assert verdict.disputed
    assert verdict.halt_witness is not None
    assert verdict.halt_witness.kind == RecordKind.HALT.value

    refused = build(tmp_path, days=(DAY,), config=config, start=False)
    assert refused.runner.start() is RunnerState.HALT
    assert is_risk_continuity_halt(refused.runner.halt_reason or "")


def test_this_modules_own_refusal_halt_is_not_read_back_as_campaign_history(tmp_path):
    """The mechanism behind blocker 1A, isolated from the recovery it broke.

    A refusal writes a HALT record like any other halt. Read back as campaign
    evidence it says "this campaign is halted and nothing cleared it", which is
    a statement about R1-c and not about the campaign -- and it is the statement
    that condemned every correct restore.
    """
    harness = campaign(tmp_path)
    harness.runner.shutdown("clean stop")
    config, state_dir = harness.runner.config, harness.state_dir
    saved = backup_of(state_dir, tmp_path / "risk.json.backup")
    (state_dir / "risk.json").unlink()

    refused = build(tmp_path, days=(DAY,), config=config, start=False)
    assert refused.runner.start() is RunnerState.HALT
    halts = [
        record for record in records(state_dir) if record["kind"] == RecordKind.HALT.value
    ]
    assert halts, "the refusal must really have written a HALT record"
    assert is_risk_continuity_halt(halts[-1]["veto_or_rejection"]["detail"])

    (state_dir / "risk.json").write_bytes(saved)
    verdict = verdict_for(state_dir)
    assert (
        verdict.halt_witness is None
    ), "the only HALT in this log is R1-c's own, and it is not campaign history"
    assert verdict.outcome is ContinuityOutcome.CONTINUOUS


# ---------------------------------------------------------------------------
# 14. the review's blocker 3: a startup gate that stops the process before
#     RECOVER must not take the finding with it.
# ---------------------------------------------------------------------------
def test_a_startup_that_halts_before_recover_still_records_the_missing_state(tmp_path):
    """A SELF_CHECK gate that stops the process first must not take the finding.

    A dirty working tree -- section 8.1's source-identity row -- halts the
    process at SELF_CHECK, before RECOVER can refuse on continuity. That used to
    end the process with `build_risk_engine`'s freshly seeded placeholder on disk
    and the gate's own reason halted into it; the next start found a file, and
    the absence it replaced was known only to the log. Now the engine that gate
    halts is one that writes nothing, so the absence is still the absence and
    the next process finds it for itself.
    """
    harness = campaign(tmp_path)
    harness.runner.shutdown("clean stop")
    config, state_dir = harness.runner.config, harness.state_dir
    saved = backup_of(state_dir, tmp_path / "risk.json.backup")
    evidence_before = _evidence_count(state_dir)
    (state_dir / "risk.json").unlink()

    dirty = build(tmp_path, days=(DAY,), config=config, start=False)
    dirty.runner.software["dirty"] = True
    assert dirty.runner.risk_continuity.outcome is ContinuityOutcome.RISK_STATE_MISSING
    assert dirty.runner.start() is RunnerState.HALT
    assert dirty.runner.halt_reason is not None
    assert dirty.runner.halt_reason.startswith(
        "source_identity"
    ), "the fixture must really stop at the EARLIER gate, or it proves nothing"
    assert not (state_dir / "risk.json").exists(), "no placeholder, whichever gate won"

    # The log says what was found, too: one record, written before SELF_CHECK.
    written = continuity_records(state_dir)
    assert len(written) == 1
    block = written[0]["recovery"]["risk_continuity"]
    assert block["outcome"] == ContinuityOutcome.RISK_STATE_MISSING.value
    assert block["risk_state_load"] == RiskStateLoad.MISSING.value
    assert (
        _evidence_count(state_dir) == evidence_before
    ), "a refused startup decides nothing, so it adds no evidence record"

    # And the next process, with a clean tree, is still refused, for the same
    # event and without a second record.
    after = build(tmp_path, days=(DAY,), config=config, start=False)
    assert after.runner.risk_continuity.outcome is ContinuityOutcome.RISK_STATE_MISSING
    assert after.runner.start() is RunnerState.HALT
    assert len(continuity_records(state_dir)) == 1

    # The supported recovery still ends it.
    (state_dir / "risk.json").write_bytes(saved)
    healed = build(tmp_path, days=(DAY,), config=config, start=False)
    assert healed.runner.start() is RunnerState.READY, healed.runner.halt_reason


# ---------------------------------------------------------------------------
# 15. the reporting defect: a log nobody can read is not a first start
# ---------------------------------------------------------------------------
def test_a_log_whose_committed_lines_cannot_be_read_is_not_a_first_start(tmp_path):
    """`read_records` stops at the first line it cannot read, and did so silently.

    A corrupt first line therefore produced no records at all, which read as "no
    history", which read as FIRST_START -- and `demo_run status` told an operator
    that a campaign whose history is unreadable has none. It does not create a
    READY bypass, because the log verification refuses the campaign anyway, but
    the answer was affirmatively wrong.
    """
    harness = campaign(tmp_path)
    harness.runner.shutdown("clean stop")
    state_dir = harness.state_dir
    (state_dir / "risk.json").unlink()

    day_file = sorted((state_dir / "decision_log").glob("*.ndjson"))[0]
    lines = day_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) > 1
    lines[0] = "{ this line is not a record"
    day_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    verdict = verdict_for(state_dir)
    assert verdict.outcome is ContinuityOutcome.HISTORY_UNVERIFIABLE
    assert verdict.outcome is not ContinuityOutcome.FIRST_START
    assert verdict.disputed
    assert not verdict.has_history


def test_a_torn_tail_on_a_genuine_first_start_is_still_a_first_start(tmp_path):
    """The other side, and the one that must not be broken to get the first.

    A line the writer never finished has no trailing newline. That is section
    9.3's TORN_TAIL, `recover_tail` removes it, and a first start interrupted
    part-way through its own STARTUP record leaves exactly one. Refusing it would
    mean a campaign could be bricked by being killed at the worst moment of its
    first minute.
    """
    state_dir = tmp_path / "state"
    (state_dir / "decision_log").mkdir(parents=True)
    (state_dir / "decision_log" / f"{DAY}.ndjson").write_bytes(
        b'{"seq": 1, "kind": "STARTUP", "minute": "2026-09-19T00:00'
    )

    verdict = verdict_for(state_dir)
    assert verdict.outcome is ContinuityOutcome.FIRST_START
    assert not verdict.disputed
    assert not verdict.has_history


# ---------------------------------------------------------------------------
# 16. the identity itself: what is inside it, and what is deliberately not
# ---------------------------------------------------------------------------
#: Every `RiskState` field that governs a permission or an account guard. Listed
#: with a value that differs from the fixture's, so each can be moved one at a
#: time and the hash asked whether it noticed. Widening `RISK_HASH_EXCLUDED` is
#: how a stale state stops being distinguishable from a current one, and a
#: comparison that cannot see a field is a comparison a rolled-back file passes.
_FIELDS_INSIDE_THE_IDENTITY = {
    "equity": 999_000.0,
    "peak_equity": 1_001_000.0,
    "day_start_equity": 998_000.0,
    "daily_pnl": -1_234.0,
    "open_positions": {"BTC/USDT": 1.5},
    "consecutive_losses": 3,
    "halted": True,
    "halt_reason": "a reason",
    "kill_switch": True,
    "stale_feed_since": 1_700_000_000.0,
    "reconciliation_disputed": {"BTC/USDT": "venue says otherwise"},
    "funding_adverse_streak": 4,
    "funding_halt": True,
}

#: And the three that are wall clock or host date rather than decision
#: semantics. Moving one must NOT move the identity, or section 10's replay
#: comparison would fail for a reason that is not a decision.
_FIELDS_OUTSIDE_THE_IDENTITY = {
    "order_times": [1_700_000_000.0],
    "cooldown_until": 1_700_000_060.0,
    "day": "2026-09-20",
}


@pytest.mark.parametrize("field_name,value", sorted(_FIELDS_INSIDE_THE_IDENTITY.items()))
def test_a_field_that_governs_permission_moves_the_identity(field_name, value):
    base = RiskState(equity=1_000_000.0, peak_equity=1_000_000.0)
    moved = replace(base, **{field_name: value})
    assert getattr(moved, field_name) != getattr(base, field_name), "the fixture is vacuous"
    assert risk_state_hash(moved) != risk_state_hash(base)


@pytest.mark.parametrize("field_name,value", sorted(_FIELDS_OUTSIDE_THE_IDENTITY.items()))
def test_a_wall_clock_field_does_not_move_the_identity(field_name, value):
    base = RiskState(equity=1_000_000.0, peak_equity=1_000_000.0)
    moved = replace(base, **{field_name: value})
    assert getattr(moved, field_name) != getattr(base, field_name), "the fixture is vacuous"
    assert risk_state_hash(moved) == risk_state_hash(base)


def test_the_identity_accounts_for_every_field_of_the_persisted_state():
    """No field may be silently neither inside nor outside. A named accounting.

    `RISK_HASH_EXCLUDED` is three names; the rest of `RiskState` is inside. A
    field added later that neither list mentions would be hashed or not by
    accident, and the two parametrised tests above would not notice because they
    only test the names they already know.
    """
    from dataclasses import fields as dataclass_fields

    accounted = set(_FIELDS_INSIDE_THE_IDENTITY) | set(_FIELDS_OUTSIDE_THE_IDENTITY)
    assert {f.name for f in dataclass_fields(RiskState)} == accounted
    assert set(_FIELDS_OUTSIDE_THE_IDENTITY) == set(RISK_HASH_EXCLUDED)


def test_the_continuity_block_carries_both_identities_and_no_verification_claim():
    """What the RECOVERY record must say, and the two things it must not say."""
    witness = risk_continuity.Witness(
        seq=7, kind=RecordKind.DECISION.value, state_hash="sha256:" + "a" * 64
    )
    verdict = RiskContinuity(
        outcome=ContinuityOutcome.STATE_HASH_REGRESSED,
        reason=f"{RISK_CONTINUITY_PREFIX} rolled back",
        load=RiskStateLoad.LOADED,
        observed_state_hash="sha256:" + "b" * 64,
        unhalted_state_hash="sha256:" + "c" * 64,
        hash_witness=witness,
        has_history=True,
    )
    block = verdict.block()
    assert block["observed_state_hash"] == "sha256:" + "b" * 64
    assert block["unhalted_state_hash"] == "sha256:" + "c" * 64
    assert block["witness_state_hash"] == "sha256:" + "a" * 64
    assert block["fingerprint_hash"] == verdict.fingerprint
    assert "records_verified" not in block

    # A hash that does not exist is OMITTED, never written as null: the log's
    # validator holds every `*_hash` field to `sha256:<hex>`.
    bare = RiskContinuity(
        outcome=ContinuityOutcome.RISK_STATE_MISSING,
        reason="r",
        load=RiskStateLoad.MISSING,
        has_history=True,
    ).block()
    assert "observed_state_hash" not in bare
    assert "unhalted_state_hash" not in bare
    assert "witness_state_hash" not in bare


def test_the_fingerprint_distinguishes_materially_different_discontinuities():
    """Two discontinuities that differ in evidence must not share one record."""
    base = RiskContinuity(
        outcome=ContinuityOutcome.STATE_HASH_REGRESSED,
        reason="r",
        load=RiskStateLoad.LOADED,
        observed_state_hash="sha256:" + "b" * 64,
        unhalted_state_hash="sha256:" + "c" * 64,
        hash_witness=risk_continuity.Witness(
            seq=7, kind=RecordKind.DECISION.value, state_hash="sha256:" + "a" * 64
        ),
        has_history=True,
    )
    assert base.fingerprint == replace(base, reason="worded differently").fingerprint
    for changed in (
        replace(base, outcome=ContinuityOutcome.HALT_NOT_HELD),
        replace(base, load=RiskStateLoad.LEGACY),
        replace(base, observed_state_hash="sha256:" + "d" * 64),
        replace(base, unhalted_state_hash="sha256:" + "e" * 64),
        replace(
            base,
            hash_witness=risk_continuity.Witness(
                seq=8, kind=RecordKind.DECISION.value, state_hash="sha256:" + "a" * 64
            ),
        ),
        replace(
            base,
            matched_witness=risk_continuity.Witness(seq=3, kind=RecordKind.DECISION.value),
        ),
    ):
        assert changed.fingerprint != base.fingerprint


def test_the_recovery_record_excludes_no_minute_and_verifies_no_chain(tmp_path):
    """Section 9.3's two fields, and why a refusal answers them differently.

    A crash leaves a minute part-decided and section 9.3 excludes it. A
    continuity refusal decides nothing at all -- it halts before RECOVER
    finishes -- so there is no minute to exclude, and excluding one would drop
    real evidence. `records_verified` is absent rather than zero because this
    check verifies no chain, and a zero would assert that one had been counted.
    """
    harness = _halted_campaign(tmp_path)
    config, state_dir = harness.runner.config, harness.state_dir
    (state_dir / "risk.json").unlink()

    refused = build(tmp_path, days=(DAY,), config=config, start=False)
    refused.runner.start()

    recovery = continuity_records(state_dir)[0]["recovery"]
    assert recovery["evidence_excluded_minute"] is None
    assert recovery["truncated_bytes"] == 0
    assert recovery["minute_finished"] is True
    assert "records_verified" not in recovery
    from tools.replay_parity import EXCLUDING_RECOVERY_CAUSES

    assert RecoveryCause.RISK_STATE_DISCONTINUITY.value not in EXCLUDING_RECOVERY_CAUSES


def test_a_campaign_that_decided_nothing_cannot_pass_its_placeholder_off_as_history(
    tmp_path,
):
    """A campaign with no `risk.state_hash` anywhere, halted before its first minute.

    There is no identity in the log for a later state to be pinned against, so
    what a restore is checked against is the halt the log records -- and what the
    placeholder used to be was a default seed halted on a DIFFERENT gate's reason
    (the dirty-tree refusal), which passed that check and the old one's
    placeholder recognition alike, depending on which halt won. No placeholder
    is written now; and the production writer's default state, put there by
    hand, is refused for not holding the campaign's halt.
    """
    from chimera.demo.risk_wiring import build_risk_engine

    harness = build(tmp_path, days=(DAY,))
    config, state_dir = harness.runner.config, harness.state_dir
    harness.runner._halt("a campaign halt nothing decided around")
    harness.runner.shutdown("clean stop")
    saved = backup_of(state_dir, tmp_path / "risk.json.backup")

    assert not any(
        (record.get("risk") or {}).get("state_hash") for record in records(state_dir)
    ), "the fixture must hold no risk.state_hash at all, or it tests the wrong thing"
    assert records(state_dir), "and it must still hold history"
    _, before = loaded_state(state_dir)
    assert before.halted and before.halt_reason == "a campaign halt nothing decided around"

    (state_dir / "risk.json").unlink()
    dirty = build(tmp_path, days=(DAY,), config=config, start=False)
    dirty.runner.software["dirty"] = True
    assert dirty.runner.start() is RunnerState.HALT
    assert (dirty.runner.halt_reason or "").startswith("source_identity")
    assert not (state_dir / "risk.json").exists(), "no placeholder, whichever gate won"

    after = build(tmp_path, days=(DAY,), config=config, start=False)
    assert after.runner.risk_continuity.outcome is ContinuityOutcome.RISK_STATE_MISSING
    assert after.runner.start() is RunnerState.HALT
    assert is_risk_continuity_halt(after.runner.halt_reason or "")
    assert len(refusal_records(state_dir)) == 1

    # The default state, from the production writer, standing in for the file.
    # Not the campaign's: it holds none of the campaign's halt.
    build_risk_engine(config, capital=CAPITAL, state_dir=state_dir)
    _, default = loaded_state(state_dir)
    assert not default.halted and default.equity == pytest.approx(float(CAPITAL))
    fabricated = build(tmp_path, days=(DAY,), config=config, start=False)
    assert fabricated.runner.risk_continuity.outcome is ContinuityOutcome.HALT_NOT_HELD
    assert fabricated.runner.start() is RunnerState.HALT
    assert len(settlement_records(state_dir)) == 0

    # And the campaign is not bricked: the backup settles it, and the halt it
    # really held comes back with it.
    (state_dir / "risk.json").write_bytes(saved)
    healed = build(tmp_path, days=(DAY,), config=config, start=False)
    assert healed.runner.risk_continuity.outcome is ContinuityOutcome.CONTINUOUS
    assert healed.runner.start() is RunnerState.HALT
    assert healed.runner.halt_reason == "a campaign halt nothing decided around"
    assert len(settlement_records(state_dir)) == 1


def test_a_refusal_that_leaves_no_file_behind_still_writes_only_one_record(tmp_path):
    """The dedup case the log's own answer cannot see, on its natural path.

    An UNREADABLE `risk.json` is preserved rather than replaced —
    `RiskEngine._persist` writes nothing at all while the load was unreadable —
    so a refusal on one leaves no state for the next start to recognise. The
    verdict is therefore re-derived from scratch on every start, and only the
    fingerprint can tell a second detection of one event from a second event.
    """
    harness = campaign(tmp_path)
    harness.runner.shutdown("clean stop")
    config, state_dir = harness.runner.config, harness.state_dir
    damaged = b'{"schema": "chimera.risk-state/1", "equity": '
    (state_dir / "risk.json").write_bytes(damaged)

    for _ in range(3):
        attempt = build(tmp_path, days=(DAY,), config=config, start=False)
        verdict = attempt.runner.risk_continuity
        assert verdict.outcome is ContinuityOutcome.RISK_STATE_UNREADABLE
        assert verdict.load is RiskStateLoad.UNREADABLE
        assert attempt.runner.start() is RunnerState.HALT
        assert is_risk_continuity_halt(attempt.runner.halt_reason or "")
        # The suspect bytes are preserved, which is why there is nothing to
        # recognise on the next start.
        assert (state_dir / "risk.json").read_bytes() == damaged

    assert len(continuity_records(state_dir)) == 1


def test_restoring_the_halt_an_operator_edited_out_settles_the_refusal(tmp_path):
    """An edited-out halt is refused, and restoring the halt settles the refusal.

    An operator who clears `halted` by hand leaves a state whose halt-free
    identity is, by construction, identical to the correct file's -- clearing the
    halt is exactly what setting the halt aside does. An earlier revision
    recognised "the file the refusal left" by that identity, so the two collided
    and the correct restore needed a special case of its own. Nothing collides
    now: the refusal writes nothing, and the restored file settles it by being
    what the log says -- the newest witness's identity once its halt is set
    aside, and halted, as the campaign's own HALT record says it must be.
    """
    harness = _halted_campaign(tmp_path)
    config, state_dir = harness.runner.config, harness.state_dir
    saved = backup_of(state_dir, tmp_path / "risk.json.backup")

    load, correct = loaded_state(state_dir)
    document = correct.to_dict(updated_at="2026-09-19T00:00:00+00:00")
    document["halted"] = False
    document["halt_reason"] = ""
    (state_dir / "risk.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _, edited = loaded_state(state_dir)
    assert halt_free_state_hash(edited) == halt_free_state_hash(correct), (
        "the two must really collide once the halt is set aside, or this test is "
        "about nothing"
    )

    refused = build(tmp_path, days=(DAY,), config=config, start=False)
    assert refused.runner.risk_continuity.outcome is ContinuityOutcome.HALT_NOT_HELD
    assert refused.runner.start() is RunnerState.HALT
    assert len(continuity_records(state_dir)) == 1
    assert (state_dir / "risk.json").read_bytes() != saved, "the edit is still there"

    (state_dir / "risk.json").write_bytes(saved)
    healed = build(tmp_path, days=(DAY,), config=config, start=False)
    assert healed.runner.risk_continuity.outcome is ContinuityOutcome.CONTINUOUS
    assert healed.runner.start() is RunnerState.HALT
    assert healed.runner.halt_reason == correct.halt_reason
    assert len(refusal_records(state_dir)) == 1
    assert len(settlement_records(state_dir)) == 1


# ---------------------------------------------------------------------------
# 17. the second independent review. A disputed file is EVIDENCE: nothing a
#     disputed process does may write it, and a refusal is settled only by a
#     file the log positively vouches for -- never by a file that changed.
# ---------------------------------------------------------------------------
_CLEAN_SOFTWARE = {
    "revision": "synthetic",
    "source_digest": "synthetic",
    "dirty": False,
    "python": "3.11",
}


def _cli(harness, tmp_path: Path, monkeypatch, capsys):
    """`tools/demo_run.main` on this campaign's files: one fresh process per call.

    Every call parses the config and constructs a new `DemoRunner` from what is
    on disk, exactly as an operator's invocation does. `_software` is pinned to a
    clean identity because what is under test is continuity and not the checkout
    this suite happens to run from: a dirty development tree would halt `run` at
    SELF_CHECK before any of this is reached (R1-a's gate, tested on its own in
    `tests/test_demo_cli.py`). Returns ``call(*argv) -> (exit_code, stdout,
    stderr)`` and the module, for its exit codes.
    """
    from tests.demo_harness import CARRY_PARAMS
    from tools import demo_run

    payload = json.loads(
        (Path(__file__).parents[1] / "conf/demo/pvc1.json").read_text("utf-8")
    )
    payload.update(
        {
            "profile": harness.runner.config.profile.value,
            "runner": {"state_dir": str(harness.state_dir)},
            "rules": {"R1_carry": dict(CARRY_PARAMS)},
        }
    )
    config_path = tmp_path / "cli_config.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(demo_run, "_software", lambda root=None: dict(_CLEAN_SOFTWARE))
    argv = ["--config", str(config_path), "--root", str(harness.root), "--profile", "TEST"]

    def call(*command: str) -> tuple[int, str, str]:
        capsys.readouterr()
        code = demo_run.main(argv + list(command))
        out, err = capsys.readouterr()
        return code, out, err

    return call, demo_run


def _status(call) -> dict[str, Any]:
    code, out, _ = call("status")
    assert code == 0
    return json.loads(out)


def _held(state_dir: Path) -> Decimal:
    """What both legs hold, as their stores record it on disk.

    Read from the files and not from `position.state`: a refused start stops
    before `reconstruct()`, so a refused process's in-memory hedge state is
    whatever it was constructed as, and only the stores say what is held.
    """
    total = Decimal("0")
    for name in ("spot_store.json", "perp_store.json"):
        positions = json.loads((state_dir / name).read_text("utf-8")).get("positions") or {}
        total += sum(
            (abs(Decimal(str(held.get("quantity", "0")))) for held in positions.values()),
            Decimal("0"),
        )
    return total


def test_a_cli_flatten_during_a_continuity_dispute_settles_nothing(
    tmp_path, monkeypatch, capsys
):
    """Mandatory test A, and the review's CRITICAL finding, through the real CLI.

    Measured at eedcef74: `flatten` is permitted during a dispute and it moved
    Aegis -- `update_equity`, and the executors' exposures -- so the halt-free
    identity of `risk.json` stopped matching the one the refusal had recorded,
    the dispute was read as settled, `status` said STATE_AHEAD_OF_LOG,
    `resolve --equity` succeeded, and `run` decided three minutes on the
    rolled-back state. Now the refused process writes no risk state at all, so
    the file after the flatten is byte-for-byte the file before it.
    """
    harness, earlier = across_the_boundary(tmp_path)
    state_dir = harness.state_dir
    current = backup_of(state_dir, tmp_path / "risk.json.current")
    _, newest = newest_hash_witness(state_dir)
    assert (
        harness.runner.position.state is not HedgeState.FLAT
    ), "the campaign must hold a position, or a flatten reduces nothing"
    if harness.runner._log is not None:
        harness.runner._log.close()

    shutil.copyfile(earlier, state_dir / "risk.json")
    stale = (state_dir / "risk.json").read_bytes()
    _, stale_state = loaded_state(state_dir)
    assert risk_state_hash(stale_state) != newest, "the restored copy must be older"
    accounted = ledger_equity(state_dir, capital=CAPITAL)
    assert (
        accounted is not None and float(accounted) != stale_state.equity
    ), "the ledger must be genuinely newer, or the R1-b overlap does not exist"
    call, demo_run = _cli(harness, tmp_path, monkeypatch, capsys)
    before_refusal = last_seq(state_dir)

    code, out, _ = call("run")
    assert code == demo_run.EXIT_HALTED
    assert is_risk_continuity_halt(json.loads(out)["reason"])
    assert len(refusal_records(state_dir)) == 1
    code, _, err = call("resolve", "--equity", "--note", "the ledger is right")
    assert code == demo_run.EXIT_REFUSED and "does not continue" in err

    assert _held(state_dir) > 0, "the campaign must hold something to flatten"
    code, out, _ = call("flatten", "--note", "reduce while the risk state is disputed")
    assert code == demo_run.EXIT_OK and json.loads(out)["flattened"] is True
    assert _held(state_dir) == 0, "the flatten reduced exposure, during the dispute"
    assert (
        state_dir / "risk.json"
    ).read_bytes() == stale, "a disputed process writes no risk state, whatever it moved"

    status = _status(call)["risk_continuity"]
    assert status["disputed"] is True
    assert status["outcome"] == ContinuityOutcome.STATE_HASH_REGRESSED.value
    code, _, err = call("resolve", "--equity", "--note", "the ledger is right")
    assert code == demo_run.EXIT_REFUSED and "does not continue" in err
    code, _, _ = call("resume", "--note", "operator: I looked at it")
    assert code == demo_run.EXIT_REFUSED
    code, out, _ = call("run")
    assert code == demo_run.EXIT_HALTED
    assert is_risk_continuity_halt(json.loads(out)["reason"])
    assert decisions_after(state_dir, before_refusal) == [], "no minute was decided"
    assert len(refusal_records(state_dir)) == 1, "one event, however many processes"

    # The supported way out: the copy the campaign last wrote.
    (state_dir / "risk.json").write_bytes(current)
    status = _status(call)["risk_continuity"]
    assert status["disputed"] is False
    assert status["outcome"] == ContinuityOutcome.CONTINUOUS.value
    code, out, _ = call("run")
    # The flatten moved the ledger and not Aegis's persisted equity, so R1-b now
    # sees a real disagreement -- persisted this time, by a settled process.
    assert code == demo_run.EXIT_HALTED
    _, halted = loaded_state(state_dir)
    assert halted.halt_reason.startswith(EQUITY_DISPUTE_PREFIX)
    assert len(settlement_records(state_dir)) == 1
    code, out, _ = call("resolve", "--equity", "--note", "the ledger is the accounting")
    assert code == demo_run.EXIT_OK and json.loads(out)["state"] == "READY"
    code, _, _ = call("run")
    assert code == demo_run.EXIT_OK
    assert decisions_after(state_dir, before_refusal), "the campaign decides again"


def test_a_kill_switch_during_a_continuity_dispute_halts_and_settles_nothing(tmp_path):
    """Mandatory test B. The switch keeps its safety effect and gains no other.

    Measured at eedcef74: the engine's constructor persisted the switch's mirror
    into the disputed file, removing the switch persisted it back, and the moved
    identity read as a restore -- STATE_AHEAD_OF_LOG, not disputed. Now the
    mirror and the halt are held in memory, where they still stop everything.
    """
    from chimera.demo.runner import RunnerError

    harness, earlier = across_the_boundary(tmp_path)
    config, state_dir = harness.runner.config, harness.state_dir
    current = backup_of(state_dir, tmp_path / "risk.json.current")
    shutil.copyfile(earlier, state_dir / "risk.json")
    stale = (state_dir / "risk.json").read_bytes()
    days = (DAY, NEXT_DAY)

    first = build(tmp_path, days=days, config=config, start=False)
    assert first.runner.start() is RunnerState.HALT
    fingerprint = first.runner.risk_continuity.fingerprint

    (state_dir / "KILL_SWITCH").write_text("engaged", encoding="utf-8")
    switched = build(tmp_path, days=days, config=config, start=False)
    assert switched.runner.start() is RunnerState.HALT
    # The safety effect, in the engine the runner enforces with.
    assert switched.runner.risk.state.kill_switch is True
    assert switched.runner.risk.halted
    with pytest.raises(RunnerError):
        switched.runner.resume("operator: the switch is on and I want to trade")
    assert (state_dir / "risk.json").read_bytes() == stale

    (state_dir / "KILL_SWITCH").unlink()
    cleared = build(tmp_path, days=days, config=config, start=False)
    verdict = cleared.runner.risk_continuity
    assert verdict.disputed
    assert verdict.outcome is ContinuityOutcome.STATE_HASH_REGRESSED
    assert verdict.fingerprint == fingerprint
    assert cleared.runner.start() is RunnerState.HALT
    assert (state_dir / "risk.json").read_bytes() == stale
    assert len(refusal_records(state_dir)) == 1

    # And after a real restore the switch halts and persists exactly as before.
    (state_dir / "KILL_SWITCH").write_text("engaged", encoding="utf-8")
    (state_dir / "risk.json").write_bytes(current)
    restored = build(tmp_path, days=days, config=config, start=False)
    assert restored.runner.risk_continuity.outcome is ContinuityOutcome.CONTINUOUS
    assert restored.runner.start() is RunnerState.HALT
    assert restored.runner.halt_reason == "kill_switch"
    _, persisted = loaded_state(state_dir)
    assert persisted.halted and persisted.kill_switch


def _a_stale_copy_carrying_a_reconciliation_dispute(tmp_path: Path):
    """A campaign whose OLDER `risk.json` held a reconciliation dispute it later cleared.

    The dispute is put into the store and into Aegis exactly as `_reconcile`
    does on a mismatch, and the next DECISION witnesses it. The copy is taken
    there; the campaign then resolves it and decides past it.
    """
    harness = build(tmp_path, days=(DAY,))
    first = harness.first_minute_ms()
    harness.run(5, start=first)
    symbol = harness.runner.position.config.perp_symbol
    harness.runner.position.perp.store.state.disputed[symbol] = "reconciliation_mismatch"
    harness.risk.note_reconciliation(symbol, "reconciliation_mismatch")
    harness.tick(first + 5 * 60_000)
    earlier = tmp_path / "risk.json.with-the-dispute"
    shutil.copyfile(harness.state_dir / "risk.json", earlier)
    harness.runner.resolve(symbol, "cleared in the campaign itself")
    harness.run(3, start=first + 6 * 60_000)
    harness.runner.shutdown("clean stop")
    return harness, earlier, symbol


def test_resolve_symbol_refuses_during_a_continuity_dispute_and_moves_nothing(
    tmp_path, monkeypatch, capsys
):
    """Mandatory test C. The reconciliation path may not move a disputed state.

    `resolve --symbol` clears Aegis's copy of a dispute through
    `note_reconciliation`, which persists. It is refused before anything is
    touched -- Aegis would clear only in memory here, but the store's copy would
    be cleared and SAVED, and restoring the right `risk.json` afterwards would
    bring back a dispute no store still records.
    """
    from chimera.demo.runner import RunnerError

    harness, earlier, symbol = _a_stale_copy_carrying_a_reconciliation_dispute(tmp_path)
    config, state_dir = harness.runner.config, harness.state_dir
    shutil.copyfile(earlier, state_dir / "risk.json")
    stale = (state_dir / "risk.json").read_bytes()

    refused = build(tmp_path, days=(DAY,), config=config, start=False)
    assert refused.runner.risk_continuity.outcome is ContinuityOutcome.STATE_HASH_REGRESSED
    assert refused.runner.start() is RunnerState.HALT
    assert refused.runner.risk.state.reconciliation_disputed.get(
        symbol
    ), "the stale state must really carry the dispute the command would clear"
    refused.runner.position.perp.store.state.disputed[symbol] = "reconciliation_mismatch"
    store_before = (state_dir / "perp_store.json").read_bytes()
    count_before = len(records(state_dir))

    with pytest.raises(RunnerError, match="does not continue the decision log"):
        refused.runner.resolve(symbol, "operator: checked the venue")

    assert refused.runner.risk.state.reconciliation_disputed.get(symbol)
    assert refused.runner.position.perp.store.state.disputed.get(symbol)
    assert (state_dir / "perp_store.json").read_bytes() == store_before
    assert (state_dir / "risk.json").read_bytes() == stale
    assert len(records(state_dir)) == count_before, "no OPERATOR record either"
    if refused.runner._log is not None:
        refused.runner._log.close()

    after = build(tmp_path, days=(DAY,), config=config, start=False)
    assert after.runner.risk_continuity.disputed

    # And from the CLI, a fresh process that never starts: refused on the same
    # ground, before the clock or a store is consulted.
    call, demo_run = _cli(harness, tmp_path, monkeypatch, capsys)
    code, _, err = call("resolve", "--symbol", symbol, "--note", "operator: checked")
    assert code == demo_run.EXIT_REFUSED
    assert "does not continue the decision log" in err
    assert (state_dir / "risk.json").read_bytes() == stale


def test_a_forged_log_beside_a_missing_risk_state_fabricates_nothing(tmp_path):
    """Mandatory test D, and the forged-log double fault the review measured.

    Measured at eedcef74: the forged log was refused before STARTUP, but by then
    `build_risk_engine` had seeded a placeholder and the log refusal's halt was
    persisted into it -- so when the log was later restored from a verified copy,
    the placeholder read as CONTINUOUS, `resume` reached READY, and the campaign's
    own historical halt was simply gone. Now nothing is written: the forged log
    still refuses, and when it is repaired the absence is still there to refuse.
    """
    from chimera.demo.runner import RunnerError

    harness = campaign(tmp_path)
    harness.runner._halt("a real historical risk halt")
    harness.runner.shutdown("clean stop")
    config, state_dir = harness.runner.config, harness.state_dir
    if harness.runner._log is not None:
        harness.runner._log.close()
    saved = backup_of(state_dir, tmp_path / "risk.json.backup")
    log_dir = state_dir / "decision_log"
    verified = tmp_path / "decision_log.verified"
    shutil.copytree(log_dir, verified)
    before_loss = last_seq(state_dir)
    (state_dir / "risk.json").unlink()

    day_file = sorted(log_dir.glob("*.ndjson"))[0]
    lines = day_file.read_bytes().split(b"\n")
    forged = json.loads(lines[2])
    forged["runner_now_ns"] = int(forged["runner_now_ns"]) + 1
    lines[2] = json.dumps(forged, sort_keys=True, separators=(",", ":")).encode()
    day_file.write_bytes(b"\n".join(lines))
    forged_bytes = day_file.read_bytes()

    first = build(tmp_path, days=(DAY,), config=config, start=False)
    assert first.runner.risk_continuity.outcome is ContinuityOutcome.RISK_STATE_MISSING
    assert first.runner.start() is RunnerState.HALT
    assert (first.runner.halt_reason or "").startswith("log_forged")
    assert not (state_dir / "risk.json").exists(), "no placeholder behind a forged log"
    assert day_file.read_bytes() == forged_bytes, "and nothing appended to it"
    if first.runner._log is not None:
        first.runner._log.close()

    shutil.rmtree(log_dir)
    shutil.copytree(verified, log_dir)
    repaired = build(tmp_path, days=(DAY,), config=config, start=False)
    assert repaired.runner.risk_continuity.outcome is ContinuityOutcome.RISK_STATE_MISSING
    assert repaired.runner.start() is RunnerState.HALT
    assert is_risk_continuity_halt(repaired.runner.halt_reason or "")
    with pytest.raises(RunnerError):
        repaired.runner.resume("operator: the log is repaired")
    with pytest.raises(RunnerError):
        repaired.tick(repaired.runner.cursor.next_minute_ms())
    assert decisions_after(state_dir, before_loss) == []
    assert not (state_dir / "risk.json").exists()

    (state_dir / "risk.json").write_bytes(saved)
    restored = build(tmp_path, days=(DAY,), config=config, start=False)
    assert restored.runner.risk_continuity.outcome is ContinuityOutcome.CONTINUOUS
    assert restored.runner.start() is RunnerState.HALT
    assert (
        restored.runner.halt_reason == "a real historical risk halt"
    ), "the campaign's own halt is what comes back, not a placeholder's"
    restored.runner.resume("operator: the historical halt's cause is gone")
    restored.tick(restored.runner.cursor.next_minute_ms())
    assert decisions_after(state_dir, before_loss)


def test_an_unverifiable_log_beside_a_missing_risk_state_fabricates_nothing(tmp_path):
    """The same root on the HISTORY_UNVERIFIABLE path: the absence is kept."""
    harness = campaign(tmp_path)
    harness.runner.shutdown("clean stop")
    config, state_dir = harness.runner.config, harness.state_dir
    if harness.runner._log is not None:
        harness.runner._log.close()
    (state_dir / "risk.json").unlink()
    day_file = sorted((state_dir / "decision_log").glob("*.ndjson"))[0]
    lines = day_file.read_bytes().split(b"\n")
    lines[0] = b"{ this line is not a record"
    day_file.write_bytes(b"\n".join(lines))

    refused = build(tmp_path, days=(DAY,), config=config, start=False)
    assert refused.runner.risk_continuity.outcome is ContinuityOutcome.HISTORY_UNVERIFIABLE
    assert refused.runner.start() is RunnerState.HALT
    assert not (state_dir / "risk.json").exists()


class _Crash(Exception):
    """Stands in for a SIGKILL, a full disk or a refused append at one statement."""


def _tear_last_line(state_dir: Path, runner) -> None:
    """Leave the record just appended half-written, as a crash mid-write would."""
    if runner._log is not None:
        runner._log.close()
    day_file = sorted((state_dir / "decision_log").glob("*.ndjson"))[-1]
    payload = day_file.read_bytes()
    assert payload.endswith(b"\n")
    last = payload[:-1].rsplit(b"\n", 1)[-1]
    day_file.write_bytes(payload[: len(payload) - 1 - len(last) // 2])


def _arm(failpoint: str, runner, state_dir: Path, monkeypatch) -> None:
    """Make ``runner.start()`` die at exactly one point of the startup sequence."""
    import chimera.demo.risk_wiring as risk_wiring

    real_append = runner._append

    def die(*args, **kwargs):
        raise _Crash(failpoint)

    if failpoint == "before_construction":
        monkeypatch.setattr(runner, "_risk_factory", die)
    elif failpoint == "during_construction":
        real_seed = risk_wiring.seed_or_reconcile_equity

        def seeded_then_die(engine, **kwargs):
            real_seed(engine, **kwargs)
            raise _Crash(failpoint)

        monkeypatch.setattr(risk_wiring, "seed_or_reconcile_equity", seeded_then_die)
    elif failpoint == "before_triage":
        monkeypatch.setattr(runner, "_triage_log", die)
    elif failpoint == "during_triage":
        monkeypatch.setattr(runner, "_state_ahead_of_log", die)
    elif failpoint in ("before_startup_append", "during_startup_append"):
        torn = failpoint == "during_startup_append"

        def startup(kind, *args, **kwargs):
            if kind is RecordKind.STARTUP:
                if torn:
                    real_append(kind, *args, **kwargs)
                    _tear_last_line(state_dir, runner)
                raise _Crash(failpoint)
            return real_append(kind, *args, **kwargs)

        monkeypatch.setattr(runner, "_append", startup)
    elif failpoint == "before_continuity_record":
        monkeypatch.setattr(runner, "_record_risk_continuity", die)
    elif failpoint == "during_continuity_record":

        def recovery(kind, *args, **kwargs):
            if kind is RecordKind.RECOVERY:
                real_append(kind, *args, **kwargs)
                _tear_last_line(state_dir, runner)
                raise _Crash(failpoint)
            return real_append(kind, *args, **kwargs)

        monkeypatch.setattr(runner, "_append", recovery)
    elif failpoint == "before_refusal_halt":
        monkeypatch.setattr(runner, "_halt", die)
    else:  # pragma: no cover - a typo in the parameter list
        raise AssertionError(failpoint)


#: The prompt's nine points, in the order `start()` reaches them.
_FAILPOINTS = (
    "before_construction",
    "during_construction",
    "before_triage",
    "during_triage",
    "before_startup_append",
    "during_startup_append",
    "before_continuity_record",
    "during_continuity_record",
    "before_refusal_halt",
)


@pytest.mark.parametrize("damage", ["missing", "stale"])
@pytest.mark.parametrize("failpoint", _FAILPOINTS)
def test_a_crash_anywhere_in_a_refused_startup_leaves_the_evidence_intact(
    tmp_path, monkeypatch, failpoint, damage
):
    """Mandatory test E: the startup crash matrix.

    Measured at eedcef74: a missing `risk.json` was replaced by a seeded
    placeholder inside `build_risk_engine`, and a process that died in triage,
    in the STARTUP append or in the continuity RECOVERY append left that
    placeholder and no record. The next start read it as STATE_AHEAD_OF_LOG,
    wrote a false LOG_BEHIND_STATE recovery, and reached READY and a DECISION
    after `resume`.

    At every point below the file is exactly as it was found -- absent, or the
    same stale bytes -- so the next process reaches the same verdict from the
    evidence itself, whether or not the log managed to say so first.
    """
    from chimera.demo.runner import RunnerError

    if damage == "missing":
        harness = campaign(tmp_path)
        harness.runner.shutdown("clean stop")
        config, state_dir = harness.runner.config, harness.state_dir
        saved = backup_of(state_dir, tmp_path / "risk.json.backup")
        (state_dir / "risk.json").unlink()
        days: tuple[str, ...] = (DAY,)
        expected = ContinuityOutcome.RISK_STATE_MISSING
    else:
        harness, earlier = across_the_boundary(tmp_path)
        config, state_dir = harness.runner.config, harness.state_dir
        saved = backup_of(state_dir, tmp_path / "risk.json.backup")
        shutil.copyfile(earlier, state_dir / "risk.json")
        days = (DAY, NEXT_DAY)
        expected = ContinuityOutcome.STATE_HASH_REGRESSED
    if harness.runner._log is not None:
        harness.runner._log.close()
    found = (state_dir / "risk.json").read_bytes() if damage == "stale" else None
    before_loss = last_seq(state_dir)

    def intact() -> bool:
        if damage == "missing":
            return not (state_dir / "risk.json").exists()
        return (state_dir / "risk.json").read_bytes() == found

    victim = build(tmp_path, days=days, config=config, start=False)
    assert victim.runner.risk_continuity.outcome is expected
    _arm(failpoint, victim.runner, state_dir, monkeypatch)
    with pytest.raises(_Crash):
        victim.runner.start()
    monkeypatch.undo()
    if victim.runner._log is not None:
        victim.runner._log.close()
    assert intact(), f"{failpoint}: the evidence was overwritten"
    assert not (state_dir / "risk.json.tmp").exists()

    after = build(tmp_path, days=days, config=config, start=False)
    assert after.runner.risk_continuity.outcome is expected
    assert after.runner.start() is RunnerState.HALT
    assert is_risk_continuity_halt(after.runner.halt_reason or "")
    with pytest.raises(RunnerError):
        after.runner.resume("operator: pressing on")
    assert intact()
    assert decisions_after(state_dir, before_loss) == []
    assert len(refusal_records(state_dir)) == 1
    # No false crash recovery for the risk state: the legs and the ledger agree
    # with the log, and the risk state is refused rather than recovered from.
    assert not [
        record
        for record in records(state_dir)
        if int(record["seq"]) > before_loss
        and (record.get("recovery") or {}).get("cause") == RecoveryCause.LOG_BEHIND_STATE.value
    ]
    if after.runner._log is not None:
        after.runner._log.close()

    (state_dir / "risk.json").write_bytes(saved)
    healed = build(tmp_path, days=days, config=config, start=False)
    assert healed.runner.start() is RunnerState.READY, healed.runner.halt_reason


def test_a_campaign_that_never_decided_recovers_from_its_real_state(tmp_path):
    """Mandatory test F, the unhalted side, and the review's never-decided edge.

    Measured at eedcef74: a campaign with records and no `risk.state_hash`, never
    halted, lost its file, and restoring the REAL file left it ALREADY_DISPUTED
    for ever -- the real state and the placeholder the refusal had seeded were
    one identity. Nothing is seeded now, so there is nothing to collide with.

    The limit, stated as an assertion rather than hidden: the genuine state of
    a campaign that never decided anything IS the default first-start seed, so
    the production writer's default state has its identity. No identity check
    can refuse one and accept the other; what protects the campaign is that no
    process writes one (the halted side, where they do differ, is
    `test_a_campaign_that_decided_nothing_cannot_pass_its_placeholder_off_as_history`).
    """
    from chimera.demo.risk_wiring import build_risk_engine

    harness = build(tmp_path, days=(DAY,))
    harness.runner.shutdown("clean stop")
    config, state_dir = harness.runner.config, harness.state_dir
    saved = backup_of(state_dir, tmp_path / "risk.json.backup")
    assert records(state_dir), "the campaign has history"
    assert not any(
        (record.get("risk") or {}).get("state_hash") for record in records(state_dir)
    ), "and has never recorded a risk identity"
    _, genuine = loaded_state(state_dir)
    assert not genuine.halted

    (state_dir / "risk.json").unlink()
    for _ in range(2):
        refused = build(tmp_path, days=(DAY,), config=config, start=False)
        assert refused.runner.risk_continuity.outcome is ContinuityOutcome.RISK_STATE_MISSING
        assert refused.runner.start() is RunnerState.HALT
        assert not (state_dir / "risk.json").exists(), "the system fabricates nothing"
    assert len(refusal_records(state_dir)) == 1

    (state_dir / "risk.json").write_bytes(saved)
    restored = build(tmp_path, days=(DAY,), config=config, start=False)
    assert restored.runner.risk_continuity.outcome is ContinuityOutcome.CONTINUOUS
    assert restored.runner.start() is RunnerState.READY, restored.runner.halt_reason
    restored.tick(restored.runner.cursor.next_minute_ms())
    assert restored.runner.state is not RunnerState.HALT
    assert len(settlement_records(state_dir)) == 1

    default = build_risk_engine(config, capital=CAPITAL, state_dir=tmp_path / "elsewhere")
    assert risk_state_hash(default.state) == risk_state_hash(genuine)


def test_after_a_refusal_only_the_state_the_log_vouches_for_settles_it(tmp_path):
    """Mandatory test G: the stale copy is refused and the current one accepted.

    Both are restored AFTER a refusal, into a stretch the log holds open -- the
    case "a different file means a restore" got wrong: the stale copy differs
    from what the refusal found, and is still older than the log.
    """
    harness, earlier = across_the_boundary(tmp_path)
    config, state_dir = harness.runner.config, harness.state_dir
    current = backup_of(state_dir, tmp_path / "risk.json.current")
    stale = earlier.read_bytes()
    _, newest = newest_hash_witness(state_dir)
    days = (DAY, NEXT_DAY)
    (state_dir / "risk.json").unlink()

    refused = build(tmp_path, days=days, config=config, start=False)
    assert refused.runner.start() is RunnerState.HALT

    (state_dir / "risk.json").write_bytes(stale)
    _, stale_state = loaded_state(state_dir)
    stale_attempt = build(tmp_path, days=days, config=config, start=False)
    verdict = stale_attempt.runner.risk_continuity
    assert verdict.outcome is ContinuityOutcome.STATE_HASH_REGRESSED
    assert verdict.settles_seq is None
    assert stale_attempt.runner.start() is RunnerState.HALT
    assert (state_dir / "risk.json").read_bytes() == stale

    (state_dir / "risk.json").write_bytes(current)
    _, current_state = loaded_state(state_dir)
    assert risk_state_hash(stale_state) != risk_state_hash(current_state), "non-vacuous"
    assert risk_state_hash(current_state) == newest
    current_attempt = build(tmp_path, days=days, config=config, start=False)
    assert current_attempt.runner.risk_continuity.outcome is ContinuityOutcome.CONTINUOUS
    assert current_attempt.runner.start() is RunnerState.READY
    assert len(refusal_records(state_dir)) == 2, "missing, then the stale attempt"
    assert len(settlement_records(state_dir)) == 1


def test_a_natural_r1b_overlap_survives_a_flatten_and_restarts(tmp_path):
    """Mandatory test H: the review's Blocker 1B, end to end with a flatten in it.

    An older `risk.json` beside a ledger the campaign has since marked: R1-b's
    equity dispute owns Aegis's halt reason, R1-c owns the runner's. Across
    restarts and a flatten nothing settles the continuity dispute and nothing
    resolves the equity one; the actual file settles the first, and R1-b's own
    path then settles the second.
    """
    from chimera.demo.runner import RunnerError

    harness = build(tmp_path, days=(DAY,))
    first = harness.first_minute_ms()
    harness.run(MINUTE_BEFORE_A_SETTLEMENT, start=first)
    earlier = tmp_path / "risk.json.before-the-settlement"
    shutil.copyfile(harness.state_dir / "risk.json", earlier)
    harness.run(2, start=first + MINUTE_BEFORE_A_SETTLEMENT * 60_000)
    harness.runner.shutdown("clean stop after the settlement")
    config, state_dir = harness.runner.config, harness.state_dir
    current = backup_of(state_dir, tmp_path / "risk.json.current")
    before_loss = last_seq(state_dir)

    shutil.copyfile(earlier, state_dir / "risk.json")
    stale = (state_dir / "risk.json").read_bytes()
    _, stale_state = loaded_state(state_dir)
    accounted = ledger_equity(state_dir, capital=CAPITAL)
    assert accounted is not None and float(accounted) != stale_state.equity

    def refused_process():
        process = build(tmp_path, days=(DAY,), config=config, start=False)
        assert process.runner.risk_continuity.outcome is (
            ContinuityOutcome.STATE_HASH_REGRESSED
        )
        assert process.runner.start() is RunnerState.HALT
        assert process.runner.risk.state.halt_reason.startswith(EQUITY_DISPUTE_PREFIX)
        with pytest.raises(RunnerError, match="does not continue the decision log"):
            process.runner.resolve_equity("the ledger is right")
        with pytest.raises(RunnerError, match="does not continue the decision log"):
            process.runner.resume("operator: pressing on")
        with pytest.raises(RunnerError, match="does not continue the decision log"):
            process.tick(process.runner.cursor.next_minute_ms())
        assert (state_dir / "risk.json").read_bytes() == stale
        return process

    refused_process()
    flattening = refused_process()
    assert _held(state_dir) > 0, "the campaign must hold something to flatten"
    flattening.runner.flatten("operator: reduce while both disputes stand")
    assert _held(state_dir) == 0, "the flatten reduced exposure, during the dispute"
    assert (state_dir / "risk.json").read_bytes() == stale
    refused_process()
    assert decisions_after(state_dir, before_loss) == []
    assert len(refusal_records(state_dir)) == 1

    (state_dir / "risk.json").write_bytes(current)
    settled = build(tmp_path, days=(DAY,), config=config, start=False)
    assert settled.runner.risk_continuity.outcome is ContinuityOutcome.CONTINUOUS
    assert settled.runner.start() is RunnerState.HALT
    assert settled.runner.risk.state.halt_reason.startswith(
        EQUITY_DISPUTE_PREFIX
    ), "the flatten moved the ledger, so a real R1-b dispute is what remains"
    settled.runner.resolve_equity("the ledger is the campaign's accounting")
    assert settled.runner.state is RunnerState.READY
    settled.tick(settled.runner.cursor.next_minute_ms())
    assert settled.runner.state is not RunnerState.HALT, settled.runner.halt_reason
    assert decisions_after(state_dir, before_loss)


def _crash_ahead_campaign(tmp_path: Path):
    """The crash window, and the copy the log's newest witness describes.

    Returns ``(harness, witnessed, ahead)``: the bytes the campaign held at its
    newest DECISION, and the bytes a tick persisted before it died without
    committing the DECISION that would have quoted them.
    """
    harness = build(tmp_path, days=(DAY, NEXT_DAY))
    first = harness.first_minute_ms() + BOUNDARY_START * 60_000
    harness.run(MINUTES_BEFORE_THE_BOUNDARY, start=first)
    witnessed = (harness.state_dir / "risk.json").read_bytes()
    real_append = harness.runner._append

    def killed(kind, *args, **kwargs):
        if kind is RecordKind.DECISION:
            raise _Crash("between the persist and the DECISION")
        return real_append(kind, *args, **kwargs)

    harness.runner._append = killed
    with pytest.raises(_Crash):
        harness.tick(first + MINUTES_BEFORE_THE_BOUNDARY * 60_000)
    if harness.runner._log is not None:
        harness.runner._log.close()
    ahead = (harness.state_dir / "risk.json").read_bytes()
    return harness, witnessed, ahead


def test_an_identity_no_record_quotes_does_not_settle_a_refusal(tmp_path):
    """Outside a refusal it is the crash window; inside one it proves nothing.

    Both sides use the same bytes: the state a tick persisted before it died.
    With no refusal open that state recovers as section 9.3's crash. After a
    refusal no process of this campaign could have written a new file -- a
    disputed process writes none -- so a file no record quotes is NOT_SETTLED,
    and the copy the newest witness describes is what settles it.
    """
    harness, witnessed, ahead = _crash_ahead_campaign(tmp_path)
    config, state_dir = harness.runner.config, harness.state_dir
    days = (DAY, NEXT_DAY)
    _, newest = newest_hash_witness(state_dir)
    assert witnessed != ahead

    assert verdict_for(state_dir).outcome is ContinuityOutcome.STATE_AHEAD_OF_LOG

    (state_dir / "risk.json").unlink()
    refused = build(tmp_path, days=days, config=config, start=False)
    assert refused.runner.start() is RunnerState.HALT
    refusal_seq = int(refusal_records(state_dir)[0]["seq"])
    # A flatten inside the refusal writes an OPERATOR record -- a kind that CAN
    # move the risk state without a hash. It is the refused process's, so it
    # admits nothing; if it were read as campaign history it would excuse the
    # unquoted file below exactly as a pre-refusal flatten does.
    assert _held(state_dir) > 0, "the campaign must hold something to flatten"
    refused.runner.flatten("operator: reduce while the risk state is disputed")
    assert _held(state_dir) == 0
    assert not (state_dir / "risk.json").exists()

    (state_dir / "risk.json").write_bytes(ahead)
    unsettled = build(tmp_path, days=days, config=config, start=False)
    verdict = unsettled.runner.risk_continuity
    assert verdict.outcome is ContinuityOutcome.NOT_SETTLED
    assert verdict.disputed and not verdict.ahead_of_log
    assert str(refusal_seq) in verdict.reason and newest in verdict.reason
    assert unsettled.runner.start() is RunnerState.HALT
    assert (state_dir / "risk.json").read_bytes() == ahead

    (state_dir / "risk.json").write_bytes(witnessed)
    _, witnessed_state = loaded_state(state_dir)
    assert risk_state_hash(witnessed_state) == newest
    settled = build(tmp_path, days=days, config=config, start=False)
    assert settled.runner.risk_continuity.outcome is ContinuityOutcome.CONTINUOUS
    assert settled.runner.risk_continuity.settles_seq == refusal_seq
    if settled.runner.start() is RunnerState.HALT:
        # The legs and the ledger are still ahead of the log, so R1-b may see
        # the witnessed equity disagree with the ledger; its own path clears it.
        assert settled.runner.risk.state.halt_reason.startswith(EQUITY_DISPUTE_PREFIX)
        settled.runner.resolve_equity("the ledger is the campaign's accounting")
    assert settled.runner.state is RunnerState.READY, settled.runner.halt_reason


def test_a_hashless_move_before_the_refusal_still_admits_its_own_state(tmp_path):
    """The other side of NOT_SETTLED: the campaign's OWN history can move the state.

    A `flatten` before the loss writes an OPERATOR record with no hash, so the
    correct backup holds an identity no record quotes. That record is the
    campaign's, written before any refusal, and it is what admits the backup --
    exactly as it does with no refusal at all. A flatten run DURING the dispute
    admits nothing, because its record is the refused process's and is skipped.
    """
    harness = campaign(tmp_path, minutes=12)
    harness.runner.flatten("operator: reduce to flat before maintenance")
    harness.runner.shutdown("clean stop")
    config, state_dir = harness.runner.config, harness.state_dir
    saved = backup_of(state_dir, tmp_path / "risk.json.backup")
    _, newest = newest_hash_witness(state_dir)
    assert risk_state_hash(loaded_state(state_dir)[1]) != newest, "the flatten moved it"
    (state_dir / "risk.json").unlink()

    refused = build(tmp_path, days=(DAY,), config=config, start=False)
    assert refused.runner.start() is RunnerState.HALT

    (state_dir / "risk.json").write_bytes(saved)
    restored = build(tmp_path, days=(DAY,), config=config, start=False)
    assert restored.runner.risk_continuity.outcome is ContinuityOutcome.CONTINUOUS
    assert restored.runner.start() is not RunnerState.HALT, restored.runner.halt_reason


def test_what_a_settled_process_halts_on_is_campaign_history_again(tmp_path):
    """The settlement record ends the skipped stretch, and not a DECISION.

    The settled process can halt before it decides anything -- here on the kill
    switch, and the halt is persisted, because this process may write. Rolling
    `risk.json` back to the restored, unhalted copy must then be HALT_NOT_HELD,
    as it would be on any campaign. At eedcef74 the stretch ran until the next
    hash-bearing record, so that HALT was skipped and the rollback passed.
    """
    harness = campaign(tmp_path)
    harness.runner.shutdown("clean stop")
    config, state_dir = harness.runner.config, harness.state_dir
    saved = backup_of(state_dir, tmp_path / "risk.json.backup")
    (state_dir / "risk.json").unlink()
    refused = build(tmp_path, days=(DAY,), config=config, start=False)
    assert refused.runner.start() is RunnerState.HALT

    (state_dir / "KILL_SWITCH").write_text("engaged", encoding="utf-8")
    (state_dir / "risk.json").write_bytes(saved)
    settled = build(tmp_path, days=(DAY,), config=config, start=False)
    assert settled.runner.start() is RunnerState.HALT
    assert settled.runner.halt_reason == "kill_switch"
    assert loaded_state(state_dir)[1].halted, "a settled process persists its halt"
    assert len(settlement_records(state_dir)) == 1

    (state_dir / "KILL_SWITCH").unlink()
    (state_dir / "risk.json").write_bytes(saved)
    rolled_back = build(tmp_path, days=(DAY,), config=config, start=False)
    verdict = rolled_back.runner.risk_continuity
    assert verdict.outcome is ContinuityOutcome.HALT_NOT_HELD
    assert verdict.halt_witness is not None
    assert verdict.halt_witness.kind == RecordKind.HALT.value


def test_a_second_loss_after_a_restore_that_decided_nothing_is_its_own_event(tmp_path):
    """The review's RECOVERY dedup edge: two losses, one READY between, no minute.

    At eedcef74 the two refusals had the same fingerprint and nothing between
    them that the dedup could see, so the second loss was never written down.
    The settlement record between them is what the dedup sees now: one event is
    the stretch from a refusal to its settlement.
    """
    harness = campaign(tmp_path)
    harness.runner.shutdown("clean stop")
    config, state_dir = harness.runner.config, harness.state_dir
    saved = backup_of(state_dir, tmp_path / "risk.json.backup")

    (state_dir / "risk.json").unlink()
    first = build(tmp_path, days=(DAY,), config=config, start=False)
    assert first.runner.start() is RunnerState.HALT
    (state_dir / "risk.json").write_bytes(saved)
    restored = build(tmp_path, days=(DAY,), config=config, start=False)
    assert restored.runner.start() is RunnerState.READY
    restored.runner.shutdown("stopped again before deciding anything")
    decided_before = _evidence_count(state_dir)

    (state_dir / "risk.json").unlink()
    for _ in range(2):
        second = build(tmp_path, days=(DAY,), config=config, start=False)
        assert second.runner.start() is RunnerState.HALT
    assert _evidence_count(state_dir) == decided_before, "no minute was decided"
    refusals = refusal_records(state_dir)
    assert len(refusals) == 2, "two losses, two records, and no more for restarts"
    settlements = settlement_records(state_dir)
    assert len(settlements) == 1
    assert int(refusals[0]["seq"]) < int(settlements[0]["seq"]) < int(refusals[1]["seq"])


def test_a_disputed_runner_decides_no_minute_even_when_asked_directly(tmp_path):
    """`tick` is the one method that writes a `risk.state_hash`, so it refuses.

    `catch_up` and `run_minutes` already stop at HALT; this is the method they
    call. A DECISION from a disputed process would quote an engine that
    persisted nothing, become the log's newest witness, and end the refusal on
    a state no file holds.
    """
    from chimera.demo.runner import RunnerError

    harness = campaign(tmp_path)
    harness.runner.shutdown("clean stop")
    config, state_dir = harness.runner.config, harness.state_dir
    (state_dir / "risk.json").unlink()
    before = last_seq(state_dir)

    refused = build(tmp_path, days=(DAY,), config=config, start=False)
    assert refused.runner.start() is RunnerState.HALT
    with pytest.raises(RunnerError, match="cannot decide a minute"):
        refused.tick(refused.runner.cursor.next_minute_ms())
    assert refused.runner.catch_up() == []
    assert decisions_after(state_dir, before) == []


def test_an_engine_held_as_evidence_writes_nothing_through_any_mutation(tmp_path):
    """The persistence boundary itself, two-sided, across every writer.

    `RiskEngine._persist` is the one path to the file, so holding it holds every
    mutation -- listed here so a writer added later that bypassed it would have
    to be added to this list to pass.
    """
    from datetime import datetime, timezone

    from chimera.risk import RiskViolation

    path = tmp_path / "risk.json"
    switch = tmp_path / "KILL_SWITCH"

    def clock() -> float:
        return 1_758_240_000.0  # 2025-09-19, fixed: the point is the file

    def exercise(engine: RiskEngine) -> None:
        engine.update_equity(990_000.0, now=datetime(2026, 9, 19, tzinfo=timezone.utc))
        engine.record_order()
        engine.set_position_exposure("BTC/USDT", 1_000.0)
        engine.close_position("BTC/USDT")
        engine.note_reconciliation("BTC/USDT", "venue disagreed")
        engine.note_reconciliation("BTC/USDT", None)
        engine.record_trade_result(-10.0)
        engine.halt("an operator halt")
        engine.resume()
        switch.write_text("engaged", encoding="utf-8")
        engine.check_kill_switch()
        switch.unlink()
        engine.check_kill_switch()

    # Held: an existing file keeps its bytes, and a missing one stays missing.
    seed = RiskEngine(RiskLimits(), state_path=path, clock=clock)
    seed.update_equity(1_000_000.0)
    found = path.read_bytes()
    held = RiskEngine(
        RiskLimits(), state_path=path, clock=clock, kill_switch_path=switch, persist=False
    )
    exercise(held)
    assert path.read_bytes() == found
    assert (
        held.state.equity != RiskEngine(RiskLimits(), state_path=path).state.equity
    ), "the held engine really did move, in memory"
    absent = tmp_path / "absent.json"
    exercise(
        RiskEngine(
            RiskLimits(),
            state_path=absent,
            clock=clock,
            kill_switch_path=switch,
            persist=False,
        )
    )
    assert not absent.exists()

    # And an unreadable file is not moved aside by an engine holding it.
    (tmp_path / "damaged.json").write_bytes(b"{ not a state")
    damaged = RiskEngine(
        RiskLimits(), state_path=tmp_path / "damaged.json", clock=clock, persist=False
    )
    with pytest.raises(RiskViolation, match="evidence"):
        damaged.adopt_after_unreadable("operator: start again")
    assert (tmp_path / "damaged.json").read_bytes() == b"{ not a state"

    # The other side: the same mutations on an ordinary engine do write.
    exercise(RiskEngine(RiskLimits(), state_path=path, clock=clock, kill_switch_path=switch))
    assert path.read_bytes() != found
