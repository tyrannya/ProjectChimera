"""R1-c (AEG-4): whether ``risk.json`` continues this campaign's decision log.

The adopted roadmap's item:

    a missing or unreadable ``risk.json`` when the decision log already holds
    records is a dispute, not a default; the loaded snapshot is compared with
    the log's last ``risk.state_hash``/HALT record; a ``RECOVERY`` record is
    written; the same continuity the store and ledger already have.

R1-b stopped one step short of this on purpose: its
``seed_or_reconcile_equity`` reads ``MISSING`` as a genuine first start because
it reads no log. Delete ``risk.json`` from a campaign whose log holds a month of
records and the next start seeded the configured capital over the top of it.

**Every test here is two-sided.** The first-start control is paired with the
missing-file dispute; the healthy restart is paired with the mismatch; the
matching persisted HALT is paired with the halt cleared behind the log's back;
and the deferral to section 9.3 is paired with the swap that has no crash to
hide behind. A one-sided "it halts" would pass against an engine that halts on
everything, and a one-sided "it does not halt" against one that checks nothing.

**Nothing here hand-writes the state it is about.** The campaigns are driven
through ``tests.demo_harness``, which builds its engine through the production
``build_risk_engine`` and its runner through the production ``DemoRunner``. What
the tests do edit is the thing an operator's accident really edits: a state file
is deleted, truncated or replaced *after* a real campaign wrote it. A fixture
that wrote the exact final state it then asserted would be asserting about
itself.

Nothing here is scientific evidence. The prices come from
``chimera.demo.fixtures`` and every number is the test author's arithmetic.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Callable

import pytest

import chimera.demo.risk_continuity as risk_continuity
import chimera.demo.risk_wiring as risk_wiring
import chimera.demo.runner as runner_module
from chimera.demo.decision_log import LOG_DIR_NAME, RecordKind
from chimera.demo.fixtures import SyntheticFeed
from chimera.demo.risk_continuity import (
    RISK_CONTINUITY_PREFIX,
    RISK_CONTINUITY_SEALED_FIELD,
    LogRiskStatement,
    RiskContinuityFault,
    assess_risk_continuity,
    read_log_risk_history,
    risk_state_hash,
)
from chimera.demo.runner import (
    CAUSES_WITHOUT_AN_AFFECTED_MINUTE,
    RecoveryCause,
    RunnerError,
    RunnerState,
    _risk_hash,
)
from chimera.demo.rules import RuleError
from chimera.demo.rules_carry import CarryRule
from chimera.risk import RiskEngine, RiskLimits, RiskState, RiskStateLoad, RiskViolation
from tests.demo_harness import CAPITAL, DAY, NEXT_DAY, build, campaign_config
from tests.test_demo_cli import written_config
from tools import demo_run

#: Enough minutes for the campaign to open a position, book fees and write
#: several hash-bearing records, with minutes left over for a restart to decide.
MINUTES = 5


# ---------------------------------------------------------------------------
# reading what a campaign left behind
# ---------------------------------------------------------------------------
def records(state_dir: Path) -> list[dict[str, Any]]:
    """Every committed record, in chain order."""
    out: list[dict[str, Any]] = []
    for path in sorted((state_dir / LOG_DIR_NAME).glob("*.ndjson")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
    return out


def continuity_recoveries(state_dir: Path) -> list[dict[str, Any]]:
    """The ``RECOVERY`` records R1-c wrote, and only those.

    Selected by the block R1-c puts inside ``recovery`` rather than by cause, so
    a test that asserts on the cause is asserting about something this helper
    did not already assume.
    """
    return [
        record
        for record in records(state_dir)
        if record["kind"] == RecordKind.RECOVERY.value
        and "risk_continuity" in record.get("recovery", {})
    ]


def causes(state_dir: Path) -> list[str]:
    return [record["recovery"]["cause"] for record in continuity_recoveries(state_dir)]


def risk_json(state_dir: Path) -> Path:
    return state_dir / "risk.json"


def bytes_of(path: Path) -> bytes | None:
    """The file's bytes, or ``None`` when it is not there. Both are evidence."""
    return path.read_bytes() if path.exists() else None


# ---------------------------------------------------------------------------
# driving one
# ---------------------------------------------------------------------------
def campaign(tmp_path: Path, *, minutes: int = MINUTES):
    """A real campaign that has decided ``minutes`` minutes and persisted them."""
    harness = build(tmp_path, days=(DAY,))
    harness.run(minutes)
    return harness


def restart(tmp_path: Path, config, *, days: tuple[str, ...] = (DAY,)) -> Any:
    """A second process over the same files, as production really builds one.

    ``now_ns`` is seeded past the log's tail because the campaigns here are
    driven over a fixture day and a halted tick does not mark its minute
    processed -- so ``start()``'s "first unprocessed minute" can be BEHIND the
    instant the previous process last recorded, and ``DecisionLog.append``
    rightly refuses a record whose clock moved backwards. Seeding the clock from
    the log's tail is canonical R1-i's item for the CLI; here it is the harness
    standing in for the recorder minutes a real campaign would have.
    """
    state_dir = Path(config.runner_setting("state_dir"))
    committed = records(state_dir)
    resumed = build(tmp_path, days=days, config=config, start=False)
    tail = max((int(r["runner_now_ns"]) for r in committed), default=None)
    resumed.runner.start(now_ns=None if tail is None else tail + 60_000_000_000)
    return resumed


def halt_the_campaign(harness) -> None:
    """Halt through a real guard: the operator's kill switch, then a tick.

    Not ``runner._halt(...)`` directly. The point of the HALT records below is
    that they are the ones a campaign really writes, including what Aegis
    persisted beside them.
    """
    switch = harness.state_dir / "KILL_SWITCH"
    switch.write_text("stop\n", encoding="utf-8")
    harness.runner.tick(harness.first_minute_ms() + MINUTES * 60_000)
    assert harness.runner.state is RunnerState.HALT
    switch.unlink()


def clear_the_halt_in_the_file(state_dir: Path) -> None:
    """What "cleared behind the log's back" means, spelled out on disk."""
    document = json.loads(risk_json(state_dir).read_text(encoding="utf-8"))
    document["halted"] = False
    document["halt_reason"] = ""
    document["kill_switch"] = False
    risk_json(state_dir).write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def move_the_peak(state_dir: Path) -> None:
    """A readable, current-schema state that is not the one the log hashed.

    ``peak_equity`` is chosen because it is hashed, it is not the equity -- so
    R1-b's ledger reconciliation cannot be what catches this -- and raising it
    is what a stale copy of the file would carry.
    """
    document = json.loads(risk_json(state_dir).read_text(encoding="utf-8"))
    document["peak_equity"] = float(document["peak_equity"]) + 12_345.0
    risk_json(state_dir).write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def pre_b5_sealed_run_rule(found: LogRiskStatement, *, sealed_run: bool) -> LogRiskStatement:
    """``risk_continuity._statement_in_run`` as it was before PR #102's B5 fix.

    A sealed run's HALT, RESUME and MOVED records were ignored and its
    restated ``risk.state_hash`` was still read as a statement. Monkeypatched
    in as a mutant only.
    """
    suppressed = (LogRiskStatement.HALTED, LogRiskStatement.RUNNING, LogRiskStatement.MOVED)
    return LogRiskStatement.NONE if sealed_run and found in suppressed else found


def forge_the_log(state_dir: Path) -> None:
    """Rewrite a COMPLETE record so it no longer hashes to what it declares."""
    path = sorted((state_dir / LOG_DIR_NAME).glob("*.ndjson"))[-1]
    lines = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[-1])
    record["runner_now_ns"] = int(record["runner_now_ns"]) + 1
    lines[-1] = json.dumps(record, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. the negative control: a genuine first start
# ---------------------------------------------------------------------------
def test_a_genuine_first_start_is_still_a_first_start(tmp_path):
    """Case A. No file and no log: nothing to be continuous with.

    The control the whole item stands on. A check that disputed a missing
    ``risk.json`` unconditionally would make every campaign unstartable, and
    would pass the missing-with-history test below just as well.
    """
    harness = build(tmp_path, days=(DAY,))

    assert harness.runner.state is RunnerState.READY
    assert harness.runner.risk.load_outcome is RiskStateLoad.MISSING
    assert harness.runner.risk.state.equity == pytest.approx(float(CAPITAL))
    assert not harness.runner.risk.state.halted
    assert not harness.runner.risk_continuity.disputed
    assert not continuity_recoveries(harness.state_dir)
    assert risk_json(harness.state_dir).is_file(), "a first start still persists"


def test_a_first_start_is_read_off_the_log_and_not_off_the_equity(tmp_path):
    """The distinction R1-b made for ``load_outcome``, made again for the log.

    A campaign that has decided minutes and is worth exactly its capital is
    indistinguishable from a first start by its numbers. It is not
    indistinguishable by its log, which is the whole point of reading one.
    """
    harness = campaign(tmp_path)
    history = read_log_risk_history(harness.state_dir)

    assert history.has_history
    assert history.records == len(records(harness.state_dir))
    assert history.statement is LogRiskStatement.STATE_HASH


# ---------------------------------------------------------------------------
# 2. a missing file against a history
# ---------------------------------------------------------------------------
def test_a_missing_risk_state_with_a_verified_history_is_a_dispute(tmp_path):
    """Case B. The defect itself: capital may not seed over a campaign's history."""
    harness = campaign(tmp_path)
    config = harness.runner.config
    state_dir = harness.state_dir
    persisted = json.loads(risk_json(state_dir).read_text(encoding="utf-8"))
    assert persisted["peak_equity"] > 0.0, "the campaign must have something to lose"
    risk_json(state_dir).unlink()

    resumed = restart(tmp_path, config)

    assert resumed.runner.state is RunnerState.HALT
    assert resumed.runner.risk.state.halted
    assert resumed.runner.risk.state.halt_reason.startswith(RISK_CONTINUITY_PREFIX)
    assert causes(state_dir) == [RecoveryCause.RISK_STATE_ABSENT.value]
    # Nothing was seeded, and nothing was invented in the file's place.
    assert resumed.runner.risk.state.equity == 0.0
    assert not risk_json(state_dir).exists(), (
        "halting must not write the file the dispute is about into existence; the "
        "next start would then find a LOADED state this process made up"
    )


def test_the_missing_file_dispute_does_not_seed_the_configured_capital(tmp_path):
    """The other half of B, stated as the number rather than as the halt."""
    harness = campaign(tmp_path)
    config = harness.runner.config
    risk_json(harness.state_dir).unlink()

    resumed = restart(tmp_path, config)

    assert resumed.runner.risk.state.equity != pytest.approx(float(CAPITAL))
    assert resumed.runner.risk.state.peak_equity == 0.0


# ---------------------------------------------------------------------------
# 3. an unreadable file against a history
# ---------------------------------------------------------------------------
def test_an_unreadable_risk_state_with_a_history_is_recorded_and_preserved(tmp_path):
    """Case C. R1-b's fail-closed halt stands; R1-c puts the loss on the record."""
    harness = campaign(tmp_path)
    config = harness.runner.config
    state_dir = harness.state_dir
    risk_json(state_dir).write_text("{not json", encoding="utf-8")
    truncated = bytes_of(risk_json(state_dir))

    resumed = restart(tmp_path, config)

    assert resumed.runner.state is RunnerState.HALT
    assert resumed.runner.risk.load_outcome is RiskStateLoad.UNREADABLE
    assert causes(state_dir) == [RecoveryCause.RISK_STATE_UNREADABLE.value]
    assert bytes_of(risk_json(state_dir)) == truncated, (
        "R1-b preserves the one record of what the account was doing, and R1-c "
        "may not take that away by halting on top of it"
    )
    assert resumed.runner.risk.state.halt_reason.startswith(
        "unreadable persisted risk state"
    ), "halt keeps the FIRST reason: R1-b named the file, and that is the useful one"


def test_an_unreadable_risk_state_without_a_history_raises_no_continuity_finding(
    tmp_path,
):
    """The two-sided pair: no log, so there is nothing for continuity to be about.

    R1-b still fails the engine closed -- that is its own witness -- and R1-c
    adds no record, because a campaign with no committed record has no history
    the file could have lost.
    """
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True)
    risk_json(state_dir).write_text("{not json", encoding="utf-8")

    engine = risk_wiring.build_risk_engine(
        campaign_config(state_dir), capital=CAPITAL, state_dir=state_dir
    )

    assert engine.load_outcome is RiskStateLoad.UNREADABLE
    assert engine.state.halted, "R1-b's fail-closed halt is untouched"
    assert not engine.continuity_disputed
    verdict = assess_risk_continuity(
        load=engine.load_outcome, snapshot=engine.loaded_snapshot, state_dir=state_dir
    )
    assert not verdict.disputed


# ---------------------------------------------------------------------------
# 4. a matching file: the false-positive control
# ---------------------------------------------------------------------------
def test_a_matching_risk_state_restarts_without_a_dispute(tmp_path):
    """Case D. The main false-positive control, and it must stay boring."""
    harness = campaign(tmp_path)
    config = harness.runner.config
    state_dir = harness.state_dir
    before = bytes_of(risk_json(state_dir))

    resumed = restart(tmp_path, config)

    assert resumed.runner.state is RunnerState.READY, resumed.runner.halt_reason
    assert not resumed.runner.risk_continuity.disputed
    assert not continuity_recoveries(state_dir)
    assert bytes_of(risk_json(state_dir)) == before, "a healthy restart writes nothing"
    # And it can really decide again: a campaign that starts and cannot tick is
    # not a campaign that restarted.
    minute = resumed.runner.cursor.next_minute_ms()
    assert minute is not None
    resumed.tick(minute)
    assert resumed.runner.state is not RunnerState.HALT, resumed.runner.halt_reason


def test_the_file_the_campaign_wrote_hashes_to_what_the_log_recorded(tmp_path):
    """The invariant the whole comparison rests on, asserted directly.

    If this ever stops holding for an ordinary campaign, every restart becomes a
    dispute -- so it is worth an assertion of its own rather than being implied
    by the test above.
    """
    harness = campaign(tmp_path)
    history = read_log_risk_history(harness.state_dir)
    engine = RiskEngine(
        risk_wiring.risk_limits(harness.runner.config.limits),
        state_path=risk_json(harness.state_dir),
    )

    assert history.statement is LogRiskStatement.STATE_HASH
    assert risk_state_hash(engine.loaded_snapshot) == history.state_hash


@pytest.mark.parametrize("after", ["shutdown", "flatten"])
def test_the_last_risk_statement_is_not_the_last_record(tmp_path, after):
    """The load-bearing one: an ordinary tail must not read as a disagreement.

    A ``SHUTDOWN`` record carries no risk block and moves nothing. An
    ``OPERATOR`` flatten carries none EITHER and does move the equity -- it
    hands the flattened figure to Aegis, which persists it -- so a comparison
    that took "the newest record carrying a hash" and demanded equality would
    call the most ordinary operator action in the runbook a continuity failure.
    """
    harness = campaign(tmp_path)
    config = harness.runner.config
    state_dir = harness.state_dir
    if after == "shutdown":
        harness.runner.shutdown("cli run complete")
    else:
        harness.runner.flatten("operator: reduce to flat for maintenance")
    committed = records(state_dir)
    assert "risk" not in committed[-1], "the tail must really carry no risk block"

    resumed = restart(tmp_path, config)

    assert resumed.runner.state is RunnerState.READY, resumed.runner.halt_reason
    assert not continuity_recoveries(state_dir)


def test_an_operator_flatten_really_moves_the_state_the_log_last_hashed(tmp_path):
    """Otherwise the test above is witnessing a tail that changed nothing."""
    harness = campaign(tmp_path)
    history_before = read_log_risk_history(harness.state_dir)
    harness.runner.flatten("operator: reduce to flat for maintenance")

    engine = RiskEngine(
        risk_wiring.risk_limits(harness.runner.config.limits),
        state_path=risk_json(harness.state_dir),
    )
    assert risk_state_hash(engine.loaded_snapshot) != history_before.state_hash
    assert read_log_risk_history(harness.state_dir).statement is LogRiskStatement.MOVED


# ---------------------------------------------------------------------------
# 5. a file that is not the one the log hashed
# ---------------------------------------------------------------------------
def test_a_risk_state_that_does_not_match_the_logs_hash_is_a_dispute(tmp_path):
    """Case E. A stale or foreign file, with no crash anywhere to explain it."""
    harness = campaign(tmp_path)
    config = harness.runner.config
    state_dir = harness.state_dir
    move_the_peak(state_dir)
    replaced = bytes_of(risk_json(state_dir))

    resumed = restart(tmp_path, config)

    assert resumed.runner.state is RunnerState.HALT
    assert causes(state_dir) == [RecoveryCause.RISK_STATE_MISMATCH.value]
    assert resumed.runner.risk.state.halt_reason.startswith(RISK_CONTINUITY_PREFIX)
    assert bytes_of(risk_json(state_dir)) == replaced, (
        "neither side is rewritten to make them agree, and the disputed bytes are "
        "what an operator has to look at"
    )
    block = continuity_recoveries(state_dir)[0]["recovery"]["risk_continuity"]
    assert block["log_state_hash"] != block["found_state_hash"]
    assert block["log_statement"] == LogRiskStatement.STATE_HASH.value


def test_a_recorded_dispute_does_not_become_its_own_evidence(tmp_path):
    """The hash R1-c writes into its RECOVERY record may never be read back as
    the log's ``risk.state_hash``.

    It is in the ``recovery`` block and not in a ``risk`` one, and the statement
    scan looks for the ``risk`` block by name. A scan that took "the newest
    record containing a state_hash" would find the disputed hash here, compare
    the file against itself, and let the campaign run.
    """
    harness = campaign(tmp_path)
    config = harness.runner.config
    state_dir = harness.state_dir
    move_the_peak(state_dir)
    restart(tmp_path, config)
    assert causes(state_dir) == [RecoveryCause.RISK_STATE_MISMATCH.value]

    disputed_hash = continuity_recoveries(state_dir)[0]["recovery"]["risk_continuity"][
        "found_state_hash"
    ]

    again = restart(tmp_path, config)

    assert again.runner.state is RunnerState.HALT, "the dispute survives its own record"
    assert again.runner.risk_continuity.disputed
    history = read_log_risk_history(state_dir)
    assert history.state_hash != disputed_hash, (
        "the statement is still the campaign's own last restatement, not the hash "
        "the dispute recorded about the file it refused"
    )
    assert history.state_hash == again.runner.risk_continuity.history.state_hash


# ---------------------------------------------------------------------------
# 6. HALT continuity
# ---------------------------------------------------------------------------
def test_a_matching_persisted_halt_is_not_a_mismatch(tmp_path):
    """Case F, first half. A halted campaign restarts halted FOR ITS OWN REASON.

    The false-positive that matters most here: the ``HALT`` record carries no
    risk block, so the hash the log last restated is the one from before the
    halt and the file has legitimately moved past it. Treating that as a
    disagreement would turn every halted restart -- including R1-b's equity
    dispute -- into an unclearable continuity halt.
    """
    harness = campaign(tmp_path)
    halt_the_campaign(harness)
    config = harness.runner.config
    state_dir = harness.state_dir
    assert read_log_risk_history(state_dir).statement is LogRiskStatement.HALTED

    resumed = restart(tmp_path, config)

    assert resumed.runner.state is RunnerState.HALT
    assert resumed.runner.risk.state.halt_reason == "kill_switch", (
        "the campaign stays halted for the reason it halted for, not for a "
        "continuity failure R1-c invented"
    )
    assert not continuity_recoveries(state_dir)
    assert not resumed.runner.risk_continuity.disputed


def test_a_logged_halt_with_no_risk_state_cannot_be_started_from(tmp_path):
    """Case F, second half: the halt must not be lost with the file."""
    harness = campaign(tmp_path)
    halt_the_campaign(harness)
    config = harness.runner.config
    state_dir = harness.state_dir
    risk_json(state_dir).unlink()

    resumed = restart(tmp_path, config)

    assert resumed.runner.state is RunnerState.HALT
    assert causes(state_dir) == [RecoveryCause.RISK_STATE_ABSENT.value]
    assert not risk_json(state_dir).exists()
    block = continuity_recoveries(state_dir)[0]["recovery"]["risk_continuity"]
    assert block["log_statement"] == LogRiskStatement.HALTED.value


def test_a_halt_cleared_behind_the_log_is_a_dispute(tmp_path):
    """Case F, third half: the log says halted and the file says running."""
    harness = campaign(tmp_path)
    halt_the_campaign(harness)
    config = harness.runner.config
    state_dir = harness.state_dir
    clear_the_halt_in_the_file(state_dir)
    cleared = bytes_of(risk_json(state_dir))

    resumed = restart(tmp_path, config)

    assert resumed.runner.state is RunnerState.HALT
    assert causes(state_dir) == [RecoveryCause.RISK_STATE_MISMATCH.value]
    assert bytes_of(risk_json(state_dir)) == cleared, "the edited file is the evidence"


def test_a_file_more_cautious_than_the_log_is_not_a_dispute(tmp_path):
    """The direction R1-c deliberately does not police.

    A ``RESUME`` is the log's newest statement and the file is halted -- which a
    crash between ``RiskEngine.resume``'s write and the record that follows it
    leaves behind. The file claims MORE restriction than the log, not less, and
    halting a campaign for being too careful would be a refusal with no safety
    in it.
    """
    history = risk_continuity.LogRiskHistory(
        records=7, statement=LogRiskStatement.RUNNING, statement_kind="RESUME"
    )
    snapshot = dict(_snapshot_of(halted=True))

    verdict = assess_risk_continuity(
        load=RiskStateLoad.LOADED,
        snapshot=snapshot,
        state_dir=tmp_path,
        history=history,
    )

    assert not verdict.disputed


def _snapshot_of(*, halted: bool) -> dict[str, Any]:
    from chimera.risk import RiskState

    return RiskState(halted=halted, halt_reason="x" if halted else "").snapshot()


# ---------------------------------------------------------------------------
# 7. the pre-schema document
# ---------------------------------------------------------------------------
def test_a_legacy_risk_state_against_a_history_is_refused(tmp_path):
    """A halt claim and no account state cannot be a campaign's persisted state.

    R1-b seeds the configured capital for a ``LEGACY`` load, exactly as it does
    for ``MISSING``, because the pre-schema file makes no equity claim. Against a
    log that holds records that seed is the same fabrication the missing-file
    case is, so it is refused and said so.
    """
    harness = campaign(tmp_path)
    config = harness.runner.config
    state_dir = harness.state_dir
    risk_json(state_dir).write_text(
        json.dumps({"halted": False, "halt_reason": "", "updated_at": "2026-09-19"}),
        encoding="utf-8",
    )
    legacy = bytes_of(risk_json(state_dir))

    resumed = restart(tmp_path, config)

    assert resumed.runner.state is RunnerState.HALT
    assert resumed.runner.risk.load_outcome is RiskStateLoad.LEGACY
    assert causes(state_dir) == [RecoveryCause.RISK_STATE_PRE_SCHEMA.value]
    assert resumed.runner.risk.state.equity == 0.0, "no capital was seeded"
    assert bytes_of(risk_json(state_dir)) == legacy


def test_a_legacy_risk_state_without_a_history_keeps_r1bs_treatment(tmp_path):
    """The pair: with no committed record there is nothing to contradict."""
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True)
    risk_json(state_dir).write_text(
        json.dumps({"halted": False, "halt_reason": "", "updated_at": "2026-09-19"}),
        encoding="utf-8",
    )

    engine = risk_wiring.build_risk_engine(
        campaign_config(state_dir), capital=CAPITAL, state_dir=state_dir
    )

    assert engine.load_outcome is RiskStateLoad.LEGACY
    assert not engine.state.halted
    assert engine.state.equity == pytest.approx(float(CAPITAL))


# ---------------------------------------------------------------------------
# 8. composition with the machinery that already exists
# ---------------------------------------------------------------------------
def test_a_forged_log_keeps_its_own_fail_closed_behaviour(tmp_path):
    """Case G. A log nobody can trust may not condemn a state file...

    ...and it may not excuse a fabricated one either. The campaign halts on
    ``LOG_FORGED``, which is the graver finding and the one the existing triage
    owns, and the engine is still refused a seed: nothing hands it the
    configured capital on the way past.
    """
    harness = campaign(tmp_path)
    config = harness.runner.config
    state_dir = harness.state_dir
    forge_the_log(state_dir)
    risk_json(state_dir).unlink()

    resumed = restart(tmp_path, config)

    assert resumed.runner.state is RunnerState.HALT
    assert (
        resumed.runner.risk.state.equity == 0.0
    ), "a bad log plus a missing risk.json may never resolve to a healthy state"
    assert not risk_json(state_dir).exists()


def test_a_forged_log_does_not_produce_a_hash_mismatch(tmp_path):
    """The other direction of case G: an edited log cannot condemn a good file."""
    harness = campaign(tmp_path)
    state_dir = harness.state_dir
    forge_the_log(state_dir)
    history = read_log_risk_history(state_dir)
    assert not history.trustworthy

    engine = RiskEngine(
        risk_wiring.risk_limits(harness.runner.config.limits),
        state_path=risk_json(state_dir),
    )
    verdict = assess_risk_continuity(
        load=engine.load_outcome,
        snapshot=engine.loaded_snapshot,
        state_dir=state_dir,
        history=history,
    )

    assert not verdict.disputed


def test_a_crash_inside_a_tick_stays_section_9_3s_recovery(tmp_path):
    """The deferral, and why R1-c may not claim section 9.3's crash.

    A booking persisted before the record that quotes it leaves the state files
    -- Aegis's among them -- genuinely AHEAD of the log, which is
    ``LOG_BEHIND_STATE``: the runner names it, excludes what it must and
    continues. The hash comparison sees the same thing a swapped file looks
    like, so R1-c stands down when the triage has really found a crash.
    """
    harness = campaign(tmp_path, minutes=3)
    config = harness.runner.config
    state_dir = harness.state_dir
    crash_the_campaign(harness)

    resumed = restart(tmp_path, config)

    crash = [
        r
        for r in records(state_dir)
        if r["kind"] == RecordKind.RECOVERY.value
        and r["recovery"]["cause"] == RecoveryCause.LOG_BEHIND_STATE.value
    ]
    assert crash, "the crash window must still be recovered from"
    assert not continuity_recoveries(state_dir), (
        "one crash is one finding; R1-c must not report it a second time under a "
        "name that says a file was swapped"
    )
    assert resumed.runner.state is not RunnerState.HALT, resumed.runner.halt_reason


def test_the_deferral_is_narrow_enough_to_still_catch_a_swap(tmp_path):
    """The pair for the deferral: no crash, so there is nothing to defer to.

    ``test_a_risk_state_that_does_not_match_the_logs_hash_is_a_dispute`` is that
    case end to end; this states the predicate directly so a widening of the
    deferral is a failure here as well as there.
    """
    harness = campaign(tmp_path)
    move_the_peak(harness.state_dir)
    resumed = build(tmp_path, days=(DAY,), config=harness.runner.config, start=False)

    assert resumed.runner.risk_continuity.disputed
    assert resumed.runner.risk_continuity.crash_could_explain
    assert resumed.runner._risk_continuity_stands(None), "no crash, no deferral"
    assert not resumed.runner._risk_continuity_stands(
        runner_module._Recovery(
            cause=RecoveryCause.LOG_BEHIND_STATE, reason="", recoverable=True
        )
    )


def crash_the_campaign(harness) -> None:
    """Leave the state files genuinely AHEAD of the log, as a kill mid-tick does.

    A booking is persisted and the record that would quote it is never written,
    which is what ``DemoRunner._state_ahead_of_log`` reports as
    ``LOG_BEHIND_STATE``.
    """
    from chimera.futures.executor import FlattenCause

    state = harness.runner.cursor.state_for(
        harness.first_minute_ms() + 2 * 60_000, now_ns=harness.runner.clock.now_ns
    )
    harness.runner.position.install_quote(state)
    harness.runner.position.emergency_reduce(FlattenCause.RISK_HALT, state)
    harness.runner._save_ledger()


@pytest.mark.parametrize(
    ("mutate", "cause"),
    [
        (lambda sd: risk_json(sd).unlink(), RecoveryCause.RISK_STATE_ABSENT),
        (
            lambda sd: risk_json(sd).write_text("{not json", encoding="utf-8"),
            RecoveryCause.RISK_STATE_UNREADABLE,
        ),
        (
            lambda sd: risk_json(sd).write_text(
                json.dumps({"halted": False, "halt_reason": "", "updated_at": "x"}),
                encoding="utf-8",
            ),
            RecoveryCause.RISK_STATE_PRE_SCHEMA,
        ),
    ],
    ids=("absent", "unreadable", "pre-schema"),
)
def test_the_faults_no_crash_produces_are_never_deferred(tmp_path, mutate, cause):
    """A crash cannot delete a file, corrupt one, or make it pre-schema.

    The narrow half of the deferral, end to end: the campaign really has the
    crash section 9.3 recovers from AND has lost its risk state, and the second
    finding is not swallowed by the first. A deferral written as "any crash
    stands the finding down" passes
    `test_a_crash_inside_a_tick_stays_section_9_3s_recovery` and fails here,
    which is the only reason that test cannot be the whole witness.
    """
    harness = campaign(tmp_path, minutes=3)
    config = harness.runner.config
    state_dir = harness.state_dir
    crash_the_campaign(harness)
    mutate(state_dir)

    resumed = restart(tmp_path, config)

    crash = [
        r
        for r in records(state_dir)
        if r["kind"] == RecordKind.RECOVERY.value
        and r["recovery"]["cause"] == RecoveryCause.LOG_BEHIND_STATE.value
    ]
    assert crash, "the crash must really be there, or the deferral is not exercised"
    assert causes(state_dir) == [
        cause.value
    ], "and the lost risk state is reported beside it, not instead of it"
    assert resumed.runner.state is RunnerState.HALT


def test_an_empty_log_beside_a_live_risk_state_is_not_r1cs_question(tmp_path):
    """Case H, pinned rather than answered.

    The roadmap's R1-c is one direction only: a risk state that cannot account
    for the log's records. The OPPOSITE -- a log that has lost the records a
    live risk state was written against -- is section 9.3's persisted
    ``last_record_hash`` question and canonical R1-i's dispute lifecycle, and
    this item does not widen into it. What is asserted here is that R1-c raises
    nothing, so a later item can decide that direction without first undoing a
    semantic invented here.
    """
    harness = campaign(tmp_path)
    config = harness.runner.config
    state_dir = harness.state_dir
    # Through `shutdown`, and not because the test wants a SHUTDOWN record --
    # the whole directory goes a line later. The runner holds the day file OPEN
    # for append, and Windows refuses to unlink a file another handle has open
    # (POSIX allows it, which is why this passed locally and failed on the
    # windows-latest job). The production entry point is what closes the log, so
    # it is what this uses; the assertion below is what stops the fix being
    # quietly undone.
    harness.runner.shutdown("closing the log before the directory is removed")
    assert harness.runner._log is None, "the day file must not still be open"
    shutil.rmtree(state_dir / LOG_DIR_NAME)

    resumed = build(tmp_path, days=(DAY,), config=config, start=False)

    assert not resumed.runner.risk_continuity.disputed
    assert resumed.runner.risk_continuity.history.records == 0


# ---------------------------------------------------------------------------
# 9. idempotence
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("mutate", "cause"),
    [
        (lambda sd: risk_json(sd).unlink(), RecoveryCause.RISK_STATE_ABSENT),
        (
            lambda sd: risk_json(sd).write_text("{not json", encoding="utf-8"),
            RecoveryCause.RISK_STATE_UNREADABLE,
        ),
        (move_the_peak, RecoveryCause.RISK_STATE_MISMATCH),
    ],
    ids=("absent", "unreadable", "mismatch"),
)
def test_repeated_restarts_record_the_finding_once(tmp_path, mutate, cause):
    """Restart three times: one RECOVERY record, and nothing walks anywhere.

    A disputed campaign gets restarted -- by the operator looking at it, by a
    service manager, by a command that has to ``start()`` first -- and a record
    per restart would bury the finding in copies of itself. The HALT record each
    restart writes is unchanged: that is the existing treatment of every
    repeated halt and R1-c does not quietly alter it.
    """
    harness = campaign(tmp_path)
    config = harness.runner.config
    state_dir = harness.state_dir
    mutate(state_dir)
    after_mutation = bytes_of(risk_json(state_dir))

    for _ in range(3):
        resumed = restart(tmp_path, config)
        assert resumed.runner.state is RunnerState.HALT

    assert causes(state_dir) == [cause.value], "one finding, one record"
    assert (
        bytes_of(risk_json(state_dir)) == after_mutation
    ), "and the evidence does not drift between restarts"


def test_a_fault_that_recurs_after_a_repair_is_recorded_again(tmp_path):
    """The pair for idempotence: silence must not outlive the finding.

    A campaign that was repaired, decided another minute and then lost its
    ``risk.json`` a second time has a hash-bearing record in between, so the
    second loss is the new event it is.
    """
    harness = campaign(tmp_path)
    config = harness.runner.config
    state_dir = harness.state_dir
    saved = bytes_of(risk_json(state_dir))
    risk_json(state_dir).unlink()
    restart(tmp_path, config)
    assert causes(state_dir) == [RecoveryCause.RISK_STATE_ABSENT.value]

    # The repair, and a minute decided under it: the log restates the hash.
    risk_json(state_dir).write_bytes(saved)
    repaired = restart(tmp_path, config)
    assert repaired.runner.state is RunnerState.READY, repaired.runner.halt_reason
    minute = repaired.runner.cursor.next_minute_ms()
    assert minute is not None
    repaired.tick(minute)

    risk_json(state_dir).unlink()
    restart(tmp_path, config)

    assert causes(state_dir) == [RecoveryCause.RISK_STATE_ABSENT.value] * 2


# ---------------------------------------------------------------------------
# 10. what the record says, and what it must not disturb
# ---------------------------------------------------------------------------
def test_the_recovery_record_excludes_no_minute(tmp_path):
    """R1-c's findings are about a FILE, so no minute was lost to them.

    Writing an ``evidence_excluded_minute`` for one would exclude a minute
    ``tools/replay_parity.py`` goes on reading -- two files disagreeing about
    what this campaign excluded, which is exactly what the null for
    ``LOG_AHEAD_OF_STATE`` exists to avoid.
    """
    harness = campaign(tmp_path)
    config = harness.runner.config
    state_dir = harness.state_dir
    risk_json(state_dir).unlink()

    restart(tmp_path, config)

    recovery = continuity_recoveries(state_dir)[0]["recovery"]
    assert recovery["evidence_excluded_minute"] is None
    assert recovery["minute_finished"] is True
    assert recovery["truncated_bytes"] == 0


def test_the_daily_report_reads_the_finding_it_was_not_written_for(tmp_path):
    """The record is new; the report that has to read it is not.

    `reports._halts_block` counts RECOVERY records and quotes each one's
    `veto_or_rejection.detail`. A block shaped so that the report could not read
    it would leave the day a campaign lost its risk state looking, in the one
    document an operator files, like a day nothing happened.

    The halt itself collapses to `other`, which is R1-b's `equity_dispute:`
    treatment too: naming a new cause in `HALT_CAUSES` is a reports decision and
    PR-12 owns it.
    """
    from chimera.demo.reports import daily_report

    harness = campaign(tmp_path)
    config = harness.runner.config
    state_dir = harness.state_dir
    risk_json(state_dir).unlink()
    restart(tmp_path, config)

    halts = daily_report(state_dir, DAY)["halts"]

    assert halts["recoveries"] == 1
    assert "there is no persisted risk.json" in halts["recovery_events"][0]["detail"]
    assert halts["count"] == 1, "and the halt beside it is counted too"


def test_the_r1c_causes_are_not_in_the_parity_tools_excluding_set():
    """The two sets are written down twice and must never overlap."""
    from tools.replay_parity import EXCLUDING_RECOVERY_CAUSES

    names = {cause.value for cause in CAUSES_WITHOUT_AN_AFFECTED_MINUTE}
    assert names == {fault.value for fault in RiskContinuityFault}
    assert not (names & EXCLUDING_RECOVERY_CAUSES)


def test_every_continuity_fault_has_a_recovery_cause():
    """The vocabularies are bounded and the runner maps one onto the other.

    ``RecoveryCause(fault.value)`` raises on import when a fault has no cause, so
    this is belt and braces -- and it is the assertion a reviewer looks for when
    a fault is added.
    """
    for fault in RiskContinuityFault:
        assert RecoveryCause(fault.value).value == fault.value


@pytest.mark.parametrize(
    ("load", "history"),
    [
        (
            RiskStateLoad.MISSING,
            risk_continuity.LogRiskHistory(
                records=9,
                statement=LogRiskStatement.STATE_HASH,
                statement_kind="DECISION",
                statement_seq=7,
                state_hash="sha256:" + "a" * 64,
            ),
        ),
        (
            RiskStateLoad.UNREADABLE,
            risk_continuity.LogRiskHistory(records=9),
        ),
        (
            RiskStateLoad.LEGACY,
            risk_continuity.LogRiskHistory(records=9),
        ),
        (
            RiskStateLoad.LOADED,
            risk_continuity.LogRiskHistory(
                records=9,
                statement=LogRiskStatement.STATE_HASH,
                statement_kind="DECISION",
                statement_seq=7,
                state_hash="sha256:" + "a" * 64,
            ),
        ),
    ],
    ids=("absent", "unreadable", "pre-schema", "mismatch"),
)
def test_a_continuity_halt_reason_is_the_same_on_every_host(load, history):
    """No host path in a halt reason, because a halt reason is HASHED.

    `RiskState.snapshot` carries `halt_reason`, so the reason is part of
    `risk.state_hash`. `RiskEngine._load_state` already refuses to put a path in
    one -- "a path or errno string would differ between hosts in the same
    semantic state" -- and R1-b's equity dispute names no path either. A reason
    that carried the state directory would make two hosts in the same semantic
    state hash differently, and a replay of a disputed campaign from another
    directory would diverge for a reason that is not a decision.

    Two directories that share nothing, one finding: the reasons must be equal
    character for character, and neither may quote either directory.
    """
    here = Path("/srv/chimera/campaign-one/state")
    there = Path("/var/lib/other/deep/nested/elsewhere")
    snapshot = _snapshot_of(halted=False)

    reasons = {
        str(root): assess_risk_continuity(
            load=load, snapshot=snapshot, state_dir=root, history=history
        ).halt_reason
        for root in (here, there)
    }

    assert len(set(reasons.values())) == 1, reasons
    reason = next(iter(reasons.values()))
    assert reason.startswith(RISK_CONTINUITY_PREFIX), "the fixture must really dispute"
    assert "risk.json" in reason, "and it still names the file an operator opens"
    for root in (here, there):
        assert str(root) not in reason


def test_the_continuity_hash_is_the_one_the_runner_writes(tmp_path):
    """One hash function, used to write the record and to read it back."""
    harness = campaign(tmp_path)
    engine = harness.runner.risk

    assert _risk_hash(engine) == risk_state_hash(engine.snapshot())
    assert risk_continuity.RISK_HASH_EXCLUDED == frozenset(
        {"order_times", "cooldown_until", "day"}
    )


def test_the_load_time_snapshot_is_not_moved_by_the_kill_switch(tmp_path):
    """Why the comparison uses the READ and not the engine a moment later.

    A switch an operator engaged while the process was down is mirrored into the
    state by the constructor and changes the hash. That is a legitimate move and
    reporting it as a continuity failure would halt a campaign for being
    stopped on purpose.
    """
    harness = campaign(tmp_path)
    config = harness.runner.config
    state_dir = harness.state_dir
    (state_dir / "KILL_SWITCH").write_text("stop\n", encoding="utf-8")

    resumed = restart(tmp_path, config)

    assert resumed.runner.state is RunnerState.HALT
    assert resumed.runner.risk.state.halt_reason == "kill_switch"
    assert not continuity_recoveries(state_dir)
    assert resumed.runner.risk.state.kill_switch
    assert not resumed.runner.risk.loaded_snapshot[
        "kill_switch"
    ], "the load-time snapshot is what the FILE said, before the switch was seen"


def test_the_status_command_reports_the_finding_without_starting(tmp_path):
    """An operator whose campaign will not start must not have to start it."""
    from tools import demo_run

    harness = campaign(tmp_path)
    config = harness.runner.config
    risk_json(harness.state_dir).unlink()
    resumed = build(tmp_path, days=(DAY,), config=config, start=False)

    status = demo_run._status(resumed.runner)

    assert status["risk_continuity"]["disputed"] is True
    assert status["risk_continuity"]["fault"] == RiskContinuityFault.RISK_STATE_ABSENT.value
    assert status["risk_continuity"]["risk_state_load"] == RiskStateLoad.MISSING.value
    assert resumed.runner.state is RunnerState.STARTUP, "status started nothing"


# ---------------------------------------------------------------------------
# 11. mutation witnesses
# ---------------------------------------------------------------------------
# Each installs a plausible half-implementation over the real one and asserts
# that a named witness above stops holding. A check that survives its own
# mutant is a check that was not deciding anything.
def _install(monkeypatch, replacement: Callable[..., Any]) -> None:
    """Put ``replacement`` where BOTH callers reach it.

    ``build_risk_engine`` and ``DemoRunner`` each import the function by name,
    so a mutant installed on one module only would leave the other doing the
    right thing -- and the test would then be witnessing the half that was not
    mutated.
    """
    monkeypatch.setattr(risk_wiring, "assess_risk_continuity", replacement)
    monkeypatch.setattr(runner_module, "assess_risk_continuity", replacement)


def _blind_to(**overrides: Any) -> Callable[..., Any]:
    """A mutant assessment that edits the history before the real one reads it."""
    real = assess_risk_continuity

    def mutant(*, load, snapshot, state_dir, history=None, **inputs):
        history = read_log_risk_history(state_dir) if history is None else history
        from dataclasses import replace

        return real(
            load=load,
            snapshot=snapshot,
            state_dir=state_dir,
            history=replace(history, **overrides),
            **inputs,
        )

    return mutant


def test_mutant_a_a_log_with_no_history_detection_loses_the_missing_file(
    tmp_path, monkeypatch
):
    """Mutation A: the log never says it holds records."""
    harness = campaign(tmp_path)
    config = harness.runner.config
    state_dir = harness.state_dir
    risk_json(state_dir).unlink()
    _install(monkeypatch, _blind_to(records=0))

    resumed = restart(tmp_path, config)

    assert not continuity_recoveries(state_dir), "the mutant must really be blind"
    assert resumed.runner.risk.state.equity == pytest.approx(float(CAPITAL)), (
        "and blind means the defect is back: capital seeded over a campaign's "
        "history. `test_a_missing_risk_state_with_a_verified_history_is_a_dispute` "
        "is the witness this kills"
    )


def test_mutant_b_treating_an_unreadable_state_as_a_first_start(tmp_path, monkeypatch):
    """Mutation B: ``UNREADABLE`` reported to the assessment as ``MISSING``.

    The witness killed is
    `test_an_unreadable_risk_state_with_a_history_is_recorded_and_preserved`:
    the cause stops naming the unreadable file.
    """
    harness = campaign(tmp_path)
    config = harness.runner.config
    state_dir = harness.state_dir
    risk_json(state_dir).write_text("{not json", encoding="utf-8")
    real = assess_risk_continuity

    def mutant(*, load, snapshot, state_dir, history=None, **inputs):
        if load is RiskStateLoad.UNREADABLE:
            load = RiskStateLoad.MISSING
        return real(
            load=load, snapshot=snapshot, state_dir=state_dir, history=history, **inputs
        )

    _install(monkeypatch, mutant)

    restart(tmp_path, config)

    assert causes(state_dir) == [RecoveryCause.RISK_STATE_ABSENT.value]
    assert causes(state_dir) != [RecoveryCause.RISK_STATE_UNREADABLE.value]


def test_mutant_c_suppressing_the_hash_comparison(tmp_path, monkeypatch):
    """Mutation C: the log never reports a ``STATE_HASH`` statement.

    Kills `test_a_risk_state_that_does_not_match_the_logs_hash_is_a_dispute`.
    """
    harness = campaign(tmp_path)
    config = harness.runner.config
    state_dir = harness.state_dir
    move_the_peak(state_dir)
    _install(monkeypatch, _blind_to(statement=LogRiskStatement.NONE, state_hash=""))

    resumed = restart(tmp_path, config)

    assert not continuity_recoveries(state_dir)
    assert resumed.runner.state is not RunnerState.HALT


def test_mutant_d_reading_the_statement_off_the_physically_last_record(tmp_path, monkeypatch):
    """Mutation D: "the newest record containing a state_hash anywhere" wins.

    That is the shortcut this module refuses, and it is wrong in exactly the
    case R1-c creates: R1-c's own RECOVERY record carries the DISPUTED hash
    inside its ``recovery`` block, so a scan by shape rather than by structure
    compares the file against itself on the next start.

    Kills `test_a_recorded_dispute_does_not_become_its_own_evidence`.

    Since PR #102's B5 remediation this RECOVERY record is ALSO masked by a
    second, independent defense: it is written by a SEALED run, and no record of
    a sealed run is a statement about ``risk.json`` at all. So the mutant is
    first shown to be stopped by that defense alone, and then exercised with
    that one rule put back to its pre-B5 form (a sealed run's HALT, RESUME and
    MOVED records ignored, its restated hashes read), to show the structural
    read is still load-bearing by itself.
    """
    harness = campaign(tmp_path)
    config = harness.runner.config
    state_dir = harness.state_dir
    move_the_peak(state_dir)
    restart(tmp_path, config)
    assert causes(state_dir) == [RecoveryCause.RISK_STATE_MISMATCH.value]

    real_statement = risk_continuity._statement_of

    def mutant(record):
        block = record.get("recovery", {})
        found = block.get("risk_continuity", {}).get("found_state_hash")
        if isinstance(found, str) and found:
            return LogRiskStatement.STATE_HASH, found
        return real_statement(record)

    monkeypatch.setattr(risk_continuity, "_statement_of", mutant)

    assert (
        restart(tmp_path, config).runner.state is RunnerState.HALT
    ), "the sealed-run rule alone keeps R1-c's own record from being a statement"

    monkeypatch.setattr(risk_continuity, "_statement_in_run", pre_b5_sealed_run_rule)

    again = restart(tmp_path, config)

    assert (
        again.runner.state is not RunnerState.HALT
    ), "the mutant lets the campaign run on the state its own dispute recorded"


def test_mutant_d2_ignoring_the_records_that_move_the_state(tmp_path, monkeypatch):
    """Mutation D again, the other way: the suppressing kinds are forgotten.

    With ``OPERATOR`` no longer counted as a statement, the newest hash-bearing
    record wins even though a flatten moved the state after it.

    Kills `test_the_last_risk_statement_is_not_the_last_record[flatten]`.
    """
    harness = campaign(tmp_path)
    config = harness.runner.config
    state_dir = harness.state_dir
    harness.runner.flatten("operator: reduce to flat for maintenance")
    monkeypatch.setattr(risk_continuity, "_MOVING_KINDS", frozenset())

    resumed = restart(tmp_path, config)

    assert resumed.runner.state is RunnerState.HALT
    assert causes(state_dir) == [RecoveryCause.RISK_STATE_MISMATCH.value]


def test_mutant_e_silently_clearing_a_logged_halt(tmp_path, monkeypatch):
    """Mutation E: a ``HALT`` statement stops being one.

    Kills `test_a_halt_cleared_behind_the_log_is_a_dispute`.
    """
    harness = campaign(tmp_path)
    halt_the_campaign(harness)
    config = harness.runner.config
    state_dir = harness.state_dir
    clear_the_halt_in_the_file(state_dir)
    _install(monkeypatch, _blind_to(statement=LogRiskStatement.MOVED))

    resumed = restart(tmp_path, config)

    assert not continuity_recoveries(state_dir)
    assert (
        resumed.runner.state is not RunnerState.HALT
    ), "the mutant restarts a halted campaign as healthy"


def test_mutant_f_writing_no_recovery_record(tmp_path, monkeypatch):
    """Mutation F: the campaign halts and says nothing in the log.

    Kills every ``causes(...)`` assertion above. The halt alone is not enough:
    canonical R1-c asks for a ``RECOVERY`` record, and a halt reason lives in one
    HALT record's free text where an audit reads the recovery ones.
    """
    harness = campaign(tmp_path)
    config = harness.runner.config
    state_dir = harness.state_dir
    risk_json(state_dir).unlink()
    monkeypatch.setattr(
        runner_module.DemoRunner,
        "_write_risk_continuity_recovery",
        lambda self: None,
    )

    resumed = restart(tmp_path, config)

    assert resumed.runner.state is RunnerState.HALT, "it still fails closed..."
    assert not continuity_recoveries(state_dir), "...and the evidence is gone"


def test_mutant_g_recording_the_finding_on_every_restart(tmp_path, monkeypatch):
    """Mutation G: idempotence dropped.

    Kills `test_repeated_restarts_record_the_finding_once`.
    """
    harness = campaign(tmp_path)
    config = harness.runner.config
    state_dir = harness.state_dir
    risk_json(state_dir).unlink()
    real = assess_risk_continuity

    def mutant(*, load, snapshot, state_dir, history=None, **inputs):
        from dataclasses import replace

        return replace(
            real(load=load, snapshot=snapshot, state_dir=state_dir, history=history, **inputs),
            already_recorded=False,
        )

    _install(monkeypatch, mutant)

    for _ in range(3):
        restart(tmp_path, config)

    assert len(continuity_recoveries(state_dir)) == 3


def test_mutant_h_the_r1b_unconditional_reseed(tmp_path, monkeypatch):
    """Mutation H: R1-b's defect restored, with R1-c watching.

    R1-b's own suite kills this on a state directory with no log. Here it is
    killed a second way: the seed must not be reachable AT ALL when the log
    holds records, whatever ``seed_or_reconcile_equity`` would have done.
    """

    def unconditional_reseed(engine, *, capital, state_dir) -> None:
        engine.update_equity(float(capital))

    harness = campaign(tmp_path)
    config = harness.runner.config
    state_dir = harness.state_dir
    risk_json(state_dir).unlink()
    monkeypatch.setattr(risk_wiring, "seed_or_reconcile_equity", unconditional_reseed)

    resumed = restart(tmp_path, config)

    assert (
        resumed.runner.risk.state.equity == 0.0
    ), "the dispute returns before the seed is reached, so the mutant never runs"
    assert not risk_json(state_dir).exists()
    assert causes(state_dir) == [RecoveryCause.RISK_STATE_ABSENT.value]


def test_the_seal_is_what_keeps_the_absence_absent(tmp_path, monkeypatch):
    """The pair for the seal: without it, halting creates the missing file.

    The mutant halts ordinarily instead of sealing, which writes an all-defaults
    halted document where there was nothing -- and the NEXT start then finds a
    ``LOADED`` state this process invented and reports a different finding about
    it.
    """
    harness = campaign(tmp_path)
    config = harness.runner.config
    state_dir = harness.state_dir
    risk_json(state_dir).unlink()
    monkeypatch.setattr(
        RiskEngine,
        "halt_for_continuity_dispute",
        lambda self, reason: RiskEngine.halt(self, reason),
    )

    restart(tmp_path, config)

    assert risk_json(state_dir).is_file(), "the mutant must really create it"
    assert json.loads(risk_json(state_dir).read_text(encoding="utf-8"))["halted"] is True
    second = restart(tmp_path, config)
    assert (
        second.runner.risk.load_outcome is RiskStateLoad.LOADED
    ), "and the absence is gone: the next start reads a state nothing recorded"


# ---------------------------------------------------------------------------
# PR #102 remediation: B1, the crash windows `_triage_log` cannot see
# ---------------------------------------------------------------------------
class ProcessKilled(BaseException):
    """A process death. Deliberately NOT an ``Exception``: ``DemoRunner._halt``
    catches those around its append, and nothing catches a real kill."""


def die_before_append(harness, kind: RecordKind, action: Callable[[], Any]) -> None:
    """Run ``action`` through the production path and kill it at its first ``kind`` append.

    Nothing is hand-edited: every file on disk is exactly what production
    persisted up to the line before the append. The day file the dead runner
    held open is then closed, as the OS closes a dead process's handles.
    """
    real = runner_module.DemoRunner._append

    def append(self, record_kind, minute_ns, payload):
        if record_kind is kind:
            raise ProcessKilled(f"killed before the {kind.value} append")
        return real(self, record_kind, minute_ns, payload)

    runner_module.DemoRunner._append = append
    try:
        with pytest.raises(ProcessKilled):
            action()
    finally:
        runner_module.DemoRunner._append = real
    if harness.runner._log is not None:
        harness.runner._log.close()


def next_minute(harness) -> int:
    """The minute the campaign would decide next."""
    return harness.runner.cursor.last_minute_processed + 60_000


def widen_the_basis(monkeypatch, *, from_minute: int, by: float) -> None:
    """Market data, not state: the perpetual closes ``by`` wider from one minute on.

    The fixture's hedged position is otherwise worth the same every minute, so
    this is what gives Aegis an ordinary mark-to-market move to persist. A few
    dollars is below anything the carry rule trades on, so the minute books no
    fill, fee or funding -- the case `_triage_log` has nothing to compare for.
    """
    real = SyntheticFeed.perp_close
    monkeypatch.setattr(
        SyntheticFeed,
        "perp_close",
        lambda self, index: round(
            real(self, index) + (by if index >= from_minute else 0.0), 2
        ),
    )


def startup_flags(state_dir: Path) -> list[Any]:
    """Each run's own ``STARTUP.risk_continuity_sealed``, in order."""
    return [
        record.get(RISK_CONTINUITY_SEALED_FIELD, "absent")
        for record in records(state_dir)
        if record["kind"] == RecordKind.STARTUP.value
    ]


def assert_proved_and_deferred(resumed, state_dir: Path, transition: str) -> dict[str, Any]:
    """The shared half of every proved crash window: recorded, deferred, not sealed."""
    verdict = resumed.runner.risk_continuity
    assert verdict.fault is RiskContinuityFault.RISK_STATE_MISMATCH, "the hash did move"
    assert verdict.crash_transition == transition
    assert not resumed.runner.risk.continuity_disputed, "proved, so not sealed"
    assert not (resumed.runner.halt_reason or "").startswith(RISK_CONTINUITY_PREFIX)
    assert startup_flags(state_dir)[-1] is False
    block = continuity_recoveries(state_dir)[-1]["recovery"]["risk_continuity"]
    assert block["fault"] == RiskContinuityFault.RISK_STATE_MISMATCH.value
    assert f"({transition}:" in block["detail"], "the record names the window"
    return block


def raise_in_the_rule(monkeypatch) -> str:
    """A real halt site: the actionable rule raises, and `tick` halts on it."""

    def evaluate(self, state, portfolio):
        raise RuleError("synthetic rule failure")

    monkeypatch.setattr(CarryRule, "evaluate", evaluate)
    return "rule_exception: R1_carry: synthetic rule failure"


def test_a_real_bare_halt_crash_is_proved_and_deferred(tmp_path, monkeypatch):
    """B1 shape 2, the bare halt: ``RiskEngine.halt`` persisted, the kill came
    before ``_halt``'s ``HALT`` append. Reproduced through a real halt site (a
    rule exception) rather than by editing ``risk.json``.
    """
    harness = campaign(tmp_path)
    config = harness.runner.config
    state_dir = harness.state_dir
    reason = raise_in_the_rule(monkeypatch)
    minute = next_minute(harness)
    die_before_append(harness, RecordKind.HALT, lambda: harness.tick(minute))
    assert json.loads(risk_json(state_dir).read_text(encoding="utf-8"))["halted"] is True

    resumed = restart(tmp_path, config)

    block = assert_proved_and_deferred(resumed, state_dir, "halt")
    assert block["halt_transition_explains_mismatch"] is True
    assert resumed.runner.state is RunnerState.HALT
    assert resumed.runner.halt_reason == reason, "the guard that really tripped"


@pytest.mark.parametrize("switch_removed_before_restart", [False, True])
def test_a_real_kill_switch_halt_crash_is_proved_and_deferred(
    tmp_path, switch_removed_before_restart
):
    """B1 shape 2, the kill switch. ``check_kill_switch`` moves THREE hashed
    fields -- the mirror, ``halted`` and ``halt_reason`` -- so the bare-halt
    revert alone never reproduced the log's hash and this window sealed for
    ever, with the switch left in place or not. The mirror is what persisted,
    so whether the file is still there at the restart does not enter the proof.
    """
    harness = campaign(tmp_path)
    config = harness.runner.config
    state_dir = harness.state_dir
    before = json.loads(risk_json(state_dir).read_text(encoding="utf-8"))
    switch = state_dir / "KILL_SWITCH"
    switch.write_text("stop\n", encoding="utf-8")
    minute = next_minute(harness)
    die_before_append(harness, RecordKind.HALT, lambda: harness.tick(minute))
    after = json.loads(risk_json(state_dir).read_text(encoding="utf-8"))
    moved = {k for k in after if k != "updated_at" and after[k] != before[k]}
    assert moved == {"kill_switch", "halted", "halt_reason"}, "the real window's shape"
    if switch_removed_before_restart:
        switch.unlink()

    resumed = restart(tmp_path, config)

    block = assert_proved_and_deferred(resumed, state_dir, "kill_switch_halt")
    assert block["halt_transition_explains_mismatch"] is True
    assert resumed.runner.halt_reason == "kill_switch"


def test_a_forged_kill_switch_reason_is_not_proved(tmp_path):
    """Negative control (constructed): the real kill-switch window, then the
    halt reason rewritten to one ``check_kill_switch`` never writes. Every
    other field is still the log's own prior state, and it must still seal.
    """
    harness = campaign(tmp_path)
    config = harness.runner.config
    state_dir = harness.state_dir
    (state_dir / "KILL_SWITCH").write_text("stop\n", encoding="utf-8")
    minute = next_minute(harness)
    die_before_append(harness, RecordKind.HALT, lambda: harness.tick(minute))
    document = json.loads(risk_json(state_dir).read_text(encoding="utf-8"))
    document["halt_reason"] = "kill_switch: something else"
    risk_json(state_dir).write_text(json.dumps(document), encoding="utf-8")

    resumed = restart(tmp_path, config)

    assert resumed.runner.risk.continuity_disputed
    assert resumed.runner.risk_continuity.crash_transition == ""
    assert startup_flags(state_dir)[-1] is True


def test_a_real_drawdown_halt_crash_is_proved_and_deferred(tmp_path, monkeypatch):
    """A2. The drawdown halt fires INSIDE ``update_equity``, so the window moves
    ``equity`` and ``daily_pnl`` as well as the halt, and the record it precedes
    is the minute's ``DECISION``, not a ``HALT``. The limit is the campaign's
    own, set between the drawdown the campaign already carries and the one the
    widened basis produces, so the breach is a real one on the real path.
    """
    widen_the_basis(monkeypatch, from_minute=MINUTES, by=5.0)
    config = campaign_config(tmp_path / "state", limits={"max_drawdown_pct": 0.0005})
    harness = build(tmp_path, days=(DAY,), config=config)
    harness.run(MINUTES)
    assert not harness.runner.risk.state.halted
    minute = next_minute(harness)
    die_before_append(harness, RecordKind.DECISION, lambda: harness.tick(minute))
    state_dir = harness.state_dir
    halted = json.loads(risk_json(state_dir).read_text(encoding="utf-8"))
    assert halted["halted"] and halted["halt_reason"].startswith("max drawdown breached")

    resumed = restart(tmp_path, config)

    block = assert_proved_and_deferred(resumed, state_dir, "equity_halt")
    assert block["halt_transition_explains_mismatch"] is True
    assert resumed.runner.halt_reason == halted["halt_reason"]


def test_a_real_intraday_equity_crash_is_proved_and_deferred(tmp_path, monkeypatch):
    """B1 shape 1, the ordinary form: ``_save_ledger()`` then ``update_equity``
    persisted a new mark, and the kill came before the ``DECISION`` append.

    The prior equity is NOT hidden behind the hash: the last ``DECISION``
    carries it in plaintext as ``ledger_effect.equity``. The proof reverts
    ``equity`` and ``daily_pnl`` to it, requires the FULL prior hash, replays the
    real ``update_equity``, and requires the ledger to hold the found equity.
    The campaign then re-decides the minute and the log catches up.
    """
    widen_the_basis(monkeypatch, from_minute=MINUTES, by=5.0)
    harness = campaign(tmp_path)
    config = harness.runner.config
    state_dir = harness.state_dir
    last_decision = [r for r in records(state_dir) if r["kind"] == "DECISION"][-1]
    minute = next_minute(harness)
    die_before_append(harness, RecordKind.DECISION, lambda: harness.tick(minute))
    found = json.loads(risk_json(state_dir).read_text(encoding="utf-8"))
    assert found["equity"] != float(last_decision["ledger_effect"]["equity"]), "it moved"

    resumed = restart(tmp_path, config)

    block = assert_proved_and_deferred(resumed, state_dir, "equity")
    assert block["halt_transition_explains_mismatch"] is False, "nothing halted"
    assert resumed.runner.state is RunnerState.READY
    resumed.tick(minute)
    history = read_log_risk_history(state_dir)
    on_disk = RiskState.from_dict(json.loads(risk_json(state_dir).read_text(encoding="utf-8")))
    assert history.state_hash == risk_state_hash(
        on_disk.snapshot()
    ), "the re-decided minute restates exactly the state on disk"


def test_a_real_funding_minute_crash_is_proved_and_deferred(tmp_path):
    """B1 shape 1 on a funding minute. The ``FUNDING`` record restates the hash
    WITHOUT moving Aegis's equity, so its own ``ledger_effect.equity`` is not
    the prior equity; the last ``DECISION``'s is, and that is what is used.
    """
    harness = build(tmp_path, days=(DAY,))
    harness.feed.write_settlements([DAY], hours=(1,))
    harness.runner.cursor._settlements = None
    harness.run(59)
    config = harness.runner.config
    state_dir = harness.state_dir
    minute = next_minute(harness)
    die_before_append(harness, RecordKind.DECISION, lambda: harness.tick(minute))
    funding = [r for r in records(state_dir) if r["kind"] == "FUNDING"]
    assert funding, "the settlement was booked and recorded before the kill"
    found = json.loads(risk_json(state_dir).read_text(encoding="utf-8"))
    assert found["equity"] == float(funding[-1]["ledger_effect"]["equity"])

    resumed = restart(tmp_path, config)

    assert_proved_and_deferred(resumed, state_dir, "equity")
    assert resumed.runner.state is RunnerState.READY


def test_a_real_day_roll_crash_stays_sealed_for_r1i(tmp_path):
    """PINNED, not fixed: the UTC day-roll window.

    ``update_equity`` on a new UTC day overwrites ``day_start_equity`` and
    ``daily_pnl``. The prior baseline is not restated by the record the crash
    preceded -- it is the equity of whichever write first touched the previous
    day, which the configured capital or a bounded scan of earlier records may
    reconstruct -- so proving this window means historical reconstruction of
    the campaign's equity history, replay-shaped logic beyond R1-c's one-step
    local inverse: canonical R1-i. It stays sealed, and this deterministic boundary is reached every
    UTC midnight (here on the 23:59 minute, which Aegis dates by its close).
    """
    harness = build(tmp_path, days=(DAY, NEXT_DAY))
    start = harness.first_minute_ms() + 1435 * 60_000
    harness.run(4, start=start)
    config = harness.runner.config
    state_dir = harness.state_dir
    before = json.loads(risk_json(state_dir).read_text(encoding="utf-8"))
    minute = next_minute(harness)
    die_before_append(harness, RecordKind.DECISION, lambda: harness.tick(minute))
    after = json.loads(risk_json(state_dir).read_text(encoding="utf-8"))
    assert after["day"] != before["day"], "reproduced: the day rolled in the window"
    assert after["day_start_equity"] != before["day_start_equity"]

    resumed = restart(tmp_path, config, days=(DAY, NEXT_DAY))

    assert resumed.runner.risk.continuity_disputed, "sealed: R1-i's to prove"
    assert resumed.runner.risk_continuity.crash_transition == ""
    assert causes(state_dir) == [RecoveryCause.RISK_STATE_MISMATCH.value]


def test_a_real_new_peak_crash_stays_sealed_for_r1i(tmp_path):
    """PINNED, not fixed: the new-peak window. ``update_equity`` above the old
    peak overwrites ``peak_equity``, whose prior value is the running maximum of
    every equity Aegis was ever given -- history again, so R1-i's. A large
    funding receipt is the fixture's way to lift the hedged equity past the
    seeded capital.
    """
    harness = build(tmp_path, days=(DAY,))
    harness.feed.write_settlements([DAY], hours=(1,), rates={(DAY, 1): "0.005"})
    harness.runner.cursor._settlements = None
    harness.run(59)
    config = harness.runner.config
    state_dir = harness.state_dir
    before = json.loads(risk_json(state_dir).read_text(encoding="utf-8"))
    minute = next_minute(harness)
    die_before_append(harness, RecordKind.DECISION, lambda: harness.tick(minute))
    after = json.loads(risk_json(state_dir).read_text(encoding="utf-8"))
    assert after["peak_equity"] > before["peak_equity"], "reproduced: a new peak"

    resumed = restart(tmp_path, config)

    assert resumed.runner.risk.continuity_disputed, "sealed: R1-i's to prove"
    assert resumed.runner.risk_continuity.crash_transition == ""


@pytest.mark.parametrize(
    "tamper",
    ["peak_equity", "daily_pnl", "equity_and_daily_pnl", "halted_by_drawdown"],
)
def test_the_equity_proof_refuses_a_file_that_is_not_the_crash(tmp_path, monkeypatch, tamper):
    """Negative controls (constructed after a REAL intra-day crash).

    ``peak_equity``: a foreign file carrying the ledger's own equity and the
    right ``daily_pnl`` -- the full prior hash catches it.
    ``daily_pnl``: the prior hash still matches (the candidate's ``daily_pnl`` is
    derived, not copied), so it is the forward replay of ``update_equity`` that
    catches it.
    ``equity_and_daily_pnl``: a consistent state whose equity is not the ledger's
    -- the ledger condition catches it.
    ``halted_by_drawdown``: a drawdown halt the campaign's limits never raise
    at this equity -- the forward replay catches it.
    """
    widen_the_basis(monkeypatch, from_minute=MINUTES, by=5.0)
    harness = campaign(tmp_path)
    config = harness.runner.config
    state_dir = harness.state_dir
    minute = next_minute(harness)
    die_before_append(harness, RecordKind.DECISION, lambda: harness.tick(minute))
    document = json.loads(risk_json(state_dir).read_text(encoding="utf-8"))
    if tamper == "peak_equity":
        document["peak_equity"] += 1.0
    elif tamper == "daily_pnl":
        document["daily_pnl"] += 1.0
    elif tamper == "equity_and_daily_pnl":
        document["equity"] += 1.0
        document["daily_pnl"] += 1.0
    else:
        document["halted"] = True
        document["halt_reason"] = "max drawdown breached: 9.00% >= 5.00%"
    risk_json(state_dir).write_text(json.dumps(document), encoding="utf-8")

    resumed = restart(tmp_path, config)

    assert resumed.runner.risk.continuity_disputed
    assert resumed.runner.risk_continuity.crash_transition == ""
    assert startup_flags(state_dir)[-1] is True


def test_the_halt_transition_proof_still_catches_a_swap(tmp_path):
    """Negative control (constructed): a halted file that ALSO differs
    somewhere else must still be reported, or the proof would excuse exactly
    the foreign/swapped state R1-c exists to catch.
    """
    harness = campaign(tmp_path)
    config = harness.runner.config
    state_dir = harness.state_dir
    move_the_peak(state_dir)
    document = json.loads(risk_json(state_dir).read_text(encoding="utf-8"))
    document["halted"] = True
    document["halt_reason"] = "max drawdown breached: 6.00% >= 5.00%"
    risk_json(state_dir).write_text(json.dumps(document), encoding="utf-8")

    resumed = restart(tmp_path, config)

    assert resumed.runner.state is RunnerState.HALT
    assert resumed.runner.risk.continuity_disputed
    assert causes(state_dir) == [RecoveryCause.RISK_STATE_MISMATCH.value]
    block = continuity_recoveries(state_dir)[0]["recovery"]["risk_continuity"]
    assert block["halt_transition_explains_mismatch"] is False


def test_a_crash_window_proof_needs_a_state_hash_statement(tmp_path):
    """The proofs are scoped to a ``STATE_HASH`` statement, stated directly.

    A ``HALTED`` or ``RUNNING`` statement already has its own rule (section 5);
    no crash-window proof may additionally fire for those.
    """
    harness = campaign(tmp_path)
    halt_the_campaign(harness)
    state_dir = harness.state_dir
    history = read_log_risk_history(state_dir)
    assert history.statement is LogRiskStatement.HALTED
    snapshot = json.loads(risk_json(state_dir).read_text(encoding="utf-8"))

    assert risk_continuity._crash_transition(snapshot, history) == ""


def advance_equity_only(state_dir: Path, delta: float) -> None:
    """Move ``equity`` and ``daily_pnl`` in ``risk.json`` alone, ledger untouched."""
    document = json.loads(risk_json(state_dir).read_text(encoding="utf-8"))
    document["equity"] = float(document["equity"]) + delta
    document["daily_pnl"] = float(document["daily_pnl"]) + delta
    risk_json(state_dir).write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def test_an_equity_move_the_ledger_never_saw_is_not_the_crash_window(tmp_path):
    """Negative control (constructed). This used to be the "shape 1" pin, and it
    did not model the crash: the real window persists the LEDGER first
    (``_save_ledger()`` then ``update_equity``), so a ``risk.json`` whose equity
    moved while the ledger did not is not that window. It stays sealed.
    """
    harness = campaign(tmp_path)
    config = harness.runner.config
    state_dir = harness.state_dir
    advance_equity_only(state_dir, 123.45)

    resumed = restart(tmp_path, config)

    assert resumed.runner.state is RunnerState.HALT
    assert resumed.runner.risk.continuity_disputed
    assert resumed.runner.risk_continuity.crash_transition == ""


# ---------------------------------------------------------------------------
# PR #102 remediation: B2, kill-switch ordering around a not-yet-triaged file
# ---------------------------------------------------------------------------
def test_a_kill_switch_does_not_create_the_file_a_missing_dispute_is_about(tmp_path):
    """B2. A ``risk.json`` gone from a campaign with history is a dispute
    (Case: ``RISK_STATE_ABSENT``); the constructor's own kill-switch look must
    not create the very file that dispute says is absent before R1-c can seal
    it, or the campaign's next restart finds a ``LOADED`` state this process
    invented.
    """
    harness = campaign(tmp_path, minutes=3)
    config = harness.runner.config
    state_dir = harness.state_dir
    risk_json(state_dir).unlink()
    (state_dir / "KILL_SWITCH").write_text("stop\n", encoding="utf-8")

    engine = risk_wiring.build_risk_engine(config, capital=CAPITAL, state_dir=state_dir)

    assert not risk_json(
        state_dir
    ).exists(), "the kill switch must not create the file the ABSENT dispute is about"
    assert engine.continuity_disputed
    assert engine.state.halt_reason.startswith(RISK_CONTINUITY_PREFIX), (
        "the continuity reason, not one 'kill_switch' persisted ahead of it and kept "
        "by RiskEngine.halt's first-reason rule"
    )


def test_a_kill_switch_does_not_overwrite_a_disputed_file_before_triage(tmp_path):
    """B2. A mismatched ``risk.json`` is deferred to the runner's own crash
    triage (``crash_could_explain``), so ``build_risk_engine`` must not let a
    present kill switch halt-and-persist over the disputed bytes before that
    triage has had a chance to run; a halted, kill-switch-mirrored copy of the
    disputed file is not the evidence an operator has to look at.
    """
    harness = campaign(tmp_path)
    config = harness.runner.config
    state_dir = harness.state_dir
    move_the_peak(state_dir)
    disputed_bytes = bytes_of(risk_json(state_dir))
    (state_dir / "KILL_SWITCH").write_text("stop\n", encoding="utf-8")

    engine = risk_wiring.build_risk_engine(config, capital=CAPITAL, state_dir=state_dir)

    assert bytes_of(risk_json(state_dir)) == disputed_bytes, (
        "the disputed bytes must survive a kill switch present at the same restart, "
        "before the runner's triage has run"
    )
    assert (
        not engine.continuity_disputed
    ), "build_risk_engine defers a MISMATCH; it does not seal one itself"


def test_the_runner_still_halts_on_a_kill_switch_once_triage_has_run(tmp_path):
    """The pair for both B2 cases: a kill switch beside an ORDINARY restart
    (no continuity dispute at all) still halts exactly as before -- the fix
    defers the check for a disputed, not-yet-triaged file; it does not remove
    it.
    """
    harness = campaign(tmp_path)
    config = harness.runner.config
    state_dir = harness.state_dir
    (state_dir / "KILL_SWITCH").write_text("stop\n", encoding="utf-8")

    resumed = restart(tmp_path, config)

    assert resumed.runner.state is RunnerState.HALT
    assert resumed.runner.halt_reason == "kill_switch"
    assert resumed.runner.risk.state.halt_reason == "kill_switch"


@pytest.mark.parametrize("already_halted_by", ["continuity_seal", "a_logged_halt"])
def test_a_kill_switch_at_start_keeps_the_first_halt_reason(
    tmp_path, monkeypatch, already_halted_by
):
    """B2, the runner half. ``start()`` consults the switch for its effect and
    reports the engine's FIRST reason, so an engine already halted -- sealed by
    a continuity dispute, or restored halted from a logged halt -- is not
    relabelled ``kill_switch``, and the sealed file's bytes are not touched.
    """
    harness = campaign(tmp_path)
    config = harness.runner.config
    state_dir = harness.state_dir
    if already_halted_by == "continuity_seal":
        move_the_peak(state_dir)
    else:
        raise_in_the_rule(monkeypatch)
        harness.tick(next_minute(harness))
        assert harness.runner.state is RunnerState.HALT
    first = json.loads(risk_json(state_dir).read_text(encoding="utf-8"))["halt_reason"]
    disputed_bytes = bytes_of(risk_json(state_dir))
    (state_dir / "KILL_SWITCH").write_text("stop\n", encoding="utf-8")

    resumed = restart(tmp_path, config)

    assert resumed.runner.state is RunnerState.HALT
    if already_halted_by == "continuity_seal":
        assert resumed.runner.halt_reason.startswith(RISK_CONTINUITY_PREFIX)
        assert bytes_of(risk_json(state_dir)) == disputed_bytes
    else:
        assert resumed.runner.halt_reason == first
        assert first.startswith("rule_exception:")


# ---------------------------------------------------------------------------
# PR #102 remediation: B3, which records a SEALED run wrote (per-run provenance)
# ---------------------------------------------------------------------------
def test_a_flatten_during_a_continuity_halt_does_not_clear_the_dispute(tmp_path):
    """B3 (a). A ``flatten`` is permitted while halted, including while sealed by
    a continuity dispute, and its ``OPERATOR`` record must not read on the
    next restart as ``MOVED`` -- which makes no comparable claim -- and
    silently clear a dispute nothing has actually repaired. The sealed run says
    so in its own ``STARTUP``.
    """
    harness = campaign(tmp_path)
    config = harness.runner.config
    state_dir = harness.state_dir
    move_the_peak(state_dir)
    disputed_bytes = bytes_of(risk_json(state_dir))

    disputed = restart(tmp_path, config)
    assert disputed.runner.state is RunnerState.HALT
    assert causes(state_dir) == [RecoveryCause.RISK_STATE_MISMATCH.value]
    assert startup_flags(state_dir) == [False, True]
    disputed.runner.flatten("operator: reduce to flat while investigating")
    assert (
        bytes_of(risk_json(state_dir)) == disputed_bytes
    ), "the seal must still refuse the write a flatten would otherwise make"
    operator_records = [r for r in records(state_dir) if r["kind"] == "OPERATOR"]
    assert operator_records, "the flatten must really have been recorded"

    again = restart(tmp_path, config)

    assert (
        again.runner.state is RunnerState.HALT
    ), "an OPERATOR record written while sealed must not clear the dispute"
    assert again.runner.risk_continuity.disputed
    assert read_log_risk_history(state_dir).statement is LogRiskStatement.STATE_HASH, (
        "the OPERATOR record written while sealed must not read as a statement about "
        "the file -- the DECISION before it must still stand, not be shadowed by a "
        "MOVED that lets rule 5 check nothing"
    )


@pytest.mark.parametrize("damage", ["mismatch", "absent"])
def test_a_repaired_campaign_may_flatten_and_restart(tmp_path, damage):
    """B3 (b), the reviewed P3 regression. Dispute, sealed run, the operator
    restores the exact pre-dispute bytes, a new run starts UNSEALED and READY,
    flattens before any minute restates the hash, and restarts. That flatten
    moved ``risk.json`` legitimately; reading it as a sealed run's record (as
    the "from the RECOVERY until the next hash" rule did) sealed it again.

    ``absent`` also pins that a repair is possible at all after ``ABSENT``: the
    sealed run's own HALT must not become a statement once the file is back.
    """
    harness = campaign(tmp_path)
    config = harness.runner.config
    state_dir = harness.state_dir
    good = bytes_of(risk_json(state_dir))
    if damage == "absent":
        risk_json(state_dir).unlink()
    else:
        move_the_peak(state_dir)
    restart(tmp_path, config)
    risk_json(state_dir).write_bytes(good)

    repaired = restart(tmp_path, config)
    assert repaired.runner.state is RunnerState.READY
    repaired.runner.flatten("operator: flat after the repair")
    again = restart(tmp_path, config)

    assert again.runner.state is RunnerState.READY
    assert not again.runner.risk_continuity.disputed
    assert len(continuity_recoveries(state_dir)) == 1, "no second, bogus finding"
    assert startup_flags(state_dir) == [False, True, False, False]


def test_a_proved_crash_then_flatten_and_resume_restarts_healthy(tmp_path, monkeypatch):
    """B3 (c), the reviewed P4 regression. A proved bare-halt crash writes an
    R1-c RECOVERY record WITHOUT sealing anything, and its run's STARTUP says
    ``False``. The flatten and resume that follow are that unsealed run's own,
    real transitions and must stay statements.
    """
    harness = campaign(tmp_path)
    config = harness.runner.config
    state_dir = harness.state_dir
    raise_in_the_rule(monkeypatch)
    minute = next_minute(harness)
    die_before_append(harness, RecordKind.HALT, lambda: harness.tick(minute))
    deferred = restart(tmp_path, config)
    assert deferred.runner.risk_continuity.crash_transition == "halt"
    deferred.runner.flatten("operator: flat")
    deferred.runner.resume("operator: checked the rule")

    again = restart(tmp_path, config)

    assert again.runner.state is RunnerState.READY
    assert not again.runner.risk_continuity.disputed
    assert len(continuity_recoveries(state_dir)) == 1
    assert startup_flags(state_dir) == [False, False, False]


def test_repeated_sealed_restarts_each_say_sealed_and_record_once(tmp_path):
    """B3 (d). Every sealed run states ``True`` for itself, and the finding is
    still recorded once: per-run provenance does not change idempotence.
    """
    harness = campaign(tmp_path)
    config = harness.runner.config
    state_dir = harness.state_dir
    risk_json(state_dir).unlink()

    for _ in range(3):
        restart(tmp_path, config)

    assert startup_flags(state_dir) == [False, True, True, True]
    assert causes(state_dir) == [RecoveryCause.RISK_STATE_ABSENT.value]


def test_a_sealed_run_does_not_contaminate_the_repaired_run_after_it(tmp_path):
    """B3 (e). After a sealed run, the repaired run's own HALT (a real kill
    switch, persisted) is a statement again: the next STARTUP starts a new run
    with its own value, and nothing sticky carries the seal across.
    """
    harness = campaign(tmp_path)
    config = harness.runner.config
    state_dir = harness.state_dir
    good = bytes_of(risk_json(state_dir))
    risk_json(state_dir).unlink()
    restart(tmp_path, config)
    risk_json(state_dir).write_bytes(good)
    repaired = restart(tmp_path, config)
    assert repaired.runner.state is RunnerState.READY
    halt_the_campaign(repaired)

    history = read_log_risk_history(state_dir)
    again = restart(tmp_path, config)

    assert history.statement is LogRiskStatement.HALTED, "the unsealed run's HALT stands"
    assert not again.runner.risk_continuity.disputed
    assert again.runner.risk_continuity.crash_transition == ""
    assert startup_flags(state_dir) == [False, True, False, False]


def test_a_startup_without_the_field_is_read_as_unsealed(tmp_path, monkeypatch):
    """B3 (f). Logs written before the field existed have no key. For a log
    from a pre-PR-#102 ``main`` build that is right: ``main`` has no
    ``halt_for_continuity_dispute`` at all, so absent is ``False`` and its HALT
    records keep main's meaning. (Intermediate PR #102 builds ``dded833`` to
    ``471c1b7`` could seal without writing the field; that compatibility
    condition is stated in the runbook for the owner to attest, and is not
    something this test, or the code, can establish.) Driven by stripping the key from a real
    runner's STARTUP payload, which is what such a build wrote.
    """
    real = runner_module.DemoRunner._append

    def append(self, kind, minute_ns, payload):
        if kind is RecordKind.STARTUP:
            payload = {k: v for k, v in payload.items() if k != RISK_CONTINUITY_SEALED_FIELD}
        return real(self, kind, minute_ns, payload)

    monkeypatch.setattr(runner_module.DemoRunner, "_append", append)
    harness = campaign(tmp_path)
    halt_the_campaign(harness)
    state_dir = harness.state_dir
    clear_the_halt_in_the_file(state_dir)

    history = read_log_risk_history(state_dir)
    again = restart(tmp_path, harness.runner.config)

    assert startup_flags(state_dir)[0] == "absent"
    assert history.statement is LogRiskStatement.HALTED, "honoured, as on main"
    assert again.runner.risk_continuity.fault is RiskContinuityFault.RISK_STATE_MISMATCH


def test_a_crash_before_the_startup_record_leaves_no_stale_marker(tmp_path):
    """B3 (g). A start that adjudicated a dispute and died before its STARTUP
    append left nothing: no marker, no RECOVERY, and the file untouched. The
    next start reaches the same finding from scratch.
    """
    harness = campaign(tmp_path)
    config = harness.runner.config
    state_dir = harness.state_dir
    risk_json(state_dir).unlink()
    committed = records(state_dir)
    tail = max(int(r["runner_now_ns"]) for r in committed)
    dying = build(tmp_path, days=(DAY,), config=config, start=False)
    die_before_append(
        dying, RecordKind.STARTUP, lambda: dying.runner.start(now_ns=tail + 60_000_000_000)
    )
    assert records(state_dir) == committed
    assert not risk_json(state_dir).exists()

    again = restart(tmp_path, config)

    assert again.runner.risk.continuity_disputed
    assert startup_flags(state_dir) == [False, True]
    assert causes(state_dir) == [RecoveryCause.RISK_STATE_ABSENT.value]


# ---------------------------------------------------------------------------
# PR #102 remediation 3, B4: nothing writes the questioned file before the verdict
# ---------------------------------------------------------------------------
# `assess_risk_continuity` defers one finding, a crash-could-explain
# RISK_STATE_MISMATCH, to the runner. `build_risk_engine` used to run R1-b's
# reconciliation in the meantime, and a stale or foreign file whose equity is
# not the ledger's made R1-b halt on `equity_dispute:` and PERSIST that halt over
# the disputed bytes -- before the runner had decided anything.
def seed_or_reconcile_calls(monkeypatch) -> list[tuple[str, bool]]:
    """Record who ran R1-b's reconciliation, and whether R1-c still held the engine.

    Both call sites are wrapped -- `build_risk_engine`'s and `DemoRunner.start`'s
    -- and each still runs the real function, so the campaign is unchanged.
    """
    calls: list[tuple[str, bool]] = []
    real = risk_wiring.seed_or_reconcile_equity

    def spy(site: str) -> Callable[..., None]:
        def wrapped(engine, *, capital, state_dir):
            calls.append((site, engine.continuity_pending))
            real(engine, capital=capital, state_dir=state_dir)

        return wrapped

    monkeypatch.setattr(risk_wiring, "seed_or_reconcile_equity", spy("build_risk_engine"))
    monkeypatch.setattr(runner_module, "seed_or_reconcile_equity", spy("DemoRunner.start"))
    return calls


def foreign_equity_campaign(tmp_path: Path):
    """A real campaign, then a current-schema ``risk.json`` whose equity is not
    the ledger's: the file R1-b's reconciliation would dispute, and R1-c too."""
    harness = campaign(tmp_path)
    state_dir = harness.state_dir
    advance_equity_only(state_dir, 777.0)
    disputed = bytes_of(risk_json(state_dir))
    assert disputed is not None
    accounted = risk_wiring.ledger_equity(state_dir, capital=CAPITAL)
    assert accounted is not None
    assert float(accounted) != json.loads(disputed)["equity"], "R1-b would halt on it"
    return harness, disputed


def test_b4_r1b_does_not_write_over_a_mismatch_before_the_verdict(tmp_path, monkeypatch):
    """B4-1. The engine is HELD, not reconciled; the finding stands; the bytes
    are the ones that were found; the halt is R1-c's; one RECOVERY, keyed on the
    found file's own hash."""
    harness, disputed = foreign_equity_campaign(tmp_path)
    config = harness.runner.config
    state_dir = harness.state_dir
    found = risk_state_hash(RiskState.from_dict(json.loads(disputed)).snapshot())

    engine = risk_wiring.build_risk_engine(config, capital=CAPITAL, state_dir=state_dir)
    assert engine.continuity_pending and not engine.continuity_disputed
    assert not engine.halted, "held, not halted: the verdict is not in yet"
    assert bytes_of(risk_json(state_dir)) == disputed, "and nothing reconciled over it"

    calls = seed_or_reconcile_calls(monkeypatch)
    resumed = restart(tmp_path, config)

    assert calls == [], "the finding stood, so R1-b is never asked about the wrong file"
    assert resumed.runner.state is RunnerState.HALT
    assert resumed.runner.halt_reason.startswith(RISK_CONTINUITY_PREFIX)
    assert resumed.runner.risk.state.halt_reason.startswith(RISK_CONTINUITY_PREFIX)
    assert resumed.runner.risk.continuity_disputed
    assert bytes_of(risk_json(state_dir)) == disputed, "byte-identical"
    assert causes(state_dir) == [RecoveryCause.RISK_STATE_MISMATCH.value]
    block = continuity_recoveries(state_dir)[0]["recovery"]["risk_continuity"]
    assert block["found_state_hash"] == found, "the evidence the runner found, not made"
    halts = [r for r in records(state_dir) if r["kind"] == RecordKind.HALT.value]
    assert [
        h["veto_or_rejection"]["detail"][: len(RISK_CONTINUITY_PREFIX)] for h in halts
    ] == [RISK_CONTINUITY_PREFIX], "no equity_dispute halt anywhere in the log"
    assert startup_flags(state_dir)[-1] is True


def test_b4_repeated_restarts_find_the_same_file_and_record_it_once(tmp_path):
    """B4-2. Each restart sees the file it was left, because no start wrote it:
    the same ``found_state_hash`` every time and one RECOVERY for the incident.
    Before the fix the first restart's R1-b halt changed the hash, and the
    second restart recorded the "new" file as a second finding."""
    harness, disputed = foreign_equity_campaign(tmp_path)
    config = harness.runner.config
    state_dir = harness.state_dir

    seen = []
    for _ in range(3):
        resumed = restart(tmp_path, config)
        assert resumed.runner.halt_reason.startswith(RISK_CONTINUITY_PREFIX)
        assert bytes_of(risk_json(state_dir)) == disputed
        seen.append(resumed.runner.risk_continuity.identity)

    assert len(set(seen)) == 1, seen
    assert causes(state_dir) == [RecoveryCause.RISK_STATE_MISMATCH.value]
    assert startup_flags(state_dir) == [False, True, True, True]


def crash_before_a_paid_settlement_is_recorded(tmp_path: Path):
    """A real kill between a PAID funding settlement's writes and its ``FUNDING``
    record. Nothing is hand-edited.

    ``DemoRunner._settle_funding`` books the settlement into the ledger (which
    marks it) and reports it to Aegis (``note_funding_settlement``, which moves
    the hashed adverse streak) before it appends the record that restates the
    hash. So the kill leaves all three at once: a ``RISK_STATE_MISMATCH`` no
    Aegis-only proof explains (the funding-streak window is unproved), a
    section 9.3 ``LOG_BEHIND_STATE`` triage that does explain it, and a ledger
    equity Aegis never received -- the disagreement R1-b exists to catch.
    """
    harness = build(tmp_path, days=(DAY,))
    harness.feed.write_settlements([DAY], hours=(1,), rate="-0.0001")
    harness.runner.cursor._settlements = None
    harness.run(59)
    before = json.loads(risk_json(harness.state_dir).read_text(encoding="utf-8"))
    minute = next_minute(harness)
    die_before_append(harness, RecordKind.FUNDING, lambda: harness.tick(minute))
    after = json.loads(risk_json(harness.state_dir).read_text(encoding="utf-8"))
    moved = {k for k in after if k != "updated_at" and after[k] != before[k]}
    assert moved == {"funding_adverse_streak"}, "the real window's shape"
    return harness


def test_b4_a_real_crash_mismatch_is_reconciled_after_the_verdict(tmp_path, monkeypatch):
    """B4-3. The crash explains the mismatch, so the file is released -- and R1-b
    runs THEN, exactly once, on the released engine, and still disputes the
    equity the crash left behind. The operator's clearing path works and the
    campaign runs again."""
    harness = crash_before_a_paid_settlement_is_recorded(tmp_path)
    config = harness.runner.config
    state_dir = harness.state_dir
    found = json.loads(risk_json(state_dir).read_text(encoding="utf-8"))
    accounted = risk_wiring.ledger_equity(state_dir, capital=CAPITAL)
    assert accounted is not None and float(accounted) != found["equity"]

    calls = seed_or_reconcile_calls(monkeypatch)
    resumed = restart(tmp_path, config)

    verdict = resumed.runner.risk_continuity
    assert verdict.fault is RiskContinuityFault.RISK_STATE_MISMATCH
    assert verdict.crash_transition == "", "not an Aegis-only window"
    assert RecoveryCause.LOG_BEHIND_STATE.value in [
        r["recovery"]["cause"] for r in records(state_dir) if r["kind"] == "RECOVERY"
    ], "section 9.3's triage is what explains it"
    assert not resumed.runner.risk.continuity_disputed and not continuity_recoveries(state_dir)
    assert calls == [("DemoRunner.start", False)], "R1-b ran once, after the release"
    assert resumed.runner.state is RunnerState.HALT
    assert resumed.runner.halt_reason.startswith(risk_wiring.EQUITY_DISPUTE_PREFIX)
    assert json.loads(risk_json(state_dir).read_text(encoding="utf-8"))["halt_reason"] == (
        resumed.runner.halt_reason
    ), "and, the verdict being in, it may persist its halt"

    settling = restart(tmp_path, config)
    settling.runner.resolve_equity("operator: crash at the settlement; the ledger is right")
    assert settling.runner.state is RunnerState.READY
    assert settling.runner.risk.state.equity == float(accounted)

    again = restart(tmp_path, config)
    assert again.runner.state is RunnerState.READY, again.runner.halt_reason
    assert not again.runner.risk_continuity.disputed


def test_b4_skipping_the_post_verdict_reconciliation_would_be_caught(tmp_path, monkeypatch):
    """Mutation witness for B4-3: a runner that released the hold and did NOT
    run R1-b would start this campaign READY on an equity its ledger does not
    hold. The test above fails against it."""
    harness = crash_before_a_paid_settlement_is_recorded(tmp_path)
    monkeypatch.setattr(runner_module, "seed_or_reconcile_equity", lambda *a, **k: None)

    resumed = restart(tmp_path, harness.runner.config)

    assert resumed.runner.state is RunnerState.READY, "the dispute went unnoticed"


def test_b4_an_ordinary_restart_still_gets_r1bs_equity_dispute(tmp_path, monkeypatch):
    """B4-4. The control: a loaded restart with NO R1-c finding and a genuine
    equity disagreement (R1-b's own crash window, on a minute that traded
    nothing, so ``risk.json`` did not move). R1-b runs where it always did --
    inside ``build_risk_engine`` -- and halts and persists exactly as before."""
    widen_the_basis(monkeypatch, from_minute=MINUTES, by=5.0)
    harness = campaign(tmp_path)
    config = harness.runner.config
    state_dir = harness.state_dir
    unmoved = risk_state_hash(
        RiskState.from_dict(json.loads(risk_json(state_dir).read_text("utf-8"))).snapshot()
    )
    real = RiskEngine.update_equity

    def killed(self, *args, **kwargs):
        raise ProcessKilled("killed in update_equity")

    monkeypatch.setattr(RiskEngine, "update_equity", killed)
    with pytest.raises(ProcessKilled):
        harness.tick(next_minute(harness))
    monkeypatch.setattr(RiskEngine, "update_equity", real)
    harness.runner._log.close()
    on_disk = RiskState.from_dict(json.loads(risk_json(state_dir).read_text("utf-8")))
    assert risk_state_hash(on_disk.snapshot()) == unmoved, "risk.json did not move"

    calls = seed_or_reconcile_calls(monkeypatch)
    resumed = restart(tmp_path, config)

    assert not resumed.runner.risk_continuity.disputed
    assert calls == [("build_risk_engine", False)], "where, and as, it always ran"
    assert resumed.runner.state is RunnerState.HALT
    assert resumed.runner.halt_reason.startswith(risk_wiring.EQUITY_DISPUTE_PREFIX)
    persisted = json.loads(risk_json(state_dir).read_text(encoding="utf-8"))
    assert persisted["halt_reason"].startswith(risk_wiring.EQUITY_DISPUTE_PREFIX)
    assert not continuity_recoveries(state_dir)
    assert startup_flags(state_dir)[-1] is False


def test_b4_resolve_equity_refuses_a_sealed_engine_through_the_cli(tmp_path, capsys):
    """B4-5. ``resolve --equity`` against an unresolved continuity seal changes
    nothing and says so: exit REFUSED, no READY on stdout, no OPERATOR record,
    the file untouched, and the next start still disputes the same finding.
    Before the fix the seal carried R1-b's ``equity_dispute:`` reason, so the
    command adopted the equity in memory, wrote an OPERATOR record, printed
    READY -- and the refused write left the next start disputing again."""
    harness, disputed = foreign_equity_campaign(tmp_path)
    state_dir = harness.state_dir
    config_path = written_config(tmp_path, harness)
    argv = ["--config", str(config_path), "--root", str(harness.root), "--profile", "TEST"]

    assert demo_run.main(argv + ["run"]) == demo_run.EXIT_HALTED
    assert json.loads(capsys.readouterr().out)["reason"].startswith(RISK_CONTINUITY_PREFIX)
    committed = records(state_dir)

    code = demo_run.main(argv + ["resolve", "--equity", "--note", "the ledger is right"])

    out = capsys.readouterr()
    assert code == demo_run.EXIT_REFUSED
    assert out.out == "", "nothing claims the dispute was settled"
    assert "sealed by an R1-c continuity dispute" in out.err
    assert bytes_of(risk_json(state_dir)) == disputed
    appended = records(state_dir)[len(committed) :]
    assert [r["kind"] for r in appended if r["kind"] == RecordKind.OPERATOR.value] == []
    assert demo_run.main(argv + ["run"]) == demo_run.EXIT_HALTED
    assert json.loads(capsys.readouterr().out)["reason"].startswith(RISK_CONTINUITY_PREFIX)
    assert causes(state_dir) == [RecoveryCause.RISK_STATE_MISMATCH.value]


def test_b4_resolve_equity_refuses_a_seal_that_kept_an_equity_dispute_reason(tmp_path):
    """B4-5, the other door. A foreign file that was itself halted on
    ``equity_dispute:`` keeps that reason under the seal (a halt keeps its
    first reason), so the reason alone looks settleable. The runner refuses
    with its own error -- not the engine's exception -- and changes nothing."""
    harness = campaign(tmp_path)
    config = harness.runner.config
    state_dir = harness.state_dir
    move_the_peak(state_dir)
    document = json.loads(risk_json(state_dir).read_text(encoding="utf-8"))
    document["halted"] = True
    document["halt_reason"] = f"{risk_wiring.EQUITY_DISPUTE_PREFIX} a stale copy's own halt"
    risk_json(state_dir).write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    disputed = bytes_of(risk_json(state_dir))

    sealed = restart(tmp_path, config)
    assert sealed.runner.risk.continuity_disputed
    assert risk_wiring.is_equity_reconciliation_halt(sealed.runner.risk.state.halt_reason)
    before = sealed.runner.risk.snapshot()
    committed = records(state_dir)

    with pytest.raises(RunnerError, match="sealed by an R1-c continuity dispute"):
        sealed.runner.resolve_equity("operator: the ledger is right")

    assert sealed.runner.risk.snapshot() == before
    assert sealed.runner.state is RunnerState.HALT
    assert records(state_dir) == committed
    assert bytes_of(risk_json(state_dir)) == disputed


# ---------------------------------------------------------------------------
# PR #102 remediation 3, B5: a sealed engine cannot move, even in memory
# ---------------------------------------------------------------------------
def loaded_engine(tmp_path: Path) -> tuple[RiskEngine, Path]:
    """A plain Aegis over a state file it wrote itself, reloaded: ``LOADED``,
    running, with an open position and a live day -- so every mutation below has
    something real to move."""
    state_path = tmp_path / "aegis" / "risk.json"
    writer = RiskEngine(RiskLimits(), state_path=state_path)
    writer.update_equity(1_000_000.0)
    writer.set_position_exposure("BTC/USDT:USDT", 100_000.0)
    engine = RiskEngine(
        RiskLimits(), state_path=state_path, kill_switch_path=tmp_path / "KILL"
    )
    assert engine.load_outcome is RiskStateLoad.LOADED and not engine.halted
    return engine, state_path


#: Every public RiskEngine method that changes ``RiskState``, and one call of it
#: that really would. ``halt`` and ``check_kill_switch`` are separate below.
OBSERVATIONS: dict[str, Callable[[RiskEngine], Any]] = {
    "update_equity": lambda e: e.update_equity(1_000_500.0),
    "record_order": lambda e: e.record_order(),
    "record_trade_result": lambda e: e.record_trade_result(-1.0),
    "set_position_exposure": lambda e: e.set_position_exposure("ETH/USDT:USDT", 5.0),
    "close_position": lambda e: e.close_position("BTC/USDT:USDT"),
    "note_feed": lambda e: e.note_feed(0, 10**18),
    "note_reconciliation": lambda e: e.note_reconciliation("BTC/USDT", "disputed"),
    "note_funding_settlement": lambda e: e.note_funding_settlement(
        "BTC/USDT:USDT", "LONG", 0.001
    ),
}
OPERATOR_DECISIONS: dict[str, Callable[[RiskEngine], Any]] = {
    "resume": lambda e: e.resume(),
    "adopt_reconciled_equity": lambda e: e.adopt_reconciled_equity(
        1_000_500.0, clearing=e.state.halt_reason, note="operator: adopt"
    ),
    "adopt_after_unreadable": lambda e: e.adopt_after_unreadable("operator: adopt"),
}


@pytest.mark.parametrize("name", sorted(OBSERVATIONS))
def test_b5_the_mutation_list_is_not_vacuous(tmp_path, name):
    """B5-5 (engine level), the control for the two tests below: on an ordinary
    engine every listed call really moves the state and really persists it."""
    engine, state_path = loaded_engine(tmp_path)
    before, disk = engine.snapshot(), bytes_of(state_path)

    OBSERVATIONS[name](engine)

    assert engine.snapshot() != before
    assert bytes_of(state_path) != disk


@pytest.mark.parametrize("name", sorted(OBSERVATIONS))
def test_b5_a_sealed_engine_does_not_move_on_an_observation(tmp_path, name):
    """Every observation is refused BEFORE it moves anything, and without an
    exception: the flatten and reconstruction paths that call these must keep
    working while the campaign is halted. Snapshot and bytes both unchanged."""
    engine, state_path = loaded_engine(tmp_path)
    engine.halt_for_continuity_dispute(f"{RISK_CONTINUITY_PREFIX} test")
    sealed, disk = engine.snapshot(), bytes_of(state_path)

    OBSERVATIONS[name](engine)

    assert engine.snapshot() == sealed
    assert bytes_of(state_path) == disk


@pytest.mark.parametrize("name", sorted(OBSERVATIONS) + ["halt"])
def test_b5_a_held_engine_refuses_every_mutation_loudly(tmp_path, name):
    """B4's hold: while R1-c's verdict is pending every mutation RAISES -- an
    ordering defect is made loud -- and nothing moves in memory or on disk."""
    engine, state_path = loaded_engine(tmp_path)
    engine.hold_for_continuity_adjudication()
    held, disk = engine.snapshot(), bytes_of(state_path)
    call = OBSERVATIONS.get(name, lambda e: e.halt("some guard"))

    with pytest.raises(RiskViolation, match="R1-c has not yet decided"):
        call(engine)

    assert engine.snapshot() == held
    assert bytes_of(state_path) == disk


@pytest.mark.parametrize("hold", ["sealed", "pending"])
@pytest.mark.parametrize("name", sorted(OPERATOR_DECISIONS))
def test_b5_operator_decisions_are_refused_before_anything_moves(tmp_path, name, hold):
    """B5-1 / B5-2. ``resume``, ``adopt_reconciled_equity`` and
    ``adopt_after_unreadable`` raise on a sealed (or held) engine, before any
    field moves and before any file is moved aside."""
    engine, state_path = loaded_engine(tmp_path)
    if hold == "sealed":
        engine.halt_for_continuity_dispute(f"{risk_wiring.EQUITY_DISPUTE_PREFIX} stale")
    else:
        engine.hold_for_continuity_adjudication()
    before, disk = engine.snapshot(), bytes_of(state_path)

    with pytest.raises(RiskViolation, match="R1-c"):
        OPERATOR_DECISIONS[name](engine)

    assert engine.snapshot() == before
    assert bytes_of(state_path) == disk


def test_b5_ordinary_resume_and_adoption_are_unchanged(tmp_path):
    """B5-5. The same two decisions on an unsealed engine move the state and
    persist it, exactly as before."""
    engine, state_path = loaded_engine(tmp_path)
    reason = f"{risk_wiring.EQUITY_DISPUTE_PREFIX} the two files disagree"
    engine.halt(reason)
    engine.adopt_reconciled_equity(1_000_500.0, clearing=reason, note="operator: ledger")
    assert not engine.halted and engine.state.equity == 1_000_500.0
    assert json.loads(state_path.read_text("utf-8"))["equity"] == 1_000_500.0

    engine.halt("some guard")
    engine.resume()
    assert not engine.halted
    assert json.loads(state_path.read_text("utf-8"))["halted"] is False


@pytest.mark.parametrize("hold", ["sealed", "pending"])
def test_b5_the_kill_switch_is_only_looked_at_while_held(tmp_path, hold):
    """``check_kill_switch`` still reports an engaged switch, and moves neither
    the hashed mirror nor the halt. A sealed engine is already halted; a held
    one is waiting for a verdict nothing may pre-empt."""
    engine, state_path = loaded_engine(tmp_path)
    if hold == "sealed":
        engine.halt_for_continuity_dispute(f"{RISK_CONTINUITY_PREFIX} test")
    else:
        engine.hold_for_continuity_adjudication()
    before, disk = engine.snapshot(), bytes_of(state_path)
    (tmp_path / "KILL").write_text("stop\n", encoding="utf-8")

    assert engine.check_kill_switch() is True

    assert engine.snapshot() == before
    assert bytes_of(state_path) == disk


def test_b5_a_held_engine_approves_nothing(tmp_path):
    """The hold is not a halt, so the entry gate refuses on it by name."""
    engine, _ = loaded_engine(tmp_path)
    assert engine.evaluate_entry("ETH/USDT:USDT", 1_000_000.0, 100.0, 97.0).allowed
    engine.hold_for_continuity_adjudication()

    decision = engine.evaluate_entry("ETH/USDT:USDT", 1_000_000.0, 100.0, 97.0)

    assert not decision.allowed and "continuity" in decision.reason


@pytest.mark.parametrize("start", ["running", "halted", "missing", "held"])
def test_b5_the_seal_establishes_its_own_halt_and_writes_nothing(tmp_path, start):
    """B5-6. ``halt_for_continuity_dispute`` halts in memory -- with R1-c's
    reason on a running state, keeping the first reason on a halted one -- seals,
    ends a hold, and writes nothing: the found bytes stay, an absent file stays
    absent."""
    reason = f"{RISK_CONTINUITY_PREFIX} test"
    if start == "missing":
        state_path = tmp_path / "aegis" / "risk.json"
        engine = RiskEngine(RiskLimits(), state_path=state_path)
    else:
        engine, state_path = loaded_engine(tmp_path)
    if start == "halted":
        engine.halt("an earlier guard")
    if start == "held":
        engine.hold_for_continuity_adjudication()
    disk = bytes_of(state_path)

    engine.halt_for_continuity_dispute(reason)

    assert engine.halted and engine.continuity_disputed and not engine.continuity_pending
    assert engine.state.halt_reason == ("an earlier guard" if start == "halted" else reason)
    assert bytes_of(state_path) == disk
    if start == "missing":
        assert not state_path.exists()


def sealed_restart(tmp_path: Path, *, halted_file: bool):
    """A foreign ``risk.json`` (the peak moved), optionally one that was itself
    halted, restarted into a real continuity seal."""
    harness = campaign(tmp_path)
    config = harness.runner.config
    state_dir = harness.state_dir
    move_the_peak(state_dir)
    if halted_file:
        document = json.loads(risk_json(state_dir).read_text(encoding="utf-8"))
        document["halted"] = True
        document["halt_reason"] = "a stale copy's own halt"
        risk_json(state_dir).write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    disputed = bytes_of(risk_json(state_dir))
    sealed = restart(tmp_path, config)
    assert sealed.runner.risk.continuity_disputed and sealed.runner.state is RunnerState.HALT
    return sealed, config, state_dir, disputed


def attempt_the_laundering(sealed) -> dict[str, Any]:
    """The review's sequence against a sealed runner, in process: clear the
    halt, adopt an equity, then decide a minute so a ``DECISION`` restates
    ``risk.state_hash``. Returns that DECISION. Refusals are swallowed here
    and asserted by the callers."""
    engine = sealed.runner.risk
    for attempt in (
        lambda: engine.resume(),
        lambda: engine.adopt_reconciled_equity(
            engine.state.equity, clearing=engine.state.halt_reason, note="operator: adopt"
        ),
    ):
        try:
            attempt()
        except RiskViolation:
            pass
    sealed.tick(next_minute(sealed))
    decision = records(sealed.state_dir)[-1]
    assert decision["kind"] == RecordKind.DECISION.value, "a DECISION really was reached"
    return decision


@pytest.mark.parametrize("halted_file", [False, True], ids=["running-file", "halted-file"])
def test_b5_the_review_laundering_sequence_does_not_clear_the_dispute(tmp_path, halted_file):
    """B5-3 / B5-4. The review's probe ended READY, with no continuity fault and
    ``STARTUP`` false. Now: the decisions are refused, the DECISION the sealed
    run still reaches is not a statement, and the restart finds the same finding,
    already recorded, over the same bytes.

    The halted-file case is why the scanner rule is needed ON TOP of the engine
    refusal: a sealed engine that did not move at all still hashes to exactly
    the disputed file when that file was already halted, so its DECISION
    restates the disputed file's own hash."""
    sealed, config, state_dir, disputed = sealed_restart(tmp_path, halted_file=halted_file)
    engine = sealed.runner.risk
    finding = sealed.runner.risk_continuity
    before = engine.snapshot()
    with pytest.raises(RunnerError, match="sealed by an R1-c continuity dispute"):
        sealed.runner.resume("operator: I looked at it")

    decision = attempt_the_laundering(sealed)

    assert engine.snapshot()["halted"] and engine.continuity_disputed
    assert {k: v for k, v in engine.snapshot().items() if k != "order_times"} == {
        k: v for k, v in before.items() if k != "order_times"
    }, "nothing the sequence asked for moved the sealed state"
    assert decision["risk"]["state_hash"] == _risk_hash(engine)
    if halted_file:
        assert decision["risk"]["state_hash"] == finding.found_state_hash, (
            "the laundering precondition itself: the sealed run restated the "
            "disputed file's own hash"
        )
    history = read_log_risk_history(state_dir)
    assert history.state_hash == finding.history.state_hash, "not a statement"

    again = restart(tmp_path, config)

    assert again.runner.state is RunnerState.HALT
    assert again.runner.risk_continuity.fault is RiskContinuityFault.RISK_STATE_MISMATCH
    assert again.runner.risk_continuity.identity == finding.identity
    assert again.runner.risk_continuity.already_recorded
    assert causes(state_dir) == [RecoveryCause.RISK_STATE_MISMATCH.value]
    assert startup_flags(state_dir) == [False, True, True]
    assert bytes_of(risk_json(state_dir)) == disputed


@pytest.mark.parametrize(
    ("engine_refuses", "scanner_ignores", "laundered"),
    [
        (True, True, False),
        (True, False, False),
        (False, True, False),
        (False, False, True),
    ],
    ids=["both", "engine-only", "scanner-only", "neither"],
)
def test_b5_each_defense_alone_stops_the_review_sequence(
    tmp_path, monkeypatch, engine_refuses, scanner_ignores, laundered
):
    """Mutation witness. With BOTH defenses removed the review's exact outcome
    is reproduced -- READY, no continuity fault, ``STARTUP`` false -- so this
    test can fail. Either defense alone stops it for a running foreign file;
    the halted-file case above is the one only the scanner rule stops."""
    if not engine_refuses:
        monkeypatch.setattr(RiskEngine, "_continuity_guard", lambda self, *a, **k: False)
    if not scanner_ignores:
        monkeypatch.setattr(risk_continuity, "_statement_in_run", pre_b5_sealed_run_rule)
    sealed, config, state_dir, _ = sealed_restart(tmp_path, halted_file=False)
    attempt_the_laundering(sealed)

    again = restart(tmp_path, config)

    if laundered:
        assert again.runner.state is RunnerState.READY
        assert not again.runner.risk_continuity.disputed
        assert startup_flags(state_dir)[-1] is False
    else:
        assert again.runner.state is RunnerState.HALT
        assert again.runner.risk_continuity.fault is RiskContinuityFault.RISK_STATE_MISMATCH
        assert startup_flags(state_dir)[-1] is True


def test_a_real_kill_switch_mirror_crash_is_proved_and_deferred(tmp_path, monkeypatch):
    """The ``kill_switch_mirror`` window's direct witness, which the suite
    lacked. Aegis is ALREADY halted -- a real drawdown breach inside a decided
    minute -- so the next minute's switch look persists the mirror alone, and
    the kill lands before ``_halt`` appends its ``HALT``."""
    widen_the_basis(monkeypatch, from_minute=MINUTES, by=5.0)
    config = campaign_config(tmp_path / "state", limits={"max_drawdown_pct": 0.0005})
    harness = build(tmp_path, days=(DAY,), config=config)
    harness.run(MINUTES + 1)
    state_dir = harness.state_dir
    breached = json.loads(risk_json(state_dir).read_text(encoding="utf-8"))
    assert breached["halted"] and breached["halt_reason"].startswith("max drawdown breached")
    assert records(state_dir)[-1]["kind"] == RecordKind.DECISION.value, "restated halted"
    (state_dir / "KILL_SWITCH").write_text("stop\n", encoding="utf-8")
    minute = next_minute(harness)
    die_before_append(harness, RecordKind.HALT, lambda: harness.tick(minute))
    after = json.loads(risk_json(state_dir).read_text(encoding="utf-8"))
    moved = {k for k in after if k != "updated_at" and after[k] != breached[k]}
    assert moved == {"kill_switch"}, "the real window's shape: the mirror alone"

    resumed = restart(tmp_path, config)

    block = assert_proved_and_deferred(resumed, state_dir, "kill_switch_mirror")
    assert block["halt_transition_explains_mismatch"] is False, "it halted nothing"
    assert resumed.runner.halt_reason == breached["halt_reason"], "the first reason"
