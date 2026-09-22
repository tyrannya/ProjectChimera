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
from chimera.demo.risk_continuity import (
    RISK_CONTINUITY_PREFIX,
    LogRiskStatement,
    RiskContinuityFault,
    assess_risk_continuity,
    read_log_risk_history,
    risk_state_hash,
)
from chimera.demo.runner import (
    CAUSES_WITHOUT_AN_AFFECTED_MINUTE,
    RecoveryCause,
    RunnerState,
    _risk_hash,
)
from chimera.risk import RiskEngine, RiskStateLoad
from tests.demo_harness import CAPITAL, DAY, build, campaign_config

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


def restart(tmp_path: Path, config) -> Any:
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
    resumed = build(tmp_path, days=(DAY,), config=config, start=False)
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

    def mutant(*, load, snapshot, state_dir, history=None):
        history = read_log_risk_history(state_dir) if history is None else history
        from dataclasses import replace

        return real(
            load=load,
            snapshot=snapshot,
            state_dir=state_dir,
            history=replace(history, **overrides),
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

    def mutant(*, load, snapshot, state_dir, history=None):
        if load is RiskStateLoad.UNREADABLE:
            load = RiskStateLoad.MISSING
        return real(load=load, snapshot=snapshot, state_dir=state_dir, history=history)

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

    def mutant(*, load, snapshot, state_dir, history=None):
        from dataclasses import replace

        return replace(
            real(load=load, snapshot=snapshot, state_dir=state_dir, history=history),
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
