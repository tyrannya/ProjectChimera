"""R1-c (AEG-4): whether ``risk.json`` continues this campaign's decision log.

The adopted roadmap's item, verbatim:

    **R1-c ``risk.json`` continuity (AEG-4):** a missing or unreadable
    ``risk.json`` when the decision log already holds records is a dispute, not
    a default; the loaded snapshot is compared with the log's last
    ``risk.state_hash``/HALT record; a ``RECOVERY`` record is written; the same
    continuity the store and ledger already have.

R1-b closed AEG-1 and stopped there on purpose. Its
:func:`chimera.demo.risk_wiring.seed_or_reconcile_equity` reads
:attr:`chimera.risk.RiskEngine.load_outcome` and treats ``MISSING`` as a genuine
first start, "and whether a missing ``risk.json`` is itself suspicious -- because
the decision log already holds records for this campaign -- is canonical R1-c's
question". This module is that question's answer, and it is asked of the one
file that can settle it: the campaign's own append-only decision log.

**What the defect was.** Nothing compared the two. Delete ``risk.json`` from a
campaign whose log holds a month of records and the next start seeded the
configured capital, wrote a brand-new state file, and ran -- with the peak
equity, the day's baseline, the order window, the streaks, the reconciliation
disputes and any HALT the campaign was carrying all silently gone. The log still
proved every one of them. Nothing read it.

**What is compared, and why it is not simply a hash.** The log records the risk
state in two different ways and the difference is load-bearing:

* ``DECISION``, ``FUNDING``, ``RECONCILIATION`` and ``LIQUIDATION_TOUCH`` records
  carry a ``risk.state_hash`` block -- the full snapshot, hashed. Those are the
  records that RESTATE the state.
* ``HALT`` and ``RESUME`` say that the state moved and do not restate it.
  ``OPERATOR`` (a flatten, an adopted equity) and ``INCOMPLETE_STATE`` (whose
  tick calls ``note_feed`` before it gives up on the minute) can move it and
  restate nothing either.

So "the log's last ``risk.state_hash``" is **not** the last line of the file and
not even the last record that happens to contain the string: an ordinary
operator flatten moves ``equity`` and writes an ``OPERATOR`` record with no risk
block at all, and a campaign that halts moves ``halted`` and ``halt_reason`` and
writes a ``HALT`` record with no risk block either. A comparison that took the
newest hash-bearing record and demanded equality would report both of those --
the two most ordinary things an operator does -- as continuity failures.

This module therefore reads the log's **last risk statement**: the newest record
that says anything about the risk state at all, and what kind of thing it says.
See :class:`LogRiskStatement`. The comparison is then made against what that
statement actually claims and against nothing it does not claim, which is what
keeps it free of both false disputes and invented tolerance.

**The direction of the check.** A dispute is raised when the file claims LESS
than the log does -- it is absent, it cannot be read, it is not the state the log
hashed, or the log says the campaign is halted and the file says it is running.
A file that is MORE restrictive than the log (halted where the log's newest
statement is a ``RESUME``) is not a dispute: R1-c exists to stop a halt or a
history being silently lost, not to stop one being kept.

**What this module does not do.** It writes nothing, opens nothing for append,
and takes no operator decision. It reads the log through
:func:`chimera.demo.decision_log.verify_log` and
:func:`chimera.demo.decision_log.read_records` -- the verified records, never a
string search over the NDJSON -- and returns a verdict.
:func:`chimera.demo.risk_wiring.build_risk_engine` uses it to decide whether the
configured capital may seed an engine; :class:`chimera.demo.runner.DemoRunner`
uses it to write the ``RECOVERY`` record the roadmap asks for. Neither the log
nor the state file is rewritten to make the two agree, by this module or by
either caller.

**The decision log is evidence, not a second risk authority.** Aegis remains the
sole central risk authority (``CLAUDE.md``). What the log provides here is the
answer to one question -- *is the persisted Aegis state believable?* -- and when
the answer is no, the campaign stops rather than trading on a state nobody can
vouch for.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from chimera.demo.clock import no_authoritative_time
from chimera.demo.decision_log import (
    LOG_DIR_NAME,
    RecordKind,
    day_files,
    read_records,
    verify_log,
)
from chimera.demo.rules import canonical_hash
from chimera.risk import (
    KILL_SWITCH_HALT_REASONS,
    RiskEngine,
    RiskLimits,
    RiskState,
    RiskStateLoad,
)

#: Risk-state fields that are WALL CLOCK or HOST DATE rather than decision
#: semantics, and are therefore outside ``risk.state_hash``.
#:
#: ``order_times`` is the rate limiter's record of when orders were placed and
#: ``cooldown_until`` is a deadline computed from one; ``day`` is the date the
#: host happened to be running on. All three move between a live run and a
#: replay of the same files, and none of them is a fact about the decision -- so
#: hashing them would make section 10's byte comparison fail for a reason that
#: has nothing to do with whether the two runs decided alike.
#:
#: Everything that governs permission and IS reproducible stays in: ``halted``,
#: ``halt_reason``, ``kill_switch``, the equity series, the streaks, and the
#: reconciliation disputes.
#:
#: It lives here rather than in the runner because R1-c hashes a snapshot that
#: never became a live engine, and the two hashes have to be the same function
#: or the comparison is between two different things.
RISK_HASH_EXCLUDED: frozenset[str] = frozenset({"order_times", "cooldown_until", "day"})

#: What a continuity dispute's halt reason begins with. A prefix, because the
#: reason names the finding after it.
RISK_CONTINUITY_PREFIX = "risk_continuity:"

#: The ``STARTUP`` payload field that says whether THIS run's Aegis was sealed
#: by an R1-c dispute (:attr:`chimera.risk.RiskEngine.continuity_disputed`,
#: settled before ``STARTUP`` is appended and fixed for the run's life). A
#: sealed engine persists nothing, so no record its run appends -- a restated
#: ``risk.state_hash`` included -- describes a write to ``risk.json``;
#: :func:`read_log_risk_history` reads the field per run, from one ``STARTUP``
#: to the next, and never carries it across.
#:
#: Absent means ``False``. That is right for every log written by a ``main``
#: build from before PR #102 (none can seal) and moot for every build from
#: ``74edc25`` on (each writes the field on every ``STARTUP``). It is WRONG for
#: a log written by an intermediate, unmerged PR #102 build from ``dded833`` up
#: to and including ``471c1b7``, which could already seal but did not yet write
#: the field; a sealed run of one of those reads here as unsealed. Reading such
#: a log correctly is outside this code: the compatibility condition -- no state
#: directory was ever run by one of those builds -- is for the owner to attest,
#: and ``docs/demo_runbook.md`` states it.
RISK_CONTINUITY_SEALED_FIELD = "risk_continuity_sealed"


def risk_state_hash(snapshot: Mapping[str, Any]) -> str:
    """Section 9.1's ``risk.state_hash`` for a :meth:`chimera.risk.RiskState.snapshot`.

    One function, used by the runner when it writes a record and by this module
    when it reads one back. ``chimera.demo.runner._risk_hash`` delegates here.
    """
    return canonical_hash(
        {
            key: value
            for key, value in snapshot.items()
            if isinstance(key, str) and key not in RISK_HASH_EXCLUDED
        }
    )


#: The crash windows :func:`_crash_transition` can prove, by name, and which of
#: them persisted a halt. The name is what the ``RECOVERY`` record's ``detail``
#: reports; the record's ``halt_transition_explains_mismatch`` is true exactly
#: for the halting ones.
CRASH_TRANSITIONS: frozenset[str] = frozenset(
    {"halt", "kill_switch_halt", "kill_switch_mirror", "equity", "equity_halt"}
)
_HALTING_TRANSITIONS: frozenset[str] = frozenset({"halt", "kill_switch_halt", "equity_halt"})


def _crash_transition(
    snapshot: Mapping[str, Any],
    history: "LogRiskHistory",
    *,
    limits: RiskLimits | None = None,
    ledger_equity: Decimal | None = None,
) -> str:
    """Which ONE production persist-then-append window provably produced this file.

    Returns the window's name (see :data:`CRASH_TRANSITIONS`), or ``""`` when
    none is proved -- and ``""`` is what keeps the file disputed.

    Every window here has the same shape. An Aegis method persists
    ``risk.json`` and the record that would restate the hash is appended after
    it; a process killed in between leaves the file one transition ahead of the
    log's last ``STATE_HASH`` statement, and section 9.3's ``_triage_log`` sees
    none of it, because no store, ledger accumulator or chain head moved.

    **The proof, and the only thing accepted.** For each window a candidate
    PRIOR state is built by reverting exactly the fields that window writes, to
    the one value they can have held before it. The candidate is accepted only
    if its FULL ``risk.state_hash`` equals the log's -- so every other hashed
    field (peak, day baseline, streaks, positions, disputes, feed mark) is
    proved unmoved, not assumed -- and, for an equity window, only if the real
    :meth:`chimera.risk.RiskEngine.update_equity` applied to that candidate
    reproduces the found file's hash exactly. Nothing is accepted because it
    "looks like" a crash. A foreign or swapped file passes only if it is, in
    every hashed field, the log's own prior state carried through one real
    transition.

    The windows:

    ``halt``
        :meth:`RiskEngine.halt` from a running state -- a rule exception, a
        dispute, a funding or execution failure, every ``DemoRunner._halt``
        site -- persisted before ``_halt`` appended its ``HALT``. ``halt`` moves
        ``halted`` and ``halt_reason`` only, and only from ``halted: False``
        (it is a no-op otherwise), so the prior is ``False``/``""``.
    ``kill_switch_halt``
        :meth:`RiskEngine.check_kill_switch` finding the switch present on a
        running engine: the mirror goes ``False`` -> ``True`` and it halts with
        one of :data:`chimera.risk.KILL_SWITCH_HALT_REASONS`, then the tick or
        ``start()`` appends ``HALT``. Three fields, and the reason must be one of
        the two that method writes. Whether the switch file is still there at
        the restart does not enter the proof: the mirror is what was persisted.
    ``kill_switch_mirror``
        The same look on an engine that was ALREADY halted (a ``DECISION``
        whose own ``update_equity`` breached a limit, then a later minute's
        switch): only the mirror moves.
    ``equity`` / ``equity_halt``
        :meth:`RiskEngine.update_equity` in the tick's PERSISTENCE step,
        persisted before the minute's ``DECISION`` -- with or without the
        drawdown / daily-loss halt ``_record_equity`` raises inside it. The
        prior equity is NOT hidden behind the hash: every ``DECISION`` (and
        every flatten's ``OPERATOR``) carries the exact equity it handed Aegis
        as ``ledger_effect.equity``, and the newest one is
        :attr:`LogRiskHistory.equity_given`. The prior ``daily_pnl`` follows
        from it and the unmoved ``day_start_equity``. Two further conditions:
        the found equity must equal the carry ledger's persisted equity,
        compared exactly as R1-b compares them, because the same crash wrote
        the ledger immediately before Aegis (``DemoRunner._decide``:
        ``_save_ledger()`` then ``update_equity``); and the forward replay
        above. A funding minute is the same window: its ``FUNDING`` record
        restates the hash without moving Aegis's equity, and the tick's
        ``update_equity`` follows it.

    **What this does not prove, and why each stays disputed.** A ``DECISION``
    that rolled the UTC day overwrote ``day_start_equity`` and ``daily_pnl``,
    and one that set a new peak overwrote ``peak_equity``. The values they held
    before are not restated by the record the crash preceded: the prior day's
    baseline is the equity of whichever write first touched that day, and the
    prior peak is the running maximum over every equity Aegis was ever given.
    Neither is necessarily unavailable -- the configured capital or a bounded
    scan of the log's earlier records may reconstruct them -- but that is
    historical reconstruction, replay-shaped logic over the campaign's equity
    history, well beyond the one-step local inverse every proof here is. It is
    canonical **R1-i**'s crash/restart work, so those windows stay
    ``RISK_STATE_MISMATCH``. So do the windows closed by a ``STATE_HASH``
    record rather than a ``HALT`` or ``DECISION`` (``note_funding_settlement``'s
    streak before its ``FUNDING`` record), those that move the feed mark
    (``note_feed``), and any combination of two windows. Deliberately: an
    unproved window is sealed, never waved through.
    """
    if history.statement is not LogRiskStatement.STATE_HASH:
        return ""
    found = dict(snapshot)
    running = {"halted": False, "halt_reason": ""}
    candidates: list[tuple[str, dict[str, Any]]] = []
    if found.get("halted"):
        candidates.append(("halt", running))
        if found.get("kill_switch"):
            if found.get("halt_reason") in KILL_SWITCH_HALT_REASONS:
                candidates.append(("kill_switch_halt", {"kill_switch": False, **running}))
            candidates.append(("kill_switch_mirror", {"kill_switch": False}))
    for name, prior in candidates:
        if risk_state_hash({**found, **prior}) == history.state_hash:
            return name
    return _equity_transition(found, history, limits=limits, ledger_equity=ledger_equity)


def _equity_transition(
    found: dict[str, Any],
    history: "LogRiskHistory",
    *,
    limits: RiskLimits | None,
    ledger_equity: Decimal | None,
) -> str:
    """The ``equity`` / ``equity_halt`` half of :func:`_crash_transition`.

    Needs the campaign's limits (to replay ``update_equity``'s own guards) and
    the ledger's equity. A caller that has neither -- ``build_risk_engine``,
    which defers every ``RISK_STATE_MISMATCH`` to the runner anyway -- gets no
    equity proof, never a weaker one.
    """
    given = history.equity_given
    if limits is None or ledger_equity is None or given is None:
        return ""
    if float(ledger_equity) != found["equity"]:
        return ""
    try:
        day = datetime.fromisoformat(str(found["day"])).replace(tzinfo=timezone.utc)
    except ValueError:
        return ""
    before = {**found, "equity": given, "daily_pnl": given - found["day_start_equity"]}
    options = [("equity", before)]
    if found["halted"]:
        options.append(("equity_halt", {**before, "halted": False, "halt_reason": ""}))
    target = risk_state_hash(found)
    for name, prior in options:
        if risk_state_hash(prior) != history.state_hash:
            continue
        # A detached engine -- no state file, no switch -- so the replay writes
        # nothing anywhere. Same day as the found file, so no day roll. Its day
        # comes from `now=` and it persists nothing, so it never needs a clock;
        # the raising one makes that a fact rather than the `time.time` default
        # quietly standing by on the demo path (R1-e).
        replay = RiskEngine(
            limits, clock=no_authoritative_time, check_kill_switch_at_construction=False
        )
        replay.state = RiskState.from_dict(prior)
        replay.update_equity(float(found["equity"]), now=day)
        if risk_state_hash(replay.snapshot()) == target:
            return name
    return ""


class LogRiskStatement(str, Enum):
    """What the log's newest record about the risk state actually says.

    Bounded, because the comparison branches on it and a statement nobody
    classified would have to be guessed at. The four members are not four
    strengths of the same claim; they are four different claims, and each admits
    a different comparison:

    ``STATE_HASH``
        A record carrying a ``risk.state_hash`` block: the whole snapshot,
        hashed, and nothing has moved it since. The file's hash must equal it.

    ``HALTED``
        A ``HALT`` record. The campaign stopped; the hash was not restated,
        because ``DemoRunner._halt`` appends no risk block. What it does claim is
        that the campaign is halted, and a file that says otherwise has had a
        halt cleared behind the log's back.

        With one exception, and it is R1-c's own doing: a ``HALT`` appended by a
        run whose engine was SEALED by a continuity dispute did not persist
        anything, so it claims nothing about the file. Which runs were sealed is
        read off each run's own ``STARTUP`` record
        (:data:`RISK_CONTINUITY_SEALED_FIELD`); see :func:`read_log_risk_history`.

    ``RUNNING``
        A ``RESUME`` record. An operator cleared a halt, and again no hash was
        restated. Nothing is required of the file: see the module docstring on
        the direction of the check.

    ``MOVED``
        An ``OPERATOR`` or ``INCOMPLETE_STATE`` record. The risk state may have
        moved -- an operator flatten hands Aegis a new equity, and the tick
        behind an incomplete minute has already called ``note_feed`` -- and
        nothing was restated. The log makes no comparable claim, so none is
        checked. The campaign's very next decided minute restates the hash and
        the window closes.

        With the same one exception as ``HALTED``: a ``MOVED`` record appended
        by a sealed run claims nothing either, because the seal also refused the
        write it would otherwise describe -- a ``flatten`` issued during a
        continuity halt persists no equity to ``risk.json``. The same holds for
        ``RUNNING``.

    ``NONE``
        The log says nothing about the risk state: it holds no records, or only
        records of kinds that neither carry nor move one.
    """

    NONE = "NONE"
    STATE_HASH = "STATE_HASH"
    HALTED = "HALTED"
    RUNNING = "RUNNING"
    MOVED = "MOVED"


#: Record kinds that MOVE the risk state without restating its hash. Listed
#: literally, with the mutation that makes each one true, because an omission
#: here is a false dispute on an ordinary operation and an addition here is a
#: blind spot:
#:
#: ``OPERATOR`` -- ``DemoRunner.flatten`` hands the flattened equity to
#:     ``update_equity``; ``resolve_equity`` adopts an equity through
#:     ``RiskEngine.adopt_reconciled_equity``. Both persist, and the record
#:     carries no ``risk`` block.
#: ``INCOMPLETE_STATE`` -- ``DemoRunner.tick`` calls ``RiskEngine.note_feed``
#:     before it decides whether the minute is complete, and ``note_feed``
#:     persists ``stale_feed_since``, which IS hashed.
#:
#: ``SKIPPED_STALE`` is deliberately absent: the builds that wrote it (before
#: R1-g retired the catch-up cap) appended it from ``catch_up`` without a tick,
#: and "nothing else about the position, the ledger or Aegis" moved. ``STARTUP``,
#: ``SHUTDOWN`` and ``RECOVERY`` are absent for the same reason -- none of the
#: three mutates Aegis, and the constructor's ``check_kill_switch`` (which does)
#: runs after the load-time snapshot this compares is taken.
_MOVING_KINDS: frozenset[str] = frozenset(
    {RecordKind.OPERATOR.value, RecordKind.INCOMPLETE_STATE.value}
)


class RiskContinuityFault(str, Enum):
    """What R1-c found. A bounded vocabulary, and a new one on purpose.

    The existing :class:`chimera.demo.runner.RecoveryCause` members describe
    LOG/STATE WRITE-ORDER crash windows -- ``LOG_BEHIND_STATE``,
    ``LOG_AHEAD_OF_STATE``, ``TORN_TAIL`` -- and a forgery. A ``risk.json`` that
    is absent, corrupt or foreign is none of those: no crash produces it, the
    direction those names carry does not apply to it, and reporting it under one
    of them would say something untrue about which file was behind. The runner
    adds one ``RecoveryCause`` member per fault here, with the same value, so a
    ``RECOVERY`` record names the finding precisely and section 9.3's names keep
    section 9.3's meanings.

    Three members, because an operator's next move differs for each:

    ``RISK_STATE_ABSENT``
        There is no ``risk.json`` and the verified log holds records. The
        campaign had state and the file that held it is gone.

    ``RISK_STATE_UNREADABLE``
        There is a ``risk.json``, it could not be believed
        (:meth:`chimera.risk.RiskEngine._load_state` failed closed on it), and
        the verified log holds records. R1-b already halts on this; what R1-c
        adds is that the continuity failure is on the record rather than only in
        a log line on one host.

    ``RISK_STATE_PRE_SCHEMA``
        There is a ``risk.json``, it is the pre-schema halt record -- a halt
        claim and no account state at all -- and the verified log holds records.

    ``RISK_STATE_MISMATCH``
        There is a readable, current-schema ``risk.json`` and it is not the
        state the log last recorded: a different hash, or a state that is
        running where the log says the campaign halted.

    The first three are findings about the FILE and the last is a finding about
    its CONTENT, which is a difference with teeth; see
    :attr:`RiskContinuity.crash_could_explain`.
    """

    RISK_STATE_ABSENT = "RISK_STATE_ABSENT"
    RISK_STATE_UNREADABLE = "RISK_STATE_UNREADABLE"
    RISK_STATE_PRE_SCHEMA = "RISK_STATE_PRE_SCHEMA"
    RISK_STATE_MISMATCH = "RISK_STATE_MISMATCH"


@dataclass(frozen=True)
class LogRiskHistory:
    """What the campaign's decision log says about its risk state.

    Read once, from the VERIFIED records. ``trustworthy`` is false for a forged
    log, and a forged log's statements are not used: deciding that a ``risk.json``
    disagrees with records that do not verify would let an edited log condemn a
    healthy state file. What a forged log can still establish is that this
    campaign HAS a history, which is why ``records`` is reported either way --
    "bad log plus missing risk.json" must not resolve to a fabricated healthy
    state, and it does not.
    """

    #: Complete, well-formed records the verification walked. Zero means the log
    #: holds nothing: no day file, or day files with nothing in them.
    records: int = 0
    #: False when a COMPLETE record does not verify. A torn tail is not forgery
    #: and does not clear this: the bytes after the last newline never formed a
    #: record, `read_records` stops before them, and the verified prefix is still
    #: the campaign's evidence.
    trustworthy: bool = True
    #: Whether the chain verification found a torn tail. Reported so the
    #: ``RECOVERY`` record can say the statements came from a repaired file's
    #: complete prefix; `DemoRunner._triage_log` owns the repair itself.
    torn: bool = False
    statement: LogRiskStatement = LogRiskStatement.NONE
    #: The record kind the statement came from. Empty for ``NONE``.
    statement_kind: str = ""
    #: That record's ``seq``. ``None`` for ``NONE``.
    statement_seq: int | None = None
    #: The hash the statement carries. Empty unless ``statement`` is
    #: ``STATE_HASH``.
    state_hash: str = ""
    #: The ``seq`` of the newest record that RESTATED the hash, whatever the
    #: newest statement is. Used only for idempotence; see
    #: :attr:`RiskContinuity.already_recorded`.
    last_restated_seq: int | None = None
    #: The newest R1-c ``RECOVERY`` record already in the log, as
    #: ``(seq, identity)``. ``None`` when the log holds none.
    recorded_dispute: tuple[int, tuple[str, str, str]] | None = None
    #: The equity the log last records Aegis being HANDED by
    #: ``update_equity``: the ``ledger_effect.equity`` of the newest ``DECISION``
    #: or ``OPERATOR`` record that carries one. Both writers
    #: pass the very same mark to ``update_equity`` and to the block
    #: (``DemoRunner._decide`` and ``DemoRunner.flatten``). ``FUNDING`` and
    #: ``LIQUIDATION_TOUCH`` blocks are NOT this: those records do not call
    #: ``update_equity``. Used only as the candidate prior equity of
    #: :func:`_crash_transition`, whose hash check decides whether it was.
    equity_given: float | None = None

    @property
    def has_history(self) -> bool:
        """Whether a campaign has already committed records under this log.

        This is the whole of R1-c's "when the decision log already holds
        records", and it is deliberately not narrowed to records that carry a
        risk block. Every entry point builds Aegis BEFORE the runner appends its
        first ``STARTUP`` record -- and on a genuine first start that build
        seeds equity and persists ``risk.json`` -- so a log holding any record
        at all is a log whose campaign had a state file by the time that record
        was written.
        """
        return self.records > 0


@dataclass(frozen=True)
class RiskContinuity:
    """R1-c's verdict on one state directory. Computed once, before any mutation."""

    load: RiskStateLoad
    history: LogRiskHistory
    fault: RiskContinuityFault | None = None
    detail: str = ""
    #: The hash of the snapshot as the read found it. Empty when there was no
    #: readable account state to hash (``MISSING``, ``UNREADABLE``).
    found_state_hash: str = ""
    #: Whether the verified log already holds a ``RECOVERY`` record for this same
    #: finding, with nothing restating the risk state since.
    already_recorded: bool = False
    #: The one production crash window PROVED to turn the log's last
    #: ``STATE_HASH`` into the found file (:data:`CRASH_TRANSITIONS`), or ``""``.
    #: See :func:`_crash_transition`. Non-empty only on a
    #: ``RISK_STATE_MISMATCH``, and a non-empty one is what defers it.
    crash_transition: str = ""

    @property
    def disputed(self) -> bool:
        return self.fault is not None

    @property
    def halt_transition_explains_mismatch(self) -> bool:
        """Whether the proved crash window is one that persisted a HALT.

        The ``RECOVERY`` block's field of the same name. Narrower than
        :attr:`crash_transition` on purpose: an ``equity`` window halted
        nothing, and the name says "halt". The window itself is named in
        :attr:`detail`, and whether the run was sealed in its ``STARTUP``.
        """
        return self.crash_transition in _HALTING_TRANSITIONS

    @property
    def identity(self) -> tuple[str, str, str]:
        """What makes two findings the same finding, for idempotence.

        The fault, how the file was found, and what it hashed to -- and NOT the
        log's statement. A restart that raises this dispute writes a ``HALT``
        record of its own, which becomes the log's newest statement, so the next
        restart looking at an unchanged file would otherwise see a "different"
        finding and record it again, and again. What must produce a second
        record is a change on the RISK side: a file that appears, disappears,
        stops parsing, or is replaced with different contents.
        """
        return (
            "" if self.fault is None else self.fault.value,
            self.load.value,
            self.found_state_hash,
        )

    @property
    def crash_could_explain(self) -> bool:
        """Whether the campaign's own crash windows could have produced this.

        True for ``RISK_STATE_MISMATCH`` alone, and the distinction is what keeps
        R1-c from claiming section 9.3's work.

        A tick mutates Aegis and *then* appends the record that restates the
        hash -- ``update_equity``, ``record_order``, ``update_position`` and
        ``note_feed`` all persist before the ``DECISION`` is written, and
        ``RiskEngine.halt`` persists before ``_halt`` appends its ``HALT``. A
        process killed in any of those windows leaves a ``risk.json`` that is
        genuinely AHEAD of the log, and the hash comparison sees exactly what a
        swapped file looks like. Section 9.3 already names that crash --
        ``LOG_BEHIND_STATE``, ``LOG_AHEAD_OF_STATE``, ``TORN_TAIL`` -- excludes
        the minute it touched and continues, and turning it into an unclearable
        continuity halt would be R1-c taking a recovery away.
        :meth:`chimera.demo.runner.DemoRunner._risk_continuity_stands` is where
        the deferral happens, and it defers only when the triage really found a
        crash, or when :func:`_crash_transition` PROVES one of the Aegis-only
        windows the triage cannot see (:attr:`crash_transition`).

        Nothing a crash does can DELETE ``risk.json``, corrupt it, or replace it
        with a pre-schema document, so the other three faults stand whatever
        else startup finds -- and they are also the three where the alternative
        is a fabrication: ``MISSING`` and ``LEGACY`` are what
        :func:`chimera.demo.risk_wiring.seed_or_reconcile_equity` seeds the
        configured capital for.
        """
        return self.fault is RiskContinuityFault.RISK_STATE_MISMATCH

    @property
    def halt_reason(self) -> str:
        """The halt this verdict raises, or an empty string when there is none.

        Prefixed with :data:`RISK_CONTINUITY_PREFIX` so an operator reading a
        ``HALT`` record can tell an R1-c dispute from every other halt the
        campaign can carry without parsing the sentence after it.
        """
        if self.fault is None:
            return ""
        return f"{RISK_CONTINUITY_PREFIX} {self.detail}"

    def record_block(self) -> dict[str, Any]:
        """The ``recovery.risk_continuity`` block of R1-c's ``RECOVERY`` record.

        Deliberately NOT a ``risk`` block. Section 9.1's ``risk.state_hash`` is
        what the next start compares against, so writing the disputed hash there
        would make the dispute its own evidence and the restart after this one
        would compare the file against itself. The hashes are named inside
        ``recovery`` instead, and they are OMITTED rather than written as
        ``null`` when there is none -- ``chimera.demo.decision_log`` holds every
        field whose name ends in ``_hash`` to ``sha256:<64 hex>``, and
        ``_append(STARTUP)`` already takes the same route with ``protocol_hash``.
        ``risk_state_load`` and ``log_statement`` are what say whether each one
        exists, so nothing has to be inferred from an absent key.
        """
        block: dict[str, Any] = {
            "fault": "" if self.fault is None else self.fault.value,
            "detail": self.detail,
            "risk_state_load": self.load.value,
            "log_statement": self.history.statement.value,
            "log_statement_kind": self.history.statement_kind,
            "log_statement_seq": self.history.statement_seq,
            "log_records": self.history.records,
            "log_trustworthy": self.history.trustworthy,
            "log_torn_tail": self.history.torn,
            "halt_transition_explains_mismatch": self.halt_transition_explains_mismatch,
        }
        if self.history.state_hash:
            block["log_state_hash"] = self.history.state_hash
        if self.found_state_hash:
            block["found_state_hash"] = self.found_state_hash
        return block

    def status_block(self) -> dict[str, Any]:
        """The same finding for the CLI's ``status``, which never starts a runner.

        An operator has to be able to see this WITHOUT starting the campaign,
        for the same reason ``ledger_complaint`` is in that output: the command
        that says what is wrong must not be one that first has to run.
        """
        return {
            "disputed": self.disputed,
            **self.record_block(),
            "already_recorded": self.already_recorded,
        }


def _statement_of(record: Mapping[str, Any]) -> tuple[LogRiskStatement, str]:
    """What one record says about the risk state, and the hash if it states one.

    The ``risk`` block is looked for BY NAME and its ``state_hash`` read out of
    it. Section 9.1 puts the block only on the records that restate the state,
    and ``chimera.demo.decision_log`` holds every ``*_hash`` field to one form,
    so this is a structural read rather than a search for a hash-shaped string
    anywhere in the record -- which would find, among others, the
    ``found_state_hash`` R1-c writes into its own ``RECOVERY`` record.
    """
    risk = record.get("risk")
    if isinstance(risk, Mapping):
        state_hash = risk.get("state_hash")
        if isinstance(state_hash, str) and state_hash:
            return LogRiskStatement.STATE_HASH, state_hash
    kind = str(record.get("kind", ""))
    if kind == RecordKind.HALT.value:
        return LogRiskStatement.HALTED, ""
    if kind == RecordKind.RESUME.value:
        return LogRiskStatement.RUNNING, ""
    if kind in _MOVING_KINDS:
        return LogRiskStatement.MOVED, ""
    return LogRiskStatement.NONE, ""


def _statement_in_run(found: LogRiskStatement, *, sealed_run: bool) -> LogRiskStatement:
    """What a record's statement counts for, given the run that appended it.

    Nothing, when the run's own ``STARTUP`` says its Aegis was SEALED -- of any
    kind, a restated ``risk.state_hash`` included; the previous statement
    stands. See the ``sealed_run`` notes in :func:`read_log_risk_history`.
    """
    return LogRiskStatement.NONE if sealed_run else found


def _recorded_dispute(record: Mapping[str, Any]) -> tuple[str, str, str] | None:
    """The identity of an R1-c finding a previous start already recorded."""
    if str(record.get("kind", "")) != RecordKind.RECOVERY.value:
        return None
    recovery = record.get("recovery")
    if not isinstance(recovery, Mapping):
        return None
    block = recovery.get("risk_continuity")
    if not isinstance(block, Mapping):
        return None
    return (
        str(block.get("fault", "")),
        str(block.get("risk_state_load", "")),
        str(block.get("found_state_hash", "")),
    )


def read_log_risk_history(state_dir: str | Path) -> LogRiskHistory:
    """What the campaign's verified decision log says about its risk state.

    One forward walk over the verified records. ``verify_log`` decides whether
    they may be believed at all and ``read_records`` yields only the complete,
    canonical ones -- it stops at the first line it cannot read, so a torn tail's
    bytes are never presented as a record.

    **A disclosed limit, not a designed one.** "Is there a log at all" is
    :func:`chimera.demo.decision_log.day_files`' answer, and it asks
    ``Path.is_dir()`` -- which returns ``False`` for a directory it merely could
    not EXAMINE, exactly the fail-open shape
    :meth:`chimera.risk.RiskEngine._load_state` and
    :meth:`chimera.carry.ledger.CarryLedger.open` were both fixed to avoid. A log
    directory on a degraded mount therefore reads here as "no history", and a
    missing ``risk.json`` beside it would pass as a first start. R1-c does not
    add a second reader for this: ``day_files`` and ``verify_log`` are the
    decision log's own, R1-c composes with them rather than around them, and
    hardening them is a change to the log module that would reach every caller.
    Through the runner the campaign still fails closed by another route --
    ``DecisionLog.open`` raises on the first append, so nothing decides a minute
    -- and the exposure is a ``build_risk_engine`` call with no runner behind it.
    """
    root = Path(state_dir) / LOG_DIR_NAME
    if not day_files(root):
        return LogRiskHistory()

    verification = verify_log(root)
    statement = LogRiskStatement.NONE
    statement_kind = ""
    statement_seq: int | None = None
    state_hash = ""
    last_restated_seq: int | None = None
    recorded: tuple[int, tuple[str, str, str]] | None = None

    equity_given: float | None = None

    #: Whether the RUN the current record belongs to had a SEALED engine, as
    #: that run's own ``STARTUP`` record says (:data:`RISK_CONTINUITY_SEALED_FIELD`).
    #: A run is every record from one ``STARTUP`` to the next, and each run
    #: states its own value: nothing is carried from one run into another.
    #:
    #: A sealed engine persists nothing (`RiskEngine._persist` refuses every
    #: write while `_continuity_disputed` is set), so a ``HALT``, ``RESUME``,
    #: ``OPERATOR`` or ``INCOMPLETE_STATE`` appended by a sealed run corresponds
    #: to no write at all. Reading one as a statement would be R1-c condemning a
    #: file for the halt R1-c itself raised -- restore the right `risk.json` and
    #: the sealed run's own HALT would disagree with it for ever -- or, for an
    #: ``OPERATOR``, would let a flatten issued during the seal erase the seal:
    #: ``MOVED`` makes no comparable claim, so the next restart would check
    #: nothing and unseal a `risk.json` nobody had repaired.
    #:
    #: Per run, and not "from an R1-c RECOVERY record until the next restated
    #: hash" as an earlier revision had it. A RECOVERY record is not evidence
    #: that any later record was written while sealed: the run that REPAIRED the
    #: file starts unsealed and may flatten before it ticks, and a proved crash
    #: window writes an R1-c RECOVERY without sealing anything. Reading either
    #: run's records as sealed made a legitimate flatten a false dispute.
    #:
    #: ``INCOMPLETE_STATE`` shares the rule structurally, not because a sealed
    #: run reaches it in production: the CLI never ticks after ``start()``
    #: returns HALT, and every sealed run does. Only a harness driving ``tick``
    #: past a HALT can append one from a sealed run.
    #:
    #: So does a ``STATE_HASH`` -- a ``DECISION``, ``FUNDING``,
    #: ``RECONCILIATION`` or ``LIQUIDATION_TOUCH`` record carrying
    #: ``risk.state_hash`` -- and this is not only defence in depth. The engine
    #: refuses every mutation while sealed (``RiskEngine._continuity_guard``), so
    #: a sealed run's hash is the hash of the state as it was LOADED, plus the
    #: continuity halt if it was not already halted. When the disputed file was
    #: itself halted, that is exactly the disputed file's own hash: a sealed run
    #: that restated it would hand the next start a "matching" statement and
    #: clear the dispute with no repair at all. A record from a run that could
    #: not write ``risk.json`` cannot prove what ``risk.json`` holds.
    sealed_run = False

    for path in day_files(root):
        for record in read_records(path):
            seq = record.get("seq")
            seq = int(seq) if isinstance(seq, int) else None
            kind = str(record.get("kind", ""))
            if kind == RecordKind.STARTUP.value:
                sealed_run = record.get(RISK_CONTINUITY_SEALED_FIELD) is True
            found, record_hash = _statement_of(record)
            found = _statement_in_run(found, sealed_run=sealed_run)
            if found is not LogRiskStatement.NONE:
                statement = found
                statement_kind = kind
                statement_seq = seq
                state_hash = record_hash
                if found is LogRiskStatement.STATE_HASH:
                    last_restated_seq = seq
            if kind in (RecordKind.DECISION.value, RecordKind.OPERATOR.value):
                effect = record.get("ledger_effect")
                if isinstance(effect, Mapping):
                    try:
                        equity_given = float(effect["equity"])
                    except (KeyError, TypeError, ValueError):
                        pass
            identity = _recorded_dispute(record)
            if identity is not None and seq is not None:
                recorded = (seq, identity)

    return LogRiskHistory(
        records=verification.records,
        trustworthy=not verification.is_forged,
        torn=verification.is_torn,
        statement=statement,
        statement_kind=statement_kind,
        statement_seq=statement_seq,
        state_hash=state_hash,
        last_restated_seq=last_restated_seq,
        recorded_dispute=recorded,
        equity_given=equity_given,
    )


def _no_dispute(load: RiskStateLoad, history: LogRiskHistory, found: str) -> RiskContinuity:
    return RiskContinuity(load=load, history=history, found_state_hash=found)


def assess_risk_continuity(
    *,
    load: RiskStateLoad,
    snapshot: Mapping[str, Any],
    state_dir: str | Path,
    history: LogRiskHistory | None = None,
    limits: RiskLimits | None = None,
    ledger_equity: Decimal | None = None,
) -> RiskContinuity:
    """R1-c's verdict: is this ``risk.json`` the continuation of this log?

    ``load`` and ``snapshot`` are :attr:`chimera.risk.RiskEngine.load_outcome`
    and :attr:`chimera.risk.RiskEngine.loaded_snapshot` -- the READ's answer and
    the bytes it found, taken before anything this process does can move either.
    A caller that re-derives them from a live engine is asking a different
    question: an operator who engaged the kill switch between two runs, or a
    first start that has just seeded its capital, has moved the state
    legitimately, and comparing THAT against the log would report the move as a
    continuity failure.

    ``history`` is injectable so a test can drive each branch against a log it
    really built, and so a caller holding one need not read the log twice.

    ``limits`` and ``ledger_equity`` (the campaign's limits and
    :func:`chimera.demo.risk_wiring.ledger_equity`) are what the ``equity``
    crash-window proof needs; see :func:`_crash_transition`. Without them that
    one proof is not attempted and the finding stands, which is the safe side.

    The rules, in the order they are applied:

    1. **A log with no records is a log with no claim.** Whatever the file says,
       there is nothing to be continuous with: no dispute. This is what keeps a
       genuine first start a first start, and it is the negative control the
       whole item stands on.
    2. **``UNREADABLE`` with history** -- the file exists and could not be
       believed. R1-b has already failed the engine closed and is preserving the
       bytes; R1-c records that the campaign it belongs to has a history the
       file can no longer account for.
    3. **``MISSING`` with history** -- the file is gone and the log says the
       campaign had one. NOT a first start, so the configured capital may not
       seed anything.
    4. **``LEGACY`` with history** (``RISK_STATE_PRE_SCHEMA``) -- the pre-schema
       document carries a halt claim and no account state at all. It cannot be the state a
       ``risk.state_hash`` was taken of, and reading it as an ordinary
       ``MISSING`` (seed capital) or as a trustworthy current state (believe its
       halt and invent the rest) are both fabrications. Fail closed and say so.
    5. **``LOADED`` against the log's last statement** -- see
       :class:`LogRiskStatement`. A ``STATE_HASH`` statement must match exactly;
       a ``HALTED`` statement must find a halted file; ``RUNNING``, ``MOVED`` and
       ``NONE`` make no claim this can check.

    A log that does not verify (``trustworthy`` false) can still establish rules
    2-4, because those depend on nothing but the existence of records. Its
    statements are not used for rule 5: the campaign halts on
    ``LOG_FORGED`` through the runner's own triage, which is the finding that
    matters, and a log nobody can trust may not condemn a state file either.
    """
    root = Path(state_dir)
    history = read_log_risk_history(root) if history is None else history
    no_account_claim = load in (RiskStateLoad.MISSING, RiskStateLoad.UNREADABLE)
    found = "" if no_account_claim else risk_state_hash(snapshot)

    if not history.has_history:
        return _no_dispute(load, history, found)

    fault: RiskContinuityFault | None = None
    detail = ""
    transition = ""
    # The FILE, never the path to it. A halt reason is hashed into
    # `risk.state_hash` (`RiskState.snapshot` carries `halt_reason`), and
    # `RiskEngine._load_state` already refuses to put a path in one for that
    # reason: "a path or errno string would differ between hosts in the same
    # semantic state". R1-b's equity dispute names no path either. A campaign
    # has exactly one risk state file and the runbook says where it lives, so
    # the name is what an operator needs; the resolved path goes to the logger.
    where = "risk.json"
    seen = f"the decision log holds {history.records} verified record(s)"

    if load is RiskStateLoad.UNREADABLE:
        fault = RiskContinuityFault.RISK_STATE_UNREADABLE
        detail = (
            f"the persisted {where} could not be read and {seen}. The "
            "campaign has a history this file can no longer account for, so the file "
            "is preserved exactly as it is, nothing is seeded, and the campaign does "
            "not run"
        )
    elif load is RiskStateLoad.MISSING:
        fault = RiskContinuityFault.RISK_STATE_ABSENT
        detail = (
            f"there is no persisted {where} and {seen}. This is not a "
            "first start: the equity, the peak, the day's baseline, the streaks, the "
            "open reconciliation disputes and any halt the campaign was carrying are "
            "not recoverable from the configured capital, and seeding it would put a "
            "number nobody recorded where a history used to be"
        )
    elif load is RiskStateLoad.LEGACY:
        fault = RiskContinuityFault.RISK_STATE_PRE_SCHEMA
        detail = (
            f"the persisted {where} is the pre-schema halt record -- it "
            f"carries a halt claim and no account state at all -- and {seen}. A "
            "campaign whose log holds records had an account, so this document cannot "
            "be the state those records were written against; it is neither an absent "
            "file to seed over nor a current state to believe"
        )
    elif history.trustworthy:
        if history.statement is LogRiskStatement.STATE_HASH:
            if found != history.state_hash:
                fault = RiskContinuityFault.RISK_STATE_MISMATCH
                detail = (
                    f"the persisted {where} hashes to {found} and the "
                    f"log's last risk.state_hash, on the {history.statement_kind} "
                    f"record at seq {history.statement_seq}, is {history.state_hash}. "
                    "Nothing has been recorded since that could have moved it. "
                    "Neither side is rewritten to make them agree"
                )
                transition = _crash_transition(
                    snapshot, history, limits=limits, ledger_equity=ledger_equity
                )
                if transition:
                    detail += (
                        f". It is exactly the log's own last state carried through ONE "
                        f"production crash window ({transition}: an Aegis write persisted "
                        "and the process died before the record that restates it), so "
                        "the finding is recorded and deferred rather than sealed"
                    )
        elif history.statement is LogRiskStatement.HALTED:
            if not snapshot.get("halted"):
                fault = RiskContinuityFault.RISK_STATE_MISMATCH
                detail = (
                    f"the log's last statement about the risk state is the HALT record "
                    f"at seq {history.statement_seq} and the persisted {where} is "
                    "not halted. A halt is cleared by an operator resume, which the "
                    "log would hold; this one was cleared behind it"
                )

    if fault is None:
        return _no_dispute(load, history, found)

    verdict = RiskContinuity(
        load=load,
        history=history,
        fault=fault,
        detail=detail,
        found_state_hash=found,
        crash_transition=transition,
    )
    return replace(verdict, already_recorded=_already_recorded(verdict))


def _already_recorded(verdict: RiskContinuity) -> bool:
    """Whether this exact finding is already in the log, unanswered.

    Two conditions, and both are needed.

    The recorded finding must have the same :attr:`RiskContinuity.identity` --
    otherwise a file that has changed since (appeared, vanished, stopped
    parsing, been replaced) would be silently folded into the old record.

    And nothing may have RESTATED the risk state since that record was written.
    A campaign that was repaired, ran again and then lost its ``risk.json`` a
    second time has a ``DECISION`` record in between, so the second loss is
    recorded as the new event it is. Without this a fault that recurred after a
    successful repair would be recorded once, for ever.
    """
    recorded = verdict.history.recorded_dispute
    if recorded is None:
        return False
    seq, recorded_identity = recorded
    if recorded_identity != verdict.identity:
        return False
    restated = verdict.history.last_restated_seq
    return restated is None or restated < seq


__all__ = [
    "CRASH_TRANSITIONS",
    "RISK_CONTINUITY_SEALED_FIELD",
    "RISK_CONTINUITY_PREFIX",
    "RISK_HASH_EXCLUDED",
    "LogRiskHistory",
    "LogRiskStatement",
    "RiskContinuity",
    "RiskContinuityFault",
    "assess_risk_continuity",
    "read_log_risk_history",
    "risk_state_hash",
]
