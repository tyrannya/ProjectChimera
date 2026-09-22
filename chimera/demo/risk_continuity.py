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
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from chimera.demo.decision_log import (
    LOG_DIR_NAME,
    RecordKind,
    day_files,
    read_records,
    verify_log,
)
from chimera.demo.rules import canonical_hash
from chimera.risk import RiskStateLoad

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


def _halt_transition_explains_mismatch(
    snapshot: Mapping[str, Any], history: "LogRiskHistory"
) -> bool:
    """Whether a bare, first halt is PROVABLY the only difference from the log.

    A crash between :meth:`chimera.risk.RiskEngine.halt`'s persist and
    ``DemoRunner._halt``'s ``HALT`` append (:meth:`RiskEngine.check_kill_switch`,
    a rule exception, a dispute, and every other halt site) leaves ``risk.json``
    halted while the log's last risk statement is still the ``STATE_HASH`` from
    before it -- the hash comparison below sees exactly what a swapped file
    looks like, for a reason no different from section 9.3's own crash windows.

    Unlike an equity-moving crash window, this one is provable rather than
    merely plausible, because :meth:`chimera.risk.RiskEngine.halt` touches
    exactly two fields (``halted``, ``halt_reason``) and only when
    ``self.state.halted`` was ``False`` a moment before -- it returns
    immediately otherwise, "idempotent: re-halting does not re-alert". So
    whenever the log's last statement is a ``STATE_HASH`` (never a ``HALTED``
    or a ``RUNNING`` one) and this crash window is what produced the found
    file, THAT record's own state was ``halted: False``, because a halt that
    was already on would have made ``RiskEngine.halt`` a no-op and left no
    crash window to fall into. Reverting the found snapshot's ``halted`` and
    ``halt_reason`` to that known prior value and re-hashing therefore either
    reproduces the log's own hash EXACTLY -- proving nothing else moved -- or
    it does not, in which case some other field also differs and this returns
    ``False`` as it must: a foreign or stale file that merely happens to be
    halted is not let through by only checking the one field a legitimate
    crash could explain.

    This is deliberately narrower than "the file is halted": an equity change,
    a peak, a streak or a reconciliation dispute all being unmoved is exactly
    what is being proved, not assumed.
    """
    if history.statement is not LogRiskStatement.STATE_HASH:
        return False
    if not snapshot.get("halted"):
        return False
    reverted = dict(snapshot)
    reverted["halted"] = False
    reverted["halt_reason"] = ""
    return risk_state_hash(reverted) == history.state_hash


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

        With one exception, and it is R1-c's own doing: a ``HALT`` appended while
        the engine was SEALED by a continuity dispute did not persist anything,
        so it claims nothing about the file. See ``sealed`` in
        :func:`read_log_risk_history`.

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
        while the engine was SEALED claims nothing either, because the seal
        also refused the write it would otherwise describe -- a `flatten`
        issued during a continuity halt persists no equity to `risk.json`. See
        ``sealed`` in :func:`read_log_risk_history`.

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
#: ``SKIPPED_STALE`` is deliberately absent: ``DemoRunner._skipped_stale`` says
#: in as many words that "nothing else about the position, the ledger or Aegis
#: moves", and it is appended from ``catch_up`` without a tick. ``STARTUP``,
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
    #: Whether a bare, first halt is PROVABLY the only difference between the
    #: found file and the log's last ``STATE_HASH``. See
    #: :func:`_halt_transition_explains_mismatch`.
    halt_transition_explains_mismatch: bool = False

    @property
    def disputed(self) -> bool:
        return self.fault is not None

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
        crash.

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

    #: True from an R1-c RECOVERY record until the next record that restates the
    #: hash. While it is set the engine was SEALED -- `halt_for_continuity_dispute`
    #: halts in memory and writes nothing -- so the HALT records the disputed
    #: restarts append correspond to no write at all, and reading one as "the
    #: campaign is halted, and the file should say so" would be R1-c condemning
    #: a file for the halt R1-c itself raised. It would also make a REPAIR
    #: impossible: restore the right `risk.json` and the log's own halt would
    #: disagree with it for ever.
    #:
    #: The same reasoning covers ``MOVED`` (``OPERATOR``/``INCOMPLETE_STATE``)
    #: while sealed. `RiskEngine._persist` refuses every write while
    #: `_continuity_disputed` is set, so a `flatten` issued during the seal --
    #: permitted, because "HALT is what it is for" -- persists nothing to
    #: `risk.json` either, and its `OPERATOR` record is no more a statement
    #: about the file than the HALT record beside it. Reading it as `MOVED`
    #: ("the log makes no comparable claim") let a flatten during the seal erase
    #: the seal itself: the very next restart saw the log's newest risk
    #: statement as `MOVED`, ran rule 5's `STATE_HASH`/`HALTED` branches never,
    #: found no fault, and unsealed a `risk.json` nobody had repaired.
    sealed = False

    for path in day_files(root):
        for record in read_records(path):
            seq = record.get("seq")
            seq = int(seq) if isinstance(seq, int) else None
            found, record_hash = _statement_of(record)
            if found is LogRiskStatement.STATE_HASH:
                sealed = False
            elif sealed and found in (
                LogRiskStatement.HALTED,
                LogRiskStatement.RUNNING,
                LogRiskStatement.MOVED,
            ):
                # Not a statement about the file. The previous one stands.
                found = LogRiskStatement.NONE
            if found is not LogRiskStatement.NONE:
                statement = found
                statement_kind = str(record.get("kind", ""))
                statement_seq = seq
                state_hash = record_hash
                if found is LogRiskStatement.STATE_HASH:
                    last_restated_seq = seq
            identity = _recorded_dispute(record)
            if identity is not None and seq is not None:
                recorded = (seq, identity)
                sealed = True

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
    )


def _no_dispute(load: RiskStateLoad, history: LogRiskHistory, found: str) -> RiskContinuity:
    return RiskContinuity(load=load, history=history, found_state_hash=found)


def assess_risk_continuity(
    *,
    load: RiskStateLoad,
    snapshot: Mapping[str, Any],
    state_dir: str | Path,
    history: LogRiskHistory | None = None,
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
    halt_explains = False
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
                halt_explains = _halt_transition_explains_mismatch(snapshot, history)
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
        halt_transition_explains_mismatch=halt_explains,
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
