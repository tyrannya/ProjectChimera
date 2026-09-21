"""R1-c (AEG-4): whether the ``risk.json`` on disk continues the log beside it.

The store and the carry ledger already refuse to let their own absence be read
as a fact about the campaign. :class:`chimera.futures.store.FuturesStore` starts
UNBOOTSTRAPPED rather than flat, :meth:`chimera.carry.ledger.CarryLedger.open`
loads UNREADABLE rather than replacing a file it could not parse, and
:meth:`chimera.demo.runner.DemoRunner._ledger_regressed_against_the_log` halts a
campaign whose ledger holds less than the log has already committed. ``risk.json``
had none of it. A **missing** one loaded as
:attr:`chimera.risk.RiskStateLoad.MISSING`, which
:func:`chimera.demo.risk_wiring.seed_or_reconcile_equity` correctly reads as a
genuine first start -- and on a campaign that had already halted, run and
persisted, deleting one file therefore cleared the halt, reset the peak equity,
the day, the cooldown, the order window, the funding streak and every open
reconciliation dispute, and left nothing anywhere saying that it had happened.
That is the audit's AEG-4, measured end to end: halt a campaign, remove
``risk.json``, start again, and the runner reaches READY unhalted with equity
re-seeded to the configured capital.

This module is the missing comparison. It answers one question --

    *does the persisted risk state continue the decision log that sits beside
    it?*

-- and it answers it from the state **as loaded**, before anything has seeded,
reconciled, repaired or bootstrapped. Nothing here writes a file, opens a socket,
reads a clock or mutates a :class:`chimera.risk.RiskEngine`. What to DO about a
verdict is the runner's; see :meth:`chimera.demo.runner.DemoRunner.start`.

**A disputed file is evidence, and nothing a disputed process does may write
it.** The runner takes this verdict before it builds Aegis, and when the verdict
is a dispute it builds Aegis with ``persist=False``
(:meth:`chimera.risk.RiskEngine._persist`). The engine still loads, seeds,
reconciles, halts and flattens -- in memory, which is what the runner needs to
refuse and to reduce exposure -- and ``risk.json`` keeps exactly the bytes, or
exactly the absence, it was found with. That is what makes every verdict below
REPEATABLE: the next process reads the same file and reaches the same answer,
whatever the processes in between did. An earlier revision let a disputed
process keep writing -- the seed over a missing file, the kill-switch mirror,
`flatten`'s equity and exposures, a reconciliation note -- and then had to guess
from the changed file whether it had been restored. It guessed "a different file
means a restore", and the independent review measured a permitted `flatten`
dissolving the dispute through the CLI, and a placeholder seeded before a crash,
a failed append or a forged log standing in for the missing state afterwards.

**A hashless record may not switch the comparison off.** Only DECISION, FUNDING,
RECONCILIATION and LIQUIDATION_TOUCH carry ``risk.state_hash``; HALT, RESUME,
OPERATOR, STARTUP, SHUTDOWN, INCOMPLETE_STATE, SKIPPED_STALE and RECOVERY carry
none. The scan walks past **every** record that carries no hash and takes the
newest one that does. That record pins an identity, and the question the
comparison really asks is directional:

``the loaded identity IS the newest witness's``
    the file is the state the campaign last recorded. Continuous.
``the loaded identity is one an EARLIER witness records``
    the campaign has already moved past this state: an older copy restored, a
    file rolled back, a state replaced. Every persist precedes the record that
    quotes it, so no crash produces this direction. **Refused.**
``the loaded identity is one NO witness records``
    either the ordinary crash window -- ``tick`` persists the risk state and then
    commits the DECISION that quotes it, and a process killed between the two
    leaves precisely this -- or a hashless record after the witness legitimately
    moved the state and said nothing about where to. See ``STATE_AHEAD_OF_LOG``,
    and ``NOT_SETTLED`` for why neither excuse is available after a refusal.

**A halt is normalised away before the identities are compared, because
``RiskEngine.halt`` is the one transition the log does not record and CAN be
inverted.** ``halted`` and ``halt_reason`` are inside ``risk.state_hash`` and a
HALT record carries no hash, so a halted state never hashes to any witness.
:meth:`chimera.risk.RiskEngine.halt` sets exactly those two fields and nothing
else, so the identity the same state had *before* it was halted is computable,
and both identities are compared. No other transition is inverted: RESUME's and
each OPERATOR command's effects are not reconstructed here.

**Nothing a disputed process wrote is campaign history.** A refusal is written
down as a continuity ``RECOVERY`` record, and every record after it until the
matching settlement -- the continuity ``RECOVERY`` a later start writes when the
file on disk positively continues the log again -- was written by a process
holding the risk state as evidence: its HALT records halted an engine that wrote
nothing, and its OPERATOR records moved nothing on disk. Reading one of them back
as "this campaign is halted" or "this campaign's state was moved" would condemn
the correct restore, or excuse an incorrect one. So that stretch is skipped, and
the settlement record is what ends it: the settled process's own halts are
ordinary history again, and a later loss is a new event with a record of its
own. See :func:`_scan`.

**A refusal is settled by a positive match, never by a change.** While a refusal
is open, the file is compared against the campaign's history exactly as it would
be on any start, with one difference: an identity that no record quotes is not
the crash window any more, because no process could have written it -- a
disputed process writes nothing -- and so it proves nothing. The refusal is
settled only by a file the log itself vouches for: the newest witness's state,
or that state with a halt written into it; or, where the campaign's own history
before the refusal moved the state by a hashless command, a state consistent
with that; or, on a campaign that has never recorded an identity at all, one
that holds whatever halt the log says it held.

**A disagreement has a DIRECTION, and only one of them fails closed.** This is
what "the same continuity the store and ledger already have" means. The carry
ledger is refused when it holds **less** than the log has already committed and
recovered from when it holds **more** (section 9.3's ``LOG_BEHIND_STATE``).
``risk.json`` gets exactly that treatment outside a refusal, and
`tests/test_demo_runner.py`'s two-sided control refuses the alternative on the
ledger's behalf.

The limits are stated rather than implied. Outside a refusal, a ``risk.json``
edited to a state no witness records is indistinguishable from the crash window
and is accepted as ahead, exactly as an edited store or ledger is. A campaign
whose last risk-moving record before a refusal is hashless (``flatten``,
``resume``, ``resolve``) cannot tell its correct post-command backup from any
other state no record quotes; giving those records a hash is R1-j's. And a
campaign that has never recorded a ``risk.state_hash`` cannot tell its genuine
first-start state from another one holding the same halt -- the genuine one IS a
default seed. What continuity buys is that a state which is *absent*,
*unreadable*, *unhalted where the log records a halt*, *demonstrably older than
the log*, or *unvouched for after a refusal* can no longer pass, and that no
process fabricates one.

Nothing here creates scientific evidence, a prospective boundary, an alpha claim
or any real-money authority.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Mapping

from chimera.demo.decision_log import (
    LOG_DIR_NAME,
    RecordKind,
    day_files,
    read_records,
)
from chimera.risk import RiskState, RiskStateLoad, state_snapshot

__all__ = [
    "RISK_CONTINUITY_PREFIX",
    "RISK_HASH_EXCLUDED",
    "ContinuityOutcome",
    "RiskContinuity",
    "Witness",
    "continuity_already_recorded",
    "evaluate_risk_continuity",
    "halt_free_state_hash",
    "is_risk_continuity_halt",
    "risk_state_hash",
]


#: What a continuity refusal's halt reason begins with. A prefix, because the
#: reason names the outcome and the witness after it. The same shape
#: :data:`chimera.demo.risk_wiring.EQUITY_DISPUTE_PREFIX` uses, and for the same
#: purpose: the runner and an operator tool recognise the halt by asking this
#: module rather than by matching a sentence that a later rewording would break.
RISK_CONTINUITY_PREFIX = "risk_continuity:"

#: Risk-state fields that are WALL CLOCK or HOST DATE rather than decision
#: semantics, and are therefore outside ``risk.state_hash``.
#:
#: ``order_times`` is the rate limiter's record of when orders were placed and
#: ``cooldown_until`` is a deadline computed from one; ``day`` is the date the
#: host happened to be running on. All three move between a live run and a replay
#: of the same files, and none of them is a fact about the decision -- so hashing
#: them would make section 10's byte comparison fail for a reason that has
#: nothing to do with whether the two runs decided alike.
#:
#: Everything that governs permission and IS reproducible stays in: ``halted``,
#: ``halt_reason``, ``kill_switch``, the equity series, the streaks, and the
#: reconciliation disputes. Widening this set is how a stale state stops being
#: distinguishable from a current one, which is why
#: `tests/test_r1c_risk_continuity.py` pins each of those fields individually.
RISK_HASH_EXCLUDED = frozenset({"order_times", "cooldown_until", "day"})

#: The kinds that can move the risk state without recording where to.
#:
#: Every kind :class:`chimera.demo.decision_log.RecordKind` admits, and what it
#: can do to a :class:`chimera.risk.RiskState`. The scan's behaviour follows from
#: this table and from nothing else, so a kind added later has to be added here
#: before it can be treated as anything:
#:
#: ===================== ======== =======================================
#: kind                  hash?    what it can move
#: ===================== ======== =======================================
#: DECISION              yes      anything; the hash pins the result
#: FUNDING               yes      the streak and the equity; pinned
#: RECONCILIATION        yes      the disputes; pinned
#: LIQUIDATION_TOUCH     yes      halts Aegis; pinned (the halt is inside)
#: HALT                  no       ``halted`` + ``halt_reason``, and nothing
#:                                else -- the one transition that is exactly
#:                                invertible; see `halt_free_state_hash`
#: RESUME                no       clears the halt, the kill-switch mirror,
#:                                ``funding_halt`` and the adverse streak
#: OPERATOR              no       ``flatten`` -> ``update_equity``;
#:                                ``resolve`` -> one dispute cleared;
#:                                ``resolve-equity`` ->
#:                                ``adopt_reconciled_equity``
#: STARTUP               no       nothing
#: SHUTDOWN              no       nothing
#: RECOVERY              no       nothing
#: INCOMPLETE_STATE      no       nothing
#: SKIPPED_STALE         no       nothing
#: ===================== ======== =======================================
#:
#: The three below are the hashless kinds that CAN move it. They do not END the
#: backward scan, but a state whose identity no witness records is told apart
#: from the crash window by whether one of them is in the campaign's own tail.
#: Giving them a ``risk.state_hash`` would close that residual and is a change to
#: what the runner WRITES, which reaches replay parity; it is R1-j's, not this
#: item's.
_RISK_MOVING_KINDS = frozenset(
    {RecordKind.HALT.value, RecordKind.RESUME.value, RecordKind.OPERATOR.value}
)

#: The records that can leave a halt CLEARED, so that a HALT before one of them
#: no longer proves anything about the loaded state.
#:
#: ``RESUME`` is :meth:`chimera.risk.RiskEngine.resume`. The OPERATOR command is
#: ``resolve-equity``, which reaches
#: :meth:`chimera.risk.RiskEngine.adopt_reconciled_equity` -- the one other
#: method that clears ``halted``. ``flatten`` and ``resolve --symbol`` are
#: deliberately absent: neither clears a halt, so a HALT that one of them follows
#: is still in force and still has to be held.
_HALT_CLEARING_COMMANDS = frozenset({"resolve-equity"})


def is_risk_continuity_halt(reason: str) -> bool:
    """Whether ``reason`` is the halt THIS module's verdict produces.

    Asked rather than matched, so a rewording of the reason cannot leave the
    campaign holding a halt that no code admits to recognising. The runner's
    refusal halts on it; the file never carries it, because the engine that halts
    on it writes nothing.
    """
    return isinstance(reason, str) and reason.startswith(RISK_CONTINUITY_PREFIX)


def risk_state_hash(state: RiskState) -> str:
    """``sha256:`` over the decision-relevant fields of one risk state.

    The identity the decision log records as ``risk.state_hash`` and the identity
    of the bytes in ``risk.json`` are the same function of the same fields, which
    is what makes comparing them meaningful at all. It is computed from a
    :class:`chimera.risk.RiskState` rather than from a live engine because the
    comparison has to be made against the state AS LOADED: by the time an engine
    exists, ``build_risk_engine`` has already seeded a first start or R1-b's
    reconciliation has already halted on a disagreement, and either would have
    moved the very identity under test.
    """
    from chimera.demo.rules import canonical_hash

    snapshot = state_snapshot(state)
    return canonical_hash(
        {
            key: value
            for key, value in snapshot.items()
            if isinstance(key, str) and key not in RISK_HASH_EXCLUDED
        }
    )


def halt_free_state_hash(state: RiskState) -> str:
    """``risk_state_hash`` of the same state as it was before it was halted.

    :meth:`chimera.risk.RiskEngine.halt` sets ``halted`` and ``halt_reason`` and
    touches nothing else, so this really is its inverse rather than an
    approximation of one. A HALT record carries no ``risk.state_hash``, so a
    halted state's own identity is in no witness -- without this the log could
    say nothing at all about a halted file, which is how an older halted state
    used to pass a newer HALT history merely for being halted too.
    """
    return risk_state_hash(replace(state, halted=False, halt_reason=""))


class ContinuityOutcome(str, Enum):
    """What the comparison found. A bounded vocabulary; the record quotes it."""

    #: The log holds no record, so there is no campaign for a risk state to
    #: continue. A genuine first start, whether or not ``risk.json`` exists --
    #: ``build_risk_engine`` writes one BEFORE the first STARTUP record, so a
    #: file with an empty log beside it is the ordinary shape of a first start
    #: that was interrupted, not a discontinuity.
    FIRST_START = "FIRST_START"
    #: History exists and nothing refuses the persisted state. Either its
    #: identity is the newest witness's, or no witness records it and a hashless
    #: record in the campaign's tail legitimately moved it. Written into the log
    #: as the SETTLEMENT of a refusal when one is open; see
    #: :attr:`RiskContinuity.settles_seq`.
    CONTINUOUS = "CONTINUOUS"
    #: Whether the log holds durable history could not be established: day files
    #: hold committed lines that no record could be read from. Reported as its
    #: own answer rather than as ``FIRST_START``, which would tell an operator
    #: that a campaign whose history is unreadable has none.
    HISTORY_UNVERIFIABLE = "HISTORY_UNVERIFIABLE"
    #: History exists and there is no ``risk.json``. The audit's AEG-4.
    RISK_STATE_MISSING = "RISK_STATE_MISSING"
    #: History exists and ``risk.json`` could not be read or believed.
    RISK_STATE_UNREADABLE = "RISK_STATE_UNREADABLE"
    #: History exists and ``risk.json`` is the pre-schema halt record, which
    #: carries no account state at all.
    RISK_STATE_WITHOUT_ACCOUNT = "RISK_STATE_WITHOUT_ACCOUNT"
    #: The log's newest halt witness is a HALT that nothing cleared, and the
    #: loaded state is not halted.
    HALT_NOT_HELD = "HALT_NOT_HELD"
    #: The loaded state's identity -- as loaded, or as it was before a halt was
    #: written into it -- is one the log records at an EARLIER witness and not at
    #: the newest: the campaign has already moved past this state, so the file
    #: was rolled back, restored from an older copy or replaced. The risk-state
    #: counterpart of ``ledger_behind_log``, and refused for the same reason:
    #: every persist precedes the record that quotes it, so no crash produces
    #: this direction.
    STATE_HASH_REGRESSED = "STATE_HASH_REGRESSED"
    #: The loaded state's identity is one NO witness records, nothing in the
    #: campaign's tail moved it without a hash, and a continuity refusal is open.
    #: Without the refusal this is ``STATE_AHEAD_OF_LOG`` -- the crash window.
    #: With it, it is a file that appeared after a process found the risk state
    #: in dispute, and no process could have written it: a disputed process
    #: writes nothing. It is not evidence that the state was restored, so the
    #: refusal stands.
    NOT_SETTLED = "NOT_SETTLED"
    #: The loaded state's identity is one NO witness records, and no hashless
    #: record moved it since the newest one. The ordinary crash window: the state
    #: was persisted and the record quoting it was not. Not a dispute;
    #: :meth:`chimera.demo.runner.DemoRunner._state_ahead_of_log` reports it as
    #: section 9.3's ``LOG_BEHIND_STATE`` and the campaign recovers. Never
    #: reached while a refusal is open; that is ``NOT_SETTLED``.
    STATE_AHEAD_OF_LOG = "STATE_AHEAD_OF_LOG"


#: The outcomes that refuse to start. Stated as a set rather than as "everything
#: that is not CONTINUOUS", so an outcome added later is a deliberate decision
#: about whether a campaign may run rather than a default -- and so that
#: ``STATE_AHEAD_OF_LOG``'s absence from it is visible as the decision it is.
_DISPUTED_OUTCOMES = frozenset(
    {
        ContinuityOutcome.HISTORY_UNVERIFIABLE,
        ContinuityOutcome.RISK_STATE_MISSING,
        ContinuityOutcome.RISK_STATE_UNREADABLE,
        ContinuityOutcome.RISK_STATE_WITHOUT_ACCOUNT,
        ContinuityOutcome.HALT_NOT_HELD,
        ContinuityOutcome.STATE_HASH_REGRESSED,
        ContinuityOutcome.NOT_SETTLED,
    }
)


@dataclass(frozen=True)
class Witness:
    """One log record, in the only four respects this comparison uses."""

    seq: int
    kind: str
    #: ``risk.state_hash`` when the record carries one, otherwise ``""``.
    state_hash: str = ""
    #: ``operator.command`` on an OPERATOR record, otherwise ``""``.
    command: str = ""

    def describe(self) -> str:
        """The witness in words, for a halt reason that must read the same on
        every host: a kind, a sequence number and a command, and no path."""
        named = f"{self.kind} record seq {self.seq}"
        return f"{named} (`{self.command}`)" if self.command else named


@dataclass(frozen=True)
class RiskContinuity:
    """The verdict, everything it was taken from, and what it means to do."""

    outcome: ContinuityOutcome
    #: The runner's halt reason, already prefixed. Empty when nothing is refused.
    reason: str
    load: RiskStateLoad
    #: The loaded state's identity, or ``""`` when there was no state to hash.
    observed_state_hash: str = ""
    #: The identity the same state had before a halt was written into it, or
    #: ``""`` when the state is not halted and the two would be equal.
    unhalted_state_hash: str = ""
    hash_witness: Witness | None = None
    halt_witness: Witness | None = None
    #: The EARLIER record whose ``risk.state_hash`` one of the loaded state's two
    #: identities still carries, when there is one. What tells a rolled-back file
    #: apart from the ordinary crash window, and the evidence the refusal quotes.
    matched_witness: Witness | None = None
    #: Whether the log holds any record at all.
    has_history: bool = False
    #: The ``seq`` of the continuity refusal this verdict SETTLES: set only on a
    #: ``CONTINUOUS`` verdict taken while one is open. The runner writes the
    #: settlement down, which is what ends the stretch :func:`_scan` skips.
    settles_seq: int | None = None

    @property
    def disputed(self) -> bool:
        """Whether this verdict refuses to let the campaign run."""
        return self.outcome in _DISPUTED_OUTCOMES

    @property
    def ahead_of_log(self) -> bool:
        """Whether the state is the ordinary crash window's, and not a dispute.

        The runner asks this from :meth:`DemoRunner._state_ahead_of_log`, beside
        the legs and the ledger, so one crash is reported once and under section
        9.3's own name.
        """
        return self.outcome is ContinuityOutcome.STATE_AHEAD_OF_LOG

    def block(self) -> dict[str, Any]:
        """The ``recovery.risk_continuity`` block of the RECOVERY record.

        Every hash is written under a ``*_hash`` name, which
        :func:`chimera.demo.decision_log._validate_hashes` holds to
        ``sha256:<hex>`` -- so a hash that does not exist is OMITTED rather than
        written as ``null``, which that validator refuses and which would in any
        case be a claim that the state had an identity of ``None``.
        """
        block: dict[str, Any] = {
            "outcome": self.outcome.value,
            "risk_state_load": self.load.value,
            "witness_kind": None if self.hash_witness is None else self.hash_witness.kind,
            "witness_seq": None if self.hash_witness is None else self.hash_witness.seq,
            "halt_witness_kind": (
                None if self.halt_witness is None else self.halt_witness.kind
            ),
            "halt_witness_seq": None if self.halt_witness is None else self.halt_witness.seq,
            "matched_witness_seq": (
                None if self.matched_witness is None else self.matched_witness.seq
            ),
            "fingerprint_hash": self.fingerprint,
        }
        if self.observed_state_hash:
            block["observed_state_hash"] = self.observed_state_hash
        if self.unhalted_state_hash:
            block["unhalted_state_hash"] = self.unhalted_state_hash
        if self.hash_witness is not None and self.hash_witness.state_hash:
            block["witness_state_hash"] = self.hash_witness.state_hash
        if self.settles_seq is not None:
            block["settles_seq"] = self.settles_seq
        return block

    @property
    def fingerprint(self) -> str:
        """What makes two detections of this discontinuity the same one.

        A start that refuses writes one RECOVERY record; a start that refuses
        again for the same reason must not write a second. The identity is the
        verdict plus what it was taken from -- never a timestamp and never a seq
        of this process's own records -- so repeated starts against an unchanged
        pair of files produce the same value and repeated starts against a
        CHANGED one do not. The files stay unchanged across a disputed process
        because nothing it does is persisted, and the witnesses stay unchanged
        because everything it wrote is skipped (:func:`_scan`).
        """
        from chimera.demo.rules import canonical_hash

        return canonical_hash(
            {
                "outcome": self.outcome.value,
                "risk_state_load": self.load.value,
                "observed_state_hash": self.observed_state_hash,
                "unhalted_state_hash": self.unhalted_state_hash,
                "witness_kind": None if self.hash_witness is None else self.hash_witness.kind,
                "witness_seq": None if self.hash_witness is None else self.hash_witness.seq,
                "witness_state_hash": (
                    "" if self.hash_witness is None else self.hash_witness.state_hash
                ),
                "halt_witness_kind": (
                    None if self.halt_witness is None else self.halt_witness.kind
                ),
                "halt_witness_seq": (
                    None if self.halt_witness is None else self.halt_witness.seq
                ),
                "matched_witness_seq": (
                    None if self.matched_witness is None else self.matched_witness.seq
                ),
            }
        )


@dataclass(frozen=True)
class _LogFacts:
    """Everything one backward pass over the log tells this comparison."""

    has_history: bool
    #: Day files hold committed lines that produced no record at all. Only
    #: computed when the pass found nothing, which is the one case where it
    #: changes an answer.
    unverifiable: bool
    #: The newest record carrying a ``risk.state_hash``, hashless records walked
    #: past. ``None`` when the log has never recorded an identity.
    hash_witness: Witness | None
    #: The newest HALT, RESUME or ``resolve-equity`` NEWER than the hash witness,
    #: among the records that are campaign history.
    halt_witness: Witness | None
    #: Whether a hashless record that can move the risk state sits between the
    #: hash witness and the end of the log, among the records that are campaign
    #: history.
    moved_since_witness: bool
    #: The ``seq`` of the continuity refusal that opened the stretch still open
    #: at the end of the log, or ``None`` when no refusal is open.
    open_refusal_seq: int | None


def _records_newest_first(state_dir: str | Path) -> Iterator[Mapping[str, Any]]:
    """Every committed record of the log, newest first.

    Day files in reverse name order -- which :func:`day_files` documents as chain
    order -- and each file's records reversed within it. One file is materialised
    at a time and the caller stops as soon as it has what it needs, so an ordinary
    start reads the newest day and no more.

    :func:`read_records` stops at the first line it cannot read, which is exactly
    right here: a torn tail is a record that was never committed, and this runs
    BEFORE :func:`chimera.demo.decision_log.recover_tail` removes the bytes. A
    record the writer never finished may not witness anything. What it cannot
    tell apart is a torn tail from a corruption earlier in the file, and
    :func:`_history_unverifiable` is what stops that from being read as "this
    campaign has no history".
    """
    for path in reversed(day_files(Path(state_dir) / LOG_DIR_NAME)):
        for record in reversed(list(read_records(path))):
            yield record


def _history_unverifiable(state_dir: str | Path) -> bool:
    """Whether day files hold committed lines that produced no record.

    A line the writer never finished has no trailing newline and is an ordinary
    torn tail -- section 9.3's TORN_TAIL, which `recover_tail` removes and which a
    genuine first start interrupted mid-STARTUP leaves behind. A line that IS
    newline-terminated and still cannot be read is a corruption, and a log with
    one of those has history that nobody can establish. Saying ``FIRST_START``
    about it would be an affirmative claim that the campaign has none.

    Only consulted when the backward pass found no record at all, which is the
    only case where the distinction changes an answer.
    """
    for path in day_files(Path(state_dir) / LOG_DIR_NAME):
        with open(path, "rb") as handle:
            committed = sum(1 for raw in handle if raw.endswith(b"\n") and raw.strip())
        if committed > sum(1 for _ in read_records(path)):
            return True
    return False


def _witness(record: Mapping[str, Any]) -> Witness:
    """One record as the comparison sees it."""
    risk = record.get("risk")
    state_hash = ""
    if isinstance(risk, Mapping):
        found = risk.get("state_hash")
        if isinstance(found, str):
            state_hash = found
    operator = record.get("operator")
    command = ""
    if isinstance(operator, Mapping):
        found = operator.get("command")
        if isinstance(found, str):
            command = found
    seq = record.get("seq")
    return Witness(
        seq=int(seq) if isinstance(seq, int) else -1,
        kind=str(record.get("kind", "")),
        state_hash=state_hash,
        command=command,
    )


def _clears_a_halt(witness: Witness) -> bool:
    """Whether this record can have left Aegis unhalted."""
    if witness.kind == RecordKind.RESUME.value:
        return True
    return (
        witness.kind == RecordKind.OPERATOR.value
        and witness.command in _HALT_CLEARING_COMMANDS
    )


def _continuity_block(record: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """This record's ``recovery.risk_continuity`` block, when it is one of ours."""
    if str(record.get("kind", "")) != RecordKind.RECOVERY.value:
        return None
    recovery = record.get("recovery")
    if not isinstance(recovery, Mapping):
        return None
    block = recovery.get("risk_continuity")
    return block if isinstance(block, Mapping) else None


def _settles_a_refusal(block: Mapping[str, Any]) -> bool:
    """Whether this continuity record is a SETTLEMENT rather than a refusal.

    Only ``CONTINUOUS`` settles, and only that is recognised: an outcome this
    build does not know is read as a refusal, so a record written by some later
    vocabulary can never open the log up by being misread.
    """
    return block.get("outcome") == ContinuityOutcome.CONTINUOUS.value


def _scan(state_dir: str | Path) -> _LogFacts:
    """Everything the log says, in one backward pass that stops as soon as it can.

    The pass ends at the first record carrying a ``risk.state_hash``, and that is
    what makes each answer well defined as well as cheap:

    * the **hash witness** is that record -- every hashless record newer than it
      is walked past rather than being allowed to end the comparison;
    * the **halt witness** is only looked for in the tail newer than it, because
      the hash pins ``halted`` and a HALT older than the hash witness can add
      nothing;
    * a **refusal** is only looked for there too: one that a hash-bearing record
      follows is one the campaign has since run past.

    An ordinary start therefore reads one record. Only a log that has never
    recorded an identity is walked in full, and such a log is short.

    **The tail is then read forwards, in stretches.** A continuity refusal opens
    a stretch and the continuity settlement after it closes it; everything in
    between -- the refusal's own HALT, a `flatten` an operator ran meanwhile, a
    kill switch that halted a process, further refusals as other files were
    tried -- was written by a process that held the risk state as evidence and
    persisted none of it, so none of it describes the file and none of it is
    read. Records before the first refusal and after a settlement are the
    campaign's own history and are read as usual. A stretch still open at the
    end of the log is the refusal nothing has settled.
    """
    has_history = False
    hash_witness: Witness | None = None
    tail: list[tuple[Mapping[str, Any], Witness]] = []

    for record in _records_newest_first(state_dir):
        has_history = True
        witness = _witness(record)
        if witness.state_hash:
            hash_witness = witness
            break
        tail.append((record, witness))

    halt_witness: Witness | None = None
    moved_since_witness = False
    open_refusal_seq: int | None = None
    for record, witness in reversed(tail):  # chain order: the newest wins
        block = _continuity_block(record)
        if block is not None:
            if _settles_a_refusal(block):
                open_refusal_seq = None
            elif open_refusal_seq is None:
                open_refusal_seq = witness.seq
            continue
        if open_refusal_seq is not None:
            continue
        if witness.kind in _RISK_MOVING_KINDS:
            moved_since_witness = True
        if witness.kind == RecordKind.HALT.value or _clears_a_halt(witness):
            halt_witness = witness

    return _LogFacts(
        has_history=has_history,
        unverifiable=(not has_history) and _history_unverifiable(state_dir),
        hash_witness=hash_witness,
        halt_witness=halt_witness,
        moved_since_witness=moved_since_witness,
        open_refusal_seq=open_refusal_seq,
    )


def _earlier_witness_carrying(
    state_dir: str | Path, identities: tuple[str, ...], *, before: Witness
) -> Witness | None:
    """The newest record BEFORE ``before`` whose ``risk.state_hash`` is one of these.

    What decides the direction of a disagreement, and therefore whether it is a
    dispute or a crash to recover from. A state the log records earlier and not
    at its newest witness is one the campaign has already moved past.

    ``seq`` is the ordering, not the file position: it is assigned by
    :meth:`chimera.demo.decision_log.DecisionLog.append` and is strictly
    increasing across the whole chain, so "before" is decidable without any
    assumption about which day file a record landed in.

    This walks the retained log, and it runs only when the newest witness already
    disagrees -- so an ordinary start stops at the first record it reads and only
    a start that is already in trouble pays for the walk.
    """
    if not identities:
        return None
    for record in _records_newest_first(state_dir):
        witness = _witness(record)
        if witness.seq >= before.seq:
            continue
        if witness.state_hash and witness.state_hash in identities:
            return witness
    return None


def continuity_already_recorded(state_dir: str | Path, fingerprint: str) -> bool:
    """Whether the log's newest continuity RECOVERY record names this same one.

    A refused campaign gets restarted -- by an operator, by a supervisor, by a
    systemd restart loop -- and the discontinuity is still there each time,
    unchanged, because the refusal wrote nothing to the file. One event deserves
    one record, so the writer asks this first.

    "Newest" means the newest RECOVERY record that carries a ``risk_continuity``
    block, refusal or settlement, and not the newest RECOVERY record of any kind:
    a torn tail repaired between two refused starts writes a RECOVERY of its own,
    and letting that one answer would append a second record for a discontinuity
    that had already been written down. A settlement answering is what makes a
    LATER loss of the same file a new event rather than a repeat of the first.
    """
    for record in _records_newest_first(state_dir):
        block = _continuity_block(record)
        if block is not None:
            return block.get("fingerprint_hash") == fingerprint
    return False


def evaluate_risk_continuity(
    state_dir: str | Path, *, load: RiskStateLoad, state: RiskState
) -> RiskContinuity:
    """Does the persisted risk state continue the log beside it? Reads only.

    ``load`` and ``state`` are the READ's own answers -- ``RiskEngine.load_outcome``
    and ``RiskEngine.state`` from a read-only inspection of ``risk.json``. Both
    have to come from the same untouched load: ``load`` because every cheaper test
    of "was there a file" is wrong in the case that matters (``Path.exists()``
    answers ``False`` for a path it merely could not examine), and ``state``
    because the identity under comparison is the one the FILE holds and not the
    one a startup has since seeded or halted.

    The order of the checks is the order of what they can prove, strongest first:
    a campaign with no history cannot have broken any, a state that is not there
    cannot be compared, and only a state that loaded is measured against a
    witness.
    """
    facts = _scan(state_dir)
    hash_witness = facts.hash_witness
    observed = risk_state_hash(state) if load is RiskStateLoad.LOADED else ""
    # Both identities the file can honestly claim: the one it holds, and -- when
    # a halt was written into it -- the one it held before that halt, which is
    # the only transition the log records without recording where to.
    unhalted = (
        halt_free_state_hash(state) if load is RiskStateLoad.LOADED and state.halted else ""
    )
    identities = tuple(identity for identity in (observed, unhalted) if identity)
    pinned = hash_witness is not None and hash_witness.state_hash in identities
    matched: Witness | None = None

    def verdict(outcome: ContinuityOutcome, reason: str = "") -> RiskContinuity:
        return RiskContinuity(
            outcome=outcome,
            reason=f"{RISK_CONTINUITY_PREFIX} {reason}" if reason else "",
            load=load,
            observed_state_hash=observed,
            unhalted_state_hash=unhalted,
            hash_witness=hash_witness,
            halt_witness=facts.halt_witness,
            matched_witness=matched,
            has_history=facts.has_history,
            settles_seq=(
                facts.open_refusal_seq if outcome is ContinuityOutcome.CONTINUOUS else None
            ),
        )

    if not facts.has_history:
        if facts.unverifiable:
            return verdict(
                ContinuityOutcome.HISTORY_UNVERIFIABLE,
                "the decision log holds committed lines that no record could be read "
                "from, so whether this campaign has durable history cannot be "
                "established and the risk state cannot be compared against it. Nothing "
                "here is repaired or removed, and risk.json is not written while this "
                "stands; verify the log (docs/demo_runbook.md, section 4)",
            )
        # Nothing has been decided under this state directory, so there is no
        # continuity to break. This is the case R1-c must NOT convert into a
        # dispute: a pristine deployment has no risk.json and no log, and
        # refusing it would mean no campaign could ever start.
        return verdict(ContinuityOutcome.FIRST_START)

    if load is RiskStateLoad.UNREADABLE:
        return verdict(
            ContinuityOutcome.RISK_STATE_UNREADABLE,
            "the decision log holds records for this campaign and risk.json could not "
            "be read, so what Aegis was enforcing cannot be established. The file is "
            "left exactly as it was found and nothing is written over it; restore the "
            "copy the campaign last wrote (docs/demo_runbook.md, section 4)",
        )

    if load is RiskStateLoad.MISSING:
        return verdict(
            ContinuityOutcome.RISK_STATE_MISSING,
            "the decision log holds records for this campaign and there is no risk.json, "
            "so a halt, a peak equity, a cooldown, an order window, a funding streak or "
            "an open reconciliation dispute would be silently reset to a default nothing "
            "recorded. An absent risk state is not evidence that the campaign had none, "
            "and nothing is written in its place; restore the copy the campaign last "
            "wrote (docs/demo_runbook.md, section 4)",
        )

    if load is RiskStateLoad.LEGACY:
        return verdict(
            ContinuityOutcome.RISK_STATE_WITHOUT_ACCOUNT,
            "the decision log holds records for this campaign and risk.json is the "
            "pre-schema halt record, which carries no equity, no peak, no day, no "
            "streak and no dispute. It cannot continue a campaign that has an account "
            "(docs/demo_runbook.md, section 4)",
        )

    if (
        facts.halt_witness is not None
        and facts.halt_witness.kind == RecordKind.HALT.value
        and not state.halted
    ):
        return verdict(
            ContinuityOutcome.HALT_NOT_HELD,
            f"the log's newest halt witness is a {facts.halt_witness.describe()} that no "
            "RESUME and no `resolve-equity` follows, and the risk state on disk is not "
            "halted. A halt is written to risk.json before the record that reports it "
            "and only an operator command clears one, so a persisted halt that has gone "
            "missing means the file was replaced, restored from an older copy or edited "
            "(docs/demo_runbook.md, section 4)",
        )

    if hash_witness is not None and not pinned:
        # The two are the same function of the same fields, so a campaign that
        # stopped cleanly restarts with them equal -- and one that stopped on a
        # halt restarts with the halt-normalised identity equal, which is why
        # both were offered. Which way they disagree is what decides whether this
        # is a refusal or a crash to recover from, and the log itself answers it:
        # a state the campaign has already moved past is one an EARLIER record
        # still records.
        matched = _earlier_witness_carrying(state_dir, identities, before=hash_witness)
        if matched is not None:
            held = (
                f"{observed}"
                if matched.state_hash == observed
                else f"{unhalted} once the halt written into it is set aside"
            )
            return verdict(
                ContinuityOutcome.STATE_HASH_REGRESSED,
                f"the risk state on disk hashes to {held}, which the log records "
                f"at the {matched.describe()} and not at its newest risk.state_hash "
                f"-- the {hash_witness.describe()}, which is "
                f"{hash_witness.state_hash}. The campaign has already moved past the "
                "state this file holds, and every persist precedes the record that "
                "quotes it, so no crash produces this: risk.json was rolled back, "
                "restored from an older copy or replaced. Restore the copy the "
                "campaign last wrote (docs/demo_runbook.md, section 4)",
            )
        if not facts.moved_since_witness:
            if facts.open_refusal_seq is not None:
                # Outside a refusal this is the crash window, and only a live
                # `tick` produces it. Inside one, no process wrote this file --
                # the refusing processes write nothing -- so an identity no record
                # quotes says nothing about whether the state was restored, and a
                # changed file is not a settled one.
                return verdict(
                    ContinuityOutcome.NOT_SETTLED,
                    "the log holds a continuity refusal (RECOVERY record seq "
                    f"{facts.open_refusal_seq}) that nothing has settled, and the risk "
                    f"state on disk hashes to {observed}, which no record quotes. That "
                    "is not evidence of a restore: a file that appears after a refusal "
                    "was written by no process of this campaign, and nothing the "
                    "campaign recorded before the refusal moved its state without a "
                    "hash. Restore the copy the log's newest risk.state_hash describes "
                    f"-- the {hash_witness.describe()}, which is "
                    f"{hash_witness.state_hash} (docs/demo_runbook.md, section 4)",
                )
            return verdict(ContinuityOutcome.STATE_AHEAD_OF_LOG)

    return verdict(ContinuityOutcome.CONTINUOUS)
