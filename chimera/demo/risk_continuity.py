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

**A hashless record may not switch the comparison off.** Only DECISION, FUNDING,
RECONCILIATION and LIQUIDATION_TOUCH carry ``risk.state_hash``; HALT, RESUME,
OPERATOR, STARTUP, SHUTDOWN, INCOMPLETE_STATE, SKIPPED_STALE and RECOVERY carry
none. An earlier revision of this module stopped its backward scan at the first
HALT, RESUME or OPERATOR record and, finding no hash there, compared nothing at
all -- so a ``risk.json`` restored from four minutes earlier passed behind a
``flatten``, behind a ``resume``, and behind a second HALT, and an OLDER halted
state passed a newer HALT history merely because both said ``halted``. The
independent review measured all three.

So the scan walks past **every** record that carries no hash and takes the newest
one that does. That record pins an identity, and the question the comparison
really asks is directional:

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
    moved the state and said nothing about where to. The two are told apart by
    whether such a record is in the tail at all; see ``STATE_AHEAD_OF_LOG``.

**A halt is normalised away before the identities are compared, because
``RiskEngine.halt`` is the one transition the log does not record and CAN be
inverted.** ``halted`` and ``halt_reason`` are inside ``risk.state_hash`` and a
HALT record carries no hash, so a halted state never hashes to any witness --
which is why a naive comparison disputes every healthy restart after a halt, and
why an older halted state used to pass. :meth:`chimera.risk.RiskEngine.halt`
sets exactly those two fields and nothing else, so the identity the same state
had *before* it was halted is computable, and both identities are compared. A
healthy restart after a halt matches the newest witness on the second; an older
halted state matches an EARLIER witness on it and is refused. No other
transition is inverted: RESUME's and each OPERATOR command's effects are not
reconstructed here, and a state they moved is reported as one no witness
records.

**Nothing a refused startup wrote is campaign history.** Refusing writes a
RECOVERY record and halts, and the halt writes a HALT record of its own. Read
back as ordinary evidence, that HALT condemned the very file the runbook tells an
operator to restore: the restored state is legitimately NOT halted, the log's
newest halt witness was R1-c's own refusal, and the documented recovery looped --
one more RECOVERY per attempt, the correct backup replaced by another halted
placeholder each time. Recognising this module's own halt reason would not have
been enough on its own, because a startup gate that refuses for its OWN reason
leaves the same shape: a ``source_identity`` halt on the placeholder
``build_risk_engine`` had just seeded says "this campaign is halted" just as
loudly. So the rule is drawn one level up. Every record at or after the newest
continuity RECOVERY that no hash-bearing record follows is a process starting,
finding the risk state in dispute and stopping -- no campaign decided anything in
that stretch, by construction -- and none of it witnesses anything about the risk
state. See :func:`_scan`.

**The finding is durable in the LOG, not in ``risk.json``.** A continuity refusal
used to be remembered only as ``risk.json``'s halt reason, and
:meth:`chimera.risk.RiskEngine.halt` keeps the FIRST reason -- so when R1-b's
``equity_dispute`` halted the engine while it was still being built, the
continuity reason never reached the file and the next process found nothing to
re-raise. The review measured the dispute evaporating across one ordinary
restart and ``resolve --equity`` then clearing a rolled-back state. Both are
consulted now: the halt reason in the file, AND the log's own continuity
RECOVERY record.

That record stands as an OPEN dispute until one of two things happens. A
hash-bearing record follows it -- the campaign ran again, so it was settled and
the dispute is behind the newest witness where the scan never reaches it. Or the
file on disk stops being the one the refusal left behind, which is what restoring
a backup does. The second is decided by an identity the record carries:
:func:`halt_free_state_hash` of the state the refusing process was holding, taken
with any halt set aside because WHICH halt ends up in that file is precisely what
is not stable -- `build_risk_engine` seeds a placeholder over a missing
``risk.json`` and then a dirty working tree, an equity dispute or R1-c itself
halts it, and each of those writes a different reason into the same state. It
needs no ``risk.state_hash`` in the log, so a campaign that halted before it
decided its first minute is neither bricked by a refusal nor able to pass its
placeholder off as history. See :class:`ContinuityOutcome.ALREADY_DISPUTED`.

**A disagreement has a DIRECTION, and only one of them fails closed.** This is
what "the same continuity the store and ledger already have" means, and getting
it wrong breaks the runner's crash semantics rather than tightening them. The
carry ledger is refused when it holds **less** than the log has already committed
-- `DemoRunner._ledger_regressed_against_the_log`, which halts. It is *recovered
from* when it holds **more**: that is section 9.3's ``LOG_BEHIND_STATE``, the
state was persisted and the record that quotes it was not, a ``RECOVERY`` record
is written, the affected minute leaves the evidence and the campaign carries on.
``risk.json`` gets exactly that treatment, and `tests/test_demo_runner.py`'s
two-sided control refuses the alternative on the ledger's behalf.

The limit of that is stated rather than implied: a ``risk.json`` edited to a
state no witness records is indistinguishable from the crash window and is
accepted as ahead, exactly as an edited store or ledger is. What continuity buys
is that a state which is *absent*, *unreadable*, *unhalted where the log records
a halt*, *demonstrably older than the log* or *already the subject of an
unsettled refusal* can no longer pass.

**Fail closed, and preserve the evidence.** Every dispute below leaves both files
exactly as they were found: this module writes nothing, and the runner's refusal
is a halt plus a ``RECOVERY`` record, never a repair. A ``risk.json`` that could
not be read is additionally protected by :meth:`chimera.risk.RiskEngine._persist`,
which writes nothing at all while the load was UNREADABLE.

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
    "OpenDispute",
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
#: The three below are the hashless kinds that CAN move it. They no longer END
#: the backward scan -- that was the defect -- but a state whose identity no
#: witness records is told apart from the crash window by whether one of them is
#: in the tail. Giving them a ``risk.state_hash`` would close the residual and is
#: a change to what the runner WRITES, which reaches replay parity; it is not
#: this item's.
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
    campaign holding a halt that no code admits to recognising. It is also what
    keeps this module's own refusal out of its own evidence: see
    :class:`ContinuityOutcome`'s ``ALREADY_DISPUTED``.
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
    approximation of one. Two things need it, and both are about a halt the log
    does not record:

    * a HALT record carries no ``risk.state_hash``, so a halted state's own
      identity is in no witness -- without this the log could say nothing at all
      about a halted file, which is how an older halted state used to pass a
      newer HALT history merely for being halted too;
    * the state a refused start leaves on disk gets halted by whatever refused
      it, and WHICH reason wins is not stable. This identity is, which is what
      lets a later start recognise that file and what
      :attr:`OpenDispute.seeded_hash` records.
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
    #: record in the tail legitimately moved it.
    CONTINUOUS = "CONTINUOUS"
    #: Whether the log holds durable history could not be established: day files
    #: hold committed lines that no record could be read from. Reported as its
    #: own answer rather than as ``FIRST_START``, which would tell an operator
    #: that a campaign whose history is unreadable has none.
    HISTORY_UNVERIFIABLE = "HISTORY_UNVERIFIABLE"
    #: A discontinuity a previous start already detected and wrote down, and
    #: which nothing has settled since. Re-deriving it would report a LATER shape
    #: of the same problem under a different name and append a second record for
    #: one event. Two things say so and either is enough, because each covers the
    #: other's gap: ``risk.json``'s own halt reason, which `RiskEngine.halt`'s
    #: first-reason-wins can lose to an `equity_dispute` raised while the engine
    #: was being built; and the log's newest continuity RECOVERY record, which is
    #: OPEN while no hash-bearing record follows it and the file on disk is still
    #: the one that refusal left behind (:attr:`OpenDispute.seeded_hash`).
    ALREADY_DISPUTED = "ALREADY_DISPUTED"
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
    #: The loaded state's identity is one NO witness records, and no hashless
    #: record moved it since the newest one. The ordinary crash window: the state
    #: was persisted and the record quoting it was not. Not a dispute;
    #: :meth:`chimera.demo.runner.DemoRunner._state_ahead_of_log` reports it as
    #: section 9.3's ``LOG_BEHIND_STATE`` and the campaign recovers.
    STATE_AHEAD_OF_LOG = "STATE_AHEAD_OF_LOG"


#: The outcomes that refuse to start. Stated as a set rather than as "everything
#: that is not CONTINUOUS", so an outcome added later is a deliberate decision
#: about whether a campaign may run rather than a default -- and so that
#: ``STATE_AHEAD_OF_LOG``'s absence from it is visible as the decision it is.
_DISPUTED_OUTCOMES = frozenset(
    {
        ContinuityOutcome.ALREADY_DISPUTED,
        ContinuityOutcome.HISTORY_UNVERIFIABLE,
        ContinuityOutcome.RISK_STATE_MISSING,
        ContinuityOutcome.RISK_STATE_UNREADABLE,
        ContinuityOutcome.RISK_STATE_WITHOUT_ACCOUNT,
        ContinuityOutcome.HALT_NOT_HELD,
        ContinuityOutcome.STATE_HASH_REGRESSED,
    }
)


@dataclass(frozen=True)
class Witness:
    """One log record, in the only five respects this comparison uses."""

    seq: int
    kind: str
    #: ``risk.state_hash`` when the record carries one, otherwise ``""``.
    state_hash: str = ""
    #: ``operator.command`` on an OPERATOR record, otherwise ``""``.
    command: str = ""
    #: ``veto_or_rejection.detail`` -- on a HALT record, the reason the runner
    #: halted for, which `DemoRunner._halt` also hands to `RiskEngine.halt` and
    #: so is the reason that lands in ``risk.json`` when nothing had halted it
    #: already. See :func:`_settles`.
    detail: str = ""

    def describe(self) -> str:
        """The witness in words, for a halt reason that must read the same on
        every host: a kind, a sequence number and a command, and no path."""
        named = f"{self.kind} record seq {self.seq}"
        return f"{named} (`{self.command}`)" if self.command else named


@dataclass(frozen=True)
class OpenDispute:
    """A continuity RECOVERY record the log holds and nothing has settled."""

    seq: int
    fingerprint: str
    detail: str
    #: :func:`halt_free_state_hash` of the risk state the refusing process was
    #: holding when it wrote this record -- the file it was about to leave on
    #: disk. A later start that still finds that identity has not been given a
    #: different file, whatever has since been halted into it. ``""`` on a record
    #: written before this field existed, which then settles nothing and disputes
    #: nothing.
    seeded_hash: str = ""


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

    @property
    def already_recorded(self) -> bool:
        """Whether the discontinuity is one a previous start already wrote down."""
        return self.outcome is ContinuityOutcome.ALREADY_DISPUTED

    def block(self, *, seeded_state_hash: str = "") -> dict[str, Any]:
        """The ``recovery.risk_continuity`` block of the RECOVERY record.

        ``seeded_state_hash`` comes from the RUNNER rather than from this
        read-only verdict, and it is the whole of how a refusal is later known to
        be unsettled: :func:`halt_free_state_hash` of the risk state the refusing
        process is holding, which is the file it is about to leave behind.
        `build_risk_engine` has to run before anything can halt it, so on a
        MISSING load that file is a freshly seeded placeholder holding the
        configured capital; on a rolled-back load it is the rolled-back state.
        Either way a later start that finds the same identity has been given no
        new file, and one that finds a different identity has.

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
        if seeded_state_hash:
            block["seeded_state_hash"] = seeded_state_hash
        return block

    @property
    def fingerprint(self) -> str:
        """What makes two detections of this discontinuity the same one.

        A start that refuses writes one RECOVERY record; a start that refuses
        again for the same reason must not write a second. The identity is the
        verdict plus what it was taken from -- never a timestamp and never a seq
        of this process's own records -- so repeated starts against an unchanged
        pair of files produce the same value and repeated starts against a
        CHANGED one do not.
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
    #: The newest HALT, RESUME or ``resolve-equity`` NEWER than the hash witness.
    #: This module's own refusal halts are not among them.
    halt_witness: Witness | None
    #: Whether a hashless record that can move the risk state sits between the
    #: hash witness and the end of the log.
    moved_since_witness: bool
    #: The newest continuity RECOVERY with no hash-bearing record after it.
    open_dispute: OpenDispute | None


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
    veto = record.get("veto_or_rejection")
    detail = ""
    if isinstance(veto, Mapping):
        found = veto.get("detail")
        if isinstance(found, str):
            detail = found
    seq = record.get("seq")
    return Witness(
        seq=int(seq) if isinstance(seq, int) else -1,
        kind=str(record.get("kind", "")),
        state_hash=state_hash,
        command=command,
        detail=detail,
    )


def _clears_a_halt(witness: Witness) -> bool:
    """Whether this record can have left Aegis unhalted."""
    if witness.kind == RecordKind.RESUME.value:
        return True
    return (
        witness.kind == RecordKind.OPERATOR.value
        and witness.command in _HALT_CLEARING_COMMANDS
    )


def _open_dispute(record: Mapping[str, Any], witness: Witness) -> OpenDispute | None:
    """This record as an unsettled continuity refusal, when that is what it is."""
    if witness.kind != RecordKind.RECOVERY.value:
        return None
    recovery = record.get("recovery")
    if not isinstance(recovery, Mapping):
        return None
    block = recovery.get("risk_continuity")
    if not isinstance(block, Mapping):
        return None
    return OpenDispute(
        seq=witness.seq,
        fingerprint=str(block.get("fingerprint_hash", "")),
        detail=str(recovery.get("detail", "")),
        seeded_hash=str(block.get("seeded_state_hash", "")),
    )


def _settles(
    dispute: OpenDispute, *, state: RiskState, halt_free: str, halt_witness: Witness | None
) -> bool:
    """Whether what is on disk now settles the refusal the log still holds open.

    Only asked when the log holds a continuity RECOVERY that no hash-bearing
    record follows, so the campaign has not run past it. Three ways out, and each
    exists because one of the others cannot answer:

    ``there is nothing on disk to recognise``
        the ``risk.json`` is still missing, or still unreadable, so it has no
        identity at all. There is no file to say "this is the one the refusal
        left", and the fresh verdict below refuses it again on its own terms.
        `DemoRunner._record_risk_continuity`'s fingerprint is what stops the
        second detection becoming a second record.
    ``a different file is there``
        the state's identity, with any halt set aside, is not the one the
        refusing process was holding. Something has been restored. This is the
        ordinary way out and the one the runbook documents.
    ``the same state is there, carrying a halt the log recorded BEFORE the
    refusal``
        the narrow case where the first two cannot tell the difference: a
        campaign that halted before it decided its first minute holds nothing but
        the first start's seed plus a halt, and a placeholder seeded over its
        missing ``risk.json`` holds the same seed plus a DIFFERENT halt. Setting
        the halt aside makes the two equal by construction. What separates them
        is which halt: the campaign's is a HALT record the log wrote before the
        refusal, and no process after the refusal could have put that reason into
        the file. Without this the documented restore could not terminate for
        such a campaign, which is not a safety property, it is a brick.
    """
    if not halt_free:
        return True
    if halt_free != dispute.seeded_hash:
        return True
    return (
        state.halted
        and halt_witness is not None
        and halt_witness.kind == RecordKind.HALT.value
        and halt_witness.detail == state.halt_reason
    )


def _scan(state_dir: str | Path) -> _LogFacts:
    """Everything the log says, in one backward pass that stops as soon as it can.

    The pass ends at the first record carrying a ``risk.state_hash``, and that is
    what makes each answer well defined as well as cheap:

    * the **hash witness** is that record -- every hashless record newer than it
      is walked past rather than being allowed to end the comparison;
    * the **halt witness** is only looked for in the tail newer than it, because
      the hash pins ``halted`` and a HALT older than the hash witness can add
      nothing;
    * the **open dispute** is likewise only looked for there, which is precisely
      the definition: a continuity RECOVERY that a hash-bearing record follows is
      one the campaign has since run past, so it is settled.

    An ordinary start therefore reads one record. Only a log that has never
    recorded an identity is walked in full, and such a log is short.

    **Nothing a refused startup wrote is campaign evidence.** The tail is then
    read forwards, and everything at or after the first continuity RECOVERY in it
    is skipped. No hash-bearing record can be in that stretch -- the tail ends at
    one by construction -- so nothing there is a campaign deciding anything; it
    is a sequence of processes starting, finding the risk state in dispute and
    stopping. The HALT records among them are halts of the very state R1-c
    refused, and reading one back as "this campaign is halted and nothing cleared
    it" is how the correct restored backup came to be condemned. A ``source_identity``
    refusal on a freshly seeded placeholder produced exactly that HALT without
    R1-c even being the reason for it, which is why recognising this module's own
    halt reason would not have been enough: this one is nobody's refusal, it is a
    different gate refusing a state R1-c had already written off.
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
    open_dispute: OpenDispute | None = None
    refused_since = False
    for record, witness in reversed(tail):  # chain order: the newest wins
        found = _open_dispute(record, witness)
        if found is not None:
            open_dispute = found
            refused_since = True
            continue
        if refused_since:
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
        open_dispute=open_dispute,
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
    systemd restart loop -- and the discontinuity is still there each time. One
    event deserves one record, so the writer asks this first.

    ``ALREADY_DISPUTED`` is the other half of that guarantee and the one that
    carries most of it: it fires when the log still holds an unsettled refusal,
    whatever shape the files have taken since. This catches the remaining case,
    where a log that has never recorded an identity leaves nothing to pin a state
    against and the same fresh verdict is therefore re-derived on every start.

    "Newest" means the newest RECOVERY record that carries a ``risk_continuity``
    block, not the newest RECOVERY record of any kind: a torn tail repaired
    between two refused starts writes a RECOVERY of its own, and letting that one
    answer would append a second record for a discontinuity that had already been
    written down.
    """
    for record in _records_newest_first(state_dir):
        if str(record.get("kind", "")) != RecordKind.RECOVERY.value:
            continue
        recovery = record.get("recovery")
        if not isinstance(recovery, Mapping):
            continue
        block = recovery.get("risk_continuity")
        if not isinstance(block, Mapping):
            continue
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
    a campaign with no history cannot have broken any, a discontinuity already
    written down is not re-derived under a second name, a state that is not there
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
    # This state with any halt set aside: what it is, independently of what has
    # been halted into it since. `unhalted` when it is halted, `observed` when it
    # is not, and the two are one function of one state.
    halt_free = unhalted or observed
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
        )

    def already_disputed(reason: str) -> RiskContinuity:
        """The refusal a previous start already made, quoted rather than remade.

        The reason is the one that was recorded, and it is already prefixed --
        it came either from `risk.json`'s halt reason or from the RECOVERY
        record's own detail, both of which this module wrote. Passing it through
        `verdict` would prefix it twice.
        """
        return RiskContinuity(
            outcome=ContinuityOutcome.ALREADY_DISPUTED,
            reason=reason,
            load=load,
            observed_state_hash=observed,
            unhalted_state_hash=unhalted,
            hash_witness=hash_witness,
            halt_witness=facts.halt_witness,
            has_history=facts.has_history,
        )

    if not facts.has_history:
        if facts.unverifiable:
            return verdict(
                ContinuityOutcome.HISTORY_UNVERIFIABLE,
                "the decision log holds committed lines that no record could be read "
                "from, so whether this campaign has durable history cannot be "
                "established and the risk state cannot be compared against it. Nothing "
                "here is repaired or removed; verify the log (docs/demo_runbook.md, "
                "section 4)",
            )
        # Nothing has been decided under this state directory, so there is no
        # continuity to break. This is the case R1-c must NOT convert into a
        # dispute: a pristine deployment has no risk.json and no log, and
        # refusing it would mean no campaign could ever start.
        return verdict(ContinuityOutcome.FIRST_START)

    if load is RiskStateLoad.LOADED and is_risk_continuity_halt(state.halt_reason):
        # A previous start detected a discontinuity, wrote its RECOVERY record
        # and halted, and the halt is still in the file.
        return already_disputed(state.halt_reason)

    if facts.open_dispute is not None and not _settles(
        facts.open_dispute,
        state=state,
        halt_free=halt_free,
        halt_witness=facts.halt_witness,
    ):
        # The log still holds an unsettled refusal -- nothing hash-bearing has
        # been written since, so the campaign has not run past it -- and the file
        # on disk is the one that refusal left behind, identified by what the
        # refusing process was holding when it wrote the record. Nothing has been
        # restored; only a halt has been written into the same state, and which
        # halt won is not something to rest on.
        #
        # That is what survives `RiskEngine.halt` keeping the FIRST reason: an
        # `equity_dispute` raised while the engine was being built leaves no
        # continuity reason in `risk.json` at all. And it is what survives a
        # startup gate -- a dirty working tree, say -- ending the process before
        # RECOVER could refuse, with `build_risk_engine`'s freshly seeded
        # placeholder standing where the missing state had been and that gate's
        # own reason halted into it.
        #
        # It settles the moment a different file appears, which is what the
        # documented restore does, and it needs no `risk.state_hash` in the log
        # to do it -- so a campaign that never decided a minute is neither
        # bricked by the refusal nor able to pass its placeholder off as history.
        return already_disputed(facts.open_dispute.detail)

    if load is RiskStateLoad.UNREADABLE:
        return verdict(
            ContinuityOutcome.RISK_STATE_UNREADABLE,
            "the decision log holds records for this campaign and risk.json could not "
            "be read, so what Aegis was enforcing cannot be established. The file is "
            "left exactly as it was found and nothing is written over it; restore it "
            "from a copy at least as recent as the log's last risk.state_hash record "
            "(docs/demo_runbook.md, section 4)",
        )

    if load is RiskStateLoad.MISSING:
        return verdict(
            ContinuityOutcome.RISK_STATE_MISSING,
            "the decision log holds records for this campaign and there is no risk.json, "
            "so a halt, a peak equity, a cooldown, an order window, a funding streak or "
            "an open reconciliation dispute would be silently reset to a default nothing "
            "recorded. An absent risk state is not evidence that the campaign had none; "
            "restore it from a copy at least as recent as the log's last risk.state_hash "
            "record (docs/demo_runbook.md, section 4)",
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
                "restored from an older copy or replaced. Restore it from a copy at "
                "least as recent as the log's last risk.state_hash record "
                "(docs/demo_runbook.md, section 4)",
            )
        if not facts.moved_since_witness:
            return verdict(ContinuityOutcome.STATE_AHEAD_OF_LOG)

    return verdict(ContinuityOutcome.CONTINUOUS)
