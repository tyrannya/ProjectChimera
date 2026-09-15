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

**The witness is chosen by kind, not by position.** Only DECISION, FUNDING,
RECONCILIATION and LIQUIDATION_TOUCH records carry ``risk.state_hash``; HALT,
RESUME, OPERATOR, STARTUP, SHUTDOWN, INCOMPLETE_STATE, SKIPPED_STALE and RECOVERY
carry none. Comparing against "the log's last record" would therefore be wrong in
both directions at once, and both were measured:

* a campaign that halted ends on a HALT record, and ``halted`` and
  ``halt_reason`` are *inside* the hash -- so the restored state legitimately
  hashes differently from the newest hash-bearing record, and a naive comparison
  would dispute **every healthy restart after a halt**;
* a stale or replaced ``risk.json`` followed by any operational record --
  STARTUP, SKIPPED_STALE, INCOMPLETE_STATE -- would slip past a comparison that
  gave up when the physically last record carried no hash.

So the scan walks back through the kinds that cannot have moved the risk state
and stops at the first record that pins it. Three outcomes, and the difference
between them is the whole of this module:

``a hash-bearing record``
    the state is pinned exactly; the hashes must be equal.
``a HALT record``
    the state is not pinned, but :meth:`chimera.demo.runner.DemoRunner._halt`
    halts Aegis *before* it writes the record and only
    :meth:`chimera.risk.RiskEngine.resume` and
    :meth:`chimera.risk.RiskEngine.adopt_reconciled_equity` ever clear a halt --
    each of which writes a record of its own. So a HALT that no RESUME and no
    ``resolve-equity`` follows means the loaded state MUST still be halted. This
    is the direct answer to AEG-4's "deleting one file silently clears a
    persisted halt".
``a RESUME or an OPERATOR record``
    the risk state moved and no record says what to. Nothing can be compared;
    the state's mere *existence* is all this module may assert. That residual is
    stated rather than papered over: making those records carry a
    ``risk.state_hash`` too would close it, and doing so is a change to what the
    runner writes rather than to what it checks.

**A disagreement has a DIRECTION, and only one of them fails closed.** This is
the whole of what "the same continuity the store and ledger already have" means,
and getting it wrong breaks the runner's crash semantics rather than tightening
them. The carry ledger is refused when it holds **less** than the log has already
committed -- `DemoRunner._ledger_regressed_against_the_log`, which halts, because
every save precedes the record that quotes it and so no crash can produce that
direction. It is *recovered from* when it holds **more**: that is section 9.3's
``LOG_BEHIND_STATE``, the state was persisted and the record that quotes it was
not, a ``RECOVERY`` record is written, the affected minute leaves the evidence
and the campaign carries on.

``risk.json`` gets exactly that treatment:

* a loaded state whose identity the log records at an **earlier** witness but not
  at the newest is a state the campaign has already moved past -- an older copy
  restored, a file rolled back. No crash produces it, so it is a **dispute**;
* a loaded state whose identity **no** witness records is the ordinary crash
  window: ``tick`` persists the risk state and then commits the DECISION that
  quotes it, and a process killed between the two leaves precisely this. It is
  reported through :meth:`chimera.demo.runner.DemoRunner._state_ahead_of_log` as
  ``LOG_BEHIND_STATE``, beside the legs and the ledger it already compares, and
  the campaign recovers. Refusing it instead would convert an ordinary SIGKILL
  into a terminal halt, which `tests/test_demo_runner.py`'s two-sided control
  refuses on the ledger's behalf and would refuse here for the same reason.

The limit of that is stated rather than implied: a ``risk.json`` edited to a
state no witness records is indistinguishable from that crash and is accepted as
ahead, exactly as an edited store or ledger is. What continuity buys is that a
state which is *absent*, *unreadable*, *unhalted where the log records a halt* or
*demonstrably older than the log* can no longer pass, and those are the four
shapes AEG-4 was about.

**Fail closed, and preserve the evidence.** Every dispute below leaves both files
exactly as they were found: this module writes nothing, and the runner's refusal
is a halt plus a ``RECOVERY`` record, never a repair. A ``risk.json`` that could
not be read is additionally protected by :meth:`chimera.risk.RiskEngine._persist`,
which writes nothing at all while the load was UNREADABLE.

Nothing here creates scientific evidence, a prospective boundary, an alpha claim
or any real-money authority.
"""

from __future__ import annotations

from dataclasses import dataclass
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
#: reconciliation disputes.
RISK_HASH_EXCLUDED = frozenset({"order_times", "cooldown_until", "day"})

#: The kinds after which the risk state may legitimately differ from every
#: earlier ``risk.state_hash``, without any record saying how.
#:
#: HALT halts Aegis, RESUME clears the halt and the funding streak with it, and
#: each OPERATOR command moves something: ``flatten`` hands the flattened equity
#: to ``update_equity``, ``resolve`` clears a reconciliation dispute, and
#: ``resolve-equity`` adopts the ledger's equity. None of the three kinds carries
#: a hash, so each ENDS the backward scan rather than being walked past.
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
    makes a re-detection idempotent: see :class:`ContinuityOutcome`'s
    ``ALREADY_DISPUTED``.
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


class ContinuityOutcome(str, Enum):
    """What the comparison found. A bounded vocabulary; the record quotes it."""

    #: The log holds no record, so there is no campaign for a risk state to
    #: continue. A genuine first start, whether or not ``risk.json`` exists --
    #: ``build_risk_engine`` writes one BEFORE the first STARTUP record, so a
    #: file with an empty log beside it is the ordinary shape of a first start
    #: that was interrupted, not a discontinuity.
    FIRST_START = "FIRST_START"
    #: History exists and the persisted state continues it.
    CONTINUOUS = "CONTINUOUS"
    #: The loaded state is already halted on a continuity refusal. The
    #: discontinuity has been detected and recorded; re-deriving it would report
    #: a LATER shape of the same problem under a different name.
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
    #: The loaded state's identity is one the log records at an EARLIER witness
    #: and not at the newest: the campaign has already moved past this state, so
    #: the file was rolled back, restored from an older copy or replaced. The
    #: risk-state counterpart of ``ledger_behind_log``, and refused for the same
    #: reason -- every persist precedes the record that quotes it, so no crash
    #: produces this direction.
    STATE_HASH_REGRESSED = "STATE_HASH_REGRESSED"
    #: The loaded state's identity is one NO witness records. The ordinary crash
    #: window: the state was persisted and the record quoting it was not. Not a
    #: dispute; :meth:`chimera.demo.runner.DemoRunner._state_ahead_of_log`
    #: reports it as section 9.3's ``LOG_BEHIND_STATE`` and the campaign
    #: recovers.
    STATE_AHEAD_OF_LOG = "STATE_AHEAD_OF_LOG"


#: The outcomes that refuse to start. Stated as a set rather than as "everything
#: that is not CONTINUOUS", so an outcome added later is a deliberate decision
#: about whether a campaign may run rather than a default -- and so that
#: ``STATE_AHEAD_OF_LOG``'s absence from it is visible as the decision it is.
_DISPUTED_OUTCOMES = frozenset(
    {
        ContinuityOutcome.ALREADY_DISPUTED,
        ContinuityOutcome.RISK_STATE_MISSING,
        ContinuityOutcome.RISK_STATE_UNREADABLE,
        ContinuityOutcome.RISK_STATE_WITHOUT_ACCOUNT,
        ContinuityOutcome.HALT_NOT_HELD,
        ContinuityOutcome.STATE_HASH_REGRESSED,
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
    hash_witness: Witness | None = None
    halt_witness: Witness | None = None
    #: The EARLIER record whose ``risk.state_hash`` the loaded state still
    #: carries, when there is one. What tells a rolled-back file apart from the
    #: ordinary crash window, and the evidence the refusal quotes.
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
        if self.hash_witness is not None and self.hash_witness.state_hash:
            block["witness_state_hash"] = self.hash_witness.state_hash
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


def _records_newest_first(state_dir: str | Path) -> Iterator[Mapping[str, Any]]:
    """Every committed record of the log, newest first.

    Day files in reverse name order -- which :func:`day_files` documents as chain
    order -- and each file's records reversed within it. One file is materialised
    at a time and the caller stops as soon as it has what it needs, so an ordinary
    start reads the newest day and no more.

    :func:`read_records` stops at the first line it cannot read, which is exactly
    right here: a torn tail is a record that was never committed, and this runs
    BEFORE :func:`chimera.demo.decision_log.recover_tail` removes the bytes. A
    record the writer never finished may not witness anything.
    """
    for path in reversed(day_files(Path(state_dir) / LOG_DIR_NAME)):
        for record in reversed(list(read_records(path))):
            yield record


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


def _scan(state_dir: str | Path) -> tuple[bool, Witness | None, Witness | None]:
    """``(has_history, hash_witness, halt_witness)``, in one backward pass.

    The two witnesses answer different questions and are found by different
    rules, but they are found together because the hash witness usually IS the
    halt witness and a second pass would read the same bytes twice.

    The scan stops as soon as both are known. For the hash witness that is the
    first record that either carries a ``risk.state_hash`` or is one of the kinds
    that moved the state without recording it. For the halt witness it is the
    first HALT, RESUME or ``resolve-equity`` -- so the only case that reads
    further back than the hash witness is a tail of ``flatten`` or
    ``resolve --symbol``, neither of which can clear a halt and neither of which
    is common.

    A hash witness that CARRIES a hash makes the halt witness redundant --
    ``halted`` is inside the hash -- so the scan stops there and reports no halt
    witness rather than walking a whole campaign to find one it will not use.
    """
    has_history = False
    hash_witness: Witness | None = None
    halt_witness: Witness | None = None
    for record in _records_newest_first(state_dir):
        has_history = True
        witness = _witness(record)
        if hash_witness is None and (witness.state_hash or witness.kind in _RISK_MOVING_KINDS):
            hash_witness = witness
            if witness.state_hash:
                # The hash pins `halted` too, so nothing further back can add
                # anything this comparison would use.
                return has_history, hash_witness, halt_witness
        if halt_witness is None and (
            witness.kind == RecordKind.HALT.value or _clears_a_halt(witness)
        ):
            halt_witness = witness
        if hash_witness is not None and halt_witness is not None:
            return has_history, hash_witness, halt_witness
    return has_history, hash_witness, halt_witness


def _earlier_witness_carrying(
    state_dir: str | Path, state_hash: str, *, before: Witness
) -> Witness | None:
    """The newest record BEFORE ``before`` whose ``risk.state_hash`` is ``state_hash``.

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
    for record in _records_newest_first(state_dir):
        witness = _witness(record)
        if witness.seq >= before.seq:
            continue
        if witness.state_hash and witness.state_hash == state_hash:
            return witness
    return None


def continuity_already_recorded(state_dir: str | Path, fingerprint: str) -> bool:
    """Whether the log's newest continuity RECOVERY record names this same one.

    A refused campaign gets restarted -- by an operator, by a supervisor, by a
    systemd restart loop -- and the discontinuity is still there each time. One
    event deserves one record, so the writer asks this first.

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
    a state that is not there cannot be compared, a state that is already refused
    is not re-derived, and only a state that loaded is measured against a witness.
    """
    has_history, hash_witness, halt_witness = _scan(state_dir)
    observed = risk_state_hash(state) if load is RiskStateLoad.LOADED else ""
    matched: Witness | None = None

    def verdict(outcome: ContinuityOutcome, reason: str = "") -> RiskContinuity:
        return RiskContinuity(
            outcome=outcome,
            reason=f"{RISK_CONTINUITY_PREFIX} {reason}" if reason else "",
            load=load,
            observed_state_hash=observed,
            hash_witness=hash_witness,
            halt_witness=halt_witness,
            matched_witness=matched,
            has_history=has_history,
        )

    if not has_history:
        # Nothing has been decided under this state directory, so there is no
        # continuity to break. This is the case R1-c must NOT convert into a
        # dispute: a pristine deployment has no risk.json and no log, and
        # refusing it would mean no campaign could ever start.
        return verdict(ContinuityOutcome.FIRST_START)

    if load is RiskStateLoad.LOADED and is_risk_continuity_halt(state.halt_reason):
        # A previous start detected a discontinuity, wrote its RECOVERY record
        # and halted, and the halt is in the file. Re-deriving the verdict here
        # would describe a LATER shape of the same problem -- after a refused
        # MISSING start, `build_risk_engine` has seeded and the halt has
        # persisted, so the very next start would report a hash mismatch and
        # append a second, differently-named RECOVERY record for one event. The
        # campaign stays refused on the reason it was first refused for.
        return RiskContinuity(
            outcome=ContinuityOutcome.ALREADY_DISPUTED,
            reason=state.halt_reason,
            load=load,
            observed_state_hash=observed,
            hash_witness=hash_witness,
            halt_witness=halt_witness,
            has_history=has_history,
        )

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
        halt_witness is not None
        and halt_witness.kind == RecordKind.HALT.value
        and not state.halted
    ):
        return verdict(
            ContinuityOutcome.HALT_NOT_HELD,
            f"the log's newest halt witness is a {halt_witness.describe()} that no "
            "RESUME and no `resolve-equity` follows, and the risk state on disk is not "
            "halted. A halt is written to risk.json before the record that reports it "
            "and only an operator command clears one, so a persisted halt that has gone "
            "missing means the file was replaced, restored from an older copy or edited "
            "(docs/demo_runbook.md, section 4)",
        )

    if (
        hash_witness is not None
        and hash_witness.state_hash
        and hash_witness.state_hash != observed
    ):
        # The two are the same function of the same fields, so a campaign that
        # stopped cleanly restarts with them equal. Which way they disagree is
        # what decides whether this is a refusal or a crash to recover from, and
        # the log itself answers it: a state the campaign has already moved past
        # is one an EARLIER record still records.
        matched = _earlier_witness_carrying(state_dir, observed, before=hash_witness)
        if matched is not None:
            return verdict(
                ContinuityOutcome.STATE_HASH_REGRESSED,
                f"the risk state on disk hashes to {observed}, which the log records "
                f"at the {matched.describe()} and not at its newest risk.state_hash "
                f"-- the {hash_witness.describe()}, which is "
                f"{hash_witness.state_hash}. The campaign has already moved past the "
                "state this file holds, and every persist precedes the record that "
                "quotes it, so no crash produces this: risk.json was rolled back, "
                "restored from an older copy or replaced. Restore it from a copy at "
                "least as recent as the log's last risk.state_hash record "
                "(docs/demo_runbook.md, section 4)",
            )
        return verdict(ContinuityOutcome.STATE_AHEAD_OF_LOG)

    return verdict(ContinuityOutcome.CONTINUOUS)
