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
reads a clock or touches the :class:`chimera.risk.RiskEngine` the runner builds;
the one engine it constructs, to know the first-start seed, has no file, no
switch and a pinned clock (:func:`first_start_state_hash`). What to DO about a
verdict is the runner's; see :meth:`chimera.demo.runner.DemoRunner.start`.

**A disputed file is evidence, and nothing a disputed process does may write
it.** The runner takes this verdict before it builds Aegis, and when the verdict
is a dispute it builds Aegis with ``persist=False``
(:meth:`chimera.risk.RiskEngine._persist`). The engine still loads, seeds,
reconciles, halts and flattens -- in memory, which is what the runner needs to
refuse and to reduce exposure -- and ``risk.json`` keeps exactly the bytes, or
exactly the absence, it was found with. That is what makes every verdict below
REPEATABLE: the next process reads the same file and reaches the same answer,
whatever the processes in between did. For the same reason none of a disputed
process's records carries a ``risk.state_hash``: the engine it would describe is
one no file holds. An earlier revision let a disputed process keep writing -- the
seed over a missing file, the kill-switch mirror, `flatten`'s equity and
exposures, a reconciliation note -- and then had to guess from the changed file
whether it had been restored. It guessed "a different file means a restore", and
the independent review measured a permitted `flatten` dissolving the dispute
through the CLI, and a placeholder seeded before a crash, a failed append or a
forged log standing in for the missing state afterwards.

**Every record that moves the risk state says where it left it.** DECISION,
FUNDING, RECONCILIATION and LIQUIDATION_TOUCH have always carried
``risk.state_hash``. HALT, RESUME and OPERATOR carried none until the third
remediation, and that absence was the independent review's B1: a state no record
quoted was admitted merely because one of them stood after the newest record
that did, and the newest record's own state was admitted even when one of them
had since moved the campaign past it. Nothing those records held could tell the
correct file from a stale or fabricated one. ``resume`` zeroes the funding streak
and ``resolve --symbol`` erases the dispute it clears, so neither can be run
backwards; ``flatten``'s ``update_equity`` moves the peak and the day's P&L by
amounts that depend on fields the log holds only as a hash, so it cannot be run
forwards either. Even a HALT, whose own two fields do invert, does not say which
reason Aegis was left holding: :meth:`chimera.risk.RiskEngine.halt` keeps the
FIRST, so R1-b's ``equity_dispute:``, the kill switch's unexaminable-path reason
or an earlier guard's halt survives under a HALT record naming something else,
and the kill switch moves ``kill_switch`` beside it. So those three kinds now
carry the same hash, of the same fields, taken after the transition they record
(:meth:`chimera.demo.runner.DemoRunner._risk_block`) -- a narrow slice of R1-j
pulled forward because no rule over the old records could be exact.

The scan walks past every record that carries no hash -- STARTUP, SHUTDOWN,
INCOMPLETE_STATE, SKIPPED_STALE, RECOVERY, a disputed process's records, and the
hashless HALT, RESUME and OPERATOR records of a log written before they carried
one -- and takes the newest one that does. That record pins an identity, and the
question the comparison really asks is directional:

``the loaded identity IS the newest witness's``
    the file is the state the campaign last recorded. Continuous -- unless a
    hashless HALT, RESUME or OPERATOR the campaign wrote after that witness can
    have moved the state since, which only an older log holds; then the log
    cannot vouch for this state or any other.
``the loaded identity is one an EARLIER witness records``
    the campaign has already moved past this state: an older copy restored, a
    file rolled back, a state replaced. Every persist precedes the record that
    quotes it, so no crash produces this direction. **Refused.**
``the loaded identity is one NO witness records``
    the ordinary crash window -- Aegis persists and then the record quoting it
    is committed, and a process killed between the two leaves precisely this.
    See ``STATE_AHEAD_OF_LOG``, and ``NOT_SETTLED`` for why it proves nothing
    after a refusal.

**A halt the log records has to be held.** The newest HALT, RESUME or
``resolve-equity`` among the campaign's records decides whether it is halted, and
a file that is not halted where that record is a HALT is refused
(``HALT_NOT_HELD``). A state that matches its witness already carries the halt
that witness recorded, so the question is only asked of one that does not.

**Nothing a disputed process wrote is campaign history.** A refusal is written
down as a continuity ``RECOVERY`` record, and every record after it until the
matching settlement -- the continuity ``RECOVERY`` a later start writes when the
file on disk is proved to continue the log again -- was written by a process
holding the risk state as evidence: its HALT records halted an engine that wrote
nothing, and its OPERATOR records moved nothing on disk. Reading one of them back
as "this campaign is halted" or "this campaign's state was moved" would condemn
the correct restore, or excuse an incorrect one. So that stretch is skipped, and
the settlement record is what ends it: the settled process's own halts are
ordinary history again, and a later loss is a new event with a record of its
own. See :func:`_campaign_history`.

**A refusal is settled by exact proof, never by a change.** While a refusal is
open, an identity no record quotes is not the crash window any more, because no
process could have written it -- a disputed process writes nothing -- and so it
proves nothing. The refusal is settled by exactly two things:

* a file whose identity IS the one the log's newest witness records, with no
  hashless record after that witness that can have moved the state; or
* on a log that has never recorded an identity and holds no such record either,
  a file whose identity IS the first-start seed the runner's own capital
  produces -- the only state such a campaign has ever held.

Not a file that is different, not an identity nobody quotes, not a halt, and not
a HALT, RESUME or OPERATOR record merely being there.

**A disagreement has a DIRECTION, and only one of them fails closed.** This is
what "the same continuity the store and ledger already have" means. The carry
ledger is refused when it holds **less** than the log has already committed and
recovered from when it holds **more** (section 9.3's ``LOG_BEHIND_STATE``).
``risk.json`` gets exactly that treatment outside a refusal, and
`tests/test_demo_runner.py`'s two-sided control refuses the alternative on the
ledger's behalf.

The limits are stated rather than implied. Outside a refusal, a ``risk.json``
edited to a state no witness records is indistinguishable from the crash window
and is recovered from as one, exactly as an edited store or ledger is. A log
written before HALT, RESUME and OPERATOR carried a hash, whose tail after its
newest witness holds one of them, cannot vouch for ANY state: after a refusal
nothing settles it, its correct backup included, because old history is not
retroactively trusted. The tick that writes INCOMPLETE_STATE has already run
``note_feed``, which can move ``stale_feed_since``; a state moved that way is one
no record quotes and is treated as every such state is. What continuity buys is
that a state which is *absent*, *unreadable*, *unhalted where the log records a
halt*, *demonstrably older than the log*, or -- after a refusal -- *anything but
exactly the state the log vouches for* can no longer pass, and that no process
fabricates one.

Nothing here creates scientific evidence, a prospective boundary, an alpha claim
or any real-money authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from chimera.demo.decision_log import (
    LOG_DIR_NAME,
    RecordKind,
    day_files,
    read_records,
)
from chimera.risk import RiskEngine, RiskState, RiskStateLoad, state_snapshot

__all__ = [
    "RISK_CONTINUITY_PREFIX",
    "RISK_HASH_EXCLUDED",
    "ContinuityOutcome",
    "RiskContinuity",
    "Witness",
    "continuity_already_recorded",
    "evaluate_risk_continuity",
    "first_start_state_hash",
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

#: The kinds that move the risk state as an operator command or a halt.
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
#: HALT                  yes*     ``halted`` + ``halt_reason``, and the
#:                                kill-switch mirror when the switch is
#:                                what halted it; pinned
#: RESUME                yes*     clears the halt, the kill-switch mirror,
#:                                ``funding_halt`` and the adverse streak;
#:                                pinned
#: OPERATOR              yes*     ``flatten`` -> exposures, ``update_equity``;
#:                                ``resolve`` -> one dispute cleared;
#:                                ``resolve-equity`` ->
#:                                ``adopt_reconciled_equity``; pinned
#: STARTUP               no       nothing itself; a halt raised while
#:                                Aegis is built is followed by a HALT
#:                                before the start can finish
#: SHUTDOWN              no       nothing
#: RECOVERY              no       nothing
#: INCOMPLETE_STATE      no       ``stale_feed_since``, through the tick's
#:                                ``note_feed``; unquoted, see the header
#: SKIPPED_STALE         no       nothing
#: ===================== ======== =======================================
#:
#: \* from the third remediation, and only from a process whose risk state is
#: not in dispute. The three are named here because a log written BEFORE then
#: holds them without a hash, and one of them in the campaign's tail after its
#: newest witness means that witness may no longer describe the state: it was
#: moved and nothing says where to. That is a reason the log cannot vouch for a
#: state, never a reason to accept one -- which is exactly what it used to be.
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


def first_start_state_hash(capital: Any) -> str:
    """``risk_state_hash`` of the state a first start seeds for this capital.

    Built the way :func:`chimera.demo.risk_wiring.build_risk_engine` builds it --
    an engine that found no file, seeded by
    :func:`chimera.demo.risk_wiring.seed_or_reconcile_equity` -- rather than
    spelled out a second time here, so the two cannot drift. The engine has no
    file and no switch, and its clock is pinned: the only field the clock could
    reach is ``day``, which is outside the identity.

    It is what a campaign that has never recorded an identity has ever held, and
    so the one state such a campaign's log can vouch for after a refusal.
    """
    from chimera.demo.risk_wiring import seed_or_reconcile_equity

    seeded = RiskEngine(clock=lambda: 0.0)
    # A first start reads no ledger: `state_dir` is consulted on a restart only.
    seed_or_reconcile_equity(seeded, capital=capital, state_dir=".")
    return risk_state_hash(seeded.state)


class ContinuityOutcome(str, Enum):
    """What the comparison found. A bounded vocabulary; the record quotes it."""

    #: The log holds no record, so there is no campaign for a risk state to
    #: continue. A genuine first start, whether or not ``risk.json`` exists --
    #: ``build_risk_engine`` writes one BEFORE the first STARTUP record, so a
    #: file with an empty log beside it is the ordinary shape of a first start
    #: that was interrupted, not a discontinuity.
    FIRST_START = "FIRST_START"
    #: History exists and nothing refuses the persisted state. With no refusal
    #: open, its identity is the newest witness's, or the log has never recorded
    #: one. With a refusal open this is the SETTLEMENT, and it is reached only
    #: by exact proof: the identity IS the newest witness's and nothing after it
    #: moved the state without a hash, or the log has never recorded an
    #: identity and this one IS the first-start seed. See
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
    #: The newest HALT, RESUME or ``resolve-equity`` among the campaign's records
    #: is a HALT, and the loaded state is not halted.
    HALT_NOT_HELD = "HALT_NOT_HELD"
    #: The loaded state's identity is one the log records at an EARLIER witness
    #: and not at the newest: the campaign has already moved past this state, so
    #: the file was rolled back, restored from an older copy or replaced. The
    #: risk-state counterpart of ``ledger_behind_log``, and refused for the same
    #: reason: every persist precedes the record that quotes it, so no crash
    #: produces this direction.
    STATE_HASH_REGRESSED = "STATE_HASH_REGRESSED"
    #: A continuity refusal is open and nothing proves the loaded state is the
    #: one the campaign holds: its identity is one no witness records; or a
    #: hashless HALT, RESUME or OPERATOR after the newest witness can have moved
    #: the state since, so not even that witness's identity is vouched for; or
    #: the log has never recorded an identity and this one is not the
    #: first-start seed. A file that appeared after a process found the risk
    #: state in dispute was written by no process of this campaign -- a disputed
    #: process writes nothing -- so it is not evidence of a restore, and the
    #: refusal stands.
    NOT_SETTLED = "NOT_SETTLED"
    #: The loaded state's identity is one NO witness records, and no refusal is
    #: open. The ordinary crash window: the state was persisted and the record
    #: quoting it was not. Not a dispute;
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
    hash_witness: Witness | None = None
    #: The newest HALT, RESUME or ``resolve-equity`` among the campaign's
    #: records, where the verdict needed to know it.
    halt_witness: Witness | None = None
    #: The EARLIER record whose ``risk.state_hash`` the loaded state's identity
    #: still carries, when there is one. What tells a rolled-back file apart
    #: from the ordinary crash window, and the evidence the refusal quotes.
    matched_witness: Witness | None = None
    #: Whether the log holds any record at all.
    has_history: bool = False
    #: The ``seq`` of the continuity refusal this verdict SETTLES: set only on a
    #: ``CONTINUOUS`` verdict taken while one is open. The runner writes the
    #: settlement down, which is what ends the stretch :func:`_campaign_history` skips.
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
        because everything it wrote is skipped (:func:`_campaign_history`).
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
    #: The newest HALT, RESUME or OPERATOR newer than the hash witness -- and
    #: therefore carrying no hash -- among the records that are campaign
    #: history: a record that can have moved the risk state without saying where
    #: to. Only a log written before those kinds carried a hash holds one, and
    #: it is a reason the log cannot vouch for a state, never a reason to
    #: accept one.
    unwitnessed_move: Witness | None
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


def _campaign_history(
    records: Iterable[tuple[Mapping[str, Any], Witness]],
) -> tuple[list[Witness], int | None]:
    """The records that are campaign history, in chain order, and the open refusal.

    ``records`` is read forwards, in stretches. A continuity refusal opens a
    stretch and the continuity settlement after it closes it; everything in
    between -- the refusal's own HALT, a `flatten` an operator ran meanwhile, a
    kill switch that halted a process, further refusals as other files were
    tried -- was written by a process that held the risk state as evidence and
    persisted none of it, so none of it describes the file and none of it is
    returned. Records before the first refusal and after a settlement are the
    campaign's own history. A stretch still open at the end is the refusal
    nothing has settled, and its ``seq`` is the second value.
    """
    history: list[Witness] = []
    open_refusal_seq: int | None = None
    for record, witness in records:
        block = _continuity_block(record)
        if block is not None:
            if _settles_a_refusal(block):
                open_refusal_seq = None
            elif open_refusal_seq is None:
                open_refusal_seq = witness.seq
            continue
        if open_refusal_seq is None:
            history.append(witness)
    return history, open_refusal_seq


def _newest_halt_event(history: list[Witness]) -> Witness | None:
    """The newest HALT, RESUME or ``resolve-equity`` in ``history``, if any."""
    for witness in reversed(history):
        if witness.kind == RecordKind.HALT.value or _clears_a_halt(witness):
            return witness
    return None


def _scan(state_dir: str | Path) -> _LogFacts:
    """Everything the log says, in one backward pass that stops as soon as it can.

    The pass ends at the first record carrying a ``risk.state_hash``, and that is
    what makes each answer well defined as well as cheap:

    * the **hash witness** is that record -- every hashless record newer than it
      is walked past rather than being allowed to end the comparison;
    * a **refusal** is only looked for in the tail newer than it: one that a
      hash-bearing record follows is one the campaign has since run past, and no
      process holding a disputed state writes a hash;
    * a hashless HALT, RESUME or OPERATOR there is an **unwitnessed move**, and a
      HALT, RESUME or ``resolve-equity`` there is the **halt witness** -- both
      things only an older log's tail can hold, now that those kinds carry a
      hash of their own.

    An ordinary start therefore reads one record. Only a log that has never
    recorded an identity is walked in full here, and such a log is short. The
    tail is read forwards through :func:`_campaign_history`, so nothing a
    disputed process wrote is read as the campaign's.
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

    history, open_refusal_seq = _campaign_history(reversed(tail))
    return _LogFacts(
        has_history=has_history,
        unverifiable=(not has_history) and _history_unverifiable(state_dir),
        hash_witness=hash_witness,
        halt_witness=_newest_halt_event(history),
        unwitnessed_move=next(
            (witness for witness in reversed(history) if witness.kind in _RISK_MOVING_KINDS),
            None,
        ),
        open_refusal_seq=open_refusal_seq,
    )


def _whole_history(state_dir: str | Path) -> list[Witness]:
    """Every record of the retained log that is campaign history, in chain order.

    Two questions need more than the tail, and both are asked only of a state
    that does not match its newest witness -- so an ordinary start never pays
    for this walk, and only a start that is already in trouble does:

    * **which way the disagreement points**: a state the log records at an
      EARLIER witness is one the campaign has already moved past;
    * **whether the campaign is halted**: the newest HALT, RESUME or
      ``resolve-equity`` can be older than the newest witness -- a HALT followed
      by a `flatten`, both hashed -- and a halt the log records is still in force
      whatever hash-bearing records came after it.

    ``seq`` is the ordering, not the file position: it is assigned by
    :meth:`chimera.demo.decision_log.DecisionLog.append` and is strictly
    increasing across the whole chain.
    """

    def chain_order() -> Iterator[tuple[Mapping[str, Any], Witness]]:
        for path in day_files(Path(state_dir) / LOG_DIR_NAME):
            for record in read_records(path):
                yield record, _witness(record)

    history, _ = _campaign_history(chain_order())
    return history


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
    state_dir: str | Path, *, load: RiskStateLoad, state: RiskState, capital: Any
) -> RiskContinuity:
    """Does the persisted risk state continue the log beside it? Reads only.

    ``load`` and ``state`` are the READ's own answers -- ``RiskEngine.load_outcome``
    and ``RiskEngine.state`` from a read-only inspection of ``risk.json``. Both
    have to come from the same untouched load: ``load`` because every cheaper test
    of "was there a file" is wrong in the case that matters (``Path.exists()``
    answers ``False`` for a path it merely could not examine), and ``state``
    because the identity under comparison is the one the FILE holds and not the
    one a startup has since seeded or halted. ``capital`` is the runner's own,
    and is read only to know the first-start seed of a campaign whose log has
    never recorded an identity (:func:`first_start_state_hash`).

    The order of the checks is the order of what they can prove, strongest first:
    a campaign with no history cannot have broken any, a state that is not there
    cannot be compared, and only a state that loaded is measured against a
    witness.
    """
    facts = _scan(state_dir)
    hash_witness = facts.hash_witness
    observed = risk_state_hash(state) if load is RiskStateLoad.LOADED else ""
    pinned = hash_witness is not None and hash_witness.state_hash == observed
    halt_witness = facts.halt_witness
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

    if hash_witness is not None and not pinned:
        # The two are the same function of the same fields, so a campaign that
        # stopped cleanly restarts with them equal. Which way they disagree is
        # what decides whether this is a refusal or a crash to recover from, and
        # the log itself answers it: a state the campaign has already moved past
        # is one an EARLIER record still records. Whether the campaign is halted
        # is asked of the whole history too -- a matching state already carries
        # the halt its witness recorded, and this one does not match.
        history = _whole_history(state_dir)
        halt_witness = _newest_halt_event(history)
        matched = next(
            (
                witness
                for witness in reversed(history)
                if witness.seq < hash_witness.seq and witness.state_hash == observed
            ),
            None,
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

    if matched is not None:
        assert hash_witness is not None  # `matched` is only looked for against one
        return verdict(
            ContinuityOutcome.STATE_HASH_REGRESSED,
            f"the risk state on disk hashes to {observed}, which the log records at "
            f"the {matched.describe()} and not at its newest risk.state_hash -- the "
            f"{hash_witness.describe()}, which is {hash_witness.state_hash}. The "
            "campaign has already moved past the state this file holds, and every "
            "persist precedes the record that quotes it, so no crash produces this: "
            "risk.json was rolled back, restored from an older copy or replaced. "
            "Restore the copy the campaign last wrote (docs/demo_runbook.md, section 4)",
        )

    if facts.open_refusal_seq is not None:
        # Inside a refusal nothing short of exact proof settles it: no process of
        # this campaign wrote the file on disk -- the refusing processes write
        # nothing -- so a changed file, an identity no record quotes and a record
        # that merely COULD have moved the state all prove nothing.
        refusal = (
            f"the log holds a continuity refusal (RECOVERY record seq "
            f"{facts.open_refusal_seq}) that nothing has settled"
        )
        move = facts.unwitnessed_move
        if move is not None:
            return verdict(
                ContinuityOutcome.NOT_SETTLED,
                f"{refusal}, and after its newest risk.state_hash the campaign wrote a "
                f"{move.describe()}, which moves the risk state and records no identity: "
                "this log was written before those records carried one. Nothing it "
                "holds can tell the state that record left from a stale or fabricated "
                "one -- the newest witness's own state included, which it may have "
                "moved past -- so no file settles this refusal automatically "
                "(docs/demo_runbook.md, section 4)",
            )
        if hash_witness is None:
            seed = first_start_state_hash(capital)
            if observed != seed:
                return verdict(
                    ContinuityOutcome.NOT_SETTLED,
                    f"{refusal}, and the log has never recorded a risk.state_hash, so "
                    "the only state this campaign has ever held is the first-start "
                    f"seed, which hashes to {seed}. The risk state on disk hashes to "
                    f"{observed}. Restore the copy the campaign wrote "
                    "(docs/demo_runbook.md, section 4)",
                )
        elif not pinned:
            return verdict(
                ContinuityOutcome.NOT_SETTLED,
                f"{refusal}, and the risk state on disk hashes to {observed}, which no "
                "record quotes. That is not evidence of a restore: a file that appears "
                "after a refusal was written by no process of this campaign. Restore "
                "the copy the log's newest risk.state_hash describes -- the "
                f"{hash_witness.describe()}, which is {hash_witness.state_hash} "
                "(docs/demo_runbook.md, section 4)",
            )
        # Proved: the newest witness's identity exactly, nothing after it moved,
        # or -- on a log that never recorded one -- the first-start seed exactly.
        return verdict(ContinuityOutcome.CONTINUOUS)

    if hash_witness is not None and not pinned:
        # Outside a refusal this is the crash window: Aegis persisted and the
        # record quoting it was never committed.
        return verdict(ContinuityOutcome.STATE_AHEAD_OF_LOG)
    return verdict(ContinuityOutcome.CONTINUOUS)
