"""`DemoRunner`: section 8.1's state machine, and nothing that decides anything.

Section 8 chose a new production-shaped runner over evolving `tools/paper_run.py`,
which stays untouched as the generator of `artifacts/paper_smoke`.

The runner owns sequencing and evidence. It owns no arithmetic: sizing is the
rule's, execution is `HedgedPosition`'s, permission is `RiskEngine`'s, and cash
is `CarryLedger`'s. What it guarantees is the ORDER those happen in, and that
every one of them is written down.

Three orderings are load-bearing and are asserted by tests rather than trusted:

**State files, then the log record, then `last_record_hash`** (section 9.3). A
crash between the first and the second leaves a position with no record, which
is recoverable by reading the stores; a crash the other way round would leave a
record claiming a position that was never taken, which is not.

**The perpetual leg before the spot leg**, which is `HedgedPosition`'s and is
simply not overridden here.

**A veto skips EXECUTION and RECONCILIATION and goes forward to PERSISTENCE.**
Section 8.1's RISK_CHECK row says "veto -> log, back to PERSISTENCE"; PERSISTENCE
comes *after* RISK_CHECK in the same tick, so "back to" is a wording slip for
"forward to, skipping the two states between". Recorded here rather than
silently interpreted.

**No clock is read.** Every instant comes from `RunnerClock`, which is advanced
only by observed record instants, so a replay of the same files produces the
same records -- which is what section 10's byte comparison rests on.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from chimera.carry.hedge import PERP, SPOT, HedgedPosition, HedgeState
from chimera.carry.ledger import LoadOutcome
from chimera.demo.clock import RunnerClock
from chimera.demo.config import DemoConfig
from chimera.demo.decision_log import (
    LOG_DIR_NAME,
    DecisionLog,
    RecordKind,
    iso_minute,
    recover_tail,
    verify_log,
)
from chimera.demo.feed import (
    PERP_MARKET,
    SETTLEMENT_INSTANT_FIELD,
    FeedCursor,
    MarketState,
    plain_json,
    settlement_from_row,
)
from chimera.demo.rules import HedgeTarget, RuleDecision, RuleError, RuleRegistry
from chimera.demo.telemetry import RunnerTelemetry
from chimera.futures.domain import PositionSide
from chimera.futures.executor import (
    FlattenCause,
    ReconciliationOutcome,
    ReconciliationRequired,
)
from chimera.recorder.sink import write_json_atomic
from chimera.risk import RiskEngine

logger = logging.getLogger(__name__)

__all__ = [
    "DemoRunner",
    "RunnerState",
    "RunnerError",
    "TickOutcome",
    "RUNNER_STATE_SCHEMA",
    "RUNNER_STATE_SCHEMAS_READ",
    "RecoveryCause",
]

RUNNER_STATE_SCHEMA = "chimera.demo-runner-state/2"

#: The runner-state schemas this build will read, newest first.
#:
#: Version 2 adds ``last_reconcile_minute_ms``, the anchor of section 8.1's
#: hourly reconciliation. A version 1 file does not carry it, and its absence is
#: read as "no reconciliation has been performed", which makes the next complete
#: minute reconcile. That is the conservative direction and it is stated rather
#: than assumed: the other reading -- "reconciled just now" -- would let a
#: restart onto an old file postpone the check, which is precisely the
#: restart-avoidance the cadence is persisted to prevent.
#:
#: Nothing a version 1 file already carried changes meaning. The bump exists so
#: that the absence of the new field is a fact this build KNOWS about, rather
#: than a default it silently supplies under a schema id that promised the field
#: was never there.
RUNNER_STATE_SCHEMAS_READ: tuple[str, ...] = (
    RUNNER_STATE_SCHEMA,
    "chimera.demo-runner-state/1",
)

MINUTE_NS = 60_000_000_000
_MS_TO_NS = 1_000_000
ZERO = Decimal("0")
SPOT_LEG = "spot"
PERP_LEG = "perp"


class RunnerError(RuntimeError):
    """The runner cannot proceed, and proceeding anyway would produce false evidence."""


class RunnerState(str, Enum):
    """Section 8.1's states.

    RECOVERY appears in section 8.1's transition block but has no row in the
    state table, and RECOVER -- one character apart -- has a row. Recorded in the
    PR rather than resolved by invention: RECOVERY is implemented as the
    *transition* out of HALT, which lands in RECOVER, so it has no dwell and
    therefore needs no row. `RecordKind.RECOVERY` (PR-09's enum) remains the name
    of the log record written on that path.
    """

    STARTUP = "STARTUP"
    SELF_CHECK = "SELF_CHECK"
    RECOVER = "RECOVER"
    READY = "READY"
    DATA_READY = "DATA_READY"
    RULE_EVALUATION = "RULE_EVALUATION"
    RISK_CHECK = "RISK_CHECK"
    EXECUTION = "EXECUTION"
    RECONCILIATION = "RECONCILIATION"
    PERSISTENCE = "PERSISTENCE"
    REPORTING = "REPORTING"
    HALT = "HALT"
    SHUTDOWN = "SHUTDOWN"


#: Every state name, for the metric sweep. Module-level so `_enter` works from
#: the first line of `__init__`, before any instance attribute exists.
_STATE_NAMES: tuple[str, ...] = tuple(state.value for state in RunnerState)


class RecoveryCause(str, Enum):
    """Why section 9.3's RECOVERY record was written. A bounded vocabulary.

    The two crash causes are named for WHICH FILE IS BEHIND, and the direction is
    the whole content of the name:

    ``LOG_BEHIND_STATE``
        Section 9.3's own: the state files were written and the record was not,
        so the stores describe a position no record does.

    ``LOG_AHEAD_OF_STATE``
        The record was committed and the runner-state file naming it was not.
        Before PR-10R this condition was reported under the name
        ``LOG_BEHIND_STATE``, which says the opposite of what had happened. The
        old name is not reused for it and not redefined: the inverted label is
        replaced by the accurate one, and section 9.3's name keeps section 9.3's
        meaning.

    ``TORN_TAIL``
        A crash between the write and the ``fsync``. Not a claim about any
        record: the bytes never formed one.

    ``LOG_FORGED``
        A complete record that does not verify. Never recovered from, never
        repaired, and here only so that a refusal has a name.
    """

    TORN_TAIL = "TORN_TAIL"
    LOG_BEHIND_STATE = "LOG_BEHIND_STATE"
    LOG_AHEAD_OF_STATE = "LOG_AHEAD_OF_STATE"
    LOG_FORGED = "LOG_FORGED"


@dataclass(frozen=True)
class _Recovery:
    """What :meth:`DemoRunner._triage_log` found, carried to the RECOVERY record.

    Triage runs before the log can be appended to, so what it learns has to
    survive until there is somewhere to write it down.
    """

    cause: RecoveryCause
    reason: str
    recoverable: bool
    truncated_bytes: int = 0
    records_verified: int = 0
    #: The chain head the log actually ends on, adopted when the state file lagged.
    adopted_head: str = ""
    #: The minute the log's newest committed record carries, read during triage
    #: for the same reason the chain head is: `_append(STARTUP)` writes a record
    #: of its own, so by the time the RECOVERY record is written the "newest
    #: record" is the runner's own STARTUP and no longer the crash's.
    committed_minute_ms: int | None = None
    #: That record's KIND. A minute is finished by exactly one record, and it is
    #: not always the last one appended for it: FUNDING, RECONCILIATION and
    #: LIQUIDATION_TOUCH are all written for minute M BEFORE `mark_processed(M)`.
    #: So a tail of one of those means the minute was STARTED and never decided,
    #: which is a different crash from one that ends on the minute's DECISION.
    committed_minute_kind: str = ""


@dataclass
class TickOutcome:
    """What one minute did. Returned so a caller can assert on it."""

    minute_ms: int
    state: RunnerState
    kind: RecordKind
    decisions: list[RuleDecision] = field(default_factory=list)
    vetoed: bool = False
    executed: bool = False
    detail: str = ""
    record_hash: str = ""


class DemoRunner:
    """Section 12.3's interface: `DemoRunner(config, root, clock)`.

    ``root`` is the recorder's storage root, read-only. The runner's own files
    live under the configured ``state_dir``, which is never inside it.
    """

    def __init__(
        self,
        config: DemoConfig,
        root: str | Path,
        clock: RunnerClock | None = None,
        *,
        contract: Any,
        risk: RiskEngine,
        position: HedgedPosition,
        rules: RuleRegistry,
        capital: Decimal,
        software: Mapping[str, Any] | None = None,
        telemetry: Any | None = None,
    ) -> None:
        self.config = config
        self.root = Path(root)
        self.clock = clock or RunnerClock()
        self.contract = contract
        self.risk = risk
        self.position = position
        self.rules = rules
        self.capital = Decimal(capital)
        self.software = dict(software or {})

        self.state_dir = Path(config.runner_setting("state_dir"))
        self.state_dir.mkdir(parents=True, exist_ok=True)
        # Built before the first `_enter`, because `_enter` publishes through it.
        # Injectable so a test can drive the same campaign through a no-op
        # emitter and compare the two decision logs byte for byte; the runner
        # never reads anything back off it, so what is injected cannot decide.
        self.telemetry = (
            telemetry
            if telemetry is not None
            else RunnerTelemetry(
                state_dir=self.state_dir, states=_STATE_NAMES, rules=self.rules.ids
            )
        )
        self._enter(RunnerState.STARTUP)
        self.halt_reason: str | None = None

        persisted = self._load_state()
        self.cursor = FeedCursor(self.root, contract, persisted)
        # `save_state` writes `clock_now_ns` and nothing reads it back. That is
        # deliberate now, and the reason is worth keeping: an earlier revision of
        # this change DID carry it over, so a new process could record an
        # operator command without `start()`. It made `start()` raise on the very
        # crash section 9.3 exists for. `RunnerClock.observe` sets `started`, and
        # `start()` reads the feed only `elif not self.clock.started` -- so a
        # carried instant suppressed that read, and after a crash between
        # `_append` and `save_state` the carried instant is the PREVIOUS minute's
        # while the log's tail is the current one. `DecisionLog.append` refuses a
        # `runner_now_ns` that precedes the tail, so the STARTUP record raised
        # `DecisionLogError` and the campaign died on a traceback at exactly the
        # moment it was supposed to recover.
        #
        # The operator commands therefore still refuse rather than run on a
        # process that has not started; `resolve` fails before it touches
        # anything, which is the property that mattered. Making them usable needs
        # the clock seeded from the LOG's tail, not from the state file that
        # lagged it, and that is a change with its own crash matrix to prove.
        self.last_record_hash: str = persisted.get("last_record_hash", "")
        #: The minute the last reconciliation was performed for, or None when
        #: none has been. A version 1 state file carries no such field, and its
        #: absence means "none has been" -- see RUNNER_STATE_SCHEMAS_READ.
        raw_reconcile = persisted.get("last_reconcile_minute_ms")
        self.last_reconcile_minute_ms: int | None = (
            None if raw_reconcile is None else int(raw_reconcile)
        )
        self._log: DecisionLog | None = None
        self._config_hash = _config_hash(config)
        self._enter(RunnerState.STARTUP)

    def _enter(self, state: RunnerState) -> RunnerState:
        """The one place `self.state` changes, so the gauge cannot drift from it.

        Section 8.1's STARTUP row opens a metrics endpoint and REPORTING emits
        metrics; a state gauge updated at each call site would eventually miss
        one, and a dashboard showing the wrong state during an incident is worse
        than one showing none.

        What is published is the state's NAME, by value, to something that
        returns nothing. `self.state` is already set when the call is made, so
        even an emitter that raised could not leave the runner in a state it does
        not believe it is in.
        """
        self.state = state
        self.telemetry.on_state(state.value)
        return state

    # ------------------------------------------------------------------
    # persisted runner state
    # ------------------------------------------------------------------
    @property
    def state_path(self) -> Path:
        return self.state_dir / "runner_state.json"

    def _load_state(self) -> dict[str, Any]:
        path = self.state_dir / "runner_state.json"
        if not path.is_file():
            return {}
        import json

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RunnerError(
                f"the runner state at {path} could not be read: {exc}. It is refused rather "
                "than replaced, because starting from an empty cursor would reprocess "
                "minutes that already have decision records"
            ) from exc
        if payload.get("schema") not in RUNNER_STATE_SCHEMAS_READ:
            raise RunnerError(
                f"{path} declares schema {payload.get('schema')!r}, not one of "
                f"{list(RUNNER_STATE_SCHEMAS_READ)}"
            )
        return payload

    def save_state(self) -> None:
        """Atomic, via the recorder's own writer rather than a second one."""
        payload = {
            "schema": RUNNER_STATE_SCHEMA,
            "campaign_id": self.config.campaign_id,
            "last_minute_processed": self.cursor.last_minute_processed,
            "last_record_hash": self.last_record_hash,
            "clock_now_ns": self.clock.now_ns if self.clock.started else None,
            # Section 8.1's "every 60 minutes", anchored to a PROCESSED MINUTE
            # and persisted. Both halves are load-bearing. Anchoring to the
            # minute rather than to a counter of ticks since this process started
            # is what makes the schedule a function of the campaign's minutes, so
            # a replay of those minutes reconciles at the same ones; persisting
            # it is what stops a restart every fifty-nine minutes from postponing
            # the check for ever.
            "last_reconcile_minute_ms": self.last_reconcile_minute_ms,
        }
        write_json_atomic(self.state_path, payload)

    # ------------------------------------------------------------------
    # the log
    # ------------------------------------------------------------------
    @property
    def log(self) -> DecisionLog:
        if self._log is None:
            self._log = DecisionLog.open(self.state_dir)
        return self._log

    def _append(self, kind: RecordKind, minute_ns: int, payload: Mapping[str, Any]) -> str:
        """Write one record, then move `last_record_hash`.

        Section 9.3's ordering: the chain head only moves once the record it
        names is on disk, so a crash between the two re-reads a log whose tail
        is one ahead of the persisted head, which SELF_CHECK detects.
        """
        record = {
            "kind": kind.value,
            "minute": iso_minute(minute_ns),
            "runner_now_ns": self.clock.now_ns,
            "config_hash": self._config_hash,
            "software": self.software,
            **dict(payload),
        }
        try:
            appended = self.log.append(record)
        except Exception:
            # Counted, then re-raised unchanged. A failed append is a failure of
            # the evidence and the runner's behaviour on it is not this line's to
            # change; without the counter the failure is visible only in a log
            # file nobody is watching at 03:00.
            self.telemetry.on_log_write_error()
            raise
        self.last_record_hash = appended.record_hash
        self.telemetry.on_record(kind.value)
        return appended.record_hash

    # ------------------------------------------------------------------
    # STARTUP -> SELF_CHECK -> RECOVER -> READY
    # ------------------------------------------------------------------
    def start(self, *, now_ns: int | None = None, allow_dirty: bool = False) -> RunnerState:
        """Run the three states before the tick loop. Returns where it stopped."""
        self._enter(RunnerState.STARTUP)
        if now_ns is not None:
            self.clock.observe(int(now_ns))
        elif not self.clock.started:
            first = self.cursor.next_minute_ms()
            if first is None:
                raise RunnerError(
                    "there is no minute to start from: the recorder has written no "
                    "normalized day under this root. The runner does not invent a start "
                    "instant; a campaign begins where the evidence begins"
                )
            self.clock.observe(first * _MS_TO_NS + MINUTE_NS)

        # Triage the log BEFORE the first append, and the order is the whole
        # point. Appending opens the log, and `DecisionLog.open` refuses a tail
        # that does not verify -- so a crash that left a torn tail made every
        # branch of the recovery path below unreachable through the only entry
        # point there is: `start()` raised `DecisionLogTailError` and the
        # campaign died on a traceback. Triage also has to read the PERSISTED
        # chain head before the STARTUP record replaces it, or section 9.3's
        # "its record_hash equals last_record_hash" compares the runner's own
        # STARTUP record against itself and can never disagree.
        triage = self._triage_log()
        if triage is not None and not triage.recoverable:
            return self._halt(triage.reason)

        self._append(
            RecordKind.STARTUP,
            self._minute_ns(),
            {
                "campaign_id": self.config.campaign_id,
                "profile": self.config.profile.value,
                # `protocol_frozen` always; `protocol_hash` only once there IS
                # one. The log holds every field whose name ends in `_hash` to
                # sha256:<64 hex>, and a campaign whose S2 protocol is not yet
                # frozen has a null one -- so the fact is recorded as a boolean
                # rather than by writing a null into a hash field, and the
                # record still says plainly that no protocol governed the run.
                "protocol_frozen": self.config.protocol_frozen,
                **(
                    {"protocol_hash": self.config.protocol_hash}
                    if self.config.protocol_hash
                    else {}
                ),
                "allow_dirty": bool(allow_dirty),
                "rules": list(self.rules.ids),
            },
        )

        self._enter(RunnerState.SELF_CHECK)
        problem = self.self_check(allow_dirty=allow_dirty)
        if problem is not None:
            return self._halt(problem)

        self._enter(RunnerState.RECOVER)
        if triage is not None:
            self._write_recovery(triage)
        outcome = self.position.reconstruct()
        if outcome.state is HedgeState.DISPUTED:
            return self._halt(f"dispute: {outcome.detail}")
        if self.risk.check_kill_switch():
            return self._halt("kill_switch")
        if self.risk.state.halted:
            return self._halt(self.risk.state.halt_reason or "risk halted")

        self._enter(RunnerState.READY)
        self.save_state()
        return self.state

    def self_check(self, *, allow_dirty: bool = False) -> str | None:
        """Section 8.1's SELF_CHECK row. Returns a reason, or None if sound.

        The source-identity check is the one `--allow-dirty` may skip, and only
        for a soak: a campaign whose records are evidence may not be produced by
        a working tree nobody can reconstruct.
        """
        if not allow_dirty and self.software.get("dirty"):
            return (
                "source_identity: the working tree is dirty and this is not a soak run. "
                "A campaign's records name the revision that produced them"
            )
        if allow_dirty and self.config.profile.value == "CAMPAIGN":
            return (
                "allow_dirty was requested for a CAMPAIGN profile. --allow-dirty is for "
                "soak runs, whose records are operational rather than evidence"
            )
        try:
            self.position.spot.store.state  # noqa: B018 - loadable is the assertion
            self.position.perp.store.state  # noqa: B018
        except Exception as exc:  # pragma: no cover - defensive
            return f"store_error: {exc}"
        if self.position.ledger.disputed:
            return f"dispute: {self.position.ledger.disputed}"
        return None

    # ------------------------------------------------------------------
    # section 9.3: what a crash left behind, and what may be done about it
    # ------------------------------------------------------------------
    def _triage_log(self) -> "_Recovery | None":
        """What a crash left behind, decided BEFORE anything is appended.

        Returns None when there is nothing to recover. Otherwise a
        :class:`_Recovery` naming what was found; ``recoverable`` is False only
        for a forgery.

        Three things have to happen here rather than later, and each of them was
        wrong when they happened later:

        * **The torn tail is repaired first.** Opening the log for writing
          refuses a tail that does not verify, so a repair attempted after the
          first append never runs.
        * **The persisted chain head is read before it moves.**
          :meth:`_append` assigns ``self.last_record_hash``, so a comparison made
          after the STARTUP record compares that record against itself.
        * **The store and ledger are compared against the log before
          ``reconstruct()``**, because that is the only moment at which the two
          sides can still disagree about the crash.

        Section 9.3 names one cause and this build names four, because four are
        distinguishable and only one of them is section 9.3's. ``LOG_BEHIND_STATE``
        keeps section 9.3's meaning -- the state files are on disk and the record
        is not. ``LOG_AHEAD_OF_STATE`` is the opposite window, which this runner
        used to report under section 9.3's name; the inverted label is replaced
        rather than redefined. ``TORN_TAIL`` is a crash between the write and the
        ``fsync``, and ``LOG_FORGED`` is a complete record that is wrong, which no
        crash produces and nothing here repairs.
        """
        root = self.state_dir / LOG_DIR_NAME
        persisted_head = self.last_record_hash
        committed = self._last_record_minute()
        verification = verify_log(root)
        if verification.is_forged:
            return _Recovery(
                cause=RecoveryCause.LOG_FORGED,
                reason=(
                    "log_forged: a complete decision record does not verify "
                    f"({verification.summary()}). A crash cannot produce one, so it is "
                    "not repaired, not truncated and not continued past"
                ),
                recoverable=False,
            )

        cause: RecoveryCause | None = None
        detail = ""
        repaired_bytes = 0
        if verification.is_torn:
            repair = recover_tail(self.state_dir)
            repaired_bytes = repair.truncated_bytes
            cause = RecoveryCause.TORN_TAIL
            detail = (
                f"{repaired_bytes} unterminated byte(s) removed from "
                f"{repair.path.name if repair.path else 'the log'} and preserved beside "
                "it; the record they would have formed was never committed"
            )
            verification = verify_log(root)

        tail = verification.last_record_hash or ""
        if cause is None and persisted_head and tail != persisted_head:
            # Only when nothing else was found. A torn tail leaves the head
            # disagreeing too, and reporting that as LOG_AHEAD_OF_STATE would
            # lose the truncation -- the more specific cause, and the one whose
            # bytes were removed.
            cause = RecoveryCause.LOG_AHEAD_OF_STATE
            detail = (
                f"the log's tail is {tail} and the runner state names {persisted_head}. "
                "The record was committed and the state file naming it was not, so the "
                "log is one record ahead of the state"
            )
        elif cause is None:
            behind = self._state_ahead_of_log()
            if behind:
                cause = RecoveryCause.LOG_BEHIND_STATE
                detail = (
                    f"{behind} The state files were written and the record was not, so "
                    "the log is behind the state"
                )

        if cause is None:
            return None
        return _Recovery(
            cause=cause,
            reason=detail,
            recoverable=True,
            truncated_bytes=repaired_bytes,
            records_verified=verification.records,
            # The chain head is adopted from the log, which is the file that
            # actually holds the records; the state file is the one that lagged.
            adopted_head=tail,
            committed_minute_ms=None if committed is None else committed[0],
            committed_minute_kind="" if committed is None else committed[1],
        )

    def _save_ledger(self) -> None:
        """Persist the carry ledger, unless it is the placeholder for a damaged one.

        `CarryLedger.save` refuses an UNREADABLE ledger outright, which is right:
        writing the placeholder back would replace the only record of the
        campaign's cash with one saying it never traded. But that refusal is a
        `LedgerError`, and on the operator paths it fired AFTER
        `emergency_reduce` had already moved both legs and BEFORE the OPERATOR
        record -- so the command that the runbook tells an operator to reach for
        ended in a traceback with the position changed and nothing written down,
        which is the same shape as the defect the refusal was added to prevent.

        There is nothing to persist for a ledger that holds no real state, so
        skipping is not a loss; the dispute is already on the position and the
        record still gets written. The damaged bytes stay on disk either way.
        """
        if self.position.ledger.outcome is LoadOutcome.UNREADABLE:
            logger.warning(
                "Carry ledger at %s is UNREADABLE; not persisting the placeholder. "
                "The damaged file is left exactly as it is.",
                self.position.ledger.path,
            )
            return
        self.position.ledger.save()

    def _require_recordable(self, command: str) -> int:
        """The instant to record with, or a refusal that has changed nothing.

        Both halves of "can this be written down" belong before the first
        mutation, and checking only one of them is how this went wrong twice.
        `DecisionLog.open` verifies the tail and refuses a torn one -- and the
        CLI never runs `start()`, so triage never repairs it -- while `append`
        refuses a ``runner_now_ns`` that precedes the tail's. An operator command
        that saved a store and cleared Aegis's dispute and only THEN discovered
        either has done the one thing section 8.3 forbids: changed the safety
        state with nothing in the log to say who did it or why.

        Raising :class:`RunnerError` is part of the contract, not incidental:
        ``tools/demo_run.py`` catches that and nothing else, so a
        ``RunnerClockError`` or ``DecisionLogError`` escaping here reaches the
        operator as a traceback.
        """
        try:
            now_ns = self.clock.now_ns
            log = self.log
        except Exception as exc:  # RunnerClockError, DecisionLogError, and kin
            raise RunnerError(
                f"{command} cannot be recorded ({exc}). Changing state that no record "
                "explains is worse than not changing it, so nothing has been changed."
            ) from exc
        if log.last_runner_now_ns is not None and now_ns < log.last_runner_now_ns:
            raise RunnerError(
                f"{command} cannot be recorded: the runner clock is at {now_ns} and the "
                f"log's last record is at {log.last_runner_now_ns}, so the record would "
                "be refused as moving the clock backwards. Nothing has been changed."
            )
        return now_ns

    def _last_record_minute(self) -> tuple[int, str] | None:
        """The newest committed record's ``(minute_ms, kind)``, or None if empty."""
        from chimera.demo.decision_log import day_files, read_records

        for path in reversed(day_files(self.state_dir / LOG_DIR_NAME)):
            newest: tuple[str, str] | None = None
            for record in read_records(path):
                minute = record.get("minute")
                if isinstance(minute, str) and minute:
                    newest = (minute, str(record.get("kind", "")))
            if newest is not None:
                stamp, kind = newest
                return int(datetime.fromisoformat(stamp).timestamp() * 1000), kind
        return None

    #: The kinds a minute can be left in the MIDDLE of, and only those.
    #:
    #: Each is appended for minute M *before* `mark_processed(M)`, so a log
    #: ending on one says the minute was started and never decided.
    #:
    #: Defined as the exception rather than the rule, and that direction is the
    #: point. The first version of this named the three kinds that DO finish a
    #: minute and treated everything else as unfinished -- which swept in HALT,
    #: STARTUP, SHUTDOWN, OPERATOR, RESUME and RECOVERY. None of those is a
    #: minute's record at all: they are stamped with `_minute_ns()`, the last
    #: minute already PROCESSED, whose DECISION is committed. A crash between
    #: `_append(HALT)` and `save_state` therefore excluded a decided minute from
    #: the parity comparison and dropped its DECISION with it -- exactly the harm
    #: `EXCLUDING_RECOVERY_CAUSES` refuses LOG_AHEAD_OF_STATE to avoid. Listing
    #: the exceptions means an unlisted kind defaults to "nothing was lost",
    #: which is the direction that never silently drops evidence.
    _MID_MINUTE_KINDS = frozenset(
        {
            RecordKind.FUNDING.value,
            RecordKind.RECONCILIATION.value,
            RecordKind.LIQUIDATION_TOUCH.value,
        }
    )

    def _minute_was_finished(self, triage: "_Recovery") -> bool:
        """Whether the log's tail leaves its minute finished.

        True for a tail that finished the minute (DECISION, INCOMPLETE_STATE,
        SKIPPED_STALE) and true for a tail that is not a minute's record at all
        (HALT, STARTUP, SHUTDOWN, OPERATOR, RESUME, RECOVERY) -- in both cases
        there is nothing to exclude.
        """
        return triage.committed_minute_kind not in self._MID_MINUTE_KINDS

    def _affected_minute_ms(self, triage: "_Recovery") -> int | None:
        """The minute the crash actually left in doubt.

        Not ``last_minute_processed``, which is what this used to be and is the
        last minute that COMPLETED. ``mark_processed`` runs before ``_append``
        and ``save_state`` runs after it, so a crash anywhere in that window
        leaves the persisted cursor naming the previous minute. Excluding that
        one from the parity comparison dropped a minute whose record is
        committed, complete and comparable -- the exact harm
        :data:`EXCLUDING_RECOVERY_CAUSES` refuses ``LOG_AHEAD_OF_STATE`` to avoid
        -- while the minute the crash really touched was left in and diverged.

        For ``LOG_AHEAD_OF_STATE`` the log itself holds the answer, so it is read
        rather than derived: the committed record's own minute -- whether or not
        that record FINISHED the minute. See :meth:`_minute_was_finished`: a tail
        of FUNDING, RECONCILIATION or LIQUIDATION_TOUCH means the minute was
        started and never decided, and that minute is still the affected one.
        """
        if triage.cause is RecoveryCause.LOG_AHEAD_OF_STATE:
            if triage.committed_minute_ms is not None:
                return triage.committed_minute_ms
        return self.cursor.next_minute_ms()

    def _write_recovery(self, triage: "_Recovery") -> None:
        """Section 9.3's RECOVERY record, written once the log is appendable."""
        if triage.adopted_head:
            self.last_record_hash = triage.adopted_head
        affected = self._affected_minute_ms(triage)
        # Whether the tail FINISHED its minute decides what may be claimed about
        # it. A DECISION (or INCOMPLETE_STATE, or SKIPPED_STALE) is the minute's
        # last word, so nothing is excluded and the cursor moves past it. A
        # FUNDING, RECONCILIATION or LIQUIDATION_TOUCH tail is not: those are
        # appended for minute M before `mark_processed(M)`, so the minute holds
        # part of its evidence and no decision. Re-deciding it would append a
        # SECOND record of that kind for the minute, so the cursor still moves --
        # and the minute is then excluded, because half a minute's evidence is
        # what section 9.3's exclusion is for. Advancing without excluding was a
        # silent hole: the minute vanished from the campaign with nothing saying
        # so, and the replay's DECISION for it came back as an unexplainable
        # `replay_only`.
        finished = self._minute_was_finished(triage)
        if triage.cause is RecoveryCause.LOG_AHEAD_OF_STATE and affected is not None:
            # The record for this minute is committed and complete; only the
            # state file naming it was lost. Leaving the cursor behind it made
            # `catch_up` decide the minute a second time, so the log ended up
            # holding TWO DECISION records for one minute -- one saying the
            # position opened and one, from a now-flat re-decision, saying it did
            # not. That is evidence corruption, and parity reported it as an
            # unexplainable `live_only` record. Adopting the log's minute is the
            # same choice as adopting the log's chain head, for the same reason:
            # the log is the file that actually holds the records.
            self.cursor.mark_processed(affected)
        minute_ns = int(affected) * _MS_TO_NS if affected is not None else self._minute_ns()
        self._append(
            RecordKind.RECOVERY,
            minute_ns,
            {
                "recovery": {
                    "cause": triage.cause.value,
                    "detail": triage.reason,
                    "affected_minute": iso_minute(minute_ns),
                    "truncated_bytes": triage.truncated_bytes,
                    "records_verified": triage.records_verified,
                    # Section 9.3: "the affected minute is excluded from the
                    # campaign's evidence and counted in the monthly report".
                    #
                    # Null for LOG_AHEAD_OF_STATE, and that is the whole point of
                    # separating the causes. There the record WAS committed and
                    # is complete, canonical and correctly linked, so there is
                    # nothing to exclude and excluding it would drop real
                    # evidence -- which is exactly why
                    # `tools/replay_parity.py::EXCLUDING_RECOVERY_CAUSES` leaves
                    # that cause out. Writing a minute here that the parity tool
                    # then declines to read would be two files disagreeing about
                    # what this campaign excluded.
                    "evidence_excluded_minute": (
                        None
                        if triage.cause is RecoveryCause.LOG_AHEAD_OF_STATE and finished
                        else iso_minute(minute_ns)
                    ),
                    # Null above means "nothing was lost". This says which of the
                    # two LOG_AHEAD_OF_STATE shapes it was, so a reader never has
                    # to infer it from the absence of a field.
                    "minute_finished": finished,
                },
                "veto_or_rejection": {
                    "stage": "recovery",
                    "label": triage.cause.value.lower(),
                    "detail": triage.reason,
                },
            },
        )
        self.save_state()
        logger.warning("Runner RECOVERED: %s: %s", triage.cause.value, triage.reason)

    def _state_ahead_of_log(self) -> str:
        """What the persisted state holds that the log's last record does not.

        Section 9.3's own crash window, and the only one that leaves both files
        internally consistent: each verifies, each is complete, and the two
        disagree about what happened. Returns a description, or an empty string
        when they agree.

        Two things are compared, because a position change is not the only thing
        a lost record can hide. The **legs** catch a fill that was persisted and
        never recorded; the **ledger** catches a funding settlement or a set of
        frictions that was booked and never recorded -- which moves no quantity
        at all, and which the leg comparison alone therefore cannot see.

        Both sides are read from files that are already loaded: the executors'
        stores and the carry ledger. ``self.position.state`` is deliberately NOT
        consulted -- it is an in-memory attribute that reads FLAT on a freshly
        constructed runner whatever the stores hold, and reading it here (before
        ``reconstruct()`` has run) made every branch of this answer False.
        """
        last_position = self._last_block("position_after")
        legs = self._position_block()
        if last_position is None:
            if legs["spot_qty"] != "0" or legs["perp_qty"] != "0":
                return (
                    f"the stores hold {legs['spot_qty']} spot and {legs['perp_qty']} perp "
                    "and no record in the log describes a position."
                )
        else:
            for field_name in ("spot_qty", "perp_qty"):
                if str(last_position.get(field_name)) != legs[field_name]:
                    return (
                        f"the stores hold {field_name}={legs[field_name]} and the log's "
                        f"last position record says {last_position.get(field_name)}."
                    )

        last_ledger = self._last_block("ledger_effect")
        if last_ledger is not None:
            ledger = self.position.ledger.state
            booked = {
                "funding": str(ledger.net_funding),
                "fees": str(ledger.fees),
                "slippage": str(ledger.slippage),
                "realised": str(ledger.realised),
            }
            for field_name, value in booked.items():
                if str(last_ledger.get(field_name)) != value:
                    return (
                        f"the carry ledger holds {field_name}={value} and the log's last "
                        f"ledger record says {last_ledger.get(field_name)}."
                    )
        return ""

    def _last_block(self, name: str) -> Mapping[str, Any] | None:
        """The newest ``name`` block in the log, or None if no record carries one."""
        from chimera.demo.decision_log import day_files, read_records

        for path in reversed(day_files(self.state_dir / LOG_DIR_NAME)):
            found: Mapping[str, Any] | None = None
            for record in read_records(path):
                block = record.get(name)
                if isinstance(block, Mapping):
                    found = block
            if found is not None:
                return found
        return None

    # ------------------------------------------------------------------
    # the tick loop
    # ------------------------------------------------------------------
    def tick(self, minute_ms: int) -> TickOutcome:
        """One minute, through section 8.1's tick loop."""
        minute_ns = int(minute_ms) * _MS_TO_NS
        self.clock.observe(minute_ns + MINUTE_NS)

        self._enter(RunnerState.DATA_READY)
        try:
            state = self.cursor.state_for(minute_ms, now_ns=self.clock.now_ns)
        except Exception as exc:
            # A feed the runner cannot read is a halt, never a traceback out of
            # the tick loop. `state_for` reads the funding settlements too (for
            # the minute's `funding_last` input), so an unreadable settlements
            # row reaches this line before `_settle_funding` ever runs -- and a
            # process that died here would leave no HALT record, no reason and no
            # saved state, which is the one outcome section 8.1 has no row for.
            reason = f"feed_unreadable: {exc}"
            self._halt(reason)
            return TickOutcome(minute_ms, self.state, RecordKind.HALT, detail=reason)
        self.risk.note_feed(minute_ns + MINUTE_NS, self.clock.now_ns)
        # Counted here, before anything is decided about the minute, because the
        # quantity is "minutes attempted": a minute that halts inside a rule is
        # still one the runner took on, and a counter that skipped it would read
        # healthiest exactly when the campaign was failing. Nothing in this call
        # is decision-derived, and nothing reads it back.
        self.telemetry.on_minute(minute_ns=minute_ns, missing=state.missing)

        # The minute's book, on the model both legs fill against, and the model's
        # clock moved whether or not one arrived. Before the incomplete branch,
        # because a minute with no book still has to age the previous one.
        self.position.install_quote(state)

        if not state.complete:
            return self._incomplete(minute_ms, state)

        if self.risk.check_kill_switch():
            self._halt("kill_switch")
            return TickOutcome(minute_ms, self.state, RecordKind.HALT, detail="kill_switch")

        # Section 6.5's funding, before anything decides. A settlement is a cash
        # flow that has ALREADY happened by this minute's close, so booking it
        # first is what makes the equity the rule is sized against, the equity
        # Aegis judges, and the equity section 6.7's liquidation check reads all
        # describe the same instant. Booking it after the decision would have the
        # minute decided on money the position no longer had.
        problem = self._settle_funding(minute_ms, state)
        if problem is not None:
            self._halt(problem)
            return TickOutcome(minute_ms, self.state, RecordKind.HALT, detail=problem)

        # Section 6.7, per minute while HEDGED or PARTIAL, and before the rule
        # for the same reason: a touched position is flattened and halted, and a
        # rule that had already sized an increase against it would be sizing
        # against a position that no longer exists.
        touched = self._liquidation_check(minute_ms, state)
        if touched is not None:
            return touched

        self._enter(RunnerState.RULE_EVALUATION)
        portfolio = self._portfolio(state)
        decisions: list[RuleDecision] = []
        for rule in self.rules:
            try:
                decision = rule.evaluate(state, portfolio)
                # Defence in depth. `RuleDecision.is_actionable` is decided by
                # the TYPE of the target, which is the guarantee that matters:
                # a shadow rule returns `SignalOnly` and `HedgedPosition.plan`
                # cannot accept one. This cross-check catches the other
                # direction -- a rule that DECLARES itself signal-only and then
                # returns a `HedgeTarget` -- which the type alone cannot, and
                # which would be a shadow rule quietly acquiring the ability to
                # trade.
                if decision.is_actionable and not getattr(rule, "actionable", False):
                    raise RuleError(
                        f"{rule.rule_id} declares actionable=False but returned a "
                        "HedgeTarget. A signal-only rule may never size a position"
                    )
                decisions.append(decision)
            except RuleError as exc:
                self._halt(f"rule_exception: {rule.rule_id}: {exc}")
                return TickOutcome(
                    minute_ms, self.state, RecordKind.HALT, detail=f"rule_exception: {exc}"
                )
            except Exception as exc:  # a rule may raise anything; none of it decides
                self._halt(f"rule_exception: {rule.rule_id}: {exc!r}")
                return TickOutcome(
                    minute_ms, self.state, RecordKind.HALT, detail=f"rule_exception: {exc!r}"
                )

        return self._decide(minute_ms, state, decisions, portfolio)

    def _decide(
        self,
        minute_ms: int,
        state: MarketState,
        decisions: Sequence[RuleDecision],
        portfolio: Mapping[str, Any],
    ) -> TickOutcome:
        minute_ns = int(minute_ms) * _MS_TO_NS
        actionable = [d for d in decisions if d.is_actionable]
        if len(actionable) > 1:
            self._halt(
                "more than one rule produced an actionable target: "
                f"{[d.rule_id for d in actionable]}. A campaign runs exactly one"
            )
            return TickOutcome(
                minute_ms, self.state, RecordKind.HALT, decisions=list(decisions)
            )

        self._enter(RunnerState.RISK_CHECK)
        intents: list[Any] = []
        veto: dict[str, Any] | None = None
        if actionable:
            target = actionable[0].target
            assert isinstance(target, HedgeTarget)
            target = self._target_to_act_on(target)
            intents = self.position.plan(target, state)
            if not intents and target.quantity != self.position.leg("spot").quantity:
                veto = {
                    "stage": "risk",
                    "label": "no_intent",
                    "detail": (
                        "the position planned nothing for a target that differs from the "
                        "held quantity: Aegis, a halt, or an incomplete minute refused it"
                    ),
                }

        executed = False
        outcome = None
        orders_before = _order_ids(self.position)
        if intents and veto is None:
            self._enter(RunnerState.EXECUTION)
            try:
                outcome = self.position.apply(
                    intents, state, equity=float(portfolio["equity"])
                )
                executed = True
            except ReconciliationRequired as exc:
                self._halt(f"dispute: {exc}")
                return TickOutcome(
                    minute_ms, self.state, RecordKind.HALT, decisions=list(decisions)
                )

            self._enter(RunnerState.RECONCILIATION)
            if outcome.state is HedgeState.DISPUTED:
                self._halt(f"dispute: {outcome.detail}")
                return TickOutcome(
                    minute_ms, self.state, RecordKind.HALT, decisions=list(decisions)
                )

        # Section 8.1's RECONCILIATION row: "every 60 minutes and after any
        # execution". Both arms land here, and a veto reaches neither -- a minute
        # that changed nothing has nothing new to compare, and the periodic arm
        # is what covers the long stretches of holding.
        if executed or self._reconcile_due(minute_ms):
            self._enter(RunnerState.RECONCILIATION)
            reason = self._reconcile(minute_ms, state, after_execution=executed)
            if reason is not None:
                self._halt(reason)
                return TickOutcome(
                    minute_ms, self.state, RecordKind.HALT, decisions=list(decisions)
                )

        # Section 8.1's RISK_CHECK exit: a veto skips EXECUTION and
        # RECONCILIATION and continues FORWARD to PERSISTENCE.
        self._enter(RunnerState.PERSISTENCE)
        mark = self.position.mark_to_market(state)
        if self.position.state is HedgeState.DISPUTED:
            self._halt("identity_violation")
            return TickOutcome(
                minute_ms, self.state, RecordKind.HALT, decisions=list(decisions)
            )
        self.position.ledger.save()
        self.risk.update_equity(float(mark.equity))
        self.cursor.mark_processed(minute_ms)

        record_hash = self._append(
            RecordKind.DECISION,
            minute_ns,
            self._decision_payload(
                state, decisions, intents, veto, outcome, mark, orders_before
            ),
        )
        self.save_state()

        self._enter(RunnerState.REPORTING)
        # Section 8.1's REPORTING state, and the only place a decision-derived
        # number reaches Prometheus. It runs after the DECISION record and the
        # runner state file are both on disk, so every series it moves describes
        # a minute that is already evidence; the emitter returns nothing and the
        # tick's outcome below does not mention it.
        self.telemetry.on_reporting(
            position=self.position,
            risk=self.risk,
            mark=mark,
            market=state,
            decisions=decisions,
            veto=veto,
        )
        self._enter(RunnerState.READY)
        return TickOutcome(
            minute_ms,
            self.state,
            RecordKind.DECISION,
            decisions=list(decisions),
            vetoed=veto is not None,
            executed=executed,
            record_hash=record_hash,
        )

    # ------------------------------------------------------------------
    # reconciliation (section 8.1)
    # ------------------------------------------------------------------
    def _reconcile_due(self, minute_ms: int) -> bool:
        """Whether this minute is one of section 8.1's hourly ones.

        A pure function of the processed minute and the persisted anchor, with no
        clock and no counter of ticks. That is what makes it survive both a
        replay and a restart: the replay walks the same minutes and reconciles at
        the same ones, and a process that restarts every fifty-nine minutes still
        meets the sixtieth, because the anchor is the last minute a
        reconciliation was PERFORMED FOR and not the last time this process
        started.

        With no anchor -- a fresh campaign, or a version 1 state file -- the
        answer is yes. A campaign reconciles the first complete minute it sees.
        """
        every = int(self.config.runner_setting("reconcile_every_minutes"))
        if every <= 0:
            return False
        if self.last_reconcile_minute_ms is None:
            return True
        return int(minute_ms) - self.last_reconcile_minute_ms >= every * 60_000

    def _reconcile(
        self, minute_ms: int, state: MarketState, *, after_execution: bool
    ) -> str | None:
        """Compare both legs against the venue's view. Returns a halt reason.

        The comparison itself is :meth:`FuturesExecutor.reconcile`, unchanged and
        not reimplemented: it is already the fail-closed one. A mismatch there
        marks the symbol disputed in the store, moves every open order on it to
        ``RECONCILIATION_REQUIRED`` -- which ``require_ready`` then refuses to
        plan through -- and raises ``chimera_futures_reconciliation_total``. What
        this method adds is the three things the plan asks of the runner and the
        executor cannot do for itself: tell Aegis (section 7.3's
        ``note_reconciliation``, which vetoes increases on the disputed symbol
        and SURVIVES a restart, cleared only by an operator note), write the
        evidence, and halt.

        The record is written on agreement too. A reconciliation that ran and
        agreed is the evidence that the check is alive; a log that only ever held
        mismatches could not tell "reconciled hourly, always agreed" from "never
        reconciled", which is exactly the state this build was in before PR-10R.
        """
        minute_ns = int(minute_ms) * _MS_TO_NS
        legs = (
            (SPOT_LEG, self.position.spot, self.position.config.spot_symbol, state.spot_close),
            (PERP_LEG, self.position.perp, self.position.config.perp_symbol, state.mark),
        )
        results: list[dict[str, Any]] = []
        mismatched: list[str] = []
        for leg, executor, symbol, reference in legs:
            try:
                report = executor.reconcile(symbol, mark_price=reference)
            except Exception as exc:
                return f"reconciliation_error: {leg} {symbol}: {exc}"
            agreed = report.outcome is ReconciliationOutcome.AGREED
            if not agreed:
                # Only ever OPENED here. `note_reconciliation(symbol, None)` is
                # the operator's clear (section 7.2: "cleared only by operator
                # note"), and calling it on an agreement would let two states
                # that came back into agreement for an unexplained reason -- a
                # later fill happening to land on the disputed quantity -- erase
                # the dispute by themselves. `FuturesExecutor.reconcile` refuses
                # exactly that for the store's copy; Aegis's copy is held to the
                # same rule.
                self.risk.note_reconciliation(symbol, report.detail)
            results.append(
                {
                    "leg": leg,
                    "symbol": symbol,
                    "outcome": report.outcome.value,
                    "local_side": report.local.side.value,
                    "local_qty": str(report.local.quantity),
                    "reported_side": report.reported.side.value,
                    "reported_qty": str(report.reported.quantity),
                    "detail": report.detail,
                }
            )
            if not agreed:
                mismatched.append(f"{leg}:{symbol}")

        self.last_reconcile_minute_ms = int(minute_ms)
        detail = "agreed" if not mismatched else f"mismatch on {', '.join(sorted(mismatched))}"
        self.position.ledger.save()
        self._append(
            RecordKind.RECONCILIATION,
            minute_ns,
            {
                "inputs": {
                    "contract_hash": _prefixed(getattr(self.contract, "contract_hash", "")),
                    "um_minute_digest": state.perp_digest,
                    "spot_minute_digest": state.spot_digest,
                    "inputs_hash": _inputs_hash(state),
                },
                "reconciliation": {
                    "trigger": "after_execution" if after_execution else "periodic",
                    "every_minutes": int(
                        self.config.runner_setting("reconcile_every_minutes")
                    ),
                    "legs": results,
                    "outcome": (
                        ReconciliationOutcome.AGREED.value
                        if not mismatched
                        else ReconciliationOutcome.MISMATCH.value
                    ),
                },
                "risk": {"state_hash": _risk_hash(self.risk), "decisions": []},
                "position_after": self._position_block(),
                "veto_or_rejection": (
                    None
                    if not mismatched
                    else {
                        "stage": "reconciliation",
                        "label": "reconciliation_mismatch",
                        "detail": detail,
                    }
                ),
            },
        )
        self.save_state()
        if mismatched:
            # Section 8.1: "mismatch -> HALT". The dispute is already recorded in
            # both the store and Aegis, and neither is cleared by anything but an
            # operator note, so a restart cannot walk away from it.
            return f"dispute: reconciliation_mismatch: {detail}"
        return None

    # ------------------------------------------------------------------
    # funding (section 6.5)
    # ------------------------------------------------------------------
    def _settle_funding(self, minute_ms: int, state: MarketState) -> str | None:
        """Book every recorded settlement this minute closed over. Returns a halt reason.

        The window is section 6.9's, and :meth:`HedgedPosition.settle_funding` is
        its authority: ``open_instant < settlement <= now``, with ``now`` the
        minute's close. Nothing here computes a funding number -- the rate, the
        mark and the notional all come from the recorded row and the position,
        and this method only decides WHICH settlements are in scope and writes
        down what was booked.

        **Exactly once**, across every path a settlement can be met twice:

        * two ticks -- the second sees the instant in ``CarryLedgerState.settled``
          and does not call the primitive at all, so no second record is written;
        * catch-up -- the same, because the window is anchored to the position's
          open instant and not to the previous tick;
        * a restart -- ``settled`` and the executor's ``applied_funding`` are both
          persisted before the next process starts;
        * a replay -- it reads the same rows in the same order from an empty
          state directory and books the same set;
        * a duplicated row in the settlements file -- the first copy books it and
          the second is skipped by the same membership test, so the log holds one
          effective record per settlement instant, which is what
          ``reports.FUNDING_SOURCE``'s per-record differencing requires.

        A settlement in scope that cannot be read -- no mark price, a rate that
        fails the plausibility refusal, another instrument's symbol -- halts the
        campaign. It is not skipped: skipping would leave the perpetual leg
        holding a position whose funding this build knows it did not book, with
        nothing in the log saying so.
        """
        ledger = self.position.ledger
        perp = self.position.leg(PERP_LEG)
        if perp.quantity == ZERO and self.position.leg(SPOT_LEG).quantity == ZERO:
            return None  # flat: section 6.5 charges the perpetual leg, and it holds nothing

        open_instant = ledger.state.open_instant_ns
        if open_instant is None:
            return (
                "funding_window_unknown: this position is not flat and the ledger records "
                "no open instant, so section 6.9's window `open_instant < settlement <= "
                "now` cannot be formed. A ledger written before the instant was recorded "
                "is refused rather than given a guessed lower bound"
            )

        minute_ns = int(minute_ms) * _MS_TO_NS
        now_ns = minute_ns + MINUTE_NS
        venue_symbol = _venue_symbol(self.contract)
        try:
            rows = list(self.cursor.settlements())
        except Exception as exc:
            return f"funding_source_unreadable: {exc}"

        for row in rows:
            try:
                instant_ns = int(row[_SETTLEMENT_FIELD]) * _MS_TO_NS
            except (KeyError, TypeError, ValueError) as exc:
                return f"funding_source_unreadable: {exc}"
            if not (open_instant < instant_ns <= now_ns):
                continue
            if instant_ns in ledger.state.settled:
                continue  # already booked: no economics, and no second record
            try:
                settlement = settlement_from_row(
                    row,
                    position_symbol=self.position.config.perp_symbol,
                    venue_symbol=venue_symbol,
                )
            except Exception as exc:
                return f"funding_unbookable: {exc}"

            try:
                flow = self.position.settle_funding(
                    settlement, open_instant_ns=open_instant, now_ns=now_ns
                )
            except Exception as exc:
                return f"funding_unbookable: {exc}"
            if instant_ns not in ledger.state.settled:
                # The primitive declined it. It agrees with the window this
                # method just applied, so a disagreement is a defect in one of
                # them and is refused rather than logged as a settlement.
                return (
                    f"funding_not_booked: settlement {settlement.settlement_id} fell in "
                    "this minute's window and the position did not book it"
                )

            # Aegis first, because the streak it keeps is what vetoes the next
            # increase, and the record written below carries the risk state hash.
            if perp.side in (PositionSide.LONG, PositionSide.SHORT):
                self.risk.note_funding_settlement(
                    self.position.config.perp_symbol, perp.side, float(settlement.rate)
                )

            mark = self.position.mark_to_market(state)
            if self.position.state is HedgeState.DISPUTED:
                return f"identity_violation: {self.position.ledger.disputed}"
            # Section 9.3's order: state files, then the record, then the head.
            self.position.ledger.save()
            self._append(
                RecordKind.FUNDING,
                minute_ns,
                self._funding_payload(state, settlement, flow, mark, row),
            )
            self.save_state()
        return None

    def _funding_payload(
        self,
        state: MarketState,
        settlement: Any,
        flow: Decimal,
        mark: Any,
        row: Mapping[str, Any],
    ) -> dict[str, Any]:
        """One settlement, in enough detail to reconstruct the flow independently.

        ``ledger_effect`` carries all five fields ``reports._ledger_and_funding``
        reads, because that function selects on the presence of the block and
        raises on a missing one; ``funding`` is the ledger's running
        ``net_funding`` after this settlement, which is the series the daily
        report differences to recover the paid/received split.
        """
        ledger = self.position.ledger.state
        perp = self.position.leg(PERP_LEG)
        return {
            "inputs": {
                "contract_hash": _prefixed(getattr(self.contract, "contract_hash", "")),
                "um_minute_digest": state.perp_digest,
                "spot_minute_digest": state.spot_digest,
                "inputs_hash": _inputs_hash(state),
            },
            "funding": {
                "settlement_id": settlement.settlement_id,
                "settlement_instant": iso_minute(
                    (settlement.instant_ns // MINUTE_NS) * MINUTE_NS
                ),
                "settlement_instant_ns": int(settlement.instant_ns),
                "leg": PERP_LEG,
                "symbol": settlement.symbol,
                "venue_symbol": str(row.get("symbol", "")),
                "side": perp.side.value,
                "rate": str(settlement.rate),
                "mark_price": str(settlement.mark_price),
                "quantity": str(perp.quantity),
                "notional": str(perp.quantity * settlement.mark_price),
                "cash_flow": str(flow),
                # Stated rather than left to the reader's sign convention. A10
                # fixes the cash-flow sign; this says which way it went in words.
                "direction": (
                    "received" if flow > ZERO else "paid" if flow < ZERO else "none"
                ),
                "funding_paid_total": str(ledger.funding_paid),
                "funding_received_total": str(ledger.funding_received),
            },
            "risk": {"state_hash": _risk_hash(self.risk), "decisions": []},
            "position_after": self._position_block(),
            "ledger_effect": {
                "fees": str(ledger.fees),
                "slippage": str(ledger.slippage),
                "funding": str(ledger.net_funding),
                "realised": str(ledger.realised),
                "equity": str(mark.equity),
            },
            "veto_or_rejection": None,
        }

    # ------------------------------------------------------------------
    # liquidation (section 6.7)
    # ------------------------------------------------------------------
    def _liquidation_check(self, minute_ms: int, state: MarketState) -> TickOutcome | None:
        """Section 6.7's per-minute touch. Returns an outcome when it fired.

        Evaluated while HEDGED or PARTIAL and not otherwise: a flat position has
        nothing to liquidate, and section 6.7 names those two states. The
        equity is the carry ledger's current one, which is why funding is booked
        first -- 6.7 exists for funding-driven equity erosion, and a check run
        against yesterday's equity would be checking the wrong number.

        Order, and it is the order a crash may interrupt at any point:

        1. evaluate against the recorded minute;
        2. halt Aegis, so no increase can be planned even if step 4 fails;
        3. write the LIQUIDATION_TOUCH record, so the log says a touch happened
           BEFORE anything claims a flatten;
        4. reduce both legs (reductions are permitted while halted, by design);
        5. persist the ledger and write the OPERATOR-free position record.

        3 before 4 is deliberate. A crash between them leaves a log that says
        "touched, and does not say flattened" over stores that still hold the
        position -- true, and recoverable. The other order would leave a log
        claiming a flatten that the stores show never happened, which is the one
        thing section 9.3 says a log may never do.
        """
        if self.position.state not in (HedgeState.HEDGED, HedgeState.PARTIAL):
            return None
        if state.mark is None:
            # A minute with no mark cannot answer 6.7 on a non-flat position, and
            # `complete` already required one, so this is unreachable through
            # `tick`. Refused rather than read as "not touched".
            return self._touch_halt(minute_ms, "liquidation_unknown: the minute has no mark")

        # Marked HERE, so the two sides of section 6.7's
        # `equity < Q * mark_high * maintenance_margin_rate` describe the same
        # minute. Reading the ledger's `last_equity` alone compared an equity
        # marked at the PREVIOUS tick against this minute's mark -- a
        # sixty-second lag on exactly the quantity section 6.7 exists to watch
        # erode.
        mark = self.position.mark_to_market(state)
        if self.position.state is HedgeState.DISPUTED:
            return self._touch_halt(
                minute_ms, f"identity_violation: {self.position.ledger.disputed}"
            )
        equity = mark.equity
        try:
            touched = self.position.liquidation_touched(state, equity=Decimal(equity))
        except Exception as exc:
            return self._touch_halt(minute_ms, f"liquidation_unknown: {exc}")
        if not touched:
            return None

        minute_ns = int(minute_ms) * _MS_TO_NS
        # Aegis first, and before any record: from this line on no increase can
        # be planned even if this process dies before the flatten, because the
        # halt is persisted by `RiskEngine.halt` itself.
        self.risk.halt("liquidation_touch")
        before = self._position_block()
        self._append(
            RecordKind.LIQUIDATION_TOUCH,
            minute_ns,
            {
                "inputs": {
                    "contract_hash": _prefixed(getattr(self.contract, "contract_hash", "")),
                    "um_minute_digest": state.perp_digest,
                    "spot_minute_digest": state.spot_digest,
                    "inputs_hash": _inputs_hash(state),
                },
                "risk": {"state_hash": _risk_hash(self.risk), "decisions": []},
                "position_after": before,
                "veto_or_rejection": {
                    "stage": "carry",
                    "label": "liquidation_touch",
                    "detail": (
                        f"equity {equity} against mark {state.mark} at maintenance rate "
                        f"{self.position.config.maintenance_margin_rate}; section 6.7's "
                        "portfolio and isolated checks"
                    ),
                },
            },
        )
        self.save_state()

        outcome = self.position.emergency_reduce(FlattenCause.RISK_HALT, state)
        # Re-marked after the flatten and before the save, so the `ledger_effect`
        # the HALT record carries describes the position the campaign stopped
        # with rather than the one it was touched at.
        self.position.mark_to_market(state)
        self._save_ledger()
        # A liquidation flatten moves the hedge and then HALTS, so unlike every
        # other position change there is no next minute to refresh the gauges at.
        # Without this `chimera_demo_hedge_state{state="HEDGED"}` stays 1 for as
        # long as the process is scraped, on a position that is flat -- an
        # operator reading the dashboard during the incident sees a live hedged
        # position that no longer exists. The operator `flatten` command guards
        # the same hazard for the same reason.
        self.telemetry.on_position(self.position)
        # `_halt` writes the HALT record, enters the state and publishes the
        # Aegis series. Reached through it rather than repeated here so the
        # runner keeps exactly one place that halts and one telemetry emission
        # for it; `risk.halt` above has already run, and `_halt` does not repeat
        # it.
        self._halt(
            f"liquidation_touch: {outcome.detail}",
            # Built here, after `_save_ledger`, so the record describes the file.
            ledger_effect=self._ledger_effect(self.position.ledger.state.last_equity),
        )
        return TickOutcome(
            minute_ms,
            self.state,
            RecordKind.LIQUIDATION_TOUCH,
            detail="liquidation_touch",
            record_hash=self.last_record_hash,
        )

    def _touch_halt(self, minute_ms: int, reason: str) -> TickOutcome:
        """Section 7.2's liquidation rule: unknown on a non-flat position is refused."""
        self._halt(reason)
        return TickOutcome(minute_ms, self.state, RecordKind.HALT, detail=reason)

    def _target_to_act_on(self, target: HedgeTarget) -> HedgeTarget:
        """What the position is actually asked for, which is not the raw target.

        Section 6.4 fixes v1's rebalance policy: "correction of imbalance from
        partial fills and rounding only. There is deliberately **no periodic
        strategy rehedge**."

        The rule re-sizes every minute, because `hedge_quantity` is a function of
        the current book and the book moves. Handing that straight to
        `HedgedPosition.plan` would re-hedge on every tick -- which is exactly
        the periodic rehedge section 6.4 forbids, and which the 48-hour
        acceptance run demonstrated is not merely inelegant: it placed two orders
        a minute until Aegis's rate limiter halted the campaign, correctly, at
        minute 201.

        So the runner acts on the ENTRY and the EXIT, and holds in between:

        * flat and the rule wants a position -> open at the rule's size;
        * holding and the rule wants nothing -> close;
        * holding and the rule still wants something -> hold what is held, and
          let `correct()` deal with any imbalance.

        The rule's own target is still recorded in full on every minute, so the
        log shows what it wanted even on the minutes the runner did not act.
        """
        held = self.position.leg(SPOT_LEG).quantity
        if target.quantity == ZERO or held == ZERO:
            return target
        return HedgeTarget(held)

    # ------------------------------------------------------------------
    # the paths that are not a decision
    # ------------------------------------------------------------------
    def _incomplete(self, minute_ms: int, state: MarketState) -> TickOutcome:
        """Section 2.2: log INCOMPLETE_STATE, run the stale check, back to READY.

        No rule evaluates. The minute is marked processed because it HAS been
        processed -- the answer for it is "the feed could not say" -- and a
        cursor that stuck here would replay the gap for ever.
        """
        minute_ns = int(minute_ms) * _MS_TO_NS
        record_hash = self._append(
            RecordKind.INCOMPLETE_STATE,
            minute_ns,
            {
                "inputs": {
                    "contract_hash": _prefixed(getattr(self.contract, "contract_hash", "")),
                    "um_minute_digest": state.perp_digest,
                    "spot_minute_digest": state.spot_digest,
                    "inputs_hash": _inputs_hash(state),
                },
                "veto_or_rejection": {
                    "stage": "feed",
                    "label": "incomplete_state",
                    "detail": ", ".join(state.missing),
                },
            },
        )
        self.cursor.mark_processed(minute_ms)
        self.save_state()
        if self.risk.state.halted:
            self._enter(RunnerState.HALT)
            self.halt_reason = self.risk.state.halt_reason
        else:
            self._enter(RunnerState.READY)
        return TickOutcome(
            minute_ms,
            self.state,
            RecordKind.INCOMPLETE_STATE,
            detail=", ".join(state.missing),
            record_hash=record_hash,
        )

    def _ledger_effect(self, equity: Any) -> dict[str, str]:
        """Section 9.1's ``ledger_effect`` block, from the carry ledger.

        One builder, because three records carry it and the daily report derives
        the whole cost and equity series from this block alone
        (`reports._ledger_and_funding`, which deliberately never reads
        `carry_ledger.json`). A record that moves the ledger and omits the block
        is a booking with no evidence: an operator flatten used to move fees,
        slippage and realised PnL and leave the log's last `ledger_effect` at its
        pre-flatten values, so the close's economics were reported nowhere -- and
        on the liquidation path, which always halts, nowhere ever.
        """
        if equity is None:
            # Never `str(None)`. The decision log is append-only and
            # `reports._ledger_and_funding` pushes this field through `_decimal`,
            # so one record carrying the text "None" makes the whole day's report
            # refuse for ever. A caller that has no equity to report has no
            # `ledger_effect` to write, and finding that out here is better than
            # finding it out in an auditor's report.
            raise RunnerError(
                "a ledger_effect needs an equity; the position has not been marked, so "
                "there is no equity to record and this record must not carry the block"
            )
        ledger = self.position.ledger.state
        return {
            "fees": str(ledger.fees),
            "slippage": str(ledger.slippage),
            "funding": str(ledger.net_funding),
            "realised": str(ledger.realised),
            "equity": str(equity),
        }

    def _halt(
        self, reason: str, *, ledger_effect: Mapping[str, str] | None = None
    ) -> RunnerState:
        """Enter HALT and write the record. Idempotent on the reason.

        The FIRST reason is kept. A halt that renamed itself as later symptoms
        arrived would lose the cause an operator needs.

        ``ledger_effect`` is passed by the ONE caller that books before it halts
        -- section 6.7's liquidation flatten -- and by no other. Putting the
        block on every halt was wrong twice. Most halts run BEFORE the tick's
        `mark_to_market` and before `_save_ledger`, so the block would have
        carried `last_equity` of ``None``, and `reports._ledger_and_funding`
        selects every record that has the block and pushes its equity through
        `_decimal`: a halt on a campaign's first minutes made the whole day's
        report refuse, and the day a campaign halted is the day whose report is
        wanted. It would also have asserted fees and slippage that no file held,
        because those halts return before the ledger is persisted -- evidence for
        a booking that never happened, which is the mirror of the defect the
        block was added to close.
        """
        if self.state is RunnerState.HALT and self.halt_reason:
            return self.state
        self._enter(RunnerState.HALT)
        self.halt_reason = reason
        if not self.risk.state.halted:
            self.risk.halt(reason)
        try:
            self._append(
                RecordKind.HALT,
                self._minute_ns(),
                {
                    "veto_or_rejection": {
                        "stage": "runner",
                        "label": "halt",
                        "detail": reason,
                    },
                    # What was held when the campaign stopped. Without it a halt
                    # that FOLLOWS a reduction -- section 6.7's liquidation
                    # flatten is the one that must -- leaves the log saying the
                    # position was touched at its full size and never saying it
                    # was closed, so the flattened position is invisible in the
                    # evidence. Reading the stores would answer it; the log is
                    # what an audit reads.
                    "position_after": self._position_block(),
                    # And what the ledger held, when the caller has just booked
                    # and persisted one. Section 6.7's liquidation flatten is the
                    # only halt that does.
                    **({"ledger_effect": dict(ledger_effect)} if ledger_effect else {}),
                },
            )
            self.save_state()
        except Exception:  # pragma: no cover - a log failure must not mask the halt
            logger.critical("HALT (%s) could not be written to the decision log", reason)
        # After the record and the state file, never before: every `_halt` path
        # returns before REPORTING, so this is the only place the Aegis series
        # can learn that the engine is halted.
        self.telemetry.on_halt(self.risk)
        logger.critical("Runner HALTED: %s", reason)
        return self.state

    # ------------------------------------------------------------------
    # operator commands
    # ------------------------------------------------------------------
    def flatten(self, note: str) -> TickOutcome:
        """Reduce to flat. Permitted while halted; that is what HALT is for."""
        note = (note or "").strip()
        if not note:
            raise RunnerError(
                "flatten requires an operator note; an unexplained flatten "
                "is an unexplained position change"
            )
        minute = self.cursor.last_minute_processed
        if minute is None:
            raise RunnerError("nothing has been processed yet, so there is nothing to flatten")
        # Prove the record is writable BEFORE the legs move, for the same reason
        # `resolve` does: on a forged log `start()`'s halt swallows the open
        # failure, so `_append` raised `DecisionLogTailError` -- not a
        # `RunnerError`, so the CLI showed a traceback -- AFTER both legs had
        # been flattened, with nothing in the log to say a flatten happened.
        self._require_recordable("flatten")
        state = self.cursor.state_for(minute, now_ns=self.clock.now_ns)
        # The book, before the orders. `install_quote` was put on the tick loop
        # only, so a `flatten` from a fresh process -- which is every flatten
        # through `tools/demo_run.py`, since the CLI constructs a runner and
        # calls this straight away -- sent both reduce-only orders against a
        # model holding no quote. Every one came back REJECTED `no_fresh_quote`,
        # the executor recorded "did not reach zero", and the position an
        # operator was trying to close in an emergency simply stood.
        self.position.install_quote(state)
        outcome = self.position.emergency_reduce(FlattenCause.RISK_HALT, state)
        # Marked BEFORE the ledger is persisted, so the equity that reaches the
        # record is the flattened position's and is the one on disk. Marking
        # after the save would record a number no file holds.
        mark = self.position.mark_to_market(state)
        self._save_ledger()
        record_hash = self._append(
            RecordKind.OPERATOR,
            int(minute) * _MS_TO_NS,
            {
                "operator": {"command": "flatten", "note": note},
                "position_after": self._position_block(),
                "ledger_effect": self._ledger_effect(mark.equity),
            },
        )
        self.save_state()
        # A flatten moves the hedge without a tick, so the position gauges would
        # otherwise keep showing the pre-flatten size until the next minute.
        self.telemetry.on_position(self.position)
        return TickOutcome(
            minute,
            self.state,
            RecordKind.OPERATOR,
            detail=outcome.detail,
            record_hash=record_hash,
        )

    def resume(self, note: str) -> TickOutcome:
        """Leave HALT by an explicit operator action, and only by one.

        Section 8.1's HALT row also offers "automatic after a transient
        stale-feed cause clears". Section 7.2 makes a stale feed a VETO rather
        than a halt, so no stale feed can put the runner in HALT in the first
        place and that clause names an unreachable path. Recorded in the PR;
        implemented as operator-only, which is also what `RiskEngine.resume`
        already requires ("Only ever called by an explicit operator action").
        """
        note = (note or "").strip()
        if not note:
            raise RunnerError("resume requires an operator note stating what was checked")
        if self.state is not RunnerState.HALT:
            raise RunnerError("the runner is not halted; there is nothing to resume from")
        self.risk.resume()
        record_hash = self._append(
            RecordKind.RESUME,
            self._minute_ns(),
            {"operator": {"command": "resume", "note": note, "cleared": self.halt_reason}},
        )
        self.halt_reason = None
        self._enter(RunnerState.RECOVER)
        outcome = self.position.reconstruct()
        if outcome.state is HedgeState.DISPUTED:
            return TickOutcome(
                self.cursor.last_minute_processed or 0,
                self._halt(f"dispute: {outcome.detail}"),
                RecordKind.HALT,
            )
        self._enter(RunnerState.READY)
        self.save_state()
        return TickOutcome(
            self.cursor.last_minute_processed or 0,
            self.state,
            RecordKind.RESUME,
            record_hash=record_hash,
        )

    def resolve(self, symbol: str, note: str) -> TickOutcome:
        """Clear one leg's reconciliation dispute, with a mandatory note.

        The note requirement is not this method's invention:
        `FuturesExecutor.resolve_reconciliation` already refuses an empty one.
        Checked here too so the CLI fails before touching a store, and so the
        OPERATOR record and the executor's own record carry the same text.

        **What is adopted, and how.** ``resolve_reconciliation(symbol, adopted,
        note)`` takes the position the operator has decided is the true one. The
        two candidates are the local view and the venue's, and this build adopts
        the LOCAL one. That is not a preference: the venue here is
        ``DryRunFuturesVenue``, which reports from state this process itself
        wrote, so "the venue's view" is not independent evidence an operator
        could have checked anything against. Adopting local leaves the position
        exactly as the stores record it and clears only the dispute flag and the
        orders it froze; the operator's note is the evidence, which is what
        section 11.5 says it is. A build with a real venue must revisit this line
        rather than this docstring.

        Before PR-10R this method called ``resolve_reconciliation(symbol, note)``
        -- two of the three required positionals. Every real invocation raised
        ``TypeError`` before touching a store, so the one operator command that
        can clear a dispute could not be run at all. The single test on the path
        passed a symbol that is not a leg and returned one line earlier, which is
        why the suite never saw it.
        """
        note = (note or "").strip()
        if not note:
            raise RunnerError("resolve requires an operator note stating what was checked")
        legs = {
            self.position.config.spot_symbol: (SPOT, self.position.spot),
            self.position.config.perp_symbol: (PERP, self.position.perp),
        }
        if symbol not in legs:
            raise RunnerError(
                f"{symbol!r} is not one of this position's legs ({sorted(legs)})"
            )
        leg_name, executor = legs[symbol]
        # Nothing above this line has changed anything; nothing below may change
        # anything until the record it will be written into is reachable. The
        # clock is the one part of that which can fail: `tools/demo_run.py`
        # constructs the runner without one and does not `start()` for this
        # command, so `now_ns` raised -- AFTER the store had been saved and
        # Aegis's dispute cleared, and out through an `except RunnerError` that
        # does not catch it. The safety-critical dispute was cleared with no
        # OPERATOR record, which is the one outcome section 8.3 forbids. Reading
        # the clock first turns that into a refusal that changes nothing.
        now_ns = self._require_recordable("resolve")
        adopted = executor.position(symbol)
        executor.resolve_reconciliation(symbol, adopted, note)
        # Aegis keeps its own copy of the dispute (section 7.2's reconciliation
        # row: "cleared only by operator note"), and it is what vetoes increases
        # on the symbol. Clearing the store without clearing this would leave the
        # campaign permanently unable to increase over a dispute nothing records.
        self.risk.note_reconciliation(symbol, None)
        # Only the RECONCILIATION dispute, and only when the ledger is the real
        # one. This used to clear whatever the carry ledger happened to be
        # disputing, which is not the same question: `ledger_unreadable` says the
        # file could not be parsed, `ledger_capital_mismatch` that it disagrees
        # with the configuration, `funding_booking_torn` that a settlement is
        # booked on one side only. Clearing those here cleared a flag and fixed
        # nothing -- and on an unreadable ledger it also SAVED, overwriting the
        # only record of the campaign's cash with a placeholder that says it
        # never traded. `CarryLedger.save` now refuses that outright; this stops
        # asking.
        dispute = self.position.ledger.disputed
        if dispute is not None and dispute.startswith(f"{leg_name}_reconciliation_mismatch"):
            self.position.ledger.resolve(note, now_ns=now_ns)
            self.position.ledger.save()
        record_hash = self._append(
            RecordKind.OPERATOR,
            self._minute_ns(),
            {
                "operator": {
                    "command": "resolve",
                    "symbol": symbol,
                    "note": note,
                    "adopted_side": adopted.side.value,
                    "adopted_qty": str(adopted.quantity),
                },
                "position_after": self._position_block(),
            },
        )
        self.save_state()
        return TickOutcome(
            self.cursor.last_minute_processed or 0,
            self.state,
            RecordKind.OPERATOR,
            record_hash=record_hash,
        )

    def shutdown(self, note: str = "") -> TickOutcome:
        """Finish, persist, write SHUTDOWN, and stop. Never mid-record."""
        self._enter(RunnerState.SHUTDOWN)
        self._save_ledger()
        record_hash = self._append(
            RecordKind.SHUTDOWN, self._minute_ns(), {"operator": {"note": note}}
        )
        self.save_state()
        if self._log is not None:
            self._log.close()
            self._log = None
        # Last, and after the log is closed: `chimera_demo_up` says the process
        # stopped on purpose, which is what tells a clean shutdown apart from the
        # scrape failure of a process that died mid-record.
        self.telemetry.on_shutdown()
        return TickOutcome(
            self.cursor.last_minute_processed or 0,
            self.state,
            RecordKind.SHUTDOWN,
            record_hash=record_hash,
        )

    # ------------------------------------------------------------------
    # driving
    # ------------------------------------------------------------------
    def run_minutes(self, minutes: Iterable[int]) -> list[TickOutcome]:
        """Process the given minutes in order, stopping at HALT."""
        outcomes = []
        for minute in minutes:
            if self.state is RunnerState.HALT:
                break
            outcomes.append(self.tick(minute))
        return outcomes

    def replay(self, start_ms: int, end_ms: int) -> list[TickOutcome]:
        """Section 12.3's `replay(from, to)`. Inclusive of both ends."""
        minutes = list(range(int(start_ms), int(end_ms) + 1, 60_000))
        return self.run_minutes(minutes)

    def catch_up(self, *, now_ms: int | None = None) -> list[TickOutcome]:
        """Process every pending minute in order; decide only the recent ones.

        Section 2.2 line 120, in full: "minutes between the persisted cursor and
        now are processed in order with `catch_up=True` in the log, and no
        position change is executed for catch-up minutes older than the
        configured `max_catchup_minutes` (default 3): older minutes are logged as
        `SKIPPED_STALE`."

        Two clauses, and before PR-10R only half of the first was implemented:
        the runner ticked the OLDEST `max_catchup_minutes` pending minutes and
        abandoned the rest with no record. That is the wrong end of the queue --
        a restart after a two-hour outage decided the two-hour-old minutes at the
        two-hour-old book and left the current ones unread -- and it left the
        campaign's log with a hole where the abandoned minutes should be, because
        nothing in the log said they had been passed over.

        What runs now: every minute from the cursor up to `now_ms` is accounted
        for, in order. The last `max_catchup_minutes` of them are ticked
        normally. Every older one advances the cursor and writes one
        `SKIPPED_STALE` record naming its age -- no rule evaluates, no Aegis
        check runs and no position changes, which is exactly "no position change
        is executed".

        `now_ms` is the instant catching up is happening at. Without one the
        newest minute the feed holds is used, because that is the newest minute
        that could be decided; a caller that means something else says so.

        Section 9.1's schema has no `catch_up` key, so it is written at the TOP
        LEVEL of the record -- the log's writer permits extra keys, and section
        10's parity comparison lists the fields that must match without naming
        this one, so a flag both live and replay produce identically cannot break
        parity. Recorded in the PR.
        """
        limit = int(self.config.runner_setting("max_catchup_minutes"))
        # Read once. The window is what it was when catching up began; letting it
        # move as minutes are processed would make the answer depend on how long
        # the catch-up itself took, which is a wall-clock dependency by another
        # name and would not replay.
        newest = self._newest_minute_ms(now_ms=now_ms)
        if newest is None:
            # No minute exists to catch up TO. The cursor's `next_minute_ms`
            # answers "cursor + one minute" for ever whether or not a file holds
            # it, so without this the loop below walks forward without end
            # writing INCOMPLETE_STATE records for minutes no recorder ever
            # wrote -- fabricated evidence, and an unbounded log.
            return []
        outcomes: list[TickOutcome] = []
        while True:
            if self.state is RunnerState.HALT:
                break
            # Bounded by `newest`, never by `now_ms` alone: with no `now_ms` the
            # cursor's own `next_minute_ms` hands back cursor + one minute for
            # ever, whether or not a file holds it, so the loop's end has to be
            # the newest minute that EXISTS.
            minute = self.cursor.next_minute_ms(now_ms=newest)
            if minute is None:
                break
            age = 0 if newest is None else (int(newest) - int(minute)) // 60_000
            if age >= limit:
                outcomes.append(self._skipped_stale(minute, age=age, limit=limit))
            else:
                outcomes.append(self.tick(minute))
        return outcomes

    def _newest_minute_ms(self, *, now_ms: int | None = None) -> int | None:
        """The newest minute catching up could decide.

        `now_ms` when the caller named one, and otherwise the newest minute the
        feed actually holds. Read from the files rather than from a clock: the
        runner reads no clock, and a replay of the same files has to find the
        same answer.
        """
        if now_ms is not None:
            return int(now_ms)
        return self.cursor.latest_minute_ms()

    def _skipped_stale(self, minute_ms: int, *, age: int, limit: int) -> TickOutcome:
        """One catch-up minute too old to act on. Recorded, never silently dropped.

        The cursor advances because the minute HAS been dealt with: the answer
        for it is "too old to decide", which is a fact about the campaign and not
        a gap in it. Nothing else about the position, the ledger or Aegis moves,
        so a SKIPPED_STALE minute cannot change what a later minute decides.
        """
        minute_ns = int(minute_ms) * _MS_TO_NS
        self.clock.observe(minute_ns + MINUTE_NS)
        record_hash = self._append(
            RecordKind.SKIPPED_STALE,
            minute_ns,
            {
                "catch_up": True,
                "stale": {
                    "age_minutes": int(age),
                    "max_catchup_minutes": int(limit),
                },
                "veto_or_rejection": {
                    "stage": "feed",
                    "label": "skipped_stale",
                    "detail": (
                        f"the minute is {age} minute(s) behind the newest available one "
                        f"and max_catchup_minutes is {limit}; no rule evaluated and no "
                        "position changed"
                    ),
                },
            },
        )
        self.cursor.mark_processed(minute_ms)
        self.save_state()
        return TickOutcome(
            minute_ms,
            self.state,
            RecordKind.SKIPPED_STALE,
            detail=f"stale by {age} minute(s)",
            record_hash=record_hash,
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _minute_ns(self) -> int:
        last = self.cursor.last_minute_processed
        if last is not None:
            return int(last) * _MS_TO_NS
        first = self.cursor.next_minute_ms()
        if first is not None:
            return int(first) * _MS_TO_NS
        # A record still needs a minute. Use the clock's own minute floor rather
        # than inventing one: iso_minute refuses a non-boundary instant.
        return (self.clock.now_ns // MINUTE_NS) * MINUTE_NS

    def _portfolio(self, state: MarketState) -> dict[str, Any]:
        """The read-only view a rule is given. Never the position object itself."""
        ledger = self.position.ledger.state
        equity = ledger.last_equity if ledger.last_equity is not None else self.capital
        return {
            "equity": str(equity),
            "hedge_state": self.position.state.value,
            "spot_qty": str(self.position.leg("spot").quantity),
            "perp_qty": str(self.position.leg("perp").quantity),
            "imbalance": str(self.position.imbalance()),
        }

    def _position_block(self) -> dict[str, str]:
        return {
            "hedge_state": self.position.state.value,
            "spot_qty": str(self.position.leg("spot").quantity),
            "perp_qty": str(self.position.leg("perp").quantity),
            "imbalance": str(self.position.imbalance()),
        }

    def _decision_payload(
        self,
        state: MarketState,
        decisions: Sequence[RuleDecision],
        intents: Sequence[Any],
        veto: Mapping[str, Any] | None,
        outcome: Any,
        mark: Any,
        orders_before: Mapping[str, frozenset[str]],
    ) -> dict[str, Any]:
        ledger = self.position.ledger.state
        actionable = [d for d in decisions if d.is_actionable]
        primary = actionable[0] if actionable else (decisions[0] if decisions else None)
        payload: dict[str, Any] = {
            "inputs": {
                "contract_hash": _prefixed(getattr(self.contract, "contract_hash", "")),
                "um_minute_digest": state.perp_digest,
                "spot_minute_digest": state.spot_digest,
                "book_um": _plain(state.book_perp),
                "book_spot": _plain(state.book_spot),
                "mark": str(state.mark) if state.mark is not None else None,
                "funding_last": (
                    str(state.funding_rate_last)
                    if state.funding_rate_last is not None
                    else None
                ),
                "inputs_hash": _inputs_hash(state),
            },
            "requested_action": [
                {
                    "leg": intent.leg,
                    "purpose": getattr(intent, "purpose", ""),
                    "side": getattr(intent.side, "value", str(intent.side)),
                    "qty": str(intent.quantity),
                }
                for intent in intents
            ],
            "risk": {
                "state_hash": _risk_hash(self.risk),
                "decisions": [] if veto is None else [dict(veto)],
            },
            "execution": _execution_block(self.position, orders_before),
            "position_after": self._position_block(),
            "ledger_effect": {
                "fees": str(ledger.fees),
                "slippage": str(ledger.slippage),
                "funding": str(ledger.net_funding),
                "realised": str(ledger.realised),
                "equity": str(mark.equity),
            },
            "veto_or_rejection": dict(veto) if veto else None,
            "shadow": [
                {"rule_id": d.rule_id, "reason": d.reason, **d.to_record()["signal"]}
                for d in decisions
                if not d.is_actionable
            ],
        }
        if primary is not None:
            payload.update(primary.to_record())
        return payload


# ----------------------------------------------------------------------
# small pure helpers, kept out of the class so tests can reach them
# ----------------------------------------------------------------------
#: The settlement instant's field name, taken from the feed rather than repeated.
_SETTLEMENT_FIELD = SETTLEMENT_INSTANT_FIELD


def _venue_symbol(contract: Any) -> str:
    """The perpetual market's symbol AS THE VENUE SPELLS IT, from the contract.

    ``BTCUSDT``, not ``BTC/USDT:USDT``. It is read off the contract rather than
    written down here so a campaign on another instrument cannot silently book
    this one's settlements; see :func:`chimera.demo.feed.settlement_from_row` for
    why the two spellings may never be confused.
    """
    try:
        return str(contract.market(PERP_MARKET).symbol)
    except Exception as exc:  # pragma: no cover - a contract without the market
        raise RunnerError(
            f"the campaign's contract does not name a {PERP_MARKET!r} market, so a "
            f"funding settlement cannot be attributed to an instrument: {exc}"
        ) from exc


def _prefixed(value: str) -> str:
    """Render a bare 64-hex digest as section 9.2's ``sha256:<hex>``.

    `RecorderContract.contract_hash` is stored bare; section 9.1 writes it
    prefixed, and `chimera.demo.decision_log` holds every field whose name ends
    in `_hash` to the prefixed form. The prefix is part of the value rather than
    decoration -- a prefixed and an unprefixed digest compare unequal -- so it
    is applied here, at the boundary, rather than by changing what the recorder
    stores.
    """
    if not value:
        return ""
    return value if value.startswith("sha256:") else f"sha256:{value}"


def _plain(mapping: Mapping[str, Any]) -> dict[str, Any]:
    """The feed's coercion, not a second one. See `chimera.demo.feed.plain_json`."""
    return plain_json(dict(mapping))


def _inputs_hash(state: MarketState) -> str:
    from chimera.demo.rules import canonical_hash

    return canonical_hash(state.canonical())


#: Risk-state fields that are WALL CLOCK or HOST DATE rather than decision
#: semantics, and are therefore outside `risk.state_hash`.
#:
#: `order_times` is the rate limiter's record of when orders were placed and
#: `cooldown_until` is a deadline computed from one; `day` is the date the host
#: happened to be running on. All three move between a live run and a replay of
#: the same files, and none of them is a fact about the decision -- so hashing
#: them would make section 10's byte comparison fail for a reason that has
#: nothing to do with whether the two runs decided alike.
#:
#: Everything that governs permission and IS reproducible stays in: `halted`,
#: `halt_reason`, `kill_switch`, the equity series, the streaks, and the
#: reconciliation disputes.
_RISK_HASH_EXCLUDED = frozenset({"order_times", "cooldown_until", "day"})


def _risk_hash(risk: RiskEngine) -> str:
    from chimera.demo.rules import canonical_hash

    snapshot = risk.snapshot()
    return canonical_hash(
        {
            key: value
            for key, value in snapshot.items()
            if isinstance(key, str) and key not in _RISK_HASH_EXCLUDED
        }
    )


def _config_hash(config: DemoConfig) -> str:
    from chimera.demo.config import config_hash

    return config_hash(config)


def _execution_block(
    position: Any, before: Mapping[str, frozenset[str]]
) -> list[dict[str, Any]]:
    """Section 9.1's ``execution`` list: the orders THIS minute produced.

    `HedgeOutcome` carries which legs filled, not the orders that filled them,
    so the order records come from each executor's own store -- the same store
    the executor persisted before this ran, which is what makes the record and
    the store agree by construction rather than by copying.

    ``before`` is the set of order ids each leg held before the minute, so the
    block names only what this minute added. Without it a record would restate
    the whole order history every tick, and the log would grow quadratically.
    """
    block: list[dict[str, Any]] = []
    for leg, executor in (("spot", position.spot), ("perp", position.perp)):
        orders = getattr(getattr(executor, "store", None), "state", None)
        if orders is None:
            continue
        for order_id, record in sorted(orders.orders.items()):
            if order_id in before.get(leg, frozenset()):
                continue
            block.append(
                {
                    "leg": leg,
                    "order_id": order_id,
                    "state": getattr(getattr(record, "state", None), "value", ""),
                    "filled_qty": str(getattr(record, "filled_quantity", "")),
                    "average_price": str(getattr(record, "average_price", "")),
                    "fees": str(getattr(record, "fees", "")),
                }
            )
    return block


def _order_ids(position: Any) -> dict[str, frozenset[str]]:
    """Each leg's order ids right now, so the next record can name only new ones."""
    snapshot: dict[str, frozenset[str]] = {}
    for leg, executor in (("spot", position.spot), ("perp", position.perp)):
        state = getattr(getattr(executor, "store", None), "state", None)
        snapshot[leg] = frozenset(state.orders) if state is not None else frozenset()
    return snapshot
