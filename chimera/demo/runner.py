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
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from chimera.carry.hedge import HedgedPosition, HedgeState
from chimera.demo.clock import RunnerClock
from chimera.demo.config import DemoConfig
from chimera.demo.decision_log import DecisionLog, RecordKind, iso_minute
from chimera.demo.feed import FeedCursor, MarketState, plain_json
from chimera.demo.rules import HedgeTarget, RuleDecision, RuleError, RuleRegistry
from chimera.demo.telemetry import RunnerTelemetry
from chimera.futures.executor import FlattenCause, ReconciliationRequired
from chimera.recorder.sink import write_json_atomic
from chimera.risk import RiskEngine

logger = logging.getLogger(__name__)

__all__ = ["DemoRunner", "RunnerState", "RunnerError", "TickOutcome", "RUNNER_STATE_SCHEMA"]

RUNNER_STATE_SCHEMA = "chimera.demo-runner-state/1"
MINUTE_NS = 60_000_000_000
_MS_TO_NS = 1_000_000
ZERO = Decimal("0")
SPOT_LEG = "spot"


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
        self.last_record_hash: str = persisted.get("last_record_hash", "")
        self._log: DecisionLog | None = None
        self._minutes_since_reconcile = 0
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
        if payload.get("schema") != RUNNER_STATE_SCHEMA:
            raise RunnerError(
                f"{path} declares schema {payload.get('schema')!r}, not "
                f"{RUNNER_STATE_SCHEMA!r}"
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
        tail = self.log.last_record_hash
        if self.last_record_hash and tail != self.last_record_hash:
            return (
                f"log_behind_state: the decision log's tail is {tail} and the runner state "
                f"names {self.last_record_hash}. One of them is from before a crash"
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
    # the tick loop
    # ------------------------------------------------------------------
    def tick(self, minute_ms: int) -> TickOutcome:
        """One minute, through section 8.1's tick loop."""
        minute_ns = int(minute_ms) * _MS_TO_NS
        self.clock.observe(minute_ns + MINUTE_NS)

        self._enter(RunnerState.DATA_READY)
        state = self.cursor.state_for(minute_ms, now_ns=self.clock.now_ns)
        self.risk.note_feed(minute_ns + MINUTE_NS, self.clock.now_ns)
        # Counted here, before anything is decided about the minute, because the
        # quantity is "minutes attempted": a minute that halts inside a rule is
        # still one the runner took on, and a counter that skipped it would read
        # healthiest exactly when the campaign was failing. Nothing in this call
        # is decision-derived, and nothing reads it back.
        self.telemetry.on_minute(minute_ns=minute_ns, missing=state.missing)

        if not state.complete:
            return self._incomplete(minute_ms, state)

        if self.risk.check_kill_switch():
            self._halt("kill_switch")
            return TickOutcome(minute_ms, self.state, RecordKind.HALT, detail="kill_switch")

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
        self._minutes_since_reconcile += 1
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

    def _halt(self, reason: str) -> RunnerState:
        """Enter HALT and write the record. Idempotent on the reason.

        The FIRST reason is kept. A halt that renamed itself as later symptoms
        arrived would lose the cause an operator needs.
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
                {"veto_or_rejection": {"stage": "runner", "label": "halt", "detail": reason}},
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
        state = self.cursor.state_for(minute, now_ns=self.clock.now_ns)
        outcome = self.position.emergency_reduce(FlattenCause.RISK_HALT, state)
        self.position.ledger.save()
        record_hash = self._append(
            RecordKind.OPERATOR,
            int(minute) * _MS_TO_NS,
            {
                "operator": {"command": "flatten", "note": note},
                "position_after": self._position_block(),
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
        """
        note = (note or "").strip()
        if not note:
            raise RunnerError("resolve requires an operator note stating what was checked")
        legs = {
            self.position.config.spot_symbol: self.position.spot,
            self.position.config.perp_symbol: self.position.perp,
        }
        executor = legs.get(symbol)
        if executor is None:
            raise RunnerError(
                f"{symbol!r} is not one of this position's legs ({sorted(legs)})"
            )
        executor.resolve_reconciliation(symbol, note)
        record_hash = self._append(
            RecordKind.OPERATOR,
            self._minute_ns(),
            {"operator": {"command": "resolve", "symbol": symbol, "note": note}},
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
        self.position.ledger.save()
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
        """Process pending minutes, at most `max_catchup_minutes` of them.

        Section 2.2 line 119: catch-up minutes are "processed in order with
        `catch_up=True` in the log". Section 9.1's schema has no `catch_up` key,
        so it is written at the TOP LEVEL of the record -- the log's writer
        permits extra keys, and section 10's parity comparison lists the fields
        that must match without naming this one, so a flag both live and replay
        produce identically cannot break parity. Recorded in the PR.
        """
        limit = int(self.config.runner_setting("max_catchup_minutes"))
        outcomes: list[TickOutcome] = []
        for _ in range(limit):
            if self.state is RunnerState.HALT:
                break
            minute = self.cursor.next_minute_ms(now_ms=now_ms)
            if minute is None:
                break
            outcomes.append(self.tick(minute))
        return outcomes

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
