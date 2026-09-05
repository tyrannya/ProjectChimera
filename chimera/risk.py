"""Central risk engine.

Every entry in the system passes through :meth:`RiskEngine.evaluate_entry`.
The engine owns account state (equity, drawdown, exposure, open positions),
trade sizing, and the kill switch.

Design notes
------------

*The kill switch is local state, not a network call.* The previous
implementation fired ``requests.post("http://localhost:8080/api/v1/stop")`` and
treated that as the guarantee. A guard that depends on an HTTP round trip
succeeding is not a guard. Here, :attr:`halted` is checked synchronously at the
top of ``evaluate_entry``, so a halted engine cannot approve an entry even if
every network path in the process is down. The halt is additionally persisted
to disk so a process restart does not silently clear it.

*The whole decision-relevant state is persisted, not just the halt.* A file that
recorded only ``halted`` let a restart reset the peak equity, the day's starting
equity, the loss streak, the cooldown deadline and the order-rate window to
their defaults. Every one of those is an input to a guard, so a restart between
the peak and the drawdown that breaches it measured the fall from the wrong
peak and allowed the trade the limit exists to stop. :class:`RiskState` is now
written in full after every mutation, under the schema string
:data:`RISK_STATE_SCHEMA`, and the drawdown deliberately stays *derived* through
:meth:`RiskEngine.current_drawdown` rather than being stored beside the two
numbers it is computed from, which could then disagree with them.

*The state file is written atomically and read fail-closed.* The write goes to a
temporary file that is flushed and ``fsync``-ed before ``os.replace`` swaps it
in — the pattern :meth:`chimera.futures.store.FuturesStore.save` uses — so a
crash in the middle of a write leaves the previous file intact rather than a
truncated one. A file that cannot be read, or that carries a schema this build
does not know, starts the engine halted: the alternative is to treat "I cannot
tell what this account was doing" as "this account was doing nothing", which is
the one reading that can lose money. That file is then *preserved* rather than
written over — every mutation persists, so without the refusal the next tick
would replace the only record of what the account was doing with an
all-defaults document. :meth:`RiskEngine.adopt_after_unreadable` is the
deliberate way through, as it is for
:class:`chimera.futures.store.FuturesStore`.

*Sizing is risk-based, not wallet-fraction-based.* ``risk_per_trade_pct`` is the
fraction of equity lost **if the stop is hit**, which is what "1% risk" means.
Converting that to a position size requires dividing by the stop distance; using
1% of the wallet as the position size instead would mean the actual risk is
0.01 * stop_distance, i.e. 20x smaller than intended for a 5% stop.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:  # pragma: no cover - for type checkers only; see _position_sign
    from chimera.futures.domain import PositionSide

logger = logging.getLogger(__name__)

#: Identifies the on-disk shape of :class:`RiskState`. A file carrying anything
#: else is refused rather than best-effort parsed; see ``RiskEngine._load_state``.
RISK_STATE_SCHEMA = "chimera.risk-state/1"

#: The only keys the pre-schema state file ever contained. A file without a
#: schema is accepted as that legacy halt record and as nothing else, so a
#: truncated or foreign document cannot be mistaken for one.
_LEGACY_KEYS = frozenset({"halted", "halt_reason", "updated_at"})

#: Width of the order-rate window, in seconds. Orders older than this are not
#: evidence about the current rate, so they are pruned before every write; the
#: file then holds the live window rather than a log that grows forever.
_ORDER_WINDOW_S = 60.0

#: Where a deployment is expected to put the operator's kill switch. A file, not
#: a config key the process would have to be restarted to re-read, so a human
#: with a shell can stop new exposure in one command.
#:
#: It is a constant an entry point passes, **not** a default
#: :class:`RiskEngine` applies on its own. The path is relative, so a default
#: would resolve against whatever directory the process happened to start in:
#: every engine in the repository — the Freqtrade strategy, the smoke and paper
#: tools, and the generator behind the frozen ``artifacts/futures_dry_run_v1``
#: — would then read an untracked file that no committed input names, and an
#: engaged switch on one host would change the frozen protocol's output and fail
#: much of the test suite. Evidence must be a function of committed inputs; a
#: guard whose reach depends on the current directory is not a guard anyway.
DEFAULT_KILL_SWITCH_PATH = Path("user_data/KILL_SWITCH")


class RiskViolation(Exception):
    """Raised by the guards that are meant to abort an operation outright."""


@dataclass(frozen=True)
class RiskLimits:
    """Configured limits. All fractions are of current equity unless noted."""

    # --- account limits -------------------------------------------------
    max_drawdown_pct: float = 0.15
    max_daily_loss_pct: float = 0.05
    max_open_positions: int = 3
    max_total_exposure_pct: float = 1.0
    max_exposure_per_asset_pct: float = 0.35

    # --- per-trade limits -----------------------------------------------
    risk_per_trade_pct: float = 0.01
    max_position_pct: float = 0.25
    max_leverage: float = 3.0
    min_stop_distance_pct: float = 0.005
    max_stop_distance_pct: float = 0.15

    # --- operational guards ---------------------------------------------
    max_orders_per_minute: int = 10
    loss_streak_limit: int = 3
    cooldown_seconds: float = 3600.0
    #: How late the newest candle may be *past its close* before entries are
    #: blocked. Not the candle's age: an OHLCV timestamp is the candle's OPEN
    #: time, so a just-closed 1h candle is already 3600s "old" while being
    #: perfectly fresh. Freqtrade measures the same way in
    #: ``IStrategy.ignore_expired_candle``. Because this is a delay past close,
    #: one value is meaningful across every timeframe.
    max_data_delay_s: float = 300.0
    max_inference_staleness_s: float = 300.0

    # --- futures guards --------------------------------------------------
    #: Sign-blind ceiling on the funding rate, applied when the caller names no
    #: position side. Without a side there is no way to know whether the rate is
    #: a cost or a rebate, so its magnitude is all that can be judged.
    max_funding_rate: float = 0.0005
    #: Ceiling on the funding a position would actually *pay* per settlement,
    #: applied when the caller does name a side. Kept as a second field rather
    #: than as a rename of ``max_funding_rate`` because the two answer different
    #: questions and are configured independently: ``max_funding_rate`` is an
    #: existing configurable key whose meaning is the sign-blind ceiling, and
    #: quietly giving that key the side-aware meaning would change what any
    #: config that sets it enforces without anybody editing that config.
    max_funding_cost_rate: float = 0.0005
    #: How many consecutive settlements a position may pay before Aegis stops
    #: approving increases. One adverse settlement is weather; a run of them is
    #: the carry having inverted, and adding to the position through that is the
    #: mistake this counter exists to make impossible.
    funding_adverse_streak_limit: int = 3
    min_liquidation_distance_pct: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RiskLimits":
        fields = set(cls.__dataclass_fields__)
        return cls(**{k: type(getattr(cls(), k))(v) for k, v in data.items() if k in fields})


@dataclass
class RiskDecision:
    """Result of an entry evaluation."""

    allowed: bool
    reason: str
    #: Stake in quote currency. Zero when ``allowed`` is False.
    stake: float = 0.0

    def __bool__(self) -> bool:
        return self.allowed


@dataclass
class RiskState:
    """The account state the guards read, in the shape it is persisted in.

    Everything here is an input to a decision, which is why all of it survives a
    restart. The drawdown is the deliberate exception: it is a function of
    ``peak_equity`` and ``equity`` and is computed on demand by
    :meth:`RiskEngine.current_drawdown`, because a stored copy is a second
    authority that can disagree with the two numbers it came from — and that
    disagreement would only ever be discovered by a limit failing to fire.
    """

    equity: float = 0.0
    peak_equity: float = 0.0
    day_start_equity: float = 0.0
    #: UTC date, ISO format: the day the daily-loss budget is measured against.
    day: str = ""
    #: ``equity - day_start_equity``. Persisted rather than left implicit
    #: because the daily report is written from the file, not from a live engine.
    daily_pnl: float = 0.0
    open_positions: dict[str, float] = field(default_factory=dict)
    #: Approval instants inside the live rate window, oldest first.
    order_times: list[float] = field(default_factory=list)
    consecutive_losses: int = 0
    cooldown_until: float = 0.0
    halted: bool = False
    halt_reason: str = ""
    #: Whether the kill-switch file was present, or could not be examined, the
    #: last time :meth:`RiskEngine.check_kill_switch` looked.
    kill_switch: bool = False
    #: When the feed was first observed stale, in epoch seconds, or ``None``
    #: while it is fresh. Kept rather than recomputed so a restart cannot
    #: present a stale feed as an unknown one.
    stale_feed_since: float | None = None
    #: Symbol -> operator-readable description of the dispute. A symbol in this
    #: map is one whose local and venue positions disagree, so its size is not
    #: known and may not be added to.
    reconciliation_disputed: dict[str, str] = field(default_factory=dict)
    #: Consecutive funding settlements paid by the position.
    funding_adverse_streak: int = 0
    funding_halt: bool = False

    def to_dict(self, *, updated_at: str) -> dict[str, Any]:
        """The persisted document: the schema, the write time, and the state."""
        return {
            "schema": RISK_STATE_SCHEMA,
            "updated_at": updated_at,
            "equity": float(self.equity),
            "peak_equity": float(self.peak_equity),
            "day_start_equity": float(self.day_start_equity),
            "day": str(self.day),
            "daily_pnl": float(self.daily_pnl),
            "open_positions": {str(k): float(v) for k, v in self.open_positions.items()},
            "order_times": [float(t) for t in self.order_times],
            "consecutive_losses": int(self.consecutive_losses),
            "cooldown_until": float(self.cooldown_until),
            "halted": bool(self.halted),
            "halt_reason": str(self.halt_reason),
            "kill_switch": bool(self.kill_switch),
            "stale_feed_since": (
                None if self.stale_feed_since is None else float(self.stale_feed_since)
            ),
            "reconciliation_disputed": {
                str(k): str(v) for k, v in self.reconciliation_disputed.items()
            },
            "funding_adverse_streak": int(self.funding_adverse_streak),
            "funding_halt": bool(self.funding_halt),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RiskState":
        """Rebuild the state, raising rather than substituting a default.

        A missing or mistyped field means the file does not say what this
        account was doing, and the caller's answer to that is a halt. Filling
        the gap here with a zero would turn "unknown" into a confident and wrong
        "flat, no drawdown, no cooldown". Every field is required, including
        ``stale_feed_since``, whose ``None`` is written as JSON ``null`` and
        round-trips: an absent key is therefore never "the feed was fresh", it is
        a document this build did not write.
        """
        missing = set(cls.__dataclass_fields__) - set(data)
        if missing:
            raise ValueError(f"missing field(s): {', '.join(sorted(missing))}")
        positions = data["open_positions"]
        disputed = data["reconciliation_disputed"]
        times = data["order_times"]
        if not isinstance(positions, Mapping):
            raise ValueError("open_positions is not a mapping")
        if not isinstance(disputed, Mapping):
            raise ValueError("reconciliation_disputed is not a mapping")
        if isinstance(times, (str, bytes)) or not isinstance(times, (list, tuple)):
            raise ValueError("order_times is not a list")
        stale = data["stale_feed_since"]
        return cls(
            equity=float(data["equity"]),
            peak_equity=float(data["peak_equity"]),
            day_start_equity=float(data["day_start_equity"]),
            day=str(data["day"]),
            daily_pnl=float(data["daily_pnl"]),
            open_positions={str(k): float(v) for k, v in positions.items()},
            order_times=[float(t) for t in times],
            consecutive_losses=int(data["consecutive_losses"]),
            cooldown_until=float(data["cooldown_until"]),
            halted=bool(data["halted"]),
            halt_reason=str(data["halt_reason"]),
            kill_switch=bool(data["kill_switch"]),
            stale_feed_since=None if stale is None else float(stale),
            reconciliation_disputed={str(k): str(v) for k, v in disputed.items()},
            funding_adverse_streak=int(data["funding_adverse_streak"]),
            funding_halt=bool(data["funding_halt"]),
        )


def _position_sign(position_side: "PositionSide | str | None") -> int | None:
    """+1 for a long, -1 for a short, ``None`` when the side is neither.

    Duck-typed on purpose. :class:`chimera.futures.domain.PositionSide` lives in
    a package whose ``__init__`` imports the executor, which imports this module,
    so a real import here would be a cycle: ``import chimera.risk`` would run
    ``chimera.futures.__init__``, which would re-enter this half-built module and
    fail on a name that does not exist yet. The enum is a ``str`` enum whose
    values are the wire format, so reading ``.value`` when it is there and the
    string otherwise costs nothing and keeps the risk engine importable alone.

    ``FLAT`` and anything unrecognised return ``None``, which sends the caller
    back to the sign-blind funding check. That is the conservative direction:
    with equal thresholds the sign-blind check vetoes a superset of what the
    side-aware one vetoes, so an unreadable side can only over-reject.
    """
    raw = getattr(position_side, "value", position_side)
    if not isinstance(raw, str):
        return None
    return {"LONG": 1, "SHORT": -1}.get(raw.upper())


class RiskEngine:
    """Stateful risk controller. Not thread-safe; one per bot process."""

    def __init__(
        self,
        limits: RiskLimits | None = None,
        state_path: str | Path | None = None,
        clock=time.time,
        kill_switch_path: str | Path | None = None,
    ) -> None:
        self.limits = limits or RiskLimits()
        self.state = RiskState()
        self._clock = clock
        self._state_path = Path(state_path) if state_path else None
        #: ``None`` means no switch is configured for this engine, not that the
        #: switch is absent. See :data:`DEFAULT_KILL_SWITCH_PATH` for why the
        #: engine does not pick a path of its own.
        self._kill_switch_path = Path(kill_switch_path) if kill_switch_path else None
        #: Set when the state file existed and could not be believed. While it is
        #: set nothing may overwrite that file; see :meth:`_persist`.
        self._state_unreadable = False
        self._load_state()
        # Before anything can be approved, for an engine that was given a switch.
        # A kill switch consulted only when the caller remembers to consult it is
        # exactly the kind of guard this module's header refuses to rely on.
        self.check_kill_switch()

    # ------------------------------------------------------------------
    # kill switch
    # ------------------------------------------------------------------
    @property
    def halted(self) -> bool:
        return self.state.halted

    def halt(self, reason: str) -> None:
        """Block all new entries. Idempotent: re-halting does not re-alert."""
        if self.state.halted:
            return
        self.state.halted = True
        self.state.halt_reason = reason
        logger.critical("RISK HALT: %s", reason)
        self._persist()

    def resume(self) -> None:
        """Clear the halt. Only ever called by an explicit operator action.

        Clears the kill-switch mirror with it, because leaving that set would
        misreport what the last look found. It does not remove the file: if the
        switch is still on disk the next check halts again, which is the point of
        putting it on disk.

        It also clears the funding halt and the adverse streak, because this is
        the **only** exit they have. The streak's own remedy is that the runner
        reduces, and a settlement that would reset it belongs to a position; once
        the position is flat no further settlement arrives, so a funding halt
        reached by following the rule would otherwise refuse every increase
        forever. The alternatives were both worse: hand-editing the state file,
        or calling :meth:`note_funding_settlement` with a favourable rate that
        never happened — putting a fabricated number into a guard, which is
        exactly what that method raises :class:`RiskViolation` to prevent.

        The reconciliation disputes and the stale-feed mark are untouched. Each
        already has an explicit clearing path that works while the account is
        flat — an operator note and a fresh minute — so a blanket resume that
        silently forgot a disputed position would be the failure the dispute
        exists to prevent.
        """
        self.state.halted = False
        self.state.halt_reason = ""
        self.state.kill_switch = False
        self.state.funding_halt = False
        self.state.funding_adverse_streak = 0
        logger.warning("Risk halt cleared by operator")
        self._persist()

    def check_kill_switch(self) -> bool:
        """Look for the kill-switch file and halt if it is there.

        Returns whether the switch is engaged. An engine constructed without a
        ``kill_switch_path`` has no switch configured and this is a no-op that
        returns ``False``; it reports "nothing is watching", not "the switch is
        off". The caller that wants a switch names the file — see
        :data:`DEFAULT_KILL_SWITCH_PATH` — and is responsible for calling this on
        start and on every tick. Nothing in this package calls it per tick yet;
        the demo runner is where that loop lives.

        Fails closed on a path that cannot be examined: only
        :class:`FileNotFoundError` is read as "absent". Any other ``OSError`` — a
        parent that is not a directory, a permission denial, an I/O error on the
        mount — means the answer is unknown, and an unknown kill switch is
        treated as an engaged one. Deliberately not ``Path.exists()``, which
        turns some of those errors into a confident ``False``.

        The halt is **level**-triggered, not edge-triggered on the mirror: while
        the file is there every call re-asserts the halt. Halting only on the
        absent-to-present transition left one state that traded through an
        engaged switch — a persisted mirror already set to ``True`` beside
        ``halted: False``, which is what a state file hand-edited to clear a halt
        produces — where this method answered "engaged" and approved entries
        anyway. :meth:`halt` is idempotent, so re-asserting costs nothing and
        does not re-alert.

        Removing the file clears the mirror but not the halt it caused;
        :meth:`resume` is the only way back, as it is for every halt.
        """
        if self._kill_switch_path is None:
            return False

        problem: str | None = None
        try:
            self._kill_switch_path.stat()
        except FileNotFoundError:
            present = False
        except OSError as exc:
            present, problem = True, str(exc)
        else:
            present = True

        mirror_moved = self.state.kill_switch != present
        self.state.kill_switch = present
        if present and not self.state.halted:
            if problem is None:
                self.halt("kill_switch")
            else:
                # The path and the errno go to the log, not into the reason.
                # The reason is persisted and hashed into the decision log by
                # snapshot(), and an absolute path or an OS-specific errno string
                # would make two hosts in the same semantic state hash
                # differently.
                logger.critical(
                    "Kill switch at %s could not be examined: %s",
                    self._kill_switch_path,
                    problem,
                )
                self.halt(
                    "kill_switch: the switch path could not be examined, which is "
                    "not evidence that it is absent"
                )
        elif mirror_moved:
            self._persist()
        return self.state.kill_switch

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def _persist(self) -> None:
        """Write the whole state atomically. An interrupted write changes nothing.

        A failed write is logged rather than raised: a halt that cannot be
        written down is still a halt in memory, and letting the ``OSError``
        unwind the caller would turn a disk problem into a skipped guard.

        While the loaded file was unreadable this writes **nothing**. The load
        path's decision to leave that file alone is worth only as much as the
        next write, and the next write comes almost immediately: any mutation
        persists, so the first tick would have replaced the one record of what
        the account was doing with an all-defaults document. Two steps then turn
        "I cannot tell what this account was doing" into a confident "flat, no
        drawdown, no cooldown, no dispute", with the original bytes gone.
        :meth:`chimera.futures.store.FuturesStore` refuses the same substitution
        for the same reason; :meth:`adopt_after_unreadable` is the deliberate way
        through.

        ``updated_at`` is stamped from the engine's own clock, not from the wall
        clock, so an injected or replayed clock governs every timestamp the
        engine writes rather than only the ones a decision reads.
        """
        if self._state_path is None:
            return
        if self._state_unreadable:
            logger.error(
                "Not writing risk state to %s: the file there could not be read and is "
                "being preserved for inspection. adopt_after_unreadable() is how an "
                "operator moves it aside and starts recording again.",
                self._state_path,
            )
            return
        self._prune_order_times()
        stamped = datetime.fromtimestamp(self._clock(), timezone.utc).isoformat()
        payload = (
            json.dumps(
                self.state.to_dict(updated_at=stamped),
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        temporary = self._state_path.with_suffix(self._state_path.suffix + ".tmp")
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(temporary, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self._state_path)
        except OSError as exc:
            logger.error("Could not persist risk state: %s", exc)

    def _load_state(self) -> None:
        """Restore the state, or halt because it could not be restored.

        The read decides whether there is a file, rather than ``Path.exists()``
        deciding first. ``exists()`` answers ``False`` for a path it merely could
        not examine — a parent that is not a directory, a symlink loop — so a
        state file on a degraded mount was read as "there is no state file" and
        the engine started on unhalted defaults: the exact substitution the
        fail-closed rule exists to prevent, and the one
        :meth:`check_kill_switch` refuses one screen above. Only
        :class:`FileNotFoundError` means absent here.
        """
        if self._state_path is None:
            return
        try:
            text = self._state_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return
        except OSError as exc:
            # The path goes to the log; the halt reason stays free of it, because
            # snapshot() hashes the reason and a path or errno string would differ
            # between hosts in the same semantic state.
            logger.critical("Risk state at %s could not be read: %s", self._state_path, exc)
            self._fail_closed(
                "unreadable persisted risk state: the file exists and could not be read"
            )
            return
        try:
            raw = json.loads(text)
        except ValueError as exc:
            self._fail_closed(f"unreadable persisted risk state: {exc}")
            return
        if not isinstance(raw, Mapping):
            self._fail_closed("unreadable persisted risk state: not a JSON object")
            return

        schema = raw.get("schema")
        if schema is None:
            # The pre-schema file. It carried a halt and nothing else, so that is
            # all that may be read out of it; anything wider would be inventing
            # an account state the file never claimed.
            if not set(raw).issubset(_LEGACY_KEYS):
                self._fail_closed(
                    "unreadable persisted risk state: no schema, and the keys "
                    f"({', '.join(sorted(str(k) for k in raw))}) are not the legacy "
                    "halt record"
                )
                return
            if raw.get("halted"):
                self.state.halted = True
                self.state.halt_reason = str(raw.get("halt_reason", "persisted halt"))
                logger.critical(
                    "Starting in HALTED state from %s: %s",
                    self._state_path,
                    self.state.halt_reason,
                )
            return

        if schema != RISK_STATE_SCHEMA:
            self._fail_closed(f"unknown persisted risk state schema: {schema!r}")
            return
        try:
            self.state = RiskState.from_dict(raw)
        except (KeyError, TypeError, ValueError) as exc:
            self._fail_closed(f"unreadable persisted risk state: {exc}")
            return
        if self.state.halted:
            logger.critical(
                "Starting in HALTED state from %s: %s",
                self._state_path,
                self.state.halt_reason,
            )

    def _fail_closed(self, reason: str) -> None:
        """Adopt a halted, empty state because the file could not be believed.

        The emptiness is not a claim that the account is flat — it is the
        absence of a claim, and the halt is what stops that absence being traded
        on. The reason names the problem so an operator reading the log knows
        which file to look at rather than only that trading stopped.

        It also marks the engine so that :meth:`_persist` writes nothing until an
        operator resolves it. Without that mark the emptiness would be written
        over the file it came from on the very next mutation, and the absence of
        a claim would become a recorded one.
        """
        self.state = RiskState(halted=True, halt_reason=reason)
        self._state_unreadable = True
        logger.critical("Starting in HALTED state: %s", reason)

    def adopt_after_unreadable(self, note: str) -> Path | None:
        """An operator's explicit decision to start recording again from empty.

        Moves the unreadable file aside — to ``<name>.corrupt`` — rather than
        overwriting it, because it is the only record of what the account was
        doing and a later investigation needs it. Returns where it was put, or
        ``None`` when there was no file to move.

        Requires a written reason, as
        :meth:`chimera.futures.store.FuturesStore.adopt_after_unreadable` does
        and for the same purpose: the decision is a human one, and the log should
        say who made it and why rather than merely that state appeared.

        This is not a resume. The engine stays halted with the reason it failed
        closed on; clearing that is :meth:`resume`, a separate deliberate act, so
        that "I have preserved the evidence" and "I accept trading from an empty
        state" are never the same keystroke.
        """
        if not self._state_unreadable:
            raise RiskViolation(
                "adopt_after_unreadable() is for a state file that exists and could "
                "not be read; this engine loaded its state without a problem"
            )
        if not note:
            raise RiskViolation(
                "adopting an empty risk state after an unreadable file requires a "
                "stated reason"
            )

        preserved: Path | None = None
        if self._state_path is not None:
            preserved = self._state_path.with_suffix(self._state_path.suffix + ".corrupt")
            try:
                os.replace(self._state_path, preserved)
            except FileNotFoundError:
                preserved = None
        logger.critical(
            "Operator adopted an empty risk state after an unreadable file. Reason: "
            "%s. The unreadable file was preserved at %s.",
            note,
            preserved,
        )
        self._state_unreadable = False
        self._persist()
        return preserved

    def _prune_order_times(self) -> None:
        """Drop approvals that have fallen out of the rate window."""
        now = self._clock()
        self.state.order_times = [
            t for t in self.state.order_times if now - t < _ORDER_WINDOW_S
        ]

    def snapshot(self) -> dict[str, Any]:
        """The semantic state, for hashing into a decision log.

        Identical inputs must give identical bytes, so this carries the fields a
        decision depends on and nothing that merely describes *this* process: no
        write time, no state-file path, no host, no PID. ``updated_at``
        therefore stays in the file and out of here — it changes on every write,
        including writes that changed no decision, and a hash that moved for
        that reason would report two identical states as different.

        ``schema`` *is* included: it names the contract the other fields are to
        be read under, so two states that agree field-for-field under different
        contracts should not hash alike.

        The derived drawdown is excluded for the same reason it is not
        persisted: it is a function of ``peak_equity`` and ``equity``, both of
        which are here, so carrying it would add a second reading of one fact
        rather than any information.

        The order window is reported as stored. It is pruned at every mutation
        and before every write, which keeps this a pure function of the state
        rather than of the clock at the moment somebody asked.
        """
        state = self.state
        return {
            "schema": RISK_STATE_SCHEMA,
            "equity": float(state.equity),
            "peak_equity": float(state.peak_equity),
            "day_start_equity": float(state.day_start_equity),
            "day": str(state.day),
            "daily_pnl": float(state.daily_pnl),
            "open_positions": {
                key: float(state.open_positions[key]) for key in sorted(state.open_positions)
            },
            "order_times": [float(t) for t in state.order_times],
            "consecutive_losses": int(state.consecutive_losses),
            "cooldown_until": float(state.cooldown_until),
            "halted": bool(state.halted),
            "halt_reason": str(state.halt_reason),
            "kill_switch": bool(state.kill_switch),
            "stale_feed_since": (
                None if state.stale_feed_since is None else float(state.stale_feed_since)
            ),
            "reconciliation_disputed": {
                key: str(state.reconciliation_disputed[key])
                for key in sorted(state.reconciliation_disputed)
            },
            "funding_adverse_streak": int(state.funding_adverse_streak),
            "funding_halt": bool(state.funding_halt),
        }

    # ------------------------------------------------------------------
    # account state
    # ------------------------------------------------------------------
    def update_equity(self, equity: float, now: datetime | None = None) -> None:
        """Record current equity and trip account-level guards if breached.

        The UTC day defaults to the engine's own clock rather than to the wall
        clock, so a replayed or fixture-driven run rolls its day on the clock
        that drives everything else it does. ``day`` and ``day_start_equity`` are
        persisted and hashed into :meth:`snapshot`, so two of them being governed
        by a clock that disagrees with the cooldown's and the rate window's would
        be a divergence a replay could not reproduce.

        A wipeout is recorded before it is halted on. ``equity`` and
        ``daily_pnl`` are persisted for the daily report, and returning first
        left the file holding the last healthy reading — so the report for the
        day an account went to zero showed a flat P&L. The halt is unaffected;
        this is about what the file says happened.
        """
        if equity <= 0:
            self.state.equity = equity
            self.state.daily_pnl = equity - self.state.day_start_equity
            self.halt(f"equity is non-positive: {equity}")
            self._persist()
            return

        stamp = now or datetime.fromtimestamp(self._clock(), timezone.utc)
        today = stamp.date().isoformat()
        if self.state.day != today:
            self.state.day = today
            self.state.day_start_equity = equity

        self.state.equity = equity
        self.state.peak_equity = max(self.state.peak_equity, equity)
        self.state.daily_pnl = equity - self.state.day_start_equity

        drawdown = (self.state.peak_equity - equity) / self.state.peak_equity
        if drawdown >= self.limits.max_drawdown_pct:
            self.halt(
                f"max drawdown breached: {drawdown:.2%} >= "
                f"{self.limits.max_drawdown_pct:.2%}"
            )
            self._persist()
            return

        if self.state.day_start_equity > 0:
            daily_loss = (self.state.day_start_equity - equity) / self.state.day_start_equity
            if daily_loss >= self.limits.max_daily_loss_pct:
                self.halt(
                    f"max daily loss breached: {daily_loss:.2%} >= "
                    f"{self.limits.max_daily_loss_pct:.2%}"
                )
        self._persist()

    def current_drawdown(self) -> float:
        if self.state.peak_equity <= 0:
            return 0.0
        return (self.state.peak_equity - self.state.equity) / self.state.peak_equity

    def record_order(self) -> None:
        """Register an order for rate limiting. Halts if the rate is exceeded."""
        now = self._clock()
        self._prune_order_times()
        self.state.order_times.append(now)
        if len(self.state.order_times) > self.limits.max_orders_per_minute:
            self.halt(
                f"order rate limit exceeded: {len(self.state.order_times)} orders/min > "
                f"{self.limits.max_orders_per_minute}"
            )
        self._persist()

    def record_trade_result(self, profit_abs: float) -> None:
        """Track the loss streak and start a cooldown when it is hit."""
        if profit_abs < 0:
            self.state.consecutive_losses += 1
            if self.state.consecutive_losses >= self.limits.loss_streak_limit:
                self.state.cooldown_until = self._clock() + self.limits.cooldown_seconds
                logger.warning(
                    "%d consecutive losses, entries paused for %.0fs",
                    self.state.consecutive_losses,
                    self.limits.cooldown_seconds,
                )
        else:
            self.state.consecutive_losses = 0
        self._persist()

    def set_position_exposure(self, pair: str, stake: float) -> None:
        """Record ``pair``'s *current total* exposure, replacing any previous value.

        Assignment, not accumulation. Freqtrade calls ``order_filled`` once per
        order that reaches a closed state — entries, partial fills, position
        adjustments and partial exits alike — and
        ``LocalTrade.recalc_trade_from_orders`` rewrites ``trade.stake_amount``
        to the trade's total stake each time. Adding that total on every
        callback double-counted: two fills of one 200 position reported 400.

        Keyed by pair because Freqtrade opens at most one trade per pair — it
        removes pairs with an open trade from the candidate whitelist in
        ``FreqtradeBot.enter_positions``. Position adjustments extend that one
        trade rather than creating a second.
        """
        if stake > 0:
            self.state.open_positions[pair] = stake
        else:
            self.state.open_positions.pop(pair, None)
        self._persist()

    def close_position(self, pair: str) -> None:
        self.state.open_positions.pop(pair, None)
        self._persist()

    @property
    def total_exposure(self) -> float:
        return sum(self.state.open_positions.values())

    # ------------------------------------------------------------------
    # runner and operator notes
    # ------------------------------------------------------------------
    def note_feed(self, last_minute_close_ns: int, now_ns: int) -> None:
        """Record how far the feed is behind its last complete minute.

        The delay is ``now - last_minute_close``; a delay *above*
        ``max_data_delay_s`` marks the feed stale and exactly at the limit does
        not, which is the comparison :meth:`evaluate_entry` already makes on
        ``data_delay_s``. The instant it first went stale is what is kept, not
        the latest observation, so the mark answers "since when" instead of
        drifting forward on every tick.

        A **negative** delay is not a fresh feed, it is an impossible one: the
        last complete minute cannot close after the present. It means the
        recorder's clock and the runner's disagree, or that one of the two
        arguments arrived in the wrong unit. Either way nothing here knows how
        far behind the feed is, so it is marked stale rather than cleared —
        otherwise a skewed clock would silently lift an existing stale mark and
        re-enable entries on data nobody has checked. The mirrored unit mistake
        (a close in milliseconds against a now in nanoseconds) produces a huge
        positive delay and already vetoes, so this closes the one direction that
        failed open.

        Persisting it is the point: a runner that restarts into a dead feed
        would otherwise begin with no opinion about freshness and approve
        entries on data nobody has checked.
        """
        delay_s = (now_ns - last_minute_close_ns) / 1e9
        if delay_s < 0:
            if self.state.stale_feed_since is None:
                self.state.stale_feed_since = now_ns / 1e9
                logger.critical(
                    "Feed reports a last minute close %.0fs in the future; the clocks "
                    "or the units disagree, so the feed's age is unknown",
                    -delay_s,
                )
        elif delay_s > self.limits.max_data_delay_s:
            if self.state.stale_feed_since is None:
                self.state.stale_feed_since = now_ns / 1e9
                logger.warning(
                    "Feed is %.0fs past the last minute close, above the %.0fs limit",
                    delay_s,
                    self.limits.max_data_delay_s,
                )
        else:
            self.state.stale_feed_since = None
        self._persist()

    def note_reconciliation(self, symbol: str, disputed: str | None) -> None:
        """Open or close a reconciliation dispute on one symbol.

        A dispute means the local position and the venue's disagree, so the size
        every limit would be applied to is not known. Increases on that symbol
        are refused until an operator says the disagreement is resolved, by
        calling this with ``None``. Nothing else clears it — in particular a
        restart does not, which is why it lives in the persisted state: a
        dispute a reboot forgets is a dispute that gets traded through.
        """
        if disputed is None:
            self.state.reconciliation_disputed.pop(symbol, None)
            logger.warning("Reconciliation dispute on %s cleared by operator", symbol)
        else:
            self.state.reconciliation_disputed[symbol] = disputed
            logger.critical("Reconciliation dispute on %s: %s", symbol, disputed)
        self._persist()

    def note_funding_settlement(
        self, pair: str, side: "PositionSide | str", rate: float
    ) -> None:
        """Count a settlement as paid or received by the position that held it.

        The cost to a position is ``sign(side) * rate``: a long pays a positive
        funding rate and a short pays a negative one, the table
        :mod:`chimera.futures.accounting` states once for the whole system. That
        module writes the same fact as a *cash flow*, ``-sign(side) * notional *
        rate``, which is negative when the position pays; a cost is the negation
        of that, so the two signs are the same statement read from opposite ends
        and neither is free to drift from the table.

        A positive cost is adverse and extends the streak; a negative cost is a
        rebate and resets it, together with any funding halt, because the
        condition that raised the halt has gone. A settlement of exactly zero is
        neither paid nor received and leaves the streak where it is rather than
        quietly forgiving it.

        Raises :class:`RiskViolation` for a side that is neither LONG nor SHORT:
        a settlement belongs to a position, and guessing which way an
        unattributed one cut would put a fabricated number into a guard.
        """
        sign = _position_sign(side)
        if sign is None:
            raise RiskViolation(
                f"{pair}: funding settlement at rate {rate} has no position side to "
                f"attribute it to (got {side!r}); it cannot be scored paid or received"
            )
        cost = sign * rate
        if cost > 0:
            self.state.funding_adverse_streak += 1
            if self.state.funding_adverse_streak >= self.limits.funding_adverse_streak_limit:
                self.state.funding_halt = True
                logger.critical(
                    "%s paid funding on %d settlements in a row; increases are refused",
                    pair,
                    self.state.funding_adverse_streak,
                )
        elif cost < 0:
            self.state.funding_adverse_streak = 0
            self.state.funding_halt = False
        self._persist()

    # ------------------------------------------------------------------
    # sizing
    # ------------------------------------------------------------------
    def position_size(
        self,
        equity: float,
        entry_price: float,
        stop_price: float,
        leverage: float = 1.0,
    ) -> float:
        """Stake (in quote currency) such that hitting the stop costs
        ``risk_per_trade_pct`` of equity.

        Returns 0.0 when the stop distance is outside the configured band: too
        tight a stop implies an unbounded position, too wide a stop means the
        trade's risk cannot be controlled.

        This deliberately does *not* shrink the stake to fit the remaining
        exposure headroom. Sizing answers "how large should this trade be?";
        whether there is room for it is :meth:`evaluate_entry`'s decision. Were
        sizing to clamp instead, a portfolio at 97% of its exposure cap would
        quietly open a position 3% of the intended size — all of the fees, none
        of the edge — and the exposure limit would never actually reject
        anything.
        """
        if equity <= 0 or entry_price <= 0:
            return 0.0

        stop_distance = abs(entry_price - stop_price) / entry_price
        if not (
            self.limits.min_stop_distance_pct
            <= stop_distance
            <= self.limits.max_stop_distance_pct
        ):
            logger.warning(
                "Stop distance %.4f outside [%.4f, %.4f], refusing to size",
                stop_distance,
                self.limits.min_stop_distance_pct,
                self.limits.max_stop_distance_pct,
            )
            return 0.0

        risk_capital = equity * self.limits.risk_per_trade_pct
        # Notional such that a `stop_distance` adverse move loses risk_capital.
        notional = risk_capital / stop_distance

        capped_leverage = max(1.0, min(leverage, self.limits.max_leverage))
        # Stake is the margin the account actually commits.
        stake = notional / capped_leverage

        return max(0.0, min(stake, equity * self.limits.max_position_pct))

    # ------------------------------------------------------------------
    # entry gate
    # ------------------------------------------------------------------
    def evaluate_entry(
        self,
        pair: str,
        equity: float,
        entry_price: float,
        stop_price: float,
        leverage: float = 1.0,
        proposed_stake: float | None = None,
        data_delay_s: float | None = None,
        inference_age_s: float | None = None,
        funding_rate: float | None = None,
        liquidation_price: float | None = None,
        exchange_healthy: bool = True,
        position_side: "PositionSide | str | None" = None,
    ) -> RiskDecision:
        """The one gate every entry must pass.

        Checks run cheapest-and-most-fatal first. The returned reason is what
        gets logged and exported as a ``rejected_entries`` metric label, so it
        is written to be readable in a dashboard.

        ``proposed_stake`` is the stake the caller is *actually about to
        commit*, in quote currency. When it is given, every limit is applied to
        that number and the trade is additionally rejected if it exceeds what
        risk-based sizing would have allowed — so an exchange minimum, a
        rounding step or any other Freqtrade adjustment cannot quietly inflate a
        position past the risk envelope. When it is omitted the engine falls
        back to its own sizing, which is what a caller that has not yet built an
        order (a pre-trade check) needs.

        ``data_delay_s`` is how late the newest candle is *past its close*, not
        its age since opening. See :attr:`RiskLimits.max_data_delay_s`. It is the
        caller's live measurement; the persisted staleness mark that
        :meth:`note_feed` sets is checked beside it and does not depend on the
        caller remembering to pass anything.

        ``position_side`` turns the funding check from a sign-blind bound on the
        magnitude of the rate into a bound on what this side would actually pay
        — ``sign(side) * rate``, the convention
        :mod:`chimera.futures.accounting` fixes for the whole system: a long pays
        a positive rate, a short pays a negative one. Callers that give no side
        keep the old behaviour, which vetoes a superset and so cannot approve
        anything the side-aware check would refuse.

        This method reads state and does no I/O. The kill-switch *file* is
        consulted by :meth:`check_kill_switch`, which a runner is required to
        call on start and on every tick; by the time the gate runs an engaged
        switch is already a halt. Nothing in this package drives that loop yet —
        the demo runner is where it will live — so today the switch is read only
        when an entry point constructs an engine with a path or calls the method
        itself.
        """
        lim = self.limits

        if self.state.halted:
            return RiskDecision(False, f"halted: {self.state.halt_reason}")

        if self._clock() < self.state.cooldown_until:
            remaining = self.state.cooldown_until - self._clock()
            return RiskDecision(False, f"cooldown active for another {remaining:.0f}s")

        if not exchange_healthy:
            return RiskDecision(False, "exchange/API reported unhealthy")

        if equity <= 0:
            return RiskDecision(False, "no equity")

        dispute = self.state.reconciliation_disputed.get(pair)
        if dispute is not None:
            return RiskDecision(False, f"reconciliation dispute on {pair}: {dispute}")

        if self.state.funding_halt:
            return RiskDecision(
                False,
                f"funding halt: {self.state.funding_adverse_streak} consecutive adverse "
                f"settlements >= {lim.funding_adverse_streak_limit}",
            )

        if self.state.stale_feed_since is not None:
            return RiskDecision(
                False,
                "market data late: the feed has been marked stale since "
                f"{self.state.stale_feed_since:.0f} (epoch seconds) and no complete "
                "minute has arrived since",
            )

        if data_delay_s is not None and data_delay_s > lim.max_data_delay_s:
            return RiskDecision(
                False,
                f"market data late: {data_delay_s:.0f}s past candle close > "
                f"{lim.max_data_delay_s:.0f}s",
            )

        if inference_age_s is not None and inference_age_s > lim.max_inference_staleness_s:
            return RiskDecision(
                False,
                f"inference stale: {inference_age_s:.0f}s > "
                f"{lim.max_inference_staleness_s:.0f}s",
            )

        if pair not in self.state.open_positions and (
            len(self.state.open_positions) >= lim.max_open_positions
        ):
            return RiskDecision(
                False,
                f"max open positions reached: {len(self.state.open_positions)} >= "
                f"{lim.max_open_positions}",
            )

        if funding_rate is not None:
            sign = _position_sign(position_side)
            if sign is None:
                if abs(funding_rate) > lim.max_funding_rate:
                    return RiskDecision(
                        False,
                        f"funding rate {funding_rate:.5f} exceeds "
                        f"{lim.max_funding_rate:.5f}",
                    )
            else:
                cost = sign * funding_rate
                if cost > lim.max_funding_cost_rate:
                    return RiskDecision(
                        False,
                        f"funding rate would cost this position {cost:.5f} per "
                        f"settlement, over {lim.max_funding_cost_rate:.5f}",
                    )

        if leverage > lim.max_leverage:
            return RiskDecision(
                False, f"leverage {leverage:.2f} exceeds max {lim.max_leverage:.2f}"
            )

        if liquidation_price is not None and entry_price > 0:
            distance = abs(entry_price - liquidation_price) / entry_price
            if distance < lim.min_liquidation_distance_pct:
                return RiskDecision(
                    False,
                    f"liquidation only {distance:.2%} away, need "
                    f"{lim.min_liquidation_distance_pct:.2%}",
                )

        allowed_stake = self.position_size(equity, entry_price, stop_price, leverage)
        if allowed_stake <= 0:
            return RiskDecision(False, "position sizing returned zero stake")

        stake = allowed_stake if proposed_stake is None else proposed_stake
        if stake <= 0:
            return RiskDecision(False, "order stake is zero")

        # An exchange minimum, a precision step or any other Freqtrade
        # adjustment may have raised the order above what risk-based sizing
        # permits. Getting filled at a size we would never have chosen is the
        # failure this check exists to stop; taking the trade anyway would mean
        # the stated risk-per-trade is simply untrue.
        if stake > allowed_stake + 1e-9:
            return RiskDecision(
                False,
                f"order stake {stake:.2f} exceeds the risk-based maximum "
                f"{allowed_stake:.2f}",
            )

        projected = self.total_exposure + stake
        if projected > equity * lim.max_total_exposure_pct:
            return RiskDecision(
                False,
                f"total exposure {projected:.2f} would exceed "
                f"{equity * lim.max_total_exposure_pct:.2f}",
            )

        pair_exposure = self.state.open_positions.get(pair, 0.0) + stake
        if pair_exposure > equity * lim.max_exposure_per_asset_pct:
            return RiskDecision(
                False,
                f"exposure in {pair} would reach {pair_exposure:.2f}, over "
                f"{equity * lim.max_exposure_per_asset_pct:.2f}",
            )

        return RiskDecision(True, "ok", stake)
