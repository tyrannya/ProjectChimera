"""``CarryLedger``: the persistent cash and attribution record of one carry position.

The executors already persist positions and orders (:mod:`chimera.futures.store`).
This ledger persists what neither leg can know on its own: the capital the
position was opened against, where every unit of it went, and whether the two
legs still tell the same story.

**Fails closed, always.** A ledger that cannot be read, whose schema this build
does not recognise, or whose identity does not reconcile is a ledger that is
*disputed* -- never one that is silently reset. A reset ledger looks exactly
like a position that never traded, and that is the one thing an operator must
never be shown by accident. Every refusal here leaves the file on disk
untouched, because it is the only record of what the process did.

**Funding is never netted.** ``funding_received`` and ``funding_paid`` are kept
apart (section 6.5) and reported apart. A single net number hides the case the
demo exists to observe: a position that received a large amount early and has
been paying since.

**The running identity.** Section 6.5 requires that the price PnL of a hedged
position equals ``Q * (basis_entry - basis_now)``, checked against the two legs'
marked PnL every minute. :meth:`CarryLedger.check_identity` is that check. It
has a tolerance because fees and rounding are real, and any disagreement wider
than the tolerance raises a dispute rather than being absorbed.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from chimera.carry.accounting import ZERO, CarryError

logger = logging.getLogger(__name__)

#: The persisted schema. A file that does not declare exactly this is refused.
LEDGER_SCHEMA = "chimera.carry-ledger/1"

#: How far the running basis identity may disagree with the legs' marked PnL
#: before the position is disputed, in quote currency. Fees and step rounding
#: are real, so the check cannot demand exact equality; a tolerance this tight
#: still catches a leg that is wrong by a fill.
IDENTITY_TOLERANCE = Decimal("0.01")


class LedgerError(CarryError):
    """The ledger cannot be read, written, or reconciled."""


class LoadOutcome(str, Enum):
    """How :meth:`CarryLedger.open` found the file. Never inferred later."""

    MISSING = "MISSING"
    LOADED = "LOADED"
    UNREADABLE = "UNREADABLE"


def _decimal(value: Any, where: str) -> Decimal:
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise LedgerError(f"{where} is not a decimal: {value!r}") from exc


def _optional_decimal(value: Any, where: str) -> Decimal | None:
    return None if value is None else _decimal(value, where)


def _text(value: Decimal | None) -> str | None:
    return None if value is None else str(value)


@dataclass
class LegAccrual:
    """One leg's cost and realisation accumulators, reported per leg (section 6.5)."""

    fees: Decimal = ZERO
    slippage: Decimal = ZERO
    realised: Decimal = ZERO

    def to_dict(self) -> dict[str, Any]:
        return {
            "fees": str(self.fees),
            "slippage": str(self.slippage),
            "realised": str(self.realised),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], where: str) -> "LegAccrual":
        return cls(
            fees=_decimal(data.get("fees", "0"), f"{where}.fees"),
            slippage=_decimal(data.get("slippage", "0"), f"{where}.slippage"),
            realised=_decimal(data.get("realised", "0"), f"{where}.realised"),
        )


@dataclass
class CarryLedgerState:
    """Everything the ledger persists. Decision-relevant fields only."""

    capital: Decimal
    free_cash: Decimal
    spot: LegAccrual = field(default_factory=LegAccrual)
    perp: LegAccrual = field(default_factory=LegAccrual)
    funding_received: Decimal = ZERO
    funding_paid: Decimal = ZERO
    rebalance_cost: Decimal = ZERO
    quantity: Decimal = ZERO
    spot_entry: Decimal | None = None
    perp_entry: Decimal | None = None
    perp_margin: Decimal = ZERO
    entry_basis: Decimal | None = None
    current_basis: Decimal | None = None
    last_equity: Decimal | None = None
    worst_equity: Decimal | None = None
    #: Settlement instants already booked. The dedup key, persisted so a restart
    #: cannot re-book a settlement the previous process already applied.
    settled: list[int] = field(default_factory=list)
    #: When the position now held was opened, in integer ns. Section 6.9's
    #: funding window is ``open_instant < settlement <= now``, so the lower bound
    #: is a fact about THIS position and has to outlive the process that opened
    #: it; a restart that forgot it would either re-charge settlements from
    #: before the open or, defaulting the other way, charge none at all.
    #:
    #: ``None`` means one of two things and they are deliberately not
    #: distinguished here: the position is flat, or the ledger was written by a
    #: build that did not record the instant. Both are answered the same way by
    #: the caller -- a non-flat position with no open instant has an unknowable
    #: funding window and is refused rather than guessed at. This field is
    #: additive: :meth:`from_dict` reads it with a ``None`` default, so a file
    #: written before it existed still loads under the same schema id, and the
    #: absence means exactly what it says.
    open_instant_ns: int | None = None
    #: The most recent identity residual, kept so a report can show how close the
    #: position runs to its tolerance rather than only whether it broke it.
    identity_gap: Decimal | None = None
    #: Non-empty means DISPUTED. Cleared only by :meth:`resolve`, with a note.
    disputed: str | None = None
    #: Every operator resolution, with its mandatory note. Append-only evidence.
    resolutions: list[dict[str, str]] = field(default_factory=list)
    #: Per-leg last-observed instant, in integer ns. Section 6.8's stale-leg rule
    #: is defined on the two legs' observation times; `FuturesState` persists no
    #: such field, and adding one would change a store schema the frozen dry-run
    #: protocol exercises, so the carry package owns these instants itself.
    marked_at_ns: dict[str, int] = field(default_factory=dict)
    updated_at: str = ""

    @property
    def net_funding(self) -> Decimal:
        return self.funding_received - self.funding_paid

    @property
    def fees(self) -> Decimal:
        return self.spot.fees + self.perp.fees

    @property
    def slippage(self) -> Decimal:
        return self.spot.slippage + self.perp.slippage

    @property
    def realised(self) -> Decimal:
        return self.spot.realised + self.perp.realised

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": LEDGER_SCHEMA,
            "capital": str(self.capital),
            "free_cash": str(self.free_cash),
            "spot": self.spot.to_dict(),
            "perp": self.perp.to_dict(),
            "funding_received": str(self.funding_received),
            "funding_paid": str(self.funding_paid),
            "rebalance_cost": str(self.rebalance_cost),
            "quantity": str(self.quantity),
            "spot_entry": _text(self.spot_entry),
            "perp_entry": _text(self.perp_entry),
            "perp_margin": str(self.perp_margin),
            "entry_basis": _text(self.entry_basis),
            "current_basis": _text(self.current_basis),
            "last_equity": _text(self.last_equity),
            "worst_equity": _text(self.worst_equity),
            "settled": list(self.settled),
            "open_instant_ns": self.open_instant_ns,
            "identity_gap": _text(self.identity_gap),
            "disputed": self.disputed,
            "resolutions": [dict(r) for r in self.resolutions],
            "marked_at_ns": dict(sorted(self.marked_at_ns.items())),
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CarryLedgerState":
        schema = str(data.get("schema", ""))
        if schema != LEDGER_SCHEMA:
            raise LedgerError(
                f"persisted carry ledger declares schema {schema!r}, not {LEDGER_SCHEMA!r}. "
                "A ledger this build cannot read is not one it may guess at."
            )
        settled_raw = data.get("settled", [])
        if not isinstance(settled_raw, list):
            raise LedgerError("settled must be a list of settlement instants")
        try:
            settled = [int(instant) for instant in settled_raw]
        except (TypeError, ValueError) as exc:
            raise LedgerError(f"settled holds a non-integer instant: {settled_raw!r}") from exc

        marked_raw = data.get("marked_at_ns", {})
        if not isinstance(marked_raw, Mapping):
            raise LedgerError("marked_at_ns must be a mapping of leg to instant")
        try:
            marked = {str(k): int(v) for k, v in marked_raw.items()}
        except (TypeError, ValueError) as exc:
            raise LedgerError(
                f"marked_at_ns holds a non-integer instant: {marked_raw!r}"
            ) from exc

        opened_raw = data.get("open_instant_ns")
        try:
            opened = None if opened_raw is None else int(opened_raw)
        except (TypeError, ValueError) as exc:
            raise LedgerError(
                f"open_instant_ns holds a non-integer instant: {opened_raw!r}"
            ) from exc

        disputed = data.get("disputed")
        return cls(
            capital=_decimal(data.get("capital", "0"), "capital"),
            free_cash=_decimal(data.get("free_cash", "0"), "free_cash"),
            spot=LegAccrual.from_dict(dict(data.get("spot", {})), "spot"),
            perp=LegAccrual.from_dict(dict(data.get("perp", {})), "perp"),
            funding_received=_decimal(data.get("funding_received", "0"), "funding_received"),
            funding_paid=_decimal(data.get("funding_paid", "0"), "funding_paid"),
            rebalance_cost=_decimal(data.get("rebalance_cost", "0"), "rebalance_cost"),
            quantity=_decimal(data.get("quantity", "0"), "quantity"),
            spot_entry=_optional_decimal(data.get("spot_entry"), "spot_entry"),
            perp_entry=_optional_decimal(data.get("perp_entry"), "perp_entry"),
            perp_margin=_decimal(data.get("perp_margin", "0"), "perp_margin"),
            entry_basis=_optional_decimal(data.get("entry_basis"), "entry_basis"),
            current_basis=_optional_decimal(data.get("current_basis"), "current_basis"),
            last_equity=_optional_decimal(data.get("last_equity"), "last_equity"),
            worst_equity=_optional_decimal(data.get("worst_equity"), "worst_equity"),
            settled=settled,
            open_instant_ns=opened,
            identity_gap=_optional_decimal(data.get("identity_gap"), "identity_gap"),
            disputed=None if disputed is None else str(disputed),
            resolutions=[
                {str(k): str(v) for k, v in dict(r).items()}
                for r in data.get("resolutions", [])
            ],
            marked_at_ns=marked,
            updated_at=str(data.get("updated_at", "")),
        )


def _now_text(instant_ns: int | None) -> str:
    """The runner clock's instant as ISO-8601 UTC, or the epoch when unknown.

    Never ``datetime.now()``: section 2.4 forbids decision-path code from
    reading the wall clock, and a persisted timestamp that moves between a live
    run and its replay would break replay parity (section 10).
    """
    if instant_ns is None:
        return ""
    moment = datetime.fromtimestamp(instant_ns / 1_000_000_000, tz=timezone.utc)
    return moment.isoformat().replace("+00:00", "+00:00")


@dataclass
class CarryLedger:
    """Load, mutate and atomically persist a :class:`CarryLedgerState`."""

    path: Path | None
    state: CarryLedgerState
    outcome: LoadOutcome = LoadOutcome.MISSING

    # -- lifecycle ---------------------------------------------------------

    @classmethod
    def open(cls, path: str | Path | None, *, capital: Decimal) -> "CarryLedger":
        """Read the ledger if there is one; otherwise start one against ``capital``.

        An unreadable or schema-foreign file is **not** replaced by a fresh
        ledger. It loads as ``UNREADABLE`` with the position already disputed,
        and the file is left exactly as it was found.
        """
        if path is None:
            return cls(
                path=None,
                state=CarryLedgerState(capital=capital, free_cash=capital),
                outcome=LoadOutcome.MISSING,
            )

        location = Path(path)
        if not location.exists():
            return cls(
                path=location,
                state=CarryLedgerState(capital=capital, free_cash=capital),
                outcome=LoadOutcome.MISSING,
            )

        try:
            data = json.loads(location.read_text(encoding="utf-8"))
            state = CarryLedgerState.from_dict(data)
        except (OSError, ValueError, ArithmeticError, LedgerError, KeyError, TypeError) as exc:
            logger.critical(
                "Carry ledger at %s could not be read (%s). The position is DISPUTED and "
                "the file is left untouched: it is the only record of what this position "
                "did, and a ledger that resets itself is indistinguishable from one that "
                "never traded.",
                location,
                exc,
            )
            damaged = CarryLedgerState(capital=capital, free_cash=capital)
            damaged.disputed = f"ledger_unreadable: {exc}"
            return cls(path=location, state=damaged, outcome=LoadOutcome.UNREADABLE)

        if state.capital != capital:
            state.disputed = (
                f"ledger_capital_mismatch: file says {state.capital}, configuration says "
                f"{capital}"
            )
        return cls(path=location, state=state, outcome=LoadOutcome.LOADED)

    def save(self, *, now_ns: int | None = None) -> None:
        """Write atomically: temp -> flush -> fsync -> replace."""
        if self.path is None:
            return
        self.state.updated_at = _now_text(now_ns)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        payload = json.dumps(self.state.to_dict(), indent=2, sort_keys=True) + "\n"
        try:
            with open(temporary, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        except OSError as exc:
            raise LedgerError(
                f"could not persist the carry ledger to {self.path}: {exc}. Continuing "
                "would mean the next restart cannot know what this position did."
            ) from exc

    # -- dispute and resolution -------------------------------------------

    @property
    def disputed(self) -> str | None:
        return self.state.disputed

    def dispute(self, reason: str) -> None:
        """Raise a reconciliation dispute. The first reason is kept, not the last."""
        if not reason:
            raise LedgerError("a dispute must state a reason")
        if self.state.disputed is None:
            self.state.disputed = reason

    def resolve(self, note: str, *, now_ns: int | None = None) -> None:
        """Clear a dispute. The note is mandatory and is evidence."""
        if not note or not note.strip():
            raise LedgerError(
                "resolving a carry ledger dispute requires an operator note. The note is "
                "the evidence that a person looked at both legs and decided."
            )
        if self.state.disputed is None:
            raise LedgerError("there is no dispute to resolve")
        self.state.resolutions.append(
            {
                "cleared": self.state.disputed,
                "note": note.strip(),
                "at": _now_text(now_ns),
            }
        )
        self.state.disputed = None

    # -- booking -----------------------------------------------------------

    def book_entry(
        self,
        *,
        quantity: Decimal,
        spot_fill: Decimal,
        perp_fill: Decimal,
        spot_fee: Decimal,
        perp_fee: Decimal,
        spot_slippage: Decimal,
        perp_slippage: Decimal,
        perp_margin: Decimal,
    ) -> None:
        """Record an opened hedge: quantities, entry prices, entry basis and costs."""
        self.state.quantity = quantity
        self.state.spot_entry = spot_fill
        self.state.perp_entry = perp_fill
        self.state.perp_margin = perp_margin
        self.state.entry_basis = perp_fill - spot_fill
        self.state.current_basis = self.state.entry_basis
        self.state.spot.fees += spot_fee
        self.state.perp.fees += perp_fee
        self.state.spot.slippage += spot_slippage
        self.state.perp.slippage += perp_slippage
        self.state.free_cash -= quantity * spot_fill
        self.state.free_cash -= perp_margin
        self.state.free_cash -= spot_fee + perp_fee + spot_slippage + perp_slippage

    def book_funding(self, instant_ns: int, flow: Decimal) -> Decimal:
        """Book one settlement's signed flow. Returns 0 if already booked.

        Deduplicated by settlement instant, and the instants are persisted, so a
        restart between the venue's settlement and this ledger's save cannot book
        it twice.
        """
        if instant_ns in self.state.settled:
            return ZERO
        self.state.settled.append(instant_ns)
        self.state.free_cash += flow
        if flow < ZERO:
            self.state.funding_paid += -flow
        elif flow > ZERO:
            self.state.funding_received += flow
        return flow

    def book_costs(
        self,
        *,
        leg: str,
        fee: Decimal = ZERO,
        slippage: Decimal = ZERO,
        realised: Decimal = ZERO,
        rebalance: bool = False,
    ) -> None:
        """Attribute a fill's frictions and realisation to one leg."""
        accrual = self._leg(leg)
        accrual.fees += fee
        accrual.slippage += slippage
        accrual.realised += realised
        self.state.free_cash -= fee + slippage
        self.state.free_cash += realised
        if rebalance:
            self.state.rebalance_cost += fee + slippage

    def _leg(self, leg: str) -> LegAccrual:
        if leg == "spot":
            return self.state.spot
        if leg == "perp":
            return self.state.perp
        raise LedgerError(f"unknown leg {leg!r}; the carry position has exactly spot and perp")

    def note_open_instant(self, *, flat: bool, instant_ns: int) -> None:
        """Move section 6.9's funding window with the position it belongs to.

        ``instant_ns`` is the instant the position came into EXISTENCE, which is
        the minute's CLOSE -- the price a leg fills at. The caller converts; this
        method stores what it is given and never guesses which end of a minute it
        was handed.

        Set when a flat position becomes non-flat, cleared when it returns to
        flat, and left alone in between -- so a position that is increased,
        corrected or rebalanced keeps the instant it was OPENED at, which is what
        ``open_instant < settlement <= now`` names.

        Clearing on flat is the half that matters. Without it a position that
        closed and later reopened would carry the FIRST open's instant, and a
        settlement that fell in the gap -- while nothing was held -- would land
        inside the window and be charged against the second position. The
        settlement instants already in :attr:`CarryLedgerState.settled` happen to
        cover the common case, but only because the runner books each settlement
        as it passes; the window is what makes the answer right by construction
        rather than by luck.
        """
        if flat:
            self.state.open_instant_ns = None
        elif self.state.open_instant_ns is None:
            self.state.open_instant_ns = int(instant_ns)

    def note_leg_mark(self, leg: str, instant_ns: int) -> None:
        """Record when a leg was last observed, for the stale-leg rule."""
        self._leg(leg)  # validates the name
        self.state.marked_at_ns[leg] = int(instant_ns)

    def stale_leg(self, *, tolerance_ns: int) -> str | None:
        """The leg whose last observation lags the other by more than ``tolerance_ns``."""
        spot = self.state.marked_at_ns.get("spot")
        perp = self.state.marked_at_ns.get("perp")
        if spot is None or perp is None:
            return None
        if spot - perp > tolerance_ns:
            return "perp"
        if perp - spot > tolerance_ns:
            return "spot"
        return None

    # -- marking and the running identity ----------------------------------

    def mark(self, *, spot_close: Decimal, perp_close: Decimal, equity: Decimal) -> Decimal:
        """Mark the position, update the basis and the worst equity, return equity."""
        self.state.current_basis = perp_close - spot_close
        self.state.last_equity = equity
        if self.state.worst_equity is None or equity < self.state.worst_equity:
            self.state.worst_equity = equity
        return equity

    def check_identity(
        self,
        *,
        spot_close: Decimal,
        perp_close: Decimal,
        tolerance: Decimal = IDENTITY_TOLERANCE,
    ) -> Decimal:
        """``Q * (basis_entry - basis_now)`` against the two legs' marked PnL.

        Returns the residual. A residual wider than ``tolerance`` raises a
        dispute: the two legs disagree about what the position is worth, and an
        unexplained disagreement is not something to average away.
        """
        if self.state.entry_basis is None or self.state.spot_entry is None:
            self.state.identity_gap = None
            return ZERO
        if self.state.perp_entry is None:
            self.state.identity_gap = None
            return ZERO

        quantity = self.state.quantity
        basis_now = perp_close - spot_close
        from_basis = quantity * (self.state.entry_basis - basis_now)
        spot_pnl = quantity * (spot_close - self.state.spot_entry)
        perp_pnl = quantity * (self.state.perp_entry - perp_close)
        residual = from_basis - (spot_pnl + perp_pnl)
        self.state.identity_gap = residual
        if abs(residual) > tolerance:
            self.dispute(
                f"identity_violation: Q*(basis_entry - basis_now) = {from_basis} but the "
                f"legs mark to {spot_pnl + perp_pnl} (residual {residual}, tolerance "
                f"{tolerance})"
            )
        return residual

    def snapshot(self) -> dict[str, Any]:
        """The ledger as a plain mapping, for the decision log and the reports."""
        return replace(self.state).to_dict()
