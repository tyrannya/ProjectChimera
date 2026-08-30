"""Persisted futures execution state, and what a restart is allowed to assume.

The rule this module exists to enforce is one sentence long: **an empty memory is
not a flat account.** A process that starts, finds nothing, and proceeds as if it
holds no position will happily open a second one on top of the first. So the
store distinguishes three situations that a plain ``dict.get`` would flatten into
one:

``loaded``
    a state file was read. Its positions, orders, ledger and applied-event ids
    are the local truth, and the executor may act on them once reconciliation
    against the venue agrees.

``missing``
    no state file exists. The executor starts **unbootstrapped** and refuses to
    plan anything until :meth:`FuturesStore.bootstrap` is given a venue-reported
    position to adopt. That is a deliberate stop, not an error to route around.

``unreadable``
    a state file exists and could not be parsed. This is the worst case and it is
    treated as the worst case: the executor is unbootstrapped *and* the file is
    left exactly where it is, because overwriting the one record of what the
    account was doing is how a recoverable incident becomes an unrecoverable one.

Writes are atomic — temp file, ``fsync``, ``os.replace`` — so a crash midway
through a write leaves the previous state rather than half of the new one. Every
mutation the executor makes is followed by a write, which is what makes the
"filled but local completion not persisted" case recoverable: the fill's event id
is either in the file or it is not, and re-applying it is a no-op either way.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from chimera.futures.accounting import Ledger
from chimera.futures.domain import FuturesError, OrderRecord, Position

logger = logging.getLogger(__name__)

#: Bumped when the on-disk shape changes in a way an older reader would
#: misinterpret. A file from a schema this build does not know is refused, not
#: best-effort parsed.
STORE_SCHEMA = "chimera.futures-execution-state/1"


class StoreError(FuturesError):
    """The persisted state cannot be used, and using it anyway would be worse."""


class LoadOutcome(str, Enum):
    """How the store came to hold what it holds. A bounded telemetry label."""

    LOADED = "LOADED"
    MISSING = "MISSING"
    UNREADABLE = "UNREADABLE"


@dataclass
class FuturesState:
    """Everything one executor process must not lose across a restart."""

    positions: dict[str, Position] = field(default_factory=dict)
    orders: dict[str, OrderRecord] = field(default_factory=dict)
    ledger: Ledger = field(default_factory=Ledger)
    #: Why the last emergency flatten happened, and when. Persisted because an
    #: operator arriving after the fact needs to know the account was flattened
    #: on purpose rather than by something they have yet to find.
    flatten_reasons: list[dict[str, str]] = field(default_factory=list)
    #: Symbols whose local and reported state disagree, and how. Held per
    #: *symbol* rather than per order because a mismatch outlives the orders that
    #: caused it: a position can be disputed with every order already terminal,
    #: and a stop that only looked at open orders would let the next signal size
    #: itself against a position nobody can vouch for. Persisted, so a restart
    #: does not clear a dispute by forgetting it.
    disputed: dict[str, str] = field(default_factory=dict)
    #: False until the local state has been adopted or confirmed against the
    #: venue. Nothing may be planned while it is False.
    bootstrapped: bool = False

    def position(self, symbol: str) -> Position:
        return self.positions.get(symbol, Position(symbol=symbol))

    def set_position(self, position: Position) -> None:
        if position.is_flat:
            self.positions.pop(position.symbol, None)
        else:
            self.positions[position.symbol] = position

    def open_orders(self) -> list[OrderRecord]:
        return [o for o in self.orders.values() if not o.is_terminal]

    def to_dict(self) -> dict[str, Any]:
        return {
            "store_schema": STORE_SCHEMA,
            "bootstrapped": self.bootstrapped,
            "positions": {s: p.to_dict() for s, p in sorted(self.positions.items())},
            "orders": {k: o.to_dict() for k, o in sorted(self.orders.items())},
            "ledger": self.ledger.to_dict(),
            "flatten_reasons": list(self.flatten_reasons),
            "disputed": dict(sorted(self.disputed.items())),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FuturesState":
        schema = str(data.get("store_schema", ""))
        if schema != STORE_SCHEMA:
            raise StoreError(
                f"persisted futures state declares schema {schema!r}, not {STORE_SCHEMA!r}. "
                "A state file this build cannot read is not a state file it may guess at."
            )
        return cls(
            positions={
                s: Position.from_dict(p) for s, p in dict(data.get("positions", {})).items()
            },
            orders={
                k: OrderRecord.from_dict(o) for k, o in dict(data.get("orders", {})).items()
            },
            ledger=Ledger.from_dict(data.get("ledger", {})),
            flatten_reasons=[dict(r) for r in data.get("flatten_reasons", [])],
            disputed={str(k): str(v) for k, v in dict(data.get("disputed", {})).items()},
            bootstrapped=bool(data.get("bootstrapped", False)),
        )


@dataclass
class FuturesStore:
    """Load and save :class:`FuturesState`, atomically, and say which case it was."""

    path: Path | None
    state: FuturesState = field(default_factory=FuturesState)
    outcome: LoadOutcome = LoadOutcome.MISSING

    @classmethod
    def open(cls, path: str | Path | None) -> "FuturesStore":
        """Read the state file if there is one, and record how that went.

        An in-memory store (``path=None``) is for tests and for the deterministic
        replay protocol. It reports :attr:`LoadOutcome.MISSING`, so even in memory
        the executor still has to be bootstrapped explicitly rather than starting
        life believing it is flat.
        """
        if path is None:
            return cls(path=None, state=FuturesState(), outcome=LoadOutcome.MISSING)

        location = Path(path)
        if not location.exists():
            logger.warning(
                "No futures execution state at %s. Starting UNBOOTSTRAPPED: nothing will "
                "be planned until the venue-reported position has been adopted. An "
                "absent file is not evidence of a flat account.",
                location,
            )
            return cls(path=location, state=FuturesState(), outcome=LoadOutcome.MISSING)

        try:
            data = json.loads(location.read_text())
            state = FuturesState.from_dict(data)
        # ArithmeticError is in the tuple because the likeliest corruption of all
        # is a mangled number in a persisted field, and `Decimal("0.5O")` raises
        # `decimal.InvalidOperation` — whose MRO is DecimalException ->
        # ArithmeticError, never ValueError. Without it the one outcome this
        # module promises for an unreadable file escaped to the caller instead.
        except (OSError, ValueError, ArithmeticError, StoreError, KeyError) as exc:
            logger.critical(
                "Futures execution state at %s could not be read (%s). Starting "
                "UNBOOTSTRAPPED and leaving the file untouched: it is the only record "
                "of what this account was doing.",
                location,
                exc,
            )
            return cls(path=location, state=FuturesState(), outcome=LoadOutcome.UNREADABLE)

        return cls(path=location, state=state, outcome=LoadOutcome.LOADED)

    def save(self) -> None:
        """Write the state atomically. A crash leaves the previous file intact."""
        if self.path is None:
            return
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
            raise StoreError(
                f"could not persist futures execution state to {self.path}: {exc}. "
                "Continuing would mean the next restart cannot know what this process "
                "did."
            ) from exc

    def bootstrap(self, reported: Mapping[str, Position]) -> None:
        """Adopt the venue's positions as the starting local truth.

        Only legal while unbootstrapped and holding no local positions. Adopting a
        reported position over an existing local one is not bootstrapping, it is
        silently resolving a reconciliation mismatch in the venue's favour, and
        :mod:`chimera.futures.executor` is the only thing allowed to decide what
        happens to a mismatch — by refusing to trade through it.

        It is also illegal **after a state file that was unreadable rather than
        absent**. The refusal exists because the obvious call — ``recover({})``
        on a cold start — is byte-for-byte the same call in both cases, so
        without it the load path's careful decision to leave a corrupt file alone
        was undone one line later by this method's own ``save``.
        :meth:`adopt_after_unreadable` is the deliberate way through, and it
        takes a written reason and preserves the file.
        """
        if self.outcome is LoadOutcome.UNREADABLE:
            raise StoreError(
                f"the state file at {self.path} exists and could not be read, so an "
                "empty or venue-reported view is not something this process may adopt "
                "as fact — the account may hold a position that file was the only "
                "record of. An operator resolves this with adopt_after_unreadable(), "
                "which takes a reason and preserves the unreadable file."
            )
        if self.state.bootstrapped:
            raise StoreError(
                "already bootstrapped; adopting a reported state twice is not a no-op"
            )
        if self.state.positions:
            raise StoreError(
                "local positions already exist, so adopting the venue's view would "
                "overwrite them. That is reconciliation, and it is not this method's "
                "to decide."
            )
        for symbol, position in reported.items():
            if position.symbol != symbol:
                raise StoreError(
                    f"reported position for {symbol} is labelled {position.symbol}"
                )
            self.state.set_position(position)
        self.state.bootstrapped = True
        self.save()

    def adopt_after_unreadable(
        self, reported: Mapping[str, Position], note: str
    ) -> Path | None:
        """An operator's explicit decision to start over from an unreadable file.

        Moves the unreadable file aside — to ``<name>.corrupt`` — rather than
        overwriting it, because it is the only record of what the account was
        doing and a later investigation needs it. Returns where it was put.

        Requires a written reason for the same purpose
        :meth:`chimera.futures.executor.FuturesExecutor.resolve_reconciliation`
        does: the decision is a human one, and the log should say who made it and
        why rather than merely that state appeared.
        """
        if self.outcome is not LoadOutcome.UNREADABLE:
            raise StoreError(
                "adopt_after_unreadable() is for a state file that exists and could "
                f"not be read; this store loaded {self.outcome.value}"
            )
        if not note:
            raise StoreError(
                "adopting state after an unreadable file requires a stated reason"
            )

        preserved: Path | None = None
        if self.path is not None and self.path.exists():
            preserved = self.path.with_suffix(self.path.suffix + ".corrupt")
            self.path.replace(preserved)
        logger.critical(
            "Operator adopted a fresh futures state after an unreadable file. Reason: "
            "%s. The unreadable file was preserved at %s.",
            note,
            preserved,
        )
        self.outcome = LoadOutcome.MISSING
        self.state = FuturesState()
        self.bootstrap(reported)
        return preserved

    def record_flatten(self, symbol: str, reason: str, at: str) -> None:
        self.state.flatten_reasons.append({"symbol": symbol, "reason": reason, "at": at})
        self.save()
