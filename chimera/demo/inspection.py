"""Read-only persisted-state inspection for an unstarted demo runner.

The active demo objects cannot exist until :class:`RunnerClock` has observed a
recorded instant: constructing Aegis seeds equity and may persist a kill-switch
halt, and both futures executors carry a clock used by their persisted actions.
Operator ``status`` and the ledger-regression guard nevertheless have to inspect
what a previous process left behind *before* startup mutates or repairs anything.

This module is that deliberately smaller view.  It opens the existing state
files through their authoritative loaders and exposes only fields that are safe
to read.  It does not reconstruct a hedge, bootstrap a store, check a kill
switch, save a file, observe a clock, or manufacture an instant.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

from chimera.carry.factory import PERP_SYMBOL, SPOT_SYMBOL
from chimera.carry.hedge import HedgeState
from chimera.carry.ledger import CarryLedger
from chimera.demo.config import DemoConfig
from chimera.demo.risk_wiring import risk_limits
from chimera.futures.store import FuturesStore
from chimera.risk import RiskEngine, RiskState

if TYPE_CHECKING:  # pragma: no cover - imports used only by the active snapshot
    from chimera.carry.hedge import HedgedPosition


def _no_authoritative_time() -> float:
    """Fail if a supposedly read-only load ever starts asking for decision time."""
    raise RuntimeError(
        "read-only demo inspection has no authoritative time; start the runner "
        "from a recorded instant before performing a timed operation"
    )


@dataclass(frozen=True)
class DemoInspection:
    """The persisted fields ``status`` and pre-start recovery checks may read."""

    risk_state: RiskState
    hedge_state: HedgeState
    spot_quantity: Decimal
    perp_quantity: Decimal
    ledger: CarryLedger

    @property
    def imbalance(self) -> Decimal:
        return self.spot_quantity - self.perp_quantity

    def position_block(self) -> dict[str, str]:
        return {
            "hedge_state": self.hedge_state.value,
            "spot_qty": str(self.spot_quantity),
            "perp_qty": str(self.perp_quantity),
            "imbalance": str(self.imbalance),
        }

    @classmethod
    def from_active(cls, risk: RiskEngine, position: "HedgedPosition") -> "DemoInspection":
        """Take the same view from an already-started runtime, without mutation."""
        spot = position.leg("spot").quantity
        perp = position.leg("perp").quantity
        return cls(
            risk_state=risk.state,
            hedge_state=position.state,
            spot_quantity=spot,
            perp_quantity=perp,
            ledger=position.ledger,
        )


def inspect_demo_state(
    config: DemoConfig,
    *,
    capital: Decimal,
    state_dir: str | Path | None = None,
) -> DemoInspection:
    """Open the demo's persisted state without starting or changing it.

    ``RiskEngine`` remains the sole parser of ``risk.json``.  Omitting a kill
    switch path makes its constructor a persisted-state read only, and the
    raising clock makes a future accidental timed operation fail instead of
    falling back to host time.  The stores and carry ledger likewise retain the
    existing ``MISSING`` / ``LOADED`` / ``UNREADABLE`` meanings from their own
    ``open`` methods.

    ``hedge_state`` intentionally remains ``FLAT`` until active startup runs
    ``HedgedPosition.reconstruct``.  That preserves the documented status
    contract: quantities and imbalance are read from the stores, while the
    derived hedge state and disputes may be stale before reconstruction.
    """
    root = Path(state_dir if state_dir is not None else config.runner_setting("state_dir"))
    risk = RiskEngine(
        risk_limits(config.limits),
        state_path=root / "risk.json",
        clock=_no_authoritative_time,
    )
    spot_store = FuturesStore.open(root / "spot_store.json")
    perp_store = FuturesStore.open(root / "perp_store.json")
    ledger = CarryLedger.open(root / "carry_ledger.json", capital=Decimal(capital))
    return DemoInspection(
        risk_state=risk.state,
        hedge_state=HedgeState.FLAT,
        spot_quantity=spot_store.state.position(SPOT_SYMBOL).quantity,
        perp_quantity=perp_store.state.position(PERP_SYMBOL).quantity,
        ledger=ledger,
    )
