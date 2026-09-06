"""``build_hedged_position``: the ONLY demo-path constructor of venues.

Section 7.6 of the adopted master plan proves architecturally that no position
increase can bypass Aegis, and step 3 of that proof is that "the demo runner
constructs venues only inside ``HedgedPosition``'s factory". This module is
that factory, and ``tests/test_demo_no_live_path.py`` is what holds it to it:
no other module under ``chimera/carry/`` or ``chimera/demo/`` may import
:mod:`chimera.futures.venue`.

**No sockets, no credentials, no new Venue class.** Both legs are
:class:`~chimera.futures.venue.DryRunFuturesVenue` instances -- the same
in-process venue the frozen dry-run protocol already exercises -- priced by the
already-merged :class:`~chimera.futures.fills.RecordedQuoteFillModel` from
PR-07. There is no live venue to select and no configuration switch that could
reach one.

**Spot constraints are declared, not fetched.** ``chimera/futures/venue.py``
publishes the perpetual's committed table; the spot leg gets its own entry
here, built through :meth:`SymbolConstraints.from_dict` so it passes the same
fail-closed validation. The spot fee is ``0.001`` and the perpetual's is
``0.0005`` (section 6.5), and the coarser of the two step sizes is what the
common hedge quantity is floored to.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

from chimera.carry.hedge import HedgeConfig, HedgedPosition
from chimera.carry.ledger import CarryLedger
from chimera.futures.executor import FuturesExecutionConfig, FuturesExecutor
from chimera.futures.fills import RecordedQuoteFillModel
from chimera.futures.store import FuturesStore
from chimera.futures.venue import (
    DryRunFuturesVenue,
    StaticConstraintSource,
    default_constraints_table,
)
from chimera.risk import RiskEngine

#: Binance spot BTCUSDT. Declared rather than fetched, for the reason
#: `default_constraints_table` gives about the perpetual: an order planned here
#: is planned against numbers a reviewer can read.
#:
#: The values are section 5.3's spot row, taken exactly as adopted:
#: ``step_size 0.00001``, ``min_notional 10``, ``taker_fee_rate 0.001``,
#: ``maker_fee_rate 0.001``, ``supported_position_sides {"LONG"}`` and
#: ``maintenance_margin_rate 0.004`` -- the last unused at 1x LONG and kept only
#: because the constraints parser requires a valid one.
#:
#: `supports_reduce_only` is left at the parser's default of True, and that is
#: load-bearing rather than incidental: `FuturesExecutor.emergency_flatten`
#: closes a leg with a reduce-only order, so a spot entry declaring False makes
#: the spot leg impossible to flatten -- every hedge correction and every
#: emergency reduction would end in a reconciliation dispute instead. The flag
#: is also true in substance on spot: a SELL of a held inventory cannot reverse
#: into a short, which is exactly the guarantee reduce-only exists to give.
SPOT_SYMBOL = "BTC/USDT"
PERP_SYMBOL = "BTC/USDT:USDT"

SPOT_CONSTRAINTS: dict[str, Any] = {
    "status": "TRADING",
    "tick_size": "0.01",
    "step_size": "0.00001",
    "quantity_precision": 5,
    "price_precision": 2,
    "min_quantity": "0.00001",
    "min_notional": "10",
    "maintenance_margin_rate": "0.004",
    "taker_fee_rate": "0.001",
    "maker_fee_rate": "0.001",
    "supported_order_types": ["MARKET"],
    "supported_position_sides": ["LONG"],
}


def demo_constraints_table() -> dict[str, dict[str, Any]]:
    """Both legs' constraints: the committed perpetual table plus spot."""
    table = dict(default_constraints_table())
    table[SPOT_SYMBOL] = dict(SPOT_CONSTRAINTS)
    return table


def common_step_size(table: Mapping[str, Mapping[str, Any]] | None = None) -> Decimal:
    """The coarser of the two legs' step sizes.

    The hedge holds one quantity on both legs, so the quantity has to be
    representable on both. Flooring to the coarser step is what makes the two
    legs equal by construction rather than equal up to a rounding the finer
    venue would have accepted.
    """
    entries = table or demo_constraints_table()
    return max(
        Decimal(str(entries[symbol]["step_size"])) for symbol in (SPOT_SYMBOL, PERP_SYMBOL)
    )


def _venue(
    source: StaticConstraintSource, model: RecordedQuoteFillModel, store: FuturesStore
) -> DryRunFuturesVenue:
    """One leg's simulated venue, with its view restored from the leg's store.

    Section 6.8 records the fact this exists for: **the venue is in-memory.** It
    holds the simulated account's positions in a dict that a new process starts
    empty, while the store on disk still holds what the account was left
    holding. Section 8.1 then reconciles the two every sixty minutes, and
    `FuturesExecutor.reconcile` asks the venue -- so without this the first
    reconciliation after any restart compares a held position against an empty
    simulator, reports MISMATCH, disputes both legs and halts the campaign. A
    restart is a normal event this design expects (section 2.1), so a build that
    could not survive one would have no campaign at all.

    What is restored is only what this process's predecessor already persisted,
    so the reconciliation it makes possible is still a real check: within a
    process the venue applies fills to its own position as it reports them
    (`DryRunFuturesVenue.submit`), and a fill the executor dropped still shows up
    as a disagreement. Nothing is adopted INTO the executor here -- that is
    `recover()`'s job, and section 6.8 has it read the stores for the same
    reason.
    """
    venue = DryRunFuturesVenue(source=source, fill_model=model)
    for symbol, position in store.state.positions.items():
        venue.apply_settlement(symbol, position)
    return venue


def build_hedged_position(
    *,
    risk: RiskEngine,
    capital: Decimal,
    state_dir: str | Path | None = None,
    config: HedgeConfig | None = None,
    fill_model: RecordedQuoteFillModel | None = None,
    constraints: Mapping[str, Mapping[str, Any]] | None = None,
) -> HedgedPosition:
    """Build both legs, their stores and the ledger. The only demo venue path.

    ``state_dir`` of ``None`` builds an in-memory position, which is what the
    tests and the replay protocol use; a path gives each leg its own store file
    and the ledger its own, all under that directory.

    One :class:`RecordedQuoteFillModel` instance is shared by both legs on
    purpose: the runner installs the decision minute's book once and every leg
    of that minute is priced from it, and the model's clock must be advanced
    every cycle so a frozen quote can age out.
    """
    source = StaticConstraintSource.from_mapping(constraints or demo_constraints_table())
    model = fill_model if fill_model is not None else RecordedQuoteFillModel()

    root = Path(state_dir) if state_dir is not None else None
    spot_store = FuturesStore.open(root / "spot_store.json" if root else None)
    perp_store = FuturesStore.open(root / "perp_store.json" if root else None)
    ledger = CarryLedger.open(root / "carry_ledger.json" if root else None, capital=capital)

    execution = FuturesExecutionConfig(dry_run=True, leverage=Decimal("1"))
    spot = FuturesExecutor(
        venue=_venue(source, model, spot_store),
        risk=risk,
        store=spot_store,
        config=execution,
    )
    perp = FuturesExecutor(
        venue=_venue(source, model, perp_store),
        risk=risk,
        store=perp_store,
        config=execution,
    )
    return HedgedPosition(
        spot=spot,
        perp=perp,
        risk=risk,
        ledger=ledger,
        config=config or HedgeConfig(spot_symbol=SPOT_SYMBOL, perp_symbol=PERP_SYMBOL),
    )
