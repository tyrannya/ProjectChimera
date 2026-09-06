"""Two-leg carry position for the demo runtime: accounting, ledger, hedge.

Re-exports only. The modules under this package own the behaviour:

* :mod:`chimera.carry.accounting` -- the pure ``Decimal`` formulas, ported from
  :mod:`nn.p13_carry` and proved equal to it on synthetic worlds by
  ``tests/test_carry_accounting_parity.py``.

Nothing in this package imports historical research code (section 16's tier
table), opens a socket, reads a credential, or constructs a venue outside
:mod:`chimera.carry.factory`.
"""

from __future__ import annotations

from chimera.carry.accounting import (
    ONE,
    TOUCH_MARK_CLOSE,
    TOUCH_MARK_HIGH,
    TOUCH_SOURCES,
    ZERO,
    Allocation,
    CarryError,
    CarryPosition,
    Costs,
    FundingSettlement,
    Quote,
    Venue,
    apply_funding,
    close_carry,
    hedge_quantity,
    is_liquidated,
    open_carry,
)

__all__ = [
    "ONE",
    "TOUCH_MARK_CLOSE",
    "TOUCH_MARK_HIGH",
    "TOUCH_SOURCES",
    "ZERO",
    "Allocation",
    "CarryError",
    "CarryPosition",
    "Costs",
    "FundingSettlement",
    "Quote",
    "Venue",
    "apply_funding",
    "close_carry",
    "hedge_quantity",
    "is_liquidated",
    "open_carry",
]
