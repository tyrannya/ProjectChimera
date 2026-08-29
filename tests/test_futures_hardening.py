"""Six defects the futures test matrix found, and the fixes that closed them.

Each one was reachable, silent, and none of them was caught by the tests written
against the intended behaviour — which is the point of keeping them here rather
than folding them into the files that own each module. They fail in six
different ways and share one shape: something that should have refused missing or
impossible input accepted it and produced a number.
"""

from __future__ import annotations

import json
from decimal import Decimal

import pytest

from chimera.futures import (
    DeterministicFillModel,
    DryRunFuturesVenue,
    FuturesExecutor,
    FuturesStore,
    LoadOutcome,
    OrderIntent,
    OrderPurpose,
    OrderRecord,
    OrderSide,
    Position,
    PositionError,
    PositionSide,
    StoreError,
    TargetPosition,
    gross_exposure,
    load_constraint_source,
    net_exposure,
    unrealised_pnl,
)
from chimera.risk import RiskEngine, RiskLimits

SYMBOL = "BTC/USDT:USDT"
OTHER = "ETH/USDT:USDT"


def _executor(store: FuturesStore) -> FuturesExecutor:
    risk = RiskEngine(
        RiskLimits(
            max_position_pct=1.0,
            risk_per_trade_pct=0.5,
            max_total_exposure_pct=10.0,
            max_exposure_per_asset_pct=10.0,
        )
    )
    risk.update_equity(1_000_000.0)
    return FuturesExecutor(
        venue=DryRunFuturesVenue(
            source=load_constraint_source(), fill_model=DeterministicFillModel()
        ),
        risk=risk,
        store=store,
    )


# --- 1. an open position must have been entered at some price ---------------


def test_an_open_position_cannot_have_a_zero_entry_price():
    """`unrealised_pnl` priced a zero entry as the whole notional in profit.

    `(mark - 0) * quantity * sign` treats a missing entry as a free position, so a
    2 BTC LONG marked at 30000 reported 60000 of gain. `liquidation_price`
    already refused the same object, so the package contradicted itself about
    whether such a position could exist.
    """
    with pytest.raises(PositionError, match="entry_price"):
        Position(SYMBOL, PositionSide.LONG, Decimal("2"), Decimal("0"))
    with pytest.raises(PositionError, match="entry_price"):
        Position(SYMBOL, PositionSide.SHORT, Decimal("2"), Decimal("-5"))
    # The flat case is unaffected: a flat position has no entry price to have.
    assert Position(SYMBOL).entry_price == Decimal("0")


def test_the_position_that_used_to_fabricate_profit_can_no_longer_be_built():
    """Belt and braces: the consequence, not just the constructor."""
    good = Position(SYMBOL, PositionSide.LONG, Decimal("2"), Decimal("30000"))
    assert unrealised_pnl(good, Decimal("30000")) == Decimal("0")
    with pytest.raises(PositionError):
        unrealised_pnl(
            Position.from_dict(
                {
                    "symbol": SYMBOL,
                    "side": "LONG",
                    "quantity": "2",
                    "entry_price": "0",
                    "leverage": "1",
                    "margin_mode": "ISOLATED",
                }
            ),
            Decimal("30000"),
        )


# --- 2. exposure must fail closed on a missing price ------------------------


def test_exposure_refuses_to_report_a_position_it_has_no_price_for():
    """Skipping an unpriced position under-reports exposure exactly when it matters.

    A LONG of 1 at 10 plus an unpriced SHORT of 100 used to report 10 for both
    gross and net: the large position was simply invisible to whatever risk check
    read the number, and the under-report was largest when a symbol's feed was
    broken.
    """
    held = [
        Position(SYMBOL, PositionSide.LONG, Decimal("1"), Decimal("10")),
        Position(OTHER, PositionSide.SHORT, Decimal("100"), Decimal("10")),
    ]
    prices = {SYMBOL: Decimal("10")}
    for function in (gross_exposure, net_exposure):
        with pytest.raises(PositionError, match=OTHER):
            function(held, prices)


def test_exposure_is_correct_once_every_open_position_is_priced():
    held = [
        Position(SYMBOL, PositionSide.LONG, Decimal("1"), Decimal("10")),
        Position(OTHER, PositionSide.SHORT, Decimal("100"), Decimal("10")),
    ]
    prices = {SYMBOL: Decimal("10"), OTHER: Decimal("10")}
    assert gross_exposure(held, prices) == Decimal("1010")
    assert net_exposure(held, prices) == Decimal("-990")


def test_a_flat_position_needs_no_price():
    """Flat holds nothing, so a missing price cannot hide anything."""
    assert gross_exposure([Position(SYMBOL)], {}) == Decimal("0")
    assert net_exposure([Position(SYMBOL)], {}) == Decimal("0")


# --- 3. an order cannot fill more than it asked for -------------------------


def _record(quantity: str) -> OrderRecord:
    return OrderRecord(
        order_id="ORD-1",
        intent=OrderIntent(
            symbol=SYMBOL,
            side=OrderSide.BUY,
            quantity=Decimal(quantity),
            purpose=OrderPurpose.OPEN,
            reduce_only=False,
            position_side=PositionSide.LONG,
        ),
    )


def test_an_over_delivered_fill_is_refused_rather_than_producing_a_negative_remainder():
    """`remaining_quantity` going negative is a negative order size downstream."""
    record = _record("1")
    record.book_fill(Decimal("0.6"), Decimal("60000"), Decimal("0"))
    assert record.remaining_quantity == Decimal("0.4")
    with pytest.raises(PositionError, match="over-delivers"):
        record.book_fill(Decimal("0.5"), Decimal("60000"), Decimal("0"))
    assert record.over_delivered is True
    assert record.remaining_quantity == Decimal("0.4"), "the refused fill booked nothing"


def test_remaining_quantity_is_never_negative_even_if_a_record_is_loaded_over_filled():
    """A persisted record from an older build must not resurrect the negative."""
    record = OrderRecord.from_dict(
        {
            **_record("1").to_dict(),
            "filled_quantity": "3",
        }
    )
    assert record.remaining_quantity == Decimal("0")


def test_book_fill_volume_weights_the_average_price():
    record = _record("2")
    record.book_fill(Decimal("1"), Decimal("30000"), Decimal("1"))
    record.book_fill(Decimal("1"), Decimal("34000"), Decimal("2"))
    assert record.average_price == Decimal("32000")
    assert record.fees == Decimal("3")


# --- 4. an unreadable state file may not be adopted by accident -------------


def test_recover_refuses_to_adopt_anything_after_an_unreadable_state_file(tmp_path):
    """The worst-case path used to be undone one line after it was taken.

    `FuturesStore.open` deliberately leaves a corrupt file alone. `recover({})` —
    byte-for-byte the same call an ordinary cold start makes — then adopted an
    empty view as a flat account and `bootstrap`'s own `save` overwrote the file.
    """
    path = tmp_path / "state.json"
    path.write_text('{"store_schema": "chimera.futures-execution-state/1", "positions": ')
    original = path.read_bytes()

    store = FuturesStore.open(path)
    assert store.outcome is LoadOutcome.UNREADABLE

    executor = _executor(store)
    assert executor.recover({}) is None
    assert store.state.bootstrapped is False
    assert path.read_bytes() == original, "the unreadable file was overwritten"

    with pytest.raises(Exception) as excinfo:
        executor.execute_target(
            TargetPosition(SYMBOL, PositionSide.LONG, Decimal("0.01")),
            Decimal("60000"),
            equity=1_000_000.0,
        )
    assert type(excinfo.value).__name__ == "NotBootstrapped"


def test_bootstrap_itself_refuses_after_an_unreadable_file(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{not json")
    store = FuturesStore.open(path)
    with pytest.raises(StoreError, match="adopt_after_unreadable"):
        store.bootstrap({})


def test_an_operator_can_adopt_after_an_unreadable_file_and_the_original_is_kept(tmp_path):
    """The deliberate way through: a written reason, and the file preserved."""
    path = tmp_path / "state.json"
    path.write_text("{not json")
    original = path.read_bytes()
    store = FuturesStore.open(path)

    with pytest.raises(StoreError, match="stated reason"):
        store.adopt_after_unreadable({}, "")

    preserved = store.adopt_after_unreadable(
        {SYMBOL: Position(SYMBOL, PositionSide.LONG, Decimal("0.5"), Decimal("60000"))},
        "operator checked the venue by hand; the account holds 0.5 BTC long",
    )
    assert preserved is not None and preserved.read_bytes() == original
    assert store.state.bootstrapped is True
    assert store.state.position(SYMBOL).quantity == Decimal("0.5")
    assert json.loads(path.read_text())["bootstrapped"] is True


def test_adopt_after_unreadable_refuses_a_store_that_loaded_cleanly(tmp_path):
    store = FuturesStore.open(tmp_path / "absent.json")
    with pytest.raises(StoreError, match="MISSING"):
        store.adopt_after_unreadable({}, "no reason should get this far")


# --- 5. a mangled number is the likeliest corruption of all -----------------


def test_a_mangled_decimal_in_a_persisted_field_reports_unreadable(tmp_path):
    """`Decimal("0.5O")` raises InvalidOperation, whose MRO never reaches ValueError.

    So the one outcome the store promises for a file that exists and cannot be
    parsed used to escape to the caller instead.
    """
    path = tmp_path / "state.json"
    path.write_text(
        json.dumps(
            {
                "store_schema": "chimera.futures-execution-state/1",
                "bootstrapped": True,
                "positions": {
                    SYMBOL: {
                        "symbol": SYMBOL,
                        "side": "LONG",
                        "quantity": "0.5O",
                        "entry_price": "60000",
                        "leverage": "1",
                        "margin_mode": "ISOLATED",
                    }
                },
                "orders": {},
                "ledger": {},
                "flatten_reasons": [],
                "disputed": {},
            }
        )
    )
    original = path.read_bytes()
    store = FuturesStore.open(path)
    assert store.outcome is LoadOutcome.UNREADABLE
    assert store.state.bootstrapped is False
    assert path.read_bytes() == original


# --- 6. a closed position must not leave its gauge open ---------------------


def test_closing_a_position_zeroes_its_quantity_gauge():
    """A panel reading a stale gauge reports an open position on a flat account."""
    from chimera import metrics

    if not metrics.PROMETHEUS_AVAILABLE:  # pragma: no cover - environment dependent
        pytest.skip("prometheus_client is not installed")

    executor = _executor(FuturesStore.open(None))
    executor.recover({})
    executor.execute_target(
        TargetPosition(SYMBOL, PositionSide.LONG, Decimal("0.5")),
        Decimal("60000"),
        equity=1_000_000.0,
    )
    gauge = metrics.FUT_POSITION_QUANTITY.labels(symbol=SYMBOL, side="LONG")
    assert gauge._value.get() == pytest.approx(0.5)

    executor.execute_target(TargetPosition.flat(SYMBOL), Decimal("60000"), equity=1_000_000.0)
    assert executor.position(SYMBOL).is_flat
    assert gauge._value.get() == pytest.approx(0.0)
