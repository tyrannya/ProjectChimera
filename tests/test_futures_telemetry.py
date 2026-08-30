"""The Argus half of Futures Execution v1: what the package tells Prometheus.

Two failures this file exists to catch, and they pull in opposite directions.

*A metric that nothing moves.* A ``FUT_*`` series the executor never writes is a
panel that reads as healthy because it is empty. So the tests here do not
inspect the emitting code — they drive a real open, close, partial fill, funding
settlement, veto, flatten, invalid transition, reconciliation and restart
through :class:`chimera.futures.FuturesExecutor` and read the series before and
after. A deleted ``.inc()`` fails a delta, not a docstring. Sixteen of the
twenty-one series are driven that way. The other five — ``FUT_SIGNALS``,
``FUT_ORDERS_REJECTED``, ``FUT_REALISED_PNL``, ``FUT_NET_PNL`` and
``FUT_DRAWDOWN`` — are only pinned structurally here and are written by paths
this file does not drive, so deleting the write would not fail anything: that is
a gap, not a decision.

*A label whose value set grows with traffic.* Prometheus keeps a time series per
distinct label combination, forever, so a free-text ``reason`` on a counter is an
unbounded memory leak wearing a metric's clothes. The label names are pinned in
:data:`EXPECTED_LABELS`, every label domain is asserted to be a bounded enum, and
the Aegis veto test checks the *value* that actually reaches the label: a short
collapsed token, never the human-readable reason the record keeps.

Values are read the way ``tests/test_observability.py`` reads them — off the
metric object (``._value.get()``, ``._sum``/``._buckets`` for histograms) — not
through the registry, so there is one technique in the suite rather than two.
The declarations are read from the module's own source, so the structural checks
still run on a machine where ``prometheus_client`` is absent and
:mod:`chimera.metrics` is nothing but stubs.

Everything runs in-process: an in-memory store, a dry-run venue, no credentials,
no environment and no network.
"""

from __future__ import annotations

import ast
from decimal import Decimal
from pathlib import Path

import pytest

from chimera import metrics
from chimera.futures import (
    DeterministicFillModel,
    DryRunFuturesVenue,
    EventKind,
    FlattenCause,
    FundingEvent,
    FuturesExecutor,
    FuturesStore,
    InvalidTransition,
    LoadOutcome,
    OrderEvent,
    OrderState,
    Position,
    PositionSide,
    ReconciliationOutcome,
    TargetPosition,
    load_constraint_source,
)
from chimera.risk import RiskEngine, RiskLimits
from chimera.safety import _SECRET_HINTS

SYMBOL = "BTC/USDT:USDT"
PRICE = Decimal("60000")
QUANTITY = Decimal("0.5")
EQUITY = 100_000.0

requires_prometheus = pytest.mark.skipif(
    not metrics.PROMETHEUS_AVAILABLE, reason="prometheus_client is not installed"
)

#: Every ``FUT_*`` series and the labels it is allowed to carry. Written out
#: rather than read from the objects so that adding a metric, or adding a label
#: to one, has to be a deliberate edit here as well.
EXPECTED_LABELS = {
    "FUT_SIGNALS": ("outcome",),
    "FUT_RISK_VETOES": ("reason",),
    "FUT_ORDERS_PLANNED": ("side", "purpose"),
    "FUT_ORDERS_SUBMITTED": ("side", "purpose"),
    "FUT_ORDERS_REJECTED": ("reason",),
    "FUT_FILLS": ("side", "kind"),
    "FUT_SLIPPAGE_BPS": (),
    "FUT_TRADING_FEES": (),
    "FUT_FUNDING": ("direction",),
    "FUT_TURNOVER": (),
    "FUT_POSITION_QUANTITY": ("symbol", "side"),
    "FUT_GROSS_EXPOSURE": (),
    "FUT_NET_EXPOSURE": (),
    "FUT_REALISED_PNL": (),
    "FUT_NET_PNL": (),
    "FUT_DRAWDOWN": (),
    "FUT_RECONCILIATION": ("outcome",),
    "FUT_INVALID_TRANSITIONS": ("from_state",),
    "FUT_EMERGENCY_FLATTEN": ("cause",),
    "FUT_RECOVERY": ("outcome",),
    "FUT_EXECUTION_LATENCY": (),
}


# --- reading the declarations ---------------------------------------------
def _declared_string(node):
    """The string a name or help argument in ``metrics.py`` evaluates to.

    The names are f-strings interpolating ``_PREFIX``; nothing else is
    interpolated, and an argument that did something cleverer would fail here
    rather than be silently skipped by the structural tests.
    """
    if isinstance(node, ast.Constant):
        return str(node.value)
    assert isinstance(node, ast.JoinedStr), f"unreadable metric argument: {ast.dump(node)}"
    parts = []
    for piece in node.values:
        if isinstance(piece, ast.Constant):
            parts.append(str(piece.value))
            continue
        assert isinstance(piece.value, ast.Name) and piece.value.id == "_PREFIX", (
            "a futures metric name interpolates something other than _PREFIX, so its "
            "series name cannot be read without importing prometheus_client"
        )
        parts.append(metrics._PREFIX)
    return "".join(parts)


def declared_futures_metrics():
    """``{attribute: {kind, name, documentation, labels}}`` for every ``FUT_*``.

    Read from the source, not the objects, so the structural assertions hold up
    on a machine where ``prometheus_client`` is missing and every metric in the
    module is a no-op stub with no name to inspect.
    """
    tree = ast.parse(Path(metrics.__file__).read_text())
    declared = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or not target.id.startswith("FUT_"):
            continue
        call = node.value
        # Labels are positional in this module today, but ``labelnames=[...]`` is
        # the same declaration to prometheus_client. Reading only the positional
        # form would report such a metric as unlabelled and wave an unbounded
        # label through the cardinality review — silently, on the machine with no
        # prometheus_client where these are the only tests that run. Any other
        # keyword fails here rather than being read as "no labels".
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        unreadable = set(keywords) - {"buckets", "labelnames"}
        assert not unreadable, (
            f"{target.id} is declared with {sorted(map(str, unreadable))}, which this "
            "reader does not understand, so its labels cannot be reviewed"
        )
        if "labelnames" in keywords:
            labels = tuple(ast.literal_eval(keywords["labelnames"]))
        elif len(call.args) > 2:
            labels = tuple(ast.literal_eval(call.args[2]))
        else:
            labels = ()
        declared[target.id] = {
            "kind": call.func.id,
            "name": _declared_string(call.args[0]),
            "documentation": _declared_string(call.args[1]),
            "labels": labels,
        }
    return declared


# --- reading the values ----------------------------------------------------
def counter_value(metric, **labels):
    """One counter series, creating the child at zero if it has never fired."""
    return (metric.labels(**labels) if labels else metric)._value.get()


def gauge_value(metric, **labels):
    """One gauge series, creating the child at zero if it has never been set."""
    return (metric.labels(**labels) if labels else metric)._value.get()


def histogram_sum(metric):
    return metric._sum.get()


def histogram_count(metric):
    return sum(bucket.get() for bucket in metric._buckets)


def label_values(metric):
    """Every label tuple this counter has ever been given a value for."""
    return set(metric._metrics)


# --- driving the executor --------------------------------------------------
def make_executor(*, store=None, fill_model=None):
    """A bootstrapped executor over a flat dry-run venue.

    The limits are widened only so far as it takes for a 0.5 BTC order at 60000
    to pass the gate; the veto test narrows them again by halting.
    """
    risk = RiskEngine(RiskLimits(max_position_pct=1.0, risk_per_trade_pct=0.5))
    risk.update_equity(EQUITY)
    executor = FuturesExecutor(
        venue=DryRunFuturesVenue(
            source=load_constraint_source(),
            fill_model=fill_model or DeterministicFillModel(),
        ),
        risk=risk,
        store=store if store is not None else FuturesStore.open(None),
    )
    executor.recover({})
    return executor


def move_to(executor, side, quantity=QUANTITY):
    return executor.execute_target(
        TargetPosition(SYMBOL, side, quantity), PRICE, equity=EQUITY
    )


# --- what is declared ------------------------------------------------------
def test_every_declared_futures_metric_is_also_a_live_metric_object():
    """Guards the source reader: a metric it cannot see is one it cannot check."""
    assert set(declared_futures_metrics()) == {a for a in dir(metrics) if a.startswith("FUT_")}


def test_the_declared_futures_metrics_are_exactly_the_ones_under_cardinality_review():
    """A new FUT_ metric must be added to EXPECTED_LABELS, not just to the module."""
    assert set(declared_futures_metrics()) == set(EXPECTED_LABELS)


def test_every_futures_metric_is_named_under_the_chimera_futures_prefix():
    """A futures series named outside the family is one no dashboard will find."""
    for attribute, declaration in declared_futures_metrics().items():
        assert declaration["name"].startswith(
            "chimera_futures_"
        ), f"{attribute} is exported as {declaration['name']!r}"


def test_every_futures_metric_carries_a_non_empty_documentation_string():
    """HELP text is what a reader gets instead of guessing from the series name."""
    for attribute, declaration in declared_futures_metrics().items():
        assert declaration["documentation"].strip(), f"{attribute} has no help text"


def test_every_futures_metric_declares_exactly_the_expected_label_names():
    """A label added without review is a new time series per distinct value."""
    for attribute, declaration in declared_futures_metrics().items():
        assert (
            declaration["labels"] == EXPECTED_LABELS[attribute]
        ), f"{attribute} labels changed to {declaration['labels']}"


def test_no_futures_metric_name_or_help_text_contains_a_credential_shaped_token():
    """Telemetry is scraped and stored; a secret that reaches it has leaked."""
    for attribute, declaration in declared_futures_metrics().items():
        blob = f"{declaration['name']} {declaration['documentation']}".upper()
        for hint in _SECRET_HINTS:
            assert hint not in blob, f"{attribute} mentions {hint!r} in its telemetry text"


# --- label cardinality in practice -----------------------------------------
@requires_prometheus
def test_an_aegis_veto_is_labelled_with_a_collapsed_token_not_the_human_reason():
    """The reason is unbounded free text; the label it collapses to must not be.

    A veto reason names a number ("drawdown 21.5% exceeds ..."), so putting it on
    a label would mint a fresh time series for every basis point the account ever
    drew down. The record keeps the whole sentence; the metric gets "halted".
    """
    executor = make_executor()
    executor.risk.halt("drawdown 21.5% exceeds the configured 15.0%")

    before = counter_value(metrics.FUT_RISK_VETOES, reason="halted")
    records = move_to(executor, PositionSide.LONG)

    assert len(records) == 1
    assert records[0].state is OrderState.REJECTED
    assert records[0].reason == "halted: drawdown 21.5% exceeds the configured 15.0%"
    assert counter_value(metrics.FUT_RISK_VETOES, reason="halted") == before + 1

    for (reason,) in label_values(metrics.FUT_RISK_VETOES):
        assert " " not in reason, f"veto label {reason!r} is a sentence, not a token"
        assert len(reason) <= 24, f"veto label {reason!r} is too long to be collapsed"
        assert "21.5%" not in reason


# --- the open -> close cycle -----------------------------------------------
@requires_prometheus
def test_a_long_open_and_close_cycle_moves_each_execution_counter_by_an_exact_delta():
    """One BUY leg and one SELL leg, each planned, submitted and filled once.

    The fees and turnover are the arithmetic of the committed BTCUSDT table at a
    5bp adverse fill: 0.5 at 60030 then 0.5 at 59970, taker 0.0005. An off-by-one
    in the event loop, or a fill booked twice, moves one of these numbers.
    """
    executor = make_executor()
    before = {
        "planned_buy": counter_value(metrics.FUT_ORDERS_PLANNED, side="BUY", purpose="OPEN"),
        "planned_sell": counter_value(
            metrics.FUT_ORDERS_PLANNED, side="SELL", purpose="CLOSE"
        ),
        "submitted_buy": counter_value(
            metrics.FUT_ORDERS_SUBMITTED, side="BUY", purpose="OPEN"
        ),
        "submitted_sell": counter_value(
            metrics.FUT_ORDERS_SUBMITTED, side="SELL", purpose="CLOSE"
        ),
        "fills_buy": counter_value(metrics.FUT_FILLS, side="BUY", kind="full"),
        "fills_sell": counter_value(metrics.FUT_FILLS, side="SELL", kind="full"),
        "fees": counter_value(metrics.FUT_TRADING_FEES),
        "turnover": counter_value(metrics.FUT_TURNOVER),
    }

    opened = move_to(executor, PositionSide.LONG)
    closed = move_to(executor, PositionSide.FLAT, Decimal("0"))
    assert [r.state for r in opened + closed] == [OrderState.FILLED, OrderState.FILLED]

    assert (
        counter_value(metrics.FUT_ORDERS_PLANNED, side="BUY", purpose="OPEN")
        == before["planned_buy"] + 1
    )
    assert (
        counter_value(metrics.FUT_ORDERS_PLANNED, side="SELL", purpose="CLOSE")
        == before["planned_sell"] + 1
    )
    assert (
        counter_value(metrics.FUT_ORDERS_SUBMITTED, side="BUY", purpose="OPEN")
        == before["submitted_buy"] + 1
    )
    assert (
        counter_value(metrics.FUT_ORDERS_SUBMITTED, side="SELL", purpose="CLOSE")
        == before["submitted_sell"] + 1
    )
    assert counter_value(metrics.FUT_FILLS, side="BUY", kind="full") == before["fills_buy"] + 1
    assert (
        counter_value(metrics.FUT_FILLS, side="SELL", kind="full") == before["fills_sell"] + 1
    )
    assert counter_value(metrics.FUT_TRADING_FEES) == pytest.approx(before["fees"] + 30.0)
    assert counter_value(metrics.FUT_TURNOVER) == pytest.approx(before["turnover"] + 60_000.0)


@requires_prometheus
def test_a_long_position_sets_positive_net_exposure_and_closing_returns_both_gauges_to_zero():
    """Gauges are set, not incremented: a stale exposure reading is a wrong one."""
    executor = make_executor()

    move_to(executor, PositionSide.LONG)
    assert gauge_value(metrics.FUT_GROSS_EXPOSURE) == 30_000.0
    assert gauge_value(metrics.FUT_NET_EXPOSURE) == 30_000.0

    move_to(executor, PositionSide.FLAT, Decimal("0"))
    assert gauge_value(metrics.FUT_GROSS_EXPOSURE) == 0.0
    assert gauge_value(metrics.FUT_NET_EXPOSURE) == 0.0


@requires_prometheus
def test_a_short_position_sets_net_exposure_negative_while_gross_stays_positive():
    """SHORT is a side, so net carries its sign and gross carries its magnitude.

    A net exposure that came out positive for a SHORT would read on the dashboard
    as twice the directional risk the account is actually running.
    """
    executor = make_executor()
    move_to(executor, PositionSide.SHORT)

    assert gauge_value(metrics.FUT_NET_EXPOSURE) == -30_000.0
    assert gauge_value(metrics.FUT_GROSS_EXPOSURE) == 30_000.0


@requires_prometheus
def test_an_open_position_publishes_its_size_on_its_own_side_and_zero_on_the_other():
    """One symbol is two series, and the side it is not on has to read zero.

    Both series are set to a sentinel first: the gauge is process-global and an
    earlier test in the same session leaves a real position size on it, so
    reading 0.5 proves nothing unless the value this open wrote is the only way
    the sentinel could have been replaced.
    """
    executor = make_executor()
    for side in (PositionSide.LONG, PositionSide.SHORT):
        metrics.FUT_POSITION_QUANTITY.labels(symbol=SYMBOL, side=side.value).set(-1.0)

    move_to(executor, PositionSide.LONG)

    assert gauge_value(
        metrics.FUT_POSITION_QUANTITY, symbol=SYMBOL, side=PositionSide.LONG.value
    ) == float(QUANTITY)
    assert (
        gauge_value(
            metrics.FUT_POSITION_QUANTITY, symbol=SYMBOL, side=PositionSide.SHORT.value
        )
        == 0.0
    )


@requires_prometheus
def test_closing_a_position_returns_its_quantity_gauge_to_zero():
    """The gauge a dashboard reads per symbol, on an account that holds nothing.

    ``FUT_GROSS_EXPOSURE`` is accumulated from scratch on every publish and does
    return to zero, which is what makes this survivable in review: the two
    panels disagree, and only the exposure one is right.
    """
    executor = make_executor()
    move_to(executor, PositionSide.LONG)
    move_to(executor, PositionSide.FLAT, Decimal("0"))

    assert gauge_value(metrics.FUT_GROSS_EXPOSURE) == 0.0
    assert (
        gauge_value(metrics.FUT_POSITION_QUANTITY, symbol=SYMBOL, side=PositionSide.LONG.value)
        == 0.0
    )


@requires_prometheus
def test_a_partially_filled_order_is_counted_under_the_partial_fill_kind():
    """Partial and final fills are separate label values, not one fill count."""
    executor = make_executor(fill_model=DeterministicFillModel(max_fill_ratio=Decimal("0.5")))
    before_partial = counter_value(metrics.FUT_FILLS, side="BUY", kind="partial")
    before_full = counter_value(metrics.FUT_FILLS, side="BUY", kind="full")

    records = move_to(executor, PositionSide.LONG)

    assert records[0].state is OrderState.FILLED
    assert counter_value(metrics.FUT_FILLS, side="BUY", kind="partial") == before_partial + 1
    assert counter_value(metrics.FUT_FILLS, side="BUY", kind="full") == before_full + 1


@requires_prometheus
def test_slippage_is_observed_as_an_adverse_positive_distance_for_a_buy_and_a_sell():
    """Adverse is positive on both sides, or a SELL's cost cancels a BUY's.

    The fill model is 5bp against the order either way: a BUY fills above the
    reference and a SELL below it. Recording the raw signed difference instead
    would put -5 in the histogram for one of the two legs and report a strategy
    that pays no spread at all.
    """
    executor = make_executor()

    before_sum = histogram_sum(metrics.FUT_SLIPPAGE_BPS)
    before_count = histogram_count(metrics.FUT_SLIPPAGE_BPS)
    move_to(executor, PositionSide.LONG)
    buy_sum = histogram_sum(metrics.FUT_SLIPPAGE_BPS)
    assert histogram_count(metrics.FUT_SLIPPAGE_BPS) == before_count + 1
    assert buy_sum - before_sum == pytest.approx(5.0)

    move_to(executor, PositionSide.FLAT, Decimal("0"))
    assert histogram_count(metrics.FUT_SLIPPAGE_BPS) == before_count + 2
    assert histogram_sum(metrics.FUT_SLIPPAGE_BPS) - buy_sum == pytest.approx(5.0)


@requires_prometheus
def test_funding_paid_and_funding_received_land_on_separate_labels_and_are_not_netted():
    """Paying 3 and receiving 6 is not the same account as receiving 3.

    A single net counter would report both as +3 and hide that the market is
    charging this position to be held.
    """
    executor = make_executor()
    move_to(executor, PositionSide.LONG)
    before_paid = counter_value(metrics.FUT_FUNDING, direction="paid")
    before_received = counter_value(metrics.FUT_FUNDING, direction="received")

    paid = executor.settle_funding(FundingEvent(SYMBOL, Decimal("0.0001"), PRICE, "settle-1"))
    received = executor.settle_funding(
        FundingEvent(SYMBOL, Decimal("-0.0002"), PRICE, "settle-2")
    )

    assert paid == Decimal("-3.00000")
    assert received == Decimal("6.00000")
    assert counter_value(metrics.FUT_FUNDING, direction="paid") == pytest.approx(
        before_paid + 3.0
    )
    assert counter_value(metrics.FUT_FUNDING, direction="received") == pytest.approx(
        before_received + 6.0
    )


# --- the exceptional paths -------------------------------------------------
@requires_prometheus
def test_an_emergency_flatten_is_counted_under_its_exact_cause_and_only_those_causes():
    """Why the account was flattened is the first thing an operator needs.

    The cause label is bounded by the enum, so a flatten reached by a path that
    invented its own reason string would show up here as a label value that is
    not a FlattenCause.
    """
    executor = make_executor()
    for cause in FlattenCause:
        before = counter_value(metrics.FUT_EMERGENCY_FLATTEN, cause=cause.value)
        executor.emergency_flatten(SYMBOL, cause, PRICE)
        assert (
            counter_value(metrics.FUT_EMERGENCY_FLATTEN, cause=cause.value) == before + 1
        ), f"{cause.value} did not increment its own series"

    observed = {values[0] for values in label_values(metrics.FUT_EMERGENCY_FLATTEN)}
    assert observed == {cause.value for cause in FlattenCause}


@requires_prometheus
def test_an_invalid_transition_is_counted_against_the_state_it_was_refused_from():
    """The refused-from state is the diagnosis; the order id would be unbounded."""
    executor = make_executor()
    record = move_to(executor, PositionSide.LONG)[0]
    assert record.state is OrderState.FILLED

    before = counter_value(metrics.FUT_INVALID_TRANSITIONS, from_state="FILLED")
    with pytest.raises(InvalidTransition, match="FILLED -> ACKNOWLEDGED"):
        executor.apply_event(
            record.order_id,
            OrderEvent(event_id="late-acknowledgement", kind=EventKind.ACKNOWLEDGED),
            PRICE,
        )

    assert counter_value(metrics.FUT_INVALID_TRANSITIONS, from_state="FILLED") == before + 1
    for (from_state,) in label_values(metrics.FUT_INVALID_TRANSITIONS):
        assert from_state in {state.value for state in OrderState}


@requires_prometheus
def test_a_reconciliation_mismatch_and_an_agreement_land_on_separate_outcome_labels():
    """An unlabelled reconciliation count cannot alert on disagreement alone."""
    executor = make_executor()

    before_agreed = counter_value(
        metrics.FUT_RECONCILIATION, outcome=ReconciliationOutcome.AGREED.value
    )
    agreement = executor.reconcile(SYMBOL)
    assert agreement.outcome is ReconciliationOutcome.AGREED
    assert (
        counter_value(metrics.FUT_RECONCILIATION, outcome=ReconciliationOutcome.AGREED.value)
        == before_agreed + 1
    )

    before_mismatch = counter_value(
        metrics.FUT_RECONCILIATION, outcome=ReconciliationOutcome.MISMATCH.value
    )
    executor.venue.apply_settlement(
        SYMBOL, Position(SYMBOL, PositionSide.LONG, Decimal("1"), PRICE)
    )
    mismatch = executor.reconcile(SYMBOL)
    assert mismatch.outcome is ReconciliationOutcome.MISMATCH
    assert (
        counter_value(metrics.FUT_RECONCILIATION, outcome=ReconciliationOutcome.MISMATCH.value)
        == before_mismatch + 1
    )


@requires_prometheus
def test_a_restart_recovery_is_counted_under_the_load_outcome_it_started_from(tmp_path):
    """MISSING and UNREADABLE are not the same restart, and neither is a flat one.

    A single recovery count would make an unreadable state file — the case where
    the process comes up knowing nothing — indistinguishable from a clean start.
    """
    before_missing = counter_value(metrics.FUT_RECOVERY, outcome=LoadOutcome.MISSING.value)
    state = tmp_path / "futures-state.json"
    make_executor(store=FuturesStore.open(state))
    assert state.exists()
    assert (
        counter_value(metrics.FUT_RECOVERY, outcome=LoadOutcome.MISSING.value)
        == before_missing + 1
    )

    before_loaded = counter_value(metrics.FUT_RECOVERY, outcome=LoadOutcome.LOADED.value)
    make_executor(store=FuturesStore.open(state))
    assert (
        counter_value(metrics.FUT_RECOVERY, outcome=LoadOutcome.LOADED.value)
        == before_loaded + 1
    )

    corrupt = tmp_path / "corrupt-state.json"
    corrupt.write_text("{not json")
    before_unreadable = counter_value(
        metrics.FUT_RECOVERY, outcome=LoadOutcome.UNREADABLE.value
    )
    make_executor(store=FuturesStore.open(corrupt))
    assert (
        counter_value(metrics.FUT_RECOVERY, outcome=LoadOutcome.UNREADABLE.value)
        == before_unreadable + 1
    )

    observed = {values[0] for values in label_values(metrics.FUT_RECOVERY)}
    assert observed <= {outcome.value for outcome in LoadOutcome}


@requires_prometheus
def test_executing_a_target_records_one_execution_latency_observation():
    """A latency histogram with no observations is a panel that reads as healthy."""
    executor = make_executor()
    before_count = histogram_count(metrics.FUT_EXECUTION_LATENCY)
    before_sum = histogram_sum(metrics.FUT_EXECUTION_LATENCY)

    move_to(executor, PositionSide.LONG)

    assert histogram_count(metrics.FUT_EXECUTION_LATENCY) == before_count + 1
    # The upper bound is what makes the sum an assertion. An in-process open
    # takes milliseconds, so observing the clock itself instead of the elapsed
    # time would add ~1e9 here — and would satisfy any "is it non-negative"
    # check, which is the shape this assertion used to have.
    assert 0.0 <= histogram_sum(metrics.FUT_EXECUTION_LATENCY) - before_sum < 60.0
