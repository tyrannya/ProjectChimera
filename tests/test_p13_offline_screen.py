"""End to end, offline: evidence, design identity, and a hand-traced block.

Three jobs.

The first is the evidence contract: the active design travels with the numbers, a
superseded hash is refused as governing, and a terminated screen reports its
refusal WITHOUT reporting the blocks it had already computed.

The second is dimensional. A carry result is a small number obtained by
subtracting large nearly-equal ones, so every unit is checked explicitly — BTC
against USDT, a fee against a notional rather than a quantity, a fraction against
an absolute — and one block is traced by hand against arithmetic written out in
the test rather than taken from the engine.

The third is the standing guarantee this whole phase rests on: nothing here
reaches a network, and P13 remains economically unrun.
"""

from __future__ import annotations

import json
from decimal import Decimal

import pytest

from nn.p13_blocks import NOT_EVALUABLE
from nn.p13_evidence import (
    FROZEN_ARTIFACT_ROOTS,
    EvidenceError,
    active_design_identity,
    assert_governing_hash,
    block_events,
    write_evidence,
)
from nn.p13_preregistration import (
    ACTIVE_DESIGN,
    CAPITAL_CONTRACT,
    COST_MODEL,
    CURRENT_RESULT_STATE,
    SUPERSEDED_HASHES,
    VENUE_CONSTRAINTS,
    preregistration_hash,
)
from nn.p13_screen import (
    ScreenOutcome,
    frozen_allocation,
    frozen_costs,
    frozen_venue,
    run_offline_screen,
)
from tests.p13_synthetic import block, funding_row, ns, world

SPOT = Decimal("30000")
PERP = Decimal("30030")
MARK = Decimal("30010")


def _screen(aligned, blocks=None) -> ScreenOutcome:
    return run_offline_screen(aligned, min_settlements=0, blocks=blocks or [block(hours=12)])


# ---------------------------------------------------------------------------
# Design identity
# ---------------------------------------------------------------------------


def test_the_active_hash_travels_with_every_evidence_object():
    """Witness 23."""
    outcome = _screen(world(hours=12))
    payload = outcome.evidence.as_dict()
    assert payload["design"]["preregistration_hash"] == preregistration_hash()
    assert payload["design"]["active_design"] == ACTIVE_DESIGN == "P13-A2R1"
    assert payload["design"]["evidence_ceiling"].startswith("EXPLORATORY")


def test_every_superseded_hash_is_refused_as_a_governing_hash():
    """Witness 24. Three retired designs, none of which may govern a run."""
    assert len(SUPERSEDED_HASHES) == 3
    for entry in SUPERSEDED_HASHES:
        with pytest.raises(EvidenceError, match="SUPERSEDED"):
            assert_governing_hash(entry["hash"])
    assert_governing_hash(preregistration_hash())
    with pytest.raises(EvidenceError, match="neither the active"):
        assert_governing_hash("sha256:" + "0" * 64)


def test_the_identity_lists_the_retired_hashes_so_a_reader_can_check_them():
    identity = active_design_identity()
    assert identity["preregistration_hash"] not in identity["superseded_hashes"]
    assert len(identity["superseded_hashes"]) == 3


# ---------------------------------------------------------------------------
# Terminal outcomes reach evidence without their numbers
# ---------------------------------------------------------------------------


def test_a_terminated_screen_reports_the_refusal_and_no_blocks():
    """Witness 15, at the evidence layer."""
    outcome = _screen(world(hours=12, missing_mark=[5]))
    payload = outcome.evidence.as_dict()
    assert payload["evaluable"] is False
    assert payload["result_state"] == NOT_EVALUABLE
    assert payload["blocks"] == []
    assert payload["gate"] is None
    assert payload["stresses"] is None
    assert payload["terminal_refusal"]["is_economic_failure"] is False
    assert "not reported" in payload["partial_results_withheld"]
    assert "not an economic finding" in payload["not_evaluable_is_not_not_viable"]
    # And the whole thing still serialises, so a refusal is a first-class artifact.
    assert json.loads(outcome.evidence.to_json())["result_state"] == NOT_EVALUABLE


def test_no_gate_or_stress_is_computed_once_a_screen_has_terminated():
    outcome = _screen(world(hours=12, missing_mark=[5]))
    assert outcome.gate is None
    assert outcome.stresses is None
    assert outcome.result_state == NOT_EVALUABLE


def test_evidence_refuses_to_be_written_under_a_frozen_artifact_path(tmp_path):
    outcome = _screen(world(hours=12))
    for frozen in FROZEN_ARTIFACT_ROOTS:
        with pytest.raises(EvidenceError, match="frozen primary artifact path"):
            write_evidence(outcome.evidence, tmp_path / frozen / "decision.json")
    written = write_evidence(outcome.evidence, tmp_path / "scratch" / "screen.json")
    assert written.exists()
    assert json.loads(written.read_text())["design"]["active_design"] == "P13-A2R1"


# ---------------------------------------------------------------------------
# A whole offline screen
# ---------------------------------------------------------------------------


def test_an_offline_screen_runs_end_to_end_and_produces_a_verdict():
    settlements = (funding_row(3, "0.0001"), funding_row(7, "0.0001"))
    outcome = _screen(world(hours=12, funding=settlements))
    assert outcome.screen.evaluable
    assert outcome.gate is not None
    # One block cannot satisfy the frozen five-block minimum, so the honest
    # outcome of a one-block synthetic world is INVALID, not a verdict.
    assert outcome.result_state.endswith("INVALID")
    payload = outcome.evidence.as_dict()
    assert len(payload["blocks"]) == 1
    assert payload["events"][0]["event"] == "open"
    assert payload["events"][-1]["event"] == "close"


def test_the_event_ledger_records_every_settlement_between_open_and_close():
    settlements = (funding_row(3, "0.0001"), funding_row(7, "-0.0002"))
    outcome = _screen(world(hours=12, funding=settlements))
    events = block_events(outcome.screen.blocks[0])
    kinds = [event["event"] for event in events]
    assert kinds.count("funding_settlement") == 2
    assert kinds[0] == "open" and kinds[-1] == "close"


def test_the_evidence_reports_how_many_settlements_used_the_substituted_base():
    """MARK_PRICE_FALLBACK.reporting_granularity, and it is funding-only."""
    outcome = _screen(
        world(hours=12, funding=(funding_row(3, "0.0001"),), published_mark_periods=())
    )
    payload = outcome.evidence.as_dict()
    fallback = payload["funding_notional_fallback"]
    assert fallback["settlements_on_substituted_base"] == 1
    assert "never for liquidation" in fallback["authorised_for"]


# ---------------------------------------------------------------------------
# Dimensions and the hand trace
# ---------------------------------------------------------------------------


def test_one_block_traced_by_hand_in_a_flat_world():
    """Every term computed here, in the test, and compared against the engine."""
    allocation, costs, venue = frozen_allocation(), frozen_costs(), frozen_venue()
    one = Decimal("1")

    # Sizing: the minimum over BOTH legs' bounds, floored to the step.
    spot_bound = allocation.spot / (SPOT * (one + costs.spot_fee + costs.spot_slippage))
    perp_bound = allocation.perp / (PERP * (one + costs.perp_fee + costs.perp_slippage))
    steps = (min(spot_bound, perp_bound) / venue.step_size).to_integral_value(
        rounding="ROUND_DOWN"
    )
    quantity = steps * venue.step_size

    outcome = _screen(world(hours=12))
    result = outcome.screen.blocks[0].result
    assert result.quantity == quantity

    # BTC x USDT/BTC = USDT. Fees are charged on the NOTIONAL, never the quantity.
    spot_notional = quantity * SPOT
    perp_notional = quantity * PERP
    one_way_fees = spot_notional * costs.spot_fee + perp_notional * costs.perp_fee
    one_way_slip = spot_notional * costs.spot_slippage + perp_notional * costs.perp_slippage
    assert result.fees == one_way_fees * 2
    assert result.slippage == one_way_slip * 2

    # A flat world telescopes to zero basis PnL and charges only friction.
    assert result.basis_entry == PERP - SPOT
    assert result.basis_exit == PERP - SPOT
    assert result.basis_pnl == 0
    assert result.net_pnl == -(result.fees + result.slippage)
    assert result.net_return == result.net_pnl / allocation.total_capital

    # The excursion is a FRACTION of total capital, comparable with the -0.02
    # floor, and its absolute twin is in quote units.
    assert result.max_adverse_excursion == result.max_adverse_excursion_pnl / (
        allocation.total_capital
    )
    assert result.max_adverse_excursion <= result.net_return


def test_a_short_receives_on_a_positive_rate_and_pays_on_a_negative_one():
    """The sign convention, checked as cash rather than restated."""
    received = _screen(world(hours=12, funding=(funding_row(4, "0.0001"),)))
    paid = _screen(world(hours=12, funding=(funding_row(4, "-0.0001"),)))
    got = received.screen.blocks[0].result
    lost = paid.screen.blocks[0].result
    assert got.funding_received > 0 and got.funding_paid == 0
    assert lost.funding_paid > 0 and lost.funding_received == 0
    # And the magnitude is quantity x notional-base x rate, in USDT.
    expected = got.quantity * MARK * Decimal("0.0001")
    assert got.funding_received == expected
    assert lost.funding_paid == expected
    assert got.net_return > lost.net_return


def test_the_capital_and_cost_constants_come_from_the_frozen_design():
    allocation, costs, venue = frozen_allocation(), frozen_costs(), frozen_venue()
    assert allocation.total_capital == Decimal(CAPITAL_CONTRACT["total_starting_capital"])
    assert allocation.spot + allocation.perp == allocation.total_capital
    assert costs.spot_fee == Decimal(COST_MODEL["spot_entry_fee_rate"])
    assert costs.perp_slippage == Decimal(COST_MODEL["perp_slippage_rate"])
    assert venue.maintenance_margin_rate == Decimal(
        VENUE_CONSTRAINTS["perpetual"]["tier_1_maintenance_margin_rate"]
    )
    assert venue.step_size == Decimal("0.001")


def test_every_reported_economic_value_is_a_string_not_a_float():
    """A Decimal rendered through binary floating point is a different number."""
    outcome = _screen(world(hours=12, funding=(funding_row(3, "0.0001"),)))
    payload = outcome.evidence.as_dict()
    for entry in payload["blocks"]:
        for key, value in entry.items():
            if key.endswith(("_quote", "_fraction", "_btc")) or key.startswith("basis"):
                assert isinstance(value, str), f"{key} is {type(value).__name__}"
    # And every one of them round-trips back to the exact Decimal the engine held.
    result = outcome.screen.blocks[0].result
    entry = payload["blocks"][0]
    assert Decimal(entry["net_return_fraction"]) == result.net_return
    assert Decimal(entry["net_pnl_quote"]) == result.net_pnl
    assert Decimal(entry["quantity_btc"]) == result.quantity


# ---------------------------------------------------------------------------
# The standing guarantees
# ---------------------------------------------------------------------------


def test_no_p13_runtime_module_imports_a_network_library():
    """Witness 29-30. Checked on the import graph of every module in the chain."""
    import ast
    import inspect

    from nn import (
        p13_alignment,
        p13_blocks,
        p13_carry,
        p13_evidence,
        p13_gate,
        p13_screen,
        p13_sources,
        p13_stress,
    )

    banned = {"requests", "urllib", "urllib3", "http", "socket", "ftplib", "aiohttp", "ccxt"}
    for module in (
        p13_sources,
        p13_alignment,
        p13_blocks,
        p13_carry,
        p13_gate,
        p13_stress,
        p13_screen,
        p13_evidence,
    ):
        tree = ast.parse(inspect.getsource(module))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert not (imported & banned), f"{module.__name__} imports {imported & banned}"


def test_p13_remains_economically_unrun_after_this_phase():
    """No governed artifact exists, and the declared state still says so."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    assert CURRENT_RESULT_STATE.endswith("NOT YET RUN")
    assert not (
        root / "artifacts" / "benchmark" / "btc_p13_decision" / "decision.json"
    ).exists()
    carry_dir = root / "artifacts" / "benchmark" / "btc_p13_carry"
    for economic in (
        "blocks.json",
        "gate.json",
        "decision.json",
        "events.jsonl",
        "screen.json",
    ):
        assert not (carry_dir / economic).exists()


def test_no_leverage_above_one_is_reachable_from_the_runtime():
    """The perpetual leg is margined at exactly its notional, so leverage is 1x."""
    from nn.p13_carry import open_carry

    allocation, costs, venue = frozen_allocation(), frozen_costs(), frozen_venue()
    aligned = world(hours=3)
    position = open_carry(
        aligned.quote(ns("2021-03-01T00:00:00+00:00")), allocation, costs, venue
    )
    assert position.leverage == 1
    assert position.perp_margin == position.quantity * PERP
    assert position.free_cash >= 0
