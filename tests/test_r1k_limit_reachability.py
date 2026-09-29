"""R1-k: the reachability inventory, and the loss streak and cooldown retired.

`chimera.demo.limit_reachability.INVENTORY` classifies every limit the campaign
schema carries or once carried as REACHABLE, UNREACHABLE or RETIRED_FROM_CONFIG,
and `check_inventory` refuses the table the moment it stops describing the code.
Every refusal is driven here with a table that really is broken; a guard only
ever asserted against the correct table is a guard that cannot fail.

The loss streak and the cooldown were configured and enforced by nothing:
`RiskEngine.record_trade_result` has no caller on the demo path. R1-k retires
both from the schema. Both sides are witnessed: the parser refuses each by name,
and the scan that justifies the retirement finds a caller when one exists.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from chimera.demo.config import (
    LIMIT_FIELDS,
    RETIRED_LIMITS,
    ConfigProfile,
    DemoConfigError,
    DemoLimits,
    load_demo_config,
    parse_demo_config,
)
from chimera.demo.limit_reachability import (
    INVENTORY,
    LimitReachabilityError,
    Reachability,
    check_inventory,
    record_trade_result_callers,
)
from chimera.demo.risk_wiring import DIRECT_LIMITS, UNCONFIGURED_LIMITS
from tests.demo_harness import build

pytestmark = pytest.mark.unsafe_default

REPO = Path(__file__).resolve().parents[1]
PVC1 = REPO / "conf" / "demo" / "pvc1.json"
RETIRED = ("loss_streak_limit", "cooldown_seconds")


def committed_payload() -> dict:
    return json.loads(PVC1.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# the inventory
# ---------------------------------------------------------------------------
def test_the_inventory_describes_the_code():
    check_inventory()


def test_every_configured_limit_is_reachable_and_every_retired_one_is_retired():
    classes = {name: row.classification for name, row in INVENTORY.items()}
    assert {n for n, c in classes.items() if c is Reachability.REACHABLE} == set(LIMIT_FIELDS)
    assert {n for n, c in classes.items() if c is Reachability.RETIRED_FROM_CONFIG} == set(
        RETIRED
    )
    assert Reachability.UNREACHABLE not in classes.values()


def test_every_reachable_row_says_reductions_stay_possible():
    assert all(row.reductions_allowed for row in INVENTORY.values())


def test_the_funding_row_names_the_rate_in_effect_not_the_realised_settlement():
    row = INVENTORY["max_funding_cost_rate"]
    assert "funding_rate_current" in row.runtime_input
    assert (
        "chimera.carry.hedge:HedgedPosition._funding_input",
        "state.funding_rate_current",
    ) in (row.chain)


def _without(name: str) -> dict:
    return {k: v for k, v in INVENTORY.items() if k != name}


def test_a_configured_limit_with_no_row_is_refused():
    with pytest.raises(LimitReachabilityError, match="no reachability row.*max_leverage"):
        check_inventory(_without("max_leverage"))


def test_a_row_for_no_such_limit_is_refused():
    extra = {**INVENTORY, "max_imaginary": INVENTORY["max_leverage"]}
    with pytest.raises(LimitReachabilityError, match="rows for no such limit"):
        check_inventory(extra)


def test_a_configured_limit_classified_unreachable_is_refused():
    broken = {
        **INVENTORY,
        "max_leverage": dataclasses.replace(
            INVENTORY["max_leverage"], classification=Reachability.UNREACHABLE
        ),
    }
    with pytest.raises(LimitReachabilityError, match="must be wired or retired"):
        check_inventory(broken)


def test_a_retired_limit_still_in_the_schema_is_refused():
    with pytest.raises(LimitReachabilityError, match="still configurable"):
        check_inventory(configured=set(LIMIT_FIELDS) | {"cooldown_seconds"})


def test_a_retired_row_classified_reachable_is_refused():
    broken = {
        **INVENTORY,
        "cooldown_seconds": dataclasses.replace(
            INVENTORY["cooldown_seconds"], classification=Reachability.REACHABLE
        ),
    }
    with pytest.raises(LimitReachabilityError, match="retired and classified REACHABLE"):
        check_inventory(broken)


def test_a_broken_link_in_a_chain_is_refused():
    """The funding wiring removed from the hedge: the chain no longer holds."""
    row = INVENTORY["max_funding_cost_rate"]
    chain = tuple(
        (
            (q, "funding_rate=float(this_is_not_in_the_source)")
            if q.endswith("HedgedPosition._funding_input")
            else (q, t)
        )
        for q, t in row.chain
    )
    broken = {**INVENTORY, "max_funding_cost_rate": dataclasses.replace(row, chain=chain)}
    with pytest.raises(LimitReachabilityError, match="no longer contains"):
        check_inventory(broken)


def test_a_chain_naming_a_function_that_does_not_exist_is_refused():
    """The source is read statically; a renamed or deleted function is a
    refusal, not an empty source that no token could be missing from."""
    row = INVENTORY["max_leverage"]
    chain = (("chimera.risk:RiskEngine.no_such_method", "lim.max_leverage"),) + row.chain[1:]
    broken = {**INVENTORY, "max_leverage": dataclasses.replace(row, chain=chain)}
    with pytest.raises(LimitReachabilityError, match="cannot be read"):
        check_inventory(broken)


def test_a_reachable_row_with_no_chain_is_refused():
    broken = {
        **INVENTORY,
        "max_leverage": dataclasses.replace(INVENTORY["max_leverage"], chain=()),
    }
    with pytest.raises(LimitReachabilityError, match="no chain"):
        check_inventory(broken)


# ---------------------------------------------------------------------------
# the retirement: no caller, so no configured limit
# ---------------------------------------------------------------------------
def test_nothing_in_the_production_trees_calls_record_trade_result():
    assert record_trade_result_callers() == []


def test_the_caller_scan_finds_a_caller_when_there_is_one(tmp_path):
    """The other side: a caller anywhere in a production tree is found."""
    tree = tmp_path / "chimera" / "demo"
    tree.mkdir(parents=True)
    (tree / "wired.py").write_text(
        "def close(risk, pnl):\n    risk.record_trade_result(pnl)\n", encoding="utf-8"
    )
    (tmp_path / "tools").mkdir()
    assert record_trade_result_callers(tmp_path) == ["chimera/demo/wired.py:2"]


def test_the_scan_covers_the_real_trees_not_an_empty_directory():
    """`record_trade_result` itself is defined in `chimera/risk.py`; the scan
    must be reading that tree, or its silence would be about nothing."""
    assert (REPO / "chimera" / "risk.py").is_file() and (
        REPO / "tools" / "demo_run.py"
    ).is_file()
    assert "def record_trade_result" in (REPO / "chimera" / "risk.py").read_text(
        encoding="utf-8"
    )


@pytest.mark.parametrize("name", RETIRED)
def test_a_retired_limit_is_refused_by_name_with_the_reason(name):
    payload = committed_payload()
    payload["limits"] = {**payload["limits"], name: 3}
    with pytest.raises(DemoConfigError, match=rf"limits\.{name} is no longer a limit.*R1-k"):
        parse_demo_config(payload)


def test_both_retired_limits_are_named_with_what_became_of_them():
    assert set(RETIRED_LIMITS) == set(RETIRED)
    assert all("R1-k" in reason for reason in RETIRED_LIMITS.values())


def test_the_retired_limits_are_not_fields_any_more():
    assert not set(RETIRED) & {f.name for f in dataclasses.fields(DemoLimits)}
    assert not set(RETIRED) & set(LIMIT_FIELDS)
    with pytest.raises(TypeError):
        dataclasses.replace(load_demo_config(PVC1).limits, loss_streak_limit=3)


def test_the_committed_campaign_no_longer_carries_them():
    raw = committed_payload()
    assert not set(RETIRED) & set(raw["limits"])
    load_demo_config(PVC1)


def test_the_engine_keeps_the_default_and_the_wiring_names_why():
    assert set(RETIRED) <= set(UNCONFIGURED_LIMITS)
    assert not set(RETIRED) & set(DIRECT_LIMITS)
    assert all("R1-k" in UNCONFIGURED_LIMITS[name] for name in RETIRED)


def test_a_retired_limit_in_a_test_profile_is_refused_too():
    payload = {**committed_payload(), "profile": "TEST"}
    payload["limits"] = {**payload["limits"], "cooldown_seconds": 60}
    with pytest.raises(DemoConfigError, match="cooldown_seconds is no longer a limit"):
        parse_demo_config(payload, expected_profile=ConfigProfile.TEST)


def test_a_real_campaign_run_never_opens_a_cooldown(tmp_path):
    """Two hundred decided minutes, opening and holding: nothing ever counts a
    loss, which is the fact the retirement rests on, observed on the real path."""
    harness = build(tmp_path)
    harness.run(200)
    assert harness.risk.state.consecutive_losses == 0
    assert harness.risk.state.cooldown_until == 0.0
