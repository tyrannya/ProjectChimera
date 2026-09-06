"""A runnable demo campaign over synthetic files. Shared by the PR-10 tests.

Everything here is invented: the prices come from `chimera.demo.fixtures`, the
venue is a `DryRunFuturesVenue`, and the rule parameters are the test's own.
No market data is read and no economic claim is available from any of it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

from chimera.carry.factory import build_hedged_position
from chimera.demo.config import ConfigProfile, DemoConfig, parse_demo_config
from chimera.demo.feed import FeedCursor
from chimera.demo.fixtures import MinuteShape, SyntheticFeed
from chimera.demo.rules import RuleRegistry
from chimera.demo.rules_carry import CarryParams, CarryRule
from chimera.demo.rules_shadow import DailyMomentumRule, FrozenLogisticRule, ShadowParams
from chimera.demo.runner import DemoRunner
from chimera.futures.fills import RecordedQuoteFillModel
from chimera.recorder.contract import load_recorder_contract
from chimera.risk import RiskEngine, RiskLimits

REPO = Path(__file__).resolve().parents[1]
DAY = "2026-09-19"
NEXT_DAY = "2026-09-20"
CAPITAL = Decimal("1000000")

#: Rule parameters for the tests. Chosen by the test author for a synthetic
#: world; they are not, and may not become, campaign parameters. The S2
#: protocol freezes those and it is PR-14's.
CARRY_PARAMS: Mapping[str, str] = {
    "min_basis": "10",
    "max_notional_fraction": "0.5",
    "step_size": "0.001",
    "min_notional": "10",
    "spot_leg_fraction": "0.5",
    "spot_fee_rate": "0.001",
    "perp_fee_rate": "0.0005",
    "slippage_rate": "0.0002",
}
SHADOW_PARAMS: Mapping[str, Any] = {
    "coefficients": ["1", "0", "0"],
    "intercept": "0",
    "threshold": "20",
}
MOMENTUM_PARAMS: Mapping[str, Any] = {
    "coefficients": ["1", "1"],
    "intercept": "0",
    "threshold": "0",
}


def campaign_config(
    state_dir: Path,
    *,
    profile: str = "TEST",
    rules: Mapping[str, Any] | None = None,
    runner: Mapping[str, Any] | None = None,
    faults: Mapping[str, Any] | None = None,
) -> DemoConfig:
    base = json.loads((REPO / "conf" / "demo" / "pvc1.json").read_text(encoding="utf-8"))
    payload: dict[str, Any] = {
        **base,
        "profile": profile,
        "runner": {"state_dir": str(state_dir), **dict(runner or {})},
        "rules": dict(
            rules
            if rules is not None
            else {
                "R1_carry": dict(CARRY_PARAMS),
                "R2_frozen_logistic": dict(SHADOW_PARAMS),
                "R3_daily_momentum": dict(MOMENTUM_PARAMS),
            }
        ),
    }
    if faults is not None:
        payload["faults"] = dict(faults)
    return parse_demo_config(payload, expected_profile=ConfigProfile(profile))


@dataclass
class Harness:
    """A runner and everything it was built from, so a test can reach in."""

    runner: DemoRunner
    model: RecordedQuoteFillModel
    root: Path
    state_dir: Path
    feed: SyntheticFeed
    risk: RiskEngine

    def first_minute_ms(self) -> int:
        minute = self.runner.cursor.next_minute_ms()
        assert minute is not None, "the fixture wrote no minutes"
        return minute

    def tick(self, minute_ms: int):
        """One minute, exactly as the production entry point drives it.

        This helper used to install the decision minute's book on the fill model
        before each tick, and take a ``quote=False`` flag to suppress it. Both are
        gone: `DemoRunner.tick` installs the book itself now, which is what
        `RecordedQuoteFillModel` always said the runner owed it.

        The harness doing that job is what hid its absence. Every test here ran
        against a model somebody else had loaded, so a suite that exercised
        funding, reconciliation and liquidation end to end still never noticed
        that through `tools/demo_run.py` no order could fill and a campaign could
        not open a position at all. A harness that supplies what production does
        not is a harness that tests itself.
        """
        return self.runner.tick(minute_ms)

    def run(self, count: int, *, start: int | None = None) -> list:
        first = start if start is not None else self.first_minute_ms()
        return [self.tick(first + i * 60_000) for i in range(count)]

    def records(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for path in sorted((self.state_dir / "decision_log").glob("*.ndjson")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    out.append(json.loads(line))
        return out


def build(
    tmp_path: Path,
    *,
    days: tuple[str, ...] = (DAY,),
    shapes: Mapping[str, Mapping[int, MinuteShape]] | None = None,
    config: DemoConfig | None = None,
    with_shadow: bool = True,
    start: bool = True,
    telemetry: Any | None = None,
) -> Harness:
    root = tmp_path / "recorder"
    state_dir = tmp_path / "state"
    contract = load_recorder_contract("btcusdt-prospective-gen3")

    feed = SyntheticFeed(root, contract)
    feed.write_days(list(days), shapes=shapes)
    feed.write_settlements(list(days))

    cfg = config or campaign_config(state_dir)
    risk = RiskEngine(
        RiskLimits(max_position_pct=1.0, risk_per_trade_pct=0.5),
        state_path=state_dir / "risk.json",
        kill_switch_path=state_dir / "KILL_SWITCH",
    )
    risk.update_equity(float(CAPITAL))
    model = RecordedQuoteFillModel()
    position = build_hedged_position(
        risk=risk, capital=CAPITAL, state_dir=state_dir, fill_model=model
    )
    position.spot.recover({})
    position.perp.recover({})

    rules = RuleRegistry([CarryRule(CarryParams.from_config(cfg.rule_params("R1_carry")))])
    if with_shadow:
        rules.register(
            FrozenLogisticRule(
                ShadowParams.from_config("R2", cfg.rule_params("R2_frozen_logistic"))
            )
        )
        rules.register(
            DailyMomentumRule(
                ShadowParams.from_config("R3", cfg.rule_params("R3_daily_momentum"))
            )
        )

    runner = DemoRunner(
        cfg,
        root,
        contract=contract,
        risk=risk,
        position=position,
        rules=rules,
        capital=CAPITAL,
        software={"revision": "synthetic", "dirty": False, "python": "3.11"},
        telemetry=telemetry,
    )
    harness = Harness(runner, model, root, state_dir, feed, risk)
    if start:
        runner.start()
    return harness


def cursor_for(harness: Harness) -> FeedCursor:
    return FeedCursor(harness.root, harness.runner.contract)
