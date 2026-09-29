"""R1-k's unsafe-default detector: a configured limit that no code path enforces.

For EVERY limit a campaign configuration carries, the real `DemoRunner` runs the
same synthetic scenario twice -- once on the committed value, once on a legal
value that should bind -- and only that one limit differs. The committed run
must be allowed (the limit did not bind); the breach run must be refused or
halted with a reason that names the limit. A limit whose value changes nothing
is a limit nothing enforces, and fails here.

The probe table is exhaustive over `LIMIT_FIELDS`, so a limit added to the
schema without a probe fails too. The last test prints one
``R1K-UNSAFE-DEFAULT`` marker line; CI's `unsafe-default-detector` job reads it
and refuses a run in which any configured limit went unprobed.

Synthetic files only. The breach values are chosen to bind in the synthetic
world, never proposed as campaign values.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import pandas as pd
import pytest

from chimera.carry.hedge import HedgeState
from chimera.demo.config import LIMIT_FIELDS
from chimera.demo.runner import RunnerState
from chimera.futures import OrderState
from tests.demo_harness import DAY, build, campaign_config

pytestmark = pytest.mark.unsafe_default

T0 = int(pd.Timestamp(DAY, tz="UTC").timestamp() * 1000)
T0_NS = T0 * 1_000_000
MINUTE_MS = 60_000


@dataclass(frozen=True)
class Probe:
    """One configured limit's two runs and how to read each."""

    #: The legal value that must bind in the scenario.
    breach: Any
    #: What a breach's evidence starts with (an order's reason or a halt reason).
    evidence: tuple[str, ...]
    #: Minutes decided before the evidence is read.
    minutes: int = 3
    #: `build` keywords shared by both runs.
    scenario: Mapping[str, Any] = field(default_factory=dict)
    #: A settlement rate written for the day, when the scenario needs one paid.
    settlement_rate: str | None = None
    #: Replaces the default drive for limits not enforced inside a tick.
    drive: Callable[[Any], None] | None = None


def _order_evidence(harness) -> str | None:
    for executor in (harness.runner.position.perp, harness.runner.position.spot):
        for _, record in sorted(executor.store.state.orders.items()):
            if record.state is OrderState.REJECTED:
                return record.reason
    return None


def _halt_evidence(harness) -> str | None:
    return harness.risk.state.halt_reason or None


def _feed_gate(harness) -> None:
    """`max_data_delay_s` is enforced by the service's READY gate, not a tick:
    the recorder's heartbeat vouches for the feed up to the day's midnight, and
    the operational clock stands 100 s after it."""
    harness.runner.check_feed(T0_NS + 100 * 1_000_000_000)


def _streak_then_retry(harness) -> None:
    """Hold across the 08:00 settlement (paid: the rate is negative for the short
    perpetual), then flatten and ask for the position again: the funding halt, if
    raised, refuses that increase."""
    harness.run(490)
    harness.runner.flatten("R1-k detector: re-open after the settlement")
    harness.tick(T0 + 490 * MINUTE_MS)


PROBES: Mapping[str, Probe] = {
    "funding_adverse_streak_limit": Probe(
        breach=1,
        evidence=("funding halt:",),
        settlement_rate="-0.0001",
        drive=_streak_then_retry,
    ),
    "max_daily_loss_pct": Probe(breach=1e-6, evidence=("max daily loss breached",)),
    "max_data_delay_s": Probe(breach=60, evidence=("FEED_STALLED",), drive=_feed_gate),
    "max_drawdown_pct": Probe(breach=1e-6, evidence=("max drawdown breached",)),
    "max_exposure_per_asset_pct": Probe(
        breach=0.1,
        # Through the per-order cap it derives (`max_position_pct`), which binds
        # first at equal fractions; either reason is this limit's.
        evidence=("order stake", "exposure in"),
    ),
    "max_funding_cost_rate": Probe(
        breach=0.0002,
        evidence=("funding rate would cost this position",),
        # A SHORT perpetual pays 0.0003: under the committed 0.0005, over 0.0002.
        scenario={"mark_funding_rate": -0.0003},
    ),
    "max_leverage": Probe(breach=0.5, evidence=("leverage",)),
    "max_open_positions": Probe(breach=1, evidence=("max open positions",)),
    "max_orders_per_minute": Probe(breach=1, evidence=("order rate limit exceeded",)),
    "max_total_exposure_pct": Probe(breach=0.1, evidence=("total exposure",)),
    "min_liquidation_distance_pct": Probe(breach=5.0, evidence=("liquidation only",)),
}

#: Filled by each probe, printed by the last test.
RESULTS: dict[str, dict[str, str]] = {}


def _run(tmp_path, name: str, probe: Probe, *, breach: bool):
    limits = {name: probe.breach} if breach else None
    config = campaign_config(tmp_path / "state", limits=limits)
    harness = build(tmp_path, config=config, start=False, **dict(probe.scenario))
    if probe.settlement_rate is not None:
        harness.feed.write_settlements([DAY], rate=probe.settlement_rate)
    harness.runner.start()
    (probe.drive or (lambda h: h.run(probe.minutes)))(harness)
    return harness


def _evidence(harness, name: str) -> str | None:
    if name == "max_data_delay_s":
        return "FEED_STALLED" if harness.runner.state is RunnerState.FEED_STALLED else None
    return _halt_evidence(harness) or _order_evidence(harness)


def test_every_configured_limit_has_a_probe():
    """A limit added to the schema without one is a limit nothing proves."""
    assert set(PROBES) == set(LIMIT_FIELDS)


@pytest.mark.parametrize("name", sorted(PROBES))
def test_a_configured_limit_changes_what_the_real_runner_does(tmp_path, name):
    probe = PROBES[name]

    control = _run(tmp_path / "committed", name, probe, breach=False)
    control_evidence = _evidence(control, name)
    assert control_evidence is None, (
        f"{name}: the committed campaign already refuses in the probe's scenario "
        f"({control_evidence}); the probe cannot tell the breach apart"
    )
    if probe.drive is None:
        assert (
            control.runner.position.state is HedgeState.HEDGED
        ), f"{name}: the committed run never opened, so nothing was allowed"

    breach = _run(tmp_path / "breach", name, probe, breach=True)
    breach_evidence = _evidence(breach, name)
    assert breach_evidence is not None, (
        f"{name}: changing it from the committed value to {probe.breach!r} changed "
        "nothing the real runner did. A configured limit no code path enforces"
    )
    assert breach_evidence.startswith(
        probe.evidence
    ), f"{name}: the breach was refused, but for another reason: {breach_evidence!r}"
    RESULTS[name] = {"committed": "allowed", "breach": breach_evidence[:120]}


def test_zz_the_detector_covered_every_configured_limit(capsys):
    """Last in the file: every probe above ran and recorded its two sides."""
    assert set(RESULTS) == set(LIMIT_FIELDS), sorted(set(LIMIT_FIELDS) - set(RESULTS))
    with capsys.disabled():
        print(
            "\nR1K-UNSAFE-DEFAULT "
            + json.dumps(
                {"limits": RESULTS, "configured": sorted(LIMIT_FIELDS)}, sort_keys=True
            )
        )
