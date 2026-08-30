"""The runnable path: market data -> Pythia -> mode -> Aegis -> Hermes -> Argus.

    python -m tools.paper_run --smoke --bars 500 --out artifacts/paper_smoke

**What this is.** One loop that walks closed candles in time order and drives the
whole chain for each one: specialist predictions, the per-mode consensus, the
mode controller's eligibility check, and — when a mode is eligible and reaches a
direction — a target position through `FuturesExecutor` into the dry-run venue,
with Argus recording bounded telemetry throughout. It exists so that sustained
paper operation later is a matter of pointing it at a live feed and leaving it
running, rather than of writing the loop then.

**What this is not.** It is **not** sustained paper validation, and a run of it
must never be described as one. It is not live: `DryRunFuturesVenue` is the only
venue in the package, `FuturesExecutionConfig` refuses anything but dry-run 1x
isolated at construction, and no credential, socket or exchange endpoint is
reachable from `chimera.futures` at all. Nothing here places a real order.

**Two market-data sources, one loop.**

* ``ReplaySource`` reads the committed multi-clock snapshot and the frozen P6
  predictions. Deterministic, offline, and what ``--smoke`` uses.
* ``LiveSource`` is the seam a sustained run plugs into: it needs a feed of
  closed candles and a specialist that serves predictions on each clock. P6
  measured its specialists but did not persist estimators, so no such specialist
  exists yet — the class states that and refuses rather than pretending.

**The honest expected outcome today.** P6, P6-EXT and P7 were all negative, so no
mode is eligible and the controller can only return `FLAT`. A smoke therefore
exercises ingestion, alignment, consensus, eligibility, the Aegis and Hermes
wiring and the telemetry, and correctly places **no order at all**. That is the
system working, and the report says so rather than reading as a failure.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterator, Protocol

import pandas as pd

from chimera import metrics
from chimera.contracts import Signal
from chimera.futures import (
    DeterministicFillModel,
    DryRunFuturesVenue,
    FuturesExecutionConfig,
    FuturesExecutor,
    FuturesStore,
    StaticConstraintSource,
    default_constraints_table,
)
from chimera.modes import (
    MODE_SPECS,
    ModeDecision,
    SpecialistStatus,
    TradingMode,
    decide_mode,
    evaluate_eligibility,
    plan_mode_transition,
)
from nn.multiclock import constituent_count
from nn.p7 import align_to_decision_clock, load_specialist

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = REPO_ROOT / "artifacts" / "benchmark"

SYMBOL = "BTC/USDT:USDT"
REPORT_NAME = "paper_run.json"
STATUS_NAME = "STATUS.md"

#: Stamped into every report. A reader who finds this file has to be told, in the
#: file, what it is not.
RUN_CLASS = (
    "an engineering smoke of the Pythia -> mode -> Aegis -> Hermes -> dry-run venue -> "
    "Argus path. NOT sustained paper validation, NOT live, and NOT evidence about alpha."
)


class PaperRunError(SystemExit):
    """The runnable path cannot be run the way it has been asked to be."""


class MarketSource(Protocol):
    """A feed of closed decision bars, with each clock's specialist prediction."""

    decision_clock: str

    def bars(self) -> Iterator[tuple[pd.Timestamp, float, dict[str, Signal | None]]]:
        """Yield ``(close_time, price, {clock: signal or None})`` in time order."""


@dataclass
class ReplaySource:
    """Frozen P6 predictions, replayed in time order on a mode's decision clock.

    Deterministic and offline. It reads the same committed prediction files P7
    replayed, through the same causal alignment, so the signal a bar sees here is
    the signal that bar would have seen — no forward fill, and nothing from a bar
    that had not closed.
    """

    decision_clock: str
    clocks: tuple[str, ...]
    limit: int | None = None
    fold: int = 0
    _frames: dict[str, pd.DataFrame] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        for clock in self.clocks:
            frame = load_specialist(clock)
            self._frames[clock] = frame.loc[frame["fold"] == self.fold].reset_index(drop=True)
        if self._frames[self.decision_clock].empty:
            raise PaperRunError(
                f"the frozen {self.decision_clock} specialist has no fold {self.fold}"
            )

    def bars(self) -> Iterator[tuple[pd.Timestamp, float, dict[str, Signal | None]]]:
        decision = self._frames[self.decision_clock]
        if self.limit is not None:
            decision = decision.iloc[: self.limit]
        lookup = {SHORT: Signal.SHORT, HOLD: Signal.HOLD, LONG: Signal.LONG}
        aligned = {
            clock: align_to_decision_clock(
                decision["timestamp"].to_numpy(dtype="datetime64[ns]"),
                self.decision_clock,
                self._frames[clock]["timestamp"].to_numpy(dtype="datetime64[ns]"),
                clock,
            )
            for clock in self.clocks
        }
        width = pd.Timedelta(minutes=constituent_count(self.decision_clock))
        for position in range(len(decision)):
            row = decision.iloc[position]
            signals: dict[str, Signal | None] = {}
            for clock in self.clocks:
                index = int(aligned[clock][position])
                if index < 0:
                    signals[clock] = None
                    continue
                action = int(self._frames[clock]["selected_action"].iloc[index])
                signals[clock] = lookup[action]
            # A synthetic reference price: the replay carries predictions, not
            # candles, and the venue only needs a price to quantise against. It
            # is deliberately constant so that a smoke's fills cannot be mistaken
            # for a backtest.
            yield (pd.Timestamp(row["timestamp"]) + width, REPLAY_REFERENCE_PRICE, signals)


SHORT, HOLD, LONG = 0, 1, 2

#: Constant on purpose. A replay drives the execution chain, not a PnL: a varying
#: price would produce a simulated equity curve that somebody could mistake for a
#: result, and there is no result here.
REPLAY_REFERENCE_PRICE = 60_000.0


@dataclass
class LiveSource:
    """The seam a sustained paper run plugs into. Not usable yet, and says so."""

    decision_clock: str
    clocks: tuple[str, ...]

    def bars(self) -> Iterator[tuple[pd.Timestamp, float, dict[str, Signal | None]]]:
        raise PaperRunError(
            "a live paper run needs a specialist that serves predictions on each clock, and "
            "none exists: P6 measured its specialists but did not persist estimators. "
            "docs/paper_operation_runbook.md section 3 records what has to be built. Until "
            "then --smoke replays the frozen predictions instead of pretending."
        )


def committed_specialist_status() -> dict[str, SpecialistStatus]:
    """Specialist viability, read from the committed decision artifacts.

    Eligibility is a fact about the evidence tree, so the runner derives it the
    way a reader would rather than carrying a table that could go stale.
    """
    status: dict[str, SpecialistStatus] = {}
    for directory, checkpoint in (("btc_p6_decision", "P6"), ("btc_p6ext_decision", "P6-EXT")):
        path = BENCHMARK / directory / "decision.json"
        if not path.is_file():
            continue
        for row in json.loads(path.read_text())["clocks"]:
            status[row["clock"]] = SpecialistStatus(
                clock=row["clock"],
                screened=True,
                viable=bool(row["viable"]),
                checkpoint=checkpoint,
            )
    return status


def build_executor(state_path: Path) -> FuturesExecutor:
    """Hermes over the dry-run venue. Refuses to be anything else at construction."""
    from chimera.risk import RiskEngine

    venue = DryRunFuturesVenue(
        # `from_mapping`, not the constructor: the table is plain dicts and the
        # source holds SymbolConstraints. `tools.futures_dry_run` builds it the
        # same way, and the two must not drift into two venue configurations.
        source=StaticConstraintSource.from_mapping(default_constraints_table()),
        fill_model=DeterministicFillModel(),
    )
    store = FuturesStore(state_path)
    executor = FuturesExecutor(
        venue=venue,
        risk=RiskEngine(),
        store=store,
        config=FuturesExecutionConfig(),
    )
    executor.recover({})
    return executor


@dataclass
class RunTotals:
    """Everything a smoke reports. Counts and bounded states, never a return."""

    bars: int = 0
    decisions: dict[str, int] = field(default_factory=dict)
    modes: dict[str, int] = field(default_factory=dict)
    reasons: dict[str, int] = field(default_factory=dict)
    orders_planned: int = 0
    orders_filled: int = 0
    risk_vetoes: int = 0
    transitions: int = 0
    flattens: int = 0

    def observe(self, decision: ModeDecision) -> None:
        self.bars += 1
        self.modes[decision.mode.value] = self.modes.get(decision.mode.value, 0) + 1
        self.reasons[decision.reason.value] = self.reasons.get(decision.reason.value, 0) + 1
        self.decisions[decision.signal.value] = (
            self.decisions.get(decision.signal.value, 0) + 1
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "bars": self.bars,
            "signals": dict(sorted(self.decisions.items())),
            "modes": dict(sorted(self.modes.items())),
            "reasons": dict(sorted(self.reasons.items())),
            "orders_planned": self.orders_planned,
            "orders_filled": self.orders_filled,
            "risk_vetoes": self.risk_vetoes,
            "mode_transitions": self.transitions,
            "transition_flattens": self.flattens,
            "flat_fraction": (
                round(self.modes.get("FLAT", 0) / self.bars, 6) if self.bars else None
            ),
        }


def run(
    source: MarketSource,
    declared: TradingMode,
    executor: FuturesExecutor,
    *,
    equity: float = 100_000.0,
    quantity: Decimal = Decimal("0.01"),
) -> RunTotals:
    """Walk closed bars in time order and drive the chain for each one."""
    eligibility = evaluate_eligibility(committed_specialist_status())
    totals = RunTotals()
    active = TradingMode.FLAT

    for close_time, price, signals in source.bars():
        decision = decide_mode(declared, signals, eligibility)
        totals.observe(decision)
        metrics.mark_mode_decision(decision)

        if decision.mode is not active:
            position = executor.position(SYMBOL)
            plan = plan_mode_transition(
                active, decision.mode, position_is_flat=position.is_flat
            )
            metrics.mark_mode_transition(plan)
            totals.transitions += 1
            if plan.must_flatten:
                totals.flattens += 1
                from chimera.futures import FlattenCause

                executor.emergency_flatten(SYMBOL, FlattenCause.OPERATOR, Decimal(str(price)))
            active = decision.mode

        if decision.is_flat:
            # FLAT is a successful outcome, not a skipped iteration: the target
            # is explicitly no position, and Hermes is asked for it so that a
            # position left over from an earlier bar is closed rather than held.
            target = executor.target_for(Signal.HOLD, SYMBOL, quantity)
        else:
            target = executor.target_for(decision.signal, SYMBOL, quantity)

        records = executor.execute_target(
            target, Decimal(str(price)), equity=equity, data_delay_s=0.0, inference_age_s=0.0
        )
        totals.orders_planned += len(records)
        totals.orders_filled += sum(1 for record in records if record.state.name == "FILLED")
        totals.risk_vetoes += sum(1 for record in records if record.state.name == "REJECTED")
        logger.debug("%s %s -> %s", close_time, decision.mode.value, decision.signal.value)

    return totals


def report(
    totals: RunTotals, declared: TradingMode, source: MarketSource, executor: FuturesExecutor
) -> dict[str, Any]:
    eligibility = evaluate_eligibility(committed_specialist_status())
    return {
        "run_class": RUN_CLASS,
        "claims": {
            "sustained_paper_validation": False,
            "live": False,
            "real_money": False,
            "alpha": False,
        },
        "declared_mode": declared.value,
        "decision_clock": source.decision_clock,
        "source": type(source).__name__,
        "eligibility": {mode.value: row.to_dict() for mode, row in eligibility.items()},
        "execution": {
            "venue": type(executor.venue).__name__,
            "dry_run": executor.config.dry_run,
            "leverage": str(executor.config.leverage),
            "margin_mode": executor.config.margin_mode,
        },
        "totals": totals.to_dict(),
        "ledger": {
            "realised_pnl": str(executor.ledger.realised_pnl),
            "trading_fees": str(executor.ledger.trading_fees),
            "turnover": str(executor.ledger.turnover),
            "note": (
                "simulated, from a constant reference price, and not evidence about alpha"
            ),
        },
    }


def status_markdown(payload: dict[str, Any]) -> str:
    totals = payload["totals"]
    eligible = [name for name, row in payload["eligibility"].items() if row["eligible"]]
    return "\n".join(
        [
            # `# OPERATIONAL` on the first line: this is the directory's status
            # marker, and `artifacts/README.md` indexes it as operational rather
            # than CURRENT. It is not research evidence and must not be filed
            # beside anything that is.
            "# OPERATIONAL",
            "",
            "## Paper-path engineering smoke",
            "",
            f"**{payload['run_class']}**",
            "",
            f"- declared mode: `{payload['declared_mode']}` on the "
            f"`{payload['decision_clock']}` clock, source `{payload['source']}`",
            f"- eligible modes: {eligible or 'none'}",
            f"- bars driven: {totals['bars']:,}",
            f"- modes entered: {totals['modes']}",
            f"- reasons: {totals['reasons']}",
            f"- orders planned: {totals['orders_planned']}, "
            f"filled: {totals['orders_filled']}, vetoed: {totals['risk_vetoes']}",
            f"- venue: `{payload['execution']['venue']}`, "
            f"dry_run={payload['execution']['dry_run']}, "
            f"leverage={payload['execution']['leverage']}, "
            f"margin={payload['execution']['margin_mode']}",
            "",
            "Sustained multi-day paper validation is **not** claimed by this run and has "
            "not been performed. See `docs/paper_operation_runbook.md`.",
            "",
        ]
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke", action="store_true", help="replay frozen predictions offline"
    )
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in TradingMode],
        default=TradingMode.SCALPING.value,
        help="the operator's declared mode; honoured only if the evidence supports it",
    )
    parser.add_argument("--bars", type=int, default=500)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--state", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_argparser().parse_args(argv)
    declared = TradingMode(args.mode)

    if declared is TradingMode.FLAT:
        clocks: tuple[str, ...] = ("1m",)
        decision_clock = "1m"
    else:
        spec = MODE_SPECS[declared]
        clocks = spec.specialists
        decision_clock = spec.decision_clock

    if not args.smoke:
        source: MarketSource = LiveSource(decision_clock, clocks)
        source.bars()  # raises, with the reason
        return 1

    source = ReplaySource(decision_clock, clocks, limit=args.bars, fold=args.fold)
    out = args.out or (REPO_ROOT / "artifacts" / "paper_smoke")
    state = args.state or (out / "futures_state.json")
    state.parent.mkdir(parents=True, exist_ok=True)
    executor = build_executor(state)

    totals = run(source, declared, executor)
    payload = report(totals, declared, source, executor)

    out.mkdir(parents=True, exist_ok=True)
    (out / REPORT_NAME).write_text(json.dumps(payload, indent=2) + "\n")
    (out / STATUS_NAME).write_text(status_markdown(payload))
    logger.info(json.dumps(payload["totals"], indent=2))
    logger.info(
        "engineering smoke only — sustained paper validation is NOT claimed. Wrote %s", out
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
