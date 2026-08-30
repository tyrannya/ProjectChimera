"""The runnable paper path: that it runs, what it refuses, and what it never claims.

Three things are worth testing here and the rest is plumbing.

1. **With no eligible mode the chain runs and places nothing.** That is the
   current, correct behaviour, and a smoke that reported it as a failure would
   push somebody towards making a mode eligible to make the red go away.
2. **With an eligible mode and agreeing specialists the chain actually trades.**
   Otherwise "the path is runnable" would be an untested claim: a loop that never
   reaches Hermes proves nothing about Hermes.
3. **No run can claim to be live, sustained, or evidence about alpha.** The
   report says so in fields, not only in prose, and the live source refuses
   rather than pretending.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from chimera.contracts import Signal
from chimera.modes import SpecialistStatus, TradingMode
from tools import paper_run
from tools.paper_run import (
    LiveSource,
    PaperRunError,
    ReplaySource,
    RunTotals,
    build_executor,
    committed_specialist_status,
    report,
    run,
    status_markdown,
)

REPO = Path(__file__).resolve().parents[1]

L, S, H = Signal.LONG, Signal.SHORT, Signal.HOLD


class ScriptedSource:
    """A source that emits exactly the votes a test wants, on a 1m clock."""

    decision_clock = "1m"

    def __init__(self, script):
        self._script = list(script)

    def bars(self):
        start = pd.Timestamp("2024-01-01T00:00:00Z")
        for index, signals in enumerate(self._script):
            yield (start + pd.Timedelta(minutes=index), 60_000.0, signals)


def viable_status():
    return {
        clock: SpecialistStatus(clock, True, True)
        for clock in ("1m", "5m", "15m", "30m", "1h", "4h", "1d")
    }


# --------------------------------------------------------------------------- #
# A. the current, correct behaviour: nothing is eligible, so nothing trades
# --------------------------------------------------------------------------- #


def test_the_committed_evidence_makes_every_mode_ineligible():
    status = committed_specialist_status()
    assert set(status) == {"1m", "5m", "15m", "30m", "1h", "4h", "1d"}
    assert not any(row.viable for row in status.values())


def test_with_no_eligible_mode_the_chain_runs_and_places_nothing(tmp_path):
    executor = build_executor(tmp_path / "state.json")
    source = ScriptedSource([{"1m": L, "5m": L, "15m": L}] * 20)
    totals = run(source, TradingMode.SCALPING, executor)

    assert totals.bars == 20
    assert totals.modes == {"FLAT": 20}
    assert totals.reasons == {"specialist_not_viable": 20}
    assert totals.orders_planned == 0
    assert totals.orders_filled == 0
    assert totals.to_dict()["flat_fraction"] == 1.0
    assert executor.position(paper_run.SYMBOL).is_flat


# --------------------------------------------------------------------------- #
# B. the path is genuinely wired: with an eligible mode it reaches Hermes
# --------------------------------------------------------------------------- #


def test_an_eligible_mode_with_agreement_actually_trades(tmp_path, monkeypatch):
    """Otherwise 'the path is runnable' would be an untested claim."""
    monkeypatch.setattr(paper_run, "committed_specialist_status", viable_status)
    executor = build_executor(tmp_path / "state.json")
    source = ScriptedSource([{"1m": L, "5m": L, "15m": H}] * 5)
    totals = run(source, TradingMode.SCALPING, executor)

    assert totals.modes == {"SCALPING": 5}
    assert totals.reasons == {"consensus_long": 5}
    assert totals.orders_planned >= 1
    assert totals.orders_filled >= 1
    position = executor.position(paper_run.SYMBOL)
    assert not position.is_flat
    assert position.side.name == "LONG"


def test_a_reversal_goes_through_the_venue_rather_than_flipping_silently(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(paper_run, "committed_specialist_status", viable_status)
    executor = build_executor(tmp_path / "state.json")
    source = ScriptedSource(
        [{"1m": L, "5m": L, "15m": H}] * 3 + [{"1m": S, "5m": S, "15m": H}] * 3
    )
    totals = run(source, TradingMode.SCALPING, executor)
    assert totals.reasons == {"consensus_long": 3, "consensus_short": 3}
    assert executor.position(paper_run.SYMBOL).side.name == "SHORT"
    assert totals.orders_filled >= 2


def test_losing_consensus_returns_the_position_to_flat(tmp_path, monkeypatch):
    """HOLD means flat, not 'do nothing' — the position is closed, not held."""
    monkeypatch.setattr(paper_run, "committed_specialist_status", viable_status)
    executor = build_executor(tmp_path / "state.json")
    source = ScriptedSource(
        [{"1m": L, "5m": L, "15m": H}] * 3 + [{"1m": L, "5m": S, "15m": H}] * 3
    )
    run(source, TradingMode.SCALPING, executor)
    assert executor.position(paper_run.SYMBOL).is_flat


def test_a_missing_specialist_holds_and_never_votes_partially(tmp_path, monkeypatch):
    monkeypatch.setattr(paper_run, "committed_specialist_status", viable_status)
    executor = build_executor(tmp_path / "state.json")
    source = ScriptedSource([{"1m": L, "5m": L, "15m": None}] * 4)
    totals = run(source, TradingMode.SCALPING, executor)
    assert totals.reasons == {"specialist_unavailable": 4}
    assert totals.orders_filled == 0


def test_a_mode_change_with_an_open_position_flattens_first(tmp_path, monkeypatch):
    """The transition semantics, exercised through the real executor."""
    monkeypatch.setattr(paper_run, "committed_specialist_status", viable_status)
    executor = build_executor(tmp_path / "state.json")
    # Agreement, then an unavailable specialist, which drops the mode to FLAT
    # while a position is open.
    source = ScriptedSource(
        [{"1m": L, "5m": L, "15m": H}] * 3 + [{"1m": L, "5m": L, "15m": None}] * 2
    )
    totals = run(source, TradingMode.SCALPING, executor)
    assert totals.transitions >= 1
    assert totals.flattens >= 1
    assert executor.position(paper_run.SYMBOL).is_flat


# --------------------------------------------------------------------------- #
# C. the replay source is causal
# --------------------------------------------------------------------------- #


def test_the_replay_source_holds_where_a_slower_specialist_has_not_closed():
    source = ReplaySource("1m", ("1m", "5m", "15m"), limit=30, fold=0)
    rows = list(source.bars())
    assert len(rows) == 30
    # The first fourteen minutes of an outer block precede the first closed 15m
    # bar inside it, exactly as P7's measured availability recorded.
    unavailable = [
        index for index, (_, _, signals) in enumerate(rows) if signals["15m"] is None
    ]
    assert unavailable == list(range(14))
    assert all(signals["1m"] is not None for _, _, signals in rows)


def test_the_replay_source_is_deterministic():
    first = list(ReplaySource("5m", ("5m", "15m", "30m", "1h"), limit=50, fold=1).bars())
    second = list(ReplaySource("5m", ("5m", "15m", "30m", "1h"), limit=50, fold=1).bars())
    assert first == second


def test_the_replay_price_is_constant_so_no_pnl_can_be_read_from_it():
    prices = {
        price for _, price, _ in ReplaySource("1m", ("1m", "5m", "15m"), limit=40).bars()
    }
    assert prices == {paper_run.REPLAY_REFERENCE_PRICE}


# --------------------------------------------------------------------------- #
# D. what it refuses, and what it never claims
# --------------------------------------------------------------------------- #


def test_a_live_run_refuses_and_says_what_is_missing():
    with pytest.raises(PaperRunError, match="did not persist estimators"):
        LiveSource("1m", ("1m", "5m", "15m")).bars()


def test_the_report_denies_every_claim_it_could_be_mistaken_for(tmp_path):
    executor = build_executor(tmp_path / "state.json")
    source = ScriptedSource([{"1m": H, "5m": H, "15m": H}] * 3)
    totals = run(source, TradingMode.SCALPING, executor)
    payload = report(totals, TradingMode.SCALPING, source, executor)

    assert payload["claims"] == {
        "sustained_paper_validation": False,
        "live": False,
        "real_money": False,
        "alpha": False,
    }
    assert "NOT sustained paper validation" in payload["run_class"]
    assert payload["execution"]["dry_run"] is True
    assert payload["execution"]["leverage"] == "1"
    assert payload["execution"]["margin_mode"] == "ISOLATED"
    assert payload["execution"]["venue"] == "DryRunFuturesVenue"


def test_the_status_marker_says_sustained_validation_is_not_claimed(tmp_path):
    executor = build_executor(tmp_path / "state.json")
    source = ScriptedSource([{"1m": H, "5m": H, "15m": H}])
    payload = report(
        run(source, TradingMode.SCALPING, executor), TradingMode.SCALPING, source, executor
    )
    text = status_markdown(payload)
    assert text.startswith("# OPERATIONAL")
    assert "not** claimed" in text or "not claimed" in text.replace("**", "")
    assert "paper_operation_runbook" in text


def test_the_executor_cannot_be_built_live(tmp_path):
    from chimera.futures import FuturesExecutionConfig, LiveFuturesNotImplemented

    with pytest.raises(LiveFuturesNotImplemented):
        FuturesExecutionConfig(dry_run=False)
    executor = build_executor(tmp_path / "state.json")
    assert executor.config.dry_run is True
    assert executor.config.leverage == Decimal("1")


def test_totals_report_no_return_of_any_kind():
    """A per-mode return would be the profit input the scaffold forbids."""
    keys = set(RunTotals().to_dict())
    for token in ("pnl", "return", "profit", "sharpe"):
        assert not any(token in key for key in keys), keys


def test_the_committed_smoke_report_if_present_denies_the_same_claims():
    path = REPO / "artifacts" / "paper_smoke" / "paper_run.json"
    if not path.is_file():
        pytest.skip("no smoke report is committed")
    payload = json.loads(path.read_text())
    assert payload["claims"]["sustained_paper_validation"] is False
    assert payload["claims"]["live"] is False
    assert payload["claims"]["alpha"] is False
    assert payload["execution"]["dry_run"] is True
    assert payload["totals"]["orders_filled"] == 0


def test_zero_bars_means_the_whole_fold_not_an_empty_run():
    """The runbook's long-soak command passes `--bars 0`."""
    from tools.paper_run import bar_limit, build_argparser

    assert bar_limit(0) is None
    assert bar_limit(-1) is None
    assert bar_limit(500) == 500
    assert build_argparser().parse_args(["--smoke", "--bars", "0"]).bars == 0

    unbounded = ReplaySource("1m", ("1m", "5m", "15m"), limit=bar_limit(0), fold=0)
    bounded = ReplaySource("1m", ("1m", "5m", "15m"), limit=bar_limit(20), fold=0)
    assert sum(1 for _ in unbounded.bars()) > sum(1 for _ in bounded.bars()) == 20
