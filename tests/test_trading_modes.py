"""The trading-mode scaffold: eligibility, transitions, and what it cannot do.

Two properties carry this file. The first is that **a mode is eligible only when
the committed evidence says its specialists are viable** — so a scaffold cannot
start trading a style the research did not support. The second is that
**nothing in the selection path can read a realised return**, which is the
failure `P8` exists to test and has not tested.

Both are asserted structurally rather than by reading prose: the eligibility
table is derived from the committed decision artifacts, and the profit tripwire
is applied to the source of every function that can influence which mode is
entered.
"""

from __future__ import annotations

import inspect
import re
import json
from pathlib import Path

import pytest

from chimera import metrics
from chimera.consensus import decide as consensus_decide
from chimera.contracts import Signal
from chimera.modes import (
    MODE_SPECS,
    PROFIT_TOKENS,
    Eligibility,
    ModeDecision,
    ModeError,
    ReasonCode,
    SpecialistStatus,
    TradingMode,
    assert_no_profit_input,
    decide_mode,
    evaluate_eligibility,
    plan_mode_transition,
    selection_source,
)

REPO = Path(__file__).resolve().parents[1]
DOCUMENT = REPO / "docs" / "trading_modes_v1.md"
BENCHMARK = REPO / "artifacts" / "benchmark"

L, S, H = Signal.LONG, Signal.SHORT, Signal.HOLD

DIRECTIONAL = [TradingMode.SCALPING, TradingMode.DAY_TRADING, TradingMode.SWING]


def committed_status() -> dict[str, SpecialistStatus]:
    """Specialist viability, read from the committed decision artifacts.

    Not a hand-written table: the whole point of eligibility is that it is a
    fact about the evidence tree, so the test derives it from the same artifacts
    a reader would.
    """
    status: dict[str, SpecialistStatus] = {}
    for directory, checkpoint in (("btc_p6_decision", "P6"), ("btc_p6ext_decision", "P6-EXT")):
        payload = json.loads((BENCHMARK / directory / "decision.json").read_text())
        for row in payload["clocks"]:
            status[row["clock"]] = SpecialistStatus(
                clock=row["clock"],
                screened=True,
                viable=bool(row["viable"]),
                checkpoint=checkpoint,
            )
    return status


# --------------------------------------------------------------------------- #
# A. the modes are the ones the documents describe
# --------------------------------------------------------------------------- #


def test_the_four_operating_states():
    assert [mode.value for mode in TradingMode] == [
        "SCALPING",
        "DAY_TRADING",
        "SWING",
        "FLAT",
    ]
    assert set(MODE_SPECS) == set(DIRECTIONAL)
    assert TradingMode.FLAT not in MODE_SPECS


@pytest.mark.parametrize("mode", DIRECTIONAL, ids=[m.value for m in DIRECTIONAL])
def test_each_mode_names_its_clocks_and_decides_on_the_fastest(mode):
    spec = MODE_SPECS[mode]
    assert spec.decision_clock == spec.primary_clocks[0]
    assert spec.decision_clock in spec.primary_clocks
    assert not set(spec.primary_clocks) & set(spec.context_clocks)
    assert spec.rule.specialists == spec.specialists
    assert spec.rule.veto_specialist == spec.specialists[-1]
    assert spec.decision_cadence == f"once per closed {spec.decision_clock} bar"


def test_the_clock_sets_are_the_ones_the_document_states():
    expected = {
        TradingMode.SCALPING: (("1m", "5m"), ("15m",), "1m", 2),
        TradingMode.DAY_TRADING: (("5m", "15m"), ("30m", "1h"), "5m", 3),
        TradingMode.SWING: (("30m", "1h", "4h"), ("1d",), "30m", 3),
    }
    for mode, (primary, context, clock, agreement) in expected.items():
        spec = MODE_SPECS[mode]
        assert spec.primary_clocks == primary
        assert spec.context_clocks == context
        assert spec.decision_clock == clock
        assert spec.rule.agreement_required == agreement


def test_the_two_measured_modes_use_the_rules_p7_measured():
    """The scaffold expresses the rule that was tested, not a second one."""
    from nn.p7_preregistration import DAY_TRADING, SCALPING

    for design, mode in (
        (SCALPING, TradingMode.SCALPING),
        (DAY_TRADING, TradingMode.DAY_TRADING),
    ):
        spec = MODE_SPECS[mode]
        assert list(spec.specialists) == design["specialists"]
        assert spec.rule.agreement_required == design["agreement_required"]
        assert spec.rule.veto_specialist == design["veto_specialist"]
        assert spec.decision_clock == design["decision_clock"]


def test_a_malformed_mode_spec_is_refused():
    from chimera.consensus import ConsensusRule
    from chimera.modes import ModeSpec

    rule = ConsensusRule("X", "1m", ("1m", "5m", "15m"), "15m", 2)
    with pytest.raises(ModeError, match="not among the primary clocks"):
        ModeSpec(TradingMode.SCALPING, "5m", ("1m",), ("15m",), rule, "x")
    with pytest.raises(ModeError, match="both primary and context"):
        ModeSpec(TradingMode.SCALPING, "1m", ("1m", "5m"), ("5m",), rule, "x")
    with pytest.raises(ModeError, match="votes on"):
        ModeSpec(TradingMode.SCALPING, "1m", ("1m",), ("30m",), rule, "x")


# --------------------------------------------------------------------------- #
# B. eligibility comes from the evidence, and currently permits nothing
# --------------------------------------------------------------------------- #


def test_a_specialist_cannot_be_viable_without_being_screened():
    with pytest.raises(ModeError, match="viability is a verdict"):
        SpecialistStatus("1m", screened=False, viable=True)


def test_an_unscreened_clock_makes_a_mode_ineligible():
    status = {clock: SpecialistStatus(clock, True, True) for clock in ("1m", "5m")}
    result = evaluate_eligibility(status, [TradingMode.SCALPING])[TradingMode.SCALPING]
    assert result.eligible is False
    assert result.reason is ReasonCode.SPECIALIST_UNSCREENED
    assert result.unscreened == ("15m",)


def test_a_screened_but_not_viable_clock_makes_a_mode_ineligible():
    status = {clock: SpecialistStatus(clock, True, True) for clock in ("1m", "5m", "15m")}
    status["15m"] = SpecialistStatus("15m", screened=True, viable=False)
    result = evaluate_eligibility(status, [TradingMode.SCALPING])[TradingMode.SCALPING]
    assert result.eligible is False
    assert result.reason is ReasonCode.SPECIALIST_NOT_VIABLE
    assert result.not_viable == ("15m",)


def test_a_mode_is_eligible_only_when_every_specialist_is_viable():
    status = {
        clock: SpecialistStatus(clock, True, True)
        for clock in ("1m", "5m", "15m", "30m", "1h")
    }
    result = evaluate_eligibility(status, [TradingMode.SCALPING, TradingMode.DAY_TRADING])
    assert result[TradingMode.SCALPING].eligible is True
    assert result[TradingMode.DAY_TRADING].eligible is True
    assert result[TradingMode.SCALPING].reason is None


def test_no_directional_mode_is_eligible_on_the_committed_evidence():
    """P6 and P6-EXT found no viable clock, so the scaffold may only be FLAT."""
    status = committed_status()
    assert set(status) == {"1m", "5m", "15m", "30m", "1h", "4h", "1d"}
    assert all(row.screened for row in status.values())
    assert not any(row.viable for row in status.values())

    eligibility = evaluate_eligibility(status)
    for mode in DIRECTIONAL:
        assert eligibility[mode].eligible is False
        assert eligibility[mode].reason is ReasonCode.SPECIALIST_NOT_VIABLE
        assert eligibility[mode].unscreened == ()


def test_swing_would_be_ineligible_even_with_every_short_clock_viable():
    """The mode is defined by the clocks it names; 30m/1h-only is not swing."""
    status = {
        clock: SpecialistStatus(clock, True, True)
        for clock in ("1m", "5m", "15m", "30m", "1h")
    }
    status |= {clock: SpecialistStatus(clock, True, False) for clock in ("4h", "1d")}
    eligibility = evaluate_eligibility(status)
    assert eligibility[TradingMode.SCALPING].eligible is True
    assert eligibility[TradingMode.SWING].eligible is False
    assert set(eligibility[TradingMode.SWING].not_viable) == {"4h", "1d"}


# --------------------------------------------------------------------------- #
# C. the decision, and what it refuses
# --------------------------------------------------------------------------- #


def test_an_ineligible_declaration_yields_flat_with_the_reason():
    eligibility = evaluate_eligibility(committed_status())
    decision = decide_mode(TradingMode.SCALPING, {"1m": L, "5m": L, "15m": L}, eligibility)
    assert decision.mode is TradingMode.FLAT
    assert decision.signal is Signal.HOLD
    assert decision.reason is ReasonCode.SPECIALIST_NOT_VIABLE
    assert decision.eligible_modes == ()
    assert decision.is_flat


def test_no_declaration_yields_flat_and_is_not_an_error():
    decision = decide_mode(None, {}, evaluate_eligibility(committed_status()))
    assert decision.mode is TradingMode.FLAT
    assert decision.reason is ReasonCode.NO_MODE_DECLARED


def test_there_is_no_auto_mode_because_p8_is_not_opened():
    assert not hasattr(TradingMode, "AUTO")
    assert "AUTO" not in {mode.value for mode in TradingMode}
    # The reason code exists so a caller asking for automatic routing can be
    # told why it is refused, rather than silently getting one.
    assert ReasonCode.AUTO_ROUTING_NOT_OPENED.value == "auto_routing_not_opened"
    from nn.research_state import ANSWERED, WITHDRAWN, checkpoint_states

    # `withdrawn` joined the vocabulary when P8 was withdrawn as moot on
    # 2026-09-03. It is still not an answer: the checkpoint was closed without
    # ever being opened, so there is no router and no number either way.
    state = checkpoint_states(REPO)["P8"]
    assert state != ANSWERED
    assert state in {"unrun", "preregistered", WITHDRAWN}


def test_an_eligible_mode_decides_through_the_shared_consensus():
    status = {clock: SpecialistStatus(clock, True, True) for clock in ("1m", "5m", "15m")}
    eligibility = evaluate_eligibility(status, [TradingMode.SCALPING])
    for actions, expected, reason in (
        ({"1m": L, "5m": L, "15m": H}, Signal.LONG, ReasonCode.CONSENSUS_LONG),
        ({"1m": S, "5m": S, "15m": H}, Signal.SHORT, ReasonCode.CONSENSUS_SHORT),
        ({"1m": L, "5m": S, "15m": H}, Signal.HOLD, ReasonCode.NO_CONSENSUS),
        ({"1m": L, "5m": L, "15m": S}, Signal.HOLD, ReasonCode.NO_CONSENSUS),
    ):
        decision = decide_mode(TradingMode.SCALPING, actions, eligibility)
        assert decision.signal is expected
        assert decision.reason is reason
        assert decision.mode is TradingMode.SCALPING
        # The scaffold's answer is the shared decider's answer.
        assert decision.signal is consensus_decide(
            actions, MODE_SPECS[TradingMode.SCALPING].rule
        )


def test_a_missing_specialist_prediction_yields_flat_not_a_partial_vote():
    status = {clock: SpecialistStatus(clock, True, True) for clock in ("1m", "5m", "15m")}
    eligibility = evaluate_eligibility(status, [TradingMode.SCALPING])
    decision = decide_mode(TradingMode.SCALPING, {"1m": L, "5m": L}, eligibility)
    assert decision.mode is TradingMode.FLAT
    assert decision.reason is ReasonCode.SPECIALIST_UNAVAILABLE
    assert decision.consensus_state["unavailable"] == ["15m"]


def test_the_decision_is_deterministic():
    status = {clock: SpecialistStatus(clock, True, True) for clock in ("1m", "5m", "15m")}
    eligibility = evaluate_eligibility(status, [TradingMode.SCALPING])
    actions = {"1m": L, "5m": L, "15m": H}
    first = decide_mode(TradingMode.SCALPING, actions, eligibility)
    second = decide_mode(TradingMode.SCALPING, dict(actions), eligibility)
    assert first.to_dict() == second.to_dict()


# --------------------------------------------------------------------------- #
# D. no profit may reach the selection
# --------------------------------------------------------------------------- #


def test_no_selection_function_can_read_a_realised_return():
    """The property the whole scaffold rests on, applied to the real source."""
    assert_no_profit_input(selection_source())


def test_the_profit_tripwire_actually_fires():
    """A check that has never failed is not evidence."""
    for token in PROFIT_TOKENS:
        with pytest.raises(ModeError, match="may not read"):
            assert_no_profit_input(f"def choose():\n    return best_by_{token}()\n")


def test_the_selection_source_covers_every_function_that_can_choose_a_mode():
    """Discovered, not listed — a list checked against itself proves nothing.

    `selection_source` concatenates three named functions, so asserting those
    three appear in it is true by construction. What has to hold instead is that
    those three are *all* of them: every module-level function in
    `chimera.modes` that can hand back a mode, an eligibility or a transition is
    inside the text the profit tripwire scans. A fourth selector added tomorrow
    and forgotten fails here.
    """
    from chimera import modes as module

    decisive = {"ModeDecision", "Eligibility", "TransitionPlan", "TradingMode"}
    selectors = {
        name
        for name, function in vars(module).items()
        if inspect.isfunction(function)
        and function.__module__ == module.__name__
        and not name.startswith("_")
        and decisive & set(re.findall(r"\w+", str(function.__annotations__.get("return", ""))))
    }
    assert selectors == {
        "evaluate_eligibility",
        "decide_mode",
        "plan_mode_transition",
    }, f"a function that can choose a mode is outside the tripwire's reach: {selectors}"

    # And the private helpers those three delegate to. `_absent` supplies the
    # status of an unreported specialist, which is an eligibility decision.
    assert inspect.getsource(module._absent) in selection_source()

    source = selection_source()
    for name in selectors:
        assert inspect.getsource(getattr(module, name)) in source


def test_a_specialist_status_carries_no_performance_field():
    assert set(SpecialistStatus.__dataclass_fields__) == {
        "clock",
        "screened",
        "viable",
        "checkpoint",
    }


def test_the_decision_carries_only_bounded_values():
    """Everything on a ModeDecision is safe as a metric label."""
    assert set(ModeDecision.__dataclass_fields__) == {
        "mode",
        "signal",
        "reason",
        "eligible_modes",
        "consensus_state",
    }
    assert set(Eligibility.__dataclass_fields__) == {
        "mode",
        "eligible",
        "reason",
        "unscreened",
        "not_viable",
    }


# --------------------------------------------------------------------------- #
# E. transitions
# --------------------------------------------------------------------------- #


def test_no_mode_change_needs_no_plan():
    plan = plan_mode_transition(
        TradingMode.SCALPING, TradingMode.SCALPING, position_is_flat=False
    )
    assert plan.must_flatten is False and plan.must_reconcile is False


def test_a_mode_change_while_flat_unwinds_nothing():
    plan = plan_mode_transition(
        TradingMode.SCALPING, TradingMode.DAY_TRADING, position_is_flat=True
    )
    assert plan.must_flatten is False and plan.must_reconcile is False


def test_a_mode_change_with_an_open_position_flattens_and_reconciles_first():
    """SCALPING LONG -> DAY_TRADING LONG is never a silent inheritance."""
    plan = plan_mode_transition(
        TradingMode.SCALPING, TradingMode.DAY_TRADING, position_is_flat=False
    )
    assert plan.must_flatten is True
    assert plan.must_reconcile is True
    assert "flatten and reconcile" in plan.note


@pytest.mark.parametrize("to_mode", list(TradingMode))
def test_every_transition_is_deterministic(to_mode):
    for flat in (True, False):
        first = plan_mode_transition(TradingMode.SWING, to_mode, position_is_flat=flat)
        second = plan_mode_transition(TradingMode.SWING, to_mode, position_is_flat=flat)
        assert first.to_dict() == second.to_dict()


# --------------------------------------------------------------------------- #
# F. telemetry stays bounded
# --------------------------------------------------------------------------- #


def test_mode_metrics_are_labelled_only_by_bounded_enums():
    import re

    source = (REPO / "chimera" / "metrics.py").read_text()
    block = source.split("# --- trading modes")[1].split("# --- system")[0]
    labels = set(re.findall(r'"(\w+)"\]', block)) | set(re.findall(r'\["(\w+)",', block))
    assert labels <= {"mode", "reason", "from_mode", "to_mode"}
    # A per-mode return would be the selection input the scaffold forbids.
    for token in ("return", "pnl", "profit", "sharpe"):
        assert f"_mode_{token}" not in block


def test_recording_a_decision_and_a_transition_does_not_raise():
    eligibility = evaluate_eligibility(committed_status())
    metrics.mark_mode_decision(decide_mode(TradingMode.SCALPING, {}, eligibility))
    metrics.mark_mode_transition(
        plan_mode_transition(TradingMode.SCALPING, TradingMode.FLAT, position_is_flat=False)
    )


# --------------------------------------------------------------------------- #
# G. the document says what the code does
# --------------------------------------------------------------------------- #


def test_the_document_records_the_current_eligibility_honestly():
    text = DOCUMENT.read_text()
    assert "Current eligibility: none" in text
    assert "never measured" in text
    for mode in DIRECTIONAL:
        assert f"`{mode.value}`" in text
    assert "P8" in text and "not opened" in text


def test_the_document_does_not_claim_alpha_for_any_mode():
    import re

    # Wrapping and emphasis normalised away: the claim is what matters, not how
    # the paragraph happened to break.
    text = re.sub(r"\s+", " ", DOCUMENT.read_text().replace("*", "").replace("`", "")).lower()
    for phrase in ("profitable mode", "proven mode", "this mode makes money"):
        assert phrase not in text
    assert "nothing here claims alpha for any mode" in text
    assert "flat is a first-class successful outcome" in text


# --------------------------------------------------------------------------- #
# H. the telemetry the runbook tells an operator to watch
# --------------------------------------------------------------------------- #


def test_every_mode_series_the_runbook_names_is_written_by_something():
    """A metric nothing writes is a dashboard panel that is always empty."""
    from chimera import metrics

    runbook = (REPO / "docs" / "paper_operation_runbook.md").read_text()
    named = set(re.findall(r"chimera_mode_[a-z_]+", runbook))
    assert named, "the runbook names no mode series at all"

    source = inspect.getsource(metrics) + inspect.getsource(paper_run_module())
    for series in sorted(named):
        # `prometheus_client` appends `_total` to a Counter's exposed name, so
        # the series an operator greps for is not spelled like the constant.
        stem = series[len("chimera_") :]
        attribute = stem[: -len("_total")].upper() if stem.endswith("_total") else stem.upper()
        assert hasattr(metrics, attribute), f"{series} is not defined"
        # Defined is not enough: something has to write it.
        written = re.search(rf"\b{attribute}\.labels\(", source)
        assert written, f"{series} is defined and nothing ever writes it"


def test_selected_and_eligible_come_back_down():
    """Both are gauges, so a mode that stops being selected must read 0.

    A gauge only ever set to 1 is worse than no gauge: the mode that was active
    an hour ago still reads selected, and two modes read selected at once.
    """
    from chimera import metrics

    def value(gauge, mode):
        return gauge.labels(mode=mode)._value.get()

    clocks = tuple(MODE_SPECS[TradingMode.SCALPING].specialists)
    status = {clock: SpecialistStatus(clock, True, True) for clock in clocks}
    eligibility = evaluate_eligibility(status)
    actions = {clock: Signal.LONG for clock in clocks}

    metrics.mark_mode_decision(decide_mode(TradingMode.SCALPING, actions, eligibility))
    assert value(metrics.MODE_SELECTED, "SCALPING") == 1
    assert value(metrics.MODE_SELECTED, "FLAT") == 0
    assert value(metrics.MODE_ELIGIBLE, "SCALPING") == 1

    metrics.mark_mode_decision(decide_mode(TradingMode.FLAT, actions, eligibility))
    assert value(metrics.MODE_SELECTED, "SCALPING") == 0
    assert value(metrics.MODE_SELECTED, "FLAT") == 1

    # And an eligibility that goes away comes back down too.
    dead = {clock: SpecialistStatus(clock, True, False) for clock in clocks}
    metrics.mark_mode_decision(
        decide_mode(TradingMode.SCALPING, actions, evaluate_eligibility(dead))
    )
    assert value(metrics.MODE_ELIGIBLE, "SCALPING") == 0


def paper_run_module():
    from tools import paper_run

    return paper_run
