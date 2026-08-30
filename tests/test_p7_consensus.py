"""P7's consensus battery, and a positive control for every item in it.

`docs/p7_preregistration.md` §9 declares ten properties that must hold before any
P7 delta is read, and declares that each carries a positive control: a test that
deliberately introduces the fault and asserts the check catches it. That second
half is the load-bearing one — **a check that has never failed is not evidence**.

The rule itself is `chimera.consensus.decide`, which is the same function the
live trading-mode controller calls. Testing it here tests the thing that would
run, not a research copy of it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from chimera.consensus import ConsensusError, ConsensusRule, decide, explain
from chimera.contracts import HOLD_IDX, LONG_IDX, SHORT_IDX, Signal
from nn.p7 import (
    UNAVAILABLE,
    P7Error,
    align_to_decision_clock,
    aligned_actions,
    consensus_signals,
    constituent_signals,
    load_specialist,
    rule_for,
    validity_gate,
)
from nn.p7_preregistration import DAY_TRADING, MODES, SCALPING

SCALP = rule_for(SCALPING)
DAY = rule_for(DAY_TRADING)

L, S, H = Signal.LONG, Signal.SHORT, Signal.HOLD


def frame(clock_opens, actions, fold=0):
    """A minimal specialist prediction frame."""
    return pd.DataFrame(
        {
            "fold": fold,
            "timestamp": pd.DatetimeIndex(clock_opens, tz="UTC"),
            "row_index": np.arange(len(actions), dtype=np.int64),
            "future_return": np.zeros(len(actions)),
            "selected_action": np.asarray(actions, dtype=np.int64),
        }
    )


# --------------------------------------------------------------------------- #
# The rule: C7, C8, C9 and the guards on the rule itself
# --------------------------------------------------------------------------- #


def test_c7_hold_propagates_no_agreement_no_position():
    assert decide({"1m": H, "5m": H, "15m": H}, SCALP) is Signal.HOLD
    assert decide({"1m": L, "5m": H, "15m": H}, SCALP) is Signal.HOLD


def test_long_agreement_reaches_the_threshold():
    assert decide({"1m": L, "5m": L, "15m": H}, SCALP) is Signal.LONG
    assert decide({"1m": L, "5m": L, "15m": L}, SCALP) is Signal.LONG


def test_short_agreement_reaches_the_threshold():
    assert decide({"1m": S, "5m": S, "15m": H}, SCALP) is Signal.SHORT
    assert decide({"1m": S, "5m": S, "15m": S}, SCALP) is Signal.SHORT


def test_c8_the_slow_specialist_vetoes_agreement_it_disagrees_with():
    """Two of three LONG, but the 15m specialist is actively SHORT."""
    assert decide({"1m": L, "5m": L, "15m": S}, SCALP) is Signal.HOLD
    assert decide({"1m": S, "5m": S, "15m": L}, SCALP) is Signal.HOLD
    # And the veto is not a vote: a HOLD from it blocks nothing.
    assert decide({"1m": L, "5m": L, "15m": H}, SCALP) is Signal.LONG


def test_c9_disagreement_yields_hold():
    assert decide({"1m": L, "5m": S, "15m": H}, SCALP) is Signal.HOLD
    assert decide({"5m": L, "15m": S, "30m": L, "1h": H}, DAY) is Signal.HOLD


def test_day_trading_needs_three_of_four():
    assert decide({"5m": L, "15m": L, "30m": H, "1h": H}, DAY) is Signal.HOLD
    assert decide({"5m": L, "15m": L, "30m": L, "1h": H}, DAY) is Signal.LONG
    assert decide({"5m": L, "15m": L, "30m": L, "1h": S}, DAY) is Signal.HOLD


def test_c4_a_missing_specialist_holds_rather_than_voting_partially():
    assert decide({"1m": L, "5m": L, "15m": None}, SCALP) is Signal.HOLD
    assert decide({"1m": L, "5m": L}, SCALP) is Signal.HOLD
    assert decide({"5m": L, "15m": L, "30m": L, "1h": None}, DAY) is Signal.HOLD


def test_a_rule_that_could_need_a_tie_break_is_refused():
    with pytest.raises(ConsensusError, match="strict majority"):
        ConsensusRule("X", "1m", ("1m", "5m"), "5m", 1)
    with pytest.raises(ConsensusError, match="does not vote"):
        ConsensusRule("X", "1m", ("1m", "5m", "15m"), "1h", 2)
    with pytest.raises(ConsensusError, match="not among the specialists"):
        ConsensusRule("X", "30m", ("1m", "5m", "15m"), "15m", 2)
    with pytest.raises(ConsensusError, match="may not vote twice"):
        ConsensusRule("X", "1m", ("1m", "1m", "15m"), "15m", 2)


def test_explain_reports_bounded_values_and_agrees_with_decide():
    record = explain({"1m": L, "5m": L, "15m": S}, SCALP)
    assert record["signal"] == "HOLD"
    assert record["long_votes"] == 2 and record["short_votes"] == 1
    assert record["veto_blocked"] is True and record["agreement_reached"] is True
    assert record["unavailable"] == 0
    for combination in (
        {"1m": L, "5m": L, "15m": H},
        {"1m": H, "5m": H, "15m": H},
        {"1m": S, "5m": S, "15m": None},
    ):
        assert explain(combination, SCALP)["signal"] == decide(combination, SCALP).value


# --------------------------------------------------------------------------- #
# The alignment: C1, C2, C3
# --------------------------------------------------------------------------- #


def test_c3_the_own_clock_alignment_is_the_identity():
    opens = pd.date_range("2024-01-01", periods=20, freq="1min", tz="UTC")
    as_of = align_to_decision_clock(
        opens.to_numpy(dtype="datetime64[ns]"),
        "1m",
        opens.to_numpy(dtype="datetime64[ns]"),
        "1m",
    )
    assert np.array_equal(as_of, np.arange(20))


def test_c2_a_row_landing_exactly_on_a_specialist_close_sees_that_bar():
    """The boundary case, asserted in both directions.

    A 1m row opening at 10:04 closes at 10:05, which is exactly when the 5m bar
    opening at 10:00 closes. It sees it. The row opening at 10:03 does not.
    """
    minutes = pd.date_range("2024-01-01 10:00", periods=10, freq="1min", tz="UTC")
    fives = pd.date_range("2024-01-01 09:55", periods=3, freq="5min", tz="UTC")
    as_of = align_to_decision_clock(
        minutes.to_numpy(dtype="datetime64[ns]"),
        "1m",
        fives.to_numpy(dtype="datetime64[ns]"),
        "5m",
    )
    at = {stamp.strftime("%H:%M"): index for stamp, index in zip(minutes, as_of)}
    # 09:55 bar closes 10:00; 10:00 bar closes 10:05.
    assert at["10:03"] == 0, "a row closing at 10:04 may not see a bar that closes at 10:05"
    assert at["10:04"] == 1, "a row closing at 10:05 must see the bar that closed at 10:05"
    assert at["10:08"] == 1
    assert at["10:09"] == 2


def test_c1_a_specialist_shifted_one_bar_earlier_changes_the_alignment():
    """The positive control: pretending a bar closed sooner is detectable."""
    minutes = pd.date_range("2024-01-01 10:00", periods=30, freq="1min", tz="UTC")
    fives = pd.date_range("2024-01-01 10:00", periods=6, freq="5min", tz="UTC")
    honest = align_to_decision_clock(
        minutes.to_numpy(dtype="datetime64[ns]"),
        "1m",
        fives.to_numpy(dtype="datetime64[ns]"),
        "5m",
    )
    leaked = align_to_decision_clock(
        minutes.to_numpy(dtype="datetime64[ns]"),
        "1m",
        (fives - pd.Timedelta(minutes=5)).to_numpy(dtype="datetime64[ns]"),
        "5m",
    )
    assert not np.array_equal(honest, leaked)
    # The leak is that a row now reads a bar one period ahead of the honest one.
    # It saturates at the last available bar, so the relation is checked where a
    # next bar exists and the direction is checked everywhere.
    assert (leaked >= honest).all(), "a shifted specialist may only move the join forward"
    interior = honest < len(fives) - 1
    assert np.array_equal(leaked[interior], honest[interior] + 1)


def test_no_decision_row_ever_reads_a_bar_that_had_not_closed():
    """The invariant behind C1, checked over every pair of clocks used by P7."""
    for design in MODES:
        rule = rule_for(design)
        decision_opens = pd.date_range(
            "2024-03-01", periods=500, freq=f"{_minutes(rule.decision_clock)}min", tz="UTC"
        )
        for clock in rule.specialists:
            spec_opens = pd.date_range(
                "2024-03-01", periods=500, freq=f"{_minutes(clock)}min", tz="UTC"
            )
            as_of = align_to_decision_clock(
                decision_opens.to_numpy(dtype="datetime64[ns]"),
                rule.decision_clock,
                spec_opens.to_numpy(dtype="datetime64[ns]"),
                clock,
            )
            available = as_of >= 0
            spec_close = spec_opens[as_of[available]] + pd.Timedelta(minutes=_minutes(clock))
            reference = decision_opens[available] + pd.Timedelta(
                minutes=_minutes(rule.decision_clock)
            )
            assert (spec_close <= reference).all()


def _minutes(clock: str) -> int:
    from nn.multiclock import constituent_count

    return constituent_count(clock)


def test_c4_control_a_specialist_truncated_at_the_block_head_holds():
    minutes = pd.date_range("2024-01-01 10:00", periods=20, freq="1min", tz="UTC")
    decision = frame(minutes, [LONG_IDX] * 20)
    # A 15m specialist whose first bar opens at 10:00 has closed nothing until
    # 10:15, so the first fourteen decision rows have it unavailable.
    slow = frame(
        pd.date_range("2024-01-01 10:00", periods=2, freq="15min", tz="UTC"),
        [LONG_IDX, LONG_IDX],
    )
    actions = aligned_actions(decision, slow, "1m", "15m")
    assert int((actions == UNAVAILABLE).sum()) == 14
    fast = aligned_actions(decision, decision, "1m", "1m")
    signals = consensus_signals({"1m": fast, "5m": fast, "15m": actions}, SCALP)
    assert (signals[:14] == HOLD_IDX).all(), "a missing veto specialist must not permit a vote"
    assert (signals[14:] == LONG_IDX).all()
    # The constituent replay of the same specialist holds where it is unavailable.
    assert (constituent_signals(actions)[:14] == HOLD_IDX).all()


# --------------------------------------------------------------------------- #
# Determinism and ordering: C5, C6, C10
# --------------------------------------------------------------------------- #


def test_c6_equal_input_gives_equal_output():
    rng = np.random.default_rng(7)
    actions = {clock: rng.integers(0, 3, size=5_000) for clock in SCALP.specialists}
    first = consensus_signals(actions, SCALP)
    second = consensus_signals({k: v.copy() for k, v in actions.items()}, SCALP)
    assert np.array_equal(first, second)


def test_c10_row_order_does_not_change_a_row_s_decision():
    """A shuffled input reproduces the same per-row consensus."""
    rng = np.random.default_rng(11)
    n = 2_000
    actions = {clock: rng.integers(0, 3, size=n) for clock in SCALP.specialists}
    straight = consensus_signals(actions, SCALP)
    order = rng.permutation(n)
    shuffled = consensus_signals({k: v[order] for k, v in actions.items()}, SCALP)
    assert np.array_equal(shuffled, straight[order])


def test_the_vectorised_rule_agrees_with_the_row_by_row_rule():
    """`consensus_signals` is the shared decider, not a re-implementation of it."""
    rng = np.random.default_rng(3)
    n = 3_000
    actions = {clock: rng.integers(-1, 3, size=n) for clock in SCALP.specialists}
    vectorised = consensus_signals(actions, SCALP)
    lookup = {SHORT_IDX: Signal.SHORT, HOLD_IDX: Signal.HOLD, LONG_IDX: Signal.LONG}
    for position in range(0, n, 37):
        row = {
            clock: (
                None
                if actions[clock][position] == UNAVAILABLE
                else lookup[int(actions[clock][position])]
            )
            for clock in SCALP.specialists
        }
        expected = decide(row, SCALP)
        assert (
            vectorised[position]
            == {Signal.SHORT: SHORT_IDX, Signal.HOLD: HOLD_IDX, Signal.LONG: LONG_IDX}[
                expected
            ]
        )


def test_c5_duplicate_specialist_predictions_are_refused(tmp_path, monkeypatch):
    minutes = pd.date_range("2024-01-01 10:00", periods=5, freq="1min", tz="UTC")
    duplicated = pd.concat([frame(minutes, [LONG_IDX] * 5), frame(minutes[[2]], [SHORT_IDX])])
    path = tmp_path / "outer_predictions.parquet"
    duplicated.to_parquet(path, index=False)
    monkeypatch.setattr("nn.p7.specialist_path", lambda clock: path)
    with pytest.raises(P7Error, match="duplicate"):
        load_specialist("1m")


def test_an_unknown_action_code_is_refused(tmp_path, monkeypatch):
    minutes = pd.date_range("2024-01-01 10:00", periods=4, freq="1min", tz="UTC")
    bad = frame(minutes, [LONG_IDX, HOLD_IDX, 9, SHORT_IDX])
    path = tmp_path / "outer_predictions.parquet"
    bad.to_parquet(path, index=False)
    monkeypatch.setattr("nn.p7.specialist_path", lambda clock: path)
    with pytest.raises(P7Error, match="not in CLASS_ORDER"):
        load_specialist("1m")


# --------------------------------------------------------------------------- #
# C3's teeth: the validity gate refuses a replay that is not the identity
# --------------------------------------------------------------------------- #


def test_the_validity_gate_refuses_a_non_identity_replay():
    minutes = pd.date_range("2024-01-01 10:00", periods=10, freq="1min", tz="UTC")
    decision = frame(minutes, [LONG_IDX] * 10)
    honest = aligned_actions(decision, decision, "1m", "1m")
    assert validity_gate(decision, honest, "1m")["identity_rows"] == 10

    shifted = np.roll(honest, 1)
    shifted[0] = SHORT_IDX
    with pytest.raises(P7Error, match="does not align to itself as the identity"):
        validity_gate(decision, shifted, "1m")


def test_a_mode_declaring_another_horizon_is_refused_rather_than_scored_at_six():
    """The declared field is used, so a mode cannot declare one and get another."""
    from nn.p7 import P7Error, run_mode

    design = dict(SCALPING)
    design["horizon_bars"] = 12
    with pytest.raises(P7Error, match="A replay is not a refit"):
        run_mode(design)


def test_the_measured_staleness_is_the_one_this_document_publishes():
    """The alignment's unbounded staleness, recomputed from the frozen cells.

    P7 v1 has no staleness bound and the closure of `docs/p7_preregistration.md`
    says so with numbers. A disclosure that nothing recomputes is a sentence, so
    this recomputes it: the worst age of an aligned vote, per mode and
    specialist, must be what the document publishes, and zero on each mode's own
    decision clock.
    """
    import numpy as np

    from nn.multiclock import constituent_count
    from nn.p7 import align_to_decision_clock, load_specialist

    expected = {
        ("SCALPING", "1m"): "0 days 00:00:00",
        ("SCALPING", "5m"): "0 days 09:29:00",
        ("SCALPING", "15m"): "1 days 00:14:00",
        ("DAY_TRADING", "5m"): "0 days 00:00:00",
        ("DAY_TRADING", "15m"): "1 days 00:10:00",
        ("DAY_TRADING", "30m"): "1 days 22:55:00",
        ("DAY_TRADING", "1h"): "3 days 20:55:00",
    }

    for design in (SCALPING, DAY_TRADING):
        clock = design["decision_clock"]
        frames = {name: load_specialist(name) for name in design["specialists"]}
        for name in design["specialists"]:
            worst = pd.Timedelta(0)
            for fold in sorted(frames[clock]["fold"].unique()):
                decision = frames[clock].loc[frames[clock]["fold"] == fold]
                specialist = frames[name].loc[frames[name]["fold"] == fold]
                opens = decision["timestamp"].to_numpy(dtype="datetime64[ns]")
                theirs = specialist["timestamp"].to_numpy(dtype="datetime64[ns]")
                index = align_to_decision_clock(opens, clock, theirs, name)
                available = index >= 0
                reference = opens[available] + np.timedelta64(constituent_count(clock), "m")
                closes = theirs[index[available]] + np.timedelta64(
                    constituent_count(name), "m"
                )
                ages = pd.to_timedelta(reference - closes)
                assert (ages >= pd.Timedelta(0)).all(), "a vote from the future"
                if len(ages):
                    worst = max(worst, ages.max())
            assert str(worst) == expected[(design["mode"], name)], (
                f"{design['mode']} {name}: worst staleness is {worst} and the closure "
                f"publishes {expected[(design['mode'], name)]}"
            )
