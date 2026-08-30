"""Unit coverage for the two P6 modules that decide things.

:mod:`tests.test_p6_evidence` checks the committed evidence, and it can only
check the paths the committed evidence happened to take. What it cannot check is
what these modules do when handed something wrong: a clock whose coverage does
not reach a frozen period, two directories claiming the same cell, cells fitted
under two different designs, a fold reporting an outer block outside the window
it was supposed to score.

Every one of those is a refusal in the source, and a refusal nothing exercises is
a comment. These tests exercise them, on synthetic inputs, so the checks are
known to fire rather than assumed to.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from nn.p6 import ARTIFACT_NAME, manifest_label, plan_folds, registration
from nn.p6_decision import (
    DecisionError,
    _fold_rows,
    check_cells_agree,
    decide,
    load_cells,
    secondary_context,
    verdict_for,
)
from nn.p6_preregistration import (
    FOLD_PERIODS,
    MODELS,
    PRIMARY_MODEL,
    VIABILITY_GATE,
    preregistration_hash,
)

REPO = Path(__file__).resolve().parents[1]
P6 = registration("p6")

FIRST_TRAIN = pd.Timestamp(FOLD_PERIODS[0]["train_start"])
LAST_OUTER_END = pd.Timestamp(FOLD_PERIODS[-1]["outer_end"])


def hourly(start=FIRST_TRAIN, end=LAST_OUTER_END):
    """A clock whose bars cover the frozen periods exactly once an hour."""
    return SimpleNamespace(
        dates=pd.date_range(start, end, freq="1h", inclusive="left").to_numpy()
    )


# --------------------------------------------------------------------------- #
# A. plan_folds: the frozen calendar, resolved into one clock's rows
# --------------------------------------------------------------------------- #


def test_the_four_frozen_periods_resolve_to_the_instants_they_name():
    data = hourly()
    dates = pd.DatetimeIndex(pd.to_datetime(data.dates, utc=True))
    plans = plan_folds(data, "1h")

    assert len(plans) == len(FOLD_PERIODS)
    for plan, frozen in zip(plans, FOLD_PERIODS):
        assert dates[plan.train.start] == pd.Timestamp(frozen["train_start"])
        assert dates[plan.inner.start] == pd.Timestamp(frozen["inner_start"])
        assert dates[plan.outer.start] == pd.Timestamp(frozen["outer_start"])
        # Half-open: the last scored bar opens strictly before the period ends.
        assert dates[plan.outer.end - 1] < pd.Timestamp(frozen["outer_end"])
        # Contiguous, in order, and no block reaches past the next one's start.
        assert plan.train.end == plan.inner.start
        assert plan.inner.end == plan.outer.start


def test_the_outer_blocks_tile_forward_and_never_overlap():
    plans = plan_folds(hourly(), "1h")
    for earlier, later in zip(plans, plans[1:]):
        assert later.outer.start == earlier.outer.end


def test_a_faster_clock_gets_more_rows_in_the_same_windows():
    """Rows are not folds: a 12x faster clock still gets four blocks."""
    slow = plan_folds(hourly(), "1h")
    fast = plan_folds(
        SimpleNamespace(
            dates=pd.date_range(
                FIRST_TRAIN, LAST_OUTER_END, freq="5min", inclusive="left"
            ).to_numpy()
        ),
        "5m",
    )
    assert len(fast) == len(slow) == 4
    for quick, plodding in zip(fast, slow):
        quick_rows = quick.outer.end - quick.outer.start
        slow_rows = plodding.outer.end - plodding.outer.start
        assert quick_rows == pytest.approx(slow_rows * 12, rel=1e-6)


def test_a_clock_whose_coverage_stops_early_is_refused_not_truncated():
    short = hourly(end=pd.Timestamp(FOLD_PERIODS[2]["outer_start"]))
    with pytest.raises(SystemExit, match="block is empty"):
        plan_folds(short, "1h")


def test_a_clock_that_starts_after_a_training_block_is_refused():
    late = hourly(start=pd.Timestamp(FOLD_PERIODS[0]["inner_start"]))
    with pytest.raises(SystemExit, match="block is empty"):
        plan_folds(late, "1h")


def test_unordered_timestamps_are_refused_before_any_searchsorted():
    data = hourly()
    scrambled = data.dates.copy()
    scrambled[[10, 20]] = scrambled[[20, 10]]
    with pytest.raises(SystemExit, match="not increasing"):
        plan_folds(SimpleNamespace(dates=scrambled), "1h")


def test_the_last_scored_bar_stops_before_the_research_boundary():
    from nn.multiclock import RESEARCH_VISIBLE_END

    data = hourly()
    dates = pd.DatetimeIndex(pd.to_datetime(data.dates, utc=True))
    plans = plan_folds(data, "1h")
    assert dates[plans[-1].outer.end - 1] < RESEARCH_VISIBLE_END


def test_a_cell_names_the_manifest_it_was_produced_from():
    inside = REPO / "data" / "research" / "btc_usdt_multiclock_gen2_manifest.json"
    assert manifest_label(inside) == "data/research/btc_usdt_multiclock_gen2_manifest.json"
    assert manifest_label(Path("/etc/hostname")) == "/etc/hostname"


# --------------------------------------------------------------------------- #
# B. synthetic cells
# --------------------------------------------------------------------------- #


def fold_record(index: int, net: float, momentum: float, model: str) -> dict:
    frozen = FOLD_PERIODS[index]
    # Spelled the way pandas renders a timestamp, which is not how the frozen
    # periods are spelled. The decision module has to parse rather than compare.
    start = str(pd.Timestamp(frozen["outer_start"]))
    end = str(pd.Timestamp(frozen["outer_end"]) - pd.Timedelta(hours=1))
    return {
        "fold": index,
        "periods": {"outer_validation": {"start": start, "end": end}},
        "model": {"selection": {"threshold": 0.5}},
        "outer_validation": {
            model: {"trading": {"net_return": net, "n_trades": 10, "turnover": 20.0}},
            "momentum_baseline": {"trading": {"net_return": momentum}},
            "economic_references": {"buy_and_hold": {"net_return": 0.5}},
        },
    }


def cell(clock: str, model: str, returns, momentum=-1.0, **overrides) -> dict:
    payload = {
        "checkpoint": "P6",
        "clock": clock,
        "model": model,
        "horizon": P6.horizons[clock],
        "preregistration_hash": preregistration_hash(),
        "source": {"minutes_digest": "d" * 64, "clock_digest": "c" * 64},
        "folds": [
            fold_record(index, net, momentum, model) for index, net in enumerate(returns)
        ],
        "_dir": f"synthetic/{clock}_{model}",
    }
    payload.update(overrides)
    return payload


def every_cell(returns_by_clock: dict[str, list[float]]) -> dict[tuple[str, str], dict]:
    return {
        (clock, model): cell(clock, model, returns)
        for clock, returns in returns_by_clock.items()
        for model in MODELS
    }


PASSING = [0.10, 0.05, -0.01, 0.04]  # 3 positive, mean > 0, beats momentum 4/4
FAILING = [0.10, -0.20, -0.01, 0.04]  # 2 positive, mean < 0


# --------------------------------------------------------------------------- #
# C. verdict_for: the three conditions, and their conjunction
# --------------------------------------------------------------------------- #


def test_a_cell_clearing_all_three_conditions_is_viable():
    verdict = verdict_for(cell("1h", PRIMARY_MODEL, PASSING))
    assert verdict["viable"] is True
    assert [item["passed"] for item in verdict["conditions"].values()] == [True] * 3
    assert verdict["conditions"]["positive_folds"]["observed"] == 3
    assert verdict["conditions"]["beats_native_momentum_folds"]["observed"] == 4


@pytest.mark.parametrize(
    "returns, momentum, failed",
    [
        ([0.10, -0.05, -0.01, 0.04], -1.0, "positive_folds"),
        ([0.01, 0.01, 0.01, -0.20], -1.0, "mean_outer_net_return"),
        ([0.10, 0.05, -0.01, 0.04], 0.09, "beats_native_momentum_folds"),
    ],
)
def test_failing_any_one_condition_fails_the_conjunction(returns, momentum, failed):
    verdict = verdict_for(cell("1h", PRIMARY_MODEL, returns, momentum=momentum))
    assert verdict["viable"] is False
    assert verdict["conditions"][failed]["passed"] is False
    others = [name for name in verdict["conditions"] if name != failed]
    assert all(verdict["conditions"][name]["passed"] for name in others)


def test_the_gate_thresholds_come_from_the_preregistration_not_from_here():
    verdict = verdict_for(cell("1h", PRIMARY_MODEL, PASSING))
    assert verdict["conditions"]["positive_folds"]["required"] == (
        VIABILITY_GATE["positive_folds_required"]
    )
    assert verdict["conditions"]["beats_native_momentum_folds"]["required"] == (
        VIABILITY_GATE["beats_momentum_folds_required"]
    )


def test_a_zero_return_fold_is_not_a_positive_fold():
    verdict = verdict_for(cell("1h", PRIMARY_MODEL, [0.0, 0.0, 0.0, 0.0]))
    assert verdict["conditions"]["positive_folds"]["observed"] == 0
    assert verdict["viable"] is False


def test_the_descriptive_block_decides_nothing():
    verdict = verdict_for(cell("1h", PRIMARY_MODEL, PASSING))
    assert verdict["descriptive"]["worst_fold"] == min(PASSING)
    assert verdict["descriptive"]["best_fold"] == max(PASSING)
    assert "decide nothing" in verdict["descriptive"]["note"]


def test_a_cell_whose_outer_block_lacks_its_own_model_is_refused():
    broken = cell("1h", PRIMARY_MODEL, PASSING)
    del broken["folds"][2]["outer_validation"][PRIMARY_MODEL]
    with pytest.raises(DecisionError, match="the outer block reports"):
        _fold_rows(broken)


def test_a_cell_with_no_momentum_floor_is_refused_rather_than_scored_alone():
    broken = cell("1h", PRIMARY_MODEL, PASSING)
    del broken["folds"][0]["outer_validation"]["momentum_baseline"]
    with pytest.raises(DecisionError, match="momentum_baseline"):
        _fold_rows(broken)


# --------------------------------------------------------------------------- #
# D. load_cells and check_cells_agree: what the decision refuses to decide on
# --------------------------------------------------------------------------- #


def write_cell(root: Path, name: str, payload: dict) -> Path:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    body = {key: value for key, value in payload.items() if key != "_dir"}
    (directory / ARTIFACT_NAME).write_text(json.dumps(body, indent=2))
    return directory


def committed_dirs(tmp_path: Path, cells: dict[tuple[str, str], dict]) -> list[Path]:
    return [
        write_cell(tmp_path, f"btc_p6_{clock}_{model}", payload)
        for (clock, model), payload in cells.items()
    ]


def full_set() -> dict[tuple[str, str], dict]:
    return every_cell({clock: list(FAILING) for clock in P6.clocks})


def test_the_full_cell_set_loads_and_agrees(tmp_path):
    dirs = committed_dirs(tmp_path, full_set())
    loaded = load_cells(dirs, P6)
    assert len(loaded) == len(P6.clocks) * len(MODELS)
    identity = check_cells_agree(loaded, P6)
    assert identity["cells"] == len(loaded)
    assert identity["preregistration_hash"] == preregistration_hash()
    assert set(identity["clock_digests"]) == set(P6.clocks)


def test_a_missing_cell_is_refused_rather_than_decided_on_a_subset(tmp_path):
    cells = full_set()
    del cells[(P6.clocks[-1], "lightgbm")]
    dirs = committed_dirs(tmp_path, cells)
    with pytest.raises(DecisionError, match="winner-shopping"):
        load_cells(dirs, P6)


def test_two_directories_claiming_one_cell_are_refused(tmp_path):
    cells = full_set()
    dirs = committed_dirs(tmp_path, cells)
    duplicate = cell("1h", PRIMARY_MODEL, PASSING)
    dirs.append(write_cell(tmp_path, "btc_p6_1h_xgboost_rerun", duplicate))
    with pytest.raises(DecisionError, match="both claim"):
        load_cells(dirs, P6)


def test_another_checkpoints_cell_is_refused(tmp_path):
    cells = full_set()
    dirs = committed_dirs(tmp_path, cells)
    foreign = cell("1h", PRIMARY_MODEL, PASSING, checkpoint="P6-EXT")
    foreign["clock"] = "4h"
    dirs.append(write_cell(tmp_path, "btc_p6ext_4h_xgboost", foreign))
    with pytest.raises(DecisionError, match="refuses another checkpoint"):
        load_cells(dirs, P6)


def test_a_directory_without_an_artifact_is_skipped_not_guessed(tmp_path):
    dirs = committed_dirs(tmp_path, full_set())
    empty = tmp_path / "btc_p6_notes"
    empty.mkdir()
    dirs.append(empty)
    assert len(load_cells(dirs, P6)) == len(P6.clocks) * len(MODELS)


def test_cells_fitted_under_two_designs_may_not_be_decided_together():
    cells = full_set()
    cells[("1h", PRIMARY_MODEL)]["preregistration_hash"] = "sha256:" + "0" * 64
    with pytest.raises(DecisionError, match="an edited design"):
        check_cells_agree(cells, P6)


def test_cells_read_from_two_sources_may_not_be_decided_together():
    cells = full_set()
    cells[("5m", "lightgbm")]["source"]["minutes_digest"] = "e" * 64
    with pytest.raises(DecisionError, match="different 1m sources"):
        check_cells_agree(cells, P6)


def test_a_fold_scored_outside_its_frozen_window_is_refused():
    cells = full_set()
    record = cells[("15m", PRIMARY_MODEL)]["folds"][1]
    record["periods"]["outer_validation"]["start"] = str(
        pd.Timestamp(FOLD_PERIODS[1]["outer_start"]) - pd.Timedelta(hours=1)
    )
    with pytest.raises(DecisionError, match="not inside the frozen period"):
        check_cells_agree(cells, P6)


def test_a_fold_scored_past_the_end_of_its_window_is_refused():
    cells = full_set()
    record = cells[("30m", PRIMARY_MODEL)]["folds"][3]
    record["periods"]["outer_validation"]["end"] = str(
        pd.Timestamp(FOLD_PERIODS[3]["outer_end"])
    )
    with pytest.raises(DecisionError, match="not inside the frozen period"):
        check_cells_agree(cells, P6)


def test_a_cell_reporting_the_wrong_number_of_folds_is_refused():
    cells = full_set()
    cells[("1m", "logistic_regression")]["folds"].pop()
    with pytest.raises(DecisionError, match="reports 3 folds"):
        check_cells_agree(cells, P6)


# --------------------------------------------------------------------------- #
# E. decide: the outcome, and who is allowed to produce it
# --------------------------------------------------------------------------- #


def test_no_viable_clock_makes_the_outcome_negative():
    decision = decide(full_set(), P6)
    assert decision["viable_clocks"] == []
    assert decision["outcome"] == "negative"
    assert [row["clock"] for row in decision["clocks"]] == list(P6.clocks)
    assert decision["decided_by"] == PRIMARY_MODEL
    assert "best_clock" not in decision


def test_one_viable_clock_makes_the_outcome_supportive():
    cells = full_set()
    for model in MODELS:
        cells[("5m", model)] = cell("5m", model, PASSING)
    decision = decide(cells, P6)
    assert decision["viable_clocks"] == ["5m"]
    assert decision["outcome"] == "supportive_adaptive"


def test_only_the_primary_family_can_move_the_outcome():
    """The exact shopping route the design forbids, attempted and ineffective."""
    cells = full_set()
    for model in set(MODELS) - {PRIMARY_MODEL}:
        cells[("1m", model)] = cell("1m", model, PASSING)
    decision = decide(cells, P6)

    assert decision["viable_clocks"] == []
    assert decision["outcome"] == "negative"
    context = {
        (row["clock"], row["model"]): row["would_have_passed"]
        for row in secondary_context(cells, P6)
    }
    assert all(context[("1m", model)] for model in set(MODELS) - {PRIMARY_MODEL})
    assert PRIMARY_MODEL not in {model for _, model in context}


def test_the_secondary_context_covers_every_clock_and_every_other_family():
    rows = secondary_context(full_set(), P6)
    assert len(rows) == len(P6.clocks) * (len(MODELS) - 1)
    assert {row["model"] for row in rows} == set(MODELS) - {PRIMARY_MODEL}
    assert {row["clock"] for row in rows} == set(P6.clocks)
    assert all(set(row) == {
        "clock",
        "model",
        "would_have_passed",
        "positive_folds",
        "mean_outer_net_return",
        "beats_momentum_folds",
    } for row in rows)
