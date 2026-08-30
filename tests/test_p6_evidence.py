"""P6's committed evidence, checked against the design it was produced under.

The load-bearing test here is
:func:`test_every_deciding_verdict_recomputes_from_the_predictions`. It does not
read the verdict, the summary, or any aggregate: it takes each XGBoost cell's
per-sample ``outer_predictions.parquet``, replays the trades through
:func:`nn.evaluate.realised_trades`, applies the three preregistered conditions
itself, and asserts the answer is the one the decision artifact published.

That is the difference between checking a number and checking a result. Every
other test in this file confirms that the cells are what they claim to be; this
one confirms that the *verdict* follows from the predictions, along a path that
shares no code with the one that produced it beyond the cost model both are
obliged to use.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chimera.contracts import TargetSpec
from nn.evaluate import realised_trades
from nn.multiclock import RESEARCH_VISIBLE_END, STYX_START
from nn.p6 import ARTIFACT_NAME, PREDICTIONS_NAME
from nn.p6_decision import DECISION_NAME, VERDICT_NOT_VIABLE, VERDICT_VIABLE
from nn.p6_preregistration import (
    CLOCKS,
    COSTS,
    FOLD_PERIODS,
    HORIZON_BARS,
    MODELS,
    PRIMARY_MODEL,
    SEED,
    SEQ_LEN,
    VIABILITY_GATE,
    preregistration_hash,
)

REPO = Path(__file__).resolve().parents[1]
BENCHMARK = REPO / "artifacts" / "benchmark"
DECISION_DIR = BENCHMARK / "btc_p6_decision"
MANIFEST = REPO / "artifacts" / "btc_p6_SHA256SUMS.txt"

CELLS = [(clock, model) for clock in CLOCKS for model in MODELS]


def cell_dir(clock: str, model: str) -> Path:
    return BENCHMARK / f"btc_p6_{clock}_{model}"


@pytest.fixture(scope="module")
def cells() -> dict[tuple[str, str], dict]:
    return {
        (clock, model): json.loads((cell_dir(clock, model) / ARTIFACT_NAME).read_text())
        for clock, model in CELLS
    }


@pytest.fixture(scope="module")
def decision() -> dict:
    return json.loads((DECISION_DIR / DECISION_NAME).read_text())


# --------------------------------------------------------------------------- #
# A. every planned cell exists, and none of them is something else
# --------------------------------------------------------------------------- #


def test_all_fifteen_cells_are_committed():
    assert len(CELLS) == 15
    missing = [
        f"{clock}x{model}" for clock, model in CELLS if not cell_dir(clock, model).is_dir()
    ]
    assert missing == []


@pytest.mark.parametrize("clock,model", CELLS, ids=[f"{c}-{m}" for c, m in CELLS])
def test_each_cell_declares_the_frozen_design(cells, clock, model):
    cell = cells[(clock, model)]
    assert cell["checkpoint"] == "P6"
    assert cell["preregistration_hash"] == preregistration_hash()
    assert cell["clock"] == clock
    assert cell["model"] == model
    assert cell["horizon_bars"] == HORIZON_BARS
    assert cell["config"] == {"seed": SEED, "seq_len": SEQ_LEN, "min_trades": 10}
    assert cell["target"]["horizon"] == HORIZON_BARS
    assert cell["target"]["fee_rate"] == COSTS["fee_rate"]
    assert cell["target"]["slippage_rate"] == COSTS["slippage_rate"]
    assert cell["tuning"].startswith("none")
    assert len(cell["folds"]) == 4


def test_every_cell_read_the_same_1m_source(cells):
    digests = {cell["source"]["minutes_digest"] for cell in cells.values()}
    assert len(digests) == 1
    per_clock = {}
    for (clock, _), cell in cells.items():
        per_clock.setdefault(clock, set()).add(cell["source"]["clock_digest"])
    assert all(len(values) == 1 for values in per_clock.values())
    # Five clocks, five distinct derived series.
    assert len({next(iter(v)) for v in per_clock.values()}) == len(CLOCKS)


@pytest.mark.parametrize("clock,model", CELLS, ids=[f"{c}-{m}" for c, m in CELLS])
def test_thresholds_were_chosen_on_the_inner_block(cells, clock, model):
    for record in cells[(clock, model)]["folds"]:
        selection = record["model"]["selection"]
        assert selection["block"] == "inner_validation"
        assert selection["min_trades"] == 10
        assert 0.34 <= selection["threshold"] <= 0.90


# --------------------------------------------------------------------------- #
# B. boundaries
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("clock,model", CELLS, ids=[f"{c}-{m}" for c, m in CELLS])
def test_no_scored_row_reaches_the_research_boundary(clock, model):
    predictions = pd.read_parquet(cell_dir(clock, model) / PREDICTIONS_NAME)
    stamps = pd.to_datetime(predictions["timestamp"], utc=True)
    assert int((stamps >= RESEARCH_VISIBLE_END).sum()) == 0
    assert int((stamps >= STYX_START).sum()) == 0


@pytest.mark.parametrize("clock,model", CELLS, ids=[f"{c}-{m}" for c, m in CELLS])
def test_outer_periods_are_the_four_frozen_windows(cells, clock, model):
    for record, frozen in zip(cells[(clock, model)]["folds"], FOLD_PERIODS):
        period = record["periods"]["outer_validation"]
        start = pd.Timestamp(period["start"])
        end = pd.Timestamp(period["end"])
        assert pd.Timestamp(frozen["outer_start"]) <= start
        assert end < pd.Timestamp(frozen["outer_end"])


def test_every_clock_scores_the_same_four_real_world_windows(cells):
    """Five clocks, one experiment: the windows coincide to within one bar."""
    for position in range(4):
        starts = {
            pd.Timestamp(
                cells[(clock, PRIMARY_MODEL)]["folds"][position]["periods"][
                    "outer_validation"
                ]["start"]
            )
            for clock in CLOCKS
        }
        # Every clock's outer block opens at the frozen instant; a clock whose
        # bar does not land on it opens at the first bar after, never before.
        frozen = pd.Timestamp(FOLD_PERIODS[position]["outer_start"])
        assert min(starts) >= frozen
        assert max(starts) - min(starts) <= pd.Timedelta(hours=1)


# --------------------------------------------------------------------------- #
# C. the verdicts, recomputed from the predictions
# --------------------------------------------------------------------------- #


def _recompute_net_return(frame: pd.DataFrame, spec: TargetSpec) -> float:
    """Replay one fold's signals through the shared cost model."""
    ordered = frame.sort_values("row_index")
    _, _, returns = realised_trades(
        ordered["selected_action"].to_numpy(dtype=np.int64),
        ordered["future_return"].to_numpy(dtype=np.float64),
        spec,
        row_index=ordered["row_index"].to_numpy(dtype=np.int64),
    )
    if len(returns) == 0:
        return 0.0
    return round(float(np.cumprod(1.0 + returns)[-1] - 1.0), 6)


@pytest.mark.parametrize("clock", CLOCKS)
def test_every_deciding_verdict_recomputes_from_the_predictions(cells, decision, clock):
    """The five XGBoost verdicts, derived again from per-sample predictions.

    Reads no summary and no verdict: the fold returns come from replaying the
    stored actions, the momentum floor comes from the cell's own outer block,
    and the three conditions are applied here rather than looked up.
    """
    cell = cells[(clock, PRIMARY_MODEL)]
    spec = TargetSpec(
        horizon=HORIZON_BARS,
        fee_rate=COSTS["fee_rate"],
        slippage_rate=COSTS["slippage_rate"],
    )
    predictions = pd.read_parquet(cell_dir(clock, PRIMARY_MODEL) / PREDICTIONS_NAME)

    returns: list[float] = []
    beats = 0
    for record in cell["folds"]:
        fold = int(record["fold"])
        recomputed = _recompute_net_return(predictions.loc[predictions["fold"] == fold], spec)
        published = float(record["outer_validation"][PRIMARY_MODEL]["trading"]["net_return"])
        assert recomputed == pytest.approx(
            published, abs=1e-6
        ), f"{clock} fold {fold}: predictions replay to {recomputed}, cell says {published}"
        returns.append(published)
        momentum = float(
            record["outer_validation"]["momentum_baseline"]["trading"]["net_return"]
        )
        beats += published > momentum

    positive = sum(1 for value in returns if value > 0.0)
    mean = float(np.mean(returns))
    viable = (
        positive >= VIABILITY_GATE["positive_folds_required"]
        and mean > 0.0
        and beats >= VIABILITY_GATE["beats_momentum_folds_required"]
    )

    published_verdict = next(row for row in decision["clocks"] if row["clock"] == clock)
    assert published_verdict["verdict"] == (VERDICT_VIABLE if viable else VERDICT_NOT_VIABLE)
    assert published_verdict["conditions"]["positive_folds"]["observed"] == positive
    assert published_verdict["conditions"]["beats_native_momentum_folds"]["observed"] == beats
    assert published_verdict["conditions"]["mean_outer_net_return"][
        "observed"
    ] == pytest.approx(mean, abs=1e-9)


def test_the_decision_reports_all_five_clocks_and_no_summary_row(decision):
    assert [row["clock"] for row in decision["clocks"]] == list(CLOCKS)
    assert decision["decided_by"] == PRIMARY_MODEL
    assert decision["preregistration_hash"] == preregistration_hash()
    assert "best_clock" not in decision
    assert set(decision["viable_clocks"]) <= set(CLOCKS)
    expected = OUTCOMES[bool(decision["viable_clocks"])]
    assert decision["outcome"] == expected


OUTCOMES = {True: "supportive_adaptive", False: "negative"}


def test_secondary_families_decide_nothing(decision):
    """A clock's verdict is XGBoost's even where another family would pass.

    The second assertion is what makes the rest worth making: there has to be at
    least one clock where a secondary family cleared the gate and XGBoost did
    not. Without it a ``decide()`` that shopped families could satisfy this test
    by never having been offered the opportunity.

    Where the opportunity exists, the published row must be XGBoost's arithmetic
    and not the passing family's — the observed condition values are checked
    against the published fold list, which
    :func:`test_every_deciding_verdict_recomputes_from_the_predictions` ties back
    to the XGBoost cell's own per-sample predictions.
    """
    rows = decision["secondary_context"]
    assert {row["model"] for row in rows} == set(MODELS) - {PRIMARY_MODEL}
    assert {row["clock"] for row in rows} == set(CLOCKS)
    assert decision["decided_by"] == PRIMARY_MODEL
    assert {row["model"] for row in decision["clocks"]} == {PRIMARY_MODEL}

    published = {row["clock"]: row for row in decision["clocks"]}
    shopped = [
        row
        for row in rows
        if row["would_have_passed"] and not published[row["clock"]]["viable"]
    ]
    assert shopped, "no secondary family passed where XGBoost failed; this test proves nothing"

    for row in shopped:
        verdict = published[row["clock"]]
        # Exactly the case the design forbids acting on. Recorded, not acted on.
        assert verdict["verdict"] == VERDICT_NOT_VIABLE
        folds = [float(fold["net_return"]) for fold in verdict["folds"]]
        conditions = verdict["conditions"]
        assert conditions["positive_folds"]["observed"] == sum(
            1 for value in folds if value > 0.0
        )
        assert conditions["mean_outer_net_return"]["observed"] == pytest.approx(
            float(np.mean(folds)), abs=1e-9
        )
        # ... and that arithmetic is not the passing family's, so a substitution
        # would have been visible here rather than silent.
        assert (
            conditions["positive_folds"]["observed"],
            conditions["mean_outer_net_return"]["observed"],
        ) != (row["positive_folds"], row["mean_outer_net_return"])


# --------------------------------------------------------------------------- #
# D. the manifest
# --------------------------------------------------------------------------- #


def test_the_primary_evidence_is_frozen_and_still_hashes():
    from tools.freeze_evidence import check

    assert MANIFEST.is_file()
    problems = check(MANIFEST)
    assert problems == []


def test_the_manifest_covers_every_primary_file():
    covered = {
        line.split(maxsplit=1)[1].strip()
        for line in MANIFEST.read_text().splitlines()
        if line.strip()
    }
    for clock, model in CELLS:
        directory = cell_dir(clock, model).relative_to(REPO)
        for name in (ARTIFACT_NAME, PREDICTIONS_NAME):
            assert f"{directory}/{name}" in covered
