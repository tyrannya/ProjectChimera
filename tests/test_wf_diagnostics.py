"""Safeguards for the walk-forward diagnostics CLI.

The tool reads ``walkforward.json`` artifacts from paths given at runtime, so
these tests build artifacts on disk rather than reaching for any real run. Two
kinds of fixture, deliberately:

* **synthetic artifacts** built by :func:`write_run`, whose row geometry comes
  from the production planner and whose summary comes from the production
  aggregator. That keeps the fixture honest while letting a test break exactly
  one thing and check the diagnostics notices;
* **a real artifact**, produced by running ``nn.walkforward`` end to end on
  generated candles. That is the compatibility check: it is the only way to know
  the reader still matches what the writer emits, and no hand-written fixture
  can stand in for it.

No test here needs a committed dataset, a committed artifact, or a checkpoint.
"""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from chimera.contracts import CLASS_ORDER, TargetSpec
from chimera.features import FeatureSpec
from nn import evaluate as ev
from nn import regime, walkforward, wf_diagnostics
from nn.data_pipeline import build_dataset, save_dataset
from nn.dataset import Split, build_windows, resolve_sealed_boundary, sample_indices
from nn.research_contract import SYNTHETIC_CONTRACT_ID, load_contract
from tools.make_sample_data import generate_candles

#: The committed contract the synthetic fixtures below belong to, and the
#: instant it seals at. Read from the registry so these tests exercise the
#: contract the research entrypoints actually resolve.
CONTRACT = load_contract(SYNTHETIC_CONTRACT_ID)
SEALED_TEST_START_UTC = CONTRACT.sealed_test_start

TINY = [
    "--research-contract",
    SYNTHETIC_CONTRACT_ID,
    "--epochs",
    "1",
    "--seq-len",
    "16",
    "--d-model",
    "16",
    "--n-heads",
    "2",
    "--num-layers",
    "1",
    "--device",
    "cpu",
]

N_ROWS = 1200
FOLDS = 3

#: Sequence length the synthetic artifacts are treated as having been run with.
SEQ_LEN = 16


# --- synthetic artifacts ------------------------------------------------------
def synthetic_boundary(n_rows: int) -> int:
    """An arbitrary sealed start row for a hand-written artifact.

    Diagnostics *reads* a boundary out of an artifact; it never derives one, so
    for these fixtures the number only has to be a plausible row that the
    artifact then records. It is deliberately not resolved from the real anchor:
    these payloads have no dataset behind them, and pretending otherwise would
    tie a fixture to a contract it is not testing.
    """
    return int(n_rows * 0.85)


def plan_for(n_rows: int = N_ROWS, folds: int = FOLDS):
    """Fold geometry straight from the production planner."""
    boundary = synthetic_boundary(n_rows)
    args = walkforward.build_argparser().parse_args(["--dataset", "x", "--folds", str(folds)])
    sizes = walkforward.resolve_sizes(args, boundary)
    return boundary, walkforward.plan_nested_folds(boundary, folds, *sizes)


def period(split) -> dict[str, Any]:
    """A period block in the shape ``ResearchData.period`` produces."""
    return {
        "start": f"2023-01-01T{split.start % 24:02d}:00:00+00:00",
        "end": f"2023-01-02T{(split.end - 1) % 24:02d}:00:00+00:00",
        "rows": len(split),
        "row_range": [split.start, split.end],
    }


def report(
    net_return: float, macro_f1: float = 0.4, *, schema: str = "current"
) -> dict[str, Any]:
    """One model's evaluation report, carrying every aggregated metric.

    The shared base below is the eleven trading fields both generations record.
    ``schema="legacy"`` then adds the removed ``sharpe`` and neither of the two
    that replaced it, which makes the block exactly
    :data:`nn.regime.LEGACY_TRADING_KEYS` — the shape every committed
    pre-correction artifact carries, and the only one
    :func:`nn.wf_diagnostics.classify_schema` is allowed to read with fields
    skipped. Anything else adds the current risk-report fields instead, so the
    fixture is the complete current evaluator shape rather than a sketch of it.
    """
    trading: dict[str, Any] = {
        "n_trades": 20,
        "net_return": net_return,
        "gross_return": net_return + 0.012,
        "total_costs": 0.012,
        "avg_trade": net_return / 20,
        "win_rate": 0.45,
        "profit_factor": 1.1,
        "max_drawdown": 0.05,
        "exposure": 0.25,
        "turnover": 40.0,
        "cost_per_trade": 0.0006,
    }
    if schema == "legacy":
        trading["sharpe"] = net_return * 10
    else:
        trading.update(
            annualised_sharpe=net_return * 2,
            annualised_sharpe_reason="",
            sharpe_basis=ev.SHARPE_BASIS,
            candle_max_drawdown=0.05,
            elapsed_intervals=83,
            per_trade_sharpe=net_return,
            per_trade_sharpe_reason="",
        )
    return {
        "classification": {
            "n_samples": 83,
            "macro_f1": macro_f1,
            "accuracy": 0.5,
            "directional_accuracy": 0.55,
            "coverage": 0.75,
            "calibration_error": 0.1,
        },
        "trading": trading,
    }


def _summarise(fold_payloads: list[dict[str, Any]], schema: str) -> dict[str, Any]:
    """The summary a run of this generation would actually have recorded.

    A pre-correction run's aggregator did not know the two candle-level metrics,
    so its summary does not carry them. Producing one that does would make the
    fixture a shape no real artifact has ever had, and the legacy path would
    then be tested against fiction.
    """
    if schema != "legacy":
        return walkforward.summarise(fold_payloads)

    # `summarise` reads the current metric set, so aggregate a current-shaped
    # copy and then strip what a pre-correction run would not have recorded.
    shaped = copy.deepcopy(fold_payloads)
    for fold in shaped:
        for model in walkforward.MODELS:
            trading = fold["outer_validation"][model]["trading"]
            for metric in wf_diagnostics.CURRENT_ONLY_METRICS:
                trading.setdefault(metric, 0.0)
    summary = walkforward.summarise(shaped)
    for stats in summary["per_model"].values():
        for metric in wf_diagnostics.CURRENT_ONLY_METRICS:
            stats.pop(metric, None)
        stats["sharpe"] = {"mean": 0.0, "std": 0.0, "values": [], "defined_folds": 0}
    return summary


def write_run(
    directory: Path,
    *,
    seed: int = 42,
    n_rows: int = N_ROWS,
    folds: int = FOLDS,
    returns: dict[str, list[float]] | None = None,
    thresholds: list[float] | None = None,
    epochs: list[int] | None = None,
    dataset: dict[str, Any] | None = None,
    outer_samples: list[int] | None = None,
    schema: str = "current",
) -> Path:
    """Write a well-formed artifact, then let a caller break one thing.

    ``returns`` maps model name to one net return per fold; the defaults make
    MTST win every fold, so a test about the baseline comparison has to say so
    explicitly rather than inherit it.
    """
    boundary, plans = plan_for(n_rows, folds)
    returns = returns or {
        "majority_baseline": [-0.01] * folds,
        "momentum_baseline": [0.0] * folds,
        "mtst": [0.1 * (i + 1) for i in range(folds)],
    }
    thresholds = thresholds or [0.4] * folds
    epochs = epochs or [2] * folds

    fold_payloads = []
    for index, plan in enumerate(plans):
        fold_payloads.append(
            {
                "fold": index,
                "seed": seed + index,
                "samples": {
                    "train": 400,
                    "inner_validation": 83,
                    "outer_validation": (outer_samples[index] if outer_samples else 83),
                },
                "periods": {
                    "train": period(plan.train),
                    "inner_validation": period(plan.inner),
                    "outer_validation": period(plan.outer),
                },
                "selection": {
                    "best_epoch": epochs[index],
                    "threshold": thresholds[index],
                    "inner_validation_loss": 1.0,
                },
                "outer_validation": {
                    model: report(values[index], schema=schema)
                    for model, values in returns.items()
                },
            }
        )

    directory.mkdir(parents=True, exist_ok=True)
    (directory / wf_diagnostics.ARTIFACT_NAME).write_text(
        json.dumps(
            {
                "dataset": dataset or {"rows": n_rows, "pair": "BTC/USDT", "timeframe": "1h"},
                "config": {"seed": seed, "folds": folds, "seq_len": SEQ_LEN},
                "sealed_test": {
                    "start_row": boundary,
                    "period": {"row_range": [boundary, n_rows], "rows": n_rows - boundary},
                    "evaluated": False,
                },
                "research_rows": boundary,
                "test_evaluated": False,
                "reported_block": "outer_validation",
                "folds": fold_payloads,
                # The production aggregator, so the fixture's summary is exactly
                # what a real run would have recorded for these folds.
                "summary": _summarise(fold_payloads, schema),
            },
            indent=2,
        )
    )
    return directory


def edit(directory: Path, mutate) -> Path:
    """Apply ``mutate`` to an artifact already on disk."""
    artifact = directory / wf_diagnostics.ARTIFACT_NAME
    payload = json.loads(artifact.read_text())
    mutate(payload)
    artifact.write_text(json.dumps(payload, indent=2))
    return directory


def problems_for(directory: Path) -> list[str]:
    return wf_diagnostics.audit_run(wf_diagnostics.load_run(directory))


# --- 1. loading ---------------------------------------------------------------
def test_a_directory_and_the_json_file_both_load(tmp_path):
    directory = write_run(tmp_path / "btc_nested_v1")
    from_dir = wf_diagnostics.load_run(directory)
    from_file = wf_diagnostics.load_run(directory / wf_diagnostics.ARTIFACT_NAME)

    assert from_dir.folds == from_file.folds
    # Either way the run is named after the directory it came from.
    assert from_dir.name == from_file.name == "btc_nested_v1"
    assert from_dir.seed == 42
    assert from_dir.n_folds == FOLDS


def test_a_missing_artifact_names_the_path_it_looked_for(tmp_path):
    with pytest.raises(SystemExit, match=r"no walk-forward artifact at .*walkforward\.json"):
        wf_diagnostics.load_run(tmp_path / "not_a_run")


def test_unreadable_json_is_refused(tmp_path):
    directory = write_run(tmp_path / "run")
    (directory / wf_diagnostics.ARTIFACT_NAME).write_text('{"folds": [')
    with pytest.raises(SystemExit, match="not readable JSON"):
        wf_diagnostics.load_run(directory)


def test_an_artifact_with_no_folds_is_refused(tmp_path):
    directory = edit(write_run(tmp_path / "run"), lambda p: p.update(folds=[]))
    with pytest.raises(SystemExit, match="has no folds"):
        wf_diagnostics.load_run(directory)


def test_a_pre_nested_artifact_is_refused_and_says_why(tmp_path):
    """The two-region schema must not be read as if it were an evaluation.

    Its single ``validation`` block was both selected on and reported, so
    averaging it beside nested runs would quietly reintroduce the optimism the
    nested design removed.
    """

    def to_legacy(payload):
        for fold in payload["folds"]:
            fold["validation"] = fold.pop("outer_validation")
            fold["periods"]["validation"] = fold["periods"].pop("outer_validation")

    directory = edit(write_run(tmp_path / "old_run"), to_legacy)
    with pytest.raises(SystemExit, match="pre-nested artifact"):
        wf_diagnostics.load_run(directory)


def test_a_fold_missing_a_region_is_refused(tmp_path):
    directory = edit(
        write_run(tmp_path / "run"),
        lambda p: p["folds"][1]["periods"].pop("inner_validation"),
    )
    with pytest.raises(SystemExit, match="missing the inner_validation period"):
        wf_diagnostics.load_run(directory)


def test_an_artifact_without_a_sealed_boundary_is_refused(tmp_path):
    directory = edit(write_run(tmp_path / "run"), lambda p: p.pop("sealed_test"))
    with pytest.raises(SystemExit, match="no sealed_test.start_row"):
        wf_diagnostics.load_run(directory)


# --- 2. the integrity audit ---------------------------------------------------
def test_a_sound_artifact_audits_clean(tmp_path):
    assert problems_for(write_run(tmp_path / "run")) == []


def test_overlapping_outer_blocks_are_caught(tmp_path):
    """Two folds reporting the same row would count it twice in the mean."""

    def overlap(payload):
        first = payload["folds"][0]["periods"]["outer_validation"]["row_range"]
        second = payload["folds"][1]["periods"]["outer_validation"]
        second["row_range"] = [first[0], first[1]]

    problems = problems_for(edit(write_run(tmp_path / "run"), overlap))
    assert any("overlaps fold 0" in problem for problem in problems)


def test_a_reported_block_that_was_selected_on_is_caught(tmp_path):
    """Inner running past the start of outer is the leak this design removed."""

    def bleed(payload):
        periods = payload["folds"][0]["periods"]
        outer_start = periods["outer_validation"]["row_range"][0]
        periods["inner_validation"]["row_range"][1] = outer_start + 10

    problems = problems_for(edit(write_run(tmp_path / "run"), bleed))
    assert any("the reported block was selected on" in problem for problem in problems)


def test_training_overlapping_the_inner_block_is_caught(tmp_path):
    def bleed(payload):
        periods = payload["folds"][0]["periods"]
        periods["train"]["row_range"][1] = periods["inner_validation"]["row_range"][0] + 5

    problems = problems_for(edit(write_run(tmp_path / "run"), bleed))
    assert any("training overlaps the block it is selected on" in p for p in problems)


@pytest.mark.parametrize("region", wf_diagnostics.REGIONS)
def test_any_region_reaching_the_sealed_block_is_caught(tmp_path, region):
    """Asserted on rows for all three regions, not just the reported one."""
    boundary = synthetic_boundary(N_ROWS)

    def cross(payload):
        payload["folds"][-1]["periods"][region]["row_range"][1] = boundary + 1

    problems = problems_for(edit(write_run(tmp_path / "run"), cross))
    assert any(
        f"{region} ends at row {boundary + 1}" in problem and "sealed" in problem
        for problem in problems
    )


@pytest.mark.parametrize(
    "mutate, expected",
    [
        pytest.param(
            lambda p: p.update(test_evaluated=True), "test_evaluated", id="test-evaluated"
        ),
        pytest.param(
            lambda p: p.pop("test_evaluated"), "test_evaluated", id="test-flag-missing"
        ),
        pytest.param(
            lambda p: p["sealed_test"].update(evaluated=True),
            "sealed_test.evaluated",
            id="sealed-evaluated",
        ),
        pytest.param(
            lambda p: p.update(reported_block="inner_validation"),
            "reported_block",
            id="wrong-reported-block",
        ),
        pytest.param(
            lambda p: p["summary"].update(aggregated_from="inner_validation"),
            "summary.aggregated_from",
            id="wrong-aggregation-source",
        ),
        pytest.param(
            lambda p: p["summary"].update(folds=99), "summary.folds", id="fold-count"
        ),
    ],
)
def test_the_sealing_and_provenance_flags_fail_closed(tmp_path, mutate, expected):
    problems = problems_for(edit(write_run(tmp_path / "run"), mutate))
    assert any(expected in problem for problem in problems)


def test_a_summary_that_does_not_follow_from_the_folds_is_caught(tmp_path):
    """The headline has to be reproducible from the per-fold reports beside it."""

    def rewrite_headline(payload):
        payload["summary"]["per_model"]["mtst"]["net_return"]["mean"] = 0.9

    problems = problems_for(edit(write_run(tmp_path / "run"), rewrite_headline))
    assert any(
        "summary mtst.net_return mean is 0.9" in problem and "folds average" in problem
        for problem in problems
    )


def test_editing_a_fold_out_from_under_the_summary_is_caught(tmp_path):
    """The same check from the other side: the folds changed, the headline did not."""

    def rewrite_fold(payload):
        payload["folds"][0]["outer_validation"]["mtst"]["trading"]["net_return"] = 5.0

    problems = problems_for(edit(write_run(tmp_path / "run"), rewrite_fold))
    assert any("summary mtst.net_return" in problem for problem in problems)


# --- 3. comparing runs --------------------------------------------------------
def test_runs_with_the_same_geometry_compare_clean(tmp_path):
    runs = [
        wf_diagnostics.load_run(write_run(tmp_path / "a", seed=42)),
        wf_diagnostics.load_run(write_run(tmp_path / "b", seed=142)),
    ]
    assert wf_diagnostics.compare_runs(runs) == []


def test_runs_whose_outer_blocks_are_different_rows_are_not_comparable(tmp_path):
    def shorten_fold_1(payload):
        payload["folds"][1]["periods"]["outer_validation"]["row_range"][1] = 700

    runs = [
        wf_diagnostics.load_run(write_run(tmp_path / "a")),
        wf_diagnostics.load_run(edit(write_run(tmp_path / "b"), shorten_fold_1)),
    ]
    problems = wf_diagnostics.compare_runs(runs)
    assert any(
        "does not share" in problem and "fold 1 outer_validation" in problem
        for problem in problems
    )


def test_runs_with_different_fold_counts_are_not_comparable(tmp_path):
    runs = [
        wf_diagnostics.load_run(write_run(tmp_path / "a", folds=3)),
        wf_diagnostics.load_run(write_run(tmp_path / "b", folds=2)),
    ]
    assert any("folds" in problem for problem in wf_diagnostics.compare_runs(runs))


def test_runs_on_a_different_dataset_are_not_comparable(tmp_path):
    runs = [
        wf_diagnostics.load_run(write_run(tmp_path / "a")),
        wf_diagnostics.load_run(
            edit(write_run(tmp_path / "b"), lambda p: p["dataset"].update(pair="ETH/USDT"))
        ),
    ]
    assert any("dataset pair" in problem for problem in wf_diagnostics.compare_runs(runs))


def test_a_different_sealed_boundary_is_not_comparable(tmp_path):
    runs = [
        wf_diagnostics.load_run(write_run(tmp_path / "a")),
        wf_diagnostics.load_run(
            edit(write_run(tmp_path / "b"), lambda p: p["sealed_test"].update(start_row=999))
        ),
    ]
    assert any("sealed_test_start" in p for p in wf_diagnostics.compare_runs(runs))


def test_a_row_only_boundary_is_not_comparable_with_a_timestamp_anchored_one(tmp_path):
    """Two runs landing on the same row is not evidence they sealed the same data.

    A run that records only ``start_row`` cannot say which instant that row meant,
    so pairing it with one that *was* sealed at a stated anchor would put a single
    sealed-test claim on runs that do not share one. The five committed artifacts
    are all row-only, which is why this fails closed on the mismatch rather than
    on the absence.
    """
    anchored = edit(
        write_run(tmp_path / "b"),
        lambda p: p["sealed_test"].update(anchor_timestamp=SEALED_TEST_START_UTC.isoformat()),
    )
    runs = [
        wf_diagnostics.load_run(write_run(tmp_path / "a")),
        wf_diagnostics.load_run(anchored),
    ]

    assert runs[0].sealed_anchor is None
    assert runs[1].sealed_anchor == SEALED_TEST_START_UTC.isoformat()
    # Same row, so the row check passes and only the provenance check fires.
    assert runs[0].sealed_test_start == runs[1].sealed_test_start
    problems = wf_diagnostics.compare_runs(runs)
    assert any("row index only" in problem for problem in problems)


def test_runs_sharing_row_only_provenance_stay_comparable(tmp_path):
    """The committed generation must not become unreadable for predating the anchor."""
    runs = [
        wf_diagnostics.load_run(write_run(tmp_path / "a")),
        wf_diagnostics.load_run(write_run(tmp_path / "b", seed=142)),
    ]
    assert all(run.sealed_anchor is None for run in runs)
    assert wf_diagnostics.compare_runs(runs) == []


def test_a_historical_artifact_is_never_relabelled_as_timestamp_anchored(tmp_path):
    """Absent provenance is reported as absent, never filled in from the anchor."""
    run = wf_diagnostics.load_run(write_run(tmp_path / "a"))
    assert run.sealed_anchor is None
    assert wf_diagnostics.audit_run(run) == [], "row-only provenance is not an integrity fault"


# --- 4. the seed-stability aggregate ------------------------------------------
@pytest.fixture
def two_runs(tmp_path):
    """Two runs differing only in seed, with controlled per-fold net returns."""
    a = write_run(
        tmp_path / "btc_nested_v1",
        seed=42,
        returns={
            "majority_baseline": [-0.01, -0.01, -0.01],
            # Momentum wins fold 2 against the first run's MTST but not the second's.
            "momentum_baseline": [0.0, 0.0, 0.35],
            "mtst": [0.1, 0.2, 0.3],
        },
        thresholds=[0.40, 0.40, 0.50],
        epochs=[2, 3, 4],
    )
    b = write_run(
        tmp_path / "btc_nested_seed_142",
        seed=142,
        returns={
            "majority_baseline": [-0.01, -0.01, -0.01],
            "momentum_baseline": [0.0, 0.0, 0.35],
            "mtst": [0.2, 0.3, 0.4],
        },
        thresholds=[0.40, 0.42, 0.80],
        epochs=[2, 3, 6],
    )
    return [wf_diagnostics.load_run(a), wf_diagnostics.load_run(b)]


def test_the_aggregate_reports_each_run_mean_and_the_spread_of_those_means(two_runs):
    summary = wf_diagnostics.aggregate(two_runs)
    net_return = summary["per_model"]["mtst"]["net_return"]

    assert summary["runs"] == ["btc_nested_v1", "btc_nested_seed_142"]
    assert summary["folds"] == 3
    assert net_return["per_run_mean"] == {
        "btc_nested_v1": pytest.approx(0.2),
        "btc_nested_seed_142": pytest.approx(0.3),
    }
    across = net_return["across_runs"]
    assert across["mean"] == pytest.approx(0.25)
    assert across["std"] == pytest.approx(0.0707107, abs=1e-6)
    assert (across["min"], across["max"]) == (pytest.approx(0.2), pytest.approx(0.3))


def test_the_aggregate_reports_each_fold_across_runs(two_runs):
    """Which fold is stable and which is a seed lottery."""
    per_fold = wf_diagnostics.aggregate(two_runs)["per_model"]["mtst"]["net_return"][
        "per_fold"
    ]

    assert [entry["fold"] for entry in per_fold] == [0, 1, 2]
    assert per_fold[0]["values"] == {
        "btc_nested_v1": pytest.approx(0.1),
        "btc_nested_seed_142": pytest.approx(0.2),
    }
    assert per_fold[0]["mean"] == pytest.approx(0.15)
    assert per_fold[2]["mean"] == pytest.approx(0.35)
    for entry in per_fold:
        assert entry["std"] == pytest.approx(0.0707107, abs=1e-6)


def test_every_aggregated_metric_is_one_walkforward_already_summarises(two_runs):
    """One list of metrics, so the two modules cannot drift apart."""
    summary = wf_diagnostics.aggregate(two_runs)
    for model in walkforward.MODELS:
        assert set(summary["per_model"][model]) == set(walkforward.SUMMARY_METRICS)


def test_selection_stability_reports_threshold_and_epoch_per_fold(two_runs):
    selection = wf_diagnostics.aggregate(two_runs)["selection"]

    assert [entry["fold"] for entry in selection] == [0, 1, 2]
    # Fold 0 picked the same threshold in both runs; fold 2 did not.
    assert selection[0]["threshold"]["std"] == pytest.approx(0.0)
    assert selection[2]["threshold"]["std"] > 0.2
    assert selection[2]["threshold"]["values"] == {
        "btc_nested_v1": pytest.approx(0.5),
        "btc_nested_seed_142": pytest.approx(0.8),
    }
    assert selection[2]["best_epoch"]["mean"] == pytest.approx(5.0)


def test_the_baseline_comparison_counts_every_run_fold(two_runs):
    """MTST loses fold 2 in the first run and wins it in the second."""
    comparison = wf_diagnostics.aggregate(two_runs)["baseline_comparison"]

    assert comparison["run_folds"] == 6
    assert comparison["per_run"] == {"btc_nested_v1": 2, "btc_nested_seed_142": 3}
    assert comparison["mtst_beat_both_baselines"] == 5
    assert comparison["runs_with_majority"] == 2


def test_a_single_run_still_aggregates_with_zero_spread(tmp_path):
    runs = [wf_diagnostics.load_run(write_run(tmp_path / "only"))]
    across = wf_diagnostics.aggregate(runs)["per_model"]["mtst"]["net_return"]["across_runs"]
    assert across["std"] == 0.0
    assert across["mean"] == across["min"] == across["max"]


# --- 5. the CLI ---------------------------------------------------------------
def test_the_cli_audits_and_aggregates_and_writes_both_files(tmp_path, capsys):
    a = write_run(tmp_path / "btc_nested_v1", seed=42)
    b = write_run(tmp_path / "btc_nested_seed_142", seed=142)
    out = tmp_path / "diag"

    assert wf_diagnostics.main([str(a), str(b), "--out", str(out)]) == 0

    printed = capsys.readouterr().out
    assert "# Walk-forward regime diagnostics" in printed
    assert "btc_nested_seed_142" in printed

    payload = json.loads((out / wf_diagnostics.REPORT_JSON).read_text())
    assert payload["comparability_problems"] == []
    assert payload["sealed_test_evaluated"] is False
    assert payload["aggregated_from"] == "outer_validation"
    assert [run["integrity_problems"] for run in payload["runs"]] == [[], []]
    assert payload["summary"]["runs"] == ["btc_nested_v1", "btc_nested_seed_142"]
    # The file on disk is the report that was printed, not a second rendering.
    assert (out / wf_diagnostics.REPORT_MD).read_text().strip() == printed.strip()


def test_the_cli_exits_nonzero_and_withholds_the_aggregate_on_a_broken_run(tmp_path, capsys):
    """A headline over an artifact that failed its audit is the thing to distrust."""
    good = write_run(tmp_path / "good", seed=42)
    bad = edit(write_run(tmp_path / "bad", seed=142), lambda p: p.update(test_evaluated=True))
    out = tmp_path / "diag"

    assert wf_diagnostics.main([str(good), str(bad), "--out", str(out)]) == 1

    printed = capsys.readouterr().out
    assert "## Integrity problems" in printed
    assert "Outer validation across runs" not in printed
    assert json.loads((out / wf_diagnostics.REPORT_JSON).read_text())["summary"] is None


def test_the_cli_withholds_the_aggregate_when_runs_are_not_comparable(tmp_path, capsys):
    a = write_run(tmp_path / "a", folds=3)
    b = write_run(tmp_path / "b", folds=2)

    assert wf_diagnostics.main([str(a), str(b)]) == 1
    printed = capsys.readouterr().out
    assert "## Runs are not comparable" in printed
    assert "averaging them would compare different blocks" in printed


def test_two_runs_with_the_same_directory_name_are_refused(tmp_path):
    a = write_run(tmp_path / "one" / "btc_nested_v1")
    b = write_run(tmp_path / "two" / "btc_nested_v1")
    with pytest.raises(SystemExit, match="same name"):
        wf_diagnostics.main([str(a), str(b)])


def test_the_report_says_what_the_numbers_are_not(tmp_path, capsys):
    """The tool must not let a seed study read as an out-of-sample result."""
    wf_diagnostics.main([str(write_run(tmp_path / "run"))])
    printed = capsys.readouterr().out

    assert "## Limitations" in printed
    assert "not out-of-sample performance" in printed
    assert "not a claim of profitability" in printed
    assert "sealed test block remains unopened" in printed
    assert "nothing was fitted on them" in printed
    assert "coincidence in the data" in printed


def test_the_report_shows_the_geometry_the_runs_share(tmp_path, capsys):
    _, plans = plan_for()
    wf_diagnostics.main([str(write_run(tmp_path / "run"))])
    printed = capsys.readouterr().out

    for plan in plans:
        assert f"[{plan.outer.start}, {plan.outer.end})" in printed
        assert f"[{plan.inner.start}, {plan.inner.end})" in printed


# --- 6. compatibility with what nn.walkforward actually writes -----------------
@pytest.fixture(scope="module")
def real_runs(tmp_path_factory):
    """Two genuine walk-forward runs, differing only in --seed.

    Slow next to the synthetic fixtures and worth it: it is the only check that
    the reader still matches the writer. If the artifact schema changes, this
    fails and the hand-written fixtures above do not.
    """
    workspace = tmp_path_factory.mktemp("real_wf")
    dataset = workspace / "ds.parquet"
    frame, metadata = build_dataset(
        generate_candles(rows=1200, seed=17),
        FeatureSpec(),
        TargetSpec(horizon=4),
        exchange="synthetic",
        pair="SYNTH/USDT",
        timeframe="1h",
    )
    save_dataset(dataset, frame, metadata)

    directories = []
    for seed in (42, 142):
        out = workspace / f"synthetic_nested_seed_{seed}"
        assert (
            walkforward.main(
                [
                    "--dataset",
                    str(dataset),
                    "--out",
                    str(out),
                    "--folds",
                    "2",
                    "--seed",
                    str(seed),
                    *TINY,
                ]
            )
            == 0
        )
        directories.append(out)
    return directories


def test_a_real_walkforward_artifact_loads_and_audits_clean(real_runs):
    for directory in real_runs:
        run = wf_diagnostics.load_run(directory)
        assert run.n_folds == 2
        assert wf_diagnostics.audit_run(run) == [], f"{run.name} failed its audit"


def test_real_runs_that_differ_only_in_seed_are_comparable(real_runs):
    runs = [wf_diagnostics.load_run(directory) for directory in real_runs]
    assert [run.seed for run in runs] == [42, 142]
    assert wf_diagnostics.compare_runs(runs) == []


def test_the_cli_reports_on_real_artifacts(real_runs, tmp_path, capsys):
    out = tmp_path / "diag"
    assert wf_diagnostics.main([str(d) for d in real_runs] + ["--out", str(out)]) == 0

    printed = capsys.readouterr().out
    assert "Outer validation across runs" in printed
    assert "Selection stability" in printed

    payload = json.loads((out / wf_diagnostics.REPORT_JSON).read_text())
    assert payload["summary"]["folds"] == 2
    assert set(payload["summary"]["per_model"]) == set(walkforward.MODELS)
    for run in payload["runs"]:
        assert run["integrity_problems"] == []


def test_the_aggregate_matches_the_summaries_the_runs_recorded(real_runs):
    """Cross-check: each run's own across-fold mean is what the tool reports.

    ``nn.walkforward`` already computes a per-run mean; the diagnostics recompute
    it from the folds. They must agree, or one of the two is reading the wrong
    block.
    """
    runs = [wf_diagnostics.load_run(directory) for directory in real_runs]
    summary = wf_diagnostics.aggregate(runs)

    for run in runs:
        for model in walkforward.MODELS:
            for metric in walkforward.SUMMARY_METRICS:
                recorded = run.summary["per_model"][model][metric]["mean"]
                reported = summary["per_model"][model][metric]["per_run_mean"][run.name]
                assert reported == pytest.approx(recorded, abs=1e-6)


# --- 7. dataset-backed regime diagnostics -------------------------------------
#
# The processed dataset and the raw candles are built here, per test, from
# generated candles. Nothing reads a committed dataset, and the assertions are
# on exact numbers computed independently of the code under test.

RAW_ROWS = 900


def build_dataset_pair(
    directory: Path, *, rows: int = RAW_ROWS, seed: int = 5, horizon: int = 6
):
    """A processed dataset and the raw candles it was built from."""
    directory.mkdir(parents=True, exist_ok=True)
    candles = generate_candles(rows=rows, seed=seed)
    raw_path = directory / "raw.parquet"
    candles.to_parquet(raw_path, index=False)

    frame, meta = build_dataset(
        candles,
        FeatureSpec(),
        TargetSpec(horizon=horizon),
        exchange="synthetic",
        pair="SYNTH/USDT",
        timeframe="1h",
    )
    processed = directory / "processed.parquet"
    save_dataset(processed, frame, meta)
    return processed, raw_path, frame, meta


@pytest.fixture(scope="module")
def dataset_pair(tmp_path_factory):
    return build_dataset_pair(tmp_path_factory.mktemp("regime_data"))


def artifact_for(directory: Path, frame, meta, *, seed: int = 42, folds: int = 3) -> Path:
    """A run whose geometry — and scored-sample counts — match the real dataset.

    The sample counts are computed with the production index logic, so the
    artifact is self-consistent in the way a real run's is: the diagnostics
    cross-check the two, and a hand-picked number would fail that check.
    """
    _, plans = plan_for(len(frame), folds)
    segment_ids = (
        frame["segment_id"].to_numpy(dtype=np.int64) if "segment_id" in frame.columns else None
    )
    horizon = dict(meta.target_spec)["horizon"]
    outer_samples = [
        len(sample_indices(plan.outer, SEQ_LEN, horizon, segment_ids=segment_ids))
        for plan in plans
    ]
    return write_run(
        directory,
        seed=seed,
        n_rows=len(frame),
        folds=folds,
        outer_samples=outer_samples,
        dataset={
            "rows": len(frame),
            "pair": meta.pair,
            "timeframe": meta.timeframe,
            "exchange": meta.exchange,
            "start": meta.start,
            "end": meta.end,
            "feature_names": list(meta.feature_names),
            "feature_spec": dict(meta.feature_spec),
            "target_spec": dict(meta.target_spec),
        },
    )


def research_for(dataset_pair, frame):
    processed, _, _, _ = dataset_pair
    return regime.load_research_frame(
        processed,
        sealed_test_start=resolve_sealed_boundary(frame["date"], contract=CONTRACT).start_row,
    )


def test_the_research_frame_stops_at_the_sealed_boundary(dataset_pair):
    """Sealed rows are absent from the object, not merely unread."""
    processed, _, frame, _ = dataset_pair
    boundary = resolve_sealed_boundary(frame["date"], contract=CONTRACT).start_row
    research = regime.load_research_frame(processed, sealed_test_start=boundary)

    assert len(research.frame) == boundary
    assert boundary < len(frame), "the fixture must actually have sealed rows to withhold"
    # The last row in the frame is the last research row, by index and by date.
    assert research.frame["date"].iloc[-1] == pd.to_datetime(
        frame["date"].iloc[boundary - 1], utc=True
    )


def test_a_block_at_or_past_the_boundary_is_refused(dataset_pair):
    processed, _, frame, _ = dataset_pair
    boundary = resolve_sealed_boundary(frame["date"], contract=CONTRACT).start_row
    research = regime.load_research_frame(processed, sealed_test_start=boundary)

    with pytest.raises(regime.RegimeDataError, match="beyond the sealed"):
        research.block(boundary - 10, boundary + 1)
    # Right up to the boundary is fine; one past it is not.
    assert len(research.block(boundary - 10, boundary)) == 10


def test_block_statistics_use_only_the_rows_of_that_block(dataset_pair):
    """Every statistic recomputed independently from the same slice."""
    processed, _, frame, _ = dataset_pair
    boundary = resolve_sealed_boundary(frame["date"], contract=CONTRACT).start_row
    research = regime.load_research_frame(processed, sealed_test_start=boundary)

    start, end = 400, 520
    stats = regime.block_statistics(research, start, end, SEQ_LEN)
    scored = research.scored_rows(start, end, SEQ_LEN)
    block = frame.iloc[scored]
    future = block["future_return"].to_numpy(dtype=float)

    assert stats["row_range"] == [start, end]
    assert stats["block_rows"] == end - start
    assert stats["scored_rows"] == len(scored)
    # The scored set is a strict subset: warm-up and label embargo cost rows.
    assert 0 < len(scored) < end - start
    assert stats["future_return"]["mean"] == pytest.approx(float(future.mean()), abs=1e-8)
    assert stats["future_return"]["median"] == pytest.approx(
        float(np.median(future)), abs=1e-8
    )
    assert stats["future_return"]["std"] == pytest.approx(float(future.std(ddof=1)), abs=1e-8)
    assert stats["future_return"]["mean_abs"] == pytest.approx(
        float(np.abs(future).mean()), abs=1e-8
    )
    assert stats["future_return"]["fraction_positive"] == pytest.approx(
        float((future > 0).mean()), abs=1e-8
    )
    assert stats["future_return"]["fraction_negative"] == pytest.approx(
        float((future < 0).mean()), abs=1e-8
    )


def test_feature_statistics_are_exact_including_the_percentiles(dataset_pair):
    processed, _, frame, _ = dataset_pair
    boundary = resolve_sealed_boundary(frame["date"], contract=CONTRACT).start_row
    research = regime.load_research_frame(processed, sealed_test_start=boundary)

    start, end = 300, 480
    stats = regime.block_statistics(research, start, end, SEQ_LEN)["features"]
    block = frame.iloc[research.scored_rows(start, end, SEQ_LEN)]

    for name, wanted in regime.FEATURE_STATS.items():
        values = block[name].to_numpy(dtype=float)
        assert set(stats[name]) == set(wanted), f"{name} reported the wrong statistics"
        if "mean" in wanted:
            assert stats[name]["mean"] == pytest.approx(float(values.mean()), abs=1e-8)
        if "median" in wanted:
            assert stats[name]["median"] == pytest.approx(float(np.median(values)), abs=1e-8)
        if "std" in wanted:
            assert stats[name]["std"] == pytest.approx(float(values.std(ddof=1)), abs=1e-8)
        if "p90" in wanted:
            assert stats[name]["p90"] == pytest.approx(
                float(np.percentile(values, 90)), abs=1e-8
            )
        if "p90_abs" in wanted:
            assert stats[name]["p90_abs"] == pytest.approx(
                float(np.percentile(np.abs(values), 90)), abs=1e-8
            )
        if "fraction_positive" in wanted:
            assert stats[name]["fraction_positive"] == pytest.approx(
                float((values > 0).mean()), abs=1e-8
            )


def test_the_target_distribution_is_exact(dataset_pair):
    processed, _, frame, _ = dataset_pair
    boundary = resolve_sealed_boundary(frame["date"], contract=CONTRACT).start_row
    research = regime.load_research_frame(processed, sealed_test_start=boundary)

    start, end = 250, 500
    distribution = regime.block_statistics(research, start, end, SEQ_LEN)[
        "target_distribution"
    ]
    scored = research.scored_rows(start, end, SEQ_LEN)
    block = frame.iloc[scored]

    assert set(distribution) == {c.value for c in CLASS_ORDER}
    assert sum(entry["count"] for entry in distribution.values()) == len(scored)
    assert sum(entry["fraction"] for entry in distribution.values()) == pytest.approx(1.0)

    for index, klass in enumerate(CLASS_ORDER):
        selected = block[block["target"] == index]
        entry = distribution[klass.value]
        assert entry["count"] == len(selected)
        assert entry["fraction"] == pytest.approx(len(selected) / len(scored), abs=1e-8)
        if len(selected):
            returns = selected["future_return"].to_numpy(dtype=float)
            assert entry["mean_future_return"] == pytest.approx(
                float(returns.mean()), abs=1e-8
            )
            assert entry["median_future_return"] == pytest.approx(
                float(np.median(returns)), abs=1e-8
            )


def identity_of(frame, meta) -> dict[str, Any]:
    """The artifact's `dataset` block for a dataset built in this suite."""
    return {
        "rows": len(frame),
        "exchange": meta.exchange,
        "pair": meta.pair,
        "timeframe": meta.timeframe,
        "start": meta.start,
        "end": meta.end,
        "feature_names": list(meta.feature_names),
        "feature_spec": dict(meta.feature_spec),
        "target_spec": dict(meta.target_spec),
    }


def test_the_matching_dataset_identity_is_accepted(dataset_pair):
    """The control: the right dataset loads, so the refusals below mean something."""
    processed, _, frame, meta = dataset_pair
    research = regime.load_research_frame(
        processed,
        sealed_test_start=resolve_sealed_boundary(frame["date"], contract=CONTRACT).start_row,
        identity=identity_of(frame, meta),
    )
    assert research.timeframe == meta.timeframe
    assert research.feature_names == list(meta.feature_names)


@pytest.mark.parametrize(
    "field, wrong_value",
    [
        pytest.param("rows", 999_999, id="rows"),
        pytest.param("exchange", "kraken", id="exchange"),
        pytest.param("pair", "ETH/USDT", id="pair"),
        pytest.param("timeframe", "4h", id="timeframe"),
        pytest.param("start", "2019-01-01T00:00:00+00:00", id="start"),
        pytest.param("end", "2030-01-01T00:00:00+00:00", id="end"),
        pytest.param("feature_names", ["not", "these"], id="feature_names"),
        pytest.param("feature_spec", {"ema_fast": 999}, id="feature_spec"),
        pytest.param("target_spec", {"horizon": 99}, id="target_spec"),
    ],
)
def test_every_identity_dimension_fails_closed(dataset_pair, field, wrong_value):
    """A dataset that disagrees on any recorded field is refused, not reindexed.

    Row count alone is not enough: a different pair, exchange, timeframe, span,
    feature contract or target definition is a different experiment, and its row
    indices point at different candles however well the shapes line up.
    """
    processed, _, frame, meta = dataset_pair
    identity = {**identity_of(frame, meta), field: wrong_value}

    with pytest.raises(regime.RegimeDataError) as raised:
        regime.load_research_frame(
            processed,
            sealed_test_start=resolve_sealed_boundary(
                frame["date"], contract=CONTRACT
            ).start_row,
            identity=identity,
        )
    assert field in str(raised.value)
    assert "row indices would address different candles" in str(raised.value)


def test_metadata_is_cross_checked_against_the_frames_own_timestamps(dataset_pair, tmp_path):
    """A stale or edited sidecar must not be taken at its word."""
    processed, _, frame, meta = dataset_pair
    copied = tmp_path / "copy.parquet"
    copied.write_bytes(Path(processed).read_bytes())
    sidecar = Path(str(copied) + ".meta.json")
    payload = json.loads(Path(str(processed) + ".meta.json").read_text())
    payload["start"] = "2019-06-01T00:00:00+00:00"
    sidecar.write_text(json.dumps(payload))

    with pytest.raises(regime.RegimeDataError, match="metadata says the data starts"):
        regime.load_research_frame(
            copied,
            sealed_test_start=resolve_sealed_boundary(
                frame["date"], contract=CONTRACT
            ).start_row,
            identity={**identity_of(frame, meta), "start": payload["start"]},
        )


# --- 8. raw OHLCV alignment ---------------------------------------------------
def test_raw_candles_are_matched_on_timestamps_not_position(dataset_pair):
    """The processed dataset dropped warm-up rows, so the offset is real."""
    processed, raw_path, frame, _ = dataset_pair
    boundary = resolve_sealed_boundary(frame["date"], contract=CONTRACT).start_row
    research = regime.load_research_frame(processed, sealed_test_start=boundary)
    raw = regime.load_raw_ohlcv(raw_path)

    start, end = 200, 260
    timestamps = research.block(start, end)["date"]
    aligned = regime.align_raw(raw, timestamps, "1h")

    assert list(aligned.index) == list(pd.to_datetime(timestamps, utc=True))
    # Positional alignment would have picked different candles entirely.
    positional = raw.iloc[start:end]
    assert list(positional.index) != list(
        aligned.index
    ), "the fixture must have a warm-up offset, or this test proves nothing"


def test_a_missing_raw_timestamp_fails_closed(dataset_pair, tmp_path):
    processed, raw_path, frame, _ = dataset_pair
    boundary = resolve_sealed_boundary(frame["date"], contract=CONTRACT).start_row
    research = regime.load_research_frame(processed, sealed_test_start=boundary)
    timestamps = research.block(200, 260)["date"]

    raw = pd.read_parquet(raw_path)
    holed = raw[raw["date"] != timestamps.iloc[5]]
    path = tmp_path / "holed.parquet"
    holed.to_parquet(path, index=False)

    with pytest.raises(regime.RegimeDataError, match="no raw candle"):
        regime.align_raw(regime.load_raw_ohlcv(path), timestamps, "1h")


def test_duplicate_raw_timestamps_fail_closed(dataset_pair, tmp_path):
    _, raw_path, _, _ = dataset_pair
    raw = pd.read_parquet(raw_path)
    path = tmp_path / "duplicated.parquet"
    pd.concat([raw, raw.iloc[[7]]], ignore_index=True).to_parquet(path, index=False)

    with pytest.raises(regime.RegimeDataError, match="duplicate timestamps"):
        regime.load_raw_ohlcv(path)


def test_an_incompatible_timeframe_fails_closed(dataset_pair):
    processed, raw_path, frame, _ = dataset_pair
    boundary = resolve_sealed_boundary(frame["date"], contract=CONTRACT).start_row
    research = regime.load_research_frame(processed, sealed_test_start=boundary)
    timestamps = research.block(200, 240)["date"]

    with pytest.raises(regime.RegimeDataError, match="spaced .* apart"):
        regime.align_raw(regime.load_raw_ohlcv(raw_path), timestamps, "4h")


def test_raw_without_ohlcv_columns_fails_closed(tmp_path):
    path = tmp_path / "not_ohlcv.parquet"
    pd.DataFrame({"date": pd.date_range("2023-01-01", periods=3, tz="UTC")}).to_parquet(
        path, index=False
    )
    with pytest.raises(regime.RegimeDataError, match="missing OHLCV column"):
        regime.load_raw_ohlcv(path)


def test_raw_block_statistics_are_exact(dataset_pair):
    processed, raw_path, frame, _ = dataset_pair
    boundary = resolve_sealed_boundary(frame["date"], contract=CONTRACT).start_row
    research = regime.load_research_frame(processed, sealed_test_start=boundary)
    raw = regime.load_raw_ohlcv(raw_path)

    start, end = 300, 400
    timestamps = research.block(start, end)["date"]
    stats = regime.raw_block_statistics(raw, timestamps, "1h")

    candles = raw.loc[pd.to_datetime(timestamps, utc=True).to_numpy()]
    close = candles["close"].to_numpy(dtype=float)
    returns = close[1:] / close[:-1] - 1.0

    assert stats["candles"] == end - start
    assert stats["start_close"] == pytest.approx(float(close[0]), abs=1e-8)
    assert stats["end_close"] == pytest.approx(float(close[-1]), abs=1e-8)
    assert stats["market_return"] == pytest.approx(float(close[-1] / close[0] - 1), abs=1e-8)
    assert stats["mean_candle_return"] == pytest.approx(float(returns.mean()), abs=1e-8)
    assert stats["candle_return_std"] == pytest.approx(float(returns.std(ddof=1)), abs=1e-8)
    assert stats["annualised_volatility"] == pytest.approx(
        float(returns.std(ddof=1) * np.sqrt(24 * 365)), abs=1e-6
    )
    assert stats["mean_abs_candle_return"] == pytest.approx(
        float(np.abs(returns).mean()), abs=1e-8
    )
    peak = np.maximum.accumulate(close)
    assert stats["max_drawdown"] == pytest.approx(float((1 - close / peak).max()), abs=1e-8)
    assert stats["positive_candle_fraction"] + stats["negative_candle_fraction"] <= 1.0


def test_raw_statistics_skip_returns_across_a_market_gap(dataset_pair, tmp_path):
    """A missing exchange candle is not a price move."""
    processed, raw_path, frame, _ = dataset_pair
    boundary = resolve_sealed_boundary(frame["date"], contract=CONTRACT).start_row
    research = regime.load_research_frame(processed, sealed_test_start=boundary)
    timestamps = research.block(300, 360)["date"]

    raw = regime.load_raw_ohlcv(raw_path)
    # Drop a candle that is *not* in the block, but sits inside its span, so the
    # block's own timestamps still all resolve while one pair is non-adjacent.
    gapped = raw.drop(index=timestamps.iloc[10])
    remaining = timestamps[timestamps != timestamps.iloc[10]]

    stats = regime.raw_block_statistics(gapped, remaining, "1h")
    assert stats["gap_pairs_skipped"] == 1
    assert stats["candles"] == len(remaining)


# --- 9. best/worst selection, seed-invariant market, identity -----------------
def stability_runs(tmp_path, net_by_fold):
    """Runs sharing geometry, whose per-fold MTST returns are dictated per seed."""
    runs = []
    for index, (name, values) in enumerate(net_by_fold.items()):
        directory = write_run(
            tmp_path / name,
            seed=42 + 100 * index,
            folds=len(values),
            returns={
                "majority_baseline": [-0.01] * len(values),
                "momentum_baseline": [0.0] * len(values),
                "mtst": values,
            },
        )
        runs.append(wf_diagnostics.load_run(directory))
    return runs


def test_best_and_worst_folds_come_from_the_data(tmp_path):
    """Fold 1 wins and fold 2 loses only because the numbers say so."""
    runs = stability_runs(
        tmp_path,
        {"run_a": [0.10, 0.90, -0.30], "run_b": [0.20, 0.70, -0.10]},
    )
    stability = wf_diagnostics.fold_model_stability(runs)
    assert wf_diagnostics.best_and_worst(stability) == (1, 2)

    # Reverse the ordering and the answer follows the data, not a constant.
    reversed_runs = stability_runs(
        tmp_path / "reversed",
        {"run_a": [0.90, 0.10, -0.30], "run_b": [0.70, 0.20, -0.10]},
    )
    assert wf_diagnostics.best_and_worst(
        wf_diagnostics.fold_model_stability(reversed_runs)
    ) == (0, 2)


def test_fold_stability_reports_every_requested_quantity(tmp_path):
    runs = stability_runs(tmp_path, {"a": [0.1, -0.2, 0.3], "b": [0.3, 0.2, 0.5]})
    stability = wf_diagnostics.fold_model_stability(runs)

    assert [entry["fold"] for entry in stability] == [0, 1, 2]
    fold_0 = stability[0]
    assert fold_0["net_return"]["mean"] == pytest.approx(0.2)
    assert fold_0["net_return"]["median"] == pytest.approx(0.2)
    assert fold_0["net_return"]["positive_seeds"] == 2
    assert fold_0["net_return"]["seeds"] == 2
    # Fold 1 has one negative seed out of two.
    assert stability[1]["net_return"]["positive_seeds"] == 1
    for field in ("directional_accuracy", "coverage", "n_trades", "threshold", "best_epoch"):
        assert {"mean", "std", "min", "max"} <= set(fold_0[field])


def test_market_statistics_are_identical_across_seeds(dataset_pair, tmp_path):
    """The market in a fold does not depend on which seed trained on it."""
    processed, raw_path, frame, meta = dataset_pair
    runs = [
        wf_diagnostics.load_run(artifact_for(tmp_path / name, frame, meta, seed=seed))
        for name, seed in (("seed_42", 42), ("seed_142", 142), ("seed_242", 242))
    ]
    assert wf_diagnostics.compare_runs(runs) == []

    research = regime.load_research_frame(
        processed, sealed_test_start=runs[0].sealed_test_start
    )
    raw = regime.load_raw_ohlcv(raw_path)
    reports = [wf_diagnostics.regime_report([run], research, raw) for run in runs]

    assert reports[0] == reports[1] == reports[2]
    assert all("market" in block for block in reports[0])


def test_a_differing_target_spec_blocks_aggregation(dataset_pair, tmp_path):
    processed, _, frame, meta = dataset_pair
    good = wf_diagnostics.load_run(artifact_for(tmp_path / "good", frame, meta))
    other = artifact_for(tmp_path / "other", frame, meta, seed=142)
    edit(other, lambda p: p["dataset"]["target_spec"].update(horizon=12))

    problems = wf_diagnostics.compare_runs([good, wf_diagnostics.load_run(other)])
    assert any("target_spec" in problem for problem in problems)


def test_a_differing_feature_contract_blocks_aggregation(dataset_pair, tmp_path):
    processed, _, frame, meta = dataset_pair
    good = wf_diagnostics.load_run(artifact_for(tmp_path / "good", frame, meta))
    other = artifact_for(tmp_path / "other", frame, meta, seed=142)
    edit(other, lambda p: p["dataset"]["feature_names"].append("an_extra_feature"))

    problems = wf_diagnostics.compare_runs([good, wf_diagnostics.load_run(other)])
    assert any("feature_names" in problem for problem in problems)


def test_seed_only_differences_do_not_block_aggregation(dataset_pair, tmp_path):
    """The whole point is comparing seeds; the seed itself must not disqualify them."""
    _, _, frame, meta = dataset_pair
    runs = [
        wf_diagnostics.load_run(artifact_for(tmp_path / f"s{seed}", frame, meta, seed=seed))
        for seed in (42, 142, 242, 342, 442)
    ]
    assert {run.seed for run in runs} == {42, 142, 242, 342, 442}
    assert wf_diagnostics.compare_runs(runs) == []


# --- 10. LONG / SHORT attribution ---------------------------------------------
def prediction_frame(**overrides) -> pd.DataFrame:
    """Six samples with hand-chosen actions, so the attribution is checkable by hand."""
    short_idx, hold_idx, long_idx = range(len(CLASS_ORDER))
    data = {
        "fold": [0] * 6,
        "seed": [42] * 6,
        "row_index": [100, 101, 102, 103, 104, 105],
        "timestamp": pd.date_range("2023-01-01", periods=6, freq="h", tz="UTC"),
        "true_target": [long_idx, hold_idx, short_idx, hold_idx, long_idx, hold_idx],
        "future_return": [0.05, 0.01, -0.04, 0.0, -0.02, 0.03],
        "p_short": [0.1, 0.2, 0.8, 0.2, 0.2, 0.1],
        "p_hold": [0.1, 0.6, 0.1, 0.6, 0.1, 0.1],
        "p_long": [0.8, 0.2, 0.1, 0.2, 0.7, 0.8],
        "selected_action": [long_idx, hold_idx, short_idx, hold_idx, long_idx, long_idx],
        "threshold": [0.6] * 6,
    }
    data.update(overrides)
    return pd.DataFrame(data)


def test_attribution_is_exact_and_uses_the_shared_cost_model():
    """Hand-computed: horizon 1 means every non-HOLD sample is its own trade."""
    spec = TargetSpec(horizon=1)
    cost = spec.cost_threshold
    result = regime.direction_attribution(prediction_frame(), spec)

    # LONG at rows 100 (+0.05), 104 (-0.02), 105 (+0.03); SHORT at 102 (-0.04 -> +0.04).
    long_returns = [0.05 - cost, -0.02 - cost, 0.03 - cost]
    short_returns = [0.04 - cost]

    assert result["long"]["trades"] == 3
    assert result["short"]["trades"] == 1
    assert result["hold_samples"] == 2
    assert result["long"]["mean_net_return"] == pytest.approx(np.mean(long_returns))
    assert result["long"]["median_net_return"] == pytest.approx(np.median(long_returns))
    assert result["long"]["additive_trade_return_sum"] == pytest.approx(sum(long_returns))
    assert result["long"]["hit_rate"] == pytest.approx(2 / 3, abs=1e-6)
    assert result["short"]["additive_trade_return_sum"] == pytest.approx(sum(short_returns))
    assert result["short"]["hit_rate"] == pytest.approx(1.0)
    assert result["long_coverage"] == pytest.approx(3 / 6, abs=1e-6)
    assert result["short_coverage"] == pytest.approx(1 / 6, abs=1e-6)


def test_attribution_totals_match_the_shared_trade_generator():
    """The split is a partition of exactly the trades nn.evaluate would take."""
    spec = TargetSpec(horizon=2)
    predictions = prediction_frame()
    result = regime.direction_attribution(predictions, spec)

    _, directions, returns = ev.realised_trades(
        predictions["selected_action"].to_numpy(np.int64),
        predictions["future_return"].to_numpy(float),
        spec,
    )
    assert result["long"]["trades"] + result["short"]["trades"] == len(returns)
    assert result["long"]["additive_trade_return_sum"] + result["short"][
        "additive_trade_return_sum"
    ] == pytest.approx(float(returns.sum()), abs=1e-8)
    assert result["long"]["trades"] == int((directions > 0).sum())


def test_attribution_generates_trades_per_run_not_across_runs():
    """One model walking one block: a second seed must not suppress the first's trade."""
    spec = TargetSpec(horizon=6)
    one_seed = prediction_frame()
    two_seeds = pd.concat([one_seed, prediction_frame(seed=[142] * 6)], ignore_index=True)

    single = regime.direction_attribution(one_seed, spec)
    doubled = regime.direction_attribution(two_seeds, spec)
    assert doubled["long"]["trades"] == 2 * single["long"]["trades"]
    assert doubled["short"]["trades"] == 2 * single["short"]["trades"]


def test_a_prediction_file_reaching_the_sealed_block_is_refused(tmp_path):
    path = tmp_path / "outer_predictions.parquet"
    prediction_frame(row_index=[100, 101, 102, 103, 104, 999]).to_parquet(path, index=False)

    with pytest.raises(regime.RegimeDataError, match="at or beyond the sealed"):
        regime.load_predictions(path, sealed_test_start=500)
    # Below the boundary it loads, so the guard above is doing the work.
    assert len(regime.load_predictions(path, sealed_test_start=1000)) == 6


def test_a_malformed_prediction_file_fails_clearly(tmp_path):
    path = tmp_path / "outer_predictions.parquet"
    prediction_frame().drop(columns=["p_long", "threshold"]).to_parquet(path, index=False)
    with pytest.raises(regime.RegimeDataError, match="missing prediction column"):
        regime.load_predictions(path, sealed_test_start=1000)

    empty = tmp_path / "empty.parquet"
    prediction_frame().iloc[:0].to_parquet(empty, index=False)
    with pytest.raises(regime.RegimeDataError, match="no predictions"):
        regime.load_predictions(empty, sealed_test_start=1000)


def test_absent_predictions_are_reported_not_approximated(tmp_path):
    runs = [wf_diagnostics.load_run(write_run(tmp_path / "legacy"))]
    report = wf_diagnostics.attribution_report(runs, TargetSpec())

    assert report["available"] is False
    assert "outer sample predictions were not persisted" in report["reason"]
    assert report["expected_file"] == walkforward.PREDICTIONS_NAME
    # No numbers are offered in place of the measurement.
    assert "long" not in report and "short" not in report


# --- 11. the regime mode end to end -------------------------------------------
@pytest.fixture(scope="module")
def real_regime_runs(tmp_path_factory):
    """Real walk-forward runs over a real processed dataset, plus its raw candles.

    Three seeds, so per-fold spread across seeds is a real number rather than a
    placeholder, and every artifact is written by ``nn.walkforward`` itself.
    """
    workspace = tmp_path_factory.mktemp("regime_e2e")
    processed, raw_path, frame, _ = build_dataset_pair(workspace, rows=1600, seed=13)

    directories = []
    for seed in (42, 142, 242):
        out = workspace / f"nested_seed_{seed}"
        assert (
            walkforward.main(
                [
                    "--dataset",
                    str(processed),
                    "--out",
                    str(out),
                    "--folds",
                    "3",
                    "--seed",
                    str(seed),
                    *TINY,
                ]
            )
            == 0
        )
        directories.append(out)
    return {
        "runs": directories,
        "dataset": processed,
        "raw": raw_path,
        "rows": len(frame),
        "boundary": resolve_sealed_boundary(frame["date"], contract=CONTRACT).start_row,
    }


def test_walkforward_persists_outer_predictions_below_the_boundary(real_regime_runs):
    """Every persisted row is an outer row, and none is sealed."""
    for directory in real_regime_runs["runs"]:
        predictions = pd.read_parquet(directory / walkforward.PREDICTIONS_NAME)
        assert list(predictions.columns) == list(regime.PREDICTION_COLUMNS)
        assert predictions["row_index"].max() < real_regime_runs["boundary"]

        artifact = json.loads((directory / wf_diagnostics.ARTIFACT_NAME).read_text())
        assert artifact["outer_predictions"] == walkforward.PREDICTIONS_NAME
        # Each persisted row falls inside its own fold's outer block.
        for fold in artifact["folds"]:
            start, end = fold["periods"]["outer_validation"]["row_range"]
            rows = predictions[predictions["fold"] == fold["fold"]]["row_index"]
            assert rows.min() >= start and rows.max() < end


def test_persisted_probabilities_agree_with_the_persisted_action(real_regime_runs):
    """selected_action is the action the reports scored, not a second reading."""
    predictions = pd.read_parquet(real_regime_runs["runs"][0] / walkforward.PREDICTIONS_NAME)
    proba = predictions[["p_short", "p_hold", "p_long"]].to_numpy(dtype=float)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    for threshold, group in predictions.groupby("threshold"):
        expected = ev.signals_from_proba(
            group[["p_short", "p_hold", "p_long"]].to_numpy(dtype=float), float(threshold)
        )
        np.testing.assert_array_equal(group["selected_action"].to_numpy(), expected)


def test_the_regime_cli_reports_every_requested_section(real_regime_runs, tmp_path, capsys):
    out = tmp_path / "btc_regimes_v1"
    exit_code = wf_diagnostics.main(
        [str(d) for d in real_regime_runs["runs"]]
        + [
            "--dataset",
            str(real_regime_runs["dataset"]),
            "--raw",
            str(real_regime_runs["raw"]),
            "--out",
            str(out),
        ]
    )
    assert exit_code == 0

    printed = capsys.readouterr().out
    for heading in (
        "## Integrity",
        "## Dataset / sealed-test status",
        "## Fold geometry",
        "## Market regime statistics",
        "## Outer validation across runs",
        "## Per-fold model stability across seeds",
        "## Best vs worst regime",
        "## LONG vs SHORT attribution",
        "## Candidate hypotheses",
        "## Limitations",
    ):
        assert heading in printed, f"{heading} missing from the report"

    payload = json.loads((out / wf_diagnostics.REPORT_JSON).read_text())
    analysis = payload["analysis"]
    assert payload["sealed_test_evaluated"] is False
    assert analysis["sealed_test_evaluated"] is False
    assert analysis["highest_outer_row"] < real_regime_runs["boundary"]
    assert len(analysis["regime"]) == 3
    assert analysis["best_fold"] != analysis["worst_fold"]
    assert len(analysis["hypotheses"]) == 3
    assert [h["rank"] for h in analysis["hypotheses"]] == [1, 2, 3]
    assert analysis["attribution"]["available"] is True
    assert (
        (out / wf_diagnostics.REPORT_MD)
        .read_text()
        .startswith("# Walk-forward regime diagnostics")
    )


def test_the_regime_cli_never_reads_a_row_at_or_beyond_the_boundary(
    real_regime_runs, tmp_path, monkeypatch
):
    """Spied on the block reads themselves, asserted on row indices."""
    boundary = real_regime_runs["boundary"]
    seen: list[tuple[int, int]] = []
    original = regime.ResearchFrame.block

    def spy(self, start, end):
        seen.append((start, end))
        return original(self, start, end)

    monkeypatch.setattr(regime.ResearchFrame, "block", spy)
    assert (
        wf_diagnostics.main(
            [str(d) for d in real_regime_runs["runs"]]
            + [
                "--dataset",
                str(real_regime_runs["dataset"]),
                "--raw",
                str(real_regime_runs["raw"]),
                "--out",
                str(tmp_path / "diag"),
            ]
        )
        == 0
    )

    assert seen, "the spy must have fired"
    assert max(end for _, end in seen) <= boundary
    # And the frame handed to those reads physically stops at the boundary.
    research = regime.load_research_frame(
        real_regime_runs["dataset"], sealed_test_start=boundary
    )
    assert len(research.frame) == boundary


def test_the_regime_statistics_match_a_direct_computation(real_regime_runs, tmp_path):
    """The reported block statistics equal the same statistics taken by hand."""
    out = tmp_path / "diag"
    wf_diagnostics.main(
        [str(d) for d in real_regime_runs["runs"]]
        + ["--dataset", str(real_regime_runs["dataset"]), "--out", str(out)]
    )
    analysis = json.loads((out / wf_diagnostics.REPORT_JSON).read_text())["analysis"]

    frame = pd.read_parquet(real_regime_runs["dataset"])
    research = regime.load_research_frame(
        real_regime_runs["dataset"], sealed_test_start=real_regime_runs["boundary"]
    )
    for block in analysis["regime"]:
        start, end = block["row_range"]
        assert end <= real_regime_runs["boundary"]
        # The rows the run actually scored, reconstructed independently.
        scored = research.scored_rows(start, end, analysis["seq_len"])
        rows = frame.iloc[scored]

        assert block["scored_rows"] == len(scored)
        assert block["block_rows"] == end - start
        assert block["future_return"]["mean"] == pytest.approx(
            float(rows["future_return"].mean()), abs=1e-8
        )
        assert block["features"]["atr_norm"]["p90"] == pytest.approx(
            float(np.percentile(rows["atr_norm"], 90)), abs=1e-8
        )


def test_the_attribution_is_exact_on_real_persisted_predictions(real_regime_runs, tmp_path):
    """Recomputed straight from the parquet files, independent of the report."""
    out = tmp_path / "diag"
    wf_diagnostics.main(
        [str(d) for d in real_regime_runs["runs"]]
        + ["--dataset", str(real_regime_runs["dataset"]), "--out", str(out)]
    )
    attribution = json.loads((out / wf_diagnostics.REPORT_JSON).read_text())["analysis"][
        "attribution"
    ]
    assert attribution["available"] is True

    combined = pd.concat(
        [pd.read_parquet(d / walkforward.PREDICTIONS_NAME) for d in real_regime_runs["runs"]],
        ignore_index=True,
    )
    spec = TargetSpec.from_dict(
        json.loads((real_regime_runs["runs"][0] / wf_diagnostics.ARTIFACT_NAME).read_text())[
            "dataset"
        ]["target_spec"]
    )
    expected = regime.direction_attribution(combined, spec)
    assert attribution["overall"] == expected
    assert attribution["overall"]["long"]["trades"] > 0


def test_raw_is_refused_without_a_dataset(real_regime_runs):
    with pytest.raises(SystemExit, match="--raw needs --dataset"):
        wf_diagnostics.main(
            [str(d) for d in real_regime_runs["runs"]]
            + ["--raw", str(real_regime_runs["raw"])]
        )


def test_a_mismatched_dataset_is_refused_by_the_cli(real_regime_runs, tmp_path):
    """Pointing at a differently sized dataset must not silently reindex."""
    other, _, _, _ = build_dataset_pair(tmp_path / "other", rows=1000, seed=99)
    with pytest.raises(SystemExit, match="[Rr]ow indices would address"):
        wf_diagnostics.main(
            [str(d) for d in real_regime_runs["runs"]] + ["--dataset", str(other)]
        )


def test_without_a_dataset_the_report_still_audits_and_says_what_is_missing(
    real_regime_runs, tmp_path, capsys
):
    """The five existing runs stay analysable with no dataset at hand."""
    assert (
        wf_diagnostics.main(
            [str(d) for d in real_regime_runs["runs"]] + ["--out", str(tmp_path / "diag")]
        )
        == 0
    )
    printed = capsys.readouterr().out

    assert "## Per-fold model stability across seeds" in printed
    assert "## Best vs worst regime" in printed
    assert "no regime statistics below" in printed
    assert "## Market regime statistics" not in printed


# --- 12. scored rows, not block rows ------------------------------------------
#
# The model is evaluated on the samples a block can produce, not on every row it
# spans: the first seq_len-1 rows cannot open a window, the last `horizon` rows
# cannot close a label, and a candidate straddling a market-data gap is dropped.
# Statistics used to interpret that evaluation have to be taken over the same
# rows, or they describe a slightly different stretch of market than the number
# they are explaining.


def gapped_research_frame(tmp_path, *, rows=900, gap_at=300, gap_len=5, horizon=4):
    """A processed dataset with a real market-data gap in the middle of a block."""
    candles = generate_candles(rows=rows, seed=21)
    # Delete candles so build_features records two segments, exactly as a real
    # exchange outage would.
    kept = pd.concat([candles.iloc[:gap_at], candles.iloc[gap_at + gap_len :]])
    frame, meta = build_dataset(
        kept.reset_index(drop=True),
        FeatureSpec(),
        TargetSpec(horizon=horizon),
        exchange="synthetic",
        pair="SYNTH/USDT",
        timeframe="1h",
    )
    processed = tmp_path / "gapped.parquet"
    save_dataset(processed, frame, meta)
    boundary = resolve_sealed_boundary(frame["date"], contract=CONTRACT).start_row
    research = regime.load_research_frame(processed, sealed_test_start=boundary)
    return research, frame, boundary


def test_the_scored_set_is_the_one_build_windows_produces(tmp_path):
    """Same rows, from the same function the run used — not a second reading."""
    research, frame, boundary = gapped_research_frame(tmp_path)
    assert research.segment_ids is not None, "the fixture must carry segment ids"

    seq_len, start, end = 16, 200, 400
    scored = research.scored_rows(start, end, seq_len)

    # The research frame stops at the boundary, so compare against the same rows.
    research_rows = frame.iloc[:boundary]
    features = research_rows[list(research.feature_names)].to_numpy(dtype=np.float64)
    _, _, expected = build_windows(
        features,
        research_rows["target"].to_numpy(dtype=np.int64),
        Split("outer_validation", start, end),
        seq_len,
        research.target_spec.horizon,
        segment_ids=research.segment_ids,
    )
    np.testing.assert_array_equal(scored, expected)


def test_gaps_and_warmup_make_the_scored_set_a_strict_subset(tmp_path):
    """The regression this section exists for: the two row sets really differ."""
    research, _, _ = gapped_research_frame(tmp_path)
    seq_len, start, end = 16, 200, 400
    scored = set(int(i) for i in research.scored_rows(start, end, seq_len))
    block = set(range(start, end))

    excluded = block - scored
    assert scored < block, "the scored set must be a strict subset"
    # Warm-up at the head, embargo at the tail, and the gap in between.
    assert set(range(start, start + seq_len - 1)) <= excluded
    assert set(range(end - research.target_spec.horizon, end)) <= excluded
    gap_excluded = {
        row
        for row in excluded
        if start + seq_len - 1 <= row < end - research.target_spec.horizon
    }
    assert gap_excluded, "the market-data gap must exclude rows in the interior too"


def test_excluded_rows_cannot_affect_the_processed_statistics(tmp_path):
    """Poison every row the model was not scored on; the statistics must not move.

    This is the assertion that would have caught the original bug: statistics
    taken over the whole block change when an unscored row changes, and
    statistics taken over the scored rows do not.
    """
    research, frame, boundary = gapped_research_frame(tmp_path)
    seq_len, start, end = 16, 200, 400
    baseline = regime.block_statistics(research, start, end, seq_len)

    scored = set(int(i) for i in research.scored_rows(start, end, seq_len))
    excluded = sorted(set(range(start, end)) - scored)
    assert excluded, "nothing would be proven if no row were excluded"

    poisoned = research.frame.copy()
    for column in ("future_return", *regime.FEATURE_STATS):
        poisoned.loc[poisoned.index[excluded], column] = 999.0
    poisoned.loc[poisoned.index[excluded], "target"] = 0
    spoiled = replace(research, frame=poisoned)

    assert regime.block_statistics(spoiled, start, end, seq_len) == baseline

    # Control: poisoning a *scored* row does move the numbers, so the assertion
    # above is not passing because the statistics are insensitive to everything.
    control = research.frame.copy()
    control.loc[control.index[sorted(scored)[0]], "future_return"] = 999.0
    moved = regime.block_statistics(replace(research, frame=control), start, end, seq_len)
    assert moved["future_return"]["mean"] != baseline["future_return"]["mean"]


def test_block_statistics_report_both_counts(tmp_path):
    """A reader must be able to see how many rows were dropped, not infer it."""
    research, _, _ = gapped_research_frame(tmp_path)
    seq_len, start, end = 16, 200, 400
    stats = regime.block_statistics(research, start, end, seq_len)

    assert stats["block_rows"] == end - start
    assert stats["scored_rows"] < stats["block_rows"]
    assert stats["seq_len"] == seq_len
    assert stats["horizon"] == research.target_spec.horizon
    first, last = stats["scored_row_range"]
    assert start <= first and last <= end


def test_a_block_too_short_to_score_is_refused(tmp_path):
    """Zero samples is an error, not a report over an empty set."""
    research, _, _ = gapped_research_frame(tmp_path)
    with pytest.raises(regime.RegimeDataError, match="no scored samples"):
        research.scored_rows(200, 210, seq_len=64)


def test_an_artifact_without_seq_len_cannot_be_regime_analysed(dataset_pair, tmp_path):
    """Guessing a sequence length would summarise rows the model never saw."""
    processed, _, frame, meta = dataset_pair
    directory = artifact_for(tmp_path / "no_seq_len", frame, meta)
    edit(directory, lambda p: p["config"].pop("seq_len"))

    with pytest.raises(SystemExit, match="does not record config.seq_len"):
        wf_diagnostics.main([str(directory), "--dataset", str(processed)])

    # Without --dataset the same artifact still audits and reports.
    assert wf_diagnostics.main([str(directory)]) == 0


def test_the_reconstruction_is_checked_against_the_recorded_sample_count(
    real_regime_runs, tmp_path
):
    """The run recorded how many samples it scored; the reconstruction must match.

    This turns "the same rows" from a claim into a checked invariant: if the
    dataset and the artifact disagree about which rows were evaluated, the report
    is refused rather than published with statistics over the wrong stretch.
    """
    out = tmp_path / "diag"
    assert (
        wf_diagnostics.main(
            [str(d) for d in real_regime_runs["runs"]]
            + ["--dataset", str(real_regime_runs["dataset"]), "--out", str(out)]
        )
        == 0
    )
    analysis = json.loads((out / wf_diagnostics.REPORT_JSON).read_text())["analysis"]
    artifact = json.loads(
        (real_regime_runs["runs"][0] / wf_diagnostics.ARTIFACT_NAME).read_text()
    )

    recorded = [fold["samples"]["outer_validation"] for fold in artifact["folds"]]
    assert [block["scored_rows"] for block in analysis["regime"]] == recorded
    # And the reconstruction is genuinely fewer rows than the block spans.
    assert all(block["scored_rows"] < block["block_rows"] for block in analysis["regime"])


def test_a_disagreement_about_the_scored_rows_is_refused(real_regime_runs, tmp_path):
    """A seq_len that does not reproduce the recorded count stops the report."""
    workspace = tmp_path / "tampered"
    workspace.mkdir()
    source = real_regime_runs["runs"][0]
    directory = workspace / source.name
    directory.mkdir()
    payload = json.loads((source / wf_diagnostics.ARTIFACT_NAME).read_text())
    payload["config"]["seq_len"] = payload["config"]["seq_len"] + 8
    (directory / wf_diagnostics.ARTIFACT_NAME).write_text(json.dumps(payload))

    with pytest.raises(SystemExit, match="disagree about which rows were evaluated"):
        wf_diagnostics.main([str(directory), "--dataset", str(real_regime_runs["dataset"])])


# --- 13. deterministic baselines are seed-invariant ---------------------------
#
# The market in a fold does not depend on the seed, and neither do the rules.
# The majority baseline is fitted on training rows and the momentum baseline has
# no fitted state at all, so two runs differing only in --seed must report
# byte-identical baseline numbers on every outer block. They did not: every model
# was scored at the MTST-selected threshold, so a rule whose probabilities were
# the class prior (or a fabricated confidence) changed action when that threshold
# moved.


def test_deterministic_baselines_are_identical_across_seed_only_reruns(
    real_regime_runs,
):
    """The floor must not move when only the seed moves.

    Same dataset, same geometry, different seed. The MTST reports are expected to
    differ; the baselines are not.
    """
    artifacts = [
        json.loads((directory / wf_diagnostics.ARTIFACT_NAME).read_text())
        for directory in real_regime_runs["runs"]
    ]
    seeds = {artifact["config"]["seed"] for artifact in artifacts}
    assert len(seeds) == len(artifacts) > 1, "the runs must actually differ in seed"

    reference = artifacts[0]
    for artifact in artifacts[1:]:
        for fold_a, fold_b in zip(reference["folds"], artifact["folds"]):
            assert (
                fold_a["periods"] == fold_b["periods"]
            ), "the runs must share geometry for this comparison to mean anything"
            for baseline in ("majority_baseline", "momentum_baseline"):
                assert (
                    fold_a["outer_validation"][baseline]
                    == fold_b["outer_validation"][baseline]
                ), f"fold {fold_a['fold']} {baseline} differs across seeds"


def test_the_mtst_reports_do_differ_across_seeds(real_regime_runs):
    """The control: if nothing differed, the test above would be vacuous."""
    artifacts = [
        json.loads((directory / wf_diagnostics.ARTIFACT_NAME).read_text())
        for directory in real_regime_runs["runs"]
    ]
    mtst = [
        [fold["outer_validation"]["mtst"]["trading"]["net_return"] for fold in a["folds"]]
        for a in artifacts
    ]
    assert len({tuple(values) for values in mtst}) > 1, "seeds must change the model"


def test_the_diagnostics_report_zero_baseline_seed_spread(real_regime_runs, tmp_path):
    """End to end: the seed-stability table shows the rules as flat lines."""
    out = tmp_path / "diag"
    assert (
        wf_diagnostics.main(
            [str(d) for d in real_regime_runs["runs"]]
            + ["--dataset", str(real_regime_runs["dataset"]), "--out", str(out)]
        )
        == 0
    )
    summary = json.loads((out / wf_diagnostics.REPORT_JSON).read_text())["summary"]

    for baseline in ("majority_baseline", "momentum_baseline"):
        for metric in walkforward.SUMMARY_METRICS:
            stats = summary["per_model"][baseline][metric]
            assert stats["across_runs"]["std"] == 0.0, f"{baseline}.{metric} moved"
            for fold in stats["per_fold"]:
                assert fold["std"] == 0.0, f"{baseline}.{metric} moved within a fold"

    # The model, by contrast, does move: the assertion above is not vacuous.
    assert (
        summary["per_model"]["mtst"]["net_return"]["across_runs"]["std"] > 0.0
    ), "the model must vary across seeds, or this proves nothing"


# --- 14. hypothesis wording follows the observed distributions ----------------
def separability_of(net_by_fold, tmp_path):
    """The separability hypothesis for a controlled set of per-fold returns."""
    runs = stability_runs(tmp_path, net_by_fold)
    stability = wf_diagnostics.fold_model_stability(runs)
    best, worst = wf_diagnostics.best_and_worst(stability)
    comparison = wf_diagnostics.compare_best_worst(None, stability, best, worst)
    return (
        wf_diagnostics._separability_candidate(
            stability,
            best,
            worst,
            lambda *metrics: max(
                (
                    abs(row["relative_difference"] or 0.0)
                    for row in comparison
                    if row["metric"] in metrics
                ),
                default=0.0,
            ),
        ),
        best,
        worst,
    )


def test_non_overlapping_folds_are_not_called_inseparable(tmp_path):
    """The real BTC shape: worst fold negative in every seed, ranges disjoint.

    The wording used to be fixed, and said the folds were "not cleanly separable
    yet" regardless of what the numbers showed.
    """
    candidate, best, worst = separability_of(
        {
            "a": [0.388, -0.125],
            "b": [0.150, -0.249],
            "c": [-0.012, -0.200],
        },
        tmp_path,
    )
    observed = candidate["observed"]

    assert (best, worst) == (0, 1)
    assert observed["ranges_overlap"] is False
    assert observed["worst_positive_seeds"] == 0
    assert observed["seeds"] == 3
    assert "consistently worse" in candidate["hypothesis"]
    assert "negative in all 3 observed seeds" in candidate["evidence"]
    assert "no overlap" in candidate["evidence"]
    # It must still refuse to generalise from one fold per regime.
    assert "sample of one" in candidate["evidence"]
    assert "independent periods" in candidate["hypothesis"]
    assert "not cleanly separable" not in candidate["evidence"]


def test_overlapping_folds_are_called_inseparable(tmp_path):
    """The other branch, on data where the ranges genuinely overlap."""
    candidate, best, worst = separability_of(
        {"a": [0.30, 0.20], "b": [0.10, 0.25], "c": [0.22, 0.05]},
        tmp_path,
    )
    observed = candidate["observed"]

    assert observed["ranges_overlap"] is True
    assert "unlucky draw" in candidate["hypothesis"]
    assert "the ranges overlap" in candidate["evidence"]
    assert "consistently worse" not in candidate["hypothesis"]


def test_a_worst_fold_that_is_sometimes_positive_is_described_as_such(tmp_path):
    """Disjoint ranges, but the worst fold is not negative everywhere."""
    candidate, _, _ = separability_of(
        {"a": [0.90, 0.20], "b": [0.80, 0.10], "c": [0.70, -0.05]},
        tmp_path,
    )
    assert candidate["observed"]["ranges_overlap"] is False
    assert candidate["observed"]["worst_positive_seeds"] == 2
    assert "positive in only 2/3 observed seeds" in candidate["evidence"]


def test_the_observed_numbers_back_the_wording(tmp_path):
    """The claim and the numbers that chose it travel together in the report."""
    candidate, best, worst = separability_of(
        {"a": [0.40, -0.10], "b": [0.20, -0.30]}, tmp_path
    )
    observed = candidate["observed"]

    assert observed["best_fold"] == best and observed["worst_fold"] == worst
    assert observed["best_range"] == [pytest.approx(0.2), pytest.approx(0.4)]
    assert observed["worst_range"] == [pytest.approx(-0.3), pytest.approx(-0.1)]
    for value in observed["best_range"] + observed["worst_range"]:
        assert f"{value:+.6f}" in candidate["evidence"]


def test_the_separability_hypothesis_reaches_the_report(real_regime_runs, tmp_path):
    """End to end: whichever branch fires, it carries its observed numbers."""
    out = tmp_path / "diag"
    wf_diagnostics.main(
        [str(d) for d in real_regime_runs["runs"]]
        + ["--dataset", str(real_regime_runs["dataset"]), "--out", str(out)]
    )
    analysis = json.loads((out / wf_diagnostics.REPORT_JSON).read_text())["analysis"]

    with_observed = [h for h in analysis["hypotheses"] if "observed" in h]
    if with_observed:
        observed = with_observed[0]["observed"]
        assert observed["best_fold"] == analysis["best_fold"]
        assert observed["worst_fold"] == analysis["worst_fold"]
        assert isinstance(observed["ranges_overlap"], bool)


# --- 15. artifacts produced before the baseline fix are labelled --------------
def test_legacy_baseline_scoring_is_detected_and_labelled(tmp_path, capsys):
    """A deterministic baseline that moved across seeds dates the artifact.

    It cannot be a finding: the majority rule is fitted on training rows and the
    momentum rule on nothing, so on identical geometry and data their numbers are
    a constant. Spread means the runs predate the fix that stopped baselines being
    scored at the threshold selected for the model.
    """
    runs = []
    for index, (name, majority) in enumerate(
        {"run_a": [-0.01, -0.02, -0.03], "run_b": [-0.05, -0.02, -0.03]}.items()
    ):
        runs.append(
            write_run(
                tmp_path / name,
                seed=42 + 100 * index,
                returns={
                    "majority_baseline": majority,
                    "momentum_baseline": [0.0, 0.0, 0.0],
                    "mtst": [0.1, 0.2, 0.3],
                },
            )
        )

    assert (
        wf_diagnostics.main(
            [str(r) for r in runs],
        )
        == 0
    )
    printed = capsys.readouterr().out
    assert "predate the baseline scoring fix" in printed
    assert "majority_baseline shows" in printed
    assert "MTST columns are unaffected" in printed


def test_a_clean_run_set_carries_no_legacy_warning(tmp_path, capsys):
    """The control: constant baselines produce no note at all."""
    runs = [
        write_run(
            tmp_path / name,
            seed=seed,
            returns={
                "majority_baseline": [-0.01, -0.02, -0.03],
                "momentum_baseline": [0.0, 0.0, 0.0],
                "mtst": mtst,
            },
        )
        for name, seed, mtst in (
            ("run_a", 42, [0.1, 0.2, 0.3]),
            ("run_b", 142, [0.2, 0.1, 0.4]),
        )
    ]
    out = tmp_path / "diag"
    assert wf_diagnostics.main([str(r) for r in runs] + ["--out", str(out)]) == 0

    assert "predate the baseline scoring fix" not in capsys.readouterr().out
    payload = json.loads((out / wf_diagnostics.REPORT_JSON).read_text())
    assert payload["legacy_baseline_scoring"] == []


def test_real_runs_written_after_the_fix_have_constant_baselines(real_regime_runs):
    """End to end on artifacts nn.walkforward wrote with the current code."""
    runs = [wf_diagnostics.load_run(d) for d in real_regime_runs["runs"]]
    summary = wf_diagnostics.aggregate(runs)
    assert wf_diagnostics.deterministic_baseline_drift(summary) == []


# An undefined metric is not a moved one. A deterministic baseline can
# legitimately never trade — a majority rule whose majority class is HOLD takes
# no position at any threshold — and then its risk-adjusted statistics are
# undefined in *every* run, reported as `None` rather than a fabricated zero.
# `_spread` passes that through as `std: None`, which `None != 0.0` used to read
# as non-zero spread, so a perfectly current study was told its artifacts predate
# the baseline scoring fix.
def never_trades(model: str):
    """Give one model the report `trading_metrics` writes when it takes no trade.

    The summary is re-derived by the production aggregator afterwards, so the
    artifact stays as self-consistent as the run that would have produced it.
    """

    def mutate(payload):
        for fold in payload["folds"]:
            fold["outer_validation"][model]["trading"].update(
                n_trades=0,
                net_return=0.0,
                exposure=0.0,
                max_drawdown=0.0,
                annualised_sharpe=None,
                per_trade_sharpe=None,
            )
        payload["summary"] = walkforward.summarise(payload["folds"])

    return mutate


def _seed_study(tmp_path, majority: dict[str, list[float]]) -> list[Path]:
    """Two current-schema seed runs whose momentum baseline never trades."""
    return [
        edit(
            write_run(
                tmp_path / name,
                seed=seed,
                returns={
                    "majority_baseline": majority[name],
                    "momentum_baseline": [0.0, 0.0, 0.0],
                    "mtst": mtst,
                },
            ),
            never_trades("momentum_baseline"),
        )
        for name, seed, mtst in (
            ("run_a", 42, [0.1, 0.2, 0.3]),
            ("run_b", 142, [0.2, 0.1, 0.4]),
        )
    ]


def test_a_baseline_that_never_trades_is_not_reported_as_drift(tmp_path, capsys):
    """The regression: undefined everywhere must not date a current study."""
    runs = _seed_study(
        tmp_path,
        {"run_a": [-0.01, -0.02, -0.03], "run_b": [-0.01, -0.02, -0.03]},
    )
    out = tmp_path / "diag"
    assert wf_diagnostics.main([str(r) for r in runs] + ["--out", str(out)]) == 0

    printed = capsys.readouterr().out
    assert "predate the baseline scoring fix" not in printed

    payload = json.loads((out / wf_diagnostics.REPORT_JSON).read_text())
    assert payload["legacy_baseline_scoring"] == []
    # Still current artifacts: the false banner was the only thing wrong.
    assert payload["metric_schema"] == wf_diagnostics.SCHEMA_CURRENT
    assert payload["skipped_metrics"] == []

    # And the metrics really are undefined rather than zero, or this proves
    # nothing about the case it is named for.
    momentum = payload["summary"]["per_model"]["momentum_baseline"]
    for metric in walkforward.NULLABLE_METRICS:
        assert momentum[metric]["across_runs"]["std"] is None
        assert all(fold["std"] is None for fold in momentum[metric]["per_fold"])


def test_a_baseline_that_moved_is_still_caught_beside_an_undefined_one(tmp_path):
    """The detector is narrowed, not switched off.

    Same two runs, but the majority baseline's net return differs between them —
    impossible for a rule with no fitted parameters on identical data, and still
    reported — while the never-trading momentum baseline is not.
    """
    runs = _seed_study(
        tmp_path,
        {"run_a": [-0.01, -0.02, -0.03], "run_b": [-0.05, -0.02, -0.03]},
    )
    summary = wf_diagnostics.aggregate([wf_diagnostics.load_run(r) for r in runs])

    assert wf_diagnostics.deterministic_baseline_drift(summary) == ["majority_baseline"]


# --- metric schema: narrow legacy tolerance, everything else fails closed -------
#
# The tempting implementation is "aggregate whichever metrics are present in
# every fold". It would read the committed pre-correction artifacts, and it
# would also let a *current* artifact that lost a field pass by quietly dropping
# it — which is the failure this module exists to catch. So the tolerance is
# keyed on a positive legacy fingerprint and nothing else.
def test_a_pre_correction_artifact_is_read_with_only_the_new_metrics_skipped(tmp_path):
    write_run(tmp_path / "legacy", schema="legacy")
    run = wf_diagnostics.load_run(tmp_path / "legacy")

    # The fixture must be a shape a pre-correction run actually wrote, not a
    # current report with fields deleted: otherwise the legacy path below is
    # tested against fiction. `regime.LEGACY_TRADING_KEYS` is itself pinned to
    # the historical read-off in tests/test_prediction_integrity_hotfix.py.
    trading = run.folds[0]["outer_validation"]["mtst"]["trading"]
    assert set(trading) == set(regime.LEGACY_TRADING_KEYS)

    assert wf_diagnostics.audit_run(run) == []
    schema, problems = wf_diagnostics.classify_schema(run)
    assert schema == wf_diagnostics.SCHEMA_LEGACY
    assert problems == []

    summary = wf_diagnostics.aggregate([run])
    # Two kinds of skip, for two different reasons: fields these runs never
    # recorded, and one they recorded under different semantics.
    skipped = set(wf_diagnostics.CURRENT_ONLY_METRICS) | set(wf_diagnostics.REDEFINED_METRICS)
    assert summary["skipped_metrics"] == sorted(skipped)
    # Exactly those, and every other metric still aggregated.
    aggregated = set(summary["per_model"]["mtst"])
    assert aggregated == set(walkforward.SUMMARY_METRICS) - skipped
    assert "net_return" in aggregated and "exposure" in aggregated


def test_the_pre_correction_warning_reaches_the_report(tmp_path):
    write_run(tmp_path / "legacy", schema="legacy")
    run = wf_diagnostics.load_run(tmp_path / "legacy")
    markdown = wf_diagnostics.to_markdown(
        [run], {run.name: []}, [], wf_diagnostics.aggregate([run])
    )
    assert "predate the risk-metric correction" in markdown
    assert "candles_per_year / horizon" in markdown
    assert "**not** comparable" in markdown
    for metric in wf_diagnostics.CURRENT_ONLY_METRICS:
        assert metric in markdown, "the warning must name what it skipped"


def test_a_current_artifact_missing_a_metric_fails_rather_than_dropping_it(tmp_path):
    """The corruption case. A gap with no legacy fingerprint is a problem."""
    directory = tmp_path / "corrupt"
    write_run(directory)
    artifact = directory / wf_diagnostics.ARTIFACT_NAME
    payload = json.loads(artifact.read_text())
    del payload["folds"][1]["outer_validation"]["mtst"]["trading"]["annualised_sharpe"]
    artifact.write_text(json.dumps(payload))

    run = wf_diagnostics.load_run(directory)
    schema, problems = wf_diagnostics.classify_schema(run)
    assert schema == ""
    assert any("refusing to aggregate around the gap" in p for p in problems)
    assert wf_diagnostics.audit_run(run), "a missing metric must be an integrity problem"


@pytest.mark.parametrize(
    "field",
    [
        "annualised_sharpe_reason",
        "sharpe_basis",
        "candle_max_drawdown",
        "elapsed_intervals",
        "per_trade_sharpe_reason",
    ],
)
def test_a_current_artifact_missing_any_required_risk_field_fails_closed(tmp_path, field):
    """Every current risk-report field is required, not just summary metrics."""
    directory = tmp_path / "missing_risk_field"
    write_run(directory)
    artifact = directory / wf_diagnostics.ARTIFACT_NAME
    payload = json.loads(artifact.read_text())
    del payload["folds"][0]["outer_validation"]["mtst"]["trading"][field]
    artifact.write_text(json.dumps(payload))

    run = wf_diagnostics.load_run(directory)
    schema, problems = wf_diagnostics.classify_schema(run)
    assert schema == ""
    assert any(field in problem for problem in problems)
    assert wf_diagnostics.audit_run(run)


def test_a_defined_current_sharpe_with_a_different_basis_is_refused(tmp_path):
    """Defined annualised Sharpes under unlike bases cannot be aggregated."""
    directory = tmp_path / "wrong_basis"
    write_run(directory)
    artifact = directory / wf_diagnostics.ARTIFACT_NAME
    payload = json.loads(artifact.read_text())
    payload["folds"][0]["outer_validation"]["mtst"]["trading"][
        "sharpe_basis"
    ] = "a different annualisation basis"
    artifact.write_text(json.dumps(payload))

    run = wf_diagnostics.load_run(directory)
    schema, problems = wf_diagnostics.classify_schema(run)
    assert schema == ""
    assert any("sharpe_basis" in problem for problem in problems)
    assert any("Refusing to aggregate unlike risk metrics" in problem for problem in problems)
    assert wf_diagnostics.audit_run(run)


def _legacy_run_with(tmp_path, name: str, mutate) -> wf_diagnostics.RunArtifact:
    """A pre-correction run whose first mtst trading block has been edited."""
    directory = tmp_path / name
    write_run(directory, schema="legacy")
    artifact = directory / wf_diagnostics.ARTIFACT_NAME
    payload = json.loads(artifact.read_text())
    mutate(payload["folds"][0]["outer_validation"]["mtst"]["trading"])
    artifact.write_text(json.dumps(payload))
    return wf_diagnostics.load_run(directory)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("annualised_sharpe_reason", ""),
        ("sharpe_basis", ev.SHARPE_BASIS),
        ("candle_max_drawdown", 0.05),
        ("elapsed_intervals", 83),
        ("per_trade_sharpe_reason", ""),
    ],
)
def test_a_legacy_report_carrying_any_current_only_risk_field_fails_closed(
    tmp_path, field, value
):
    """A hybrid is not a degraded read: the legacy tolerance is fingerprint-only.

    The pre-correction `sharpe` alongside a field only the corrected evaluator
    emits means one of the two is not what it claims, and which numbers are
    comparable to which is then unknown. `per_trade_sharpe_reason` is the case
    the shared validator alone cannot catch — `NON_REPRODUCIBLE_TRADING_METRICS`
    does not name it — so it is covered here explicitly.
    """
    run = _legacy_run_with(
        tmp_path, f"hybrid_{field}", lambda trading: trading.__setitem__(field, value)
    )

    schema, problems = wf_diagnostics.classify_schema(run)
    assert schema == ""
    assert any(field in problem for problem in problems)
    assert wf_diagnostics.audit_run(run), "a hybrid report must be an integrity problem"


def test_a_legacy_report_missing_a_historical_trading_key_fails_closed(tmp_path):
    """`turnover` is not an aggregated metric, so only the exact-set rule sees it."""
    run = _legacy_run_with(tmp_path, "legacy_short", lambda trading: trading.pop("turnover"))

    schema, problems = wf_diagnostics.classify_schema(run)
    assert schema == ""
    assert any("turnover" in problem for problem in problems)
    assert wf_diagnostics.audit_run(run)


def test_a_legacy_report_with_an_unknown_trading_key_fails_closed(tmp_path):
    """A key in neither generation dates the artifact to neither."""
    run = _legacy_run_with(
        tmp_path, "legacy_extra", lambda trading: trading.__setitem__("sortino", 1.5)
    )

    schema, problems = wf_diagnostics.classify_schema(run)
    assert schema == ""
    assert any("sortino" in problem for problem in problems)
    assert wf_diagnostics.audit_run(run)


def test_a_half_renamed_schema_fails_rather_than_being_treated_as_legacy(tmp_path):
    """Both field generations at once: semantics unknown, so no aggregate."""
    directory = tmp_path / "mixed"
    write_run(directory)
    artifact = directory / wf_diagnostics.ARTIFACT_NAME
    payload = json.loads(artifact.read_text())
    trading = payload["folds"][0]["outer_validation"]["mtst"]["trading"]
    trading["sharpe"] = 12.0  # the removed field, alongside the ones that replaced it
    artifact.write_text(json.dumps(payload))

    run = wf_diagnostics.load_run(directory)
    schema, problems = wf_diagnostics.classify_schema(run)
    assert schema == ""
    assert any("neither the old schema nor the new one" in p for p in problems)


def test_a_legacy_artifact_missing_an_unrelated_metric_still_fails(tmp_path):
    """The legacy fingerprint excuses exactly two absences, not a third."""
    directory = tmp_path / "legacy_gap"
    write_run(directory, schema="legacy")
    artifact = directory / wf_diagnostics.ARTIFACT_NAME
    payload = json.loads(artifact.read_text())
    del payload["folds"][0]["outer_validation"]["mtst"]["trading"]["exposure"]
    artifact.write_text(json.dumps(payload))

    run = wf_diagnostics.load_run(directory)
    schema, problems = wf_diagnostics.classify_schema(run)
    assert schema == ""
    assert problems


def test_a_legacy_run_and_a_current_run_are_not_comparable(tmp_path):
    """Averaging them would put two definitions under one column heading."""
    write_run(tmp_path / "legacy", schema="legacy")
    write_run(tmp_path / "current")
    runs = [
        wf_diagnostics.load_run(tmp_path / "legacy"),
        wf_diagnostics.load_run(tmp_path / "current"),
    ]
    problems = wf_diagnostics.compare_runs(runs)
    assert any("metric schema" in p for p in problems)


def test_the_economic_section_says_buy_and_hold_is_unavailable_for_old_runs(tmp_path):
    """Never inferred from a net-return sign."""
    write_run(tmp_path / "legacy", schema="legacy")
    run = wf_diagnostics.load_run(tmp_path / "legacy")
    markdown = wf_diagnostics.to_markdown(
        [run], {run.name: []}, [], wf_diagnostics.aggregate([run])
    )
    assert "## Against doing nothing" in markdown
    assert "CASH" in markdown
    assert "not available for these runs" in markdown
    assert markdown.index("Against the statistical / rule baselines") < markdown.index(
        "Against doing nothing"
    )


def test_the_baseline_section_disclaims_profitability(tmp_path):
    write_run(tmp_path / "current")
    run = wf_diagnostics.load_run(tmp_path / "current")
    markdown = wf_diagnostics.to_markdown(
        [run], {run.name: []}, [], wf_diagnostics.aggregate([run])
    )
    assert "not evidence of profitability" in markdown


def test_an_undefined_sharpe_survives_aggregation_as_undefined(tmp_path):
    """`None` must not be averaged in as a zero anywhere in the chain."""
    directory = tmp_path / "undefined"
    write_run(directory)
    artifact = directory / wf_diagnostics.ARTIFACT_NAME
    payload = json.loads(artifact.read_text())
    for fold in payload["folds"]:
        fold["outer_validation"]["mtst"]["trading"]["annualised_sharpe"] = None
    artifact.write_text(json.dumps(payload))

    run = wf_diagnostics.load_run(directory)
    assert wf_diagnostics.classify_schema(run)[0] == wf_diagnostics.SCHEMA_CURRENT
    summary = wf_diagnostics.aggregate([run])
    across = summary["per_model"]["mtst"]["annualised_sharpe"]["across_runs"]
    assert across["mean"] is None and across["defined"] == 0
    markdown = wf_diagnostics.to_markdown([run], {run.name: []}, [], summary)
    assert "| mtst | annualised_sharpe | n/a | n/a | n/a | n/a |" in markdown


def test_legacy_max_drawdown_is_skipped_as_redefined_not_averaged(tmp_path):
    """Present in both generations, computed differently in each.

    Pre-correction runs measured the running peak from the first completed
    trade instead of from starting capital, so their `max_drawdown` understates
    a strategy that was under water from the start. Averaging those values
    beside corrected ones would put two definitions under one heading, so the
    metric is skipped — and the report distinguishes "absent" from "recorded
    under different semantics", because the remedies differ.
    """
    write_run(tmp_path / "legacy", schema="legacy")
    run = wf_diagnostics.load_run(tmp_path / "legacy")
    summary = wf_diagnostics.aggregate([run])

    assert "max_drawdown" not in summary["per_model"]["mtst"]
    assert summary["skipped_because_redefined"] == ["max_drawdown"]
    assert summary["skipped_because_absent"] == sorted(wf_diagnostics.CURRENT_ONLY_METRICS)
    # Metrics whose definition did not change are still aggregated.
    assert "net_return" in summary["per_model"]["mtst"]
    assert "exposure" in summary["per_model"]["mtst"]

    markdown = wf_diagnostics.to_markdown([run], {run.name: []}, [], summary)
    assert "`max_drawdown` is skipped for a different" in markdown
    assert "starting capital" in markdown


def test_a_current_run_still_aggregates_max_drawdown(tmp_path):
    write_run(tmp_path / "current")
    run = wf_diagnostics.load_run(tmp_path / "current")
    summary = wf_diagnostics.aggregate([run])
    assert "max_drawdown" in summary["per_model"]["mtst"]
    assert summary["skipped_metrics"] == []
