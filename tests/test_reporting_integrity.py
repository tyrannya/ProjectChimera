"""Reports must not let a losing model read as a winning one.

Three separate ways a reader has been misled by this repository's own output,
each pinned here:

* a verdict that stated only the baseline result — true, and read as evidence of
  profitability while the model lost money in every fold it won;
* a results table with no economic reference in it, so "did it beat doing
  nothing?" had no answer to find;
* an ``artifacts/`` tree with five walk-forward runs and two diagnostics
  aggregates and nothing at all saying which generation was current.

These tests assert *wording and structure*, which is unusual, and deliberate:
the numbers were never the thing that misled anyone.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nn import walkforward

REPO = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO / "artifacts"
INDEX = ARTIFACTS / "README.md"

#: The one artifact directory the index may mark CURRENT.
CURRENT_BASELINE = "diagnostics/btc_regimes_v3"


# --- verdict wording ------------------------------------------------------------
def _reports(net_return: float):
    """One model's outer report, carrying every metric the summary aggregates."""
    return {
        "classification": {
            "macro_f1": 0.2,
            "directional_accuracy": 0.5,
            "coverage": 0.05,
            "calibration_error": 0.1,
        },
        "trading": {
            "n_trades": 10,
            "net_return": net_return,
            "annualised_sharpe": -0.5,
            "per_trade_sharpe": -0.2,
            "exposure": 0.05,
            "max_drawdown": 0.1,
        },
    }


def _fold(net_return: float, majority: float, momentum: float, hold: float | None):
    """One fold's outer reports, in the shape `summarise` reads."""
    outer = {
        "mtst": _reports(net_return),
        "majority_baseline": _reports(majority),
        "momentum_baseline": _reports(momentum),
    }
    if hold is not None:
        outer[walkforward.REFERENCES_KEY] = {
            "cash": {"net_return": 0.0},
            "buy_and_hold": {"available": True, "net_return": hold},
        }
    return {"outer_validation": outer}


def test_a_losing_model_that_beats_the_baselines_reads_as_losing():
    """The exact situation in `artifacts/walkforward/btc_nested_seed_342`.

    That artifact records "model beat both baselines in a majority of
    outer-validation folds" while its mean outer net return is -4.18%. Both
    halves of that are true; printing only the first is what made it misleading.
    """
    results = [_fold(-0.03, -0.20, -0.30, 0.10) for _ in range(4)]
    summary = walkforward.summarise(results)

    assert summary["mtst_beat_baselines_in_folds"] == 4
    verdict = summary["verdict"]

    # It says it beat the baselines...
    assert "beat both statistical/rule baselines in 4/4" in verdict
    # ...and it is not allowed to stop there.
    assert "LOST money" in verdict
    assert "-0.03" in verdict
    assert "0/4" in verdict, "it beat CASH in no fold, and the verdict must say so"
    assert "not evidence of profitability" in verdict


def test_the_verdict_never_states_the_baseline_result_alone():
    """Whatever the numbers, both clauses are present.

    A quote-sized fragment of this sentence must not be able to mean 'this
    works'.
    """
    for net, hold in ((-0.03, 0.10), (0.05, 0.01), (-0.01, -0.20), (0.0, 0.0)):
        summary = walkforward.summarise([_fold(net, -0.2, -0.3, hold) for _ in range(4)])
        verdict = summary["verdict"]
        assert "baselines" in verdict
        assert ("LOST money" in verdict) or ("made money" in verdict)
        assert "CASH" in verdict
        assert "buy-and-hold" in verdict
        assert "floor test, not evidence of profitability" in verdict


def test_beating_cash_is_counted_from_the_net_return_not_asserted():
    results = [_fold(0.02, -0.2, -0.3, 0.01), _fold(-0.02, -0.2, -0.3, 0.01)]
    economic = walkforward.summarise(results)["economic_comparison"]
    assert economic["beat_cash_folds"] == 1
    assert economic["beat_buy_and_hold_folds"] == 1
    assert economic["mean_buy_and_hold_net_return"] == pytest.approx(0.01)


def test_a_profitable_model_is_allowed_to_say_so():
    """The wording rule is honesty, not pessimism."""
    summary = walkforward.summarise([_fold(0.08, -0.2, -0.3, 0.01) for _ in range(4)])
    assert "made money" in summary["verdict"]
    assert "LOST money" not in summary["verdict"]


def test_missing_references_are_reported_as_missing_not_inferred():
    """A pre-references artifact must not have buy-and-hold guessed for it."""
    summary = walkforward.summarise([_fold(-0.03, -0.2, -0.3, None) for _ in range(3)])
    assert summary["economic_comparison"]["mean_buy_and_hold_net_return"] is None
    assert summary["economic_comparison"]["references_available"] == 0
    assert "not recorded" in summary["verdict"]


# --- report structure -----------------------------------------------------------
def test_the_markdown_separates_baselines_from_economic_references():
    """Two headings, because they answer two questions.

    A no-trade policy in the same table as the models reads as one more
    contender that scored badly, rather than the line they all have to clear.
    """
    results = []
    for fold in range(2):
        record = _fold(-0.03, -0.20, -0.30, 0.10)
        record["fold"] = fold
        record["periods"] = {
            region: {
                "start": "2023-01-01T00:00:00+00:00",
                "end": "2023-02-01T00:00:00+00:00",
                "rows": 100,
                "row_range": [fold * 100, (fold + 1) * 100],
            }
            for region in ("train", "inner_validation", "outer_validation")
        }
        record["selection"] = {"threshold": 0.5, "best_epoch": 2}
        references = record["outer_validation"][walkforward.REFERENCES_KEY]
        references["buy_and_hold"].update(
            gross_return=0.102,
            candle_max_drawdown=0.2,
            annualised_sharpe=0.4,
            spans_market_data_gap=False,
            candles_held=100,
            missing_candles=0,
        )
        results.append(record)

    summary = walkforward.summarise(results)
    sealed = {
        "row_range": [1000, 1200],
        "start": "2023-06-01T00:00:00+00:00",
        "end": "2023-07-01T00:00:00+00:00",
    }
    markdown = walkforward.to_markdown(results, summary, sealed)

    assert "### Statistical / rule baselines and the model" in markdown
    assert "### Economic references (not models, not baselines)" in markdown
    assert markdown.index("Statistical / rule baselines") < markdown.index(
        "Economic references"
    )
    assert "| CASH |" in markdown and "| buy-and-hold |" in markdown
    assert "floor, not a business case" in markdown
    # The two Sharpe columns are distinguished by name, and neither is "Sharpe".
    assert "ann. Sharpe" in markdown and "per-trade Sharpe" in markdown
    assert "| Sharpe |" not in markdown


# --- the artifact index ----------------------------------------------------------
def _index_rows() -> list[dict[str, str]]:
    """Parse the status table out of `artifacts/README.md`."""
    text = INDEX.read_text()
    rows = []
    header: list[str] | None = None
    for line in text.splitlines():
        if not line.startswith("|"):
            header = None
            continue
        cells = [cell.strip().strip("`*") for cell in line.strip("|").split("|")]
        if header is None:
            header = cells
            continue
        if set("".join(cells)) <= set("- "):
            continue
        if len(cells) == len(header):
            rows.append(dict(zip(header, cells)))
    return [row for row in rows if row.get("status")]


def test_the_index_names_exactly_one_current_artifact():
    rows = _index_rows()
    current = [row for row in rows if row["status"] == "CURRENT"]
    assert len(current) == 1, f"expected one CURRENT entry, got {[r['path'] for r in current]}"
    assert current[0]["path"] == CURRENT_BASELINE


def test_no_legacy_generation_is_labelled_current():
    """The failure this index exists to prevent.

    v1 and v2 are earlier generations of the same question. Marking one CURRENT
    — by edit, by copy, or by adding a row — would point a reader at superseded
    evidence, so it is asserted rather than trusted.
    """
    for row in _index_rows():
        if row["generation"] in ("v1", "v2"):
            assert row["status"] != "CURRENT", row["path"]


def test_every_committed_artifact_directory_appears_in_the_index():
    """An artifact absent from the index has no stated status at all."""
    listed = {row["path"] for row in _index_rows()}
    on_disk = {
        str(path.relative_to(ARTIFACTS)) for path in ARTIFACTS.glob("*/*") if path.is_dir()
    }
    assert on_disk, "no artifact directories found"
    assert (
        on_disk == listed
    ), f"unlisted: {sorted(on_disk - listed)}; stale: {sorted(listed - on_disk)}"


def test_every_indexed_path_exists_and_carries_a_status_file():
    for row in _index_rows():
        directory = ARTIFACTS / row["path"]
        assert directory.is_dir(), row["path"]
        status = directory / "STATUS.md"
        assert status.is_file(), f"{row['path']} has no STATUS.md"
        assert status.read_text().startswith(f"# {row['status']}")


def test_the_index_records_that_committed_artifacts_predate_the_metric_change():
    """Their `sharpe` column is not comparable to a current run's, and says so."""
    rows = _index_rows()
    assert rows
    for row in rows:
        assert row["metric semantics"] == "pre-correction", row["path"]
    text = INDEX.read_text()
    assert "not comparable" in text
    assert "annualised_sharpe" in text


def test_the_index_states_that_the_v3_source_runs_are_absent():
    """Missing provenance is reported, never solved by relabelling v1.

    `btc_regimes_v3` aggregates walk-forward runs that are not in this
    repository. The index has to say so, and has to say that the committed v1
    directories are not those runs.
    """
    rows = {row["path"]: row for row in _index_rows()}
    assert rows[CURRENT_BASELINE]["source runs present"] == "no"
    assert "btc_nested_v3_seed" in rows[CURRENT_BASELINE]["source runs"]

    text = INDEX.read_text()
    assert "not fixed by copying or renaming" in text.lower()
    for missing in ("btc_nested_v3_seed_", "btc_nested_v2_seed_"):
        assert missing in text


def test_no_directory_named_as_a_v3_source_run_exists():
    """Guards the other half: the index says absent, so they must be absent.

    If someone later creates one of these directories by copying a v1 run, this
    fails rather than letting fabricated provenance pass review.
    """
    for seed in (42, 142, 242, 342, 442):
        for generation in ("v2", "v3"):
            path = ARTIFACTS / "walkforward" / f"btc_nested_{generation}_seed_{seed}"
            assert not path.exists(), (
                f"{path} exists, but the index records it as absent. If it is a real "
                "run, update artifacts/README.md; if it is a copy of another run, "
                "delete it."
            )
