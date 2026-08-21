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

import re
import subprocess
from pathlib import Path

import pytest

from nn import walkforward
from nn.data_fingerprint import ResearchInputFingerprint
from nn.research_contract import load_contract

REPO = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO / "artifacts"
INDEX = ARTIFACTS / "README.md"

#: Stable identifiers for research questions/checkpoints. CURRENT is unique
#: per question, not across the index: two rows answering different questions
#: may both be CURRENT at once.
BASELINE_QUESTION = "btc_ohlcv14_mtst_baseline"
P2A_QUESTION = "btc_p2a_model_family_benchmark"
P2B_QUESTION = "btc_p2b_information_set_benchmark"

#: Generations produced after the reporting-integrity metric-semantics fix,
#: whose `sharpe`/`max_drawdown` are directly comparable across runs.
POST_CORRECTION_GENERATIONS = {"v4", "P2a", "P2b"}

#: The current generation's aggregate for the baseline research question.
CURRENT_BASELINE = "diagnostics/btc_regimes_v4"

#: Its own committed source runs, each independently marked CURRENT.
CURRENT_SOURCE_RUNS = {
    f"walkforward/btc_nested_v4_seed_{seed}" for seed in (42, 142, 242, 342, 442)
}

#: The current generation's aggregate for the P2a model-family question.
CURRENT_P2A = "benchmark/btc_p2a_comparison"

#: Its own committed source runs, each independently marked CURRENT.
CURRENT_P2A_SOURCE_RUNS = {
    f"benchmark/btc_p2a_seed_{seed}" for seed in (42, 142, 242, 342, 442)
}

#: The current generation's aggregate for the P2b information-set question, and
#: its nine cells. `btc_p2b_regimes` is CURRENT for the same question but is
#: descriptive rather than a result, so it is listed beside them.
CURRENT_P2B = "benchmark/btc_p2b_comparison"
CURRENT_P2B_SOURCE_RUNS = {
    f"benchmark/btc_p2b_{arm}_{model}"
    for arm in ("ohlcv14", "smc_v1", "ohlcv14_plus_smc_v1")
    for model in ("logistic_regression", "lightgbm", "xgboost")
} | {"benchmark/btc_p2b_regimes"}

#: The prior generation's aggregate, whose own source runs are absent.
V3_BASELINE = "diagnostics/btc_regimes_v3"


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
    contract = load_contract()
    markdown = walkforward.to_markdown(
        results,
        summary,
        sealed,
        anchor=contract.sealed_test_start.isoformat(),
        contract=contract,
        fingerprint=ResearchInputFingerprint(
            research_input_hash="a" * 64,
            full_table_hash="b" * 64,
            research_rows=1000,
            total_rows=1200,
            research_start="2023-01-01T00:00:00+00:00",
            research_end="2023-06-01T00:00:00+00:00",
            columns=("date", "close", "target"),
        ),
    )

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


def _current_generation_by_question(rows: list[dict[str, str]]) -> dict[str, str]:
    """The single CURRENT generation for each research question in `rows`.

    CURRENT is scoped per research question, not globally unique across the
    index: two different questions may each carry their own CURRENT
    generation at once. A single question may still only ever have one.
    """
    by_question: dict[str, set[str]] = {}
    for row in rows:
        if row["status"] != "CURRENT":
            continue
        by_question.setdefault(row["research question"], set()).add(row["generation"])
    for question, generations in by_question.items():
        assert (
            len(generations) == 1
        ), f"{question!r} has more than one CURRENT generation: {sorted(generations)}"
    return {question: next(iter(gens)) for question, gens in by_question.items()}


_SOURCE_RUN_SHORTHAND = re.compile(r"([\w/]+_seed_)\{([\d,]+)\}")


def _expected_source_run_paths(cell: str) -> set[str] | None:
    """Expand a `source runs` cell's `prefix_seed_{a,b,c}` shorthand.

    Returns `None` for a cell with nothing to expand (e.g. a source run's own
    `itself (seed 42)` cell).
    """
    match = _SOURCE_RUN_SHORTHAND.search(cell)
    if match is None:
        return None
    prefix, numbers = match.groups()
    return {f"{prefix}{n}" for n in numbers.split(",")}


def _validate_current_aggregates(rows: list[dict[str, str]]) -> None:
    """A CURRENT aggregate's committed source runs must exist, be CURRENT
    themselves, and answer the same research question as the aggregate they
    back.
    """
    by_path = {row["path"]: row for row in rows}
    for row in rows:
        if row["status"] != "CURRENT" or row["source runs present"] != "yes":
            continue
        expected = _expected_source_run_paths(row["source runs"])
        if expected is None:
            continue
        for source_path in expected:
            source_row = by_path.get(source_path)
            assert (
                source_row is not None
            ), f"{row['path']} names source run {source_path}, which is not in the index"
            assert source_row["status"] == "CURRENT", (
                f"{row['path']} is CURRENT but its source run {source_path} is "
                f"{source_row['status']}"
            )
            assert source_row["research question"] == row["research question"], (
                f"{source_path} backs {row['path']} but answers a different research "
                "question"
            )


def test_the_index_names_exactly_one_current_generation_per_research_question():
    """CURRENT is scoped per research question, not globally unique across the
    index: the aggregate and its own committed source runs for a question
    share its CURRENT generation, matching each source run's own `STATUS.md`.
    Two different questions may each carry their own CURRENT generation at
    once; a single question may still only ever have one.
    """
    rows = _index_rows()
    current = [row for row in rows if row["status"] == "CURRENT"]
    assert current, "expected at least one CURRENT entry"

    assert _current_generation_by_question(rows) == {
        BASELINE_QUESTION: "v4",
        P2A_QUESTION: "P2a",
        P2B_QUESTION: "P2b",
    }

    def paths_for(question: str) -> set[str]:
        return {row["path"] for row in current if row["research question"] == question}

    assert paths_for(BASELINE_QUESTION) == {CURRENT_BASELINE, *CURRENT_SOURCE_RUNS}
    assert paths_for(P2A_QUESTION) == {CURRENT_P2A, *CURRENT_P2A_SOURCE_RUNS}
    assert paths_for(P2B_QUESTION) == {CURRENT_P2B, *CURRENT_P2B_SOURCE_RUNS}


def test_current_aggregates_are_backed_by_their_committed_source_runs():
    _validate_current_aggregates(_index_rows())


def _row(
    path: str,
    generation: str,
    question: str,
    status: str,
    source_runs_present: str = "n/a",
    source_runs: str = "itself",
) -> dict[str, str]:
    return {
        "path": path,
        "generation": generation,
        "research question": question,
        "status": status,
        "source runs present": source_runs_present,
        "source runs": source_runs,
    }


def test_two_competing_current_generations_for_the_same_question_is_rejected():
    """The failure this scoping exists to prevent: a new generation marked
    CURRENT for a question whose old generation was never demoted.
    """
    rows = [
        _row("diagnostics/btc_regimes_v4", "v4", BASELINE_QUESTION, "CURRENT", "yes"),
        _row("diagnostics/btc_regimes_v5", "v5", BASELINE_QUESTION, "CURRENT", "yes"),
    ]
    with pytest.raises(AssertionError):
        _current_generation_by_question(rows)


def test_two_unrelated_current_questions_are_allowed():
    """The behaviour the old, globally-unique-CURRENT test forbade."""
    rows = [
        _row("diagnostics/btc_regimes_v4", "v4", BASELINE_QUESTION, "CURRENT", "yes"),
        _row("benchmark/btc_p2a_comparison", "P2a", P2A_QUESTION, "CURRENT", "yes"),
    ]
    assert _current_generation_by_question(rows) == {
        BASELINE_QUESTION: "v4",
        P2A_QUESTION: "P2a",
    }


def test_a_superseded_generation_does_not_count_as_current():
    rows = [
        _row("diagnostics/btc_regimes_v4", "v4", BASELINE_QUESTION, "CURRENT", "yes"),
        _row("diagnostics/btc_regimes_v3", "v3", BASELINE_QUESTION, "SUPERSEDED", "no"),
    ]
    assert _current_generation_by_question(rows) == {BASELINE_QUESTION: "v4"}


def test_a_current_aggregate_missing_a_source_run_is_rejected():
    rows = [
        _row(
            "diagnostics/btc_regimes_v4",
            "v4",
            BASELINE_QUESTION,
            "CURRENT",
            "yes",
            "walkforward/btc_nested_v4_seed_{42,142}",
        ),
        _row("walkforward/btc_nested_v4_seed_42", "v4", BASELINE_QUESTION, "CURRENT"),
        # seed 142 is missing from the index entirely.
    ]
    with pytest.raises(AssertionError):
        _validate_current_aggregates(rows)


def test_a_source_run_status_disagreeing_with_its_aggregate_is_rejected():
    rows = [
        _row(
            "diagnostics/btc_regimes_v4",
            "v4",
            BASELINE_QUESTION,
            "CURRENT",
            "yes",
            "walkforward/btc_nested_v4_seed_{42}",
        ),
        _row("walkforward/btc_nested_v4_seed_42", "v4", BASELINE_QUESTION, "HISTORICAL"),
    ]
    with pytest.raises(AssertionError):
        _validate_current_aggregates(rows)


def test_a_source_run_answering_a_different_question_is_rejected():
    rows = [
        _row(
            "benchmark/btc_p2a_comparison",
            "P2a",
            P2A_QUESTION,
            "CURRENT",
            "yes",
            "benchmark/btc_p2a_seed_{42}",
        ),
        _row("benchmark/btc_p2a_seed_42", "P2a", BASELINE_QUESTION, "CURRENT"),
    ]
    with pytest.raises(AssertionError):
        _validate_current_aggregates(rows)


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
    # Committed directories, not whatever happens to be on this filesystem: a
    # developer's untracked run directories are legitimate local state and are
    # not part of what the repository contains.
    committed = {str(Path(p).parent) for p in _tracked_paths() if "/" in p}
    assert committed, "no committed artifact directories found"
    assert (
        committed == listed
    ), f"unlisted: {sorted(committed - listed)}; stale: {sorted(listed - committed)}"


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
        if row["generation"] in POST_CORRECTION_GENERATIONS:
            continue
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
    assert rows[V3_BASELINE]["source runs present"] == "no"
    assert "btc_nested_v3_seed" in rows[V3_BASELINE]["source runs"]

    text = INDEX.read_text()
    assert "not fixed by copying or renaming" in text.lower()
    for missing in ("btc_nested_v3_seed_", "btc_nested_v2_seed_"):
        assert missing in text


def _tracked_paths() -> set[str]:
    """Every path git has under `artifacts/`, relative to it.

    The index makes a claim about what this *repository* contains, so the guard
    below has to ask git, not the filesystem.
    """
    result = subprocess.run(
        ["git", "ls-files", "-z", "artifacts/"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    return {
        str(Path(entry).relative_to("artifacts"))
        for entry in result.stdout.split("\0")
        if entry
    }


def test_no_v2_or_v3_source_run_is_committed():
    """The index says those runs are absent *from the repository*.

    Deliberately a git question, not a filesystem one. On the machine that
    produced them the genuine `btc_nested_v3_seed_*` directories still exist,
    untracked and gitignored — that is the normal state of a developer checkout
    and must not fail the suite. What must never happen is one of them being
    *committed*, because the only way that can occur without the real run data
    is by copying or relabelling a different generation, which would manufacture
    provenance the index says does not exist.
    """
    committed = _tracked_paths()
    for seed in (42, 142, 242, 342, 442):
        for generation in ("v2", "v3"):
            name = f"walkforward/btc_nested_{generation}_seed_{seed}"
            offenders = sorted(p for p in committed if p.startswith(name + "/"))
            assert not offenders, (
                f"{name} is committed ({offenders}), but artifacts/README.md records it "
                "as absent from this repository. If the real run data has been added, "
                "update the index; if these are copies of another generation, remove "
                "them — a relabelled run is fabricated provenance."
            )


def test_a_genuine_untracked_source_run_does_not_fail_the_suite(tmp_path, monkeypatch):
    """Regression for the guard above: local ignored artifacts are legitimate.

    The first version of this check called `Path.exists()`, which would have
    failed on the very machine that generated the v3 evidence — the runs are on
    disk there, correctly gitignored. Creating one here proves the guard reads
    committed state instead.
    """
    genuine = ARTIFACTS / "walkforward" / "btc_nested_v3_seed_42"
    created = not genuine.exists()
    if created:
        genuine.mkdir(parents=True)
        (genuine / "walkforward.json").write_text("{}")
    try:
        assert genuine.exists(), "the fixture must actually be on disk"
        # Untracked, so git does not see it and the guard passes.
        assert not any(
            p.startswith("walkforward/btc_nested_v3_seed_42/") for p in _tracked_paths()
        )
        test_no_v2_or_v3_source_run_is_committed()
        # And the index still describes the repository correctly.
        test_the_index_states_that_the_v3_source_runs_are_absent()
    finally:
        if created:
            (genuine / "walkforward.json").unlink()
            genuine.rmdir()


def test_the_index_lists_only_committed_artifact_directories():
    """Every indexed directory is one git actually carries."""
    committed_dirs = {str(Path(p).parent) for p in _tracked_paths() if "/" in p}
    for row in _index_rows():
        assert (
            row["path"] in committed_dirs
        ), f"{row['path']} is in the index but has no committed files"
