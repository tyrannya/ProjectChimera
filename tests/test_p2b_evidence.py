"""The committed P2b and P2c evidence, checked rather than described.

Three kinds of claim live here, and they are checked three different ways
because they are three different kinds of thing.

**Primary evidence is pinned by its bytes.** A cell's metrics and its per-sample
predictions cannot be rebuilt without re-fitting, so a byte change in one is a
change in the research result. Every checksum manifest in the repository is
re-hashed here, with no exemptions of any kind, through the same
:func:`tools.freeze_evidence.check` that ``--verify`` calls — so "frozen" has
exactly one meaning whether a person or CI is asking.

**Derived evidence is pinned by what it says.** A comparison is regenerated from
the cells whenever the aggregator improves, and it is supposed to: widening a
recomputation from ten trading keys to twenty-three rewrites the file without
touching a number the research reported. Hashing it alongside primary evidence
produced three manifests that failed their own verification and a test that
excused the failures by matching ``_comparison/`` in the path. So the fold
counts, the verdicts and the integrity counters are asserted directly instead,
and a regenerated report that changed a finding fails while one that only
improved its own prose does not.

**The freeze tool itself is pinned by adversarial use.** A verifier that only
ever sees good input proves nothing — a function returning zero problems would
pass. Every rejection it claims is exercised against a sandboxed tree.

Nothing here needs a dataset, a fit or a network. It all reads committed files.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from nn.information_sets import CHECKPOINTS
from tools import freeze_evidence
from tools.freeze_evidence import DERIVED, EVIDENCE_CLASS_KEY, PRIMARY

ROOT = Path(__file__).resolve().parent.parent
BENCHMARK = ROOT / "artifacts" / "benchmark"

#: Every checksum manifest this repository stands behind, discovered rather than
#: listed. A checkpoint that freezes new evidence is covered the moment its
#: manifest lands, instead of when someone remembers to add a name here.
MANIFESTS = tuple(sorted(path.name for path in (ROOT / "artifacts").glob("*_SHA256SUMS.txt")))

#: Manifests written before a manifest covered primary evidence only, and the
#: primary-only manifest that replaced each. They are kept byte for byte and
#: renamed rather than edited: a manifest is the repository's own statement
#: about what a past run produced, so the answer to one that covered the wrong
#: kind of file is a successor, never a rewrite — `tools.freeze_evidence`
#: refuses to overwrite one for the same reason. The `.superseded.txt` suffix is
#: what keeps a manifest nobody should expect to verify out of `MANIFESTS`,
#: because a checksum file that fails on purpose teaches a reader to shrug at
#: one that fails for real. `None` means there was nothing to succeed —
#: `btc_p2b_recheck` covered a comparison and nothing else.
SUPERSEDED = {
    "btc_p2b_SHA256SUMS.superseded.txt": "btc_p2b_cells_SHA256SUMS.txt",
    "btc_p2b_ablation_SHA256SUMS.superseded.txt": ("btc_p2b_ablation_cells_SHA256SUMS.txt"),
    "btc_p2b_recheck_SHA256SUMS.superseded.txt": None,
    "btc_p2c_SHA256SUMS.superseded.txt": "btc_p2c_cells_SHA256SUMS.txt",
}

#: The two comparison directories this branch regenerates. Derived by
#: declaration, and asserted to be so below rather than assumed.
DERIVED_DIRS = (
    BENCHMARK / "btc_p2b_comparison",
    BENCHMARK / "btc_p2c_comparison",
    BENCHMARK / "btc_p2b_ablation_xgboost",
    BENCHMARK / "btc_p2b_regimes",
)


# --------------------------------------------------------------------------- #
# A. primary evidence is pinned by its bytes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("manifest", MANIFESTS)
def test_every_frozen_file_still_hashes_to_its_manifest(manifest):
    """No exemptions. Not by directory, not by suffix, not by evidence class.

    The previous version of this test allowed any covered path containing
    ``_comparison/`` to fail its hash, which made three manifests permanently
    and deliberately wrong. A checksum that means "frozen unless it is not"
    trains a reader to ignore the one signal it exists to send.
    """
    path = ROOT / "artifacts" / manifest
    assert path.is_file(), f"{manifest} is missing"
    entries = freeze_evidence.manifest_entries(path)
    assert entries, f"{manifest} is empty"
    assert freeze_evidence.check(path) == []


@pytest.mark.parametrize("manifest", MANIFESTS)
def test_no_manifest_covers_derived_evidence(manifest):
    """The rule that keeps the test above from ever needing an exemption again."""
    for _, name in freeze_evidence.manifest_entries(ROOT / "artifacts" / manifest):
        directory = (ROOT / name).parent
        assert freeze_evidence.evidence_class_of(directory) != DERIVED, (
            f"{manifest} freezes {name}, which its own directory declares derived. "
            "A derived file is regenerated by design; hashing it guarantees a "
            "manifest that fails for a reason nobody should act on."
        )


def test_the_regenerated_reports_declare_themselves_derived():
    """The other half: the exclusion must be a declaration, not a path pattern."""
    for directory in DERIVED_DIRS:
        assert directory.is_dir(), f"{directory} is missing"
        assert freeze_evidence.evidence_class_of(directory) == DERIVED


def test_every_frozen_cell_is_primary_to_the_tool_that_froze_it():
    """Nothing a manifest covers would be excluded if it were frozen again.

    The committed cells declare no evidence class: they were written before the
    field existed, and `evidence_class_of` returns `None` for them. That is the
    intended default rather than a gap — a new aggregator has to opt *into*
    being excluded, so an undeclared directory is frozen as primary and only an
    explicit `derived` is refused. What must never happen is a covered
    directory that would now be refused, which is what this asserts.
    """
    covered = {
        (ROOT / name).parent
        for manifest in MANIFESTS
        for _, name in freeze_evidence.manifest_entries(ROOT / "artifacts" / manifest)
    }
    assert covered, "the manifests cover nothing"
    for directory in covered:
        assert freeze_evidence.evidence_class_of(directory) in (None, PRIMARY)


# --------------------------------------------------------------------------- #
# B. the freeze tool, used adversarially
#
# `ROOT` is redirected at a sandbox so these can mutate and delete files. The
# committed evidence is never written to.
# --------------------------------------------------------------------------- #
@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """A repository-shaped tree with one primary cell and one derived report."""
    monkeypatch.setattr(freeze_evidence, "ROOT", tmp_path)
    cell = tmp_path / "artifacts" / "benchmark" / "demo_cell"
    cell.mkdir(parents=True)
    (cell / "p2b.json").write_text(json.dumps({EVIDENCE_CLASS_KEY: PRIMARY, "fold": 0}))
    (cell / "outer_predictions.parquet").write_bytes(b"not really a parquet")
    (cell / "STATUS.md").write_text("# CURRENT\n")

    report = tmp_path / "artifacts" / "benchmark" / "demo_comparison"
    report.mkdir(parents=True)
    (report / "p2b_comparison.json").write_text(json.dumps({EVIDENCE_CLASS_KEY: DERIVED}))
    (report / "p2b_comparison.md").write_text("# derived\n")
    return tmp_path


def _freeze(sandbox, *directories) -> Path:
    out = sandbox / "artifacts" / "DEMO_SHA256SUMS.txt"
    assert freeze_evidence.freeze([Path(d) for d in directories], out) == 0
    return out


def test_freezing_a_primary_cell_covers_every_evidence_file_in_it(sandbox):
    manifest = _freeze(sandbox, sandbox / "artifacts" / "benchmark" / "demo_cell")
    covered = {name for _, name in freeze_evidence.manifest_entries(manifest)}
    assert covered == {
        "artifacts/benchmark/demo_cell/p2b.json",
        "artifacts/benchmark/demo_cell/outer_predictions.parquet",
        "artifacts/benchmark/demo_cell/STATUS.md",
    }
    assert freeze_evidence.check(manifest) == []


def test_freezing_refuses_a_directory_that_declares_itself_derived(sandbox):
    with pytest.raises(SystemExit, match="refusing to freeze derived evidence"):
        _freeze(sandbox, sandbox / "artifacts" / "benchmark" / "demo_comparison")


def test_freezing_a_glob_that_caught_a_report_refuses_rather_than_dropping_it(sandbox):
    """The realistic mistake: `--out ... artifacts/benchmark/btc_p2b_*`.

    Silently freezing the cells and skipping the report would produce a manifest
    that looks complete and is not, which is worse than a refusal naming the
    directory.
    """
    with pytest.raises(SystemExit, match="demo_comparison"):
        _freeze(
            sandbox,
            sandbox / "artifacts" / "benchmark" / "demo_cell",
            sandbox / "artifacts" / "benchmark" / "demo_comparison",
        )


def test_freezing_refuses_to_overwrite_an_existing_manifest(sandbox):
    _freeze(sandbox, sandbox / "artifacts" / "benchmark" / "demo_cell")
    with pytest.raises(SystemExit, match="already exists"):
        _freeze(sandbox, sandbox / "artifacts" / "benchmark" / "demo_cell")


def test_a_mutated_covered_file_fails_verification(sandbox):
    manifest = _freeze(sandbox, sandbox / "artifacts" / "benchmark" / "demo_cell")
    target = sandbox / "artifacts" / "benchmark" / "demo_cell" / "outer_predictions.parquet"
    target.write_bytes(b"not really a parquet either")
    problems = freeze_evidence.check(manifest)
    assert len(problems) == 1 and problems[0].startswith("CHANGED")
    assert freeze_evidence.verify(manifest) == 1


def test_a_removed_covered_file_fails_verification(sandbox):
    manifest = _freeze(sandbox, sandbox / "artifacts" / "benchmark" / "demo_cell")
    (sandbox / "artifacts" / "benchmark" / "demo_cell" / "STATUS.md").unlink()
    problems = freeze_evidence.check(manifest)
    assert len(problems) == 1 and problems[0].startswith("MISSING")
    assert freeze_evidence.verify(manifest) == 1


def test_a_changed_file_is_not_excused_by_living_in_a_comparison_directory(sandbox):
    """The exact exemption that used to exist, aimed at the exact path shape.

    A directory named `..._comparison` whose artifact declares itself *primary*
    is primary. The name is not the classifier and must not become one again.
    """
    directory = sandbox / "artifacts" / "benchmark" / "misnamed_comparison"
    directory.mkdir(parents=True)
    (directory / "p2b_comparison.json").write_text(json.dumps({EVIDENCE_CLASS_KEY: PRIMARY}))
    manifest = _freeze(sandbox, directory)
    (directory / "p2b_comparison.json").write_text(
        json.dumps({EVIDENCE_CLASS_KEY: PRIMARY, "x": 1})
    )
    problems = freeze_evidence.check(manifest)
    assert len(problems) == 1
    assert "_comparison/" in problems[0]


def test_the_cli_exit_code_and_the_test_suites_check_are_the_same_decision(sandbox):
    manifest = _freeze(sandbox, sandbox / "artifacts" / "benchmark" / "demo_cell")
    assert freeze_evidence.check(manifest) == []
    assert freeze_evidence.verify(manifest) == 0
    (sandbox / "artifacts" / "benchmark" / "demo_cell" / "STATUS.md").write_text("# edited\n")
    assert freeze_evidence.check(manifest) != []
    assert freeze_evidence.verify(manifest) == 1


# --------------------------------------------------------------------------- #
# C. the findings themselves
# --------------------------------------------------------------------------- #
def _comparison(name: str) -> dict:
    return json.loads((BENCHMARK / name / "p2b_comparison.json").read_text())


def test_each_checkpoints_artifacts_identify_as_that_checkpoint():
    """The provenance failure this branch was opened to fix.

    Nine P2c cells and their comparison identified as P2b and stated P2b's
    market-structure question, over chart-structure columns. Every number was
    P2c's; only the identity was wrong, which is why nothing failed.
    """
    for checkpoint, prefix, arms in (
        ("P2b", "btc_p2b", ("ohlcv14", "smc_v1", "ohlcv14_plus_smc_v1")),
        (
            "P2c",
            "btc_p2c",
            ("ohlcv14", "chart_structure_v1", "ohlcv14_plus_chart_structure_v1"),
        ),
    ):
        question = CHECKPOINTS[checkpoint].question
        comparison = _comparison(f"{prefix}_comparison")
        assert comparison["checkpoint"] == checkpoint
        assert comparison["question"] == question
        assert comparison["evidence_class"] == DERIVED
        assert (
            (BENCHMARK / f"{prefix}_comparison" / "p2b_comparison.md")
            .read_text()
            .startswith(f"# {checkpoint} —")
        )
        # And says where that identity came from, because a reader who opens
        # the nine cells behind it will find "P2b" written in every one.
        assert "recovered from the arms" in comparison["checkpoint_identity"]
        for arm in arms:
            for model in ("logistic_regression", "lightgbm", "xgboost"):
                cell = json.loads(
                    (BENCHMARK / f"{prefix}_{arm}_{model}" / "p2b.json").read_text()
                )
                # The arm is the trustworthy field, and `load_cell` checks it
                # against the columns the cell actually flattened.
                assert cell["information_set"] == arm
                # The label is not. Every committed cell predates the field in
                # which a cell states its own question, so all twenty-five say
                # "P2b" — including the nine that ran chart structure. They are
                # deliberately not relabelled: editing a frozen artifact to say
                # something it did not say manufactures provenance, and
                # re-running them to correct a string would replace the primary
                # evidence the manifests exist to pin. The report above is
                # titled correctly because `nn.p2b_compare` reads the arms
                # instead. If someone ever does correct these cells, this fails
                # and the change gets discussed rather than absorbed.
                assert cell["checkpoint"] == "P2b", f"{prefix}_{arm}_{model}"
                assert "question" not in cell


def test_the_p2b_question_and_the_p2c_question_are_not_the_same_question():
    assert (
        _comparison("btc_p2b_comparison")["question"]
        != _comparison("btc_p2c_comparison")["question"]
    )
    assert "smc_v1" in _comparison("btc_p2b_comparison")["question"]
    assert "chart_structure_v1" in _comparison("btc_p2c_comparison")["question"]


@pytest.mark.parametrize("name", ["btc_p2b_comparison", "btc_p2c_comparison"])
def test_every_persisted_row_is_a_planned_row(name):
    """The counters that say the scored sample is the planned sample.

    `snapshot_anchoring` asks whether each persisted row agrees with the
    snapshot at the index it claims. This asks the other question — whether
    those indices are the ones the fold plan selected — and a scorer that
    persisted a consistent but wrong selection passes the first and fails this.
    """
    bound = _comparison(name)["planned_row_alignment"]
    assert bound["cells_checked"] == 9
    assert bound["folds_checked"] == 36
    assert bound["rows_checked"] == 170_451
    assert bound["problems"] == 0
    for counter in (
        "missing_folds",
        "unplanned_folds",
        "non_integer_row_index",
        "duplicate_rows",
        "unsorted_rows",
        "count_mismatches",
        "sample_index_hash_mismatches",
        "first_last_mismatches",
        "cross_fold_rows",
        "snapshot_value_mismatches",
    ):
        assert bound[counter] == 0, counter


def test_both_information_set_checkpoints_reproduce_the_same_frozen_control():
    """P2b and P2c each re-ran the OHLCV14 control, and it is P2a's.

    Both must equal P2a's frozen seed-42 XGBoost evidence, and therefore each
    other. If adding a feature family ever perturbs the control's sample
    universe, this is what notices.
    """
    p2a = json.loads((BENCHMARK / "btc_p2a_seed_42" / "benchmark.json").read_text())
    frozen = [f["outer_validation"]["xgboost"] for f in p2a["folds"]]
    for checkpoint in ("btc_p2b_ohlcv14_xgboost", "btc_p2c_ohlcv14_xgboost"):
        cell = json.loads((BENCHMARK / checkpoint / "p2b.json").read_text())
        assert [f["outer_validation"]["xgboost"] for f in cell["folds"]] == frozen


def test_the_p2b_comparison_reports_a_negative_result():
    """The frozen verdicts, pinned so a rerun cannot quietly improve them.

    Not a test of the machinery — a test of the *finding*. P2b answered no, and
    the six fold counts below are what that means. If a later change to the
    comparison moves any of them, the result changed, and that has to be a
    deliberate act with its own evidence rather than a diff nobody read.
    """
    payload = _comparison("btc_p2b_comparison")
    assert payload["sealed_test"] is False
    assert payload["independent_recompute"]["mismatches"] == 0
    assert payload["snapshot_anchoring"]["problems"] == 0

    expected = {
        ("logistic_regression", "smc_v1"): 0,
        ("logistic_regression", "ohlcv14_plus_smc_v1"): 2,
        ("lightgbm", "smc_v1"): 2,
        ("lightgbm", "ohlcv14_plus_smc_v1"): 1,
        ("xgboost", "smc_v1"): 1,
        ("xgboost", "ohlcv14_plus_smc_v1"): 1,
    }
    for (model, arm), folds in expected.items():
        assert payload["deltas"][model][arm]["net_return_improved_folds"] == folds

    # The predeclared bar is three of four. Nothing reached it, which is the
    # finding; asserting it here stops "P2b was positive" from ever being true
    # of this directory without someone changing this line.
    assert all(v < 3 for v in expected.values())


def test_the_p2c_comparison_reports_a_negative_result():
    """P2c's frozen verdicts. Same reasoning as the P2b pin above.

    Every mean delta here is negative, which P2b's was not — there is no arm in
    P2c where pooling the four periods would even have flattered the result.
    """
    payload = _comparison("btc_p2c_comparison")
    assert payload["sealed_test"] is False
    assert payload["independent_recompute"]["mismatches"] == 0
    assert payload["snapshot_anchoring"]["problems"] == 0

    expected = {
        ("logistic_regression", "chart_structure_v1"): 1,
        ("logistic_regression", "ohlcv14_plus_chart_structure_v1"): 0,
        ("lightgbm", "chart_structure_v1"): 1,
        ("lightgbm", "ohlcv14_plus_chart_structure_v1"): 2,
        ("xgboost", "chart_structure_v1"): 1,
        ("xgboost", "ohlcv14_plus_chart_structure_v1"): 1,
    }
    for (model, arm), folds in expected.items():
        entry = payload["deltas"][model][arm]
        assert entry["net_return_improved_folds"] == folds
        assert entry["aggregate"]["net_return"]["mean"] < 0
    assert all(v < 3 for v in expected.values())


def test_the_committed_predictions_are_the_bytes_the_manifests_pin():
    """A cell's parquet is primary evidence and is covered, not merely present."""
    covered = {
        name
        for manifest in (
            "btc_p2b_cells_SHA256SUMS.txt",
            "btc_p2c_cells_SHA256SUMS.txt",
        )
        for _, name in freeze_evidence.manifest_entries(ROOT / "artifacts" / manifest)
    }
    for prefix, arms in (
        ("btc_p2b", ("ohlcv14", "smc_v1", "ohlcv14_plus_smc_v1")),
        (
            "btc_p2c",
            ("ohlcv14", "chart_structure_v1", "ohlcv14_plus_chart_structure_v1"),
        ),
    ):
        for arm in arms:
            for model in ("logistic_regression", "lightgbm", "xgboost"):
                relative = (
                    f"artifacts/benchmark/{prefix}_{arm}_{model}/outer_predictions.parquet"
                )
                assert relative in covered, relative
                assert (ROOT / relative).is_file()


def test_a_manifest_entry_is_a_sha256_of_the_file_it_names():
    """Independent of `check`: re-derive one digest by hand.

    `check` could be wrong in the same direction as `freeze` and both tests
    above would still pass. This one hashes a file directly.
    """
    manifest = ROOT / "artifacts" / "btc_p2b_cells_SHA256SUMS.txt"
    expected, name = freeze_evidence.manifest_entries(manifest)[0]
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected


# --------------------------------------------------------------------------- #
# D. the documentation says what the history supports
#
# The last item on the adversarial list, and the only one a checksum cannot
# reach: a specification claiming more independence than git gives it. These
# assert wording, which is unusual and deliberate — `chart_structure_v1.md`
# opened with `smc_v1`'s checkpoint and `smc_v1`'s question for the length of
# two checkpoints, and no test in this repository could have noticed.
# --------------------------------------------------------------------------- #
DOCS = ROOT / "docs"


@pytest.mark.parametrize(
    "document, checkpoint, family",
    [
        ("smc_v1.md", "P2b", "smc_v1"),
        ("chart_structure_v1.md", "P2c", "chart_structure_v1"),
    ],
)
def test_each_feature_spec_names_its_own_checkpoint(document, checkpoint, family):
    """Both specs use the same header line, and it has to name the right one.

    `chart_structure_v1.md` carried `Research checkpoint: **P2b**` — `smc_v1`'s
    checkpoint, with `smc_v1`'s question — through two checkpoints.
    """
    text = (DOCS / document).read_text()
    other = "P2c" if checkpoint == "P2b" else "P2b"
    header, _, rest = text.partition("Research checkpoint:")
    assert rest, f"{document} no longer states a research checkpoint"
    assert rest.lstrip().startswith(f"**{checkpoint}**"), rest[:80]
    assert f"Research checkpoint: **{other}**" not in text
    # The question beside the header must be about this document's own family.
    assert family in rest[:400], rest[:400]


def test_the_chart_spec_states_its_adaptive_status_rather_than_implying_none():
    chart = (DOCS / "chart_structure_v1.md").read_text()
    assert "Status: adaptive research evidence" in chart
    assert "Not a pristine out-of-sample confirmation" in chart
    # The claim it is allowed to make, and the one it is not.
    assert "before any P2b outer-validation number existed" in chart
    assert "predeclared before any P2b outer-validation result was observed" not in chart


def test_no_document_calls_a_frozen_hash_stale_by_design():
    """The phrase that meant "this checksum is allowed to be wrong"."""
    for path in [ROOT / "artifacts" / "README.md", *DOCS.glob("*.md"), ROOT / "README.md"]:
        assert "stale by design" not in path.read_text(), path


def test_no_document_claims_ohlcv_alpha_has_been_disproved():
    """The overstatement two negative checkpoints invite.

    The evidence is against spending the next checkpoint on another
    hand-designed transformation of the same hourly bars. It is not a proof
    about a space of possible features.
    """
    forbidden = (
        "OHLCV information is proven exhausted",
        "there is no OHLCV alpha",
        "five hourly values contain no predictive information",
        "proven exhausted",
    )
    for path in [ROOT / "README.md", ROOT / "artifacts" / "README.md", *DOCS.glob("*.md")]:
        text = path.read_text()
        for phrase in forbidden:
            assert phrase not in text, f"{path.name}: {phrase!r}"


def test_the_repository_has_the_manifests_these_tests_think_it_has():
    """Discovery is convenient and silent; a deleted manifest would just vanish.

    Globbing means a new checkpoint is covered automatically. It also means a
    manifest that disappeared would take its own verification with it and no
    test above would fail, so the set is named once, here.
    """
    assert MANIFESTS == (
        "btc_p2a_SHA256SUMS.txt",
        "btc_p2b_ablation_cells_SHA256SUMS.txt",
        "btc_p2b_cells_SHA256SUMS.txt",
        "btc_p2c_cells_SHA256SUMS.txt",
        "btc_v4_SHA256SUMS.txt",
    )
    for retired in SUPERSEDED:
        assert (ROOT / "artifacts" / retired).is_file(), (
            f"{retired} was retired, not deleted; it is the record of what its "
            "checkpoint froze and the repository keeps it"
        )


@pytest.mark.parametrize("retired,successor", sorted(SUPERSEDED.items()))
def test_a_reissued_manifest_changed_no_digest_it_inherited(retired, successor):
    """Re-issuing a manifest dropped derived entries and did nothing else.

    This is what makes the supersession safe to trust. Narrowing a manifest is
    exactly the shape of laundering a result — drop the row that stopped
    matching, keep the file, call it frozen — so a successor is required to be a
    strict projection of what it replaced: every path they share carries the
    same digest, nothing was added, and every entry that went away lives in a
    directory that declares itself derived.

    A successor that re-froze a moved primary file fails here, and so does one
    that quietly dropped a cell.
    """
    old = {
        name: digest
        for digest, name in freeze_evidence.manifest_entries(ROOT / "artifacts" / retired)
    }
    new = (
        {
            name: digest
            for digest, name in freeze_evidence.manifest_entries(
                ROOT / "artifacts" / successor
            )
        }
        if successor
        else {}
    )
    assert old, f"{retired} is empty"

    rehashed = [name for name in old.keys() & new.keys() if old[name] != new[name]]
    assert not rehashed, (
        f"{successor} records a different digest than {retired} for {rehashed}. A "
        "successor may narrow what a manifest covers; it may not restate what a "
        "covered file hashed to."
    )
    assert not new.keys() - old.keys(), (
        f"{successor} covers files {retired} did not: "
        f"{sorted(new.keys() - old.keys())}. New evidence gets its own manifest "
        "rather than being folded into a re-issue."
    )
    for name in sorted(old.keys() - new.keys()):
        directory = (ROOT / name).parent
        assert freeze_evidence.evidence_class_of(directory) == DERIVED, (
            f"{retired} covered {name} and {successor} does not, but "
            f"{directory.name} does not declare itself {DERIVED}. Only derived "
            "evidence may be dropped from a re-issued manifest."
        )
