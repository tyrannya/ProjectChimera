"""What a research cell means when it says which source produced it.

The audited P3 cells recorded 1,805, 1,811 and 1,824 "source" files against a
repository with 92 tracked Python files, and their digests split three ways *by
model family* while every line of ProjectChimera source was identical. The cause
was an in-repository ``.venv``: the digest covered every module the process had
imported whose file resolved inside the repository directory, and site-packages
resolves inside the repository directory. The comparator was then relaxed to let
those cells join.

These tests are the adversarial cases that failure suggests, each against a real
git checkout built in a temporary directory rather than against a mock.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from nn.p2b_compare import ComparisonError, check_cells_agree
from nn.source_identity import (
    SOURCE_IDENTITY_SCHEME,
    SOURCE_ROOTS,
    SourceIdentityError,
    source_identity,
)


def git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    """A minimal git checkout shaped like this repository."""
    root = tmp_path / "repo"
    for name in SOURCE_ROOTS:
        (root / name).mkdir(parents=True)
        (root / name / "__init__.py").write_text(f'"""{name}"""\n')
    (root / "nn" / "engine.py").write_text("VALUE = 1\n")
    (root / "tests").mkdir()
    (root / "tests" / "test_engine.py").write_text("def test_engine():\n    assert True\n")
    (root / "README.md").write_text("# repo\n")
    (root / ".gitignore").write_text(".venv/\n")

    git(root.parent, "init", "-q", str(root))
    git(root, "config", "user.email", "test@example.invalid")
    git(root, "config", "user.name", "test")
    git(root, "add", "-A")
    git(root, "commit", "-qm", "initial")
    return root


def digest(root: Path) -> str:
    return source_identity(root)["source_digest"]


# --- what must not move it ---------------------------------------------------
def test_an_in_repository_virtualenv_cannot_enter_the_digest(checkout):
    """The failure this replaces, reproduced in its real shape.

    A ``.venv`` inside the checkout holding thousands of Python files, exactly
    where the P3 cells' 1,800 "source" files came from.
    """
    before = digest(checkout)
    site = checkout / ".venv" / "lib" / "python3.11" / "site-packages" / "sklearn"
    site.mkdir(parents=True)
    (site / "__init__.py").write_text("__version__ = '1.9.0'\n")
    (site / "linear_model.py").write_text("class LogisticRegression: pass\n")
    assert digest(checkout) == before


def test_a_site_packages_directory_under_a_source_root_cannot_enter_it_either(checkout):
    """Belt and braces: the structural exclusion, not only the path depth."""
    before = digest(checkout)
    vendored = checkout / "tools" / "site-packages" / "lightgbm"
    vendored.mkdir(parents=True)
    (vendored / "__init__.py").write_text("__version__ = '4.7.0'\n")
    git(checkout, "add", "-A")
    assert digest(checkout) == before


def test_importing_another_model_library_cannot_move_it(checkout):
    """The model-family split, stated as the property that now holds.

    The digest is a function of files on disk under the source roots and of
    nothing about the running process, so three cells that imported three
    different estimator libraries record one digest by construction.
    """
    import json  # noqa: F401  - a module this process had not imported before

    before = digest(checkout)
    import statistics  # noqa: F401
    import wave  # noqa: F401

    assert digest(checkout) == before


def test_a_test_only_change_does_not_move_it(checkout):
    """`tests/` is outside the roots, so a batch is not split by a new test."""
    before = digest(checkout)
    (checkout / "tests" / "test_engine.py").write_text("def test_engine():\n    assert 1\n")
    git(checkout, "add", "-A")
    git(checkout, "commit", "-qm", "a test")
    assert digest(checkout) == before


def test_a_documentation_commit_moves_the_revision_and_not_the_digest(checkout):
    before = source_identity(checkout)
    (checkout / "README.md").write_text("# repo\n\nmore words\n")
    git(checkout, "add", "-A")
    git(checkout, "commit", "-qm", "docs")
    after = source_identity(checkout)

    assert after["revision"] != before["revision"]
    assert after["source_digest"] == before["source_digest"]


# --- what must move it -------------------------------------------------------
def test_dirty_tracked_source_moves_it(checkout):
    before = digest(checkout)
    (checkout / "nn" / "engine.py").write_text("VALUE = 2\n")
    after = source_identity(checkout)
    assert after["source_digest"] != before
    assert after["dirty"] is True


def test_an_untracked_python_file_under_a_source_root_moves_it(checkout):
    """A module research can import is source whether or not git knows it."""
    before = digest(checkout)
    (checkout / "nn" / "scratch.py").write_text("HACK = True\n")
    after = source_identity(checkout)
    assert after["source_digest"] != before
    assert after["untracked_source_files"] == 1


def test_deleting_a_tracked_module_moves_it(checkout):
    before = digest(checkout)
    (checkout / "nn" / "engine.py").unlink()
    after = source_identity(checkout)
    assert after["source_digest"] != before
    assert after["missing_tracked_source_files"] == 1


def test_a_committed_source_change_moves_it(checkout):
    before = digest(checkout)
    (checkout / "chimera" / "features.py").write_text("def compute():\n    return 3\n")
    git(checkout, "add", "-A")
    git(checkout, "commit", "-qm", "a feature")
    after = source_identity(checkout)
    assert after["source_digest"] != before
    assert after["dirty"] is False


# --- what must refuse to answer ---------------------------------------------
def test_a_gitignored_python_file_under_a_source_root_is_refused(checkout):
    """The hole this digest exists to close, in its remaining shape.

    A module git is told to ignore is still importable, so a digest that
    silently skipped it would be back to describing something other than what
    ran. There is no legitimate ignored ``.py`` under a package root, so the
    right answer is a refusal rather than a wider rule.
    """
    (checkout / ".gitignore").write_text(".venv/\nnn/private.py\n")
    (checkout / "nn" / "private.py").write_text("SECRET = 1\n")
    with pytest.raises(SourceIdentityError, match="hidden by"):
        source_identity(checkout)


def test_a_tree_that_is_not_a_checkout_has_no_source_identity(tmp_path):
    plain = tmp_path / "tarball"
    (plain / "nn").mkdir(parents=True)
    (plain / "nn" / "engine.py").write_text("VALUE = 1\n")
    with pytest.raises(SourceIdentityError, match="not a git checkout"):
        source_identity(plain)


def test_this_repository_identifies_itself(request):
    """The real checkout, so the count cannot silently become site-packages again."""
    identity = source_identity(Path(__file__).resolve().parent.parent)
    assert identity["scheme"] == SOURCE_IDENTITY_SCHEME
    # Two orders of magnitude below the 1,805 the audited P3 cells recorded, and
    # a ceiling rather than an exact count so adding a module does not fail here.
    assert 10 < identity["source_files"] < 300


# --- and what the comparator does with it ------------------------------------
def _cell(name: str, **code):
    payload = {
        "checkpoint": "P2b",
        "question": "q",
        "code": {"scheme": SOURCE_IDENTITY_SCHEME, "source_digest": "a" * 64, **code},
        "numerics": {"python": "3.11.9"},
        "contract": {"contract_id": "btc-usdt-1h-gen1"},
        "sizes": {"research_rows": 10},
        "target": {"horizon": 6},
        "threshold_selection": {"grid": [0.4]},
        "snapshot": {"rows": 10},
        "feature_spec": {"combined_spec_hash": "d" * 64},
        "alignment": {"folds": [{"fold": 0}]},
        "folds": [
            {
                "samples": {"outer_validation": 3},
                "periods": {},
                "outer_validation": {
                    "majority_baseline": {},
                    "momentum_baseline": {},
                    "economic_references": {},
                },
            }
        ],
    }
    return {"information_set": name, "model": "xgboost", "payload": payload}


def test_two_cells_from_one_source_join():
    parity = check_cells_agree([_cell("ohlcv14"), _cell("smc_v1")])
    assert parity["code"]["source_digest"] == "a" * 64
    assert parity["code"]["schemes"] == [SOURCE_IDENTITY_SCHEME]


def test_a_same_model_source_disagreement_is_refused_even_on_one_clean_revision():
    """The exemption, gone for every cell recorded under the new scheme.

    Both cells are the same model and the same clean revision — the case the old
    rule let through — and under a digest that no longer moves with the import
    graph, a disagreement can only mean different executable source.
    """
    revision = "1" * 40
    control = _cell("ohlcv14", revision=revision, dirty=False)
    other = _cell("smc_v1", source_digest="e" * 64, revision=revision, dirty=False)
    with pytest.raises(ComparisonError, match="materially different ProjectChimera source"):
        check_cells_agree([control, other])


def test_the_legacy_exemption_still_applies_to_cells_that_recorded_no_scheme():
    """So the committed P2b, P2c and P3 comparisons stay reproducible.

    Selected by the cell's own recorded scheme, never by a path or a checkpoint
    name, so no cell produced after the change can reach it.
    """
    revision = "1" * 40
    control = _cell("ohlcv14", revision=revision, dirty=False)
    other = _cell("smc_v1", source_digest="e" * 64, revision=revision, dirty=False)
    for cell in (control, other):
        cell["payload"]["code"].pop("scheme")
    parity = check_cells_agree([control, other])
    assert parity["code"]["schemes"] == ["import-graph/legacy"]


def test_the_two_schemes_may_not_be_mixed_in_one_comparison():
    control = _cell("ohlcv14")
    other = _cell("smc_v1")
    other["payload"]["code"].pop("scheme")
    with pytest.raises(ComparisonError, match="different rules"):
        check_cells_agree([control, other])


def test_cells_fitted_under_different_numerical_environments_are_refused():
    control = _cell("ohlcv14")
    other = _cell("smc_v1")
    other["payload"]["numerics"] = {"python": "3.12.4"}
    with pytest.raises(ComparisonError, match="different numerical environments"):
        check_cells_agree([control, other])


def test_a_cell_that_records_no_numerical_environment_cannot_join_one_that_does():
    control = _cell("ohlcv14")
    other = _cell("smc_v1")
    other["payload"].pop("numerics")
    with pytest.raises(ComparisonError, match="numerical environment was recorded"):
        check_cells_agree([control, other])
