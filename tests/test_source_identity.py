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
    LEGACY_SOURCE_IDENTITY_SCHEMES,
    RECOGNISED_SOURCE_IDENTITY_SCHEMES,
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


def test_a_real_virtualenv_under_a_source_root_is_still_excluded(checkout):
    """The case that must keep working after the name-based filter is gone.

    A virtualenv is recognised by the ``pyvenv.cfg`` at its root, not by being
    called ``.venv``, so its contents stay out of the digest — which is the
    whole reason the exclusion exists.
    """
    before = digest(checkout)
    venv = checkout / "tools" / ".venv"
    venv.mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n")
    library = venv / "lib" / "python3.11" / "site-packages" / "sklearn"
    library.mkdir(parents=True)
    (library / "__init__.py").write_text("__version__ = '1.9.0'\n")
    (venv / "lib" / "python3.11" / "shim.py").write_text("X = 1\n")

    identity = source_identity(checkout)
    assert identity["source_digest"] == before
    assert identity["excluded_environment_files"] >= 2


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


def test_a_tracked_module_under_a_directory_called_build_is_hashed(checkout):
    """The blind spot the first version of the exclusion opened from the far side.

    ``nn/build/evil.py`` is importable as ``nn.build.evil``. The name-based
    filter dropped it because one of its directories was called ``build``, so a
    module research could execute was invisible to the identity that exists to
    say what research executed — the same failure as the in-repository
    virtualenv, arrived at from the opposite direction.
    """
    before = digest(checkout)
    evil = checkout / "nn" / "build"
    evil.mkdir(parents=True)
    (evil / "__init__.py").write_text("")
    (evil / "evil.py").write_text("def patch():\n    return 'owned'\n")
    git(checkout, "add", "-A")
    git(checkout, "commit", "-qm", "a module under a directory called build")

    identity = source_identity(checkout)
    assert identity["source_digest"] != before
    assert identity["excluded_environment_files"] == 0
    # And it is the content that moves it, not merely the file existing.
    (evil / "evil.py").write_text("def patch():\n    return 'owned differently'\n")
    assert digest(checkout) != identity["source_digest"]


@pytest.mark.parametrize("directory", ["build", "dist", "env", "venv", ".venv", "ENV"])
def test_an_environment_shaped_name_alone_never_hides_project_source(checkout, directory):
    """A directory's name is not evidence about what it holds.

    Two acceptable outcomes and one unacceptable one. If git can see the file it
    enters the digest; if ``.gitignore`` hides it, identity refuses to answer at
    all. What must never happen is the third thing — the file silently absent
    from a digest that reports success, which is what a name-based filter did.
    """
    before = digest(checkout)
    target = checkout / "nn" / directory
    target.mkdir(parents=True)
    (target / "helper.py").write_text("VALUE = 2\n")
    git(checkout, "add", "-A")

    try:
        after = digest(checkout)
    except SourceIdentityError as exc:
        assert "hidden by" in str(exc)
        return
    assert after != before, f"nn/{directory}/helper.py vanished from the digest"


def test_a_gitignored_module_under_a_build_directory_is_refused(checkout):
    """Hiding it from git as well must not buy silence either."""
    (checkout / ".gitignore").write_text(".venv/\nnn/build/\n")
    evil = checkout / "nn" / "build"
    evil.mkdir(parents=True)
    (evil / "evil.py").write_text("X = 1\n")
    with pytest.raises(SourceIdentityError, match="hidden by"):
        source_identity(checkout)


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


# --- a cell may not name its own way out of the current rule -----------------
@pytest.mark.parametrize(
    "scheme",
    [
        "import-graph/legacy",
        "git-tracked-source/2",
        "git-tracked-source/0",
        "whatever/1",
        "",
    ],
)
def test_an_unrecognised_scheme_is_refused_outright(scheme):
    """Recognised by allow-list, before any digest is looked at.

    The check used to be "are the two schemes equal, and is either the current
    one" — so two cells that both declared the same invented scheme agreed with
    each other, failed the current-scheme test, and landed in the historical
    exemption. Inventing a name was a way back into a rule written for cells
    that predate names.
    """
    control = _cell("ohlcv14", scheme=scheme)
    other = _cell("smc_v1", scheme=scheme)
    with pytest.raises(ComparisonError, match="does not recognise"):
        check_cells_agree([control, other])


def test_an_invented_scheme_cannot_reach_the_historical_exemption():
    """The exploit the allow-list closes, spelled out.

    Same clean revision, differing digests, a scheme nobody has ever defined:
    under the old ordering this joined, on a rule that exists only to keep the
    committed P2b/P2c/P3 comparisons reproducible.
    """
    revision = "1" * 40
    control = _cell("ohlcv14", scheme="legacy-ish/1", revision=revision, dirty=False)
    other = _cell(
        "smc_v1",
        scheme="legacy-ish/1",
        source_digest="e" * 64,
        revision=revision,
        dirty=False,
    )
    with pytest.raises(ComparisonError, match="does not recognise"):
        check_cells_agree([control, other])


def test_the_recognised_set_is_the_current_scheme_and_the_unnamed_historical_one():
    assert RECOGNISED_SOURCE_IDENTITY_SCHEMES == {SOURCE_IDENTITY_SCHEME, None}
    assert LEGACY_SOURCE_IDENTITY_SCHEMES == {None}
    assert SOURCE_IDENTITY_SCHEME not in LEGACY_SOURCE_IDENTITY_SCHEMES


def test_the_committed_cells_are_the_population_the_exemption_is_for():
    """Falsifies the three above: the exemption still applies to real history."""
    import json

    for cell in ("btc_p3_ohlcv14_xgboost", "btc_p3_ohlcv14_lightgbm"):
        payload = json.loads(
            (
                Path(__file__).resolve().parent.parent
                / "artifacts"
                / "benchmark"
                / cell
                / "p2b.json"
            ).read_text()
        )
        assert payload["code"].get("scheme") in LEGACY_SOURCE_IDENTITY_SCHEMES


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
