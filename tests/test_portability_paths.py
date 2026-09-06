"""Portability guards: explicit encodings, OS-neutral paths, numpy-safe dtypes.

Three defects made this suite Windows-only-red, and each is the kind that comes
back the moment someone writes the convenient thing:

* ``Path.read_text()`` with no ``encoding`` decodes with the *locale* codec. On
  a Windows runner that is cp1252, and 477 tracked files in this repository
  carry non-ASCII bytes, so the convenient call raises ``UnicodeDecodeError``
  on content that is perfectly valid UTF-8.
* ``str(Path(...))`` renders ``\\`` on Windows. The frozen manifests and the
  artifact index record POSIX paths, so a path rebuilt that way never matches
  the string it is compared against -- and in one case emptied a set through a
  ``"/" in p`` filter until the test tripped its own ``assert``.
* ``.to_numpy()`` on a tz-aware pandas object yields an *object* array of
  ``Timestamp``s, and datetime arithmetic on object arrays is exactly what
  numpy has been narrowing.

These guards are deliberately written to fail on **every** OS rather than only
on Windows: a check that can only go red on a platform most contributors never
run is a check that goes unnoticed until CI reddens.
"""

from __future__ import annotations

import ast
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
PACKAGES = ("chimera", "nn", "tools", "tests", "strategies")

#: Where ``encoding`` sits positionally in each call, so that a call which
#: already passes it there counts as explicit rather than as bare.
#: ``Path.read_text(encoding=None, errors=None)``
#: ``Path.write_text(data, encoding=None, errors=None, newline=None)``
TEXT_IO = {"read_text": 0, "write_text": 1}


def _python_files() -> list[Path]:
    return sorted(p for pkg in PACKAGES for p in (REPO / pkg).rglob("*.py"))


def _bare_text_io(source: str) -> list[tuple[int, str]]:
    """Every ``read_text``/``write_text`` call in ``source`` with no encoding."""
    offenders: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr not in TEXT_IO:
            continue
        if any(kw.arg == "encoding" or kw.arg is None for kw in node.keywords):
            continue
        if len(node.args) > TEXT_IO[func.attr]:  # passed positionally, still explicit
            continue
        offenders.append((node.lineno, func.attr))
    return offenders


def test_every_text_read_and_write_declares_its_encoding():
    """No call may fall back to the locale codec."""
    offenders: list[str] = []
    for path in _python_files():
        for lineno, attr in _bare_text_io(path.read_text(encoding="utf-8")):
            offenders.append(f"{path.relative_to(REPO).as_posix()}:{lineno}: {attr}()")
    assert not offenders, (
        "these calls decode or encode with the locale codec, which is cp1252 on a "
        "Windows runner and fails on the repository's own UTF-8 content:\n  "
        + "\n  ".join(offenders)
    )


def _duplicated_encoding(source: str) -> list[tuple[int, str]]:
    """Calls that pass ``encoding`` both positionally and by keyword.

    ``read_text("utf-8", encoding="utf-8")`` raises ``TypeError: got multiple
    values for argument 'encoding'``. A sweep that adds the keyword without
    looking at the positional arguments produces exactly this, and it produces
    it in the files least likely to be read closely afterwards.
    """
    duplicated: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr not in TEXT_IO:
            continue
        by_keyword = any(kw.arg == "encoding" for kw in node.keywords)
        positionally = len(node.args) > TEXT_IO[func.attr]
        if by_keyword and positionally:
            duplicated.append((node.lineno, func.attr))
    return duplicated


def test_no_call_passes_encoding_twice():
    offenders: list[str] = []
    for path in _python_files():
        for lineno, attr in _duplicated_encoding(path.read_text(encoding="utf-8")):
            offenders.append(f"{path.relative_to(REPO).as_posix()}:{lineno}: {attr}()")
    assert not offenders, (
        "these calls pass encoding positionally *and* by keyword, which raises "
        "TypeError at call time:\n  " + "\n  ".join(offenders)
    )


def test_the_duplicate_encoding_guard_actually_bites():
    assert _duplicated_encoding('Path("x").read_text("utf-8", encoding="utf-8")') == [
        (1, "read_text")
    ]
    assert _duplicated_encoding('p.write_text(body, "utf-8", encoding="utf-8")') == [
        (1, "write_text")
    ]
    # A positional encoding on its own is already explicit and must not be flagged.
    assert _duplicated_encoding('Path("x").read_text("utf-8")') == []
    assert _duplicated_encoding('p.write_text(body, encoding="utf-8")') == []


def test_the_encoding_guard_actually_bites():
    """Negative control: the guard above must see a bare call."""
    assert _bare_text_io("Path('x').read_text()") == [(1, "read_text")]
    assert _bare_text_io("Path('x').write_text(body)") == [(1, "write_text")]
    # ...and must not fire on a call that declares one.
    assert _bare_text_io("Path('x').read_text(encoding='utf-8')") == []
    assert _bare_text_io("Path('x').read_text('utf-8')") == []
    assert _bare_text_io("Path('x').write_text(body, encoding='utf-8')") == []


#: The evidence tests that rebuild a manifest-relative path and compare it as a
#: string. Each must render POSIX separators explicitly rather than relying on
#: the host separator matching the recorded one.
POSIX_PATH_SITES = {
    "tests/test_reporting_integrity.py": 3,
    "tests/test_p6_evidence.py": 1,
    "tests/test_p7_evidence.py": 1,
    "tests/test_p13_evidence.py": 1,
}


@pytest.mark.parametrize("relative,expected", sorted(POSIX_PATH_SITES.items()))
def test_evidence_tests_render_paths_as_posix(relative: str, expected: int):
    """A rebuilt evidence path is compared against a POSIX string, so make it one."""
    source = (REPO / relative).read_text(encoding="utf-8")
    rendered = source.count(".as_posix()") + source.count("PurePosixPath(")
    assert rendered >= expected, (
        f"{relative} should render at least {expected} path(s) as POSIX; found "
        f"{rendered}. str(Path(...)) yields backslashes on Windows and never "
        "matches the manifest."
    )
    assert "str(Path(entry).relative_to" not in source
    assert "{str(Path(p).parent)" not in source


def test_git_style_paths_round_trip_through_pureposixpath():
    """`git ls-files` prints POSIX separators; PurePosixPath keeps them on any OS."""
    entry = "artifacts/walkforward/btc_nested_v3_seed_42/walkforward.json"
    relative = PurePosixPath(entry).relative_to("artifacts")
    assert relative.as_posix() == "walkforward/btc_nested_v3_seed_42/walkforward.json"
    assert relative.parent.as_posix() == "walkforward/btc_nested_v3_seed_42"
    assert "\\" not in relative.as_posix()


# The daily timeframe needs more than a 78-bar warm-up, so the fixture has to
# span more than 78 days of hourly candles before any row is eligible.
def _synthetic_candles(rows: int = 2600) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    dates = pd.date_range("2021-01-01", periods=rows, freq="h", tz="UTC")
    close = 30000 + np.cumsum(rng.normal(0, 20, rows))
    return pd.DataFrame(
        {
            "date": dates,
            "open": close + rng.normal(0, 4, rows),
            "high": close + np.abs(rng.normal(0, 10, rows)),
            "low": close - np.abs(rng.normal(0, 10, rows)),
            "close": close,
            "volume": np.abs(rng.normal(100, 20, rows)),
        }
    )


def test_tz_aware_to_numpy_without_a_dtype_is_an_object_array():
    """The two-sided half: this is why `nn.mtf` asks for a dtype explicitly."""
    dates = pd.date_range("2021-01-01", periods=4, freq="h", tz="UTC")
    assert dates.to_numpy().dtype == np.dtype("O")
    assert dates.to_numpy(dtype="datetime64[ns]").dtype == np.dtype("<M8[ns]")


def test_mtf_staleness_is_computed_in_native_datetime64():
    """`nn.mtf` must not do datetime arithmetic on object arrays."""
    from nn.mtf import build_mtf_context

    context = build_mtf_context(_synthetic_candles())
    for name, timeframe in context.per_timeframe.items():
        assert timeframe.staleness_hours.dtype == np.float64, name
        assert timeframe.as_of.dtype.kind == "i", name
        closes = timeframe.bars["close_time"].to_numpy(dtype="datetime64[ns]")
        assert closes.dtype == np.dtype("<M8[ns]"), name


def test_mtf_source_asks_for_the_datetime_dtype():
    """Regression guard: the repair is a dtype request, and it must stay one."""
    source = (REPO / "nn" / "mtf.py").read_text(encoding="utf-8")
    assert source.count('to_numpy(dtype="datetime64[ns]")') >= 4
    assert 'as_of = np.searchsorted(closes, dates.to_numpy(), side="right")' not in source


LOCK = REPO / "requirements-lock.txt"


def test_the_lock_file_exists_and_pins_every_line():
    assert LOCK.is_file(), "requirements-lock.txt is missing"
    pins = [
        line.strip()
        for line in LOCK.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert pins, "the lock file declares no dependency"
    unpinned = [p for p in pins if "==" not in p]
    assert not unpinned, f"these lines are not pinned to an exact version: {unpinned}"
    names = {p.split("==")[0].lower().replace("_", "-") for p in pins}
    # Everything `chimera/` imports at runtime has to be in the locked set.
    for core in ("numpy", "pandas", "pyarrow", "requests", "prometheus-client"):
        assert core in names, f"{core} is a core dependency but is not locked"
    assert "projectchimera" not in names, "the checkout is installed with -e ., not pinned"


def test_the_lock_file_is_used_by_ci():
    workflow = (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "-r requirements-lock.txt" in workflow, "CI does not install from the lock"
    assert "windows-latest" in workflow, "the test matrix does not cover Windows"
    assert "ubuntu-latest" in workflow


GITATTRIBUTES = REPO / ".gitattributes"


def test_line_endings_are_never_converted():
    """Frozen manifests hash bytes, so a CRLF checkout would fail all of them."""
    assert GITATTRIBUTES.is_file(), (
        ".gitattributes is missing; Git for Windows would check text files out "
        "with CRLF and every frozen manifest would fail on the Windows runner"
    )
    rules = [
        line.strip()
        for line in GITATTRIBUTES.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert "* -text" in rules, f"expected a repository-wide `* -text` rule, found {rules}"


def test_frozen_manifests_cover_text_files_and_are_hashed_as_bytes():
    """The two facts that make the rule above load-bearing, asserted together."""
    from tools.freeze_evidence import digest

    manifests = sorted((REPO / "artifacts").glob("*SHA256SUMS.txt"))
    assert manifests, "no frozen manifest found"
    covered = [name for manifest in manifests for _, name in _manifest_entries(manifest)]
    assert any(
        name.endswith((".md", ".json")) for name in covered
    ), "no manifest covers a text file, which would make the CRLF rule moot"
    # `digest` is a byte hash, so a line-ending rewrite changes it.
    sample = REPO / covered[0]
    if sample.is_file():
        import hashlib

        assert digest(sample) == hashlib.sha256(sample.read_bytes()).hexdigest()


def _manifest_entries(manifest: Path) -> list[tuple[str, str]]:
    from tools.freeze_evidence import manifest_entries

    return manifest_entries(manifest)


#: Production helpers that serialise a repository-relative path into a manifest,
#: an artifact payload or a snapshot. Each must render POSIX separators: the
#: committed manifests use them, and a file written on Windows has to agree.
PRODUCTION_POSIX_SITES = (
    "tools/freeze_evidence.py",
    "nn/p6.py",
    "tools/export_trade_snapshot.py",
    "tools/export_derivatives_snapshot.py",
    "tools/acquire_multiclock_source.py",
)


@pytest.mark.parametrize("relative", PRODUCTION_POSIX_SITES)
def test_production_helpers_serialise_posix_paths(relative: str):
    source = (REPO / relative).read_text(encoding="utf-8")
    assert ".as_posix()" in source, f"{relative} renders a path without as_posix()"
    assert "str(path.resolve().relative_to(" not in source
    assert "str(resolved.relative_to(" not in source
    assert "str(candidate.resolve().relative_to(" not in source


def test_freeze_evidence_records_posix_paths():
    """Functional: the manifest writer's own renderer, on a nested path."""
    from tools import freeze_evidence

    rendered = freeze_evidence.relative(REPO / "artifacts" / "btc_v4_SHA256SUMS.txt")
    assert rendered == "artifacts/btc_v4_SHA256SUMS.txt"
    assert "\\" not in rendered


def test_ci_verifies_every_frozen_manifest():
    workflow = (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "Verify every frozen evidence manifest" in workflow
    assert "from tools.freeze_evidence import check" in workflow


def test_pyyaml_is_declared_rather_than_inherited():
    """tests/test_observability.py imports yaml directly."""
    pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert "PyYAML" in pyproject, "PyYAML is imported by the suite but never declared"
