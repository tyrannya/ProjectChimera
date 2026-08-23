"""Identity of the ProjectChimera source a research process executed.

Every other identity a research cell records — the contract hash, the snapshot
hashes, the feature-spec hash, the library versions — describes the *data* and
the *definitions*. None of them changes when the runner does, so cells built by
different revisions of this repository would join into one comparison without a
word. This module answers the remaining question: **which ProjectChimera source
produced this cell.**

**What it is a digest of, and what it deliberately is not.** The first version
of this hashed every module a process had imported whose file resolved inside
the repository directory. That sounded like "what code ran" and was not: a
developer's in-repository ``.venv/lib/python3.11/site-packages`` resolves inside
the repository too, so the digest covered scikit-learn, LightGBM and XGBoost as
well as ``nn/``. The nine P3 cells recorded 1,805, 1,811 and 1,824 "source"
files against this repository's 92 tracked Python files, and the digest split
three ways *by model family* — because the three families import different
libraries — while every line of ProjectChimera source was identical. A
comparator exemption was then added to let those cells join.

The identity here is defined the other way round: it is the content of the
Python files git tracks under this repository's executable package roots, plus
any untracked Python file sitting among them. An installed environment is not
under those roots and cannot enter the digest whatever a process imports; a
change to a module research can execute always does.

**Fail-closed.** A Python file under a source root that ``.gitignore`` hides is
refused rather than silently omitted, because that is precisely the shape of the
hole this replaces — a module research can import that source identity cannot
see. A tracked file deleted from the working tree is recorded as absent rather
than skipped, so removing a module moves the digest.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable

#: The package roots this repository ships as executable source — the same four
#: ``pyproject.toml`` installs. ``tests/`` is deliberately absent: it is not
#: importable by a research run, and including it would split a batch of cells
#: across a commit that only added a test.
SOURCE_ROOTS: tuple[str, ...] = ("chimera", "nn", "strategies", "tools")

#: Directory names that are an installed or generated environment rather than
#: this project's source. None of them can legitimately appear under a source
#: root; the check is here because the failure it prevents already happened
#: once, one directory level up.
ENVIRONMENT_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "env",
        "ENV",
        "site-packages",
        "dist-packages",
        "node_modules",
        "__pycache__",
        "build",
        "dist",
    }
)

#: Names the rule the digest is computed under. Recorded in every artifact so a
#: reader — and :mod:`nn.p2b_compare` — can tell a digest taken under this rule
#: from one taken under the import-graph rule it replaced.
SOURCE_IDENTITY_SCHEME = "git-tracked-source/1"

#: What a tracked file that is not in the working tree hashes to.
ABSENT = "absent"


class SourceIdentityError(RuntimeError):
    """The executable source of this repository cannot be identified."""


def _git(root: Path, *args: str) -> str | None:
    """Run git in ``root``; ``None`` when git or the checkout is unavailable."""
    try:
        out = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - environment
        return None
    return out.stdout if out.returncode == 0 else None


def _listing(root: Path, *args: str) -> list[str] | None:
    raw = _git(root, "ls-files", "-z", *args, "--", *SOURCE_ROOTS)
    if raw is None:
        return None
    return [entry for entry in raw.split("\0") if entry]


def _python_files(entries: Iterable[str]) -> list[str]:
    """The ``.py`` entries that are project source rather than an environment."""
    out = []
    for entry in entries:
        path = Path(entry)
        if path.suffix != ".py":
            continue
        if ENVIRONMENT_DIRS.intersection(path.parts):
            continue
        out.append(entry)
    return sorted(set(out))


def source_identity(root: Path) -> dict[str, Any]:
    """Identify the ProjectChimera source present in the checkout at ``root``.

    Raises :class:`SourceIdentityError` when the tree is not a git checkout —
    research source identity is the whole point of the record, and a run that
    cannot establish one must not produce evidence that looks as if it had —
    and when a Python file under a source root is hidden by ``.gitignore``.
    """
    root = Path(root).resolve()
    tracked = _listing(root, "--cached")
    if tracked is None:
        raise SourceIdentityError(
            f"{root} is not a git checkout, or git is unavailable, so the executable "
            "ProjectChimera source that produced this run cannot be identified. Research "
            "evidence records which source produced it; run from a checkout."
        )
    untracked = _listing(root, "--others", "--exclude-standard") or []
    hidden = _python_files(_listing(root, "--others", "--ignored", "--exclude-standard") or [])
    if hidden:
        raise SourceIdentityError(
            f"{len(hidden)} Python file(s) under {list(SOURCE_ROOTS)} are hidden by "
            f".gitignore (first: {hidden[0]}). A module research can import but source "
            "identity cannot see is the exact hole this digest exists to close; either "
            "track the file or move it out of the package roots."
        )

    names = _python_files(tracked) + _python_files(untracked)
    files = sorted(set(names))
    sources: dict[str, str] = {}
    for name in files:
        path = root / name
        sources[name] = (
            hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ABSENT
        )

    material = {
        "scheme": SOURCE_IDENTITY_SCHEME,
        "roots": list(SOURCE_ROOTS),
        "files": sources,
    }
    digest = hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    head = _git(root, "rev-parse", "HEAD")
    status = _git(root, "status", "--porcelain", "--", *SOURCE_ROOTS)
    untracked_source = set(_python_files(untracked))
    missing = sorted(name for name, value in sources.items() if value == ABSENT)

    return {
        "scheme": SOURCE_IDENTITY_SCHEME,
        "roots": list(SOURCE_ROOTS),
        "revision": head.strip() if head else None,
        "dirty": None if status is None else bool(status.strip()),
        "source_digest": digest,
        "source_files": len(files),
        "untracked_source_files": len(untracked_source),
        "missing_tracked_source_files": len(missing),
        "note": (
            "source_digest is a SHA-256 over the content of every Python file git tracks "
            f"under {list(SOURCE_ROOTS)}, plus any untracked Python file among them. An "
            "installed environment — a virtualenv inside the checkout, site-packages, a "
            "model library imported lazily by one model family and not another — is not "
            "under those roots and cannot move it. revision and dirty are recorded beside "
            "it because neither one alone says what source ran"
        ),
    }
