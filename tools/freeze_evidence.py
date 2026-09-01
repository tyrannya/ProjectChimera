"""Freeze a checkpoint's primary artifacts behind a checksum manifest, or verify one.

    python -m tools.freeze_evidence --out artifacts/btc_p2b_SHA256SUMS.txt \
        artifacts/benchmark/btc_p2b_*
    python -m tools.freeze_evidence --verify artifacts/btc_p2b_SHA256SUMS.txt

The repository already keeps ``artifacts/btc_v4_SHA256SUMS.txt`` and
``artifacts/btc_p2a_SHA256SUMS.txt``, and a test asserts every covered file
still hashes to the value the freeze recorded. Both were produced by hand. This
is the same format and the same guarantee, produced by something that cannot
skip a file or transcribe a digit wrong.

**A manifest covers primary evidence only, and freezing enforces it.** A
directory whose artifact declares itself ``derived`` is refused, because a
derived file is regenerated whenever its aggregator improves and a hash over it
is a promise the workflow is designed to break. That is not hypothetical: three
manifests on this branch covered comparison directories, all three failed their
own ``--verify``, and the test that should have caught it excused the failures
by matching ``_comparison/`` in the path. A checksum that means "frozen unless
it is not" trains a reader to ignore the one signal it exists to send.

Derived evidence is not unchecked; it is checked differently. It is regenerated
from the frozen primary evidence and what it *says* is asserted — the fold
counts, the verdicts, the recomputation and anchoring problem counts — so a
regenerated report that changed a finding fails, and one that only improved its
own prose does not.

**Freezing refuses to overwrite.** An existing manifest is the repository's own
statement about what a past run produced; regenerating it in place is how a
result quietly becomes whatever the code does today. A checkpoint that needs new
numbers gets a new manifest under a new name, and the old one stays.

``--verify`` is the read side and is what belongs in a test: it re-hashes every
covered file and reports the ones that moved or vanished. It has no exemptions,
and :func:`check` is the single implementation both the CLI and the test suite
call, so the two cannot drift into disagreeing about what "frozen" means.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent

#: The two evidence classes, and the artifact key that declares one.
#:
#: **Primary** evidence is what a run produced and nothing can rebuild: a fitted
#: cell's metrics and its per-sample outer predictions. Reproducing it means
#: re-fitting, so a byte change in it is a change in the research result.
#:
#: **Derived** evidence is recomputed from primary evidence by an aggregator —
#: a comparison, an ablation table, a regime description. It changes whenever
#: the aggregator improves, and it is *supposed* to: widening a recomputation
#: from ten trading keys to twenty-three rewrites the file without touching a
#: single number the research reported.
#:
#: The distinction exists because mixing them was incoherent. Three manifests
#: here covered comparison files, so three manifests failed their own
#: ``--verify`` the moment a reporter was improved, and a test excused the
#: failures by matching the directory name — which is a checksum that means
#: "this file is frozen unless it is not". Immutable manifests now cover primary
#: evidence only, and derived evidence is pinned by regenerating it and checking
#: what it says.
PRIMARY = "primary"
DERIVED = "derived"
EVIDENCE_CLASS_KEY = "evidence_class"

#: Extensions worth freezing. A checkpoint directory can also hold caches and
#: editor droppings, and a manifest that covers those fails for reasons that
#: have nothing to do with the evidence.
#:
#: **The compressed half is not decoration.** P13's preregistered sources are
#: Binance archive objects — ``{SYMBOL}-{interval}-{YYYY}-{MM}.zip`` and its
#: ``.zip.CHECKSUM`` companion, per ``nn.p13_preregistration.DATA_SOURCES`` — and
#: a covering rule keyed on extension would have let every one of them land in a
#: checkpoint directory UNMANIFESTED, for no better reason than being compressed.
#: The two directions of the rule fail in opposite ways there: the freezer would
#: not hash them, and the coverage test would not notice they were unhashed. A
#: source object is the most load-bearing evidence a checkpoint has, so the
#: suffixes are widened to reach it rather than left to be discovered by the
#: acquisition run.
#:
#: Widened deliberately narrowly: the archive formats the frozen designs name,
#: and ``.gz`` beside ``.zip`` because it is the other shape published source
#: data arrives in. Not a general "freeze every binary" rule — the scan still
#: only ever walks the directories it is given.
EVIDENCE_SUFFIXES = (
    ".json",
    ".md",
    ".parquet",
    ".csv",
    ".txt",
    ".zip",
    ".gz",
    ".CHECKSUM",
)


def evidence_class_of(directory: Path) -> str | None:
    """What a run directory says it is, read from the artifact the run wrote.

    ``None`` when nothing in the directory declares a class — the walk-forward
    and diagnostics artifacts predate the distinction and are frozen as they
    always were. Only an explicit ``derived`` is refused, so a new aggregator
    has to opt *into* being excluded rather than remember to opt out.
    """
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (ValueError, OSError):
            continue
        if isinstance(payload, dict) and EVIDENCE_CLASS_KEY in payload:
            return str(payload[EVIDENCE_CLASS_KEY])
    return None


def evidence_files(directories: Iterable[Path]) -> list[Path]:
    """Every primary evidence file under ``directories``, repo-relative and sorted.

    Raises rather than silently skipping a derived directory: an operator who
    globbed ``btc_p2b_*`` and caught the comparison meant to freeze it, and a
    manifest that quietly dropped part of what it was given would be a worse
    answer than a refusal naming the directory.
    """
    found: set[Path] = set()
    derived: list[Path] = []
    unclassified: list[Path] = []
    for directory in directories:
        directory = Path(directory)
        if directory.is_file():
            found.add(directory.resolve())
            continue
        if not directory.is_dir():
            raise SystemExit(f"{directory} is neither a file nor a directory")
        for path in directory.rglob("*"):
            if not (path.is_file() and path.suffix in EVIDENCE_SUFFIXES):
                continue
            kind = evidence_class_of(path.parent)
            if kind == DERIVED:
                derived.append(path.parent)
                continue
            if kind is None:
                unclassified.append(path.parent)
            found.add(path.resolve())
    if derived:
        raise SystemExit(
            "refusing to freeze derived evidence: "
            + ", ".join(sorted({relative(d) for d in derived}))
            + f". A directory whose artifact declares {EVIDENCE_CLASS_KEY}={DERIVED!r} is "
            "regenerated from primary evidence whenever its aggregator improves, so a "
            "hash over it is a promise this workflow breaks on purpose. Freeze the cells; "
            "pin the report by regenerating it and checking what it says."
        )
    for name in sorted({relative(d) for d in unclassified}):
        print(f"note: {name} declares no {EVIDENCE_CLASS_KEY}; freezing it as primary")
    return sorted(found)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        raise SystemExit(f"{path} is outside the repository; refusing to freeze it")


def freeze(directories: Iterable[Path], out: Path) -> int:
    if out.exists():
        raise SystemExit(
            f"{out} already exists. A frozen manifest is a statement about what a past "
            "run produced; regenerating it in place is how a result quietly becomes "
            "whatever the code does today. Write a new manifest under a new name."
        )
    files = evidence_files(directories)
    if not files:
        raise SystemExit("no evidence files found; refusing to write an empty manifest")
    lines = [f"{digest(path)}  {relative(path)}" for path in files]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"froze {len(lines)} files into {out}")
    return 0


def manifest_entries(manifest: Path) -> list[tuple[str, str]]:
    """``(sha256, repo-relative path)`` per covered file, in manifest order."""
    entries: list[tuple[str, str]] = []
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        expected, name = line.split(maxsplit=1)
        entries.append((expected, name.strip()))
    return entries


def check(manifest: Path, *, root: Path | None = None) -> list[str]:
    """Every covered file that vanished or moved, described. Empty means frozen.

    The whole of the verification, with no exemption of any kind — not by
    directory, not by suffix, not by evidence class. Both ``--verify`` and the
    test suite call this, so "frozen" has exactly one meaning in this
    repository.

    ``root`` is the tree the manifest's repository-relative paths resolve
    against, and it defaults to this repository. It is an argument because
    :func:`nn.p4_holdout.assert_frozen_stage_one` has to verify a manifest in
    whatever tree it was handed — a guard that checked one tree while being
    asked about another would report "frozen" about files nobody looked at.
    """
    tree = ROOT if root is None else Path(root)
    problems: list[str] = []
    for expected, name in manifest_entries(manifest):
        path = tree / name
        if not path.is_file():
            problems.append(f"MISSING  {name}")
            continue
        actual = digest(path)
        if actual != expected:
            problems.append(
                f"CHANGED  {name}\n           was {expected}\n           now {actual}"
            )
    return problems


def verify(manifest: Path) -> int:
    entries = manifest_entries(manifest)
    problems = check(manifest)
    for problem in problems:
        print(problem, file=sys.stderr)
    print(f"{len(entries) - len(problems)}/{len(entries)} files match {manifest}")
    return 1 if problems else 0


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("directories", nargs="*", type=Path)
    parser.add_argument("--out", type=Path, help="write a new manifest here")
    parser.add_argument(
        "--verify", type=Path, help="re-hash the files an existing manifest covers"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.verify:
        return verify(args.verify)
    if not args.out or not args.directories:
        raise SystemExit("give --out and at least one directory, or --verify MANIFEST")
    return freeze(args.directories, args.out)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
