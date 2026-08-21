"""Freeze a checkpoint's artifacts behind a checksum manifest, or verify one.

    python -m tools.freeze_evidence --out artifacts/btc_p2b_SHA256SUMS.txt \
        artifacts/benchmark/btc_p2b_*
    python -m tools.freeze_evidence --verify artifacts/btc_p2b_SHA256SUMS.txt

The repository already keeps ``artifacts/btc_v4_SHA256SUMS.txt`` and
``artifacts/btc_p2a_SHA256SUMS.txt``, and a test asserts every covered file
still hashes to the value the freeze recorded. Both were produced by hand. This
is the same format and the same guarantee, produced by something that cannot
skip a file or transcribe a digit wrong.

**Freezing refuses to overwrite.** An existing manifest is the repository's own
statement about what a past run produced; regenerating it in place is how a
result quietly becomes whatever the code does today. A checkpoint that needs new
numbers gets a new manifest under a new name, and the old one stays.

``--verify`` is the read side and is what belongs in a test: it re-hashes every
covered file and reports the ones that moved or vanished.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent

#: Extensions worth freezing. A checkpoint directory can also hold caches and
#: editor droppings, and a manifest that covers those fails for reasons that
#: have nothing to do with the evidence.
EVIDENCE_SUFFIXES = (".json", ".md", ".parquet", ".csv", ".txt")


def evidence_files(directories: Iterable[Path]) -> list[Path]:
    """Every evidence file under ``directories``, relative to the repo root, sorted."""
    found: set[Path] = set()
    for directory in directories:
        directory = Path(directory)
        if directory.is_file():
            found.add(directory.resolve())
            continue
        if not directory.is_dir():
            raise SystemExit(f"{directory} is neither a file nor a directory")
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix in EVIDENCE_SUFFIXES:
                found.add(path.resolve())
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


def verify(manifest: Path) -> int:
    entries = [
        line.split(maxsplit=1) for line in manifest.read_text().splitlines() if line.strip()
    ]
    problems: list[str] = []
    for expected, name in entries:
        path = ROOT / name.strip()
        if not path.is_file():
            problems.append(f"MISSING  {name.strip()}")
            continue
        actual = digest(path)
        if actual != expected:
            problems.append(
                f"CHANGED  {name.strip()}\n           was {expected}\n           now {actual}"
            )
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
