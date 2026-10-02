"""R1-n's backup/restore drill: prove the backup restores, not that it exists.

    python -m tools.backup_drill \
        --recorder-root /var/lib/chimera/data \
        --state-dir     /var/lib/chimera-demo \
        --archive       /backup/chimera-$(date -u +%Y%m%dT%H%M%SZ).tar.gz \
        --restore-to    /var/tmp/chimera-restore-drill

WHAT A DRILL IS, AND WHY THE WORD MATTERS. An untested backup is a belief. The
failure this guards against is not "nobody made a copy" -- it is the ordinary
one where a copy exists, is made nightly, is monitored, and turns out at the
worst possible moment to be missing the directory that mattered, or to have been
written while a file was half-flushed, or to extract into a path nobody can use.
So this tool does all three steps in one invocation and fails if the third
disagrees with the first: archive the two roots, restore them somewhere else,
and compare every file by SHA-256. A run that exits 0 is a restore that was
actually performed and checked, minutes ago, on this host.

WHAT IT BACKS UP. Two roots, which after R1-n belong to two different Unix
accounts and are the only durable state the demo has:

* the recorder's storage root -- raw NDJSON.gz, normalized parquet, the day
  manifests and everything under ``health/``;
* the runner's state directory -- the carry ledger, the risk snapshot, the
  decision log, the runner state.

The recorder's own files are named here by directory rather than one by one, and
that is deliberate rather than vague. R1-h asserts that nothing outside
``chimera/recorder/`` so much as NAMES the recorder's lifecycle log -- the check
is a text search over ``chimera``, ``nn`` and ``tools`` against an exact
allow-list of two modules, and it does not care that the mention is in a
docstring. This tool archives whole directories and never opens that file by
name, so it has no need to say it, and earning a place on another item's
allow-list for the sake of a doc phrase is how such a list stops meaning
anything. CI caught this; the first draft of this docstring named the file.

WHAT IT DOES NOT DO, stated here rather than discovered during an incident:

* it is not a snapshot. It reads a live tree, so a file being rewritten while
  the archive walks past it can be captured mid-write. R1-h made every parquet
  and manifest write a temp-file + fsync + rename, which is what makes this
  acceptable: a reader -- including this one -- sees either the old file or the
  new one, never half of either. The append-only logs can end at a partial last
  line, and that is the honest state of an append-only log;
* it does not stop, pause or signal either service, and it must not: a backup
  that halts a campaign is a backup nobody will run;
* it does not rotate, encrypt, or copy anything off the host. Where the archive
  goes afterwards is the operator's decision and belongs in the runbook, not
  in a tool that would then need a credential;
* it does not restore into production. A restore target that is, contains, or
  sits inside either source root is REFUSED -- the one mistake in this
  procedure that destroys the thing it was meant to protect.

EXIT CODES: ``0`` archived, restored and verified identical; ``1`` the restore
did not match, or a source was missing; ``2`` usage error, including an unsafe
restore target.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

#: Read size for digesting. Large enough that a multi-GB recorder root does not
#: turn into a syscall benchmark, small enough to stay off the heap.
CHUNK = 1024 * 1024


@dataclass(frozen=True)
class Source:
    """One root to archive, under the name it takes inside the archive."""

    label: str
    path: Path

    @property
    def arcname(self) -> str:
        return self.label


def digest_file(path: Path) -> str:
    """SHA-256 of one file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def digest_tree(root: Path) -> dict[str, str]:
    """Every regular file under ``root``, as relative path -> SHA-256.

    Symlinks are recorded by their target text rather than followed: a tree that
    restored a link as a copy of its target would compare equal here while being
    a different tree, and the recorder's roots are not expected to contain links
    at all -- so the honest thing is to notice them.
    """
    digests: dict[str, str] = {}
    for current, _dirs, files in os.walk(root):
        for name in files:
            path = Path(current) / name
            relative = str(path.relative_to(root)).replace(os.sep, "/")
            if path.is_symlink():
                digests[relative] = "symlink:" + os.readlink(path)
            else:
                digests[relative] = digest_file(path)
    return digests


def archive(sources: Iterable[Source], destination: Path) -> Path:
    """Write a gzipped tar of every source root. Returns the archive path."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, "w:gz") as tar:
        for source in sources:
            tar.add(source.path, arcname=source.arcname)
    return destination


def restore(source_archive: Path, target: Path) -> Path:
    """Extract the archive under ``target``.

    ``filter="data"`` is not decoration: without it, tar extraction follows
    absolute paths and ``..`` components out of the target directory, and an
    archive is exactly the sort of file an operator copies back from somewhere
    else before trusting it.
    """
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(source_archive, "r:gz") as tar:
        tar.extractall(target, filter="data")
    return target


def compare(sources: Iterable[Source], target: Path) -> list[str]:
    """Every way the restored tree differs from the live one.

    Empty list means the restore is byte-identical, which is the only result
    that makes the backup a fact rather than a belief.
    """
    problems: list[str] = []
    for source in sources:
        live = digest_tree(source.path)
        restored_root = target / source.arcname
        if not restored_root.is_dir():
            problems.append(f"{source.label}: the restore produced no {restored_root}")
            continue
        restored = digest_tree(restored_root)

        for missing in sorted(set(live) - set(restored)):
            problems.append(f"{source.label}: {missing} is missing from the restore")
        for extra in sorted(set(restored) - set(live)):
            problems.append(f"{source.label}: {extra} is in the restore but not live")
        for shared in sorted(set(live) & set(restored)):
            if live[shared] != restored[shared]:
                problems.append(
                    f"{source.label}: {shared} restored with a different digest "
                    f"({live[shared][:12]} live, {restored[shared][:12]} restored)"
                )
    return problems


def unsafe_target(sources: Iterable[Source], target: Path) -> str | None:
    """Why this restore target must be refused, or ``None`` if it is fine.

    The check is in both directions. A target inside a source root would have
    the drill restore a copy of the data into the data, which at best doubles
    the disk use mid-campaign. A target that CONTAINS a source root is the
    serious one: the extraction would write straight over the live tree, and a
    drill that can destroy production is not a drill.
    """
    target = target.resolve()
    for source in sources:
        root = source.path.resolve()
        if target == root:
            return f"the restore target is {source.label} itself ({root})"
        if str(target).startswith(str(root) + os.sep):
            return f"the restore target is inside {source.label} ({root})"
        if str(root).startswith(str(target) + os.sep):
            return (
                f"the restore target contains {source.label} ({root}); extracting "
                "there would write over the live tree"
            )
    return None


def drill(sources: Sequence[Source], archive_path: Path, restore_to: Path) -> list[str]:
    """Archive, restore, compare. Returns the problems found, empty if clean."""
    missing = [s.label for s in sources if not s.path.is_dir()]
    if missing:
        return [f"{label}: source root does not exist" for label in missing]
    archive(sources, archive_path)
    restore(archive_path, restore_to)
    return compare(sources, restore_to)


def build_parser() -> argparse.ArgumentParser:
    """The CLI, as a factory, for the same reason as in `tools.deadman_check`:
    the runbook test checks every documented command against the real parser."""
    parser = argparse.ArgumentParser(
        description="Archive the recorder root and the runner state, restore them "
        "elsewhere, and verify the restore byte for byte.",
    )
    parser.add_argument("--recorder-root", type=Path, help="the recorder's storage root")
    parser.add_argument("--state-dir", type=Path, help="the runner's state directory")
    parser.add_argument("--archive", type=Path, required=True, help="archive to write")
    parser.add_argument(
        "--restore-to",
        type=Path,
        required=True,
        help="an empty scratch directory to restore into; never a live root",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    sources: list[Source] = []
    if args.recorder_root:
        sources.append(Source("recorder-root", args.recorder_root))
    if args.state_dir:
        sources.append(Source("state-dir", args.state_dir))
    if not sources:
        parser.error("give at least one of --recorder-root or --state-dir")

    reason = unsafe_target(sources, args.restore_to)
    if reason is not None:
        parser.error(f"refusing to restore: {reason}")

    problems = drill(sources, args.archive, args.restore_to)
    if problems:
        print("BACKUP DRILL FAILED", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        print(
            "The archive is not a restorable copy of these roots. Do not rely on it.",
            file=sys.stderr,
        )
        return 1

    total = sum(len(digest_tree(s.path)) for s in sources)
    print(
        f"backup drill passed: {len(sources)} root(s), {total} file(s) archived to "
        f"{args.archive} and restored identical under {args.restore_to}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
