"""Describe the acquired P13 sources, and compute no economics while doing it.

Reads the acquisition manifest and the cached bytes it names, parses every object
under the committed A2R2 loader, and writes a source-closure record: rows, units,
first and last instants, gaps, duplicates, boundary truncation, mark publication
coverage, funding settlement counts as SOURCE counts, and — per calendar block —
whether every held hour supplies the rows A2R2 requires.

**No network.** Everything is read from the local cache the acquisition wrote. The
manifest names each object's published checksum, and the loader re-verifies it
against the bytes on disk, so a cache that changed since acquisition fails here
rather than being described.

**No economics.** This tool never calls the governed screen, the block runner's
evaluator or the gate. What it may say about a future run is a statement about
SOURCE COVERAGE — a held hour with no mark row would make that run terminate NOT
EVALUABLE — and that is source insufficiency, never an economic finding.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nn.p13_source_closure import closure_payload, load_acquired_sources
from tools.acquire_p13_sources import source_provenance


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--manifest", type=Path, required=True, help="the acquisition manifest to close over"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help=(
            "where the acquired archives live. REQUIRED: the manifest deliberately "
            "records no machine-local path, so the cache location is supplied here as "
            "local state rather than read out of committed evidence"
        ),
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not manifest.get("complete"):
        raise SystemExit(
            "the acquisition manifest does not report a COMPLETE acquisition. Source "
            "closure over a partial source set would describe a universe the frozen "
            "design did not specify."
        )
    cache_dir = args.cache_dir

    loaded = load_acquired_sources(manifest, cache_dir)
    payload = closure_payload(loaded, manifest=manifest, provenance=source_provenance())

    text = json.dumps(payload, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"source closure written to {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
