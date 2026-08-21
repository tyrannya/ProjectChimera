"""Refuse a research snapshot that cannot be trusted, before anyone trains on it.

``tools.export_research_snapshot`` proves at export time that the snapshot under
``data/research/`` holds no Styx row. A fresh clone inherits no such proof: it
has four files and a manifest asserting things about them. Assertions and files
drift apart silently — a rebuilt Parquet, a hand-edited manifest, a partial
checkout, a merge that resurrected an older file — because the research
entrypoints read the data and never the manifest.

So every claim is recomputed rather than read, and the first disagreement fails
closed. The digests come from the exporter's own helpers
(:func:`nn.data_fingerprint.research_input_digest`,
:func:`nn.data_fingerprint.fingerprint_raw_input`,
:func:`tools.export_research_snapshot._sha256`): a second implementation of a
hash is a second definition of identity, and the two would drift apart at the
first change to either.

**The seal is checked against the data, not against the manifest's word for it.**
``contains_styx`` and ``safety_checks`` are claims made by whoever wrote the
manifest, so they are held to exactly the values the exporter writes and then
the timestamps in both Parquet files are compared against the contract's sealed
instant directly. That comparison runs before any digest, row-count or span
check, so a snapshot carrying a sealed row is rejected as a seal breach rather
than as an incidental hash mismatch.

**A rejection names the breach, never its content.** The seal check reports how
many rows are at or after the instant and nothing else: a verifier that printed
the offending candle would leak the rows it exists to keep sealed.

Nothing outside ``data/research/`` and ``nn/research_contracts/`` is opened. The
canonical dataset and the frozen walk-forward artifact the exporter compared
against are absent from a research clone, so ``canonical_reference`` and
``safety_checks.canonical_research_input_verified`` are checked for internal
consistency only. They are the exporter's testimony; this tool cannot re-derive
them and does not pretend to.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from nn.data_fingerprint import fingerprint_raw_input, research_input_digest
from nn.data_pipeline import load_dataset
from nn.regime import load_raw_ohlcv
from nn.research_contract import ResearchContractError, load_contract
from tools.export_research_snapshot import MANIFEST_NAME, SNAPSHOT_SCHEMA, _sha256

DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "data" / "research" / MANIFEST_NAME

#: Keys a manifest must carry, with the type each must have. Checked before any
#: value is used, so a wrong-typed field is a named rejection rather than a
#: ``TypeError`` three checks later.
REQUIRED_KEYS: tuple[tuple[str, type], ...] = (
    ("contract", dict),
    ("contract.contract_id", str),
    ("contract.contract_hash", str),
    ("contract.sealed_test_start", str),
    ("contains_styx", bool),
    ("canonical_reference", dict),
    ("canonical_reference.research_rows", int),
    ("canonical_reference.max_outer_end_row", int),
    ("raw_pre_styx", dict),
    ("raw_pre_styx.path", str),
    ("raw_pre_styx.rows", int),
    ("raw_pre_styx.start", str),
    ("raw_pre_styx.end", str),
    ("raw_pre_styx.semantic_hash", str),
    ("raw_pre_styx.sha256", str),
    ("processed_outer_coverage", dict),
    ("processed_outer_coverage.path", str),
    ("processed_outer_coverage.rows", int),
    ("processed_outer_coverage.row_range", list),
    ("processed_outer_coverage.start", str),
    ("processed_outer_coverage.end", str),
    ("processed_outer_coverage.semantic_prefix_hash", str),
    ("processed_outer_coverage.sha256", str),
    ("processed_outer_coverage.metadata_path", str),
    ("processed_outer_coverage.metadata_sha256", str),
    ("safety_checks", dict),
    ("safety_checks.canonical_research_input_verified", bool),
    ("safety_checks.raw_strictly_before_styx", bool),
    ("safety_checks.processed_strictly_before_styx", bool),
    ("safety_checks.outer_horizon_within_research", bool),
    ("safety_checks.styx_rows_exported", int),
)

_MISSING = object()


class SnapshotVerificationError(ValueError):
    """A research snapshot failed one check, named by :attr:`check`.

    The name is an attribute rather than something to be recovered from the
    message, so a caller can branch on *which* guarantee broke without parsing
    prose that is free to change.
    """

    def __init__(self, check: str, detail: str) -> None:
        super().__init__(f"{check}: {detail}")
        self.check = check


@dataclass(frozen=True)
class CheckResult:
    """One check that held. A check that did not raises instead."""

    name: str
    detail: str


def _utc(text: str) -> pd.Timestamp:
    """Parse a manifest instant as UTC, the way contracts parse theirs."""
    stamp = pd.Timestamp(text)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _lookup(payload: dict[str, Any], dotted: str) -> Any:
    node: Any = payload
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return _MISSING
        node = node[part]
    return node


def _wrong_type(value: Any, kind: type) -> bool:
    """``bool`` is an ``int`` in Python; a row count spelled ``true`` is not."""
    if kind is int:
        return not isinstance(value, int) or isinstance(value, bool)
    return not isinstance(value, kind)


def verify_snapshot(manifest_path: str | Path = DEFAULT_MANIFEST) -> list[CheckResult]:
    """Check a research snapshot end to end, raising on the first failure.

    Paths inside the manifest are repository-relative, so the tree they are
    resolved against is the one holding the manifest: two directories above
    ``data/research/<manifest>``. A copy of the snapshot therefore verifies as
    itself rather than silently against the committed files.
    """
    manifest_path = Path(manifest_path).resolve()
    results: list[CheckResult] = []

    def held(name: str, detail: str) -> None:
        results.append(CheckResult(name, detail))

    if not manifest_path.is_file():
        raise SnapshotVerificationError("manifest_readable", f"no manifest at {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise SnapshotVerificationError(
            "manifest_readable", f"{manifest_path} is not readable JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise SnapshotVerificationError(
            "manifest_readable", f"{manifest_path} is not a JSON object"
        )
    held("manifest_readable", str(manifest_path))

    schema = payload.get("snapshot_schema")
    if schema != SNAPSHOT_SCHEMA:
        raise SnapshotVerificationError(
            "snapshot_schema", f"expected {SNAPSHOT_SCHEMA!r}, manifest declares {schema!r}"
        )
    held("snapshot_schema", SNAPSHOT_SCHEMA)

    for dotted, kind in REQUIRED_KEYS:
        value = _lookup(payload, dotted)
        if value is _MISSING:
            raise SnapshotVerificationError(
                "manifest_keys", f"missing required key {dotted!r}"
            )
        if _wrong_type(value, kind):
            raise SnapshotVerificationError(
                "manifest_keys",
                f"{dotted} must be {kind.__name__}, got {type(value).__name__}",
            )
    held("manifest_keys", f"{len(REQUIRED_KEYS)} required keys present and well typed")

    declared = payload["contract"]
    try:
        contract = load_contract(declared["contract_id"])
    except ResearchContractError as exc:
        raise SnapshotVerificationError("contract_id", str(exc)) from exc
    held("contract_id", f"{contract.contract_id} committed at {contract.source}")

    if declared["contract_hash"] != contract.contract_hash:
        raise SnapshotVerificationError(
            "contract_hash",
            f"manifest records {declared['contract_hash']}, {contract.source} hashes to "
            f"{contract.contract_hash}",
        )
    held("contract_hash", contract.contract_hash)

    sealed = contract.sealed_test_start
    try:
        declared_seal = _utc(declared["sealed_test_start"])
    except ValueError as exc:
        raise SnapshotVerificationError(
            "contract_sealed_test_start",
            f"sealed_test_start {declared['sealed_test_start']!r} is not a timestamp: {exc}",
        ) from exc
    if declared_seal != sealed:
        raise SnapshotVerificationError(
            "contract_sealed_test_start",
            f"manifest seals at {declared_seal.isoformat()}, contract "
            f"{contract.contract_id} seals at {sealed.isoformat()}",
        )
    held("contract_sealed_test_start", sealed.isoformat())

    if payload["contains_styx"] is not False:
        raise SnapshotVerificationError(
            "contains_styx",
            f"must be exactly false, manifest says {payload['contains_styx']!r}",
        )
    held("contains_styx", "false")

    safety = payload["safety_checks"]
    if safety["styx_rows_exported"] != 0:
        raise SnapshotVerificationError(
            "styx_rows_exported",
            f"must be exactly 0, manifest says {safety['styx_rows_exported']!r}",
        )
    held("styx_rows_exported", "0")

    flags = sorted(name for name in safety if name != "styx_rows_exported")
    for name in flags:
        if safety[name] is not True:
            raise SnapshotVerificationError(
                "safety_check_flags",
                f"safety_checks.{name} must be exactly true, manifest says {safety[name]!r}",
            )
    held("safety_check_flags", f"{len(flags)} flags true")

    root = manifest_path.parents[2]
    raw_block = payload["raw_pre_styx"]
    processed_block = payload["processed_outer_coverage"]
    raw_path = root / raw_block["path"]
    processed_path = root / processed_block["path"]
    metadata_path = root / processed_block["metadata_path"]
    for key, path in (
        ("raw_pre_styx.path", raw_path),
        ("processed_outer_coverage.path", processed_path),
        ("processed_outer_coverage.metadata_path", metadata_path),
    ):
        if not path.is_file():
            raise SnapshotVerificationError("files_present", f"{key} names {path}, not a file")
    held("files_present", f"3 referenced files under {root}")

    for check, path, expected in (
        ("raw_sha256", raw_path, raw_block["sha256"]),
        ("processed_sha256", processed_path, processed_block["sha256"]),
        ("metadata_sha256", metadata_path, processed_block["metadata_sha256"]),
    ):
        actual = _sha256(path)
        if actual != expected:
            raise SnapshotVerificationError(
                check, f"{path.name} hashes to {actual}, manifest records {expected}"
            )
        held(check, expected)

    # The seal comes before every check below it. A snapshot holding a sealed
    # row must be rejected for holding one, not for whatever digest that row
    # happened to disturb — and the rejection counts rows without naming them.
    raw = load_raw_ohlcv(raw_path)
    raw_dates = pd.DatetimeIndex(raw.index)
    frame, metadata = load_dataset(processed_path)
    processed_dates = pd.DatetimeIndex(pd.to_datetime(frame["date"], utc=True))
    for check, dates in (
        ("raw_before_styx", raw_dates),
        ("processed_before_styx", processed_dates),
    ):
        breaches = int((dates >= sealed).sum())
        if breaches:
            raise SnapshotVerificationError(
                check,
                f"{breaches} of {len(dates)} rows are at or after the sealed instant "
                f"{sealed.isoformat()}; the rows themselves are withheld",
            )
        held(check, f"{len(dates)} rows strictly before {sealed.isoformat()}")

    for check, declared_rows, actual_rows in (
        ("raw_rows", raw_block["rows"], len(raw_dates)),
        ("processed_rows", processed_block["rows"], len(frame)),
    ):
        if declared_rows != actual_rows:
            raise SnapshotVerificationError(
                check,
                f"manifest records {declared_rows} rows, the Parquet holds {actual_rows}",
            )
        held(check, str(actual_rows))

    rows = processed_block["rows"]
    if list(processed_block["row_range"]) != [0, rows]:
        raise SnapshotVerificationError(
            "processed_row_range",
            f"row_range {processed_block['row_range']} does not describe the {rows} rows "
            "exported, which are [0, rows) of the canonical dataset",
        )
    held("processed_row_range", f"[0, {rows})")

    for check, block, dates in (
        ("raw_span", raw_block, raw_dates),
        ("processed_span", processed_block, processed_dates),
    ):
        try:
            start, end = _utc(block["start"]), _utc(block["end"])
        except ValueError as exc:
            raise SnapshotVerificationError(
                check,
                f"start/end {block['start']!r}..{block['end']!r} are not timestamps: {exc}",
            ) from exc
        if start != dates[0] or end != dates[-1]:
            raise SnapshotVerificationError(
                check,
                f"manifest spans {start.isoformat()}..{end.isoformat()}, the data spans "
                f"{dates[0].isoformat()}..{dates[-1].isoformat()}",
            )
        held(check, f"{dates[0].isoformat()} .. {dates[-1].isoformat()}")

    fingerprint = fingerprint_raw_input(
        raw,
        exchange=metadata.exchange,
        pair=metadata.pair,
        timeframe=metadata.timeframe,
        visible_through=raw_dates[-1],
    )
    if fingerprint.raw_input_hash != raw_block["semantic_hash"]:
        raise SnapshotVerificationError(
            "raw_semantic_hash",
            f"recomputed {fingerprint.raw_input_hash}, manifest records "
            f"{raw_block['semantic_hash']}",
        )
    held("raw_semantic_hash", fingerprint.raw_input_hash)

    prefix_hash = research_input_digest(
        {name: frame[name] for name in frame.columns},
        feature_names=metadata.feature_names,
        exchange=metadata.exchange,
        pair=metadata.pair,
        timeframe=metadata.timeframe,
        feature_spec=metadata.feature_spec,
        target_spec=metadata.target_spec,
        rows=len(frame),
    )
    if prefix_hash != processed_block["semantic_prefix_hash"]:
        raise SnapshotVerificationError(
            "processed_semantic_prefix_hash",
            f"recomputed {prefix_hash}, manifest records "
            f"{processed_block['semantic_prefix_hash']}",
        )
    held("processed_semantic_prefix_hash", prefix_hash)

    reference = payload["canonical_reference"]
    outer_end = reference["max_outer_end_row"]
    research_rows = reference["research_rows"]
    if outer_end > research_rows:
        raise SnapshotVerificationError(
            "outer_coverage_rows",
            f"max_outer_end_row {outer_end} is past research_rows {research_rows}, so the "
            "frozen outer folds would reach sealed rows",
        )
    if len(frame) < outer_end:
        raise SnapshotVerificationError(
            "outer_coverage_rows",
            f"the processed snapshot holds {len(frame)} rows, too few to cover outer folds "
            f"ending at row {outer_end}",
        )
    held("outer_coverage_rows", f"{len(frame)} rows cover row {outer_end} of {research_rows}")

    return results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        results = verify_snapshot(args.manifest)
    except SnapshotVerificationError as exc:
        print(f"research snapshot REJECTED - {exc}")
        return 1
    print(f"research snapshot verified: {args.manifest}")
    for result in results:
        print(f"  PASS  {result.name:<30} {result.detail}")
    print(f"{len(results)} checks passed; no row at or after the sealed instant.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
