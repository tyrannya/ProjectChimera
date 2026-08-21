"""Refuse a P3 trade source that cannot be trusted, before anyone fits on it.

``tools.export_trade_snapshot`` proves at acquisition time that the hourly
aggregate under ``data/research/`` holds no sealed hour and covers every candle
the research spine has. A fresh clone inherits none of that: it has a Parquet
file and a manifest asserting things about it, and the two drift apart silently
— a rebuilt table, a hand-edited manifest, a partial checkout, a merge that
resurrected an older file.

So every claim is recomputed rather than read, and the first disagreement fails
closed. The digests come from the exporter's own helpers
(:func:`nn.data_fingerprint.fingerprint_trade_aggregates`,
:func:`tools.export_trade_snapshot._sha256`) and the table's internal consistency
from :func:`nn.trade_aggregates.check_table`, because a second implementation of
a hash or of a validity rule is a second definition of it, and the two drift
apart at the first change to either.

**The seal is checked against the data, not against the manifest's word for it,
and before anything else touches the table.** ``contains_styx`` and
``safety_checks`` are claims made by whoever wrote the manifest, so they are held
to exactly the values the exporter writes and then the hours in the Parquet are
compared against the contract's sealed instant directly. A snapshot carrying a
sealed hour is rejected as a seal breach rather than as an incidental hash
mismatch three checks later.

**A rejection names the breach, never its content.** The seal check reports how
many rows are at or after the instant and nothing else.

**Two identities, both bound.** A trade source is only research evidence in
combination with the candles it is aligned to, so this checks the aggregate's own
semantic hash *and* that the OHLCV snapshot the manifest was exported against is
the one committed here — and then that every spine hour actually has a row. A
trade table that is internally perfect and covers a different five years is the
failure this last check exists for.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from nn.data_fingerprint import fingerprint_trade_aggregates
from nn.research_contract import ResearchContractError, load_contract
from nn.trade_aggregates import (
    AGGREGATE_COLUMNS,
    StyxBreach,
    TradeAggregationError,
    aggregation_spec_hash,
    check_table,
    coverage_report,
)
from tools.export_trade_snapshot import (
    MANIFEST_NAME,
    MARKET_TYPE,
    SOURCE_TYPE,
    SYMBOL,
    TRADE_SNAPSHOT_SCHEMA,
    VENUE,
    _sha256,
)

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
    ("styx_instant", str),
    ("source", dict),
    ("source.venue", str),
    ("source.market_type", str),
    ("source.symbol", str),
    ("source.source_type", str),
    ("source.identity_column", str),
    ("source.ordering_rule", str),
    ("source.duplicate_rule", str),
    ("source.aggressor_semantics", str),
    ("source.uses_buyer_maker", bool),
    ("source.known_limitations", list),
    ("acquisition", dict),
    ("acquisition.archive_count", int),
    ("acquisition.raw_rows_total", int),
    ("acquisition.archives", list),
    ("aggregation", dict),
    ("aggregation.aggregation_spec_hash", str),
    ("trade_stream", dict),
    ("trade_stream.trades", int),
    ("trade_stream.duplicate_ids", int),
    ("trade_stream.timestamp_regressions", int),
    ("trade_stream.invalid_rows", int),
    ("trade_stream.nonpositive_price", int),
    ("trade_stream.nonpositive_qty", int),
    ("trade_stream.strictly_sorted", bool),
    ("trade_stream.first_timestamp", str),
    ("trade_stream.last_timestamp", str),
    ("aggregate", dict),
    ("aggregate.path", str),
    ("aggregate.rows", int),
    ("aggregate.start", str),
    ("aggregate.end", str),
    ("aggregate.semantic_hash", str),
    ("aggregate.sha256", str),
    ("aggregate.columns", list),
    ("coverage", dict),
    ("coverage.hours_present", int),
    ("coverage.hours_missing", int),
    ("alignment", dict),
    ("alignment.ohlcv_manifest", str),
    ("alignment.ohlcv_processed_semantic_prefix_hash", str),
    ("alignment.spine_rows", int),
    ("alignment.spine_hours_covered", int),
    ("alignment.spine_hours_missing", int),
    ("normalisation", dict),
    ("normalisation.canonical_timezone", str),
    ("normalisation.epoch_representation", str),
    ("safety_checks", dict),
    ("safety_checks.styx_rows_exported", int),
)

_MISSING = object()


class TradeSnapshotVerificationError(ValueError):
    """A trade snapshot failed one check, named by :attr:`check`.

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


def verify_trade_snapshot(
    manifest_path: str | Path = DEFAULT_MANIFEST,
) -> list[CheckResult]:
    """Check a trade snapshot end to end, raising on the first failure.

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
        raise TradeSnapshotVerificationError(
            "manifest_readable", f"no manifest at {manifest_path}"
        )
    try:
        payload = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise TradeSnapshotVerificationError(
            "manifest_readable", f"{manifest_path} is not readable JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise TradeSnapshotVerificationError(
            "manifest_readable", f"{manifest_path} is not a JSON object"
        )
    held("manifest_readable", str(manifest_path))

    schema = payload.get("snapshot_schema")
    if schema != TRADE_SNAPSHOT_SCHEMA:
        raise TradeSnapshotVerificationError(
            "snapshot_schema",
            f"expected {TRADE_SNAPSHOT_SCHEMA!r}, manifest declares {schema!r}",
        )
    held("snapshot_schema", TRADE_SNAPSHOT_SCHEMA)

    for dotted, kind in REQUIRED_KEYS:
        value = _lookup(payload, dotted)
        if value is _MISSING:
            raise TradeSnapshotVerificationError(
                "manifest_keys", f"missing required key {dotted!r}"
            )
        if _wrong_type(value, kind):
            raise TradeSnapshotVerificationError(
                "manifest_keys",
                f"{dotted} must be {kind.__name__}, got {type(value).__name__}",
            )
    held("manifest_keys", f"{len(REQUIRED_KEYS)} required keys present and well typed")

    declared = payload["contract"]
    try:
        contract = load_contract(declared["contract_id"])
    except ResearchContractError as exc:
        raise TradeSnapshotVerificationError("contract_id", str(exc)) from exc
    held("contract_id", f"{contract.contract_id} committed at {contract.source}")

    if declared["contract_hash"] != contract.contract_hash:
        raise TradeSnapshotVerificationError(
            "contract_hash",
            f"manifest records {declared['contract_hash']}, {contract.source} hashes to "
            f"{contract.contract_hash}",
        )
    held("contract_hash", contract.contract_hash)

    sealed = contract.sealed_test_start
    for check, text in (
        ("contract_sealed_test_start", declared["sealed_test_start"]),
        ("styx_instant", payload["styx_instant"]),
    ):
        try:
            stamp = _utc(text)
        except ValueError as exc:
            raise TradeSnapshotVerificationError(
                check, f"{text!r} is not a timestamp: {exc}"
            ) from exc
        if stamp != sealed:
            raise TradeSnapshotVerificationError(
                check,
                f"manifest seals at {stamp.isoformat()}, contract "
                f"{contract.contract_id} seals at {sealed.isoformat()}",
            )
        held(check, sealed.isoformat())

    if payload["contains_styx"] is not False:
        raise TradeSnapshotVerificationError(
            "contains_styx",
            f"must be exactly false, manifest says {payload['contains_styx']!r}",
        )
    held("contains_styx", "false")

    safety = payload["safety_checks"]
    if safety["styx_rows_exported"] != 0:
        raise TradeSnapshotVerificationError(
            "styx_rows_exported",
            f"must be exactly 0, manifest says {safety['styx_rows_exported']!r}",
        )
    held("styx_rows_exported", "0")

    flags = sorted(name for name in safety if name != "styx_rows_exported")
    if not flags:
        raise TradeSnapshotVerificationError(
            "safety_check_flags", "safety_checks declares no flag; there is nothing to hold"
        )
    for name in flags:
        if safety[name] is not True:
            raise TradeSnapshotVerificationError(
                "safety_check_flags",
                f"safety_checks.{name} must be exactly true, manifest says {safety[name]!r}",
            )
    held("safety_check_flags", f"{len(flags)} flags true")

    source = payload["source"]
    expected_source = {
        "venue": VENUE,
        "market_type": MARKET_TYPE,
        "symbol": SYMBOL,
        "source_type": SOURCE_TYPE,
    }
    for key, want in expected_source.items():
        if str(source[key]).strip().lower() != want.lower():
            raise TradeSnapshotVerificationError(
                "source_identity",
                f"source.{key} is {source[key]!r}; this checkpoint's canonical source is "
                f"{want!r}, and a snapshot of another venue, market or representation is "
                "not the same research input",
            )
    if source["identity_column"] != "agg_trade_id":
        raise TradeSnapshotVerificationError(
            "source_identity",
            f"source.identity_column is {source['identity_column']!r}; the duplicate rule "
            "is defined over agg_trade_id and means nothing over another key",
        )
    held("source_identity", f"{VENUE} {MARKET_TYPE} {SYMBOL} {SOURCE_TYPE}")

    declared_aggregation = payload["aggregation"]["aggregation_spec_hash"]
    if declared_aggregation != aggregation_spec_hash():
        raise TradeSnapshotVerificationError(
            "aggregation_spec_hash",
            f"the snapshot was aggregated under {declared_aggregation} "
            f"but this checkout's nn.trade_aggregates hashes to {aggregation_spec_hash()}. "
            "The bucket, the aggressor rule, the size grid or the column set has changed, "
            "so the committed table is not what this code would produce",
        )
    held("aggregation_spec_hash", aggregation_spec_hash())

    root = manifest_path.parents[2]
    block = payload["aggregate"]
    aggregate_path = root / block["path"]
    if not aggregate_path.is_file():
        raise TradeSnapshotVerificationError(
            "files_present", f"aggregate.path names {aggregate_path}, not a file"
        )
    held("files_present", f"1 referenced file under {root}")

    actual = _sha256(aggregate_path)
    if actual != block["sha256"]:
        raise TradeSnapshotVerificationError(
            "aggregate_sha256",
            f"{aggregate_path.name} hashes to {actual}, manifest records {block['sha256']}",
        )
    held("aggregate_sha256", actual)

    frame = pd.read_parquet(aggregate_path)
    missing = [name for name in AGGREGATE_COLUMNS if name not in frame.columns]
    if missing:
        raise TradeSnapshotVerificationError(
            "aggregate_columns", f"the table is missing column(s) {missing}"
        )
    held("aggregate_columns", f"{len(AGGREGATE_COLUMNS)} columns present")

    # The seal comes before every check below it. A snapshot holding a sealed
    # hour must be rejected for holding one, not for whatever digest that hour
    # happened to disturb — and the rejection counts rows without naming them.
    dates = pd.DatetimeIndex(pd.to_datetime(frame["date"], utc=True))
    breaches = int((dates >= sealed).sum())
    if breaches:
        raise TradeSnapshotVerificationError(
            "aggregate_before_styx",
            f"{breaches} of {len(dates)} hours are at or after the sealed instant "
            f"{sealed.isoformat()}; the rows themselves are withheld",
        )
    held("aggregate_before_styx", f"{len(dates)} hours strictly before {sealed.isoformat()}")

    try:
        check_table(frame, styx=sealed)
    except StyxBreach as exc:  # pragma: no cover - the check above already fired
        raise TradeSnapshotVerificationError("aggregate_before_styx", str(exc)) from exc
    except TradeAggregationError as exc:
        raise TradeSnapshotVerificationError("table_integrity", str(exc)) from exc
    held(
        "table_integrity",
        "hours strictly increasing, on the hour, and internally consistent",
    )

    if block["rows"] != len(frame):
        raise TradeSnapshotVerificationError(
            "aggregate_rows",
            f"manifest records {block['rows']} rows, the Parquet holds {len(frame)}",
        )
    held("aggregate_rows", str(len(frame)))

    try:
        start, end = _utc(block["start"]), _utc(block["end"])
    except ValueError as exc:
        raise TradeSnapshotVerificationError(
            "aggregate_span",
            f"start/end {block['start']!r}..{block['end']!r} are not timestamps: {exc}",
        ) from exc
    if start != dates[0] or end != dates[-1]:
        raise TradeSnapshotVerificationError(
            "aggregate_span",
            f"manifest spans {start.isoformat()}..{end.isoformat()}, the table spans "
            f"{dates[0].isoformat()}..{dates[-1].isoformat()}",
        )
    held("aggregate_span", f"{dates[0].isoformat()} .. {dates[-1].isoformat()}")

    if list(block["columns"]) != list(AGGREGATE_COLUMNS):
        raise TradeSnapshotVerificationError(
            "aggregate_column_order",
            "the manifest's column list is not this checkout's aggregate schema, in order",
        )
    held("aggregate_column_order", f"{len(AGGREGATE_COLUMNS)} columns in schema order")

    fingerprint = fingerprint_trade_aggregates(
        frame,
        exchange=source["venue"],
        symbol=source["symbol"],
        market_type=source["market_type"],
        source_type=source["source_type"],
        aggregation_spec_hash=aggregation_spec_hash(),
    )
    if fingerprint.aggregate_hash != block["semantic_hash"]:
        raise TradeSnapshotVerificationError(
            "aggregate_semantic_hash",
            f"recomputed {fingerprint.aggregate_hash}, manifest records "
            f"{block['semantic_hash']}",
        )
    held("aggregate_semantic_hash", fingerprint.aggregate_hash)

    stream = payload["trade_stream"]
    for name in ("duplicate_ids", "timestamp_regressions", "invalid_rows"):
        if stream[name] != 0:
            raise TradeSnapshotVerificationError(
                "stream_integrity",
                f"trade_stream.{name} is {stream[name]}, and a source with any of those "
                "is not the single ordered sequence the duplicate rule is defined over",
            )
    if stream["strictly_sorted"] is not True:
        raise TradeSnapshotVerificationError(
            "stream_integrity", "trade_stream.strictly_sorted must be exactly true"
        )
    declared_trades = int(frame["n_trades"].sum())
    if stream["trades"] != declared_trades:
        raise TradeSnapshotVerificationError(
            "stream_trade_count",
            f"trade_stream.trades is {stream['trades']} but the hourly counts sum to "
            f"{declared_trades}; the manifest and the table disagree about how many "
            "trades were aggregated",
        )
    held("stream_trade_count", f"{declared_trades} trades across {len(frame)} hours")

    buy_total = int(frame["buy_trades"].sum())
    if stream.get("buy_trades") != buy_total:
        raise TradeSnapshotVerificationError(
            "stream_trade_count",
            f"trade_stream.buy_trades is {stream.get('buy_trades')!r} but the hourly "
            f"taker-buy counts sum to {buy_total}",
        )
    held("stream_aggressor_count", f"{buy_total} taker buys")

    archives = payload["acquisition"]["archives"]
    if not archives:
        raise TradeSnapshotVerificationError(
            "archives_declared",
            "acquisition.archives is empty, so the committed table names no source it "
            "was derived from and the derivation cannot be audited",
        )
    if payload["acquisition"]["archive_count"] != len(archives):
        raise TradeSnapshotVerificationError(
            "archives_declared",
            f"archive_count is {payload['acquisition']['archive_count']} but "
            f"{len(archives)} archives are listed",
        )
    names = [a.get("name") for a in archives]
    if len(set(names)) != len(names):
        raise TradeSnapshotVerificationError(
            "archives_declared", "the same archive is listed twice"
        )
    for entry in archives:
        digest = str(entry.get("sha256", ""))
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise TradeSnapshotVerificationError(
                "archives_declared",
                f"archive {entry.get('name')!r} records sha256 {digest!r}, which is not a "
                "sha-256 digest; the raw bytes this table was derived from are unidentified",
            )
    held("archives_declared", f"{len(archives)} archives with byte digests")

    recomputed = coverage_report(frame)
    for key in ("first_hour", "last_hour", "hours_in_span", "hours_present", "hours_missing"):
        if payload["coverage"].get(key) != recomputed[key]:
            raise TradeSnapshotVerificationError(
                "coverage_claims",
                f"coverage.{key} is {payload['coverage'].get(key)!r}, recomputed "
                f"{recomputed[key]!r}",
            )
    held(
        "coverage_claims",
        f"{recomputed['hours_present']} of {recomputed['hours_in_span']} hours present, "
        f"{recomputed['hours_missing']} missing",
    )

    alignment = payload["alignment"]
    ohlcv_manifest = root / alignment["ohlcv_manifest"]
    if not ohlcv_manifest.is_file():
        raise TradeSnapshotVerificationError(
            "ohlcv_binding",
            f"alignment.ohlcv_manifest names {ohlcv_manifest}, not a file; the candles "
            "this source claims to align to are not present",
        )
    ohlcv = json.loads(ohlcv_manifest.read_text())
    if (
        ohlcv["processed_outer_coverage"]["semantic_prefix_hash"]
        != alignment["ohlcv_processed_semantic_prefix_hash"]
    ):
        raise TradeSnapshotVerificationError(
            "ohlcv_binding",
            "this trade source was exported against an OHLCV snapshot whose processed "
            "semantic hash was "
            f"{alignment['ohlcv_processed_semantic_prefix_hash']}, but the committed "
            f"snapshot hashes to {ohlcv['processed_outer_coverage']['semantic_prefix_hash']}. "
            "Two research inputs that were never aligned to each other must not be "
            "presented as one",
        )
    if ohlcv["contract"]["contract_hash"] != contract.contract_hash:
        raise TradeSnapshotVerificationError(
            "ohlcv_binding",
            "the OHLCV snapshot names a different research contract than this trade source",
        )
    held("ohlcv_binding", alignment["ohlcv_processed_semantic_prefix_hash"])

    spine = pd.read_parquet(root / ohlcv["processed_outer_coverage"]["path"], columns=["date"])
    spine_hours = pd.DatetimeIndex(pd.to_datetime(spine["date"], utc=True))
    absent = spine_hours.difference(dates)
    if len(absent):
        raise TradeSnapshotVerificationError(
            "spine_coverage",
            f"{len(absent)} of {len(spine_hours)} spine hours have no trade-aggregate "
            "row, so the microstructure row for those candles would have to be invented",
        )
    if alignment["spine_rows"] != len(spine_hours):
        raise TradeSnapshotVerificationError(
            "spine_coverage",
            f"alignment.spine_rows is {alignment['spine_rows']}, the committed spine "
            f"holds {len(spine_hours)}",
        )
    if alignment["spine_hours_missing"] != 0 or alignment["spine_hours_covered"] != len(
        spine_hours
    ):
        raise TradeSnapshotVerificationError(
            "spine_coverage",
            "alignment claims a coverage this snapshot does not have: "
            f"covered={alignment['spine_hours_covered']}, "
            f"missing={alignment['spine_hours_missing']}, spine={len(spine_hours)}",
        )
    held("spine_coverage", f"all {len(spine_hours)} spine hours have a trade row")

    return results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        results = verify_trade_snapshot(args.manifest)
    except TradeSnapshotVerificationError as exc:
        print(f"trade snapshot REJECTED - {exc}")
        return 1
    print(f"trade snapshot verified: {args.manifest}")
    for result in results:
        print(f"  PASS  {result.name:<30} {result.detail}")
    print(f"{len(results)} checks passed; no hour at or after the sealed instant.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
