"""Recompute every claim P4's derivatives snapshot manifest makes about itself.

    python -m tools.verify_derivatives_snapshot

An independent reader, not a second look at the exporter's own bookkeeping. It
opens the Parquet, re-derives the numbers, and compares them with what the
manifest says — schema, contract identity, both boundaries, the file digest, the
source-spec hash, the semantic fingerprint, the table's internal consistency, the
staleness bounds, the coverage claims, the missing-day accounting, the
exact-duplicate accounting of the metrics archives, the observation-validity
accounting beside it, and the binding to the OHLCV snapshot the derivatives are
joined to.

**Nothing here fits anything, and nothing here reads a model output.** It is a
statement about an input.

**It runs inside the only function that loads the source.**
:func:`nn.p2b.load_derivatives_snapshot` calls :func:`verify_derivatives_snapshot`
before it returns a row, for the reason ``nn.p2b.load_trade_snapshot`` does: a
``make`` target is a convention and ``python -m nn.p2b`` bypasses it. A corrupt
derivatives source must not reach a model fit by any path, and the only way to
guarantee that is for the load to fail.

**A failure names the check and never the data.** The boundary checks report how
many hours are at or after the sealed instant, or at or after P4-HOLD's first
instant, and nothing about them — so a snapshot that breaches either is rejected
without printing what it breached with.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nn.data_fingerprint import fingerprint_derivatives_hourly
from nn.derivatives_sources import (
    DERIVATIVES_COLUMNS,
    FUNDING,
    HOUR_NS,
    OPEN_INTEREST,
    PERPETUAL,
    staleness_bound_ns,
)
from nn.p4_holdout import assert_stage_one_instants, holdout_first_instant
from nn.p4_preregistration import preregistration_hash
from nn.research_contract import load_contract
from tools.export_derivatives_snapshot import (
    DERIVATIVES_SNAPSHOT_SCHEMA,
    MANIFEST_NAME,
    check_hourly_table,
    coverage_report,
    source_spec_hash,
)

DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "data" / "research" / MANIFEST_NAME

#: How far a recomputed available-fraction may sit from the manifest's copy.
#: Zero: both are computed from the same rows by the same function, so any
#: difference at all means the manifest is describing another table.
COVERAGE_TOLERANCE = 0.0


class DerivativesSnapshotVerificationError(ValueError):
    """The committed derivatives snapshot does not match its own manifest."""


@dataclass(frozen=True)
class CheckResult:
    name: str
    detail: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DerivativesSnapshotVerificationError(message)


def cross_source_consistency(
    frame: pd.DataFrame, raw_candles: pd.DataFrame, *, rtol: float = 0.05
) -> dict[str, Any]:
    """§3.5's value-level check: the perpetual close against the spot close.

    Not an equality — they are different instruments and the basis between them
    is the point of the feature. What this catches is the failure mode that would
    make the basis meaningless: a perpetual series joined to the wrong hours, a
    unit or scale error, or a column read from the wrong position, all of which
    move the two apart by far more than a carry premium ever does.

    ``rtol`` is 5%, and it is a bound on a *cost of carry*, not a fitted number.
    ``drv_basis`` is clipped at ±2% by ``docs/p4_preregistration.md`` §5, so a
    tolerance a little above the clip is the widest reading under which the
    feature still means what it says; anything further apart is a join or a unit,
    not a market. A recorded justification rather than a knob: the check reports
    the worst observed deviation so a reader can see how much room was left.
    """
    candles = raw_candles.copy()
    candles["date"] = pd.to_datetime(candles["date"], utc=True)
    spot = pd.Series(
        candles["close"].to_numpy(dtype=np.float64),
        index=pd.DatetimeIndex(candles["date"]),
    )
    hours = pd.DatetimeIndex(pd.to_datetime(frame["date"], utc=True))
    aligned = spot.reindex(hours).to_numpy(dtype=np.float64)
    available = frame["perp_available"].to_numpy(dtype=np.int64) == 1
    perp = frame["perp_close"].to_numpy(dtype=np.float64)
    comparable = available & np.isfinite(aligned) & (aligned > 0)
    if not comparable.any():
        raise DerivativesSnapshotVerificationError(
            "no hour has both a perpetual close and a spot candle, so the basis cannot "
            "be cross-checked against the source it divides by"
        )
    deviation = np.abs(perp[comparable] / aligned[comparable] - 1.0)
    worst = float(deviation.max())
    offenders = int((deviation > rtol).sum())
    _require(
        offenders == 0,
        f"{offenders} hour(s) have a perpetual close more than {rtol:.0%} from the spot "
        f"close (worst {worst:.4f}). docs/p4_preregistration.md §5 clips drv_basis at "
        "±2%, so a gap this wide is a join, a unit or a column position — not a cost of "
        "carry.",
    )
    return {
        "hours_compared": int(comparable.sum()),
        "hours_uncomparable": int((~comparable).sum()),
        "worst_relative_deviation": round(worst, 6),
        "tolerance": rtol,
        "justification": (
            "a bound on a cost of carry: drv_basis is clipped at ±2%, so anything "
            "beyond 5% is a join or a unit rather than a market"
        ),
    }


def verify_derivatives_snapshot(manifest_path: Path = DEFAULT_MANIFEST) -> list[CheckResult]:
    """Recompute every claim, or raise naming the first one that failed."""
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise DerivativesSnapshotVerificationError(f"no manifest at {manifest_path}")
    payload = json.loads(manifest_path.read_text())
    root = manifest_path.resolve().parent.parent.parent
    checks: list[CheckResult] = []

    _require(
        payload.get("snapshot_schema") == DERIVATIVES_SNAPSHOT_SCHEMA,
        f"{manifest_path} declares schema {payload.get('snapshot_schema')!r}, not "
        f"{DERIVATIVES_SNAPSHOT_SCHEMA!r}",
    )
    checks.append(CheckResult("schema", DERIVATIVES_SNAPSHOT_SCHEMA))

    _require(
        payload.get("preregistration_hash") == preregistration_hash(),
        f"the snapshot was exported under preregistration "
        f"{payload.get('preregistration_hash')!r} and this checkout is "
        f"{preregistration_hash()!r}. A source acquired under one design is not the "
        "source another design preregistered.",
    )
    checks.append(CheckResult("preregistration", preregistration_hash()[:16]))

    contract = load_contract(payload["contract"]["contract_id"])
    _require(
        payload["contract"]["contract_hash"] == contract.contract_hash,
        f"the snapshot declares contract hash {payload['contract']['contract_hash']} "
        f"and {contract.contract_id} hashes to {contract.contract_hash}",
    )
    checks.append(
        CheckResult("contract", f"{contract.contract_id} {contract.contract_hash[:12]}")
    )

    source_path = root / payload["hourly"]["path"]
    _require(source_path.is_file(), f"the manifest names {source_path}, which does not exist")
    digest = _sha256(source_path)
    _require(
        digest == payload["hourly"]["sha256"],
        f"{source_path} hashes to {digest} and the manifest says "
        f"{payload['hourly']['sha256']}",
    )
    checks.append(CheckResult("file digest", digest[:16]))

    _require(
        payload["hourly"].get("source_spec_hash") == source_spec_hash(),
        f"the snapshot was written under source spec "
        f"{payload['hourly'].get('source_spec_hash')!r} and this checkout computes "
        f"{source_spec_hash()!r}. Two staleness rules are two sources.",
    )
    checks.append(CheckResult("source spec", source_spec_hash()[:16]))

    frame = pd.read_parquet(source_path)
    _require(
        list(frame.columns) == list(DERIVATIVES_COLUMNS),
        f"the source holds columns {list(frame.columns)}, not {list(DERIVATIVES_COLUMNS)}",
    )
    checks.append(CheckResult("columns", f"{len(frame.columns)} in schema order"))

    # The exporter's own structural checks, re-run by the reader rather than
    # trusted from the writer: contiguous hours, 0/1 availability flags, ages
    # inside their preregistered bounds, finite values, positive prices.
    check_hourly_table(frame)
    checks.append(CheckResult("table structure", f"{len(frame)} contiguous hours"))

    _require(
        int(payload["hourly"]["rows"]) == len(frame),
        f"the manifest says {payload['hourly']['rows']} rows and the file holds "
        f"{len(frame)}",
    )
    dates = pd.DatetimeIndex(pd.to_datetime(frame["date"], utc=True))
    _require(
        dates[0].isoformat() == payload["hourly"]["start"]
        and dates[-1].isoformat() == payload["hourly"]["end"],
        "the manifest's span does not match the file's first and last hour",
    )
    checks.append(CheckResult("span", f"{dates[0].isoformat()}..{dates[-1].isoformat()}"))

    fingerprint = fingerprint_derivatives_hourly(frame, source_spec_hash=source_spec_hash())
    _require(
        fingerprint.source_hash == payload["hourly"]["semantic_hash"],
        f"the source's semantic hash is {fingerprint.source_hash} and the manifest says "
        f"{payload['hourly']['semantic_hash']}",
    )
    checks.append(CheckResult("semantic hash", fingerprint.short_hash))

    sealed = contract.sealed_test_start
    breaching = int((dates >= sealed).sum())
    _require(
        breaching == 0,
        f"{breaching} hour(s) are at or after the sealed instant {sealed.isoformat()}",
    )
    checks.append(CheckResult("styx", f"0 of {len(dates)} hours at or after the seal"))

    boundary = pd.Timestamp(holdout_first_instant()).tz_convert("UTC")
    assert_stage_one_instants(dates, what=f"the derivatives source at {source_path}")
    _require(
        payload.get("contains_p4_hold") is False
        and payload["safety_checks"]["p4_hold_rows_exported"] == 0,
        "the manifest does not declare that it holds no P4-HOLD hour",
    )
    checks.append(CheckResult("p4-hold", f"0 hours at or after {boundary.isoformat()}"))

    recomputed = coverage_report(frame, payload["coverage"]["missing_metrics_days"])
    for field in (FUNDING, OPEN_INTEREST, PERPETUAL):
        declared = payload["coverage"][field]
        actual = recomputed[field]
        for key in (
            "hours_available",
            "hours_unavailable",
            "max_contiguous_unavailable_hours",
        ):
            _require(
                declared[key] == actual[key],
                f"the manifest says {field}.{key} is {declared[key]} and the file gives "
                f"{actual[key]}",
            )
        _require(
            abs(float(declared["available_fraction"]) - float(actual["available_fraction"]))
            <= COVERAGE_TOLERANCE,
            f"the manifest says {field} is available on "
            f"{declared['available_fraction']} of hours and the file gives "
            f"{actual['available_fraction']}",
        )
        bound = staleness_bound_ns(field) // HOUR_NS
        _require(
            int(declared["staleness_bound_hours"]) == bound,
            f"the manifest records a {declared['staleness_bound_hours']}h staleness "
            f"bound for {field} and the preregistration fixes {bound}h",
        )
    checks.append(
        CheckResult(
            "coverage",
            ", ".join(
                f"{field}={recomputed[field]['available_fraction']}"
                for field in (FUNDING, OPEN_INTEREST, PERPETUAL)
            ),
        )
    )

    declared_missing = payload["acquisition"]["missing_metrics_days"]
    _require(
        declared_missing == len(payload["coverage"]["missing_metrics_days"]),
        f"the manifest counts {declared_missing} missing metrics day(s) and lists "
        f"{len(payload['coverage']['missing_metrics_days'])}",
    )
    reasons = {entry.get("reason") for entry in payload["coverage"]["missing_metrics_days"]}
    permitted = {"not_published", "checksum_mismatch", "empty_member"}
    _require(
        reasons <= permitted,
        f"a metrics day is recorded missing for reason(s) {sorted(reasons - permitted)}. "
        f"docs/p4_preregistration.md §3.0a names {sorted(permitted)} and nothing else; "
        "anything the acquisition could not classify must have stopped it.",
    )
    checks.append(CheckResult("missing days", f"{declared_missing}, all classified"))

    # The exact-duplicate accounting, re-added here rather than imported. This
    # verifier never opens a metrics ZIP — the archives are not committed and
    # cannot be — so it cannot re-collapse the rows itself. What it can refuse to
    # trust is the aggregate: every number below is recomputed from the manifest's
    # own per-archive records, and each record must be internally consistent with
    # the rule. A manifest that summarised a collapse it did not record, or
    # recorded a retained count its raw count does not support, fails here.
    integrity = payload["acquisition"]["open_interest_source_integrity"]
    normalised = [
        record for record in payload["acquisition"]["archives"] if "normalisation" in record
    ]
    entries = [record["normalisation"] for record in normalised]
    _require(
        int(integrity["conflicting_duplicate_instants"]) == 0,
        f"the manifest records {integrity['conflicting_duplicate_instants']} conflicting "
        "duplicate instant(s). Rows sharing an instant that disagree in any field stop "
        "the acquisition, so a written snapshot cannot have seen one.",
    )
    for record in normalised:
        entry = record["normalisation"]
        read, kept = int(entry["rows_read"]), int(entry["observations_retained"])
        collapsed, instants_hit = (
            int(entry["exact_duplicate_rows_collapsed"]),
            int(entry["duplicate_instants"]),
        )
        _require(
            kept == read - collapsed,
            f"{record['name']} records {read} raw row(s), {collapsed} collapsed and "
            f"{kept} retained, which do not add up",
        )
        # What the archive contributed is no longer what A1 retained: amendment
        # A4 classifies those retained rows and only the valid ones reach the
        # reducer. The two claims are checked together in the A4 block below, and
        # the block itself is mandatory here so that a manifest cannot drop the
        # classification and leave `rows` accountable to nothing.
        _require(
            "validity" in record,
            f"{record['name']} records a duplicate normalisation and no "
            "observation-validity block. Amendment A4 classifies every logical row a "
            "metrics archive retains, so a manifest without that accounting was "
            "written under a superseded protocol.",
        )
        _require(
            instants_hit <= collapsed and (instants_hit == 0) == (collapsed == 0),
            f"{record['name']} records {collapsed} collapsed row(s) across "
            f"{instants_hit} duplicate instant(s); every affected instant loses at "
            "least one row, and no row is lost without one",
        )
    for key, recomputed_total in (
        ("archives_read", len(entries)),
        ("raw_rows_read", sum(int(e["rows_read"]) for e in entries)),
        (
            "logical_observations_retained",
            sum(int(e["observations_retained"]) for e in entries),
        ),
        (
            "exact_duplicate_rows_collapsed",
            sum(int(e["exact_duplicate_rows_collapsed"]) for e in entries),
        ),
        ("duplicate_instants_collapsed", sum(int(e["duplicate_instants"]) for e in entries)),
        (
            "archives_with_exact_duplicates",
            sum(1 for e in entries if int(e["exact_duplicate_rows_collapsed"])),
        ),
    ):
        _require(
            int(integrity[key]) == recomputed_total,
            f"the manifest says open_interest_source_integrity.{key} is "
            f"{integrity[key]} and its own per-archive records give {recomputed_total}",
        )
    checks.append(
        CheckResult(
            "exact duplicates",
            f"{integrity['exact_duplicate_rows_collapsed']} row(s) collapsed across "
            f"{integrity['duplicate_instants_collapsed']} instant(s) in "
            f"{integrity['archives_with_exact_duplicates']} archive(s), 0 conflicting",
        )
    )

    # The A4 accounting, on the same terms and for the same reason: this verifier
    # cannot re-classify rows it cannot open, so what it refuses to trust is the
    # arithmetic. A rejected observation must be visible as a rejected
    # observation — counted, partitioned, and adding up against the logical row
    # count A1 left behind — rather than quietly indistinguishable from a row the
    # archive never published.
    classified = [
        record for record in payload["acquisition"]["archives"] if "validity" in record
    ]
    validities = [record["validity"] for record in classified]
    for key in ("negative_observations", "nonfinite_observations"):
        _require(
            int(integrity[key]) == 0,
            f"the manifest records {integrity[key]} {key.replace('_', ' ')}. A negative "
            "or non-finite consumed open-interest value stops the acquisition under "
            "amendment A4, so a written snapshot cannot have seen one.",
        )
    for record in classified:
        entry = record["validity"]
        logical, valid = (
            int(entry["logical_observations"]),
            int(entry["valid_positive_observations"]),
        )
        invalid = int(entry["invalid_zero_observations"])
        parts = (
            int(entry["invalid_both_zero_observations"]),
            int(entry["invalid_zero_contracts_only"]),
            int(entry["invalid_zero_notional_only"]),
        )
        _require(
            logical == valid + invalid,
            f"{record['name']} records {logical} logical observation(s), {valid} valid "
            f"and {invalid} invalid, which do not add up",
        )
        _require(
            invalid == sum(parts),
            f"{record['name']} records {invalid} invalid observation(s) and "
            f"{sum(parts)} across the three disjoint zero shapes",
        )
        _require(
            int(entry["negative_observations"]) == 0
            and int(entry["nonfinite_observations"]) == 0,
            f"{record['name']} records a negative or non-finite consumed value; "
            "amendment A4 stops the acquisition on either",
        )
        _require(
            int(record["rows"]) == valid,
            f"{record['name']} claims it contributed {record['rows']} observation(s) "
            f"and its classification found {valid} valid one(s)",
        )
        if "normalisation" in record:
            _require(
                int(record["normalisation"]["observations_retained"]) == logical,
                f"{record['name']} retained "
                f"{record['normalisation']['observations_retained']} logical row(s) "
                f"after the duplicate collapse and classified {logical}. Amendment A4 "
                "runs on what A1 left behind, so the two counts are the same number.",
            )
    for key, recomputed_total in (
        ("archives_classified", len(validities)),
        ("logical_observations", sum(int(e["logical_observations"]) for e in validities)),
        (
            "valid_positive_observations",
            sum(int(e["valid_positive_observations"]) for e in validities),
        ),
        (
            "invalid_zero_observations",
            sum(int(e["invalid_zero_observations"]) for e in validities),
        ),
        (
            "invalid_both_zero_observations",
            sum(int(e["invalid_both_zero_observations"]) for e in validities),
        ),
        (
            "invalid_zero_contracts_only",
            sum(int(e["invalid_zero_contracts_only"]) for e in validities),
        ),
        (
            "invalid_zero_notional_only",
            sum(int(e["invalid_zero_notional_only"]) for e in validities),
        ),
        (
            "archives_with_invalid_zero_observations",
            sum(1 for e in validities if int(e["invalid_zero_observations"])),
        ),
    ):
        _require(
            int(integrity[key]) == recomputed_total,
            f"the manifest says open_interest_source_integrity.{key} is "
            f"{integrity[key]} and its own per-archive records give {recomputed_total}",
        )
    checks.append(
        CheckResult(
            "invalid zero observations",
            f"{integrity['invalid_zero_observations']} of "
            f"{integrity['logical_observations']} logical row(s) rejected across "
            f"{integrity['archives_with_invalid_zero_observations']} archive(s) "
            f"({integrity['invalid_both_zero_observations']} both-zero, "
            f"{integrity['invalid_zero_contracts_only']} zero-contracts, "
            f"{integrity['invalid_zero_notional_only']} zero-notional), "
            "0 negative, 0 non-finite",
        )
    )

    _require(
        int(payload["safety_checks"]["rest_rows_exported"]) == 0,
        "the manifest does not declare that no REST row was exported. §3.0a forbids "
        "fapi.binance.com standing in for a missing archive day.",
    )
    checks.append(CheckResult("no rest substitution", "0 rows"))

    ohlcv_manifest = root / payload["alignment"]["ohlcv_manifest"]
    _require(ohlcv_manifest.is_file(), f"the OHLCV manifest {ohlcv_manifest} does not exist")
    ohlcv = json.loads(ohlcv_manifest.read_text())
    _require(
        payload["alignment"]["ohlcv_processed_semantic_prefix_hash"]
        == ohlcv["processed_outer_coverage"]["semantic_prefix_hash"],
        "the derivatives source was aligned to a different OHLCV research input than the "
        "one committed beside it",
    )
    spine = pd.read_parquet(root / ohlcv["processed_outer_coverage"]["path"], columns=["date"])
    spine_hours = pd.DatetimeIndex(pd.to_datetime(spine["date"], utc=True))
    missing = spine_hours.difference(dates)
    _require(
        len(missing) == 0,
        f"{len(missing)} spine hour(s) have no derivatives row, so the P4 row for those "
        "candles does not exist",
    )
    checks.append(CheckResult("spine coverage", f"{len(spine_hours)} hours, none missing"))

    raw = pd.read_parquet(root / ohlcv["raw_pre_styx"]["path"], columns=["date", "close"])
    consistency = cross_source_consistency(frame, raw)
    checks.append(
        CheckResult(
            "cross-source",
            f"{consistency['hours_compared']} hours, worst basis "
            f"{consistency['worst_relative_deviation']}",
        )
    )
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    try:
        checks = verify_derivatives_snapshot(args.manifest)
    except DerivativesSnapshotVerificationError as exc:
        print(f"FAILED: {exc}")
        return 1
    for check in checks:
        print(f"  ok  {check.name}: {check.detail}")
    print(f"{len(checks)} checks passed against {args.manifest}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
