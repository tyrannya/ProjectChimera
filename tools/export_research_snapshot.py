"""Export a research-only BTC snapshot that cannot contain Styx rows.

This helper is intentionally conservative. It verifies the canonical source
dataset against the frozen v4 artifact, then writes:

* raw OHLCV strictly before the contract seal;
* the exact processed OHLCV14 prefix needed by the frozen outer folds;
* a manifest with contract, hashes, row ranges, and safety checks.

It never writes rows at or after ``sealed_test_start``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from chimera.contracts import HOLD_IDX, LONG_IDX, SHORT_IDX
from nn.data_fingerprint import (
    fingerprint_raw_input,
    research_input_digest,
)
from nn.data_pipeline import DatasetMetadata, load_dataset, save_dataset
from nn.regime import load_raw_ohlcv
from nn.research_contract import load_contract

SNAPSHOT_SCHEMA = "chimera.research-snapshot/1"
DEFAULT_REFERENCE = Path(
    "artifacts/walkforward/btc_nested_v4_seed_42/walkforward.json"
)
RAW_NAME = "btc_usdt_1h_gen1_raw_pre_styx.parquet"
PROCESSED_NAME = "btc_usdt_1h_gen1_ohlcv14_outer_coverage.parquet"
MANIFEST_NAME = "btc_usdt_1h_gen1_snapshot_manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _outer_end(reference: dict[str, Any]) -> int:
    ends = [
        int(fold["periods"]["outer_validation"]["row_range"][1])
        for fold in reference["folds"]
    ]
    if not ends:
        raise ValueError("reference artifact contains no outer folds")
    return max(ends)


def _class_balance(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["target"].value_counts().to_dict()
    return {
        "SHORT": int(counts.get(SHORT_IDX, 0)),
        "HOLD": int(counts.get(HOLD_IDX, 0)),
        "LONG": int(counts.get(LONG_IDX, 0)),
    }


def export_snapshot(
    *,
    dataset_path: Path,
    raw_path: Path,
    reference_path: Path,
    out_dir: Path,
    contract_id: str,
) -> dict[str, Any]:
    contract = load_contract(contract_id)
    frame, metadata = load_dataset(dataset_path)
    contract.require_scope(
        exchange=metadata.exchange,
        pair=metadata.pair,
        timeframe=metadata.timeframe,
        source=str(dataset_path),
    )

    reference = json.loads(reference_path.read_text())
    recorded_contract = reference["sealed_test"]["research_contract"]
    if recorded_contract["contract_id"] != contract.contract_id:
        raise ValueError(
            "reference artifact contract id does not match selected contract: "
            f"{recorded_contract['contract_id']!r} != {contract.contract_id!r}"
        )
    if recorded_contract["contract_hash"] != contract.contract_hash:
        raise ValueError(
            "reference artifact contract hash does not match selected contract"
        )

    dates = pd.DatetimeIndex(pd.to_datetime(frame["date"], utc=True))
    if not dates.is_monotonic_increasing or dates.has_duplicates:
        raise ValueError("processed dataset timestamps must be sorted and unique")

    seal_row = int(dates.searchsorted(contract.sealed_test_start, side="left"))
    recorded_seal_row = int(reference["sealed_test"]["start_row"])
    recorded_research_rows = int(reference["research_input"]["research_rows"])
    if seal_row != recorded_seal_row or seal_row != recorded_research_rows:
        raise ValueError(
            "resolved seal row disagrees with frozen v4 evidence: "
            f"resolved={seal_row}, artifact={recorded_seal_row}, "
            f"research_input={recorded_research_rows}"
        )
    if seal_row <= 0 or seal_row >= len(frame):
        raise ValueError("seal does not split the processed dataset")
    if not (dates[:seal_row] < contract.sealed_test_start).all():
        raise ValueError("processed research prefix reaches Styx")
    if not (dates[seal_row:] >= contract.sealed_test_start).all():
        raise ValueError("processed sealed suffix starts before the contract seal")

    canonical_hash = research_input_digest(
        {name: frame[name] for name in frame.columns},
        feature_names=metadata.feature_names,
        exchange=metadata.exchange,
        pair=metadata.pair,
        timeframe=metadata.timeframe,
        feature_spec=metadata.feature_spec,
        target_spec=metadata.target_spec,
        rows=seal_row,
    )
    recorded_hash = reference["research_input"]["research_input_hash"]
    if canonical_hash != recorded_hash:
        raise ValueError(
            "source dataset does not match the frozen canonical research input: "
            f"{canonical_hash} != {recorded_hash}"
        )

    outer_end = _outer_end(reference)
    horizon = int(metadata.target_spec["horizon"])
    if not 0 < outer_end < seal_row:
        raise ValueError(
            f"outer coverage end {outer_end} is not inside research rows [0, {seal_row})"
        )
    if outer_end + horizon > seal_row:
        raise ValueError(
            "frozen outer coverage is too close to Styx for its target horizon"
        )

    processed = frame.iloc[:outer_end].copy()
    processed_dates = pd.DatetimeIndex(pd.to_datetime(processed["date"], utc=True))
    if not (processed_dates < contract.sealed_test_start).all():
        raise ValueError("processed outer-coverage snapshot reaches Styx")

    raw = load_raw_ohlcv(raw_path)
    raw_safe = raw.loc[raw.index < contract.sealed_test_start].copy()
    if raw_safe.empty:
        raise ValueError("raw pre-Styx snapshot would be empty")
    if not (raw_safe.index < contract.sealed_test_start).all():
        raise ValueError("raw research snapshot reaches Styx")

    out_dir.mkdir(parents=True, exist_ok=True)
    raw_out = out_dir / RAW_NAME
    processed_out = out_dir / PROCESSED_NAME
    manifest_out = out_dir / MANIFEST_NAME

    raw_safe.reset_index().to_parquet(raw_out, index=False)

    prefix_metadata = DatasetMetadata(
        exchange=metadata.exchange,
        pair=metadata.pair,
        timeframe=metadata.timeframe,
        rows=len(processed),
        start=processed_dates[0].isoformat(),
        end=processed_dates[-1].isoformat(),
        feature_names=list(metadata.feature_names),
        feature_spec=dict(metadata.feature_spec),
        target_spec=dict(metadata.target_spec),
        class_balance=_class_balance(processed),
        validation=dict(metadata.validation),
        created_at=metadata.created_at,
    )
    save_dataset(processed_out, processed, prefix_metadata)

    prefix_hash = research_input_digest(
        {name: processed[name] for name in processed.columns},
        feature_names=metadata.feature_names,
        exchange=metadata.exchange,
        pair=metadata.pair,
        timeframe=metadata.timeframe,
        feature_spec=metadata.feature_spec,
        target_spec=metadata.target_spec,
        rows=len(processed),
    )
    raw_fingerprint = fingerprint_raw_input(
        raw_safe,
        exchange=metadata.exchange,
        pair=metadata.pair,
        timeframe=metadata.timeframe,
        visible_through=raw_safe.index[-1],
    )

    manifest = {
        "snapshot_schema": SNAPSHOT_SCHEMA,
        "contract": contract.provenance(),
        "contains_styx": False,
        "canonical_reference": {
            "artifact": str(reference_path),
            "research_input_hash": recorded_hash,
            "research_rows": seal_row,
            "max_outer_end_row": outer_end,
            "outer_to_styx_margin_rows": seal_row - outer_end,
            "target_horizon": horizon,
        },
        "raw_pre_styx": {
            "path": str(raw_out),
            "rows": len(raw_safe),
            "start": raw_safe.index[0].isoformat(),
            "end": raw_safe.index[-1].isoformat(),
            "semantic_hash": raw_fingerprint.raw_input_hash,
            "sha256": _sha256(raw_out),
            "columns": ["date", "open", "high", "low", "close", "volume"],
        },
        "processed_outer_coverage": {
            "path": str(processed_out),
            "rows": len(processed),
            "row_range": [0, outer_end],
            "start": processed_dates[0].isoformat(),
            "end": processed_dates[-1].isoformat(),
            "semantic_prefix_hash": prefix_hash,
            "sha256": _sha256(processed_out),
            "metadata_path": str(
                processed_out.with_suffix(processed_out.suffix + ".meta.json")
            ),
            "metadata_sha256": _sha256(
                processed_out.with_suffix(processed_out.suffix + ".meta.json")
            ),
        },
        "safety_checks": {
            "canonical_research_input_verified": True,
            "raw_strictly_before_styx": bool(
                raw_safe.index[-1] < contract.sealed_test_start
            ),
            "processed_strictly_before_styx": bool(
                processed_dates[-1] < contract.sealed_test_start
            ),
            "outer_horizon_within_research": outer_end + horizon <= seal_row,
            "styx_rows_exported": 0,
        },
    }
    manifest_out.write_text(json.dumps(manifest, indent=2) + "\n")

    print(
        "research snapshot exported:",
        f"raw_rows={len(raw_safe)}",
        f"processed_rows={len(processed)}",
        f"seal_row={seal_row}",
        f"outer_end={outer_end}",
        "styx_rows=0",
    )
    print(f"manifest={manifest_out}")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--out-dir", type=Path, default=Path("data/research"))
    parser.add_argument("--contract", default="btc-usdt-1h-gen1")
    return parser


def main() -> None:
    args = _parser().parse_args()
    export_snapshot(
        dataset_path=args.dataset,
        raw_path=args.raw,
        reference_path=args.reference,
        out_dir=args.out_dir,
        contract_id=args.contract,
    )


if __name__ == "__main__":
    main()
