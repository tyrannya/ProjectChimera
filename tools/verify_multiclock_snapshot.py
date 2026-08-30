"""Refuse a multi-clock source that cannot be trusted, before anyone fits on it.

    python -m tools.verify_multiclock_snapshot

``tools.acquire_multiclock_source`` proves what it wrote at the moment it wrote
it. A fresh clone inherits none of that: it has one Parquet file and a manifest
asserting things about it. So every claim is **recomputed** rather than read,
and the first disagreement fails closed.

The order matters. Boundary breaches are checked before digests, so a snapshot
carrying a row from ``P4-HOLD`` is rejected as a boundary breach rather than as
an incidental hash mismatch — and the rejection reports how many rows, never
which, because a verifier that printed the offending candle would publish the
region it exists to keep closed.

Every derived clock is re-cut from the 1m source through the same
:func:`nn.multiclock.resample_from_minutes` the research entrypoints call, and
its digest is compared against the manifest. That is what makes the derived
clocks reproducible rather than merely asserted: nothing but the committed 1m
file and this repository's code is needed to obtain them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from nn.multiclock import (
    ALL_CLOCKS,
    PARITY_TOLERANCE,
    RESEARCH_VISIBLE_END,
    STYX_START,
    assert_manifest_clock,
    assert_minute_grid,
    bar_availability,
    candle_digest,
    minute_gaps,
    parity_against,
    resample_from_minutes,
)
from tools.acquire_multiclock_source import MANIFEST_NAME, SNAPSHOT_SCHEMA

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "data" / "research" / MANIFEST_NAME

#: Keys a manifest must carry, with the type each must have, checked before any
#: value is used so a wrong-typed field is a named rejection rather than a
#: ``TypeError`` three checks later.
REQUIRED_KEYS: tuple[tuple[str, type], ...] = (
    ("snapshot_schema", str),
    ("symbol", str),
    ("market", str),
    ("source", dict),
    ("source.canonical_base_url", str),
    ("source.months", list),
    ("boundaries", dict),
    ("boundaries.research_visible_end", str),
    ("boundaries.styx_start", str),
    ("boundaries.p4_hold_opened", bool),
    ("boundaries.styx_opened", bool),
    ("minutes", dict),
    ("minutes.path", str),
    ("minutes.rows", int),
    ("minutes.start", str),
    ("minutes.end", str),
    ("minutes.sha256", str),
    ("minutes.digest", str),
    ("clocks", dict),
    ("minutes.gaps", list),
    ("parity_1h", dict),
    ("parity_1h.reference", str),
    ("parity_1h.tolerance", float),
    ("parity_1h.overlapping_bars", int),
    ("parity_1h.mismatching_bars", int),
    ("parity_1h.only_in_left", int),
    ("parity_1h.only_in_right", int),
    ("parity_1h.mismatching_timestamps", list),
)

#: What every entry of ``source.months`` must carry. The per-object digests are
#: the whole provenance claim — that each archive is the object Binance
#: published — so a manifest that omitted one would be asserting provenance it
#: does not record.
REQUIRED_MONTH_KEYS: tuple[str, ...] = (
    "month",
    "object",
    "zip_sha256",
    "member",
    "member_sha256",
    "rows",
    "open_time_unit",
)


class SnapshotError(SystemExit):
    """The committed multi-clock source and its manifest disagree."""


def _lookup(payload: dict[str, Any], path: str) -> Any:
    node: Any = payload
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            raise SnapshotError(f"the manifest is missing {path!r}")
        node = node[part]
    return node


def check_shape(manifest: dict[str, Any]) -> None:
    for path, kind in REQUIRED_KEYS:
        value = _lookup(manifest, path)
        if not isinstance(value, kind) or (kind is int and isinstance(value, bool)):
            raise SnapshotError(
                f"the manifest's {path!r} is {type(value).__name__}, expected {kind.__name__}"
            )
    months = manifest["source"]["months"]
    if not months:
        raise SnapshotError("the manifest records no source objects at all")
    for position, record in enumerate(months):
        missing = [key for key in REQUIRED_MONTH_KEYS if key not in record]
        if missing:
            raise SnapshotError(f"source.months[{position}] is missing {missing}")
        if record["open_time_unit"] not in {"ms", "us"}:
            raise SnapshotError(
                f"source.months[{position}] declares open_time_unit "
                f"{record['open_time_unit']!r}; Binance publishes ms or us"
            )
        for key in ("zip_sha256", "member_sha256"):
            digest = record[key]
            if not isinstance(digest, str) or len(digest) != 64:
                raise SnapshotError(f"source.months[{position}].{key} is not a SHA-256")
    declared_rows = sum(int(record["rows"]) for record in months)
    if declared_rows < manifest["minutes"]["rows"]:
        raise SnapshotError(
            f"the source objects declare {declared_rows} rows and the committed 1m file "
            f"holds {manifest['minutes']['rows']}; the file cannot hold more than its sources"
        )
    if manifest["snapshot_schema"] != SNAPSHOT_SCHEMA:
        raise SnapshotError(
            f"the manifest declares schema {manifest['snapshot_schema']!r}; this tool "
            f"verifies {SNAPSHOT_SCHEMA!r} and refuses to guess at another"
        )


def check_boundaries(manifest: dict[str, Any], minutes: pd.DataFrame) -> dict[str, Any]:
    """The seal, checked against the data rather than against the manifest's word."""
    declared_visible = pd.Timestamp(manifest["boundaries"]["research_visible_end"])
    declared_styx = pd.Timestamp(manifest["boundaries"]["styx_start"])
    if declared_visible != RESEARCH_VISIBLE_END:
        raise SnapshotError(
            f"the manifest declares a research-visible boundary of {declared_visible} "
            f"and this repository's is {RESEARCH_VISIBLE_END}. The boundary is the first "
            "instant of the retired P4-HOLD region and does not move."
        )
    if declared_styx != STYX_START:
        raise SnapshotError(
            f"the manifest declares a Styx instant of {declared_styx} and the "
            f"repository's is {STYX_START}. Styx never moves."
        )
    for claim in ("p4_hold_opened", "styx_opened"):
        if manifest["boundaries"][claim] is not False:
            raise SnapshotError(f"the manifest claims {claim} is true; nothing may open it")

    dates = pd.to_datetime(minutes["date"], utc=True)
    beyond_visible = int((dates >= RESEARCH_VISIBLE_END).sum())
    if beyond_visible:
        raise SnapshotError(
            f"the committed 1m source carries {beyond_visible} row(s) at or after "
            f"{RESEARCH_VISIBLE_END.isoformat()}, which is the start of P4-HOLD"
        )
    beyond_styx = int((dates >= STYX_START).sum())
    if beyond_styx:  # pragma: no cover - implied by the check above
        raise SnapshotError(f"the committed 1m source carries {beyond_styx} sealed row(s)")
    return {"rows_at_or_after_research_boundary": 0, "rows_at_or_after_styx": 0}


def check_minutes(
    manifest: dict[str, Any], path: Path, minutes: pd.DataFrame
) -> dict[str, Any]:
    record = manifest["minutes"]
    assert_minute_grid(minutes, what="the committed 1m source")

    actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_sha != record["sha256"]:
        raise SnapshotError(
            f"{path.name} hashes to {actual_sha} and the manifest says {record['sha256']}"
        )
    actual_digest = candle_digest(minutes)
    if actual_digest != record["digest"]:
        raise SnapshotError(
            f"the 1m candles digest to {actual_digest} and the manifest says "
            f"{record['digest']}; the file was rebuilt with different values"
        )
    dates = pd.to_datetime(minutes["date"], utc=True)
    for key, actual in (
        ("rows", int(len(minutes))),
        ("start", dates.iloc[0].isoformat()),
        ("end", dates.iloc[-1].isoformat()),
    ):
        if actual != record[key]:
            raise SnapshotError(
                f"the 1m source's {key} is {actual!r} and the manifest says {record[key]!r}"
            )

    gaps = minute_gaps(minutes)
    if gaps != record["gaps"]:
        raise SnapshotError(
            f"the 1m source has {len(gaps)} discontinuit(ies) and the manifest records "
            f"{len(record['gaps'])}; the gap structure decides every clock's completeness"
        )
    return {"rows": len(minutes), "gaps": len(gaps), "sha256": actual_sha}


def check_clocks(manifest: dict[str, Any], minutes: pd.DataFrame) -> dict[str, Any]:
    """Re-cut every clock from the 1m source and hold it to the manifest."""
    declared = manifest["clocks"]
    missing = [clock for clock in ALL_CLOCKS if clock not in declared]
    if missing:
        raise SnapshotError(f"the manifest describes no {missing} clock(s)")

    summary: dict[str, Any] = {}
    for timeframe in ALL_CLOCKS:
        frame = resample_from_minutes(minutes, timeframe)
        assert_manifest_clock(frame, declared[timeframe])
        # Availability is recomputed too. It is what says how many bars the
        # completeness rule dropped, and a manifest that merely *stated* it could
        # under-report an outage without any digest moving.
        availability = bar_availability(minutes, timeframe)
        for key in ("buckets_touched", "complete_bars", "incomplete_bars_dropped"):
            # Required, not optional. Skipping a key the manifest omits means a
            # manifest can pass this check by saying less.
            if key not in declared[timeframe]:
                raise SnapshotError(f"the {timeframe} clock record declares no {key!r}")
            if declared[timeframe][key] != availability[key]:
                raise SnapshotError(
                    f"the {timeframe} clock recomputes {key}={availability[key]} and the "
                    f"manifest says {declared[timeframe][key]}"
                )
        if availability["complete_bars"] != len(frame):
            raise SnapshotError(
                f"the {timeframe} clock has {len(frame)} bars and its availability record "
                f"counts {availability['complete_bars']} complete ones"
            )
        summary[timeframe] = int(len(frame))
    return summary


def check_parity(manifest: dict[str, Any], minutes: pd.DataFrame) -> dict[str, Any]:
    """The 1h parity claim, **recomputed** rather than read.

    The claim that licenses fitting on this source is that the derived 1h clock
    agrees with the committed 1h history. A verifier that only checked the
    manifest against itself would accept any parity result a manifest cared to
    state, which is precisely the class of assertion this tool exists to replace.

    So the comparison is run again here, from the committed 1m file and the
    committed 1h history, and the manifest is held to every figure it reports —
    including the enumerated timestamps, which must be the same hours and not
    merely the same count.
    """
    parity = manifest["parity_1h"]
    listed = len(parity["mismatching_timestamps"])
    if listed != parity["mismatching_bars"]:
        raise SnapshotError(
            f"the manifest says {parity['mismatching_bars']} hour(s) disagree and "
            f"enumerates {listed}"
        )

    reference_path = REPO_ROOT / str(parity["reference"])
    if not reference_path.is_file():
        raise SnapshotError(f"the 1h reference {reference_path} the manifest names is absent")
    reference = pd.read_parquet(reference_path)
    reference = reference.loc[reference["date"] < RESEARCH_VISIBLE_END].reset_index(drop=True)
    # This repository's tolerance, not the manifest's. Taking it from the artifact
    # under test lets a manifest declare a loose one and be verified against it,
    # which is the verifier agreeing with whatever it was handed.
    declared_tolerance = float(parity["tolerance"])
    if declared_tolerance != PARITY_TOLERANCE:
        raise SnapshotError(
            f"the manifest declares a parity tolerance of {declared_tolerance} and this "
            f"repository's is {PARITY_TOLERANCE}; the comparison it describes is not "
            "the one this tool performs"
        )
    recomputed = parity_against(
        resample_from_minutes(minutes, "1h"),
        reference,
        timeframe="1h",
        tolerance=PARITY_TOLERANCE,
    )
    for key, actual in (
        ("overlapping_bars", recomputed.overlapping_bars),
        ("mismatching_bars", recomputed.mismatching_bars),
        ("only_in_left", recomputed.only_in_left),
        ("only_in_right", recomputed.only_in_right),
    ):
        if actual != parity[key]:
            raise SnapshotError(
                f"the 1h parity recomputes {key}={actual} and the manifest says "
                f"{parity[key]}; the manifest describes a comparison that is not this one"
            )
    if list(recomputed.mismatching_timestamps) != list(parity["mismatching_timestamps"]):
        raise SnapshotError(
            "the 1h parity disagrees with the manifest about *which* hours differ, not "
            "only how many"
        )

    stamps = [pd.Timestamp(value) for value in parity["mismatching_timestamps"]]
    late = [value for value in stamps if value >= RESEARCH_VISIBLE_END]
    if late:
        raise SnapshotError(
            f"{len(late)} disagreeing hour(s) lie at or after the research boundary"
        )

    # Recomputed like everything else in this block. Reading it from the manifest
    # and reporting it under `"recomputed": True` would be the one number in the
    # report that the report had not checked.
    agreement = 0.0
    if recomputed.overlapping_bars:
        agreeing = recomputed.overlapping_bars - recomputed.mismatching_bars
        agreement = round(agreeing / recomputed.overlapping_bars, 9)
    declared = parity.get("agreement_fraction")
    if declared is not None and abs(float(declared) - agreement) > 1e-9:
        raise SnapshotError(
            f"the 1h parity recomputes an agreement fraction of {agreement} and the "
            f"manifest says {declared}"
        )
    return {
        "overlapping_bars": recomputed.overlapping_bars,
        "mismatching_bars": recomputed.mismatching_bars,
        "recomputed": True,
        "agreement_fraction": agreement,
    }


def verify(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.is_file():
        raise SnapshotError(f"no multi-clock manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    check_shape(manifest)

    minutes_path = REPO_ROOT / manifest["minutes"]["path"]
    if not minutes_path.is_file():
        raise SnapshotError(f"the manifest points at {minutes_path}, which is absent")
    minutes = pd.read_parquet(minutes_path)

    report = {
        "manifest": str(manifest_path),
        "boundaries": check_boundaries(manifest, minutes),
        "minutes": check_minutes(manifest, minutes_path, minutes),
        "clocks": check_clocks(manifest, minutes),
        "parity_1h": check_parity(manifest, minutes),
    }
    return report


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_argparser().parse_args(argv)
    report = verify(args.manifest)
    logger.info(json.dumps(report, indent=2))
    logger.info("multi-clock source verified.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
