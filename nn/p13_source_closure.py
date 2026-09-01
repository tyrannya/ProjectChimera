"""What the acquired P13 sources ARE — and deliberately nothing about what they pay.

Source closure is the chronology step between having the bytes and running the
screen. It answers questions about the SOURCES: how many rows each object
carries, which unit its timestamps are in, where the holes are, which months
published a mark series, how many funding settlements the archive contains, and
how many rows the research boundary truncated.

**It computes no economics, and that is the point of it being a separate module.**
No position is opened, no settlement is applied, no basis is differenced, no block
return is formed and no gate condition is evaluated. It never calls
``run_offline_screen``, ``run_screen`` or ``evaluate_block``, and
``tests/test_p13_acquisition.py`` asserts that on the import graph and on the
call graph rather than trusting this paragraph.

**Why it may still speak about the A2R2 held window.** Whether a held hour has a
mark row is a fact about ROW PRESENCE, decidable from the source manifest before
any number exists — the same availability-only question ``MARK_PRICE_FALLBACK``
and A2 are both triggered by. So this module reports, per block, the opening
instant A2R2's own :func:`nn.p13_blocks.find_opening_instant` selects and how many
held hours lack a required row. If that coverage makes a future governed screen
terminate, the consequence is stated as a SOURCE-VALIDITY fact and never as an
economic one: ``NOT EVALUABLE`` says nothing about carry, and partial economics
are not computed on the way to saying so.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from nn.p13_alignment import AlignedSources, grid_instants
from nn.p13_blocks import CalendarBlock, calendar_blocks, find_opening_instant
from nn.p13_preregistration import ACTIVE_DESIGN, DATA_BOUNDARY, preregistration_hash
from nn.p13_sources import (
    RESEARCH_BOUNDARY_NS,
    FundingTable,
    KlineTable,
    ObjectProvenance,
    extract_single_member,
    read_funding_object,
    read_kline_object,
)

__all__ = [
    "CLOSURE_SCHEMA",
    "ClosureError",
    "LoadedSources",
    "load_acquired_sources",
    "family_summary",
    "grid_coverage",
    "held_window_mark_coverage",
    "closure_payload",
]

CLOSURE_SCHEMA = "chimera.p13-source-closure/1"

#: Which planned ``field`` maps to which loader. ``markPriceKlines`` shares the
#: kline layout, which ``KLINE_COLUMNS`` already records as a first-party fact.
KLINE_FIELDS = ("spot_price", "perpetual_price", "mark_price")
FUNDING_FIELD = "funding_settlement"


class ClosureError(RuntimeError):
    """The acquired sources cannot be described honestly."""


@dataclass(frozen=True)
class LoadedSources:
    """Every acquired object, parsed, with the provenance of each reading."""

    provenance: tuple[ObjectProvenance, ...]
    aligned: AlignedSources
    published_mark_periods: tuple[str, ...]
    #: Per family, the periods whose object was read. Kept so a family that is
    #: short a month is visible as a missing OBJECT rather than only as a hole in
    #: the row grid.
    periods_by_field: dict[str, tuple[str, ...]]


def load_acquired_sources(manifest: dict[str, Any], cache_dir: Path) -> LoadedSources:
    """Parse every object the manifest names, from the bytes on disk.

    The manifest's per-object digests are RE-CHECKED here by handing each object's
    published checksum back to the loader together with its archive bytes, so a
    cache file that changed between acquisition and closure fails rather than
    being described. Closure that trusted the manifest would be describing a
    record rather than the data.
    """
    spot: list = []
    perp: list = []
    mark: list = []
    funding: list = []
    provenance: list[ObjectProvenance] = []
    periods: dict[str, list[str]] = {}
    mark_periods: list[str] = []

    for record in manifest["objects"]:
        field = record["field"]
        name = record["object_name"]
        path = cache_dir / name
        if not path.is_file():
            raise ClosureError(
                f"{name} is named by the manifest and absent from {cache_dir}. The closure "
                "describes bytes, not records of bytes."
            )
        raw = path.read_bytes()
        member_name, payload = extract_single_member(raw)
        periods.setdefault(field, []).append(record["period"])

        if field == FUNDING_FIELD:
            table: FundingTable | KlineTable = read_funding_object(
                payload,
                object_name=name,
                period=record["period"],
                raw_object=raw,
                member_name=member_name,
                published_checksum=record["published_checksum"],
            )
            funding.extend(table.rows)
        elif field in KLINE_FIELDS:
            table = read_kline_object(
                payload,
                field=field,
                object_name=name,
                period=record["period"],
                raw_object=raw,
                member_name=member_name,
                published_checksum=record["published_checksum"],
            )
            if field == "spot_price":
                spot.extend(table.rows)
            elif field == "perpetual_price":
                perp.extend(table.rows)
            else:
                mark.extend(table.rows)
                mark_periods.append(record["period"])
        else:  # pragma: no cover - the plan emits no other field
            raise ClosureError(f"{name}: unknown source field {field!r}")
        provenance.append(table.provenance)

    aligned = AlignedSources.build(
        spot=spot,
        perpetual=perp,
        mark=mark,
        funding=funding,
        published_mark_periods=sorted(set(mark_periods)),
    )
    return LoadedSources(
        provenance=tuple(provenance),
        aligned=aligned,
        published_mark_periods=tuple(sorted(set(mark_periods))),
        periods_by_field={k: tuple(sorted(v)) for k, v in periods.items()},
    )


def family_summary(provenance: Sequence[ObjectProvenance], field: str) -> dict[str, Any]:
    """One source family, described from the readings rather than from the rows."""
    records = [p for p in provenance if p.field == field]
    if not records:
        raise ClosureError(f"no objects read for field {field!r}")
    firsts = [p.first_instant_ns for p in records if p.first_instant_ns is not None]
    lasts = [p.last_instant_ns for p in records if p.last_instant_ns is not None]
    units = Counter(p.resolved_epoch_unit for p in records)
    return {
        "field": field,
        "objects": len(records),
        "periods": sorted(p.period for p in records),
        "rows_read": sum(p.rows_read for p in records),
        "first_instant_ns": min(firsts) if firsts else None,
        "last_instant_ns": max(lasts) if lasts else None,
        "first_instant_utc": _iso(min(firsts)) if firsts else None,
        "last_instant_utc": _iso(max(lasts)) if lasts else None,
        "resolved_epoch_units": dict(sorted(units.items())),
        "objects_by_unit": {
            unit: sorted(p.period for p in records if p.resolved_epoch_unit == unit)
            for unit in sorted(units)
        },
        "rows_dropped_at_boundary": sum(p.rows_dropped_at_boundary for p in records),
        "instants_withheld_contradictory": sum(p.ambiguous_instants for p in records),
        "instants_withheld_non_positive": sum(p.non_positive_instants for p in records),
        "repeated_instants_passed_through": sum(p.repeated_instants for p in records),
        "archive_bytes": sum(p.byte_size or 0 for p in records),
        "member_bytes": sum(p.member_byte_size for p in records),
        "checksum_states": dict(sorted(Counter(p.checksum_state for p in records).items())),
    }


def grid_coverage(
    instants: Iterable[int], *, start_ns: int, end_exclusive_ns: int
) -> dict[str, Any]:
    """Which hours of the reference grid this family supplies, and where it does not.

    The grid is generated from the CALENDAR, never from the rows present, for the
    same reason the block runner does it that way: a completeness check driven by
    the rows can never notice an hour for which nothing was published at all.
    """
    present = set(instants)
    expected = list(grid_instants(start_ns, end_exclusive_ns))
    missing = [i for i in expected if i not in present]
    runs: list[dict[str, Any]] = []
    for instant in missing:
        if runs and instant == runs[-1]["end_ns"] + 3_600_000_000_000:
            runs[-1]["end_ns"] = instant
            runs[-1]["hours"] += 1
        else:
            runs.append({"start_ns": instant, "end_ns": instant, "hours": 1})
    return {
        "expected_hours": len(expected),
        "present_hours": len([i for i in expected if i in present]),
        "missing_hours": len(missing),
        "coverage_fraction": (
            f"{(len(expected) - len(missing)) / len(expected):.9f}" if expected else None
        ),
        "gap_runs": len(runs),
        "longest_gap_hours": max((r["hours"] for r in runs), default=0),
        "gaps": [
            {
                "start_utc": _iso(r["start_ns"]),
                "end_utc": _iso(r["end_ns"]),
                "hours": r["hours"],
            }
            for r in runs[:200]
        ],
        "gaps_listed": min(len(runs), 200),
        "rows_outside_the_reference_grid": len(present - set(expected)),
    }


def held_window_mark_coverage(
    aligned: AlignedSources, blocks: Sequence[CalendarBlock] | None = None
) -> list[dict[str, Any]]:
    """Per block: where A2R2 opens, and which held hours lack a required row.

    **Row presence only.** ``find_opening_instant`` is the committed A2R2 runtime's
    own function and reads presence; the held window is calendar arithmetic; and
    the scan below asks ``instant_validity`` which sources supply a row. Nothing
    here reads a price, opens a position or forms a return.

    The A2R2 rule this reports against: the position opens at the first
    EXECUTION-valid instant, bar 0 is a held bar, and every held bar needs a mark.
    A held hour missing any required source would make a future governed screen
    terminate ``NOT EVALUABLE`` — a SOURCE-VALIDITY consequence, stated here as
    one and not converted into an economic finding.
    """
    chosen = tuple(blocks) if blocks is not None else calendar_blocks()
    report: list[dict[str, Any]] = []
    for block in chosen:
        entry: dict[str, Any] = {
            "block": block.label,
            "calendar_start_utc": _iso(block.start_ns),
            "calendar_end_exclusive_utc": _iso(block.end_exclusive_ns),
        }
        try:
            opening = find_opening_instant(aligned, block)
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            entry.update(
                {
                    "opens": False,
                    "reason": f"{type(exc).__name__}: {exc}",
                    "held_hours": None,
                }
            )
            report.append(entry)
            continue

        instants = list(
            grid_instants(block.start_ns, min(block.end_exclusive_ns, RESEARCH_BOUNDARY_NS))
        )
        exit_ns = instants[-1]
        held = [i for i in instants if opening.opened_at_ns <= i < exit_ns]
        missing_mark: list[int] = []
        missing_execution: list[int] = []
        for instant in held:
            validity = aligned.instant_validity(instant)
            if not validity.has_execution:
                missing_execution.append(instant)
            elif not validity.has_liquidation_mark:
                missing_mark.append(instant)
        exit_validity = aligned.instant_validity(exit_ns)
        entry.update(
            {
                "opens": True,
                "opened_at_utc": _iso(opening.opened_at_ns),
                "opened_at_calendar_boundary": not opening.delayed,
                "opening_delayed_by_hours": opening.skipped_instants,
                "opening_delay_reason": opening.reason,
                "opening_consulted_mark": opening.OPENING_CONSULTED_MARK,
                "intended_close_utc": _iso(exit_ns),
                "held_hours": len(held),
                "held_hours_missing_execution_row": len(missing_execution),
                "held_hours_missing_mark_row": len(missing_mark),
                "first_held_hour_missing_mark_utc": (
                    _iso(missing_mark[0]) if missing_mark else None
                ),
                "first_held_hour_missing_execution_utc": (
                    _iso(missing_execution[0]) if missing_execution else None
                ),
                "exit_bar_execution_valid": exit_validity.has_execution,
                "source_coverage_consequence": _consequence(
                    missing_execution, missing_mark, exit_validity.has_execution
                ),
            }
        )
        report.append(entry)
    return report


def _consequence(
    missing_execution: Sequence[int], missing_mark: Sequence[int], exit_ok: bool
) -> str:
    """The SOURCE-VALIDITY consequence of this block's coverage. Never economic."""
    if missing_execution or missing_mark:
        return (
            "A HELD hour lacks a required source row, so a future governed screen would "
            "terminate P13 ALWAYS-ON ANNUAL SPOT/PERP CARRY: NOT EVALUABLE at that hour "
            "under A2R2. This is source insufficiency for a required risk quantity and "
            "says NOTHING about carry; no economics are computed here or implied."
        )
    if not exit_ok:
        return (
            "Every held hour is covered, but the intended close lacks an execution row, "
            "which is POSITION_LIFECYCLE.close_instant's UNCLOSED case governed by "
            "amendment A1. Reported as a source fact; no economics are computed."
        )
    return (
        "Every held hour supplies both execution rows and a mark row, and the intended "
        "close supplies both execution rows. No source-validity obstacle to a future "
        "governed screen is present in this block. That is a statement about COVERAGE "
        "ONLY and is not a prediction, a result, or any claim about carry."
    )


def closure_payload(
    loaded: LoadedSources,
    *,
    manifest: dict[str, Any],
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The whole source-closure record, deterministic in its own content."""
    boundary_start = int(
        datetime.fromisoformat(DATA_BOUNDARY["span_start_inclusive"]).timestamp()
        * 1_000_000_000
    )
    families = {
        field: family_summary(loaded.provenance, field)
        for field in (*KLINE_FIELDS, FUNDING_FIELD)
    }
    coverage = {
        "spot_price": grid_coverage(
            loaded.aligned.spot, start_ns=boundary_start, end_exclusive_ns=RESEARCH_BOUNDARY_NS
        ),
        "perpetual_price": grid_coverage(
            loaded.aligned.perpetual,
            start_ns=boundary_start,
            end_exclusive_ns=RESEARCH_BOUNDARY_NS,
        ),
        "mark_price": grid_coverage(
            loaded.aligned.mark, start_ns=boundary_start, end_exclusive_ns=RESEARCH_BOUNDARY_NS
        ),
    }
    funding_instants = [row.instant_ns for row in loaded.aligned.funding]
    intervals = Counter(
        str(row.interval_hours) for row in loaded.aligned.funding if row.interval_hours
    )
    settlements_per_year = Counter(_iso(i)[:4] for i in funding_instants)
    blocks = held_window_mark_coverage(loaded.aligned)

    records = [p.as_dict() for p in loaded.provenance]
    blob = json.dumps(
        sorted(records, key=lambda r: (str(r["field"]), str(r["period"]))),
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "schema": CLOSURE_SCHEMA,
        "checkpoint": "P13",
        "active_design": ACTIVE_DESIGN,
        "preregistration_hash": preregistration_hash(),
        "acquisition_manifest_digest": manifest["manifest_digest"],
        "acquisition_plan_digest": manifest["plan_digest"],
        "archive_host": manifest["archive_host"],
        "span_start_inclusive": DATA_BOUNDARY["span_start_inclusive"],
        "research_boundary_exclusive": DATA_BOUNDARY["span_end_exclusive"],
        "objects_read": len(loaded.provenance),
        "families": families,
        "grid_coverage": coverage,
        "mark_publication_coverage": {
            "months_planned": len(loaded.periods_by_field.get("mark_price", ())),
            "months_published": len(loaded.published_mark_periods),
            "published_months": list(loaded.published_mark_periods),
            "unpublished_months": sorted(
                set(loaded.periods_by_field.get("mark_price", ()))
                - set(loaded.published_mark_periods)
            ),
            "what_this_drives": (
                "MARK_PRICE_FALLBACK's per-object funding-notional substitution, and "
                "NOTHING else. Under A2R2 mark availability never moves an opening "
                "instant, and it never authorises a liquidation touch."
            ),
        },
        "funding_source_coverage": {
            "settlements_in_source": len(funding_instants),
            "first_settlement_utc": _iso(min(funding_instants)) if funding_instants else None,
            "last_settlement_utc": _iso(max(funding_instants)) if funding_instants else None,
            "distinct_settlement_instants": len(set(funding_instants)),
            "repeated_settlement_instants": len(funding_instants) - len(set(funding_instants)),
            "published_interval_hours": dict(sorted(intervals.items())),
            "settlements_by_calendar_year": dict(sorted(settlements_per_year.items())),
            "this_is_a_source_count": (
                "the number of settlement ROWS the archive publishes in the span. It is "
                "NOT the number a block applies, which depends on each block's holding "
                "window and is an economic quantity this closure does not compute."
            ),
        },
        "research_boundary_truncation": {
            "rows_dropped_at_or_after_boundary": sum(
                p.rows_dropped_at_boundary for p in loaded.provenance
            ),
            "by_field": {
                field: families[field]["rows_dropped_at_boundary"] for field in families
            },
            "boundary": DATA_BOUNDARY["span_end_exclusive"],
            "rule": (
                "the ONE boundary-straddling month is truncated at load and every other "
                "month carrying a boundary-crossing row is refused outright"
            ),
            "max_surviving_instant_utc": max(
                (
                    families[field]["last_instant_utc"]
                    for field in families
                    if families[field]["last_instant_utc"]
                ),
                default=None,
            ),
        },
        "a2r2_held_window_source_coverage": blocks,
        "closure_digest": "sha256:" + hashlib.sha256(blob.encode()).hexdigest(),
        "objects": sorted(records, key=lambda r: (str(r["field"]), str(r["period"]))),
        "provenance": provenance or {},
        "what_was_not_computed": [
            "funding PnL",
            "basis PnL",
            "net PnL",
            "block returns",
            "maximum adverse excursion economics",
            "G1-G6",
            "S1-S4 economic outputs",
            "D1-D3 economic outputs",
            "VIABLE / NOT VIABLE",
            "any governed decision",
        ],
        "what_this_is_not": (
            "a result. This record describes SOURCES: which objects exist, what their "
            "bytes hash to, what units their timestamps carry, where their holes are and "
            "how many settlements they publish. No P13 economic number was computed, the "
            "governed screen was never called over this history, and CURRENT_RESULT_STATE "
            "remains NOT YET RUN."
        ),
    }


def _iso(instant_ns: int) -> str:
    return (
        datetime.fromtimestamp(instant_ns / 1_000_000_000, tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "+00:00")
    )
