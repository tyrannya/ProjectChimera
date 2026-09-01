"""Acquiring P13's four preregistered archive families, and refusing everything else.

``tools/acquire_p13_sources.py`` owns the PLAN — which 260 objects the frozen
design requires and where each one lives — and has always been able to compute it
without a network. This module owns what happens once bytes are actually fetched:
the published checksum is parsed and COMPARED, the single member is extracted, and
both digests are recorded so the object's identity is reconstructible from the
manifest rather than trusted.

**One host, enforced structurally.** :func:`assert_allowed_url` refuses any URL
whose host is not ``data.binance.vision``, and every fetch goes through it. That
is the frozen ``VENUE_POLICY`` and ``DATA_SOURCES`` made executable: no alternate
venue, no REST endpoint standing in for the historical archive, no third-party
mirror, and no authenticated endpoint — the archive is public and no credential is
ever presented.

**A mismatch is a refusal.** ``nn.p13_sources.verify_published_checksum`` raises
rather than returning a flag, and this module does not catch it. An object whose
bytes disagree with the digest its publisher publishes for it is not acquired,
not cached as good, and not described in a manifest.

**This module computes no economics.** It reads bytes and describes them. It never
opens a position, never applies a settlement, and never evaluates a gate; the
functions that would are in :mod:`nn.p13_screen`, :mod:`nn.p13_blocks` and
:mod:`nn.p13_carry`, and nothing here imports them.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence
from urllib.parse import urlparse

from nn.p13_sources import (
    CHECKSUM_VERIFIED,
    SourceError,
    extract_single_member,
    verify_published_checksum,
)

__all__ = [
    "AcquisitionError",
    "ACQUISITION_SCHEMA",
    "ALLOWED_HOST",
    "AcquiredObject",
    "assert_allowed_url",
    "parse_checksum_companion",
    "acquire_object",
    "acquire_all",
    "acquisition_manifest",
]

ACQUISITION_SCHEMA = "chimera.p13-source-acquisition/1"

#: The ONE host P13 may fetch from. Not a default and not a preference: the frozen
#: design names ``https://data.binance.vision`` as the archive host, and
#: ``docs/p4_preregistration.md`` §3 already settled that a row sourced from
#: anywhere else is not a row of the preregistered historical source.
ALLOWED_HOST = "data.binance.vision"


class AcquisitionError(RuntimeError):
    """A required object cannot be obtained under the frozen rules.

    Raised rather than recorded-and-skipped. A partial acquisition that continued
    past a failure would produce a manifest describing a universe the screen does
    not have, and the missing months would be exactly the ones nobody looked at.
    """


def assert_allowed_url(url: str) -> None:
    """Refuse any URL that is not a public object on the frozen archive host.

    Checked on the PARSED host rather than by substring, because
    ``https://data.binance.vision.example.com/`` contains the allowed host as a
    substring and is a different server. The scheme is pinned to HTTPS and a
    userinfo component is refused outright — credentials have no place on a public
    archive, and their presence would be the shape of an authenticated request.
    """
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise AcquisitionError(f"{url}: only https is permitted, not {parsed.scheme!r}")
    if parsed.username or parsed.password:
        raise AcquisitionError(
            f"{url}: carries credentials. P13 reads a PUBLIC archive and presents no "
            "credential to any endpoint; an authenticated request is a different source."
        )
    if parsed.hostname != ALLOWED_HOST:
        raise AcquisitionError(
            f"{url}: host {parsed.hostname!r} is not {ALLOWED_HOST!r}. The frozen design "
            "names one archive host. No alternate venue, no REST endpoint, no mirror and "
            "no S3 origin may stand in for it without an explicit amendment."
        )


def parse_checksum_companion(text: str, object_name: str) -> str:
    """The sha256 Binance publishes for one object, from its ``.CHECKSUM`` file.

    The published format is ``<digest>  <filename>``, which is what
    ``sha256sum -c`` consumes. BOTH halves are checked: a digest whose companion
    names a DIFFERENT object would verify the wrong bytes against the wrong
    expectation and report success, which is the one failure a checksum exists to
    make impossible.
    """
    stripped = text.strip()
    if not stripped:
        raise AcquisitionError(f"{object_name}: the published .CHECKSUM is empty")
    parts = stripped.split()
    digest = parts[0].lower()
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise AcquisitionError(
            f"{object_name}: {parts[0]!r} is not a sha256 hex digest. The companion was "
            "not parsed as one rather than being coerced into one."
        )
    if len(parts) >= 2:
        named = parts[1].strip().rsplit("/", 1)[-1]
        if named != object_name:
            raise AcquisitionError(
                f"{object_name}: its .CHECKSUM names {named!r}. A companion that names a "
                "different object would verify one file's bytes against another file's "
                "digest, so it is refused rather than used."
            )
    return digest


@dataclass(frozen=True)
class AcquiredObject:
    """One published object, its companion digest, and both recomputed digests.

    Every field is one of ``SOURCE_FREEZE_FIELDS``. The record is what makes the
    acquisition reconstructible: an auditor with the manifest can re-fetch the
    object, recompute both digests, and disagree with this record if it is wrong.
    """

    field: str
    market: str
    data_type: str
    symbol: str
    interval: str | None
    period: str
    object_name: str
    url: str
    checksum_url: str
    #: The whole published object — the digest checkable against the publisher.
    archive_byte_size: int
    archive_sha256: str
    #: The publisher's own digest, and whether it was actually COMPARED. The state
    #: comes from ``nn.p13_sources.verify_published_checksum``, which raises on a
    #: mismatch, so a record can only exist with the state ``verified_match``.
    published_checksum: str
    checksum_state: str
    #: The single CSV inside the object — the bytes the rows are parsed from.
    member_name: str
    member_sha256: str
    member_byte_size: int
    acquired_at: str
    cache_path: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


Fetcher = Callable[[str], bytes]


def acquire_object(
    planned: Any,
    cache_dir: Path,
    *,
    fetch: Fetcher,
    reuse_cache: bool = True,
) -> AcquiredObject:
    """Obtain ONE object and its companion, verify it, and describe it.

    ``planned`` is a ``tools.acquire_p13_sources.PlannedObject``; it is taken
    structurally rather than by import so this module does not depend on the CLI.

    A cached object is REUSED but never trusted: its digest is recomputed and
    re-compared against the companion on every run, so a corrupted or replaced
    cache file fails exactly as a corrupted download would. That is the property
    that makes "do not delete acquired bytes after verification" safe.
    """
    assert_allowed_url(planned.url)
    assert_allowed_url(planned.checksum_url)

    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_path = cache_dir / planned.object_name
    companion_path = cache_dir / f"{planned.object_name}.CHECKSUM"

    for path, url in ((archive_path, planned.url), (companion_path, planned.checksum_url)):
        if reuse_cache and path.is_file() and path.stat().st_size > 0:
            continue
        path.write_bytes(fetch(url))

    raw = archive_path.read_bytes()
    published = parse_checksum_companion(
        companion_path.read_text(encoding="utf-8"), planned.object_name
    )
    # Raises on a mismatch. Not caught: an object whose bytes disagree with its
    # publisher is not acquired at all.
    state = verify_published_checksum(raw, published)
    if state != CHECKSUM_VERIFIED:  # pragma: no cover - defensive
        raise AcquisitionError(
            f"{planned.object_name}: verification returned {state!r} with both the bytes "
            "and the published digest in hand, which should be impossible"
        )

    member_name, member = extract_single_member(raw)
    return AcquiredObject(
        field=planned.field,
        market=planned.market,
        data_type=planned.data_type,
        symbol=planned.symbol,
        interval=planned.interval,
        period=planned.period,
        object_name=planned.object_name,
        url=planned.url,
        checksum_url=planned.checksum_url,
        archive_byte_size=len(raw),
        archive_sha256=hashlib.sha256(raw).hexdigest(),
        published_checksum=published,
        checksum_state=state,
        member_name=member_name,
        member_sha256=hashlib.sha256(member).hexdigest(),
        member_byte_size=len(member),
        acquired_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        cache_path=str(archive_path),
    )


def acquire_all(
    planned: Sequence[Any],
    cache_dir: Path,
    *,
    fetch: Fetcher,
    reuse_cache: bool = True,
    progress: Callable[[int, int, Any], None] | None = None,
) -> list[AcquiredObject]:
    """Every planned object, or a refusal naming the first one that failed.

    **It stops.** A source-access failure is not a month to be skipped, an object
    to be marked optional, or a hole to be filled from somewhere else: it is a
    reason to stop and report, because a screen assembled from the months that
    happened to be reachable is a screen over a universe the design did not
    specify. Everything already fetched and verified stays in the cache.
    """
    acquired: list[AcquiredObject] = []
    for index, item in enumerate(planned, start=1):
        try:
            acquired.append(
                acquire_object(item, cache_dir, fetch=fetch, reuse_cache=reuse_cache)
            )
        except (SourceError, AcquisitionError, OSError) as exc:
            raise AcquisitionError(
                f"acquisition STOPPED at object {index} of {len(planned)}: "
                f"{item.field} {item.period} ({item.object_name}) — {type(exc).__name__}: "
                f"{exc}. {len(acquired)} object(s) were acquired and verified before this "
                "one and are preserved in the cache. Nothing is skipped, substituted or "
                "marked optional, and the acquisition is NOT complete."
            ) from exc
        if progress is not None:
            progress(index, len(planned), acquired[-1])
    return acquired


def acquisition_manifest(
    acquired: Sequence[AcquiredObject],
    *,
    symbol: str,
    plan_digest: str,
    planned_count: int,
    cache_dir: str,
    active_design: str,
    preregistration_hash: str,
    span_start_inclusive: str,
    span_end_exclusive: str,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The frozen record of what was obtained, deterministic in its own content.

    ``manifest_digest`` covers the OBJECT RECORDS only — not the wall-clock
    acquisition timestamps, which differ between two runs that fetched identical
    bytes. Two acquisitions of the same published objects therefore produce the
    same digest, which is what makes the manifest checkable rather than merely
    present.
    """
    records = [obj.as_dict() for obj in acquired]
    stable = [
        {k: v for k, v in record.items() if k not in ("acquired_at", "cache_path")}
        for record in records
    ]
    blob = json.dumps(
        sorted(stable, key=lambda r: (r["field"], r["period"])),
        sort_keys=True,
        separators=(",", ":"),
    )
    by_field: dict[str, int] = {}
    for record in records:
        by_field[record["field"]] = by_field.get(record["field"], 0) + 1
    verified = sum(1 for r in records if r["checksum_state"] == CHECKSUM_VERIFIED)
    return {
        "schema": ACQUISITION_SCHEMA,
        "checkpoint": "P13",
        "active_design": active_design,
        "preregistration_hash": preregistration_hash,
        "archive_host": ALLOWED_HOST,
        "symbol": symbol,
        "span_start_inclusive": span_start_inclusive,
        "span_end_exclusive": span_end_exclusive,
        "plan_digest": plan_digest,
        "planned_object_count": planned_count,
        "acquired_object_count": len(records),
        "complete": len(records) == planned_count,
        "objects_by_field": by_field,
        "checksum_verified_count": verified,
        "checksum_unverified_count": len(records) - verified,
        "total_archive_bytes": sum(r["archive_byte_size"] for r in records),
        "total_member_bytes": sum(r["member_byte_size"] for r in records),
        "cache_dir": cache_dir,
        "manifest_digest": "sha256:" + hashlib.sha256(blob.encode()).hexdigest(),
        "provenance": provenance or {},
        "what_this_is_not": (
            "a result. This manifest records which published objects were obtained and "
            "what their bytes hash to. No P13 return, funding total, basis figure, block "
            "result or gate condition is computed here, and none may be inferred from it."
        ),
        "objects": sorted(records, key=lambda r: (r["field"], r["period"])),
    }


def total_bytes(acquired: Iterable[AcquiredObject]) -> int:
    return sum(obj.archive_byte_size for obj in acquired)
