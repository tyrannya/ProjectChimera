"""Acquisition and source closure, tested without a network.

Every byte here is constructed by the test and handed to the acquirer through an
injected fetcher. Nothing in this file opens a socket, and the two things it most
needs to prove — that a checksum is actually COMPARED, and that a mismatch is a
REFUSAL — are exactly the things a test against the real archive could not
demonstrate, because the real archive's objects match.

Nothing here computes a P13 economic quantity. The closure module is asserted, on
its own call graph, never to reach the governed screen.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath

import pytest

from nn import p13_acquisition as acq
from nn import p13_preregistration as prereg
from nn.p13_sources import CHECKSUM_VERIFIED, SourceError
from tests.p13_synthetic import kline_csv, kline_row_fields, ms, zip_bytes
from tools.acquire_p13_sources import plan_objects, plan_payload

PERIOD = "2021-03"
START = "2021-03-01T00:00:00+00:00"

#: Spelled as a code point so no test in this file has to escape one, and so a
#: reader cannot mistake an escaped backslash in a literal for a doubled one.
BACKSLASH = chr(92)


def _payload(count: int = 3) -> bytes:
    rows = [kline_row_fields(ms(START) + i * 3_600_000, "30000") for i in range(count)]
    return kline_csv(rows)


def _object(name: str = "BTCUSDT-1h-2021-03.csv") -> bytes:
    return zip_bytes(name, _payload())


class FakeArchive:
    """A published object and its companion, served without a network.

    ``served`` records every URL asked for, so a test can assert that the acquirer
    reached for the companion as well as the object and reached for nothing else.
    """

    def __init__(self, objects: dict[str, bytes], *, corrupt: set[str] | None = None):
        self.objects = objects
        self.corrupt = corrupt or set()
        self.served: list[str] = []

    def __call__(self, url: str) -> bytes:
        self.served.append(url)
        name = url.rsplit("/", 1)[-1]
        if name.endswith(".CHECKSUM"):
            target = name[: -len(".CHECKSUM")]
            raw = self.objects[target]
            digest = hashlib.sha256(raw).hexdigest()
            if target in self.corrupt:
                digest = "0" * 64
            return f"{digest}  {target}\n".encode()
        if name not in self.objects:
            raise OSError(f"HTTP 404 for {url}")
        return self.objects[name]


def _planned(field: str = "spot_price", period: str = PERIOD):
    """One PlannedObject from the frozen plan, so paths are never invented here."""
    return next(o for o in plan_objects() if o.field == field and o.period == period)


# ---------------------------------------------------------------------------
# The published checksum companion
# ---------------------------------------------------------------------------


def test_the_publisher_companion_format_is_parsed():
    digest = "ab" * 32
    assert acq.parse_checksum_companion(f"{digest}  X.zip\n", "X.zip") == digest
    assert acq.parse_checksum_companion(f"{digest.upper()}  X.zip", "X.zip") == digest
    # A bare digest, which some mirrors publish, is also accepted.
    assert acq.parse_checksum_companion(digest, "X.zip") == digest


def test_a_companion_naming_a_different_object_is_refused():
    """Otherwise one file's bytes are verified against another file's digest."""
    with pytest.raises(acq.AcquisitionError, match="names"):
        acq.parse_checksum_companion(f"{'ab' * 32}  OTHER.zip", "X.zip")


def test_a_companion_that_is_not_a_digest_is_refused_rather_than_coerced():
    for text in ("", "   ", "not-a-digest  X.zip", "abc  X.zip", "zz" * 32 + "  X.zip"):
        with pytest.raises(acq.AcquisitionError):
            acq.parse_checksum_companion(text, "X.zip")


# ---------------------------------------------------------------------------
# One host, and no authenticated endpoint
# ---------------------------------------------------------------------------


def test_only_the_frozen_archive_host_is_permitted():
    acq.assert_allowed_url("https://data.binance.vision/data/spot/monthly/x.zip")
    for url in (
        "https://api.binance.com/api/v3/klines",
        "https://fapi.binance.com/fapi/v1/fundingRate",
        "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision/x.zip",
        "https://data.binance.vision.example.com/x.zip",
        "https://example.com/data.binance.vision/x.zip",
    ):
        with pytest.raises(acq.AcquisitionError, match="host"):
            acq.assert_allowed_url(url)


def test_plaintext_and_credentialed_urls_are_refused():
    with pytest.raises(acq.AcquisitionError, match="https"):
        acq.assert_allowed_url("http://data.binance.vision/x.zip")
    with pytest.raises(acq.AcquisitionError, match="credentials"):
        acq.assert_allowed_url("https://key:secret@data.binance.vision/x.zip")


def test_every_planned_url_is_on_the_permitted_host():
    """The plan cannot name an object the fetcher would refuse to fetch."""
    for planned in plan_objects():
        acq.assert_allowed_url(planned.url)
        acq.assert_allowed_url(planned.checksum_url)


# ---------------------------------------------------------------------------
# Acquiring one object
# ---------------------------------------------------------------------------


def test_an_acquired_object_records_both_digests_and_a_real_verification(tmp_path):
    planned = _planned()
    raw = _object()
    fetch = FakeArchive({planned.object_name: raw})
    obtained = acq.acquire_object(planned, tmp_path, fetch=fetch)

    assert obtained.archive_sha256 == hashlib.sha256(raw).hexdigest()
    assert obtained.archive_byte_size == len(raw)
    assert obtained.published_checksum == obtained.archive_sha256
    assert obtained.checksum_state == CHECKSUM_VERIFIED
    assert obtained.member_name == "BTCUSDT-1h-2021-03.csv"
    assert obtained.member_sha256 == hashlib.sha256(_payload()).hexdigest()
    assert obtained.member_sha256 != obtained.archive_sha256
    assert obtained.object_name == "BTCUSDT-1h-2021-03.zip"
    assert obtained.period == PERIOD
    # Both the object AND its companion were fetched, and nothing else.
    assert len(fetch.served) == 2
    assert fetch.served[1].endswith(".CHECKSUM")


def test_a_checksum_mismatch_refuses_the_object(tmp_path):
    """The deliberate mismatch: the object is not acquired at all."""
    planned = _planned()
    raw = _object()
    fetch = FakeArchive({planned.object_name: raw}, corrupt={planned.object_name})
    with pytest.raises(SourceError, match="does not match"):
        acq.acquire_object(planned, tmp_path, fetch=fetch)


def test_an_archive_holding_two_members_is_refused(tmp_path):
    import io
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("a.csv", _payload())
        archive.writestr("b.csv", _payload())
    planned = _planned()
    fetch = FakeArchive({planned.object_name: buffer.getvalue()})
    with pytest.raises(SourceError, match="exactly one"):
        acq.acquire_object(planned, tmp_path, fetch=fetch)


def test_a_cached_object_is_reused_but_re_verified(tmp_path):
    """Caching never becomes trusting: the digest is recomputed every run."""
    planned = _planned()
    raw = _object()
    fetch = FakeArchive({planned.object_name: raw})
    acq.acquire_object(planned, tmp_path, fetch=fetch)
    assert len(fetch.served) == 2

    again = acq.acquire_object(planned, tmp_path, fetch=fetch)
    assert len(fetch.served) == 2, "a cached object was re-downloaded"
    assert again.checksum_state == CHECKSUM_VERIFIED

    # Corrupt the cache and the SAME call must now refuse. The file lives under
    # the PUBLISHER's path, not under a bare filename.
    acq.cache_location(tmp_path, planned).write_bytes(_object("OTHER.csv"))
    with pytest.raises(SourceError, match="does not match"):
        acq.acquire_object(planned, tmp_path, fetch=fetch)


# ---------------------------------------------------------------------------
# Acquiring the set: it stops, it never substitutes
# ---------------------------------------------------------------------------


def test_an_unobtainable_object_stops_the_acquisition(tmp_path):
    """No skipping, no optional months, no partial completeness."""
    planned = plan_objects()[:4]
    served = {p.object_name: _object() for p in planned}
    del served[planned[2].object_name]
    fetch = FakeArchive(served)
    with pytest.raises(acq.AcquisitionError) as raised:
        acq.acquire_all(planned, tmp_path, fetch=fetch)
    message = str(raised.value)
    assert "STOPPED at object 3 of 4" in message
    assert planned[2].object_name in message
    assert "2 object(s) were acquired" in message
    assert "NOT complete" in message
    # And what was already obtained is preserved rather than rolled back.
    assert acq.cache_location(tmp_path, planned[0]).is_file()
    assert acq.cache_location(tmp_path, planned[1]).is_file()


def test_the_acquirer_never_reaches_for_an_object_outside_the_plan(tmp_path):
    planned = plan_objects()[:3]
    fetch = FakeArchive({p.object_name: _object() for p in planned})
    acq.acquire_all(planned, tmp_path, fetch=fetch)
    permitted = {p.url for p in planned} | {p.checksum_url for p in planned}
    assert set(fetch.served) == permitted


# ---------------------------------------------------------------------------
# The manifest
# ---------------------------------------------------------------------------


def _manifest(tmp_path, count: int = 4) -> dict:
    planned = plan_objects()[:count]
    fetch = FakeArchive({p.object_name: _object() for p in planned})
    acquired = acq.acquire_all(planned, tmp_path, fetch=fetch)
    plan = plan_payload()
    return acq.acquisition_manifest(
        acquired,
        symbol="BTCUSDT",
        plan_digest=plan["plan_digest"],
        planned_count=count,
        active_design=prereg.ACTIVE_DESIGN,
        preregistration_hash=prereg.preregistration_hash(),
        span_start_inclusive=plan["span_start_inclusive"],
        span_end_exclusive=plan["span_end_exclusive"],
    )


def test_the_manifest_pins_the_active_design_and_hash(tmp_path):
    manifest = _manifest(tmp_path)
    assert manifest["active_design"] == "P13-A2R2"
    assert manifest["preregistration_hash"] == prereg.preregistration_hash()
    assert manifest["archive_host"] == "data.binance.vision"
    assert manifest["complete"] is True
    assert manifest["checksum_unverified_count"] == 0
    assert manifest["checksum_verified_count"] == manifest["acquired_object_count"]


def test_a_superseded_hash_can_never_govern_new_acquisition_evidence(tmp_path):
    """The manifest quotes the ACTIVE design, and the retired ones are refused."""
    from nn.p13_evidence import EvidenceError, assert_governing_hash

    manifest = _manifest(tmp_path)
    assert_governing_hash(manifest["preregistration_hash"])
    for entry in prereg.SUPERSEDED_HASHES:
        assert manifest["preregistration_hash"] != entry["hash"]
        with pytest.raises(EvidenceError, match="SUPERSEDED"):
            assert_governing_hash(entry["hash"])


def test_the_manifest_digest_is_deterministic_across_runs(tmp_path):
    """Two acquisitions of identical bytes produce an identical digest.

    The digest deliberately excludes the wall-clock acquisition time and the cache
    path, which differ between runs that obtained exactly the same objects.
    """
    first = _manifest(tmp_path / "a")
    second = _manifest(tmp_path / "b")
    assert first["manifest_digest"] == second["manifest_digest"]
    # Two DIFFERENT cache directories, and the digest does not notice — because no
    # machine-local path is part of it, or of the manifest at all.
    assert "cache_dir" not in first and "cache_dir" not in second
    # The one field deliberately excluded from the digest is the wall clock.
    assert all("acquired_at" in record for record in first["objects"])
    stripped = json.dumps(
        [{k: v for k, v in r.items() if k != "acquired_at"} for r in first["objects"]],
        sort_keys=True,
    )
    other = json.dumps(
        [{k: v for k, v in r.items() if k != "acquired_at"} for r in second["objects"]],
        sort_keys=True,
    )
    assert stripped == other


def test_the_manifest_states_that_it_is_not_a_result(tmp_path):
    """The key names the question and the value answers it: "a result"."""
    manifest = _manifest(tmp_path)
    disclaimer = manifest["what_this_is_not"]
    assert disclaimer.startswith("a result.")
    assert "No P13 return" in disclaimer
    assert "none may be inferred from it" in disclaimer


# ---------------------------------------------------------------------------
# Object IDENTITY is the publisher's path, not the object's name
# ---------------------------------------------------------------------------


def test_three_families_share_one_object_name_and_differ_only_by_path():
    """The fact the cache design has to survive, asserted before the behaviour.

    Binance names spot klines, USD-M perpetual klines and USD-M markPriceKlines
    identically. If this ever stops being true the collision guards below stop
    testing anything, and this test says so out loud.
    """
    for period in ("2020-01", "2021-07", "2025-05"):
        names = {
            _planned(field, period).object_name
            for field in ("spot_price", "perpetual_price", "mark_price")
        }
        paths = {
            _planned(field, period).path
            for field in ("spot_price", "perpetual_price", "mark_price")
        }
        assert len(names) == 1, "the families no longer share a filename"
        assert len(paths) == 3, "the families must differ by published path"


def test_the_cache_keys_on_the_published_path_so_families_cannot_collide(tmp_path):
    """**Regression guard.** A name-keyed cache silently acquired ONE family.

    With reuse enabled, the second and third families found the first family's
    file already present, skipped their own download entirely, verified the first
    family's bytes against the first family's companion — and passed. The manifest
    then recorded three families that were one family repeated three times.

    Here the three published objects carry DIFFERENT bytes. A cache that collapses
    them returns identical digests; a correct one returns three distinct ones and
    fetches three times.
    """
    period = "2021-07"
    families = ("spot_price", "perpetual_price", "mark_price")
    objects = {}
    planned = {}
    for index, field in enumerate(families):
        item = _planned(field, period)
        planned[field] = item
        # Distinct content per family, so a collision is visible as a digest.
        objects[item.path] = zip_bytes(
            f"{item.object_name[:-4]}.csv", _payload(count=index + 1)
        )

    class PathKeyedArchive(FakeArchive):
        """Serves by published PATH, which is what a real archive does."""

        def __call__(self, url: str) -> bytes:
            self.served.append(url)
            path = url.split("data.binance.vision/", 1)[1]
            if path.endswith(".CHECKSUM"):
                target = path[: -len(".CHECKSUM")]
                raw = self.objects[target]
                name = target.rsplit("/", 1)[-1]
                return f"{hashlib.sha256(raw).hexdigest()}  {name}\n".encode()
            return self.objects[path]

    fetch = PathKeyedArchive(objects)
    acquired = [
        acq.acquire_object(planned[field], tmp_path, fetch=fetch) for field in families
    ]

    digests = {obj.archive_sha256 for obj in acquired}
    assert len(digests) == 3, "three published objects collapsed onto one cache file"
    assert len({obj.archive_relative_path for obj in acquired}) == 3
    assert len({obj.object_name for obj in acquired}) == 1, "the premise changed"
    # Every family was actually FETCHED — six requests, not two.
    assert len(fetch.served) == 6, "a family was never downloaded at all"
    # And each landed at its own place on disk.
    for field in families:
        assert acq.cache_location(tmp_path, planned[field]).is_file()


def test_a_manifest_holding_two_records_for_one_published_path_is_refused(tmp_path):
    """Belt and braces: even if a cache collided, the manifest would not ship."""
    planned = _planned("spot_price", "2021-07")
    fetch = FakeArchive({planned.object_name: _object()})
    one = acq.acquire_object(planned, tmp_path, fetch=fetch)
    plan = plan_payload()
    with pytest.raises(acq.AcquisitionError, match="more than once"):
        acq.acquisition_manifest(
            [one, one],
            symbol="BTCUSDT",
            plan_digest=plan["plan_digest"],
            planned_count=2,
            active_design=prereg.ACTIVE_DESIGN,
            preregistration_hash=prereg.preregistration_hash(),
            span_start_inclusive=plan["span_start_inclusive"],
            span_end_exclusive=plan["span_end_exclusive"],
        )


# ---------------------------------------------------------------------------
# Canonical, platform-independent paths in committed evidence
# ---------------------------------------------------------------------------


def test_a_windows_spelling_renders_with_forward_slashes_at_the_string_boundary():
    """The rendering claim, checked where a Windows spelling actually means one.

    ``PureWindowsPath`` interprets backslashes as separators on EVERY platform,
    which a plain ``Path`` does not: on POSIX a backslash is an ordinary filename
    character, so ``Path("a\\b")`` is one file called ``a\\b`` rather than ``b``
    inside ``a``. This asserts the mechanism the canonicaliser relies on —
    ``as_posix()`` — without fabricating a filesystem object that does not exist.
    """
    windows = PureWindowsPath(
        r"artifacts\benchmark\btc_p13_a2r2_source_acquisition\acquisition_manifest.json"
    )
    assert windows.as_posix() == (
        "artifacts/benchmark/btc_p13_a2r2_source_acquisition/acquisition_manifest.json"
    )
    assert BACKSLASH not in windows.as_posix()
    # And the same relative path spelled POSIX-style renders identically, so the
    # canonical form does not depend on how it was written down.
    posix = PurePosixPath(
        "artifacts/benchmark/btc_p13_a2r2_source_acquisition/acquisition_manifest.json"
    )
    assert posix.as_posix() == windows.as_posix()


def test_posix_repo_relative_renders_a_real_file_with_forward_slashes(tmp_path):
    """The production canonicaliser, on a real nested file, on this platform."""
    root = tmp_path
    nested = root / "artifacts" / "benchmark" / "btc_p13_a2r2_source_acquisition"
    nested.mkdir(parents=True)
    target = nested / "acquisition_manifest.json"
    target.write_text("{}", encoding="utf-8")

    rendered = acq.posix_repo_relative(target, root)
    assert rendered == (
        "artifacts/benchmark/btc_p13_a2r2_source_acquisition/acquisition_manifest.json"
    )
    assert BACKSLASH not in rendered
    # ``str()`` on the same Path is what the canonicaliser exists to avoid; on
    # Windows it renders separators this must never emit.
    assert os.sep in str(target)


def test_natively_equivalent_spellings_of_one_file_render_identically(tmp_path):
    """Filesystem identity: same file, several legal spellings, one rendering.

    ``os.path.join`` uses the PLATFORM separator, so on Windows this genuinely
    feeds a backslash-spelled path through the canonicaliser and on POSIX a
    forward-slash one. Both are real paths to the real file, which is the
    difference between this and the witness it replaces.
    """
    root = tmp_path
    nested = root / "artifacts" / "benchmark" / "btc_p13_a2r2_source_acquisition"
    nested.mkdir(parents=True)
    target = nested / "acquisition_manifest.json"
    target.write_text("{}", encoding="utf-8")

    expected = acq.posix_repo_relative(target, root)
    spellings = [
        target,
        Path(
            os.path.join(
                str(root),
                "artifacts",
                "benchmark",
                "btc_p13_a2r2_source_acquisition",
                "acquisition_manifest.json",
            )
        ),
        root
        / "artifacts"
        / "."
        / "benchmark"
        / "btc_p13_a2r2_source_acquisition"
        / "acquisition_manifest.json",
        nested / ".." / "btc_p13_a2r2_source_acquisition" / "acquisition_manifest.json",
    ]
    for spelling in spellings:
        assert spelling.resolve() == target.resolve(), "the spelling names another file"
        assert acq.posix_repo_relative(spelling, root) == expected
        assert BACKSLASH not in acq.posix_repo_relative(spelling, root)


def test_the_evidence_manifest_is_identical_however_its_inputs_were_spelled(tmp_path):
    """Determinism: how a path was SPELLED must not change the bytes written.

    The spellings are all NATIVELY valid on the platform running the test —
    ``os.path.join`` supplies the platform separator, so on Windows this really is
    a backslash-spelled input — and every one denotes the same file. A synthetic
    Windows spelling on POSIX would denote a DIFFERENT file, which is why the
    earlier version of this witness failed on Ubuntu and was right to.
    """
    root = tmp_path
    nested = root / "artifacts" / "benchmark" / "btc_p13_a2r2_source_acquisition"
    nested.mkdir(parents=True)
    files = []
    for name in ("acquisition_manifest.json", "source_closure.json", "STATUS.md"):
        path = nested / name
        path.write_text(f"content of {name}", encoding="utf-8")
        files.append(path)

    canonical = acq.evidence_manifest_text(files, root=root)

    native = [
        Path(
            os.path.join(
                str(root), "artifacts", "benchmark", "btc_p13_a2r2_source_acquisition", f.name
            )
        )
        for f in files
    ]
    assert acq.evidence_manifest_text(native, root=root) == canonical

    detoured = [nested / ".." / nested.name / f.name for f in files]
    assert acq.evidence_manifest_text(detoured, root=root) == canonical

    # Shuffling the input order must not move a byte either: the manifest sorts.
    assert acq.evidence_manifest_text(list(reversed(files)), root=root) == canonical

    # And the emitted text is forward-slashed on EVERY platform, which is the
    # property the committed manifest depends on.
    assert BACKSLASH not in canonical
    for line in canonical.strip().splitlines():
        digest, name = line.split("  ", 1)
        assert len(digest) == 64
        assert name.startswith("artifacts/benchmark/")
        assert BACKSLASH not in name


def test_a_path_outside_the_repository_is_refused_rather_than_recorded(tmp_path):
    outside = tmp_path.parent / "elsewhere.json"
    outside.write_text("{}", encoding="utf-8")
    with pytest.raises(acq.AcquisitionError, match="outside"):
        acq.posix_repo_relative(outside, tmp_path)


def test_no_absolute_or_machine_local_path_reaches_the_manifest(tmp_path):
    """The cache lives on some machine; the evidence must not say which."""
    import re

    manifest = _manifest(tmp_path)
    assert "cache_dir" not in manifest
    assert "cache_location_is_not_recorded" in manifest

    drive = re.compile(r"^[A-Za-z]:[\\/]")
    offenders: list[tuple[str, str]] = []

    def walk(node, trail="manifest"):
        if isinstance(node, dict):
            for key, value in node.items():
                assert key != "cache_path", f"{trail}: a cache path reached the evidence"
                walk(value, f"{trail}.{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{trail}[{index}]")
        elif isinstance(node, str):
            if drive.match(node) or node.startswith("/") or "\\" in node:
                offenders.append((trail, node))

    walk(manifest)
    assert offenders == [], f"machine-local paths in the manifest: {offenders[:3]}"

    # And the identity that IS recorded is the publisher's own path.
    for record in manifest["objects"]:
        identity = record["archive_relative_path"]
        assert identity.startswith("data/")
        assert identity.endswith(".zip")
        assert "\\" not in identity
        assert record["url"].endswith(identity)


def test_the_manifest_records_the_published_path_as_each_objects_identity(tmp_path):
    manifest = _manifest(tmp_path, count=4)
    identities = [record["archive_relative_path"] for record in manifest["objects"]]
    assert len(set(identities)) == len(identities)
    for record in manifest["objects"]:
        assert record["archive_relative_path"].rsplit("/", 1)[-1] == record["object_name"]


# ---------------------------------------------------------------------------
# The plan the acquisition is bound to
# ---------------------------------------------------------------------------


def test_the_plan_is_exactly_the_frozen_span_and_object_count():
    plan = plan_payload()
    assert plan["object_count"] == 260
    assert plan["objects_by_field"] == {
        "spot_price": 65,
        "perpetual_price": 65,
        "mark_price": 65,
        "funding_settlement": 65,
    }
    periods = sorted({o.period for o in plan_objects()})
    assert periods[0] == "2020-01"
    assert periods[-1] == "2025-05"
    assert len(periods) == 65


def test_no_planned_object_begins_at_or_after_the_research_boundary():
    """The exclusive boundary, enforced in the plan rather than downstream."""
    boundary = prereg.DATA_BOUNDARY["span_end_exclusive"]
    assert boundary == "2025-05-19T08:00:00+00:00"
    for planned in plan_objects():
        assert planned.period <= "2025-05"
    assert not any(o.period >= "2025-06" for o in plan_objects())


def test_every_object_name_matches_binances_published_grammar():
    for planned in plan_objects():
        if planned.interval is None:
            expected = f"BTCUSDT-{planned.data_type}-{planned.period}.zip"
        else:
            expected = f"BTCUSDT-{planned.interval}-{planned.period}.zip"
        assert planned.object_name == expected
        assert planned.url.endswith(planned.path)
        assert planned.checksum_url == planned.url + ".CHECKSUM"


# ---------------------------------------------------------------------------
# The committed A2R2 acquisition evidence
# ---------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[1]
A2R2_EVIDENCE_DIR = REPO / "artifacts" / "benchmark" / "btc_p13_a2r2_source_acquisition"
A2R2_EVIDENCE_MANIFEST = REPO / "artifacts" / "btc_p13_a2r2_source_acquisition_SHA256SUMS.txt"

#: Independently stated, in a test, rather than discovered from the directory or
#: read out of the manifest — both of which change when someone deletes evidence,
#: which is the event this has to survive. Scoped to THIS checkpoint's own
#: acquisition output; the repository-wide evidence contract is not touched.
A2R2_REQUIRED_EVIDENCE = (
    "artifacts/benchmark/btc_p13_a2r2_source_acquisition/STATUS.md",
    "artifacts/benchmark/btc_p13_a2r2_source_acquisition/acquisition_manifest.json",
    "artifacts/benchmark/btc_p13_a2r2_source_acquisition/source_closure.json",
)


@pytest.mark.skipif(
    not A2R2_EVIDENCE_MANIFEST.is_file(), reason="acquisition evidence not committed yet"
)
def test_the_committed_a2r2_evidence_is_exactly_what_its_manifest_covers():
    """Deleting a file AND its manifest line must not pass unnoticed."""
    listed = [
        line.split("  ", 1)[1]
        for line in A2R2_EVIDENCE_MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert sorted(listed) == sorted(A2R2_REQUIRED_EVIDENCE)
    for name in A2R2_REQUIRED_EVIDENCE:
        assert (REPO / name).is_file(), f"{name} is missing"
    regenerated = acq.evidence_manifest_text(
        [REPO / name for name in A2R2_REQUIRED_EVIDENCE], root=REPO
    )
    assert regenerated == A2R2_EVIDENCE_MANIFEST.read_text(encoding="utf-8")


@pytest.mark.skipif(
    not A2R2_EVIDENCE_MANIFEST.is_file(), reason="acquisition evidence not committed yet"
)
def test_the_committed_manifest_uses_forward_slashes_only():
    text = A2R2_EVIDENCE_MANIFEST.read_text(encoding="utf-8")
    assert chr(92) not in text, "a backslash reached the committed evidence manifest"
    for line in text.strip().splitlines():
        digest, name = line.split("  ", 1)
        assert len(digest) == 64
        assert name.startswith("artifacts/")


@pytest.mark.skipif(
    not (A2R2_EVIDENCE_DIR / "acquisition_manifest.json").is_file(),
    reason="acquisition evidence not committed yet",
)
def test_the_committed_acquisition_evidence_leaks_no_machine_local_path():
    """A drive letter or home directory in committed evidence is not reproducible."""
    import re

    drive = re.compile(r"^[A-Za-z]:[\/]")
    for name in ("acquisition_manifest.json", "source_closure.json"):
        payload = json.loads((A2R2_EVIDENCE_DIR / name).read_text(encoding="utf-8"))
        offenders: list[str] = []

        def walk(node, trail=name):
            if isinstance(node, dict):
                for key, value in node.items():
                    walk(value, f"{trail}.{key}")
            elif isinstance(node, list):
                for index, value in enumerate(node):
                    walk(value, f"{trail}[{index}]")
            elif isinstance(node, str):
                if drive.match(node) or node.startswith("/") or chr(92) in node:
                    offenders.append(f"{trail} = {node[:60]!r}")

        walk(payload)
        assert offenders == [], f"machine-local paths in {name}: {offenders[:3]}"


@pytest.mark.skipif(
    not (A2R2_EVIDENCE_DIR / "acquisition_manifest.json").is_file(),
    reason="acquisition evidence not committed yet",
)
def test_the_committed_acquisition_is_complete_and_governed_by_the_active_design():
    manifest = json.loads(
        (A2R2_EVIDENCE_DIR / "acquisition_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["complete"] is True
    assert manifest["acquired_object_count"] == manifest["planned_object_count"] == 260
    assert manifest["checksum_verified_count"] == 260
    assert manifest["checksum_unverified_count"] == 0
    assert manifest["active_design"] == "P13-A2R2"
    assert manifest["preregistration_hash"] == prereg.preregistration_hash()
    # Every published path distinct: the collision that once made 260 into 130.
    paths = [record["archive_relative_path"] for record in manifest["objects"]]
    assert len(set(paths)) == 260


# ---------------------------------------------------------------------------
# Source closure computes no economics
# ---------------------------------------------------------------------------


def test_the_closure_module_never_calls_the_governed_screen():
    """Asserted on the CALL GRAPH, not on the docstring that promises it."""
    from nn import p13_source_closure

    tree = ast.parse(inspect.getsource(p13_source_closure))
    called: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                called.add(func.id)
            elif isinstance(func, ast.Attribute):
                called.add(func.attr)
    for forbidden in (
        "run_offline_screen",
        "run_screen",
        "run_block",
        "evaluate_block",
        "evaluate_gate",
        "run_stresses",
        "open_carry",
        "close_carry",
        "apply_funding",
        "build_quotes",
    ):
        assert forbidden not in called, f"the closure calls {forbidden}"


def test_the_closure_module_imports_nothing_that_could_reach_a_network():
    from nn import p13_source_closure

    tree = ast.parse(inspect.getsource(p13_source_closure))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    for forbidden in ("requests", "urllib", "urllib3", "http", "socket", "aiohttp", "httpx"):
        assert forbidden not in imported


def test_the_closure_records_the_units_it_resolved_per_object(tmp_path):
    """TIMESTAMP_UNIT_POLICY: recorded per object, never assumed."""
    from nn.p13_source_closure import family_summary
    from nn.p13_sources import read_kline_object

    raw = _object()
    table = read_kline_object(
        _payload(),
        field="spot_price",
        object_name="BTCUSDT-1h-2021-03.zip",
        period=PERIOD,
        raw_object=raw,
        member_name="BTCUSDT-1h-2021-03.csv",
    )
    summary = family_summary([table.provenance], "spot_price")
    assert summary["resolved_epoch_units"] == {"ms": 1}
    assert summary["objects_by_unit"] == {"ms": [PERIOD]}
    assert summary["rows_read"] == 3
