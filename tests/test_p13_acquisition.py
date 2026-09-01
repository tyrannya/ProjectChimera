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

import pytest

from nn import p13_acquisition as acq
from nn import p13_preregistration as prereg
from nn.p13_sources import CHECKSUM_VERIFIED, SourceError
from tests.p13_synthetic import kline_csv, kline_row_fields, ms, zip_bytes
from tools.acquire_p13_sources import plan_objects, plan_payload

PERIOD = "2021-03"
START = "2021-03-01T00:00:00+00:00"


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

    # Corrupt the cache and the SAME call must now refuse.
    (tmp_path / planned.object_name).write_bytes(_object("OTHER.csv"))
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
    assert (tmp_path / planned[0].object_name).is_file()
    assert (tmp_path / planned[1].object_name).is_file()


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
        cache_dir=str(tmp_path),
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
    assert first["cache_dir"] != second["cache_dir"]
    # And the digest MOVES when the bytes do.
    assert json.dumps(first, sort_keys=True) != json.dumps(second, sort_keys=True)


def test_the_manifest_states_that_it_is_not_a_result(tmp_path):
    """The key names the question and the value answers it: "a result"."""
    manifest = _manifest(tmp_path)
    disclaimer = manifest["what_this_is_not"]
    assert disclaimer.startswith("a result.")
    assert "No P13 return" in disclaimer
    assert "none may be inferred from it" in disclaimer


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
