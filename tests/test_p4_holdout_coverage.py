"""P4-HOLD's archive-day coverage: the pre-fit plumbing, and what it must refuse.

``nn.p2b.load_holdout_coverage`` has always said that P4-HOLD's coverage is
established from archive-day metadata by ``tools.export_derivatives_snapshot
--probe``, and the probe had always documented itself as writing nothing. The
gate therefore reported P4-HOLD unavailable for the reason "coverage file
absent" — which is a statement about missing plumbing and not about the archive.
These tests are about the plumbing: that the probe establishes the claim from
publication metadata alone, that it refuses to invent one, and that every way of
not knowing still reads as unavailable.

**Nothing here reaches the network and nothing here reads a P4-HOLD row.** The
transport is replaced by a recorder that fails the test if anything but a HEAD
request is issued, or if a request names a day outside the region.
"""

from __future__ import annotations

import json
import urllib.error
from pathlib import Path

import pytest

import tools.export_derivatives_snapshot as exporter
from nn.p2b import DEFAULT_HOLDOUT_COVERAGE, load_holdout_coverage
from nn.p4_holdout import (
    COVERAGE_PATH,
    COVERAGE_QUERY_METHOD,
    COVERAGE_SCHEMA,
    HOLDOUT_ROWS,
    coverage_semantic_hash,
    expected_coverage_binding,
    holdout_archive_days,
    read_ledger,
)
from nn.p4_preregistration import preregistration_hash
from nn.p4_stage1 import describe, read_authorisation
from nn.p4_universe import availability_gate

ROOT = Path(__file__).resolve().parent.parent
OHLCV_MANIFEST = ROOT / "data" / "research" / "btc_usdt_1h_gen1_snapshot_manifest.json"

#: The hash the active design carries. Pinned here as well as in
#: ``tests/test_p4_preregistration.py`` because this change is machinery and not
#: science: if plumbing a coverage file through moved the preregistration hash,
#: something in the hashed payload moved with it, and that is a different commit.
ACTIVE_HASH = "b52ce5dda17ff065bd70d4f4a62ef6b0e221dd18ed2cf4c0f20017ec3bae59a7"

#: Two available exploratory blocks: the minimum §3.6 requires, so that the gate's
#: verdict below is a function of the holdout entry and of nothing else.
TWO_AVAILABLE_BLOCKS = [
    {"label": "outer_block_0", "available": True},
    {"label": "outer_block_1", "available": True},
]


def _day_of(url: str) -> str:
    """The UTC day a metrics archive URL names, e.g. ``...-metrics-2025-05-19.zip``."""
    parts = url.rsplit("-", 3)
    return "-".join([parts[1], parts[2], parts[3].removesuffix(".zip")])


class _Response:
    """What a HEAD answer is: a status, and by protocol no body at all."""

    status = 200

    def read(self, *_args):
        return b""

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class Transport:
    """A recorder standing in for ``urllib``, and a tripwire on what it is asked.

    Every request is remembered with its method. ``published`` decides which days
    answer 200 and which 404; a day named in ``fails`` raises the way a timeout
    or a DNS failure does, which is what distinguishes "not published" from "not
    reached" in :func:`tools.export_derivatives_snapshot._request`.
    """

    def __init__(self, *, published: set[str] | None = None, fails: set[str] | None = None):
        self.published = published
        self.fails = fails or set()
        self.calls: list[tuple[str, str]] = []

    def __call__(self, request, timeout=None):
        url = request.full_url
        self.calls.append((request.get_method(), url))
        day = _day_of(url)
        if day in self.fails:
            raise TimeoutError(f"synthetic transport failure for {day}")
        if self.published is not None and day not in self.published:
            raise urllib.error.HTTPError(url, 404, "synthetic", {}, None)
        return _Response()

    @property
    def days_requested(self) -> list[str]:
        return [_day_of(url) for _, url in self.calls]


@pytest.fixture
def transport(monkeypatch):
    """Install a recorder and remove the retry sleeps, so a failure test is fast."""

    def install(**kwargs):
        recorder = Transport(**kwargs)
        monkeypatch.setattr(exporter.urllib.request, "urlopen", recorder)
        monkeypatch.setattr(exporter, "BACKOFF_SECONDS", ())
        return recorder

    return install


def _establish(generated_at: str = "2026-01-01T00:00:00Z") -> dict:
    return exporter.probe_holdout_coverage(
        ohlcv_manifest=OHLCV_MANIFEST, timeout=5, generated_at=generated_at
    )


def _probe(out_dir: Path, install, **kwargs) -> tuple[dict, Transport]:
    recorder = install(**kwargs)
    payload = _establish()
    exporter.write_holdout_coverage(payload, out_dir)
    return payload, recorder


# --- the region's own days ----------------------------------------------------
def test_the_required_days_are_the_days_the_region_spans():
    days = holdout_archive_days()
    first, last = (part.strip() for part in read_ledger()["region_span"].split(".."))
    assert days[0] == first[:10]
    assert days[-1] == last[:10]
    assert len(days) == 101
    assert days == sorted(set(days))
    assert HOLDOUT_ROWS == (45802, 48211)


def test_the_probe_writes_where_the_gate_reads(tmp_path, transport):
    """The two halves of the plumbing name the same file."""
    assert DEFAULT_HOLDOUT_COVERAGE == COVERAGE_PATH
    assert COVERAGE_PATH.name == exporter.HOLDOUT_COVERAGE_NAME
    _probe(tmp_path, transport, published=set(holdout_archive_days()))
    assert (tmp_path / DEFAULT_HOLDOUT_COVERAGE.name).is_file()


# --- absent is not available --------------------------------------------------
def test_absent_coverage_is_unavailable_coverage(tmp_path):
    entry = load_holdout_coverage(tmp_path / exporter.HOLDOUT_COVERAGE_NAME)
    assert entry["available"] is False
    assert entry["basis"] == "absent"
    assert "--probe" in entry["reasons"][0]
    gate = availability_gate(TWO_AVAILABLE_BLOCKS, entry)
    assert gate["gate_passed"] is False
    assert gate["outcome"]["reason_code"] == "insufficient_coverage"


# --- probe metadata makes the holdout evaluable -------------------------------
def test_probe_metadata_makes_holdout_availability_evaluable(tmp_path, transport):
    payload, _ = _probe(tmp_path, transport, published=set(holdout_archive_days()))
    assert payload["days_published"] == len(holdout_archive_days())
    assert payload["days_absent"] == []

    entry = load_holdout_coverage(tmp_path / exporter.HOLDOUT_COVERAGE_NAME)
    assert entry["available"] is True
    assert entry["label"] == "p4_hold"
    assert entry["block"] == list(HOLDOUT_ROWS)
    assert entry["rows"] == HOLDOUT_ROWS[1] - HOLDOUT_ROWS[0]
    assert entry["basis"] == "archive-day metadata only; no P4-HOLD row was read"

    gate = availability_gate(TWO_AVAILABLE_BLOCKS, entry)
    assert gate["gate_passed"] is True
    assert gate["holdout"]["available"] is True


def test_a_long_run_of_unpublished_days_is_still_unavailable(tmp_path, transport):
    """Establishing the coverage is not the same as the coverage passing."""
    days = holdout_archive_days()
    absent = set(days[10:14])
    payload, _ = _probe(tmp_path, transport, published=set(days) - absent)
    assert payload["days_absent"] == sorted(absent)

    entry = load_holdout_coverage(tmp_path / exporter.HOLDOUT_COVERAGE_NAME)
    assert entry["available"] is False
    assert entry["max_contiguous_missing_hours"] > 48
    assert availability_gate(TWO_AVAILABLE_BLOCKS, entry)["gate_passed"] is False


# --- metadata only ------------------------------------------------------------
def test_only_publication_metadata_is_consulted(tmp_path, transport):
    days = holdout_archive_days()
    payload, recorder = _probe(tmp_path, transport, published=set(days))

    assert recorder.calls, "the probe answered without asking the archive anything"
    assert {method for method, _ in recorder.calls} == {"HEAD"}
    assert sorted(set(recorder.days_requested)) == days
    assert all("/daily/metrics/" in url for _, url in recorder.calls)
    assert payload["reads"]["archive_bodies_requested"] == 0
    assert payload["reads"]["archive_rows_parsed"] == 0
    assert payload["reads"]["p4_hold_rows_read"] == 0


def test_no_p4_hold_row_contents_are_read(tmp_path, transport, monkeypatch):
    """Every reader of an archive's *contents* is made to explode, and none fires.

    A HEAD request carries no body by protocol, so the method assertion above is
    what proves nothing was fetched. These three cover the other direction: if
    the coverage probe ever grows a path that opens or parses a metrics archive,
    this test names it.
    """

    def forbidden(*_args, **_kwargs):
        raise AssertionError("the coverage probe read archive contents")

    monkeypatch.setattr(exporter, "read_metrics", forbidden)
    monkeypatch.setattr(exporter, "download_archive", forbidden)
    monkeypatch.setattr(exporter.zipfile, "ZipFile", forbidden)

    payload, _ = _probe(tmp_path, transport, published=set(holdout_archive_days()))
    assert set(payload["published_days"].values()) == {True}


def test_the_claim_is_bound_to_its_design_source_and_period(tmp_path, transport):
    payload, _ = _probe(tmp_path, transport, published=set(holdout_archive_days()))
    days = holdout_archive_days()

    assert payload["coverage_schema"] == COVERAGE_SCHEMA
    assert payload["preregistration_hash"] == preregistration_hash()
    assert payload["source"]["venue"] == "binance"
    assert payload["source"]["market_type"] == "um"
    assert payload["source"]["symbol"] == "BTCUSDT"
    assert payload["source"]["archive_family"] == "futures/um/daily/metrics"
    assert payload["source"]["archive_kind"] == "daily"
    assert payload["queried"]["first_day"] == days[0]
    assert payload["queried"]["last_day"] == days[-1]
    assert payload["queried"]["days"] == len(days)
    assert payload["region"]["rows"] == list(HOLDOUT_ROWS)
    assert payload["region"]["first_instant"].startswith(days[0])
    assert payload["region"]["last_instant"].startswith(days[-1])
    assert payload["generated_at"] == "2026-01-01T00:00:00Z"
    assert set(payload["published_days"]) == set(days)


def test_two_probes_of_the_same_archive_differ_only_in_when_they_ran(tmp_path, transport):
    first, _ = _probe(tmp_path, transport, published=set(holdout_archive_days()))
    second = _establish(generated_at="2026-06-30T12:34:56Z")

    assert second["generated_at"] != first["generated_at"]
    assert second["semantic_hash"] == first["semantic_hash"]
    assert {k: v for k, v in second.items() if k != "generated_at"} == {
        k: v for k, v in first.items() if k != "generated_at"
    }


def test_a_different_publication_answer_moves_the_semantic_hash(tmp_path, transport):
    days = holdout_archive_days()
    full, _ = _probe(tmp_path / "a", transport, published=set(days))
    holed, _ = _probe(tmp_path / "b", transport, published=set(days) - {days[5]})
    assert holed["semantic_hash"] != full["semantic_hash"]


# --- fail closed --------------------------------------------------------------
def test_a_transport_failure_is_not_an_unpublished_day(tmp_path, transport):
    """The distinction §3.0a is built on, applied to the one region nobody reads.

    Recording an unreachable day as absent would fabricate an unavailability
    verdict about P4-HOLD. The probe stops instead, and writes nothing.
    """
    days = holdout_archive_days()
    with pytest.raises(exporter.DerivativesExportError, match="not an archive the source"):
        _probe(tmp_path, transport, published=set(days), fails={days[7]})
    assert not (tmp_path / exporter.HOLDOUT_COVERAGE_NAME).exists()


def test_a_404_day_is_a_measurement_even_though_a_failure_is_not(tmp_path, transport):
    """The two answers a day can give, side by side, so neither collapses."""
    days = holdout_archive_days()
    payload, _ = _probe(tmp_path, transport, published=set(days) - {days[3]})
    assert payload["published_days"][days[3]] is False
    assert payload["published_days"][days[4]] is True
    assert payload["days_absent"] == [days[3]]


def test_incomplete_date_coverage_fails_closed(tmp_path, transport):
    """A file about eighty of the region's days is not a claim about the region."""
    days = holdout_archive_days()
    payload, _ = _probe(tmp_path, transport, published=set(days))
    for day in days[-21:]:
        payload["published_days"].pop(day)
    exporter.write_holdout_coverage(payload, tmp_path)

    entry = load_holdout_coverage(tmp_path / exporter.HOLDOUT_COVERAGE_NAME)
    assert entry["available"] is False
    assert entry["basis"] == "refused"
    assert "unaccounted for" in entry["reasons"][0]
    assert availability_gate(TWO_AVAILABLE_BLOCKS, entry)["gate_passed"] is False


def test_days_outside_the_region_are_not_coverage_of_it(tmp_path, transport):
    payload, _ = _probe(tmp_path, transport, published=set(holdout_archive_days()))
    payload["published_days"]["2019-01-01"] = True
    exporter.write_holdout_coverage(payload, tmp_path)

    entry = load_holdout_coverage(tmp_path / exporter.HOLDOUT_COVERAGE_NAME)
    assert entry["available"] is False
    assert "outside the region" in entry["reasons"][0]


def test_coverage_established_under_a_superseded_design_is_refused(tmp_path, transport):
    payload, _ = _probe(tmp_path, transport, published=set(holdout_archive_days()))
    payload["preregistration_hash"] = "0" * 64
    exporter.write_holdout_coverage(payload, tmp_path)

    entry = load_holdout_coverage(tmp_path / exporter.HOLDOUT_COVERAGE_NAME)
    assert entry["available"] is False
    assert "does not carry over" in entry["reasons"][0]


def test_a_document_under_another_schema_is_refused(tmp_path):
    path = tmp_path / exporter.HOLDOUT_COVERAGE_NAME
    path.write_text(
        json.dumps(
            {
                "coverage_schema": "something.else/1",
                "preregistration_hash": preregistration_hash(),
                "published_days": {day: True for day in holdout_archive_days()},
            }
        )
    )
    entry = load_holdout_coverage(path)
    assert entry["available"] is False
    assert "another schema" in entry["reasons"][0]


def test_a_bare_available_flag_cannot_pass_the_gate(tmp_path):
    """The shape a hand-written file would take, and the shape that is refused."""
    path = tmp_path / exporter.HOLDOUT_COVERAGE_NAME
    path.write_text(json.dumps({"label": "p4_hold", "available": True}))
    entry = load_holdout_coverage(path)
    assert entry["available"] is False
    assert availability_gate(TWO_AVAILABLE_BLOCKS, entry)["gate_passed"] is False


# --- the CLI wiring -----------------------------------------------------------
def test_probe_writes_the_coverage_file_and_says_where(tmp_path, monkeypatch, capsys):
    """``--probe`` no longer writes nothing, and the one file it writes is this.

    The field-schema half of the probe is replaced here rather than fixtured:
    what this asserts is the wiring — that ``--probe`` establishes the coverage
    and persists it under ``--out-dir`` — and that half needs real archives to
    say anything at all.
    """
    established = {"coverage_schema": COVERAGE_SCHEMA, "published_days": {}}
    monkeypatch.setattr(exporter, "probe_holdout_coverage", lambda **_: established)
    monkeypatch.setattr(exporter, "probe", lambda *_a, **_k: {"fields": {}})

    assert exporter.main(["--probe", "--out-dir", str(tmp_path)]) == 0
    written = tmp_path / exporter.HOLDOUT_COVERAGE_NAME
    assert json.loads(written.read_text()) == established
    out = capsys.readouterr().out
    assert f"p4_hold_coverage={written}" in out
    assert "write nothing" not in exporter.build_argparser().format_help()


def test_a_probe_that_cannot_establish_coverage_writes_nothing(tmp_path, monkeypatch):
    """Fail closed, and do not go on to measure schemas as if coverage were known."""

    def unreachable(**_kwargs):
        raise exporter.DerivativesExportError("synthetic transport failure")

    def refuse(*_args, **_kwargs):
        raise AssertionError("the probe went on measuring after failing closed")

    monkeypatch.setattr(exporter, "probe_holdout_coverage", unreachable)
    monkeypatch.setattr(exporter, "probe", refuse)

    with pytest.raises(exporter.DerivativesExportError):
        exporter.main(["--probe", "--out-dir", str(tmp_path)])
    assert list(tmp_path.iterdir()) == []


# --- one mutated field at a time ----------------------------------------------
#: Every mutation below is applied to a **pristine probe-generated record** and
#: its digest is **recomputed afterwards**, so nothing is caught by the digest
#: alone. What has to catch these is the binding check: each field is compared
#: against a value `expected_coverage_binding` derives from the ledger, the
#: preregistration and the published archive layout, never read from the file.
def _mutate(payload: dict, path: tuple, value) -> dict:
    forged = json.loads(json.dumps(payload))
    target = forged
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    forged["semantic_hash"] = coverage_semantic_hash(forged)
    return forged


def _refuses(tmp_path: Path, payload: dict) -> dict:
    exporter.write_holdout_coverage(payload, tmp_path)
    entry = load_holdout_coverage(tmp_path / exporter.HOLDOUT_COVERAGE_NAME)
    assert entry["available"] is False
    assert entry["basis"] == "refused"
    assert availability_gate(TWO_AVAILABLE_BLOCKS, entry)["gate_passed"] is False
    return entry


@pytest.fixture
def pristine(tmp_path, transport):
    """A record a real probe would have written, with every day published."""
    payload, _ = _probe(tmp_path, transport, published=set(holdout_archive_days()))
    return payload


def test_the_positive_control_twin_still_passes(tmp_path, pristine):
    """The other half of every rejection below: unmutated, this record is good."""
    exporter.write_holdout_coverage(pristine, tmp_path)
    entry = load_holdout_coverage(tmp_path / exporter.HOLDOUT_COVERAGE_NAME)
    assert entry["available"] is True
    assert availability_gate(TWO_AVAILABLE_BLOCKS, entry)["gate_passed"] is True
    assert pristine["semantic_hash"] == coverage_semantic_hash(pristine)


@pytest.mark.parametrize(
    "value",
    ["false", "true", 1, 0, None, [], {}, 1.0, "yes"],
    ids=["str-false", "str-true", "int-1", "int-0", "null", "list", "dict", "float", "yes"],
)
def test_a_publication_status_that_is_not_a_boolean_fails_closed(tmp_path, pristine, value):
    """Truthiness is not an answer. ``"false"`` is a true string; ``1`` is not a bool.

    A day whose status arrived as any of these is a day nobody measured, and
    reading it as a verdict about P4-HOLD would invent one.
    """
    day = holdout_archive_days()[9]
    entry = _refuses(tmp_path, _mutate(pristine, ("published_days", day), value))
    assert "not a boolean" in entry["reasons"][0]
    assert day in entry["reasons"][0]


@pytest.mark.parametrize(
    "field,value",
    [
        ("venue", "bybit"),
        ("symbol", "ETHUSDT"),
        ("market_type", "cm"),
        ("field", "funding_rate"),
        ("provider", "somewhere-else"),
        ("base_url", "https://example.invalid"),
        ("archive_family", "futures/um/monthly/metrics"),
        ("archive_kind", "monthly"),
        ("url_template", "{base}/data/spot/daily/metrics/{symbol}.zip"),
    ],
)
def test_a_record_about_another_source_is_not_coverage_of_this_one(
    tmp_path, pristine, field, value
):
    entry = _refuses(tmp_path, _mutate(pristine, ("source", field), value))
    assert "source block" in entry["reasons"][0]


@pytest.mark.parametrize(
    "field,value",
    [
        ("label", "outer_block_0"),
        ("rows", [45802, 48212]),
        ("rows", [45801, 48211]),
        ("first_instant", "2025-05-20T08:00:00+00:00"),
        ("last_instant", "2025-08-28T16:00:00+00:00"),
        ("stage_1_last_instant", "2025-05-19T08:00:00+00:00"),
    ],
)
def test_a_record_about_another_region_is_not_coverage_of_this_one(
    tmp_path, pristine, field, value
):
    """Shifted rows or a shifted boundary describe a region this is not."""
    entry = _refuses(tmp_path, _mutate(pristine, ("region", field), value))
    assert "region block" in entry["reasons"][0]


@pytest.mark.parametrize(
    "field,value",
    [
        ("method", "HTTP GET on each daily archive URL"),
        ("method", "HTTP HEAD"),
        ("first_day", "2025-05-20"),
        ("last_day", "2025-08-26"),
        ("days", 100),
        ("days", 102),
        ("transport_failure_is_not_an_absent_day", False),
    ],
)
def test_a_record_about_another_query_is_not_coverage_of_this_period(
    tmp_path, pristine, field, value
):
    """Including the flag: a probe that treats a failure as an absence is refused."""
    entry = _refuses(tmp_path, _mutate(pristine, ("queried", field), value))
    assert "queried block" in entry["reasons"][0]


def test_the_query_method_is_the_one_the_design_names(tmp_path, pristine):
    assert pristine["queried"]["method"] == COVERAGE_QUERY_METHOD
    assert expected_coverage_binding()["queried"]["method"] == COVERAGE_QUERY_METHOD


# --- the digest -----------------------------------------------------------------
def test_a_missing_semantic_hash_fails_closed(tmp_path, pristine):
    forged = json.loads(json.dumps(pristine))
    forged.pop("semantic_hash")
    entry = _refuses(tmp_path, forged)
    assert "semantic_hash" in entry["reasons"][0]


def test_a_mismatched_semantic_hash_fails_closed(tmp_path, pristine):
    """A record edited in one place and left inconsistent in another.

    The edit is to a field no other check reads — the provenance path — so the
    digest is the only thing standing between it and the gate.
    """
    forged = json.loads(json.dumps(pristine))
    forged["boundary_derived_from"] = "somewhere/else.json"
    entry = _refuses(tmp_path, forged)
    assert "different measurements" in entry["reasons"][0]


def test_a_stale_digest_left_behind_by_an_edit_fails_closed(tmp_path, pristine):
    entry = _refuses(tmp_path, {**json.loads(json.dumps(pristine)), "semantic_hash": "0" * 64})
    assert "different measurements" in entry["reasons"][0]


def test_generated_at_is_the_one_field_that_may_move(tmp_path, pristine):
    """Non-semantic, so re-stamping it neither breaks the digest nor the verdict."""
    restamped = json.loads(json.dumps(pristine))
    restamped["generated_at"] = "2027-12-31T23:59:59Z"
    assert coverage_semantic_hash(restamped) == pristine["semantic_hash"]
    exporter.write_holdout_coverage(restamped, tmp_path)
    entry = load_holdout_coverage(tmp_path / exporter.HOLDOUT_COVERAGE_NAME)
    assert entry["available"] is True


def test_day_counts_that_disagree_with_the_day_map_fail_closed(tmp_path, pristine):
    """The record must describe the measurement it reports."""
    entry = _refuses(tmp_path, _mutate(pristine, ("days_published",), 7))
    assert "day counts disagree" in entry["reasons"][0]


def test_the_writer_and_the_reader_derive_the_same_binding(pristine):
    """The bindings are not two hand-kept copies; they are one function's output."""
    binding = expected_coverage_binding()
    assert pristine["region"] == binding["region"]
    assert pristine["source"] == binding["source"]
    assert pristine["queried"] == binding["queried"]
    assert set(pristine["published_days"]) == set(binding["days"])


# --- what this change did not do ----------------------------------------------
def test_the_preregistration_hash_did_not_move():
    """No scientific rule changed, so the hashed payload must be byte-identical.

    Plumbing a coverage file through is machinery: no feature, window, clip,
    staleness bound, block geometry, availability threshold, gate criterion,
    stage-1 geometry or holdout rule is touched by it.
    """
    assert preregistration_hash() == ACTIVE_HASH
    authorisation = json.loads(
        (ROOT / "data" / "research" / "p4_stage1_authorisation.json").read_text()
    )
    assert authorisation["preregistration_hash"] == ACTIVE_HASH


def test_stage_one_is_authorised_but_no_fit_has_run():
    assert read_authorisation()["state"] == "authorised"
    described = describe()
    assert described["interlock"]["state"] == "authorised"
    assert described["fits_run"] == 0


def test_no_p4_fit_or_result_exists():
    assert not (ROOT / "artifacts" / "benchmark" / "btc_p4_comparison").exists()
    assert not list((ROOT / "artifacts" / "benchmark").glob("btc_p4_*"))


def test_the_holdout_ledger_and_styx_are_untouched():
    """Neither region moved, and neither was reached to establish the coverage.

    The ledger is still `unspent` and still names the same region, and the
    committed research snapshot still stops before both boundaries — which is
    what makes the probe's HEAD requests the only thing that looked at P4-HOLD's
    period at all, and Styx untouched by this change as by every other.
    """
    import pandas as pd

    from nn.research_contract import load_contract

    ledger = read_ledger()
    assert ledger["state"] == "unspent"
    assert ledger["checkpoint"] is None
    assert ledger["region"] == list(HOLDOUT_ROWS)
    assert ledger["evaluations_permitted"] == 1
    assert ledger["region_span"].startswith(holdout_archive_days()[0])

    contract = load_contract("btc-usdt-1h-gen1")
    manifest = json.loads(OHLCV_MANIFEST.read_text())
    assert manifest["contains_styx"] is False
    processed = manifest["processed_outer_coverage"]
    assert processed["row_range"] == [0, HOLDOUT_ROWS[0]]
    end = pd.Timestamp(processed["end"]).tz_convert("UTC")
    assert end < contract.sealed_test_start
    assert end < pd.Timestamp(ledger["region_span"].split("..")[0].strip())
