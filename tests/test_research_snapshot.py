"""The committed research snapshot is verifiable, and every way it can rot is caught.

Two properties, and the second is what makes the first worth having.

The committed snapshot under ``data/research/`` must verify. That test is the
regression guard: it is the only thing standing between a fresh clone and a
research snapshot whose manifest quietly stopped describing its files.

A verifier that only ever sees a good snapshot proves nothing, though — a
function returning ``True`` would pass that test. So every failure mode the
verifier claims to catch is exercised here against a *corrupted copy* built in
``tmp_path``: the four real files are copied out, exactly one thing is changed,
and the verifier is required to reject it by name. The committed files are never
written to.

Corruptions that push a timestamp to or past the seal build a synthetic row from
a duplicated pre-Styx one. That is fabricated data with a chosen timestamp, not
sealed market data: this repository holds no candle at or after the seal, and
this suite reads none.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import pytest

from tools.verify_research_snapshot import (
    SnapshotVerificationError,
    main,
    verify_snapshot,
)

REPO = Path(__file__).resolve().parents[1]
COMMITTED = REPO / "data" / "research" / "btc_usdt_1h_gen1_snapshot_manifest.json"

#: The checks the verifier runs, in order. Pinned so that deleting one is a test
#: failure rather than a quietly shorter report.
EXPECTED_CHECKS = (
    "manifest_readable",
    "snapshot_schema",
    "manifest_keys",
    "contract_id",
    "contract_hash",
    "contract_sealed_test_start",
    "contains_styx",
    "styx_rows_exported",
    "safety_check_flags",
    "files_present",
    "raw_sha256",
    "processed_sha256",
    "metadata_sha256",
    "raw_before_styx",
    "processed_before_styx",
    "raw_rows",
    "processed_rows",
    "processed_row_range",
    "raw_span",
    "processed_span",
    "raw_semantic_hash",
    "processed_semantic_prefix_hash",
    "outer_coverage_rows",
)

#: A timestamp at the seal itself, to prove the boundary is closed at the top.
AT_THE_SEAL = "2025-08-27T23:00:00+00:00"


@pytest.fixture
def snapshot(tmp_path: Path) -> Path:
    """A writable copy of the committed snapshot, laid out as the repository is."""
    research = tmp_path / "data" / "research"
    research.mkdir(parents=True)
    for path in COMMITTED.parent.iterdir():
        shutil.copy2(path, research / path.name)
    return research / COMMITTED.name


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def edit(manifest: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    payload = json.loads(manifest.read_text())
    mutate(payload)
    manifest.write_text(json.dumps(payload, indent=2) + "\n")


def rejects(manifest: Path, check: str) -> SnapshotVerificationError:
    with pytest.raises(SnapshotVerificationError) as excinfo:
        verify_snapshot(manifest)
    assert excinfo.value.check == check
    return excinfo.value


def raw_path(manifest: Path) -> Path:
    block = json.loads(manifest.read_text())["raw_pre_styx"]
    return manifest.parents[2] / block["path"]


def processed_path(manifest: Path) -> Path:
    block = json.loads(manifest.read_text())["processed_outer_coverage"]
    return manifest.parents[2] / block["path"]


def rewrite(frame: pd.DataFrame, path: Path, manifest: Path, block: str) -> None:
    """Replace a Parquet and refresh only its sha256, leaving other claims stale.

    The point of most corruptions is to reach a check *past* the byte hash, so
    the byte hash is repaired deliberately and nothing else is.
    """
    frame.to_parquet(path, index=False)
    edit(manifest, lambda payload: payload[block].update(sha256=sha256(path)))


def append_synthetic_row(frame: pd.DataFrame, when: str) -> pd.DataFrame:
    """Duplicate the last row at a chosen timestamp. Fabricated, never sealed data."""
    extra = frame.iloc[[-1]].copy()
    extra["date"] = pd.Timestamp(when)
    return pd.concat([frame, extra], ignore_index=True)


def test_committed_snapshot_verifies() -> None:
    results = verify_snapshot(COMMITTED)
    assert tuple(result.name for result in results) == EXPECTED_CHECKS


def test_snapshot_copy_verifies_away_from_the_repository(snapshot: Path) -> None:
    """Nothing outside data/research/ and the committed contracts is needed."""
    assert tuple(result.name for result in verify_snapshot(snapshot)) == EXPECTED_CHECKS


def test_cli_reports_and_exits_zero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["verify_research_snapshot"])
    assert main() == 0
    out = capsys.readouterr().out
    assert out.count("PASS") == len(EXPECTED_CHECKS)
    assert "raw_before_styx" in out


def test_cli_exits_non_zero_on_rejection(
    snapshot: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    edit(snapshot, lambda payload: payload.update(contains_styx=True))
    monkeypatch.setattr("sys.argv", ["verify_research_snapshot", "--manifest", str(snapshot)])
    assert main() == 1
    assert "contains_styx" in capsys.readouterr().out


def test_missing_manifest_is_rejected(snapshot: Path) -> None:
    snapshot.unlink()
    rejects(snapshot, "manifest_readable")


def test_unparseable_manifest_is_rejected(snapshot: Path) -> None:
    snapshot.write_text("{ this is not json")
    rejects(snapshot, "manifest_readable")


def test_wrong_snapshot_schema_is_rejected(snapshot: Path) -> None:
    edit(
        snapshot, lambda payload: payload.update(snapshot_schema="chimera.research-snapshot/2")
    )
    rejects(snapshot, "snapshot_schema")


def test_missing_manifest_key_is_rejected(snapshot: Path) -> None:
    edit(snapshot, lambda payload: payload["raw_pre_styx"].pop("semantic_hash"))
    rejects(snapshot, "manifest_keys")


def test_wrong_typed_manifest_key_is_rejected(snapshot: Path) -> None:
    edit(snapshot, lambda payload: payload["raw_pre_styx"].update(rows="49551"))
    rejects(snapshot, "manifest_keys")


def test_unknown_contract_id_is_rejected(snapshot: Path) -> None:
    edit(snapshot, lambda payload: payload["contract"].update(contract_id="btc-usdt-1h-gen2"))
    rejects(snapshot, "contract_id")


def test_wrong_contract_hash_is_rejected(snapshot: Path) -> None:
    edit(snapshot, lambda payload: payload["contract"].update(contract_hash="00" * 32))
    rejects(snapshot, "contract_hash")


def test_wrong_sealed_instant_is_rejected(snapshot: Path) -> None:
    edit(
        snapshot,
        lambda payload: payload["contract"].update(
            sealed_test_start="2025-09-27T23:00:00+00:00"
        ),
    )
    rejects(snapshot, "contract_sealed_test_start")


def test_contains_styx_true_is_rejected(snapshot: Path) -> None:
    edit(snapshot, lambda payload: payload.update(contains_styx=True))
    rejects(snapshot, "contains_styx")


def test_non_zero_styx_rows_exported_is_rejected(snapshot: Path) -> None:
    edit(snapshot, lambda payload: payload["safety_checks"].update(styx_rows_exported=1))
    rejects(snapshot, "styx_rows_exported")


def test_false_safety_flag_is_rejected(snapshot: Path) -> None:
    edit(
        snapshot,
        lambda payload: payload["safety_checks"].update(raw_strictly_before_styx=False),
    )
    rejects(snapshot, "safety_check_flags")


def test_missing_referenced_file_is_rejected(snapshot: Path) -> None:
    raw_path(snapshot).unlink()
    rejects(snapshot, "files_present")


def test_raw_byte_corruption_is_rejected(snapshot: Path) -> None:
    path = raw_path(snapshot)
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    path.write_bytes(bytes(data))
    rejects(snapshot, "raw_sha256")


def test_processed_byte_corruption_is_rejected(snapshot: Path) -> None:
    path = processed_path(snapshot)
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    path.write_bytes(bytes(data))
    rejects(snapshot, "processed_sha256")


def test_metadata_corruption_is_rejected(snapshot: Path) -> None:
    path = processed_path(snapshot).with_suffix(".parquet.meta.json")
    path.write_text(path.read_text() + " ")
    rejects(snapshot, "metadata_sha256")


def test_raw_row_at_the_seal_is_rejected(snapshot: Path) -> None:
    path = raw_path(snapshot)
    rewrite(
        append_synthetic_row(pd.read_parquet(path), AT_THE_SEAL),
        path,
        snapshot,
        "raw_pre_styx",
    )
    error = rejects(snapshot, "raw_before_styx")
    assert "1 of 49552 rows" in str(error)


def test_processed_row_at_the_seal_is_rejected(snapshot: Path) -> None:
    path = processed_path(snapshot)
    rewrite(
        append_synthetic_row(pd.read_parquet(path), AT_THE_SEAL),
        path,
        snapshot,
        "processed_outer_coverage",
    )
    rejects(snapshot, "processed_before_styx")


def test_seal_breach_never_names_the_offending_rows(snapshot: Path) -> None:
    """A leak-proof rejection: the count, never the candle."""
    path = raw_path(snapshot)
    frame = append_synthetic_row(pd.read_parquet(path), AT_THE_SEAL)
    frame.loc[frame.index[-1], "close"] = 123456.75
    rewrite(frame, path, snapshot, "raw_pre_styx")
    message = str(rejects(snapshot, "raw_before_styx"))
    assert "123456" not in message


def test_wrong_raw_row_count_is_rejected(snapshot: Path) -> None:
    edit(snapshot, lambda payload: payload["raw_pre_styx"].update(rows=49550))
    rejects(snapshot, "raw_rows")


def test_wrong_processed_row_count_is_rejected(snapshot: Path) -> None:
    edit(snapshot, lambda payload: payload["processed_outer_coverage"].update(rows=45801))
    rejects(snapshot, "processed_rows")


def test_row_range_disagreeing_with_row_count_is_rejected(snapshot: Path) -> None:
    edit(
        snapshot,
        lambda payload: payload["processed_outer_coverage"].update(row_range=[0, 45801]),
    )
    rejects(snapshot, "processed_row_range")


def test_wrong_raw_start_is_rejected(snapshot: Path) -> None:
    edit(
        snapshot,
        lambda payload: payload["raw_pre_styx"].update(start="2020-01-02T00:00:00+00:00"),
    )
    rejects(snapshot, "raw_span")


def test_wrong_processed_end_is_rejected(snapshot: Path) -> None:
    edit(
        snapshot,
        lambda payload: payload["processed_outer_coverage"].update(
            end="2025-05-19T08:00:00+00:00"
        ),
    )
    rejects(snapshot, "processed_span")


def test_altered_raw_candle_is_rejected(snapshot: Path) -> None:
    """Same bytes count, same span, different market. Only the semantic hash sees it."""
    path = raw_path(snapshot)
    frame = pd.read_parquet(path)
    frame.loc[frame.index[10], "close"] = float(frame.loc[frame.index[10], "close"]) + 1.0
    rewrite(frame, path, snapshot, "raw_pre_styx")
    rejects(snapshot, "raw_semantic_hash")


def test_altered_processed_feature_is_rejected(snapshot: Path) -> None:
    path = processed_path(snapshot)
    frame = pd.read_parquet(path)
    frame.loc[frame.index[7], "rsi_centered"] = 0.5
    rewrite(frame, path, snapshot, "processed_outer_coverage")
    rejects(snapshot, "processed_semantic_prefix_hash")


def test_outer_coverage_past_the_research_region_is_rejected(snapshot: Path) -> None:
    edit(
        snapshot,
        lambda payload: payload["canonical_reference"].update(max_outer_end_row=48218),
    )
    rejects(snapshot, "outer_coverage_rows")


def test_processed_snapshot_too_short_for_outer_coverage_is_rejected(snapshot: Path) -> None:
    edit(
        snapshot,
        lambda payload: payload["canonical_reference"].update(max_outer_end_row=45803),
    )
    rejects(snapshot, "outer_coverage_rows")
