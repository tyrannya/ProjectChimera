"""The R0 governance records are immutable, and this is what keeps them so.

`docs/governance/r0_adoption_2026-09-14/` holds two externally produced
documents — the corrected independent audit and the final owner corrigendum —
imported byte for byte on 2026-09-14 when the owner adopted the corrected §37
master roadmap. Their SHA-256 digests are recorded in `SHA256SUMS.txt` beside
them and in the adoption record. Both files keep bytes the repository's own
hooks would otherwise rewrite (an intentional `================` separator in
the §37.0 phase map; three trailing spaces inside a quoted table), so each is
exempted by exact path in `test_no_merge_conflict_markers_remain` and in
`.pre-commit-config.yaml`. An exemption without a hash check would let the
bytes drift silently; this test is the other half of that arrangement.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORDS_DIR = ROOT / "docs" / "governance" / "r0_adoption_2026-09-14"
MANIFEST = RECORDS_DIR / "SHA256SUMS.txt"

#: The digests of the files exactly as the owner supplied them, computed with
#: `sha256sum` before import and again on the committed copies.
EXPECTED: dict[str, str] = {
    "docs/governance/r0_adoption_2026-09-14/"
    "ProjectChimera_full_audit_and_master_roadmap_2026-09-14_corrected.md": (
        "3bddc12e577f5e6e385860ef8484fc0ac023976e4bbc515fd0b8e8cff501e62f"
    ),
    "docs/governance/r0_adoption_2026-09-14/"
    "ProjectChimera_final_owner_corrigendum_and_adoption_ready_roadmap_2026-09-14.md": (
        "baaf2d0b7cfc4082aacc61a9ce7fa08d76036af044b50d2b2adcefd2b3d18797"
    ),
}

#: The line whose presence is the reason for the exact-path exemption in
#: `tests/test_config_and_cli.py`; if it ever disappears, the exemption should
#: go with it.
PHASE_MAP_SEPARATOR = "================  REAL-MONEY AUTHORITY BOUNDARY (four conditions)"


def _manifest_entries() -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        digest, _, path = line.partition("  ")
        entries[path] = digest
    return entries


def test_the_manifest_lists_exactly_the_two_records():
    assert _manifest_entries() == EXPECTED


def test_every_record_still_hashes_to_its_recorded_digest():
    for relative, digest in EXPECTED.items():
        data = (ROOT / relative).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest, relative


def test_the_records_keep_the_bytes_the_exemptions_exist_for():
    for relative in EXPECTED:
        text = (ROOT / relative).read_text(encoding="utf-8")
        lines = text.splitlines()
        assert any(line.startswith(PHASE_MAP_SEPARATOR) for line in lines), relative


def test_the_derived_master_roadmap_is_normalised_instead():
    """The authoritative roadmap is repository-authored and follows house style."""
    text = (ROOT / "docs" / "master_roadmap_r0_r18.md").read_text(encoding="utf-8")
    lines = text.splitlines()
    assert not any(line.startswith("=======") for line in lines)
    assert not any(line != line.rstrip() for line in lines)
    assert text.endswith("\n") and not text.endswith("\n\n")
