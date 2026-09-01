"""P13's committed evidence, and the rule that it cannot go missing quietly.

The external audit removed a line from ``artifacts/btc_p13_SHA256SUMS.txt``,
deleted the file that line covered, and watched the whole suite stay green. That
is the failure mode a checksum manifest is least able to catch by itself: it
verifies what it *lists*, so a deletion that also edits the manifest is invisible
to it. Every mature generation here is protected against that by a test that
knows independently which files must exist; P13 had no such test.

:data:`REQUIRED_EVIDENCE` is that independent statement. It is written here, in a
test, from the frozen stopping rule — NOT EVALUABLE must be "recorded ... with
the acquisition evidence that establishes it" — and not read from the manifest,
which is exactly what stops the manifest from vouching for itself.

The rule has three halves, and the mutation tests at the bottom of this file run
each of them against a scratch copy of the tree with a defect injected, so the
claim "deleting the file and its line still fails" is executed rather than
asserted in prose.

Nothing here computes or reads a P13 economic quantity. There are none: the
checkpoint is NOT EVALUABLE for an environment reason, and
``tests/test_p13_preregistration.py::test_no_p13_result_artifact_exists`` is what
holds it there.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from tools.freeze_evidence import EVIDENCE_SUFFIXES, check, manifest_entries

REPO = Path(__file__).resolve().parents[1]

#: Where P13's primary evidence lives, and the manifest that freezes it.
PRIMARY_DIR = "artifacts/benchmark/btc_p13_carry"
MANIFEST = "artifacts/btc_p13_SHA256SUMS.txt"

#: **The independent contract.** Every file P13's current NOT EVALUABLE state
#: rests on, named here rather than discovered from the manifest or from the
#: directory listing — both of which change when someone deletes evidence, which
#: is precisely the event this must survive.
#:
#: Each entry earns its place from the frozen design rather than from taste:
#:
#: * ``acquisition_refusal.json`` is the machine-readable refusal — the per-family
#:   probes, the plan digest and the generation provenance. It IS the evidence
#:   behind the verdict.
#: * ``acquisition_plan.json`` is the networkless plan naming all 260 objects the
#:   screen would have required. Without it the refusal names a host but not what
#:   was asked of it.
#: * ``STATUS.md`` is the human front door to both.
#:
#: Adding a fourth file to the directory without adding it here is a failure, not
#: a convenience: an evidence file nothing vouches for is the same problem from
#: the other side.
REQUIRED_EVIDENCE: tuple[str, ...] = (
    f"{PRIMARY_DIR}/STATUS.md",
    f"{PRIMARY_DIR}/acquisition_plan.json",
    f"{PRIMARY_DIR}/acquisition_refusal.json",
)


def coverage_problems(root: Path) -> list[str]:
    """Every way ``root``'s P13 evidence fails the contract above, described.

    A function over an arbitrary tree rather than a body of assertions over this
    one, so the mutation tests can inject a defect into a scratch copy and prove
    the rule notices. Empty means covered.
    """
    problems: list[str] = []
    manifest = root / MANIFEST
    if not manifest.is_file():
        return [f"MISSING MANIFEST  {MANIFEST}"]

    listed = [name for _, name in manifest_entries(manifest)]
    for name in REQUIRED_EVIDENCE:
        if not (root / name).is_file():
            problems.append(f"MISSING EVIDENCE  {name}")
        count = listed.count(name)
        if count != 1:
            problems.append(f"LISTED {count} TIMES  {name}")

    for name in listed:
        if not (root / name).is_file():
            problems.append(f"MANIFEST POINTS AT NOTHING  {name}")

    # The other direction: a primary evidence file nobody vouches for. Scanned
    # with the freezer's own definition of what counts as evidence, so the two
    # cannot disagree about which suffixes matter.
    directory = root / PRIMARY_DIR
    present = {
        str(path.relative_to(root))
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.suffix in EVIDENCE_SUFFIXES
    }
    for name in sorted(present - set(REQUIRED_EVIDENCE)):
        problems.append(f"UNDECLARED EVIDENCE  {name}")
    for name in sorted(present - set(listed)):
        problems.append(f"UNCOVERED BY MANIFEST  {name}")
    return problems


@pytest.fixture()
def scratch(tmp_path: Path) -> Path:
    """A copy of just the P13 evidence and its manifest, for injecting defects."""
    root = tmp_path / "tree"
    (root / PRIMARY_DIR).mkdir(parents=True)
    (root / "artifacts").mkdir(exist_ok=True)
    shutil.copy(REPO / MANIFEST, root / MANIFEST)
    for name in REQUIRED_EVIDENCE:
        shutil.copy(REPO / name, root / name)
    return root


def drop(root: Path, name: str) -> None:
    """Delete an evidence file AND the manifest line covering it.

    The audit's exact manoeuvre. A rule that only re-hashes what the manifest
    lists is blind to it.
    """
    (root / name).unlink()
    manifest = root / MANIFEST
    manifest.write_text(
        "".join(
            line + "\n"
            for line in manifest.read_text().splitlines()
            if line.strip() and line.split(maxsplit=1)[1].strip() != name
        )
    )


# --------------------------------------------------------------------------- #
# The evidence as committed
# --------------------------------------------------------------------------- #


def test_the_committed_p13_evidence_satisfies_the_contract():
    assert coverage_problems(REPO) == []


def test_the_manifest_still_hashes_to_what_the_freeze_recorded():
    assert check(REPO / MANIFEST) == []


@pytest.mark.parametrize("name", REQUIRED_EVIDENCE)
def test_every_required_evidence_file_exists_and_is_listed_once(name):
    assert (REPO / name).is_file()
    listed = [entry for _, entry in manifest_entries(REPO / MANIFEST)]
    assert listed.count(name) == 1


def test_the_manifest_covers_exactly_the_required_set_and_nothing_else():
    listed = {name for _, name in manifest_entries(REPO / MANIFEST)}
    assert listed == set(REQUIRED_EVIDENCE)


def test_the_scratch_copy_of_the_tree_is_itself_covered(scratch):
    """The control the mutations below are measured against. Without it a rule
    that reported a problem for every tree would look like it was working."""
    assert coverage_problems(scratch) == []


# --------------------------------------------------------------------------- #
# The mutations the external audit performed, executed
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", REQUIRED_EVIDENCE)
def test_deleting_an_evidence_file_and_its_manifest_line_still_fails(scratch, name):
    drop(scratch, name)
    problems = coverage_problems(scratch)
    assert any(name in problem for problem in problems), (
        f"{name} was deleted along with the line that covered it and the rule stayed "
        "silent. That is the defect this file exists for."
    )


@pytest.mark.parametrize("name", REQUIRED_EVIDENCE)
def test_deleting_only_the_evidence_file_still_fails(scratch, name):
    (scratch / name).unlink()
    assert coverage_problems(scratch) != []
    assert check(scratch / MANIFEST, root=scratch) != []


@pytest.mark.parametrize("name", REQUIRED_EVIDENCE)
def test_deleting_only_the_manifest_line_still_fails(scratch, name):
    manifest = scratch / MANIFEST
    manifest.write_text(
        "".join(
            line + "\n"
            for line in manifest.read_text().splitlines()
            if line.strip() and line.split(maxsplit=1)[1].strip() != name
        )
    )
    assert coverage_problems(scratch) != []


def test_a_new_primary_evidence_file_cannot_appear_uncovered(scratch):
    """The other direction: evidence that appears without anything vouching for
    it. A P13 economic artifact dropped into this directory is the case that
    matters, and it must not be able to arrive quietly."""
    sneaked = scratch / PRIMARY_DIR / "block_results.json"
    sneaked.write_text("{}\n")
    problems = coverage_problems(scratch)
    assert any("block_results.json" in problem for problem in problems)


def test_a_corrupted_evidence_file_still_fails(scratch):
    """Belt to the contract's braces: the digest half of the guarantee."""
    target = scratch / REQUIRED_EVIDENCE[0]
    target.write_text(target.read_text() + "\nedited\n")
    assert check(scratch / MANIFEST, root=scratch) != []


# --------------------------------------------------------------------------- #
# Coverage cannot be escaped by being compressed
# --------------------------------------------------------------------------- #
#
# The rule above is keyed on file extension, and P13's PREREGISTERED SOURCES are
# `.zip` archive objects with `.zip.CHECKSUM` companions. Under the original
# suffix list every one of the 260 objects the acquisition will fetch could have
# landed in this directory unmanifested — not because anyone excluded them, but
# because the covering rule did not recognise the shape source data arrives in.
#
# These are synthetic filesystem tests. No archive is downloaded, no host is
# contacted, and the files written below are empty placeholders with the right
# names.
# --------------------------------------------------------------------------- #


def source_object_suffixes() -> set[str]:
    """Every archive suffix the frozen P13 design says it will fetch.

    Derived from the preregistration rather than restated, so widening the source
    set later cannot leave this test agreeing with a list that no longer matches
    the design.
    """
    from nn.p13_preregistration import DATA_SOURCES

    suffixes: set[str] = set()
    for source in DATA_SOURCES:
        for key in ("object", "checksum_object"):
            # `checksum_object` carries a trailing ", sha256" note; the object
            # name is the first comma-separated field.
            name = str(source[key]).split(",")[0].strip()
            suffixes.add(Path(name).suffix)
    return suffixes


def test_the_frozen_design_really_does_name_compressed_source_objects():
    """The premise, checked. Without it the next test could pass vacuously."""
    assert source_object_suffixes() == {".zip", ".CHECKSUM"}


@pytest.mark.parametrize("suffix", sorted(source_object_suffixes()))
def test_no_preregistered_source_suffix_can_escape_the_coverage_rule(suffix):
    assert suffix in EVIDENCE_SUFFIXES, (
        f"a preregistered P13 source object ending in {suffix} would not be seen by the "
        "coverage scan, so the acquisition could leave it unmanifested silently"
    )


@pytest.mark.parametrize(
    "name",
    [
        "BTCUSDT-1h-2020-01.zip",
        "BTCUSDT-1h-2020-01.zip.CHECKSUM",
        "BTCUSDT-fundingRate-2020-01.zip",
        "block_results.csv.gz",
    ],
)
def test_a_source_archive_dropped_into_the_evidence_directory_cannot_arrive_quietly(
    scratch, name
):
    """The acquisition stage's exact failure mode, executed against a scratch tree."""
    (scratch / PRIMARY_DIR / name).write_bytes(b"")
    problems = coverage_problems(scratch)
    assert any(name in problem for problem in problems), (
        f"{name} landed in the primary evidence directory and nothing objected"
    )
    assert any("UNCOVERED BY MANIFEST" in problem and name in problem for problem in problems)


def test_the_widening_did_not_reach_files_that_are_not_evidence(scratch):
    """Narrow on purpose: a cache or an editor dropping must still be ignored.

    A rule that covered everything would fail for reasons having nothing to do
    with the evidence, which is how a checksum stops being a signal.
    """
    for name in ("notes.swp", "cache.pyc", "scratch.tmp"):
        (scratch / PRIMARY_DIR / name).write_bytes(b"")
    assert coverage_problems(scratch) == []
