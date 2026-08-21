"""The committed P2b evidence, checked rather than described.

Two claims in this repository's documentation were, until this file existed,
prose alone.

The first is the strongest single defence of P2b's negative result: that its
OHLCV14 control reproduces P2a's frozen seed-42 XGBoost evidence exactly, from a
different code path reading the committed research snapshot rather than the
canonical dataset. If that is true, the negative result cannot be blamed on a
broken control. It was true when it was written down and nothing would have
noticed if it stopped being true.

The second is :mod:`tools.freeze_evidence`'s own docstring, which says a test
asserts every file a manifest covers still hashes to the value the freeze
recorded. That was true of exactly one of the four manifests in the repository.

Neither test needs a dataset, a fit or a network. Both read committed files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.freeze_evidence import DERIVED, check, evidence_class_of, manifest_entries

ROOT = Path(__file__).resolve().parent.parent
BENCHMARK = ROOT / "artifacts" / "benchmark"

#: Every checksum manifest this repository stands behind, discovered rather than
#: listed. A checkpoint that freezes new evidence is covered the moment its
#: manifest lands, instead of when someone remembers to add a name here.
MANIFESTS = sorted(path.name for path in (ROOT / "artifacts").glob("*_SHA256SUMS.txt"))

#: Manifests written before a manifest covered primary evidence only, and the
#: primary-only manifest that replaced each. They are kept byte for byte and
#: renamed rather than edited: a manifest is the repository's own statement
#: about what a past run produced, so the answer to one that covered the wrong
#: kind of file is a successor, never a rewrite. The `.superseded.txt` suffix is
#: what keeps a manifest nobody should expect to verify out of the set below,
#: because a checksum file that fails on purpose teaches a reader to shrug at
#: one that fails for real. `None` means there was nothing to succeed —
#: `btc_p2b_recheck` covered a comparison and nothing else.
SUPERSEDED = {
    "btc_p2b_SHA256SUMS.superseded.txt": "btc_p2b_cells_SHA256SUMS.txt",
    "btc_p2b_ablation_SHA256SUMS.superseded.txt": ("btc_p2b_ablation_cells_SHA256SUMS.txt"),
    "btc_p2b_recheck_SHA256SUMS.superseded.txt": None,
    "btc_p2c_SHA256SUMS.superseded.txt": "btc_p2c_cells_SHA256SUMS.txt",
}


def test_the_repository_has_the_manifests_these_tests_think_it_has():
    """Discovery is convenient and silent; a deleted manifest would just vanish.

    Globbing means a new checkpoint is covered automatically. It also means a
    manifest that disappeared would take its own verification with it and no
    test would fail, so the set is named once, here.
    """
    assert MANIFESTS == [
        "btc_p2a_SHA256SUMS.txt",
        "btc_p2b_ablation_cells_SHA256SUMS.txt",
        "btc_p2b_cells_SHA256SUMS.txt",
        "btc_p2c_cells_SHA256SUMS.txt",
        "btc_v4_SHA256SUMS.txt",
    ]
    for retired in SUPERSEDED:
        assert (ROOT / "artifacts" / retired).is_file(), (
            f"{retired} was retired, not deleted; it is the record of what its "
            "checkpoint froze and the repository keeps it"
        )


@pytest.mark.parametrize("manifest", MANIFESTS)
def test_frozen_evidence_still_hashes_to_its_manifest(manifest):
    """Every file a manifest covers is byte-identical to what was frozen.

    No exemption, of any kind. `tools.freeze_evidence.check` is the single
    implementation the CLI and this test share, so "frozen" cannot come to mean
    one thing under `make freeze-evidence` and another here.

    That is only enforceable because a manifest covers primary evidence alone —
    cells and their per-sample outer predictions, which cannot be rebuilt
    without re-fitting. An earlier arrangement hashed the comparisons alongside
    them, so improving a reporter broke three manifests at once and this test
    excused the breakage by matching `_comparison/` in the path, which is a
    checksum that means "frozen unless it is not". What pins a derived report
    instead is the rest of this file: its fold counts and verdicts are asserted
    directly, so a regenerated report that changed a finding fails and one that
    only improved its own prose does not.
    """
    problems = check(ROOT / "artifacts" / manifest)
    assert problems == [], f"{manifest} no longer describes the repository:\n" + "\n".join(
        problems
    )


@pytest.mark.parametrize("retired,successor", sorted(SUPERSEDED.items()))
def test_a_reissued_manifest_changed_no_digest_it_inherited(retired, successor):
    """Re-issuing a manifest dropped derived entries and did nothing else.

    This is the check that makes the supersession safe to trust. Narrowing a
    manifest is exactly the shape of laundering a result — drop the row that
    stopped matching, keep the file, call it frozen — so a successor is required
    to be a strict projection of what it replaced: every path they share carries
    the same digest, nothing was added, and every entry that went away lives in
    a directory that now declares itself `derived`.

    A successor that re-froze a moved primary file fails here, and so does one
    that quietly dropped a cell.
    """
    old = dict(
        (name, digest) for digest, name in manifest_entries(ROOT / "artifacts" / retired)
    )
    new = (
        dict(
            (name, digest) for digest, name in manifest_entries(ROOT / "artifacts" / successor)
        )
        if successor
        else {}
    )
    assert old, f"{retired} is empty"

    rehashed = [name for name in old.keys() & new.keys() if old[name] != new[name]]
    assert not rehashed, (
        f"{successor} records a different digest than {retired} for {rehashed}. A "
        "successor may narrow what a manifest covers; it may not restate what a "
        "covered file hashed to."
    )
    assert not new.keys() - old.keys(), (
        f"{successor} covers files {retired} did not: {sorted(new.keys() - old.keys())}. "
        "New evidence gets its own manifest rather than being folded into a re-issue."
    )
    for name in sorted(old.keys() - new.keys()):
        directory = (ROOT / name).parent
        assert evidence_class_of(directory) == DERIVED, (
            f"{retired} covered {name} and {successor} does not, but "
            f"{directory.name} does not declare itself {DERIVED}. Only derived "
            "evidence may be dropped from a re-issued manifest."
        )


def test_the_p2c_comparison_reports_a_negative_result():
    """P2c's frozen verdicts. Same reasoning as the P2b pin below.

    Every mean delta here is negative, which P2b's was not — there is no arm in
    P2c where pooling the four periods would even have flattered the result.
    """
    payload = json.loads(
        (BENCHMARK / "btc_p2c_comparison" / "p2b_comparison.json").read_text()
    )
    assert payload["sealed_test"] is False
    assert payload["independent_recompute"]["mismatches"] == 0
    assert payload["snapshot_anchoring"]["problems"] == 0

    expected = {
        ("logistic_regression", "chart_structure_v1"): 1,
        ("logistic_regression", "ohlcv14_plus_chart_structure_v1"): 0,
        ("lightgbm", "chart_structure_v1"): 1,
        ("lightgbm", "ohlcv14_plus_chart_structure_v1"): 2,
        ("xgboost", "chart_structure_v1"): 1,
        ("xgboost", "ohlcv14_plus_chart_structure_v1"): 1,
    }
    for (model, arm), folds in expected.items():
        entry = payload["deltas"][model][arm]
        assert entry["net_return_improved_folds"] == folds
        assert entry["aggregate"]["net_return"]["mean"] < 0
    assert all(v < 3 for v in expected.values())


def test_both_information_set_checkpoints_reproduce_the_same_frozen_control():
    """P2b and P2c each re-ran the OHLCV14 control, under different code.

    The two checkpoints have different source digests — P2c's wiring of a second
    feature family changed the module `nn.p2b` imports — so their controls are
    two independent runs of the same configuration through two versions of the
    alignment layer. Both must equal P2a's frozen seed-42 XGBoost evidence, and
    therefore each other. If adding a feature family ever perturbs the control's
    sample universe, this is what notices.
    """
    p2a = json.loads((BENCHMARK / "btc_p2a_seed_42" / "benchmark.json").read_text())
    frozen = [f["outer_validation"]["xgboost"] for f in p2a["folds"]]
    for checkpoint in ("btc_p2b_ohlcv14_xgboost", "btc_p2c_ohlcv14_xgboost"):
        cell = json.loads((BENCHMARK / checkpoint / "p2b.json").read_text())
        assert [f["outer_validation"]["xgboost"] for f in cell["folds"]] == frozen


def test_the_p2b_comparison_reports_a_negative_result():
    """The frozen verdicts, pinned so a rerun cannot quietly improve them.

    Not a test of the machinery — a test of the *finding*. P2b answered no, and
    the six fold counts below are what that means. If a later change to the
    comparison moves any of them, the result changed, and that has to be a
    deliberate act with its own evidence rather than a diff nobody read.
    """
    payload = json.loads(
        (BENCHMARK / "btc_p2b_comparison" / "p2b_comparison.json").read_text()
    )
    assert payload["sealed_test"] is False
    assert payload["independent_recompute"]["mismatches"] == 0
    assert payload["snapshot_anchoring"]["problems"] == 0

    expected = {
        ("logistic_regression", "smc_v1"): 0,
        ("logistic_regression", "ohlcv14_plus_smc_v1"): 2,
        ("lightgbm", "smc_v1"): 2,
        ("lightgbm", "ohlcv14_plus_smc_v1"): 1,
        ("xgboost", "smc_v1"): 1,
        ("xgboost", "ohlcv14_plus_smc_v1"): 1,
    }
    for (model, arm), folds in expected.items():
        assert payload["deltas"][model][arm]["net_return_improved_folds"] == folds

    # The predeclared bar is three of four. Nothing reached it, which is the
    # finding; asserting it here stops "P2b was positive" from ever being true
    # of this directory without someone changing this line.
    assert all(v < 3 for v in expected.values())
