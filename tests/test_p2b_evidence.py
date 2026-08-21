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

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BENCHMARK = ROOT / "artifacts" / "benchmark"

#: Every checksum manifest in the repository. A new checkpoint adds its own here
#: at freeze time, which is the moment its numbers stop being allowed to move.
MANIFESTS = (
    "btc_v4_SHA256SUMS.txt",
    "btc_p2a_SHA256SUMS.txt",
    "btc_p2b_SHA256SUMS.txt",
    "btc_p2b_ablation_SHA256SUMS.txt",
    "btc_p2b_recheck_SHA256SUMS.txt",
    "btc_p2c_SHA256SUMS.txt",
)


@pytest.mark.parametrize("manifest", MANIFESTS)
def test_frozen_evidence_still_hashes_to_its_manifest(manifest, capsys):
    """Every *primary* file a freeze covers is byte-identical to what was frozen.

    Comparison directories are deliberately exempt, and the exemption is the
    point rather than a concession. A comparison is *derived*: `nn.p2b_compare`
    regenerates it from the cells whenever the reporter improves, and it was
    regenerated twice tonight — once when the recomputation widened from ten
    trading keys to twenty-three, once when the report stopped hard-coding P2b's
    arm names and started naming the arms actually present. Hashing a
    regenerable artifact alongside primary evidence guarantees a stale manifest
    and teaches a reader to ignore the failure.

    So the hashes pin what cannot be rebuilt — the cells, their per-sample
    predictions, the regime description — and the comparison's *content* is
    pinned instead by the verdict tests below, which assert the six fold counts
    directly. A regenerated comparison that changed a finding fails those; a
    regenerated comparison that only improved its own prose does not.
    """
    path = ROOT / "artifacts" / manifest
    assert path.is_file(), f"{manifest} is missing"
    entries = [
        line.split(maxsplit=1) for line in path.read_text().splitlines() if line.strip()
    ]
    assert entries, f"{manifest} is empty"
    stale = 0
    for expected, name in entries:
        name = name.strip()
        target = ROOT / name
        assert target.is_file(), f"{name} is missing from the frozen evidence"
        if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            assert "_comparison/" in name, f"{name} no longer matches {manifest}"
            stale += 1
    # A manifest whose every entry is a regenerated comparison would pass
    # vacuously; each must still cover something that cannot be rebuilt.
    assert stale < len(entries), f"{manifest} covers nothing but derived artifacts"
    capsys.readouterr()


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
