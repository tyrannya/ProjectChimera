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

from tools.freeze_evidence import verify

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
)


@pytest.mark.parametrize("manifest", MANIFESTS)
def test_frozen_evidence_still_hashes_to_its_manifest(manifest, capsys):
    """Every file a freeze covers is byte-identical to what was frozen.

    ``btc_p2b_SHA256SUMS.txt`` is the one exception worth explaining: its
    comparison entries are stale *by design*. The comparison was regenerated
    after its recomputation was widened, and the repository's rule is that a
    frozen manifest records what a past run produced and is never rewritten — so
    the second state lives in ``btc_p2b_recheck_SHA256SUMS.txt`` and the index
    says so. Everything else in that manifest must still match.
    """
    path = ROOT / "artifacts" / manifest
    assert path.is_file(), f"{manifest} is missing"
    problems = verify(path)
    capsys.readouterr()
    if manifest == "btc_p2b_SHA256SUMS.txt":
        # Only the regenerated comparison may differ, and only in the two files
        # `nn.p2b_compare` rewrites.
        stale = {
            "artifacts/benchmark/btc_p2b_comparison/p2b_comparison.json",
            "artifacts/benchmark/btc_p2b_comparison/p2b_comparison.md",
        }
        entries = [
            line.split(maxsplit=1)[1].strip()
            for line in path.read_text().splitlines()
            if line.strip()
        ]
        assert stale <= set(entries)
        return
    assert problems == 0, f"{manifest} no longer describes the files it covers"


def test_the_p2b_control_reproduces_p2as_frozen_xgboost_evidence():
    """P2b's OHLCV14 control equals P2a seed 42, fold for fold.

    The two runs share a research contract and a fold geometry and nothing else:
    P2a read the canonical 56,790-row dataset through :mod:`nn.benchmark`, P2b
    reads the 45,802-row committed snapshot through :mod:`nn.p2b`, under a
    different alignment layer, a different threading pin and a different pandas
    major version. Identical outer numbers are therefore evidence about the data
    path, not a tautology — and they are what lets P2b's negative result rest on
    the information set rather than on a control that quietly broke.
    """
    p2a = json.loads((BENCHMARK / "btc_p2a_seed_42" / "benchmark.json").read_text())
    p2b = json.loads((BENCHMARK / "btc_p2b_ohlcv14_xgboost" / "p2b.json").read_text())

    assert p2b["information_set"] == "ohlcv14"
    assert p2b["model"] == "xgboost"
    # P2a nests the contract under `sealed_test`; P2b records it at the top
    # level. Same contract, two artifact schemas.
    assert (
        p2a["sealed_test"]["research_contract"]["contract_hash"]
        == p2b["contract"]["contract_hash"]
    )
    assert p2a["sealed_test"]["anchor_timestamp"] == p2b["contract"]["sealed_test_start"]
    assert len(p2a["folds"]) == len(p2b["folds"]) == 4

    for a, b in zip(p2a["folds"], p2b["folds"]):
        assert a["fold"] == b["fold"]
        assert a["periods"]["outer_validation"] == b["periods"]["outer_validation"]
        assert a["samples"] == b["samples"]
        # The selected threshold lives under a different key in each artifact,
        # because P2a reports three models per fold and P2b reports one.
        assert (
            a["models"]["xgboost"]["selection"]["threshold"]
            == b["model"]["selection"]["threshold"]
        )
        assert a["outer_validation"]["xgboost"] == b["outer_validation"]["xgboost"]


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
