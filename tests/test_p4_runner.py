"""The real P4 entrypoint, driven end to end, and stopped where it must stop.

``python -m nn.p2b --checkpoint P4`` is the production research path. These tests
run it — the whole of it, on the committed spine, through the real verifier and
the real universe construction — and assert that it refuses to fit. That is the
only way to know the guards are wired into the path rather than merely existing
beside it, and it is why these are here instead of a fifth set of unit tests over
the same functions.

**Nothing here fits a P4 model.** Every run below ends at a refusal, and the
refusals are the assertions.
"""

from __future__ import annotations

import json

import pytest

from nn import p2b
from nn.information_sets import CHECKPOINTS, P4_INFORMATION_SETS
from nn.p2b_compare import ComparisonError, checkpoint_of
from nn.p4_holdout import HoldoutError
from nn.p4_preregistration import HOLDOUT_ROWS, PRIMARY_MODEL
from nn.p4_stage1 import Stage1Interlock
from tools.export_derivatives_snapshot import (
    MANIFEST_NAME,
    acquisition_window,
    write_snapshot,
)
from tools.verify_derivatives_snapshot import verify_derivatives_snapshot

from tests.test_p4_snapshot import CONTRACT, FIXTURE_ARCHIVES


@pytest.fixture(scope="module")
def runnable_tree(tmp_path_factory, p4_spine, p4_hourly):
    """A research tree holding both snapshots, ready for the real entrypoint."""
    import shutil

    root = tmp_path_factory.mktemp("chimera-p4-run")
    shutil.copytree(p4_spine["research"], root / "data" / "research")
    manifest = root / "data" / "research" / "btc_usdt_1h_gen1_snapshot_manifest.json"
    start, end, alignment = acquisition_window(manifest, CONTRACT)
    write_snapshot(
        p4_hourly,
        FIXTURE_ARCHIVES,
        [],
        contract=CONTRACT,
        out_dir=root / "data" / "research",
        ohlcv_manifest=manifest,
        alignment_source=alignment,
        requested_from=start,
        requested_through=end,
        acquired_at="1970-01-01T00:00:00Z",
    )
    return {
        "root": root,
        "manifest": manifest,
        "derivatives": root / "data" / "research" / MANIFEST_NAME,
    }


def _argv(tree, out, arm="ohlcv14_plus_derivatives_v1", model=PRIMARY_MODEL, extra=()):
    return [
        "--checkpoint",
        "P4",
        "--information-set",
        arm,
        "--model",
        model,
        "--manifest",
        str(tree["manifest"]),
        "--derivatives-manifest",
        str(tree["derivatives"]),
        "--out",
        str(out),
        *extra,
    ]


# --- the checkpoint is reachable, and its arms are the preregistered three ----
def test_p4_is_a_choice_the_runner_offers():
    parser = p2b.build_argparser()
    checkpoint = next(a for a in parser._actions if a.dest == "checkpoint")
    information_set = next(a for a in parser._actions if a.dest == "information_set")
    assert "P4" in checkpoint.choices
    for arm in P4_INFORMATION_SETS:
        assert arm in information_set.choices


def test_an_arm_of_another_checkpoint_is_refused_under_p4(runnable_tree, tmp_path):
    out = tmp_path / "cell"
    with pytest.raises(SystemExit, match="not an arm of P4"):
        p2b.main(_argv(runnable_tree, out, arm="smc_v1"))
    assert not out.exists()


# --- the whole path runs, and stops at the interlock -------------------------
@pytest.mark.parametrize("arm", P4_INFORMATION_SETS)
def test_every_p4_arm_reaches_the_interlock_and_is_refused(runnable_tree, tmp_path, arm):
    """The source verifies, the universe builds, the gate is computed — and no fit.

    The refusal is the committed interlock's, which is the last gate before a
    model is built. Reaching it proves the steps before it ran; the empty output
    directory proves nothing after it did.
    """
    out = tmp_path / f"cell_{arm}"
    with pytest.raises(Stage1Interlock, match="not_authorised"):
        p2b.main(_argv(runnable_tree, out, arm=arm))
    assert not out.exists()


def test_the_confirmation_flag_alone_does_not_get_past_the_interlock(runnable_tree, tmp_path):
    out = tmp_path / "cell"
    with pytest.raises(Stage1Interlock, match="not_authorised"):
        p2b.main(_argv(runnable_tree, out, extra=("--authorise-fit",)))
    assert not out.exists()


def test_a_missing_derivatives_snapshot_stops_the_run_before_anything_else(
    runnable_tree, tmp_path
):
    out = tmp_path / "cell"
    argv = _argv(runnable_tree, out)
    argv[argv.index("--derivatives-manifest") + 1] = str(tmp_path / "absent.json")
    with pytest.raises(SystemExit, match="P4 needs a derivatives snapshot"):
        p2b.main(argv)


def test_a_derivatives_snapshot_that_fails_verification_never_reaches_a_fit(
    runnable_tree, tmp_path
):
    """The verifier runs inside the loader, so `python -m nn.p2b` cannot skip it."""
    import shutil

    root = tmp_path / "broken"
    shutil.copytree(runnable_tree["root"], root)
    path = root / "data" / "research" / MANIFEST_NAME
    payload = json.loads(path.read_text())
    payload["hourly"]["semantic_hash"] = "0" * 64
    path.write_text(json.dumps(payload, indent=2))
    out = tmp_path / "cell"
    with pytest.raises(SystemExit, match="failed verification"):
        p2b.main(
            _argv(
                {
                    "manifest": root
                    / "data"
                    / "research"
                    / "btc_usdt_1h_gen1_snapshot_manifest.json",
                    "derivatives": path,
                },
                out,
            )
        )
    assert not out.exists()


def test_a_snapshot_bound_to_other_candles_never_reaches_a_fit(runnable_tree, tmp_path):
    import shutil

    root = tmp_path / "mismatched"
    shutil.copytree(runnable_tree["root"], root)
    research = root / "data" / "research"
    path = research / MANIFEST_NAME
    payload = json.loads(path.read_text())
    payload["alignment"]["ohlcv_processed_semantic_prefix_hash"] = "7" * 64
    path.write_text(json.dumps(payload, indent=2))
    out = tmp_path / "cell"
    with pytest.raises(SystemExit):
        p2b.main(
            _argv(
                {
                    "manifest": research / "btc_usdt_1h_gen1_snapshot_manifest.json",
                    "derivatives": path,
                },
                out,
            )
        )
    assert not out.exists()


def test_the_committed_snapshot_and_source_agree_about_where_stage_one_stops(
    runnable_tree,
):
    checks = verify_derivatives_snapshot(runnable_tree["derivatives"])
    assert any(check.name == "p4-hold" for check in checks)
    payload = json.loads(runnable_tree["derivatives"].read_text())
    assert payload["safety_checks"]["acquisition_bounded_before_p4_hold"] is True
    assert payload["safety_checks"]["table_strictly_before_p4_hold"] is True


# --- the comparison path carries the same guard ------------------------------
def _cell(arm, model, folds):
    return {
        "information_set": arm,
        "model": model,
        "payload": {
            "checkpoint": "P4",
            "question": CHECKPOINTS["P4"].question,
            "alignment": {"folds": folds},
        },
    }


def _folds(last_outer_end):
    return [
        {
            "fold": 0,
            "train": {"row_range": [0, 26518], "last_row": 26510},
            "inner_validation": {"row_range": [26518, 31339], "last_row": 31330},
            "outer_validation": {
                "row_range": [31339, last_outer_end],
                "last_row": last_outer_end - 7,
            },
        }
    ]


def test_the_comparison_accepts_p4_cells_that_stop_before_the_holdout():
    cells = [
        _cell("ohlcv14", PRIMARY_MODEL, _folds(HOLDOUT_ROWS[0])),
        _cell("ohlcv14_plus_derivatives_v1", PRIMARY_MODEL, _folds(HOLDOUT_ROWS[0])),
    ]
    assert checkpoint_of(cells).name == "P4"


def test_the_comparison_refuses_a_p4_cell_whose_folds_reach_the_holdout():
    """A join is a real path into P4's rows, so it carries the guard too."""
    cells = [
        _cell("ohlcv14", PRIMARY_MODEL, _folds(HOLDOUT_ROWS[0])),
        _cell("ohlcv14_plus_derivatives_v1", PRIMARY_MODEL, _folds(HOLDOUT_ROWS[0] + 1)),
    ]
    with pytest.raises(HoldoutError, match="P4-HOLD"):
        checkpoint_of(cells)


def test_the_comparison_still_refuses_cells_of_two_checkpoints():
    cells = [_cell("ohlcv14", PRIMARY_MODEL, _folds(HOLDOUT_ROWS[0]))]
    other = _cell("ohlcv14", PRIMARY_MODEL, _folds(HOLDOUT_ROWS[0]))
    other["payload"]["checkpoint"] = "P3"
    with pytest.raises(ComparisonError, match="different research questions"):
        checkpoint_of([*cells, other])
