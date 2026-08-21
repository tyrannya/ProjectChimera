"""Safeguards for the P2b machinery: the views, the runner and the comparison.

P2b's answer is a *difference* between two numbers. A difference is only about
the information sets while everything else — which rows are scored, what the
label is, where the folds fall, what a trade costs — is identical, so the parts
of this repository that assert that identity are the parts the whole checkpoint
rests on. They had no tests. This file gives them one apiece:

1. **The sets themselves.** 14 / 39 / 53 columns, combined in that order, six
   ablations that can only remove a *declared* family. A set whose families do
   not partition its columns cannot be constructed at all.
2. **The alignment layer.** :func:`nn.information_sets.build_information_set_views`
   is checked for the property the proof relies on — the views hold the *same
   array objects*, not equal ones — and for failing closed on every input that
   would move a row: a spine timestamp with no candle, unsorted or duplicated
   raw candles, a segment the raw history contradicts, a non-finite feature.
3. **The proof.** :meth:`AlignedResearchSamples.prove_alignment` is checked to
   produce evidence a later reader can recompute, and to refuse views that were
   built from a copied array, a different label, or different segment ids.
4. **The runner and the comparison**, on a synthetic dataset small enough to fit
   in a second: one cell end to end, and — the assertion that matters most —
   that :func:`nn.p2b_compare.recompute_cell` *reports* a fold whose stored net
   return has been edited away from what its own predictions imply. A recompute
   that cannot fail is decoration, and the self-checking evidence is the reason
   nine independent processes may be joined into one report.

The fold geometry assertions are written out as literal row numbers taken from
the committed manifest, not derived from the module under test: a plan that
agrees with the code that produced it is not evidence that the plan is right.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from chimera.contracts import TargetSpec
from chimera.features import FeatureSpec, feature_columns
from nn import evaluate as ev
from nn import p2b, p2b_compare
from nn.data_pipeline import build_dataset
from nn.dataset import Split, sample_indices
from nn.information_sets import (
    ABLATABLE_FAMILIES,
    COMBINED,
    OHLCV14,
    P2B_INFORMATION_SETS,
    SMC_V1,
    AlignmentError,
    InformationSet,
    ablation_set_names,
    build_information_set_views,
    combined_set,
    information_set,
    ohlcv14_set,
    smc_v1_set,
)
from nn.simple_models import LOGISTIC_REGRESSION, fit_simple_model
from nn.smc import SMC_FEATURE_FAMILIES, smc_feature_columns
from nn.train import prepare_research_windows
from nn.walkforward import FoldPlan, REFERENCES_KEY
from tools.make_sample_data import generate_candles

ROOT = Path(__file__).resolve().parents[1]
REAL_MANIFEST = ROOT / "data" / "research" / "btc_usdt_1h_gen1_snapshot_manifest.json"

#: Small enough that a logistic fit is a second's work, long enough that every
#: block holds many windows.
SEQ_LEN = 6
HORIZON = 4

#: Two folds over the synthetic spine, in the shape ``plan_nested_folds`` emits.
FOLDS = (
    FoldPlan(
        train=Split("train", 0, 300),
        inner=Split("inner_validation", 300, 400),
        outer=Split("outer_validation", 400, 500),
    ),
    FoldPlan(
        train=Split("train", 0, 400),
        inner=Split("inner_validation", 400, 500),
        outer=Split("outer_validation", 500, 600),
    ),
)


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def raw_candles() -> pd.DataFrame:
    """Deterministic synthetic OHLCV, dated well before any sealed instant."""
    return generate_candles(rows=700, seed=11, start="2021-01-01")


@pytest.fixture(scope="module")
def spine(raw_candles):
    """The processed research frame and its metadata, built from those candles."""
    return build_dataset(
        raw_candles,
        FeatureSpec(),
        TargetSpec(horizon=HORIZON),
        exchange="synthetic",
        pair="SYNTH/USDT",
        timeframe="1h",
    )


@pytest.fixture(scope="module")
def aligned(spine, raw_candles):
    frame, ds_meta = spine
    return build_information_set_views(frame, ds_meta, raw_candles, names=P2B_INFORMATION_SETS)


@pytest.fixture(scope="module")
def cell(aligned):
    """One real P2b cell: ``smc_v1`` x logistic regression over both folds."""
    records, predictions = p2b.run_cell(
        aligned,
        SMC_V1,
        LOGISTIC_REGRESSION,
        FOLDS,
        seed=42,
        seq_len=SEQ_LEN,
        min_trades=3,
    )
    return records, predictions


@pytest.fixture(scope="module")
def real_manifest() -> dict[str, Any]:
    return json.loads(REAL_MANIFEST.read_text())


# --------------------------------------------------------------------------- #
# A. the information sets
# --------------------------------------------------------------------------- #
def test_information_set_rejects_a_duplicate_column():
    with pytest.raises(ValueError, match="lists a column twice"):
        InformationSet(name="dupe", columns=("a", "a"), families={"f": ("a", "a")})


def test_information_set_rejects_families_that_do_not_partition_the_columns():
    """A column no family owns could never be ablated, and one owned twice would
    be removed by two different ablations."""
    with pytest.raises(ValueError, match="do not partition its columns"):
        InformationSet(name="short", columns=("a", "b"), families={"f": ("a",)})
    with pytest.raises(ValueError, match="do not partition its columns"):
        InformationSet(name="long", columns=("a",), families={"f": ("a",), "g": ("a",)})


def test_the_three_arms_have_the_declared_widths_and_ordering():
    ohlcv, smc, both = ohlcv14_set(), smc_v1_set(), combined_set()

    assert len(ohlcv.columns) == 14
    assert len(smc.columns) == 39
    assert len(both.columns) == 53
    # Exactly ohlcv14's columns, then smc's, in that order — the ordering every
    # persisted prediction and the combined spec hash depend on.
    assert both.columns == tuple(feature_columns()) + tuple(smc_feature_columns())
    assert both.columns[:14] == ohlcv.columns
    assert both.columns[14:] == smc.columns
    assert set(both.families) == {"ohlcv14", *SMC_FEATURE_FAMILIES}


def test_without_drops_exactly_one_family_and_keeps_the_rest_in_order():
    both = combined_set()
    for family, members in SMC_FEATURE_FAMILIES.items():
        ablated = both.without(family)
        assert ablated.name == f"{COMBINED}_minus_{family}"
        assert set(ablated.columns) == set(both.columns) - set(members)
        assert len(ablated.columns) == 53 - len(members)
        # Order preserved: the survivors appear exactly as they did before.
        assert ablated.columns == tuple(c for c in both.columns if c not in set(members))
        assert family not in ablated.families


def test_without_an_undeclared_family_raises():
    with pytest.raises(KeyError, match="has no family"):
        combined_set().without("momentum")


def test_information_set_resolves_the_three_arms_and_the_six_ablations():
    for name in P2B_INFORMATION_SETS:
        assert information_set(name).name == name
    names = ablation_set_names()
    assert len(names) == 6 == len(ABLATABLE_FAMILIES)
    for name, family in zip(names, ABLATABLE_FAMILIES):
        resolved = information_set(name)
        assert resolved.name == name
        assert len(resolved.columns) == 53 - len(SMC_FEATURE_FAMILIES[family])


def test_an_unknown_information_set_names_the_known_ones():
    with pytest.raises(KeyError) as excinfo:
        information_set("ohlcv14_plus_rsi")
    message = str(excinfo.value)
    assert "unknown information set" in message
    for known in (OHLCV14, SMC_V1, COMBINED, *ablation_set_names()):
        assert known in message


# --------------------------------------------------------------------------- #
# B. build_information_set_views — the alignment guarantees
# --------------------------------------------------------------------------- #
def test_the_views_share_the_spine_arrays_by_object_identity(aligned):
    """``is``, not ``==``. Equality today would still let a later copy drift."""
    control = aligned.views[OHLCV14]
    for name in (SMC_V1, COMBINED):
        view = aligned.views[name]
        assert view.targets is control.targets
        assert view.dates is control.dates
        assert view.future_return is control.future_return
        assert view.close is control.close
        assert view.segment_ids is control.segment_ids


def test_only_the_features_differ_between_views(aligned):
    control = aligned.views[OHLCV14]
    smc = aligned.views[SMC_V1]
    both = aligned.views[COMBINED]

    for name in ("ds_meta", "feature_spec", "target_spec"):
        assert getattr(smc, name) is getattr(control, name)
    assert smc.candles_per_year == control.candles_per_year
    assert smc.n_rows == control.n_rows == both.n_rows == aligned.n_rows

    assert control.feature_names == feature_columns()
    assert smc.feature_names == smc_feature_columns()
    assert both.feature_names == feature_columns() + smc_feature_columns()
    assert control.features.shape[1] == 14
    assert smc.features.shape[1] == 39
    # The combined matrix is the two others side by side, in the declared order.
    assert np.array_equal(both.features, np.column_stack([control.features, smc.features]))


def test_a_spine_timestamp_with_no_raw_candle_fails_closed(spine, raw_candles):
    frame, ds_meta = spine
    # Row 200 of the raw history is inside the spine's coverage (the spine starts
    # after the feature warm-up), so removing it orphans a research row.
    holed = raw_candles.drop(index=200).reset_index(drop=True)
    with pytest.raises(ValueError) as excinfo:
        build_information_set_views(frame, ds_meta, holed, names=(OHLCV14,))
    message = str(excinfo.value)
    assert "no raw candle" in message
    assert str(raw_candles["date"].iloc[200]) in message


def test_unsorted_raw_candles_fail_closed(spine, raw_candles):
    frame, ds_meta = spine
    shuffled = raw_candles.copy()
    order = list(range(len(shuffled)))
    order[100], order[101] = order[101], order[100]
    shuffled = shuffled.iloc[order].reset_index(drop=True)
    with pytest.raises(ValueError, match="sorted and free of duplicate timestamps"):
        build_information_set_views(frame, ds_meta, shuffled, names=(OHLCV14,))


def test_duplicate_raw_timestamps_fail_closed(spine, raw_candles):
    frame, ds_meta = spine
    duplicated = raw_candles.copy()
    duplicated.loc[101, "date"] = duplicated.loc[100, "date"]
    with pytest.raises(ValueError, match="sorted and free of duplicate timestamps"):
        build_information_set_views(frame, ds_meta, duplicated, names=(OHLCV14,))


def test_a_spine_without_segment_ids_fails_closed(spine, raw_candles):
    """Without segment ids a window could bridge a market-data gap unnoticed, so
    the module refuses the frame and says how to rebuild it."""
    frame, ds_meta = spine
    with pytest.raises(ValueError, match="no 'segment_id' column"):
        build_information_set_views(
            frame.drop(columns=["segment_id"]), ds_meta, raw_candles, names=(OHLCV14,)
        )


def test_a_spine_segment_the_raw_history_contradicts_fails_closed(spine, raw_candles):
    """The spine calls two rows contiguous; the raw candles put a candle between
    them. Market structure would then carry across a step the windowing believes
    does not exist."""
    frame, ds_meta = spine
    gapped = frame.drop(index=200).reset_index(drop=True)
    with pytest.raises(ValueError) as excinfo:
        build_information_set_views(gapped, ds_meta, raw_candles, names=(OHLCV14,))
    message = str(excinfo.value)
    assert "research rows 199 and 200" in message
    assert "2 candles apart" in message
    assert "bridge" in message


def test_a_processed_segment_spanning_two_raw_segments_fails_closed(spine, raw_candles):
    """The same disagreement in its other form: the rows stay adjacent in the raw
    frame, but the raw frame itself now has a gap there."""
    frame, ds_meta = spine
    dropped = raw_candles["date"].iloc[200]
    holed_raw = raw_candles.drop(index=200).reset_index(drop=True)
    holed_spine = frame[frame["date"] != dropped].reset_index(drop=True)
    with pytest.raises(ValueError, match="bridge a market-data gap"):
        build_information_set_views(holed_spine, ds_meta, holed_raw, names=(OHLCV14,))


def test_a_non_finite_market_structure_value_fails_closed(spine, raw_candles, monkeypatch):
    """smc_v1 guarantees a finite value on every row; a NaN here would silently
    change which rows the arms are compared on, so it is refused by name."""
    frame, ds_meta = spine

    def poisoned(candles, timeframe, spec):
        out = pd.DataFrame(
            0.0, index=range(len(candles)), columns=smc_feature_columns(), dtype=np.float64
        )
        out.iloc[150, 0] = np.nan
        return out

    monkeypatch.setattr(
        "nn.information_sets.compute_smc_features_segmented", poisoned
    )
    with pytest.raises(ValueError) as excinfo:
        build_information_set_views(frame, ds_meta, raw_candles, names=(SMC_V1,))
    message = str(excinfo.value)
    assert "non-finite market-structure value" in message
    assert smc_feature_columns()[0] in message
    assert "docs/smc_v1.md" in message and "§5" in message


# --------------------------------------------------------------------------- #
# C. prove_alignment
# --------------------------------------------------------------------------- #
def test_prove_alignment_returns_recomputable_per_fold_evidence(aligned):
    evidence = aligned.prove_alignment(FOLDS, SEQ_LEN)

    assert evidence["information_sets"] == list(P2B_INFORMATION_SETS)
    assert evidence["seq_len"] == SEQ_LEN
    assert evidence["horizon"] == HORIZON
    assert evidence["rows"] == aligned.n_rows
    assert len(evidence["folds"]) == len(FOLDS)

    blocks = ("train", "inner_validation", "outer_validation")
    for fold, (record, plan) in enumerate(zip(evidence["folds"], FOLDS)):
        assert record["fold"] == fold
        for block, split in zip(blocks, (plan.train, plan.inner, plan.outer)):
            entry = record[block]
            assert entry["row_range"] == [split.start, split.end]
            # Independently recomputed, per view: the counts and the digest are
            # one value across the three information sets.
            for name in P2B_INFORMATION_SETS:
                idx = sample_indices(
                    split, SEQ_LEN, HORIZON, segment_ids=aligned.views[name].segment_ids
                )
                assert entry["samples"] == len(idx)
                assert entry["first_row"] == int(idx[0])
                assert entry["last_row"] == int(idx[-1])
                digest = hashlib.sha256(
                    np.ascontiguousarray(idx, dtype=np.int64).tobytes()
                ).hexdigest()
                assert entry["sample_index_sha256"] == digest

    # Different rows must hash differently, or the digest proves nothing. Within
    # a fold the three blocks are disjoint; across folds the outer blocks
    # partition one stretch and can never repeat. (Fold k's outer block *is*
    # fold k+1's inner block at step == outer_size, so those two agreeing is the
    # geometry, not a collision.)
    for record in evidence["folds"]:
        digests = [record[block]["sample_index_sha256"] for block in blocks]
        assert len(set(digests)) == len(blocks)
    outer = [record["outer_validation"]["sample_index_sha256"] for record in evidence["folds"]]
    assert len(set(outer)) == len(FOLDS)


def test_prove_alignment_refuses_a_view_built_from_a_copied_array(aligned):
    copied = dataclasses.replace(
        aligned.views[SMC_V1], targets=aligned.views[SMC_V1].targets.copy()
    )
    broken = dataclasses.replace(aligned, views={**aligned.views, SMC_V1: copied})
    with pytest.raises(AlignmentError, match="does not share the spine's target array"):
        broken.prove_alignment(FOLDS, SEQ_LEN)


def test_prove_alignment_refuses_views_whose_label_or_costs_differ(aligned):
    control = aligned.views[OHLCV14]
    for replacement in (
        dataclasses.replace(control.target_spec, horizon=control.target_spec.horizon + 1),
        dataclasses.replace(control.target_spec, fee_rate=control.target_spec.fee_rate * 2),
    ):
        view = dataclasses.replace(aligned.views[SMC_V1], target_spec=replacement)
        broken = dataclasses.replace(aligned, views={**aligned.views, SMC_V1: view})
        with pytest.raises(AlignmentError, match="horizon or costs differ"):
            broken.prove_alignment(FOLDS, SEQ_LEN)


def test_prove_alignment_refuses_views_with_different_segment_ids(aligned):
    """Segment ids are the only per-view input to sample selection, so a view
    holding its own copy could score different rows and must be refused."""
    control = aligned.views[OHLCV14]
    moved = control.segment_ids.copy()
    moved[350:] += 1

    # The refusal is load-bearing: these ids really would select other rows.
    assert not np.array_equal(
        sample_indices(FOLDS[0].inner, SEQ_LEN, HORIZON, segment_ids=moved),
        sample_indices(FOLDS[0].inner, SEQ_LEN, HORIZON, segment_ids=control.segment_ids),
    )

    view = dataclasses.replace(aligned.views[SMC_V1], segment_ids=moved)
    broken = dataclasses.replace(aligned, views={**aligned.views, SMC_V1: view})
    with pytest.raises(AlignmentError, match="segment ids"):
        broken.prove_alignment(FOLDS, SEQ_LEN)


# --------------------------------------------------------------------------- #
# D. nn.p2b plumbing
# --------------------------------------------------------------------------- #
def test_the_committed_manifest_plans_the_declared_four_folds(real_manifest):
    """Literal row numbers, so a change to the fractions has to be argued for
    rather than absorbed."""
    folds, sizes = p2b.plan_from_manifest(real_manifest, 45802)

    assert sizes == {
        "research_rows": 48217,
        "min_train": 21697,
        "inner_val_size": 4821,
        "outer_val_size": 4821,
        "step": 4821,
    }
    assert len(folds) == 4
    assert [plan.train.end for plan in folds] == [21697, 26518, 31339, 36160]
    assert [(plan.outer.start, plan.outer.end) for plan in folds] == [
        (26518, 31339),
        (31339, 36160),
        (36160, 40981),
        (40981, 45802),
    ]
    assert all(plan.train.start == 0 for plan in folds)
    assert all(plan.inner.start == plan.train.end for plan in folds)
    assert all(plan.outer.start == plan.inner.end for plan in folds)


def test_a_snapshot_one_row_short_of_the_plan_is_refused(real_manifest):
    with pytest.raises(p2b.SnapshotError) as excinfo:
        p2b.plan_from_manifest(real_manifest, 45801)
    assert "reaches row 45802" in str(excinfo.value)
    assert "holds only 45801" in str(excinfo.value)


def test_a_manifest_that_disagrees_with_the_plan_is_refused(real_manifest):
    mutated = copy.deepcopy(real_manifest)
    mutated["canonical_reference"]["max_outer_end_row"] = 45803
    with pytest.raises(p2b.SnapshotError, match="max_outer_end_row=45803"):
        p2b.plan_from_manifest(mutated, 45802)


def test_load_snapshot_refuses_a_manifest_that_will_not_declare_itself_styx_free(
    real_manifest, tmp_path
):
    for value in ("false", None, True):
        mutated = copy.deepcopy(real_manifest)
        mutated["contains_styx"] = value
        path = tmp_path / "contains_styx.json"
        path.write_text(json.dumps(mutated))
        with pytest.raises(p2b.SnapshotError, match="contains_styx=false"):
            p2b.load_snapshot(path)


def test_load_snapshot_refuses_a_manifest_declaring_exported_sealed_rows(
    real_manifest, tmp_path
):
    mutated = copy.deepcopy(real_manifest)
    mutated["safety_checks"]["styx_rows_exported"] = 1
    path = tmp_path / "styx_rows.json"
    path.write_text(json.dumps(mutated))
    with pytest.raises(p2b.SnapshotError, match="declares exported sealed rows"):
        p2b.load_snapshot(path)


def test_fold_spread_reports_order_statistics_and_carries_undefined_folds():
    stats = p2b.fold_spread([0.5, None, -0.25, 1.0])
    assert stats["values"] == [0.5, None, -0.25, 1.0]
    assert stats["defined_folds"] == 3
    assert stats["mean"] == pytest.approx((0.5 - 0.25 + 1.0) / 3)
    assert stats["min"] == -0.25
    assert stats["max"] == 1.0
    assert stats["median"] == 0.5
    # A metric no fold defines stays undefined rather than collapsing to zero.
    empty = p2b.fold_spread([None, None])
    assert empty["mean"] is None
    assert (empty["min"], empty["max"], empty["median"]) == (None, None, None)
    assert empty["defined_folds"] == 0


def test_write_predictions_refuses_a_row_at_the_sealed_boundary(tmp_path):
    predictions = pd.DataFrame({"row_index": [10, 11, 12], "p_long": [0.1, 0.2, 0.3]})

    path = p2b.write_predictions(tmp_path, predictions, 13)
    assert path == tmp_path / p2b.PREDICTIONS_NAME
    assert path.is_file()

    with pytest.raises(AssertionError, match="row 12 is at or beyond"):
        p2b.write_predictions(tmp_path, predictions, 12)


def test_run_cell_produces_one_record_per_fold_with_its_provenance(cell):
    records, predictions = cell

    assert [r["fold"] for r in records] == [0, 1]
    for fold, (record, plan) in enumerate(zip(records, FOLDS)):
        assert record["run_seed"] == 42
        assert record["fold_seed"] == 42 + fold
        assert record["periods"]["outer_validation"]["row_range"] == [
            plan.outer.start,
            plan.outer.end,
        ]
        provenance = record["model"]["provenance"]
        assert provenance["model"] == "logistic_regression"
        assert provenance["flattened_input_dim"] == SEQ_LEN * 39
        threading = provenance["threading"]
        assert threading["limit_applied"] == p2b.FIT_THREADS == 1
        assert isinstance(threading["pools"], list)
        for pool in threading["pools"]:
            assert pool["num_threads"] == p2b.FIT_THREADS
        # The rule floors travel with the fold and come from the control view.
        assert set(record["outer_validation"]) == {
            "logistic_regression",
            "majority_baseline",
            "momentum_baseline",
            REFERENCES_KEY,
        }


def test_run_cell_persists_predictions_tagged_with_the_information_set(cell):
    records, predictions = cell

    assert list(predictions.columns)[:3] == ["information_set", "model", "fold"]
    assert set(predictions.columns) == {
        "information_set",
        "model",
        "fold",
        "seed",
        "row_index",
        "timestamp",
        "true_target",
        "future_return",
        "p_short",
        "p_hold",
        "p_long",
        "selected_action",
        "threshold",
    }
    assert (predictions["information_set"] == SMC_V1).all()
    assert (predictions["model"] == "logistic_regression").all()
    for fold, record in enumerate(records):
        rows = predictions[predictions["fold"] == fold]
        assert len(rows) == record["samples"]["outer_validation"]
        assert (rows["threshold"] == record["model"]["selection"]["threshold"]).all()
        # Every persisted row is an outer row of that fold and nothing else.
        assert rows["row_index"].min() >= FOLDS[fold].outer.start
        assert rows["row_index"].max() < FOLDS[fold].outer.end


def test_the_selected_threshold_comes_from_the_inner_block(aligned, cell):
    """Refit fold 0 and re-select from scratch. The recorded threshold has to be
    the one the *inner* rows chose — the outer block never enters the argument."""
    records, _ = cell
    data = aligned.views[SMC_V1]
    plan = FOLDS[0]

    prepared = prepare_research_windows(data, plan.train, plan.inner, SEQ_LEN)
    fitted = fit_simple_model(
        LOGISTIC_REGRESSION, prepared.X_train, prepared.y_train, seed=records[0]["fold_seed"]
    )
    threshold, report = ev.select_threshold(
        fitted.predict_proba(prepared.X_val),
        data.future_return[prepared.idx_val],
        data.target_spec,
        grid=p2b.threshold_grid(),
        min_trades=3,
        row_index=prepared.idx_val,
    )

    selection = records[0]["model"]["selection"]
    assert selection["block"] == "inner_validation"
    assert selection["threshold"] == threshold
    assert selection["inner_validation_trading"] == report
    # The rows the selection saw are inner rows, never outer ones.
    assert prepared.idx_val.min() >= plan.inner.start
    assert prepared.idx_val.max() < plan.inner.end


# --------------------------------------------------------------------------- #
# E. nn.p2b_compare
# --------------------------------------------------------------------------- #
def _fake_cell(information_set: str, model: str) -> dict[str, Any]:
    """A cell shaped like an artifact, small enough to read at a glance."""
    payload = {
        "checkpoint": p2b.CHECKPOINT,
        "information_set": information_set,
        "model": model,
        "outer_predictions": p2b.PREDICTIONS_NAME,
        "contract": {"contract_id": "btc-usdt-1h-gen1", "contract_hash": "dca6"},
        "sizes": {"research_rows": 48217, "min_train": 21697},
        "target": TargetSpec().to_dict(),
        "threshold_selection": {"block": "inner_validation", "min_trades": 10},
        "snapshot": {"rows": 45802, "contains_styx": False},
        "feature_spec": {"combined_spec_hash": "0f0f"},
        "alignment": {
            "folds": [
                {"fold": 0, "outer_validation": {"samples": 4750, "sample_index_sha256": "aa"}}
            ]
        },
        "folds": [
            {
                "fold": 0,
                "samples": {"train": 21000, "inner_validation": 4750, "outer_validation": 4750},
                "periods": {"outer_validation": {"start": "2023-01-01", "end": "2023-07-01"}},
                "outer_validation": {
                    "majority_baseline": {"trading": {"net_return": 0.0}},
                    "momentum_baseline": {"trading": {"net_return": -0.02}},
                    REFERENCES_KEY: {"cash": {"net_return": 0.0}},
                },
            }
        ],
    }
    return {
        "dir": Path("/nonexistent") / f"{information_set}_{model}",
        "information_set": information_set,
        "model": model,
        "payload": payload,
        "predictions": pd.DataFrame(),
    }


def _cell_pair() -> tuple[dict[str, Any], dict[str, Any]]:
    return _fake_cell(OHLCV14, "logistic_regression"), _fake_cell(SMC_V1, "logistic_regression")


def test_check_cells_agree_accepts_two_cells_that_scored_the_same_rows():
    control, other = _cell_pair()
    parity = p2b_compare.check_cells_agree([control, other])
    assert parity["cells"] == [
        "ohlcv14 x logistic_regression",
        "smc_v1 x logistic_regression",
    ]
    assert "every cell scored the same outer rows" in parity["conclusion"]


def test_check_cells_agree_refuses_cells_that_disagree_on_the_contract():
    control, other = _cell_pair()
    other["payload"]["contract"]["contract_hash"] = "beef"
    with pytest.raises(p2b_compare.ComparisonError, match="disagree on 'contract'"):
        p2b_compare.check_cells_agree([control, other])


def test_check_cells_agree_refuses_cells_with_a_different_sample_index_hash():
    control, other = _cell_pair()
    other["payload"]["alignment"]["folds"][0]["outer_validation"]["sample_index_sha256"] = "bb"
    with pytest.raises(p2b_compare.ComparisonError, match="scored a different sample index"):
        p2b_compare.check_cells_agree([control, other])


def test_check_cells_agree_refuses_cells_with_a_different_baseline_report():
    """The baselines are fitted on the control view's columns, so a difference
    between two cells cannot be the model — it can only be the rows."""
    control, other = _cell_pair()
    other["payload"]["folds"][0]["outer_validation"]["momentum_baseline"]["trading"][
        "net_return"
    ] = 0.5
    with pytest.raises(
        p2b_compare.ComparisonError, match="reports a different momentum_baseline"
    ):
        p2b_compare.check_cells_agree([control, other])


def _recomputable_cell(records, predictions) -> dict[str, Any]:
    return {
        "dir": Path("/nonexistent"),
        "information_set": SMC_V1,
        "model": "logistic_regression",
        "payload": {
            "target": TargetSpec(horizon=HORIZON).to_dict(),
            "folds": copy.deepcopy(records),
        },
        "predictions": predictions,
    }


def test_recompute_cell_agrees_with_an_honest_cell(cell):
    records, predictions = cell
    findings = p2b_compare.recompute_cell(_recomputable_cell(records, predictions))
    assert [f["fold"] for f in findings] == [0, 1]
    assert [f["mismatches"] for f in findings] == [[], []]
    assert all(f["recomputed_from"] == p2b.PREDICTIONS_NAME for f in findings)


def test_recompute_cell_reports_a_net_return_edited_away_from_its_predictions(cell):
    """The single assertion this file exists for. A recompute that cannot fail
    is decoration, and the whole nine-cell report rests on this one being able
    to say no."""
    records, predictions = cell
    edited = _recomputable_cell(records, predictions)
    reported = edited["payload"]["folds"][0]["outer_validation"]["logistic_regression"]
    honest = reported["trading"]["net_return"]
    reported["trading"]["net_return"] = honest + 0.05

    findings = p2b_compare.recompute_cell(edited)
    assert findings[1]["mismatches"] == []
    assert len(findings[0]["mismatches"]) == 1
    complaint = findings[0]["mismatches"][0]
    assert complaint.startswith("trading.net_return:")
    assert f"recomputed {honest}" in complaint
    assert f"reported {honest + 0.05}" in complaint


def test_recompute_cell_reports_a_classification_metric_edited_away(cell):
    records, predictions = cell
    edited = _recomputable_cell(records, predictions)
    reported = edited["payload"]["folds"][0]["outer_validation"]["logistic_regression"]
    reported["classification"]["macro_f1"] = reported["classification"]["macro_f1"] + 0.1

    findings = p2b_compare.recompute_cell(edited)
    assert any(m.startswith("classification.macro_f1:") for m in findings[0]["mismatches"])


def _delta_matrix(control_net: list[float], candidate_net: list[float]) -> dict[str, Any]:
    """A four-fold matrix in the shape :func:`nn.p2b_compare.build_matrix` emits."""

    def rows(net: list[float]) -> list[dict[str, Any]]:
        return [
            {
                "fold": fold,
                "period_start": f"202{fold}-01-01",
                "period_end": f"202{fold}-07-01",
                "net_return": value,
                "annualised_sharpe": None if fold == 3 else 0.5 + fold,
                "max_drawdown": 0.1 * fold,
                "exposure": 0.2,
                "n_trades": 10 + fold,
                "macro_f1": 0.3,
                "directional_accuracy": 0.5,
            }
            for fold, value in enumerate(net)
        ]

    return {
        "logistic_regression": {
            OHLCV14: {"folds": rows(control_net)},
            SMC_V1: {"folds": rows(candidate_net)},
        }
    }


@pytest.mark.parametrize("improved", [0, 1, 2, 3, 4])
def test_build_deltas_counts_improved_folds_and_picks_the_stated_verdict(improved):
    control = [0.01, 0.01, 0.01, 0.01]
    candidate = [0.02 if fold < improved else 0.0 for fold in range(4)]
    deltas = p2b_compare.build_deltas(_delta_matrix(control, candidate))

    entry = deltas["logistic_regression"][SMC_V1]
    assert entry["vs"] == OHLCV14
    assert entry["total_folds"] == 4
    assert entry["net_return_improved_folds"] == improved
    assert entry["verdict"] == p2b_compare.VERDICTS[improved]
    assert [d["net_return"] for d in entry["per_fold"]] == [
        round(c - b, 6) for c, b in zip(candidate, control)
    ]
    # The control itself is never differenced against itself.
    assert OHLCV14 not in deltas["logistic_regression"]


def test_build_deltas_carries_an_undefined_metric_through_as_undefined():
    deltas = p2b_compare.build_deltas(
        _delta_matrix([0.01] * 4, [0.02, 0.0, 0.0, 0.0])
    )
    per_fold = deltas["logistic_regression"][SMC_V1]["per_fold"]
    assert per_fold[3]["annualised_sharpe"] is None
    assert per_fold[0]["annualised_sharpe"] == 0.0
    assert deltas["logistic_regression"][SMC_V1]["aggregate"]["annualised_sharpe"][
        "defined_folds"
    ] == 3


def _write_artifact(directory: Path, payload: dict[str, Any]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / p2b.ARTIFACT_NAME).write_text(json.dumps(payload))
    return directory


def test_load_cell_refuses_a_cell_that_names_another_predictions_file(tmp_path):
    payload = _fake_cell(OHLCV14, "logistic_regression")["payload"]
    payload["outer_predictions"] = "predictions.parquet"
    run_dir = _write_artifact(tmp_path / "renamed", payload)
    with pytest.raises(p2b_compare.ComparisonError, match="declares its predictions as"):
        p2b_compare.load_cell(run_dir)


def test_load_cell_refuses_an_artifact_from_another_checkpoint(tmp_path):
    payload = _fake_cell(OHLCV14, "logistic_regression")["payload"]
    payload["checkpoint"] = "P2a"
    run_dir = _write_artifact(tmp_path / "p2a", payload)
    with pytest.raises(p2b_compare.ComparisonError, match="is checkpoint 'P2a', not P2b"):
        p2b_compare.load_cell(run_dir)
