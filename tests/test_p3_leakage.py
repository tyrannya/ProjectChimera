"""P3's causal guarantees, and the corruptions each one is supposed to catch.

``docs/microstructure_v1.md`` §2 states one rule — a row may read its own hour
and the hours strictly before it in the same segment, and nothing else — and §4
lists the properties that follow. This file is those properties, each paired
with the attack it is meant to survive and with the **positive control** that
the unattacked input passes. A guard that always fired would satisfy every
corruption test here on its own, which is why none of them stands alone.

The two that matter most are at the end. A verifier that runs in a ``make``
target is a convention; a verifier that runs in the only function that loads the
data is a guarantee. So one test proves a corrupt trade snapshot produces
**exactly zero model fits**, and its falsifying twin proves a valid one reaches
a real fit call — because "zero fits" is also what a runner that never fits
anything would report.

Nothing here fits a model on synthetic data. The fit spy records the call and
raises, so the fit path is proven reachable without a synthetic number ever
being produced, let alone recorded as evidence.
"""

from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd
import pytest

from chimera.contracts import TargetSpec
from chimera.features import FeatureSpec
from nn import p2b
from nn.data_pipeline import build_dataset
from nn.information_sets import (
    CHECKPOINTS,
    MICROSTRUCTURE_V1,
    OHLCV14,
    OHLCV14_PLUS_MICROSTRUCTURE,
    P3_INFORMATION_SETS,
    build_information_set_views,
    information_set,
    microstructure_spec_identity,
)
from nn.microstructure import (
    MicrostructureSpec,
    compute_microstructure_features,
    microstructure_feature_columns,
)
from nn.p2b_compare import ComparisonError, check_cells_agree
from nn.research_contract import load_contract
from nn.trade_aggregates import HourlyTradeAggregator
from tools.export_trade_snapshot import MANIFEST_NAME
from tools.make_sample_data import generate_candles, generate_trades

BTC = load_contract("btc-usdt-1h-gen1")
STYX = BTC.sealed_test_start
STYX_NS = int(STYX.value)
SHORT = MicrostructureSpec().with_windows(w_short=3, w_long=6)


def aggregate(ts, price, qty, maker, ids) -> pd.DataFrame:
    aggregator = HourlyTradeAggregator(styx_ns=STYX_NS)
    aggregator.add_chunk(ts, price, qty, maker, ids)
    return aggregator.finish()


# --------------------------------------------------------------------------- #
# A. append and future-mutation invariance
# --------------------------------------------------------------------------- #
def test_appending_hours_changes_no_earlier_row():
    """The property that makes a growing source safe: the past does not move.

    Built by *truncating* one series rather than by generating two, because two
    generator calls with different lengths are two different random streams and
    would compare unrelated trades. The short table plus the hours after it is
    the long table, exactly, which is what "appending" means.
    """
    ts, price, qty, maker, ids = generate_trades(hours=45, seed=61, trades_per_hour=10)
    early = (ts - ts[0]) // 3_600_000_000_000 < 30
    assert early.any() and not early.all()

    before = compute_microstructure_features(
        aggregate(ts[early], price[early], qty[early], maker[early], ids[early]), SHORT
    )
    after = compute_microstructure_features(aggregate(ts, price, qty, maker, ids), SHORT)
    assert len(after) > len(before)
    pd.testing.assert_frame_equal(before, after.iloc[: len(before)])


def test_mutating_the_future_changes_no_earlier_row():
    """Editing hours after T must leave every row at or before T byte-identical."""
    ts, price, qty, maker, ids = generate_trades(hours=40, seed=62, trades_per_hour=12)
    cutoff_hour = 25
    hour_of = (ts - ts[0]) // 3_600_000_000_000
    future = hour_of >= cutoff_hour
    assert future.any() and (~future).any()

    original = compute_microstructure_features(aggregate(ts, price, qty, maker, ids), SHORT)

    tampered_price = price.copy()
    tampered_qty = qty.copy()
    tampered_maker = maker.copy()
    tampered_price[future] *= 3.0
    tampered_qty[future] *= 17.0
    tampered_maker[future] = ~tampered_maker[future]
    mutated = compute_microstructure_features(
        aggregate(ts, tampered_price, tampered_qty, tampered_maker, ids), SHORT
    )

    kept = int(np.unique(hour_of[~future]).size)
    pd.testing.assert_frame_equal(original.iloc[:kept], mutated.iloc[:kept])
    # The positive control: the mutation did reach the rows after the cutoff, so
    # the equality above is causality rather than a no-op.
    assert not original.iloc[kept:].equals(mutated.iloc[kept:])


def test_trade_hours_after_the_spine_cannot_reach_a_joined_feature(spine_and_trades):
    """Causality at the join, made structural rather than inherited.

    The trade table is cut at the spine's last hour *before* the engine runs, so
    a future feature written with a centred window could not read a later hour
    even if it tried. This appends a wildly different fortnight past the spine
    and asserts that not one joined value moves.
    """
    frame, ds_meta, candles, aggregates = spine_and_trades
    tail = aggregates.tail(14 * 24).copy().reset_index(drop=True)
    tail["date"] = (
        tail["date"]
        + (pd.Timestamp(aggregates["date"].iloc[-1]) - pd.Timestamp(tail["date"].iloc[0]))
        + pd.Timedelta(hours=1)
    )
    for name in ("qty", "buy_qty", "notional", "buy_notional"):
        tail[name] = tail[name] * 91.0
    extended = pd.concat([aggregates, tail], ignore_index=True)

    plain = build_information_set_views(
        frame, ds_meta, candles, names=(MICROSTRUCTURE_V1,), trade_aggregates=aggregates
    )
    with_future = build_information_set_views(
        frame, ds_meta, candles, names=(MICROSTRUCTURE_V1,), trade_aggregates=extended
    )
    assert np.array_equal(
        plain.views[MICROSTRUCTURE_V1].features,
        with_future.views[MICROSTRUCTURE_V1].features,
    )


# --------------------------------------------------------------------------- #
# B. the alignment layer
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def spine_and_trades():
    """A processed spine and a trade table covering every one of its hours."""
    candles = generate_candles(rows=900, seed=71, start="2021-06-01")
    frame, ds_meta = build_dataset(
        candles,
        FeatureSpec(),
        TargetSpec(horizon=6),
        exchange="synthetic",
        pair="SYNTH/USDT",
        timeframe="1h",
    )
    start = pd.Timestamp(candles["date"].iloc[0]).tz_convert("UTC")
    hours = (
        int(
            (pd.Timestamp(candles["date"].iloc[-1]).tz_convert("UTC") - start)
            / pd.Timedelta(hours=1)
        )
        + 1
    )
    trades = generate_trades(hours=hours, seed=72, start=start, trades_per_hour=10)
    return frame, ds_meta, candles, aggregate(*trades)


def test_a_p3_arm_joins_onto_the_spine_and_proves_its_join(spine_and_trades):
    frame, ds_meta, candles, aggregates = spine_and_trades
    aligned = build_information_set_views(
        frame, ds_meta, candles, names=P3_INFORMATION_SETS, trade_aggregates=aggregates
    )
    evidence = aligned.join_evidence["microstructure"]
    assert evidence["matches"] is True
    assert evidence["matches_under_plus_one_shift"] is False
    assert evidence["matches_under_minus_one_shift"] is False
    assert evidence["rows"] == len(frame)
    assert evidence["warmup_hours_before_the_spine"] > 0
    assert set(aligned.views) == set(P3_INFORMATION_SETS)
    assert aligned.views[MICROSTRUCTURE_V1].feature_names == microstructure_feature_columns()
    assert aligned.micro_spec is not None


def test_a_p3_arm_without_a_trade_source_is_refused_by_name(spine_and_trades):
    frame, ds_meta, candles, _ = spine_and_trades
    with pytest.raises(ValueError, match="given no trade source"):
        build_information_set_views(
            frame, ds_meta, candles, names=(MICROSTRUCTURE_V1,), trade_aggregates=None
        )


def test_a_spine_hour_with_no_trade_row_fails_closed(spine_and_trades):
    frame, ds_meta, candles, aggregates = spine_and_trades
    holed = aggregates.drop(index=aggregates.index[len(aggregates) // 2]).reset_index(
        drop=True
    )
    with pytest.raises(ValueError, match="no trade-aggregate hour"):
        build_information_set_views(
            frame, ds_meta, candles, names=(MICROSTRUCTURE_V1,), trade_aggregates=holed
        )


def test_a_rolled_microstructure_matrix_is_rejected(spine_and_trades, monkeypatch):
    """The attack no shared-array check can see: every column one hour early.

    A shift of one row is a one-hour look-ahead on all 32 columns at once, and
    it leaves the shape, the row count and every other array in the alignment
    proof untouched.
    """
    frame, ds_meta, candles, aggregates = spine_and_trades
    real = compute_microstructure_features

    def rolled(table, spec):
        computed = real(table, spec)
        # `np.roll`, not `shift`: a shift leaves a NaN behind and would be
        # caught by the finiteness guard instead of by the join check, which
        # would make this test pass for the wrong reason.
        return pd.DataFrame(
            np.roll(computed.to_numpy(), -1, axis=0),
            columns=computed.columns,
            index=computed.index,
        )

    monkeypatch.setattr("nn.information_sets.compute_microstructure_features", rolled)
    with pytest.raises(ValueError, match="microstructure join is wrong"):
        build_information_set_views(
            frame, ds_meta, candles, names=(MICROSTRUCTURE_V1,), trade_aggregates=aggregates
        )


def test_the_unrolled_matrix_is_accepted(spine_and_trades):
    """The positive control for the test above."""
    frame, ds_meta, candles, aggregates = spine_and_trades
    aligned = build_information_set_views(
        frame, ds_meta, candles, names=(MICROSTRUCTURE_V1,), trade_aggregates=aggregates
    )
    assert aligned.join_evidence["microstructure"]["matches"] is True


def test_a_non_finite_microstructure_value_fails_closed(spine_and_trades, monkeypatch):
    frame, ds_meta, candles, aggregates = spine_and_trades
    real = compute_microstructure_features

    def poisoned(table, spec):
        computed = real(table, spec).copy()
        # A row the join actually selects. The first few hours of the table are
        # warm-up that precedes the spine, and poisoning one of those would
        # prove only that the guard cannot see rows nothing reads.
        computed.iloc[-3, 0] = np.nan
        return computed

    monkeypatch.setattr("nn.information_sets.compute_microstructure_features", poisoned)
    with pytest.raises(ValueError, match="non-finite microstructure"):
        build_information_set_views(
            frame, ds_meta, candles, names=(MICROSTRUCTURE_V1,), trade_aggregates=aggregates
        )


def test_a_spine_segment_the_trade_source_contradicts_is_refused():
    """The spine and the trade source must agree about where the gaps are.

    Exercised as a unit rather than through `build_information_set_views`, and
    the reason is worth writing down: at integration level this check is
    unreachable on well-formed inputs. The join already requires every spine
    hour to have a trade row, and `_check_segment_contiguity` already requires
    two rows of one spine segment to be adjacent *candles* — between them those
    two force the trade rows to be adjacent too. So this is defence in depth
    against a future join that relaxes one of them, and testing it through a
    contrived integration scenario would prove less, not more, because the
    scenario would have to be one the other guards cannot produce.
    """
    from nn.information_sets import _check_trade_segment_contiguity

    spine = pd.DataFrame({"segment_id": np.array([0, 0, 0, 0], dtype=np.int64)})
    # The positive control first: adjacent trade rows for one segment are fine.
    _check_trade_segment_contiguity(spine, np.array([10, 11, 12, 13], dtype=np.int64))
    # And a two-hour jump inside one claimed segment is not.
    with pytest.raises(ValueError, match="hours apart in the trade source"):
        _check_trade_segment_contiguity(spine, np.array([10, 11, 13, 14], dtype=np.int64))
    # A jump *between* segments is allowed, because the windowing already knows.
    across = pd.DataFrame({"segment_id": np.array([0, 0, 1, 1], dtype=np.int64)})
    _check_trade_segment_contiguity(across, np.array([10, 11, 13, 14], dtype=np.int64))


def test_the_alignment_proof_covers_a_p3_arm(spine_and_trades):
    from nn.walkforward import plan_nested_folds

    frame, ds_meta, candles, aggregates = spine_and_trades
    aligned = build_information_set_views(
        frame, ds_meta, candles, names=P3_INFORMATION_SETS, trade_aggregates=aggregates
    )
    folds = plan_nested_folds(len(frame), 2, int(len(frame) * 0.5), 80, 80, 80)
    evidence = aligned.prove_alignment(folds, 16)
    assert evidence["information_sets"] == list(P3_INFORMATION_SETS)
    assert len(evidence["folds"]) == 2
    for fold in evidence["folds"]:
        assert fold["outer_validation"]["samples"] > 0
        assert len(fold["outer_validation"]["sample_index_sha256"]) == 64


# --------------------------------------------------------------------------- #
# C. the checkpoint's identity
# --------------------------------------------------------------------------- #
def test_p3_declares_the_question_the_checkpoint_was_opened_to_ask():
    checkpoint = CHECKPOINTS["P3"]
    assert checkpoint.arms == (OHLCV14, MICROSTRUCTURE_V1, OHLCV14_PLUS_MICROSTRUCTURE)
    assert checkpoint.control == OHLCV14
    assert checkpoint.family == MICROSTRUCTURE_V1
    assert checkpoint.trade_source is True
    assert checkpoint.question == (
        "does causal microstructure_v1, alone or combined with OHLCV14, add usable "
        "information beyond OHLCV14?"
    )
    assert checkpoint.question != CHECKPOINTS["P2b"].question
    assert checkpoint.question != CHECKPOINTS["P2c"].question


def test_only_p3_reads_the_trade_source():
    assert [name for name, c in CHECKPOINTS.items() if c.trade_source] == ["P3"]


def test_the_combined_arm_is_ohlcv14_then_microstructure_in_order():
    combined = information_set(OHLCV14_PLUS_MICROSTRUCTURE)
    control = information_set(OHLCV14)
    family = information_set(MICROSTRUCTURE_V1)
    assert combined.columns == control.columns + family.columns
    assert len(combined.columns) == 14 + 32


# --------------------------------------------------------------------------- #
# D. the comparator refuses cells measured on different sources
# --------------------------------------------------------------------------- #
def _cell(name: str, **overrides):
    payload = {
        "checkpoint": "P3",
        "question": CHECKPOINTS["P3"].question,
        "code": {"source_digest": "a" * 64},
        "contract": {"contract_id": "btc-usdt-1h-gen1"},
        "sizes": {"research_rows": 10},
        "target": {"horizon": 6},
        "threshold_selection": {"grid": [0.4]},
        "snapshot": {"rows": 10},
        "trade_snapshot": {"semantic_hash": "b" * 64},
        "microstructure_spec": {"microstructure_spec_hash": "c" * 64},
        "feature_spec": {"combined_spec_hash": "d" * 64},
        "alignment": {"folds": [{"fold": 0}]},
        "folds": [
            {
                "samples": {"outer_validation": 3},
                "periods": {},
                "outer_validation": {
                    "majority_baseline": {},
                    "momentum_baseline": {},
                    "economic_references": {},
                },
            }
        ],
    }
    payload.update(overrides)
    return {"information_set": name, "model": "xgboost", "payload": payload}


def test_two_cells_from_one_trade_source_agree():
    """The positive control: the checker is not simply always-refuse."""
    agreement = check_cells_agree([_cell(OHLCV14), _cell(MICROSTRUCTURE_V1)])
    assert agreement["conclusion"].startswith("every cell scored the same outer rows")


@pytest.mark.parametrize(
    "key,value",
    [
        ("trade_snapshot", {"semantic_hash": "e" * 64}),
        ("microstructure_spec", {"microstructure_spec_hash": "f" * 64}),
    ],
)
def test_cells_measured_on_different_trade_sources_are_refused(key, value):
    with pytest.raises(ComparisonError, match=key):
        check_cells_agree([_cell(OHLCV14), _cell(MICROSTRUCTURE_V1, **{key: value})])


def test_a_cell_with_no_trade_source_cannot_join_one_that_has_it():
    """An absence is a different identity, not a wildcard."""
    stripped = _cell(MICROSTRUCTURE_V1)
    stripped["payload"].pop("trade_snapshot")
    with pytest.raises(ComparisonError, match="trade_snapshot"):
        check_cells_agree([_cell(OHLCV14), stripped])


def test_model_specific_source_digests_accept_same_clean_revision():
    control = _cell(OHLCV14)
    other = _cell(MICROSTRUCTURE_V1)
    revision = "1" * 40
    control["payload"]["code"].update({"revision": revision, "dirty": False})
    other["payload"]["code"].update(
        {"source_digest": "e" * 64, "revision": revision, "dirty": False}
    )

    agreement = check_cells_agree([control, other])
    assert agreement["code"]["source_digest"] is None
    assert agreement["code"]["source_digests"] == ["a" * 64, "e" * 64]


def test_different_source_digests_refuse_different_revisions():
    control = _cell(OHLCV14)
    other = _cell(MICROSTRUCTURE_V1)
    control["payload"]["code"].update({"revision": "1" * 40, "dirty": False})
    other["payload"]["code"].update(
        {"source_digest": "e" * 64, "revision": "2" * 40, "dirty": False}
    )

    with pytest.raises(ComparisonError, match="same clean"):
        check_cells_agree([control, other])


def test_different_source_digests_refuse_dirty_cells():
    control = _cell(OHLCV14)
    other = _cell(MICROSTRUCTURE_V1)
    revision = "1" * 40
    control["payload"]["code"].update({"revision": revision, "dirty": False})
    other["payload"]["code"].update(
        {"source_digest": "e" * 64, "revision": revision, "dirty": True}
    )

    with pytest.raises(ComparisonError, match="same clean"):
        check_cells_agree([control, other])


def test_the_microstructure_identity_records_both_halves():
    identity = microstructure_spec_identity(
        MicrostructureSpec(), {"semantic_hash": "b" * 64, "venue": "binance"}
    )
    assert identity["microstructure_spec_version"] == "microstructure_v1"
    assert len(identity["microstructure_spec_hash"]) == 64
    assert identity["trade_source"]["semantic_hash"] == "b" * 64
    other = microstructure_spec_identity(
        MicrostructureSpec(), {"semantic_hash": "0" * 64, "venue": "binance"}
    )
    assert other["combined_microstructure_hash"] != identity["combined_microstructure_hash"]


# --------------------------------------------------------------------------- #
# E. the runner verifies before it fits — and the falsifying twin
# --------------------------------------------------------------------------- #
class FitSpy:
    """Records that a fit was reached, then stops the run before one happens.

    Raising rather than returning a stub is deliberate. This suite must never
    produce a fitted number from synthetic trades — a synthetic P3 result is
    exactly the thing this checkpoint may not manufacture — and a spy that
    completed the fold would produce metrics, an artifact and a predictions file
    that look like evidence.
    """

    class Reached(RuntimeError):
        pass

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        raise FitSpy.Reached("the fit path was reached")


@pytest.fixture
def runnable(synthetic_trade_snapshot, tmp_path):
    """A writable copy of both snapshots, and the arguments to run against them."""
    tree = tmp_path / "tree"
    shutil.copytree(synthetic_trade_snapshot["root"], tree)
    research = tree / "data" / "research"
    return {
        "tree": tree,
        "ohlcv": research / "synthetic_snapshot_manifest.json",
        "trades": research / MANIFEST_NAME,
        "out": tmp_path / "cell",
    }


def argv(runnable, arm: str = MICROSTRUCTURE_V1) -> list[str]:
    return [
        "--checkpoint",
        "P3",
        "--information-set",
        arm,
        "--model",
        "logistic_regression",
        "--out",
        str(runnable["out"]),
        "--manifest",
        str(runnable["ohlcv"]),
        "--trade-manifest",
        str(runnable["trades"]),
        "--research-contract",
        "synth-usdt-1h-gen1",
        "--seq-len",
        "16",
        "--min-trades",
        "2",
    ]


def test_a_valid_snapshot_reaches_a_real_fit_call(runnable, monkeypatch):
    """The falsifying twin. "Zero fits" is also what a broken runner reports."""
    spy = FitSpy()
    monkeypatch.setattr(p2b, "fit_simple_model", spy)
    with pytest.raises(FitSpy.Reached):
        p2b.main(argv(runnable))
    assert spy.calls == 1


@pytest.mark.parametrize(
    "mutate,expect",
    [
        (lambda p: p["aggregate"].__setitem__("semantic_hash", "0" * 64), "semantic_hash"),
        (lambda p: p["aggregate"].__setitem__("sha256", "0" * 64), "aggregate_sha256"),
        (lambda p: p.__setitem__("contains_styx", True), "contains_styx"),
        (lambda p: p["trade_stream"].__setitem__("duplicate_ids", 1), "stream_integrity"),
        (lambda p: p["source"].__setitem__("venue", "kraken"), "source_identity"),
        (
            lambda p: p["alignment"].__setitem__(
                "ohlcv_processed_semantic_prefix_hash", "0" * 64
            ),
            "ohlcv_binding",
        ),
    ],
)
def test_a_corrupt_trade_snapshot_produces_exactly_zero_fits(
    runnable, monkeypatch, mutate, expect
):
    spy = FitSpy()
    monkeypatch.setattr(p2b, "fit_simple_model", spy)
    payload = json.loads(runnable["trades"].read_text())
    mutate(payload)
    runnable["trades"].write_text(json.dumps(payload, indent=2) + "\n")

    with pytest.raises(SystemExit) as excinfo:
        p2b.main(argv(runnable))
    assert spy.calls == 0
    assert expect in str(excinfo.value)
    assert not (runnable["out"] / "p2b.json").exists()


def test_a_sealed_hour_stops_the_run_before_any_fit_and_says_nothing_about_it(
    runnable, monkeypatch
):
    """The seal, at the only layer that matters: the one that loads the data."""
    import hashlib

    spy = FitSpy()
    monkeypatch.setattr(p2b, "fit_simple_model", spy)
    payload = json.loads(runnable["trades"].read_text())
    path = runnable["tree"] / payload["aggregate"]["path"]
    frame = pd.read_parquet(path)
    marker = 424242.42
    frame.loc[frame.index[-1], "date"] = STYX
    frame.loc[frame.index[-1], "last_price"] = marker
    frame.loc[frame.index[-1], "high_price"] = marker
    frame.to_parquet(path, index=False)
    payload["aggregate"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    runnable["trades"].write_text(json.dumps(payload, indent=2) + "\n")

    with pytest.raises(SystemExit) as excinfo:
        p2b.main(argv(runnable))
    assert spy.calls == 0
    message = str(excinfo.value)
    assert "aggregate_before_styx" in message
    assert "withheld" in message
    assert "424242" not in message


def test_a_missing_trade_snapshot_names_the_tool_that_produces_it(runnable, monkeypatch):
    spy = FitSpy()
    monkeypatch.setattr(p2b, "fit_simple_model", spy)
    runnable["trades"].unlink()
    with pytest.raises(SystemExit) as excinfo:
        p2b.main(argv(runnable))
    assert spy.calls == 0
    assert "tools.export_trade_snapshot" in str(excinfo.value)


def test_a_p2_checkpoint_never_opens_the_trade_snapshot(runnable, monkeypatch):
    """P2b and P2c read the candles alone, and must not acquire a P3 provenance."""
    opened: list[str] = []
    monkeypatch.setattr(
        p2b,
        "load_trade_snapshot",
        lambda path: opened.append(str(path)) or (_ for _ in ()).throw(AssertionError),
    )
    spy = FitSpy()
    monkeypatch.setattr(p2b, "fit_simple_model", spy)
    arguments = argv(runnable, arm=OHLCV14)
    arguments[1] = "P2b"
    with pytest.raises(FitSpy.Reached):
        p2b.main(arguments)
    assert opened == []
    assert spy.calls == 1
