"""P6's preregistration, held to the things it claims about itself.

Two kinds of test live here. The first kind checks the design is internally
coherent and matches the document beside it — a preregistration whose
machine-readable twin and prose disagree is two designs, and the reader has no
way to know which one ran.

The second kind is a **tripwire**: while this file is the whole of P6, no P6
artifact may exist. Registration is not permission. The tripwire is flipped in
the commit that produces the evidence, and until then it fails the moment a cell
appears.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from nn.multiclock import RESEARCH_VISIBLE_END, STYX_START
from nn.p6_preregistration import (
    CHECKPOINT,
    CLOCKS,
    COSTS,
    FOLD_PERIODS,
    HORIZON_BARS,
    HORIZONS,
    MEASURED_UNIVERSE,
    MODELS,
    OUTER_PERIODS,
    PRIMARY_MODEL,
    REGION,
    SEED,
    SEQ_LEN,
    VIABILITY_GATE,
    describe,
    payload,
    preregistration_hash,
)

REPO = Path(__file__).resolve().parents[1]
DOCUMENT = REPO / "docs" / "p6_preregistration.md"

#: The hash the document publishes and every P6 cell must record. Written down
#: here so that moving any constant in the twin fails this test rather than
#: silently producing cells under a design nobody reviewed.
FROZEN_HASH = "sha256:2785b1a7b19ecceca58cd0e936d14a5cbbbe6eb10f7ddf2796800409a0eaaaf2"


# --------------------------------------------------------------------------- #
# A. the design is what it says it is
# --------------------------------------------------------------------------- #


def test_the_five_clocks_are_frozen():
    assert CLOCKS == ("1m", "5m", "15m", "30m", "1h")
    assert set(HORIZONS) == set(CLOCKS)


def test_the_horizon_is_six_native_bars_on_every_clock():
    assert HORIZON_BARS == 6
    expected = {
        "1m": "6 minutes",
        "5m": "30 minutes",
        "15m": "90 minutes",
        "30m": "3 hours",
        "1h": "6 hours",
    }
    assert HORIZONS == expected
    # Each horizon really is six of that clock's bars, checked arithmetically
    # rather than trusted to the prose.
    for clock, text in HORIZONS.items():
        native = pd.Timedelta(clock.replace("m", "min") if clock.endswith("m") else clock)
        assert pd.Timedelta(text) == 6 * native


def test_costs_are_the_established_model_in_basis_points():
    assert COSTS["fee_rate"] == 0.0005
    assert COSTS["slippage_rate"] == 0.0005
    assert COSTS["cost_threshold"] == pytest.approx(
        2.0 * (COSTS["fee_rate"] + COSTS["slippage_rate"])
    )
    assert COSTS["cost_threshold"] == 0.002


def test_the_model_families_are_the_established_three():
    assert MODELS == ("logistic_regression", "lightgbm", "xgboost")
    assert PRIMARY_MODEL == "xgboost"
    assert SEED == 42


def test_one_bar_of_features_on_every_clock():
    assert SEQ_LEN == 1


def test_the_gate_requires_all_three_conditions():
    assert VIABILITY_GATE["conjunction"] == "all three"
    assert len(VIABILITY_GATE["conditions"]) == 3
    assert VIABILITY_GATE["positive_folds_required"] == 3
    assert VIABILITY_GATE["beats_momentum_folds_required"] == 3
    assert VIABILITY_GATE["total_folds"] == 4
    assert VIABILITY_GATE["decided_by"] == PRIMARY_MODEL
    assert VIABILITY_GATE["per_clock"] is True


def test_the_hash_is_frozen():
    """Moving any constant in the twin moves this, which is the point."""
    assert preregistration_hash() == FROZEN_HASH
    assert describe()["preregistration_hash"] == FROZEN_HASH


@pytest.mark.parametrize("key", sorted(payload()))
def test_every_payload_key_is_populated(key):
    assert payload()[key] not in (None, "", [], {})


def test_the_document_publishes_the_same_hash():
    text = DOCUMENT.read_text()
    assert FROZEN_HASH in text
    assert f"# {CHECKPOINT} — preregistration" in text


def test_the_document_names_all_five_clocks_and_horizons():
    text = DOCUMENT.read_text()
    for clock, horizon in HORIZONS.items():
        assert f"`{clock}`" in text
        assert horizon in text


# --------------------------------------------------------------------------- #
# B. the folds are the same four real-world periods, and they derive
# --------------------------------------------------------------------------- #


def test_fold_periods_recompute_from_the_committed_1h_snapshot():
    """The freeze is checkable: these instants are the 1h plan, rendered.

    `nn.p2b.plan_from_manifest` is the planner every checkpoint since P2b has
    run. Its output is row indices into the committed 1h outer coverage; this
    reads those rows' timestamps and asserts they are exactly what P6 froze.
    """
    from nn.p2b import DEFAULT_MANIFEST, plan_from_manifest

    manifest = json.loads(DEFAULT_MANIFEST.read_text())
    coverage = pd.read_parquet(
        REPO / manifest["processed_outer_coverage"]["path"], columns=["date"]
    )
    dates = pd.to_datetime(coverage["date"], utc=True)
    folds, _ = plan_from_manifest(manifest, len(coverage))
    assert len(folds) == len(FOLD_PERIODS)

    for plan, frozen in zip(folds, FOLD_PERIODS):
        assert dates.iloc[plan.train.start].isoformat() == frozen["train_start"]
        assert dates.iloc[plan.inner.start].isoformat() == frozen["inner_start"]
        assert dates.iloc[plan.outer.start].isoformat() == frozen["outer_start"]
        # The outer block is half-open, so its end is the instant after its last
        # bar opens. The final fold's is the research boundary itself.
        last_open = dates.iloc[plan.outer.end - 1]
        assert (last_open + pd.Timedelta(hours=1)).isoformat() == frozen["outer_end"]


def test_the_last_outer_block_ends_exactly_at_the_research_boundary():
    assert pd.Timestamp(FOLD_PERIODS[-1]["outer_end"]) == RESEARCH_VISIBLE_END
    assert pd.Timestamp(REGION["end_exclusive"]) == RESEARCH_VISIBLE_END
    assert RESEARCH_VISIBLE_END < STYX_START


def test_outer_blocks_tile_forward_and_never_overlap():
    for earlier, later in zip(OUTER_PERIODS, OUTER_PERIODS[1:]):
        assert earlier[1] == later[0]
    for start, end in OUTER_PERIODS:
        assert pd.Timestamp(start) < pd.Timestamp(end)


def test_training_never_reaches_its_own_inner_or_outer_block():
    for row in FOLD_PERIODS:
        assert pd.Timestamp(row["train_start"]) == pd.Timestamp(REGION["start"])
        assert pd.Timestamp(row["inner_start"]) < pd.Timestamp(row["outer_start"])
        assert pd.Timestamp(row["outer_start"]) < pd.Timestamp(row["outer_end"])


def test_no_frozen_instant_reaches_the_boundary_or_styx():
    for row in FOLD_PERIODS:
        for key in ("train_start", "inner_start", "outer_start"):
            assert pd.Timestamp(row[key]) < RESEARCH_VISIBLE_END
        assert pd.Timestamp(row["outer_end"]) <= RESEARCH_VISIBLE_END


def test_the_upstream_parity_disagreements_reach_no_scored_block():
    """The claim `docs/multiclock_v1.md` §6 makes, checked against every block.

    Outer blocks are what the verdicts rest on, but they are not the only blocks
    whose numbers a cell publishes: the inner block selects each fold's
    threshold, and a cell records its size and period. A guard that checked only
    the outer blocks would leave "no block whose numbers a checkpoint reports"
    half-proved, so this checks the inner blocks too. All 29 fall in training
    windows alone.
    """
    manifest = json.loads(
        (REPO / "data/research/btc_usdt_multiclock_gen2_manifest.json").read_text()
    )
    stamps = [pd.Timestamp(v) for v in manifest["parity_1h"]["mismatching_timestamps"]]
    assert stamps

    for row in FOLD_PERIODS:
        for name, start, end in (
            ("inner", row["inner_start"], row["outer_start"]),
            ("outer", row["outer_start"], row["outer_end"]),
        ):
            inside = [s for s in stamps if pd.Timestamp(start) <= s < pd.Timestamp(end)]
            assert inside == [], (
                f"{len(inside)} disagreeing hour(s) inside fold {row['fold']}'s "
                f"{name} block {start} .. {end}"
            )

    # Every one of them precedes the earliest instant any fold scores or selects
    # on, which is fold 0's inner block.
    earliest_scored = min(pd.Timestamp(row["inner_start"]) for row in FOLD_PERIODS)
    assert max(stamps) < earliest_scored
    assert max(stamps) < pd.Timestamp(OUTER_PERIODS[0][0])


def test_the_measured_universe_covers_every_clock():
    assert set(MEASURED_UNIVERSE) == set(CLOCKS)
    # HOLD share falls monotonically as the bar lengthens, because a fixed
    # 20 bps threshold is a shrinking fraction of a longer bar's move. Recorded
    # before any fit and asserted here so that it cannot later be discovered.
    shares = [MEASURED_UNIVERSE[clock]["hold_fraction"] for clock in CLOCKS]
    assert shares == sorted(shares, reverse=True)
    rows = [MEASURED_UNIVERSE[clock]["region_rows"] for clock in CLOCKS]
    assert rows == sorted(rows, reverse=True)


# --------------------------------------------------------------------------- #
# C. tripwires: registration is not permission
# --------------------------------------------------------------------------- #

#: Flipped in the commit that produces P6 evidence. While it is False, a P6
#: artifact appearing anywhere fails the suite.
P6_EVIDENCE_EXPECTED = True


def test_no_p6_artifact_exists_before_the_evidence_commit():
    if P6_EVIDENCE_EXPECTED:
        pytest.skip("P6 evidence is expected; its own tests check it")
    found = sorted(p.name for p in (REPO / "artifacts" / "benchmark").glob("btc_p6*"))
    assert (
        found == []
    ), f"P6 artifacts exist while the preregistration is the whole of P6: {found}"


def test_the_research_state_reports_p6_as_preregistered_or_answered():
    from nn.research_state import CHECKPOINTS, checkpoint_states

    names = [checkpoint.name for checkpoint in CHECKPOINTS]
    assert "P6" in names
    state = checkpoint_states(REPO)["P6"]
    assert state in {"preregistered", "answered"}
    if not P6_EVIDENCE_EXPECTED:
        assert state == "preregistered"


def _flat(text: str) -> str:
    """Markdown prose as one comparable line: no wrapping, emphasis or backticks."""
    return re.sub(r"\s+", " ", text.replace("`", "").replace("*", "")).lower()


def test_every_forbidden_item_appears_in_the_document():
    """The twin's prohibitions and the document's are one list, not two.

    Matched item by item rather than by counting: a prohibition that exists only
    in the module is one a reader of the document never sees, and a preregistration
    a reader cannot check is not one.
    """
    from nn.p6_preregistration import FORBIDDEN_AFTER_RESULTS

    section = _flat(DOCUMENT.read_text().split("## 10. Forbidden")[1].split("## 11.")[0])
    for item in FORBIDDEN_AFTER_RESULTS:
        assert _flat(item) in section, f"the document does not forbid: {item}"


def test_the_document_states_the_gate_and_reports_every_clock():
    text = _flat(DOCUMENT.read_text())
    assert "3 of the 4" in text
    assert "five separate specialist verdicts" in text
    assert "native-timeframe momentum baseline" in text
    assert "reference, not a required competitor" in text
    # The gate's three conditions, each identifiable in the prose.
    assert "cost-aware outer net return > 0 in at least 3 of the 4 folds" in text
    assert "mean outer cost-aware net return across the four folds > 0" in text


def test_the_document_has_no_unresolved_placeholder():
    text = DOCUMENT.read_text()
    assert not re.search(r"\bTODO\b|\bTBD\b|XXX", text)
