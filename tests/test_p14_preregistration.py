"""P14's design, and the one fact about P14 that is currently true: it is not opened.

Most of this file is the usual preregistration coherence check. The tests that
matter are the ones asserting **P14 has not been opened**: no signal module, no
P14 artifact carrying a number, no P14 result anywhere, and a research state that
says `preregistered` and never `answered`.

The rest pins the constants a result could later be argued into or out of --
the sign, the clock, the horizon, the gate conditions, the theta grid, the cost
model and the activity floor -- so that moving one of them moves the hash and is
visible in Git rather than silent.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from nn.p14_preregistration import (
    CHECKPOINT,
    CLOCK,
    COSTS,
    CURRENT_RESULT_STATE,
    EXTERNAL_ANCHOR,
    FORBIDDEN_AFTER_RESULTS,
    HORIZON_BARS,
    INNER_BLOCKS,
    MINIMUM_ACTIVITY,
    OUTCOME,
    OUTER_BLOCKS,
    RESULT_STATES,
    SAFETY_PROHIBITIONS,
    SIGNAL_NAME,
    SOURCE,
    STAGE0,
    STAGE1,
    STAGE2_GATED_ON_STAGE1,
    THRESHOLD_SELECTION,
    VIABILITY_GATE,
    describe,
    payload,
    preregistration_hash,
)
from nn.research_state import CHECKPOINTS, FRONT_DOOR_DOCUMENTS, checkpoint_states

REPO = Path(__file__).resolve().parents[1]
DOCUMENT = REPO / "docs" / "p14_preregistration.md"
PREFLIGHT = REPO / "artifacts" / "benchmark" / "btc_p14_source_preflight"

FROZEN_HASH = "sha256:830943664906c8cffbdae3b03b8f78e23339123c5d10831a85957ac958eb9b12"


def _flat(text: str) -> str:
    stripped = text.replace("`", "").replace("*", "").replace(">", " ")
    return re.sub(r"\s+", " ", stripped).lower()


@pytest.fixture(scope="module")
def document() -> str:
    return DOCUMENT.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads((PREFLIGHT / "preflight_manifest.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# A. P14 is not opened
# --------------------------------------------------------------------------- #


def test_the_outcome_is_not_opened():
    assert OUTCOME == "NOT OPENED"
    assert CURRENT_RESULT_STATE == "P14 NATIVE 1m TRADE-FLOW SCREEN: NOT YET RUN"
    assert CURRENT_RESULT_STATE == RESULT_STATES[0]
    assert len(RESULT_STATES) == 6


def test_no_evaluator_has_been_implemented():
    """A design, not a component. Nothing implements or provides one."""
    for name in ("p14.py", "p14_decision.py", "p14_screen.py", "tradeflow.py"):
        assert not (REPO / "nn" / name).exists(), name


def test_no_p14_result_artifact_exists():
    """The source preflight is allowed. A scored aggregate is not."""
    named = {p.relative_to(REPO).as_posix() for p in (REPO / "artifacts").rglob("*p14*")}
    assert named == {
        "artifacts/benchmark/btc_p14_source_preflight",
        "artifacts/btc_p14_source_preflight_SHA256SUMS.txt",
    }, sorted(named)
    contents = {p.name for p in PREFLIGHT.iterdir()}
    assert contents == {"STATUS.md", "preflight_manifest.json"}, sorted(contents)
    assert not (REPO / "artifacts" / "benchmark" / "btc_p14_decision").exists()


def test_the_research_state_says_preregistered_and_never_answered():
    states = checkpoint_states(REPO)
    assert states["P14"] == "preregistered"
    assert states["P8"] == "preregistered"
    assert states["P13"] == "preregistered"


def test_p14_is_registered_as_a_checkpoint_and_a_front_door_document():
    entry = next(c for c in CHECKPOINTS if c.name == "P14")
    assert entry.question == "btc_p14_native_tradeflow_screen"
    assert entry.preregistration == "docs/p14_preregistration.md"
    assert entry.evidence == "artifacts/benchmark/btc_p14_decision/decision.json"
    assert CHECKPOINTS[-1].name == "P14", "P14 is the newest checkpoint"
    assert "docs/p14_preregistration.md" in FRONT_DOOR_DOCUMENTS


def test_the_preflight_evidence_carries_no_economic_or_predictive_outcome(manifest):
    """An economic quantity would have to be a field. Prose saying there is none is not one."""

    def keys(node):
        if isinstance(node, dict):
            for k, v in node.items():
                yield k.lower()
                yield from keys(v)
        elif isinstance(node, list):
            for item in node:
                yield from keys(item)

    words = {w for key in keys(manifest) for w in key.split("_")}
    for forbidden in (
        "return",
        "sharpe",
        "pnl",
        "drawdown",
        "accuracy",
        "correlation",
        "agreement",
        "signal",
        "tfi",
        "gate",
        "profit",
        "alpha",
    ):
        assert forbidden not in words, forbidden
    assert manifest["result_state"] == CURRENT_RESULT_STATE
    assert "no gate evaluation" in manifest["what_this_is"]


# --------------------------------------------------------------------------- #
# B. The constants a result could be argued into or out of
# --------------------------------------------------------------------------- #


def test_the_hash_is_frozen():
    assert preregistration_hash() == FROZEN_HASH
    assert describe()["preregistration_hash"] == FROZEN_HASH


def test_the_document_quotes_the_same_hash(document):
    assert FROZEN_HASH in document


def test_the_hash_moves_when_any_decision_relevant_value_moves():
    """A payload whose hash did not move would make the freeze decorative."""
    import hashlib

    base = payload()
    for key, mutate in (
        ("horizon_bars", lambda v: v + 5),
        ("clock", lambda v: "5m"),
        ("signal_name", lambda v: "tfi_ratio_v2"),
        ("viability_gate", lambda v: v[:-1]),
        ("costs", lambda v: {**v, "cost_threshold": 0.001}),
    ):
        edited = dict(base)
        edited[key] = mutate(base[key])
        blob = json.dumps(edited, sort_keys=True, separators=(",", ":"))
        assert "sha256:" + hashlib.sha256(blob.encode()).hexdigest() != FROZEN_HASH, key


def test_one_signal_one_clock_one_horizon():
    assert SIGNAL_NAME == "tfi_ratio"
    assert CLOCK == "1m"
    assert HORIZON_BARS == 1


def test_the_cost_model_is_the_one_every_prior_checkpoint_used():
    assert COSTS["fee_rate"] == 0.0005
    assert COSTS["slippage_rate"] == 0.0005
    assert COSTS["cost_threshold"] == 0.002


def test_the_viability_gate_is_p6s_three_conditions():
    assert len(VIABILITY_GATE) == 3
    joined = _flat(" ".join(VIABILITY_GATE))
    assert "at least 3 of the 4" in joined
    assert "mean" in joined
    assert "momentum" in joined


def test_the_theta_grid_and_threshold_discipline_are_frozen():
    assert THRESHOLD_SELECTION["min_trades"] == 10
    assert "0.05" in THRESHOLD_SELECTION["grid"] and "0.95" in THRESHOLD_SELECTION["grid"]
    assert _flat(THRESHOLD_SELECTION["selected_on"]) == "the inner block only, never the outer"


def test_the_minimum_activity_floor_is_fixed_in_advance():
    assert MINIMUM_ACTIVITY["outer_trades_per_fold"] == 30
    assert "not a pass" in _flat(MINIMUM_ACTIVITY["rule"])
    assert "p7" in _flat(MINIMUM_ACTIVITY["why_30"])


def test_the_folds_are_the_frozen_p6_instants():
    assert OUTER_BLOCKS[0][0] == "2023-03-04T07:00:00+00:00"
    assert OUTER_BLOCKS[-1][1] == "2025-05-19T08:00:00+00:00"
    assert INNER_BLOCKS[0][0] == "2022-08-15T10:00:00+00:00"
    assert len(OUTER_BLOCKS) == len(INNER_BLOCKS) == 4
    for i in range(3):
        assert OUTER_BLOCKS[i][1] == OUTER_BLOCKS[i + 1][0], "outer blocks are contiguous"
        # each fold selects its threshold on the block the previous fold reported on
        assert INNER_BLOCKS[i + 1] == OUTER_BLOCKS[i], i


def test_the_last_outer_block_stops_at_the_research_boundary():
    from nn.p14_preregistration import RESEARCH_BOUNDARY

    assert RESEARCH_BOUNDARY == "2025-05-19T08:00:00+00:00"
    assert OUTER_BLOCKS[-1][1] == RESEARCH_BOUNDARY


# --------------------------------------------------------------------------- #
# C. The stages cannot be reordered or short-circuited
# --------------------------------------------------------------------------- #


def test_the_economic_screen_is_gated_on_the_predictive_gate():
    flat = _flat(STAGE2_GATED_ON_STAGE1)
    assert "if and only if" in flat
    assert "stage 0 passed and stage 1 passed" in flat
    assert "never run" in _flat(STAGE1["on_failure"])


def test_the_mechanism_control_can_never_make_p14_positive():
    flat = _flat(STAGE0["it_is_not_evidence_about_alpha"])
    assert "cannot make p14 positive" in flat
    assert "not evaluable" in _flat(STAGE0["on_failure"])
    assert "all four" in _flat(STAGE0["pass_condition"])


def test_stage_one_fits_nothing_and_uses_a_hindsight_floor():
    assert "no model family" in _flat(STAGE1["model"])
    assert "constant-direction" in _flat(STAGE1["baseline"])
    assert len(STAGE1["conditions"]) == 2


def test_both_stages_measure_statistic_and_baseline_over_the_same_rows():
    """A difference in exclusions must never be readable as a difference in skill."""
    assert "both measured over d" in _flat(STAGE1["decision_set"])
    assert "both measured over d0" in _flat(STAGE0["decision_set"])
    for key in ("statistic", "baseline"):
        assert "|d|" in STAGE1[key].lower()
        assert "d0" in STAGE0["baseline"].lower()


def test_a_stage_one_pass_is_not_reportable_as_a_result():
    flat = _flat(STAGE1["what_passing_does_not_mean"])
    assert "permissive filter" in flat
    assert "not evidence of alpha" in flat
    assert "licenses exactly one thing" in flat


def test_the_p6_1m_cells_are_not_available_as_a_control():
    flat = _flat(STAGE1["why_not_the_p6_1m_cells"])
    assert "not a control" in flat
    assert "rescue" in flat


# --------------------------------------------------------------------------- #
# D. The anchor is recorded honestly
# --------------------------------------------------------------------------- #


def test_the_anchor_is_a_single_named_source():
    assert "Silantyev" in EXTERNAL_ANCHOR["citation"]
    assert "10.1007/s42521-019-00007-w" in EXTERNAL_ANCHOR["citation"]
    assert "BitMEX" in EXTERNAL_ANCHOR["market_studied"]


def test_the_anchor_records_that_its_full_text_was_not_read():
    flat = _flat(EXTERNAL_ANCHOR["full_text_was_not_read"])
    assert "paywalled" in flat
    assert "no numeric value from the article is a target" in flat


def test_the_anchor_claim_is_marked_contemporaneous_not_predictive():
    adaptations = _flat(" ".join(EXTERNAL_ANCHOR["unavoidable_adaptations"]))
    assert "contemporaneous" in adaptations
    assert "causal" in adaptations
    assert "order flow imbalance is not reproduced at all" in adaptations


# --------------------------------------------------------------------------- #
# E. Source, and the P13 lesson
# --------------------------------------------------------------------------- #


def test_the_source_is_one_binance_spot_archive_family():
    assert SOURCE["layout"] == "data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-YYYY-MM.zip"
    assert SOURCE["canonical_base_url"] == "https://data.binance.vision"
    flat = _flat(SOURCE["no_other_source"])
    for absent in ("aggtrades", "order book", "futures", "funding", "open interest"):
        assert absent in flat, absent


def test_the_preflight_passed_and_proves_row_level_sufficiency(manifest):
    assert manifest["verdict"] == "PASS"
    assert manifest["objects"]["zip_objects"] == 65
    assert manifest["objects"]["checksums_verified"] == 65
    assert manifest["objects"]["checksum_mismatches"] == 0
    assert manifest["integrity"]["taker_gt_volume"] == 0
    assert manifest["integrity"]["malformed_rows"] == 0
    assert manifest["integrity"]["duplicate_open_times"] == 0
    assert manifest["sufficiency"]["rows_match_exactly"] is True
    assert manifest["sufficiency"]["committed_price_grid_rows"] == 2827755
    assert manifest["coverage"]["rows_before_boundary"] == 2827755
    assert manifest["raw_archives_committed"] is False


def test_the_preflight_reproduces_the_multiclock_gap_structure(manifest):
    """The same 15 exchange outages multiclock_v1 section 5 enumerates."""
    assert manifest["coverage"]["missing_interval_count"] == 15
    assert manifest["coverage"]["longest_missing_run_minutes"] == 354  # 5h54m
    assert manifest["coverage"]["longest_missing_run"]["start"].startswith("2020-02-19")
    assert manifest["coverage"]["last_missing_interval_start"].startswith("2023-03-24")


def test_the_sign_convention_was_proved_against_the_trade_tape(manifest):
    check = manifest["aggtrades_cross_check"]
    assert check["minutes_compared"] == 4320
    assert check["minutes_agreeing_taker_buy"] == 4320
    assert check["minutes_agreeing_volume"] == 4320
    assert check["minutes_present_in_one_source_only"] == 0
    assert check["max_relative_difference_taker_buy"] < 1e-12
    assert "identity, not approximation" in check["conclusion"]


def test_the_boundary_is_respected_and_truncation_is_recorded(manifest):
    assert manifest["span"]["intended_last_timestamp_exclusive"] == "2025-05-19T08:00:00+00:00"
    assert manifest["coverage"]["rows_at_or_after_boundary_truncated"] == 18240
    total = (
        manifest["coverage"]["rows_before_boundary"]
        + manifest["coverage"]["rows_at_or_after_boundary_truncated"]
    )
    assert total == manifest["coverage"]["rows_parsed"]


# --------------------------------------------------------------------------- #
# F. Safety and the forbidden list
# --------------------------------------------------------------------------- #


def test_the_sign_flip_is_forbidden():
    flat = _flat(" ".join(FORBIDDEN_AFTER_RESULTS))
    assert "flipping the sign convention" in flat


def test_the_forbidden_list_closes_the_usual_doors():
    flat = _flat(" ".join(FORBIDDEN_AFTER_RESULTS))
    for door in (
        "changing the clock, the horizon",
        "running the economic screen after a failed predictive gate",
        "another venue, another symbol",
        "opening p8, reading p4-hold, or approaching styx",
        "tradeflow_v2",
    ):
        assert door in flat, door


def test_the_safety_prohibitions_are_intact():
    flat = _flat(" ".join(SAFETY_PROHIBITIONS))
    assert "no real money" in flat
    assert "no leverage above 1x" in flat
    assert "aegis remains the sole risk authority" in flat
    assert "p4-hold stays retired and unread" in flat
    assert "styx stays sealed" in flat


def test_the_document_discloses_the_unexecutable_short_leg(document):
    flat = _flat(document)
    assert "not executable on binance spot" in flat
    assert "futures execution v1" in flat


def test_the_document_and_the_payload_agree_on_the_checkpoint(document):
    assert CHECKPOINT == "P14"
    assert (
        _flat(document).startswith("# p14 — preregistration") or "p14" in _flat(document)[:60]
    )
    assert "btc_p14_source_preflight" in document


def test_condition_three_is_disclosed_as_a_weak_floor():
    """P6 already learned this. Recording it now stops it being discovered after a result."""
    from nn.p14_preregistration import (
        CONDITION_3_IS_A_WEAK_DISCRIMINATOR,
        MOMENTUM_BASELINE_IS_NOT_AN_INPUT,
    )

    flat = _flat(CONDITION_3_IS_A_WEAK_DISCRIMINATOR)
    assert "very low one" in flat
    assert "condition 1 is what binds" in flat
    assert "only column any p14 rule reads" in _flat(MOMENTUM_BASELINE_IS_NOT_AN_INPUT)
