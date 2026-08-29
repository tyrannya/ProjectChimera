"""P5's design, as values, fixed before any P5 model was fitted.

The machine-readable twin of ``docs/p5_preregistration.md``. Every constant in
that document is a value here, ``tests/test_p5_preregistration.py`` asserts the
two agree, and every P5 cell records :func:`preregistration_hash` so that a cell
produced under an edited design is a *different object* rather than the same one
with a different story.

This module is deliberately separate from :mod:`nn.p4_preregistration` and shares
no constant with it. P4's hash is load-bearing in four committed files —
``data/research/p4_stage1_authorisation.json``,
``data/research/p4_holdout_ledger.json``,
``artifacts/benchmark/btc_p4_stage1/stage1.json`` and its checksum manifest — so
editing a P4 constant to serve P5 would move that hash and break the release path
of a checkpoint that is closed. P4 is finished; nothing here touches it.

What P5 changes, and it is exactly one axis: **timeframe context**. Everything
else is held at the value four earlier checkpoints ran under, and :data:`HELD_FIXED`
lists each one by name so "unchanged" is a checkable claim rather than a promise.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

CHECKPOINT = "P5"

QUESTION = (
    "does strictly causal higher-timeframe OHLCV context provide robust incremental "
    "information beyond the current 1h OHLCV14 baseline, in the unchanged BTC/USDT "
    "1h-observation / 6-candle-horizon cost-aware setup?"
)

HYPOTHESIS = (
    "the 1h bar is not the only clock a directional decision could read, and every "
    "family tested so far — smc_v1, chart_structure_v1, microstructure_v1, "
    "derivatives_v1 — was computed on one. If trend and volatility context on a 4h "
    "and a daily clock carry information the 1h OHLCV14 vector does not, adding them "
    "improves the cost-aware net return of the deciding model in most temporal folds. "
    "If they do not, they do not."
)

#: Why a positive P5 would still not be a confirmation.
EVIDENCE_CEILING = (
    "P5 is exploratory adaptive evidence and cannot confirm anything. Its four outer "
    "blocks have already been read by v4, P2a, P2b, the P2b ablation, the P2b regime "
    "description, P2c, P3 and P4 — eight prior readings — so a positive result is a "
    "lead that would need evidence these blocks structurally cannot supply. The sealed "
    "Styx region is not opened to supply it, and P4-HOLD was retired unread and is not "
    "available. A negative result needs no discounting, which is the asymmetry that "
    "makes negative results the cheap ones to trust."
)

RESEARCH_CLASSIFICATION = (
    "P5 was designed after P4's outer results had been seen, and by the time it runs "
    "these four outer blocks will have been read by v4, P2a, P2b, the P2b ablation, the "
    "P2b regime description, P2c, P3 and P4. Its feature definition, arms, models, "
    "folds and decision rule were fixed before its own outer numbers existed, and the "
    "axis was chosen because four handcrafted families on one clock had failed — so "
    "this is exploratory adaptive evidence: it generates hypotheses and cannot confirm "
    "one"
)

# --- the family under test -------------------------------------------------

FAMILY = "mtf_v1"
CONTROL = "ohlcv14"
COMBINED = "ohlcv14_plus_mtf_v1"
ARMS: tuple[str, ...] = (CONTROL, FAMILY, COMBINED)

#: The higher timeframes, and the fixed UTC grid each is cut on. Not a rolling
#: window: a 4h bar starts at 00, 04, 08, 12, 16 or 20 UTC and a daily bar at 00
#: UTC, which is the grid the venue itself publishes and the only one a live
#: system could reproduce without knowing when it happened to start.
TIMEFRAMES: dict[str, dict[str, Any]] = {
    "4h": {
        "hours": 4,
        "grid": "UTC, bars beginning at 00/04/08/12/16/20",
        "prefix": "mtf_4h_",
    },
    "1d": {"hours": 24, "grid": "UTC, bars beginning at 00", "prefix": "mtf_1d_"},
}

#: A higher-timeframe bar is `open=first, high=max, low=min, close=last,
#: volume=sum` over its constituent 1h candles, and is used ONLY if every one of
#: them is present. Nothing is forward-filled, nothing is approximated from a
#: partial bar, and a partial bar is not a bar.
BAR_CONSTRUCTION: dict[str, Any] = {
    "open": "first 1h open in the bar",
    "high": "max 1h high in the bar",
    "low": "min 1h low in the bar",
    "close": "last 1h close in the bar",
    "volume": "sum of 1h volume in the bar",
    "completeness": "every constituent 1h candle must be present, or the bar is dropped",
    "incomplete_bar_policy": "dropped entirely; never partially formed, never filled",
}

#: The feature engine, unchanged. `chimera.features.compute_features` is the
#: OHLCV14 engine the control is built from; running it on a 4h or 1d bar series
#: is what "the same semantics on a different clock" means. The window lengths
#: are counts of BARS, not hours, and are not rescaled — rescaling them would be
#: a second axis (window length) changing at the same time as the first.
FEATURE_ENGINE: dict[str, Any] = {
    "function": "chimera.features.compute_features",
    "spec": "chimera.features.FeatureSpec()  # the defaults, unchanged",
    "windows_are_measured_in": "bars of the higher timeframe, not hours",
    "columns_per_timeframe": 14,
    "total_columns": 28,
    "why_not_rescaled": (
        "a 26-bar EMA on 4h bars is a different lookback in hours than on 1h bars, and "
        "that is the point: the arm is asking whether a slower clock carries "
        "information. Rescaling the windows to match the 1h lookback in hours would "
        "make the higher-timeframe arm a smoothed copy of the control"
    ),
}

#: Bars of each higher-timeframe series that are not usable, matching how the 1h
#: research spine itself was built: `nn.data_pipeline.build_dataset` drops
#: `FeatureSpec.warmup` rows at the head of each segment because Wilder- and
#: EMA-smoothed quantities converge rather than becoming exact at a fixed row.
WARMUP_BARS = 78

#: How a 1h decision row sees a higher-timeframe bar. The bar must have CLOSED.
ALIGNMENT: dict[str, Any] = {
    "rule": (
        "at the 1h row timestamped t, the context is the last complete higher-timeframe "
        "bar whose close time is <= t"
    ),
    "close_time": "a bar beginning at s with width w closes at s + w",
    "strictly_causal": "a bar that has not closed at t is never visible at t",
    "staleness_bound_bars": 1,
    "staleness_rule": (
        "the as-of bar must be the immediately preceding complete bar: t - close_time < w. "
        "A row whose nearest complete bar is older than that — because a bar in between "
        "was dropped as incomplete — is INELIGIBLE rather than served stale context"
    ),
}

#: The complete-bar series is treated as contiguous when features are computed.
#: This is a real decision with a real cost and it is recorded rather than
#: buried, together with the measurement that settled it.
CONTIGUITY_POLICY: dict[str, Any] = {
    "policy": (
        "the complete-bar series is contiguous; feature state is not reset at a dropped bar"
    ),
    "consequence": (
        "a handful of bar-to-bar differences span two bar widths rather than one, "
        "wherever an incomplete bar was dropped"
    ),
    "measured_dropped_bars": {"4h": [20, 11792], "1d": [16, 1966]},
    "alternative_considered": "reset feature state at every dropped bar, as smc_v1 and "
    "chart_structure_v1 reset at a market-data gap",
    "why_rejected": (
        "measured before any fit: a per-segment reset leaves 30,563 of 45,802 spine rows "
        "eligible (0.667), with outer block 0 only 0.621 eligible and fold 1's inner "
        "block the same. That does not merely cost sample — it fails the availability "
        "rule below and CHANGES WHICH FOLDS EXIST, which would make the '3 of 4' bar a "
        "statement about a different experiment. The contiguous policy leaves 44,171 of "
        "45,802 rows eligible (0.964) with every inner and outer block at 1.000"
    ),
    "why_it_is_not_a_leak": (
        "dropping a bar removes an unobserved bar from the series. Every bar that "
        "remains is fully observed and strictly in the past of the row that reads it. "
        "The reset those two families need is about STRUCTURE — a swing high inferred "
        "across a hole is a claim about prices nobody saw — and a moving average over "
        "observed closes makes no such claim"
    ),
}

# --- sample universe and availability --------------------------------------

#: Every condition a spine row must satisfy to be scored at all. Computed once
#: and applied to all three arms from the same array object, so "the arms were
#: compared on the same rows" is a property of construction rather than a claim.
#: This is the mechanism `docs/p4_preregistration.md` §6.2 established, reused.
ELIGIBILITY_CONDITIONS: tuple[str, ...] = (
    "an as-of complete 4h bar exists at or before the row",
    "an as-of complete 1d bar exists at or before the row",
    "each as-of bar is at index >= WARMUP_BARS in its own complete-bar series",
    "each as-of bar satisfies the one-bar staleness bound",
    "all 28 mtf_v1 columns are finite at the row",
)

#: The shape and the numbers are P4's `BLOCK_AVAILABILITY_RULE`, reused verbatim.
#: What is new is that it is applied to each fold's INNER block as well as its
#: outer one: the decision threshold is selected on the inner block, so a
#: punctured inner block would select a threshold on a different period than the
#: fold reports on. P4 had no inner-block condition; this is a strengthening,
#: declared here before any fit rather than after seeing which folds it costs.
BLOCK_AVAILABILITY_RULE: dict[str, Any] = {
    "min_eligible_row_fraction": 0.98,
    "max_contiguous_ineligible_hours": 48,
    "applies_to": "each fold's inner-validation block and each fold's outer-validation block",
    "measured_on": (
        "rows surviving ELIGIBILITY_CONDITIONS, computed before any model is fitted and "
        "reported per block whatever the gate decides"
    ),
    "training_blocks": (
        "reported, never gating. Training loss is identical across arms by construction, "
        "and requiring 98% of a training block that begins at the dataset's own start "
        "would disqualify every fold over a warm-up cost that is a property of the start "
        "date rather than of the fold"
    ),
}

#: All four folds, or nothing. P4 could tolerate three because its availability
#: was genuinely unknown when it was written — the archive's inception date had
#: not been established. P5's availability is measurable before any fit from
#: timestamps alone, and it was measured (see MEASURED_AVAILABILITY), so there is
#: no reason to leave the denominator free. A denominator that can move after
#: results are seen is the failure this forecloses.
AVAILABILITY_GATE: dict[str, Any] = {
    "folds_required": 4,
    "of": 4,
    "on_failure": "not_evaluable",
    "on_failure_means": (
        "P5 reports invalid/unanswered. It does NOT re-derive a bar over the surviving "
        "folds, and it does not drop a fold"
    ),
}

#: Measured before any P5 model was fitted, from timestamps and completeness
#: alone. This carries availability information and no outcome information: it is
#: a fact about which rows have a higher-timeframe context, computed without
#: fitting anything, exactly as P4's `--probe` established its own window.
MEASURED_AVAILABILITY: dict[str, Any] = {
    "measured_before_any_fit": True,
    "spine_rows": 45802,
    "eligible_rows": 44171,
    "eligible_fraction": 0.964368,
    "ineligible_rows": 1631,
    "ineligible_span": [0, 1630],
    "ineligible_is_contiguous_at_the_head": True,
    "binding_constraint": "the 78-bar warm-up of the daily series",
    "first_eligible_timestamp": "2020-03-23T00:00:00+00:00",
    "per_fold_inner_eligible_fraction": [1.0, 1.0, 1.0, 1.0],
    "per_fold_outer_eligible_fraction": [1.0, 1.0, 1.0, 1.0],
    "per_fold_train_eligible_fraction": [0.9248, 0.9385, 0.9480, 0.9549],
    "folds_available": 4,
    "note": (
        "every ineligible row lies in the training block of every fold; no inner or outer "
        "block loses a single row. All four folds are available and the deciding bar is "
        "3 of 4"
    ),
}

# --- models and the decision ----------------------------------------------

PRIMARY_MODEL = "xgboost"
SECONDARY_MODELS: tuple[str, ...] = ("logistic_regression", "lightgbm")
MODELS: tuple[str, ...] = (PRIMARY_MODEL, *SECONDARY_MODELS)

SECONDARY_MODELS_ROLE = (
    "context only. They are reported in full whatever they show, and they cannot switch "
    "the deciding cell after the fact: a checkpoint whose winner is chosen from three "
    "models after seeing all three has three chances to pass a bar written for one"
)

#: The one comparison that decides P5.
PRIMARY_COMPARISON: tuple[str, str] = (COMBINED, CONTROL)

#: Strict. A delta of exactly zero is not an improvement: the claim under test is
#: that the new information adds something, and adding nothing is the null.
IMPROVED_RULE: dict[str, Any] = {
    "improved_when": "delta > 0",
    "zero_is_improved": False,
    "metric": "outer-validation cost-aware net return",
    "delta": "ohlcv14_plus_mtf_v1 minus ohlcv14, per fold, same fold index",
}

#: The bar, and nothing else. This is P2b's, P2c's and P3's rule unchanged —
#: count the folds — because P5 asks P2b's question about a different family, and
#: an information-set checkpoint that scored itself differently from the four it
#: is compared against would not be comparable to them.
DECISION_RULE: dict[str, Any] = {
    "statistic": "number of outer folds in which the deciding delta is > 0",
    "folds": 4,
    "improved_folds_required": 3,
    "decided_by": {"model": PRIMARY_MODEL, "comparison": list(PRIMARY_COMPARISON)},
    "cost_multiplier": 1.0,
    "mean_delta": "descriptive; may not rescue a fold-count failure and may not veto a pass",
    "worst_fold_delta": "descriptive; same",
    "false_positive_rate_under_coin_null": "5/16 = 0.3125",
    "why_that_rate_is_stated": (
        "P4's §8.1 established it and it does not stop being true here. At n = 4 "
        "dependent folds no threshold has both a usable false-positive rate and usable "
        "power. P5 does not pretend otherwise; it reports an exploratory answer with its "
        "error rate written down, which is why a positive P5 is a lead and not a finding"
    ),
}

#: Reported, never decisive. P4 invalidated a fold in which either arm realised
#: fewer than 10 non-overlapping outer trades; P2b, P2c and P3 did not. P5
#: follows the information-set precedent — no exclusion — and reports the count so
#: the difference between the two conventions is visible instead of silent.
TRADE_COUNT_DIAGNOSTIC: dict[str, Any] = {
    "flag_below_outer_trades": 10,
    "effect_on_the_denominator": "none",
    "effect_on_the_decision": "none",
    "why": (
        "excluding a fold changes the denominator, and a denominator that can move is a "
        "denominator someone can move after seeing the numbers. The count is reported and "
        "limits the interpretation; it does not change the arithmetic"
    ),
}

# --- what may not move -----------------------------------------------------

#: One axis changes. Each entry below is a value four earlier checkpoints ran
#: under and P5 does not touch, named so that "unchanged" is checkable.
HELD_FIXED: dict[str, Any] = {
    "exchange": "binance",
    "pair": "BTC/USDT",
    "base_timeframe": "1h",
    "research_contract": "btc-usdt-1h-gen1",
    "source": "the committed OHLCV research snapshot; no new source is acquired",
    "label_horizon_candles": 6,
    "target": "chimera.contracts.TargetSpec(horizon=6, fee_rate=0.0005, slippage_rate=0.0005)",
    "cost_threshold": 0.002,
    "round_trip_cost": 0.002,
    "folds": 4,
    "fold_plan": "nn.p2b.plan_from_manifest over the committed manifest, unchanged",
    "seq_len": 64,
    "seed": 42,
    "min_trades": 10,
    "threshold_selection": "on the inner-validation block only, applied unchanged to outer",
    "model_families": "the P2a configurations, untuned",
    "control": CONTROL,
}

#: After the first valid P5 outer number exists, none of these may change. An
#: amendment is legal only before the first fit and only as its own commit.
FORBIDDEN_AFTER_RESULTS: tuple[str, ...] = (
    "the question or the hypothesis",
    "the mtf_v1 feature definition, including the timeframes, the bar grid, the "
    "completeness rule, the feature engine, the warm-up and the alignment rule",
    "the arms or their names",
    "the model families or their hyperparameters",
    "the deciding model",
    "the deciding comparison",
    "the target, the horizon or the cost model",
    "the fold plan or the number of folds",
    "the eligibility conditions or the availability rule",
    "the decision rule, the improved rule or the required fold count",
    "the addition of any further success criterion",
    "re-running a valid cell because its number is disappointing",
)

STOPPING_RULE: dict[str, Any] = {
    "stages": 1,
    "holdout": None,
    "on_pass": (
        "P5 is supportive exploratory ADAPTIVE evidence. It is not confirmation, it does "
        "not open Styx, and it does not license a live allocation. The next checkpoint "
        "would have to be designed to test it on evidence these blocks cannot supply"
    ),
    "on_fail": (
        "P5 is negative. Do not tune mtf_v1's constants — they were predeclared, and "
        "searching them against these four outer blocks would convert a negative result "
        "into a fitted one. Do not create mtf_v2. The next research move changes axis"
    ),
    "on_not_evaluable": (
        "P5 is reported invalid/unanswered. An experiment that could not run is not a "
        "negative result and is not written up as one"
    ),
}

#: The instant itself is deliberately NOT restated here. It lives in exactly one
#: place — `sealed_test_start` in the committed contract at
#: `nn/research_contracts/btc-usdt-1h-gen1.json` — and
#: `tests/test_research_contracts.py::test_the_source_carries_no_second_copy_of_the_anchor`
#: fails any module that writes it down a second time. A preregistration that
#: hard-coded the boundary would be a second copy that could drift from the one
#: the loader actually reads.
STYX_PROHIBITION = (
    "the sealed region beginning at the committed contract's sealed_test_start is not "
    "opened, not read, not planned over and not available to rescue an ambiguous P5. P5 "
    "adds no new source, so it adds no new way to reach it"
)

P4_HOLD_UNAVAILABILITY = (
    "P4-HOLD, rows [45802, 48211), was retired unread by P4 with checkpoint: null. The "
    "ledger in nn.p4_holdout is one-way — there is no path back to unspent — and P5 does "
    "not ask for it, does not read it, and could not spend it if it did"
)

# --- artifacts -------------------------------------------------------------

ARTIFACT_POLICY: dict[str, Any] = {
    "primary": {
        "directories": "artifacts/benchmark/btc_p5_{arm}_{model}/ for each of the 9 cells",
        "files": ["p2b.json", "p2b.md", "STATUS.md", "outer_predictions.parquet"],
        "manifest": "artifacts/btc_p5_SHA256SUMS.txt",
        "immutable": True,
    },
    "decision": {
        "directory": "artifacts/benchmark/btc_p5_decision/",
        "files": ["decision.json", "STATUS.md"],
        "manifest": "artifacts/btc_p5_decision_SHA256SUMS.txt",
        "immutable": True,
        "why_frozen": (
            "the decision record is what the checkpoint answered. P4 froze its Stage-1 "
            "screen for the same reason, and `nn.p4_holdout.assert_frozen_stage_one` "
            "compares the live report against the frozen one by equality"
        ),
    },
    "derived": {
        "directory": "artifacts/benchmark/btc_p5_comparison/",
        "immutable": False,
        "why": (
            "the comparison is regenerated whenever the aggregator improves, so a hash "
            "over it is a promise this workflow breaks on purpose. It is pinned by "
            "regenerating it and checking what it says"
        ),
    },
}

#: What the leakage battery must show BEFORE any P5 cell is fitted. Each item has
#: a positive control: a test that deliberately introduces the leak and asserts
#: the check catches it. A check that has never failed is not evidence.
LEAKAGE_BATTERY: tuple[dict[str, str], ...] = (
    {
        "id": "L1",
        "must_show": "no not-yet-closed 4h bar is used: every as-of 4h bar closes at or "
        "before its row's timestamp",
        "positive_control": "advance the as-of index by one bar; the check must fail",
    },
    {
        "id": "L2",
        "must_show": "the same for 1d",
        "positive_control": "advance the as-of index by one bar; the check must fail",
    },
    {
        "id": "L3",
        "must_show": "exact boundary timestamps are causal: at a row landing exactly on a "
        "bar's close time that bar IS visible and the next is not",
        "positive_control": "shift the close-time convention by one bar width; the "
        "boundary rows must change",
    },
    {
        "id": "L4",
        "must_show": "no higher-timeframe bar is built from an hour at or after the Styx "
        "boundary, or from an hour inside P4-HOLD",
        "positive_control": "extend the source past the truncation point; the check must fail",
    },
    {
        "id": "L5",
        "must_show": "labels are unchanged: targets, future returns, closes, dates and "
        "segment ids are the spine's own arrays, shared by object identity",
        "positive_control": "nn.information_sets.AlignedResearchSamples.prove_alignment "
        "already raises on a view that does not share them",
    },
    {
        "id": "L6",
        "must_show": "the control arm's columns are byte-identical to the spine's own "
        "OHLCV14 columns",
        "positive_control": "perturb one control column; the check must fail",
    },
    {
        "id": "L7",
        "must_show": "the join preserves sample identity: every arm scores the same rows "
        "in every block of every fold",
        "positive_control": "prove_alignment already raises on differing sample indices",
    },
    {
        "id": "L8",
        "must_show": "no value is forward-filled from a future close: the as-of index is "
        "monotone non-decreasing and every as-of bar closed in the past",
        "positive_control": "roll the joined column by -1 across a bar boundary; the "
        "check must fail",
    },
    {
        "id": "L9",
        "must_show": "a deliberately future-shifted mtf_v1 is detected",
        "positive_control": "build the family from the NEXT bar rather than the last "
        "closed one; every causal check must fail",
    },
    {
        "id": "L10",
        "must_show": "causality is structural, not inherited: mtf_v1 computed on the full "
        "raw history and on the history truncated at the spine's last row agree on every "
        "spine row",
        "positive_control": "a centred or backfilled window would disagree; the test "
        "constructs one and asserts the check catches it",
    },
)


def payload() -> dict[str, Any]:
    """Everything the hash covers. Adding a constant here is a design change."""
    return {
        "checkpoint": CHECKPOINT,
        "question": QUESTION,
        "hypothesis": HYPOTHESIS,
        "evidence_ceiling": EVIDENCE_CEILING,
        "research_classification": RESEARCH_CLASSIFICATION,
        "family": FAMILY,
        "control": CONTROL,
        "combined": COMBINED,
        "arms": list(ARMS),
        "timeframes": TIMEFRAMES,
        "bar_construction": BAR_CONSTRUCTION,
        "feature_engine": FEATURE_ENGINE,
        "warmup_bars": WARMUP_BARS,
        "alignment": ALIGNMENT,
        "contiguity_policy": CONTIGUITY_POLICY,
        "eligibility_conditions": list(ELIGIBILITY_CONDITIONS),
        "block_availability_rule": BLOCK_AVAILABILITY_RULE,
        "availability_gate": AVAILABILITY_GATE,
        "measured_availability": MEASURED_AVAILABILITY,
        "primary_model": PRIMARY_MODEL,
        "secondary_models": list(SECONDARY_MODELS),
        "secondary_models_role": SECONDARY_MODELS_ROLE,
        "models": list(MODELS),
        "primary_comparison": list(PRIMARY_COMPARISON),
        "improved_rule": IMPROVED_RULE,
        "decision_rule": DECISION_RULE,
        "trade_count_diagnostic": TRADE_COUNT_DIAGNOSTIC,
        "held_fixed": HELD_FIXED,
        "forbidden_after_results": list(FORBIDDEN_AFTER_RESULTS),
        "stopping_rule": STOPPING_RULE,
        "styx_prohibition": STYX_PROHIBITION,
        "p4_hold_unavailability": P4_HOLD_UNAVAILABILITY,
        "artifact_policy": ARTIFACT_POLICY,
        "leakage_battery": [dict(item) for item in LEAKAGE_BATTERY],
    }


def preregistration_hash() -> str:
    """SHA-256 over :func:`payload`. Every P5 cell records it."""
    blob = json.dumps(payload(), sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(blob.encode()).hexdigest()


def mtf_columns() -> list[str]:
    """The 28 column names, in the order the family emits them.

    Imported by :mod:`nn.mtf` rather than spelled again there: an arm whose
    columns were named somewhere other than the preregistration would be
    answering a question nobody registered.
    """
    from chimera.features import feature_columns

    return [
        f"{TIMEFRAMES[tf]['prefix']}{name}"
        for tf in ("4h", "1d")
        for name in feature_columns()
    ]


def describe() -> dict[str, Any]:
    """The preregistration plus its hash, for a report or an artifact."""
    return {"preregistration_hash": preregistration_hash(), **payload()}


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(describe(), indent=2))
