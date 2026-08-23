"""P4's design, as data, fixed before any P4 observation exists.

``docs/p4_preregistration.md`` is the document a reader should start from. This
module is the same commitments in a form a later run cannot quietly disagree
with: the arms, the columns, the windows, the clips, the sample-universe rules,
the trade-validity bar, the two-stage decision rule and the stopping rule are
values here, and ``tests/test_p4_preregistration.py`` pins them against the
document.

**Nothing here runs anything.** There is no engine, no acquisition, no feature
implementation, and ``P4`` is deliberately *not* registered in
:data:`nn.information_sets.CHECKPOINTS` — so ``python -m nn.p2b --checkpoint P4``
is refused rather than silently producing a cell from columns that do not exist
yet. Preregistration is a commitment about a future run, not the start of one.

**Why a module and not only prose.** Every constant below is a researcher degree
of freedom, and the way they get spent is one at a time, each with a reason, after
a number has been seen. A document can be edited with the same commit that reports
the result. A value that a test asserts, and that the eventual runner must read
rather than restate, cannot be moved without the move being the diff.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

#: The research question, in one sentence, as it will be recorded in every cell.
QUESTION = (
    "does causal derivatives positioning and carry information — perpetual funding, "
    "open interest and futures basis — add usable information beyond OHLCV14?"
)

#: The checkpoint identity `artifacts/README.md` scopes CURRENT to.
RESEARCH_QUESTION_ID = "btc_p4_derivatives_positioning_benchmark"

#: The new information set's name, and the arms of the checkpoint.
DERIVATIVES_V1 = "derivatives_v1"
CONTROL = "ohlcv14"
COMBINED = "ohlcv14_plus_derivatives_v1"

#: Three arms, and only three. `derivatives_v1` alone answers "does this family
#: carry standalone signal"; the combined arm answers the question that was
#: asked. A leave-one-family-out ablation is *not* preregistered as evidence: it
#: is six more comparisons over the same rows, and P2b's ablation is the model
#: for how it must be labelled if it is ever run — post-hoc, descriptive,
#: nothing fitted, no bearing on the verdict.
ARMS: tuple[str, ...] = (CONTROL, DERIVATIVES_V1, COMBINED)

#: The one comparison the decision rule reads. Everything else is secondary and
#: is reported without being able to change the answer.
PRIMARY_COMPARISON = (COMBINED, CONTROL)
PRIMARY_MODEL = "xgboost"
SECONDARY_MODELS: tuple[str, ...] = ("logistic_regression", "lightgbm")

#: The target is not changed. See the document, section "Target and horizon":
#: the control's entire evidence base exists at 1h/6h, and moving the horizon
#: after four negative checkpoints would change the control and the question in
#: the same step.
TARGET = {
    "timeframe": "1h",
    "horizon": 6,
    "fee_rate": 0.0005,
    "slippage_rate": 0.0005,
    "cost_threshold": 0.002,
    "unchanged_from": "v4 / P2a / P2b / P2c / P3",
}

#: Reported for every arm, fixed now, and never the basis of the headline.
COST_SENSITIVITY_MULTIPLIERS: tuple[float, ...] = (1.0, 1.5, 2.0)

#: The eight columns of `derivatives_v1`, with the window and clip each is
#: computed under. Nothing here may be searched against a P4 outer result.
#:
#: `visible_from` states the causality rule in the same row as the feature,
#: because "which observation was available at the decision instant" is where a
#: positioning feature leaks if it leaks at all.
FEATURES: tuple[dict[str, Any], ...] = (
    {
        "name": "drv_funding_last",
        "family": "funding",
        "definition": "the most recent realised 8h funding rate visible at row t",
        "window": None,
        "clip": [-0.01, 0.01],
        "visible_from": "settlement instant T is visible to row t iff T <= t (candle open)",
    },
    {
        "name": "drv_funding_sum_9",
        "family": "funding",
        "definition": "sum of the last 9 realised settlements visible at row t (three days)",
        "window": 9,
        "clip": [-0.09, 0.09],
        "visible_from": "settlement instant T is visible to row t iff T <= t (candle open)",
    },
    {
        "name": "drv_funding_z",
        "family": "funding",
        "definition": (
            "(drv_funding_last - mean of the last 30 visible settlements) / "
            "(std of the same 30 + 1e-12)"
        ),
        "window": 30,
        "clip": [-5.0, 5.0],
        "visible_from": "settlement instant T is visible to row t iff T <= t (candle open)",
    },
    {
        "name": "drv_oi_log_change_24h",
        "family": "open_interest",
        "definition": "log(OI_t / OI_{t-24h}) on contract-count open interest",
        "window": 24,
        "clip": [-1.0, 1.0],
        "visible_from": "the last OI snapshot at or before t, at most 1h stale",
    },
    {
        "name": "drv_oi_notional_ratio",
        "family": "open_interest",
        "definition": "OI notional at t over the mean OI notional of the last 168h",
        "window": 168,
        "clip": [0.0, 10.0],
        "visible_from": "the last OI snapshot at or before t, at most 1h stale",
    },
    {
        "name": "drv_oi_price_divergence",
        "family": "open_interest",
        "definition": (
            "log(OI_t / OI_{t-24h}) - log(close_t / close_{t-24h}): positioning "
            "building against price rather than with it"
        ),
        "window": 24,
        "clip": [-1.0, 1.0],
        "visible_from": "the last OI snapshot at or before t, at most 1h stale",
    },
    {
        "name": "drv_basis",
        "family": "basis",
        "definition": "perpetual close at t over spot close at t, minus one",
        "window": None,
        "clip": [-0.02, 0.02],
        "visible_from": "both closes are the candle at t, complete at t + 1h",
    },
    {
        "name": "drv_basis_z",
        "family": "basis",
        "definition": "(drv_basis - mean of the last 168h) / (std of the same 168h + 1e-12)",
        "window": 168,
        "clip": [-5.0, 5.0],
        "visible_from": "both closes are the candle at t, complete at t + 1h",
    },
)

FEATURE_NAMES: tuple[str, ...] = tuple(feature["name"] for feature in FEATURES)

#: Hours of history every `derivatives_v1` column needs before it is defined.
#: The binding window is 30 funding settlements = 240 hours; the 168h windows
#: are shorter. A row inside the warm-up is outside the sample universe for
#: *every* arm, control included.
WARMUP_HOURS = 240

#: Exact primary sources. `--plan` and `--probe` must establish availability
#: against these before anything is downloaded; neither may be substituted after
#: a P4 number exists.
DATA_SOURCES: tuple[dict[str, Any], ...] = (
    {
        "field": "funding_rate",
        "venue": "binance",
        "market_type": "usd-m perpetual futures",
        "symbol": "BTCUSDT",
        "primary": (
            "https://data.binance.vision/data/futures/um/monthly/fundingRate/BTCUSDT/"
            "BTCUSDT-fundingRate-{year}-{month}.zip"
        ),
        "fallback": "https://fapi.binance.com/fapi/v1/fundingRate (paged, limit 1000)",
        "timestamp_column": "fundingTime",
        "timestamp_semantics": "the settlement instant, 00:00/08:00/16:00 UTC",
        "publication": (
            "the realised rate is final at the settlement instant; the continuously "
            "published *predicted* rate is never read"
        ),
        "cadence_hours": 8,
    },
    {
        "field": "open_interest",
        "venue": "binance",
        "market_type": "usd-m perpetual futures",
        "symbol": "BTCUSDT",
        "primary": (
            "https://data.binance.vision/data/futures/um/monthly/metrics/BTCUSDT/"
            "BTCUSDT-metrics-{year}-{month}.zip "
            "(sum_open_interest, sum_open_interest_value)"
        ),
        "fallback": (
            "https://fapi.binance.com/futures/data/openInterestHist — 30 days of "
            "retention only, so it cannot build this history and is a spot check"
        ),
        "timestamp_column": "create_time",
        "timestamp_semantics": "the instant of a 5-minute snapshot of the open book",
        "publication": "the snapshot is a state at its own instant, not an interval",
        "cadence_hours": 1 / 12,
    },
    {
        "field": "perpetual_price",
        "venue": "binance",
        "market_type": "usd-m perpetual futures",
        "symbol": "BTCUSDT",
        "primary": (
            "https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/1h/"
            "BTCUSDT-1h-{year}-{month}.zip"
        ),
        "fallback": "https://fapi.binance.com/fapi/v1/klines",
        "timestamp_column": "open_time",
        "timestamp_semantics": "candle open instant; the close is complete at open + 1h",
        "publication": "complete at the candle's close",
        "cadence_hours": 1,
    },
    {
        "field": "spot_price",
        "venue": "binance",
        "market_type": "spot",
        "symbol": "BTCUSDT",
        "primary": "already committed: data/research/btc_usdt_1h_gen1_raw_pre_styx.parquet",
        "fallback": None,
        "timestamp_column": "date",
        "timestamp_semantics": "candle open instant; the close is complete at open + 1h",
        "publication": "complete at the candle's close",
        "cadence_hours": 1,
    },
)

#: How long an observation may be carried forward before its row leaves the
#: sample universe. Funding is 8-hourly by construction, so holding the last
#: settlement is the definition of the feature rather than a repair; one extra
#: hour is tolerance for a late archive row. Open interest is a 5-minute
#: snapshot, so an hour is already twelve missed snapshots.
MAX_STALENESS_HOURS = {"funding_rate": 9, "open_interest": 1, "perpetual_price": 1}

#: The two evaluation regions, in canonical processed-dataset row numbers.
#:
#: `research_rows` is 48,217 — the row the contract's sealed instant resolves to
#: — and the horizon is 6, so the last row whose label can be read without
#: touching a sealed close is 48,210. Everything about the second region follows
#: from those two numbers and nothing about its content.
EXPLORATORY_OUTER_BLOCKS: tuple[tuple[int, int], ...] = (
    (26518, 31339),
    (31339, 36160),
    (36160, 40981),
    (40981, 45802),
)
HOLDOUT_ROWS = (45802, 48211)
RESEARCH_ROWS = 48217

#: Every artifact that has read the exploratory blocks. The list is why they
#: cannot confirm anything, and it is recorded so that the claim is checkable
#: rather than asserted.
EXPLORATORY_BLOCKS_READ_BY: tuple[str, ...] = (
    "v4 (five seeds)",
    "P2a",
    "P2b",
    "P2b ablation",
    "P2b regime description",
    "P2c",
    "P3",
)

#: Minimum non-overlapping outer trades, **for each of the two compared arms**,
#: before a fold's net-return delta counts as an observation of anything.
#:
#: The principle is this repository's own: `nn.p2b.MIN_TRADES = 10` already
#: declares that a trading statistic computed on fewer than ten realised trades
#: is too thin to *select* a threshold on. It is incoherent to refuse to select
#: on four trades and then report a fold built on four as one of four temporal
#: observations, which is exactly what P3 did — its frozen `xgboost x ohlcv14`
#: control realised 4 trades in fold 2 and 11 in fold 1, and both counted the
#: same as fold 0's 80.
#:
#: A higher bar was considered and rejected on evidence rather than on taste.
#: Twenty — twice the selection floor, on the argument that an evaluation claim
#: should need more exposure than a selection — would have invalidated **three**
#: of that control's four folds, so no arm behaving like the control this
#: programme has been comparing against could ever reach three valid folds.
#: Preregistering a bar nothing can clear is not rigour; it is a way of
#: guaranteeing a negative and calling it a decision rule. Ten bites on the
#: pathological fold, leaves three, and makes the thinness of the evaluation a
#: recorded limitation instead of a hidden one.
MIN_OUTER_TRADES = 10

#: Under an independent p = 0.5 null, P(X >= 3 of 4) = 5/16.
#:
#: Recorded because it is the number that makes "3 of 4" not evidence: a coin
#: clears that bar 31% of the time. It is also why stage 1 is a *screen* and not
#: a test — at n = 4 dependent folds no threshold has both a usable false-positive
#: rate and usable power, and pretending otherwise is the failure mode this
#: constant exists to name.
NULL_PROBABILITY_THREE_OF_FOUR = 5 / 16

#: Stage 1 continues only if every one of these holds. All three, not any.
STAGE_1_CONTINUATION = {
    "valid_folds_required": 3,
    "improved_folds_required": 3,
    "mean_delta_above": 0.0,
    "worst_fold_delta_at_least": -0.02,
    "screen_false_positive_rate_under_coin_null": NULL_PROBABILITY_THREE_OF_FOUR,
    "note": (
        "a screening rule, not a test. Its only job is to decide whether to spend "
        "the single-use holdout; a false continuation costs one holdout evaluation "
        "and is recorded as such"
    ),
}

#: What each stage-2 outcome means, and what happens next. Fixed before the run.
STAGE_2_OUTCOMES: tuple[dict[str, str], ...] = (
    {
        "label": "negative",
        "when": "holdout net-return delta (combined - control) <= 0",
        "classification": "negative evidence",
        "then": (
            "P4 is negative and closed. No re-fit, no window change, no second "
            "arm on these rows, and the holdout is not evaluated again"
        ),
    },
    {
        "label": "inconclusive",
        "when": (
            "delta > 0, but either arm realises fewer than MIN_OUTER_TRADES outer "
            "trades on the holdout, or the combined arm's absolute net return after "
            "costs is <= 0"
        ),
        "classification": "hypothesis-generating",
        "then": (
            "recorded as hypothesis-generating. A follow-up needs a new research "
            "generation with its own evaluation data; it may not re-use these rows"
        ),
    },
    {
        "label": "supported",
        "when": (
            "delta > 0, both arms realise at least MIN_OUTER_TRADES outer trades, "
            "the combined arm's absolute net return after costs is > 0, and the sign "
            "of the delta survives the 1.5x cost sensitivity"
        ),
        "classification": "single-region supported; NOT confirmatory",
        "then": (
            "P4 is worth confirming. Confirmation is a new research generation — a "
            "new contract over a market or a time region this programme has not "
            "evaluated. Styx is not opened to resolve it"
        ),
    },
)

#: The strongest label P4's evaluation data can carry, whatever it returns.
RESEARCH_CLASSIFICATION = (
    "adaptive on the four exploratory blocks; single-region and never-sealed on the "
    "holdout. P4 cannot produce confirmatory evidence."
)

#: Every choice P4 introduces, and the sentence that closes it.
DEGREES_OF_FREEDOM: tuple[dict[str, str], ...] = (
    {
        "choice": "which derivatives fields to read",
        "constrained_by": "three, fixed here: funding, open interest, basis. No others.",
    },
    {
        "choice": "how many features to build from them",
        "constrained_by": f"exactly {len(FEATURES)}, named in FEATURES. No additions.",
    },
    {
        "choice": "the trailing windows (9, 24, 30, 168)",
        "constrained_by": "declared per feature here; no window may be re-chosen.",
    },
    {
        "choice": "the clips",
        "constrained_by": "declared per feature here; a clip is not a threshold to tune.",
    },
    {
        "choice": "which venue and instrument",
        "constrained_by": "Binance USD-M BTCUSDT perpetual, in DATA_SOURCES.",
    },
    {
        "choice": "which model",
        "constrained_by": (
            "one primary, xgboost, in its unchanged P2a configuration. The other two "
            "are secondary and cannot change the answer."
        ),
    },
    {
        "choice": "which comparison decides",
        "constrained_by": "one: combined vs control, on the primary model.",
    },
    {
        "choice": "the horizon and the target",
        "constrained_by": "unchanged from v4; changing them is a different question.",
    },
    {
        "choice": "the cost model",
        "constrained_by": (
            "unchanged from v4, with a fixed 1.0x/1.5x/2.0x sensitivity reported for "
            "every arm and never used to pick the headline."
        ),
    },
    {
        "choice": "the sample universe",
        "constrained_by": (
            "the intersection rule below, applied identically to every arm including "
            "the control, which is re-run on it rather than copied."
        ),
    },
    {
        "choice": "which folds count",
        "constrained_by": (
            f"a fold in which either compared arm realises fewer than "
            f"{MIN_OUTER_TRADES} outer trades is invalid for that comparison."
        ),
    },
    {
        "choice": "whether to look at the holdout",
        "constrained_by": (
            "only if stage 1 continues, once, on a snapshot that stage 1's own data "
            "file structurally cannot reach."
        ),
    },
    {
        "choice": "what a result means",
        "constrained_by": "STAGE_2_OUTCOMES, fixed before the run.",
    },
)


def preregistration() -> dict[str, Any]:
    """The whole commitment, as one JSON-serialisable object."""
    return {
        "checkpoint": "P4",
        "question": QUESTION,
        "research_question_id": RESEARCH_QUESTION_ID,
        "arms": list(ARMS),
        "primary_comparison": list(PRIMARY_COMPARISON),
        "primary_model": PRIMARY_MODEL,
        "secondary_models": list(SECONDARY_MODELS),
        "target": dict(TARGET),
        "cost_sensitivity_multipliers": list(COST_SENSITIVITY_MULTIPLIERS),
        "features": [dict(feature) for feature in FEATURES],
        "warmup_hours": WARMUP_HOURS,
        "data_sources": [dict(source) for source in DATA_SOURCES],
        "max_staleness_hours": dict(MAX_STALENESS_HOURS),
        "exploratory_outer_blocks": [list(block) for block in EXPLORATORY_OUTER_BLOCKS],
        "exploratory_blocks_read_by": list(EXPLORATORY_BLOCKS_READ_BY),
        "holdout_rows": list(HOLDOUT_ROWS),
        "research_rows": RESEARCH_ROWS,
        "min_outer_trades": MIN_OUTER_TRADES,
        "stage_1_continuation": dict(STAGE_1_CONTINUATION),
        "stage_2_outcomes": [dict(outcome) for outcome in STAGE_2_OUTCOMES],
        "research_classification": RESEARCH_CLASSIFICATION,
        "degrees_of_freedom": [dict(entry) for entry in DEGREES_OF_FREEDOM],
        "styx": (
            "not opened, not moved, and not available to resolve an ambiguous P4 "
            "result under any outcome"
        ),
    }


def preregistration_hash() -> str:
    """SHA-256 over :func:`preregistration`, canonically encoded.

    What a P4 cell will record, so that a cell produced under an edited
    preregistration is a different object rather than the same one with a
    different story.
    """
    material = json.dumps(
        preregistration(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()
