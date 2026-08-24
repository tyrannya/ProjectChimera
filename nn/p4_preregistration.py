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
        # A *daily* archive. The first version of this preregistration named a
        # monthly metrics path; Binance does not publish one, so the source it
        # committed to did not exist. Corrected here, before any probe and
        # before any P4 number, which is the only time a source may be changed.
        "primary": (
            "https://data.binance.vision/data/futures/um/daily/metrics/BTCUSDT/"
            "BTCUSDT-metrics-{year}-{month}-{day}.zip "
            "(sum_open_interest, sum_open_interest_value)"
        ),
        "archive_granularity": "daily; one archive per UTC day",
        "earliest_intended_availability": "2020-09-01",
        "availability_note": (
            "2020-09-01 is the earliest BTCUSDT metrics day this preregistration "
            "intends to request. It is a stated intent, not a measured fact: nothing "
            "here has touched the network. The probe step establishes the real first "
            "available day from metadata only, and a first day later than this one "
            "narrows the sample universe rather than being worked around"
        ),
        "missing_day_behaviour": (
            "a daily archive that 404s, fails its checksum, or is short is a MISSING "
            "DAY, never an interpolated one. One missing day removes 288 five-minute "
            "snapshots, so under the 1-hour staleness bound every hour of that UTC day "
            "leaves the sample universe for EVERY arm, the control included. Missing "
            "days are counted, reported per block, and fed to the availability gate"
        ),
        "coverage_failure_behaviour": (
            "fail closed: an archive that cannot be fetched or verified stops the "
            "acquisition rather than being skipped, and a block that fails the "
            "availability rule makes the checkpoint not_evaluable rather than negative"
        ),
        "fallback": (
            "https://fapi.binance.com/futures/data/openInterestHist — DIAGNOSTIC AND "
            "FALLBACK ONLY. It retains 30 days, so it cannot build this history at "
            "all. It may be used to spot-check a handful of recent archive rows and "
            "may NEVER silently stand in for a missing archive day: a row sourced "
            "from REST is not a row of the preregistered historical source, and an "
            "acquisition that substituted one would be reporting a universe it did "
            "not have"
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

#: Which CSV columns the funding archive is allowed to be read from.
#:
#: Fixed **before any archive has been opened**, because "we looked at the file
#: and mapped the columns that worked" is a researcher degree of freedom wearing
#: the clothes of an implementation detail. Binance has published funding data
#: under more than one column layout; a reader that inferred the mapping at
#: acquisition time would be choosing, after seeing the data, which column is
#: the rate.
#:
#: The rule is an allow-list, not a heuristic. A header row must match one of
#: the maps below *exactly* by column-name set. A headerless file is accepted
#: only in the single unambiguous two-column shape, and only when its first
#: column parses as an epoch instant inside the archive's own calendar day —
#: the same "let the archive's period decide" test
#: :func:`nn.trade_aggregates.resolve_epoch_unit` applies to trades. Anything
#: else is a refusal, and extending this list is a commit that moves
#: :func:`preregistration_hash` and may only happen before a P4 outer number
#: exists.
FUNDING_CSV_COLUMN_POLICY: dict[str, Any] = {
    "canonical_fields": ["settlement_instant", "realised_funding_rate"],
    "allowed_header_maps": [
        {
            "layout": "fundingTime/fundingRate",
            "columns": ["fundingTime", "fundingRate"],
            "settlement_instant": "fundingTime",
            "realised_funding_rate": "fundingRate",
        },
        {
            "layout": "calc_time/funding_interval_hours/last_funding_rate",
            "columns": ["calc_time", "funding_interval_hours", "last_funding_rate"],
            "settlement_instant": "calc_time",
            "realised_funding_rate": "last_funding_rate",
        },
        {
            "layout": "calc_time/last_funding_rate",
            "columns": ["calc_time", "last_funding_rate"],
            "settlement_instant": "calc_time",
            "realised_funding_rate": "last_funding_rate",
        },
    ],
    "headerless_positional_layout": {
        "columns": 2,
        "settlement_instant": 0,
        "realised_funding_rate": 1,
        "condition": (
            "accepted only when the first column parses as an epoch instant inside the "
            "archive's own calendar period under exactly one supported unit"
        ),
    },
    "on_unrecognised_layout": (
        "refuse the acquisition and stop. Do not infer a mapping, do not fall back to "
        "positional order, and do not read the REST endpoint instead. Extending "
        "allowed_header_maps is a commit that moves the preregistration hash and is "
        "permitted only while no P4 outer number exists"
    ),
    "extra_columns": (
        "a recognised layout may carry columns this policy does not name; they are "
        "ignored. An unrecognised column-name SET is a refusal"
    ),
}

#: When rows repeating one instant may be collapsed, and when they may not.
#:
#: **This is amendment A1, and it is not an original preregistration rule.** The
#: rule written before any archive was opened rejected *every* duplicate
#: timestamp in *every* source, with no normalisation of any kind. That rule was
#: narrowed after the first bounded real-source inspection and before any P4
#: model fit, Stage-1 result or outcome existed. The original wording is kept in
#: ``supersedes`` rather than overwritten, and ``docs/p4_preregistration.md``
#: §3.4a records the same history in prose. Nothing about a P4 *result*
#: motivated it, because no P4 result exists.
#:
#: What forced it is in ``adopted_because``: two official archives were measured
#: and found to publish every row twice, byte-identically. That is a
#: source-format defect, not two conflicting observations — and the original rule
#: was written for conflicting observations. A reader that cannot tell which of
#: two rows is the observation must stop; a reader looking at two copies of one
#: row has nothing to tell apart and nothing to choose between.
#:
#: The narrowing is deliberately as small as it can be: one field, one archive,
#: one grouping key, equality over the *complete* published row, and a refusal
#: everywhere else. Widening it — another source, a subset of columns, a
#: tie-break of any kind — is a commit that moves :func:`preregistration_hash`
#: and is permitted only while no P4 outer number exists.
#:
#: :func:`nn.derivatives_sources.source_spec` reads this object rather than
#: restating it, so the acquisition cannot come to mean something the
#: preregistration does not say.
OPEN_INTEREST_DUPLICATE_POLICY: dict[str, Any] = {
    "amendment": "A1",
    "amendment_status": (
        "adopted after inspection of real source archives revealed a source-format "
        "defect, and before any P4 model fit, Stage-1 result, outcome or holdout "
        "access existed. This is a source-protocol amendment, not an original "
        "preregistration rule"
    ),
    "supersedes": (
        "the original rule, under which a duplicate timestamp in ANY source was a "
        "rejection of the acquisition and never a de-duplication"
    ),
    "adopted_because": (
        "the official Binance USD-M daily metrics archives for BTCUSDT dated "
        "2020-09-01 and 2020-09-02 were observed to hold 576 data rows for 288 "
        "create_time instants, every observation published twice with the paired "
        "rows identical across every column of the CSV. The observed defect is "
        "exact row duplication, not conflicting observations. Two archives were "
        "inspected; no claim is made that any other day is duplicated"
    ),
    "scope": {
        "field": "open_interest",
        "archive": "Binance USD-M daily metrics, data/futures/um/daily/metrics",
        "applies_to": ["open_interest"],
        "does_not_apply_to": ["funding_rate", "perpetual_price"],
    },
    "grouping_key": "create_time",
    "acceptance": (
        "rows sharing one grouping_key value are accepted only when the complete "
        "published CSV row is identical across every column. The comparison runs "
        "after schema validation and covers every column the archive carries, "
        "including columns no derivatives_v1 feature consumes"
    ),
    "normalisation": (
        "N identical rows at one instant collapse to exactly one logical "
        "observation; the N - 1 removed rows are counted"
    ),
    "on_conflict": (
        "if any field differs, reject the acquisition. Never choose the first row, "
        "never choose the last, never average, never infer which row is correct"
    ),
    "provenance_required": [
        "rows_read",
        "observations_retained",
        "exact_duplicate_rows_collapsed",
        "duplicate_instants",
    ],
    "other_sources": (
        "a duplicate timestamp in the funding or perpetual-price archive remains a "
        "rejection of the acquisition, with no normalisation of any kind"
    ),
    "schema_precedence": (
        "schema identity is validated before normalisation, so an archive whose "
        "header is not the preregistered one is refused for its schema and never "
        "reaches this rule"
    ),
}

#: Where the published funding archive *begins*, and what a month before it is.
#:
#: **This is amendment A2, and it is not an original preregistration rule.** The
#: original design planned funding as a continuous monthly source over whatever
#: window the generic warm-up planner asked for, and treated every absent month
#: as a hard failure without distinguishing a hole inside a published sequence
#: from a month before the sequence starts. The distinction is made here, after
#: the first real full-acquisition attempt met the second case, and before any
#: P4 model fit, Stage-1 result, outcome or holdout access existed. The original
#: wording is kept in ``supersedes`` rather than overwritten, and
#: ``docs/p4_preregistration.md`` §3.4b records the same history in prose.
#:
#: What forced it is in ``adopted_because``: the acquisition asks for 2019-12
#: only to give the 30-settlement funding window somewhere to warm up, and that
#: month is not published. The checks that established it are bounded and are
#: recorded as status codes rather than as a conclusion about the market: four
#: consecutive months before 2020-01 are absent and the two months from 2020-01
#: are present, so the first required missing archive is *before* the observed
#: beginning of the sequence rather than inside it.
#:
#: **This is a source-inception boundary and not a relaxation of §3.4.** After
#: 2020-01 every expected month is still mandatory and a missing one still stops
#: the acquisition. Nothing here permits a synthetic value, a REST backfill, an
#: interpolation, or a different source for the months the archive does not
#: publish — the history simply begins where the source begins, and the existing
#: causality, warm-up and sample-universe rules decide when a funding feature
#: first becomes defined.
#:
#: :func:`nn.derivatives_sources.plan_archives` and
#: :func:`nn.derivatives_sources.source_spec` read this object rather than
#: restating it, so a checkout in which the design says 2020-01 and the planner
#: asks for 2019-12 cannot be constructed.
FUNDING_ARCHIVE_INCEPTION_POLICY: dict[str, Any] = {
    "amendment": "A2",
    "amendment_status": (
        "adopted after the first real full-acquisition attempt found the requested "
        "warm-up month absent from the published archive, and before any P4 model "
        "fit, Stage-1 result, outcome or holdout access existed. This is a "
        "source-protocol amendment, not an original preregistration rule"
    ),
    "supersedes": (
        "the original rule, under which funding was planned as a continuous monthly "
        "source from whatever month the generic warm-up planner requested, and every "
        "absent month was an internal continuity gap and a hard failure"
    ),
    "scope": {
        "field": "funding_rate",
        "archive": (
            "Binance USD-M BTCUSDT monthly fundingRate archive, "
            "data/futures/um/monthly/fundingRate"
        ),
        "applies_to": ["funding_rate"],
        "does_not_apply_to": ["open_interest", "perpetual_price"],
    },
    "first_protocol_month": "2020-01",
    "first_protocol_instant": "2020-01-01T00:00:00+00:00",
    "adopted_because": (
        "the acquisition plan requests funding from 2019-12 to warm the 30-settlement "
        "window, and real HEAD checks of the monthly fundingRate archive on the "
        "acquisition host returned 404 for 2019-12 and 200 for 2020-01. The first "
        "required missing archive therefore falls BEFORE the observed beginning of the "
        "published monthly sequence rather than inside an established one. The "
        "downloaded BTCUSDT-fundingRate-2020-01.zip begins at 2020-01-01T00:00:00Z "
        "under the already-allowed calc_time/funding_interval_hours/last_funding_rate "
        "layout, so no column rule changes"
    ),
    "observed_evidence": {
        "checked": "HTTP status of the published monthly archive, one request per month",
        "months": [
            {"month": "2019-09", "status": 404, "published": False},
            {"month": "2019-10", "status": 404, "published": False},
            {"month": "2019-11", "status": 404, "published": False},
            {"month": "2019-12", "status": 404, "published": False},
            {"month": "2020-01", "status": 200, "published": True},
            {"month": "2020-02", "status": 200, "published": True},
        ],
        "first_observed_row": "2020-01-01T00:00:00+00:00",
        "observed_layout": "calc_time/funding_interval_hours/last_funding_rate",
    },
    "generalisation_limit": (
        "six months were checked and that is all that is claimed. This says nothing "
        "about whether BTCUSDT funding of any kind existed before January 2020, only "
        "that the preregistered Binance monthly fundingRate archive is not published "
        "for the months checked before it and is published from it"
    ),
    "pre_inception_behaviour": (
        "a requested month strictly before first_protocol_month is OUTSIDE the "
        "published archive source and is NOT an internal continuity gap. It is not "
        "counted as a missing month, is not requested, and never reaches the "
        "fail-closed rule that governs the months the source does publish"
    ),
    "acquisition_start_rule": (
        "the funding archive plan begins at max(generic requested start, "
        "first_protocol_month). The generic research spine boundary and the requested "
        "feature warm-up are unchanged; only the funding archive iterator is clamped"
    ),
    "no_substitution": (
        "never. No synthetic funding value is created for the pre-inception region, no "
        "REST endpoint backfills it, nothing is interpolated or carried backwards, and "
        "no other venue, instrument or archive stands in for it"
    ),
    "post_inception_continuity": (
        "unchanged and fail-closed: every expected month from first_protocol_month "
        "onward is mandatory, and a month that 404s, fails its checksum or cannot be "
        "read stops the acquisition. No gap after inception may be skipped, "
        "interpolated, forward-invented or replaced by a different source"
    ),
    "downstream_consequence": (
        "the funding history begins where the source begins, and the unchanged "
        "causality, warm-up and sample-universe rules decide the rest: a row at which "
        "fewer than the required settlements have accumulated has no defined "
        "drv_funding_sum_9 or drv_funding_z and is outside the common sample universe "
        "for EVERY arm, the control included. Losing early training rows this way is "
        "the intended consequence and is reported, not repaired"
    ),
    "windows_unchanged": (
        "no feature, window, clip, staleness bound, availability threshold, target, "
        "horizon, Stage-1 geometry, cost model or comparison rule is changed by this "
        "amendment. The 30-settlement funding z-score and the 9-settlement funding sum "
        "are exactly what they were"
    ),
    "provenance_required": [
        "generic_requested_from",
        "source_inception_month",
        "effective_from",
        "months_clamped",
    ],
    "other_sources": (
        "open interest keeps its own §3.0a first-day rule and perpetual klines keep "
        "theirs; A2 clamps the funding archive plan and nothing else"
    ),
}

#: Where the published perpetual kline archive *begins*, and what a month before
#: it is.
#:
#: **This is amendment A3, and it is not an original preregistration rule.** The
#: original design planned the perpetual klines exactly as it planned funding: a
#: continuous monthly source over whatever window the generic warm-up planner
#: asked for, with every absent month a hard failure and no distinction between a
#: hole inside a published sequence and a month before the sequence starts. A2
#: drew that distinction for the funding archive and explicitly left the kline
#: archive alone, because at that point the kline archive had not been checked.
#: It has now been, on the next real acquisition attempt, and it has the same
#: boundary in the same place. The distinction is drawn here for the perpetual
#: source specifically, before any P4 model fit, Stage-1 result, outcome or
#: holdout access existed. The original wording is kept in ``supersedes`` rather
#: than overwritten, and ``docs/p4_preregistration.md`` §3.4c records the same
#: history in prose.
#:
#: **A3 is not A2 and does not extend it.** Two archives were measured separately
#: and each carries its own object, its own evidence and its own scope. That they
#: agree on the month is an observation about two sources, not one rule about a
#: venue: an edit that moved one month would move exactly one plan, which is what
#: keeping them apart is for.
#:
#: **This is a source-inception boundary and not a relaxation of §3.4.** After
#: 2020-01 every expected month is still mandatory and a missing one still stops
#: the acquisition. Nothing here permits a synthetic price, a REST backfill, an
#: interpolation, a carried-back close, or the spot candle history standing in for
#: the perpetual leg over the months the archive does not publish — the history
#: simply begins where the source begins, and the existing causality, staleness
#: and sample-universe rules decide when a basis feature first becomes defined.
#:
#: :func:`nn.derivatives_sources.plan_archives` and
#: :func:`nn.derivatives_sources.source_spec` read this object rather than
#: restating it, so a checkout in which the design says 2020-01 and the planner
#: asks for 2019-12 cannot be constructed.
PERPETUAL_KLINE_ARCHIVE_INCEPTION_POLICY: dict[str, Any] = {
    "amendment": "A3",
    "amendment_status": (
        "adopted after the next real full-acquisition attempt found the requested "
        "warm-up month absent from the published monthly kline archive, and before any "
        "P4 model fit, Stage-1 result, outcome or holdout access existed. This is a "
        "source-protocol amendment, not an original preregistration rule"
    ),
    "supersedes": (
        "the original rule, under which the perpetual klines were planned as a "
        "continuous monthly source from whatever month the generic warm-up planner "
        "requested, and every absent month was an internal continuity gap and a hard "
        "failure. A2 left this source explicitly untouched; A3 is what touches it"
    ),
    "relationship_to_a2": (
        "separate and parallel, not an extension. A2 measured the monthly fundingRate "
        "archive and clamps the funding plan; A3 measures the monthly 1h kline archive "
        "and clamps the perpetual-price plan. The two objects are independent: they "
        "happen to name the same month because two archives were observed to begin in "
        "the same month, and moving either month moves exactly one plan"
    ),
    "scope": {
        "field": "perpetual_price",
        "archive": (
            "Binance USD-M BTCUSDT MONTHLY 1h perpetual kline archive, "
            "data/futures/um/monthly/klines/BTCUSDT/1h"
        ),
        "applies_to": ["perpetual_price"],
        "does_not_apply_to": ["funding_rate", "open_interest"],
    },
    "first_protocol_month": "2020-01",
    "first_protocol_instant": "2020-01-01T00:00:00+00:00",
    "adopted_because": (
        "the acquisition plan requests perpetual-price history from 2019-12 to provide "
        "pre-spine warm-up and source overlap, and real checks of the monthly 1h kline "
        "archive on the acquisition host returned 404 for 2019-12 and 200 for 2020-01. "
        "The first required missing archive therefore falls BEFORE the observed "
        "beginning of the published monthly sequence rather than inside an established "
        "one. The downloaded BTCUSDT-1h-2020-01.zip begins at 2020-01-01T00:00:00Z "
        "under the already-allowed 12-field Binance kline layout, so no column rule "
        "changes"
    ),
    "observed_evidence": {
        "checked": "HTTP status of the published monthly archive, one request per month",
        "months": [
            {"month": "2019-09", "status": 404, "published": False},
            {"month": "2019-10", "status": 404, "published": False},
            {"month": "2019-11", "status": 404, "published": False},
            {"month": "2019-12", "status": 404, "published": False},
            {"month": "2020-01", "status": 200, "published": True},
            {"month": "2020-02", "status": 200, "published": True},
        ],
        "first_observed_row": "2020-01-01T00:00:00+00:00",
        "first_observed_open_time_ms": 1577836800000,
        "observed_layout": "the existing Binance 12-field kline schema, unchanged",
        "observed_columns": 12,
    },
    "generalisation_limit": (
        "six months were checked and that is all that is claimed. This says nothing "
        "about whether BTCUSDT perpetual-price data of any kind existed before January "
        "2020, only that the preregistered Binance USD-M monthly 1h kline archive is "
        "not published for the months checked before it and is published from it"
    ),
    "pre_inception_behaviour": (
        "a requested month strictly before first_protocol_month is OUTSIDE the "
        "published archive source and is NOT an internal continuity gap. It is not "
        "counted as a missing month, is not requested, and never reaches the "
        "fail-closed rule that governs the months the source does publish"
    ),
    "acquisition_start_rule": (
        "the perpetual kline archive plan begins at max(generic requested start, "
        "first_protocol_month). The generic research spine boundary, the requested "
        "feature warm-up, the spot denominator and the basis formula are unchanged; "
        "only the perpetual kline archive iterator is clamped"
    ),
    "no_substitution": (
        "never. No synthetic perpetual price is created for the pre-inception region, "
        "no REST endpoint backfills it, nothing is interpolated or carried backwards, "
        "the spot candle history does NOT stand in for the missing perpetual leg, and "
        "no other venue, instrument or archive is read in its place"
    ),
    "no_spot_substitution": (
        "explicitly. The spot close is the basis DENOMINATOR and is not a proxy for its "
        "numerator: substituting it would make drv_basis identically zero and report a "
        "measured spread where none was observed. A row whose perpetual input is "
        "unavailable has no basis, and that is the answer rather than a problem"
    ),
    "post_inception_continuity": (
        "unchanged and fail-closed: every expected month from first_protocol_month "
        "onward is mandatory, and a month that 404s, fails its checksum or cannot be "
        "read stops the acquisition. No gap after inception may be skipped, "
        "interpolated, forward-filled, backward-filled or replaced by a different "
        "source"
    ),
    "downstream_consequence": (
        "basis-derived features become available only where their required "
        "perpetual-price input is actually present. The perpetual history begins where "
        "the source begins, and the unchanged causality, staleness and sample-universe "
        "rules decide the rest: a row with no perpetual close inside the 1-hour "
        "staleness bound has no defined drv_basis and no defined drv_basis_z, and is "
        "outside the common sample universe for EVERY arm, the control included. "
        "Losing early rows this way is the intended consequence and is reported, not "
        "repaired"
    ),
    "windows_unchanged": (
        "no feature, window, clip, staleness bound, availability threshold, target, "
        "horizon, Stage-1 geometry, cost model or comparison rule is changed by this "
        "amendment. drv_basis is still the perpetual close over the spot close minus "
        "one, clipped to [-0.02, 0.02], and drv_basis_z is still its 168-hour z-score"
    ),
    "provenance_required": [
        "generic_requested_from",
        "source_inception_month",
        "effective_from",
        "months_clamped",
        "not_an_internal_gap",
        "amendment",
    ],
    "other_sources": (
        "funding keeps A2's own inception object and open interest keeps its §3.0a "
        "first-day rule; A3 clamps the perpetual kline archive plan and nothing else"
    ),
}

#: What counts as an open-interest *observation*, and what is merely a row.
#:
#: **This is amendment A4, and it is not an original preregistration rule.** The
#: original design had no notion of an invalid observation at all: every row that
#: survived schema validation and the duplicate rule was an observation, and the
#: only question left was how stale it was. That worked because nothing had ever
#: read the source at row level, and the assumption underneath — that a published
#: metrics row carries a positive open-interest state — had never been checked.
#: It has now been, and it is false. The original wording is kept in
#: ``supersedes`` rather than overwritten, and ``docs/p4_preregistration.md``
#: §3.4d records the same history in prose.
#:
#: **What forced it is in ``adopted_because``.** The first real full acquisition
#: after A3 completed all 1722 planned open-interest days and then failed during
#: construction and validation of the hourly table, on the pre-write check that
#: refuses a non-positive open interest on an hour marked available. The source
#: was then inspected directly, before any P4 fit, Stage-1 result or outcome
#: existed, and the rows behind that failure are published rows carrying zero.
#:
#: **What this rule does NOT say.** It does not say that BTCUSDT's real economic
#: open interest was zero during those rows, and it does not say that Binance
#: documents zero as an outage sentinel — no official statement to that effect is
#: known to this repository, so none is claimed. What is claimed is narrower and
#: is the whole of the rule: a consumed metric of exactly zero is not a valid
#: positive open-interest state under this protocol, so it is not admitted as an
#: observation. Nothing is inferred about what the true state was; the protocol
#: refuses to name one.
#:
#: **Where it sits in the order.** After schema validation and after A1's exact
#: duplicate normalisation, and nowhere else. A1 must continue to describe what
#: the archive *published*, so a zero row duplicated identically is collapsed by
#: A1 first and classified once by A4 afterwards. Classifying before A1 would
#: make the duplicate accounting a statement about rows this rule had already
#: removed, which is not what that accounting is for.
#:
#: :func:`nn.derivatives_sources.classify_open_interest_observations` is this
#: rule's implementation and :func:`nn.derivatives_sources.source_spec` reads this
#: object rather than restating it, so the acquisition cannot come to mean
#: something the preregistration does not say.
OPEN_INTEREST_OBSERVATION_VALIDITY_POLICY: dict[str, Any] = {
    "amendment": "A4",
    "amendment_status": (
        "adopted after the first real full acquisition following A3 completed the whole "
        "planned open-interest download and then failed at pre-write hourly-table "
        "validation, and after the source was inspected directly — and before any P4 "
        "model fit, Stage-1 result, outcome or holdout access existed. This is a "
        "source-protocol amendment, not an original preregistration rule"
    ),
    "supersedes": (
        "the original rule, under which every metrics row surviving schema validation "
        "was an open-interest observation regardless of its value, the only remaining "
        "question was staleness, and a published zero would have been carried into the "
        "hourly table as a real zero open-interest state"
    ),
    "scope": {
        "field": "open_interest",
        "archive": (
            "Binance USD-M BTCUSDT DAILY metrics archives, "
            "data/futures/um/daily/metrics/BTCUSDT/BTCUSDT-metrics-YYYY-MM-DD.zip"
        ),
        "applies_to": ["open_interest"],
        "does_not_apply_to": ["funding_rate", "perpetual_price"],
        "consumed_fields": ["sum_open_interest", "sum_open_interest_value"],
    },
    "applies_after": (
        "schema validation and A1 exact-duplicate normalisation, in that order. The "
        "complete published CSV rows are read, A1 collapses rows that repeat one "
        "create_time identically, the consumed numeric fields are parsed, and only then "
        "is each retained logical row classified here"
    ),
    "validity_rule": (
        "a retained logical row is a VALID open-interest observation if and only if "
        "sum_open_interest > 0 AND sum_open_interest_value > 0. Both consumed metrics "
        "must be strictly positive"
    ),
    "exactly_zero_is_invalid": (
        "a row in which either consumed field is EXACTLY ZERO is an invalid observation "
        "and not a valid zero open-interest state. It is a published zero-valued "
        "observation that is invalid under this protocol's positive-state requirement, "
        "and no further interpretation of it is asserted"
    ),
    "invalid_row_handling": [
        "remains a real published source row, counted in provenance and accounting",
        "is classified as an invalid OI observation",
        "does NOT enter the causal OI observation sequence",
        "does NOT update the hourly last-observation reducer",
        "is NOT replaced with another numeric value",
        "is NOT interpolated",
        "is NOT averaged",
        "is NOT repaired from the REST endpoint",
        "is NOT repaired from a future observation",
        "does NOT turn the archive or the UTC day into a missing day on its own",
    ],
    "on_negative_or_nonfinite": (
        "HARD FAIL. A consumed OI field that is negative, non-finite or unparseable "
        "stops the acquisition. It is never classified as an ordinary zero-invalid "
        "observation, never dropped and never repaired: none was observed, so a run "
        "that meets one has met something this inspection did not measure"
    ),
    "staleness_unchanged": (
        "after an invalid row the existing as-of rule is unchanged: the most recent "
        "PRIOR VALID observation stays visible only while its age satisfies the "
        "preregistered open_interest staleness bound of 1 hour in MAX_STALENESS_HOURS. "
        "Once it is older than that bound, open interest is UNAVAILABLE. A long run of "
        "invalid rows therefore produces unavailable hours; the prior valid observation "
        "is never carried across it"
    ),
    "later_valid_observation": (
        "a valid observation after an invalid run restores availability PROSPECTIVELY "
        "only. It is never applied backwards to the hours the run left unavailable"
    ),
    "missing_day_relationship": (
        "an archive or UTC day containing invalid rows is NOT thereby a §3.0a missing "
        "day. The missing-day rule is unchanged and still applies only to an archive "
        "that 404s, fails its published checksum or arrives short"
    ),
    "downstream_consequence": (
        "hours with no valid open-interest observation inside the staleness bound have "
        "no defined open-interest input and are outside the common sample universe for "
        "EVERY arm, ohlcv14 included. Losing those hours is the intended consequence "
        "and is reported, not repaired"
    ),
    "provenance_required": [
        "logical_observations",
        "valid_positive_observations",
        "invalid_zero_observations",
        "invalid_both_zero_observations",
        "invalid_zero_contracts_only",
        "invalid_zero_notional_only",
        "negative_observations",
        "nonfinite_observations",
    ],
    "accounting_identity": (
        "for every successfully exported archive: logical_observations (after A1 "
        "duplicate collapse) == valid_positive_observations + "
        "invalid_zero_observations, and invalid_zero_observations == "
        "invalid_both_zero_observations + invalid_zero_contracts_only + "
        "invalid_zero_notional_only, the three being disjoint. "
        "negative_observations and nonfinite_observations are 0 in any exported "
        "snapshot because either one stops the acquisition"
    ),
    "adopted_because": (
        "the first real full acquisition after A3 completed all 1722 planned "
        "open-interest days and then failed during construction and validation of the "
        "hourly table with 'open interest is non-positive on an available hour'. Direct "
        "inspection of the source then established that the daily metrics archives "
        "publish rows whose consumed open-interest metrics are exactly zero, in two "
        "distinct shapes: paired zeros in both consumed fields, and positive contracts "
        "with zero notional. The original protocol had no way to describe such a row "
        "except as a real zero open-interest state, which it is not"
    ),
    "observed_evidence": {
        "inspection": (
            "direct inspection of the published archives, before any P4 fit, Stage-1 "
            "result, P4 outcome, P4-HOLD access or Styx access"
        ),
        "paired_zero_rows_exist": (
            "rows with sum_open_interest == 0 AND sum_open_interest_value == 0 were "
            "observed on multiple dates and are not isolated to one date. Observed "
            "examples include 2021-05-22 04:25..05:10 UTC and multiple later dates "
            "through 2025"
        ),
        "max_consecutive_paired_zero_run": {
            "observations": 102,
            "observation_spacing_minutes": 5,
            "minutes": 510,
            "hours": 8.5,
            "note": (
                "the longest consecutive run of paired-zero five-minute observations "
                "found by the scan. It is 8.5 times the 1-hour open-interest staleness "
                "bound, so it cannot be absorbed by the existing carry-forward"
            ),
        },
        "one_sided_zero_rows": {
            "found": 12,
            "shape": "sum_open_interest > 0 and sum_open_interest_value == 0",
            "date": "2023-04-10",
            "all_on_one_date": True,
            "instants_utc": [
                "2023-04-10T08:25:00+00:00",
                "2023-04-10T08:45:00+00:00",
                "2023-04-10T08:55:00+00:00",
                "2023-04-10T09:00:00+00:00",
                "2023-04-10T09:05:00+00:00",
                "2023-04-10T09:10:00+00:00",
                "2023-04-10T09:15:00+00:00",
                "2023-04-10T09:20:00+00:00",
                "2023-04-10T09:25:00+00:00",
                "2023-04-10T09:30:00+00:00",
                "2023-04-10T09:35:00+00:00",
                "2023-04-10T09:40:00+00:00",
            ],
        },
        "scan_coverage": (
            "the complete planned 1722-day open-interest acquisition range, scanned "
            "with errors=0"
        ),
        "scan_days": 1722,
        "scan_errors": 0,
        "negative_consumed_values_observed": 0,
    },
    "generalisation_limit": (
        "this is what one complete scan of the planned range found and that is all that "
        "is claimed. Exactly 12 one-sided zero rows were found and all fell on "
        "2023-04-10; the longest paired-zero run found was 102 observations; no negative "
        "consumed value was found. Nothing here asserts a cause, a mechanism, a "
        "frequency outside the scanned range, or that the pattern will persist"
    ),
    "no_sentinel_claim": (
        "no claim is made that Binance documents zero as a sentinel value, an outage "
        "marker or anything else. No official source saying so is known to this "
        "repository. The rows are described only as published zero-valued observations "
        "that are invalid under this protocol's positive-state requirement"
    ),
    "no_economic_inference": (
        "A4 does not infer the true economic open interest during an invalid row. It "
        "does not assert that open interest was zero, that it was unchanged, or that it "
        "was anything else. The protocol simply refuses to treat a non-positive "
        "consumed metric as a valid positive open-interest state"
    ),
    "windows_unchanged": (
        "no modelling, feature, window, clip, warm-up, target, horizon, cost model, "
        "staleness bound, availability threshold, Stage-1 geometry, holdout rule or "
        "gate is changed by this amendment. drv_oi_log_change_24h, "
        "drv_oi_notional_ratio and drv_oi_price_divergence keep their definitions and "
        "their windows exactly. No epsilon is added to open interest to make a log "
        "ratio defined, and no zero is replaced by a tiny positive value"
    ),
    "other_sources": (
        "funding and the perpetual klines are untouched. A4 governs which metrics rows "
        "become open-interest observations and nothing else; A1 still decides which "
        "published rows are one logical row, A2 and A3 still clamp their own archive "
        "plans, and §3.0a still decides what a missing day is"
    ),
}

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

#: The exact geometry of the one-shot evaluation, in the same row space.
#:
#: These were prose in the document and nothing else, which meant the three
#: numbers that decide what stage 2 fits on, selects on and reports could be
#: changed by an edit that moved no hash. They are values now, they are inside
#: :func:`preregistration`, and moving any of them moves
#: :func:`preregistration_hash`.
STAGE_2_TRAIN_ROWS = (0, 40981)
STAGE_2_SELECTION_ROWS = (40981, 45802)
STAGE_2_EVALUATION_ROWS = HOLDOUT_ROWS

#: The last row a stage-1 dataset may contain, exclusive.
#:
#: Stage 1 reads the committed snapshot, which holds rows [0, 45802) and
#: therefore cannot reach the holdout at all. :mod:`nn.p4_holdout` turns that
#: from a fact about today's file into a checked precondition.
STAGE_1_MAX_ROW_EXCLUSIVE = HOLDOUT_ROWS[0]

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

#: What "improved" means, stated so that a tie cannot be argued either way.
#:
#: A strict inequality on the net-return delta. An exactly-zero delta is **not**
#: an improvement: the claim under test is that the new information adds
#: something, and adding nothing is the null, not a win. Zero is reachable in
#: practice — two arms that take no trades in a fold both return exactly 0.0 —
#: which is precisely why it needed saying before it happened rather than after.
IMPROVED_RULE = {
    "statistic": "outer net return after costs, combined arm minus control arm",
    "improved_when": "delta > 0",
    "zero_is_improved": False,
    "note": (
        "strict. A fold in which both arms take no trades has delta exactly 0.0 and "
        "is not an improvement; it is also invalid under min_outer_trades, and both "
        "readings are recorded so neither has to be decided later"
    ),
}

#: When a block counts as available, given a punctured daily open-interest feed.
#:
#: "Two blocks in full" was the gate and "in full" had no operational meaning.
#: One missing daily OI archive removes a whole UTC day of rows from every arm,
#: so under a literal reading a single 404 anywhere in five years would make
#: every block unavailable and the checkpoint unrunnable — and under a loose
#: reading, any amount of loss could be waved through after the fact.
#:
#: The numbers are set from the mechanism, not from a desired outcome, and
#: before any probe: an outer block is 4,821 rows, so 2% is about 96 hours —
#: four missing days spread across roughly seven months, which is a feed with
#: holes and still the same block. 48 hours is two consecutive missing archives;
#: a longer unbroken outage removes a market episode rather than a sample of
#: hours, and a block missing an episode is not that block.
BLOCK_AVAILABILITY_RULE = {
    "min_surviving_row_fraction": 0.98,
    "max_contiguous_missing_hours": 48,
    "applies_to": "each exploratory outer block, and P4-HOLD, under the same rule",
    "measured_on": (
        "rows surviving the sample-universe conditions, computed before any model is "
        "fitted and reported per block whatever the gate decides"
    ),
    "note": (
        "'in full' means this rule and nothing else. A block that fails it is "
        "UNAVAILABLE and is excluded from stage 1's fold count entirely; it is never "
        "included at a discount"
    ),
}

#: The gate that decides whether P4 can be evaluated at all.
#:
#: Failing it is **not** a research result about derivatives information. It is a
#: statement about what public archives contain, and it is classified as such.
AVAILABILITY_GATE = {
    "requires_exploratory_blocks_available": 2,
    "requires_holdout_available": True,
    "block_rule": "BLOCK_AVAILABILITY_RULE",
    "on_failure": "not_evaluable",
    "note": (
        "two available blocks is below stage 1's three-valid-fold requirement, so an "
        "intersection that only just clears this gate fails stage 1 by construction "
        "and the holdout is never opened. That is the intended behaviour: the gate "
        "stops the acquisition, and stage 1 stops the evaluation"
    ),
}

#: The four conditions a row must satisfy to be in P4's sample universe, applied
#: identically to every arm with the control re-run on the result.
UNIVERSE_CONDITIONS: tuple[dict[str, str], ...] = (
    {
        "condition": "ohlcv14_row",
        "requires": (
            "the row is in the OHLCV14 research spine: past indicator warm-up, no NaN "
            "feature, and its 6-candle label knowable without touching a sealed close"
        ),
    },
    {
        "condition": "derivatives_defined",
        "requires": (
            "every one of the eight derivatives_v1 columns is defined at the row, "
            "after the 240-hour derivatives warm-up"
        ),
    },
    {
        "condition": "within_staleness_bounds",
        "requires": (
            "no observation the row depends on is stale beyond its MAX_STALENESS_HOURS "
            "bound; a missing OI day therefore removes that whole UTC day"
        ),
    },
    {
        "condition": "no_segment_bridge",
        "requires": "no feature's window bridges a spine segment boundary",
    },
)

#: The holdout is spent once, by one checkpoint, ever — or not at all.
HOLDOUT_SPEND_POLICY = {
    "region": list(HOLDOUT_ROWS),
    "evaluations_permitted": 1,
    "checkpoints_permitted": 1,
    "gated_on": "a frozen stage-1 pass artifact satisfying STAGE_1_CONTINUATION",
    "retired_if_unspent": True,
    "note": (
        "if P4 does not spend P4-HOLD — because stage 1 failed, because the "
        "availability gate failed, or because the checkpoint was abandoned — the "
        "region is RETIRED from research-confirmation use anyway. A later P4b or P5 "
        "may not read P4's published stage-1 results, retune against them, and then "
        "present the same rows as a fresh holdout: by then the rows have informed a "
        "design decision and are adaptive like every other block. Retirement does not "
        "forbid describing the region; it forbids treating a result on it as "
        "independent evidence"
    ),
    "does_not_upgrade": (
        "spending the holdout does not make P4 confirmatory. Its maximum label is "
        "single-region supported, never confirmatory"
    ),
    "ledger": "data/research/p4_holdout_ledger.json",
    "enforced_by": "nn.p4_holdout",
}

#: Every label P4 can end with, including the one that is not a research result.
EVIDENCE_CLASSIFICATION = {
    "not_evaluable": (
        "the availability gate failed. This is a fact about what public archives "
        "contain and is NOT negative evidence about derivatives information: nothing "
        "was measured. Recorded as not_evaluable / insufficient_coverage, with the "
        "per-block coverage that produced it"
    ),
    "screened_out": (
        "the availability gate passed and stage 1 did not continue. Exploratory, on "
        "burned blocks; reported as negative for this design at this horizon and not "
        "as a result about the information itself"
    ),
    "negative": "the holdout was spent and the delta was <= 0",
    "hypothesis_generating": "the holdout was spent and the result was inconclusive",
    "single_region_supported": (
        "the holdout was spent and the result was supported. The ceiling. Never "
        "confirmatory, because one never-sealed region of about 2,400 rows cannot be"
    ),
    "maximum_label": "single-region supported; never confirmatory",
}

#: The reason an unavailable checkpoint is not a negative one, as a value.
NOT_EVALUABLE_OUTCOME = {
    "label": "not_evaluable",
    "reason_code": "insufficient_coverage",
    "when": "the availability gate in AVAILABILITY_GATE fails",
    "classification": "not a research result",
    "then": (
        "record the per-block coverage and stop. Do not run stage 1 on the blocks "
        "that did survive, do not open the holdout, and do not report P4 as negative "
        "evidence about derivatives positioning — nothing was measured"
    ),
}

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
    "improved_rule": dict(IMPROVED_RULE),
    "mean_delta_above": 0.0,
    "mean_delta_strict": True,
    "worst_fold_delta_at_least": -0.02,
    "screen_false_positive_rate_under_coin_null": NULL_PROBABILITY_THREE_OF_FOUR,
    "evaluated_on": "the exploratory blocks that passed BLOCK_AVAILABILITY_RULE",
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
        "constrained_by": (
            "STAGE_2_OUTCOMES and EVIDENCE_CLASSIFICATION, fixed before the run, "
            "including that an availability failure is not_evaluable and not negative."
        ),
    },
    {
        "choice": "how a funding CSV's columns are read",
        "constrained_by": (
            "FUNDING_CSV_COLUMN_POLICY, an allow-list fixed before any archive was "
            "opened. An unrecognised layout is a refusal, not an inference."
        ),
    },
    {
        "choice": "what counts as a block being available",
        "constrained_by": (
            "BLOCK_AVAILABILITY_RULE: 98% of rows surviving and no contiguous outage "
            "over 48 hours. Set from the mechanism before any probe."
        ),
    },
    {
        "choice": "whether an exactly-zero fold delta is an improvement",
        "constrained_by": "IMPROVED_RULE: it is not. delta > 0, strictly.",
    },
    {
        "choice": "what stage 2 fits, selects and reports on",
        "constrained_by": (
            "STAGE_2_TRAIN_ROWS, STAGE_2_SELECTION_ROWS and STAGE_2_EVALUATION_ROWS, "
            "inside the hashed preregistration rather than only in prose."
        ),
    },
    {
        "choice": "whether duplicate source rows may be normalised, and which",
        "constrained_by": (
            "OPEN_INTEREST_DUPLICATE_POLICY, amendment A1: exact duplicate rows in the "
            "daily metrics archive only, collapsed only when the complete published CSV "
            "row is identical. A conflicting duplicate is still a refusal, funding and "
            "perpetual klines are unchanged, and every removal is counted in provenance. "
            "Adopted after source inspection and before any P4 fit — the one rule here "
            "that is an amendment rather than an original commitment, which is why it "
            "carries its own supersedes and amendment_status."
        ),
    },
    {
        "choice": "where the funding archive's history is allowed to begin",
        "constrained_by": (
            "FUNDING_ARCHIVE_INCEPTION_POLICY, amendment A2: the Binance USD-M BTCUSDT "
            "monthly fundingRate archive's first protocol month is 2020-01, and the "
            "funding plan begins at max(generic requested start, that month). A month "
            "before it is outside the source rather than an internal gap, and is never "
            "synthesised, backfilled from REST, interpolated or sought elsewhere. Every "
            "expected month from 2020-01 onward stays mandatory and fail-closed. "
            "Adopted after the first full-acquisition attempt met the boundary and "
            "before any P4 fit — the second rule here that is an amendment rather than "
            "an original commitment, which is why it carries its own supersedes and "
            "amendment_status."
        ),
    },
    {
        "choice": "where the perpetual kline archive's history is allowed to begin",
        "constrained_by": (
            "PERPETUAL_KLINE_ARCHIVE_INCEPTION_POLICY, amendment A3: the Binance USD-M "
            "BTCUSDT monthly 1h kline archive's first protocol month is 2020-01, and "
            "the perpetual-price plan begins at max(generic requested start, that "
            "month). A month before it is outside the source rather than an internal "
            "gap, and is never synthesised, backfilled from REST, interpolated, or "
            "replaced by the spot candle history that is the basis denominator. Every "
            "expected month from 2020-01 onward stays mandatory and fail-closed, and "
            "basis features become available only where their perpetual input actually "
            "is. A separate object from A2's, measured separately on a separate "
            "archive. Adopted after the acquisition attempt that met the boundary and "
            "before any P4 fit — the third rule here that is an amendment rather than "
            "an original commitment, which is why it carries its own supersedes and "
            "amendment_status."
        ),
    },
    {
        "choice": "which open-interest source rows count as observations at all",
        "constrained_by": (
            "OPEN_INTEREST_OBSERVATION_VALIDITY_POLICY, amendment A4: a retained "
            "logical metrics row is a valid open-interest observation only when "
            "sum_open_interest and sum_open_interest_value are BOTH strictly positive. "
            "A row with either consumed field exactly zero stays a published source row "
            "and is counted, but is not a valid zero OI state and never enters the "
            "causal observation sequence — it is not interpolated, averaged, substituted "
            "from REST or repaired from a later row, and it does not make its day a "
            "missing day. A negative, non-finite or unparseable consumed value is a hard "
            "failure instead. Classification runs AFTER schema validation and A1's "
            "exact-duplicate collapse, the 1-hour staleness bound is unchanged, and the "
            "hours a long invalid run leaves without a valid observation simply leave "
            "the common sample universe for every arm. Adopted after the first full "
            "acquisition after A3 failed at pre-write hourly-table validation and the "
            "source was inspected directly, and before any P4 fit — the fourth rule "
            "here that is an amendment rather than an original commitment, which is why "
            "it carries its own supersedes and amendment_status."
        ),
    },
    {
        "choice": "who may spend the holdout, and how often",
        "constrained_by": (
            "HOLDOUT_SPEND_POLICY: once, by one checkpoint, gated on a frozen stage-1 "
            "pass, enforced by nn.p4_holdout, and retired if unspent."
        ),
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
        "funding_csv_column_policy": dict(FUNDING_CSV_COLUMN_POLICY),
        "open_interest_duplicate_policy": dict(OPEN_INTEREST_DUPLICATE_POLICY),
        "funding_archive_inception_policy": dict(FUNDING_ARCHIVE_INCEPTION_POLICY),
        "perpetual_kline_archive_inception_policy": dict(
            PERPETUAL_KLINE_ARCHIVE_INCEPTION_POLICY
        ),
        "open_interest_observation_validity_policy": dict(
            OPEN_INTEREST_OBSERVATION_VALIDITY_POLICY
        ),
        "max_staleness_hours": dict(MAX_STALENESS_HOURS),
        "exploratory_outer_blocks": [list(block) for block in EXPLORATORY_OUTER_BLOCKS],
        "exploratory_blocks_read_by": list(EXPLORATORY_BLOCKS_READ_BY),
        "holdout_rows": list(HOLDOUT_ROWS),
        "research_rows": RESEARCH_ROWS,
        "stage_1_max_row_exclusive": STAGE_1_MAX_ROW_EXCLUSIVE,
        "stage_2_train_rows": list(STAGE_2_TRAIN_ROWS),
        "stage_2_selection_rows": list(STAGE_2_SELECTION_ROWS),
        "stage_2_evaluation_rows": list(STAGE_2_EVALUATION_ROWS),
        "block_availability_rule": dict(BLOCK_AVAILABILITY_RULE),
        "availability_gate": dict(AVAILABILITY_GATE),
        "universe_conditions": [dict(entry) for entry in UNIVERSE_CONDITIONS],
        "holdout_spend_policy": dict(HOLDOUT_SPEND_POLICY),
        "improved_rule": dict(IMPROVED_RULE),
        "min_outer_trades": MIN_OUTER_TRADES,
        "stage_1_continuation": dict(STAGE_1_CONTINUATION),
        "stage_2_outcomes": [dict(outcome) for outcome in STAGE_2_OUTCOMES],
        "not_evaluable_outcome": dict(NOT_EVALUABLE_OUTCOME),
        "evidence_classification": dict(EVIDENCE_CLASSIFICATION),
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
