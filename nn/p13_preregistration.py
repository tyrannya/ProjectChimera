"""P13's design, committed as a design, before any P13 economic number exists.

P13 asks whether a *mechanically defined* delta-hedged BTC spot/perpetual carry
position can earn robust positive net returns from funding and basis after both
legs' realistic frictions — using only information available at each decision
instant.

**This is not P4 again.** P4 read funding, open interest and basis as *predictive
information* for a directional model and answered no. Here funding and basis are
the *payoff mechanism itself*, and the position is deliberately indifferent to
the next price move. A negative P4 says nothing about this question, and a
positive P13 would say nothing about P4.

**It is also not a directional model.** Nothing here fits anything. There is no
train/test split, no hyperparameter, no threshold selected from data, and no
model family. The strategy is a fixed arithmetic rule, which is why the design
can be frozen this completely before it runs.

The design is committed now for the usual reason: the only moment at which a
carry construction, a capital denominator, a cost model and a viability gate can
be fixed without being fitted to the answer is before the answer exists. §14 of
the governing task requires the preregistration commit to be pushed before the
first economic result, and §5 requires it to be written without inspecting one.

**Evidence ceiling.** Historical funding and basis were already publicly knowable
in 2026, when this was designed. Any result is
``EXPLORATORY / ADAPTIVE HISTORICAL STRUCTURAL FEASIBILITY EVIDENCE`` — never
pristine, prospective or confirmatory. See :data:`HINDSIGHT_DISCLOSURE`.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from nn.research_contract import load_contract

CHECKPOINT = "P13"

#: Why this identifier. `P9`-`P12` are reserved in `docs/research_roadmap.md`
#: for target/horizon, cross-asset, regime-conditioning and architecture work.
#: None has been designed, and reassigning one of them to a structural question
#: would silently overwrite historical numbering.
IDENTIFIER_RATIONALE = (
    "P9 (target/horizon), P10 (cross-asset/context), P11 (regime conditioning) and P12 "
    "(architecture/representation) are reserved in docs/research_roadmap.md and remain "
    "unused and undesigned. P13 is the first free identifier and is taken rather than "
    "renumbering a reserved axis."
)

QUESTION = (
    "Can a mechanically defined delta-hedged BTC spot/perpetual carry position — LONG "
    "Binance spot BTCUSDT against SHORT Binance USD-M BTCUSDT perpetual — earn robust "
    "positive net returns from funding and/or basis after BOTH legs' realistic trading "
    "frictions, hedge maintenance and adverse basis movement, using only information "
    "available at each decision instant?"
)

HYPOTHESIS_CLASS = "structural / non-directional. Not a directional BTC prediction model."

EVIDENCE_CEILING = "EXPLORATORY / ADAPTIVE HISTORICAL STRUCTURAL FEASIBILITY EVIDENCE"

HINDSIGHT_DISCLOSURE = (
    "Historical Binance funding and basis were already public knowledge in 2026, when this "
    "checkpoint was designed. The designer was not blind to the era. This screen is "
    "therefore exploratory and adaptive: it can falsify the structural mechanism, and it "
    "cannot confirm a future edge. A genuinely stronger confirmation requires a strategy "
    "frozen before wall-clock data that has not yet occurred, followed by sustained "
    "prospective paper observation. This ceiling is not lifted by a positive result."
)

NOT_P4 = (
    "P4 tested funding/open-interest/basis as INPUT FEATURES for directional prediction and "
    "was screened out at Stage 1. P13 tests funding/basis as the PAYOFF MECHANISM of a "
    "delta-hedged position. Different hypothesis, different accounting, different failure "
    "mode. Neither result transfers to the other."
)

# ---------------------------------------------------------------------------
# §1  Sources
# ---------------------------------------------------------------------------

#: Same-venue by construction. Introducing a second exchange would add basis and
#: settlement risk this first structural screen is not trying to measure, and
#: `docs/current_development_plan.md` standing constraint 20 makes same-venue the
#: default. Cross-venue carry is a different, later checkpoint.
VENUE_POLICY = (
    "Binance only, for both legs. No cross-exchange construction in this screen. A "
    "cross-venue carry is a separate checkpoint with its own preregistration."
)

ARCHIVE_HOST = "https://data.binance.vision"

#: Where every source fact below was established, independently of this
#: repository's own P4 constants. `binance/binance-public-data` is Binance's
#: official public-data repository: the path grammar, the object and checksum
#: naming, the checksum algorithm, the archive column layouts and the list of
#: archive revisions are read from Binance's own published code and README rather
#: than inferred from P4 or from a third-party description. P4's amendments A2
#: and A3 corroborate the 2020-01 inception; they are not the primary evidence
#: for it.
FIRST_PARTY_SOURCE_EVIDENCE = {
    "repository": "https://github.com/binance/binance-public-data",
    "commit": "5c7f3197591c0d54d85dc43066226bc4c671d47a",
    "read_offline_at": "a shallow clone of that commit, read locally",
    "path_grammar": {
        "evidence": "python/utility.py lines 105-113, function get_path",
        "spot": "data/spot/{time_period}/{market_data_type}/{SYMBOL}/{interval}/",
        "futures": (
            "data/futures/{um|cm}/{time_period}/{market_data_type}/{SYMBOL}/{interval}/"
        ),
        "no_interval_segment": (
            "get_path omits the {interval} segment when interval is None, which is the shape "
            "the fundingRate archive uses"
        ),
    },
    "base_url": "python/enums.py: BASE_URL = 'https://data.binance.vision/'",
    "object_naming": {
        "evidence": (
            "python/download-kline.py line 45 and "
            "download-futures-markPriceKlines.py line 48"
        ),
        "monthly": "{SYMBOL}-{interval}-{YYYY}-{MM}.zip",
        "daily": "{SYMBOL}-{interval}-{YYYY-MM-DD}.zip",
    },
    "checksum": {
        "evidence": "README.md 'CHECKSUM' section; download-*.py checksum_file_name lines",
        "object": "{the .zip name}.CHECKSUM, published beside every archive",
        "algorithm": "sha256, verified by `sha256sum -c <file>.zip.CHECKSUM`",
    },
    "mark_price_archive_exists": {
        "evidence": (
            "python/download-futures-markPriceKlines.py is an official Binance downloader for "
            "market_data_type 'markPriceKlines' under data/futures/um, in both monthly and "
            "daily granularity, each with a .CHECKSUM companion"
        ),
        "consequence": (
            "the mark-price source is established as a real published archive family rather "
            "than an assumption. indexPriceKlines and premiumIndexKlines exist alongside it "
            "and are deliberately NOT acquired: neither is required by this design."
        ),
    },
    "funding_archive_note": (
        "binance-public-data ships no download-futures-fundingRate.py, so the fundingRate "
        "archive is NOT covered by an official downloader script. Its existence rests on "
        "nn.p4_preregistration.FUNDING_ARCHIVE_INCEPTION_POLICY, which records real HTTP "
        "checks against data/futures/um/monthly/fundingRate/BTCUSDT returning 200 for "
        "2020-01. "
        "This is stated plainly rather than glossed: it is the one source whose path is "
        "corroborated by measurement rather than by Binance's published code."
    ),
    "period_start_date": (
        "python/enums.py sets PERIOD_START_DATE = '2020-01-01', Binance's own default archive "
        "start, independently consistent with the measured 2020-01 inception of both the "
        "funding and perpetual-kline archives"
    ),
    "monthly_publication_lag": (
        "README.md: 'new daily data becoming available the next day and new monthly data at "
        "the first monday of the month'. Availability semantics, not revision semantics."
    ),
}

#: **Corrects an assumption inherited from P4.** Binance's own README states that
#: "Archived files may be updated at a later date as a result of recently
#: discovered issues", and publishes an exhaustive changelog of such updates —
#: including a 2022-08-08 KLINE update, which is directly in scope for this
#: checkpoint's kline sources. A design that asserted archives are never revised
#: would be asserting something the publisher contradicts.
ARCHIVE_REVISION_POLICY = {
    "archives_may_be_revised": True,
    "evidence": (
        "binance-public-data README.md, 'Updates' section: an exhaustive table of archive "
        "revisions, listing 2022-08-08 kline_updates ('Fixed inconsistent data') and "
        "2022-04-21 aggregate_trade_updates"
    ),
    "what_this_changes": (
        "a realised funding settlement and a closed kline are final AS MARKET FACTS, but the "
        "ARCHIVE OBJECT carrying them is not guaranteed immutable. Those are different "
        "claims and this design keeps them apart."
    ),
    "handling": (
        "the acquisition records, per object, both the published .CHECKSUM digest and an "
        "independently recomputed sha256 of the bytes actually received, plus the byte size. "
        "A later re-fetch that disagrees is a REVISION EVENT: it is reported and disclosed, "
        "never silently accepted and never silently rejected. The frozen manifest therefore "
        "pins the exact bytes the result was computed from, which is what reproducibility "
        "requires when the upstream is mutable."
    ),
    "known_revisions_in_scope": (
        "the 2022-08-08 kline update falls inside this checkpoint's 2020-2025 span. The "
        "acquisition records which published changelog entries overlap the acquired span so "
        "an auditor can see them without re-deriving the list."
    ),
}

#: **A live dimensional trap, and it falls inside this checkpoint's span.**
TIMESTAMP_UNIT_POLICY = {
    "spot_unit_change": (
        "binance-public-data README.md: 'The timestamp for SPOT Data from January 1st 2025 "
        "onwards will be in microseconds.' Before that date spot timestamps are milliseconds."
    ),
    "why_it_matters_here": (
        "this checkpoint spans 2020-01-01 to 2025-05-19, so it straddles the change. Parsing "
        "2025 spot rows as milliseconds would place them about fifty thousand years in the "
        "future and silently destroy every alignment; parsing 2020 rows as microseconds would "
        "place them in 1970. The futures archives show no equivalent note, so spot and "
        "perpetual sources may carry DIFFERENT units for the same calendar month — a "
        "cross-source hazard, not merely a per-source one."
    ),
    "rule": (
        "the unit is never assumed and never hard-coded per source. Every archive's unit is "
        "resolved by nn.trade_aggregates.resolve_epoch_unit, which requires the parsed "
        "instants to fall inside the archive's OWN calendar period under exactly one "
        "supported unit, and refuses otherwise. The resolved unit is recorded per object in "
        "the manifest so the choice is auditable rather than implicit."
    ),
    "fail_closed": (
        "an archive whose unit cannot be resolved unambiguously is a refusal, not a guess."
    ),
}

#: Every source is an official Binance published archive. REST endpoints are
#: **not** a fallback for history: `docs/p4_preregistration.md` §3 already
#: established that a row sourced from REST is not a row of the preregistered
#: historical source, and substituting one would report a universe the run did
#: not have.
DATA_SOURCES: tuple[dict[str, Any], ...] = (
    {
        "field": "spot_price",
        "role": "the LONG leg's execution price, and the basis denominator",
        "venue": "binance",
        "market_type": "spot",
        "symbol": "BTCUSDT",
        "archive": (
            "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h/"
            "BTCUSDT-1h-{year}-{month}.zip"
        ),
        "committed_alternative": (
            "data/research/btc_usdt_1h_gen1_raw_pre_styx.parquet, the already-frozen and "
            "hash-verified Binance spot BTCUSDT 1h snapshot described by "
            "data/research/btc_usdt_1h_gen1_snapshot_manifest.json. Preferred over a "
            "re-acquisition because its provenance is already committed."
        ),
        "committed_alternative_is_pre_STYX_not_pre_boundary": (
            "STATED PRECISELY, because the filename says pre_styx and it means it. Its true "
            "span is 2020-01-01T00:00:00+00:00 .. 2025-08-27T22:00:00+00:00, 49,551 rows, and "
            "it therefore CONTAINS roughly 2,415 rows at or after the P13 research boundary — "
            "the retired P4-HOLD region. It is not a pre-boundary file and must never be "
            "described as one. Committed digests: sha256 "
            "04f9748e09adfd4cb38dd84898a43aa7eb5de1ff9f6aa938e3e1302b20cefeee, semantic_hash "
            "c62de8d92a1eb80397d17ff89e480005d381525ec14c5f6519efddbc2cbe86ed."
        ),
        "truncating_read_carve_out": (
            "for this one pre-existing committed snapshot the boundary treatment is a "
            "TRUNCATING READ applied at load, before any P13 computation touches a row: rows "
            "at or after RESEARCH_BOUNDARY_EXCLUSIVE are dropped by the loader, the loader "
            "asserts the maximum surviving instant is strictly below the boundary, and the "
            "surviving row count and maximum instant are recorded in the evidence manifest. "
            "This is the ONE place DATA_BOUNDARY's 'a row at or after the boundary is a "
            "refusal, not a filter' is relaxed, it is relaxed only for a file whose extra "
            "rows "
            "were committed long before P13 existed, and it is relaxed loudly rather than "
            "silently. A freshly acquired archive is still refused rather than truncated."
        ),
        "timestamp_column": "date",
        "timestamp_semantics": "the candle OPEN; the candle is complete at open + 1h",
        "timestamp_unit": (
            "MILLISECONDS before 2025-01-01 and MICROSECONDS from 2025-01-01, per Binance's "
            "own README. Resolved per object, never assumed. See TIMESTAMP_UNIT_POLICY. "
            "Normalised to int64 UTC nanoseconds downstream."
        ),
        "revision_policy": (
            "the closed candle is final as a market fact; the ARCHIVE OBJECT carrying it is "
            "not guaranteed immutable. See ARCHIVE_REVISION_POLICY."
        ),
        "object": "{SYMBOL}-{interval}-{YYYY}-{MM}.zip",
        "checksum_object": "{SYMBOL}-{interval}-{YYYY}-{MM}.zip.CHECKSUM, sha256",
    },
    {
        "field": "perpetual_price",
        "role": "the SHORT leg's execution price, and the basis numerator",
        "venue": "binance",
        "market_type": "usd-m perpetual futures",
        "symbol": "BTCUSDT",
        "archive": (
            "https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/1h/"
            "BTCUSDT-1h-{year}-{month}.zip"
        ),
        "timestamp_column": "open_time",
        "timestamp_semantics": "the candle OPEN; the candle is complete at open + 1h",
        "timestamp_unit": (
            "epoch milliseconds in the futures archives, with no microsecond change "
            "documented for futures. Still resolved per object rather than assumed, because "
            "spot and futures may differ for the same month. See TIMESTAMP_UNIT_POLICY."
        ),
        "first_published_month": "2020-01",
        "first_published_month_provenance": (
            "corroborated from two independent directions. Binance's own enums.py sets "
            "PERIOD_START_DATE = '2020-01-01'; and nn.p4_preregistration."
            "PERPETUAL_KLINE_ARCHIVE_INCEPTION_POLICY (amendment A3) records real HTTP checks "
            "returning 404 for 2019-12 and 200 for 2020-01 on this exact archive, adopted "
            "before any P4 result existed. The acquisition probe re-establishes it rather "
            "than trusting either."
        ),
        "revision_policy": (
            "the closed candle is final as a market fact; the archive object is not "
            "guaranteed immutable, and a 2022-08-08 kline revision is published. See "
            "ARCHIVE_REVISION_POLICY."
        ),
        "object": "{SYMBOL}-{interval}-{YYYY}-{MM}.zip",
        "checksum_object": "{SYMBOL}-{interval}-{YYYY}-{MM}.zip.CHECKSUM, sha256",
    },
    {
        "field": "funding_settlement",
        "role": "the carry payoff itself",
        "venue": "binance",
        "market_type": "usd-m perpetual futures",
        "symbol": "BTCUSDT",
        "archive": (
            "https://data.binance.vision/data/futures/um/monthly/fundingRate/BTCUSDT/"
            "BTCUSDT-fundingRate-{year}-{month}.zip"
        ),
        "timestamp_column": "per FUNDING_CSV_COLUMN_POLICY (fundingTime or calc_time)",
        "timestamp_semantics": (
            "the SETTLEMENT instant. The realised rate is final AT that instant and is not "
            "knowable before it."
        ),
        "timestamp_unit": "epoch milliseconds in the archive, normalised to int64 UTC ns",
        "first_published_month": "2020-01",
        "first_published_month_provenance": (
            "measured, not assumed: nn.p4_preregistration.FUNDING_ARCHIVE_INCEPTION_POLICY "
            "(amendment A2) records real HTTP checks returning 404 for 2019-09 through "
            "2019-12 and 200 for 2020-01, adopted before any P4 result existed"
        ),
        "revision_policy": (
            "a realised settlement is final as a market fact; the archive object is not "
            "guaranteed immutable. See ARCHIVE_REVISION_POLICY."
        ),
        "object": "{SYMBOL}-fundingRate-{YYYY}-{MM}.zip",
        "checksum_object": "{SYMBOL}-fundingRate-{YYYY}-{MM}.zip.CHECKSUM, sha256",
        "path_shape_note": (
            "no {interval} path segment, matching get_path's interval-is-None branch in "
            "Binance's own utility.py"
        ),
        "interval_policy": (
            "the settlement CADENCE IS NOT ASSUMED. Binance has changed the funding "
            "settlement frequency of USD-M perpetuals (8h to 4h, and to 1h while a rate sits "
            "at its cap). Every row of the archive is one settlement event and is treated as "
            "one; where the layout carries funding_interval_hours it is recorded per event. "
            "Nothing in this design multiplies a rate by an assumed number of settlements "
            "per day."
        ),
    },
    {
        "field": "mark_price",
        "role": "the notional base on which Binance charges funding",
        "venue": "binance",
        "market_type": "usd-m perpetual futures",
        "symbol": "BTCUSDT",
        "archive": (
            "https://data.binance.vision/data/futures/um/monthly/markPriceKlines/BTCUSDT/1h/"
            "BTCUSDT-1h-{year}-{month}.zip"
        ),
        "timestamp_column": "open_time",
        "timestamp_semantics": "the candle OPEN; the candle is complete at open + 1h",
        "timestamp_unit": (
            "epoch milliseconds in the futures archives; resolved per object rather than "
            "assumed. See TIMESTAMP_UNIT_POLICY."
        ),
        "availability": (
            "ESTABLISHED FROM FIRST-PARTY CODE, not assumed: Binance publishes an official "
            "downloader, python/download-futures-markPriceKlines.py, for market_data_type "
            "'markPriceKlines' under data/futures/um in monthly and daily granularity, each "
            "with a .CHECKSUM companion. The FIRST MONTH published for BTCUSDT is still "
            "unmeasured — no HTTP request has been made from this session — so the "
            "acquisition probe establishes the span, and MARK_PRICE_FALLBACK covers the one "
            "substitution allowed if the family proves unpublished for BTCUSDT."
        ),
        "object": "{SYMBOL}-{interval}-{YYYY}-{MM}.zip",
        "checksum_object": "{SYMBOL}-{interval}-{YYYY}-{MM}.zip.CHECKSUM, sha256",
        "siblings_deliberately_not_acquired": (
            "indexPriceKlines and premiumIndexKlines exist in the same family and are NOT "
            "acquired: neither is required by this design, and acquiring a source a design "
            "does not use invites it being used later."
        ),
    },
)

#: Fixed **now**, and triggered by an availability fact rather than by anything
#: economic. Binance charges funding on the position's notional at the mark price
#: when the funding countdown reaches zero. Mark tracks the *index* (a spot
#: composite), not the perpetual's last trade, so substituting the perpetual
#: close would be wrong by exactly the basis — the quantity under study. Spot is
#: the closer proxy, and the residual error is second order.
MARK_PRICE_FALLBACK = {
    "trigger": (
        "PER ARCHIVE OBJECT, not all-or-nothing: any month for which the markPriceKlines "
        "object is unpublished falls back for that month alone, established by the "
        "acquisition probe. The design admits the first published month for BTCUSDT is "
        "unmeasured, so partial availability is the likely case and an all-or-nothing rule "
        "would have had no defined behaviour for it."
    ),
    "substitution": "the Binance spot BTCUSDT close is used as the funding notional base",
    "never_triggered_by": (
        "any economic observation. Not by a disappointing return, not by a stress case, not "
        "by anything a run computes. Availability only."
    ),
    "reporting_granularity": (
        "the substitution flag is reported per block AND as a count of settlements that used "
        "the substituted base, so a reader can see exactly how much of the funding total "
        "rests on the approximation rather than on the venue's own notional price."
    ),
    "error_bound": (
        "per settlement the substitution mis-states funding by basis_fraction x rate of "
        "notional, a product of two small numbers. It is reported as an explicit "
        "substitution flag on every artifact and in every summary, never silently."
    ),
    "forbidden_alternative": (
        "the perpetual close is NEVER used as the funding notional base: it is wrong by the "
        "basis, which is the quantity this checkpoint measures"
    ),
}

#: Reused verbatim from P4 rather than restated, so a layout rule cannot drift
#: between two checkpoints reading the same archive.
FUNDING_COLUMN_POLICY_SOURCE = "nn.p4_preregistration.FUNDING_CSV_COLUMN_POLICY"

SOURCE_FREEZE_FIELDS: tuple[str, ...] = (
    "official source archive family and exact object path",
    "instrument and market type",
    "symbol",
    "timestamp column, semantics and unit",
    "publication and availability semantics",
    "object names",
    "byte sizes",
    "published Binance .CHECKSUM digest",
    "independently recomputed sha256 of the fetched bytes",
    "sha256 of the extracted member",
    "acquisition cutoff instant",
    "availability span actually observed",
    "duplicate rows detected",
    "gaps detected",
    "timezone",
    "fail-closed rule applied",
)

# ---------------------------------------------------------------------------
# §2  Data boundary
# ---------------------------------------------------------------------------

#: The retired P4-HOLD region begins here. `docs/current_development_plan.md`
#: standing constraint 12 makes it the research-visible cutoff, and it is
#: stricter than the Styx seal, so it is the operative boundary.
RESEARCH_BOUNDARY_EXCLUSIVE = "2025-05-19T08:00:00+00:00"

#: **Resolved from the committed contract, never restated.**
#: ``tests/test_research_contracts.py::test_the_source_carries_no_second_copy_of_the_anchor``
#: forbids a second literal copy of the sealed anchor anywhere under ``nn/`` or
#: ``chimera/``, because a copy is a thing that can silently disagree with the
#: contract it claims to quote. ``nn/p6_preregistration.py`` resolves it the same
#: way and for the same reason.
STYX_SEALED_INSTANT = load_contract("btc-usdt-1h-gen1").sealed_test_start.isoformat()

DATA_BOUNDARY = {
    "span_start_inclusive": "2020-01-01T00:00:00+00:00",
    "span_end_exclusive": RESEARCH_BOUNDARY_EXCLUSIVE,
    "start_rationale": (
        "the first month published by BOTH the funding archive and the perpetual kline "
        "archive, as measured by P4 amendments A2 and A3, and also the first instant of the "
        "committed spot snapshot. Not chosen; it is where the sources begin."
    ),
    "end_rationale": (
        "the first instant of the retired P4-HOLD region. No P13 observation of any kind may "
        "be at or after it."
    ),
    "p4_hold": "NOT READ. Retired, checkpoint null, and not a spare holdout.",
    "styx": "NOT READ. Sealed.",
    "no_manufactured_holdout": (
        "P13 does not carve a 'fresh holdout' out of history that was already knowable in "
        "2026. Splitting already-public data does not manufacture independence."
    ),
    "enforcement": (
        "fail closed. A row at or after the boundary is a refusal, not a filter: the "
        "acquisition and the evaluator both assert it, and the frozen evidence records the "
        "maximum observed instant so an auditor can check it without trusting either."
    ),
    "the_boundary_straddling_month": (
        "the boundary falls mid-month, so the FINAL archive month (2025-05) necessarily "
        "contains rows on both sides of it. Named explicitly because the fail-closed rule "
        "above would otherwise forbid the very month the declared span requires.\n"
        "The rule is about research OBSERVATIONS, not about bytes on disk: that one month's "
        "archive is fetched whole — a partial object cannot be checksum-verified against "
        "Binance's published digest — and is TRUNCATED AT LOAD before any P13 computation "
        "touches a row. The loader asserts the maximum surviving instant is strictly below "
        "the boundary and records it, the count of dropped rows is recorded, and the digest "
        "recorded in the manifest is that of the WHOLE published object so the acquisition "
        "stays verifiable against Binance.\n"
        "This carve-out covers exactly two things: the one boundary-straddling archive month, "
        "and the pre-existing committed spot snapshot. Every other archive month lies wholly "
        "before the boundary and is refused rather than truncated if it does not."
    ),
}

# ---------------------------------------------------------------------------
# §3  Funding causality — load-bearing
# ---------------------------------------------------------------------------

FUNDING_CAUSALITY = {
    "rule": (
        "a funding settlement is knowable only AT its settlement instant. No decision taken "
        "at time t may use any settlement whose instant is greater than t."
    ),
    "predicted_rate": (
        "NEVER READ. Binance continuously publishes a predicted funding rate; it is a "
        "forecast, it is revised, and reading it would put a later number into an earlier "
        "row. Only realised settlements from the archive are used."
    ),
    "does_the_strategy_use_funding_as_a_signal": (
        "NO. The frozen strategy is always-on within a block and takes no entry, exit or "
        "sizing decision from any funding observation. This removes the entire class of "
        "funding-lookahead error from the decision path by construction rather than by "
        "checking for it. Funding enters only as a cash flow, applied at its own settlement "
        "instant while a position is already open."
    ),
    "why_this_is_stronger_than_a_filter": (
        "a causal entry filter on observed funding would be defensible, but it would be a "
        "parameter, and a parameter chosen before results is still a parameter a later reader "
        "must take on trust. The always-on rule removes every funding-derived ENTRY, EXIT and "
        "SIZING parameter from the design."
    ),
    "the_one_parameter_that_remains": (
        "HONESTLY STATED, because the sentence it replaces ('an always-on rule has none to "
        "trust') was false. One parameter survives and cannot be removed by any construction: "
        "the LEG DIRECTION. Under the imported sign convention a SHORT perpetual receives "
        "when "
        "funding is positive and pays when it is negative, so choosing LONG spot / SHORT "
        "perpetual rather than the reverse fixes the SIGN of the entire funding payoff. That "
        "choice is hindsight-informed: perpetual funding on major crypto pairs has been "
        "predominantly positive over the historical era being screened, and the designer knew "
        "that in 2026. It is a binary, mechanically motivated choice — retail leverage demand "
        "is structurally long, which is the mechanism a carry harvests — but it is not a "
        "choice made in ignorance, and the reader is entitled to weigh the result knowing it."
    ),
    "the_instrument_for_that_disclosure": (
        "funding received and funding paid are reported as two separate magnitudes per block, "
        "never netted, precisely so a reader can see how much of any positive result came "
        "from "
        "the direction being right about the era rather than from the carry paying for its "
        "frictions."
    ),
    "positive_controls_required": (
        "the suite must FAIL when funding is shifted one settlement into the future, when a "
        "settlement is made visible before its instant, and when a settlement is applied "
        "twice. A suite that passes under those injections is not testing causality."
    ),
}

# ---------------------------------------------------------------------------
# §4  Position and capital contract
# ---------------------------------------------------------------------------

CONSTRUCTION = (
    "LONG Binance spot BTCUSDT, and SHORT Binance USD-M BTCUSDT perpetual, in EQUAL BTC "
    "quantity. Equal quantity is what makes the position delta neutral: net BTC exposure is "
    "identically zero for the life of the position, not approximately zero — under S0, S1 "
    "and S3. S2 deliberately leaves the position directionally exposed for one bar at each "
    "end, so its zero-delta window is the perpetual leg's own open-to-close."
)

#: **Which number in a row is a fill, and which is only a mark.** Both kline
#: sources are timestamped by candle OPEN, so at grid instant ``t`` the only
#: price in that row already knowable at ``t`` is the open. Filling at the close
#: of the candle labelled ``t`` would execute at a price revealed an hour later —
#: at BOTH ends of the position, which is 100% of the price PnL, since the hedged
#: position's price PnL is exactly ``Q x (basis_in - basis_out)``. Leaving the
#: field unstated would leave that choice to whoever writes the executor.
EXECUTION_PRICE_POLICY = {
    "fill_price": (
        "the OPEN of the candle whose open_time is the decision instant t, on BOTH legs, for "
        "the entry, the exit and any liquidation-forced close. It is the only field in that "
        "row knowable at t."
    ),
    "close_is_never_a_fill": (
        "the candle CLOSE is used only for end-of-hour marking: the basis series, "
        "mark-to-market, equity and the funding notional. It is never an execution price."
    ),
    "high_is_only_a_liquidation_touch": (
        "the hourly HIGH of the mark series is used only as the conservative intra-bar "
        "liquidation touch, never as a fill and never as a mark."
    ),
    "why_stated": (
        "an unstated fill field is a researcher degree of freedom wearing the clothes of an "
        "implementation detail, and here it would be a one-hour lookahead at the two instants "
        "that determine the entire price result."
    ),
}

CAPITAL_CONTRACT = {
    "total_starting_capital": "1000000",
    "capital_units": "USDT",
    "why_a_real_scale_and_not_1.0": (
        "a denominator of 1.0 USDT is arithmetically incompatible with the venue filters this "
        "same design invokes. Binance's committed BTCUSDT perpetual filters are step_size "
        "0.001 BTC, min_quantity 0.001 and min_notional 100 USDT (chimera/futures/venue.py). "
        "At a 2020 spot price near 7,195 USDT, half of 1.0 USDT buys about 0.00007 BTC, which "
        "floors to ZERO at a 0.001 step and is far below min_notional — so a design "
        "denominated in 1.0 would open no block at all and terminate INVALID for a reason "
        "having nothing to do with carry. 1,000,000 USDT is declared instead: large enough "
        "that lot granularity is negligible at every price in the span, and an ordinary size "
        "for a carry book."
    ),
    "returns_are_fractions": (
        "every block and aggregate return is net PnL divided by total_starting_capital and is "
        "reported as a FRACTION, so the gate's thresholds are scale-free and the 1.0-style "
        "denominator semantics are unchanged."
    ),
    "scale_is_frozen_and_is_not_a_parameter": (
        "1,000,000 USDT is fixed here and never tuned. Any scale at which lot granularity is "
        "negligible yields the same fractional returns to within rounding, and the accounting "
        "controls assert exactly that, so the choice cannot move a result."
    ),
    "slippage_at_this_scale": (
        "the frozen 5 bps per-leg slippage is charged against roughly 500,000 USDT per leg on "
        "BTCUSDT, the deepest crypto pair. Market impact beyond that assumption is NOT "
        "modelled, and is recorded as a limitation rather than argued away."
    ),
    "spot_allocation": "500000",
    "perp_margin_allocation": "500000",
    "hedge_ratio": "1.0 BTC of short perpetual per 1.0 BTC of long spot",
    "quantity_rule": (
        "Q is the LARGEST equal quantity BOTH allocations can fund, rounded DOWN to the "
        "venue step size:\n"
        "    Q_spot_bound = spot_allocation / (spot_price  x (1 + spot_fee + spot_slippage))\n"
        "    Q_perp_bound = perp_allocation / (perp_price  x (1 + perp_fee + perp_slippage))\n"
        "    Q            = step_floor(min(Q_spot_bound, Q_perp_bound))\n"
        "step_floor uses the COARSER of the spot and perpetual step sizes, so one Q is legal "
        "on both instruments at once. A per-leg step would produce two different quantities "
        "and silently break the delta neutrality the whole construction rests on.\n"
        "The minimum over both legs, and not the spot leg alone. Sizing from the spot "
        "allocation only would make the perpetual leg's cash requirement exceed its "
        "allocation whenever the perpetual traded more than about 5 bps above spot — that "
        "is, in precisely the contango regimes a carry position exists to harvest — and the "
        "margin rule would then refuse to open in a way correlated with the phenomenon under "
        "study. Taking the minimum removes that bias, keeps the two legs at exactly equal "
        "quantity, and guarantees gross notional never exceeds total capital. Rounding down, "
        "never up, so the position never exceeds the capital that authorised it."
    ),
    "unused_allocation": (
        "whichever leg does not bind leaves a residual that stays as idle quote cash at zero "
        "yield. It is never redeployed to enlarge either leg."
    ),
    "gross_leverage": (
        "spot notional + perpetual notional <= 1.0 x total capital by construction, so gross "
        "leverage never exceeds 1x and net BTC exposure is zero."
    ),
    "residual_delta": (
        "exactly zero in BTC terms, by construction, for the whole holding period under S0, "
        "S1 and S3. Under S2 the legs are opened and closed one bar apart by design, so delta "
        "is zero only BETWEEN the perpetual leg's open and close, and the zero-delta control "
        "is scoped accordingly — the same trap identity_scope closes for the basis "
        "identity. The two "
        "legs' VALUES still differ, and that difference is the basis — which is the quantity "
        "under study, not an error."
    ),
    "leverage": "exactly 1x. Never more, under any stress case or any result.",
    "margin_mode": (
        "PORTFOLIO (cross) collateral across the two legs — see "
        "MARGIN_AND_LIQUIDATION.primary_model. The 0.5/0.5 split is a SIZING DEVICE, not a "
        "wallet segregation: it bounds each leg's notional so gross leverage stays at 1x. The "
        "strictly isolated case is stress S4 and sits outside the gate."
    ),
    "evaluator_flag": (
        "the primary run passes isolated=False to nn.p13_carry.evaluate_block; S4 passes "
        "isolated=True. Named here so the model is fixed by the preregistration rather than "
        "chosen by whoever calls the module."
    ),
    "margin_sufficiency_rule": (
        "an ASSERTION, not a branch. Under the min() sizing rule the perpetual's initial "
        "margin at 1x satisfies Q x perp_entry_price <= perp_allocation / (1 + perp_fee + "
        "perp_slippage) < perp_allocation for every basis, so the requirement cannot fail. "
        "Reaching it is therefore a BUG rather than a market condition, and it raises rather "
        "than silently shrinking the hedge — exactly as nn.p13_carry treats its analogous "
        "free-cash check. It is never rescued by borrowing across legs, by raising leverage, "
        "or by shrinking the hedge below Q."
    ),
    "idle_capital": (
        "any part of total capital not consumed by the two legs' notional and their entry "
        "frictions sits as idle quote cash earning EXACTLY ZERO. No interest, no yield, no "
        "sweep. Assuming a return on idle collateral would be inventing income the venue does "
        "not pay."
    ),
    "equity_definition": (
        "equity_t = free_cash_t + Q x spot_price_t + perp_margin + perp_unrealised_pnl_t\n"
        "  perp_unrealised_pnl_t = (perp_entry_price - perp_price_t) x Q   [SHORT]\n"
        "and free_cash evolves — it is not a residual computed twice:\n"
        "  free_cash starts at total_starting_capital;\n"
        "  minus Q x spot_entry_price x (1 + spot_fee + spot_slippage)   at entry;\n"
        "  minus perp_margin, and minus Q x perp_entry_price x (perp_fee + perp_slippage);\n"
        "  plus or minus each funding cash flow as it settles;\n"
        "  plus the exit proceeds of both legs, net of their exit fees and slippage.\n"
        "Fees, slippage and funding therefore reach equity EXACTLY ONCE, through free_cash. "
        "They are also accumulated separately for REPORTING, and those reporting "
        "accumulators are never added to equity a second time. Stating this explicitly "
        "because an equity line written as 'cash + legs + funding - fees' double-counts every "
        "term that cash already reflects."
    ),
    "equity_invariant": (
        "at entry, equity equals total capital minus the entry fees and slippage of both "
        "legs, and nothing else. The accounting controls assert exactly that."
    ),
    "return_denominator": (
        "TOTAL committed capital, 1.0. Never one leg's capital. A return quoted on the spot "
        "leg alone would double the apparent performance of a position that consumed both "
        "allocations."
    ),
}

REBALANCE_POLICY = {
    "policy": "none within a block",
    "why": (
        "the hedge is quantity-matched, so net BTC delta is identically zero for the life of "
        "the position and NOTHING about the hedge degrades as price moves. No rebalance is "
        "mechanically required. This is a property of the construction, not an omission."
    ),
    "consequence": (
        "rebalance count and rebalance cost are structurally zero and are reported as zero "
        "with this reason attached, rather than silently absent."
    ),
    "not_a_tuning_surface": (
        "a rebalance cadence is not introduced later to improve a disappointing result. "
        "Doing so would be a different construction and a different checkpoint."
    ),
}

POSITION_LIFECYCLE = {
    "per_block": "exactly one position: one open, one close",
    "open_instant": (
        "the first hourly grid instant inside the block at which spot, perpetual and (where "
        "used) mark observations are all present and valid"
    ),
    "close_instant": (
        "the INTENDED close is the last hourly grid instant of the block, strictly before the "
        "block end and strictly before the research boundary. If that instant is not valid, "
        "the position closes at the FIRST valid instant AT OR AFTER it — an operator who "
        "cannot trade keeps holding — bounded by the BLOCK END as well as by the research "
        "boundary, so a search for a valid exit can never run into the next block and "
        "overlap the next position. If no valid instant exists at or before the block's last "
        "hour, the block is reported UNCLOSED with its reason rather than back-dated.\n"
        "Stated this way deliberately: 'the LAST instant at which observations are valid' "
        "would require scanning forward to the end of the block to discover which instant "
        "that is, so an operator standing at t could not implement it. It is the one "
        "lookahead the always-on rule does not remove, and it is not benign — the exit "
        "instant sets basis_out, which is half of the entire price PnL under the "
        "telescoping identity."
    ),
    "validity_definition": (
        "an instant is VALID when the row is present in every required source, every "
        "preregistered field is present, every price is strictly positive, and no duplicate "
        "row makes the instant ambiguous. Anything else is invalid and fails closed."
    ),
    "carry_across_blocks": (
        "none. Each block opens and closes its own position, so blocks are not linked by an "
        "inherited inventory and a bad block cannot be hidden inside a neighbour."
    ),
    "opens_closes_rebalances_per_block": "1 open, 1 close, 0 rebalances",
}

#: **Frozen here because they are decision-relevant and would otherwise be
#: chosen after the hash.** ``nn.p13_carry.Venue`` takes each of these as an
#: injected field, and the maintenance margin rate sets S4's liquidation
#: threshold while the step size and minimum notional decide whether a block can
#: open at all. ``chimera/futures/venue.py`` refuses metadata missing them
#: precisely because "there is no default that is safer than refusing".
VENUE_CONSTRAINTS = {
    "perpetual": {
        "symbol": "BTCUSDT (USD-M perpetual)",
        "step_size": "0.001",
        "min_quantity": "0.001",
        "min_notional": "100",
        "tier_1_maintenance_margin_rate": "0.004",
        "taker_fee_rate": "0.0005",
        "provenance": (
            "chimera/futures/venue.py, the repository's committed Binance USD-M BTCUSDT "
            "filter table, already used by the futures dry-run validation"
        ),
    },
    "spot": {
        "symbol": "BTCUSDT (spot)",
        "step_size": "0.00001",
        "min_notional": "10",
        "provenance": (
            "Binance's published spot LOT_SIZE and NOTIONAL filters for BTCUSDT. Declared "
            "here rather than fetched, and immaterial at this capital scale: the binding "
            "step is the COARSER of the two, which is the perpetual's 0.001."
        ),
    },
    "effective_step_size": "0.001, the coarser of the two, so one Q is legal on both legs",
    "era_limitation": (
        "STATED, NOT GLOSSED: this is a CURRENT-ERA filter table applied uniformly across "
        "2020-2025. Binance's leverage brackets, maintenance margin rates and lot filters "
        "have changed over that span, and no historical bracket archive is preregistered "
        "here. The consequence is confined: at 1x with a portfolio margin model the "
        "maintenance requirement is a rounding term against equity, and at this capital scale "
        "the lot filters never bind. It would matter to S4's liquidation threshold, and S4 is "
        "a diagnostic outside the gate."
    ),
}

# ---------------------------------------------------------------------------
# §5  Cost model — frozen, and never lowered
# ---------------------------------------------------------------------------

#: Binance published VIP-0 taker rates with no BNB discount and no maker
#: assumption. Taker on both legs because a carry position that must be *in* the
#: market cannot assume its passive order filled.
COST_MODEL = {
    "spot_entry_fee_rate": "0.001",
    "spot_exit_fee_rate": "0.001",
    "perp_entry_fee_rate": "0.0005",
    "perp_exit_fee_rate": "0.0005",
    "spot_slippage_rate": "0.0005",
    "perp_slippage_rate": "0.0005",
    "fee_basis": "the fill's NOTIONAL, quantity x price. Never quantity alone.",
    "slippage_basis": "the fill's NOTIONAL, charged as a cost, never as a price improvement",
    "rebalance_cost": "0.0, because REBALANCE_POLICY performs none",
    "financing_or_borrow_cost": (
        "0.0, and this is a consequence of the construction rather than an optimistic "
        "assumption: the spot leg is bought outright with allocated capital and the "
        "perpetual leg is 1x isolated. Nothing is borrowed, so there is nothing to charge "
        "interest on."
    ),
    "idle_cash_yield": "0.0",
    "rationale": (
        "VIP-0 taker rates, no BNB discount, taker on both legs, slippage charged on every "
        "fill. The repository's directional research uses a 20 bps round-trip cost threshold "
        "for a single spot leg; this model charges 30 bps round-trip on spot and 20 bps on "
        "the perpetual, which is stricter per leg and covers two legs."
    ),
    "per_block_round_trip_cost_of_total_capital": (
        "approximately 0.0025 of total capital: 0.003 round-trip on a spot leg of about half "
        "of it, plus 0.002 round-trip on a perpetual leg of about half of it"
    ),
    "never_lowered": (
        "no fee, no slippage and no friction in this model is reduced because the strategy "
        "looks unprofitable. Lowering a cost after seeing a result is the rescue this "
        "preregistration exists to prevent."
    ),
}

# ---------------------------------------------------------------------------
# §6  Funding and basis accounting
# ---------------------------------------------------------------------------

FUNDING_SEMANTICS = {
    "sign_convention": (
        "cash_flow = -sign(side) x notional x rate, the single convention already written "
        "once in chimera.futures.accounting. Positive rate: longs pay shorts. So the SHORT "
        "perpetual leg RECEIVES when the rate is positive and PAYS when it is negative."
    ),
    "notional_base": (
        "the position notional at the MARK price at the settlement instant, per "
        "MARK_PRICE_FALLBACK when mark is unavailable"
    ),
    "application": (
        "exactly once per settlement event, while the position is open, deduplicated by "
        "settlement instant. A redelivered or duplicated archive row changes nothing."
    ),
    "settlements_outside_holding": "not applied. A settlement before the open or after the "
    "close is not this position's cash flow.",
    "boundary_tie_rule": (
        "explicit because it fires in EVERY block: each block opens at 00:00:00 UTC on 1 "
        "January, which is itself a Binance USD-M settlement instant, and P4 amendment A2 "
        "records a real settlement row at exactly 2020-01-01T00:00:00+00:00.\n"
        "  * a settlement whose instant EQUALS the open instant is NOT applied — the position "
        "did not hold through that accrual window, and crediting it would be a small gift on "
        "the payoff variable;\n"
        "  * a settlement whose instant EQUALS the close or liquidation instant IS applied — "
        "the position was held through that window.\n"
        "Formally: apply settlements with open_instant < settlement_instant <= close_instant."
    ),
    "dedup_key": (
        "the settlement instant. Where a settlement is handed to "
        "chimera.futures.accounting.Ledger.book_funding, its settlement_id is derived "
        "deterministically from that instant, so the module's id-based dedup and this "
        "design's instant-based dedup are one rule rather than two that could disagree."
    ),
    "cadence": "read from the archive, never assumed. See the funding source's "
    "interval_policy.",
    "rate_unit": (
        "a signed DECIMAL FRACTION of notional, charged once per settlement event exactly as "
        "published. NOT a percent, NOT basis points, NOT annualised, and NEVER multiplied by "
        "a settlements-per-day count. A typical BTCUSDT 8-hourly rate is on the order of "
        "1e-4; read as a percent it would become 1e-2 per settlement, which over a year of "
        "settlements does not shade a verdict but manufactures one."
    ),
    "unit_fail_closed": (
        "if any |rate| in an acquired archive exceeds 0.01, the RUN IS REFUSED as a unit "
        "error rather than clipped, filtered or winsorised. 0.01 is far outside any "
        "legitimate 8-hourly BTCUSDT settlement and is a corruption detector, not an "
        "economic assumption about caps."
    ),
    "reported_separately": (
        "funding paid and funding received are reported as two non-negative magnitudes, not "
        "collapsed into one net number, so a position the market charged to hold is "
        "distinguishable from one that simply earned little."
    ),
}

BASIS_DEFINITION = {
    "basis": "basis_t = perpetual_close_t - spot_close_t, in quote units per BTC",
    "basis_fraction": "basis_t / spot_close_t",
    "which_series_plays_which_role": (
        "named so no invariant depends on an implementer's choice.\n"
        "  * MARK-TO-MARKET of each leg uses the CLOSE of that leg's own hourly candle: the "
        "spot close for the spot leg, the perpetual close for the perpetual leg. Where "
        "chimera.futures.accounting.unrealised_pnl is used, the perpetual CLOSE is passed in "
        "its mark-price argument position, so the basis identity stays exact rather than "
        "failing by Q x (mark - perp_close).\n"
        "  * The MARK series is used for two things only: the funding notional and the "
        "liquidation test.\n"
        "  * The funding notional at a settlement instant uses the mark candle CLOSE at or "
        "immediately preceding that instant; under MARK_PRICE_FALLBACK it is the spot candle "
        "CLOSE at or immediately preceding it. Both are knowable at the settlement instant."
    ),
    "structural_price_pnl_identity": (
        "for the equal-quantity hedge, the two legs' price PnL is exactly "
        "Q x (basis_at_entry - basis_at_exit). The LONG spot leg earns "
        "Q x (spot_out - spot_in) and the SHORT perpetual leg earns "
        "Q x (perp_in - perp_out); their sum telescopes. The delta-hedged position's price "
        "PnL is therefore PURE BASIS CONVERGENCE, which is a hard invariant the accounting "
        "controls assert rather than a description."
    ),
    "identity_scope": (
        "the identity holds whenever both legs are opened at one instant and closed at one "
        "instant with a single quantity Q — that is, under S0, S1 and S3, since S3 only "
        "shifts the two basis values. It does NOT hold under S2, where the perpetual leg is "
        "opened and closed one bar apart from the spot leg and the position is genuinely "
        "directional at each end. The invariant test is therefore SCOPED to the simultaneous "
        "cases and must not be asserted against S2; asserting it there would either fail "
        "spuriously or, worse, be 'fixed' by weakening the accounting."
    ),
    "no_double_counting": (
        "basis PnL and funding are disjoint by construction. Funding is a cash flow at "
        "settlement instants; basis PnL is the mark-to-market difference between entry and "
        "exit. Neither is derived from the other and neither is added twice."
    ),
}

# ---------------------------------------------------------------------------
# §7  Margin and liquidation
# ---------------------------------------------------------------------------

MARGIN_AND_LIQUIDATION = {
    "leverage": "1x gross in every model below. Never above 1x, under any variant.",
    "primary_model": (
        "PORTFOLIO (cross) margin. The two legs are one portfolio: the spot holding "
        "collateralises the perpetual short, which is how a delta-hedged carry position is "
        "actually financed. Liquidation is evaluated against TOTAL portfolio equity versus "
        "the perpetual's maintenance-margin requirement, not against a walled-off margin "
        "balance."
    ),
    "why_not_isolated_as_primary": (
        "because at equal quantity the portfolio is price-invariant, and isolating the "
        "perpetual leg from the spot leg that hedges it measures an artifact rather than the "
        "carry. Total equity moves only with basis, funding and costs; a price rise that "
        "costs the short exactly what it pays the spot leg cannot make the PORTFOLIO "
        "insolvent. Modelling the perpetual as if the spot leg did not exist would report "
        "liquidations that a real carry book financing both legs from one balance would "
        "never take."
    ),
    "disclosure_of_how_this_choice_was_made": (
        "HONESTLY DISCLOSED: this margin model was selected during design after checking, "
        "against the already-committed pre-boundary Binance SPOT snapshot, whether a strictly "
        "isolated 1x short could survive a calendar-year holding period. It cannot: the "
        "repository's own liquidation_price puts a 1x isolated SHORT at entry x "
        "(2 - maintenance_margin_rate), roughly a 99.6% adverse move, and BTC's intra-year "
        "high exceeded that multiple of its first close in 2020, 2021, 2023 and 2024. No "
        "funding, basis, PnL or P13 return was computed to establish this — only price "
        "extrema of a source already frozen in this repository, strictly before the research "
        "boundary.\n"
        "The claim is deliberately narrow, because the price check supports only a narrow "
        "one: it establishes that a strictly isolated 1x short is LIQUIDATED in four of the "
        "six blocks. It does NOT establish the SIGN of those blocks' returns, and none was "
        "computed. At equal quantity the spot leg's gain roughly offsets the forfeited "
        "margin, so a liquidated block's realised return is dominated by the funding accrued "
        "before the liquidation instant — which is exactly what S4 measures and what this "
        "preregistration declines to predict. What the check does justify is that the "
        "isolated model truncates the holding period in most blocks, so it answers a "
        "different question from the one asked; it does not by itself prove that model would "
        "have failed the gate."
    ),
    "isolated_is_measured_not_assumed_away": (
        "the strict isolated case is not discarded — it is stress S4, predeclared here, "
        "reported for every block, and deliberately OUTSIDE the viability gate. Its purpose "
        "is to quantify exactly how much of the result depends on the margin model, which is "
        "the honest treatment of a modelling choice this consequential."
    ),
    "liquidation_check": (
        "two different tests, named separately because they are not the same computation.\n"
        "  * PRIMARY (portfolio): the inequality `portfolio_equity_t < Q x mark_t x "
        "maintenance_margin_rate`, evaluated at every hourly grid instant while the position "
        "is open. It does NOT call liquidation_price, and it can only fire after cumulative "
        "funding and cost losses have eaten the book — not from price, since the hedged "
        "portfolio is price-invariant.\n"
        "  * S4 (isolated): chimera.futures.accounting.liquidation_price at leverage 1 and "
        "the venue's tier-1 maintenance margin rate — the repository's own function, not a "
        "second copy of the formula — against the isolated balance defined in S4."
    ),
    "forced_close_price": (
        "a liquidation-forced close fills at the OPEN of the FOLLOWING bar, not at the "
        "trigger price. The trigger is detected from within-bar price action that an hourly "
        "grid cannot resolve, so filling at the trigger would assume an execution precision "
        "the data does not support; the next open is the first price an operator could "
        "actually have transacted at, and the choice errs against the position."
    ),
    "maintenance_margin_rate_is_an_approximation": (
        "DISCLOSED: at roughly half of 1,000,000 USDT of perpetual notional the position sits "
        "ABOVE Binance's tier-1 leverage bracket, so the tier-1 maintenance margin rate is an "
        "approximation rather than the exact applicable rate, and the true rate is higher. "
        "Per chimera/futures/accounting.py the omission is anti-conservative for a SHORT, so "
        "it biases S4 toward LATER liquidation — the diagnostic is if anything too kind to "
        "the isolated case, which is the safe direction for a bound that exists to be "
        "unforgiving. No historical bracket archive is preregistered, so the exact schedule "
        "is not recoverable here."
    ),
    "on_liquidation": (
        "the block is recorded as LIQUIDATED at that instant: the perpetual margin at risk is "
        "lost, the spot leg is closed at that instant paying its exit costs, and the "
        "resulting REALISED return — which need not be a loss, since the spot leg has gained "
        "roughly what the short lost — is the block's return. A liquidated block is INCLUDED "
        "in the evaluation at its realised number and is never re-simulated with a top-up."
    ),
    "no_margin_top_up": (
        "no discretionary transfer, and no top-up trigger. A top-up threshold would be a "
        "parameter. The primary model needs none because the collateral is already one pool; "
        "S4 needs none because refusing the top-up is the whole point of S4."
    ),
    "no_quantity_change": (
        "neither model ever changes Q. Margin treatment affects financing and solvency, never "
        "the hedge, so delta stays identically zero in both."
    ),
    "what_the_portfolio_model_assumes_operationally": (
        "STATED because it is the model's weakest operational assumption. Treating the spot "
        "holding as collateral for the perpetual short requires an account arrangement that "
        "actually does that — Binance Portfolio Margin, or an operator moving quote "
        "collateral "
        "between the spot and USD-M wallets in time. A plain USD-M cross-margin account does "
        "NOT automatically collateralise from spot BTC holdings. The primary model therefore "
        "describes a carry book financed as one pool, which is how such a book is run, and "
        "NOT the default retail account configuration. Transfer latency, transfer limits and "
        "the eligibility rules of Portfolio Margin are not modelled. S4 is the bound on how "
        "much this assumption is worth."
    ),
    "what_the_simulation_cannot_determine": (
        "intra-hour liquidation between hourly grid points, the tiered maintenance-margin "
        "schedule above tier 1, the maintenance-amount deduction, auto-deleveraging, and any "
        "wallet-transfer latency between the spot and futures accounts. Each is unknowable "
        "from the preregistered sources. The fail-closed treatment is to test liquidation "
        "against the hourly HIGH of the mark series where available and the hourly close "
        "otherwise, and to RECORD which was used, so the check is never quietly weaker than "
        "it claims to be."
    ),
    "dry_run_venue_limitation": (
        "docs/futures_dry_run_validation.md records that the dry-run venue has no full "
        "liquidation engine. That absence is an engineering limitation and is NOT permitted "
        "to become an economic assumption here: P13 performs its own liquidation check "
        "against price history rather than inheriting the venue's silence."
    ),
}

#: What happens when a liquidation trigger has no permitted following bar.
#:
#: **This is amendment A1, and it is not an original preregistration rule.**
#:
#: The original design settled the FILL: ``forced_close_price`` says a
#: liquidation-forced close executes at the OPEN OF THE FOLLOWING BAR, and
#: ``POSITION_LIFECYCLE.close_instant`` bounds every exit search by the block end
#: and by the research boundary. Read together those two sentences already
#: determine that a trigger on the block's FINAL quote has no permitted fill: the
#: trigger bar's own open is a price stamped before the touch that caused the
#: liquidation, the next block's first bar is outside the bound, and for the
#: final block that bar is at or after the research boundary. That much is
#: deduction, not choice, and this amendment does not restate it as a decision.
#:
#: What the original text did NOT settle is what such a block does to the GATE.
#: ``VIABILITY_GATE.excluded_blocks`` authorises exclusion for blocks that could
#: not be OPENED, and this block opened. ``VIABILITY_GATE.liquidated_blocks``
#: says a liquidated block is "counted as included blocks ... Not excluded", and
#: this block cannot be counted at a realised return because it has none.
#: ``close_instant`` names the state — UNCLOSED, "reported ... with its reason
#: rather than back-dated" — and stops there. Two frozen rules point in opposite
#: directions and neither reaches this case, so a future run would have had to
#: choose one WITH THE DATA IN FRONT OF IT. That is the researcher degree of
#: freedom this amendment removes.
#:
#: **Adopted after an environment-only NOT EVALUABLE outcome and before any P13
#: economic observation of any kind.** The sources were never obtained, no block
#: was opened, no return, funding total, basis figure or gate condition was ever
#: computed, and nothing about the market informed this choice. The only
#: information that entered it is the shape of the frozen text itself.
#:
#: **The direction is fail-closed, and that is checkable rather than asserted.**
#: Excluding the block would have been the flattering option: it drops a block
#: that was being liquidated out of G2's mean and out of G3's worst-block test,
#: while G1's required count of four positives is unchanged by removing a
#: non-positive block. Terminating the screen INVALID cannot flatter anything —
#: it forfeits the possibility of a VIABLE verdict outright. An amendment that
#: can only make the verdict harder to obtain is the one safe kind to make after
#: a design is frozen.
FORCED_CLOSE_WITHOUT_A_FOLLOWING_BAR: dict[str, Any] = {
    "amendment": "A1",
    "amendment_status": (
        "adopted after P13's environment-only NOT EVALUABLE closure — the preregistered "
        "Binance archives were unreachable — and BEFORE any P13 economic observation "
        "exists. No block was opened, no return, funding total, basis figure or gate "
        "condition was computed, and no market data informed this rule. It is an "
        "execution-semantics amendment, not an original preregistration rule"
    ),
    "supersedes": (
        "silence. The original text fixed the forced-close FILL (the following bar's open) "
        "and bounded the exit search by the block end and the research boundary, but said "
        "nothing about what an opened-and-unclosable block does to the viability gate"
    ),
    "what_the_original_text_already_determines": (
        "NOT re-decided here, and recorded so this amendment is not read as wider than it "
        "is:\n"
        "  * the forced close fills at the OPEN of the FOLLOWING bar "
        "(MARGIN_AND_LIQUIDATION.forced_close_price);\n"
        "  * no fill may use the trigger bar's own open, which is stamped BEFORE the "
        "intra-bar touch that caused the liquidation;\n"
        "  * the exit search may not run past the block end or the research boundary "
        "(POSITION_LIFECYCLE.close_instant), so the following bar of the NEXT block is not "
        "available and, for the final block, is inside the retired P4-HOLD region."
    ),
    "the_rule": (
        "when a liquidation trigger fires on the FINAL quote of a block, no permitted "
        "forced-close fill exists. The block is recorded UNCLOSED and LIQUIDATED at the "
        "trigger instant, with the funding, fees and slippage actually incurred up to that "
        "instant reported as the facts they are. Its close-dependent economics — exit "
        "basis, basis PnL, net PnL and net return — are NOT DETERMINABLE and are emitted "
        "as NaN rather than as zero, so a gate that ignores the UNCLOSED flag raises "
        "instead of averaging in a number nobody measured. The SCREEN terminates "
        "'P13 ALWAYS-ON ANNUAL SPOT/PERP CARRY: INVALID'."
    ),
    "why_invalid_rather_than_excluded": (
        "excluding the block would drop a block that was being liquidated out of G2's mean "
        "and out of G3's worst-block test, which is precisely the flattering treatment "
        "VIABILITY_GATE.liquidated_blocks already refuses for every OTHER liquidated block. "
        "INVALID is the only treatment that neither invents a price nor improves the "
        "verdict."
    ),
    "why_not_a_price": (
        "no price is invented and nothing outside the permitted region is read. The "
        "alternatives all require one: the trigger bar's open is acausal, a mark or close "
        "is never a fill under EXECUTION_PRICE_POLICY, a zero valuation is a fabricated "
        "number, and the next block's open crosses a bound the frozen text sets."
    ),
    "applies_equally_to_the_other_unclosed_cause": (
        "POSITION_LIFECYCLE.close_instant already reports a block UNCLOSED when no valid "
        "instant exists at or before the block's last hour. That block is unclosable for a "
        "different reason and has the same missing quantity, so it takes the same "
        "treatment. Defining UNCLOSED once, for both causes, is a narrower commitment than "
        "defining it only for the case that prompted the question."
    ),
    "direction_of_conservatism": (
        "strictly one-way. This rule can only remove a verdict, never produce or improve "
        "one: it makes VIABLE unreachable for the whole screen. It is therefore incapable "
        "of rescuing a negative result, which is the property that makes a post-freeze "
        "amendment admissible at all."
    ),
    "not_chosen_from_data": (
        "no market observation of any kind was available when this was adopted, and none "
        "was consulted. P13's sources have never been obtained."
    ),
    "does_not_disturb_the_acquisition_evidence": (
        "the committed acquisition plan, refusal record and STATUS at "
        "artifacts/benchmark/btc_p13_carry/ were generated at 2b1b400e under the ORIGINAL "
        "design and quote its hash. They are NOT regenerated and NOT rewritten. This "
        "amendment governs block execution — a rule about which bar a forced close fills "
        "at — and the acquisition never obtained a single row for any bar, so nothing in "
        "that evidence was produced under a rule this changes. Historical evidence stays "
        "historical; the superseded hash it carries is recorded in "
        "docs/p13_preregistration.md as provenance"
    ),
    "reachability": (
        "STATED so the amendment is not read as academic. Under the PRIMARY portfolio "
        "model the hedged book is price-invariant, so a liquidation can only follow "
        "cumulative funding and cost losses and a trigger on a block's very last hour is "
        "remote. Under S4, the isolated diagnostic, a price touch alone can trigger it and "
        "the design already records that a strictly isolated 1x short is liquidated in four "
        "of six blocks — so the case is reachable in the diagnostic that exists to be "
        "unforgiving, which is exactly where an undefined rule would have been resolved "
        "against a visible number."
    ),
}

#: The three distinct states a missing liquidation mark can put a block in.
#:
#: Named as constants because a future block runner and this design must agree on
#: the vocabulary, and because the whole point of A2 is that these three are NOT
#: the same state and must never be collapsed into one "no mark" branch.
MARKLESS_STATE_PRE_OPEN = "pre_open_mark_absent"
MARKLESS_STATE_HELD = "held_bar_mark_absent"
MARKLESS_STATE_NO_VALID_OPEN = "no_valid_opening_instant_in_block"

MARKLESS_STATES: tuple[str, ...] = (
    MARKLESS_STATE_PRE_OPEN,
    MARKLESS_STATE_HELD,
    MARKLESS_STATE_NO_VALID_OPEN,
)

#: What a mark-less period does to a block — settled before any number exists that
#: could settle it.
#:
#: **This is amendment A2, and it is not an original preregistration rule.**
#:
#: The frozen design already admits the fact that forces the question. The
#: ``mark_price`` entry in :data:`DATA_SOURCES` records that the first published
#: month for BTCUSDT is UNMEASURED, and :data:`MARK_PRICE_FALLBACK` is explicitly
#: "PER ARCHIVE OBJECT, not all-or-nothing" — so a mark series present for some
#: months and absent for others is this design's own stated expectation rather than
#: a surprise.
#:
#: **What the frozen text already settles, and A2 does not re-decide.**
#: ``MARK_PRICE_FALLBACK`` authorises the spot close as a substitute for the
#: FUNDING NOTIONAL BASE. ``BASIS_DEFINITION.which_series_plays_which_role`` names
#: the liquidation test as a SECOND, separate use of the mark series and does not
#: extend that substitution to it.
#: ``MARGIN_AND_LIQUIDATION.what_the_simulation_cannot_determine`` requires the
#: liquidation test to use "the hourly HIGH of the mark series where available and
#: the hourly close otherwise", and to record which was used. Read together those
#: already determine that a HELD bar carrying neither a mark high nor a mark close
#: has NO authorised liquidation touch, which is why ``nn.p13_carry.Quote`` refuses
#: one. That is deduction, and this amendment does not restate it as a choice.
#:
#: **What the frozen text does NOT settle** is what the block runner — which does
#: not exist yet — must DO with that refusal, and it fails to settle it in two
#: places that pull in opposite directions:
#:
#:   * BEFORE a position exists. ``POSITION_LIFECYCLE.open_instant`` requires
#:     "spot, perpetual and (where used) mark observations" to be present and valid
#:     at the opening instant. Whether the mark is "used" there is precisely the
#:     ambiguity: it is not a fill price, and it IS the series every subsequent held
#:     bar's risk check depends on.
#:   * AFTER a position exists. ``VIABILITY_GATE.excluded_blocks`` authorises
#:     exclusion only for a block that could not be OPENED; ``liquidated_blocks``
#:     refuses to drop a block that was opened; ``NOT_EVALUABLE_MEANING`` speaks to
#:     sources that could not be obtained at all. None of the three reaches a block
#:     that opened and then met a bar whose required risk quantity is not
#:     computable.
#:
#: A run meeting that gap with the data in front of it would have to choose, and
#: the available choices are not equally flattering: excluding the affected block,
#: or jumping the hole, keeps a screen evaluable by discarding exactly the periods
#: where the venue's own risk series is missing. That is the researcher degree of
#: freedom this amendment removes.
#:
#: **Adopted before any P13 economic observation of any kind.** No archive object
#: has been fetched, no block opened, no funding total, basis figure, return or gate
#: condition computed, and no market data informed this rule. The only inputs were
#: the shape of the frozen text and the coverage fact the design already declares.
#:
#: **The two TERMINAL branches are fail-closed, and the pre-open branch is
#: economically INDETERMINATE — revision A2R1 corrects a false claim the first
#: committed A2 made here.** That first version asserted the pre-open rule "can
#: only SHORTEN a block's exposure, which can only reduce accrued funding" and that
#: no branch "can add a positive block, raise a mean, lift a worst block". **That
#: monotonicity claim is false**, by arithmetic rather than by observation: a
#: skipped pre-open interval may contain funding of EITHER sign, and moving the
#: opening instant moves ``basis_entry``, which enters block PnL through the
#: telescoping identity in ``BASIS_DEFINITION.structural_price_pnl_identity`` in
#: EITHER direction. Skipping an interval of negative funding, or entering at a
#: wider basis, can therefore IMPROVE a block. Shorter exposure is not
#: economically monotone.
#:
#: What survives, and is the actual ground of admissibility: the rule was frozen
#: BEFORE acquisition and before any P13 economic observation, and it is triggered
#: ONLY by source validity. An ex-ante causal convention whose economic direction
#: is unknown cannot be a rescue of a result nobody has seen. That is a different
#: and weaker argument than one-way conservatism, and stating the weaker true one
#: is the point of this revision.
#:
#: The two terminal branches remain genuinely fail-closed: each forfeits every
#: verdict, ``VIABLE`` included, rather than producing or improving one.
MARKLESS_LIQUIDATION_VALIDITY_POLICY: dict[str, Any] = {
    "amendment": "A2",
    "revision": "A2R1",
    "revision_history": (
        {
            "revision": "A2",
            "hash": "sha256:86050b6897143cd0abfa2106ee6cc37baaf3c637e396b23f30bdeeb935fbfe6c",
            "committed_at": "0d023ba5a33bbcae55bb079146838f5f22959607",
            "status": "SUPERSEDED",
            "superseded_because": (
                "it asserted that the pre-open branch 'can only SHORTEN a block's exposure, "
                "which can only reduce accrued funding' and that no branch 'can add a "
                "positive block, raise a mean, lift a worst block'. That monotonicity claim "
                "is FALSE: skipped pre-open funding is not sign-constrained and a delayed "
                "open moves basis_at_entry in either direction, so a delayed open may "
                "improve a block. An independent review identified it before implementation "
                "and before acquisition"
            ),
        },
        {
            "revision": "A2R1",
            "status": "ACTIVE",
            "what_changed": (
                "the JUSTIFICATION ONLY. The false monotonicity claim is withdrawn and "
                "replaced by an explicit statement that the pre-open branch's economic "
                "direction is INDETERMINATE EX ANTE, with admissibility resting on ex-ante "
                "timing, a source-only trigger and causal implementability instead"
            ),
        },
    ),
    "behaviour_unchanged_from_first_a2": (
        "STATED because a hash movement should never hide a silent behaviour change. A2R1 "
        "alters NO source-validity treatment. The authorised liquidation sources and their "
        "priority, the empty authorised-surrogate list and the forbidden-surrogate list, the "
        "pre-open causal search and its bounds, the prohibition on attributing anything to "
        "the strategy before the valid open, the held-bar terminal NOT EVALUABLE and its "
        "forbidden treatments, the no-valid-open terminal NOT EVALUABLE, the refusal to "
        "convert either into an excluded block, the exit-bar exemption, the independence of "
        "MARK_PRICE_FALLBACK, and the evidence requirement are all IDENTICAL to the first "
        "committed A2. Only the claimed economic direction changed, and it changed from a "
        "false claim to no claim"
    ),
    "amendment_status": (
        "adopted after P13's environment-only NOT EVALUABLE closure and after the A1-era "
        "evaluator repair that removed the unauthorised third liquidation tier, and BEFORE "
        "any P13 economic observation exists. No archive object was fetched, no block was "
        "opened, no return, funding total, basis figure or gate condition was computed, and "
        "no market data informed this rule. It is a SOURCE-VALIDITY amendment, not an "
        "original preregistration rule and not an economic one"
    ),
    "amendment_timing": (
        "BEFORE ANY P13 ECONOMIC OBSERVATION, and specifically before acquisition. The "
        "timing is not a formality: this rule can change WHICH historical observations enter "
        "a block, and a rule that selects observations is inadmissible once the observations "
        "are visible. It is therefore an explicit hash-moving amendment rather than a "
        "clarification, even though it moves no threshold"
    ),
    "supersedes": (
        "silence, in two distinct places. POSITION_LIFECYCLE.open_instant requires the mark "
        "to be valid at the open '(where used)' without saying whether it is used there, and "
        "no frozen rule reaches a block that OPENED and then met a held bar carrying no "
        "authorised liquidation touch"
    ),
    "what_the_original_text_already_determines": (
        "NOT re-decided here, and recorded so this amendment is not read as wider than it "
        "is:\n"
        "  * the liquidation test takes the hourly mark HIGH where available and the hourly "
        "mark CLOSE otherwise, recording which was used "
        "(MARGIN_AND_LIQUIDATION.what_the_simulation_cannot_determine);\n"
        "  * MARK_PRICE_FALLBACK's spot substitution is authorised for the FUNDING NOTIONAL "
        "BASE and for nothing else;\n"
        "  * BASIS_DEFINITION.which_series_plays_which_role keeps the liquidation test a "
        "SEPARATE use of the mark series, so the funding substitution does not reach it;\n"
        "  * therefore a HELD bar with neither mark series has no computable liquidation "
        "quantity, and nn.p13_carry.Quote already refuses it."
    ),
    "scope": (
        "SOURCE VALIDITY ONLY. A2 decides which historical instants the frozen design is "
        "entitled to evaluate, and what happens when it is entitled to evaluate none. It "
        "introduces no signal, no filter conditioned on anything economic, no threshold, no "
        "parameter and no new instrument"
    ),
    "authorised_liquidation_sources": ("mark_high", "mark_close"),
    "source_priority": (
        "the mark HIGH is PREFERRED whenever the bar carries it, because adverse for a SHORT "
        "means HIGH and an hourly grid cannot otherwise resolve a within-bar touch. The mark "
        "CLOSE is the authorised fallback and is a materially weaker test, which is why the "
        "two are recorded by name rather than by a boolean. If the bar carries neither, no "
        "liquidation quantity is computable and there is no third tier"
    ),
    "authorised_liquidation_surrogates": (),
    "forbidden_liquidation_surrogates": (
        "the Binance spot close",
        "the Binance spot high",
        "the perpetual TRADE close",
        "the perpetual TRADE high",
        "any REST endpoint value",
        "any other venue's price",
        "a reconstructed or synthetic mark series",
        "zero",
        "infinity",
        "any other surrogate whatsoever",
    ),
    "why_no_surrogate": (
        "each substitution answers a different question from the one the liquidation test "
        "asks, and every one of them errs in the same anti-conservative direction. A spot "
        "close or spot high cannot see a perpetual mark spike at all. A perpetual TRADE "
        "price is wrong by the basis — the quantity this checkpoint exists to measure — and "
        "is not the series the venue liquidates against. A REST value is not a row of the "
        "preregistered historical source, which docs/p4_preregistration.md section 3 already "
        "settled. Another venue breaks VENUE_POLICY. A reconstructed mark is a number this "
        "design invented, and zero or infinity are fabricated numbers dressed as data. "
        "Refusing all of them is the same refusal nn.p13_carry.Quote already performs"
    ),
    "states": {
        MARKLESS_STATE_PRE_OPEN: {
            "when": (
                "a candidate opening grid instant inside a block at which the required "
                "liquidation mark is unavailable, while NO position exists yet"
            ),
            "treatment": (
                "the instant is NOT a valid opening instant. The block's opening search "
                "advances CAUSALLY to the first valid executable instant at or after the "
                "calendar block boundary, provided it remains inside that same block, "
                "remains strictly before the research boundary, and satisfies every other "
                "preregistered validity requirement including "
                "POSITION_LIFECYCLE.validity_definition and both execution legs' opens. This "
                "is the forward search POSITION_LIFECYCLE.open_instant already describes, "
                "with the mark's role in it made explicit"
            ),
            "terminal": False,
            "result_state": None,
            "scope": "this block's opening instant only",
            "no_attribution_before_the_open": (
                "no position exists during the skipped pre-open interval, so NOTHING may be "
                "attributed to the strategy across it: no liquidation exposure, no funding "
                "cash flow, no basis PnL, no fee and no slippage. The block's holding period "
                "begins at the valid opening instant, and the capital sits idle at zero "
                "yield before it, exactly as VIABILITY_GATE.the_hurdle_is_zero_yield_cash "
                "already requires of idle capital"
            ),
            "is_not": (
                "not block deletion, not block exclusion, not a redefined block span, and "
                "NOT a new economic signal. The block's calendar period is unchanged; only "
                "the instant at which a valid position can first exist inside it moves"
            ),
            "economic_direction": "INDETERMINATE EX ANTE",
            "why_indeterminate": (
                "CORRECTED IN A2R1, and the correction is arithmetic rather than empirical — "
                "no market data was consulted to reach it. The first committed A2 claimed a "
                "later open 'can only reduce the funding a block accrues'. It cannot: a "
                "delayed open changes the ENTRY INSTANT, and the skipped interval is not "
                "sign-constrained.\n"
                "  * FUNDING. A Binance USD-M funding rate takes EITHER sign, and this "
                "construction is SHORT the perpetual, so it RECEIVES on a positive rate and "
                "PAYS on a negative one. A skipped interval may therefore contain funding "
                "the block would have received (skipping it WORSENS the block) or funding "
                "the block would have paid (skipping it IMPROVES the block). Nothing frozen "
                "here fixes which.\n"
                "  * BASIS. Under BASIS_DEFINITION.structural_price_pnl_identity the hedged "
                "price PnL is Q x (basis_at_entry - basis_at_exit). A later open sets a "
                "DIFFERENT basis_at_entry, so it moves basis PnL in EITHER direction; "
                "entering at a wider basis raises it and entering at a narrower one lowers "
                "it.\n"
                "  * FEES AND SLIPPAGE are unchanged in count: the construction still "
                "performs exactly one open and one close per block, so a delayed open does "
                "not add or remove a round trip.\n"
                "Shorter exposure is NOT economically monotone, and this design claims no "
                "monotonicity for it in either direction"
            ),
            "why_it_is_nonetheless_admissible": (
                "NOT because it is unfavourable — it may be favourable — but because of WHEN "
                "and ON WHAT it was frozen. It is an ex-ante, causal, source-validity "
                "convention: adopted before acquisition and before any P13 economic "
                "observation, triggered only by whether an archive row exists, and "
                "implementable by an operator standing at the instant. A rule whose economic "
                "direction was unknown to its author, and which no observation informed, "
                "cannot have been selected to rescue a result nobody had seen. The "
                "alternative it refuses — opening at an instant whose risk series is absent "
                "— would buy exposure the design cannot audit"
            ),
        },
        MARKLESS_STATE_HELD: {
            "when": (
                "a HELD bar — one of bars 0 .. N-1 of an OPENED block's holding window — "
                "carrying neither a mark high nor a mark close"
            ),
            "treatment": (
                "the historical screen TERMINATES. The required liquidation quantity for a "
                "bar the position was actually exposed through is not computable from the "
                "preregistered sources, and no authorised substitute exists"
            ),
            "terminal": True,
            "result_state": "P13 ALWAYS-ON ANNUAL SPOT/PERP CARRY: NOT EVALUABLE",
            "scope": (
                "SCREEN-WIDE and terminal for this governed run. Not per bar, not per block"
            ),
            "forbidden_treatments": (
                "silently skipping the bar",
                "jumping across the missing period",
                "closing the position before it",
                "reopening after it",
                "excluding only the affected block",
                "converting the block to opened=False",
                "treating the missing period as a zero return",
                "recording the block UNCLOSED or INVALID merely to preserve evaluability",
                "altering the G1-G6 denominators",
                "continuing with five blocks",
            ),
            "why_terminal_rather_than_local": (
                "every local treatment is a selection rule that fires on source coverage and "
                "lands on the result. Dropping the affected block removes precisely the "
                "period whose risk series is missing from G2's mean and G3's worst-block "
                "test; jumping the hole reports an exposure the run never audited; a zero "
                "return is a number nobody measured. A2 refuses the whole screen instead, "
                "which is the one treatment that cannot improve a verdict"
            ),
            "what_it_is_not": (
                "the reason is SOURCE INSUFFICIENCY FOR A REQUIRED RISK QUANTITY, not an "
                "observed economic failure. NOT EVALUABLE says nothing about carry, is not a "
                "negative result, and must never be cited as one — the same reading "
                "docs/current_development_plan.md already fixes for the environment-only NOT "
                "EVALUABLE closure"
            ),
            "partial_numbers_are_not_a_result": (
                "STATED because this state, unlike the acquisition-time NOT EVALUABLE that "
                "preceded it, can be reached after computation has begun. Any block "
                "economics computed before the refusal fires are NOT a result: they are not "
                "written as primary evidence, not reported as a partial answer, and do not "
                "enter any gate. The run reports the terminal state and what was missing"
            ),
        },
        MARKLESS_STATE_NO_VALID_OPEN: {
            "when": (
                "an entire calendar block in which the causal opening search finds no valid "
                "opening instant at all, because the required liquidation mark source never "
                "becomes available inside that block before the research boundary"
            ),
            "treatment": (
                "the historical screen TERMINATES. The block is not opened, and it is NOT "
                "converted into an excluded block"
            ),
            "terminal": True,
            "result_state": "P13 ALWAYS-ON ANNUAL SPOT/PERP CARRY: NOT EVALUABLE",
            "scope": "SCREEN-WIDE and terminal for this governed run",
            "why_not_an_excluded_block": (
                "VIABILITY_GATE.excluded_blocks exists for a block the CONSTRUCTION could "
                "not open — a quantity flooring to zero at the step size, a leg below the "
                "venue minimum notional, required rows absent or invalid. Reading it as "
                "covering 'the mark archive was never published for this year' would turn a "
                "source-availability fact into an escape hatch that drops whole years out of "
                "G1's and G2's denominators, and it would drop exactly the years whose risk "
                "series is missing. The existing excluded-block rule is NOT broadened here, "
                "and it must not be broadened later"
            ),
        },
    },
    "exit_bar": {
        "rule": (
            "a NORMAL exit bar needs no liquidation mark. Under the repaired holding-window "
            "semantics the held bars are 0 .. N-1 and the exit bar N is POST-EXIT: the "
            "position closes at bar N's OPEN, before that bar's post-open high or close "
            "exists, so there is no intra-bar window at N the position was exposed through "
            "and no causal liquidation test to perform there"
        ),
        "what_is_still_required_at_the_exit_bar": (
            "BOTH execution legs' opens, under the existing execution validity rules. The "
            "exemption is from the LIQUIDATION mark and from nothing else: an exit bar "
            "missing a leg's open is invalid exactly as POSITION_LIFECYCLE.close_instant "
            "already provides, and the exit search moves to the first valid instant at or "
            "after it within the same bounds"
        ),
        "not_a_relaxation": (
            "this exempts no bar the position was actually held through. It states the "
            "consequence of a fill rule already frozen: a position closed at an open was "
            "never exposed to that bar's subsequent price action"
        ),
    },
    "funding_fallback_independence": {
        "rule": (
            "MARK_PRICE_FALLBACK is INDEPENDENT of this policy and is not narrowed by it. If "
            "a funding settlement at the close instant needs a notional base and no mark "
            "exists there, the already-authorised substitution of the spot candle CLOSE "
            "applies — for the FUNDING NOTIONAL BASE ALONE"
        ),
        "the_coupling_that_is_forbidden": (
            "in BOTH directions. A mark-less bar does not disable the funding fallback, and "
            "the funding fallback does not authorise a liquidation touch. The two are "
            "different uses of one series with different error properties: the funding "
            "substitution is wrong by basis_fraction x rate of notional, a product of two "
            "small numbers, while a spot-based liquidation touch is structurally blind to a "
            "perpetual mark spike"
        ),
        "reporting": (
            "unchanged. MARK_PRICE_FALLBACK.reporting_granularity still requires the "
            "substitution flag per block and a count of settlements that used the "
            "substituted base, and the liquidation touch provenance remains a separate count "
            "of liquidation TESTS by source"
        ),
    },
    "evidence_requirement": {
        "fields": (
            "the reason a block's opening was delayed, named as one of MARKLESS_STATES",
            "the amount of pre-open time skipped inside the block, and the count of skipped "
            "grid instants",
            "the realised opening instant, against the calendar block boundary",
        ),
        "relationship_to_the_per_block_report": (
            "IN ADDITION to TEMPORAL_PARTITION.per_block_report_fields, which is unchanged"
        ),
        "not_implemented_yet": (
            "STATED so this commit is not read as more than it is. The preregistration "
            "commit freezes the requirement; it writes no reporting code, no block runner, "
            "no loader and no downloader, and none of those exists"
        ),
    },
    "gate_structure_unchanged": (
        "six UTC calendar-year inference blocks",
        "G1 breadth, G2 mean, G3 downside floor, G4 sample, G5 stress, G6 effect size",
        "4-of-6 breadth",
        "the strict positive-mean rule",
        "the -0.02 worst-block floor",
        "the 200-settlement and 5-included-block rule",
        "the S1 and S3 gate requirements",
        "the 0.0025 minimum effect size",
        "the S2 and S4 diagnostics",
        "D1-D3",
        "the cost model",
        "leverage",
        "the venue",
        "the hedge ratio",
        "the research boundary",
    ),
    "no_gate_denominator_change": (
        "A2 never adds a block to, removes a block from, or reweights the denominators of G1 "
        "and G2. Its terminal branches stop the screen instead of editing a denominator, "
        "which is exactly why they cannot be used to improve a verdict"
    ),
    "no_affected_block_exclusion": (
        "no block is ever excluded, dropped, shortened, split, re-dated or re-labelled "
        "BECAUSE its mark coverage is incomplete. Source coverage may terminate the screen; "
        "it may never select which periods the screen reports"
    ),
    "direction_of_conservatism": (
        "NOT UNIFORM ACROSS BRANCHES, and A2R1 says so rather than overclaiming.\n"
        "  * The two TERMINAL branches — a held bar with no authorised mark, and a block "
        "with no valid opening instant — are genuinely fail-closed. Each forfeits the whole "
        "screen, VIABLE included, so neither can produce or improve a verdict.\n"
        "  * The PRE-OPEN branch is NOT one-way. Its economic direction is INDETERMINATE EX "
        "ANTE: see states.pre_open_mark_absent.why_indeterminate. It may improve or worsen "
        "funding PnL, and it may improve or worsen basis PnL. No monotonicity is claimed for "
        "it in either direction.\n"
        "The first committed A2 asserted one-way conservatism for the whole policy. That was "
        "wrong about the pre-open branch and is withdrawn here"
    ),
    "no_economic_monotonicity_is_claimed": (
        "stated as its own commitment so it cannot be lost in prose. This design does NOT "
        "claim that the pre-open rule can only reduce accrued funding, cannot add a positive "
        "block, cannot raise the mean of G2, cannot lift the worst block of G3, cannot "
        "improve any block, or is necessarily economically unfavourable. Every one of those "
        "claims is false, and each was asserted by the first committed A2 "
        "(sha256:86050b6897143cd0abfa2106ee6cc37baaf3c637e396b23f30bdeeb935fbfe6c) and is "
        "withdrawn by A2R1"
    ),
    "admissibility_basis": (
        "the ground on which this amendment stands, now that one-way conservatism is not "
        "available to it. Three facts, each checkable from the repository rather than "
        "asserted:\n"
        "  * TIMING. Frozen before P13 source acquisition and before any P13 economic "
        "observation. No archive object has been fetched, no coverage probe run, no block "
        "opened, and no funding total, basis figure, return or gate condition computed.\n"
        "  * TRIGGER. Fired only by whether a required archive row exists, never by any "
        "quantity a run computes — see why_this_is_source_validity_not_an_economic_rule.\n"
        "  * CAUSALITY. The opening search advances forward only, inside the same block and "
        "strictly before the research boundary, so it is implementable by an operator "
        "standing at the instant.\n"
        "A rule whose economic direction was unknown to its author, which no observation "
        "informed, and which could not have been evaluated against a result that did not "
        "exist, cannot be a rescue of that result. That is a WEAKER argument than one-way "
        "conservatism, and it is the true one"
    ),
    "evidence_ceiling_unchanged": (
        "this correction does not strengthen anything. Any future P13 result remains "
        "EXPLORATORY / ADAPTIVE HISTORICAL STRUCTURAL FEASIBILITY EVIDENCE under "
        "EVIDENCE_CEILING and HINDSIGHT_DISCLOSURE. Withdrawing a false conservatism claim "
        "removes an overstatement; it adds no evidential weight, and no positive P13 result "
        "becomes more credible because this text is now accurate"
    ),
    "why_this_is_source_validity_not_an_economic_rule": (
        "the trigger is the PRESENCE OF AN ARCHIVE ROW and nothing else. It is never fired "
        "by a return, a funding total, a basis level, a drawdown, a stress outcome or any "
        "other quantity a run computes — the same availability-only discipline "
        "MARK_PRICE_FALLBACK.never_triggered_by already imposes on the funding substitution. "
        "A rule that fired on an economic observation would be a filter chosen against a "
        "result; this one is decidable from the source manifest before a single economic "
        "number exists"
    ),
    "not_chosen_from_data": (
        "no market observation of any kind was available when this was adopted, and none was "
        "consulted. P13's sources have never been obtained, no coverage probe has been run "
        "against Binance, and the mark archive's first published month for BTCUSDT remains "
        "unmeasured. The policy is written to be correct under EVERY coverage outcome "
        "precisely because the actual coverage is unknown"
    ),
    "relation_to_not_evaluable_meaning": (
        "NOT_EVALUABLE_MEANING is unchanged and is not rewritten. It describes the "
        "acquisition-time case — sources that could not be obtained — and A2 adds two "
        "further routes to the SAME terminal state, both of them source insufficiency rather "
        "than economics. The one difference worth stating is that A2's held-bar route can be "
        "reached after computation has begun, and "
        "states.held_bar_mark_absent.partial_numbers_are_not_a_result governs that case "
        "explicitly"
    ),
    "boundaries_and_seals_unchanged": (
        "the exclusive research boundary remains 2025-05-19T08:00:00+00:00. P4-HOLD remains "
        "retired and unread, Styx remains sealed, and P8 remains unopened. The causal "
        "opening search introduced here is bounded by the block AND by that boundary, so it "
        "can never reach a sealed or retired instant"
    ),
    "does_not_disturb_the_acquisition_evidence": (
        "the committed acquisition plan, refusal record and STATUS at "
        "artifacts/benchmark/btc_p13_carry/ were generated at 2b1b400e under the ORIGINAL "
        "design and quote its hash. They are NOT regenerated and NOT rewritten. That "
        "acquisition obtained no row for any bar, so nothing in it was produced under a rule "
        "this amendment changes. Historical evidence stays historical; the superseded hashes "
        "are recorded in SUPERSEDED_HASHES and in docs/p13_preregistration.md as provenance"
    ),
    "implementation_status": (
        "DESIGN ONLY. This amendment is frozen before the runtime that will obey it exists. "
        "nn/p13_carry.py is unchanged by it, no downloader, loader or block runner was "
        "written with it, and P13 remains economically NOT YET RUN"
    ),
}

# ---------------------------------------------------------------------------
# §8  Temporal partition
# ---------------------------------------------------------------------------

TEMPORAL_PARTITION = {
    "unit": "UTC calendar year",
    "blocks": ("2020", "2021", "2022", "2023", "2024", "2025-partial"),
    "block_2025_span": "2025-01-01T00:00:00+00:00 to 2025-05-19T08:00:00+00:00 exclusive",
    "chosen_because": (
        "a calendar year spans multiple funding regimes, which is exactly what a structural "
        "carry screen must not slice away, and the UTC calendar year is a convention nobody "
        "chose for this result."
    ),
    "what_was_and_was_not_chosen": (
        "STATED HONESTLY rather than overclaimed. The OUTER edges are not a choice: the span "
        "starts where the sources begin and ends at the research boundary. The five INTERIOR "
        "boundaries ARE a choice — a calendar-year partition — and because each block opens "
        "and closes a position, those five boundaries fix ten of the twelve fill instants. "
        "Block returns are therefore sensitive to them. Saying 'no boundary was selected' "
        "would be false; what is true is that the convention was fixed before any result and "
        "cannot be moved after one. D3 reports how much the choice mattered."
    ),
    "explicitly_not": (
        "not P6's four-fold gate. Funding carry has a different event structure from a "
        "directional bar-classification screen, and copying that partition because it "
        "existed would be borrowing a design rather than choosing one."
    ),
    "partial_block_is_included": (
        "the 2025 partial year is INCLUDED, and its thin-sample status is decided by the "
        "predeclared minimum settlement count rather than by its length. Dropping it after "
        "seeing it would be selection."
    ),
    "forbidden": (
        "splitting a block after seeing a bad period; excluding negative-funding regimes; "
        "slicing one regime finely to manufacture independence; and treating the thousands "
        "of individual funding payments as thousands of independent market experiments — "
        "the BLOCK is the unit of inference, and there are six of them."
    ),
    "inferential_units": 6,
    "per_block_report_fields": (
        "calendar period",
        "funding settlement count",
        "active position duration",
        "opens",
        "closes",
        "rebalances",
        "gross funding received",
        "gross funding paid",
        "basis PnL",
        "fees",
        "slippage",
        "other costs",
        "net return on total capital",
        "maximum adverse excursion",
        "liquidation flag",
        "S4 isolated net return",
        "S4 isolated liquidation flag",
        "S4 isolated liquidation instant",
        "thin sample flag",
        "mark substitution flag",
    ),
}

# ---------------------------------------------------------------------------
# §9  The viability gate — frozen before the first economic number
# ---------------------------------------------------------------------------

MIN_SETTLEMENTS_PER_BLOCK = 200
MIN_INCLUDED_BLOCKS = 5
BREADTH_REQUIRED = 4
BREADTH_OF = 6
WORST_BLOCK_FLOOR = "-0.02"
#: One block's round-trip friction under the frozen cost model. The minimum
#: effect size is anchored here and nowhere else.
MIN_MEAN_NET_RETURN = "0.0025"

VIABILITY_GATE = {
    "denominator": (
        "every block return is block net PnL divided by total_starting_capital, reported as a "
        "fraction. Never one leg's capital."
    ),
    "block_net_pnl": (
        "equity at the block's close MINUS total_starting_capital. The entry fees and "
        "slippage are INSIDE the numerator. Measuring instead from post-entry equity would "
        "hide them — the two bases differ by roughly 12.5 bps of capital, which is precisely "
        "the scale at which G1's strict sign test and G2's strict mean test are decided."
    ),
    "maximum_adverse_excursion": (
        "the most negative value of (equity_t - total_starting_capital) over the holding "
        "period, as a fraction of total capital — the same base as the block return, so the "
        "two are comparable."
    ),
    "horizon_convention": (
        "block returns are PER BLOCK PERIOD and are NOT annualised. G1, G2 and G3 operate on "
        "them unweighted, so the 2025 partial block (4.6 months) carries the same weight in "
        "the mean as a full year, and the -0.02 floor is a looser ANNUALISED bar for it "
        "(-5.2%/yr against -2.0%/yr). This is DECLARED rather than corrected: renormalising "
        "would move a frozen threshold, and the disparity is a property of where the sources "
        "end rather than a choice. The per-block report carries each block's calendar period "
        "and duration so a reader can weight them differently if they wish."
    ),
    "conditions": {
        "G1_breadth": (
            "net block return strictly greater than 0 in at least 4 of the 6 blocks. Two "
            "thirds rather than a bare majority, and deliberately NOT a copy of P6's 3-of-4: "
            "with six blocks, 4-of-6 is a genuine but not extreme bar, and a structural "
            "mechanism that only pays in half its years is not a carry."
        ),
        "G2_central_tendency": "the mean net block return across included blocks is strictly "
        "greater than 0",
        "G3_downside": (
            "the worst block's net return is at least -0.02 of total capital. Tied to the "
            "frozen cost model rather than to any observed return: -0.02 is roughly eight "
            "times a block's round-trip friction, so the rule permits a bad year that "
            "materially exceeds costs while refusing one that is catastrophic for something "
            "described as delta-neutral carry."
        ),
        "G4_sample": (
            "every included block has at least 200 funding settlements, and at least 5 "
            "blocks are included. 200 settlements is about two months of continuous 8-hourly "
            "funding; a block with fewer has not observed enough of a regime to inform "
            "anything."
        ),
        "G5_stress": (
            "the S1 higher-friction stress AND the S3 adverse-basis stress must EACH also "
            "satisfy G1 and G2, and G3 must hold under S1 as well as under S0.\n"
            "S3 is in the gate deliberately. This construction performs exactly one open and "
            "one close per block, so doubling frictions moves a block by a fixed ~0.0025 of "
            "capital once a year — S1 alone is the stress this particular design is "
            "STRUCTURALLY LEAST SENSITIVE TO, and gating on it alone would read as robustness "
            "while buying almost none. S3 perturbs the basis, which is the variable the "
            "payoff identity is actually built on, and is therefore the stress that can "
            "genuinely invalidate the result."
        ),
        "G6_minimum_effect_size": (
            "the mean net block return must exceed 0.0025 — one block's round-trip friction "
            "under the frozen cost model — not merely exceed zero.\n"
            "Without it the gate can emit VIABLE on a mean of a few basis points a year, "
            "which is economically indistinguishable from zero and an order of magnitude "
            "below the frictions already modelled. The floor is derived from the COST MODEL, "
            "which was frozen before any result, and from nothing observed."
        ),
    },
    "conjunction": "ALL of G1, G2, G3, G4, G5 and G6. Any single failure is a FAIL.",
    "the_hurdle_is_zero_yield_cash": (
        "stated so a VIABLE verdict is not read as beating an alternative use of the money. "
        "The gate's hurdle is ZERO — idle capital is explicitly credited no yield anywhere in "
        "this design — so the comparison is against cash under a mattress, not against a "
        "risk-free rate. G6's 0.0025 floor is a FRICTION-derived minimum effect size, not an "
        "economic hurdle rate, and no risk-free series is preregistered. Any caption "
        "accompanying a VIABLE result repeats this."
    ),
    "operating_characteristic": (
        "stated so the gate is not read as stronger than it is. Preregistration protects "
        "against selection; it does not create statistical power. Against a symmetric "
        "coin-flip null over six blocks, G1 alone (at least 4 positive) admits about 34% of "
        "outcomes. G1 is therefore NOT the discriminating condition — the conjunction with "
        "G2, G3, G6 and the two stresses is. With six inferential units this screen can "
        "FALSIFY a mechanism convincingly and can only ever weakly support one, which is why "
        "STOPPING_RULE.on_viable stops the search rather than promoting anything."
    ),
    "tie_handling": {
        "block_return_exactly_zero": "NOT positive. Strict inequality; a zero block does not "
        "count toward G1.",
        "mean_exactly_zero": "FAILS G2. Strict inequality.",
        "mean_exactly_0.0025": "FAILS G6. Strict inequality: the mean must EXCEED the floor.",
        "worst_block_exactly_minus_0.02": "PASSES G3. Inclusive bound.",
        "settlements_exactly_200": "PASSES G4. Inclusive bound.",
    },
    "excluded_blocks": (
        "a block in which the position could not be opened — required source rows absent or "
        "invalid at every candidate instant, the quantity flooring to zero at the step size, "
        "or either leg falling below the venue minimum notional — is EXCLUDED from the "
        "denominators of G1 and G2 and is "
        "reported with its reason. If fewer than 5 blocks remain, the screen is INVALID "
        "rather than PASS or FAIL: too little was measured to decide."
    ),
    "liquidated_blocks": "counted as included blocks, at their realised negative return. Not "
    "excluded.",
    "frozen": (
        "chosen from mechanism and from the frozen cost model, before any P13 economic "
        "number existed. No threshold here may be moved after a result."
    ),
}

#: Scoped to the construction, following the repository's own precedent that a
#: negative is narrower than its headline: standing constraints 5 and 6 in
#: docs/current_development_plan.md record P6's negative as deciding-family-
#: specific and P7's as consensus-v1-specific. An unscoped "STRUCTURAL CARRY:
#: NOT VIABLE" would read as a verdict on carry itself.
RESULT_STATES: tuple[str, ...] = (
    "P13 ALWAYS-ON ANNUAL SPOT/PERP CARRY: VIABLE UNDER THIS EXPLORATORY SCREEN",
    "P13 ALWAYS-ON ANNUAL SPOT/PERP CARRY: NOT VIABLE",
    "P13 ALWAYS-ON ANNUAL SPOT/PERP CARRY: INVALID",
    "P13 ALWAYS-ON ANNUAL SPOT/PERP CARRY: NOT EVALUABLE",
)

WHAT_A_NEGATIVE_WOULD_AND_WOULD_NOT_MEAN = (
    "a NOT VIABLE verdict would say that THIS construction — always-on, one position per "
    "calendar year, quantity-matched, 1x, taker on both legs, on Binance alone — did not "
    "clear this gate on 2020-2025 history. It would NOT say that funding/basis carry is "
    "unprofitable: a shorter holding period, a maker execution assumption, a causal funding "
    "filter, a different venue pair or a different hedge instrument are all untested by it, "
    "and each would be a separate preregistration. The scope of the label is the scope of "
    "the claim."
)

#: Updated only by the commit that produces the governed result.
CURRENT_RESULT_STATE = "P13 ALWAYS-ON ANNUAL SPOT/PERP CARRY: NOT YET RUN"

NOT_EVALUABLE_MEANING = (
    "the preregistered sources could not be obtained, so no economic number exists and none "
    "was estimated. NOT EVALUABLE is a terminal state for this design, not an invitation to "
    "redesign it against whatever data happens to be reachable."
)

# ---------------------------------------------------------------------------
# §10  Stress cases — diagnostics, never a tuning surface
# ---------------------------------------------------------------------------

STRESS_CASES: tuple[dict[str, str], ...] = (
    {
        "id": "S0",
        "name": "base",
        "definition": "the frozen COST_MODEL exactly as written",
    },
    {
        "id": "S1",
        "name": "higher friction",
        "definition": "every fee and slippage rate in COST_MODEL doubled",
        "role": "IN THE GATE, via G5, alongside S3",
    },
    {
        "id": "S2",
        "name": "delayed hedge",
        "definition": (
            "the legs are not simultaneous: one leg is opened one hourly bar after the "
            "other, and closed one bar before it, leaving the position directionally exposed "
            "at each end. Evaluated BOTH WAYS — spot leg first, and perpetual leg first — and "
            "the WORSE of the two is the reported diagnostic value. A one-sided delay is not "
            "a stress: in a rising sample, hedging late is a benefit, and reporting only that "
            "ordering would dress a windfall as a robustness check."
        ),
        "role": "diagnostic only, outside the gate",
    },
    {
        "id": "S3",
        "name": "adverse basis",
        "definition": (
            "the basis is moved AGAINST the position by a fixed 10 bps of spot at each end: "
            "the ENTRY basis is REDUCED by 10 bps of spot (the position is opened at a less "
            "favourable spread) and the EXIT basis is INCREASED by 10 bps of spot (it is "
            "closed at a less favourable one). Since price PnL is Q x (basis_in - basis_out), "
            "both moves subtract, for a total charge of 20 bps of spot notional — about "
            "0.0010 of total capital against S1's roughly 0.0025. The direction is stated "
            "because 'worsened' is not a direction, and the magnitude is frozen and is never "
            "raised after seeing a basis series."
        ),
        "role": (
            "IN THE GATE, via G5, alongside S1. It is the only stress that perturbs the "
            "variable the payoff identity is built on. The 10 bps magnitude is frozen here "
            "and is never revised against any observed basis series."
        ),
    },
    {
        "id": "S4",
        "name": "strict isolated margin, no top-up",
        "definition": (
            "the perpetual leg is margined in isolation with exactly its entry notional at 1x "
            "and receives NO collateral from the spot leg and NO rescue from free cash. Its "
            "funding flows are debited from and credited to the ISOLATED MARGIN BALANCE, not "
            "to the portfolio's cash, and the liquidation test is\n"
            "    perp_margin + perp_unrealised_pnl + cumulative_perp_funding "
            "<= Q x mark x maintenance_margin_rate\n"
            "Routing funding to free cash instead would make the 'strict' case systematically "
            "LENIENT — a short paying funding for months would never feel it in the balance "
            "that is supposed to be walled off — and the whole purpose of S4 is to be the "
            "unforgiving bound."
        ),
        "role": (
            "diagnostic only, and the most important diagnostic here: it quantifies how much "
            "of the primary result depends on the portfolio-margin modelling choice. It is "
            "deliberately OUTSIDE the viability gate because the isolated construction "
            "measures an artifact of wallet segregation rather than the carry, and it is "
            "reported for every block precisely so that claim can be checked rather than "
            "taken on trust."
        ),
    },
)

#: **Every stress above perturbs the COST side. None perturbs the PAYOFF side**,
#: in a sample era whose perpetual funding was historically elevated — which is
#: also the era the leg direction was chosen knowing about. These three are
#: predeclared now, before any result, with frozen definitions, no thresholds and
#: no bearing on the gate, so they add no parameter surface.
PAYOFF_SIDE_DIAGNOSTICS: tuple[dict[str, str], ...] = (
    {
        "id": "D1",
        "name": "funding halved",
        "definition": "every realised funding rate multiplied by 0.5, reported per block and "
        "in aggregate",
        "answers": "how much of the result survives if the era's funding was half as rich. A "
        "carry that only pays at 2020-2024 funding levels is a claim about that era, not "
        "about the mechanism.",
    },
    {
        "id": "D2",
        "name": "leave one block out",
        "definition": "the mean net block return recomputed six times, each omitting one "
        "block, reported as the full set",
        "answers": "whether the aggregate is carried by a single exceptional year. With six "
        "inferential units that is the most likely way a mean misleads.",
    },
    {
        "id": "D3",
        "name": "partition offset",
        "definition": "the identical rule applied to six blocks offset by six months, "
        "truncated at the research boundary",
        "answers": "how sensitive the result is to where the interior block boundaries fall, "
        "which the calendar-year partition fixes arbitrarily.",
    },
)

PAYOFF_DIAGNOSTIC_DISCIPLINE = (
    "D1, D2 and D3 are REPORTED, never gated on, and never used to select anything. A "
    "disappointing base result is not rescued by a flattering diagnostic, and a flattering "
    "base result is not defended by ignoring an unflattering one. They exist because a "
    "reader deciding what a VIABLE verdict is worth needs to know whether it depended on one "
    "year, on one partition, or on the era's funding level."
)

STRESS_DISCIPLINE = (
    "stress cases are diagnostics. They are not a parameter surface, the best of them is "
    "never promoted, and a strategy is never re-specified to survive one. Only S1 and S3 "
    "bear on the verdict, via G5, and that bearing was fixed before the first result. S2 and "
    "S4, and the payoff-side diagnostics below, are reported and never gate anything."
)

# ---------------------------------------------------------------------------
# §11  Stopping rules and prohibitions
# ---------------------------------------------------------------------------

STOPPING_RULE = {
    "on_viable": (
        "STOP the alpha search. A pass establishes only that this preregistered historical "
        "structural mechanism survived one exploratory screen under these stated "
        "assumptions. It authorises no real money, no Styx read, no leverage above 1x, no "
        "live futures routing, no parameter expansion and no promotion. Close it and audit "
        "it first."
    ),
    "on_not_viable": (
        "STOP. Publish the negative. Do not change thresholds, the funding rule, the "
        "rebalance cadence, the costs, the leverage, the venue; do not add market making; do "
        "not write P13-v2 in the task that produced v1; do not inspect Styx or P4-HOLD."
    ),
    "on_invalid": "recorded as invalid, with what was insufficient. Not re-run against a "
    "relaxed rule.",
    "on_not_evaluable": (
        "recorded as NOT EVALUABLE with the acquisition evidence that establishes it. The "
        "design stays frozen and executable; it is not redesigned to fit reachable data."
    ),
}

FORBIDDEN_AFTER_RESULTS: tuple[str, ...] = (
    "moving any cost, fee, slippage or friction",
    "moving any viability-gate threshold, count or floor",
    "changing the capital denominator or the allocation split",
    "changing the hedge ratio, the rebalance policy or the position lifecycle",
    "changing the temporal partition, or re-splitting a block after seeing it",
    "excluding a negative-funding regime or a losing block",
    "raising leverage above 1x",
    "switching venue, instrument or symbol",
    "adding a funding or basis entry filter that was not preregistered",
    "adding market making, or any second strategy, to rescue the first",
    "writing P13-v2 in the task that produced P13",
    "reading P4-HOLD or Styx for any purpose whatsoever",
)

SAFETY_PROHIBITIONS: tuple[str, ...] = (
    "no real money",
    "no authenticated live futures order route, and no expansion of live reachability",
    "no leverage above 1x",
    "no P4-HOLD read",
    "no Styx read",
    "no manufactured historical holdout",
    "no live Freqtrade experiment",
    "no operational PnL selecting alpha",
    "no parameter shopping after results",
    "no rewriting of P1 through P8 frozen economic evidence",
    "Aegis remains the central risk authority",
    "risk-reducing close and emergency flatten paths remain available under halt",
)

PROVENANCE_REQUIREMENT = (
    "primary result generation MUST begin from a clean, committed source tree. A run from a "
    "dirty tree is refused rather than annotated. This exists because P6's fifteen primary "
    "cells recorded dirty: true and their fits are consequently not reconstructible from a "
    "clean checkout; that defect is disclosed in the front-door documents and is not "
    "repeated here."
)

#: What the run must leave behind. A summary JSON is not enough: an independent
#: reviewer has to be able to reimplement the accounting loop from the primary
#: evidence WITHOUT importing this checkpoint's evaluator, and that is only
#: possible if every event is recorded rather than every total.
ARTIFACT_POLICY = {
    "root": "artifacts/benchmark/btc_p13_carry/",
    "decision_aggregate": "artifacts/benchmark/btc_p13_decision/decision.json",
    "primary_evidence": (
        "an event-level ledger, one row per event, covering every position open, every "
        "close, every funding settlement applied, every fee, every slippage charge, every "
        "liquidation test that fired, and the per-instant equity series. It carries the "
        "quantity, both leg prices, the mark, the basis, the signed cash flow and the "
        "resulting equity at each event, so the block totals are RECOMPUTABLE from it rather "
        "than asserted by it."
    ),
    "primary_evidence_has_its_own_manifest": (
        "the event ledger is frozen under its own SHA-256 manifest, separate from the source "
        "manifests. Derived reports — per-block tables, the decision aggregate, any prose "
        "summary — are regenerable FROM the primary evidence and are pinned by regenerating "
        "them and checking what they say, following artifacts/README.md."
    ),
    "source_manifests": (
        "one per acquired archive object, carrying every field in SOURCE_FREEZE_FIELDS. A "
        "source manifest is not the evidence manifest and neither substitutes for the other."
    ),
    "provenance_record": (
        "each artifact records the preregistration hash, the git revision it was generated "
        "at, and whether the tree was dirty. A dirty tree refuses to generate primary "
        "evidence at all, per PROVENANCE_REQUIREMENT."
    ),
    "stress_artifacts": (
        "S1 through S4 are written beside the base result under the same root, each labelled "
        "by stress id. They are diagnostics and are never presented as the result."
    ),
    "not_frozen_only_as_a_summary": (
        "stated explicitly because it is the failure mode this policy exists to prevent: a "
        "checkpoint whose only artifact is a totals JSON cannot be audited, only believed."
    ),
}

TRIPWIRE = (
    "no P13 economic result artifact may exist at the moment this preregistration is "
    "committed. tests/test_p13_preregistration.py asserts that "
    "artifacts/benchmark/btc_p13_decision/decision.json is absent, so a preregistration "
    "commit that silently carried a result would fail the suite."
)

TEST_REQUIREMENTS: tuple[str, ...] = (
    "the suite fails when funding is shifted one settlement into the future",
    "the suite fails when a settlement becomes visible before its instant",
    "the suite fails when the funding sign is reversed",
    "the suite fails when a settlement is applied twice",
    "the suite fails when leverage is applied twice",
    "the suite fails when a fee is computed on quantity instead of notional",
    "the suite fails when either leg's close fee is omitted",
    "the suite fails when percent and basis points are confused",
    "the suite fails when the portfolio denominator is one leg instead of both",
    "the suite fails when a row at or after the research boundary is admitted",
    "the suite fails when a manifest digest is corrupted",
    "the suite fails when a price is non-positive or a precision rule is violated",
    "supportive two-sided controls: a synthetic world in which the gate MUST pass, and one "
    "in which it MUST fail, so the decision function is not tested only against the "
    "committed outcome",
)


# ---------------------------------------------------------------------------
# §15  Which design is active, and which are history
# ---------------------------------------------------------------------------

#: The design a future economic run is governed by. Named in the payload so the
#: answer is machine-readable rather than inferred from a document.
ACTIVE_DESIGN = "P13-A2R1"

#: Every hash this preregistration has ever had, EXCEPT the active one, each
#: marked SUPERSEDED.
#:
#: Written as literals, because the whole value of a superseded hash is that it is
#: not recomputed from the thing that superseded it. Kept INSIDE the hashed payload
#: for one specific reason: it makes the claim "hash X is the active P13 design"
#: falsifiable from the module alone. A future run that quotes a hash listed here
#: is quoting a design this file states is retired, and
#: ``tests/test_p13_preregistration.py`` asserts the active hash is never a member
#: of this tuple.
#:
#: Nothing here is an alternative design. **No P13 economic number, block, gate
#: condition or decision was ever produced under any of them, and none may be
#: produced under any of them now.** What *was* produced under the original is the
#: acquisition evidence in ``artifacts/benchmark/btc_p13_carry/`` — a record of
#: sources that could not be obtained, not an economic result — which keeps the
#: hash it was generated with and is deliberately not rewritten.
SUPERSEDED_HASHES: tuple[dict[str, str], ...] = (
    {
        "design": "P13 original preregistration",
        "hash": "sha256:1369c8828767c04e5b0609fc0125947c91f1cb5f15e977804ff1d1d70fd68767",
        "status": "SUPERSEDED",
        "committed_at": "939f38151cfa607e04c4d74846e081a8ab91ed49",
        "superseded_by": "A1",
        "what_it_governs_now": (
            "historical provenance only, plus the committed acquisition evidence generated "
            "under it. It must not govern new economic evidence"
        ),
    },
    {
        "design": "P13-A1",
        "hash": "sha256:4397109858249c6923b72418d756a3e8504c7cb7abed15deebf300c252f4b099",
        "status": "SUPERSEDED",
        "committed_at": "51f2a9b8e585b75a3d797c7a19953461196eba65",
        "superseded_by": "A2",
        "what_it_governs_now": (
            "historical provenance only. A1's rule itself is NOT withdrawn — "
            "FORCED_CLOSE_WITHOUT_A_FOLLOWING_BAR remains in the hashed payload and in "
            "force — but the A1 HASH is retired, and no P13 economic run may claim it as "
            "the active design"
        ),
    },
    {
        "design": "P13-A2",
        "hash": "sha256:86050b6897143cd0abfa2106ee6cc37baaf3c637e396b23f30bdeeb935fbfe6c",
        "status": "SUPERSEDED",
        "committed_at": "0d023ba5a33bbcae55bb079146838f5f22959607",
        "superseded_by": "A2R1",
        "what_it_governs_now": (
            "historical provenance only, and it is retained rather than quietly dropped "
            "because it is the first committed A2 design and the chronology must show it. "
            "A2's source-validity BEHAVIOUR is not withdrawn — A2R1 keeps every treatment "
            "unchanged — but its claim that the pre-open branch 'can only reduce accrued "
            "funding' was false and is withdrawn. No P13 economic run may claim this hash "
            "as the active design"
        ),
    },
)


def payload() -> dict[str, Any]:
    """The canonical, hashable design. Every decision-relevant constant is here."""
    return {
        "checkpoint": CHECKPOINT,
        "identifier_rationale": IDENTIFIER_RATIONALE,
        "question": QUESTION,
        "hypothesis_class": HYPOTHESIS_CLASS,
        "evidence_ceiling": EVIDENCE_CEILING,
        "hindsight_disclosure": HINDSIGHT_DISCLOSURE,
        "not_p4": NOT_P4,
        "venue_policy": VENUE_POLICY,
        "archive_host": ARCHIVE_HOST,
        "first_party_source_evidence": FIRST_PARTY_SOURCE_EVIDENCE,
        "archive_revision_policy": ARCHIVE_REVISION_POLICY,
        "timestamp_unit_policy": TIMESTAMP_UNIT_POLICY,
        "data_sources": [dict(s) for s in DATA_SOURCES],
        "mark_price_fallback": MARK_PRICE_FALLBACK,
        "funding_column_policy_source": FUNDING_COLUMN_POLICY_SOURCE,
        "source_freeze_fields": list(SOURCE_FREEZE_FIELDS),
        "research_boundary_exclusive": RESEARCH_BOUNDARY_EXCLUSIVE,
        "styx_sealed_instant": STYX_SEALED_INSTANT,
        "data_boundary": DATA_BOUNDARY,
        "funding_causality": FUNDING_CAUSALITY,
        "construction": CONSTRUCTION,
        "execution_price_policy": EXECUTION_PRICE_POLICY,
        "capital_contract": CAPITAL_CONTRACT,
        "rebalance_policy": REBALANCE_POLICY,
        "position_lifecycle": POSITION_LIFECYCLE,
        "venue_constraints": VENUE_CONSTRAINTS,
        "cost_model": COST_MODEL,
        "funding_semantics": FUNDING_SEMANTICS,
        "basis_definition": BASIS_DEFINITION,
        "margin_and_liquidation": MARGIN_AND_LIQUIDATION,
        "forced_close_without_a_following_bar": FORCED_CLOSE_WITHOUT_A_FOLLOWING_BAR,
        "markless_liquidation_validity_policy": MARKLESS_LIQUIDATION_VALIDITY_POLICY,
        "markless_states": list(MARKLESS_STATES),
        "temporal_partition": TEMPORAL_PARTITION,
        "min_settlements_per_block": MIN_SETTLEMENTS_PER_BLOCK,
        "min_included_blocks": MIN_INCLUDED_BLOCKS,
        "min_mean_net_return": MIN_MEAN_NET_RETURN,
        "breadth_required": BREADTH_REQUIRED,
        "breadth_of": BREADTH_OF,
        "worst_block_floor": WORST_BLOCK_FLOOR,
        "viability_gate": VIABILITY_GATE,
        "result_states": list(RESULT_STATES),
        "current_result_state": CURRENT_RESULT_STATE,
        "what_a_negative_would_and_would_not_mean": WHAT_A_NEGATIVE_WOULD_AND_WOULD_NOT_MEAN,
        "not_evaluable_meaning": NOT_EVALUABLE_MEANING,
        "stress_cases": [dict(s) for s in STRESS_CASES],
        "stress_discipline": STRESS_DISCIPLINE,
        "payoff_side_diagnostics": [dict(d) for d in PAYOFF_SIDE_DIAGNOSTICS],
        "payoff_diagnostic_discipline": PAYOFF_DIAGNOSTIC_DISCIPLINE,
        "stopping_rule": STOPPING_RULE,
        "forbidden_after_results": list(FORBIDDEN_AFTER_RESULTS),
        "safety_prohibitions": list(SAFETY_PROHIBITIONS),
        "provenance_requirement": PROVENANCE_REQUIREMENT,
        "artifact_policy": ARTIFACT_POLICY,
        "tripwire": TRIPWIRE,
        "test_requirements": list(TEST_REQUIREMENTS),
        "active_design": ACTIVE_DESIGN,
        "superseded_hashes": [dict(h) for h in SUPERSEDED_HASHES],
    }


def preregistration_hash() -> str:
    blob = json.dumps(payload(), sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(blob.encode()).hexdigest()


def describe() -> dict[str, Any]:
    return {"preregistration_hash": preregistration_hash(), **payload()}


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(describe(), indent=2))
