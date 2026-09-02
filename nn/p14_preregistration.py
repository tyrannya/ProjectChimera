"""P14's design, committed as a design. **P14 IS NOT OPENED.**

No signal has been computed, no fold has been scored, no P14 number exists, and
:data:`OUTCOME` says so. The design is committed now because that is the only
moment at which a trade-flow rule can be fixed without being fitted to its answer.

**What P14 asks.** Whether the *sign* of aggressive trade flow, measured on one
closed Binance spot BTCUSDT 1-minute bar, carries causal directional information
about the next 1-minute bar — and, only if it does, whether that information
survives the same 20 bps round-trip cost model every checkpoint since v4 has paid.

**Why this axis.** `P3` folded the same trade tape into *hourly* sufficient
statistics and evaluated them under the 1h/6h programme; its own closure says in
as many words that it did not test short-horizon trade flow at its own clock.
`P6` fitted specialists on 1m bars but gave them only the fourteen OHLCV columns.
The cell {fast clock} x {trade flow} has never been read. P14 reads it once.

**The one thing that is genuinely new here is not the clock and not the source.**
It is that the signal is fixed by an *external* published claim rather than by
this repository's taste: Silantyev (2019) reports that trade flow imbalance
explains contemporaneous price change in a BTC market better than aggregate order
flow imbalance does. That claim is used to fix the mechanism, the interval and the
sign *before* any ProjectChimera number exists, and it is used as a **positive
control on the construction** — never as evidence that the causal question comes
out positive. The anchor's claim is contemporaneous. P14's question is causal.
Those are different claims and the design keeps them apart.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

CHECKPOINT = "P14"

#: The one thing about P14 that is currently true.
OUTCOME = "NOT OPENED"

QUESTION = (
    "Does mechanically defined signed trade-flow imbalance, measured on one closed Binance "
    "spot BTCUSDT 1-minute bar, carry causal directional information about the next "
    "1-minute bar beyond a no-information floor -- and, only if it does, does that "
    "information survive the frozen 20 bps round-trip cost model?"
)

PRIMARY_HYPOTHESIS = (
    "H1: sign(tfi_ratio) on the closed bar t agrees with sign(the next bar's close-to-close "
    "return) more often than the same fold's best constant-direction rule, in at least 3 of "
    "the 4 outer blocks and on average. No hypothesis is asserted about which way this "
    "comes out."
)

# --------------------------------------------------------------------------- #
# 1. The external replication anchor
# --------------------------------------------------------------------------- #

#: One anchor. Not a synthesis of several, and not the most profitable one found.
EXTERNAL_ANCHOR = {
    "citation": (
        "Silantyev, E. (2019). Order flow analysis of cryptocurrency markets. "
        "Digital Finance 1(1-4), 191-218. https://doi.org/10.1007/s42521-019-00007-w"
    ),
    "market_studied": "BitMEX XBTUSD perpetual contract, trade and quote data",
    "qualitative_claim": (
        "trade flow imbalance explains contemporaneous price change better than aggregate "
        "order flow imbalance does, and contemporaneous price change is linearly related to "
        "flow imbalance over large enough intervals"
    ),
    "intervals_studied": "1 second to 1 hour; 1 minute is among the intervals presented",
    "why_this_anchor": (
        "it is peer-reviewed, it is about Bitcoin rather than equities, and its trade-flow "
        "construction needs only a trade tape with an aggressor flag -- which is exactly "
        "what Binance publishes"
    ),
    "full_text_was_not_read": (
        "the article is paywalled and this session could not read it. The qualitative claim, "
        "the market and the interval range above are taken from the publisher's own summary "
        "and from independent citing literature. NO NUMERIC VALUE FROM THE ARTICLE IS A "
        "TARGET, and none is quoted. The algebra below is therefore frozen by ProjectChimera "
        "as its own definition rather than presented as a quotation."
    ),
    "unavoidable_adaptations": (
        "venue and instrument: BitMEX XBTUSD -> Binance spot BTCUSDT, because every "
        "checkpoint in this repository reads Binance BTCUSDT and switching venue would "
        "change two things at once;",
        "dependent variable: mid-price change from quote data -> close-to-close return of "
        "the 1m kline, because Binance publishes no historical spot quote archive and a mid "
        "price is therefore unobtainable;",
        "order flow imbalance is NOT reproduced at all, because it needs book data this "
        "source does not carry. Only the trade-flow half is reproduced, and no claim about "
        "the OFI-vs-TFI ordering is made or tested;",
        "the anchor's claim is CONTEMPORANEOUS. P14's primary question is CAUSAL "
        "(next bar). The anchor is not evidence about the causal question.",
    ),
    "fixed_before_outcomes": (
        "every adaptation above was written into this payload before any P14 statistic "
        "existed, and none may be revisited to explain a result"
    ),
}

# --------------------------------------------------------------------------- #
# 2. Source
# --------------------------------------------------------------------------- #

INSTRUMENT = "binance spot BTCUSDT"

#: One archive family. The same publisher, the same objects and the same layout
#: `multiclock_v1` already acquired and checksum-verified for P6 -- P14 adds no
#: venue, no market, no instrument and no second source.
SOURCE = {
    "publisher": "Binance public data archive",
    "canonical_base_url": "https://data.binance.vision",
    "layout": "data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-YYYY-MM.zip",
    "digest_source": "the .zip.CHECKSUM object Binance publishes beside each archive",
    "months": "2020-01 through 2025-05, inclusive: 65 monthly objects",
    "columns_used": (
        "open_time (column 0)",
        "close (column 4)",
        "volume (column 5)",
        "taker_buy_base_asset_volume (column 9)",
    ),
    "timestamp_units": (
        "resolved per file from the magnitude of open_time, exactly as multiclock_v1 does: "
        "60 months are milliseconds and 5 are microseconds"
    ),
    "price_grid": (
        "data/research/btc_usdt_multiclock_gen2_1m_pre_boundary.parquet -- the committed, "
        "already-verified 1m source P6 was cut from. P14 adds one column to the rows that "
        "file already has and creates no new bar."
    ),
    "no_other_source": (
        "no aggTrades stream enters the frozen construction, no order book, no futures, no "
        "funding, no open interest, no second venue and no second symbol"
    ),
}

#: Why the taker-side split is used instead of re-summing the raw aggTrades tape.
#: This is a SOURCE decision, taken before any P14 statistic existed, on
#: reproducibility and gap-surface grounds -- never on an outcome.
SOURCE_SUBSTITUTION = {
    "what": (
        "signed trade flow is read from the kline archive's taker_buy_base_asset_volume "
        "rather than recomputed by summing aggTrades quantity where is_buyer_maker is false"
    ),
    "it_is_an_identity_not_an_approximation": (
        "verified against the publisher's own trade tape on three days spanning both "
        "timestamp eras -- 2020-01-15, 2023-06-15 and 2025-03-14, 2,694,256 aggregated "
        "trades, 4,320 of 4,320 minutes agreeing to within float64 summation noise "
        "(maximum relative difference 1.3e-14) for both volume and taker-buy volume"
    ),
    "why": (
        "the derived column then lives on exactly the rows the committed price grid already "
        "has, so the information family has no gap surface of its own. That is the P13 "
        "failure mode -- an archive object existing while the rows the design needs do not "
        "-- made structurally impossible rather than merely checked."
    ),
    "and_why_else": (
        "142 MB of archives a reader can re-derive every feature value from, instead of "
        "48.75 GB and a 3.4-billion-row streaming step"
    ),
}

RESEARCH_BOUNDARY = "2025-05-19T08:00:00+00:00"

BOUNDARY_RULE = (
    "exclusive. No row at or after it is read, exactly as multiclock_v1 requires. The "
    "boundary is the first instant of the retired P4-HOLD region and it does not move."
)

# --------------------------------------------------------------------------- #
# 3. The signal -- one column, no window, no constant to search
# --------------------------------------------------------------------------- #

SIGNAL_NAME = "tfi_ratio"

SIGNAL_DEFINITION = (
    "tfi_ratio(t) = (2 * taker_buy_base_asset_volume(t) - volume(t)) / volume(t), evaluated "
    "on the bar opening at t once that bar has closed. Equivalently "
    "(aggressive buy volume - aggressive sell volume) / (total volume), in [-1, +1]."
)

SIGN_CONVENTION = (
    "Binance publishes taker_buy_base_asset_volume as the base volume in which the BUYER "
    "was the TAKER, i.e. aggressive buying. The complement, volume - taker_buy_base, is "
    "aggressive selling. This is the same convention the committed P3 trade snapshot "
    "records for aggTrades -- is_buyer_maker false is an aggressive BUY -- and it is the "
    "convention that reconciles the two archives exactly in SOURCE_SUBSTITUTION."
)

SIGNAL_PROPERTIES = (
    "one column, not a family",
    "no lookback window, so no window length to search",
    "no clip, no z-score, no smoothing, no normalising constant",
    "scale-free and bounded in [-1, +1], so no volume trend has to be detrended",
    "strictly causal: it reads the bar it is stamped on and nothing later",
    "a bar with volume == 0 has tfi_ratio 0 and is excluded from every denominator",
)

#: The direction is fixed HERE, before any number. A negative verdict may not be
#: rescued by discovering that the opposite sign would have worked.
SIGNAL_DIRECTION = (
    "positive tfi_ratio is predicted to precede a positive return, matching the anchor's "
    "sign. Flipping the sign after seeing a result is forbidden by FORBIDDEN_AFTER_RESULTS."
)

# --------------------------------------------------------------------------- #
# 4. Clock, horizon, target
# --------------------------------------------------------------------------- #

CLOCK = "1m"

CLOCK_REASON = (
    "fixed ex ante by three converging reasons, none of them an outcome: the anchor presents "
    "a 1-minute interval, 1m is the native resolution of the committed source every clock in "
    "this repository is cut from, and 1m is the fastest clock P6 established fold geometry "
    "for. No other clock is evaluated and no clock is chosen from a result."
)

HORIZON_BARS = 1

HORIZON_REASON = (
    "one native bar. The anchor's mechanism is defined at the interval scale; extending it "
    "to six bars would test persistence, which the anchor does not claim. This deliberately "
    "departs from the six-native-bar convention v4 through P6-EXT used, and the departure is "
    "fixed here rather than chosen later. Horizon families are P9 and are forbidden here."
)

#: One definition of "the return of bar k", used by every stage so that the
#: contemporaneous control and the causal gate cannot drift apart:
#:
#:     r(k) = close(k) / close(k-1) - 1
#:
#: which is the return realised OVER bar k, since the bar opening at k spans
#: [k, k+1min) and close(k-1) is the last trade before it. Stage 0 pairs
#: tfi_ratio(t) with r(t) -- the same bar. Stage 1 pairs tfi_ratio(t) with
#: r(t+1) -- the next bar, which is the only causal pairing.
RETURN_DEFINITION = "r(k) = close(k) / close(k-1) - 1, the return realised over bar k"

TARGET = (
    "r(t+1), the return realised over the single 1m bar following the decision bar, as "
    "RETURN_DEFINITION fixes it. There is NO three-class label and no class threshold: "
    "introducing one would add a constant the anchor does not fix. The target enters the "
    "primary gate through its sign and the economic screen through "
    "nn.evaluate.realised_trades."
)

TARGET_EXCLUSIONS = (
    "a decision bar whose immediately following minute is absent from the source is excluded "
    "-- the segment discipline nn.data_pipeline already applies, so no target crosses a "
    "market-data gap",
    "a decision bar with r(t+1) exactly 0 is excluded from the agreement denominators",
    "a decision bar with tfi_ratio exactly 0 is excluded from the agreement denominators",
    "the last decision bar of each block has no successor and is excluded",
)

# --------------------------------------------------------------------------- #
# 5. Folds
# --------------------------------------------------------------------------- #

FOLDS = (
    "the same four real-world temporal periods every checkpoint since P2b has read, mapped "
    "by timestamp and never by row number, from nn.p2b.plan_from_manifest over the committed "
    "1h snapshot -- identical to the instants p6_preregistration.md section 6 froze"
)

OUTER_BLOCKS = (
    ("2023-03-04T07:00:00+00:00", "2023-09-24T17:00:00+00:00"),
    ("2023-09-24T17:00:00+00:00", "2024-04-12T14:00:00+00:00"),
    ("2024-04-12T14:00:00+00:00", "2024-10-30T11:00:00+00:00"),
    ("2024-10-30T11:00:00+00:00", "2025-05-19T08:00:00+00:00"),
)

INNER_BLOCKS = (
    ("2022-08-15T10:00:00+00:00", "2023-03-04T07:00:00+00:00"),
    ("2023-03-04T07:00:00+00:00", "2023-09-24T17:00:00+00:00"),
    ("2023-09-24T17:00:00+00:00", "2024-04-12T14:00:00+00:00"),
    ("2024-04-12T14:00:00+00:00", "2024-10-30T11:00:00+00:00"),
)

FOLD_RULES = (
    "four periods, whatever the row count. Rows are not folds, and the periods are never "
    "subdivided to manufacture independence",
    "no training block is used at all: the signal fits nothing",
    "the only fitted quantity in the whole checkpoint is the economic screen's threshold, "
    "selected on the INNER block and never on the outer",
    "purge and embargo: one bar, applied by nn.dataset.sample_indices' horizon embargo at "
    "horizon 1, so no inner row's target is drawn from the outer block",
)

# --------------------------------------------------------------------------- #
# 6. Stage 0 -- mechanism-presence control
# --------------------------------------------------------------------------- #

#: A check that has never failed is not evidence. This one can fail, and if it
#: does the checkpoint forfeits rather than reporting a verdict.
STAGE0 = {
    "name": "mechanism-presence control",
    "decision_set": (
        "D0(fold) = {t in the outer block : bar t exists, tfi_ratio(t) != 0, r(t) != 0}. The "
        "statistic and its baseline are BOTH measured over D0."
    ),
    "what": (
        "the CONTEMPORANEOUS agreement rate between sign(tfi_ratio(t)) and sign(r(t)) -- the "
        "return realised over the SAME bar the flow was measured on -- over D0, per outer "
        "block"
    ),
    "baseline": (
        "the same block's best constant-direction rule over D0: "
        "max(#{t in D0 : r(t) > 0}, #{t in D0 : r(t) < 0}) / |D0|"
    ),
    "pass_condition": (
        "contemporaneous agreement exceeds that baseline in ALL FOUR outer blocks"
    ),
    "why_it_is_here": (
        "it is the anchor's own claim, and it is the only part of the anchor P14 expects to "
        "reproduce. If aggressive buying does not coincide with the price rising in this data "
        "then the construction, the sign convention or the source is wrong, and no causal "
        "statistic computed on top of it means anything."
    ),
    "it_is_not_evidence_about_alpha": (
        "same-bar agreement is not available at the decision instant and can never enter a "
        "model, a feature, a threshold or a trading rule. It cannot make P14 positive and it "
        "is reported as a control, never as a result."
    ),
    "on_failure": (
        "P14 terminates NOT EVALUABLE, screen-wide. The disposition is explicitly ambiguous "
        "between a construction defect and a genuine absence of the mechanism in this market, "
        "and an independent reviewer must settle which before any re-run. Stage 1 is not "
        "reached and no causal statistic is computed."
    ),
}

# --------------------------------------------------------------------------- #
# 7. Stage 1 -- the predictive gate (primary)
# --------------------------------------------------------------------------- #

STAGE1 = {
    "name": "causal predictive gate",
    "decision_set": (
        "D(fold) = {t in the outer block : bar t exists, bar t+1 exists and is the "
        "immediately next minute, tfi_ratio(t) != 0, r(t+1) != 0}. The statistic and its "
        "baseline are BOTH measured over D, so a difference in exclusions can never be "
        "mistaken for a difference in skill."
    ),
    "statistic": ("A(fold) = #{t in D : sign(tfi_ratio(t)) == sign(r(t+1))} / |D|"),
    "baseline": (
        "B(fold) = max(#{t in D : r(t+1) > 0}, #{t in D : r(t+1) < 0}) / |D| -- the best a "
        "constant-direction rule could have done on that block. It is a hindsight floor, "
        "deliberately the hardest fair one, in the spirit of P7's fold-wise-best benchmark."
    ),
    "conditions": (
        "A > B in at least 3 of the 4 outer blocks",
        "mean(A - B) across the four blocks > 0",
    ),
    "model": (
        "none. There is no model family, no fit, no seed and no hyperparameter anywhere in "
        "Stage 1 -- the statistic is a sign agreement over published arithmetic. This is the "
        "'direct statistical signal test' option, chosen over a logistic regression because "
        "it removes the model-family choice entirely."
    ),
    "why_not_the_p6_1m_cells": (
        "P6's 1m cells are NOT a control for this. They were fitted on the fourteen OHLCV "
        "columns at a six-bar horizon, so neither their predictions nor their returns are "
        "commensurable with a one-bar trade-flow rule. They may not be used to contextualise, "
        "rescue or discount a P14 verdict, and the secondary P6 logistic-regression and "
        "LightGBM leads are not promoted by anything here."
    ),
    "what_passing_does_not_mean": (
        "Each outer block holds on the order of 1e5 to 1e6 decision rows, so an edge far too "
        "small to pay 20 bps would still clear A > B. Stage 1 is deliberately a PERMISSIVE "
        "FILTER, not a finding: its only job is to stop a dead signal from reaching the cost "
        "model. A Stage 1 pass is NOT evidence of alpha, is NOT reportable as a positive "
        "result on its own, and licenses exactly one thing -- running Stage 2."
    ),
    "on_failure": (
        "P14 is NEGATIVE and the economic screen is NEVER RUN. A signal that cannot beat a "
        "constant direction does not get a second chance at a cost model."
    ),
}

# --------------------------------------------------------------------------- #
# 8. Stage 2 -- the economic screen, reached only through Stage 1
# --------------------------------------------------------------------------- #

STAGE2_GATED_ON_STAGE1 = (
    "Stage 2 runs if and only if Stage 0 passed and Stage 1 passed. There is no branch in "
    "which a failed predictive gate is followed by an economic evaluation, and no reading of "
    "this document licenses one."
)

TRADING_RULE = (
    "action(t) = LONG if tfi_ratio(t) > +theta, SHORT if tfi_ratio(t) < -theta, else HOLD. "
    "Entered at the close of bar t and held for exactly HORIZON_BARS, with non-overlapping "
    "trades resolved by nn.evaluate.realised_trades, which is the single definition in this "
    "repository of what trades a signal took and what each netted."
)

THRESHOLD_SELECTION = {
    "grid": "theta in 0.05, 0.10, ..., 0.95 -- 19 predeclared points, fixed here",
    "selected_on": "the INNER block only, never the outer",
    "objective": (
        "maximise net return after the round-trip cost, the objective every prior cell used"
    ),
    "min_trades": 10,
    "one_theta_per_fold": "selected independently per fold on that fold's inner block",
}

COSTS = {
    "fee_rate": 0.0005,
    "slippage_rate": 0.0005,
    "cost_threshold": 0.002,
    "applied": "once per realised trade, by nn.evaluate.realised_trades",
    "unchanged": (
        "identical to v4, P2a, P2b, P2c, P3, P4, P5, P6 and P6-EXT. P14 introduces no new "
        "cost model, and a cost model expressed per trade does not become cheaper because "
        "trades are shorter."
    ),
    "no_funding_no_margin_no_liquidation": (
        "the instrument is spot and the position is unlevered, so there is no funding "
        "cashflow, no margin and no liquidation to account for. P14 changes the information "
        "set, not the instrument."
    ),
}

#: The programme-wide limitation, restated rather than quietly inherited.
SHORT_LEG_DISCLOSURE = (
    "A SHORT action is not executable on Binance spot, and never has been in this "
    "repository: every economic number from v4 onward prices a synthetic long/short spot "
    "instrument. Futures Execution v1 exists to close that gap operationally and is dry-run "
    "only. P14 does not close it and does not claim to. The instrument question -- whether "
    "this machinery behaves differently when the market modelled and traded is the USD-M "
    "perpetual -- is a separate checkpoint that this one deliberately does not open."
)

VIABILITY_GATE = (
    "cost-aware outer net return > 0 in at least 3 of the 4 folds",
    "mean outer cost-aware net return across the four folds > 0",
    "beats the native 1m momentum baseline's outer net return in at least 3 of the 4 folds",
)

VIABILITY_GATE_PROVENANCE = (
    "these three conditions are P6's viability gate, section 9.2, imported verbatim rather "
    "than invented now. Importing a gate frozen before P6's first fit is what stops a gate "
    "from being tuned to a signal that already exists."
)

MOMENTUM_BASELINE_IS_NOT_AN_INPUT = (
    "condition 3 needs ema_cross on the 1m clock, which nn.benchmark.fit_baselines resolves "
    "from chimera.features. Computing it does NOT widen P14's information set: tfi_ratio "
    "remains the only column any P14 rule reads, and no OHLCV column may enter the signal, "
    "the threshold or the trading rule."
)

#: Disclosed here rather than discovered after a result.
CONDITION_3_IS_A_WEAK_DISCRIMINATOR = (
    "P6's own closure records that on the fast clocks the native momentum baseline returns "
    "about -1.000 -- it takes a position on the sign of ema_cross at every bar and pays 20 "
    "bps a hundred times a fold -- so 'beats momentum' at 1m means 'does not trade itself to "
    "death'. It is a real floor and a very low one. A Stage 2 pass must NEVER be reported as "
    "having cleared three demanding conditions. Condition 1 is what binds, as it did in P6."
)

#: P7's disclosed failure mode, applied prospectively instead of retrofitted.
MINIMUM_ACTIVITY = {
    "outer_trades_per_fold": 30,
    "rule": (
        "a Stage 2 PASS requires at least 30 realised outer trades in EVERY one of the four "
        "folds. A fold below the floor is flagged THIN and Stage 2 is reported "
        "INSUFFICIENT_ACTIVITY, which is not a pass."
    ),
    "why_30": (
        "three times select_threshold's own min_trades floor, fixed here because P7 realised "
        "13 trades across four folds -- one of them zero -- and the audit required a minimum "
        "effective trade count to be fixed in a NEW preregistration before new evidence, not "
        "retrofitted into an old one. Any number is arbitrary; this one is arbitrary in "
        "advance."
    ),
}

# --------------------------------------------------------------------------- #
# 9. Reporting, multiplicity and the ceiling
# --------------------------------------------------------------------------- #

MULTIPLICITY = (
    "one signal, one clock, one horizon, one sign, one primary gate",
    "no model-family tournament, no second signal, no feature family, no second venue",
    "every fold's Stage 0, Stage 1 and Stage 2 number is published whatever it says",
    "there is no best-fold line, no best-threshold line and no summary row that hides a fold",
)

EVIDENCE_CEILING = (
    "Exploratory and adaptive, and more so than any checkpoint before it. These are the same "
    "four real-world blocks v4, P2a, P2b, P2c, P3, P4, P5, P6, P6-EXT and P7 have already "
    "read. No P14 result is confirmatory. A positive P14 is a CANDIDATE and nothing else: it "
    "would still have to survive a frozen architecture, sustained genuinely-future paper "
    "validation, a mature-system freeze, and only then a separate decision about whether the "
    "hindsight-era-capped Styx adds anything, before any separately authorised very small "
    "live allocation could even be discussed. A negative P14 needs no discounting, which is "
    "the asymmetry that makes negative results the cheap ones to trust."
)

RESULT_STATES: tuple[str, ...] = (
    "P14 NATIVE 1m TRADE-FLOW SCREEN: NOT YET RUN",
    "P14 MECHANISM-PRESENCE CONTROL FAILED: NOT EVALUABLE",
    "P14 PREDICTIVE GATE FAILED: NEGATIVE",
    "P14 PREDICTIVE GATE PASSED, ECONOMIC SCREEN FAILED: NEGATIVE ON TRADABILITY",
    "P14 PREDICTIVE GATE PASSED, ECONOMIC SCREEN INSUFFICIENT_ACTIVITY: NOT EVALUABLE",
    "P14 PREDICTIVE GATE PASSED, ECONOMIC SCREEN PASSED: EXPLORATORY CANDIDATE",
)

CURRENT_RESULT_STATE = "P14 NATIVE 1m TRADE-FLOW SCREEN: NOT YET RUN"

FORBIDDEN_AFTER_RESULTS: tuple[str, ...] = (
    "flipping the sign convention because the opposite direction would have worked",
    "changing the clock, the horizon, the target or the exclusion rules",
    "changing the signal definition, adding a window, a clip, a z-score or a second column",
    "changing the theta grid, the threshold objective, the cost model or any gate condition",
    "changing the minimum-activity floor after seeing a trade count",
    "adding a model family, a second signal, regime conditioning or adaptive retraining",
    "running the economic screen after a failed predictive gate, under any pretext",
    "switching to the aggTrades tape, another venue, another symbol or another market "
    "because the result disappoints",
    "re-reading the P6 secondary logistic-regression or LightGBM cells as support",
    "opening P8, reading P4-HOLD, or approaching Styx",
    "writing a tradeflow_v2 in the task that produced v1's result",
)

SAFETY_PROHIBITIONS: tuple[str, ...] = (
    "no real money and no live allocation, tiny or otherwise",
    "no authenticated order route is created, enabled or approached",
    "no leverage above 1x is contemplated anywhere in this design; the instrument is spot "
    "and the position is unlevered",
    "Aegis remains the sole risk authority",
    "P4-HOLD stays retired and unread; Styx stays sealed",
)

STOPPING_RULE = {
    "on_not_evaluable": (
        "recorded as a source or construction disposition, never as evidence about the "
        "market. No parameter is rescued and no A2-style amendment is written against "
        "coverage that is already visible."
    ),
    "on_negative": (
        "recorded as negative and left visible. The correct response is a different "
        "question, not a second signal against these same four blocks."
    ),
    "on_candidate": (
        "adaptive evidence that one mechanically defined trade-flow rule cleared a "
        "cost-aware floor on burned blocks. Not a deployable strategy, not permission for "
        "real money, and not a reason to open P8."
    ),
    "on_not_opened": "nothing has been measured. This is the current state.",
}


def payload() -> dict[str, Any]:
    return {
        "checkpoint": CHECKPOINT,
        "outcome": OUTCOME,
        "question": QUESTION,
        "primary_hypothesis": PRIMARY_HYPOTHESIS,
        "external_anchor": EXTERNAL_ANCHOR,
        "instrument": INSTRUMENT,
        "source": SOURCE,
        "source_substitution": SOURCE_SUBSTITUTION,
        "research_boundary": RESEARCH_BOUNDARY,
        "boundary_rule": BOUNDARY_RULE,
        "signal_name": SIGNAL_NAME,
        "signal_definition": SIGNAL_DEFINITION,
        "sign_convention": SIGN_CONVENTION,
        "signal_properties": list(SIGNAL_PROPERTIES),
        "signal_direction": SIGNAL_DIRECTION,
        "clock": CLOCK,
        "clock_reason": CLOCK_REASON,
        "horizon_bars": HORIZON_BARS,
        "horizon_reason": HORIZON_REASON,
        "return_definition": RETURN_DEFINITION,
        "target": TARGET,
        "target_exclusions": list(TARGET_EXCLUSIONS),
        "folds": FOLDS,
        "outer_blocks": [list(b) for b in OUTER_BLOCKS],
        "inner_blocks": [list(b) for b in INNER_BLOCKS],
        "fold_rules": list(FOLD_RULES),
        "stage0": STAGE0,
        "stage1": STAGE1,
        "stage2_gated_on_stage1": STAGE2_GATED_ON_STAGE1,
        "trading_rule": TRADING_RULE,
        "threshold_selection": THRESHOLD_SELECTION,
        "costs": COSTS,
        "short_leg_disclosure": SHORT_LEG_DISCLOSURE,
        "viability_gate": list(VIABILITY_GATE),
        "viability_gate_provenance": VIABILITY_GATE_PROVENANCE,
        "momentum_baseline_is_not_an_input": MOMENTUM_BASELINE_IS_NOT_AN_INPUT,
        "condition_3_is_a_weak_discriminator": CONDITION_3_IS_A_WEAK_DISCRIMINATOR,
        "minimum_activity": MINIMUM_ACTIVITY,
        "multiplicity": list(MULTIPLICITY),
        "evidence_ceiling": EVIDENCE_CEILING,
        "result_states": list(RESULT_STATES),
        "current_result_state": CURRENT_RESULT_STATE,
        "forbidden_after_results": list(FORBIDDEN_AFTER_RESULTS),
        "safety_prohibitions": list(SAFETY_PROHIBITIONS),
        "stopping_rule": STOPPING_RULE,
    }


def preregistration_hash() -> str:
    blob = json.dumps(payload(), sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(blob.encode()).hexdigest()


def describe() -> dict[str, Any]:
    return {"preregistration_hash": preregistration_hash(), **payload()}


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(describe(), indent=2))
