"""P7's design, frozen after P6 closed and before any P7 number exists.

The machine-readable twin of ``docs/p7_preregistration.md``. Every constant a
result could be argued into or out of lives here, :func:`preregistration_hash`
covers all of them, and every P7 artifact records it.

**P7 fits nothing.** It replays the frozen P6 XGBoost specialists' committed
per-sample predictions. No specialist is refitted, no threshold is retuned, no
weight is searched, and the specialist set for each mode is fixed by the
architecture rather than chosen from P6's results — which is the point, because
P6 was negative and a P7 that picked its specialists after seeing that would be
selecting on the outcome it is about to be compared against.

**Two modes, two verdicts.** Scalping and day trading are separate experiments
reported separately. "Both supportive", "one supportive", and "neither
supportive" are all real answers and none of them is collapsed into a
winner-selection.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from nn.p6_preregistration import PRIMARY_MODEL as P6_PRIMARY_MODEL
from nn.p6_preregistration import preregistration_hash as p6_hash

CHECKPOINT = "P7"

QUESTION = (
    "Does causal agreement between independently trained temporal specialists add robust "
    "cost-aware value beyond the corresponding individual specialists?"
)

HYPOTHESIS = (
    "Unknown, and deliberately not asserted. P6 found no individually viable clock, which "
    "makes the consensus question more interesting rather than less: agreement is a "
    "different object from any of its members, and a filter that trades only where "
    "independently fitted models concur could in principle be better or worse than each of "
    "them. Neither direction is claimed here."
)

EVIDENCE_CEILING = (
    "Exploratory, adaptive, and a rung lower than P6's. These are the same four real-world "
    "windows v4, P2a, P2b, P2c, P3, P4, P5 and P6 have read — the ninth reading — and P7 "
    "is designed with P6's results already known. No P7 result is confirmatory. A positive "
    "P7 would need confirmation these blocks cannot supply; a negative one needs no "
    "discounting."
)

RESEARCH_CLASSIFICATION = (
    "Two mode experiments over frozen predictions. P7 selects nothing for deployment; a "
    "supportive mode has earned a scaffold that can express it, not a claim of "
    "profitability, and not permission to trade real money."
)

# --------------------------------------------------------------------------- #
# 1. The specialists
# --------------------------------------------------------------------------- #

#: What P7 replays: the deciding family's frozen P6 cells and their committed
#: per-sample outer predictions. Named as a dependency rather than re-derived,
#: so a P7 artifact carries the identity of the P6 design it stands on.
SPECIALIST_SOURCE = {
    "checkpoint": "P6",
    "model": P6_PRIMARY_MODEL,
    "p6_preregistration_hash": p6_hash(),
    "cells": "artifacts/benchmark/btc_p6_{clock}_xgboost/",
    "predictions": "outer_predictions.parquet",
    "frozen_under": "artifacts/btc_p6_SHA256SUMS.txt",
    "column": "selected_action",
    "semantics": (
        "each specialist's own predetermined decision, produced by its own frozen "
        "threshold on its own inner block during P6. P7 reads the action and never the "
        "probabilities, so there is no second threshold anywhere in this checkpoint."
    ),
}

REFIT_PROHIBITION = (
    "No specialist is refitted, no threshold is re-selected, no probability is re-scored "
    "and no weight is searched. P7 reads committed predictions. A P7 that refitted "
    "anything would be measuring a different specialist from the one P6 published."
)

SPECIALIST_SET_IS_ARCHITECTURAL = (
    "Each mode's specialist set is fixed by what the mode *is* — a scalping decision reads "
    "fast clocks and a day-trading decision reads intraday ones — and not by P6's results. "
    "P6 was negative on all five clocks, so there was no attractive subset to pick even if "
    "picking were allowed, and P6 section 9.4 predeclared that P7 must not depend on which "
    "specialists cleared an absolute floor."
)

# --------------------------------------------------------------------------- #
# 2. Causal alignment
# --------------------------------------------------------------------------- #

#: The instant a decision-clock row's decision is taken, and therefore the
#: instant that decides which specialist predictions exist yet.
#:
#: A row's trade is entered at its **close**: the label is
#: ``close[t + 6] / close[t] - 1``, so the position is taken at the close of bar
#: ``t``. Everything that closed at or before that instant is available, and
#: nothing else is.
ALIGNMENT = {
    "reference_instant": "the decision bar's close, i.e. its open plus one decision-clock bar",
    "why": (
        "the trade is entered at the decision bar's close, because the label is "
        "close[t+6]/close[t]-1; a specialist bar that closes at that same instant has "
        "printed and is available, and one that closes later has not"
    ),
    "rule": (
        "as_of = searchsorted(specialist_close_times, reference_instant, side='right') - 1, "
        "the last specialist bar whose close is at or before the reference instant"
    ),
    "close_time": "a specialist bar's close is its open plus its own bar width",
    "side_right": (
        "a decision row landing exactly on a specialist close does see that bar — the same "
        "boundary convention nn.mtf uses, and the boundary rows are where a shift is "
        "detectable at all"
    ),
    "own_clock_is_the_identity": (
        "for the mode's own decision clock the rule maps row t to row t exactly, because a "
        "bar closes at its own close. That is asserted rather than assumed: see "
        "VALIDITY_GATE."
    ),
    "never": (
        "a prediction whose source candle had not closed at the reference instant, a "
        "forward fill across a specialist's missing bar, or an interpolation of any kind"
    ),
}

#: What happens when a specialist has nothing available yet — at the head of an
#: outer block, before its first bar has closed inside that block.
#:
#: The strict choice: the consensus is HOLD, rather than voting with a partial
#: set. A missing specialist is not evidence of anything, and letting a rule
#: reach its threshold on the members that happen to be present is exactly the
#: shape of a result that depends on missing data.
UNAVAILABILITY_RULE = {
    "condition": (
        "any required specialist has no bar closed at or before the reference instant"
    ),
    "consensus": "HOLD",
    "constituent_replay": "HOLD for that constituent on that row",
    "why_strict": (
        "a partial vote would let a rule reach its threshold on the members that happen to "
        "be present, which makes the result a function of where a block starts"
    ),
}

#: Measured before any consensus was computed, from the committed P6 predictions.
#: Every count is the head of an outer block, and no fold loses more than 14 of
#: its decision rows on any specialist.
MEASURED_AVAILABILITY = {
    "scalping": {
        "decision_rows": 1161875,
        "unavailable_rows": {"1m": 0, "5m": 16, "15m": 56},
        "worst_fold_unavailable": 14,
    },
    "day_trading": {
        "decision_rows": 232285,
        "unavailable_rows": {"5m": 0, "15m": 8, "30m": 20, "1h": 44},
        "worst_fold_unavailable": 11,
    },
    "note": (
        "every unavailable row is at the head of an outer block, before the slower "
        "specialist's first bar has closed inside it. The largest loss on any fold is 14 "
        "rows of 289,248, so the unavailability rule cannot carry a result either way."
    ),
}

# --------------------------------------------------------------------------- #
# 3. The two modes
# --------------------------------------------------------------------------- #

LONG = "LONG"
SHORT = "SHORT"
HOLD = "HOLD"

#: P7A. A scalping decision is taken every minute and reads the three fastest
#: clocks; the 15m specialist is the slow member and holds the veto.
SCALPING = {
    "mode": "SCALPING",
    "decision_clock": "1m",
    "specialists": ["1m", "5m", "15m"],
    "veto_specialist": "15m",
    "agreement_required": 2,
    "of": 3,
    "rule": (
        "LONG when at least 2 of the 3 specialists are actively LONG and the 15m "
        "specialist is not actively SHORT; SHORT when at least 2 of the 3 are actively "
        "SHORT and the 15m specialist is not actively LONG; otherwise HOLD"
    ),
    "horizon_bars": 6,
    "horizon": "6 minutes",
}

#: P7B. A day-trading decision is taken every five minutes and reads the
#: intraday clocks; the 1h specialist is the slow member and holds the veto.
DAY_TRADING = {
    "mode": "DAY_TRADING",
    "decision_clock": "5m",
    "specialists": ["5m", "15m", "30m", "1h"],
    "veto_specialist": "1h",
    "agreement_required": 3,
    "of": 4,
    "rule": (
        "LONG when at least 3 of the 4 specialists are actively LONG and the 1h "
        "specialist is not actively SHORT; SHORT when at least 3 of the 4 are actively "
        "SHORT and the 1h specialist is not actively LONG; otherwise HOLD"
    ),
    "horizon_bars": 6,
    "horizon": "30 minutes",
}

MODES: tuple[dict[str, Any], ...] = (SCALPING, DAY_TRADING)

CONSENSUS_VERSION = "consensus_v1"

CONSENSUS_PROHIBITIONS: tuple[str, ...] = (
    "no discretionary weighting; every specialist's vote counts exactly one",
    "no weight search, no threshold search, no agreement-count search",
    "no per-fold or per-mode variation of the rule",
    "no consensus_v2 in the task that produced consensus_v1's result",
    "'actively LONG' and 'actively SHORT' mean the specialist's own frozen action, and "
    "HOLD is neither",
)

# --------------------------------------------------------------------------- #
# 4. Controls and accounting
# --------------------------------------------------------------------------- #

#: The comparison P7 is actually about. A consensus taken every minute cannot be
#: compared against a specialist's native P6 return, because those were measured
#: under different execution cadences — a 15m specialist in P6 took a position
#: every 15 minutes at most and here it is asked at every minute. So every
#: constituent is replayed on the *mode's* decision clock, through the same
#: alignment and the same accounting, and the consensus is compared against that.
CONTROLS = {
    "what": (
        "each constituent specialist, replayed on the mode's own decision clock through "
        "the same causal alignment and the same cost and trade accounting as the consensus"
    ),
    "why": (
        "the consensus must be compared against its components under identical execution "
        "cadence, not against native P6 returns computed under a different one"
    ),
    "accounting": (
        "nn.evaluate.trading_metrics on the decision clock's own future_return and row "
        "index — the same greedy non-overlapping trade rule, the same six-native-bar hold, "
        "and the same 20 bps round trip charged once per realised trade"
    ),
}

COSTS = {
    "fee_rate": 0.0005,
    "slippage_rate": 0.0005,
    "cost_threshold": 0.002,
    "unchanged_from": "P6, and every checkpoint since v4",
    "forbidden": (
        "reducing costs because a consensus on a 1m clock otherwise looks bad. Turnover "
        "and trade count are reported for the consensus and for every constituent."
    ),
}

#: The per-fold benchmark. Computed mechanically inside each fold from the
#: constituent replays; it is a *benchmark*, not a model selected for deployment,
#: and no constituent is promoted by being the best on a fold.
BEST_CONSTITUENT = {
    "definition": "max over the mode's constituents of that constituent's replayed outer "
    "net return, computed within each fold",
    "is": "a benchmark computed after the fact inside each fold",
    "is_not": (
        "a selection; nothing is deployed, carried forward or preferred by winning a fold"
    ),
    "why_hardest": (
        "comparing against the fold-wise best is deliberately the hardest fair bar: a "
        "consensus that only beats the average of its members has not shown it is worth "
        "the machinery"
    ),
}

FOLD_DELTA = "consensus outer net return minus the fold's best constituent outer net return"

# --------------------------------------------------------------------------- #
# 5. Validity, and the decision
# --------------------------------------------------------------------------- #

#: P7 is invalid unless this holds, and it is checked before any delta is read.
#:
#: The alignment maps a mode's own decision clock onto itself as the identity, so
#: replaying that specialist through P7's machinery must reproduce the frozen P6
#: cell's four outer net returns *exactly*. If it does not, the replay is not the
#: thing P6 measured and no comparison built on it means anything.
VALIDITY_GATE = {
    "check": (
        "the mode's own decision-clock specialist, replayed by P7, reproduces the frozen "
        "P6 cell's four outer net returns exactly"
    ),
    "tolerance": "exact to the 1e-6 rounding nn.evaluate.trading_metrics applies",
    "on_failure": "P7 is invalid; no verdict is issued and no delta is reported",
    "why": (
        "it is the one property that ties P7's accounting to P6's. A replay that cannot "
        "reproduce the specialist it claims to be replaying is measuring something else."
    ),
}

#: Applied per mode, separately. Both conditions required.
DECISION_RULE = {
    "conditions": [
        "the consensus beats the fold-wise best constituent in at least 3 of the 4 "
        "outer folds",
        "the mean fold delta across the 4 folds is > 0",
    ],
    "conjunction": "both",
    "improved_folds_required": 3,
    "total_folds": 4,
    "per_mode": True,
    "no_further_criterion": (
        "these two, and nothing else. A criterion added after the deltas are visible is "
        "not a criterion, it is a description of the result."
    ),
}

OUTCOMES: tuple[str, ...] = (
    "both modes supportive",
    "scalping supportive only",
    "day trading supportive only",
    "neither supportive",
    "invalid",
)

OUTCOME_INDEPENDENCE = (
    "The two modes are reported separately and neither is collapsed into the other. "
    "'Scalping supportive, day trading not' is a result, not a reason to report scalping "
    "as P7's answer."
)

DIAGNOSTICS = {
    "turnover": "reported for the consensus and every constituent, per fold",
    "trade_count": "reported per fold; flagged below 10 outer trades, changes no denominator",
    "hold_fraction": "the share of decision rows on which the consensus is HOLD, reported",
    "agreement_counts": "how often the rule reached its threshold and how often the veto "
    "fired, reported and decisive in nothing",
}

# --------------------------------------------------------------------------- #
# 6. Boundaries
# --------------------------------------------------------------------------- #

STYX_PROHIBITION = (
    "P7 reads no candle at all. It reads committed P6 predictions, which exist only for "
    "rows inside the four outer blocks, so the sealed region is not reachable from this "
    "checkpoint by any path."
)

P4_HOLD_UNAVAILABILITY = (
    "P4-HOLD is retired with checkpoint null and available to nobody. P7 adds no data "
    "source and no new way to reach it, and manufactures no replacement holdout."
)

FORBIDDEN_AFTER_RESULTS: tuple[str, ...] = (
    "changing either consensus rule, its agreement count or its veto specialist",
    "changing either mode's specialist set or decision clock",
    "refitting, re-thresholding or re-weighting any specialist",
    "changing the alignment rule or the unavailability rule",
    "changing the cost model, the fee rate or the slippage rate",
    "changing the control definition or the fold-wise best-constituent benchmark",
    "changing either condition of the decision rule, or adding a third",
    "reporting one mode's verdict as P7's answer",
    "writing consensus_v2 in the task that produced consensus_v1's result",
)

STOPPING_RULE = {
    "on_supportive": (
        "a supportive mode has adaptive evidence that agreement added value over its own "
        "components on these four burned blocks. It is not a deployable strategy, it is "
        "not out-of-sample evidence, and it does not license real money."
    ),
    "on_negative": (
        "a mode whose consensus did not beat its own components is recorded as negative. "
        "The trading-mode scaffold may still describe it, and must not claim alpha for it."
    ),
    "on_invalid": (
        "the validity gate failed; the replay is not reproducing the specialists it claims "
        "to replay, and no verdict is issued until that is fixed."
    ),
}

ARTIFACT_POLICY = {
    "modes": "artifacts/benchmark/btc_p7_{mode}/ holding p7.json, p7.md and STATUS.md",
    "decision": "artifacts/benchmark/btc_p7_decision/decision.json and STATUS.md",
    "manifest": "artifacts/btc_p7_SHA256SUMS.txt over the primary mode evidence",
    "reruns": (
        "a valid negative mode is never re-run. A mode invalidated by a software defect "
        "may be repaired and re-run, and only the affected mode."
    ),
}

LEAKAGE_BATTERY: tuple[dict[str, str], ...] = (
    {
        "id": "C1",
        "property": "no decision row reads a specialist bar that had not closed",
        "positive_control": "a specialist prediction shifted one bar earlier is caught",
    },
    {
        "id": "C2",
        "property": "a decision row landing exactly on a specialist close does see that bar",
        "positive_control": "the boundary row is constructed and asserted in both directions",
    },
    {
        "id": "C3",
        "property": "the own-clock alignment is the identity",
        "positive_control": "asserted per mode, and the validity gate depends on it",
    },
    {
        "id": "C4",
        "property": "a missing specialist yields HOLD and never a partial vote",
        "positive_control": "a specialist truncated at the block head is asserted to HOLD",
    },
    {
        "id": "C5",
        "property": "duplicate specialist predictions for one bar are refused",
        "positive_control": "a duplicated row is injected and the loader raises",
    },
    {
        "id": "C6",
        "property": "the consensus is deterministic: equal input gives equal output",
        "positive_control": "the same frame is replayed twice and the digests compared",
    },
    {
        "id": "C7",
        "property": "HOLD propagates — no agreement, no position",
        "positive_control": "an all-HOLD frame yields zero trades",
    },
    {
        "id": "C8",
        "property": "the veto blocks agreement it disagrees with",
        "positive_control": "a 2-of-3 LONG with a SHORT veto yields HOLD",
    },
    {
        "id": "C9",
        "property": "disagreement yields HOLD",
        "positive_control": "one LONG, one SHORT, one HOLD yields HOLD",
    },
    {
        "id": "C10",
        "property": "same-timestamp ordering is stable and does not affect the result",
        "positive_control": "a shuffled input frame reproduces the same consensus",
    },
)

HELD_FIXED = (
    "the specialists and their frozen predictions, the alignment rule, the unavailability "
    "rule, both consensus rules, both specialist sets, both decision clocks, the cost "
    "model, the control definition, the best-constituent benchmark and the decision rule."
)


def payload() -> dict[str, Any]:
    """Everything the hash covers. Adding a constant here is a design change."""
    return {
        "checkpoint": CHECKPOINT,
        "question": QUESTION,
        "hypothesis": HYPOTHESIS,
        "evidence_ceiling": EVIDENCE_CEILING,
        "research_classification": RESEARCH_CLASSIFICATION,
        "specialist_source": SPECIALIST_SOURCE,
        "refit_prohibition": REFIT_PROHIBITION,
        "specialist_set_is_architectural": SPECIALIST_SET_IS_ARCHITECTURAL,
        "alignment": ALIGNMENT,
        "unavailability_rule": UNAVAILABILITY_RULE,
        "measured_availability": MEASURED_AVAILABILITY,
        "consensus_version": CONSENSUS_VERSION,
        "modes": [dict(mode) for mode in MODES],
        "consensus_prohibitions": list(CONSENSUS_PROHIBITIONS),
        "controls": CONTROLS,
        "costs": COSTS,
        "best_constituent": BEST_CONSTITUENT,
        "fold_delta": FOLD_DELTA,
        "validity_gate": VALIDITY_GATE,
        "decision_rule": DECISION_RULE,
        "outcomes": list(OUTCOMES),
        "outcome_independence": OUTCOME_INDEPENDENCE,
        "diagnostics": DIAGNOSTICS,
        "styx_prohibition": STYX_PROHIBITION,
        "p4_hold_unavailability": P4_HOLD_UNAVAILABILITY,
        "forbidden_after_results": list(FORBIDDEN_AFTER_RESULTS),
        "stopping_rule": STOPPING_RULE,
        "artifact_policy": ARTIFACT_POLICY,
        "leakage_battery": [dict(item) for item in LEAKAGE_BATTERY],
        "held_fixed": HELD_FIXED,
    }


def preregistration_hash() -> str:
    """SHA-256 over :func:`payload`. Every P7 artifact records it."""
    blob = json.dumps(payload(), sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(blob.encode()).hexdigest()


def mode(name: str) -> dict[str, Any]:
    """One mode's frozen design, by name."""
    for item in MODES:
        if item["mode"] == name:
            return item
    raise KeyError(
        f"P7 registered no mode {name!r}; it registered {[m['mode'] for m in MODES]}"
    )


def describe() -> dict[str, Any]:
    return {"preregistration_hash": preregistration_hash(), **payload()}


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(describe(), indent=2))
