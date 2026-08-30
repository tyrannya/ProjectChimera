"""P6's design, frozen before any P6 model is fitted.

The machine-readable twin of ``docs/p6_preregistration.md``. Every constant a
result could be argued into or out of lives here, and
:func:`preregistration_hash` covers all of them, so a cell produced under an
edited design is a different object rather than the same one with a different
story. Every P6 cell records the hash.

**What P6 changes, and what it does not.** Five checkpoints varied the columns
attached to a 1h bar. P6 varies the bar and holds the columns: the same fourteen
`chimera.features` columns, the same cost-aware label definition, the same three
untuned model families, the same four real-world temporal periods. The only
thing that moves is the clock — and, because the horizon is stated in *native*
bars, the physical distance the label looks ahead.

**P6 is a screen, not a comparison.** There is no control arm, because there is
nothing to be incrementally better than: a 1m specialist has no 1m predecessor.
Each clock gets its own verdict against an absolute viability gate, and all five
are reported. Choosing the best clock and calling P6 positive is forbidden by
§9 and by the shape of the artifact, which has five rows and no summary row.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from nn.multiclock import BASE_CONTRACT_ID
from nn.research_contract import load_contract

CHECKPOINT = "P6"

QUESTION = (
    "Can independent native-timeframe specialists extract robust cost-aware predictive "
    "signal at temporal scales the previous 1h/6h programme did not test?"
)

HYPOTHESIS = (
    "Unknown, and deliberately not asserted. Five negative information-set results were "
    "all obtained on one clock, which is the axis none of them varied. That makes the "
    "clock worth testing and says nothing about which way it will come out."
)

EVIDENCE_CEILING = (
    "Exploratory, adaptive. The four temporal periods are the same real-world windows "
    "v4, P2a, P2b, P2c, P3, P4 and P5 have already read, mapped onto faster clocks. "
    "Faster clocks multiply rows, not independent temporal periods: 2.8 million minutes "
    "of one asset over four windows is four observations of a market regime, not "
    "2.8 million. No P6 result is confirmatory, and a positive P6 needs confirmation "
    "these blocks cannot supply."
)

RESEARCH_CLASSIFICATION = (
    "A screen over five clocks. Every clock's verdict is published whatever it says. "
    "P6 selects nothing for deployment, and a clock that passes the gate has earned "
    "the right to be carried into P7's consensus test, not a claim of profitability."
)

# --------------------------------------------------------------------------- #
# 1. The clocks
# --------------------------------------------------------------------------- #

#: The five clocks, frozen. Not four, not six, and not "whichever of these five
#: survives": §9 requires all five verdicts in the artifact.
CLOCKS: tuple[str, ...] = ("1m", "5m", "15m", "30m", "1h")

#: The source every clock is cut from, and the rules it is cut under. Described
#: in ``docs/multiclock_v1.md`` and enforced by :mod:`nn.multiclock`.
SOURCE = {
    "instrument": "binance spot BTCUSDT",
    "base_clock": "1m",
    "committed": "data/research/btc_usdt_multiclock_gen2_1m_pre_boundary.parquet",
    "manifest": "data/research/btc_usdt_multiclock_gen2_manifest.json",
    "derivation": (
        "epoch-anchored UTC grid; a bar exists only with its full constituent minute "
        "count; incomplete bars are dropped, never forward-completed"
    ),
    "parity_1h": (
        "47,094 of 47,123 overlapping hours agree with the committed 1h history at "
        "relative tolerance 1e-9; the 29 that do not are an upstream inconsistency "
        "between two series Binance publishes itself, all of them between 2020-04 and "
        "2022-05 and none of them in any outer block"
    ),
}

# --------------------------------------------------------------------------- #
# 2. Features, target, horizon and costs
# --------------------------------------------------------------------------- #

#: Unchanged from every checkpoint since v4: `chimera.features.compute_features`,
#: fourteen columns, window lengths included. P6 adds no family. `smc_v1`,
#: `chart_structure_v1`, `microstructure_v1`, `derivatives_v1` and `mtf_v1` are
#: all absent, because a checkpoint that changed the clock *and* the information
#: would answer neither question.
FEATURE_ENGINE = {
    "module": "chimera.features.compute_features",
    "columns": 14,
    "spec": "FeatureSpec() defaults, unchanged",
    "warmup_bars": 78,
    "evaluated_on": "the specialist's own native bars, not on 1h bars",
}

#: Six *native* bars. The number is inherited from every previous checkpoint and
#: the unit is what changes, which is the whole point: a scale-consistent rule
#: keeps "six bars" fixed and lets the physical distance follow the clock.
#: Searching horizons is a different checkpoint (P9) and is forbidden here.
HORIZON_BARS = 6

#: What six native bars means in wall-clock time, per clock.
HORIZONS = {
    "1m": "6 minutes",
    "5m": "30 minutes",
    "15m": "90 minutes",
    "30m": "3 hours",
    "1h": "6 hours",
}

#: Identical on every clock, in basis points, unchanged from v4 onward. A cost
#: model expressed per trade does not become cheaper because trades are shorter:
#: that is precisely the effect P6 is measuring, and discounting it would delete
#: the result rather than produce one.
COSTS = {
    "fee_rate": 0.0005,
    "slippage_rate": 0.0005,
    "cost_threshold": 0.002,
    "meaning": "round-trip: entry and exit, each paying fee plus slippage, 20 bps",
    "applied": "once per realised trade, by nn.evaluate.realised_trades",
    "forbidden": (
        "reducing costs because a fast clock looks worse under them. Turnover and trade "
        "count are reported per cell so the reader can see what the cost model is doing."
    ),
}

#: The label, applied mechanically on each clock's own closes. Class definitions
#: do not move with the resulting balance — see MEASURED_UNIVERSE, which is
#: recorded before any fit precisely so that nobody can later claim the labels
#: were chosen once the balance was known.
TARGET = {
    "definition": "nn.data_pipeline.compute_target on the native clock's close series",
    "future_return": "close[t + 6] / close[t] - 1, in native bars",
    "classes": "LONG if future_return > 0.002, SHORT if < -0.002, else HOLD",
    "forbidden": (
        "changing the class definition, the cost threshold or the horizon in response to "
        "a clock's class balance"
    ),
}

# --------------------------------------------------------------------------- #
# 3. Samples
# --------------------------------------------------------------------------- #

#: One native bar. Every clock uses the same value, so no clock is advantaged.
#:
#: **Why not 64.** P2a and P2b window 64 bars because they scored MTST's own
#: samples and had to build the same array MTST built. P6 has no MTST and no
#: such constraint, and "native OHLCV14 semantics on this clock" reads most
#: exactly as the fourteen columns of the bar the decision is taken on. The
#: fourteen are already multi-bar aggregates — EMA ratios, RSI-14, MACD, ATR-14,
#: realised volatility, a volume z-score — so history reaches the model through
#: the indicators rather than through 896 flattened columns.
#:
#: A second, weaker reason is recorded because hiding it would be worse than
#: stating it: the 1m clock's fold-3 training block is 2.24 million samples. At
#: seq_len 16 a single logistic-regression fit on it reached 9.0 GB of resident
#: memory on a 15 GB machine before either tree family ran, and at seq_len 64 the
#: input array alone is 8 GB. That measurement was taken on a *training* block,
#: before any outer block existed in any form, and it did not choose the value —
#: it ruled out the alternative.
SEQ_LEN = 1

#: The research region, identical for every clock. It begins where the committed
#: 1h research coverage begins, so the five specialists see the same real-world
#: history, and it ends at the retired P4-HOLD boundary.
REGION = {
    "start": "2020-01-04T06:00:00+00:00",
    "end_exclusive": "2025-05-19T08:00:00+00:00",
    "start_is": "the first row of the committed 1h outer coverage",
    "end_is": "the first instant of the retired P4-HOLD region",
}

# --------------------------------------------------------------------------- #
# 4. Folds — the same four real-world periods on every clock
# --------------------------------------------------------------------------- #

#: The four outer periods, as instants, taken from the 1h fold plan every
#: checkpoint since P2b has run: `nn.p2b.plan_from_manifest` over the committed
#: snapshot, rendered from row indices into timestamps.
#:
#: **Mapped by timestamp, never by row number.** A 1m row 21,697 and a 1h row
#: 21,697 are two different fortnights. Freezing the instants is what makes
#: "the same four temporal periods" true of all five clocks rather than a hope.
#:
#: They are literals here and recomputed from the committed 1h snapshot by
#: `tests/test_p6_preregistration.py`, so the freeze is checkable rather than
#: asserted.
FOLD_PERIODS: tuple[dict[str, str], ...] = (
    {
        "fold": "0",
        "train_start": "2020-01-04T06:00:00+00:00",
        "inner_start": "2022-08-15T10:00:00+00:00",
        "outer_start": "2023-03-04T07:00:00+00:00",
        "outer_end": "2023-09-24T17:00:00+00:00",
    },
    {
        "fold": "1",
        "train_start": "2020-01-04T06:00:00+00:00",
        "inner_start": "2023-03-04T07:00:00+00:00",
        "outer_start": "2023-09-24T17:00:00+00:00",
        "outer_end": "2024-04-12T14:00:00+00:00",
    },
    {
        "fold": "2",
        "train_start": "2020-01-04T06:00:00+00:00",
        "inner_start": "2023-09-24T17:00:00+00:00",
        "outer_start": "2024-04-12T14:00:00+00:00",
        "outer_end": "2024-10-30T11:00:00+00:00",
    },
    {
        "fold": "3",
        "train_start": "2020-01-04T06:00:00+00:00",
        "inner_start": "2024-04-12T14:00:00+00:00",
        "outer_start": "2024-10-30T11:00:00+00:00",
        "outer_end": "2025-05-19T08:00:00+00:00",
    },
)

#: The outer periods alone, as (start, end_exclusive) pairs, for the checks that
#: only care where results are reported.
OUTER_PERIODS: tuple[tuple[str, str], ...] = tuple(
    (row["outer_start"], row["outer_end"]) for row in FOLD_PERIODS
)

FOLD_POLICY = (
    "Four folds, expanding training window, no shuffle. The four periods are not "
    "subdivided to manufacture more independence: a faster clock delivers more rows "
    "inside the same four windows, and rows are not folds. Training reaches every "
    "available row before the fold's inner block; the label's six-bar embargo is "
    "applied by nn.dataset.sample_indices, so no training row's label can be drawn "
    "from the block it is selected or scored on."
)

# --------------------------------------------------------------------------- #
# 5. Models
# --------------------------------------------------------------------------- #

#: The three predeclared families from nn.simple_models, at the configurations
#: P2a froze. No hyperparameter search, no early stopping, no per-clock tuning.
MODELS: tuple[str, ...] = ("logistic_regression", "lightgbm", "xgboost")

#: The family whose number is the verdict, chosen before any P6 fit for the
#: reason P2a established: it was the strongest OHLCV14 family there. The other
#: two are reported in full and decide nothing.
PRIMARY_MODEL = "xgboost"

SECONDARY_MODELS_ROLE = (
    "Reported for every clock and decisive for none. If logistic regression passes the "
    "gate on a clock where XGBoost does not, the clock's verdict is XGBoost's. Switching "
    "to whichever family passed is the search this design exists to prevent."
)

SEED = 42

SEED_POLICY = (
    "One run. Per-fold seed is SEED + fold, exactly as nn.benchmark derives it. Logistic "
    "regression takes no seed and the tree families are deterministic given theirs, so a "
    "second seed would not be a second observation of these four periods."
)

#: How the decision threshold is chosen: on the inner block, never on the outer.
THRESHOLD = {
    "selector": "nn.evaluate.select_threshold",
    "block": "inner_validation",
    "objective": (
        "maximise net return after round-trip costs on the inner-validation block, "
        "subject to at least min_trades realised trades"
    ),
    "grid": "numpy.arange(0.34, 0.91, 0.02), rounded to 4 decimals",
    "min_trades": 10,
    "forbidden": "any use of an outer block in choosing a threshold",
}

# --------------------------------------------------------------------------- #
# 6. The gate
# --------------------------------------------------------------------------- #

#: The floors every clock is measured against. The momentum baseline is the
#: *native* one: nn.benchmark.fit_baselines resolves `ema_cross` by name in the
#: clock's own feature set, so a 1m specialist is compared with a 1m momentum
#: rule and not with an hourly one.
BASELINES = {
    "momentum_baseline": (
        "nn.baselines.MomentumBaseline on this clock's own ema_cross, deadband 0.001"
    ),
    "majority_baseline": "nn.baselines.MajorityClassBaseline fitted on the fold's labels",
    "economic_references": (
        "CASH and buy-and-hold, reported as references. Buy-and-hold is not a required "
        "competitor: it is a full-exposure long position over four periods that all rose."
    ),
}

#: **The viability gate, frozen before the first fit.**
#:
#: No directly applicable precedent existed. P2b, P2c, P3 and P5 all gate an
#: *incremental* comparison — an arm against its control — and P4's stage-1 rule
#: is a continuation screen with the same shape. P6 has no control to be
#: incrementally better than, so a gate had to be written, and it is written here
#: rather than after the numbers arrive.
#:
#: All three conditions are required. A clock that fails any one of them is not
#: viable, whatever the other two say.
VIABILITY_GATE = {
    "conditions": [
        "cost-aware outer net return > 0 in at least 3 of the 4 folds",
        "mean outer cost-aware net return across the 4 folds > 0",
        "beats the matching native-timeframe momentum baseline's outer net return "
        "in at least 3 of the 4 folds",
    ],
    "conjunction": "all three",
    "positive_folds_required": 3,
    "beats_momentum_folds_required": 3,
    "total_folds": 4,
    "decided_by": PRIMARY_MODEL,
    "per_clock": True,
    "buy_and_hold": "a reference, not a required competitor",
    "why_a_new_gate": (
        "every existing gate in this repository compares an arm against a control, and "
        "P6 has no control. A screen for absolute viability needs an absolute floor."
    ),
}

#: Reported per cell and decisive in neither direction, as P5's was.
DIAGNOSTICS = {
    "trade_count": "flagged below 10 outer trades; changes no denominator",
    "turnover": "2 x realised trades, reported per cell",
    "class_balance": "reported per clock; it is an outcome of the frozen labels, not an input",
    "mean_and_worst_fold": "reported; the mean is one of the three gate conditions and the "
    "worst fold is descriptive only",
}

# --------------------------------------------------------------------------- #
# 7. Measured before any fit
# --------------------------------------------------------------------------- #

#: Facts about the data, measured before a single P6 model existed, so that they
#: cannot later be presented as a discovery or used to argue a rule.
#:
#: The class balance is the one worth staring at: a fixed 20 bps round-trip cost
#: threshold is a large fraction of a six-minute move and a small fraction of a
#: six-hour one, so HOLD runs from 80.5% of 1m rows to 20.3% of 1h rows. That is
#: a mechanical consequence of holding costs fixed while the clock changes. It is
#: **not** a reason to move the threshold, and §2's `forbidden` clause says so.
MEASURED_UNIVERSE = {
    "1m": {"region_rows": 2821619, "segments": 17, "hold_fraction": 0.805},
    "5m": {"region_rows": 563328, "segments": 17, "hold_fraction": 0.544},
    "15m": {"region_rows": 186936, "segments": 16, "hold_fraction": 0.367},
    "30m": {"region_rows": 92834, "segments": 16, "hold_fraction": 0.278},
    "1h": {"region_rows": 45783, "segments": 15, "hold_fraction": 0.203},
}

MEASURED_UNIVERSE_NOTE = (
    "The 1h clock's 45,783 rows against the committed snapshot's 45,802 is the 13 hours "
    "the full-constituent rule makes unavailable, plus the segment boundaries they "
    "create. P6's 1h cell is therefore not a reproduction of P2a's control and is not "
    "offered as one."
)

# --------------------------------------------------------------------------- #
# 8. Boundaries
# --------------------------------------------------------------------------- #

#: The instant is *resolved* from the committed contract rather than restated.
#: `tests/test_research_contracts.py` forbids a second copy of the sealed anchor
#: anywhere under `nn/` or `chimera/`, and it is right to: two constants that
#: agree today are two constants that can disagree tomorrow. The rendered string
#: — and therefore the preregistration hash — is unchanged by this.
STYX_PROHIBITION = (
    f"Styx ({load_contract(BASE_CONTRACT_ID).sealed_test_start.isoformat()}) is not "
    "read, scored, inspected or planned over by "
    "P6. The research-visible boundary binds three months earlier, so no P6 code path "
    "reaches within three months of it."
)

P4_HOLD_UNAVAILABILITY = (
    "P4-HOLD, rows [45802, 48211) from 2025-05-19T08:00:00+00:00, is retired with "
    "checkpoint null. It was never opened and is available to nobody. Every clock stops "
    "before it, enforced on constituent minutes by nn.multiclock, and P6 does not "
    "manufacture a replacement holdout because the clock changed."
)

FORBIDDEN_AFTER_RESULTS: tuple[str, ...] = (
    "changing the clock set, or dropping a clock whose verdict disappoints",
    "changing the horizon rule from six native bars",
    "changing the cost model, the fee rate or the slippage rate",
    "changing the class definition or the cost threshold",
    "changing seq_len, the seed, or any model configuration",
    "changing the fold periods, or subdividing them",
    "changing any condition of the viability gate, or the model that decides it",
    "reporting the best clock as though it were the checkpoint's answer",
    "refitting a clock that failed, under any pretext",
    "adding a fourth model family, or a second seed, after seeing a number",
)

STOPPING_RULE = {
    "on_pass": (
        "A clock that passes the gate is viable on adaptive research folds and is carried "
        "into P7 as a frozen specialist. It is not a deployable strategy and its number is "
        "not out-of-sample evidence."
    ),
    "on_fail": (
        "A clock that fails is recorded as failing and is still carried into P7 if P7's "
        "own design names it, because P7 asks whether agreement adds value over the "
        "individual specialists — a question whose answer must not depend on which "
        "specialists happened to pass an absolute floor."
    ),
    "on_all_fail": (
        "If no clock passes, P6 is negative: changing the clock, on this asset, over these "
        "four periods, under this information set and these costs, did not produce robust "
        "cost-aware signal. P7 may still run, because consensus among unprofitable "
        "specialists is a separate question, and a negative P7 on top is a cleaner answer "
        "than not asking."
    ),
}

ARTIFACT_POLICY = {
    "cells": "one directory per (clock, model): artifacts/benchmark/btc_p6_{clock}_{model}/",
    "cell_files": "p6.json, p6.md, STATUS.md, outer_predictions.parquet",
    "decision": "artifacts/benchmark/btc_p6_decision/decision.json and STATUS.md",
    "manifest": "artifacts/btc_p6_SHA256SUMS.txt over the primary cells",
    "reruns": (
        "a valid negative cell is never re-run. A cell invalidated by a software defect "
        "may be repaired and re-run, and only the affected cells."
    ),
}

LEAKAGE_BATTERY: tuple[dict[str, str], ...] = (
    {
        "id": "L1",
        "property": "no bar is built from a minute at or after the research boundary",
        "positive_control": (
            "a minute moved to the boundary is refused by resample_from_minutes"
        ),
    },
    {
        "id": "L2",
        "property": "no derived bar contains a constituent from the following period",
        "positive_control": "a bar shifted one period is caught by the parity comparison",
    },
    {
        "id": "L3",
        "property": "an incomplete bar does not exist rather than existing shortened",
        "positive_control": "deleting one minute removes its whole bar, asserted by count",
    },
    {
        "id": "L4",
        "property": "features at row t depend only on rows <= t on the same clock",
        "positive_control": "chimera.features' own append-invariance tests, per clock",
    },
    {
        "id": "L5",
        "property": "no training or inner row's label is drawn from a later block",
        "positive_control": (
            "sample_indices' horizon embargo, asserted against the block edges"
        ),
    },
    {
        "id": "L6",
        "property": "no feature, label or window crosses a market-data gap",
        "positive_control": (
            "segment ids recomputed on each clock and asserted against the folds"
        ),
    },
    {
        "id": "L7",
        "property": "the threshold is chosen on the inner block and never on the outer",
        "positive_control": "the selector is handed inner indices only; asserted per cell",
    },
    {
        "id": "L8",
        "property": "every clock's four outer periods are the same four instants",
        "positive_control": "recomputed from the committed 1h snapshot and compared",
    },
)

HELD_FIXED = (
    "asset, exchange, instrument, information set, feature engine, warm-up, label "
    "definition, horizon in native bars, cost model, model families and their "
    "configurations, seed, seq_len, threshold selector and grid, fold periods, and the "
    "gate. The clock is the only thing that varies between cells of different clocks, "
    "and the model family the only thing that varies within a clock."
)


def payload() -> dict[str, Any]:
    """Everything the hash covers. Adding a constant here is a design change."""
    return {
        "checkpoint": CHECKPOINT,
        "question": QUESTION,
        "hypothesis": HYPOTHESIS,
        "evidence_ceiling": EVIDENCE_CEILING,
        "research_classification": RESEARCH_CLASSIFICATION,
        "clocks": list(CLOCKS),
        "source": SOURCE,
        "feature_engine": FEATURE_ENGINE,
        "horizon_bars": HORIZON_BARS,
        "horizons": HORIZONS,
        "costs": COSTS,
        "target": TARGET,
        "seq_len": SEQ_LEN,
        "region": REGION,
        "fold_periods": [dict(row) for row in FOLD_PERIODS],
        "fold_policy": FOLD_POLICY,
        "models": list(MODELS),
        "primary_model": PRIMARY_MODEL,
        "secondary_models_role": SECONDARY_MODELS_ROLE,
        "seed": SEED,
        "seed_policy": SEED_POLICY,
        "threshold": THRESHOLD,
        "baselines": BASELINES,
        "viability_gate": VIABILITY_GATE,
        "diagnostics": DIAGNOSTICS,
        "measured_universe": MEASURED_UNIVERSE,
        "measured_universe_note": MEASURED_UNIVERSE_NOTE,
        "styx_prohibition": STYX_PROHIBITION,
        "p4_hold_unavailability": P4_HOLD_UNAVAILABILITY,
        "forbidden_after_results": list(FORBIDDEN_AFTER_RESULTS),
        "stopping_rule": STOPPING_RULE,
        "artifact_policy": ARTIFACT_POLICY,
        "leakage_battery": [dict(item) for item in LEAKAGE_BATTERY],
        "held_fixed": HELD_FIXED,
    }


def preregistration_hash() -> str:
    """SHA-256 over :func:`payload`. Every P6 cell records it."""
    blob = json.dumps(payload(), sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(blob.encode()).hexdigest()


def describe() -> dict[str, Any]:
    """The preregistration plus its hash, for a report or an artifact."""
    return {"preregistration_hash": preregistration_hash(), **payload()}


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(describe(), indent=2))
