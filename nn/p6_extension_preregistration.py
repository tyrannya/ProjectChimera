"""P6-EXT: the same screen, on the two slow clocks a SWING mode needs.

Frozen before any 4h or 1d specialist was fitted, and **separate from P6**
because P6's design was frozen over five clocks and closed. Editing a closed
preregistration to add two more is not an extension, it is a rewrite; this is the
extension.

**Everything except the clock set is P6's, imported rather than restated.** The
features, the horizon rule, the costs, the label, ``seq_len``, the models, the
seed, the threshold selector, the region, the fold periods and the viability gate
are the objects :mod:`nn.p6_preregistration` defines, so a change to any of them
would move both hashes and neither checkpoint could quietly diverge from the
other.

**Why it exists.** ``docs/trading_modes_v1.md`` describes a SWING mode whose
primary clocks are 30m, 1h and 4h with 1d as slow context. Two of those had no
specialist. A mode is not eligible until the specialists it names exist and have
been screened, and calling 30m/1h-only operation "swing" would be describing a
different thing by the same name.

**What it is not.** It is not P5. P5 evaluated 4h and 1d bars as *context columns
attached to a 1h row*; this fits a model on 4h samples and a model on 1d samples.
Those are different objects and the roadmap says so.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from nn.p6_preregistration import (
    BASELINES,
    COSTS,
    DIAGNOSTICS,
    FEATURE_ENGINE,
    FOLD_PERIODS,
    FOLD_POLICY,
    HELD_FIXED,
    HORIZON_BARS,
    LEAKAGE_BATTERY,
    MODELS,
    P4_HOLD_UNAVAILABILITY,
    PRIMARY_MODEL,
    REGION,
    SEED,
    SEED_POLICY,
    SEQ_LEN,
    SOURCE,
    STYX_PROHIBITION,
    TARGET,
    THRESHOLD,
    VIABILITY_GATE,
)
from nn.p6_preregistration import preregistration_hash as p6_hash

CHECKPOINT = "P6-EXT"

QUESTION = (
    "Do independent native-timeframe specialists on the two slow clocks a SWING mode "
    "needs — 4h and 1d — extract robust cost-aware predictive signal?"
)

HYPOTHESIS = (
    "Unknown, and deliberately not asserted. P6 was negative on all five short clocks, "
    "which is a reason to expect little and not a reason to skip the measurement: a mode "
    "whose specialists were never screened is not eligible, whatever anyone expects."
)

EVIDENCE_CEILING = (
    "Exploratory, adaptive, and designed with P6's results known. The same four real-world "
    "windows every checkpoint since v4 has read. No P6-EXT result is confirmatory."
)

RESEARCH_CLASSIFICATION = (
    "A two-clock extension of P6's screen, run under P6's gate. It exists so that SWING "
    "eligibility is a measured fact rather than an assumption, and it selects nothing."
)

#: The two clocks. Not five, and not seven: P6 answered for the five and this
#: answers for the two it did not cover.
CLOCKS: tuple[str, ...] = ("4h", "1d")

HORIZONS = {"4h": "1 day", "1d": "6 days"}

#: The checkpoint this extends, named by hash so that an extension of an edited
#: P6 would be a different object.
EXTENDS = {
    "checkpoint": "P6",
    "preregistration_hash": p6_hash(),
    "clocks": ["1m", "5m", "15m", "30m", "1h"],
    "relationship": (
        "identical design on two further clocks. Everything except the clock set is "
        "imported from nn.p6_preregistration rather than restated, so the two checkpoints "
        "cannot drift apart."
    ),
}

NOT_P5 = (
    "P5 evaluated 4h and 1d bars as context columns attached to a 1h row, and its decision "
    "was taken hourly by a model fitted on hourly samples. P6-EXT fits a model on 4h "
    "samples and a model on 1d samples. They are different objects and neither is evidence "
    "about the other."
)

#: Measured before any 4h or 1d model existed. The 1d clock is the thinnest
#: universe this programme has ever fitted on, and saying so before the result
#: exists is the point of recording it here.
MEASURED_UNIVERSE = {
    "4h": {"region_rows": 10569, "hold_fraction": 0.096},
    "1d": {"region_rows": 1253, "hold_fraction": 0.040},
}

THINNESS_NOTE = (
    "The 1d clock is by far the thinnest universe this programme has fitted on, and for a "
    "reason worth stating before the result exists: the shared 78-bar warm-up and 6-bar "
    "label embargo are applied per contiguous segment, and the 1m source's fifteen exchange "
    "outages cut the 1d series into segments too. Each costs 84 daily bars, so 1,946 "
    "in-region 1d bars become 1,253 usable rows. A 1d outer block is then roughly 130 rows "
    "and a 1d trade is held six days, so a fold can contain only a few dozen "
    "non-overlapping trades. That is a hard limit on what a 1d verdict generalises to. It "
    "is recorded here, before the verdict, and it changes no condition of the gate: a "
    "thin universe is a reason to distrust a *positive*, not a reason to move the bar."
)

FORBIDDEN_AFTER_RESULTS: tuple[str, ...] = (
    "changing the clock set, or dropping a clock whose verdict disappoints",
    "changing anything imported from P6, which would move both hashes",
    "reporting a 4h or 1d verdict as though P5 had already answered it",
    "declaring SWING eligible on a clock that did not pass",
    "refitting a clock that failed, under any pretext",
)

STOPPING_RULE = {
    "on_pass": (
        "a clock that passes is viable on adaptive research folds and its specialist "
        "exists for a mode to name. It is not a deployable strategy."
    ),
    "on_fail": (
        "a clock that fails is recorded as failing. A SWING mode may still be *described*, "
        "and must be marked not eligible for anything that would claim alpha for it."
    ),
}

ARTIFACT_POLICY = {
    "cells": "artifacts/benchmark/btc_p6ext_{clock}_{model}/",
    "decision": "artifacts/benchmark/btc_p6ext_decision/",
    "manifest": "artifacts/btc_p6ext_SHA256SUMS.txt",
    "reruns": "a valid negative cell is never re-run",
}


def payload() -> dict[str, Any]:
    """Everything the hash covers, including everything imported from P6."""
    return {
        "checkpoint": CHECKPOINT,
        "question": QUESTION,
        "hypothesis": HYPOTHESIS,
        "evidence_ceiling": EVIDENCE_CEILING,
        "research_classification": RESEARCH_CLASSIFICATION,
        "clocks": list(CLOCKS),
        "horizons": HORIZONS,
        "extends": EXTENDS,
        "not_p5": NOT_P5,
        "source": SOURCE,
        "feature_engine": FEATURE_ENGINE,
        "horizon_bars": HORIZON_BARS,
        "costs": COSTS,
        "target": TARGET,
        "seq_len": SEQ_LEN,
        "region": REGION,
        "fold_periods": [dict(row) for row in FOLD_PERIODS],
        "fold_policy": FOLD_POLICY,
        "models": list(MODELS),
        "primary_model": PRIMARY_MODEL,
        "seed": SEED,
        "seed_policy": SEED_POLICY,
        "threshold": THRESHOLD,
        "baselines": BASELINES,
        "viability_gate": VIABILITY_GATE,
        "diagnostics": DIAGNOSTICS,
        "measured_universe": MEASURED_UNIVERSE,
        "thinness_note": THINNESS_NOTE,
        "styx_prohibition": STYX_PROHIBITION,
        "p4_hold_unavailability": P4_HOLD_UNAVAILABILITY,
        "forbidden_after_results": list(FORBIDDEN_AFTER_RESULTS),
        "stopping_rule": STOPPING_RULE,
        "artifact_policy": ARTIFACT_POLICY,
        "leakage_battery": [dict(item) for item in LEAKAGE_BATTERY],
        "held_fixed": HELD_FIXED,
    }


def preregistration_hash() -> str:
    blob = json.dumps(payload(), sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(blob.encode()).hexdigest()


def describe() -> dict[str, Any]:
    return {"preregistration_hash": preregistration_hash(), **payload()}


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(describe(), indent=2))
