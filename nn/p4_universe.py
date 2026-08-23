"""P4's common sample universe, and the availability gate that precedes it.

Two things live here, and they are the two that decide whether P4 can say
anything at all.

**The universe (§6.2).** Three arms are compared, and a comparison of two
information sets is only a comparison of information sets while both are scored
on the same rows. The rule is an intersection of four conditions, it is computed
**once**, and every arm — the ``ohlcv14`` control included — is scored on the
result. That is the failure this module exists to make impossible: if the
derivatives arm were scored where derivatives data happens to exist and the
control everywhere, the delta would measure the difference between two market
periods and would be reported as the difference between two information sets.

**The gate (§3.6, §8.0).** A block is *available* when at least 98% of its rows
survive the universe conditions and no contiguous run of excluded rows exceeds
48 hours. If fewer than two exploratory blocks and P4-HOLD are available, P4 is
``not_evaluable`` / ``insufficient_coverage`` — **not** negative. Nothing was
measured. That distinction is the whole reason this is computed before a model
is fitted rather than reported beside one.

**One arithmetic consequence, recorded because it is larger than §3.0a's
worked example.** ``drv_oi_log_change_24h`` and ``drv_oi_price_divergence`` read
the open-interest snapshot at ``t`` *and* at ``t − 24h``. A missing metrics day
therefore removes its own 24 hours **and** the 24 hours exactly one day later:
48 rows per missing day, not 24. That follows from §5's feature definitions and
from §3.4's ban on carrying an observation forward past its bound; it is not a
choice this module makes, and :func:`block_availability` reports the excluded
runs so the number is visible rather than inferred.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from nn.derivatives import (
    DERIVATIVES_FEATURE_COLUMNS,
    DerivativesError,
    DerivativesSpec,
    compute_derivatives_features,
    undefined_reasons,
)
from nn.p4_preregistration import (
    AVAILABILITY_GATE,
    BLOCK_AVAILABILITY_RULE,
    EXPLORATORY_OUTER_BLOCKS,
    HOLDOUT_ROWS,
    NOT_EVALUABLE_OUTCOME,
    UNIVERSE_CONDITIONS,
    preregistration_hash,
)

#: Names the meaning of a universe identity. Recorded in every P4 cell, so two
#: cells built over different intersections cannot be joined by a comparison
#: that only checked their fold plans.
UNIVERSE_SCHEMA = "chimera.p4-universe/1"


class UniverseError(ValueError):
    """P4's arms cannot be put on one set of rows honestly."""


@dataclass(frozen=True)
class Universe:
    """Which spine rows all three P4 arms are scored on, and why the rest are not."""

    #: One flag per research-spine row. The *only* thing every arm reads.
    eligible: np.ndarray
    #: ``(n_spine_rows, 8)`` of ``derivatives_v1``, zero where ineligible.
    features: np.ndarray
    feature_names: tuple[str, ...]
    #: How many rows each of §6.2's four conditions removed, and per column.
    excluded_by: dict[str, Any]
    universe_hash: str
    spec_hash: str
    rows_eligible: int
    rows_total: int
    first_eligible_row: int | None
    last_eligible_row: int | None
    first_eligible_instant: str | None
    last_eligible_instant: str | None
    join_evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "universe_schema": UNIVERSE_SCHEMA,
            "universe_hash": self.universe_hash,
            "derivatives_spec_hash": self.spec_hash,
            "preregistration_hash": preregistration_hash(),
            "rows_eligible": self.rows_eligible,
            "rows_total": self.rows_total,
            "first_eligible_row": self.first_eligible_row,
            "last_eligible_row": self.last_eligible_row,
            "first_eligible_instant": self.first_eligible_instant,
            "last_eligible_instant": self.last_eligible_instant,
            "excluded_by": self.excluded_by,
            "conditions": [dict(entry) for entry in UNIVERSE_CONDITIONS],
            "applied_to": "every arm identically, with the ohlcv14 control re-run on it",
            "join_evidence": self.join_evidence,
        }


def _mask_hash(mask: np.ndarray) -> str:
    """A stable digest of which rows survived. Cheap enough to store in every cell."""
    packed = np.ascontiguousarray(np.asarray(mask, dtype=bool))
    return hashlib.sha256(
        np.packbits(packed).tobytes() + str(len(packed)).encode()
    ).hexdigest()


def index_hash(idx: np.ndarray) -> str:
    """A stable digest of a sample-index array, in :mod:`nn.information_sets`'s form."""
    return hashlib.sha256(np.ascontiguousarray(idx, dtype=np.int64).tobytes()).hexdigest()


def frame_segment_ids(frame_dates: pd.Series, spine_dates: pd.Series) -> np.ndarray:
    """Segment ids for the hourly frame that respect the **spine's** boundaries.

    §6.2's fourth condition is about the *spine's* segments, and they are
    stricter than the candle history's hour gaps: the dataset builder drops an
    indicator warm-up after every gap, so a two-hour hole in the candles becomes
    a three-and-a-half-day hole in the spine. A trailing window that reset only
    at the candles' gaps would reach back across the dropped warm-up and bridge a
    boundary the fold geometry believes is there.

    So a new segment starts at an hour gap in the frame **or** at any hour that
    begins a run of spine rows. The result is at least as strict as both.
    """
    frame = pd.DatetimeIndex(pd.to_datetime(frame_dates, utc=True))
    spine = pd.DatetimeIndex(pd.to_datetime(spine_dates, utc=True))
    if not frame.is_monotonic_increasing or frame.has_duplicates:
        raise UniverseError("the hourly frame must be strictly increasing in time")

    step = frame.to_series().diff()
    new = (step != pd.Timedelta(hours=1)).to_numpy(dtype=bool, copy=True)
    new[0] = True

    spine_step = spine.to_series().diff()
    spine_starts = spine[(spine_step != pd.Timedelta(hours=1)).to_numpy(dtype=bool, copy=True)]
    if len(spine):
        spine_starts = spine_starts.union(spine[:1])
    positions = frame.get_indexer(spine_starts)
    if (positions < 0).any():
        missing = spine_starts[positions < 0]
        raise UniverseError(
            f"{len(missing)} spine segment start(s) are absent from the hourly frame "
            f"(first: {missing[0].isoformat()}). The derivatives source does not cover "
            "the candles it is being aligned to."
        )
    new[positions] = True
    return np.cumsum(new) - 1


def build_universe(
    spine: pd.DataFrame,
    hourly: pd.DataFrame,
    raw_candles: pd.DataFrame,
    *,
    spec: DerivativesSpec | None = None,
) -> Universe:
    """The intersection of §6.2's four conditions, computed once for every arm.

    ``spine`` is the processed research dataset — the rows the fold plan is
    expressed in, which is condition 1 by construction. ``hourly`` is the
    committed derivatives source. ``raw_candles`` is the unprocessed candle
    history: it supplies the spot close the basis divides by and the price
    divergence differences, and its hours are the frame the trailing windows run
    over, because a window measured in hours must be measured on hours.

    Fails closed on anything that would move a row: a spine timestamp the
    derivatives source does not cover, a spine segment the frame does not break
    at, a non-finite feature on a row the engine called defined.
    """
    spec = spec or DerivativesSpec()
    spine = spine.reset_index(drop=True)
    spine_dates = pd.to_datetime(spine["date"], utc=True)

    raw = raw_candles.reset_index(drop=True).copy()
    raw["date"] = pd.to_datetime(raw["date"], utc=True)
    grid = pd.DatetimeIndex(pd.to_datetime(hourly["date"], utc=True))
    if not grid.is_monotonic_increasing or grid.has_duplicates:
        raise UniverseError("the derivatives source must be strictly increasing in time")

    # The frame is the candle hours the derivatives source covers, and nothing
    # later than the last spine row: the engine is causal, but truncating makes
    # that structural here rather than inherited, exactly as the OHLCV raw
    # candles are truncated in `nn.information_sets.build_information_set_views`.
    last_used = spine_dates.iloc[-1]
    in_window = raw["date"].isin(pd.Series(grid)) & (raw["date"] <= last_used)
    frame_candles = raw.loc[in_window].reset_index(drop=True)
    if frame_candles.empty:
        raise UniverseError(
            "no candle hour is inside the derivatives source's window; the two sources "
            "do not describe the same period"
        )
    positions = grid.get_indexer(pd.DatetimeIndex(frame_candles["date"]))
    if (positions < 0).any():
        raise UniverseError("a candle hour vanished from the derivatives source mid-join")
    frame = hourly.iloc[positions].reset_index(drop=True)

    segments = frame_segment_ids(frame["date"], spine_dates)
    features = compute_derivatives_features(
        frame,
        frame_candles["close"].to_numpy(dtype=np.float64),
        spec,
        segment_ids=segments,
    )

    frame_index = pd.DatetimeIndex(pd.to_datetime(frame["date"], utc=True))
    rows = frame_index.get_indexer(pd.DatetimeIndex(spine_dates))
    if (rows < 0).any():
        missing = int((rows < 0).sum())
        first = spine_dates[rows < 0].iloc[0]
        raise UniverseError(
            f"{missing} research row(s) have no derivatives hour (first: {first}). The "
            "derivatives source does not cover the candles it is being aligned to, and "
            "a P4 row for those candles would have to be invented."
        )
    _check_segment_contiguity(spine, rows)

    values = features[list(DERIVATIVES_FEATURE_COLUMNS)].to_numpy(dtype=np.float64)[rows]
    defined = features["defined"].to_numpy(dtype=bool)[rows]
    if not np.isfinite(values[defined]).all():
        raise DerivativesError(
            "a derivatives column is non-finite on a row the engine called defined"
        )
    values = np.where(defined[:, None], values, 0.0)

    on_spine = features.iloc[rows].reset_index(drop=True)
    per_column = undefined_reasons(on_spine)
    warmup_short = per_column.pop("warmup_hours")
    excluded = {
        # Condition 1 is satisfied by construction: `spine` *is* the OHLCV14
        # research spine, so a row that failed it is not in this array at all.
        "ohlcv14_row": 0,
        "derivatives_undefined": int((~defined).sum()),
        "derivatives_warmup_hours": warmup_short,
        "per_column": per_column,
        "hourly_frame_rows": int(len(frame)),
        "note": (
            "a missing open-interest day removes its own 24 hours and the 24 hours "
            "exactly one day later, because drv_oi_log_change_24h and "
            "drv_oi_price_divergence read the snapshot at t and at t - 24h and §3.4 "
            "forbids carrying either forward"
        ),
    }
    eligible_rows = np.flatnonzero(defined)
    return Universe(
        eligible=defined,
        features=values,
        feature_names=tuple(DERIVATIVES_FEATURE_COLUMNS),
        excluded_by=excluded,
        universe_hash=_mask_hash(defined),
        spec_hash=spec.spec_hash(),
        rows_eligible=int(defined.sum()),
        rows_total=int(len(defined)),
        first_eligible_row=int(eligible_rows[0]) if eligible_rows.size else None,
        last_eligible_row=int(eligible_rows[-1]) if eligible_rows.size else None,
        first_eligible_instant=(
            spine_dates.iloc[int(eligible_rows[0])].isoformat() if eligible_rows.size else None
        ),
        last_eligible_instant=(
            spine_dates.iloc[int(eligible_rows[-1])].isoformat()
            if eligible_rows.size
            else None
        ),
        join_evidence=_join_evidence(frame, rows, spine_dates, values, features),
    )


def _check_segment_contiguity(spine: pd.DataFrame, rows: np.ndarray) -> None:
    """Two spine rows in one segment must be adjacent hours in the derivatives frame.

    The same check :func:`nn.information_sets._check_trade_segment_contiguity`
    makes of the trade source, and for the same reason: the spine's segment ids
    decide which windows may span which rows, and if the two sources disagreed a
    trailing window would carry state across a gap the windowing believes is not
    there.
    """
    if "segment_id" not in spine.columns:
        raise UniverseError(
            "the research spine has no 'segment_id' column, so a derivatives window "
            "could bridge a market-data gap. Rebuild it with the current "
            "tools.build_features."
        )
    segments = spine["segment_id"].to_numpy(dtype=np.int64)
    same_segment = segments[1:] == segments[:-1]
    step = np.diff(rows)
    offenders = np.flatnonzero(same_segment & (step != 1))
    if offenders.size:
        row = int(offenders[0])
        raise UniverseError(
            f"research rows {row} and {row + 1} share segment {int(segments[row])} but "
            f"are {int(step[row])} hours apart in the derivatives source. The two "
            "disagree about the gap structure, so a trailing window would bridge a gap "
            "the windowing does not know about."
        )


def _join_evidence(
    frame: pd.DataFrame,
    rows: np.ndarray,
    spine_dates: pd.Series,
    values: np.ndarray,
    features: pd.DataFrame,
) -> dict[str, Any]:
    """Re-derive one column from the source, two ways, and score a shifted join.

    The join is where a derivatives value would end up on the wrong hour, and a
    one-row shift is both the likeliest mistake and the most damaging: it is a
    one-hour look-ahead on all eight columns at once. ``drv_basis`` is rebuilt
    positionally from the raw perpetual close, and again by timestamp, because a
    positional check alone passes when the join and the re-derivation are shifted
    together.
    """
    column = list(DERIVATIVES_FEATURE_COLUMNS).index("drv_basis")
    joined = values[:, column]
    engine = features["drv_basis"].to_numpy(dtype=np.float64)
    expected = engine[rows]
    # A basis of zero on every row is what an empty universe looks like, and a
    # shift check over it would "match" trivially. That is not a failure of the
    # join — there is no join to prove — so it is reported as inapplicable rather
    # than raised, because §9.0's `not_evaluable` path has to stay reachable: a
    # checkpoint whose archive gave it nothing must be able to say so.
    # The *universe* mask, not the basis column's own: `values` and the engine's
    # column are both zeroed wherever a row is outside the universe, so a row
    # with a defined basis and an undefined open interest carries a zero in both
    # and discriminates nothing.
    defined = features["defined"].to_numpy(dtype=bool)[rows]
    if int(defined.sum()) < 3:
        return {
            "recomputed": [],
            "rows": int(len(rows)),
            "applicable": False,
            "reason": (
                f"{int(defined.sum())} research row(s) have a defined basis, too few for a "
                "shift check to discriminate. No P4 arm can be scored on this universe"
            ),
            "matches": None,
            "matches_under_plus_one_shift": None,
            "matches_under_minus_one_shift": None,
        }
    if not np.allclose(joined, expected, rtol=0.0, atol=1e-12):
        worst = int(np.argmax(np.abs(joined - expected)))
        raise UniverseError(
            f"the derivatives join is wrong: drv_basis at research row {worst} is "
            f"{joined[worst]!r} but the source hour gives {expected[worst]!r}. A value "
            "is sitting on the wrong hour."
        )
    by_hour = pd.Series(
        engine, index=pd.DatetimeIndex(pd.to_datetime(frame["date"], utc=True))
    )
    looked_up = by_hour.reindex(pd.DatetimeIndex(spine_dates)).to_numpy(dtype=np.float64)
    if not np.allclose(joined, looked_up, rtol=0.0, atol=1e-12):
        raise UniverseError(
            "the derivatives join is wrong: drv_basis does not match the source hour "
            "carrying that timestamp"
        )

    def shifted_matches(offset: int) -> bool:
        rolled = np.roll(expected, offset)
        live = defined[1:-1]
        return bool(np.allclose(joined[1:-1][live], rolled[1:-1][live], rtol=0.0, atol=1e-12))

    evidence = {
        "applicable": True,
        "recomputed": [
            "drv_basis from the derivatives source, positionally",
            "drv_basis from the derivatives source, by timestamp",
        ],
        "rows": int(len(rows)),
        "matches": True,
        "matches_under_plus_one_shift": shifted_matches(1),
        "matches_under_minus_one_shift": shifted_matches(-1),
        "source_rows_first": int(rows[0]),
        "source_rows_last": int(rows[-1]),
        "warmup_hours_before_the_spine": int(rows[0]),
    }
    if evidence["matches_under_plus_one_shift"] or evidence["matches_under_minus_one_shift"]:
        raise UniverseError(
            "the join check cannot tell a correct join from a shifted one on this data, "
            "so it is not evidence and must not be reported as though it were"
        )
    return evidence


# --------------------------------------------------------------------------- #
# the availability gate
# --------------------------------------------------------------------------- #
def _longest_false_run(flags: np.ndarray) -> int:
    """Longest run of ``False``, which is §8.0's contiguous-outage statistic."""
    values = np.asarray(flags, dtype=bool)
    if values.all():
        return 0
    padded = np.concatenate(([True], values, [True]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    starts, ends = edges[0::2], edges[1::2]
    return int((ends - starts).max()) if len(starts) else 0


def block_availability(
    eligible: np.ndarray, block: Sequence[int], *, label: str
) -> dict[str, Any]:
    """Whether one block is *available* under §8.0, and every number behind it.

    Two conditions and no third: at least 98% of the block's rows survive the
    universe conditions, and no contiguous run of excluded rows exceeds 48 hours.
    Both numbers come from :data:`nn.p4_preregistration.BLOCK_AVAILABILITY_RULE`
    rather than from here.

    Reported whatever it decides. A block that fails is *unavailable*, is
    excluded from stage 1's fold count entirely, and is never included at a
    discount.
    """
    start, end = int(block[0]), int(block[1])
    window = np.asarray(eligible, dtype=bool)[start:end]
    total = end - start
    if len(window) != total:
        raise UniverseError(
            f"{label} spans rows [{start}, {end}) but only {len(window)} of them are in "
            "the universe mask. A block that is not fully covered cannot be judged "
            "available at a discount."
        )
    surviving = int(window.sum())
    fraction = surviving / total if total else 0.0
    outage = _longest_false_run(window)
    min_fraction = float(BLOCK_AVAILABILITY_RULE["min_surviving_row_fraction"])
    max_outage = int(BLOCK_AVAILABILITY_RULE["max_contiguous_missing_hours"])
    reasons = []
    if fraction < min_fraction:
        reasons.append(
            f"{surviving}/{total} rows survive ({fraction:.4f}); at least {min_fraction} "
            "is required"
        )
    if outage > max_outage:
        reasons.append(
            f"the longest contiguous run of excluded rows is {outage}h; at most "
            f"{max_outage}h is permitted"
        )
    return {
        "label": label,
        "block": [start, end],
        "rows": total,
        "rows_surviving": surviving,
        "surviving_fraction": round(fraction, 6),
        "max_contiguous_missing_hours": outage,
        "available": not reasons,
        "reasons": reasons,
        "rule": dict(BLOCK_AVAILABILITY_RULE),
    }


def availability_gate(
    blocks: Sequence[Mapping[str, Any]], holdout: Mapping[str, Any] | None
) -> dict[str, Any]:
    """§3.6's gate: enough blocks and P4-HOLD, or ``not_evaluable``.

    ``holdout`` is P4-HOLD's own availability under the same rule. It is a
    separate argument because stage 1's source file structurally does not contain
    the region — that is the point of §6.3 — so its coverage is established from
    archive metadata (:func:`holdout_coverage_from_archive_days`) rather than
    from rows nobody may read yet, and recomputed exactly when the stage-2
    snapshot is exported.

    A failure here is **not** a negative result. Nothing was measured, and the
    outcome is the ``not_evaluable`` / ``insufficient_coverage`` row of §9.0 with
    the per-block coverage that produced it.
    """
    available = [entry for entry in blocks if entry["available"]]
    required = int(AVAILABILITY_GATE["requires_exploratory_blocks_available"])
    holdout_ok = bool(holdout and holdout.get("available"))
    reasons = []
    if len(available) < required:
        reasons.append(f"{len(available)} exploratory block(s) available; {required} required")
    if AVAILABILITY_GATE["requires_holdout_available"] and not holdout_ok:
        reasons.append(
            "P4-HOLD is not available under the block rule"
            + (f": {holdout.get('reasons')}" if holdout else " (no coverage established)")
        )
    passed = not reasons
    return {
        "gate_passed": passed,
        "reasons": reasons,
        "exploratory_blocks": list(blocks),
        "exploratory_blocks_available": len(available),
        "holdout": dict(holdout) if holdout else None,
        "rule": dict(AVAILABILITY_GATE),
        "preregistration_hash": preregistration_hash(),
        "outcome": None if passed else dict(NOT_EVALUABLE_OUTCOME),
        "note": (
            "a failed gate is a statement about what public archives contain. It is not "
            "evidence about whether derivatives positioning carries information, and it "
            "may not be reported as a negative P4"
        ),
    }


def evaluate_blocks(universe: Universe) -> list[dict[str, Any]]:
    """§8.0 applied to each of the four preregistered exploratory blocks."""
    return [
        block_availability(universe.eligible, block, label=f"outer_block_{index}")
        for index, block in enumerate(EXPLORATORY_OUTER_BLOCKS)
    ]


def holdout_coverage_from_archive_days(
    published_days: Mapping[str, bool], *, hours_per_day: int = 24
) -> dict[str, Any]:
    """P4-HOLD's availability, estimated from which archive days exist. No rows read.

    §8.0 applies the block rule to P4-HOLD, and stage 1 must know the answer
    before it decides whether to continue — but stage 1's source file stops at
    row 45802 and does not contain the region. The only outcome-blind way to
    answer is to ask the archive *which days it publishes*, which is metadata:
    no price, no label, no return and no prediction is read, and nothing here
    returns holdout data.

    ``published_days`` maps each UTC day of the region to whether the metrics
    archive publishes it. The estimate is deliberately **conservative**: an
    absent day costs its own 24 hours and the 24 hours after it, because
    ``drv_oi_log_change_24h`` reads ``t`` and ``t − 24h``. The exact figure is
    recomputed from rows when the stage-2 snapshot is exported, and a release
    records both.
    """
    days = sorted(published_days)
    if not days:
        raise UniverseError(
            "no archive days were established for P4-HOLD, so its availability is "
            "unknown. An unknown coverage is not an available one."
        )
    absent = {day for day in days if not published_days[day]}
    lost_days: set[str] = set()
    for index, day in enumerate(days):
        if day in absent:
            lost_days.add(day)
            if index + 1 < len(days):
                lost_days.add(days[index + 1])
    total_rows = HOLDOUT_ROWS[1] - HOLDOUT_ROWS[0]
    lost_rows = min(len(lost_days) * hours_per_day, total_rows)
    surviving = total_rows - lost_rows
    fraction = surviving / total_rows if total_rows else 0.0
    outage = _longest_run_of_days(days, lost_days) * hours_per_day
    min_fraction = float(BLOCK_AVAILABILITY_RULE["min_surviving_row_fraction"])
    max_outage = int(BLOCK_AVAILABILITY_RULE["max_contiguous_missing_hours"])
    reasons = []
    if fraction < min_fraction:
        reasons.append(
            f"an estimated {surviving}/{total_rows} rows survive ({fraction:.4f}); at "
            f"least {min_fraction} is required"
        )
    if outage > max_outage:
        reasons.append(
            f"the longest estimated contiguous outage is {outage}h; at most "
            f"{max_outage}h is permitted"
        )
    return {
        "label": "p4_hold",
        "block": list(HOLDOUT_ROWS),
        "rows": total_rows,
        "rows_surviving": surviving,
        "surviving_fraction": round(fraction, 6),
        "max_contiguous_missing_hours": outage,
        "available": not reasons,
        "reasons": reasons,
        "basis": "archive-day metadata only; no P4-HOLD row was read",
        "days_checked": len(days),
        "days_absent": sorted(absent),
        "rule": dict(BLOCK_AVAILABILITY_RULE),
        "exact_figure_recomputed_at": "stage-2 snapshot export",
    }


def _longest_run_of_days(days: Sequence[str], lost: set[str]) -> int:
    longest = current = 0
    for day in days:
        current = current + 1 if day in lost else 0
        longest = max(longest, current)
    return longest


def fold_sample_hashes(
    universe: Universe, folds: Sequence[Any], seq_len: int, horizon: int, spine: pd.DataFrame
) -> dict[str, Any]:
    """The per-fold sample indices every arm must agree on, and their digests.

    Computed from the universe mask and the fold plan alone — no model, no
    features, no arm. ``nn.p2b_compare``'s parity check refuses to join two cells
    whose per-fold sample-index hashes disagree, so recording these is what makes
    §6.2's "the universe is applied to every arm identically" enforced rather
    than promised.
    """
    from nn.dataset import sample_indices

    segments = spine["segment_id"].to_numpy(dtype=np.int64)
    out = []
    for index, plan in enumerate(folds):
        entry: dict[str, Any] = {"fold": index}
        for name in ("train", "inner", "outer"):
            split = getattr(plan, name)
            idx = sample_indices(
                split,
                seq_len,
                horizon,
                segment_ids=segments,
                eligible=universe.eligible,
            )
            entry[name] = {
                "block": [int(split.start), int(split.end)],
                "samples": int(len(idx)),
                "first_row": int(idx[0]) if len(idx) else None,
                "last_row": int(idx[-1]) if len(idx) else None,
                "index_hash": index_hash(idx),
            }
        out.append(entry)
    return {
        "universe_hash": universe.universe_hash,
        "seq_len": seq_len,
        "horizon": horizon,
        "folds": out,
        "note": (
            "identical across every arm by construction: the mask and the fold plan are "
            "the only inputs, and neither is a function of the information set"
        ),
    }


def universe_identity(universe: Universe, availability: Mapping[str, Any]) -> dict[str, Any]:
    """One object binding the universe, the gate and the preregistration together."""
    material = {
        "universe_hash": universe.universe_hash,
        "derivatives_spec_hash": universe.spec_hash,
        "preregistration_hash": preregistration_hash(),
        "rows_eligible": universe.rows_eligible,
        "rows_total": universe.rows_total,
        "gate_passed": bool(availability.get("gate_passed")),
    }
    return {
        **material,
        "identity": hashlib.sha256(
            json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
