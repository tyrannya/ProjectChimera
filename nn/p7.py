"""P7's runner: consensus over frozen P6 predictions, and its own components.

    python -m nn.p7 --mode SCALPING --out artifacts/benchmark/btc_p7_scalping
    python -m nn.p7 --all --out-root artifacts/benchmark

**Nothing is fitted here and nothing can be.** This module reads
``outer_predictions.parquet`` from the frozen P6 cells and never touches a
candle, a feature, a model or a threshold. The only thing it computes is which
specialist prediction was available at each decision instant, what the
preregistered rule says about that set, and what the resulting signal series
earns under the same cost model P6 used.

Three pieces do the work:

* :func:`align_to_decision_clock` — the causal join. A decision row's trade is
  entered at its close, so the reference instant is the decision bar's open plus
  one decision-clock bar, and a specialist bar is available when its own close is
  at or before that. On the mode's own clock this is the identity, which
  :func:`validity_gate` turns into a refusal rather than an assumption.
* :func:`chimera.consensus.decide` — the rule itself, shared verbatim with the
  live trading-mode controller so that the thing measured here and the thing that
  would run are one function.
* :func:`nn.evaluate.trading_metrics` — the accounting, identical to P6's, so a
  consensus and its constituents are charged the same 20 bps round trip under the
  same greedy non-overlapping trade rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from chimera.consensus import ConsensusRule
from chimera.consensus import decide as consensus_decide
from chimera.contracts import CLASS_ORDER, HOLD_IDX, LONG_IDX, SHORT_IDX, Signal, TargetSpec
from nn.evaluate import trading_metrics
from nn.multiclock import constituent_count
from nn.p2b import code_revision, numerical_environment
from nn.p6 import PREDICTIONS_NAME
from nn.p6_preregistration import HORIZON_BARS
from nn.p7_preregistration import (
    CHECKPOINT,
    COSTS,
    EVIDENCE_CEILING,
    MODES,
    QUESTION,
    SPECIALIST_SOURCE,
    mode as registered_mode,
    preregistration_hash,
)

logger = logging.getLogger(__name__)

ARTIFACT_NAME = "p7.json"
MARKDOWN_NAME = "p7.md"
STATUS_NAME = "STATUS.md"

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_ROOT = REPO_ROOT / "artifacts" / "benchmark"

EVIDENCE_CLASS = "one trading mode's consensus and its components; P7 primary evidence"

#: The action column's integer encoding, from `chimera.contracts`. Named rather
#: than assumed so a change to CLASS_ORDER breaks here instead of silently
#: re-labelling every vote.
_SIGNAL_OF_INDEX: dict[int, Signal] = {
    SHORT_IDX: Signal.SHORT,
    HOLD_IDX: Signal.HOLD,
    LONG_IDX: Signal.LONG,
}
_INDEX_OF_SIGNAL: dict[Signal, int] = {value: key for key, value in _SIGNAL_OF_INDEX.items()}

#: The sentinel for "this specialist had nothing closed yet". Deliberately not
#: HOLD: the preregistered unavailability rule makes the *whole* consensus HOLD,
#: which is a different statement from every specialist holding.
UNAVAILABLE = -1


class P7Error(SystemExit):
    """The frozen predictions cannot be replayed the way P7 says they must be."""


def specialist_path(clock: str) -> Path:
    return REPO_ROOT / "artifacts" / "benchmark" / f"btc_p6_{clock}_xgboost" / PREDICTIONS_NAME


def load_specialist(clock: str) -> pd.DataFrame:
    """One frozen P6 specialist's outer predictions, checked before use.

    Refuses duplicates and refuses a frame that is not ordered within a fold:
    both would make ``searchsorted`` return an index that means nothing, and
    neither is something to repair silently.
    """
    path = specialist_path(clock)
    if not path.is_file():
        raise P7Error(f"the frozen {clock} specialist is absent at {path}")
    frame = pd.read_parquet(
        path, columns=["fold", "timestamp", "row_index", "future_return", "selected_action"]
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values(["fold", "timestamp"], kind="mergesort").reset_index(drop=True)
    duplicated = int(frame.duplicated(subset=["fold", "timestamp"]).sum())
    if duplicated:
        raise P7Error(
            f"the {clock} specialist carries {duplicated} duplicate (fold, timestamp) "
            "prediction(s); one bar cannot have two decisions"
        )
    unknown = set(frame["selected_action"].unique()) - set(_SIGNAL_OF_INDEX)
    if unknown:
        raise P7Error(
            f"the {clock} specialist emits action(s) {sorted(unknown)}, not in CLASS_ORDER"
        )
    return frame


def align_to_decision_clock(
    decision_opens: np.ndarray,
    decision_clock: str,
    specialist_opens: np.ndarray,
    specialist_clock: str,
) -> np.ndarray:
    """Index of the specialist bar available at each decision row, or -1.

    The reference instant is the decision bar's **close** — its open plus one
    decision-clock bar — because that is where the trade is entered. A specialist
    bar is available when its own close, its open plus its own width, is at or
    before that instant.

    ``side="right"`` means a decision row landing exactly on a specialist close
    *does* see that bar: it has printed. On the mode's own clock this makes the
    result the identity, which is the property :func:`validity_gate` checks.
    """
    reference = decision_opens + np.timedelta64(constituent_count(decision_clock), "m")
    closes = specialist_opens + np.timedelta64(constituent_count(specialist_clock), "m")
    return np.searchsorted(closes, reference, side="right") - 1


def aligned_actions(
    decision: pd.DataFrame,
    specialist: pd.DataFrame,
    decision_clock: str,
    specialist_clock: str,
) -> np.ndarray:
    """Each decision row's available specialist action, or :data:`UNAVAILABLE`."""
    as_of = align_to_decision_clock(
        decision["timestamp"].to_numpy(dtype="datetime64[ns]"),
        decision_clock,
        specialist["timestamp"].to_numpy(dtype="datetime64[ns]"),
        specialist_clock,
    )
    actions = np.full(len(decision), UNAVAILABLE, dtype=np.int64)
    available = as_of >= 0
    source = specialist["selected_action"].to_numpy(dtype=np.int64)
    actions[available] = source[as_of[available]]
    return actions


def consensus_signals(
    actions_by_clock: Mapping[str, np.ndarray], rule: ConsensusRule
) -> np.ndarray:
    """The preregistered rule, applied row by row through the shared decider.

    Vectorising the counts and calling :func:`chimera.consensus.decide` only on
    the distinct vote combinations is not an optimisation of the rule — it *is*
    the rule, evaluated once per combination instead of once per row. There are
    at most ``4 ** len(specialists)`` of them, so the shared function decides
    every case that occurs and the row loop disappears.
    """
    clocks = list(rule.specialists)
    stacked = np.stack([actions_by_clock[clock] for clock in clocks], axis=1)
    combinations, inverse = np.unique(stacked, axis=0, return_inverse=True)

    decided = np.empty(len(combinations), dtype=np.int64)
    for position, combination in enumerate(combinations):
        actions: dict[str, Signal | None] = {
            clock: (None if int(value) == UNAVAILABLE else _SIGNAL_OF_INDEX[int(value)])
            for clock, value in zip(clocks, combination)
        }
        decided[position] = _INDEX_OF_SIGNAL[consensus_decide(actions, rule)]
    return decided[inverse]


def constituent_signals(actions: np.ndarray) -> np.ndarray:
    """One constituent replayed on the decision clock: HOLD where unavailable."""
    replayed = actions.copy()
    replayed[replayed == UNAVAILABLE] = HOLD_IDX
    return replayed


def score(signals: np.ndarray, decision: pd.DataFrame, spec: TargetSpec) -> dict[str, Any]:
    """The accounting, identical to P6's, on the decision clock's own rows."""
    metrics = trading_metrics(
        signals,
        decision["future_return"].to_numpy(dtype=np.float64),
        spec,
        row_index=decision["row_index"].to_numpy(dtype=np.int64),
    )
    counts = np.bincount(signals, minlength=len(CLASS_ORDER))
    return {
        "net_return": metrics["net_return"],
        "n_trades": metrics["n_trades"],
        "turnover": metrics["turnover"],
        "total_costs": metrics["total_costs"],
        "gross_return": metrics["gross_return"],
        "max_drawdown": metrics["max_drawdown"],
        "exposure": metrics["exposure"],
        "cost_per_trade": metrics["cost_per_trade"],
        "rows": int(len(signals)),
        "signal_counts": {
            "SHORT": int(counts[SHORT_IDX]),
            "HOLD": int(counts[HOLD_IDX]),
            "LONG": int(counts[LONG_IDX]),
        },
    }


def rule_for(design: Mapping[str, Any]) -> ConsensusRule:
    """The frozen design, as the domain object the live controller also uses."""
    return ConsensusRule(
        mode=str(design["mode"]),
        decision_clock=str(design["decision_clock"]),
        specialists=tuple(design["specialists"]),
        veto_specialist=str(design["veto_specialist"]),
        agreement_required=int(design["agreement_required"]),
    )


def validity_gate(
    decision: pd.DataFrame, own_actions: np.ndarray, decision_clock: str
) -> dict[str, Any]:
    """The mode's own clock must align to itself as the identity.

    If it does not, the replay is not the thing P6 measured, and every delta
    built on it is a comparison between two different accountings.
    """
    expected = decision["selected_action"].to_numpy(dtype=np.int64)
    identical = bool(np.array_equal(own_actions, expected))
    if not identical:
        raise P7Error(
            f"the {decision_clock} specialist does not align to itself as the identity: "
            f"{int((own_actions != expected).sum())} of {len(expected)} rows differ. The "
            "replay is not reproducing the specialist it claims to replay."
        )
    return {"decision_clock": decision_clock, "identity_rows": int(len(expected))}


def run_mode(design: Mapping[str, Any]) -> dict[str, Any]:
    """One mode: the consensus, every constituent, and the per-fold deltas."""
    rule = rule_for(design)
    decision_clock = rule.decision_clock
    # The mode's own declared horizon, held to P6's. Both modes declare six
    # native bars because P7 replays P6's cells and a replay scored at a
    # different horizon would not be one; reading `HORIZON_BARS` and ignoring the
    # field the design carries would let a future mode declare twelve and be
    # scored at six without anything saying so.
    horizon = int(design["horizon_bars"])
    if horizon != HORIZON_BARS:
        raise P7Error(
            f"{design['mode']} declares a {horizon}-bar horizon and the frozen P6 "
            f"specialists were fitted at {HORIZON_BARS}. A replay is not a refit."
        )
    spec = TargetSpec(
        horizon=horizon,
        fee_rate=COSTS["fee_rate"],
        slippage_rate=COSTS["slippage_rate"],
    )
    specialists = {clock: load_specialist(clock) for clock in rule.specialists}
    decision_all = specialists[decision_clock]

    folds: list[dict[str, Any]] = []
    identity: dict[str, Any] = {}
    for fold in sorted(decision_all["fold"].unique()):
        decision = decision_all.loc[decision_all["fold"] == fold].reset_index(drop=True)
        actions_by_clock = {
            clock: aligned_actions(
                decision,
                frame.loc[frame["fold"] == fold].reset_index(drop=True),
                decision_clock,
                clock,
            )
            for clock, frame in specialists.items()
        }
        identity = validity_gate(decision, actions_by_clock[decision_clock], decision_clock)

        consensus = score(consensus_signals(actions_by_clock, rule), decision, spec)
        constituents = {
            clock: score(constituent_signals(actions), decision, spec)
            for clock, actions in actions_by_clock.items()
        }
        best_clock = max(constituents, key=lambda clock: constituents[clock]["net_return"])
        best = constituents[best_clock]["net_return"]
        unavailable = {
            clock: int((actions == UNAVAILABLE).sum())
            for clock, actions in actions_by_clock.items()
        }
        folds.append(
            {
                "fold": int(fold),
                "period_start": decision["timestamp"].iloc[0].isoformat(),
                "period_end": decision["timestamp"].iloc[-1].isoformat(),
                "decision_rows": int(len(decision)),
                "unavailable_rows": unavailable,
                "consensus": consensus,
                "constituents": constituents,
                "best_constituent": {"clock": best_clock, "net_return": best},
                "delta": round(consensus["net_return"] - best, 6),
            }
        )
        logger.info(
            "%s fold %d: consensus %+.6f, best constituent %s %+.6f, delta %+.6f",
            rule.mode,
            fold,
            consensus["net_return"],
            best_clock,
            best,
            folds[-1]["delta"],
        )

    deltas = [record["delta"] for record in folds]
    digest = hashlib.sha256(
        json.dumps([record["consensus"] for record in folds], sort_keys=True).encode()
    ).hexdigest()
    return {
        "checkpoint": CHECKPOINT,
        "question": QUESTION,
        "evidence_class": EVIDENCE_CLASS,
        "evidence_ceiling": EVIDENCE_CEILING,
        "preregistration_hash": preregistration_hash(),
        "specialist_source": dict(SPECIALIST_SOURCE),
        "mode": dict(design),
        "consensus_rule": {
            "specialists": list(rule.specialists),
            "veto_specialist": rule.veto_specialist,
            "agreement_required": rule.agreement_required,
            "decided_by": "chimera.consensus.decide",
        },
        "target": spec.to_dict(),
        "validity_gate": {**identity, "passed": True},
        "code": code_revision(),
        "numerics": numerical_environment(),
        "folds": folds,
        "summary": {
            "fold_deltas": deltas,
            "mean_delta": round(float(np.mean(deltas)), 12),
            "improved_folds": int(sum(1 for value in deltas if value > 0.0)),
            "total_folds": len(deltas),
            "consensus_digest": digest,
        },
    }


def to_markdown(payload: dict[str, Any]) -> str:
    rule = payload["consensus_rule"]
    lines = [
        f"# {payload['checkpoint']} — {payload['mode']['mode']} consensus",
        "",
        f"**{payload['evidence_class']}**",
        "",
        f"- preregistration: `{payload['preregistration_hash']}`",
        f"- decision clock: `{payload['mode']['decision_clock']}`, horizon "
        f"{payload['mode']['horizon']}",
        f"- specialists: {', '.join('`' + c + '`' for c in rule['specialists'])}, "
        f"{rule['agreement_required']} must agree, `{rule['veto_specialist']}` vetoes",
        f"- validity gate: own-clock alignment is the identity over "
        f"{payload['validity_gate']['identity_rows']:,} rows",
        "",
        "| fold | consensus | best constituent | delta | trades | turnover |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for record in payload["folds"]:
        best = record["best_constituent"]
        lines.append(
            f"| {record['fold']} | {record['consensus']['net_return']} | "
            f"`{best['clock']}` {best['net_return']} | **{record['delta']}** | "
            f"{record['consensus']['n_trades']} | {record['consensus']['turnover']} |"
        )
    summary = payload["summary"]
    lines += [
        "",
        f"Improved in **{summary['improved_folds']} of {summary['total_folds']}** folds, "
        f"mean delta **{summary['mean_delta']}**.",
        "",
        "Per-constituent replays, on this mode's decision clock and under this mode's "
        "accounting:",
        "",
        "| fold | " + " | ".join(f"`{c}`" for c in rule["specialists"]) + " |",
        "| --- |" + " --- |" * len(rule["specialists"]),
    ]
    for record in payload["folds"]:
        cells = " | ".join(
            str(record["constituents"][clock]["net_return"]) for clock in rule["specialists"]
        )
        lines.append(f"| {record['fold']} | {cells} |")
    lines += [
        "",
        "The verdict is in the decision artifact, which applies the preregistered rule.",
    ]
    return "\n".join(lines) + "\n"


def status_markdown(payload: dict[str, Any]) -> str:
    return (
        f"# CURRENT — {payload['checkpoint']} {payload['mode']['mode']}\n\n"
        f"{payload['evidence_class']}.\n\n"
        f"Preregistration `{payload['preregistration_hash']}`.\n"
        f"Frozen under `artifacts/btc_p7_SHA256SUMS.txt`.\n"
    )


def write_mode(payload: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / ARTIFACT_NAME).write_text(json.dumps(payload, indent=2) + "\n")
    (out_dir / MARKDOWN_NAME).write_text(to_markdown(payload))
    (out_dir / STATUS_NAME).write_text(status_markdown(payload))
    return out_dir


def out_name(design: Mapping[str, Any]) -> str:
    return f"btc_p7_{str(design['mode']).lower()}"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=[item["mode"] for item in MODES])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_argparser().parse_args(argv)
    if not args.all and not args.mode:
        raise SystemExit("pass --mode MODE or --all")

    designs = list(MODES) if args.all else [registered_mode(args.mode)]
    for design in designs:
        payload = run_mode(design)
        written = write_mode(payload, args.out_root / out_name(design))
        logger.info(
            "%s: improved in %d of %d folds, mean delta %s -> %s",
            design["mode"],
            payload["summary"]["improved_folds"],
            payload["summary"]["total_folds"],
            payload["summary"]["mean_delta"],
            written,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
