"""Two-sided synthetic controls for decision modules whose committed evidence is negative.

The frozen P5/P7 evidence tests prove that real artifacts recompute to the published
negative answers. These controls prove the same decision functions can also return a
supportive answer when synthetic inputs satisfy the preregistered conditions, and that
one satisfied condition cannot rescue failure of the other.
"""

from __future__ import annotations

from nn.p5_decision import OUTCOME_NEGATIVE, OUTCOME_SUPPORTIVE, decide as decide_p5
from nn.p5_preregistration import PRIMARY_COMPARISON, PRIMARY_MODEL
from nn.p7_decision import VERDICT_NEGATIVE, VERDICT_SUPPORTIVE, verdict_for


def _p5_cell(returns: list[float], trades: int = 20) -> dict:
    folds = []
    for fold, value in enumerate(returns):
        folds.append(
            {
                "fold": fold,
                "outer_validation": {
                    PRIMARY_MODEL: {
                        "trading": {
                            "net_return": value,
                            "n_trades": trades,
                        }
                    }
                },
                "periods": {
                    "outer_validation": {
                        "start": f"2025-01-{fold + 1:02d}T00:00:00+00:00",
                        "end": f"2025-01-{fold + 2:02d}T00:00:00+00:00",
                    }
                },
            }
        )
    return {"model": PRIMARY_MODEL, "folds": folds, "_dir": "synthetic"}


def _p5_cells(control: list[float], combined: list[float]) -> dict:
    combined_arm, control_arm = PRIMARY_COMPARISON
    return {
        (PRIMARY_MODEL, control_arm): _p5_cell(control),
        (PRIMARY_MODEL, combined_arm): _p5_cell(combined),
    }


def test_p5_decision_has_a_real_supportive_path():
    result = decide_p5(
        _p5_cells(
            control=[0.00, 0.00, 0.00, 0.00],
            combined=[0.10, 0.10, 0.10, -0.01],
        ),
        {"gate_passed": True},
    )
    assert result["improved_folds"] == 3
    assert result["passed"] is True
    assert result["outcome"] == OUTCOME_SUPPORTIVE


def test_p5_positive_mean_cannot_replace_the_three_fold_gate():
    result = decide_p5(
        _p5_cells(
            control=[0.00, 0.00, 0.00, 0.00],
            combined=[0.20, 0.20, -0.01, -0.01],
        ),
        {"gate_passed": True},
    )
    assert result["descriptive"]["mean_delta"] > 0
    assert result["improved_folds"] == 2
    assert result["passed"] is False
    assert result["outcome"] == OUTCOME_NEGATIVE


def _p7_payload(deltas: list[float]) -> dict:
    folds = []
    for fold, delta in enumerate(deltas):
        folds.append(
            {
                "fold": fold,
                "period_start": f"2025-02-{fold + 1:02d}T00:00:00+00:00",
                "period_end": f"2025-02-{fold + 2:02d}T00:00:00+00:00",
                "delta": delta,
                "consensus": {
                    "net_return": 0.10 + delta,
                    "n_trades": 20,
                    "turnover": 1.0,
                    "signal_counts": {"HOLD": 100},
                },
                "best_constituent": {"clock": "1m", "net_return": 0.10},
                "decision_rows": 1000,
            }
        )
    return {
        "mode": {"mode": "SCALPING", "decision_clock": "1m"},
        "consensus_rule": {
            "specialists": ["1m", "5m", "15m"],
            "agreement_required": 2,
            "veto_specialist": "15m",
        },
        "folds": folds,
    }


def test_p7_decision_has_a_real_supportive_path():
    verdict = verdict_for(_p7_payload([0.10, 0.10, 0.10, -0.01]))
    assert verdict["conditions"]["improved_folds"]["passed"] is True
    assert verdict["conditions"]["mean_fold_delta"]["passed"] is True
    assert verdict["verdict"] == VERDICT_SUPPORTIVE


def test_p7_positive_mean_cannot_replace_three_improved_folds():
    verdict = verdict_for(_p7_payload([0.20, 0.20, -0.01, -0.01]))
    assert verdict["conditions"]["improved_folds"]["passed"] is False
    assert verdict["conditions"]["mean_fold_delta"]["passed"] is True
    assert verdict["verdict"] == VERDICT_NEGATIVE


def test_p7_three_positive_folds_cannot_replace_positive_mean():
    verdict = verdict_for(_p7_payload([0.01, 0.01, 0.01, -1.00]))
    assert verdict["conditions"]["improved_folds"]["passed"] is True
    assert verdict["conditions"]["mean_fold_delta"]["passed"] is False
    assert verdict["verdict"] == VERDICT_NEGATIVE
