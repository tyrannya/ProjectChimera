"""Training and CLI smoke tests.

These run the real entrypoints on a tiny synthetic dataset. They are the
substitute for training on every commit: fast enough for CI, and they fail if
the pipeline stops producing a loadable, self-consistent artifact.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from chimera.contracts import TargetSpec
from chimera.features import FeatureSpec, feature_columns
from nn.data_pipeline import build_dataset, save_dataset
from nn.registry import load_model, resolve_current
from nn.research_contract import SYNTHETIC_CONTRACT_ID
from nn.train import build_argparser, class_weights, main, resolve_device, set_seed
from tools.make_sample_data import generate_candles


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "ds.parquet"
    candles = generate_candles(rows=1000, seed=5)
    frame, metadata = build_dataset(
        candles,
        FeatureSpec(),
        TargetSpec(horizon=4),
        exchange="synthetic",
        pair="SYNTH/USDT",
        timeframe="1h",
    )
    save_dataset(path, frame, metadata)
    return path


TINY = [
    # The fixtures below are synthetic candles, so they run under the committed
    # synthetic research contract. The BTC one would (correctly) refuse them.
    "--research-contract",
    SYNTHETIC_CONTRACT_ID,
    "--epochs",
    "1",
    "--seq-len",
    "16",
    "--batch-size",
    "64",
    "--d-model",
    "16",
    "--n-heads",
    "2",
    "--num-layers",
    "1",
    "--device",
    "cpu",
]


# --- helpers ---------------------------------------------------------------
def test_seeding_makes_training_reproducible():
    set_seed(123)
    a = torch.randn(5)
    b = np.random.rand(5)
    set_seed(123)
    torch.testing.assert_close(a, torch.randn(5))
    np.testing.assert_allclose(b, np.random.rand(5))


def test_device_resolution_prefers_cpu_when_cuda_is_absent(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto").type == "cpu"
    assert resolve_device("cuda").type == "cpu", "must fall back, not crash"


def test_class_weights_counter_the_majority_class():
    """HOLD dominates cost-aware labels; without weighting the model learns to
    always predict it and matches the majority baseline exactly."""
    y = np.array([1] * 90 + [0] * 5 + [2] * 5)
    weights = class_weights(y, 3)
    assert weights[1] < weights[0]
    assert weights[1] < weights[2]


def test_argparser_defaults_to_no_tuning():
    """A plain training run must not require a 100-trial search."""
    args = build_argparser().parse_args(["--dataset", "x"])
    assert args.tune_trials == 0
    assert args.promote is False, "training must not auto-promote"
    assert args.mlflow is False, "MLflow must be opt-in"


# --- the real thing ----------------------------------------------------------
def test_training_produces_a_loadable_artifact(dataset, tmp_path):
    models_dir = tmp_path / "models"
    assert main(["--dataset", str(dataset), "--models-dir", str(models_dir), *TINY]) == 0

    versions = [p for p in models_dir.iterdir() if p.is_dir()]
    assert len(versions) == 1, "one training run must produce exactly one version"

    model, metadata = load_model(versions[0])
    assert metadata.sequence_length == 16
    assert metadata.feature_names == feature_columns()
    assert len(metadata.scaler_mean) == len(feature_columns())
    assert 0.0 < metadata.decision_threshold <= 1.0
    assert metadata.target_spec.horizon == 4
    assert metadata.pair == "SYNTH/USDT"

    # And it actually runs.
    with torch.no_grad():
        out = model(torch.zeros(1, 16, len(feature_columns())))
    assert out.shape == (1, 3)


def test_training_does_not_promote_by_default(dataset, tmp_path):
    models_dir = tmp_path / "models"
    main(["--dataset", str(dataset), "--models-dir", str(models_dir), *TINY])
    with pytest.raises(FileNotFoundError):
        resolve_current(models_dir)


def test_the_report_records_splits_baselines_and_the_gate(dataset, tmp_path):
    models_dir = tmp_path / "models"
    main(["--dataset", str(dataset), "--models-dir", str(models_dir), *TINY])
    version = next(p for p in models_dir.iterdir() if p.is_dir())
    report = json.loads((version / "report.json").read_text())

    assert set(report["split_plan"]) == {"train", "validation", "test"}
    for split in ("validation", "test"):
        assert set(report[split]) == {"majority_baseline", "momentum_baseline", "mtst"}
    assert "passed" in report["promotion"]
    assert report["threshold_selection"]["n_trades"] >= 0


def test_training_is_reproducible_for_a_fixed_seed(dataset, tmp_path):
    outputs = []
    for run in range(2):
        models_dir = tmp_path / f"models{run}"
        main(
            ["--dataset", str(dataset), "--models-dir", str(models_dir), "--seed", "7", *TINY]
        )
        version = next(p for p in models_dir.iterdir() if p.is_dir())
        model, metadata = load_model(version)
        with torch.no_grad():
            outputs.append(model(torch.ones(2, 16, len(feature_columns()))))
        outputs.append(metadata.decision_threshold)

    torch.testing.assert_close(outputs[0], outputs[2])
    assert outputs[1] == outputs[3]


def test_splits_recorded_in_the_report_do_not_overlap(dataset, tmp_path):
    models_dir = tmp_path / "models"
    main(["--dataset", str(dataset), "--models-dir", str(models_dir), *TINY])
    version = next(p for p in models_dir.iterdir() if p.is_dir())
    plan = json.loads((version / "report.json").read_text())["split_plan"]

    assert plan["train"]["end"] == plan["validation"]["start"]
    assert plan["validation"]["end"] == plan["test"]["start"]
    assert plan["train"]["start"] == 0


def test_training_fails_loudly_on_too_short_a_dataset(dataset, tmp_path):
    with pytest.raises(SystemExit, match="no samples"):
        main(
            [
                "--dataset",
                str(dataset),
                "--models-dir",
                str(tmp_path / "m"),
                "--research-contract",
                SYNTHETIC_CONTRACT_ID,
                "--seq-len",
                "5000",
                "--epochs",
                "1",
                "--device",
                "cpu",
            ]
        )


# --- CLI entrypoints ------------------------------------------------------------
@pytest.mark.parametrize(
    "module",
    [
        "tools.backfill",
        "tools.build_features",
        "tools.make_sample_data",
        "nn.train",
        "nn.walkforward",
        "nn.wf_diagnostics",
    ],
)
def test_cli_help_works(module):
    """Every documented entrypoint must at least be importable and describe itself."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", module, "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


def test_backfill_requires_its_arguments():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "tools.backfill"], capture_output=True, text=True
    )
    assert result.returncode == 2
    assert "required" in result.stderr.lower()


def test_build_features_roundtrips_through_the_cli(tmp_path):
    from tools.build_features import main as build_main

    candles_path = tmp_path / "candles.parquet"
    generate_candles(rows=800, seed=3).to_parquet(candles_path, index=False)

    out = tmp_path / "ds.parquet"
    assert (
        build_main(
            [
                "--candles",
                str(candles_path),
                "--out",
                str(out),
                "--exchange",
                "synthetic",
                "--pair",
                "S/USDT",
                "--timeframe",
                "1h",
                "--horizon",
                "3",
            ]
        )
        == 0
    )

    from nn.data_pipeline import load_dataset

    frame, metadata = load_dataset(out)
    assert len(frame) > 0
    assert metadata.target_spec["horizon"] == 3
    assert metadata.pair == "S/USDT"
