"""End-to-end smoke test, no network and no money.

    python -m tools.smoke

Walks the whole chain in one process::

    synthetic candles -> validation -> features + labels -> tiny training
    -> model artifact -> reload from disk -> inference service -> prediction
    -> strategy interpretation -> risk decision

Everything is written to a temporary directory and removed afterwards. It runs
on CPU in well under a minute, which is what makes it usable both as a
developer sanity check and as the CI substitute for real training.

Exit code 0 means every stage produced the shape the next stage expects. It
says nothing about whether the model is any good — with synthetic data it
cannot.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger("smoke")


def _step(number: int, description: str) -> None:
    logger.info("[%d/8] %s", number, description)


def run_smoke(workdir: Path, rows: int = 1200, epochs: int = 1, seq_len: int = 24) -> dict:
    from fastapi.testclient import TestClient

    from chimera.contracts import Signal, TargetSpec, decide
    from chimera.features import FeatureSpec, compute_features, feature_columns
    from chimera.risk import RiskEngine, RiskLimits
    from nn.data_pipeline import build_dataset, save_dataset, validate_ohlcv
    from nn.registry import load_model
    from nn.train import main as train_main
    from tools.make_sample_data import generate_candles

    results: dict = {}

    _step(1, "generate synthetic candles")
    candles = generate_candles(rows, seed=11)
    candles, report = validate_ohlcv(candles, "1h")
    results["candles"] = {"rows": len(candles), "validation": report.to_dict()}

    _step(2, "build features and cost-aware labels")
    target_spec = TargetSpec(horizon=4)
    frame, metadata = build_dataset(
        candles,
        FeatureSpec(),
        target_spec,
        exchange="synthetic",
        pair="SYNTH/USDT",
        timeframe="1h",
        validation=report.to_dict(),
    )
    dataset_path = workdir / "dataset.parquet"
    save_dataset(dataset_path, frame, metadata)
    results["dataset"] = {
        "rows": len(frame),
        "class_balance": metadata.class_balance,
        "features": len(feature_columns()),
    }

    _step(3, "train a tiny model")
    models_dir = workdir / "models"
    exit_code = train_main(
        [
            "--dataset",
            str(dataset_path),
            "--models-dir",
            str(models_dir),
            "--epochs",
            str(epochs),
            "--seq-len",
            str(seq_len),
            "--batch-size",
            "64",
            "--d-model",
            "32",
            "--n-heads",
            "2",
            "--num-layers",
            "1",
            "--device",
            "cpu",
        ]
    )
    if exit_code != 0:
        raise RuntimeError(f"training exited with {exit_code}")

    versions = sorted(p for p in models_dir.iterdir() if p.is_dir())
    if not versions:
        raise RuntimeError("training produced no model artifact")
    model_dir = versions[-1]
    results["training"] = {"version": model_dir.name}

    _step(4, "reload the artifact from disk")
    model, model_metadata = load_model(model_dir)
    results["reload"] = {
        "version": model_metadata.model_version,
        "sequence_length": model_metadata.sequence_length,
        "n_features": model_metadata.n_features,
        "threshold": model_metadata.decision_threshold,
    }
    if model_metadata.feature_names != feature_columns():
        raise RuntimeError("reloaded metadata disagrees with the feature contract")

    _step(5, "serve it and check readiness")
    # promote() is normally gated; the smoke path promotes directly because it
    # is exercising plumbing, not judging a model.
    from nn.registry import promote

    promote(models_dir, model_dir.name)

    import os

    os.environ["CHIMERA_MODELS_DIR"] = str(models_dir)
    from nn.infer_service import app

    with TestClient(app) as client:
        assert client.get("/livez").status_code == 200, "livez failed"
        ready = client.get("/readyz")
        if ready.status_code != 200:
            raise RuntimeError(f"readyz failed: {ready.text}")
        info = ready.json()
        results["readyz"] = info

        _step(6, "predict from real computed features")
        features = compute_features(candles, FeatureSpec()).dropna()
        window = features.iloc[-info["sequence_length"] :]
        response = client.post(
            "/predict",
            json={
                "pair": "SYNTH/USDT",
                "timeframe": "1h",
                "timestamp": candles["date"].iloc[-1].isoformat(),
                "features": window.to_numpy().tolist(),
            },
        )
        if response.status_code != 200:
            raise RuntimeError(f"predict failed: {response.status_code} {response.text}")
        prediction = response.json()
        results["prediction"] = prediction

        _step(7, "confirm bad input is rejected, not answered")
        bad = client.post(
            "/predict",
            json={
                "pair": "SYNTH/USDT",
                "timeframe": "1h",
                "timestamp": candles["date"].iloc[-1].isoformat(),
                "features": [[0.0] * info["n_features"]],
            },
        )
        if bad.status_code != 400:
            raise RuntimeError(f"expected 400 for a short window, got {bad.status_code}")
        results["shape_rejection"] = bad.status_code

    _step(8, "strategy interpretation and risk decision")
    signal = decide(prediction["probabilities"], prediction["decision_threshold"])
    if signal.value != prediction["signal"]:
        raise RuntimeError(
            f"client-side decide() says {signal.value} but the service said "
            f"{prediction['signal']}: the decision rule has diverged"
        )

    engine = RiskEngine(RiskLimits(), state_path=workdir / "risk_state.json")
    engine.update_equity(10_000.0)
    entry_price = float(candles["close"].iloc[-1])
    decision = engine.evaluate_entry(
        pair="SYNTH/USDT",
        equity=10_000.0,
        entry_price=entry_price,
        stop_price=entry_price * 0.95,
        data_age_s=0.0,
        inference_age_s=0.0,
    )
    results["risk"] = {
        "signal": signal.value,
        "allowed": decision.allowed,
        "reason": decision.reason,
        "stake": round(decision.stake, 2),
    }

    # 1% of 10k risked over a 5% stop is a 2000 notional, capped at 25% of
    # equity by max_position_pct. Assert the arithmetic, not just the plumbing.
    if signal is not Signal.HOLD and decision.allowed:
        expected = min(10_000.0 * 0.01 / 0.05, 10_000.0 * 0.25)
        if abs(decision.stake - expected) > 1e-6:
            raise RuntimeError(f"sizing wrong: expected {expected}, got {decision.stake}")

    # A halted engine must refuse regardless of the signal.
    engine.halt("smoke test")
    halted = engine.evaluate_entry(
        pair="SYNTH/USDT",
        equity=10_000.0,
        entry_price=entry_price,
        stop_price=entry_price * 0.95,
    )
    if halted.allowed:
        raise RuntimeError("halted risk engine approved an entry")
    results["kill_switch"] = {"allowed": halted.allowed, "reason": halted.reason}

    return results


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description="End-to-end smoke test.")
    parser.add_argument("--rows", type=int, default=1200)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--keep", action="store_true", help="Keep the temp directory.")
    args = parser.parse_args(argv)

    workdir = Path(tempfile.mkdtemp(prefix="chimera-smoke-"))
    try:
        results = run_smoke(workdir, rows=args.rows, epochs=args.epochs, seq_len=args.seq_len)
    except Exception as exc:  # noqa: BLE001 - the point is to report the failure
        logger.error("SMOKE FAILED: %s", exc, exc_info=True)
        return 1
    finally:
        if args.keep:
            logger.info("Kept %s", workdir)
        else:
            shutil.rmtree(workdir, ignore_errors=True)

    print("\n" + json.dumps(results, indent=2, default=str))
    print("\nSMOKE OK: every stage produced what the next stage expects.")
    print("This proves the plumbing, not profitability.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
