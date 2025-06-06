"""
Train script для MinimalTST.

USAGE:
    python train.py --features data/features/binance_BTC_USDT_1h_2020-01-01_features.parquet --epochs 20

• Mixed precision: включено через torch.cuda.amp.autocast.
• Ray Tune: 100 trials по lr∈[1e-5,1e-3].
• Loss: Huber(pred-truth) + 0.2 * policy_gradient_reward.
• Финал: torch.jit.trace → nn/model_ts.pt, torch.onnx.export → nn/model_ts.onnx,
  mlflow.register_model(..., name="nn_predictor");
  MlflowClient().set_registered_model_alias(..., alias="prod").
• Лучший конфиг выбирается через analysis.get_best_config(metric="loss", mode="min").
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import mlflow
from mlflow.tracking import MlflowClient
from torch.cuda.amp import autocast, GradScaler
import mlflow
import ray
from ray import tune
import argparse
from model_def import MTST


def load_data(path: str):
    df = pd.read_parquet(path)
    X = []
    y = []
    window = 100
    for i in range(window, len(df) - 1):
        X.append(
            df.iloc[i - window : i][
                ["close", "return_1", "ema_9", "ema_21", "volume"]
            ].values
        )
        y.append(df.iloc[i + 1]["close"])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def policy_gradient_reward(pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    return (
        (torch.sign(pred[1:] - pred[:-1]) == torch.sign(truth[1:] - truth[:-1]))
        .float()
        .mean()
    )


def live_sharpe(pred: torch.Tensor, truth: torch.Tensor) -> float:
    pnl = torch.sign(pred[1:] - pred[:-1]) * (truth[1:] - truth[:-1])
    if pnl.numel() < 2 or pnl.std() == 0:
        return 0.0
    return (pnl.mean() / pnl.std() * np.sqrt(pnl.numel())).item()


def train_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    for xb, yb in loader:
        optimizer.zero_grad()
        with autocast():
            out = model(xb)
            loss = criterion(out, yb) + (1 - policy_gradient_reward(out, yb))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


def evaluate(model, loader):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for xb, yb in loader:
            with autocast():
                out = model(xb)
            preds.append(out.detach())
            trues.append(yb.detach())
    preds = torch.cat(preds)
    trues = torch.cat(trues)
    return live_sharpe(preds, trues)


def train_tst(config, features, device, epochs=10):
    X, y = load_data(features)
    split = int(len(X) * 0.8)
    X_train, X_val = torch.tensor(X[:split]).to(device), torch.tensor(X[split:]).to(
        device
    )
    y_train, y_val = torch.tensor(y[:split]).to(device), torch.tensor(y[split:]).to(
        device
    )
    train_ds = torch.utils.data.TensorDataset(X_train, y_train)
    val_ds = torch.utils.data.TensorDataset(X_val, y_val)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=32)
    model = MTST(input_dim=X.shape[-1], seq_len=X.shape[1]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"])
    criterion = nn.HuberLoss()
    scaler = GradScaler()
    for _ in range(epochs):
        train_epoch(model, train_loader, optimizer, criterion, scaler, device)
    sharpe = evaluate(model, val_loader)
    tune.report(sharpe=sharpe)
    return model


def tune_trainable(config, features, device, epochs=10):
    return train_tst(config, features, device, epochs)


def save_and_register(model, example_input, output_dir="nn"):
    model.eval()
    traced = torch.jit.trace(model, example_input)
    os.makedirs(output_dir, exist_ok=True)
    traced.save(os.path.join(output_dir, "model_ts.pt"))
    torch.onnx.export(
        model,
        example_input,
        os.path.join(output_dir, "model_ts.onnx"),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("nn_predictor")
    with mlflow.start_run():
        mlflow.pytorch.log_model(model, "nn_predictor")
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/nn_predictor"
        reg_model = mlflow.register_model(model_uri, name="nn_predictor")
        client = mlflow.tracking.MlflowClient()
        client.set_registered_model_alias("nn_predictor", reg_model.version, "prod")

        model_version = mlflow.register_model(model_uri, name="nn_predictor")
        MlflowClient().set_registered_model_alias(
            name="nn_predictor", alias="prod", version=model_version.version
        )


# main # Removed this line


if __name__ == "__main__":
    import logging  # Added import

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )  # Added logging config
    logger = logging.getLogger(__name__)  # Added logger instance

    logger.info("Starting NN training script.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    logger.info("Initializing Ray...")
    ray.init(ignore_reinit_error=True, include_dashboard=False)

    search_space = {"lr": tune.loguniform(1e-5, 1e-3)}

    logger.info("Starting Ray Tune hyperparameter tuning...")
    analysis = tune.run(
        tune.with_parameters(
            tune_trainable, features=args.features, device=device, epochs=args.epochs
        ),
        resources_per_trial={"cpu": 2, "gpu": int(device == "cuda")},
        config=search_space,
        num_samples=100,  # Consider reducing for faster testing if needed
        local_dir="./ray_results",
        metric="sharpe",
        mode="max",
    )
    logger.info("Ray Tune hyperparameter tuning completed.")

    best_config = analysis.get_best_config(
        metric="sharpe", mode="max"
    )  # Corrected, was "sharpe" then "loss"
    logger.info(
        f"Best config found by Ray Tune (metric='sharpe', mode='max'): {best_config}"
    )

    logger.info("Training final model with best config...")
    model = train_tst(best_config, args.features, device, args.epochs)
    logger.info("Final model training completed.")

    X, _ = load_data(args.features)
    example_input = torch.tensor(X[:2]).to(device)  # Using first 2 samples as example

    logger.info("Saving and registering model with MLflow...")
    save_and_register(model, example_input)
    logger.info("Model saved and registered successfully.")
    logger.info("NN training script finished.")
