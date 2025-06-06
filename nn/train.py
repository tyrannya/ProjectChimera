"""
Train script для MinimalTST.

USAGE:
    python train.py --features data/features/binance_BTC_USDT_1h_2020-01-01_features.parquet --epochs 20

• Mixed precision: включено через torch.cuda.amp.autocast.
• Ray Tune: 100 trials по lr∈[1e-5,1e-3].
• Loss: Huber(pred-truth) + 0.2 * policy_gradient_reward.
• Финал: torch.jit.trace → nn/model_ts.pt, torch.onnx.export → nn/model_ts.onnx, mlflow.register_model(..., name="nn_predictor", alias="prod").
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import mlflow
from torch.cuda.amp import autocast, GradScaler
from model_def import MinimalTST
import argparse
import ray
from ray import tune

def load_data(path):
    df = pd.read_parquet(path)
    X = []
    y = []
    window = 100
    for i in range(window, len(df) - 1):
        X.append(df.iloc[i - window:i][['close', 'return_1', 'ema_9', 'ema_21', 'volume']].values)
        y.append(df.iloc[i + 1]['close'])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return X, y

def policy_gradient_reward(pred, truth):
    # Toy reward: positive if predicted return sign matches actual, else 0
    return ((torch.sign(pred[1:] - pred[:-1]) == torch.sign(truth[1:] - truth[:-1])).float().mean())

def train_tst(config, features, device, epochs=10):
    X, y = load_data(features)
    X = torch.tensor(X).to(device)
    y = torch.tensor(y).to(device)
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    model = MinimalTST(input_dim=X.shape[-1], seq_len=X.shape[1]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"])
    criterion = nn.HuberLoss()
    scaler = GradScaler()
    for epoch in range(epochs):
        model.train()
        epoch_loss = []
        for xb, yb in loader:
            optimizer.zero_grad()
            with autocast():
                out = model(xb)
                loss1 = criterion(out, yb)
                reward = policy_gradient_reward(out, yb)
                loss = loss1 + 0.2 * (1 - reward)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss.append(loss.item())
        tune.report(loss=np.mean(epoch_loss))
    return model

def tune_trainable(config, features, device, epochs=10):
    model = train_tst(config, features, device, epochs)
    return model

def save_and_register(model, example_input, output_dir="nn"):
    model.eval()
    traced = torch.jit.trace(model, example_input)
    traced.save(os.path.join(output_dir, "model_ts.pt"))
    torch.onnx.export(model, example_input, os.path.join(output_dir, "model_ts.onnx"),
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input':{0:'batch'}, 'output':{0:'batch'}}, opset_version=17)
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("nn_predictor")
    with mlflow.start_run():
        mlflow.pytorch.log_model(model, "nn_predictor")
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/nn_predictor"
        reg_model = mlflow.register_model(model_uri, name="nn_predictor")
        client = mlflow.tracking.MlflowClient()
        client.set_registered_model_alias("nn_predictor", reg_model.version, "prod")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ray.init(ignore_reinit_error=True, include_dashboard=False)
    search_space = {
        "lr": tune.loguniform(1e-5, 1e-3)
    }
    analysis = tune.run(
        tune.with_parameters(tune_trainable, features=args.features, device=device, epochs=args.epochs),
        resources_per_trial={"cpu": 2, "gpu": int(device=="cuda")},
        config=search_space,
        num_samples=100,
        local_dir="./ray_results",
        metric="loss",
        mode="min"
    )
    best_config = analysis.get_best_config(metric="loss")
    model = train_tst(best_config, args.features, device, args.epochs)
    os.makedirs("nn", exist_ok=True)
    # Use an example input for tracing and export
    X, _ = load_data(args.features)
    example_input = torch.tensor(X[:2]).to(device)
    save_and_register(model, example_input)
