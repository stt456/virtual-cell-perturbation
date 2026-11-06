"""Small multi-output models, train-only scaling, and explicit mean fallback."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import random
import time
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
import torch
from torch import nn

from src.c_common import load_npz, read_json, require, write_json


def macro_mse(prediction, truth):
    return float(np.mean((np.asarray(prediction, dtype=np.float64) - truth) ** 2))


def standardizer(x):
    require(len(x) > 0 and np.isfinite(x).all(), "No finite training features")
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return mean, scale


def network(inputs, hidden, outputs, dropout):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, outputs))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def predict(model, x, available, device="cpu"):
    available = np.asarray(available, dtype=bool)
    require(available.shape == (len(x),), "Feature availability shape")
    mu = model["train_mean"]
    if model["family"] == "zero":
        return np.zeros((len(x), len(mu)), dtype=np.float64)
    result = np.broadcast_to(mu, (len(x), len(mu))).copy()
    if model["family"] == "train_mean" or not available.any():
        return result
    scaled = (np.asarray(x[available], dtype=np.float64) - model["scale_mean"]) / model["scale_std"]
    if model["family"] == "ridge":
        result[available] = scaled @ model["coef"].T + model["intercept"]
    else:
        net = network(scaled.shape[1], model["hidden_dim"], len(mu), model["dropout"]).to(device)
        net.load_state_dict(model["state_dict"])
        net.eval()
        with torch.no_grad():
            residual = net(torch.as_tensor(scaled, dtype=torch.float32, device=device)).cpu().numpy()
        result[available] = residual.astype(np.float64) + mu
    require(np.isfinite(result).all(), "Nonfinite prediction")
    return result


def fit_ridge(x_train, y_train, available, alpha):
    require(available.any(), "No mapped training targets")
    mean, scale = standardizer(x_train[available])
    estimator = Ridge(alpha=alpha, fit_intercept=True, solver="svd")
    estimator.fit((x_train[available] - mean) / scale, y_train[available])
    return {"family": "ridge", "alpha": float(alpha), "scale_mean": mean, "scale_std": scale,
            "train_mean": y_train.mean(axis=0), "coef": estimator.coef_, "intercept": estimator.intercept_}


def fit_mlp(x_train, y_train, available, x_val, y_val, val_available, settings, seed, device):
    require(available.any() and val_available.any(), "No available train/validation targets")
    set_seed(seed)
    mean, scale = standardizer(x_train[available])
    mu = y_train.mean(axis=0)
    xt = torch.as_tensor((x_train[available] - mean) / scale, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y_train[available] - mu, dtype=torch.float32, device=device)
    xv = torch.as_tensor((x_val[val_available] - mean) / scale, dtype=torch.float32, device=device)
    net = network(xt.shape[1], settings["hidden_dim"], yt.shape[1], settings["dropout"]).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    generator = torch.Generator(device=device).manual_seed(seed)
    best, best_state, best_epoch, stale, history = float("inf"), None, 0, 0, []
    base = np.broadcast_to(mu, y_val.shape).copy()
    for epoch in range(1, settings["max_epochs"] + 1):
        net.train()
        order = torch.randperm(len(xt), generator=generator, device=device)
        loss_sum = torch.zeros((), device=device)
        for start in range(0, len(order), settings["batch_size"]):
            batch = order[start:start + settings["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((net(xt[batch]) - yt[batch]) ** 2)
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach() * len(batch)
        net.eval()
        with torch.no_grad():
            pred = base.copy()
            pred[val_available] = net(xv).cpu().numpy().astype(np.float64) + mu
        validation = macro_mse(pred, y_val)
        train_loss = float((loss_sum / len(xt)).item())
        require(np.isfinite(validation) and np.isfinite(train_loss), "Nonfinite MLP optimization")
        history.append({"epoch": epoch, "train_available_mse": train_loss, "val_all_mse": validation,
                        "val_available_mse": macro_mse(pred[val_available], y_val[val_available])})
        if validation < best - 1e-12:
            best, best_epoch, stale = validation, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}
        else:
            stale += 1
        if stale >= settings["patience"]:
            break
    require(best_state is not None, "No finite MLP checkpoint")
    model = {"family": "mlp", "hidden_dim": settings["hidden_dim"], "dropout": settings["dropout"],
             "scale_mean": mean, "scale_std": scale, "train_mean": mu, "state_dict": best_state,
             "best_epoch": best_epoch, "learning_rate": settings["learning_rate"]}
    return model, history


def search(family, x_train, y_train, available, x_val, y_val, val_available, config, seed, device):
    if family == "ridge":
        candidates = [{"alpha": alpha} for alpha in config["ridge"]["alphas"]]
    else:
        candidates = [{**config["mlp"], "hidden_dim": hidden, "learning_rate": lr}
                      for hidden in config["mlp"]["hidden_dims"] for lr in config["mlp"]["learning_rates"]]
    best_model, best_loss, best_key, records = None, float("inf"), None, []
    for settings in candidates:
        started = time.monotonic()
        if family == "ridge":
            model = fit_ridge(x_train, y_train, available, settings["alpha"])
            history, key = [], (-settings["alpha"],)
        else:
            model, history = fit_mlp(x_train, y_train, available, x_val, y_val, val_available, settings, seed, device)
            key = (settings["hidden_dim"], settings["learning_rate"])
        prediction = predict(model, x_val, val_available, device=device if family == "mlp" else "cpu")
        loss = macro_mse(prediction, y_val)
        record = {"settings": settings, "val_all_mse": loss,
                  "val_available_mse": macro_mse(prediction[val_available], y_val[val_available]),
                  "best_epoch": model.get("best_epoch"), "epochs_run": len(history), "history": history,
                  "elapsed_seconds": time.monotonic() - started}
        records.append(record)
        print(f"  {family} {settings.get('alpha', (settings.get('hidden_dim'), settings.get('learning_rate')))} val_mse={loss:.8g}, {record['elapsed_seconds']:.2f}s", flush=True)
        tolerance = config["selection"]["tie_atol"]
        if loss < best_loss - tolerance or (abs(loss - best_loss) <= tolerance and (best_key is None or key < best_key)):
            best_model, best_loss, best_key = model, loss, key
    return best_model, records


def save_model(folder, model):
    folder = Path(folder)
    arrays = {key: value for key, value in model.items() if isinstance(value, np.ndarray)}
    meta = {key: value for key, value in model.items() if key not in arrays and key != "state_dict"}
    np.savez_compressed(folder / "model_arrays.npz", **arrays)
    if "state_dict" in model:
        torch.save(model["state_dict"], folder / "model_weights.pt")
    write_json(folder / "model.json", meta)


def load_model(folder):
    folder = Path(folder)
    result = {**read_json(folder / "model.json"), **load_npz(folder / "model_arrays.npz")}
    if result["family"] == "mlp":
        result["state_dict"] = torch.load(folder / "model_weights.pt", map_location="cpu", weights_only=True)
    return result
