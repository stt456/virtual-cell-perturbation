"""C diagnostic implementation of v0.1; independent D review is still required."""
import numpy as np
from scipy.stats import rankdata
from src.c_common import require, unique_strings

METRICS = ("mse", "pearson", "spearman", "top20_overlap", "top20_direction", "residual_pearson", "residual_spearman", "mse_without_self")


def correlations(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    a, b = a - a.mean(axis=1, keepdims=True), b - b.mean(axis=1, keepdims=True)
    denominator = np.sqrt(np.sum(a*a, axis=1) * np.sum(b*b, axis=1))
    out = np.full(len(a), np.nan)
    good = denominator > 1e-14
    out[good] = np.clip(np.sum(a[good]*b[good], axis=1) / denominator[good], -1, 1)
    return out


def evaluate(prediction, truth, train_mean, target_ids, gene_ids, available, fallback, model, seed, split, top_k=20):
    prediction, truth = np.asarray(prediction, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    genes, targets = np.asarray(gene_ids, dtype=str), np.asarray(target_ids, dtype=str)
    unique_strings(genes, "evaluation genes")
    unique_strings(targets, "evaluation targets")
    require(prediction.shape == truth.shape == (len(targets), len(genes)), "Evaluation shape mismatch")
    require(np.isfinite(prediction).all() and np.isfinite(truth).all(), "Nonfinite evaluation input")
    require(np.asarray(available).shape == np.asarray(fallback).shape == (len(targets),), "Evaluation flags")
    require(np.asarray(train_mean).shape == (len(genes),), "Evaluation train mean shape")
    residual_pred, residual_true = prediction - train_mean, truth - train_mean
    values = {"mse": np.mean((prediction-truth)**2, axis=1),
              "pearson": correlations(prediction, truth),
              "spearman": correlations(rankdata(prediction, axis=1), rankdata(truth, axis=1)),
              "residual_pearson": correlations(residual_pred, residual_true),
              "residual_spearman": correlations(rankdata(residual_pred, axis=1), rankdata(residual_true, axis=1))}
    k = min(top_k, len(genes))
    require(k > 0, "Empty gene output")
    rows = []
    gene_map = {gene: i for i, gene in enumerate(genes)}
    for i, target in enumerate(targets):
        ptop = np.lexsort((genes, -np.abs(prediction[i])))[:k]
        ttop = np.lexsort((genes, -np.abs(truth[i])))[:k]
        intersection = np.intersect1d(ptop, ttop)
        direction = float(np.mean(np.sign(prediction[i, intersection]) == np.sign(truth[i, intersection]))) if len(intersection) else np.nan
        if model == "zero":
            direction = np.nan
        coordinates = np.ones(len(genes), dtype=bool)
        if target in gene_map:
            coordinates[gene_map[target]] = False
        self_excluded = float(np.mean((prediction[i, coordinates]-truth[i, coordinates])**2)) if coordinates.any() else np.nan
        rows.append({"model": model, "seed": seed, "split": split, "target_gene": target,
                     "feature_available": bool(available[i]), "used_fallback": bool(fallback[i]),
                     "data_version": "GEARS_550fde89_B_v0.1", "zero_topk_is_no_signal_tie": model == "zero",
                     "self_coordinate_present": target in gene_map,
                     **{key: float(value[i]) for key, value in values.items()},
                     "top20_overlap": len(intersection)/k, "top20_direction": direction,
                     "mse_without_self": self_excluded})
    return rows


def summarize(rows):
    out = []
    for subset in ("all", "available", "unavailable"):
        selected = [row for row in rows if subset == "all" or row["feature_available"] == (subset == "available")]
        summary = {key: rows[0][key] for key in ("model", "seed", "split", "data_version")}
        summary.update(subset=subset, n_targets=len(selected), n_fallback=sum(row["used_fallback"] for row in selected))
        for metric in METRICS:
            values = np.asarray([row[metric] for row in selected], dtype=float)
            valid = values[np.isfinite(values)]
            summary[metric] = float(valid.mean()) if len(valid) else np.nan
            summary[metric + "_n_valid"] = len(valid)
        out.append(summary)
    return out


def check_prediction_package(data, targets, genes):
    require(data["target_ids"].tolist() == list(targets), "Prediction target order mismatch")
    require(data["gene_ids"].tolist() == list(genes), "Prediction gene order mismatch")
    require(data["delta_pred"].shape == (len(targets), len(genes)) and np.isfinite(data["delta_pred"]).all(), "Invalid predictions")
    for key in ("feature_available", "used_fallback"):
        require(data[key].dtype == np.bool_ and data[key].shape == (len(targets),), f"Invalid {key}")
