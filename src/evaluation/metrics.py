"""Population mean response metrics; each perturbation target has equal weight."""
import numpy as np

from .io import ids, require

METRICS = ("mse", "pearson", "spearman", "top20_overlap", "top20_direction",
           "residual_pearson", "residual_spearman", "mse_without_self")


def average_ranks(vector):
    """One-based ranks, average ties; independent of SciPy/C's rankdata."""
    vector = np.asarray(vector, dtype=np.float64)
    require(vector.ndim == 1 and np.isfinite(vector).all(), "rank input must be finite vector")
    order = np.argsort(vector, kind="stable")
    sorted_values = vector[order]
    starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
    ends = np.r_[starts[1:], len(order)]
    ranks = np.empty(len(order), dtype=float)
    for start, end in zip(starts, ends):
        ranks[order[start:end]] = (start + 1 + end) / 2
    return ranks


def pearson(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    require(a.shape == b.shape and a.ndim == 1 and len(a) > 0, "correlation shape mismatch")
    require(np.isfinite(a).all() and np.isfinite(b).all(), "correlation input is nonfinite")
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan
    ac, bc = a - a.mean(), b - b.mean()
    # Scale separately: no arbitrary product-norm threshold or overflow/underflow.
    ac, bc = ac / np.max(np.abs(ac)), bc / np.max(np.abs(bc))
    return float(np.clip(np.dot(ac, bc) / np.sqrt(np.dot(ac, ac) * np.dot(bc, bc)), -1, 1))


def top_indices(vector, genes, k=20):
    require(isinstance(k, int) and k > 0, "top_k must be positive integer")
    return np.lexsort((np.asarray(genes), -np.abs(vector)))[:min(k, len(vector))]


def baselines(training_delta, n_test):
    values = np.asarray(training_delta, dtype=np.float64)
    require(values.ndim == 2 and values.shape[0] > 0 and values.shape[1] > 0,
            "training targets required for mean baseline")
    require(np.isfinite(values).all() and n_test > 0, "invalid baseline input")
    mean = values.mean(axis=0)  # Equal perturbation weights, never n_cells weights.
    return mean, np.zeros((n_test, values.shape[1])), np.tile(mean, (n_test, 1))


def evaluate(prediction, truth, train_mean, targets, genes, available, fallback,
             model, seed, split="test", top_k=20):
    require(truth is not None, "ground truth missing; missing labels cannot be scored as zeros")
    targets, genes = ids(np.asarray(targets), "metric targets"), ids(np.asarray(genes), "metric genes")
    prediction, truth, train_mean = (np.asarray(x, dtype=np.float64)
                                     for x in (prediction, truth, train_mean))
    require(prediction.shape == truth.shape == (len(targets), len(genes)), "metric matrix shape mismatch")
    require(train_mean.shape == (len(genes),), "metric mean shape mismatch")
    require(all(np.isfinite(x).all() for x in (prediction, truth, train_mean)), "nonfinite metric input")
    require(np.asarray(available).shape == np.asarray(fallback).shape == (len(targets),), "metric flags mismatch")
    gene_map, rows = {g: i for i, g in enumerate(genes)}, []
    for i, target in enumerate(targets):
        p, t = prediction[i], truth[i]
        rp, rt = p - train_mean, t - train_mean
        intersection = np.intersect1d(top_indices(p, genes, top_k), top_indices(t, genes, top_k))
        zero_prediction = not np.any(p)
        direction = (float(np.mean(np.sign(p[intersection]) == np.sign(t[intersection])))
                     if len(intersection) and not zero_prediction else np.nan)
        keep = np.arange(len(genes)) != gene_map.get(target, -1)
        row = dict(model=model, seed=seed, split=split, target_gene=str(target),
                   feature_available=bool(available[i]), used_fallback=bool(fallback[i]),
                   zero_topk_is_no_signal_tie=zero_prediction,
                   self_coordinate_present=target in gene_map,
                   mse=float(np.mean((p-t)**2)), pearson=pearson(p, t),
                   spearman=pearson(average_ranks(p), average_ranks(t)),
                   residual_pearson=pearson(rp, rt),
                   residual_spearman=pearson(average_ranks(rp), average_ranks(rt)),
                   top20_overlap=len(intersection) / min(top_k, len(genes)),
                   top20_direction=direction,
                   mse_without_self=float(np.mean((p[keep]-t[keep])**2)) if keep.any() else np.nan,
                   truth_rms=float(np.sqrt(np.mean(t*t))),
                   residual_truth_rms=float(np.sqrt(np.mean(rt*rt))))
        rows.append(row)
    return rows
