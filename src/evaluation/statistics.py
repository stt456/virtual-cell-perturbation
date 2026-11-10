import numpy as np
import pandas as pd

from .io import require
from .metrics import METRICS


def summarize(frame, stratified=False):
    rows = []
    for (seed, split, model), group in frame.groupby(["seed", "split", "model"], sort=True):
        subsets = {"all": group, "available": group[group.feature_available],
                   "unavailable": group[~group.feature_available]}
        if stratified and "strength_stratum" in frame:
            subsets.update({f"strength_{s}": group[group.strength_stratum == s]
                            for s in ("weak", "medium", "strong")})
        for subset, selected in subsets.items():
            if selected.empty:
                continue
            row = dict(seed=int(seed), split=split, model=model, subset=subset,
                       n_targets=len(selected), n_fallback=int(selected.used_fallback.sum()))
            for metric in METRICS:
                valid = selected[metric].dropna()
                row[metric] = valid.mean() if len(valid) else np.nan
                row[f"{metric}_n_valid"] = len(valid)
            rows.append(row)
    return pd.DataFrame(rows)


def across_seeds(summary):
    rows = []
    for (model, split, subset), group in summary.groupby(["model", "split", "subset"], sort=True):
        require(not group.seed.duplicated().any(), "duplicate seed in summary")
        row = dict(model=model, split=split, subset=subset, n_seeds=len(group),
                   mean_targets=group.n_targets.mean())
        for metric in METRICS:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)
            row[f"{metric}_n_valid_seeds"] = int(group[metric].notna().sum())
        rows.append(row)
    return pd.DataFrame(rows)


def paired_bootstrap(reference, current, repeats=10000, random_seed=1729):
    """Positive improvement = baseline MSE minus model MSE. Within one split only."""
    require(repeats >= 100, "bootstrap repeats must be at least 100")
    for frame in (reference, current):
        require(not frame.target_gene.duplicated().any() and len(frame) > 0, "bootstrap duplicate/empty target")
        require(len(frame[["seed", "split"]].drop_duplicates()) == 1, "do not pool overlapping seeds/splits")
    require(reference[["seed", "split"]].iloc[0].tolist() == current[["seed", "split"]].iloc[0].tolist(),
            "bootstrap split mismatch")
    require(set(reference.target_gene) == set(current.target_gene), "bootstrap target mismatch")
    r = reference.set_index("target_gene").sort_index()
    c = current.set_index("target_gene").loc[r.index]
    diff = r.mse.to_numpy(float) - c.mse.to_numpy(float)
    require(np.isfinite(diff).all(), "bootstrap nonfinite error")
    rng, distribution = np.random.default_rng(random_seed), []
    for start in range(0, repeats, 500):
        draws = rng.integers(len(diff), size=(min(500, repeats-start), len(diff)))
        distribution.extend(diff[draws].mean(axis=1))
    lo, hi = np.quantile(distribution, [0.025, 0.975])
    base_mse = float(r.mse.mean())
    return dict(n_targets=len(diff), mean_mse_improvement_over_train_mean=float(diff.mean()),
                relative_improvement_pct=100*diff.mean()/base_mse if base_mse else np.nan,
                ci95_low=float(lo), ci95_high=float(hi), bootstrap_repeats=repeats,
                bootstrap_random_seed=random_seed, target_win_fraction=float(np.mean(diff > 0)))


def bootstrap_table(frame, repeats=10000):
    rows = []
    for (seed, split), group in frame.groupby(["seed", "split"], sort=True):
        reference = group[group.model == "train_mean"]
        for offset, (subset, mask) in enumerate((("all", np.ones(len(reference), bool)),
                                               ("available", reference.feature_available.to_numpy()),
                                               ("unavailable", ~reference.feature_available.to_numpy()))):
            ref = reference.loc[mask]
            if ref.empty:
                continue
            for model, current in group.groupby("model", sort=True):
                if model == "train_mean":
                    continue
                selected = current[current.target_gene.isin(ref.target_gene)]
                values = paired_bootstrap(ref, selected, repeats, 1729+int(seed)+10000*(split=="test")+offset*100)
                rows.append(dict(seed=int(seed), split=split, subset=subset, model=model, **values))
    return pd.DataFrame(rows)


def add_strength_strata(frame):
    """Post-hoc, seed-specific tertiles; thresholds shared by all models in that seed."""
    pieces, thresholds = [], []
    for (seed, split), group in frame.groupby(["seed", "split"], sort=True):
        zero = group[group.model == "zero"].set_index("target_gene")
        strength = (zero.truth_rms if "truth_rms" in zero else np.sqrt(zero.mse))
        require(np.isfinite(strength).all(), "invalid response strength")
        q1, q2 = np.quantile(strength, [1/3, 2/3])
        labels = pd.Series(np.where(strength <= q1, "weak", np.where(strength <= q2, "medium", "strong")),
                           index=strength.index)
        current = group.copy()
        current["truth_rms"] = current.target_gene.map(strength)
        current["strength_stratum"] = current.target_gene.map(labels)
        require(current.strength_stratum.notna().all(), "strength target mapping failed")
        pieces.append(current)
        thresholds.append(dict(seed=int(seed), split=split, rms_q1=q1, rms_q2=q2,
                               definition="sqrt(mean(delta_true**2)); within-seed tertiles; post-hoc"))
    return pd.concat(pieces, ignore_index=True), pd.DataFrame(thresholds)
