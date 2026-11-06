"""Gene loadings and source-X summary features fitted only on training controls."""
from collections import Counter
import numpy as np
from sklearn.decomposition import PCA
from src.c_common import require


def fit_control_features(controls, gene_names, target_ids, components=50, seed=42):
    require(controls.shape[1] == len(gene_names), "Measured gene order mismatch")
    require(controls.shape[0] >= 2, "PCA needs at least two training controls")
    x = controls.astype(np.float64)
    require(np.isfinite(x.data).all() and (x.data >= 0).all(), "Invalid control values")
    mean = np.asarray(x.mean(axis=0)).ravel()
    variance = np.maximum(np.asarray(x.multiply(x).mean(axis=0)).ravel() - mean**2, 0)
    detection = np.asarray((x > 0).mean(axis=0)).ravel()
    variable = variance > 1e-12
    k = min(components, int(variable.sum()), x.shape[0] - 1)
    require(k > 0, "No variable features available for PCA")
    pca = PCA(n_components=k, svd_solver="randomized", random_state=seed, iterated_power=4,
              n_oversamples=10, whiten=False)
    pca.fit(x[:, variable].toarray())
    keep = pca.explained_variance_ > max(pca.explained_variance_[0] * 1e-10, 1e-12)
    require(keep.any(), "Numerically zero PCA rank")
    axes = np.zeros((int(keep.sum()), x.shape[1]), dtype=np.float64)
    axes[:, variable] = pca.components_[keep]
    names = list(gene_names)
    counts = Counter(names)
    index = {name: i for i, name in enumerate(names) if counts[name] == 1}
    available = np.array([name in index for name in target_ids], dtype=bool)
    mapped = np.array([index.get(name, -1) for name in target_ids])
    features_pca = np.zeros((len(target_ids), axes.shape[0]))
    features_simple = np.zeros((len(target_ids), 2))
    features_pca[available] = axes[:, mapped[available]].T
    features_simple[available] = np.column_stack((mean[mapped[available]], detection[mapped[available]]))
    return {
        "target_ids": np.asarray(target_ids, dtype=str), "feature_available": available, "mapped_column": mapped,
        "pca": features_pca, "mean_detection": features_simple,
        "combined": np.column_stack((features_pca, features_simple)),
        "pca_components": axes, "pca_mean": mean, "gene_variance": variance,
        "gene_detection": detection, "explained_variance": pca.explained_variance_[keep],
        "explained_variance_ratio": pca.explained_variance_ratio_[keep],
        "zero_variance_target": available & np.array([variance[i] <= 1e-12 if i >= 0 else False for i in mapped]),
    }
