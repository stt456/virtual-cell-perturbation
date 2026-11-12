"""A reference model using C's frozen inputs and D's prediction format."""
import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.c_common import (load_npz, read_json, require, save_predictions, sha256,
                          unique_strings, validate_split, validate_training_labels, write_json)

PARTS = ("train", "val", "test")
INPUTS = Path("data/processed/c_inputs")
FEATURES = Path("data/processed/c_features")


def load_seed(data_root, seed, frozen_hashes):
    inputs = INPUTS / f"seed_{seed}"
    feature_path = FEATURES / f"seed_{seed}/features.npz"
    paths = [inputs / name for name in ("split.json", "output_genes.txt", "labels_train_val.npz")]
    paths.append(feature_path)
    missing = [p.as_posix() for p in paths if not (data_root / p).is_file()]
    require(not missing, f"Seed {seed} missing C inputs: {', '.join(missing)}")
    hashes = {p.as_posix(): sha256(data_root / p) for p in paths}
    for name, digest in hashes.items():
        require(digest == frozen_hashes.get(name), f"C frozen input hash mismatch: {name}")

    split = read_json(data_root / paths[0])
    require(split["seed"] == seed, "Seed mismatch")
    validate_split(split)
    genes = np.asarray((data_root / paths[1]).read_text(encoding="utf-8").splitlines(), dtype=str)
    unique_strings(genes, "output genes")
    require(len(genes) > 0, "No output genes")
    labels = load_npz(data_root / paths[2])
    validate_training_labels(labels, split, genes)
    features = load_npz(data_root / feature_path)
    targets = [target for part in PARTS for target in split["targets"][part]]
    require(features["target_ids"].tolist() == targets, "Feature target order mismatch")
    require(features["fit_control_ids"].tolist() == split["control_cell_ids"]["train"],
            "Features must be fitted on the training controls")
    return split, genes, labels["delta_true"], features, hashes


def cosine_similarity(train, query):
    center = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale == 0] = 1.0

    def unit(rows):
        values = (rows - center) / scale
        norm = np.linalg.norm(values, axis=1, keepdims=True)
        return np.divide(values, norm, out=np.zeros_like(values), where=norm != 0)

    return np.clip(unit(query) @ unit(train).T, -1, 1)


def weighted_predictions(similarity, neighbors, train_delta, k):
    chosen = neighbors[:, :k]
    weights = np.maximum(np.take_along_axis(similarity, chosen, axis=1), 0.0)
    weights[weights.sum(axis=1) == 0] = 1.0
    weights /= weights.sum(axis=1, keepdims=True)
    return np.einsum("ij,ijk->ik", weights, train_delta[chosen])


def fit_predict(split, labels, features, representation, candidates):
    counts = [len(split["targets"][part]) for part in PARTS]
    n_train, n_val, _ = counts
    x = np.asarray(features[representation], dtype=np.float64)
    available = features["feature_available"]
    require(available.dtype == np.bool_ and available.shape == (sum(counts),), "Invalid coverage mask")
    require(x.ndim == 2 and x.shape[0] == len(available) and x.shape[1] > 0,
            "Invalid feature matrix")
    require(np.isfinite(x[available]).all(), "Nonfinite features for available targets")

    training = np.flatnonzero(available[:n_train])
    train_ids = np.asarray(split["targets"]["train"], dtype=str)
    training = training[np.argsort(train_ids[training], kind="stable")]
    k_values = [k for k in sorted(set(candidates)) if k <= len(training)]
    require(k_values and k_values[0] > 0, "No valid k for available training targets")
    training_delta = labels[training].astype(np.float64)
    # Use the same all-target mean as C's baseline, including targets without features.
    train_mean = labels[:n_train].mean(axis=0)
    mask_val = available[n_train:n_train + n_val]
    mask_test = available[n_train + n_val:]
    require(mask_val.any(), "No validation target has features for k selection")
    truth = labels[n_train:n_train + n_val]

    def predict_for(k, mask, similarity, neighbors):
        values = np.broadcast_to(train_mean, (len(mask), len(train_mean))).astype(np.float64, copy=True)
        values[mask] = weighted_predictions(similarity, neighbors, training_delta, k)
        return values

    sim_val = cosine_similarity(x[training], x[n_train:n_train + n_val][mask_val])
    val_order = np.argsort(-sim_val, axis=1, kind="stable")
    scores = {}
    for k in k_values:
        pred = predict_for(k, mask_val, sim_val, val_order)
        scores[k] = float(np.mean((pred - truth) ** 2))
    best = min(scores.values())
    selected = min(k for k, error in scores.items() if error <= best + 1e-12)
    val = predict_for(selected, mask_val, sim_val, val_order)
    sim_test = cosine_similarity(x[training], x[n_train + n_val:][mask_test])
    test = predict_for(selected, mask_test, sim_test, np.argsort(-sim_test, axis=1, kind="stable"))
    return val, test, mask_val, mask_test, selected, scores


def run(data_root, output, freeze, representation, candidates, synthetic=False):
    require(not output.exists(), f"Output already exists: {output}")
    seeds = freeze["seeds"]
    require(seeds and len(set(seeds)) == len(seeds), "Invalid frozen seeds")
    runs = []
    for seed in seeds:
        split, genes, labels, features, hashes = load_seed(data_root, seed, freeze["external_input_hashes"])
        result = fit_predict(split, labels, features, representation, candidates)
        runs.append((seed, split, genes, hashes, result))

    output.mkdir(parents=True, exist_ok=False)
    manifest = {"model": "knn", "synthetic": synthetic, "test_labels_accessed": False, "seeds": {}}
    for seed, split, genes, hashes, result in runs:
        val, test, mask_val, mask_test, k, scores = result
        folder = output / f"seed_{seed}"
        folder.mkdir()
        val_file, test_file = folder / "validation_predictions.npz", folder / "predictions.npz"
        save_predictions(val_file, split["targets"]["val"], genes, val, mask_val, ~mask_val)
        save_predictions(test_file, split["targets"]["test"], genes, test, mask_test, ~mask_test)
        write_json(folder / "run.json", {
            "seed": seed, "synthetic": synthetic, "representation": representation,
            "k_candidates": candidates, "selected_k": k, "validation_macro_mse": scores,
            "input_hashes": hashes, "validation_sha256": sha256(val_file),
            "test_sha256": sha256(test_file), "test_labels_accessed": False,
        })
        manifest["seeds"][str(seed)] = {
            "predictions": test_file.relative_to(output).as_posix(), "sha256": sha256(test_file),
            "validation_predictions": val_file.relative_to(output).as_posix(),
            "validation_sha256": sha256(val_file),
        }
        print(f"seed {seed}: k={k}, val MSE={scores[k]:.8g}, "
              f"coverage val={mask_val.sum()}/{len(mask_val)}, test={mask_test.sum()}/{len(mask_test)}")
    name = "smoke_manifest.json" if synthetic else "manifest.json"
    write_json(output / name, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--freeze", type=Path, default=ROOT / "results/c_v1_20261106/freeze.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--representation", choices=("combined", "pca", "mean_detection"), default="combined")
    parser.add_argument("--k", nargs="+", type=int, default=[3, 5, 10, 20])
    args = parser.parse_args()
    if min(args.k) < 1:
        parser.error("Every k must be positive")
    try:
        run(args.data_root, args.output, read_json(args.freeze), args.representation, args.k)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"kNN: {error}\n")


if __name__ == "__main__":
    main()
