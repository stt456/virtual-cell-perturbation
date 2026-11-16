"""Record rebuilt features after checking them against C's frozen Ridge outputs."""

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
C_RESULTS = Path("results/c_v1_20261106")
TOLERANCE = 1e-10


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_frozen_array(archive, relative, hashes):
    data = archive.read((C_RESULTS / relative).as_posix())
    require(hashlib.sha256(data).hexdigest() == hashes[relative],
            f"C handoff file changed: {relative}")
    with np.load(BytesIO(data), allow_pickle=False) as values:
        return dict(values)


def verify_seed(archive, frozen, seed):
    hashes = frozen["external_input_hashes"]
    directory = Path(f"data/processed/c_inputs/seed_{seed}")
    required = [directory / name for name in ("split.json", "output_genes.txt",
                                               "labels_train_val.npz")]
    actual = {path.as_posix(): digest(ROOT / path) for path in required}
    for key, value in actual.items():
        require(value == hashes[key], f"Frozen input changed: {key}")
    spec = json.loads((ROOT / required[0]).read_text(encoding="utf-8"))
    genes = (ROOT / required[1]).read_text(encoding="utf-8").splitlines()
    with np.load(ROOT / required[2], allow_pickle=False) as labels:
        train_mean = labels["delta_true"][:len(spec["targets"]["train"])].mean(axis=0)

    feature_path = Path(f"data/processed/c_features/seed_{seed}/features.npz")
    actual[feature_path.as_posix()] = digest(ROOT / feature_path)
    feature_meta = json.loads((ROOT / feature_path.with_name("features.json"))
                              .read_text(encoding="utf-8"))
    require(actual[feature_path.as_posix()] == feature_meta["features_sha256"],
            f"Feature cache changed for seed {seed}")
    with np.load(ROOT / feature_path, allow_pickle=False) as feature:
        train, val, test = (spec["targets"][part] for part in ("train", "val", "test"))
        require(feature["target_ids"].tolist() == train + val + test, "Target order changed")
        require(feature["fit_control_ids"].tolist() == spec["control_cell_ids"]["train"],
                "Features must use frozen training controls")
        x = feature["combined"]
        available = feature["feature_available"]
    n_train = len(train)
    val_slice = slice(n_train, n_train + len(val))
    train_features = x[:n_train][available[:n_train]]
    val_mask = available[val_slice]
    folder = f"ridge_combined/seed_{seed}/"
    model = load_frozen_array(archive, folder + "model_arrays.npz", frozen["artifact_hashes"])
    validation = load_frozen_array(archive, folder + "validation_predictions.npz",
                                   frozen["artifact_hashes"])
    require(val == validation["target_ids"].tolist(), "C validation targets changed")
    require(genes == validation["gene_ids"].tolist(), "C validation genes changed")
    require(np.array_equal(val_mask, validation["feature_available"]), "C feature coverage changed")
    values = np.broadcast_to(model["train_mean"], validation["delta_pred"].shape).copy()
    scaled = (x[val_slice][val_mask] - model["scale_mean"]) / model["scale_std"]
    values[val_mask] = scaled @ model["coef"].T + model["intercept"]
    errors = {
        "ridge_validation_max_abs_difference": float(
            np.max(np.abs(values - validation["delta_pred"]))),
        "ridge_training_label_mean_max_abs_difference": float(
            np.max(np.abs(train_mean - model["train_mean"]))),
        "ridge_training_feature_mean_max_abs_difference": float(
            np.max(np.abs(train_features.mean(axis=0) - model["scale_mean"]))),
        "ridge_training_feature_std_max_abs_difference": float(
            np.max(np.abs(train_features.std(axis=0) - model["scale_std"]))),
    }
    require(all(np.isfinite(value) and value <= TOLERANCE for value in errors.values()),
            f"Rebuilt feature verification failed for seed {seed}: {errors}")
    errors["feature_sha256_matches_original"] = (
        actual[feature_path.as_posix()] == hashes[feature_path.as_posix()])
    errors["original_feature_sha256"] = hashes[feature_path.as_posix()]
    return actual, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c-handoff", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "results/a_local_freeze.json")
    args = parser.parse_args()
    original = ROOT / C_RESULTS / "freeze.json"
    frozen = json.loads(original.read_text(encoding="utf-8"))
    inputs, checks = {}, {}
    with ZipFile(args.c_handoff) as archive:
        for seed in frozen["seeds"]:
            actual, errors = verify_seed(archive, frozen, seed)
            inputs.update(actual)
            checks[str(seed)] = errors
    record = {"seeds": frozen["seeds"], "external_input_hashes": inputs,
              "mode": "locally_rebuilt_c_features", "original_c_freeze_sha256": digest(original),
              "tolerance": TOLERANCE, "verification": checks}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(record, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
