"""Shared file contracts and provenance for C; no data-dependent global state."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(json_ready(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def unique_strings(values, name):
    values = np.asarray(values)
    require(values.ndim == 1 and values.dtype.kind == "U", f"{name}: expected Unicode vector")
    require(len(values) == len(set(values.tolist())), f"{name}: duplicate IDs")
    require(all(x.strip() for x in values.tolist()), f"{name}: blank ID")


def validate_split(spec):
    for field in ("targets", "control_cell_ids"):
        seen = set()
        for part in ("train", "val", "test"):
            values = spec[field][part]
            unique_strings(np.asarray(values, dtype=str), f"{field}/{part}")
            require(bool(values) and not seen.intersection(values), f"{field}: empty set or leakage")
            seen.update(values)


def validate_training_labels(data, spec, genes):
    expected = spec["targets"]["train"] + spec["targets"]["val"]
    require(data["target_ids"].tolist() == expected, "Training target order or test-label contamination")
    require(data["gene_ids"].tolist() == list(genes), "Training gene order mismatch")
    expected_split = ["train"] * len(spec["targets"]["train"]) + ["val"] * len(spec["targets"]["val"])
    require(data["split"].tolist() == expected_split, "Training split mismatch")
    require(data["delta_true"].shape == (len(expected), len(genes)), "Training label shape")
    require(np.isfinite(data["delta_true"]).all(), "Training labels are nonfinite")


def save_predictions(path, target_ids, gene_ids, values, available, fallback):
    target_ids, gene_ids = np.asarray(target_ids, dtype=str), np.asarray(gene_ids, dtype=str)
    unique_strings(target_ids, "prediction targets")
    unique_strings(gene_ids, "prediction genes")
    values = np.asarray(values, dtype=np.float64)
    require(values.shape == (len(target_ids), len(gene_ids)) and np.isfinite(values).all(), "Invalid prediction matrix")
    available, fallback = np.asarray(available, dtype=bool), np.asarray(fallback, dtype=bool)
    require(available.shape == fallback.shape == (len(target_ids),), "Prediction flags")
    np.savez_compressed(path, target_ids=target_ids, gene_ids=gene_ids, delta_pred=values,
                        feature_available=available, used_fallback=fallback)


def code_provenance():
    paths = sorted({*ROOT.glob("src/**/*.py"), *ROOT.glob("scripts/*.py"),
                    *ROOT.glob("tests/test_c*.py"), ROOT / "configs/c_models.json"})
    files = {str(p.relative_to(ROOT)): sha256(p) for p in paths}
    combined = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    return {"git_commit": commit, "source_tree_sha256": combined, "source_files": files}


def environment():
    versions = {}
    for name in ("numpy", "scipy", "scikit-learn", "torch", "anndata", "h5py", "pandas"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


def utc_now():
    return datetime.now(timezone.utc).isoformat()
