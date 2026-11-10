import hashlib
import json
from pathlib import Path

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(json_value(value), ensure_ascii=False, indent=2,
                                   allow_nan=False) + "\n", encoding="utf-8")


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def ids(values, name):
    values = np.asarray(values)
    require(values.ndim == 1 and values.dtype.kind == "U", f"{name}: Unicode vector required")
    require(len(values) > 0 and all(str(v).strip() for v in values), f"{name}: empty ID")
    require(len(set(values.tolist())) == len(values), f"{name}: duplicate ID")
    return values


def validate_split(spec):
    for field in ("targets", "control_cell_ids"):
        require(field in spec, f"split: missing {field}")
        seen = set()
        for part in ("train", "val", "test"):
            values = ids(np.asarray(spec[field][part], dtype=str), f"{field}/{part}")
            require(not seen.intersection(values), f"{field}: leakage across splits")
            seen.update(values)


def validate_prediction(package, targets, genes):
    for key in ("target_ids", "gene_ids", "delta_pred", "feature_available", "used_fallback"):
        require(key in package, f"prediction missing {key}")
    t, g = ids(package["target_ids"], "target_ids"), ids(package["gene_ids"], "gene_ids")
    require(t.tolist() == list(targets), "prediction target order mismatch")
    require(g.tolist() == list(genes), "prediction gene order mismatch")
    values = package["delta_pred"]
    require(values.shape == (len(t), len(g)) and values.dtype.kind in "fi",
            "prediction matrix shape/type mismatch")
    require(np.isfinite(values).all(), "prediction has nonfinite values")
    for key in ("feature_available", "used_fallback"):
        require(package[key].dtype == np.bool_ and package[key].shape == (len(t),),
                f"prediction invalid {key}")


def validate_labels(package, targets, genes, split):
    require(ids(package["target_ids"], "label targets").tolist() == list(targets),
            "label target order mismatch")
    require(ids(package["gene_ids"], "label genes").tolist() == list(genes),
            "label gene order mismatch")
    values = package["delta_true"]
    require(values.shape == (len(targets), len(genes)) and np.isfinite(values).all(),
            "label matrix shape/nonfinite mismatch")
    require(package["split"].tolist() == list(split), "label split mismatch")
