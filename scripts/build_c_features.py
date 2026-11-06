"""Fit PCA features separately for each seed, using exported training controls."""
import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scipy import sparse
import numpy as np
from threadpoolctl import threadpool_limits
from src.c_common import environment, load_npz, read_json, require, sha256, validate_split, write_json
from src.features.control_features import fit_control_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/c_models.json")
    args = parser.parse_args()
    config = read_json(args.config)
    fixed = {"center": True, "scale_genes": False, "whiten": False, "solver": "randomized", "iterated_power": 4}
    require(all(config["features"].get(key) == value for key, value in fixed.items()),
            "This v1 feature implementation requires the documented centered, unscaled PCA settings")
    for seed in config["seeds"]:
        started = time.monotonic()
        folder = ROOT / config["paths"]["inputs"] / f"seed_{seed}"
        ready = read_json(folder / "ready.json")
        for name, value in ready["files"].items():
            require(sha256(folder / name) == value, f"Input cache hash: {name}")
        spec = read_json(folder / "split.json")
        validate_split(spec)
        info = load_npz(folder / "control_metadata.npz")
        require(info["cell_ids"].tolist() == spec["control_cell_ids"]["train"], "PCA control leakage/order")
        signature = {"input_ready_sha256": sha256(folder / "ready.json"), "feature_config": config["features"],
                     "feature_code_sha256": sha256(ROOT / "src/features/control_features.py")}
        out = ROOT / config["paths"]["features"] / f"seed_{seed}"
        if out.exists():
            existing = read_json(out / "features.json")
            require(existing["signature"] == signature and sha256(out / "features.npz") == existing["features_sha256"],
                    "Feature cache changed; use a new feature directory")
            print(f"REUSE features seed {seed}", flush=True)
            continue
        controls = sparse.load_npz(folder / "controls_train.npz")
        targets = sum((spec["targets"][part] for part in ("train", "val", "test")), [])
        with threadpool_limits(limits=config["runtime"]["cpu_threads"]):
            result = fit_control_features(controls, info["measured_gene_names"].tolist(), targets,
                                          components=config["features"]["pca_components"], seed=seed)
        out.mkdir(parents=True)
        np.savez_compressed(out / "features.npz", **result, measured_gene_ids=info["measured_gene_ids"],
                            measured_gene_names=info["measured_gene_names"], fit_control_ids=info["cell_ids"])
        meta = {"signature": signature, "features_sha256": sha256(out / "features.npz"), "seed": seed,
                "pca_dimensions": result["pca"].shape[1], "fit_controls": controls.shape[0],
                "available_targets": int(result["feature_available"].sum()),
                "zero_variance_targets": result["target_ids"][result["zero_variance_target"]].tolist(),
                "explained_variance_ratio_sum": float(result["explained_variance_ratio"].sum()),
                "positive_fraction_semantics": "fraction of source X > 0; count normalization provenance unverified",
                "environment": environment(), "elapsed_seconds": time.monotonic() - started}
        write_json(out / "features.json", meta)
        print(f"FEATURES seed {seed}: {meta['pca_dimensions']} PCA dimensions, coverage {meta['available_targets']}/{len(targets)}, {meta['elapsed_seconds']:.2f}s", flush=True)


if __name__ == "__main__":
    main()
