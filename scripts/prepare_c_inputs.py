"""Export training controls and train/val labels; never load D_test labels."""
import argparse
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import anndata as ad
import numpy as np
from scipy import sparse
from src.c_common import (environment, load_npz, read_json, require, sha256,
                         validate_split, validate_training_labels, write_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/c_inputs")
    args = parser.parse_args()
    installed = read_json(ROOT / "results/c_data_verification_20261106/installed_data_manifest.json")
    expression = ROOT / "data/processed/expression.h5ad"
    require(sha256(expression) == installed[str(expression.relative_to(ROOT))]["sha256"], "Expression hash changed")
    data = ad.read_h5ad(expression, backed="r")
    try:
        for seed in (42, 123, 2026):
            source = ROOT / f"data/processed/seed_{seed}"
            split_path = ROOT / f"data/splits/seed_{seed}.json"
            spec = read_json(split_path)
            validate_split(spec)
            genes = (source / "output_genes.txt").read_text().splitlines()
            labels = load_npz(source / "C_train_val.npz")
            validate_training_labels(labels, spec, genes)
            inputs = [split_path, source / "C_train_val.npz", source / "output_genes.txt", source / "metadata.json"]
            hashes = {str(p.relative_to(ROOT)): sha256(p) for p in inputs}
            for name, value in hashes.items():
                require(value == installed[name]["sha256"], f"Verified input changed: {name}")
            signature = {"seed": seed, "inputs": hashes, "expression_sha256": installed[str(expression.relative_to(ROOT))]["sha256"],
                         "preparer_sha256": sha256(__file__)}
            folder = args.output / f"seed_{seed}"
            if folder.exists():
                ready = read_json(folder / "ready.json")
                require(ready["signature"] == signature, "Input cache changed; use a new output directory")
                for name, value in ready["files"].items():
                    require(sha256(folder / name) == value, f"Corrupt input cache: {name}")
                print(f"REUSE seed {seed}", flush=True)
                continue
            folder.mkdir(parents=True)
            ids = data.obs_names.get_indexer(spec["control_cell_ids"]["train"])
            require((ids >= 0).all() and len(set(ids)) == len(ids), "Training control ID missing/duplicated")
            require(data.obs.iloc[ids].is_control.astype(bool).all(), "Non-control cell in training controls")
            pieces = [data.X[ids[start:start+512], :].tocsr() for start in range(0, len(ids), 512)]
            controls = sparse.vstack(pieces, format="csr")
            require(np.isfinite(controls.data).all() and (controls.data >= 0).all(), "Unexpected source control values")
            sparse.save_npz(folder / "controls_train.npz", controls)
            np.savez_compressed(folder / "control_metadata.npz", cell_ids=np.asarray(spec["control_cell_ids"]["train"], dtype=str),
                                measured_gene_ids=np.asarray(data.var_names, dtype=str),
                                measured_gene_names=np.asarray(data.var.gene_name, dtype=str))
            for src, name in [(split_path, "split.json"), (source / "C_train_val.npz", "labels_train_val.npz"),
                              (source / "output_genes.txt", "output_genes.txt"), (source / "metadata.json", "metadata.json")]:
                shutil.copyfile(src, folder / name)
            files = {p.name: sha256(p) for p in sorted(folder.iterdir()) if p.is_file()}
            write_json(folder / "ready.json", {"signature": signature, "files": files, "environment": environment(),
                                               "shape": list(controls.shape), "contains_test_labels": False})
            print(f"PREPARED seed {seed}: controls {controls.shape}, train/val labels {labels['delta_true'].shape}", flush=True)
    finally:
        data.file.close()


if __name__ == "__main__":
    main()
