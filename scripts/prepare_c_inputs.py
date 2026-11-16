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


def verify_exported_expression(data):
    source = ROOT / "data/replogle_k562_essential/perturb_processed.h5ad"
    expected = "550fde89ac85c258c9fd517638fe34fb42c0e748c9305fe6355761a2bb754170"
    require(sha256(source) == expected, "K562 source hash changed")
    raw = ad.read_h5ad(source, backed="r")
    try:
        require(raw.var_names.equals(data.var_names), "Measured gene order changed")
        require(raw.var.gene_name.astype(str).equals(data.var.gene_name.astype(str)),
                "Measured gene symbols changed")
        conditions = raw.obs["condition"].astype(str)
        counts = conditions.value_counts()
        kept = conditions.eq("ctrl") | (
            conditions.str.fullmatch(r"[^+]+\+ctrl") & conditions.ne("ctrl+ctrl")
            & conditions.map(counts).ge(20)
        )
        kept_ids = np.flatnonzero(kept.to_numpy())
        require(raw.obs_names[kept_ids].equals(data.obs_names), "Filtered cell order changed")
        expected_control = conditions.iloc[kept_ids].eq("ctrl").to_numpy()
        require(np.array_equal(expected_control, data.obs.is_control.to_numpy()),
                "Control flags changed")
        for start in range(0, len(kept_ids), 2048):
            selected = kept_ids[start:start + 2048]
            original = raw.X[selected, :].tocsr()
            exported = data.X[start:start + len(selected), :].tocsr()
            for field in ("indptr", "indices", "data"):
                require(np.array_equal(getattr(original, field), getattr(exported, field)),
                        f"Expression values differ at row {start}")
    finally:
        raw.file.close()
    print("Verified exported cell IDs, gene IDs and all sparse expression values "
          "against the original K562 file.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/c_inputs")
    parser.add_argument("--verify-local-expression", action="store_true",
                        help="Independently verify a local export whose HDF5 bytes differ.")
    args = parser.parse_args()
    installed = read_json(ROOT / "results/c_data_verification_20261106/installed_data_manifest.json")
    expression = ROOT / "data/processed/expression.h5ad"
    expression_key = expression.relative_to(ROOT).as_posix()
    expression_sha = sha256(expression)
    expression_matches = expression_sha == installed[expression_key]["sha256"]
    require(expression_matches or args.verify_local_expression, "Expression hash changed")
    data = ad.read_h5ad(expression, backed="r")
    try:
        if not expression_matches:
            verify_exported_expression(data)
        for seed in (42, 123, 2026):
            source = ROOT / f"data/processed/seed_{seed}"
            split_path = ROOT / f"data/splits/seed_{seed}.json"
            spec = read_json(split_path)
            validate_split(spec)
            genes = (source / "output_genes.txt").read_text().splitlines()
            labels = load_npz(source / "C_train_val.npz")
            validate_training_labels(labels, spec, genes)
            inputs = [split_path, source / "C_train_val.npz", source / "output_genes.txt", source / "metadata.json"]
            hashes = {p.relative_to(ROOT).as_posix(): sha256(p) for p in inputs}
            for name, value in hashes.items():
                require(value == installed[name]["sha256"], f"Verified input changed: {name}")
            signature = {"seed": seed, "inputs": hashes, "expression_sha256": expression_sha,
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
