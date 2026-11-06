"""Independently check the extracted source and an isolated B reproduction.

Run after the original B scripts, without fitting models or using their helpers.
The scan uses float64 sparse row reductions; B aggregates via group assignment.
"""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    audit = args.audit_dir.resolve()
    source = root / "data/replogle_k562_essential/perturb_processed.h5ad"
    reproduction = audit / "b_reproduction"
    results = reproduction / "results"
    interface = reproduction / "仓库接口交付_v0.1"
    reference = json.loads((audit / "reference_data_report.json").read_text())
    reproduced = json.loads((reproduction / "reports/data_report.json").read_text())
    source_sha = digest(source)
    extraction = json.loads((audit / "extraction.json").read_text())
    require(source_sha == extraction["files"][0]["sha256"] ==
            "550fde89ac85c258c9fd517638fe34fb42c0e748c9305fe6355761a2bb754170", "Source hash mismatch")
    installed_path = audit / "installed_data_manifest.json"
    installed = json.loads(installed_path.read_text()) if installed_path.exists() else {}
    for name, expected in installed.items():
        require(digest(root / name) == expected["sha256"], f"Installed data changed: {name}")
    script_hashes = json.loads((audit / "b_script_hashes.json").read_text())
    for name, expected in script_hashes.items():
        require(digest(root / "src" / name) == expected, f"B source changed: {name}")
        require(digest(reproduction / "src" / name) == expected, f"B copy changed: {name}")

    data = ad.read_h5ad(source, backed="r")
    obs, var = data.obs.copy(), data.var.copy()
    shape = list(data.shape)
    raw_shape = list(data.raw.shape) if data.raw is not None else None
    layer_names, uns_keys = list(data.layers), list(data.uns)
    data.file.close()
    require(obs.index.is_unique and var.index.is_unique, "Nonunique matrix identifiers")
    require(not obs.condition.isna().any(), "Missing condition")
    labels = obs.condition.astype(str).to_numpy()
    condition_to_target = {}
    for label in np.unique(labels):
        if label == "ctrl":
            condition_to_target[label] = "ctrl"
        else:
            parts = label.split("+")
            require(len(parts) == 2 and parts[1] == "ctrl" and parts[0] not in ("", "ctrl"),
                    f"Unsupported condition: {label}")
            condition_to_target[label] = parts[0]
    targets = np.array([condition_to_target[label] for label in labels])
    counts = Counter(targets[targets != "ctrl"])
    valid = sorted(target for target, count in counts.items() if count >= 20)
    eligible = np.isin(targets, valid) | (targets == "ctrl")
    symbols = Counter(var.gene_name.astype(str))
    covered = [target for target in valid if target in symbols]
    ambiguous = [target for target in valid if symbols[target] > 1]
    numeric = {
        "shape": shape, "conditions": len(np.unique(labels)),
        "controls": int(np.sum(targets == "ctrl")), "kept_cells": int(eligible.sum()),
        "excluded_cells": int((~eligible).sum()), "targets_before_filter": len(counts),
        "targets_after_filter": len(valid), "duplicate_gene_symbols": int(var.gene_name.duplicated().sum()),
        "missing_conditions": int(obs.condition.isna().sum()),
    }
    for field, actual in numeric.items():
        require(actual == reference[field] == reproduced[field], f"Report mismatch: {field}")

    coverage_rows, split_rows, samples, seed_inputs = [], [], [], []
    for seed in (42, 123, 2026):
        split = pd.read_csv(results / f"split_{seed}.csv", keep_default_na=False)
        require(split.cell_id.tolist() == obs.index.tolist(), f"Cell order: {seed}")
        require(np.array_equal(split.target.to_numpy(), targets), f"Target parsing: {seed}")
        rng = np.random.RandomState(seed)
        ordered = np.array(valid)
        rng.shuffle(ordered)
        first, second = int(len(ordered) * .7), int(len(ordered) * .15)
        parts = ("train", "val", "test")
        target_sets = dict(zip(parts, (ordered[:first], ordered[first:first+second], ordered[first+second:])))
        mapping = {target: part for part, items in target_sets.items() for target in items}
        expected_split = np.array([mapping.get(target, "excluded") for target in targets], dtype="<U8")
        controls = np.flatnonzero(targets == "ctrl")
        rng.shuffle(controls)
        first, second = int(len(controls) * .7), int(len(controls) * .15)
        control_sets = dict(zip(parts, (controls[:first], controls[first:first+second], controls[first+second:])))
        for part, ids in control_sets.items():
            expected_split[ids] = part
        require(np.array_equal(split["split"].to_numpy(), expected_split), f"Split assignment: {seed}")
        row = {"seed": seed, **{f"{part}_targets": len(target_sets[part]) for part in parts},
               **{f"{part}_cells": int(np.sum(expected_split == part)) for part in parts}, "hvg": 2000}
        split_rows.append(row)
        output = pd.read_csv(results / f"hvg_{seed}.csv")
        columns = var.index.get_indexer(output.gene_id)
        require(len(columns) == 2000 and (columns >= 0).all(), f"Output mapping: {seed}")
        require(output.gene_id.is_unique and output.gene_name.is_unique, f"Duplicate output: {seed}")
        require(var.gene_name.iloc[columns].astype(str).tolist() == output.gene_name.tolist(), "Symbol mismatch")
        means = pd.read_csv(results / f"mean_expression_{seed}.csv.gz", index_col=0)
        mean_counts = pd.read_csv(results / f"mean_counts_{seed}.csv").set_index("condition")
        require(means.columns.tolist() == output.gene_id.tolist(), f"Mean gene order: {seed}")
        require(np.isfinite(means.to_numpy()).all(), f"Invalid means: {seed}")
        for part in parts:
            ts = target_sets[part].tolist()
            coverage_rows.append({"seed": seed, "split": part, "targets": len(ts),
                                  "symbol_present": sum(target in symbols for target in ts),
                                  "unique_mapping": sum(symbols[target] == 1 for target in ts),
                                  "ambiguous_mapping": sum(symbols[target] > 1 for target in ts),
                                  "symbol_absent": sum(target not in symbols for target in ts),
                                  "control_cells": len(control_sets[part])})
            chosen = sorted(ts)[0]
            control_mask = (targets == "ctrl") & (expected_split == part)
            for name, mask in (("ctrl_" + part, control_mask), (chosen, targets == chosen)):
                require(int(mask.sum()) == int(mean_counts.loc[name, "n_cells"]), f"Counts: {seed}/{name}")
                samples.append({"seed": seed, "condition": name, "mask": mask,
                                "sum": np.zeros(shape[1]), "count": int(mask.sum()),
                                "columns": columns, "expected": means.loc[name].to_numpy()})
        folder = interface / f"data/processed/seed_{seed}"
        spec = json.loads((interface / f"data/splits/seed_{seed}.json").read_text())
        gene_ids = (folder / "output_genes.txt").read_text().splitlines()
        for filename, allowed in (("aggregates", parts), ("C_train_val", parts[:2]), ("D_test", parts[2:])):
            with np.load(folder / f"{filename}.npz", allow_pickle=False) as package:
                expected_ids = [target for part in allowed for target in sorted(target_sets[part])]
                require(package["target_ids"].tolist() == expected_ids, f"Package targets: {seed}/{filename}")
                require(package["gene_ids"].tolist() == gene_ids == output.gene_name.tolist(), "Package genes")
                require(set(package["split"].tolist()) == set(allowed), "Package split contamination")
                require(package["delta_true"].shape == (len(expected_ids), 2000), "Package dimensions")
                for key in ("delta_true", "mean_perturbed", "mean_control"):
                    require(np.isfinite(package[key]).all(), f"Invalid package values: {key}")
                require(np.allclose(package["delta_true"], package["mean_perturbed"] - package["mean_control"],
                                    rtol=0, atol=1e-12), "Incorrect delta")
                expected_control = np.stack([means.loc["ctrl_" + part].to_numpy() for part in package["split"]])
                require(np.allclose(package["mean_control"], expected_control, rtol=0, atol=1e-12), "Wrong controls")
                require(np.allclose(package["mean_perturbed"], means.loc[expected_ids].to_numpy(),
                                    rtol=0, atol=1e-12), "Wrong means")
        for part in parts:
            require(spec["targets"][part] == sorted(target_sets[part]), "JSON target mismatch")
            require(spec["control_cell_ids"][part] == obs.index[np.sort(control_sets[part])].tolist(), "JSON control mismatch")
        seed_inputs.append({"seed": seed, "mask": expected_split == "train", "sum": np.zeros(shape[1]),
                            "sum_sq": np.zeros(shape[1]), "count": int(np.sum(expected_split == "train")),
                            "output": output, "columns": columns})
    require(split_rows == reference["split_summary"] == reproduced["split_summary"], "Split summary mismatch")

    matrix = {"finite": True, "min_stored": float("inf"), "max_stored": float("-inf"),
              "stored_entries": 0, "noninteger_stored_entries": 0, "explicit_zeros": 0}
    with h5py.File(source, "r") as handle:
        group = handle["X"]
        require(group.attrs["encoding-type"] == "csr_matrix", "Expected CSR input")
        ptr = group["indptr"][:]
        require(len(ptr) == shape[0] + 1 and ptr[0] == 0 and (np.diff(ptr) >= 0).all(), "Invalid CSR pointers")
        require(ptr[-1] == len(group["data"]) == len(group["indices"]), "Invalid CSR lengths")
        for start in range(0, shape[0], 2048):
            end = min(start + 2048, shape[0])
            lower, upper = int(ptr[start]), int(ptr[end])
            values = group["data"][lower:upper].astype(np.float64)
            indices = group["indices"][lower:upper]
            require(np.isfinite(values).all(), "Nonfinite source expression")
            require((indices >= 0).all() and (indices < shape[1]).all(), "Invalid CSR column")
            if len(values):
                matrix["min_stored"] = min(matrix["min_stored"], float(values.min()))
                matrix["max_stored"] = max(matrix["max_stored"], float(values.max()))
            matrix["stored_entries"] += len(values)
            matrix["noninteger_stored_entries"] += int(np.sum(np.abs(values - np.rint(values)) > 1e-6))
            matrix["explicit_zeros"] += int(np.sum(values == 0))
            batch = csr_matrix((values, indices, ptr[start:end+1] - lower), shape=(end-start, shape[1]))
            for sample in samples:
                mask = sample["mask"][start:end]
                if mask.any():
                    sample["sum"] += np.asarray(batch[mask].sum(axis=0)).ravel()
            for item in seed_inputs:
                subset = batch[item["mask"][start:end]]
                item["sum"] += np.asarray(subset.sum(axis=0)).ravel()
                item["sum_sq"] += np.asarray(subset.multiply(subset).sum(axis=0)).ravel()
            if start % (2048 * 20) == 0:
                print(f"Scanned {end}/{shape[0]} rows", flush=True)

    aggregation = []
    for sample in samples:
        calculated = sample["sum"][sample["columns"]] / sample["count"]
        error = float(np.max(np.abs(calculated - sample["expected"])))
        require(error < 1e-5, f"Independent aggregation failed: {sample['condition']}")
        aggregation.append({"seed": sample["seed"], "condition": sample["condition"],
                            "cells": sample["count"], "max_abs_error_float64": error})
    hvg_checks = []
    for item in seed_inputs:
        mean = item["sum"] / item["count"]
        variance = np.maximum(item["sum_sq"] / item["count"] - mean**2, 0)
        dispersion = variance / (mean + 1e-12)
        bins = pd.qcut(pd.Series(mean).rank(method="first"), 20, labels=False, duplicates="drop").to_numpy()
        scores = np.zeros(shape[1])
        for label in np.unique(bins):
            mask = bins == label
            values = np.log1p(dispersion[mask])
            scores[mask] = (values - values.mean()) / (values.std() + 1e-12)
        selected = np.argsort(-scores, kind="stable")[:2000]
        original = item["columns"]
        require(set(selected) == set(original), f"Independent output gene set differs: {item['seed']}")
        hvg_checks.append({"seed": item["seed"], "independent_float64_overlap": len(set(selected) & set(original)),
                           "same_order": bool(np.array_equal(selected, original)),
                           "max_abs_train_mean_difference": float(np.max(np.abs(mean[original] - item["output"].train_mean))),
                           "max_abs_train_variance_difference": float(np.max(np.abs(variance[original] - item["output"].train_variance)))})

    verification = {
        "status": "PASS", "source_sha256": source_sha,
        "verification_script_sha256": digest(Path(__file__)),
        "installed_data_hashes_checked": len(installed),
        "numeric_report_matches": numeric, "raw_shape": raw_shape, "layers": layer_names,
        "obs_columns": obs.columns.tolist(), "upstream_uns_keys_not_used_for_fitting": uns_keys,
        "source_matrix_scan": matrix, "symbol_coverage": len(covered),
        "unique_symbol_coverage": len(covered) - len(ambiguous), "ambiguous_targets": ambiguous,
        "missing_target_count": len(valid) - len(covered), "coverage_by_split": coverage_rows,
        "split_summary": split_rows, "independent_aggregation": aggregation,
        "max_independent_aggregation_error": max(item["max_abs_error_float64"] for item in aggregation),
        "independent_hvg_precision_check": hvg_checks, "b_original_sources_unchanged": True,
        "all_three_interface_packages_passed": True,
        "limitations": ["No original B aggregate files were supplied for bytewise comparison.",
                        "Normalization provenance and upstream gene-selection scope remain unverified.",
                        "Aggregation recheck covers 18 condition/seed pairs, not every target.",
                        "This is data validation, not model training or evaluation."],
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "pandas", "h5py", "anndata")},
    }
    (audit / "independent_verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2) + "\n")
    pd.DataFrame(coverage_rows).to_csv(audit / "coverage_by_split.csv", index=False)
    pd.DataFrame(aggregation).to_csv(audit / "independent_aggregation.csv", index=False)
    files = [path for path in audit.rglob("*") if path.is_file() and "matplotlib-cache" not in path.parts
             and "data/replogle_k562_essential" not in path.as_posix() and path.name != "artifact_manifest.json"]
    manifest = {str(path.relative_to(audit)): {"bytes": path.stat().st_size, "sha256": digest(path)} for path in sorted(files)}
    (audit / "artifact_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: verification[key] for key in ("status", "symbol_coverage", "unique_symbol_coverage",
          "ambiguous_targets", "missing_target_count", "max_independent_aggregation_error", "independent_hvg_precision_check")},
          ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
