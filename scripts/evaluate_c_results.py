"""Freeze validation-selected artifacts, then evaluate in a separate process."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from src.c_common import code_provenance, load_npz, read_json, require, sha256, utc_now, write_json
from src.evaluation.c_metrics import METRICS, check_prediction_package, evaluate, summarize


def verify_files(base, files):
    for name, expected in files.items():
        require(sha256(base / name) == expected, f"Frozen file changed: {name}")


def validated_training_mean(frozen_mean, recomputed_mean):
    # C-/Fortran-contiguous reductions may differ in the final float64 bits.
    require(np.asarray(frozen_mean).shape == np.asarray(recomputed_mean).shape and
            np.allclose(frozen_mean, recomputed_mean, rtol=1e-12, atol=1e-14),
            "Model baseline is not the training equal-weight mean")
    return np.asarray(frozen_mean, dtype=np.float64)


def model_names(config):
    return ["zero", "train_mean"] + [family + "_" + rep for rep in config["representations"] for family in ("ridge", "mlp")]


def aggregate_summary(summary):
    output = []
    for (model, split, subset), group in summary.groupby(["model", "split", "subset"], sort=True):
        row = {"model": model, "split": split, "subset": subset, "n_seeds": len(group),
               "mean_targets": float(group.n_targets.mean())}
        for metric in METRICS:
            row[metric + "_mean"] = float(group[metric].mean())
            row[metric + "_std"] = float(group[metric].std(ddof=1))
        output.append(row)
    return pd.DataFrame(output)


def freeze(base):
    require(not (base / "freeze.json").exists(), "Experiment already frozen")
    experiment, completed = read_json(base / "experiment.json"), read_json(base / "completed.json")
    require(experiment["signature"] == completed["signature"] and experiment["signature"]["mode"] == "formal", "Only completed formal experiments can be frozen")
    provenance = code_provenance()
    require(provenance["source_tree_sha256"] == experiment["signature"]["source_tree_sha256"], "Source changed after training")
    config = experiment["signature"]["config"]
    names = model_names(config)
    validation, files, external = [], {}, {}
    verified = read_json(ROOT / "results/c_data_verification_20261106/installed_data_manifest.json")
    expected_test_hashes = {}
    for seed in config["seeds"]:
        inputs = ROOT / config["paths"]["inputs"] / f"seed_{seed}"
        features = ROOT / config["paths"]["features"] / f"seed_{seed}"
        spec = read_json(inputs / "split.json")
        genes = (inputs / "output_genes.txt").read_text().splitlines()
        verify_files(inputs, read_json(inputs / "ready.json")["files"])
        current_inputs = {"input_ready": sha256(inputs / "ready.json"), "split": sha256(inputs / "split.json"),
                          "training_labels": sha256(inputs / "labels_train_val.npz"), "genes": sha256(inputs / "output_genes.txt"),
                          "features": sha256(features / "features.npz"), "feature_metadata": sha256(features / "features.json")}
        for path in [*inputs.iterdir(), *features.iterdir()]:
            if path.is_file():
                external[str(path.relative_to(ROOT))] = sha256(path)
        expected_test_hashes[str(seed)] = verified[f"data/processed/seed_{seed}/D_test.npz"]["sha256"]
        for name in names:
            folder = base / name / f"seed_{seed}"
            run = read_json(folder / "run.json")
            require(run["mode"] == "formal" and not run["test_labels_accessed"], "Invalid formal run")
            require(run["source_tree_sha256"] == provenance["source_tree_sha256"], "Different source among runs")
            require(run["train_target_ids"] == spec["targets"]["train"] and run["val_target_ids"] == spec["targets"]["val"], "Formal run subset mismatch")
            require(run["input_hashes"] == current_inputs, "Inputs changed before freeze")
            verify_files(folder, run["artifact_hashes"])
            check_prediction_package(load_npz(folder / "predictions.npz"), spec["targets"]["test"], genes)
            validation.append(pd.read_csv(folder / "validation_summary.csv"))
            for path in folder.iterdir():
                if path.is_file():
                    files[str(path.relative_to(base))] = sha256(path)
    summary = pd.concat(validation, ignore_index=True)
    summary.to_csv(base / "validation_summary.csv", index=False)
    comparison = aggregate_summary(summary)
    comparison.to_csv(base / "validation_model_comparison.csv", index=False)
    best_learner, best_including_baselines = {}, {}
    for seed in config["seeds"]:
        main = summary[(summary.subset == "all") & (summary.seed == seed)].sort_values(["mse", "model"])
        best_learner[str(seed)] = main[~main.model.isin(["zero", "train_mean"])].iloc[0].model
        best_including_baselines[str(seed)] = main.iloc[0].model
    for path in [base / "experiment.json", base / "completed.json", base / "validation_summary.csv", base / "validation_model_comparison.csv"]:
        files[str(path.relative_to(base))] = sha256(path)
    write_json(base / "freeze.json", {"frozen_utc": utc_now(), "artifact_hashes": files, "external_input_hashes": external,
                                      "test_label_expected_sha256": expected_test_hashes, "source_tree_sha256": provenance["source_tree_sha256"],
                                      "models": names, "seeds": config["seeds"], "validation_selected_learner_by_seed": best_learner,
                                      "validation_selected_method_including_baselines_by_seed": best_including_baselines,
                                      "test_labels_accessed": False, "evaluation_config": config["evaluation"]})
    print(f"FROZEN {len(names)*len(config['seeds'])} runs; per-seed validation selection: {best_learner}; no test labels loaded", flush=True)


def evaluate_frozen(base):
    frozen, experiment = read_json(base / "freeze.json"), read_json(base / "experiment.json")
    require(not (base / "evaluation").exists(), "Evaluation already exists; do not silently repeat test-based work")
    current_code = code_provenance()
    correction = None
    if current_code["source_tree_sha256"] != frozen["source_tree_sha256"]:
        original = experiment["provenance"]["source_files"]
        current = current_code["source_files"]
        changes = {name: {"original": original.get(name), "corrected": current.get(name)}
                   for name in set(original) | set(current) if original.get(name) != current.get(name)}
        allowed = {"scripts/evaluate_c_results.py", "tests/test_c_evaluator_precision.py"}
        require(set(changes) <= allowed, "Training/metric source changed after freeze")
        correction = read_json(base / "evaluation_correction.json")
        require(correction["source_changes"] == changes and correction["freeze_sha256"] == sha256(base / "freeze.json"),
                "Evaluation correction is not documented for these frozen artifacts")
    verify_files(base, frozen["artifact_hashes"])
    verify_files(ROOT, frozen["external_input_hashes"])
    config = experiment["signature"]["config"]
    all_rows, all_summary = [], []
    for seed in frozen["seeds"]:
        inputs = ROOT / config["paths"]["inputs"] / f"seed_{seed}"
        spec = read_json(inputs / "split.json")
        c_labels = load_npz(inputs / "labels_train_val.npz")
        recomputed_means = c_labels["delta_true"][c_labels["split"] == "train"].mean(axis=0)
        d_path = ROOT / f"data/processed/seed_{seed}/D_test.npz"
        require(sha256(d_path) == frozen["test_label_expected_sha256"][str(seed)], "D labels changed after data verification")
        test_labels = load_npz(d_path)
        genes = c_labels["gene_ids"].tolist()
        require(test_labels["gene_ids"].tolist() == genes and test_labels["target_ids"].tolist() == spec["targets"]["test"], "D label order mismatch")
        for name in frozen["models"]:
            folder = base / name / f"seed_{seed}"
            model_arrays = load_npz(folder / "model_arrays.npz")
            means = validated_training_mean(model_arrays["train_mean"], recomputed_means)
            for split, file in (("val", "validation_predictions.npz"), ("test", "predictions.npz")):
                prediction = load_npz(folder / file)
                check_prediction_package(prediction, spec["targets"][split], genes)
                truth = test_labels["delta_true"] if split == "test" else c_labels["delta_true"][c_labels["split"] == "val"]
                if prediction["used_fallback"].any():
                    subset = prediction["delta_pred"][prediction["used_fallback"]]
                    require(np.array_equal(subset, np.broadcast_to(means, subset.shape)), "Fallback mismatch during evaluation")
                rows = evaluate(prediction["delta_pred"], truth, means, prediction["target_ids"], genes,
                                prediction["feature_available"], prediction["used_fallback"], name, seed, split, config["evaluation"]["top_k"])
                all_rows.extend(rows)
                all_summary.extend(summarize(rows))
    per_target = pd.DataFrame(all_rows)
    summary = pd.DataFrame(all_summary)
    bootstrap = []
    for (seed, split), group in per_target.groupby(["seed", "split"]):
        reference = group[group.model == "train_mean"].set_index("target_gene")
        for subset in ("all", "available", "unavailable"):
            ids = reference.index if subset == "all" else reference.index[reference.feature_available == (subset == "available")]
            if not len(ids):
                continue
            rng = np.random.default_rng(int(seed) + (10000 if split == "test" else 0) + (0 if subset == "all" else 100 if subset == "available" else 200))
            draws = rng.integers(0, len(ids), size=(config["evaluation"]["bootstrap_repeats"], len(ids)))
            for name in frozen["models"]:
                current = group[group.model == name].set_index("target_gene").loc[ids]
                difference = reference.loc[ids, "mse"].to_numpy() - current.mse.to_numpy()
                distribution = difference[draws].mean(axis=1)
                lo, hi = np.quantile(distribution, [.025, .975])
                bootstrap.append({"seed": seed, "split": split, "subset": subset, "model": name,
                                  "n_targets": len(ids), "mean_mse_improvement_over_train_mean": difference.mean(),
                                  "ci95_low": lo, "ci95_high": hi, "bootstrap_repeats": len(draws)})
    destination = base / "evaluation"
    destination.mkdir()
    per_target.to_csv(destination / "per_target.csv", index=False)
    summary.to_csv(destination / "summary_by_seed.csv", index=False)
    aggregate_summary(summary).to_csv(destination / "summary_across_seeds.csv", index=False)
    pd.DataFrame(bootstrap).to_csv(destination / "paired_bootstrap.csv", index=False)
    report = {"completed_utc": utc_now(), "implementation": config["evaluation"]["implementation"],
              "freeze_sha256": sha256(base / "freeze.json"), "source_tree_sha256": current_code["source_tree_sha256"],
              "training_source_tree_sha256": frozen["source_tree_sha256"], "evaluation_code_correction": correction,
              "test_label_sha256": frozen["test_label_expected_sha256"], "models": frozen["models"], "seeds": frozen["seeds"],
              "validation_selected_learner_by_seed": frozen["validation_selected_learner_by_seed"],
              "notes": ["No hyperparameters were changed after the test evaluation.",
                        "MSE after subtracting the same training mean is identical to the original MSE and is not duplicated.",
                        "Bootstrap pairs targets within a seed; overlapping seeds are not pooled as independent observations.",
                        "Method selection is separate per seed; validation labels from another split do not select its model.",
                        "Preparation/audit previously held full data; this is process isolation, not independent third-party blinding.",
                        "A reference method and independent D acceptance are not part of these C diagnostic results."]}
    report["files"] = {p.name: sha256(p) for p in destination.iterdir() if p.is_file()}
    write_json(destination / "evaluation.json", report)
    print(aggregate_summary(summary).query("split == 'test' and subset == 'all'")[["model", "mse_mean", "mse_std", "pearson_mean", "residual_pearson_mean"]].to_string(index=False), flush=True)
    print("COMPLETE: frozen predictions evaluated; C diagnostic results require independent D review", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=ROOT / "results/c_v1_20261106")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--freeze", action="store_true")
    action.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    (freeze if args.freeze else evaluate_frozen)(args.experiment.resolve())


if __name__ == "__main__":
    main()
