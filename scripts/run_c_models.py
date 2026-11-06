"""Validation-only model selection and label-free test prediction for C."""
import argparse
from copy import deepcopy
from pathlib import Path
import resource
import shlex
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from src.c_common import (code_provenance, environment, load_npz, read_json, require, save_predictions,
                         sha256, utc_now, validate_split, validate_training_labels, write_json)
from src.evaluation.c_metrics import evaluate, summarize
from src.models.c_models import load_model, macro_mse, predict, save_model, search


def load_seed(config, seed):
    folder = ROOT / config["paths"]["inputs"] / f"seed_{seed}"
    features_folder = ROOT / config["paths"]["features"] / f"seed_{seed}"
    ready, feature_meta = read_json(folder / "ready.json"), read_json(features_folder / "features.json")
    require(feature_meta["signature"]["input_ready_sha256"] == sha256(folder / "ready.json"), "Feature/input provenance mismatch")
    for name, value in ready["files"].items():
        require(sha256(folder / name) == value, f"Input changed: {name}")
    require(sha256(features_folder / "features.npz") == feature_meta["features_sha256"], "Features changed")
    spec = read_json(folder / "split.json")
    validate_split(spec)
    genes = (folder / "output_genes.txt").read_text().splitlines()
    labels = load_npz(folder / "labels_train_val.npz")
    validate_training_labels(labels, spec, genes)
    features = load_npz(features_folder / "features.npz")
    require(features["target_ids"].tolist() == sum((spec["targets"][p] for p in ("train", "val", "test")), []), "Feature target order")
    require(features["fit_control_ids"].tolist() == spec["control_cell_ids"]["train"], "Feature control leakage")
    hashes = {"input_ready": sha256(folder / "ready.json"), "split": sha256(folder / "split.json"),
              "training_labels": sha256(folder / "labels_train_val.npz"), "genes": sha256(folder / "output_genes.txt"),
              "features": sha256(features_folder / "features.npz"), "feature_metadata": sha256(features_folder / "features.json")}
    return spec, genes, labels, features, hashes, read_json(folder / "metadata.json")


def effective_config(config, mode):
    config = deepcopy(config)
    if mode != "formal":
        config["seeds"] = [42]
        config["representations"] = ["combined"]
        config["ridge"]["alphas"] = [1.0]
        config["mlp"]["hidden_dims"] = [64]
        config["mlp"]["learning_rates"] = [0.001]
        if mode == "smoke":
            config["mlp"].update(max_epochs=5, patience=3)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/c_models.json")
    parser.add_argument("--mode", choices=["smoke", "pilot", "formal"], default="smoke")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = effective_config(read_json(args.config), args.mode)
    config["runtime"]["device"] = args.device
    require(config["runtime"]["deterministic"] and config["mlp"]["residual_to_train_mean"],
            "This v1 runner implements deterministic, training-mean-residual MLPs")
    if args.device == "cuda":
        require(torch.cuda.is_available(), "CUDA unavailable; run outside the restricted sandbox or explicitly choose CPU")
    torch.set_num_threads(config["runtime"]["cpu_threads"])
    torch.set_num_interop_threads(1)
    base = ROOT / config["paths"]["results"]
    out = args.output or (base if args.mode == "formal" else base.with_name(base.name + "_" + args.mode))
    out = out.resolve()
    provenance = code_provenance()
    signature = {"mode": args.mode, "config": config, "source_tree_sha256": provenance["source_tree_sha256"]}
    if out.exists():
        require(args.resume, "Output exists; use --resume for a matching completed run or a new --output")
        require(read_json(out / "experiment.json")["signature"] == signature, "Cannot resume a different experiment")
    else:
        out.mkdir(parents=True)
        write_json(out / "experiment.json", {"signature": signature, "started_utc": utc_now(), "provenance": provenance,
                                             "environment": environment(), "python": sys.version,
                                             "device_name": torch.cuda.get_device_name(0) if args.device == "cuda" else "CPU",
                                             "command": "python " + shlex.join(sys.argv), "synthetic": False,
                                             "test_labels_accessed": False})
        for name in provenance["source_files"]:
            dest = out / "source_snapshot" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, dest)
    with threadpool_limits(limits=config["runtime"]["cpu_threads"]):
        for seed in config["seeds"]:
            spec, genes, labels, features, hashes, metadata = load_seed(config, seed)
            n_train, n_val = len(spec["targets"]["train"]), len(spec["targets"]["val"])
            indices = {"train": np.arange(n_train), "val": np.arange(n_train, n_train+n_val),
                       "test": np.arange(n_train+n_val, len(features["target_ids"]))}
            columns = np.arange(len(genes))
            if args.mode == "smoke":
                for part, good_count, missing_count in (("train", 40, 8), ("val", 12, 4)):
                    selected = indices[part]
                    mask = features["feature_available"][selected]
                    indices[part] = np.sort(np.r_[selected[mask][:good_count], selected[~mask][:missing_count]])
                columns = columns[:64]
            genes = np.asarray(genes, dtype=str)[columns]
            tr, va, te = (indices[p] for p in ("train", "val", "test"))
            y_train, y_val = labels["delta_true"][tr][:, columns], labels["delta_true"][va][:, columns]
            available = features["feature_available"]
            tasks = [("zero", "combined"), ("train_mean", "combined")]
            tasks += [(family, representation) for representation in config["representations"] for family in ("ridge", "mlp")]
            for family, representation in tasks:
                name = family if family in ("zero", "train_mean") else family + "_" + representation
                folder = out / name / f"seed_{seed}"
                if folder.exists():
                    old = read_json(folder / "run.json")
                    require(old["input_hashes"] == hashes and old["source_tree_sha256"] == provenance["source_tree_sha256"], "Completed run differs")
                    for file, value in old["artifact_hashes"].items():
                        require(sha256(folder / file) == value, f"Completed artifact changed: {file}")
                    print(f"REUSE {name}/seed_{seed}", flush=True)
                    continue
                started = time.monotonic()
                print(f"START {name}/seed_{seed}", flush=True)
                x = features[representation]
                if family in ("zero", "train_mean"):
                    model, trials = {"family": family, "train_mean": y_train.mean(axis=0)}, []
                else:
                    if family == "mlp" and args.device == "cuda":
                        torch.cuda.reset_peak_memory_stats()
                    model, trials = search(family, x[tr], y_train, available[tr], x[va], y_val, available[va], config, seed,
                                           args.device if family == "mlp" else "cpu")
                folder.mkdir(parents=True)
                save_model(folder, model)
                reloaded = load_model(folder)
                device = args.device if family == "mlp" else "cpu"
                pred_val = predict(model, x[va], available[va], device=device)
                pred_test = predict(model, x[te], available[te], device=device)
                reloaded_pred = predict(reloaded, x[va], available[va], device="cpu")
                reload_error = float(np.max(np.abs(pred_val-reloaded_pred)))
                require(np.allclose(pred_val, reloaded_pred, rtol=2e-5, atol=2e-6), "Checkpoint reload prediction mismatch")
                flags_val = ~available[va] if family not in ("zero", "train_mean") else np.zeros(len(va), dtype=bool)
                flags_test = ~available[te] if family not in ("zero", "train_mean") else np.zeros(len(te), dtype=bool)
                if flags_val.any():
                    require(np.array_equal(pred_val[flags_val], np.broadcast_to(model["train_mean"], pred_val[flags_val].shape)), "Fallback changed")
                save_predictions(folder / "validation_predictions.npz", features["target_ids"][va], genes, pred_val, available[va], flags_val)
                save_predictions(folder / "predictions.npz", features["target_ids"][te], genes, pred_test, available[te], flags_test)
                rows = evaluate(pred_val, y_val, model["train_mean"], features["target_ids"][va], genes,
                                available[va], flags_val, name, seed, "val", config["evaluation"]["top_k"])
                pd.DataFrame(rows).to_csv(folder / "validation_per_target.csv", index=False)
                summary = summarize(rows)
                pd.DataFrame(summary).to_csv(folder / "validation_summary.csv", index=False)
                write_json(folder / "search.json", trials)
                write_json(folder / "validation_evaluation.json", {"implementation": config["evaluation"]["implementation"], "summary": summary})
                artifacts = {p.name: sha256(p) for p in sorted(folder.iterdir()) if p.is_file()}
                run = {"schema_version": "0.1", "synthetic": False, "mode": args.mode, "seed": seed,
                       "model": name, "model_family": family, "representation": representation,
                       "config_snapshot": config, "input_hashes": hashes, "expression_metadata": metadata,
                       "git_commit": provenance["git_commit"], "source_tree_sha256": provenance["source_tree_sha256"],
                       "training_command": "python " + shlex.join(sys.argv), "environment": environment(),
                       "test_labels_accessed": False, "val_all_mse": macro_mse(pred_val, y_val),
                       "train_target_ids": features["target_ids"][tr].tolist(), "val_target_ids": features["target_ids"][va].tolist(),
                       "train_available_targets": int(available[tr].sum()), "n_output_genes": len(genes),
                       "reload_max_abs_error_cpu": reload_error, "selected_parameters": read_json(folder / "model.json"),
                       "device_used": device, "elapsed_seconds": time.monotonic()-started,
                       "cpu_max_rss_mib_process": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                       "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated()/2**20 if family == "mlp" and device == "cuda" else 0,
                       "completed_utc": utc_now(), "artifact_hashes": artifacts}
                write_json(folder / "run.json", run)
                print(f"DONE {name}/seed_{seed}: val_mse={run['val_all_mse']:.8g}, reload_error={reload_error:.3g}, {run['elapsed_seconds']:.2f}s", flush=True)
    write_json(out / "completed.json", {"completed_utc": utc_now(), "signature": signature, "test_labels_accessed": False})
    print(f"COMPLETE {args.mode}: {out.name}; no test labels loaded", flush=True)


if __name__ == "__main__":
    main()
