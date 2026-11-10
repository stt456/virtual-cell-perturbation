from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import platform
import shutil
import sys

import numpy as np
import pandas as pd

from .io import (ids, load_npz, read_json, require, sha256, validate_labels,
                 validate_prediction, validate_split, write_json)
from .metrics import METRICS, baselines, evaluate, top_indices
from .statistics import across_seeds, add_strength_strata, bootstrap_table, summarize


def audit_handoff(base):
    """Verify available frozen evidence. B data provenance remains unverified here."""
    frozen = read_json(base / "freeze.json")
    experiment = read_json(base / "experiment.json")
    hashes, inventory, coverage, packages, means, runs = [], [], [], {}, {}, {}

    def check(path, expected, scope):
        require(path.is_file(), f"missing frozen file: {path}")
        actual = sha256(path)
        require(actual == expected, f"hash mismatch: {path}")
        hashes.append(dict(scope=scope, path=str(path.relative_to(base)), sha256=actual, passed=True))

    for relative, expected in frozen["artifact_hashes"].items():
        path = (base / relative).resolve()
        require(path.is_relative_to(base.resolve()), "unsafe frozen path")
        check(path, expected, "C_frozen_artifact")
    evaluation = read_json(base / "evaluation/evaluation.json")
    require(evaluation["freeze_sha256"] == sha256(base / "freeze.json"), "C evaluation freeze mismatch")
    for relative, expected in evaluation["files"].items():
        path = (base / "evaluation" / relative).resolve()
        require(path.is_relative_to((base / "evaluation").resolve()), "unsafe evaluation path")
        check(path, expected, "C_self_evaluation")
    source_files = experiment.get("provenance", {}).get("source_files", {})
    for relative, expected in source_files.items():
        check(base / "source_snapshot" / relative, expected, "C_training_source_snapshot")
    correction_path = base / "evaluation_correction.json"
    if correction_path.exists():
        correction = read_json(correction_path)
        require(correction["freeze_sha256"] == sha256(base / "freeze.json"), "correction freeze mismatch")
        for relative, change in correction["source_changes"].items():
            check(base / "evaluation_source_snapshot" / relative, change["corrected"], "C_evaluation_correction")

    for seed in frozen["seeds"]:
        for model in frozen["models"]:
            folder = base / model / f"seed_{seed}"
            run = read_json(folder / "run.json")
            require(run["seed"] == seed and run["model"] == model and run["mode"] == "formal",
                    "run identity mismatch")
            require(run["test_labels_accessed"] is False, "C run declares test-label use")
            require(run["source_tree_sha256"] == frozen["source_tree_sha256"], "training source mismatch")
            input_names = dict(input_ready=f"data/processed/c_inputs/seed_{seed}/ready.json",
                               split=f"data/processed/c_inputs/seed_{seed}/split.json",
                               training_labels=f"data/processed/c_inputs/seed_{seed}/labels_train_val.npz",
                               genes=f"data/processed/c_inputs/seed_{seed}/output_genes.txt",
                               features=f"data/processed/c_features/seed_{seed}/features.npz",
                               feature_metadata=f"data/processed/c_features/seed_{seed}/features.json")
            for field, relative in input_names.items():
                require(run["input_hashes"][field] == frozen["external_input_hashes"][relative],
                        "run/freeze declared input hash mismatch")
            runs[seed, model] = run
            mean = load_npz(folder / "model_arrays.npz")["train_mean"]
            require(np.isfinite(mean).all(), "nonfinite frozen training mean")
            if seed in means:
                require(np.array_equal(mean, means[seed]), "different frozen training means")
            means[seed] = mean
            for split, filename in (("test", "predictions.npz"), ("val", "validation_predictions.npz")):
                package = load_npz(folder / filename)
                targets = ids(package["target_ids"], "package targets").tolist()
                genes = ids(package["gene_ids"], "package genes").tolist()
                key = seed, split
                if split == "val":
                    require(genes == packages[seed, "test"]["gene_ids"].tolist(),
                            "test/validation output gene order mismatch")
                if key in packages:
                    reference = packages[key]
                    validate_prediction(package, reference["target_ids"], reference["gene_ids"])
                    require(np.array_equal(package["feature_available"], reference["feature_available"]),
                            "model coverage mask mismatch")
                else:
                    validate_prediction(package, targets, genes)
                    packages[key] = package
                require(mean.shape == (len(genes),), "training mean output dimension mismatch")
                sets = [set(run["train_target_ids"]), set(run["val_target_ids"]),
                        set(packages.get((seed, "test"), package)["target_ids"])]
                require(not any(sets[i] & sets[j] for i, j in ((0, 1), (0, 2), (1, 2))),
                        "train/val/test target leakage in run metadata")
                ids(np.asarray(run["train_target_ids"], dtype=str), "train targets")
                if split == "val":
                    require(targets == run["val_target_ids"], "validation metadata target order mismatch")
                values, fallback = package["delta_pred"], package["used_fallback"]
                if model == "zero":
                    require(not np.any(values) and not fallback.any(), "invalid zero baseline")
                elif model == "train_mean":
                    require(np.array_equal(values, np.broadcast_to(mean, values.shape)) and not fallback.any(),
                            "invalid submitted training-mean baseline")
                else:
                    require(np.array_equal(fallback, ~package["feature_available"]), "fallback/coverage mismatch")
                    require(np.array_equal(values[fallback], np.broadcast_to(mean, values[fallback].shape)),
                            "fallback is not exactly frozen training mean")
                inventory.append(dict(seed=seed, model=model, split=split, n_targets=len(targets),
                                      n_genes=len(genes), n_available=int(package["feature_available"].sum()),
                                      n_fallback=int(fallback.sum()), prediction_sha256=sha256(folder / filename)))
        test = packages[seed, "test"]
        coverage.append(dict(seed=seed, n_train=len(runs[seed, frozen["models"][0]]["train_target_ids"]),
                             n_val=len(packages[seed, "val"]["target_ids"]), n_test=len(test["target_ids"]),
                             n_output_genes=len(test["gene_ids"]), n_available=int(test["feature_available"].sum()),
                             n_unavailable=int((~test["feature_available"]).sum()),
                             coverage_pct=100*test["feature_available"].mean()))

    supplied = pd.read_csv(base / "evaluation/per_target.csv")
    require(not supplied.duplicated(["seed", "split", "model", "target_gene"]).any(), "duplicate C metric row")
    require(set(supplied.model) == set(frozen["models"]) and set(supplied.seed) == set(frozen["seeds"]),
            "C metric model/seed inventory mismatch")
    require(set(supplied.split) == {"test", "val"}, "unexpected C metric split")
    for row in inventory:
        current = supplied[(supplied.seed == row["seed"]) & (supplied.model == row["model"]) &
                           (supplied.split == row["split"])]
        p = load_npz(base / row["model"] / f"seed_{row['seed']}" /
                     ("predictions.npz" if row["split"] == "test" else "validation_predictions.npz"))
        require(current.target_gene.tolist() == p["target_ids"].tolist(), "C metrics/prediction target order mismatch")
        for flag in ("feature_available", "used_fallback"):
            require(current[flag].dtype == bool and np.array_equal(current[flag], p[flag]), "C metric flags mismatch")
    require(np.isfinite(supplied.mse).all() and (supplied.mse >= 0).all(), "invalid reported C MSE")
    recomputed = summarize(supplied)
    old = pd.read_csv(base / "evaluation/summary_by_seed.csv")
    keys = ["seed", "split", "model", "subset"]
    left, right = recomputed.set_index(keys).sort_index(), old.set_index(keys).sort_index()
    require(left.index.equals(right.index), "summary group mismatch")
    for column in ["n_targets", "n_fallback", *METRICS, *[f"{m}_n_valid" for m in METRICS]]:
        require(np.allclose(left[column], right[column], rtol=1e-11, atol=1e-14, equal_nan=True),
                f"C summary does not match target rows: {column}")
    left = across_seeds(recomputed).set_index(["model", "split", "subset"]).sort_index()
    right = pd.read_csv(base / "evaluation/summary_across_seeds.csv").set_index(["model", "split", "subset"]).sort_index()
    require(left.index.equals(right.index), "across-seed group mismatch")
    for m in METRICS:
        for suffix in ("mean", "std"):
            require(np.allclose(left[f"{m}_{suffix}"], right[f"{m}_{suffix}"], rtol=1e-11, atol=1e-14, equal_nan=True),
                    f"C across-seed summary mismatch: {m}")
    audit = dict(artifact_integrity_passed=True, n_hashes_checked=len(hashes),
                 prediction_contracts_passed=True, reported_metric_aggregation_passed=True,
                 b_label_hashes_verified=False, independent_metric_recalculation=False,
                 control_split_checked=False, train_mean_recomputed_from_labels=False,
                 note="Only submitted files and declared metadata are checked without B labels.")
    return frozen, supplied, audit, pd.DataFrame(hashes), pd.DataFrame(inventory), pd.DataFrame(coverage)


def score_independently(base, data_root, frozen, a_manifest=None):
    """Require exact frozen B versions; never substitute missing labels."""
    rows, evidence = [], []
    a = read_json(a_manifest) if a_manifest else None
    if a:
        require(a["model"] not in frozen["models"] and a["test_labels_accessed"] is False, "invalid A manifest")
        require(set(a["seeds"]) == {str(s) for s in frozen["seeds"]}, "A seed inventory mismatch")
    for seed in frozen["seeds"]:
        folder = data_root / f"data/processed/c_inputs/seed_{seed}"
        direct = data_root / f"data/processed/seed_{seed}"
        split_path = folder / "split.json" if (folder / "split.json").exists() else data_root / f"data/splits/seed_{seed}.json"
        labels_path = folder / "labels_train_val.npz" if (folder / "labels_train_val.npz").exists() else direct / "C_train_val.npz"
        test_path = direct / "D_test.npz"
        genes_path = folder / "output_genes.txt" if (folder / "output_genes.txt").exists() else direct / "output_genes.txt"
        checks = [(split_path, frozen["external_input_hashes"][f"data/processed/c_inputs/seed_{seed}/split.json"]),
                  (labels_path, frozen["external_input_hashes"][f"data/processed/c_inputs/seed_{seed}/labels_train_val.npz"]),
                  (genes_path, frozen["external_input_hashes"][f"data/processed/c_inputs/seed_{seed}/output_genes.txt"]),
                  (test_path, frozen["test_label_expected_sha256"][str(seed)])]
        for path, expected in checks:
            require(path.is_file(), f"B input missing: {path}; not scored")
            require(sha256(path) == expected, f"B version/hash mismatch: {path}")
            evidence.append(dict(path=str(path), sha256=expected))
        spec = read_json(split_path)
        require(spec["seed"] == seed, "B seed mismatch")
        validate_split(spec)
        genes = genes_path.read_text(encoding="utf-8").splitlines()
        trainval, test = load_npz(labels_path), load_npz(test_path)
        train_ids, val_ids, test_ids = [spec["targets"][s] for s in ("train", "val", "test")]
        validate_labels(trainval, train_ids+val_ids, genes, ["train"]*len(train_ids)+["val"]*len(val_ids))
        validate_labels(test, test_ids, genes, ["test"]*len(test_ids))
        mean, _, _ = baselines(trainval["delta_true"][:len(train_ids)], len(test_ids))
        canonical = load_npz(base / f"train_mean/seed_{seed}/model_arrays.npz")["train_mean"]
        require(np.allclose(mean, canonical, rtol=1e-12, atol=1e-14), "independently computed training mean mismatch")
        evidence.append(dict(seed=seed, train_mean_max_abs_difference=float(np.max(np.abs(mean-canonical)))))
        for split, targets, truth in (("val", val_ids, trainval["delta_true"][len(train_ids):]),
                                      ("test", test_ids, test["delta_true"])):
            ref = load_npz(base / f"train_mean/seed_{seed}" /
                           ("predictions.npz" if split == "test" else "validation_predictions.npz"))
            validate_prediction(ref, targets, genes)
            for model in frozen["models"]:
                p = load_npz(base / model / f"seed_{seed}" /
                             ("predictions.npz" if split == "test" else "validation_predictions.npz"))
                run = read_json(base / model / f"seed_{seed}/run.json")
                require(run["train_target_ids"] == train_ids and run["val_target_ids"] == val_ids,
                        "C run/B split mismatch")
                validate_prediction(p, targets, genes)
                values = (np.zeros_like(truth) if model == "zero" else np.tile(canonical, (len(targets), 1))
                          if model == "train_mean" else p["delta_pred"])
                if model in ("zero", "train_mean"):
                    require(np.array_equal(values, p["delta_pred"]), "independently checked baseline mismatch")
                rows.extend(evaluate(values, truth, canonical, p["target_ids"], p["gene_ids"],
                                     p["feature_available"], p["used_fallback"], model, seed, split))
            if a and split == "test":
                item = a["seeds"][str(seed)]
                path = (a_manifest.parent / item["predictions"]).resolve()
                require(sha256(path) == item["sha256"], "A prediction hash mismatch")
                p = load_npz(path)
                validate_prediction(p, targets, genes)
                require(np.array_equal(p["feature_available"], ref["feature_available"]), "A coverage mask mismatch")
                rows.extend(evaluate(p["delta_pred"], truth, canonical, p["target_ids"], p["gene_ids"],
                                     p["feature_available"], p["used_fallback"], a["model"], seed, split))
                evidence.append(dict(path=str(path), sha256=item["sha256"]))
    return pd.DataFrame(rows), evidence


def export_e(base, output, frame, frozen, independent, data_root):
    destination = output / "E_handoff"
    destination.mkdir()
    cases, top_genes, choices = [], [], {}
    for seed in frozen["seeds"]:
        model = frozen["validation_selected_learner_by_seed"][str(seed)]
        p = load_npz(base / f"{model}/seed_{seed}/predictions.npz")
        # Fixed, reproducible illustrative selection; upstream test summaries were already available.
        chosen = []
        for available in (True, False):
            candidates = p["target_ids"][p["feature_available"] == available].tolist()
            candidates.sort(key=lambda t: __import__('hashlib').sha256(f"D_E_v1|{seed}|{t}".encode()).hexdigest())
            chosen.extend(candidates[:3])
        choices[str(seed)] = dict(model=model, targets=chosen)
        idx = np.array([p["target_ids"].tolist().index(t) for t in chosen])
        arrays = {k: (v if k == "gene_ids" else v[idx]) for k, v in p.items()}
        arrays["train_mean"] = load_npz(base / f"{model}/seed_{seed}/model_arrays.npz")["train_mean"]
        if independent:
            labels = load_npz(data_root / f"data/processed/seed_{seed}/D_test.npz")
            arrays["delta_true"] = labels["delta_true"][idx]
        np.savez_compressed(destination / f"cases_seed_{seed}.npz", **arrays)
        for i, target in enumerate(chosen):
            current = frame[(frame.seed == seed) & (frame.split == "test") &
                            (frame.model == model) & (frame.target_gene == target)].iloc[0].to_dict()
            reference = frame[(frame.seed == seed) & (frame.split == "test") &
                              (frame.model == "train_mean") & (frame.target_gene == target)].iloc[0]
            current["mse_improvement_over_train_mean"] = reference.mse-current["mse"]
            current["truth_matrix_available"] = independent
            cases.append(current)
            for rank, j in enumerate(top_indices(arrays["delta_pred"][i], arrays["gene_ids"]), 1):
                top_genes.append(dict(seed=seed, model=model, target_gene=target, rank=rank,
                                      gene_id=arrays["gene_ids"][j], delta_pred=arrays["delta_pred"][i, j],
                                      delta_true=arrays["delta_true"][i, j] if independent else np.nan))
    pd.DataFrame(cases).to_csv(destination / "case_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(top_genes).to_csv(destination / "case_top20_predictions.csv", index=False, encoding="utf-8-sig")
    exploratory = []
    for seed in frozen["seeds"]:
        model = frozen["validation_selected_learner_by_seed"][str(seed)]
        current = frame[(frame.seed == seed) & (frame.split == "test") &
                        (frame.model == model) & frame.feature_available].copy()
        reference = frame[(frame.seed == seed) & (frame.split == "test") &
                          (frame.model == "train_mean")].set_index("target_gene")
        current["mse_improvement_over_train_mean"] = current.target_gene.map(reference.mse) - current.mse
        current = current.sort_values(["mse_improvement_over_train_mean", "target_gene"], ascending=[False, True])
        for label, selected in (("better", current.head(5)), ("worse", current.tail(5))):
            selected = selected.copy()
            selected["posthoc_case_group"] = label
            exploratory.append(selected)
    pd.concat(exploratory, ignore_index=True).to_csv(destination / "posthoc_best_worst.csv", index=False, encoding="utf-8-sig")
    write_json(destination / "case_selection.json", dict(selection="SHA256 D_E_v1|seed|target; 3 available + 3 unavailable per seed",
               selection_timing="post-hoc illustrative; not a pre-test preregistration", choices=choices,
               truth_matrices_available=independent, ground_truth_missing_value=None))


def run(args):
    base, output = args.experiment.resolve(), args.output.resolve()
    require(not output.exists(), f"output already exists; choose a new --output: {output}")
    frozen, supplied, audit, hashes, inventory, coverage = audit_handoff(base)
    independent = args.mode == "independent"
    if independent:
        frame, evidence = score_independently(base, args.data_root.resolve(), frozen, args.a_manifest)
        audit.update(b_label_hashes_verified=True, independent_metric_recalculation=True,
                     control_split_checked=True, train_mean_recomputed_from_labels=True)
        origin = "D_independent_metrics_from_B_labels"
    else:
        frame, evidence, origin = supplied.copy(), [], "C_self_evaluation_target_rows_reaggregated_by_D"
    frame, thresholds = add_strength_strata(frame)
    frame["evaluation_origin"] = origin
    summary = summarize(frame, stratified=True)
    across = across_seeds(summary)
    bootstrap = bootstrap_table(frame, args.bootstrap_repeats)
    selected = pd.concat([frame[(frame.seed == seed) & (frame.model == frozen["validation_selected_learner_by_seed"][str(seed)])]
                          for seed in frozen["seeds"]], ignore_index=True)
    selected["source_model"] = selected.model
    selected["model"] = "validation_selected_learner"
    selected_summary = summarize(selected)
    output.mkdir(parents=True)
    if independent:
        keys = ["seed", "split", "model", "target_gene"]
        checked = frame.merge(supplied, on=keys, suffixes=("_D", "_C"), validate="one_to_one")
        differences = checked[keys].copy()
        for metric in METRICS:
            differences[f"{metric}_D_minus_C"] = checked[f"{metric}_D"] - checked[f"{metric}_C"]
            differences[f"{metric}_validity_matches"] = checked[f"{metric}_D"].notna() == checked[f"{metric}_C"].notna()
        differences.to_csv(output / "independent_vs_C_metric_differences.csv", index=False, encoding="utf-8-sig")
    tables = {"hash_checks": hashes, "prediction_inventory": inventory, "coverage": coverage,
              "per_target": frame, "summary_by_seed": summary, "summary_across_seeds": across,
              "paired_bootstrap": bootstrap, "strength_thresholds": thresholds,
              "validation_selected_by_seed": selected_summary,
              "validation_selected_across_seeds": across_seeds(selected_summary)}
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False, encoding="utf-8-sig")
    write_json(output / "audit.json", audit)
    pending_inputs = dict(
        B_test_labels=None if not independent else "verified",
        B_training_labels=None if not independent else "verified",
        A_reference_predictions=None if not args.a_manifest else "provided",
        missing_means="not scored; never replaced by zero changes",
        independent_evaluation_complete=independent,
        full_planned_method_comparison_complete=independent and args.a_manifest is not None)
    write_json(output / "pending_inputs.json", pending_inputs)
    export_e(base, output, frame, frozen, independent, args.data_root.resolve())
    from .plots import make_figures
    make_figures(output, frame, summary, across, bootstrap, coverage, origin)
    from .report import write_reports
    write_reports(output, base, frozen, audit, frame, summary, across, bootstrap, coverage, origin)
    source_root = Path(__file__).resolve().parents[2]
    repository_artifacts = publish_repository_artifacts(output, source_root, across, bootstrap, coverage,
                                                        summary, audit, pending_inputs)
    manifest = dict(completed_utc=datetime.now(timezone.utc).isoformat(), mode=args.mode,
                    evaluation_origin=origin, freeze_sha256=sha256(base / "freeze.json"),
                    command=sys.argv, python=sys.version, platform=platform.platform(),
                    environment={k: importlib.metadata.version(k) for k in ("numpy", "pandas", "matplotlib")},
                    independent_input_evidence=evidence,
                    source_hashes={str(p.relative_to(source_root)): sha256(p)
                                   for p in sorted([*(source_root / "src/evaluation").glob("*.py"),
                                                    source_root / "scripts/run_d_evaluation.py",
                                                    source_root / "tests/test_d_evaluation.py",
                                                    source_root / "requirements-d.txt"])},
                    repository_artifacts=repository_artifacts,
                    files={str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()})
    write_json(output / "manifest.json", manifest)
    return f"D output: {output}; origin={origin}; independent={independent}"


def publish_repository_artifacts(output, source_root, across, bootstrap, coverage, summary, audit, pending_inputs):
    """Copy compact, shareable D outputs into the repository's tracked folders."""
    figures = source_root / "figures"
    reports = source_root / "reports"
    figures.mkdir(exist_ok=True)
    reports.mkdir(exist_ok=True)
    figure_names = (
        ("01_model_comparison", "d_01_model_comparison"),
        ("02_ridge_paired_bootstrap", "d_02_ridge_paired_bootstrap"),
        ("03_feature_coverage", "d_03_feature_coverage"),
        ("04_response_strength", "d_04_response_strength"),
        ("05_representation_ablation", "d_05_representation_ablation"),
    )
    published = []
    for source_name, target_name in figure_names:
        for extension in ("png", "svg"):
            target = figures / f"{target_name}.{extension}"
            shutil.copy2(output / "figures" / f"{source_name}.{extension}", target)
            published.append(target)

    report = (output / "D_统一评测与结果比较.md").read_text(encoding="utf-8")
    for source_name, target_name in figure_names:
        report = report.replace(f"figures/{source_name}.png", f"../figures/{target_name}.png")
    report_path = reports / "d_results.md"
    report_path.write_text(report, encoding="utf-8")
    html_path = reports / "d_results.html"
    shutil.copy2(output / "D_结果浏览.html", html_path)
    published.extend((report_path, html_path))

    across[(across.split == "test") & (across.subset == "all")].to_csv(
        reports / "d_summary.csv", index=False, encoding="utf-8-sig")
    across.to_csv(reports / "d_metrics_across_seeds.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(reports / "d_metrics_by_seed.csv", index=False, encoding="utf-8-sig")
    bootstrap[bootstrap.split == "test"].to_csv(
        reports / "d_paired_bootstrap.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(reports / "d_coverage.csv", index=False, encoding="utf-8-sig")
    write_json(reports / "d_scope.json", dict(audit=audit, pending_inputs=pending_inputs))
    published.extend(reports / name for name in (
        "d_summary.csv", "d_metrics_across_seeds.csv", "d_metrics_by_seed.csv",
        "d_paired_bootstrap.csv", "d_coverage.csv", "d_scope.json"))
    return {str(path.relative_to(source_root)): sha256(path) for path in published}
