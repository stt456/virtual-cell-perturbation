"""Numerical and leakage checks with hand-computable/learnable synthetic data."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import csr_matrix
import torch

from src.c_common import load_npz, save_predictions, validate_split, validate_training_labels
from src.features.control_features import fit_control_features
from src.models.c_models import fit_ridge, fit_mlp, predict, save_model, load_model, macro_mse
from src.evaluation.c_metrics import check_prediction_package, evaluate, summarize
from scripts.evaluate_c_results import verify_files
from src.c_common import sha256, write_json
from scripts import run_c_models


class CChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_pca_gene_loadings_and_duplicate_mapping(self):
        controls = csr_matrix([[1., 2., 1., 5.], [2., 4., 1., 5.], [3., 6., 1., 5.], [4., 8., 1., 5.]])
        result = fit_control_features(controls, ["A", "B", "DUP", "DUP"], ["A", "B", "DUP", "UNKNOWN"], components=3)
        self.assertEqual(result["pca"].shape, (4, 1))
        np.testing.assert_allclose(abs(result["pca_components"][0, :2]), np.array([1, 2])/np.sqrt(5))
        np.testing.assert_allclose(result["mean_detection"][:2], [[2.5, 1], [5., 1]])
        self.assertEqual(result["feature_available"].tolist(), [True, True, False, False])

    def test_ridge_generalization_scaling_and_mean_fallback(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=(90, 4))
        weights = rng.normal(size=(4, 5))
        y = x @ weights + np.arange(5)
        available = np.ones(60, dtype=bool)
        available[-1] = False
        x[59] = 1e8
        model = fit_ridge(x[:60], y[:60], available, .001)
        np.testing.assert_allclose(model["scale_mean"], x[:59].mean(axis=0))
        mapped = np.ones(30, dtype=bool)
        prediction = predict(model, x[60:], mapped)
        self.assertLess(macro_mse(prediction, y[60:]), macro_mse(np.broadcast_to(y[:60].mean(axis=0), y[60:].shape), y[60:]) * .001)
        mapped[-1] = False
        np.testing.assert_array_equal(predict(model, x[60:], mapped)[-1], y[:60].mean(axis=0))
        with tempfile.TemporaryDirectory() as folder:
            save_model(folder, model)
            np.testing.assert_array_equal(predict(load_model(folder), x[60:], mapped), predict(model, x[60:], mapped))

    def test_mlp_learns_signal_and_reloads(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=(120, 3))
        y = np.column_stack((x[:, 0] + 2*x[:, 1], x[:, 2] - x[:, 0]))
        settings = {"hidden_dim": 16, "dropout": 0., "learning_rate": .01, "weight_decay": 0.,
                    "max_epochs": 100, "patience": 20, "batch_size": 32}
        model, history = fit_mlp(x[:90], y[:90], np.ones(90, bool), x[90:], y[90:], np.ones(30, bool), settings, 7, "cpu")
        prediction = predict(model, x[90:], np.ones(30, bool))
        self.assertLess(macro_mse(prediction, y[90:]), macro_mse(np.broadcast_to(y[:90].mean(axis=0), y[90:].shape), y[90:]) * .1)
        self.assertTrue(history)
        with tempfile.TemporaryDirectory() as folder:
            save_model(folder, model)
            np.testing.assert_array_equal(prediction, predict(load_model(folder), x[90:], np.ones(30, bool)))

    def test_test_labels_cannot_enter_training_contract(self):
        spec = {"targets": {"train": ["A"], "val": ["B"], "test": ["C"]},
                "control_cell_ids": {"train": ["C1"], "val": ["C2"], "test": ["C3"]}}
        validate_split(spec)
        labels = {"target_ids": np.array(["A", "B"]), "gene_ids": np.array(["G"]),
                  "split": np.array(["train", "val"]), "delta_true": np.ones((2, 1))}
        validate_training_labels(labels, spec, ["G"])
        bad = deepcopy(labels)
        bad["target_ids"][1] = "C"
        with self.assertRaisesRegex(ValueError, "contamination"):
            validate_training_labels(bad, spec, ["G"])
        spec["control_cell_ids"]["test"] = ["C1"]
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate_split(spec)

    def test_training_loader_is_isolated_from_external_test_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            inputs, features = root / "inputs/seed_42", root / "features/seed_42"
            inputs.mkdir(parents=True)
            features.mkdir(parents=True)
            spec = {"targets": {"train": ["A", "B"], "val": ["C"], "test": ["D"]},
                    "control_cell_ids": {"train": ["C1"], "val": ["C2"], "test": ["C3"]}}
            write_json(inputs / "split.json", spec)
            (inputs / "output_genes.txt").write_text("G1\nG2\n")
            write_json(inputs / "metadata.json", {"synthetic": True})
            np.savez(inputs / "labels_train_val.npz", target_ids=np.array(["A", "B", "C"]), gene_ids=np.array(["G1", "G2"]),
                     split=np.array(["train", "train", "val"]), delta_true=np.array([[1., 2.], [3., 4.], [5., 6.]]))
            files = {p.name: sha256(p) for p in inputs.iterdir()}
            write_json(inputs / "ready.json", {"files": files})
            np.savez(features / "features.npz", target_ids=np.array(["A", "B", "C", "D"]),
                     fit_control_ids=np.array(["C1"]), combined=np.array([[0., 1.], [1., 0.], [2., 3.], [4., 5.]]),
                     feature_available=np.ones(4, bool))
            write_json(features / "features.json", {"signature": {"input_ready_sha256": sha256(inputs / "ready.json")},
                                                     "features_sha256": sha256(features / "features.npz")})
            file = root / "D_test.npz"
            np.savez(file, delta_true=np.zeros((1, 2)))
            original_load = np.load
            permitted = {inputs / "labels_train_val.npz", features / "features.npz"}
            def guarded_load(path, *args, **kwargs):
                self.assertIn(Path(path), permitted)
                return original_load(path, *args, **kwargs)
            config = {"paths": {"inputs": "inputs", "features": "features"}}
            with patch.object(run_c_models, "ROOT", root), patch.object(np, "load", guarded_load):
                first = run_c_models.load_seed(config, 42)
                np.savez(file, delta_true=np.full((1, 2), 1e20))
                second = run_c_models.load_seed(config, 42)
            np.testing.assert_array_equal(first[2]["delta_true"], second[2]["delta_true"])
            np.testing.assert_array_equal(first[3]["combined"], second[3]["combined"])
            self.assertEqual(first[4], second[4])

    def test_metrics_hand_values_ties_and_zero_baseline(self):
        rows = evaluate(np.zeros((1, 3)), np.array([[1., -2., 0.]]), np.zeros(3), ["A"], ["B", "A", "C"],
                        [True], [False], "zero", 42, "val", top_k=2)
        row = rows[0]
        self.assertAlmostEqual(row["mse"], 5/3)
        self.assertEqual(row["top20_overlap"], 1.)
        self.assertTrue(np.isnan(row["pearson"]) and np.isnan(row["top20_direction"]))
        self.assertEqual(row["mse_without_self"], .5)
        self.assertEqual(summarize(rows)[0]["pearson_n_valid"], 0)

    def test_correlations_and_nonoverlapping_topk(self):
        rows = evaluate(np.array([[1., 2., 3.]]), np.array([[3., 2., 1.]]), np.zeros(3), ["X"], ["A", "B", "C"],
                        [True], [False], "ridge", 42, "val", top_k=1)
        self.assertAlmostEqual(rows[0]["pearson"], -1)
        self.assertAlmostEqual(rows[0]["spearman"], -1)
        self.assertEqual(rows[0]["top20_overlap"], 0)
        self.assertTrue(np.isnan(rows[0]["top20_direction"]))

    def test_prediction_order_and_frozen_tampering(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            save_predictions(folder / "pred.npz", ["A"], ["G1", "G2"], [[1, 2]], [True], [False])
            package = load_npz(folder / "pred.npz")
            with self.assertRaisesRegex(ValueError, "gene order"):
                check_prediction_package(package, ["A"], ["G2", "G1"])
            expected = {"pred.npz": sha256(folder / "pred.npz")}
            verify_files(folder, expected)
            with (folder / "pred.npz").open("ab") as handle:
                handle.write(b"changed")
            with self.assertRaisesRegex(ValueError, "Frozen file changed"):
                verify_files(folder, expected)


if __name__ == "__main__":
    unittest.main()
