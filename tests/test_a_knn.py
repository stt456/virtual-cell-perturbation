"""Exercise the A pipeline with explicitly synthetic three-seed C-style files."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.run_a_knn import FEATURES, INPUTS, run
from src.c_common import load_npz, read_json, sha256, write_json

ROOT = Path(__file__).resolve().parents[1]


def fixture(root):
    freeze = {"seeds": [42, 123, 2026], "external_input_hashes": {}}
    for seed in freeze["seeds"]:
        inp, feat = (root / base / f"seed_{seed}" for base in (INPUTS, FEATURES))
        inp.mkdir(parents=True)
        feat.mkdir(parents=True)
        targets = {"train": ["A", "B", "C", "M"], "val": ["V", "W"], "test": ["X", "Y"]}
        write_json(inp / "split.json", {
            "seed": seed, "targets": targets,
            "control_cell_ids": {"train": ["ctrl1"], "val": ["ctrl2"], "test": ["ctrl3"]},
        })
        (inp / "output_genes.txt").write_text("G1\nG2\n", encoding="utf-8")
        np.savez(inp / "labels_train_val.npz", target_ids=targets["train"] + targets["val"],
                 gene_ids=["G1", "G2"], split=["train"] * 4 + ["val"] * 2,
                 delta_true=[[2., 1.], [-1., 3.], [0., -2.], [3., 2.], [2., 1.], [1., 1.]])
        np.savez(feat / "features.npz", target_ids=sum(targets.values(), []),
                 fit_control_ids=["ctrl1"], feature_available=[True, True, True, False, True, False, True, False],
                 combined=[[1, 0], [-1, 1], [-1, -1], [np.nan, np.nan],
                           [1, 0], [np.nan, np.nan], [-1, 1], [np.nan, np.nan]])
        for path in (*inp.iterdir(), *feat.iterdir()):
            freeze["external_input_hashes"][path.relative_to(root).as_posix()] = sha256(path)
    return freeze


class KNNPrototypeTest(unittest.TestCase):
    def test_three_seed_predictions_and_version_guard(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as folder:
            root = Path(folder)
            freeze = fixture(root)
            output = root / "smoke"
            run(root, output, freeze, "combined", [1, 2, 3], synthetic=True)
            manifest = read_json(output / "smoke_manifest.json")
            self.assertTrue(manifest["synthetic"])
            self.assertFalse(manifest["test_labels_accessed"])
            self.assertEqual(set(manifest["seeds"]), {"42", "123", "2026"})
            for seed, item in manifest["seeds"].items():
                test_file = output / item["predictions"]
                val_file = output / item["validation_predictions"]
                self.assertEqual(sha256(test_file), item["sha256"])
                self.assertEqual(sha256(val_file), item["validation_sha256"])
                prediction, val = load_npz(test_file), load_npz(val_file)
                np.testing.assert_array_equal(prediction["target_ids"], ["X", "Y"])
                np.testing.assert_array_equal(prediction["gene_ids"], ["G1", "G2"])
                np.testing.assert_array_equal(prediction["used_fallback"], [False, True])
                np.testing.assert_allclose(prediction["delta_pred"], [[-1, 3], [1, 1]])
                np.testing.assert_allclose(val["delta_pred"], [[2, 1], [1, 1]])
                self.assertEqual(read_json(test_file.parent / "run.json")["selected_k"], 1)
            (root / INPUTS / "seed_42/output_genes.txt").write_text("G2\nG1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                run(root, root / "bad", freeze, "combined", [1], synthetic=True)
            self.assertFalse((root / "bad").exists())
