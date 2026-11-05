import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("checks", ROOT / "scripts/check_framework.py")
checks = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checks)


class InterfaceChecks(unittest.TestCase):
    def setUp(self):
        self.split = json.loads((ROOT / "examples/synthetic_split.json").read_text(encoding="utf-8"))

    def test_valid_split(self):
        checks.validate_split(self.split)

    def test_target_leakage_rejected(self):
        bad = copy.deepcopy(self.split)
        bad["targets"]["test"].append(bad["targets"]["train"][0])
        with self.assertRaisesRegex(ValueError, "leakage"):
            checks.validate_split(bad)

    def test_control_leakage_rejected(self):
        bad = copy.deepcopy(self.split)
        bad["control_cell_ids"]["val"] = bad["control_cell_ids"]["train"]
        with self.assertRaisesRegex(ValueError, "leakage"):
            checks.validate_split(bad)

    def test_gene_order_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "gene order"):
            checks.validate_prediction_csv(ROOT / "examples/synthetic_predictions.csv", self.split["targets"]["test"], ["MOCK_GENE_2", "MOCK_GENE_1", "MOCK_GENE_3"])

    def test_nonfinite_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            path = Path(folder) / "bad.csv"
            path.write_text("target_gene,G1\nT1,nan\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-finite"):
                checks.validate_prediction_csv(path, ["T1"], ["G1"])


if __name__ == "__main__":
    unittest.main()
