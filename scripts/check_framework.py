"""Validate the project framework and explicitly synthetic interface examples."""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_split(data):
    for field in ("targets", "control_cell_ids"):
        values = data[field]
        seen = set()
        for split in SPLITS:
            items = values[split]
            require(isinstance(items, list) and bool(items), f"{field}/{split}: empty or invalid")
            require(all(isinstance(x, str) and x.strip() for x in items), f"{field}: invalid identifier")
            require(len(items) == len(set(items)), f"{field}/{split}: duplicate IDs")
            require(not (seen & set(items)), f"{field}: leakage across splits")
            seen.update(items)


def validate_prediction_csv(path, expected_targets, expected_genes):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    require(bool(rows), "prediction CSV is empty")
    require(rows[0] == ["target_gene", *expected_genes], "gene order mismatch")
    require(len(expected_genes) == len(set(expected_genes)), "duplicate gene IDs")
    require([r[0] for r in rows[1:] if r] == expected_targets, "target order mismatch")
    for row in rows[1:]:
        require(len(row) == len(expected_genes) + 1, "prediction shape mismatch")
        require(all(math.isfinite(float(x)) for x in row[1:]), "non-finite prediction")


def main():
    config = json.loads((ROOT / "configs/default.json").read_text(encoding="utf-8"))
    require(math.isclose(sum(config["split"][s] for s in SPLITS), 1), "split ratios must sum to one")
    require(all(0 < config["split"][s] < 1 for s in SPLITS), "invalid split ratio")
    require(config["split"]["unit"] == "target_gene", "split must group target genes")
    require(config["seeds"] == [42, 123, 2026], "unexpected seeds")
    for directory in ("data/raw", "data/processed", "data/splits", "data/gene_sets", "src/data", "src/features", "src/models", "src/evaluation", "src/biology", "results", "figures", "reports", "tests"):
        require((ROOT / directory).is_dir(), f"missing directory: {directory}")
    split = json.loads((ROOT / "examples/synthetic_split.json").read_text(encoding="utf-8"))
    run = json.loads((ROOT / "examples/synthetic_run.json").read_text(encoding="utf-8"))
    require(split.get("synthetic") is True and run.get("synthetic") is True, "examples must be synthetic")
    require(split["seed"] == run["seed"], "example seed mismatch")
    validate_split(split)
    validate_prediction_csv(ROOT / "examples/synthetic_predictions.csv", split["targets"]["test"], run["gene_ids"])
    print("PASS: configuration, directory structure, split isolation and synthetic prediction interface")
    print("Real data and biological results have not been validated.")


if __name__ == "__main__":
    main()
