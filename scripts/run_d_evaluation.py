"""Run D's unified evaluation on the frozen C handoff or independent B labels."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.pipeline import run

HANDOFF_EXPERIMENT = ROOT.parent / "results/c_v1_20261106"
if not HANDOFF_EXPERIMENT.is_dir():
    HANDOFF_EXPERIMENT = ROOT / "results/c_v1_20261106"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("handoff", "independent"), default="handoff")
    parser.add_argument("--experiment", type=Path, default=HANDOFF_EXPERIMENT)
    parser.add_argument("--output", type=Path, default=ROOT / "results/d_v1_20261110")
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--a-manifest", type=Path, help="JSON reference prediction paths and hashes")
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    args = parser.parse_args()
    try:
        result = run(args)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        parser.exit(2, f"D evaluation stopped: {exc}\n")
    print(result)


if __name__ == "__main__":
    main()
