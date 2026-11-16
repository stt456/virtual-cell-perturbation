"""Run the local K562 workflow in order and summarize A's validation results."""

import argparse
import hashlib
import json
from io import BytesIO
from pathlib import Path
import shutil
import struct
import subprocess
import sys
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
SOURCE_MEMBER = "replogle_k562_essential/perturb_processed.h5ad"
SOURCE_SHA256 = "550fde89ac85c258c9fd517638fe34fb42c0e748c9305fe6355761a2bb754170"
ARCHIVE_SHA256 = "e769c89aa876061fdb3ac02a8de274f19a741a4f95decf14a52fd621d1eea319"
FREEZE = ROOT / "results/c_v1_20261106/freeze.json"
INSTALLED = ROOT / "results/c_data_verification_20261106/installed_data_manifest.json"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def launch(python, script, *arguments):
    command = [str(python), "-X", "utf8", script, *map(str, arguments)]
    print(f"\n> {subprocess.list2cmdline(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def normalize_frozen_file(path, expected):
    if path.suffix not in {".json", ".txt", ".csv", ".npz"} or digest(path) == expected:
        return False
    content = path.read_bytes()
    if path.suffix == ".npz":
        # NumPy uses ZIP. Windows and Linux mark the writer OS differently.
        normalized = bytearray(content)
        with ZipFile(BytesIO(content)) as archive:
            position = archive.start_dir
            for item in archive.infolist():
                if item.create_system == 0:
                    normalized[position + 5] = 3
                lengths = struct.unpack_from("<HHH", normalized, position + 28)
                position += 46 + sum(lengths)
    else:
        normalized = content.replace(b"\r\n", b"\n")
    if hashlib.sha256(normalized).hexdigest() == expected:
        path.write_bytes(normalized)
        print(f"Matched frozen file after format normalization: {path.relative_to(ROOT)}",
              flush=True)
        return True
    return False


def prepare_data(python):
    data = ROOT / "data"
    archive = data / "replogle_k562_essential.zip"
    source = data / SOURCE_MEMBER
    if not source.is_file():
        if archive.is_file():
            if digest(archive) != ARCHIVE_SHA256:
                raise ValueError(f"K562 ZIP hash differs: {archive}")
            with ZipFile(archive) as package:
                package.extract(SOURCE_MEMBER, data)
        else:
            launch(python, "src/download_b.py")
    if digest(source) != SOURCE_SHA256:
        raise ValueError(f"K562 source hash differs: {source}")

    for name in ("process_b", "report_b", "audit_b", "export_repo_interface",
                 "separate_labels", "validate_repo_export"):
        launch(python, f"src/{name}.py")

    exported = ROOT / "仓库接口交付_v0.1/data"
    installed = json.loads(INSTALLED.read_text(encoding="utf-8"))
    for key, record in installed.items():
        if not key.startswith(("data/processed/", "data/splits/")):
            continue
        original = exported / Path(key).relative_to("data")
        normalize_frozen_file(original, record["sha256"])
        target = ROOT / key
        if target.exists():
            normalize_frozen_file(target, record["sha256"])
        if target.exists() and digest(target) != record["sha256"]:
            if digest(target) != digest(original):
                raise ValueError(f"Existing local file differs from the new export: {target}")
        elif target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)


def prepare_features(data_python, feature_python):
    installed = json.loads(INSTALLED.read_text(encoding="utf-8"))
    for key, record in installed.items():
        path = ROOT / key
        if path.is_file():
            normalize_frozen_file(path, record["sha256"])
    launch(data_python, "scripts/prepare_c_inputs.py", "--verify-local-expression")
    launch(feature_python, "scripts/build_c_features.py")
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))["external_input_hashes"]
    for key, expected in frozen.items():
        if key.endswith("/features.npz") and normalize_frozen_file(ROOT / key, expected):
            metadata_path = (ROOT / key).with_name("features.json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["features_sha256_before_normalization"] = metadata["features_sha256"]
            metadata["features_sha256"] = expected
            metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                                     encoding="utf-8")


def summarize(output, freeze):
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    seeds = list(manifest["seeds"])
    if manifest["synthetic"] or seeds != ["42", "123", "2026"]:
        raise ValueError("Expected a formal three-seed result")
    rows = ["| 种子 | 最优 k | 验证集 MSE | 各候选 k 的验证集 MSE |",
            "| --- | ---: | ---: | --- |"]
    for seed in seeds:
        result = json.loads((output / f"seed_{seed}/run.json").read_text(encoding="utf-8"))
        k = result["selected_k"]
        errors = result["validation_macro_mse"]
        candidates = "；".join(f"k={v}：{errors[str(v)]:.8g}" for v in sorted(map(int, errors)))
        rows.append(f"| {seed} | {k} | {errors[str(k)]:.8g} | {candidates} |")
    frozen = json.loads(freeze.read_text(encoding="utf-8"))
    rebuilt = frozen.get("mode") == "locally_rebuilt_c_features"
    source_note = (
        "三套划分、输出基因与训练及验证标签均匹配 C 的原始哈希。"
        "基因特征在本地按 C 的代码和配置重建；对照原 Ridge 验证预测的最大绝对误差为 "
        f"{max(v['ridge_validation_max_abs_difference'] for v in frozen['verification'].values()):.3g}。"
        if rebuilt else "四类输入文件均匹配 C 的原始冻结记录。"
    )
    report = [
        "# A 的 kNN 正式验证结果",
        "",
        "在 C 的冻结数据划分上运行余弦加权 kNN，以验证集目标等权 MSE 选择 k。",
        "",
        source_note,
        "",
        *rows,
        "",
        f"K562 源数据 SHA256：`{SOURCE_SHA256}`",
        "",
        f"本次输入清单 SHA256：`{digest(freeze)}`",
        "",
        f"运行命令：`python -X utf8 scripts/run_a_knn.py --freeze \"{freeze}\" --output \"{output}\"`",
        "",
        "本次输入清单随附于 input_freeze.json。各预测文件索引见 manifest.json。",
        "",
    ]
    (output / "validation_summary.md").write_text("\n".join(report), encoding="utf-8")
    shutil.copyfile(freeze, output / "input_freeze.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("data", "features", "validate", "all"), default="all")
    parser.add_argument("--data-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--feature-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output", type=Path, default=Path("results/a_knn_v1"))
    parser.add_argument("--freeze", type=Path, default=FREEZE)
    args = parser.parse_args()
    output = ROOT / args.output
    freeze = ROOT / args.freeze
    try:
        if args.stage in ("data", "all"):
            prepare_data(args.data_python)
        if args.stage in ("features", "all"):
            prepare_features(args.data_python, args.feature_python)
        if args.stage in ("validate", "all"):
            launch(args.feature_python, "scripts/run_a_knn.py", "--freeze", freeze, "--output", output)
            summarize(output, freeze)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Stopped: {error}\n")


if __name__ == "__main__":
    main()
