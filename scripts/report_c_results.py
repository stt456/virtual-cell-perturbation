"""Render the already frozen C diagnostic results; does not train or select models."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.c_common import read_json, require, sha256


def number(value):
    return "NA" if pd.isna(value) else f"{value:.6g}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=ROOT / "results/c_v1_20261106")
    args = parser.parse_args()
    base = args.experiment.resolve()
    frozen = read_json(base / "freeze.json")
    evaluation = read_json(base / "evaluation/evaluation.json")
    require(evaluation["freeze_sha256"] == sha256(base / "freeze.json"), "Freeze changed after evaluation")
    for name, expected in evaluation["files"].items():
        require(sha256(base / "evaluation" / name) == expected, f"Evaluation file changed: {name}")
    summary = pd.read_csv(base / "evaluation/summary_across_seeds.csv")
    by_seed = pd.read_csv(base / "evaluation/summary_by_seed.csv")
    experiment = read_json(base / "experiment.json")
    names = ["zero", "train_mean", "ridge_pca", "ridge_mean_detection", "ridge_combined",
             "mlp_pca", "mlp_mean_detection", "mlp_combined"]
    test = summary[summary.split == "test"].set_index(["model", "subset"])
    rows = []
    for name in names:
        all_row, mapped = test.loc[(name, "all")], test.loc[(name, "available")]
        baseline, mapped_baseline = test.loc[("train_mean", "all"), "mse_mean"], test.loc[("train_mean", "available"), "mse_mean"]
        rows.append(f"| {name} | {number(all_row.mse_mean)} ± {number(all_row.mse_std)} | {number(mapped.mse_mean)} ± {number(mapped.mse_std)} | {(baseline-all_row.mse_mean)/baseline*100:.2f}% | {(mapped_baseline-mapped.mse_mean)/mapped_baseline*100:.2f}% | {number(mapped.residual_pearson_mean)} |")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    labels = [name.replace("mean_detection", "mean+det").replace("_", "\n") for name in names]
    colors = ["#929292" if name in ("zero", "train_mean") else "#2878b5" if name.startswith("ridge") else "#db8a2c" for name in names]
    for ax, subset, title in zip(axes, ("all", "available"), ("All test targets", "Targets with mapped input features")):
        values = np.array([test.loc[(name, subset), "mse_mean"] for name in names])
        errors = np.array([test.loc[(name, subset), "mse_std"] for name in names])
        ax.bar(np.arange(len(names)), values, yerr=errors, capsize=3, color=colors)
        ax.set_xticks(np.arange(len(names)), labels, fontsize=8)
        ax.set_ylabel("Macro MSE (lower is better)")
        ax.set_title(title)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle("C v1: frozen test predictions, mean ± sample SD across 3 seeds")
    (ROOT / "figures").mkdir(exist_ok=True)
    fig.savefig(ROOT / "figures/c_v1_mse.png", dpi=180)
    fig.savefig(ROOT / "figures/c_v1_mse.pdf")
    plt.close(fig)
    selections = []
    hyperparameters, timings = [], []
    for seed in frozen["seeds"]:
        learner = frozen["validation_selected_learner_by_seed"][str(seed)]
        overall = frozen["validation_selected_method_including_baselines_by_seed"][str(seed)]
        selections.append(f"| {seed} | {learner} | {overall} |")
        for name in names:
            run = read_json(base / name / f"seed_{seed}/run.json")
            timings.append(run)
            if name in ("zero", "train_mean"):
                continue
            params = run["selected_parameters"]
            detail = f"alpha={params['alpha']:g}" if params["family"] == "ridge" else f"hidden={params['hidden_dim']}, lr={params['learning_rate']:g}, epoch={params['best_epoch']}"
            hyperparameters.append(f"| {seed} | {name} | {detail} | {run['val_all_mse']:.6g} | {run['elapsed_seconds']:.2f} |")
    gpu_peak = max(run["gpu_peak_allocated_mib"] for run in timings)
    total_seconds = sum(run["elapsed_seconds"] for run in timings)
    validation_rows = []
    for name in names:
        row = summary[(summary.model == name) & (summary.split == "val") & (summary.subset == "all")].iloc[0]
        validation_rows.append(f"| {name} | {number(row.mse_mean)} ± {number(row.mse_std)} |")
    relative = base.relative_to(ROOT).as_posix()
    content = "\n".join([
        "# C 主模型实验结果", "", "状态：C 的特征、Ridge/MLP、三种表示消融、三种子训练、冻结预测和诊断评估已经完成。以下为本项目实际运行结果；尚未包含 A 的参考方法或 D 的独立验收。", "",
        f"实验目录：`{relative}`。冻结时间：`{frozen['frozen_utc']}`；独立评估进程完成时间：`{evaluation['completed_utc']}`。", "",
        "## 数据和评价对象", "",
        "沿用已核验的 GEARS 5000 基因版本与 B v0.1 接口，目标级划分种子为 42、123、2026。每个 seed 有 760/162/164 个 train/val/test 目标；可唯一映射到输入列的训练目标为 284/279/302，测试目标为 66/73/59。", "",
        "PCA 仅使用各 seed 的 7483 个训练对照，按基因中心化、不按方差缩放、不 whitening；取 `components_.T` 作为 50 维基因表示，追加源 X 均值与正值比例。标准化仅用可映射的训练目标拟合。缺少输入列的目标统一使用所有训练目标的等权平均 delta 回退。", "",
        "## 冻结测试结果", "", "以下均为三个种子的算术平均 ± 样本标准差。MSE 降低率相对于训练均值基线，正数表示误差下降，负数表示误差上升。残差相关是减去训练均值后的 Pearson，不是 DRF。", "",
        "| 方法 | 全部目标 MSE | 可映射目标 MSE | 全部 MSE 降低率 | 可映射 MSE 降低率 | 可映射残差 Pearson |",
        "|---|---:|---:|---:|---:|---:|", *rows, "",
        "![三种子测试误差](../figures/c_v1_mse.png)", "",
        "误差条是三个随机划分间的样本标准差，不是目标 bootstrap 置信区间。逐目标配对 bootstrap 另存证据目录，按每个种子独立重采样，不能把重叠划分当作互相独立的所有样本。", "",
        "## 验证结果与选择规则", "", "所有超参数只按该 seed 的验证目标平均 MSE 选择。不同 seed 的验证标签不会替另一个 seed 选择模型。以下逐 seed 的方法选择在测试标签载入前已写入 freeze.json。", "",
        "| seed | 仅在学习模型中选择 | 包含基线时选择 |", "|---|---|---|", *selections, "",
        "下表仅描述各方法的验证成绩，不用于跨种子重新选择某个种子的模型：", "",
        "| 方法 | 全部验证目标 MSE |", "|---|---:|", *validation_rows, "",
        "## 参数与资源", "",
        "正式搜索共 72 个候选拟合：3 种表示 × 3 个种子 ×（4 个 Ridge alpha + 4 个 MLP 组合）。另有 6 个零变化/训练均值基线运行；冒烟和 pilot 独立保存，不混入正式表格。", "",
        "| seed | 模型 | 验证选出的参数 | 验证 MSE | 搜索及交付耗时/s |", "|---|---|---|---:|---:|", *hyperparameters, "",
        f"运行设备：`{experiment['device_name']}`；Ridge/PCA 使用 CPU，MLP 使用本地 CUDA。各正式任务记录的耗时合计 {total_seconds:.2f} 秒（不含数据导出、PCA、测试评估和解释器启动），MLP 最大 PyTorch allocated 显存约 {gpu_peak:.2f} MiB，不代表整张卡的总显存使用量。", "",
        "项目 `.venv` 负责 AnnData 数据准备；现有 `cs224n` 环境负责 PCA、模型及评估。运行元数据保存 Python/依赖版本、输入哈希、代码提交和源码树哈希；未修改其他 Conda 环境。", "",
        "## 验证和边界", "",
        "13 项单元/接口检查通过，包括手算指标、PCA 载荷定义、模拟信号学习、训练标准化范围、缺失回退、测试标签读取隔离、预测顺序和冻结文件篡改拒绝。真实数据先做冒烟，再做 seed 42 pilot，随后执行完整搜索。所有正式模型都检查了 CPU 重载预测与运行时预测的一致性。", "",
        "测试预测在读取 D_test.npz 前已生成并冻结；评估由单独进程完成。早前数据审计物理持有过全量文件，因此应称为训练/评估流程隔离，不是第三方严格盲测。", "",
        "当前上游只有 5000 个测量基因，约 62% 目标没有输入列。应同时看全部目标和可映射子集，不能把回退部分误解成模型学会了这些操作。归一化来源、上游基因选择范围、guide/batch 字段仍未解决；不能宣称全流程排除了上游选择信息。", "",
        "PCA/MLP 是否更好应按实际结果判断；本轮没有根据测试分数追加搜索。总体 MSE、残差相关和 Top-20/方向指标可能给出不同排序，完整指标及其有效目标数均已保留。", "",
        "## 交付与复现", "",
        f"- [逐目标成绩](../{relative}/evaluation/per_target.csv)",
        f"- [逐种子完整指标](../{relative}/evaluation/summary_by_seed.csv)",
        f"- [三种子汇总](../{relative}/evaluation/summary_across_seeds.csv)",
        f"- [配对 bootstrap](../{relative}/evaluation/paired_bootstrap.csv)",
        f"- [冻结清单](../{relative}/freeze.json)",
        f"- [评估元数据](../{relative}/evaluation/evaluation.json)",
        "- [C 输入输出与运行说明](../docs/c_models.md)", "",
        "每个方法的 `seed_<seed>/predictions.npz` 按 v0.1 交给 D；旁边保存 `run.json`、模型、标准化参数、验证结果和搜索历史。大型数据、权重和完整运行目录按现有 .gitignore 留在本地；小型汇总 CSV 与图可随 C 代码交付。GitHub 远程仓库未修改。", ""
    ])
    (ROOT / "reports/c_results.md").write_text(content, encoding="utf-8")
    summary.to_csv(ROOT / "reports/c_results_summary.csv", index=False)
    by_seed.to_csv(ROOT / "reports/c_results_by_seed.csv", index=False)
    print("WROTE reports/c_results.md, two summary CSVs, and figures/c_v1_mse.png/.pdf", flush=True)


if __name__ == "__main__":
    main()
