from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd

ORDER = ["zero", "train_mean", "ridge_pca", "ridge_mean_detection", "ridge_combined",
         "mlp_pca", "mlp_mean_detection", "mlp_combined"]
LABELS = {"zero": "无变化", "train_mean": "训练平均扰动", "ridge_pca": "Ridge PCA",
          "ridge_mean_detection": "Ridge 均值与检出率", "ridge_combined": "Ridge 组合",
          "mlp_pca": "MLP PCA", "mlp_mean_detection": "MLP 均值与检出率", "mlp_combined": "MLP 组合"}


def make_figures(output, frame, summary, across, bootstrap, coverage, origin):
    destination = output / "figures"
    destination.mkdir()
    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.exists():
        plt.rcParams["font.family"] = FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"font.size": 10, "axes.unicode_minus": False, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none", "figure.dpi": 120})
    caption = ("来源：D 独立重算 B 标签" if origin.startswith("D_") else
               "来源：C 自检逐目标记录，D 统计复算；未完成独立标签验收")
    caption += f"  |  C 冻结 2026-11-06，D 整理 {date.today().isoformat()}"

    def save(fig, name):
        fig.text(.02, .012, caption, fontsize=8, color="#59616d")
        fig.savefig(destination / f"{name}.png", dpi=200, facecolor="white")
        fig.savefig(destination / f"{name}.svg", facecolor="white")
        plt.close(fig)

    methods = [m for m in ORDER if m in frame.model.unique()]
    methods += [m for m in sorted(frame.model.unique()) if m not in methods]
    all_test = across[(across.split == "test") & (across.subset == "all")].set_index("model").loc[methods]
    y = np.arange(len(methods))
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.3), gridspec_kw={"width_ratios": [1.1, 1]})
    fig.subplots_adjust(left=.19, right=.98, top=.87, bottom=.31, wspace=.55)
    axes[0].barh(y, all_test.mse_mean, xerr=all_test.mse_std, color="#527b96", capsize=3)
    axes[0].set_yticks(y, [LABELS.get(m, m) for m in methods]); axes[0].invert_yaxis()
    axes[0].set_xlabel("逐目标 MSE 的三种子均值（越低越好）\n源 X 表达单位²")
    axes[0].set_ylabel("模型与表示")
    axes[0].set_title("表达变化误差  均值 ± 种子间样本标准差")
    axes[0].axvline(all_test.loc["train_mean", "mse_mean"], color="#444444", ls="--", lw=1,
                    label="训练平均扰动"); axes[0].legend(fontsize=8)
    valid = all_test.residual_pearson_mean.notna().to_numpy()
    axes[1].barh(y[valid], all_test.residual_pearson_mean[valid],
                 xerr=all_test.residual_pearson_std[valid], color="#527b96", capsize=3)
    axes[1].set_yticks(y, [LABELS.get(m, m) for m in methods]); axes[1].invert_yaxis()
    axes[1].set_xlabel("去训练平均扰动后的 Pearson r（无量纲）")
    axes[1].set_ylabel("模型与表示")
    axes[1].set_title("扰动特异性  仅在相关系数有效的目标上取均值")
    fig.suptitle("统一模型比较  每个种子 164 个测试目标和 2000 个输出基因", fontsize=14)
    fig.text(.19, .095, "训练平均扰动的残差相关系数不可定义；可学习模型的有效目标数为 66 / 73 / 59。\n"
             "无变化基线减去训练均值后也可产生相关系数，该值不表示识别了目标差异。", fontsize=8)
    save(fig, "01_model_comparison")

    ridge = [m for m in methods if m.startswith("ridge_")]
    fig, axes = plt.subplots(1, len(ridge), figsize=(12, 5.2), squeeze=False)
    fig.subplots_adjust(left=.08, right=.97, bottom=.34, top=.78, wspace=.4)
    for ax, model in zip(axes[0], ridge):
        rows = bootstrap[(bootstrap.split == "test") & (bootstrap.subset == "all") &
                         (bootstrap.model == model)].sort_values("seed")
        mean = rows.mean_mse_improvement_over_train_mean.to_numpy()
        for i, (_, row) in enumerate(rows.iterrows()):
            ax.plot([row.ci95_low, row.ci95_high], [i, i], color="#527b96", lw=2)
            ax.scatter(row.mean_mse_improvement_over_train_mean, i, color="#527b96", s=35)
        ax.axvline(0, color="#444444", ls="--", lw=1)
        ax.set_yticks(range(len(rows)), rows.seed.astype(str)); ax.invert_yaxis()
        ax.set_title(LABELS[model]); ax.set_ylabel("随机划分种子")
        ax.set_xlabel("平均 MSE 改善（基线 − 模型）\n源 X 表达单位²，右侧更好")
        ax.ticklabel_format(style="sci", axis="x", scilimits=(-3, -3))
    fig.suptitle("Ridge 相对平均扰动的配对目标 bootstrap 95% 区间", fontsize=14)
    fig.text(.08, .09, f"每个种子单独重采样 164 个目标，{int(bootstrap.bootstrap_repeats.iloc[0]):,} 次；点为平均差，线为百分位区间。\n"
             "不同种子的测试目标有重叠，未将 492 条目标记录当作独立样本合并推断。", fontsize=9)
    save(fig, "02_ridge_paired_bootstrap")

    fig, ax = plt.subplots(figsize=(8.8, 4.9))
    fig.subplots_adjust(left=.12, right=.97, bottom=.22, top=.82)
    x = np.arange(len(coverage))
    ax.bar(x, coverage.n_available, color="#527b96", label="表示可用")
    ax.bar(x, coverage.n_unavailable, bottom=coverage.n_available, color="#cbd0d4", label="表示缺失，模型回退训练均值")
    for i, row in enumerate(coverage.itertuples()):
        ax.text(i, row.n_available/2, f"{row.n_available}\n{row.coverage_pct:.1f}%", ha="center", va="center", color="white")
        ax.text(i, row.n_available+row.n_unavailable/2, str(row.n_unavailable), ha="center", va="center")
    ax.set_xticks(x, coverage.seed.astype(str)); ax.set_xlabel("随机划分种子")
    ax.set_ylabel("测试扰动目标数（个）"); ax.set_ylim(0, 195)
    ax.set_title("目标基因表示覆盖率  三个种子各 164 个测试目标", fontsize=13)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.17), ncol=2, fontsize=8)
    save(fig, "03_feature_coverage")

    strength = summary[(summary.split == "test") & summary.subset.str.startswith("strength_")]
    reference = strength[strength.model == "train_mean"][["seed", "subset", "mse"]].rename(columns={"mse": "baseline_mse"})
    joined = strength.merge(reference, on=["seed", "subset"], validate="many_to_one")
    joined["relative_mse_improvement_pct"] = 100*(joined.baseline_mse-joined.mse)/joined.baseline_mse
    joined.to_csv(output / "strength_comparison.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(9, 5.2))
    fig.subplots_adjust(left=.12, right=.98, bottom=.24, top=.84)
    categories = ["strength_weak", "strength_medium", "strength_strong"]
    selected_models = ["ridge_mean_detection", "ridge_combined", "mlp_mean_detection"]
    for offset, model in enumerate(selected_models):
        group = joined[joined.model == model].groupby("subset").relative_mse_improvement_pct
        ax.errorbar(np.arange(3)+(offset-1)*.07, group.mean().reindex(categories),
                    yerr=group.std(ddof=1).reindex(categories), marker=["o", "s", "^"][offset],
                    color=["#315b78", "#738b9a", "#636363"][offset], capsize=3, label=LABELS[model])
    ax.axhline(0, color="#555555", lw=1, ls="--")
    ax.set_xticks(range(3), ["弱响应", "中等响应", "强响应"])
    ax.set_xlabel("真实变化 RMS 三分位分层（种子内定义，事后描述）")
    ax.set_ylabel("相对训练平均扰动的 MSE 改善（%）")
    ax.set_title("真实响应强度分层  均值 ± 种子间样本标准差", fontsize=13)
    ax.legend(fontsize=8, loc="best")
    fig.text(.12, .095, "响应强度 = sqrt(mean(真实变化²))；分层阈值见 strength_thresholds.csv。\n"
             "handoff 模式下由 C 无变化基线的逐目标 MSE 推得，不是从缺失标签重建表达矩阵。", fontsize=8)
    save(fig, "04_response_strength")

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.9), sharey=True)
    fig.subplots_adjust(left=.1, right=.97, bottom=.24, top=.82, wspace=.22)
    for ax, subset in zip(axes, ("all", "available")):
        for family, color, marker in (("ridge", "#315b78", "o"), ("mlp", "#747474", "s")):
            group = across[(across.split == "test") & (across.subset == subset)].set_index("model")
            names = [f"{family}_{rep}" for rep in ("pca", "mean_detection", "combined")]
            values = group.loc[names]
            ax.errorbar(range(3), values.mse_mean, yerr=values.mse_std, marker=marker,
                        capsize=3, color=color, label=family.upper())
        baseline = across[(across.split == "test") & (across.subset == subset) & (across.model == "train_mean")].iloc[0]
        ax.axhline(baseline.mse_mean, color="#555555", ls="--", label="训练平均扰动")
        ax.set_xticks(range(3), ["PCA", "均值与检出率", "组合"])
        ax.set_xlabel("目标基因表示"); ax.set_ylabel("逐目标 MSE（源 X 表达单位²，越低越好）")
        ax.set_title("全部目标（164/种子）" if subset == "all" else "表示可用目标（66/73/59）")
    axes[0].legend(fontsize=8)
    fig.suptitle("表示消融比较  均值 ± 三种子间样本标准差", fontsize=14)
    save(fig, "05_representation_ablation")
