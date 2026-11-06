# C 方法调研与选择依据

核验日期：2026-11-06。围绕本仓库的“训练未见单基因 CRISPRi 目标 -> 群体平均表达变化”任务整理。下文区分文献结论、本项目的设计选择和仍待验证的问题；不将论文成绩填入本项目结果。

数据状态更新：用户提供的源 ZIP 已实际解压并核验，411/1086 的输入目标覆盖得到确认，文件没有额外全基因表达层。详见 [本地真实数据核验报告](c_data_verification_20261106.md)。本方法笔记中的覆盖限制继续成立。

## 1. 一手来源与本项目采用的部分

| 来源 | 已核验内容 | 对 C 的影响 |
|---|---|---|
| Ahlmann-Eltze、Huber、Anders，Nature Methods 2025 | 在作者的任务、数据与指标设置中，所比较的深度方法没有稳定超过简单均值/线性基线；线性模型可使用基因和扰动表示 | 先建立训练均值和 Ridge，再检验 MLP 是否有增益；不预设深度模型更好 |
| Miller 等，Nature Biotechnology 2026 | 在 14 个数据集、18 种指标上讨论评价灵敏度，并用正负对照定义 DRF | 不根据一个总体 MSE 就推断是否学到了扰动特异性；同时报告相关、方向、覆盖和基线 |
| Roohani、Huang、Leskovec，GEARS | 使用基因关系图预测遗传扰动响应；官方数据加载代码明确注明 Replogle 分发集经过过滤、不是完整数据 | GEARS 属于 A 的参考方法，C 不为它引入整套图计算环境；5000 基因分发集的覆盖限制是真实的方法约束 |
| Replogle 等，Cell 2022 | 原论文的数据可用性部分指向 GWPS 门户；原始测序在 PRJNA831566；实验区分 K562 essential 与 genome-wide | 若补全输入基因，应先找同一 essential 实验的可追溯矩阵和元数据，避免直接下载所有测序数据或混合实验 |
| scikit-learn Ridge/PCA 官方文档 | Ridge 原生支持多输出；PCA 默认中心化而不缩放，`components_` 为主轴方向 | 固定具体载荷定义和输入形状，使用成熟实现，避免把细胞 score 当作目标基因特征 |

链接：

- [Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines](https://www.nature.com/articles/s41592-025-02772-6)
- [上述论文的官方复现仓库](https://github.com/const-ae/linear_perturbation_prediction-Paper)
- [Deep learning perturbation models can outperform baselines on calibrated metrics](https://www.nature.com/articles/s41587-026-03307-w)
- [Predicting transcriptional outcomes of novel multigene perturbations with GEARS](https://www.nature.com/articles/s41587-023-01905-6)
- [GEARS 官方数据加载实现](https://github.com/snap-stanford/GEARS/blob/master/gears/pertdata.py)
- [Mapping information-rich genotype-phenotype landscapes with genome-scale Perturb-seq](https://doi.org/10.1016/j.cell.2022.05.013)
- [Replogle 论文开放全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC9380471/)
- [GWPS 官方数据门户](https://gwps.wi.mit.edu/)
- [Ridge 官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
- [PCA 官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- [Dataverse 文件元数据](https://dataverse.harvard.edu/api/files/7458695)

已直接读取论文正文/方法或官方文档；Replogle 正文通过 Europe PMC 的 `PMC9380471/fullTextXML` 核验数据来源。官方代码仓库链接用于溯源，本轮没有复现这些论文的模型。

## 2. 为什么选择 Ridge 和小型 MLP

本任务先将同一目标的细胞聚合为一个标签向量。B 报告每个 seed 只有 760 个训练目标，而且在现有测量基因集合中符号覆盖总共只有 411/1086。模型有效训练样本数会更少。因此先考察正则化线性映射，比直接增加网络规模更有依据。

输入特征为 `F_train: [N_available_train, K]`，标签为 `Y_train: [N_available_train, G]`。K 为 2、最多 50 或最多 52，G 为约 2000。Ridge 拟合从目标表示到完整 delta 向量的映射，L2 正则限制少量目标下的系数放大。MLP 在同一输入、同一标签、同一划分上增加非线性，用固定的小搜索范围检验是否值得增加模型复杂度。

2025 年线性基线论文的模型是具有读出基因表示 G、扰动表示 P 的双侧映射，并可从训练扰动伪 bulk 或外部数据取得表示。本项目的“只用训练对照拟合 PCA + 普通多输出 Ridge”与其训练数据和参数化都不同。该论文支持强基线的必要性，不构成对我们这套具体表示有效性的证明。

2026 年指标校准论文也不证明本项目的 MLP 必须胜出。两项研究共同要求把模型、数据、划分和指标说清楚，再依据本项目实验作判断。

## 3. PCA 载荷如何成为目标基因表示

从该 seed 的训练对照矩阵开始：`X_ctrl` 的行是对照细胞，列是测量基因。按列中心化后拟合 PCA，得到：

- `pca.transform(X_ctrl)`：每个细胞的低维 score，形状为 `[N_control, K]`。
- `pca.components_.T`：每个测量基因在各主轴上的系数，形状为 `[G_measured, K]`。

C 需要的是后者中目标基因所在的一行。它概括该基因在正常对照细胞变动结构中的位置；它是否能预测扰动后的响应是待检验假设，不能直接解释为因果调控关系。

追加对照均值和正值比例后，可以比较三个问题：

1. 只有表达量与稀疏程度，能预测到什么程度？
2. PCA 表达结构是否提供额外信息？
3. 两类信息组合是否比各自单独使用更好？

这里采用主轴系数作为“载荷”，不乘特征值平方根，避免不同文献中同名量混用。中心化不等于单位方差缩放；这一版固定为中心化、不按基因缩放、不 whitening。特征输入随后在训练目标之间标准化，是另一项操作。

输出的约 2000 个 HVG 和输入的 5000 个测量基因是不同集合。一个目标不在输出列表，不妨碍它拥有输入表示；一个目标不在测量矩阵里，则不能由当前 PCA 凭空得到表示。仅有聚合后的目标响应均值也不足以恢复对照细胞中的协方差和检出率。

## 4. 评价结论的边界

共同响应可能让训练均值基线取得较好的总体误差。因此全部目标成绩必须和有表示子集成绩一起给出；缺失目标统一用同一个训练均值回退，不能当作每个模型分别成功预测了这些目标。

同时减去训练均值会改变相关性和排序，但不会改变逐坐标预测误差：

`(delta_pred - mean_train) - (delta_true - mean_train) = delta_pred - delta_true`

所以“残差 MSE”与原 MSE 完全相同。仓库提出的残差相关可作为辅助观察，但它不是 Miller 等定义的 DRF。DRF 还需要明确定义的正负对照，不能只给已有指标改名。

Top-20 最大变化不自动等于统计显著 DEG；重合率不等于通路富集。若补充排除目标自身坐标后的指标，需提前定义和全方法一致计算，避免直接抑制目标基因的变化掩盖下游响应表现。

## 5. 数据版本对研究问题的影响

GEARS 官方代码将文件 7458695 标记为 filtered / not complete。B 进一步报告其测量基因仅 5000 个，保留目标的符号覆盖为 37.8%。据此，现有数据可以回答“在这个受限输入与既定回退规则下，各方法表现如何”，不适合仅凭总体分数否定“更完整的基因表示能否帮助预测”这个更宽的问题。

补齐全基因训练对照是改善表示覆盖的直接方向；不需要为了 Ridge/MLP 下载原始 FASTQ。能否与 B 的细胞和表达空间正确匹配，仍须实际文件验证。若连输出基因选择和标签处理也更换，则属于新数据版本，必须让各模型重新在同一版本上比较。

实施方案、数据清单和每阶段验收见 [C 执行计划](c_execution_plan.md)。
