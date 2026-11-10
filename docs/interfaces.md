# 统一接口 v0.1

本文件是 A 整理的可执行协作草案，供 B、C、D、E 对齐；尚未声称已获组员确认。变更字段或表达空间时同时修改配置与版本号。

## 共同定义

- `target_gene`：扰动目标基因的统一人类基因符号，不能以 guide ID 代替。
- `gene_ids`：模型输出的基因符号，去重，顺序固定。
- `delta`：在确定的表达空间中，扰动均值减去匹配对照均值。
- `split`：`train`、`val`、`test`；划分单位为扰动目标。
- `seed`：42、123、2026。每个 seed 单独发布完整产物。
- 任何模拟样例必须带 `synthetic: true`，不得混入真实评测。

## B 交付数据

### 1 数据概览

`reports/data_report.md` 至少记录：下载源、文件哈希与大小、原始/过滤后细胞与基因数量、扰动目标数量、对照数量、原始标签值、guide/批次字段是否存在、处理空间、缺失和过滤原因。缺少的字段明确记为未提供。

### 2 细胞信息与划分

`data/processed/cell_metadata.csv`：

```csv
cell_id,target_gene,is_control,guide_id,batch
```

`is_control` 使用 `true`/`false`；对照的 `target_gene` 统一为 `control`，原始标签映射另存字典。未知 guide 或批次可为空，禁止编造。

每个 `data/splits/seed_42.json`：

```json
{
  "schema_version": "0.1",
  "seed": 42,
  "targets": {"train": ["GENE_A"], "val": ["GENE_B"], "test": ["GENE_C"]},
  "control_cell_ids": {"train": ["CTRL_1"], "val": ["CTRL_2"], "test": ["CTRL_3"]}
}
```

以上名称仅说明格式。真实目标集合必须两两不相交；同目标的所有细胞和 guide 一起划分；对照 ID 也互不重叠。默认最少 20 个扰动细胞；小数据比例取整规则由 B 写入元数据，不能静默丢目标。

### 3 表达与聚合产物

- `expression.h5ad`：细胞 × 测量基因，保留稀疏矩阵；obs 包含以上标准字段。未核验前不假定 `.X` 是原始 count。
- `seed_42/output_genes.txt`：每行一个输出基因，仅用训练可用细胞选取，约 2000 个。
- `seed_42/aggregates.npz`：`target_ids` [T]、`gene_ids` [G]、`split` [T]、`mean_perturbed` [T,G]、`mean_control` [T,G]、`delta_true` [T,G]、`n_cells` [T]。
- `seed_42/metadata.json`：schema、seed、数据哈希、表达层、归一化/log1p 状态、目标过滤阈值、输出基因选择方法、对照匹配规则和源字段映射。

`delta_true = mean_perturbed - mean_control`。对照应在相应划分内按可用批次匹配；若跨批次聚合，明确权重。若无法匹配，记录并由组员决定排除或统一处理。不得把测试扰动结果用于训练基线、特征或调参。

## C 的输入与表示

在训练对照细胞上对测量基因拟合 PCA，取目标基因的前 50 维载荷，再追加训练对照均值和检出率，组合表示预期为 [T,52]。可用维数不足时记录实际维数，不靠补维掩盖问题。标准化参数仅从训练目标特征拟合。

输出 HVG 集合与特征构建的测量基因集合分开：目标不在输出 HVG 中不等于目标没有表示。明确 PCA 前的中心化、缩放和载荷定义，并保存拟合参数。

建议接口：`train(features_train, delta_train, features_val, delta_val, config)`；`predict(target_ids, features, config)`。验证集仅调参和早停，测试标签由 D 使用。

## A/C 向 D 交付预测

`results/{model}/seed_42/predictions.npz`：

| 字段 | 形状/类型 | 含义 |
|---|---|---|
| target_ids | [T_test] 字符串 | 与测试清单严格一致 |
| gene_ids | [G] 字符串 | 与输出基因清单严格一致 |
| delta_pred | [T_test,G] float32/64 | 有限数，不得有 NaN/Inf |
| feature_available | [T_test] bool | 目标表示是否可用 |
| used_fallback | [T_test] bool | 是否使用训练平均变化回退 |

同目录 `run.json` 保存：schema、seed、model、配置快照、数据/划分/基因列表哈希、训练命令、代码提交、环境版本、表达空间和 `synthetic: false`。模型文件、参数和日志同目录保存。NPZ 字符串数组采用 Unicode 类型，读取用 `allow_pickle=False`。

模型名建议 `zero`、`train_mean`、`knn` 或 `gears`、`ridge`、`mlp`。如 GEARS 最终预测是绝对表达，A 必须按同一对照定义转成 delta；不能直接当作变化矩阵提交。

## D 的指标约定

- 逐目标计算后等权平均，避免细胞多的目标主导结果。
- MSE：全输出基因变化误差。Pearson/Spearman：变化向量相关；常数向量记缺失并报有效目标数。
- Top-k：真实与预测分别按绝对变化降序取前 k，k=min(20,G)，并列按 `gene_ids` 的字典序打破。重合率=交集数/k；方向一致率只对交集计算，无交集记缺失。零变化基线的方向率记缺失，其 Top-k 属于无信号并列选择，须标记解释。
- “变化最多的基因”不是已经经过统计检验的 DEG；若做 DEG 分析，需另定义检验方法和 FDR。
- 特异性：预测和真实 delta 各减去训练目标等权平均 delta 后，重新计算误差和相关；这不是论文 DRF 的同义词。
- 三种子分别报告，再汇总均值和样本标准差；bootstrap 按目标重采样，配对比较同目标误差。覆盖可用子集和全部目标同时报告。

输出 `per_target.csv`、`summary.csv` 与 `evaluation.json`，至少包含 model、seed、target_gene、指标值、有效数量、fallback 信息与数据版本。评测脚本和小样本正确性验证由 D 负责。

## D 向 E 交付

真实/预测 delta、目标/基因顺序、逐目标指标和训练均值基线；在查看测试成绩前确定案例选择规则。富集分析固定基因集版本、背景集合和多重检验方法。若只使用 HVG，背景应限于实际可评估且有注释的基因，并说明覆盖限制。

## 当前阶段的模拟样例

`examples/synthetic_predictions.csv` 用宽表演示相同顺序的预测矩阵，`examples/synthetic_split.json` 演示目标隔离。它们只验证接口，不是 NPZ 正式交付，也不包含真实数据或成绩。

## D 统一评测实现与当前交付

- 入口：`scripts/run_d_evaluation.py`；指标、IO、统计、图表及报告实现位于 `src/evaluation/`。
- 输入：C 冻结实验目录中的 `freeze.json`、预测包、模型训练均值、运行元数据与 C 自检评估表。handoff 模式会核对冻结文件 SHA256、预测契约和 C 已报告指标的聚合一致性；不会把缺失的 B 标签替换为零，也不会声称完成独立重算。
- 独立评测：`--mode independent` 必须找到并匹配 `freeze.json` 内的 B 划分、训练/验证标签、输出基因清单和测试标签哈希；缺失或版本不一致时停止。A 预测通过带逐种子文件哈希的 JSON manifest 加入。
- 本地完整结果：`results/d_v1_20261110/`，包括逐目标表、分层汇总、bootstrap、核验清单、图表、manifest 和 E 案例交接。该目录遵守仓库规则，不提交大型逐目标产物；仅跟踪小型 `manifest.json`、`audit.json` 和 `pending_inputs.json`。
- 可共享摘要：`reports/d_summary.csv`、`reports/d_metrics_by_seed.csv`、`reports/d_metrics_across_seeds.csv`、`reports/d_paired_bootstrap.csv`、`reports/d_coverage.csv` 与 `reports/d_scope.json`；图表放在 `figures/d_*.png` 和 `figures/d_*.svg`。
- 当前范围和运行命令见 `docs/d_evaluation.md`；结果解释见 `reports/d_results.md`。指标定义仍遵循本页上文，缺失相关值保持为空并同时报告有效目标数。
