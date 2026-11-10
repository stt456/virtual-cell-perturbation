# D 统一评测：方法、复现与交接

## 当前结果的证据范围

当前结果来自冻结实验 `c_v1_20261106`。D 独立核对交接文件的哈希、预测包的目标/基因顺序、矩阵形状、有限数值、覆盖掩码和训练均值回退，并从 C 提供的逐目标指标表重新汇总统计量。

本轮按用户要求将 B 的训练/测试标签与独立划分，以及 A 的参考预测暂作空值处理。故 `audit.json` 中 `b_label_hashes_verified`、`independent_metric_recalculation`、`control_split_checked` 和 `train_mean_recomputed_from_labels` 均为 `false`。当前指标是 C 自检逐目标记录的 D 统计复算，不能表述为 D 使用独立真值重新计算，也不能据此验证标签真实性。A 或 B 缺项不作为零分。

## 指标与统计

- 先对每个扰动目标在输出基因上计算指标，再对目标等权平均；不按细胞数加权。
- 主指标为 MSE、Pearson、Spearman、Top-20 重合率、Top-20 交集方向一致率，以及扣除训练平均扰动后的残差 Pearson/Spearman。辅助项包括剔除目标自身输出坐标后的 MSE。
- Top-20 以绝对变化从大到小排序，按 gene ID 字典序打破并列；常数向量相关性和没有有效交集的方向率记为空，并报告有效目标数。全零预测的 Top-20 只表示稳定的并列选择，不代表模型识别了变化基因。
- 逐种子报告后，对三个种子的宏平均指标等权汇总；标准差采用样本标准差（ddof=1）。配对 bootstrap 在同一个种子、同一 split 内按目标重采样 10,000 次。正的 MSE 差值表示优于训练平均扰动基线；不把不同种子的重复目标当成独立样本合并。
- “Top-20 变化基因”不是经过统计检验的 DEG。本轮没有独立真值，不能据此开展真实/预测散点图、DEG 检验或通路富集。

详细参数见 [`configs/d_evaluation.json`](../configs/d_evaluation.json)，结果解读见 [`reports/d_results.md`](../reports/d_results.md)。

## 运行当前交接包

从仓库根目录运行。若解压后的 C 冻结交接包位于仓库目录的上一级，入口会自动使用该目录；否则将读取仓库本身的 `results/c_v1_20261106`。

```powershell
python -m pip install -r requirements-d.txt
python scripts/run_d_evaluation.py --mode handoff --output results/d_v1_20261110
```

输出目录必须不存在，以免覆盖之前的审计包。完整结果保存在 `results/d_v1_20261110/`，该目录含逐目标表、分层统计、bootstrap、冻结文件检查、图表、manifest 和 E 案例交接。仓库只跟踪其中小型的 `manifest.json`、`audit.json` 和 `pending_inputs.json`，忽略完整本地运行数据；汇总表、图表和报告会写入 `reports/` 与 `figures/` 供共享。

## 补齐 B 后的独立重算

提供与 C 冻结时版本完全相同、且哈希匹配 `freeze.json` 的文件。对每个 seed，程序要求：

```text
data/processed/c_inputs/seed_{seed}/split.json
data/processed/c_inputs/seed_{seed}/labels_train_val.npz
data/processed/c_inputs/seed_{seed}/output_genes.txt
data/processed/seed_{seed}/D_test.npz
```

也支持 B 接口路径 `data/splits/seed_{seed}.json`、`data/processed/seed_{seed}/C_train_val.npz` 和 `output_genes.txt`；每个文件仍须与 C 冻结记录中的对应副本哈希一致。`--data-root` 指向包含 `data/` 的目录。

```powershell
python scripts/run_d_evaluation.py --mode independent --data-root "D:\完整项目" --output results/d_independent_v1
```

独立模式会核验目标划分、训练均值、目标和基因顺序，并从 B 标签逐目标重新计算指标。缺文件或哈希不符时停止，不会输出伪造的独立结果。重新评测应使用新的输出目录。

## 加入 A 的参考模型

A 为每个 seed 提交相同测试目标、输出基因顺序和覆盖掩码的 `predictions.npz`，并提供 SHA256。通过 JSON manifest 传入，例如：

```json
{
  "model": "knn",
  "test_labels_accessed": false,
  "seeds": {
    "42": {"predictions": "seed_42/predictions.npz", "sha256": "实际文件的 SHA256"},
    "123": {"predictions": "seed_123/predictions.npz", "sha256": "实际文件的 SHA256"},
    "2026": {"predictions": "seed_2026/predictions.npz", "sha256": "实际文件的 SHA256"}
  }
}
```

相对预测路径以 manifest 所在目录为基准。获得 A 与 B 的真实文件后，运行：

```powershell
python scripts/run_d_evaluation.py --mode independent --data-root "D:\完整项目" --a-manifest "D:\A交接\manifest.json" --output results/d_with_reference_v1
```

## 文件位置

- 入口：[`scripts/run_d_evaluation.py`](../scripts/run_d_evaluation.py)；实现：[`src/evaluation/`](../src/evaluation/)。
- 方法说明和当前边界：[`reports/d_results.md`](../reports/d_results.md)；离线交互表：[`reports/d_results.html`](../reports/d_results.html)。
- 完整逐目标审计包：本地忽略目录 `results/d_v1_20261110/`。汇总表、覆盖率、bootstrap 和交互结果页保存在 `reports/`；PNG/SVG 图在 `figures/`。
- `reports/d_scope.json` 记录本轮核验状态。以后完成 B 标签独立重算后，应以新报告和新结果目录更新，不覆盖本次证据范围。
