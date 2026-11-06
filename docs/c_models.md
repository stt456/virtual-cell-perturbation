# C 模型：输入、运行和交付

C 将一个扰动目标作为一个监督样本，输入是训练对照来源的基因特征，输出是约 2000 个指标的平均变化。三种表示为 PCA、源矩阵均值/正值比例、二者组合；模型为多输出 Ridge 与单隐藏层 MLP。

## 数据边界

`scripts/prepare_c_inputs.py` 从已冻结的 B 产物导出 `data/processed/c_inputs/seed_<seed>/`，包含训练对照稀疏矩阵、基因/细胞 ID、划分及 `labels_train_val.npz`，不导出测试标签。完整原始 h5ad 的检查属于预处理步骤；模型入口不读取原始 h5ad、`aggregates.npz` 或 `D_test.npz`。

`scripts/build_c_features.py` 只用训练对照拟合 PCA，保存到 `data/processed/c_features/seed_<seed>/`。目标映射使用全部 5000 个输入基因，不限于 2000 个输出基因；歧义/缺失目标被标记，模型拟合排除它们，预测时采用训练等权均值回退。

训练入口按每个 seed 独立调参；跨种子汇总只作描述，不利用另一个划分的验证标签替当前 seed 选模型。每种模型/表示单独保存参数。输出基因顺序始终沿用 B 的接口文件。

## 已使用的运行方式

先取得 B 的 v0.1 标准接口，把 `data/processed/` 和 `data/splits/` 放到仓库相应目录；源 ZIP 本身还不是这些处理产物。可按 B 的 `process_b.py → report_b.py → audit_b.py → export_repo_interface.py → separate_labels.py → validate_repo_export.py` 在数据副本目录重建后导入。仓库附带本次冻结数据的 `results/c_data_verification_20261106/installed_data_manifest.json`，训练准备会核对实际哈希；不匹配时先核对数据版本，不能直接跳过检查。

原始数据、预测矩阵及模型权重不随 Git 提交，D 接收预测文件时需要单独取得[交接包](../reports/c_handoff.md)。报告中指向大型 `data/`、`results/` 文件的链接是本地证据路径；Git 仓库提供代码、小型指标表、哈希和图表。

依赖入口为 `requirements-c.txt`，其中范围用于环境安装；本轮数据环境的确切版本见数据核验报告，训练环境版本见 `results/c_v1_20261106/experiment.json`。建议复用已有可用 PyTorch 环境，或在独立环境中安装 `python -m pip install -r requirements-c.txt`。已验证平台为 Linux/WSL，原生 Windows 运行尚未验证。

在仓库根目录执行。数据准备复用本项目 `.venv`；数学计算和 PyTorch 复用已有 `cs224n` 环境。下面的 `conda run -n cs224n` 可替换为该环境的 Python 路径，无需安装到系统 Python。

```bash
.venv/bin/python scripts/prepare_c_inputs.py
conda run -n cs224n python scripts/build_c_features.py
conda run -n cs224n python -m unittest discover -s tests -v
conda run -n cs224n python scripts/run_c_models.py --mode smoke --device cuda
conda run -n cs224n python scripts/run_c_models.py --mode pilot --device cuda
conda run -n cs224n python scripts/run_c_models.py --mode formal --device cuda
conda run -n cs224n python scripts/evaluate_c_results.py --freeze
conda run -n cs224n python scripts/evaluate_c_results.py --evaluate
conda run -n cs224n python scripts/report_c_results.py
```

CPU 可运行同一程序，将 `--device cuda` 改成 `--device cpu`。本轮用户授权使用本地 GPU，只有服务器任务才受物理 1、3 号卡限制。本地只有一张 GPU，进程内编号为 0。若受执行沙箱限制而看不到 CUDA，需要在获准的非沙箱进程中运行，不应把这种情况解释为显卡或驱动损坏。

脚本默认不覆盖已有实验。完整成功的匹配训练结果可通过 `--resume` 核对并复用；改变参数/代码/输入后需使用新的 `--output` 目录。新的实验目录传给后续步骤的 `--experiment`。冻结之后禁止默默修改模型或按测试成绩追加搜索，评估脚本会验证冻结哈希。

## 关键设置

配置是 `configs/c_models.json`。PCA 取中心化矩阵的主轴系数，至多 50 维；MLP 学习相对于所有训练目标等权均值的残差。特征标准化只拟合有表示的训练目标，验证集仅选参数和早停。MSE 不对输出坐标另行标准化，也不按每个目标的细胞数加权。

Ridge 候选 alpha 为 0.1/1/10/100。MLP 的隐藏维为 64/128，学习率为 0.001/0.0003，dropout=0.1，AdamW weight decay=0.0001，batch=64，最多 200 epochs，patience=20。固定配置下共 72 个候选拟合。

MLP 的 dropout 和 weight decay 在本轮预先固定，没有做正则强度搜索；验证集选择的是隐藏维度、学习率和最佳停止 epoch。这是与组内规划泛称“正则化由验证集选择”相比的明确范围差异，不能写成已经搜索了所有正则参数。C 的提交不包含新的状态条件模型、正式 DEG 检验、A 的参考模型或 E 的通路分析。

## 交给 D 的文件

正式实验默认放在 `results/c_v1_20261106/<model>/seed_<seed>/`，`model` 是 `zero`、`train_mean` 或 `ridge_pca`、`ridge_mean_detection`、`ridge_combined`、`mlp_pca`、`mlp_mean_detection`、`mlp_combined`。

- `predictions.npz`：测试 `target_ids`、`gene_ids`、`delta_pred`、`feature_available`、`used_fallback`，均可用 `allow_pickle=False` 读取。
- `run.json`：配置、参数、数据/源码/环境记录、资源消耗和文件哈希。
- `model_arrays.npz`：训练均值、标准化参数、Ridge 系数等；MLP 另有 `model_weights.pt` 和 `model.json`。
- `validation_predictions.npz`、`validation_per_target.csv`、`validation_summary.csv`：验证预测与评估。
- `search.json`：全部候选参数、验证成绩、训练日志和耗时。

零变化和训练均值是独立基线，其 `used_fallback` 标志为 false；模型因输入缺失而使用训练均值时才标记 true。所有方法的 `feature_available` 使用同一输入覆盖掩码，便于 D 统一报告可映射子集。

本轮诊断评估位于 `src/evaluation/c_metrics.py`，遵循 v0.1 的逐目标 MSE、变化相关、Top-20、方向与残差相关定义，并保留有效数量。它是 C 自检，不能标为 D 已独立审核。A 的 kNN/GEARS 参考方法尚未接入。
