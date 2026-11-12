# A 的 11 月 12 日交付与接口联调

本次交付沿用[四篇核心文献对照笔记](literature_review.md)，新增[余弦加权 kNN 原型](../scripts/run_a_knn.py)。原型已用模拟数据生成三套验证集与测试集预测，D 的预测校验和指标函数已读取并评估三套模拟验证集结果。真实数据运行待取得同版本输入文件。

## 模型和评价口径

每个目标基因对应一行特征和一个群体平均表达变化向量。默认复用 C 的组合特征，包括 PCA、训练对照均值和检出率。只用有特征的训练目标拟合标准化参数，并按余弦相似度在其中检索最近邻。相似度相同时按目标基因 ID 排序。对邻居响应取加权平均，负相似度权重置零，权重和为零时对这组邻居等权平均。

每个种子单独用验证集的目标等权 MSE 从 3、5、10、20 中选 k，近似同分选较小的 k。无可用特征的目标使用全部训练目标的等权平均响应，沿用 C 的覆盖标记和回退口径。生成测试预测时只使用测试目标 ID 和其特征。

## 各角色之间的交接

| 交接 | 需要的输入或输出 | 当前状态 |
|---|---|---|
| B、C → A | 每个种子的划分、输出基因列表、训练/验证标签、C 的特征矩阵 | GitHub 有代码与哈希记录，实际文件需从组内交接包取得 |
| A → D | 验证预测、测试预测、覆盖与回退标记、预测 SHA256、每种子的 k 和验证误差 | 模拟验证通过，真实预测待运行 |
| B、C → D | B 的独立训练和测试标签、划分，C 的冻结预测包 | D 的 11 月 10 日报告已复算 C 的结果，独立评测仍需 B 的对应文件 |
| D → E | 匹配的真实响应和预测、输出基因顺序、经确认的案例清单 | E 已提交 Reactome 和案例规则草案，正式案例还需 D 的真值与评测交接 |

在正式案例分析前，A、D、E 需统一 Reactome 的背景基因范围和案例规则。E 可以先用示例数据测试通路分析。各方法应复用每个种子自己的输出基因列表，不能把三个种子的预测矩阵按列直接拼接。

## 运行和交付

使用含 C、D 已合并代码的仓库，在根目录运行模拟测试（需要 NumPy）：

```powershell
python -m unittest discover -s tests -p test_a_knn.py -v
```

真实输入使用 C 的冻结版本。请向 C 索取以下文件，各三份（种子 42、123、2026）：

- `data/processed/c_inputs/seed_<seed>/split.json`
- `data/processed/c_inputs/seed_<seed>/output_genes.txt`
- `data/processed/c_inputs/seed_<seed>/labels_train_val.npz`
- `data/processed/c_features/seed_<seed>/features.npz`

将上述目录放在仓库内，或用 `--data-root` 指向包含 `data/` 的交接目录。使用 C 仓库中的 `results/c_v1_20261106/freeze.json` 核对输入哈希，哈希不一致时先确认数据版本。

```powershell
python scripts/run_a_knn.py --data-root "D:\交接目录" --output results/a_knn_v1
```

把新输出目录整体交给 D。每个种子含 `validation_predictions.npz`、`predictions.npz` 和 `run.json`，顶层 `manifest.json` 给出路径与 SHA256。D 当前的 `--a-manifest` 入口读取测试预测。验证预测可用 D 的 `validate_prediction` 和 `evaluate` 函数，与同种子的验证标签配对核验。本次模拟联调用了这两个函数，真实数据上的独立评测仍待完成。
