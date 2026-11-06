# C 接入前的真实数据核验

日期：2026-11-06。参考仓库提交：`3bf46ba`。结论：用户提供的压缩包与 B 使用的 GEARS 分发包相同；本轮实际读取、复跑和独立复算支持 B 报告的主要数量、三套划分及聚合结果。此次未训练 Ridge/MLP，没有模型成绩。

## 文件已落地

- 解压文件：[data/replogle_k562_essential/perturb_processed.h5ad](../data/replogle_k562_essential/perturb_processed.h5ad)，2,898,717,719 字节。
- 已核验的标准接口：[data/processed/](../data/processed/) 和 [data/splits/](../data/splits/)，共导入 23 个文件，逐文件复核 SHA256，没有覆盖已有数据文件。
- 原版 B 脚本的隔离复跑目录：[results/c_data_verification_20261106/b_reproduction/](../results/c_data_verification_20261106/b_reproduction/)。仓库原 B 脚本、报告和图未修改。
- 新建项目 `.venv`，只安装本轮数据处理依赖；没有更改已有 Conda 环境，没有操作服务器或修改 GitHub 远程仓库。

ZIP 的 SHA256 为 `e769c89aa876061fdb3ac02a8de274f19a741a4f95decf14a52fd621d1eea319`，与 B 报告相同；MD5 为 `84eed779531de88f83d3eb16a773a261`，与此前查询的官方文件元数据相同。解压时通过 ZIP CRC 检查。

h5ad 的 SHA256 为 `550fde89ac85c258c9fd517638fe34fb42c0e748c9305fe6355761a2bb754170`，与 B 报告相同。

## 主要统计核验

| 项目 | 实际核验值 | 与原报告比较 |
|---|---:|---|
| 原矩阵 | 162751 × 5000，CSR 稀疏矩阵 | 一致 |
| 过滤前扰动目标 | 1092 | 一致 |
| 至少 20 个细胞的目标 | 1086 | 一致 |
| 对照细胞 | 10691 | 一致 |
| 过滤后总细胞 | 162648 | 一致 |
| 被排除细胞 | 103，来自 6 个低细胞数目标 | 一致 |
| 缺失 condition | 0 | 一致 |
| 重复基因符号 | TBCE 对应两个 Ensembl ID | 一致 |
| 保留目标的输入符号覆盖 | 411/1086，37.8453% | 一致 |
| 无对应输入列的目标 | 675/1086，62.1547% | 一致 |

411 个可匹配目标均能唯一映射到输入列；TBCE 的重复并未造成这 411 个目标的映射歧义。这里验证的是列的可用性，不是 PCA 信息量、预测能力或模型效果。

完整扫描了 CSR 的 316,427,573 个存储值，未发现 NaN/Inf 或非法索引。存储值均为正值，范围约 0.1829565–8.3939848，316,427,549 个值与最近整数的差大于 1e-6。稀疏矩阵中仍有隐式零值；上述统计不证明具体的归一化或 log1p 来源。

文件没有 `.raw`，`layers` 为空。因此这是完整的 GEARS 筛选版分发文件，但不含可直接恢复缺失输入基因的另一份全基因矩阵。先前“仅 5000 个测量基因、675 个目标缺少输入列”的判断得到实际文件支持。

## 三套划分与 C 的实际输入规模

每个 seed 的目标总数都是训练 760、验证 162、测试 164。独立重新生成目标和对照的随机划分，逐细胞与 B 复跑结果一致；同一目标不跨集合，对照 ID 也不跨集合。

| seed | 训练目标可匹配/总数 | 验证目标可匹配/总数 | 测试目标可匹配/总数 |
|---|---:|---:|---:|
| 42 | 284/760 | 61/162 | 66/164 |
| 123 | 279/760 | 59/162 | 73/164 |
| 2026 | 302/760 | 50/162 | 59/164 |

三套划分的训练/验证/测试对照细胞均为 7483/1603/1605。包含扰动与对照的细胞总数分别为：

| seed | train | val | test |
|---|---:|---:|---:|
| 42 | 116178 | 24156 | 22314 |
| 123 | 110196 | 26226 | 26226 |
| 2026 | 115799 | 23979 | 22870 |

这意味着在当前输入表示规则下，C 的模型拟合对象只有约 280–300 个可映射训练目标；不能把全部细胞数当成独立监督样本数。实际表示的信息量仍需在特征构建阶段检查。

## 复跑及独立检查的范围

将原版 `process_b.py`、`report_b.py`、`audit_b.py`、`export_repo_interface.py`、`separate_labels.py`、`validate_repo_export.py` 复制到隔离目录，核对代码哈希后依次运行，六步均通过。原 B 自查复跑的最大聚合误差为约 `1.4028e-6`，与其报告的约 `1.41e-6` 相符。

另写 [scripts/verify_c_data.py](../scripts/verify_c_data.py)，不调用 B 的处理函数，直接从源 HDF5 CSR 分块读取并以 float64 求和：

- 核对原始数量、标签解析、完整划分、目标和基因顺序以及全部矩阵值的有限性。
- 独立复算每个 seed 的三个对照池和 train/val/test 各一个目标，共 18 个条件/seed 组合。计数完全一致，2000 个输出坐标上的最大绝对误差为 `4.440892098500626e-16`。
- 使用训练细胞重新计算 float64 均值、方差及 B 的自定义分箱离散度。三个 seed 选出的基因集合均与 B 复跑的 2000 个基因完全一致。
- 更高精度统计下，三个 seed 的基因排序均有差异；训练均值最大差约 `3.87e-7`，方差最大差约 `2.59e-6`。本次没有改用新的排序，接口文件严格沿用 B 的导出顺序；后续必须按基因 ID 对齐。
- 三套 `aggregates.npz` 均为 1086 × 2000；`C_train_val.npz` 为 922 × 2000，`D_test.npz` 为 164 × 2000。以 `allow_pickle=False` 读取，标签隔离、顺序、有限值、均值与 delta 关系均通过核对。

独立核验复跑命令：

```bash
.venv/bin/python scripts/verify_c_data.py --audit-dir results/c_data_verification_20261106
```

此命令读取已提取的源文件和本轮 B 复跑产物，重新写入该核验目录中的 JSON/CSV 证据。它不重新执行 B 的六个步骤，也不训练模型。

项目框架检查及现有 5 项模拟接口单元测试均通过；`pip check` 通过。真实数据结论来自上述实际文件和独立检查，不能归因于那 5 项模拟测试。

## 证据和仍未解决的问题

- [独立核验 JSON](../results/c_data_verification_20261106/independent_verification.json)
- [各划分覆盖率](../results/c_data_verification_20261106/coverage_by_split.csv)
- [18 组独立聚合复算](../results/c_data_verification_20261106/independent_aggregation.csv)
- [解压与源文件校验](../results/c_data_verification_20261106/extraction.json)
- [本地导入文件哈希](../results/c_data_verification_20261106/installed_data_manifest.json)
- [复跑产物哈希清单](../results/c_data_verification_20261106/artifact_manifest.json)
- [实际依赖版本](../results/c_data_verification_20261106/environment-lock.txt)

本轮环境为 Python 3.11.15、numpy 1.26.4、scipy 1.14.1、pandas 2.2.3、h5py 3.12.1、anndata 0.10.9、matplotlib 3.9.4。它复现了已报告的数量与处理行为，不等于 B 原环境的字节级重现；B 原本生成的聚合文件没有随 ZIP 提供，无法与其逐文件比较。

准确的归一化流程、上游 5000 基因的选择范围、guide/batch 元数据仍未解决。文件里有上游统计字段，但本轮没有用它们拟合 C 特征或选择超参数。675 个目标缺少输入列的限制仍在；需要回退或另行扩展输入数据。完整 h5ad 和 `aggregates.npz` 仍含测试信息，C 训练入口应只读取按 seed 导出的训练对照和 `C_train_val.npz`。

数据获取、B 接口重建及本轮验收已完成；下一步是导出 C 的训练对照输入、构造特征并做模型冒烟测试。
