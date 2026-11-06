# K562 虚拟细胞扰动响应预测

利用公开 Perturb-seq 数据，研究目标基因的表达结构信息能否帮助模型预测未见基因受到 CRISPRi 抑制后的群体平均转录响应。主模型为 Ridge 和 MLP，参考模型计划优先尝试 GEARS，环境适配困难时使用基因表示 kNN。

仓库已包含 A 的框架、B 的数据处理流程、E 的分析规划，以及 C 的基因表示、Ridge/MLP、三种子消融与诊断评估结果。原始数据和大型运行产物通过独立文件渠道交接；A 的参考方法、D 的独立评测和 E 的结果解释继续按分工接入。

项目仓库：https://github.com/stt456/virtual-cell-perturbation

## 先看这些文件

- [A 阶段交付清单](reports/a_nov08_delivery.md)：本次成果与交接事项。
- [四篇核心文献笔记](reports/literature_review.md)：研究问题、数据、方法、评价、局限和项目启发。
- [统一接口 v0.1](docs/interfaces.md)：B→C、A/C→D、D→E 的文件格式。
- [协作与阶段验收](docs/collaboration.md)：分工、交付顺序和检查清单。
- [统一配置](configs/default.json)：种子、划分、模型候选参数与路径。
- [GitHub 上传说明](docs/github_setup.md)：远程仓库连接步骤。
- [C 运行与交付说明](docs/c_models.md)：数据前提、特征、训练、冻结和预测接口。
- [C 实际结果](reports/c_results.md)：72 个候选实验、三种子结果与限制。
- [C 数据核验](reports/c_data_verification_20261106.md)：源文件哈希、覆盖率和独立复算。

## 任务定义

对扰动目标 g，预测 `delta_g = mean(expression | perturbation=g) - mean(expression | control)`，表达空间由 B 核实后固定。输入为目标基因表示，输出为目标数 T × 输出基因数 G 的变化矩阵。CRISPRi 是基因表达抑制，不能把本任务写成已经验证的完全基因敲除。

主要评价对象是群体平均响应。把预测变化加到对照细胞表达上仅能得到展示用示意，不能据此声称模型预测了单细胞异质性。

## 统一目录

```text
configs/              统一配置
data/raw/             原始数据，本地保存
data/processed/       B 交付的数据和聚合矩阵
data/splits/          三套随机种子的划分
data/gene_sets/       E 使用的基因集和版本说明
docs/                 接口和协作说明
examples/             明确标记的模拟接口样例
src/data/             B 的数据代码
src/features/         C 的基因表示代码
src/models/           A 的参考模型与 C 的主模型
src/evaluation/       D 的评测代码
src/biology/          E 的生物学分析代码
scripts/              运行入口和框架检查
tests/                接口检查的正确性测试
results/              模型预测和指标，本地保存
figures/              正式图表
reports/              文献、数据与结果报告
```

## 快速检查框架

在仓库根目录运行，Python 标准库即可，无需先下载数据：

```powershell
python scripts/check_framework.py
python -m unittest discover -s tests -p test_interfaces.py -v
```

这两条命令检查配置、目录和模拟接口，不能替代真实数据实验。

C 的全部模型测试需要科学计算与 PyTorch 依赖：`python -m pip install -r requirements-c.txt`，然后运行 `python -m unittest discover -s tests -v`。C 的实际运行已在 Linux/WSL 验证；环境和数据前提见 `docs/c_models.md`。

## 实验环境

建议 Python 3.11。正式项目环境与 GEARS 环境分开，避免图计算依赖冲突。

```powershell
conda env create -f environment.yml
conda activate virtual-cell
```

也可以使用已有 Python：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

依赖文件是起始兼容范围，尚未在真实数据上验证，也不是已验证的锁定环境。首次正式运行后保存实际环境：`python -m pip freeze > results/environment-lock.txt`。

## 实验约定

1. 按扰动目标基因划分 70%/15%/15%；同一目标的细胞和 guide 不得跨集合。种子为 42、123、2026。
2. 对照细胞另行划分；PCA、标准化和输出高变基因选择只用训练允许的数据。
3. 约 2000 个输出基因；目标基因的特征覆盖与输出基因集合分开管理。
4. 同一划分中的所有方法使用相同目标、基因顺序和表达空间；未知表示目标使用训练平均扰动回退，单独记录。
5. 基线包括零变化、训练扰动等权均值，以及 A 提供的参考方法；D 实现正式指标。
6. 数据下载地址、处理状态和标签由 B 检查实际文件后确认。配置中的 `null` 是待核实项，不能当成默认生物学事实。

## 角色与当前阶段

| 角色 | 本阶段工作 | 后续工作 |
|---|---|---|
| A | 文献、仓库、README、目录和接口协调 | 参考模型实验、端到端集成 |
| B | 下载与检查数据，提交概览 | 预处理、划分、聚合和数据字典 |
| C | 表示、Ridge/MLP、三种子消融及预测交付 | 配合 D 独立核验与组内集成 |
| D | 确定基线指标并用模拟数据验证 | 三种子统一评测和统计分析 |
| E | 生物背景、基因集来源和案例规则 | 通路分析、图表和汇报 |

本项目的 C 实际分数见 `reports/c_results.md`，目前属于 C 诊断评估，尚不能称为 D 已独立验收。文献里的结果属于原论文，不混入本项目结果表。
