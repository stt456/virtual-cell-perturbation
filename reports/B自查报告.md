# B 部分逐项自查

## 已完成
- B.3.1 下载 GEARS 文件并检查 X、obs、var、稀疏格式及可用字段；新增 src/download_b.py。原始文件没有单独 guide 和 batch 字段，不能报告其分布。
- B.3.2 统一使用原文件 X，未追加变换；抽样为非整数，文件缺少足够的处理来源，不能确认是否经过何种归一化或 log1p。不得当作已确认的原始计数或标准 log1p 数据。
- B.3.3 对 ctrl 和形如 gene+ctrl 的条件解析，1086 个单目标通过至少 20 细胞过滤；103 个细胞来自 6 个低于阈值的目标，逐细胞原因见 excluded_cells_reasons.csv。
- B.3.4—5 对目标及对照按 70/15/余数划分；保存 42、123、2026 三套划分；目标集合不交叉。对照按细胞分配，见 coverage_and_controls_by_split.csv。每个目标整体分配，理论上其所有 guide 随目标走；因没有 guide 信息，无法单独验证。
- B.3.6 使用每套训练细胞的统计选择 2000 基因，输出含 ID 和符号的基因列表。原文件上游预选 5000 基因，其是否使用全数据尚未确认，因此不能宣称全流程无泄漏。每套采用自定义均值分箱标准化离散度排序，不是 Scanpy Seurat HVG。
- B.3.7 稀疏矩阵分块读取，按条件和对照分别聚合；独立复算每套三个对照及三个目标，共 18 组，最大绝对误差约 1.41e-6，计数一致，见 aggregation_independent_checks.csv。
- B.4 绘制扰动细胞数、划分数量和表达概况，检查重复符号、缺失 condition 及目标覆盖。目标符号覆盖及各集合对照数量见 target_mapping.csv、coverage_and_controls_by_split.csv；基因符号存在重复 TBCE，输出使用 Ensembl ID。
- B.5 已提供下载/处理/自查脚本、报告、划分、基因列表、每条件均值及接口说明；原始 h5ad 与映射文件可供下游使用。

## 不能声称完全完成的事项
1. 对原文件是否严格归一化、log1p 的来源未确认；统一沿用 X 是暂定约定。若项目必须确认原始流程，需要查明上游数据生产记录或从原始计数开始处理。
2. guide 和 batch 的原始字段不存在，无法核验 guide 标签准确性或提供分批分布；恢复这些信息需要与上游原始细胞元数据匹配。
3. 上游预选基因的时间及是否全数据选取未证实，因此需要警惕可能泄漏。不要使用原 h5ad 中 uns 的上游差异表达、HVG 等统计作为训练专用信息。
4. 当前 5000 基因内目标覆盖仅为 411/1086（保留目标；各划分详见 coverage_and_controls_by_split.csv）。C 中许多目标可能需要回退预测；这不应理解为全转录组中不存在这些基因。

## 可复现运行
使用现有 Python 3.9.13、anndata 0.8.0、numpy 1.21.5、pandas 1.4.4、scipy 1.9.1、h5py 3.7.0、matplotlib 3.5.1。anndata 位于项目 python_packages，其余包来自现有 Python 环境；还未提供完全锁定的独立环境。

PowerShell：
$env:PYTHONPATH=".\python_packages"
python ".\src\process_b.py"
python ".\src\report_b.py"
python ".\src\audit_b.py"

压缩包 SHA256 E769C89AA876061FDB3AC02A8DE274F19A741A4F95DECF14A52FD621D1EEA319；h5ad SHA256 550FDE89AC85C258C9FD517638FE34FB42C0E748C9305FE6355761A2BB754170。
