# B 数据检查与预处理报告

## 数据来源和概况
GEARS 分发的 Replogle K562 Essential 数据，下载地址 https://dataverse.harvard.edu/api/access/datafile/7458695 。文件为预处理版本，162751 个细胞、5000 个基因；稀疏表达矩阵。原数据并非完整原始数据，已有上游基因选择。共 1092 个扰动目标、10691 个对照细胞；按每个目标至少 20 个细胞过滤后保留 1086 个目标、162648 个细胞，排除 103 个细胞。缺失 condition 标签 0。重复基因符号：[{'gene_id': 'ENSG00000285053', 'gene_name': 'TBCE'}, {'gene_id': 'ENSG00000284770', 'gene_name': 'TBCE'}]；矩阵和输出均以 Ensembl gene_id 标识以避免歧义。目标符号在 5000 个输入基因符号中的覆盖：412/1092（含过滤前目标；详见 target_mapping.csv）。

## 表达空间与限制
直接使用 GEARS 提供的 X；抽样中非零值非整数，与转换后的表达量一致，但文件本身不足以证明归一化方法或 log1p 流程。未再次归一化，不能将此推断当成已核实的标准化方案。未发现独立 guide 和 batch 字段，无法核查 guide 分布或批次分布；目标分组可避免同一目标跨集合，但无法独立检验 guide。需要上游来源补充元数据。原始版本预选 5000 个基因，可能基于全部细胞，存在上游信息泄漏，严格无泄漏实验需要从原始全基因数据重新按训练集选基因。

## 划分及输出基因
目标以随机种子 42、123、2026 独立打乱，按目标数 70/15/余数划分；对照细胞独立以相同比例划分。每个种子训练/验证/测试目标为 760/162/164，集合互不相交。细胞数量和各条件数量见 split_summary.csv、mean_counts 文件。各 seed 在训练细胞内计算基因方差与均值比，按均值分箱后标准化分数，选择 2000 基因；这是本项目自定义离散度排序，不等同 Scanpy Seurat HVG。每套输出基因可能不同。

## 接口和使用
- data/replogle_k562_essential/perturb_processed.h5ad：原始分发矩阵；行和列对应 cell_metadata.csv、gene_metadata.csv。
- results/split_<seed>.csv：cell_id、target、split（train/val/test/excluded）。ctrl 为非靶向对照。
- results/hvg_<seed>.csv：训练集筛选的 2000 基因、训练统计及分数。
- results/mean_expression_<seed>.csv.gz：行是扰动基因或 ctrl_train/ctrl_val/ctrl_test，列为基因 ID，值为所选基因的平均表达；对应样本数见 mean_counts_<seed>.csv。扰动目标只归属对应划分。评测时用相应划分的对照，明确控制组定义。
- target_counts.csv 和 target_mapping.csv：过滤前目标数量及目标符号覆盖；gene_metadata.csv 保存基因 ID 和符号。
- figures：扰动细胞数、划分目标及细胞数量、seed 42 对照与扰动表达概览。

运行：在项目目录设置 PYTHONPATH 指向 python_packages，执行 python src/process_b.py，然后 python src/report_b.py。处理使用 2048 行分块；HVG 统计仅使用该种子的训练细胞。检查已验证三个划分的目标不交叉、细胞 ID 唯一、均值矩阵形状及有限值。

## 自查补充
请同时阅读 B自查报告.md、coverage_and_controls_by_split.csv、aggregation_independent_checks.csv 和 excluded_cells_reasons.csv。上游 GEARS 文件预选基因的具体时点尚未核实；潜在泄漏是风险而非已证实事实。
