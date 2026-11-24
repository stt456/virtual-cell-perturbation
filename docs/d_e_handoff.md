# D→E 案例评测与通路交付（2026-11-24）

本次将仓库协作产物与发给 E 的数据材料分别管理。

## 合并到 GitHub 的内容

- `scripts/build_d_e_delivery.py`：读取已发布的真实案例，核对来源哈希、重算指标、复核案例抽取、执行 Reactome ORA，并生成 D 完整构建目录及 E 专用包。
- 本文：案例规则、通路背景、复现步骤及交付边界；`reports/d_e_20261124.md` 为本轮结果摘要。
- `reports/d_e_20261124/`：18 个案例的精简误差表、通路比较、背景注释覆盖、指标重算差异、B 冻结版本索引、实际依赖版本和 E 包索引。

本轮新建的原始矩阵、完整长表、GMT、图表、下载输入 ZIP 和 E 压缩包留作本地交付，不新增进 Git。PR #6 之前已跟踪的案例源文件仍作为本轮可复现数据来源。

## 单独给 E 的内容

专用文件名：`E_通路分析案例包_20261124.zip`。包内首读 `给E_阅读说明.md` 和 `案例选择与背景说明.md`。

包内包含：三个种子各 6 个案例的原 NPZ；36000 行真实/预测/训练均值比较 CSV；模型与基线误差；1080 行 Top-20 基因及方向；完整和显著通路表；完整候选哈希排序；各自的 2000 基因背景；Reactome GMT；散点图；逐文件 SHA-256。

E 包不包含 D 构建输入归档、源码快照或全量旧评测目录。发布索引见 [e_delivery_index.json](../reports/d_e_20261124/e_delivery_index.json)，可按压缩包哈希核对最终发送版本。

## 案例与模型规则

保留 2026-11-16 已发布的事后展示案例，不根据本次误差或通路显著性替换。每个种子在完整发布测试目标表中，按 `feature_available` 分为有表示/无表示两组，计算 UTF-8 字符串 `D_E_v1|{seed}|{target_gene}` 的 SHA-256，按哈希升序、目标符号升序处理并列，各取 3 个。已复核全部候选排名，所选目标与原案例包完全一致。

- 42：RRP12、MAD2L1、CCDC84、NUP98、MRPL43、DMAP1；模型 `mlp_mean_detection`。
- 123：GRB2、LSG1、MED7、MRPL3、SMG5、POP5；模型 `ridge_mean_detection`。
- 2026：HSPE1、EIF4G2、RAD21、PPP1R8、RANGAP1、CTNNBL1；模型 `mlp_mean_detection`。

各模型沿用冻结的验证集选择结果。原 E 草案 `E-case-v1|42|...` 的前 3 名方案并未用于这些已发布案例；不能把本次案例称为测试前预注册或称其代表全测试集。细胞数和 QC 原文件未包含在案例包内，不声称重新验收这类信息。

## 通路背景与统计口径

本包采用 E 草案中的全部共同输出基因方案：每个种子使用真实、模型预测及训练均值共同具有有限值的全部 2000 个输出基因，包含无 Reactome 注释的输出基因。同一种子的所有案例/方法共用背景；不同种子各自保留输出基因及顺序，不用全基因组、各 Top-20 列表或跨种子并集充当背景。

三份背景使用 LF 换行恢复，其 SHA-256 与 C 冻结 `output_genes.txt` 完全一致。原符号精确匹配，2000 个符号均唯一，不静默转大写、补别名或合并。该口径在本次分析阶段明确，不能追溯称为测试前与组员已达成的约定。

真实、模型、训练均值分别按绝对变化降序、基因符号升序打破并列，各取 Top-20。主 ORA 不分上下调，基因方向单独保留。Top-20 是幅度排序，不是统计学 DEG。交集方向一致率在双方变化均非零的交集上计算；分母为零记 NA。

Reactome GMT 来自[官方下载](https://reactome.org/download-data)，本次下载于 2026-11-24；与 E 原记录（团队标注 Reactome 98）完全相同，2874 行，SHA-256 为 `a9feaa3ae9bbb746422cba77a669813de627fb15b7d7478b67c8d57020ec26ae`。文件哈希不符则构建停止，不静默使用新版本。

每通路先与背景取交集，保留大小 10–500 的通路；单侧超几何过度富集检验，零交集也纳入且 p=1。在每个“种子×案例×方法”的全部保留通路内做 BH 校正，以 q<0.05 标记显著。校正范围未覆盖所有案例/方法的联合检验，跨案例发现只用于探索。背景应覆盖所有有机会进入选中列表的基因，参见 [GO 官方说明](https://geneontology.org/docs/go-enrichment-analysis/)。

## 复现

在仓库根目录准备环境：`python -m pip install -r requirements-d.txt`。本轮实际版本见 `reports/d_e_20261124/environment.txt`。

准备两个输入文件到本地 `results/d_e_build_inputs/`（Git 忽略此目录）：

```powershell
New-Item -ItemType Directory -Force results/d_e_build_inputs
git archive --format=zip --output=results/d_e_build_inputs/published_evaluation.zip e280f87b7696dc8a0fbb48870691d83c0c458fe0 results/d_with_knn_v1 results/c_v1_20261106/freeze.json reports/a_knn_experiment.md
Invoke-WebRequest -Uri https://reactome.org/download/current/ReactomePathways.gmt.zip -OutFile results/d_e_build_inputs/ReactomePathways.gmt.zip
python scripts/build_d_e_delivery.py --work results/d_e_build_inputs
```

构建脚本会核验下载 GMT 的固定哈希；若 `current` 已更新，应从 E 的既有交付或官方归档取得上述同哈希版本。输出目录分别为 `D_E_案例与通路分析交付/`（D 内部完整构建）和 `E_通路分析案例包_20261124/`（E 专用），各有同名 ZIP。

## 数据与验证边界

本次使用现有 18 个真实案例，按用户确认不再向 B 索取；案例真实变化已并入 E 专用包。完整 B 训练/验证/测试矩阵和控制细胞划分未随案例交付，本轮不能从完整训练标签再次核验训练均值，不能把案例子集当成完整测试标签。

本次对 54 个“案例×方法”指标重新打分，对已发布完整逐目标指标复做统计汇总，未声称本轮全量真值重算。原 X 的归一化/log1p 来源尚未确认，保持原表达单位。

已完成原发布文件哈希核验、三份输出基因冻结哈希核验、候选抽取规则复核、案例 MSE 对照、全量发布 MSE 汇总一致性检查，以及超几何/BH 的已知小例子与输出范围检查。无表示回退预测仅在与保存的训练均值相差不超过 1e-12 时归一至该均值，避免浮点噪声造成伪残差相关；原始案例 NPZ 保留不改。
