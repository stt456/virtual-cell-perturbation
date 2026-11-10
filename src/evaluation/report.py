import base64
from datetime import date
import html
import json

import numpy as np
import pandas as pd

from .io import json_value
from .metrics import METRICS
from .plots import LABELS, ORDER


def write_reports(output, base, frozen, audit, frame, summary, across, bootstrap, coverage, origin):
    independent = origin.startswith("D_")
    source = "D 使用冻结的 B 标签独立计算" if independent else "D 对 C 自检逐目标记录的统计复算"
    main = across[(across.split == "test") & (across.subset == "all")].set_index("model")
    baseline = main.loc["train_mean", "mse_mean"]
    selected = pd.read_csv(output / "validation_selected_across_seeds.csv")
    selected = selected[(selected.split == "test") & (selected.subset == "all")].iloc[0]
    evaluation_date = date.today().isoformat()
    lines = ["# D 统一评测与结果比较", "", f"评测日期：{evaluation_date}。对象为 K562 CRISPRi 未见目标扰动的群体平均表达变化。", "",
             f"本轮结果来源为**{source}**。当前文件完整性核验、预测接口核验、指标程序和统计汇总已完成。" +
             ("已核对 B 标签版本并重算指标。" if independent else
              "B 的训练与测试标签、独立划分文件及 A 的参考预测按本轮要求保留为空；未完成基于 B 标签的独立指标验收。"), "",
             "现有结果显示，简单的训练平均扰动已解释相当一部分可预测变化；Ridge 和 MLP 的改进有限。"
             "目标表示缺失时的回退占比较高，因此需要同时看总体误差、表示可用子集和去平均扰动后的相关性。", "",
             "## 1 评测范围与证据", "",
             f"冻结实验为 `c_v1_20261106`，种子为 {frozen['seeds']}。独立核对 {audit['n_hashes_checked']} 项 SHA256，"
             "检查测试与验证预测的目标和基因顺序、矩阵形状、有限数值、表示掩码、回退值以及声明的目标划分。"
             "C 自检逐目标表与种子汇总、跨种子均值和标准差一致。文件哈希一致只能证明交接完整性，不能代替标签真实性和指标重算。", "",
             "每种子按 760 个训练目标、162 个验证目标和 164 个测试目标进行记录；同一目标未在该种子的训练、验证、测试声明中交叉。"
             "每种子输出 2000 个基因，同一种子内所有方法使用完全相同的顺序。输出基因随种子而变，跨种子结果是三套实验的描述性均值。"
             "未合并不同种子的重复目标进行独立样本推断。", "",
             "已纳入无变化、训练平均扰动、Ridge 的三种表示和 MLP 的三种表示，共 8 种方法、24 个测试预测包。"
             "A 的 kNN 或 GEARS 未提供，不计作零分，也不据此比较其性能。", "",
             "## 2 指标和统计口径", "",
             "- MSE：先在每个目标的输出基因上计算平方误差均值，再对目标等权平均，不按细胞数量加权。",
             "- 平均扰动基线：先得到每个训练扰动的平均变化，再对训练扰动等权平均；本轮仅核对 C 保存的训练均值及对应预测。独立模式会从 B 的训练标签重算并核验。",
             "- Pearson 和 Spearman：逐目标计算；Spearman 对并列数值使用平均秩。常数向量的相关系数为缺失，汇总同时给出有效目标数。",
             "- Top-20：按绝对变化降序、基因 ID 升序处理并列值，交集大小除以 20；方向一致率只在交集内计算，再对有效目标等权平均。全零预测的 Top-20 仅是并列值的确定性选取，其方向指标记缺失。",
             "- 扰动特异性：预测和真实变化都减去同一种子的训练平均扰动，再计算 Pearson 与 Spearman。两者同时减去同一向量不改变 MSE，所以不重复报告等价误差。",
             "- 辅助误差：若被干预目标本身出现在输出基因中，另报告剔除该坐标后的 MSE；当前 gene_ids 为基因符号，按精确 ID 匹配。",
             "- 三种子汇总：对每个种子的宏平均等权求均值，标准差采用样本标准差，ddof=1；它不是目标级误差条或置信区间。",
             f"- Bootstrap：每个种子内配对同一扰动目标，{int(bootstrap.bootstrap_repeats.iloc[0]):,} 次有放回重采样，给出均值差的百分位 95% 区间。正值表示基线 MSE 减去模型 MSE 后有改善；不同种子不合并。这些比较是描述性分析，未作多重比较显著性宣称。", "",
             "D 的指标实现不导入 C 的评测代码。handoff 模式只复算统计，指标值仍来自 C；independent 模式才重新读取 B 标签逐目标计算。", "",
             "## 3 模型比较", "",
             "以下 MSE 单位为源 X 表达单位的平方，相关系数为无量纲；数值为三种子均值 ± 样本标准差。", ""]
    for model in [m for m in ORDER if m in main.index] + [m for m in main.index if m not in ORDER]:
        row = main.loc[model]
        text = f"- **{LABELS.get(model, model)}**：MSE {row.mse_mean:.6f} ± {row.mse_std:.6f}"
        if np.isfinite(row.pearson_mean):
            text += f"；Pearson {row.pearson_mean:.4f}；Spearman {row.spearman_mean:.4f}"
        else:
            text += "；变化相关系数不可定义"
        if model != "zero":
            text += f"；Top-20 重合率 {100*row.top20_overlap_mean:.2f}%"
        if model not in ("zero", "train_mean"):
            text += f"；相对平均扰动 MSE 改善 {100*(baseline-row.mse_mean)/baseline:.2f}%"
        lines.append(text + "。")
    lines += ["", "![模型比较](figures/01_model_comparison.png)", "",
              "均值与检出率表示下的 Ridge 在总体 MSE 上表现较好；组合表示下的 Ridge 在残差相关性上较好。"
              "这说明不同指标考察的内容不同，不能据单一指标宣称某种模型全面最优。", "",
              "模型选择仍沿用测试前 C 冻结的、各自验证集选出的学习器：" +
              "；".join(f"seed {s} 选择 {frozen['validation_selected_learner_by_seed'][str(s)]}" for s in frozen['seeds']) +
              f"。该验证选择流程的测试 MSE 为 {selected.mse_mean:.6f} ± {selected.mse_std:.6f}，"
              f"相对训练平均扰动改善 {100*(baseline-selected.mse_mean)/baseline:.2f}%。"
              "固定模型表用于比较，不用测试排名重新选择参数。详细指标见 validation_selected_across_seeds.csv。", "",
              "## 4 Ridge 配对置信区间", ""]
    for model in ("ridge_mean_detection", "ridge_pca", "ridge_combined"):
        if model not in main.index:
            continue
        rows = bootstrap[(bootstrap.split == "test") & (bootstrap.subset == "all") & (bootstrap.model == model)]
        lines.append(f"**{LABELS[model]}** 相对训练平均扰动的 MSE 差值：")
        lines.append("")
        for row in rows.itertuples():
            qualifier = "区间跨 0" if row.ci95_low <= 0 <= row.ci95_high else "区间在 0 的同一侧"
            lines.append(f"- seed {row.seed}：改善 {row.mean_mse_improvement_over_train_mean:.6f}，"
                         f"95% CI [{row.ci95_low:.6f}, {row.ci95_high:.6f}]，{qualifier}。")
        lines.append("")
    lines += ["上述区间反映给定划分中目标抽样的不确定性；没有覆盖完整训练重运行、表达测量、不同细胞系和不同数据集的不确定性。", "",
              "![配对置信区间](figures/02_ridge_paired_bootstrap.png)", "",
              "## 5 表示覆盖与响应强度分层", ""]
    for row in coverage.itertuples():
        lines.append(f"- seed {row.seed}：表示可用 {row.n_available}/{row.n_test}（{row.coverage_pct:.2f}%），"
                     f"缺失 {row.n_unavailable} 个；六种可学习模型对缺失目标都严格回退至该种子的训练平均扰动。")
    lines += ["", "模型在缺失子集与平均扰动基线的预测完全相同，所以该子集的 MSE 改善严格为 0。"
              "残差相关性在这些回退目标上不可定义；可学习模型的残差相关有效目标数为各自可用目标数，"
              "而训练均值基线为 0。有效样本的变化必须与相关性数值一起阅读。", "",
              "![表示覆盖](figures/03_feature_coverage.png)", "",
              "响应强度定义为真实变化向量的 RMS，并按每种子内的 1/3 和 2/3 分位数划为弱、中、强响应。"
              "本轮 handoff 模式由 C 无变化基线逐目标 MSE 的平方根获得该强度，未据此重建真实表达矩阵。"
              "阈值、样本量、各层全部方法的误差和相关性均保存在结果表中；这种分层属于测试结果的事后描述，不用于重新调参。", "",
              "![响应强度分层](figures/04_response_strength.png)", "",
              "## 6 表示消融与解释边界", "",
              "PCA、均值与检出率、组合表示分别比较，保持同种子的目标、输出基因和评测口径一致。"
              "当前记录中，加入 PCA 的 MLP 总体 MSE 没有优于仅均值与检出率；Ridge 的组合表示提高残差相关性，"
              "但未同时取得最低总体 MSE。该比较反映本轮固定模型和表示，不证明 PCA 或非线性方法普遍无效。", "",
              "![表示消融](figures/05_representation_ablation.png)", "",
              "C 的运行元数据显示：源 X 的归一化与 log1p 状态未核实；上游 5000 基因已预选，选择过程的泄漏状态不明；"
              "缺少 guide 和 batch 字段，使用划分内混合对照。以上约束影响对表达单位、批次匹配和外部泛化的解释，"
              "本报告不将这些元数据声明视为 D 已完成的数据审计。MLP 本轮固定 dropout 和 weight decay，仅选择隐藏维度、学习率与早停轮次。", "",
              "## 7 交给 E 的案例与数据", "",
              "E_handoff 内包含每种子固定散列抽取的 3 个表示可用目标和 3 个缺失目标，使用各自验证集选出的模型。"
              "选择规则可复现，但制定时已有上游测试汇总可见，因此明确标注为事后展示案例，而非测试前预注册。"
              "case_metrics.csv 给出误差、相关性和相对基线改善；cases_seed_*.npz 含预测矩阵、目标顺序、基因顺序及训练均值。"
              + ("真实变化矩阵已一并输出。" if independent else
                 "真实变化矩阵保持缺失，NPZ 中没有 delta_true 键；Top-20 预测表的 delta_true 留空，不能据此生成真假散点图或真实通路富集。"), "",
              "posthoc_best_worst.csv 另列表示可用目标中按误差改善排序的较好与较差案例，仅用于探索性讨论。"
              "B 的细胞数量和注释未提供，本轮不进行细胞数与误差的关系分析；也不在缺少独立真值时给出 DEG 检验或校准指标。", "",
              "## 8 交付与复现", "",
              "主入口为 scripts/run_d_evaluation.py；指标与统计在 src/evaluation；自动检查在 tests/test_d_evaluation.py。"
              "docs/d_evaluation.md 提供运行方式、缺项处理、后续独立模式和 A 预测接口。manifest.json 记录命令、环境、源文件及产物哈希。", "",
              "结果表包含逐目标记录、每种子与跨种子汇总、有效指标数量、表示覆盖、响应强度分层、配对 bootstrap 和验证选择流程结果。"
              "A 与 B 的待补项在 pending_inputs.json 中保持 null，不作为模型得分。所有图均由本轮结果表生成，SVG 可编辑、PNG 可直接用于汇报。", ""]
    (output / "D_统一评测与结果比较.md").write_text("\n".join(lines), encoding="utf-8")

    payload = dict(origin=source, independent=independent,
                   across=across.to_dict("records"), summary=summary.to_dict("records"),
                   bootstrap=bootstrap.to_dict("records"), coverage=coverage.to_dict("records"),
                   labels=LABELS, order=ORDER, audit=audit)
    data = json.dumps(json_value(payload), ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    images = []
    for name, title in (("01_model_comparison", "模型比较"), ("02_ridge_paired_bootstrap", "Ridge 配对置信区间"),
                        ("03_feature_coverage", "表示覆盖"), ("04_response_strength", "响应强度分层"),
                        ("05_representation_ablation", "表示消融")):
        encoded = base64.b64encode((output / f"figures/{name}.png").read_bytes()).decode()
        images.append(f'<section><h2>{title}</h2><img alt="{title}" src="data:image/png;base64,{encoded}"></section>')
    template = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>D 统一评测与结果比较</title><style>
*{box-sizing:border-box}body{margin:0;background:#f5f6f7;color:#20252a;font:15px/1.7 'Microsoft YaHei',sans-serif}main{max-width:1180px;margin:auto;padding:40px 28px}h1{font-size:28px;margin:6px 0}h2{font-size:20px;margin:26px 0 12px}p{max-width:960px}.scope{color:#5b4830;border-left:3px solid #9b7c48;padding-left:16px}.stats{display:flex;gap:40px;padding:18px 0;border-bottom:1px solid #d1d6da}.stats strong{display:block;font-size:24px}.muted{color:#63707b;font-size:13px}section{margin:24px 0 38px}img{max-width:100%;height:auto;background:white}table{border-collapse:collapse;width:100%;background:white;font-size:13px}th,td{padding:10px 12px;border-bottom:1px solid #d9dfe3;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#edf0f2;white-space:nowrap}select{padding:8px 12px;border:1px solid #c2cbd1;border-radius:4px;background:white;font:inherit;margin-right:18px}label{display:inline-block}.scroll{overflow:auto}.good{color:#245773}.negative{color:#814d48}.legend{margin:10px 0;color:#596773}a{color:#245773}footer{border-top:1px solid #cbd1d5;padding:18px 0;color:#5f6a74;font-size:13px}@media(max-width:600px){main{padding:20px 14px}.stats{gap:20px;flex-wrap:wrap}h1{font-size:24px}}
</style><main><p class="muted">K562 CRISPRi · 未见目标扰动 · D 统计评测 · 2026-11-07</p>
<h1>统一评测与结果比较</h1><p class="scope" id="scope"></p>
<div class="stats"><div><strong>8</strong>已提供方法</div><div><strong>3 × 164</strong>种子 × 测试目标</div><div><strong>36.0–44.5%</strong>目标表示覆盖率</div></div>
<p>训练平均扰动是主要对照。模型的改善应同时结合覆盖率、有效相关目标数和目标级置信区间判断。</p>
<section><h2>模型指标交互表</h2><label>目标子集 <select id="subset"><option value="all">全部目标</option><option value="available">表示可用</option><option value="unavailable">表示缺失</option><option value="strength_weak">弱响应</option><option value="strength_medium">中等响应</option><option value="strength_strong">强响应</option></select></label>
<label>实验 <select id="seed"><option value="mean">三种子均值 ± 标准差</option><option>42</option><option>123</option><option>2026</option></select></label>
<p class="muted">MSE 越低越好，其余指标越高越好。方向一致率仅针对 Top-20 交集。— 为不可定义，不是 0。单种子相关系数旁括号为有效目标数。</p><div class="scroll"><table><thead id="head"></thead><tbody id="body"></tbody></table></div></section>
<section><h2>置信区间读法</h2><p>正值表示“平均扰动基线 MSE − 模型 MSE”。区间跨 0 表示该划分下平均改善的方向仍不确定；不将三个重叠划分合并为 492 个独立目标。</p></section>
__IMAGES__
<section><h2>缺项与 E 的使用范围</h2><p id="missing"></p><p>案例选择为可复现的事后展示；缺失目标的模型预测为训练平均扰动。当前结论适用于本次 K562 群体平均响应实验，不能外推到细胞级分布或其他细胞系。</p></section>
<footer>来源：C 冻结预测与自检记录；D 输出含独立代码、检查、统计复算、5 组 PNG/SVG、逐目标数据和 E 案例。详细定义与限制见同目录中文报告。此页面完全离线，无外部数据请求。</footer></main>
<script>const d=__DATA__;const sub=document.getElementById('subset'),seed=document.getElementById('seed');document.getElementById('scope').textContent=d.origin+'。'+(d.independent?'已核验 B 标签并重新计算指标。':'B 标签与 A 预测暂为空；这里的指标来自 C 自检逐目标记录，不代表 D 已完成独立标签验收。');document.getElementById('missing').textContent=d.independent?'E 的案例包包含真实与预测变化矩阵。':'B 训练标签、测试标签及 A 参考模型预测按要求保留为空，不计零分。E 案例包仅含预测与元数据，真实表达矩阵未提供，不能制作真实与预测散点图或真实通路富集。';
const metrics=[['mse','MSE'],['pearson','Pearson'],['spearman','Spearman'],['top20_overlap','Top-20 重合'],['top20_direction','交集方向一致'],['residual_pearson','残差 Pearson']];function f(v,k){if(v===null||v===undefined)return '—';return k.startsWith('top')?(100*v).toFixed(2)+'%':v.toFixed(k==='mse'?6:4)}function render(){const mean=seed.value==='mean';let rows=(mean?d.across:d.summary).filter(r=>r.split==='test'&&r.subset===sub.value&&(mean||String(r.seed)===seed.value));rows.sort((a,b)=>d.order.indexOf(a.model)-d.order.indexOf(b.model));document.getElementById('head').innerHTML='<tr><th>方法</th>'+metrics.map(([k,n])=>'<th>'+n+'</th>').join('')+'</tr>';document.getElementById('body').innerHTML=rows.map(r=>'<tr><td>'+(d.labels[r.model]||r.model)+'</td>'+metrics.map(([k,n])=>{const v=r[k+(mean?'_mean':'')],sd=r[k+'_std'];return '<td>'+f(v,k)+(mean&&v!==null?' ± '+f(sd,k):!mean&&k!=='mse'?' ('+r[k+'_n_valid']+')':'')+'</td>'}).join('')+'</tr>').join('')}sub.addEventListener('change',render);seed.addEventListener('change',render);render();</script></html>"""
    (output / "D_结果浏览.html").write_text(
        template.replace("__IMAGES__", "\n".join(images))
        .replace("__DATA__", data)
        .replace("2026-11-07", evaluation_date),
        encoding="utf-8",
    )
