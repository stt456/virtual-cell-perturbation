"""Build a reproducible D/E case handoff from the published frozen case matrices."""
import argparse
import csv
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import zipfile
import importlib.metadata

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR if (SCRIPT_DIR/'src').is_dir() else SCRIPT_DIR.parent
sys.path.insert(0, str(REPO))
from src.evaluation.metrics import evaluate, top_indices
from src.evaluation.statistics import summarize, across_seeds

EXPECTED_GMT = 'a9feaa3ae9bbb746422cba77a669813de627fb15b7d7478b67c8d57020ec26ae'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False, encoding='utf-8-sig')


def unpack(archive, folder):
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if not (folder / name).resolve().is_relative_to(folder.resolve()):
                raise ValueError('Unsafe archive path')
        z.extractall(folder)


@lru_cache(maxsize=None)
def hypergeom_tail(M, K, n, k):
    # Exact integer combinations avoid underflow for the small Top-20 lists.
    if k == 0:
        return 1.0
    den = math.comb(M, n)
    return min(1.0, sum(math.comb(K, j)*math.comb(M-K, n-j)
                       for j in range(max(k, n-(M-K)), min(n, K)+1))/den)


def bh(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p, kind='stable')
    q = np.minimum.accumulate((p[order]*len(p)/np.arange(1, len(p)+1))[::-1])[::-1]
    result = np.empty(len(p)); result[order] = np.minimum(q, 1)
    return result


def markdown_table(frame, columns):
    names = list(columns)
    lines = ['| ' + ' | '.join(columns.values()) + ' |', '| ' + ' | '.join(['---']*len(names)) + ' |']
    for _, row in frame.iterrows():
        cells=[]
        for k in names:
            x=row[k]
            cells.append('NA' if pd.isna(x) else f'{x:.6g}' if isinstance(x,(float,np.floating)) else str(x))
        lines.append('| '+' | '.join(cells)+' |')
    return '\n'.join(lines)


def write_report(out, frame, across, comparison, coverage, choices):
    comp=pd.DataFrame(comparison)
    model_rows=frame[frame.method=='model'].copy()
    mean_rows=frame[frame.method=='train_mean'].set_index(['seed','target_gene'])
    model_rows['baseline_mse']=[mean_rows.loc[(r.seed,r.target_gene),'mse'] for r in model_rows.itertuples()]
    model_rows['pct_improvement']=100*model_rows.mse_improvement_over_train_mean/model_rows.baseline_mse
    model_rows['self_gene_covered']=model_rows.self_coordinate_present
    write_csv(out/'tables/case_model_vs_baseline.csv',model_rows.to_dict('records'))
    full=across[(across.split=='test')&(across.subset=='all')].copy()
    baseline=float(full.loc[full.model=='train_mean','mse_mean'].iloc[0])
    full['improvement_pct']=100*(baseline-full.mse_mean)/baseline
    full_table=markdown_table(full,{'model':'方法','mse_mean':'MSE 均值','mse_std':'种子间样本标准差',
        'pearson_mean':'Pearson','top20_overlap_mean':'Top-20 重合率','improvement_pct':'相对均值基线改善 %'})
    case_table=markdown_table(model_rows,{'seed':'种子','target_gene':'目标','model':'模型',
        'used_fallback':'回退','mse':'MSE','baseline_mse':'均值基线 MSE','pct_improvement':'改善 %','pearson':'Pearson'})
    pathway_table=markdown_table(comp,{'seed':'种子','target_gene':'目标','top20_intersection_n':'Top-20 交集数',
        'truth_significant_n':'真实显著通路数','model_significant_n':'预测显著通路数',
        'baseline_significant_n':'基线显著通路数','shared_truth_model_n':'真实/预测显著交集'})
    bg_table=markdown_table(pd.DataFrame(coverage),{'seed':'种子','background_n':'背景基因数',
        'annotated_n':'有注释','unannotated_n':'无注释','tested_pathways':'参与检验通路数'})
    selection_text='''# 案例选择与背景基因规则（本次交付采用）

本包保留 2026-11-16 已发布的 18 个案例；2026-11-24 仅核验原规则并分析，不按本次误差或富集结果换案例。属于事后展示分析，不声称测试前预注册。

1. 对种子 42、123、2026 分别使用原冻结验证集选出的模型；对应模型为 MLP 均值/检出率、Ridge 均值/检出率、MLP 均值/检出率。
2. 在每个种子完整测试目标清单中，按 `feature_available` 分成有表示与无表示两组。每组计算 UTF-8 字符串 `D_E_v1|{seed}|{target_gene}` 的 SHA-256，按十六进制哈希升序、目标符号字典序打破并列，各取前三名。
3. 本次用已发布逐目标表重建候选清单和哈希排序，核对所选目标与原案例包完全一致。完整候选、组内排名、哈希和选中标记见 `tables/case_selection_candidates.csv`。没有细胞数/质量标记原文件，不额外声称完成细胞质量重新验收。
4. 不因预测差、通路不显著或故事难解释替换案例。E 早期提出的 `E-case-v1|42|...` 前三规则与本次已发布规则不同，不能把两者混称为同一方案。
5. 种子 42 为首要展示组，123 和 2026 为补充；这些案例均不代表全测试集，也不合并种子做独立样本显著性推断。

背景：每个种子使用真实、模型预测与训练均值共同具有有限值的全部 2000 个输出基因，同一种子的所有案例与方法保持相同背景。包括无 Reactome 注释的输出基因，因为它们同样有机会进入 Top-20。三个种子各用自己的输出集合及顺序，不混为跨种子并集；不用全基因组、仅 Top-20、仅显著基因作背景。原接口提过“只保留有注释基因”，本包明确采用 E 生物学草案的“保留全部共同输出基因”口径，制定时间为本次分析阶段。

背景文件 `background/background_seed_{seed}.txt` 使用 LF 换行，文件 SHA-256 与 C 冻结的原输出基因列表完全一致。符号按原样精确匹配，不转大写、不补别名、不做静默合并；2000 个符号唯一。注释覆盖率另表报告。

真实值、模型、均值基线各自按绝对变化降序选 Top-20，并列按基因符号升序，不按真值为预测筛基因。主 ORA 使用不分方向的 Top-20；方向保存在基因表中。零值不计入上/下调，交集方向一致率仅在双方非零的交集上计算，无有效交集记 NA。Top-20 是幅度排序，不是经过显著性检验的 DEG。

Reactome GMT 与 E 原记录 SHA-256 完全一致，共 2874 行（团队记录为 Reactome 98）；下载日期 2026-11-24。通路先与背景取交集，只检验交集大小 10–500 的通路。单侧超几何检验后，在每个“种子 × 案例 × 方法”的全部保留通路内做 BH 校正，零交集通路也纳入且 p=1，以 q<0.05 标记显著。校正范围没有覆盖所有案例/方法联合检验，因此跨案例筛发现仅用于探索。
'''
    (out/'案例选择与背景说明.md').write_text(selection_text,encoding='utf-8')
    report=f'''# D 统一评测与 E 案例通路交付

日期：2026-11-24。对象：K562 CRISPRi 扰动的群体平均表达变化。按用户本次要求，使用已找到的现有数据，不再向 B 发消息索取。

## 本次完成与数据范围

本次从 GitHub main 提交 `e280f87b7696dc8a0fbb48870691d83c0c458fe0` 的 [PR #6](https://github.com/stt456/virtual-cell-perturbation/pull/6) 取回案例包，实际读取 18 个案例的真实变化与预测，独立重算模型、训练均值和零变化三种方法的 54 行逐案例指标，生成 36000 行全输出基因比较表、1080 行 Top-20 基因表、三张真实/预测散点图，并计算 Reactome ORA。

原始包每种子含 6×2000 的 `delta_true` 和 `delta_pred`，以及 `train_mean`、目标/基因顺序和覆盖标记。案例真实变化已经并入 CSV 和 NPZ，直接交给 E 即可。`delta_true` 表示扰动均值减对应对照均值，保持源 X 表达空间；原始归一化/log1p 来源尚未确认，不能改写为已确认的原始计数或 log2 fold change。

同时对已发布的完整逐目标指标重新做种子及跨种子汇总；它们来自 11 月 16 日的独立评测记录，本次没有取得完整 B 训练/验证/测试标签矩阵，因此不把全量结果称为本次从全量真值重新打分。完整 B 版本索引保留在 `source/results/c_v1_20261106/freeze.json` 和 `tables/B_version_inventory.csv`。从案例恢复出的三份输出基因列表已与 C 冻结哈希逐字节匹配；其余完整标签和对照划分不由案例子集替代。

## D 统一评测口径与完整结果复汇总

MSE 先对输出基因平均、再对目标等权平均；Pearson 与 Spearman 逐目标计算，常数向量相关记 NA。Top-20 按绝对变化和符号并列规则选择；残差相关在真实与预测同时减去训练均值后计算。跨种子均值与样本标准差仅描述三套实验，不把重叠目标当作独立样本。原完整测试每种子 164 目标、训练 760、验证 162；这些数量取自发布记录。

{full_table}

完整方法表、分表示可用/缺失子集、响应强度分层和验证集结果见 `tables/published_metrics_reaggregated_*.csv`。原发布配对 bootstrap 表位于 `source/results/d_with_knn_v1/paired_bootstrap.csv`：按种子、同目标配对、10000 次重采样；不从这 18 个事后案例推断全测试集优势。kNN 全体测试 MSE 最低，但不据测试排名重新选参数；C 模型案例保持验证选择结果。种子 123 的某些改善区间跨零，不能宣称所有划分稳定显著优于均值基线。

## 案例规则与逐案例比较

规则为每种子按有/无表示分层后各取 SHA-256 排序前三；具体字符串、全部候选排名、模型和事后选择标记见 [案例选择与背景说明](案例选择与背景说明.md)。已用完整发布测试目标表复核排序，与原案例包一致。表中改善为 `(基线 MSE−模型 MSE)/基线 MSE`；正值表示误差更小。

{case_table}

9 个无表示案例使用训练均值回退，不能从其结果推断模型学到了目标特异响应。对已声明回退且与训练均值相差不超过 1e-12 的预测，重算残差前只做浮点精度归一到已保存训练均值，避免理论零向量产生伪相关；原始 NPZ 和发布清单保留。案例 MSE 与发布记录的最大差异为约 9.7×10⁻¹⁷。训练均值在本次来自案例包，未从完整训练标签再次独立重算。

![种子 42 的真实与预测变化](figures/cases_seed_42.png)

## 通路分析背景与参数

本包采用每种子的全部 2000 个共同输出基因作为背景，保留未注释基因；不是全基因组或 Top-20 自身。背景文件与 C 原基因列表哈希完全一致。

{bg_table}

Reactome GMT 文件 SHA-256 为 `{EXPECTED_GMT}`，与 E 早期记录完全一致。真实、模型和训练均值各自取绝对变化 Top-20，通路与背景交集大小保留 10–500，做单侧超几何检验；每案例/方法的全部保留通路做 BH 校正，q<0.05 标记显著。本次共有 23238 个通路检验行，包含重复方法/案例的背景检验，不代表 23238 个独立通路。

## 通路结果与解释

{pathway_table}

完整检验结果共 23238 行，其中 94 行达到各列表内 BH q<0.05，包含不同方法和回退案例重复的同一通路。18 个案例中，真实与模型各自显著通路的交集均为 0。该现象只说明在本 Top-20/背景/阈值下未出现双方显著通路重叠，不能据此证明所有通路预测失败；也不能仅凭一方显著另一方不显著声称二者存在显著差异。

例如，种子 2026 的 RAD21 模型列表出现 31 个显著条目而真实列表为 0；种子 123 的无表示案例复制训练均值，所以其显著结果与基线相同，不能当作目标特异通路被准确恢复。Top-20 列表较短，显著性受注释和共同背景限制；相关通路共享基因，也不能把多个显著条目解释为独立机制。当前 ORA 不分上下调，结果不等于通路激活或抑制，更不是因果证明。

完整检验及重叠基因见 `tables/pathway_results_all.csv`；显著行见 `tables/pathway_results_significant.csv`；逐案例比较见 `tables/pathway_comparison.csv`。上下调和真实/预测数值见 `tables/top20_gene_lists.csv`，不把幅度 Top-20 称作统计学 DEG。

## 文件使用与复现

- `cases/cases_seed_*.npz`：原案例真值/预测、训练均值及目标/输出顺序，使用 numpy 的 `allow_pickle=False` 读取。
- `tables/case_changes_all_genes.csv`：36000 行，每行是一个种子/扰动目标/输出基因，已并列真实变化、预测和均值基线。
- `tables/case_metrics_recomputed.csv` 与 `case_model_vs_baseline.csv`：本次重新计算的逐案例指标及基线差值。
- `tables/top20_gene_lists.csv`：真实、模型、基线各自的 Top-20、方向及对应真实/预测值。
- `background/background_seed_*.txt`：保持原顺序的完整基因背景；`gene_annotation_seed_*.csv` 记录精确符号匹配覆盖。
- `source/`：从 Git 取回且未经改写的原发布记录；其中存在旧 handoff 模板措辞，以本报告说明的实际执行范围及文件为准。
- `manifest.json`：本包逐文件哈希；`tables/source_integrity.csv`：发布文件及背景基因哈希核验；`tables/case_vs_published_metric_check.csv`：重算差异。
- `code/`：本次构建脚本与引用的 D 指标函数；`requirements.txt`：实际使用的依赖版本。

在当前项目环境执行 `python scripts/build_d_e_delivery.py --work ../results/D_E_delivery_20261124` 可生成分析产物。压缩包内的 `inputs/` 含两份构建输入，解压后执行 `python code/build_d_e_delivery.py --work inputs`；新结果生成到 inputs 子目录，不覆盖原交付。Python 版本记录在 manifest 中。

本次没有联系 B；按用户确认使用现有数据完成本次案例交付。全量训练/验证/测试矩阵和原始控制细胞划分仍未包含在此案例包中，不能用这 18 个案例冒充完整 B 交付。

## 方法来源

- [Reactome 官方下载](https://reactome.org/download-data)：基因集来源；本次已核对与 E 记录一致的文件哈希，未依赖 current 链接长期不变。
- [GO 官方背景集合说明](https://geneontology.org/docs/go-enrichment-analysis/)：背景应覆盖可能进入选中列表的基因；本包据此采用全部共同输出基因。
- 团队方案：`reports/e_nov08_biology.md`；模型/评测记录：[仓库 PR #6](https://github.com/stt456/virtual-cell-perturbation/pull/6)。
'''
    (out/'D_统一评测与案例通路交付.md').write_text(report,encoding='utf-8')
    frozen=json.loads((out/'source/results/c_v1_20261106/freeze.json').read_text(encoding='utf-8'))
    inventory=[]
    for seed in (42,123,2026):
        for name in ('labels_train_val.npz','split.json','output_genes.txt'):
            path=f'data/processed/c_inputs/seed_{seed}/{name}'
            inventory.append(dict(seed=seed,file=path,expected_sha256=frozen['external_input_hashes'][path],
                local_status='recovered_exact_from_case_gene_ids' if name=='output_genes.txt' else 'full_file_not_in_case_package'))
        inventory.append(dict(seed=seed,file=f'data/processed/seed_{seed}/D_test.npz',
            expected_sha256=frozen['test_label_expected_sha256'][str(seed)],local_status='18_case_subset_only_6_per_seed'))
    write_csv(out/'tables/B_version_inventory.csv',inventory)


def run(args):
    work = args.work.resolve()
    out = work / 'D_E_案例与通路分析交付'
    out.mkdir(exist_ok=True)
    for name in ('source', 'cases', 'background', 'tables', 'figures', 'code'):
        (out/name).mkdir(exist_ok=True)
    unpack(work/'published_evaluation.zip', out/'source')
    published = out/'source/results/d_with_knn_v1'
    old_manifest = json.loads((published/'manifest.json').read_text(encoding='utf-8'))
    checks = []
    for relative, expected in old_manifest['files'].items():
        p = published/relative.replace('\\', '/')
        assert p.is_file() and digest(p) == expected, str(p)
        checks.append(dict(file=p.relative_to(out).as_posix(), sha256=digest(p), passed=True))
    with zipfile.ZipFile(work/'ReactomePathways.gmt.zip') as z:
        gmt = z.read('ReactomePathways.gmt')
    assert hashlib.sha256(gmt).hexdigest() == EXPECTED_GMT, 'Reactome version mismatch'
    (out/'background/ReactomePathways.gmt').write_bytes(gmt)
    pathways = {}
    for line in gmt.decode().splitlines():
        name, pid, *genes = line.split('\t')
        assert pid not in pathways
        pathways[pid] = (name, set(genes))
    annotated = set().union(*(x[1] for x in pathways.values()))
    selected = json.loads((published/'E_handoff/case_selection.json').read_text(encoding='utf-8'))
    published_metrics = pd.read_csv(published/'per_target.csv')
    metrics, top_rows, change_rows, ora_rows, comparison, candidates = [], [], [], [], [], []
    coverage, metric_checks = [], []
    for seed_text, choice in selected['choices'].items():
        seed = int(seed_text); model = choice['model']
        path = published/f'E_handoff/cases_seed_{seed}.npz'
        with np.load(path, allow_pickle=False) as z:
            a = {k:z[k] for k in z.files}
        genes, targets = a['gene_ids'].astype(str), a['target_ids'].astype(str)
        assert len(set(genes)) == len(genes) == 2000
        assert targets.tolist() == choice['targets']
        assert a['delta_true'].shape == a['delta_pred'].shape == (6, 2000)
        assert all(np.isfinite(a[k]).all() for k in ('delta_true','delta_pred','train_mean'))
        assert np.array_equal(a['used_fallback'], ~a['feature_available'])
        # Confirm the original selection against the whole published test population.
        all_targets = published_metrics[(published_metrics.seed == seed) &
            (published_metrics.split == 'test') & (published_metrics.model == model)]
        chosen = []
        for flag in (True, False):
            pool = all_targets.loc[all_targets.feature_available == flag, 'target_gene'].tolist()
            pool.sort(key=lambda t: (hashlib.sha256(f'D_E_v1|{seed}|{t}'.encode()).hexdigest(), t))
            chosen.extend(pool[:3])
            for rank, t in enumerate(pool, 1):
                candidates.append(dict(seed=seed, target_gene=t, feature_available=flag,
                    hash=hashlib.sha256(f'D_E_v1|{seed}|{t}'.encode()).hexdigest(),
                    rank_in_stratum=rank, selected=rank <= 3))
        assert chosen == targets.tolist(), 'Published case selection differs from declared rule'
        np.savez_compressed(out/f'cases/cases_seed_{seed}.npz', **a)
        background = set(genes)
        gene_path = out/f'background/background_seed_{seed}.txt'
        gene_path.write_bytes(('\n'.join(genes)+'\n').encode('utf-8'))
        frozen = json.loads((out/'source/results/c_v1_20261106/freeze.json').read_text(encoding='utf-8'))
        gene_hash = frozen['external_input_hashes'][f'data/processed/c_inputs/seed_{seed}/output_genes.txt']
        assert digest(gene_path) == gene_hash, 'Recovered output gene order differs from C freeze'
        checks.append(dict(file=gene_path.relative_to(out).as_posix(),sha256=gene_hash,passed=True))
        write_csv(out/f'background/gene_annotation_seed_{seed}.csv',
            [dict(gene_id=g, reactome_annotated=g in annotated) for g in genes])
        retained = {pid:(name, gs & background) for pid,(name,gs) in pathways.items()
                    if 10 <= len(gs & background) <= 500}
        coverage.append(dict(seed=seed, background_n=len(genes), annotated_n=len(background & annotated),
            unannotated_n=len(background-annotated), tested_pathways=len(retained)))
        matrices = {'model':a['delta_pred'], 'train_mean':np.tile(a['train_mean'], (6,1)),
                    'zero':np.zeros((6,2000))}
        # Canonicalize only declared mean fallbacks within machine-precision tolerance.
        for flag_index in np.flatnonzero(a['used_fallback']):
            assert np.allclose(matrices['model'][flag_index],a['train_mean'],rtol=0,atol=1e-12)
            matrices['model'][flag_index] = a['train_mean']
        for method, prediction in matrices.items():
            rows = evaluate(prediction, a['delta_true'], a['train_mean'], targets, genes,
                a['feature_available'], a['used_fallback'] if method=='model' else np.zeros(6,bool),
                model if method=='model' else method, seed)
            for row in rows:
                row['method'] = method
                row['scope'] = 'recomputed_case_subset'
                row['mse_improvement_over_train_mean'] = float(np.mean((a['train_mean']-
                    a['delta_true'][targets.tolist().index(row['target_gene'])])**2))-row['mse']
            metrics.extend(rows)
        for i,t in enumerate(targets):
            orig = published_metrics[(published_metrics.seed==seed)&(published_metrics.split=='test')&
                (published_metrics.model==model)&(published_metrics.target_gene==t)].iloc[0]
            current = next(r for r in metrics if r['seed']==seed and r['target_gene']==t and r['method']=='model')
            err=abs(current['mse']-float(orig.mse)); assert err < 1e-12
            metric_checks.append(dict(seed=seed,target_gene=t,mse_difference_from_published=err))
            for j,g in enumerate(genes):
                change_rows.append(dict(seed=seed,target_gene=t,gene_id=g,model=model,
                    delta_true=a['delta_true'][i,j],delta_pred=matrices['model'][i,j],
                    train_mean=a['train_mean'][j],feature_available=bool(a['feature_available'][i]),
                    used_fallback=bool(a['used_fallback'][i])))
            vector_sets = {'truth':a['delta_true'][i],'model':matrices['model'][i],
                           'train_mean':a['train_mean']}
            significant = {}
            for method, vector in vector_sets.items():
                top = top_indices(vector, genes,20); names=set(genes[top]); significant[method]=set()
                for rank,j in enumerate(top,1):
                    top_rows.append(dict(seed=seed,target_gene=t,method=method,model=model,rank=rank,
                        gene_id=genes[j],delta=vector[j],direction='up' if vector[j]>0 else 'down' if vector[j]<0 else 'zero',
                        delta_true=a['delta_true'][i,j],delta_pred=matrices['model'][i,j],train_mean=a['train_mean'][j]))
                current_ora=[]
                for pid,(name,gs) in retained.items():
                    overlap=names&gs; k=len(overlap)
                    current_ora.append(dict(seed=seed,target_gene=t,method=method,model=model,list_type='top20_abs',
                        pathway_id=pid,pathway_name=name,background_n=2000,list_n=20,pathway_n=len(gs),
                        overlap_n=k,overlap_genes=';'.join(sorted(overlap)),p_value=hypergeom_tail(2000,len(gs),20,k)))
                q=bh([r['p_value'] for r in current_ora])
                for row,value in zip(current_ora,q):
                    row['p_adj']=value; row['significant']=bool(value<.05)
                    if value<.05: significant[method].add(row['pathway_id'])
                ora_rows.extend(current_ora)
            ti=set(top_indices(vector_sets['truth'],genes)); pi=set(top_indices(vector_sets['model'],genes))
            common=sorted(ti&pi); nonzero=[j for j in common if a['delta_true'][i,j]!=0 and matrices['model'][i,j]!=0]
            comparison.append(dict(seed=seed,target_gene=t,model=model,feature_available=bool(a['feature_available'][i]),
                top20_intersection_n=len(common),top20_overlap=len(common)/20,
                nonzero_intersection_n=len(nonzero),direction_agreement_nonzero=float(np.mean([
                    np.sign(a['delta_true'][i,j])==np.sign(matrices['model'][i,j]) for j in nonzero])) if nonzero else np.nan,
                truth_significant_n=len(significant['truth']),model_significant_n=len(significant['model']),
                baseline_significant_n=len(significant['train_mean']),
                shared_truth_model_n=len(significant['truth']&significant['model']),
                shared_truth_baseline_n=len(significant['truth']&significant['train_mean'])))
        fig,axs=plt.subplots(2,3,figsize=(12,7))
        for i,ax in enumerate(axs.flat):
            ax.scatter(a['delta_true'][i],matrices['model'][i],s=4,alpha=.35)
            bounds=[min(a['delta_true'][i].min(),matrices['model'][i].min()),max(a['delta_true'][i].max(),matrices['model'][i].max())]
            ax.plot(bounds,bounds,color='gray',lw=.8); ax.set_title(f'{targets[i]} | fallback={bool(a["used_fallback"][i])}')
            ax.set_xlabel('Observed delta'); ax.set_ylabel('Predicted delta')
        fig.suptitle(f'Seed {seed}: {model} | post-hoc cases'); fig.tight_layout()
        fig.savefig(out/f'figures/cases_seed_{seed}.png',dpi=160); plt.close(fig)
    frame=pd.DataFrame(metrics)
    write_csv(out/'tables/case_metrics_recomputed.csv',metrics)
    write_csv(out/'tables/case_changes_all_genes.csv',change_rows)
    write_csv(out/'tables/top20_gene_lists.csv',top_rows)
    write_csv(out/'tables/pathway_comparison.csv',comparison)
    write_csv(out/'tables/pathway_results_all.csv',ora_rows)
    write_csv(out/'tables/pathway_results_significant.csv',[r for r in ora_rows if r['significant']])
    write_csv(out/'tables/case_selection_candidates.csv',candidates)
    write_csv(out/'tables/background_coverage.csv',coverage)
    write_csv(out/'tables/case_vs_published_metric_check.csv',metric_checks)
    write_csv(out/'tables/source_integrity.csv',checks)
    # Reaggregate the full, previously published target metrics; no full-label rescore is claimed.
    summary=summarize(published_metrics, stratified=True)
    summary.to_csv(out/'tables/published_metrics_reaggregated_by_seed.csv',index=False,encoding='utf-8-sig')
    across=across_seeds(summary)
    across.to_csv(out/'tables/published_metrics_reaggregated_across_seeds.csv',index=False,encoding='utf-8-sig')
    original=pd.read_csv(published/'summary_across_seeds.csv')
    keys=['model','split','subset']; l=across.set_index(keys).sort_index(); r=original.set_index(keys).sort_index()
    assert l.index.equals(r.index)
    assert np.allclose(l['mse_mean'],r['mse_mean'],rtol=1e-10,atol=1e-13)
    if Path(__file__).resolve() != (out/'code/build_d_e_delivery.py').resolve():
        shutil.copy2(__file__,out/'code/build_d_e_delivery.py')
    for f in ('metrics.py','statistics.py','io.py','__init__.py'):
        (out/'code/src/evaluation').mkdir(parents=True,exist_ok=True)
        shutil.copy2(REPO/'src/evaluation'/f,out/'code/src/evaluation'/f)
    write_json(out/'analysis_summary.json',dict(n_cases=len(comparison),n_case_metric_rows=len(metrics),
        n_pathway_tests=len(ora_rows),n_significant=sum(r['significant'] for r in ora_rows),
        background=coverage,reactome_sha256=EXPECTED_GMT,case_metric_max_difference=max(r['mse_difference_from_published'] for r in metric_checks),
        selection_verified_from_published_population=True,full_B_labels_present=False,
        source_commit='e280f87b7696dc8a0fbb48870691d83c0c458fe0'))
    write_report(out, frame, across, comparison, coverage, selected)
    (out/'inputs').mkdir(exist_ok=True)
    for name in ('published_evaluation.zip','ReactomePathways.gmt.zip'):
        shutil.copy2(work/name,out/'inputs'/name)
    (out/'requirements.txt').write_text('\n'.join(f'{name}=={importlib.metadata.version(name)}'
        for name in ('numpy','pandas','matplotlib'))+'\n',encoding='utf-8')
    write_json(out/'manifest.json',dict(source_commit='e280f87b7696dc8a0fbb48870691d83c0c458fe0',
        python=sys.version,full_label_recalculation_this_run=False,case_recalculation_this_run=True,
        pathway_analysis_this_run=True,reactome_sha256=EXPECTED_GMT,
        files={p.relative_to(out).as_posix():digest(p) for p in sorted(out.rglob('*'))
               if p.is_file() and p != out/'manifest.json'}))
    archive=work/'D_E_案例与通路分析交付_20261124.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file(): z.write(p, p.relative_to(work).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    # E receives only the case analysis materials; repository/source archives stay in D's bundle.
    share = work/'E_通路分析案例包_20261124'
    share.mkdir(exist_ok=True)
    for name in ('cases','background','figures'):
        shutil.copytree(out/name,share/name,dirs_exist_ok=True)
    (share/'tables').mkdir(exist_ok=True)
    for name in ('case_changes_all_genes.csv','case_metrics_recomputed.csv','case_model_vs_baseline.csv',
                 'top20_gene_lists.csv','pathway_comparison.csv','pathway_results_all.csv',
                 'pathway_results_significant.csv','case_selection_candidates.csv','background_coverage.csv'):
        shutil.copy2(out/'tables'/name,share/'tables'/name)
    shutil.copy2(out/'案例选择与背景说明.md',share/'案例选择与背景说明.md')
    write_json(share/'case_selection.json',selected)
    (share/'给E_阅读说明.md').write_text('''# 给 E：通路分析案例包

本包为 2026-11-24 的 D→E 交付。包含三个种子各 6 个案例的真实变化、模型预测、训练均值及本次通路结果；共 18 个案例。直接阅读本说明、案例选择与背景说明，然后打开 tables 中的结果表。

## 文件清单

- cases/cases_seed_*.npz：目标顺序、输出基因顺序、真实变化 delta_true、模型预测 delta_pred、训练均值和回退标记。每包 6×2000，numpy 读取需 allow_pickle=False。
- tables/case_changes_all_genes.csv：36000 行完整真实/预测并列表，包含 seed、target_gene、gene_id、model、delta_true、delta_pred、train_mean 及覆盖标记。
- tables/case_model_vs_baseline.csv：18 个模型案例与均值基线的误差比较；case_metrics_recomputed.csv 同时给出模型、均值基线和零变化三种方法。
- tables/top20_gene_lists.csv：真实、模型、基线各自选取的幅度 Top-20、方向和数值。
- tables/pathway_results_all.csv：23238 行完整 Reactome 检验及 BH q 值；pathway_results_significant.csv 为其中 94 行列表内 q<0.05 的结果，包含跨案例重复条目。
- tables/pathway_comparison.csv：逐案例的真实/模型/基线显著通路数量与重叠；background_coverage.csv 记录注释覆盖。
- tables/case_selection_candidates.csv：依据已发布完整测试目标表复核的候选、分层、哈希、排名和选中标记。
- background/background_seed_*.txt：每种子完整、有序的 2000 个输出背景基因，哈希与 C 冻结列表完全一致；gene_annotation_seed_*.csv 标记注释情况；ReactomePathways.gmt 为与 E 原记录一致的基因集。
- figures/cases_seed_*.png：逐案例真实与预测变化散点图。
- manifest.json：本包逐文件哈希、来源提交和分析范围。

## 规则与结果要点

使用已发布的事后展示案例，每种子分别从有表示/无表示目标中按 SHA-256 排序各选 3 个；模型按原验证选择确定，不根据本次通路结果替换。详见案例选择与背景说明.md。种子 42/123/2026 对应模型分别为 mlp_mean_detection/ridge_mean_detection/mlp_mean_detection。

背景采用每种子真实、模型和训练均值共同覆盖的全部 2000 个输出基因，包括无 Reactome 注释的基因。分别按绝对变化取 Top-20，通路交背景后大小 10–500，单侧超几何检验，在每个案例/方法内 BH 校正，q<0.05。方向列可作辅助解释；Top-20 不能称作经统计检验的 DEG，ORA 不等于通路激活/抑制。

18 个案例在该阈值下没有真实和预测均显著的通路交集。一方显著另一方不显著不证明二者存在显著差异；无表示案例的模型预测回退到训练均值，不代表恢复了目标特异通路。

本次按用户要求使用现有数据，不再向 B 索取。案例真实变化已并入本包；本包不含完整 B 训练/验证/测试标签及原始对照细胞划分，不能替代全量标签包。源 X 的归一化/log1p 来源仍未确认，保留原表达单位，不改称 log2 fold change。

公开结果来源：https://github.com/stt456/virtual-cell-perturbation/pull/6 。通路方法依据：https://reactome.org/download-data 和 https://geneontology.org/docs/go-enrichment-analysis/ 。
''',encoding='utf-8')
    write_json(share/'manifest.json',dict(audience='E',source_commit='e280f87b7696dc8a0fbb48870691d83c0c458fe0',
        n_cases=18,seeds=[42,123,2026],full_B_labels_present=False,
        selection_timing='post-hoc illustrative',reactome_sha256=EXPECTED_GMT,
        files={p.relative_to(share).as_posix():digest(p) for p in sorted(share.rglob('*'))
               if p.is_file() and p != share/'manifest.json'}))
    with zipfile.ZipFile(work/'E_通路分析案例包_20261124.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(share.rglob('*')):
            if p.is_file(): z.write(p,p.relative_to(work).as_posix())
    print(json.dumps(json.loads((out/'analysis_summary.json').read_text(encoding='utf-8')),ensure_ascii=False,indent=2))
    print('OUT',out)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--work',type=Path,required=True)
    run(parser.parse_args())
