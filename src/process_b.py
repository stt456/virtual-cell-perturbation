import sys, json, pathlib, re
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import anndata as ad
from scipy.sparse import csr_matrix
P=pathlib.Path(__file__).resolve().parents[1]; data=P/'data/replogle_k562_essential/perturb_processed.h5ad'
d=ad.read_h5ad(data,backed='r'); obs=d.obs.copy(); var=d.var.copy(); labels=obs['condition'].astype(str)
def target(s):
 if s=='ctrl': return 'ctrl'
 parts=s.split('+')
 return parts[0] if len(parts)==2 and parts[1]=='ctrl' and parts[0] not in ('ctrl','') else None
obs['target']=labels.map(target); counts=obs.loc[obs.target.notna() & (obs.target!='ctrl'),'target'].value_counts(); valid=counts[counts>=20].index
keep=obs.target.eq('ctrl') | obs.target.isin(valid); obs['included']=keep; counts=counts.sort_values(ascending=False)
obs['target']=obs.target.fillna(''); symbols=var.gene_name.astype(str); n=d.n_vars
(P/'results').mkdir(exist_ok=True); (P/'figures').mkdir(exist_ok=True); (P/'reports').mkdir(exist_ok=True)
obs[['condition','control','target','included']].to_csv(P/'results/cell_metadata.csv'); var.to_csv(P/'results/gene_metadata.csv'); counts.rename('cell_count').to_csv(P/'results/target_counts.csv')
summary=[]
for seed in [42,123,2026]:
 rng=np.random.RandomState(seed); genes=np.array(sorted(valid)); rng.shuffle(genes); a=int(len(genes)*.7); b=a+int(len(genes)*.15)
 mapping={g:s for s,gs in [('train',genes[:a]),('val',genes[a:b]),('test',genes[b:])] for g in gs}
 split=np.full(len(obs),'excluded',dtype=object); targets=obs.target.to_numpy(); included=keep.to_numpy(); ctrl=np.flatnonzero((targets=='ctrl') & included); rng.shuffle(ctrl); c=int(len(ctrl)*.7); e=c+int(len(ctrl)*.15)
 for s,ids in [('train',ctrl[:c]),('val',ctrl[c:e]),('test',ctrl[e:])]: split[ids]=s
 for i in np.flatnonzero(included & (targets!='ctrl')): split[i]=mapping[targets[i]]
 pd.DataFrame({'cell_id':obs.index,'target':targets,'split':split}).to_csv(P/f'results/split_{seed}.csv',index=False)
 sets=[set(targets[(split==s)&(targets!='ctrl')]) for s in ['train','val','test']]; assert all(not sets[i]&sets[j] for i,j in [(0,1),(0,2),(1,2)])
 groups=sorted(set(targets[included])-{'ctrl'}); groups=['ctrl_train','ctrl_val','ctrl_test']+groups; groupid={g:i for i,g in enumerate(groups)}; sums=np.zeros((len(groups),n),dtype=np.float64); nums=np.zeros(len(groups),dtype=np.int64); sx=np.zeros(n); sx2=np.zeros(n); nt=0
 for start in range(0,d.n_obs,2048):
  end=min(start+2048,d.n_obs); x=d.X[start:end].tocsr(); ls=split[start:end]; ts=targets[start:end]; idx=np.flatnonzero(ls!='excluded')
  if len(idx):
   group=np.array([groupid['ctrl_'+ls[i] if ts[i]=='ctrl' else ts[i]] for i in idx]); assign=csr_matrix((np.ones(len(idx)),(group,idx)),shape=(len(groups),end-start)); sums+=assign@x; nums+=np.bincount(group,minlength=len(groups))
  ti=np.flatnonzero(ls=='train')
  if len(ti):
   t=x[ti]; sx+=np.asarray(t.sum(axis=0)).ravel(); sx2+=np.asarray(t.multiply(t).sum(axis=0)).ravel(); nt+=len(ti)
 mean=sx/nt; variance=np.maximum(sx2/nt-mean**2,0); dispersion=variance/(mean+1e-12)
 bins=pd.qcut(pd.Series(mean).rank(method='first'),20,labels=False,duplicates='drop'); score=np.zeros(n)
 for k in np.unique(bins):
  ids=np.flatnonzero(bins==k); v=np.log1p(dispersion[ids]); score[ids]=(v-v.mean())/(v.std()+1e-12)
 order=np.argsort(-score,kind='stable')[:2000]; pd.DataFrame({'gene_id':var.index[order], 'gene_name':symbols.iloc[order].values,'train_mean':mean[order],'train_variance':variance[order],'dispersion_score':score[order]}).to_csv(P/f'results/hvg_{seed}.csv',index=False)
 means=sums/np.maximum(nums[:,None],1); pd.DataFrame(means[:,order],index=groups,columns=var.index[order]).rename_axis('condition').to_csv(P/f'results/mean_expression_{seed}.csv.gz',compression='gzip'); pd.DataFrame({'condition':groups,'n_cells':nums}).to_csv(P/f'results/mean_counts_{seed}.csv',index=False)
 row={'seed':seed,'train_targets':len(sets[0]),'val_targets':len(sets[1]),'test_targets':len(sets[2]),'train_cells':int((split=='train').sum()),'val_cells':int((split=='val').sum()),'test_cells':int((split=='test').sum()),'hvg':len(order)}; summary.append(row); print(row,flush=True)
pd.DataFrame(summary).to_csv(P/'results/split_summary.csv',index=False)
fig,ax=plt.subplots(); ax.hist(counts.values,bins=40); ax.axvline(20,color='red'); ax.set(xlabel='Cells per perturbation',ylabel='Targets'); fig.tight_layout(); fig.savefig(P/'figures/target_cell_counts.png',dpi=160); plt.close(fig)
fig,ax=plt.subplots(); df=pd.DataFrame(summary).set_index('seed'); df[['train_targets','val_targets','test_targets']].plot.bar(ax=ax); ax.set_ylabel('Targets'); fig.tight_layout(); fig.savefig(P/'figures/split_targets.png',dpi=160); plt.close(fig)
fig,ax=plt.subplots(); df[['train_cells','val_cells','test_cells']].plot.bar(ax=ax); ax.set_ylabel('Cells'); fig.tight_layout(); fig.savefig(P/'figures/split_cells.png',dpi=160); plt.close(fig)
report={'shape':[d.n_obs,d.n_vars], 'conditions':int(labels.nunique()),'controls':int((targets=='ctrl').sum()),'kept_cells':int(included.sum()),'excluded_cells':int((~included).sum()),'targets_before_filter':len(counts),'targets_after_filter':len(valid),'duplicate_gene_symbols':int(symbols.duplicated().sum()),'missing_conditions':int(obs.condition.isna().sum()),'matrix':'sparse CSR; sampled values non-integer, consistent with transformed expression; normalization and log1p provenance not independently established','guide_batch':'not provided as standalone fields; barcode suffix is not assumed to be batch','hvg':'within each training split, rank variance/mean after log1p and mean-bin z-score; computed on input expression, not Scanpy Seurat HVG','upstream_leakage':'GEARS distribution already selected 5000 genes upstream; upstream HVG selection may use all cells. This prevents a strictly leakage-free benchmark.', 'split_summary':summary}
(P/'reports/data_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
