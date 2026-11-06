import pathlib,sys,json,hashlib,platform,importlib.metadata as md
import numpy as np,pandas as pd,anndata as ad
p=pathlib.Path(__file__).resolve().parents[1]; r=p/'results'; d=ad.read_h5ad(p/'data/replogle_k562_essential/perturb_processed.h5ad',backed='r'); v=d.var; obs=d.obs; t=pd.read_csv(r/'target_mapping.csv'); s0=pd.read_csv(r/'split_42.csv',keep_default_na=False); sym=set(v.gene_name.astype(str)); rows=[]; checks=[]
for seed in [42,123,2026]:
 s=pd.read_csv(r/f'split_{seed}.csv',keep_default_na=False); assert list(s.cell_id)==list(d.obs_names); means=pd.read_csv(r/f'mean_expression_{seed}.csv.gz',index_col=0); h=pd.read_csv(r/f'hvg_{seed}.csv'); counts=pd.read_csv(r/f'mean_counts_{seed}.csv').set_index('condition'); assert list(means.columns)==list(h.gene_id); assert len(set(h.gene_id))==2000
 sets=[]
 for part in ['train','val','test']:
  z=s[s['split']==part]; gs=sorted(set(z.target)-{'ctrl'}); sets.append(set(gs)); rows.append(dict(seed=seed,split=part,targets=len(gs),target_symbol_present=sum(g in sym for g in gs),target_symbol_absent=sum(g not in sym for g in gs),control_cells=int((z.target=='ctrl').sum()),perturbed_cells=int((z.target!='ctrl').sum())))
 assert not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2]
 sample=[('ctrl_'+x,(s['target']=='ctrl')&(s['split']==x)) for x in ['train','val','test']]; sample += [(g,s.target==g) for g in [sorted(sets[i])[0] for i in range(3)]]
 columns=d.var_names.get_indexer(h.gene_id); assert (columns>=0).all()
 for name,mask in sample:
  ids=np.flatnonzero(mask); acc=np.zeros(2000)
  for start in range(0,len(ids),256): acc+=np.asarray(d.X[ids[start:start+256],:][:,columns].sum(axis=0)).ravel()
  calculated=acc/len(ids); diff=float(np.max(np.abs(calculated-means.loc[name].to_numpy()))); assert diff<1e-5 and int(counts.loc[name,'n_cells'])==len(ids); checks.append(dict(seed=seed,condition=name,cells=len(ids),max_abs_error=diff))
pd.DataFrame(rows).to_csv(r/'coverage_and_controls_by_split.csv',index=False); pd.DataFrame(checks).to_csv(r/'aggregation_independent_checks.csv',index=False)
orig=s0; excluded=orig[orig['split']=='excluded'].copy(); excluded['reason']=np.where(excluded.target=='','unrecognized_condition','target_has_fewer_than_20_cells'); excluded.to_csv(r/'excluded_cells_reasons.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False)); print('excluded reasons',excluded.reason.value_counts().to_dict()); print('max error',max(x['max_abs_error'] for x in checks)); print('versions',platform.python_version(),{x:md.version(x) for x in ['anndata','numpy','pandas','scipy','h5py','matplotlib']})
