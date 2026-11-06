import json,pathlib,sys,importlib.util
import numpy as np,pandas as pd,anndata as ad
P=pathlib.Path(__file__).resolve().parents[1]; O=P/'仓库接口交付_v0.1'; d=ad.read_h5ad(O/'data/processed/expression.h5ad',backed='r'); cm=pd.read_csv(O/'data/processed/cell_metadata.csv',keep_default_na=False); assert list(d.obs_names)==cm.cell_id.tolist() and len(d.obs)==162648 and d.n_vars==5000
checks=[]
for seed in (42,123,2026):
 s=json.loads((O/f'data/splits/seed_{seed}.json').read_text(encoding='utf-8')); folder=O/f'data/processed/seed_{seed}'; data=np.load(folder/'aggregates.npz',allow_pickle=False); g=(folder/'output_genes.txt').read_text(encoding='utf-8').splitlines(); assert len(g)==len(set(g))==2000 and data['gene_ids'].tolist()==g; assert len(set(data['target_ids']))==1086
 for key in ('mean_perturbed','mean_control','delta_true'): assert data[key].shape==(1086,2000) and np.isfinite(data[key]).all()
 assert np.allclose(data['delta_true'],data['mean_perturbed']-data['mean_control'],rtol=1e-12,atol=1e-12)
 ids=set(); ctrls=set()
 for split in ('train','val','test'):
  x=set(s['targets'][split]); c=set(s['control_cell_ids'][split]); assert not ids&x and not ctrls&c; ids|=x; ctrls|=c; assert data['target_ids'][data['split']==split].tolist()==s['targets'][split]; assert c<=set(cm.loc[cm.is_control==True,'cell_id'])
 assert len(ctrls)==10691; checks.append({'seed':seed,'targets':len(ids),'control_cells':len(ctrls),'genes':len(g),'npz_shape':list(data['delta_true'].shape)})
print(json.dumps(checks,ensure_ascii=False,indent=2)); (O/'validation.json').write_text(json.dumps({'schema':'0.1','checks':checks,'scope':'structural and numerical checks; does not establish normalization provenance or batch matching'},ensure_ascii=False,indent=2),encoding='utf-8')
