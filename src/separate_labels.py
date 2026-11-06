import pathlib,numpy as np
p=pathlib.Path(__file__).resolve().parents[1]/'仓库接口交付_v0.1'
for seed in (42,123,2026):
 folder=p/f'data/processed/seed_{seed}'; z=np.load(folder/'aggregates.npz',allow_pickle=False)
 for name,mask in [('C_train_val',z['split']!='test'),('D_test',z['split']=='test')]:
  d={k:(z[k] if k=='gene_ids' else z[k][mask]) for k in z.files}; np.savez_compressed(folder/(name+'.npz'),**d)
  assert not any(d['split']=='test') if name=='C_train_val' else all(d['split']=='test')
 print(seed,'done')
