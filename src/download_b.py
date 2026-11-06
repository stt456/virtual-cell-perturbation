import pathlib,time,requests,zipfile,hashlib
p=pathlib.Path(__file__).resolve().parents[1]; out=p/'data/replogle_k562_essential.zip'; out.parent.mkdir(exist_ok=True); u='https://dataverse.harvard.edu/api/access/datafile/7458695?fresh='+str(time.time_ns())
with requests.get(u,stream=True,timeout=(30,120)) as r:
 r.raise_for_status()
 with out.open('wb') as f:
  for chunk in r.iter_content(1048576):
   if chunk: f.write(chunk)
with zipfile.ZipFile(out) as z:
 for name in z.namelist():
  target=(out.parent/name).resolve()
  if out.parent.resolve() not in target.parents and target!=out.parent.resolve(): raise ValueError('Unsafe ZIP path')
 z.extractall(out.parent)
print('Downloaded and extracted:',out)
