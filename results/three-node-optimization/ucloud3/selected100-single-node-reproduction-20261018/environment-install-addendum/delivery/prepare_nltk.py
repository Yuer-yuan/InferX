from pathlib import Path
import json,hashlib,urllib.request,zipfile
R=Path(__file__).resolve().parent;root=R/'eval-envs/nltk_data';archives=root/'.archives';archives.mkdir(parents=True)
for item in json.loads((R/'nltk-resources.json').read_text()):
 p=archives/item['file']
 with urllib.request.urlopen(item['source'],timeout=120)as response:p.write_bytes(response.read())
 data=p.read_bytes();assert len(data)==item['bytes'];assert hashlib.sha256(data).hexdigest()==item['sha256']
 kind=item['source'].split('/packages/',1)[1].split('/',1)[0]
 with zipfile.ZipFile(p)as archive:
  for name in archive.namelist():assert not name.startswith('/')and '..'not in Path(name).parts
  archive.extractall(root/kind)
 print(json.dumps(dict(file=item['file'],sha256=hashlib.sha256(data).hexdigest(),verified=True)))
