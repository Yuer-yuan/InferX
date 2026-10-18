import pathlib,json,hashlib,random,struct
import pyarrow.parquet as pq
from tokenizers import Tokenizer
R=pathlib.Path(__file__).resolve().parent;P=pathlib.Path('/workspace/InferX/models/tokenizer/tokenizer.json')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(P)=='0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3'
meta=json.loads((R/'dataset-identity.json').read_text());tok=Tokenizer.from_file(str(P));payloads=[];cohort=[]
for split,count,seed in [('train',4,20261017),('validation',2,20261018)]:
 p=R/(split+'.parquet');f=next(v for v in meta['files'] if v['split']==split);assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256'];table=pq.read_table(p,columns=['text']);texts=table.column('text').to_pylist();text='\n\n'.join(texts);ids=tok.encode(text,add_special_tokens=False).ids;assert len(ids)>8192*count;rng=random.Random(seed);windows=[]
 while len(windows)<count:
  start=rng.randrange(0,len(ids)-8192+1)
  if all(abs(start-z)>=8192 for z in windows):windows.append(start)
 for i,start in enumerate(windows):
  prompt=ids[start:start+8192];name=f'{split}-{i}';payload=dict(model='qwen38-27b',prompt=prompt,max_tokens=1,ignore_eos=True,temperature=0,seed=42,cache_salt='ucloud2-calibration014:'+name,stream=True,stream_options={'include_usage':True});payloads.append(dict(name=name,split=split,payload=payload));cohort.append(dict(name=name,split=split,token_offset=start,length=8192,seed=seed,prompt_I32_sha256=hashlib.sha256(struct.pack('<8192I',*prompt)).hexdigest(),prompt_JSON_sha256=hashlib.sha256(json.dumps(prompt,separators=(',',':')).encode()).hexdigest(),payload_sha256=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest(),whole_split_text_sha256=hashlib.sha256(text.encode()).hexdigest(),whole_split_token_count=len(ids)))
(R/'requests.json').write_text(json.dumps(payloads,ensure_ascii=False)+'\n');(R/'input-identity.json').write_text(json.dumps(dict(dataset=meta,tokenizer_path=str(P),tokenizer_sha256=sha(P),add_special_tokens=False,rendering='raw token-ID completion, no chat template or answer labels',training_requests=4,validation_requests=2,cohort=cohort,capture_layers=[0,31,61,63],sampled_token_indices=list(range(0,2048,4)),train_chunks=list(range(16)),validation_chunks=list(range(16,24)),per_layer_training_samples=8192,per_layer_validation_samples=4096),indent=2)+'\n');print('CPU_inputs_ready',len(payloads),flush=True)
