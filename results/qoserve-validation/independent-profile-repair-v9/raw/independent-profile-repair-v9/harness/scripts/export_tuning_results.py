"""Export immutable summaries after a campaign has finished writing raw JSONL."""
import collections,hashlib,json,os,statistics
from pathlib import Path
root=Path(os.environ['INFERX_ROOT'])
rows=[]
for p in sorted((root/'tuning').glob('*.jsonl')):
 for line in p.read_text().splitlines():
  j=json.loads(line)
  if j.get('kind')!='trial':continue
  s=j['summary']
  row=dict(file=p.name,config=j['config'],prompt_id=j['prompt_id'],output_budget=j['request']['max_tokens'],repetition=j['repetition'],concurrency=j['concurrency'],index=j['index'],wave_s=j['wave_s'],request_sha256=j['request_sha256'],text_sha256=hashlib.sha256(s['text'].encode()).hexdigest(),**{k:v for k,v in s.items() if k not in ('text','reasoning','tool_calls')})
  rows.append(row)
(root/'tuning/trials-compact.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
groups=collections.defaultdict(list)
for r in rows:groups[(r['config'],r['prompt_id'],r['cache_mode'],r['concurrency'],r['file'],r['output_budget'])].append(r)
out=[]
for key,rs in sorted(groups.items()):
 good=[r for r in rs if r['ok']]
 d=dict(config=key[0],prompt_id=key[1],cache_mode=key[2],concurrency=key[3],file=key[4],output_budget=key[5],n=len(rs),successful=len(good),truncated=sum(r['truncated'] for r in rs))
 for metric in ('observed_decode_tps','e2e_tps','ttft_s','wall_s','cached_tokens'):
  values=[r[metric] for r in good if r[metric] is not None]
  d[metric]=dict(median=statistics.median(values),min=min(values),max=max(values)) if values else None
 out.append(d)
(root/'tuning/summary-current.json').write_text(json.dumps(out,indent=2))
print(len(rows),'trials,',len(out),'groups')
