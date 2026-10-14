"""Export compact observations; raw prompts, conversations and diffs stay on HDD."""
import collections
import hashlib
import json
import os
import statistics
from pathlib import Path

root = Path(os.environ['INFERX_ROOT'])
directory = root/'context-agents'
raw_names = {f'trials-{client}.jsonl' for client in ['all', 'opencode', 'pi', 'claude']}
rows = [json.loads(l) for p in sorted(directory.iterdir()) if p.name in raw_names
        for l in p.read_text().splitlines() if l.strip()]
identities = set()
for row in rows:
    identity = (row['client'], row['task_id'])
    if identity in identities:
        raise ValueError(f'duplicate client/task identity: {identity}')
    identities.add(identity)
compact = [{k:v for k,v in row.items() if k not in {'server_requests','command','baseline'}} for row in rows]
(directory/'trials-compact.jsonl').write_text(''.join(json.dumps(j)+'\n' for j in compact))
summary = {}
for client in ['opencode','pi','claude']:
    selected = [j for j in compact if j['client']==client]
    summary[client] = dict(n=len(selected), statuses=dict(collections.Counter(j['status'] for j in selected)),
        wall_s_median=statistics.median(j['wall_s'] for j in selected) if selected else None,
        wall_s_total=sum(j.get('wall_s',0) for j in selected),
        completion_tokens=sum(j['server_completion_tokens'] for j in selected),
        max_prompt_tokens=max((j['server_max_prompt_tokens'] for j in selected),default=0),
        cached_fraction=sum(j['server_cached_tokens'] for j in selected)/max(1,sum(j['server_prompt_tokens'] for j in selected)))
contexts=[]
for capacity,kv in [(131072,'int8'),(262144,'rk4v4-e8')]:
    name=f'context-{capacity}-{kv}'
    result=json.loads((directory/(name+'-result.json')).read_text())
    probe=json.loads((directory/(name+'-lookup.jsonl')).read_text().splitlines()[0])['summary']
    heldout=[json.loads(l) for l in (directory/(name+'-heldout.jsonl')).read_text().splitlines()]
    rate={}
    for task in ['code_lru','code_toposort','reason_probability','prose_migration']:
        xs=[j['summary'] for j in heldout if j['prompt_id']==task]
        rate[task]=dict(n=len(xs), decode_median=statistics.median(x['observed_decode_tps'] for x in xs),
            decode_min=min(x['observed_decode_tps'] for x in xs),decode_max=max(x['observed_decode_tps'] for x in xs),
            truncated=sum(x['truncated'] for x in xs))
    contexts.append(dict(**result, prompt_tokens=probe['prompt_tokens'], ttft_s=probe['ttft_s'],
                         response_tokens=probe['completion_tokens'],heldout=rate))
output=dict(contexts=contexts,clients=summary,trial_count=len(compact),
    runtime=json.loads((directory/'runtime.json').read_text()),
    client_versions=json.loads((directory/'clients.json').read_text()),
    frozen_tasks=json.loads((directory/'tasks-manifest.json').read_text()))
(directory/'summary.json').write_text(json.dumps(output,indent=2))
files=[]
static=[p for p in directory.glob('context-*.jsonl')]
static += [p for p in directory.glob('context-*.log')]
static += [p for p in directory.glob('context-*-gpu.csv')]
static += list(directory.glob('smoke-*.jsonl'))
static += [p for p in directory.rglob('runs/*/events.jsonl')]
static += [p for p in directory.rglob('runs/*/changes.diff')]
for p in sorted(set(static)):
    files.append(dict(file=str(p.relative_to(directory)),bytes=p.stat().st_size,
                      sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(directory/'raw-manifest.json').write_text(json.dumps(dict(root=str(directory),files=files),indent=2))
print(json.dumps(summary,indent=2))
