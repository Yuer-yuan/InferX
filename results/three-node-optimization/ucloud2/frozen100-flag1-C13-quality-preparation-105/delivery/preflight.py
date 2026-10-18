"""CPU-only original EvalScope adapter/defaults/cohort selfcheck, no model call."""
import hashlib
import importlib.metadata
import json
import os
import pathlib

ROOT=pathlib.Path(__file__).resolve().parent
os.environ.update(json.loads((ROOT/'client-environment.json').read_text()))
from evalscope.config import TaskConfig
from evalscope.api.registry import get_benchmark
import evalscope.benchmarks.ifbench.ifbench_adapter

assert importlib.metadata.version('evalscope')=='1.13.0'
raw=json.loads((ROOT/'task_config.json').read_text())
assert raw['generation_config']=={'max_tokens':32768} and raw['limit']==100 and raw['eval_batch_size']==13
config=TaskConfig(**{k:v for k,v in raw.items() if k!='api_key_env'},api_key='EMPTY')
resolved=config.model_dump(mode='json')
(ROOT/'expanded-defaults-preflight.json').write_text(json.dumps(resolved,ensure_ascii=False,indent=2)+'\n')
assert resolved['generation_config']['retries']==5
assert resolved['generation_config']['batch_size']==13
assert all(resolved['generation_config'].get(k) is None for k in ['temperature','top_p','top_k','seed','reasoning_effort'])
adapter=get_benchmark('ifbench',config)
data=adapter.load_dataset()
assert list(data.keys())==['default']
actual=[]
for index,sample in enumerate(list(data['default'])[:100]):
    actual.append(dict(index=index,messages=[dict(role=m.role,content=m.content) for m in sample.input],metadata=sample.metadata))
expected=json.loads(pathlib.Path('/workspace/InferX/framework-c4/sglang/ifbench100-input-identity.json').read_text())
assert actual==expected
digest=hashlib.sha256(json.dumps(actual,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
assert digest=='7f8b8adbf9b4234ac188cf62b52613e0068ca46eccaf1658cf550b234f4e4f04'
frozen=[json.loads(l) for l in (ROOT/'first100-frozen-cohort.jsonl').read_text().splitlines()]
def compact(v):
    if isinstance(v,dict):return {k:compact(a) for k,a in v.items() if a is not None}
    if isinstance(v,list):return [compact(a) for a in v]
    return v
for a,b in zip(actual,frozen):
    assert a['messages']==b['messages']
    m=a['metadata'];c=b['corpus']
    for k in ['key','prompt','instruction_id_list','kwargs']:assert compact(m[k])==compact(c[k])
instruction_count=sum(len(r['metadata']['instruction_id_list']) for r in actual)
assert len(actual)==100 and instruction_count==111
(ROOT/'adapter-first100-actual-inputs.json').write_text(json.dumps(actual,ensure_ascii=False,indent=2)+'\n')
result=dict(status='CPU_PREFLIGHT_PASS',samples=100,instructions=111,input_identity_sha256=digest,
    generation_config_explicit={'max_tokens':32768},resolved_generation_config=resolved['generation_config'],
    dataset_dir=config.dataset_dir,dataset_hub=config.dataset_hub)
(ROOT/'preflight.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result),flush=True)
