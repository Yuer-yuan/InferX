"""CPU contract probe: exact pinned fallback forward on a tiny real GDN model.
This is not a BulletServe engine/GPU/performance reproduction.
"""
import ast,hashlib,json,os,subprocess,types
from pathlib import Path
import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
r=Path(os.environ['INFERX_ROOT']);out=r/'sota-validation/bullet-gdn-contract-v2';out.mkdir(exist_ok=False)
source=subprocess.check_output(['git','show','c4aa493b737c861128e6a5b6b0b3413cb26aed63:python/sglang/srt/models/transformers.py'],cwd=r/'sota-validation/sources/BulletServe-repo',text=True);(out/'fallback.py').write_text(source)
module=ast.parse(source);cls=next(x for x in module.body if isinstance(x,ast.ClassDef) and x.name=='TransformersForCausalLM');fn=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='forward');standalone=ast.Module(body=[fn],type_ignores=[])
ns={'torch':torch,'ForwardBatch':object,'LogitsProcessorOutput':object};exec(compile(ast.fix_missing_locations(standalone),'pinned-bullet-fallback-forward','exec'),ns)
torch.set_num_threads(1);torch.manual_seed(20261016)
config=Qwen3_5TextConfig(vocab_size=256,hidden_size=64,intermediate_size=128,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=16,layer_types=['linear_attention','full_attention'],linear_num_key_heads=2,linear_num_value_heads=4,linear_key_head_dim=16,linear_value_head_dim=16,linear_conv_kernel_dim=4)
config._attn_implementation='eager';model=Qwen3_5TextModel(config).eval();ids=torch.tensor([[3,11,57,98,142,233,19,31]]);positions=torch.arange(ids.shape[1])[None,:]
owner=types.SimpleNamespace(model=model,attention_instances=[],lm_head=None,logits_processor=lambda ids,hidden,*args:hidden)
with torch.no_grad():
 full=model(ids,position_ids=positions,use_cache=False).last_hidden_state[0,-1]
 first=ns['forward'](owner,ids[0,:-1],positions[0,:-1],object())
 dropped=ns['forward'](owner,ids[0,-1:],positions[0,-1:],object())[0]
 pre=model(ids[:,:-1],position_ids=positions[:,:-1],use_cache=True)
 print('cache fields',[(type(x).__name__,list(vars(x))) for x in pre.past_key_values.layers],flush=True)
 resumed=model(ids[:,-1:],position_ids=positions[:,-1:],past_key_values=pre.past_key_values,use_cache=True).last_hidden_state[0,-1]
 error_drop=(dropped-full).abs().max().item();error_preserved=(resumed-full).abs().max().item()
 assert torch.allclose(resumed,full,rtol=1e-4,atol=1e-4),(error_preserved,'cache-preserving control failed')
 assert not torch.allclose(dropped,full,rtol=1e-4,atol=1e-4),'probe did not expose lost state'
report=dict(scope='Exact fallback forward CPU hybrid state contract probe; native tinyHF CPU, not fullBullet engine/GPU/INT4/SOTA efficacy',bullet_commit='c4aa493b737c861128e6a5b6b0b3413cb26aed63',fallback_sha256=hashlib.sha256(source.encode()).hexdigest(),model=type(model).__name__,layers=2,layer_types=config.layer_types,hidden_size=64,sequence_tokens=8,fallback_passes_no_hf_cache=True,dropped_state_max_abs=error_drop,preserved_state_max_abs=error_preserved,preserved_state_matches_full=True,missing_integration='Per-request recurrent conv/SSM state ownership, batching/prefix forks and P/D handoff; cached HF control proves state-preserving path exists but does not implement Bullet engine integration.')
(out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
