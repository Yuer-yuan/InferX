import json,inspect,hashlib,os
from pathlib import Path
from types import SimpleNamespace as NS
import torch
from sglang.srt.models.qwen3_5 import Qwen3_5ForConditionalGeneration as Qwen
from sglang.srt.model_executor.model_runner import ModelRunner
from inferx_pd_runtime import install_qwen_split
# Exercise the real class/runner with scalar layers; avoid constructing weights or CUDA state.
install_qwen_split(Qwen,lambda:1)
q=Qwen.__new__(Qwen);torch.nn.Module.__init__(q)
def layer(i):
 def f(*,positions,hidden_states,residual,forward_batch):return hidden_states+i+1,(residual or 0)+hidden_states
 return f
q.model=NS(embed_tokens=lambda ids:ids,layers=[layer(i) for i in range(3)],norm=lambda h,r:(h+r,None),flashinfer_mnnvl_cutedsl_fusion=None)
q.config=NS(num_hidden_layers=3,num_experts=0);q.language_model_only=True;q.capture_aux_hidden_states=False;q.is_mrope_enabled=False
q.pp_group=NS(world_size=1,is_first_rank=True,is_last_rank=True);q.lm_head=None
q.logits_processor=lambda ids,h,head,batch,aux=None:h*3
runner=ModelRunner.__new__(ModelRunner);runner.model=q;runner.model_config=NS(num_hidden_layers=3);runner.device_timer=None
calls=[];runner.attn_backend=NS(init_forward_metadata=lambda b:calls.append(b.split_index))
batch=NS(input_ids=2,positions=None,split_index=0,spec_info=None)
assert runner.forward_split_prefill(batch,forward_count=1) is None
assert runner.forward_split_prefill(batch,forward_count=2)==54
assert batch.split_index==3 and calls==[0]
p=Path(inspect.getsourcefile(Qwen));report=dict(status='passed',scope='real Qwen wrapper and ModelRunner CPU invocation with scalar layers; not GPU/GDN correctness',source=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),result=54,intervals=[[0,1],[1,3]])
r=Path(os.environ['INFERX_ROOT'])/'pd-strategies/compatibility';r.mkdir(parents=True,exist_ok=True);(r/'actual-split-cpu.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
