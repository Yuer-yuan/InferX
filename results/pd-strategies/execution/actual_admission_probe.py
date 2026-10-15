import json,hashlib,inspect,os
from pathlib import Path
from types import SimpleNamespace as NS
from sglang.srt.managers.schedule_policy import PrefillAdder
from inferx.sglang_pd import cap_chunk_budget
rows=[]
for cap in [None,128,256]:
 a=PrefillAdder.__new__(PrefillAdder)
 a.__dict__.update(dllm_config=None,is_hybrid_swa=False,is_all_swa=False,is_hybrid_ssm_cache=True,
  rem_chunk_tokens=512,rem_total_token_offset=0,cur_rem_token_offset=0,rem_input_tokens=4096,
  _mamba_slot_cost=0,rem_mamba_slots=None,page_size=1,waiting_queue_len=1,
  prefill_delayer_single_pass=None,can_run_list=[],token_to_kv_pool_allocator=NS(available_size=lambda:200000),
  tree_cache=NS(full_evictable_size=lambda:0),log_hit_tokens=0,log_input_tokens=0,
  reprocessed_log_hit_tokens=0,reprocessed_log_input_tokens=0)
 req=NS(full_untruncated_fill_ids=[0]*2000,prefix_indices=[],sampling_params=NS(max_new_tokens=64),
  retracted_stain=False)
 req.set_extend_range=lambda start,end:setattr(req,'extend_range',NS(length=end-start))
 original=lambda r:PrefillAdder.add_chunked_req(a,r)
 result=original(req) if cap is None else cap_chunk_budget(a,req,original,cap)
 expected=512 if cap is None else cap
 assert req.extend_range.length==expected and a.rem_chunk_tokens==512-expected
 assert a.rem_total_token_offset==expected+1 and a.rem_input_tokens==4096-expected
 assert result is req and a.can_run_list==[req]
 rows.append(dict(cap=cap,consumed=req.extend_range.length,round_remaining=a.rem_chunk_tokens,
                  total_offset=a.rem_total_token_offset))
p=Path(inspect.getsourcefile(PrefillAdder))
report=dict(status='passed',scope='actual installed PrefillAdder method and accounting with CPU scalar allocator fixture; no GPU capacity or efficacy claim',
 source=str(p),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),rows=rows)
r=Path(os.environ['INFERX_ROOT'])/'pd-strategies/compatibility';r.mkdir(parents=True,exist_ok=True)
(r/'actual-admission-cpu.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
