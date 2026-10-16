"""Unscored matched-prefix CPU profiles; no kernel or GPU worker replacement."""
import copy,cProfile,hashlib,json,os,pstats,sys
from pathlib import Path
from vllm.v1.engine.core import EngineCore
from vllm.v1.request import RequestStatus
from scripts import calibrate_qoserve_model as calibration

output=Path(sys.argv[sys.argv.index('--output')+1]);directory=output.parent
original_specs=calibration.rollout_specs;original_step=EngineCore.step
probes={};captures={};aborts={}

def specs(plan,engine_id):
 result=[s for s in original_specs(plan,engine_id) if s['kind']=='initial-pps']
 sequence=[('fixed-before',0,512,[128],65),('variable-before',0,512,plan['profile']['candidates'],11),
           ('warm-mixed',3,512,[128],None),('variable-after-mixed',0,512,plan['profile']['candidates'],11),
           ('fixed-after-mixed',0,512,[128],65)]
 for name,d,context,candidates,target in sequence:
  rid='diagnostic-'+name
  spec=dict(rollout_id=rid,kind='diagnostic',phase='holdout',budget_mode='total-cycle',
    holdout_protocol='unscored-cpu-profile',candidates=candidates,prefill_target=128,
    decoders=d,prompt_len=131072,decoder_prompt_len=context,decoder_max_tokens=1024,
    warmup_steps=5,long_salt=engine_id+'/'+rid+'/long',decoder_salt=engine_id+'/'+rid+'/decoder',token_seed=777)
  probes[rid]=dict(name=name,target=target,abort_after=1 if target is None else target+1)
  result.append(spec)
 (directory/'diagnostic-definition.json').write_text(json.dumps(dict(scope='Unscored CPU profiles at identical context7840 and total128; full131072-token inputs but intentionally aborted after probes, not complete long tasks.',specs=result),indent=2)+'\n')
 return result

def profiled_step(self):
 scheduler=self.scheduler;config=getattr(scheduler,'_profile_config',None)
 probe=probes.get(config['rollout_id']) if config else None
 index=len(scheduler._profile_rows) if probe else -1
 profile=None
 if probe and index==probe['target']:
  request=scheduler.requests[config['long_request_id']]
  before=dict(computed_tokens=request.num_computed_tokens,prompt_tokens=request.num_prompt_tokens)
  profile=cProfile.Profile();profile.enable()
 try:result=original_step(self)
 finally:
  if profile is not None:profile.disable()
 if profile is not None:
  row=copy.deepcopy(scheduler._profile_rows[-1]);name=probe['name']
  profile.dump_stats(str(directory/(name+'.pstats')))
  with (directory/(name+'-cumulative.txt')).open('x') as stream:pstats.Stats(profile,stream=stream).sort_stats('cumulative').print_stats(60)
  with (directory/(name+'-self.txt')).open('x') as stream:pstats.Stats(profile,stream=stream).sort_stats('tottime').print_stats(60)
  stats=pstats.Stats(profile)
  functions=[dict(file=k[0],line=k[1],function=k[2],primitive_calls=v[0],calls=v[1],self_s=v[2],cumulative_s=v[3]) for k,v in stats.stats.items()]
  captures[name]=dict(before=before,row=row,functions=sorted(functions,key=lambda x:x['self_s'],reverse=True)[:80],
      profiler_total_s=stats.total_tt,primitive_calls=stats.prim_calls,total_calls=stats.total_calls)
 if probe and index+1>=probe['abort_after'] and config['long_request_id'] in scheduler.requests:
  scheduler.finish_requests([config['long_request_id']],RequestStatus.FINISHED_ABORTED)
  aborts[probe['name']]='intentional diagnostic stop after matched step'
 return result

calibration.rollout_specs=specs;EngineCore.step=profiled_step
try:calibration.main()
finally:
 EngineCore.step=original_step;calibration.rollout_specs=original_specs
 scope=dict(scope='Instrumented diagnostic only; never predictor fitting, qualification, throughput or SLO evidence.',
   training_allowed=False,qualification_allowed=False,captures=captures,intentional_aborts=aborts,
   driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 expected=dict(batch_num_tokens=128,batch_num_prefill_tokens=128,batch_num_decode_tokens=0,batch_decode_context=0,batch_prefill_context=7840)
 scope['matched_features_verified']=len(captures)==4 and all(c['row']['features']==expected and c['before']==dict(computed_tokens=7840,prompt_tokens=131072) for c in captures.values())
 (directory/'cpu-profile-audit.json').write_text(json.dumps(scope,indent=2,allow_nan=False)+'\n')
 if output.exists():
  state=json.loads(output.read_text());state.update(matrix_kind='unscored-shape-transition-cpu-profile',training_allowed=False,qualification_allowed=False)
  for entry in state['rollouts']:
   if entry['spec']['kind']=='diagnostic':entry.update(source_loop_stop_reason=entry.get('stop_reason'),stop_reason='intentional-diagnostic-abort')
  output.write_text(json.dumps(state,indent=2,allow_nan=False)+'\n')
