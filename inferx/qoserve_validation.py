"""QoServe caller contracts and source-policy views of native requests.

Native cache/dispatch integration is qualified separately in the pinned vLLM
environment. This module's lightweight part has no CUDA/runtime imports.
"""
import math
import ast
import copy
import hashlib
import inspect
import io
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

from inferx.qoserve_policy import load_author_policy, policy_manifest

NATIVE_SHA256 = 'f820bee4f42b878b262a30f3728a97d7413477b2a277748e20f2528ac2ec2f2a'
NATIVE_CAPSULE_SHA256 = '3dd377a94ef72661647dd8af5278f96adb44efdc961debf587befd4eaa577df3'
NATIVE_LICENSE_SHA256 = 'c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4'


def parse_contract(extra_args):
    """Validate flat vllm_xargs; only untagged code checks receive defaults."""
    tags = {key:value for key,value in (extra_args or {}).items() if key.startswith('qoserve_')}
    if not tags:
        return dict(tier=0, deadline_s=6., tbt_s=.1, trace_id=None)
    required = {'qoserve_tier', 'qoserve_deadline_s', 'qoserve_tbt_s'}
    if not required <= tags.keys() or tags.keys() - required - {'qoserve_trace_id'}:
        raise ValueError('incomplete or unknown QoServe contract fields')
    tier = tags['qoserve_tier']
    if type(tier) is not int or tier not in (0, 1):
        raise ValueError('QoServe tier must be integer 0 or 1')
    values = []
    for key in ('qoserve_deadline_s', 'qoserve_tbt_s'):
        value = tags[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError('QoServe deadlines and TBT must be positive finite seconds')
        values.append(float(value))
    trace = tags.get('qoserve_trace_id')
    if 'qoserve_trace_id' in tags and (not isinstance(trace, str) or not trace):
        raise ValueError('QoServe trace ID must be a nonempty string')
    return dict(tier=tier, deadline_s=values[0], tbt_s=values[1], trace_id=trace)


def sequence_view_type(policy):
    """Map real fields to the extracted author methods, retaining request age.

    The view owns policy metadata only. It never owns, allocates or frees KV.
    Its inherited comparison/deadline methods are the pinned author's code.
    """
    class NativeSequenceView(policy.Sequence):
        def __init__(self, request, *, wall_to_monotonic):
            self.request = request
            self.contract = parse_contract(request.sampling_params.extra_args)
            self.seq_id = request.request_id
            self.arrival_time = request.arrival_time + wall_to_monotonic
            self.state = SimpleNamespace(_arrived_at=self.arrival_time, e2e_prefill_time=0.)
            self.scheduler_type = policy.scheduler_type
            self.hybrid_prioritization_param = policy.hybrid_prioritization_param
            self.drop = 0
            self.cached_prefix_tokens = 0
            self.last_slack = 0.
            self.tier = self.contract['tier']
            self.tier_deadline = self.contract['deadline_s']
            self.ttft_deadline = self.contract['deadline_s']
            self.num_decode_tokens = -1  #max_tokens is a cap, not known actual output length.
            context = SimpleNamespace(output_len_pred=policy.output_len_pred,
                                      execution_threshold_batched=self.contract['tbt_s'])
            policy._qoserve_ttft_budget(context, self)

        @property
        def output_token_ids(self):
            return self.request.output_token_ids

        @property
        def mem_allocated(self):
            return self.request.num_computed_tokens > 0

        @property
        def prompt_processing_finished(self):
            return self.request.num_computed_tokens >= self.get_prompt_len()

        def get_prompt_len(self):
            # Native replay of past output is prefill work too.
            return max(self.request.num_prompt_tokens, self.request.num_tokens - 1)

        def get_num_prompt_tokens_processed(self):
            return min(max(self.request.num_computed_tokens, self.cached_prefix_tokens), self.get_prompt_len())

        def _get_context_size(self):
            # Original decode context includes the latest sampled input token;
            # prefill context is only the already-computed prompt/cache prefix.
            return (self.request.num_tokens if self.prompt_processing_finished
                    else self.get_num_prompt_tokens_processed())

    return NativeSequenceView


def _native_dispatch(native_source):
    tree = ast.parse(native_source)
    cls = next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='Scheduler')
    method = copy.deepcopy(next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=='schedule'))
    method.name = '_qoserve_schedule_native'
    counts = dict(running_gate=0,waiting_gate=0,running_allocation=0,waiting_allocation=0,
                  running_quota=0,waiting_quota=0,no_preempt=0)

    class Boundary(ast.NodeTransformer):
        def visit_While(self,node):
            self.generic_visit(node)
            if ast.unparse(node.test)=='req_index < len(self.running) and token_budget > 0':
                node.body[1:1] = ast.parse('''
if self._qoserve_policy_enabled and self._qoserve_quota(request) == 0:
    req_index += 1
    continue
''').body
                counts['running_gate']+=1
            elif ast.unparse(node.test)=='token_budget > 0':
                # Preserve native readiness/grammar handling before eligibility.
                index=next(i for i,n in enumerate(node.body) if isinstance(n,ast.If)
                           and ast.unparse(n.test)=='not ready_to_schedule')
                node.body[index+1:index+1]=ast.parse('''
if self._qoserve_policy_enabled and self._qoserve_quota(request) == 0:
    skip_request(request_queue)
    continue
''').body
                counts['waiting_gate']+=1
            elif ast.unparse(node.test)=='True' and any(isinstance(n,ast.Assign)
                    and ast.unparse(n.targets[0])=='new_blocks' for n in node.body):
                index=next(i for i,n in enumerate(node.body) if isinstance(n,ast.If)
                           and ast.unparse(n.test)=='new_blocks is not None')
                node.body.insert(index+1,ast.Break())
                counts['no_preempt']+=1
            return node

        def visit_Call(self,node):
            self.generic_visit(node)
            if ast.unparse(node.func)=='self.kv_cache_manager.allocate_slots':
                reservation=next((n for n in node.keywords if n.arg=='reserved_blocks'),None)
                if reservation is None:
                    node.keywords.append(ast.keyword(arg='reserved_blocks',value=ast.parse(
                        'self._qoserve_other_commitments(request)',mode='eval').body))
                    counts['running_allocation']+=1
                else:
                    reservation.value=ast.BinOp(left=reservation.value,op=ast.Add(),right=ast.parse(
                        'self._qoserve_other_commitments(request)',mode='eval').body)
                    counts['waiting_allocation']+=1
            elif isinstance(node.func,ast.Name) and node.func.id=='min':
                args=[ast.unparse(n) for n in node.args]
                if args==['num_new_tokens','token_budget','input_budget - draft_slots']:
                    node.args.append(ast.parse('self._qoserve_quota(request)',mode='eval').body)
                    counts['running_quota']+=1
                elif args==['num_new_tokens','request_token_budget']:
                    node.args.append(ast.parse('self._qoserve_quota(request)',mode='eval').body)
                    counts['waiting_quota']+=1
            return node

    method=ast.fix_missing_locations(Boundary().visit(method))
    if counts!={name:1 for name in counts}:
        raise ValueError(f'native QoServe boundary census mismatch: {counts}')
    return ast.unparse(method),counts


def prepare_adapter(author_root: Path, native_scheduler: Path, destination: Path,
                    *, coefficients: dict | None = None) -> dict:
    """Prepare an immutable overlay; never patch the installed native engine."""
    destination=Path(destination)
    if destination.exists() or destination.is_symlink():raise FileExistsError(destination)
    native_bytes=Path(native_scheduler).read_bytes()
    sha=lambda raw:hashlib.sha256(raw).hexdigest()
    if sha(native_bytes)!=NATIVE_SHA256:raise ValueError('native scheduler hash mismatch')
    root=Path(author_root)
    author=json.loads((root/'author-source-v1.json').read_text())
    policy=load_author_policy(root/'author-source-v1.tar.gz',author,coefficients)
    evidence=policy_manifest(root/'author-source-v1.tar.gz',author,coefficients,
                             timing_source=root/'author-timing-source-v1.py')
    native_capsule=(root/'native-source-v1.tar.gz').read_bytes()
    if sha(native_capsule)!=NATIVE_CAPSULE_SHA256:raise ValueError('native capsule hash mismatch')
    references=json.loads((root/'native-source-v1.json').read_text())['files']
    with tarfile.open(fileobj=io.BytesIO(native_capsule),mode='r:gz') as archive:
        actual={member.name:sha(archive.extractfile(member).read()) for member in archive.getmembers()}
    if actual!=references:raise ValueError('native reference inventory mismatch')
    native_license=(root/'native-source-v1.LICENSE').read_bytes()
    if sha(native_license)!=NATIVE_LICENSE_SHA256:raise ValueError('native license hash mismatch')
    with tarfile.open(root/'author-source-v1.tar.gz') as archive:
        author_license=archive.extractfile('LICENSE').read()
    dispatch,census=_native_dispatch(native_bytes.decode())
    header=f'''# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Native scheduler boundary; QoServe policy is in a separate Apache-2.0 module.
from __future__ import annotations
import hashlib,json,math,os,time,heapq
from pathlib import Path
from types import SimpleNamespace
import vllm.v1.core.sched.scheduler as _native
_NativeScheduler=_native.Scheduler
_site=Path(_native.__file__).parents[4]
for _name,_sha in {references!r}.items():
    if hashlib.sha256((_site/_name).read_bytes()).hexdigest()!=_sha:
        raise RuntimeError('installed native source hash mismatch: '+_name)
globals().update({{k:v for k,v in vars(_native).items() if not k.startswith('__')}})
from vllm.v1.core.sched.request_queue import FCFSRequestQueue
from inferx_qoserve_author_policy import AuthorPolicy
'''
    from inferx.ebplus_validation import record_native_runtime
    source=(header+'\n'+inspect.getsource(parse_contract)+'\n'+inspect.getsource(sequence_view_type)
            +'\n'+inspect.getsource(record_native_runtime)+'\n'+_NATIVE_CLASSES+'\n')
    # Add the transformed native method to the capacity-compatible class only.
    module=ast.parse(source)
    compatible=next(node for node in module.body if isinstance(node,ast.ClassDef)
                    and node.name=='CompatibleNativeScheduler')
    compatible.body.append(ast.parse(dispatch).body[0])
    source=ast.unparse(ast.fix_missing_locations(module))+'\n'
    source='# SPDX-License-Identifier: Apache-2.0\n# Native vLLM boundary; see LICENSE.native.\n'+source
    files={'inferx_qoserve_author_policy.py':policy._module_source.encode(),
           'inferx_qoserve_scheduler.py':source.encode(),
           'LICENSE.author':author_license,'LICENSE.native':native_license}
    manifest=dict(schema_version=1,policy=evidence,native_scheduler_sha256=NATIVE_SHA256,
        native_reference_capsule_sha256=NATIVE_CAPSULE_SHA256,native_source_files=references,
        native_boundary_census=census,files={name:sha(raw) for name,raw in files.items()},
        mappings=['Native reserved_blocks protects other full-input commitments; actual allocator owns all blocks.',
                  'Native readiness, prefix adoption, checkpoint/CoW and output ownership remain native.',
                  'QoServe per-step total budget and per-request quotas bound native allocation; no decoder eviction.',
                  'Pure preliminary native prefix/whole-input queries select eligible policy views; native admission rechecks.',
                  'Restore only new empty native defaultdict lookup entries after pure capacity forecasts.',
                  'Begin the pinned idempotent Mamba step-cache clear before policy forecasts; native call clears it again before allocation.',
                  'Caller TBT enters original deadline helpers per request; source tier1 output prediction remains15.',
                  'Hardware coefficient/initial-throughput presence is required; qualification is a separate real-model gate.'])
    destination.mkdir(parents=True,exist_ok=False)
    for name,raw in files.items():(destination/name).write_bytes(raw)
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


_NATIVE_CLASSES='''
def _record_qoserve_runtime(scheduler):
    record_native_runtime(scheduler)
    path=os.environ.get('INFERX_VLLM_RUNTIME')
    if not path:return
    target=Path(path);data=json.loads(target.read_text())
    data.update(scheduler_class=type(scheduler).__name__,
        full_input_commitments=isinstance(scheduler,CompatibleNativeScheduler),
        qoserve_policy_enabled=scheduler._qoserve_policy_enabled,
        profiling_mode=getattr(scheduler,'_qoserve_profiling_mode','off'),
        wall_to_monotonic_offset=scheduler._qoserve_clock_offset)
    temporary=target.with_suffix('.tmp');temporary.write_text(json.dumps(data,indent=2)+'\\n')
    temporary.replace(target)

class ObservedNativeScheduler(_NativeScheduler):
    _qoserve_policy_enabled=False
    def __init__(self,*args,**kwargs):
        _NativeScheduler.__init__(self,*args,**kwargs)
        if self.max_num_running_reqs>4 or self.max_num_active_reqs>4:
            raise ValueError('QoServe qualification requires hard C4')
        if (self.vllm_config.max_concurrent_batches!=1 or self.scheduler_config.async_scheduling
                or self.vllm_config.speculative_config is not None):
            raise ValueError('QoServe qualification requires synchronous MTP0')
        if self.connector is not None or self.ec_connector is not None:
            raise ValueError('external connectors are outside this native qualification')
        self._qoserve_clock_offset=time.monotonic()-time.time()
        self._qoserve_view_type=sequence_view_type(AuthorPolicy)
        self._qoserve_views={}
        self._qoserve_records={}
        self._qoserve_trace_ids=set()
        self._qoserve_decisions=[]
        self._qoserve_quotas={}
        _record_qoserve_runtime(self)

    def add_request(self,request):
        if request.request_id in self.requests:
            raise ValueError('duplicate native request ID')
        view=self._qoserve_view_type(request,wall_to_monotonic=self._qoserve_clock_offset)
        trace=view.contract['trace_id']
        if trace is not None and trace in self._qoserve_trace_ids:
            raise ValueError('duplicate QoServe trace ID')
        _NativeScheduler.add_request(self,request)
        self._qoserve_views[request.request_id]=view
        self._qoserve_records[request.request_id]=dict(request_id=request.request_id,
            trace_id=trace,contract=view.contract,arrival_monotonic=view.arrival_time,
            model_token_times_s=[],finish_status=None)
        if trace is not None:self._qoserve_trace_ids.add(trace)

    def update_from_output(self,scheduler_output,model_runner_output):
        before={rid:self._qoserve_views[rid] for rid in scheduler_output.num_scheduled_tokens
                if rid in self._qoserve_views}
        result=_NativeScheduler.update_from_output(self,scheduler_output,model_runner_output)
        now=time.monotonic()
        for batch in result.values():
            for output in batch.outputs:
                if not output.new_token_ids:continue
                if len(output.new_token_ids)!=1:
                    raise RuntimeError('unqualified multiple model tokens in synchronous MTP0')
                view=before[output.request_id]
                record=self._qoserve_records[output.request_id]
                if not record['model_token_times_s']:
                    view.state.e2e_prefill_time=now-view.arrival_time
                record['model_token_times_s'].append(now-view.arrival_time)
        return result

    def _free_request(self,request,*args,**kwargs):
        result=_NativeScheduler._free_request(self,request,*args,**kwargs)
        if request.request_id in self._qoserve_records:
            self._qoserve_records[request.request_id]['finish_status']=request.status.name
        self._qoserve_views.pop(request.request_id,None)
        return result

    def shutdown(self):
        path=os.environ.get('INFERX_QOSERVE_EVIDENCE')
        if path:
            target=Path(path);tmp=target.with_suffix('.tmp')
            tmp.write_text(json.dumps(dict(scheduler_class=type(self).__name__,
                requests=self._qoserve_records,decisions=self._qoserve_decisions,
                remaining_views=sorted(self._qoserve_views)),indent=2)+'\\n')
            tmp.replace(target)
        return _NativeScheduler.shutdown(self)


class CompatibleNativeScheduler(ObservedNativeScheduler):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if not self.scheduler_reserve_full_isl:
            raise ValueError('full-input commitments require native full-sequence admission check')

    def _qoserve_owners(self):
        running={r.request_id for r in self.running}
        return [r for rid,r in self.requests.items()
                if not r.is_finished() and (rid in running or self._holds_kv_blocks(r))]

    def _qoserve_other_commitments(self,request):
        excluded=getattr(request,'request_id',None)
        return sum(self._request_remaining_blocks(r) for r in self._qoserve_owners()
                   if r.request_id!=excluded)

    def _qoserve_effective_free_blocks(self):
        return self.kv_cache_manager.block_pool.get_num_free_blocks()-self._qoserve_other_commitments(None)

    def _qoserve_can_admit(self,request):
        if request in self._qoserve_owners():return True
        if len(self._qoserve_owners())+self.num_waiting_for_streaming_input>=self.max_num_active_reqs:
            return False
        if request.num_computed_tokens==0:
            blocks,local,_,_=self._get_local_prefix_cache_hit(request)
        else:
            blocks=self.kv_cache_manager.empty_kv_cache_blocks;local=0
        view=self._qoserve_views[request.request_id]
        view.cached_prefix_tokens=local
        computed=request.num_computed_tokens+local
        full=min(request.num_tokens,self.max_model_len)
        tables=[manager.req_to_blocks for manager in self.kv_cache_manager.coordinator.single_type_managers]
        existed=[request.request_id in table for table in tables]
        try:
            need=self.kv_cache_manager.coordinator.get_num_blocks_to_allocate(
                request_id=request.request_id,num_tokens=full,new_computed_blocks=blocks.blocks,
                num_encoder_tokens=0,total_computed_tokens=computed,num_local_computed_tokens=computed,
                num_tokens_main_model=full,apply_admission_cap=True,
                prefill_end=max(request.num_prompt_tokens,request.num_tokens-1))
        finally:
            # Pinned Mamba's read query indexes a defaultdict for a new waiter.
            # Undo only that empty lookup insertion; never release owned blocks.
            for table,present in zip(tables,existed):
                if not present and request.request_id in table:
                    if table[request.request_id]:
                        raise RuntimeError('capacity forecast unexpectedly acquired native blocks')
                    del table[request.request_id]
        watermark=self.kv_cache_manager.watermark_blocks if self.running else 0
        return need+self._qoserve_other_commitments(request)+watermark <= self.kv_cache_manager.block_pool.get_num_free_blocks()

    def _qoserve_quota(self,request):
        return (self._qoserve_quotas.get(request.request_id,0) if self._qoserve_policy_enabled
                else self.max_num_scheduled_tokens)

    def schedule(self,throttle_prefills=False):
        if throttle_prefills:raise ValueError('DP prefill throttling is outside qualification')
        return self._qoserve_schedule_native()


class QoServeScheduler(AuthorPolicy,CompatibleNativeScheduler):
    _qoserve_policy_enabled=True
    def __init__(self,*args,**kwargs):
        self.require_deployment_coefficients()
        CompatibleNativeScheduler.__init__(self,*args,**kwargs)
        if self.scheduler_config.long_prefill_token_threshold!=0:
            raise ValueError('QoServe dynamic candidates require native long-prefill cap0')
        if self.policy!=SchedulingPolicy.FCFS:
            raise ValueError('native policy must be FCFS before QoServe ordering')
        initial=float(os.environ['INFERX_QOSERVE_INITIAL_PPS'])
        if not math.isfinite(initial) or initial<=0:raise ValueError('invalid calibrated initial prefill throughput')
        self.expected_prefill_throughput=initial
        self.chunk_size=self.max_num_scheduled_tokens
        self._prefill_queue=[]
        self.last_prefill_size=0
        self.prev_call_time=-1

    def _qoserve_key(self,view,now):
        context=SimpleNamespace(execution_threshold=view.contract['tbt_s'],
                                execution_threshold_batched=view.contract['tbt_s'])
        return AuthorPolicy._sorting_key(context,view,now)

    def schedule(self,throttle_prefills=False):
        if throttle_prefills:raise ValueError('DP prefill throttling is outside qualification')
        # The previous synchronous model step is complete. The pinned native
        # method only clears Mamba cached_blocks_this_step and is idempotent;
        # planning must not mistake completed checkpoints for same-step writes.
        self.kv_cache_manager.new_step_starts()
        now=time.monotonic()
        self._qoserve_update_prefill_throughput(now)
        decoders=[self._qoserve_views[r.request_id] for r in self.running
                  if self._qoserve_views[r.request_id].prompt_processing_finished]
        decoders.sort(key=lambda view:self._qoserve_key(view,now))
        prefills=[]
        running={r.request_id for r in self.running}
        for rid,view in self._qoserve_views.items():
            if view.prompt_processing_finished:continue
            view.cached_prefix_tokens=0
            request=view.request
            if rid in running or (request.status in (RequestStatus.WAITING,RequestStatus.PREEMPTED)
                                  and self._qoserve_can_admit(request)):
                prefills.append(view)
        heapq.heapify(prefills)
        self._prefill_queue=prefills
        min_slack=min((view._get_TBT_deadline(view.contract['tbt_s'],view.contract['tbt_s'])-now
                       for view in decoders),default=1e18)
        decode_context=sum(view._get_context_size() for view in decoders)
        allowance=0;predicted=None
        if prefills:
            allowance,predicted=self._get_prefill_size_by_slack(
                min_slack,decode_context,prefills[0]._get_context_size(),len(decoders))
        self._qoserve_quotas={view.seq_id:1 for view in decoders}
        ranks={};used=0
        while self._prefill_queue:
            view=heapq.heappop(self._prefill_queue)
            if self._qoserve_relegate_once(view,now):continue
            count=min(allowance-used,view.get_prompt_len()-view.get_num_prompt_tokens_processed())
            if count<=0:break
            ranks[view.seq_id]=len(ranks)
            self._qoserve_quotas[view.seq_id]=count;used+=count
        decode_order={view.seq_id:i for i,view in enumerate(decoders)}
        self.running=sorted(self.running,key=lambda request:
            (0,decode_order[request.request_id]) if request.request_id in decode_order
            else (1,ranks.get(request.request_id,10**9)))
        for name in ('waiting','kv_holding_waiting'):
            setattr(self,name,FCFSRequestQueue(sorted(getattr(self,name),
                key=lambda request:ranks.get(request.request_id,10**9))))
        ceiling=self.max_num_scheduled_tokens
        selected=min(ceiling,max(1,used+len(decoders)))
        prefill_ids={view.seq_id for view in self._qoserve_views.values()
                     if not view.prompt_processing_finished}
        try:
            self.max_num_scheduled_tokens=selected
            output=self._qoserve_schedule_native()
        finally:
            self.max_num_scheduled_tokens=ceiling
        self.last_prefill_size=sum(count for rid,count in output.num_scheduled_tokens.items() if rid in prefill_ids)
        self._qoserve_decisions.append(dict(step=self.current_step,min_slack=min_slack,
            allowance=allowance,predicted_s=predicted,selected_total=selected,
            actual=dict(output.num_scheduled_tokens),effective_free_blocks=self._qoserve_effective_free_blocks()))
        return output
'''
