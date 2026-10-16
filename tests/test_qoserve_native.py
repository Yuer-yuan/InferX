"""Real native Request/hybrid allocator contracts, explicitly outside GPU timing.

ModelRunnerOutput supplies token IDs to exercise the real CPU state machine;
this suite neither simulates performance nor substitutes for model gates.
"""
import copy
import importlib
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import time
import unittest
from unittest.mock import patch


@unittest.skipUnless(os.environ.get('INFERX_QOSERVE_CPU_CONTRACT') == '1',
                     'pinned native CPU gate; prohibited during remote GPU timing')
class QoServeNativeContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from inferx.qoserve_validation import prepare_adapter
        from vllm.engine.arg_utils import EngineArgs
        cls.tmp = tempfile.TemporaryDirectory(prefix='inferx-qoserve-native-')
        cls.root = Path(cls.tmp.name)
        cls.evidence = Path(__file__).resolve().parents[1] / 'results/qoserve-validation'
        with tarfile.open(cls.evidence / 'native-source-v1.tar.gz') as archive:
            scheduler = cls.root / 'scheduler.py'
            scheduler.write_bytes(archive.extractfile('vllm/v1/core/sched/scheduler.py').read())
        # Presence-only unit coefficients, not a qualified hardware profile.
        table = dict(coefficients=dict(batch_num_tokens=.00001, batch_num_prefill_tokens=0.,
            batch_num_decode_tokens=0., batch_decode_context=0., batch_prefill_context=0.), intercept=.01)
        cls.overlay = cls.root / 'overlay'
        prepare_adapter(cls.evidence, scheduler, cls.overlay,
                        coefficients={name: copy.deepcopy(table) for name in ('tokens_0_512','tokens_513_plus')})
        sys.path.insert(0, str(cls.overlay))
        cls.module = importlib.import_module('inferx_qoserve_scheduler')
        cls.config = EngineArgs(
            model=str(Path(os.environ['INFERX_ROOT']) / 'models/qwen38-int4-redhat'),
            dtype='bfloat16', max_model_len=131328, max_num_seqs=4,
            max_num_batched_tokens=4096, long_prefill_token_threshold=0,
            enable_chunked_prefill=True, enable_prefix_caching=True,
            enforce_eager=True, async_scheduling=False, language_model_only=True,
            mamba_cache_mode='align', kv_cache_dtype='fp8',
        ).create_engine_config()

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(str(cls.overlay))
        for name in ('inferx_qoserve_scheduler', 'inferx_qoserve_author_policy'):
            sys.modules.pop(name, None)
        cls.tmp.cleanup()

    def scheduler(self, name='QoServeScheduler', blocks=256, cap=0):
        import torch
        from vllm.v1.kv_cache_interface import (
            FullAttentionSpec, MambaSpec, KVCacheConfig, KVCacheGroupSpec)
        from vllm.v1.structured_output import StructuredOutputManager
        config = copy.deepcopy(self.config)
        config.cache_config.num_gpu_blocks = blocks
        config.scheduler_config.long_prefill_token_threshold = cap
        config.scheduler_config.scheduler_reserve_full_isl = True
        attention = FullAttentionSpec(block_size=256, num_kv_heads=1,
                                      head_size=16, dtype=torch.bfloat16)
        state = MambaSpec(block_size=256, shapes=((4, 4), (4, 4)),
            dtypes=(torch.bfloat16, torch.bfloat16), page_size_padded=attention.page_size_bytes,
            mamba_cache_mode='align', num_prefill_checkpoint_blocks=1,
            prefill_checkpoint_alignment=256)
        cache = KVCacheConfig(num_blocks=blocks, kv_cache_tensors=[], kv_cache_groups=[
            KVCacheGroupSpec(layer_names=['attention'], kv_cache_spec=attention),
            KVCacheGroupSpec(layer_names=['gdn'], kv_cache_spec=state)])
        with patch.dict(os.environ,INFERX_QOSERVE_INITIAL_PPS='8500'):
            return getattr(self.module, name)(config, cache, StructuredOutputManager(config), block_size=256)

    def add(self, scheduler, name, length=32, output=4, token=42, tags=None, arrival=None):
        from vllm import SamplingParams
        from vllm.v1.request import Request
        from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
        from vllm.utils.hashing import get_hash_fn_by_name
        hasher = get_hash_fn_by_name('sha256')
        init_none_hash(hasher)
        request = Request(request_id=name, prompt_token_ids=[token] * length,
            sampling_params=SamplingParams(max_tokens=output, ignore_eos=True, extra_args=tags),
            pooling_params=None, block_hasher=get_request_block_hasher(256, hasher), arrival_time=arrival)
        scheduler.add_request(request)
        return request

    def step(self, scheduler):
        from vllm.v1.outputs import ModelRunnerOutput
        output = scheduler.schedule()
        ids = list(output.num_scheduled_tokens)
        runner = ModelRunnerOutput(req_ids=ids, req_id_to_index={rid:i for i,rid in enumerate(ids)},
            sampled_token_ids=[[] if scheduler.requests[rid].is_prefill_chunk else [7] for rid in ids])
        result = scheduler.update_from_output(output, runner)
        return output, result

    def finish(self, scheduler, bound=500):
        for _ in range(bound):
            # Finished-ID notifications are not remaining request work.
            if not scheduler.requests:
                return
            output, _ = self.step(scheduler)
            self.assertGreater(output.total_num_scheduled_tokens, 0, 'owned work starved without a model step')
        self.fail('native requests did not finish within the declared CPU step bound')

    def ownership(self, scheduler):
        pool = scheduler.kv_cache_manager.block_pool
        return dict(free=pool.get_num_free_blocks(),
            refs=[block.ref_cnt for block in pool.blocks],
            manager_tables=[dict(request_blocks={rid:tuple(block.block_id for block in blocks)
                for rid,blocks in manager.req_to_blocks.items()},
                cached=dict(manager.num_cached_block),
                checkpoints=dict(getattr(manager,'_checkpoints',{})))
                for manager in scheduler.kv_cache_manager.coordinator.single_type_managers],
            queues=[[r.request_id for r in queue] for queue in
                    (scheduler.running,scheduler.waiting,scheduler.kv_holding_waiting)],
            blocks={rid:scheduler.kv_cache_manager.get_blocks(rid).get_block_ids()
                    for rid in scheduler.requests if scheduler.requests[rid].num_computed_tokens > 0})

    def test_full_promises_prevent_joint_overadmission_and_both_requests_finish(self):
        # Each20K prompt fits in128 logical hybrid blocks; their combined inputs do not.
        for name in ('CompatibleNativeScheduler','QoServeScheduler'):
            with self.subTest(name=name):
                scheduler = self.scheduler(name, blocks=128)
                first = self.add(scheduler,'first',length=20000,token=41)
                second = self.add(scheduler,'second',length=20000,token=42)
                output, _ = self.step(scheduler)
                self.assertIn(first.request_id, output.num_scheduled_tokens)
                self.assertNotIn(second.request_id, output.num_scheduled_tokens)
                self.assertIn(second, scheduler.waiting)
                self.assertGreater(scheduler._qoserve_other_commitments(None), 0)
                self.assertGreaterEqual(scheduler._qoserve_effective_free_blocks(), 0)
                self.finish(scheduler)
                self.assertEqual((len(first.output_token_ids),len(second.output_token_ids)),(4,4))
                self.assertEqual(scheduler._qoserve_other_commitments(None), 0)
                self.assertEqual((first.num_preemptions,second.num_preemptions),(0,0))

    def test_capacity_queries_preserve_real_native_ownership(self):
        scheduler = self.scheduler()
        request = self.add(scheduler,'partial',length=8000)
        self.step(scheduler)
        waiting = self.add(scheduler,'candidate',length=8000,token=43)
        before = self.ownership(scheduler)
        for _ in range(20):
            self.assertGreaterEqual(scheduler._qoserve_effective_free_blocks(), 0)
            self.assertGreaterEqual(scheduler._qoserve_other_commitments(request), 0)
            self.assertTrue(scheduler._qoserve_can_admit(waiting))
        self.assertEqual(self.ownership(scheduler), before)

    def test_abort_releases_promises_and_admits_waiter(self):
        from vllm.v1.request import RequestStatus
        scheduler = self.scheduler(blocks=128)
        first = self.add(scheduler,'abort-owner',length=20000,token=41)
        second = self.add(scheduler,'waiter',length=20000,token=42)
        self.step(scheduler)
        scheduler.finish_requests([first.request_id], RequestStatus.FINISHED_ABORTED)
        self.assertNotIn(first.request_id, scheduler.requests)
        output,_ = self.step(scheduler)
        self.assertIn(second.request_id, output.num_scheduled_tokens)
        self.finish(scheduler)
        self.assertEqual(scheduler._qoserve_other_commitments(None),0)

    def test_small_dynamic_chunk_advances_private_mamba_state_and_keeps_decode_first(self):
        scheduler = self.scheduler()
        short = self.add(scheduler,'decoder',output=8,tags={
            'qoserve_tier':0,'qoserve_deadline_s':1e-9,'qoserve_tbt_s':1e-9,'qoserve_trace_id':'short'})
        self.step(scheduler)
        self.assertEqual(len(short.output_token_ids),1)
        long = self.add(scheduler,'long',length=8000,token=43,tags={
            'qoserve_tier':1,'qoserve_deadline_s':600.,'qoserve_tbt_s':.4,'qoserve_trace_id':'long'})
        before = long.num_computed_tokens
        output,_ = self.step(scheduler)
        self.assertEqual(list(output.num_scheduled_tokens)[0],short.request_id)
        self.assertLessEqual(output.total_num_scheduled_tokens,128)
        self.assertGreater(long.num_computed_tokens,before)
        self.assertEqual(set(scheduler.running),{short,long})
        self.assertEqual((short.num_preemptions,long.num_preemptions),(0,0))
        self.finish(scheduler)

    def test_urgent_waiter_can_pause_owned_partial_prefill_without_hiding_it(self):
        scheduler = self.scheduler()
        long = self.add(scheduler,'partial-long',length=8000,output=8,token=43,tags={
            'qoserve_tier':1,'qoserve_deadline_s':600.,'qoserve_tbt_s':.4})
        self.step(scheduler)
        self.assertLess(long.num_computed_tokens,long.num_prompt_tokens)
        blocks = scheduler.kv_cache_manager.get_blocks(long.request_id).get_block_ids()
        short = self.add(scheduler,'urgent',length=3000,output=8,token=44,tags={
            'qoserve_tier':0,'qoserve_deadline_s':1.,'qoserve_tbt_s':.1})
        before = long.num_computed_tokens
        output,_ = self.step(scheduler)
        self.assertIn(short.request_id,output.num_scheduled_tokens)
        self.assertEqual(long.num_computed_tokens,before)
        self.assertIn(long,scheduler.running)
        self.assertEqual(scheduler.kv_cache_manager.get_blocks(long.request_id).get_block_ids(),blocks)
        self.finish(scheduler)

    def test_blocked_waiter_cannot_starve_already_committed_partial_prefill(self):
        scheduler = self.scheduler(blocks=128)
        long = self.add(scheduler,'owned',length=20000,token=41,tags={
            'qoserve_tier':1,'qoserve_deadline_s':600.,'qoserve_tbt_s':.4})
        self.step(scheduler)
        urgent = self.add(scheduler,'cannot-fit',length=20000,token=42,tags={
            'qoserve_tier':0,'qoserve_deadline_s':1.,'qoserve_tbt_s':.1})
        before = long.num_computed_tokens
        output,_ = self.step(scheduler)
        self.assertIn(long.request_id,output.num_scheduled_tokens)
        self.assertGreater(long.num_computed_tokens,before)
        self.assertIn(urgent,scheduler.waiting)
        self.finish(scheduler)

    def test_all_allocated_owners_count_toward_hard_C4(self):
        scheduler = self.scheduler()
        active = [self.add(scheduler,f'owner-{i}',length=32,output=8,token=40+i) for i in range(4)]
        self.step(scheduler)
        fifth = self.add(scheduler,'fifth',length=32,output=8,token=49)
        self.step(scheduler)
        self.assertEqual(set(scheduler.running),set(active))
        self.assertIn(fifth,scheduler.waiting)
        self.assertEqual(fifth.num_computed_tokens,0)
        self.finish(scheduler)

    def test_actual_shared_prefix_is_adopted_with_native_cow_accounting(self):
        scheduler = self.scheduler(blocks=128)
        seed = self.add(scheduler,'seed',length=8192,token=45)
        self.finish(scheduler)
        first = self.add(scheduler,'shared-a',length=8192,output=8,token=45)
        second = self.add(scheduler,'shared-b',length=8192,output=8,token=45)
        self.step(scheduler)
        self.assertGreater(first.num_computed_tokens,4096)
        # Mamba refuses reuse of a newly published same-step checkpoint. It is
        # available after that real CPU/model step has completed, on the next step.
        if second.num_computed_tokens==0:self.step(scheduler)
        self.assertGreater(second.num_computed_tokens,4096)
        a=scheduler.kv_cache_manager.get_blocks(first.request_id).get_block_ids()
        b=scheduler.kv_cache_manager.get_blocks(second.request_id).get_block_ids()
        self.assertTrue(set(a[0]) & set(b[0]), 'shared attention prefix was discarded')
        before=self.ownership(scheduler)
        for _ in range(5):scheduler._qoserve_effective_free_blocks()
        self.assertEqual(self.ownership(scheduler),before)
        self.finish(scheduler)
        self.assertEqual((len(seed.output_token_ids),len(first.output_token_ids),len(second.output_token_ids)),(4,8,8))

    def test_late_request_is_requeued_once_and_still_finishes(self):
        scheduler = self.scheduler()
        request = self.add(scheduler,'late',length=1024,tags={
            'qoserve_tier':0,'qoserve_deadline_s':1e-9,'qoserve_tbt_s':.1})
        self.step(scheduler)
        view = scheduler._qoserve_views[request.request_id]
        self.assertEqual(view.drop,1)
        self.finish(scheduler)
        self.assertEqual(len(request.output_token_ids),4)
        self.assertNotIn(request.request_id,scheduler._qoserve_views)

    def test_one_wall_to_monotonic_mapping_keeps_original_request_age(self):
        with patch.object(self.module.time,'time',return_value=2000.), \
             patch.object(self.module.time,'monotonic',return_value=1000.):
            scheduler=self.scheduler()
        request=self.add(scheduler,'old-arrival',arrival=1990.,output=8,tags={
            'qoserve_tier':1,'qoserve_deadline_s':600.,'qoserve_tbt_s':.4})
        view=scheduler._qoserve_views[request.request_id]
        self.assertEqual(view.arrival_time,990.)
        self.assertEqual(view.ttft_deadline,594.)
        self.step(scheduler)
        with patch.object(self.module.time,'time',return_value=50000.):
            self.step(scheduler)
        self.assertIs(scheduler._qoserve_views[request.request_id],view)
        self.assertEqual(view.arrival_time,990.)
        self.finish(scheduler)

    def test_invalid_qos_metadata_fails_before_native_allocation(self):
        for tags in ({'qoserve_tier':2}, {'qoserve_tier':True},
                     {'qoserve_tier':1,'qoserve_deadline_s':-1.,'qoserve_tbt_s':.4},
                     {'qoserve_tier':0,'qoserve_deadline_s':float('nan'),'qoserve_tbt_s':.1},
                     {'qoserve_tier':0,'qoserve_deadline_s':6.,'qoserve_tbt_s':True}):
            scheduler=self.scheduler()
            before=self.ownership(scheduler)
            with self.subTest(tags=tags),self.assertRaises(ValueError):
                self.add(scheduler,'bad',tags=tags)
            self.assertEqual(self.ownership(scheduler),before)
            self.assertNotIn('bad',scheduler.requests)

    def test_first_actual_token_time_drives_original_cumulative_deadline(self):
        with patch.object(self.module.time,'time',return_value=2000.), \
             patch.object(self.module.time,'monotonic',return_value=1000.):
            scheduler=self.scheduler()
        request=self.add(scheduler,'token-clock',arrival=1990.,output=8,tags={
            'qoserve_tier':0,'qoserve_deadline_s':6.,'qoserve_tbt_s':.2})
        with patch.object(self.module.time,'monotonic',return_value=1000.):self.step(scheduler)
        view=scheduler._qoserve_views[request.request_id]
        self.assertEqual(view.state.e2e_prefill_time,10.)
        self.assertAlmostEqual(view._get_TBT_deadline(.2,.8),1000.4)
        with patch.object(self.module.time,'monotonic',return_value=1000.3):self.step(scheduler)
        self.assertEqual(view.state.e2e_prefill_time,10.)
        self.assertAlmostEqual(view._get_TBT_deadline(.2,.8),1000.6)

    def test_allocation_exception_keeps_native_owners_available_for_retry(self):
        scheduler=self.scheduler()
        request=self.add(scheduler,'retain',length=8000)
        with patch.object(scheduler.kv_cache_manager,'allocate_slots',side_effect=RuntimeError('allocation fault')):
            with self.assertRaisesRegex(RuntimeError,'allocation fault'):scheduler.schedule()
        self.assertIs(scheduler.requests[request.request_id],request)
        self.assertIn(request,scheduler.waiting)
        self.finish(scheduler)

    def test_runtime_records_actual_scheduler_capacity_boundary_and_profile_mode(self):
        path=self.root/'runtime-observed.json'
        for name,committed,policy in [('ObservedNativeScheduler',False,False),
                                       ('CompatibleNativeScheduler',True,False),
                                       ('QoServeScheduler',True,True)]:
            with self.subTest(name=name),patch.dict(os.environ,INFERX_VLLM_RUNTIME=str(path)):
                scheduler=self.scheduler(name)
                state=json.loads(path.read_text())
                self.assertEqual(state.get('scheduler_class'),name)
                self.assertIs(state.get('full_input_commitments'),committed)
                self.assertIs(state.get('qoserve_policy_enabled'),policy)
                self.assertEqual(state.get('profiling_mode'),'off')
                self.assertEqual((state['num_gpu_blocks'],state['chunk'],state['long_prefill_token_threshold']),
                                 (256,4096,0))
                self.assertAlmostEqual(state['wall_to_monotonic_offset'],scheduler._qoserve_clock_offset)


if __name__ == '__main__':
    unittest.main()
