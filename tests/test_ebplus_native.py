"""Dedicated CPU gate using real installed vLLM Request and hybrid allocator.

Run only after timed GPU campaigns finish, with INFERX_EBPLUS_CPU_CONTRACT=1
in the pinned vLLM environment. No model weights or GPU kernels are loaded.
The small logical cache is a real native cache manager, not a replacement API.
"""
import importlib
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipUnless(os.environ.get('INFERX_EBPLUS_CPU_CONTRACT') == '1',
                     'dedicated pinned-vLLM CPU gate; do not run during GPU timing')
class NativeHybridContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from inferx.ebplus_validation import prepare_adapter
        cls.tmp = tempfile.TemporaryDirectory(prefix='inferx-ebplus-native-')
        root = Path(cls.tmp.name)
        capsule = Path(__file__).parents[1] / 'results/ebplus-validation/source-capsule-v1.tar.gz'
        with tarfile.open(capsule) as archive:
            for name in ('author_scheduler.py', 'author_calibration.py', 'current_scheduler.py', 'author_LICENSE'):
                (root / name).write_bytes(archive.extractfile(name).read())
        cls.overlay = root / 'overlay'
        prepare_adapter(root / 'author_scheduler.py', root / 'author_calibration.py',
                        root / 'current_scheduler.py', cls.overlay)
        sys.path.insert(0, str(cls.overlay))
        cls.scheduler_cls = importlib.import_module('inferx_ebplus_scheduler').EBPlusScheduler
        import torch
        from vllm.engine.arg_utils import EngineArgs
        cls.torch = torch
        cls.config = EngineArgs(
            model=str(Path(os.environ['INFERX_ROOT']) / 'models/qwen38-int4-redhat'),
            dtype='bfloat16', max_model_len=131328, max_num_seqs=4,
            max_num_batched_tokens=4096, long_prefill_token_threshold=2048,
            enable_chunked_prefill=True, enable_prefix_caching=True,
            enforce_eager=True, async_scheduling=False, language_model_only=True,
            mamba_cache_mode='align', kv_cache_dtype='fp8',
        ).create_engine_config()
        cls.config.cache_config.num_gpu_blocks = 256

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(str(cls.overlay))
        cls.tmp.cleanup()

    def scheduler(self, mode='eb', cls=None, blocks=256):
        from vllm.v1.kv_cache_interface import (
            FullAttentionSpec, MambaSpec, KVCacheConfig, KVCacheGroupSpec)
        from vllm.v1.structured_output import StructuredOutputManager
        import copy
        config = copy.deepcopy(self.config)
        config.cache_config.num_gpu_blocks = blocks
        attn = FullAttentionSpec(block_size=256, num_kv_heads=1, head_size=16,
                                 dtype=self.torch.bfloat16)
        state = MambaSpec(block_size=256, shapes=((4, 4), (4, 4)),
                          dtypes=(self.torch.bfloat16, self.torch.bfloat16),
                          page_size_padded=attn.page_size_bytes,
                          mamba_cache_mode='align', num_prefill_checkpoint_blocks=1,
                          prefill_checkpoint_alignment=256)
        cache = KVCacheConfig(num_blocks=blocks, kv_cache_tensors=[], kv_cache_groups=[
            KVCacheGroupSpec(layer_names=['attention'], kv_cache_spec=attn),
            KVCacheGroupSpec(layer_names=['gdn'], kv_cache_spec=state)])
        env = {key: value for key, value in os.environ.items() if not key.startswith('VLLM_PD_')}
        # Unit branch coefficients only. Actual model calibration is a later gate.
        env.update(VLLM_PD_SCHEDULER_MODE=mode, VLLM_PD_K_MODE='ifr',
                   VLLM_PD_ALPHA_P='0.001', VLLM_PD_BETA_P='0.00001',
                   VLLM_PD_ALPHA_D='0.001', VLLM_PD_BETA_D='0.0001',
                   VLLM_PD_MB_COST_A='0.01')
        with patch.dict(os.environ, env, clear=True):
            return (cls or self.scheduler_cls)(config, cache, StructuredOutputManager(config), block_size=256)

    def test_native_control_observes_actual_allocator_without_policy(self):
        import json
        module=importlib.import_module('inferx_ebplus_scheduler')
        path=Path(self.tmp.name)/'native-runtime.json'
        with patch.dict(os.environ,INFERX_VLLM_RUNTIME=str(path)):
            scheduler=self.scheduler(cls=module.ObservedNativeScheduler)
        self.add(scheduler,'native-control')
        self.assertEqual(set(self.step(scheduler).num_scheduled_tokens),{'native-control'})
        state=json.loads(path.read_text())
        self.assertEqual(state['concurrency'],4)
        self.assertEqual(state['num_gpu_blocks'],256)
        self.assertEqual({g['type'] for g in state['kv_groups']},{'MambaSpec','FullAttentionSpec'})
        self.assertFalse(hasattr(scheduler,'pd_phase'))

    def test_calibration_request_uses_current_interface_and_no_cross_request_hits(self):
        from scripts.calibrate_ebplus_hardware import calibration_request
        first=calibration_request(256,5)
        second=calibration_request(256,5)
        self.assertEqual(len(first.prompt_token_ids),256)
        self.assertTrue(first.sampling_params.ignore_eos)
        self.assertEqual(first.sampling_params.temperature,0.0)
        self.assertEqual(first.cache_salt,first.request_id)
        self.assertNotEqual(first.cache_salt,second.cache_salt)

    def test_calibration_preserves_native_hybrid_alignment(self):
        from scripts.calibrate_ebplus_hardware import calibration_engine_args
        import json
        plan=json.loads((Path(__file__).parents[1]/'configs/ebplus-validation.json').read_text())
        config=calibration_engine_args(plan,Path(os.environ['INFERX_ROOT'])).create_engine_config()
        self.assertTrue(config.cache_config.enable_prefix_caching)
        self.assertEqual(config.cache_config.mamba_cache_mode,'align')
        self.assertEqual(config.model_config.max_model_len,131328)

    def add(self, scheduler, name, length=32, output=64, token_id=42):
        from vllm import SamplingParams
        from vllm.v1.request import Request
        from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
        from vllm.utils.hashing import get_hash_fn_by_name
        hasher = get_hash_fn_by_name('sha256')
        init_none_hash(hasher)
        request = Request(request_id=name, prompt_token_ids=[token_id] * length,
                          sampling_params=SamplingParams(max_tokens=output, ignore_eos=True),
                          pooling_params=None, block_hasher=get_request_block_hasher(256, hasher))
        scheduler.add_request(request)
        return request

    def step(self, scheduler):
        from vllm.v1.outputs import ModelRunnerOutput
        output = scheduler.schedule()
        ids = list(output.num_scheduled_tokens)
        runner = ModelRunnerOutput(
            req_ids=ids, req_id_to_index={rid: i for i, rid in enumerate(ids)},
            sampled_token_ids=[[] if scheduler.requests[rid].is_prefill_chunk else [7] for rid in ids])
        scheduler.update_from_output(output, runner)
        return output

    def test_full_native_running_queue_keeps_fifth_request_out_at_C4(self):
        scheduler = self.scheduler()
        active = [self.add(scheduler, f'active-{i}') for i in range(4)]
        waiting = self.add(scheduler, 'fifth')
        first = self.step(scheduler)
        self.assertEqual(set(first.num_scheduled_tokens), {r.request_id for r in active})
        self.assertEqual(len(scheduler.running), 4)
        blocks = {r.request_id: scheduler.kv_cache_manager.get_blocks(r.request_id).get_block_ids()
                  for r in active}
        second = self.step(scheduler)
        self.assertEqual(set(second.num_scheduled_tokens), set(blocks))
        self.assertIn(waiting, scheduler.waiting)
        self.assertEqual(len(scheduler.running), 4)
        for request in active:
            self.assertEqual(scheduler.kv_cache_manager.get_blocks(request.request_id).get_block_ids(),
                             blocks[request.request_id])
        self.assertEqual(scheduler._compute_adaptive_N(), 4)

    def test_partial_prefill_preserves_both_native_hybrid_owners(self):
        scheduler = self.scheduler()
        long = self.add(scheduler, 'long', length=5000)
        short = self.add(scheduler, 'short')
        self.step(scheduler)
        self.assertEqual(len(short.output_token_ids), 1)
        before = long.num_computed_tokens
        second = self.step(scheduler)
        self.assertEqual(set(second.num_scheduled_tokens), {'long'})
        self.assertGreater(long.num_computed_tokens, before)
        self.assertEqual(len(short.output_token_ids), 1)
        self.assertEqual(set(scheduler.running), {long, short})
        for _ in range(12):
            self.step(scheduler)
            if scheduler.pd_phase == 1:
                break
        self.assertEqual(scheduler.pd_phase, 1)
        self.assertGreater(len(short.output_token_ids), 1)

    def test_allocator_exception_retains_native_queues_and_clears_phase_guard(self):
        scheduler = self.scheduler()
        request = self.add(scheduler, 'retained')
        with patch.object(scheduler.kv_cache_manager, 'allocate_slots',
                          side_effect=RuntimeError('injected allocation failure')):
            with self.assertRaisesRegex(RuntimeError, 'injected allocation failure'):
                scheduler.schedule()
        self.assertFalse(scheduler._ebplus_exclusive)
        self.assertIs(scheduler.requests['retained'], request)
        self.assertIn(request, scheduler.waiting)
        self.assertEqual(scheduler.running, [])
        self.assertIn('retained', self.step(scheduler).num_scheduled_tokens)

    def test_actual_native_completion_hook_drives_default_crossover(self):
        scheduler = self.scheduler(mode='auto')
        for i in range(100):
            request = self.add(scheduler, f'sample-{i}', output=1)
            self.step(scheduler)
            self.assertNotIn(request.request_id, scheduler.requests)
            if i == 24:
                self.assertFalse(scheduler.pd_param_initialized)
                self.assertEqual(scheduler._active_scheduler, 'mb')
                self.assertEqual(scheduler._ebplus_evaluations,0)
        self.assertEqual(scheduler._active_scheduler, 'eb')
        self.assertEqual(scheduler._mode_switch_count, 1)
        self.assertEqual(scheduler.pd_total_completed, 100)
        self.assertEqual(scheduler._ebplus_evaluations,1)

    def test_shutdown_keeps_actual_policy_evidence(self):
        import json
        scheduler=self.scheduler()
        self.add(scheduler,'evidence',output=4)
        for _ in range(8):
            if not scheduler.has_requests():break
            self.step(scheduler)
        path=Path(self.tmp.name)/'evidence.json'
        with patch.dict(os.environ,INFERX_EBPLUS_EVIDENCE=str(path)):
            scheduler.shutdown()
        state=json.loads(path.read_text())
        self.assertEqual(state['mode'],'eb')
        self.assertGreater(state['execution_counts']['prefill'],0)
        self.assertGreater(state['execution_counts']['decode'],0)

    def test_online_snapshot_counts_the_100th_completion_once_and_keeps_decision_inputs(self):
        import json
        scheduler=self.scheduler(mode='auto')
        path=Path(self.tmp.name)/'online-evidence.json'
        with patch.dict(os.environ,INFERX_EBPLUS_EVIDENCE=str(path)):
            for i in range(100):
                if i==99:prompt_before=scheduler._avg_prompt_len
                self.add(scheduler,f'online-{i}',output=1)
                self.step(scheduler)
        state=json.loads(path.read_text())
        self.assertEqual(state['completed'],100)
        inputs=state['last_evaluation_inputs']
        self.assertEqual(inputs['mu_L'],prompt_before)
        self.assertEqual(inputs['mu_o'],1.)
        self.assertEqual(inputs['alpha_p'],scheduler.pd_alpha_p)
        self.assertEqual(inputs['mb_cost_a'],scheduler._mb_cost_a)
        self.assertEqual(state['evaluations'],1)

    def test_native_decode_progress_when_full_sequence_admission_refuses_a_distinct_long(self):
        module=importlib.import_module('inferx_ebplus_scheduler')
        scheduler=self.scheduler(cls=module.ObservedNativeScheduler,blocks=128)
        decoders=[self.add(scheduler,f'base-{i}',length=2560,output=64) for i in range(3)]
        for _ in range(16):
            self.step(scheduler)
            if all(len(r.output_token_ids)>0 for r in decoders):break
        long=self.add(scheduler,'native-full-fit-only-after-decode',length=30720,output=4,token_id=43)
        output=self.step(scheduler)
        self.assertEqual(set(output.num_scheduled_tokens),{r.request_id for r in decoders})
        self.assertIn(long,scheduler.waiting)

    def test_full_sequence_admission_refusal_cannot_freeze_live_decoders(self):
        isolated=self.scheduler(blocks=128)
        self.add(isolated,'fits-alone',length=30720,output=4,token_id=43)
        for _ in range(32):
            if not isolated.has_requests():break
            self.step(isolated)
        self.assertFalse(isolated.has_requests(),'The distinct long must fit and complete alone')
        scheduler = self.scheduler(blocks=128)
        decoders = [self.add(scheduler, f'held-{i}', length=2560, output=64) for i in range(3)]
        for _ in range(16):
            self.step(scheduler)
            if all(len(r.output_token_ids)>0 for r in decoders):break
        self.assertTrue(all(len(r.output_token_ids)>0 for r in decoders))
        self.assertTrue(scheduler.scheduler_reserve_full_isl)
        scheduler.pd_phase=1
        scheduler.pd_switch_threshold_k=1
        before={r.request_id:len(r.output_token_ids) for r in decoders}
        long=self.add(scheduler,'full-fit-only-after-decode',length=30720,output=4,token_id=43)
        # The native refusal can be learned from one empty schedule; it must not repeat forever.
        outputs=[self.step(scheduler),self.step(scheduler)]
        self.assertGreater(sum(o.total_num_scheduled_tokens for o in outputs),0,
            'Native full-sequence refusal must cause decode progress rather than an empty-schedule loop')
        self.assertTrue(any(set(o.num_scheduled_tokens)==set(before) for o in outputs))
        self.assertIn(long,scheduler.waiting)
        self.assertTrue(all(len(r.output_token_ids)>before[r.request_id] for r in decoders))
        empty_streak=0
        for _ in range(120):
            if not scheduler.has_requests():break
            output=self.step(scheduler)
            empty_streak=empty_streak+1 if output.total_num_scheduled_tokens==0 else 0
            self.assertLess(empty_streak,2,'No repeated empty-refill loop while decoders can release capacity')
        self.assertFalse(scheduler.has_requests())

    def test_sustained_empty_schedules_preserve_actual_queue_state_for_debugging(self):
        import json
        scheduler=self.scheduler(blocks=128)
        decoder=self.add(scheduler,'cooldown-decoder',length=2560,output=64)
        for _ in range(8):
            self.step(scheduler)
            if decoder.output_token_ids:break
        decoder.next_decode_eligible_step=10000
        scheduler.pd_phase=1
        scheduler.pd_switch_threshold_k=1
        long=self.add(scheduler,'waiting-distinct-long',length=30720,output=4,token_id=43)
        path=Path(self.tmp.name)/'empty-schedule-state.json'
        with patch.dict(os.environ,INFERX_EBPLUS_EVIDENCE=str(path)):
            for _ in range(80):self.step(scheduler)
        self.assertTrue(path.exists(),'A sustained empty loop needs an inspectable snapshot without ptrace')
        state=json.loads(path.read_text())
        self.assertGreaterEqual(state['zero_schedule_streak'],64)
        self.assertIn(decoder.request_id,state['scheduler_snapshot']['running'])
        self.assertIn(long.request_id,state['scheduler_snapshot']['waiting'])
        self.assertEqual(state['scheduler_snapshot']['requests'][decoder.request_id]['next_decode_eligible_step'],10000)

    def test_policy_preemption_uses_native_hybrid_free_and_resumes(self):
        from vllm.v1.request import RequestStatus
        scheduler = self.scheduler()
        request = self.add(scheduler, 'partial', length=5000)
        self.step(scheduler)
        self.assertGreater(request.num_computed_tokens, 0)
        self.assertLess(request.num_computed_tokens, request.num_prompt_tokens)
        computed_before = request.num_computed_tokens
        count, freed_computed_tokens = scheduler._preempt_chunk_prefilling()
        self.assertEqual(count, 1)
        self.assertEqual(freed_computed_tokens, computed_before)
        self.assertEqual(request.status, RequestStatus.PREEMPTED)
        self.assertEqual(request.num_computed_tokens, 0)
        self.assertNotIn(request, scheduler.running)
        self.assertIn(request, scheduler.waiting)
        self.assertEqual(scheduler.chunk_prefilling, [])
        self.assertIn('partial', self.step(scheduler).num_scheduled_tokens)


if __name__ == '__main__':
    unittest.main()
