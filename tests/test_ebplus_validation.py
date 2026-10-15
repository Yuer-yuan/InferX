"""Pinned author-policy execution; no GPU or substitute production allocator."""
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace


class AuthorPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        capsule = Path(__file__).parents[1] / 'results/ebplus-validation/source-capsule-v1.tar.gz'
        with tarfile.open(capsule) as archive:
            for name in ('author_scheduler.py', 'author_calibration.py', 'current_scheduler.py', 'author_LICENSE'):
                (cls.root / name).write_bytes(archive.extractfile(name).read())

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def api(self):
        import inferx.ebplus_validation as api
        return api

    def policy(self, extra_env=None):
        source = self.api().extract_author_policy(
            self.root / 'author_scheduler.py', self.root / 'author_calibration.py')
        path = self.root / 'policy.py'
        path.write_text(source)
        spec = importlib.util.spec_from_file_location('inferx_ebplus_policy_test', path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        state = module.AuthorPolicy()
        state.max_num_running_reqs = 4
        state.running = []
        state.waiting = []
        env = {key: value for key, value in os.environ.items() if not key.startswith('VLLM_PD_')}
        env.update(VLLM_PD_SCHEDULER_MODE='auto', VLLM_PD_K_MODE='ifr',
                   VLLM_PD_ALPHA_P='0.001', VLLM_PD_BETA_P='0.00001',
                   VLLM_PD_ALPHA_D='0.001', VLLM_PD_BETA_D='0.0001',
                   VLLM_PD_MB_COST_A='0.01')
        env.update(extra_env or {})
        with patch.dict(os.environ, env, clear=True):
            state._initialize_author_policy()
        return state

    def test_source_mismatch_is_rejected_before_extraction(self):
        # A changed upstream controller must never silently keep the old label.
        altered = self.root / 'altered.py'
        altered.write_bytes((self.root / 'author_scheduler.py').read_bytes() + b'\n# changed\n')
        with self.assertRaisesRegex(ValueError, 'author.*hash'):
            self.api().extract_author_policy(altered, self.root / 'author_calibration.py')

    def test_exact_cfr_threshold_and_small_batch_midpoint_execute_author_math(self):
        # Wrong sign/bisection or dropping the N=1 branch changes these values.
        state = self.policy()
        state.pd_alpha_p = state.pd_alpha_d = 1.0
        self.assertAlmostEqual(state._compute_theta_zero_exact(0.5), 0.6266353822983259)
        self.assertEqual(state._compute_midpoint_k(0.5, 0.5, 1, 10), (1.0, 0, 0, 0))
        self.assertEqual(state._compute_midpoint_k(0.5, 0.5, 4, 10), (1.0, 1, 2, 1))

    def test_default_cold_start_does_not_pretend_to_switch_on_25_requests(self):
        state = self.policy()
        for _ in range(25):
            state._update_params_online(64)
        self.assertFalse(state.pd_param_initialized)
        self.assertEqual(state._active_scheduler, 'mb')
        self.assertEqual(state._mode_switch_history, [])

    def test_actual_crossover_runs_after_author_completion_threshold(self):
        state = self.policy()
        for _ in range(99):
            state._update_params_online(64)
        self.assertEqual(state._active_scheduler, 'mb')
        state._update_params_online(64)
        self.assertTrue(state.pd_param_initialized)
        self.assertEqual(state._active_scheduler, 'eb')
        self.assertEqual(state._mode_switch_count, 1)
        self.assertGreater(state._mode_switch_history[0]['LHS'],
                           state._mode_switch_history[0]['RHS'])

    def test_calibration_mutation_is_rejected_independently(self):
        altered = self.root / 'bad_calibration.py'
        altered.write_bytes((self.root / 'author_calibration.py').read_bytes() + b'\n')
        with self.assertRaisesRegex(ValueError, 'calibration.*hash'):
            self.api().extract_author_policy(self.root / 'author_scheduler.py', altered)

    def test_prepare_refuses_existing_destination_without_overwriting(self):
        destination = self.root / 'existing'
        destination.mkdir(exist_ok=True)
        marker = destination / 'untouched'
        marker.write_text('keep')
        with self.assertRaises(FileExistsError):
            self.api().prepare_adapter(self.root / 'author_scheduler.py',
                                       self.root / 'author_calibration.py',
                                       self.root / 'current_scheduler.py', destination)
        self.assertEqual(marker.read_text(), 'keep')

    def test_phase_gates_keep_decode_out_of_prefill_and_prefill_out_of_decode(self):
        prefill = SimpleNamespace(request_id='long', num_computed_tokens=32, num_prompt_tokens=128)
        decode = SimpleNamespace(request_id='short', num_computed_tokens=128, num_prompt_tokens=128)
        gate = self.api().phase_allows_running
        self.assertTrue(gate(0, prefill, {'short'}))
        self.assertFalse(gate(0, decode, {'short'}))
        self.assertTrue(gate(1, decode, {'short'}))
        self.assertFalse(gate(1, decode, set()))
        self.assertFalse(gate(1, prefill, {'short'}))
        self.assertTrue(gate(2, prefill, {'short'}))
        self.assertFalse(gate(2, decode, {'short'}))

    def test_pending_native_prefill_completions_count_toward_refill_target(self):
        requests = {
            'partial': SimpleNamespace(num_computed_tokens=32, num_prompt_tokens=128),
            'complete': SimpleNamespace(num_computed_tokens=96, num_prompt_tokens=128),
            'decode': SimpleNamespace(num_computed_tokens=128, num_prompt_tokens=128),
        }
        gate = self.api().phase_allows_admission
        self.assertTrue(gate(2, 2, 0, requests, {'partial': 32, 'complete': 32}))
        self.assertFalse(gate(2, 2, 1, requests, {'partial': 32, 'complete': 32}))
        self.assertFalse(gate(1, 4, 0, requests, {}))
        self.assertTrue(gate(0, 4, 2, requests, {'decode': 1}))

    def test_changed_native_source_cannot_publish_a_partially_prepared_adapter(self):
        altered = self.root / 'bad_native.py'
        altered.write_bytes((self.root / 'current_scheduler.py').read_bytes() + b'\n')
        destination = self.root / 'invalid_native_overlay'
        with self.assertRaisesRegex(ValueError, 'native.*hash'):
            self.api().prepare_adapter(self.root / 'author_scheduler.py',
                                       self.root / 'author_calibration.py', altered, destination)
        self.assertFalse(destination.exists())

    def test_prepared_overlay_is_compilable_and_binds_all_input_bytes(self):
        destination = self.root / 'complete_overlay'
        manifest = self.api().prepare_adapter(
            self.root / 'author_scheduler.py', self.root / 'author_calibration.py',
            self.root / 'current_scheduler.py', destination)
        for name in ('inferx_ebplus_policy.py', 'inferx_ebplus_scheduler.py'):
            raw = (destination / name).read_bytes()
            compile(raw, name, 'exec')
            self.assertEqual(hashlib.sha256(raw).hexdigest(), manifest['files'][name])
        self.assertEqual(hashlib.sha256((destination / 'LICENSE').read_bytes()).hexdigest(),
                         'c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4')
        self.assertEqual(json.loads((destination / 'manifest.json').read_text()), manifest)
        # Failed publication must leave the exact prior overlay intact.
        prior = (destination / 'manifest.json').read_bytes()
        with self.assertRaises(FileExistsError):
            self.api().prepare_adapter(self.root / 'author_scheduler.py',
                                       self.root / 'author_calibration.py',
                                       self.root / 'current_scheduler.py', destination)
        self.assertEqual((destination / 'manifest.json').read_bytes(), prior)


if __name__ == '__main__':
    unittest.main()
