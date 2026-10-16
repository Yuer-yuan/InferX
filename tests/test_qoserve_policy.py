"""Execute pinned QoServe mathematics; no substitute native allocator."""
import hashlib
import heapq
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'results/qoserve-validation'


class QoServePolicyTests(unittest.TestCase):
    def api(self):
        try:
            import inferx.qoserve_policy as api
        except ModuleNotFoundError as exc:
            self.fail(f'pinned QoServe policy loader is missing: {exc}')
        return api

    def manifest(self):
        return json.loads((EVIDENCE / 'author-source-v1.json').read_text())

    def policy_type(self, coefficients=None):
        return self.api().load_author_policy(
            EVIDENCE / 'author-source-v1.tar.gz', self.manifest(), coefficients)

    def policy(self, free_blocks=100):
        cls = self.policy_type()

        class State(cls):
            def _qoserve_effective_free_blocks(self):
                return free_blocks

        state = State()
        state.block_size = 1568
        state._prefill_queue = []
        return state

    def seq(self, **overrides):
        cls = self.policy_type().Sequence

        class View(cls):
            def get_prompt_len(self):
                return self.prompt_tokens

            def get_num_prompt_tokens_processed(self):
                return self.computed_tokens

        seq = View()
        values = dict(arrival_time=100., ttft_deadline=6., prompt_tokens=1000,
                      computed_tokens=0, scheduler_type='deadline',
                      hybrid_prioritization_param=.008, drop=0,
                      mem_allocated=False, prompt_processing_finished=False,
                      state=SimpleNamespace(_arrived_at=100., e2e_prefill_time=2.),
                      output_token_ids=[1, 2, 3], tier=0, tier_deadline=600.,
                      num_decode_tokens=-1)
        values.update(overrides)
        seq.__dict__.update(values)
        return seq

    def test_changed_capsule_cannot_retain_author_label(self):
        # Removing the byte check would accept a modified source artifact.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'changed.tar.gz'
            path.write_bytes((EVIDENCE / 'author-source-v1.tar.gz').read_bytes() + b'changed')
            with self.assertRaisesRegex(ValueError, 'capsule.*hash'):
                self.api().load_author_policy(path, self.manifest())

    def test_changed_commit_license_or_inventory_is_rejected(self):
        for mutation in ('commit', 'license', 'inventory'):
            manifest = self.manifest()
            if mutation == 'commit':
                manifest['commit'] = '0' * 40
            elif mutation == 'license':
                manifest['files']['LICENSE'] = '0' * 64
            else:
                del manifest['files']['LICENSE']
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.api().load_author_policy(EVIDENCE / 'author-source-v1.tar.gz', manifest)

    def test_reference_a100_tables_cannot_be_used_as_deployment_defaults(self):
        cls = self.policy_type()
        self.assertFalse(cls.deployment_coefficients_supplied)
        with self.assertRaisesRegex(ValueError, 'deployment.*coefficients'):
            cls.require_deployment_coefficients()

    def test_original_predictor_selects_upper_model_at_513(self):
        # An off-by-one model boundary changes both literal predictions.
        state = self.policy()
        self.assertAlmostEqual(state._predict_batch_time(0, 0, 512, 0), .05656380672)
        self.assertAlmostEqual(state._predict_batch_time(0, 0, 513, 0), .05416013709)

    def injected_coefficients(self):
        lower = {'coefficients': dict(batch_num_tokens=.001, batch_num_prefill_tokens=.002,
                                      batch_num_decode_tokens=.003, batch_decode_context=.004,
                                      batch_prefill_context=.005), 'intercept': .01}
        upper = {'coefficients': {key: value * 2 for key, value in lower['coefficients'].items()},
                 'intercept': .02}
        return {'tokens_0_512': lower, 'tokens_513_plus': upper}

    def test_hardware_tables_replace_both_bins_without_changing_features(self):
        coefficients = self.injected_coefficients()
        cls = self.policy_type(coefficients)
        cls.require_deployment_coefficients()
        state = cls()
        self.assertAlmostEqual(state._predict_batch_time(2, 3, 4, 1), .049)
        self.assertAlmostEqual(state._predict_batch_time(0, 0, 512, 1), 3.1)
        # Caller mutation after extraction must not retune a live policy.
        coefficients['tokens_0_512']['intercept'] = 9000
        self.assertAlmostEqual(state._predict_batch_time(2, 3, 4, 1), .049)

    def test_invalid_coefficients_fail_before_policy_execution(self):
        for mutation in ('missing', 'nan', 'bool', 'extra'):
            coefficients = self.injected_coefficients()
            if mutation == 'missing':
                del coefficients['tokens_513_plus']
            elif mutation == 'extra':
                coefficients['tokens_0_512']['coefficients']['unknown'] = 1
            else:
                coefficients['tokens_0_512']['intercept'] = float('nan') if mutation == 'nan' else True
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.policy_type(coefficients)
        coefficients = self.injected_coefficients()
        coefficients['tokens_0_512']['coefficients']['batch_num_decode_tokens'] = -.003
        self.assertAlmostEqual(self.policy_type(coefficients)()._predict_batch_time(2, 3, 4, 1), .043)

    def test_chunk_search_uses_original_guard_candidates_and_native_capacity(self):
        state = self.policy()
        self.assertEqual(state._get_prefill_size_by_slack(100, 0, 0, 3)[0], 2549)
        self.assertEqual(state._get_prefill_size_by_slack(.06, 0, 0, 3)[0], 253)
        constrained = self.policy(free_blocks=1)
        self.assertEqual(constrained._get_prefill_size_by_slack(100, 0, 0, 3)[0], 509)
        self.assertEqual(constrained._get_prefill_size_by_slack(.06, 0, 0, 3)[0], 381)

    def test_nonpositive_slack_still_uses_minimum_chunk(self):
        for slack in (0, -10):
            with self.subTest(slack=slack):
                self.assertEqual(self.policy()._get_prefill_size_by_slack(slack, 0, 0, 3)[0], 125)

    def test_prefill_heap_preserves_drop_then_hybrid_deadline_priority(self):
        first = self.seq(arrival_time=10, ttft_deadline=6, prompt_tokens=100)
        second = self.seq(arrival_time=9, ttft_deadline=8, prompt_tokens=0)
        queue = [second, first]
        heapq.heapify(queue)
        self.assertIs(heapq.heappop(queue), first)  #16.8 <17
        first.drop = 1
        queue = [first, second]
        heapq.heapify(queue)
        self.assertIs(heapq.heappop(queue), second)

    def test_fcfs_edf_and_srpf_keep_their_distinct_order(self):
        first = self.seq(arrival_time=10, ttft_deadline=6, prompt_tokens=100)
        second = self.seq(arrival_time=9, ttft_deadline=8, prompt_tokens=0)
        for mode, want in (('fcfs', False), ('edf', True), ('srpf', False)):
            first.scheduler_type = second.scheduler_type = mode
            with self.subTest(mode=mode):
                self.assertEqual(first < second, want)

    def test_decode_deadlines_are_cumulative_and_use_actual_prefill_if_later(self):
        seq = self.seq(prompt_processing_finished=True)
        self.assertAlmostEqual(seq._get_TBT_deadline(.1, .4), 106.4)
        key = self.policy()._sorting_key(seq, 105)
        self.assertEqual(key[0], 0)
        self.assertAlmostEqual(key[1], -1.4)
        seq.state.e2e_prefill_time = 8
        self.assertAlmostEqual(seq._get_TBT_deadline(.1, .4), 108.4)
        seq.tier = 1
        self.assertAlmostEqual(seq._get_TBT_deadline(.1, .4), 109.6)
        self.assertEqual(self.policy()._sorting_key(self.seq(), 105), (1, 106.))

    def test_tier1_budget_uses_predicted_length_without_future_output_oracle(self):
        state = self.policy()
        unknown = self.seq(tier=1, num_decode_tokens=-1)
        self.assertEqual(state._qoserve_ttft_budget(unknown), 594.)
        known = self.seq(tier=1, num_decode_tokens=100)
        self.assertEqual(state._qoserve_ttft_budget(known), 560.)
        self.assertEqual(state._qoserve_ttft_budget(self.seq()), 6.)

    def test_eager_relegation_requeues_once_without_dropping_request(self):
        state = self.policy()
        state.expected_prefill_throughput = 100
        seq = self.seq(arrival_time=0, ttft_deadline=5)
        self.assertTrue(state._qoserve_relegate_once(seq, 0))
        self.assertEqual(seq.drop, 1)
        self.assertIs(heapq.heappop(state._prefill_queue), seq)
        self.assertFalse(state._qoserve_relegate_once(seq, 0))
        self.assertEqual(state._prefill_queue, [])
        self.assertFalse(state._qoserve_relegate_once(self.seq(arrival_time=0, ttft_deadline=5, mem_allocated=True), 0))
        state.scheduler_type = 'fcfs'
        self.assertFalse(state._qoserve_relegate_once(self.seq(arrival_time=0, ttft_deadline=5), 0))

    def test_ema_uses_elapsed_schedule_time_and_resets_observation(self):
        state = self.policy()
        state.last_prefill_size, state.prev_call_time = 1000, 10.
        state._qoserve_update_prefill_throughput(12.)
        self.assertAlmostEqual(state.expected_prefill_throughput, 8460.)
        self.assertEqual((state.last_prefill_size, state.prev_call_time), (0, 12.))
        state._qoserve_update_prefill_throughput(13.)
        self.assertAlmostEqual(state.expected_prefill_throughput, 8460.)

    def test_manifest_binds_original_and_generated_policy_and_timing_bytes(self):
        api = self.api()
        manifest = api.policy_manifest(EVIDENCE / 'author-source-v1.tar.gz', self.manifest(),
                                       timing_source=EVIDENCE / 'author-timing-source-v1.py')
        self.assertEqual(manifest['commit'], '72339db5b120614033b317880ad3a05a5e4c9415')
        self.assertEqual(manifest['timing_source_sha256'], '0ea48ab80e4236feb1d5944a30bb5acf2b165bd6b27843c881027daf9ff5e9e1')
        self.assertEqual(manifest['coefficient_source'], 'author-reference')
        self.assertEqual(set(manifest['methods']), {'_predict_batch_time', '_get_prefill_size_by_slack',
            '_sorting_key', '__lt__', '_get_TBT_deadline', '_qoserve_ttft_budget',
            '_qoserve_relegate_once', '_qoserve_update_prefill_throughput'})
        self.assertEqual(hashlib.sha256(self.policy_type()._module_source.encode()).hexdigest(),
                         manifest['generated_module_sha256'])
        self.assertTrue(all(len(row['source_sha256']) == len(row['generated_sha256']) == 64
                            for row in manifest['methods'].values()))
        with tempfile.TemporaryDirectory() as tmp:
            changed = Path(tmp) / 'timing.py'
            changed.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'timing.*hash'):
                api.policy_manifest(EVIDENCE / 'author-source-v1.tar.gz', self.manifest(), timing_source=changed)


if __name__ == '__main__':
    unittest.main()
