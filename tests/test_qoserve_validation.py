"""Caller contracts and source policy views, independent of CUDA imports."""
import importlib
import json
import hashlib
from pathlib import Path
from types import SimpleNamespace
import unittest
import tempfile
import tarfile
import subprocess
import sys


class QoServeViewContracts(unittest.TestCase):
    def api(self):
        try:
            return importlib.import_module('inferx.qoserve_validation')
        except ModuleNotFoundError as exc:
            self.fail(f'QoServe native boundary has no caller-contract/view implementation: {exc}')

    def policy(self):
        from inferx.qoserve_policy import load_author_policy
        root=Path(__file__).resolve().parents[1]/'results/qoserve-validation'
        return load_author_policy(root/'author-source-v1.tar.gz',
                                  json.loads((root/'author-source-v1.json').read_text()))

    def request(self,tags=None):
        return SimpleNamespace(request_id='r',arrival_time=1990.,num_prompt_tokens=100,
            num_tokens=100,num_computed_tokens=20,output_token_ids=[],
            sampling_params=SimpleNamespace(extra_args=tags))

    def test_only_untagged_checks_receive_source_defaults(self):
        api=self.api()
        self.assertEqual(api.parse_contract(None),dict(tier=0,deadline_s=6.,tbt_s=.1,trace_id=None))
        self.assertEqual(api.parse_contract({'unrelated':'keep'}),api.parse_contract(None))
        with self.assertRaises(ValueError):api.parse_contract({'qoserve_tier':0})

    def test_caller_contract_uses_tier_ttft_or_ttlt_without_generation_oracle(self):
        self.assertEqual(self.api().parse_contract(dict(qoserve_tier=1,qoserve_deadline_s=600,
            qoserve_tbt_s=.4,qoserve_trace_id='long-r0')),dict(tier=1,deadline_s=600.,tbt_s=.4,trace_id='long-r0'))
        tags=dict(qoserve_tier=0,qoserve_deadline_s=3.,qoserve_tbt_s=.02)
        self.assertEqual(self.api().parse_contract(tags)['deadline_s'],3.)
        self.assertEqual(tags,dict(qoserve_tier=0,qoserve_deadline_s=3.,qoserve_tbt_s=.02))

    def test_bad_numeric_tiers_deadlines_and_trace_ids_are_rejected(self):
        normal=dict(qoserve_tier=0,qoserve_deadline_s=6.,qoserve_tbt_s=.1,qoserve_trace_id='short')
        for key,value in [('qoserve_tier',True),('qoserve_tier',2),('qoserve_tier',.0),
                          ('qoserve_deadline_s',0),('qoserve_deadline_s',-1),
                          ('qoserve_deadline_s',float('inf')),('qoserve_deadline_s',float('nan')),
                          ('qoserve_tbt_s',True),('qoserve_tbt_s',0),('qoserve_trace_id',''),
                          ('qoserve_trace_id',2),('qoserve_output_tokens',512)]:
            tags=dict(normal);tags[key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                self.api().parse_contract(tags)

    def test_sequence_view_maps_original_clock_once_and_uses_unknown_length_15(self):
        request=self.request(dict(qoserve_tier=1,qoserve_deadline_s=600.,qoserve_tbt_s=.4))
        view=self.api().sequence_view_type(self.policy())(request,wall_to_monotonic=-1000.)
        self.assertEqual((view.arrival_time,view.state._arrived_at),(990.,990.))
        self.assertEqual((view.ttft_deadline,view.num_decode_tokens),(594.,-1))
        request.arrival_time=50000.
        self.assertEqual(view.arrival_time,990.)
        self.assertEqual(view.get_prompt_len()-view.get_num_prompt_tokens_processed(),80)

    def test_view_preserves_live_native_progress_and_author_context_semantics(self):
        request=self.request()
        view=self.api().sequence_view_type(self.policy())(request,wall_to_monotonic=-1000.)
        self.assertFalse(view.prompt_processing_finished)
        self.assertEqual(view._get_context_size(),20)
        request.num_computed_tokens=101
        request.num_tokens=102
        request.output_token_ids.extend([1,2])
        self.assertTrue(view.prompt_processing_finished)
        self.assertEqual(view._get_context_size(),102)  #author prompt-processed + outputs
        view.state.e2e_prefill_time=8.
        self.assertAlmostEqual(view._get_TBT_deadline(.1,.4),998.3)
        # Replayed input contains past output; age/drop/first-token time survive.
        view.drop=1
        request.num_computed_tokens=0
        self.assertFalse(view.prompt_processing_finished)
        self.assertEqual(view.get_prompt_len(),101)
        self.assertEqual((view.arrival_time,view.drop,view.state.e2e_prefill_time),(990.,1,8.))

    def test_tier_budget_uses_request_tbt_without_mutating_global_source_defaults(self):
        policy=self.policy()
        request=self.request(dict(qoserve_tier=1,qoserve_deadline_s=100.,qoserve_tbt_s=.2))
        view=self.api().sequence_view_type(policy)(request,wall_to_monotonic=-1000.)
        self.assertEqual(view.ttft_deadline,97.)
        self.assertEqual(policy.execution_threshold_batched,.4)

    def test_real_cached_prefix_informs_planning_without_advancing_native_request(self):
        request=self.request()
        request.num_computed_tokens=0
        view=self.api().sequence_view_type(self.policy())(request,wall_to_monotonic=-1000.)
        view.cached_prefix_tokens=80  #actual native cache-hit count supplied by the dispatcher.
        self.assertEqual(view.get_prompt_len()-view.get_num_prompt_tokens_processed(),20)
        self.assertEqual(view._get_context_size(),80)
        self.assertEqual(request.num_computed_tokens,0)
        view.cached_prefix_tokens=0
        self.assertEqual(view.get_num_prompt_tokens_processed(),0)

    def test_builder_refuses_existing_overlay_without_overwriting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'keep').write_text('original')
            with self.assertRaises(FileExistsError):
                self.api().prepare_adapter(root/'absent',root/'absent.py',root)
            self.assertEqual((root/'keep').read_text(),'original')

    def test_builder_rejects_changed_native_source_before_writing_overlay(self):
        evidence=Path(__file__).resolve().parents[1]/'results/qoserve-validation'
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);native=root/'scheduler.py';native.write_text('changed')
            with self.assertRaisesRegex(ValueError,'native.*hash'):
                self.api().prepare_adapter(evidence,native,root/'overlay')
            self.assertFalse((root/'overlay').exists())

    def test_reference_overlay_is_reproducible_but_marked_unqualified_for_deployment(self):
        evidence=Path(__file__).resolve().parents[1]/'results/qoserve-validation'
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);native=root/'scheduler.py'
            with tarfile.open(evidence/'native-source-v1.tar.gz') as archive:
                native.write_bytes(archive.extractfile('vllm/v1/core/sched/scheduler.py').read())
            manifest=self.api().prepare_adapter(evidence,native,root/'overlay')
            self.assertFalse(manifest['policy']['deployment_coefficients_supplied'])
            self.assertEqual(manifest['native_scheduler_sha256'],'f820bee4f42b878b262a30f3728a97d7413477b2a277748e20f2528ac2ec2f2a')
            for name,want in manifest['files'].items():
                self.assertEqual(hashlib.sha256((root/'overlay'/name).read_bytes()).hexdigest(),want)
            compile((root/'overlay/inferx_qoserve_scheduler.py').read_text(),'<native-overlay>','exec')

    def test_cli_prepares_a_source_reference_overlay_without_importing_cuda(self):
        repo=Path(__file__).resolve().parents[1];evidence=repo/'results/qoserve-validation'
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);native=root/'scheduler.py'
            with tarfile.open(evidence/'native-source-v1.tar.gz') as archive:
                native.write_bytes(archive.extractfile('vllm/v1/core/sched/scheduler.py').read())
            result=subprocess.run([sys.executable,str(repo/'scripts/prepare_qoserve_validation.py'),
                '--author-root',str(evidence),'--native-scheduler',str(native),'--output',str(root/'overlay')],
                cwd=repo,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            manifest=json.loads((root/'overlay/manifest.json').read_text())
            self.assertFalse(manifest['policy']['deployment_coefficients_supplied'])
            self.assertEqual(json.loads(result.stdout)['destination'],str(root/'overlay'))


if __name__=='__main__':unittest.main()
