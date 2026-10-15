"""Calibration identity, feasible workload, and actual-policy evidence gates."""
import copy
import json
import math
import os
from pathlib import Path
import unittest


class CalibrationContracts(unittest.TestCase):
    def api(self):
        import inferx.ebplus_calibration as api
        return api

    def plan(self):
        case = next(c for c in json.loads((Path(__file__).parents[1] / 'configs/pd-baselines.json').read_text())['cases']
                    if c['id'] == 'vllm031-int4-c4-b4096-cap2048-eager-mem95')
        return self.api().make_calibration_plan(case)

    def result(self):
        sizes=self.plan()['decomposed']['prefill_sizes']
        return dict(identity=self.plan()['identity'], measurement_type='decomposed_cuda_events',
                    sampling_temperature=0.0,ignore_eos=True,
                    parameters=dict(alpha_p=0.001, beta_p=0.00001, alpha_d=0.01, beta_d=0.0001),
                    prefill_sizes=sizes, decode_counts=[1,2,4],
                    raw_steps=[dict(stage='prefill',target=p,num_prefill=1,num_decode=0,tokens=p,elapsed_ms=1.2)
                               for p in sizes for _ in range(25)] +
                              [dict(stage='decode',target=n,num_prefill=0,num_decode=n,tokens=n,elapsed_ms=1.2)
                               for n in [1,2,4] for _ in range(25)])

    def test_calibration_cannot_be_reused_on_different_model_gpu_or_precision(self):
        for field in ('model_revision','gpu_uuid','kv_dtype','native_scheduler_sha256'):
            result=self.result();result['identity'][field]='different'
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'identity'):
                self.api().validate_hardware_result(result,self.plan())

    def test_plan_cannot_label_a_different_runtime_as_the_calibrated_identity(self):
        for field,value in [('gpu_uuid','other'),('kv_dtype','bfloat16'),('concurrency',8),
                            ('model_revision','other')]:
            plan=self.plan();plan['base_case'][field]=value
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'identity'):
                self.api().validate_plan(plan)

    def test_missing_counts_nonfinite_cost_and_cpu_timing_are_rejected(self):
        mutations=[lambda r:r.update(decode_counts=[1,2]),
                   lambda r:r['parameters'].update(alpha_d=float('nan')),
                   lambda r:r.update(measurement_type='scheduler_call_interval'),
                   lambda r:r.update(raw_steps=[]),
                   lambda r:r.update(sampling_temperature=1.0)]
        for mutate in mutations:
            result=self.result();mutate(result)
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):
                self.api().validate_hardware_result(result,self.plan())
        self.api().validate_hardware_result(self.result(),self.plan())

    def test_calibration_batch_must_fit_hard_C4_and_complete_prefill_step(self):
        plan=self.plan();plan['decomposed']['decode_counts']=[1,2,8]
        with self.assertRaisesRegex(ValueError,'C4'):
            self.api().validate_plan(plan)
        plan=self.plan();plan['decomposed']['prefill_sizes']=[256,8192]
        with self.assertRaisesRegex(ValueError,'prefill'):
            self.api().validate_plan(plan)

    def test_generated_plan_matches_qualified_hybrid_geometries(self):
        frozen=json.loads((Path(__file__).parents[1]/'configs/ebplus-validation.json').read_text())
        self.assertEqual(self.plan()['decomposed']['prefill_sizes'],frozen['decomposed']['prefill_sizes'])

    def test_old_restoration_record_does_not_authorize_current_service(self):
        from inferx.service_ownership import process_identity
        identity=process_identity(os.getpid(),Path('/proc/self/exe').resolve())
        campaign=dict(status='restored',restored=True,cases=[dict(state='measured')])
        restored=copy.deepcopy(identity);restored['proc_start_ticks']='0'
        with self.assertRaisesRegex(ValueError,'restor'):
            self.api().require_restored(campaign,restored,identity,expected_count=1)
        self.api().require_restored(campaign,identity,identity,expected_count=1)
        campaign['cases'][0]['state']='failed'
        with self.assertRaises(ValueError):
            self.api().require_restored(campaign,identity,identity,expected_count=1)

    def test_auto_evidence_requires_crossover_evaluation_with_measured_costs(self):
        state=dict(mode='auto',evaluations=0,parameters_initialized=False,mb_cost_profiled=True,
                   execution_counts=dict(mb=25,prefill=0,decode=0))
        with self.assertRaisesRegex(ValueError,'policy'):
            self.api().validate_policy_evidence(state)
        state.update(evaluations=1,parameters_initialized=True)
        self.api().validate_policy_evidence(state)  # Decision may legitimately retain MB.
        state['mb_cost_profiled']=False
        with self.assertRaises(ValueError):self.api().validate_policy_evidence(state)

    def test_failed_attempt_requires_fresh_completed_restoration_before_retry(self):
        from inferx.service_ownership import process_identity
        identity=process_identity(os.getpid(),Path('/proc/self/exe').resolve())
        state=dict(status='failed',restored=True,cases=[dict(state='failed')])
        self.api().require_terminal_restoration(state,identity,identity)
        state['status']='running'
        with self.assertRaises(ValueError):
            self.api().require_terminal_restoration(state,identity,identity)
        state.update(status='failed',restored=False)
        with self.assertRaises(ValueError):
            self.api().require_terminal_restoration(state,identity,identity)

    def test_actual_vllm_launch_can_select_calibrated_scheduler_synchronously(self):
        from inferx.pd_baselines import engine_command
        case=copy.deepcopy(self.plan()['base_case'])
        case.update(scheduler_class='inferx_ebplus_scheduler.EBPlusScheduler',async_scheduling=False,
                    num_gpu_blocks_override=98,shutdown_timeout=30)
        command=engine_command(case,Path('/hdd'),Path('/hdd/run'))
        self.assertIn('--no-async-scheduling',command)
        index=command.index('--scheduler-cls')
        self.assertEqual(command[index+1],'inferx_ebplus_scheduler.EBPlusScheduler')
        self.assertEqual(command[command.index('--num-gpu-blocks-override')+1],'98')
        self.assertEqual(command[command.index('--shutdown-timeout')+1],'30')

    def test_mb_fit_uses_best_independent_total_tps_and_requires_all_grid_points(self):
        plan=self.plan();rows=[]
        for scene in plan['mb_grid']['scenarios']:
            r=scene['output']/(scene['input']+scene['output'])
            for b in plan['mb_grid']['budgets']:
                for cap in plan['mb_grid']['prefill_caps']:
                    for n in plan['mb_grid']['counts']:
                        total=n*(scene['input']+scene['output'])
                        cost=(.001+.002*r+.003*r*r)*(4/n)
                        rows.append(dict(scenario=scene['name'],budget=b,cap=cap,count=n,
                            prompt_tokens=n*scene['input'],completion_tokens=n*scene['output'],
                            completed=n,duration_s=total*cost,total_token_throughput=1/cost))
        fit=self.api().fit_mb_cost(rows,plan)
        for k,v in [('a',.001),('b',.002),('c',.003)]:
            self.assertAlmostEqual(fit['coefficients'][k],v)
        with self.assertRaisesRegex(ValueError,'grid'):
            self.api().fit_mb_cost(rows[:-1],plan)
        rows[0]['prompt_tokens']-=1
        with self.assertRaisesRegex(ValueError,'tokens'):
            self.api().fit_mb_cost(rows,plan)

    def test_private_info_endpoint_absence_keeps_actual_native_runtime_evidence(self):
        from unittest.mock import patch
        from urllib.error import HTTPError
        with patch.dict(os.environ,INFERX_ROOT='/unused-test-root'):
            from scripts.qualify_ebplus_model import configuration_snapshot
        runtime=dict(context_window=131328,concurrency=4,num_gpu_blocks=98)
        def missing(path):raise HTTPError(path,404,'not exposed',{},None)
        snapshot=configuration_snapshot(runtime,missing)
        self.assertEqual(snapshot['native_runtime'],runtime)
        self.assertEqual(snapshot['server_info_http_status'],404)
        def rejected(path):raise HTTPError(path,403,'forbidden',{},None)
        with self.assertRaises(HTTPError):configuration_snapshot(runtime,rejected)


if __name__=='__main__':
    unittest.main()
