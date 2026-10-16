"""Actual comparison contracts: matched history, independent waves, original crossover."""
import copy
import importlib
import json
from pathlib import Path
import unittest


class ComparisonContracts(unittest.TestCase):
    def api(self):
        return importlib.import_module('inferx.ebplus_comparison')

    def test_bounded_cases_preserve_native_capacity_and_exercise_both_author_policies(self):
        plan=json.loads((Path(__file__).parents[1]/'configs/ebplus-validation.json').read_text())
        cases=self.api().comparison_cases(plan)
        self.assertEqual([c['id'] for c in cases],['native','auto-default','forced-eb'])
        for c in cases:
            self.assertEqual((c['concurrency'],c['chunk'],c['long_prefill_token_threshold'],c['draft']), (4,4096,2048,0))
            self.assertEqual(c['num_gpu_blocks_override'],98)
            self.assertFalse(c['async_scheduling'])
        forced=copy.deepcopy(cases[2]);forced['id']='auto-default'
        forced['environment']['VLLM_PD_SCHEDULER_MODE']='auto'
        self.assertEqual(forced,cases[1])

    def test_frozen_specs_use_full_long_and_serial_solos_keep_server_C4(self):
        templates={name:dict(payload={'messages':[name]},expected_tool={'name':'record','arguments':{}})
                   for name in ['tool-24576','cold-131072','resume-131072']}
        mixed=self.api().comparison_specs(templates,'cold-low','x',kind='mixed')
        self.assertEqual([s['offset_s'] for s in mixed[:-1]],list(range(0,48,2)))
        self.assertEqual((mixed[-1]['prompt_id'],mixed[-1]['offset_s']),('cold-131072',16))
        solo=self.api().comparison_specs(templates,'short-control','x',kind='isolated-serial-C4')
        self.assertEqual(len(solo),24)
        self.assertTrue(all(s['producer']==0 and 'offset_s' not in s for s in solo))
        with self.assertRaises(ValueError):self.api().comparison_specs(templates,'cold-high','x',kind='mixed')

    def test_independent_crossover_reconstruction_respects_hysteresis_and_cooldown(self):
        x=dict(mu_L=100.,mu_o=100.,N_obs=2.,theta0=.5,alpha_mb=0.,alpha_p=0.,alpha_d=.0001,
               beta_p=.0001,beta_d=.0001,mb_cost_a=.00015,mb_cost_b=0.,mb_cost_c=0.,
               delta=.0001,cooldown_before=0,active_before='mb')
        # LHS=0.00005, RHS=0.0001*ln(2)/2≈0.00003466; margin is below delta.
        self.assertEqual(self.api().crossover_decision(x)['expected_active'],'mb')
        x['delta']=0.
        self.assertEqual(self.api().crossover_decision(x)['expected_active'],'eb')
        x['cooldown_before']=1
        self.assertEqual(self.api().crossover_decision(x)['expected_active'],'mb')
        x['theta0']=0
        with self.assertRaises(ValueError):self.api().crossover_decision(x)


class PolicyEvidenceContracts(unittest.TestCase):
    def test_claimed_mode_and_calibration_must_match_observed_original_decision(self):
        from inferx import ebplus_comparison as api
        hardware=dict(parameters=dict(alpha_p=0.,beta_p=.0001,alpha_d=.0001,beta_d=.0001))
        mb=dict(fit=dict(coefficients=dict(a=.00015,b=0.,c=0.)))
        x=dict(mu_L=100.,mu_o=100.,N_obs=2.,theta0=.5,alpha_mb=0.,alpha_p=0.,alpha_d=.0001,
               beta_p=.0001,beta_d=.0001,mb_cost_a=.00015,mb_cost_b=0.,mb_cost_c=0.,
               delta=0.,cooldown_before=0,active_before='mb')
        state=dict(mode='auto',active_scheduler='mb',evaluations=1,parameters_initialized=True,
                   mb_cost_profiled=True,execution_counts=dict(mb=20,prefill=0,decode=0),
                   last_evaluation_inputs=x,completed=100,max_num_running_reqs=4,num_gpu_blocks=98)
        with self.assertRaisesRegex(ValueError,'decision'):
            api.audit_policy(state,hardware,mb,delta=0.)
        state['active_scheduler']='eb'
        self.assertEqual(api.audit_policy(state,hardware,mb,delta=0.)['expected_active'],'eb')
        state['last_evaluation_inputs']['beta_p']=.00001
        with self.assertRaisesRegex(ValueError,'calibration'):
            api.audit_policy(state,hardware,mb,delta=0.)

class RuntimeEvidenceContracts(unittest.TestCase):
    def test_actual_native_runtime_maps_calibrator_names_and_rejects_precision_changes(self):
        from inferx import ebplus_comparison as api
        plan=json.loads((Path(__file__).parents[1]/'configs/ebplus-validation.json').read_text())
        hardware=dict(native_runtime=dict(block_size=1568,num_gpu_blocks=98,kv_groups=[{'type':'MambaSpec','spec':'BF16conv FP32GDN','layers':['gdn']}]))
        runtime=dict(model='/hdd/models/qwen38-int4-redhat',context_window=131328,concurrency=4,max_active=4,
                     chunk=4096,long_prefill_token_threshold=2048,async_scheduling=False,enforce_eager=True,
                     cudagraph_mode='NONE',speculative=False,quantization='compressed-tensors',kv_dtype='fp8',
                     mamba_cache_mode='align',block_size=1568,num_gpu_blocks=98,kv_groups=hardware['native_runtime']['kv_groups'])
        api.audit_native_runtime(runtime,plan,hardware,root=Path('/hdd'))
        runtime['kv_groups']=[{'type':'MambaSpec','spec':'BF16conv BF16GDN','layers':['gdn']}]
        with self.assertRaisesRegex(ValueError,'runtime'):
            api.audit_native_runtime(runtime,plan,hardware,root=Path('/hdd'))


if __name__=='__main__':unittest.main()

class ForcedEvidenceContracts(unittest.TestCase):
    def test_forced_policy_requires_actual_phase_work_and_matched_pool(self):
        from inferx.ebplus_comparison import audit_policy
        state=dict(mode='eb',active_scheduler='eb',execution_counts=dict(mb=0,prefill=200,decode=80),
                   completed=130,max_num_running_reqs=4,num_gpu_blocks=98)
        self.assertEqual(audit_policy(state,{}, {},delta=.0001)['expected_active'],'eb')
        state['num_gpu_blocks']=120
        with self.assertRaisesRegex(ValueError,'capacity'):
            audit_policy(state,{}, {},delta=.0001)
