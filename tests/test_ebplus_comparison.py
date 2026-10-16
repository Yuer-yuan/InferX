"""Actual comparison contracts: matched history, independent waves, original crossover."""
import copy
import importlib
import json
from pathlib import Path
import unittest


class ComparisonContracts(unittest.TestCase):
    def api(self):
        return importlib.import_module('inferx.ebplus_comparison')

    def test_catalog_preserves_capacity_and_exercises_both_author_policies(self):
        plan=json.loads((Path(__file__).parents[1]/'configs/ebplus-validation.json').read_text())
        cases=self.api().comparison_cases(plan)
        self.assertEqual([c['id'] for c in cases],['native','auto-default','forced-eb','auto-tuned',
                                                 'native-chunk-admit','auto-tuned-chunk-admit'])
        for c in cases:
            self.assertEqual((c['concurrency'],c['chunk'],c['long_prefill_token_threshold'],c['draft']), (4,4096,2048,0))
            self.assertEqual(c['num_gpu_blocks_override'],98)
            self.assertFalse(c['async_scheduling'])
        forced=copy.deepcopy(cases[2]);forced['id']='auto-default'
        forced['environment']['VLLM_PD_SCHEDULER_MODE']='auto'
        self.assertEqual(forced,cases[1])
        tuned=copy.deepcopy(cases[3]);tuned['id']='auto-default'
        self.assertEqual(tuned['environment'].pop('VLLM_PD_MODE_SWITCH_DELTA'),'0.000001')
        self.assertEqual(tuned,cases[1])

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



class ForcedEvidenceContracts(unittest.TestCase):
    def test_forced_policy_requires_actual_phase_work_and_matched_pool(self):
        from inferx.ebplus_comparison import audit_policy
        state=dict(mode='eb',active_scheduler='eb',execution_counts=dict(mb=0,prefill=200,decode=80),
                   completed=130,max_num_running_reqs=4,num_gpu_blocks=98)
        self.assertEqual(audit_policy(state,{}, {},delta=.0001)['expected_active'],'eb')
        state['num_gpu_blocks']=120
        with self.assertRaisesRegex(ValueError,'capacity'):
            audit_policy(state,{}, {},delta=.0001)

class ExclusiveAutoExecutionContracts(unittest.TestCase):
    def test_final_auto_eb_claim_requires_both_exclusive_phases_after_real_crossover(self):
        from inferx.ebplus_comparison import audit_policy
        hardware=dict(parameters=dict(alpha_p=0.,beta_p=.0001,alpha_d=.0001,beta_d=.0001))
        mb=dict(fit=dict(coefficients=dict(a=.00015,b=0.,c=0.)))
        state=dict(mode='auto',active_scheduler='eb',evaluations=1,parameters_initialized=True,mb_cost_profiled=True,
            execution_counts=dict(mb=100,prefill=0,decode=0),completed=130,max_num_running_reqs=4,num_gpu_blocks=98,
            last_evaluation_inputs=dict(hardware['parameters'],mb_cost_a=.00015,mb_cost_b=0.,mb_cost_c=0.,
                mu_L=100.,mu_o=100.,N_obs=2.,theta0=.5,alpha_mb=0.,delta=.000001,cooldown_before=0,active_before='mb'))
        audit_policy(state,hardware,mb,delta=.000001)  # Transition snapshot can precede execution.
        with self.assertRaisesRegex(ValueError,'exclusive'):
            audit_policy(state,hardware,mb,delta=.000001,require_exclusive_execution=True)
        state['execution_counts'].update(prefill=1,decode=1)
        self.assertEqual(audit_policy(state,hardware,mb,delta=.000001,require_exclusive_execution=True)['expected_active'],'eb')


if __name__=='__main__':unittest.main()

class FullInputAdmissionContracts(unittest.TestCase):
    def test_existing_admission_flag_variants_preserve_every_other_runtime_setting(self):
        from inferx.ebplus_comparison import comparison_cases,is_native_case
        plan=json.loads((Path(__file__).parents[1]/'configs/ebplus-validation.json').read_text())
        cases={c['id']:c for c in comparison_cases(plan)}
        for name,base in [('native-chunk-admit','native'),('auto-tuned-chunk-admit','auto-tuned')]:
            actual=copy.deepcopy(cases[name]);actual['id']=base
            self.assertIs(actual.pop('scheduler_reserve_full_isl'),False)
            self.assertEqual(actual,cases[base])
        self.assertTrue(is_native_case(cases['native-chunk-admit']))
        self.assertFalse(is_native_case(cases['auto-tuned-chunk-admit']))

    def test_official_boolean_cli_can_disable_full_input_fit_without_changing_chunk(self):
        from inferx.pd_baselines import engine_command
        plan=json.loads((Path(__file__).parents[1]/'configs/ebplus-validation.json').read_text())
        case=copy.deepcopy(plan['base_case']);case['scheduler_reserve_full_isl']=False
        command=engine_command(case,Path('/hdd'),Path('/hdd/run'))
        self.assertIn('--no-scheduler-reserve-full-isl',command)
        self.assertEqual(command[command.index('--max-num-batched-tokens')+1],'4096')
        for wrong in [0,'false',None]:
            case['scheduler_reserve_full_isl']=wrong
            with self.assertRaises(ValueError):engine_command(case,Path('/hdd'),Path('/hdd/run'))

class AdmissionRuntimeEvidenceContracts(unittest.TestCase):
    def test_new_runtime_schema_requires_actual_admission_flag_and_rejects_opposite_value(self):
        from inferx import ebplus_comparison as api
        plan=json.loads((Path(__file__).parents[1]/'configs/ebplus-validation.json').read_text())
        hardware=dict(native_runtime=dict(block_size=1568,num_gpu_blocks=98,kv_groups=[]))
        runtime=dict(model='/hdd/models/qwen38-int4-redhat',context_window=131328,concurrency=4,max_active=4,
            chunk=4096,long_prefill_token_threshold=2048,async_scheduling=False,enforce_eager=True,
            cudagraph_mode='NONE',speculative=False,quantization='compressed-tensors',kv_dtype='fp8',
            mamba_cache_mode='align',block_size=1568,num_gpu_blocks=98,kv_groups=[],scheduler_reserve_full_isl=False)
        case=dict(plan['base_case'],scheduler_reserve_full_isl=False)
        api.audit_native_runtime(runtime,plan,hardware,root=Path('/hdd'),case=case,runtime_evidence_version=2)
        runtime['scheduler_reserve_full_isl']=True
        with self.assertRaisesRegex(ValueError,'runtime'):
            api.audit_native_runtime(runtime,plan,hardware,root=Path('/hdd'),case=case,runtime_evidence_version=2)
        runtime.pop('scheduler_reserve_full_isl')
        with self.assertRaisesRegex(ValueError,'runtime'):
            api.audit_native_runtime(runtime,plan,hardware,root=Path('/hdd'),case=case,runtime_evidence_version=2)

class AdmissionCalibrationContracts(unittest.TestCase):
    def test_policy_requires_mb_profile_measured_under_its_admission_flag(self):
        from inferx.ebplus_comparison import validate_admission_profile
        case=dict(scheduler_reserve_full_isl=False)
        with self.assertRaisesRegex(ValueError,'admission'):validate_admission_profile(case,{})
        profile=dict(scheduler_reserve_full_isl=False,native_runtimes={})
        with self.assertRaisesRegex(ValueError,'runtime'):validate_admission_profile(case,profile)
        profile['native_runtimes']={f'mb-b{b}-cap{c}':dict(scheduler_reserve_full_isl=False,
            chunk=b,long_prefill_token_threshold=c,num_gpu_blocks=98,concurrency=4)
            for b in [1024,2048,4096] for c in [0,2048]}
        validate_admission_profile(case,profile)
        profile['native_runtimes']['mb-b1024-cap0']['scheduler_reserve_full_isl']=True
        with self.assertRaisesRegex(ValueError,'runtime'):validate_admission_profile(case,profile)
        validate_admission_profile({}, {})  # Historical pinned default-True cohort remains readable.
