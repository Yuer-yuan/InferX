"""Real-model calibration protocol, independent of formal benchmark outcomes."""
import copy
import importlib
import json
import hashlib
from pathlib import Path
import tempfile
import unittest


class CollectionProtocolContracts(unittest.TestCase):
    def api(self):
        try:return importlib.import_module('scripts.calibrate_qoserve_model')
        except ModuleNotFoundError as exc:self.fail(f'missing real-model collector: {exc}')

    def plan(self):
        return json.loads((Path(__file__).resolve().parents[1]/'configs/qoserve-validation.json').read_text())

    def test_matrix_and_salts_are_fixed_before_model_execution(self):
        specs=self.api().rollout_specs(self.plan(),'engine-one')
        self.assertEqual(len(specs),39)
        self.assertEqual(len({s['long_salt'] for s in specs}),39)
        train=[s for s in specs if s['kind']=='train']
        self.assertEqual({(s['prefill_target'],s['decoders']) for s in train},
            {(p,d) for p in (128,512,2048,2552) for d in (0,1,3)})
        hold=[s for s in specs if s['kind']=='holdout']
        self.assertEqual([s['decoders'] for s in hold],[0,2])
        self.assertTrue(all(s['long_salt']!=s['decoder_salt'] for s in specs))
        self.assertTrue(set(s['long_salt'] for s in specs).isdisjoint(
            s['long_salt'] for s in self.api().rollout_specs(self.plan(),'engine-two')))

    def test_initial_throughput_warmups_are_separate_from_regression_rollouts(self):
        specs=self.api().rollout_specs(self.plan(),'engine')
        pps=[s for s in specs if s['kind']=='initial-pps']
        self.assertEqual(len(pps),25)
        self.assertEqual([s['warmup_steps'] for s in pps],[1]*5+[0]*20)
        self.assertTrue(all(s['prompt_len']==512 and s['decoders']==0 for s in pps))
        self.assertTrue(all(s['prompt_len']==131072 and s['warmup_steps']==5
            for s in specs if s['kind']!='initial-pps'))

    def test_repair_uses_new_train_salts_and_fresh_holdouts_with_bounded_context_diversity(self):
        path=Path(__file__).resolve().parents[1]/'configs/qoserve-profile-repair-v1.json'
        definition=json.loads(path.read_text());api=self.api()
        specs=api.repair_rollout_specs(self.plan(),'new-repair-engine',definition)
        self.assertEqual(len(specs),37)
        training=[s for s in specs if s['kind']=='train']
        self.assertEqual({(s['decoders'],s['decoder_prompt_len']) for s in training},
            {(0,24557),(1,512),(1,24557),(2,24557),(3,24557),(1,65536),(1,131072)})
        self.assertTrue(all(s['budget_mode']=='total-cycle' for s in specs if s['kind']!='initial-pps'))
        self.assertTrue(all(s['candidates']==[32,64,96,128,256,384,512,768,1024,1280,1536,1792,2048,2552]
            for s in training))
        held=[s for s in specs if s['kind']=='holdout']
        self.assertEqual([(s['holdout_protocol'],s['decoders']) for s in held],
            [('original-cycle-v1',0),('original-cycle-v1',2),('remainder-cycle-v1',0),('remainder-cycle-v1',2)])
        self.assertEqual(held[0]['candidates'],self.plan()['profile']['candidates'])
        diagnostic=[s for s in specs if s['kind']=='diagnostic']
        self.assertEqual(len(diagnostic),1)
        self.assertEqual((diagnostic[0]['decoders'],diagnostic[0]['candidates']),(0,[128]))
        self.assertTrue({s['long_salt'] for s in specs}.isdisjoint(
            s['long_salt'] for s in api.rollout_specs(self.plan(),'original-engine')))
        high=next(s for s in training if s['decoder_prompt_len']==131072)
        self.assertEqual(high['decoder_max_tokens'],256)
        for mutation in ('identity','old-holdout','candidate','missing-context','missing-original-cycle'):
            d=copy.deepcopy(definition)
            if mutation=='identity':d['identity']['draft']=7
            elif mutation=='old-holdout':d['replace_initial_holdout']=False
            elif mutation=='candidate':d['calibration_candidates'].remove(32)
            elif mutation=='missing-context':d['train_decoder_geometries'].pop()
            else:d['holdout_candidate_cycles'].pop(0)
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                api.repair_rollout_specs(self.plan(),'new-repair-engine',d)

    def test_repair_basis_is_the_exact_completed_independent_fit(self):
        path=Path(__file__).resolve().parents[1]/'configs/qoserve-profile-repair-v1.json'
        definition=json.loads(path.read_text());api=self.api()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);basis=root/definition['basis_fit_path'];basis.parent.mkdir(parents=True)
            value=dict(qualified=False,identity=self.plan()['identity'],input_evidence={
                'archive_sha256':definition['basis_archive_sha256']})
            basis.write_text(json.dumps(value));definition['basis_fit_sha256']=hashlib.sha256(basis.read_bytes()).hexdigest()
            inv=root/'qoserve-validation/exports/independent-profile-v5-inventory.json';inv.parent.mkdir(parents=True)
            inv.write_text(json.dumps({'archive_sha256':definition['basis_archive_sha256']}))
            api.verify_repair_basis(definition,root)
            basis.write_text(basis.read_text()+' ')
            with self.assertRaisesRegex(ValueError,'hash'):api.verify_repair_basis(definition,root)
            escaped=dict(definition,basis_fit_path='../outside.json')
            with self.assertRaises(ValueError):api.verify_repair_basis(escaped,root)

    def test_profile_configuration_rejects_runtime_or_geometry_drift(self):
        api=self.api();plan=self.plan();api.validate_profile_plan(plan)
        for mutation in ('gpu','draft','blocks','matrix','candidates','long'):
            p=copy.deepcopy(plan)
            if mutation=='gpu':p['base_case']['gpu_uuid']='other'
            elif mutation=='draft':p['identity']['draft']=7
            elif mutation=='blocks':p['profile']['num_gpu_blocks']=99
            elif mutation=='matrix':p['profile']['prefill_targets']=[128]
            elif mutation=='candidates':p['profile']['candidates'].remove(384)
            else:p['profile']['long_prompt_len']=131329
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):api.validate_profile_plan(p)

    def test_controller_requires_current_restoration_identity_and_exact_original_arguments(self):
        try:api=importlib.import_module('scripts.run_qoserve_validation')
        except ModuleNotFoundError as exc:self.fail(f'missing owned profile controller: {exc}')
        from inferx.service_ownership import process_identity
        import os
        current=process_identity(os.getpid(),Path('/proc/self/exe').resolve())
        terminal=dict(status='failed',restored=True)
        argv=['/hdd/ninfer-serve','/hdd/model','--ctx-size','262144']
        api.validate_prerequisite(terminal,current,current,argv,argv)
        for mutation in ('live','unrestored','stale','argv'):
            campaign=copy.deepcopy(terminal);restored=copy.deepcopy(current);actual=list(argv)
            if mutation=='live':campaign['status']='running'
            elif mutation=='unrestored':campaign['restored']=False
            elif mutation=='stale':restored['proc_start_ticks']='0'
            else:actual[-1]='131328'
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                api.validate_prerequisite(campaign,restored,current,actual,argv)


if __name__=='__main__':unittest.main()
