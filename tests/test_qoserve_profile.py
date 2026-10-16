"""Calibration rows must describe completed real engine steps, not targets."""
import importlib
import json
from pathlib import Path
import unittest


class ProfileRowContracts(unittest.TestCase):
    def api(self):
        try:return importlib.import_module('inferx.qoserve_profile')
        except ModuleNotFoundError as exc:self.fail(f'QoServe completed-step validation is missing: {exc}')

    def identity(self):
        path=Path(__file__).resolve().parents[1]/'configs/ebplus-validation.json'
        return json.loads(path.read_text())['identity']

    def row(self):
        return dict(row_id='train-p128-d1-step7',phase='train',engine_id='engine-0',
            rollout_id='train-p128-d1',cache_salt='independent-train-salt',step_index=7,
            warmup=False,identity=self.identity(),proposed_total_tokens=129,
            timing=dict(start_s=100.,end_s=100.125,batch_execution_time=.125,
                        boundary='schedule-entry-to-native-update-exit',completed_model_step=True),
            requests=[dict(request_id='long',computed_before=512,prefill_end=2048,
                           scheduled_tokens=128,stage='prefill',output_tokens_before=0),
                      dict(request_id='decoder',computed_before=24557,prefill_end=24557,
                           scheduled_tokens=1,stage='decode',output_tokens_before=1)],
            features=dict(batch_num_tokens=129,batch_num_prefill_tokens=128,
                batch_num_decode_tokens=1,batch_decode_context=24558,batch_prefill_context=512))

    def test_completed_step_uses_real_prefill_and_latest_decode_input_contexts(self):
        self.assertEqual(self.api().validate_step_row(self.row(),self.identity()),
            dict(batch_num_tokens=129,batch_num_prefill_tokens=128,
                 batch_num_decode_tokens=1,batch_decode_context=24558,batch_prefill_context=512))

    def test_training_budget_cycle_keeps_the_training_phase_and_original_fixed_mode(self):
        api=self.api();cfg=dict(phase='train',prefill_target=128,candidates=[32,128,384])
        self.assertEqual(api.proposed_profile_budget(cfg,3,1),131)
        cfg['budget_mode']='total-cycle'
        self.assertEqual([api.proposed_profile_budget(cfg,2,i) for i in range(4)],[32,128,384,32])
        self.assertEqual(cfg['phase'],'train')
        self.assertEqual(api.proposed_profile_budget(cfg,2,2,ceiling=256),256)
        cfg['budget_mode']='unknown'
        with self.assertRaises(ValueError):api.proposed_profile_budget(cfg,2,0)

    def test_replayed_old_outputs_remain_prefill_until_the_native_boundary(self):
        row=self.row();row['requests']=[dict(request_id='replay',computed_before=50,
            prefill_end=199,scheduled_tokens=128,stage='prefill',output_tokens_before=100)]
        row['features']=dict(batch_num_tokens=128,batch_num_prefill_tokens=128,
            batch_num_decode_tokens=0,batch_decode_context=0,batch_prefill_context=50)
        self.assertEqual(self.api().validate_step_row(row,self.identity())['batch_num_prefill_tokens'],128)

    def test_targets_and_prompt_lengths_cannot_replace_executed_geometry(self):
        for field,value in [('batch_num_tokens',512),('batch_num_prefill_tokens',512),
                            ('batch_prefill_context',2048),('batch_decode_context',24557)]:
            row=self.row();row['features'][field]=value
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'geometry'):
                self.api().validate_step_row(row,self.identity())

    def test_timing_must_span_the_same_completed_model_step(self):
        for key,value in [('completed_model_step',False),('boundary','worker-cuda-event'),
                          ('end_s',99.),('batch_execution_time',125.),
                          ('batch_execution_time',float('nan')),('start_s',True)]:
            row=self.row();row['timing'][key]=value
            if key=='start_s' and value is True:row['timing']['end_s']=1.125
            with self.subTest(key=key,value=value),self.assertRaisesRegex(ValueError,'timing|completed'):
                self.api().validate_step_row(row,self.identity())

    def test_wrong_hardware_formal_samples_and_unjoined_request_ids_are_rejected(self):
        for mutation in ('identity','formal','duplicate','c4','budget','bool-tokens','wrong-stage'):
            row=self.row()
            if mutation=='identity':row['identity']['gpu_uuid']='other-gpu'
            elif mutation=='formal':row['phase']='mixed'
            elif mutation=='duplicate':row['requests'][1]['request_id']='long'
            elif mutation=='c4':
                row['requests'] += [dict(row['requests'][1],request_id=f'd{i}') for i in range(3)]
                row['features'].update(batch_num_tokens=132,batch_num_decode_tokens=4,batch_decode_context=98232)
                row['proposed_total_tokens']=132
            elif mutation=='budget':row['proposed_total_tokens']=128
            elif mutation=='bool-tokens':
                row['requests'][0]['scheduled_tokens']=True
                row['features'].update(batch_num_tokens=2,batch_num_prefill_tokens=1)
                row['proposed_total_tokens']=2
            else:row['requests'][0]['stage']='decode'
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                self.api().validate_step_row(row,self.identity())

    def test_uncached_zero_work_or_nonpositive_clock_is_not_a_training_row(self):
        for mutation in ('zero','negative','future','missing-id','warmup-string'):
            row=self.row()
            if mutation=='zero':row['requests'][0]['scheduled_tokens']=0
            elif mutation=='negative':
                row['requests'][0]['computed_before']=-1
                row['features']['batch_prefill_context']=-1
            elif mutation=='future':row['requests'][0]['prefill_end']=200000
            elif mutation=='missing-id':row['row_id']=''
            else:row['warmup']='false'
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                self.api().validate_step_row(row,self.identity())


if __name__=='__main__':unittest.main()
