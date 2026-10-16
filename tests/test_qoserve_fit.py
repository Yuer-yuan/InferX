"""Synthetic algebra validates the fitter; it is never GPU efficacy evidence."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import unittest


@unittest.skipUnless(os.environ.get('INFERX_QOSERVE_FIT_CONTRACT')=='1' or importlib.util.find_spec('sklearn'),
                     'isolated scientific fitter environment required')
class StatisticalProfileContracts(unittest.TestCase):
    def plan(self):
        return json.loads((Path(__file__).resolve().parents[1]/'configs/qoserve-validation.json').read_text())

    def row(self,index,target,phase='train',slow=False):
        identity=self.plan()['identity'];context=(0,8192,32768,98304)[index%4]
        decoders=(0,1,3)[index%3];prefill=target-decoders
        requests=[dict(request_id='long',computed_before=context,prefill_end=131072,
            scheduled_tokens=prefill,stage='prefill',output_tokens_before=0)]
        for d in range(decoders):
            position=24557+index%100+d
            requests.append(dict(request_id=f'd{d}',computed_before=position,prefill_end=position,
                scheduled_tokens=1,stage='decode',output_tokens_before=1))
        features=dict(batch_num_tokens=target,batch_num_prefill_tokens=prefill,
            batch_num_decode_tokens=decoders,batch_decode_context=sum(r['computed_before']+1 for r in requests[1:]),
            batch_prefill_context=context)
        duration=.02+.0001*target+.003*decoders+.0000004*features['batch_decode_context']+(.00002 if slow else .0000008)*context
        return dict(row_id=f'{phase}-{index}-{target}',phase=phase,engine_id='synthetic-algebra',
            rollout_id=phase,cache_salt=phase+'-salt',step_index=index,warmup=False,identity=identity,
            proposed_total_tokens=target,requests=requests,features=features,
            timing=dict(start_s=100.,end_s=100.+duration,batch_execution_time=duration,
                boundary='schedule-entry-to-native-update-exit',completed_model_step=True))

    def data(self,slow=False):
        train=[self.row(i,(128,256,512,768,1024,2048)[i%6],slow=slow) for i in range(480)]
        holdout=[self.row(1000+i*11+j,target,'holdout',slow=slow)
            for i in range(8) for j,target in enumerate(self.plan()['profile']['candidates'])]
        return train,holdout

    def fit(self,rows=None,holdout=None,plan=None):
        from inferx.qoserve_profile import fit_profile
        if rows is None:rows,holdout=self.data()
        return fit_profile(rows,holdout,plan or self.plan())

    def test_original_ols_split_and_two_bin_formula_are_reproducible(self):
        first=self.fit();second=self.fit()
        self.assertEqual(first,second)
        self.assertTrue(first['qualified'],first['qualification_reasons'])
        self.assertEqual(first['fit_protocol'],dict(algorithm='sklearn.linear_model.LinearRegression',
            test_size=.2,random_state=42,training_upper_excludes_513=True,runtime_upper_starts_at_513=True))
        for name,info in first['bins'].items():
            self.assertGreaterEqual(info['eligible_training_rows'],100)
            self.assertLessEqual(info['internal_test']['max_absolute_error_s'],1e-8)
            self.assertEqual(info['holdout']['guard_coverage'],1.)
            # Total=P+D in both bins. The lower synthetic target/decode pairs
            # also satisfy D=(total-128)/128, adding a second dependency.
            self.assertEqual(info['rank'],{'tokens_0_512':3,'tokens_513_plus':4}[name])

    def test_exact_513_training_edge_is_excluded_but_holdout_uses_upper(self):
        rows,holdout=self.data();edge=self.row(900,513)
        rows.append(edge);holdout.append(self.row(900,513,'holdout'))
        result=self.fit(rows,holdout)
        self.assertIn(edge['row_id'],result['training_exclusions']['source-edge-513'])
        self.assertEqual(result['bins']['tokens_513_plus']['holdout']['count'],57)

    def test_small_bin_candidate_or_context_coverage_cannot_hide_in_pooled_accuracy(self):
        rows,holdout=self.data()
        result=self.fit(rows[:30],holdout)
        self.assertFalse(result['qualified']);self.assertTrue(any('training' in s for s in result['qualification_reasons']))
        result=self.fit(rows,[r for r in holdout if r['proposed_total_tokens']!=384])
        self.assertFalse(result['qualified']);self.assertTrue(any('candidate 384' in s for s in result['qualification_reasons']))
        result=self.fit(rows,[r for r in holdout if r['features']['batch_prefill_context']<65536])
        self.assertFalse(result['qualified']);self.assertTrue(any('context' in s for s in result['qualification_reasons']))

    def test_holdout_identity_salts_ids_and_formal_outcomes_never_enter_fit(self):
        for mutation in ('salt','rollout','row','identity','formal'):
            rows,holdout=self.data()
            if mutation=='salt':holdout[0]['cache_salt']=rows[0]['cache_salt']
            elif mutation=='rollout':holdout[0]['rollout_id']=rows[0]['rollout_id']
            elif mutation=='row':holdout[0]['row_id']=rows[0]['row_id']
            elif mutation=='identity':holdout[0]['identity']['draft']=7
            else:rows[0]['phase']='formal'
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.fit(rows,holdout)

    def test_five_second_fallback_requires_actual_context_or_geometry_censoring(self):
        ordinary=self.fit();self.assertEqual(ordinary['cutoff_s'],1.)
        rows,holdout=self.data(slow=True);result=self.fit(rows,holdout)
        self.assertEqual(result['cutoff_s'],5.)
        self.assertTrue(result['fallback']['activated'])
        self.assertTrue(result['fallback']['source_censored_row_ids'])
        self.assertTrue(result['qualified'],result['qualification_reasons'])
        rows,holdout=self.data();extra=copy.deepcopy(rows[0]);extra['row_id']='outlier-existing-geometry'
        extra['timing'].update(end_s=102.,batch_execution_time=2.);rows.append(extra)
        result=self.fit(rows,holdout)
        self.assertEqual(result['cutoff_s'],1.)
        self.assertIn(extra['row_id'],result['training_exclusions']['above-cutoff'])

    def test_guard_is_separate_for_every_bin_and_candidate_without_holdout_refitting(self):
        rows,holdout=self.data();baseline=self.fit(rows,holdout)
        for row in holdout:
            if row['proposed_total_tokens']==2552:
                duration=row['timing']['batch_execution_time']*3
                row['timing'].update(end_s=100.+duration,batch_execution_time=duration)
        result=self.fit(rows,holdout)
        self.assertEqual(result['coefficients'],baseline['coefficients'])
        self.assertFalse(result['qualified'])
        self.assertEqual(result['candidates']['2552']['guard_coverage'],0.)
        self.assertTrue(any('candidate 2552 guard' in s for s in result['qualification_reasons']))

    def test_positive_training_times_do_not_certify_nonpositive_tail_predictions(self):
        rows,holdout=self.data()
        for row in rows+holdout:
            duration=-.01+.0001*row['features']['batch_num_tokens']
            row['timing'].update(end_s=100.+duration,batch_execution_time=duration)
        tail=self.row(1002,1,'holdout');tail['proposed_total_tokens']=128
        tail['timing'].update(end_s=100.005,batch_execution_time=.005)
        holdout.append(tail)
        result=self.fit(rows,holdout)
        self.assertFalse(result['qualified'])
        self.assertIn(tail['row_id'],result['prediction_failures'])


if __name__=='__main__':unittest.main()
