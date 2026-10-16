"""Common SLOs come from independent native solos, never strategy outcomes."""
import copy
import json
from pathlib import Path
import unittest


class CommonBudgetContracts(unittest.TestCase):
    def data(self):
        identity=json.loads((Path(__file__).resolve().parents[1]/'configs/qoserve-validation.json').read_text())['identity']
        def record(rid,ttft,latency):
            return dict(request_id=rid,strict_ok=True,ttft_s=ttft,latency_s=latency,
                model_token_times_s=[ttft,ttft+.02,ttft+.04])
        return dict(identity=identity,phase='independent-native-solo',strategy='plain-native',
            short=[record(f's-{i}',i*.1,float(i)) for i in range(1,25)],
            cold=[record(f'c-{i}',1.,float(i)) for i in (80,90,100)],
            resume=[record(f'r-{i}',1.,float(i)) for i in (70,80,90)])

    def freeze(self,data=None):
        from inferx.qoserve_profile import freeze_common_budgets
        data=data or self.data()
        return freeze_common_budgets(data,self.data()['identity'])

    def test_quantiles_and_common_tier_budgets_have_literal_independent_values(self):
        contract=self.freeze()
        self.assertAlmostEqual(contract['interactive']['ttft_s'],3.45)
        self.assertAlmostEqual(contract['interactive']['tbt_s'],.03)
        self.assertAlmostEqual(contract['interactive']['practical_e2e_s'],34.5)
        self.assertEqual(contract['long']['cold']['ttlt_s'],112.5)
        self.assertEqual(contract['long']['resume']['ttlt_s'],100.)
        self.assertAlmostEqual(contract['long']['cold']['tbt_s'],.12)
        self.assertEqual(contract['label'],'independent proxy SLOs; not enterprise SLAs')

    def test_formal_strategy_or_incomplete_solo_records_cannot_define_common_slos(self):
        for mutation in ('formal','strategy','short-count','long-count','id','strict','time','identity'):
            data=self.data()
            if mutation=='formal':data['phase']='formal-mixed'
            elif mutation=='strategy':data['strategy']='qoserve'
            elif mutation=='short-count':data['short'].pop()
            elif mutation=='long-count':data['cold'].pop()
            elif mutation=='id':data['resume'][0]['request_id']=data['short'][0]['request_id']
            elif mutation=='strict':data['short'][0]['strict_ok']=False
            elif mutation=='time':data['short'][0]['model_token_times_s']=[.1,.1]
            else:data['identity']['draft']=7
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.freeze(data)

    def test_tags_preserve_the_entire_payload_and_use_no_output_length_oracle(self):
        from inferx.qoserve_profile import attach_common_contract
        from inferx.qoserve_validation import parse_contract
        original=dict(messages=[dict(role='user',content='original input')],tools=[dict(type='function',function=dict(name='f'))],
            max_tokens=96,temperature=0.,vllm_xargs={'unrelated':'keep'})
        before=copy.deepcopy(original);budgets=self.freeze()
        tagged=attach_common_contract(original,budgets,tier=1,kind='cold',trace_id='long-cold')
        self.assertEqual(original,before)
        tags={k:tagged['vllm_xargs'].pop(k) for k in ('qoserve_tier','qoserve_deadline_s','qoserve_tbt_s','qoserve_trace_id')}
        self.assertEqual(tagged,original)
        self.assertEqual(parse_contract(tags),dict(tier=1,deadline_s=112.5,tbt_s=budgets['long']['cold']['tbt_s'],trace_id='long-cold'))
        with self.assertRaises(ValueError):attach_common_contract(original,budgets,tier=2,kind='cold',trace_id='bad')


if __name__=='__main__':unittest.main()
