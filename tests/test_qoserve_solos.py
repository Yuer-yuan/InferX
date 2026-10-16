"""Client completion and actual engine token records must join by caller trace."""
import copy
import importlib
import unittest


class SoloJoinContracts(unittest.TestCase):
    def api(self):
        try:return importlib.import_module('scripts.calibrate_qoserve_budgets')
        except ModuleNotFoundError as exc:self.fail(f'missing matched solo collector: {exc}')

    def records(self):
        clients=[dict(id='short-0',functional_ok=True,summary={'ok':True},arrival_ttft_s=.15,latency_s=.3)]
        observer=dict(requests={'engine-generated-id':dict(trace_id='short-0',
            model_token_times_s=[.1,.12,.14],finish_status='FINISHED_STOPPED')})
        return clients,observer

    def test_engine_token_timestamps_are_joined_without_using_sse_chunk_gaps(self):
        client,observer=self.records()
        self.assertEqual(self.api().join_solo_records(client,observer),[
            dict(request_id='short-0',strict_ok=True,ttft_s=.15,latency_s=.3,model_token_times_s=[.1,.12,.14])])

    def test_missing_duplicate_or_aborted_engine_trace_is_not_successful_calibration(self):
        for mutation in ('missing','duplicate','abort','strict','client-duplicate'):
            clients,observer=self.records();engine=observer['requests']['engine-generated-id']
            if mutation=='missing':engine['trace_id']=None
            elif mutation=='duplicate':observer['requests']['second']=copy.deepcopy(engine)
            elif mutation=='abort':engine['finish_status']='FINISHED_ABORTED'
            elif mutation=='strict':clients[0]['functional_ok']=False
            else:clients.append(copy.deepcopy(clients[0]))
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.api().join_solo_records(clients,observer)


if __name__=='__main__':unittest.main()
