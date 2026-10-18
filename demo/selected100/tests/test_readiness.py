"""Readiness retries only read-only model probes for the verified child."""
import sys, types, unittest, urllib.error
from pathlib import Path
from unittest import mock
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import candidate_common as common

class EvaluationError(Exception):
    pass

def wrapped(cause):
    e=EvaluationError('not ready');e.__cause__=cause;return e

class ReadinessTests(unittest.TestCase):
    def run_wait(self, check, identity=None, clock=None):
        api=types.SimpleNamespace(EvaluationError=EvaluationError,check_model=check)
        with mock.patch.object(common,'check_service',identity or mock.Mock(return_value={'pid':123})), \
             mock.patch.object(common.time,'sleep'), \
             mock.patch.object(common.time,'monotonic',side_effect=clock or [0,1]):
            return common.wait_model_ready(api,'launch',{},'sha','quality100','http://127.0.0.1:1/v1',{'pid':123},'EMPTY')
    def test_loaded_model_passes_without_generation(self):
        c=mock.Mock(return_value={'id':'qwen38-27b'})
        r=self.run_wait(c);self.assertEqual(r['attempts'],1);self.assertEqual(r['generation_requests_sent'],0)
        c.assert_called_once_with('http://127.0.0.1:1/v1','qwen38-27b','EMPTY')
    def test_loading_timeout_then_ready_rechecks_identity(self):
        c=mock.Mock(side_effect=[wrapped(TimeoutError()),{'id':'qwen38-27b'}]);identity=mock.Mock(return_value={'pid':123})
        r=self.run_wait(c,identity,clock=[0,15,15,16]);self.assertEqual(r['attempts'],2);self.assertEqual(identity.call_count,3)
    def test_changed_process_refuses_probe(self):
        c=mock.Mock()
        with self.assertRaisesRegex(ValueError,'Service launch changed'):
            self.run_wait(c,mock.Mock(return_value={'pid':456}),clock=[0])
        c.assert_not_called()
    def test_nontransient_auth_failure_is_not_retried(self):
        c=mock.Mock(side_effect=wrapped(urllib.error.HTTPError('http://localhost',401,'unauthorized',{},None)))
        with self.assertRaises(EvaluationError):self.run_wait(c,clock=[0])
        self.assertEqual(c.call_count,1)
    def test_loading_timeout_has_bounded_wait(self):
        c=mock.Mock(side_effect=wrapped(TimeoutError()))
        with self.assertRaises(EvaluationError):self.run_wait(c,clock=[0,301])
        self.assertEqual(c.call_count,1)

if __name__=='__main__':unittest.main()
