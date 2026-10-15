"""Owned EB+ calibration jobs with exact original-service restoration."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from inferx.ebplus_calibration import validate_plan,require_restored,validate_hardware_result
from inferx.service_ownership import process_identity
from inferx.sota_results import verify_prompt_count_contract
from inferx.ebplus_validation import CALIBRATION_SHA256
from scripts.run_pd_baselines import run_with_restoration
from scripts import run_concurrency_campaign as previous


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--plan',required=True,type=Path)
    a=p.parse_args();assert Path(a.run_id).name==a.run_id and a.run_id not in {'.','..'}
    root=Path(os.environ['INFERX_ROOT']);plan=json.loads(a.plan.read_text());validate_plan(plan)
    saved=json.loads((root/'tuning/best-service.json').read_text());assert saved==json.loads((root/'context-agents/runtime.json').read_text())
    identity=process_identity(saved['pid'],previous.BINARY);assert all(saved.get(k)==v for k,v in identity.items())
    prior=root/'morphx-control/raw/solo-local-v1'
    require_restored(json.loads((prior/'campaign.json').read_text()),json.loads((prior/'restored-service.json').read_text()),saved,expected_count=6)
    saved['original_argv']=Path(f"/proc/{saved['pid']}/cmdline").read_bytes().decode().rstrip('\0').split('\0')
    assert saved['original_argv']==json.loads((prior/'original-service.json').read_text())['original_argv']
    verify_prompt_count_contract(root/plan['base_case']['model'])
    source=root/'sota-validation/sources/eb-vllm-repo/vllm/v1/core/sched/calibration.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==CALIBRATION_SHA256
    out=root/'ebplus-validation/raw'/a.run_id;out.mkdir(parents=True,exist_ok=False);previous.DIRECTORY=out
    (out/'plan.json').write_bytes(a.plan.read_bytes());(out/'controller.py').write_bytes(Path(__file__).read_bytes())
    (out/'original-service.json').write_text(json.dumps(saved,indent=2)+'\n')
    metadata=dict(status='running',controller_pid=os.getpid(),started_unix=time.time(),cases=[],restored=False)
    def flush():(out/'campaign.json').write_text(json.dumps(metadata,indent=2)+'\n')
    def measure():
        previous.gpu_free();entry=dict(case='decomposed-hardware',state='starting');metadata['cases'].append(entry);flush()
        code=Path(__file__).resolve().parents[1];worker=code/'scripts/calibrate_ebplus_hardware.py'
        command=['taskset','-c','28-35',str(root/plan['base_case']['binary']),'-u',str(worker),
                 '--plan',str(out/'plan.json'),'--output',str(out/'hardware-result.json')]
        (out/'worker.py').write_bytes(worker.read_bytes())
        env={**os.environ,**plan['base_case']['environment'],'PYTHONPATH':str(code),
             'CUDA_VISIBLE_DEVICES':plan['identity']['gpu_uuid'],'HF_ENDPOINT':'https://hf-mirror.com',
             'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TMPDIR':str(root/'tmp')}
        entry.update(command=command,worker_sha256=hashlib.sha256(worker.read_bytes()).hexdigest());flush()
        with (out/'worker.log').open('x') as log:
            child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                   env=env,start_new_session=True)
            entry['child_pid']=child.pid;flush()
            try:
                status=child.wait(timeout=1800)
            finally:
                # A live unreaped direct child cannot have its PID recycled.
                if child.poll() is None:
                    child.terminate()
                    try:child.wait(timeout=20)
                    except subprocess.TimeoutExpired:child.kill();child.wait()
            if status!=0:
                entry.update(state='failed',exit_code=status);flush();raise RuntimeError('hardware calibration failed; keep raw, no efficacy score')
        result=json.loads((out/'hardware-result.json').read_text());validate_hardware_result(result,plan)
        assert result['status']=='complete'
        entry.update(state='measured',result_sha256=hashlib.sha256((out/'hardware-result.json').read_bytes()).hexdigest());flush()
    flush()
    try:
        run_with_restoration(saved,previous.restore,[root/'context-agents/runtime.json',root/'tuning/best-service.json'],
                             lambda:previous.final_shutdown_check(saved,gpu0_state='stopped'),measure)
        metadata['status']='restored'
    except BaseException as exc:
        metadata.update(status='failed',error=repr(exc));raise
    finally:
        metadata.update(finished_unix=time.time(),restored=(out/'restored-service.json').exists());flush()


if __name__=='__main__':main()
