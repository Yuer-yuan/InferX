"""Owned QoServe experiment controller with exact service restoration."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from inferx.ebplus_calibration import require_terminal_restoration


def validate_prerequisite(campaign,restored,current,actual_argv,original_argv):
    require_terminal_restoration(campaign,restored,current)
    if actual_argv!=original_argv:
        raise ValueError('original service arguments changed')


def main():
    from inferx.pd_baselines import snapshot_harness,run_owned_command
    from inferx.service_ownership import process_identity
    from inferx.sota_results import verify_prompt_count_contract
    from scripts.calibrate_qoserve_model import validate_profile_plan
    from scripts.run_pd_baselines import run_with_restoration
    from scripts import run_concurrency_campaign as previous
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True)
    parser.add_argument('--stage',choices=('profile','budgets'),default='profile')
    parser.add_argument('--plan',required=True,type=Path);parser.add_argument('--prerequisite',required=True,type=Path)
    parser.add_argument('--overlay',required=True,type=Path)
    parser.add_argument('--repair-definition',type=Path)
    args=parser.parse_args();root=Path(os.environ['INFERX_ROOT']).resolve()
    if Path(args.run_id).name!=args.run_id or args.run_id in ('.','..'):raise ValueError('invalid run ID')
    for path in (args.plan,args.prerequisite,args.overlay):
        if not path.resolve().is_relative_to(root):raise ValueError('all experiment files must reside on HDD')
    plan=json.loads(args.plan.read_text());validate_profile_plan(plan)
    expected_rollouts=39
    if args.repair_definition:
        from scripts.calibrate_qoserve_model import verify_repair_basis,repair_rollout_specs
        if args.stage!='profile' or not args.repair_definition.resolve().is_relative_to(root):
            raise ValueError('repair definition requires an HDD profiling stage')
        definition=json.loads(args.repair_definition.read_text());verify_repair_basis(definition,root)
        expected_rollouts=len(repair_rollout_specs(plan,'preflight-only',definition))
    saved=json.loads((root/'tuning/best-service.json').read_text())
    assert saved==json.loads((root/'context-agents/runtime.json').read_text())
    current=process_identity(saved['pid'],previous.BINARY)
    assert all(saved.get(k)==v for k,v in current.items())
    saved['original_argv']=Path(f"/proc/{saved['pid']}/cmdline").read_bytes().decode().rstrip('\0').split('\0')
    prior=args.prerequisite
    validate_prerequisite(json.loads((prior/'campaign.json').read_text()),
        json.loads((prior/'restored-service.json').read_text()),current,saved['original_argv'],
        json.loads((prior/'original-service.json').read_text())['original_argv'])
    contract=verify_prompt_count_contract(root/plan['base_case']['model'])
    assert contract['revision']==plan['identity']['model_revision'] and contract['model']==plan['identity']['model']
    manifest=json.loads((args.overlay/'manifest.json').read_text())
    for name,digest in manifest['files'].items():
        assert hashlib.sha256((args.overlay/name).read_bytes()).hexdigest()==digest
    assert manifest['native_scheduler_sha256']==plan['identity']['native_scheduler_sha256']
    out=root/'qoserve-validation/raw'/args.run_id;out.mkdir(parents=True,exist_ok=False)
    previous.DIRECTORY=out
    (out/'plan.json').write_bytes(args.plan.read_bytes())
    (out/'original-service.json').write_text(json.dumps(saved,indent=2)+'\n')
    (out/'overlay-manifest.json').write_bytes((args.overlay/'manifest.json').read_bytes())
    if args.repair_definition:
        (out/'repair-definition.json').write_bytes(args.repair_definition.read_bytes())
        (out/'basis-fit.json').write_bytes((root/definition['basis_fit_path']).read_bytes())
    code=Path(__file__).resolve().parents[1]
    metadata=dict(status='running',controller_pid=os.getpid(),started_unix=time.time(),cases=[],restored=False,
        harness_sha256=snapshot_harness(code,out/'harness',
            [str(p.relative_to(code)) for d in ('inferx','scripts') for p in sorted((code/d).glob('*.py'))]))
    def flush():
        temp=out/'campaign.tmp';temp.write_text(json.dumps(metadata,indent=2)+'\n');temp.replace(out/'campaign.json')
    def measure():
        previous.gpu_free()
        if args.stage=='budgets':
            from scripts.calibrate_qoserve_budgets import run_budget_calibration
            def progress(entry):
                if entry is not None:metadata['cases'].append(entry)
                flush()
            return run_budget_calibration(plan,root,out,progress,args.overlay)
        command=['taskset','-c','28-35',str(root/plan['base_case']['binary']),'-u','-m',
            'scripts.calibrate_qoserve_model','--plan',str(out/'plan.json'),'--output',str(out/'profile-result.json')]
        if args.repair_definition:command+=['--repair-definition',str(out/'repair-definition.json')]
        env={**os.environ,**plan['base_case']['environment'],'PYTHONPATH':str(args.overlay)+':'+str(code),
            'CUDA_VISIBLE_DEVICES':plan['identity']['gpu_uuid'],'HF_ENDPOINT':'https://hf-mirror.com',
            'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TMPDIR':str(root/'tmp'),
            'INFERX_VLLM_RUNTIME':str(out/'runtime.json'),'INFERX_QOSERVE_EVIDENCE':str(out/'model-observer.json')}
        entry=dict(case='independent-completed-step-profile',state='starting',command=command)
        metadata['cases'].append(entry);flush()
        with (out/'worker.log').open('x') as log:
            outcome=run_owned_command(command,env,log,timeout=10800)
        if outcome!=0:
            entry.update(state='failed',exit_code=outcome);flush()
            raise RuntimeError('QoServe model profiling failed; preserve raw, no efficacy score')
        result=json.loads((out/'profile-result.json').read_text())
        assert result['status']=='complete' and result['identity']==plan['identity'] and len(result['rollouts'])==expected_rollouts
        entry.update(state='collected',result_sha256=hashlib.sha256((out/'profile-result.json').read_bytes()).hexdigest());flush()
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
