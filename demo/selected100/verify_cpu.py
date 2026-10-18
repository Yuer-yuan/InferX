#!/usr/bin/env python3
"""Copy this package to an unrelated path and run CPU tests/CLI dry runs only."""
import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0,str(PACKAGE))
import candidate_common as common


def write(path,data):
    path.write_text(json.dumps(data,indent=2,sort_keys=True,ensure_ascii=False)+"\n")


def normalized(argv, flags, positions=(0,)):
    result=list(argv)
    for index in positions: result[index]=f"<path{index}>"
    for flag in flags: result[result.index(flag)+1]=f"<{flag}>"
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository",type=Path,help="Optional original repo for frozen demo and historical-input checks")
    args=parser.parse_args()
    checks=PACKAGE/"checks"
    checks.mkdir(exist_ok=True)
    receipt={"scope":"CPU tests and dry runs only; no exec/HTTP/GPU/build/accuracy evaluation",
             "selection_status":"pending_not_promoted","profiles":{},"dry_run_count":0}
    with tempfile.TemporaryDirectory(prefix="candidate100 CPU relocation ") as temp:
        relocated=Path(temp)/"standalone package with spaces"
        shutil.copytree(PACKAGE,relocated,ignore=shutil.ignore_patterns("__pycache__","checks","MANIFEST.json"))
        unrelated=Path(temp)/"unrelated cwd";unrelated.mkdir()
        deploy=Path(temp)/"deployment with spaces"
        env={k:v for k,v in os.environ.items() if k not in ("OPENAI_API_KEY","PYTHONPATH","PYTHONHOME")}
        env["PYTHONDONTWRITEBYTECODE"]="1"
        env["OPENAI_API_KEY"]="EMPTY"
        tests=subprocess.run([sys.executable,"-B","-m","unittest","discover","-s",str(relocated/"tests"),
                              "-p","test_candidate.py","-v"],cwd=unrelated,env=env,capture_output=True,text=True)
        (checks/"relocated-unittest.log").write_text(tests.stdout+tests.stderr)
        if tests.returncode: raise RuntimeError("Relocated tests failed; see checks/relocated-unittest.log")
        receipt["relocated_unittest_exit_code"]=tests.returncode
        plans={}
        for inventory in ("u3-47","full64"):
            recipe=relocated/f"candidate_recipes/100-k3-{inventory}.json"
            for name in [f"perf-c{c}" for c in common.TIERS]+["quality100"]:
                common_args=["--recipe",str(recipe),"--root",str(deploy),"--dry-run"]
                commands={"service":[sys.executable,"-B",str(relocated/"serve_candidate.py"),*common_args,
                                     "--profile",name,"--log-dir",str(deploy/name/"service")]}
                if name=="quality100":
                    commands["evaluation"]=[sys.executable,"-B",str(relocated/"accuracy_candidate.py"),
                        *common_args,"--output-dir",str(deploy/name/"evaluation")]
                else:
                    commands["evaluation"]=[sys.executable,"-B",str(relocated/"perf_candidate.py"),
                        *common_args,"--concurrency",name.split("-c")[1],
                        "--output-dir",str(deploy/name/"evaluation")]
                for kind,command in commands.items():
                    done=subprocess.run(command,cwd=unrelated,env=env,capture_output=True,text=True)
                    if done.returncode: raise RuntimeError(done.stderr)
                    plan=json.loads(done.stdout)
                    assert plan.get("state")=="dry_run" or plan.get("dry_run") is True
                    assert plan.get("service_verified",False) is False
                    plans[f"{inventory}/{name}/{kind}"]=plan
                    receipt["dry_run_count"]+=1
                assert plans[f"{inventory}/{name}/service"]["profile"]==plans[f"{inventory}/{name}/evaluation"]["profile"]
                receipt["profiles"][name]=plans[f"{inventory}/{name}/service"]["profile"]
        assert receipt["dry_run_count"]==24 and not deploy.exists()
        receipt["relocation_no_output_directories_created"]=True
        actual=common.read_json(PACKAGE/"evidence/C14-original-config.json")["cases"][0]
        service=plans["u3-47/perf-c14/service"]
        assert normalized(actual["command"],["--port","--request-log-jsonl"],(0,1))==normalized(service["argv"],["--port","--request-log-jsonl"],(0,1))
        assert actual["concurrencies"]==[service["profile"]["client_concurrency"]]==[14]
        recorded=common.read_json(PACKAGE/"evidence/C14-original-AIPerf-command.json")
        rendered=plans["u3-47/perf-c14/evaluation"]["AIPerf"]
        relocated_flags=["--url","--tokenizer","--input-file","--artifact-dir"]
        # Original owner used int600; frozen runner argparse serializes float600.0.
        old_timeout=recorded[recorded.index("--request-timeout-seconds")+1]
        new_timeout=rendered[rendered.index("--request-timeout-seconds")+1]
        assert float(old_timeout)==float(new_timeout)==600
        assert normalized(recorded,relocated_flags+["--request-timeout-seconds"])==normalized(rendered,relocated_flags+["--request-timeout-seconds"])
        receipt["AIPerf_timeout_600_vs_600_float_equivalent"]={"actual":old_timeout,"rendered":new_timeout}
        receipt["C14_actual_service_argv_equivalent_after_explicit_path_port_relocation"]=True
        receipt["C14_actual_AIPerf_argv_equivalent_after_path_port_and_timeout_number_normalization"]=True
        write(checks/"portable-dry-run-plans.json",plans)
    common.validate_dataset(PACKAGE/"data/dataset.jsonl")
    assert common.hash_file(PACKAGE/"organizer/runner.py")==common.RUNNER_SHA
    receipt["frozen_runner_dataset_meta_verified"]=True
    if args.repository:
        root=args.repository.resolve()
        before=common.read_json(checks/"protected-demo-before.json")
        after={p:common.hash_file(root/p) for p in before}
        allowed={"demo/selected_common.py","demo/SELECTED.md","demo/tests/test_selected.py"}
        changed={p for p in before if before[p]!=after[p]}
        assert changed==allowed, f"Unexpected protected demo changes: {changed}"
        receipt["protected_demo_files_unchanged"]=len(before)-len(changed)
        receipt["authorized_C4_capacity_fix_files"]={p:{"before":before[p],"after":after[p]} for p in sorted(changed)}
        receipt["final_selected_index_unchanged"]=True
        write(checks/"protected-demo-after.json",after)
        run=root/"results/three-node-optimization/ucloud3/c12-c16-latency-20261018"
        inputs={}
        for suffix in ("","-r2"):
            file=run/("C14-native13-bounded100"+suffix)/"delivery/raw/candidate/c14/inputs.json"
            common.validate_inputs(file)
            inputs[str(file.relative_to(root))]={"file_sha256":common.hash_file(file),"canonical_sha256":common.INPUT_SHA}
        receipt["both_historical_C14_inputs_match_frozen137_contract"]=inputs
        changes=[]
        for original,new in (("selected_common.py","candidate_common.py"),("serve_selected.py","serve_candidate.py"),
                             ("perf_selected.py","perf_candidate.py"),("accuracy_selected.py","accuracy_candidate.py")):
            changes.extend(difflib.unified_diff((root/"demo"/original).read_text().splitlines(True),
                (PACKAGE/new).read_text().splitlines(True),fromfile="demo/"+original,tofile="candidate100/"+new))
        (checks/"entrypoint-diff.patch").write_text("".join(changes))
    write(checks/"CPU-check.json",receipt)
    print(json.dumps(receipt,indent=2))


if __name__=="__main__":
    main()
