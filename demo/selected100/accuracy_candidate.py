#!/usr/bin/env python3
"""Pending original100 IFBench profile, client/native13; no accuracy claim."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import sys

import candidate_common as common


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--root", type=Path, default=Path("/workspace/InferX"))
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--service-launch", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:18500/v1")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    recipe, sha = common.load_recipe(args.recipe)
    root = args.root.resolve()
    official = common.load_official(args.runner or common.PACKAGE/"organizer/runner.py")
    common.require(not args.output_dir.exists(), "Evaluation output directory already exists")
    launch = None
    if not args.dry_run:
        common.require(args.service_launch is not None, "--service-launch is required for evaluation")
        launch = common.check_service(args.service_launch, recipe, sha, "quality100", args.base_url)
    tokenizer = launch["paths"]["tokenizer"] if launch else str(root/"assets/tokenizer")
    forwarded = ["--mode", "accuracy", "--base-url", args.base_url, "--model", "qwen38-27b",
                 "--tokenizer", tokenizer, "--env-dir", str(root/"eval-envs"),
                 "--eval-batch-size", str(common.profile(recipe, "quality100")["client_concurrency"]),
                 "--output-dir", str(args.output_dir.resolve()/"evaluation")]
    config = official.parse_args(forwarded)
    task = official.accuracy_config(config, config.output_dir/"accuracy")
    common.require(task["limit"] == 100 and task["generation_config"] == {"max_tokens":32768},
                   "Original accuracy contract changed")
    plan = dict(recipe_id=recipe["recipe_id"], recipe_sha256=sha,
                binary_sha256=recipe["binary_sha256"], source_sha256=recipe["source_sha256"],
                profile=common.profile(recipe, "quality100"), task_config=task, runner_sha256=common.RUNNER_SHA,
                asset_contract=recipe["assets"]["contract"], selection_status=recipe["selection_status"],
                quality_status=recipe["quality_status"],
                service_verified=launch is not None, dry_run=args.dry_run)
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    output = common.exclusive_directory(args.output_dir)
    shutil.copyfile(args.recipe, output/"recipe.json")
    common.write_json(output/"service-launch.json", launch)
    common.write_json(output/"plan.json", plan)
    receipt = dict(status="failed", recipe_id=recipe["recipe_id"], recipe_sha256=sha,
                   service_pid=launch["pid"], service_start_ticks=launch["start_ticks"])
    try:
        common.require(common.check_service(args.service_launch, recipe, sha, "quality100", args.base_url) == launch,
                       "Service launch changed during evaluation")
        readiness = common.wait_model_ready(official, args.service_launch, recipe, sha, "quality100",
                                            args.base_url, launch, common.api_key())
        common.write_json(output/"model-readiness.json", readiness)
        os.environ["PATH"] = str(root/"tools")+":"+str(root/"envs/build/bin")+":"+os.environ.get("PATH", "")
        os.environ.setdefault("MODELSCOPE_CACHE", str(root/"cache/modelscope"))
        os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
        code = official.main(forwarded)
        common.require(common.check_service(args.service_launch, recipe, sha, "quality100", args.base_url) == launch,
                       "Service launch changed during evaluation")
        receipt.update(status="complete" if code == 0 else "failed", exit_code=code)
        return code
    except BaseException as error:
        receipt["error"] = common.safe_error(f"{type(error).__name__}: {error}")
        raise
    finally:
        common.write_json(output/"evaluation-identity.json", receipt)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, common.interrupted_signal)
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Evaluation interrupted; partial evidence retained.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"Error: {common.safe_error(error)}", file=sys.stderr)
        raise SystemExit(2)
