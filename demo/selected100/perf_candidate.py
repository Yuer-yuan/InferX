#!/usr/bin/env python3
"""Fixed original128 performance evaluation against a verified candidate100 service."""
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
    parser.add_argument("--concurrency", required=True, type=int, choices=common.TIERS)
    parser.add_argument("--root", type=Path, default=Path("/workspace/InferX"))
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--service-launch", type=Path)
    parser.add_argument("--frozen-dataset", type=Path, default=common.PACKAGE/"data/dataset.jsonl")
    parser.add_argument("--base-url", default="http://127.0.0.1:18499/v1")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    recipe, sha = common.load_recipe(args.recipe)
    official = common.load_official(args.runner or common.PACKAGE/"organizer/runner.py")
    dataset, meta = common.validate_dataset(args.frozen_dataset)
    common.require(not args.output_dir.exists(), "Evaluation output directory already exists")
    profile_name = f"perf-c{args.concurrency}"
    launch = None
    if not args.dry_run:
        common.require(args.service_launch is not None, "--service-launch is required for evaluation")
        launch = common.check_service(args.service_launch, recipe, sha, profile_name, args.base_url)
    tokenizer = launch["paths"]["tokenizer"] if launch else str(root/"assets/tokenizer")
    config = official.parse_args(["--mode", "perf", "--base-url", args.base_url,
        "--model", "qwen38-27b", "--tokenizer", tokenizer, "--env-dir", str(root/"eval-envs"),
        "--perf-requests", "128", "--seed", "20261016", "--request-timeout", "600",
        "--output-dir", str(args.output_dir)])
    output = config.output_dir
    local_dataset = output/"dataset.jsonl"
    dest = output/f"c{args.concurrency}"
    key = common.api_key()
    command = official.aiperf_command(config, args.concurrency, local_dataset, dest, common.NAMESPACE, key)
    plan = dict(recipe_id=recipe["recipe_id"], recipe_sha256=sha,
                binary_sha256=recipe["binary_sha256"], source_sha256=recipe["source_sha256"],
                profile=common.profile(recipe, profile_name), runner_sha256=common.RUNNER_SHA,
                asset_contract=recipe["assets"]["contract"], selection_status=recipe["selection_status"],
                dataset_source=str(dataset), dataset_sha256=common.DATA_SHA, metadata_sha256=common.META_SHA,
                performance_namespace=common.NAMESPACE, probe_salt=common.PROBE_SALT,
                expected_actual_inputs_canonical_sha256=common.INPUT_SHA,
                measured=128, warmup=8, probe_count=1, AIPerf=command,
                service_verified=launch is not None, dry_run=args.dry_run)
    if args.dry_run:
        print(official.redact(json.dumps(plan, ensure_ascii=False, indent=2), key))
        return 0
    common.exclusive_directory(output)
    dest.mkdir()
    shutil.copyfile(args.recipe, output/"recipe.json")
    common.write_json(output/"service-launch.json", launch)
    official.write_json(output/"plan.json", plan, key)
    shutil.copyfile(dataset, local_dataset)
    shutil.copyfile(meta, local_dataset.with_suffix(".meta.json"))
    receipt = dict(status="failed", recipe_id=recipe["recipe_id"], recipe_sha256=sha,
                   service_pid=launch["pid"], service_start_ticks=launch["start_ticks"])
    try:
        common.validate_dataset(local_dataset)
        # Recheck immediately before issuing any HTTP, after data preparation.
        common.require(common.check_service(args.service_launch, recipe, sha, profile_name, args.base_url) == launch,
                       "Service launch changed during evaluation")
        readiness = common.wait_model_ready(official, args.service_launch, recipe, sha, profile_name,
                                            args.base_url, launch, key)
        common.write_json(output/"model-readiness.json", readiness)
        official.probe_performance(config, local_dataset, key, common.PROBE_SALT)
        env = official.child_environment(config.env_dir/"nltk_data")
        official.execute(command, dest/"console.log", key, env)
        common.validate_inputs(dest/"inputs.json")
        summary = official.parse_performance(dest, 128, args.concurrency, warmup_expected=8, strict_isl=True)
        official.write_json(output/"summary.json", summary, key)
        common.require(common.check_service(args.service_launch, recipe, sha, profile_name, args.base_url) == launch,
                       "Service launch changed during evaluation")
        common.require(not summary.get("issues"), "Official parser reported integrity issues; raw evidence retained")
        receipt.update(status="complete", actual_inputs_canonical_sha256=common.INPUT_SHA,
                       meets_slo=summary.get("meets_slo"))
        print(official.redact(json.dumps(summary, ensure_ascii=False, indent=2), key))
        return 0
    except BaseException as error:
        receipt["error"] = official.redact(f"{type(error).__name__}: {error}", key)
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
