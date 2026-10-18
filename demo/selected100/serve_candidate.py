#!/usr/bin/env python3
"""Start a pending candidate100 profile; --dry-run performs no service/GPU operation."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

import candidate_common as common


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--root", type=Path, default=Path("/workspace/InferX"))
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18499)
    for name in ("binary", "model", "assets", "tokenizer", "build-lib-dir"):
        parser.add_argument("--"+name, type=Path)
    parser.add_argument("--cuda-lib-dir", default="/usr/local/cuda-13.2/lib64", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    recipe, sha = common.load_recipe(args.recipe)
    kwargs = {k:v for k,v in vars(args).items() if k not in ("recipe", "profile", "dry_run")}
    launch = common.render_service(recipe, sha, args.profile, **kwargs)
    launch["state"] = "dry_run"
    if args.dry_run:
        print(json.dumps(launch, ensure_ascii=False, indent=2))
        return 0
    common.require(not args.log_dir.exists(), "Service output directory already exists")
    paths = launch["paths"]
    common.require(common.hash_file(paths["binary"]) == recipe["binary_sha256"], "Frozen binary SHA mismatch")
    assets = common.validate_assets(recipe, paths)
    for directory in [Path(paths[x]) for x in ("tokenizer", "build_lib_dir", "cuda_lib_dir")]:
        common.require(directory.is_dir(), f"Missing deployed asset/runtime directory: {directory}")
    env = common.clean_environment(os.environ, launch["environment"])
    launch.update(state="exec_pending", pid=os.getpid(), start_ticks=common.process_start(os.getpid()),
                  process_environment=env,
                  asset_verification=assets)
    output = common.exclusive_directory(args.log_dir)
    shutil.copyfile(args.recipe, output/"recipe.json")
    common.write_json(output/"launch.json", launch)
    print(json.dumps(launch, ensure_ascii=False), flush=True)
    try:
        os.execvpe(launch["argv"][0], launch["argv"], env)
    except OSError as error:
        common.write_json(output/"exec-error.json", {"error":str(error), "service_started":False})
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2)
