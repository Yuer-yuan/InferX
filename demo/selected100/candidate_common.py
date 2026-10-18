"""Isolated pending candidate100 contracts; no network or GPU operations here."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import urllib.parse
import urllib.error
import time

TIERS = (4, 8, 12, 13, 14)
RUNNER_SHA = "d80df390cf6d1d50129796b31fdd7f4d1e95d0043d3b942d47ac582118702fb0"
DATA_SHA = "82f78595fde320cf91293248a94c48770f6d7116abb6dd065cc11d13d0ff9599"
META_SHA = "1698f38c56755f029e9f61c6c915812ea09b0ed829cd71d6647259f60a9cf96f"
INPUT_SHA = "6a22077fc53f81d60176e844adc01dd2dfeb7af19adebe9069f1d9e52969c7d1"
NAMESPACE = "8a90faa7364b4a6aad73eac63d71d369:candidate:c12"
PROBE_SALT = "8a90faa7364b4a6aad73eac63d71d369:candidate:probe"
CLEAN_PREFIXES = ("NINFER_", "CUDA_MPS_", "UCLOUD2_", "INFERX_OUTPUT_CAPTURE_")
INHERITED_ENV_KEYS = frozenset(("PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "LC_CTYPE",
    "TMPDIR", "TMP", "TEMP", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER",
    "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES"))
MODEL_BYTES = 18210531328
PACKAGE = Path(__file__).resolve().parent
CATALOG = PACKAGE / "candidate_recipes"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def hash_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical_sha(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def write_json(path, data):
    # Each output directory is exclusively created by its entrypoint.
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def load_recipe(path):
    path = Path(path).resolve()
    recipe = read_json(path)
    index = read_json(CATALOG / "index.json")["recipes"]
    require(recipe.get("recipe_id") in index, "Unknown frozen candidate recipe")
    sha = hash_file(path)
    require(sha == index[recipe["recipe_id"]], "Recipe differs from its frozen catalog entry")
    require(recipe.get("schema_version") == 1 and recipe.get("status") == "candidate",
            "Unsupported recipe schema/status")
    load_asset_contract(recipe)
    return recipe, sha


def load_asset_contract(recipe):
    identity = recipe["assets"]["contract"]
    require(identity["file"] in ("assets-full64.json", "assets-u3-47.json"),
            "Unknown asset inventory")
    path = CATALOG / identity["file"]
    require(hash_file(path) == identity["sha256"], "Frozen asset inventory SHA mismatch")
    return read_json(path)


def validate_assets(recipe, paths):
    """Check deployment metadata; never read large model/cache contents."""
    contract = load_asset_contract(recipe)
    model = Path(paths["model"])
    require(model.is_file() and model.stat().st_size == contract["model"]["bytes"],
            "Model container size mismatch")
    checked = {}
    for name, spec in contract["caches"].items():
        root = Path(paths["assets"]) / name
        expected_dirs = {str(Path(p).parent) for p in spec["files"]}
        actual_dirs = {p.name for p in root.glob("weights-L*") if p.is_dir()}
        require(actual_dirs == expected_dirs,
                f"Asset layer set mismatch for {name}; use this recipe's original cache inventory")
        actual_files = {str(p.relative_to(root)) for directory in actual_dirs
                        for p in (root / directory).iterdir()
                        if p.name in ("packed.bin", "scales.f32") and p.is_file()}
        require(actual_files == set(spec["files"]), f"Asset serving file set mismatch for {name}")
        for rel, item in spec["files"].items():
            require((root / rel).stat().st_size == item["bytes"],
                    f"Asset file size mismatch: {name}/{rel}")
        checked[name] = {k: spec[k] for k in ("layers", "file_count", "total_bytes")}
    return dict(method="frozen_manifest_and_current_file_metadata", content_sha_rehashed=False,
                contract=recipe["assets"]["contract"], model=contract["model"], caches=checked,
                limitation="Same-size payload corruption is not detected; content SHA is inherited from original receipts.")


def profile(recipe, name):
    require(name in recipe["profiles"], "Profile must be perf-c4/8/12/13/14 or quality100")
    return dict(recipe["profiles"][name])


def clean_environment(inherited, fixed):
    env = {k:v for k,v in inherited.items() if k in INHERITED_ENV_KEYS}
    env.update(fixed)
    return env


def render_service(recipe, recipe_sha, profile_name, *, root, log_dir, host="127.0.0.1",
                   port=18499, binary=None, model=None, assets=None, tokenizer=None,
                   build_lib_dir=None, cuda_lib_dir="/usr/local/cuda-13.2/lib64"):
    require(0 < port < 65536, "Invalid port")
    require(host in ("127.0.0.1", "localhost", "0.0.0.0", "::1", "::"),
            "Use a loopback or wildcard bind; evaluation verifies a same-host service")
    paths = {"root":str(Path(root).resolve()), "log_dir":str(Path(log_dir).resolve())}
    root = Path(paths["root"])
    paths.update(binary=str(Path(binary or root/"artifacts"/recipe["binary_artifact_id"]/"ninfer-serve").resolve()),
                 model=str(Path(model or root/"assets/model/qwen3_8_27b.ninfer").resolve()),
                 assets=str(Path(assets or root/"assets").resolve()),
                 tokenizer=str(Path(tokenizer or root/"assets/tokenizer").resolve()),
                 build_lib_dir=str(Path(build_lib_dir or root/"envs/build/lib").resolve()),
                 cuda_lib_dir=str(Path(cuda_lib_dir).resolve()))
    settings = profile(recipe, profile_name)
    require(settings["kv_capacity"] >= 65536, "KV capacity must be >= max-context (65536)")
    fixed = {k:v.format(assets=paths["assets"]) for k,v in recipe["environment"].items()}
    fixed.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                 LD_LIBRARY_PATH=paths["build_lib_dir"]+":"+paths["cuda_lib_dir"])
    argv = [paths["binary"], paths["model"], "--host", host, "--port", str(port),
            "--model-id", "qwen38-27b", "--max-context", "65536",
            "--kv-capacity", str(settings["kv_capacity"]),
            "--max-concurrency", str(settings["native_concurrency"]),
            "--max-pending-requests", "32", "--pending-timeout-ms", "600000",
            "--prefill-chunk", "2048", "--kv-dtype", "int8", "--presence-penalty", "0",
            "--frequency-penalty", "0", "--seed", "42", "--no-ui", "--spec", "mtp",
            "--draft-tokens", "3", "--lm-head-draft", "--request-log-jsonl",
            str(Path(paths["log_dir"])/"requests.jsonl"), "--no-prefix-reuse", "--no-disk-cache"]
    return dict(schema_version=1, kind="candidate100_service_launch", recipe_id=recipe["recipe_id"],
                recipe_sha256=recipe_sha, binary_sha256=recipe["binary_sha256"],
                source_sha256=recipe["source_sha256"], profile=settings, paths=paths,
                host=host, port=port, argv=argv, environment=fixed,
                asset_contract=recipe["assets"]["contract"],
                version_id=recipe["version_id"], selection_status=recipe["selection_status"],
                quality_status=recipe["quality_status"],
                qualification=recipe.get("qualification", "Frozen100 measured at C4/C8/C12/C13/C14 and original100 quality; portable entrypoint CPU checked only, not a new GPU deployment. See evidence/status.json for exact scope."))


def process_start(pid, proc_root=Path("/proc")):
    stat = (Path(proc_root)/str(pid)/"stat").read_text()
    return int(stat[stat.rfind(")")+2:].split()[19])


def check_listener(pid, port, hostname, proc_root):
    """Do not send requests while this process is loading and another service owns the port."""
    proc = Path(proc_root)/str(pid)
    inodes = set()
    for fd in (proc/"fd").iterdir():
        try:
            target = os.readlink(fd)
        except FileNotFoundError:
            continue
        if target.startswith("socket:[") and target.endswith("]"):
            inodes.add(target[8:-1])
    if hostname == "127.0.0.1":
        table, addresses = "tcp", {"00000000", "0100007F"}
    else:
        table, addresses = "tcp6", {"0"*32, "0"*24+"01000000"}
    for line in (proc/"net"/table).read_text().splitlines()[1:]:
        fields = line.split()
        if len(fields) < 10:
            continue
        address, hex_port = fields[1].split(":")
        if (fields[3] == "0A" and fields[9] in inodes and address in addresses and
                int(hex_port,16) == port):
            return
    raise ValueError("Verified service process does not own the requested listening socket; wait for readiness")


def check_service(launch_path, recipe, recipe_sha, expected_profile, base_url,
                  proc_root=Path("/proc")):
    launch = read_json(launch_path)
    require(launch.get("kind") == "candidate100_service_launch" and launch.get("schema_version") == 1,
            "Not a candidate100 service launch receipt")
    require(launch.get("recipe_sha256") == recipe_sha and launch.get("recipe_id") == recipe["recipe_id"],
            "Service recipe mismatch")
    require(launch.get("binary_sha256") == recipe["binary_sha256"] and
            launch.get("source_sha256") == recipe["source_sha256"], "Service binary/source mismatch")
    require(launch.get("profile") == profile(recipe, expected_profile), "Service native/KV/profile mismatch")
    p = launch["paths"]
    expected = render_service(recipe, recipe_sha, expected_profile, **p,
                              host=launch["host"], port=launch["port"])
    require(launch["argv"] == expected["argv"] and launch["environment"] == expected["environment"],
            "Service argv or environment differs from frozen profile")
    require(launch.get("asset_contract") == expected["asset_contract"], "Service asset inventory mismatch")
    api = urllib.parse.urlsplit(base_url)
    require(api.scheme == "http" and api.hostname in ("127.0.0.1", "::1") and
            api.port == launch["port"] and api.path.rstrip("/") == "/v1" and
            not api.query and not api.fragment and not api.username and not api.password,
            "Evaluation URL must address the same local service port and /v1")
    require(launch.get("state") == "exec_pending", "Dry-run plans are not live service receipts")
    pid, start = launch["pid"], launch["start_ticks"]
    require(type(pid) is int and pid > 0 and type(start) is int, "Invalid process identity")
    proc = Path(proc_root)/str(pid)
    require(process_start(pid, proc_root) == start, "Service PID was reused")
    require((proc/"exe").resolve(strict=True) == Path(p["binary"]).resolve(strict=True),
            "Service executable mismatch or exec has not completed")
    actual_argv = [s.decode() for s in (proc/"cmdline").read_bytes().split(b"\0") if s]
    require(actual_argv == launch["argv"], "Actual service argv mismatch")
    env = dict(s.decode().split("=", 1) for s in (proc/"environ").read_bytes().split(b"\0") if s)
    actual_fixed = {k:v for k,v in env.items() if k.startswith(CLEAN_PREFIXES) or k in
                    ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "LD_LIBRARY_PATH")}
    require(actual_fixed == launch["environment"], "Actual service environment mismatch")
    require(env == launch.get("process_environment") and clean_environment(env, launch["environment"]) == env,
            "Actual process environment is not the recorded complete whitelist")
    require(hash_file(proc/"exe") == recipe["binary_sha256"], "Actual executable SHA mismatch")
    require(launch.get("asset_verification") == validate_assets(recipe, p),
            "Service asset metadata verification mismatch")
    check_listener(pid, launch["port"], api.hostname, proc_root)
    require(process_start(pid, proc_root) == start, "Service identity changed during verification")
    return launch


def wait_model_ready(official, launch_path, recipe, recipe_sha, expected_profile,
                     base_url, expected_launch, key, timeout_seconds=300):
    """Wait for model loading using read-only /models, preserving the frozen runner.

    A listening socket is available before the large-KV engine finishes loading.
    Validate exact process identity before every attempt and after success. Never
    send generation/warmup requests here or retry evaluator requests.
    """
    started = time.monotonic()
    deadline = started + timeout_seconds
    attempts = 0
    while True:
        require(check_service(launch_path, recipe, recipe_sha, expected_profile, base_url) == expected_launch,
                "Service launch changed while waiting for model readiness")
        attempts += 1
        try:
            model = official.check_model(base_url, "qwen38-27b", key)
        except official.EvaluationError as error:
            cause = error.__cause__
            transient = isinstance(cause, (TimeoutError, ConnectionError, urllib.error.URLError))
            if isinstance(cause, urllib.error.HTTPError):
                transient = cause.code in (502, 503, 504)
            if not transient or time.monotonic() >= deadline:
                raise
            print("Waiting for verified service model readiness: " + safe_error(error), flush=True)
            time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
            continue
        require(check_service(launch_path, recipe, recipe_sha, expected_profile, base_url) == expected_launch,
                "Service launch changed during model readiness check")
        return dict(ready=True, attempts=attempts, elapsed_seconds=time.monotonic()-started,
                    model=model, generation_requests_sent=0)


def load_official(path):
    path = Path(path).resolve()
    require(hash_file(path) == RUNNER_SHA, "Official d80 runner SHA mismatch")
    spec = importlib.util.spec_from_file_location("selected_official_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_dataset(dataset):
    dataset = Path(dataset).resolve()
    meta = dataset.with_suffix(".meta.json")
    require(hash_file(dataset) == DATA_SHA, "Frozen dataset SHA mismatch")
    require(hash_file(meta) == META_SHA, "Frozen dataset metadata SHA mismatch")
    return dataset, meta


def validate_inputs(path):
    data = read_json(path)
    require(isinstance(data, dict) and len(data.get("data", [])) == 137,
            "AIPerf inputs must retain all137 original dataset records")
    require(canonical_sha(data) == INPUT_SHA, "Actual input payload/salt mismatch")


def api_key():
    value = os.environ.get("OPENAI_API_KEY", "EMPTY")
    require(bool(value) and "\n" not in value and "\r" not in value, "Invalid API key environment")
    return value


def exclusive_directory(path):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False, mode=0o700)
    return path


def interrupted_signal(*_):
    raise KeyboardInterrupt


def safe_error(error):
    text = str(error)
    secret = os.environ.get("OPENAI_API_KEY", "")
    return text.replace(secret, "<REDACTED>") if secret and secret != "EMPTY" else text
