#!/usr/bin/env python3
"""Run sequential, exact-length PILOT experiments using the untouched official runner.

Usage: python scripts/run_ninfer_sweep.py --config pilot.json --output-dir NEW_DIR
The configured command must directly own the engine; no existing server is reused.
Dependencies must already exist in the official runner's isolated perf environment.
"""
from __future__ import annotations

import argparse
import contextlib
import errno
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

DEFAULT_RUNNER = Path(__file__).resolve().parents[1] / "submission/organizer/runner.py"


class SweepCancelled(BaseException):
    pass


def write_json(path: Path, data: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_runner(path: Path):
    spec = importlib.util.spec_from_file_location("official_pilot_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_config(config: dict, case_names: list[str] | None) -> list[dict]:
    for field in ("base_url", "model", "tokenizer", "env_dir", "seed", "request_timeout", "requests", "warmup", "cases"):
        if field not in config:
            raise ValueError(f"missing config field: {field}")
    api = urllib.parse.urlsplit(config["base_url"])
    if (api.scheme != "http" or api.hostname not in {"localhost", "127.0.0.1", "::1"} or
            api.path != "/v1" or api.username or api.password or api.query or api.fragment):
        raise ValueError("base_url must be a local HTTP API ending exactly /v1")
    _ = api.port  # Reject invalid port syntax before creating any subprocess.
    for field in ("model", "tokenizer", "env_dir"):
        if not isinstance(config[field], str) or not config[field].strip():
            raise ValueError(f"{field} must be a nonempty string")
    if config["tokenizer"] == "builtin":
        raise ValueError("PILOT requires a model tokenizer and exact token-ID inputs, not builtin")
    if type(config["seed"]) is not int:
        raise ValueError("seed must be an explicit integer")
    for field, minimum in (("requests", 1), ("warmup", 0)):
        if type(config[field]) is not int or config[field] < minimum:
            raise ValueError(f"{field} must be an integer >= {minimum}")
    for field, default in (("request_timeout", 600), ("startup_timeout", 300), ("cleanup_timeout", 10), ("port_release_timeout", 65)):
        value = config.get(field, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{field} must be a positive finite number")
    if not isinstance(config["cases"], list) or not config["cases"]:
        raise ValueError("cases must be a nonempty list")
    seen = set()
    for case in config["cases"]:
        name = case.get("name", "")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or name in seen:
            raise ValueError("case names must be unique safe directory names")
        seen.add(name)
        command, tiers = case.get("command"), case.get("concurrencies")
        if not isinstance(command, list) or not command or not all(isinstance(arg, str) and arg and "\x00" not in arg for arg in command):
            raise ValueError(f"{name}: command must be a nonempty argv list")
        if not isinstance(tiers, list) or not tiers or any(type(c) is not int or c < 1 for c in tiers) or len(set(tiers)) != len(tiers):
            raise ValueError(f"{name}: concurrencies must be unique positive integers")
    if case_names and set(case_names) - seen:
        raise ValueError(f"unknown case filter: {sorted(set(case_names) - seen)}")
    return [case for case in config["cases"] if not case_names or case["name"] in case_names]


def require_unused_port(base_url: str) -> None:
    api = urllib.parse.urlsplit(base_url)
    for family, socktype, protocol, _, address in socket.getaddrinfo(api.hostname, api.port or 80, type=socket.SOCK_STREAM):
        with socket.socket(family, socktype, protocol) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(address)
            except OSError as error:
                raise RuntimeError(f"destination port {api.port or 80} is occupied or unavailable: {error}") from error



def preflight_port(base_url: str, record: dict) -> None:
    """Wait only for nonlistening EADDRINUSE after this run's owned cleanup."""
    started = time.monotonic()
    api = urllib.parse.urlsplit(base_url)
    deadline = started + record["timeout_seconds"]
    record.update(attempts=0, waited_seconds=0)
    try:
        while True:
            record["attempts"] += 1
            try:
                require_unused_port(base_url)
                record["status"] = "released" if record["attempts"] > 1 else "unused"
                return
            except RuntimeError as error:
                record.setdefault("first_error", str(error))
                record["last_error"] = str(error)
                if not record["previous_owned_server"] or getattr(error.__cause__, "errno", None) != errno.EADDRINUSE:
                    raise
                # A live listener must be refused immediately, even after our previous engine.
                for family, socktype, protocol, _, address in socket.getaddrinfo(api.hostname, api.port or 80, type=socket.SOCK_STREAM):
                    with socket.socket(family, socktype, protocol) as check:
                        check.settimeout(0.2)
                        result = check.connect_ex(address)
                    if result == 0:
                        raise RuntimeError(f"destination port {api.port or 80} has a live listener; refusing to wait or launch") from error
                    if result != errno.ECONNREFUSED:
                        raise RuntimeError(f"cannot establish absence of a listener on port {api.port or 80}: connect errno {result}") from error
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(f"port release timeout after {record['timeout_seconds']}s without a listener: {error}") from error
                if record["attempts"] == 1:
                    print(f"PILOT: waiting up to {record['timeout_seconds']}s for owned-server port release; {error}", flush=True)
                record["status"] = "waiting"
                time.sleep(min(0.1, remaining))
    except (SweepCancelled, KeyboardInterrupt) as error:
        record.update(status="cancelled", error=str(error) or "interrupted")
        raise
    except Exception as error:
        record.update(status="failed", error=str(error))
        raise
    finally:
        record["waited_seconds"] = time.monotonic() - started


def start_ticks(pid: int) -> int:
    # Linux /proc field 22, accounting for spaces/parentheses in the comm field.
    return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])


def wait_ready(child: subprocess.Popen, base_url: str, timeout: float) -> None:
    api = urllib.parse.urlsplit(base_url)
    health = urllib.parse.urlunsplit((api.scheme, api.netloc, "/health", "", ""))
    deadline = time.monotonic() + timeout
    last_error = "no health response"
    while time.monotonic() < deadline:
        if child.poll() is not None:
            raise RuntimeError(f"server exited before readiness (exit {child.returncode})")
        try:
            with urllib.request.urlopen(health, timeout=min(1, max(0.01, deadline - time.monotonic()))) as response:
                if response.status == 200:
                    if child.poll() is not None:
                        raise RuntimeError("server exited during readiness")
                    return
        except (OSError, urllib.error.URLError) as error:
            last_error = str(error)
        time.sleep(min(0.1, max(0, deadline - time.monotonic())))
    raise RuntimeError(f"server readiness timeout after {timeout}s: {last_error}")


def stop_server(child: subprocess.Popen, timeout: float) -> dict:
    if child.poll() is None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=timeout)
    # The engine should be the direct child, but also remove its owned workers.
    with contextlib.suppress(ProcessLookupError):
        os.killpg(child.pid, signal.SIGKILL)
    return {"reaped": child.poll() is not None, "exit_code": child.returncode}


def parse_tier(runner, dest: Path, requests: int, concurrency: int, warmup: int) -> dict:
    try:
        parsed = runner.parse_performance(dest, requests, concurrency, warmup_expected=warmup, strict_isl=True)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        parsed = {"status": "异常", "issues": [f"native parse failed: {error}"], "concurrency": concurrency,
                  "artifacts": str(dest)}
    parsed["native_status"] = parsed.pop("status")
    # Supplement the official result with observed failure counts, never guessed successes.
    try:
        records = [r for r in runner.read_jsonl(dest / "profile_export.jsonl")
                   if r.get("metadata", {}).get("benchmark_phase") == "profiling"]
        parsed.update(failed_input_length_records=sum(r.get("metrics", {}).get("input_sequence_length", {}).get("value") != 8192 for r in records),
                      failed_output_length_records=sum(r.get("metrics", {}).get("output_sequence_length", {}).get("value") != 512 for r in records),
                      missing_profiling_records=max(0, requests - len(records)))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        parsed.update(failed_input_length_records=None, failed_output_length_records=None, missing_profiling_records=None)
    return parsed


def run_sweep(config: dict, output_dir: Path, runner_path: Path = DEFAULT_RUNNER,
              case_names: list[str] | None = None) -> dict:
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing existing output directory: {output_dir}")
    cases = validate_config(config, case_names)
    runner_path = Path(runner_path).expanduser().resolve()
    runner = load_runner(runner_path)
    args = argparse.Namespace(**{**config, "env_dir": Path(config["env_dir"]).expanduser().resolve(),
                                 "perf_requests": config["requests"], "tokenizer_revision": config.get("tokenizer_revision")})
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "config.json", config)
    (output_dir / "config.json").chmod(0o444)
    manifest = {"label": "PILOT", "status": "running", "started_at": runner.utc_now(),
                "run_id": uuid.uuid4().hex, "runner": str(runner_path),
                "runner_sha256": hashlib.sha256(runner_path.read_bytes()).hexdigest(),
                "config_sha256": hashlib.sha256((output_dir / "config.json").read_bytes()).hexdigest(),
                "selected_cases": [case["name"] for case in cases], "cases": [], "any_slo_met": False,
                "slo": {"ttft_p95_max_ms": 10000, "tpot_p95_max_ms": 40},
                "note": "PILOT only; no official score. Probe and warmup excluded; cache isolation not verified by telemetry."}
    write_json(output_dir / "manifest.json", manifest)
    started = time.monotonic()
    previous_handlers = {}
    stopping_owned_server = False
    deferred_cancellation = None

    def cancel(signum, _frame):
        nonlocal deferred_cancellation
        # A second interrupt must not break owned-child cleanup in runner.execute.
        for number in (signal.SIGINT, signal.SIGTERM):
            signal.signal(number, signal.SIG_IGN)
        cancellation = SweepCancelled(f"received {signal.Signals(signum).name}")
        if stopping_owned_server:
            deferred_cancellation = cancellation
            return
        raise cancellation

    try:
        for number in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[number] = signal.signal(number, cancel)
        environment = runner.child_environment(args.env_dir / "nltk_data")
        dataset = output_dir / "dataset.jsonl"
        dataset_command = [str(runner.tool_python(args.env_dir, "perf")), str(runner_path), "_dataset",
                           args.tokenizer, args.tokenizer_revision or "", str(dataset),
                           str(args.perf_requests + args.warmup + 1), str(args.seed)]
        write_json(output_dir / "dataset-command.json", dataset_command)
        runner.execute(dataset_command, output_dir / "dataset.log", env=environment)
        manifest["dataset"] = {"path": str(dataset), "metadata": str(dataset.with_suffix(".meta.json")),
                               "sha256": hashlib.sha256(dataset.read_bytes()).hexdigest()}
        write_json(output_dir / "manifest.json", manifest)
        previous_owned_server = None
        for config_case in cases:
            folder = output_dir / config_case["name"]
            folder.mkdir()
            case_started = time.monotonic()
            case = {"label": "PILOT", "name": config_case["name"], "command": config_case["command"],
                    "concurrencies": config_case["concurrencies"], "started_at": runner.utc_now(),
                    "status": "running", "tiers": [], "server_log": str(folder / "server.log"),
                    "cleanup": {"reaped": True, "note": "no server started"}}
            manifest["cases"].append(case)
            write_json(output_dir / "manifest.json", manifest)
            child = None
            try:
                case["port_preflight"] = {"status": "checking", "started_at": runner.utc_now(),
                    "timeout_seconds": config.get("port_release_timeout", 65) if previous_owned_server else 0,
                    "previous_owned_server": previous_owned_server}
                write_json(output_dir / "manifest.json", manifest)
                preflight_port(args.base_url, case["port_preflight"])
                with (folder / "server.log").open("w", encoding="utf-8") as log:
                    child = subprocess.Popen(config_case["command"], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                case.update(pid=child.pid, start_ticks=start_ticks(child.pid), cleanup={"reaped": False})
                write_json(output_dir / "manifest.json", manifest)
                wait_ready(child, args.base_url, config.get("startup_timeout", 300))
                case["ready_at"] = runner.utc_now()
                runner.check_model(args.base_url, args.model, "EMPTY")
                probe_salt = f"{manifest['run_id']}:{case['name']}:probe"
                case["probe"] = {"cache_salt": probe_salt, "status": "running", "excluded_from_metrics": True}
                runner.probe_performance(args, dataset, "EMPTY", probe_salt)
                case["probe"]["status"] = "complete"
                for concurrency in config_case["concurrencies"]:
                    dest = folder / f"c{concurrency}"
                    dest.mkdir()
                    command = runner.aiperf_command(args, concurrency, dataset, dest,
                        f"{manifest['run_id']}:{case['name']}:c{concurrency}", "EMPTY")
                    command[command.index("--warmup-request-count") + 1] = str(args.warmup)
                    write_json(dest / "command.json", command)
                    tier_started = time.monotonic()
                    tier = {"label": "PILOT", "status": "running", "concurrency": concurrency,
                            "artifacts": str(dest), "started_at": runner.utc_now(), "meets_slo": False}
                    case["tiers"].append(tier)
                    write_json(output_dir / "manifest.json", manifest)
                    failure = None
                    cancelled = None
                    try:
                        if child.poll() is not None:
                            raise RuntimeError(f"server exited (exit {child.returncode})")
                        runner.execute(command, dest / "console.log", env=environment)
                        if child.poll() is not None:
                            raise RuntimeError(f"server exited (exit {child.returncode})")
                    except (SweepCancelled, KeyboardInterrupt) as error:
                        cancelled = error
                    except Exception as error:
                        failure = str(error)
                    tier.update(parse_tier(runner, dest, args.perf_requests, concurrency, args.warmup))
                    if failure or cancelled is not None:
                        tier["issues"].append(failure or str(cancelled) or "interrupted")
                    tier["status"] = ("cancelled" if cancelled is not None else
                                      "complete" if tier["native_status"] == "完整" and not tier["issues"] else "failed")
                    metrics = tier.get("metrics", {})
                    tier["meets_slo"] = (tier["status"] == "complete" and
                        metrics.get("time_to_first_token", {}).get("p95", math.inf) <= 10000 and
                        metrics.get("inter_token_latency", {}).get("p95", math.inf) <= 40)
                    tier.update(finished_at=runner.utc_now(), elapsed_seconds=time.monotonic() - tier_started)
                    write_json(dest / "summary.json", tier)
                    write_json(output_dir / "manifest.json", manifest)
                    if cancelled is not None:
                        raise cancelled
                case["status"] = "complete" if all(t["status"] == "complete" for t in case["tiers"]) else "failed"
            except (SweepCancelled, KeyboardInterrupt) as error:
                case.update(status="cancelled", error=str(error) or "interrupted")
                raise
            except Exception as error:
                case.update(status="failed", error=str(error))
            finally:
                if case.get("probe", {}).get("status") == "running":
                    case["probe"].update(status=case["status"], error=case.get("error", "probe interrupted"))
                # A first cancellation during TERM/KILL/wait must not abandon the owned child.
                stopping_owned_server = True
                try:
                    if child is not None:
                        case["cleanup"] = stop_server(child, config.get("cleanup_timeout", 10))
                except Exception as error:
                    case["cleanup"] = {"reaped": False, "error": str(error)}
                    case["status"] = "failed"
                finally:
                    stopping_owned_server = False
                if child is not None and case["cleanup"]["reaped"]:
                    previous_owned_server = {"case": case["name"], "pid": child.pid,
                        "start_ticks": case.get("start_ticks"), "cleanup": case["cleanup"]}
                if deferred_cancellation is not None:
                    case.update(status="cancelled", error=str(deferred_cancellation))
                for unfinished in case["tiers"]:
                    if unfinished["status"] == "running":
                        unfinished.update(status="cancelled" if case["status"] == "cancelled" else "failed",
                                          meets_slo=False, finished_at=runner.utc_now(),
                                          elapsed_seconds=time.monotonic() - tier_started)
                        unfinished.setdefault("issues", []).append(case.get("error", "tier interrupted"))
                        write_json(Path(unfinished["artifacts"]) / "summary.json", unfinished)
                case.update(finished_at=runner.utc_now(), elapsed_seconds=time.monotonic() - case_started)
                write_json(folder / "summary.json", case)
                write_json(output_dir / "manifest.json", manifest)
                if deferred_cancellation is not None:
                    raise deferred_cancellation
            if not case["cleanup"]["reaped"]:
                raise RuntimeError("owned server cleanup failed; refusing to start another case")
        manifest["status"] = "complete" if all(c["status"] == "complete" for c in manifest["cases"]) else "failed"
    except (SweepCancelled, KeyboardInterrupt) as error:
        manifest.update(status="cancelled", error=str(error) or "interrupted")
    except Exception as error:
        manifest.update(status="failed", error=str(error))
    finally:
        manifest["any_slo_met"] = any(t.get("meets_slo", False) for c in manifest["cases"] for t in c["tiers"])
        manifest.update(finished_at=runner.utc_now(), elapsed_seconds=time.monotonic() - started)
        write_json(output_dir / "manifest.json", manifest)
        for number, handler in previous_handlers.items():
            signal.signal(number, handler)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--case", action="append", dest="case_names", help="repeat to select named cases")
    options = parser.parse_args(argv)
    try:
        config = json.loads(options.config.read_text(encoding="utf-8"))
        result = run_sweep(config, options.output_dir, options.runner, options.case_names)
    except (OSError, ValueError, TypeError) as error:
        parser.exit(2, f"PILOT: {error}\n")
    print(json.dumps({"label": "PILOT", "status": result["status"], "any_slo_met": result["any_slo_met"],
                      "manifest": str(options.output_dir.resolve() / "manifest.json")}, ensure_ascii=False))
    return 0 if result["status"] == "complete" else 130 if result["status"] == "cancelled" else 1


if __name__ == "__main__":
    raise SystemExit(main())
