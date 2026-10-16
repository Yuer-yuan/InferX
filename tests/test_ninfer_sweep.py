"""Process-boundary contracts for the PILOT driver (no GPU or downloads)."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import threading

import pytest

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "scripts/run_ninfer_sweep.py"
RUNNER = ROOT / "submission/organizer/runner.py"


def load_driver():
    assert DRIVER.is_file(), "the PILOT sweep driver has not been implemented"
    spec = importlib.util.spec_from_file_location("ninfer_sweep", DRIVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_for(path, timeout=10):
    deadline = time.monotonic() + timeout
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert path.exists(), f"timed out waiting for {path}"


def assert_gone(pid):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        # A reparented, killed worker can briefly remain a zombie on CI.
        if Path(f"/proc/{pid}/stat").read_text().split(")", 1)[1].split()[0] == "Z":
            return
        time.sleep(0.02)
    pytest.fail(f"owned process {pid} survived cleanup")


@pytest.fixture
def harness(tmp_path, monkeypatch):
    """Only replace unavailable tokenizer/AIPerf; keep runner and processes real."""
    stub = tmp_path / "transformers.py"
    stub.write_text('''
class AutoTokenizer:
    all_special_ids = [0]
    init_kwargs = {"_commit_hash": "cpu-fixture"}
    @classmethod
    def from_pretrained(cls, name, revision=None, trust_remote_code=False):
        assert name == "fixture-tokenizer" and not trust_remote_code
        return cls()
    def get_vocab(self):
        return {"special": 0, "one": 1, "two": 2, "three": 3}
''')
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setenv("SWEEP_FAKE_MODE", "good")
    monkeypatch.setenv("SWEEP_FAKE_RECORD", str(tmp_path / "aiperf.pid"))
    perf_bin = tmp_path / "env/aiperf-0.13.0/bin"
    perf_bin.mkdir(parents=True)
    (perf_bin / "python").symlink_to(sys.executable)
    aiperf = perf_bin / "aiperf"
    aiperf.write_text(f"#!{sys.executable}\n" + '''
import json, os, pathlib, subprocess, sys, time
args = sys.argv[1:]
def value(flag): return args[args.index(flag) + 1]
mode = os.environ["SWEEP_FAKE_MODE"]
pathlib.Path(os.environ["SWEEP_FAKE_RECORD"]).with_name("benchmark-started").touch()
if mode == "hold":
    deadline = time.monotonic() + 10
    gate = pathlib.Path(os.environ["SWEEP_FAKE_RECORD"]).with_name("release-benchmark")
    while not gate.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert gate.exists(), "benchmark gate timed out"
pathlib.Path(os.environ["SWEEP_FAKE_RECORD"]).write_text(str(os.getpid()))
if mode == "hang":
    subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    time.sleep(60)
dest = pathlib.Path(value("--artifact-dir"))
requests = int(value("--request-count"))
warmup = int(value("--warmup-request-count"))
dataset = [json.loads(line) for line in pathlib.Path(value("--input-file")).read_text().splitlines()]
assert len(dataset) == requests + warmup + 1
assert all(len(r["extra"]["prompt"]) == 8192 and r["output_length"] == 512 for r in dataset)
extra = json.loads(value("--extra-inputs"))
assert extra["ignore_eos"] is True and extra["temperature"] == 0 and extra["cache_salt"]
def row(phase, ttft, tpot, input_len=8192):
    return {"metadata": {"benchmark_phase": phase}, "metrics": {
        "time_to_first_token": {"value": ttft, "unit": "ms"},
        "inter_token_latency": {"value": tpot, "unit": "ms"},
        "request_latency": {"value": ttft + tpot * 511, "unit": "ms"},
        "input_sequence_length": {"value": input_len, "unit": "tokens"},
        "output_sequence_length": {"value": 512, "unit": "tokens"}}}
rows = [row("warmup", 50000, 1000) for _ in range(warmup)]
rows += [row("profiling", 1000, 20, 8000 if mode == "length" else 8192)
         for _ in range(requests - (mode == "errors"))]
(dest / "profile_export.jsonl").write_text("".join(json.dumps(r) + "\\n" for r in rows))
native = {"is_complete": True, "was_cancelled": False,
          "request_count": {"avg": requests - (mode == "errors")},
          "error_request_count": {"avg": int(mode == "errors")},
          "benchmark_duration": {"avg": 2}, "request_throughput": {"avg": 1},
          "output_token_throughput": {"avg": 512}}
for tag, val in [("time_to_first_token", 1000), ("inter_token_latency", 20), ("request_latency", 11220)]:
    native[tag] = {"p50": val, "p95": val, "p99": val}
(dest / "profile_export_aiperf.json").write_text(json.dumps(["broken export"] if mode == "malformed" else native))
print("native fixture export retained", flush=True)
sys.exit(7 if mode == "exit" else 0)
''')
    aiperf.chmod(0o755)
    server = tmp_path / "server.py"
    server.write_text('''
from http.server import BaseHTTPRequestHandler, HTTPServer
import json, os, pathlib, signal, sys
port, root, mode = int(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
root.joinpath("server.pid").write_text(str(os.getpid()))
if mode == "exit":
    raise SystemExit(3)
if mode == "slow-stop":
    def slow_stop(signum, frame):
        root.joinpath("server-term").write_text("TERM received; still running")
    signal.signal(signal.SIGTERM, slow_stop)
class Server(HTTPServer):
    allow_reuse_address = mode != "no-reuse"
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1" if mode == "no-reuse" else "HTTP/1.0"
    def do_GET(self):
        body = json.dumps({"data":[{"id":"model"}]}).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers()
        self.wfile.write(body); self.wfile.flush()
    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        root.joinpath("probe.json").write_text(json.dumps(payload))
        assert len(payload["prompt"]) == 8192 and payload["max_tokens"] == 1
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
        usage = {"prompt_tokens": 8000 if mode == "probe-fail" else 8192, "completion_tokens": 1}
        self.wfile.write(("data: " + json.dumps({"usage": usage}) + "\\n\\ndata: [DONE]\\n\\n").encode())
Server(("127.0.0.1",port), Handler).serve_forever()
''')
    port = free_port()
    config = {"base_url": f"http://127.0.0.1:{port}/v1", "model": "model",
              "tokenizer": "fixture-tokenizer", "tokenizer_revision": "fixture-revision",
              "env_dir": str(tmp_path / "env"), "seed": 137, "requests": 2, "warmup": 1,
              "request_timeout": 3, "startup_timeout": 2, "cleanup_timeout": 1,
              "cases": [{"name": "mtp0", "command": [sys.executable, str(server), str(port), str(tmp_path), "good"],
                         "concurrencies": [1, 2]}]}
    return config


def test_existing_destination_is_never_overwritten(tmp_path):
    driver = load_driver()
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "do-not-touch"
    marker.write_text("original")
    with pytest.raises(FileExistsError):
        driver.run_sweep({}, output, RUNNER)
    assert marker.read_text() == "original"
    assert list(output.iterdir()) == [marker]


def test_occupied_port_never_launches_or_stops_foreign_server(tmp_path, harness):
    driver = load_driver()
    port = int(harness["base_url"].split(":")[2].split("/")[0])
    harness["port_release_timeout"] = 2
    with socket.socket() as foreign:
        foreign.bind(("127.0.0.1", port)); foreign.listen()
        result = driver.run_sweep(harness, tmp_path / "run", RUNNER)
        assert result["status"] == "failed"
        assert "port" in result["cases"][0]["error"].lower()
        assert result["cases"][0]["elapsed_seconds"] < 0.5
        assert result["cases"][0]["port_preflight"]["previous_owned_server"] is None
        assert not (tmp_path / "server.pid").exists()
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            pass
    assert not result["any_slo_met"]


def test_success_uses_native_artifacts_and_excludes_warmup(tmp_path, harness):
    original = RUNNER.read_bytes()
    result = load_driver().run_sweep(harness, tmp_path / "run", RUNNER)
    assert result["label"] == "PILOT" and result["status"] == "complete"
    assert result["any_slo_met"] is True
    assert len(result["cases"]) == 1
    case = result["cases"][0]
    assert case["cleanup"]["reaped"] is True and case["start_ticks"] > 0
    assert case["command"] == harness["cases"][0]["command"]
    assert_gone(case["pid"])
    salts = []
    for tier in case["tiers"]:
        assert tier["label"] == "PILOT" and tier["status"] == "complete"
        assert tier["meets_slo"] is True
        assert tier["metrics"]["time_to_first_token"]["p95"] == 1000
        assert tier["metrics"]["inter_token_latency"]["p95"] == 20
        assert tier["records"] == 2 and tier["error_requests"] == 0
        assert tier["failed_input_length_records"] == 0
        artifact = Path(tier["artifacts"])
        assert (artifact / "profile_export.jsonl").is_file()
        assert (artifact / "profile_export_aiperf.json").is_file()
        assert "native fixture export retained" in (artifact / "console.log").read_text()
        command = json.loads((artifact / "command.json").read_text())
        assert command[command.index("--warmup-request-count") + 1] == "1"
        salts.append(json.loads(command[command.index("--extra-inputs") + 1])["cache_salt"])
    assert len(set(salts)) == 2
    dataset = [json.loads(line) for line in (tmp_path / "run/dataset.jsonl").read_text().splitlines()]
    assert len(dataset) == 4
    probe = json.loads((tmp_path / "probe.json").read_text())
    assert probe["prompt"] == dataset[-1]["extra"]["prompt"]
    assert probe["cache_salt"] not in salts
    assert json.loads((tmp_path / "run/dataset.meta.json").read_text())["seed"] == 137
    assert json.loads((tmp_path / "run/config.json").read_text()) == harness
    assert (tmp_path / "run/config.json").stat().st_mode & 0o222 == 0
    assert RUNNER.read_bytes() == original


@pytest.mark.parametrize("mode", ["errors", "length", "exit", "malformed"])
def test_native_failure_is_retained_without_slo_success(tmp_path, harness, monkeypatch, mode):
    monkeypatch.setenv("SWEEP_FAKE_MODE", mode)
    result = load_driver().run_sweep(harness, tmp_path / "run", RUNNER)
    assert result["status"] == "failed" and result["any_slo_met"] is False
    case = result["cases"][0]
    assert_gone(case["pid"])
    assert case["cleanup"]["reaped"] is True
    for tier in case["tiers"]:
        assert tier["status"] == "failed" and tier["meets_slo"] is False
        assert tier["issues"]
        assert (Path(tier["artifacts"]) / "profile_export_aiperf.json").is_file()
        if mode == "errors":
            assert tier["error_requests"] == 1 and tier["missing_profiling_records"] == 1
        if mode == "length":
            assert tier["failed_input_length_records"] == 2


@pytest.mark.parametrize("mode", ["exit", "probe-fail"])
def test_server_start_or_probe_failure_is_reaped_and_next_case_runs(tmp_path, harness, mode):
    harness["cases"][0]["command"][-1] = mode
    good = {**harness["cases"][0], "name": "mtp4", "command": harness["cases"][0]["command"][:-1] + ["good"]}
    harness["cases"].append(good)
    result = load_driver().run_sweep(harness, tmp_path / "run", RUNNER)
    first, second = result["cases"]
    assert first["status"] == "failed" and first["error"]
    assert first["tiers"] == [] and first["cleanup"]["reaped"] is True
    if mode == "probe-fail":
        assert first["probe"]["status"] == "failed"
        assert first["probe"]["error"]
    assert second["status"] == "complete" and second["tiers"][0]["meets_slo"]
    assert result["status"] == "failed"
    assert_gone(first["pid"]); assert_gone(second["pid"])


def test_sigterm_cancels_owned_benchmark_and_server_and_persists_status(tmp_path, harness, monkeypatch):
    load_driver()
    monkeypatch.setenv("SWEEP_FAKE_MODE", "hang")
    config = tmp_path / "config.json"
    config.write_text(json.dumps(harness))
    output = tmp_path / "run"
    with (tmp_path / "driver.log").open("w") as log:
        child = subprocess.Popen([sys.executable, str(DRIVER), "--config", str(config),
                                  "--output-dir", str(output), "--runner", str(RUNNER)], stdout=log, stderr=log)
        try:
            wait_for(tmp_path / "aiperf.pid")
            child.send_signal(signal.SIGTERM)
            assert child.wait(timeout=15) != 0
        finally:
            if child.poll() is None:
                child.kill(); child.wait()
    result = json.loads((output / "manifest.json").read_text())
    assert result["status"] == "cancelled" and result["any_slo_met"] is False
    case = result["cases"][0]
    assert case["status"] == "cancelled" and case["cleanup"]["reaped"] is True
    assert case["tiers"][0]["status"] == "cancelled"
    assert_gone(case["pid"])
    assert_gone(int((tmp_path / "aiperf.pid").read_text()))


def test_cancel_during_native_parsing_finalizes_started_tier(tmp_path, harness, monkeypatch):
    driver = load_driver()
    original_loader = driver.load_runner

    def interrupting_runner(path):
        runner = original_loader(path)

        def interrupted_parse(*args, **kwargs):
            raise KeyboardInterrupt()

        # The external parser boundary lets cancellation land after execute has reaped AIPerf.
        runner.parse_performance = interrupted_parse
        return runner

    monkeypatch.setattr(driver, "load_runner", interrupting_runner)
    result = driver.run_sweep(harness, tmp_path / "run", RUNNER)
    assert result["status"] == "cancelled" and not result["any_slo_met"]
    case = result["cases"][0]
    assert case["cleanup"]["reaped"] is True
    assert_gone(case["pid"])
    tier = case["tiers"][0]
    assert tier["status"] == "cancelled" and not tier["meets_slo"]
    summary = json.loads((Path(tier["artifacts"]) / "summary.json").read_text())
    assert summary["status"] == "cancelled"


@pytest.mark.parametrize("cancel_signal", [signal.SIGINT, signal.SIGTERM])
def test_signal_during_server_cleanup_finishes_reaping_and_records_cancellation(tmp_path, harness, cancel_signal):
    load_driver()
    harness["cleanup_timeout"] = 0.8
    harness["cases"][0]["command"][-1] = "slow-stop"
    harness["cases"].append({**harness["cases"][0], "name": "must-not-start"})
    config = tmp_path / "config.json"
    config.write_text(json.dumps(harness))
    output = tmp_path / "run"
    server_pid = None
    with (tmp_path / "driver.log").open("w") as log:
        child = subprocess.Popen([sys.executable, str(DRIVER), "--config", str(config),
                                  "--output-dir", str(output), "--runner", str(RUNNER)], stdout=log, stderr=log)
        try:
            # TERM is sent by stop_server only after the benchmark has completed.
            wait_for(tmp_path / "server-term")
            server_pid = int((tmp_path / "server.pid").read_text())
            child.send_signal(cancel_signal)
            assert child.wait(timeout=10) == 130
            assert_gone(server_pid)
            result = json.loads((output / "manifest.json").read_text())
            assert result["status"] == "cancelled"
            assert len(result["cases"]) == 1
            case = result["cases"][0]
            assert case["status"] == "cancelled"
            assert case["cleanup"] == {"reaped": True, "exit_code": -signal.SIGKILL}
            assert signal.Signals(cancel_signal).name in case["error"]
            summary = json.loads((output / "mtp0/summary.json").read_text())
            assert summary["status"] == "cancelled" and summary["cleanup"]["reaped"] is True
        finally:
            if child.poll() is None:
                child.kill(); child.wait()
            # The pre-fix RED run must not itself leak the deliberately stubborn fixture.
            if server_pid is None and (tmp_path / "server.pid").exists():
                server_pid = int((tmp_path / "server.pid").read_text())
            if server_pid is not None:
                try:
                    os.killpg(server_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


def test_owned_no_reuse_http_shutdown_waits_boundedly_for_time_wait(tmp_path, harness, monkeypatch):
    load_driver()
    harness["port_release_timeout"] = 0.3
    harness["cases"][0]["concurrencies"] = [1]
    harness["cases"][0]["command"][-1] = "no-reuse"
    harness["cases"].append({**harness["cases"][0], "name": "next"})
    monkeypatch.setenv("SWEEP_FAKE_MODE", "hold")
    config = tmp_path / "config.json"
    config.write_text(json.dumps(harness))
    output = tmp_path / "run"
    port = int(harness["base_url"].split(":")[2].split("/")[0])
    with (tmp_path / "driver.log").open("w") as log:
        child = subprocess.Popen([sys.executable, str(DRIVER), "--config", str(config),
                                  "--output-dir", str(output), "--runner", str(RUNNER)], stdout=log, stderr=log)
        try:
            wait_for(tmp_path / "benchmark-started")
            # Keep a client connection open until the owned server actively closes it at shutdown.
            with socket.create_connection(("127.0.0.1", port), timeout=3) as client:
                client.sendall(b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: keep-alive\r\n\r\n")
                response = client.recv(4096)
                assert b"200 OK" in response
                (tmp_path / "release-benchmark").touch()
                while client.recv(4096):
                    pass
            assert child.wait(timeout=5) == 1
        finally:
            if child.poll() is None:
                child.kill(); child.wait()
    states = [line.split()[3] for line in Path("/proc/net/tcp").read_text().splitlines()[1:]
              if int(line.split()[1].split(":")[1], 16) == port]
    assert "06" in states, "fixture did not leave the owned server endpoint in TIME_WAIT"
    with socket.socket() as check:
        assert check.connect_ex(("127.0.0.1", port)) == 111
    result = json.loads((output / "manifest.json").read_text())
    first, following = result["cases"]
    assert first["cleanup"]["reaped"] is True
    assert following["status"] == "failed" and "pid" not in following
    assert following["elapsed_seconds"] >= 0.28
    preflight = following["port_preflight"]
    assert preflight["status"] == "failed" and preflight["attempts"] >= 2
    assert preflight["waited_seconds"] >= 0.28
    assert preflight["previous_owned_server"]["pid"] == first["pid"]
    assert "Address already in use" in preflight["first_error"]
    assert "timeout" in preflight["error"]


def test_following_case_waits_for_released_nonlistening_port(tmp_path, harness, monkeypatch):
    driver = load_driver()
    harness["port_release_timeout"] = 2
    harness["cases"].append({**harness["cases"][0], "name": "next"})
    original_stop = driver.stop_server
    port = int(harness["base_url"].split(":")[2].split("/")[0])
    release_thread = None
    blocking_socket = None
    stopped = 0

    def stop_and_delay_release(child, timeout):
        nonlocal release_thread, blocking_socket, stopped
        cleanup = original_stop(child, timeout)
        stopped += 1
        if stopped == 1:
            # Emulate delayed kernel release at the external socket boundary without a 60s test.
            blocking_socket = socket.socket()
            blocking_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            blocking_socket.bind(("127.0.0.1", port))
            blocking_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
            release_thread = threading.Thread(target=lambda: (time.sleep(0.25), blocking_socket.close()))
            release_thread.start()
        return cleanup

    monkeypatch.setattr(driver, "stop_server", stop_and_delay_release)
    try:
        result = driver.run_sweep(harness, tmp_path / "run", RUNNER)
    finally:
        if release_thread is not None:
            release_thread.join()
        if blocking_socket is not None:
            blocking_socket.close()
    assert result["status"] == "complete"
    first, following = result["cases"]
    assert_gone(first["pid"]); assert_gone(following["pid"])
    preflight = following["port_preflight"]
    assert preflight["status"] == "released" and preflight["attempts"] >= 2
    assert preflight["waited_seconds"] >= 0.2
    assert preflight["previous_owned_server"]["pid"] == first["pid"]


def test_following_case_still_refuses_live_foreign_listener_immediately(tmp_path, harness, monkeypatch):
    driver = load_driver()
    harness["port_release_timeout"] = 2
    harness["cases"].append({**harness["cases"][0], "name": "next"})
    original_stop = driver.stop_server
    port = int(harness["base_url"].split(":")[2].split("/")[0])
    foreign = socket.socket()
    foreign.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    stopped = 0

    def stop_and_install_foreign_listener(child, timeout):
        nonlocal stopped
        cleanup = original_stop(child, timeout)
        stopped += 1
        if stopped == 1:
            foreign.bind(("127.0.0.1", port)); foreign.listen()
        return cleanup

    monkeypatch.setattr(driver, "stop_server", stop_and_install_foreign_listener)
    try:
        result = driver.run_sweep(harness, tmp_path / "run", RUNNER)
        first, following = result["cases"]
        assert first["cleanup"]["reaped"] is True
        assert following["status"] == "failed" and "pid" not in following
        assert following["elapsed_seconds"] < 0.5
        assert "listener" in following["error"]
        with socket.create_connection(("127.0.0.1", port), timeout=0.2):
            pass
    finally:
        foreign.close()
