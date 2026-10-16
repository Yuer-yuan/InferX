"""Operator snapshot: launch frozen service, run unchanged judge, retain service."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from run_ninfer_sweep import load_runner, require_unused_port, start_ticks, wait_ready, write_json

r = Path(os.environ['INFERX_ROOT'])
config_path = r / 'config/official-r1.json'
config = json.loads(config_path.read_text())
service_dir = r / 'runs/official-r1-service'
output = r / 'runs/official-r1'
assert not service_dir.exists() and not output.exists(), 'refusing existing run directories'
runner_path = r / 'code/submission/organizer/runner.py'
runner = load_runner(runner_path)
expected = next(x for x in json.loads((r / 'code/submission/input-manifest.json').read_text())['extracted_members']
                if x['zip_member'] == 'runner.py')['sha256']
assert hashlib.sha256(runner_path.read_bytes()).hexdigest() == expected
command = config['server_command']
binary = Path(command[0])
assert hashlib.file_digest(binary.open('rb'), 'sha256').hexdigest() == config['binary_sha256']
assert hashlib.file_digest(Path(command[1]).open('rb'), 'sha256').hexdigest() == config['model_sha256']
assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip(), 'GPU in use'
gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
assert gpu == config['gpu_uuid']
require_unused_port(config['base_url'])
service_dir.mkdir(exist_ok=False)
write_json(service_dir / 'frozen-configuration.json', config)
write_json(service_dir / 'operator-process.json', {
    'pid': os.getpid(), 'start_ticks': start_ticks(os.getpid()),
    'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
    'runner_sha256': expected, 'started_at': runner.utc_now(),
    'note': 'Server intentionally remains available after evaluation, controlled by active-service.json.'
})

cleaning_sampler = False
deferred_cancellation = None

def cancel(signum, frame):
    global deferred_cancellation
    for number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(number, signal.SIG_IGN)
    if cleaning_sampler:
        deferred_cancellation = f'operator received {signal.Signals(signum).name} during cleanup'
        return
    raise KeyboardInterrupt('official operator cancelled')

for number in (signal.SIGINT, signal.SIGTERM):
    signal.signal(number, cancel)

query = 'timestamp,uuid,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm,clocks.mem,pcie.link.gen.current,pcie.link.width.current,clocks_event_reasons.active,clocks_event_reasons.sw_power_cap,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_thermal_slowdown'
telemetry = (service_dir / 'gpu.csv').open('w')
sampler = subprocess.Popen(['nvidia-smi', '--query-gpu=' + query,
                            '--format=csv,noheader,nounits', '--loop-ms=1000'],
                           stdout=telemetry, stderr=subprocess.STDOUT)
status = {'status': 'starting', 'started_at': runner.utc_now()}
rc = 1
try:
    server_log = (service_dir / 'server.log').open('ab', buffering=0)
    server = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=server_log,
                              stderr=subprocess.STDOUT, start_new_session=True,
                              env={**os.environ, 'CUDA_VISIBLE_DEVICES': '0'})
    identity = {
        'pid': server.pid, 'start_ticks': str(start_ticks(server.pid)), 'uid': os.getuid(),
        'command': command, 'started': time.time(), 'binary_sha256': config['binary_sha256'],
        'model_sha256': config['model_sha256'], 'gpu_uuid': gpu,
        'source_commit': config['engine_commit'], 'patch_sha256': config['patch_sha256'],
        'base_url': config['base_url'], 'source_main_commit': config['source_main_commit'],
        'keep_running_after_evaluation': True
    }
    for path in (r / 'config/active-service.json', service_dir / 'service.json'):
        write_json(path, identity)
    wait_ready(server, config['base_url'], 300)
    if sampler.poll() is not None:
        raise RuntimeError(f'GPU sampler exited before evaluation: {sampler.returncode}')
    runner.check_model(config['base_url'], config['model'], 'EMPTY')
    argv = [sys.executable, str(runner_path), '--base-url', config['base_url'],
            '--model', config['model'], '--tokenizer', str(r / 'models/tokenizer'),
            '--env-dir', str(r / 'eval-envs'), '--mode', 'all', '--perf-requests', '100',
            '--eval-batch-size', '2', '--request-timeout', '600', '--seed', '42',
            '--api-key-env', 'INFERX_LOCAL_EVAL_KEY', '--output-dir', str(output)]
    write_json(service_dir / 'judge-command.json', {
        'invocation': 'in_process_original_runner.main', 'argv': argv[2:],
        'equivalent_cli': argv
    })
    status.update(status='evaluating', ready_at=runner.utc_now())
    write_json(service_dir / 'operator-status.json', status)
    os.environ['INFERX_LOCAL_EVAL_KEY'] = 'EMPTY'
    # Keep the original judge's owned-worker cleanup directly in this process;
    # an outer launcher must not impose an equal, competing cleanup deadline.
    with (r / 'logs/official-r1-judge.log').open('w') as judge_log:
        with contextlib.redirect_stdout(judge_log), contextlib.redirect_stderr(judge_log):
            rc = runner.main(argv[2:])
    status['status'] = ('judge-exited-zero' if rc == 0 else
                        'cancelled' if rc == 130 else 'judge-reported-abnormal')
except KeyboardInterrupt as error:
    rc = 130
    status.update(status='cancelled', error=str(error))
except BaseException as error:
    status.update(status='failed', error=f'{type(error).__name__}: {error}')
    print(status['error'], flush=True)
finally:
    cleaning_sampler = True
    sampler.terminate()
    try:
        sampler.wait(timeout=10)
    except subprocess.TimeoutExpired:
        sampler.kill()
        sampler.wait()
    telemetry.close()
    if deferred_cancellation:
        rc = 130
        status.update(status='cancelled', error=deferred_cancellation)
    status.update(ended_at=runner.utc_now(), operator_exit_code=rc,
                  sampler_exit_code=sampler.returncode)
    write_json(service_dir / 'operator-status.json', status)
    (r / 'logs/official-r1.exitcode').write_text(str(rc) + '\n')
print(json.dumps(status), flush=True)
sys.exit(rc)
