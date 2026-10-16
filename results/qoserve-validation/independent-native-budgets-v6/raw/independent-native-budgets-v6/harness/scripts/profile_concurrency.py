"""Separate bounded CUDA timeline runs; never mixed into latency/throughput groups."""
import copy
import json
import subprocess
import time

from scripts.run_concurrency_campaign import DIRECTORY, Server, group, snapshots


def run_profiles(grid, templates):
    directory = DIRECTORY/'profiles'
    directory.mkdir(exist_ok=False)
    records = []
    for draft, concurrency in [(0,1),(7,4)]:
        configuration = dict(id=f'profile-d{draft}-c{concurrency}', draft=draft,
                             concurrency=concurrency, profiler=True)
        target = directory/configuration['id']
        report = target/'trace'
        prefix = ['nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none',
                  '--cuda-graph-trace=node','--delay=75','--duration=60','--kill=none',
                  '--force-overwrite=false','--output',str(report)]
        server = Server(configuration, grid, prefix=prefix, directory=target)
        record = dict(configuration=configuration, trace_delay_s=75, trace_duration_s=60,
                      independent_of_performance_groups=True)
        try:
            record['startup'] = server.start()
            time.sleep(max(0, 80-(time.monotonic()-server.launched_at)))
            local = copy.deepcopy(templates)
            local['code-512']['payload']['max_tokens'] = 256
            record['measurement'] = group(server, 'profile-code',
                snapshots(['code-512'], concurrency*2, local, 'profile-code'),
                ['code-512'], local, write_index=False)
            record['status'] = 'traced'
        except Exception as exc:
            record.update(status='profile_error', error=f'{type(exc).__name__}: {exc}')
        finally:
            server.stop()
        path = report.with_suffix('.nsys-rep')
        if path.exists():
            export = subprocess.run(['nsys','stats','--report','cuda_gpu_kern_sum,cuda_api_sum',
                '--format','csv','--output',str(target/'stats'),str(path)],
                capture_output=True, text=True, timeout=120)
            (target/'stats.log').write_text(export.stdout+export.stderr)
            record['stats_returncode'] = export.returncode
            record['report_bytes'] = path.stat().st_size
        else:
            record['report_missing'] = True
        (target/'profile.json').write_text(json.dumps(record, indent=2))
        records.append(record)
        print(json.dumps(dict(event='profile_done', case=configuration['id'],
                               status=record['status'], report_exists=path.exists())), flush=True)
    return records
