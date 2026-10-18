#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib==3.10.7"]
# ///
"""Audit selected build 100 campaign and export independent bilingual vector plots.

CPU only. Does not edit Typst, published slide assets, or historical evidence.
Run: uv run submission/tooling/inferx-submission/scripts/build_selected_campaign.py
"""
import hashlib
import argparse
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
REGISTRY = ROOT / 'submission/track-a/data/selected-campaign-sources.json'
DATA = ROOT / 'submission/track-a/data/selected-campaign.json'
OUT = ROOT / 'submission/records/selected-campaign'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path, value):
    text = json.dumps(value, ensure_ascii=False, indent=2) + '\n'
    if not path.exists() or path.read_text() != text:
        path.write_text(text)


def p95(values):
    vals = sorted(values)
    x = .95 * (len(vals) - 1)
    lo, hi = math.floor(x), math.ceil(x)
    return vals[lo] + (x - lo) * (vals[hi] - vals[lo])


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-8), (a, b)


def audit():
    reg = json.loads(REGISTRY.read_text())
    sources = {str(REGISTRY.relative_to(ROOT)): {'sha256': digest(REGISTRY.read_bytes())}}
    rows, common, gpu_by_node = [], [], {}
    for excluded in reg['excluded_attempts']:
        for kind, item in excluded['sources'].items():
            raw = (ROOT / item['path']).read_bytes()
            assert digest(raw) == item['sha256'], item['path']
            sources[item['path']] = {'sha256': digest(raw), 'bytes': len(raw)}
            if kind == 'summary':
                rejected = json.loads(raw)
                assert rejected['status'] == 'CONFIGURATION_REJECTED_BEFORE_MODEL_OR_REQUESTS'
                assert rejected['measured'] == rejected['native_done'] == 0
                assert rejected['performance_slo_evaluated'] is False
            elif kind == 'server_log':
                assert b'--kv-capacity must be at least --max-context' in raw
    for entry in reg['runs'] + reg.get('comparison_runs', []):
        def read(key, lines=False, only_hash=False):
            item = entry['sources'][key]
            raw = (ROOT / item['path']).read_bytes()
            sha = digest(raw)
            assert sha == item['sha256'], f"Pinned raw source changed: {item['path']}"
            sources[item['path']] = {'sha256': sha, 'bytes': len(raw)}
            if only_hash:
                return sha
            return [json.loads(line) for line in raw.splitlines() if line] if lines else json.loads(raw)

        summary, analysis, phase = read('summary'), read('analysis'), read('phase')
        identity, config, env = read('identity'), read('config'), read('environment')
        command, records, native = read('command'), read('records', lines=True), read('native', lines=True)
        payload, dataset = read('payload', only_hash=True), read('dataset', only_hash=True)
        c, cap = entry['client_c'], entry['native_c']
        assert identity['binary_sha256'] == reg['binary_sha256']
        assert identity['source_sha256'] == reg['source_sha256']
        runtime = analysis.get('runtime', {})
        assert runtime.get('binary_sha256', analysis.get('binary')) == reg['binary_sha256']
        assert runtime.get('source_sha256', analysis.get('source')) == reg['source_sha256']
        assert analysis.get('client_C', analysis.get('C')) == c == summary['concurrency']
        assert analysis.get('native_C', analysis.get('native_cap')) == cap
        assert summary['status'] == 'complete' and summary['error_requests'] == analysis['errors'] == 0
        assert analysis['measured'] == config['requests'] == summary['records'] == summary['valid_records'] == 128
        assert summary['lengths_verified'] and summary['osl_verified']
        assert config['warmup'] == analysis['warmup'] == 8 and config['seed'] == analysis['seed'] == 20261016
        assert config['cases'][0]['concurrencies'] == [c]
        for name, expected in [('--concurrency', c), ('--request-count', 128), ('--warmup-request-count', 8), ('--random-seed', 20261016)]:
            assert int(command[command.index(name) + 1]) == expected
        formal = [r for r in records if r['metadata']['benchmark_phase'] == 'profiling']
        assert len(formal) == 128 and Counter(r['metadata']['benchmark_phase'] for r in records) == {'profiling': 128, 'warmup': 8}
        assert len({r['metadata']['x_request_id'] for r in formal}) == 128
        for r in records:
            m = r['metrics']
            assert m['input_sequence_length']['value'] == 8192 and m['output_sequence_length']['value'] == 512
            assert m['usage_prompt_cache_read_tokens']['value'] == 0
            assert not r['metadata']['was_cancelled'] and not r['metadata']['context_overflow_skip']
        starts = [r for r in native if r.get('event') == 'server_start']
        done = [r for r in native if r.get('event') == 'request_done']
        assert len(starts) == 1 and len(done) == 137
        startup = starts[0]
        assert startup['engine']['max_concurrency'] == cap
        kv = max(65536, 8704 * c)
        assert startup['engine']['kv_capacity'] == kv and startup['engine']['max_context'] == 65536
        assert startup['engine']['prefix_reuse'] is False
        assert startup['engine']['speculative_draft_window'] == 3
        assert startup['engine']['speculative_backend'] == 'mtp'
        assert startup['engine']['prefill_chunk'] == 2048
        assert Counter(r['result']['completion_tokens'] for r in done) == {512: 136, 1: 1}
        assert all(r['result']['prefix_cache_hit_tokens'] == 0 and r['result']['computed_prefill_tokens'] == 8192 for r in done)
        assert all(r['result']['prompt_tokens'] == 8192 for r in done)
        gpu = startup['environment']['gpu_uuid']
        assert gpu_by_node.setdefault(entry['node'], gpu) == gpu
        argv = config['cases'][0]['command']
        flag = lambda name: argv[argv.index(name) + 1]
        assert int(flag('--max-concurrency')) == cap and int(flag('--kv-capacity')) == kv
        assert int(flag('--draft-tokens')) == 3
        assert env['NINFER_SCHED_BOUNDED_DECODE_MEMBERSHIP'] == '1'
        assert env['NINFER_PROMPT_LOOKUP_N5_PRIORITY'] == env['NINFER_PROMPT_LOOKUP_N5_HYBRID'] == '0'
        extra = json.loads(command[command.index('--extra-inputs') + 1])
        assert extra['temperature'] == 0 and extra['ignore_eos'] is True
        # Compare routing/configuration while preserving actual paths in the source record.
        normalized_env = {k: Path(v).name if k.endswith('_CACHE') else v for k, v in env.items()}
        normalized_argv = argv.copy()
        normalized_argv[0] = '<pinned-binary>'
        for name in ['--max-concurrency', '--kv-capacity', '--port', '--request-log-jsonl']:
            normalized_argv[normalized_argv.index(name) + 1] = '<varies>'
        common.append((payload, dataset, extra, normalized_env, normalized_argv))
        ttft = p95([r['metrics']['time_to_first_token']['value'] for r in formal])
        tpot = p95([(r['metrics']['request_latency']['value'] - r['metrics']['time_to_first_token']['value']) / 511 for r in formal])
        for key, value, analysis_key in [('time_to_first_token', ttft, 'TTFT_p95_ms'), ('inter_token_latency', tpot, 'TPOT_p95_ms')]:
            close(value, summary['metrics'][key]['p95'])
            close(value, analysis['metrics'][analysis_key])
            close(value, phase[key]['p95'])
        assert phase['is_complete'] and not phase['was_cancelled'] and not phase['error_summary']
        assert phase['completed_request_count']['avg'] == phase['request_count']['avg'] == 128
        output_tokens = sum(r['metrics']['output_token_count']['value'] for r in formal)
        assert output_tokens == phase['total_output_tokens']['avg'] == 65536
        phase_window = next(p for p in read('phase_manifest')['phases'] if p['phase_kind'] == 'profiling')
        assert phase_window['total_request_count'] == phase_window['successful_request_count'] == 128
        assert phase_window['error_request_count'] == 0 and not phase_window['was_cancelled']
        duration = (phase_window['end_ns'] - phase_window['start_ns']) / 1e9
        throughput = phase['output_token_throughput']['avg']
        close(output_tokens / duration, throughput)
        close(throughput, summary['output_token_throughput'])
        close(throughput, analysis['metrics']['output_tokens_per_second'])
        passed = ttft <= 10000 and tpot <= 40
        assert passed == summary['meets_slo']
        rows.append({k: entry[k] for k in ['id', 'node', 'client_c', 'native_c', 'repeat']} | {
            'kv': kv, 'k': 3, 'n': 128, 'warmup': 8, 'excluded_probe': 1,
            'gpu_uuid': gpu, 'gpu_name': startup['environment']['gpu_name'],
            'ttft_s': ttft / 1000, 'tpot_ms': tpot, 'tps': throughput, 'joint_slo': passed,
            'TTFT_margin_ms': 10000 - ttft, 'TPOT_margin_ms': 40 - tpot,
            'throughput_numerator_tokens': output_tokens,
            'throughput_phase_duration_s': duration,
            'throughput_denominator_source': entry['sources']['phase_manifest']['path'],
            'throughput_phase_start_ns': phase_window['start_ns'],
            'throughput_phase_end_ns': phase_window['end_ns'],
            'reported_phase_benchmark_duration_s': phase['benchmark_duration']['avg'],
            'summary_benchmark_duration_s': summary['benchmark_duration'],
            'payload_sha256': payload, 'dataset_sha256': dataset,
            'source_paths': {k: v['path'] for k, v in entry['sources'].items()}})
    assert all(x == common[0] for x in common), 'Payload / normalized configuration mismatch'
    assert len(set(gpu_by_node.values())) == len(gpu_by_node), 'Node labels do not identify distinct GPUs'
    # Both C14 repetitions are mandatory, with true client/native distinction.
    main_ids = {r['id'] for r in reg['runs']}
    c14 = [r for r in rows if r['client_c'] == 14 and r['id'] in main_ids]
    assert len(c14) == 2 and {r['repeat'] for r in c14} == {1, 2}
    assert all(r['native_c'] == 13 and r['node'] == 'ucloud3' for r in c14)
    result = {k: reg[k] for k in ['schema_version', 'campaign', 'binary_sha256', 'source_sha256', 'pending', 'excluded_attempts', 'quality_status', 'scope']}
    result.update({'performance': [r for r in rows if r['id'] in main_ids],
                   'comparison_runs': [r for r in rows if r['id'] not in main_ids],
                   'sources': sources, 'validation': {
        'P95': 'Linear quantile over all original 128 profiling requests; TPOT=(E2E-TTFT)/(512-1). Warmup and one probe excluded.',
        'throughput': '65536 output tokens divided by (profiling end_ns-start_ns)/1e9 from phase_manifest.json; matched raw phase output_token_throughput, parser summary, and audited analysis. Neither the nearby benchmark_duration metric nor the request timestamp span is substituted for this denominator.',
        'comparability': 'Same binary/source, exact actual payload including salt, normalized route flags and runtime arguments; node, client/native capacity, KV and repetition explicitly retained. Cache path basenames are normalized, not a claim of freshly hashing all cache bytes.',
        'C14': 'Both measured runs shown separately. No mean/error-bar substitution, no long-run guarantee, no claim that native14 passed.',
        'quality': 'No quality inference from build6632 or any earlier build.'}})
    write(DATA, result)
    return result


def plots(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.lines import Line2D
    from matplotlib.transforms import Bbox
    plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 12,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.edgecolor': '#657180', 'axes.linewidth': .7,
                         'pdf.fonttype': 42, 'axes.unicode_minus': False})
    font = FontProperties(family=['Liberation Sans', 'Droid Sans Fallback'], size=12)
    dest = OUT / 'figures'
    dest.mkdir(parents=True, exist_ok=True)
    rows = sorted(data['performance'], key=lambda r: (r['client_c'], r['repeat']))
    colors = {'ucloud1': '#7E2635', 'ucloud3': '#316C93'}
    for lang in ['zh', 'en']:
        zh = lang == 'zh'
        fig, axes = plt.subplots(1, 3, figsize=(292 / 25.4, 126 / 25.4))
        fig.subplots_adjust(left=.055, right=.985, bottom=.49, top=.84, wspace=.32)
        specs = [('ttft_s', 'TTFT P95 (s)', 12, [0, 5, 10], 10),
                 ('tpot_ms', 'TPOT P95 (ms/token)', 48, [0, 20, 40], 40),
                 ('tps', '总吞吐量 (tok/s)' if zh else 'Aggregate output (tok/s)', 350, [0, 100, 200, 300], None)]
        for ax, (key, title, ymax, yticks, target) in zip(axes, specs):
            ax.set_axisbelow(True)
            ax.yaxis.grid(True, color='#D5DBE1', linewidth=.5)
            for node, marker in [('ucloud1', 'o'), ('ucloud3', 's')]:
                regular = [r for r in rows if r['node'] == node and r['repeat'] == 1]
                ax.plot([r['client_c'] for r in regular], [r[key] for r in regular],
                        marker=marker, color=colors[node], linewidth=1.5, markersize=5)
            repeat = next(r for r in rows if r['client_c'] == 14 and r['repeat'] == 2)
            ax.plot(14, repeat[key], marker='D', fillstyle='none', color=colors['ucloud3'],
                    markersize=7, markeredgewidth=1.3, linestyle='none')
            ax.set(ylim=(0, ymax), yticks=yticks, xlim=(3, 15), xticks=[4, 8, 12, 14])
            ax.set_title(title, loc='left', fontproperties=font if zh else None, pad=10)
            ax.set_xlabel('客户端并发' if zh else 'Client concurrency', fontproperties=font if zh else None)
            if target:
                ax.axhline(target, color='#287657', linestyle=(0, (4, 3)), linewidth=.9)
                ax.text(.02, target / ymax + .025, f'SLO {target}', transform=ax.transAxes, color='#287657', fontsize=10)
        handles = [Line2D([], [], color=colors['ucloud1'], marker='o', label='ucloud1'),
                   Line2D([], [], color=colors['ucloud3'], marker='s', label='ucloud3 · r1'),
                   Line2D([], [], color=colors['ucloud3'], marker='D', fillstyle='none', linestyle='none', label='ucloud3 · C14 r2')]
        fig.legend(handles=handles, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(.52, .99))
        table_ax = fig.add_axes([.055, .135, .93, .265])
        table_ax.axis('off')
        headers = ['节点 / 轮次', '客户端 / 引擎', 'KV Token', 'TTFT P95 (s)', 'TPOT P95 (ms)', 'tok/s'] if zh else ['Node / run', 'Client / native', 'KV tokens', 'TTFT P95 (s)', 'TPOT P95 (ms)', 'tok/s']
        cells = [[f"{r['node']} · r{r['repeat']}", f"{r['client_c']} / {r['native_c']}", str(r['kv']),
                  f"{r['ttft_s']:.3f}", f"{r['tpot_ms']:.3f}", f"{r['tps']:.2f}"] for r in rows]
        table = table_ax.table(cellText=cells, colLabels=headers, loc='center', cellLoc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor('#D5DBE1')
            cell.set_linewidth(.4)
            if row == 0:
                cell.set_facecolor('#F5F0F1')
                cell.get_text().set_weight('bold')
                if zh:
                    cell.get_text().set_fontproperties(font)
        note = ('同一100版本；两台节点分别连线，非同机纯并发效应。每轮128测量＋8预热，冷8192/512，K3。\n'
                'C14为客户端14 / 引擎13（含等待）；KV=max(65536,8704×客户端并发)。新版本任务精度待测。') if zh else (
                'Same build 100; lines stay within each node, not a same-node concurrency-only effect. 128 + 8, cold 8192/512, K3.\n'
                'C14: client14 / native13, including queueing. KV=max(65536,8704×client C). New-build task quality is pending.')
        fig.text(.055, .075, note, va='top', fontproperties=font if zh else None, fontsize=11)
        if data['pending']:
            fig.text(.055, .975, 'C4待正式原始结果' if zh else 'C4 awaits completed raw results', ha='left', va='top',
                     color='#657180', fontproperties=font if zh else None, fontsize=10)
        pdf = dest / f'selected-campaign-{lang}.pdf'
        fig.savefig(pdf, metadata={'Title': 'InferX build 100 selected campaign', 'CreationDate': None, 'ModDate': None})
        # Slide edition keeps conditions as normal Typst body text, not a small plot footer.
        w, h = fig.get_size_inches()
        fig.savefig(dest / f'selected-campaign-plot-{lang}.pdf', bbox_inches=Bbox.from_bounds(0, .12*h, w, .88*h),
                    metadata={'Title': 'InferX build100 campaign curves and measurements', 'CreationDate': None, 'ModDate': None})
        fig.savefig(dest / f'selected-campaign-{lang}.png', dpi=160)
        plt.close(fig)


def include_c4():
    """Explicitly pin the already-authorized, completed legal C4 run; never launch it."""
    reg = json.loads(REGISTRY.read_text())
    if any(r['client_c'] == 4 for r in reg['runs']):
        return
    pending = next(p for p in reg['pending'] if p['client_c'] == 4)
    base = ROOT / pending['path']
    report = json.loads((base / 'summary.json').read_text())
    assert report['status'] in ['COMPLETE_JOINT_PASS', 'COMPLETE_JOINT_FAIL']
    assert report['runtime']['binary_sha256'] == reg['binary_sha256']
    assert report['KV'] == 65536 and report['client_C'] == report['native_C'] == 4
    raw, run = base / 'delivery/raw', base / 'delivery/raw/candidate/c4'
    files = {'analysis': base / 'summary.json', 'summary': run / 'summary.json',
             'records': run / 'profile_export.jsonl', 'phase': run / 'phases/profiling/profile_export_aiperf.json',
             'phase_manifest': run / 'phase_manifest.json', 'command': run / 'command.json',
             'identity': raw / 'candidate-identity.json', 'config': raw / 'config.json',
             'payload': run / 'inputs.json', 'dataset': raw / 'dataset.jsonl',
             'native': raw / 'candidate/requests.jsonl', 'environment': base / 'delivery/control/environment.json'}
    reg['runs'].insert(0, {'id': '100-ucloud3-c4-native4-kv65536-r1', 'node': 'ucloud3',
                           'client_c': 4, 'native_c': 4, 'repeat': 1,
                           'sources': {k: {'path': str(p.relative_to(ROOT)), 'sha256': digest(p.read_bytes())} for k, p in files.items()}})
    reg['pending'].remove(pending)
    write(REGISTRY, reg)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-c4', action='store_true', help='Pin the completed legal C4 run from the registered pending path; CPU-only.')
    args = parser.parse_args()
    if args.include_c4:
        include_c4()
    OUT.mkdir(parents=True, exist_ok=True)
    data = audit()
    plots(data)
    text = ['# Selected build 100 campaign', '', data['scope'], '',
            '| Node | Client / native | Run | KV | TTFT P95 s | TPOT P95 ms | Output tok/s | Phase denominator s |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in data['performance']:
        text.append(f"| {r['node']} | {r['client_c']} / {r['native_c']} | {r['repeat']} | {r['kv']} | {r['ttft_s']:.9f} | {r['tpot_ms']:.9f} | {r['tps']:.9f} | {r['throughput_phase_duration_s']:.9f} |")
    text += ['', 'Separate failed boundary configurations (not added to the six-point main curve):', '',
             '| Node | Client / native | TTFT P95 s | TPOT P95 ms | Output tok/s | SLO |',
             '|---|---:|---:|---:|---:|---|']
    for r in data['comparison_runs']:
        text.append(f"| {r['node']} | {r['client_c']} / {r['native_c']} | {r['ttft_s']:.9f} | {r['tpot_ms']:.9f} | {r['tps']:.9f} | {'PASS' if r['joint_slo'] else 'FAIL'} |")
    text += ['', 'C16/native13 is a separate admission-capacity diagnostic, not a passing capacity point. All raw measurements remain retained.']
    text += ['', 'Quality: ' + data['quality_status'], '', 'Pending: ' + json.dumps(data['pending']), '',
             'Excluded setup attempt: ' + json.dumps(data['excluded_attempts'], ensure_ascii=False), '',
             'Rebuild: `uv run submission/tooling/inferx-submission/scripts/build_selected_campaign.py`', '',
             'No edits to Typst, historical result-details data, published figures, or final PDFs. These are independent integration artifacts.']
    (OUT / 'README.md').write_text('\n'.join(text) + '\n')
    outputs = [DATA, OUT / 'README.md'] + sorted((OUT / 'figures').glob('selected-campaign-*'))
    write(OUT / 'manifest.json', {'sources': data['sources'], 'outputs': {str(p.relative_to(ROOT)): {'sha256': digest(p.read_bytes()), 'bytes': p.stat().st_size} for p in outputs},
                                'tool_sha256': digest(Path(__file__).read_bytes()), 'visual_inspection': 'pending'})
    print(f"Verified {len(data['performance'])} formal runs, {len(data['sources'])} source hashes; C4 pending={bool(data['pending'])}. Independent vector charts: {OUT / 'figures'}")


if __name__ == '__main__':
    main()
