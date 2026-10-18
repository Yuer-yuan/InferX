#!/usr/bin/env python3
"""Evidence-driven InferX materials; standard library, no model/network calls."""
import argparse
from deck_figures import build_figures, tikz_figure, drawio_document
from datetime import datetime, timezone, timedelta
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import os
import shutil
import subprocess
import sys
import tempfile

METRICS = ('prompt_level_strict', 'inst_level_strict', 'prompt_level_loose', 'inst_level_loose')
SKILL = Path(__file__).resolve().parents[1]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def dump(value):
    return json.dumps(value, ensure_ascii=False, indent=2) + '\n'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temp = Path(handle.name)
        handle.write(value.encode())
    temp.replace(path)


def p95(values):
    values = sorted(values)
    if not values:
        raise ValueError('empty percentile input')
    pos = .95 * (len(values) - 1)
    lo = math.floor(pos)
    return values[lo] + (values[math.ceil(pos)] - values[lo]) * (pos - lo)


class Evidence:
    def __init__(self, repo):
        self.repo, self.files = repo, {}

    def read(self, rel):
        path = (self.repo / rel).resolve()
        if not path.is_relative_to(self.repo):
            raise ValueError(f'evidence outside repo: {rel}')
        before = path.stat()
        data = path.read_bytes()
        after = path.stat()
        if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
            raise ValueError(f'input is changing: {rel}')
        self.files[rel] = {'sha256': sha(data), 'bytes': len(data)}
        return gzip.decompress(data) if path.suffix == '.gz' else data

    def json(self, rel):
        return json.loads(self.read(rel))

    def lines(self, rel):
        return [json.loads(line) for line in self.read(rel).splitlines() if line.strip()]

    def unchanged(self):
        for rel, item in self.files.items():
            if sha((self.repo / rel).read_bytes()) != item['sha256']:
                raise ValueError(f'input changed during collection: {rel}')


def near(actual, expected, what):
    if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-6):
        raise ValueError(f'{what}: raw {actual} != summary {expected}')


def collect(e, cfg):
    out = []
    for item in cfg['performance']:
        row = dict(item)
        d = e.json(item['summary'])
        if 'select' in item:
            for key in item['select']:
                d = d[key]
        if d.get('status') not in ('complete', '完整') or d.get('issues'):
            raise ValueError(f'incomplete performance: {item["id"]}')
        if d['concurrency'] != item['c']:
            raise ValueError(f'concurrency mismatch: {item["id"]}')
        if d['records'] != item['n'] or d['valid_records'] != item['n'] or d['error_requests'] != 0:
            raise ValueError(f'count/error mismatch: {item["id"]}')
        if not d['lengths_verified'] or not d['osl_verified']:
            raise ValueError('unverified token lengths')
        m = d['metrics']
        row.update(ttft_s=m['time_to_first_token']['p95']/1000,
                   tpot_ms=m['inter_token_latency']['p95'],
                   e2e_s=m['request_latency']['p95']/1000,
                   tps=d['output_token_throughput'], errors=d['error_requests'])
        row['joint_slo'] = row['ttft_s'] <= 10 and row['tpot_ms'] <= 40
        row['verification'] = 'original summary only'
        if item.get('identity'):
            identity = e.json(item['identity'])
            actual_binary = identity.get('binary_sha256', identity.get('binary_SHA256'))
            actual_source = identity.get('source_sha256', identity.get('source_SHA256'))
            if actual_binary != item['binary']:
                raise ValueError(f'binary identity mismatch: {item["id"]}')
            if not isinstance(actual_source, str) or len(actual_source) != 64:
                raise ValueError(f'missing full source identity: {item["id"]}')
            if item.get('source_identity') and item['source_identity'] != actual_source:
                raise ValueError(f'source identity mismatch: {item["id"]}')
            row['source_identity'] = actual_source
        if item.get('raw'):
            all_rows = e.lines(item['raw'])
            raw = [r for r in all_rows if r['metadata'].get('benchmark_phase') == 'profiling']
            warm = [r for r in all_rows if r['metadata'].get('benchmark_phase') == 'warmup']
            if len(raw) != item['n'] or len(warm) != item['warmup']:
                raise ValueError(f'raw/warmup count mismatch: {item["id"]}')
            ttft, tpot, latency = [], [], []
            for r in raw:
                rm, meta = r['metrics'], r['metadata']
                if meta.get('was_cancelled') or meta.get('context_overflow_skip') or meta.get('error'):
                    raise ValueError('failed raw request retained: cannot declare zero-error completion')
                value = lambda key: rm[key]['value']
                if value('input_sequence_length') != 8192 or value('output_sequence_length') != 512:
                    raise ValueError('raw ISL/OSL mismatch')
                ttft.append(value('time_to_first_token'))
                latency.append(value('request_latency'))
                if not (math.isfinite(ttft[-1]) and math.isfinite(latency[-1]) and 0 <= ttft[-1] <= latency[-1]):
                    raise ValueError('invalid client timing')
                token_time = (latency[-1]-ttft[-1])/511
                near(token_time, value('inter_token_latency'), 'request TPOT')
                tpot.append(token_time)
            near(p95(ttft)/1000, row['ttft_s'], 'TTFT P95')
            near(p95(tpot), row['tpot_ms'], 'TPOT P95')
            near(p95(latency)/1000, row['e2e_s'], 'E2E P95')
            row['individual_violations'] = {'ttft': sum(x>10000 for x in ttft), 'tpot': sum(x>40 for x in tpot)}
            row['verification'] = 'original summary + all client records recomputed'
        if item.get('native'):
            native = e.lines(item['native'])
            done = [r for r in native if r.get('event') == 'request_done']
            if len(done) != item['n'] + item['warmup'] + 1:
                raise ValueError(f'native count mismatch: {item["id"]}')
            for r in done:
                v = r['result']
                if (v['prompt_tokens'] != 8192 or v['computed_prefill_tokens'] != 8192
                        or v['prefix_cache_hit_tokens'] != 0 or v['prefix_reuse_path'] != 'full_reset'):
                    raise ValueError('native cold-input mismatch')
            if sum(r['result']['completion_tokens']==512 for r in done) != item['n']+item['warmup']:
                raise ValueError('native completion mismatch')
            start = next(r for r in native if r.get('event')=='server_start')
            engine = start['engine']
            if (engine['max_concurrency'] != item.get('native_capacity', item['c']) or engine['speculative_draft_window'] != item['k']
                    or engine['kv_capacity'] != item['kv']):
                raise ValueError('native configuration mismatch')
            row['native_cold'] = len(done)
            row['gpu'] = start['environment']
            row['startup_memory'] = start['memory']
        row['native_capacity'] = item.get('native_capacity', item['c'])
        out.append(row)
    quality = []
    for item in cfg['quality']:
        if not isinstance(item.get('client_c'), int) or item['client_c'] < 1:
            raise ValueError('quality requires explicit positive client concurrency')
        if not isinstance(item.get('protocol'), str) or not item['protocol']:
            raise ValueError('quality requires explicit protocol description')
        d = e.json(item['summary'])
        if d['samples'] != 100 or d['instructions'] != 111 or d['attempts'] != 100:
            raise ValueError('quality count mismatch')
        if not d['original_verifier_recomputed_all100'] or not d['all_native_score_values_match']:
            raise ValueError('quality original-verifier review missing')
        row = dict(item, scores={k:d['metrics'][k]['score'] for k in METRICS},
                   empty=d['empty_final'], length=d['length'], errors=d['failures'],
                   n=d['samples'], instructions=d['instructions'])
        if len(item['binary']) != 64:
            raise ValueError('quality requires full binary SHA256')
        if item.get('identity'):
            identity = e.json(item['identity'])
            if identity.get('binary_sha256', identity.get('binary_SHA256')) != item['binary']:
                raise ValueError('quality binary identity mismatch')
            source = identity.get('source_sha256', identity.get('source_SHA256', identity.get('source_aggregate_sha256')))
            if item.get('source_identity') and source != item['source_identity']:
                raise ValueError('quality source identity mismatch')
        if item.get('runtime'):
            runtime = e.json(item['runtime'])
            if (runtime['binary']['binary_sha256'] != item['binary']
                    or runtime['binary']['source_aggregate_sha256'] != item['source_identity']):
                raise ValueError('quality actual runtime identity mismatch')
            engine = runtime['actual_startup']['engine']
            if (runtime['resolved_generation']['batch_size'] != item['client_c']
                    or engine['max_concurrency'] != item['native_capacity']
                    or engine['speculative_draft_window'] != item['k'] or engine['kv_capacity'] != item['kv']):
                raise ValueError('quality actual runtime configuration mismatch')
            for key, value in item.get('route_flags', {}).items():
                if runtime['actual_process']['environment'].get(key) != value:
                    raise ValueError(f'quality route flag mismatch: {key}')
        for k, value in row['scores'].items():
            denominator = 100 if k.startswith('prompt') else 111
            if not 0 <= value <= 1 or not math.isclose(value*denominator,round(value*denominator),abs_tol=1e-7):
                raise ValueError('invalid quality denominator')
        quality.append(row)
    ledgers = []
    for rel in cfg['ledgers']:
        encoded = e.read(rel)
        rows = [json.loads(line) for line in encoded.splitlines() if line.strip()]
        # Only exact duplicate records are removed. A row is not necessarily a GPU run.
        unique = {json.dumps(r,sort_keys=True,ensure_ascii=False) for r in rows}
        # Freeze the exact observed ledger: live experiments can append later without
        # invalidating the snapshot. Original evidence files above remain hash-checked.
        digest = sha(encoded)
        frozen = f'submission/track-a/data/ledger-snapshots/{digest}.jsonl'
        if not (e.repo/frozen).exists():
            write(e.repo/frozen, encoded.decode())
        del e.files[rel]
        e.read(frozen)
        ledgers.append({'path':rel, 'snapshot':frozen, 'records':len(rows), 'distinct_records':len(unique)})
    for rel in cfg['supporting_evidence']:
        e.read(rel)
    return {'schema_version':1, 'state':'rolling_materials_snapshot_not_final_release',
            'team':cfg['team'], 'roles':cfg['roles'], 'performance':out, 'quality':quality,
            'workload':{'ledgers':ledgers,'distinct_records':sum(r['distinct_records'] for r in ledgers),
                        'meaning':'Exact-deduplicated ledger records, including CPU, operator, model and audit stages; not independent successful GPU experiment count.'},
            'notes':cfg['notes']}


def tex_escape(value):
    return ''.join({'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_',
                    '{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}.get(c,c) for c in str(value))


def render(snapshot, template, language='zh'):
    byid = {r['id']:r for r in snapshot['performance']}
    roles = snapshot['roles']
    rows = [byid[k] for k in roles['capacity_plot']]
    best = byid[roles['headline_capacity']]
    quality = next(r for r in snapshot['quality'] if r['id']==roles['headline_quality'])
    if not best['joint_slo']:
        raise ValueError('headline_capacity must be a reviewed SLO pass')
    replacements = {'DATE':snapshot['updated_at_shanghai'], 'TEAM':tex_escape(snapshot['team']['name']),
                    'MEMBERS':tex_escape('、'.join(snapshot['team']['members'])),
                    'RECORDS':str(snapshot['workload']['distinct_records']),
                    'BEST_C':str(best['c']), 'BEST_TPS':f'{best["tps"]:.2f}',
                    'BEST_TTFT':f'{best["ttft_s"]:.2f}', 'BEST_TPOT':f'{best["tpot_ms"]:.2f}',
                    'QUALITY':f'{quality["scores"]["prompt_level_strict"]*100:.0f}',
                    'QUALITY_BINARY':quality['binary'][:8],
                    'PERF_ROWS':'\n'.join(f'C{r["c"]} & {r["binary"][:4]} & {r["ttft_s"]:.2f} & {r["tpot_ms"]:.2f} & {r["tps"]:.2f} & '+('通过' if r['joint_slo'] else '未过')+r' \\' for r in rows),
                    'LATENCY_COORDS':' '.join(f'({r["ttft_s"]:.6f},{r["tpot_ms"]:.6f})' for r in rows),
                    'TPS_COORDS':' '.join(f'({r["c"]},{r["tps"]:.6f})' for r in rows)}
    replacements['MAIN_PERF_ROWS'] = '\n'.join(
        f'{r["c"]} & {r["ttft_s"]:.2f} & {r["tpot_ms"]:.2f} & {r["tps"]:.2f} & '+('通过' if r['joint_slo'] else '未通过')+r' \\'
        for r in rows)
    replacements['DATE_SHORT'] = snapshot['updated_at_shanghai'][:10]
    replacements['RECOMPUTED_N'] = str(sum(r['n'] for r in snapshot['performance'] if r.get('raw')))
    replacements['RECOMPUTED_WARMUP'] = str(sum(r['warmup'] for r in snapshot['performance'] if r.get('raw')))
    replacements.update(QUALITY_C=str(quality['client_c']), QUALITY_NATIVE=str(quality['native_capacity']),
                        QUALITY_KV=str(quality['kv']), QUALITY_EMPTY=str(quality['empty']),
                        QUALITY_LENGTH=str(quality['length']), QUALITY_ERRORS=str(quality['errors']),
                        QUALITY_STRICT_N=str(round(quality['scores']['prompt_level_strict']*100)),
                        QUALITY_LOOSE_N=str(round(quality['scores']['prompt_level_loose']*100)),
                        QUALITY_INST_STRICT_N=str(round(quality['scores']['inst_level_strict']*111)),
                        QUALITY_INST_LOOSE_N=str(round(quality['scores']['inst_level_loose']*111)),
                        QUALITY_STRICT_FAILURES=str(100-round(quality['scores']['prompt_level_strict']*100)))
    pass_width = 12 * quality['scores']['prompt_level_strict']
    replacements.update(QUALITY_PASS_WIDTH=f'{pass_width:.6f}',
                        QUALITY_PASS_CENTER=f'{pass_width/2:.6f}',
                        QUALITY_FAIL_CENTER=f'{(pass_width+12)/2:.6f}')
    passed=round(quality['scores']['prompt_level_strict']*100)
    replacements['QUALITY_PASS_CELLS']=','.join(f'{i%10}/{i//10}' for i in range(passed))
    replacements['QUALITY_FAIL_CELLS']=','.join(f'{i%10}/{i//10}' for i in range(passed,100))
    for r in rows:
        replacements[f'TTFT_{r["c"]}'] = f'{r["ttft_s"]:.6f}'
        replacements[f'TPOT_{r["c"]}'] = f'{r["tpot_ms"]:.6f}'
    stages = [byid[k] for k in roles['progression_plot']]
    replacements['PROGRESSION_COORDS'] = ' '.join(f'({i+1},{r["tps"]:.6f})' for i,r in enumerate(stages))
    replacements['PROGRESSION_LABELS'] = ','.join(tex_escape(r['plot_label']) for r in stages)
    replacements['PROGRESSION_GAIN'] = f'{(stages[-1]["tps"]/stages[0]["tps"]-1)*100:.1f}'
    replacements['QUALITY_COORDS'] = ' '.join(f'({i+1},{r["scores"]["prompt_level_strict"]*100:.6f})' for i,r in enumerate(snapshot['quality']))
    replacements['BASELINE_ROWS'] = '\n'.join(
        f'{r["c"]} & {r["ttft_s"]:.2f} & {r["tpot_ms"]:.2f} & {r["tps"]:.2f}'+r' \\'
        for r in (byid['zip-c2'], byid['zip-c8']))
    baseline = [byid[f'zip-c{c}'] for c in (1,2,4,8)]
    for field, token in [('tps','BASELINE_TPS_COORDS'),('tpot_ms','BASELINE_TPOT_COORDS')]:
        replacements[token] = ' '.join(f'({i+1},{r[field]:.6f})' for i,r in enumerate(baseline))
    replacements['PROGRESSION_TPOT_COORDS'] = ' '.join(
        f'({i+1},{r["tpot_ms"]:.6f})' for i,r in enumerate(stages))
    replacements['PROGRESSION_TTFT_VALUES'] = ' / '.join(f'{r["ttft_s"]:.2f}' for r in stages)
    replacements['QUALITY_ROWS'] = '\n'.join(
        f'{r["binary"][:4]} & {100*r["scores"]["prompt_level_strict"]:.0f} & '
        f'{r["errors"]}/{r["empty"]}/{r["length"]}'+r' \\' for r in snapshot['quality'])
    control, candidate = byid['gqa-c32'], byid['fusion-c32']
    replacements['GDN_TPOT_COORDS'] = f'(1,{control["tpot_ms"]:.6f}) (2,{candidate["tpot_ms"]:.6f})'
    replacements['GDN_TPS_BEFORE'] = f'{control["tps"]:.2f}'
    replacements['GDN_TPS_AFTER'] = f'{candidate["tps"]:.2f}'
    replacements['GDN_TPS_CHANGE'] = f'{100*(candidate["tps"]/control["tps"]-1):.2f}'
    replacements['GDN_TPOT_CHANGE'] = f'{100*(candidate["tpot_ms"]/control["tpot_ms"]-1):.2f}'
    cap_control, cap_candidate = byid['fusion-c16'], byid['fusion-client16-native12']
    if (any(cap_control[key] != cap_candidate[key] for key in ('node','binary','source_identity','c','n','warmup','k','kv'))
            or (cap_control['native_capacity'], cap_candidate['native_capacity']) != (16,12)):
        raise ValueError('native-cap comparison requires reviewed matching client16 configurations')
    replacements['CAPACITY_ROWS'] = '\n'.join(
        f'{r["native_capacity"]} & {r["ttft_s"]:.2f} & {r["tpot_ms"]:.2f} & {r["tps"]:.2f} & 未过'+r' \\'
        for r in (cap_control, cap_candidate))
    replacements['CAP_TPOT_REDUCTION'] = f'{100*(1-cap_candidate["tpot_ms"]/cap_control["tpot_ms"]):.2f}'
    replacements['CAP_TTFT_INCREASE'] = f'{100*(cap_candidate["ttft_s"]/cap_control["ttft_s"]-1):.2f}'
    replacements['CAP_TPS_REDUCTION'] = f'{100*(1-cap_candidate["tps"]/cap_control["tps"]):.2f}'
    for field, token in [('ttft_s','CAP_TTFT_COORDS'),('tpot_ms','CAP_TPOT_COORDS')]:
        replacements[token] = ' '.join(
            f'({i+1},{r[field]:.6f})' for i,r in enumerate((cap_control,cap_candidate)))
    progress = byid['yield-c16']
    replacements['YIELD_BEFORE'] = f'{cap_control["tpot_ms"]:.2f}'
    replacements['YIELD_AFTER'] = f'{progress["tpot_ms"]:.2f}'
    replacements['YIELD_REDUCTION'] = f'{100*(1-progress["tpot_ms"]/cap_control["tpot_ms"]):.2f}'
    chapters_zh=['项目概述','问题与目标','整体方案','核心实现','结果与证据','工程化设计','成本、性能与取舍','总结与不足']
    chapters_en=['Project overview','Problem and goals','System design','Implementation','Results and evidence','Engineering','Cost, performance and tradeoffs','Conclusions and limitations']
    replacements['LANGUAGE']=language
    replacements['LANG_INDEX']='1' if language=='zh' else '2'
    romanized={'陈文欣':'Wenxin Chen','张志雄':'Zhixiong Zhang','郭坤昌':'Kunchang Guo'}
    replacements['MEMBERS_EN']=', '.join(romanized.get(name,name) for name in snapshot['team']['members'])
    for i,title in enumerate(chapters_zh if language=='zh' else chapters_en,1):
        replacements[f'CHAPTER_{i}']=title
    for name,figure in build_figures().items():
        replacements[f'FIG_{name.upper()}']=tikz_figure(figure,language)
    result = template
    for key,val in replacements.items():
        result = result.replace('@@'+key+'@@',val)
    if re.search(r'@@[A-Z0-9_]+@@',result):
        raise ValueError('unfilled TeX template marker')
    return result


def markdown(s):
    lines=['# 测试成绩与证据索引','',f'数据整理时间：{s["updated_at_shanghai"]}。']
    for title, rows in [
        ('ZIP默认历史评测：100正式+8预热', [r for r in s['performance'] if r['id'].startswith('zip-')]),
        ('扩展容量与版本实验：128正式+8预热', [r for r in s['performance'] if not r['id'].startswith('zip-')])]:
        lines+=['', '## '+title, '',
                '| 配置/节点 | 版本 | N/预热 | TTFT P95(s) | TPOT P95(ms) | E2E P95(s) | tok/s | 两项SLO |',
                '|---|---|---:|---:|---:|---:|---:|---|']
        for r in rows:
            lines.append(f'| {r["label"]}/{r["node"]} | {r["binary"][:8]} | {r["n"]}/{r["warmup"]} | {r["ttft_s"]:.3f} | {r["tpot_ms"]:.3f} | {r["e2e_s"]:.3f} | {r["tps"]:.3f} | '+('通过' if r['joint_slo'] else '未通过')+' |')
    lines+=['','以上两表的协议/版本分别对应；只有明确标注的同机同payload增量对照才支持该增量的描述，不作为普遍加速倍数。扩展128请求不是ZIP默认100请求。','',
            '## IFBench质量：按版本分列', '',
            '| 质量版本 | 实际测试配置 | prompt strict | inst strict | prompt loose | inst loose | 错误/空final/length |','|---|---|---:|---:|---:|---:|---|']
    for r in s['quality']:
        lines.append('| '+r['label']+' / '+r['binary'][:8]+' | '+r['protocol']+' | '+' | '.join(f'{100*r["scores"][k]:.2f}%' for k in METRICS)+f' | {r["errors"]}/{r["empty"]}/{r["length"]} |')
    lines+=['','均为IFBench原前100、111指令、max32768、默认thinking及模型采样。历史C2与新C16的并发、KV及路由分别列示，不解释为纯量化收益或配对非劣；质量不自动覆盖其他binary或配置。6632质量KV532480，性能KV139264且TPOT未过，不能称同配置联合达标。','',
            '## 工作量口径','',f'三个ledger共 {s["workload"]["distinct_records"]} 条逐ledger去除完全重复行后的记录；包含CPU、算子、整模型和审计，不等于独立成功GPU实验次数。','',
            '## 可追溯来源','']
    for r in s['performance']+s['quality']:
        link='../../'+r['summary']
        lines += [f'- {r["id"]}：[汇总]({link})；binary `{r["binary"]}`；'+r.get('verification','原版verifier复核汇总')+'。']
    return '\n'.join(lines)+'\n'


def pdf_page_count(path):
    info=subprocess.run(['pdfinfo',str(path)],capture_output=True,text=True,check=True).stdout
    match=re.search(r'^Pages:\s+(\d+)',info,re.M)
    if not match or int(match.group(1))<1:
        raise ValueError('PDF must contain at least one readable page')
    return int(match.group(1))


def check_presentation_structure(tex):
    """Keep the required chapter order while allowing multi-frame chapters."""
    research='InferX research deck language:' in tex
    english='InferX research deck language: en' in tex
    tex='\n'.join(re.split(r'(?<!\\)%',line)[0] for line in tex.splitlines())
    required=['项目概述','问题与目标','整体方案','核心实现','结果与证据','工程化设计','成本、性能与取舍','总结与不足']
    if english:
        required=['Project overview','Problem and goals','System design','Implementation','Results and evidence','Engineering','Cost, performance and tradeoffs','Conclusions and limitations']
    titles=re.findall(r'\\begin\{frame\}(?:\[[^\]]*\])?\{([^}]+)\}',tex)
    if len(titles)!=len(re.findall(r'\\begin\{frame\}',tex)):
        raise ValueError('every presentation frame requires its chapter title')
    chapters=[]
    for title in titles:
        if not chapters or title!=chapters[-1]:chapters.append(title)
    if chapters!=required:
        raise ValueError('presentation requires all eight chapters in rule order; consecutive chapter pages are allowed')
    if research:
        if re.findall(r'\\usetheme\{([^}]+)\}',tex) != ['default']:
            raise ValueError('bilingual research deck requires the white default canvas')
        if r'\setbeamertemplate{navigation symbols}{}' not in tex:
            raise ValueError('research deck must omit navigation decorations')
    elif re.findall(r'\\usetheme\{([^}]+)\}',tex) != ['Madrid']:
        raise ValueError('legacy presentation requires Madrid')


def compile_pdf(repo, stem='main'):
    """Compile the current TeX and record the actual PDF page count."""
    tex=repo/f'submission/basic/presentation/{stem}.tex'
    pdf=tex.with_suffix('.pdf')
    report_path=repo/('submission/records/latex-build.json' if stem=='main' else 'submission/records/latex-build-en.json')
    engine=shutil.which('xelatex') or shutil.which('tectonic')
    if not engine:
        candidate=Path.home()/'.local/bin/tectonic'
        if candidate.is_file():engine=str(candidate)
    if not engine:
        raise ValueError('LaTeX compiler missing: install XeLaTeX or Tectonic before refresh')
    version=subprocess.run([engine,'--version'],capture_output=True,text=True,check=True).stdout.splitlines()[0]
    tex_sha=sha(tex.read_bytes())
    if report_path.exists() and pdf.exists():
        prior=json.loads(report_path.read_text())
        if (prior.get('status')=='PASS' and prior.get('tex_sha256')==tex_sha
                and prior.get('compiler_version')==version and prior.get('pdf_sha256')==sha(pdf.read_bytes())
                and prior.get('pages')==pdf_page_count(pdf)):
            bind_pdf_manifest(repo,prior,stem)
            print('PDF is current; verified source, compiler and output SHA')
            return
    build_root=repo/('submission/records/latex' if stem=='main' else 'submission/records/latex-en')
    build_root.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='build-',dir=build_root) as tmp:
        out=Path(tmp)
        if Path(engine).name=='tectonic':
            command=[engine,'--keep-logs','--outdir',str(out)]
            fonts=Path.home()/'.local/share/fonts/Fandol'
            if fonts.is_dir():command.extend(['-Z','search-path='+str(fonts)])
            command.append(str(tex))
            commands=[command]
        else:
            commands=[[engine,'-interaction=nonstopmode','-halt-on-error','-no-shell-escape',
                       '-output-directory='+str(out),str(tex)]]*2
        env=dict(os.environ)
        stamp=json.loads((repo/'submission/track-a/data/results.json').read_text())['updated_at_shanghai']
        env['SOURCE_DATE_EPOCH']=str(int(datetime.strptime(stamp,'%Y-%m-%d %H:%M %z').timestamp()))
        outputs=[]
        for command in commands:
            run=subprocess.run(command,cwd=tex.parent,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            outputs.append(run.stdout)
            if run.returncode:
                write(build_root/'compiler-output.txt','\n'.join(outputs))
                if (out/f'{stem}.log').exists():write(build_root/'main.log',(out/f'{stem}.log').read_text(errors='replace'))
                write(report_path,dump({'status':'FAIL','tex_sha256':tex_sha,'compiler_version':version,
                                       'returncode':run.returncode,'log':'submission/records/latex/compiler-output.txt'}))
                raise ValueError('LaTeX compilation failed; see submission/records/latex/compiler-output.txt')
        built=out/f'{stem}.pdf'
        if not built.exists():raise ValueError('compiler returned without PDF')
        pages=pdf_page_count(built)
        log=(out/f'{stem}.log').read_text(errors='replace') if (out/f'{stem}.log').exists() else '\n'.join(outputs)
        warnings=[line for line in log.splitlines() if re.search(r'Overfull|Missing character|undefined references|Undefined control',line)]
        write(build_root/'compiler-output.txt','\n'.join(outputs))
        write(build_root/'main.log',log)
        # Keep the previous PDF if compilation or PDF inspection fails above.
        temporary=pdf.with_suffix('.pdf.new')
        shutil.copyfile(built,temporary);temporary.replace(pdf)
        report={'status':'PASS','tex_sha256':tex_sha,'pdf_sha256':sha(pdf.read_bytes()),
                'compiler_version':version,'compiler_path':engine,'pages':pages,
                'warnings':warnings,'log':str((build_root/'main.log').relative_to(repo)),
                'visual_inspection':'pending; render and inspect every page before delivery'}
        write(report_path,dump(report))
        bind_pdf_manifest(repo,report,stem)
        print(f'PDF compiled: {pages} pages; {len(warnings)} layout/font warnings')


def bind_pdf_manifest(repo, report, stem='main'):
    path=repo/'submission/track-a/evidence-manifest.json'
    if not path.exists():return
    d=json.loads(path.read_text())
    tex=f'submission/basic/presentation/{stem}.tex'
    pdf=f'submission/basic/presentation/{stem}.pdf'
    if d['outputs'].get(tex,{}).get('sha256')==report['tex_sha256']:
        d['outputs'][pdf]={'sha256':report['pdf_sha256'],'bytes':(repo/pdf).stat().st_size}
        d['compiler']={'version':report['compiler_version'],'pages':report['pages']}
        write(path,dump(d))


def render_pdf(repo, stem='main'):
    pdf=repo/f'submission/basic/presentation/{stem}.pdf'
    path=repo/('submission/records/latex-build.json' if stem=='main' else 'submission/records/latex-build-en.json')
    report=json.loads(path.read_text())
    if report.get('status')!='PASS' or report['pdf_sha256']!=sha(pdf.read_bytes()):
        raise ValueError('cannot render stale or failed PDF')
    count=pdf_page_count(pdf)
    if report.get('pages')!=count:raise ValueError('build page count does not match actual PDF')
    target=repo/('submission/records/latex/preview' if stem=='main' else 'submission/records/latex-en/preview')
    target.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='render-',dir=target.parent) as tmp:
        rendered=Path(tmp)
        subprocess.run(['pdftoppm','-png','-scale-to','1500',str(pdf),str(rendered/'page')],check=True)
        indexed={}
        for p in rendered.glob('page-*.png'):
            match=re.fullmatch(r'page-(\d+)\.png',p.name)
            if match:
                index=int(match.group(1))
                if index in indexed:raise ValueError('duplicate PDF rendering page')
                indexed[index]=p
        if set(indexed)!=set(range(1,count+1)):raise ValueError('incomplete PDF rendering')
        # Poppler zero-pads names for larger PDFs; normalize and remove old pages.
        for index,p in indexed.items():shutil.copyfile(p,target/f'page-{index}.png')
        expected={f'page-{i}.png' for i in range(1,count+1)}
        for p in target.glob('page-*.png'):
            if re.fullmatch(r'page-\d+\.png',p.name) and p.name not in expected:p.unlink()
    pages=[target/f'page-{i}.png' for i in range(1,count+1)]
    report['rendered_pdf_sha256']=report['pdf_sha256']
    report['preview_files']={str(p.relative_to(repo)):sha(p.read_bytes()) for p in pages}
    write(path,dump(report))
    print(f'Rendered all {count} PDF pages for visual inspection')


def check(repo, stem='main'):
    manifest=json.loads((repo/'submission/track-a/evidence-manifest.json').read_text())
    for group in ['inputs','outputs']:
        for rel,item in manifest[group].items():
            if sha((repo/rel).read_bytes()) != item['sha256']:
                raise ValueError(f'{group} changed since refresh: {rel}; refresh after review')
    tex=(repo/f'submission/basic/presentation/{stem}.tex').read_text()
    build_path=repo/('submission/records/latex-build.json' if stem=='main' else 'submission/records/latex-build-en.json')
    build=json.loads(build_path.read_text())
    if (build.get('status')!='PASS'
            or build['tex_sha256']!=sha(tex.encode())
            or build['pdf_sha256']!=sha((repo/f'submission/basic/presentation/{stem}.pdf').read_bytes())):
        raise ValueError('PDF is not a successful compile of current TeX')
    pages=pdf_page_count(repo/f'submission/basic/presentation/{stem}.pdf')
    if build.get('pages')!=pages:raise ValueError('build page count does not match actual PDF')
    if build.get('warnings'):
        raise ValueError('unresolved layout/font warnings; fix and recompile before delivery')
    visual=build.get('visual_inspection',{})
    if (not isinstance(visual,dict) or visual.get('status')!='PASS'
            or visual.get('pdf_sha256')!=build['pdf_sha256'] or visual.get('pages')!=pages):
        raise ValueError('current PDF requires all-page visual inspection before delivery')
    preview_root='submission/records/latex' if stem=='main' else 'submission/records/latex-en'
    expected_previews={f'{preview_root}/preview/page-{i}.png' for i in range(1,pages+1)}
    if (build.get('rendered_pdf_sha256')!=build['pdf_sha256']
            or set(build.get('preview_files',{}))!=expected_previews):
        raise ValueError('current PDF rendering evidence missing')
    for rel,digest in build['preview_files'].items():
        if sha((repo/rel).read_bytes())!=digest:raise ValueError('PDF preview changed after inspection')
    check_presentation_structure(tex)
    for env in ['frame','columns','column','tikzpicture','axis','document','tabular']:
        if len(re.findall(r'\\begin\{'+env+r'\}',tex)) != len(re.findall(r'\\end\{'+env+r'\}',tex)):
            raise ValueError(f'unbalanced TeX environment {env}')
    if re.search(r'(?<!\\)\\[\u4e00-\u9fff0-9]',tex):
        raise ValueError('invalid single-backslash line break in TeX')
    if r'\newcommand{\note}' in tex:
        raise ValueError('custom command collides with Beamer speaker notes')
    # Conservative vocabulary for this generated template; this is a lexical
    # check, not a substitute for XeLaTeX. Extend after reviewing new constructs.
    known=set('noindent coordinate footnotesize Large addplot author begin bfseries bottomrule bullet color date definecolor dfrac documentclass draw end fill fontsize foreach hfill hspace insertframenumber insertframetitle item label large linewidth mathrm midrule newcommand node par path pgfplotsset quad raisebox rightarrow rule scriptsize selectfont setbeamercolor setbeamerfont setbeamersize setbeamertemplate setlength small tabcolsep textwidth tikzset tiny title toprule usepackage usetikzlibrary vspace y usetheme textbf leq centering texttt detokenize'.split())
    known.update(re.findall(r'\\newcommand\{\\([A-Za-z]+)\}',tex))
    known.update(('normalsize','times','section','cdots','href','textcolor','i','x','setsansfont','setmonofont','sansmath','insertframenumber','insertframetitle','sffamily','bfseries'))
    unknown=set(re.findall(r'(?<!\\)\\([A-Za-z]+)',tex))-known
    if unknown:raise ValueError(f'TeX command requires review: {sorted(unknown)}')
    depth=0
    for line in tex.splitlines():
        line=re.split(r'(?<!\\)%',line)[0]
        for match in re.finditer(r'(?<!\\)[{}]',line):
            depth += 1 if match.group()=='{' else -1
            if depth<0:raise ValueError('unbalanced TeX brace')
    if depth:raise ValueError('unbalanced TeX brace')
    if re.search(r'\\(?:input|include|includegraphics|addbibresource)\s*\{',tex):
        raise ValueError('single-file Overleaf contract: external resource found')
    for p in [repo/'submission/README.md', *list((repo/'submission/basic').rglob('*.md')),
              *list((repo/'submission/track-a').rglob('*.md'))]:
        if re.search(r'占位符|待团队|发布缺口|滚动材料快照|尚未最终冻结|赛前人工验收|未在Overleaf或本机实际编译',p.read_text()):
            raise ValueError(f'internal preparation note in submission prose: {p}; move to records')
        for target in re.findall(r'\[[^\]\n]+\]\(([^)\n]+)\)',p.read_text()):
            if re.match(r'\w+://',target) or target.startswith('#'): continue
            target=target.split('#')[0]
            if not (p.parent/target).exists():
                raise ValueError(f'broken local link {p}: {target}')
    print(f'PASS: source/output SHA, current {pages}-page PDF, chapter structure, resources and local links')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['inventory','refresh','compile','render','check'])
    ap.add_argument('--repo',type=Path,required=True)
    args=ap.parse_args(); repo=args.repo.resolve()
    stems=['main']+(['InferX-EN'] if (repo/'submission/basic/presentation/InferX-EN.tex').exists() else [])
    if args.command=='check':
        for stem in stems:check(repo,stem)
        return
    if args.command=='compile':
        for stem in stems:compile_pdf(repo,stem);render_pdf(repo,stem)
        return
    if args.command=='render':
        for stem in stems:render_pdf(repo,stem)
        return
    if args.command=='inventory':
        files=[]
        for node in ['ucloud1','ucloud2','ucloud3']:
            paths=(repo/'results/three-node-optimization'/node).rglob('*.md')
            candidates=[p for p in paths if p.name in ['REPORT.md','FINAL-RESULT.md','RESULT.md'] and not any(x.startswith('source') for x in p.relative_to(repo).parts)]
            files.extend(sorted(candidates,key=lambda p:p.stat().st_mtime,reverse=True)[:8])
        print(dump([{'path':str(p.relative_to(repo)),'modified_utc':datetime.fromtimestamp(p.stat().st_mtime,timezone.utc).isoformat()} for p in files]));return
    e=Evidence(repo); registry='submission/track-a/data/sources.json'; cfg=e.json(registry)
    snapshot=collect(e,cfg)
    template_path=SKILL/'assets/main.tex.in'; template=template_path.read_text()
    # Capture exact generator/template identity even when running the installed skill.
    tools={'refresh.py':sha(Path(__file__).read_bytes()),'main.tex.in':sha(template_path.read_bytes()),
           'deck_figures.py':sha((SKILL/'scripts/deck_figures.py').read_bytes())}
    fingerprint=sha(dump({'inputs':e.files,'tools':tools}).encode())
    target=repo/'submission/track-a/data/results.json'
    previous=json.loads(target.read_text()) if target.exists() else None
    if previous and previous.get('input_fingerprint')==fingerprint:
        snapshot['updated_at_shanghai']=previous['updated_at_shanghai']
    else:
        snapshot['updated_at_shanghai']=datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M +08:00')
    snapshot['input_fingerprint']=fingerprint
    snapshot['generator']=tools
    e.unchanged()
    outputs={'submission/track-a/data/results.json':dump(snapshot),
             'submission/track-a/generated-results.md':markdown(snapshot),
             'submission/basic/presentation/main.tex':render(snapshot,template,'zh'),
             'submission/basic/presentation/InferX-ZH.tex':render(snapshot,template,'zh'),
             'submission/basic/presentation/InferX-EN.tex':render(snapshot,template,'en')}
    for language,suffix in [('zh','ZH'),('en','EN')]:
        outputs[f'submission/basic/presentation/InferX-figures-{suffix}.drawio']=drawio_document(build_figures(),language)
    findings='# 自动汇总的材料验收事项\n\n更新时间：'+snapshot['updated_at_shanghai']+'。\n\n'
    findings+='\n'.join('- '+item for item in snapshot['notes'])+'\n'
    write(repo/'submission/records/generated-materials-findings.md',findings)
    if previous and previous.get('input_fingerprint')!=fingerprint:
        history=repo/'submission/track-a/data/history'/previous['input_fingerprint'][:16]
        for rel in outputs:
            old=repo/rel
            if old.exists(): write(history/(old.name+'.snapshot'),old.read_text())
        old_manifest=repo/'submission/track-a/evidence-manifest.json'
        if old_manifest.exists():write(history/'evidence-manifest.json.snapshot',old_manifest.read_text())
    for rel,value in outputs.items():write(repo/rel,value)
    manifest={'state':snapshot['state'],'updated_at_shanghai':snapshot['updated_at_shanghai'],
              'input_fingerprint':fingerprint,'tools':tools,'inputs':e.files,
              'outputs':{rel:{'sha256':sha(val.encode()),'bytes':len(val.encode())} for rel,val in outputs.items()},
              'scope':'Material evidence snapshot only, not private-source/model/cache release or complete clean-rebuild proof.'}
    write(repo/'submission/track-a/evidence-manifest.json',dump(manifest))
    print(f'Updated {len(outputs)} generated files; {len(e.files)} evidence inputs; snapshot {fingerprint[:16]}')
    compile_pdf(repo)
    render_pdf(repo)
    compile_pdf(repo,'InferX-EN')
    render_pdf(repo,'InferX-EN')
    chinese=repo/'submission/basic/presentation/InferX-ZH.pdf'
    shutil.copyfile(repo/'submission/basic/presentation/main.pdf',chinese)
    manifest=json.loads((repo/'submission/track-a/evidence-manifest.json').read_text())
    manifest['outputs']['submission/basic/presentation/InferX-ZH.pdf']={'sha256':sha(chinese.read_bytes()),'bytes':chinese.stat().st_size}
    write(repo/'submission/track-a/evidence-manifest.json',dump(manifest))


if __name__=='__main__':
    try:main()
    except (ValueError,KeyError,OSError,json.JSONDecodeError,subprocess.CalledProcessError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr);sys.exit(1)
