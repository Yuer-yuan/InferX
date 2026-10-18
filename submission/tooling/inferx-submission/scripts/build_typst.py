#!/usr/bin/env python3
"""Refresh shared evidence/diagram assets, compile both Typst PDFs and render previews.

python3 submission/tooling/inferx-submission/scripts/build_typst.py [--render]
Measured chart PDFs: uv run submission/tooling/inferx-submission/scripts/build_charts.py
"""
import argparse
import hashlib
import json
import math
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from deck_figures import build_figures, drawio_document, COLORS
from build_result_details import build_result_details
from build_selected_campaign import audit as audit_selected_campaign
from build_selected_quality import audit as audit_selected_quality

ROOT = Path(__file__).resolve().parents[4]
DECK = ROOT / 'submission/basic/presentation'
RECORDS = ROOT / 'submission/records'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_changed(path, text):
    if not path.exists() or path.read_text() != text:
        path.write_text(text)


def assets():
    write_changed(DECK / 'result-details.json', json.dumps(build_result_details(), ensure_ascii=False, indent=2)+'\n')
    campaign = audit_selected_campaign()
    write_changed(DECK / 'selected-campaign.json', json.dumps(campaign, ensure_ascii=False, indent=2)+'\n')
    quality = audit_selected_quality()
    write_changed(DECK / 'selected-quality.json', json.dumps(quality, ensure_ascii=False, indent=2)+'\n')
    for language in ['zh', 'en']:
        plot = RECORDS / 'selected-campaign/figures' / f'selected-campaign-plot-{language}.pdf'
        target = DECK / 'figures' / f'selected-campaign-{language}.pdf'
        if not target.exists() or target.read_bytes() != plot.read_bytes():
            shutil.copy2(plot, target)
    c14 = next(r for r in campaign['performance'] if r['client_c'] == 14 and r['repeat'] == 2)
    raw = json.loads((ROOT / c14['source_paths']['summary']).read_text())
    excerpt = {k: raw[k] for k in ['concurrency', 'meets_slo', 'error_requests']}
    excerpt['metrics'] = {k: {'p95': raw['metrics'][k]['p95']} for k in ['time_to_first_token', 'inter_token_latency']}
    excerpt['output_token_throughput'] = raw['output_token_throughput']
    quality_raw = json.loads((ROOT / quality['summary_source']).read_text())
    quality_excerpt = {k: quality_raw[k] for k in ['samples', 'instructions']}
    quality_excerpt['metrics'] = {k: {'score': v['score']} for k, v in quality_raw['metrics'].items()}
    quality_json = '{\n' + ''.join(f'  "{k}": {quality_excerpt[k]},\n' for k in ['samples', 'instructions'])
    quality_json += '  "metrics": {\n' + ',\n'.join(f'    "{k}": '+json.dumps(v) for k,v in quality_excerpt['metrics'].items()) + '\n  }\n}'
    assert json.loads(quality_json) == quality_excerpt
    write_changed(DECK / 'selected-reproduction.json', json.dumps({
        'source': c14['source_paths']['summary'], 'source_sha256': sha(ROOT / c14['source_paths']['summary']),
        'scope': 'Actual build100 C14/native13 run2 output excerpt. Not a new run of the displayed portable commands.',
        'performance_json': json.dumps(excerpt, ensure_ascii=False, indent=2),
        'quality_source': quality['summary_source'], 'quality_source_sha256': sha(ROOT / quality['summary_source']),
        'quality_scope': 'Actual complete evaluation105 original100 score excerpt. Not a new run of the displayed portable commands.',
        'quality_json': quality_json}, ensure_ascii=False, indent=2)+'\n')
    source = json.loads((ROOT / 'submission/track-a/data/results.json').read_text())
    frozen = {k: source[k] for k in ['team', 'quality']}
    frozen['performance'] = {r['id']: r for r in source['performance']}
    write_changed(DECK / 'benchmark-data.json',json.dumps(frozen, ensure_ascii=False, indent=2)+'\n')
    figs = build_figures()
    for language in ['zh', 'en']:
        write_changed(DECK / f'InferX-figures-{language.upper()}.drawio',drawio_document(figs, language))
    ports = {'east':(1,0), 'west':(-1,0), 'north':(0,-1), 'south':(0,1),
             'north west':(-1,-.65), 'south west':(-1,.65)}
    for f in figs.values():
        nodes = {n['id']: n for n in f['nodes']}
        for n in f['nodes']:
            n['stroke'],n['fill'] = COLORS[n['role']]
        f['bounds'] = [min(n['x']-n['w']/2 for n in f['nodes'])-.1,
                       min(n['y']-n['h']/2 for n in f['nodes'])-.1,
                       max(n['x']+n['w']/2 for n in f['nodes'])+.1,
                       max(n['y']+n['h']/2 for n in f['nodes'])+.1]
        def port(node, name):
            dx,dy=ports[name]
            return [node['x']+dx*node['w']/2,node['y']+dy*node['h']/2]
        def arrow(a,b):
            dx,dy=b[0]-a[0],b[1]-a[1]
            norm=math.hypot(dx,dy);dx/=norm;dy/=norm
            return [b,[b[0]-.12*dx+.045*dy,b[1]-.12*dy-.045*dx],
                       [b[0]-.12*dx-.045*dy,b[1]-.12*dy+.045*dx]]
        for e in f['edges']:
            points=[port(nodes[e['a']],e['ap'])]+e['via']+[port(nodes[e['b']],e['bp'])]
            e['points']=points;e['arrows']=[arrow(points[-2],points[-1])]
            if e['kind']=='bidirectional':e['arrows'].append(arrow(points[1],points[0]))
    write_changed(DECK / 'figure-data.json',json.dumps(figs,ensure_ascii=False,indent=2)+'\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--render',action='store_true');args=ap.parse_args()
    assets()
    typst=shutil.which('typst') or str(Path.home()/'.local/bin/typst')
    record={'updated_at':datetime.now().astimezone().isoformat(),
            'compiler':subprocess.check_output([typst,'--version'],text=True).strip(),
            'chart_backend':'Matplotlib 3.10.7, embedded vector PDF', 'editions':{}}
    for lang in ['ZH','EN']:
        entry=f'InferX-{lang}.typ';output=f'InferX-{lang}.pdf'
        compiled=subprocess.run([typst,'compile','--no-pdf-tags',entry,output],cwd=DECK,text=True,capture_output=True)
        (RECORDS / f'typst-{lang.lower()}.log').write_text(compiled.stdout+compiled.stderr)
        if compiled.returncode:raise RuntimeError(compiled.stderr)
        pdf=DECK/output
        page_info=subprocess.check_output(['pdfinfo',str(pdf)],text=True)
        pages=int(next(l.split(':')[1] for l in page_info.splitlines() if l.startswith('Pages:')))
        assert pages>0,(lang,pages)
        item={'pdf':str(pdf.relative_to(ROOT)),'sha256':sha(pdf),'pages':pages,'warnings':compiled.stderr,
              'visual_inspection':{'status':'pending','pdf_sha256':sha(pdf)}}
        if args.render:
            dest=RECORDS/'typst'/lang.lower();dest.mkdir(parents=True,exist_ok=True)
            for stale in dest.glob('page-*.png'):stale.unlink()
            subprocess.run(['pdftoppm','-scale-to','1440','-png',str(pdf),str(dest/'page')],check=True)
            item['previews']={str(p.relative_to(ROOT)):sha(p) for p in sorted(dest.glob('page-*.png'))}
        record['editions'][lang.lower()]=item
    shutil.copy2(DECK/'InferX-ZH.pdf',DECK/'main.pdf')
    write_changed(DECK/'main.typ','#import "deck.typ": render\n#render("zh")\n')
    record['inputs']={str(p.relative_to(ROOT)):sha(p) for p in sorted(list(DECK.glob('*.typ'))+list(DECK.glob('*.json'))+list((DECK/'figures').glob('*.pdf'))+[ROOT/'submission/track-a/data/cost-evidence.json'])}
    (RECORDS/'typst-build.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print('Compiled '+', '.join(f'{lang.upper()}: {item["pages"]} pages' for lang,item in record['editions'].items())+'. main.pdf matches Chinese edition.')


if __name__=='__main__':main()
