#!/usr/bin/env python3
# /// script
# dependencies = ["pymupdf==1.28.2"]
# ///
"""Audit the current compiled deck/evidence snapshot; visual review is separate."""
import hashlib
import argparse
import json
from pathlib import Path
from xml.etree import ElementTree as ET
import pymupdf

ROOT=Path(__file__).resolve().parents[4]
DECK=ROOT/'submission/basic/presentation'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def require(value,message):
    if not value:raise AssertionError(message)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--allow-pending-visual',action='store_true',help='Run mechanical checks for an interim draft; never certify visual acceptance.')
    args=ap.parse_args()
    record=json.loads((ROOT/'submission/records/typst-build.json').read_text())
    source=json.loads((ROOT/'submission/track-a/data/results.json').read_text())
    data=json.loads((DECK/'benchmark-data.json').read_text())
    cost=json.loads((ROOT/'submission/track-a/data/cost-evidence.json').read_text())
    details=json.loads((ROOT/'submission/track-a/data/result-details.json').read_text())
    require(details==json.loads((DECK/'result-details.json').read_text()),'Stale result detail data')
    for rel,item in details['sources'].items():
        require(sha(ROOT/rel)==item['sha256'],f'Result evidence changed: {rel}')
    campaign=json.loads((ROOT/'submission/track-a/data/selected-campaign.json').read_text())
    require(campaign==json.loads((DECK/'selected-campaign.json').read_text()),'Stale selected-campaign data')
    for rel,item in campaign['sources'].items():
        require(sha(ROOT/rel)==item['sha256'],f'Selected-campaign evidence changed: {rel}')
    quality=json.loads((ROOT/'submission/track-a/data/selected-quality.json').read_text())
    require(quality==json.loads((DECK/'selected-quality.json').read_text()),'Stale selected-quality data')
    for rel,item in quality['sources'].items():
        require(sha(ROOT/rel)==item['sha256'],f'Selected-quality evidence changed: {rel}')
    for rel,item in cost['sources'].items():
        require(sha(ROOT/rel)==item['sha256'],f'Cost evidence changed: {rel}')
    figure_ids=set(json.loads((DECK/'figure-data.json').read_text()))
    require(data['performance']=={r['id']:r for r in source['performance']},'Stale performance data')
    require(data['quality']==source['quality'],'Stale quality data')
    require(sha(DECK/'main.pdf')==sha(DECK/'InferX-ZH.pdf'),'main.pdf differs from Chinese PDF')
    for rel,digest in record['inputs'].items():require(sha(ROOT/rel)==digest,f'Source changed: {rel}')
    for lang in ['zh','en']:
        item=record['editions'][lang];pdf=ROOT/item['pdf'];digest=sha(pdf)
        require(item['sha256']==digest,f'Stale {lang} build record')
        if not args.allow_pending_visual:
            require(item['visual_inspection']['status']=='PASS','Visual review pending')
        require(item['visual_inspection']['pdf_sha256']==digest,'Visual review is for another PDF')
        require(not item['warnings'],f'Compiler warnings: {lang}')
        doc=pymupdf.open(pdf);require(len(doc)==item['pages'],f'Wrong page count: {lang}')
        for i,page in enumerate(doc):
            require(not page.get_images(),f'Unexpected raster image: {lang} page {i+1}')
            require(page.get_text().strip(),f'Missing selectable text: {lang} page {i+1}')
            require(page.get_drawings(),f'Missing vector objects: {lang} page {i+1}')
            for block in page.get_text('dict')['blocks']:
                for line in block.get('lines',[]):
                    for span in line['spans']:
                        box=pymupdf.Rect(span['bbox'])
                        require(box.x0>=-1 and box.y0>=-1 and box.x1<=page.rect.width+1 and box.y1<=page.rect.height+1,
                                f'Text outside page: {lang}/{i+1}/{span["text"]}')
        require(len(item['previews'])==len(doc),'Missing all-page previews')
        for rel,digest in item['previews'].items():require(sha(ROOT/rel)==digest,f'Preview changed: {rel}')
        text=''.join(p.get_text() for p in doc)
        for claim in ['85 / 100','88 / 100','96 / 111','99 / 111','48GB','306.04','309.95']:
            require(claim in text,f'Missing measured claim {claim}: {lang}')
        diagrams=ET.parse(DECK/f'InferX-figures-{lang.upper()}.drawio').getroot().findall('diagram')
        require({d.get('id') for d in diagrams}==figure_ids,
                'Missing editable overview or mechanism figures')
        for diagram in diagrams:
            cells=diagram.findall('.//mxCell')
            require(any(c.get('source') and c.get('target') for c in cells),'Unconnected draw.io export')
            require(all('data:image' not in c.get('style','') for c in cells),'Raster draw.io wrapper')
    for pdf in (DECK/'figures').glob('*.pdf'):
        page=pymupdf.open(pdf)[0]
        require(not page.get_images(),f'Raster plot: {pdf.name}')
        require(page.get_text().strip() and page.get_drawings(),f'Plot is not vector/text: {pdf.name}')
    manifest=json.loads((ROOT/'submission/track-a/evidence-manifest.json').read_text())
    for key in ['inputs','outputs']:
        for rel,item in manifest[key].items():require(sha(ROOT/rel)==item['sha256'],f'Manifest mismatch: {rel}')
    page_count=sum(item['pages'] for item in record['editions'].values())
    require(len({item['pages'] for item in record['editions'].values()})==1,'Language editions have different page counts')
    scope='MECHANICAL PASS (visual acceptance not certified)' if args.allow_pending_visual else 'PASS'
    print(f'{scope}: {page_count} slides; {len(list((DECK/"figures").glob("*.pdf")))} vector-PDF plots; {2*len(figure_ids)} editable diagrams; {len(manifest["inputs"])} evidence inputs; source/PDF/preview hashes match.')

if __name__=='__main__':main()
