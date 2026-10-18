#!/usr/bin/env python3
"""Start persistent bilingual Tinymist previews plus automatic PDF compilation.

Run from any directory. State and logs: submission/records/typst-live/.
Only recorded owned processes are stopped by --stop.
"""
import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
DECK=ROOT/'submission/basic/presentation'
HERE=Path(__file__).resolve().parent
STATE=ROOT/'submission/records/typst-live'
RECORD=STATE/'processes.json'
TINYMIST=Path.home()/'.local/bin/tinymist'


def alive(pid):
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False


def owned(pid):
    try:
        cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
        return str(Path(__file__).resolve()) in cmd or (str(TINYMIST) in cmd and 'InferX-' in cmd)
    except FileNotFoundError:return False


def dependencies():
    files=[p for p in DECK.glob('*.typ') if p.name!='main.typ']
    files += list((DECK/'figures').glob('*.pdf'))
    files += [HERE/'deck_figures.py',HERE/'build_charts.py',HERE/'cost_charts.py',HERE/'build_request_trace.py',HERE/'build_typst.py',ROOT/'submission/track-a/data/results.json',ROOT/'submission/track-a/data/cost-evidence.json']
    files += [HERE/'build_result_details.py', ROOT/'submission/track-a/data/result-details-sources.json']
    files += [HERE/'build_selected_campaign.py', ROOT/'submission/track-a/data/selected-campaign-sources.json']
    files += [HERE/'build_selected_quality.py', ROOT/'submission/track-a/data/selected-quality-sources.json']
    return {str(p):p.stat().st_mtime_ns for p in files}


def build(charts=False):
    if charts:subprocess.run(['uv','run',str(HERE/'build_charts.py')],cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(HERE/'build_typst.py')],cwd=ROOT,check=True)


def serve():
    children=[]
    signal.signal(signal.SIGTERM,lambda *_:sys.exit(0))
    try:
        for lang,http,data,control,title in [('ZH',18081,18084,18091,'InferX 中文 · Live'),('EN',18082,18085,18092,'InferX English · Live')]:
            for port in [http,data,control]:
                with socket.socket() as s:s.bind(('127.0.0.1',port))
            args=[str(TINYMIST),'preview','--no-open','--preview-mode','slide','--host',f'127.0.0.1:{http}',
                  '--data-plane-host',f'127.0.0.1:{data}','--control-plane-host',f'127.0.0.1:{control}',
                  '--page-title',title,f'InferX-{lang}.typ']
            log=(STATE/f'{lang.lower()}-preview.log').open('a')
            proc=subprocess.Popen(args,cwd=DECK,stdout=log,stderr=subprocess.STDOUT)
            children.append(proc)
        RECORD.write_text(json.dumps({'supervisor_pid':os.getpid(),'preview_pids':[p.pid for p in children],
             'urls':{'zh':'http://127.0.0.1:18081/','en':'http://127.0.0.1:18082/'},
             'cwd':str(DECK),'source':str(DECK/'deck.typ')},indent=2)+'\n')
        previous=dependencies()
        while True:
            time.sleep(.5)
            if any(p.poll() is not None for p in children):raise RuntimeError('Preview stopped; see preview logs.')
            current=dependencies()
            if current!=previous:
                time.sleep(.25)
                changed={p for p in current if current[p]!=previous.get(p)}
                charts=any(p.endswith(name) for p in changed for name in ['results.json','build_charts.py','cost_charts.py','cost-evidence.json','build_request_trace.py','build_result_details.py','result-details-sources.json','build_selected_campaign.py','selected-campaign-sources.json'])
                try:build(charts=charts)
                except subprocess.CalledProcessError as e:print('Compile failed; fix source to retry:',e,flush=True)
                previous=dependencies()
                print('Source change compiled.',flush=True)
    finally:
        for p in children:
            if p.poll() is None:p.terminate()
        for p in children:p.wait(timeout=10)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--serve',action='store_true');ap.add_argument('--stop',action='store_true');args=ap.parse_args()
    STATE.mkdir(parents=True,exist_ok=True)
    if args.serve:return serve()
    old=json.loads(RECORD.read_text()) if RECORD.exists() else {}
    pid=old.get('supervisor_pid')
    if args.stop:
        if pid and owned(pid):os.kill(pid,signal.SIGTERM)
        print('Stopped owned InferX preview processes.');return
    if pid and alive(pid) and owned(pid):
        print(json.dumps(old['urls'],ensure_ascii=False));return
    build()
    log=(STATE/'supervisor.log').open('a')
    proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--serve'],cwd=ROOT,
                          stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    for _ in range(40):
        if proc.poll() is not None:raise RuntimeError('Preview startup failed; see supervisor.log')
        if RECORD.exists() and json.loads(RECORD.read_text()).get('supervisor_pid')==proc.pid:
            print(RECORD.read_text());return
        time.sleep(.1)
    raise RuntimeError('Preview startup timed out; see supervisor.log')


if __name__=='__main__':main()
