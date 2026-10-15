"""Diagnostic admission audit; prefix progress is not physical cache accounting."""
import json,re,sys
from pathlib import Path
root,destination=map(Path,sys.argv[1:])
result=[]
for campaign in sorted(root.glob('pd-strategies-*/campaign.json')):
 c=json.loads(campaign.read_text())
 if c['status']=='running' or c.get('profiling_only'):continue
 for state in c.get('cases',[]):
  if 'group' not in state:continue
  case=campaign.parent/f"{state['case']}-r{state['repetition']}-{state['group']}"
  rp=case/state['group']/'result.json'
  if not rp.exists():continue
  measured=json.loads(rp.read_text());start=measured['started_unix_ms']/1000;end=measured['ended_unix_ms']/1000
  events=[]
  for line in (case/'server.log').read_text(errors='replace').splitlines():
   if 'PD_ADMISSION ' in line:
    e=json.loads(line.split('PD_ADMISSION ',1)[1])
    if start<=e['unix']<=end:events.append(e)
  long_events=[e for e in events if e['cached_tokens']+e['remaining_input']>=130000]
  admitted=[e for e in long_events if e['method']=='new' and e['admitted']]
  chunks=[e for e in long_events if e['method']=='chunk' and e['admitted']]
  begin=min((e['unix'] for e in admitted),default=None);last=max((e['unix'] for e in chunks),default=None)
  during={e['rid'] for e in events if begin is not None and last is not None and begin<=e['unix']<=last and e['method']=='new' and e['admitted'] and 20000<=e['cached_tokens']+e['remaining_input']<=30000}
  result.append(dict(run=campaign.parent.name,case=state['case'],group=state['group'],repetition=state['repetition'],long_first_admitted_s=begin-start if begin else None,long_last_chunk_admitted_s=last-start if last else None,short_new_admissions_during_long_chunks=len(during),long_candidate_attempts=len([e for e in long_events if e['method']=='new']),long_admitted_attempts=len(admitted)))
destination.write_text(json.dumps(dict(scope='Native scheduling events bounded by measured wave. Last chunk admission is not GPU completion; prefix length is progress, not API cache hit. Only first NEW admitted=True establishes long admission.',groups=result),indent=2)+'\n')
print('audited groups',len(result))
