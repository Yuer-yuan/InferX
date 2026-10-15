"""Freeze all candidate settings before diagnostics, only from qualified gates."""
import copy,hashlib,json,os
from pathlib import Path
from inferx.morphx_control import load_qualified_base_case
r=Path(os.environ['INFERX_ROOT']);root=r/'morphx-control';audit_path=root/'model-audit.json';audit=json.loads(audit_path.read_text());assert len(audit['rows'])==4 and all(x['status']=='passed' for x in audit['rows'])
source=r/'sota-validation/configs/sota-validation.json';original=json.loads(source.read_text());base=load_qualified_base_case(r)
cases=[copy.deepcopy(next(c for c in original['cases'] if c['id']=='compiled-primary-control'))]
libraries={'abi-only':r/'sota-validation/builds/libcuda_query_abi_v4.so',**{n:root/f'builds/libcuda_{n}_v1.so' for n in ['fill','event','both']}}
for gate in audit['rows']:
 variant=gate['variant'];library=libraries[variant];assert hashlib.sha256(library.read_bytes()).hexdigest()==gate['library_sha256']
 for split,mode,label in [(60,0,'60-authorcap'),(96,0,'96-authorcap'),(96,16,'96-ablation')]:
  case=copy.deepcopy(base);case['id']='morphx-control-'+variant+'-'+label
  case['morphx'].update(patched_launches=gate['patched_launches'],library_sha256=gate['library_sha256'],qualification_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),scope='Real full-model qualified transport port; partial native coverage, not original-vLLM/GDRCopy ceiling.')
  case['environment'].update(LD_PRELOAD=str(library)+':'+str(r/'pd-baselines/builds/libpreload.so'),NEUTRINO_HOOK_DRIVER=str(library),SMSCHED_SPLIT_HINT=str(split),SMSCHED_LLM_TRICK_MODE=str(mode),SMSCHED_LLM_TRACE_LIMIT='2048')
  case['environment'].pop('SMSCHED_CONTROL_TRACE',None);case['environment']['INFERX_MORPHX_IR']=str(root/'ir'/case['id']);cases.append(case)
for case in cases:case['environment']['INFERX_MORPHX_IR']=str(root/'ir'/case['id'])
output=dict(scope='New control transport and policy diagnostics; actualauthorcompiler/profiler/schedule_2, modified transport/SGLang orchestration. No hidden unqualified lanes.',
 selection='First cold diagnostic at96/mode16 for ABI-only/fill/event/both plus compiler control. Select jointly short/long/throughput. Then60/mode0,96/mode0,96/mode16 within selected transport; at most3policy settings. Freeze before3freshmixedrepeats and ownserialC4solos. Unrun cases have no score.',
 source_sha256={str(x.relative_to(r)):hashlib.sha256(x.read_bytes()).hexdigest() for x in [source,audit_path]},cases=cases)
p=root/'configs/morphx-control.json';p.parent.mkdir(exist_ok=True)
with p.open('x') as f:json.dump(output,f,indent=2)
print(hashlib.sha256(p.read_bytes()).hexdigest(),len(cases))
