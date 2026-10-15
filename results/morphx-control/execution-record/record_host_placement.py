"""Read-only topology evidence; does not reserve CPUs or change other processes."""
import hashlib,json,os,subprocess,time
from pathlib import Path
r=Path(os.environ['INFERX_ROOT']);raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,pci.bus_id','--format=csv,noheader'],text=True)
gpus=[]
def cpus(text):
 out=set()
 for term in text.strip().split(','):
  ends=[int(x) for x in term.split('-')];out.update(range(ends[0],ends[-1]+1))
 return out
for line in raw.splitlines():
 index,uuid,pci=[x.strip() for x in line.split(',')];domain,bus,slot=pci.lower().split(':');address=f'{int(domain,16):04x}:{bus}:{slot}'
 node=int((Path('/sys/bus/pci/devices')/address/'numa_node').read_text());assert node>=0
 local=cpus((Path('/sys/devices/system/node')/f'node{node}/cpulist').read_text());gpus.append(dict(index=int(index),uuid=uuid,pci=address,numa=node,local_cpus=sorted(local)))
selected=next(x for x in gpus if x['uuid']=='GPU-4d9bef9d-631c-bb36-3ff4-589d6a184189');agent=36;server=set(range(28,36));client={40,41,42,43,96,97,98,99};siblings=cpus(Path(f'/sys/devices/system/cpu/cpu{agent}/topology/thread_siblings_list').read_text())
assert agent in selected['local_cpus'] and not (siblings & (server|client))
report=dict(recorded_unix=time.time(),gpus=gpus,agent_cpu=agent,agent_siblings=sorted(siblings),server_cpus=sorted(server),client_cpus=sorted(client),
 source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='NUMA-local and disjoint experiment CPU placement, not whole-host exclusive core reservation. Actual agent binding is separately logged by each owned backend.')
(r/'morphx-control/host-placement.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
