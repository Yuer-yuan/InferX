"""Positive overlap evidence on GPU kernel streams identified by GDN kernels."""
import gzip,hashlib,json
from pathlib import Path
r=Path('/home/guokc/mypro/gnn-mount-sda1/InferX');run=r/'pd-baselines/raw/pd-strategies-pdmux-profile-v1';p=run/'traces/pdmux-pdmux-mixed-TP-0.trace.json.gz';raw=p.read_bytes();es=json.loads(gzip.decompress(raw))['traceEvents'];kernels=[x for x in es if x.get('cat')=='kernel' and x.get('ph')=='X' and x.get('dur',0)>0]
def key(x):
 a=x['args'];return (a['device'],a['context'],a['stream'])
pmark=[x for x in kernels if x['name']=='chunk_gated_delta_rule_fwd_kernel_h_blockdim64'];dmark=[x for x in kernels if x['name']=='fused_recurrent_gated_delta_rule_packed_decode_kernel'];ps={key(x) for x in pmark};ds={key(x) for x in dmark};assert ps and ds and not ps.intersection(ds),(ps,ds)
prefill=[x for x in kernels if key(x) in ps];decode=[x for x in kernels if key(x) in ds]
def union(events):
 out=[]
 for s,e in sorted((x['ts'],x['ts']+x['dur']) for x in events):
  if out and s<=out[-1][1]:out[-1][1]=max(out[-1][1],e)
  else:out.append([s,e])
 return out
pi,di=union(prefill),union(decode);i=j=0;overlap=[]
while i<len(pi) and j<len(di):
 s=max(pi[i][0],di[j][0]);e=min(pi[i][1],di[j][1])
 if e>s:overlap.append([s,e])
 if pi[i][1]<di[j][1]:i+=1
 else:j+=1
assert overlap,'No temporal overlap observed on classified streams'
span=max(x['ts']+x['dur'] for x in kernels)-min(x['ts'] for x in kernels)
result=dict(status='observed',scope='Separate profiler-only wave; GDN chunk-prefill and recurrent-decode kernel names identify distinct(device,context,stream)sets. Temporal overlap of kernels on those streams is positive GPU concurrency evidence, not a bandwidth counter, universal progress guarantee, or performance measurement.',trace=str(p.relative_to(r)),trace_sha256=hashlib.sha256(raw).hexdigest(),kernel_events=len(kernels),prefill_marker_count=len(pmark),decode_marker_count=len(dmark),prefill_stream_keys=sorted(ps),decode_stream_keys=sorted(ds),prefill_stream_kernel_events=len(prefill),decode_stream_kernel_events=len(decode),kernel_trace_span_s=span/1e6,prefill_kernel_union_s=sum(e-s for s,e in pi)/1e6,decode_kernel_union_s=sum(e-s for s,e in di)/1e6,overlapping_kernel_union_s=sum(e-s for s,e in overlap)/1e6,overlap_intervals=len(overlap),example_overlap_timestamp_us=overlap[:3],profiling_excluded_from_formal_statistics=True)
(r/'pd-strategies/gpu-overlap-evidence.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
