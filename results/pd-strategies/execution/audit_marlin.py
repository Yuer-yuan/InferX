import hashlib,json
from pathlib import Path
r=Path('/home/guokc/mypro/gnn-mount-sda1/InferX');s=r/'pd-baselines/sglang0519-venv/lib/python3.12/site-packages/sglang';e=[]
for relative,needles in [
 ('srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py',['self.workspace = marlin_make_workspace','workspace=self.workspace,']),
 ('srt/layers/quantization/marlin_utils.py',['def marlin_make_workspace','def apply_gptq_marlin_linear']),
 ('kernels/jit/csrc/gemm/marlin/gptq_marlin.cuh',['int* locks = (int*)workspace','cudaDevAttrMultiProcessorCount']),
 ('kernels/jit/csrc/gemm/marlin/marlin_template.h',['inline void barrier_acquire','barrier_acquire(&locks[locks_off]','barrier_release(&locks[locks_off]'])]:
 p=s/relative;b=p.read_bytes();ls=b.decode().splitlines();snippets=[]
 for needle in needles:
  found=[i for i,l in enumerate(ls) if needle in l];assert found,(relative,needle)
  for i in found:snippets.append(dict(needle=needle,line=i+1,context='\n'.join(ls[max(0,i-2):i+9])))
 e.append(dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),snippets=snippets))
out=r/'pd-strategies/compatibility/marlin-concurrency-audit.json';out.write_text(json.dumps(dict(finding='One persistent Marlin workspace per quantization scheme is passed unchanged to both P/D streams; underlying kernel uses it as mutable inter-CTA barrier locks. Independent per-stream scratch is required for overlapping calls.',causal_limit='Static concurrency hazard. Not a controlled reproduction of the original8192 mixed stall; cleanl16 already completed one cold wave without isolation. No root-cause or numerical-equivalence claim from this audit.',physical_sm_count_limit='Kernel also queries physical multiprocessor count. Whether its grid/barrier scheduling is safe under every Green Context partition is not established by this source audit.',files=e),indent=2)+'\n');print(out)
