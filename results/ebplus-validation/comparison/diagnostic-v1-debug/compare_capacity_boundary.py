import argparse,ast,copy,hashlib,json
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument('--old',required=True,type=Path);parser.add_argument('--new',required=True,type=Path)
args=parser.parse_args();old=args.old;new=args.new
def methods(folder,cls):
 tree=ast.parse((folder/'inferx_ebplus_scheduler.py').read_text())
 c=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
 return {n.name:n for n in c.body if isinstance(n,ast.FunctionDef)}
def dump(node):return ast.dump(node,include_attributes=False)
assert (old/'inferx_ebplus_policy.py').read_bytes()==(new/'inferx_ebplus_policy.py').read_bytes()
om=methods(old,'EBPlusScheduler');nm=methods(new,'EBPlusScheduler')
phase=copy.deepcopy(nm['_handle_phase_transition'])
phase.body=[n for n in phase.body if not (
 (isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='native_admission_pressure') or
 (isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='kv_cache_full' and isinstance(n.value,ast.BoolOp)) or
 (isinstance(n,ast.If) and ast.unparse(n.test).startswith('native_admission_pressure')))]
ptree=ast.parse((new/'inferx_ebplus_policy.py').read_text());pc=next(n for n in ptree.body if isinstance(n,ast.ClassDef) and n.name=='AuthorPolicy')
p=next(n for n in pc.body if isinstance(n,ast.FunctionDef) and n.name=='_handle_phase_transition')
assert dump(phase)==dump(p)
class RemoveNote(ast.NodeTransformer):
 def visit_Expr(self,node):
  if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='_ebplus_note_admission_block':return None
  return self.generic_visit(node)
assert dump(RemoveNote().visit(copy.deepcopy(nm['_schedule_exclusive_native'])))==dump(om['_schedule_exclusive_native'])
unchanged=[name for name in om if name in nm and dump(om[name])==dump(nm[name])]
assert dump(methods(old,'ObservedNativeScheduler')['__init__'])==dump(methods(new,'ObservedNativeScheduler')['__init__'])
assert all(name in unchanged for name in ['_schedule_default','_update_after_schedule','_preempt_request','_preempt_chunk_prefilling','_free_request'])
a=json.loads((old/'manifest.json').read_text());b=json.loads((new/'manifest.json').read_text());assert a['extracted_method_sha256']==b['extracted_method_sha256']
result=dict(author_policy_bytes_identical=True,original_author_method_hashes_identical=len(a['extracted_method_sha256']),
    phase_branches_identical_except_native_pressure_predicate_and_counter=True,
    native_exclusive_schedule_identical_except_real_allocation_refusal_hook=True,
    observed_native_control_identical=True,unchanged_bridge_methods=unchanged,
    changed_bridge_methods=sorted(set(om)-set(unchanged)),added_bridge_methods=sorted(set(nm)-set(om)),
    old_module_sha256=hashlib.sha256((old/'inferx_ebplus_scheduler.py').read_bytes()).hexdigest(),
    new_module_sha256=hashlib.sha256((new/'inferx_ebplus_scheduler.py').read_bytes()).hexdigest(),
    limitation='Engine-bound admission-pressure semantics explicitly extended. No arbitrary-program equivalence or GPU correctness/performance proof.')
print(json.dumps(result,indent=2))
