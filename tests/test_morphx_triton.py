"""The bridge preserves work and original entry, rejecting ambiguous modules."""
import pytest

IR='''source_filename = "LLVMDialectModule"
target datalayout = "e-p:64:64"
define ptx_kernel void @oracle(ptr captures(none) %x) {
  %v = or disjoint i32 1, 2
  ret void
}
declare noundef range(i32 0, 1024) i32 @tid()
'''


def test_ir_bridge_weakens_only_known_optimization_promises():
    from inferx.morphx_triton import llvm18_compatible_ir
    text,changes=llvm18_compatible_ir(IR)
    assert 'target triple = "nvptx64-nvidia-cuda"' in text
    assert 'ptr nocapture %x' in text and '%v = or i32 1, 2' in text
    assert 'declare noundef i32 @tid()' in text
    assert changes=={'target_triple':1,'captures_none':1,'disjoint_or':1,'return_range':1,'gep_wrap_promises':0,'trunc_wrap_promises':0,'icmp_samesign':0,'nocreateundeforpoison':0}
    assert llvm18_compatible_ir(text)[0]==text
    for bad in [IR.replace('captures(none)','captures(address)'),IR+'target triple = "x86_64-pc-linux-gnu"\n']:
        with pytest.raises(ValueError):llvm18_compatible_ir(bad)


def test_ptx_keeps_original_and_only_its_morphable_entry():
    from inferx.morphx_triton import finalize_ptx
    ptx='.version 8.6\n.target sm_89, debug\n.visible .entry oracle_pk() {}\n.visible .entry oracle() {}\n'
    text,name=finalize_ptx(ptx,'oracle',86,89,False)
    assert name=='oracle' and 'oracle_pk()' in text and 'debug' not in text
    for bad in [ptx.replace('oracle_pk','other'),ptx.replace('oracle_pk','oracle'),ptx+' .visible .entry third() {}']:
        with pytest.raises(ValueError,match='entries'):finalize_ptx(bad,'oracle',86,89,False)


def test_compiler_cache_key_changes_with_pass_and_bridge(tmp_path):
    from inferx.morphx_triton import compilation_key
    plugin=tmp_path/'pass.so';plugin.write_bytes(b'one')
    a=compilation_key(plugin,'test-v1');plugin.write_bytes(b'two')
    assert a!=compilation_key(plugin,'test-v1')
    assert compilation_key(plugin,'test-v1')!=compilation_key(plugin,'test-v2')


def test_new_gep_wrap_promises_are_weakened_without_changing_address_work():
    from inferx.morphx_triton import llvm18_compatible_ir
    source=IR+'  %p = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %78\n  %q = getelementptr nusw i64, ptr %x, i32 %i\n'
    changed,counts=llvm18_compatible_ir(source)
    assert '%p = getelementptr inbounds i8, ptr addrspace(3) @global_smem, i32 %78' in changed
    assert '%q = getelementptr i64, ptr %x, i32 %i' in changed
    assert counts['gep_wrap_promises']==2


def test_truncation_wrap_promises_keep_original_integer_operation():
    from inferx.morphx_triton import llvm18_compatible_ir
    source=IR+'  %41 = trunc nuw nsw i64 %29 to i32, !dbg !37\n  %42 = trunc nsw i64 %x to i8\n'
    changed,counts=llvm18_compatible_ir(source)
    assert '%41 = trunc i64 %29 to i32, !dbg !37' in changed
    assert '%42 = trunc i64 %x to i8' in changed
    assert counts['trunc_wrap_promises']==3


def test_comparison_signedness_promise_keeps_predicate_and_operands():
    from inferx.morphx_triton import llvm18_compatible_ir
    changed,counts=llvm18_compatible_ir(IR+'  %82 = icmp samesign ult i32 %13, 40, !dbg !48\n')
    assert '%82 = icmp ult i32 %13, 40, !dbg !48' in changed
    assert counts['icmp_samesign']==1


def test_new_function_poison_promise_is_dropped_without_changing_body():
    from inferx.morphx_triton import llvm18_compatible_ir
    changed,counts=llvm18_compatible_ir(IR+'attributes #4 = { nocallback nocreateundeforpoison nofree nounwind memory(none) }\n')
    assert 'attributes #4 = { nocallback nofree nounwind memory(none) }' in changed
    assert counts['nocreateundeforpoison']==1 and '%v = or i32 1, 2' in changed
