"""CPU compilation exercises the patched real dispatch bodies, without CUDA."""
import hashlib
from pathlib import Path
import shutil
import subprocess

import pytest

FIXTURES = Path(__file__).parent/'fixtures'


def original():
    return (FIXTURES/'morphx_launch_original.cpp').read_text()


def test_extended_and_ptsz_dispatch_preserve_config_and_native_semantics(tmp_path):
    from inferx.morphx_validation import patch_extended_launch
    changed = patch_extended_launch(original())
    source = (FIXTURES/'morphx_launch_shim.cpp').read_text().replace('// PATCHED_DISPATCH', changed)
    path=tmp_path/'dispatch.cpp';path.write_text(source)
    compiler=shutil.which('g++')
    assert compiler, 'C++ compiler required for dispatch behavior regression'
    subprocess.run([compiler,'-std=c++17','-Wall','-Wextra','-Werror',str(path),'-o',str(tmp_path/'probe')],check=True,capture_output=True,text=True)
    result=subprocess.run([str(tmp_path/'probe')],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'dispatch oracle passed' in result.stdout


def test_patch_rejects_unknown_or_already_patched_dispatch():
    from inferx.morphx_validation import patch_extended_launch
    with pytest.raises(ValueError,match='launch source'):
        patch_extended_launch('unrelated source')
    with pytest.raises(ValueError,match='launch source'):
        patch_extended_launch(patch_extended_launch(original()))


def source_tree(tmp_path):
    root=tmp_path/'upstream';(root/'runtime/src').mkdir(parents=True);(root/'include').mkdir()
    (root/'runtime/src/cuda-llm.cpp').write_text(original())
    (root/'runtime/src/preload.c').write_text('unchanged preload')
    (root/'include/common.h').write_text('unchanged transport')
    return root


def test_preparation_is_fresh_hashed_and_leaves_upstream_unchanged(tmp_path):
    from inferx.morphx_validation import prepare_runtime
    src=source_tree(tmp_path);before=(src/'runtime/src/cuda-llm.cpp').read_bytes();sha=hashlib.sha256(before).hexdigest()
    dest=tmp_path/'isolated';manifest=prepare_runtime(src,dest,sha)
    assert (src/'runtime/src/cuda-llm.cpp').read_bytes()==before
    assert (dest/'runtime/src/preload.c').read_text()=='unchanged preload'
    assert manifest['before_sha256']==sha and manifest['after_sha256']!=sha
    assert manifest['attributed_launches']=='native; preserve original config'
    assert manifest['ptsz_launches']=='native; preserve per-thread stream semantics'
    assert manifest['normal_unattributed_launches']=='existing MorphX interception'
    for name,expected in manifest['source_sha256'].items():
        assert hashlib.sha256((dest/name).read_bytes()).hexdigest()==expected
    with pytest.raises(FileExistsError):prepare_runtime(src,dest,sha)
    assert (src/'runtime/src/cuda-llm.cpp').read_bytes()==before


def test_preparation_refuses_unpinned_source_before_creating_destination(tmp_path):
    from inferx.morphx_validation import prepare_runtime
    src=source_tree(tmp_path);dest=tmp_path/'isolated'
    with pytest.raises(ValueError,match='source hash'):prepare_runtime(src,dest,'0'*64)
    assert not dest.exists()
    with pytest.raises(ValueError,match='overlap'):
        prepare_runtime(src,src/'child',hashlib.sha256(original().encode()).hexdigest())
