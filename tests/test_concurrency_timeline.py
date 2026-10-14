import sqlite3

import pytest

from scripts.analyze_concurrency import summarize_cuda_timeline


def test_cuda_timeline_uses_interval_union_and_does_not_call_copies_idle(tmp_path):
    path=tmp_path/'trace.sqlite'
    connection=sqlite3.connect(path)
    connection.execute('CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER,end INTEGER,deviceId INTEGER)')
    connection.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?)',[(10,20,0),(15,25,0),(30,40,0)])
    connection.commit();connection.close()
    result=summarize_cuda_timeline(path)
    assert result['kernel_duration_sum_ns']==35
    assert result['kernel_union_ns']==25 and result['kernel_envelope_ns']==30
    assert result['no_kernel_fraction']==pytest.approx(1/6)
    assert result['kernel_count']==3
    assert 'copies' in result['limitation']
    assert sqlite3.connect(path).execute('SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL').fetchone()[0]==3


def test_cuda_timeline_keeps_empty_or_invalid_capture_explicit(tmp_path):
    path=tmp_path/'trace.sqlite'
    connection=sqlite3.connect(path)
    connection.execute('CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER,end INTEGER,deviceId INTEGER)')
    connection.commit();connection.close()
    assert summarize_cuda_timeline(path)['status']=='no_kernels'
    connection=sqlite3.connect(path)
    connection.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (20,10,0)');connection.commit();connection.close()
    with pytest.raises(ValueError,match='invalid kernel interval'):
        summarize_cuda_timeline(path)
