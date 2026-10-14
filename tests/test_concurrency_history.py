from inferx.concurrency import tool_resume_history


def test_resume_keeps_the_entire_seed_query_and_tool_result_without_mutating_it():
    messages=[dict(role='system',content='system'),dict(role='user',content='24K registry'),
              dict(role='assistant',content='Registry loaded.'),dict(role='user',content='record status')]
    expected=dict(name='record_status',arguments=dict(status='ready',task='status-probe',attempt=3))
    history=tool_resume_history(messages,expected)
    assert history[:4]==messages
    assert len(messages)==4 and len(history)==6
    call=history[4]['tool_calls'][0]
    assert call['function']['name']==expected['name']
    import json
    assert json.loads(call['function']['arguments'])==expected['arguments']
    assert history[5]['role']=='tool' and history[5]['tool_call_id']==call['id']
    history[0]['content']='mutated'
    assert messages[0]['content']=='system'
