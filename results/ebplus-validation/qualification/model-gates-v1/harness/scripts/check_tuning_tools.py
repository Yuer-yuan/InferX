"""Protocol smoke probe, with no filesystem or shell tool execution."""
import argparse
import json
import time
import urllib.request
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--endpoint',default='http://127.0.0.1:18081')
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()

def post(route,payload):
    req=urllib.request.Request(args.endpoint+route,json.dumps(payload).encode(),
        {'Content-Type':'application/json','anthropic-version':'2023-06-01'})
    start=time.perf_counter()
    with urllib.request.urlopen(req,timeout=120) as response:
        return json.load(response),time.perf_counter()-start

tool=dict(name='add_numbers',description='Add two integers.',input_schema=dict(
    type='object',properties={'a':{'type':'integer'},'b':{'type':'integer'}},
    required=['a','b'],additionalProperties=False))
messages=[dict(role='user',content='Use add_numbers to calculate 2 + 3. After the tool result, answer with only the number.')]
payload=dict(model='qwen38-27b',max_tokens=256,temperature=0,messages=messages,
             tools=[tool],tool_choice={'type':'tool','name':'add_numbers'})
first,t1=post('/v1/messages',payload)
calls=[x for x in first.get('content',[]) if x.get('type')=='tool_use']
assert len(calls)==1 and calls[0]['name']=='add_numbers', first
call=calls[0]
assert call['input']=={'a':2,'b':3},call
messages += [dict(role='assistant',content=first['content']),
             dict(role='user',content=[dict(type='tool_result',tool_use_id=call['id'],content='5')])]
payload['tool_choice']={'type':'auto'}
second,t2=post('/v1/messages',payload)
answer=''.join(x.get('text','') for x in second.get('content',[]) if x.get('type')=='text')
passed=answer.strip().strip('`* .\n')=='5' and second.get('stop_reason')=='end_turn'
args.output.write_text(json.dumps(dict(passed=passed,first=first,second=second,
                                     first_wall_s=t1,second_wall_s=t2),indent=2))
assert passed,second
print('Anthropic tool roundtrip passed',round(t1+t2,3),'seconds')
