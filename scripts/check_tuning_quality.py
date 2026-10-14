import asyncio,json,os,sys
from pathlib import Path
from inferx.verifier import verify
from inferx.llama_client import extract_code
root=Path(os.environ['INFERX_ROOT'])
checks={
'code_merge':('merge_intervals','''def check(f):
    assert f([])==[]
    data=[(6,3),(2,1),(2,3),(10,10)]
    assert [tuple(x) for x in f(data)]==[(1,6),(10,10)]
    assert data==[(6,3),(2,1),(2,3),(10,10)]
    assert [tuple(x) for x in f([(1,1),(1,1)])]==[(1,1)]
'''),
'code_lru':('LRUCache','''def check(C):
    c=C(2);c.put(1,1);c.put(2,2);assert c.get(1)==1
    c.put(3,3);assert c.get(2)==-1;c.put(1,4);assert c.get(1)==4
    c.put(4,4);assert c.get(3)==-1;assert c.get(4)==4
    c=C(0);c.put(1,1);assert c.get(1)==-1
    try:C(-1)
    except ValueError:pass
    else:raise AssertionError('negative capacity accepted')
'''),
'code_toposort':('topological_sort','''def check(f):
    assert f({})==[]
    g={'b':['d','d'],'a':['d'],'z':[]}
    assert f(g)==['a','b','d','z'];assert g=={'b':['d','d'],'a':['d'],'z':[]}
    assert f({'a':['c'],'b':[]})==['a','b','c']
    for g in [{'a':['a']},{'a':['b'],'b':['a']}]:
        try:f(g)
        except ValueError:pass
        else:raise AssertionError('cycle accepted')
''')}
async def main():
 rows=[]
 for p in sorted((root/'tuning').glob(sys.argv[1])):
  for line in p.read_text().splitlines():
   j=json.loads(line)
   if j.get('kind')!='trial' or j['prompt_id'] not in checks:continue
   entry,tests=checks[j['prompt_id']]
   if not j['summary']['ok']:res=dict(status='infrastructure_error',error=j['summary']['errors'])
   elif j['summary']['truncated']:res=dict(status='truncated')
   else:res=await verify(extract_code(j['summary']['text']),tests,entry,3,None)
   rows.append(dict(file=p.name,config=j['config'],prompt=j['prompt_id'],repetition=j['repetition'],**res))
 print(json.dumps(rows,indent=2))
asyncio.run(main())
