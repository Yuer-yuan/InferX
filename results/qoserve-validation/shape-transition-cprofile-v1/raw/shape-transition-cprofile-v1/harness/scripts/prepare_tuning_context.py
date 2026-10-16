import json,os,urllib.request
from pathlib import Path
root=Path(os.environ['INFERX_ROOT'])

def post(route,data):
 req=urllib.request.Request('http://127.0.0.1:18080'+route,json.dumps(data).encode(),{'Content-Type':'application/json'})
 return json.load(urllib.request.urlopen(req,timeout=60))

def messages(lines,kind):
 user='Project configuration registry. Use it as reference.\n'
 for i in range(lines):
  user+=f'Record R{i:05d}: component=service_{(i*73)%1009:04d}; timeout_ms={100+i*13%5000}; retries={i%7}; owner=team_{i%31}; enabled={str(i%3!=0).lower()}.\n'
 user+='The audit marker is CYAN-ORBIT-7319.\n'
 if kind=='fit':
  user+='Return only the audit marker verbatim, with no explanation.'
 else:
  user+='Implement Python parse_registry(text) that parses the Record lines above into dicts with typed fields, ignoring malformed lines. Return only a complete Python code block, including re import and a docstring. Do not include tests.'
 return [dict(role='system',content='You are a helpful coding assistant. Answer directly.'),dict(role='user',content=user)]

def count(msg):
 rendered=post('/apply-template',dict(messages=msg,chat_template_kwargs={'enable_thinking':False}))['prompt']
 return len(post('/tokenize',dict(content=rendered,add_special=False))['tokens'])
for name,targets in [('context',[2048,8192]),('fit',[32768,64000])]:
 rows=[]
 for target in targets:
  lo,hi=1,2000
  while lo<hi:
   mid=(lo+hi+1)//2
   if count(messages(mid,name))<=target:lo=mid
   else:hi=mid-1
  msg=messages(lo,name); actual=count(msg)
  rows.append(dict(id=f'{name}_{target}',messages=msg,max_tokens=32 if name=='fit' else 512,prompt_tokens_llama=actual))
  print(name,target,lo,actual,flush=True)
 (root/f'tuning/{name}.json').write_text(json.dumps(rows,indent=2))
