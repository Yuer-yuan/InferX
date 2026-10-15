"""Generate capacity/retrieval probes, using the preserved server's CPU tokenizer."""
import json
import os
import urllib.request
from pathlib import Path
from inferx.context_agents import long_context_messages, lookup_answers, max_input_tokens

root = Path(os.environ['INFERX_ROOT'])
out = root / 'context-agents'
out.mkdir(parents=True, exist_ok=True)


def post(route, body):
    request = urllib.request.Request('http://127.0.0.1:18080' + route,
        json.dumps(body).encode(), {'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def count(records):
    rendered = post('/apply-template', {'messages': long_context_messages(records),
                         'chat_template_kwargs': {'enable_thinking': False}})['prompt']
    return len(post('/tokenize', {'content': rendered, 'add_special': False})['tokens'])


for capacity in [131072, 262144]:
    limit = max_input_tokens(capacity)
    low, high = 3, 10000
    while low < high:
        mid = (low + high + 1) // 2
        if count(mid) <= limit:
            low = mid
        else:
            high = mid - 1
    actual = count(low)
    assert actual <= limit
    corpus = [dict(id=f'lookup_{capacity}', messages=long_context_messages(low),
                   max_tokens=128, expected=lookup_answers(low), prompt_tokens_llama=actual,
                   records=low, context=capacity, output_reserve=8192)]
    path = out / f'lookup-{capacity}.json'
    with path.open('x') as f:
        json.dump(corpus, f, indent=2)
    print(path.name, low, actual, flush=True)
