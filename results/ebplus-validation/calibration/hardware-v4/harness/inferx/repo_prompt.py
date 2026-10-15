import hashlib

from inferx.repository import source_context


def head_lines(text, limit):
    if len(text) <= limit:
        return text
    return text[:limit].rsplit('\n', 1)[0] + '\n' if limit else ''


def tail_lines(text, limit):
    if len(text) <= limit:
        return text
    tail = text[-limit:] if limit else ''
    return tail.split('\n', 1)[1] if '\n' in tail else ''


async def build_prompt(client, task, max_prompt_tokens=6000):
    prefix_limit, suffix_limit, context_limit = 8000, 8000, 6000
    while True:
        prefix = tail_lines(task['prefix_code'], prefix_limit)
        suffix = head_lines(task['suffix_code'], suffix_limit)
        context = source_context(task, context_limit) if context_limit else ''
        instruction = (
            'Fill the missing Python block in the repository source file below. '
            'The resulting source is PREFIX + your code + SUFFIX, with no added characters. '
            'Return only a JSON object with one string field "code" containing the exact missing text. '
            'Preserve indentation and newlines, including indentation on the FIRST line. '
            'The prefix ends at the start of a line; your code must include the entire missing block. '
            f'The first line of your code should start with {len(task.get("infill_indent", ""))} spaces. '
            'Do not repeat the prefix or suffix. Do not include tests or explanations.\n'
            f'Repository: {task["repo_name"]}\nTarget: {task["file_name"]}\n'
            f'<RELATED_SOURCE>\n{context}\n</RELATED_SOURCE>\n'
            f'<PREFIX>\n{prefix}</PREFIX>\n<SUFFIX>{suffix}\n</SUFFIX>')
        rendered = await client.request_json('POST', '/apply-template', dict(
            messages=[dict(role='system', content='You are a careful Python repository code completion assistant.'),
                      dict(role='user', content=instruction)],
            chat_template_kwargs={'enable_thinking': False}))
        prompt = rendered['prompt']
        tokens = await client.request_json('POST', '/tokenize', {'content': prompt})
        n = len(tokens['tokens'])
        if n <= max_prompt_tokens:
            return prompt, dict(tokens=n, sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                                prefix_chars=len(prefix), suffix_chars=len(suffix), context_chars=len(context))
        if context_limit:
            context_limit = context_limit // 2 if context_limit >= 256 else 0
        elif prefix_limit or suffix_limit:
            prefix_limit = int(prefix_limit * .75) if prefix_limit >= 128 else 0
            suffix_limit = int(suffix_limit * .75) if suffix_limit >= 128 else 0
        else:
            raise ValueError('chat template and instructions exceed prompt budget')
