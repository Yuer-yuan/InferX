"""Small helpers for owned-process checks and deterministic context probes."""
import hashlib
import os
from pathlib import Path


def process_start_ticks(pid: int) -> str:
    return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]


def verify_process_identity(pid: int, start_ticks: str, executable: Path) -> bool:
    if pid <= 0:
        return False
    try:
        proc = Path(f'/proc/{pid}')
        return (proc.stat().st_uid == os.getuid()
                and process_start_ticks(pid) == start_ticks
                and (proc / 'exe').resolve(strict=True) == executable.resolve(strict=True))
    except (OSError, IndexError, ValueError):
        return False


def max_input_tokens(context: int, output_reserve: int = 8192) -> int:
    if output_reserve < 1 or context <= output_reserve:
        raise ValueError('context must exceed the positive output reserve')
    return context - output_reserve


def lookup_answers(records: int) -> dict[str, str]:
    if records < 3:
        raise ValueError('at least three records required')
    return {label: hashlib.sha256(f'inferx-registry-v1:{i}'.encode()).hexdigest()[:16]
            for label, i in [('early', 0), ('middle', records // 2), ('late', records - 1)]}


def long_context_messages(records: int) -> list[dict]:
    lookup_answers(records)
    lines = ['Read this configuration registry and retrieve the exact secret values.']
    for i in range(records):
        value = hashlib.sha256(f'inferx-registry-v1:{i}'.encode()).hexdigest()[:16]
        lines.append(f'Record R{i:05d}: secret={value}; component=service_{i % 1009:04d}; '
                     f'timeout_ms={100 + i * 13 % 5000}; retries={i % 7}.')
    lines.append(f'Return only a JSON object with keys early, middle, late containing the '
                 f'secret values of records R00000, R{records // 2:05d}, R{records - 1:05d}, '
                 'respectively. No explanation or code fence.')
    return [{'role': 'system', 'content': 'You are a precise assistant. Answer directly.'},
            {'role': 'user', 'content': '\n'.join(lines)}]


def agent_configs(context: int) -> dict:
    """Describe only the local model, including auxiliary/compaction routing."""
    max_input_tokens(context)
    if context < 32768:
        raise ValueError('agent window must be at least32768 tokens')
    model = 'qwen38-27b'
    opencode = {
        '$schema': 'https://opencode.ai/config.json',
        'model': f'inferx/{model}', 'small_model': f'inferx/{model}',
        'enabled_providers': ['inferx'], 'autoupdate': False,
        'lsp': False, 'formatter': False, 'plugin': [], 'mcp': {},
        'provider': {'inferx': {'npm': '@ai-sdk/openai-compatible', 'name': 'InferX local',
            'options': {'baseURL': 'http://127.0.0.1:18081/v1', 'apiKey': 'inferx-local'},
            'models': {model: {'name': 'Qwen3.8-27B NInfer', 'tool_call': True,
                'reasoning': False, 'attachment': False,
                'limit': {'context': context, 'output': 8192}}}}},
        'compaction': {'auto': True, 'prune': True, 'reserved': 16384},
        'agent': {'compaction': {'model': f'inferx/{model}'}},
    }
    pi = {'providers': {'inferx': {'baseUrl': 'http://127.0.0.1:18081/v1',
        'api': 'openai-completions', 'apiKey': 'inferx-local',
        'models': [{'id': model, 'name': 'Qwen3.8-27B NInfer', 'reasoning': False,
                    'input': ['text'], 'contextWindow': context, 'maxTokens': 8192,
                    'cost': {'input': 0, 'output': 0, 'cacheRead': 0, 'cacheWrite': 0},
                    'samplingParams': {'temperature': 0, 'presence_penalty': 0,
                                       'frequency_penalty': 0}}]}}}
    env = {key: model for key in ['ANTHROPIC_MODEL', 'ANTHROPIC_DEFAULT_SONNET_MODEL',
        'ANTHROPIC_DEFAULT_OPUS_MODEL', 'ANTHROPIC_DEFAULT_HAIKU_MODEL',
        'ANTHROPIC_SMALL_FAST_MODEL', 'CLAUDE_CODE_SUBAGENT_MODEL']}
    env.update(ANTHROPIC_BASE_URL='http://127.0.0.1:18081',
        ANTHROPIC_API_KEY='inferx-local', CLAUDE_CODE_MAX_OUTPUT_TOKENS='8192',
        CLAUDE_CODE_AUTO_COMPACT_WINDOW=str(max(100000, context - 16384)),
        CLAUDE_AUTOCOMPACT_PCT_OVERRIDE='80', CLAUDE_CODE_ATTRIBUTION_HEADER='0',
        CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC='1', DISABLE_AUTOUPDATER='1',
        DISABLE_TELEMETRY='1', DISABLE_ERROR_REPORTING='1')
    # The installed Claude accepts compact windows >=100K; smaller fallback uses percentage.
    if context < 116384:
        env.pop('CLAUDE_CODE_AUTO_COMPACT_WINDOW')
        env['CLAUDE_AUTOCOMPACT_PCT_OVERRIDE'] = '25'
    claude = {'env': env, 'alwaysThinkingEnabled': False}
    return {'opencode': opencode, 'pi': pi, 'claude': claude}
