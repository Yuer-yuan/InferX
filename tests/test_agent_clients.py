import pytest
from inferx.context_agents import agent_configs


def test_all_client_budgets_and_models_match_local_runtime():
    configs = agent_configs(131072)
    oc = configs['opencode']
    assert oc['model'] == oc['small_model'] == 'inferx/qwen38-27b'
    assert oc['enabled_providers'] == ['inferx']
    assert oc['provider']['inferx']['options']['baseURL'] == 'http://127.0.0.1:18081/v1'
    assert oc['provider']['inferx']['models']['qwen38-27b']['limit'] == {'context': 131072, 'output': 8192}
    pi = configs['pi']['providers']['inferx']
    assert pi['baseUrl'] == 'http://127.0.0.1:18081/v1'
    assert pi['models'][0]['contextWindow'] == 131072
    assert pi['models'][0]['maxTokens'] == 8192
    env = configs['claude']['env']
    for key in ['ANTHROPIC_MODEL', 'ANTHROPIC_DEFAULT_SONNET_MODEL',
                'ANTHROPIC_DEFAULT_OPUS_MODEL', 'ANTHROPIC_DEFAULT_HAIKU_MODEL',
                'ANTHROPIC_SMALL_FAST_MODEL', 'CLAUDE_CODE_SUBAGENT_MODEL']:
        assert env[key] == 'qwen38-27b'
    assert env['ANTHROPIC_BASE_URL'] == 'http://127.0.0.1:18081'
    assert int(env['CLAUDE_CODE_AUTO_COMPACT_WINDOW']) <= 131072 - 8192


def test_client_config_rejects_insufficient_window():
    with pytest.raises(ValueError):
        agent_configs(8192)
