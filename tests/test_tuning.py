import math
import pytest

from inferx.tuning import summarize_response


def completion(tokens=100, cached=0, reason='stop'):
    return [
        {'choices': [{'delta': {'content': 'hello '}, 'finish_reason': None}]},
        {'choices': [{'delta': {'content': 'world'}, 'finish_reason': reason}],
         'usage': {'prompt_tokens': 200, 'completion_tokens': tokens,
                   'prompt_tokens_details': {'cached_tokens': cached}}},
    ]


def test_throughput_uses_usage_not_chunks():
    result = summarize_response(completion(), 5., 1., 'cold')
    assert result['ok']
    assert result['e2e_tps'] == 20.
    assert result['observed_decode_tps'] == 24.75  # (100 - first token) / 4 s
    assert result['text'] == 'hello world'


@pytest.mark.parametrize('events', [completion()[:1], completion() + [{'error': 'failed'}]])
def test_incomplete_or_error_stream_is_not_success(events):
    result = summarize_response(events, 5., 1., 'cold')
    assert not result['ok']
    assert result['e2e_tps'] is None


def test_missing_usage_is_not_inferred_from_chunks():
    events = completion()
    del events[-1]['usage']
    result = summarize_response(events, 5., 1., 'cold')
    assert not result['ok']
    assert result['completion_tokens'] is None


@pytest.mark.parametrize('wall', [0., -1., math.inf, math.nan])
def test_bad_timing_cannot_win(wall):
    assert not summarize_response(completion(), wall, 1., 'cold')['ok']


def test_cached_tokens_cannot_be_labeled_cold():
    assert not summarize_response(completion(cached=123), 5., 1., 'cold')['ok']
    assert summarize_response(completion(cached=123), 5., 1., 'warm')['ok']


def test_empty_output_and_length_finish_are_visible():
    assert not summarize_response(completion(tokens=0), 5., None, 'cold')['ok']
    result = summarize_response(completion(reason='length'), 5., 1., 'cold')
    assert result['ok']  # performance observation, not a quality pass
    assert result['truncated']


def test_role_only_chunk_and_separate_usage_event():
    events = completion()
    usage = events[-1].pop('usage')
    events.insert(0, {'choices': [{'delta': {'role': 'assistant'}, 'finish_reason': None}]})
    events.append({'choices': [], 'usage': usage})
    result = summarize_response(events, 5., 1., 'warm')
    assert result['ok'] and result['completion_tokens'] == 100
