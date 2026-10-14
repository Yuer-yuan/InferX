import unittest


class PromptTests(unittest.IsolatedAsyncioTestCase):
    async def test_bounded_prompt_preserves_gap_and_excludes_answer_and_tests(self):
        from inferx.repo_prompt import build_prompt
        class Client:
            async def request_json(self, method, path, payload):
                if path == '/apply-template':
                    return {'prompt': '\n'.join(x['content'] for x in payload['messages'])}
                return {'tokens': list(range(len(payload['content']) // 3))}
        row = dict(file_name='/repo/a.py', repo_name='repo',
            prefix_code='before\n' * 4000 + 'def f(x):\n    ',
            suffix_code='\n    return 0\n' + 'after\n' * 4000,
            middle_code='SECRET ANSWER', context_code=[
                ['/repo/a.py', 'SECRET ANSWER'], ['/repo/tests/test_a.py', 'SECRET TEST'],
                ['/repo/helper.py', 'helper\n' * 1000]])
        prompt, meta = await build_prompt(Client(), row, max_prompt_tokens=1600)
        self.assertLessEqual(meta['tokens'], 1600)
        self.assertIn('def f(x):\n    ', prompt)
        self.assertIn('\n    return 0\n', prompt)
        self.assertNotIn('SECRET', prompt)
        self.assertLess(meta['prefix_chars'], len(row['prefix_code']))
        self.assertEqual(len(meta['sha256']), 64)

    async def test_unfit_template_fails_instead_of_unbounded_truncation(self):
        from inferx.repo_prompt import build_prompt
        class Client:
            async def request_json(self, method, path, payload):
                return {'prompt': 'huge template'} if path == '/apply-template' else {'tokens': list(range(100))}
        with self.assertRaises(ValueError):
            await build_prompt(Client(), dict(repo_name='repo', file_name='/repo/a.py',
                prefix_code='', suffix_code='', context_code=[]), max_prompt_tokens=10)
