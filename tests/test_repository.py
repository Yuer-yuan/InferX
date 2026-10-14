import ast
import json
from pathlib import Path
import stat
import tempfile
import unittest
import zipfile

try:
    from inferx.repository import (parse_completion, safe_relative,
        extract_repositories, ablated_source, source_context)
except ImportError:
    parse_completion = None


class RepositoryTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(parse_completion, 'repository helpers missing')

    def test_json_preserves_code_whitespace_and_rejects_nonstring(self):
        code = 'if x:\n        return 2\n    '
        self.assertEqual(parse_completion('```json\n' + json.dumps({'code': code}) + '\n```'), code)
        for bad in ['pass', '{"code":null}', '{"code":3}', '{}']:
            with self.assertRaises(ValueError):
                parse_completion(bad)

    def test_dataset_paths_stay_inside_named_repository(self):
        self.assertEqual(safe_relative('repo', '/repo/pkg/a.py'), Path('pkg/a.py'))
        for path in ['/other/a.py', '/repo/../outside', '/repo//a.py', '/repo/a/../../b', '/repo/a\\b.py']:
            with self.assertRaises(ValueError):
                safe_relative('repo', path)

    def test_safe_zip_extract_excludes_cache_and_rejects_symlinks(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); archive = root/'r.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('repos/repo/pkg/a.py', 'original')
                z.writestr('repos/repo/__pycache__/a.pyc', 'stale')
                z.writestr('repos/other/private.txt', 'excluded')
            extract_repositories(archive, root/'out', ['repo'])
            self.assertEqual((root/'out/repo/pkg/a.py').read_text(), 'original')
            self.assertFalse((root/'out/repo/__pycache__').exists())
            self.assertFalse((root/'out/other').exists())
            with zipfile.ZipFile(archive, 'w') as z:
                i = zipfile.ZipInfo('repos/repo/link')
                i.create_system = 3
                i.external_attr = (stat.S_IFLNK | 0o777) << 16
                z.writestr(i, '/etc/passwd')
            with self.assertRaises(ValueError):
                extract_repositories(archive, root/'out2', ['repo'])
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('repos/repo/../../outside', 'bad')
            with self.assertRaises(ValueError):
                extract_repositories(archive, root/'out3', ['repo'])

    def test_pass_ablation_is_valid_python_and_requires_block_boundary(self):
        row = dict(prefix_code='def f(x):\n    ', middle_code='if x:\n        return 2',
                   suffix_code='\n    return 0\n')
        text = ablated_source(row)
        ast.parse(text)
        self.assertEqual(text, 'def f(x):\n    pass\n    return 0\n')
        with self.assertRaises(ValueError):
            ablated_source(dict(row, prefix_code='value = '))

    def test_source_context_excludes_tests_and_target(self):
        row = dict(file_name='/repo/a.py', context_code=[
            ['/repo/a.py', 'ANSWER'], ['/repo/tests/t.py', 'SECRET TEST'],
            ['/repo/test_a.py', 'TEST'], ['/repo/pkg/helper.py', 'useful']])
        result = source_context(row)
        self.assertIn('useful', result)
        self.assertNotIn('ANSWER', result)
        self.assertNotIn('TEST', result)

    def test_reference_must_reconstruct_exact_fixture(self):
        from inferx.repository import reference_matches
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root/'repo').mkdir()
            (root/'repo/a.py').write_text('def f():\n    return 1\n')
            row = dict(repo_name='repo', file_name='/repo/a.py',
                       prefix_code='def f():\n    ', middle_code='return 1', suffix_code='\n')
            self.assertTrue(reference_matches(row, root))
            self.assertFalse(reference_matches(dict(row, middle_code='return 2'), root))

    def test_line_start_boundary_preserves_reference_and_does_not_repair_output(self):
        from inferx.repository import line_start_task
        row = dict(prefix_code='def f():\n    ', middle_code='return 1', suffix_code='\n')
        normalized = line_start_task(row)
        self.assertEqual(normalized['prefix_code'], 'def f():\n')
        self.assertEqual(normalized['middle_code'], '    return 1')
        self.assertEqual(normalized['infill_indent'], '    ')
        self.assertEqual(normalized['prefix_code'] + normalized['middle_code'] + normalized['suffix_code'],
                         row['prefix_code'] + row['middle_code'] + row['suffix_code'])
        self.assertEqual(row['prefix_code'], 'def f():\n    ')


if __name__ == '__main__':
    unittest.main()
