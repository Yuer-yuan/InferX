import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from test_repository_analysis import grid
from scripts.analyze_repository import analyze


class ExportTests(unittest.TestCase):
    def test_export_preserves_all_metrics_and_verifiable_sources(self):
        rows = grid()
        for e in rows:
            for t in e['tasks']:
                t['candidates'][0].update(code='private code', raw_output='private raw output')
                t['candidates'][0]['verification']['output_tail'] = 'private test log'
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for phase in ('dev', 'eval'):
                folder = root/phase; folder.mkdir()
                subset = [e for e in rows if e['split'] == phase]
                (folder/'episodes.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in subset))
                (folder/'COMPLETE').write_text(str(len(subset)))
                for name in ('config.json', 'prompts.json', 'warmup.json'):
                    (folder/name).write_text('{}')
                (folder/'telemetry.csv').write_text('timestamp,index\n')
            (root/'dev/selection.json').write_text('{"selected_baseline":"fixed1"}')
            prepared = root/'prepared'; prepared.mkdir()
            (prepared/'manifest.json').write_text('{}')
            (prepared/'dependencies.txt').write_text('pinned')
            (prepared/'calibration.json').write_text('[]')
            (root/'manifest.json').write_text('{"upstream":"pinned"}')
            (root/'fixtures-manifest.json').write_text('{"files":{}}')
            output = root/'export'
            script = Path(__file__).resolve().parents[1]/'scripts/export_repository_results.py'
            result = subprocess.run([sys.executable, str(script), '--dev', str(root/'dev'),
                '--eval', str(root/'eval'), '--prepared', str(prepared), '--output', str(output)],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads((output/'upstream-manifest.json').read_text()), {'upstream': 'pinned'})
            self.assertTrue((output/'fixtures-manifest.json').is_file())
            exported = [json.loads(line) for phase in ('dev', 'eval')
                        for line in (output/f'{phase}-episodes.jsonl').read_text().splitlines()]
            self.assertEqual(analyze(rows), analyze(exported))
            self.assertNotIn('private', (output/'dev-episodes.jsonl').read_text())
            for name, digest in json.loads((output/'SHA256.json').read_text()).items():
                self.assertEqual(hashlib.sha256((output/name).read_bytes()).hexdigest(), digest)
