#!/usr/bin/env python3
"""Extract only preselected repositories from the already hashed HDD dataset."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inferx.repository import extract_repositories

REPOSITORIES = ['more-itertools', 'sqlparse', 'json_repair', 'python-fastjsonschema', 'compressed_rtf',
                'xmltodict', 'PTable', 'algorithms', 'PyPattyrn', 'python-pinyin', 'boltons']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.data/'manifest.json').read_text())
    extract_repositories(args.data/'repos.zip', args.data/'fixtures', REPOSITORIES)
    hashes = {str(p.relative_to(args.data/'fixtures')): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted((args.data/'fixtures').rglob('*')) if p.is_file()}
    (args.data/'fixtures-manifest.json').write_text(json.dumps(
        dict(upstream=manifest, repositories=REPOSITORIES, files=hashes), indent=2) + '\n')
    print(json.dumps(dict(repositories=REPOSITORIES, files=len(hashes))))


if __name__ == '__main__':
    main()
