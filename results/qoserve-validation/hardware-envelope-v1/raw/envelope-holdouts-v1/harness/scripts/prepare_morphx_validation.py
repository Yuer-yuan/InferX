"""Prepare a fresh HDD-only runtime; never change old source/libraries."""
import argparse
import json
import os
from pathlib import Path

from inferx.morphx_validation import prepare_runtime


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--destination',required=True,type=Path)
    parser.add_argument('--expected-sha256',required=True)
    args=parser.parse_args()
    root=Path(os.environ['INFERX_ROOT']).resolve()
    if not all(p.resolve().is_relative_to(root) for p in [args.source,args.destination]):
        parser.error('runtime source and destination must remain on HDD root')
    manifest=prepare_runtime(args.source,args.destination,args.expected_sha256)
    print(json.dumps(dict(before=manifest['before_sha256'],after=manifest['after_sha256'],destination=str(args.destination))))


if __name__=='__main__':main()
