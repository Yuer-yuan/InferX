"""Run on the HDD-resident remote checkout; write every response before summarizing."""
import argparse
import asyncio
import json
from pathlib import Path

from inferx.tuning import benchmark


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--endpoint', default='http://127.0.0.1:18081')
    parser.add_argument('--model', default='qwen38-27b')
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--cache', choices=['cold', 'warm', 'natural'], default='cold')
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--concurrency', type=int, default=1)
    args = parser.parse_args()
    if args.repeat < 1 or args.concurrency < 1:
        parser.error('repeat and concurrency must be positive')
    corpus = json.loads(args.corpus.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        records = asyncio.run(benchmark(args.endpoint, args.model, corpus, output,
                                       args.config, args.cache, args.repeat, args.concurrency))
    if not all(r['summary']['ok'] for r in records):
        raise SystemExit('One or more trials failed; inspect raw output.')


if __name__ == '__main__':
    main()
