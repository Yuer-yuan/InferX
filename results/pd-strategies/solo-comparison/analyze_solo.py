"""Compare supplemental isolated requests with the existing formal mixed waves."""
import argparse
import gzip
import hashlib
import json
import statistics
from pathlib import Path


CASES = ['sglang-ctx-control', 'sglang-ctx-responsive128', 'sglang-pdmux-private-best']


def load(path):
    with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as stream:
        return json.load(stream)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('solo_summary', type=Path)
    parser.add_argument('formal_summary', type=Path)
    parser.add_argument('comparison', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    solo, formal, comparison = map(load, [args.solo_summary, args.formal_summary, args.comparison])
    assert len(solo['campaigns']) == 1
    assert solo['campaigns'][0]['status'] == 'restored' and solo['campaigns'][0]['restored']
    assert len(solo['groups']) == 9 and all(g['comparison_eligible'] for g in solo['groups'])
    assert len(solo['requests']) == 78
    assert all(r['ok'] and r['functional_ok'] and not r['truncated'] for r in solo['requests'])
    selected = {(r['case'], r['group']): r for r in comparison['rows']}
    rows = []
    for case in CASES:
        for group in ['cold-low', 'resume-high']:
            short_base = next(g for g in solo['groups'] if g['case'] == case and g['group'] == 'short-control')
            long_base = next(g for g in solo['groups'] if g['case'] == case and g['group'] == group)
            reference = selected[case, group]
            mixed = [g for g in formal['groups'] if g['identity'] in reference['waves']]
            assert len(mixed) == 3
            mixed_short = statistics.median(g['short']['latency_s']['p95'] for g in mixed)
            mixed_long = statistics.median(g['long']['latency_s']['median'] for g in mixed)
            short_solo = short_base['short']['latency_s']['p95']
            long_solo = long_base['long']['latency_s']['median']
            base_prefix = long_base['prefix']['long']
            if group == 'resume-high':
                assert base_prefix['cached_tokens'] == 24576
                assert base_prefix['computed_tokens'] == 106473
            rows.append(dict(
                case=case, group=group,
                solo_short_n=24, solo_long_n=1, independent_mixed_waves=3,
                short_solo_p95_s=short_solo, short_mixed_p95_median_s=mixed_short,
                short_slowdown=mixed_short / short_solo,
                long_solo_latency_s=long_solo, long_mixed_latency_median_s=mixed_long,
                long_slowdown=mixed_long / long_solo,
                short_solo_median_s=short_base['short']['latency_s']['median'],
                short_mixed_medians_s=[g['short']['latency_s']['median'] for g in mixed],
                long_solo_ttft_s=long_base['long']['ttft_s']['median'],
                long_mixed_ttft_s=[g['long']['ttft_s']['median'] for g in mixed],
                solo_long_prefix=base_prefix,
                mixed_long_prefix=[g['prefix']['long'] for g in mixed],
                mixed_waves=reference['waves'],
                solo_short_identity=short_base['identity'], solo_long_identity=long_base['identity'],
            ))
    output = dict(
        scope='Actual strict single-request client execution, retaining each formal C4 server configuration.',
        limitation='One diagnostic solo repetition per case/workload versus three earlier independent mixed waves; '
                   '24 correlated short samples per solo wave. Ratios are descriptive, not confidence intervals.',
        formula='slowdown = mixed seconds / isolated seconds; 2 means twice the latency (+100%).',
        requests_verified=78, rows=rows,
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                       [args.solo_summary, args.formal_summary, args.comparison]},
    )
    args.destination.write_text(json.dumps(output, indent=2, ensure_ascii=False) + '\n')
    for row in rows:
        print(json.dumps({key: round(value, 4) if isinstance(value, float) else value
                          for key, value in row.items() if key in
                          ['case', 'group', 'short_solo_p95_s', 'short_mixed_p95_median_s',
                           'short_slowdown', 'long_solo_latency_s', 'long_mixed_latency_median_s',
                           'long_slowdown']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
