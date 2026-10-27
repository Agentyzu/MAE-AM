import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
METRICS = ['ad_sw', 'ad_rev_sum', 'ad_utility_sum', 'user_satis', 'ad_satis', 'score',
           'advertiser_utility', 'truthful_utility', 'utility_gain']


def collect_results(source):
    rows = []
    for path in sorted(source.rglob('*_results_*.json')):
        with path.open(encoding='utf-8') as file:
            result = json.load(file)
        suffixes = [''] if 'ad_sw' in result else [key[len('ad_sw'):] for key in result if key.startswith('ad_sw_')]
        if not suffixes:
            raise ValueError(f'No auction metrics found: {path}')
        file_rows = []
        for suffix in suffixes:
            row = {'auction': path.parent.parent.name, 'split': path.parent.name,
                   'model': path.name.split('_results_', 1)[0], 'beta': None, 'advertiser': None,
                   'advertiser_utility': None, 'truthful_utility': None, 'utility_gain': None}
            utilities = result[f'ad_utility{suffix}']
            if suffix:
                beta, advertiser = suffix[1:].rsplit('_', 1)
                advertiser = int(advertiser)
                rank = result[f'ad_rank{suffix}'][:len(utilities)]
                row.update(beta=float(beta), advertiser=advertiser,
                           advertiser_utility=utilities[rank.index(advertiser)] if advertiser in rank else 0.0)
            row['ad_sw'] = result[f'ad_sw{suffix}']
            row['ad_rev_sum'] = sum(result[f'ad_rev{suffix}'])
            row['ad_utility_sum'] = sum(utilities)
            for metric in ('user_satis', 'ad_satis', 'score'):
                row[metric] = result.get(metric)
            file_rows.append(row)
        truthful = {row['advertiser']: row['advertiser_utility'] for row in file_rows if row['beta'] == 1}
        for row in file_rows:
            if row['beta'] is not None:
                if row['advertiser'] not in truthful:
                    raise ValueError(f"Missing beta=1 baseline for advertiser {row['advertiser']}: {path}")
                row['truthful_utility'] = truthful[row['advertiser']]
                row['utility_gain'] = row['advertiser_utility'] - row['truthful_utility']
        rows.extend(file_rows)
    if not rows:
        raise ValueError(f'No experiment result JSON files found under {source}')
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Aggregate experiment results and export Excel tables')
    parser.add_argument('--results-dir', type=Path, default=ROOT / 'results')
    parser.add_argument('--dataset', choices=('ADGEN', 'ATVI'), default='ADGEN')
    parser.add_argument('--auction', help='Auction name used by run_experiment.py, or its result directory name')
    parser.add_argument('--output', type=Path, help='Output Excel file')
    args = parser.parse_args(argv)
    source = args.results_dir.resolve() / args.dataset
    if args.auction:
        source /= {'ALS': 'Prom_ALS', 'AAL': 'Prom_AAL', 'SQA': 'GSP_SQA'}.get(args.auction, args.auction)
    if not source.is_dir():
        parser.error(f'Result directory does not exist: {source}')
    table = collect_results(source)
    grouped = table.groupby(['auction', 'model', 'beta', 'advertiser'], dropna=False)[METRICS]
    output = args.output or source / 'summary.xlsx'
    output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        table.to_excel(writer, sheet_name='per_split', index=False)
        grouped.mean().reset_index().to_excel(writer, sheet_name='mean', index=False)
        grouped.var().reset_index().to_excel(writer, sheet_name='variance', index=False)
        grouped.count().reset_index().to_excel(writer, sheet_name='count', index=False)
    print(f'Saved {len(table)} result row(s): {output.resolve()}')


if __name__ == '__main__':
    main()
