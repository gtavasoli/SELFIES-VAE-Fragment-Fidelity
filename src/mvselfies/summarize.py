"""Recompute archived fragment metrics from count CSVs and write a run summary."""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path

from .evaluate import jsd_full, compute_rare_recall, topk_coverage


def read_counts(path):
    """Read positive counts from an evaluation CSV into a fragment Counter.

    ``path`` is a pathlib.Path. The first CSV column identifies fragments;
    counts come from ``count``, ``count_train``, or ``count_gen``. Repeated
    fragment rows are summed and nonpositive counts are skipped.
    """
    counts = Counter()
    with path.open(encoding='utf-8', newline='') as stream:
        rows = csv.DictReader(stream)
        column = rows.fieldnames[0]
        count_column = next(key for key in rows.fieldnames if key in {'count','count_train','count_gen'})
        for row in rows:
            count = int(row[count_column])
            if count > 0:
                counts[row[column]] += count
    return counts


def verify_metrics(directory, report):
    """Check recorded metrics against the train/generated count CSVs.

    Parameters
    ----------
    directory : pathlib.Path
        Directory containing ``frag_freq_train.csv`` and ``frag_freq_gen.csv``.
    report : dict
        Parsed ``frag_eval_report.json`` with JSD, rare recall, and top-k masses.

    Raises
    ------
    ValueError
        If any recomputed value differs by more than 1e-10 in absolute terms.
    """
    train = read_counts(directory/'frag_freq_train.csv')
    gen = read_counts(directory/'frag_freq_gen.csv')
    calculated = {'jsd_full_vocab':jsd_full(train, gen),
                  'rare_fragment_recall':compute_rare_recall(train, gen)}
    for key, value in calculated.items():
        if not math.isclose(value, report[key], abs_tol=1e-10, rel_tol=0):
            raise ValueError(f'{directory}: {key}: calculated {value}, archived {report[key]}')
    for side, counts in [('train',train), ('gen',gen)]:
        for k, value in topk_coverage(counts).items():
            expected = report[f'coverage_{side}_topk'][str(k)]
            if not math.isclose(value, expected, abs_tol=1e-10, rel_tol=0):
                raise ValueError(f'{directory}: {side} top-{k} mass differs.')


def summarize(root, catalog, verify=False):
    """Collect one summary row per catalog run without aggregating experiments.

    Parameters
    ----------
    root : pathlib.Path
        Repository or extracted resource root.
    catalog : list of dict
        Entries from ``experiments/runs.json`` in the desired output order.
    verify : bool, default=False
        Recompute metrics from count tables before using each saved report.

    Returns
    -------
    list of dict
        Run identifiers, training budgets, conventions, and reported metrics.
        Different training histories and repeated seed labels remain separate.
    """
    rows = []
    for run in catalog:
        directory = root/run['results']/'frag_eval_smiles'
        report = json.loads((directory/'frag_eval_report.json').read_text(encoding='utf-8'))
        if verify:
            verify_metrics(directory, report)
        config = run['training_args']
        rows.append({
            'run':run['id'], 'role':run['role'], 'model':run['model'], 'scale':run['scale'],
            'seed':run['seed'], 'epochs':config['epochs'], 'batch_size':config['batch_size'],
            'fragment_convention':run['fragment_convention'],
            'n_train_evaluated':report['train_stats']['n_molecules'],
            'n_gen_evaluated':report['gen_stats']['n_molecules'],
            'jsd_full_vocab':report['jsd_full_vocab'], 'rare_fragment_recall':report['rare_fragment_recall'],
            'gen_top100_mass':report['coverage_gen_topk']['100'],
            'gen_top500_mass':report['coverage_gen_topk']['500'],
        })
    return rows


def main():
    """Write a catalog summary CSV, optionally checking all saved fragment metrics."""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--out', type=Path, default=Path('results/summary.csv'))
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    catalog = json.loads((args.root/'experiments/runs.json').read_text(encoding='utf-8'))
    rows = summarize(args.root, catalog, args.verify)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    print(f"{'Verified and summarized' if args.verify else 'Summarized'} {len(rows)} runs: {args.out}")


if __name__ == '__main__':
    main()
