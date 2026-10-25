"""Paired comparisons of PG, rollout horizon and reward definitions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rlrec.evaluate_v2 import bootstrap_family, holm_adjust


SPLITS = ('test', 'confirmation')
METRICS = ('ndcg@10', 'factual_completion_weighted_hit@10', 'predicted_reward@10')
IDENTITY = ('row_id', 'user_id', 'target', 'day', 'context_key', 'actual_completion', 'known_target')
CONTRASTS = {
    'kion': (
        ('anchored_h5', 'matched_control'),
        ('anchored_h5', 'anchored_h1'),
        ('appended_h5', 'anchored_h5'),
        ('retention_h5', 'anchored_h5'),
    ),
    'yambda': (
        ('anchored_h5', 'matched_control'),
        ('anchored_h5', 'anchored_h1'),
        ('duration_neutral_h5', 'anchored_h5'),
        ('seconds_h5', 'anchored_h5'),
    ),
}


def read_split(run, split, families, seeds):
    frames, reference = {}, None
    for family in families:
        for seed in seeds:
            path = run / 'evaluation' / f'{family}_{seed}_{split}.pkl'
            frame = pd.read_pickle(path).sort_values('row_id').reset_index(drop=True)
            identity = frame[list(IDENTITY)]
            if reference is None:
                reference = identity
            elif not identity.equals(reference):
                raise ValueError(f'Paired comparison requires the same observations: {path.name}')
            frames[(family, seed)] = frame
    return frames, {'n_rows': len(reference), 'n_users': int(reference.user_id.nunique())}


def analyze_run(directory, dataset):
    run = Path(directory)
    results = json.loads((run / 'results.json').read_text())
    seeds = results['config']['sft_seeds']
    families = sorted({family for pair in CONTRASTS[dataset] for family in pair})
    output, rows = {'seeds': seeds, 'splits': {}}, []
    for split in SPLITS:
        frames, population = read_split(run, split, families, seeds)
        comparisons, pvalues = {}, {}
        for candidate, comparator in CONTRASTS[dataset]:
            name = f'{candidate}_vs_{comparator}'
            by_metric = {}
            for metric in METRICS:
                first = frames[(families[0], seeds[0])]
                finite = np.isfinite(first[metric].to_numpy(float))
                if not finite.any():
                    raise ValueError(f'No finite observations for {dataset}/{split}/{metric}')
                if any(not np.array_equal(np.isfinite(frame[metric].to_numpy(float)), finite)
                       for frame in frames.values()):
                    raise ValueError(f'Paired metric populations differ: {dataset}/{split}/{metric}')
                bases = [frames[(comparator, seed)].loc[finite].copy() for seed in seeds]
                tunes = [frames[(candidate, seed)].loc[finite].copy() for seed in seeds]
                stat = bootstrap_family(bases, tunes, metric, margin=0., confidence=.95,
                                        n_bootstrap=2000, seed=42)
                stat.pop('noninferiority', None)
                stat['training_seed_delta_sd'] = (float(np.std(stat['per_seed_delta'], ddof=1))
                                                  if len(seeds) > 1 else None)
                stat.update(candidate=candidate, comparator=comparator,
                            n_excluded_missing_metric=int((~finite).sum()))
                by_metric[metric] = stat
                pvalues[f'{name}:{metric}'] = stat['paired_randomization_p_two_sided']
            comparisons[name] = {'baseline_family': comparator, 'candidate_family': candidate,
                                 'metrics': by_metric}
        adjusted = holm_adjust(pvalues)
        for name, comparison in comparisons.items():
            for metric, stat in comparison['metrics'].items():
                stat['holm_adjusted_p_two_sided'] = adjusted[f'{name}:{metric}']
                rows.append({'dataset': dataset, 'split': split, 'contrast': name,
                             'candidate': stat['candidate'], 'comparator': stat['comparator'],
                             'metric': metric, 'comparator_mean': stat['baseline_mean'],
                             'candidate_mean': stat['tuned_mean'], 'delta': stat['delta'],
                             'relative_delta': stat['relative_delta'],
                             'training_seed_delta_sd': stat['training_seed_delta_sd'],
                             **{f'{label}_{bound}': stat[field][bound]
                                for label, field in (('ci_user', 'ci_user'),
                                                     ('ci_seed', 'ci_training_seed'),
                                                     ('ci_joint', 'ci_joint'))
                                for bound in ('low', 'high')},
                             'p_two_sided': stat['paired_randomization_p_two_sided'],
                             'p_holm': stat['holm_adjusted_p_two_sided'],
                             'holm_family_n_tests': len(pvalues),
                             'n_users': stat['n_users'], 'n_rows': stat['n_examples'],
                             'n_seeds': stat['n_seeds']})
        output['splits'][split] = {'population': population,
                                   'holm_family_n_tests': len(pvalues),
                                   'comparisons': comparisons}
    return output, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kion', type=Path)
    parser.add_argument('--yambda', type=Path)
    parser.add_argument('--output', type=Path, default=Path('results'))
    args = parser.parse_args()
    if args.kion is None and args.yambda is None:
        parser.error('Provide --kion and/or --yambda')
    result = {'datasets': {}, 'uncertainty': {
        'bootstrap_samples': 2000, 'confidence': .95, 'seed': 42,
        'primary': 'Paired user-cluster intervals conditional on trained checkpoints.',
        'additional': 'Seed and crossed user/seed intervals are exploratory with only three seeds.',
        'multiplicity': 'Separate Holm family of 12 two-sided tests within each dataset and period.',
        'noninferiority': 'Not evaluated for these direct contrasts.'}}
    rows = []
    for dataset in ('kion', 'yambda'):
        directory = getattr(args, dataset)
        if directory is not None:
            result['datasets'][dataset], dataset_rows = analyze_run(directory, dataset)
            rows.extend(dataset_rows)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / 'paired_ablation_statistics.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    pd.DataFrame(rows).to_csv(args.output / 'paired_ablation_statistics.csv', index=False)
    print(f'Saved paired comparisons to {args.output}')


if __name__ == '__main__':
    main()
