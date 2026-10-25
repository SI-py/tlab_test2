"""Compare observed, capped listening-time proxies on saved Yambda predictions.

Utility is Hit@10 times min(observed completion times observed duration, 600).
It describes logged targets, not counterfactual listening time of a new policy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from rlrec.evaluate_v2 import bootstrap_family, holm_adjust

SPLITS = ('test', 'confirmation')
METRIC = 'logged_capped_single_play_seconds_hit@10'
CAP = 600
CONTRASTS = (('seconds_h5', 'anchored_h5'), ('duration_neutral_h5', 'anchored_h5'))



def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe(value):
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def target_population(frame, vocab_size):
    frame = frame.reset_index(drop=True)
    needed = {'user_id', 'target', 'day', 'history', 'item_id', 'timestamp', 'duration', 'reward'}
    require(needed <= set(frame), f'Missing observed target columns: {sorted(needed-set(frame))}')
    target = frame.target.to_numpy()
    require(np.all((target == 2) | ((target >= 3) & (target < vocab_size))), 'Target outside frozen catalog')
    oov = target == 2
    duration = frame.duration.to_numpy(float)
    valid_duration = np.isfinite(duration) & (duration > 0)
    eligible = oov | valid_duration
    completion = frame.reward.to_numpy(float)
    require(np.isfinite(completion[~oov & eligible]).all(), 'Known eligible target lacks finite observed completion')
    require(eligible.any(), 'No eligible duration audit targets')
    labels = np.zeros(len(frame), dtype=float)
    known_valid = ~oov & eligible
    labels[known_valid] = np.minimum(np.clip(completion[known_valid], 0, 1) * duration[known_valid], CAP)
    identity = frame[['user_id', 'target', 'day']].copy()
    identity.insert(0, 'row_id', np.arange(len(frame)))
    identity['context_key'] = frame.history.map(lambda history: ','.join(map(str, map(int, history))))
    all_users = set(frame.user_id)
    eligible_users = set(frame.loc[eligible, 'user_id'])
    counts = {
        'n_rows': len(frame), 'n_users': len(all_users), 'n_known_rows': int((~oov).sum()),
        'n_known_users': int(frame.loc[~oov, 'user_id'].nunique()),
        'n_oov_rows': int(oov.sum()), 'n_oov_users': int(frame.loc[oov, 'user_id'].nunique()),
        'n_valid_duration_rows': int(valid_duration.sum()),
        'n_invalid_duration_rows': int((~valid_duration).sum()),
        'n_invalid_duration_users': int(frame.loc[~valid_duration, 'user_id'].nunique()),
        'n_known_invalid_duration_rows': int((~oov & ~valid_duration).sum()),
        'n_known_invalid_duration_users': int(frame.loc[~oov & ~valid_duration, 'user_id'].nunique()),
        'n_oov_invalid_duration_rows': int((oov & ~valid_duration).sum()),
        'n_oov_invalid_duration_users': int(frame.loc[oov & ~valid_duration, 'user_id'].nunique()),
        'n_missing_duration_rows': int(np.isnan(duration).sum()),
        'n_nonfinite_duration_rows': int((~np.isfinite(duration)).sum()),
        'n_nonpositive_finite_duration_rows': int((np.isfinite(duration) & (duration <= 0)).sum()),
        'n_eligible_rows': int(eligible.sum()), 'n_excluded_rows': int((~eligible).sum()),
        'n_eligible_users': len(eligible_users), 'n_users_with_excluded_rows': int(frame.loc[~eligible, 'user_id'].nunique()),
        'n_users_excluded_entirely': len(all_users-eligible_users),
        'n_oov_invalid_completion_rows': int((oov & ~np.isfinite(completion)).sum()),
    }
    return frame, identity, eligible, labels, counts


def read_predictions(path, identity):
    frame = pd.read_pickle(path).sort_values('row_id').reset_index(drop=True)
    if len(frame) != len(identity) or not frame[list(identity)].equals(identity):
        raise ValueError(f'Prediction rows do not match observed targets: {path.name}')
    return frame



def analyze(directory):
    run = Path(directory)
    config = json.loads((run / 'config.json').read_text())
    results = json.loads((run / 'results.json').read_text())
    with (run / 'prepared.pkl').open('rb') as stream:
        prepared = pickle.load(stream)
    manifest = prepared.manifest
    seeds = config['sft_seeds']
    expected = {name: (entry['family'], entry.get('seed')) for name, entry in results['models'].items()}
    output = {'dataset': 'yambda', 'metric': METRIC,
              'unit': 'Seconds per logged target under equal eligible-user weighting',
              'cap_seconds': CAP, 'seeds': seeds, 'splits': {}}
    model_rows, family_rows, contrast_rows = [], [], []
    for split in SPLITS:
        observed, identity, eligible, labels, eligibility = target_population(getattr(prepared, split), len(manifest['item_ids']))
        models, families, frames = {}, {}, {}
        for name, (family, seed) in expected.items():
            relative = f'evaluation/{name}_{split}.pkl'
            frame = read_predictions(run / relative, identity)
            audited = identity.loc[eligible].copy()
            audited[METRIC] = (frame['recall@10'].to_numpy()*labels)[eligible]
            value = float(audited.groupby('user_id')[METRIC].mean().mean())
            models[name] = {'family': family, 'seed': seed, 'mean': value,
                            'n_eligible_rows': eligibility['n_eligible_rows'], 'n_eligible_users': eligibility['n_eligible_users']}
            model_rows.append({'split': split, 'model': name, **models[name]})
            frames[(family, seed)] = audited
        for family in sorted({family for family, seed in expected.values()}):
            members = [item for item in models.values() if item['family'] == family]
            values = [item['mean'] for item in members]
            families[family] = {'mean': float(np.mean(values)), 'seed_sd': float(np.std(values, ddof=1)) if len(values)>1 else None,
                                'n_runs': len(values), 'n_eligible_rows': eligibility['n_eligible_rows'],
                                'n_eligible_users': eligibility['n_eligible_users']}
            family_rows.append({'split': split, 'family': family, **families[family]})
        comparisons = {}
        for candidate, baseline in CONTRASTS:
            key = f'{candidate}_vs_{baseline}'
            result = bootstrap_family([frames[(baseline, seed)] for seed in seeds],
                                      [frames[(candidate, seed)] for seed in seeds], METRIC, margin=0,
                                      n_bootstrap=2000, confidence=.95, seed=42)
            result.pop('noninferiority', None)
            comparisons[key] = {'baseline_family': baseline, 'candidate_family': candidate, **result}
        adjusted = holm_adjust({key: item['paired_randomization_p_two_sided'] for key, item in comparisons.items()})
        for key, item in comparisons.items():
            item['holm_adjusted_p_two_sided'] = adjusted[key]
            item['holm_family'] = f'yambda/{split}/two_secondary_duration_utility_contrasts'
            contrast_rows.append({'split': split, 'contrast': key, **{field: value for field, value in item.items()
                                  if not isinstance(value, (dict, list))},
                                  **{f'{ci}_{bound}': item[ci][bound] for ci in ('ci_user', 'ci_training_seed', 'ci_joint') for bound in ('low', 'high')}})
        output['splits'][split] = {'eligibility': eligibility, 'models': models, 'families': families,
                                  'comparisons': comparisons}
    return output, {'models': model_rows, 'families': family_rows, 'comparisons': contrast_rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, help='Yambda experiment directory')
    parser.add_argument('--output', default='results')
    args = parser.parse_args()
    result, tables = analyze(args.run)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output/'duration_utility.json').write_text(json.dumps(safe(result), ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(output/f'duration_utility_{name}.csv', index=False)
    print(f'Duration audit saved: {output / "duration_utility.json"}')

if __name__ == '__main__':
    main()
