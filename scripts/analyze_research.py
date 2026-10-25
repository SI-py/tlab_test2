"""Explain saved recommendations without refitting models or selecting recipes."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd


USER_MODAL_DESCRIPTION = {
    'added_utc': '2026-10-25T12:43:02+00:00',
    'status': 'descriptive added after primary inspection; no fitting/reselection',
    'scope': 'Global modal top-1 item from the existing distinct-context first_counts; exact user summaries only, no new hypothesis tests',
}


def modal_user_summary(unique, slates, modal_token):
    """Give users equal weight after averaging their distinct contexts."""
    users = pd.DataFrame({'user_id': unique.user_id.to_numpy(),
                          'is_modal': [int(slate[0] == modal_token) for slate in slates]})
    shares = users.groupby('user_id').is_modal.mean()
    return {
        'modal_top1_user_macro_share': float(shares.mean()),
        'users_ever_modal_top1_fraction': float((shares > 0).mean()),
        'users_majority_modal_top1_fraction': float((shares > .5).mean()),
        'n_behavior_users': int(len(shares)),
    }


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')




def analyze(run, output, raw):
    run, output, raw = Path(run), Path(output), Path(raw)
    results = json.loads((run / 'results.json').read_text())
    config = results['config']
    manifest = results['data_manifest']
    ids = manifest['item_ids']
    dataset = results['dataset']
    titles, kinds = {}, {}
    if dataset == 'kion':
        items = pd.read_csv(raw / 'items.csv', usecols=['item_id', 'title', 'content_type'])
        titles = dict(zip(items.item_id.astype(int), items.title.fillna('')))
        kinds = dict(zip(items.item_id.astype(int), items.content_type.fillna('')))
    output.mkdir(parents=True, exist_ok=True)
    evidence = {'dataset': dataset, 'definitions': {
        'exposure': 'Equal weight per distinct user/day/history context, then equal weight per seed within family',
        'familiar_top10_share': 'Fraction of top-10 tokens in the supplied history; Yambda history is the last 48 listens, not lifetime history',
        'top10_overlap_with_baseline': 'Intersection size divided by ten, averaged over distinct user/day/history contexts within user, then across users; paired SFT seed',
        'series_share': 'Metadata content_type=series; not a proxy for runtime',
        'top_items': 'Descriptive exposure counts, not observed responses to the new policy',
        'modal_top1_user_macro_share': 'For the global modal top-1 token counted over distinct contexts, average the within-user distinct-context share, then average users equally',
        'users_ever_modal_top1_fraction': 'Fraction of users with that global modal top-1 item in at least one distinct context',
        'users_majority_modal_top1_fraction': 'Fraction of users with that global modal top-1 item in strictly more than half of their distinct contexts',
        'n_behavior_users': 'Number of users represented by the distinct contexts; same denominator for all three modal-item user summaries',
    }, 'descriptive_additions': {'global_modal_top1_user_summary': USER_MODAL_DESCRIPTION}, 'models': {}, 'families': {}}
    rows, exposures = [], []
    for split in ('test', 'confirmation'):
        for name, entry in results['models'].items():
            path = run / 'evaluation' / f'{name}_{split}.pkl'
            frame = pd.read_pickle(path)
            keys = ['user_id', 'day', 'context_key']
            slate_keys = frame.top_items.map(tuple)
            if frame.assign(slate_key=slate_keys).groupby(keys).slate_key.nunique().max() != 1:
                raise ValueError(f'One context has inconsistent slates: {name}/{split}')
            unique = frame.drop_duplicates(keys)
            slates = [list(map(int, x[:10])) for x in unique.top_items]
            counts = Counter(token for slate in slates for token in slate)
            first_counts = Counter(slate[0] for slate in slates)
            modal_token = first_counts.most_common(1)[0][0]
            total = sum(counts.values())
            familiar = []
            for slate, history in zip(slates, unique.context_key):
                seen = set(map(int, history.split(','))) if history else set()
                familiar.append(np.mean([item in seen for item in slate]))
            behavior = {
                'n_contexts': len(unique),
                'familiar_top10_share': float(np.mean(familiar)),
                'top1_item_id': int(ids[modal_token]),
                **modal_user_summary(unique, slates, modal_token),
            }
            if dataset == 'kion':
                behavior['series_exposure_share'] = sum(count for token, count in counts.items()
                    if kinds.get(int(ids[token])) == 'series') / total
            seed = entry.get('seed')
            if seed is not None:
                baseline = pd.read_pickle(run / 'evaluation' / f'baseline_{seed}_{split}.pkl')
                for key in ('row_id', 'user_id', 'target', 'day', 'context_key'):
                    if not np.array_equal(frame[key].to_numpy(), baseline[key].to_numpy()):
                        raise ValueError(f'Unpaired predictions: {name}/{split}/{key}')
                pair = pd.DataFrame({'user_id': frame.user_id, 'day': frame.day, 'context_key': frame.context_key,
                    'overlap': [len(set(a[:10]) & set(b[:10])) / 10
                                for a, b in zip(frame.top_items, baseline.top_items)],
                    'first_same': [int(a[0] == b[0]) for a, b in zip(frame.top_items, baseline.top_items)]}).drop_duplicates(keys)
                behavior['top10_overlap_with_baseline'] = float(pair.groupby('user_id').overlap.mean().mean())
                behavior['top1_agreement_with_baseline'] = float(pair.groupby('user_id').first_same.mean().mean())
            behavior['frequent_items'] = [{'item_id': int(ids[token]), 'title': str(titles.get(int(ids[token]), '')),
                'content_type': str(kinds.get(int(ids[token]), '')), 'exposure_share': count / total,
                'contexts_with_item': count} for token, count in counts.most_common(15)]
            evidence['models'].setdefault(name, {'family': entry['family'], 'seed': seed})[split] = behavior
            rows.append({'dataset': dataset, 'split': split, 'model': name,
                'family': entry['family'], 'seed': seed,
                **{k: v for k, v in behavior.items() if isinstance(v, (int, float))}})
            for token, count in counts.items():
                exposures.append({'dataset': dataset, 'split': split, 'model': name,
                    'family': entry['family'], 'seed': seed, 'item_id': int(ids[token]),
                    'title': str(titles.get(int(ids[token]), '')),
                    'content_type': str(kinds.get(int(ids[token]), '')), 'exposure_share': count / total})
    model_frame = pd.DataFrame(rows)
    exposure_frame = pd.DataFrame(exposures)
    for (family, split), group in model_frame.groupby(['family', 'split'], sort=True):
        summary = {}
        for metric in group.select_dtypes(include='number'):
            if metric in ('seed', 'top1_item_id'):
                continue
            values = group[metric].dropna().to_numpy(float)
            if len(values):
                summary[metric] = {'mean': float(values.mean()),
                    'seed_sd': float(values.std(ddof=1)) if len(values) > 1 else 0.}
        family_exposure = exposure_frame[(exposure_frame.family == family) & (exposure_frame.split == split)]
        # Missing item/seed combinations are zero, not silently dropped.
        count = len(group)
        averaged = family_exposure.groupby(['item_id', 'title', 'content_type'], dropna=False).exposure_share.sum() / count
        summary['frequent_items'] = [{'item_id': int(key[0]), 'title': str(key[1]),
            'content_type': str(key[2]), 'exposure_share': float(value)}
            for key, value in averaged.sort_values(ascending=False).head(15).items()]
        evidence['families'].setdefault(family, {})[split] = summary
    model_frame.to_csv(output / f'{dataset}_behavior.csv', index=False)
    exposure_frame.to_csv(output / f'{dataset}_item_exposure.csv', index=False)
    write_json(output / f'{dataset}_behavior.json', evidence)
    training_rows = []
    for name, entry in results['models'].items():
        path = run / 'training' / f'{name}.json'
        if entry['family'] == 'baseline' and not path.exists():
            architecture = results['sft_selection']['selected']
            path = run / 'training' / f"sft_{architecture}_{entry['seed']}.json"
        if not path.exists():
            continue
        for row in json.loads(path.read_text()):
            validation = row.get('validation', {})
            losses = {key: value for key, value in row.items() if key not in ('validation', 'completion')}
            if 'completion' in row:
                # UtilityReward transforms this head for the two duration
                # objectives; the original log's name is not a common unit.
                losses['pre_penalty_score'] = row['completion']
                utility = (entry.get('variant') or {}).get('reward', 'completion')
                if utility == 'seconds':
                    losses['score_unit'] = 'expected_seconds_divided_by_300'
                    losses['score_definition'] = 'min(expected single-play seconds, 600) / 300; bounded in [0, 2], not seconds'
                elif utility == 'duration_neutral':
                    losses['score_unit'] = 'dimensionless_completion_residual'
                    losses['score_definition'] = 'predicted completion - frozen duration-bin completion prior + global completion mean; not bounded to [0, 1]'
                else:
                    losses['score_unit'] = 'completion_fraction'
                    losses['score_definition'] = 'Frozen predicted completion fraction before uncertainty/support/retention penalties or mixing'
            training_rows.append({'dataset': dataset, 'model': name,
                'family': entry['family'], 'seed': entry.get('seed'),
                **losses,
                **{'validation_' + key: value for key, value in validation.items()}})
    pd.DataFrame(training_rows).to_csv(output / f'{dataset}_training_curves.csv', index=False)
    return {'dataset': dataset, 'models': len(evidence['models']), 'output': str(output)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kion', type=Path)
    parser.add_argument('--yambda', type=Path)
    parser.add_argument('--data', type=Path, default=Path('data/raw'))
    parser.add_argument('--output', type=Path, default=Path('results'))
    args = parser.parse_args()
    if not (args.kion or args.yambda):
        parser.error('Pass --kion and/or --yambda')
    for folder in (args.kion, args.yambda):
        if folder:
            print(json.dumps(analyze(folder, args.output, args.data)))
