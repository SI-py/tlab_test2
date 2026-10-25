"""Measure recorded 24-hour explicit feedback on saved Yambda recommendations.

Feedback is attached to the nearest strictly preceding listen of the same item.
It is a secondary logged-target outcome, not a counterfactual response.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

WINDOW_SECONDS = 24 * 60 * 60
FEEDBACK_TYPES = ('like', 'dislike', 'unlike', 'undislike')
LABEL_VERSION = 1



def safe(value):
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def write_json(path, value):
    path = Path(path)
    path.write_text(json.dumps(safe(value), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def source_end_timestamp(parquet):
    maximum = None
    for group in range(parquet.metadata.num_row_groups):
        metadata = parquet.metadata.row_group(group)
        for column_index in range(metadata.num_columns):
            column = metadata.column(column_index)
            if column.path_in_schema == 'timestamp.list.element' and column.statistics and column.statistics.has_min_max:
                maximum = max(maximum or 0, int(column.statistics.max))
    if maximum is None:
        raise ValueError('The official Parquet file must have timestamp maxima')
    return maximum


def _array(batch, name, row):
    return batch.column(batch.schema.get_field_index(name))[row].values


def match_user_feedback(timestamps, item_ids, event_types, listen_mask, target_keys):
    """Assign explicit events to unique nearest strictly preceding raw listens.

    A same-timestamp listen makes order unobservable at five-second precision.
    Such an event can refer either to that listen or to the prior listen: both
    candidate keys are flagged rather than assigning the event optimistically.
    """
    timestamps = np.asarray(timestamps, np.int64)
    item_ids = np.asarray(item_ids, np.int64)
    event_types = np.asarray(event_types, dtype=object)
    listen_mask = np.asarray(listen_mask, bool)
    if not (len(timestamps) == len(item_ids) == len(event_types) == len(listen_mask)):
        raise ValueError('Raw event arrays have different lengths')
    if np.any(np.diff(timestamps) < 0):
        raise ValueError('Raw Yambda user events are not chronological')
    if not np.array_equal(listen_mask, event_types == 'listen'):
        raise ValueError('played_ratio validity and listen event types disagree')
    unexpected = set(event_types.tolist()) - {'listen', *FEEDBACK_TYPES}
    if unexpected:
        raise ValueError(f'Unknown explicit event types: {unexpected}')
    by_item = defaultdict(list)
    multiplicity = Counter()
    for timestamp, item in zip(timestamps[listen_mask], item_ids[listen_mask]):
        key = (int(item), int(timestamp))
        by_item[key[0]].append(key[1])
        multiplicity[key] += 1
    by_item = {item: np.asarray(times, np.int64) for item, times in by_item.items()}
    missing = target_keys - set(multiplicity)
    if missing:
        raise ValueError(f'{len(missing)} prepared target keys do not match raw listens')
    assignments = {key: Counter() for key in target_keys}
    ambiguities = {key: Counter() for key in target_keys}
    for key in target_keys:
        if multiplicity[key] > 1:
            ambiguities[key]['duplicate_raw_listen_key'] = multiplicity[key]
    counters = Counter()
    for timestamp, item, event in zip(timestamps[~listen_mask], item_ids[~listen_mask], event_types[~listen_mask]):
        timestamp, item, event = int(timestamp), int(item), str(event)
        counters[f'raw_{event}_events'] += 1
        times = by_item.get(item)
        if times is None:
            counters['feedback_without_any_same_item_listen'] += 1
            continue
        insertion = int(np.searchsorted(times, timestamp, side='left'))
        preceding_index = insertion - 1
        simultaneous = insertion < len(times) and int(times[insertion]) == timestamp
        if simultaneous:
            counters['feedback_with_simultaneous_listen'] += 1
            key = (item, timestamp)
            if key in ambiguities:
                ambiguities[key]['simultaneous_feedback_listen'] += 1
            if preceding_index >= 0:
                prior = (item, int(times[preceding_index]))
                if timestamp - prior[1] <= WINDOW_SECONDS and prior in ambiguities:
                    ambiguities[prior]['intervening_simultaneous_listen_feedback'] += 1
            continue
        if preceding_index < 0:
            counters['feedback_without_preceding_same_item_listen'] += 1
            continue
        key = (item, int(times[preceding_index]))
        if timestamp - key[1] > WINDOW_SECONDS:
            counters['feedback_more_than_24h_after_nearest_listen'] += 1
            continue
        if multiplicity[key] > 1:
            counters['feedback_with_duplicate_nearest_listen'] += 1
            if key in ambiguities:
                ambiguities[key]['feedback_nearest_duplicate_listen'] += 1
            continue
        # Exactly one listen receives this event. Later repeats prevent the
        # event from being attributed to earlier listens of the same item.
        counters['uniquely_assigned_feedback_events'] += 1
        if key in assignments:
            assignments[key][event] += 1
            counters['uniquely_assigned_to_audited_target'] += 1
        else:
            counters['uniquely_assigned_to_other_raw_listen'] += 1
    return assignments, ambiguities, multiplicity, counters


def matching_rules():
    return {
        'version': LABEL_VERSION,
        'purpose': 'Secondary outcome audit after all policy/model/recipe decisions are frozen; never a policy feature or tuning objective',
        'window_seconds': WINDOW_SECONDS,
        'assignment_rule': 'Each explicit event goes to the unique nearest strictly earlier raw listen of the same user and raw item, within 86400 seconds. Organic and unsampled listens participate in matching.',
        'timestamp_rule': 'Official relative seconds, quantized to five seconds; simultaneous feedback/listen order is unknown and affected targets are excluded.',
        'duplicate_rule': 'Multiple raw listens with identical user/item/timestamp make the listen association ambiguous; affected target rows are excluded.',
        'same_timestamp_rule': 'When feedback coincides with a same-item listen, both that listen key and a nearest strictly prior key within 24h are marked ambiguous.',
        'like24_rule': 'At least one uniquely assigned like event in the window; this is event occurrence, not final favourite state.',
        'dislike24_rule': 'At least one uniquely assigned dislike event in the window; unlike and undislike are separately recorded and do not silently cancel occurrence labels.',
        'censoring_rule': 'Every target requires its full 24h window through the global source end; censored outcomes stay missing even when an early positive is observed.',
        'observation_assumption': 'The released event stream records feedback through its global endpoint; a user last-event time is not treated as an earlier observation endpoint.',
        'no_feedback_rule': 'Zero denotes no matched explicit event, not a dislike, lack of satisfaction, or response to a hypothetical recommendation.',
        'metrics_rule': 'User-macro mean of like24*Hit@K and dislike24*Hit@K on identical matured unambiguous target rows; among-hit rates are selected-subset descriptions only.',
        'causal_limit': 'Logged-target relevance weighted by explicit feedback, not click/like probability or causal value of the new policy.',
    }


def build_labels(prepared, source_path, output, splits=('test', 'confirmation'), batch_users=128, rebuild=False):
    source_path, output = Path(source_path), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frames = {split: getattr(prepared, split).reset_index(drop=True).copy() for split in splits}
    required = {'user_id', 'item_id', 'timestamp', 'target', 'day'}
    for split, frame in frames.items():
        if not required.issubset(frame):
            raise ValueError(f'{split} is not a timestamped Yambda target frame')
    manifest_path = output / 'explicit_feedback_labels.json'
    cache_paths = {split: output / f'feedback_labels_{split}.parquet' for split in splits}
    if not rebuild and manifest_path.exists() and all(path.exists() for path in cache_paths.values()):
        existing = json.loads(manifest_path.read_text())
        cached = {split: pd.read_parquet(path) for split, path in cache_paths.items()}
        keys = ['user_id', 'item_id', 'timestamp', 'target', 'day']
        if (existing.get('protocol', existing.get('definitions')) == matching_rules()
                and all(cached[split][keys].reset_index(drop=True).equals(frames[split][keys]) for split in splits)):
            manifest = {'definitions': matching_rules(), 'source_file': source_path.name,
                        'source_end_timestamp': existing['source_end_timestamp'],
                        'raw_matching_counts': existing.get('raw_matching_counts', {}), 'splits': existing['splits']}
            write_json(manifest_path, manifest)
            return cached, manifest
    parquet = pq.ParquetFile(source_path)
    source_end = source_end_timestamp(parquet)
    declared_end = prepared.manifest.get('source_statistics', {}).get('timestamp_max')
    if declared_end is not None and int(declared_end) != source_end:
        raise ValueError('Prepared source endpoint differs from official Parquet metadata')
    targets_by_user = defaultdict(set)
    for frame in frames.values():
        for row in frame[['user_id', 'item_id', 'timestamp']].itertuples(index=False):
            targets_by_user[int(row.user_id)].add((int(row.item_id), int(row.timestamp)))
    cohort = set(targets_by_user)
    records_by_user = {}
    counters = Counter()
    for batch in parquet.iter_batches(batch_size=batch_users, columns=['uid', 'timestamp', 'item_id', 'event_type', 'played_ratio_pct']):
        uids = batch.column(batch.schema.get_field_index('uid')).to_numpy(zero_copy_only=False)
        for row, uid in enumerate(uids):
            uid = int(uid)
            if uid not in cohort:
                continue
            if uid in records_by_user:
                raise ValueError(f'Duplicate raw user row {uid}')
            timestamps = _array(batch, 'timestamp', row).to_numpy(zero_copy_only=False).astype(np.int64)
            item_ids = _array(batch, 'item_id', row).to_numpy(zero_copy_only=False).astype(np.int64)
            event_types = np.asarray(_array(batch, 'event_type', row).to_pylist(), dtype=object)
            listen_mask = pc.is_valid(_array(batch, 'played_ratio_pct', row)).to_numpy(zero_copy_only=False)
            assignments, ambiguous, multiplicity, local_counts = match_user_feedback(
                timestamps, item_ids, event_types, listen_mask, targets_by_user[uid])
            records_by_user[uid] = (assignments, ambiguous, {key: multiplicity[key] for key in targets_by_user[uid]})
            counters.update(local_counts)
            counters['raw_cohort_users'] += 1
            counters['raw_cohort_events'] += len(timestamps)
            counters['raw_cohort_listens'] += int(listen_mask.sum())
    if set(records_by_user) != cohort:
        raise ValueError('Some prepared target users do not exist in the source')
    labels = {}
    split_counts = {}
    for split, frame in frames.items():
        rows = []
        for index, row in enumerate(frame.itertuples()):
            uid, raw_item, timestamp = int(row.user_id), int(row.item_id), int(row.timestamp)
            key = (raw_item, timestamp)
            assignments, ambiguous, multiplicity = records_by_user[uid]
            counts, flags = assignments[key], ambiguous[key]
            mature = timestamp + WINDOW_SECONDS <= source_end
            eligible = mature and not bool(flags)
            record = {'row_id': index, 'user_id': uid, 'target': int(row.target), 'item_id': raw_item,
                'day': int(row.day), 'timestamp': timestamp, 'window_end_timestamp': timestamp + WINDOW_SECONDS,
                'matured': mature, 'ambiguous': bool(flags), 'eligible': eligible,
                'raw_listen_multiplicity': int(multiplicity[key]),
                'ambiguity_reasons': ','.join(sorted(flags)), 'ambiguity_count': int(sum(flags.values()))}
            for event in FEEDBACK_TYPES:
                record[f'{event}24_count'] = int(counts[event])
                record[f'{event}24'] = float(counts[event] > 0) if eligible else np.nan
            rows.append(record)
        labels[split] = pd.DataFrame(rows)
        labels[split].to_parquet(output / f'feedback_labels_{split}.parquet', index=False)
        eligible_frame = labels[split][labels[split].eligible]
        split_counts[split] = {
            'target_rows': len(rows), 'target_users': int(frame.user_id.nunique()),
            'eligible_rows': len(eligible_frame), 'eligible_users': int(eligible_frame.user_id.nunique()),
            'right_censored_rows': int((~labels[split].matured).sum()),
            'ambiguous_rows': int(labels[split].ambiguous.sum()),
            'duplicate_raw_listen_rows': int((labels[split].raw_listen_multiplicity > 1).sum()),
            'positive_labels': {f'{event}24': int(eligible_frame[f'{event}24'].sum()) for event in FEEDBACK_TYPES},
            'users_with_positive_label': {f'{event}24': int(eligible_frame.loc[eligible_frame[f'{event}24'] > 0, 'user_id'].nunique()) for event in FEEDBACK_TYPES},
            'like_and_dislike_rows': int(((eligible_frame.like24 > 0) & (eligible_frame.dislike24 > 0)).sum()),
            'like_and_unlike_rows': int(((eligible_frame.like24 > 0) & (eligible_frame.unlike24 > 0)).sum()),
        }
    manifest = {'definitions': matching_rules(), 'source_file': source_path.name,
                'source_end_timestamp': source_end, 'raw_matching_counts': dict(counters),
                'splits': split_counts}
    write_json(manifest_path, manifest)
    return labels, manifest


def attach_labels(evaluation, labels):
    if evaluation.row_id.duplicated().any() or labels.row_id.duplicated().any():
        raise ValueError('Evaluation/label row IDs must be unique')
    outcome_columns = ['row_id', 'user_id', 'target', 'day', 'timestamp', 'eligible', 'matured', 'ambiguous']
    outcome_columns += [f'{event}24' for event in FEEDBACK_TYPES]
    joined = evaluation.merge(labels[outcome_columns], on='row_id', how='inner', validate='one_to_one', suffixes=('', '_label'))
    if len(joined) != len(evaluation) or len(joined) != len(labels):
        raise ValueError('Models must use exactly the same fixed target rows as the label sidecar')
    for key in ['user_id', 'target', 'day', 'timestamp']:
        if key in evaluation:
            if not np.array_equal(joined[key], joined[f'{key}_label']):
                raise ValueError(f'Evaluation and feedback labels disagree on {key}')
            joined = joined.drop(columns=f'{key}_label')
    return joined[joined.eligible].copy()


def model_metrics(joined):
    result = {'n_rows': len(joined), 'n_users': int(joined.user_id.nunique()),
              'like24_target_n': int(joined.like24.sum()), 'dislike24_target_n': int(joined.dislike24.sum())}
    user_columns = {}
    for k in [10, 20]:
        hit_name = f'recall@{k}'
        if hit_name not in joined:
            continue
        hit = joined[hit_name].to_numpy(float)
        for event in FEEDBACK_TYPES:
            name = f'{event}24_weighted_hit@{k}'
            joined[name] = joined[f'{event}24'].to_numpy(float) * hit
            user_columns[name] = joined.groupby('user_id')[name].mean()
            result[name] = float(user_columns[name].mean()) if len(joined) else None
        selected = joined[hit > 0]
        result[f'n_hits@{k}'] = len(selected)
        result[f'observed_like24_among_hits_micro@{k}'] = float(selected.like24.mean()) if len(selected) else None
        result[f'observed_dislike24_among_hits_micro@{k}'] = float(selected.dislike24.mean()) if len(selected) else None
        liked = joined[joined.like24 > 0]
        result[f'liked_target_recall@{k}'] = float(liked.groupby('user_id')[hit_name].mean().mean()) if len(liked) else None
        result[f'liked_target_users@{k}'] = int(liked.user_id.nunique())
    return result, pd.DataFrame(user_columns)


def bootstrap_delta(baseline, candidate, samples=2000):
    paired = pd.concat([baseline.rename('base'), candidate.rename('candidate')], axis=1, join='inner')
    if len(paired) != len(baseline) or len(paired) != len(candidate):
        raise ValueError('Secondary comparison must use identical eligible users')
    difference = paired.iloc[:, 1].to_numpy() - paired.iloc[:, 0].to_numpy()
    if not len(difference):
        return {'n_users': 0, 'delta': None, 'ci95': [None, None]}
    rng = np.random.default_rng(1997)
    draws = np.empty(samples)
    for index in range(samples):
        draws[index] = difference[rng.integers(0, len(difference), len(difference))].mean()
    return {'n_users': len(difference), 'delta': float(difference.mean()),
            'ci95': np.quantile(draws, [.025, .975]).tolist(), 'bootstrap_samples': samples,
            'unit': 'User cluster, conditional on saved checkpoints; family seeds averaged before resampling'}


def audit_predictions(run, labels, manifest, output, bootstrap_samples=2000):
    run, output = Path(run), Path(output)
    results = json.loads((run / 'results.json').read_text())
    names = list(results['models'])
    audited = {'definitions': matching_rules(), 'label_counts': manifest['splits'], 'splits': {},
               'interpretation': 'Secondary logged-feedback weighted relevance, not counterfactual policy value.'}
    for split, sidecar in labels.items():
        entries, user_metrics = {}, {}
        for name in names:
            path = run / 'evaluation' / f'{name}_{split}.pkl'
            evaluation = pd.read_pickle(path) if path.exists() else pd.read_csv(path.with_suffix('.csv'))
            joined = attach_labels(evaluation, sidecar)
            metrics, per_user = model_metrics(joined)
            family = results['models'][name].get('family', name)
            entries[name] = {'family': family, 'seed': results['models'][name].get('seed'), **metrics}
            user_metrics[name] = per_user
        families = {}
        family_users = {}
        for family in sorted({entry['family'] for entry in entries.values()}):
            members = [name for name, entry in entries.items() if entry['family'] == family]
            metric_names = set.intersection(*(set(user_metrics[name].columns) for name in members))
            combined = {}
            for metric in metric_names:
                aligned = pd.concat([user_metrics[name][metric] for name in members], axis=1, join='inner')
                if any(len(aligned) != len(user_metrics[name]) for name in members):
                    raise ValueError('Family members use different eligible feedback users')
                combined[metric] = aligned.mean(axis=1)
            family_users[family] = pd.DataFrame(combined)
            families[family] = {'n_runs': len(members), 'n_eligible_users': len(family_users[family]),
                                **{metric: float(series.mean()) if len(series) else None for metric, series in combined.items()}}
        comparisons = {}
        if 'baseline' in family_users:
            for family, values in family_users.items():
                if family == 'baseline':
                    continue
                comparisons[family + '_vs_baseline'] = {
                    metric: bootstrap_delta(family_users['baseline'][metric], values[metric], bootstrap_samples)
                    for metric in ['like24_weighted_hit@10', 'dislike24_weighted_hit@10']
                    if metric in values and metric in family_users['baseline']}
        audited['splits'][split] = {'models': entries, 'family_means': families, 'comparisons': comparisons}
    write_json(output / 'feedback.json', audited)
    return audited


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='artifacts/yambda')
    parser.add_argument('--data', default='data/raw/yambda50m_multi.parquet')
    parser.add_argument('--output', default='results')
    parser.add_argument('--labels-only', action='store_true')
    parser.add_argument('--rebuild-labels', action='store_true', help='Recompute cached feedback labels')
    parser.add_argument('--splits', nargs='+', default=['test', 'confirmation'], choices=['test', 'confirmation'])
    parser.add_argument('--bootstrap-samples', type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        parser.error('--bootstrap-samples must be positive')
    with (Path(args.run) / 'prepared.pkl').open('rb') as handle:
        prepared = pickle.load(handle)
    labels, manifest = build_labels(prepared, args.data, args.output, tuple(args.splits), rebuild=args.rebuild_labels)
    if args.labels_only:
        print(json.dumps({'status': 'labels_ready', 'counts': manifest['splits']}, ensure_ascii=False))
    else:
        audited = audit_predictions(args.run, labels, manifest, args.output, args.bootstrap_samples)
        print(json.dumps({'status': 'audit_ready', 'models': len(next(iter(audited['splits'].values()))['models'])}, ensure_ascii=False))

if __name__ == '__main__':
    main()
