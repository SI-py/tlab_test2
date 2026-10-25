"""Temporal adapter for the official Yambda-50M event stream.

A Parquet row is one user's complete chronological event list. Only listen
records become history tokens; explicit feedback is counted but never converted
into duplicate listens. Repeated consumed items remain valid recommendation
candidates, unlike the novel-item KION protocol.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

SOURCE_URL = 'https://huggingface.co/datasets/yandex/yambda/resolve/main/sequential/50m/multi_event.parquet'
ARTIST_URL = 'https://huggingface.co/datasets/yandex/yambda/resolve/main/artist_item_mapping.parquet'
SECONDS_PER_DAY = 86400
RETURN_SECONDS = 7 * SECONDS_PER_DAY
SPLITS = ('train', 'rm_valid', 'valid', 'test', 'confirmation')
DEFAULTS = dict(seed=42, n_users=2000, vocab_size=10000, min_history=5,
                max_length=48, max_train_examples=200000,
                train_examples_per_user=100, eval_examples_per_user=20,
                temporal_quantiles=[.6, .7, .8, .9], parquet_batch_users=128)


@dataclass
class YambdaPreparedData:
    train: pd.DataFrame
    rm_valid: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    confirmation: pd.DataFrame
    items: pd.DataFrame
    item_ids: list[int]
    manifest: dict
    config: dict

    @property
    def vocab_size(self):
        return len(self.item_ids)


def enrich_artists(items, event_path, config):
    """Attach static full artist IDs and a bounded train-selected feature set.

    Multi-artist tracks credit their training count to each associated artist.
    The 64 most supported artists have individual features; all remaining and
    missing associations share artist_other. Full IDs remain available for exact
    artist overlap/diversity independently of this compressed representation.
    """
    requested = config.get('artist_path')
    if requested is False:
        return items, {'enabled': False, 'reason': 'Explicitly disabled in config'}
    if requested:
        artist_path = Path(requested)
        if not artist_path.is_absolute() and not artist_path.exists():
            artist_path = Path(event_path).parent / artist_path
        if not artist_path.exists():
            raise FileNotFoundError(f'Configured artist mapping does not exist: {artist_path}')
    else:
        artist_path = next((candidate for candidate in [
            Path(event_path).parent / 'yambda_artist_item_mapping.parquet',
            Path(event_path).parent / 'artist_item_mapping.parquet'] if candidate.exists()), None)
    if artist_path is None:
        return items, {'enabled': False, 'reason': 'Optional artist mapping not available'}
    parquet = pq.ParquetFile(artist_path)
    if set(parquet.schema_arrow.names) != {'artist_id', 'item_id'}:
        raise ValueError('Unexpected Yambda artist mapping schema')
    vocabulary = pa.array(items.item_id.to_numpy(np.uint32), type=pa.uint32())
    relevant = []
    for batch in parquet.iter_batches(batch_size=262144, columns=['artist_id', 'item_id']):
        keep = pc.is_in(batch.column(batch.schema.get_field_index('item_id')), value_set=vocabulary)
        selected = batch.filter(keep)
        if selected.num_rows:
            relevant.append(selected.to_pandas())
    if relevant:
        pairs = pd.concat(relevant, ignore_index=True).drop_duplicates(['item_id', 'artist_id'])
        full_ids = {int(item): sorted(set(map(int, group.artist_id))) for item, group in pairs.groupby('item_id', sort=True)}
    else:
        full_ids = {}
    counts = dict(zip(items.item_id.astype(int), items.train_count.astype(int)))
    support = Counter()
    for item, artists in full_ids.items():
        for artist in artists:
            support[artist] += int(counts[item])
    limit = min(64, max(1, int(config.get('artist_feature_limit', 64))))
    selected_artists = sorted(support, key=lambda artist: (-support[artist], artist))[:limit]
    selected_set = set(selected_artists)
    result = items.copy()
    result['artist_ids'] = [full_ids.get(int(item), []) for item in result.item_id]
    result['artist_id'] = [artists[0] if artists else -1 for artists in result.artist_ids]
    feature_labels = []
    for artists in result.artist_ids:
        labels = [f'artist_{artist}' for artist in artists if artist in selected_set]
        if not artists or any(artist not in selected_set for artist in artists):
            labels.append('artist_other')
        feature_labels.append(','.join(labels))
    result['genres'] = feature_labels
    names = sorted({label for text in feature_labels for label in text.split(',') if label})
    manifest = dict(enabled=True, source=ARTIST_URL, file=str(artist_path),
        bytes=artist_path.stat().st_size,
        source_rows=parquet.metadata.num_rows, catalogue_associations=sum(map(len, full_ids.values())),
        catalogue_items_with_artists=len(full_ids), catalogue_artist_coverage=len(full_ids)/max(1,len(items)),
        feature_artists=[int(artist) for artist in selected_artists], feature_count=len(names),
        feature_names=names, training_support={str(artist): int(support[artist]) for artist in selected_artists},
        selection_rule='Top <=64 artists by summed full-cohort train listen counts of mapped in-vocabulary tracks; ties artist ID; multi-artist tracks credit every associated artist.',
        feature_rule='Individual features for selected artists; artist_other for all remaining or missing associations. Full sorted artist_ids retained separately.',
        static_metadata_assumption='Artist-track relations treated as static; no historical metadata snapshots. Only train-observed catalogue items and training counts determine selected features.',
        primary_id_rule='artist_id is the smallest associated numeric ID for convenience, not an asserted primary performer.')
    return result, manifest


def _array(batch, column, row):
    return batch.column(batch.schema.get_field_index(column))[row].values


def _listen_arrays(batch, row, full=False):
    ratio = _array(batch, 'played_ratio_pct', row)
    # The pinned official file guarantees non-null playback fields iff listen.
    mask = pc.is_valid(ratio).to_numpy(zero_copy_only=False)
    timestamp = _array(batch, 'timestamp', row).to_numpy(zero_copy_only=False)[mask].astype(np.int64)
    organic = _array(batch, 'is_organic', row).to_numpy(zero_copy_only=False)[mask].astype(np.uint8)
    output = {'timestamp': timestamp, 'is_organic': organic}
    if full:
        output.update(
            item_id=_array(batch, 'item_id', row).to_numpy(zero_copy_only=False)[mask].astype(np.int64),
            ratio=ratio.to_numpy(zero_copy_only=False)[mask].astype(np.float32),
            duration=_array(batch, 'track_length_seconds', row).to_numpy(zero_copy_only=False)[mask].astype(np.float32))
    return output


def temporal_boundaries(path, quantiles=(.6, .7, .8, .9), batch_size=128):
    """Exact inverse-CDF quantiles of all listens, independent of feedback value.

    Timestamps are relative *seconds rounded to multiples of five*, not counts of
    five-second ticks (Yambda paper, section 3.3).
    """
    quantiles = np.asarray(quantiles, dtype=float)
    if len(quantiles) != 4 or not np.all(np.diff(quantiles) > 0) or not np.all((quantiles > 0) & (quantiles < 1)):
        raise ValueError('Need four strictly increasing temporal quantiles inside (0,1)')
    parquet = pq.ParquetFile(path)
    maximum = None
    for group in range(parquet.metadata.num_row_groups):
        row_group = parquet.metadata.row_group(group)
        for index in range(row_group.num_columns):
            column = row_group.column(index)
            if column.path_in_schema == 'timestamp.list.element' and column.statistics and column.statistics.has_min_max:
                maximum = max(maximum or 0, int(column.statistics.max))
    if maximum is None:
        raise ValueError('Expected official timestamp metadata')
    histogram = np.zeros(maximum // 5 + 1, dtype=np.int64)
    total_events, total_listens, zero_ratio, repeated_ratio = 0, 0, 0, 0
    minimum = maximum
    for batch in parquet.iter_batches(batch_size=batch_size, columns=['timestamp', 'played_ratio_pct']):
        timestamp = pc.list_flatten(batch.column(0)).to_numpy(zero_copy_only=False).astype(np.int64)
        ratio = pc.list_flatten(batch.column(1))
        mask = pc.is_valid(ratio).to_numpy(zero_copy_only=False)
        listen_ts = timestamp[mask]
        if len(listen_ts) and np.any(listen_ts % 5):
            raise ValueError('Official timestamps must have five-second precision')
        if len(listen_ts):
            histogram += np.bincount(listen_ts // 5, minlength=len(histogram))
            minimum = min(minimum, int(listen_ts.min()))
            values = ratio.to_numpy(zero_copy_only=False)[mask]
            zero_ratio += int((values == 0).sum())
            repeated_ratio += int((values > 100).sum())
        total_events += len(timestamp)
        total_listens += len(listen_ts)
    cumulative = histogram.cumsum()
    ranks = np.ceil(quantiles * total_listens).astype(np.int64)
    cutoffs = (np.searchsorted(cumulative, ranks, side='left') * 5).astype(int).tolist()
    if len(set(cutoffs)) != len(cutoffs):
        raise ValueError('Temporal quantiles collapse to identical timestamps')
    return cutoffs + [maximum + 5], dict(
        timestamp_min=minimum, timestamp_max=maximum, events=total_events,
        listens=total_listens, non_listen_events=total_events-total_listens,
        zero_ratio_listens=zero_ratio, replay_ratio_gt100_listens=repeated_ratio,
        quantile_method='Exact empirical inverse CDF on all listen timestamps, ceil(q*N) rank',
        timestamp_unit='Relative seconds rounded down to multiples of 5; multiply by neither 5 nor Unix epoch')


def _eligible_users(path, train_end, min_history, batch_size):
    eligible = []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size,
            columns=['uid', 'timestamp', 'played_ratio_pct', 'is_organic']):
        uids = batch.column(batch.schema.get_field_index('uid')).to_numpy(zero_copy_only=False)
        for row, uid in enumerate(uids):
            arrays = _listen_arrays(batch, row)
            ts, organic = arrays['timestamp'], arrays['is_organic']
            prefix = ts < train_end
            # All contemporaneous events are excluded from a target's context.
            starts = np.searchsorted(ts, ts, side='left')
            if int(prefix.sum()) > min_history and np.any(prefix & (organic == 0) & (starts >= min_history)):
                eligible.append(int(uid))
    return np.sort(np.asarray(eligible, dtype=np.int64))


def prepare_yambda(path, config=None):
    path = Path(path)
    cfg = DEFAULTS | (config or {})
    batch_size = int(cfg['parquet_batch_users'])
    boundaries, source_stats = temporal_boundaries(path, cfg['temporal_quantiles'], batch_size)
    train_end = boundaries[0]
    eligible = _eligible_users(path, train_end, cfg['min_history'], batch_size)
    if not len(eligible):
        raise ValueError('No train-eligible recommended-listen users')
    rng = np.random.default_rng(cfg['seed'])
    cohort = np.sort(rng.choice(eligible, min(int(cfg['n_users']), len(eligible)), replace=False))
    cohort_set = set(cohort.tolist())
    retained = []
    popularity = Counter()
    duration_ids, duration_values = [], []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size,
            columns=['uid', 'timestamp', 'played_ratio_pct', 'is_organic', 'item_id', 'track_length_seconds']):
        uids = batch.column(batch.schema.get_field_index('uid')).to_numpy(zero_copy_only=False)
        for row, uid in enumerate(uids):
            if int(uid) not in cohort_set:
                continue
            arrays = _listen_arrays(batch, row, full=True)
            ts = arrays['timestamp']
            if len(ts) and np.any(np.diff(ts) < 0):
                raise ValueError(f'Unsorted timestamps for uid={uid}')
            prefix = ts < train_end
            popularity.update(arrays['item_id'][prefix].tolist())
            duration_ids.append(arrays['item_id'][prefix])
            duration_values.append(arrays['duration'][prefix])
            retained.append((int(uid), arrays))
    sorted_items = sorted(popularity, key=lambda item: (-popularity[item], item))[:int(cfg['vocab_size'])]
    item_ids = [-1, -2, -3] + [int(item) for item in sorted_items]
    mapping = {raw: token for token, raw in enumerate(item_ids) if token >= 3}
    frozen_duration = pd.DataFrame({'item_id': np.concatenate(duration_ids),
                                    'duration': np.concatenate(duration_values)}).groupby('item_id').duration.median()
    items = pd.DataFrame({'item_id': item_ids[3:], 'token': np.arange(3, len(item_ids)),
                          'content_type': 'track', 'release_year': 0., 'genres': '',
                          'title': [str(item) for item in item_ids[3:]]})
    items['duration'] = items.item_id.map(frozen_duration).astype(float)
    items['track_length_seconds'] = items.duration
    items['train_count'] = items.item_id.map(popularity).astype(int)
    items, artist_manifest = enrich_artists(items, path, cfg)
    result_rows = {name: [] for name in SPLITS}
    eligible_targets = {name: 0 for name in SPLITS}
    source_maximum = source_stats['timestamp_max']
    known_replay_targets = 0
    recommended_targets = 0
    for uid, arrays in retained:
        ts, raw, organic = arrays['timestamp'], arrays['item_id'], arrays['is_organic']
        starts = np.searchsorted(ts, ts, side='left')
        tokens = np.fromiter((mapping.get(int(item), 2) for item in raw), dtype=np.int32, count=len(raw))
        period = np.searchsorted(np.asarray(boundaries[:4]), ts, side='right')
        candidates = (organic == 0) & (starts >= int(cfg['min_history']))
        for split_index, name in enumerate(SPLITS):
            indices = np.flatnonzero(candidates & (period == split_index))
            original_count = len(indices)
            eligible_targets[name] += original_count
            budget = int(cfg['train_examples_per_user'] if split_index == 0 else cfg['eval_examples_per_user'])
            if budget < 1:
                raise ValueError('Per-user example budgets must be positive')
            if len(indices) > budget:
                local_rng = np.random.default_rng(np.random.SeedSequence([int(cfg['seed']), uid, split_index]))
                indices = np.sort(local_rng.choice(indices, budget, replace=False))
            for index in indices:
                start = int(starts[index])
                timestamp = int(ts[index])
                # User return = a strictly later active relative day, not next track.
                day = timestamp // SECONDS_PER_DAY
                next_day_start = (day + 1) * SECONDS_PER_DAY
                future_index = int(np.searchsorted(ts, next_day_start, side='left'))
                maturity_end = train_end if split_index == 0 else source_maximum + 1
                mature = timestamp + RETURN_SECONDS < maturity_end
                return7 = float(future_index < len(ts) and int(ts[future_index]) <= timestamp + RETURN_SECONDS) if mature else np.nan
                history = tokens[max(0, start-int(cfg['max_length'])):start].astype(int).tolist()
                previous_consumption = bool(np.any(raw[:start] == raw[index]))
                recommended_targets += 1
                known_replay_targets += int(arrays['ratio'][index] > 100)
                result_rows[name].append(dict(user_id=uid, day=day, timestamp=timestamp,
                    history=history, history_last_timestamp=int(ts[start-1]), history_length_total=start,
                    target=int(tokens[index]), item_id=int(raw[index]),
                    reward=float(np.clip(arrays['ratio'][index] / 100., 0, 1)),
                    played_ratio_raw=float(arrays['ratio'][index]), return7=return7,
                    total_dur=float(arrays['duration'][index]), duration=float(arrays['duration'][index]),
                    is_organic=0, target_seen_before=previous_consumption,
                    sampling_weight=original_count/max(1, len(indices)), user_period_candidates=original_count))
    columns = ['user_id','day','timestamp','history','history_last_timestamp','history_length_total',
               'target','item_id','reward','played_ratio_raw','return7','total_dur','duration',
               'is_organic','target_seen_before','sampling_weight','user_period_candidates']
    frames = {name: pd.DataFrame(result_rows[name], columns=columns) for name in SPLITS}
    original_train_rows = len(frames['train'])
    maximum_train = int(cfg['max_train_examples'])
    if original_train_rows > maximum_train:
        selection = np.sort(rng.choice(original_train_rows, maximum_train, replace=False))
        frames['train'] = frames['train'].iloc[selection].reset_index(drop=True)
    effective = dict(cfg)
    effective.update(dataset='yambda', exclude_seen=False,
        train_end_timestamp=int(boundaries[0]), rm_valid_end_timestamp=int(boundaries[1]),
        valid_end_timestamp=int(boundaries[2]), test_end_timestamp=int(boundaries[3]),
        confirmation_end_timestamp=int(boundaries[4]),
        train_end=int(boundaries[0]//SECONDS_PER_DAY), rm_valid_end=int(boundaries[1]//SECONDS_PER_DAY),
        valid_end=int(boundaries[2]//SECONDS_PER_DAY), test_end=int(boundaries[3]//SECONDS_PER_DAY))
    manifest = dict(dataset='Yambda-50M sequential multi-event', source=SOURCE_URL,
        license='Apache-2.0; dataset card specifies scientific/research use',
        source_statistics=source_stats,
        source_users=pq.ParquetFile(path).metadata.num_rows, train_eligible_users=len(eligible),
        cohort=cohort.tolist(), item_ids=item_ids, config=effective,
        temporal_boundaries_seconds=dict(zip(SPLITS, map(int,boundaries))),
        rows={name: len(frame) for name,frame in frames.items()},
        eligible_recommended_targets=eligible_targets, original_train_rows=original_train_rows,
        unknown_target_fraction={name: float((frame.target==2).mean()) if len(frame) else None for name,frame in frames.items()},
        repeated_target_fraction={name: float(frame.target_seen_before.mean()) if len(frame) else None for name,frame in frames.items()},
        reward_rule='clip(played_ratio_pct/100,0,1); replay beyond 100 retained separately as played_ratio_raw',
        reward_replay_fraction=known_replay_targets/max(1,recommended_targets),
        target_rule='Only recommendation-driven listens (is_organic=0); all strictly earlier listens, including organic, form context.',
        history_rule=f'Last {cfg["max_length"]} listens strictly before target timestamp; contemporaneous listens share one prefix; repeated consumed items remain eligible.',
        return7_label_rule='Activity on a strictly later relative day within 604800 seconds; training window must mature before train cutoff; evaluation window before dataset end.',
        duration_rule='Candidate metadata uses median duration from cohort train listens only; row duration is observed track length for diagnostics, never future fitted metadata.',
        sampling_rule=f'Uniform without replacement per user and period: <= {cfg["train_examples_per_user"]} train targets, <= {cfg["eval_examples_per_user"]} evaluation targets, selected independently of reward. User-macro means target the per-user event distribution; sampling_weight retained for weighted micro analyses.',
        catalog_rule='Most frequent items in all cohort train listens; UNK targets retained as misses; previously consumed music is allowed.',
        feedback_rule='Explicit like/dislike/unlike/undislike events are excluded from item histories and targets; their number is reported in source statistics.')
    manifest['artist_metadata'] = artist_manifest
    return YambdaPreparedData(*(frames[name] for name in SPLITS), items, item_ids, manifest, effective)
