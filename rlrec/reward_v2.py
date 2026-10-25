"""Time-separated completion critics and diagnostics for the second study.

These models predict outcomes conditional on logged consumption. Neither the
completion head nor return7 identifies the causal effect of a recommendation.
"""
from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path
import pickle

from catboost import CatBoostRegressor
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.decomposition import TruncatedSVD
from scipy import sparse


DEFAULTS = {
    'seeds': [17, 29, 43], 'threads': 4, 'learning_rate': .05,
    'grid': [{'depth': 4, 'iterations': 300}, {'depth': 6, 'iterations': 500},
             {'depth': 8, 'iterations': 500}],
    'early_stopping_rounds': 50, 'l2_leaf_reg': 12., 'shrinkage': 25.,
    'diagnostic_iterations': 300, 'bootstrap_samples': 400,
    'history_cache_size': 8192,
    'latent_dim': 24, 'duration_column': None, 'duration_scale': None,
}


def _safe(value):
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    return value


def _correlation(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


class ResearchReward:
    """Stronger critic, a disjoint-user auditor and observational baselines.

    ``support`` is a monotone transform of early-training item counts. It is an
    input-support heuristic, not a calibrated propensity or epistemic interval.
    """
    def __init__(self, config=None):
        self.config = DEFAULTS | (config or {})
        self.seeds = tuple(int(seed) for seed in self.config['seeds'])
        if not self.seeds:
            raise ValueError('At least one reward seed is required')
        self.models = []
        self.audit_model = None
        self.return_model = None
        self.return_history_model = None
        self.item_model = None
        self.history_model = None
        self._history_cache = OrderedDict()

    def _model(self, seed, depth=6, iterations=400):
        return CatBoostRegressor(
            iterations=int(iterations), depth=int(depth),
            learning_rate=float(self.config['learning_rate']), loss_function='RMSE',
            eval_metric='RMSE', random_seed=int(seed),
            thread_count=int(self.config['threads']), verbose=False,
            allow_writing_files=False, l2_leaf_reg=float(self.config['l2_leaf_reg']),
        )

    def _fit_statistics(self, early, items, vocab_size):
        self.vocab_size = int(vocab_size)
        self.item_labels = np.asarray([str(i) for i in range(vocab_size)], dtype=object)
        self.global_mean = float(early.reward.mean())
        self.global_std = float(early.reward.std(ddof=0))
        self.item_count = np.zeros(vocab_size, np.float32)
        self.item_mean = np.full(vocab_size, self.global_mean, np.float32)
        self.item_std = np.full(vocab_size, self.global_std, np.float32)
        self.item_high_completion = np.full(vocab_size, float((early.reward >= .8).mean()), np.float32)
        grouped = early.assign(square=early.reward**2, high=(early.reward >= .8).astype(float)).groupby('target').agg(
            count=('reward', 'size'), total=('reward', 'sum'), square=('square', 'sum'), high=('high', 'sum'))
        alpha = float(self.config['shrinkage'])
        ix = grouped.index.to_numpy(int)
        counts = grouped['count'].to_numpy(float)
        means = (grouped.total.to_numpy(float) + alpha * self.global_mean) / (counts + alpha)
        squares = (grouped.square.to_numpy(float) + alpha * (self.global_std**2 + self.global_mean**2)) / (counts + alpha)
        self.item_count[ix] = counts
        self.item_mean[ix] = means
        self.item_std[ix] = np.sqrt(np.maximum(0, squares - means**2))
        self.item_high_completion[ix] = (grouped.high.to_numpy(float) + alpha * self.item_high_completion[0]) / (counts + alpha)
        items = items.copy()
        for column, default in [('genres', ''), ('content_type', ''), ('release_year', np.nan)]:
            if column not in items:
                items[column] = default
        genres = sorted({g.strip() for text in items.genres.fillna('') for g in str(text).split(',') if g.strip()})
        self.genre_names = genres
        genre_index = {g: j for j, g in enumerate(genres)}
        self.genre = np.zeros((vocab_size, max(1, len(genres))), np.float32)
        self.kind = np.zeros(vocab_size, np.float32)
        self.year = np.zeros(vocab_size, np.float32)
        self.metadata_available = np.zeros(vocab_size, np.float32)
        self.log_duration = np.zeros(vocab_size, np.float32)
        self.duration_seconds = np.zeros(vocab_size, np.float32)
        self.duration_available = np.zeros(vocab_size, np.float32)
        duration_column = self.config.get('duration_column')
        if not duration_column:
            duration_column = next((name for name in ['duration_seconds', 'duration_sec', 'duration_s', 'duration_ms', 'duration'] if name in items), None)
        scale = self.config.get('duration_scale')
        scale = float(scale) if scale is not None else (.001 if duration_column == 'duration_ms' else 1.)
        if duration_column:
            seconds = pd.to_numeric(items[duration_column], errors='coerce').to_numpy(float) * scale
            valid_duration = np.isfinite(seconds) & (seconds > 0)
            tokens = items.token.to_numpy(int)
            valid_duration &= (tokens >= 3) & (tokens < vocab_size)
            self.log_duration[tokens[valid_duration]] = np.log1p(seconds[valid_duration])
            self.duration_seconds[tokens[valid_duration]] = seconds[valid_duration]
            self.duration_available[tokens[valid_duration]] = 1.
        self.duration_manifest = {'column': duration_column, 'scale_to_seconds': scale,
                                  'known_items': int(self.duration_available.sum())}
        self._fit_duration_prior(early)
        for row in items.itertuples():
            token = int(row.token)
            if not 3 <= token < vocab_size:
                continue
            self.metadata_available[token] = 1.
            for value in str(row.genres).split(','):
                if value.strip() in genre_index:
                    self.genre[token, genre_index[value.strip()]] = 1.
            self.kind[token] = float(str(row.content_type) == 'series')
            self.year[token] = np.clip((float(row.release_year) - 1980) / 50, -2, 2) if pd.notna(row.release_year) else 0.
        self.genre_norm = np.maximum(1, self.genre.sum(axis=1))
        self.log_item_count = np.log1p(self.item_count)
        self.support_score = 1. - 1. / np.sqrt(1. + self.item_count)
        self._fit_latent(early)
        self._history_cache.clear()

    def _fit_duration_prior(self, early):
        """Completion prior conditioned on frozen early duration quantile bins.

        Subtracting this prior is an observational adjustment. It does not
        identify an unbiased effect of duration on satisfaction or retention.
        """
        self.duration_prior = np.full(self.vocab_size, self.global_mean, np.float32)
        self.duration_prior_count = np.zeros(self.vocab_size, np.float32)
        targets = early.target.to_numpy(int)
        known = self.duration_available[targets] > 0
        durations = self.duration_seconds[targets[known]].astype(float)
        rewards = early.reward.to_numpy(float)[known]
        alpha = float(self.config['shrinkage'])
        self.duration_thresholds = np.empty(0, float)
        records = []
        boundaries = []
        if len(durations):
            boundaries = np.unique(np.quantile(durations, np.linspace(0, 1, 11))).tolist()
            self.duration_thresholds = np.asarray(boundaries[1:-1], float)
            bin_ids = np.searchsorted(self.duration_thresholds, durations, side='right')
            valid_items = np.flatnonzero(self.duration_available > 0)
            candidate_bins = np.searchsorted(self.duration_thresholds, self.duration_seconds[valid_items], side='right')
            for index in range(len(self.duration_thresholds) + 1):
                mask = bin_ids == index
                count = int(mask.sum())
                total = float(rewards[mask].sum())
                estimate = (total + alpha * self.global_mean) / (count + alpha)
                items_in_bin = valid_items[candidate_bins == index]
                self.duration_prior[items_in_bin] = estimate
                self.duration_prior_count[items_in_bin] = count
                records.append({'bin': index, 'n': count,
                    'lower_seconds': None if index == 0 else float(self.duration_thresholds[index - 1]),
                    'upper_seconds': None if index == len(self.duration_thresholds) else float(self.duration_thresholds[index]),
                    'observed_completion_mean': float(rewards[mask].mean()) if count else None,
                    'smoothed_completion_mean': estimate, 'catalog_items': len(items_in_bin)})
        self.duration_manifest['prior'] = {
            'fit_rule': 'Early-period consumed target pairs only; quantiles and completion means frozen before critic supervised period',
            'quantiles_requested': 10, 'quantile_boundaries_seconds': boundaries,
            'internal_thresholds_seconds': self.duration_thresholds.tolist(),
            'n_early_labels_with_duration': len(durations), 'shrinkage_to_global_mean': alpha,
            'bins': records, 'missing_duration_prior': self.global_mean,
            'outside_observed_range': 'Use the nearest outer bin; this is extrapolation, not evidence of calibration',
            'expected_seconds_rule': 'Predicted bounded completion ratio times frozen item duration; not repeated-listening time',
        }

    def _fit_latent(self, early):
        """Collaborative item factors from strictly earlier consumed items only."""
        dim = max(0, int(self.config['latent_dim']))
        self.latent = np.zeros((self.vocab_size, dim), np.float32)
        user_items = {}
        if dim:
            for row in early.itertuples():
                consumed = user_items.setdefault(int(row.user_id), set())
                consumed.add(int(row.target))
                consumed.update(int(token) for token in row.history if 3 <= int(token) < self.vocab_size)
        rows, columns = [], []
        for index, consumed in enumerate(user_items.values()):
            valid = sorted(token for token in consumed if 3 <= token < self.vocab_size)
            rows.extend([index] * len(valid))
            columns.extend(valid)
        rank = min(dim, max(0, len(user_items) - 1), max(0, self.vocab_size - 4))
        explained = None
        if rank:
            matrix = sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, columns)),
                                       shape=(len(user_items), self.vocab_size))
            # Binary consumption avoids overweighting repeated plays; IDF reduces
            # the leading popularity component without using future counts.
            frequency = np.diff(matrix.tocsc().indptr)
            idf = np.log1p(len(user_items) / (1. + frequency)).astype(np.float32)
            matrix = matrix.multiply(idf[None, :]).tocsr()
            svd = TruncatedSVD(n_components=rank, n_iter=5, random_state=1381)
            svd.fit(matrix)
            factors = svd.components_.T * np.sqrt(svd.singular_values_)[None, :]
            norms = np.linalg.norm(factors, axis=1, keepdims=True)
            self.latent[:, :rank] = np.divide(factors, norms, out=np.zeros_like(factors), where=norms > 1e-8)
            explained = float(svd.explained_variance_ratio_.sum())
        self.latent_manifest = {'requested_dim': dim, 'fitted_dim': rank, 'users': len(user_items),
            'binary_pairs': len(rows), 'explained_variance_ratio': explained,
            'fit_rule': 'Only early-period targets and their strictly earlier histories; binary user-item matrix, early-only IDF weighting'}

    def _summarize_history(self, history):
        key = tuple(int(token) for token in history)
        cached = self._history_cache.get(key)
        if cached is not None:
            self._history_cache.move_to_end(key)
            return cached
        h = np.asarray(key, np.int64)
        known = h[(h >= 3) & (h < self.vocab_size)]
        length = len(known)
        n_genres = self.genre.shape[1]
        genre_profile = np.zeros(n_genres, np.float32)
        recent_genre = np.zeros(n_genres, np.float32)
        last = 0
        latent_mean = np.zeros(self.latent.shape[1], np.float32)
        latent_recent = np.zeros_like(latent_mean)
        if length:
            genre_profile = self.genre[known].mean(axis=0)
            recent = known[-5:]
            recent_genre = self.genre[recent].mean(axis=0)
            last = int(known[-1])
            p = genre_profile / max(float(genre_profile.sum()), 1e-8)
            entropy = float(-(p[p > 0] * np.log(p[p > 0])).sum())
            values = [np.log1p(length), len(set(known.tolist())) / length,
                      1 - length / max(1, len(h)), float(self.item_mean[known].mean()),
                      float(self.item_mean[known].std()), float(self.log_item_count[known].mean()),
                      float(self.log_item_count[known].std()), float(self.kind[known].mean()),
                      float(self.year[known].mean()), float(self.year[known].std()), entropy,
                      float(self.item_mean[recent].mean()), float(self.kind[recent].mean()),
                      float(self.year[recent].mean())]
            duration_values = self.log_duration[known][self.duration_available[known] > 0]
            recent_durations = self.log_duration[recent][self.duration_available[recent] > 0]
            latent_mean = self.latent[known].mean(axis=0)
            latent_recent = self.latent[recent].mean(axis=0)
            values.extend([float(duration_values.mean()) if len(duration_values) else 0.,
                           float(duration_values.std()) if len(duration_values) else 0.,
                           float(recent_durations.mean()) if len(recent_durations) else 0.,
                           float(self.duration_available[known].mean()), float(np.linalg.norm(latent_mean))])
        else:
            values = [0., 0., 1. if len(h) else 0., self.global_mean, 0., 0., 0., 0., 0., 0., 0., self.global_mean, 0., 0.] + [0.] * 5
        values.extend(latent_mean.tolist())
        summary = (np.asarray(values, np.float32), genre_profile, recent_genre, last, frozenset(known.tolist()), latent_mean, latent_recent)
        self._history_cache[key] = summary
        if len(self._history_cache) > int(self.config['history_cache_size']):
            self._history_cache.popitem(last=False)
        return summary

    def features(self, histories, candidates):
        candidates = np.asarray(candidates, np.int64)
        if candidates.ndim != 2 or candidates.shape[0] != len(histories):
            raise ValueError('Candidates must have shape [len(histories), C]')
        if np.any((candidates < 0) | (candidates >= self.vocab_size)):
            raise ValueError('Candidate outside the frozen vocabulary')
        n, width = candidates.shape
        cand = candidates.reshape(-1)
        summaries = [self._summarize_history(history) for history in histories]
        history_values = np.stack([s[0] for s in summaries]) if summaries else np.empty((0, 19 + self.latent.shape[1]), np.float32)
        history_genres = np.stack([s[1] for s in summaries]) if summaries else np.empty((0, self.genre.shape[1]), np.float32)
        recent_genres = np.stack([s[2] for s in summaries]) if summaries else np.empty_like(history_genres)
        last = np.asarray([s[3] for s in summaries], np.int64)
        repeated_h = np.repeat(history_values, width, axis=0)
        repeated_g = np.repeat(history_genres, width, axis=0)
        repeated_recent = np.repeat(recent_genres, width, axis=0)
        repeat_last = np.repeat(last, width)
        mean_latent = np.repeat(np.stack([s[5] for s in summaries]), width, axis=0)
        recent_latent = np.repeat(np.stack([s[6] for s in summaries]), width, axis=0)
        item_latent = self.latent[cand]
        latent_products = mean_latent * item_latent
        latent_dot = latent_products.sum(axis=1)
        latent_cosine = latent_dot / np.maximum(1e-8, np.linalg.norm(mean_latent, axis=1))
        recent_dot = (recent_latent * item_latent).sum(axis=1)
        recent_cosine = recent_dot / np.maximum(1e-8, np.linalg.norm(recent_latent, axis=1))
        target_g = self.genre[cand]
        denominator = self.genre_norm[cand]
        affinity = (repeated_g * target_g).sum(axis=1) / denominator
        recent_affinity = (repeated_recent * target_g).sum(axis=1) / denominator
        last_affinity = (self.genre[repeat_last] * target_g).sum(axis=1) / denominator
        repeat = np.asarray([int(token) in summary[4] for summary, row in zip(summaries, candidates) for token in row], np.float32)
        item_values = np.column_stack([
            self.log_item_count[cand], self.item_mean[cand], self.item_std[cand], self.item_high_completion[cand],
            self.kind[cand], self.year[cand], self.metadata_available[cand], target_g,
            self.log_duration[cand], self.duration_available[cand], item_latent])
        item_names = ['item_log_count', 'item_mean_completion', 'item_completion_std', 'item_high_completion',
                      'item_series', 'item_year', 'item_metadata'] + [f'item_genre_{j}' for j in range(self.genre.shape[1])]
        item_names += ['item_log_duration', 'item_duration_available'] + [f'item_latent_{j}' for j in range(self.latent.shape[1])]
        history_names = ['hist_log_length', 'hist_unique_fraction', 'hist_unknown_fraction', 'hist_item_completion_mean',
                         'hist_item_completion_std', 'hist_log_popularity_mean', 'hist_log_popularity_std', 'hist_series_fraction',
                         'hist_year_mean', 'hist_year_std', 'hist_genre_entropy', 'hist_recent_completion_mean',
                         'hist_recent_series_fraction', 'hist_recent_year_mean', 'hist_log_duration_mean',
                         'hist_log_duration_std', 'hist_recent_log_duration_mean', 'hist_duration_known_fraction',
                         'hist_latent_coherence'] + [f'hist_latent_{j}' for j in range(self.latent.shape[1])]
        interactions = np.column_stack([
            affinity, recent_affinity, last_affinity,
            (self.kind[repeat_last] == self.kind[cand]).astype(np.float32),
            self.item_mean[cand] - repeated_h[:, 3],
            self.log_item_count[cand] - repeated_h[:, 5],
            np.abs(self.year[cand] - repeated_h[:, 8]),
            np.abs(self.kind[cand] - repeated_h[:, 7]), repeat,
            np.abs(self.log_duration[cand] - repeated_h[:, 14]),
            self.log_duration[cand] - repeated_h[:, 16],
            latent_dot, latent_cosine, recent_dot, recent_cosine,
            (self.latent[repeat_last] * item_latent).sum(axis=1), latent_products])
        interaction_names = ['match_genre_all', 'match_genre_recent', 'match_genre_last', 'match_kind_last',
                             'match_completion_delta', 'match_popularity_delta', 'match_year_distance',
                             'match_series_distance', 'match_repeat', 'match_duration_distance',
                             'match_recent_duration_delta', 'match_latent_dot', 'match_latent_cosine',
                             'match_recent_latent_dot', 'match_recent_latent_cosine', 'match_last_latent_cosine']
        interaction_names += [f'match_latent_product_{j}' for j in range(self.latent.shape[1])]
        x = pd.DataFrame(np.column_stack([item_values, repeated_h, interactions]).astype(np.float32),
                         columns=item_names + history_names + interaction_names)
        # Categorical IDs are strings: the token number has no ordinal meaning.
        x.insert(0, 'item_token', self.item_labels[cand])
        self.item_feature_names = ['item_token'] + item_names
        self.history_feature_names = history_names
        self.interaction_feature_names = interaction_names
        return x

    @staticmethod
    def _weights(frame, by_day=False):
        if by_day:
            counts = frame.groupby(['user_id', 'day']).user_id.transform('size').to_numpy(float)
        else:
            counts = frame.groupby('user_id').user_id.transform('size').to_numpy(float)
        weights = 1. / np.maximum(1, counts)
        return weights / weights.mean()

    def fit(self, train, rm_valid, items, vocab_size):
        labeled = train[(train.target >= 3) & train.reward.notna()].copy()
        validation = rm_valid[(rm_valid.target >= 3) & rm_valid.reward.notna()].copy()
        if len(labeled) < 60 or len(validation) < 10:
            raise ValueError('Need at least 60 train and 10 validation completion labels')
        time_column = 'timestamp' if 'timestamp' in labeled and 'timestamp' in validation else 'day'
        if labeled[time_column].isna().any() or validation[time_column].isna().any():
            raise ValueError(f'Reward training and validation require nonmissing {time_column}')
        latest_training_time = labeled[time_column].max()
        earliest_validation_time = validation[time_column].min()
        if earliest_validation_time <= latest_training_time:
            raise ValueError(f'Reward validation {time_column} must follow all training {time_column}')
        self.stats_cutoff = int(np.median(labeled.day.to_numpy()))
        early = labeled[labeled.day < self.stats_cutoff]
        supervised = labeled[labeled.day >= self.stats_cutoff]
        if len(early) < 10:
            raise ValueError('Need a nonempty strictly earlier statistics period')
        self._fit_statistics(early, items, vocab_size)
        users = np.sort(supervised.user_id.unique())
        shuffled = np.random.default_rng(1381).permutation(users)
        audit_users = set(shuffled[:max(1, len(users) // 3)].tolist())
        optimize = supervised[~supervised.user_id.isin(audit_users)]
        audit = supervised[supervised.user_id.isin(audit_users)]
        if min(len(optimize), len(audit)) < 15:
            raise ValueError('Need two nontrivial disjoint-user supervised partitions')
        x = self.features(optimize.history.tolist(), optimize.target.to_numpy()[:, None])
        xv = self.features(validation.history.tolist(), validation.target.to_numpy()[:, None])
        y, yv = optimize.reward.to_numpy(float), validation.reward.to_numpy(float)
        # User-balanced fitting aligns the critic with the policy's user-macro target.
        weights = self._weights(optimize)
        self.tuning_history = []
        winner = None
        for spec in self.config['grid']:
            candidate = self._model(self.seeds[0], **spec)
            candidate.fit(x, y, cat_features=['item_token'], sample_weight=weights,
                          eval_set=(xv, yv), use_best_model=True,
                          early_stopping_rounds=int(self.config['early_stopping_rounds']))
            pv = np.clip(candidate.predict(xv), 0, 1)
            rmse = float(np.sqrt(mean_squared_error(yv, pv)))
            record = dict(depth=int(spec['depth']), iterations=int(spec['iterations']),
                          selected_iterations=int(candidate.tree_count_), validation_rmse=rmse)
            self.tuning_history.append(record)
            if winner is None or rmse < winner[0]:
                winner = (rmse, candidate, record)
        _, first, chosen = winner
        self.selected_spec = {'depth': chosen['depth'], 'iterations': chosen['selected_iterations']}
        self.models = [first]
        for seed in self.seeds[1:]:
            model = self._model(seed, **self.selected_spec)
            model.fit(x, y, cat_features=['item_token'], sample_weight=weights)
            self.models.append(model)
        xa = self.features(audit.history.tolist(), audit.target.to_numpy()[:, None])
        self.audit_model = self._model(191, depth=max(3, self.selected_spec['depth'] - 1),
                                       iterations=self.selected_spec['iterations'])
        self.audit_model.fit(xa, audit.reward.to_numpy(float), cat_features=['item_token'], sample_weight=self._weights(audit))
        diagnostic_iterations = int(self.config['diagnostic_iterations'])
        self.item_model = self._model(281, **self.selected_spec)
        self.item_model.fit(x[self.item_feature_names], y, cat_features=['item_token'], sample_weight=weights)
        self.history_model = self._model(391, **self.selected_spec)
        self.history_model.fit(x[self.history_feature_names], y, sample_weight=weights)
        ret = optimize[optimize.return7.notna()].copy()
        ret_weights = self._weights(ret, by_day=True) if len(ret) else np.empty(0, float)
        self.return_baseline = float(np.average(ret.return7.to_numpy(float), weights=ret_weights)) if len(ret) else .5
        if not np.isfinite(self.return_baseline):
            self.return_baseline = .5
        self.return_model = None
        self.return_history_model = None
        if len(ret) >= 30 and ret.return7.nunique() > 1:
            xr = self.features(ret.history.tolist(), ret.target.to_numpy()[:, None])
            self.return_model = self._model(719, depth=5, iterations=diagnostic_iterations)
            self.return_model.fit(xr, ret.return7.to_numpy(float), cat_features=['item_token'],
                                  sample_weight=ret_weights)
            self.return_history_model = self._model(719, depth=5, iterations=diagnostic_iterations)
            self.return_history_model.fit(xr[self.history_feature_names], ret.return7.to_numpy(float),
                                          sample_weight=ret_weights)
        importance = np.mean([m.feature_importances_ for m in self.models], axis=0)
        self.feature_importance = dict(zip(x.columns, map(float, importance)))
        self.feature_groups = {
            'item_only': float(sum(self.feature_importance[c] for c in self.item_feature_names)),
            'history_only': float(sum(self.feature_importance[c] for c in self.history_feature_names)),
            'history_item_interaction': float(sum(self.feature_importance[c] for c in self.interaction_feature_names)),
        }
        self.training_manifest = {
            'stats_cutoff': self.stats_cutoff, 'stats_rows': len(early), 'supervised_rows': len(supervised),
            'optimizer_rows': len(optimize), 'audit_rows': len(audit),
            'optimizer_users': int(optimize.user_id.nunique()), 'audit_users': int(audit.user_id.nunique()),
            'return_rows': len(ret), 'return_user_days': int(ret.drop_duplicates(['user_id', 'day']).shape[0]),
            'temporal_validation_column': time_column,
            'latest_training_time': _safe(latest_training_time),
            'earliest_validation_time': _safe(earliest_validation_time),
            'seeds': list(self.seeds), 'selected_spec': self.selected_spec, 'tuning': self.tuning_history,
            'completion_training_weights': 'Equal total weight per optimizer user',
            'return_training_weights': 'Equal total weight per user-day; full, history-only and constant use identical rows, labels and weights',
            'return_head_comparison': 'Full and history-only use depth 5, identical iteration budget and random seed 719; only input columns differ',
            'statistics_rule': 'Statistics days strictly earlier than supervised days; no target outcome is a feature',
            'audit_rule': 'Disjoint supervised users; shared early statistics and observational bias',
            'support_rule': '1 - 1/sqrt(1 + early item count); not a calibrated propensity or confidence score',
            'uncertainty_rule': 'Ensemble standard deviation is disagreement, not a calibrated confidence interval',
            'feature_importance_groups': self.feature_groups,
            'collaborative_features': self.latent_manifest, 'duration_features': self.duration_manifest,
            'completion_ablation_capacity': 'Item-only and history-only models use the selected full-model depth and number of trees',
        }
        self.validation_metrics = self.evaluate(validation)
        return self

    def score_candidates(self, histories, candidates):
        candidates = np.asarray(candidates, np.int64)
        shape = candidates.shape
        if candidates.ndim != 2:
            raise ValueError('Candidates must be a two-dimensional token array')
        if candidates.size == 0:
            return {key: np.empty(shape, np.float32) for key in ['mean', 'std', 'audit', 'return7', 'support', 'return7_history',
                                                               'duration_seconds', 'duration_prior', 'expected_seconds']}
        x = self.features(histories, candidates)
        predictions = np.stack([model.predict(x) for model in self.models])
        predicted_mean = np.clip(predictions.mean(axis=0), 0, 1)
        candidate_durations = self.duration_seconds[candidates.reshape(-1)]
        values = {
            'mean': predicted_mean,
            'std': predictions.std(axis=0),
            'audit': np.clip(self.audit_model.predict(x), 0, 1),
            'return7': np.clip(self.return_model.predict(x), 0, 1) if self.return_model is not None else np.full(len(x), self.return_baseline),
            'support': self.support_score[candidates.reshape(-1)],
            'return7_history': np.clip(self.return_history_model.predict(x[self.history_feature_names]), 0, 1)
                if self.return_history_model is not None else np.full(len(x), self.return_baseline),
            'duration_seconds': candidate_durations,
            'duration_prior': self.duration_prior[candidates.reshape(-1)],
            'expected_seconds': predicted_mean * candidate_durations,
        }
        result = {key: np.asarray(value, np.float32).reshape(shape) for key, value in values.items()}
        if not all(np.isfinite(array).all() for array in result.values()):
            raise FloatingPointError('Reward critic produced nonfinite values')
        return result

    def predict(self, histories, targets):
        return {key: array[:, 0] for key, array in self.score_candidates(histories, np.asarray(targets)[:, None]).items()}

    @staticmethod
    def _metrics(y, prediction, users=None):
        y, prediction = np.asarray(y, float), np.asarray(prediction, float)
        result = {
            'mae': float(mean_absolute_error(y, prediction)),
            'rmse': float(np.sqrt(mean_squared_error(y, prediction))),
            'r2': float(r2_score(y, prediction)) if len(y) > 1 else None,
            'pearson': _correlation(y, prediction),
        }
        if users is not None:
            errors = pd.DataFrame({'user_id': np.asarray(users), 'absolute': np.abs(y - prediction), 'square': (y - prediction)**2})
            grouped = errors.groupby('user_id')[['absolute', 'square']].mean()
            result.update(user_macro_mae=float(grouped.absolute.mean()), user_macro_rmse=float(np.sqrt(grouped.square.mean())))
        return result

    def _bootstrap(self, y, prediction, users):
        errors = pd.DataFrame({'user_id': np.asarray(users), 'absolute': np.abs(y - prediction), 'square': (y - prediction)**2})
        grouped = errors.groupby('user_id')[['absolute', 'square']].mean().to_numpy()
        rng = np.random.default_rng(842)
        count = int(self.config['bootstrap_samples'])
        if count < 1:
            return {'unit': 'user_cluster_macro', 'draws': 0}
        values = np.empty((count, 2))
        for index in range(count):
            draw = grouped[rng.integers(0, len(grouped), size=len(grouped))].mean(axis=0)
            values[index] = [draw[0], np.sqrt(draw[1])]
        return {'unit': 'user_cluster_macro', 'draws': count,
                'mae_95ci': np.quantile(values[:, 0], [.025, .975]).tolist(),
                'rmse_95ci': np.quantile(values[:, 1], [.025, .975]).tolist()}

    def _sensitivity(self, frame):
        # Fixed random actions isolate within-context action dependence from the
        # variation of baseline user activity across different histories.
        columns = ['user_id', 'day'] if 'day' in frame else ['user_id']
        chosen = frame.drop_duplicates(columns).head(128)
        if chosen.empty:
            return {'n_contexts': 0}
        rng = np.random.default_rng(957)
        candidates = rng.integers(3, self.vocab_size, size=(len(chosen), 16))
        candidates[:, 0] = chosen.target.to_numpy(int)
        scores = self.score_candidates(chosen.history.tolist(), candidates)
        result = {'n_contexts': len(chosen), 'candidates_per_context': 16}
        for head in ['mean', 'return7', 'return7_history']:
            values = scores[head].astype(float)
            within = float(np.var(values, axis=1).mean())
            between = float(np.var(values.mean(axis=1)))
            result[head] = {
                'within_context_candidate_std': float(np.std(values, axis=1).mean()),
                'between_history_std': float(np.std(values.mean(axis=1))),
                'candidate_variance_fraction': within / max(within + between, 1e-12),
                'same_history_candidate_permutation_mae': float(np.abs(values[:, 0] - values[:, 1]).mean()),
            }
        result['interpretation'] = 'Observational action sensitivity, not a causal effect of an item on return'
        return result

    def evaluate(self, frame):
        known = frame[(frame.target >= 3) & (frame.target < self.vocab_size)].copy()
        labeled = known[known.reward.notna()].copy()
        if labeled.empty:
            return {'n': 0, 'return7_n': int(known.return7.notna().sum())}
        histories, targets = labeled.history.tolist(), labeled.target.to_numpy(int)
        x = self.features(histories, targets[:, None])
        predictions = self.predict(histories, targets)
        y = labeled.reward.to_numpy(float)
        users = labeled.user_id.to_numpy()
        baseline_predictions = {
            'ensemble': predictions['mean'], 'audit': predictions['audit'],
            'early_item_mean': self.item_mean[targets],
            'constant': np.full(len(y), self.global_mean),
            'item_only_model': np.clip(self.item_model.predict(x[self.item_feature_names]), 0, 1),
            'history_only_model': np.clip(self.history_model.predict(x[self.history_feature_names]), 0, 1),
        }
        result = {'n': len(labeled), 'n_users': int(labeled.user_id.nunique()),
                  **{key: self._metrics(y, prediction, users) for key, prediction in baseline_predictions.items()}}
        result['ensemble_user_bootstrap'] = self._bootstrap(y, predictions['mean'], users)
        bins = np.minimum((np.clip(predictions['mean'], 0, 1) * 10).astype(int), 9)
        calibration = []
        for index in range(10):
            mask = bins == index
            calibration.append({'lower': index / 10, 'upper': (index + 1) / 10, 'n': int(mask.sum()),
                'predicted_mean': float(predictions['mean'][mask].mean()) if mask.any() else None,
                'observed_mean': float(y[mask].mean()) if mask.any() else None})
        result['completion_calibration'] = calibration
        result['completion_calibration_gap'] = float(sum(row['n'] * abs(row['observed_mean'] - row['predicted_mean'])
                                                      for row in calibration if row['n']) / len(y))
        absolute = np.abs(y - predictions['mean'])
        disagreement = predictions['std']
        result['ensemble_disagreement_diagnostics'] = {
            'mean_std': float(disagreement.mean()),
            'std_vs_absolute_error_pearson': _correlation(disagreement, absolute),
            'std_vs_absolute_error_spearman': _correlation(pd.Series(disagreement).rank(), pd.Series(absolute).rank()),
            'naive_1_96std_coverage': float((absolute <= 1.96 * disagreement).mean()),
            'interpretation': 'Coverage is diagnostic only; ensemble spread is not a valid confidence interval',
        }
        counts = self.item_count[targets]
        support_bins = []
        for label, lo, hi in [('zero', 0, 1), ('1-4', 1, 5), ('5-19', 5, 20), ('20-99', 20, 100), ('100+', 100, np.inf)]:
            mask = (counts >= lo) & (counts < hi)
            support_bins.append({'early_item_count': label, 'n': int(mask.sum()),
                'mae': float(absolute[mask].mean()) if mask.any() else None,
                'rmse': float(np.sqrt(np.mean((y[mask] - predictions['mean'][mask])**2))) if mask.any() else None,
                'mean_ensemble_std': float(disagreement[mask].mean()) if mask.any() else None})
        result['support_error_bins'] = support_bins
        duration_known = self.duration_available[targets] > 0
        if duration_known.any():
            duration = self.duration_seconds[targets].astype(float)
            log_duration = np.log1p(duration)
            result['duration_diagnostics'] = {
                'n': int(duration_known.sum()),
                'duration_vs_predicted_completion_pearson': _correlation(duration[duration_known], predictions['mean'][duration_known]),
                'duration_vs_observed_completion_pearson': _correlation(duration[duration_known], y[duration_known]),
                'log_duration_vs_predicted_completion_pearson': _correlation(log_duration[duration_known], predictions['mean'][duration_known]),
                'log_duration_vs_observed_completion_pearson': _correlation(log_duration[duration_known], y[duration_known]),
                'duration_prior': self._metrics(y[duration_known], predictions['duration_prior'][duration_known], users[duration_known]),
                'outside_early_duration_range_fraction': float(np.mean((duration[duration_known] < self.duration_manifest['prior']['quantile_boundaries_seconds'][0]) |
                    (duration[duration_known] > self.duration_manifest['prior']['quantile_boundaries_seconds'][-1])))
                    if self.duration_manifest['prior']['quantile_boundaries_seconds'] else None,
                'bins': [],
                'interpretation': 'Observed consumed pairs only; expected seconds uses clipped single-duration completion and is not actual listening time',
            }
            duration_bins = np.searchsorted(self.duration_thresholds, duration, side='right')
            for index in range(len(self.duration_thresholds) + 1):
                mask = duration_known & (duration_bins == index)
                if not mask.any():
                    continue
                result['duration_diagnostics']['bins'].append({
                    'bin': index, 'n': int(mask.sum()), 'mean_seconds': float(duration[mask].mean()),
                    'observed_completion_mean': float(y[mask].mean()),
                    'predicted_completion_mean': float(predictions['mean'][mask].mean()),
                    'frozen_duration_prior_mean': float(predictions['duration_prior'][mask].mean()),
                    'rmse': float(np.sqrt(np.mean((y[mask] - predictions['mean'][mask])**2))),
                })
        else:
            result['duration_diagnostics'] = {'n': 0, 'reason': 'No frozen item durations available'}
        per_item = pd.DataFrame({'target': targets, 'y': y, 'prediction': predictions['mean'],
                                 'absolute': absolute, 'square': (y - predictions['mean'])**2}).groupby('target').agg(
            n=('y', 'size'), observed_mean=('y', 'mean'), predicted_mean=('prediction', 'mean'),
            mae=('absolute', 'mean'), mse=('square', 'mean'))
        per_item['rmse'] = np.sqrt(per_item.mse)
        result['frequent_item_metrics'] = per_item.sort_values(['n', 'target'], ascending=[False, True]).head(50).drop(columns='mse').reset_index().to_dict('records')
        result['feature_importance_groups'] = self.feature_groups
        result['feature_importance'] = self.feature_importance
        ret = known[known.return7.notna()].copy()
        if len(ret):
            pr = self.predict(ret.history.tolist(), ret.target.to_numpy(int))
            yr = ret.return7.to_numpy(float)
            day_counts = ret.groupby(['user_id', 'day']).user_id.transform('size').to_numpy(float)
            weights = 1 / day_counts
            def ret_metrics(values):
                return {'brier': float(np.average((yr - values)**2, weights=weights)),
                        'auc': float(roc_auc_score(yr, values, sample_weight=weights)) if len(np.unique(yr)) > 1 else None}
            result['return7'] = {'n': len(ret), 'n_user_days': int(ret.drop_duplicates(['user_id', 'day']).shape[0]),
                'weighting': 'Equal total weight per user-day',
                'full': ret_metrics(pr['return7']), 'history_only': ret_metrics(pr['return7_history']),
                'constant': ret_metrics(np.full(len(yr), self.return_baseline)),
                'label_mean': float(np.average(yr, weights=weights)),
                'training_label_mean': self.return_baseline,
                # Aliases retain the familiar headline schema of v1.
                **ret_metrics(pr['return7']),
                'baseline_brier': ret_metrics(np.full(len(yr), self.return_baseline))['brier']}
        else:
            result['return7'] = {'n': 0, 'n_user_days': 0, 'reason': 'No mature return7 labels in this interval'}
        result['action_sensitivity'] = self._sensitivity(labeled)
        result['interpretation'] = 'Observed consumed pairs only; no click probability or causal policy value'
        return _safe(result)

    def save(self, directory):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self._history_cache.clear()
        with (path / 'reward.pkl').open('wb') as handle:
            pickle.dump(self, handle)
        (path / 'reward_manifest.json').write_text(json.dumps(_safe(self.training_manifest), indent=2), encoding='utf-8')
        (path / 'reward_validation.json').write_text(json.dumps(_safe(self.validation_metrics), indent=2), encoding='utf-8')

    @classmethod
    def load(cls, directory):
        with (Path(directory) / 'reward.pkl').open('rb') as handle:
            return pickle.load(handle)
