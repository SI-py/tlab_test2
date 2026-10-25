"""Recommend from a completed experiment and an ordered JSON array of raw item IDs.

Input history is supplied by the caller and must contain only interactions before
--cutoff. Scores are ranking scores, never calibrated click probabilities.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import re

import numpy as np
import pandas as pd

from rlrec.baselines_v2 import ItemKNN
from rlrec.evaluate_v2 import RewardReranker
from rlrec.policy_v2 import load_policy, logits_function
from rlrec.reward_v2 import ResearchReward


def load_run(run):
    run = Path(run)
    config = json.loads((run / 'config.json').read_text())
    manifest = json.loads((run / 'data_manifest.json').read_text())
    selection = json.loads((run / 'frozen_selection.json').read_text())
    return config, manifest, selection


def validate_cutoff(cutoff, config, manifest):
    if config['dataset'] == 'yambda':
        if not re.fullmatch(r'\d+', str(cutoff)):
            raise ValueError('Yambda --cutoff must be an integer relative timestamp in seconds')
        query = int(cutoff)
        effective = manifest.get('config', {})
        boundary = effective.get('valid_end_timestamp')
        if boundary is None:
            boundary = manifest.get('temporal_boundaries_seconds', {}).get('valid')
        if boundary is None:
            raise ValueError('Yambda validation endpoint is missing from the manifest')
        if query < int(boundary):
            raise ValueError(f'Cutoff {query} precedes model-selection availability at {boundary} relative seconds')
        return query
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', str(cutoff)):
        raise ValueError('KION --cutoff must be a YYYY-MM-DD date')
    query = pd.Timestamp(cutoff)
    boundary = max(pd.Timestamp(config[key]) for key in ('train_end', 'rm_valid_end', 'valid_end'))
    if query < boundary:
        raise ValueError(f'Cutoff {cutoff} precedes model-selection availability at {boundary.date()}')
    return str(query.date())


def read_history(path):
    values = json.loads(Path(path).read_text())
    if not isinstance(values, list) or any(type(value) is not int or value < 0 for value in values):
        raise ValueError('--history-json must contain an ordered JSON array of nonnegative integer raw item IDs')
    return values


def resolve_family(requested, selection, config):
    if requested in ('selected_overall', 'selected'):
        family = selection.get('selected_overall')
    elif requested == 'selected_rl':
        family = selection.get('selected_rl')
        if family is None:
            raise ValueError('No RL family passed the development guardrail')
    else:
        family = 'itemknn' if requested == 'knn' else requested
    declared = {'baseline', 'reranker', 'itemknn', 'popularity'} | {variant['name'] for variant in config['variants']}
    if family not in declared or not re.fullmatch(r'[A-Za-z0-9_-]+', str(family)):
        raise ValueError(f'Unknown model family {family!r}; available: ' + ', '.join(sorted(declared)))
    return family


def neural_callback(run, family, seed, device):
    model, _ = load_policy(Path(run) / 'checkpoints' / f'{family}_{seed}.pt', device)
    return logits_function(model, device)


def rank_scores(scores, history, k, exclude_seen):
    scores = np.asarray(scores, float)
    if scores.ndim != 1 or len(scores) < 4 or not np.isfinite(scores[3:]).all():
        raise ValueError('Model returned invalid real-item ranking scores')
    if k < 1:
        raise ValueError('--k must be positive')
    eligible = np.ones(len(scores), bool)
    eligible[:3] = False
    if exclude_seen:
        eligible[[int(token) for token in set(history) if 3 <= int(token) < len(scores)]] = False
    candidates = np.flatnonzero(eligible)
    if not len(candidates):
        raise ValueError('No eligible real items remain')
    order = np.lexsort((candidates, -scores[candidates]))[:min(k, len(candidates))]
    return candidates[order], scores[candidates[order]]


def recommend(run, history_json, cutoff, model='selected_overall', seed=11, k=10, device='cpu'):
    run = Path(run)
    config, manifest, selection = load_run(run)
    cutoff = validate_cutoff(cutoff, config, manifest)
    if int(seed) not in [int(value) for value in config['sft_seeds']]:
        raise ValueError('--seed must be one of the prespecified SFT seeds')
    family = resolve_family(model, selection, config)
    item_ids = manifest['item_ids']
    vocab_size = len(item_ids)
    lookup = {int(raw): token for token, raw in enumerate(item_ids) if token >= 3}
    raw_history = read_history(history_json)
    history = [lookup.get(int(raw), 2) for raw in raw_history]
    if config['dataset'] == 'yambda':
        history = history[-int(config['max_length']):]
    reward = ResearchReward.load(run / 'reward')
    if int(reward.vocab_size) != vocab_size:
        raise ValueError('Reward/frozen vocabulary size mismatch')
    if family == 'itemknn':
        callback = ItemKNN.load(run / 'itemknn.pkl')
        if callback.vocab_size != vocab_size:
            raise ValueError('Item-kNN/frozen vocabulary size mismatch')
    elif family == 'popularity':
        if config['dataset'] == 'yambda':
            # The experiment uses cohort-train listen counts, not distinct-user
            # counts of the separate item-kNN model.
            with (run / 'prepared.pkl').open('rb') as handle:
                prepared = pickle.load(handle)
            if prepared.item_ids != item_ids:
                raise ValueError('Prepared popularity catalog does not match the run')
            popularity = np.zeros(vocab_size, float)
            tokens = prepared.items.token.to_numpy(int)
            if not np.array_equal(prepared.items.item_id.to_numpy(int), np.asarray(item_ids)[tokens]):
                raise ValueError('Prepared item metadata/token mapping differs from the catalog')
            popularity[tokens] = prepared.items.train_count.to_numpy(float)
            del prepared
        else:
            knn = ItemKNN.load(run / 'itemknn.pkl')
            if knn.vocab_size != vocab_size:
                raise ValueError('Popularity source/frozen vocabulary size mismatch')
            popularity = knn.item_user_count.astype(float)
        callback = lambda histories: np.tile(np.log1p(popularity), (len(histories), 1))
    elif family == 'reranker':
        if k > int(config.get('rerank_top_m', 200)):
            raise ValueError('--k exceeds the reranker candidate budget')
        base = neural_callback(run, 'baseline', seed, device)
        parameters = selection['reranker']
        callback = RewardReranker(base, reward, vocab_size, parameters['alpha'],
            top_m=config.get('rerank_top_m', 200), reward_type=parameters['reward_type'],
            exclude_seen=bool(config.get('exclude_seen', True)))
    else:
        callback = neural_callback(run, family, seed, device)
    values = np.asarray(callback([history]), float)
    if values.shape != (1, vocab_size):
        raise ValueError('Ranking callback returned an inconsistent catalog')
    tokens, scores = rank_scores(values[0], history, k, bool(config.get('exclude_seen', True)))
    heads = reward.score_candidates([history], tokens[None, :])
    return pd.DataFrame({
        'dataset': config['dataset'], 'model': family,
        'seed': seed if family not in ('popularity', 'itemknn') else None,
        'cutoff': cutoff, 'rank': np.arange(1, len(tokens) + 1),
        'item_id': [int(item_ids[token]) for token in tokens], 'score': scores,
        'predicted_completion': heads['mean'][0], 'ensemble_disagreement': heads['std'][0],
        'audit_completion': heads['audit'][0], 'predicted_return7': heads['return7'][0],
        'support_heuristic': heads['support'][0], 'duration_seconds': heads['duration_seconds'][0],
        'duration_prior': heads['duration_prior'][0], 'expected_seconds_proxy': heads['expected_seconds'][0],
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--model', default='selected_overall')
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--history-json', required=True)
    parser.add_argument('--cutoff', required=True)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'mps'])
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        frame = recommend(args.run, args.history_json, args.cutoff, args.model, args.seed, args.k, args.device)
    except (ValueError, FileNotFoundError, KeyError, RuntimeError) as error:
        parser.error(str(error))
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f'Saved {len(frame)} ranked items to {path}')


if __name__ == '__main__':
    main()
