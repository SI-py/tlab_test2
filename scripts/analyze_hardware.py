"""Compare CPU- and GPU-trained KION policies for one matched training seed.

The experiment is descriptive: it does not change model selection or add
statistical tests. Both checkpoint versions are evaluated on the same CPU.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import torch

from rlrec.baselines_v2 import ItemKNN
from rlrec.policy_v2 import load_policy, logits_function
from rlrec.research import evaluator
from rlrec.reward_v2 import ResearchReward

FAMILIES = ('matched_control', 'raw_h5', 'anchored_h1', 'anchored_h5',
            'appended_h5', 'conservative_h5', 'retention_h5')
METRICS = ('ndcg@10', 'factual_completion_weighted_hit@10')
SPLITS = ('test', 'confirmation')
SEEDS = (11, 22, 33)



def replacement_summary(cpu_by_seed, gpu33):
    primary_mean = float(np.mean([cpu_by_seed[seed] for seed in SEEDS]))
    replacement = float(np.mean([cpu_by_seed[11], cpu_by_seed[22], gpu33]))
    delta = float(gpu33 - cpu_by_seed[33])
    return {'primary_cpu_seed33': cpu_by_seed[33], 'secondary_gpu_seed33': float(gpu33),
            'delta_gpu_minus_cpu_seed33': delta, 'primary_three_seed_mean': primary_mean,
            'sensitivity_replacement_mean': replacement, 'delta_three_seed_mean': delta / 3,
            'unchanged_seed11': cpu_by_seed[11], 'unchanged_seed22': cpu_by_seed[22]}



def analyze(primary_directory, gpu_directory, predictions_output=None):
    primary, gpu = Path(primary_directory), Path(gpu_directory)
    config = json.loads((primary / 'config.json').read_text())
    results = json.loads((primary / 'results.json').read_text())
    with (primary / 'prepared.pkl').open('rb') as stream:
        data = pickle.load(stream)
    torch.set_num_threads(config.get('threads', 4))
    reward = ResearchReward.load(primary / 'reward')
    knn = ItemKNN.load(primary / 'itemknn.pkl')
    assess, _ = evaluator(config, data, reward, knn)
    predictions = Path(predictions_output) if predictions_output else None
    if predictions:
        predictions.mkdir(parents=True, exist_ok=True)
    output = {'dataset': 'kion', 'analysis_role': 'Descriptive one-seed software and hardware sensitivity',
              'interpretation': 'CPU and CUDA training also differ in runtime versions and sampling. This is not an isolated hardware effect.',
              'inference_scope': 'One matched seed; no new intervals, p-values, noninferiority tests or model selection.',
              'splits': {}}
    rows = []
    for split in SPLITS:
        observed = getattr(data, split).reset_index(drop=True)
        families = {}
        for family in FAMILIES:
            model, _ = load_policy(gpu / 'checkpoints' / f'{family}_33.pt', 'cpu')
            summary, per = assess(logits_function(model, 'cpu'), observed)
            del model
            metrics = {}
            for metric in METRICS:
                cpu = {seed: results['models'][f'{family}_{seed}'][split][metric] for seed in SEEDS}
                metrics[metric] = replacement_summary(cpu, summary[metric])
                rows.append({'dataset': 'kion', 'split': split, 'family': family,
                             'metric': metric, **metrics[metric]})
            families[family] = {'metrics': metrics}
            if predictions:
                per.to_pickle(predictions / f'{family}_33_{split}.pkl')
            print(f'{split}: {family}', flush=True)
        output['splits'][split] = {'n_examples': len(observed),
                                  'n_users': int(observed.user_id.nunique()), 'families': families}
    return output, rows



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--primary', required=True, help='Primary KION experiment directory')
    parser.add_argument('--gpu-source', required=True, help='Directory containing GPU-trained seed-33 checkpoints')
    parser.add_argument('--output', default='results')
    parser.add_argument('--predictions-output', help='Optional directory for per-row predictions')
    args = parser.parse_args()
    result, rows = analyze(args.primary, args.gpu_source, args.predictions_output)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'hardware_replication.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    pd.DataFrame(rows).to_csv(output / 'hardware_replication.csv', index=False)
    print(f'Saved results to {output}')


if __name__ == '__main__':
    main()
