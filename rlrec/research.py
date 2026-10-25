"""Two-dataset study with development-only selection and later confirmation."""
from __future__ import annotations
import argparse
import copy
import importlib.metadata
import json
import pickle
import shutil
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .data_v2 import prepare_kion
from .policy_v2 import make_model,save_policy,load_policy,seed_everything,logits_function
from .train_v2 import supervised_train,rl_finetune
from .reward_v2 import ResearchReward


def safe(value):
    if isinstance(value,dict):return {str(k):safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [safe(v) for v in value]
    if isinstance(value,(np.integer,)):return int(value)
    if isinstance(value,(float,np.floating)):return float(value) if np.isfinite(value) else None
    return value


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(safe(value),ensure_ascii=False,indent=2,allow_nan=False));temporary.replace(path)


def prepared(config,output,raw_dir):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'config.json').exists() and json.loads((output/'config.json').read_text())!=config:
        raise ValueError('Output directory belongs to another configuration')
    write_json(output/'config.json',config)
    cache=output/'prepared.pkl'
    if cache.exists():
        with cache.open('rb') as f:return pickle.load(f)
    start=time.monotonic()
    if config.get('dataset')=='yambda':
        from .yambda import prepare_yambda
        data=prepare_yambda(Path(raw_dir)/'yambda50m_multi.parquet',config)
    else:data=prepare_kion(raw_dir,config)
    with cache.open('wb') as f:pickle.dump(data,f)
    write_json(output/'data_manifest.json',data.manifest)
    write_json(output/'preparation.json',{'seconds':time.monotonic()-start})
    return data


def fit_sft(config,data,output):
    output=Path(output);device=runtime_device(config);seed0=config['sft_seeds'][0];entries={};models_dir=output/'sft';models_dir.mkdir(exist_ok=True)
    selection_path=output/'sft_selection.json'
    if selection_path.exists():
        selection=json.loads(selection_path.read_text());selected=selection['selected'];entries=selection['architecture_search']
    else:
        for architecture in config['architectures']:
            name=architecture['name'];cfg=config|architecture|{'seed':seed0}
            path=models_dir/f'{name}_{seed0}.pt';log_path=output/'training'/f'sft_{name}_{seed0}.json'
            if not path.exists():
                seed_everything(seed0);model=make_model(data.vocab_size,cfg).to(device)
                print('SFT START',name,seed0,flush=True);start=time.monotonic()
                model,logs=supervised_train(model,data.train,data.rm_valid,cfg,device=device,callback=lambda x:print('SFT',name,seed0,x,flush=True))
                write_json(log_path,logs)
                save_policy(model,path,{'trained_until':config.get('train_end','relative-time'), 'seed':seed0,'architecture':architecture,
                    'training_seconds':time.monotonic()-start})
                del model
            logs=json.loads(log_path.read_text());score=max(row['validation']['ndcg@10'] for row in logs)
            entries[name]={'architecture':architecture,'ndcg@10':score,'best_epoch':int(max(logs,key=lambda x:x['validation']['ndcg@10'])['epoch'])}
            write_json(output/'sft_search_progress.json',entries)
        selected=max(entries,key=lambda name:entries[name]['ndcg@10'])
        write_json(selection_path,{'selected':selected,'architecture_search':entries,'criterion':'RM-validation user-macro NDCG@10; first prespecified seed only'})
    architecture=entries[selected]['architecture']
    checkpoints=output/'checkpoints';checkpoints.mkdir(exist_ok=True)
    for seed in config['sft_seeds']:
        destination=checkpoints/f'baseline_{seed}.pt'
        if destination.exists():continue
        if seed==seed0:shutil.copy2(models_dir/f'{selected}_{seed}.pt',destination);continue
        cfg=config|architecture|{'seed':seed};seed_everything(seed);model=make_model(data.vocab_size,cfg).to(device)
        print('SFT FINAL',selected,seed,flush=True);start=time.monotonic()
        model,logs=supervised_train(model,data.train,data.rm_valid,cfg,device=device,callback=lambda x:print('SFT',selected,seed,x,flush=True))
        write_json(output/'training'/f'baseline_{seed}.json',logs)
        save_policy(model,destination,{'trained_until':config.get('train_end','relative-time'),'seed':seed,'architecture':architecture,
            'training_seconds':time.monotonic()-start})
        del model
    print('SFT COMPLETE selected',selected,flush=True)


def fit_reward(config,data,output):
    output=Path(output)
    path=output/'reward'
    if (path/'reward.pkl').exists():return ResearchReward.load(path)
    print('REWARD START',flush=True);start=time.monotonic()
    reward=ResearchReward(config.get('reward_config')).fit(data.train,data.rm_valid,data.items,data.vocab_size)
    reward.save(path)
    write_json(path/'timing.json',{'training_seconds':time.monotonic()-start})
    print('REWARD COMPLETE',reward.validation_metrics.get('ensemble'),flush=True)
    return reward


def runtime_device(config):
    return config.get('device','cuda' if torch.cuda.is_available() else 'cpu')


def evaluator(config,data,reward,knn):
    from .evaluate_v2 import evaluate
    if 'train_count' in data.items:
        popularity=np.zeros(data.vocab_size,float)
        popularity[data.items.token.to_numpy(int)]=data.items.train_count.to_numpy(float)
    else:popularity=knn.item_user_count.astype(float)
    if config['dataset']=='yambda' and 'artist_ids' in data.items:
        genres={int(row.token):row.artist_ids for row in data.items.itertuples()}
    else:genres={int(row.token):[g.strip() for g in str(row.genres).split(',') if g.strip()] for row in data.items.itertuples()}
    def run(callback,frame,reward_source=None):
        return evaluate(callback,frame,data.vocab_size,reward if reward_source is None else reward_source,
                        popularity=popularity,genres=genres,batch_size=config.get('eval_batch_size',128),exclude_seen=config.get('exclude_seen',True))
    def pop(histories):return np.tile(np.log1p(popularity),(len(histories),1))
    return run,pop


def fitted_knn(data,output):
    from .baselines_v2 import ItemKNN
    path=Path(output)/'itemknn.pkl'
    if path.exists():return ItemKNN.load(path)
    print('ITEMKNN START',flush=True)
    model=ItemKNN().fit(data.train,data.vocab_size);model.save(path)
    write_json(Path(output)/'itemknn_manifest.json',model.training_manifest)
    return model


def save_predictions(output,name,split,metrics,per,data,config):
    directory=Path(output)/'evaluation';directory.mkdir(exist_ok=True)
    per.to_pickle(directory/f'{name}_{split}.pkl')
    if config['dataset']=='kion':
        from .evaluate_v2 import day_set_metrics
        extra,contexts=day_set_metrics(per,getattr(data,split))
        metrics.update(extra);contexts.to_pickle(directory/f'{name}_{split}_sets.pkl')
    return metrics


class UtilityReward:
    """Declared alternative utilities; completion critic and its inputs stay frozen."""
    def __init__(self,base,utility):self.base=base;self.utility=utility
    def score_candidates(self,histories,candidates):
        scores=self.base.score_candidates(histories,candidates)
        scores=dict(scores)
        if self.utility=='duration_neutral':
            scores['mean']=scores['mean']-scores['duration_prior']+self.base.global_mean
        elif self.utility=='seconds':
            # Fixed 300-second unit; cap a single item at 10 min before optimisation.
            scores['mean']=np.minimum(scores['expected_seconds'],600.)/300.
        return scores


def group_metrics(entries):
    groups={}
    for key,entry in entries.items():
        family=entry['family'];groups.setdefault(family,[]).append(entry['validation'])
    result={}
    for name,rows in groups.items():
        numeric={k for row in rows for k,v in row.items() if isinstance(v,(float,int)) and not isinstance(v,bool)}
        result[name]={k:float(np.mean([r[k] for r in rows if k in r and r[k] is not None])) for k in numeric}
    return result


def select_recipe(groups,threshold):
    criterion='factual_completion_weighted_hit@10'
    eligible={name:row for name,row in groups.items() if row['ndcg@10']>=threshold}
    if not eligible:return None
    return max(eligible,key=lambda name:(eligible[name][criterion],eligible[name]['ndcg@10'],name))


def develop(config,data,output):
    from .evaluate_v2 import RewardReranker
    output=Path(output);device=runtime_device(config)
    frozen_path=output/'frozen_selection.json'
    if frozen_path.exists():
        print('Development already frozen',flush=True);return
    reward=ResearchReward.load(output/'reward');knn=fitted_knn(data,output);assess,pop=evaluator(config,data,reward,knn)
    progress=output/'development.json'
    results=json.loads(progress.read_text()) if progress.exists() else {'models':{},'rerank_grid':{},'config':config}
    for name,callback in [('popularity',pop),('itemknn',knn)]:
        if name not in results['models']:
            metrics,per=assess(callback,data.valid)
            metrics=save_predictions(output,name,'valid',metrics,per,data,config)
            results['models'][name]={'family':name,'seed':None,'validation':metrics}
            write_json(progress,results);print('VALID',name,{k:metrics[k] for k in ['ndcg@10','factual_completion_weighted_hit@10']},flush=True)
    for seed in config['sft_seeds']:
        base,_=load_policy(output/'checkpoints'/f'baseline_{seed}.pt',device)
        base_fn=logits_function(base,device)
        name=f'baseline_{seed}'
        if name not in results['models']:
            metrics,per=assess(base_fn,data.valid);metrics=save_predictions(output,name,'valid',metrics,per,data,config)
            results['models'][name]={'family':'baseline','seed':seed,'validation':metrics}
            write_json(progress,results);print('VALID',name,{k:metrics[k] for k in ['ndcg@10','factual_completion_weighted_hit@10']},flush=True)
        cache={}
        for reward_type in ['mean','mean-std']:
            for alpha in config['rerank_alphas']:
                key=f'{reward_type}:{alpha}'
                item=results['rerank_grid'].setdefault(key,{'reward_type':reward_type,'alpha':alpha,'by_seed':{}})
                if str(seed) in item['by_seed']:continue
                reranker=RewardReranker(base_fn,reward,data.vocab_size,alpha,top_m=config.get('rerank_top_m',200),reward_type=reward_type,
                    cache=cache,cache_namespace=f'{seed}',exclude_seen=config.get('exclude_seen',True))
                metrics,_=assess(reranker,data.valid,reranker)
                item['by_seed'][str(seed)]=metrics
                write_json(progress,results)
                print('RERANK VALID',seed,key,{k:metrics[k] for k in ['ndcg@10','factual_completion_weighted_hit@10']},flush=True)
        del cache
        for variant in config['variants']:
            name=f"{variant['name']}_{seed}";checkpoint=output/'checkpoints'/f'{name}.pt'
            if name in results['models']:continue
            if checkpoint.exists():model,metadata=load_policy(checkpoint,device)
            else:
                print('RL START',name,flush=True);start=time.monotonic()
                rm=UtilityReward(reward,variant['reward']) if variant['reward'] in ['duration_neutral','seconds'] else reward
                effective=dict(variant,reward='completion') if variant['reward'] in ['duration_neutral','seconds'] else variant
                model,logs=rl_finetune(base,data.train,rm,config,effective,seed+1000,device,
                    callback=lambda x:print(name,x,flush=True))
                write_json(output/'training'/f'{name}.json',logs)
                metadata={'variant':variant,'seed':seed,'trained_until':config.get('train_end','relative-time'),
                    'training_seconds':time.monotonic()-start,
                    'sampled_actions':sum(row['sampled_actions'] for row in logs)}
                save_policy(model,checkpoint,metadata)
            metrics,per=assess(logits_function(model,device),data.valid)
            metrics=save_predictions(output,name,'valid',metrics,per,data,config)
            results['models'][name]={'family':variant['name'],'variant':variant,'seed':seed,'validation':metrics,
                'training_seconds':metadata.get('training_seconds'),'sampled_actions':metadata.get('sampled_actions')}
            write_json(progress,results);print('VALID',name,{k:metrics[k] for k in ['ndcg@10','factual_completion_weighted_hit@10']},flush=True)
            del model
        del base
        if torch.cuda.is_available():torch.cuda.empty_cache()
    groups=group_metrics(results['models']);threshold=(1-config.get('noninferiority_margin',.05))*groups['baseline']['ndcg@10']
    for key,item in results['rerank_grid'].items():
        rows=list(item['by_seed'].values());numeric={k for r in rows for k,v in r.items() if isinstance(v,(float,int)) and not isinstance(v,bool)}
        item['mean_metrics']={k:float(np.mean([r[k] for r in rows if k in r and r[k] is not None])) for k in numeric}
    rerank_groups={key:item['mean_metrics'] for key,item in results['rerank_grid'].items()}
    rr_key=select_recipe(rerank_groups,threshold)
    chosen_rr=results['rerank_grid'][rr_key]
    groups['reranker']=chosen_rr['mean_metrics']
    rl_names={v['name'] for v in config['variants'] if v.get('pg_weight',1)>0}
    selected_rl=select_recipe({name:row for name,row in groups.items() if name in rl_names},threshold)
    overall=select_recipe(groups,threshold)
    frozen={'selected_overall':overall,'selected_rl':selected_rl,'reranker':{'key':rr_key,'alpha':chosen_rr['alpha'],'reward_type':chosen_rr['reward_type']},
        'validation_groups':groups,'ndcg_guardrail':threshold,'selection_metric':'factual_completion_weighted_hit@10',
        'rule':'Maximise user-macro observed-completion weighted hit among recipes with mean validation NDCG >=95% baseline. Same recipe for all seeds; baseline always eligible.',
        'heldout_use':'Primary and confirmation predictions not inspected before this file was written.'}
    results['groups']=groups;write_json(progress,results);write_json(frozen_path,frozen)
    print('DEVELOPMENT FROZEN',overall,'RL',selected_rl,'RR',frozen['reranker'],flush=True)


def evaluate_final(config,data,output):
    from .evaluate_v2 import RewardReranker,bootstrap_family,holm_adjust
    from .evaluate import evaluate_rollouts
    output=Path(output);frozen_path=output/'frozen_selection.json'
    if not frozen_path.exists():raise ValueError('Development selection must be frozen first')
    selection=json.loads(frozen_path.read_text())
    device=runtime_device(config);reward=ResearchReward.load(output/'reward');knn=fitted_knn(data,output)
    assess,pop=evaluator(config,data,reward,knn)
    development=json.loads((output/'development.json').read_text());path=output/'results.json'
    if path.exists():results=json.loads(path.read_text())
    else:
        import platform
        results={'version':2,'dataset':config['dataset'],'config':config,'data_manifest':data.manifest,
            'sft_selection':json.loads((output/'sft_selection.json').read_text()),'selection':selection,
            'models':copy.deepcopy(development['models']),'rerank_grid':development['rerank_grid'],
            'reward_metrics':{'rm_valid':reward.validation_metrics,'valid':reward.evaluate(data.valid)},
            'environment':{'platform':platform.platform(),'device':device,'torch':torch.__version__,
                'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                'packages':{p:importlib.metadata.version(p) for p in ['transformers','numpy','pandas','scipy','catboost','scikit-learn']}}}
    for seed in config['sft_seeds']:
        results['models'].setdefault(f'reranker_{seed}',{'family':'reranker','seed':seed,'rerank':selection['reranker'],
            'validation':development['rerank_grid'][selection['reranker']['key']]['by_seed'][str(seed)]})
    for split in ['test','confirmation']:
        frame=getattr(data,split)
        if split not in results['reward_metrics']:results['reward_metrics'][split]=reward.evaluate(frame)
        for name in ['popularity','itemknn']:
            entry=results['models'][name]
            if split in entry:continue
            callback=pop if name=='popularity' else knn
            metrics,per=assess(callback,frame);entry[split]=save_predictions(output,name,split,metrics,per,data,config)
            write_json(path,results);print(split.upper(),name,{k:metrics[k] for k in ['ndcg@10','factual_completion_weighted_hit@10']},flush=True)
        for seed in config['sft_seeds']:
            names=[f'baseline_{seed}']+[f"{v['name']}_{seed}" for v in config['variants']]+[f'reranker_{seed}']
            base=None
            for name in names:
                entry=results['models'][name]
                if split in entry:continue
                if entry['family']=='reranker':
                    if base is None:base,_=load_policy(output/'checkpoints'/f'baseline_{seed}.pt',device)
                    rr=selection['reranker'];callback=RewardReranker(logits_function(base,device),reward,data.vocab_size,rr['alpha'],
                        top_m=config.get('rerank_top_m',200),reward_type=rr['reward_type'],exclude_seen=config.get('exclude_seen',True))
                    rm_source=callback;model=None
                else:
                    model,_=load_policy(output/'checkpoints'/f'{name}.pt',device);callback=logits_function(model,device);rm_source=reward
                metrics,per=assess(callback,frame,rm_source)
                entry[split]=save_predictions(output,name,split,metrics,per,data,config)
                if split=='test' and entry['family'] in ['baseline','raw_h5','anchored_h1','anchored_h5','appended_h5','conservative_h5','retention_h5','duration_neutral_h5','seconds_h5','reranker']:
                    summary,rollouts=evaluate_rollouts(callback,frame,data.vocab_size,reward.score_candidates,horizon=5,max_examples=256,
                        batch_size=64,exclude_observed=config.get('exclude_seen',True))
                    entry['rollout']=summary;rollouts.to_csv(output/'evaluation'/f'{name}_rollouts.csv',index=False)
                write_json(path,results);print(split.upper(),name,{k:metrics[k] for k in ['ndcg@10','factual_completion_weighted_hit@10','predicted_reward@10']},flush=True)
                if model is not None:del model
            if base is not None:del base
            if torch.cuda.is_available():torch.cuda.empty_cache()
    # Comparisons use identical users and paired SFT initialisations.
    def frames(family,split,metric=None):
        values=[pd.read_pickle(output/'evaluation'/f'{family}_{seed}_{split}.pkl') for seed in config['sft_seeds']]
        return [f[np.isfinite(f[metric])].copy() for f in values] if metric else values
    stats={}
    for split in ['test','confirmation']:
        comparisons={};baseline=frames('baseline',split)
        families=[v['name'] for v in config['variants']]+['reranker']
        for family in families:
            comparisons[f'{family}_vs_baseline']={metric:bootstrap_family(frames('baseline',split,metric),frames(family,split,metric),metric,
                margin=config.get('noninferiority_margin',.05),n_bootstrap=config.get('bootstrap_samples',2000))
                for metric in ['ndcg@10','factual_completion_weighted_hit@10']}
        selected=selection['selected_rl']
        if selected:
            for comparator in ['matched_control','reranker']:
                comparisons[f'selected_rl_vs_{comparator}']={metric:bootstrap_family(frames(comparator,split,metric),frames(selected,split,metric),metric,
                    margin=config.get('noninferiority_margin',.05),n_bootstrap=config.get('bootstrap_samples',2000))
                    for metric in ['ndcg@10','factual_completion_weighted_hit@10']}
        pvalues={f'{name}:{metric}':item['paired_randomization_p_two_sided'] for name,row in comparisons.items() for metric,item in row.items()}
        adjusted=holm_adjust(pvalues)
        for name,row in comparisons.items():
            for metric,item in row.items():
                item['holm_adjusted_p_two_sided']=adjusted[f'{name}:{metric}']
                item['noninferiority']['primary']='user_conditional_on_trained_checkpoints'
        stats[split]=comparisons
    results['statistics']=stats
    write_json(output/'statistics.json',stats)
    write_json(path,results)
    print('RESEARCH COMPLETE',config['dataset'],str(path),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/kion.json')
    p.add_argument('--output',default='artifacts/kion')
    p.add_argument('--data',default='data/raw');p.add_argument('--stage',default='all',choices=['prepare','sft','reward','develop','evaluate','all'])
    args=p.parse_args();config=json.loads(Path(args.config).read_text());torch.set_num_threads(config.get('threads',4))
    data=prepared(config,args.output,args.data)
    print('DATA',config['dataset'],'train',len(data.train),'RM-valid',len(data.rm_valid),'policy-valid',len(data.valid),'vocab',data.vocab_size,flush=True)
    if args.stage in ('sft','all'):fit_sft(config,data,args.output)
    if args.stage in ('reward','all'):fit_reward(config,data,args.output)
    if args.stage in ('develop','all'):develop(config,data,args.output)
    if args.stage in ('evaluate','all'):evaluate_final(config,data,args.output)


if __name__=='__main__':main()
