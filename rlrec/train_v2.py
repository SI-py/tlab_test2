"""Next-item training and matched RLOO ablations for the second study."""
from __future__ import annotations
import copy
import math
import time
import numpy as np
import pandas as pd
import torch
from .policy_v2 import next_logits,masked_logits,cross_entropy,seed_everything


def rloo_advantages(rewards,gamma=.95):
    """Return-to-go minus the mean return of the other trajectories."""
    returns=torch.zeros_like(rewards)
    running=torch.zeros_like(rewards[...,0])
    for t in reversed(range(rewards.shape[-1])):
        running=rewards[...,t]+gamma*running
        returns[...,t]=running
    groups=rewards.shape[1]
    if groups<2:
        raise ValueError('RLOO requires at least two independent trajectories')
    baseline=(returns.sum(dim=1,keepdim=True)-returns)/(groups-1)
    return returns-baseline


def optimizer_for(model,lr):
    decay=[p for p in model.parameters() if p.requires_grad and p.ndim>=2]
    plain=[p for p in model.parameters() if p.requires_grad and p.ndim<2]
    return torch.optim.AdamW([{'params':decay,'weight_decay':.01},{'params':plain,'weight_decay':0.}],lr=lr)


def ranking_validation(model,frame,exclude_seen,device='cpu',batch_size=256):
    rows=[];ce_sum=0.;known=0
    model.eval()
    with torch.no_grad():
        for offset in range(0,len(frame),batch_size):
            b=frame.iloc[offset:offset+batch_size]
            logits=masked_logits(next_logits(model,b.history.tolist(),device),b.history.tolist(),exclude_seen)
            target=torch.as_tensor(b.target.to_numpy(copy=True),device=device)
            top=torch.topk(logits,min(20,logits.shape[-1]-3),dim=-1).indices
            matches=(top==target[:,None]) & (target[:,None]>=3)
            hit10=matches[:,:10].any(-1).cpu().numpy().astype(float)
            discount=1/torch.log2(torch.arange(2,top.shape[-1]+2,device=device).float())
            ndcg=(matches[:,:10]*discount[:10]).sum(-1).cpu().numpy()
            mask=target>=3
            if mask.any():
                ce_sum+=cross_entropy(logits[mask],target[mask]).sum().item();known+=int(mask.sum())
            rows.extend(zip(b.user_id.to_numpy(),hit10,ndcg))
    df=pd.DataFrame(rows,columns=['user_id','recall','ndcg'])
    means=df.groupby('user_id')[['recall','ndcg']].mean().mean()
    return {'recall@10':float(means['recall']),'ndcg@10':float(means['ndcg']),'ce':ce_sum/max(1,known)}


def supervised_train(model,train,valid,config,device='cpu',callback=None):
    known=train[train.target>=3].reset_index(drop=True)
    rng=np.random.default_rng(config['seed'])
    epochs=config.get('sft_epochs',8);batch_size=config.get('batch_size',256)
    opt=optimizer_for(model,config.get('sft_lr',.001))
    total_steps=math.ceil(len(known)/batch_size)*epochs;warm=max(1,int(total_steps*.08))
    def lr_scale(step):
        if step<warm:return (step+1)/warm
        progress=(step-warm)/max(1,total_steps-warm)
        return .1+.9*.5*(1+math.cos(math.pi*min(1.,progress)))
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lr_scale)
    best=-float('inf');state=None;stale=0;logs=[]
    for epoch in range(epochs):
        model.train();start=time.monotonic();order=rng.permutation(len(known));loss_sum=0.;n=0
        for offset in range(0,len(known),batch_size):
            b=known.iloc[order[offset:offset+batch_size]]
            histories=b.history.tolist();target=torch.as_tensor(b.target.to_numpy(copy=True),device=device)
            logits=masked_logits(next_logits(model,histories,device),histories,config.get('exclude_seen',True))
            loss=cross_entropy(logits,target,config.get('label_smoothing',.02)).mean()
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step();scheduler.step();loss_sum+=loss.item()*len(b);n+=len(b)
        validation=ranking_validation(model,valid,config.get('exclude_seen',True),device,batch_size)
        log={'epoch':epoch+1,'train_loss':loss_sum/n,'validation':validation,'seconds':time.monotonic()-start,'lr':scheduler.get_last_lr()[0]}
        logs.append(log)
        if validation['ndcg@10']>best+1e-5:
            best=validation['ndcg@10'];state=copy.deepcopy(model.state_dict());stale=0
        else:stale+=1
        if callback:callback(log)
        if stale>=config.get('sft_patience',3):break
    model.load_state_dict(state)
    return model.eval(),logs


class ContextSampler:
    """Uniform users, then observed contexts, then a known target within context."""
    def __init__(self,frame,seed):
        frame=frame[frame.target>=3].reset_index(drop=True)
        context='timestamp' if 'timestamp' in frame else 'day'
        self.frame=frame;self.rng=np.random.default_rng(seed);self.users=[];self.contexts=[]
        for uid,user in frame.groupby('user_id',sort=True):
            self.users.append(int(uid));self.contexts.append([part.index.to_numpy() for _,part in user.groupby(context,sort=True)])
    def sample(self,n):
        selected=[]
        for u in self.rng.integers(0,len(self.users),size=n):
            contexts=self.contexts[u];rows=contexts[self.rng.integers(len(contexts))]
            selected.append(rows[self.rng.integers(len(rows))])
        return self.frame.iloc[selected]


def rl_finetune(base,contexts,reward,config,variant,seed,device='cpu',callback=None):
    seed_everything(seed)
    model=copy.deepcopy(base).to(device).eval();reference=copy.deepcopy(base).to(device).eval()
    reference.requires_grad_(False)
    opt=optimizer_for(model,variant.get('lr',config.get('rl_lr',.00015)))
    sampler=ContextSampler(contexts,seed)
    group=config.get('rl_group_size',4);batch_size=config.get('rl_batch_size',24)
    horizon=variant.get('horizon',5);gamma=config.get('gamma',1.);logs=[]
    for step in range(config.get('rl_steps',160)):
        b=sampler.sample(batch_size);initial=b.history.tolist()
        original=[list(h) for h in initial for _ in range(group)]
        histories=[h.copy() for h in original];generated=[[] for _ in histories]
        logps=[];kls=[];rewards=[];factual_scores=[]
        for t in range(horizon):
            logits=masked_logits(next_logits(model,histories,device),histories,config.get('exclude_seen',True),generated)
            logp=torch.log_softmax(logits,dim=-1);probs=logp.exp()
            with torch.no_grad():
                ref_logits=masked_logits(next_logits(reference,histories,device),histories,config.get('exclude_seen',True),generated)
                ref_logp=torch.log_softmax(ref_logits,dim=-1)
                action=torch.multinomial(probs.detach(),1)
                rm_histories=histories if variant.get('rm_context','frozen')=='append' else original
                score=reward.score_candidates(rm_histories,action.cpu().numpy())
                value=score['mean'][:,0].copy();factual_scores.append(float(value.mean()))
                if variant.get('reward')=='conservative':
                    value=value-score['std'][:,0]-.05*(1-score['support'][:,0])
                elif variant.get('reward')=='retention':
                    value=.65*value+.35*score['return7'][:,0]-score['std'][:,0]
                rewards.append(torch.as_tensor(value,device=device,dtype=torch.float32))
            logps.append(logp.gather(1,action)[:,0]);kls.append((probs*(logp-ref_logp)).sum(-1))
            for history,slate,a in zip(histories,generated,action[:,0].tolist()):
                history.append(a);slate.append(a)
        values=torch.stack(rewards,dim=-1).reshape(batch_size,group,horizon)
        advantages=rloo_advantages(values,gamma).detach()
        discounts=gamma**torch.arange(horizon,device=device)
        pg=-(torch.stack(logps,dim=-1).reshape_as(values)*advantages*discounts).mean()
        kl=torch.stack(kls,dim=-1).mean()
        replay=masked_logits(next_logits(model,initial,device),initial,config.get('exclude_seen',True))
        ce=cross_entropy(replay,torch.as_tensor(b.target.to_numpy(copy=True),device=device)).mean()
        loss=variant.get('pg_weight',1.)*pg+variant.get('beta',.2)*kl+variant.get('ce_weight',.5)*ce
        opt.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        log={'step':step+1,'reward':values.mean().item(),'completion':mean(factual_scores),'kl':kl.item(),
             'ce':ce.item(),'pg_loss':pg.item(),'loss':loss.item(),'grad_norm':float(norm),'sampled_actions':batch_size*group*horizon}
        logs.append(log)
        if callback and (step==0 or (step+1)%40==0):callback(log)
    return model.eval(),logs


def mean(values):return sum(values)/len(values)
