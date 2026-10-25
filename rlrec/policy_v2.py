"""Item GPT-2 with a last-position LM head and consistent eligible catalog."""
from __future__ import annotations
import random
from pathlib import Path
import numpy as np
import torch
from transformers import GPT2Config,GPT2LMHeadModel


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_policy(model,path,metadata=None):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    torch.save({'config':model.config.to_dict(),'state_dict':model.state_dict(),
                'metadata':metadata or {}},path)


def load_policy(path,device='cpu'):
    state=torch.load(path,map_location='cpu',weights_only=True)
    cfg=GPT2Config.from_dict(state['config'])
    cfg._attn_implementation='eager'
    model=GPT2LMHeadModel(cfg)
    model.load_state_dict(state['state_dict'])
    return model.to(device).eval(),state['metadata']


def make_model(vocab_size,config):
    cfg=GPT2Config(vocab_size=vocab_size,n_positions=config.get('max_length',48),
        n_ctx=config.get('max_length',48),n_embd=config.get('embedding_dim',128),
        n_layer=config.get('layers',2),n_head=config.get('heads',4),
        resid_pdrop=config.get('dropout',.1),embd_pdrop=config.get('dropout',.1),attn_pdrop=config.get('dropout',.1),
        bos_token_id=1,pad_token_id=0,eos_token_id=None,use_cache=False)
    cfg._attn_implementation='eager'
    return GPT2LMHeadModel(cfg)


def pack(histories,max_length,device='cpu'):
    sequences=[[1]+list(h)[-max_length+1:] for h in histories]
    width=max(map(len,sequences));arr=np.zeros((len(sequences),width),dtype=np.int64)
    for i,seq in enumerate(sequences):arr[i,-len(seq):]=seq
    ids=torch.as_tensor(arr,device=device);mask=ids.ne(0).long()
    return ids,mask,(mask.cumsum(-1)-1).clamp_min(0)


def next_logits(model,histories,device='cpu'):
    ids,mask,pos=pack(histories,model.config.n_positions,device)
    return model(input_ids=ids,attention_mask=mask,position_ids=pos,logits_to_keep=1).logits[:,0]


def masked_logits(logits,histories,exclude_seen=True,generated=None):
    out=logits.clone();out[:,:3]=-1e9
    for row,hist in enumerate(histories):
        seen=set(hist) if exclude_seen else set()
        if generated is not None:seen.update(generated[row])
        seen=[int(x) for x in seen if 3<=int(x)<out.shape[-1]]
        if seen:out[row,seen]=-1e9
    if (out[:,3:]>-1e8).sum(-1).min()==0:
        raise ValueError('Eligible item catalog exhausted')
    return out


def logits_function(model,device='cpu'):
    def callback(histories):
        model.eval()
        with torch.no_grad():return next_logits(model,histories,device).cpu().numpy()
    return callback


def cross_entropy(logits,targets,smoothing=0.):
    logp=torch.log_softmax(logits,dim=-1)
    nll=-logp.gather(1,targets[:,None])[:,0]
    if smoothing:
        allowed=logits>-1e8
        smooth=-(logp*allowed).sum(-1)/allowed.sum(-1)
        nll=(1-smoothing)*nll+smoothing*smooth
    return nll
