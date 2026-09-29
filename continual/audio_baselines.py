"""Audio adaptations of the pinned official SMoPE / RainbowPrompt modules.

The pretrained wav2vec encoder (including native normalization) is untouched;
only attention K/V prefixes are added. No Real/Fake-specific prompt split.
"""
import types
import torch
from torch import nn
from torch.nn import functional as F
from transformers import Wav2Vec2FeatureExtractor
from feature_extraction import load_wav2vec2_model
from model import SSLAASIST
from continual.vendor_smope import OnePrompt
from continual.vendor_rainbow import RainbowPrompt


class AudioPromptBaseline(nn.Module):
    def __init__(self, model_dir, method, seed=2026):
        super().__init__()
        self.ssl = load_wav2vec2_model(model_dir)
        self.ssl.requires_grad_(False)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir, do_normalize=False)
        self.kind = method
        self.task = 0
        self.epoch = 0
        self.enabled = True
        self.query = None
        self.scores = {}
        self.sim_loss = None
        d = self.ssl.config.hidden_size
        h = self.ssl.config.num_attention_heads
        n = self.ssl.config.num_hidden_layers
        if method == 'smope':
            # Official CIFAR configuration: 25 K/V experts, top-5, first six layers.
            self.prompt = OnePrompt(d, 4, [50, 5, 1e-5, 1e-5, 0.4], key_dim=d, num_heads=h)
        else:
            # Official IMR 10-task configuration, all SSL layers, scaled task pool.
            self.prompt = RainbowPrompt(length=20, embed_dim=d, pool_size=4,
                top_k=1, n_tasks=4, num_layers=n, num_heads=h,
                prompt_tune_idx=list(range(n)), self_attn_idx=list(range(n//2)),
                D1=56, D2=96, use_linear=True, relation_type='attention', KI_iter=1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.w2vaasist = SSLAASIST(input_dim=d)
        for i, layer in enumerate(self.ssl.encoder.layers):
            attention = layer.attention
            original = attention.forward
            def forward(attn, hidden_states, attention_mask=None, output_attentions=False,
                        _i=i, _original=original, **kwargs):
                if not self.enabled or (self.kind == 'smope' and _i not in self.prompt.e_layers):
                    return _original(hidden_states, attention_mask=attention_mask,
                                     output_attentions=output_attentions, **kwargs)
                return self.attend(attn, hidden_states, _i, attention_mask)
            attention.forward = types.MethodType(forward, attention)

    def train(self, mode=True):
        super().train(mode)
        self.ssl.eval()
        return self

    def attend(self, attn, x, layer, mask):
        b,t,d = x.shape
        h = attn.num_heads
        def heads(v):
            return v.reshape(b,-1,h,d//h).transpose(1,2)
        q,k,v = (heads(proj(x)) for proj in (attn.q_proj,attn.k_proj,attn.v_proj))
        if self.kind == 'smope':
            (pk,pv,eps),_,_ = self.prompt(None,layer,x,train=self.training,task_id=self.task)
            score = q.mean(2,keepdim=True) @ pk.transpose(-1,-2)
            spread = (score.amax(-1,keepdim=True)-score.amin(-1,keepdim=True)).detach()
            target_score = score-eps[None,:,None,:]*spread
            indices = target_score.topk(self.prompt.topk,dim=-1).indices.squeeze(2)
            index = indices[...,None].expand(-1,-1,-1,d//h)
            pk,pv = pk.gather(2,index),pv.gather(2,index)
            self.scores[layer] = (score,target_score)
            prefix_logits = (q.mean(2,keepdim=True) @ pk.transpose(-1,-2)).expand(-1,-1,t,-1)
        else:
            if self.training:
                out = self.prompt(x,layer,cls_features=self.query,cur_id=self.task,
                                  train=True,p_type='Unique' if self.epoch<5 else 'Rainbow')
                pk,pv = map(heads,out['batched_prompt'])
                # Official engine uses the last prompted layer's similarity.
                self.sim_loss = out['sim_loss']
            else:
                # Per-example routing avoids dependence on evaluation batching.
                ids = (F.normalize(self.query,dim=-1) @
                       F.normalize(self.prompt.base_key[:self.task+1],dim=-1).T).argmax(-1)
                p = self.prompt.stored_rainbow_prompts[ids,layer]
                pk,pv = heads(p[:,:10]),heads(p[:,10:])
            prefix_logits = q @ pk.transpose(-1,-2)
        logits = torch.cat([prefix_logits,q@k.transpose(-1,-2)],-1)*attn.scaling
        if mask is not None:
            logits = logits + F.pad(mask,(pk.shape[2],0),value=0)
        weights = logits.softmax(-1)
        y = (weights @ torch.cat([pv,v],dim=2)).transpose(1,2).reshape(b,t,d)
        return attn.out_proj(y), None, None

    @torch.no_grad()
    def refresh_rainbow(self):
        if self.kind != 'rainbow':
            return
        # Cache is rebuilt from checkpoint parameters, never a stale pre-step tensor.
        for l in self.prompt.prompt_tune_idx:
            p = getattr(self.prompt,f'base_knowledge_{l}')
            current = p[self.task:self.task+1]
            if self.epoch<5:
                evolved = current.mean(0)
            else:
                key = F.normalize(self.prompt.base_key[self.task],dim=-1)
                prev = p[:self.task] if self.task else current
                self.prompt.task_id = self.task
                a = self.prompt.task_conditioning_step(prev,key)
                c = self.prompt.task_conditioning_step(current,key)
                evolved = self.prompt.Prompt_Evolution(l,a,c,p.shape[-1],56,dropout=0).mean(0)
            self.prompt.stored_rainbow_prompts[self.task,l].copy_(evolved)

    def forward(self, audio):
        values = self.processor(audio,sampling_rate=16000,return_tensors='pt').input_values.to(audio.device).squeeze(0)
        self.scores = {}
        if self.kind == 'rainbow':
            self.enabled=False
            with torch.no_grad():
                self.query = self.ssl(values).last_hidden_state.mean(1)
            self.enabled=True
        features = self.ssl(values).last_hidden_state
        return self.w2vaasist(features)

    def auxiliary_loss(self):
        if self.kind == 'rainbow':
            return -0.01*self.sim_loss
        loss = self.prompt.router_loss(self.scores,self.task,self.prompt.topk)
        with torch.no_grad():
            self.prompt.update_prompt(self.scores)
        return loss

    def meta(self):
        result = dict(task=self.task,epoch=self.epoch)
        if self.kind == 'smope':
            result['frequent']=self.prompt.used_frequently
            result['freq']={k:v for k,v in vars(self.prompt).items() if k.startswith('e_freq_')}
            result['old']={k:v.detach().cpu() for k,v in vars(self.prompt).items() if k.startswith('old_e_')}
        return result

    def restore_meta(self, meta):
        self.task,self.epoch = meta['task'],meta['epoch']
        if self.kind=='smope':
            self.prompt.used_frequently=meta['frequent']
            for k,v in meta['freq'].items(): setattr(self.prompt,k,v)
            for k,v in meta['old'].items(): setattr(self.prompt,k,v.to(next(self.parameters()).device))

    def before_task(self, task):
        self.task=task
        self.epoch=0
        if self.kind=='smope' and task:
            self.prompt.process_task_count()
        self.frozen_slices={}
        if self.kind=='rainbow':
            for name,p in self.prompt.named_parameters():
                if name.startswith('base_knowledge_') or name=='base_key':
                    self.frozen_slices[name]=(p[:task].detach().clone(),p[task+1:].detach().clone())

    @torch.no_grad()
    def restore_frozen(self):
        if self.kind=='rainbow':
            for name,p in self.prompt.named_parameters():
                if name in self.frozen_slices:
                    old,future=self.frozen_slices[name]
                    p[:self.task].copy_(old)
                    p[self.task+1:].copy_(future)
