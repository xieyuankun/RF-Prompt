"""Replay-free audio adaptations of recent and audio-specific prompt methods.

The XLS-R encoder is frozen and prompts are injected as attention K/V prefixes.
Both methods retain the seeded AASIST binary backend used by the matched suite.
"""
import copy
import types

import torch
from torch import nn
from torch.nn import functional as F
from transformers import Wav2Vec2FeatureExtractor

from feature_extraction import load_wav2vec2_model
from model import SSLAASIST


class AudioPromptComparison(nn.Module):
    def __init__(self, model_dir, method, seed=2026):
        super().__init__()
        if method not in {'singleprompt', 'kaprompt', 'oisoprompt'}:
            raise ValueError(method)
        self.kind = method
        self.ssl = load_wav2vec2_model(model_dir)
        self.ssl.requires_grad_(False)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_dir, do_normalize=False
        )
        self.task = 0
        self.epoch = 0
        self.enabled = True
        self.route_mode = 'main'
        self.last_query = None
        self.last_key_loss = None
        self.last_aux_logits = None
        self.teacher_general = None
        object.__setattr__(self, '_teacher_aasist', None)

        d = self.ssl.config.hidden_size
        self.dim = d
        self.layers = self.ssl.config.num_hidden_layers
        self.tokens = 5
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            if method == 'singleprompt':
                self.single_prompt = nn.Parameter(
                    torch.empty(self.layers, 2 * self.tokens, d)
                )
                nn.init.uniform_(self.single_prompt, -0.02, 0.02)
            elif method == 'oisoprompt':
                # Oiso et al. (INTERSPEECH 2024): one shallow five-token
                # prompt prepended before positional convolution/encoding.
                # Their default copies one pretrained positional-convolution
                # vector into every prompt slot (random_range=None).
                initial = self.ssl.encoder.pos_conv_embed.conv.weight[
                    :, 0, 0
                ].detach().clone().repeat(self.tokens, 1)
                self.input_prompt = nn.Parameter(initial)
            else:
            # Closest official DIL configuration: one prompt per task, top-1.
                self.pool_size = 4
                self.expert_tokens = 10  # 10 K + 10 V, official prompt length 20.
                self.general_tokens = 2  # 2 K + 2 V, official general length 4.
                self.general_layers = {0, 1}
                self.expert_layers = {2, 3, 4}
                self.keys = nn.Parameter(torch.empty(self.pool_size, d))
                self.task_prompts = nn.Parameter(torch.empty(
                    self.pool_size, self.layers, 2 * self.expert_tokens, d
                ))
                self.general_prompt = nn.Parameter(torch.empty(
                    self.layers, 2 * self.general_tokens, d
                ))
                nn.init.uniform_(self.keys, -0.02, 0.02)
                nn.init.uniform_(self.task_prompts, -0.02, 0.02)
                nn.init.uniform_(self.general_prompt, -0.02, 0.02)

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.w2vaasist = SSLAASIST(input_dim=d)

        if method == 'oisoprompt':
            encoder = self.ssl.encoder
            original_encoder = encoder.forward

            def encoder_forward(encoder_self, hidden_states,
                                attention_mask=None, **kwargs):
                batch = hidden_states.shape[0]
                prompt = self.input_prompt.unsqueeze(0).expand(batch, -1, -1)
                hidden_states = torch.cat([prompt, hidden_states], dim=1)
                if attention_mask is not None:
                    prompt_mask = torch.ones(
                        batch, self.tokens, dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    attention_mask = torch.cat(
                        [prompt_mask, attention_mask], dim=1
                    )
                return original_encoder(
                    hidden_states, attention_mask=attention_mask, **kwargs
                )

            encoder.forward = types.MethodType(encoder_forward, encoder)

        for layer_index, layer in enumerate(self.ssl.encoder.layers):
            if method == 'oisoprompt':
                break
            attention = layer.attention
            original = attention.forward

            def forward(attn, hidden_states, attention_mask=None,
                        output_attentions=False, _layer=layer_index,
                        _original=original, **kwargs):
                prompt = self._prompt_for(hidden_states, _layer)
                if not self.enabled or prompt is None:
                    return _original(
                        hidden_states,
                        attention_mask=attention_mask,
                        output_attentions=output_attentions,
                        **kwargs,
                    )
                return self._attend(attn, hidden_states, prompt, attention_mask)

            attention.forward = types.MethodType(forward, attention)

    def train(self, mode=True):
        super().train(mode)
        self.ssl.eval()
        teacher = object.__getattribute__(self, '_teacher_aasist')
        if teacher is not None:
            teacher.eval()
        return self

    def _prompt_for(self, x, layer):
        if not self.enabled:
            return None
        batch = x.shape[0]
        if self.kind == 'singleprompt':
            return self.single_prompt[layer].unsqueeze(0).expand(batch, -1, -1)
        if self.kind == 'oisoprompt':
            return None
        if layer in self.general_layers:
            source = (
                self.teacher_general[layer]
                if self.route_mode == 'aux' and self.teacher_general is not None
                else self.general_prompt[layer]
            )
            return source.unsqueeze(0).expand(batch, -1, -1)
        if layer not in self.expert_layers:
            return None
        if self.training and self.route_mode == 'main':
            ids = torch.full(
                (batch,), self.task, dtype=torch.long, device=x.device
            )
            return self.task_prompts[ids, layer]
        query = F.normalize(self.last_query, dim=-1)
        keys = F.normalize(self.keys[:self.task + 1], dim=-1)
        scores = query @ keys.T
        if self.route_mode == 'aux' and self.task > 0:
            old_ids = scores[:, :self.task].argmax(-1)
            old = self.task_prompts[old_ids, layer]
            current = self.task_prompts[self.task, layer].unsqueeze(0)
            return 0.5 * (old + current)
        ids = scores.argmax(-1)
        return self.task_prompts[ids, layer]

    def _attend(self, attn, x, prompt, mask):
        batch, length, dim = x.shape
        heads = attn.num_heads

        def split(value):
            return value.reshape(batch, -1, heads, dim // heads).transpose(1, 2)

        q, k, v = (split(project(x)) for project in
                   (attn.q_proj, attn.k_proj, attn.v_proj))
        half = prompt.shape[1] // 2
        pk, pv = split(prompt[:, :half]), split(prompt[:, half:])
        prefix_logits = q @ pk.transpose(-1, -2)
        logits = torch.cat([prefix_logits, q @ k.transpose(-1, -2)], -1)
        logits = logits * attn.scaling
        if mask is not None:
            logits = logits + F.pad(mask, (pk.shape[2], 0), value=0)
        weights = logits.softmax(-1)
        output = weights @ torch.cat([pv, v], dim=2)
        output = output.transpose(1, 2).reshape(batch, length, dim)
        return attn.out_proj(output), None, None

    def _values(self, audio):
        return self.processor(
            audio, sampling_rate=16000, return_tensors='pt'
        ).input_values.to(audio.device).squeeze(0)

    def _native_query(self, values):
        self.enabled = False
        with torch.no_grad():
            query = self.ssl(values).last_hidden_state.mean(1)
        self.enabled = True
        return query

    def forward(self, audio):
        values = self._values(audio)
        self.last_aux_logits = None
        if self.kind == 'kaprompt':
            self.last_query = self._native_query(values)
            current_key = F.normalize(self.keys[self.task], dim=-1)
            self.last_key_loss = (
                1.0 - F.normalize(self.last_query, dim=-1) @ current_key
            ).mean()
        self.route_mode = 'main'
        features = self.ssl(values).last_hidden_state
        output = self.w2vaasist(features)
        if self.kind == 'kaprompt' and self.training and self.task > 0:
            self.route_mode = 'aux'
            aux_features = self.ssl(values).last_hidden_state
            teacher = object.__getattribute__(self, '_teacher_aasist')
            self.last_aux_logits = teacher(aux_features)[1]
            self.route_mode = 'main'
        return output

    def auxiliary_loss(self, labels):
        zero = next(self.parameters()).sum() * 0.0
        if self.kind != 'kaprompt':
            return zero
        loss = 0.5 * self.last_key_loss
        if self.last_aux_logits is not None:
            loss = loss + F.cross_entropy(self.last_aux_logits, labels)
        return loss

    @torch.no_grad()
    def initialize_task(self, loader, device):
        if self.kind != 'kaprompt' or self.task == 0:
            return
        queries = []
        total = 0
        for batch in loader:
            waveform, _, labels = batch
            mask = labels == 1
            if mask.any():
                values = self._values(waveform[mask].to(device))
                queries.append(self._native_query(values).cpu())
                total += int(mask.sum())
            if total >= 512:
                break
        query = F.normalize(torch.cat(queries)[:512].mean(0), dim=-1)
        old_keys = F.normalize(self.keys[:self.task].detach().cpu(), dim=-1)
        parent = int((old_keys @ query).argmax())
        self.keys[self.task].copy_(self.keys[parent])
        self.task_prompts[self.task].copy_(self.task_prompts[parent])
        self.selected_parent = parent

    def before_task(self, task):
        self.task = task
        self.epoch = 0
        if self.kind == 'oisoprompt':
            # Task 1 establishes the source detector. For subsequent tasks,
            # reproduce Type B: shared input prompt + final linear layer only.
            self.w2vaasist.requires_grad_(task == 0)
            self.w2vaasist.out_layer.requires_grad_(True)
            self.input_prompt.requires_grad_(True)
            if task > 0:
                for name, parameter in self.w2vaasist.named_parameters():
                    if not name.startswith('out_layer.'):
                        parameter.grad = None
            return
        if self.kind != 'kaprompt':
            return
        self.selected_parent = None
        self.frozen_keys = (
            self.keys[:task].detach().clone(), self.keys[task + 1:].detach().clone()
        )
        self.frozen_prompts = (
            self.task_prompts[:task].detach().clone(),
            self.task_prompts[task + 1:].detach().clone(),
        )
        if task > 0:
            teacher = copy.deepcopy(self.w2vaasist).eval()
            teacher.requires_grad_(False)
            object.__setattr__(self, '_teacher_aasist', teacher)
            self.teacher_general = self.general_prompt.detach().clone()

    @torch.no_grad()
    def restore_frozen(self):
        if self.kind != 'kaprompt':
            return
        old, future = self.frozen_keys
        self.keys[:self.task].copy_(old)
        self.keys[self.task + 1:].copy_(future)
        old, future = self.frozen_prompts
        self.task_prompts[:self.task].copy_(old)
        self.task_prompts[self.task + 1:].copy_(future)

    def refresh_rainbow(self):
        return

    def meta(self):
        result = {'task': self.task, 'epoch': self.epoch}
        if self.kind == 'kaprompt':
            result['selected_parent'] = getattr(self, 'selected_parent', None)
            result['teacher_general'] = self.teacher_general
            teacher = object.__getattribute__(self, '_teacher_aasist')
            result['teacher_aasist'] = (
                None if teacher is None else
                {k: v.detach().cpu() for k, v in teacher.state_dict().items()}
            )
        return result

    def restore_meta(self, meta):
        self.task = meta['task']
        self.epoch = meta['epoch']
        if self.kind != 'kaprompt':
            return
        self.selected_parent = meta.get('selected_parent')
        self.teacher_general = meta.get('teacher_general')
        state = meta.get('teacher_aasist')
        if state is not None:
            teacher = copy.deepcopy(self.w2vaasist).eval()
            teacher.load_state_dict(state)
            teacher.requires_grad_(False)
            object.__setattr__(self, '_teacher_aasist', teacher)

