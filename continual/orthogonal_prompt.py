"""RF-Prompt modules for continual audio deepfake detection."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoFeatureExtractor,
    Wav2Vec2Config,
    Wav2Vec2FeatureExtractor,
    WavLMConfig,
)

from feature_extraction import load_wav2vec2_model, load_wavlm_model
from model import SSLAASIST


def _orthonormal_rows(matrix):
    """Return an orthonormal basis with the same number of rows."""
    return torch.linalg.qr(matrix.transpose(0, 1), mode='reduced').Q.transpose(0, 1)


def _random_prompt(layers, tokens, hidden_size, device=None):
    """Xavier-scale prompt with orthonormal rows independently per layer."""
    if tokens == 0:
        return torch.empty(layers, 0, hidden_size, device=device)
    limit = math.sqrt(3.0 / hidden_size)
    prompt = torch.empty(layers, tokens, hidden_size, device=device)
    nn.init.uniform_(prompt, -limit, limit)
    return torch.stack([_orthonormal_rows(value) for value in prompt])


def response_soft_weights(query, fake_bank, temperature=0.1):
    """Route with the actual prompt bank and reduce experts to five tokens.

    The expert signature is the mean prompt direction over layers and tokens.
    No separate key or trainable router is introduced.
    """
    if fake_bank.ndim != 4:
        raise ValueError('Expected fake bank [experts, layers, tokens, hidden]')
    if temperature <= 0:
        raise ValueError('Response temperature must be positive')
    signatures = F.normalize(fake_bank.mean(dim=(1, 2)), dim=-1)
    scores = F.normalize(query, dim=-1) @ signatures.t()
    return scores / temperature, F.softmax(scores / temperature, dim=1)


class RFPromptEncoder(nn.Module):
    """Frozen SSL encoder with a shared real prompt and expanding fake experts."""

    def __init__(
        self, model_dir, real_tokens=5, fake_tokens=5, dropout=0.1,
        router_temperature=0.1, router_floor=0.1,
        use_shared_real=True, uniform_fake_router=False,
        joint_mode='none', collect_prompt_response=False,
        fake_init_mode='orthogonal', fake_init_scale=0.1,
        collect_layer_features=False, ssl_backbone='xlsr',
    ):
        super().__init__()
        if ssl_backbone not in ('xlsr', 'wavlm', 'w2vbert'):
            raise ValueError(f'Unsupported SSL backbone: {ssl_backbone}')
        self.ssl_backbone = ssl_backbone
        config_path = str(model_dir) + '/config.json'
        if ssl_backbone == 'wavlm':
            self.config = WavLMConfig.from_json_file(config_path)
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
                model_dir, do_normalize=False
            )
            self.model = load_wavlm_model(model_dir)
        elif ssl_backbone == 'w2vbert':
            from transformers import Wav2Vec2BertConfig, Wav2Vec2BertModel
            self.config = Wav2Vec2BertConfig.from_json_file(config_path)
            self.processor = AutoFeatureExtractor.from_pretrained(model_dir)
            self.model = Wav2Vec2BertModel.from_pretrained(model_dir)
        else:
            self.config = Wav2Vec2Config.from_json_file(config_path)
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
                model_dir, do_normalize=False
            )
            self.model = load_wav2vec2_model(model_dir)
        self.model.config.output_hidden_states = False
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        self.num_layers = self.config.num_hidden_layers
        self.hidden_size = self.config.hidden_size
        self.real_tokens = real_tokens if use_shared_real else 0
        self.fake_tokens = fake_tokens
        self.router_temperature = router_temperature
        self.router_floor = router_floor
        self.uniform_fake_router = uniform_fake_router
        if joint_mode not in (
            'none', 'uniform', 'normalized', 'concat', 'single', 'response',
        ):
            raise ValueError(f'Unknown joint prompt mode: {joint_mode}')
        self.joint_mode = joint_mode
        self.collect_prompt_response = collect_prompt_response
        if fake_init_mode not in ('orthogonal', 'inherit_response'):
            raise ValueError(f'Unknown Fake Prompt initialization: {fake_init_mode}')
        if fake_init_scale < 0:
            raise ValueError('Fake Prompt initialization scale must be non-negative')
        self.fake_init_mode = fake_init_mode
        self.fake_init_scale = fake_init_scale
        self.collect_layer_features = collect_layer_features
        if self.real_tokens:
            stored_real_tokens = self.real_tokens
            self.real_prompt = nn.Parameter(_random_prompt(
                self.num_layers, stored_real_tokens, self.hidden_size
            ))
            self.group_router = (
                None if (joint_mode != 'none')
                else nn.Linear(self.hidden_size, 2)
            )
        else:
            self.register_parameter('real_prompt', None)
            self.group_router = None

        self.fake_prompts = nn.ParameterList()
        self.fake_prompt_bases = nn.ParameterList()
        self.prompt_dropout = nn.Dropout(dropout)
        self.active_tasks = 0
        self.training_task = 0
        self.last_router = None
        self.add_task()
        self.set_task(0)

    @property
    def task_count(self):
        return len(self.fake_prompts)

    def _new_fake_prompt(self):
        device = next(self.parameters()).device
        new_prompt = _random_prompt(
            self.num_layers, self.fake_tokens, self.hidden_size, device=device
        )
        if not self.fake_prompts:
            return new_prompt

        projected = []
        with torch.no_grad():
            for layer_index in range(self.num_layers):
                old = torch.cat([
                    prompt[layer_index].detach()
                    for prompt in self.fake_prompts
                ], dim=0)
                old_basis = _orthonormal_rows(old)
                value = new_prompt[layer_index]
                value = value - (value @ old_basis.t()) @ old_basis
                projected.append(_orthonormal_rows(value))
        return torch.stack(projected)

    def add_task(self):
        """Append one fake expert, initialized in the old-space complement."""
        device = next(self.parameters()).device
        prompt = nn.Parameter(self._new_fake_prompt())
        base = nn.Parameter(
            torch.zeros_like(prompt), requires_grad=False
        )
        # Preserve the exact RNG progression of the experiment code, where a
        # retired key router allocated one vector per task.  Discarding the
        # draw without storing it keeps later prompt initialization identical.
        _compatibility_draw = torch.empty(self.hidden_size, device=device)
        nn.init.normal_(_compatibility_draw, std=self.hidden_size ** -0.5)
        self.fake_prompts.append(prompt)
        self.fake_prompt_bases.append(base)
        # Preserve the retired gate head's RNG draw without storing dead state.
        _compatibility_gate = nn.Linear(
            self.hidden_size, self.num_layers, device=device
        )

    def ensure_task_count(self, count):
        while self.task_count < count:
            self.add_task()


    def initialize_fake_prompt(self, fake_query):
        """Initialize the current Fake5 from the most responsive old expert.

        Token slots are inherited directly from the parent, while a small
        orthogonal residual supplies task-specific capacity.  The parent is
        selected only from current-task frozen SSL queries; no task key is
        stored or used at inference.
        """
        if (
            self.fake_init_mode != 'inherit_response' or
            self.training_task == 0 or fake_query is None
        ):
            return None
        old_bank = torch.stack([
            self.fake_prompts[index].detach()
            for index in range(self.training_task)
        ])
        signatures = F.normalize(old_bank.mean(dim=(1, 2)), dim=-1)
        scores = F.normalize(fake_query.detach(), dim=-1) @ signatures.t()
        parent_index = int(scores.argmax())
        parent = old_bank[parent_index]
        residual = self.fake_prompts[self.training_task].detach()
        residual = F.normalize(residual, dim=-1)
        parent_scale = parent.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        initialized = parent + self.fake_init_scale * parent_scale * residual
        with torch.no_grad():
            self.fake_prompt_bases[self.training_task].copy_(parent)
            self.fake_prompts[self.training_task].copy_(initialized)
        return parent_index, scores.detach()

    def set_task(self, task_index):
        self.ensure_task_count(task_index + 1)
        self.training_task = task_index
        self.active_tasks = task_index + 1
        if self.real_prompt is not None:
            self.real_prompt.requires_grad_(True)
        if self.group_router is not None:
            for parameter in self.group_router.parameters():
                parameter.requires_grad_(True)
        for index, prompt in enumerate(self.fake_prompts):
            enabled = index == task_index
            prompt.requires_grad_(enabled)


    def route_logits(self, query):
        bank = torch.stack([
            self.fake_prompts[index]
            for index in range(self.active_tasks)
        ])
        signatures = F.normalize(bank.mean(dim=(1, 2)), dim=-1)
        return F.normalize(query, dim=-1) @ signatures.t() / self.router_temperature


    def _front_end(self, audio_data):
        processor_input = (
            [row.numpy() for row in audio_data.detach().cpu()]
            if self.ssl_backbone == 'w2vbert' else audio_data
        )
        processed = self.processor(
            processor_input, sampling_rate=16000, return_tensors='pt'
        )
        with torch.no_grad():
            if self.ssl_backbone == 'w2vbert':
                features = processed.input_features.to(audio_data.device)
                hidden, _ = self.model.feature_projection(features)
            else:
                values = processed.input_values.to(audio_data.device).squeeze(0)
                features = self.model.feature_extractor(values).transpose(1, 2)
                hidden, _ = self.model.feature_projection(features)
            query = F.normalize(hidden.mean(dim=1), dim=-1)
            if self.ssl_backbone != 'w2vbert':
                hidden = hidden + self.model.encoder.pos_conv_embed(hidden)
                if self.ssl_backbone == 'wavlm':
                    hidden = self.model.encoder.layer_norm(hidden)
            hidden = self.model.encoder.dropout(hidden)
        return hidden, query

    def extract_query(self, audio_data):
        return self._front_end(audio_data)[1]

    def initialize_router(self, real_centroid=None, fake_centroid=None):
        """Initialize the new key (and T0 group gate) from frozen queries."""
        with torch.no_grad():
            if (
                self.training_task == 0 and self.group_router is not None and
                real_centroid is not None and fake_centroid is not None
            ):
                self.group_router.weight[0].copy_(F.normalize(real_centroid, dim=0))
                self.group_router.weight[1].copy_(F.normalize(fake_centroid, dim=0))
                self.group_router.bias.zero_()

    def _router(self, query):
        fake_count = self.active_tasks
        if self.uniform_fake_router:
            beta = query.new_full((query.shape[0], fake_count), 1.0 / fake_count)
            fake_logits = beta.log()
        else:
            fake_logits = self.route_logits(query)
            soft_beta = F.softmax(fake_logits, dim=1)
            beta = soft_beta
            if (
                self.training and fake_count > 1 and
                self.router_floor > 0
            ):
                beta = (
                    (1.0 - self.router_floor) * beta +
                    self.router_floor / fake_count
                )

        if self.group_router is None:
            group_logits = None
            group_weights = query.new_ones((query.shape[0], 1))
        else:
            group_logits = self.group_router(query)
            group_weights = F.softmax(group_logits, dim=1)
        return group_logits, group_weights, fake_logits, beta


    def orthogonality_loss(self):
        """Scale-invariant overlap of current and historical fake subspaces."""
        if self.training_task == 0:
            return self.fake_prompts[0].new_zeros(())
        if self.fake_init_mode == 'inherit_response':
            current = (
                self.fake_prompts[self.training_task] -
                self.fake_prompt_bases[self.training_task]
            )
            old_values = [
                self.fake_prompts[index].detach() -
                self.fake_prompt_bases[index].detach()
                for index in range(self.training_task)
            ]
        else:
            current = self.fake_prompts[self.training_task]
            old_values = [
                prompt.detach()
                for prompt in self.fake_prompts[:self.training_task]
            ]
        current_basis = torch.linalg.qr(
            current.transpose(1, 2), mode='reduced'
        ).Q.transpose(1, 2)
        old = torch.cat(old_values, dim=1)
        old_basis = torch.linalg.qr(
            old.transpose(1, 2), mode='reduced'
        ).Q.transpose(1, 2)
        overlap = torch.einsum('lph,lqh->lpq', current_basis, old_basis)
        return overlap.pow(2).mean()

    def forward(
        self, audio_data, return_router=False, real_prompt_override=None,
        gap_gate_override=None, deterministic_gates=False,
        disable_prompt_dropout=False,
    ):
        hidden, query = self._front_end(audio_data)
        fake_bank = torch.stack([
            self.fake_prompts[index]
            for index in range(self.active_tasks)
        ])

        if self.joint_mode == 'none':
            group_logits, group_weights, fake_logits, beta = self._router(query)
        elif self.joint_mode == 'response':
            group_logits = None
            group_weights = query.new_ones((query.shape[0], 1))
            fake_logits, beta = response_soft_weights(
                query, fake_bank, self.router_temperature
            )
        else:
            group_logits = None
            group_weights = query.new_ones((query.shape[0], 1))
            fake_logits = None
            beta = query.new_full(
                (query.shape[0], self.active_tasks),
                1.0 / self.active_tasks,
            )

        response_sum = None
        layer_fake_logits = []
        layer_fake_weights = []
        layer_features = []
        real_sparse_router_losses = []
        real_sparse_selections = []
        position_bias = None
        for layer_index, layer in enumerate(self.model.encoder.layers):
            parts = []
            group_sizes = []
            if self.joint_mode == 'none':
                fake_prompt = torch.einsum(
                    'bt,tph->bph', beta, fake_bank[:, layer_index]
                )
                if self.real_prompt is not None:
                    real_prompt = self.real_prompt[layer_index].expand(
                        hidden.shape[0], -1, -1
                    )
                    if self.group_router is None:
                        parts.append(real_prompt)
                    else:
                        parts.append(
                            group_weights[:, 0, None, None] * real_prompt
                        )
                        fake_prompt = (
                            group_weights[:, 1, None, None] * fake_prompt
                        )
                parts.append(fake_prompt)
            else:
                if self.real_prompt is not None:
                    source_real_prompt = (
                        self.real_prompt if real_prompt_override is None
                        else real_prompt_override
                    )
                    real_prompt = source_real_prompt[
                        layer_index, :self.real_tokens
                    ].expand(hidden.shape[0], -1, -1)
                    parts.append(real_prompt)
                    group_sizes.append(real_prompt.shape[1])
                if self.joint_mode in ('uniform', 'normalized'):
                    fake_prompt = fake_bank[:, layer_index].mean(dim=0)
                    if self.joint_mode == 'normalized':
                        fake_prompt = F.normalize(fake_prompt, dim=-1)
                    fake_prompt = fake_prompt.expand(hidden.shape[0], -1, -1)
                    parts.append(fake_prompt)
                    group_sizes.append(self.fake_tokens)
                elif self.joint_mode == 'response':
                    fake_prompt = torch.einsum(
                        'bt,tph->bph', beta, fake_bank[:, layer_index]
                    )
                    parts.append(fake_prompt)
                    group_sizes.append(self.fake_tokens)
                elif self.joint_mode == 'single':
                    task_index = getattr(
                        self, 'inference_task', self.training_task
                    )
                    parts.append(
                        fake_bank[task_index, layer_index].expand(
                            hidden.shape[0], -1, -1
                        )
                    )
                    group_sizes.append(self.fake_tokens)
                else:
                    for task_index in range(self.active_tasks):
                        parts.append(
                            fake_bank[task_index, layer_index].expand(
                                hidden.shape[0], -1, -1
                            )
                        )
                        group_sizes.append(self.fake_tokens)
            prompts = torch.cat(parts, dim=1)
            if not disable_prompt_dropout:
                prompts = self.prompt_dropout(prompts)
            prompt_count = prompts.shape[1]
            hidden = torch.cat((prompts, hidden), dim=1)
            output_attentions = (
                self.joint_mode != 'none' and self.collect_prompt_response
            )
            if self.ssl_backbone == 'wavlm':
                layer_output = layer(
                    hidden, position_bias=position_bias,
                    output_attentions=output_attentions,
                )
                position_bias = layer_output[1]
            elif self.ssl_backbone == 'w2vbert':
                relative_position_embeddings = (
                    self.model.encoder.embed_positions(hidden)
                    if self.model.encoder.embed_positions is not None else None
                )
                layer_output = layer(
                    hidden,
                    relative_position_embeddings=relative_position_embeddings,
                    output_attentions=output_attentions,
                )
            else:
                layer_output = layer(
                    hidden, output_attentions=output_attentions,
                )
            hidden = layer_output[0]
            if self.joint_mode != 'none' and self.collect_prompt_response:
                attention = (
                    layer_output[2]
                    if self.ssl_backbone == 'wavlm' else layer_output[1]
                )
                audio_to_prompt = attention[
                    :, :, prompt_count:, :prompt_count
                ]
                layer_responses = []
                start = 0
                for size in group_sizes:
                    layer_responses.append(
                        audio_to_prompt[..., start:start + size]
                        .sum(dim=-1).mean(dim=(1, 2))
                    )
                    start += size
                values = torch.stack(layer_responses, dim=1)
                response_sum = (
                    values if response_sum is None else response_sum + values
                )
            hidden = hidden[:, prompt_count:, :]
            if self.collect_layer_features:
                layer_features.append(hidden.mean(dim=1))

        if layer_fake_weights:
            beta = torch.stack(layer_fake_weights, dim=0).mean(dim=0)
            if layer_fake_logits:
                fake_logits = torch.stack(layer_fake_logits, dim=0).mean(dim=0)

        prompt_response = None
        if response_sum is not None:
            prompt_response = response_sum / self.num_layers
            if self.joint_mode == 'concat':
                fake_response = prompt_response[:, 1:]
                beta = fake_response / (
                    fake_response.sum(dim=1, keepdim=True) + 1e-8
                )

        self.last_router = {
            'query': query,
            'group_logits': group_logits,
            'group_weights': group_weights,
            'fake_logits': fake_logits,
            'fake_weights': beta,
            'prompt_response': prompt_response,
            'joint_mode': self.joint_mode,
            'gap_gate_values': (
                None
            ),
            'real_sparse_router_loss': (
                torch.stack(real_sparse_router_losses).mean()
                if real_sparse_router_losses else query.new_zeros(())
            ),
            'real_sparse_selections': real_sparse_selections,
            'layer_features': (
                torch.stack(layer_features, dim=1)
                if layer_features else None
            ),
        }
        if return_router:
            return hidden, self.last_router
        return hidden

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, 'model'):
            self.model.eval()
        return self


class RFPromptAASIST(nn.Module):
    def __init__(self, model_dir, classifier_seed=2026, **prompt_kwargs):
        super().__init__()
        self.prompt_encoder = RFPromptEncoder(
            model_dir, **prompt_kwargs
        )
        # Ablations use different prompt shapes and therefore consume different
        # RNG counts. Isolate classifier initialization so every run starts from
        # exactly the same AASIST weights for a given experiment seed.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(classifier_seed)
            self.w2vaasist = SSLAASIST(
                input_dim=self.prompt_encoder.hidden_size
            )
        self.last_real_expert = None

    def set_task(self, task_index):
        self.prompt_encoder.set_task(task_index)
        enabled = True
        for parameter in self.w2vaasist.parameters():
            parameter.requires_grad_(enabled)

    def _forward_once(self, audio_data):
        features = self.prompt_encoder(audio_data)
        return self.w2vaasist(features)

    def forward(self, audio_data):
        self.last_real_expert = None
        return self._forward_once(audio_data)

    def forward_with_prompt_meta(self, audio_data):
        self.last_real_expert = None
        features, router = self.prompt_encoder(
            audio_data, return_router=True
        )
        hidden, logits = self.w2vaasist(features)
        return hidden, logits, router

    def train(self, mode=True):
        super().train(mode)
        self.prompt_encoder.model.eval()
        return self
