#!/usr/bin/env python3
"""Continual trainer and evaluator used by RF-Prompt and matched baselines."""

import csv
import copy
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from dataset import ProtocolManifestDataset
import eval_metrics as em
from model import WAVLMAASIST, W2V2BERTAASIST, XLSRAASIST
from continual.orthogonal_prompt import RFPromptAASIST


LEGACY_TASKS = (
    ('T0', 'ASVspoof2019-LA', 't0_asv19'),
    ('T1', 'ASVspoof5-Track1', 't1_asv5'),
    ('T2', 'Codecfake', 't2_codecfake'),
    ('T3', 'ATADD-Track2-speech', 't3_atadd_t2_speech'),
)
TASK_LAYOUTS = {
    'legacy': LEGACY_TASKS,
    'a': (
        ('A0', 'ASVspoof2019-LA', 'a0_asv19'),
        ('A1', 'ASVspoof5-Track1', 'a1_asv5'),
        ('A2', 'Codecfake', 'a2_codecfake'),
        ('A3', 'ATADD-Track2-speech', 'a3_atadd_t2_speech'),
    ),
    'b': (
        ('B0', 'Classical synthesis / conversion', 'b0_classical_waveform'),
        ('B1', 'Continuous-representation neural generation', 'b1_neural_vocoder'),
        ('B2', 'Discrete neural codec reconstruction', 'b2_neural_codec'),
        ('B3', 'Discrete-token generative modeling', 'b3_codec_token_alm'),
    ),
    'joint': (('J0', 'Joint offline upper bound', '.'),),
}
TASKS = LEGACY_TASKS


def configure_tasks(layout):
    global TASKS
    if layout not in TASK_LAYOUTS:
        raise ValueError(f'Unknown task layout: {layout}')
    TASKS = TASK_LAYOUTS[layout]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class SequentialMethod:
    name = 'sequential'

    def before_task(self, model, task_index):
        del model, task_index

    def loss(self, model, waveform, labels):
        _, logits = model(waveform)
        return F.cross_entropy(logits, labels), logits

    def initialize_task(self, model, loader, device):
        del model, loader, device

    def after_backward(self, model):
        del model

    def after_optimizer_step(self, model):
        del model

    def after_task(self, model, train_loader, device):
        del model, train_loader, device

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        if state_dict:
            raise ValueError('Sequential method has no persistent state')


class OrthogonalPromptMethod(SequentialMethod):
    """Classification + real/fake balancing + fake-subspace orthogonality."""

    name = 'oprompt'

    def __init__(
        self, orthogonal_weight=1.0, balance_weight=0.1,
        router_init_samples=512, key_match_weight=0.0,
        llb_weight=0.0, llb_delta=0.2,
    ):
        self.orthogonal_weight = orthogonal_weight
        self.balance_weight = balance_weight
        self.router_init_samples = router_init_samples
        self.key_match_weight = key_match_weight
        self.llb_weight = llb_weight
        self.llb_delta = llb_delta
        self.last_terms = {}
        self.last_hidden = None

    def before_task(self, model, task_index):
        model.set_task(task_index)

    def initialize_task(self, model, loader, device):
        if (
            getattr(model.prompt_encoder, 'fake_init_mode', 'orthogonal') ==
            'inherit_response' and model.prompt_encoder.training_task > 0
        ):
            fake_queries = []
            fake_count = 0
            model.eval()
            with torch.no_grad():
                for waveform, _, labels in loader:
                    fake = labels == 1
                    if not fake.any():
                        continue
                    waveform = waveform[fake].to(device, non_blocking=True)
                    query = model.prompt_encoder.extract_query(waveform).cpu()
                    values = query[:self.router_init_samples - fake_count]
                    fake_queries.append(values)
                    fake_count += len(values)
                    if fake_count >= self.router_init_samples:
                        break
            fake_query = (
                torch.cat(fake_queries).mean(dim=0).to(device)
                if fake_queries else None
            )
            initialized = model.prompt_encoder.initialize_fake_prompt(fake_query)
            if initialized is None:
                raise RuntimeError('Inherited Fake Prompt initialization failed')
            parent_index, scores = initialized
            print(
                f'FAKE_PROMPT_INIT mode=inherit_response samples={fake_count} '
                f'parent={parent_index} scores={scores.cpu().tolist()}',
                flush=True,
            )
        if model.prompt_encoder.joint_mode != 'none':
            print(
                f'JOINT_PROMPT mode={model.prompt_encoder.joint_mode} '
                'router_initialization=skipped',
                flush=True,
            )
            return
        real_queries = []
        fake_queries = []
        real_count = 0
        fake_count = 0
        model.eval()
        with torch.no_grad():
            for waveform, _, labels in loader:
                waveform = waveform.to(device, non_blocking=True)
                query = model.prompt_encoder.extract_query(waveform).cpu()
                real = labels == 0
                fake = labels == 1
                if real.any() and real_count < self.router_init_samples:
                    values = query[real][:self.router_init_samples - real_count]
                    real_queries.append(values)
                    real_count += len(values)
                if fake.any() and fake_count < self.router_init_samples:
                    values = query[fake][:self.router_init_samples - fake_count]
                    fake_queries.append(values)
                    fake_count += len(values)
                if (
                    real_count >= self.router_init_samples and
                    fake_count >= self.router_init_samples
                ):
                    break
        real_centroid = (
            torch.cat(real_queries).mean(dim=0).to(device)
            if real_queries else None
        )
        fake_centroid = (
            torch.cat(fake_queries).mean(dim=0).to(device)
            if fake_queries else None
        )
        model.prompt_encoder.initialize_router(real_centroid, fake_centroid)
        print(
            f'ROUTER_INIT real_samples={real_count} fake_samples={fake_count}',
            flush=True,
        )

    def loss(self, model, waveform, labels):
        hidden, logits, router = model.forward_with_prompt_meta(waveform)
        self.last_hidden = hidden
        classification = F.cross_entropy(logits, labels)
        if router['group_logits'] is None:
            balance = classification.new_zeros(())
        else:
            balance = F.cross_entropy(router['group_logits'], labels)
        orthogonal = model.prompt_encoder.orthogonality_loss()
        llb = classification.new_zeros(())
        response = router.get('prompt_response')
        if self.llb_weight > 0:
            if response is None:
                raise RuntimeError('Prompt LLB requires prompt responses')
            normalized = response / (
                response.sum(dim=1, keepdim=True) + 1e-8
            )
            coefficient = torch.full_like(
                normalized, 1.0 + self.llb_delta
            )
            real = labels == 0
            fake = labels == 1
            coefficient[real, 0] = 1.0 - self.llb_delta
            coefficient[fake, 1:] = 1.0 - self.llb_delta
            balanced_response = normalized * coefficient
            llb = (
                balanced_response.var(dim=1, unbiased=False) /
                (balanced_response.mean(dim=1) + 1e-8)
            ).mean()
        key_match = classification.new_zeros(())
        if self.key_match_weight > 0:
            fake = labels == 1
            if fake.any():
                current = model.prompt_encoder.fake_prompts[
                    model.prompt_encoder.training_task
                ]
                key = F.normalize(current.mean(dim=(0, 1)), dim=0)
                queries = F.normalize(router['query'][fake], dim=-1)
                key_match = (1.0 - queries @ key).mean()
        loss = (
            classification + self.balance_weight * balance +
            self.orthogonal_weight * orthogonal +
            self.key_match_weight * key_match + self.llb_weight * llb
        )
        self.last_terms = {
            'classification_loss': float(classification.detach()),
            'balance_loss': float(balance.detach()),
            'orthogonal_loss': float(orthogonal.detach()),
            'key_match_loss': float(key_match.detach()),
            'llb_loss': float(llb.detach()),
        }
        return loss, logits

    def state_dict(self):
        return {
            'orthogonal_weight': self.orthogonal_weight,
            'balance_weight': self.balance_weight,
            'router_init_samples': self.router_init_samples,
            'key_match_weight': self.key_match_weight,
            'llb_weight': self.llb_weight,
            'llb_delta': self.llb_delta,
        }

    def load_state_dict(self, state_dict):
        self.orthogonal_weight = state_dict.get(
            'orthogonal_weight', self.orthogonal_weight
        )
        self.balance_weight = state_dict.get(
            'balance_weight', self.balance_weight
        )
        self.router_init_samples = state_dict.get(
            'router_init_samples', self.router_init_samples
        )
        self.key_match_weight = state_dict.get(
            'key_match_weight', self.key_match_weight
        )
        self.llb_weight = state_dict.get('llb_weight', self.llb_weight)
        self.llb_delta = state_dict.get('llb_delta', self.llb_delta)


class RealProtectedPromptMethod(OrthogonalPromptMethod):
    """Keep one plastic Real Prompt while protecting its historical function."""

    TEACHER_MODES = {'logit_kd', 'full_kd', 'opd'}

    def __init__(
        self, *args, protection='none', anchor_weight=0.1,
        kd_weight=1.0, feature_weight=1.0, temperature=2.0,
        sample_fraction=0.25, ema_decay=0.999,
        stat_max_batches=16, projection_rank=8,
        projection_max_rank=32, real_freeze_layers=0,
        protect_layer_count=0,
        spd_weight=0.0, spd_real_only=True, spd_layer_count=0,
        spd_kind='feature',
        real_update_layers='all', real_update_top_k=8,
        smope_router_weight=0.0, smope_prototype_weight=0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        allowed = {
            'none', 'proximal', 'fisher', 'gradient_projection', 'ema',
            'logit_kd', 'full_kd', 'opd', 'pcgrad_global',
            'pcgrad_layerwise',
        }
        if protection not in allowed:
            raise ValueError(f'Unknown Real Prompt protection: {protection}')
        self.protection = protection
        self.anchor_weight = anchor_weight
        self.kd_weight = kd_weight
        self.feature_weight = feature_weight
        self.temperature = temperature
        self.sample_fraction = sample_fraction
        self.ema_decay = ema_decay
        self.stat_max_batches = stat_max_batches
        self.projection_rank = projection_rank
        self.projection_max_rank = projection_max_rank
        self.real_freeze_layers = real_freeze_layers
        self.protect_layer_count = protect_layer_count
        if spd_kind not in {'feature', 'prompt_cosine', 'feature_clean'}:
            raise ValueError(f'Unknown SPD kind: {spd_kind}')
        self.spd_kind = spd_kind
        self.spd_weight = spd_weight
        self.spd_real_only = spd_real_only
        self.spd_layer_count = spd_layer_count
        if real_update_layers not in {
            'all', 'early8', 'middle8', 'late8', 'gap_topk'
        }:
            raise ValueError(f'Unknown Real update layer mode: {real_update_layers}')
        if real_update_top_k < 1:
            raise ValueError('Real update Top-K must be positive')
        self.real_update_layers = real_update_layers
        self.real_update_top_k = real_update_top_k
        self.smope_router_weight = smope_router_weight
        self.smope_prototype_weight = smope_prototype_weight
        self.real_anchor = None
        self.real_fisher = None
        self.real_gradient_basis = None
        self.ema_prompt = None
        self.teacher_state = None
        self.teacher = None
        self.task_index = 0
        self.pcgrad_real = None
        self.pcgrad_fake = None
        self.pcgrad_fake_fraction = 0.0
        self.spd_gradient = None
        self.spd_teacher_prompt = None
        self.real_update_mask = None
        self.smope_prototypes = []
        self.frozen_real_layers = None

    def _protected_tensors(self, *tensors):
        count = self.protect_layer_count
        if count <= 0:
            count = tensors[0].shape[0]
        count = min(count, tensors[0].shape[0])
        return tuple(tensor[:count] for tensor in tensors)

    def _real_layer_masks(self, router, batch_size, layer_count, device):
        if self.real_update_layers == 'gap_topk':
            gates = router.get('gap_gate_values')
            if gates is None:
                raise RuntimeError('Adaptive Real layer updates require GAP gates')
            task_index = min(self.task_index, gates.shape[1] - 1)
            values = gates[:, task_index]
            top_k = min(self.real_update_top_k, values.shape[1])
            selected = values.topk(top_k, dim=1).indices
            masks = torch.zeros_like(values)
            masks.scatter_(1, selected, 1.0)
            return masks
        mask = torch.zeros(layer_count, device=device)
        if self.real_update_layers == 'all':
            mask.fill_(1.0)
        elif self.real_update_layers == 'early8':
            mask[:min(8, layer_count)] = 1.0
        elif self.real_update_layers == 'middle8':
            mask[min(8, layer_count):min(16, layer_count)] = 1.0
        elif self.real_update_layers == 'late8':
            mask[max(0, layer_count - 8):] = 1.0
        return mask.expand(batch_size, -1)

    @staticmethod
    def _capture_teacher_state(model):
        return {
            'task_count': model.prompt_encoder.task_count,
            'active_tasks': model.prompt_encoder.active_tasks,
            'training_task': model.prompt_encoder.training_task,
            'prompt_encoder': {
                key: value.detach().cpu()
                for key, value in model.prompt_encoder.state_dict().items()
                if not key.startswith('model.')
            },
            'w2vaasist': {
                key: value.detach().cpu()
                for key, value in model.w2vaasist.state_dict().items()
            },
        }

    def _build_teacher(self, model):
        if self.teacher_state is None:
            return None
        teacher = copy.deepcopy(model)
        teacher.prompt_encoder.ensure_task_count(
            self.teacher_state['task_count']
        )
        teacher.prompt_encoder.load_state_dict(
            self.teacher_state['prompt_encoder'], strict=False
        )
        teacher.w2vaasist.load_state_dict(self.teacher_state['w2vaasist'])
        teacher.prompt_encoder.active_tasks = self.teacher_state['active_tasks']
        teacher.prompt_encoder.training_task = self.teacher_state['training_task']
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        return teacher

    def before_task(self, model, task_index):
        self.task_index = task_index
        device = next(model.parameters()).device
        for name in (
            'real_anchor', 'real_fisher', 'real_gradient_basis', 'ema_prompt',
            'spd_teacher_prompt',
        ):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, value.to(device))
        if task_index > 0 and self.protection in self.TEACHER_MODES:
            self.teacher = self._build_teacher(model)
            if self.teacher is None:
                raise RuntimeError('Real distillation requires a previous teacher')
        else:
            self.teacher = None
        super().before_task(model, task_index)
        prompt = model.prompt_encoder.real_prompt
        if task_index > 0 and prompt is not None and self.real_freeze_layers > 0:
            count = min(self.real_freeze_layers, prompt.shape[0])
            self.frozen_real_layers = prompt[:count].detach().clone()
        else:
            self.frozen_real_layers = None

    @staticmethod
    def _relative_anchor_penalty(prompt, anchor, importance=None):
        scale = anchor.detach().pow(2).mean().clamp_min(1e-8)
        difference = (prompt - anchor).pow(2)
        if importance is not None:
            normalized = importance / importance.mean().clamp_min(1e-8)
            difference = difference * normalized
        return difference.mean() / scale

    def _select_real_indices(self, logits, labels):
        indices = torch.nonzero(labels == 0, as_tuple=False).flatten()
        if indices.numel() == 0:
            return indices
        count = max(1, int(math.ceil(indices.numel() * self.sample_fraction)))
        if self.protection == 'opd':
            fake_probability = F.softmax(logits.detach(), dim=1)[indices, 1]
            return indices[fake_probability.topk(count).indices]
        order = torch.randperm(indices.numel(), device=indices.device)
        return indices[order[:count]]

    def loss(self, model, waveform, labels):
        loss, logits = super().loss(model, waveform, labels)
        zero = loss.new_zeros(())
        anchor_loss = zero
        logit_kd = zero
        feature_kd = zero
        spd_loss = zero
        smope_router_loss = zero
        smope_prototype_loss = zero
        selected_fraction = 0.0
        prompt = model.prompt_encoder.real_prompt
        current_router = model.prompt_encoder.last_router
        if prompt is not None:
            sample_layer_masks = self._real_layer_masks(
                current_router, labels.shape[0], prompt.shape[0], prompt.device
            )
            self.real_update_mask = sample_layer_masks.mean(dim=0).detach()
        else:
            sample_layer_masks = None
            self.real_update_mask = None

        self.pcgrad_real = None
        self.pcgrad_fake = None
        self.pcgrad_fake_fraction = 0.0
        self.spd_gradient = None

        if (
            self.task_index > 0 and
            self.protection in {'pcgrad_global', 'pcgrad_layerwise'} and
            prompt is not None and prompt.requires_grad
        ):
            real = labels == 0
            fake = labels == 1
            if real.any() and fake.any():
                real_objective = F.cross_entropy(logits[real], labels[real])
                fake_objective = F.cross_entropy(logits[fake], labels[fake])
                self.pcgrad_real = torch.autograd.grad(
                    real_objective, prompt, retain_graph=True,
                    allow_unused=True,
                )[0]
                self.pcgrad_fake = torch.autograd.grad(
                    fake_objective, prompt, retain_graph=True,
                    allow_unused=True,
                )[0]
                self.pcgrad_fake_fraction = float(fake.float().mean())

        if (
            self.task_index > 0 and self.spd_weight > 0 and
            self.spd_teacher_prompt is not None and
            prompt is not None and prompt.requires_grad
        ):
            if self.spd_kind == 'prompt_cosine':
                from continual.prompt_cosine import prompt_cosine_anchor
                spd_loss = prompt_cosine_anchor(
                    prompt, self.spd_teacher_prompt, self.spd_layer_count
                )
                self.spd_gradient = torch.autograd.grad(
                    spd_loss, prompt, retain_graph=True, allow_unused=True
                )[0]
            else:
                selected = labels == 0 if self.spd_real_only else torch.ones_like(
                    labels, dtype=torch.bool
                )
                if selected.any():
                    selected_layer_masks = sample_layer_masks[selected]
                    gate_override = current_router.get('gap_gate_values')
                    if gate_override is not None:
                        gate_override = gate_override[selected].detach()
                    # Classification remains stochastic. The clean control uses
                    # two additional deterministic prompt-encoder passes for SPD.
                    # Preserve the main-pass router metadata used elsewhere.
                    saved_router = model.prompt_encoder.last_router
                    try:
                        if self.spd_kind == 'feature_clean':
                            _, student_router = model.prompt_encoder(
                                waveform[selected], return_router=True,
                                gap_gate_override=gate_override,
                                deterministic_gates=True,
                                disable_prompt_dropout=True,
                            )
                            student_features = student_router['layer_features']
                        else:
                            current_features = current_router.get('layer_features')
                            if current_features is None:
                                raise RuntimeError('SPD requires collected layer features')
                            student_features = current_features[selected]
                        with torch.no_grad():
                            _, teacher_router = model.prompt_encoder(
                                waveform[selected], return_router=True,
                                real_prompt_override=self.spd_teacher_prompt,
                                gap_gate_override=gate_override,
                                deterministic_gates=True,
                                disable_prompt_dropout=True,
                            )
                            teacher_features = teacher_router['layer_features']
                    finally:
                        model.prompt_encoder.last_router = saved_router
                    if self.spd_layer_count > 0:
                        legacy = torch.zeros_like(selected_layer_masks)
                        legacy[:, :self.spd_layer_count] = 1.0
                        selected_layer_masks = selected_layer_masks * legacy
                    layer_losses = 1.0 - F.cosine_similarity(
                        student_features, teacher_features, dim=-1
                    )
                    spd_loss = (
                        layer_losses * selected_layer_masks
                    ).sum() / selected_layer_masks.sum().clamp_min(1.0)
                    self.spd_gradient = torch.autograd.grad(
                        spd_loss, prompt, retain_graph=True, allow_unused=True
                    )[0]

        if self.task_index > 0 and prompt is not None:
            if self.protection == 'proximal' and self.real_anchor is not None:
                protected, anchor = self._protected_tensors(
                    prompt, self.real_anchor
                )
                anchor_loss = self._relative_anchor_penalty(protected, anchor)
            elif self.protection == 'fisher' and self.real_anchor is not None:
                protected, anchor, importance = self._protected_tensors(
                    prompt, self.real_anchor, self.real_fisher
                )
                anchor_loss = self._relative_anchor_penalty(
                    protected, anchor, importance
                )
            elif self.protection == 'ema' and self.ema_prompt is not None:
                protected, ema = self._protected_tensors(
                    prompt, self.ema_prompt
                )
                with torch.no_grad():
                    ema.mul_(self.ema_decay).add_(
                        protected.detach(), alpha=1.0 - self.ema_decay
                    )
                anchor_loss = self._relative_anchor_penalty(protected, ema)

            if self.protection in self.TEACHER_MODES and self.teacher is not None:
                selected = self._select_real_indices(logits, labels)
                if selected.numel() > 0:
                    real_count = max(int((labels == 0).sum()), 1)
                    selected_fraction = float(selected.numel() / real_count)
                    with torch.no_grad():
                        teacher_hidden, teacher_logits = self.teacher(
                            waveform[selected]
                        )
                    temperature = self.temperature
                    logit_kd = F.kl_div(
                        F.log_softmax(logits[selected] / temperature, dim=1),
                        F.softmax(teacher_logits / temperature, dim=1),
                        reduction='batchmean',
                    ) * (temperature ** 2)
                    if self.protection in {'full_kd', 'opd'}:
                        feature_kd = 1.0 - F.cosine_similarity(
                            self.last_hidden[selected], teacher_hidden, dim=-1
                        ).mean()

        loss = (
            loss + self.anchor_weight * anchor_loss +
            self.kd_weight * (logit_kd + self.feature_weight * feature_kd) +
            self.spd_weight * spd_loss.detach() +
            self.smope_router_weight * smope_router_loss +
            self.smope_prototype_weight * smope_prototype_loss
        )
        self.last_terms.update({
            'real_anchor_loss': float(anchor_loss.detach()),
            'real_logit_kd_loss': float(logit_kd.detach()),
            'real_feature_kd_loss': float(feature_kd.detach()),
            'real_selected_fraction': selected_fraction,
            'real_fake_gradient_conflict_rate': 0.0,
            'real_fake_gradient_cosine': 0.0,
            'shared_prompt_distillation_loss': float(spd_loss.detach()),
            'smope_real_router_loss': float(smope_router_loss.detach()),
            'smope_real_prototype_loss': float(
                smope_prototype_loss.detach()
            ),
        })
        return loss, logits

    def after_backward(self, model):
        prompt = model.prompt_encoder.real_prompt
        if (
            self.spd_gradient is not None and prompt is not None and
            prompt.grad is not None
        ):
            prompt.grad.add_(self.spd_gradient, alpha=self.spd_weight)
        if (
            self.protection in {'pcgrad_global', 'pcgrad_layerwise'} and
            self.pcgrad_real is not None and self.pcgrad_fake is not None and
            prompt is not None and prompt.grad is not None
        ):
            real_gradient = self.pcgrad_real.detach()
            fake_gradient = self.pcgrad_fake.detach()
            eps = torch.finfo(fake_gradient.dtype).eps
            if self.protection == 'pcgrad_global':
                real_flat = real_gradient.reshape(-1)
                fake_flat = fake_gradient.reshape(-1)
                dot = torch.dot(real_flat, fake_flat)
                denominator = torch.dot(real_flat, real_flat).clamp_min(eps)
                if dot < 0:
                    correction = -(dot / denominator) * real_gradient
                    prompt.grad.add_(
                        correction, alpha=self.pcgrad_fake_fraction
                    )
                conflict_rate = float(dot < 0)
                cosine = float(
                    dot / (
                        real_flat.norm() * fake_flat.norm()
                    ).clamp_min(eps)
                )
            else:
                real_flat = real_gradient.flatten(1)
                fake_flat = fake_gradient.flatten(1)
                dots = (real_flat * fake_flat).sum(dim=1)
                denominators = real_flat.pow(2).sum(dim=1).clamp_min(eps)
                conflict = dots < 0
                coefficients = torch.where(
                    conflict, -dots / denominators, torch.zeros_like(dots)
                )
                correction = coefficients[:, None, None] * real_gradient
                prompt.grad.add_(
                    correction, alpha=self.pcgrad_fake_fraction
                )
                cosines = dots / (
                    real_flat.norm(dim=1) * fake_flat.norm(dim=1)
                ).clamp_min(eps)
                conflict_rate = float(conflict.float().mean())
                cosine = float(cosines.mean())
            self.last_terms.update({
                'real_fake_gradient_conflict_rate': conflict_rate,
                'real_fake_gradient_cosine': cosine,
            })
        if (
            self.protection == 'gradient_projection' and
            self.real_gradient_basis is not None and
            prompt is not None and prompt.grad is not None
        ):
            flat = prompt.grad.reshape(-1)
            basis = self.real_gradient_basis.to(flat.device)
            flat.sub_(basis.t() @ (basis @ flat))
        if (
            self.task_index > 0 and prompt is not None and
            prompt.grad is not None and self.real_freeze_layers > 0
        ):
            prompt.grad[:self.real_freeze_layers].zero_()
        if (
            self.task_index > 0 and prompt is not None and
            prompt.grad is not None and self.real_update_mask is not None
        ):
            prompt.grad.mul_(
                self.real_update_mask[:, None, None].to(prompt.grad)
            )

    def after_optimizer_step(self, model):
        prompt = model.prompt_encoder.real_prompt
        if (
            self.frozen_real_layers is not None and prompt is not None and
            self.task_index > 0
        ):
            count = self.frozen_real_layers.shape[0]
            with torch.no_grad():
                prompt[:count].copy_(self.frozen_real_layers)

    def _real_gradients(self, model, train_loader, device):
        gradients = []
        model.eval()
        for batch_index, (waveform, _, labels) in enumerate(train_loader):
            if batch_index >= self.stat_max_batches:
                break
            real = labels == 0
            if not real.any():
                continue
            waveform = waveform[real].to(device, non_blocking=True)
            targets = torch.zeros(
                waveform.shape[0], dtype=torch.long, device=device
            )
            model.zero_grad(set_to_none=True)
            _, logits = model(waveform)
            F.cross_entropy(logits, targets).backward()
            gradient = model.prompt_encoder.real_prompt.grad
            if gradient is not None:
                gradients.append(gradient.detach().flatten().clone())
        model.zero_grad(set_to_none=True)
        return gradients

    def after_task(self, model, train_loader, device):
        if self.protection in {'fisher', 'gradient_projection'}:
            gradients = self._real_gradients(model, train_loader, device)
            if gradients and self.protection == 'fisher':
                current = torch.stack(gradients).pow(2).mean(dim=0).reshape_as(
                    model.prompt_encoder.real_prompt
                )
                self.real_fisher = (
                    current if self.real_fisher is None
                    else self.real_fisher.to(device) + current
                )
            elif gradients:
                matrix = torch.stack(gradients)
                _, _, right = torch.linalg.svd(matrix, full_matrices=False)
                new_basis = right[:min(self.projection_rank, right.shape[0])]
                if self.real_gradient_basis is not None:
                    new_basis = torch.cat([
                        self.real_gradient_basis.to(device), new_basis
                    ], dim=0)
                orthogonal = torch.linalg.qr(
                    new_basis.t(), mode='reduced'
                ).Q.t()
                self.real_gradient_basis = orthogonal[
                    :self.projection_max_rank
                ].detach()

        prompt = model.prompt_encoder.real_prompt
        if prompt is not None:
            self.real_anchor = prompt.detach().clone()
            if self.spd_weight > 0:
                self.spd_teacher_prompt = prompt.detach().clone()
            if self.protection == 'ema' and self.ema_prompt is None:
                self.ema_prompt = prompt.detach().clone()
        if self.protection in self.TEACHER_MODES:
            self.teacher_state = self._capture_teacher_state(model)
        self.teacher = None

    def state_dict(self):
        state = super().state_dict()
        state.update({
            'protection': self.protection,
            'anchor_weight': self.anchor_weight,
            'kd_weight': self.kd_weight,
            'feature_weight': self.feature_weight,
            'temperature': self.temperature,
            'sample_fraction': self.sample_fraction,
            'ema_decay': self.ema_decay,
            'stat_max_batches': self.stat_max_batches,
            'projection_rank': self.projection_rank,
            'projection_max_rank': self.projection_max_rank,
            'real_freeze_layers': self.real_freeze_layers,
            'protect_layer_count': self.protect_layer_count,
            'spd_kind': self.spd_kind,
            'spd_weight': self.spd_weight,
            'spd_real_only': self.spd_real_only,
            'spd_layer_count': self.spd_layer_count,
            'spd_teacher_prompt': (
                None if self.spd_teacher_prompt is None
                else self.spd_teacher_prompt.cpu()
            ),
            'real_update_layers': self.real_update_layers,
            'real_update_top_k': self.real_update_top_k,
            'smope_router_weight': self.smope_router_weight,
            'smope_prototype_weight': self.smope_prototype_weight,
            'smope_prototypes': self.smope_prototypes,
            'real_anchor': None if self.real_anchor is None else self.real_anchor.cpu(),
            'real_fisher': None if self.real_fisher is None else self.real_fisher.cpu(),
            'real_gradient_basis': (
                None if self.real_gradient_basis is None
                else self.real_gradient_basis.cpu()
            ),
            'ema_prompt': None if self.ema_prompt is None else self.ema_prompt.cpu(),
            'teacher_state': self.teacher_state,
            'task_index': self.task_index,
        })
        return state

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        for name in (
            'protection', 'anchor_weight', 'kd_weight', 'feature_weight',
            'temperature', 'sample_fraction', 'ema_decay', 'stat_max_batches',
            'projection_rank', 'projection_max_rank', 'real_freeze_layers',
            'protect_layer_count',
            'spd_kind', 'spd_weight', 'spd_real_only', 'spd_layer_count',
            'spd_teacher_prompt',
            'real_update_layers', 'real_update_top_k',
            'smope_router_weight', 'smope_prototype_weight',
            'smope_prototypes',
            'real_anchor',
            'real_fisher', 'real_gradient_basis', 'ema_prompt',
            'teacher_state', 'task_index',
        ):
            if name in state_dict:
                setattr(self, name, state_dict[name])


class EWCMethod(SequentialMethod):
    name = 'ewc'

    def __init__(self, strength=100.0, decay=1.0, fisher_max_batches=None):
        self.strength = strength
        self.decay = decay
        self.fisher_max_batches = fisher_max_batches
        self.fisher = {}
        self.means = {}

    def loss(self, model, waveform, labels):
        base_loss, logits = super().loss(model, waveform, labels)
        if not self.fisher:
            return base_loss, logits
        penalty = torch.zeros((), device=waveform.device)
        for name, parameter in model.named_parameters():
            if name in self.fisher:
                penalty = penalty + (
                    self.fisher[name] * (parameter - self.means[name]).pow(2)
                ).sum()
        return base_loss + 0.5 * self.strength * penalty, logits

    def after_task(self, model, train_loader, device):
        new_fisher = {
            name: torch.zeros_like(parameter, device=device)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        model.eval()
        batches = 0
        for batch_index, (waveform, _, labels) in enumerate(train_loader):
            if (
                self.fisher_max_batches is not None and
                batch_index >= self.fisher_max_batches
            ):
                break
            waveform = waveform.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            model.zero_grad(set_to_none=True)
            _, logits = model(waveform)
            F.cross_entropy(logits, labels).backward()
            for name, parameter in model.named_parameters():
                if name in new_fisher and parameter.grad is not None:
                    new_fisher[name].add_(parameter.grad.detach().pow(2))
            batches += 1
        for name in new_fisher:
            new_fisher[name].div_(max(batches, 1))
            if name in self.fisher:
                # Best-dev checkpoints restore method tensors on CPU.
                old_fisher = self.fisher[name].to(new_fisher[name])
                new_fisher[name].add_(old_fisher, alpha=self.decay)
        self.fisher = new_fisher
        self.means = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        model.zero_grad(set_to_none=True)

    def state_dict(self):
        return {
            'strength': self.strength,
            'decay': self.decay,
            'fisher_max_batches': self.fisher_max_batches,
            'fisher': {key: value.cpu() for key, value in self.fisher.items()},
            'means': {key: value.cpu() for key, value in self.means.items()},
        }

    def load_state_dict(self, state_dict):
        self.strength = state_dict.get('strength', self.strength)
        self.decay = state_dict.get('decay', self.decay)
        self.fisher_max_batches = state_dict.get(
            'fisher_max_batches', self.fisher_max_batches
        )
        self.fisher = state_dict.get('fisher', {})
        self.means = state_dict.get('means', {})

    def before_task(self, model, task_index):
        del task_index
        device = next(model.parameters()).device
        self.fisher = {key: value.to(device) for key, value in self.fisher.items()}
        self.means = {key: value.to(device) for key, value in self.means.items()}


class LwFMethod(SequentialMethod):
    name = 'lwf'

    def __init__(self, alpha=1.0, temperature=2.0):
        self.alpha = alpha
        self.temperature = temperature
        self.teacher = None
        self.teacher_state = None

    def before_task(self, model, task_index):
        del task_index
        if self.teacher_state is not None:
            self.teacher = copy.deepcopy(model.w2vaasist)
            self.teacher.load_state_dict(self.teacher_state)
            self.teacher.to(next(model.parameters()).device)
            self.teacher.eval()
            for parameter in self.teacher.parameters():
                parameter.requires_grad = False

    def loss(self, model, waveform, labels):
        features = model.wav2vec2.extract_features(waveform)
        _, logits = model.w2vaasist(features)
        base_loss = F.cross_entropy(logits, labels)
        if self.teacher is None:
            return base_loss, logits
        with torch.no_grad():
            _, teacher_logits = self.teacher(features.detach())
        temperature = self.temperature
        distillation = F.kl_div(
            F.log_softmax(logits / temperature, dim=1),
            F.softmax(teacher_logits / temperature, dim=1),
            reduction='batchmean',
        ) * (temperature ** 2)
        return base_loss + self.alpha * distillation, logits

    def after_task(self, model, train_loader, device):
        del train_loader, device
        self.teacher = copy.deepcopy(model.w2vaasist)
        self.teacher.eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad = False
        self.teacher_state = {
            key: value.detach().cpu()
            for key, value in self.teacher.state_dict().items()
        }

    def state_dict(self):
        return {
            'alpha': self.alpha,
            'temperature': self.temperature,
            'teacher_state': self.teacher_state,
        }

    def load_state_dict(self, state_dict):
        self.alpha = state_dict.get('alpha', self.alpha)
        self.temperature = state_dict.get('temperature', self.temperature)
        self.teacher_state = state_dict.get('teacher_state')


class OWMMethod(SequentialMethod):
    """Official OWM update adapted to AASIST's final linear layer."""

    name = 'owm'

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.task_index = 0
        self.projector = None
        self.last_hidden = None

    def before_task(self, model, task_index):
        self.task_index = task_index
        dimension = model.w2vaasist.out_layer.in_features
        device = next(model.parameters()).device
        if self.projector is None:
            self.projector = torch.eye(dimension, device=device)
        else:
            self.projector = self.projector.to(device)

    def loss(self, model, waveform, labels):
        hidden, logits = model(waveform)
        self.last_hidden = hidden.detach()
        return F.cross_entropy(logits, labels), logits

    def _update_projector(self):
        representation = self.last_hidden.mean(dim=0, keepdim=True)
        gain = self.projector @ representation.t()
        denominator = self.alpha + representation @ gain
        self.projector.sub_(gain @ gain.t() / denominator)
        self.projector.div_(self.projector.norm(p='fro').clamp_min(1e-12))

    def after_backward(self, model):
        if self.task_index == 0:
            return
        with torch.no_grad():
            self._update_projector()
            weight = model.w2vaasist.out_layer.weight
            weight.grad.copy_(weight.grad @ self.projector.t())

    def state_dict(self):
        return {
            'alpha': self.alpha,
            'projector': None if self.projector is None else self.projector.cpu(),
        }

    def load_state_dict(self, state_dict):
        self.alpha = state_dict.get('alpha', self.alpha)
        self.projector = state_dict.get('projector')


class RWMMethod(OWMMethod):
    """Official RWM rotation adapted to binary real/fake AASIST logits."""

    name = 'rwm'

    def __init__(self, alpha=1.0):
        super().__init__(alpha)
        self.complement = None
        self.last_logits = None
        self.last_labels = None

    def before_task(self, model, task_index):
        super().before_task(model, task_index)
        if self.complement is None:
            self.complement = torch.eye(
                self.projector.shape[0], device=self.projector.device
            )
        else:
            self.complement = self.complement.to(self.projector.device)

    def loss(self, model, waveform, labels):
        loss, logits = super().loss(model, waveform, labels)
        self.last_logits = logits.detach()
        self.last_labels = labels.detach()
        return loss, logits

    def _rotation_beta(self):
        correct_logits = self.last_logits.gather(
            1, self.last_labels.unsqueeze(1)
        ).squeeze(1)
        weights = torch.softmax(correct_logits, dim=0).clamp(0.0, 1.0)
        # Real (label 0) is the compact/parallel group; fake is orthogonal.
        signs = torch.where(
            self.last_labels == 0,
            torch.ones_like(weights),
            -torch.ones_like(weights),
        )
        theta = math.pi / 4 + 0.5 * (
            torch.asin(weights) * signs
        ).sum()
        return torch.tan(theta).clamp(-100.0, 100.0)

    def after_backward(self, model):
        if self.task_index == 0:
            return
        with torch.no_grad():
            beta = self._rotation_beta()
            self._update_projector()
            gram = self.projector.t() @ self.projector
            identity = torch.eye(gram.shape[0], device=gram.device)
            mapped = self.projector @ torch.linalg.solve(
                gram + identity * 1e-10, self.projector.t()
            )
            self.complement.copy_(identity - mapped)
            self.complement.div_(self.complement.norm(p='fro').clamp_min(1e-12))
            direction = self.projector + beta * self.complement
            weight = model.w2vaasist.out_layer.weight
            weight.grad.copy_(weight.grad @ direction.t())

    def state_dict(self):
        state = super().state_dict()
        state['complement'] = (
            None if self.complement is None else self.complement.cpu()
        )
        return state

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.complement = state_dict.get('complement')


class RAWMMethod(OWMMethod):
    """ICML'23 RAWM adapted to AASIST's final linear layer."""

    name = 'rawm'

    def __init__(self, alpha=1.0, reg_alpha=1.0, temperature=2.0):
        super().__init__(alpha)
        self.reg_alpha = reg_alpha
        self.temperature = temperature
        self.complement = None
        self.last_labels = None
        self.teacher = None
        self.teacher_state = None

    def before_task(self, model, task_index):
        super().before_task(model, task_index)
        if self.complement is None:
            self.complement = torch.eye(
                self.projector.shape[0], device=self.projector.device
            )
        else:
            self.complement = self.complement.to(self.projector.device)
        if self.teacher_state is not None:
            self.teacher = copy.deepcopy(model.w2vaasist)
            self.teacher.load_state_dict(self.teacher_state)
            self.teacher.to(next(model.parameters()).device)
            self.teacher.eval()
            for parameter in self.teacher.parameters():
                parameter.requires_grad = False

    def loss(self, model, waveform, labels):
        features = model.wav2vec2.extract_features(waveform)
        hidden, logits = model.w2vaasist(features)
        self.last_hidden = hidden.detach()
        self.last_labels = labels.detach()
        base_loss = F.cross_entropy(logits, labels)
        if self.teacher is None:
            return base_loss, logits
        with torch.no_grad():
            _, teacher_logits = self.teacher(features.detach())
        temperature = self.temperature
        regularization = F.kl_div(
            F.log_softmax(logits / temperature, dim=1),
            F.softmax(teacher_logits / temperature, dim=1),
            reduction='batchmean',
        ) * (temperature ** 2)
        return base_loss + self.reg_alpha * regularization, logits

    def after_backward(self, model):
        if self.task_index == 0:
            return
        with torch.no_grad():
            real_count = (self.last_labels == 0).sum().float()
            fake_count = (self.last_labels == 1).sum().float()
            beta = (real_count + 1.0) / (fake_count + 1.0)
            self._update_projector()
            gram = self.projector.t() @ self.projector
            identity = torch.eye(gram.shape[0], device=gram.device)
            mapped = self.projector @ torch.linalg.solve(
                gram + identity * 1e-10, self.projector.t()
            )
            self.complement.copy_(identity - mapped)
            self.complement.div_(
                self.complement.norm(p='fro').clamp_min(1e-12)
            )
            direction = self.projector + beta * self.complement
            weight = model.w2vaasist.out_layer.weight
            weight.grad.copy_(weight.grad @ direction.t())

    def after_task(self, model, train_loader, device):
        del train_loader, device
        self.teacher = copy.deepcopy(model.w2vaasist)
        self.teacher.eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad = False
        self.teacher_state = {
            key: value.detach().cpu()
            for key, value in self.teacher.state_dict().items()
        }

    def state_dict(self):
        state = super().state_dict()
        state.update({
            'reg_alpha': self.reg_alpha,
            'temperature': self.temperature,
            'complement': (
                None if self.complement is None else self.complement.cpu()
            ),
            'teacher_state': self.teacher_state,
        })
        return state

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.reg_alpha = state_dict.get('reg_alpha', self.reg_alpha)
        self.temperature = state_dict.get('temperature', self.temperature)
        self.complement = state_dict.get('complement')
        self.teacher_state = state_dict.get('teacher_state')


class RegOMethod(SequentialMethod):
    """RegO regions on AASIST's final linear weight, matching official scope."""

    name = 'rego'

    def __init__(
        self, ef_threshold=0.1, importance_quantile=0.75,
        importance_max_batches=None
    ):
        self.ef_threshold = ef_threshold
        self.importance_quantile = importance_quantile
        self.importance_max_batches = importance_max_batches
        self.task_index = 0
        self.real_filters = []
        self.fake_filters = []
        self.old_gradient = None
        self.last_hidden = None
        self.last_labels = None

    def before_task(self, model, task_index):
        self.task_index = task_index
        device = next(model.parameters()).device
        self.real_filters = [value.to(device) for value in self.real_filters]
        self.fake_filters = [value.to(device) for value in self.fake_filters]
        if self.old_gradient is not None:
            self.old_gradient = self.old_gradient.to(device)

    def loss(self, model, waveform, labels):
        hidden, logits = model(waveform)
        self.last_hidden = hidden.detach()
        self.last_labels = labels.detach()
        return F.cross_entropy(logits, labels), logits

    @staticmethod
    def _ebbinghaus_weights(task_count):
        if task_count <= 1:
            return [1.0] * task_count
        low, high = 1e-6, 1e6
        for _ in range(100):
            midpoint = (low + high) / 2
            total = sum(math.exp(-index / midpoint)
                        for index in range(1, task_count + 1))
            if total > 1.0:
                high = midpoint
            else:
                low = midpoint
        scale = (low + high) / 2
        return list(reversed([
            math.exp(-index / scale)
            for index in range(1, task_count + 1)
        ]))

    def after_backward(self, model):
        weight = model.w2vaasist.out_layer.weight
        with torch.no_grad():
            current = weight.grad.clone()
            if (
                self.task_index > 0 and self.old_gradient is not None and
                self.real_filters and self.fake_filters
            ):
                memory = torch.zeros_like(current)
                weights = self._ebbinghaus_weights(self.task_index)
                for index in range(self.task_index):
                    active = self.real_filters[index].bool() | \
                        self.fake_filters[index].bool()
                    memory.add_(active.to(current.dtype), alpha=weights[index])
                remembered = memory > self.ef_threshold

                real_union = torch.zeros_like(current, dtype=torch.bool)
                fake_union = torch.zeros_like(current, dtype=torch.bool)
                for index in range(self.task_index):
                    real_union.logical_or_(self.real_filters[index].bool())
                    fake_union.logical_or_(self.fake_filters[index].bool())
                real_union.logical_and_(remembered)
                fake_union.logical_and_(remembered)

                region_a = ~real_union & ~fake_union
                region_b = real_union & ~fake_union
                region_c = ~real_union & fake_union
                region_d = real_union & fake_union

                old_flat = self.old_gradient.reshape(-1)
                projection_scale = torch.dot(
                    current.reshape(-1), old_flat
                ) / old_flat.pow(2).sum().clamp_min(1e-12)
                parallel = projection_scale * self.old_gradient
                orthogonal = current - parallel
                real_ratio = (self.last_labels == 0).float().mean()
                fake_ratio = 1.0 - real_ratio

                modified = torch.zeros_like(current)
                modified.add_(current * region_a)
                modified.add_(parallel * region_b)
                modified.add_(orthogonal * region_c)
                modified.add_((real_ratio * parallel + fake_ratio * orthogonal) * region_d)
                weight.grad.copy_(modified)
            self.old_gradient = weight.grad.detach().clone()

    def after_task(self, model, train_loader, device):
        weight = model.w2vaasist.out_layer.weight
        real_importance = torch.zeros_like(weight, device=device)
        fake_importance = torch.zeros_like(weight, device=device)
        real_count = 0
        fake_count = 0
        model.eval()
        for batch_index, (waveform, _, labels) in enumerate(train_loader):
            if (
                self.importance_max_batches is not None and
                batch_index >= self.importance_max_batches
            ):
                break
            waveform = waveform.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.no_grad():
                hidden, _ = model(waveform)
            for label_value, accumulator in (
                (0, real_importance), (1, fake_importance)
            ):
                mask = labels == label_value
                count = int(mask.sum())
                if count == 0:
                    continue
                model.zero_grad(set_to_none=True)
                logits = model.w2vaasist.out_layer(hidden[mask].detach())
                F.cross_entropy(logits, labels[mask]).backward()
                accumulator.add_(weight.grad.detach().pow(2), alpha=count)
                if label_value == 0:
                    real_count += count
                else:
                    fake_count += count
        real_importance.div_(max(real_count, 1))
        fake_importance.div_(max(fake_count, 1))
        real_threshold = torch.quantile(
            real_importance, self.importance_quantile
        )
        fake_threshold = torch.quantile(
            fake_importance, self.importance_quantile
        )
        self.real_filters.append(real_importance > real_threshold)
        self.fake_filters.append(fake_importance > fake_threshold)
        model.zero_grad(set_to_none=True)

    def state_dict(self):
        return {
            'ef_threshold': self.ef_threshold,
            'importance_quantile': self.importance_quantile,
            'importance_max_batches': self.importance_max_batches,
            'real_filters': [value.cpu() for value in self.real_filters],
            'fake_filters': [value.cpu() for value in self.fake_filters],
            'old_gradient': (
                None if self.old_gradient is None else self.old_gradient.cpu()
            ),
        }

    def load_state_dict(self, state_dict):
        self.ef_threshold = state_dict.get('ef_threshold', self.ef_threshold)
        self.importance_quantile = state_dict.get(
            'importance_quantile', self.importance_quantile
        )
        self.importance_max_batches = state_dict.get(
            'importance_max_batches', self.importance_max_batches
        )
        self.real_filters = state_dict.get('real_filters', [])
        self.fake_filters = state_dict.get('fake_filters', [])
        self.old_gradient = state_dict.get('old_gradient')


class ExternalPromptMethod(SequentialMethod):
    """Adapter for the matched audio prompt baselines reported in Table 1."""

    def before_task(self, model, task_index):
        model.before_task(task_index)

    def loss(self, model, waveform, labels):
        _, logits = model(waveform)
        base = F.cross_entropy(logits, labels)
        auxiliary = (
            model.auxiliary_loss()
            if model.kind in ('smope', 'rainbow')
            else model.auxiliary_loss(labels)
        )
        if not torch.isfinite(base + auxiliary):
            raise RuntimeError('Non-finite adapted-prompt baseline loss')
        return base + auxiliary, logits

    def after_optimizer_step(self, model):
        model.restore_frozen()


def build_method(args):
    if args.method in (
        'smope', 'rainbow', 'singleprompt', 'kaprompt', 'oisoprompt'
    ):
        return ExternalPromptMethod()
    if args.method == 'sequential':
        return SequentialMethod()
    if args.method == 'ewc':
        return EWCMethod(
            args.ewc_lambda, args.ewc_decay, args.ewc_fisher_max_batches
        )
    if args.method == 'lwf':
        return LwFMethod(args.lwf_alpha, args.lwf_temperature)
    if args.method == 'owm':
        return OWMMethod(args.owm_alpha)
    if args.method == 'rwm':
        return RWMMethod(args.owm_alpha)
    if args.method == 'rawm':
        return RAWMMethod(
            args.owm_alpha, args.lwf_alpha, args.lwf_temperature
        )
    if args.method == 'rego':
        return RegOMethod(
            args.rego_ef_threshold,
            args.rego_importance_quantile,
            args.rego_importance_max_batches,
        )
    if args.method == 'oprompt':
        return RealProtectedPromptMethod(
            args.prompt_orthogonal_weight,
            args.prompt_balance_weight,
            args.prompt_router_init_samples,
            getattr(args, 'prompt_key_match_weight', 0.0),
            getattr(args, 'prompt_llb_weight', 0.0),
            getattr(args, 'prompt_llb_delta', 0.2),
            protection=getattr(args, 'prompt_real_protection', 'none'),
            anchor_weight=getattr(
                args, 'prompt_real_anchor_weight', 0.1
            ),
            kd_weight=getattr(args, 'prompt_real_kd_weight', 1.0),
            feature_weight=getattr(
                args, 'prompt_real_feature_weight', 1.0
            ),
            temperature=getattr(
                args, 'prompt_real_kd_temperature', 2.0
            ),
            sample_fraction=getattr(
                args, 'prompt_real_sample_fraction', 0.25
            ),
            ema_decay=getattr(args, 'prompt_real_ema_decay', 0.999),
            stat_max_batches=getattr(
                args, 'prompt_real_stat_max_batches', 16
            ),
            projection_rank=getattr(
                args, 'prompt_real_projection_rank', 8
            ),
            projection_max_rank=getattr(
                args, 'prompt_real_projection_max_rank', 32
            ),
            real_freeze_layers=getattr(
                args, 'prompt_real_freeze_layers', 0
            ),
            protect_layer_count=getattr(
                args, 'prompt_real_protect_layer_count', 0
            ),
            spd_kind=getattr(args, 'prompt_spd_kind', 'feature'),
            spd_weight=getattr(args, 'prompt_spd_weight', 0.0),
            spd_real_only=getattr(args, 'prompt_spd_real_only', False),
            spd_layer_count=getattr(args, 'prompt_spd_layer_count', 0),
            real_update_layers=getattr(
                args, 'prompt_real_update_layers', 'all'
            ),
            real_update_top_k=getattr(
                args, 'prompt_real_update_top_k', 8
            ),
            smope_router_weight=0.0,
            smope_prototype_weight=0.0,
        )
    raise ValueError(f'Unsupported method: {args.method}')


class ProtocolATrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda')
        self.protocol_root = Path(args.protocol_root).resolve()
        self.output_dir = Path(args.output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.freeze = args.backbone_mode == 'frozen'
        self.initial_lr = args.lr
        self.min_lr = args.min_lr

        if args.method in ('singleprompt', 'kaprompt', 'oisoprompt'):
            from continual.audio_baselines3 import AudioPromptComparison
            self.model = AudioPromptComparison(
                args.xlsr, args.method, args.seed
            ).to(self.device)
        elif args.method in ('smope', 'rainbow'):
            from continual.audio_baselines import AudioPromptBaseline
            self.model = AudioPromptBaseline(
                args.xlsr, args.method, args.seed
            ).to(self.device)
        elif args.method == 'oprompt':
            if not self.freeze:
                raise ValueError('oprompt currently requires --backbone_mode frozen')
            self.model = RFPromptAASIST(
                model_dir=args.xlsr,
                ssl_backbone=getattr(args, 'ssl_backbone', 'xlsr'),
                classifier_seed=args.seed,
                real_tokens=args.prompt_real_tokens,
                fake_tokens=args.prompt_fake_tokens,
                dropout=args.prompt_dropout,
                router_temperature=args.prompt_router_temperature,
                router_floor=args.prompt_router_floor,
                use_shared_real=not args.prompt_no_shared_real,
                uniform_fake_router=args.prompt_uniform_fake_router,
                joint_mode=getattr(args, 'prompt_joint_mode', 'none'),
                fake_init_mode=getattr(
                    args, 'prompt_fake_init_mode', 'orthogonal'
                ),
                fake_init_scale=getattr(
                    args, 'prompt_fake_init_scale', 0.1
                ),
                collect_layer_features=(
                    getattr(args, 'prompt_spd_weight', 0.0) > 0 and
                    getattr(args, 'prompt_spd_kind', 'feature') != 'prompt_cosine'
                ),
                collect_prompt_response=(
                    getattr(args, 'prompt_llb_weight', 0.0) > 0
                ),
            ).to(self.device)
        else:
            ssl_backbone = getattr(args, 'ssl_backbone', 'xlsr')
            model_class = {
                'xlsr': XLSRAASIST,
                'wavlm': WAVLMAASIST,
                'w2vbert': W2V2BERTAASIST,
            }[ssl_backbone]
            model_kwargs = {
                'model_dir': args.xlsr, 'freeze': self.freeze,
            }
            if ssl_backbone == 'w2vbert':
                model_kwargs['classifier_seed'] = args.seed
            self.model = model_class(**model_kwargs).to(self.device)
        self.method = build_method(args)
        self.results = {
            'config': vars(args),
            'tasks': [task_id for task_id, _, _ in TASKS],
            'eval_rows': [],
            'dev_history': [],
            'parameter_history': [],
        }
        self.start_task = 0
        self.start_epoch = 0
        self.resume_optimizer_state = None
        self.resume_scheduler_state = None

        if args.resume:
            self._load_checkpoint(Path(args.resume))

    def _manifest(self, task_index, split):
        return self.protocol_root / TASKS[task_index][2] / f'{split}.csv'

    def _loader(self, manifest, shuffle, seed_offset=0, num_workers=None):
        dataset = ProtocolManifestDataset(str(manifest), self.args.audio_len)
        generator = torch.Generator()
        generator.manual_seed(self.args.seed + seed_offset)
        workers = self.args.num_workers if num_workers is None else num_workers
        return DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=shuffle,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=workers > 0 and shuffle,
            worker_init_fn=seed_worker,
            generator=generator,
        )

    def _model_state(self):
        if self.args.method in (
            'smope', 'rainbow', 'singleprompt', 'kaprompt', 'oisoprompt'
        ):
            return {
                'kind': 'external_prompt',
                'meta': self.model.meta(),
                'parameters': {
                    key: value.detach().cpu()
                    for key, value in self.model.state_dict().items()
                    if not key.startswith('ssl.')
                },
            }
        if self.args.method == 'oprompt':
            return {
                'kind': 'developmental_prompt',
                'task_count': self.model.prompt_encoder.task_count,
                'prompt_encoder': {
                    key: value.detach().cpu()
                    for key, value in self.model.prompt_encoder.state_dict().items()
                    if not key.startswith('model.')
                },
                'w2vaasist': {
                    key: value.detach().cpu()
                    for key, value in self.model.w2vaasist.state_dict().items()
                },
            }
        if self.freeze:
            return {
                'kind': 'frozen_backbone',
                'w2vaasist': {
                    key: value.detach().cpu()
                    for key, value in self.model.w2vaasist.state_dict().items()
                },
            }
        return {
            'kind': 'full_model',
            'model': {
                key: value.detach().cpu()
                for key, value in self.model.state_dict().items()
            },
        }

    def _restore_model_state(self, state):
        if state['kind'] == 'external_prompt':
            missing, unexpected = self.model.load_state_dict(
                state['parameters'], strict=False
            )
            if unexpected or not all(key.startswith('ssl.') for key in missing):
                raise RuntimeError(
                    f'Invalid adapted-prompt checkpoint: missing={missing}, '
                    f'unexpected={unexpected}'
                )
            self.model.restore_meta(state['meta'])
        elif state['kind'] == 'developmental_prompt':
            self.model.prompt_encoder.ensure_task_count(state['task_count'])
            self.model.prompt_encoder.load_state_dict(
                state['prompt_encoder'], strict=False
            )
            self.model.w2vaasist.load_state_dict(state['w2vaasist'])
        elif state['kind'] == 'frozen_backbone':
            self.model.w2vaasist.load_state_dict(state['w2vaasist'])
        elif state['kind'] == 'full_model':
            self.model.load_state_dict(state['model'])
        else:
            raise ValueError(f"Unknown checkpoint model kind: {state['kind']}")

    def _atomic_save(self, payload, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + '.tmp')
        torch.save(payload, temporary)
        os.replace(temporary, path)

    def _checkpoint_payload(
        self, task_index, epoch_completed, optimizer=None, scheduler=None
    ):
        return {
            'task_index': task_index,
            'epoch_completed': epoch_completed,
            'model_state': self._model_state(),
            'method_state': self.method.state_dict(),
            'optimizer_state': optimizer.state_dict() if optimizer else None,
            'scheduler_state': scheduler.state_dict() if scheduler else None,
            'results': self.results,
            'args': vars(self.args),
        }

    def _save_checkpoint(
        self, task_index, epoch_completed, optimizer=None, scheduler=None,
        filename='latest.pt'
    ):
        self._atomic_save(
            self._checkpoint_payload(
                task_index, epoch_completed, optimizer, scheduler
            ),
            self.output_dir / filename,
        )

    def _load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        self._restore_model_state(checkpoint['model_state'])
        self.method.load_state_dict(checkpoint.get('method_state', {}))
        self.results = checkpoint.get('results', self.results)
        self.start_task = int(checkpoint['task_index'])
        self.start_epoch = int(checkpoint['epoch_completed'])
        self.resume_optimizer_state = checkpoint.get('optimizer_state')
        self.resume_scheduler_state = checkpoint.get('scheduler_state')
        print(
            f"Resumed from {path}: task={self.start_task}, "
            f"epoch_completed={self.start_epoch}",
            flush=True,
        )

    def _write_results(self):
        result_path = self.output_dir / 'results.json'
        temporary = result_path.with_suffix('.json.tmp')
        with open(temporary, 'w', encoding='utf-8') as file:
            json.dump(self.results, file, indent=2, sort_keys=True)
        os.replace(temporary, result_path)

        with open(
            self.output_dir / 'eer_matrix.csv', 'w', newline='', encoding='utf-8'
        ) as file:
            writer = csv.writer(file)
            writer.writerow(['after_task'] + [task[0] for task in TASKS])
            for row in self.results['eval_rows']:
                writer.writerow(
                    [row['after_task']] +
                    [row['metrics'][task[0]]['eer_percent'] for task in TASKS]
                )

        with open(
            self.output_dir / 'auc_matrix.csv', 'w', newline='', encoding='utf-8'
        ) as file:
            writer = csv.writer(file)
            writer.writerow(['after_task'] + [task[0] for task in TASKS])
            for row in self.results['eval_rows']:
                writer.writerow(
                    [row['after_task']] +
                    [row['metrics'][task[0]]['auc'] for task in TASKS]
                )

        with open(
            self.output_dir / 'pooled_eval.csv', 'w', newline='', encoding='utf-8'
        ) as file:
            writer = csv.writer(file)
            writer.writerow(['after_task', 'samples', 'eer_percent', 'auc', 'threshold'])
            for row in self.results['eval_rows']:
                pooled = row.get('pooled')
                if pooled:
                    writer.writerow([
                        row['after_task'], pooled['samples'], pooled['eer_percent'],
                        pooled['auc'], pooled['threshold'],
                    ])

        rows = self.results['eval_rows']
        summary = {'completed_tasks': len(rows), 'average_seen_eer': {}}
        for row_index, row in enumerate(rows):
            seen = [
                row['metrics'][TASKS[index][0]]['eer_percent']
                for index in range(row_index + 1)
            ]
            summary['average_seen_eer'][row['after_task']] = float(np.mean(seen))
        if len(rows) == len(TASKS):
            final = rows[-1]
            final_eers = [
                final['metrics'][task[0]]['eer_percent'] for task in TASKS
            ]
            forgetting = []
            for task_index in range(len(TASKS) - 1):
                history = [
                    rows[row_index]['metrics'][TASKS[task_index][0]]['eer_percent']
                    for row_index in range(task_index, len(rows))
                ]
                forgetting.append(final_eers[task_index] - min(history))
            summary['final_average_eer'] = float(np.mean(final_eers))
            summary['average_forgetting'] = (
                float(np.mean(forgetting)) if forgetting else 0.0
            )
            if 'pooled' in final:
                summary['final_pooled_eer'] = final['pooled']['eer_percent']
                summary['final_pooled_auc'] = final['pooled']['auc']
        summary_path = self.output_dir / 'summary.json'
        temporary = summary_path.with_suffix('.json.tmp')
        with open(temporary, 'w', encoding='utf-8') as file:
            json.dump(summary, file, indent=2, sort_keys=True)
        os.replace(temporary, summary_path)

    def evaluate_manifest(
        self, manifest, max_batches=None, eval_task=None, return_arrays=False
    ):
        loader = self._loader(
            manifest, shuffle=False, seed_offset=1000,
            num_workers=self.args.eval_num_workers,
        )
        scores = []
        labels = []
        all_utterance_ids = []
        losses = []
        group_predictions = []
        fake_selections = []
        fake_weights = []
        real_expert_selections = []
        real_expert_weights = []
        started = time.monotonic()
        if self.args.method in (
            'smope', 'rainbow', 'singleprompt', 'kaprompt', 'oisoprompt'
        ):
            self.model.refresh_rainbow()
        self.model.eval()
        with torch.no_grad():
            for batch_index, (waveform, utterance_ids, label) in enumerate(loader):
                if max_batches is not None and batch_index >= max_batches:
                    break
                waveform = waveform.to(self.device, non_blocking=True)
                label_device = label.to(self.device, non_blocking=True)
                _, logits = self.model(waveform)
                if self.args.method == 'oprompt':
                    real_expert = self.model.last_real_expert
                    if real_expert is not None:
                        real_expert_selections.append(
                            real_expert['selected'].cpu()
                        )
                        real_expert_weights.append(
                            real_expert['weights'].cpu()
                        )
                    router = self.model.prompt_encoder.last_router
                    if router['group_logits'] is not None:
                        group_predictions.append(
                            router['group_logits'].argmax(dim=1).cpu()
                        )
                    fake_selections.append(
                        router['fake_weights'].argmax(dim=1).cpu()
                    )
                    fake_weights.append(router['fake_weights'].cpu())
                losses.append(float(F.cross_entropy(logits, label_device)))
                real_score = F.softmax(logits, dim=1)[:, 0]
                scores.append(real_score.cpu())
                labels.append(label)
                all_utterance_ids.extend(list(utterance_ids))
        score_array = torch.cat(scores).numpy()
        label_array = torch.cat(labels).numpy()
        eer, threshold = em.compute_eer(
            score_array[label_array == 0], score_array[label_array == 1]
        )
        auc = roc_auc_score(label_array, 1.0 - score_array)
        result = {
            'samples': int(label_array.size),
            'loss': float(np.mean(losses)),
            'eer': float(eer),
            'eer_percent': float(eer * 100.0),
            'auc': float(auc),
            'threshold': float(threshold),
            'seconds': time.monotonic() - started,
        }
        real_scores = score_array[label_array == 0]
        fake_scores = score_array[label_array == 1]
        if real_scores.size:
            result['real_recall_at_0_5_percent'] = float(
                np.mean(real_scores >= 0.5) * 100.0
            )
            result['real_recall_at_eer_percent'] = float(
                np.mean(real_scores >= threshold) * 100.0
            )
            result['mean_real_score'] = float(np.mean(real_scores))
        if fake_scores.size:
            result['fake_recall_at_0_5_percent'] = float(
                np.mean(fake_scores < 0.5) * 100.0
            )
            result['fake_recall_at_eer_percent'] = float(
                np.mean(fake_scores < threshold) * 100.0
            )
            result['mean_fake_score'] = float(np.mean(fake_scores))
        if return_arrays:
            result['_score_array'] = score_array
            result['_label_array'] = label_array
            result['_utterance_ids'] = all_utterance_ids
        if fake_selections:
            selected = torch.cat(fake_selections)
            weights = torch.cat(fake_weights)
            counts = torch.bincount(
                selected, minlength=self.model.prompt_encoder.active_tasks
            )
            result['fake_router_histogram'] = counts.tolist()
            result['mean_fake_router_weights'] = weights.mean(dim=0).tolist()
            label_tensor = torch.cat(labels)
            if group_predictions:
                group = torch.cat(group_predictions)
                result['group_router_accuracy_percent'] = float(
                    (group == label_tensor).float().mean() * 100.0
                )
            if eval_task is not None and eval_task < self.model.prompt_encoder.active_tasks:
                fake_mask = label_tensor == 1
                if fake_mask.any():
                    result['fake_router_task_accuracy_percent'] = float(
                        (selected[fake_mask] == eval_task).float().mean() * 100.0
                    )
        if real_expert_selections:
            selected = torch.cat(real_expert_selections)
            weights = torch.cat(real_expert_weights)
            expert_count = self.model.prompt_encoder.active_tasks
            result['real_expert_selection_mode'] = (
                'none'
            )
            result['real_expert_histogram'] = torch.bincount(
                selected, minlength=expert_count
            ).tolist()
            result['mean_real_expert_weights'] = weights.mean(dim=0).tolist()
            if eval_task is not None and eval_task < expert_count:
                result['real_expert_task_accuracy_percent'] = float(
                    (selected == eval_task).float().mean() * 100.0
                )
        return result

    def _append_train_curve(self, row):
        path = self.output_dir / 'train_curve.csv'
        exists = path.exists()
        with open(path, 'a', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=row.keys())
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def _train_task(self, task_index, start_epoch):
        task_id, task_name, _ = TASKS[task_index]
        train_loader = self._loader(
            self._manifest(task_index, 'train'),
            shuffle=True,
            seed_offset=task_index,
        )
        self.method.before_task(self.model, task_index)
        real_lr_scale = 1.0
        if self.args.method == 'oprompt' and task_index > 0:
            real_lr_scale = getattr(self.args, 'prompt_real_lr_scale', 1.0)
            if real_lr_scale == 0 and self.model.prompt_encoder.real_prompt is not None:
                self.model.prompt_encoder.real_prompt.requires_grad_(False)
        if self.args.method == 'kaprompt' and start_epoch == 0:
            init_loader = self._loader(
                self._manifest(task_index, 'train'),
                shuffle=False,
                seed_offset=2000 + task_index,
            )
            self.model.initialize_task(init_loader, self.device)
            del init_loader
        if self.args.method == 'oprompt' and start_epoch == 0:
            init_loader = self._loader(
                self._manifest(task_index, 'train'),
                shuffle=False,
                seed_offset=2000 + task_index,
            )
            self.method.initialize_task(self.model, init_loader, self.device)
            del init_loader
        torch.cuda.reset_peak_memory_stats(self.device)
        named_parameters = [
            (name, parameter) for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        ]
        parameters = [parameter for _, parameter in named_parameters]
        parameter_row = {
            'task_index': task_index,
            'task_id': task_id,
            'total_parameters': sum(
                parameter.numel() for parameter in self.model.parameters()
            ),
            'trainable_parameters': sum(
                parameter.numel() for parameter in parameters
            ),
        }
        if self.args.method == 'oprompt':
            parameter_row['stored_fake_experts'] = (
                self.model.prompt_encoder.task_count
            )
            parameter_row['injected_prompt_tokens'] = (
                self.model.prompt_encoder.real_tokens +
                self.model.prompt_encoder.fake_tokens
            )
        self.results['parameter_history'] = [
            row for row in self.results.get('parameter_history', [])
            if row['task_index'] != task_index
        ]
        self.results['parameter_history'].append(parameter_row)
        print(
            f"PARAMETERS task={task_id} total={parameter_row['total_parameters']} "
            f"trainable={parameter_row['trainable_parameters']}",
            flush=True,
        )
        real_parameters = [
            parameter for name, parameter in named_parameters
            if name == 'prompt_encoder.real_prompt'
        ]
        other_parameters = [
            parameter for name, parameter in named_parameters
            if name != 'prompt_encoder.real_prompt'
        ]
        parameter_groups = [{'params': other_parameters, 'lr': self.initial_lr}]
        if real_parameters:
            parameter_groups.append({
                'params': real_parameters,
                'lr': self.initial_lr * real_lr_scale,
            })
        optimizer = torch.optim.Adam(
            parameter_groups,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=5e-4,
        )
        minimum_factor = self.min_lr / self.initial_lr
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: minimum_factor + (
                1.0 - minimum_factor
            ) * (1.0 + math.cos(math.pi * epoch / self.args.epochs)) / 2.0,
        )
        if start_epoch:
            if self.resume_optimizer_state is None or self.resume_scheduler_state is None:
                raise ValueError('Mid-task resume requires optimizer and scheduler state')
            optimizer.load_state_dict(self.resume_optimizer_state)
            scheduler.load_state_dict(self.resume_scheduler_state)

        print(
            f"START_TASK task={task_id} name={task_name} "
            f"start_epoch={start_epoch} train_samples={len(train_loader.dataset)}",
            flush=True,
        )
        task_started = time.monotonic()
        best_dev_path = self.output_dir / f'best_dev_{task_id}.pt'
        best_dev_eer = float('inf')
        best_dev_loss = float('inf')
        best_dev_auc = float('-inf')
        best_dev_epoch = None
        if start_epoch and best_dev_path.exists():
            previous_best = torch.load(best_dev_path, map_location='cpu')
            selection = previous_best.get('dev_selection', {})
            if int(selection.get('task_index', -1)) == task_index:
                best_dev_eer = float(selection['eer_percent'])
                best_dev_loss = float(selection['loss'])
                best_dev_auc = float(selection['auc'])
                best_dev_epoch = int(selection['epoch'])
            del previous_best
        for epoch in range(start_epoch, self.args.epochs):
            if self.args.method in (
                'smope', 'rainbow', 'singleprompt', 'kaprompt', 'oisoprompt'
            ):
                self.model.epoch = epoch
            self.model.train()
            losses = []
            epoch_started = time.monotonic()
            for batch_index, (waveform, _, labels) in enumerate(train_loader):
                if (
                    self.args.max_train_batches is not None and
                    batch_index >= self.args.max_train_batches
                ):
                    break
                waveform = waveform.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss, _ = self.method.loss(self.model, waveform, labels)
                loss.backward()
                self.method.after_backward(self.model)
                optimizer.step()
                self.method.after_optimizer_step(self.model)
                losses.append(float(loss.detach()))
            scheduler.step()

            row = {
                'task_index': task_index,
                'task_id': task_id,
                'epoch': epoch + 1,
                'train_loss': float(np.mean(losses)),
                'lr': optimizer.param_groups[0]['lr'],
                'real_lr': (
                    optimizer.param_groups[1]['lr']
                    if len(optimizer.param_groups) > 1 else 0.0
                ),
                'epoch_seconds': time.monotonic() - epoch_started,
                'task_elapsed_seconds': time.monotonic() - task_started,
            }
            if self.args.method == 'oprompt':
                row.update(self.method.last_terms)
            self._append_train_curve(row)
            print(
                f"TRAIN task={task_id} epoch={epoch + 1}/{self.args.epochs} "
                f"loss={row['train_loss']:.6f} lr={row['lr']:.3e} "
                f"real_lr={row['real_lr']:.3e} "
                f"seconds={row['epoch_seconds']:.1f}",
                flush=True,
            )

            should_monitor = (
                self.args.dev_interval > 0 and
                ((epoch + 1) % self.args.dev_interval == 0 or
                 epoch + 1 == self.args.epochs)
            )
            if should_monitor:
                dev_metrics = self.evaluate_manifest(
                    self._manifest(task_index, 'dev'),
                    max_batches=self.args.eval_max_batches,
                    eval_task=task_index,
                )
                dev_row = {
                    'task_index': task_index,
                    'task_id': task_id,
                    'epoch': epoch + 1,
                    **dev_metrics,
                }
                self.results['dev_history'].append(dev_row)
                current_eer = float(dev_metrics['eer_percent'])
                current_loss = float(dev_metrics['loss'])
                current_auc = float(dev_metrics['auc'])
                improved = (
                    current_eer < best_dev_eer or
                    (
                        math.isclose(current_eer, best_dev_eer) and
                        current_loss < best_dev_loss
                    )
                )
                if improved:
                    best_dev_eer = current_eer
                    best_dev_loss = current_loss
                    best_dev_auc = current_auc
                    best_dev_epoch = epoch + 1
                    best_payload = self._checkpoint_payload(
                        task_index, epoch + 1, optimizer, scheduler
                    )
                    best_payload['dev_selection'] = {
                        'task_index': task_index,
                        'task_id': task_id,
                        'epoch': best_dev_epoch,
                        'eer_percent': best_dev_eer,
                        'loss': best_dev_loss,
                        'auc': best_dev_auc,
                        'rule': 'min_eer_then_min_loss_then_earliest_epoch',
                    }
                    self._atomic_save(best_payload, best_dev_path)
                self._write_results()
                print(
                    f"DEV task={task_id} epoch={epoch + 1} "
                    f"eer={dev_metrics['eer_percent']:.4f}% "
                    f"loss={dev_metrics['loss']:.6f} "
                    f"auc={dev_metrics['auc']:.6f} "
                    f"best_epoch={best_dev_epoch}",
                    flush=True,
                )

            if (
                self.args.checkpoint_interval > 0 and
                (epoch + 1) % self.args.checkpoint_interval == 0
            ):
                self._save_checkpoint(
                    task_index, epoch + 1, optimizer, scheduler
                )

        if self.args.dev_interval > 0:
            if best_dev_epoch is None or not best_dev_path.exists():
                raise RuntimeError(
                    f'No best Dev checkpoint was recorded for task {task_id}'
                )
            best_checkpoint = torch.load(best_dev_path, map_location='cpu')
            self._restore_model_state(best_checkpoint['model_state'])
            self.method.load_state_dict(best_checkpoint.get('method_state', {}))
            del best_checkpoint
            print(
                f"SELECT_BEST_DEV task={task_id} epoch={best_dev_epoch} "
                f"eer={best_dev_eer:.4f}% loss={best_dev_loss:.6f} "
                f"auc={best_dev_auc:.6f}",
                flush=True,
            )

        self.method.after_task(self.model, train_loader, self.device)
        parameter_row['peak_gpu_memory_mb'] = float(
            torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
        )

    def _evaluate_all_tasks(self, after_task_index):
        metrics = {}
        pooled_scores = []
        pooled_labels = []
        pooled_utterance_ids = []
        pooled_native_tasks = []
        for eval_index, (task_id, task_name, _) in enumerate(TASKS):
            result = self.evaluate_manifest(
                self._manifest(eval_index, 'eval'),
                max_batches=self.args.eval_max_batches,
                eval_task=eval_index,
                return_arrays=True,
            )
            pooled_scores.append(result.pop('_score_array'))
            pooled_labels.append(result.pop('_label_array'))
            utterance_ids = result.pop('_utterance_ids')
            pooled_utterance_ids.extend(utterance_ids)
            pooled_native_tasks.extend([task_id] * len(utterance_ids))
            metrics[task_id] = {'name': task_name, **result}
            print(
                f"EVAL after={TASKS[after_task_index][0]} task={task_id} "
                f"samples={result['samples']} eer={result['eer_percent']:.4f}% "
                f"auc={result['auc']:.6f}",
                flush=True,
            )
        score_array = np.concatenate(pooled_scores)
        label_array = np.concatenate(pooled_labels)
        pooled_eer, pooled_threshold = em.compute_eer(
            score_array[label_array == 0], score_array[label_array == 1]
        )
        pooled = {
            'samples': int(label_array.size),
            'eer': float(pooled_eer),
            'eer_percent': float(pooled_eer * 100.0),
            'auc': float(roc_auc_score(label_array, 1.0 - score_array)),
            'threshold': float(pooled_threshold),
        }
        print(
            f"POOLED_EVAL after={TASKS[after_task_index][0]} "
            f"samples={pooled['samples']} eer={pooled['eer_percent']:.4f}% "
            f"auc={pooled['auc']:.6f}",
            flush=True,
        )
        self.results['eval_rows'].append({
            'after_task': TASKS[after_task_index][0],
            'metrics': metrics,
            'pooled': pooled,
        })
        score_path = self.output_dir / (
            f'eval_scores_after_{TASKS[after_task_index][0]}.csv'
        )
        with open(score_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['utt_id', 'label', 'score_real', 'native_task'])
            writer.writerows(zip(
                pooled_utterance_ids, label_array.tolist(),
                score_array.tolist(), pooled_native_tasks,
            ))
        self._write_results()

    def run(self):
        stop_task = min(len(TASKS), self.args.max_tasks)
        if self.start_task >= stop_task:
            print('Requested tasks are already complete.', flush=True)
            return
        for task_index in range(self.start_task, stop_task):
            start_epoch = self.start_epoch if task_index == self.start_task else 0
            self._train_task(task_index, start_epoch)
            self.resume_optimizer_state = None
            self.resume_scheduler_state = None
            self.start_epoch = 0
            self._evaluate_all_tasks(task_index)
            self._save_checkpoint(
                task_index + 1,
                0,
                filename=f'after_{TASKS[task_index][0]}.pt',
            )
            self._save_checkpoint(task_index + 1, 0)
        print('TRAINING_COMPLETE', flush=True)
