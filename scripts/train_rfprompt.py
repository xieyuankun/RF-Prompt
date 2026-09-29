#!/usr/bin/env python3
"""Train RF-Prompt or a matched continual-learning baseline."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from continual.trainer import ProtocolATrainer, configure_tasks, seed_everything


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol_root', required=True)
    parser.add_argument(
        '--task_layout', default='b', choices=('legacy', 'a', 'b', 'joint')
    )
    parser.add_argument('--xlsr', required=True)
    parser.add_argument(
        '--ssl_backbone', default='xlsr',
        choices=('xlsr', 'wavlm', 'w2vbert'),
        help='SSL architecture in --xlsr; XLS-R 300M/1B/2B all use xlsr',
    )
    parser.add_argument('--output_dir', required=True)
    parser.add_argument(
        '--method', default='rfprompt',
        choices=(
            'rfprompt', 'sequential', 'ewc', 'lwf', 'owm', 'rawm', 'rwm',
            'rego', 'oisoprompt', 'singleprompt', 'kaprompt', 'smope',
            'rainbow',
        )
    )
    parser.add_argument('--backbone_mode', default='frozen', choices=('frozen', 'ft'))
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument(
        '--batch_size', type=int,
        help='defaults to 64 for frozen backbones and 14 for full fine-tuning'
    )
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--eval_num_workers', type=int, default=0)
    parser.add_argument('--audio_len', type=int, default=64600)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--min_lr', type=float)
    parser.add_argument('--dev_interval', type=int, default=1)
    parser.add_argument('--checkpoint_interval', type=int, default=10)
    parser.add_argument('--resume')
    parser.add_argument('--max_tasks', type=int, default=4)
    parser.add_argument('--max_train_batches', type=int)
    parser.add_argument('--eval_max_batches', type=int)
    parser.add_argument('--ewc_lambda', type=float, default=100.0)
    parser.add_argument('--ewc_decay', type=float, default=1.0)
    parser.add_argument('--ewc_fisher_max_batches', type=int)
    parser.add_argument('--lwf_alpha', type=float, default=1.0)
    parser.add_argument('--lwf_temperature', type=float, default=2.0)
    parser.add_argument('--owm_alpha', type=float, default=1.0)
    parser.add_argument('--rego_ef_threshold', type=float, default=0.1)
    parser.add_argument('--rego_importance_quantile', type=float, default=0.75)
    parser.add_argument('--rego_importance_max_batches', type=int)
    parser.add_argument('--prompt_real_tokens', type=int, default=5)
    parser.add_argument('--prompt_fake_tokens', type=int, default=5)
    parser.add_argument('--prompt_dropout', type=float, default=0.1)
    parser.add_argument('--prompt_router_temperature', type=float, default=0.1)
    parser.add_argument('--prompt_router_floor', type=float, default=0.1)
    parser.add_argument('--prompt_router_init_samples', type=int, default=512)
    parser.add_argument('--prompt_balance_weight', type=float, default=0.0)
    parser.add_argument('--prompt_orthogonal_weight', type=float, default=0.1)
    parser.add_argument('--prompt_key_match_weight', type=float, default=0.0)
    parser.add_argument(
        '--prompt_joint_mode', default='response',
        choices=(
            'none', 'uniform', 'normalized', 'concat', 'response',
        ),
        help='remove learned routing and either average all fake prompt banks '
             'into five tokens or concatenate all prompt banks'
    )
    parser.add_argument('--prompt_llb_weight', type=float, default=0.0)
    parser.add_argument('--prompt_llb_delta', type=float, default=0.2)
    parser.add_argument(
        '--prompt_fake_init_mode', default='inherit_response',
        choices=('orthogonal', 'inherit_response'),
        help='initialize each new Fake5 independently or inherit token slots '
             'from the old expert most responsive to current frozen SSL queries'
    )
    parser.add_argument('--prompt_fake_init_scale', type=float, default=0.1)
    parser.add_argument(
        '--prompt_spd_weight', type=float, default=1.0,
        help='GAP-style cosine distillation weight for the inherited Shared Real Prompt'
    )
    parser.add_argument(
        '--prompt_spd_real_only', action='store_true',
        help='apply Shared Prompt Distillation only to real samples'
    )
    parser.add_argument(
        '--prompt_spd_layer_count', type=int, default=24,
        help='protect the first N prompt layers'
    )
    parser.add_argument(
        '--prompt_real_update_layers', default='all',
        choices=('all', 'early8', 'middle8', 'late8', 'gap_topk'),
        help='Real Prompt layers allowed to update after task 0'
    )
    parser.add_argument('--prompt_real_update_top_k', type=int, default=8)
    parser.add_argument(
        '--prompt_real_lr_scale', type=float, default=1.0,
        help='multiply the shared Real Prompt learning rate after task 0; '
             '0 freezes it while leaving the current Fake Prompt and AASIST trainable'
    )
    parser.add_argument(
        '--prompt_real_protection', default='none',
        choices=(
            'none', 'proximal', 'fisher', 'gradient_projection', 'ema',
            'logit_kd', 'full_kd', 'opd', 'pcgrad_global',
            'pcgrad_layerwise',
        ),
        help='training-only protection for the single plastic Shared Real Prompt'
    )
    parser.add_argument('--prompt_real_anchor_weight', type=float, default=0.1)
    parser.add_argument('--prompt_real_kd_weight', type=float, default=1.0)
    parser.add_argument('--prompt_real_feature_weight', type=float, default=1.0)
    parser.add_argument('--prompt_real_kd_temperature', type=float, default=2.0)
    parser.add_argument('--prompt_real_sample_fraction', type=float, default=0.25)
    parser.add_argument('--prompt_real_ema_decay', type=float, default=0.999)
    parser.add_argument('--prompt_real_stat_max_batches', type=int, default=16)
    parser.add_argument('--prompt_real_projection_rank', type=int, default=8)
    parser.add_argument('--prompt_real_projection_max_rank', type=int, default=32)
    parser.add_argument(
        '--prompt_real_freeze_layers', type=int, default=0,
        help='after B0, zero Shared Real5 gradients in this many early layers'
    )
    parser.add_argument(
        '--prompt_real_protect_layer_count', type=int, default=0,
        help='apply parameter protection to the first N Shared Real Prompt layers; '
             '0 protects all layers'
    )
    parser.add_argument(
        '--prompt_no_shared_real', action='store_true',
        help='ablation: remove the shared real-knowledge prompt'
    )
    parser.add_argument('--prompt_uniform_fake_router', action='store_true')
    parser.add_argument('--prompt_spd_kind', default='prompt_cosine',
                        choices=('feature', 'prompt_cosine', 'feature_clean'))
    args = parser.parse_args()
    # ``oprompt`` is retained as the internal checkpoint identifier for
    # compatibility with the experiments used in the paper.
    if args.method == 'rfprompt':
        args.method = 'oprompt'

    if args.lr is None:
        args.lr = 1e-4 if args.backbone_mode == 'frozen' else 1e-6
    if args.min_lr is None:
        args.min_lr = 1e-6 if args.backbone_mode == 'frozen' else 1e-8
    if args.batch_size is None:
        if args.method == 'oprompt':
            args.batch_size = 32
        else:
            args.batch_size = 32 if args.backbone_mode == 'frozen' else 14
    if args.num_workers < 0 or args.num_workers > 16:
        parser.error('--num_workers must be between 0 and 16')
    if args.eval_num_workers < 0 or args.eval_num_workers > 16:
        parser.error('--eval_num_workers must be between 0 and 16')
    if args.max_tasks < 1 or args.max_tasks > 4:
        parser.error('--max_tasks must be between 1 and 4')
    if args.prompt_fake_tokens < 1:
        parser.error('--prompt_fake_tokens must be positive')
    if not args.prompt_no_shared_real and args.prompt_real_tokens < 1:
        parser.error('--prompt_real_tokens must be positive with shared real')
    if not 0 <= args.prompt_router_floor < 1:
        parser.error('--prompt_router_floor must be in [0, 1)')
    if args.prompt_key_match_weight < 0:
        parser.error('--prompt_key_match_weight must be non-negative')
    if args.prompt_llb_weight < 0:
        parser.error('--prompt_llb_weight must be non-negative')
    if args.prompt_fake_init_scale < 0:
        parser.error('--prompt_fake_init_scale must be non-negative')
    if args.prompt_spd_weight < 0:
        parser.error('--prompt_spd_weight must be non-negative')
    if not 0 <= args.prompt_spd_layer_count <= 24:
        parser.error('--prompt_spd_layer_count must be between 0 and 24')
    if args.prompt_real_update_top_k < 1:
        parser.error('--prompt_real_update_top_k must be positive')
    if not 0 <= args.prompt_real_lr_scale <= 1:
        parser.error('--prompt_real_lr_scale must be in [0, 1]')
    if args.prompt_real_anchor_weight < 0 or args.prompt_real_kd_weight < 0:
        parser.error('Real Prompt protection weights must be non-negative')
    if args.prompt_real_feature_weight < 0:
        parser.error('--prompt_real_feature_weight must be non-negative')
    if args.prompt_real_kd_temperature <= 0:
        parser.error('--prompt_real_kd_temperature must be positive')
    if not 0 < args.prompt_real_sample_fraction <= 1:
        parser.error('--prompt_real_sample_fraction must be in (0, 1]')
    if not 0 <= args.prompt_real_ema_decay < 1:
        parser.error('--prompt_real_ema_decay must be in [0, 1)')
    if args.prompt_real_stat_max_batches < 1:
        parser.error('--prompt_real_stat_max_batches must be positive')
    if (
        args.prompt_real_projection_rank < 1 or
        args.prompt_real_projection_max_rank < args.prompt_real_projection_rank
    ):
        parser.error('Real projection ranks are invalid')
    if not 0 <= args.prompt_real_freeze_layers <= 24:
        parser.error('--prompt_real_freeze_layers must be between 0 and 24')
    if not 0 <= args.prompt_real_protect_layer_count <= 24:
        parser.error('--prompt_real_protect_layer_count must be between 0 and 24')
    if not 0 <= args.prompt_llb_delta < 1:
        parser.error('--prompt_llb_delta must be in [0, 1)')
    if args.prompt_llb_weight > 0 and args.prompt_joint_mode == 'none':
        parser.error('--prompt_llb_weight requires --prompt_joint_mode')
    return args


def main():
    args = parse_args()
    configure_tasks(args.task_layout)
    seed_everything(args.seed)
    ProtocolATrainer(args).run()


if __name__ == '__main__':
    main()
