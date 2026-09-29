#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 XLSR_DIR RAMI_PROTOCOL_ROOT OUTPUT_DIR" >&2
  exit 2
fi

XLSR_DIR="$1"
PROTOCOL_ROOT="$2"
OUTPUT_DIR="$3"

python -u scripts/train_rfprompt.py \
  --xlsr "$XLSR_DIR" \
  --protocol_root "$PROTOCOL_ROOT" \
  --task_layout b \
  --method rfprompt \
  --backbone_mode frozen \
  --epochs 50 \
  --batch_size 32 \
  --num_workers 8 \
  --eval_num_workers 4 \
  --seed 2026 \
  --lr 1e-4 \
  --min_lr 1e-6 \
  --dev_interval 1 \
  --checkpoint_interval 10 \
  --max_tasks 4 \
  --prompt_real_tokens 5 \
  --prompt_fake_tokens 5 \
  --prompt_fake_init_mode inherit_response \
  --prompt_router_temperature 0.1 \
  --prompt_balance_weight 0 \
  --prompt_llb_weight 0 \
  --prompt_real_protection none \
  --prompt_real_update_layers all \
  --prompt_joint_mode response \
  --prompt_spd_kind prompt_cosine \
  --prompt_spd_weight 1.0 \
  --prompt_spd_layer_count 24 \
  --prompt_orthogonal_weight 0.1 \
  --output_dir "$OUTPUT_DIR"
