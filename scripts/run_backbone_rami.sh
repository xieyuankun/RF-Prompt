#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "Usage: $0 METHOD FAMILY SSL_DIR RAMI_PROTOCOL_ROOT OUTPUT_DIR" >&2
  echo "METHOD: rfprompt | sequential; FAMILY: xlsr | wavlm | w2vbert" >&2
  exit 2
fi

METHOD="$1"
FAMILY="$2"
SSL_DIR="$3"
PROTOCOL_ROOT="$4"
OUTPUT_DIR="$5"

case "$METHOD" in rfprompt|sequential) ;; *) exit 2 ;; esac
case "$FAMILY" in xlsr|wavlm|w2vbert) ;; *) exit 2 ;; esac

python -u scripts/train_rfprompt.py \
  --method "$METHOD" \
  --ssl_backbone "$FAMILY" \
  --xlsr "$SSL_DIR" \
  --protocol_root "$PROTOCOL_ROOT" \
  --task_layout b \
  --epochs 50 \
  --batch_size 32 \
  --num_workers 8 \
  --eval_num_workers 4 \
  --seed 2026 \
  --lr 1e-4 \
  --min_lr 1e-6 \
  --dev_interval 1 \
  --prompt_fake_init_mode inherit_response \
  --prompt_joint_mode response \
  --prompt_spd_kind prompt_cosine \
  --prompt_spd_weight 1.0 \
  --prompt_spd_layer_count 24 \
  --prompt_orthogonal_weight 0.1 \
  --output_dir "$OUTPUT_DIR"
